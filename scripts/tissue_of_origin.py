#!/usr/bin/env python3
"""Tissue-of-Origin (TOO) classifier from cfDNA fragmentomics features.

This is the headline MCED sub-task: *given a sample that is already known
to be cancer*, predict **which tissue** the cfDNA is shedding from. The
closest clinical analogue is GRAIL's Galleri "Cancer Signal Origin" (CSO)
feature, which assigns one of 50+ tissue labels to a positive MCED screen.

Comparison with the existing multi-cancer classifier
(`scripts/multiclass_classification.py`):

  - Multi-cancer: cancer-vs-healthy per class, includes HEALTHY in the
    label space, reports OvR AUC with HEALTHY as one of the classes.
    - macro AUC ~0.97 across 5 classes (BRCA, HCC_J, LUAD, PAAD, HEALTHY).

  - Tissue-of-Origin (this script): cancer-ONLY cohort, HEALTHY dropped
    from the analysis. The label space is {BRCA, CRC, HCC_J, LUAD, OV,
    PAAD}. The question is harder: "given the sample is cancer, *where*
    is it from?" — not just "is it cancer?".

Inputs:
  - 5-channel fragmentomics features (5 Mb DELFI ratio + 5 Mb coverage
    + 100 kb ratio + 100 kb counts + FSD), built by `honest_benchmark.load5`.
  - labels_multiclass.tsv with a disease_class per sample.

Model:
  - OvR logistic regression with PCA-200 + StandardScaler, identical to
    the multi-cancer script. Re-using the same pipeline keeps the
    TOO/multi-cancer comparison apples-to-apples: any difference in
    performance is the disease_class task, not the model.

Evaluation:
  - 5 seeds × 5-fold StratifiedKFold, pooled OOF.
  - Per-class OOF AUC, macro AUC, top-1 + top-2 accuracy, confusion matrix.

Honest caveats (also documented in docs/TISSUE_OF_ORIGIN_RESULTS.md):
  - Fragmentomics-only — no methylation. Methylation is the dominant TOO
    signal in Galleri and most published MCED TOO work, so we expect
    per-class AUCs in the 0.6-0.75 range (vs binary ~0.97).
  - Pooled OOF on the same cohort — not an external validation cohort.
  - Six broad tissue labels — no sub-tissue (e.g. LUAD vs LUSC, BRCA
    subtype), so we cannot reproduce Galleri's 50+ tissue taxonomy.

Usage:
    python scripts/tissue_of_origin.py
    python scripts/tissue_of_origin.py --help
    python scripts/tissue_of_origin.py --quick \
        --out results/tissue_of_origin_smoke.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Reuse the repo's shared constants + 5-channel feature loader so we
# stay apples-to-apples with the multi-cancer script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FEAT_DIR, REPO_ROOT, RESULTS_DIR
from honest_benchmark import load5
from multiclass_classification import build_labels_multiclass

LABELS_MULTICLASS = REPO_ROOT / "labels_multiclass.tsv"
DEFAULT_OUT = RESULTS_DIR / "tissue_of_origin.json"

SEEDS = [42, 13, 7, 99, 1234]
N_FOLDS = 5
MIN_CLASS_N = 25  # smaller than multi-cancer (30) to keep CRC/OV
PCA_N = 200
LR_C = 1.0

# Cancer-only disease classes for TOO (HEALTHY is dropped)
TOO_CLASSES = ["BRCA", "CRC", "HCC_J", "LUAD", "OV", "PAAD"]


# --------------------------------------------------------------------------- #
# Cohort loader (cancer-only)
# --------------------------------------------------------------------------- #
def load_cohort_cancer_only(labels_multiclass: dict[str, str],
                            keep: set[str],
                            feat_dir: str):
    """Load 5-channel features for cancer samples whose class ∈ `keep`.

    Mirrors `multiclass_classification.load_cohort_5ch` but uses the
    cancer disease_class as the multiclass label and DROPS healthy
    samples entirely. Returns X, y_multi, classes, cls_to_int, sample_ids.
    """
    cancer_labels = {s: c for s, c in labels_multiclass.items()
                     if c in keep and c != "HEALTHY"}
    # load5 wants a binary {0,1} target — set 1 for everything (all cancer)
    labels01 = {s: 1 for s in cancer_labels}
    studies = {s: ("jiang" if s.startswith(("H", "HOT"))
                   else "cristiano")
               for s in labels01}

    X, _y_bin, _studies_arr = load5(labels01, studies, feat_dir=feat_dir)

    # Recover the sample-id order used by load5 (sorted keys)
    sample_ids = []
    for s in sorted(labels01.keys()):
        paths = [os.path.join(feat_dir, f"{s}.delfi_5mb_ratio.npy"),
                 os.path.join(feat_dir, f"{s}.delfi_5mb_coverage.npy"),
                 os.path.join(feat_dir, f"{s}.delfi_100kb_ratio.npy"),
                 os.path.join(feat_dir, f"{s}.delfi_100kb_counts.npy")]
        if all(os.path.exists(p) for p in paths) and \
           os.path.exists(os.path.join(feat_dir, f"{s}.fsd.json")):
            sample_ids.append(s)
    assert len(sample_ids) == X.shape[0], (len(sample_ids), X.shape[0])

    classes = sorted(keep - {"HEALTHY"})
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y_multi = np.array([cls_to_int[cancer_labels[s]] for s in sample_ids],
                       dtype=int)
    return X, y_multi, classes, cls_to_int, sample_ids


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def ovr_logreg_cv(X: np.ndarray, y: np.ndarray, classes: list[str],
                  seeds: list[int], n_folds: int, pca_n: int):
    """One-vs-rest logistic regression, pooled OOF across seeds.

    Returns (oof_proba, per_class_aucs_seed, macro_aucs, top1_accs,
             top2_accs).

    Speed note (Sept 2026): The PCA fit on the training fold is shared
    across all OvR classes — previously it was refit per class (5-class
    OvR = 5× wasted PCA work; 6-class TOO = 6× wasted). Pre-fitting PCA
    once per fold and projecting the train/test matrices before the
    per-class LR loop preserves the exact algorithm and identical
    numerical results (PCA is identical because it depends only on Xtr,
    not on y).
    """
    n, n_classes = X.shape[0], len(classes)
    oof_proba = np.zeros((n, n_classes), dtype=float)
    macro_aucs: list[float] = []
    top1_accs: list[float] = []
    top2_accs: list[float] = []
    per_class_aucs_seed: dict[str, list[float]] = {c: [] for c in classes}

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=seed)
        oof_seed = np.zeros((n, n_classes), dtype=float)
        for tr, te in cv.split(X, y):
            # Standardize once per fold
            sc = StandardScaler().fit(X[tr])
            Xtr_s = sc.transform(X[tr])
            Xte_s = sc.transform(X[te])
            del sc
            # PCA once per fold (shared across all OvR classes)
            if pca_n and pca_n < Xtr_s.shape[1]:
                pca = PCA(n_components=min(pca_n, Xtr_s.shape[0] - 1),
                          random_state=seed).fit(Xtr_s)
                Xtr_p = pca.transform(Xtr_s)
                Xte_p = pca.transform(Xte_s)
                del pca
            else:
                Xtr_p, Xte_p = Xtr_s, Xte_s
            for ci in range(n_classes):
                ytr_bin = (y[tr] == ci).astype(int)
                if ytr_bin.sum() == 0 or ytr_bin.sum() == len(ytr_bin):
                    continue
                clf = LogisticRegression(C=LR_C, max_iter=1000,
                                         solver="lbfgs", random_state=seed)
                clf.fit(Xtr_p, ytr_bin)
                oof_seed[te, ci] = clf.predict_proba(Xte_p)[:, 1]
                del clf
            # Release per-fold intermediates (LR coefs, PCA internal cov,
            # StandardScaler buffers) before the next fold allocates.
            del Xtr_s, Xte_s, Xtr_p, Xte_p
            gc.collect()

        seed_class_aucs = []
        for ci, cls in enumerate(classes):
            y_bin = (y == ci).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            try:
                a = roc_auc_score(y_bin, oof_seed[:, ci])
                per_class_aucs_seed[cls].append(a)
                seed_class_aucs.append(a)
            except ValueError:
                continue
        if seed_class_aucs:
            macro_aucs.append(float(np.mean(seed_class_aucs)))
        y_pred_seed = np.argmax(oof_seed, axis=1)
        top1_accs.append(accuracy_score(y, y_pred_seed))
        try:
            top2_accs.append(top_k_accuracy_score(
                y, oof_seed, k=2, labels=np.arange(n_classes)))
        except ValueError:
            pass
        oof_proba += oof_seed

    oof_proba /= len(seeds)
    return (oof_proba, per_class_aucs_seed, macro_aucs,
            top1_accs, top2_accs)


# --------------------------------------------------------------------------- #
# Within-Cristiano ablation (optional)
# --------------------------------------------------------------------------- #
def _cristiano_only_ablation(labels_multiclass: dict[str, str],
                             all_classes: list[str],
                             feat_dir: str,
                             seeds: list[int], n_folds: int, pca_n: int):
    """Drop Jiang HCC_J (n=89, all Jiang protocol) and re-run OvR CV on the
    5 cancer types from the Cristiano 2019 cohort only.

    Motivation: the OvR HCC_J AUC in the full cohort is confounded by
    protocol-level batch effects (every HCC_J sample is Jiang; every
    non-HCC_J sample is Cristiano), so a near-perfect HCC_J AUC is
    plausible even if HCC_J fragmentomics carries no tissue signal.
    The Cristiano-only number is the honest fragmentomics-only TOO
    estimate (still inflated by any within-Cristiano batch structure,
    but no cross-study leak).
    """
    cristiano_classes = [c for c in all_classes if c != "HCC_J"]
    keep = set(cristiano_classes)
    (X, y_multi, classes, cls_to_int, _ids) = load_cohort_cancer_only(
        labels_multiclass, keep=keep, feat_dir=feat_dir)
    if X.shape[0] == 0:
        return None
    (oof_proba, _per_class_aucs_seed, macro_aucs,
     top1_accs, top2_accs) = ovr_logreg_cv(
        X, y_multi, classes, seeds=seeds, n_folds=n_folds, pca_n=pca_n)
    y_pred = np.argmax(oof_proba, axis=1)
    cm = confusion_matrix(y_multi, y_pred, labels=np.arange(len(classes)))
    oof_per_class_auc = {}
    for i, c in enumerate(classes):
        y_bin = (y_multi == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        try:
            oof_per_class_auc[c] = float(roc_auc_score(y_bin, oof_proba[:, i]))
        except ValueError:
            pass
    return {
        "n_samples": int(X.shape[0]),
        "n_classes": len(classes),
        "classes": classes,
        "class_counts": {c: int((y_multi == cls_to_int[c]).sum())
                         for c in classes},
        "per_class_auc": oof_per_class_auc,
        "macro_auc_mean": float(np.mean(macro_aucs)) if macro_aucs else float("nan"),
        "macro_auc_std": float(np.std(macro_aucs)) if macro_aucs else float("nan"),
        "top1_accuracy_mean": float(np.mean(top1_accs)) if top1_accs else float("nan"),
        "top1_accuracy_std": float(np.std(top1_accs)) if top1_accs else float("nan"),
        "top2_accuracy_mean": float(np.mean(top2_accs)) if top2_accs else float("nan"),
        "top2_accuracy_std": float(np.std(top2_accs)) if top2_accs else float("nan"),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(LABELS_MULTICLASS),
                    help="Path to labels_multiclass.tsv")
    ap.add_argument("--features-dir", default=str(FEAT_DIR),
                    help="Directory with per-sample .npy / .fsd.json artifacts")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output JSON path")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help=f"Number of CV seeds (default: {len(SEEDS)})")
    ap.add_argument("--folds", type=int, default=N_FOLDS,
                    help=f"Number of CV folds (default: {N_FOLDS})")
    ap.add_argument("--min-class-n", type=int, default=MIN_CLASS_N,
                    help=f"Drop cancer classes with n < this "
                         f"(default: {MIN_CLASS_N})")
    ap.add_argument("--pca-n", type=int, default=PCA_N,
                    help=f"PCA components (default: {PCA_N}); 0 to disable")
    ap.add_argument("--quick", action="store_true",
                    help="Quick smoke run: 1 seed × 2 folds (for tests)")
    ap.add_argument("--include-cristiano-only-ablation",
                    action="store_true",
                    help="Add a within-Cristiano (no-Jiang) ablation row — "
                         "removes the Jiang HCC_J batch-effect contribution "
                         "to the OvR HCC_J AUC. Slower (a second 5×5 CV).")
    args = ap.parse_args(argv)

    if args.quick:
        seeds = [SEEDS[0]]
        n_folds = 2
    else:
        seeds = SEEDS[: max(1, args.seeds)]
        n_folds = max(2, args.folds)

    t0 = time.time()
    labels_multiclass = build_labels_multiclass(Path(args.labels))
    counts = Counter(labels_multiclass.values())

    # TOO label space = cancer classes only (HEALTHY dropped)
    keep = {c for c in TOO_CLASSES if counts.get(c, 0) >= args.min_class_n}
    dropped = sorted([(c, counts.get(c, 0))
                      for c in TOO_CLASSES if c not in keep],
                     key=lambda x: x[1])
    print(f"Loaded {len(labels_multiclass)} disease labels")
    cancer_counts = {c: counts.get(c, 0) for c in TOO_CLASSES}
    print(f"Cancer class counts: {cancer_counts}")
    print(f"Keeping (cancer-only, n >= {args.min_class_n}): {sorted(keep)}")
    if dropped:
        print(f"Dropped (n < {args.min_class_n}): {dropped}")
    print(f"Total cancer samples kept (before feature filter): "
          f"{sum(counts.get(c, 0) for c in keep)}")

    X, y_multi, classes, cls_to_int, _sample_ids = load_cohort_cancer_only(
        labels_multiclass, keep=keep, feat_dir=args.features_dir)
    if X.shape[0] == 0:
        print(f"ERROR: No features found in {args.features_dir}")
        print("  Check that data/features/ contains the cfDNA feature cache.")
        return 2
    print(f"Final cohort: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(classes)} cancer classes ({classes})")
    print(f"Class counts after feature filter: "
          f"{dict(Counter([classes[i] for i in y_multi]))}")

    # OvR logistic regression CV
    (oof_proba, per_class_aucs_seed, macro_aucs,
     top1_accs, top2_accs) = ovr_logreg_cv(
        X, y_multi, classes, seeds=seeds, n_folds=n_folds, pca_n=args.pca_n)

    y_pred = np.argmax(oof_proba, axis=1)
    cm = confusion_matrix(y_multi, y_pred, labels=np.arange(len(classes)))

    # Optional within-Cristiano ablation (drops Jiang HCC_J to remove the
    # cross-protocol batch effect that artificially inflates HCC_J AUC)
    cristiano_only = None
    if args.include_cristiano_only_ablation:
        cristiano_only = _cristiano_only_ablation(
            labels_multiclass, classes, feat_dir=args.features_dir,
            seeds=seeds, n_folds=n_folds, pca_n=args.pca_n)

    # Pooled-OOF per-class AUC (single number per class)
    oof_per_class_auc = {}
    for i, c in enumerate(classes):
        y_bin = (y_multi == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        try:
            oof_per_class_auc[c] = float(
                roc_auc_score(y_bin, oof_proba[:, i]))
        except ValueError:
            pass

    top1_pooled = float(accuracy_score(y_multi, y_pred))
    try:
        top2_pooled = float(top_k_accuracy_score(
            y_multi, oof_proba, k=2,
            labels=np.arange(len(classes))))
    except ValueError:
        top2_pooled = float("nan")

    out = {
        # Cohort
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": len(classes),
        "classes": classes,
        "class_counts": {c: int((y_multi == cls_to_int[c]).sum())
                         for c in classes},
        "dropped_classes": [{"class": c, "n": int(n)} for c, n in dropped],
        # CV setup
        "seeds": seeds,
        "n_folds": n_folds,
        "pca_n": args.pca_n,
        "lr_C": LR_C,
        "min_class_n": args.min_class_n,
        "cancer_only": True,  # marker — TOO is cancer-only by definition
        # TOO metrics (primary)
        "per_class_auc": oof_per_class_auc,
        "macro_auc_mean": (float(np.mean(macro_aucs))
                           if macro_aucs else float("nan")),
        "macro_auc_std": (float(np.std(macro_aucs))
                          if macro_aucs else float("nan")),
        "top1_accuracy_pooled_oof": top1_pooled,
        "top2_accuracy_pooled_oof": top2_pooled,
        "top1_accuracy_mean": (float(np.mean(top1_accs))
                               if top1_accs else float("nan")),
        "top1_accuracy_std": (float(np.std(top1_accs))
                              if top1_accs else float("nan")),
        "top2_accuracy_mean": (float(np.mean(top2_accs))
                               if top2_accs else float("nan")),
        "top2_accuracy_std": (float(np.std(top2_accs))
                              if top2_accs else float("nan")),
        # Per-seed per-class AUC for stability reporting
        "per_class_auc_mean_across_seeds": {
            c: float(np.mean(per_class_aucs_seed[c]))
            for c in classes if per_class_aucs_seed[c]},
        "per_class_auc_std_across_seeds": {
            c: float(np.std(per_class_aucs_seed[c]))
            for c in classes if per_class_aucs_seed[c]},
        # Confusion matrix
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
        # Optional within-Cristiano ablation
        "cristiano_only_ablation": cristiano_only,
        "runtime_seconds": round(time.time() - t0, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=" * 64)
    print(f"Tissue-of-Origin (cancer-only, {len(classes)} classes, "
          f"{len(seeds)} seed × {n_folds}-fold CV, pooled OOF)")
    print("=" * 64)
    print(f"{'Class':<10} {'n':>5} {'OOF AUC':>9}")
    for c in classes:
        n = int((y_multi == cls_to_int[c]).sum())
        auc = oof_per_class_auc.get(c, float("nan"))
        print(f"{c:<10} {n:>5} {auc:>9.4f}")
    print()
    print(f"Macro AUC (mean ± std across seeds): "
          f"{out['macro_auc_mean']:.4f} ± {out['macro_auc_std']:.4f}")
    print(f"Top-1 accuracy (pooled OOF)        : {top1_pooled:.4f}")
    print(f"Top-2 accuracy (pooled OOF)        : {top2_pooled:.4f}")
    print(f"Top-1 accuracy (mean ± std seeds)  : "
          f"{out['top1_accuracy_mean']:.4f} ± {out['top1_accuracy_std']:.4f}")
    print(f"Top-2 accuracy (mean ± std seeds)  : "
          f"{out['top2_accuracy_mean']:.4f} ± {out['top2_accuracy_std']:.4f}")
    # Chance baseline = 1/n_classes (uniform)
    chance = 1.0 / len(classes)
    top1_lift = top1_pooled / chance if chance else float("nan")
    print(f"Chance top-1 baseline (1/K = 1/{len(classes)}) "
          f"= {chance:.4f}; observed lift = {top1_lift:.2f}×")
    if cristiano_only is not None:
        co = cristiano_only
        print()
        print("-" * 64)
        print("Within-Cristiano ablation (HCC_J dropped, batch-effect removed)")
        print("-" * 64)
        print(f"Cohort: {co['n_samples']} samples, {co['n_classes']} classes "
              f"({co['classes']})")
        print(f"{'Class':<10} {'n':>5} {'OOF AUC':>9}")
        for c in co["classes"]:
            n = co["class_counts"][c]
            auc = co["per_class_auc"].get(c, float("nan"))
            print(f"{c:<10} {n:>5} {auc:>9.4f}")
        print(f"Macro AUC : {co['macro_auc_mean']:.4f} ± "
              f"{co['macro_auc_std']:.4f}")
        print(f"Top-1 acc : {co['top1_accuracy_mean']:.4f} ± "
              f"{co['top1_accuracy_std']:.4f}")
        print(f"Top-2 acc : {co['top2_accuracy_mean']:.4f} ± "
              f"{co['top2_accuracy_std']:.4f}")
    print()
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
