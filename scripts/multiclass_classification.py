#!/usr/bin/env python3
"""Multi-cancer (one-vs-rest) classification on the cfDNA cohort.

Builds on the same 5-channel feature loader used by `scripts/honest_benchmark.py`
(5 Mb DELFI ratio + 5 Mb coverage + 100 kb ratio + 100 kb counts + FSD), trains
one OvR logistic-regression per cancer type, and reports per-class AUC,
macro AUC, top-2 accuracy, and a confusion matrix.

Comparison vs binary cancer-vs-healthy is reported on the *same* filtered cohort
so the "information loss" from binary -> multi-class is a like-for-like number.

Mapping from sample-ID prefix to disease_class is documented in
`build_labels_multiclass()` and matches `labels_multiclass.tsv`.

Usage:
    python scripts/multiclass_classification.py
    python scripts/multiclass_classification.py --help
    python scripts/multiclass_classification.py --seeds 1 --folds 2 \
        --out results/multiclass_smoke.json
"""
from __future__ import annotations

import argparse
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
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Reuse the repo's shared constants + feature loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FEAT_DIR, REPO_ROOT, RESULTS_DIR
from honest_benchmark import fsd_vec, load5

LABELS_MULTICLASS = REPO_ROOT / "labels_multiclass.tsv"
LABELS_CROSS_STUDY = FEAT_DIR / "labels_cross_study.tsv"
DEFAULT_OUT = RESULTS_DIR / "multiclass_classification.json"

SEEDS = [42, 13, 7, 99, 1234]
N_FOLDS = 5
MIN_CLASS_N = 30  # drop classes with n < 30 (per task spec)
PCA_N = 200
LR_C = 1.0


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def _decode_prefix(sample: str) -> str:
    """Decode FinaleDB sample-ID prefix -> disease_class.

    Mapping table (verified against the AUDIT_REPORT_2.md S1 deferred-analysis
    sample-count ranges n ∈ {9, 18, 27, 28, 54, 60, 79, 88} and the
    Cristiano 2019 paper, PMC6774252):
      CGPLH*   → HEALTHY    (Cristiano healthy)
      CGPLL*   → LUAD       (Cristiano Lung)
      CGPLB*   → BRCA       (Cristiano Breast)
      CGPLP*   → PAAD       (Cristiano Pancreatic)
      CGPLO*   → OV         (Cristiano Ovarian)
      CGCRC*   → CRC        (Cristiano Colorectal)
      CGST1-8* → OTHER_C    (Cristiano gastric / bile duct / other)
      H*, HOT* → HCC_J      (Jiang 2015 HCC)
      C3xx     → HEALTHY    (Jiang 2015 healthy controls)
      CGH1x    → HEALTHY    (Cristiano healthy outliers)
    """
    if not sample:
        return "UNKNOWN"
    if sample.startswith("HOT") or (sample.startswith("H")
                                    and sample[1:].isdigit()):
        return "HCC_J"
    p5 = sample[:5]
    if p5 == "CGPLH":
        return "HEALTHY"
    if p5 == "CGPLL":
        return "LUAD"
    if p5 == "CGPLB":
        return "BRCA"
    if p5 == "CGPLP":
        return "PAAD"
    if p5 == "CGPLO":
        return "OV"
    if p5 == "CGCRC":
        return "CRC"
    if p5[:4] == "CGST":
        return "OTHER_C"
    if sample.startswith("CGH"):
        return "HEALTHY"
    if sample.startswith("C") and sample[1:].isdigit():
        return "HEALTHY"
    return "UNKNOWN"


def build_labels_multiclass(labels_path: Path) -> dict[str, str]:
    """Read labels_multiclass.tsv -> {sample: disease_class}.

    Falls back to parsing the cross-study labels file if the multiclass one
    is missing, so the script is robust to either ordering of steps.
    """
    out: dict[str, str] = {}
    if not labels_path.exists():
        labels_path = LABELS_CROSS_STUDY
    with open(labels_path) as f:
        header = f.readline().strip().split("\t")
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sample = parts[0]
            if "disease_class" in header and len(parts) >= 4:
                cls = parts[1]
            else:
                cls = _decode_prefix(sample)
            if cls != "UNKNOWN":
                out[sample] = cls
    return out


# --------------------------------------------------------------------------- #
# Feature loading (5-channel pattern reused from honest_benchmark)
# --------------------------------------------------------------------------- #
def load_cohort_5ch(labels_multiclass: dict[str, str],
                    keep: set[str],
                    feat_dir: str):
    """Load 5-channel features for samples whose disease_class ∈ `keep`.

    Mirrors the honest_benchmark.load5 pattern but:
      - uses the disease_class instead of binary cancer/healthy for the
        primary label vector
      - returns the sample-IDs aligned to rows of X so we can compute both
        multi-class and binary targets from the same X
    """
    labels01 = {s: (0 if c == "HEALTHY" else 1)
                for s, c in labels_multiclass.items() if c in keep}
    studies = {s: ("jiang" if s.startswith(("H", "HOT"))
                   else "cristiano")
               for s in labels01}
    X, y_bin, _studies_arr = load5(labels01, studies, feat_dir=feat_dir)

    # load5 iterates sorted(labels01) — recover the same sample order
    sample_ids = []
    for s in sorted(labels01.keys()):
        paths = [os.path.join(feat_dir, f"{s}.delfi_5mb_ratio.npy"),
                 os.path.join(feat_dir, f"{s}.delfi_5mb_coverage.npy"),
                 os.path.join(feat_dir, f"{s}.delfi_100kb_ratio.npy"),
                 os.path.join(feat_dir, f"{s}.delfi_100kb_counts.npy")]
        if all(os.path.exists(p) for p in paths) and fsd_vec(s, feat_dir) is not None:
            sample_ids.append(s)
    assert len(sample_ids) == X.shape[0], (len(sample_ids), X.shape[0])

    classes = sorted(keep)
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y_multi = np.array([cls_to_int[labels_multiclass[s]] for s in sample_ids],
                       dtype=int)
    return X, y_multi, y_bin, classes, cls_to_int, sample_ids


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def _lr_pipeline(Xtr, ytr_bin, Xte, seed: int, pca_n: int) -> np.ndarray:
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    if pca_n and pca_n < Xtr.shape[1]:
        pca = PCA(n_components=min(pca_n, Xtr.shape[0] - 1),
                  random_state=seed).fit(Xtr)
        Xtr = pca.transform(Xtr); Xte = pca.transform(Xte)
    clf = LogisticRegression(C=LR_C, max_iter=1000,
                             solver="lbfgs", random_state=seed)
    clf.fit(Xtr, ytr_bin)
    return clf.predict_proba(Xte)[:, 1]


def ovr_logreg_cv(X: np.ndarray, y: np.ndarray, classes: list[str],
                  seeds: list[int], n_folds: int, pca_n: int):
    """One-vs-rest logistic regression, pooled OOF across seeds."""
    n, n_classes = X.shape[0], len(classes)
    oof_proba = np.zeros((n, n_classes), dtype=float)
    macro_aucs: list[float] = []
    top2_accs: list[float] = []
    per_class_aucs_seed: dict[str, list[float]] = {c: [] for c in classes}

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_seed = np.zeros((n, n_classes), dtype=float)
        for tr, te in cv.split(X, y):
            for ci in range(n_classes):
                ytr_bin = (y[tr] == ci).astype(int)
                if ytr_bin.sum() == 0 or ytr_bin.sum() == len(ytr_bin):
                    continue
                oof_seed[te, ci] = _lr_pipeline(X[tr], ytr_bin, X[te],
                                                seed=seed, pca_n=pca_n)
        # Per-class AUC for this seed
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
        try:
            top2_accs.append(top_k_accuracy_score(y, oof_seed, k=2,
                                                   labels=np.arange(n_classes)))
        except ValueError:
            pass
        oof_proba += oof_seed

    oof_proba /= len(seeds)
    return oof_proba, per_class_aucs_seed, macro_aucs, top2_accs


def binary_cv(X: np.ndarray, y_bin: np.ndarray,
              seeds: list[int], n_folds: int, pca_n: int):
    aucs = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof = np.zeros(len(X), dtype=float)
        for tr, te in cv.split(X, y_bin):
            oof[te] = _lr_pipeline(X[tr], y_bin[tr], X[te],
                                   seed=seed, pca_n=pca_n)
        aucs.append(roc_auc_score(y_bin, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


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
                    help=f"Drop classes with n < this (default: {MIN_CLASS_N})")
    ap.add_argument("--pca-n", type=int, default=PCA_N,
                    help=f"PCA components (default: {PCA_N}); 0 to disable")
    ap.add_argument("--quick", action="store_true",
                    help="Quick smoke run: 1 seed × 2 folds (for tests)")
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
    keep = {c for c, n in counts.items() if n >= args.min_class_n}
    dropped = sorted([(c, n) for c, n in counts.items()
                      if n < args.min_class_n], key=lambda x: x[1])
    print(f"Loaded {len(labels_multiclass)} disease labels")
    print(f"Class counts: {dict(counts)}")
    print(f"Keeping (n >= {args.min_class_n}): {sorted(keep)}")
    if dropped:
        print(f"Dropped (n < {args.min_class_n}): {dropped}")

    X, y_multi, y_bin, classes, cls_to_int, _sample_ids = load_cohort_5ch(
        labels_multiclass, keep=keep, feat_dir=args.features_dir)
    if X.shape[0] == 0 or len(X.shape) < 2:
        print(f"ERROR: No features found in {args.features_dir}")
        print(f"  Check that data/features/ contains the 627-sample cfDNA feature cache.")
        print(f"  See USAGE.md for instructions on regenerating features via fetch_finaledb.py")
        return 2
    print(f"Final cohort: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(classes)} classes ({classes})")

    # Multi-class OvR CV
    oof_proba, per_class_aucs_seed, macro_aucs, top2_accs = ovr_logreg_cv(
        X, y_multi, classes, seeds=seeds, n_folds=n_folds, pca_n=args.pca_n)

    # Confusion matrix (rows=true, cols=pred)
    y_pred = np.argmax(oof_proba, axis=1)
    cm = confusion_matrix(y_multi, y_pred, labels=np.arange(len(classes)))

    # Binary cancer-vs-healthy AUC on the SAME filtered cohort
    bin_auc_mean, bin_auc_std = binary_cv(X, y_bin, seeds=seeds,
                                          n_folds=n_folds, pca_n=args.pca_n)

    macro_mean = float(np.mean(macro_aucs))
    info_delta = macro_mean - bin_auc_mean

    # OOF AUC (pooled across seeds) per class — single number per class
    oof_per_class_auc = {c: float(roc_auc_score((y_multi == i).astype(int),
                                                oof_proba[:, i]))
                         for i, c in enumerate(classes)}

    out = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": len(classes),
        "classes": classes,
        "class_counts": {c: int((y_multi == cls_to_int[c]).sum())
                         for c in classes},
        "dropped_classes": [{"class": c, "n": int(n)} for c, n in dropped],
        "seeds": seeds,
        "n_folds": n_folds,
        "pca_n": args.pca_n,
        "lr_C": LR_C,
        "min_class_n": args.min_class_n,
        # Multi-class
        "oof_per_class_auc": oof_per_class_auc,
        "per_class_auc_mean_across_seeds": {
            c: float(np.mean(per_class_aucs_seed[c]))
            for c in classes if per_class_aucs_seed[c]
        },
        "per_class_auc_std_across_seeds": {
            c: float(np.std(per_class_aucs_seed[c]))
            for c in classes if per_class_aucs_seed[c]
        },
        "macro_auc_mean": macro_mean,
        "macro_auc_std": float(np.std(macro_aucs)),
        "top2_accuracy_mean": float(np.mean(top2_accs)) if top2_accs else float("nan"),
        "top2_accuracy_std": float(np.std(top2_accs)) if top2_accs else float("nan"),
        # Binary comparison
        "binary_cancer_vs_healthy_auc": float(bin_auc_mean),
        "binary_cancer_vs_healthy_auc_std": float(bin_auc_std),
        "information_delta_multi_vs_binary": float(info_delta),
        # Confusion matrix
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
        "runtime_seconds": round(time.time() - t0, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=" * 60)
    print(f"Multi-class OvR ({len(classes)} classes, "
          f"{len(seeds)} seed × {n_folds}-fold CV, pooled OOF)")
    print("=" * 60)
    print(f"{'Class':<10} {'n':>5} {'OOF AUC':>9}")
    for c in classes:
        n = int((y_multi == cls_to_int[c]).sum())
        auc = oof_per_class_auc[c]
        print(f"{c:<10} {n:>5} {auc:>9.4f}")
    print()
    print(f"Macro AUC (mean across seeds): {macro_mean:.4f} ± "
          f"{float(np.std(macro_aucs)):.4f}")
    print(f"Top-2 accuracy                : "
          f"{out['top2_accuracy_mean']:.4f} ± {out['top2_accuracy_std']:.4f}")
    print(f"Binary AUC (cancer vs healthy): "
          f"{bin_auc_mean:.4f} ± {bin_auc_std:.4f}")
    sign = "gain" if info_delta > 0 else "loss"
    print(f"Information delta (multi-binary): "
          f"{info_delta:+.4f}  ({sign})")
    print()
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
