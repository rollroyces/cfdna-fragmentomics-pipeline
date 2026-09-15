#!/usr/bin/env python3
"""Per-cancer AUC table for the cfDNA fragmentomics pipeline.

Computes one-vs-rest AUC for each cancer type against all other samples
(cancer + healthy combined). Uses the same 5-channel feature set,
5-seed × 5-fold pooled OOF, LR + PCA(200) OvR protocol as
multiclass_classification.py, but reports a single per-cancer AUC table
optimized for the sens-optimization report.

Differences vs multiclass_classification.py:
  - Reports per-cancer AUC for ALL cancers (not dropped at n<30).
    n=27 (CRC) is small but reportable; the row is flagged.
  - Adds bootstrap 95% CI per cancer (DeLong-style not available
    without scipy; we use percentile bootstrap on the seed-mean).
  - Adds per-cancer Sens@99% spec, which is what the v3 design doc
    asks for.
  - Output schema: results/per_cancer_auc.json with both .json and
    .md rendering.

Usage:
    python scripts/per_cancer_auc.py [--min-class-n 0]
    python scripts/per_cancer_auc.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from honest_benchmark import load5
from train_classifier import _harmonize
from multiclass_classification import build_labels_multiclass, load_cohort_5ch
from sens_at_specificity import (
    sens_at_specificity,
    bootstrap_ci,
)

DEFAULT_FEAT_DIR = "data/features"
DEFAULT_SEEDS = [42, 13, 7, 99, 1234]
PCA_N = 200
LR_C = 1.0


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


def per_cancer_oof(
    X: np.ndarray,
    y_multi: np.ndarray,
    classes: list[str],
    seeds: list[int],
    n_folds: int,
    pca_n: int,
    specificities: list[float],
    n_boot: int,
    bootstrap_seed: int,
) -> dict:
    """One-vs-rest pooled OOF across seeds, per-cancer AUC + sens@spec."""
    n, n_classes = X.shape[0], len(classes)
    oof_proba = np.zeros((n, n_classes), dtype=float)
    per_seed_aucs: dict[str, list[float]] = {c: [] for c in classes}
    per_seed_sens99: dict[str, list[float]] = {c: [] for c in classes}

    for sd in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=sd)
        oof_seed = np.zeros((n, n_classes), dtype=float)
        for tr, te in cv.split(X, y_multi):
            for ci in range(n_classes):
                ytr_bin = (y_multi[tr] == ci).astype(int)
                if ytr_bin.sum() == 0 or ytr_bin.sum() == len(ytr_bin):
                    continue
                oof_seed[te, ci] = _lr_pipeline(X[tr], ytr_bin, X[te],
                                                seed=sd, pca_n=pca_n)
        # Per-class AUC + Sens@99% per seed
        for ci, cls in enumerate(classes):
            y_bin = (y_multi == ci).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            try:
                a = roc_auc_score(y_bin, oof_seed[:, ci])
                per_seed_aucs[cls].append(a)
            except ValueError:
                continue
            try:
                s99, _ = sens_at_specificity(y_bin, oof_seed[:, ci], 0.99)
                per_seed_sens99[cls].append(s99)
            except Exception:
                continue
        oof_proba += oof_seed
    oof_proba /= len(seeds)

    # Pooled OOF AUC per class + bootstrap CI
    rows = {}
    for ci, cls in enumerate(classes):
        y_bin = (y_multi == ci).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        auc_pooled = float(roc_auc_score(y_bin, oof_proba[:, ci]))
        # Per-seed mean AUC + std
        seed_aucs = per_seed_aucs[cls]
        seed_sens = per_seed_sens99[cls]
        # Bootstrap CI on the pooled score (resample n with replacement)
        rng = np.random.default_rng(bootstrap_seed)
        auc_bs = np.empty(n_boot, dtype=float)
        sens_bs = np.empty(n_boot, dtype=float)
        N = len(y_bin)
        for b in range(n_boot):
            idx = rng.integers(0, N, size=N)
            try:
                auc_bs[b] = roc_auc_score(y_bin[idx], oof_proba[idx, ci])
            except ValueError:
                auc_bs[b] = np.nan
            s, _ = sens_at_specificity(y_bin[idx], oof_proba[idx, ci], 0.99)
            sens_bs[b] = s
        valid_auc = auc_bs[~np.isnan(auc_bs)]
        valid_sens = sens_bs[~np.isnan(sens_bs)]
        n_pos = int(y_bin.sum())
        rows[cls] = {
            "n_samples": N,
            "n_positive": n_pos,
            "auc_pooled": auc_pooled,
            "auc_seed_mean": float(np.mean(seed_aucs)) if seed_aucs else None,
            "auc_seed_std": float(np.std(seed_aucs)) if seed_aucs else None,
            "auc_ci95_lo": float(np.quantile(valid_auc, 0.025)),
            "auc_ci95_hi": float(np.quantile(valid_auc, 0.975)),
            "sens_at_99pct_pooled": float(sens_at_specificity(
                y_bin, oof_proba[:, ci], 0.99)[0]),
            "sens_at_99pct_seed_mean": float(np.mean(seed_sens))
                if seed_sens else None,
            "sens_at_99pct_ci95_lo": float(np.quantile(valid_sens, 0.025)),
            "sens_at_99pct_ci95_hi": float(np.quantile(valid_sens, 0.975)),
            "small_n_flag": n_pos < 30,
        }
    return rows


def render_markdown(rows: dict, classes: list[str]) -> str:
    """Render a markdown table summarizing per-cancer AUC."""
    lines = [
        "# Per-cancer AUC table (OvR, pooled 5-seed × 5-fold OOF)",
        "",
        "Baseline protocol: LR (C=1.0) + PCA(200) on the 5-channel feature",
        "set, harmonized per study. Same as `multiclass_classification.py`.",
        "",
        "| Cancer | n_pos | AUC (pooled) | AUC 95% CI | Sens@99% (pooled) | Sens@99% 95% CI | Notes |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for cls in classes:
        r = rows.get(cls)
        if r is None:
            lines.append(f"| {cls} | - | - | - | - | - | excluded |")
            continue
        flag = "n<30 (exploratory)" if r["small_n_flag"] else ""
        lines.append(
            f"| {cls} | {r['n_positive']} | "
            f"{r['auc_pooled']:.4f} | "
            f"[{r['auc_ci95_lo']:.4f}, {r['auc_ci95_hi']:.4f}] | "
            f"{r['sens_at_99pct_pooled']:.4f} | "
            f"[{r['sens_at_99pct_ci95_lo']:.4f}, "
            f"{r['sens_at_99pct_ci95_hi']:.4f}] | {flag} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=DEFAULT_FEAT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca", type=int, default=PCA_N)
    ap.add_argument("--min-class-n", type=int, default=0,
                    help="Drop classes with n<this; default 0 (report all)")
    ap.add_argument("--specificities", type=float, nargs="+",
                    default=[0.95, 0.98, 0.99, 0.995, 0.999])
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--out",
                    default="results/per_cancer_auc.json")
    ap.add_argument("--out-md",
                    default="results/per_cancer_auc.md")
    args = ap.parse_args()

    t0 = time.time()
    labels_multiclass = build_labels_multiclass(
        Path("labels_multiclass.tsv"))
    # Restrict to all classes (regardless of n) by passing min_class_n=0
    keep = set(labels_multiclass.values())
    X, y_multi, _y_bin, classes, _cls_to_int, _sample_ids = load_cohort_5ch(
        labels_multiclass, keep=keep, feat_dir=args.features_dir)
    print(f"Cohort: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Classes: {classes}")
    for c in classes:
        n = int((y_multi == classes.index(c)).sum())
        print(f"  {c:<10} n={n}")

    rows = per_cancer_oof(
        X, y_multi, classes, args.seeds, args.folds, args.pca,
        args.specificities, args.n_bootstrap, args.bootstrap_seed)

    # Macro AUC over OvR AUCs
    aucs = [r["auc_pooled"] for r in rows.values()]
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")

    payload = {
        "cohort": {"n_total": int(X.shape[0]),
                   "n_features": int(X.shape[1]),
                   "classes": classes,
                   "class_counts": {c: int((y_multi == classes.index(c)).sum())
                                    for c in classes}},
        "config": {"seeds": args.seeds,
                   "n_folds": args.folds,
                   "pca_n": args.pca,
                   "lr_C": LR_C,
                   "n_bootstrap": args.n_bootstrap,
                   "bootstrap_seed": args.bootstrap_seed},
        "per_cancer": rows,
        "macro_ovr_auc_pooled": macro_auc,
        "min_auc_per_cancer": float(min(aucs)) if aucs else None,
        "wall_seconds": round(time.time() - t0, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    md = render_markdown(rows, classes)
    with open(args.out_md, "w") as f:
        f.write(md)

    print()
    print("=" * 80)
    print(f"{'Cancer':<10} {'n':>5} {'AUC':>9} {'AUC 95% CI':>20} "
          f"{'Sens@99%':>10} {'Sens@99% 95% CI':>20}")
    print("=" * 80)
    for c in classes:
        r = rows.get(c)
        if r is None:
            print(f"{c:<10} -")
            continue
        flag = " <30" if r["small_n_flag"] else ""
        print(f"{c:<10}{flag} {r['n_positive']:>5} "
              f"{r['auc_pooled']:>9.4f} "
              f"[{r['auc_ci95_lo']:.3f},{r['auc_ci95_hi']:.3f}]"
              f"{'':>5} "
              f"{r['sens_at_99pct_pooled']:>9.4f} "
              f"[{r['sens_at_99pct_ci95_lo']:.3f},"
              f"{r['sens_at_99pct_ci95_hi']:.3f}]")
    print("=" * 80)
    print(f"Macro OvR AUC (pooled): {macro_auc:.4f}")
    print(f"Min per-cancer AUC:      {payload['min_auc_per_cancer']:.4f}")
    print(f"\nWrote {args.out} and {args.out_md} in "
          f"{payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())