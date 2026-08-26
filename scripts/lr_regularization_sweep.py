"""
Regularization sweep on LR no-PCA.

Tests: L2 (default) vs L1 vs ElasticNet at 5 different C values,
on the 627 cross-study cohort. Goal: see if any (penalty, C) combo
beats the LR-no-PCA default (L2, C=1.0) at AUC 0.976.

Pre-registered prediction: L1 at small C (heavy sparsity) might win by
zeroing out noisy bins. But with 60k features and 627 samples, L1 with
default C=1.0 already gives a sparse solution (~1k non-zero), so the
gain from further tuning is likely <0.001 AUC.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/Users/hermes/cfdna-fragmentomics-pipeline")
sys.path.insert(0, "/Users/hermes/cfdna-fragmentomics-pipeline/scripts")

from train_classifier import _harmonize  # noqa

FEAT_DIR = "/Users/hermes/cfdna-fragmentomics-pipeline/data/features"
LABELS_TSV = "/Users/hermes/cfdna-fragmentomics-pipeline/data/features/labels_cross_study.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--c-values", nargs="+", type=float,
                    default=[0.001, 0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--out", default="/tmp/lr_reg_sweep.json")
    ap.add_argument("--skip-l1", action="store_true",
                    help="Skip the L1 saga sweep. By default it runs "
                         "L1 at C=[0.01, 0.1, 1.0] which takes ~35 "
                         "minutes and produces all-zero coefficients "
                         "(see BENCHMARK.md Appendix E.2).")
    args = ap.parse_args()

    print("[reg_sweep] Loading cohort...")
    samples, y_dict, studies = [], {}, {}
    with open(LABELS_TSV) as f:
        for line in f:
            p = line.rstrip().split("\t")
            samples.append(p[0])
            y_dict[p[0]] = 1 if p[1].lower() == "cancer" else 0
            studies[p[0]] = p[2] if len(p) >= 3 else "unknown"
    ok = [s for s in samples
          if all(os.path.exists(f"{FEAT_DIR}/{s}.{x}.npy")
                for x in ('delfi_5mb_ratio', 'delfi_5mb_coverage',
                          'delfi_100kb_ratio', 'delfi_100kb_counts'))
          and os.path.exists(f"{FEAT_DIR}/{s}.fsd.json")]
    samples = ok
    y = np.asarray([y_dict[s] for s in samples], dtype=int)
    study = np.asarray([studies[s] for s in samples])

    rows = []
    for s in samples:
        r5 = np.load(f"{FEAT_DIR}/{s}.delfi_5mb_ratio.npy")
        c5 = np.load(f"{FEAT_DIR}/{s}.delfi_5mb_coverage.npy")
        r100 = np.load(f"{FEAT_DIR}/{s}.delfi_100kb_ratio.npy")
        c100_raw = np.load(f"{FEAT_DIR}/{s}.delfi_100kb_counts.npy")
        c100 = c100_raw / np.median(c100_raw)
        with open(f"{FEAT_DIR}/{s}.fsd.json") as f:
            d = json.load(f)
        keys = sorted(d["size_bins"].keys(), key=lambda k: int(k.split("-")[0]))
        fsd = np.asarray([d["size_bins"][k] for k in keys], dtype=float)
        rows.append(np.concatenate([r5, c5, r100, c100, fsd]))
    X = np.nan_to_num(np.stack(rows).astype(float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    keep = np.nanstd(X, axis=0) > 1e-12
    X = X[:, keep]
    print(f"[reg_sweep] X shape: {X.shape}")

    results = []
    # L2 (default) sweep
    for C in args.c_values:
        print(f"\n[reg_sweep] LR L2 C={C}...")
        t0 = time.time()
        aucs = []
        for s in range(args.seeds):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            yt, ys = [], []
            for tr, te in cv.split(X, y):
                Xtr_h, sc = _harmonize(X[tr], study[tr], None)
                Xte_h, _ = _harmonize(X[te], study[te], sc)
                m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                       max_iter=20000, tol=1e-8,
                                       random_state=0)
                m.fit(Xtr_h, y[tr])
                ys.extend(m.predict_proba(Xte_h)[:, 1].tolist())
                yt.extend(y[te].tolist())
            aucs.append(roc_auc_score(yt, ys))
        r = {"penalty": "l2", "C": C,
             "auc_mean": float(np.mean(aucs)),
             "auc_std": float(np.std(aucs)),
             "per_seed_aucs": aucs,
             "elapsed": time.time() - t0}
        print(f"  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  ({r['elapsed']:.1f}s)")
        results.append(r)

    # L1 sweep (only one C, just to see if sparsity helps)
    if not args.skip_l1:
      for C in [0.01, 0.1, 1.0]:
        print(f"\n[reg_sweep] LR L1 C={C}...")
        t0 = time.time()
        aucs = []
        for s in range(args.seeds):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            yt, ys = [], []
            for tr, te in cv.split(X, y):
                Xtr_h, sc = _harmonize(X[tr], study[tr], None)
                Xte_h, _ = _harmonize(X[te], study[te], sc)
                m = LogisticRegression(penalty="l1", C=C, solver="saga",
                                       max_iter=1000, tol=1e-3,
                                       random_state=0)
                m.fit(Xtr_h, y[tr])
                ys.extend(m.predict_proba(Xte_h)[:, 1].tolist())
                yt.extend(y[te].tolist())
            aucs.append(roc_auc_score(yt, ys))
            n_nonzero_seed = int(np.sum(np.abs(m.coef_[0]) > 1e-6))
        r = {"penalty": "l1", "C": C,
             "auc_mean": float(np.mean(aucs)),
             "auc_std": float(np.std(aucs)),
             "per_seed_aucs": aucs,
             "elapsed": time.time() - t0,
             "n_nonzero": n_nonzero_seed}
        print(f"  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
              f"({r['elapsed']:.1f}s, {r.get('n_nonzero', '?')}/{X.shape[1]} features)")
        results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'penalty':<8} {'C':>8} {'AUC':>10} {'std':>5}")
    print("-" * 60)
    for r in results:
        print(f"{r['penalty']:<8} {r['C']:>8.3f} {r['auc_mean']:>10.4f} {r['auc_std']:>5.4f}")
    print("=" * 60)
    best = max(results, key=lambda r: r["auc_mean"])
    print(f"BEST: {best['penalty']} C={best['C']}  AUC={best['auc_mean']:.4f}")

    with open(args.out, "w") as f:
        json.dump({"results": results, "n_samples": len(samples),
                   "interpretation": (
                       f"BEST: {best['penalty']} C={best['C']} AUC={best['auc_mean']:.4f}"
                       if best["auc_mean"] > 0.9760 else
                       "No variant beats L2 C=1.0 default (0.9760)")}, f, indent=2)


if __name__ == "__main__":
    main()
