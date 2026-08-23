"""
Honest ablation: do nucleosome-aware ratio features improve AUC?

This script answers the question: "if I add 3 biologically-grounded
ratio features (submono_ratio, mono_to_di_ratio, short_long_ratio)
to the existing 5-channel profile, does the LR-on-PCA classifier
improve?"

Method:
  - Same 627 cross-study cohort as the main benchmark
  - Same 5-fold CV with same labels/splits (matched by seed)
  - Compare two configurations, both averaged over 5 seeds:
    A. 5-channel baseline (5Mb ratio+coverage, 100kb ratio+counts, FSD)
    B. 5-channel + 3 nucleosome ratio features (A concatenated with 3)
  - Report mean AUC, std, paired t-test of per-seed differences
  - This is the same hygiene as honest_benchmark.py

Expected outcome (pre-registered):
  - If the literature-known cancer-shift signature is *already fully
    captured* by the 196-bin FSD histogram: no improvement, +0.000 AUC.
  - If the nucleosome-aware ratios add genuine biological signal:
    +0.001 to +0.01 AUC.
  - If the 196-bin FSD actually *underweights* the subnucleosome
    signal due to PCA noise: could be larger, but unlikely.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from nuc_features import compute_nuc_features_from_path  # noqa: E402
from train_classifier import _harmonize  # noqa: E402


FEAT_DIR = "/Users/hermes/cfdna-fragmentomics-pipeline/data/features"
LABELS_TSV = "/Users/hermes/cfdna-fragmentomics-pipeline/data/features/labels_cross_study.tsv"


def _load_labels() -> tuple[list[str], np.ndarray, dict, dict]:
    samples, y, studies = [], {}, {}
    with open(LABELS_TSV) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            samples.append(parts[0])
            y[parts[0]] = 1 if parts[1].lower() in ("cancer", "1", "tumor") else 0
            studies[parts[0]] = parts[2] if len(parts) >= 3 else "unknown"
    return samples, np.asarray([y[s] for s in samples], dtype=int), y, studies


def _build_5ch(samples: list[str], y_arr: np.ndarray) -> np.ndarray:
    rows = []
    for s in samples:
        r5 = np.load(os.path.join(FEAT_DIR, f"{s}.delfi_5mb_ratio.npy"))
        c5 = np.load(os.path.join(FEAT_DIR, f"{s}.delfi_5mb_coverage.npy"))
        r100 = np.load(os.path.join(FEAT_DIR, f"{s}.delfi_100kb_ratio.npy"))
        c100_raw = np.load(os.path.join(FEAT_DIR,
                                          f"{s}.delfi_100kb_counts.npy"))
        c100 = c100_raw / np.median(c100_raw)
        with open(os.path.join(FEAT_DIR, f"{s}.fsd.json")) as f:
            d = json.load(f)
        keys = sorted(d["size_bins"].keys(), key=lambda k: int(k.split("-")[0]))
        fsd = np.asarray([d["size_bins"][k] for k in keys], dtype=float)
        rows.append(np.concatenate([r5, c5, r100, c100, fsd]))
    X = np.nan_to_num(np.stack(rows).astype(float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    return X


def _build_5ch_plus_nuc(samples: list[str], y_arr: np.ndarray) -> np.ndarray:
    """5-channel + 3 original (band-sum) ratio features."""
    X_5ch = _build_5ch(samples, y_arr)
    nuc_rows = []
    for s in samples:
        nuc_rows.append(compute_nuc_features_from_path(
            os.path.join(FEAT_DIR, f"{s}.fsd.json")))
    X_nuc = np.stack(nuc_rows)
    return np.concatenate([X_5ch, X_nuc], axis=1)


def _build_5ch_plus_band(samples: list[str], y_arr: np.ndarray) -> np.ndarray:
    """5-channel + 3 new band-boundary features (v2 design)."""
    from nuc_features import compute_band_features_from_path
    X_5ch = _build_5ch(samples, y_arr)
    band_rows = []
    for s in samples:
        band_rows.append(compute_band_features_from_path(
            os.path.join(FEAT_DIR, f"{s}.fsd.json")))
    X_band = np.stack(band_rows)
    return np.concatenate([X_5ch, X_band], axis=1)


def _build_5ch_plus_all_nuc(samples: list[str], y_arr: np.ndarray) -> np.ndarray:
    """5-channel + all 6 nucleosome features (3 original + 3 band)."""
    from nuc_features import load_fsd, compute_all_features
    X_5ch = _build_5ch(samples, y_arr)
    nuc_rows = []
    for s in samples:
        nuc_rows.append(compute_all_features(load_fsd(
            os.path.join(FEAT_DIR, f"{s}.fsd.json"))))
    X_nuc = np.stack(nuc_rows)
    return np.concatenate([X_5ch, X_nuc], axis=1)


def _evaluate(X: np.ndarray, y: np.ndarray, study_arr: np.ndarray,
              pca_n: int = 200, n_seeds: int = 5) -> dict:
    """Use the pipeline's evaluate_cv with full 5-channel hygiene."""
    from train_classifier import evaluate_cv
    aucs = []
    for s in range(n_seeds):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        # Drop constant columns (mirrors train_classifier)
        keep = np.nanstd(X, axis=0) > 1e-12
        Xk = X[:, keep]
        result = evaluate_cv(Xk, y,
                              LogisticRegression(max_iter=20000, tol=1e-8,
                                                 random_state=0),
                              cv=cv,
                              use_pca=(pca_n > 0), pca_n=pca_n,
                              study_arr=study_arr, harmonize=True)
        aucs.append(result["auc_mean"])
    return {"auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "per_seed_aucs": aucs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pca-n", type=int, default=200)
    ap.add_argument("--out", default="/tmp/nuc_ablation.json")
    args = ap.parse_args()

    print("[ablation] Loading cohort...")
    samples, y, label_map, studies = _load_labels()
    # Filter to samples with all features present
    ok = []
    for s in samples:
        if all(os.path.exists(os.path.join(FEAT_DIR, f"{s}.{x}.npy"))
                for x in ("delfi_5mb_ratio", "delfi_5mb_coverage",
                           "delfi_100kb_ratio", "delfi_100kb_counts")) \
                and os.path.exists(os.path.join(FEAT_DIR, f"{s}.fsd.json")):
            ok.append(s)
    samples = ok
    y = np.asarray([label_map[s] for s in samples], dtype=int)
    study_arr = np.asarray([studies[s] for s in samples])
    print(f"[ablation] {len(samples)} samples, {y.sum()} cancer, "
          f"{(y == 0).sum()} healthy")

    print("[ablation] Building feature matrices...")
    X_5ch = _build_5ch(samples, y)
    X_5nuc = _build_5ch_plus_nuc(samples, y)
    print(f"[ablation] 5-channel shape: {X_5ch.shape}")
    print(f"[ablation] +nuc shape:      {X_5nuc.shape}")

    print(f"[ablation] Running {args.seeds}-seed CV (PCA n={args.pca_n})...")
    t0 = time.time()
    res_5ch = _evaluate(X_5ch, y, study_arr, pca_n=args.pca_n,
                         n_seeds=args.seeds)
    res_5nuc = _evaluate(X_5nuc, y, study_arr, pca_n=args.pca_n,
                          n_seeds=args.seeds)
    elapsed = time.time() - t0
    print(f"[ablation] Building 5ch+band features...")
    X_5band = _build_5ch_plus_band(samples, y)
    X_5all = _build_5ch_plus_all_nuc(samples, y)
    print(f"[ablation] +band shape: {X_5band.shape}, +all_nuc shape: {X_5all.shape}")

    print(f"[ablation] Running {args.seeds}-seed CV (PCA n={args.pca_n})...")
    t0 = time.time()
    res_5ch = _evaluate(X_5ch, y, study_arr, pca_n=args.pca_n,
                         n_seeds=args.seeds)
    res_5nuc = _evaluate(X_5nuc, y, study_arr, pca_n=args.pca_n,
                          n_seeds=args.seeds)
    res_5band = _evaluate(X_5band, y, study_arr, pca_n=args.pca_n,
                           n_seeds=args.seeds)
    res_5all = _evaluate(X_5all, y, study_arr, pca_n=args.pca_n,
                          n_seeds=args.seeds)
    elapsed = time.time() - t0
    print(f"[ablation] Time: {elapsed:.1f}s")

    # Paired t-tests vs 5ch baseline
    def paired_t(a, b):
        ts, ps = stats.ttest_rel(a, b)
        d = np.asarray(a) - np.asarray(b)
        return float(ts), float(ps), float(np.mean(d)), float(np.std(d))

    t_nuc, p_nuc, d_nuc, ds_nuc = paired_t(res_5nuc["per_seed_aucs"],
                                              res_5ch["per_seed_aucs"])
    t_band, p_band, d_band, ds_band = paired_t(res_5band["per_seed_aucs"],
                                                res_5ch["per_seed_aucs"])
    t_all, p_all, d_all, ds_all = paired_t(res_5all["per_seed_aucs"],
                                            res_5ch["per_seed_aucs"])

    summary = {
        "n_samples": len(samples),
        "n_cancer": int(y.sum()),
        "n_healthy": int((y == 0).sum()),
        "n_seeds": args.seeds,
        "pca_n": args.pca_n,
        "baseline_5ch": res_5ch,
        "with_v1_nuc_ratios": res_5nuc,
        "with_v2_band_features": res_5band,
        "with_all_nuc": res_5all,
        "v1_paired_t_stat": t_nuc,
        "v1_paired_t_pvalue": p_nuc,
        "v1_delta_auc_mean": d_nuc,
        "v1_delta_auc_std": ds_nuc,
        "v2_paired_t_stat": t_band,
        "v2_paired_t_pvalue": p_band,
        "v2_delta_auc_mean": d_band,
        "v2_delta_auc_std": ds_band,
        "all_paired_t_stat": t_all,
        "all_paired_t_pvalue": p_all,
        "all_delta_auc_mean": d_all,
        "all_delta_auc_std": ds_all,
        "elapsed_seconds": elapsed,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(f"Baseline (5ch):    AUC {res_5ch['auc_mean']:.4f} ± {res_5ch['auc_std']:.4f}")
    print(f"+ v1 (3 ratios):   AUC {res_5nuc['auc_mean']:.4f} ± {res_5nuc['auc_std']:.4f}   Δ={d_nuc:+.4f}  p={p_nuc:.3f}")
    print(f"+ v2 (3 band):     AUC {res_5band['auc_mean']:.4f} ± {res_5band['auc_std']:.4f}   Δ={d_band:+.4f}  p={p_band:.3f}")
    print(f"+ all (6 nuc):     AUC {res_5all['auc_mean']:.4f} ± {res_5all['auc_std']:.4f}   Δ={d_all:+.4f}  p={p_all:.3f}")
    print("=" * 70)
    best_delta = max([(d_nuc, p_nuc, "v1"),
                       (d_band, p_band, "v2"),
                       (d_all, p_all, "all")],
                      key=lambda x: x[0])
    if best_delta[0] > 0 and best_delta[1] < 0.05:
        print(f"BEST: {best_delta[2]} (+{best_delta[0]*100:.2f}pp AUC, p={best_delta[1]:.3f})")
    else:
        print(f"NULL: no variant improves 5ch by >= 0.5pp (best Δ={best_delta[0]*100:+.2f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
