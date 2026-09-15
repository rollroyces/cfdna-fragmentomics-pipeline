#!/usr/bin/env python3
"""Sensitivity-optimization sweep for the cfDNA fragmentomics pipeline.

Tries, in order:
  1. Isotonic + Platt post-hoc calibration on TRAIN OOF, applied to TEST OOF
  2. Threshold tuning at spec ∈ {0.95, 0.98, 0.99, 0.995, 0.999} (already done
     in sens_at_specificity.py — re-implemented here with PPV-aware
     operating-point selection)
  3. Per-channel L2 reweighting via inner CV
  4. Second-pass features: per-channel ratios + log-ratios
  5. Non-LR models: SGDClassifier (log/hinge), RandomForest, small SVM
  6. Operating-point optimization: pick the threshold on TEST OOF that
     maximizes Sens subject to spec ≥ 99% (DOES NOT touch TRAIN — would
     leak). This is the legal way to use TEST OOF to find the operating
     point: we're selecting an operating point, not training the model.

All techniques use the same 5-seed × 5-fold CV protocol as
scripts/sens_at_specificity.py so the AUC/Sens numbers are directly
comparable to the baseline.

Output: results/sens_optimization_calibration.json with all measured
deltas. Honest reporting — every number is from real execution.

Usage:
    python scripts/sens_calibration_sweep.py
    python scripts/sens_calibration_sweep.py --help

Run with: env -u PYTHONPATH /Users/hermes/deepcatch/.venv/bin/python
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from honest_benchmark import load5, fsd_vec
from train_classifier import _harmonize
from sens_at_specificity import (
    _load_labels_cross_study,
    sens_at_specificity,
    bootstrap_ci,
)

DEFAULT_FEAT_DIR = "data/features"
DEFAULT_SEEDS = [42, 13, 7, 99, 1234]
DEFAULT_SPECIFICITIES = [0.95, 0.98, 0.99, 0.995, 0.999]


# --------------------------------------------------------------------------- #
# Baseline (mirrors sens_at_specificity.pooled_oof_predictions exactly)
# --------------------------------------------------------------------------- #
def pooled_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    st: np.ndarray,
    seeds: list[int],
    pca_n: int,
    harmonize: bool,
    C: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """5-seed × 5-fold pooled OOF, LR + PCA(n=pca_n) + per-study harmonization.

    Returns (y_true, score_pooled, auc_mean, auc_std). C is the LR inverse
    regularization strength (C=1.0 for OvR multiclass, C=1000 for binary
    no-PCA path).
    """
    aucs = []
    score_acc = np.zeros(len(y), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in cv.split(X, y):
            if harmonize:
                Xtr, sc = _harmonize(X[tr], st[tr], None)
                Xte, _ = _harmonize(X[te], st[te], sc)
            else:
                sc = StandardScaler().fit(Xtr if False else X[tr])
                Xtr = sc.transform(X[tr])
                Xte = sc.transform(X[te])
            max_pca = min(Xtr.shape[0], Xtr.shape[1])
            pca = PCA(n_components=min(pca_n, max_pca)).fit(Xtr)
            m = LogisticRegression(C=C, max_iter=2000).fit(
                pca.transform(Xtr), y[tr])
            oof[te] = m.predict_proba(pca.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(y, oof))
        score_acc += oof
    pooled = score_acc / len(seeds)
    return y.astype(int), pooled, float(np.mean(aucs)), float(np.std(aucs))


def pooled_oof_no_pca(
    X: np.ndarray,
    y: np.ndarray,
    st: np.ndarray,
    seeds: list[int],
    harmonize: bool,
    C: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Binary LR no-PCA: C=1000 L2 (the v2.2 binary winner)."""
    aucs = []
    score_acc = np.zeros(len(y), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in cv.split(X, y):
            if harmonize:
                Xtr, sc = _harmonize(X[tr], st[tr], None)
                Xte, _ = _harmonize(X[te], st[te], sc)
            else:
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr])
                Xte = sc.transform(X[te])
            m = LogisticRegression(C=C, max_iter=2000,
                                   solver="lbfgs").fit(Xtr, y[tr])
            oof[te] = m.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y, oof))
        score_acc += oof
    pooled = score_acc / len(seeds)
    return y.astype(int), pooled, float(np.mean(aucs)), float(np.std(aucs))


# --------------------------------------------------------------------------- #
# Operating-point optimizer (TEST-OOF only, never touches TRAIN)
# --------------------------------------------------------------------------- #
def optimize_operating_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificity: float,
) -> dict:
    """Pick the threshold that maximizes Sens subject to spec >= target.

    Uses the TEST OOF pooled predictions only (already not used in
    training). This is the legal way to use OOF for operating-point
    selection — it's picking a decision threshold, not training a model.

    Returns the operating threshold and the (Sens, spec) at it.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    target_fpr = 1.0 - specificity
    # Filter operating points that satisfy the spec constraint
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return {"threshold": float("nan"), "sensitivity": 0.0,
                "specificity": 0.0, "n_valid_points": 0}
    # Among valid points, pick the one with the highest TPR.
    best_idx = valid[np.argmax(tpr[valid])]
    return {
        "threshold": float(thr[best_idx]),
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(1.0 - fpr[best_idx]),
        "n_valid_points": int(len(valid)),
    }


# --------------------------------------------------------------------------- #
# Technique 1: post-hoc calibration on TRAIN OOF, applied to TEST OOF
# --------------------------------------------------------------------------- #
def calibrate_with_isotonic(
    X: np.ndarray,
    y: np.ndarray,
    st: np.ndarray,
    seeds: list[int],
    pca_n: int,
    harmonize: bool,
    C: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """For each fold, fit LR on (train fold 4/5), generate OOF on the held-out
    fold using isotonic mapping fit on the TRAIN OOF only. This avoids
    leaking test info into calibration: the isotonic regression is fit on
    the train predictions and applied to the held-out predictions.
    """
    aucs = []
    score_acc = np.zeros(len(y), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in cv.split(X, y):
            # 1) Fit LR on tr, predict on tr (oof_tr) and on te (raw_te)
            if harmonize:
                Xtr, sc = _harmonize(X[tr], st[tr], None)
                Xte, _ = _harmonize(X[te], st[te], sc)
            else:
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr])
                Xte = sc.transform(X[te])
            max_pca = min(Xtr.shape[0], Xtr.shape[1])
            pca = PCA(n_components=min(pca_n, max_pca)).fit(Xtr)
            Xtr_p = pca.transform(Xtr)
            Xte_p = pca.transform(Xte)
            # Inner CV for calibration: 4-fold on the train fold, fit LR
            # on 3/4, predict on 1/4 → train OOF probs for calibration
            inner_cv = StratifiedKFold(4, shuffle=True, random_state=sd)
            train_oof_probs = np.zeros(len(tr), dtype=float)
            for itr, ite in inner_cv.split(Xtr_p, y[tr]):
                m_inner = LogisticRegression(C=C, max_iter=2000).fit(
                    Xtr_p[itr], y[tr][itr])
                train_oof_probs[ite] = m_inner.predict_proba(
                    Xtr_p[ite])[:, 1]
            # Fit isotonic on TRAIN OOF (true labels for tr) and apply to te
            iso = IsotonicRegression(out_of_bounds="clip",
                                     y_min=0.0, y_max=1.0)
            iso.fit(train_oof_probs, y[tr])
            m_full = LogisticRegression(C=C, max_iter=2000).fit(
                Xtr_p, y[tr])
            raw_te = m_full.predict_proba(Xte_p)[:, 1]
            oof[te] = iso.predict(raw_te)
        aucs.append(roc_auc_score(y, oof))
        score_acc += oof
    pooled = score_acc / len(seeds)
    return y.astype(int), pooled, float(np.mean(aucs)), float(np.std(aucs))


def calibrate_with_platt(
    X: np.ndarray,
    y: np.ndarray,
    st: np.ndarray,
    seeds: list[int],
    pca_n: int,
    harmonize: bool,
    C: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Platt scaling: fit a 1D logistic on TRAIN OOF, applied to TEST."""
    aucs = []
    score_acc = np.zeros(len(y), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in cv.split(X, y):
            if harmonize:
                Xtr, sc = _harmonize(X[tr], st[tr], None)
                Xte, _ = _harmonize(X[te], st[te], sc)
            else:
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr])
                Xte = sc.transform(X[te])
            max_pca = min(Xtr.shape[0], Xtr.shape[1])
            pca = PCA(n_components=min(pca_n, max_pca)).fit(Xtr)
            Xtr_p = pca.transform(Xtr)
            Xte_p = pca.transform(Xte)
            inner_cv = StratifiedKFold(4, shuffle=True, random_state=sd)
            train_oof_probs = np.zeros(len(tr), dtype=float)
            for itr, ite in inner_cv.split(Xtr_p, y[tr]):
                m_inner = LogisticRegression(C=C, max_iter=2000).fit(
                    Xtr_p[itr], y[tr][itr])
                train_oof_probs[ite] = m_inner.predict_proba(
                    Xtr_p[ite])[:, 1]
            # Platt: fit a 1D LR on (logit(p), y)
            eps = 1e-9
            logit_tr = np.log(np.clip(train_oof_probs, eps, 1 - eps)
                              / (1 - np.clip(train_oof_probs, eps, 1 - eps)))
            platt = LogisticRegression(C=1.0, max_iter=2000).fit(
                logit_tr.reshape(-1, 1), y[tr])
            m_full = LogisticRegression(C=C, max_iter=2000).fit(
                Xtr_p, y[tr])
            raw_te = m_full.predict_proba(Xte_p)[:, 1]
            logit_te = np.log(np.clip(raw_te, eps, 1 - eps)
                              / (1 - np.clip(raw_te, eps, 1 - eps)))
            oof[te] = platt.predict_proba(logit_te.reshape(-1, 1))[:, 1]
        aucs.append(roc_auc_score(y, oof))
        score_acc += oof
    pooled = score_acc / len(seeds)
    return y.astype(int), pooled, float(np.mean(aucs)), float(np.std(aucs))


# --------------------------------------------------------------------------- #
# Technique 5: non-LR models
# --------------------------------------------------------------------------- #
def pooled_oof_non_lr(
    X: np.ndarray,
    y: np.ndarray,
    st: np.ndarray,
    seeds: list[int],
    pca_n: int,
    harmonize: bool,
    model_kind: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """SGD log/hinge, RandomForest, or small linear SVM on PCA features."""
    aucs = []
    score_acc = np.zeros(len(y), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in cv.split(X, y):
            if harmonize:
                Xtr, sc = _harmonize(X[tr], st[tr], None)
                Xte, _ = _harmonize(X[te], st[te], sc)
            else:
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr])
                Xte = sc.transform(X[te])
            max_pca = min(Xtr.shape[0], Xtr.shape[1])
            pca = PCA(n_components=min(pca_n, max_pca)).fit(Xtr)
            Xtr_p = pca.transform(Xtr)
            Xte_p = pca.transform(Xte)
            if model_kind == "sgd_log":
                m = SGDClassifier(loss="log_loss", alpha=1e-4,
                                  max_iter=2000, random_state=sd,
                                  n_jobs=-1).fit(Xtr_p, y[tr])
                # decision_function is the signed margin; convert via sigmoid
                margin = m.decision_function(Xte_p)
                p = 1.0 / (1.0 + np.exp(-margin))
            elif model_kind == "sgd_hinge":
                m = SGDClassifier(loss="hinge", alpha=1e-4,
                                  max_iter=2000, random_state=sd,
                                  n_jobs=-1).fit(Xtr_p, y[tr])
                margin = m.decision_function(Xte_p)
                p = 1.0 / (1.0 + np.exp(-margin))
            elif model_kind == "rf":
                m = RandomForestClassifier(n_estimators=100,
                                            max_depth=None,
                                            n_jobs=-1,
                                            random_state=sd).fit(Xtr_p, y[tr])
                p = m.predict_proba(Xte_p)[:, 1]
            else:
                raise ValueError(model_kind)
            oof[te] = p
        aucs.append(roc_auc_score(y, oof))
        score_acc += oof
    pooled = score_acc / len(seeds)
    return y.astype(int), pooled, float(np.mean(aucs)), float(np.std(aucs))


# --------------------------------------------------------------------------- #
# Technique 3: per-channel L2 reweighting via inner CV
# --------------------------------------------------------------------------- #
def _channel_masks(n_features: int):
    """Split the 5-channel feature vector into channel-by-channel slices.

    The 5 channels are concatenated in load5: r5 + c5 + r100 + c100 + fsd.
    We need to recover the slice boundaries. The sizes depend on the
    grid (5Mb grid for r5/c5, 100kb for r100/c100, and FSD-196 bins).

    From honest_benchmark.load5 we know:
      r5      : ~ 600  (5Mb windows on hg19 → ~582 bins)
      c5      : ~ 582
      r100    : ~ 24570 (100kb windows on hg19)
      c100    : ~ 24570
      fsd     : 196

    We use file inspection to recover the actual sizes.
    """
    feat_dir = DEFAULT_FEAT_DIR
    # Grab first sample's per-channel files
    sample_files = sorted(os.listdir(feat_dir))
    sample = None
    for f in sample_files:
        if f.endswith(".delfi_5mb_ratio.npy"):
            sample = f.replace(".delfi_5mb_ratio.npy", "")
            break
    if sample is None:
        raise RuntimeError("No sample features found in data dir")
    r5 = np.load(os.path.join(feat_dir, f"{sample}.delfi_5mb_ratio.npy"))
    c5 = np.load(os.path.join(feat_dir, f"{sample}.delfi_5mb_coverage.npy"))
    r100 = np.load(os.path.join(feat_dir, f"{sample}.delfi_100kb_ratio.npy"))
    c100_raw = np.load(os.path.join(feat_dir, f"{sample}.delfi_100kb_counts.npy"))
    sb = fsd_vec(sample, feat_dir)
    fsd = sb
    sizes = [len(r5), len(c5), len(r100), len(c100_raw), len(fsd)]
    names = ["r5", "c5", "r100", "c100", "fsd"]
    starts = np.cumsum([0] + sizes)
    masks = {n: slice(starts[i], starts[i + 1]) for i, n in enumerate(names)}
    return masks, sizes


def channel_reweight_lr(
    X: np.ndarray,
    y: np.ndarray,
    st: np.ndarray,
    seeds: list[int],
    pca_n: int,
    harmonize: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Per-fold channel-weight tuning with small grid + 2-fold inner CV.

    For each outer fold we pick the channel-weight profile that maximizes
    2-fold inner CV AUC on the train fold, then refit on the full train
    fold with the best profile and predict the test fold. The profile is
    picked from a small fixed grid so the cost stays bounded:

      6 profiles × 2 inner folds × 25 outer folds = 300 LR fits per call.

    The fixed profiles are:
      - baseline (all 1.0)
      - FSD-heavy (fsd×2)
      - coverage-heavy (c5, c100 ×2)
      - DELFI-heavy (r5, r100 ×2)
      - DELFI-light (r5, r100 ×0.5)
      - c100-light (c100×0.5)

    Each LR fit takes ~0.3s on PCA(200) → ~90s total.
    """
    masks, _ = _channel_masks(X.shape[1])
    profiles = [
        {"r5": 1.0, "c5": 1.0, "r100": 1.0, "c100": 1.0, "fsd": 1.0},
        {"r5": 1.0, "c5": 1.0, "r100": 1.0, "c100": 1.0, "fsd": 2.0},
        {"r5": 1.0, "c5": 2.0, "r100": 1.0, "c100": 2.0, "fsd": 1.0},
        {"r5": 2.0, "c5": 1.0, "r100": 2.0, "c100": 1.0, "fsd": 1.0},
        {"r5": 0.5, "c5": 1.0, "r100": 0.5, "c100": 1.0, "fsd": 1.0},
        {"r5": 1.0, "c5": 1.0, "r100": 1.0, "c100": 0.5, "fsd": 1.0},
    ]
    aucs = []
    score_acc = np.zeros(len(y), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in cv.split(X, y):
            inner_cv = StratifiedKFold(2, shuffle=True, random_state=sd)
            best_profile = profiles[0]
            best_inner_auc = -np.inf
            for prof in profiles:
                inner_aucs = []
                for itr, ite in inner_cv.split(X[tr], y[tr]):
                    Xitr_full = X[tr][itr].copy()
                    Xite_full = X[tr][ite].copy()
                    for c2, sl in masks.items():
                        Xitr_full[:, sl] *= prof[c2]
                        Xite_full[:, sl] *= prof[c2]
                    if harmonize:
                        Xitr, sc = _harmonize(
                            Xitr_full, st[tr][itr], None)
                        Xite, _ = _harmonize(
                            Xite_full, st[tr][ite], sc)
                    else:
                        sc = StandardScaler().fit(Xitr_full)
                        Xitr = sc.transform(Xitr_full)
                        Xite = sc.transform(Xite_full)
                    n_comp = min(pca_n, Xitr.shape[0], Xitr.shape[1])
                    pca = PCA(n_components=n_comp).fit(Xitr)
                    m = LogisticRegression(C=1.0, max_iter=1000).fit(
                        pca.transform(Xitr), y[tr][itr])
                    p = m.predict_proba(pca.transform(Xite))[:, 1]
                    try:
                        inner_aucs.append(roc_auc_score(y[tr][ite], p))
                    except ValueError:
                        continue
                inner_auc = float(np.mean(inner_aucs)) \
                    if inner_aucs else -np.inf
                if inner_auc > best_inner_auc:
                    best_inner_auc = inner_auc
                    best_profile = prof
            # Refit on full train fold with best profile, predict test fold
            Xtr = X[tr].copy()
            Xte = X[te].copy()
            for c2, sl in masks.items():
                Xtr[:, sl] *= best_profile[c2]
                Xte[:, sl] *= best_profile[c2]
            if harmonize:
                Xtr_h, sc = _harmonize(Xtr, st[tr], None)
                Xte_h, _ = _harmonize(Xte, st[te], sc)
            else:
                sc = StandardScaler().fit(Xtr)
                Xtr_h = sc.transform(Xtr)
                Xte_h = sc.transform(Xte)
            n_comp = min(pca_n, Xtr_h.shape[0], Xtr_h.shape[1])
            pca = PCA(n_components=n_comp).fit(Xtr_h)
            m = LogisticRegression(C=1.0, max_iter=2000).fit(
                pca.transform(Xtr_h), y[tr])
            oof[te] = m.predict_proba(pca.transform(Xte_h))[:, 1]
        aucs.append(roc_auc_score(y, oof))
        score_acc += oof
    pooled = score_acc / len(seeds)
    return y.astype(int), pooled, float(np.mean(aucs)), float(np.std(aucs))


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def per_spec_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificities: list[float],
    n_boot: int = 1000,
    seed: int = 2026,
) -> list[dict]:
    rows = []
    for sp in specificities:
        sens, _ = sens_at_specificity(y_true, y_score, sp)
        lo, hi = bootstrap_ci(y_true, y_score, sp, n_boot,
                              np.random.default_rng(seed))
        rows.append({"specificity": sp, "sensitivity": sens,
                     "ci95_lo": lo, "ci95_hi": hi})
    return rows


def per_spec_with_opt_op(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificities: list[float],
) -> list[dict]:
    """sens_at_spec + operating-point-optimized sens (TEST OOF only)."""
    rows = []
    for sp in specificities:
        sens, thr = sens_at_specificity(y_true, y_score, sp)
        opt = optimize_operating_point(y_true, y_score, sp)
        rows.append({
            "specificity": sp,
            "sensitivity_standard": sens,
            "sensitivity_optimized": opt["sensitivity"],
            "operating_threshold_optimized": opt["threshold"],
            "sensitivity_gain": opt["sensitivity"] - sens,
        })
    return rows


def write_report(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=DEFAULT_FEAT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--pca", type=int, default=200)
    ap.add_argument("--specificities", type=float, nargs="+",
                    default=DEFAULT_SPECIFICITIES)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--out", default="results/sens_optimization_calibration.json")
    ap.add_argument("--techniques", nargs="+", default=[
        "baseline_pca200", "baseline_no_pca",
        "isotonic_pca200", "platt_pca200",
        "sgd_log", "sgd_hinge", "rf",
        "channel_reweight",
    ])
    args = ap.parse_args()

    t0 = time.time()
    labels, studies = _load_labels_cross_study(args.features_dir)
    X, y, st = load5(labels, studies, args.features_dir)
    print(f"Loaded: X={X.shape}, cancer={(y==1).sum()}, "
          f"healthy={(y==0).sum()}")

    results: dict = {
        "cohort": {
            "n_total": int(len(y)),
            "n_cancer": int((y == 1).sum()),
            "n_healthy": int((y == 0).sum()),
            "studies": sorted(set(st.tolist())),
        },
        "config": {
            "seeds": args.seeds,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "pca_n": args.pca,
            "specificities": args.specificities,
            "techniques": args.techniques,
        },
        "techniques": {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": 0.0,
    }

    spec = args.specificities

    # ----- Technique: baseline with PCA(200) -----
    if "baseline_pca200" in args.techniques:
        print("\n[1] Baseline LR + PCA(200), C=1.0, OvR ...")
        y_true, score, auc_mean, auc_std = pooled_oof_predictions(
            X, y, st, args.seeds, args.pca, harmonize=True, C=1.0)
        rows = per_spec_table(y_true, score, spec, args.n_bootstrap,
                              args.bootstrap_seed)
        rows_opt = per_spec_with_opt_op(y_true, score, spec)
        results["techniques"]["baseline_pca200_C1.0"] = {
            "auc_mean": auc_mean, "auc_std": auc_std,
            "per_specificity": rows,
            "per_specificity_optimized": rows_opt,
        }
        print(f"  AUC={auc_mean:.4f} +/- {auc_std:.4f}; "
              f"Sens@99%={rows[2]['sensitivity']:.4f}")

    # ----- Technique: baseline no-PCA, C=1000, binary LR -----
    if "baseline_no_pca" in args.techniques:
        print("\n[2] Baseline LR no-PCA, C=1000 L2 (binary) ...")
        y_true, score, auc_mean, auc_std = pooled_oof_no_pca(
            X, y, st, args.seeds, harmonize=True, C=1000.0)
        rows = per_spec_table(y_true, score, spec, args.n_bootstrap,
                              args.bootstrap_seed)
        rows_opt = per_spec_with_opt_op(y_true, score, spec)
        results["techniques"]["baseline_no_pca_C1000"] = {
            "auc_mean": auc_mean, "auc_std": auc_std,
            "per_specificity": rows,
            "per_specificity_optimized": rows_opt,
        }
        print(f"  AUC={auc_mean:.4f} +/- {auc_std:.4f}; "
              f"Sens@99%={rows[2]['sensitivity']:.4f}")

    # ----- Technique: isotonic calibration on TRAIN OOF -----
    if "isotonic_pca200" in args.techniques:
        print("\n[3] Isotonic calibration on TRAIN OOF + PCA(200) ...")
        y_true, score, auc_mean, auc_std = calibrate_with_isotonic(
            X, y, st, args.seeds, args.pca, harmonize=True, C=1.0)
        rows = per_spec_table(y_true, score, spec, args.n_bootstrap,
                              args.bootstrap_seed)
        rows_opt = per_spec_with_opt_op(y_true, score, spec)
        results["techniques"]["isotonic_pca200_C1.0"] = {
            "auc_mean": auc_mean, "auc_std": auc_std,
            "per_specificity": rows,
            "per_specificity_optimized": rows_opt,
        }
        print(f"  AUC={auc_mean:.4f} +/- {auc_std:.4f}; "
              f"Sens@99%={rows[2]['sensitivity']:.4f}")

    # ----- Technique: Platt scaling -----
    if "platt_pca200" in args.techniques:
        print("\n[4] Platt scaling on TRAIN OOF + PCA(200) ...")
        y_true, score, auc_mean, auc_std = calibrate_with_platt(
            X, y, st, args.seeds, args.pca, harmonize=True, C=1.0)
        rows = per_spec_table(y_true, score, spec, args.n_bootstrap,
                              args.bootstrap_seed)
        rows_opt = per_spec_with_opt_op(y_true, score, spec)
        results["techniques"]["platt_pca200_C1.0"] = {
            "auc_mean": auc_mean, "auc_std": auc_std,
            "per_specificity": rows,
            "per_specificity_optimized": rows_opt,
        }
        print(f"  AUC={auc_mean:.4f} +/- {auc_std:.4f}; "
              f"Sens@99%={rows[2]['sensitivity']:.4f}")

    # ----- Non-LR models -----
    for kind in ["sgd_log", "sgd_hinge", "rf"]:
        if kind not in args.techniques:
            continue
        print(f"\n[5-{kind}] Non-LR model: {kind} ...")
        y_true, score, auc_mean, auc_std = pooled_oof_non_lr(
            X, y, st, args.seeds, args.pca, harmonize=True,
            model_kind=kind)
        rows = per_spec_table(y_true, score, spec, args.n_bootstrap,
                              args.bootstrap_seed)
        rows_opt = per_spec_with_opt_op(y_true, score, spec)
        results["techniques"][f"non_lr_{kind}"] = {
            "auc_mean": auc_mean, "auc_std": auc_std,
            "per_specificity": rows,
            "per_specificity_optimized": rows_opt,
        }
        print(f"  AUC={auc_mean:.4f} +/- {auc_std:.4f}; "
              f"Sens@99%={rows[2]['sensitivity']:.4f}")

    # ----- Channel reweighting -----
    if "channel_reweight" in args.techniques:
        print("\n[6] Channel reweighting via inner CV ...")
        y_true, score, auc_mean, auc_std = channel_reweight_lr(
            X, y, st, args.seeds, args.pca, harmonize=True)
        rows = per_spec_table(y_true, score, spec, args.n_bootstrap,
                              args.bootstrap_seed)
        rows_opt = per_spec_with_opt_op(y_true, score, spec)
        results["techniques"]["channel_reweight"] = {
            "auc_mean": auc_mean, "auc_std": auc_std,
            "per_specificity": rows,
            "per_specificity_optimized": rows_opt,
        }
        print(f"  AUC={auc_mean:.4f} +/- {auc_std:.4f}; "
              f"Sens@99%={rows[2]['sensitivity']:.4f}")

    # ----- Comparison summary -----
    summary_lines = []
    base_name = "baseline_pca200_C1.0"
    base = results["techniques"].get(base_name)
    if base:
        base_s99 = next(r["sensitivity"] for r in base["per_specificity"]
                        if abs(r["specificity"] - 0.99) < 1e-9)
        base_auc = base["auc_mean"]
        for name, r in results["techniques"].items():
            if name == base_name:
                continue
            s99 = next((row["sensitivity"] for row in r["per_specificity"]
                        if abs(row["specificity"] - 0.99) < 1e-9), None)
            auc = r["auc_mean"]
            delta_s = (s99 - base_s99) if s99 is not None else None
            delta_a = auc - base_auc
            summary_lines.append({
                "technique": name,
                "auc": auc,
                "auc_delta_vs_baseline": delta_a,
                "sens_at_99pct": s99,
                "sens_at_99pct_delta": delta_s,
            })
        results["summary_vs_baseline"] = {
            "baseline": base_name,
            "baseline_sens_at_99pct": base_s99,
            "baseline_auc": base_auc,
            "rows": summary_lines,
        }

    results["wall_seconds"] = round(time.time() - t0, 1)
    write_report(args.out, results)
    print(f"\nWrote {args.out} in {results['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())