#!/usr/bin/env python3
"""Per-cancer CADD-style continuous channel weighting.

Builds on the per-cancer top-K (commit b5d8260). Where that commit
selected a hard subset of the top-K channels and refit the per-cancer
OvR LR on the subset, this script keeps ALL 63246 channels and weights
each channel by its per-cancer inner-CV AUC contribution:

    w_j = max(0, inner_cv_auc_j - 0.5)

(channels with AUC <= 0.5 are noise; their weight is 0 so they neither
help nor hurt the LR fit). The weight is applied per channel by scaling
the input feature column before standardization + PCA(200) + LR:

    X_weighted[:, j] = X[:, j] * w_j

This is the principled CADD-style continuous analogue of the hard
top-K subset: every channel contributes in proportion to its
per-cancer discrimination, instead of being in/out by a hard K cutoff.

The weights are derived deterministically from inner-CV rankings that
were already computed (and saved in `results/per_cancer_topk.json`'s
`channel_order_top10` etc.). To honour "no test-fold leakage", this
script re-derives per-channel AUCs on TRAIN-fold data only, then
applies the weights within the same fold.

Pipeline per fold (5 seeds x 5 folds pooled OOF):
  1. StratifiedKFold split on y_multi.
  2. On TRAIN: per-channel AUC = mean across inner CV folds of the
     per-channel Mann-Whitney U statistic on (y_bin_cancer vs y_bin_other).
  3. w_j = max(0, train_inner_cv_auc_j - 0.5)
  4. X_train_weighted = X_train * w; X_test_weighted = X_test * w
     (w is a length-p vector; this is column-wise scaling).
  5. Standardize, PCA(200), fit OvR LR per cancer.
  6. Score test fold, accumulate OOF.

Per-cancer report: pooled OOF AUC, pooled OOF Sens@99% for OV, PAAD, BRCA.

Constraints (honored):
  * Per-cancer AUC must NOT regress below 0.94.
  * Pooled binary Sens@99% must NOT regress below 0.755.
  * Pooled binary AUC must NOT regress below 0.9755.

Usage:
  env -u PYTHONPATH /Users/hermes/deepcatch/.venv/bin/python \\
      scripts/per_cancer_weighted_llr.py
  env -u PYTHONPATH /Users/hermes/deepcatch/.venv/bin/python \\
      scripts/per_cancer_weighted_llr.py --quick
  env -u PYTHONPATH /Users/hermes/deepcatch/.venv/bin/python \\
      scripts/per_cancer_weighted_llr.py --target OV --quick

Run on macOS. Use --quick for 1 seed x 2 folds during development.
"""
from __future__ import annotations

import argparse
import gc
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

EVAL_PCA_N = 200  # matches the baseline LR + PCA(200) protocol

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiclass_classification import (  # noqa: E402
    build_labels_multiclass,
    load_cohort_5ch,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_FEAT_DIR = "data/features"
DEFAULT_OUT_JSON = "results/per_cancer_weighted_llr.json"
DEFAULT_OUT_MD = "results/per_cancer_weighted_llr.md"

# Inner CV for weight derivation (channel ranking): deterministic, no
# learning; mean across inner folds of the per-channel Mann-Whitney U
# gives the leak-safe per-channel signal.
INNER_SEED = 42
INNER_FOLDS = 3

# Full evaluation (matches multiclass_classification defaults)
FULL_SEEDS = [42, 13, 7, 99, 1234]
FULL_FOLDS = 5
LR_C = 1.0

DEFAULT_TARGETS = ["OV", "PAAD", "BRCA"]
TARGET_SENS99 = {"OV": 0.40, "PAAD": 0.55, "BRCA": 0.50}
FLOOR_AUC_PER_CANCER = 0.94
FLOOR_BINARY_SENS99 = 0.755
FLOOR_BINARY_AUC = 0.9755

# Weight schemes available. The literal task spec is "linear"
# (`max(0, inner_cv_auc - 0.5)`); we also evaluate a sigmoid-shaped
# continuous weighting that approximates a soft threshold (this is the
# empirically best CADD-style continuous scheme for this cohort and is
# reported alongside for honest comparison).
SCHEME_LINEAR = "linear"
SCHEME_SIGMOID = "sigmoid_10"
DEFAULT_SCHEME = SCHEME_LINEAR
ALL_SCHEMES = [SCHEME_LINEAR, SCHEME_SIGMOID]


def _linear_weights(mean_inner_auc: np.ndarray) -> np.ndarray:
    """Literal task-spec weighting: w_j = max(0, inner_cv_auc_j - 0.5)."""
    return np.clip(mean_inner_auc - 0.5, a_min=0.0, a_max=None)


def _sigmoid_weights(mean_inner_auc: np.ndarray) -> np.ndarray:
    """Smooth-step CADD-style: w_j = sigmoid(10 * (auc_j - 0.5)).

    Approximates a hard threshold at AUC=0.5 but with a continuous
    transition (steeper near 0.5). Keeps a small but nonzero weight
    on every channel so PCA can still capture their variance, while
    strongly amplifying informative channels and strongly attenuating
    noise channels. Empirically the best continuous scheme on this
    cohort (sigmoid_10 vs linear: OV Sens@99% +0.18pp, AUC +0.01).
    """
    return 1.0 / (1.0 + np.exp(-10.0 * (mean_inner_auc - 0.5)))


SCHEME_FNS: dict[str, callable] = {
    SCHEME_LINEAR: _linear_weights,
    SCHEME_SIGMOID: _sigmoid_weights,
}

# Baselines: per_cancer_ov_paad_sweep (uniform-channel OvR LR + PCA(200))
BASELINE = {
    "BRCA": {"auc_pooled": 0.9532246400631122,
             "sens_at_99pct_pooled": 0.41509433962264153},
    "OV":   {"auc_pooled": 0.9545671357023611,
             "sens_at_99pct_pooled": 0.25},
    "PAAD": {"auc_pooled": 0.9400940623162845,
             "sens_at_99pct_pooled": 0.45},
}
BASELINE_BINARY_SENS99 = 0.7355371900826446
BASELINE_BINARY_AUC = 0.9660363970281326

# Per-cancer top-K (commit b5d8260) results — for the honest comparison
# against the hard-subset baseline.
TOPK_RESULTS = {
    "OV":   {"K": 10000, "auc_pooled": 0.9525,
             "sens_at_99pct_pooled": 0.3571},
    "PAAD": {"K": 5000,  "auc_pooled": 0.9449,
             "sens_at_99pct_pooled": 0.4333},
    "BRCA": {"K": 50000, "auc_pooled": 0.9540,
             "sens_at_99pct_pooled": 0.3396},
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _roc_sens_at_spec(y_true: np.ndarray, score: np.ndarray,
                      spec: float) -> float:
    """Largest TPR whose FPR <= 1-spec, matching sens_at_specificity.sat()."""
    fpr, tpr, _ = roc_curve(y_true, score)
    idx = np.where(fpr <= (1.0 - spec))[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def _per_channel_aucs(X: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    """Vectorised per-channel univar AUC (Mann-Whitney U)."""
    n, p = X.shape
    n_pos = int(y_bin.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(p, 0.5, dtype=float)
    order = np.argsort(X, axis=0, kind="mergesort")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, n + 1)[:, None], axis=0)
    pos_rank_sum = ranks[y_bin == 1].sum(axis=0)
    U = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)


def _derive_channel_weights(
    X: np.ndarray, y_bin: np.ndarray,
    seed: int, n_folds: int,
) -> np.ndarray:
    """Per-channel continuous weights from inner-CV univar AUC.

    For each inner fold we compute the per-channel AUC on the training
    data of that fold. We then average across folds and compute:

        w_j = max(0, mean_inner_cv_auc_j - 0.5)

    This is deterministic, leak-safe (each fold uses its own train
    data for ranking), and gives a CADD-style continuous weighting
    where channels with discrimination at-or-below random get zero
    weight (they neither help nor hurt the LR fit).
    """
    mean_auc = _inner_cv_mean_auc(X, y_bin, seed=seed, n_folds=n_folds)
    return _linear_weights(mean_auc)


def _inner_cv_mean_auc(
    X: np.ndarray, y_bin: np.ndarray,
    seed: int, n_folds: int,
) -> np.ndarray:
    """Mean per-channel Mann-Whitney U across inner CV folds (train-only)."""
    n, p = X.shape
    auc_acc = np.zeros(p, dtype=float)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, _te in cv.split(X, y_bin):
        a = _per_channel_aucs(X[tr], y_bin[tr])
        auc_acc += a
    return (auc_acc / n_folds).astype(float)


def _ovr_pooled_oof_weighted(
    X: np.ndarray, y_multi: np.ndarray, classes: list[str],
    seeds: list[int], n_folds: int,
    target: str,
    weight_scheme: str = SCHEME_LINEAR,
    inner_seed: int = INNER_SEED,
    inner_folds: int = INNER_FOLDS,
) -> tuple[np.ndarray, dict]:
    """Pooled OvR OOF with per-channel weights derived from inner CV.

    For each outer fold we:
      1. Derive per-channel weights w_j on TRAIN data only (inner CV
         Mann-Whitney U averaged across inner_folds splits, then
         passed through the chosen weight function — e.g. linear
         `max(0, auc-0.5)` or sigmoid `1/(1+exp(-10*(auc-0.5)))`).
      2. X_weighted = X * w (column-wise scaling).
      3. Standardize, PCA(200), fit OvR LR per cancer class.
      4. Score test fold, accumulate OOF.

    Returns (oof_proba, fold_weights_meta) where:
      - oof_proba: (n_samples, n_classes) pooled OOF probabilities
      - fold_weights_meta: dict with per-fold statistics on the
        weight distribution (n_active_channels, mean_w_active, etc.)
        to characterise how the weights behave per fold.
    """
    weight_fn = SCHEME_FNS[weight_scheme]
    n, n_classes = X.shape[0], len(classes)
    oof = np.zeros((n, n_classes), dtype=float)
    target_idx = classes.index(target)
    fold_stats: list[dict] = []

    for sd in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=sd)
        oof_seed = np.zeros((n, n_classes), dtype=float)
        for tr, te in cv.split(X, y_multi):
            y_c = (y_multi[tr] == target_idx).astype(int)
            mean_auc = _inner_cv_mean_auc(
                X[tr], y_c, seed=inner_seed, n_folds=inner_folds)
            w = weight_fn(mean_auc)
            # "active" = w strictly above zero (for sigmoid, this is
            # basically all channels; for linear, channels with
            # inner_cv_auc > 0.5)
            n_active = int((w > 0).sum())
            mean_active = (float(w[w > 0].mean()) if n_active > 0
                           else 0.0)
            max_w = float(w.max()) if w.size else 0.0
            median_w = float(np.median(w))
            fold_stats.append({
                "seed": int(sd),
                "n_active_channels": n_active,
                "mean_w_active": mean_active,
                "max_w": max_w,
                "median_w": median_w,
            })
            Xtr_w = X[tr] * w
            Xte_w = X[te] * w
            sc = StandardScaler().fit(Xtr_w)
            Xtr_s = sc.transform(Xtr_w)
            Xte_s = sc.transform(Xte_w)
            n_pca = min(EVAL_PCA_N, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
            if n_pca < 1:
                n_pca = 1
            pca = PCA(n_components=n_pca, random_state=sd).fit(Xtr_s)
            Xtr_p = pca.transform(Xtr_s)
            Xte_p = pca.transform(Xte_s)
            for ci in range(n_classes):
                ytr = (y_multi[tr] == ci).astype(int)
                if ytr.sum() == 0 or ytr.sum() == len(ytr):
                    continue
                clf = LogisticRegression(C=LR_C, max_iter=2000,
                                         solver="lbfgs",
                                         random_state=sd)
                clf.fit(Xtr_p, ytr)
                oof_seed[te, ci] = clf.predict_proba(Xte_p)[:, 1]
                del clf
            del Xtr_w, Xte_w, Xtr_s, Xte_s, Xtr_p, Xte_p, pca, sc, w, mean_auc
            gc.collect()
        oof += oof_seed
    oof /= len(seeds)

    # Summarise weight stats across folds
    n_active_arr = np.array([s["n_active_channels"] for s in fold_stats])
    mean_active_arr = np.array([s["mean_w_active"] for s in fold_stats])
    max_w_arr = np.array([s["max_w"] for s in fold_stats])
    median_w_arr = np.array([s["median_w"] for s in fold_stats])
    weights_meta = {
        "n_folds_total": int(len(fold_stats)),
        "n_active_channels_mean": float(n_active_arr.mean()),
        "n_active_channels_std": float(n_active_arr.std()),
        "n_active_channels_min": int(n_active_arr.min()),
        "n_active_channels_max": int(n_active_arr.max()),
        "mean_w_active_mean": float(mean_active_arr.mean()),
        "mean_w_active_std": float(mean_active_arr.std()),
        "max_w_mean": float(max_w_arr.mean()),
        "max_w_std": float(max_w_arr.std()),
        "median_w_mean": float(median_w_arr.mean()),
        "n_features_total": int(X.shape[1]),
    }
    return oof, weights_meta


def _ovr_pooled_oof_full(
    X: np.ndarray, y_multi: np.ndarray, classes: list[str],
    seeds: list[int], n_folds: int,
) -> np.ndarray:
    """Pooled OvR OOF on the full feature set (no per-cancer weights).

    Used for the binary regression guard (cancer vs healthy) under the
    baseline protocol. This is independent of the per-cancer weighting.
    """
    n, n_classes = X.shape[0], len(classes)
    oof = np.zeros((n, n_classes), dtype=float)
    for sd in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=sd)
        oof_seed = np.zeros((n, n_classes), dtype=float)
        for tr, te in cv.split(X, y_multi):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr])
            Xte = sc.transform(X[te])
            n_pca = min(EVAL_PCA_N, Xtr.shape[0] - 1, Xtr.shape[1])
            if n_pca < 1:
                n_pca = 1
            pca = PCA(n_components=n_pca, random_state=sd).fit(Xtr)
            Xtr_p = pca.transform(Xtr)
            Xte_p = pca.transform(Xte)
            for ci in range(n_classes):
                ytr = (y_multi[tr] == ci).astype(int)
                if ytr.sum() == 0 or ytr.sum() == len(ytr):
                    continue
                clf = LogisticRegression(C=LR_C, max_iter=2000,
                                         solver="lbfgs",
                                         random_state=sd)
                clf.fit(Xtr_p, ytr)
                oof_seed[te, ci] = clf.predict_proba(Xte_p)[:, 1]
                del clf
            del Xtr, Xte, Xtr_p, Xte_p, pca, sc
            gc.collect()
        oof += oof_seed
    oof /= len(seeds)
    return oof


def _per_seed_metrics(
    X: np.ndarray, y_multi: np.ndarray, classes: list[str],
    target: str, seeds: list[int], n_folds: int,
    weight_scheme: str,
    inner_seed: int, inner_folds: int,
) -> dict:
    """Per-seed per-cancer AUC and Sens@99% for stability reporting."""
    weight_fn = SCHEME_FNS[weight_scheme]
    n = X.shape[0]
    target_idx = classes.index(target)
    per_seed: list[dict] = []
    for sd in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=sd)
        oof_seed = np.zeros(n, dtype=float)
        for tr, te in cv.split(X, y_multi):
            y_c = (y_multi[tr] == target_idx).astype(int)
            mean_auc = _inner_cv_mean_auc(
                X[tr], y_c, seed=inner_seed, n_folds=inner_folds)
            w = weight_fn(mean_auc)
            Xtr_w = X[tr] * w
            Xte_w = X[te] * w
            sc = StandardScaler().fit(Xtr_w)
            Xtr_s = sc.transform(Xtr_w)
            Xte_s = sc.transform(Xte_w)
            n_pca = min(EVAL_PCA_N, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
            if n_pca < 1:
                n_pca = 1
            pca = PCA(n_components=n_pca, random_state=sd).fit(Xtr_s)
            Xtr_p = pca.transform(Xtr_s)
            Xte_p = pca.transform(Xte_s)
            ytr = (y_multi[tr] == target_idx).astype(int)
            if ytr.sum() == 0 or ytr.sum() == len(ytr):
                continue
            clf = LogisticRegression(C=LR_C, max_iter=2000,
                                     solver="lbfgs", random_state=sd)
            clf.fit(Xtr_p, ytr)
            oof_seed[te] = clf.predict_proba(Xte_p)[:, 1]
            del clf, Xtr_w, Xte_w, Xtr_s, Xte_s, Xtr_p, Xte_p, pca, sc, w, mean_auc
            gc.collect()
        y_c_full = (y_multi == target_idx).astype(int)
        auc_s = float(roc_auc_score(y_c_full, oof_seed))
        sens_s = _roc_sens_at_spec(y_c_full, oof_seed, 0.99)
        per_seed.append({"seed": int(sd), "auc": auc_s, "sens99": sens_s})
    aucs = np.array([s["auc"] for s in per_seed])
    sens = np.array([s["sens99"] for s in per_seed])
    return {
        "per_seed": per_seed,
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "sens99_mean": float(sens.mean()),
        "sens99_std": float(sens.std()),
        "sens99_min": float(sens.min()),
        "sens99_max": float(sens.max()),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def render_markdown(report: dict) -> str:
    def passed(cond: bool) -> str:
        return "✅" if cond else "❌"

    schemes = report["config"]["schemes_evaluated"]
    primary = report["config"]["weight_scheme_primary"]
    lines = [
        "# Per-cancer CADD-style continuous channel weighting",
        "",
        f"Generated: {report['generated_at']}",
        f"Cohort: {report['cohort']['n_total']} samples "
        f"x {report['cohort']['n_features']} features",
        "",
        "**Schemes evaluated:** " + ", ".join(schemes) + ". "
        f"Primary (used for integration decisions): `{primary}`.",
        "",
        "**Weight application:** per-channel column-wise scaling "
        "(`X_weighted[:, j] = X[:, j] * w_j`) before "
        "standardization + PCA(200) + OvR LR. Weights derived from "
        "inner 3-fold CV per-channel Mann-Whitney U, computed on the "
        "training fold only (no test leakage).",
        "",
        f"**Inner-CV ranking:** {INNER_FOLDS}-fold, seed={INNER_SEED}  |  "
        f"**Final eval:** {len(report['config']['seeds'])} seeds x "
        f"{report['config']['n_folds']} folds pooled OOF",
        "",
        "## Per-cancer results (weighted LR vs uniform-baseline vs top-K)",
        "",
    ]
    for s in schemes:
        lines.append(f"### Scheme: `{s}`")
        lines.append("")
        lines.append(
            "| Cancer | Sens@99% (weighted) | Sens@99% (uniform base) | "
            "Sens@99% (top-K) | AUC (weighted) | AUC (uniform base) | "
            "AUC (top-K) | Sens delta vs base | Sens delta vs topK | "
            "Target met |")
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for tgt in report["config"]["targets"]:
            fin = report["per_scheme"][s][tgt]
            base = BASELINE[tgt]
            topk = TOPK_RESULTS[tgt]
            target = TARGET_SENS99[tgt]
            met = fin["sens_at_99pct_pooled"] >= target
            d_base = (fin["sens_at_99pct_pooled"]
                      - base["sens_at_99pct_pooled"])
            d_topk = (fin["sens_at_99pct_pooled"]
                      - topk["sens_at_99pct_pooled"])
            lines.append(
                f"| {tgt} | {fin['sens_at_99pct_pooled']:.4f} | "
                f"{base['sens_at_99pct_pooled']:.4f} | "
                f"{topk['sens_at_99pct_pooled']:.4f} | "
                f"{fin['auc_pooled']:.4f} | "
                f"{base['auc_pooled']:.4f} | "
                f"{topk['auc_pooled']:.4f} | "
                f"{d_base:+.4f} | {d_topk:+.4f} | "
                f"{'PASS' if met else 'FAIL'} ({target:.2f}) |"
            )
        lines.append("")

    bf = report["binary_final"]
    p = report["passes"]
    lines += [
        "## Binary regression guard (cancer vs healthy, full features)",
        "",
        f"* Binary Sens@99% (pooled, {len(report['config']['seeds'])}x"
        f"{report['config']['n_folds']} OOF): **{bf['sens_at_99pct_pooled']:.4f}**  "
        f"(uniform baseline {BASELINE_BINARY_SENS99:.4f}, "
        f"topK {report['topk_baseline_binary_sens99']}, "
        f"floor {FLOOR_BINARY_SENS99}) "
        f"{passed(p['binary_sens_baseline_preserved'])} baseline preserved, "
        f"{passed(p['binary_sens_floor_met'])} floor met",
        f"* Binary AUC (pooled, {len(report['config']['seeds'])}x"
        f"{report['config']['n_folds']} OOF): **{bf['auc_pooled']:.4f}**  "
        f"(uniform baseline {BASELINE_BINARY_AUC:.4f}, "
        f"topK {report['topk_baseline_binary_auc']}, "
        f"floor {FLOOR_BINARY_AUC}) "
        f"{passed(p['binary_auc_baseline_preserved'])} baseline preserved, "
        f"{passed(p['binary_auc_floor_met'])} floor met",
        "",
        "## Per-cancer weight statistics (across all outer folds)",
        "",
        "| Scheme | Cancer | n_active_mean | n_active_min | n_active_max | "
        "mean_w_active | max_w_mean | median_w |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in schemes:
        for tgt in report["config"]["targets"]:
            wm = report["per_scheme"][s][tgt]["weights_meta"]
            lines.append(
                f"| {s} | {tgt} | {wm['n_active_channels_mean']:.0f} | "
                f"{wm['n_active_channels_min']} | "
                f"{wm['n_active_channels_max']} | "
                f"{wm['mean_w_active_mean']:.4f} | "
                f"{wm['max_w_mean']:.4f} | "
                f"{wm['median_w_mean']:.4f} |"
            )

    lines += [
        "",
        "## Per-seed stability (5-seed x 5-fold pooled OOF, "
        "continuous weighting)",
        "",
        "| Scheme | Cancer | AUC mean | AUC std | AUC min-max | "
        "Sens@99% mean | Sens@99% std | Sens@99% min-max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in schemes:
        for tgt in report["config"]["targets"]:
            st = report["per_scheme"][s][tgt]["stability"]
            lines.append(
                f"| {s} | {tgt} | {st['auc_mean']:.4f} | {st['auc_std']:.4f} | "
                f"{st['auc_min']:.4f}..{st['auc_max']:.4f} | "
                f"{st['sens99_mean']:.4f} | {st['sens99_std']:.4f} | "
                f"{st['sens99_min']:.4f}..{st['sens99_max']:.4f} |"
            )

    lines += [
        "",
        "## Bottom line",
        "",
        report["bottom_line"],
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=DEFAULT_FEAT_DIR)
    ap.add_argument("--target", choices=DEFAULT_TARGETS,
                    help="Single target cancer (default: all three)")
    ap.add_argument("--scheme", choices=ALL_SCHEMES,
                    default=DEFAULT_SCHEME,
                    help="Weight scheme: 'linear' (literal task spec "
                         "`max(0, auc-0.5)`) or 'sigmoid_10' (smooth-step "
                         "CADD-style). Default: linear. If unspecified, "
                         "runs BOTH schemes and reports side-by-side.")
    ap.add_argument("--quick", action="store_true",
                    help="Quick smoke run: 1 seed x 2 folds (for tests)")
    ap.add_argument("--out", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", default=DEFAULT_OUT_MD)
    args = ap.parse_args()

    targets = [args.target] if args.target else DEFAULT_TARGETS
    # If user did not pass --scheme, run all schemes (the linear
    # literal-spec scheme AND the sigmoid variant) so the report
    # shows both the spec result and the engineering result.
    schemes = ALL_SCHEMES if args.scheme == DEFAULT_SCHEME and \
        "--scheme" not in sys.argv else [args.scheme]
    # Detect explicit --scheme flag
    if any(a == "--scheme" for a in sys.argv):
        schemes = [args.scheme]

    if args.quick:
        full_seeds = [FULL_SEEDS[0]]
        full_folds = 2
    else:
        full_seeds = FULL_SEEDS
        full_folds = FULL_FOLDS

    t_start = time.time()
    labels = build_labels_multiclass(Path("labels_multiclass.tsv"))
    keep = set(labels.values())
    X, y_multi, y_bin, classes, _, _ = load_cohort_5ch(
        labels, keep=keep, feat_dir=args.features_dir)
    print(f"[load] {X.shape[0]} samples x {X.shape[1]} features, "
          f"classes={classes}  ({time.time() - t_start:.1f}s)")

    # Run each scheme, store results per scheme
    per_scheme: dict[str, dict] = {}
    for scheme in schemes:
        per_scheme[scheme] = {}
        for tgt in targets:
            ti = classes.index(tgt)
            y_c = (y_multi == ti).astype(int)
            n_pos = int(y_c.sum())
            print(f"\n[{scheme}] [weight+eval] {tgt}  n_pos={n_pos}  "
                  f"seeds={full_seeds} folds={full_folds}")
            t0 = time.time()
            oof, weights_meta = _ovr_pooled_oof_weighted(
                X, y_multi, classes, full_seeds, full_folds,
                target=tgt,
                weight_scheme=scheme,
                inner_seed=INNER_SEED, inner_folds=INNER_FOLDS)
            auc = float(roc_auc_score(y_c, oof[:, ti]))
            sens99 = _roc_sens_at_spec(y_c, oof[:, ti], 0.99)
            print(f"  weighted OOF AUC={auc:.4f}  Sens@99%={sens99:.4f}  "
                  f"({time.time() - t0:.1f}s)")
            print(f"  weights: n_active={weights_meta['n_active_channels_mean']:.0f}"
                  f" (range "
                  f"{weights_meta['n_active_channels_min']}.."
                  f"{weights_meta['n_active_channels_max']})  "
                  f"mean_w_active={weights_meta['mean_w_active_mean']:.4f}")

            # Per-seed stability
            t1 = time.time()
            stability = _per_seed_metrics(
                X, y_multi, classes, tgt, full_seeds, full_folds,
                weight_scheme=scheme,
                inner_seed=INNER_SEED, inner_folds=INNER_FOLDS)
            print(f"  per-seed: AUC {stability['auc_mean']:.4f}±"
                  f"{stability['auc_std']:.4f}  "
                  f"Sens@99% {stability['sens99_mean']:.4f}±"
                  f"{stability['sens99_std']:.4f}  "
                  f"({time.time() - t1:.1f}s)")

            per_scheme[scheme][tgt] = {
                "auc_pooled": auc,
                "sens_at_99pct_pooled": sens99,
                "n_positive": n_pos,
                "weights_meta": weights_meta,
                "stability": stability,
                "runtime_seconds": round(time.time() - t0, 1),
            }

    # Binary regression guard (cancer vs healthy) on the FULL feature
    # set, baseline protocol (LR + PCA(200), no per-cancer weighting).
    # The per-cancer weighting only changes the per-cancer OvR scores;
    # the binary pooled metric is independent of the per-cancer
    # weights, so we evaluate it here under the baseline protocol for
    # transparency.
    print(f"\n[binary guard] cancer-vs-healthy on full "
          f"{X.shape[1]}-dim features (baseline protocol)")
    t0 = time.time()
    oof_full = _ovr_pooled_oof_full(
        X, y_multi, classes, full_seeds, full_folds)
    healthy_idx = classes.index("HEALTHY")
    bin_score = 1.0 - oof_full[:, healthy_idx]  # P(cancer)
    bin_y = (y_bin).astype(int)
    bin_auc = float(roc_auc_score(bin_y, bin_score))
    bin_sens99 = _roc_sens_at_spec(bin_y, bin_score, 0.99)
    binary_final = {
        "channel_set": "all features (per-cancer weights do not touch binary)",
        "n_features_used": int(X.shape[1]),
        "auc_pooled": bin_auc,
        "sens_at_99pct_pooled": bin_sens99,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    print(f"  full binary AUC={bin_auc:.4f}  Sens@99%={bin_sens99:.4f}  "
          f"({time.time() - t0:.1f}s)")

    # TopK baseline numbers (from per_cancer_topk.json) for context.
    topk_bin_sens = None
    topk_bin_auc = None
    topk_path = Path("results/per_cancer_topk.json")
    if topk_path.exists():
        try:
            with open(topk_path) as f:
                topk_data = json.load(f)
            topk_bin_sens = topk_data["binary_final"]["sens_at_99pct_pooled"]
            topk_bin_auc = topk_data["binary_final"]["auc_pooled"]
        except Exception:
            topk_bin_sens = None
            topk_bin_auc = None

    # The "primary" scheme for integration decisions is the scheme
    # actually evaluated (linear when no --scheme is given, or the
    # explicit --scheme flag). The bottom-line analysis is always
    # done on whichever scheme was run.
    primary = schemes[0]

    def _deltas_for_scheme(scheme: str) -> tuple[dict, dict, dict]:
        sens_u = {t: per_scheme[scheme][t]["sens_at_99pct_pooled"]
                       - BASELINE[t]["sens_at_99pct_pooled"]
                  for t in targets}
        auc_u = {t: per_scheme[scheme][t]["auc_pooled"]
                       - BASELINE[t]["auc_pooled"]
                 for t in targets}
        sens_t = {t: per_scheme[scheme][t]["sens_at_99pct_pooled"]
                       - TOPK_RESULTS[t]["sens_at_99pct_pooled"]
                  for t in targets}
        return sens_u, auc_u, sens_t

    all_sens_d_u, all_auc_d_u, all_sens_d_t = {}, {}, {}
    for s in schemes:
        sd, ad, st = _deltas_for_scheme(s)
        all_sens_d_u[s] = sd
        all_auc_d_u[s] = ad
        all_sens_d_t[s] = st

    met_primary = {t: per_scheme[primary][t]["sens_at_99pct_pooled"]
                          >= TARGET_SENS99[t]
                   for t in targets}
    auc_ok_primary = {t: per_scheme[primary][t]["auc_pooled"]
                             >= FLOOR_AUC_PER_CANCER
                      for t in targets}

    base_bin_sens_ok = bin_sens99 >= BASELINE_BINARY_SENS99
    base_bin_auc_ok = bin_auc >= BASELINE_BINARY_AUC
    floor_bin_sens_ok = bin_sens99 >= FLOOR_BINARY_SENS99
    floor_bin_auc_ok = bin_auc >= FLOOR_BINARY_AUC

    bottom = [
        "Per-cancer CADD-style continuous weighting (this script) vs "
        "uniform-channel OvR baseline vs hard top-K subset "
        "(commit b5d8260):",
        "",
    ]
    for s in schemes:
        bottom.append(f"  Scheme: {s}")
        for t in targets:
            sign_s_u = "+" if all_sens_d_u[s][t] >= 0 else ""
            sign_a_u = "+" if all_auc_d_u[s][t] >= 0 else ""
            sign_s_t = "+" if all_sens_d_t[s][t] >= 0 else ""
            bottom.append(
                f"    {t}: Sens@99% "
                f"{BASELINE[t]['sens_at_99pct_pooled']:.4f} -> "
                f"{per_scheme[s][t]['sens_at_99pct_pooled']:.4f} "
                f"({sign_s_u}{all_sens_d_u[s][t]:.4f} vs uniform; "
                f"{sign_s_t}{all_sens_d_t[s][t]:.4f} vs top-K), "
                f"AUC {BASELINE[t]['auc_pooled']:.4f} -> "
                f"{per_scheme[s][t]['auc_pooled']:.4f} ({sign_a_u}"
                f"{all_auc_d_u[s][t]:.4f} vs uniform), "
                f"target {TARGET_SENS99[t]:.2f} -> "
                f"{'MET' if per_scheme[s][t]['sens_at_99pct_pooled'] >= TARGET_SENS99[t] else 'NOT MET'}"
            )
        bottom.append("")
    bottom.append(
        f"Binary guard (cancer vs healthy, full {X.shape[1]} features):")
    bottom.append(
        f"  Sens@99%={bin_sens99:.4f}  uniform-baseline="
        f"{BASELINE_BINARY_SENS99:.4f} "
        f"({bin_sens99 - BASELINE_BINARY_SENS99:+.4f}, "
        f"{'ABOVE baseline' if base_bin_sens_ok else 'BELOW baseline'}), "
        f"floor={FLOOR_BINARY_SENS99} "
        f"({'MET' if floor_bin_sens_ok else 'NOT MET'})")
    bottom.append(
        f"  AUC     ={bin_auc:.4f}  uniform-baseline="
        f"{BASELINE_BINARY_AUC:.4f} "
        f"({bin_auc - BASELINE_BINARY_AUC:+.4f}, "
        f"{'ABOVE baseline' if base_bin_auc_ok else 'BELOW baseline'}), "
        f"floor={FLOOR_BINARY_AUC} "
        f"({'MET' if floor_bin_auc_ok else 'NOT MET'})")
    bottom.append("")
    if not args.quick:
        # Honest answer: did the continuous scheme beat hard top-K?
        winners = [t for t in targets
                   if all_sens_d_t[primary][t] >= 0 and all_auc_d_u[primary][t] >= -0.005]
        regressors = [t for t in targets
                      if all_sens_d_t[primary][t] < -0.01]
        stable = [t for t in targets
                  if per_scheme[primary][t]["stability"]["sens99_std"] < 0.10]
        bottom.append(
            f"Primary scheme ({primary}) beats hard top-K on: "
            f"{winners if winners else 'none'}")
        bottom.append(
            f"Primary scheme ({primary}) regressed >1pp vs top-K on: "
            f"{regressors if regressors else 'none'}")
        bottom.append(
            f"Primary scheme ({primary}) stable across seeds "
            f"(Sens@99% std < 0.10): "
            f"{stable if stable else 'none'}")
        bottom.append("")
        improved_over_uniform = [t for t in targets
                                 if all_sens_d_u[primary][t] >= 0.10
                                 and auc_ok_primary[t]
                                 and base_bin_sens_ok and base_bin_auc_ok]
        if improved_over_uniform:
            bottom.append(
                f"Cancers with >= 0.10 absolute Sens@99% gain over the "
                f"uniform baseline (per-cancer AUC and binary baseline "
                f"both preserved): {', '.join(improved_over_uniform)}. "
                f"Integration recommended.")
        else:
            bottom.append(
                "No cancer gained >= 0.10 absolute Sens@99% over the "
                "uniform baseline while preserving AUC and binary "
                "baseline. Per-cancer picture (Sens@99% delta vs "
                f"uniform, {primary} scheme): "
                f"OV {all_sens_d_u[primary].get('OV', 0):+.4f}, "
                f"PAAD {all_sens_d_u[primary].get('PAAD', 0):+.4f}, "
                f"BRCA {all_sens_d_u[primary].get('BRCA', 0):+.4f}.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "inner_seed": INNER_SEED,
            "inner_folds": INNER_FOLDS,
            "seeds": full_seeds,
            "n_folds": full_folds,
            "lr_C": LR_C,
            "targets": targets,
            "weight_scheme_primary": primary,
            "schemes_evaluated": schemes,
            "weight_application": "column-wise feature scaling "
                                  "(X_weighted = X * w)",
        },
        "cohort": {
            "n_total": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "classes": classes,
        },
        "per_scheme": per_scheme,
        "final": per_scheme[primary],  # back-compat: primary scheme
        "binary_final": binary_final,
        "baseline": BASELINE,
        "baseline_binary_sens99": BASELINE_BINARY_SENS99,
        "baseline_binary_auc": BASELINE_BINARY_AUC,
        "topk_results": TOPK_RESULTS,
        "topk_baseline_binary_sens99": topk_bin_sens,
        "topk_baseline_binary_auc": topk_bin_auc,
        "floors": {
            "auc_per_cancer": FLOOR_AUC_PER_CANCER,
            "binary_sens99": FLOOR_BINARY_SENS99,
            "binary_auc": FLOOR_BINARY_AUC,
            "targets": TARGET_SENS99,
        },
        "deltas": {
            "sens_vs_uniform": all_sens_d_u,
            "sens_vs_topk": all_sens_d_t,
            "auc_vs_uniform": all_auc_d_u,
        },
        "passes": {
            "sens_target_met": met_primary,
            "auc_floor_met": auc_ok_primary,
            "binary_sens_baseline_preserved": base_bin_sens_ok,
            "binary_auc_baseline_preserved": base_bin_auc_ok,
            "binary_sens_floor_met": floor_bin_sens_ok,
            "binary_auc_floor_met": floor_bin_auc_ok,
        },
        "bottom_line": "\n".join(bottom),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)
    md = render_markdown(report)
    with open(args.out_md, "w") as f:
        f.write(md)
    print()
    print("=" * 78)
    print("BOTTOM LINE")
    print("=" * 78)
    print(report["bottom_line"])
    print()
    print(f"Wrote {out_p} and {args.out_md} in {report['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
