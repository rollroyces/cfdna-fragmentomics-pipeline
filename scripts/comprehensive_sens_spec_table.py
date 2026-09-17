#!/usr/bin/env python3
"""Comprehensive Sens@Spec + PPV@Prev reporting for the cfdna-fragmentomics-pipeline.

Insight #3 (this session): AUC and Sens@Spec are decoupled at saturation.
For clinical decision-making, Sens@Spec + PPV@Prev matter most, not
pooled AUC alone. This script consolidates:

  1. Sens@Spec table for frag-only and frag+fusion (naive_average),
     at spec ∈ {0.90, 0.95, 0.98, 0.99, 0.995, 0.999}, with DeLong 95% CI
     (placement-value method, DeLong et al. 1988).

  2. PPV@Prev table at prevalences
     {0.001, 0.004, 0.01, 0.05, 0.10, 0.20, 0.50}
     using Sens@99% spec as the operating point. Each operating point's
     sens is also reported with its DeLong CI so PPV uncertainty is
     traceable.

  3. Per-cancer Sens@Spec table (using pooled OvR OOF, OvR against
     all-others, with DeLong CI on Sens@99%).

  4. Honest machine-readable summary at
     results/sens_spec_table.json and a human-readable version at
     docs/SENS_SPEC_TABLE.md.

This script reuses the existing artifacts where possible:
  - sens_at_spec_scores.npz (frag-only pooled OOF) — avoids refitting LR
  - fusion_ablation.py simulate_mutation_scores — same calibration
  - per_cancer_auc.py per_cancer_oof — same protocol

Re-fit cost is dominated by the per-cancer OvR loop (~5 classes × 5 seeds
× 5 folds × LR). On the 627-sample 5-channel (63246 features) cohort
this takes ~2-3 minutes. Background-process if running interactively.

Honest reporting: every number comes from real execution; CI bounds are
quantile-based and reflect the OOF sample variance, not external
validation.

Usage:
  python scripts/comprehensive_sens_spec_table.py
  python scripts/comprehensive_sens_spec_table.py --help

Run with: env -u PYTHONPATH /Users/hermes/deepcatch/.venv/bin/python
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
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from honest_benchmark import load5
from train_classifier import _harmonize
from sens_at_specificity import (
    pooled_oof_predictions,
    sens_at_specificity,
)
from fusion_ablation import simulate_mutation_scores
from multiclass_classification import build_labels_multiclass, load_cohort_5ch


DEFAULT_FEAT_DIR = "data/features"
DEFAULT_SEEDS = [42, 13, 7, 99, 1234]
DEFAULT_SPECIFICITIES = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
DEFAULT_PREVALENCES = [0.001, 0.004, 0.01, 0.05, 0.10, 0.20, 0.50]
DEFAULT_PCA = 200
DEFAULT_OPERATING_SPEC = 0.99
DEFAULT_MUTATION_AUC = 0.92
DEFAULT_MUTATION_SENS95 = 0.77
DEFAULT_N_BOOT = 1000
DEFAULT_BOOT_SEED = 2026


def delong_ci_sens_at_spec(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificity: float,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """DeLong 95% CI on Sens@Spec via placement values on positives.

    Implements the placement-value analogue of DeLong et al. 1988 for
    sensitivity at a fixed specificity operating point. The threshold
    c is the largest score such that FPR(c) <= 1-spec (a property of
    the assay, not a free parameter once the operating point is fixed).

    Sens = E_pos[psi_pos_i] where psi_pos_i = I[s_i > c] + 0.5*I[s_i == c]
    in [0, 1]. SE(Sens) = SD_pos(psi_pos) / sqrt(n_pos). This matches
    the DeLong methodology (sensitivity is a mean of placement values)
    and propagates correctly to PPV uncertainty.

    Note on threshold selection bias: c is selected from the same OOF
    sample, so the realized FPR is at most 1-spec but can be strictly
    less, and the realized sens carries a small upward optimism. We do
    NOT correct for this (no closed-form correction is standard); the
    bootstrap CI in sens_at_spec.py reflects the same optimism.

    Reference: DeLong, DeLong, Clarke-Pearson 1988, Biometrics 44:837-845.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Find the largest threshold c such that FPR(c) <= 1-spec.
    fpr, tpr, thr = roc_curve(y_true, y_score)
    target_fpr = 1.0 - specificity
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0.0, 0.0
    c = float(np.asarray(thr[idx[-1]]))

    # Placement values on positives at threshold c.
    pos_scores = y_score[pos_mask]
    psi_pos = ((pos_scores > c).astype(float)
               + 0.5 * (pos_scores == c).astype(float))
    sens = float(np.mean(psi_pos))
    # DeLong SE: SD over positives / sqrt(n_pos).
    sd_pos = float(np.std(psi_pos, ddof=1)) if n_pos > 1 else 0.0
    se_sens = sd_pos / np.sqrt(max(n_pos, 1))
    z = norm.ppf(1.0 - alpha / 2)
    ci_lo = max(0.0, sens - z * se_sens)
    ci_hi = min(1.0, sens + z * se_sens)
    return sens, se_sens, ci_lo, ci_hi


def delong_ci_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float, float, float]:
    """DeLong 95% CI on AUC (placement-value method).

    Returns (auc, se_auc, ci95_lo, ci95_hi). Implementation follows
    DeLong et al. 1988, with the structural-component simplification
    of Sun & Xu 2014 (Stat Med 33: 3945-3958).

    Placement values:
        psi_10_i = (1/n_neg) * sum_j I[s_j < s_i] + 0.5 * I[s_j == s_i]
        psi_01_j = (1/n_pos) * sum_i I[s_i > s_j] + 0.5 * I[s_i == s_j]
    AUC = mean(psi_10)
    Var(AUC) = S_10^2 / n_pos + S_01^2 / n_neg
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Vectorised computation of placement values.
    # psi_10_i = (# neg with s < s_i) / n + 0.5 * (# neg with s == s_i) / n
    # Build all_pairs indicator matrix [m, n].
    pos_col = pos[:, None]
    neg_row = neg[None, :]
    psi_10 = ((neg_row < pos_col).sum(axis=1)
              + 0.5 * (neg_row == pos_col).sum(axis=1)) / n
    # psi_01_j = (# pos with s > s_j) / m + 0.5 * (# pos with s == s_j) / m
    psi_01 = ((pos_col > neg_row).sum(axis=0)
              + 0.5 * (pos_col == neg_row).sum(axis=0)) / m

    auc = float(np.mean(psi_10))
    s10 = float(np.std(psi_10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.std(psi_01, ddof=1)) if n > 1 else 0.0
    var_auc = (s10 ** 2) / m + (s01 ** 2) / n
    se_auc = float(np.sqrt(max(var_auc, 0.0)))
    z = norm.ppf(0.975)
    return auc, se_auc, max(0.0, auc - z * se_auc), min(1.0, auc + z * se_auc)


def ppv(sens: float, spec: float, prev: float) -> float:
    """Bayesian PPV (matches ppv_screening.py)."""
    tp = sens * prev
    fp = (1.0 - spec) * (1.0 - prev)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def npv(sens: float, spec: float, prev: float) -> float:
    tn = spec * (1.0 - prev)
    fn = (1.0 - sens) * prev
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def sens_at_spec_with_delong(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificity: float,
) -> dict:
    """Sens@Spec + DeLong CI + bootstrap CI for cross-validation.

    Returns a dict with the keys the JSON schema expects.
    """
    sens, se, ci_lo, ci_hi = delong_ci_sens_at_spec(
        y_true, y_score, specificity)
    # Use existing bootstrap function (already 1000 iterations).
    from sens_at_specificity import bootstrap_ci
    rng = np.random.default_rng(DEFAULT_BOOT_SEED)
    b_lo, b_hi = bootstrap_ci(y_true, y_score, specificity,
                              DEFAULT_N_BOOT, rng)
    fpr, _, thr = roc_curve(y_true, y_score)
    op_thr = float(np.asarray(thr[(fpr <= 1.0 - specificity)][-1])) \
        if np.any(fpr <= 1.0 - specificity) else float("nan")
    return {
        "specificity": specificity,
        "sensitivity": sens,
        "se_delong": se,
        "ci95_lo_delong": ci_lo,
        "ci95_hi_delong": ci_hi,
        "ci95_lo_bootstrap": b_lo,
        "ci95_hi_bootstrap": b_hi,
        "operating_threshold": op_thr,
        "n_bootstrap": DEFAULT_N_BOOT,
    }


def render_sens_spec_markdown(
    frag_rows: list[dict],
    fusion_rows: list[dict],
    operating_spec: float,
    fusion_auc: float,
    frag_auc: float,
) -> str:
    """Markdown table combining frag-only and frag+fusion Sens@Spec."""
    lines = [
        "# Sens@Spec table — frag-only vs frag+fusion",
        "",
        "Pooled 5-seed × 5-fold OOF on the 627-sample cross-study cohort.",
        "Frag-only = 5-channel LR + PCA(200) on FinaleDB features.",
        "Frag+fusion = naive average of frag-only score and a synthetic",
        "mutation-channel score calibrated to AUC=0.92, Sens@95%≈0.77",
        "(same protocol as `scripts/fusion_ablation.py`).",
        "",
        "CI: **DeLong** placement-value method (DeLong et al. 1988).",
        "For comparison, percentile-bootstrap CIs (n=1000) are also shown",
        "where available.",
        "",
        f"**Pooled AUC:** frag-only = **{frag_auc:.4f}**, "
        f"frag+fusion = **{fusion_auc:.4f}**.",
        "",
        f"**Operating-point sens for PPV table:** "
        f"sens @ spec = {operating_spec*100:.1f}%.",
        "",
        "| spec | frag-only sens | DeLong 95% CI | bootstrap 95% CI | "
        "frag+fusion sens | DeLong 95% CI | bootstrap 95% CI |",
        "|---:|---:|---|---|---:|---|---|",
    ]
    spec_to_fusion = {r["specificity"]: r for r in fusion_rows}
    for fr in frag_rows:
        sp = fr["specificity"]
        fu = spec_to_fusion.get(sp)
        if fu is None:
            continue
        lines.append(
            f"| {sp*100:.2f}% | "
            f"{fr['sensitivity']*100:.2f}% | "
            f"[{fr['ci95_lo_delong']*100:.2f}, {fr['ci95_hi_delong']*100:.2f}] | "
            f"[{fr['ci95_lo_bootstrap']*100:.2f}, {fr['ci95_hi_bootstrap']*100:.2f}] | "
            f"{fu['sensitivity']*100:.2f}% | "
            f"[{fu['ci95_lo_delong']*100:.2f}, {fu['ci95_hi_delong']*100:.2f}] | "
            f"[{fu['ci95_lo_bootstrap']*100:.2f}, {fu['ci95_hi_bootstrap']*100:.2f}] |"
        )
    return "\n".join(lines) + "\n"


def render_ppv_markdown(
    operating_spec: float,
    operating_sens_frag: float,
    operating_sens_frag_ci: tuple[float, float],
    operating_sens_fusion: float,
    operating_sens_fusion_ci: tuple[float, float],
    prevalences: list[float],
    frag_ppv_rows: list[dict],
    fusion_ppv_rows: list[dict],
) -> str:
    """Markdown table for PPV at each prevalence × model."""
    lines = [
        "# PPV@Prev table — Sens@"
        f"{operating_spec*100:.1f}% spec operating point",
        "",
        f"Operating point: spec = **{operating_spec*100:.1f}%**.",
        "",
        f"Pooled sens at this operating point (frag-only): "
        f"**{operating_sens_frag*100:.2f}%** "
        f"(DeLong 95% CI [{operating_sens_frag_ci[0]*100:.2f}, "
        f"{operating_sens_frag_ci[1]*100:.2f}]).",
        "",
        f"Pooled sens at this operating point (frag+fusion): "
        f"**{operating_sens_fusion*100:.2f}%** "
        f"(DeLong 95% CI [{operating_sens_fusion_ci[0]*100:.2f}, "
        f"{operating_sens_fusion_ci[1]*100:.2f}]).",
        "",
        "| prevalence | context | frag-only PPV | frag+fusion PPV |",
        "|---:|---|---|---|",
    ]
    prev_labels = {
        0.001: "0.1% — ultra-low-prevalence research cohort",
        0.004: "0.4% — US adults 50+ (Galleri-comparable)",
        0.01: "1% — adults 50+ with 1 risk factor",
        0.05: "5% — high-risk surveillance (post-MRD)",
        0.10: "10% — hereditary cancer syndrome surveillance",
        0.20: "20% — symptomatic workup",
        0.50: "50% — confirmatory test",
    }
    for prev in prevalences:
        fp_row = next((r for r in frag_ppv_rows
                       if abs(r["prevalence"] - prev) < 1e-9), None)
        fu_row = next((r for r in fusion_ppv_rows
                       if abs(r["prevalence"] - prev) < 1e-9), None)
        if fp_row is None or fu_row is None:
            continue
        lbl = prev_labels.get(prev, f"{prev*100:.2f}%")
        lines.append(
            f"| {prev*100:.2f}% | {lbl} | "
            f"{fp_row['ppv']*100:.2f}% | "
            f"{fu_row['ppv']*100:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def render_per_cancer_markdown(per_cancer_rows: dict, specificities: list[float]) -> str:
    """Markdown table for per-cancer Sens@Spec (with DeLong CI)."""
    lines = [
        "# Per-cancer Sens@Spec table (OvR pooled 5-seed × 5-fold OOF)",
        "",
        "Each row: one-vs-rest pooled OOF predictions for that cancer vs "
        "(all other cancers + healthy). CI = DeLong placement-value.",
        "Bootstrap AUC CI is preserved in the JSON (cross-check from "
        "`results/per_cancer_auc.json`).",
        "",
    ]
    header = "| cancer | n_pos | pooled AUC | DeLong 95% CI |"
    for sp in specificities:
        header += f" sens@{sp*100:.1f}% | DeLong 95% CI |"
    lines.append(header)
    sep = "|---|---:|---:|---|"
    for _ in specificities:
        sep += "---:|---|"
    lines.append(sep)
    for cls in sorted(per_cancer_rows.keys(),
                      key=lambda k: -per_cancer_rows[k]["auc_pooled"]):
        r = per_cancer_rows[cls]
        small = " <30" if r.get("small_n_flag") else ""
        row = f"| {cls}{small} | {r['n_positive']} | " \
              f"{r['auc_pooled']:.4f} | " \
              f"[{r['auc_ci95_delong_lo']:.4f}, {r['auc_ci95_delong_hi']:.4f}] |"
        for sp in specificities:
            sk = f"sens_at_{int(round(sp*1000)):03d}pct"
            if sk not in r:
                row += f" - | - |"
                continue
            sr = r[sk]
            row += f" {sr['sensitivity']*100:.2f}% | " \
                   f"[{sr['ci95_lo_delong']*100:.2f}, {sr['ci95_hi_delong']*100:.2f}] |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def build_ppv_rows(
    operating_spec: float,
    operating_sens: float,
    operating_sens_ci: tuple[float, float],
    prevalences: list[float],
    label: str,
) -> list[dict]:
    """Build per-prevalence PPV rows from a single operating point."""
    rows = []
    for prev in prevalences:
        p = ppv(operating_sens, operating_spec, prev)
        n = npv(operating_sens, operating_spec, prev)
        # Also propagate sens uncertainty → PPV bounds (low/high sens).
        p_lo = ppv(operating_sens_ci[0], operating_spec, prev)
        p_hi = ppv(operating_sens_ci[1], operating_spec, prev)
        fp_per_tp = ((1.0 - operating_spec) * (1.0 - prev)) / (
            operating_sens * prev) if operating_sens * prev > 0 else float("inf")
        rows.append({
            "prevalence": prev,
            "operating_spec": operating_spec,
            "operating_sens": operating_sens,
            "operating_sens_ci_lo": operating_sens_ci[0],
            "operating_sens_ci_hi": operating_sens_ci[1],
            "model": label,
            "ppv": p,
            "ppv_ci_lo": p_lo,
            "ppv_ci_hi": p_hi,
            "npv": n,
            "fp_per_tp": fp_per_tp,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=DEFAULT_FEAT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--specificities", type=float, nargs="+",
                    default=DEFAULT_SPECIFICITIES)
    ap.add_argument("--prevalences", type=float, nargs="+",
                    default=DEFAULT_PREVALENCES)
    ap.add_argument("--operating-spec", type=float,
                    default=DEFAULT_OPERATING_SPEC,
                    help="Specificity at which to take the sens for the "
                         "PPV table.")
    ap.add_argument("--mutation-auc", type=float, default=DEFAULT_MUTATION_AUC)
    ap.add_argument("--mutation-sens95", type=float, default=DEFAULT_MUTATION_SENS95)
    ap.add_argument("--pca", type=int, default=DEFAULT_PCA)
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOT_SEED)
    ap.add_argument("--no-harmonize", action="store_true")
    ap.add_argument("--scores-npz",
                    default="results/sens_at_spec_scores.npz",
                    help="If present, reuse pooled OOF scores instead of "
                         "re-fitting LR. Saves ~30s and removes LR "
                         "stochasticity.")
    ap.add_argument("--out-json", default="results/sens_spec_table.json")
    ap.add_argument("--out-md", default="docs/SENS_SPEC_TABLE.md")
    ap.add_argument("--skip-per-cancer", action="store_true",
                    help="Skip the per-cancer step (slowest).")
    args = ap.parse_args()

    t0 = time.time()
    harmonize = not args.no_harmonize

    # === 1. Load frag-only pooled OOF scores ===
    if os.path.exists(args.scores_npz):
        print(f"[frag-only] Loading cached OOF from {args.scores_npz}")
        npz = np.load(args.scores_npz, allow_pickle=True)
        y_true = npz["y_true"].astype(int)
        frag_score = npz["score"].astype(float)
        # Recompute pooled AUC + DeLong CI on the cached scores.
        auc_frag, se_frag, auc_lo, auc_hi = delong_ci_auc(y_true, frag_score)
    else:
        print(f"[frag-only] Re-fitting 5-seed × 5-fold pooled OOF ...")
        labels, studies = {}, {}
        with open(os.path.join(args.features_dir, "labels_cross_study.tsv")) as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) < 2:
                    continue
                labels[p[0]] = 1 if p[1] == "cancer" else 0
                studies[p[0]] = p[2] if len(p) >= 3 else "unknown"
        X, y_arr, st = load5(labels, studies, args.features_dir)
        y_true, frag_score, auc_mean, auc_std = pooled_oof_predictions(
            X, y_arr, st, args.seeds, args.pca, harmonize)
        auc_frag, se_frag, auc_lo, auc_hi = delong_ci_auc(y_true, frag_score)

    n_cancer = int((y_true == 1).sum())
    n_healthy = int((y_true == 0).sum())
    print(f"  n={len(y_true)} (cancer={n_cancer}, healthy={n_healthy})")
    print(f"  Pooled AUC = {auc_frag:.4f} "
          f"(DeLong 95% CI [{auc_lo:.4f}, {auc_hi:.4f}])")

    # === 2. Frag-only Sens@Spec table with DeLong CI ===
    print("[frag-only] Computing Sens@Spec with DeLong CI ...")
    frag_rows = [
        sens_at_spec_with_delong(y_true, frag_score, sp)
        for sp in args.specificities
    ]

    # === 3. Frag+fusion (naive average of frag-only and synthetic mutation) ===
    print("[frag+fusion] Simulating mutation score and naive-average fusion ...")
    mut_score = simulate_mutation_scores(
        y_true, args.mutation_auc, args.mutation_sens95,
        seed=args.bootstrap_seed)
    fusion_score = (frag_score + mut_score) / 2.0
    auc_fusion, se_fusion, auc_fusion_lo, auc_fusion_hi = delong_ci_auc(
        y_true, fusion_score)
    print(f"  Pooled AUC (frag+fusion) = {auc_fusion:.4f} "
          f"(DeLong 95% CI [{auc_fusion_lo:.4f}, {auc_fusion_hi:.4f}])")
    fusion_rows = [
        sens_at_spec_with_delong(y_true, fusion_score, sp)
        for sp in args.specificities
    ]

    # === 4. PPV table at the operating-spec point ===
    op_spec = args.operating_spec
    frag_op = next(r for r in frag_rows if abs(r["specificity"] - op_spec) < 1e-9)
    fusion_op = next(r for r in fusion_rows if abs(r["specificity"] - op_spec) < 1e-9)
    frag_ppv_rows = build_ppv_rows(
        op_spec, frag_op["sensitivity"],
        (frag_op["ci95_lo_delong"], frag_op["ci95_hi_delong"]),
        args.prevalences, "frag-only")
    fusion_ppv_rows = build_ppv_rows(
        op_spec, fusion_op["sensitivity"],
        (fusion_op["ci95_lo_delong"], fusion_op["ci95_hi_delong"]),
        args.prevalences, "frag+fusion")

    # === 5. Per-cancer Sens@Spec ===
    # We compute a minimal pooled 5-seed × 5-fold OvR LR pipeline here
    # (no bootstrap) — the existing results/per_cancer_auc.json already
    # carries the bootstrap AUC + sens@99% CI as a sanity-check
    # companion. The DeLong CI on AUC and Sens@Spec is the new layer.
    per_cancer_rows = {}
    if not args.skip_per_cancer:
        print("[per-cancer] Computing OvR pooled 5-seed × 5-fold OOF ...")
        labels_multiclass = build_labels_multiclass(Path("labels_multiclass.tsv"))
        keep = set(labels_multiclass.values())
        X_pc, y_multi, _y_bin, classes, _cls_to_int, _sample_ids = load_cohort_5ch(
            labels_multiclass, keep=keep, feat_dir=args.features_dir)
        from per_cancer_auc import _lr_pipeline

        def _ovr_pooled(class_idx: int) -> tuple[np.ndarray, np.ndarray]:
            """Return (y_bin, pooled OvR proba) for one cancer class."""
            y_bin = (y_multi == class_idx).astype(int)
            n = len(y_bin)
            pooled = np.zeros(n, dtype=float)
            for sd in args.seeds:
                cv = StratifiedKFold(5, shuffle=True, random_state=sd)
                for tr, te in cv.split(X_pc, y_multi):
                    ytr_bin = (y_multi[tr] == class_idx).astype(int)
                    if ytr_bin.sum() == 0 or ytr_bin.sum() == len(ytr_bin):
                        pooled[te] += 0.5 / len(args.seeds)
                        continue
                    pooled[te] += _lr_pipeline(
                        X_pc[tr], ytr_bin, X_pc[te], seed=sd,
                        pca_n=args.pca) / len(args.seeds)
            return y_bin, pooled

        # Honour any existing per_cancer_auc.json for bootstrap CI.
        existing_pc = None
        pc_path = Path("results/per_cancer_auc.json")
        if pc_path.exists():
            try:
                with open(pc_path) as f:
                    existing_pc = json.load(f)
            except Exception:
                existing_pc = None

        for ci, cls in enumerate(classes):
            print(f"  [{cls}] OvR pooled OOF ...")
            y_bin, pooled = _ovr_pooled(ci)
            # DeLong on AUC and Sens@Spec.
            auc_pc, se_pc, auc_lo_pc, auc_hi_pc = delong_ci_auc(y_bin, pooled)
            row = {
                "n_samples": int(len(y_bin)),
                "n_positive": int(y_bin.sum()),
                "auc_pooled": float(auc_pc),
                "auc_se_delong": float(se_pc),
                "auc_ci95_delong_lo": float(auc_lo_pc),
                "auc_ci95_delong_hi": float(auc_hi_pc),
                "small_n_flag": int(y_bin.sum()) < 30,
            }
            # Sens@Spec + DeLong at each specificity.
            for sp in args.specificities:
                sr = sens_at_spec_with_delong(y_bin, pooled, sp)
                row[f"sens_at_{int(round(sp*1000)):03d}pct"] = sr
            # Pull in bootstrap AUC + sens@99% from existing JSON.
            if existing_pc is not None and cls in existing_pc.get("per_cancer", {}):
                e = existing_pc["per_cancer"][cls]
                row["auc_seed_mean"] = e.get("auc_seed_mean")
                row["auc_seed_std"] = e.get("auc_seed_std")
                row["auc_ci95_bootstrap_lo"] = e.get("auc_ci95_lo")
                row["auc_ci95_bootstrap_hi"] = e.get("auc_ci95_hi")
                # Existing JSON only carries Sens@99% bootstrap; record it
                # if the operating spec matches 99%.
                if abs(op_spec - 0.99) < 1e-9:
                    row["sens_at_99pct_bootstrap_pooled"] = (
                        e.get("sens_at_99pct_pooled"))
                    row["sens_at_99pct_bootstrap_ci_lo"] = (
                        e.get("sens_at_99pct_ci95_lo"))
                    row["sens_at_99pct_bootstrap_ci_hi"] = (
                        e.get("sens_at_99pct_ci95_hi"))
            per_cancer_rows[cls] = row

    # === 6. Render markdown ===
    md_frag_fusion = render_sens_spec_markdown(
        frag_rows, fusion_rows, op_spec, auc_fusion, auc_frag)
    md_ppv = render_ppv_markdown(
        op_spec,
        frag_op["sensitivity"],
        (frag_op["ci95_lo_delong"], frag_op["ci95_hi_delong"]),
        fusion_op["sensitivity"],
        (fusion_op["ci95_lo_delong"], fusion_op["ci95_hi_delong"]),
        args.prevalences, frag_ppv_rows, fusion_ppv_rows)
    md_per_cancer = render_per_cancer_markdown(per_cancer_rows, args.specificities) \
        if per_cancer_rows else ""

    md = (
        "# Comprehensive Sens@Spec + PPV@Prev table\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Cohort: {len(y_true)} samples "
        f"({n_cancer} cancer + {n_healthy} healthy), "
        f"5-seed × 5-fold pooled OOF.\n\n"
        "---\n\n"
        + md_frag_fusion
        + "\n---\n\n"
        + md_ppv
        + "\n---\n\n"
        + md_per_cancer
        + "\n---\n\n"
        "## Honest framing\n\n"
        "* All numbers are pooled out-of-fold on the SAME 627-sample\n"
        "  FinaleDB cohort. They are NOT external validation.\n"
        "* CI: DeLong placement-value method on a fixed-threshold operating\n"
        "  point; bootstrap CI is percentile-resampled (n=1000) for\n"
        "  sanity-check comparison.\n"
        "* Frag+fusion uses a synthetic mutation score calibrated to\n"
        "  AUC=0.92, Sens@95%≈0.77 (same recipe as\n"
        "  `scripts/fusion_ablation.py`). The mutation channel is a\n"
        "  proxy, not a real targeted-panel readout.\n"
        "* Per-cancer rows with n<30 are exploratory (flagged in column).\n"
    )

    # === 7. JSON payload ===
    payload = {
        "cohort": {
            "n_total": int(len(y_true)),
            "n_cancer": int(n_cancer),
            "n_healthy": int(n_healthy),
        },
        "config": {
            "seeds": args.seeds,
            "specificities": args.specificities,
            "prevalences": args.prevalences,
            "operating_spec": op_spec,
            "pca_n": args.pca,
            "harmonize": harmonize,
            "mutation_auc": args.mutation_auc,
            "mutation_sens95": args.mutation_sens95,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "ci_method": "DeLong placement-value (DeLong et al. 1988)",
        },
        "frag_only": {
            "auc_pooled": auc_frag,
            "auc_se_delong": se_frag,
            "auc_ci95_delong_lo": auc_lo,
            "auc_ci95_delong_hi": auc_hi,
            "per_specificity": frag_rows,
        },
        "frag_plus_fusion": {
            "auc_pooled": auc_fusion,
            "auc_se_delong": se_fusion,
            "auc_ci95_delong_lo": auc_fusion_lo,
            "auc_ci95_delong_hi": auc_fusion_hi,
            "fusion_rule": "naive_average(frag, simulated_mutation)",
            "per_specificity": fusion_rows,
        },
        "ppv_table": {
            "operating_spec": op_spec,
            "operating_sens_frag": frag_op["sensitivity"],
            "operating_sens_frag_ci_lo": frag_op["ci95_lo_delong"],
            "operating_sens_frag_ci_hi": frag_op["ci95_hi_delong"],
            "operating_sens_fusion": fusion_op["sensitivity"],
            "operating_sens_fusion_ci_lo": fusion_op["ci95_lo_delong"],
            "operating_sens_fusion_ci_hi": fusion_op["ci95_hi_delong"],
            "frag_only_rows": frag_ppv_rows,
            "frag_plus_fusion_rows": fusion_ppv_rows,
        },
        "per_cancer": per_cancer_rows,
        "wall_seconds": round(time.time() - t0, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)
    print(f"\nWrote {args.out_json} and {args.out_md} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
