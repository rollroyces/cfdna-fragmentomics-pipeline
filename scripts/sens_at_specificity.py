"""Sensitivity at high specificity thresholds for the cross-study 627 cohort.

Extends the decision-curve reporting in scripts/decision_curve.py (deepcatch)
and scripts/ppv_screening.py by computing Sens@Spec for
spec ∈ {0.95, 0.98, 0.99, 0.995, 0.999}.

Why this matters: published MCED tests (Galleri, Shield, CancerSEEK) all
report Sens at a specific operating point (typically 99% specificity for
population screening). To make the comparison honest, we need the same
operating points on the same pooled 5-seed 5-fold OOF predictions the
honest benchmark already uses.

CLI:
  python scripts/sens_at_specificity.py
    [--features-dir DIR]    default: data/features
    [--seeds N N N ...]      default: 42 13 7 99 1234
    [--n-bootstrap N]        default: 1000
    [--out PATH]             default: results/sens_at_spec.json
    [--pca N]                default: 200 (matches honest_benchmark Section C)
    [--no-harmonize]         disable per-study harmonization

Output JSON schema:
{
  "cohort": {"n_total", "n_cancer", "n_healthy", "studies"},
  "config": {"seeds", "n_bootstrap", "pca_n", "harmonize", "pooled_oof_n"},
  "per_specificity": [
    {"specificity": 0.95, "sensitivity": 0.91,
     "ci95_lo": 0.88, "ci95_hi": 0.94, "n_below_fpr": 1234, ...},
    ...
  ],
  "auc_5seed_mean": 0.85, "auc_5seed_std": 0.02,
  "generated_at": "2026-..."
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Reuse the existing honest_benchmark loader + harmonize helper.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from honest_benchmark import load5
from train_classifier import _harmonize

DEFAULT_FEAT_DIR = "data/features"
DEFAULT_SEEDS = [42, 13, 7, 99, 1234]
DEFAULT_SPECIFICITIES = [0.95, 0.98, 0.99, 0.995, 0.999]
DEFAULT_PCA = 200


def _load_labels_cross_study(feat_dir: str):
    """Load labels_cross_study.tsv -> (labels_dict, studies_dict)."""
    labels, studies = {}, {}
    with open(os.path.join(feat_dir, "labels_cross_study.tsv")) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 2:
                continue
            labels[p[0]] = 1 if p[1] == "cancer" else 0
            studies[p[0]] = p[2] if len(p) >= 3 else "unknown"
    return labels, studies


def pooled_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    st: np.ndarray,
    seeds: list[int],
    pca_n: int,
    harmonize: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """5-seed 5-fold OOF: each sample gets exactly one prediction per seed.

    Returns (y_true_pooled, score_pooled, auc_mean, auc_std).
    The pooled score is the *average across seeds* of per-seed OOF scores
    (so each sample is still predicted only on data it never saw during
    training). This matches the honest_benchmark convention.
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
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr])
                Xte = sc.transform(X[te])
            max_pca = min(Xtr.shape[0], Xtr.shape[1])
            pca = PCA(n_components=min(pca_n, max_pca)).fit(Xtr)
            # LR with default lbfgs solver. NOTE: on this high-dim PCA
            # input the solver often hits max_iter without full
            # convergence, so per-fold scores carry ~1-3pp run-to-run
            # noise. The pooled 5-seed OOF averages this out somewhat,
            # and the bootstrap CI captures the remaining variance.
            # We use the SAME max_iter as honest_benchmark.py for
            # apples-to-apples comparison against Section C.
            m = LogisticRegression(max_iter=2000).fit(
                pca.transform(Xtr), y[tr])
            oof[te] = m.predict_proba(pca.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(y, oof))
        score_acc += oof
    pooled = score_acc / len(seeds)
    return y.astype(int), pooled, float(np.mean(aucs)), float(np.std(aucs))


def sens_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificity: float,
) -> tuple[float, float]:
    """Return (sensitivity, operating_threshold) at the requested specificity.

    Uses sklearn's roc_curve (which gives FPR = 1 - specificity). We pick
    the largest TPR whose FPR ≤ target_fpr, matching honest_benchmark's
    `sat()` convention. This is the same rule decision_curve_cli uses, so
    the 99% / 95% numbers are directly comparable to existing reports.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    target_fpr = 1.0 - specificity
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, float("nan")
    return float(tpr[idx[-1]]), float(thr[idx[-1]])


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificity: float,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on Sens@Spec.

    Resamples (y_true, y_score) pairs with replacement and recomputes the
    sensitivity at each resample. Returns (ci_lo, ci_hi).
    """
    n = len(y_true)
    sens_samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sens_samples[b], _ = sens_at_specificity(y_true[idx], y_score[idx],
                                                 specificity)
    lo = float(np.quantile(sens_samples, alpha / 2))
    hi = float(np.quantile(sens_samples, 1 - alpha / 2))
    return lo, hi


def per_specificity_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificities: list[float],
    n_boot: int,
    seed: int = 2026,
) -> list[dict]:
    """Full per-spec report: sens, threshold, bootstrap CI, sample counts."""
    rng = np.random.default_rng(seed)
    fpr, _, _ = roc_curve(y_true, y_score)
    out = []
    for sp in specificities:
        sens, op_thr = sens_at_specificity(y_true, y_score, sp)
        ci_lo, ci_hi = bootstrap_ci(y_true, y_score, sp, n_boot, rng)
        target_fpr = 1.0 - sp
        n_below = int((fpr <= target_fpr).sum())
        out.append({
            "specificity": sp,
            "sensitivity": sens,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "operating_threshold": op_thr,
            "n_fpr_points_at_or_below": n_below,
            "n_bootstrap": n_boot,
        })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=DEFAULT_FEAT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--pca", type=int, default=DEFAULT_PCA)
    ap.add_argument("--no-harmonize", action="store_true")
    ap.add_argument("--specificities", type=float, nargs="+",
                    default=DEFAULT_SPECIFICITIES)
    ap.add_argument("--out", default="results/sens_at_spec.json")
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--save-scores", action="store_true",
                    help="also save the raw pooled OOF (y, score) arrays "
                         "as .npy alongside the JSON, so downstream "
                         "scripts (fusion_ablation.py) can reuse them "
                         "without re-fitting LR (LR is stochastic when "
                         "max_iter is hit without convergence).")
    args = ap.parse_args()

    labels, studies = _load_labels_cross_study(args.features_dir)
    X, y, st = load5(labels, studies, args.features_dir)

    n_cancer = int((y == 1).sum())
    n_healthy = int((y == 0).sum())
    unique_studies = sorted(set(st.tolist()))
    print(f"Cohort: {n_cancer} cancer + {n_healthy} healthy = {len(y)} total")
    print(f"Studies: {unique_studies}")
    print(f"Seeds: {args.seeds}, PCA n={args.pca}, harmonize={not args.no_harmonize}")

    print("Computing pooled 5-seed 5-fold OOF predictions ...")
    y_true, score, auc_mean, auc_std = pooled_oof_predictions(
        X, y, st, args.seeds, args.pca, not args.no_harmonize)
    print(f"  Pooled OOF AUC: {auc_mean:.4f} +/- {auc_std:.4f}")

    print("Computing Sens@Spec with bootstrap 95% CI ...")
    rows = per_specificity_table(
        y_true, score, args.specificities,
        args.n_bootstrap, seed=args.bootstrap_seed)

    print("\n" + "=" * 76)
    print(f"{'Specificity':>12} {'Sens':>8} {'CI95 lo':>10} {'CI95 hi':>10} "
          f"{'Threshold':>12}")
    print("=" * 76)
    for r in rows:
        print(f"{r['specificity']*100:>11.2f}% {r['sensitivity']*100:>7.2f}% "
              f"{r['ci95_lo']*100:>9.2f}% {r['ci95_hi']*100:>9.2f}% "
              f"{r['operating_threshold']:>12.4f}")
    print("=" * 76)

    payload = {
        "cohort": {
            "n_total": len(y),
            "n_cancer": n_cancer,
            "n_healthy": n_healthy,
            "studies": unique_studies,
            "features_dir": args.features_dir,
            "labels_file": "labels_cross_study.tsv",
        },
        "config": {
            "seeds": args.seeds,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "pca_n": args.pca,
            "harmonize": not args.no_harmonize,
            "cv_folds": 5,
            "classifier": "LogisticRegression(max_iter=2000)",
            "feature_set": "5-channel (5mb_ratio + 5mb_coverage + 100kb_ratio + "
                           "100kb_counts + FSD-196)",
        },
        "pooled_oof": {
            "n": len(y_true),
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "score_pooling": "mean_across_seeds",
            "in_sample": False,
            "external_validation": False,
        },
        "per_specificity": rows,
        "interpretation": {
            "honest_framing": (
                "All numbers are pooled out-of-fold on the same 627-sample "
                "cohort (5-seed x 5-fold CV). This is the standard ML "
                "benchmark protocol but does NOT replace external "
                "validation on an independent cohort."
            ),
            "specificity_choice": (
                "spec=0.999 is the most stringent clinically-actionable "
                "operating point; spec=0.95 is a relaxed screening "
                "operating point. spec=0.99 matches Galleri / "
                "CancerSEEK's headline specificity."
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}")
    if args.save_scores:
        scores_path = os.path.splitext(args.out)[0] + "_scores.npz"
        np.savez(scores_path, y_true=y_true, score=score,
                 sample_ids=np.array(sorted(labels.keys())))
        print(f"Wrote {scores_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
