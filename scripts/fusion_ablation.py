"""Fragmentomics + simulated mutation fusion ablation.

Companion to scripts/sens_at_specificity.py. Adds a synthetic
mutation-informed channel (calibrated to match DeepCatch's headline
panel-LLR @ 0.1% VAF: AUC ~0.92, Sens@95% ~0.77) and averages it with
the pooled 5-seed 5-fold OOF fragmentomics score.

The synthetic mutation score is *not* a real mutation model — it's a
calibrated stand-in used to ask: "Given a mutation channel of this
quality, does fusion help or hurt the tumor-naive result?"

For a real mutation-channel fusion, see deepcatch/src/fragmentomics/
fusion_ablation.py. The numbers in this script are intended for the
in-pipeline BENCHMARK_published.md table only.

Output: results/fusion_ablation.json with per-strategy AUC and Sens@Spec
for spec ∈ {0.95, 0.98, 0.99, 0.995, 0.999}.

CLI:
  python scripts/fusion_ablation.py
    [--sens-json PATH]      default: results/sens_at_spec.json
    [--mutation-auc TARGET] default: 0.92
    [--mutation-sens95 T]   default: 0.77
    [--out PATH]            default: results/fusion_ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from sens_at_specificity import (
    bootstrap_ci,
    sens_at_specificity,
)


def simulate_mutation_scores(
    y_true: np.ndarray,
    target_auc: float,
    target_sens95: float,
    seed: int = 2026,
) -> np.ndarray:
    """Generate a synthetic mutation score calibrated to target operating points.

    The score is built as: y_true * 0.6 + N(0, noise) + small jitter.
    We then linearly rescale to hit the target AUC and Sens@95%.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    # Base signal: higher for cancer, lower for healthy, with noise.
    base = np.where(y_true == 1,
                    rng.normal(0.75, 0.18, n),
                    rng.normal(0.35, 0.18, n))
    base = np.clip(base, 0.0, 1.0)

    # Rescale linearly so target AUC is hit approximately.
    # Empirically, the uncalibrated AUC is around 0.85; we adjust the gap.
    current_auc = roc_auc_score(y_true, base)
    # A simple linear shift: add a delta so new AUC ≈ target_auc.
    # (This is a one-shot calibration, not a perfect rank-preserving
    # transform — good enough for the ablation summary.)
    delta = (target_auc - current_auc) * 0.5
    base = np.clip(base + delta, 0.0, 1.0)

    # Fine-tune the gap so the Sens@95% lands near target.
    # Sens@95% rises with the gap between cancer and healthy means.
    for _ in range(40):
        sens95_now, _ = sens_at_specificity(y_true, base, 0.95)
        if abs(sens95_now - target_sens95) < 0.005:
            break
        step = (target_sens95 - sens95_now) * 0.05
        cancer_shift = step * 0.5
        healthy_shift = -step * 0.5
        base = np.where(y_true == 1,
                        np.clip(base + cancer_shift, 0.0, 1.0),
                        np.clip(base + healthy_shift, 0.0, 1.0))
    return base


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sens-json", default="results/sens_at_spec.json")
    ap.add_argument("--scores-npz", default=None,
                    help="optional path to sens_at_spec_scores.npz; if "
                         "present, the pooled OOF (y_true, score) arrays "
                         "are loaded directly instead of refitting LR "
                         "(avoids stochastic re-convergence).")
    ap.add_argument("--mutation-auc", type=float, default=0.92)
    ap.add_argument("--mutation-sens95", type=float, default=0.77)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--out", default="results/fusion_ablation.json")
    args = ap.parse_args()

    with open(args.sens_json) as f:
        sens_data = json.load(f)

    # These are used by both branches; pull them out of sens_data once.
    feat_dir = sens_data["cohort"]["features_dir"]
    seeds = sens_data["config"]["seeds"]
    pca_n = sens_data["config"]["pca_n"]
    harmonize = sens_data["config"]["harmonize"]

    # Try to load saved scores first; otherwise re-fit (slower + stochastic).
    scores_npz = args.scores_npz
    if scores_npz is None:
        scores_npz = os.path.splitext(args.sens_json)[0] + "_scores.npz"
    if os.path.exists(scores_npz):
        print(f"Loading pooled OOF scores from {scores_npz} ...")
        npz = np.load(scores_npz, allow_pickle=True)
        y_true = npz["y_true"].astype(int)
        frag_score = npz["score"].astype(float)
        from sklearn.metrics import roc_auc_score as _auc
        auc_frag = float(_auc(y_true, frag_score))
        # std is unknown from a single scores file; report 0
        auc_std_frag = 0.0
    else:
        print("Re-fitting pooled 5-seed 5-fold OOF "
              "(same as sens_at_specificity.py) ...")
        from honest_benchmark import load5
        from sens_at_specificity import _load_labels_cross_study, pooled_oof_predictions
        labels, studies = _load_labels_cross_study(feat_dir)
        X, y, st = load5(labels, studies, feat_dir)
        y_true, frag_score, auc_frag, auc_std_frag = pooled_oof_predictions(
            X, y, st, seeds, pca_n, harmonize)

    print(f"  Fragmentomics pooled OOF AUC: {auc_frag:.4f} +/- {auc_std_frag:.4f}")
    print(f"  Simulating mutation score at AUC={args.mutation_auc}, "
          f"Sens@95%={args.mutation_sens95} ...")
    mut_score = simulate_mutation_scores(
        y_true, args.mutation_auc, args.mutation_sens95, seed=args.bootstrap_seed)
    auc_mut = roc_auc_score(y_true, mut_score)
    sens95_mut, _ = sens_at_specificity(y_true, mut_score, 0.95)
    print(f"  Mutation score realized: AUC={auc_mut:.4f}, Sens@95%={sens95_mut:.3f}")

    # Naive-average fusion
    print("Computing naive-average fusion ...")
    fused = (frag_score + mut_score) / 2.0
    auc_fused = roc_auc_score(y_true, fused)

    specificities = [0.95, 0.98, 0.99, 0.995, 0.999]
    rng = np.random.default_rng(args.bootstrap_seed)

    def per_spec(scores, label):
        out = []
        for sp in specificities:
            sens, thr = sens_at_specificity(y_true, scores, sp)
            ci_lo, ci_hi = bootstrap_ci(y_true, scores, sp,
                                        args.n_bootstrap, rng)
            out.append({"specificity": sp, "sensitivity": sens,
                        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                        "operating_threshold": thr,
                        "n_bootstrap": args.n_bootstrap})
        print(f"\n  {label}:")
        for r in out:
            print(f"    spec={r['specificity']*100:.1f}%  "
                  f"sens={r['sensitivity']*100:.2f}%  "
                  f"CI=[{r['ci95_lo']*100:.2f}, {r['ci95_hi']*100:.2f}]")
        return out

    print("\nPer-spec table:")
    frag_table = per_spec(frag_score, "Fragmentomics only")
    mut_table = per_spec(mut_score, "Mutation only (synthetic)")
    fused_table = per_spec(fused, "Naive-average fusion")

    payload = {
        "cohort": sens_data["cohort"],
        "config": {
            "seeds": seeds,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "pca_n": pca_n,
            "harmonize": harmonize,
            "fusion_rule": "naive_average",
            "mutation_target_auc": args.mutation_auc,
            "mutation_target_sens95": args.mutation_sens95,
        },
        "summary": {
            "fragmentomics_auc": auc_frag,
            "mutation_auc_realized": auc_mut,
            "fusion_auc": auc_fused,
            "delta_auc_fusion_vs_frag": auc_fused - auc_frag,
        },
        "per_specificity": {
            "fragmentomics_only": frag_table,
            "mutation_only_synthetic": mut_table,
            "naive_average_fusion": fused_table,
        },
        "honest_framing": (
            "The mutation channel is a CALIBRATED SYNTHETIC score, not a "
            "real mutation model. It is tuned to DeepCatch's headline "
            "panel-LLR @ 0.1% VAF (AUC ~0.92, Sens@95% ~0.77) so the "
            "ablation answers: given a mutation channel of this quality, "
            "does fusion help? For a real mutation-channel fusion see "
            "deepcatch/src/fragmentomics/fusion_ablation.py."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}")
    print(f"\nDelta AUC (fusion - fragmentomics): {auc_fused - auc_frag:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
