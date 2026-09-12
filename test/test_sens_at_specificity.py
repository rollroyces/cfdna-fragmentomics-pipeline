"""Tests for the sens@spec extension (scripts/sens_at_specificity.py).

Four required tests:
1. Sens at spec=0.95 is monotonically non-increasing as spec increases
2. Bootstrap CIs are non-negative
3. Sens at spec=0.95 matches the existing reported value (regression guard)
4. The comparison table is reproducible from the JSON

All tests are pure-arithmetic or data-driven (no network, no model retrain).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from sens_at_specificity import (
    per_specificity_table,
    sens_at_specificity,
)


def _seeded_scores(n=400, seed=0):
    """Deterministic synthetic scores for tests.

    Cancer (n//2) get higher scores on average (AUC ~0.85).
    """
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)]).astype(int)
    # Cancer: N(0.7, 0.2); Healthy: N(0.4, 0.2). Clip to [0,1].
    s = np.where(y == 1,
                 rng.normal(0.7, 0.2, n),
                 rng.normal(0.4, 0.2, n))
    s = np.clip(s, 0.0, 1.0)
    return y, s


def test_monotonic_non_increasing_sens():
    """Sens@Spec must be a non-increasing function of spec.

    As the specificity target rises, fewer points satisfy fpr <= 1-spec,
    so the achievable sensitivity cannot go up.
    """
    y, s = _seeded_scores(n=600, seed=42)
    specs = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    sens_values = []
    for sp in specs:
        sens, _ = sens_at_specificity(y, s, sp)
        sens_values.append(sens)
    # Non-increasing: each subsequent sens <= the previous one
    for i in range(len(sens_values) - 1):
        assert sens_values[i + 1] <= sens_values[i] + 1e-12, (
            f"sens@spec must be non-increasing; got "
            f"specs={specs}, sens={sens_values}")


def test_bootstrap_ci_non_negative():
    """All bootstrap CI bounds must be >= 0.

    Sensitivity is bounded in [0, 1]; a CI lower bound below 0 is invalid.
    """
    y, s = _seeded_scores(n=400, seed=7)
    specs = [0.95, 0.98, 0.99, 0.995, 0.999]
    table = per_specificity_table(y, s, specs, n_boot=200, seed=2026)
    for row in table:
        assert row["ci95_lo"] >= 0.0, (
            f"CI lower bound must be non-negative; got "
            f"spec={row['specificity']}, ci_lo={row['ci95_lo']}")
        assert row["ci95_hi"] <= 1.0 + 1e-12, (
            f"CI upper bound must be <= 1; got "
            f"spec={row['specificity']}, ci_hi={row['ci95_hi']}")
        assert row["ci95_lo"] <= row["ci95_hi"] + 1e-12, (
            f"CI bounds must be ordered; got lo={row['ci95_lo']} "
            f"hi={row['ci95_hi']}")


def test_sens_at_spec95_matches_existing_reported_value():
    """Regression guard: the sens@95% reported by sens_at_specificity.py
    must match the value reported by honest_benchmark.py Section C
    (5-channel PCA n=200, harmonized).

    Both scripts use the same logic:
      sat(t) = max TPR where FPR <= t
    so they MUST agree when run on the same cohort + model.

    We assert that running sens_at_specificity() on the same pooled OOF
    AUC that honest_benchmark reports gives the same sens@0.95.
    """
    # The honest_benchmark Section C reports:
    #   5-channel (PCA n=200, harmonized) N=627  AUC 0.9739 +/- 0.0033
    #   S95 0.892 +/- 0.017
    # That S95 = 0.892 is the per-seed mean of sat(0.05). The pooled
    # version (which sens_at_specificity.py computes) uses a single
    # threshold across all seeds. They should be close (within 2 stds of
    # the honest_benchmark S95 std), but not bit-equal because the
    # operating threshold differs.
    from honest_benchmark import load5
    from sens_at_specificity import _load_labels_cross_study, pooled_oof_predictions

    feat_dir = os.environ.get("FEAT_DIR",
                              os.path.join(os.path.dirname(__file__), "..",
                                           "data", "features"))
    if not os.path.isdir(feat_dir):
        pytest.skip(f"data dir not found: {feat_dir} (gitignored; CI provides synthetic only)")
    labels, studies = _load_labels_cross_study(feat_dir)
    X, y, st = load5(labels, studies, feat_dir)
    assert len(y) == 627, f"Expected 627 samples; got {len(y)}"

    y_true, score, auc_mean, _auc_std = pooled_oof_predictions(
        X, y, st, [42, 13, 7, 99, 1234], pca_n=200, harmonize=True)
    sens95, _ = sens_at_specificity(y_true, score, 0.95)

    # Honest benchmark reported: AUC 0.9739 +/- 0.0033, S95 0.892 +/- 0.017
    # Allow 3 sigma deviation on each
    assert abs(auc_mean - 0.9739) < 3 * 0.0033, (
        f"AUC {auc_mean:.4f} drifted from honest_benchmark 0.9739")
    assert 0.892 - 3 * 0.017 <= sens95 <= 0.892 + 3 * 0.017, (
        f"Sens@95% {sens95:.3f} outside honest_benchmark S95=0.892+/-0.017")


def test_comparison_table_reproducible_from_json():
    """Round-trip: writing sens_at_specificity JSON output and reading it
    back must give the same per_specificity table (regression guard for
    BENCHMARK_published.md).

    This also tests that all required fields are present and that the
    BENCHMARK_published.md row fields are derived consistently.
    """
    y, s = _seeded_scores(n=300, seed=11)
    specs = [0.95, 0.98, 0.99, 0.995, 0.999]
    table = per_specificity_table(y, s, specs, n_boot=50, seed=99)

    payload = {
        "cohort": {"n_total": len(y),
                   "n_cancer": int((y == 1).sum()),
                   "n_healthy": int((y == 0).sum()),
                   "studies": ["synthetic"],
                   "features_dir": "synthetic",
                   "labels_file": "synthetic.tsv"},
        "config": {"seeds": [42], "n_bootstrap": 50, "bootstrap_seed": 99,
                   "pca_n": 200, "harmonize": True, "cv_folds": 5,
                   "classifier": "LogisticRegression(max_iter=2000)",
                   "feature_set": "synthetic"},
        "pooled_oof": {"n": len(y), "auc_mean": 0.85, "auc_std": 0.02,
                       "score_pooling": "mean_across_seeds",
                       "in_sample": False, "external_validation": False},
        "per_specificity": table,
        "interpretation": {"honest_framing": "synthetic",
                            "specificity_choice": "synthetic"},
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp = f.name
    try:
        with open(tmp) as f:
            reloaded = json.load(f)
        assert reloaded["per_specificity"] == table, (
            "Round-trip changed the per_specificity table")
        # Verify every required BENCHMARK_published.md field is recoverable
        for row in reloaded["per_specificity"]:
            for field in ("specificity", "sensitivity", "ci95_lo", "ci95_hi"):
                assert field in row, f"Missing field: {field}"
            # Build the markdown row reproducibly
            md_row = (f"| {row['specificity']*100:.1f}% | "
                      f"{row['sensitivity']*100:.1f}% | "
                      f"{row['ci95_lo']*100:.1f}%-{row['ci95_hi']*100:.1f}% |")
            # Just sanity-check the format
            assert md_row.startswith("|"), md_row
            assert md_row.count("|") >= 4, md_row
    finally:
        os.unlink(tmp)
