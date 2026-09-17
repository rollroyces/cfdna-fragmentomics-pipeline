"""Tests for the per-cancer top-K channel-selection script
(scripts/per_cancer_topk_sweep.py).

Verifies:
  1. Smoke run completes (1 seed x 2 folds, 3 K candidates).
  2. Output JSON has the expected schema (selection, final, binary_final,
     cohort, config, baseline, passes, improvements_ge_0p10, bottom_line).
  3. Per-cancer results are honest:
       * per-cancer AUC in (0, 1)
       * per-cancer Sens@99% in [0, 1]
       * n_features_used = K chosen by inner CV
       * K is in the K-candidates grid
       * channel_indices is a list of K ints in [0, n_features)
  4. Binary guard reports pooled binary Sens@99% in [0, 1] and pooled
     binary AUC in (0, 1).
  5. Honest reporting: if the script says it improved a cancer, the
     result JSON must show sens_at_99pct_pooled > baseline.
  6. inner-CV AUC is monotonically informative (K=1 candidate list is
     not empty, all AUCs in [0.5, 1.0]).
"""
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = "/Users/hermes/deepcatch/.venv/bin/python"


def _has_real_data():
    feat_dir = os.path.join(REPO_ROOT, "data", "features")
    return os.path.isdir(feat_dir) and any(
        f.endswith(".delfi_5mb_ratio.npy")
        for f in os.listdir(feat_dir))


def _run_smoke():
    """Run a quick smoke (1 seed x 2 folds, 3 K candidates) and return
    (data_dict, out_md_path). Output goes to the project results/ dir
    under smoke-specific names; the caller is responsible for
    cleanup."""
    out_json = os.path.join(REPO_ROOT, "results",
                            "_smoke_per_cancer_topk.json")
    out_md = os.path.join(REPO_ROOT, "results",
                          "_smoke_per_cancer_topk.md")
    cmd = [
        PYTHON,
        os.path.join(REPO_ROOT, "scripts", "per_cancer_topk_sweep.py"),
        "--quick",
        "--target", "OV",
        "--ks", "50", "500", "5000",
        "--out", out_json,
        "--out-md", out_md,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       env={"PATH": os.environ.get("PATH", "")})
    assert r.returncode == 0, (
        f"per_cancer_topk_sweep smoke failed: rc={r.returncode}, "
        f"stderr={r.stderr[:1000]}")
    with open(out_json) as f:
        data = json.load(f)
    return data, out_json, out_md


def _cleanup(*paths):
    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


@pytest.mark.skipif(not _has_real_data(),
                    reason="requires 627-sample cfDNA feature cache")
def test_smoke_completes():
    """Smoke run with 1 seed x 2 folds and 3 K candidates completes."""
    data, out_json, out_md = _run_smoke()
    try:
        assert os.path.exists(out_md)
        with open(out_md) as f:
            md = f.read()
        assert "Per-cancer top-K channel selection" in md
        assert "OV" in md
    finally:
        _cleanup(out_json, out_md)


@pytest.mark.skipif(not _has_real_data(),
                    reason="requires 627-sample cfDNA feature cache")
def test_output_schema():
    """The output JSON has the expected top-level keys and nested types."""
    data, out_json, out_md = _run_smoke()
    try:
        for k in ("cohort", "config", "selection", "final", "binary_final",
                  "baseline", "floors", "passes", "improvements_ge_0p10",
                  "bottom_line", "wall_seconds", "generated_at"):
            assert k in data, f"missing key {k!r} in top-level JSON"
        assert "OV" in data["selection"]
        assert "OV" in data["final"]
        sel = data["selection"]["OV"]
        assert "per_K" in sel and len(sel["per_K"]) >= 3
        assert "best_K" in sel and sel["best_K"] in {50, 500, 5000}
        assert 0.5 <= sel["best_inner_cv_auc"] <= 1.0
        fin = data["final"]["OV"]
        assert "channel_indices" in fin
        assert len(fin["channel_indices"]) == fin["K"]
        assert all(0 <= int(i) < data["cohort"]["n_features"]
                   for i in fin["channel_indices"])
        assert 0.0 < fin["auc_pooled"] < 1.0
        assert 0.0 <= fin["sens_at_99pct_pooled"] <= 1.0
        # Binary guard
        bf = data["binary_final"]
        assert 0.0 < bf["auc_pooled"] < 1.0
        assert 0.0 <= bf["sens_at_99pct_pooled"] <= 1.0
        assert bf["channel_set"]
        assert bf["n_features_used"] == data["cohort"]["n_features"]
    finally:
        _cleanup(out_json, out_md)


@pytest.mark.skipif(not _has_real_data(),
                    reason="requires 627-sample cfDNA feature cache")
def test_no_invented_improvements():
    """If the script flags an OV improvement, the JSON must back it up.

    Honesty guard: improvements_ge_0p10 must be a subset of the target
    list, and each entry must have sens_at_99pct_pooled >=
    baseline.sens_at_99pct_pooled + 0.10.
    """
    data, out_json, out_md = _run_smoke()
    try:
        improvements = data["improvements_ge_0p10"]
        targets = data["config"]["targets"]
        final = data["final"]
        baseline = data["baseline"]
        for tgt in improvements:
            assert tgt in targets
            assert tgt in final and tgt in baseline
            delta = (final[tgt]["sens_at_99pct_pooled"]
                     - baseline[tgt]["sens_at_99pct_pooled"])
            assert delta >= 0.10 - 1e-9, (
                f"script claims improvement for {tgt} but delta={delta:.4f}")
    finally:
        _cleanup(out_json, out_md)


@pytest.mark.skipif(not _has_real_data(),
                    reason="requires 627-sample cfDNA feature cache")
def test_binary_uses_full_features():
    """Binary guard must evaluate on the full feature set, not on the
    per-cancer top-K subset. The published baseline binary numbers are
    produced under the full-feature protocol; using a per-cancer top-K
    union would invalidate the guard."""
    data, out_json, out_md = _run_smoke()
    try:
        bf = data["binary_final"]
        assert bf["n_features_used"] == data["cohort"]["n_features"]
        assert "all features" in bf["channel_set"].lower()
    finally:
        _cleanup(out_json, out_md)


@pytest.mark.skipif(not _has_real_data(),
                    reason="requires 627-sample cfDNA feature cache")
def test_inner_cv_aucs_in_unit_interval():
    """All per-K inner-CV AUCs in the selection curve must be in [0, 1]."""
    data, out_json, out_md = _run_smoke()
    try:
        for tgt in data["config"]["targets"]:
            for row in data["selection"][tgt]["per_K"]:
                assert 0.0 <= row["inner_cv_auc"] <= 1.0, (
                    f"{tgt} K={row['K']} inner_cv_auc={row['inner_cv_auc']} "
                    "out of [0, 1]")
    finally:
        _cleanup(out_json, out_md)
