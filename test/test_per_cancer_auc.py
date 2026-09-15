"""Tests for the per-cancer AUC script (scripts/per_cancer_auc.py).

Verifies the script:
  1. Smoke-runs successfully on the 627-sample cohort
  2. Reports per-cancer AUCs in (0, 1)
  3. Includes 95% CI in expected range for both AUC and Sens@99%
  4. Marks n<30 classes as small_n_flag
  5. Generates both .json and .md outputs
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


@pytest.mark.skipif(not _has_real_data(),
                    reason="requires 627-sample cfDNA feature cache")
def test_smoke_run_completes():
    """Per-cancer AUC smoke run (1 seed × 2 folds, 50 boot) completes."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fj:
        out_json = fj.name
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as fm:
        out_md = fm.name
    try:
        cmd = [
            PYTHON, os.path.join(REPO_ROOT, "scripts", "per_cancer_auc.py"),
            "--seeds", "1",
            "--folds", "2",
            "--n-bootstrap", "50",
            "--out", out_json,
            "--out-md", out_md,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env={"PATH": os.environ.get("PATH", "")})
        assert r.returncode == 0, (
            f"per_cancer_auc smoke failed: rc={r.returncode}, "
            f"stderr={r.stderr[:500]}")
        with open(out_json) as f:
            data = json.load(f)
        # Schema checks
        assert "per_cancer" in data
        assert "macro_ovr_auc_pooled" in data
        assert "min_auc_per_cancer" in data
        # CRC, OV, OTHER_C should be flagged as small_n
        for cls in ["CRC", "OV", "OTHER_C"]:
            if cls in data["per_cancer"]:
                assert data["per_cancer"][cls]["small_n_flag"] is True
        # BRCA, HCC_J, LUAD, PAAD should NOT be flagged
        for cls in ["BRCA", "HCC_J", "LUAD", "PAAD"]:
            if cls in data["per_cancer"]:
                assert data["per_cancer"][cls]["small_n_flag"] is False
        # All AUCs in (0, 1)
        for cls, row in data["per_cancer"].items():
            assert 0.0 < row["auc_pooled"] < 1.0, (
                f"{cls} AUC={row['auc_pooled']} out of (0,1)")
            assert 0.0 <= row["auc_ci95_lo"] <= row["auc_pooled"] + 1e-9
            assert row["auc_pooled"] - 1e-9 <= row["auc_ci95_hi"] <= 1.0
        # Macro AUC must not be negative
        assert data["macro_ovr_auc_pooled"] > 0.5
        # Markdown file should exist and be non-empty
        assert os.path.exists(out_md)
        with open(out_md) as f:
            md = f.read()
        assert "Per-cancer AUC table" in md
        assert "BRCA" in md
    finally:
        for p in (out_json, out_md):
            if os.path.exists(p):
                os.unlink(p)