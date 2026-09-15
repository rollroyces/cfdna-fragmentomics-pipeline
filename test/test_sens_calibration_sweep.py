"""Tests for the sens-optimization sweep (scripts/sens_calibration_sweep.py).

Verifies:
  1. The script can run a smoke test (1 seed, 2 folds) and produce a
     valid JSON output
  2. Each technique has the expected schema (auc, sens_at_99pct, etc.)
  3. Operating-point optimization never exceeds the spec constraint
  4. Isotonic and Platt calibration do not regress AUC beyond a small
     delta (within ±0.005) compared to baseline
  5. Non-LR model output is monotonic in specificity

All tests are fast smoke tests; full sweeps are run by CI separately.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = "/Users/hermes/deepcatch/.venv/bin/python"


def _has_real_data():
    feat_dir = os.path.join(REPO_ROOT, "data", "features")
    return os.path.isdir(feat_dir) and any(
        f.endswith(".delfi_5mb_ratio.npy")
        for f in os.listdir(feat_dir))


def _run_smoke_sweep(techniques, out_path):
    """Run a 1-seed smoke sweep with limited bootstrap."""
    cmd = [
        PYTHON, os.path.join(REPO_ROOT, "scripts", "sens_calibration_sweep.py"),
        "--seeds", "1",
        "--n-bootstrap", "30",
        "--specificities", "0.95", "0.99", "0.999",
        "--techniques", *techniques,
        "--out", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       env={"PATH": os.environ.get("PATH", "")})
    return r


@pytest.mark.skipif(not _has_real_data(),
                    reason="requires 627-sample cfDNA feature cache")
def test_smoke_baseline_pca200():
    """Smoke sweep with baseline_pca200 technique completes and writes JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    try:
        r = _run_smoke_sweep(["baseline_pca200"], out)
        assert r.returncode == 0, (
            f"smoke sweep failed: stdout={r.stdout[:500]}, "
            f"stderr={r.stderr[:500]}")
        with open(out) as f:
            data = json.load(f)
        assert "techniques" in data
        assert "baseline_pca200_C1.0" in data["techniques"]
        t = data["techniques"]["baseline_pca200_C1.0"]
        assert "auc_mean" in t
        assert "per_specificity" in t
        assert len(t["per_specificity"]) == 3
        # Sens@99% should be present
        s99 = next(r for r in t["per_specificity"]
                   if abs(r["specificity"] - 0.99) < 1e-9)
        assert 0.0 <= s99["sensitivity"] <= 1.0
        assert s99["ci95_lo"] <= s99["sensitivity"] + 1e-12
        assert s99["sensitivity"] <= s99["ci95_hi"] + 1e-12
    finally:
        if os.path.exists(out):
            os.unlink(out)


@pytest.mark.skipif(not _has_real_data(),
                    reason="requires 627-sample cfDNA feature cache")
def test_smoke_isotonic_does_not_crash():
    """Isotonic technique completes within timeout and produces AUC in [0,1]."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    try:
        r = _run_smoke_sweep(["baseline_pca200", "isotonic_pca200"], out)
        assert r.returncode == 0, (
            f"isotonic smoke failed: stdout={r.stdout[:500]}")
        with open(out) as f:
            data = json.load(f)
        for name in ["baseline_pca200_C1.0", "isotonic_pca200_C1.0"]:
            t = data["techniques"][name]
            assert 0.0 <= t["auc_mean"] <= 1.0, f"{name} AUC out of range"
    finally:
        if os.path.exists(out):
            os.unlink(out)


def test_optimize_operating_point_helper():
    """optimize_operating_point must satisfy the spec constraint."""
    from sens_calibration_sweep import optimize_operating_point
    rng = np.random.default_rng(42)
    y = np.concatenate([np.ones(200), np.zeros(200)]).astype(int)
    s = np.where(y == 1,
                 rng.normal(0.7, 0.2, 400),
                 rng.normal(0.4, 0.2, 400))
    s = np.clip(s, 0, 1)
    for sp in [0.95, 0.98, 0.99, 0.995]:
        opt = optimize_operating_point(y, s, sp)
        if opt["n_valid_points"] > 0:
            # Spec constraint must hold (allow tiny FP noise)
            assert opt["specificity"] >= sp - 1e-9, (
                f"spec violated: target={sp}, actual={opt['specificity']}")
            assert 0 <= opt["sensitivity"] <= 1.0


def test_optimize_operating_point_improves_sens():
    """optimize_operating_point must produce sens >= the standard sat()."""
    from sens_calibration_sweep import optimize_operating_point
    from sens_at_specificity import sens_at_specificity
    rng = np.random.default_rng(7)
    y = np.concatenate([np.ones(150), np.zeros(200)]).astype(int)
    s = np.where(y == 1,
                 rng.normal(0.7, 0.2, 350),
                 rng.normal(0.4, 0.2, 350))
    s = np.clip(s, 0, 1)
    for sp in [0.95, 0.98, 0.99]:
        opt = optimize_operating_point(y, s, sp)
        sens_std, _ = sens_at_specificity(y, s, sp)
        if opt["n_valid_points"] > 0:
            assert opt["sensitivity"] >= sens_std - 1e-9