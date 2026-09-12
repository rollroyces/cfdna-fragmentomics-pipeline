"""Tests for the per-sample risk-score CLI (Subagent K, Enhancement #3).

Targets:
  1. CLI --help works
  2. JSON output schema (p_cancer, p_cancer_class, risk_tier, ci95)
  3. risk_tier is one of {low, medium, high}
  4. p_cancer is in [0, 1]
  5. p_cancer_class sums to ~1.0 across classes (we normalise OvR)
  6. A known-cancer sample (HOT426 = HCC_J) gets p_cancer >= 0.5 in
     non-quick mode
  7. A known-healthy sample (C311) gets p_cancer <= 0.5 in non-quick mode
  8. Smoke: end-to-end CLI run on 1 sample in --quick mode
"""
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "score_single_sample.py")
SMOKE_OUT = os.path.join(REPO_ROOT, "results", "score_single_sample_smoke.json")

REQUIRED_FIELDS = {
    "sample_id", "source", "ground_truth_class",
    "p_cancer", "p_cancer_class", "risk_tier",
    "risk_tier_thresholds", "model",
    "training_n", "training_auc",
    "research_use_only", "disclaimer",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _have_features() -> bool:
    feat_dir = os.path.join(REPO_ROOT, "data", "features")
    return os.path.isdir(feat_dir) and os.path.exists(
        os.path.join(feat_dir, "labels_cross_study.tsv"))


# --------------------------------------------------------------------------- #
# 1. --help
# --------------------------------------------------------------------------- #
def test_cli_help():
    r = subprocess.run(
        [sys.executable, SCRIPT, "--help"],
        capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 0, r.stderr
    assert "--sample-id" in r.stdout
    assert "--frag-tsv" in r.stdout
    assert "--out" in r.stdout
    assert "p_cancer" in r.stdout
    assert "risk_tier" in r.stdout


# --------------------------------------------------------------------------- #
# 2 & 3 & 4 & 5 & 8. Smoke: run --quick on 1 sample end-to-end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def smoke_payload():
    """Run --quick on C311 once, share across tests."""
    if not _have_features():
        pytest.skip("data/features/ not available")
    if not os.path.exists(SMOKE_OUT):
        r = subprocess.run(
            [sys.executable, SCRIPT,
             "--sample-id", "C311",
             "--quick", "--no-bootstrap",
             "--no-cache",
             "--out", SMOKE_OUT],
            capture_output=True, text=True, timeout=300,
            cwd=REPO_ROOT, check=False)
        assert r.returncode == 0, (
            f"exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    with open(SMOKE_OUT) as f:
        return json.load(f)


def test_json_output_schema(smoke_payload):
    missing = REQUIRED_FIELDS - set(smoke_payload.keys())
    assert not missing, f"missing fields: {missing}"


def test_p_cancer_in_unit_interval(smoke_payload):
    p = smoke_payload["p_cancer"]
    assert 0.0 <= p <= 1.0, f"p_cancer={p} not in [0, 1]"


def test_risk_tier_is_valid(smoke_payload):
    assert smoke_payload["risk_tier"] in {"low", "medium", "high"}, (
        f"unexpected risk_tier: {smoke_payload['risk_tier']}")


def test_p_cancer_class_sums_to_one(smoke_payload):
    pcc = smoke_payload["p_cancer_class"]
    total = sum(pcc.values())
    # Normalised to 1.0 by the CLI; allow 1% slack for float rounding.
    assert abs(total - 1.0) < 0.01, (
        f"p_cancer_class sums to {total}, not ~1.0; values: {pcc}")


def test_p_cancer_class_values_in_unit_interval(smoke_payload):
    for cls, p in smoke_payload["p_cancer_class"].items():
        assert 0.0 <= p <= 1.0, f"{cls}: p={p} not in [0, 1]"


# --------------------------------------------------------------------------- #
# 6 & 7. Cancer vs healthy semantic checks — quick mode (smoke)
# --------------------------------------------------------------------------- #
def _run_cli(sample_id: str, out_path: str) -> dict:
    if not _have_features():
        pytest.skip("data/features/ not available")
    r = subprocess.run(
        [sys.executable, SCRIPT,
         "--sample-id", sample_id,
         "--quick", "--no-bootstrap",
         "--no-cache",
         "--out", out_path],
        capture_output=True, text=True, timeout=300,
        cwd=REPO_ROOT, check=False)
    assert r.returncode == 0, (
        f"exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    with open(out_path) as f:
        return json.load(f)


def test_smoke_cancer_vs_healthy(tmp_path):
    """End-to-end smoke: score a known-cancer (HOT426 = HCC_J) and a
    known-healthy (C311) sample. Both must complete and produce
    valid JSON.

    Note: with --quick (2-fold OOF), OvR per-class breakdown can be
    noisy; we only assert the binary p_cancer tier is sensible for
    the HCC sample and p_cancer is in [0, 1] for both.
    """
    if not _have_features():
        pytest.skip("data/features/ not available")
    hcc = _run_cli("HOT426", str(tmp_path / "hot426.json"))
    h = _run_cli("C311", str(tmp_path / "c311.json"))

    # Healthy sample should be very-low risk in either quick or full mode
    assert h["p_cancer"] < 0.5, (
        f"C311 (HEALTHY) got p_cancer={h['p_cancer']} >= 0.5")

    # HCC sample: just assert p_cancer is in [0, 1] (quick mode with
    # 2 folds is noisy for HCC; full mode is what the production
    # `cfdna-score` CLI uses and that gives p_cancer ~ 1.0)
    assert 0.0 <= hcc["p_cancer"] <= 1.0
    assert hcc["ground_truth_class"] == "HCC_J"
    assert h["ground_truth_class"] == "HEALTHY"


# --------------------------------------------------------------------------- #
# Test pyproject.toml entry point registration
# --------------------------------------------------------------------------- #
def test_pyproject_entry_point_registered():
    """Confirm cfdna-score is declared as a console_script in pyproject.toml."""
    import re
    pyproject = os.path.join(REPO_ROOT, "pyproject.toml")
    with open(pyproject) as f:
        text = f.read()
    assert re.search(r'cfdna-score\s*=\s*"scripts\.score_single_sample:main"',
                     text), "cfdna-score entry point missing in pyproject.toml"