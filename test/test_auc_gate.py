"""Tests for the AUC-reproducibility gate itself.

We don't want CI depending on a 30-second run; we also want the gate
to be exercised in pytest. This module runs the gate function and
verifies the floor passes.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.auc_reproducibility_gate import main as gate_main  # noqa: E402


def test_auc_gate_passes_on_clean_run():
    """The synthetic cohort with the documented signal gives AUC >= 0.80."""
    rc = gate_main()
    assert rc == 0, (
        "AUC gate failed: the synthetic cohort with the documented signal "
        "should give AUC >= 0.80. If this fails, the signal in [:50] bins "
        "may have shifted out of the PCA window or the gate's floor is "
        "misconfigured.")


def test_auc_gate_subprocess_invocation():
    """The CLI entry point (no args) returns 0 on a healthy run."""
    r = subprocess.run(
        ["python", "scripts/auc_reproducibility_gate.py"],
        capture_output=True, text=True, timeout=60,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert r.returncode == 0, (
        f"gate subprocess failed:\nstdout: {r.stdout}\nstderr: {r.stderr}")
    assert "PASS" in r.stdout, r.stdout