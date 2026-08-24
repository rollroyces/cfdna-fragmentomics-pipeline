"""Smoke test: lr_regularization_sweep's print statements reference
variables in scope. Catches the n_nonzero bug that crashed L1
sweeps on commit 7bbdef5."""
import os
import subprocess
import sys


def test_lr_sweep_runs_l1_without_name_error():
    """The L1 sweep previously crashed because the print statement
    referenced an out-of-scope variable name. This regression test
    forces the L1 path to run and verifies the script exits cleanly."""
    repo = "/Users/hermes/cfdna-fragmentomics-pipeline"
    py = "/Users/hermes/deepcatch/.venv/bin/python"
    # Use a small config so the L1 sweep finishes quickly.
    # L1 with saga + 60k features can be slow; reduce seeds to 1
    # and skip past L2 by setting --c-values only includes a single L2 value.
    # But we need L1 to actually run. The L1 code lives after the L2 loop,
    # so passing any L2 value triggers L1 next.
    # Use a single L2 C value and a single L1 C value to keep this fast.
    cmd = [
        py, "scripts/lr_regularization_sweep.py",
        "--seeds", "1",
        "--c-values", "1.0",
        "--out", "/tmp/test_lr_sweep.json",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                        cwd=repo, timeout=600,
                        env={**os.environ, "PYTHONPATH": ""})
    assert r.returncode == 0, (
        f"lr_regularization_sweep crashed with exit {r.returncode}.\n"
        f"stdout tail: {r.stdout[-500:]}\nstderr tail: {r.stderr[-500:]}")
    assert "n_nonzero" not in r.stderr or "NameError" not in r.stderr, (
        "L1 sweep printed NameError — the n_nonzero scope bug returned")
