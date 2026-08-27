"""Smoke test: honest_benchmark.py --help must complete in <10 seconds
and NOT trigger the full benchmark.

The previous version of honest_benchmark.py had all work at module
level (sections A through E), so 'python scripts/honest_benchmark.py
--help' would load data, run 5-seed CV on 5+5+5+3+1 = 19 model
configs (≈5 minutes wall-clock). This test guards against the
regression.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TIMEOUT_SEC = 10


def test_honest_benchmark_help_is_fast():
    """python scripts/honest_benchmark.py --help should take <10 seconds."""
    cmd = [PY, "scripts/honest_benchmark.py", "--help"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       timeout=TIMEOUT_SEC,
                       env={**os.environ, "PYTHONPATH": ""})
    elapsed = time.time() - t0
    assert r.returncode == 0, (
        f"honest_benchmark.py --help returned {r.returncode}\n"
        f"stdout: {r.stdout[-500:]}\nstderr: {r.stderr[-500:]}")
    assert elapsed < TIMEOUT_SEC, (
        f"honest_benchmark.py --help took {elapsed:.1f}s "
        f"(should be <{TIMEOUT_SEC}s). If this is failing because the "
        f"script is running the full benchmark, the module-level work "
        f"needs to be wrapped in run_honest_benchmark() and called "
        f"from main().")
    assert "--features-dir" in r.stdout, (
        f"--help output missing --features-dir argument: {r.stdout}")


def test_honest_benchmark_help_does_not_mention_data_loading():
    """A correct --help should NOT print any 'Cohort:' or 'N=' lines.

    If it does, the module-level work is being run before main()
    parses args, which is the original bug.
    """
    cmd = [PY, "scripts/honest_benchmark.py", "--help"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       timeout=TIMEOUT_SEC,
                       env={**os.environ, "PYTHONPATH": ""})
    assert "Cohort:" not in r.stdout, (
        f"--help output contains 'Cohort:' which means the script is "
        f"running data-loading code before arg parsing:\n{r.stdout}")
    assert "AUC" not in r.stdout or "AUC" in r.stdout.split("--features-dir")[0], (
        f"--help output contains AUC which means the script is running "
        f"the benchmark before arg parsing:\n{r.stdout}")
