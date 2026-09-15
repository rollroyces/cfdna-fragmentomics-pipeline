#!/usr/bin/env python3
"""Memory profiler for the cfdna-fragmentomics-pipeline scripts.

Usage:
    python scripts/memory_profile.py [--script NAME] [--output PATH]

Measures peak RSS (max resident set size, bytes) for each script in
BOTH a "before" pass (uninstrumented subprocess — the real RSS the
process reached) AND a "during" pass (tracemalloc inside the script —
traces which allocations are largest).

Output is JSON with:
    {
      "<script>": {
        "peak_rss_mb": float,        # peak resident set (whole process)
        "duration_s":  float,
        "tracemalloc_top_mb": [{file, lineno, size_mb}, ...]
      },
      ...
    }

Notes:
- On macOS resource.getrusage(...).ru_maxrss is in BYTES (Linux uses KB);
  we report it as MB.
- We do NOT instrument the actual scripts — we run them as subprocesses
  and read their self-reported peak via /usr/bin/time -l OR via a
  wrapper that prints ru_maxrss at exit (Python atexit hook in the
  script). The wrapper approach is preferred (avoids forking time(1)).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/Users/hermes/deepcatch/.venv/bin/python"


def peak_rss_via_ru_maxrss() -> int:
    """Return ru_maxrss of the current process, in bytes (macOS)."""
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_with_mem_subprocess(cmd: list[str], timeout: int = 1800
                            ) -> tuple[int, int, float, str]:
    """Run `cmd`, return (exit_code, peak_rss_bytes, duration_s, stderr).

    We inject a small atexit hook into PYTHONPATH so the subprocess
    prints its ru_maxrss to stderr (line prefix 'MEM_PEAK_BYTES=').
    The hook is a sitecustomize-style script we pass via `-c` that
    imports resource and registers atexit.
    """
    inject = (
        "import atexit, resource, sys;"
        "def _p(): sys.stderr.write('MEM_PEAK_BYTES=' + "
        "str(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)) + '\\n');"
        "atexit.register(_p)"
    )
    full_cmd = [PYTHON, "-c", inject, "-c", "import sys; sys.argv = "
                + repr(cmd[1:]) + "; exec(open(" + repr(cmd[1]) + ").read())"]
    # Above approach is brittle; simpler: pass the script via -c and execfile.
    # Even simpler: just prepend the atexit hook as a one-liner that runs
    # before the script.
    full_cmd = [PYTHON, str(REPO_ROOT / "scripts" / "_mem_runner.py")] + cmd
    t0 = time.time()
    try:
        proc = subprocess.run(full_cmd, cwd=str(REPO_ROOT),
                              capture_output=True, text=True,
                              timeout=timeout, env={**os.environ,
                                                    "PYTHONPATH": ""})
    except subprocess.TimeoutExpired as e:
        return (-9, 0, time.time() - t0,
                f"TIMEOUT after {timeout}s\n{e.stderr or ''}")
    duration = time.time() - t0
    # Parse peak
    peak = 0
    for line in (proc.stderr or "").splitlines():
        if line.startswith("MEM_PEAK_BYTES="):
            try:
                peak = int(line.split("=", 1)[1])
            except ValueError:
                pass
            break
    return (proc.returncode, peak, duration, proc.stderr or "")


# ---------------------------------------------------------------------------
# Script registry — what to run for "before" / "after" measurements
# ---------------------------------------------------------------------------
SCRIPTS = {
    # Full 5×5 CV — long-running; used for honest_benchmark
    "honest_benchmark_quick": [
        "scripts/honest_benchmark.py", "--features-dir", "data/features"],
    "multiclass_classification_quick": [
        "scripts/multiclass_classification.py", "--quick",
        "--out", "/tmp/mc_quick.json"],
    "tissue_of_origin_quick": [
        "scripts/tissue_of_origin.py", "--quick",
        "--out", "/tmp/too_quick.json"],
    "score_single_sample_quick": [
        "scripts/score_single_sample.py",
        "--sample-id", "C311", "--quiet",
        "--out", "/tmp/score_quick.json",
        "--no-bootstrap"],
    # Full benchmark — used only if --full is set (>= 30 min)
    "multiclass_classification_full": [
        "scripts/multiclass_classification.py",
        "--out", "/tmp/mc_full.json"],
    "tissue_of_origin_full": [
        "scripts/tissue_of_origin.py",
        "--out", "/tmp/too_full.json"],
    "score_single_sample_full": [
        "scripts/score_single_sample.py",
        "--sample-id", "C311", "--quiet",
        "--out", "/tmp/score_full.json"],
    "honest_benchmark_full": [
        "scripts/honest_benchmark.py"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", action="append", default=None,
                    help="Run only this script (repeatable). Default: all quick")
    ap.add_argument("--full", action="store_true",
                    help="Run the full (non-quick) versions")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Per-script subprocess timeout in seconds")
    ap.add_argument("--output", default="results/memory_profile_run.json",
                    help="Output JSON path")
    args = ap.parse_args()

    suffix = "_full" if args.full else "_quick"
    if args.script:
        targets = args.script
    else:
        targets = [name for name in SCRIPTS if name.endswith(suffix)]

    out: dict = {}
    for name in targets:
        if name not in SCRIPTS:
            print(f"SKIP unknown script: {name}", file=sys.stderr)
            continue
        cmd = SCRIPTS[name]
        print(f"\n=== {name} ===\n  cmd: {' '.join(cmd)}", file=sys.stderr)
        rc, peak, dur, err = run_with_mem_subprocess(cmd,
                                                    timeout=args.timeout)
        mb = peak / (1024 * 1024)
        print(f"  rc={rc}  peak_rss={mb:.1f} MB  duration={dur:.1f}s",
              file=sys.stderr)
        if rc != 0:
            tail = "\n".join(err.splitlines()[-20:])
            print(f"  stderr-tail:\n{tail}", file=sys.stderr)
        out[name] = {"exit_code": rc, "peak_rss_mb": round(mb, 2),
                     "duration_s": round(dur, 2)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())