"""Memory-peak-reporting runner.

Used by scripts/memory_profile.py to run another script as a subprocess
while reporting its peak RSS via stderr at exit.

Usage:
    /Users/hermes/deepcatch/.venv/bin/python scripts/_mem_runner.py \
        <script_path> [<arg> ...]

Writes `MEM_PEAK_BYTES=<int>\n` to stderr at exit (atexit hook).
"""
import atexit
import resource
import runpy
import sys

# Inject the script's real argv BEFORE runpy resets sys.argv[0].
# runpy.run_path() preserves sys.argv as-is.
script_path = sys.argv[1]
sys.argv = [script_path] + sys.argv[2:]


def _report_peak() -> None:
    """Emit peak RSS to stderr at process exit."""
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    sys.stderr.write(f"MEM_PEAK_BYTES={peak}\n")
    sys.stderr.flush()


atexit.register(_report_peak)
runpy.run_path(script_path, run_name="__main__")