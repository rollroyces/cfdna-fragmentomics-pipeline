"""Tests for the PPV-screening calculator (pure arithmetic, no model)."""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ppv_screening import ppv, npv


def test_ppv_basic_known_values():
    """A 100% sens / 100% spec test should give PPV = 1.0 (perfect)."""
    for prev in [0.001, 0.01, 0.05, 0.1]:
        assert abs(ppv(1.0, 1.0, prev) - 1.0) < 1e-12, (
            f"PPV(sens=1, spec=1, prev={prev}) should be 1.0")


def test_ppv_random_classifier():
    """A random classifier (sens = 1-spec) should give PPV = prev."""
    for prev in [0.001, 0.01, 0.05]:
        p = ppv(0.7, 0.3, prev)
        assert abs(p - prev) < 1e-12, (
            f"PPV for random classifier should equal prev, got {p}")


def test_npv_basic_known_values():
    """Perfect sensitivity + 100% specificity -> NPV = 1 (never a false negative)."""
    # sens=1, spec=1: all healthy correctly identified, all cancer caught
    # TN = spec*(1-prev), FN = (1-sens)*prev = 0
    # NPV = TN / (TN + FN) = 1.0
    assert abs(npv(1.0, 1.0, 0.5) - 1.0) < 1e-12
    # sens=1, spec=0: spec=0 means no healthy correctly identified, but
    # sens=1 means no cancer missed. TN=0, FN=0. So NPV is undefined
    # (division by zero in formula). npv() returns 0.0 as safe default.
    assert npv(1.0, 0.0, 0.5) == 0.0
    # sens=0, spec=1: no cancer caught, all healthy correctly identified.
    # TN = spec*(1-prev), FN = (1-sens)*prev = prev. NPV = (1-prev)/1 = 1-prev.
    assert abs(npv(0.0, 1.0, 0.4) - 0.6) < 1e-12


def test_ppv_higher_prevalence_increases_ppv():
    """PPV must monotonically increase with prevalence (Bayes)."""
    sens, spec = 0.9, 0.95
    ppv_low = ppv(sens, spec, 0.001)
    ppv_mid = ppv(sens, spec, 0.01)
    ppv_high = ppv(sens, spec, 0.1)
    assert ppv_low < ppv_mid < ppv_high


def test_ppv_higher_specificity_increases_ppv():
    """PPV must monotonically increase with specificity (Bayes)."""
    sens, prev = 0.9, 0.01
    ppv_low_spec = ppv(sens, 0.7, prev)
    ppv_high_spec = ppv(sens, 0.99, prev)
    assert ppv_low_spec < ppv_high_spec


def test_ppv_screening_main_writes_json():
    """End-to-end: ppv_screening.main() produces a valid JSON file."""
    import tempfile
    from ppv_screening import main
    import io
    from contextlib import redirect_stdout

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        # Re-run with --out pointing at tmp
        sys.argv = ["ppv_screening.py", "--out", tmp_path]
        with redirect_stdout(io.StringIO()):
            main()
        with open(tmp_path) as f:
            data = json.load(f)
        assert "results" in data
        assert "operating_points" in data
        assert "prevalences" in data
        assert len(data["results"]) == len(data["operating_points"]) * len(data["prevalences"])
        # Verify PPV monotonically increases with prevalence
        sens95 = [r for r in data["results"]
                  if r["operating_point"] == "Sens@95% (LR no-PCA C=1000)"]
        ppvs = [r["ppv"] for r in sorted(sens95, key=lambda x: x["prevalence"])]
        assert all(ppvs[i] < ppvs[i+1] for i in range(len(ppvs)-1)), (
            f"PPV should increase with prevalence; got {ppvs}")
    finally:
        os.unlink(tmp_path)
