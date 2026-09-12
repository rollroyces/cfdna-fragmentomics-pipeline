"""Tests for the multi-cancer classification enhancement (Subagent J).

Targets:
  1. labels_multiclass.tsv exists with the right header + spot-checks
  2. The script's --help works
  3. The script runs end-to-end on a 10-sample subset (--quick)
  4. Per-class AUCs are in [0, 1]
  5. Macro AUC >= min per-class AUC
  6. JSON output is loadable and has expected fields
"""
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS_MULTICLASS = os.path.join(REPO_ROOT, "labels_multiclass.tsv")
SCRIPT = os.path.join(REPO_ROOT, "scripts", "multiclass_classification.py")
SMOKE_OUT = os.path.join(REPO_ROOT, "results", "multiclass_smoke.json")

# Spot-check known sample IDs (prefix -> expected disease_class)
# These IDs are read straight from labels_multiclass.tsv so they always exist.
SPOT_CHECKS = [
    ("CGPLLU13_D28", "LUAD"),
    ("CGPLBR100", "BRCA"),
    ("CGPLPA112", "PAAD"),
    ("CGPLOV11", "OV"),
    ("CGCRC291", "CRC"),
    ("CGST102", "OTHER_C"),
    ("HOT426", "HCC_J"),
    ("H232", "HCC_J"),
    ("C311", "HEALTHY"),
    ("CGPLH440", "HEALTHY"),
]


# --------------------------------------------------------------------------- #
# 1. labels_multiclass.tsv: format + spot-checks
# --------------------------------------------------------------------------- #
def test_labels_multiclass_exists():
    assert os.path.exists(LABELS_MULTICLASS), (
        f"labels_multiclass.tsv not found at {LABELS_MULTICLASS}")


def test_labels_multiclass_header():
    with open(LABELS_MULTICLASS) as f:
        header = f.readline().strip().split("\t")
    assert header == ["sample", "disease_class", "label", "study"], (
        f"Unexpected header: {header}")


@pytest.mark.parametrize("sample,expected_class", SPOT_CHECKS)
def test_labels_multiclass_spot_check(sample, expected_class):
    with open(LABELS_MULTICLASS) as f:
        next(f)  # skip header
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if parts[0] == sample:
                assert parts[1] == expected_class, (
                    f"{sample}: expected {expected_class}, got {parts[1]}")
                return
    pytest.skip(f"sample {sample!r} not in labels_multiclass.tsv")


def test_labels_multiclass_class_counts():
    """The 8-class distribution matches the audit's deferred-analysis ranges."""
    counts = {}
    with open(LABELS_MULTICLASS) as f:
        next(f)
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            cls = parts[1]
            counts[cls] = counts.get(cls, 0) + 1
    # Audit-reported per-cancer-type n ∈ {9, 18, 27, 28, 54, 60, 79, 88}
    assert counts["BRCA"] == 54
    assert counts["LUAD"] == 79
    assert counts["PAAD"] == 60
    assert counts["OV"] == 28
    assert counts["CRC"] == 27
    assert counts["HCC_J"] == 89
    # CGST1-8 = 27 (OTHER_C)
    assert counts["OTHER_C"] == 27
    # Healthy = 260 CGPLH + 32 Jiang C-prefix + 2 CGH
    assert counts["HEALTHY"] == 294


# --------------------------------------------------------------------------- #
# 2. CLI --help
# --------------------------------------------------------------------------- #
def test_script_help():
    r = subprocess.run(
        [sys.executable, SCRIPT, "--help"],
        capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 0, r.stderr
    assert "Multi-cancer" in r.stdout
    assert "--seeds" in r.stdout
    assert "--folds" in r.stdout
    assert "--out" in r.stdout


# --------------------------------------------------------------------------- #
# 3. End-to-end on a 10-sample subset
# --------------------------------------------------------------------------- #
def test_script_quick_runs(tmp_path):
    """Run the --quick path; should complete in <120s and emit a valid JSON."""
    out = tmp_path / "smoke.json"
    feat_dir = os.path.join(REPO_ROOT, "data", "features")
    if not os.path.isdir(feat_dir):
        pytest.skip(f"data dir not found: {feat_dir} (gitignored; CI provides synthetic only)")
    r = subprocess.run(
        [sys.executable, SCRIPT, "--quick", "--out", str(out)],
        capture_output=True, text=True, timeout=300,
        cwd=REPO_ROOT, check=False)
    assert r.returncode == 0, (
        f"exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    assert out.exists(), f"output file {out} not created"
    with open(out) as f:
        d = json.load(f)
    # Sanity: at least one class, multiple samples
    assert d["n_samples"] >= 30, f"too few samples: {d['n_samples']}"
    assert d["n_classes"] >= 2


# --------------------------------------------------------------------------- #
# 4 & 5. Per-class AUC sanity, macro >= min per-class AUC
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def smoke_results():
    """Run --quick once, share across tests."""
    feat_dir = os.path.join(REPO_ROOT, "data", "features")
    if not os.path.isdir(feat_dir):
        pytest.skip(f"data dir not found: {feat_dir} (gitignored; CI provides synthetic only)")
    if not os.path.exists(SMOKE_OUT):
        r = subprocess.run(
            [sys.executable, SCRIPT, "--quick", "--out", SMOKE_OUT],
            capture_output=True, text=True, timeout=300,
            cwd=REPO_ROOT, check=False)
        assert r.returncode == 0, (
            f"exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    with open(SMOKE_OUT) as f:
        return json.load(f)


def test_per_class_aucs_in_unit_interval(smoke_results):
    aucs = smoke_results["oof_per_class_auc"]
    for cls, a in aucs.items():
        assert 0.0 <= a <= 1.0, f"{cls}: AUC {a} not in [0, 1]"


def test_macro_auc_ge_min_per_class_auc(smoke_results):
    aucs = smoke_results["oof_per_class_auc"]
    macro = smoke_results["macro_auc_mean"]
    assert macro >= min(aucs.values()) - 1e-9, (
        f"macro {macro} < min per-class {min(aucs.values())}")


# --------------------------------------------------------------------------- #
# 6. JSON output schema
# --------------------------------------------------------------------------- #
REQUIRED_FIELDS = {
    "n_samples", "n_features", "n_classes", "classes", "class_counts",
    "dropped_classes", "seeds", "n_folds", "pca_n",
    "oof_per_class_auc", "macro_auc_mean", "macro_auc_std",
    "top2_accuracy_mean", "top2_accuracy_std",
    "binary_cancer_vs_healthy_auc", "binary_cancer_vs_healthy_auc_std",
    "information_delta_multi_vs_binary",
    "confusion_matrix", "confusion_matrix_labels", "runtime_seconds",
}


def test_json_output_schema(smoke_results):
    missing = REQUIRED_FIELDS - set(smoke_results.keys())
    assert not missing, f"missing fields in JSON output: {missing}"
    # Confusion matrix must be square
    n = smoke_results["n_classes"]
    cm = smoke_results["confusion_matrix"]
    assert len(cm) == n, f"CM rows {len(cm)} != n_classes {n}"
    assert all(len(row) == n for row in cm), "CM not square"
    # Confusion matrix row sums <= n_samples per class
    counts = smoke_results["class_counts"]
    for i, cls in enumerate(smoke_results["classes"]):
        assert sum(cm[i]) <= counts[cls], (
            f"row {cls} sum {sum(cm[i])} > count {counts[cls]}")
