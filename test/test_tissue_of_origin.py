"""Tests for the Tissue-of-Origin (TOO) classifier (Subagent L, Enhancement #4).

Targets:
  1. CLI --help works
  2. JSON output schema (per_class_auc, top1_acc, top2_acc, confusion_matrix)
  3. Per-class AUCs are in [0, 1]
  4. Top-2 accuracy >= Top-1 accuracy
  5. Smoke test runs end-to-end
  6. cancer_only flag is set in JSON output
  7. Confusion matrix is square and rows sum correctly
"""
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS_MULTICLASS = os.path.join(REPO_ROOT, "labels_multiclass.tsv")
SCRIPT = os.path.join(REPO_ROOT, "scripts", "tissue_of_origin.py")
SMOKE_OUT = os.path.join(REPO_ROOT, "results", "tissue_of_origin_smoke.json")

# Expected cancer-only label space (HEALTHY dropped)
TOO_CLASSES = ["BRCA", "CRC", "HCC_J", "LUAD", "OV", "PAAD"]


# --------------------------------------------------------------------------- #
# 1. CLI --help
# --------------------------------------------------------------------------- #
def test_script_help():
    r = subprocess.run(
        [sys.executable, SCRIPT, "--help"],
        capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 0, r.stderr
    assert "Tissue-of-Origin" in r.stdout
    assert "cancer" in r.stdout.lower()
    assert "--seeds" in r.stdout
    assert "--folds" in r.stdout
    assert "--out" in r.stdout
    assert "--quick" in r.stdout


# --------------------------------------------------------------------------- #
# 2. End-to-end smoke run (1 seed × 2 folds) — generates SMOKE_OUT
# --------------------------------------------------------------------------- #
def test_script_quick_runs(tmp_path):
    """The --quick path completes and emits a valid JSON."""
    out = tmp_path / "smoke.json"
    feat_dir = os.path.join(REPO_ROOT, "data", "features")
    if not os.path.isdir(feat_dir):
        pytest.skip(f"data dir not found: {feat_dir} "
                    f"(gitignored; CI provides synthetic only)")
    r = subprocess.run(
        [sys.executable, SCRIPT, "--quick", "--out", str(out)],
        capture_output=True, text=True, timeout=600,
        cwd=REPO_ROOT, check=False)
    assert r.returncode == 0, (
        f"exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    assert out.exists(), f"output file {out} not created"
    with open(out) as f:
        d = json.load(f)
    # Cancer-only cohort has 6 classes, ≥200 samples (337 - 1 dropped)
    assert d["cancer_only"] is True
    assert d["n_classes"] >= 4, f"too few classes: {d['n_classes']}"
    assert d["n_samples"] >= 200, f"too few samples: {d['n_samples']}"


# --------------------------------------------------------------------------- #
# 3 & 4 & 6 & 7. Sanity checks against the shared smoke output
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def smoke_results():
    """Run --quick once, share across sanity tests."""
    feat_dir = os.path.join(REPO_ROOT, "data", "features")
    if not os.path.isdir(feat_dir):
        pytest.skip(f"data dir not found: {feat_dir} "
                    f"(gitignored; CI provides synthetic only)")
    if not os.path.exists(SMOKE_OUT):
        r = subprocess.run(
            [sys.executable, SCRIPT, "--quick", "--out", SMOKE_OUT],
            capture_output=True, text=True, timeout=600,
            cwd=REPO_ROOT, check=False)
        assert r.returncode == 0, (
            f"exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    with open(SMOKE_OUT) as f:
        return json.load(f)


REQUIRED_FIELDS = {
    # Cohort
    "n_samples", "n_features", "n_classes", "classes", "class_counts",
    "dropped_classes",
    # CV setup
    "seeds", "n_folds", "pca_n", "lr_C", "min_class_n", "cancer_only",
    # Metrics (TOO contract)
    "per_class_auc", "macro_auc_mean", "macro_auc_std",
    "top1_accuracy_pooled_oof", "top2_accuracy_pooled_oof",
    "top1_accuracy_mean", "top1_accuracy_std",
    "top2_accuracy_mean", "top2_accuracy_std",
    "per_class_auc_mean_across_seeds", "per_class_auc_std_across_seeds",
    # Confusion matrix
    "confusion_matrix", "confusion_matrix_labels",
    "runtime_seconds",
}


def test_json_output_schema(smoke_results):
    missing = REQUIRED_FIELDS - set(smoke_results.keys())
    assert not missing, f"missing fields in JSON output: {missing}"


def test_cancer_only_flag_is_true(smoke_results):
    """TOO must be cancer-only — HEALTHY excluded from label space."""
    assert smoke_results["cancer_only"] is True
    assert "HEALTHY" not in smoke_results["classes"], (
        f"HEALTHY must NOT be in TOO label space, "
        f"got {smoke_results['classes']}")


def test_classes_are_cancer_only(smoke_results):
    classes = set(smoke_results["classes"])
    expected = set(TOO_CLASSES)
    # All expected cancer classes should be present (with min_class_n=25)
    missing = expected - classes
    assert not missing, (
        f"missing cancer classes from TOO output: {missing}")


def test_per_class_aucs_in_unit_interval(smoke_results):
    aucs = smoke_results["per_class_auc"]
    assert aucs, "per_class_auc is empty"
    for cls, a in aucs.items():
        assert 0.0 <= a <= 1.0, f"{cls}: AUC {a} not in [0, 1]"


def test_top2_ge_top1(smoke_results):
    """Top-2 accuracy must be >= Top-1 accuracy by definition."""
    t1 = smoke_results["top1_accuracy_pooled_oof"]
    t2 = smoke_results["top2_accuracy_pooled_oof"]
    assert t2 + 1e-9 >= t1, (
        f"top2_acc {t2} < top1_acc {t1} (impossible)")
    # Also check the per-seed means
    t1m = smoke_results["top1_accuracy_mean"]
    t2m = smoke_results["top2_accuracy_mean"]
    assert t2m + 1e-9 >= t1m, (
        f"top2_acc (seed-mean) {t2m} < top1_acc (seed-mean) {t1m}")


def test_topk_better_than_chance(smoke_results):
    """Top-1 must beat 1/K random baseline (otherwise the model learned nothing)."""
    k = smoke_results["n_classes"]
    chance = 1.0 / k
    t1 = smoke_results["top1_accuracy_pooled_oof"]
    assert t1 > chance, (
        f"top1_acc {t1:.4f} <= chance baseline {chance:.4f} "
        f"(K={k} classes) — classifier is at or below random")


def test_confusion_matrix_is_square_and_consistent(smoke_results):
    n = smoke_results["n_classes"]
    cm = smoke_results["confusion_matrix"]
    assert len(cm) == n, f"CM rows {len(cm)} != n_classes {n}"
    assert all(len(row) == n for row in cm), "CM not square"
    # Each row sums to ≤ n_samples for that class
    counts = smoke_results["class_counts"]
    for i, cls in enumerate(smoke_results["classes"]):
        assert sum(cm[i]) <= counts[cls], (
            f"row {cls} sum {sum(cm[i])} > count {counts[cls]}")
    # All CM entries are non-negative integers
    for row in cm:
        for x in row:
            assert isinstance(x, int) and x >= 0, f"bad CM entry: {x}"


def test_macro_auc_ge_min_per_class_auc(smoke_results):
    aucs = smoke_results["per_class_auc"]
    macro = smoke_results["macro_auc_mean"]
    assert macro + 1e-9 >= min(aucs.values()), (
        f"macro {macro} < min per-class {min(aucs.values())}")
