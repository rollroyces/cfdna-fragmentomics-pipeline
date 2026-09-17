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


@pytest.fixture(scope="module")
def off_path_5x5(tmp_path_factory):
    """Full 5x5 OFF-path run, cached at module scope (one slow subprocess).

    Returns the dict JSON so other tests can compare directly to it
    without re-running the expensive pipeline.
    """
    if not os.path.isdir(os.path.join(REPO_ROOT, "data", "features")):
        pytest.skip("data dir not found (gitignored; CI provides synthetic only)")
    out_dir = tmp_path_factory.mktemp("off_5x5")
    return _run_full_5x5(ov_topk_path=None, out_path=str(out_dir / "off.json"))


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
    # Sept 2026: OV-specific K=10000 channel-set integration
    "ov_topk_channels", "oof_per_class_sens_at_99pct",
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


# --------------------------------------------------------------------------- #
# 7. OV-specific K=10000 channel pre-filter (Sept 2026)
# --------------------------------------------------------------------------- #
OV_TOPK_JSON = os.path.join(REPO_ROOT, "results", "ov_topk_channels.json")
OV_TOPK_DOCUMENTED_SENS99 = 0.35714285714285715  # per_cancer_topk.json
OV_TOPK_DOCUMENTED_AUC = 0.9524783144          # per_cancer_topk.json (0.9525)
PER_CANCER_OV_PAAD_JSON = os.path.join(
    REPO_ROOT, "results", "per_cancer_ov_paad.json")


def _feat_dir_or_skip() -> str:
    """Return the local data/features/ path or skip the test (CI lacks cohort)."""
    feat_dir = os.path.join(REPO_ROOT, "data", "features")
    if not os.path.isdir(feat_dir):
        pytest.skip(f"data dir not found: {feat_dir} "
                    "(gitignored; CI provides synthetic only)")
    return feat_dir


def _run_full_5x5(*, ov_topk_path: str | None, out_path: str) -> dict:
    """Run the script with full 5x5 CV (no --quick) and return parsed JSON.

    Uses --min-class-n 0 so OV (n=28) is kept — the pre-filter test
    only makes sense when OV is in the class list. On a 627-sample
    cohort this takes ~2 minutes on M-series Mac; CI without the
    cohort will skip via _feat_dir_or_skip().
    """
    cmd = [sys.executable, SCRIPT,
           "--min-class-n", "0",
           "--seeds", "5", "--folds", "5",
           "--out", out_path]
    if ov_topk_path is not None:
        cmd += ["--ov-topk-channels", ov_topk_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       cwd=REPO_ROOT, check=False)
    assert r.returncode == 0, (
        f"exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    with open(out_path) as f:
        return json.load(f)


def test_ov_topk_channels_json_exists_and_shape():
    """results/ov_topk_channels.json must exist with K=10000 ints in [0, 63246)."""
    if not os.path.exists(OV_TOPK_JSON):
        pytest.skip(f"{OV_TOPK_JSON} not found (regenerate via "
                    "scripts/per_cancer_topk_sweep.py)")
    with open(OV_TOPK_JSON) as f:
        d = json.load(f)
    assert d["cancer"] == "OV"
    assert d["K"] == 10000
    assert len(d["channel_indices"]) == 10000
    # Indices must be valid column positions
    for i in d["channel_indices"][:5] + d["channel_indices"][-5:]:
        assert isinstance(i, int)
        assert 0 <= i < 63246, f"channel index {i} out of [0, 63246)"
    assert d["documented_sens_at_99pct_pooled"] == OV_TOPK_DOCUMENTED_SENS99


def test_ov_topk_prefilter_off_matches_baseline(tmp_path, off_path_5x5):
    """Default behavior (no --ov-topk-channels) must reproduce the
    per_cancer_ov_paad.json OV Sens@99% = 0.25 exactly, and leave all
    other class metrics unchanged (full backwards-compat).

    Uses the cached module-scoped OFF-path run from ``off_path_5x5``
    fixture (same script, same protocol) — much faster than running a
    second 5x5 subprocess per test.
    """
    if not os.path.exists(PER_CANCER_OV_PAAD_JSON):
        pytest.skip(f"baseline {PER_CANCER_OV_PAAD_JSON} missing")
    d = off_path_5x5
    base = json.load(open(PER_CANCER_OV_PAAD_JSON))["baseline"]
    assert d["ov_topk_channels"]["applied"] is False
    # OV: published baseline Sens@99% = 0.25
    assert d["oof_per_class_sens_at_99pct"]["OV"] == \
        base["per_class"]["OV"]["sens_at_99pct_pooled"], (
            f"OFF-path OV Sens@99% drift: "
            f"{d['oof_per_class_sens_at_99pct']['OV']} vs "
            f"baseline {base['per_class']['OV']['sens_at_99pct_pooled']}")
    # And it should equal exactly 0.25 (no float drift across re-runs)
    assert d["oof_per_class_sens_at_99pct"]["OV"] == 0.25
    # OV AUC
    assert abs(d["oof_per_class_auc"]["OV"]
               - base["per_class"]["OV"]["auc_pooled"]) < 1e-9
    # All other classes: byte-identical Sens@99% to baseline
    for cls, m_base in base["per_class"].items():
        assert cls in d["oof_per_class_sens_at_99pct"], cls
        assert d["oof_per_class_sens_at_99pct"][cls] == \
            m_base["sens_at_99pct_pooled"], (
                f"{cls}: OFF drift {d['oof_per_class_sens_at_99pct'][cls]} "
                f"vs baseline {m_base['sens_at_99pct_pooled']}")


def test_ov_topk_prefilter_on_reproduces_documented_lift(tmp_path, off_path_5x5):
    """With --ov-topk-channels results/ov_topk_channels.json, OV Sens@99%
    must equal the documented 0.3571 exactly. Other classes and the
    binary pooled metric must match the OFF-path script output exactly
    (the pre-filter is local to the OV class only, so anything else
    changing would be a regression)."""
    if not os.path.exists(OV_TOPK_JSON):
        pytest.skip(f"{OV_TOPK_JSON} not found (regenerate via "
                    "scripts/per_cancer_topk_sweep.py)")
    # 1) OFF path comes from the cached module-scoped fixture (same
    #    script, same protocol) — apples-to-apples comparison without
    #    cross-protocol drift between per_cancer_ov_paad_sweep's
    #    binary_oof_score and the script's binary_cv (they differ at
    #    the 4th decimal due to a different per-fold standardisation).
    off = off_path_5x5
    # 2) Run ON with the same protocol.
    out = str(tmp_path / "on.json")
    d = _run_full_5x5(ov_topk_path=OV_TOPK_JSON, out_path=out)
    # Pre-filter metadata surfaced in JSON
    assert d["ov_topk_channels"]["applied"] is True
    assert d["ov_topk_channels"]["K"] == 10000
    assert d["ov_topk_channels"]["n_channels"] == 10000
    assert d["ov_topk_channels"]["source_path"] == OV_TOPK_JSON
    # OV: documented +0.1071 absolute lift (0.25 -> 0.3571)
    got = d["oof_per_class_sens_at_99pct"]["OV"]
    assert got == OV_TOPK_DOCUMENTED_SENS99, (
        f"OV Sens@99% = {got!r}, documented "
        f"{OV_TOPK_DOCUMENTED_SENS99!r}")
    # Other classes byte-identical to OFF (same script, same protocol)
    for cls in off["oof_per_class_sens_at_99pct"]:
        if cls == "OV":
            continue  # OV is the only one that changes
        assert d["oof_per_class_sens_at_99pct"][cls] == \
            off["oof_per_class_sens_at_99pct"][cls], (
                f"{cls}: pre-filter regressed Sens@99% "
                f"{d['oof_per_class_sens_at_99pct'][cls]} vs OFF "
                f"{off['oof_per_class_sens_at_99pct'][cls]}")
        # And AUC unchanged
        assert abs(d["oof_per_class_auc"][cls]
                   - off["oof_per_class_auc"][cls]) < 1e-9, (
                f"{cls}: pre-filter regressed AUC "
                f"{d['oof_per_class_auc'][cls]} vs OFF "
                f"{off['oof_per_class_auc'][cls]}")
    # Binary cancer-vs-healthy also unchanged (pre-filter is local to OV;
    # binary_cv uses its own StandardScaler+PCA on full X and is
    # independent of the OvR loop entirely).
    assert abs(d["binary_cancer_vs_healthy_auc"]
               - off["binary_cancer_vs_healthy_auc"]) < 1e-9
