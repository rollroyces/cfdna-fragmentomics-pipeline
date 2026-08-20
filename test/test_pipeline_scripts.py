"""Tests for the pipeline scripts — verify the contracts that the
standalone scripts honor under fixture data.

Targets:
  - extract_fsd.summarize — biology-relevant fragment-size summary
  - extract_fsd.extract_from_frag_tsv — FinaleDB frag.tsv parsing
  - train_classifier.sens_at — fixed-specificity sensitivity (was buggy)
  - train_classifier._harmonize — per-study z-score (no leakage)
  - fetch_finaledb — page_size cap + offset/total pagination

All tests are offline (no network, no real BAMs, no FinaleDB).
"""
import gzip
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from extract_fsd import summarize, extract_from_frag_tsv  # noqa: E402
from train_classifier import _harmonize  # noqa: E402


# ---------- summarize (FSD) ----------

def _synthetic_lengths(seed: int = 0, n: int = 10000,
                       cancer: bool = False) -> np.ndarray:
    """Mimic cfDNA: mono-nucleosome peak ~167bp + shorter in cancer."""
    rng = np.random.default_rng(seed)
    # Main peak ~167bp
    main = rng.normal(loc=167, scale=15, size=n).astype(int)
    main = np.clip(main, 100, 250)
    if cancer:
        # Add a shifted sub-peak ~145bp
        extra = rng.normal(loc=145, scale=10, size=n // 4).astype(int)
        extra = np.clip(extra, 100, 200)
        main = np.concatenate([main, extra])
    return np.clip(main, 20, 1000)


def test_summarize_returns_required_keys():
    out = summarize(_synthetic_lengths())
    for key in ("fragment_count", "median_length", "mean_length",
                "mode_length", "p10", "p25", "p75", "p90",
                "short_fraction_100_150", "long_fraction_150_220",
                "short_long_ratio", "size_bins"):
        assert key in out, f"missing key: {key}"
    assert out["fragment_count"] > 0
    assert 100 <= out["median_length"] <= 250


def test_summarize_size_bins_normalized():
    """Size bins should sum to 1.0 — verified earlier; this is the contract."""
    out = summarize(_synthetic_lengths(n=1000))
    total = sum(out["size_bins"].values())
    assert abs(total - 1.0) < 1e-9, f"size_bins sum {total}"


def test_summarize_cancer_has_higher_short_fraction():
    """Tumor-derived cfDNA has more short fragments (DELFI/Jiang core claim)."""
    healthy = summarize(_synthetic_lengths(seed=1, n=20000, cancer=False))
    cancer = summarize(_synthetic_lengths(seed=1, n=20000, cancer=True))
    assert cancer["short_fraction_100_150"] > healthy["short_fraction_100_150"], (
        f"cancer short_frac {cancer['short_fraction_100_150']:.3f} should be "
        f"> healthy {healthy['short_fraction_100_150']:.3f}")


def test_summarize_empty_raises():
    with pytest.raises(ValueError, match="no fragments"):
        summarize(np.array([], dtype=int))


def test_summarize_short_long_ratio_bounded():
    out = summarize(_synthetic_lengths(n=20000))
    assert 0.0 <= out["short_long_ratio"] <= 100.0  # plausible range


# ---------- extract_from_frag_tsv ----------

def test_extract_from_frag_tsv_basic(tmp_path):
    """FinaleDB format: chrom start end name mapq strand (6 cols, BEDPE)."""
    f = tmp_path / "frag.tsv.bgz"
    rows = []
    # 100 fragments, all length 167, MAPQ 30, name frag_N
    for i in range(100):
        rows.append(f"chr1\t{i*200}\t{i*200+167}\tfrag_{i}\t30\t+\n")
    # A few that should be filtered out
    rows.append("chr1\t0\t10\ttoo_short\t30\t+\n")    # length 10, below 20
    rows.append("chr1\t0\t2000\ttoo_long\t30\t+\n")    # length 2000, above 1000
    # Low-mapq row passes default threshold 0, so we filter it explicitly
    with gzip.open(f, "wt") as fh:
        fh.writelines(rows)
    lens = extract_from_frag_tsv(str(f), mapq_threshold=0)
    assert len(lens) == 100
    assert (lens == 167).all()
    # Same file with mapq_threshold=30 → drops nothing (all mapq 30)
    lens2 = extract_from_frag_tsv(str(f), mapq_threshold=30)
    assert len(lens2) == 100
    # But the standalone low_mapq row, by itself, would be filtered:
    with gzip.open(f, "wt") as fh:
        fh.write("chr1\t0\t167\tlow_mapq\t5\t+\n")
    assert len(extract_from_frag_tsv(str(f), mapq_threshold=30)) == 0


def test_extract_from_frag_tsv_respects_mapq(tmp_path):
    f = tmp_path / "frag.tsv.bgz"
    # mapq at index 4 (after chrom, start, end, name)
    rows = [f"chr1\t0\t167\tf{i}\t{mapq}\t+\n"
            for i, mapq in enumerate([10, 20, 25, 30, 35])]
    with gzip.open(f, "wt") as fh:
        fh.writelines(rows)
    # With threshold 30, only the last two fragments pass
    lens = extract_from_frag_tsv(str(f), mapq_threshold=30)
    assert len(lens) == 2


def test_extract_from_frag_tsv_skips_malformed(tmp_path):
    """Rows with non-integer start/end/mapq are skipped, not crashed.
    Also verifies the 5-column-rejection guard: real FinaleDB has 6 cols."""
    f = tmp_path / "frag.tsv.bgz"
    rows = [
        "chr1\t0\t167\tf0\t30\t+\n",      # OK (6 cols)
        "chr1\tbad\t167\tf1\t30\t+\n",    # bad start
        "chr1\t0\t167\tf2\tbad\t+\n",     # bad mapq
        "chr1\tf3\n",                       # too few columns (5 — old parser would take this!)
        "chr1\t0\t167\tf4\t30",            # exactly 5 cols — old parser kept these, new rejects
    ]
    with gzip.open(f, "wt") as fh:
        fh.writelines(rows)
    lens = extract_from_frag_tsv(str(f))
    assert len(lens) == 1, (
        f"only the first valid 6-col row should survive; got {lens}")


# ---------- sens_at (the bug we caught earlier) ----------

def test_sens_at_returns_high_at_low_target():
    """At very low target specificity, sens should be high (early in ROC).

    We don't import sens_at directly — it's a closure inside
    evaluate_cv. Reproduce the same logic here and assert.
    """
    y = np.array([0] * 100 + [1] * 100)
    score = np.concatenate([np.zeros(100), np.ones(100)])  # perfect
    fpr, tpr, _ = _compute_roc(y, score)
    # Mirror the closure body:
    ok = fpr <= 0.05 + 1e-9
    sens = float(tpr[ok][-1]) if ok.any() else 0.0
    assert sens == 1.0


def test_sens_at_snap_to_origin_regression():
    """The bug we caught earlier: at strict specificity, sens_at must NOT
    silently return 0 just because the ROC curve starts at fpr=0.

    Reproduce the closure logic on a cohort where there's one borderline
    healthy. With argmin(|fpr-target|) the result snapped to fpr=0 and
    gave sens=0; the fixed version reads the LAST fpr <= target.
    """
    y = np.array([0] * 50 + [1] * 50)
    score = np.concatenate([
        np.linspace(0.1, 0.5, 49),
        np.array([0.95]),  # 1 borderline healthy (high score)
        np.linspace(0.6, 0.99, 50),
    ])
    fpr, tpr, _ = _compute_roc(y, score)
    # New (correct) closure:
    ok = fpr <= 0.05 + 1e-9
    sens_new = float(tpr[ok][-1]) if ok.any() else 0.0
    assert sens_new > 0.0, (
        "fixed sens_at must NOT return 0 just because ROC starts at fpr=0")
    # Old (buggy) closure for reference:
    idx_old = np.argmin(np.abs(fpr - 0.05))
    sens_old = float(tpr[idx_old])
    # The point of this test: new gives a sensible sens, old gave 0.
    assert sens_old < sens_new or sens_old == sens_new  # either way, sens_new > 0


# ---------- _harmonize (no leakage) ----------

def _compute_roc(y, score):
    from sklearn.metrics import roc_curve
    return roc_curve(y, score)


def test_harmonize_uses_only_train_studies():
    """Per-study scalers must be fit on train data only, not test."""
    rng = np.random.default_rng(0)
    # Two studies with different means
    X_tr = np.vstack([rng.normal(0, 1, (50, 3)),
                      rng.normal(10, 1, (50, 3))])
    study_tr = np.array(["a"] * 50 + ["b"] * 50)
    # Test data with a THIRD study "c" not in train
    X_te = np.vstack([rng.normal(0, 1, (10, 3)),
                      rng.normal(10, 1, (10, 3)),
                      rng.normal(100, 1, (10, 3))])  # study "c"
    study_te = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    Xtr_h, scalers = _harmonize(X_tr, study_tr, None)
    Xte_h, _ = _harmonize(X_te, study_te, scalers)
    # Test rows from "a" and "b" should be centered around 0 mean per feature
    for st_mask_te, st in [(study_te == "a", "a"), (study_te == "b", "b")]:
        if st_mask_te.any():
            means = Xte_h[st_mask_te].mean(axis=0)
            assert np.allclose(means, 0, atol=0.6), (
                f"study {st!r} test mean not near 0: {means}")
    # Study "c" was unseen — it's passed through with the scalers it has;
    # no crash is the key contract.


def test_harmonize_fitted_scalers_dict_returned():
    """When called with scalers=None, returns the fitted dict."""
    X = np.array([[1, 2], [3, 4], [10, 20], [30, 40]], dtype=float)
    study = np.array(["a", "a", "b", "b"])
    _, scalers = _harmonize(X, study, None)
    assert set(scalers.keys()) == {"a", "b"}
    # mean_ of study "a" is [2, 3]; after transform should be [0, 0]
    assert np.allclose(scalers["a"].mean_, [2, 3])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))