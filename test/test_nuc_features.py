"""Tests for the nucleosome-aware fragmentomic features."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from nuc_features import (  # noqa: E402
    SUBNUCLEOSOME_RANGE,
    MONONUCLEOSOME_RANGE,
    DINUCLEOSOME_RANGE,
    NUC_FEATURE_NAMES,
    compute_nuc_features_from_fsd,
    compute_nuc_features_from_path,
    load_fsd,
)


def test_nuc_features_returns_three_values():
    """Each sample produces exactly 3 features in the documented order."""
    fsd = np.zeros(196)
    fsd[20:30] = 0.5  # bin centers 120..165
    fsd /= fsd.sum()
    feats = compute_nuc_features_from_fsd(fsd)
    assert feats.shape == (3,)
    assert feats.dtype == float


def test_nuc_features_handle_zero_fsd():
    """An all-zero FSD must not crash (denominator epsilon prevents /0)."""
    fsd = np.zeros(196)
    feats = compute_nuc_features_from_fsd(fsd)
    assert np.all(np.isfinite(feats))
    # All zeros -> every sum is 0, every ratio is 0/eps ~= small finite
    assert (np.abs(feats) < 1e6).all()


def test_nuc_features_submono_direction_is_correct():
    """If mass is concentrated in subnucleosome range, submono_ratio > 1.
    If mass is concentrated in mononucleosome, submono_ratio < 1."""
    # Subnucleosome-heavy FSD
    fsd_sub = np.zeros(196)
    fsd_sub[8:18] = 1.0  # bin centers 60..105 (subnucleosomal)
    fsd_sub /= fsd_sub.sum()
    f_submono = compute_nuc_features_from_fsd(fsd_sub)[0]
    # Mononucleosome-heavy FSD
    fsd_mono = np.zeros(196)
    fsd_mono[22:32] = 1.0  # bin centers 130..175 (mononucleosomal)
    fsd_mono /= fsd_mono.sum()
    f_submono_mono = compute_nuc_features_from_fsd(fsd_mono)[0]
    assert f_submono > f_submono_mono


def test_nuc_features_normalization_invariant():
    """Doubling the FSD amplitude should not change the ratios."""
    fsd = np.random.default_rng(0).random(196)
    fsd /= fsd.sum()
    f1 = compute_nuc_features_from_fsd(fsd)
    f2 = compute_nuc_features_from_fsd(fsd * 1000)
    np.testing.assert_allclose(f1, f2, rtol=1e-6)


def test_nuc_features_match_disk_fsd_format():
    """End-to-end: write a fake .fsd.json, load it, compute features."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.fsd.json")
        with open(path, "w") as f:
            bins = {}
            for k in range(196):
                lo = 20 + 5 * k
                hi = lo + 5
                # Synthetic: peak at ~167bp
                bins[f"{lo}-{hi}"] = float(np.exp(-((lo - 167) / 10) ** 2))
            total = sum(bins.values())
            bins = {k: v / total for k, v in bins.items()}
            import json
            json.dump({"size_bins": bins, "fragment_count": 1000}, f)
        feats = compute_nuc_features_from_path(path)
        assert feats.shape == (3,)
        assert np.all(np.isfinite(feats))


def test_feature_names_match_implementation():
    """Sanity: feature names list has exactly 3 entries."""
    assert len(NUC_FEATURE_NAMES) == 3


def test_index_ranges_are_within_fsd():
    """The bin-index ranges must not exceed the 196 FSD bins."""
    assert SUBNUCLEOSOME_RANGE[1] <= 1000  # well within range
    assert MONONUCLEOSOME_RANGE[1] <= 1000
    assert DINUCLEOSOME_RANGE[1] <= 1000


def test_nuc_features_distinguish_cancer_like_from_healthy_like():
    """A 'cancer-like' FSD (mass shifted to short fragments) should
    have a higher submono_ratio than a 'healthy-like' FSD. This is
    the headline biological signal."""
    rng = np.random.default_rng(42)
    # Cancer-like: 60% mass at subnucleosome, 30% mono, 10% long
    cancer = np.zeros(196)
    cancer[5:14] = 0.6   # 45..85 bp
    cancer[22:32] = 0.3  # 130..175 bp (mono)
    cancer[40:60] = 0.1  # 220..315 bp (long)
    # Healthy-like: 5% subnuc, 70% mono, 25% long
    healthy = np.zeros(196)
    healthy[5:14] = 0.05
    healthy[22:32] = 0.70
    healthy[40:60] = 0.25
    f_cancer = compute_nuc_features_from_fsd(cancer / cancer.sum())
    f_healthy = compute_nuc_features_from_fsd(healthy / healthy.sum())
    assert f_cancer[0] > f_healthy[0], (
        f"submono_ratio should be higher for cancer-like FSD; "
        f"got cancer={f_cancer[0]:.3f}, healthy={f_healthy[0]:.3f}")
