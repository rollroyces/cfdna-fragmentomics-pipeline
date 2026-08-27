"""Tests for the 8-channel evaluation script.

The 8-channel evaluation runs on the 98-sample subset where motif
features are available. Tests verify the load_channels function
returns correct shapes and that the alignment between 5-channel and
8-channel subsets is preserved.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from eval_8channel import fsd_vec, load_channels  # noqa


def test_fsd_vec_returns_196_elements():
    """FSD histogram has 196 bins (5bp each, 20-1000bp range)."""
    # Use a known sample
    p = "/Users/hermes/cfdna-fragmentomics-pipeline/data/features"
    if not os.path.exists(p):
        import pytest
        pytest.skip("data/features not on disk")
    # Pick first available sample
    for f in os.listdir(p):
        if f.endswith(".fsd.json"):
            sample = f.replace(".fsd.json", "")
            v = fsd_vec(sample)
            assert v is not None
            assert v.shape == (196,), f"Expected 196 bins, got {v.shape}"
            # Normalized
            assert abs(v.sum() - 1.0) < 0.01
            return
    raise RuntimeError("No .fsd.json files in data/features/")


def test_load_channels_returns_kept_sample_ids():
    """load_channels must return (X, y, st, kept_ids) so that
    downstream code can align 5-channel and 8-channel subsets."""
    p = "/Users/hermes/cfdna-fragmentomics-pipeline/data/features"
    if not os.path.exists(p):
        import pytest
        pytest.skip("data/features not on disk")
    labels = {"C309": 1, "C310": 0}
    studies = {"C309": "cristiano", "C310": "cristiano"}
    X, y, st, kept = load_channels(labels, studies)
    # Check return shape: 4 elements
    assert len((X, y, st, kept)) == 4
    # X rows should match `kept` length
    assert X.shape[0] == len(kept)
    assert y.shape[0] == len(kept)
    assert st.shape[0] == len(kept)
    # The kept list must be in the same order as X rows
    for i, sid in enumerate(kept):
        assert labels[sid] == y[i], (
            f"Row {i}: y={y[i]} but labels[{sid}]={labels[sid]}")
        assert studies[sid] == st[i]
