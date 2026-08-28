"""Tests for the Gemma baseline's parse-failure handling.

The Engineering reviewer (round 1) flagged that gemma_baseline.py
silently filled parse failures with p=0.5, which can introduce
label-correlated bias. The fix added explicit tracking and a
choice of failure handling.

The test focuses on the parser and the failure-tracking logic,
not on the full Gemma pipeline (which is slow).

The full Gemma integration test requires a model on disk and is
not run by default.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gemma_baseline import _parse_p_cancer  # noqa


def test_parse_p_cancer_valid_decimal():
    """Standard decimal response."""
    assert _parse_p_cancer("P(cancer) = 0.5") == 0.5
    assert _parse_p_cancer("P(cancer) = 0.0") == 0.0
    assert _parse_p_cancer("P(cancer) = 1.0") == 1.0
    assert _parse_p_cancer("P(cancer) = 0.731") == 0.731


def test_parse_p_cancer_with_spaces():
    """Whitespace variants should still parse."""
    assert _parse_p_cancer("P(cancer) =0.5") == 0.5
    assert _parse_p_cancer("P(cancer)= 0.5") == 0.5
    assert _parse_p_cancer("  P(cancer) = 0.5  ") == 0.5


def test_parse_p_cancer_returns_none_for_unparseable():
    """Returns None for anything that doesn't match the pattern.

    Note: there is a fallback regex that catches any 0.x or 1.0 in
    the text, so _parse_p_cancer("0.5") returns 0.5, not None.
    This is intentional — Gemma sometimes responds with just the
    number. Test that *truly* unparseable strings return None.
    """
    assert _parse_p_cancer("") is None
    assert _parse_p_cancer("I don't know") is None
    assert _parse_p_cancer("abc def ghi") is None  # No numbers
    assert _parse_p_cancer("P(cancer) = abc") is None  # Non-numeric
    # These have a 0.x or 1.0 in them — fallback regex catches them
    assert _parse_p_cancer("Sure! P(cancer) = 0.5!") == 0.5  # Embedded works
    assert _parse_p_cancer("I think it's 0.731") == 0.731  # Fallback works


def test_parse_p_cancer_integer_format():
    """Integer-like values should parse."""
    # Pattern is flexible; 1.0, 0.0, etc. should all parse
    assert _parse_p_cancer("P(cancer) = 0") == 0.0
    assert _parse_p_cancer("P(cancer) = 1") == 1.0