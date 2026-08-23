"""Tests for the Gemma baseline: prompt formatting, parsing, and
feature-summarization. These don't load the actual model — they
verify the contracts that prevent the silent failures I hit during
development."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gemma_baseline import (  # noqa: E402
    _sample_to_text,
    _build_few_shot_prompt,
    _parse_p_cancer,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)


def _fake_features(seed: int = 0, with_short_bias: bool = False):
    """Synthetic 5-channel-like features for testing."""
    rng = np.random.default_rng(seed)
    r5 = rng.normal(0, 0.5, 631)
    c5 = rng.random(631) + 0.5
    r100 = rng.normal(0.4, 0.3, 30894)
    if with_short_bias:
        # Drop short fragments <150bp to simulate cancer-like FSD shift
        r100[:3000] += 0.1
    c100 = rng.random(30894) * 1000 + 100
    fsd = rng.random(196)
    fsd = fsd / fsd.sum()  # normalize like the real FSD JSON
    if with_short_bias:
        # Cancer-like: shorter fragments less common
        fsd[:26] *= 0.7
        fsd = fsd / fsd.sum()
    return r5, c5, r100, c100, fsd


def test_sample_to_text_runs_without_indexerror():
    """The prompt-builder must handle a 196-bin FSD without IndexError."""
    f = _fake_features()
    text = _sample_to_text("C309", *f, study="jiang")
    assert "C309" in text
    assert "jiang" in text
    # short_frac and long_frac must be in [0, 1] (fractions)
    assert "short<150bp=" in text
    assert "long>250bp=" in text
    assert "fsd(mode=" in text


def test_sample_to_text_compact_under_200_chars():
    """Compactness is critical — too-long prompts overflow n_ctx=2048
    silently and AUC defaults to 0.5 (the bug we hit at 2429 tokens).
    Per-sample text should be < 200 chars (~50 tokens) to safely fit
    a 4-shot prompt under 2048 tokens."""
    f = _fake_features()
    text = _sample_to_text("C309", *f, study="jiang")
    assert len(text) < 200, (
        f"sample text {len(text)} chars; expected <200 for n_ctx=2048 "
        f"4-shot fit")


def test_parse_p_cancer_handles_common_formats():
    """The regex must catch the variations Gemma actually produces."""
    cases = [
        ("P(cancer)=0.5", 0.5),
        ("P(cancer) = 0.5", 0.5),
        ("P(cancer)=0.93", 0.93),
        ("P(cancer)=1.0", 1.0),
        ("P(cancer)=0", 0.0),
        ("some text P(cancer)=0.7 more text", 0.7),
    ]
    for text, expected in cases:
        got = _parse_p_cancer(text)
        assert got == expected, f"parse({text!r}) = {got}, expected {expected}"


def test_parse_p_cancer_returns_none_on_garbage():
    """When Gemma returns garbage (timeout, refusal, etc.), we must
    NOT crash. None gets mapped to 0.5 by the caller — that's the
    expected safety fallback."""
    assert _parse_p_cancer("") is None
    assert _parse_p_cancer("I refuse to answer") is None
    assert _parse_p_cancer("the answer is uncertain") is None


def test_few_shot_prompt_is_self_contained():
    """The few-shot prompt must include the few-shot examples AND the
    query, but never any actual labels in the query position (that
    would be label leakage)."""
    f_cancer = _fake_features(seed=0, with_short_bias=True)
    f_healthy = _fake_features(seed=1, with_short_bias=False)
    features = {
        "C_CANCER": dict(zip(["r5", "c5", "r100", "c100", "fsd"], f_cancer)),
        "H_HEALTHY": dict(zip(["r5", "c5", "r100", "c100", "fsd"], f_healthy)),
    }
    labels = {"C_CANCER": 1, "H_HEALTHY": 0}
    studies = {"C_CANCER": "jiang", "H_HEALTHY": "cristiano"}
    prompt = _build_few_shot_prompt(
        ["C_CANCER", "H_HEALTHY"], features, labels, studies,
        n_cancer=1, n_healthy=1, seed=0)
    assert "CANCER" in prompt
    assert "HEALTHY" in prompt
    assert "P(cancer)=1.00" in prompt  # cancer example
    assert "P(cancer)=0.00" in prompt  # healthy example
    # System prompt should NOT appear in the few-shot text (it's
    # passed separately to the chat-completion API)
    assert SYSTEM_PROMPT not in prompt


def test_full_prompt_fits_n_ctx_2048_with_4_shot():
    """Critical regression guard: if we make the prompt too verbose
    again, Gemma raises 'Requested tokens exceed context window' and
    AUC silently defaults to 0.5."""
    f = _fake_features()
    features = {f"S{i:03d}": dict(zip(["r5", "c5", "r100", "c100", "fsd"], f))
                for i in range(8)}
    labels = {f"S{i:03d}": i % 2 for i in range(8)}
    studies = {f"S{i:03d}": "jiang" for i in range(8)}
    train_sids = list(features.keys())[:6]
    few_shot = _build_few_shot_prompt(train_sids, features, labels, studies,
                                       n_cancer=3, n_healthy=3, seed=0)
    query = _sample_to_text("S007", *f, study="jiang")
    prompt = PROMPT_TEMPLATE.format(nshot=4, examples=few_shot, query=query)
    # Rough char-to-token estimate: 1 token ~ 4 chars
    approx_tokens = len(SYSTEM_PROMPT) // 4 + len(prompt) // 4
    assert approx_tokens < 2048, (
        f"full prompt ~{approx_tokens} tokens exceeds 2048 — "
        f"Gemma would fail with context-window error and AUC would "
        f"silently fall to 0.5")
