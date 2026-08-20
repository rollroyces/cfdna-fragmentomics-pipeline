"""Tests for fetch_finaledb — the data-ingestion layer.

Targets:
  - frag_file_size → HEAD request contract
  - download_frag → size guard for deep-WGS (the bug that caused the
    earlier 9.7GB disk-fill incident: when HEAD returns 0, the guard
    was bypassed)

These tests don't touch the network — they monkeypatch urllib.
"""
import os
import sys
import urllib.error
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import fetch_finaledb  # noqa: E402


def test_frag_file_size_returns_int_from_content_length():
    """HEAD with Content-Length → that byte count."""
    fake = MagicMock()
    fake.__enter__.return_value = fake
    fake.__exit__.return_value = None
    fake.headers = {"Content-Length": "170000000"}
    with patch.object(fetch_finaledb.urllib.request, "urlopen", return_value=fake):
        size = fetch_finaledb.frag_file_size(seqrun_id=123, max_mb=500)
    assert size == 170_000_000


def test_frag_file_size_returns_zero_on_404():
    """HEAD 404 → 0 (caller must handle)."""
    with patch.object(
        fetch_finaledb.urllib.request, "urlopen",
        side_effect=urllib.error.HTTPError("url", 404, "Not Found", {}, None),
    ):
        size = fetch_finaledb.frag_file_size(seqrun_id=999)
    assert size == 0


def test_frag_file_size_returns_zero_on_network_error():
    """HEAD timeout → 0."""
    with patch.object(
        fetch_finaledb.urllib.request, "urlopen",
        side_effect=urllib.error.URLError("timeout"),
    ):
        size = fetch_finaledb.frag_file_size(seqrun_id=999)
    assert size == 0


def test_download_frag_rejects_oversize_via_head():
    """If HEAD says 2 GB and max_mb=500, download must be skipped."""
    fake_head = MagicMock()
    fake_head.__enter__.return_value = fake_head
    fake_head.__exit__.return_value = None
    fake_head.headers = {"Content-Length": str(2 * 1024 * 1024 * 1024)}
    with patch.object(fetch_finaledb.urllib.request, "urlopen", return_value=fake_head), \
         patch.object(fetch_finaledb, "shutil") as mock_shutil:
        result = fetch_finaledb.download_frag(seqrun_id=123, out_path="/tmp/x.bgz",
                                              max_mb=500)
    assert result is False, "oversize must be rejected (would have caused 9.7GB runaway)"
    assert not mock_shutil.copyfileobj.called, "no download should have started"


def test_download_frag_rejects_when_head_returns_zero_REGRESSION():
    """REGRESSION: if HEAD returns 0 (timeout/404), the size guard used to
    be bypassed (it tested `0 < size > max_mb`, which is False when size=0,
    so the guard did NOT fire). Then the download proceeded with no
    limit, causing the 9.7GB disk-fill incident.

    The fixed behavior must reject the download when HEAD fails — i.e.
    we cannot know the size, so we err on the side of caution.

    This test will fail on the buggy version. Fix by changing the guard
    to `if size == 0 or size > max_mb * 1024 * 1024:`.
    """
    with patch.object(
        fetch_finaledb.urllib.request, "urlopen",
        side_effect=urllib.error.URLError("timeout"),
    ):
        result = fetch_finaledb.download_frag(seqrun_id=123, out_path="/tmp/x.bgz",
                                              max_mb=500)
    assert result is False, (
        "When HEAD fails (size=0), download MUST be skipped — otherwise "
        "an oversize deep-WGS file can fill the disk. This is the fix "
        "for the 9.7GB runaway bug.")


def test_download_frag_streaming_cap_aborts_oversize(tmp_path):
    """Defense-in-depth: even if the HEAD size lies (e.g., a 2 GB file
    advertises 200 MB), the streaming byte-count cap MUST abort the
    download before max_mb * 1.1 is reached."""
    # Fake HEAD says 100 MB (lies)
    fake_head = MagicMock()
    fake_head.__enter__.return_value = fake_head
    fake_head.__exit__.return_value = None
    fake_head.headers = {"Content-Length": str(100 * 1024 * 1024)}
    # Fake GET streams 1MB chunks; would produce 3 GB if not aborted
    fake_get = MagicMock()
    fake_get.__enter__.return_value = fake_get
    fake_get.__exit__.return_value = None
    big_chunk = b"x" * (1024 * 1024)
    fake_get.read.side_effect = [big_chunk] * 4000 + [b""]  # many 1MB chunks
    with patch.object(fetch_finaledb.urllib.request, "urlopen",
                      side_effect=[fake_head, fake_get]):
        out = str(tmp_path / "frag.tsv.bgz")
        result = fetch_finaledb.download_frag(seqrun_id=123, out_path=out,
                                              max_mb=500)
    assert result is False, "streaming cap should have aborted"
    # Confirm the .part file was cleaned up
    assert not (tmp_path / "frag.tsv.bgz.part").exists(), \
        ".part file should be cleaned on abort"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))