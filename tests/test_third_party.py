"""brain_to_voice must be byte-identical to the pinned commit (no unlogged local edits)."""

import importlib.util

import pytest

from wvx.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("fetch_third_party", REPO_ROOT / "scripts" / "fetch_third_party.py")
ftp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ftp)


def test_brain_to_voice_matches_pinned_checksums():
    dest = REPO_ROOT / "third_party" / "brain_to_voice"
    if not dest.exists():
        pytest.skip("not fetched; run scripts/fetch_third_party.py")
    expected = ftp.load_expected(REPO_ROOT / "third_party" / "brain_to_voice.sha256")
    assert ftp.verify(dest, expected) == []
