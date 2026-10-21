"""wvx.provenance: code version from git, from the pushed version file, or unknown."""

import json
import subprocess

from wvx import provenance


def test_reads_pushed_version_file_without_git(tmp_path):
    (tmp_path / provenance.VERSION_FILE).write_text(json.dumps(
        {"commit": "abc123", "dirty": False, "diff_sha256": None, "pushed_at": "2026-10-21T00:00:00Z"}))
    v = provenance.code_version(tmp_path)
    assert v["source"] == "sync" and v["commit"] == "abc123" and v["dirty"] is False


def test_unknown_without_git_or_file(tmp_path):
    assert provenance.code_version(tmp_path) == {"source": None, "commit": None, "dirty": None, "diff_sha256": None}


def test_git_dirty_flag_and_diff_hash(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("1\n")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "x")
    clean = provenance.code_version(tmp_path)
    assert clean["source"] == "git" and not clean["dirty"] and clean["diff_sha256"] is None
    (tmp_path / "a.txt").write_text("2\n")
    d1 = provenance.code_version(tmp_path)
    (tmp_path / "a.txt").write_text("3\n")
    d2 = provenance.code_version(tmp_path)
    assert d1["dirty"] and d1["commit"] == clean["commit"] and d1["diff_sha256"] != d2["diff_sha256"]


def test_config_hash_is_order_independent():
    assert provenance.config_sha256({"a": 1, "b": {"c": 2}}) == provenance.config_sha256({"b": {"c": 2}, "a": 1})
    assert provenance.config_sha256({"a": 1}) != provenance.config_sha256({"a": 2})
