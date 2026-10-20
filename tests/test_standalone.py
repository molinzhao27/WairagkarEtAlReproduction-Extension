"""The repo must be standalone: no tracked file may point at a user's home directory."""

import re
import subprocess

import pytest

from wvx.config import REPO_ROOT

# Built from pieces so this file doesn't match itself.
HOME_PATH_RE = re.compile("/" + r"Users/[A-Za-z0-9._-]+/|~/" + "Documents")
SKIP = {".githooks/pre-commit"}


def _tracked_files():
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.decode().split("\0") if f and f not in SKIP]


def test_no_absolute_home_paths_in_tracked_files():
    offenders = []
    for rel in _tracked_files():
        path = REPO_ROOT / rel
        try:
            text = path.read_text(errors="strict")
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue  # binary or vanished
        for i, line in enumerate(text.splitlines(), 1):
            if HOME_PATH_RE.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()[:120]}")
    assert not offenders, "absolute home paths found:\n" + "\n".join(offenders)
