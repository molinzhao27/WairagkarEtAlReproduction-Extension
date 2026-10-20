"""Fetch third-party code that this repo does not redistribute, pinned to an exact commit.

    python scripts/fetch_third_party.py            # fetch if missing, then verify
    python scripts/fetch_third_party.py --verify   # verify only (detects local edits)

brain-to-voice-2025 is under an academic, non-commercial license that limits who may receive
copies, so it is cloned from the authors' public repository at the commit recorded in
configs/third_party.yaml (its .git is removed). Every file is then checked against
third_party/brain_to_voice.sha256: no file may be missing, extra or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "third_party.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_expected(sha_list: Path) -> dict[str, str]:
    expected = {}
    for line in sha_list.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            expected[rel] = digest
    return expected


def verify(dest: Path, expected: dict[str, str]) -> list[str]:
    """Return a list of problems (empty if dest matches expected exactly)."""
    if not dest.is_dir():
        return [f"missing directory {dest.relative_to(REPO_ROOT)}"]
    # Python bytecode caches appear when their modules are imported; they are not edits.
    present = {p.relative_to(dest).as_posix() for p in dest.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    problems = [f"missing: {r}" for r in sorted(expected.keys() - present)]
    problems += [f"extra: {r}" for r in sorted(present - expected.keys())]
    problems += [f"changed: {r}" for r in sorted(expected.keys() & present)
                 if sha256(dest / r) != expected[r]]
    return problems


def fetch(url: str, commit: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        clone = Path(tmp) / "clone"
        subprocess.run(["git", "clone", "--quiet", url, str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", commit], check=True)
        shutil.rmtree(clone / ".git")
        clone.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only verify the existing copy")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())["brain_to_voice"]
    dest = REPO_ROOT / cfg["dest"]
    expected = load_expected(REPO_ROOT / cfg["sha256_list"])

    if not dest.exists() and not args.verify:
        print(f"[fetch] {cfg['url']} @ {cfg['commit'][:7]} -> {cfg['dest']}")
        fetch(cfg["url"], cfg["commit"], dest)

    problems = verify(dest, expected)
    if problems:
        print(f"[verify] {cfg['dest']} does NOT match the pinned checksums:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"[verify] {cfg['dest']}: all {len(expected)} files match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
