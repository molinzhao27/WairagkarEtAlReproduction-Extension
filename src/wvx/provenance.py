"""Which code produced a result: the git commit (and whether the tree was dirty) for every result file.

On the Mac the repo is a git checkout. On the GPU host it is an rsync copy with no ``.git``, so
``scripts/remote/sync.sh push`` writes ``.wvx_version.json`` (commit, dirty flag, hash of the uncommitted
changes) next to the code, and :func:`code_version` reads that file when git is unavailable.

    from wvx.provenance import code_version, config_sha256
    summary["code_version"] = code_version()      # {"source", "commit", "dirty", "diff_sha256", ...}
    summary["config_sha256"] = config_sha256(cfg)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

from wvx.config import REPO_ROOT

VERSION_FILE = ".wvx_version.json"  # written by scripts/remote/sync.sh push (gitignored)


def _git(root: Path, *args: str) -> Optional[bytes]:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _from_git(root: Path) -> Optional[dict]:
    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        return None
    status = _git(root, "status", "--porcelain") or b""
    out = {"source": "git", "commit": head.decode().strip(), "dirty": bool(status.strip()), "diff_sha256": None}
    if out["dirty"]:
        h = hashlib.sha256(_git(root, "diff", "HEAD") or b"")
        for name in sorted((_git(root, "ls-files", "--others", "--exclude-standard", "-z") or b"").split(b"\0")):
            p = root / name.decode()
            if name and p.is_file():
                h.update(name + b"\0" + p.read_bytes())
        out["diff_sha256"] = h.hexdigest()
    return out


def code_version(root: Path = REPO_ROOT) -> dict:
    """Git state of ``root``; else the pushed ``.wvx_version.json``; else ``source: None`` (unknown)."""
    root = Path(root)
    got = _from_git(root)
    if got is not None:
        return got
    f = root / VERSION_FILE
    if f.is_file():
        return {"source": "sync", **json.loads(f.read_text())}
    return {"source": None, "commit": None, "dirty": None, "diff_sha256": None}


def config_sha256(cfg: Mapping[str, Any]) -> str:
    """SHA-256 of the resolved config (keys sorted, non-JSON values stringified)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
