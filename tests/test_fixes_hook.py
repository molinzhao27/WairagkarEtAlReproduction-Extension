""".githooks/pre-commit (2026-10-21 audit): case-insensitive blocked extensions and more binary types.
Each case runs the hook on a throwaway git repository under pytest's tmp_path, never on this repo."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from wvx.config import REPO_ROOT

HOOK = REPO_ROOT / ".githooks" / "pre-commit"
BASHES = [b for b in dict.fromkeys(["/bin/bash", shutil.which("bash")]) if b and shutil.which(b)]

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not BASHES or not HOOK.is_file(),
                                reason="git/bash or the hook not available")

BLOCKED = [
    # case variants of the old list
    "results/a.MAT", "results/a.WAV", "results/m.H5", "results/a.Npz", "results/a.Flac", "results/m.PTH",
    # audio
    "results/a.aiff", "results/a.aif", "results/a.ogg", "results/a.m4a", "results/a.opus",
    # pickles
    "results/d.pkl", "results/d.pickle", "results/d.joblib",
    # models and weights
    "results/m.safetensors", "results/model.bin", "results/m.onnx", "results/m.pt", "results/m.ckpt",
    "results/m.keras", "results/m.hdf5", "results/model.ckpt.index", "results/ckpt-3.data-00000-of-00001",
    # archives
    "results/a.tar", "results/a.tar.gz", "results/a.gz", "results/a.tgz", "results/a.7z", "results/a.bz2",
    "results/a.xz", "results/a.zip",
    # data tables
    "results/t.parquet", "results/t.feather", "results/a.npy", "results/a.npz", "a.mat",
    # path rules (unchanged), now case-insensitive too
    "data/raw/a.txt", "Data/raw/a.txt", "checkpoints/x.txt", "results/Checkpoints/x.txt", "results/x.tf/saved_model.pb",
]
ALLOWED = ["src/wvx/asr.py", "results/summary.json", "results/per_trial.csv", "notes/x.md", "configs/a.yaml",
           "data/MANIFEST.md", "data/README.md", "results/binder.md", "src/wvx/bin.py", "results/a.binx",
           "results/matrix.txt", "results/notes.mat.md", "scripts/remote/sync.sh", ".githooks/pre-commit"]


def _hook_rejects(tmp_path, rel, content=b"x", bash="bash"):
    repo = tmp_path / "r"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    f = repo / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content)
    subprocess.run(["git", "-C", str(repo), "add", "-f", rel], check=True)
    r = subprocess.run([bash, str(HOOK)], cwd=repo, capture_output=True)
    return r.returncode != 0


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("rel", BLOCKED)
def test_blocked(tmp_path, rel, bash):
    assert _hook_rejects(tmp_path, rel, bash=bash)


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("rel", ALLOWED)
def test_allowed(tmp_path, rel, bash):
    assert not _hook_rejects(tmp_path, rel, bash=bash)


def test_size_cap_unchanged(tmp_path):
    cap = 5 * 1024 * 1024
    assert _hook_rejects(tmp_path / "big", "results/big.json", b"0" * (cap + 1))
    assert not _hook_rejects(tmp_path / "ok", "results/ok.json", b"0" * cap)


def test_home_paths_still_blocked(tmp_path):
    home = "/" + "Users/someone/project"   # split so this file does not trip the hook itself
    assert _hook_rejects(tmp_path, "notes/x.md", f"see {home}/file\n".encode())
