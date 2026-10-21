"""Small helpers shared by scripts/syllable_*.py that must run BEFORE numpy, TensorFlow or PyTorch load
(thread caps are read once by those libraries): config loading, thread caps, niceness, provenance bits.
Standard library and PyYAML only."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from wvx.config import REPO_ROOT

PathLike = Union[str, Path]


def thread_overrides(n: Optional[int]) -> dict:
    """Config overrides for ``--threads N``: TF/BLAS/torch pools and the thread-pool workers."""
    if n is None:
        return {}
    n = int(n)
    return {"runtime": {"threads": {"intra_op": n, "omp": n, "inter_op": min(2, n)}},
            "benchmark": {"n_workers": n}, "alignment": {"n_workers": n}}


def start(config: PathLike, threads: Optional[int] = None, overrides: Optional[Mapping[str, Any]] = None) -> tuple[dict, dict]:
    """Load the config (recursive base experiments), export the thread caps, raise the niceness.

    Returns (cfg, the environment variables set). Call before importing numpy.
    """
    from wvx.config import deep_merge
    from wvx.experiment_config import apply_thread_env, load_experiment_config

    cfg = load_experiment_config(config, overrides=deep_merge(thread_overrides(threads), dict(overrides or {})))
    env_set = apply_thread_env(cfg)
    nice = cfg["runtime"].get("nice")
    if nice is not None and os.nice(0) < int(nice):     # absolute niceness, only ever raised; children inherit it
        os.nice(int(nice) - os.nice(0))
    return cfg, env_set


def runtime_record(cfg: Mapping[str, Any], env_set: Mapping[str, str]) -> dict:
    return {"threads": (cfg.get("runtime") or {}).get("threads"), "env": dict(env_set), "nice": os.nice(0),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def rel(p: PathLike) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def sha256(path: PathLike) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
