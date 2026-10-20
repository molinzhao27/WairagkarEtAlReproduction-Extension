"""Config loading: configs/default.yaml deep-merged with an experiment YAML.

Paths in the ``paths`` section are stored relative to the repo root and are
resolved to absolute ``Path`` objects at load time, so code never hardcodes
locations.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

PathLike = Union[str, Path]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Return a new dict: ``override`` merged into ``base`` recursively."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: PathLike) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(p: PathLike) -> Path:
    """Resolve a repo-relative path (absolute paths are returned unchanged)."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(
    experiment: Optional[PathLike] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Load default.yaml, merge an optional experiment YAML and overrides.

    The ``paths`` section is resolved to absolute Paths.
    """
    cfg = load_yaml(DEFAULT_CONFIG)
    if experiment is not None:
        cfg = deep_merge(cfg, load_yaml(resolve_path(experiment)))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["paths"] = {k: resolve_path(v) for k, v in cfg.get("paths", {}).items()}
    return cfg
