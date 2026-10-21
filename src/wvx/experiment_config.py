"""Experiment configs with several base experiments, and CPU thread caps.

Imports neither numpy nor TensorFlow, so a script can read its config and cap the thread pools
(``runtime.threads``) BEFORE either library starts its pools:

    cfg = load_experiment_config(path)
    apply_thread_env(cfg)          # env vars: OMP/MKL/OpenBLAS/numba, TF_NUM_*_THREADS
    ... import numpy / tensorflow (via wvx.b2v) ...
    apply_tf_threads(cfg)          # tf.config.threading, before TensorFlow runs any op
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Optional

from wvx.config import deep_merge, load_config, load_yaml, resolve_path

_OMP_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
_TF_VARS = {"intra_op": "TF_NUM_INTRAOP_THREADS", "inter_op": "TF_NUM_INTEROP_THREADS"}


def resolve_chain(path) -> dict:
    """An experiment YAML merged over its ``base_experiment`` (a path or a list, resolved recursively)."""
    exp = load_yaml(resolve_path(path))
    bases = exp.pop("base_experiment", None) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, resolve_chain(b))
    return deep_merge(merged, exp)


def load_experiment_config(path, overrides: Optional[Mapping] = None) -> dict:
    """configs/default.yaml <- base experiments (in order, recursively) <- ``path`` <- ``overrides``."""
    merged = resolve_chain(path)
    if overrides:
        merged = deep_merge(merged, overrides)
    return load_config(None, overrides=merged)


def _threads(cfg: Mapping[str, Any]) -> dict:
    return dict((cfg.get("runtime") or {}).get("threads") or {})


def apply_thread_env(cfg: Mapping[str, Any]) -> dict:
    """Export ``runtime.threads`` as environment variables. Raises if TensorFlow is already imported."""
    if "tensorflow" in sys.modules:
        raise RuntimeError("apply_thread_env must run before TensorFlow is imported")
    th = _threads(cfg)
    set_vars = {}
    if th.get("omp") is not None:
        if "numpy" in sys.modules:
            raise RuntimeError("apply_thread_env must run before numpy is imported (BLAS reads its env once)")
        for v in _OMP_VARS:
            os.environ[v] = set_vars[v] = str(int(th["omp"]))
    for key, var in _TF_VARS.items():
        if th.get(key) is not None:
            os.environ[var] = set_vars[var] = str(int(th[key]))
    return set_vars


def apply_tf_threads(cfg: Mapping[str, Any]) -> dict:
    """Set TensorFlow's intra/inter-op pools from ``runtime.threads``; returns what TensorFlow reports."""
    import tensorflow as tf

    th = _threads(cfg)
    try:
        if th.get("intra_op") is not None:
            tf.config.threading.set_intra_op_parallelism_threads(int(th["intra_op"]))
        if th.get("inter_op") is not None:
            tf.config.threading.set_inter_op_parallelism_threads(int(th["inter_op"]))
    except RuntimeError as e:
        raise RuntimeError("runtime.threads must be applied before TensorFlow runs any op") from e
    return {"tf_intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
            "tf_inter_op": tf.config.threading.get_inter_op_parallelism_threads()}
