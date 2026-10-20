"""Seed every RNG we use. Call ``set_seed(cfg["seed"])`` at the top of each script."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_ops: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()
