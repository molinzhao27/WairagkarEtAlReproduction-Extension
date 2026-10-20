"""Common in-memory format for one recording session, shared by every data source.

Neural features follow the decoder's input layout: one row per 10 ms bin, spike-band power for
all electrodes first, then threshold crossings, in the decoder's electrode order. Targets are
the decoder's normalized LPCNet features, one row per bin. Sizes come from the config
(``model`` section), not from constants here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trial:
    neural: np.ndarray                    # (T, n_features) float32
    target: Optional[np.ndarray] = None   # (T, n_targets) float32, or None if not generated yet
    text: str = ""                        # cue sentence
    block: Optional[int] = None
    meta: dict = field(default_factory=dict)


@dataclass
class SessionData:
    name: str        # e.g. "t15_day195" or "t15.2024.03.03_attempted"
    layer: int       # index of this session's input layer in the decoder
    trials: list[Trial]

    def validate(self, n_features: int, n_targets: int) -> None:
        """Raise ValueError if any trial has the wrong shape or dtype."""
        for i, t in enumerate(self.trials):
            where = f"{self.name} trial {i}"
            if t.neural.ndim != 2 or t.neural.shape[1] != n_features:
                raise ValueError(f"{where}: neural shape {t.neural.shape}, want (T, {n_features})")
            if t.neural.dtype != np.float32:
                raise ValueError(f"{where}: neural dtype {t.neural.dtype}, want float32")
            if t.target is not None:
                if t.target.shape != (t.neural.shape[0], n_targets):
                    raise ValueError(
                        f"{where}: target shape {t.target.shape}, want ({t.neural.shape[0]}, {n_targets})"
                    )
                if t.target.dtype != np.float32:
                    raise ValueError(f"{where}: target dtype {t.target.dtype}, want float32")
