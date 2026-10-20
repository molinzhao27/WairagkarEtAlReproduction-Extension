"""Common in-memory format for one recording session, shared by every data source.

Neural features follow the decoder's input layout: one row per 10 ms bin, spike-band power for
all electrodes first, then threshold crossings, in the decoder's electrode order. Targets are
the decoder's normalized LPCNet features, one row per bin. Sizes come from the config
(``model`` section), not from constants here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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

    def save(self, path) -> None:
        """Write to one compressed .npz (trials concatenated, metadata as JSON)."""
        lengths = [t.neural.shape[0] for t in self.trials]
        has_target = [t.target is not None for t in self.trials]
        arrays = {"neural": np.concatenate([t.neural for t in self.trials])}
        if any(has_target):
            arrays["target"] = np.concatenate([t.target for t in self.trials if t.target is not None])
        header = {
            "name": self.name,
            "layer": self.layer,
            "trials": [
                {"length": n, "has_target": h, "text": t.text, "block": t.block, "meta": t.meta}
                for n, h, t in zip(lengths, has_target, self.trials)
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, header=np.array(json.dumps(header)), **arrays)

    @classmethod
    def load(cls, path) -> "SessionData":
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            neural, target = z["neural"], z["target"] if "target" in z else None
        trials, row, trow = [], 0, 0
        for h in header["trials"]:
            n = h["length"]
            tgt = None
            if h["has_target"]:
                tgt, trow = target[trow:trow + n], trow + n
            trials.append(Trial(neural=neural[row:row + n], target=tgt, text=h["text"],
                                block=h["block"], meta=h["meta"]))
            row += n
        return cls(name=header["name"], layer=header["layer"], trials=trials)
