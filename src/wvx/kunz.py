"""Kunz et al. 2025 sentence files -> SessionData in the decoder's input layout.

Facts used here are documented in notes/DATA_FORMATS.md (Kunz section):
- Level-5 .mat files; continuous ``spikePow`` (float64, µV²) and ``binnedTX`` (uint8 counts), T x 256 each,
  same column = same electrode, 10 ms bins.
- ``goTrialEpochs`` / ``delayTrialEpochs`` are N x 2, 1-based, inclusive. ``blockNum`` labels every bin.
- ``sentenceFeatures`` holds threshold crossings first; the decoder wants spike-band power first, so we
  build trials from the continuous arrays in [spikePow, binnedTX] order instead.

PROVISIONAL until Wairagkar's released features are inspected (DATA_FORMATS.md, mismatches 5, 6, 8):
channel permutation (default identity), normalization (default none) and pre-go context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io

from wvx.config import resolve_path
from wvx.session import SessionData, Trial


def _cellstr(cell: np.ndarray) -> list[str]:
    return [str(np.asarray(c).squeeze()) if np.asarray(c).size else "" for c in cell.ravel()]


def _permutation(path: Optional[str], n_channels: int) -> np.ndarray:
    """0-based column permutation from a JSON list (identity if path is None)."""
    if path is None:
        return np.arange(n_channels)
    perm = np.asarray(json.loads(resolve_path(path).read_text()), dtype=int)
    if sorted(perm.tolist()) != list(range(n_channels)):
        raise ValueError(f"{path} is not a permutation of 0..{n_channels - 1}")
    return perm


def _normalize(x: np.ndarray, blocks: np.ndarray, method: str, eps: float) -> np.ndarray:
    """Normalize continuous features (T x C). 'zscore_block' uses each block's own statistics."""
    if method == "none":
        return x
    if method == "zscore_file":
        return (x - x.mean(0)) / (x.std(0) + eps)
    if method == "zscore_block":
        out = np.empty_like(x)
        for b in np.unique(blocks):
            rows = blocks == b
            out[rows] = (x[rows] - x[rows].mean(0)) / (x[rows].std(0) + eps)
        return out
    raise ValueError(f"unknown normalization method {method!r}")


def load_condition(path, name: str, layer: int, adapter: dict) -> SessionData:
    """Load one Kunz file (e.g. the attempted or imagined 50-word file) as a SessionData.

    ``adapter`` is the ``adapter`` section of configs/data/kunz.yaml.
    """
    m = scipy.io.loadmat(path)
    sp = m["spikePow"].astype(np.float64)
    tx = m["binnedTX"].astype(np.float64)
    if sp.shape != tx.shape:
        raise ValueError(f"spikePow {sp.shape} and binnedTX {tx.shape} differ")
    if int(np.asarray(m["binSize"]).squeeze()) != adapter["bin_ms"]:
        raise ValueError(f"binSize {m['binSize']} != configured {adapter['bin_ms']} ms")
    blocks = m["blockNum"].ravel().astype(int)
    go = m["goTrialEpochs"].astype(int)
    delay = m["delayTrialEpochs"].astype(int)
    cues = _cellstr(m["sentenceText"])
    decoded = _cellstr(m["decodedSentenceText"])

    if adapter.get("clip_spikepow") is not None:
        sp = np.minimum(sp, adapter["clip_spikepow"])
    perm = _permutation(adapter.get("channel_permutation"), sp.shape[1])
    norm = adapter["normalization"]
    sp = _normalize(sp[:, perm], blocks, norm["method"], norm["eps"])
    tx = _normalize(tx[:, perm], blocks, norm["method"], norm["eps"])
    x = np.concatenate([sp, tx], axis=1).astype(np.float32)   # spike-band power first

    trials = []
    for i in range(go.shape[0]):
        g0, g1 = go[i, 0] - 1, go[i, 1]                        # 1-based inclusive -> python slice
        block = blocks[g0]
        if blocks[g1 - 1] != block:
            raise ValueError(f"{name} trial {i}: go epoch crosses a block boundary")
        start = max(g0 - adapter["pre_go_bins"], 0)
        start += int(np.argmax(blocks[start:g0 + 1] == block))  # never reach into the previous block
        trials.append(
            Trial(
                neural=x[start:g1],
                text=cues[i],
                block=int(block),
                meta={
                    "trial_index": i,
                    "go_offset": g0 - start,               # row of the go cue within the trial
                    "go_bins": (int(go[i, 0]), int(go[i, 1])),
                    "delay_bins": (int(delay[i, 0]), int(delay[i, 1])),
                    "closed_loop": decoded[i] != "",
                    "decoded_text": decoded[i],
                },
            )
        )
    return SessionData(name=name, layer=layer, trials=trials)
