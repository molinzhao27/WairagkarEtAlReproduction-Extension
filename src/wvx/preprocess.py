"""Causal per-block feature preprocessing of Wairagkar et al. 2025, for continuous 10 ms binned features.

Their readme and Methods: "Each bin was log-transformed, normalized using rolling means and s.d. from the
past 10 s and then causally smoothed using a sigmoid kernel of length 1.5 s of the past activity." No
formula, offset or kernel parameter is published. What is implemented here, for one feature type
(T bins x C channels, one recording block at a time):

1. log:      l_t = log(x_t + offset)
2. rolling:  z_t = (l_t - mean(W_t)) / sd(W_t),  W_t = the last N bins of the same block
             (``include_current``: W_t = l[t-N+1 .. t], else l[t-N .. t-1]); sd uses ``ddof``.
             Warm-up: fewer than N bins exist after a block starts, so W_t holds all the block's
             bins so far (expanding window). Strictly causal. An sd <= ``min_std`` (all values in W_t
             equal) is replaced by ``zero_std_value``, as BRAND's z_window node does.
3. smooth:   y_t = sum_{k=0}^{L-1} w_k z_{t-k},  w_k = 1 / (1 + exp((k - center) / scale)), sum_k w_k = 1.
             k = 0 is the current bin, so recent bins weigh most and the weight falls off as a
             sigmoid of the lag. In the first L-1 bins of a block the missing past is dropped and the
             remaining weights are renormalized.

Every number (offsets, N, L, center, scale, ...) comes from the ``params`` block of
``adapter.normalization`` in configs/data/kunz.yaml, where the evidence for each default is recorded.
Blocks are processed independently: nothing crosses a change in the block label.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.signal import lfilter

FEATURES = ("spikepow", "threshcross")


def _ms_to_bins(ms: float, bin_ms: float, what: str, integer: bool) -> float:
    bins = ms / bin_ms
    if integer and (bins != round(bins) or bins < 1):
        raise ValueError(f"{what} = {ms} ms is not a positive whole number of {bin_ms} ms bins")
    return int(round(bins)) if integer else float(bins)


def sigmoid_kernel(length: int, center: float, scale: float) -> np.ndarray:
    """Causal smoothing weights w_0..w_{length-1} (w_0 = current bin), sigmoid in the lag, summing to 1."""
    if length < 1 or scale <= 0:
        raise ValueError(f"need length >= 1 and scale > 0, got {length}, {scale}")
    k = np.arange(length, dtype=np.float64)
    w = 1.0 / (1.0 + np.exp((k - center) / scale))
    return w / w.sum()


def log_transform(x: np.ndarray, offset: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(x + offset <= 0):
        raise ValueError(f"log of a non-positive value: min(x) + offset = {x.min() + offset}")
    return np.log(x + offset)


def rolling_zscore(x: np.ndarray, window: int, include_current: bool, ddof: int,
                   min_std: float, zero_std_value: float) -> np.ndarray:
    """Causal rolling z-score of one block (T x C) over the last ``window`` bins (expanding at the start)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    if T == 0:
        return x.copy()
    # Subtracting the block's first row (known at t = 0, so still causal) changes no z-score but keeps
    # the running sums small, so the variance below does not lose precision to cancellation.
    d = x - x[0]
    zero = np.zeros((1,) + x.shape[1:])
    s1 = np.concatenate([zero, np.cumsum(d, axis=0)])        # s1[i] = sum of d[:i]
    s2 = np.concatenate([zero, np.cumsum(d * d, axis=0)])
    t = np.arange(T)
    end = t + 1 if include_current else t                      # window is d[start:end]
    start = np.maximum(end - window, 0)
    n = (end - start).astype(np.float64)
    n_shape = (T,) + (1,) * (x.ndim - 1)
    n = n.reshape(n_shape)
    sum1 = s1[end] - s1[start]
    sum2 = s2[end] - s2[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, sum1 / np.maximum(n, 1), 0.0)
        var = (sum2 - n * mean * mean) / np.maximum(n - ddof, 1)
    var = np.where(n - ddof > 0, np.maximum(var, 0.0), 0.0)
    sd = np.sqrt(var)
    sd = np.where(sd <= min_std, zero_std_value, sd)
    z = (d - mean) / sd
    return np.where(n > 0, z, 0.0)                            # no past at all (only if include_current=False)


def causal_smooth(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """y_t = sum_k kernel[k] x_{t-k} within one block; the first len(kernel)-1 rows use the renormalized
    weights that are available."""
    x = np.asarray(x, dtype=np.float64)
    y = lfilter(kernel, [1.0], x, axis=0)                     # zero initial state = no past before the block
    m = min(len(kernel) - 1, x.shape[0])
    if m > 0:
        avail = np.cumsum(kernel)[:m].reshape((m,) + (1,) * (x.ndim - 1))
        y[:m] = y[:m] / avail
    return y


def block_runs(blocks: np.ndarray) -> list[slice]:
    """Contiguous runs of equal block labels, in order (a recording block per run)."""
    blocks = np.asarray(blocks).ravel()
    if blocks.size == 0:
        return []
    cuts = np.flatnonzero(blocks[1:] != blocks[:-1]) + 1
    edges = np.concatenate([[0], cuts, [blocks.size]])
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def wairagkar_block(x: np.ndarray, feature: str, params: Mapping, bin_ms: float) -> np.ndarray:
    """log -> rolling z-score -> sigmoid smoothing for one block of one feature type (T x C)."""
    if feature not in FEATURES:
        raise ValueError(f"feature must be one of {FEATURES}, got {feature!r}")
    roll = params["rolling"]
    sm = params["smoothing"]
    if sm["kernel"] != "sigmoid":
        raise ValueError(f"unknown smoothing kernel {sm['kernel']!r}")
    window = _ms_to_bins(roll["window_ms"], bin_ms, "rolling.window_ms", integer=True)
    kernel = sigmoid_kernel(
        _ms_to_bins(sm["length_ms"], bin_ms, "smoothing.length_ms", integer=True),
        _ms_to_bins(sm["center_ms"], bin_ms, "smoothing.center_ms", integer=False),
        _ms_to_bins(sm["scale_ms"], bin_ms, "smoothing.scale_ms", integer=False),
    )
    z = rolling_zscore(log_transform(x, params["log_offset"][feature]), window,
                       bool(roll["include_current"]), int(roll["ddof"]),
                       float(roll["min_std"]), float(roll["zero_std_value"]))
    return causal_smooth(z, kernel)


def wairagkar_transform(x: np.ndarray, blocks: np.ndarray, feature: str, params: Mapping,
                        bin_ms: float) -> np.ndarray:
    """Apply :func:`wairagkar_block` to every contiguous block of a continuous (T x C) array."""
    x = np.asarray(x, dtype=np.float64)
    blocks = np.asarray(blocks).ravel()
    if blocks.shape[0] != x.shape[0]:
        raise ValueError(f"{blocks.shape[0]} block labels for {x.shape[0]} bins")
    out = np.empty_like(x)
    for run in block_runs(blocks):
        out[run] = wairagkar_block(x[run], feature, params, bin_ms)
    return out
