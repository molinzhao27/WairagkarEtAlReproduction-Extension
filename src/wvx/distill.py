"""Self-distillation targets for inner speech (D-086): the decoder's own voice for ATTEMPTED speech of a sentence,
time-warped onto an inner-speech trial of the same sentence.

Both trials' 50 ms phoneme streams are force-aligned to the sentence (``wvx.phonemes.ctc_align``); each label's first
token gives a pair of anchor times (inner, attempted); the warp is piecewise linear through the anchors (plus both
trials' first and last frames), and the attempted features are read at the warped times (linear interpolation)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from wvx.phonemes import ctc_align


def label_onsets(path: np.ndarray, n_labels: int) -> np.ndarray:
    """First frame of each label's state in a ``ctc_align`` path (-1 where the label never appears)."""
    out = np.full(n_labels, -1)
    for t, s in enumerate(path):
        if s % 2 and out[s // 2] < 0:
            out[s // 2] = t
    return out


def warp_map(inner_lp: np.ndarray, att_lp: np.ndarray, label: Sequence[int], n_inner: int, n_att: int,
             pool: int) -> np.ndarray:
    """For each 10 ms frame of the inner trial (``n_inner``), the (fractional) attempted frame it maps to.
    ``*_lp``: phase-0 log-probabilities per ``pool``-bin token; token k's centre is frame ``pool * k + pool // 2``."""
    n = len(label)
    oi = label_onsets(ctc_align(inner_lp, label), n)
    oa = label_onsets(ctc_align(att_lp, label), n)
    xs, ys = [0.0], [0.0]
    for a, b in zip(oi, oa):
        if a < 0 or b < 0:
            continue
        x, y = float(pool * a + pool // 2), float(pool * b + pool // 2)
        if x > xs[-1] and y > ys[-1] and x < n_inner - 1 and y < n_att - 1:
            xs.append(x)
            ys.append(y)
    xs.append(float(n_inner - 1))
    ys.append(float(n_att - 1))
    return np.interp(np.arange(n_inner, dtype=float), xs, ys)


def warp_features(att: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``att`` (T_att, F) read at fractional frames ``f`` (linear interpolation)."""
    lo = np.clip(np.floor(f).astype(int), 0, len(att) - 1)
    hi = np.clip(lo + 1, 0, len(att) - 1)
    w = (f - lo)[:, None]
    return ((1 - w) * att[lo] + w * att[hi]).astype(np.float32)
