"""Data for the brain2voice-2.0-style decoder (D-075): windows, augmentation, live-style decoding windows (NumPy).

The dataset (``scripts/build_b2v2_data.py``) is one directory: ``neural.f16`` (every trial's neural features
concatenated, (N, 512) float16, 10 ms bins, the release's channel order [spike-band power | threshold crossings]),
``target.f32`` ((N, 20) float32, their LPCNet targets scaled to [-1, 1], unsmoothed) and ``meta.json`` (shapes and
the trial table: per trial its global id ``gid`` (their benchmark indexing), session, ``start`` / ``length`` in
``neural.f16``, cue, phoneme label per word, word timings, closed-loop audio flag, benchmark set).

Training windows follow the paper (6.4 s, downsampled to 50 ms by averaging 5 bins) with our documented choices:
- a window is token-aligned to its END bin, and each token's target is the LPCNet frame of its LAST bin (the frame
  a live decoder emits at that moment);
- the window is snapped so that it cuts no word (its end moves back to the onset of a word it would cut, its start
  forward past one); its CTC label is the phonemes of the words it contains, each followed by SIL. Trials whose word
  timings do not match their pronunciation list get a CTC label only when the window holds all of their speech;
  trials without usable timings get none (acoustic loss only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

N_SBP = 256        # spike-band power columns first, then threshold crossings


class B2V2Data:
    """Memory-mapped dataset directory (see module docstring)."""

    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        n, f, k = self.meta["n_bins"], self.meta["n_features"], self.meta["n_targets"]
        self.neural = np.memmap(self.root / "neural.f16", dtype=np.float16, mode="r", shape=(n, f))
        self.target = np.memmap(self.root / "target.f32", dtype=np.float32, mode="r", shape=(n, k))
        self.trials = self.meta["trials"]
        self.by_gid = {t["gid"]: t for t in self.trials}

    def trial_arrays(self, t: Mapping) -> tuple[np.ndarray, np.ndarray]:
        a, n = int(t["start"]), int(t["length"])
        return np.asarray(self.neural[a:a + n], np.float32), np.asarray(self.target[a:a + n], np.float32)


# ---------------------------------------------------------------------------------------------- windows


def _cuts(on: int, off: int, edge: int) -> bool:
    return on < edge < off


def snap_window(s: int, e: int, words: Optional[Sequence[Sequence[int]]], pool: int) -> tuple[int, int, list[int]]:
    """[s, e) -> (s', e', indices of the words inside): a window of whole ``pool``-bin tokens ending at e' that cuts
    no word ([onset, offset) bins). e' is e, or the onset of the word e would cut; s' >= s is the first token start
    that cuts no word. A window with no token left has s' == e'."""
    w = [(int(a), int(b)) for a, b in (words or [])]
    for on, off in w:
        if _cuts(on, off, e):
            e = on
            break
    n_tok = max(0, (e - s) // pool)
    s = e - pool * n_tok
    moved = True
    while moved and n_tok > 0:
        moved = False
        for on, off in w:
            if _cuts(on, off, s):
                n_tok = max(0, (e - off) // pool)
                s = e - pool * n_tok
                moved = True
    inside = [k for k, (on, off) in enumerate(w) if on >= s and off <= e] if n_tok > 0 else []
    return s, e, inside


def sample_window(t: Mapping, rng: np.random.Generator, max_bins: int, pool: int) -> dict:
    """One training window of trial ``t``: ``{"s", "e", "label" (ids), "ctc"}`` (``s``/``e`` relative to the trial)."""
    n = int(t["length"])
    L = min(n, max_bins)
    s0 = int(rng.integers(0, n - L + 1))
    words, per_word = t.get("words"), t.get("word_labels")
    if words is not None and per_word is not None:
        s, e, inside = snap_window(s0, s0 + L, words, pool)
        if e - s >= pool:
            return {"s": s, "e": e, "label": [i for k in inside for i in per_word[k]], "ctc": True}
    span, full = t.get("speech_span"), t.get("label")
    if span is not None and full is not None and span[1] - span[0] <= L:
        lo, hi = max(0, span[1] - L), min(int(span[0]), n - L)
        if lo <= hi:
            s0 = int(rng.integers(lo, hi + 1))
            e = s0 + L
            s = e - pool * (L // pool)
            if s <= span[0]:
                return {"s": s, "e": e, "label": list(full), "ctc": True}
    e = s0 + L
    return {"s": e - pool * (L // pool), "e": e, "label": [], "ctc": False}


def pool_tokens(x: np.ndarray, pool: int) -> np.ndarray:
    """(n_tok * pool, F) -> (n_tok, F): mean of each ``pool`` consecutive bins."""
    return x.reshape(-1, pool, x.shape[-1]).mean(1)


def make_batch(data: B2V2Data, trial_list: Sequence[Mapping], rng: np.random.Generator, cfg: Mapping[str, Any],
               augment: bool = False) -> dict:
    """Padded batch (right padding) of one window per trial: ``x`` (B, L, F) float32, ``y`` (B, L, 20), ``valid``
    (B, L) bool, ``speech`` (B, L) bool, ``labels`` (B, S) int64 (0-padded), ``label_len`` (B,), ``ctc`` (B,) bool."""
    m = cfg["data"]
    pool, max_bins = int(m["pool"]), int(m["context_tokens"]) * int(m["pool"])
    c0, thr = int(m["speech_feature"]), float(m["speech_threshold"])
    xs, ys, labels, ctc = [], [], [], []
    for t in trial_list:
        w = sample_window(t, rng, max_bins, pool)
        a = int(t["start"])
        x = np.asarray(data.neural[a + w["s"]: a + w["e"]], np.float32)
        xs.append(pool_tokens(x, pool))
        ys.append(np.asarray(data.target[a + w["s"] + pool - 1: a + w["e"]: pool], np.float32))
        labels.append(w["label"])
        ctc.append(w["ctc"])
    B, L = len(xs), max(len(x) for x in xs)
    F, K = xs[0].shape[1], ys[0].shape[1]
    out = {"x": np.zeros((B, L, F), np.float32), "y": np.zeros((B, L, K), np.float32),
           "valid": np.zeros((B, L), bool), "labels": np.zeros((B, max(1, max(map(len, labels)))), np.int64),
           "label_len": np.array([len(lab) for lab in labels], np.int64), "ctc": np.array(ctc, bool)}
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        out["x"][i, :len(x)], out["y"][i, :len(y)], out["valid"][i, :len(x)] = x, y, True
        out["labels"][i, :len(lab)] = lab
    out["speech"] = out["valid"] & (out["y"][..., c0] > thr)
    if augment:
        out["x"] = augment_batch(out["x"], out["valid"], rng, cfg["augmentation"])
    return out


def augment_batch(x: np.ndarray, valid: np.ndarray, rng: np.random.Generator, a: Mapping[str, Any]) -> np.ndarray:
    """The paper's augmentations on the 50 ms model input (our choice of where): white noise (sd ``noise``), a constant
    per-feature offset over the window (sd ``offset``; threshold-crossing columns times ``offset_tx_scale``), a
    per-feature random-walk drift (step sd ``drift``), and ``channel_mask`` of the features zeroed (per window)."""
    B, L, F = x.shape
    x = x + rng.normal(0.0, float(a["noise"]), x.shape).astype(np.float32)
    off = rng.normal(0.0, float(a["offset"]), (B, 1, F)).astype(np.float32)
    off[..., N_SBP:] *= float(a["offset_tx_scale"])
    x = x + off + np.cumsum(rng.normal(0.0, float(a["drift"]), x.shape), axis=1).astype(np.float32)
    keep = rng.random((B, 1, F)) >= float(a["channel_mask"])
    return np.where(keep & valid[..., None], x, 0.0).astype(np.float32)


# ------------------------------------------------------------------------------------ live-style decoding


def live_windows(x: np.ndarray, window_bins: int, pool: int, start: int = 0, stop: Optional[int] = None):
    """Decoding windows for every 10 ms step t in [max(start, pool - 1), stop): the bins (t - window_bins, t] as
    ``pool``-bin tokens ending at t (fewer at the trial start). Returns ``tokens`` (n_steps, n_max, F) left-aligned
    with zero right padding, ``n_tok`` (n_steps,), and the steps ``t``."""
    T, F = x.shape
    stop = T if stop is None else min(stop, T)
    steps = np.arange(max(start, pool - 1), stop)
    n_max = window_bins // pool
    n_tok = np.minimum(n_max, (steps + 1) // pool)
    tokens = np.zeros((len(steps), n_max, F), np.float32)
    for j, (t, n) in enumerate(zip(steps, n_tok)):
        tokens[j, :n] = pool_tokens(x[t + 1 - pool * n: t + 1], pool)
    return tokens, n_tok, steps
