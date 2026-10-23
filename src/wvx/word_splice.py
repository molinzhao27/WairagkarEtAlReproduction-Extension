"""Decoder targets spliced from THEIR word recordings (D-052).

Their stored targets (``target_speech_feats_word``) are their own voice, stretched word by word onto each trial's
``word_indices``. Every Kunz word also occurs in their 50-word-vocabulary trials, so a target in their voice can be
built for any sentence of that vocabulary: cut each word out of their targets (an inventory), and for a new trial
place, per word of the cue, the exemplar with the closest duration, linearly resampled to the word's span, with
their silence frame in between. No TTS voice is involved; only the timing comes from elsewhere.

    inv = build_inventory(raw_targets, word_spans, texts)        # their sessions (raw (T, 20) features)
    raw, used = splice(words, spans, n_bins, inv)                # words of the cue, (K, 2) bin spans
    target = wvx.targets.decoder_target(raw, cfg)                # their scaling + smoothing

Pure numpy.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


def words_of(text: str) -> list[str]:
    """The cue's words as the inventory keys them (``wvx.metrics.normalize_text``)."""
    from wvx.metrics import normalize_text

    return normalize_text(text).split()


def build_inventory(raw_targets: Sequence[np.ndarray], word_spans: Sequence[np.ndarray], texts: Sequence[str],
                    sources: Optional[Sequence[str]] = None) -> dict:
    """{"words": {word: [(segment (n, F) float32, source), ...]}, "silence": (F,), "n_trials", "n_skipped"}.

    A trial is used only when its number of word spans equals its number of words and the spans are well formed;
    the silence frame is the per-feature median of every frame outside the word spans of the used trials."""
    words: dict = {}
    sil, used, skipped = [], 0, 0
    for i, (raw, spans, text) in enumerate(zip(raw_targets, word_spans, texts)):
        raw = np.asarray(raw, dtype=np.float32)
        spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
        ws = words_of(text)
        ok = (len(ws) == len(spans) and len(spans) and np.all(spans[:, 1] > spans[:, 0])
              and np.all(spans[1:, 0] >= spans[:-1, 1]) and spans[0, 0] >= 0 and spans[-1, 1] <= raw.shape[0])
        if not ok:
            skipped += 1
            continue
        used += 1
        src = sources[i] if sources is not None else str(i)
        inside = np.zeros(raw.shape[0], bool)
        for k, (w, (a, b)) in enumerate(zip(ws, spans)):
            words.setdefault(w, []).append((raw[a:b].copy(), f"{src}:w{k}"))
            inside[a:b] = True
        sil.append(raw[~inside])
    if not used:
        raise ValueError("no usable trial for the inventory")
    return {"words": words, "silence": np.median(np.concatenate(sil), axis=0).astype(np.float32),
            "n_trials": used, "n_skipped": skipped}


def resample(seg: np.ndarray, n: int) -> np.ndarray:
    """Linear resampling of (m, F) frames to (n, F) (endpoints kept)."""
    seg = np.asarray(seg, dtype=np.float64)
    m = seg.shape[0]
    if n <= 0:
        return np.zeros((0, seg.shape[1]), np.float32)
    if m == 1:
        return np.repeat(seg, n, 0).astype(np.float32)
    x = np.linspace(0.0, m - 1.0, n)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, m - 1)
    f = (x - lo)[:, None]
    return ((1 - f) * seg[lo] + f * seg[hi]).astype(np.float32)


def choose(inventory: Mapping, word: str, n: int):
    """The exemplar of ``word`` whose length is closest to ``n`` frames (ties: inventory order); None if absent."""
    ex = inventory["words"].get(word)
    if not ex:
        return None
    return min(ex, key=lambda e: abs(e[0].shape[0] - n))


def splice(words: Sequence[str], spans: np.ndarray, n_bins: int, inventory: Mapping) -> tuple[Optional[np.ndarray], list]:
    """Raw (n_bins, F) target: silence, with each word's chosen exemplar resampled onto its span. Returns
    (raw, sources); raw is None when a word is missing from the inventory. Spans with no frames are skipped."""
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    if len(words) != len(spans):
        raise ValueError(f"{len(words)} words, {len(spans)} spans")
    out = np.repeat(inventory["silence"][None], n_bins, 0).astype(np.float32)
    used = []
    for w, (a, b) in zip(words, spans):
        a, b = max(int(a), 0), min(int(b), n_bins)
        if b <= a:
            used.append(None)
            continue
        e = choose(inventory, w, b - a)
        if e is None:
            return None, used
        out[a:b] = resample(e[0], b - a)
        used.append(e[1])
    return out, used


def word_spans_from_syllables(mapped: np.ndarray, syllable_words: Sequence[int], n_words: int) -> np.ndarray:
    """(n_words, 2) word spans from (K, 2) mapped syllable spans and each syllable's word index: the union of the
    word's syllables that have frames ([0, 0] when none)."""
    mapped = np.asarray(mapped, dtype=np.int64).reshape(-1, 2)
    out = np.zeros((n_words, 2), np.int64)
    for w in range(n_words):
        segs = [mapped[k] for k, sw in enumerate(syllable_words) if sw == w and mapped[k, 1] > mapped[k, 0]]
        if segs:
            out[w] = (min(s[0] for s in segs), max(s[1] for s in segs))
    return out
