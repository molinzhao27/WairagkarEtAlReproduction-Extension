"""Syllable timing for the decoder training targets: the D-033 benchmark and the Kunz v2 targets.

The option-(b) alignment (``wvx.alignment``, D-020/D-027) needs syllables on two timelines: the TTS of
the cue and the decoder output for the trial. The DTW mapping and the per-syllable stretch stay as they
are; this module only supplies the syllables (``align_trial(..., tts_segments=, pred_segments=)``).
Settings: ``syllables`` and ``benchmark`` in ``configs/experiments/syllable_benchmark.yaml``.

**TTS side (every method), ``tts_side: forced``.** The stored TTS features of each cue are
resynthesized with LPCNet (``scripts/syllable_tts.py audio``), so audio time t belongs to feature frame
``(t - audio_lag_ms) / 10 ms``. The Montreal Forced Aligner (english_us_arpa acoustic model and
dictionary) aligns phones and words to that audio. ``tts_syllables`` syllabifies the phones of each
word by the maximal onset principle (``syllabify``): one syllable per vowel (ARPAbet vowels carry a
stress digit), and the consonants between two vowels go to the second syllable as far as they form a
legal onset. Legal onsets (``legal_onsets``) are the consonant clusters that begin at least
``min_onset_words`` CMUdict words (first pronunciation of each word), the textbook data-driven
inventory. Syllables never cross a word boundary. Check: the syllable count of every cue against its
CMUdict count (``dictionary_syllable_count``).
``tts_side: envelope`` keeps D-027's envelope dips (``wvx.alignment.segment_syllables``) instead.

**Decoder side** (``method``), each with its published defaults:

- ``current``: envelope dips of at least 3 dB in the LPCNet c0 envelope (``wvx.alignment``, D-027).
- ``nuclei``: de Jong & Wempe (2009) syllable nuclei, run by ``wvx.syllable_segmenters.praat_nuclei``
  on the LPCNet audio of the decoder output. Nuclei are time points, so ``nuclei_to_segments`` makes
  segments from them: each speech run of the decoder output (the speech mask the DTW already uses, so
  pause edges are shared with ``current``) that holds nuclei is split at the intensity minimum between
  consecutive nuclei. Runs without a nucleus hold no syllable.
- ``sylber``: Sylber's segmenter (Cho et al., ICLR 2025, arXiv 2410.07168), released checkpoint,
  ``wvx.syllable_segmenters.SylberSegmenter``. Its segments are used as they are.

How far the decoder side can matter: in the unchanged mapping, DTW runs on the two envelopes (speech
frames from the envelope masks) whatever the method; the decoder syllables enter only through ``snap``
(each DTW-mapped TTS boundary moves to the nearest detected decoder boundary of the same kind within
``snap_tolerance_ms``). The TTS syllables, in contrast, are the pieces that get stretched.

**Count-constrained variant** (``count_constraint``; secondary, never used for the choice):
``keep_strongest`` keeps the N decoder syllables with the highest envelope peak, N = the number of TTS
syllables; fewer than N are left as they are.

**Mapping arms** (``arm``; D-033 addition, fixed before any benchmark run):

- ``A``: the unchanged D-027 mapping (``wvx.alignment``: envelope DTW, decoder syllables only snap).
- ``B``, segment-driven (``align_trial_segment_driven``, settings ``syllables.segment_mapping``): the
  decoder syllables ARE the target spans. TTS syllables are paired with decoder syllables one to one when
  the counts match, else by a monotone DTW over the two syllable sequences (``syllable_cost``,
  ``syllable_path``; every syllable of both sides on the path, many-to-one and one-to-many allowed). The
  cost uses what both sides have: each syllable's onset as a share of its side's speech time (summed
  syllable durations, so pauses do not count; differences scaled to about one per syllable) plus the
  |log ratio| of its share of that speech time. Each TTS syllable then goes to ONE decoder syllable, the
  lowest-cost of those the path pairs it with (``assign_syllables``; monotone by construction). TTS
  syllables that share a decoder syllable split its span in proportion to their TTS durations
  (``segment_driven_spans``); a decoder syllable with no TTS syllable stays silent; everything outside the
  spans is silence with the same pitch rule as arm A (``wvx.alignment.stretch_features``). No decoder
  syllable at all: the target is silence.

Pure NumPy and the standard library; the tools that need Praat or PyTorch are in
``wvx.syllable_segmenters`` and run in the separate syllable environment (requirements-syllables.txt).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from wvx.config import load_yaml, resolve_path

PathLike = Union[str, Path]

DEFAULT_CONFIG = "configs/experiments/syllable_benchmark.yaml"
METHODS = ("current", "nuclei", "sylber")
TTS_SIDES = ("envelope", "forced")
ARMS = ("A", "B")


# ------------------------------------------------------------------------------------- config


def load_benchmark_config(path: PathLike = DEFAULT_CONFIG, overrides: Optional[Mapping] = None) -> dict:
    """default.yaml <- every base_experiment, recursively <- ``path`` <- ``overrides`` (paths resolved).

    No numpy import, so a script can cap its thread pools afterwards (``wvx.experiment_config``).
    """
    from wvx.experiment_config import load_experiment_config

    return load_experiment_config(path, overrides)


def load_any_targets_config(path: PathLike, overrides: Optional[Mapping] = None) -> dict:
    """``wvx.targets.load_targets_config`` for a config whose base has no base of its own (unchanged
    result and hash for kunz_targets.yaml), else the recursive loader (syllable_benchmark.yaml)."""
    exp = load_yaml(resolve_path(path))
    base = exp.get("base_experiment")
    if isinstance(base, str) and load_yaml(resolve_path(base)).get("base_experiment") is None:
        from wvx.targets import load_targets_config

        return load_targets_config(path, overrides)
    return load_benchmark_config(path, overrides)


def settings(cfg: Mapping[str, Any]) -> dict:
    s = cfg["syllables"]
    if s["method"] not in METHODS:
        raise ValueError(f"syllables.method must be one of {METHODS}, got {s['method']!r}")
    if s["tts_side"] not in TTS_SIDES:
        raise ValueError(f"syllables.tts_side must be one of {TTS_SIDES}, got {s['tts_side']!r}")
    if s["arm"] not in ARMS:
        raise ValueError(f"syllables.arm must be one of {ARMS}, got {s['arm']!r}")
    return s


def frame_ms(cfg: Mapping[str, Any]) -> float:
    s = cfg["synthesis"]
    return 1000.0 * float(s["samples_per_frame"]) / float(s["sample_rate"])


# ----------------------------------------------------------------------------------- locations


def syllables_root(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["paths"]["syllables"])


def decoded_dir(cfg: Mapping[str, Any], cell: str) -> Path:
    """Decoder output of one cell (session x decoder): predictions.npz and wav/<trial>.wav (host only)."""
    return syllables_root(cfg) / "decoded" / cell


def tts_dir(cfg: Mapping[str, Any], voice: str, corpus: str) -> Path:
    """TTS audio for the aligner (mfa_corpus/), its output (mfa_output/) and syllables.json (host only)."""
    from wvx.targets import voice_slug

    return syllables_root(cfg) / "tts" / voice_slug(voice) / corpus


def segments_path(cfg: Mapping[str, Any], cell: str, method: str) -> Path:
    return syllables_root(cfg) / "segments" / f"{cell}__{method}.json"


def utterance_name(index: int) -> str:
    return f"u{index:04d}"


def trial_wav_name(index: int) -> str:
    return f"{index:04d}.wav"


# ------------------------------------------------------------------------- ARPAbet + max onset


def is_vowel(phone: str) -> bool:
    """ARPAbet (CMUdict, MFA english_us_arpa): vowels and only vowels end in a stress digit."""
    return phone[-1:].isdigit()


def base_phone(phone: str) -> str:
    return phone.rstrip("0123456789")


def legal_onsets(pronunciations: Iterable[Sequence[str]], min_words: int) -> frozenset:
    """Consonant clusters (stress-free phone tuples) that begin at least ``min_words`` pronunciations.

    The classic data-driven onset inventory for maximal-onset syllabification: a cluster is a legal
    syllable onset if words begin with it. Pass one pronunciation per word.
    """
    counts: Counter = Counter()
    for pron in pronunciations:
        onset = []
        for p in pron:
            if is_vowel(p):
                if onset:
                    counts[tuple(onset)] += 1
                break
            onset.append(base_phone(p))
    return frozenset(o for o, c in counts.items() if c >= int(min_words))


def syllabify(phones: Sequence[str], onsets: frozenset) -> list[tuple[int, int]]:
    """[start, end) phone indices of the syllables of ONE word, by the maximal onset principle.

    One syllable per vowel. Consonants before the first vowel belong to the first syllable, those
    after the last vowel to the last one. Between two vowels, the next syllable takes the longest
    final part of the cluster that is a legal onset (possibly none). A word without a vowel has no
    syllable.
    """
    vowels = [i for i, p in enumerate(phones) if is_vowel(p)]
    if not vowels:
        return []
    starts = [0]
    for prev, nxt in zip(vowels[:-1], vowels[1:]):
        j = prev + 1
        while j < nxt and tuple(base_phone(p) for p in phones[j:nxt]) not in onsets:
            j += 1
        starts.append(j)
    return list(zip(starts, starts[1:] + [len(phones)]))


def cmudict_first_pronunciations(cmu: Mapping[str, Sequence[Sequence[str]]]) -> dict:
    """{word: first pronunciation} of a CMUdict mapping (``cmudict.dict()``)."""
    return {w: list(p[0]) for w, p in cmu.items() if p}


def dictionary_syllable_count(words: Sequence[str], first_pron: Mapping[str, Sequence[str]]) -> Optional[int]:
    """Vowels in the first CMUdict pronunciation of every word; None if a word is not in the dictionary."""
    n = 0
    for w in words:
        pron = first_pron.get(w.lower())
        if pron is None:
            return None
        n += sum(is_vowel(p) for p in pron)
    return n


# --------------------------------------------------------------------- forced-alignment output


Interval = tuple[float, float, str]


def read_aligner_json(path: PathLike, cfg: Mapping[str, Any]) -> tuple[list[Interval], list[Interval]]:
    """(words, phones) intervals from the aligner's JSON output (``--output_format json``).

    MFA 3 writes ``{"tiers": {"words": {"entries": [[start, end, label], ...]}, "phones": {...}}}``.
    """
    fa = cfg["syllables"]["forced_alignment"]
    with open(path) as f:
        d = json.load(f)
    tiers = d["tiers"]

    def entries(name):
        return [(float(s), float(e), str(lab)) for s, e, lab in tiers[name]["entries"]]

    return entries(fa["word_tier"]), entries(fa["phone_tier"])


def tts_syllables(words: Sequence[Interval], phones: Sequence[Interval], onsets: frozenset,
                  cfg: Mapping[str, Any]) -> dict:
    """Syllables (seconds, audio time) of one force-aligned utterance.

    Each phone belongs to the word interval that holds its midpoint. Silence labels are skipped; a word
    that the aligner marks as unknown is reported (``unknown_words``).
    """
    fa = cfg["syllables"]["forced_alignment"]
    silence, unknown = set(fa["silence_labels"]), str(fa["unknown_label"])
    ph = [p for p in phones if p[2] not in silence]
    mids = np.array([(s + e) / 2 for s, e, _ in ph]) if ph else np.zeros(0)
    out_words, syl, unk, no_vowel, syl_word, syl_phones = [], [], [], [], [], []
    for ws, we, w in words:
        if w in silence:
            continue
        idx = np.flatnonzero((mids >= ws) & (mids < we))
        labels = [ph[i][2] for i in idx]
        out_words.append(w)
        if unknown in labels or w == unknown:
            unk.append(w)
        parts = syllabify(labels, onsets)
        if not parts:
            no_vowel.append(w)
        for a, b in parts:
            syl.append((ph[idx[a]][0], ph[idx[b - 1]][1]))
            syl_word.append(len(out_words) - 1)
            syl_phones.append(labels[a:b])
    return {"words": out_words, "syllables_s": [list(s) for s in syl], "unknown_words": unk,
            "words_without_vowel": no_vowel, "syllable_words": syl_word, "syllable_phones": syl_phones}


def seconds_to_frames(seg_s: Sequence[Sequence[float]], cfg: Mapping[str, Any], n_frames: Optional[int] = None,
                      keep_empty: bool = False) -> np.ndarray:
    """(K, 2) audio times (s) -> [onset, offset) feature frames, minus the LPCNet synthesis lag.

    ``keep_empty`` (TTS): every segment keeps at least one frame and segments never overlap, so the
    count is preserved. Otherwise segments that round to zero frames are dropped.
    """
    lag = float(cfg["syllables"]["audio_lag_ms"])
    seg = np.asarray(seg_s, dtype=np.float64).reshape(-1, 2)
    f = np.rint((seg * 1000.0 - lag) / frame_ms(cfg)).astype(np.int64)
    if n_frames is not None:
        f = np.clip(f, 0, int(n_frames))
    if keep_empty:
        prev = 0
        for k in range(len(f)):
            on = max(int(f[k, 0]), prev)
            off = max(int(f[k, 1]), on + 1)
            if n_frames is not None and off > n_frames:
                raise ValueError(f"syllable {k} ends after the last of {n_frames} frames")
            f[k] = on, off
            prev = off
        return f
    return f[f[:, 1] > f[:, 0]]


def seconds_to_points(t_s: Sequence[float], cfg: Mapping[str, Any], n_frames: int, boundary: bool = False) -> np.ndarray:
    """Audio time points (s) -> feature frames, minus the synthesis lag: the frame that holds each point
    (in [0, n_frames - 1]), or with ``boundary`` the nearest frame boundary (in [0, n_frames])."""
    lag = float(cfg["syllables"]["audio_lag_ms"])
    x = (np.asarray(t_s, dtype=np.float64).ravel() * 1000.0 - lag) / frame_ms(cfg)
    if boundary:
        return np.clip(np.rint(x).astype(np.int64), 0, int(n_frames))
    return np.clip(np.floor(x).astype(np.int64), 0, int(n_frames) - 1)


def audio_feature_lag_ms(raw20: np.ndarray, wav: np.ndarray, cfg: Mapping[str, Any]) -> float:
    """Delay (ms) of LPCNet audio behind its features: the shift that best correlates the c0 power contour
    (frame k placed at its nominal centre, 10k + 5 ms) with the audio power in 10 ms windows centred on
    the same grid, both smoothed like the alignment envelope; searched on a ``lag_check.resolution_ms``
    grid up to ``lag_check.max_lag_ms``. Measured on TTS only (``scripts/syllable_tts.py audio``)."""
    from scipy.ndimage import gaussian_filter1d

    from wvx import alignment as al

    lc = cfg["syllables"]["lag_check"]
    res, fms = float(lc["resolution_ms"]), frame_ms(cfg)
    spf, sr = int(cfg["synthesis"]["samples_per_frame"]), float(cfg["synthesis"]["sample_rate"])
    raw20 = np.asarray(raw20, dtype=np.float64)
    n = raw20.shape[0]
    t = np.arange(0.0, n * fms, res) + res / 2
    c = np.interp(t, np.arange(n) * fms + fms / 2, al.c0_to_db(raw20[:, 0], cfg))
    x = np.asarray(wav, dtype=np.float64) ** 2
    cs = np.concatenate([[0.0], np.cumsum(x)])
    s = np.clip(np.rint((t - fms / 2) * sr / 1000.0).astype(np.int64), 0, x.size)
    e = np.clip(s + spf, 0, x.size)
    amin = float(cfg["alignment"]["envelope"]["amin"])
    p = 10.0 * np.log10(np.maximum((cs[e] - cs[s]) / np.maximum(e - s, 1), amin ** 2))
    sig = float(cfg["alignment"]["envelope"]["smooth_ms"]) / res
    c, p = gaussian_filter1d(c, sig), gaussian_filter1d(p, sig)
    m, L = t.size, int(round(float(lc["max_lag_ms"]) / res))
    best, best_r = 0, -np.inf
    for d in range(-L, L + 1):
        a, b = c[max(0, -d): m - max(0, d)], p[max(0, d): m - max(0, -d)]
        if a.size > 2 and a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
            if r > best_r:
                best, best_r = d, r
    return best * res


# ------------------------------------------------------------------------------ decoder side


def nuclei_to_segments(nuclei_f: Sequence[int], cuts_f: Sequence[int], speech: np.ndarray) -> np.ndarray:
    """Syllable segments from nuclei (frames, sorted), the cut between each consecutive pair, and a
    speech mask: every speech run with nuclei is split at the cuts between its nuclei; runs without a
    nucleus hold no syllable; nuclei outside every run are ignored. ``cuts_f[i]`` lies between
    ``nuclei_f[i]`` and ``nuclei_f[i + 1]``.
    """
    from wvx.alignment import runs

    nuc = np.asarray(nuclei_f, dtype=np.int64).ravel()
    cuts = np.asarray(cuts_f, dtype=np.int64).ravel()
    if nuc.size and cuts.size != nuc.size - 1:
        raise ValueError(f"{nuc.size} nuclei need {nuc.size - 1} cuts, got {cuts.size}")
    if np.any(np.diff(nuc) < 0):
        raise ValueError("nuclei must be sorted")
    segs: list[tuple[int, int]] = []
    for a, b in runs(np.asarray(speech, dtype=bool)):
        idx = np.flatnonzero((nuc >= a) & (nuc < b))
        edges = [int(a)]
        for i in idx[:-1]:
            c = int(cuts[i])
            if edges[-1] < c < b:
                edges.append(c)
        if idx.size:
            edges.append(int(b))
            segs.extend(zip(edges[:-1], edges[1:]))
    return np.asarray(segs, dtype=np.int64).reshape(-1, 2)


def keep_strongest(segs: np.ndarray, env_db: np.ndarray, n: Optional[int]) -> np.ndarray:
    """At most ``n`` segments: those with the highest envelope peak (earlier wins ties), in time order."""
    segs = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    if n is None or len(segs) <= int(n):
        return segs
    env = np.asarray(env_db, dtype=np.float64)
    strength = np.array([env[a:b].max() for a, b in segs])
    keep = np.sort(np.argsort(-strength, kind="stable")[: int(n)])
    return segs[keep]


def decoder_segments(method: str, env_db: np.ndarray, cfg: Mapping[str, Any],
                     record: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """(J, 2) syllables of one trial's decoder output on the neural timeline (len(env_db) frames).

    ``current`` segments the envelope itself; ``nuclei`` / ``sylber`` read the segmenter's record for the
    trial (``scripts/syllable_segment.py``), in audio seconds.
    """
    from wvx import alignment as al

    n = int(np.asarray(env_db).size)
    if method == "current":
        return al.segment_syllables(env_db, cfg)
    if record is None:
        raise ValueError(f"method {method!r} needs the segmenter's record for the trial")
    if method == "nuclei":
        nuc = seconds_to_points(record["nuclei_s"], cfg, n)
        cuts = seconds_to_points(record["cuts_s"], cfg, n, boundary=True)
        return nuclei_to_segments(nuc, cuts, al.speech_mask(env_db, cfg))
    if method == "sylber":
        return seconds_to_frames(record["segments_s"], cfg, n_frames=n)
    if method == "manual":
        return manual_segments(record["spans_s"], cfg, n)
    raise ValueError(f"unknown method {method!r}")


def manual_segments(spans_s: Sequence[Sequence[float]], cfg: Mapping[str, Any], n_frames: int) -> np.ndarray:
    """Manual syllable spans (``wvx.manual_alignment``: one per TTS syllable, in seconds read off the NEURAL timeline,
    so no synthesis lag) -> (K, 2) [onset, offset) frames, each time to the nearest frame boundary (half up).

    Not a ``METHODS`` entry: ``decoder_segments("manual", ...)`` reads ``record["spans_s"]``, and arm B pairs the K
    spans with the K TTS syllables one to one. Invalid spans (empty, overlapping, outside) raise, never get fixed.
    """
    from wvx.alignment import check_segments

    seg = np.asarray(spans_s, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(seg)):
        raise ValueError("manual spans must be finite")
    return check_segments(np.floor(seg * 1000.0 / frame_ms(cfg) + 0.5).astype(np.int64), int(n_frames), "manual")


# ------------------------------------------------------------- arm B: segment-driven mapping


def speech_time_features(segs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(onset share, duration share) of each syllable within its side's speech time (summed durations)."""
    d = np.diff(np.asarray(segs, dtype=np.float64).reshape(-1, 2), axis=1).ravel()
    if d.size == 0 or np.any(d <= 0):
        raise ValueError("syllables must be non-empty")
    total = d.sum()
    return np.concatenate([[0.0], np.cumsum(d)[:-1]]) / total, d / total


def syllable_cost(tts_segs: np.ndarray, dec_segs: np.ndarray, mcfg: Mapping[str, Any]) -> np.ndarray:
    """(K, J) pair costs: position_weight * |onset share difference| * (K + J) / 2
    + log_duration_weight * |ln(TTS duration share / decoder duration share)|.

    Both terms are dimensionless and invariant to each side's speaking rate and pauses; the position
    difference is scaled so that one syllable's worth of speech time costs about 1.
    """
    pt, rt = speech_time_features(tts_segs)
    pd, rd = speech_time_features(dec_segs)
    unit = (pt.size + pd.size) / 2.0
    return (float(mcfg["position_weight"]) * unit * np.abs(pt[:, None] - pd[None, :])
            + float(mcfg["log_duration_weight"]) * np.abs(np.log(rt[:, None] / rd[None, :])))


def syllable_path(cost: np.ndarray) -> np.ndarray:
    """Monotone (L, 2) path [tts, decoder] from (0, 0) to (K - 1, J - 1) minimising the summed pair cost,
    steps (1, 1), (1, 0), (0, 1) with equal weights (every syllable of both sides is on the path).
    Ties prefer the diagonal step, then (1, 0), then (0, 1): deterministic."""
    C = np.asarray(cost, dtype=np.float64)
    K, J = C.shape
    D = np.full((K + 1, J + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, K + 1):
        for j in range(1, J + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    path = [(K - 1, J - 1)]
    i, j = K, J
    while (i, j) != (1, 1):
        steps = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]          # tie order: diagonal, (1, 0), (0, 1)
        i, j = min(steps, key=lambda ij: D[ij])                   # min() keeps the first of equal values
        path.append((i - 1, j - 1))
    return np.asarray(path[::-1], dtype=np.int64)


def assign_syllables(path: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """(K,) decoder syllable of each TTS syllable: the lowest-cost of its path pairs (earliest on ties).
    Monotone: consecutive TTS syllables share at most one decoder syllable on a DTW path."""
    path = np.asarray(path, dtype=np.int64)
    K = int(cost.shape[0])
    out = np.empty(K, dtype=np.int64)
    for i in range(K):
        js = path[path[:, 0] == i, 1]
        out[i] = js[int(np.argmin(cost[i, js]))]
    return out


def segment_driven_spans(tts_segs: np.ndarray, dec_segs: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """(K, 2) target span of each TTS syllable: its decoder syllable, split among the TTS syllables assigned
    to it in proportion to their TTS durations (rounded; a span can be empty if the decoder syllable has
    fewer frames than TTS syllables)."""
    tts_segs = np.asarray(tts_segs, dtype=np.int64).reshape(-1, 2)
    dec_segs = np.asarray(dec_segs, dtype=np.int64).reshape(-1, 2)
    d = np.diff(tts_segs, axis=1).ravel().astype(np.float64)
    out = np.zeros_like(tts_segs)
    for j in np.unique(assignment):
        idx = np.flatnonzero(assignment == j)
        on, off = (int(x) for x in dec_segs[j])
        ends = on + np.rint((off - on) * np.cumsum(d[idx]) / d[idx].sum()).astype(np.int64)
        ends[-1] = off
        out[idx, 0] = np.concatenate([[on], ends[:-1]])
        out[idx, 1] = ends
    return out


def align_trial_segment_driven(pred_raw: np.ndarray, tts_raw: np.ndarray, silence: np.ndarray, cfg: Mapping[str, Any],
                               tts_segments: Optional[np.ndarray] = None, pred_segments: Optional[np.ndarray] = None,
                               pred_env: Optional[np.ndarray] = None, synth=None):
    """Arm B for one trial (module docstring): decoder output (T, 20) + TTS frames (n, 20) -> aligned raw target.

    Missing ``tts_segments`` / ``pred_segments`` are the envelope dips, as in arm A. Returns a
    ``wvx.alignment.Alignment`` whose ``info`` holds the pairing diagnostics.
    """
    from wvx import alignment as al

    mcfg = cfg["syllables"]["segment_mapping"]
    pred_raw = np.asarray(pred_raw, dtype=np.float32)
    tts_raw = np.asarray(tts_raw, dtype=np.float32)
    n_out = pred_raw.shape[0]
    env_p = al.envelope_db(pred_raw, cfg, synth) if pred_env is None else np.asarray(pred_env, np.float64)
    env_t = al.envelope_db(tts_raw, cfg, synth)
    seg_p = al.segment_syllables(env_p, cfg) if pred_segments is None else al.check_segments(pred_segments, n_out, "decoder")
    seg_t = al.segment_syllables(env_t, cfg) if tts_segments is None else al.check_segments(tts_segments, tts_raw.shape[0], "TTS")
    if not len(seg_t):
        raise ValueError("no TTS syllables")
    K, J = len(seg_t), len(seg_p)
    if J == 0:                                     # nothing detected in the decoder output: silence throughout
        spans = np.zeros_like(seg_t)
        info = {"pairing": "none", "n_decoder_syllables": 0, "n_decoder_syllables_without_tts": 0,
                "n_tts_syllables_without_frames": K}
        cost = float("nan")
    else:
        C = syllable_cost(seg_t, seg_p, mcfg)
        one_to_one = K == J and mcfg["equal_counts"] == "one_to_one"
        path = np.stack([np.arange(K), np.arange(K)], axis=1) if one_to_one else syllable_path(C)
        assignment = assign_syllables(path, C)
        spans = segment_driven_spans(seg_t, seg_p, assignment)
        cost = float(C[path[:, 0], path[:, 1]].mean())
        info = {"pairing": "one_to_one" if one_to_one else "dtw", "n_decoder_syllables": J,
                "n_decoder_syllables_without_tts": int(J - np.unique(assignment).size),
                "n_tts_syllables_without_frames": int(np.sum(spans[:, 1] <= spans[:, 0])),
                "assignment": assignment.tolist()}
    target = al.stretch_features(tts_raw, seg_t, spans, n_out, silence, cfg)
    return al.Alignment(target, env_p, env_t, seg_p, seg_t, spans, cost, info)


# ------------------------------------------------------------------------------- file formats


def save_json(path: PathLike, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False))
    tmp.replace(path)
    return path


def load_json(path: PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def save_decoded(path: PathLike, preds: Sequence[np.ndarray], header: Mapping[str, Any]) -> Path:
    """Normalized decoder predictions of every trial of a cell (``header`` must hold ``n_bins``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = dict(header)
    h["pred_lengths"] = [int(p.shape[0]) for p in preds]
    if len(h.get("n_bins", [])) != len(preds):
        raise ValueError("header['n_bins'] must list one bin count per trial")
    np.savez_compressed(path, header=np.array(json.dumps(h)),
                        preds=np.concatenate([np.asarray(p, np.float32) for p in preds]))
    return path


def load_decoded(path: PathLike) -> tuple[list[np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        flat = z["preds"]
    bounds = np.cumsum([0] + header["pred_lengths"])
    if bounds[-1] != flat.shape[0]:
        raise ValueError(f"{path}: header lengths sum to {bounds[-1]}, file has {flat.shape[0]} rows")
    return [flat[a:b] for a, b in zip(bounds[:-1], bounds[1:])], header


def load_tts_syllables(cfg: Mapping[str, Any], voice: str, corpus: str) -> dict:
    """{text_key: (K, 2) frames} from ``scripts/syllable_tts.py syllabify`` (TTS feature timeline)."""
    from wvx.targets import text_key

    d = load_json(tts_dir(cfg, voice, corpus) / cfg["syllables"]["forced_alignment"]["syllables_file"])
    return {text_key(u["text"]): np.asarray(u["frames"], dtype=np.int64).reshape(-1, 2) for u in d["utterances"]}


def segment_records(cfg: Mapping[str, Any], cell: str, method: str) -> tuple[list[dict], dict]:
    """Per-trial records of a segmenter run, in trial order, and the file's header."""
    d = load_json(segments_path(cfg, cell, method))
    recs = sorted(d["trials"], key=lambda r: r["trial"])
    if [r["trial"] for r in recs] != list(range(len(recs))):
        raise ValueError(f"{segments_path(cfg, cell, method)}: trials are not 0..{len(recs) - 1}")
    return recs, {k: v for k, v in d.items() if k != "trials"}


# -------------------------------------------------------------------------------- choice rule


def candidates(rule: Mapping[str, Any]) -> list[str]:
    """Candidate cells "<arm>/<method>" in tie-break order: arm A before B, then ``simplicity_order``."""
    return [f"{a}/{m}" for a in rule["arms"] for m in rule["simplicity_order"]]


def choose_method(target_r: Mapping[str, Mapping[str, float]], rule: Mapping[str, Any]) -> dict:
    """The pre-registered choice (D-033): the (arm, method) cell with the highest mean target r over
    ``rule['primary_cells']``; every candidate within ``tie_tolerance`` of the best is tied, and the tie goes
    to the earliest in ``candidates(rule)`` (arm A before B; current < nuclei < sylber).

    ``target_r["<arm>/<method>"][cell]`` = mean target r of that candidate (free variant, forced TTS side).
    """
    cells, order, tol = list(rule["primary_cells"]), candidates(rule), float(rule["tie_tolerance"])
    means = {}
    for m in order:
        vals = [target_r.get(m, {}).get(c) for c in cells]
        if any(v is None for v in vals):
            raise ValueError(f"candidate {m!r} lacks a result for one of {cells}")
        means[m] = float(np.mean(vals))
    best = max(means.values())
    tied = [m for m in order if means[m] >= best - tol]
    arm, method = tied[0].split("/")
    return {"mean_target_r": means, "best_mean": best, "tied_within_tolerance": tied, "winner": tied[0],
            "winner_arm": arm, "winner_method": method,
            "rule": {"primary_cells": cells, "tie_tolerance": tol, "order": order}}
