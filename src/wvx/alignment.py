"""Align TTS target speech to a trial's neural timeline: the paper's method for later sessions.

Wairagkar et al. (Methods, "Generating decoder training target voice"): a decoder trained on earlier
sessions synthesizes speech for the new trials; syllable boundaries are segmented automatically from
the envelope of that synthetic speech; DTW between the TTS envelope and the predicted envelope maps
the TTS syllable boundaries onto the predicted ones; each TTS segment is then time-stretched to the
duration of its neural counterpart. Their code for this is not released, so every step below that
the paper leaves open is our choice, set in ``configs/experiments/kunz_targets.yaml`` (``alignment``):

1. **Envelope** (``envelope_db``). One value per 10 ms LPCNet frame (envelope frame k = neural bin
   k), Gaussian-smoothed (sigma ``smooth_ms``, about a 9 Hz low-pass, i.e. syllable rate), in dB.
   ``source: c0`` (default) uses LPCNet's c0, which is exactly the mean log band power
   (``c0 = sqrt(18) * mean log10(1e-2 + E_b) - 4``) of the features, decoder output and TTS alike.
   ``source: audio`` is the paper's literal wording: both are rendered by the same LPCNet synthesis
   and the frame RMS is taken. On the release, c0 aligned better on t15_day195 (where it was picked)
   and on t15_day193 (held out for this choice): target r 0.279/0.290 vs 0.233/0.251. Plausibly
   because the vocoder's excitation noise makes the RMS envelope ragged inside words.
2. **Syllables** (``segment_syllables``). Speech = frames above ``max(peak - speech_threshold_db,
   noise floor + floor_margin_db)``, short gaps bridged, short blips dropped. Inside each speech
   region, syllable nuclei are envelope peaks; a boundary is the minimum between two peaks when the
   dip is at least ``min_dip_db`` below both (Mermelstein-style dip criterion), with no syllable
   shorter than ``min_syllable_ms``. The same function segments TTS and decoder output.
3. **DTW** (``dtw_path``, ``map_syllables``). On each envelope clipped at its own silence threshold
   and scaled to [0, 1] (level-invariant; silence is flat, so DTW cannot chase noise in pauses), plus
   the envelope slope (onsets/offsets), Euclidean cost, symmetric steps, optional Sakoe-Chiba band.
   With ``speech_only`` (default) the silent frames of both envelopes are removed first and the
   result is mapped back. Reason: T15 spoke one word at a time with long pauses (release readme),
   while the TTS speaks the sentence as one phrase; over whole envelopes DTW can absorb several of
   his words into the TTS's leading silence (seen on t15_day195). Without pauses, the two speech
   sequences are matched directly and each pause becomes a syllable boundary.
4. **Mapping** (``map_segments``). A TTS onset after silence maps to the first decoder frame DTW
   pairs with it, an offset before silence to the last. A boundary between two touching syllables
   (an envelope dip, frame b) maps to the centre of the decoder frames DTW pairs with b; if b pairs
   with at least ``min_gap_ms`` of decoder output, that stretch is a pause in the decoder output
   (T15 spoke one word at a time) and stays silent: the syllable ends where the stretch starts and
   the next one begins where it ends. With ``snap``, each mapped
   boundary then moves to the nearest detected decoder boundary of the same kind (onset/offset)
   within ``snap_tolerance_ms``, as long as order is kept: "maps TTS syllable boundaries onto the
   predicted ones". A dip in the TTS can thus become a pause if the decoder output has one there.
5. **Stretch** (``stretch_features`` / ``stretch_audio``). ``domain: features`` resamples the TTS
   LPCNet frames of each syllable to the mapped length (linear interpolation per feature). LPCNet
   frames are a source-filter description (spectral envelope + pitch per 10 ms), so this changes
   timing only: pitch and timbre are preserved by construction, with no phase-vocoder artefacts and
   no second vocoding pass. ``domain: audio`` is the audio-domain alternative (LPCNet-resynthesized
   TTS, phase-vocoder stretch per syllable, re-encoded); it needs synthesis and encoding callables.
   Everything outside the mapped syllables (before, between, after) is the LPCNet frame of digital
   silence, which their norm JSON shows their targets contain (``wvx.targets``). Their released
   targets hold the pitch period of the last speech frame through each pause (what LPCNet's encoder
   does over zeros in one continuous stream; only before the first word is it 3.1), so with
   ``silence_pitch: hold`` our silence frames do the same; ``code`` keeps the constant 3.1.

The decoder-target transform (scale with their JSON, smooth sigma 2) is applied afterwards by
``wvx.targets.decoder_target``. Functions here are pure NumPy/SciPy except where a synthesis or
encode callable is passed in, so they are tested on synthetic signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

PITCH = 18                                       # LPCNet feature index of the pitch period

Synth = Callable[[np.ndarray], np.ndarray]      # (T, 20) raw features -> waveform of T * spf samples
Encode = Callable[[np.ndarray], np.ndarray]     # waveform (16 kHz) -> (n, >= 20) raw features


# -------------------------------------------------------------------------------- helpers


def frame_ms(cfg: Mapping[str, Any]) -> float:
    s = cfg["synthesis"]
    return 1000.0 * float(s["samples_per_frame"]) / float(s["sample_rate"])


def ms_to_frames(ms: float, cfg: Mapping[str, Any]) -> int:
    return int(round(float(ms) / frame_ms(cfg)))


def runs(mask: np.ndarray) -> np.ndarray:
    """(K, 2) [start, end) of the True runs of a boolean vector."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1).astype(np.int64)


# ------------------------------------------------------------------------------- envelopes


def _smooth_db(amplitude: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    ecfg = cfg["alignment"]["envelope"]
    sigma = float(ecfg["smooth_ms"]) / frame_ms(cfg)
    a = np.asarray(amplitude, dtype=np.float64)
    if sigma > 0:
        a = gaussian_filter1d(a, sigma, mode="nearest")
    return 20.0 * np.log10(np.maximum(a, float(ecfg["amin"])))


def envelope_from_audio(wav: np.ndarray, n_frames: int, cfg: Mapping[str, Any]) -> np.ndarray:
    """Smoothed frame-RMS envelope in dB, one value per LPCNet frame (n_frames values)."""
    spf = int(cfg["synthesis"]["samples_per_frame"])
    x = np.asarray(wav, dtype=np.float64)
    need = n_frames * spf
    x = x[:need] if x.size >= need else np.concatenate([x, np.zeros(need - x.size)])
    rms = np.sqrt(np.mean(x.reshape(n_frames, spf) ** 2, axis=1))
    return _smooth_db(rms, cfg)


def c0_to_db(c0: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """LPCNet c0 -> mean log band power in dB (10 * mean_b log10(1e-2 + E_b))."""
    a = cfg["alignment"]
    return 10.0 * (np.asarray(c0, np.float64) + float(a["c0_offset"])) / float(a["c0_scale"])


def envelope_from_c0(raw: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    amp = 10.0 ** (c0_to_db(np.asarray(raw)[:, 0], cfg) / 20.0)
    return _smooth_db(amp, cfg)


def envelope_db(raw: np.ndarray, cfg: Mapping[str, Any], synth: Optional[Synth] = None) -> np.ndarray:
    """Envelope (dB, one value per frame) of raw LPCNet features, per ``alignment.envelope.source``."""
    src = cfg["alignment"]["envelope"]["source"]
    if src == "audio":
        if synth is None:
            raise ValueError("envelope source 'audio' needs a synthesis callable")
        return envelope_from_audio(synth(raw), raw.shape[0], cfg)
    if src == "c0":
        return envelope_from_c0(raw, cfg)
    raise ValueError(f"unknown envelope source {src!r}")


# ---------------------------------------------------------------------------- segmentation


def silence_threshold(env_db: np.ndarray, cfg: Mapping[str, Any]) -> float:
    s = cfg["alignment"]["segmentation"]
    peak = float(np.max(env_db))
    rel = peak - float(s["speech_threshold_db"])
    floor = float(np.percentile(env_db, float(s["floor_percentile"]))) + float(s["floor_margin_db"])
    thr = max(rel, floor)
    return thr if thr < peak else rel


def speech_mask(env_db: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Speech frames: above the silence threshold, gaps < min_gap bridged, regions < min_region dropped."""
    s = cfg["alignment"]["segmentation"]
    m = np.asarray(env_db) > silence_threshold(env_db, cfg)
    min_gap, min_region = ms_to_frames(s["min_gap_ms"], cfg), ms_to_frames(s["min_region_ms"], cfg)
    r = runs(m)
    for (a0, a1), (b0, _) in zip(r[:-1], r[1:]):
        if b0 - a1 < min_gap:
            m[a1:b0] = True
    for a, b in runs(m):
        if b - a < min_region:
            m[a:b] = False
    return m


def segment_syllables(env_db: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """(K, 2) int [onset, offset) frames of syllables found in an envelope (module docstring, step 2)."""
    s = cfg["alignment"]["segmentation"]
    dip = float(s["min_dip_db"])
    min_syl = max(1, ms_to_frames(s["min_syllable_ms"], cfg))
    env = np.asarray(env_db, dtype=np.float64)
    segs: list[tuple[int, int]] = []
    for r0, r1 in runs(speech_mask(env, cfg)):
        x = env[r0:r1]
        peaks, _ = find_peaks(x, prominence=dip, distance=min_syl)
        cuts = []
        for p, q in zip(peaks[:-1], peaks[1:]):
            v = p + int(np.argmin(x[p:q + 1]))
            if x[p] - x[v] >= dip and x[q] - x[v] >= dip:
                cuts.append(v)
        edges = [0]
        for c in cuts:
            if c - edges[-1] >= min_syl and (r1 - r0) - c >= min_syl:
                edges.append(c)
        edges.append(r1 - r0)
        segs.extend((r0 + a, r0 + b) for a, b in zip(edges[:-1], edges[1:]))
    return np.asarray(segs, dtype=np.int64).reshape(-1, 2)


# ------------------------------------------------------------------------------------- DTW


def dtw_features(env_db: np.ndarray, cfg: Mapping[str, Any], thr: Optional[float] = None) -> np.ndarray:
    """Per-frame DTW features; ``thr`` = silence threshold (default: computed on ``env_db`` itself)."""
    d = cfg["alignment"]["dtw"]
    env = np.asarray(env_db, dtype=np.float64)
    if d["feature"] == "range":
        thr = silence_threshold(env, cfg) if thr is None else float(thr)
        peak = float(env.max())
        x = (np.clip(env, thr, peak) - thr) / max(peak - thr, 1e-9)
    elif d["feature"] == "zscore":
        x = (env - env.mean()) / max(env.std(), 1e-9)
    else:
        raise ValueError(f"unknown dtw feature {d['feature']!r}")
    feats = [x]
    w = float(d["delta_weight"])
    if w > 0:
        feats.append(w * np.gradient(x))
    return np.stack(feats, axis=1)


def dtw_path(
    env_src: np.ndarray, env_dst: np.ndarray, cfg: Mapping[str, Any],
    thr_src: Optional[float] = None, thr_dst: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Monotone (L, 2) path [src_frame, dst_frame] from (0, 0) to (n-1, m-1), and cost per step."""
    import librosa

    d = cfg["alignment"]["dtw"]
    a, b = dtw_features(env_src, cfg, thr_src), dtw_features(env_dst, cfg, thr_dst)
    band = d.get("band_rad")
    D, wp = librosa.sequence.dtw(
        X=a.T, Y=b.T, metric=d["metric"], backtrack=True,
        global_constraints=band is not None, band_rad=float(band) if band is not None else 0.25,
    )
    path = np.asarray(wp[::-1], dtype=np.int64)
    return path, float(D[-1, -1] / len(path))


# ---------------------------------------------------------------------------------- mapping


def map_segments(
    src_segs: np.ndarray, path: np.ndarray, n_dst: int, dst_segs: Optional[np.ndarray], cfg: Mapping[str, Any],
    pause_rule: bool = True,
) -> np.ndarray:
    """Map (K, 2) source syllables through a DTW path onto the destination timeline (n_dst frames).

    ``pause_rule=False`` (speech-only DTW, where the destination has no pauses left) always maps a
    touching boundary to the centre of the frames paired with the dip frame.
    """
    m = cfg["alignment"]["mapping"]
    src_segs = np.asarray(src_segs, dtype=np.int64).reshape(-1, 2)
    path = np.asarray(path, dtype=np.int64)
    n_src = int(path[:, 0].max()) + 1
    first = np.full(n_src, np.iinfo(np.int64).max)
    last = np.full(n_src, -1)
    np.minimum.at(first, path[:, 0], path[:, 1])
    np.maximum.at(last, path[:, 0], path[:, 1])
    centre = np.bincount(path[:, 0], weights=path[:, 1], minlength=n_src) / np.bincount(path[:, 0], minlength=n_src)

    min_pause = ms_to_frames(cfg["alignment"]["segmentation"]["min_gap_ms"], cfg)
    src_b = src_segs.reshape(-1)                                # on0, off0, on1, off1, ... (source frames)
    bounds = np.empty(src_b.size, dtype=np.int64)
    for k, b in enumerate(src_b):
        joined = (k % 2 == 1 and k + 1 < src_b.size and src_b[k + 1] == b) or (
            k % 2 == 0 and k > 0 and src_b[k - 1] == b)
        if joined:          # boundary inside a speech run (syllable cut at an envelope dip, frame b)
            if pause_rule and last[b] - first[b] + 1 >= min_pause:
                # the dip frame absorbs a pause of the decoder output: the pause stays silent
                bounds[k] = first[b] if k % 2 == 1 else last[b] + 1
            else:
                bounds[k] = int(round(centre[b]))               # centre of the frames DTW pairs with b
        elif k % 2 == 0:    # onset after silence: first frame paired with the first speech frame
            bounds[k] = first[b]
        else:               # offset before silence: after the last frame paired with the last speech frame
            bounds[k] = last[b - 1] + 1
    bounds = np.maximum.accumulate(np.clip(bounds, 0, n_dst))

    if m["snap"] and dst_segs is not None and len(dst_segs):
        tol = ms_to_frames(m["snap_tolerance_ms"], cfg)
        cand = (np.asarray(dst_segs)[:, 0], np.asarray(dst_segs)[:, 1])
        for k in range(len(bounds)):
            lo = bounds[k - 1] if k else 0
            later = np.flatnonzero(src_b[k + 1:] > src_b[k])   # next boundary at a later source frame
            hi = bounds[k + 1 + later[0]] if later.size else n_dst
            c = cand[k % 2]
            ok = c[(np.abs(c - bounds[k]) <= tol) & (c >= lo) & (c <= hi)]
            if ok.size:
                bounds[k] = ok[np.argmin(np.abs(ok - bounds[k]))]

    out = bounds.reshape(-1, 2).copy()
    min_len = int(m["min_syllable_frames"])
    prev = 0
    for k in range(len(out)):                                     # keep order, give every syllable room
        on = max(out[k, 0], prev)
        off = max(out[k, 1], on + min_len)
        if off > n_dst:
            off, on = n_dst, max(prev, n_dst - min_len)
        out[k] = on, off
        prev = off
    return out


def to_compressed(segs: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Segments on the full timeline -> indices among the kept frames ``keep`` (sorted frame numbers)."""
    segs = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    return np.stack([np.searchsorted(keep, segs[:, 0]), np.searchsorted(keep, segs[:, 1])], axis=1)


def to_absolute(segs_c: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Inverse of ``to_compressed``: onset -> its kept frame, offset -> one past the last kept frame."""
    segs_c = np.asarray(segs_c, dtype=np.int64).reshape(-1, 2)
    on = keep[np.clip(segs_c[:, 0], 0, keep.size - 1)]
    off = keep[np.clip(segs_c[:, 1] - 1, 0, keep.size - 1)] + 1
    return np.stack([on, np.maximum(off, on + 1)], axis=1)


def map_syllables(
    env_src: np.ndarray, env_dst: np.ndarray, src_segs: np.ndarray, dst_segs: np.ndarray, cfg: Mapping[str, Any]
) -> tuple[np.ndarray, float]:
    """DTW + mapping of source syllables onto the destination timeline (steps 3-4 of the docstring).

    With ``dtw.speech_only`` the DTW runs on the speech frames of each envelope only (silence removed,
    as in the paper's evaluation metric) and the result is mapped back: a TTS spoken as one connected
    phrase is then matched to speech produced one word at a time without having to place the pauses,
    and every pause of the decoder output becomes a boundary between syllables.
    """
    n_dst = int(np.asarray(env_dst).size)
    if not cfg["alignment"]["dtw"]["speech_only"]:
        path, cost = dtw_path(env_src, env_dst, cfg)
        return map_segments(src_segs, path, n_dst, dst_segs, cfg), cost
    keep_s = np.flatnonzero(speech_mask(env_src, cfg))
    keep_d = np.flatnonzero(speech_mask(env_dst, cfg))
    if keep_s.size < 2 or keep_d.size < 2:                     # nothing to match: fall back to full DTW
        path, cost = dtw_path(env_src, env_dst, cfg)
        return map_segments(src_segs, path, n_dst, dst_segs, cfg), cost
    path, cost = dtw_path(env_src[keep_s], env_dst[keep_d], cfg,
                          silence_threshold(env_src, cfg), silence_threshold(env_dst, cfg))
    dst_c = to_compressed(dst_segs, keep_d) if len(dst_segs) else None
    mapped_c = map_segments(to_compressed(src_segs, keep_s), path, keep_d.size, dst_c, cfg, pause_rule=False)
    return to_absolute(mapped_c, keep_d), cost


# --------------------------------------------------------------------------------- stretch


def _resample(seg: np.ndarray, n_out: int, interp: str) -> np.ndarray:
    n_in = seg.shape[0]
    pos = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5            # centre-aligned frame positions
    if interp == "nearest":
        return seg[np.clip(np.round(pos).astype(int), 0, n_in - 1)]
    if interp == "linear":
        pos = np.clip(pos, 0, n_in - 1)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, n_in - 1)
        f = (pos - i0)[:, None]
        return (1 - f) * seg[i0] + f * seg[i1]
    raise ValueError(f"unknown interp {interp!r}")


def _pieces(src_segs: np.ndarray, dst_segs: np.ndarray, gap_fill: str):
    """Source -> destination intervals to copy: syllables, plus inter-syllable gaps if gap_fill='tts'."""
    pairs = [(tuple(s), tuple(d)) for s, d in zip(src_segs, dst_segs)]
    if gap_fill == "tts":
        for k in range(len(src_segs) - 1):
            pairs.append(((src_segs[k, 1], src_segs[k + 1, 0]), (dst_segs[k, 1], dst_segs[k + 1, 0])))
    elif gap_fill != "silence":
        raise ValueError(f"unknown gap_fill {gap_fill!r}")
    return [(s, d) for s, d in pairs if s[1] > s[0] and d[1] > d[0]]


def stretch_features(
    src: np.ndarray, src_segs: np.ndarray, dst_segs: np.ndarray, n_out: int, fill: np.ndarray,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """(n_out, d) features: each source segment resampled into its destination, ``fill`` elsewhere."""
    st = cfg["alignment"]["stretch"]
    src = np.asarray(src, dtype=np.float64)
    out = np.tile(np.asarray(fill, dtype=np.float64), (n_out, 1))
    filled = np.zeros(n_out, dtype=bool)
    for (a, b), (c, e) in _pieces(np.asarray(src_segs), np.asarray(dst_segs), st["gap_fill"]):
        out[c:e] = _resample(src[a:b], e - c, st["interp"])
        filled[c:e] = True
    mode = st["silence_pitch"]
    if mode == "hold":
        out[:, PITCH] = hold_pitch(out[:, PITCH], filled)
    elif mode != "code":
        raise ValueError(f"unknown silence_pitch {mode!r}")
    return out.astype(np.float32)


def hold_pitch(pitch: np.ndarray, speech: np.ndarray) -> np.ndarray:
    """Non-speech frames take the pitch of the last speech frame before them (unchanged before the first)."""
    pitch = np.asarray(pitch, dtype=np.float64).copy()
    idx = np.where(speech, np.arange(pitch.size), -1)
    last = np.maximum.accumulate(idx)
    m = (~np.asarray(speech, dtype=bool)) & (last >= 0)
    pitch[m] = pitch[last[m]]
    return pitch


def stretch_audio(
    wav: np.ndarray, src_segs: np.ndarray, dst_segs: np.ndarray, n_out: int, cfg: Mapping[str, Any]
) -> np.ndarray:
    """Waveform of n_out frames: each source segment phase-vocoder stretched into place, zeros elsewhere."""
    import librosa

    st = cfg["alignment"]["stretch"]
    spf = int(cfg["synthesis"]["samples_per_frame"])
    wav = np.asarray(wav, dtype=np.float32)
    out = np.zeros(n_out * spf, dtype=np.float32)
    for (a, b), (c, e) in _pieces(np.asarray(src_segs), np.asarray(dst_segs), st["gap_fill"]):
        seg, n = wav[a * spf:b * spf], (e - c) * spf
        n_fft = min(int(st["audio_n_fft"]), 2 ** int(np.floor(np.log2(max(seg.size, 2)))))
        y = librosa.effects.time_stretch(seg, rate=seg.size / n, n_fft=n_fft)
        out[c * spf:e * spf] = librosa.util.fix_length(y, size=n)
    return out


# ------------------------------------------------------------------------------ one trial


@dataclass
class Alignment:
    target_raw: np.ndarray        # (T, 20) raw LPCNet features on the neural timeline
    pred_env: np.ndarray          # (T,) dB
    tts_env: np.ndarray           # (n_tts,) dB
    pred_segments: np.ndarray     # (J, 2) syllables detected in the decoder output
    tts_segments: np.ndarray      # (K, 2) syllables detected in the TTS
    mapped_segments: np.ndarray   # (K, 2) TTS syllables on the neural timeline
    dtw_cost: float

    def summary(self) -> dict:
        return {
            "n_pred_syllables": int(len(self.pred_segments)),
            "n_tts_syllables": int(len(self.tts_segments)),
            "pred_segments": self.pred_segments.tolist(),
            "tts_segments": self.tts_segments.tolist(),
            "mapped_segments": self.mapped_segments.tolist(),
            "dtw_cost": self.dtw_cost,
        }


def align_trial(
    pred_raw: np.ndarray, tts_raw: np.ndarray, silence: np.ndarray, cfg: Mapping[str, Any],
    synth: Optional[Synth] = None, encode: Optional[Encode] = None, pred_env: Optional[np.ndarray] = None,
) -> Alignment:
    """Decoder output on the neural timeline (T, 20) + TTS frames (n, 20) -> aligned raw target (T, 20).

    ``pred_env`` (T,) replaces the envelope of ``pred_raw`` (e.g. the envelope of their closed-loop
    audio); ``pred_raw`` then only sets T.
    """
    pred_raw = np.asarray(pred_raw, dtype=np.float32)
    tts_raw = np.asarray(tts_raw, dtype=np.float32)
    n_out = pred_raw.shape[0]
    env_p = envelope_db(pred_raw, cfg, synth) if pred_env is None else np.asarray(pred_env, np.float64)
    if env_p.shape != (n_out,):
        raise ValueError(f"decoder envelope {env_p.shape} does not match {n_out} frames")
    env_t = envelope_db(tts_raw, cfg, synth)
    seg_p = segment_syllables(env_p, cfg)
    seg_t = segment_syllables(env_t, cfg)
    if not len(seg_t):
        raise ValueError("no syllables found in the TTS envelope")
    mapped, cost = map_syllables(env_t, env_p, seg_t, seg_p, cfg)

    domain = cfg["alignment"]["stretch"]["domain"]
    if domain == "features":
        target = stretch_features(tts_raw, seg_t, mapped, n_out, silence, cfg)
    elif domain == "audio":
        if synth is None or encode is None:
            raise ValueError("stretch domain 'audio' needs synthesis and encode callables")
        enc = np.asarray(encode(stretch_audio(synth(tts_raw), seg_t, mapped, n_out, cfg)), np.float32)
        enc = enc[:n_out, : tts_raw.shape[1]]
        target = np.concatenate([enc, np.tile(silence, (n_out - enc.shape[0], 1))]).astype(np.float32)
    else:
        raise ValueError(f"unknown stretch domain {domain!r}")
    return Alignment(target, env_p, env_t, seg_p, seg_t, mapped, cost)


# --------------------------------------------------------------------------- evaluation aids


def speech_bounds(raw: np.ndarray, cfg: Mapping[str, Any], threshold_db: float) -> Optional[tuple[int, int]]:
    """[first, last + 1) frame whose c0 power is within ``threshold_db`` of the loudest frame."""
    db = c0_to_db(np.asarray(raw)[:, 0], cfg)
    idx = np.flatnonzero(db > db.max() - float(threshold_db))
    return (int(idx[0]), int(idx[-1]) + 1) if idx.size else None


def boundary_errors(ours: np.ndarray, theirs: np.ndarray) -> np.ndarray:
    """For every reference boundary, the signed distance (frames) to the nearest of ours."""
    ours, theirs = np.asarray(ours, float).ravel(), np.asarray(theirs, float).ravel()
    if not ours.size or not theirs.size:
        return np.full(theirs.size, np.nan)
    d = ours[None, :] - theirs[:, None]
    return d[np.arange(theirs.size), np.argmin(np.abs(d), axis=1)]
