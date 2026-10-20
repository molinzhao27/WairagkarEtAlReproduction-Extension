"""Objective metrics for synthesized speech, following Wairagkar et al. 2025 (Nature).

Exact definitions, the paper's wording, sources, and every ambiguity with the choice made are
in ``notes/METRICS.md``. Every tunable lives in ``configs/metrics.yaml`` (section ``metrics``).

Public API (``ref`` = target speech, ``est`` = synthesized speech; 1-D real arrays, same ``sr``):

- ``mel_spectrogram_correlation(ref, est, sr, cfg) -> float``: mean over mel bands of the Pearson
  r between dB mel spectrograms, after independent silence removal and DTW (paper's headline metric).
- ``mel_band_correlations(ref, est, sr, cfg) -> np.ndarray``: the per-band r before averaging.
- ``mel_cepstral_distortion(ref, est, sr, cfg) -> float``: Kubichek-style MCD in dB on
  DTW-aligned mel-cepstra, silences removed.
- ``speech_frame_mask(wav, sr, cfg) -> np.ndarray``: the silence detector both metrics use.
- ``word_error_rate`` / ``char_error_rate`` / ``normalize_text`` / ``edit_distance``: text metrics
  for a later ASR extension (the paper reports no ASR-based WER).
- ``evaluate(ref_wav, est_wav, sr, cfg) -> dict[str, float]``: everything above that applies to a
  pair of waveforms, sharing one STFT per signal.

``cfg`` may be None (use configs/metrics.yaml), a full config holding a ``metrics`` section, or the
``metrics`` section itself. Whatever is given is deep-merged over configs/metrics.yaml, so partial
overrides such as ``{"mel_corr": {"dtw": False}}`` work. Nothing here is random.
"""

from __future__ import annotations

import functools
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import scipy.fft
import scipy.signal

from wvx.config import REPO_ROOT, deep_merge, load_yaml, resolve_path

METRICS_CONFIG = REPO_ROOT / "configs" / "metrics.yaml"

ConfigLike = Optional[Mapping[str, Any]]
TextLike = Union[str, Sequence[str]]


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


def load_metrics_config(
    path: Optional[Union[str, Path]] = None, overrides: ConfigLike = None
) -> dict:
    """Return the ``metrics`` section of configs/metrics.yaml (or of ``path``), plus overrides."""
    data = load_yaml(resolve_path(path) if path is not None else METRICS_CONFIG)
    cfg = data.get("metrics", data)
    return deep_merge(cfg, overrides) if overrides else cfg


@functools.lru_cache(maxsize=1)
def _default_cfg() -> dict:
    return load_metrics_config()


def _cfg(cfg: ConfigLike) -> dict:
    """Resolve ``cfg`` (None / full config / metrics section) to a complete metrics section."""
    base = _default_cfg()
    if cfg is None:
        return deep_merge(base, {})
    if "metrics" in cfg:
        return deep_merge(base, cfg["metrics"])
    if set(cfg) & set(base):
        return deep_merge(base, cfg)
    return deep_merge(base, {})  # a full config without a metrics section


# --------------------------------------------------------------------------------------------
# Signal front end (shared by both spectral metrics)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Analysis:
    power: np.ndarray  # STFT power, (n_freq, n_frames)
    speech: np.ndarray  # bool, (n_frames,): True = kept, False = silence


def _as_signal(x: Any, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D waveform, got shape {x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} is empty")
    if np.iscomplexobj(x) or not np.issubdtype(x.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array, got dtype {x.dtype}")
    x = x.astype(np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or inf")
    return x


def _resample(x: np.ndarray, sr: float, cfg: dict) -> np.ndarray:
    target = int(cfg["sample_rate"])
    if sr == target:
        return x
    sr_int = int(round(sr))
    if sr <= 0 or sr_int != sr:
        raise ValueError(f"sample rate must be a positive integer, got {sr}")
    g = math.gcd(target, sr_int)
    window = ("kaiser", float(cfg["resample"]["kaiser_beta"]))
    return scipy.signal.resample_poly(x, target // g, sr_int // g, window=window)


def _frame_params(cfg: dict) -> Tuple[int, int, int]:
    """(n_fft, win_length, hop_length) in samples at cfg['sample_rate']."""
    sr = cfg["sample_rate"]
    st = cfg["stft"]
    win = int(round(st["win_ms"] * sr / 1000.0))
    hop = int(round(st["hop_ms"] * sr / 1000.0))
    n_fft = int(st["n_fft"]) if st["n_fft"] is not None else win
    if n_fft < win:
        raise ValueError(f"stft.n_fft ({n_fft}) must be >= the window length ({win})")
    return n_fft, win, hop


def _power_spectrogram(x: np.ndarray, cfg: dict) -> np.ndarray:
    n_fft, win, hop = _frame_params(cfg)
    st = cfg["stft"]
    spec = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=st["window"],
        center=st["center"],
        pad_mode=st["pad_mode"],
    )
    return np.abs(spec) ** 2


def _fill_short_silences(speech: np.ndarray, min_run: int) -> np.ndarray:
    """Relabel internal silent runs shorter than ``min_run`` frames as speech.

    Leading and trailing silence is never refilled (it is not "between words").
    """
    out = speech.copy()
    idx = np.flatnonzero(speech)
    if idx.size == 0:
        return out
    first, last = idx[0], idx[-1]
    t = first
    while t <= last:
        if out[t]:
            t += 1
            continue
        start = t
        while t <= last and not speech[t]:
            t += 1
        if t - start < min_run:
            out[start:t] = True
    return out


def _speech_mask_from_power(power: np.ndarray, cfg: dict) -> np.ndarray:
    s = cfg["silence"]
    if s["method"] != "relative_energy":
        raise ValueError(f"unknown silence.method {s['method']!r}")
    energy = power.sum(axis=0)
    floor = float(s["amin"])
    above_floor = energy > floor
    if not above_floor.any():
        return np.zeros(energy.shape, dtype=bool)
    energy_db = 10.0 * np.log10(np.maximum(energy, floor))
    speech = above_floor & (energy_db >= energy_db.max() - float(s["threshold_db"]))
    min_run = int(round(float(s["min_silence_ms"]) / cfg["stft"]["hop_ms"]))
    if min_run > 1:
        speech = _fill_short_silences(speech, min_run)
    return speech


def _analyze(x: Any, sr: float, cfg: dict, name: str) -> _Analysis:
    x = _resample(_as_signal(x, name), sr, cfg)
    power = _power_spectrogram(x, cfg)
    return _Analysis(power=power, speech=_speech_mask_from_power(power, cfg))


def _mel_db(power: np.ndarray, n_mels: int, cfg: dict) -> np.ndarray:
    """dB mel power spectrogram, (n_mels, n_frames)."""
    n_fft, _, _ = _frame_params(cfg)
    m = cfg["mel"]
    fb = librosa.filters.mel(
        sr=cfg["sample_rate"],
        n_fft=n_fft,
        n_mels=int(n_mels),
        fmin=m["fmin"],
        fmax=m["fmax"],
        htk=m["htk"],
        norm=m["norm"],
        dtype=np.float64,
    )
    d = cfg["db"]
    return librosa.power_to_db(fb @ power, ref=d["ref"], amin=d["amin"], top_db=d["top_db"])


def _norm_for_dtw(x: np.ndarray, mode: str) -> np.ndarray:
    """Per-row normalization of a (D, T) feature matrix, used only to choose the DTW path."""
    if mode == "none":
        return x
    centred = x - x.mean(axis=1, keepdims=True)
    if mode == "mean":
        return centred
    if mode == "zscore":
        std = centred.std(axis=1, keepdims=True)
        return centred / np.where(std > 0, std, 1.0)
    raise ValueError(f"unknown dtw_feature_norm {mode!r}")


def _align(
    a: np.ndarray, b: np.ndarray, use_dtw: bool, cfg: dict, feature_norm: str = "none"
) -> Tuple[np.ndarray, np.ndarray]:
    """Frame index pairs (ia, ib) aligning feature matrices a (D, Ta) and b (D, Tb).

    With DTW, the path is computed on ``_norm_for_dtw(a)`` vs ``_norm_for_dtw(b)``.
    """
    if use_dtw:
        d = cfg["alignment"]["dtw"]
        _, wp = librosa.sequence.dtw(
            X=_norm_for_dtw(a, feature_norm),
            Y=_norm_for_dtw(b, feature_norm),
            metric=d["metric"],
            step_sizes_sigma=np.asarray(d["step_sizes"], dtype=np.uint32),
            weights_add=np.asarray(d["weights_add"], dtype=np.float64),
            weights_mul=np.asarray(d["weights_mul"], dtype=np.float64),
            global_constraints=d["global_constraints"],
            band_rad=d["band_rad"],
        )
        wp = wp[::-1]  # librosa returns the path end -> start
        return wp[:, 0], wp[:, 1]
    mode = cfg["alignment"]["without_dtw"]
    if mode == "error" and a.shape[1] != b.shape[1]:
        raise ValueError(f"lengths differ ({a.shape[1]} vs {b.shape[1]} frames) and DTW is off")
    if mode not in ("error", "truncate"):
        raise ValueError(f"unknown alignment.without_dtw {mode!r}")
    idx = np.arange(min(a.shape[1], b.shape[1]))
    return idx, idx


def _select_frames(
    feats: np.ndarray, analysis: _Analysis, remove_silence: bool
) -> np.ndarray:
    return feats[:, analysis.speech] if remove_silence else feats


def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between matching rows; NaN where either row is constant."""
    ok = (np.ptp(a, axis=1) > 0) & (np.ptp(b, axis=1) > 0)
    r = np.full(a.shape[0], np.nan)
    if not ok.any():
        return r
    a0 = a[ok] - a[ok].mean(axis=1, keepdims=True)
    b0 = b[ok] - b[ok].mean(axis=1, keepdims=True)
    num = (a0 * b0).sum(axis=1)
    den = np.sqrt((a0 * a0).sum(axis=1) * (b0 * b0).sum(axis=1))
    r[ok] = np.clip(num / den, -1.0, 1.0)
    return r


# --------------------------------------------------------------------------------------------
# Mel-spectrogram correlation
# --------------------------------------------------------------------------------------------


def _band_corr(
    ref: _Analysis, est: _Analysis, cfg: dict, remove_silence: bool, use_dtw: bool
) -> np.ndarray:
    n_mels = int(cfg["mel_corr"]["n_mels"])
    r_db = _select_frames(_mel_db(ref.power, n_mels, cfg), ref, remove_silence)
    e_db = _select_frames(_mel_db(est.power, n_mels, cfg), est, remove_silence)
    if min(r_db.shape[1], e_db.shape[1]) < int(cfg["min_frames"]):
        return np.full(n_mels, np.nan)
    ia, ib = _align(r_db, e_db, use_dtw, cfg, cfg["mel_corr"]["dtw_feature_norm"])
    return _rowwise_pearson(r_db[:, ia], e_db[:, ib])


def _aggregate_bands(r: np.ndarray, cfg: dict) -> float:
    policy = cfg["mel_corr"]["constant_band"]
    if policy not in ("skip", "nan"):
        raise ValueError(f"unknown mel_corr.constant_band {policy!r}")
    if np.isnan(r).all() or (policy == "nan" and np.isnan(r).any()):
        return float("nan")
    return float(np.nanmean(r))


def mel_band_correlations(ref: Any, est: Any, sr: float, cfg: ConfigLike = None) -> np.ndarray:
    """Per-mel-band Pearson r between target and synthesized speech (NaN = undefined band).

    Settings come from ``cfg['mel_corr']`` (n_mels, remove_silence, dtw) and the shared front end.
    """
    c = _cfg(cfg)
    m = c["mel_corr"]
    return _band_corr(
        _analyze(ref, sr, c, "ref"), _analyze(est, sr, c, "est"), c, m["remove_silence"], m["dtw"]
    )


def mel_spectrogram_correlation(ref: Any, est: Any, sr: float, cfg: ConfigLike = None) -> float:
    """Mean over mel bands of the Pearson r between dB mel spectrograms (paper: 40 bands).

    Silences are removed from each signal independently, the remaining frames are DTW-aligned,
    and r is computed per band over the aligned frame pairs, then averaged (notes/METRICS.md §2).
    Returns NaN if either signal has fewer than ``min_frames`` speech frames.
    """
    c = _cfg(cfg)
    m = c["mel_corr"]
    r = _band_corr(
        _analyze(ref, sr, c, "ref"), _analyze(est, sr, c, "est"), c, m["remove_silence"], m["dtw"]
    )
    return _aggregate_bands(r, c)


# --------------------------------------------------------------------------------------------
# Mel-cepstral distortion
# --------------------------------------------------------------------------------------------


def _mcd_constant(value: Union[str, float]) -> float:
    if value == "kubichek":
        return 10.0 * math.sqrt(2.0) / math.log(10.0)
    return float(value)


def _mel_cepstra(power: np.ndarray, cfg: dict) -> np.ndarray:
    """Mel-cepstral coefficients used by MCD, (n_used, n_frames) (c0 dropped unless include_c0).

    log mel spectrum (natural log of amplitude or power, with the shared dB floor) -> DCT-II over
    the mel axis. ``cepstrum_scale: sptk`` scales it to the one-sided cepstrum convention
    (log|X| = c0 + sum_d c_d cos(d*w)) that Kubichek's 10*sqrt(2)/ln(10) constant assumes;
    ``ortho`` is the orthonormal DCT used by librosa-style MFCCs.
    """
    m = cfg["mcd"]
    n_mels, n_ceps = int(m["n_mels"]), int(m["n_ceps"])
    if not 0 < n_ceps <= n_mels:
        raise ValueError(f"mcd.n_ceps must be in 1..n_mels ({n_mels}), got {n_ceps}")
    log_power = _mel_db(power, n_mels, cfg) * (math.log(10.0) / 10.0)  # ln P
    if m["log_spectrum"] == "amplitude":
        log_spec = 0.5 * log_power  # ln|X| = ln(P) / 2
    elif m["log_spectrum"] == "power":
        log_spec = log_power
    else:
        raise ValueError(f"unknown mcd.log_spectrum {m['log_spectrum']!r}")
    scale = m["cepstrum_scale"]
    if scale == "sptk":
        ceps = scipy.fft.dct(log_spec, type=2, axis=0, norm=None) / n_mels
        ceps[0] /= 2.0  # c0 = mean log spectrum
    elif scale == "ortho":
        ceps = scipy.fft.dct(log_spec, type=2, axis=0, norm="ortho")
    else:
        raise ValueError(f"unknown mcd.cepstrum_scale {scale!r}")
    first = 0 if m["include_c0"] else 1
    return ceps[first:n_ceps]


def _mcd(ref: _Analysis, est: _Analysis, cfg: dict) -> float:
    m = cfg["mcd"]
    c_ref = _select_frames(_mel_cepstra(ref.power, cfg), ref, m["remove_silence"])
    c_est = _select_frames(_mel_cepstra(est.power, cfg), est, m["remove_silence"])
    if min(c_ref.shape[1], c_est.shape[1]) < int(cfg["min_frames"]):
        return float("nan")
    ia, ib = _align(c_ref, c_est, m["dtw"], cfg, m["dtw_feature_norm"])
    diff = c_ref[:, ia] - c_est[:, ib]
    per_pair = _mcd_constant(m["constant"]) * np.sqrt((diff * diff).sum(axis=0))
    return float(per_pair.mean())


def mel_cepstral_distortion(ref: Any, est: Any, sr: float, cfg: ConfigLike = None) -> float:
    """Mel-cepstral distortion in dB (lower is better; 0 for identical signals).

    MCD = mean over DTW path pairs (t, u) of
          (10*sqrt(2)/ln 10) * sqrt( sum_{d=1}^{24} (c_d^ref[t] - c_d^est[u])^2 )
    with 25 mel bands -> 25 cepstral coefficients, c0 excluded, silences removed from each signal
    independently (defaults; notes/METRICS.md §3). Returns NaN if either signal has fewer than
    ``min_frames`` speech frames.
    """
    c = _cfg(cfg)
    return _mcd(_analyze(ref, sr, c, "ref"), _analyze(est, sr, c, "est"), c)


def speech_frame_mask(wav: Any, sr: float, cfg: ConfigLike = None) -> np.ndarray:
    """Boolean per-frame mask (True = speech kept, False = silence removed) for one waveform."""
    c = _cfg(cfg)
    return _analyze(wav, sr, c, "wav").speech


# --------------------------------------------------------------------------------------------
# One call for the inference script
# --------------------------------------------------------------------------------------------


def evaluate(ref_wav: Any, est_wav: Any, sr: float, cfg: ConfigLike = None) -> dict:
    """All waveform metrics for one (target, synthesized) pair, as ``dict[str, float]``.

    Keys:
      mel_corr             paper's metric (40 bands, silence removed, DTW)
      mcd                  mel-cepstral distortion, dB
      mel_corr_framewise   (if cfg evaluate.mel_corr_framewise) no silence removal, no DTW,
                           frame t vs frame t (truncated to the shorter): the preprint's reading
      ref_speech_fraction, est_speech_fraction   share of frames kept by the silence detector
    """
    c = _cfg(cfg)
    ref = _analyze(ref_wav, sr, c, "ref")
    est = _analyze(est_wav, sr, c, "est")
    m = c["mel_corr"]
    out = {
        "mel_corr": _aggregate_bands(_band_corr(ref, est, c, m["remove_silence"], m["dtw"]), c),
        "mcd": _mcd(ref, est, c),
    }
    if c["evaluate"]["mel_corr_framewise"]:
        framewise_cfg = deep_merge(c, {"alignment": {"without_dtw": "truncate"}})
        out["mel_corr_framewise"] = _aggregate_bands(
            _band_corr(ref, est, framewise_cfg, False, False), framewise_cfg
        )
    out["ref_speech_fraction"] = float(ref.speech.mean())
    out["est_speech_fraction"] = float(est.speech.mean())
    return out


# --------------------------------------------------------------------------------------------
# Text metrics (for a later ASR extension; not reported in the paper)
# --------------------------------------------------------------------------------------------


def normalize_text(text: str, cfg: ConfigLike = None) -> str:
    """Normalize a transcript per ``cfg['text']``; returns space-separated tokens."""
    t = _cfg(cfg)["text"]
    s = unicodedata.normalize(t["unicode_form"], text) if t["unicode_form"] else text
    for apostrophe in t["apostrophes"]:
        s = s.replace(apostrophe, "'")
    if t["lowercase"]:
        s = s.lower()
    if t["hyphens_to_space"]:
        s = "".join(" " if unicodedata.category(ch) == "Pd" else ch for ch in s)
    keep = {"'"} if t["keep_apostrophes"] else set()
    if t["remove_punctuation"]:
        s = "".join(ch for ch in s if ch in keep or not unicodedata.category(ch).startswith("P"))
    tokens = s.split()
    if t["remove_punctuation"] and t["keep_apostrophes"]:
        tokens = [tok.strip("'") for tok in tokens]  # quotes, not contractions
    return " ".join(tok for tok in tokens if tok)


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Levenshtein distance (unit-cost substitutions, insertions, deletions) between sequences."""
    a, b = list(a), list(b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _pairs(ref: TextLike, hyp: TextLike):
    if isinstance(ref, str) != isinstance(hyp, str):
        raise TypeError("ref_text and hyp_text must both be strings or both be sequences")
    if isinstance(ref, str):
        return [(ref, hyp)]
    ref, hyp = list(ref), list(hyp)
    if len(ref) != len(hyp):
        raise ValueError(f"{len(ref)} references but {len(hyp)} hypotheses")
    return list(zip(ref, hyp))


def _error_rate(ref: TextLike, hyp: TextLike, tokenize) -> float:
    errors = total = 0
    for r, h in _pairs(ref, hyp):
        r_tok, h_tok = tokenize(r), tokenize(h)
        errors += edit_distance(r_tok, h_tok)
        total += len(r_tok)
    if total == 0:
        raise ValueError("reference is empty after normalization; the error rate is undefined")
    return errors / total


def word_error_rate(ref_text: TextLike, hyp_text: TextLike, cfg: ConfigLike = None) -> float:
    """(S + D + I) / N over normalized word tokens.

    Strings give one pair; equal-length sequences give the corpus rate (total edits / total
    reference words), not the mean of per-sentence rates. Can exceed 1 (insertions).
    """
    c = _cfg(cfg)
    return _error_rate(ref_text, hyp_text, lambda s: normalize_text(s, c).split())


def char_error_rate(ref_text: TextLike, hyp_text: TextLike, cfg: ConfigLike = None) -> float:
    """Character edit distance / reference characters, after the same normalization as WER.

    Single spaces between words count as characters unless ``text.cer_count_spaces`` is false.
    """
    c = _cfg(cfg)
    keep_spaces = c["text"]["cer_count_spaces"]

    def chars(s: str):
        n = normalize_text(s, c)
        return list(n if keep_spaces else n.replace(" ", ""))

    return _error_rate(ref_text, hyp_text, chars)
