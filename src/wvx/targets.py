"""Target speech features: TTS feature files, the silence frame, voice statistics, decoder targets.

The decoder's training target (``Trial.target``) is their ``get_speech_features_from_single_trial``
applied to raw LPCNet features: keep 18 cepstra + pitch period + pitch correlation, min-max scale to
[-1, 1] with ``T15_lpcnet_features_norm_macOS_TTS.json``, Gaussian smoothing along time
(sigma 2 frames). ``decoder_target`` calls exactly that function with the ``wairagkar_data``
settings, so our targets go through the same transform as their released ones.

**Silence.** LPCNet encodes digital zeros as one constant frame: c0 = -12.48528 (= 18 x log10(1e-2)
/ sqrt(18) - 4), cepstra 0, pitch 3.1 (period 255, the encoder's maximum) and correlation -0.5. Their
norm JSON's c0 minimum (-12.48528) and pitch maximum (3.10) are exactly these values, so their TTS
set contained digital silence. Our targets use this frame wherever the alignment puts no speech.
``TTSCorpus.silence`` holds it as encoded by the same toolchain as the TTS frames.

**Voice statistics.** Their JSON also has per-feature mean and s.d. over their TTS set (unused by
their code). That set mixes speech frames of one voice with an unknown share ``w`` of silence
frames. ``fit_silence_mixture`` models their moments as ``(1 - w) * speech + w * silence`` and fits
``w`` per candidate voice; ``voice_distance`` is the remaining mismatch, used to rank voices.

TTS feature files (``TTSCorpus``) hold raw LPCNet frames (n, 36) per sentence, as their
``encode_lpcnet_features_from_wav`` returns them, plus the silence frame and provenance. No audio is
stored anywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from wvx import b2v
from wvx.config import deep_merge, load_config, load_yaml, resolve_path

PathLike = Union[str, Path]

DEFAULT_TARGETS_CONFIG = "configs/experiments/kunz_targets.yaml"


# ------------------------------------------------------------------------------------- config


def load_targets_config(path: PathLike = DEFAULT_TARGETS_CONFIG, overrides: Optional[Mapping] = None) -> dict:
    """default.yaml <- ``base_experiment`` <- this file <- ``overrides`` (paths resolved)."""
    exp = load_yaml(resolve_path(path))
    base = exp.get("base_experiment")
    merged = deep_merge(exp, overrides or {})
    return load_config(base, overrides=merged)


# ---------------------------------------------------------------------------------- TTS files


def voice_slug(voice: str) -> str:
    """Directory-safe voice name: ``"Reed (English (US))"`` -> ``"reed_english_us"``."""
    slug = re.sub(r"[^0-9a-z]+", "_", voice.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot make a slug from voice {voice!r}")
    return slug


def text_key(text: str) -> str:
    """Lookup key for a cue: surrounding whitespace removed, inner runs of whitespace collapsed.

    Their cues are space-padded char matrices; Kunz cues are exact strings. Case and punctuation are
    kept because they change what the TTS says (``?`` gives a rising contour).
    """
    return " ".join(str(text).split())


@dataclass
class TTSCorpus:
    voice: str
    texts: list[str]
    frames: list[np.ndarray]          # per sentence (n_i, 36) float32, raw LPCNet range
    silence: np.ndarray               # (36,) float32: LPCNet frame of digital zeros
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.frames):
            raise ValueError(f"{len(self.texts)} texts but {len(self.frames)} feature arrays")
        keys = [text_key(t) for t in self.texts]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate texts in a TTS corpus")
        self._index = {k: i for i, k in enumerate(keys)}

    def __len__(self) -> int:
        return len(self.texts)

    def __contains__(self, text: str) -> bool:
        return text_key(text) in self._index

    def lookup(self, text: str) -> np.ndarray:
        try:
            return self.frames[self._index[text_key(text)]]
        except KeyError:
            raise KeyError(f"no TTS features for {text!r} (voice {self.voice!r})") from None

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"voice": self.voice, "texts": self.texts, "lengths": [int(f.shape[0]) for f in self.frames],
                  "meta": self.meta}
        np.savez_compressed(
            path,
            header=np.array(json.dumps(header)),
            frames=np.concatenate(self.frames).astype(np.float32),
            silence=np.asarray(self.silence, dtype=np.float32),
        )
        return path

    @classmethod
    def load(cls, path: PathLike) -> "TTSCorpus":
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            flat, silence = z["frames"], z["silence"]
        bounds = np.cumsum([0] + header["lengths"])
        if bounds[-1] != flat.shape[0]:
            raise ValueError(f"{path}: header lengths sum to {bounds[-1]}, file has {flat.shape[0]} frames")
        frames = [flat[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        return cls(voice=header["voice"], texts=header["texts"], frames=frames, silence=silence,
                   meta=header.get("meta", {}))


def corpus_path(cfg: Mapping[str, Any], voice: str, corpus: str) -> Path:
    return Path(cfg["paths"]["tts_features"]) / voice_slug(voice) / f"{corpus}.npz"


# ------------------------------------------------------------------------- feature transforms


def n_targets(cfg: Mapping[str, Any]) -> int:
    return int(cfg["model"]["n_targets"])


def reduce(frames: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """(n, 36|20) raw frames -> (n, 20) cepstra + pitch (their dimensionality reduction)."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] < n_targets(cfg):
        raise ValueError(f"frames shape {frames.shape}, want (n, >= {n_targets(cfg)})")
    if frames.shape[1] == n_targets(cfg):
        return frames
    sp = b2v.load("lib.speech_processor")
    n_ceps = int(cfg["wairagkar_data"]["n_cepstral_coeff"])
    return sp.lpcnet_features_dimensionality_reduction(frames, n_cepstral_coeff=n_ceps).astype(np.float32)


def silence_frame(corpus: TTSCorpus, cfg: Mapping[str, Any]) -> np.ndarray:
    """(20,) raw silence frame of a corpus."""
    return reduce(np.asarray(corpus.silence, dtype=np.float32)[None, :], cfg)[0]


def lpc_minmax(cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from wvx.wairagkar_data import lpc_norm

    return lpc_norm(cfg)


def decoder_target(raw: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Raw LPCNet features (T, 20|36) -> decoder target (T, 20) float32, exactly their transform."""
    dcfg = cfg["wairagkar_data"]
    lo, hi = lpc_minmax(cfg)
    out = b2v.load("lib.data_utils").get_speech_features_from_single_trial(
        np.asarray(raw, dtype=np.float32),
        lo,
        hi,
        n_cepstral_coeff=int(dcfg["n_cepstral_coeff"]),
        smoothing=bool(dcfg["target_smoothing"]),
        smoothing_sigma=float(dcfg["target_smoothing_sigma"]),
    )
    return np.ascontiguousarray(out, dtype=np.float32)


def scale(raw: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Raw (T, 20) -> [-1, 1] scale, no smoothing (their scale_between_minus_one_to_one)."""
    lo, hi = lpc_minmax(cfg)
    return b2v.load("lib.data_utils").scale_between_minus_one_to_one(np.asarray(raw, np.float32), lo, hi)


def prediction_timeline(pred_norm: np.ndarray, n_bins: int, silence20: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Decoder output (n_bins - window_len + 1, 20) in [-1, 1] -> (n_bins, 20) raw, one row per bin.

    Prediction k belongs to bin k + window_len - 1 (their pairing). The first window_len - 1 bins have
    no prediction; they get the silence frame (speech does not start within 0.59 s of the go cue).
    """
    from wvx.synthesis import denormalize

    pred_norm = np.asarray(pred_norm, dtype=np.float32)
    n_front = n_bins - pred_norm.shape[0]
    if pred_norm.ndim != 2 or pred_norm.shape[1] != n_targets(cfg) or n_front < 0:
        raise ValueError(f"predictions {pred_norm.shape} do not fit {n_bins} bins")
    raw = denormalize(pred_norm, cfg).astype(np.float32)
    front = np.tile(np.asarray(silence20, np.float32), (n_front, 1))
    return np.concatenate([front, raw], axis=0)


# ------------------------------------------------------------------------- voice statistics


def is_silence(frames: np.ndarray, silence: np.ndarray, atol: float = 1e-4) -> np.ndarray:
    """Rows equal to the silence frame (digital zeros) on the first 20 features."""
    k = min(frames.shape[1], silence.shape[0], 20)
    return np.all(np.abs(frames[:, :k] - silence[None, :k]) <= atol, axis=1)


@dataclass
class Moments:
    n: int
    mean: np.ndarray       # (20,)
    second: np.ndarray     # (20,) E[x^2]
    min: np.ndarray
    max: np.ndarray

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.second - self.mean ** 2, 0.0))


def moments(frames: np.ndarray) -> Moments:
    x = np.asarray(frames, dtype=np.float64)
    return Moments(n=x.shape[0], mean=x.mean(0), second=(x ** 2).mean(0), min=x.min(0), max=x.max(0))


def mixture_moments(speech: Moments, silence: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and s.d. of (1 - w) * speech + w * point mass at ``silence``."""
    s = np.asarray(silence, dtype=np.float64)
    mean = (1 - w) * speech.mean + w * s
    second = (1 - w) * speech.second + w * s ** 2
    return mean, np.sqrt(np.maximum(second - mean ** 2, 0.0))


def _distance(mean, std, ref_mean, ref_std, idx) -> float:
    z = (mean[idx] - ref_mean[idx]) / ref_std[idx]
    r = np.log(np.maximum(std[idx], 1e-12) / ref_std[idx])
    return float(np.mean(z ** 2 + r ** 2))


def fit_silence_mixture(
    speech: Moments, silence: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray,
    features: Sequence[int], grid: Sequence[float],
) -> dict:
    """Grid-search the silence share w; distance = mean over features of z(mean)^2 + log(sd ratio)^2."""
    start, stop, step = (float(g) for g in grid)
    ws = np.arange(start, stop + step / 2, step)
    idx = np.asarray(features, dtype=int)
    ref_mean, ref_std = np.asarray(ref_mean, float), np.asarray(ref_std, float)
    d = np.array([_distance(*mixture_moments(speech, silence, w), ref_mean, ref_std, idx) for w in ws])
    best = int(np.argmin(d))
    mean, std = mixture_moments(speech, silence, ws[best])
    return {
        "silence_share": float(ws[best]),
        "distance": float(d[best]),
        "mean_z": ((mean - ref_mean) / ref_std).tolist(),
        "sd_log_ratio": np.log(np.maximum(std, 1e-12) / ref_std).tolist(),
    }


def voiced_f0_hz(frames: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """F0 (Hz) of voiced frames (pitch correlation above the configured threshold)."""
    v = cfg["voice_id"]
    voiced = frames[:, 19] > float(v["voiced_corr_min"])
    period = frames[voiced, 18] / float(v["pitch_period_scale"]) + float(v["pitch_period_offset"])
    return float(cfg["synthesis"]["sample_rate"]) / period


def load_norm_json(cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with open(cfg["paths"]["lpc_norm"]) as f:
        d = json.load(f)
    return {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}


def rank_voice(frames_per_sentence: Sequence[np.ndarray], silence: np.ndarray, cfg: Mapping[str, Any]) -> dict:
    """Statistics of one voice's corpus against their norm JSON (see module docstring)."""
    vcfg = cfg["voice_id"]
    allf = np.concatenate([reduce(f, cfg) for f in frames_per_sentence]).astype(np.float64)
    sil = reduce(np.asarray(silence, np.float32)[None, :], cfg)[0].astype(np.float64)
    speech = allf[~is_silence(allf, sil)]
    m = moments(speech)
    ref = load_norm_json(cfg)
    fit = fit_silence_mixture(m, sil, ref["lpcnet_mean"], ref["lpcnet_std"],
                              vcfg["distance_features"], vcfg["silence_share_grid"])
    f0 = voiced_f0_hz(speech, cfg)
    ceps = slice(1, 18)
    span_ratio = (m.max[ceps] - m.min[ceps]) / (ref["lpcnet_max"][ceps] - ref["lpcnet_min"][ceps])
    return {
        "n_frames_speech": int(m.n),
        "median_voiced_f0_hz": float(np.median(f0)) if f0.size else float("nan"),
        "voiced_fraction": float(f0.size / max(m.n, 1)),
        **fit,
        "c0_speech_mean": float(m.mean[0]),
        "c0_speech_max": float(m.max[0]),
        "cepstral_span_log_ratio_mean_abs": float(np.mean(np.abs(np.log(span_ratio)))),
        "fraction_outside_json_range": float(np.mean((speech < ref["lpcnet_min"]) | (speech > ref["lpcnet_max"]))),
    }


# ------------------------------------------------------------------------- whole sessions


def lpcnet_callables(cfg: Mapping[str, Any]):
    """(synth, encode) for ``alignment.align_trial``: LPCNet synthesis of raw features, and LPCNet
    encoding of a 16 kHz waveform (temp files under ``synthesis.tmp_root``)."""
    import tempfile

    from wvx import synthesis

    def synth(raw):
        return synthesis.features_to_waveform(raw, cfg)

    def encode(wav):
        root = cfg["synthesis"].get("tmp_root")
        if root is not None:
            Path(resolve_path(root)).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=resolve_path(root) if root is not None else None) as d:
            return synthesis.encode_wav(synthesis.write_wav(Path(d) / "stretched.wav", wav, cfg), cfg)

    return synth, encode


def align_trials(model, trials: Sequence, corpus: TTSCorpus, cfg: Mapping[str, Any], reorder: bool,
                 log=print, preds: Optional[Sequence[np.ndarray]] = None,
                 tts_segments: Optional[Sequence[Optional[np.ndarray]]] = None,
                 pred_segments: Optional[Sequence[Optional[np.ndarray]]] = None,
                 arm: str = "A") -> tuple[list[np.ndarray], list]:
    """Decoder predictions and alignments for trials (anything with ``.neural`` and ``.text``).

    Predictions run in this process (TensorFlow); alignments run in ``alignment.n_workers`` threads,
    each calling LPCNet as a subprocess. Returns (normalized predictions, ``alignment.Alignment``s).
    ``preds``: normalized predictions computed earlier (``model`` is then unused). ``tts_segments`` /
    ``pred_segments``: per trial, syllables that replace the envelope-dip ones (``align_trial``); None, or
    a None entry, keeps the envelope dips. ``arm``: "A" = ``wvx.alignment.align_trial`` (D-027 mapping),
    "B" = ``wvx.syllables.align_trial_segment_driven`` (decoder syllables define the spans; D-033).
    """
    from concurrent.futures import ThreadPoolExecutor

    from wvx import inference
    from wvx.alignment import align_trial
    from wvx.syllables import align_trial_segment_driven

    if arm not in ("A", "B"):
        raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")

    sil20 = silence_frame(corpus, cfg)
    missing = sorted({t.text for t in trials if t.text not in corpus})
    if missing:
        raise KeyError(f"{len(missing)} cues have no TTS features in {corpus.voice!r}, e.g. {missing[:3]}")
    for name, seq in (("preds", preds), ("tts_segments", tts_segments), ("pred_segments", pred_segments)):
        if seq is not None and len(seq) != len(trials):
            raise ValueError(f"{name} has {len(seq)} entries for {len(trials)} trials")
    if preds is None:
        preds = []
        for i, t in enumerate(trials):
            preds.append(inference.predict_trial(model, t.neural, cfg, reorder=reorder))
            if log and (i + 1) % 25 == 0:
                log(f"predicted {i + 1}/{len(trials)} trials")
    else:
        preds = [np.asarray(p, dtype=np.float32) for p in preds]

    synth, encode = lpcnet_callables(cfg)

    def one(i):
        t = trials[i]
        pred_raw = prediction_timeline(preds[i], t.neural.shape[0], sil20, cfg)
        segs = {"tts_segments": None if tts_segments is None else tts_segments[i],
                "pred_segments": None if pred_segments is None else pred_segments[i]}
        tts = reduce(corpus.lookup(t.text), cfg)
        if arm == "B":
            return align_trial_segment_driven(pred_raw, tts, sil20, cfg, synth=synth, **segs)
        return align_trial(pred_raw, tts, sil20, cfg, synth=synth, encode=encode, **segs)

    with ThreadPoolExecutor(max_workers=int(cfg["alignment"]["n_workers"])) as ex:
        results = list(ex.map(one, range(len(trials))))
    return preds, results


def speech_moments(frames_per_utterance: Sequence[np.ndarray], cfg: Mapping[str, Any], mask_db: float) -> dict:
    """Moments of the speech frames (c0 power within ``mask_db`` of each utterance's loudest frame).

    Used to compare a TTS voice with their released targets (their TTS voice, time-stretched): the
    same rule picks speech frames in both, whatever their silence convention.
    """
    from wvx.alignment import c0_to_db

    sp = []
    for f in frames_per_utterance:
        r = reduce(f, cfg).astype(np.float64)
        db = c0_to_db(r[:, 0], cfg)
        sp.append(r[db > db.max() - float(mask_db)])
    x = np.concatenate(sp)
    m = moments(x)
    f0 = voiced_f0_hz(x, cfg)
    return {"n": int(m.n), "mean": m.mean.tolist(), "std": m.std.tolist(),
            "voiced_fraction": float(f0.size / max(m.n, 1)),
            "f0_percentiles_hz": {str(q): float(np.percentile(f0, q)) for q in (10, 25, 50, 75, 90)} if f0.size else {}}


def reference_distance(ours: dict, ref: dict, features: Sequence[int]) -> dict:
    """Distance between two ``speech_moments`` results (z of means by the reference s.d., log s.d. ratio)."""
    idx = np.asarray(features, dtype=int)
    m_o, s_o = np.asarray(ours["mean"]), np.asarray(ours["std"])
    m_r, s_r = np.asarray(ref["mean"]), np.asarray(ref["std"])
    z = (m_o - m_r) / s_r
    r = np.log(s_o / s_r)
    f0_o, f0_r = ours["f0_percentiles_hz"].get("50"), ref["f0_percentiles_hz"].get("50")
    return {"distance": float(np.mean(z[idx] ** 2 + r[idx] ** 2)), "mean_z": z.tolist(), "sd_log_ratio": r.tolist(),
            "median_f0_semitones_vs_ref": float(12 * np.log2(f0_o / f0_r)) if f0_o and f0_r else None}


def decoder_output_sanity(trials: Sequence, results: Sequence, cfg: Mapping[str, Any]) -> dict:
    """Ground-truth-free signs that the decoder output is speech-like (used to compare channel orderings).

    T15 attempted one word at a time, so syllables detected in a meaningful decoder output should track
    the length of the cue across trials; an output unrelated to the cue should not.
    """
    from wvx import alignment as al

    words = np.array([len(t.text.split()) for t in trials], float)
    det = np.array([len(r.pred_segments) for r in results], float)
    tts = np.array([len(r.tts_segments) for r in results], float)
    w = int(cfg["model"]["window_len"])
    rng = np.array([np.percentile(r.pred_env[w - 1:], 95) - np.percentile(r.pred_env[w - 1:], 5) for r in results])
    speech = np.array([al.speech_mask(r.pred_env, cfg)[w - 1:].mean() for r in results])

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None

    return {
        "r_detected_syllables_vs_cue_words": corr(det, words),
        "r_detected_syllables_vs_tts_syllables": corr(det, tts),
        "detected_syllables_per_trial_median": float(np.median(det)),
        "fraction_trials_without_speech": float(np.mean(det == 0)),
        "envelope_range_db_p95_minus_p5_median": float(np.median(rng)),
        "speech_fraction_median": float(np.median(speech)),
    }
