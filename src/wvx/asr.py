"""ASR scoring of synthesized speech: Whisper (faster-whisper / CTranslate2, CPU, int8) -> WER, CER, PER.

Extension metric (notes/PLAN.md Phase 3 d). Wairagkar et al. report no ASR-based WER; their WER/PER come
from human listeners, so nothing computed here is comparable to the paper. Every setting lives in
``configs/asr.yaml`` (section ``asr``); ``cfg`` arguments follow wvx.metrics: None (that file), a full config
holding an ``asr`` section, or the section itself, deep-merged over the file.

Public API
- ``transcribe(wav, sr, cfg) -> str``: raw Whisper text, UNCONSTRAINED decoding (no prompt, no LM).
- ``evaluate_wav(wav, sr, reference_text, cfg) -> dict``: transcribe, then ``score_text``. Adds the
  transcript diagnostics (``avg_logprob``, ``no_speech_prob``, ``n_segments``) and timing (``audio_s``, ``asr_s``).
- ``score_text(reference_text, hypothesis_text, cfg, constrained_hypothesis_text=None) -> dict``: no model needed.
  Keys: ``ref``, ``hyp``, ``hyp_constrained`` (normalized token strings); ``wer``, ``cer``, ``per`` (unconstrained)
  and ``wer_constrained``, ``cer_constrained``, ``per_constrained``; the raw counts behind each rate
  (``ref_words``, ``word_errors``, ``word_errors_constrained``, ``ref_chars``, ..., ``ref_phones``, ...) so a caller
  can pool rates over trials (sum of errors / sum of reference units); ``ref_oov_words``.
- ``normalize(text, cfg) -> str``, ``snap_to_vocabulary(tokens, vocabulary, cfg) -> list``,
  ``number_to_words(n) -> str``, ``phonemes(text, cfg) -> list``, ``vocabulary(cfg) -> list``,
  ``load_asr_config(path, overrides) -> dict``.

Definitions
- Text normalization (reference and hypothesis alike): digits -> words, then ``wvx.metrics.normalize_text`` with
  configs/metrics.yaml ``text`` (NFKC, lowercase, hyphens -> space, punctuation removed except in-word
  apostrophes), then the whole-token rewrites in ``text.replacements`` (spelling variants such as ok -> okay).
- WER = word edit distance / reference words; CER = character edit distance / reference characters (spaces
  between words counted per metrics.yaml ``text.cer_count_spaces``); PER = ARPAbet phoneme edit distance /
  reference phonemes (g2p_en, stress stripped, word boundaries ignored). All can exceed 1 (insertions).
- UNCONSTRAINED: the normalized Whisper output as is.
- CONSTRAINED (closed 50-word vocabulary, ``constrained.vocabulary``), primary method ``snap``: every word of the
  unconstrained hypothesis is replaced by the nearest vocabulary word (character Levenshtein distance; ties ->
  smaller length difference -> alphabetical). In-vocabulary words are untouched and no word is added or removed
  (unless ``snap.max_normalized_distance`` is set), so snapping can only change substitutions. Alternatives:
  ``prompt`` (decode again with the vocabulary as Whisper's initial prompt) and ``prompt_snap`` (both). When the
  reference itself has out-of-vocabulary words the constrained scores are NaN (``oov_reference: nan``).

Heavy dependencies (faster_whisper, g2p_en/nltk) are imported lazily, so the rest of the repo and the pure text
functions work without them. Model files and NLTK data live inside the repo (``data/processed/models``).
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import re
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from wvx import metrics
from wvx.config import REPO_ROOT, deep_merge, load_yaml, resolve_path

ASR_CONFIG = REPO_ROOT / "configs" / "asr.yaml"
ConfigLike = Optional[Mapping[str, Any]]
CONSTRAINED_METHODS = ("snap", "prompt", "prompt_snap")


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


def load_asr_config(path: Optional[Union[str, Path]] = None, overrides: ConfigLike = None) -> dict:
    """Return the ``asr`` section of configs/asr.yaml (or of ``path``), plus overrides."""
    data = load_yaml(resolve_path(path) if path is not None else ASR_CONFIG)
    cfg = data.get("asr", data)
    return deep_merge(cfg, overrides) if overrides else cfg


@functools.lru_cache(maxsize=1)
def _default_cfg() -> dict:
    return load_asr_config()


def _cfg(cfg: ConfigLike) -> dict:
    """Resolve ``cfg`` (None / full config / asr section / partial override) to a complete asr section."""
    base = _default_cfg()
    if cfg is None:
        return deep_merge(base, {})
    if "asr" in cfg:
        return deep_merge(base, cfg["asr"])
    if set(cfg) & set(base):
        return deep_merge(base, cfg)
    return deep_merge(base, {})


@functools.lru_cache(maxsize=1)
def _metrics_cfg() -> dict:
    return metrics.load_metrics_config()


def vocabulary(cfg: ConfigLike = None) -> list:
    """The closed vocabulary, checked: normalized, unique, non-empty."""
    c = _cfg(cfg)
    vocab = list(c["constrained"]["vocabulary"])
    bad = [w for w in vocab if not isinstance(w, str) or normalize(w, c) != w or " " in w]
    if bad:
        raise ValueError(f"vocabulary words must already be normalized single tokens: {bad}")
    if len(set(vocab)) != len(vocab) or not vocab:
        raise ValueError("vocabulary must be non-empty and free of duplicates")
    return vocab


# --------------------------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------------------------

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand"))
# A digit run with optional thousands separators and decimals, not glued to letters ("3rd", "21st" stay).
_NUMBER_RE = re.compile(r"(?<!\w)(\d+(?:,\d{3})*)(?:\.(\d+))?(?!\w)")


def number_to_words(n: int) -> str:
    """Cardinal number in words (no hyphens, no "and"): 21 -> "twenty one", 1005 -> "one thousand five"."""
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " " + number_to_words(r))
    for scale, name in _SCALES:
        if n >= scale:
            q, r = divmod(n, scale)
            if q >= 1000 and scale == _SCALES[0][0]:
                raise ValueError(f"number too large: {n}")
            return number_to_words(q) + " " + name + ("" if r == 0 else " " + number_to_words(r))
    raise AssertionError("unreachable")


def _digits_to_words(text: str) -> str:
    def repl(m: re.Match) -> str:
        words = number_to_words(int(m.group(1).replace(",", "")))
        if m.group(2):
            words += " point " + " ".join(_ONES[int(d)] for d in m.group(2))
        return f" {words} "

    return _NUMBER_RE.sub(repl, text)


def _apply_replacements(tokens: list, replacements: Sequence[Sequence[str]]) -> list:
    for src, dst in replacements:
        s, d = str(src).split(), str(dst).split()
        out, i = [], 0
        while i < len(tokens):
            if s and tokens[i:i + len(s)] == s:
                out += d
                i += len(s)
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens


def normalize(text: str, cfg: ConfigLike = None) -> str:
    """Normalized, space-separated tokens (see module docstring). Applied to references and hypotheses."""
    c = _cfg(cfg)
    t = c["text"]
    s = _digits_to_words(text) if t["digits_to_words"] else text
    tokens = metrics.normalize_text(s, _metrics_cfg()).split()
    if t.get("replacements"):
        repl = [(metrics.normalize_text(a, _metrics_cfg()), metrics.normalize_text(b, _metrics_cfg()))
                for a, b in t["replacements"]]
        tokens = _apply_replacements(tokens, repl)
    return " ".join(tokens)


# --------------------------------------------------------------------------------------------
# Constrained (closed-vocabulary) hypothesis
# --------------------------------------------------------------------------------------------


def snap_to_vocabulary(tokens: Sequence[str], vocab: Sequence[str], cfg: ConfigLike = None) -> list:
    """Replace each token by its nearest vocabulary word (see module docstring); in-vocabulary tokens are kept."""
    c = _cfg(cfg)["constrained"]["snap"]
    if c["distance"] != "char":
        raise ValueError(f"unknown snap distance {c['distance']!r} (only 'char')")
    max_nd = c["max_normalized_distance"]
    vset = set(vocab)
    out = []
    for tok in tokens:
        if tok in vset:
            out.append(tok)
            continue
        best = min(vocab, key=lambda v: (metrics.edit_distance(tok, v), abs(len(tok) - len(v)), v))
        if max_nd is not None and metrics.edit_distance(tok, best) / max(len(tok), 1) > max_nd:
            continue  # too far from every vocabulary word: deleted
        out.append(best)
    return out


def vocabulary_prompt(cfg: ConfigLike = None) -> str:
    c = _cfg(cfg)
    return c["constrained"]["prompt_template"].format(words=", ".join(vocabulary(c)))


# --------------------------------------------------------------------------------------------
# Phonemes (g2p_en; lazy)
# --------------------------------------------------------------------------------------------

_G2P = None
_G2P_LOCK = threading.Lock()
_ARPABET_RE = re.compile(r"^[A-Z]{1,2}[0-2]?$")


def _g2p(cfg: dict):
    global _G2P
    with _G2P_LOCK:
        if _G2P is not None:
            return _G2P
        p = cfg["per"]
        if p["engine"] != "g2p_en":
            raise ValueError(f"unknown PER engine {p['engine']!r}")
        try:
            import nltk
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("PER needs g2p_en and nltk (requirements-linux-cuda.txt); "
                              "set asr.per.enabled: false to skip PER") from e
        data_dir = resolve_path(p["nltk_data"])
        data_dir.mkdir(parents=True, exist_ok=True)
        if str(data_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(data_dir))  # found first; g2p_en's import-time checks then pass
        for name, resource in p["nltk_resources"]:
            try:
                nltk.data.find(resource)
            except LookupError:
                if not p["allow_download"]:
                    raise
                if not nltk.download(name, download_dir=str(data_dir), quiet=True):
                    raise RuntimeError(f"nltk.download({name!r}) failed")
        try:
            from g2p_en import G2p
        except ImportError as e:  # pragma: no cover
            raise ImportError("PER needs g2p_en (requirements-linux-cuda.txt); "
                              "set asr.per.enabled: false to skip PER") from e
        _G2P = G2p()
        return _G2P


def phonemes(text: str, cfg: ConfigLike = None) -> list:
    """ARPAbet phonemes of already-normalized ``text`` (word boundaries dropped)."""
    c = _cfg(cfg)
    if not text.strip():
        return []
    out = [ph for ph in _g2p(c)(text) if _ARPABET_RE.match(ph)]
    if c["per"]["strip_stress"]:
        out = [ph.rstrip("012") for ph in out]
    return out


# --------------------------------------------------------------------------------------------
# Scoring (no model)
# --------------------------------------------------------------------------------------------


def _rate(errors: Optional[int], n: Optional[int]) -> float:
    if errors is None or not n:
        return float("nan")
    return errors / n


def _chars(norm: str) -> list:
    keep_spaces = _metrics_cfg()["text"]["cer_count_spaces"]
    return list(norm if keep_spaces else norm.replace(" ", ""))


def score_text(reference_text: str, hypothesis_text: str, cfg: ConfigLike = None,
               constrained_hypothesis_text: Optional[str] = None) -> dict:
    """WER/CER/PER, unconstrained and constrained, with the counts behind them (module docstring).

    ``constrained_hypothesis_text`` is the raw hypothesis the constrained score starts from (the prompted decode for
    methods prompt/prompt_snap); default ``hypothesis_text``. Snapping is applied here for snap/prompt_snap.
    """
    c = _cfg(cfg)
    method = c["constrained"]["method"]
    if method not in CONSTRAINED_METHODS:
        raise ValueError(f"constrained.method must be one of {CONSTRAINED_METHODS}, not {method!r}")
    vocab = vocabulary(c)
    ref = normalize(reference_text, c)
    hyp = normalize(hypothesis_text, c)
    hyp_c_tokens = normalize(constrained_hypothesis_text if constrained_hypothesis_text is not None
                             else hypothesis_text, c).split()
    if method in ("snap", "prompt_snap"):
        hyp_c_tokens = snap_to_vocabulary(hyp_c_tokens, vocab, c)
    hyp_c = " ".join(hyp_c_tokens)

    ref_w, hyp_w = ref.split(), hyp.split()
    vset = set(vocab)
    oov = sum(w not in vset for w in ref_w)
    constrained_ok = oov == 0 or c["constrained"]["oov_reference"] == "score"
    if c["constrained"]["oov_reference"] not in ("nan", "score"):
        raise ValueError("constrained.oov_reference must be 'nan' or 'score'")

    out = {
        "ref": ref, "hyp": hyp, "hyp_constrained": hyp_c,
        "ref_words": len(ref_w), "ref_chars": len(_chars(ref)), "ref_oov_words": oov,
        "hyp_words": len(hyp_w), "hyp_empty": not hyp_w,
        "word_errors": metrics.edit_distance(ref_w, hyp_w),
        "char_errors": metrics.edit_distance(_chars(ref), _chars(hyp)),
        "word_errors_constrained": metrics.edit_distance(ref_w, hyp_c_tokens) if constrained_ok else None,
        "char_errors_constrained": metrics.edit_distance(_chars(ref), _chars(hyp_c)) if constrained_ok else None,
        "ref_phones": None, "phone_errors": None, "phone_errors_constrained": None,
    }
    if c["per"]["enabled"]:
        ref_p = phonemes(ref, c)
        out["ref_phones"] = len(ref_p)
        out["phone_errors"] = metrics.edit_distance(ref_p, phonemes(hyp, c))
        if constrained_ok:
            out["phone_errors_constrained"] = metrics.edit_distance(ref_p, phonemes(hyp_c, c))
    out.update({
        "wer": _rate(out["word_errors"], out["ref_words"]),
        "cer": _rate(out["char_errors"], out["ref_chars"]),
        "per": _rate(out["phone_errors"], out["ref_phones"]),
        "wer_constrained": _rate(out["word_errors_constrained"], out["ref_words"]),
        "cer_constrained": _rate(out["char_errors_constrained"], out["ref_chars"]),
        "per_constrained": _rate(out["phone_errors_constrained"], out["ref_phones"]),
    })
    return out


# --------------------------------------------------------------------------------------------
# Whisper (faster-whisper; lazy)
# --------------------------------------------------------------------------------------------

_MODELS: dict = {}
_MODEL_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def model_dir(cfg: ConfigLike = None) -> Path:
    m = _cfg(cfg)["model"]
    return resolve_path(m["cache_dir"]) / m["local_name"]


def _set_cache_env(c: dict) -> None:
    """Keep every Hugging Face / OpenMP side effect inside the repo and within the thread budget."""
    hf_home = resolve_path(c["model"]["hf_home"])
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", str(c["cpu_threads"]))


def ensure_model(cfg: ConfigLike = None) -> Path:
    """Local model directory: downloaded once (pinned revision) into the repo, sha256-verified."""
    c = _cfg(cfg)
    m = c["model"]
    d = model_dir(c)
    missing = [f for f in m["files"] if not (d / f).is_file()]
    if missing:
        if not m["allow_download"]:
            raise FileNotFoundError(f"ASR model files missing in {d}: {missing} (asr.model.allow_download is false)")
        _set_cache_env(c)
        from huggingface_hub import snapshot_download

        d.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=m["repo_id"], revision=m["revision"], local_dir=str(d),
                          allow_patterns=list(m["files"]))
    if m["verify_sha256"]:
        for name, expected in (m.get("sha256") or {}).items():
            got = _sha256(d / name)
            if got != expected:
                raise RuntimeError(f"{d / name}: sha256 {got} != expected {expected}")
    return d


def _model(c: dict):
    key = (str(model_dir(c)), c["device"], c["compute_type"], int(c["cpu_threads"]), int(c["num_workers"]))
    with _MODEL_LOCK:
        if key not in _MODELS:
            if c["device"] != "cpu":
                raise ValueError("asr.device must be 'cpu' (the GPU is never used for ASR)")
            d = ensure_model(c)
            _set_cache_env(c)
            from faster_whisper import WhisperModel

            import ctranslate2

            from wvx.config import load_config

            # CTranslate2's RNG matters only for sampling (temperature > 0 fallback); seeded once per process.
            ctranslate2.set_random_seed(int(c["seed"] if c["seed"] is not None else load_config()["seed"]))
            _MODELS[key] = WhisperModel(str(d), device="cpu", compute_type=c["compute_type"],
                                        cpu_threads=int(c["cpu_threads"]), num_workers=int(c["num_workers"]),
                                        local_files_only=True)
        return _MODELS[key]


def prepare_audio(wav: np.ndarray, sr: int, cfg: ConfigLike = None) -> np.ndarray:
    """Mono float32 at Whisper's rate, full scale 1.0. Integer PCM is divided by ``audio.int_full_scale``; float
    audio must already be in full-scale-1 units (int16-scaled floats, such as their closed-loop audio, are
    rejected: divide them by ``audio.int_full_scale`` first)."""
    c = _cfg(cfg)["audio"]
    x = np.asarray(wav)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D (mono) signal, got shape {x.shape}")
    is_int = np.issubdtype(x.dtype, np.integer)
    x = x.astype(np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("audio contains NaN or inf")
    if is_int:
        x = x / c["int_full_scale"]
    elif x.size and np.abs(x).max() > c["max_float_peak"]:
        raise ValueError(f"float audio peaks at {np.abs(x).max():.4g} > audio.max_float_peak "
                         f"({c['max_float_peak']}): int16-scaled? divide by audio.int_full_scale first")
    target = int(c["sample_rate"])
    if int(sr) != target:
        import scipy.signal

        frac = Fraction(target, int(sr))
        x = scipy.signal.resample_poly(x, frac.numerator, frac.denominator,
                                       window=("kaiser", c["resample_kaiser_beta"]))
    if c["peak_normalize"] is not None and x.size and np.abs(x).max() > 0:
        x = x * (c["peak_normalize"] / np.abs(x).max())
    return x.astype(np.float32)


def transcribe_detailed(wav: np.ndarray, sr: int, cfg: ConfigLike = None,
                        initial_prompt: Optional[str] = None) -> dict:
    """Whisper transcript plus diagnostics: text, n_segments, avg_logprob (mean), no_speech_prob (max),
    max_compression_ratio (gzip ratio of the text; > decode.compression_ratio_threshold flags repetition loops) and
    max_temperature (> 0 = a temperature-fallback re-decode was used)."""
    c = _cfg(cfg)
    audio = prepare_audio(wav, sr, c)
    dec = dict(c["decode"])
    if initial_prompt is not None:
        dec["initial_prompt"] = initial_prompt
    segments, _info = _model(c).transcribe(audio, **dec)
    segments = list(segments)  # the generator does the decoding
    return {
        "text": " ".join(s.text.strip() for s in segments).strip(),
        "n_segments": len(segments),
        "avg_logprob": float(np.mean([s.avg_logprob for s in segments])) if segments else float("nan"),
        "no_speech_prob": float(max(s.no_speech_prob for s in segments)) if segments else float("nan"),
        "max_compression_ratio": float(max(s.compression_ratio for s in segments)) if segments else float("nan"),
        "max_temperature": float(max(s.temperature for s in segments)) if segments else float("nan"),
    }


def transcribe(wav: np.ndarray, sr: int, cfg: ConfigLike = None) -> str:
    """Raw Whisper text of ``wav`` (UNCONSTRAINED: no prompt, no LM, no vocabulary)."""
    return transcribe_detailed(wav, sr, cfg)["text"]


def evaluate_wav(wav: np.ndarray, sr: int, reference_text: str, cfg: ConfigLike = None) -> dict:
    """Transcribe ``wav`` and score it against ``reference_text`` (keys: see ``score_text``, plus ``hyp_raw``,
    ``hyp_raw_prompted`` (prompt methods only), the ``transcribe_detailed`` diagnostics, ``audio_s``, ``asr_s``)."""
    c = _cfg(cfg)
    t0 = time.perf_counter()
    tr = transcribe_detailed(wav, sr, c)
    constrained_raw = None
    if c["constrained"]["method"] in ("prompt", "prompt_snap"):
        constrained_raw = transcribe_detailed(wav, sr, c, initial_prompt=vocabulary_prompt(c))["text"]
    asr_s = time.perf_counter() - t0
    out = {"hyp_raw": tr["text"]}
    if constrained_raw is not None:
        out["hyp_raw_prompted"] = constrained_raw
    out.update(score_text(reference_text, tr["text"], c, constrained_hypothesis_text=constrained_raw))
    out.update({k: tr[k] for k in ("n_segments", "avg_logprob", "no_speech_prob", "max_compression_ratio",
                                   "max_temperature")})
    out.update({"audio_s": float(np.asarray(wav).size / sr), "asr_s": asr_s})
    return out


def versions() -> dict:
    """Installed versions of the ASR stack (None if absent), for provenance in result files."""
    from importlib import metadata

    out = {}
    for pkg in ("faster-whisper", "ctranslate2", "tokenizers", "huggingface-hub", "g2p-en", "nltk"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def pooled_rate(errors: Sequence[Optional[int]], totals: Sequence[Optional[int]]) -> float:
    """Sum of errors / sum of reference units over the trials where both are defined (NaN if none)."""
    pairs = [(e, n) for e, n in zip(errors, totals) if e is not None and n is not None
             and not (isinstance(e, float) and math.isnan(e))]
    tot = sum(n for _, n in pairs)
    return sum(e for e, _ in pairs) / tot if tot else float("nan")
