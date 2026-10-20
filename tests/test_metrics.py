"""Tests for wvx.metrics on synthetic speech-like audio (no data needed; fast)."""

import math

import numpy as np
import pytest

from wvx import metrics as M
from wvx.config import load_config

SEED = load_config()["seed"]
CFG = M.load_metrics_config()
SR = CFG["sample_rate"]
HOP = int(round(CFG["stft"]["hop_ms"] * SR / 1000))
WIN = int(round(CFG["stft"]["win_ms"] * SR / 1000))

# Test fixture: three "words" (duration s, f0 start/end Hz, F1 start/end, F2 start/end).
WORDS = [
    (0.35, 120, 150, 700, 400, 1200, 2200),
    (0.25, 140, 110, 300, 800, 2300, 1000),
    (0.45, 130, 160, 500, 600, 1700, 1300),
]


def _word(dur, f0a, f0b, f1a, f1b, f2a, f2b, sr=SR, stretch=1.0, seed=SEED):
    """Harmonic source with gliding pitch and two moving formants, plus a short noise burst."""
    n = int(round(dur * stretch * sr))
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    f0 = f0a + (f0b - f0a) * u
    phase = 2 * np.pi * np.cumsum(f0) / sr
    f1 = f1a + (f1b - f1a) * u
    f2 = f2a + (f2b - f2a) * u
    x = np.zeros(n)
    for h in range(1, int(sr / 2 // max(f0a, f0b)) + 1):
        fh = h * f0
        amp = np.exp(-0.5 * ((fh - f1) / 150) ** 2) + 0.6 * np.exp(-0.5 * ((fh - f2) / 200) ** 2) + 0.02
        x += amp * np.sin(h * phase) * (fh < sr / 2)
    burst = np.random.default_rng(seed).normal(size=n) * 0.3 * np.exp(-(((u - 0.1) / 0.05) ** 2))
    return (x * np.sqrt(np.sin(np.pi * u)) + burst) / 5


def _sentence(sr=SR, stretch=1.0, gap=0.2, lead=0.2, order=(0, 1, 2)):
    parts = [np.zeros(int(round(lead * sr)))]
    for k in order:
        parts += [_word(*WORDS[k], sr=sr, stretch=stretch, seed=SEED + k), np.zeros(int(round(gap * sr)))]
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def ref():
    return _sentence()


@pytest.fixture(scope="module")
def noise(ref):
    return np.random.default_rng(SEED).normal(size=ref.size)


def _with_snr(ref, noise, snr_db):
    return ref + noise * np.sqrt(np.mean(ref**2) / 10 ** (snr_db / 10))


# ---------------------------------------------------------------- identity / API


def test_identical_signals_give_corr_one_and_mcd_zero(ref):
    assert M.mel_spectrogram_correlation(ref, ref, SR, CFG) == pytest.approx(1.0, abs=1e-12)
    assert M.mel_cepstral_distortion(ref, ref, SR, CFG) == pytest.approx(0.0, abs=1e-12)
    r = M.mel_band_correlations(ref, ref, SR, CFG)
    assert r.shape == (CFG["mel_corr"]["n_mels"],)
    assert np.allclose(r[~np.isnan(r)], 1.0)


def test_evaluate_returns_required_float_keys(ref, noise):
    out = M.evaluate(ref, _with_snr(ref, noise, 20), SR, CFG)
    for key in ("mel_corr", "mcd", "mel_corr_framewise", "ref_speech_fraction", "est_speech_fraction"):
        assert isinstance(out[key], float), key
    assert out["mel_corr"] == pytest.approx(M.mel_spectrogram_correlation(ref, _with_snr(ref, noise, 20), SR, CFG))
    assert out["mcd"] == pytest.approx(M.mel_cepstral_distortion(ref, _with_snr(ref, noise, 20), SR, CFG))


def test_cfg_forms_none_full_and_section_are_equivalent(ref, noise):
    est = _with_snr(ref, noise, 10)
    full = load_config(experiment="configs/metrics.yaml")
    a = M.evaluate(ref, est, SR, None)
    b = M.evaluate(ref, est, SR, full)
    c = M.evaluate(ref, est, SR, CFG)
    assert a == b == c


def test_config_overrides_are_respected(ref):
    r = M.mel_band_correlations(ref, ref, SR, {"mel_corr": {"n_mels": 32}})
    assert r.shape == (32,)
    out = M.evaluate(ref, ref, SR, {"metrics": {"evaluate": {"mel_corr_framewise": False}}})
    assert "mel_corr_framewise" not in out


def test_other_sample_rates_are_resampled():
    sr = 22050
    x = _sentence(sr=sr)
    out = M.evaluate(x, x, sr, CFG)
    assert out["mel_corr"] == pytest.approx(1.0, abs=1e-12)
    assert out["mcd"] == pytest.approx(0.0, abs=1e-12)
    n_frames = M.speech_frame_mask(x, sr, CFG).size
    assert n_frames == 1 + round(x.size * SR / sr) // HOP  # analysed at 16 kHz, 10 ms frames


def test_invalid_inputs_raise(ref):
    with pytest.raises(ValueError):
        M.evaluate(np.stack([ref, ref]), ref, SR, CFG)
    bad = ref.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError):
        M.evaluate(ref, bad, SR, CFG)
    with pytest.raises(ValueError):
        M.evaluate(ref, np.array([]), SR, CFG)


# ---------------------------------------------------------------- alignment (DTW)


def test_leading_silence_shift_is_invisible(ref):
    est = np.concatenate([np.zeros(25 * HOP), ref])  # whole-frame shift
    out = M.evaluate(ref, est, SR, CFG)
    assert out["mel_corr"] == pytest.approx(1.0, abs=1e-9)
    assert out["mcd"] == pytest.approx(0.0, abs=1e-9)
    assert out["mel_corr_framewise"] < 0.5  # without silence removal + DTW the shift hurts


def test_subframe_shift_is_nearly_invisible(ref):
    est = np.concatenate([np.zeros(HOP // 4 + 1), ref])
    assert M.mel_spectrogram_correlation(ref, est, SR, CFG) > 0.97
    assert M.mel_cepstral_distortion(ref, est, SR, CFG) < 0.5


def test_time_stretch_is_handled_by_dtw(ref):
    est = _sentence(stretch=1.25, gap=0.35)  # slower words, longer pauses
    off = {"mel_corr": {"dtw": False}, "mcd": {"dtw": False}}
    corr_dtw, mcd_dtw = M.mel_spectrogram_correlation(ref, est, SR, CFG), M.mel_cepstral_distortion(ref, est, SR, CFG)
    corr_off, mcd_off = M.mel_spectrogram_correlation(ref, est, SR, off), M.mel_cepstral_distortion(ref, est, SR, off)
    assert corr_dtw > 0.98 and mcd_dtw < 1.0
    assert corr_off < 0.5 and mcd_off > 3.0


def test_level_difference_does_not_change_metrics(ref):
    est = 0.05 * _sentence(stretch=1.25, gap=0.35)
    same_level = _sentence(stretch=1.25, gap=0.35)
    assert M.mel_spectrogram_correlation(ref, est, SR, CFG) == pytest.approx(
        M.mel_spectrogram_correlation(ref, same_level, SR, CFG), abs=1e-3
    )
    assert M.mel_cepstral_distortion(ref, 0.05 * ref, SR, CFG) == pytest.approx(0.0, abs=1e-9)
    assert M.mel_spectrogram_correlation(ref, 0.05 * ref, SR, CFG) == pytest.approx(1.0, abs=1e-9)


def test_wrong_words_score_worse_than_right_words(ref, noise):
    wrong = _sentence(order=(2, 0, 1))
    right = _with_snr(ref, noise, 20)
    assert M.mel_spectrogram_correlation(ref, wrong, SR, CFG) < M.mel_spectrogram_correlation(ref, right, SR, CFG)


# ---------------------------------------------------------------- noise


def test_noise_lowers_corr_and_raises_mcd_monotonically(ref, noise):
    snrs = [40, 30, 20, 10, 0]
    outs = [M.evaluate(ref, _with_snr(ref, noise, s), SR, CFG) for s in snrs]
    corr = [o["mel_corr"] for o in outs]
    mcd = [o["mcd"] for o in outs]
    assert corr[0] < 1.0 and mcd[0] > 0.0
    assert all(a > b for a, b in zip(corr, corr[1:])), corr
    assert all(a < b for a, b in zip(mcd, mcd[1:])), mcd


# ---------------------------------------------------------------- silence removal


def test_silence_mask_marks_gaps_and_words(ref):
    mask = M.speech_frame_mask(ref, SR, CFG)
    centres = np.arange(mask.size) * HOP  # center=True: frame t is centred on sample t*HOP
    half = WIN // 2
    # Frames whose whole window lies in digital silence must be silent ...
    silent_window = np.array([not np.any(ref[max(0, c - half): c + half]) for c in centres])
    assert not mask[silent_window].any()
    # ... and frames in the middle 60% of every word must be speech.
    start = int(round(0.2 * SR))
    for k in (0, 1, 2):
        n = int(round(WORDS[k][0] * SR))
        core = (centres >= start + 0.2 * n) & (centres <= start + 0.8 * n)
        assert mask[core].all(), k
        start += n + int(round(0.2 * SR))


def test_extra_silence_between_words_is_removed(ref):
    est = _sentence(gap=0.8, lead=0.6)
    out = M.evaluate(ref, est, SR, CFG)
    assert out["mel_corr"] == pytest.approx(1.0, abs=1e-9)
    assert out["mcd"] == pytest.approx(0.0, abs=1e-9)
    assert out["est_speech_fraction"] < out["ref_speech_fraction"]
    off = {"mel_corr": {"remove_silence": False, "dtw": False}, "mcd": {"remove_silence": False, "dtw": False}}
    assert M.mel_spectrogram_correlation(ref, est, SR, off) < 0.8
    assert M.mel_cepstral_distortion(ref, est, SR, off) > 3.0


def test_silent_estimate_gives_nan(ref):
    out = M.evaluate(ref, np.zeros_like(ref), SR, CFG)
    assert math.isnan(out["mel_corr"]) and math.isnan(out["mcd"])
    assert out["est_speech_fraction"] == 0.0


def test_min_silence_ms_keeps_short_internal_gaps_only():
    gap = np.zeros(int(round(0.03 * SR)))  # 30 ms pause inside an utterance
    lead = np.zeros(int(round(0.2 * SR)))
    x = np.concatenate([lead, _word(*WORDS[0]), gap, _word(*WORDS[1]), lead])
    tight = {"silence": {"min_silence_ms": 0.0, "threshold_db": 20.0}}
    loose = {"silence": {"min_silence_ms": 100.0, "threshold_db": 20.0}}
    m_tight, m_loose = M.speech_frame_mask(x, SR, tight), M.speech_frame_mask(x, SR, loose)
    idx = np.flatnonzero(m_loose)
    assert np.all(m_loose[idx[0]: idx[-1] + 1])  # short internal gap refilled
    assert not np.all(m_tight[idx[0]: idx[-1] + 1])  # ... but dropped with min_silence_ms = 0
    assert not m_loose[0] and not m_loose[-1]  # leading/trailing silence always removed


# ---------------------------------------------------------------- MCD definition


def test_mcd_equals_rms_mel_db_distance_when_all_coefficients_used(ref, noise):
    """With n_ceps == n_mels, MCD = mean over frames of RMS_band(dB difference - its band mean)."""
    est = _with_snr(ref, noise, 20)
    cfg = M._cfg({"mcd": {"remove_silence": False, "dtw": False}})
    n = cfg["mcd"]["n_mels"]
    assert cfg["mcd"]["n_ceps"] == n
    d = M._mel_db(M._analyze(ref, SR, cfg, "r").power, n, cfg) - M._mel_db(M._analyze(est, SR, cfg, "e").power, n, cfg)
    d -= d.mean(axis=0, keepdims=True)
    expected = np.sqrt((d**2).mean(axis=0)).mean()
    assert M.mel_cepstral_distortion(ref, est, SR, cfg) == pytest.approx(expected, rel=1e-10)


def test_mcd_constant_and_c0_options(ref, noise):
    est = _with_snr(ref, noise, 20)
    kub = M.mel_cepstral_distortion(ref, est, SR, CFG)
    plain = M.mel_cepstral_distortion(ref, est, SR, {"mcd": {"constant": 1.0}})
    assert kub == pytest.approx(plain * 10 * math.sqrt(2) / math.log(10), rel=1e-10)
    assert M.mel_cepstral_distortion(ref, 0.05 * ref, SR, {"mcd": {"include_c0": True}}) > 1.0


# ---------------------------------------------------------------- WER / CER


@pytest.mark.parametrize(
    "ref_text, hyp_text, expected",
    [
        ("the cat sat", "the cat sat", 0.0),
        ("the cat sat on the mat", "the cat sit on mat", 2 / 6),  # 1 sub + 1 del
        ("a b", "a x b y", 1.0),  # 2 insertions
        ("a", "x y z", 3.0),  # WER can exceed 1
        ("a b c", "", 1.0),
        ("Hello, World!", "hello world", 0.0),  # case + punctuation
        ("don’t stop", "don't stop", 0.0),  # curly apostrophe
        ("don't stop", "dont stop", 0.5),  # apostrophe is kept inside words
        ("a well-known 'quote'", "a well known quote", 0.0),  # hyphen -> space, edge quotes
    ],
)
def test_word_error_rate_known_cases(ref_text, hyp_text, expected):
    assert M.word_error_rate(ref_text, hyp_text, CFG) == pytest.approx(expected)


def test_word_error_rate_corpus_level_pools_edits():
    assert M.word_error_rate(["a b", "c d e"], ["a b", "c x e"], CFG) == pytest.approx(1 / 5)


def test_char_error_rate_known_cases():
    assert M.char_error_rate("kitten", "sitting", CFG) == pytest.approx(3 / 6)
    assert M.char_error_rate("ab cd", "abcd", CFG) == pytest.approx(1 / 5)  # missing space counts
    assert M.char_error_rate("ab cd", "abcd", {"text": {"cer_count_spaces": False}}) == 0.0
    assert M.char_error_rate("Hi, there.", "hi there", CFG) == 0.0


def test_edit_distance_and_empty_reference():
    assert M.edit_distance("flaw", "lawn") == 2
    assert M.edit_distance("lawn", "flaw") == 2
    assert M.edit_distance([], ["a"]) == 1
    with pytest.raises(ValueError):
        M.word_error_rate("", "anything", CFG)
    with pytest.raises(ValueError):
        M.word_error_rate(["a"], ["a", "b"], CFG)
