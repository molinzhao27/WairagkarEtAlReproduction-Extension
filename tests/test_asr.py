"""Tests for wvx.asr: text normalization, constrained snapping and WER/CER/PER bookkeeping (fast, no model).

Whisper and g2p_en tests run only where those packages and their files are already present (the GPU host);
nothing here downloads anything.
"""

import importlib.util
import math

import numpy as np
import pytest

from wvx import asr, metrics

NO_PER = {"per": {"enabled": False}}
VOCAB = asr.vocabulary()


def _has(mod):
    return importlib.util.find_spec(mod) is not None


# --------------------------------------------------------------------------- config / vocabulary


def test_vocabulary_is_the_50_word_set():
    assert len(VOCAB) == 50 and len(set(VOCAB)) == 50
    assert all(asr.normalize(w) == w for w in VOCAB)
    for w in ("i", "okay", "goodbye", "hello", "thirsty", "comfortable"):
        assert w in VOCAB


def test_config_defaults_are_cpu_and_small():
    c = asr.load_asr_config()
    assert c["device"] == "cpu" and c["cpu_threads"] <= 8
    assert c["model"]["download_bytes"] < 1e9
    assert c["constrained"]["method"] in asr.CONSTRAINED_METHODS
    assert c["decode"]["initial_prompt"] is None  # unconstrained decoding has no prompt


def test_variants_merge_to_valid_configs():
    from wvx.config import deep_merge

    base = asr.load_asr_config()
    for name, override in base["variants"].items():
        c = deep_merge(base, override)
        assert c["constrained"]["method"] in asr.CONSTRAINED_METHODS, name
        assert c["device"] == "cpu" and c["cpu_threads"] <= 8, name
        assert asr.vocabulary(c) == VOCAB, name


# --------------------------------------------------------------------------- normalization


@pytest.mark.parametrize("n, words", [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (20, "twenty"), (21, "twenty one"), (100, "one hundred"),
    (105, "one hundred five"), (1000, "one thousand"), (1005, "one thousand five"),
    (21_300, "twenty one thousand three hundred"), (2_000_001, "two million one"),
])
def test_number_to_words(n, words):
    assert asr.number_to_words(n) == words


@pytest.mark.parametrize("text, norm", [
    ("I need 2 glasses!", "i need two glasses"),
    ("OK, good-bye.", "okay goodbye"),
    ("Ok.  Good bye", "okay goodbye"),
    ("1,000 people", "one thousand people"),
    ("It's 3.5 now", "it's three point five now"),
    ("the 21st time", "the 21st time"),        # ordinals are left alone
    ("Hello?", "hello"),
    ("“Don’t” — stop", "don't stop"),
    ("", ""),
])
def test_normalize(text, norm):
    assert asr.normalize(text) == norm


def test_normalize_is_idempotent_and_matches_metrics_without_digits():
    s = "Is my nurse very hungry? They're coming-outside."
    n = asr.normalize(s)
    assert asr.normalize(n) == n
    assert n == metrics.normalize_text(s)


# --------------------------------------------------------------------------- snapping


def test_snap_keeps_in_vocabulary_words():
    toks = "i hope it is you".split()
    assert asr.snap_to_vocabulary(toks, VOCAB) == toks


def test_snap_to_nearest_word():
    assert asr.snap_to_vocabulary(["nurses", "hungary", "thirty", "glass"], VOCAB) == \
        ["nurse", "hungry", "thirsty", "glasses"]


def test_snap_tie_breaks_by_length_then_alphabet():
    # "aa": distance 1 to "ab", "ba" and "aab"; same-length candidates win, then alphabetical.
    assert asr.snap_to_vocabulary(["aa"], ["ba", "aab", "ab"]) == ["ab"]


def test_snap_never_changes_word_count_by_default_and_can_delete_when_bounded():
    toks = ["xylophone", "my", "zzzz"]
    assert len(asr.snap_to_vocabulary(toks, VOCAB)) == 3
    cfg = {"constrained": {"snap": {"max_normalized_distance": 0.3}}}
    assert asr.snap_to_vocabulary(["nurses", "xylophone"], VOCAB, cfg) == ["nurse"]


# --------------------------------------------------------------------------- scoring bookkeeping


def test_score_counts_and_rates():
    s = asr.score_text("I need good glasses.", "I need glasses, please", NO_PER)
    assert s["ref"] == "i need good glasses" and s["hyp"] == "i need glasses please"
    assert s["ref_words"] == 4 and s["word_errors"] == 2 and s["wer"] == 0.5
    assert s["wer"] == pytest.approx(metrics.word_error_rate(s["ref"], s["hyp"]))
    assert s["cer"] == pytest.approx(metrics.char_error_rate(s["ref"], s["hyp"]))
    assert s["ref_chars"] == len("i need good glasses")
    assert s["ref_oov_words"] == 0
    # every hypothesis word is in the vocabulary, so snapping changes nothing
    assert s["hyp_constrained"] == s["hyp"] and s["wer_constrained"] == s["wer"]
    assert math.isnan(s["per"]) and s["phone_errors"] is None


def test_constrained_snapping_fixes_near_misses_only():
    s = asr.score_text("Is my nurse very hungry?", "Is my nurses very hungary", NO_PER)
    assert s["wer"] == pytest.approx(2 / 5)
    assert s["hyp_constrained"] == "is my nurse very hungry" and s["wer_constrained"] == 0.0
    # insertions stay insertions after snapping
    s = asr.score_text("You up?", "you up up", NO_PER)
    assert s["word_errors_constrained"] == 1 and s["wer_constrained"] == 0.5


def test_empty_hypothesis_is_all_deletions():
    s = asr.score_text("I hope it is you.", "", NO_PER)
    assert s["hyp_empty"] and s["wer"] == 1.0 and s["wer_constrained"] == 1.0 and s["cer"] == 1.0


def test_out_of_vocabulary_reference_gives_nan_constrained_scores():
    s = asr.score_text("The child tried to play with me today.", "the child tried to play", NO_PER)
    assert s["ref_oov_words"] > 0
    assert math.isnan(s["wer_constrained"]) and s["word_errors_constrained"] is None
    assert not math.isnan(s["wer"])
    s = asr.score_text("The child tried to play.", "the child tried to play",
                       {**NO_PER, "constrained": {"oov_reference": "score"}})
    assert s["wer_constrained"] > 0  # every OOV reference word is an unavoidable error


def test_prompt_method_uses_the_prompted_hypothesis_without_snapping():
    cfg = {**NO_PER, "constrained": {"method": "prompt"}}
    s = asr.score_text("I am tired.", "I am tyred", cfg, constrained_hypothesis_text="I am tyred")
    assert s["hyp_constrained"] == "i am tyred" and s["wer_constrained"] == pytest.approx(1 / 3)
    cfg = {**NO_PER, "constrained": {"method": "prompt_snap"}}
    s = asr.score_text("I am tired.", "eye am tyred", cfg, constrained_hypothesis_text="I am tyred")
    assert s["hyp_constrained"] == "i am tired" and s["wer_constrained"] == 0.0
    assert s["wer"] == pytest.approx(2 / 3)


def test_pooled_rate_is_sum_over_sum_not_mean():
    a = asr.score_text("I hope it is you.", "I hope it is", NO_PER)       # 1/5
    b = asr.score_text("You up?", "", NO_PER)                              # 2/2
    pooled = asr.pooled_rate([a["word_errors"], b["word_errors"]], [a["ref_words"], b["ref_words"]])
    assert pooled == pytest.approx(3 / 7)
    assert pooled != pytest.approx(np.mean([a["wer"], b["wer"]]))
    assert pooled == pytest.approx(metrics.word_error_rate(["I hope it is you.", "You up?"], ["I hope it is", ""]))
    assert asr.pooled_rate([None, a["word_errors"]], [b["ref_words"], a["ref_words"]]) == pytest.approx(1 / 5)
    assert math.isnan(asr.pooled_rate([None], [3]))


# --------------------------------------------------------------------------- audio front end


def test_prepare_audio_scaling_and_resampling():
    sr = 8000
    x = (0.25 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    y = asr.prepare_audio(x, sr)
    assert y.dtype == np.float32 and y.size == 2 * x.size
    pcm = (x * 32768).astype(np.int16)
    assert np.abs(asr.prepare_audio(pcm, sr)).max() == pytest.approx(0.25, abs=0.02)
    with pytest.raises(ValueError, match="int16-scaled"):
        asr.prepare_audio(x.astype(np.float64) * 32768, sr)
    with pytest.raises(ValueError):
        asr.prepare_audio(np.zeros((2, 10)), sr)


# --------------------------------------------------------------------------- optional: g2p_en / Whisper


def _nltk_ready():
    if not (_has("g2p_en") and _has("nltk")):
        return False
    from wvx.config import resolve_path

    c = asr.load_asr_config()
    root = resolve_path(c["per"]["nltk_data"])
    return all((root / res).exists() for _, res in c["per"]["nltk_resources"])


@pytest.mark.skipif(not _nltk_ready(), reason="g2p_en / its NLTK data not installed here")
def test_per_with_g2p():
    cfg = {"per": {"allow_download": False}}
    assert asr.phonemes("i hope it is you", cfg) == ["AY", "HH", "OW", "P", "IH", "T", "IH", "Z", "Y", "UW"]
    s = asr.score_text("I hope it is you.", "I hope it is you", cfg)
    assert s["per"] == 0.0 and s["ref_phones"] == 10
    s = asr.score_text("I hope it is you.", "I hope it is", cfg)
    assert s["phone_errors"] == 2 and s["per"] == pytest.approx(0.2)


def _whisper_ready():
    return _has("faster_whisper") and all(
        (asr.model_dir() / f).is_file() for f in asr.load_asr_config()["model"]["files"])


@pytest.mark.skipif(not _whisper_ready(), reason="faster-whisper or the small.en model not installed here")
def test_transcribe_silence_and_bookkeeping():
    cfg = {**NO_PER, "model": {"allow_download": False, "verify_sha256": False}}
    sr = 16000
    silence = np.zeros(2 * sr, np.float32)
    text = asr.transcribe(silence, sr, cfg)
    assert isinstance(text, str)
    res = asr.evaluate_wav(silence, sr, "I hope it is you.", cfg)
    assert res["hyp_raw"] == text and res["audio_s"] == pytest.approx(2.0)
    assert res["wer"] == pytest.approx(res["word_errors"] / 5)
