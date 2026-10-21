"""wvx.targets: TTS feature files, their target transform, silence-mixture voice statistics."""

import numpy as np
import pytest

from wvx import b2v
from wvx import targets as tg
from wvx.config import deep_merge

pytestmark = pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py")

SILENCE36 = np.r_[-12.48528, np.zeros(17), 3.1, -0.5, np.zeros(16)].astype(np.float32)


@pytest.fixture(scope="module")
def cfg():
    return tg.load_targets_config()


def test_config_merges_base_experiment(cfg):
    assert cfg["synthesis"]["samples_per_frame"] == 160          # from reproduce_inference.yaml
    assert cfg["wairagkar_data"]["target_smoothing_sigma"] == 2.0
    assert cfg["model"]["n_targets"] == 20                        # from default.yaml
    assert "alignment" in cfg and cfg["paths"]["tts_features"].is_absolute()


def test_voice_slug_and_text_key():
    assert tg.voice_slug("Reed (English (US))") == "reed_english_us"
    assert tg.voice_slug("Fred") == "fred"
    with pytest.raises(ValueError):
        tg.voice_slug("()")
    assert tg.text_key("  I am  here.   ") == "I am here."


def test_corpus_round_trip_and_lookup(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.normal(size=(n, 36)).astype(np.float32) for n in (5, 9)]
    c = tg.TTSCorpus("Fred", ["You up?", "I am here."], frames, SILENCE36, {"lpcnet": "7dc9942"})
    back = tg.TTSCorpus.load(c.save(tmp_path / "fred" / "x.npz"))
    assert back.voice == "Fred" and back.meta == {"lpcnet": "7dc9942"} and len(back) == 2
    np.testing.assert_array_equal(back.lookup("I am here.   "), frames[1])
    np.testing.assert_array_equal(back.silence, SILENCE36)
    assert "You up?" in back and "you up?" not in back
    with pytest.raises(KeyError):
        back.lookup("missing")
    with pytest.raises(ValueError):
        tg.TTSCorpus("Fred", ["a", " a"], frames, SILENCE36)


def test_corpus_path(cfg):
    p = tg.corpus_path(cfg, "Reed (English (US))", "kunz_50words")
    assert p.parts[-2:] == ("reed_english_us", "kunz_50words.npz")


def test_decoder_target_is_their_transform(cfg):
    data_utils = b2v.load("lib.data_utils")
    lo, hi = data_utils.load_lpc_norm(str(cfg["paths"]["lpc_norm"]))
    rng = np.random.default_rng(1)
    raw = (lo + rng.random((50, 36)).astype(np.float32)[:, :20] * (hi - lo))
    raw36 = np.concatenate([raw, rng.normal(size=(50, 16)).astype(np.float32)], axis=1)
    out = tg.decoder_target(raw36, cfg)
    want = data_utils.get_speech_features_from_single_trial(raw36, lo, hi, 18, True, 2.0)
    assert out.dtype == np.float32 and out.shape == (50, 20)
    np.testing.assert_allclose(out, want, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(tg.decoder_target(raw36[:, :20], cfg), out)   # 20 or 36 columns


def test_silence_frame_hits_their_json_extremes(cfg):
    sil = tg.scale(SILENCE36[None, :20], cfg)[0]
    assert sil[0] == pytest.approx(-1.0, abs=1e-5)       # c0 min of their JSON is digital silence
    assert sil[18] == pytest.approx(1.0, abs=1e-5)       # pitch max of their JSON is the silence code


def test_prediction_timeline(cfg):
    n_bins, w = 100, cfg["model"]["window_len"]
    pred = np.random.default_rng(2).uniform(-1, 1, (n_bins - w + 1, 20)).astype(np.float32)
    sil20 = SILENCE36[:20]
    raw = tg.prediction_timeline(pred, n_bins, sil20, cfg)
    assert raw.shape == (n_bins, 20)
    np.testing.assert_array_equal(raw[: w - 1], np.tile(sil20, (w - 1, 1)))
    np.testing.assert_allclose(tg.scale(raw[w - 1:], cfg), pred, atol=1e-5)    # denormalize inverts scale
    with pytest.raises(ValueError):
        tg.prediction_timeline(pred, 10, sil20, cfg)


def test_silence_mixture_fit_recovers_share():
    rng = np.random.default_rng(3)
    speech = rng.normal(loc=np.linspace(-1, 1, 20), scale=np.linspace(0.2, 2, 20), size=(20000, 20))
    sil = SILENCE36[:20].astype(np.float64)
    m = tg.moments(speech)
    ref_mean, ref_std = tg.mixture_moments(m, sil, 0.3)
    fit = tg.fit_silence_mixture(m, sil, ref_mean, ref_std, list(range(1, 20)), [0.0, 0.9, 0.005])
    assert fit["silence_share"] == pytest.approx(0.3, abs=0.006)
    assert fit["distance"] < 1e-4
    other = tg.moments(speech + 0.5)
    assert tg.fit_silence_mixture(other, sil, ref_mean, ref_std, list(range(1, 20)),
                                  [0.0, 0.9, 0.005])["distance"] > 0.05


def test_is_silence_and_voiced_f0(cfg):
    frames = np.stack([SILENCE36[:20], SILENCE36[:20] + 0.01])
    np.testing.assert_array_equal(tg.is_silence(frames, SILENCE36[:20]), [True, False])
    f = np.zeros((3, 20), np.float32)
    f[:, 18] = 0.02 * (160 - 100)      # period 160 samples at 16 kHz = 100 Hz
    f[:, 19] = [0.3, 0.2, -0.3]        # last one unvoiced
    np.testing.assert_allclose(tg.voiced_f0_hz(f, cfg), [100.0, 100.0])


def test_rank_voice_on_synthetic_corpus(cfg):
    rng = np.random.default_rng(4)
    ref = tg.load_norm_json(cfg)
    frames = [np.concatenate([rng.normal(ref["lpcnet_mean"], 0.5 * ref["lpcnet_std"], (40, 20)),
                              np.zeros((40, 16))], axis=1).astype(np.float32) for _ in range(3)]
    frames.append(np.tile(SILENCE36, (10, 1)))
    out = tg.rank_voice(frames, SILENCE36, cfg)
    assert out["n_frames_speech"] == 120                      # silence frames excluded
    for k in ("distance", "silence_share", "median_voiced_f0_hz", "mean_z", "sd_log_ratio"):
        assert k in out
    assert np.isfinite(out["distance"])


def test_speech_moments_and_reference_distance(cfg):
    rng = np.random.default_rng(5)
    speech = rng.normal(size=(200, 36)).astype(np.float32)
    speech[:, 0] = 8.0                                   # loud frames
    speech[:, 18], speech[:, 19] = 0.02 * (125 - 100), 0.3   # 128 Hz, voiced
    utt = np.concatenate([np.tile(SILENCE36, (50, 1)), speech])
    m = tg.speech_moments([utt], cfg, 30.0)
    assert m["n"] == 200                                 # silence frames dropped by the c0 rule
    assert m["f0_percentiles_hz"]["50"] == pytest.approx(128.0)
    same = tg.reference_distance(m, m, list(range(1, 18)))
    assert same["distance"] == pytest.approx(0.0) and same["median_f0_semitones_vs_ref"] == pytest.approx(0.0)
    shifted = dict(m, mean=(np.asarray(m["mean"]) + 0.5).tolist())
    assert tg.reference_distance(shifted, m, list(range(1, 18)))["distance"] > 0.1


def test_decoder_output_sanity(cfg):
    from types import SimpleNamespace

    trials = [SimpleNamespace(text=" ".join(["w"] * k)) for k in (2, 4, 6)]
    env = np.full(200, -100.0)
    results = []
    for k in (2, 4, 6):
        e = env.copy()
        for j in range(k):
            e[60 + 20 * j: 70 + 20 * j] = -10.0
        results.append(SimpleNamespace(pred_segments=np.zeros((k, 2)), tts_segments=np.zeros((k, 2)), pred_env=e))
    s = tg.decoder_output_sanity(trials, results, cfg)
    assert s["r_detected_syllables_vs_cue_words"] == pytest.approx(1.0)
    assert s["fraction_trials_without_speech"] == 0.0 and s["envelope_range_db_p95_minus_p5_median"] == pytest.approx(90.0)


def _c0_frames(env_db):
    """(n, 36) raw frames whose c0 is the given dB power (silence elsewhere), with a ramp in feature 1."""
    f = np.tile(SILENCE36, (env_db.size, 1)).astype(np.float64)
    f[:, 0] = env_db * np.sqrt(18.0) / 10.0 - 4.0
    f[:, 1] = np.linspace(-1, 1, env_db.size)
    return f.astype(np.float32)


def _bumps(n, centres, width):
    t = np.arange(n)[:, None]
    return -20.0 + 40.0 * np.minimum(np.exp(-0.5 * ((t - np.asarray(centres)[None, :]) / width) ** 2).sum(1), 1.0)


def test_align_trials_with_saved_predictions_and_segments(cfg):
    from types import SimpleNamespace

    w, T = cfg["model"]["window_len"], 320
    corpus = tg.TTSCorpus("V", ["You up?"], [_c0_frames(_bumps(100, [30, 50], 5))], SILENCE36)
    pred_raw = _c0_frames(_bumps(T, [150, 230], 12))[:, :20]
    preds = [tg.scale(pred_raw[w - 1:], cfg)]
    trials = [SimpleNamespace(neural=np.zeros((T, cfg["model"]["n_features"]), np.float32), text="You up?")]
    _, ref = tg.align_trials(None, trials, corpus, cfg, reorder=False, log=None, preds=preds)
    assert len(ref[0].tts_segments) == 2 and len(ref[0].pred_segments) == 2
    _, got = tg.align_trials(None, trials, corpus, cfg, reorder=False, log=None, preds=preds,
                             tts_segments=[ref[0].tts_segments], pred_segments=[ref[0].pred_segments])
    np.testing.assert_array_equal(got[0].mapped_segments, ref[0].mapped_segments)
    np.testing.assert_array_equal(got[0].target_raw, ref[0].target_raw)
    one = np.array([[ref[0].tts_segments[0, 0], ref[0].tts_segments[-1, 1]]])          # one TTS syllable instead
    _, merged = tg.align_trials(None, trials, corpus, cfg, reorder=False, log=None, preds=preds, tts_segments=[one])
    assert merged[0].mapped_segments.shape == (1, 2)
    with pytest.raises(ValueError):
        tg.align_trials(None, trials, corpus, cfg, reorder=False, log=None, preds=preds * 2)
