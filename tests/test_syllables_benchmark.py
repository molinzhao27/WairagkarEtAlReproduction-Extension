"""scripts/syllable_benchmark.py score_run end to end on synthetic trials: every method and variant runs through
the unchanged alignment and validate_alignment's scoring (no data, no LPCNet, no segmenter tools)."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from wvx import alignment as al
from wvx import b2v
from wvx import syllables as sy
from wvx import targets as tg
from wvx.config import REPO_ROOT

pytestmark = pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py")

SILENCE36 = np.r_[-12.48528, np.zeros(17), 3.1, -0.5, np.zeros(16)].astype(np.float32)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("syllable_benchmark", REPO_ROOT / "scripts" / "syllable_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg():
    return sy.load_benchmark_config()


def _frames(env_db, seed):
    """(n, 36) raw frames: c0 carries the envelope (dB), cepstra a seeded random walk, voiced pitch."""
    rng = np.random.default_rng(seed)
    f = np.tile(SILENCE36, (env_db.size, 1)).astype(np.float64)
    f[:, 0] = env_db * np.sqrt(18.0) / 10.0 - 4.0
    f[:, 1:18] = np.cumsum(rng.normal(scale=0.1, size=(env_db.size, 17)), axis=0)
    f[:, 18], f[:, 19] = 0.5, 0.3
    return f.astype(np.float32)


def _bumps(n, centres, width):
    t = np.arange(n)[:, None]
    return -20.0 + 40.0 * np.minimum(np.exp(-0.5 * ((t - np.asarray(centres)[None, :]) / width) ** 2).sum(1), 1.0)


@pytest.fixture(scope="module")
def world(cfg):
    """Two trials whose decoder output and 'their' targets put each TTS syllable at a known place."""
    w, T = cfg["model"]["window_len"], 420
    texts = ["You up?", "I hope it."]
    tts_c = [[30, 50], [30, 50, 70]]
    corpus = tg.TTSCorpus("V", texts, [_frames(_bumps(100, c, 5), k) for k, c in enumerate(tts_c)], SILENCE36)
    sil20 = tg.silence_frame(corpus, cfg)
    their_syl = [np.array([[150, 200], [260, 310]]), np.array([[120, 160], [200, 240], [300, 350]])]
    trials, preds, their_raw, tts_forced, rec_n, rec_s = [], [], [], [], [], []
    lag = cfg["syllables"]["audio_lag_ms"] / 1000.0
    for k, text in enumerate(texts):
        tts = tg.reduce(corpus.lookup(text), cfg)
        tts_segs = al.segment_syllables(al.envelope_db(tts, cfg), cfg)          # stand-in for the forced syllables
        assert len(tts_segs) == len(their_syl[k])
        raw = al.stretch_features(tts, tts_segs, their_syl[k], T, sil20, cfg)
        their_raw.append(raw)
        trials.append(SimpleNamespace(neural=np.zeros((T, cfg["model"]["n_features"]), np.float32), text=text,
                                      target=tg.decoder_target(raw, cfg)))
        preds.append(tg.scale(raw[w - 1:], cfg))                                # a decoder that is exactly right
        tts_forced.append(tts_segs)
        centres = their_syl[k].mean(axis=1) / 100.0 + lag
        rec_n.append({"nuclei_s": centres.tolist(), "cuts_s": (their_syl[k][1:, 0] / 100.0 + lag).tolist()})
        rec_s.append({"segments_s": (their_syl[k] / 100.0 + lag).tolist()})
    pred_env = [al.envelope_db(tg.prediction_timeline(p, T, sil20, cfg), cfg) for p in preds]
    return dict(trials=trials, preds=preds, corpus=corpus, their_raw=their_raw, syl=their_syl, words=their_syl,
                desc=["synthetic"] * 2, pred_env=pred_env, tts_forced=tts_forced,
                records={"current": None, "nuclei": rec_n, "sylber": rec_s})


@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("method", ["current", "nuclei", "sylber"])
@pytest.mark.parametrize("count", [False, True])
def test_score_run_every_method(bench, cfg, world, method, count, arm):
    w = world
    out = bench.score_run(method, count, "forced", w["trials"], w["preds"], w["corpus"], cfg, w["their_raw"], w["syl"],
                          w["words"], w["desc"], w["pred_env"], w["tts_forced"], w["records"][method], False, arm=arm)
    h = out["headline"]
    assert h["n_trials"] == 2 and h["oracle_n_trials"] == 2 and h["fraction_tts_count_equal_theirs"] == 1.0
    assert h["target_r_mean"] > 0.8 and h["target_r_mean"] > h["chance_r_mean"] + 0.3
    assert h["onsets_within_tolerance"] == 1.0 and h["span_onset_error_median_ms"] <= 50
    assert h["tts_syllables_without_speech"] == 0 and h["tts_syllables_total"] == 5
    assert len(out["per_trial"]) == 2 and out["offline_decoder"]["n_trials"] == 2
    if arm == "B":     # the synthetic decoder finds exactly the right syllables: one to one, nothing left silent
        assert h["trials_paired_one_to_one"] == 2 and h["decoder_syllables_without_tts"] == 0
        assert h["tts_syllables_without_frames"] == 0 and h["trials_without_decoder_syllables"] == 0
        assert out["per_trial"][0]["mapping"]["pairing"] == "one_to_one"
    else:
        assert h["decoder_syllables_without_tts"] is None and out["per_trial"][0]["mapping"] is None


def test_score_run_reference_and_guards(bench, cfg, world):
    w = world
    out = bench.score_run("current", False, "envelope", w["trials"], w["preds"], w["corpus"], cfg, w["their_raw"],
                          w["syl"], w["words"], w["desc"], w["pred_env"], None, None, False)
    assert out["headline"]["target_r_mean"] > 0.8
    with pytest.raises(ValueError):
        bench.score_run("current", True, "envelope", w["trials"], w["preds"], w["corpus"], cfg, w["their_raw"],
                        w["syl"], w["words"], w["desc"], w["pred_env"], None, None, False)
    assert bench.run_name("nuclei", True, "forced", "B") == "nuclei__count__tts-forced__arm-B"
    runs = bench.planned_runs(cfg)
    assert len(runs) == 13 and runs[-1] == ("current", False, "envelope", "A")
    assert {(m, c, a) for m, c, _, a in runs[:-1]} == {(m, c, a) for m in ("current", "nuclei", "sylber")
                                                      for c in (False, True) for a in ("A", "B")}
