"""Scoring in worker processes (scripts/kunz_experiment.py ``audio_scorer``), on synthetic audio.

``wvx.kunz_experiment.audio_scores`` / ``score_output`` run in spawned processes with one BLAS thread each, as the
experiment runs them, must give exactly the in-process result (same fields, same order, same float bits), and the
same numbers as calling the metrics directly, as the scoring threads did before this work moved to processes.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

from wvx import b2v
from wvx import kunz_experiment as kx
from wvx import metrics
from wvx.config import REPO_ROOT

SR = 16000
TTS_METRICS = {"evaluate": {"mel_corr_framewise": False}}   # configs/experiments/kunz_full.yaml
TARGET_METRICS: dict = {}
HOP_S = float(metrics.load_metrics_config()["stft"]["hop_ms"]) / 1000.0


def _voice(f0, seconds, seed):
    """Harmonic bursts with silent gaps (syllable-like), plus a little noise."""
    t = np.arange(int(seconds * SR)) / SR
    on = (np.sin(2 * np.pi * 3.0 * t) > -0.3).astype(float)
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * on
    return (0.3 * x + 0.003 * np.random.default_rng(seed).standard_normal(t.size)).astype(np.float32)


def _bits(v):
    """A value with every float as its 8 bytes (so NaN == NaN, and 0.0 != -0.0), keys in their order."""
    if isinstance(v, float):
        return ("float", struct.pack("<d", v))
    if isinstance(v, dict):
        return ("dict", [(k, _bits(x)) for k, x in v.items()])
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, [_bits(x) for x in v])
    if isinstance(v, np.ndarray):
        return ("ndarray", v.dtype.str, v.shape, v.tobytes())
    return (type(v).__name__, v)


def _workers(refs, n=2):
    return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                               initializer=kx.init_output_scoring, initargs=(*refs, 1))


def _inline(refs, fn, *args):
    kx.init_output_scoring(*refs)
    try:
        return [fn(*a) for a in zip(*args)]
    finally:
        kx._REFS.clear()


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    tts = {f"cue {j}": _voice(110 + 30 * j, 0.6 + 0.2 * j, j) for j in range(4)}
    target = {7: _voice(140, 1.0, 10)}
    pred = rng.uniform(-1, 1, (90, 20)).astype(np.float32)
    trial_target = rng.uniform(-1, 1, (149, 20)).astype(np.float32)
    out = _voice(150, 1.1, 20)
    cases = [
        (out, kx.OutputJob(pred, "cue 1", partner=(3, "cue 2"), cues=list(tts), heldout=(7, trial_target))),  # real
        (_voice(125, 0.9, 21), kx.OutputJob(pred, "cue 1", heldout=(7, trial_target))),          # time-shuffled
        (_voice(170, 1.3, 22), kx.OutputJob(pred, "cue 3", partner=(0, "cue 0"), cues=list(tts))),     # imagined
        (np.zeros(8000, np.float32), kx.OutputJob(pred, "cue 0", partner=(1, "cue 1"), cues=list(tts))),  # silent
        (out, kx.OutputJob(pred, "cue 2", heldout=(7, trial_target[:50]))),                      # target too short
    ]
    wavs, jobs = [c[0] for c in cases], [c[1] for c in cases]
    refs = (None, tts, target, SR, TTS_METRICS, TARGET_METRICS, HOP_S, 0.5)
    return refs, wavs, jobs, _inline(refs, kx.audio_scores, wavs, jobs)


def test_worker_processes_give_the_in_process_scores_bit_for_bit(world):
    refs, wavs, jobs, inline = world
    with _workers(refs) as ex:
        scores = list(ex.map(kx.audio_scores, wavs, jobs))
    assert [_bits(r) for r in scores] == [_bits(r) for r in inline]   # fields, their order, types and float bits


def test_audio_scores_are_the_metrics_called_directly(world):
    (_, tts, target, sr, tts_cfg, tgt_cfg, hop_s, early_s), wavs, jobs, inline = world
    real, shuffled, imagined, silent, short = inline
    wav = wavs[0]
    own, partner = metrics.evaluate(tts["cue 1"], wav, sr, tts_cfg), metrics.evaluate(tts["cue 2"], wav, sr, tts_cfg)
    assert [real[f"tts_{k}"] for k in own] == list(own.values())
    assert [real[f"shuffled_trial_tts_{k}"] for k in partner] == list(partner.values())
    assert real["shuffled_trial_partner"] == 3
    assert real["_cue_scores"] == [metrics.mel_spectrogram_correlation(tts[c], wav, sr, tts_cfg) for c in tts]
    assert np.isfinite(real["_cue_scores"]).all()
    stats = kx.output_speech_stats(wav, metrics.speech_frame_mask(wav, sr, tts_cfg), hop_s, early_s)
    assert all(real[k] == v for k, v in stats.items())
    tgt = metrics.evaluate(target[7], wav, sr, tgt_cfg)
    assert [real[f"target_{k}"] for k in tgt] == list(tgt.values()) and "target_mel_corr_framewise" in real
    feat = kx.feature_scores(jobs[0].pred, jobs[0].heldout[1][-90:])
    assert real["feat_mse"] == feat["feat_mse"] and real["feat_corr_mean"] == feat["feat_corr_mean"]
    assert "_cue_scores" not in shuffled and "shuffled_trial_partner" not in shuffled and "target_mcd" in shuffled
    assert "feat_mse" not in imagined and len(imagined["_cue_scores"]) == len(tts)
    assert np.isnan(silent["_cue_scores"]).all() and silent["est_speech_frames"] == 0   # undefined, not an error
    assert short["error"].startswith("ValueError: cannot align 90 predictions") and "feat_mse" not in short


@pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py")
def test_lpcnet_audio_and_scores_in_a_worker_process(world, tmp_path):
    from wvx import synthesis
    from wvx.experiment_config import load_experiment_config

    cfg = load_experiment_config("configs/experiments/kunz_pilot.yaml",
                                 {"synthesis": {"tmp_root": str(tmp_path / "tmp")}})
    try:
        synthesis.preflight(cfg)
    except synthesis.LPCNetError:
        pytest.skip("LPCNet binary not built (scripts/build_lpcnet.sh)")
    refs, _, jobs, _ = world
    refs = (cfg, *refs[1:])
    with _workers(refs, n=1) as ex:
        got = list(ex.map(kx.score_output, jobs[:2]))
    want = _inline(refs, kx.score_output, jobs[:2])
    assert [_bits(r) for r in got] == [_bits(r) for r in want]
    wav = synthesis.decoder_output_to_waveform(jobs[0].pred, cfg)
    np.testing.assert_array_equal(got[0][0], wav)
    assert got[0][0].shape == (90 * int(cfg["synthesis"]["samples_per_frame"]),)


# ----------------------------------------------------------------------------------------- ASR


def fake_transcribe(wav, sr, reference_text, cfg=None):
    """Stands in for wvx.asr.evaluate_wav (importable by the worker processes: this module is on their path)."""
    return {"hyp_raw": f"{reference_text} @{cfg['asr']['tag']}", "samples": int(np.size(wav)),
            "checksum": float(np.sum(np.asarray(wav, float) ** 2)), "pid": os.getpid()}


def fake_rescore(reference_text, hypothesis_text, cfg):
    return {"word_errors": int(reference_text != hypothesis_text), "pid": os.getpid()}


def test_asr_in_worker_processes_returns_what_it_returns_in_process():
    spec = importlib.util.spec_from_file_location("kunz_experiment_asr", REPO_ROOT / "scripts" / "kunz_experiment.py")
    ke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ke)
    asr = ke.ASR(sys.modules[__name__], {"module": __name__, "function": "fake_transcribe",
                                         "rescore_function": "fake_rescore", "cfg": {"asr": {"tag": 3}}})
    clips = [(_voice(100 + 20 * j, 0.3 + 0.1 * j, j), f"cue {j}") for j in range(6)]

    def calls():
        with ThreadPoolExecutor(4) as threads:   # as from the scoring threads
            out = list(threads.map(lambda c: asr(c[0], SR, c[1]), clips))
        return out + [asr.rescore("cue 1", "cue one")]

    inline = calls()
    with asr.worker_processes(2):
        pooled = calls()
    assert asr.pool is None
    assert {r["asr_pid"] for r in inline} == {os.getpid()} and os.getpid() not in {r["asr_pid"] for r in pooled}
    strip = [{k: v for k, v in r.items() if k != "asr_pid"} for r in inline]
    assert [_bits(r) for r in strip] == [_bits({k: v for k, v in r.items() if k != "asr_pid"}) for r in pooled]
    assert strip[1] == {"asr_hyp_raw": "cue 1 @3", "asr_samples": clips[1][0].size,
                        "asr_checksum": float(np.sum(clips[1][0].astype(float) ** 2))}
    assert strip[-1] == {"asr_word_errors": 1}
