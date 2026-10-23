"""Fixes to wvx.asr (2026-10-21 audit): per-clip sampling seeds and silent-clip skipping.

No faster-whisper needed: Whisper is replaced by a stand-in that behaves like faster-whisper's temperature fallback
on top of CTranslate2's RNG (one generator per worker thread, created from the global seed at the thread's first
sampling op; a newly loaded model has new threads). With that RNG, reseeding a loaded model does nothing, which is why
wvx.asr runs the sampling steps of a decode on a fresh model right after setting the clip's seed (D-056; until then it
redid the whole decode on a fresh model, ``_old_scheme`` below).
"""

from __future__ import annotations

import inspect
import math
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from wvx import asr, metrics

NO_PER = {"per": {"enabled": False}}
SR = 16000


def _clip(k: int, seconds: float = 1.0) -> np.ndarray:
    """A deterministic non-silent clip; odd k -> its first sample is positive (the stand-in then needs fallback)."""
    rng = np.random.default_rng(1000 + k)
    t = np.arange(int(seconds * SR)) / SR
    x = 0.2 * np.sin(2 * np.pi * (150 + 10 * k) * t) * (0.6 + 0.4 * rng.random(t.size))
    x[0] = 0.05 if k % 2 else -0.05
    return x


# ------------------------------------------------------------------------------------ stand-in for Whisper


class _CT2:
    """Global seed + one lazily created generator per 'worker thread' (= per model instance), as CTranslate2."""

    seed = None
    created = []

    def __init__(self):
        self._rng = None
        self.generate_calls = []
        self.seed_at_load = _CT2.seed
        _CT2.created.append(self)

    def generate(self, audio, **kw):
        self.generate_calls.append(kw)
        if kw.get("sampling_topk", 1) != 1:
            if self._rng is None:  # thread_local std::mt19937 generator(get_random_seed())
                self._rng = np.random.default_rng(_CT2.seed if _CT2.seed is not None else None)
            return int(self._rng.integers(1 << 30))
        return 0


class _FakeWhisper:
    """faster-whisper-like transcribe: temperature 0 = beam search (no sampling kwargs); a 'degenerate' clip (first
    sample > 0) falls back to the sampled temperatures, each draw changing the text."""

    def __init__(self):
        self.model = asr._SamplingRecorder(_CT2())
        self.n_transcribe = 0

    def transcribe(self, audio, **dec):
        self.n_transcribe += 1
        text, temp = "beam", 0.0
        for t in dec["temperature"]:
            kw = {"beam_size": dec["beam_size"]} if t == 0 else {"beam_size": 1, "sampling_topk": 0,
                                                                  "sampling_temperature": t}
            draw = self.model.generate(audio, **kw)
            text, temp = ("beam" if t == 0 else f"sampled {draw}"), t
            if not audio[0] > 0 or t > 0:
                break
        seg = SimpleNamespace(text=text, avg_logprob=-0.5, no_speech_prob=0.1, compression_ratio=1.2,
                              temperature=temp)
        return iter([seg]), None


def _settle():
    """Wait for background preloads (the stand-in loads instantly, but in another thread)."""
    for f in list(asr._SPARES.values()):
        f.result()


@pytest.fixture
def fake_whisper(monkeypatch):
    """Persistent model + fresh models + the global seed, all stand-ins. Yields a namespace for inspection.
    Fresh models: ``_load_ct2`` makes a stand-in CTranslate2 model (also when preloaded in the background)."""
    _CT2.seed, _CT2.created = None, []
    asr._SPARES.clear()
    persistent = _FakeWhisper()
    ns = SimpleNamespace(persistent=persistent, seeds_set=[])

    def set_seed(seed):
        ns.seeds_set.append(int(seed))
        _CT2.seed = int(seed)

    monkeypatch.setattr(asr, "_model", lambda c: persistent)
    monkeypatch.setattr(asr, "_load_ct2", lambda c, d: _CT2())
    monkeypatch.setattr(asr, "_set_random_seed", set_seed)
    yield ns
    _settle()
    asr._SPARES.clear()


def _old_scheme(audio_wav, cfg):
    """The pre-D-056 scheme with the same stand-ins: decode on the persistent model; if it sampled, set the clip's seed,
    load a new model and redo the whole decode on it."""
    c = asr._cfg(cfg)
    audio = asr.prepare_audio(audio_wav, SR, c)
    seed = asr.clip_seed(audio, asr.base_seed(c))
    segments, sampled = asr._decode(asr._model(c), audio, dict(c["decode"]))
    if sampled:
        asr._set_random_seed(seed)
        segments, sampled = asr._decode(_FakeWhisper(), audio, dict(c["decode"]))
    return segments[0].text, sampled


# ------------------------------------------------------------------------------------------- seed derivation


def test_clip_seed_depends_only_on_the_samples_and_the_config_seed():
    a = _clip(1)
    s = asr.clip_seed(a, 42)
    assert s == asr.clip_seed(a.copy(), 42)                       # another array, same samples
    assert s == asr.clip_seed(a.astype(np.float32), 42)           # what Whisper is given: float32
    assert s == asr.clip_seed(np.concatenate([a, a])[: a.size], 42)
    assert s == asr.clip_seed(np.repeat(a, 2)[::2], 42)          # non-contiguous view
    assert s != asr.clip_seed(a, 43)
    b = a.copy()
    b[-1] += 1e-3
    assert s != asr.clip_seed(b, 42)
    assert s != asr.clip_seed(a[:-1], 42)
    assert all(0 <= asr.clip_seed(_clip(k), 42) < 2**32 - 1 for k in range(20))   # 2**32 - 1 = CTranslate2 "unset"
    assert len({asr.clip_seed(_clip(k), 42) for k in range(20)}) == 20


def test_base_seed_defaults_to_the_repo_seed():
    from wvx.config import load_config

    assert asr.base_seed() == load_config()["seed"]
    assert asr.base_seed({"seed": 7}) == 7


def test_sampling_recorder_flags_only_sampling_calls():
    rec = asr._SamplingRecorder(_CT2())
    rec.generate(np.zeros(3), beam_size=5)
    assert not rec.sampled
    rec.generate(np.zeros(3), beam_size=1, sampling_topk=0, sampling_temperature=0.2)
    assert rec.sampled
    assert rec.generate_calls[0] == {"beam_size": 5}   # other attributes are forwarded


# -------------------------------------------------------------------------- reproducibility (order, threads)


def _run(order, threads=1):
    cfg = {**NO_PER}
    clips = {k: _clip(k) for k in order}
    if threads == 1:
        out = {k: asr.transcribe_detailed(clips[k], SR, cfg) for k in order}
    else:
        with ThreadPoolExecutor(threads) as ex:
            out = dict(zip(order, ex.map(lambda k: asr.transcribe_detailed(clips[k], SR, cfg), order)))
    return out


def test_stand_in_reproduces_the_original_bug(fake_whisper):
    """Sanity check of the stand-in: decoding on the persistent model only (the old code, seeded once per process)
    makes a sampled transcript depend on what was sampled before."""
    _CT2.seed = 42
    m = _FakeWhisper()
    audio = _clip(1).astype(np.float32)
    first = next(m.transcribe(audio, temperature=[0.0, 0.2], beam_size=5)[0]).text
    _CT2.seed = 42   # reseeding a loaded model has no effect (thread-local generator already exists)
    second = next(m.transcribe(audio, temperature=[0.0, 0.2], beam_size=5)[0]).text
    assert first.startswith("sampled") and first != second


def test_transcripts_do_not_depend_on_order_or_threads(fake_whisper):
    order = list(range(6))
    a = _run(order)
    b = _run(order[::-1])
    c = _run([3, 0, 5, 1, 4, 2] * 2, threads=4)   # repeats and a thread pool
    for k in order:
        assert a[k]["text"] == b[k]["text"] == c[k]["text"], k
        assert a[k]["asr_seed"] == b[k]["asr_seed"] == c[k]["asr_seed"] == \
            asr.clip_seed(asr.prepare_audio(_clip(k), SR), 42)
        assert a[k]["asr_sampled"] is (k % 2 == 1)
        assert a[k]["skipped_silent"] is False and a[k]["speech_frames"] > 0
        assert a[k]["text"].startswith("sampled" if k % 2 else "beam")
    assert len({a[k]["text"] for k in order if k % 2}) == 3   # different clips, different seeds


def test_sampling_steps_run_on_a_fresh_model_after_seeding(fake_whisper):
    out = asr.transcribe_detailed(_clip(1), SR, {**NO_PER, "preload_fresh_model": False})
    assert out["asr_sampled"] and out["max_temperature"] > 0
    assert fake_whisper.seeds_set == [out["asr_seed"]]
    persistent = fake_whisper.persistent.model._inner
    fresh = [m for m in _CT2.created if m is not persistent]
    assert len(fresh) == 1 and fake_whisper.persistent.n_transcribe == 1
    # beam search on the cached model, the first sampling step (and everything after it) on the fresh one
    assert [k.get("sampling_topk", 1) for k in persistent.generate_calls] == [1]
    assert [k.get("sampling_topk", 1) for k in fresh[0].generate_calls] == [0]
    # the draw is what a model loaded right after setting the clip's seed would make
    assert out["text"] == f"sampled {np.random.default_rng(out['asr_seed']).integers(1 << 30)}"
    assert fake_whisper.persistent.model.fresh is None and fake_whisper.persistent.model.seed is None


def test_same_transcripts_as_redoing_the_whole_decode(fake_whisper):
    """D-056 identity: routing the sampling steps to a fresh model (with or without a preloaded spare) gives the
    transcripts of the old scheme, which redid the whole decode on a model loaded right after seeding."""
    old = {k: _old_scheme(_clip(k), NO_PER) for k in range(6)}
    assert sum(sampled for _, sampled in old.values()) == 3
    for preload in (False, True):
        cfg = {**NO_PER, "preload_fresh_model": preload}
        for k in [5, 0, 3, 3, 1, 4, 2, 1]:
            out = asr.transcribe_detailed(_clip(k), SR, cfg)
            assert (out["text"], out["asr_sampled"]) == old[k], (preload, k)


def test_preloaded_spare_is_loaded_in_the_background_and_used_once(fake_whisper):
    cfg = {**NO_PER, "preload_fresh_model": True}
    asr.transcribe_detailed(_clip(0), SR, cfg)              # no sampling: nothing loaded
    assert not asr._SPARES and len(_CT2.created) == 1
    first = asr.transcribe_detailed(_clip(1), SR, cfg)      # loads a fresh model now, then preloads a spare
    _settle()
    assert len(asr._SPARES) == 1 and len(_CT2.created) == 3
    spare = _CT2.created[2]
    assert spare.seed_at_load == first["asr_seed"] and not spare.generate_calls   # loaded under ANOTHER clip's seed
    out = asr.transcribe_detailed(_clip(3), SR, cfg)        # takes the spare, seeds it, preloads the next one
    _settle()
    assert len(spare.generate_calls) == 1 and out["asr_seed"] != first["asr_seed"]
    assert out["text"] == f"sampled {np.random.default_rng(out['asr_seed']).integers(1 << 30)}"
    assert len(_CT2.created) == 4 and not _CT2.created[3].generate_calls
    asr.transcribe_detailed(_clip(2), SR, cfg)              # beam only: the spare stays unused
    _settle()
    assert len(_CT2.created) == 4 and len(asr._SPARES) == 1


def test_recorder_switches_once_and_forwards_everything_after():
    inner, fresh, seeds = _CT2(), _CT2(), []
    rec = asr._SamplingRecorder(inner)
    rec.seed, rec.take_fresh = 7, lambda: (seeds.append("taken"), fresh)[1]
    import unittest.mock

    with unittest.mock.patch.object(asr, "_set_random_seed", lambda s: seeds.append(s)):
        rec.generate(np.zeros(3), beam_size=5)
        assert rec.fresh is None and not rec.sampled and rec.generate_calls is inner.generate_calls
        rec.generate(np.zeros(3), beam_size=1, sampling_topk=0, sampling_temperature=0.2)
        rec.generate(np.zeros(3), beam_size=1, sampling_topk=0, sampling_temperature=0.4)
        rec.generate(np.zeros(3), beam_size=5)
    assert seeds == ["taken", 7] and rec.sampled and rec.fresh is fresh
    assert len(inner.generate_calls) == 1 and len(fresh.generate_calls) == 3
    assert rec.generate_calls is fresh.generate_calls               # attributes now come from the fresh model


def test_beam_only_decodes_never_seed_or_reload(fake_whisper):
    for k in range(4):
        asr.transcribe_detailed(_clip(k), SR, {**NO_PER, "decode": {"temperature": [0.0]}})
    assert fake_whisper.seeds_set == [] and _CT2.created == [fake_whisper.persistent.model._inner]
    out = asr.transcribe_detailed(_clip(0), SR, NO_PER)    # fallback configured but not needed: no reload either
    assert out["asr_sampled"] is False and fake_whisper.seeds_set == [] and not asr._SPARES


# --------------------------------------------------------------------------------------------- silent clips


@pytest.mark.parametrize("wav", [np.zeros(2 * SR), np.zeros(2 * SR, np.int16), np.zeros(SR, np.float32),
                                 np.zeros(0)], ids=["float64", "int16", "float32", "empty"])
def test_silent_clip_skips_whisper(fake_whisper, wav):
    out = asr.evaluate_wav(wav, SR, "I hope it is you.", NO_PER)
    assert fake_whisper.persistent.n_transcribe == 0 and fake_whisper.seeds_set == []
    assert out["skipped_silent"] is True and out["speech_frames"] == 0
    assert out["hyp_raw"] == "" and out["hyp_empty"] and out["n_segments"] == 0
    assert out["asr_seed"] is None and out["asr_sampled"] is False
    assert out["wer"] == 1.0 and out["word_errors"] == 5
    assert math.isnan(out["max_temperature"])


def test_all_zero_audio_has_no_speech_frames_under_the_metrics_detector():
    assert not metrics.speech_frame_mask(np.zeros(SR), SR).any()
    assert asr.speech_frames(np.zeros(SR), SR) == 0
    # LPCNet's output for minimal energy (c0) is exact digital zero (checked with third_party/LPCNet on the Mac:
    # 0 of 32000 samples non-zero), i.e. this case. One LSB of int16 noise is NOT silence for the relative detector:
    lsb = np.zeros(SR, np.int16)
    lsb[SR // 2] = 1
    assert asr.speech_frames(lsb, SR) > 0


def test_speech_frames_is_the_metrics_mask_on_the_full_scale_signal():
    x = _clip(2)
    x[: SR // 4] = 0.0                      # digital silence: never speech
    x[SR // 4: SR // 2] *= 0.1              # -20 dB: speech at threshold_db 40, silence at 3
    want = int(metrics.speech_frame_mask(x, SR).sum())
    assert 0 < want < metrics.speech_frame_mask(x, SR).size
    assert asr.speech_frames(x, SR) == want
    assert asr.speech_frames((x * 32767).astype(np.int16), SR) == int(
        metrics.speech_frame_mask((x * 32767).astype(np.int16) / 32768.0, SR).sum())
    stricter = {"silence": {"metrics_overrides": {"silence": {"threshold_db": 3.0}}}}
    assert asr.speech_frames(x, SR, stricter) < want


def test_skip_can_be_switched_off(fake_whisper):
    out = asr.transcribe_detailed(np.zeros(SR), SR, {**NO_PER, "silence": {"skip_silent": False}})
    assert fake_whisper.persistent.n_transcribe == 1
    assert out["skipped_silent"] is False and out["speech_frames"] is None and out["text"] == "beam"


def test_near_silent_clip_is_still_transcribed(fake_whisper):
    x = np.zeros(SR)
    x[100:400] = 1e-3 * np.sin(np.arange(300))
    out = asr.transcribe_detailed(x, SR, NO_PER)
    assert out["skipped_silent"] is False and fake_whisper.persistent.n_transcribe == 1


# ------------------------------------------------------------------------------------------- schema / settings


def test_public_signatures_unchanged_for_existing_callers():
    """scripts/kunz_experiment.py binds evaluate_wav(wav, sr, reference_text, cfg) and score_text(ref, hyp, cfg)."""
    assert list(inspect.signature(asr.evaluate_wav).parameters) == ["wav", "sr", "reference_text", "cfg"]
    assert list(inspect.signature(asr.score_text).parameters)[:3] == ["reference_text", "hypothesis_text", "cfg"]


def test_evaluate_wav_keeps_old_keys_and_adds_diagnostics(fake_whisper):
    out = asr.evaluate_wav(_clip(1), SR, "I hope it is you.", NO_PER)
    old = {"hyp_raw", "ref", "hyp", "hyp_constrained", "wer", "cer", "per", "wer_constrained", "word_errors",
           "ref_words", "n_segments", "avg_logprob", "no_speech_prob", "max_compression_ratio", "max_temperature",
           "audio_s", "asr_s"}
    assert old <= set(out)
    assert {"skipped_silent", "speech_frames", "asr_seed", "asr_sampled"} <= set(out)


def test_prompt_method_records_its_own_sampling_flag(fake_whisper):
    out = asr.evaluate_wav(_clip(1), SR, "I hope it is you.", {**NO_PER, "constrained": {"method": "prompt_snap"}})
    assert out["asr_sampled_prompted"] is True and "hyp_raw_prompted" in out


def test_effective_settings_track_what_changes_a_score():
    import json

    base = asr.effective_settings()
    assert json.loads(json.dumps(base)) == base
    assert base["seeding"] == asr.SEEDING and base["seed"] == 42
    assert base["silence"]["skip_silent"] is True and "threshold_db" in base["silence"]["detector"]["silence"]
    assert "cpu_threads" not in base and "preload_fresh_model" not in base
    assert asr.effective_settings({"cpu_threads": 2}) == base      # speed only
    assert asr.effective_settings({"preload_fresh_model": False}) == base
    for override in ({"decode": {"temperature": [0.0]}}, {"silence": {"skip_silent": False}},
                     {"silence": {"metrics_overrides": {"silence": {"threshold_db": 30.0}}}}, {"seed": 1},
                     {"constrained": {"method": "prompt"}}):
        assert asr.effective_settings(override) != base, override


def test_config_documents_the_seeding_and_has_a_beam_variant():
    c = asr.load_asr_config()
    assert c["silence"]["skip_silent"] is True and c["silence"]["metrics_overrides"] == {}
    assert c["variants"]["beam"]["decode"]["temperature"] == [0.0]
    assert any(t > 0 for t in c["decode"]["temperature"])

