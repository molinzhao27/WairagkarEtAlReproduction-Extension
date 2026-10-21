"""Overwrite guards (2026-10-21 audit): scripts/asr_eval.py and scripts/run_inference.py never silently replace
outputs that were produced with different settings. No data, no Whisper; TensorFlow only for run_inference.run."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from wvx import asr
from wvx.config import REPO_ROOT, deep_merge, load_config

DAY195 = REPO_ROOT / "results" / "reproduce_inference_t15_day195"


def _script(name):
    spec = importlib.util.spec_from_file_location(f"fixes_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ae():
    return _script("asr_eval")


@pytest.fixture(scope="module")
def ri():
    return _script("run_inference")


# ------------------------------------------------------------------------------------------------ asr_eval


def _settings(ae, override=None, sources=("target", "pred")):
    acfg = asr.load_asr_config(overrides=override)
    return ae.run_settings(acfg, list(sources), None, None)


def test_asr_eval_writes_into_an_empty_directory(ae, tmp_path):
    assert ae.check_existing(tmp_path / "a.csv", tmp_path / "a.json", _settings(ae), False) is None


def test_asr_eval_reruns_with_identical_settings(ae, tmp_path):
    s = _settings(ae)
    (tmp_path / "a.csv").write_text("x\n")
    (tmp_path / "a.json").write_text(json.dumps({"effective_settings": s}))
    assert ae.check_existing(tmp_path / "a.csv", tmp_path / "a.json", _settings(ae), False) is None


def test_asr_eval_refuses_different_settings_unless_overwrite(ae, tmp_path):
    (tmp_path / "a.csv").write_text("x\n")
    (tmp_path / "a.json").write_text(json.dumps({"effective_settings": _settings(ae, {"decode": {"temperature": [0.0]}})}))
    msg = ae.check_existing(tmp_path / "a.csv", tmp_path / "a.json", _settings(ae), False)
    assert msg and "asr.decode.temperature: [0.0] -> [0.0, 0.2" in msg and "--overwrite" in msg
    assert ae.check_existing(tmp_path / "a.csv", tmp_path / "a.json", _settings(ae), True) is None
    other_sources = _settings(ae, sources=("pred",))
    (tmp_path / "a.json").write_text(json.dumps({"effective_settings": _settings(ae)}))
    assert "sources" in ae.check_existing(tmp_path / "a.csv", tmp_path / "a.json", other_sources, False)


def test_asr_eval_refuses_a_csv_without_summary(ae, tmp_path):
    (tmp_path / "a.csv").write_text("x\n")
    assert "settings are unknown" in ae.check_existing(tmp_path / "a.csv", tmp_path / "a.json", _settings(ae), False)


@pytest.mark.skipif(not (DAY195 / "asr_summary.json").is_file(), reason="committed ASR summary not present")
def test_asr_eval_refuses_to_replace_the_committed_beam_only_summary(ae, tmp_path):
    """The committed unsuffixed files are beam-only and predate effective_settings: the default (fallback) config
    and even the beam variant (now with silent-clip skipping and per-clip seeding) must not replace them."""
    shutil.copy(DAY195 / "asr_summary.json", tmp_path / "asr_summary.json")
    (tmp_path / "asr_per_trial.csv").write_text("x\n")
    msg = ae.check_existing(tmp_path / "asr_per_trial.csv", tmp_path / "asr_summary.json",
                            _settings(ae, sources=("target", "pred", "live")), False)
    assert "predates effective-settings" in msg and "asr.decode.temperature: [0.0] -> [0.0, 0.2" in msg
    beam = _settings(ae, asr.load_asr_config()["variants"]["beam"], sources=("target", "pred", "live"))
    msg = ae.check_existing(tmp_path / "asr_per_trial.csv", tmp_path / "asr_summary.json", beam, False)
    assert msg and "decode.temperature" not in msg and "asr.silence" in msg


def test_asr_eval_main_stops_before_loading_the_model(ae, tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "per_trial_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "text", "target_wav", "pred_wav"])
        w.writeheader()
        w.writerow({"trial": 0, "text": "hello", "target_wav": "t.wav", "pred_wav": "p.wav"})
    old = {"effective_settings": _settings(ae, {"decode": {"temperature": [0.0]}})}
    (run / "asr_summary.json").write_text(json.dumps(old))
    (run / "asr_per_trial.csv").write_text("old rows\n")

    def no_model(*a, **k):
        raise AssertionError("the model must not be loaded when the run is refused")

    seeds = []
    monkeypatch.setattr(ae.asr, "ensure_model", no_model)
    # wvx.seeding.set_seed imports TensorFlow (and with it Keras 3, which would break later wvx.b2v users in
    # this pytest process): record the call instead.
    monkeypatch.setattr(ae, "set_seed", seeds.append)
    monkeypatch.setattr(sys, "argv", ["asr_eval.py", "--results-dir", str(run)])
    assert ae.main() == 2
    assert seeds == [42]
    assert (run / "asr_per_trial.csv").read_text() == "old rows\n"
    assert json.loads((run / "asr_summary.json").read_text()) == old


def test_asr_eval_summary_counts_skipped_and_sampled_clips(ae):
    nan = float("nan")
    base = {"type": "t", "source": "pred", "hyp_empty": True, "ref_oov_words": 0, "max_compression_ratio": nan,
            "asr_s": 0.0, "audio_s": 1.0, **{k: nan for k in ae.RATES}}
    rows = [{**base, "skipped_silent": True, "asr_sampled": False, "max_temperature": nan},
            {**base, "skipped_silent": False, "asr_sampled": True, "max_temperature": 0.2,
             "max_compression_ratio": 1.0}]
    s = ae.summarize(rows, 2.4)["pred"]["ALL"]
    assert s["skipped_silent_fraction"] == 0.5 and s["sampled_fraction"] == 0.5 and s["fallback_used_fraction"] == 0.5


# ------------------------------------------------------------------------------------------- run_inference


def _ri_cfg(tmp_path, **inference):
    cfg = load_config("configs/experiments/reproduce_inference.yaml")
    model = tmp_path / "model.h5"
    model.write_bytes(b"not a real model")
    return deep_merge(cfg, {"run_name": "run", "paths": {"results": tmp_path / "results", "pretrained_model": model,
                                                         "wairagkar_neural_data": tmp_path / "no_data"},
                            "inference": inference})


def _write_run(ri, cfg, n_trials_run, extra_files=()):
    out = Path(cfg["paths"]["results"]) / cfg["run_name"]
    out.mkdir(parents=True)
    mp = Path(cfg["paths"]["pretrained_model"])
    (out / "summary.json").write_text(json.dumps({"run_settings": ri.run_settings(cfg, mp, ri._sha256(mp)),
                                                  "n_trials_run": n_trials_run}))
    (out / "per_trial_metrics.csv").write_text("trial\n0\n")
    for name in extra_files:
        (out / name).write_text("{}")
    return out


def test_run_inference_guard(ri, tmp_path):
    cfg305 = _ri_cfg(tmp_path)
    out = _write_run(ri, cfg305, 305, extra_files=["asr_summary.json"])
    mp = Path(cfg305["paths"]["pretrained_model"])
    sha = ri._sha256(mp)
    assert ri.check_existing(out, ri.run_settings(cfg305, mp, sha), False) == (None, None)
    cfg5 = deep_merge(cfg305, {"inference": {"max_trials": 5}})
    msg, prev = ri.check_existing(out, ri.run_settings(cfg5, mp, sha), False)
    assert "max_trials: None -> 5" in msg and "n_trials_run 305" in msg and "asr_summary.json" in msg
    for change in ({"inference": {"session": "t15_day193"}}, {"inference": {"reorder_channels": False}}):
        assert ri.check_existing(out, ri.run_settings(deep_merge(cfg305, change), mp, sha), False)[0]
    assert ri.check_existing(out, ri.run_settings(cfg305, mp, "0" * 64), False)[0]          # another model file
    msg, prev = ri.check_existing(out, ri.run_settings(cfg5, mp, sha), True)
    assert msg is None and prev["n_trials_run"] == 305 and prev["derived_files_now_stale"] == ["asr_summary.json"]


@pytest.mark.skipif(not (DAY195 / "summary.json").is_file(), reason="committed day195 summary not present")
def test_run_inference_guard_reads_summaries_written_before_run_settings(ri, tmp_path):
    out = tmp_path / "reproduce_inference_t15_day195"
    out.mkdir()
    shutil.copy(DAY195 / "summary.json", out / "summary.json")
    old = json.loads((out / "summary.json").read_text())
    assert "run_settings" not in old
    cfg = load_config("configs/experiments/reproduce_inference.yaml")
    mp = Path(cfg["paths"]["pretrained_model"])
    same = ri.run_settings(cfg, mp, old["model"]["sha256"])
    assert ri.check_existing(out, same, False) == (None, None)
    msg, _ = ri.check_existing(out, ri.run_settings(deep_merge(cfg, {"inference": {"max_trials": 5}}), mp,
                                                    old["model"]["sha256"]), False)
    assert "max_trials: None -> 5" in msg and "n_trials_run 305" in msg


def test_run_inference_refuses_before_loading_any_data(ri, tmp_path):
    if importlib.util.find_spec("tf_keras") is None:   # run() sets up Keras 2 itself; importing TF here would not
        pytest.skip("tf_keras not installed")
    cfg = _ri_cfg(tmp_path)
    out = _write_run(ri, cfg, 305)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError, match="max_trials: None -> 5"):
        ri.run(deep_merge(cfg, {"inference": {"max_trials": 5}}))   # no data exists: refused before reading any
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    with pytest.raises(Exception) as e:                                # with --overwrite it gets as far as the data
        ri.run(deep_merge(cfg, {"inference": {"max_trials": 5}}), overwrite=True)
    assert not isinstance(e.value, FileExistsError)


def test_asr_eval_records_settings_and_accepts_a_faithful_rerun(ae, tmp_path, monkeypatch):
    """Full write path with Whisper replaced by a stand-in: the summary records effective_settings, code_version and
    config_sha256; the same command runs again; a different max_trials is refused."""
    import numpy as np
    import soundfile as sf
    import yaml
    from types import SimpleNamespace

    from wvx.config import load_yaml

    acfg = load_yaml(REPO_ROOT / "configs" / "asr.yaml")
    acfg["asr"]["per"]["enabled"] = False                       # g2p_en is not needed for this check
    (tmp_path / "asr.yaml").write_text(yaml.safe_dump(acfg, sort_keys=False))
    run = tmp_path / "run"
    run.mkdir()
    sr = 16000
    tone = 0.2 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
    sf.write(run / "silent.wav", np.zeros(sr), sr)
    sf.write(run / "tone.wav", tone, sr)
    with open(run / "per_trial_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "text", "target_wav", "pred_wav"])
        w.writeheader()
        w.writerow({"trial": 0, "text": "hello", "target_wav": str(run / "tone.wav"), "pred_wav": str(run / "silent.wav")})
        w.writerow({"trial": 1, "text": "you up", "target_wav": str(run / "tone.wav"), "pred_wav": str(run / "tone.wav")})

    class Fake:
        model = asr._SamplingRecorder(SimpleNamespace(generate=lambda *a, **k: None))
        n = 0

        def transcribe(self, audio, **dec):
            Fake.n += 1
            return iter([SimpleNamespace(text="hello", avg_logprob=-0.2, no_speech_prob=0.0, compression_ratio=1.0,
                                         temperature=0.0)]), None

    monkeypatch.setattr(ae.asr, "ensure_model", lambda c: tmp_path)
    monkeypatch.setattr(ae.asr, "_model", lambda c: Fake())
    monkeypatch.setattr(ae, "set_seed", lambda seed: None)       # keeps TensorFlow out of this process
    argv = ["asr_eval.py", "--results-dir", str(run), "--asr-config", str(tmp_path / "asr.yaml")]
    monkeypatch.setattr(sys, "argv", argv)
    assert ae.main() == 0
    assert Fake.n == 3                                          # the silent clip never reached Whisper
    s = json.loads((run / "asr_summary.json").read_text())
    assert s["effective_settings"]["asr"]["silence"]["skip_silent"] is True
    assert s["effective_settings"]["sources"] == ["target", "pred"] and len(s["config_sha256"]) == 64
    assert s["code_version"]["source"] in ("git", "sync", None)
    assert s["results"]["pred"]["ALL"]["skipped_silent_fraction"] == 0.5
    with open(run / "asr_per_trial.csv") as f:
        rows = list(csv.DictReader(f))
    silent = [r for r in rows if r["wav"].endswith("silent.wav")]
    assert len(silent) == 1 and silent[0]["skipped_silent"] == "True" and silent[0]["hyp_raw"] == ""
    assert ae.main() == 0                                       # same settings: a faithful re-run is allowed
    monkeypatch.setattr(sys, "argv", argv + ["--max-trials", "1"])
    assert ae.main() == 2                                       # different settings: refused
    monkeypatch.setattr(sys, "argv", argv + ["--max-trials", "1", "--overwrite"])
    assert ae.main() == 0
    assert json.loads((run / "asr_summary.json").read_text())["effective_settings"]["max_trials"] == 1
