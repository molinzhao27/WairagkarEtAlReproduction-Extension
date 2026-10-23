"""scripts/kunz_experiment.py end to end on a tiny synthetic fold, with TensorFlow, LPCNet and the audio metrics
replaced by deterministic fakes (CPU, < 2 s, no data).

The real run_fold / train_variant / decode_model / score_model / summarize run. Every trial's neural input is
a constant equal to its cue's id + 1, the fake decoder copies it into the output, the fake vocoder into the
waveform, and the fake mel_corr(ref, est) = 1 / (1 + |ref code - est code|). So an output scores exactly 1.0
against its own cue and less against every other cue, which makes the keying of predictions, trials, cues
and shuffled-trial partners checkable value by value. Code 0 is a silent output.

Covered: what is (re)trained / (re)decoded / (re)scored on a re-run and why (config hash, model hash,
decode settings, scoring settings, file times, --retrain, --overwrite-predictions, --stage), refusals with
a clear error, provenance in every output, and the cue nulls / ranks / score matrix.
"""

from __future__ import annotations

import concurrent.futures
import csv
import importlib.util
import json
import math
import os
from argparse import Namespace
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wvx import b2v
from wvx import kunz_experiment as kx
from wvx.config import REPO_ROOT
from wvx.experiment_config import load_experiment_config
from wvx.metrics import normalize_text
from wvx.session import SessionData, Trial

pytestmark = pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py (electrode map)")

PAIRS = {3: 4, 6: 7, 11: 9, 13: 12}
K, F, NT, W, HOP = 4, 512, 20, 60, 160
CV = {"source": "test", "commit": "abc123", "dirty": False, "diff_sha256": None}


def _script():
    spec = importlib.util.spec_from_file_location("kunz_experiment_resume", REPO_ROOT / "scripts" / "kunz_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------ fake world


def _cue_ids(data):
    keys = sorted({normalize_text(t.text) for s in data for t in s.trials})
    return {k: i for i, k in enumerate(keys)}


def _data(silent_cue: str):
    """Attempted (targets) and imagined blocks with the same cues (imagined in reverse order), eval block 14
    (one cue shared with A6). Neural input = cue id + 1 everywhere; the trial with ``silent_cue`` in I4 is 0."""
    rng = np.random.default_rng(0)
    att, img, idx_a, idx_i = [], [], 0, 0
    texts = {a: [f"Sentence {a} number {j}." for j in range(K)] for a in PAIRS}
    ev_texts = [f"Novel cue {j}" for j in range(K - 1)] + ["sentence 6 NUMBER 0"]
    ids = {k: i for i, k in enumerate(sorted({normalize_text(t) for v in texts.values() for t in v}
                                             | {normalize_text(t) for t in ev_texts}))}

    def trial(text, block, idx, target, code=None):
        n = int(rng.integers(W + 40, W + 120))
        c = ids[normalize_text(text)] + 1 if code is None else code
        return Trial(neural=np.full((n, F), c, np.float32), text=text, block=block, meta={"trial_index": idx},
                     target=rng.uniform(-1, 1, (n, NT)).astype(np.float32) if target else None)

    for a, i in PAIRS.items():
        for c in texts[a]:
            att.append(trial(c, a, idx_a, True))
            idx_a += 1
        for c in texts[a][::-1]:
            img.append(trial(c, i, idx_i, False, 0 if (i == 4 and c == silent_cue) else None))
            idx_i += 1
    ev = [trial(c, 14, j, False) for j, c in enumerate(ev_texts)]
    return SessionData("att", 0, att), SessionData("img", 0, img), SessionData("ev", 0, ev)


def _code(x):
    return float(np.asarray(x).flat[0])


def fake_mel(ref, est):
    e, r = _code(est), _code(ref)
    return float("nan") if e == 0 else 1.0 / (1.0 + abs(r - e))


class InlinePool:
    """ProcessPoolExecutor stand-in: initializer and jobs run in this process, so the fake metrics apply."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        assert mp_context.get_start_method() == "spawn" and initargs[-1] in (1, 2)   # real workers: spawned, <= 2 BLAS
        initializer(*initargs[:-1])                                                    # (no BLAS cap in the test process)

    def submit(self, fn, *args):
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as ex:
            f.set_exception(ex)
        return f

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class World:
    """The script module with fakes installed, a config pointing into tmp_path, and call counters."""

    def __init__(self, tmp_path, monkeypatch):
        from wvx import inference, metrics, synthesis, training
        from wvx import kunz_experiment as kxm

        self.ke = ke = _script()
        self.tmp = tmp_path
        self.silent_cue = "Sentence 3 number 1."
        self.data = _data(self.silent_cue)
        self.ids = _cue_ids(self.data)
        self.calls = {"train": 0, "load": [], "score": []}
        self.n_trained = 0
        for name, content in (("zs.h5", b"zero-shot model v1"), ("init.h5", b"init"), ("att.npz", b"attempted data")):
            (tmp_path / name).write_bytes(content)
        self.overrides = {
            "experiment": {"name": "t", "results_root": str(tmp_path / "results"), "audio_root": str(tmp_path / "audio"),
                           "checkpoints_root": str(tmp_path / "ckpt"), "zero_shot_model": str(tmp_path / "zs.h5"),
                           "attempted_file": str(tmp_path / "att.npz"), "folds": [3], "run_variants": ["frozen"],
                           "nulls": {"shuffled_time_models": ["trained"], "length_matched_k": 2},
                           "evaluation": {"n_workers": 2, "progress_every": 1000, "asr": {"enabled": False},
                                          "save_audio": {"predictions": [], "references": False}}},
            "training": {"init": {"path": str(tmp_path / "init.h5")}},
        }
        self.cfg = load_experiment_config("configs/experiments/kunz_pilot.yaml", self.overrides)
        world = self

        def fake_train(sessions, cfg, extra_callbacks=()):
            world.calls["train"] += 1
            world.n_trained += 1
            run = cfg["run_name"]
            h5 = Path(cfg["paths"]["checkpoints"]) / run / "inference_layer00.h5"
            h5.parent.mkdir(parents=True, exist_ok=True)
            h5.write_bytes(b"trained model #%d" % world.n_trained)   # every training exports a new file
            res = Path(cfg["paths"]["results"]) / run
            res.mkdir(parents=True, exist_ok=True)
            n = len(sessions[0].trials)
            split = res / "split.json"
            split.write_text(json.dumps({"sessions": {sessions[0].name: {
                "train": list(range(1, n)), "validation": [0], "excluded_too_short": []}}}))
            return SimpleNamespace(history={"loss": [1.0], "val_loss": [0.9]}, split_json=split, inference_models={0: h5})

        def fake_load(path, cfg=None):
            world.calls["load"].append(Path(path).name)
            return object()

        real_score = ke.score_model

        def spy_score(model, *a, **k):
            world.calls["score"].append(model)
            return real_score(model, *a, **k)

        def fake_refs(fold, cfg, dirs, pool, asr=None):
            texts = sorted({t.text for s in fold.sets for t in s.trials})
            code = {t: world.ids[normalize_text(t)] + 1 for t in texts}
            held = [t for s in fold.sets if s.role == "heldout_attempted" for t in s.trials]
            return SimpleNamespace(
                sr=16000, asr_rows=[], tts={t: np.full(2 * HOP, code[t], float) for t in texts},
                tts_speech_s={t: world.duration(t) for t in texts},
                target={t.meta["trial_index"]: np.full(2 * HOP, code[t.text], float) for t in held},
                target_vs_tts={t.meta["trial_index"]: {"mel_corr": 1.0} for t in held})

        monkeypatch.setattr(training, "train", fake_train)
        monkeypatch.setattr(inference, "load_model", fake_load)
        monkeypatch.setattr(kxm, "decode", lambda net, x, cfg: np.full((len(x) - W + 1, NT), x[0, 0], np.float32))
        monkeypatch.setattr(synthesis, "decoder_output_to_waveform", lambda pred, cfg: np.repeat(pred[:, 0], HOP))
        monkeypatch.setattr(metrics, "mel_spectrogram_correlation", lambda ref, est, sr, cfg=None: fake_mel(ref, est))
        monkeypatch.setattr(metrics, "evaluate", lambda ref, est, sr, cfg=None: {
            "mel_corr": fake_mel(ref, est), "mcd": 1.0, "ref_speech_fraction": 1.0,
            "est_speech_fraction": float(_code(est) != 0)})
        monkeypatch.setattr(metrics, "speech_frame_mask", lambda wav, sr, cfg=None: np.abs(np.asarray(wav)[::HOP]) > 0)
        monkeypatch.setattr(ke, "References", fake_refs)
        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InlinePool)   # audio metrics: see InlinePool
        monkeypatch.setattr(ke, "make_asr", lambda cfg: (None, "disabled in test"))
        monkeypatch.setattr(ke, "make_progress_callback", lambda every: None)
        monkeypatch.setattr(ke, "_clear_tf_session", lambda: None)
        monkeypatch.setattr(ke, "_code_version", lambda: CV)
        monkeypatch.setattr(ke, "score_model", spy_score)

    def duration(self, text):
        return 0.5 + 0.3 * (self.ids[normalize_text(text)] % 4)

    def run(self, stage="all", retrain=False, overwrite=False, **over):
        cfg = load_experiment_config("configs/experiments/kunz_pilot.yaml",
                                     _merge(self.overrides, over)) if over else self.cfg
        self.calls = {"train": 0, "load": [], "score": []}
        args = Namespace(stage=stage, retrain=retrain, overwrite_predictions=overwrite, config="test")
        return self.ke.run_fold(3, self.data, cfg, args, {"code_version": CV, "runtime": {}})

    @property
    def dirs(self):
        return self.ke.Dirs(self.cfg, "fold_A03")

    def scores(self, model):
        return json.loads(self.dirs.scores(model).read_text())


def _merge(a, b):
    from wvx.config import deep_merge

    return deep_merge(a, b)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


# ------------------------------------------------------------------------------------ keying


def test_first_run_keys_predictions_trials_cues_and_partners(world):
    s = world.run()
    assert world.calls == {"train": 1, "load": ["zs.h5", "inference_layer00.h5"], "score": ["zero_shot", "frozen"]}
    d = world.dirs
    fold = json.loads((d.results / "fold.json").read_text())
    for model in ("zero_shot", "frozen"):
        rows = world.scores(model)
        real = [r for r in rows if r["input"] == "real"]
        assert len(real) == sum(len(v["trials"]) for v in fold["sets"].values())
        for r in real:
            trials = fold["sets"][r["set"]]["trials"]
            texts = [t["text"] for t in trials]
            own_id = world.ids[normalize_text(r["text"])]
            cue_ids = list(dict.fromkeys(world.ids[normalize_text(t)] for t in texts))  # columns, first appearance
            if r["set"] == "paired_imagined_I04" and r["text"] == world.silent_cue:
                assert math.isnan(r["tts_mel_corr"] if r["tts_mel_corr"] is not None else float("nan"))
                assert r["own_rank"] is None and r["n_cues"] == 0 and r["est_speech_frames"] == 0
                assert r["peak_dbfs"] is None
                continue
            # prediction of THIS trial (own cue scores 1.0, rank 1), every cue of the set ranked
            assert r["tts_mel_corr"] == 1.0 and r["own_rank"] == 1 and r["n_cues"] == len(cue_ids)
            others = [1 / (1 + abs(c - own_id)) for c in cue_ids if c != own_id]
            assert r["null_all_mean"] == pytest.approx(np.mean(others))
            durs = {c: 0.5 + 0.3 * (c % 4) for c in cue_ids}
            near = sorted((c for c in cue_ids if c != own_id), key=lambda c: (abs(durs[c] - durs[own_id]), cue_ids.index(c)))
            assert r["null_lenmatched_mean"] == pytest.approx(np.mean([1 / (1 + abs(c - own_id)) for c in near[:2]]))
            assert r["cue_tts_speech_s"] == pytest.approx(durs[own_id])
            # shuffled-trial partner: the trial named in fold.json, scored against ITS cue
            partner = next(t for t in trials if t["trial_index"] == r["trial_index"])["shuffled_trial_partner"]
            assert r["shuffled_trial_partner"] == partner
            p_text = next(t["text"] for t in trials if t["trial_index"] == partner)
            assert r["shuffled_trial_tts_mel_corr"] == pytest.approx(1 / (1 + abs(world.ids[normalize_text(p_text)] - own_id)))
            assert r["est_speech_frames"] == r["n_frames"] and r["peak_dbfs"] == pytest.approx(20 * math.log10(own_id + 1))
            assert r["early_speech_s"] == pytest.approx(min(r["n_frames"], 400) * 0.01)
        # score matrix: one row per real output, own column = 1.0
        with open(d.score_matrix(model)) as f:
            matrix = list(csv.DictReader(f))
        assert len(matrix) == len(real)
        for m in matrix:
            if m["text"] != world.silent_cue or not m["set"].startswith("paired"):
                assert float(m[f"cue_{int(m['own_cue']):03d}"]) == 1.0
    shuf = [r for r in world.scores("frozen") if r["input"] == "shuffled_time"]
    assert shuf and all("own_rank" not in r for r in shuf)        # cue nulls only for real outputs
    assert not d.predictions("zero_shot", "shuffled_time").exists()

    # provenance everywhere
    assert fold["code_version"] == CV and len(fold["config_sha256"]) == 64
    assert s["code_version"] == CV and s["config_sha256"] == fold["config_sha256"] and s["git_commit"] == "abc123"
    info = json.loads(d.training_info("frozen").read_text())
    assert info["code_version"] == CV and len(info["training_config_sha256"]) == 64
    assert info["exported_sha256"] == world.ke._sha256(d.h5("frozen"))
    for r in world.scores("frozen"):
        assert r["model_sha256"] == info["exported_sha256"] and r["code_version"] == "abc123"
        assert r["predictions_sha256"] == world.ke._sha256(d.predictions("frozen", r["input"]))
    assert s["models"]["frozen"]["predictions"]["real"]["model_sha256"] == info["exported_sha256"]
    header = (d.results / "per_trial.csv").read_text().splitlines()[0].split(",")
    assert "predictions_sha256" not in header and "null_lenmatched_mean" in header

    # per-fold summary: cue nulls, primary first, identification, sensitivity
    g = s["results"]["per_model"]["zero_shot"]["heldout_attempted_A03"]
    assert list(g["nulls"]) == ["lenmatched", "all", "shuffled_trial"]
    assert g["identification"]["top1_rate"] == 1.0 and g["identification"]["n"] == K
    img = s["results"]["per_model"]["zero_shot"]["paired_imagined_I04"]
    assert img["identification"]["n_undefined"] == 1 and img["silent_outputs"]["n_silent"] == 1
    avi = s["results"]["per_model"]["frozen"]["attempted_vs_paired_imagined"]
    assert list(avi["excess_difference"]) == ["lenmatched", "all", "shuffled_trial"]
    assert "sensitivity_min_speech" in g and s["results"]["settings"]["primary_null"] == "lenmatched"


class FakeASR:
    """Transcript = the cue (lower case); fails on a silent output (code 0)."""

    def __call__(self, wav, sr, text):
        if _code(wav) == 0:
            raise RuntimeError("no audio")
        return {"asr_hyp_raw": text.lower(), "asr_word_errors": 0, "asr_ref_words": len(text.split())}

    def rescore(self, text, hyp):
        return {"asr_word_errors": int(text.lower() != hyp), "asr_ref_words": len(text.split())}


def test_row_fields_keep_their_order_with_audio_scores_from_worker_processes(world, monkeypatch):
    """Audio scores come back from a worker process while the thread runs ASR; the row is still assembled in the
    order of the per-trial table: trial, audio scores, target_vs_tts, ASR, shuffled-trial ASR, provenance, nulls."""
    monkeypatch.setattr(world.ke, "make_asr", lambda cfg: (FakeASR(), "fake"))
    world.run()
    rows = world.scores("frozen")
    evaluate = ["mel_corr", "mcd", "ref_speech_fraction", "est_speech_fraction"]
    nulls = list(kx.cue_null_stats([1.0, 0.5], 0, [1.0, 2.0], 1))
    head = list(world.ke.BASE_FIELDS) + [f"tts_{k}" for k in evaluate] + [
        "est_speech_frames", "est_speech_s", "early_speech_s", "peak_dbfs"]
    asr = ["asr_hyp_raw", "asr_word_errors", "asr_ref_words"]
    prov = ["predictions_sha256", "model_sha256", "score_settings_sha256", "config_sha256", "code_version"]
    held = next(r for r in rows if r["input"] == "real" and r["role"] == "heldout_attempted")
    assert list(held) == head + ["shuffled_trial_partner"] + [f"shuffled_trial_tts_{k}" for k in evaluate] + [
        "feat_mse", "feat_corr_mean"] + [f"target_{k}" for k in evaluate] + ["target_vs_tts_mel_corr"] + asr + [
        "shuffled_trial_asr_word_errors", "shuffled_trial_asr_ref_words"] + prov + nulls
    assert held["asr_hyp_raw"] == held["text"].lower() and held["shuffled_trial_asr_word_errors"] == 1
    shuf = next(r for r in rows if r["input"] == "shuffled_time" and r["role"] == "heldout_attempted")
    assert list(shuf) == head + ["feat_mse", "feat_corr_mean"] + [f"target_{k}" for k in evaluate] + [
        "target_vs_tts_mel_corr"] + asr + prov
    img = next(r for r in rows if r["input"] == "real" and r["role"] == "paired_imagined")
    assert list(img) == head + ["shuffled_trial_partner"] + [f"shuffled_trial_tts_{k}" for k in evaluate] + asr + [
        "shuffled_trial_asr_word_errors", "shuffled_trial_asr_ref_words"] + prov + nulls
    silent = next(r for r in world.scores("zero_shot")
                  if r["input"] == "real" and r["set"] == "paired_imagined_I04" and r["text"] == world.silent_cue)
    assert silent["error"] == "RuntimeError: no audio" and "asr_hyp_raw" not in silent   # ASR failure recorded
    assert list(silent).index("error") == list(silent).index("shuffled_trial_tts_est_speech_fraction") + 1


# ------------------------------------------------------------------------------------ resume


def test_rerun_reuses_up_to_date_outputs(world):
    world.run()
    before = world.dirs.scores("frozen").stat().st_mtime_ns
    world.run()
    assert world.calls == {"train": 0, "load": [], "score": []}
    assert world.dirs.scores("frozen").stat().st_mtime_ns == before
    world.run(experiment={"evaluation": {"progress_every": 7, "n_workers": 1}})   # changes nothing that is scored
    assert world.calls == {"train": 0, "load": [], "score": []}


def test_changed_training_config_refuses_instead_of_reusing_or_retraining(world):
    world.run()
    with pytest.raises(world.ke.StaleOutputError, match=r"training\.epochs.*--retrain"):
        world.run(training={"epochs": 6})
    assert world.calls == {"train": 0, "load": [], "score": []}
    with pytest.raises(world.ke.StaleOutputError, match="--retrain"):    # --stage decode checks it too
        world.run(stage="decode", training={"epochs": 6})
    (world.tmp / "att.npz").write_bytes(b"other attempted data")           # input data identity
    with pytest.raises(world.ke.StaleOutputError, match="input_files_sha256.attempted_file"):
        world.run()
    world.run(retrain=True)
    assert world.calls["train"] == 1


def test_retrain_redecodes_and_rescores_that_variant(world):
    world.run()
    old = world.ke._sha256(world.dirs.h5("frozen"))
    s = world.run(retrain=True)
    new = world.ke._sha256(world.dirs.h5("frozen"))
    assert new != old
    assert world.calls == {"train": 1, "load": ["inference_layer00.h5"], "score": ["frozen"]}
    assert {r["model_sha256"] for r in world.scores("frozen")} == {new}
    assert s["models"]["frozen"]["sha256"] == new and s["training"]["frozen"]["exported_sha256"] == new


def test_replaced_zero_shot_model_is_redecoded_and_rescored(world):
    world.run()
    (world.tmp / "zs.h5").write_bytes(b"zero-shot model, a different file")
    world.run()
    assert world.calls == {"train": 0, "load": ["zs.h5"], "score": ["zero_shot"]}
    assert {r["model_sha256"] for r in world.scores("zero_shot")} == {world.ke._sha256(world.tmp / "zs.h5")}


def test_changed_decode_setting_is_redecoded(world):
    world.run()
    with pytest.raises(world.ke.StaleOutputError, match="reorder_channels"):   # training-relevant: refused ...
        world.run(experiment={"reorder_channels": False})
    assert world.calls == {"train": 0, "load": [], "score": []}                # ... before any other work
    world.run(experiment={"reorder_channels": False, "run_variants": []})      # zero-shot only: re-decoded
    assert world.calls == {"train": 0, "load": ["zs.h5"], "score": ["zero_shot"]}
    with np.load(world.dirs.predictions("zero_shot", "real")) as z:
        assert json.loads(str(z["header"]))["decode_settings"]["reorder_channels"] is False


def test_changed_scoring_setting_rescores_without_decoding(world):
    world.run()
    s = world.run(experiment={"nulls": {"length_matched_k": 3}})
    assert world.calls == {"train": 0, "load": [], "score": ["zero_shot", "frozen"]}
    assert s["results"]["settings"]["length_matched_k"] == 3


def test_newer_predictions_or_overwrite_rescore(world):
    world.run()
    p = world.dirs.predictions("zero_shot", "real")
    t = world.dirs.scores("zero_shot").stat().st_mtime + 10
    os.utime(p, (t, t))
    world.run()
    assert world.calls == {"train": 0, "load": [], "score": ["zero_shot"]}
    world.run(overwrite=True)
    assert world.calls == {"train": 0, "load": ["zs.h5", "inference_layer00.h5"], "score": ["zero_shot", "frozen"]}


def test_stage_decode_leaves_no_summary_built_on_stale_scores(world):
    world.run()
    summary = world.dirs.results / "summary.json"
    before = summary.read_text()
    (world.tmp / "zs.h5").write_bytes(b"zero-shot model v2 (longer)")
    out = world.run(stage="decode")
    assert world.calls == {"train": 0, "load": ["zs.h5"], "score": []}
    assert out["scored"] is False and "zero_shot" in out["stale_scores"]
    assert summary.read_text() == before                                  # not rewritten from stale scores
    world.run(stage="score")
    assert world.calls["score"] == ["zero_shot", "frozen"] and world.calls["load"] == []
    assert json.loads(summary.read_text())["models"]["zero_shot"]["sha256"] == world.ke._sha256(world.tmp / "zs.h5")


def test_scoring_refuses_predictions_of_another_model_or_decode_setting(world):
    world.run()
    with pytest.raises(world.ke.StaleOutputError, match="reorder_channels was True, now False.*decode again"):
        world.run(stage="score", experiment={"reorder_channels": False, "run_variants": []})
    (world.tmp / "zs.h5").write_bytes(b"zero-shot model v3 (not decoded yet)")
    with pytest.raises(world.ke.StaleOutputError, match="current model is.*decode again"):
        world.run(stage="score")


def test_older_training_info_and_foreign_h5(world):
    world.run()
    d = world.dirs
    info = json.loads(d.training_info("frozen").read_text())
    legacy = {k: v for k, v in info.items() if k not in ("training_config_sha256", "training_fingerprint")}
    d.training_info("frozen").write_text(json.dumps(legacy))
    with pytest.raises(world.ke.StaleOutputError, match="cannot be checked"):
        world.run()                                                         # never skips training unverified
    s = world.run(stage="score")                                            # reuse is possible, flagged
    assert s["training"]["frozen"]["training_config_verified"] is False
    d.training_info("frozen").write_text(json.dumps(info))
    d.h5("frozen").write_bytes(b"some other model file")
    with pytest.raises(world.ke.StaleOutputError, match="not the file its training exported"):
        world.run()
