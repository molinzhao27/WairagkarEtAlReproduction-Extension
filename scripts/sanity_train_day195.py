#!/usr/bin/env python
"""Training-code sanity check: the Kunz pilot recipe on THEIR data (t15_day195) with THEIR targets (host, CPU).

    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/sanity_train_day195.py \\
        --config configs/experiments/sanity_train_day195.yaml [--smoke] [--variants theirs ours] [--stage all|score]

Why: the Kunz pilot (D-029) made held-out attempted decoding slightly worse than the zero-shot day-193 decoder.
This repeats the pilot's training exactly (``wvx.training.train``; one new input layer copied from the day-193
``.h5``, frozen base, 5 epochs, same optimizer/augmentation/validation size; every input through
``wvx.kunz_experiment.prepare_neural`` = reorder_channels ON) on t15_day195, whose real targets are known.

Data: ``wvx.wairagkar_data.load_session`` (targets = ``target_speech_feats_word``, scaled + smoothed as their
loader does); trial types from ``trial_description`` in the neural .mat. Training = the "Training data..."
trials (validation drawn from them, seeded); test = every "Evaluation trial..." trial.

Steps:
1. Zero-shot: the day-193 decoder as is (reorder ON) decodes the test trials.
2. Per target variant (``sanity.target_variants``): ``theirs`` = their targets; ``ours`` (optional) = our
   option-(b) targets for the same training trials (``wvx.targets.align_trials`` with the day-193 decoder and the
   Aman TTS of the cue, exactly as for Kunz; cached under data/processed/<name>/); ``tts_oracle`` / ``tts_auto``
   (D-049): the Aman TTS with forced-aligned syllables, arm B, placed on THEIR true syllables / on the c0-dip
   syllables of the init decoder's own output (the Kunz v2 pipeline; settings from ``sanity.tts_targets.config``). The validation loss of the
   initial model (epoch 0) is measured first, then ``wvx.training.train`` runs, then the exported model decodes.
3. Scoring per test trial, always against THEIR targets: frame-wise feature correlation (mean over the 20
   features) with the decoder target; ``mel_corr`` etc. (``wvx.metrics.evaluate``, configs/metrics.yaml) vs the
   LPCNet resynthesis of their raw stored targets; ASR (``wvx.asr.evaluate_wav``, unconstrained), also on the
   target audio as a ceiling. ASR runs serially in trial order (reproducible temperature fallback).
4. Summary: per trial type and pooled; paired Wilcoxon (``wvx.kunz_experiment.paired``) trained - zero-shot.

Re-runnable: training is skipped when its exported ``.h5`` exists, decoding when its predictions exist, scoring
when its scores exist (``--stage score`` rescoring everything from the saved predictions).

Outputs (``sanity.name`` N):
    results/N/ per_trial.csv, summary.json, scores_<model>.json, asr_target_audio.json, data.json
    results/N/train_<variant>/ history.csv, split.json, config.yaml, training_info.json
    results/audio/N/ predictions/<model>.npz, <model>/trNNN.wav, target/trNNN.wav (gitignored)
    data/checkpoints/N/train_<variant>/ best.tf, final.tf, inference_layer00.h5
    data/processed/N/ours_targets.npz (variant ours), <variant>_targets.npz (tts_oracle, tts_auto)

``--device gpu`` runs on a GPU (the spare PC, D-042); the device is recorded in summary.json.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; keep third_party/ free of __pycache__

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import REPO_ROOT, deep_merge, resolve_path  # noqa: E402  (yaml only)
from wvx.experiment_config import apply_tf_threads, apply_thread_env, load_experiment_config  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/sanity_train_day195.yaml"
ZERO_SHOT = "zero_shot"
# Per-trial metrics summarized and compared (direction: higher is better except feat_mse, mcd and the ASR rates).
METRICS = ("feat_corr_mean", "feat_mse", "mel_corr", "mel_corr_framewise", "mcd", "est_speech_fraction",
           "asr_wer", "asr_cer", "asr_per")
LOWER_IS_BETTER = ("feat_mse", "mcd", "asr_wer", "asr_cer", "asr_per")


def log(msg: str) -> None:
    print(f"[sanity_train_day195 {datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.decode().strip() + ("+uncommitted" if dirty.strip() else "")


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return _rel(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and obj != obj:
        return None  # NaN -> null in JSON
    return obj


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2))
    os.replace(tmp, path)


def _read_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    for r in rows:  # JSON null -> NaN for the statistics
        for k, v in list(r.items()):
            if v is None:
                r[k] = float("nan")
    return rows


class Dirs:
    def __init__(self, cfg):
        s = cfg["sanity"]
        self.results = resolve_path(s["results_root"]) / s["name"]
        self.audio = resolve_path(s["audio_root"]) / s["name"]
        self.ckpt = resolve_path(s["checkpoints_root"]) / s["name"]
        self.targets = resolve_path(s["targets_root"]) / s["name"]
        self.pred = self.audio / "predictions"
        for d in (self.results, self.audio, self.pred):
            d.mkdir(parents=True, exist_ok=True)

    def h5(self, variant: str) -> Path:
        return self.ckpt / f"train_{variant}" / "inference_layer00.h5"

    def model_path(self, model: str, cfg) -> Path:
        return resolve_path(cfg["sanity"]["zero_shot_model"]) if model == ZERO_SHOT else self.h5(model[len("trained_"):])

    def predictions(self, model: str) -> Path:
        return self.pred / f"{model}.npz"


# ---------------------------------------------------------------------------------------- data


def load_day195(cfg, max_train=None, max_test_per_type=None):
    """Session (their targets), raw targets, per-trial type; training and test trial indices."""
    import numpy as np
    import scipy.io

    from wvx import wairagkar_data

    s = cfg["sanity"]
    data_dir = resolve_path(s["data_dir"])
    sess = wairagkar_data.load_session(data_dir, s["session"], int(s["layer"]), cfg)
    raw = wairagkar_data.load_raw_targets(data_dir, s["session"], cfg)
    neural_path = wairagkar_data.session_files(data_dir, s["session"], cfg)[0]
    key = s["description_key"]
    desc = [str(x).strip() for x in np.atleast_1d(
        scipy.io.loadmat(str(neural_path), squeeze_me=True, variable_names=[key])[key])]
    if len(desc) != len(sess.trials):
        raise ValueError(f"{len(desc)} trial descriptions for {len(sess.trials)} trials")
    for t, d in zip(sess.trials, desc):
        t.meta["type"] = d
    train = [i for i, d in enumerate(desc) if d in s["train_types"]]
    test = [i for i, d in enumerate(desc) if d in s["test_types"]]
    other = sorted({desc[i] for i in range(len(desc)) if i not in set(train) | set(test)})
    if other:
        raise ValueError(f"trial types neither trained nor tested: {other}")
    want = s["expected_counts"]
    if (len(train), len(test)) != (int(want["train"]), int(want["test"])):
        raise ValueError(f"{len(train)} training / {len(test)} test trials, expected {want}")
    counts = {d: desc.count(d) for d in sorted(set(desc))}
    if max_train:
        train = train[:max_train]
    if max_test_per_type:
        test = sorted(i for ty in s["test_types"] for i in [j for j in test if desc[j] == ty][:max_test_per_type])
    log(f"loaded {s['session']}: {len(sess.trials)} trials {counts}; training {len(train)}, test {len(test)}")
    return sess, raw, desc, train, test, counts


def ours_targets(sess, train, cfg, dirs: Dirs) -> tuple[dict, dict]:
    """Our option-(b) targets for the training trials (cached), and how they compare with theirs."""
    import numpy as np
    import tensorflow as tf

    from wvx import inference, kunz_experiment as kx, targets as tg

    s, o = cfg["sanity"], cfg["sanity"]["ours_targets"]
    path = dirs.targets / "ours_targets.npz"
    keys = [f"tr{i:03d}" for i in train]
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            if all(k in z.files for k in keys):
                log(f"ours: targets loaded from {_rel(path)}")
                return {i: z[k] for i, k in zip(train, keys)}, header["info"]
        log(f"ours: {_rel(path)} lacks some trials; regenerating")
    t0 = time.time()
    voice = o["voice"] or cfg["tts"]["voice"]
    corpus_file = tg.corpus_path(cfg, voice, o["corpus"])
    corpus = tg.TTSCorpus.load(corpus_file)
    model_path = resolve_path(o["decoder"])
    model = inference.load_model(model_path, cfg)
    reorder = bool(cfg["experiment"]["reorder_channels"])
    trials = [sess.trials[i] for i in train]
    log(f"ours: aligning the TTS ({voice}) of {len(trials)} training cues to the day-193 decoder output "
        f"(reorder {reorder})")
    _, results = tg.align_trials(model, trials, corpus, cfg, reorder=reorder, log=log)
    out = {i: tg.decoder_target(res.target_raw, cfg) for i, res in zip(train, results)}
    for i, y in out.items():
        if y.shape != sess.trials[i].target.shape:
            raise ValueError(f"trial {i}: our target {y.shape}, theirs {sess.trials[i].target.shape}")
    r = [kx.feature_scores(out[i], sess.trials[i].target)["feat_corr_mean"] for i in train]
    info = {
        "method": "wvx.targets.align_trials + decoder_target (D-020/D-027), as scripts/align_targets.py",
        "voice": voice, "corpus": _rel(corpus_file), "decoder": _rel(model_path),
        "decoder_sha256": _sha256(model_path), "reorder_channels": reorder, "n_trials": len(train),
        "ours_vs_theirs_word_target_r": kx.describe(r),
        "n_mapped_syllables": kx.describe([len(res.mapped_segments) for res in results]),
        "wall_time_s": time.time() - t0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, header=np.array(json.dumps(_jsonable({"info": info, "trials": train}))),
                        **{k: out[i] for i, k in zip(train, keys)})
    os.replace(tmp, path)
    log(f"ours: {len(out)} targets in {info['wall_time_s']:.0f} s -> {_rel(path)}; r vs their targets "
        f"{info['ours_vs_theirs_word_target_r'].get('mean', float('nan')):.3f}")
    del model
    tf.keras.backend.clear_session()
    return out, info


TTS_VARIANTS = {
    "tts_oracle": "TTS syllables (forced) on THEIR true syllables (syl_indices), arm B",
    "tts_auto": "TTS syllables (forced) on the c0-dip syllables of the init decoder's own output (B/current), arm B",
}


def _valid_spans(segs, n_bins: int) -> bool:
    import numpy as np

    segs = np.asarray(segs)
    return bool(segs.size and segs.ndim == 2 and np.all(segs[:, 1] > segs[:, 0]) and np.all(segs[1:, 0] >= segs[:-1, 1])
                and segs[0, 0] >= 0 and segs[-1, 1] <= n_bins)


def tts_targets(variant: str, sess, train, cfg, dirs: Dirs) -> tuple[dict, dict]:
    """D-049 targets from the TTS voice for the training trials (cached): the Kunz target pipeline
    (``sanity.tts_targets.config``: forced TTS syllables, arm B, ``wvx.targets.decoder_target``) with the decoder-side
    syllables taken from THEIR syl_indices (``tts_oracle``) or from the c0 dips of the init decoder's own output
    (``tts_auto``). A trial whose syl_indices are malformed uses the c0-dip syllables (counted)."""
    import numpy as np
    import tensorflow as tf

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_alignment as va
    from wvx import alignment as al, inference, kunz_experiment as kx, syllables as sy, targets as tg

    s = cfg["sanity"]
    path = dirs.targets / f"{variant}_targets.npz"
    keys = [f"tr{i:03d}" for i in train]
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            if all(k in z.files for k in keys):
                log(f"{variant}: targets loaded from {_rel(path)}")
                return {i: z[k] for i, k in zip(train, keys)}, header["info"]
        log(f"{variant}: {_rel(path)} lacks some trials; regenerating")
    t0 = time.time()
    cfg_t = load_experiment_config(s["tts_targets"]["config"])
    voice = cfg_t["tts"]["voice"]
    corpus_file = tg.corpus_path(cfg_t, voice, s["session"])
    corpus = tg.TTSCorpus.load(corpus_file)
    sil20 = tg.silence_frame(corpus, cfg_t)
    table = sy.load_tts_syllables(cfg_t, voice, s["session"])
    trials = [sess.trials[i] for i in train]
    tts_segs = [table[tg.text_key(t.text)] for t in trials]
    reorder = bool(cfg["experiment"]["reorder_channels"])
    model_path = resolve_path(cfg["training"]["init"]["path"])
    model = inference.load_model(model_path, cfg_t)
    preds = [inference.predict_trial(model, t.neural, cfg_t, reorder=reorder) for t in trials]
    auto = [sy.decoder_segments("current", al.envelope_db(tg.prediction_timeline(p, t.neural.shape[0], sil20, cfg_t),
                                                          cfg_t), cfg_t, None) for p, t in zip(preds, trials)]
    n_fallback = 0
    if variant == "tts_oracle":
        v = cfg_t["validation"]
        si = resolve_path(s["data_dir"]) / s["session"] / v["speech_indices_file"].format(session=s["session"])
        syl = va.load_intervals(si, v["syllable_key"], len(sess.trials))
        pred_segs = []
        for i, t, a in zip(train, trials, auto):
            ok = _valid_spans(syl[i], t.neural.shape[0])
            n_fallback += not ok
            pred_segs.append(syl[i] if ok else a)
    else:
        pred_segs = auto
    log(f"{variant}: placing the TTS ({voice}) of {len(trials)} training cues ({TTS_VARIANTS[variant]}; "
        f"{n_fallback} fallbacks)")
    _, results = tg.align_trials(None, trials, corpus, cfg_t, reorder=reorder, preds=preds, tts_segments=tts_segs,
                                 pred_segments=pred_segs, arm="B", log=log)
    out = {i: tg.decoder_target(res.target_raw, cfg_t) for i, res in zip(train, results)}
    for i, y in out.items():
        if y.shape != sess.trials[i].target.shape:
            raise ValueError(f"trial {i}: {variant} target {y.shape}, theirs {sess.trials[i].target.shape}")
    r = [kx.feature_scores(out[i], sess.trials[i].target)["feat_corr_mean"] for i in train]
    pairing = [(res.info or {}).get("pairing") for res in results]
    info = {
        "variant": variant, "method": TTS_VARIANTS[variant] + "; wvx.targets.align_trials + decoder_target",
        "settings_config": s["tts_targets"]["config"], "voice": voice, "corpus": _rel(corpus_file),
        "decoder_for_auto_syllables": _rel(model_path), "decoder_sha256": _sha256(model_path),
        "reorder_channels": reorder, "n_trials": len(train), "n_fallback_to_auto": n_fallback,
        "pairing": {k: pairing.count(k) for k in sorted(set(map(str, pairing)))},
        "vs_theirs_word_target_r": kx.describe(r),
        "n_mapped_syllables": kx.describe([len(res.mapped_segments) for res in results]),
        "wall_time_s": time.time() - t0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, header=np.array(json.dumps(_jsonable({"info": info, "trials": train}))),
                        **{k: out[i] for i, k in zip(train, keys)})
    os.replace(tmp, path)
    log(f"{variant}: {len(out)} targets in {info['wall_time_s']:.0f} s -> {_rel(path)}; r vs their targets "
        f"{info['vs_theirs_word_target_r'].get('mean', float('nan')):.3f}; pairing {info['pairing']}")
    del model
    tf.keras.backend.clear_session()
    return out, info


# ------------------------------------------------------------------------------------ training


def make_progress_callback(every: int):
    import tensorflow as tf

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch, self.t0 = epoch, time.time()

        def on_train_batch_end(self, batch, logs=None):
            if every and (batch + 1) % every == 0:
                log(f"  epoch {self.epoch + 1} batch {batch + 1}/{self.params.get('steps')} "
                    f"loss {logs.get('loss', float('nan')):.5f} ({time.time() - self.t0:.0f} s)")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            log(f"  epoch {epoch + 1} done in {time.time() - self.t0:.0f} s: loss {logs.get('loss', float('nan')):.5f} "
                f"val_loss {logs.get('val_loss', float('nan')):.5f}")

    return Progress()


def initial_validation_loss(tsess, cfg) -> dict:
    """Validation loss of the INITIAL model (epoch 0), on the same validation trials ``train`` will use."""
    import tensorflow as tf

    from wvx import training as tr

    t, m = cfg["training"], cfg["model"]
    split = tr.split_trials([tsess], cfg)
    val_b = tr.SessionBatcher([tsess], split.validation, shuffle=False, batch_size=t["batch_size"],
                              time_steps=m["window_len"], step_size=m["step_size"], seed=cfg["seed"],
                              drop_remainder=t["drop_remainder"])
    model = tr._initial_model(cfg, tr.n_session_layers([tsess], cfg))
    out = model.evaluate(val_b.dataset(m["n_features"], m["n_targets"]), verbose=0, return_dict=True)
    del model
    tf.keras.backend.clear_session()
    return {k: float(v) for k, v in out.items()}


def train_variant(variant: str, sess, train, targets: dict, cfg, dirs: Dirs, retrain: bool) -> dict:
    from wvx import kunz_experiment as kx
    from wvx.session import SessionData, Trial
    from wvx.training import train as fit

    s, t = cfg["sanity"], cfg["training"]
    run = f"train_{variant}"
    info_path = dirs.results / run / "training_info.json"
    if dirs.h5(variant).is_file() and not retrain:
        log(f"{variant}: exported model exists, training skipped ({_rel(dirs.h5(variant))})")
        return json.loads(info_path.read_text()) if info_path.is_file() else {"skipped": True}
    vcfg = deep_merge(cfg, {
        "run_name": run,
        "paths": {"results": dirs.results, "checkpoints": dirs.ckpt},
        "training": {"overwrite": bool(retrain) or bool(t["overwrite"])},
    })
    trials = [Trial(neural=kx.prepare_neural(sess.trials[i].neural, vcfg), target=targets[i],
                    text=sess.trials[i].text, meta={"trial_index": i, "type": sess.trials[i].meta["type"]})
              for i in train]
    tsess = SessionData(name=s["train_session_name"], layer=int(s["layer"]), trials=trials)
    log(f"{variant}: initial validation loss ...")
    init_val = initial_validation_loss(tsess, vcfg)
    log(f"{variant}: initial (epoch 0) {init_val}")
    log(f"{variant}: training on {len(trials)} trials (freeze_base={t['init']['freeze_base']}, epochs {t['epochs']}, "
        f"validation {t['validation']['n_trials']}, reorder {cfg['experiment']['reorder_channels']})")
    t0 = time.time()
    result = fit([tsess], vcfg, extra_callbacks=[make_progress_callback(int(t.get("log_every_batches") or 0))])
    wall = time.time() - t0
    h = result.history
    split = json.loads(result.split_json.read_text())["sessions"][tsess.name]
    info = {
        "variant": variant,
        "targets": {"theirs": "their target_speech_feats_word (scaled + smoothed by their loader)",
                    "ours": "our option-(b) alignment (wvx.targets), scaled + smoothed by their transform"}.get(
                        variant, TTS_VARIANTS.get(variant, variant)),
        "freeze_base": t["init"]["freeze_base"],
        "init": _rel(resolve_path(t["init"]["path"])),
        "n_train_trials": len(split["train"]),
        "n_validation_trials": len(split["validation"]),
        "validation_trials": [tsess.trials[j].meta["trial_index"] for j in split["validation"]],
        "validation_types": {ty: sum(tsess.trials[j].meta["type"] == ty for j in split["validation"])
                             for ty in s["train_types"]},
        "excluded_too_short": split["excluded_too_short"],
        "epochs_configured": t["epochs"],
        "epochs_run": len(h.get("loss", [])),
        "initial_validation": init_val,
        "history": h,
        "loss_note": "loss = their train_step loss (Huber + 1.5 x Huber on features 4-9; dropout on; augmented data); "
                     "val_loss = plain Huber, dropout off (Keras test_step); initial_validation = val_loss before training",
        "export_weights": t["export_weights"],
        "wall_time_s": wall,
        "exported": _rel(result.inference_models[0]),
        "exported_sha256": _sha256(result.inference_models[0]),
    }
    _write_json(info_path, info)
    log(f"{variant}: trained in {wall / 60:.1f} min; val_loss {init_val.get('loss')} (init) -> {h.get('val_loss')}; "
        f"loss {h.get('loss')}")
    import tensorflow as tf

    del result
    tf.keras.backend.clear_session()
    return info


# ------------------------------------------------------------------------------------ decoding


def decode_model(model: str, sess, test, cfg, dirs: Dirs, overwrite: bool) -> None:
    import numpy as np
    import tensorflow as tf

    from wvx import inference, kunz_experiment as kx

    out = dirs.predictions(model)
    if out.is_file() and not overwrite:
        log(f"{model}: predictions exist, decoding skipped")
        return
    path = dirs.model_path(model, cfg)
    net = inference.load_model(path, cfg)
    t0, preds = time.time(), {}
    for i in test:
        preds[f"tr{i:03d}"] = kx.decode(net, kx.prepare_neural(sess.trials[i].neural, cfg), cfg)
        if len(preds) % int(cfg["sanity"]["evaluation"]["progress_every"]) == 0:
            log(f"  {model}: decoded {len(preds)}/{len(test)} ({time.time() - t0:.0f} s)")
    header = {"model": model, "model_path": _rel(path), "model_sha256": _sha256(path),
              "reorder_channels": bool(cfg["experiment"]["reorder_channels"]), "seed": cfg["seed"],
              "created": datetime.datetime.now().isoformat(timespec="seconds")}
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez_compressed(tmp, header=np.array(json.dumps(header)), **preds)
    os.replace(tmp, out)
    log(f"{model}: {len(preds)} trials decoded in {time.time() - t0:.0f} s -> {_rel(out)}")
    del net
    tf.keras.backend.clear_session()


# ------------------------------------------------------------------------------------- scoring


class Scorer:
    """Target audio (LPCNet resynthesis of their raw targets over the predicted frames) and the ASR."""

    def __init__(self, sess, raw, test, cfg, dirs: Dirs, pool):
        import threading

        from wvx import inference, synthesis

        self.cfg, self.dirs, self.pool, self.sess, self.test = cfg, dirs, pool, sess, test
        self.sr = int(cfg["synthesis"]["sample_rate"])
        e = cfg["sanity"]["evaluation"]
        self.save = bool(e["save_audio"])
        self.asr_cfg = e["asr"]["cfg"] if e["asr"]["enabled"] else None
        self.lock = threading.Lock()
        w = int(cfg["model"]["window_len"])

        def target(i):
            n = sess.trials[i].neural.shape[0] - w + 1
            wav = synthesis.features_to_waveform(inference.align_to_predictions(raw[i], n), cfg)
            if self.save:
                synthesis.write_wav(dirs.audio / "target" / f"tr{i:03d}.wav", wav, cfg)
            return i, wav

        t0 = time.time()
        self.target = dict(pool.map(target, test))
        log(f"target audio: {len(self.target)} trials resynthesized in {time.time() - t0:.0f} s")

    def asr(self, wav, text) -> dict:
        from wvx import asr

        with self.lock:
            out = asr.evaluate_wav(wav, self.sr, text, self.asr_cfg)
        return {f"asr_{k}": (v.item() if hasattr(v, "item") else v) for k, v in out.items()}

    def base_row(self, model, i) -> dict:
        t = self.sess.trials[i]
        return {"model": model, "trial_index": i, "type": t.meta["type"], "text": t.text,
                "n_bins": int(t.neural.shape[0])}

    def asr_target_audio(self) -> list[dict]:
        rows, t0 = [], time.time()
        for i in self.test:  # serial, in trial order: reproducible temperature fallback
            row = self.base_row("target_audio", i)
            try:
                row.update(self.asr(self.target[i], self.sess.trials[i].text))
            except Exception as ex:
                row["asr_error"] = f"{type(ex).__name__}: {ex}"
            rows.append(row)
        log(f"target audio: ASR on {len(rows)} clips in {time.time() - t0:.0f} s")
        return rows

    def score(self, model: str) -> list[dict]:
        import numpy as np

        from wvx import inference, kunz_experiment as kx, metrics, synthesis

        cfg, sess = self.cfg, self.sess
        with np.load(self.dirs.predictions(model), allow_pickle=False) as z:
            preds = {k: z[k] for k in z.files if k != "header"}

        def safe(row, fn, prefix=""):
            try:
                row.update({f"{prefix}{k}": v for k, v in fn().items()})
            except Exception as ex:  # record and go on (e.g. silent output -> metric undefined)
                row[f"{prefix}error"] = f"{type(ex).__name__}: {ex}"

        def one(i):
            pred = preds[f"tr{i:03d}"]
            row = self.base_row(model, i)
            row["n_frames"] = int(pred.shape[0])
            safe(row, lambda: kx.feature_scores(pred, inference.align_to_predictions(sess.trials[i].target, len(pred))))
            wav = synthesis.decoder_output_to_waveform(pred, cfg)
            if self.save:
                synthesis.write_wav(self.dirs.audio / model / f"tr{i:03d}.wav", wav, cfg)
            safe(row, lambda: metrics.evaluate(self.target[i], wav, self.sr, cfg.get("metrics") or {}))
            return row, wav

        t0 = time.time()
        done = list(self.pool.map(one, self.test))
        log(f"{model}: features, audio and mel metrics for {len(done)} trials in {time.time() - t0:.0f} s")
        if self.asr_cfg is not None:
            t0 = time.time()
            for k, (row, wav) in enumerate(done):  # serial, in trial order
                safe(row, lambda: self.asr(wav, row["text"]))
                if (k + 1) % int(cfg["sanity"]["evaluation"]["progress_every"]) == 0:
                    log(f"  {model}: ASR {k + 1}/{len(done)} ({time.time() - t0:.0f} s)")
            log(f"{model}: ASR on {len(done)} clips in {time.time() - t0:.0f} s")
        return [r for r, _ in done]


# ------------------------------------------------------------------------------------- summary


def _num(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        out.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan"))
    return out


def pooled_asr(rows) -> dict:
    """Pooled rates over trials (sum of errors / sum of reference units) and the empty-transcript fraction."""
    import math

    out = {}
    for name, err, tot in (("wer", "word_errors", "ref_words"), ("cer", "char_errors", "ref_chars"),
                           ("per", "phone_errors", "ref_phones")):
        e, n = _num(rows, f"asr_{err}"), _num(rows, f"asr_{tot}")
        ok = [(a, b) for a, b in zip(e, n) if math.isfinite(a) and math.isfinite(b)]
        if ok and sum(b for _, b in ok) > 0:
            out[f"{name}_pooled"] = sum(a for a, _ in ok) / sum(b for _, b in ok)
            out[f"{name}_n_trials"] = len(ok)
    empty = [bool(r["asr_hyp_empty"]) for r in rows if isinstance(r.get("asr_hyp_empty"), bool)]
    if empty:
        out["empty_transcript_fraction"] = sum(empty) / len(empty)
    defined = [r for r in rows if isinstance(r.get("asr_word_errors_constrained"), (int, float))
               and r["asr_word_errors_constrained"] == r["asr_word_errors_constrained"]]
    out["n_trials_constrained_defined"] = len(defined)  # reference fully inside the 50-word vocabulary
    return out


def summarize(rows: list[dict], target_rows: list[dict], cfg) -> dict:
    from wvx import kunz_experiment as kx

    types = list(cfg["sanity"]["test_types"])
    groups = [(ty, (lambda r, ty=ty: r["type"] == ty)) for ty in types] + [("pooled", lambda r: True)]
    models = [m for m in dict.fromkeys(r["model"] for r in rows)]
    out = {"per_model": {}, "comparisons": {}, "target_audio_asr": {},
           "direction": f"differences are a - b; lower is better for {list(LOWER_IS_BETTER)}"}
    for m in models:
        res = out["per_model"][m] = {}
        for g, f in groups:
            sel = [r for r in rows if r["model"] == m and f(r)]
            if not sel:
                continue
            res[g] = {"n": len(sel), **{k: kx.describe(_num(sel, k)) for k in METRICS},
                      "asr_pooled": pooled_asr(sel),
                      "n_errors": {k: sum(k in r for r in sel) for k in ("error", "asr_error")}}
    for g, f in groups:
        sel = [r for r in target_rows if f(r)]
        if sel:
            out["target_audio_asr"][g] = {"n": len(sel), "asr_wer": kx.describe(_num(sel, "asr_wer")),
                                          "asr_pooled": pooled_asr(sel)}
    trained = [m for m in models if m != ZERO_SHOT]
    pairs = [(m, ZERO_SHOT) for m in trained if ZERO_SHOT in models]
    for a, b in (("trained_ours", "trained_theirs"), ("trained_tts_oracle", "trained_theirs"),
                 ("trained_tts_auto", "trained_theirs"), ("trained_tts_auto", "trained_tts_oracle")):
        if a in trained and b in trained:
            pairs.append((a, b))
    for a, b in pairs:
        comp = out["comparisons"][f"{a}_minus_{b}"] = {}
        for g, f in groups:
            ra = {r["trial_index"]: r for r in rows if r["model"] == a and f(r)}
            rb = {r["trial_index"]: r for r in rows if r["model"] == b and f(r)}
            common = sorted(set(ra) & set(rb))
            if common:
                comp[g] = {k: kx.paired(_num([ra[i] for i in common], k), _num([rb[i] for i in common], k))
                           for k in METRICS}
    return out


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    tmp = path.with_suffix(".csv.partial")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and v != v else v) for k, v in r.items()})
    os.replace(tmp, path)


def headline(summary: dict) -> list[str]:
    def fmt(d):
        return f"{d.get('mean', float('nan')):.3f}±{d.get('sd', float('nan')):.3f}"

    lines = []
    for m, res in summary["per_model"].items():
        for g, v in res.items():
            lines.append(f"{m:15s} {g:34s} n={v['n']:3d} feat_r {fmt(v['feat_corr_mean'])} mel_corr {fmt(v['mel_corr'])} "
                         f"WER {fmt(v['asr_wer'])} (pooled {v['asr_pooled'].get('wer_pooled', float('nan')):.3f})")
    for c, res in summary["comparisons"].items():
        for g, v in res.items():
            lines.append(f"{c:32s} {g:34s} " + " ".join(
                f"{k} {v[k].get('mean', float('nan')):+.3f} (p={v[k].get('wilcoxon_p', float('nan')):.2g})"
                for k in ("feat_corr_mean", "mel_corr", "asr_wer")))
    return lines


# ---------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=["all", "score"], default="all",
                    help="score = rescore every model from saved predictions (no training or decoding)")
    ap.add_argument("--variants", nargs="+", default=None, help="override sanity.target_variants")
    ap.add_argument("--retrain", action="store_true", help="train even if an exported model exists")
    ap.add_argument("--overwrite-predictions", action="store_true", help="decode even if predictions exist")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: name <name>_smoke, 1 epoch, few trials (not a result)")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="override runtime.device")
    args = ap.parse_args(argv)

    over: dict = {"sanity": {}}
    if args.device:
        over["runtime"] = {"device": args.device}
    if args.variants:
        over["sanity"]["target_variants"] = args.variants
    cfg = load_experiment_config(args.config, over)
    s = cfg["sanity"]
    if args.smoke:
        sm = s["smoke"]
        cfg = deep_merge(cfg, {
            "sanity": {"name": s["name"] + "_smoke"},
            "training": {"epochs": sm["epochs"], "validation": {"n_trials": sm["val_trials"]},
                         "log_every_batches": sm["log_every_batches"]},
        })
        s = cfg["sanity"]
    unknown = set(s["target_variants"]) - {"theirs", "ours", *TTS_VARIANTS}
    if unknown:
        raise SystemExit(f"unknown target variants {sorted(unknown)} (theirs, ours, {', '.join(TTS_VARIANTS)})")
    if set(s["target_variants"]) & set(TTS_VARIANTS) and not (s.get("tts_targets") or {}).get("config"):
        raise SystemExit("tts_oracle / tts_auto need sanity.tts_targets.config")
    if resolve_path(cfg["training"]["init"]["path"]) != resolve_path(s["zero_shot_model"]):
        raise SystemExit("training.init.path must be the zero-shot model (the new input layer copies its layer)")

    env_set = apply_thread_env(cfg)  # before numpy / TensorFlow
    nice = cfg["runtime"].get("nice")
    if nice is not None and os.nice(0) < int(nice):  # absolute niceness (only ever raised); children inherit it
        os.nice(int(nice) - os.nice(0))

    from wvx import b2v

    b2v.use_keras2()
    tf_threads = apply_tf_threads(cfg)

    import numpy as np
    import tensorflow as tf

    from wvx.seeding import set_seed
    from wvx.training import configure_device

    configure_device(cfg["runtime"]["device"])
    set_seed(int(cfg["seed"]), deterministic_ops=bool(cfg["runtime"]["deterministic_ops"]))
    runtime = {
        "device": cfg["runtime"]["device"], "threads": cfg["runtime"].get("threads"), "env": env_set, **tf_threads,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "nice": os.nice(0),
        "devices": [d.name for d in tf.config.list_logical_devices()], "python": platform.python_version(),
        "numpy": np.__version__, "tensorflow": tf.__version__, "platform": platform.platform(),
    }
    log(f"config {args.config} ({s['name']}), stage {args.stage}, variants {s['target_variants']}, "
        f"epochs {cfg['training']['epochs']}; runtime {runtime}")
    t_start = time.time()
    dirs = Dirs(cfg)
    sm = s["smoke"] if args.smoke else {}
    sess, raw, desc, train, test, counts = load_day195(cfg, sm.get("max_train_trials"), sm.get("max_test_per_type"))
    _write_json(dirs.results / "data.json", {
        "session": s["session"], "trial_types": counts, "train_trials": train, "test_trials": test,
        "test_types": {ty: sum(desc[i] == ty for i in test) for ty in s["test_types"]},
        "targets": cfg["wairagkar_data"]["keys"]["target"], "reorder_channels": cfg["experiment"]["reorder_channels"],
    })

    timing, training, ours_info, tts_info = {}, {}, None, {}
    models = [ZERO_SHOT] + [f"trained_{v}" for v in s["target_variants"]]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=int(s["evaluation"]["n_workers"])) as pool:
        scorer = None

        def score(model):
            nonlocal scorer
            path = dirs.results / f"scores_{model}.json"
            if path.is_file() and args.stage != "score":
                log(f"{model}: scores exist, scoring skipped")
                return
            if not dirs.predictions(model).is_file():
                log(f"{model}: no predictions, not scored")
                return
            if scorer is None:
                scorer = Scorer(sess, raw, test, cfg, dirs, pool)
                tpath = dirs.results / "asr_target_audio.json"
                if s["evaluation"]["asr"]["enabled"] and s["evaluation"]["asr"]["target_audio"] and (
                        not tpath.is_file() or args.stage == "score"):
                    _write_json(tpath, scorer.asr_target_audio())
            t0 = time.time()
            _write_json(path, scorer.score(model))
            timing[f"score_{model}_s"] = time.time() - t0

        def write_summary():
            rows = [r for m in models for p in [dirs.results / f"scores_{m}.json"] if p.is_file() for r in _read_rows(p)]
            tpath = dirs.results / "asr_target_audio.json"
            target_rows = _read_rows(tpath) if tpath.is_file() else []
            if not rows:
                return
            write_rows_csv(dirs.results / "per_trial.csv", rows)
            results = summarize(rows, target_rows, cfg)
            _write_json(dirs.results / "summary.json", {
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "config_file": args.config, "smoke": bool(args.smoke), "git_commit": _git_commit(), "seed": cfg["seed"],
                "design": {
                    "session": s["session"], "n_train_pool": len(train), "n_test": len(test), "trial_types": counts,
                    "init_and_zero_shot": _rel(resolve_path(s["zero_shot_model"])),
                    "reorder_channels": bool(cfg["experiment"]["reorder_channels"]),
                    "recipe": {k: cfg["training"][k] for k in ("epochs", "batch_size", "learning_rate", "n_augmentations",
                                                               "validation", "init", "export_weights")},
                    "references": "their targets: feat_* vs the scaled+smoothed decoder target; mel_corr etc. vs the "
                                  "LPCNet resynthesis of their raw stored targets (target_audio_source raw)",
                    "asr": "wvx.asr.evaluate_wav, unconstrained (constrained is NaN when the reference has words "
                           "outside the 50-word vocabulary)",
                },
                "models": {m: {"path": _rel(dirs.model_path(m, cfg)),
                               "sha256": _sha256(dirs.model_path(m, cfg)) if dirs.model_path(m, cfg).is_file() else None}
                           for m in models},
                "training": training, "ours_targets": ours_info, "tts_targets": tts_info, "results": results,
                "timing_s": timing,
                "outputs": {"per_trial_csv": _rel(dirs.results / "per_trial.csv"), "audio": _rel(dirs.audio),
                            "checkpoints": _rel(dirs.ckpt)},
                "runtime": runtime,
            })
            for line in headline(results):
                log(line)

        def decode(model):
            t0 = time.time()
            decode_model(model, sess, test, cfg, dirs, overwrite=args.overwrite_predictions)
            timing[f"decode_{model}_s"] = time.time() - t0

        if args.stage == "all":
            decode(ZERO_SHOT)
        score(ZERO_SHOT)
        write_summary()
        for v in s["target_variants"]:
            model = f"trained_{v}"
            if args.stage == "all":
                if v == "theirs":
                    tgt = {i: sess.trials[i].target for i in train}
                elif v == "ours":
                    t0 = time.time()
                    tgt, ours_info = ours_targets(sess, train, cfg, dirs)
                    timing["ours_targets_s"] = time.time() - t0
                else:
                    t0 = time.time()
                    tgt, tts_info[v] = tts_targets(v, sess, train, cfg, dirs)
                    timing[f"{v}_targets_s"] = time.time() - t0
                t0 = time.time()
                training[v] = train_variant(v, sess, train, tgt, cfg, dirs, retrain=args.retrain)
                timing[f"train_{v}_s"] = time.time() - t0
                decode(model)
            else:
                p = dirs.results / f"train_{v}" / "training_info.json"
                training[v] = json.loads(p.read_text()) if p.is_file() else None
                op = dirs.targets / ("ours_targets.npz" if v == "ours" else f"{v}_targets.npz")
                if v != "theirs" and op.is_file():
                    with np.load(op, allow_pickle=False) as z:
                        info = json.loads(str(z["header"]))["info"]
                    if v == "ours":
                        ours_info = info
                    else:
                        tts_info[v] = info
            score(model)
            write_summary()
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {_rel(dirs.results / 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
