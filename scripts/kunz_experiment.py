#!/usr/bin/env python
"""Core experiment, Kunz T15 2024.03.03: train on ATTEMPTED, decode attempted and IMAGINED (host, CPU).

    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/kunz_experiment.py \\
        --config configs/experiments/kunz_pilot.yaml [--stage all|train|decode|score]

Per fold (held-out attempted block X; ``wvx.kunz_experiment.build_fold``), in this order:

1. **Zero-shot baseline:** the day-193 decoder as released (channel reorder ON, no training) decodes every
   evaluation set; scored at once, so the evaluation pipeline is exercised before the long training.
2. **Training** per variant (``experiment.run_variants``): ONE new input layer initialised from the day-193
   ``.h5``, base frozen or fine-tuned, on the attempted blocks other than X (validation drawn from them,
   seeded), their hyperparameters and early stopping (``wvx.training.train``).
3. **Decoding** of every set by each trained model: real inputs, plus time-shuffled inputs (seeded
   permutation of the bins within each trial) for the models in ``nulls.shuffled_time_models``.
4. **Scoring** per trial: LPCNet audio of the output vs the LPCNet resynthesis of its cue's TTS features
   (``mel_corr``, MCD; target-free, same pipeline for every set), the shuffled-trial null (vs another
   trial's cue in the same set), and for the held-out attempted block also vs the aligned-target audio
   and the frame-wise feature correlation with the targets. ASR when ``wvx.asr`` exists.

Every Kunz trial goes through ``wvx.kunz_experiment.prepare_neural`` (reorder_channels) before training
and before every decode. Thread caps (``runtime.threads``) are applied before numpy/TensorFlow load.

Re-runnable: training is skipped when its exported ``.h5`` exists, decoding when its prediction file
exists, and ``--stage score`` recomputes all metrics from the saved predictions (e.g. once ASR exists).

Outputs (``experiment.name`` N, fold F = fold_AXX):
    results/N/F/fold.json, per_trial.csv, summary.json, scores_<model>.json,
    results/N/F/train_<variant>/ history.csv, split.json, config.yaml, training_info.json
    results/audio/N/F/predictions/<model>__<input>.npz and wavs (gitignored)
    data/checkpoints/N/F/train_<variant>/ best.tf, final.tf, inference_layer00.h5
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; keep third_party/ free of __pycache__

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import REPO_ROOT, deep_merge, resolve_path  # noqa: E402  (yaml only)
from wvx.experiment_config import apply_tf_threads, apply_thread_env, load_experiment_config  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/kunz_pilot.yaml"
INPUTS = ("real", "shuffled_time")
BASE_FIELDS = ["model", "input", "set", "role", "condition", "block", "trial_index", "text", "text_in_training",
               "n_bins", "n_frames"]


def log(msg: str) -> None:
    print(f"[kunz_experiment {datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


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


# ------------------------------------------------------------------------------------ settings


def model_names(cfg) -> list[str]:
    e = cfg["experiment"]
    return (["zero_shot"] if e["zero_shot"] else []) + list(e["run_variants"])


def _expand(names, cfg) -> set[str]:
    """Config model lists: "trained" stands for every trained variant."""
    out = set()
    for n in names or []:
        out |= set(cfg["experiment"]["run_variants"]) if n == "trained" else {n}
    return out


def inputs_for(model: str, cfg) -> list[str]:
    st = _expand(cfg["experiment"]["nulls"]["shuffled_time_models"], cfg)
    return ["real", "shuffled_time"] if model in st else ["real"]


class Dirs:
    def __init__(self, cfg, fold_name: str):
        e = cfg["experiment"]
        self.results = resolve_path(e["results_root"]) / e["name"] / fold_name
        self.audio = resolve_path(e["audio_root"]) / e["name"] / fold_name
        self.ckpt = resolve_path(e["checkpoints_root"]) / e["name"] / fold_name
        self.pred = self.audio / "predictions"
        for d in (self.results, self.audio, self.pred):
            d.mkdir(parents=True, exist_ok=True)

    def h5(self, variant: str) -> Path:
        return self.ckpt / f"train_{variant}" / "inference_layer00.h5"

    def model_path(self, model: str, cfg) -> Path:
        return resolve_path(cfg["experiment"]["zero_shot_model"]) if model == "zero_shot" else self.h5(model)

    def predictions(self, model: str, inp: str) -> Path:
        return self.pred / f"{model}__{inp}.npz"


def pred_key(set_name: str, trial_index: int) -> str:
    return f"{set_name}__tr{int(trial_index):03d}"


# ---------------------------------------------------------------------------------------- data


def load_data(cfg):
    from wvx.session import SessionData

    e, fmt = cfg["experiment"], {k: str(v) for k, v in cfg["paths"].items()}
    out = {}
    for key in ("attempted_file", "imagined_file", "imagined_eval_file"):
        p = resolve_path(e[key].format(**fmt))
        out[key] = SessionData.load(p)
        log(f"loaded {_rel(p)}: {out[key].name}, {len(out[key].trials)} trials")
    return out["attempted_file"], out["imagined_file"], out["imagined_eval_file"]


def make_fold(block: int, data, cfg):
    from wvx import kunz_experiment as kx
    from wvx.metrics import normalize_text

    att, img, ev = data
    fold = kx.build_fold(att, img, ev, block, cfg, normalize=normalize_text)
    n = cfg["experiment"]["max_trials_per_set"]
    if n:
        for s in fold.sets:
            s.trials, s.text_in_training = s.trials[:n], s.text_in_training[:n]
    partners = {}
    if cfg["experiment"]["nulls"]["shuffled_trial"]:
        for s in fold.sets:
            partners[s.name] = kx.shuffled_trial_partner(s.texts, cfg["seed"], s.key, normalize=normalize_text)
    return fold, partners


def fold_record(fold, partners, cfg) -> dict:
    n_train = cfg["experiment"]["max_train_trials"]
    train = fold.train_trials[:n_train] if n_train else fold.train_trials
    rec = dict(fold.info)
    rec.update({
        "reorder_channels": bool(cfg["experiment"]["reorder_channels"]),
        "seed": cfg["seed"],
        # wvx.training's split.json indexes into this list
        "train_trial_order": [{"block": t.block, "trial_index": t.meta["trial_index"]} for t in train],
        "sets": {
            s.name: {
                **fold.info["sets"][s.name], "n": len(s.trials), "key": s.key,
                "trials": [
                    {"trial_index": t.meta["trial_index"], "text": t.text, "n_bins": int(t.neural.shape[0]),
                     "text_in_training": bool(s.text_in_training[i]),
                     "shuffled_trial_partner": (int(s.trials[partners[s.name][i]].meta["trial_index"])
                                                if s.name in partners else None)}
                    for i, t in enumerate(s.trials)
                ],
            }
            for s in fold.sets
        },
    })
    return rec


# ------------------------------------------------------------------------------------ training


def make_progress_callback(every: int):
    import tensorflow as tf

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch, self.t0 = epoch, time.time()

        def on_train_batch_end(self, batch, logs=None):
            if every and (batch + 1) % every == 0:
                n = self.params.get("steps")
                log(f"  epoch {self.epoch + 1} batch {batch + 1}/{n} loss {logs.get('loss', float('nan')):.5f} "
                    f"({time.time() - self.t0:.0f} s)")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            log(f"  epoch {epoch + 1} done in {time.time() - self.t0:.0f} s: loss {logs.get('loss', float('nan')):.5f} "
                f"val_loss {logs.get('val_loss', float('nan')):.5f}")

    return Progress()


def _history_from_csv(path: Path) -> dict:
    hist: dict = {}
    if path.is_file():
        for row in csv.DictReader(path.open()):
            for k, v in row.items():
                hist.setdefault(k, []).append(float(v))
    return hist


def train_variant(fold, variant: str, cfg, dirs: Dirs, retrain: bool) -> dict:
    from wvx import kunz_experiment as kx
    from wvx.training import train

    e, t = cfg["experiment"], cfg["training"]
    run = f"train_{variant}"
    info_path = dirs.results / run / "training_info.json"
    if dirs.h5(variant).is_file() and not retrain:
        log(f"{fold.name}/{variant}: exported model exists, training skipped ({_rel(dirs.h5(variant))})")
        return json.loads(info_path.read_text()) if info_path.is_file() else {"skipped": True}
    vcfg = deep_merge(cfg, {
        "run_name": run,
        "paths": {"results": dirs.results, "checkpoints": dirs.ckpt},  # Paths: config.yaml stays repo-relative
        "training": {"init": {"freeze_base": bool(e["variants"][variant]["freeze_base"])},
                     "overwrite": bool(retrain) or bool(t["overwrite"])},
    })
    sess = kx.training_session(fold, f"t15.{e['session']}_attempted_{fold.name}", 0, vcfg,
                               max_trials=e["max_train_trials"])
    log(f"{fold.name}/{variant}: training on {len(sess.trials)} attempted trials from blocks {fold.train_blocks} "
        f"(freeze_base={vcfg['training']['init']['freeze_base']}, epochs {t['epochs']}, "
        f"patience {t['early_stopping_patience']})")
    t0 = time.time()
    result = train([sess], vcfg, extra_callbacks=[make_progress_callback(int(t.get("log_every_batches") or 0))])
    wall = time.time() - t0
    h = result.history
    monitor = t["monitor"]
    vals = h.get(monitor, [])
    best = int(min(range(len(vals)), key=vals.__getitem__)) + 1 if vals else None
    split = json.loads(result.split_json.read_text())["sessions"][sess.name]
    info = {
        "variant": variant,
        "freeze_base": vcfg["training"]["init"]["freeze_base"],
        "init": _rel(resolve_path(t["init"]["path"])),
        "n_train_trials": len(split["train"]),
        "n_validation_trials": len(split["validation"]),
        "validation_trials": [{"block": sess.trials[j].block, "trial_index": sess.trials[j].meta["trial_index"]}
                              for j in split["validation"]],
        "excluded_too_short": split["excluded_too_short"],
        "epochs_configured": t["epochs"],
        "epochs_run": len(h.get("loss", [])),
        "early_stopped": len(h.get("loss", [])) < t["epochs"],
        "best_epoch_by_" + monitor: best,
        "export_weights": t["export_weights"],
        "history": h,
        "wall_time_s": wall,
        "exported": _rel(result.inference_models[0]),
        "exported_sha256": _sha256(result.inference_models[0]),
    }
    _write_json(info_path, info)
    log(f"{fold.name}/{variant}: trained in {wall / 60:.1f} min, {info['epochs_run']} epochs, "
        f"loss {h.get('loss')} val_loss {h.get('val_loss')}")
    import tensorflow as tf

    del result
    tf.keras.backend.clear_session()
    return info


# ------------------------------------------------------------------------------------ decoding


def decode_model(model: str, fold, cfg, dirs: Dirs, overwrite: bool) -> None:
    import numpy as np
    import tensorflow as tf

    from wvx import inference
    from wvx import kunz_experiment as kx

    path = dirs.model_path(model, cfg)
    todo = [i for i in inputs_for(model, cfg) if overwrite or not dirs.predictions(model, i).is_file()]
    if not todo:
        log(f"{fold.name}/{model}: predictions exist, decoding skipped")
        return
    net = inference.load_model(path, cfg)
    for inp in todo:
        t0, preds, n = time.time(), {}, sum(len(s.trials) for s in fold.sets)
        for s in fold.sets:
            for t in s.trials:
                x = kx.prepare_neural(t.neural, cfg)
                if inp == "shuffled_time":
                    x = kx.shuffle_time(x, cfg["seed"], t.block, t.meta["trial_index"])
                preds[pred_key(s.name, t.meta["trial_index"])] = kx.decode(net, x, cfg)
                if len(preds) % int(cfg["experiment"]["evaluation"]["progress_every"]) == 0:
                    log(f"  {model}/{inp}: decoded {len(preds)}/{n} ({time.time() - t0:.0f} s)")
        header = {"model": model, "model_path": _rel(path), "model_sha256": _sha256(path), "input": inp,
                  "reorder_channels": bool(cfg["experiment"]["reorder_channels"]), "seed": cfg["seed"],
                  "fold": fold.name, "created": datetime.datetime.now().isoformat(timespec="seconds")}
        out = dirs.predictions(model, inp)
        tmp = out.with_name(out.stem + ".partial.npz")
        np.savez_compressed(tmp, header=np.array(json.dumps(header)), **preds)
        os.replace(tmp, out)
        log(f"{fold.name}/{model}/{inp}: {len(preds)} trials decoded in {time.time() - t0:.0f} s -> {_rel(out)}")
    del net
    tf.keras.backend.clear_session()


def load_predictions(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        preds = {k: z[k] for k in z.files if k != "header"}
    return header, preds


# ------------------------------------------------------------------------------------- scoring


class ASR:
    """The configured ASR function (``experiment.evaluation.asr``), called with its OWN config section.

    ``__call__(wav, sr, text)`` transcribes and scores; ``rescore(text, hyp)`` scores an existing transcript
    against another cue (shuffled-trial null). Calls are serialized: one model, and the text front end
    (g2p/NLTK) is not known to be thread-safe.
    """

    _ROLES = {"wav": ("wav", "audio", "waveform", "signal"), "sr": ("sr", "sample_rate", "fs"),
              "text": ("text", "ref_text", "reference", "reference_text"), "cfg": ("cfg", "config")}

    def __init__(self, mod, a):
        import threading

        self.mod, self.a = mod, a
        self.fn = getattr(mod, a["function"])
        self.rescore_fn = getattr(mod, a.get("rescore_function") or "", None)
        self.bind = self._bind(self.fn)
        self.lock = threading.Lock()
        self.status = f"{a['module']}.{a['function']}{inspect.signature(self.fn)} with cfg {a.get('cfg')}"

    @classmethod
    def _bind(cls, fn) -> dict:
        bind = {}
        for pname, p in inspect.signature(fn).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            role = next((r for r, alias in cls._ROLES.items() if pname in alias), None)
            if role is None and p.default is inspect.Parameter.empty:
                raise TypeError(f"{fn.__name__}{inspect.signature(fn)}: cannot map parameter {pname!r}")
            if role:
                bind[pname] = role
        return bind

    def __call__(self, wav, sr, text) -> dict:
        vals = {"wav": wav, "sr": sr, "text": text, "cfg": self.a.get("cfg")}
        with self.lock:
            out = self.fn(**{p: vals[r] for p, r in self.bind.items()})
        return _prefixed("asr_", out)

    def rescore(self, text, hyp) -> dict:
        if self.rescore_fn is None:
            return {}
        with self.lock:
            out = self.rescore_fn(text, hyp, self.a.get("cfg"))
        return _prefixed("asr_", out)


def make_asr(cfg):
    """(ASR or None, status). Only used when the configured module/function exists at scoring time."""
    a = cfg["experiment"]["evaluation"]["asr"]
    if not a.get("enabled", True):
        return None, "disabled in config"
    if importlib.util.find_spec(a["module"]) is None:
        return None, f"{a['module']} not present at scoring time; ASR columns empty (re-run --stage score later)"
    mod = importlib.import_module(a["module"])
    if getattr(mod, a["function"], None) is None:
        return None, f"{a['module']}.{a['function']} not present; ASR columns empty"
    try:
        asr = ASR(mod, a)
    except TypeError as ex:
        return None, f"ASR not used: {ex}"
    versions = getattr(mod, "versions", None)
    if callable(versions):
        asr.status += f"; versions {versions()}"
    load = getattr(mod, "load_asr_config", None)
    if callable(load):  # the effective settings (their file + our override), for the record
        c = load(None, (a.get("cfg") or {}).get("asr"))
        eff = {"model": c.get("model", {}).get("name"), "compute_type": c.get("compute_type"),
               "cpu_threads": c.get("cpu_threads"), "seed": c.get("seed"),
               "temperature": c.get("decode", {}).get("temperature"),
               "beam_size": c.get("decode", {}).get("beam_size"),
               "constrained_method": c.get("constrained", {}).get("method")}
        asr.status += f"; effective settings {eff}"
    return asr, asr.status


def _prefixed(prefix: str, scores: dict) -> dict:
    return {(k if k.startswith(prefix) else f"{prefix}{k}"): (v.item() if hasattr(v, "item") else v)
            for k, v in dict(scores).items()}


class References:
    """TTS reference audio per cue and aligned-target audio per held-out attempted trial (shared by models)."""

    def __init__(self, fold, cfg, dirs: Dirs, pool, asr=None):
        from wvx import inference, metrics, synthesis, targets

        e = cfg["experiment"]["evaluation"]
        voice = e["tts_voice"] or cfg["tts"]["voice"]
        corpus_name = e["tts_corpus"] or cfg["kunz"]["corpus"]
        self.corpus_file = targets.corpus_path(cfg, voice, corpus_name)
        corpus = targets.TTSCorpus.load(self.corpus_file)
        self.voice, self.sr = voice, int(cfg["synthesis"]["sample_rate"])
        save = bool(e["save_audio"]["references"])
        texts = sorted({t.text for s in fold.sets for t in s.trials})
        corpus_index = {targets.text_key(x): k for k, x in enumerate(corpus.texts)}
        missing = [t for t in texts if t not in corpus]
        if missing:
            raise KeyError(f"{len(missing)} cues have no TTS in {_rel(self.corpus_file)}, e.g. {missing[:3]}")

        def tts(text):
            wav = synthesis.features_to_waveform(corpus.lookup(text), cfg)
            if save:
                k = corpus_index[targets.text_key(text)]
                synthesis.write_wav(dirs.audio / "tts" / f"s{k:03d}.wav", wav, cfg)
            return text, wav

        self.tts = dict(pool.map(tts, texts))
        w = int(cfg["model"]["window_len"])
        held = [t for s in fold.sets if s.role == "heldout_attempted" for t in s.trials]

        def target(t):
            n = t.neural.shape[0] - w + 1
            wav = synthesis.decoder_output_to_waveform(inference.align_to_predictions(t.target, n), cfg)
            if save:
                synthesis.write_wav(dirs.audio / "target" / f"tr{t.meta['trial_index']:03d}.wav", wav, cfg)
            vs = metrics.evaluate(self.tts[t.text], wav, self.sr, cfg["experiment"]["evaluation"]["tts_metrics"])
            return t.meta["trial_index"], (wav, vs)

        both = dict(pool.map(target, held))
        self.target = {k: v[0] for k, v in both.items()}
        self.target_vs_tts = {k: v[1] for k, v in both.items()}
        log(f"references: {len(self.tts)} TTS cues ({voice}), {len(self.target)} aligned targets")
        # ASR ceiling: the references themselves (TTS once per cue; aligned targets of the held-out block).
        self.asr_rows: list[dict] = []
        if asr is not None and cfg["experiment"]["evaluation"]["asr"].get("references"):
            def one(item):
                kind, key, text, wav = item
                row = {"reference": kind, "key": key, "text": text}
                try:
                    row.update(asr(wav, self.sr, text))
                except Exception as ex:
                    row["asr_error"] = f"{type(ex).__name__}: {ex}"
                return row

            items = [("tts", text, text, wav) for text, wav in self.tts.items()]
            items += [("target", idx, t.text, self.target[t.meta["trial_index"]])
                      for t in held for idx in [t.meta["trial_index"]]]
            self.asr_rows = list(pool.map(one, items))
            log(f"references: ASR on {len(self.asr_rows)} reference clips")


def score_model(model: str, fold, partners, cfg, dirs: Dirs, refs: References, asr, pool) -> list[dict]:
    from wvx import inference, kunz_experiment as kx, metrics, synthesis

    e = cfg["experiment"]["evaluation"]
    save = model in _expand(e["save_audio"]["predictions"], cfg)
    tts_cfg, tgt_cfg = e["tts_metrics"], e["target_metrics"]
    asr_inputs, hyp_key = set(e["asr"].get("inputs") or []), e["asr"].get("hypothesis_key")
    sr = refs.sr
    tasks = []
    for inp in inputs_for(model, cfg):
        path = dirs.predictions(model, inp)
        if not path.is_file():
            log(f"{fold.name}/{model}/{inp}: no predictions ({_rel(path)}), not scored")
            continue
        header, preds = load_predictions(path)
        for s in fold.sets:
            for i, t in enumerate(s.trials):
                tasks.append((inp, s, i, t, preds[pred_key(s.name, t.meta["trial_index"])]))

    def safe(row, prefix, fn):
        try:
            row.update(_prefixed(prefix, fn()))
        except Exception as ex:  # record and go on (e.g. silent output -> metric undefined)
            row[f"{prefix}error"] = f"{type(ex).__name__}: {ex}"

    def one(task):
        inp, s, i, t, pred = task
        idx = int(t.meta["trial_index"])
        row = {"model": model, "input": inp, "set": s.name, "role": s.role, "condition": s.condition,
               "block": s.block, "trial_index": idx, "text": t.text, "text_in_training": bool(s.text_in_training[i]),
               "n_bins": int(t.neural.shape[0]), "n_frames": int(pred.shape[0])}
        wav = synthesis.decoder_output_to_waveform(pred, cfg)
        if save and inp == "real":
            synthesis.write_wav(dirs.audio / model / s.name / f"tr{idx:03d}.wav", wav, cfg)
        safe(row, "tts_", lambda: metrics.evaluate(refs.tts[t.text], wav, sr, tts_cfg))
        partner = s.trials[partners[s.name][i]] if s.name in partners else None
        if inp == "real" and partner is not None:
            row["shuffled_trial_partner"] = int(partner.meta["trial_index"])
            safe(row, "shuffled_trial_tts_", lambda: metrics.evaluate(refs.tts[partner.text], wav, sr, tts_cfg))
        if s.role == "heldout_attempted":
            safe(row, "", lambda: kx.feature_scores(pred, inference.align_to_predictions(t.target, len(pred))))
            safe(row, "target_", lambda: metrics.evaluate(refs.target[idx], wav, sr, tgt_cfg))
            row["target_vs_tts_mel_corr"] = refs.target_vs_tts[idx].get("mel_corr")
        if asr is not None and inp in asr_inputs:
            safe(row, "", lambda: asr(wav, sr, t.text))
            hyp = row.get(f"asr_{hyp_key}")
            if inp == "real" and s.name in partners and isinstance(hyp, str):
                safe(row, "shuffled_trial_", lambda: asr.rescore(partner.text, hyp))
        return row

    t0, rows = time.time(), []
    every = int(e["progress_every"])
    for row in pool.map(one, tasks):
        rows.append(row)
        if len(rows) % every == 0:
            log(f"  {model}: scored {len(rows)}/{len(tasks)} ({time.time() - t0:.0f} s)")
    log(f"{fold.name}/{model}: {len(rows)} outputs scored in {time.time() - t0:.0f} s")
    return rows


# ------------------------------------------------------------------------------------- summary


def _num(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        out.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan"))
    return out


def _match(a_rows, b_rows, key_a, key_b, by=("trial_index",)):
    """Values of key_a in a_rows and key_b in b_rows, matched on ``by``."""
    idx = {tuple(r[k] for k in by): r for r in b_rows}
    a, b = [], []
    for r in a_rows:
        o = idx.get(tuple(r[k] for k in by))
        if o is not None:
            a.append(r)
            b.append(o)
    return _num(a, key_a), _num(b, key_b)


def _silent(rows) -> dict:
    """Outputs with no speech frame (``tts_est_speech_fraction`` == 0): count and fraction."""
    f = _num(rows, "tts_est_speech_fraction")
    n = sum(v == 0 for v in f)
    return {"n_silent": int(n), "n": len(rows), "fraction": n / len(rows) if rows else float("nan")}


def _silent_as_zero(rows, key) -> list:
    return [0.0 if s == 0 else v for v, s in zip(_num(rows, key), _num(rows, "tts_est_speech_fraction"))]


_ASR_UNITS = (("wer", "word_errors", "ref_words"), ("cer", "char_errors", "ref_chars"),
              ("per", "phone_errors", "ref_phones"))


def pooled_asr(rows: list[dict], prefix: str = "asr_") -> dict:
    """Pooled ASR rates over trials: sum of errors / sum of reference units (unconstrained and constrained)."""
    import math

    out = {}
    for name, err, tot in _ASR_UNITS:
        for suf in ("", "_constrained"):
            e, n = _num(rows, f"{prefix}{err}{suf}"), _num(rows, f"{prefix}{tot}")
            ok = [(a, b) for a, b in zip(e, n) if math.isfinite(a) and math.isfinite(b)]
            if ok and sum(b for _, b in ok) > 0:
                out[f"{name}{suf}_pooled"] = sum(a for a, _ in ok) / sum(b for _, b in ok)
                out[f"{name}{suf}_n_trials"] = len(ok)
    empty = [bool(r[f"{prefix}hyp_empty"]) for r in rows if isinstance(r.get(f"{prefix}hyp_empty"), bool)]
    if empty:
        out["empty_transcript_fraction"] = sum(empty) / len(empty)
    return out


def summarize(rows: list[dict], fold, cfg) -> dict:
    from wvx import kunz_experiment as kx
    from wvx.metrics import normalize_text

    def sel(**kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    models = [m for m in model_names(cfg) if sel(model=m)]
    numeric = sorted({k for r in rows for k, v in r.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("block", "trial_index")})
    out = {"per_model": {}, "comparisons": {}}
    groups = [(s.name, {"set": s.name}) for s in fold.sets]
    groups.append(("leaky_imagined_pooled", {"role": "leaky_imagined"}))
    for m in models:
        res = out["per_model"][m] = {}
        for gname, flt in groups:
            real, shuf = sel(model=m, input="real", **flt), sel(model=m, input="shuffled_time", **flt)
            if not real:
                continue
            g = res[gname] = {
                "role": real[0]["role"], "n": len(real),
                # Outputs the silence detector finds no speech in (LPCNet digital silence): mel_corr undefined.
                "silent_outputs": _silent(real),
                "mel_corr_vs_tts": kx.describe(_num(real, "tts_mel_corr")),
                # Convention, labelled: silent outputs scored 0 (avoids comparing sets on selected subsets).
                "mel_corr_vs_tts_silent_as_zero": kx.describe(_silent_as_zero(real, "tts_mel_corr")),
                "null_shuffled_trial": kx.describe(_num(real, "shuffled_trial_tts_mel_corr")),
                "real_minus_shuffled_trial": kx.paired(_num(real, "tts_mel_corr"),
                                                       _num(real, "shuffled_trial_tts_mel_corr")),
            }
            by = ("set", "trial_index")
            if shuf:
                g["silent_outputs_shuffled_time"] = _silent(shuf)
                g["null_shuffled_time"] = kx.describe(_num(shuf, "tts_mel_corr"))
                g["null_shuffled_time_silent_as_zero"] = kx.describe(_silent_as_zero(shuf, "tts_mel_corr"))
                g["real_minus_shuffled_time"] = kx.paired(*_match(real, shuf, "tts_mel_corr", "tts_mel_corr", by))
            novel = [r for r in real if not r["text_in_training"]]
            if len(novel) != len(real):
                g["mel_corr_vs_tts_novel_cues_only"] = kx.describe(_num(novel, "tts_mel_corr"))
            if g["role"] == "heldout_attempted":
                for k in ("target_mel_corr", "target_mel_corr_framewise", "feat_corr_mean", "target_vs_tts_mel_corr"):
                    g[k] = kx.describe(_num(real, k))
                if shuf:
                    for k in ("target_mel_corr", "target_mel_corr_framewise", "feat_corr_mean"):
                        g[f"null_shuffled_time_{k}"] = kx.describe(_num(shuf, k))
            if any("asr_word_errors" in r for r in real):
                g["asr_real"] = pooled_asr(real)
                g["asr_null_shuffled_trial"] = pooled_asr(real, "shuffled_trial_asr_")
                if any("asr_word_errors" in r for r in shuf):
                    g["asr_null_shuffled_time"] = pooled_asr(shuf)
            g["all_metrics_real"] = {k: kx.describe(_num(real, k)) for k in numeric if any(k in r for r in real)}
            if shuf:
                g["all_metrics_shuffled_time"] = {k: kx.describe(_num(shuf, k)) for k in numeric
                                                  if any(k in r for r in shuf)}
        # Within model: attempted vs imagined on the SAME cues (held-out block X vs its paired block).
        held = sel(model=m, input="real", role="heldout_attempted")
        pair = sel(model=m, input="real", role="paired_imagined")
        if held and pair:
            for r in held + pair:
                r["_cue"] = normalize_text(r["text"])
            a, b = _match(held, pair, "tts_mel_corr", "tts_mel_corr", by=("_cue",))
            res["attempted_minus_paired_imagined_same_cue"] = kx.paired(a, b)
            for r in held + pair:
                r.pop("_cue", None)
    if "zero_shot" in models:
        for m in models:
            if m == "zero_shot":
                continue
            comp = out["comparisons"][f"{m}_minus_zero_shot"] = {}
            for gname, flt in groups:
                a_rows, b_rows = sel(model=m, input="real", **flt), sel(model="zero_shot", input="real", **flt)
                if a_rows and b_rows:
                    comp[gname] = kx.paired(*_match(a_rows, b_rows, "tts_mel_corr", "tts_mel_corr",
                                                    ("set", "trial_index")))
    return out


def _asr_reference_summary(path: Path) -> dict:
    """Pooled ASR rates of the reference audio (TTS per cue; aligned targets), the ceiling for the decoders."""
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text())
    for r in rows:
        for k, v in list(r.items()):
            if v is None:
                r[k] = float("nan")
    return {kind: {"n": sum(r["reference"] == kind for r in rows), **pooled_asr([r for r in rows if r["reference"] == kind])}
            for kind in sorted({r["reference"] for r in rows})}


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    fields = list(BASE_FIELDS)
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
    lines = []
    for m, res in summary["per_model"].items():
        for g, v in res.items():
            if not isinstance(v, dict) or "mel_corr_vs_tts" not in v:
                continue
            a, n1 = v["mel_corr_vs_tts"], v["null_shuffled_trial"]
            n2 = v.get("null_shuffled_time", {})
            fmt = lambda d: (f"{d.get('mean', float('nan')):.3f}±{d.get('sd', float('nan')):.3f} (n={d.get('n', 0)}"  # noqa: E731
                             + (f", {d['n_nan']} undefined" if d.get("n_nan") else "") + ")")
            asr = v.get("asr_real", {})
            asr_s = (f" | WER {asr['wer_pooled']:.2f} (null trial {v['asr_null_shuffled_trial'].get('wer_pooled', float('nan')):.2f})"
                     if "wer_pooled" in asr else "")
            lines.append(f"{m:10s} {g:28s} mel_corr vs TTS {fmt(a)} | shuffled-trial {fmt(n1)}"
                         + (f" | shuffled-time {fmt(n2)}" if n2 else "") + asr_s)
    return lines


# ---------------------------------------------------------------------------------------- main


def run_fold(block, data, cfg, args, ctx) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    fold, partners = make_fold(block, data, cfg)
    dirs = Dirs(cfg, fold.name)
    _write_json(dirs.results / "fold.json", fold_record(fold, partners, cfg))
    log(f"{fold.name}: train blocks {fold.train_blocks}; sets "
        + ", ".join(f"{s.name} ({len(s.trials)}, {sum(s.text_in_training)} cues in training)" for s in fold.sets))
    e = cfg["experiment"]
    stage = args.stage
    timing: dict = {}
    training: dict = {}
    asr, asr_status = (None, "not scored in this stage")
    n_workers = int(e["evaluation"]["n_workers"])
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        refs = None

        def score(model):
            nonlocal refs, asr, asr_status
            if refs is None:
                asr, asr_status = make_asr(cfg)
                log(f"ASR: {asr_status}")
                refs = References(fold, cfg, dirs, pool, asr)
                if refs.asr_rows:
                    _write_json(dirs.results / "asr_references.json", refs.asr_rows)
            t0 = time.time()
            rows = score_model(model, fold, partners, cfg, dirs, refs, asr, pool)
            _write_json(dirs.results / f"scores_{model}.json", rows)
            timing[f"score_{model}_s"] = time.time() - t0

        def decode(model):
            t0 = time.time()
            decode_model(model, fold, cfg, dirs, overwrite=args.overwrite_predictions)
            timing[f"decode_{model}_s"] = time.time() - t0

        def needs_score(model):
            return stage == "score" or (stage == "all" and not (dirs.results / f"scores_{model}.json").is_file())

        if e["zero_shot"] and stage in ("all", "decode"):
            decode("zero_shot")
        if e["zero_shot"] and needs_score("zero_shot"):
            score("zero_shot")
        for v in e["run_variants"]:
            if stage in ("all", "train"):
                t0 = time.time()
                training[v] = train_variant(fold, v, cfg, dirs, retrain=args.retrain)
                timing[f"train_{v}_s"] = time.time() - t0
            elif (dirs.results / f"train_{v}" / "training_info.json").is_file():
                training[v] = json.loads((dirs.results / f"train_{v}" / "training_info.json").read_text())
            if stage in ("all", "decode"):
                if not dirs.h5(v).is_file():
                    raise FileNotFoundError(f"no trained model for {v}: {_rel(dirs.h5(v))}")
                decode(v)
            if needs_score(v):
                score(v)

    rows = []
    for m in model_names(cfg):
        p = dirs.results / f"scores_{m}.json"
        if p.is_file():
            rows.extend(json.loads(p.read_text()))
    if not rows:
        return {"fold": fold.name, "scored": False}
    for r in rows:  # JSON null -> NaN for the statistics
        for k, v in list(r.items()):
            if v is None:
                r[k] = float("nan")
    write_rows_csv(dirs.results / "per_trial.csv", rows)
    summary = {
        "fold": fold.name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config_file": args.config,
        "git_commit": ctx["git"],
        "seed": cfg["seed"],
        "design": fold.info,
        "reorder_channels": bool(e["reorder_channels"]),
        "models": {m: {"path": _rel(dirs.model_path(m, cfg)),
                       "sha256": _sha256(dirs.model_path(m, cfg)) if dirs.model_path(m, cfg).is_file() else None,
                       "inputs": inputs_for(m, cfg)} for m in model_names(cfg)},
        "training": training,
        "tts_reference": {"voice": cfg["tts"]["voice"] if not e["evaluation"]["tts_voice"] else e["evaluation"]["tts_voice"],
                          "corpus": e["evaluation"]["tts_corpus"] or cfg["kunz"]["corpus"],
                          "audio": "LPCNet resynthesis of the cue's TTS features (same pipeline as decoder output)"},
        "target_reference": "held-out attempted only: LPCNet resynthesis of the aligned training target "
                            "(decoder_target, smoothed + scaled, mapped back to LPCNet's range)",
        "asr": asr_status,
        "asr_references": _asr_reference_summary(dirs.results / "asr_references.json"),
        "caveats": [
            "Targets were aligned with the day-193 decoder's own output on these attempted trials (D-020), so "
            "target-based scores (target_*, feat_corr_mean) of the zero-shot model are partly circular; the "
            "target-free mel_corr vs TTS is the primary metric.",
            "mel_corr uses DTW, which raises the floor for unrelated speech (METRICS.md 2.5); compare with the nulls.",
            "leaky_imagined sets: their cues were trained on (attempted); not a clean generalization test.",
        ],
        "results": summarize(rows, fold, cfg),
        "timing_s": timing,
        "outputs": {"per_trial_csv": _rel(dirs.results / "per_trial.csv"), "audio": _rel(dirs.audio),
                    "predictions": _rel(dirs.pred), "checkpoints": _rel(dirs.ckpt)},
        "runtime": ctx["runtime"],
    }
    _write_json(dirs.results / "summary.json", summary)
    for line in headline(summary["results"]):
        log(line)
    log(f"{fold.name}: wrote {_rel(dirs.results / 'per_trial.csv')} and {_rel(dirs.results / 'summary.json')}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=["all", "train", "decode", "score"], default="all")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="override experiment.folds")
    ap.add_argument("--variants", nargs="+", default=None, help="override experiment.run_variants")
    ap.add_argument("--retrain", action="store_true", help="train even if an exported model exists")
    ap.add_argument("--overwrite-predictions", action="store_true", help="decode even if predictions exist")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: name <name>_smoke, 1 epoch, few trials (not a result)")
    args = ap.parse_args(argv)

    over: dict = {"experiment": {}}
    if args.folds:
        over["experiment"]["folds"] = args.folds
    if args.variants:
        over["experiment"]["run_variants"] = args.variants
    cfg = load_experiment_config(args.config, over)
    if args.smoke:
        sm = cfg["experiment"]["smoke"]
        cfg = deep_merge(cfg, {
            "experiment": {"name": cfg["experiment"]["name"] + "_smoke",
                           "max_train_trials": sm["max_train_trials"],
                           "max_trials_per_set": sm["max_trials_per_set"]},
            "training": {"epochs": sm["epochs"], "validation": {"n_trials": sm["val_trials"]},
                         "log_every_batches": sm["log_every_batches"]},
        })
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
    ctx = {
        "git": _git_commit(),
        "runtime": {
            "device": cfg["runtime"]["device"], "threads": cfg["runtime"].get("threads"), "env": env_set,
            **tf_threads, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nice": os.nice(0), "devices": [d.name for d in tf.config.list_logical_devices()],
            "python": platform.python_version(), "numpy": np.__version__, "tensorflow": tf.__version__,
            "platform": platform.platform(),
        },
    }
    log(f"config {args.config} (experiment {cfg['experiment']['name']}), stage {args.stage}, "
        f"folds {cfg['experiment']['folds']}, variants {cfg['experiment']['run_variants']}, "
        f"epochs {cfg['training']['epochs']}; runtime {ctx['runtime']}")
    t0 = time.time()
    data = load_data(cfg)
    for block in cfg["experiment"]["folds"]:
        run_fold(int(block), data, cfg, args, ctx)
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
