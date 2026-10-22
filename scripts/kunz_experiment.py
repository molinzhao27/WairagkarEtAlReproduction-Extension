#!/usr/bin/env python
"""Core experiment, Kunz T15 2024.03.03: train on ATTEMPTED, decode attempted and IMAGINED (host, CPU).

    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/kunz_experiment.py \\
        --config configs/experiments/kunz_pilot.yaml [--stage all|train|decode|score]
    # the user's spare PC (own GPU, no gate; scripts/remote/pc_launch.sh): add --device gpu

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
   Every real output is also scored against EVERY distinct cue of its set (the cue nulls: mean over all
   other cues, mean over the ``nulls.length_matched_k`` cues closest in TTS speech duration, and the rank of
   the own cue; ``wvx.kunz_experiment.cue_null_stats``), plus its speech content (speech frames, speech
   seconds within the first ``evaluation.early_window_s``, peak dBFS) from the metrics' silence detector.
   LPCNet and the metrics run in ``evaluation.n_workers`` worker processes, the ASR in ``n_workers // its
   cpu_threads`` more, each with its own model (D-046: in threads, the GIL and one serialized ASR model kept a
   run at ~4 cores). Same scores either way: per-clip ASR seeds, and no metric depends on the process.

Every Kunz trial goes through ``wvx.kunz_experiment.prepare_neural`` (reorder_channels) before training
and before every decode. Thread caps (``runtime.threads``) are applied before numpy/TensorFlow load.

**Re-runnable, never silently stale:**
- Training is skipped only when ``training_info.json`` records the SAME training fingerprint (training-
  relevant config + SHA-256 of the input files; ``wvx.kunz_experiment.training_fingerprint``) and the
  ``.h5`` is the file it exported. Otherwise it stops with an error: pass ``--retrain`` (expensive) or use a
  new ``experiment.name``. ``--stage decode/score`` re-check it (older runs without a fingerprint: warning).
- Decoding is skipped only when the prediction file's header names the current model file (sha256), the
  same decode settings, and every needed trial; otherwise it is re-decoded, with a loud log line.
- Scoring (``--stage all``) is redone when ``scores_<model>.json`` is missing, older than its predictions,
  or records other predictions / model / scoring settings (hashes stored per row). ``--stage score``
  always rescores. ``--retrain`` re-decodes and rescores that variant, ``--overwrite-predictions`` every
  model. If any scores are stale at the end (e.g. after ``--stage decode``), no summary is written.

Outputs (``experiment.name`` N, fold F = fold_AXX):
    results/N/F/fold.json, per_trial.csv, summary.json, scores_<model>.json, score_matrix_<model>.csv
    results/N/F/train_<variant>/ history.csv, split.json, config.yaml, training_info.json
    results/audio/N/F/predictions/<model>__<input>.npz and wavs (gitignored)
    data/checkpoints/N/F/train_<variant>/ best.tf, final.tf, inference_layer00.h5
Every JSON output records ``code_version`` (``wvx.provenance``) and ``config_sha256`` (of the resolved config
with repo-relative paths).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; keep third_party/ free of __pycache__

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import REPO_ROOT, deep_merge, resolve_path  # noqa: E402  (yaml only)
from wvx.experiment_config import apply_tf_threads, apply_thread_env, load_experiment_config  # noqa: E402
from wvx.provenance import code_version, config_sha256  # noqa: E402  (stdlib only)

DEFAULT_CONFIG = "configs/experiments/kunz_pilot.yaml"
INPUTS = ("real", "shuffled_time")
BASE_FIELDS = ["model", "input", "set", "role", "condition", "block", "trial_index", "text", "text_in_training",
               "n_bins", "n_frames"]
# Per-row provenance in scores_<model>.json (left out of per_trial.csv, which holds the data columns).
PROVENANCE_FIELDS = ("predictions_sha256", "model_sha256", "score_settings_sha256", "config_sha256", "code_version")
# experiment.evaluation keys that change how scoring runs or what is saved, never a score.
EVALUATION_KEYS_NOT_AFFECTING_SCORES = ("n_workers", "progress_every", "save_audio")


class StaleOutputError(RuntimeError):
    """An existing output was made under other settings than this run's; it is never reused silently."""


def log(msg: str) -> None:
    print(f"[kunz_experiment {datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


_SHA_CACHE: dict = {}


def _sha256(path) -> str:
    """SHA-256 of a file, cached per (path, size, mtime) so large inputs are hashed once per process."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        with open(p, "rb") as f:
            _SHA_CACHE[key] = hashlib.file_digest(f, "sha256").hexdigest()
    return _SHA_CACHE[key]


@functools.lru_cache(maxsize=1)
def _code_version() -> dict:
    return code_version()


def version_label(cv) -> str | None:
    """Compact code version for per-row records: <commit>[+dirty:<first 12 of the diff hash>]."""
    if not cv or not cv.get("commit"):
        return None
    out = str(cv["commit"])
    if cv.get("dirty"):
        out += "+dirty" + (f":{str(cv['diff_sha256'])[:12]}" if cv.get("diff_sha256") else "")
    return out


def run_config_sha256(cfg) -> str:
    """config_sha256 of the resolved config with repo-relative paths (same on the Mac and the host)."""
    from wvx import kunz_experiment as kx

    return config_sha256(kx.portable(cfg))


def run_provenance(cfg, ctx=None) -> dict:
    cv = (ctx or {}).get("code_version") or _code_version()
    return {"code_version": cv, "config_sha256": run_config_sha256(cfg)}


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


def _clear_tf_session() -> None:
    import tensorflow as tf

    tf.keras.backend.clear_session()


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
    st = _expand((cfg["experiment"].get("nulls") or {}).get("shuffled_time_models"), cfg)
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

    def scores(self, model: str) -> Path:
        return self.results / f"scores_{model}.json"

    def score_matrix(self, model: str) -> Path:
        return self.results / f"score_matrix_{model}.csv"

    def training_info(self, variant: str) -> Path:
        return self.results / f"train_{variant}" / "training_info.json"


def pred_key(set_name: str, trial_index: int) -> str:
    """Name of one trial's prediction in a predictions .npz (set name + Kunz trial index)."""
    return f"{set_name}__tr{int(trial_index):03d}"


def expected_pred_keys(fold) -> list[str]:
    return [pred_key(s.name, t.meta["trial_index"]) for s in fold.sets for t in s.trials]


# ---------------------------------------------------------------------------------------- data


def data_path(cfg, key: str) -> Path:
    """An experiment input file (``experiment.<key>``, formatted with ``paths``)."""
    fmt = {k: str(v) for k, v in cfg["paths"].items()}
    return resolve_path(cfg["experiment"][key].format(**fmt))


def load_data(cfg):
    from wvx.session import SessionData

    out = {}
    for key in ("attempted_file", "imagined_file", "imagined_eval_file"):
        p = data_path(cfg, key)
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


def train_trials(fold, cfg) -> list:
    n_train = cfg["experiment"]["max_train_trials"]
    return fold.train_trials[:n_train] if n_train else fold.train_trials


def fold_record(fold, partners, cfg) -> dict:
    rec = dict(fold.info)
    rec.update({
        "reorder_channels": bool(cfg["experiment"]["reorder_channels"]),
        "seed": cfg["seed"],
        # wvx.training's split.json indexes into this list
        "train_trial_order": [{"block": t.block, "trial_index": t.meta["trial_index"]} for t in train_trials(fold, cfg)],
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


def training_input_files(cfg) -> dict:
    """The files a trained variant depends on: training data (inputs + targets), electrode map, init model."""
    files = {"attempted_file": data_path(cfg, "attempted_file")}
    if cfg["experiment"]["reorder_channels"]:
        files["electrode_mapping"] = Path(cfg["paths"]["electrode_mapping"])
    init = cfg["training"].get("init") or {}
    if init.get("source"):
        files["init"] = resolve_path(init["path"])
    return files


def current_training(fold, variant: str, cfg) -> tuple[dict, str]:
    """(fingerprint, its sha256) of training ``variant`` on this fold under the current config."""
    from wvx import kunz_experiment as kx

    shas = {k: (_sha256(p) if p.is_file() else None) for k, p in training_input_files(cfg).items()}
    fp = kx.training_fingerprint(cfg, variant, train_trials(fold, cfg), shas)
    return fp, config_sha256(fp)


def check_trained_model(fold, variant: str, cfg, dirs: Dirs, strict: bool) -> dict:
    """``training_info.json`` of an existing exported model, verified against the current config.

    Raises StaleOutputError if it records another training fingerprint or the ``.h5`` is not the file it
    exported. If nothing can be verified (no training_info.json, or one written before fingerprints were
    recorded): raises when ``strict`` (deciding whether to skip training), else warns loudly and marks the
    returned info ``training_config_verified: false``.
    """
    from wvx import kunz_experiment as kx

    info_path, h5 = dirs.training_info(variant), dirs.h5(variant)
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    fp, sha = current_training(fold, variant, cfg)
    what = f"{fold.name}/{variant}: the existing model {_rel(h5)}"
    fix = ("Pass --retrain to train it again under the current config (expensive), or set a new "
           "experiment.name to keep both.")
    problems = []
    stored = info.get("training_config_sha256")
    if stored is not None and stored != sha:
        diff = kx.fingerprint_differences(info.get("training_fingerprint") or {}, fp)
        problems.append(f"was trained under other settings (differs in: {', '.join(diff) or 'fingerprint'})")
    if info.get("exported_sha256") and h5.is_file() and info["exported_sha256"] != _sha256(h5):
        problems.append("is not the file its training exported (sha256 differs from training_info.json)")
    if problems:
        raise StaleOutputError(f"{what} {'; '.join(problems)}. {fix}")
    if stored is None:
        why = ("has no training_info.json" if not info else
               "has a training_info.json without a training fingerprint (written before they were recorded)")
        if strict:
            raise StaleOutputError(f"{what} {why}, so it cannot be checked against the current config and is not "
                                   f"reused for training. {fix} (--stage decode / --stage score reuse it, flagged "
                                   f"as unverified.)")
        log(f"!!! WARNING {what} {why}; reused UNVERIFIED (training_config_verified: false)")
        return {**info, "training_config_verified": False}
    return {**info, "training_config_verified": True}


def train_variant(fold, variant: str, cfg, dirs: Dirs, retrain: bool) -> dict:
    from wvx import kunz_experiment as kx
    from wvx.training import train

    e, t = cfg["experiment"], cfg["training"]
    run = f"train_{variant}"
    info_path = dirs.training_info(variant)
    if dirs.h5(variant).is_file() and not retrain:
        info = check_trained_model(fold, variant, cfg, dirs, strict=True)
        log(f"{fold.name}/{variant}: exported model matches the current training config "
            f"({info['training_config_sha256'][:12]}), training skipped ({_rel(dirs.h5(variant))})")
        return info
    fp, fp_sha = current_training(fold, variant, cfg)
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
        f"patience {t['early_stopping_patience']}; fingerprint {fp_sha[:12]})")
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
        # resume check (check_trained_model): what this model depends on
        "training_config_sha256": fp_sha,
        "training_fingerprint": fp,
        "input_files": {k: _rel(p) for k, p in training_input_files(cfg).items()},
        **run_provenance(cfg),
    }
    _write_json(info_path, info)
    log(f"{fold.name}/{variant}: trained in {wall / 60:.1f} min, {info['epochs_run']} epochs, "
        f"loss {h.get('loss')} val_loss {h.get('val_loss')}")
    del result
    _clear_tf_session()
    return {**info, "training_config_verified": True}


# ------------------------------------------------------------------------------------ decoding


def decode_settings(cfg, inp: str) -> dict:
    """What a prediction file depends on besides the model file (stored in its header, checked on reuse)."""
    reorder = bool(cfg["experiment"]["reorder_channels"])
    out = {"input": inp, "reorder_channels": reorder,
           "window_len": int(cfg["model"]["window_len"]), "step_size": int(cfg["model"]["step_size"])}
    if reorder:
        out["electrode_mapping_sha256"] = _sha256(cfg["paths"]["electrode_mapping"])
    if inp == "shuffled_time":
        out["seed"] = int(cfg["seed"])  # the time permutation
    return out


def read_prediction_header(path: Path) -> tuple[dict, list[str]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["header"])), [k for k in z.files if k != "header"]


def prediction_staleness(model: str, inp: str, fold, cfg, dirs: Dirs, model_sha: str) -> list[str]:
    from wvx import kunz_experiment as kx

    try:
        header, keys = read_prediction_header(dirs.predictions(model, inp))
    except Exception as ex:  # unreadable or not a prediction file: never reuse
        return [f"unreadable ({type(ex).__name__}: {ex})"]
    return kx.prediction_staleness(header, keys, model_sha, decode_settings(cfg, inp), expected_pred_keys(fold))


def save_predictions(path: Path, header: dict, preds: dict) -> None:
    import numpy as np

    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, header=np.array(json.dumps(header)), **preds)
    os.replace(tmp, path)


def decode_model(model: str, fold, cfg, dirs: Dirs, overwrite: bool) -> list[str]:
    """Decode every set with ``model`` for its inputs; returns the inputs that were (re-)decoded.

    An existing prediction file is reused only if it was made by the current model file with the current
    decode settings and holds every trial of the fold (``prediction_staleness``); otherwise it is re-decoded
    with a loud log line. ``overwrite`` re-decodes regardless.
    """
    from wvx import inference
    from wvx import kunz_experiment as kx

    path = dirs.model_path(model, cfg)
    if not path.is_file():
        raise FileNotFoundError(f"{model}: model file not found: {_rel(path)}")
    model_sha = _sha256(path)
    todo = []
    for inp in inputs_for(model, cfg):
        p = dirs.predictions(model, inp)
        if overwrite or not p.is_file():
            if overwrite and p.is_file():
                log(f"{fold.name}/{model}/{inp}: re-decoding (overwrite requested)")
            todo.append(inp)
            continue
        reasons = prediction_staleness(model, inp, fold, cfg, dirs, model_sha)
        if reasons:
            log(f"!!! {fold.name}/{model}/{inp}: STALE predictions {_rel(p)}: {'; '.join(reasons)}. Re-decoding.")
            todo.append(inp)
        else:
            log(f"{fold.name}/{model}/{inp}: predictions match model {model_sha[:12]} and decode settings, "
                f"decoding skipped")
    if not todo:
        return []
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
        header = {"model": model, "model_path": _rel(path), "model_sha256": model_sha, "input": inp,
                  "reorder_channels": bool(cfg["experiment"]["reorder_channels"]), "seed": cfg["seed"],
                  "fold": fold.name, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                  "decode_settings": decode_settings(cfg, inp), "code_version": version_label(_code_version()),
                  "runtime_device": (cfg.get("runtime") or {}).get("device")}   # provenance only, not a reuse criterion
        out = dirs.predictions(model, inp)
        save_predictions(out, header, preds)
        log(f"{fold.name}/{model}/{inp}: {len(preds)} trials decoded in {time.time() - t0:.0f} s -> {_rel(out)}")
    del net
    _clear_tf_session()
    return todo


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
    (g2p/NLTK) is not known to be thread-safe. Within ``worker_processes(n)`` the calls run in n spawned worker
    processes instead, one call at a time in each, with a model of its own: wvx.asr seeds every clip from the clip
    and the config alone, so a transcript does not depend on the process, the order or earlier clips.
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
        self.cpu_threads = None   # threads per call, if the module reports them (make_asr)
        self.pool = None          # worker processes (worker_processes)

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

    def _run(self, function: str, *args, **kwargs) -> dict:
        from wvx import kunz_experiment as kx

        if self.pool is not None:
            return self.pool.submit(kx.call, self.a["module"], function, *args, **kwargs).result()
        with self.lock:
            return getattr(self.mod, function)(*args, **kwargs)

    def __call__(self, wav, sr, text) -> dict:
        vals = {"wav": wav, "sr": sr, "text": text, "cfg": self.a.get("cfg")}
        out = self._run(self.a["function"], **{p: vals[r] for p, r in self.bind.items()})
        return _prefixed("asr_", out)

    def rescore(self, text, hyp) -> dict:
        if self.rescore_fn is None:
            return {}
        return _prefixed("asr_", self._run(self.a["rescore_function"], text, hyp, self.a.get("cfg")))

    @contextlib.contextmanager
    def worker_processes(self, n: int):
        """Run the calls in ``n`` worker processes (spawned: this process runs TensorFlow threads) while inside."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(n), mp_context=multiprocessing.get_context("spawn")) as pool:
            self.pool = pool
            try:
                yield self
            finally:
                self.pool = None


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
        asr.cpu_threads = c.get("cpu_threads")
    return asr, asr.status


def _prefixed(prefix: str, scores: dict) -> dict:
    from wvx.kunz_experiment import prefixed

    return prefixed(prefix, scores)


def metric_cfg(override) -> dict:
    """The resolved metrics section for an ``experiment.evaluation.*_metrics`` override."""
    from wvx import metrics

    o = dict(override or {})
    return metrics.load_metrics_config(overrides=o.get("metrics", o) or None)


def metric_hop_s(override) -> float:
    return float(metric_cfg(override)["stft"]["hop_ms"]) / 1000.0


def score_settings(cfg) -> dict:
    """What the per-trial scores depend on besides the predictions and the model (hashed into every row)."""
    from wvx import kunz_experiment as kx

    e = cfg["experiment"]
    ev = e["evaluation"]
    return kx.portable({
        "seed": cfg["seed"],
        "nulls": {k: v for k, v in e["nulls"].items() if k != "shuffled_time_models"},
        "max_trials_per_set": e["max_trials_per_set"],
        "evaluation": {k: v for k, v in ev.items() if k not in EVALUATION_KEYS_NOT_AFFECTING_SCORES},
        "tts_voice": ev["tts_voice"] or cfg["tts"]["voice"],
        "tts_corpus": ev["tts_corpus"] or cfg["kunz"]["corpus"],
        "tts_metrics_resolved": metric_cfg(ev["tts_metrics"]),
        "target_metrics_resolved": metric_cfg(ev["target_metrics"]),
        "synthesis": {k: cfg["synthesis"][k] for k in ("sample_rate", "samples_per_frame", "n_frame_floats")},
    })


def score_staleness(model: str, fold, cfg, dirs: Dirs, rows=None) -> list[str]:
    """Why ``scores_<model>.json`` can NOT be reused (empty list: it can).

    Missing; older than one of its prediction files; computed from other predictions, another model file or
    other scoring settings (hashes stored per row; rows without them are only checked by time); or covering
    other trials than the current fold. Prediction or model files that are absent (e.g. on the Mac, which
    holds no predictions) cannot be compared and are not held against the scores.
    """
    p = dirs.scores(model)
    if not p.is_file():
        return ["no scores file"]
    if rows is None:
        rows = json.loads(p.read_text())
    reasons = []
    by_input: dict = {}
    for r in rows:
        by_input.setdefault(r.get("input"), []).append(r)
    for inp in inputs_for(model, cfg):
        f = dirs.predictions(model, inp)
        if not f.is_file():
            continue
        if f.stat().st_mtime > p.stat().st_mtime:
            reasons.append(f"the {inp} predictions are newer than the scores")
        rs = by_input.get(inp, [])
        if not rs:
            reasons.append(f"no scores for the {inp} predictions")
            continue
        rec = {r.get("predictions_sha256") for r in rs} - {None}
        if rec and rec != {_sha256(f)}:
            reasons.append(f"the {inp} scores were computed from other predictions")
    model_path = dirs.model_path(model, cfg)
    rec_m = {r.get("model_sha256") for r in rows} - {None}
    if rec_m and model_path.is_file() and rec_m != {_sha256(model_path)}:
        reasons.append("the scores were computed with another model file")
    rec_s = {r.get("score_settings_sha256") for r in rows} - {None}
    if rec_s and rec_s != {config_sha256(score_settings(cfg))}:
        reasons.append("the scoring settings changed")
    expected = {(s.name, int(t.meta["trial_index"])) for s in fold.sets for t in s.trials}
    for inp, rs in by_input.items():
        got = {(r.get("set"), r.get("trial_index")) for r in rs}
        if got != expected:
            reasons.append(f"the {inp} scores cover {len(got)} trials, the fold has {len(expected)} "
                           f"(other sets or trial limits)")
    return reasons


class References:
    """TTS reference audio per cue and aligned-target audio per held-out attempted trial (shared by models).

    ``tts_speech_s[text]``: speech seconds of the cue's TTS reference by the metrics' silence detector (used
    to pick the length-matched null cues).
    """

    def __init__(self, fold, cfg, dirs: Dirs, pool, asr=None):
        import numpy as np

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
        hop_s = metric_hop_s(e["tts_metrics"])
        self.tts_speech_s = {text: float(np.sum(metrics.speech_frame_mask(wav, self.sr, e["tts_metrics"]))) * hop_s
                             for text, wav in self.tts.items()}
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


def write_score_matrix(path: Path, rows: list[dict]) -> None:
    """Per-set cue score matrix of the real outputs: one row per output, ``cue_JJJ`` = mel_corr vs the TTS of
    the set's J-th distinct cue (order of first appearance among the set's trials; ``own_cue`` = the output's
    own column, ``own_cue_tts_speech_s`` its TTS speech seconds). Empty cell = undefined."""
    n = max((len(r["scores"]) for r in rows), default=0)
    fields = ["set", "trial_index", "text", "own_cue", "own_cue_tts_speech_s"] + [f"cue_{j:03d}" for j in range(n)]
    tmp = path.with_suffix(".csv.partial")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in fields[:5]}
            out.update({f"cue_{j:03d}": ("" if v != v else v) for j, v in enumerate(r["scores"])})
            w.writerow(out)
    os.replace(tmp, path)


def audio_scorer(refs: References, cfg):
    """Worker processes (``evaluation.n_workers``) that vocode and score the outputs (``wvx.kunz_experiment.
    score_output``; the config and the references go to each worker once).

    The LPCNet feature conversion and the metrics' DTW (librosa, numba without ``nogil``) hold the GIL, so in this
    process's threads they ran one at a time. Spawned, not forked: this process runs TensorFlow threads. BLAS: 2
    threads per worker, or 1 if this process has 1 (same results as this process's count, see init_output_scoring;
    uncapped, 8 workers x 8 OpenBLAS threads slowed a pass by 40%).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from threadpoolctl import threadpool_info

    from wvx import kunz_experiment as kx

    e = cfg["experiment"]["evaluation"]
    blas = min((i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"), default=1)
    return ProcessPoolExecutor(
        max_workers=int(e["n_workers"]), mp_context=multiprocessing.get_context("spawn"),
        initializer=kx.init_output_scoring,
        initargs=(cfg, refs.tts, refs.target, refs.sr, e["tts_metrics"], e["target_metrics"],
                  metric_hop_s(e["tts_metrics"]), float(e["early_window_s"]), min(2, blas)))


def score_model(model: str, fold, partners, cfg, dirs: Dirs, refs: References, asr, pool) -> list[dict]:
    """Score every output of ``model``: in ``pool``'s threads, each output goes to a worker process for its audio
    and audio scores (``audio_scorer``), then to the ASR (worker processes of its own, see run_fold)."""
    import numpy as np

    from wvx import kunz_experiment as kx, synthesis
    from wvx.metrics import normalize_text

    ex = cfg["experiment"]
    e = ex["evaluation"]
    save = model in _expand(e["save_audio"]["predictions"], cfg)
    asr_inputs, hyp_key = set(e["asr"].get("inputs") or []), e["asr"].get("hypothesis_key")
    k_len = int(ex["nulls"]["length_matched_k"])
    sr = refs.sr
    model_path = dirs.model_path(model, cfg)
    model_sha = _sha256(model_path) if model_path.is_file() else None
    common = {"score_settings_sha256": config_sha256(score_settings(cfg)), "config_sha256": run_config_sha256(cfg),
              "code_version": version_label(_code_version())}
    cues = {s.name: kx.set_cues(s.texts, normalize_text) for s in fold.sets}
    cue_s = {s.name: np.array([refs.tts_speech_s[x] for x in cues[s.name][1]], float) for s in fold.sets}
    tasks = []
    for inp in inputs_for(model, cfg):
        path = dirs.predictions(model, inp)
        if not path.is_file():
            log(f"{fold.name}/{model}/{inp}: no predictions ({_rel(path)}), not scored")
            continue
        header, preds = load_predictions(path)
        # Same check as decode_model: never score predictions of another model file / other decode settings.
        stale = kx.prediction_staleness(header, list(preds), model_sha, decode_settings(cfg, inp),
                                        expected_pred_keys(fold))
        if stale:
            raise StaleOutputError(f"{_rel(path)} does not match the current model and config ({'; '.join(stale)}): "
                                   f"decode again first (--stage decode or all)")
        meta = {"predictions_sha256": _sha256(path), "model_sha256": header.get("model_sha256")}
        for s in fold.sets:
            for i, t in enumerate(s.trials):
                tasks.append((inp, meta, s, i, t, preds[pred_key(s.name, t.meta["trial_index"])]))

    def safe(row, prefix, fn):
        try:
            row.update(_prefixed(prefix, fn()))
        except Exception as ex_:  # record and go on (e.g. silent output -> metric undefined)
            row[f"{prefix}error"] = f"{type(ex_).__name__}: {ex_}"

    def one(task):
        inp, meta, s, i, t, pred = task
        idx = int(t.meta["trial_index"])
        row = {"model": model, "input": inp, "set": s.name, "role": s.role, "condition": s.condition,
               "block": s.block, "trial_index": idx, "text": t.text, "text_in_training": bool(s.text_in_training[i]),
               "n_bins": int(t.neural.shape[0]), "n_frames": int(pred.shape[0])}
        partner = s.trials[partners[s.name][i]] if s.name in partners else None
        wav, scores = audio.submit(kx.score_output, kx.OutputJob(
            pred, t.text,
            partner=(int(partner.meta["trial_index"]), partner.text) if inp == "real" and partner is not None else None,
            cues=cues[s.name][1] if inp == "real" else None,     # cue nulls: the same output vs EVERY cue of its set
            heldout=(idx, t.target) if s.role == "heldout_attempted" else None)).result()
        if save and inp == "real":
            synthesis.write_wav(dirs.audio / model / s.name / f"tr{idx:03d}.wav", wav, cfg)
        row.update(scores)
        if s.role == "heldout_attempted":
            row["target_vs_tts_mel_corr"] = refs.target_vs_tts[idx].get("mel_corr")
        if asr is not None and inp in asr_inputs:
            safe(row, "", lambda: asr(wav, sr, t.text))
            hyp = row.get(f"asr_{hyp_key}")
            if inp == "real" and s.name in partners and isinstance(hyp, str):
                safe(row, "shuffled_trial_", lambda: asr.rescore(partner.text, hyp))
        row.update(meta)
        row.update(common)
        return row

    t0, rows = time.time(), []
    every = int(e["progress_every"])
    with audio_scorer(refs, cfg) as audio:
        for row in pool.map(one, tasks):
            rows.append(row)
            if len(rows) % every == 0:
                log(f"  {model}: scored {len(rows)}/{len(tasks)} ({time.time() - t0:.0f} s)")
    matrix = []
    for row, (inp, _, s, i, t, _) in zip(rows, tasks):
        scores = row.pop("_cue_scores", None)
        if scores is None:
            continue
        own = cues[s.name][2][i]
        row.update(kx.cue_null_stats(scores, own, cue_s[s.name], k_len))
        matrix.append({"set": s.name, "trial_index": row["trial_index"], "text": t.text, "own_cue": own,
                       "own_cue_tts_speech_s": float(cue_s[s.name][own]), "scores": scores})
    if matrix:
        write_score_matrix(dirs.score_matrix(model), matrix)
    log(f"{fold.name}/{model}: {len(rows)} outputs scored ({len(matrix)} against every cue of their set) "
        f"in {time.time() - t0:.0f} s")
    return rows


# ------------------------------------------------------------------------------------- summary


def _num(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        out.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan"))
    return out


def _match(a_rows, b_rows, key_a, key_b, by=("trial_index",)):
    """Values of key_a in a_rows and key_b in b_rows, matched on ``by``. A key repeated on either side is
    ambiguous and raises ValueError (it would otherwise drop or double-count rows)."""
    from wvx import kunz_experiment as kx

    kx.index_unique([tuple(r[k] for k in by) for r in a_rows], f"_match on {by}, first rows")
    idx = kx.index_unique([tuple(r[k] for k in by) for r in b_rows], f"_match on {by}, second rows")
    a, b = [], []
    for r in a_rows:
        j = idx.get(tuple(r[k] for k in by))
        if j is not None:
            a.append(r)
            b.append(b_rows[j])
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


def summary_settings(cfg) -> dict:
    """Settings of the cue-null summaries (``experiment.summary`` and the null/early-window parameters)."""
    e = cfg["experiment"]
    return {"primary_null": e["summary"]["primary_null"], "min_speech_s": float(e["summary"]["min_speech_s"]),
            "length_matched_k": int(e["nulls"]["length_matched_k"]),
            "early_window_s": float(e["evaluation"]["early_window_s"])}


def has_cue_nulls(rows) -> bool:
    """Rows scored with the cue nulls (``n_cues`` is set on every real output, 0 when silent)."""
    return bool(rows) and all("n_cues" in r for r in rows)


def _null_excess(rows, order) -> dict:
    from wvx import kunz_experiment as kx

    own = _num(rows, "tts_mel_corr")
    return {n: {"column": kx.NULL_COLUMNS[n], "null": kx.describe(_num(rows, kx.NULL_COLUMNS[n])),
                "excess": kx.paired(own, _num(rows, kx.NULL_COLUMNS[n]))} for n in order}


def _excess_values(rows, null_name) -> list:
    from wvx import kunz_experiment as kx

    return [a - b for a, b in zip(_num(rows, "tts_mel_corr"), _num(rows, kx.NULL_COLUMNS[null_name]))]


def summarize(rows: list[dict], fold, cfg) -> dict:
    from wvx import kunz_experiment as kx
    from wvx.metrics import normalize_text

    def sel(**kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    st = summary_settings(cfg)
    order = kx.null_order(st["primary_null"])
    min_s = st["min_speech_s"]
    models = [m for m in model_names(cfg) if sel(model=m)]
    numeric = sorted({k for r in rows for k, v in r.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("block", "trial_index")})
    out = {"per_model": {}, "comparisons": {}, "settings": {**st, "null_order": order}}
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
            if has_cue_nulls(real):
                # Same output vs other cues of the set (primary null first) and identification of the own cue.
                g["nulls"] = _null_excess(real, order)
                g["identification"] = kx.identification(_num(real, "own_rank"), _num(real, "n_cues"))
                g["speech_s"] = kx.describe(_num(real, "est_speech_s"))
                g["early_speech_s"] = kx.describe(_num(real, "early_speech_s"))
                g["peak_dbfs"] = kx.describe(_num(real, "peak_dbfs"))
                kept = [r for r, v in zip(real, _num(real, "est_speech_s")) if v >= min_s]
                g["sensitivity_min_speech"] = {
                    "analysis": "sensitivity analysis, not the primary test: only outputs with >= min_speech_s "
                                "of decoded speech (selected on the output itself)",
                    "min_speech_s": min_s, "n_kept": len(kept), "n": len(real),
                    "nulls": {n: v["excess"] for n, v in _null_excess(kept, order).items()},
                    "identification": kx.identification(_num(kept, "own_rank"), _num(kept, "n_cues")),
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
            try:
                a, b = _match(held, pair, "tts_mel_corr", "tts_mel_corr", by=("_cue",))
                res["attempted_minus_paired_imagined_same_cue"] = kx.paired(a, b)
                if has_cue_nulls(held) and has_cue_nulls(pair):
                    avi = res["attempted_vs_paired_imagined"] = {
                        "primary_null": order[0],
                        # unpaired (Mann-Whitney): each output has its own null
                        "excess_difference": {n: kx.unpaired(_excess_values(held, n), _excess_values(pair, n))
                                              for n in order},
                        "early_speech_s_same_cue": kx.paired(*_match(held, pair, "early_speech_s", "early_speech_s",
                                                                     by=("_cue",))),
                    }
                    hk = [r for r, v in zip(held, _num(held, "est_speech_s")) if v >= min_s]
                    pk = [r for r, v in zip(pair, _num(pair, "est_speech_s")) if v >= min_s]
                    avi["sensitivity_min_speech"] = {
                        "analysis": "sensitivity analysis, not the primary test", "min_speech_s": min_s,
                        "n_attempted_kept": len(hk), "n_imagined_kept": len(pk),
                        "excess_difference": {n: kx.unpaired(_excess_values(hk, n), _excess_values(pk, n))
                                              for n in order},
                    }
            finally:
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
        fields.extend(k for k in r if k not in fields and k not in PROVENANCE_FIELDS)
    tmp = path.with_suffix(".csv.partial")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
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
            cue = ""
            if "nulls" in v:
                p, ex = next(iter(v["nulls"].items()))
                ident = v["identification"]
                cue = (f" | excess vs {p} null {ex['excess'].get('mean', float('nan')):+.3f} "
                       f"(p {ex['excess'].get('wilcoxon_p', float('nan')):.1e}) | top-1 "
                       f"{ident.get('top1_rate', float('nan')):.2f} (chance {ident.get('top1_chance', float('nan')):.2f})")
            lines.append(f"{m:10s} {g:28s} mel_corr vs TTS {fmt(a)} | shuffled-trial {fmt(n1)}"
                         + (f" | shuffled-time {fmt(n2)}" if n2 else "") + cue + asr_s)
    return lines


# ---------------------------------------------------------------------------------------- main


def _forced(overwrite: bool, retrained: bool, decoded) -> str:
    why = []
    if retrained:
        why.append("--retrain")
    if overwrite:
        why.append("--overwrite-predictions")
    if decoded:
        why.append(f"re-decoded {list(decoded)}")
    return ", ".join(why)


def run_fold(block, data, cfg, args, ctx) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    fold, partners = make_fold(block, data, cfg)
    dirs = Dirs(cfg, fold.name)
    e = cfg["experiment"]
    stage = args.stage
    # Fail fast: refuse a trained model made under other settings BEFORE writing or computing anything.
    training_stage = stage in ("all", "train")
    verified = {v: check_trained_model(fold, v, cfg, dirs, strict=training_stage)
                for v in e["run_variants"] if dirs.h5(v).is_file() and not (training_stage and args.retrain)}
    prov = run_provenance(cfg, ctx)
    _write_json(dirs.results / "fold.json", {**fold_record(fold, partners, cfg), **prov})
    log(f"{fold.name}: train blocks {fold.train_blocks}; sets "
        + ", ".join(f"{s.name} ({len(s.trials)}, {sum(s.text_in_training)} cues in training)" for s in fold.sets))
    timing: dict = {}
    training: dict = {}
    asr, asr_status = (None, "not scored in this stage")
    n_workers = int(e["evaluation"]["n_workers"])
    with ThreadPoolExecutor(max_workers=n_workers) as pool, contextlib.ExitStack() as workers:
        refs = None

        def score(model):
            nonlocal refs, asr, asr_status
            if refs is None:
                asr, asr_status = make_asr(cfg)
                log(f"ASR: {asr_status}")
                if isinstance(asr, ASR):   # one model per worker process, about as many cores as the audio workers
                    n_asr = max(1, n_workers // int(asr.cpu_threads or n_workers))
                    workers.enter_context(asr.worker_processes(n_asr))
                    log(f"ASR: {n_asr} worker processes; audio and metrics: {n_workers} worker processes")
                refs = References(fold, cfg, dirs, pool, asr)
                if refs.asr_rows:
                    _write_json(dirs.results / "asr_references.json", refs.asr_rows)
            t0 = time.time()
            rows = score_model(model, fold, partners, cfg, dirs, refs, asr, pool)
            _write_json(dirs.scores(model), rows)
            timing[f"score_{model}_s"] = time.time() - t0

        def decode(model, overwrite):
            t0 = time.time()
            done = decode_model(model, fold, cfg, dirs, overwrite=overwrite)
            timing[f"decode_{model}_s"] = time.time() - t0
            return done or []

        def needs_score(model, forced: str) -> bool:
            if stage == "score":
                return True
            if stage != "all":
                return False
            if forced:
                log(f"{fold.name}/{model}: re-scoring ({forced})")
                return True
            reasons = score_staleness(model, fold, cfg, dirs)
            if reasons == ["no scores file"]:
                return True
            if reasons:
                log(f"!!! {fold.name}/{model}: STALE scores ({'; '.join(reasons)}); re-scoring")
                return True
            log(f"{fold.name}/{model}: scores match their predictions, model and settings; scoring skipped")
            return False

        if e["zero_shot"]:
            decoded = decode("zero_shot", args.overwrite_predictions) if stage in ("all", "decode") else []
            if needs_score("zero_shot", _forced(args.overwrite_predictions, False, decoded)):
                score("zero_shot")
        for v in e["run_variants"]:
            retrained = False
            if training_stage:
                t0 = time.time()
                training[v] = train_variant(fold, v, cfg, dirs, retrain=args.retrain)
                timing[f"train_{v}_s"] = time.time() - t0
                retrained = bool(args.retrain)
            elif v in verified:
                training[v] = verified[v]
            elif dirs.training_info(v).is_file():
                training[v] = json.loads(dirs.training_info(v).read_text())
            decoded = []
            if stage in ("all", "decode"):
                if not dirs.h5(v).is_file():
                    raise FileNotFoundError(f"no trained model for {v}: {_rel(dirs.h5(v))}")
                decoded = decode(v, args.overwrite_predictions or retrained)
            if needs_score(v, _forced(args.overwrite_predictions, retrained, decoded)):
                score(v)

    rows, stale = [], {}
    for m in model_names(cfg):
        p = dirs.scores(m)
        if p.is_file():
            mrows = json.loads(p.read_text())
            reasons = [r for r in score_staleness(m, fold, cfg, dirs, mrows) if r != "no scores file"]
            if reasons:
                stale[m] = reasons
            rows.extend(mrows)
    if stale:
        for m, reasons in stale.items():
            log(f"!!! {fold.name}/{m}: scores are STALE ({'; '.join(reasons)}). per_trial.csv and summary.json NOT "
                f"written; run --stage score (or all).")
        return {"fold": fold.name, "scored": False, "stale_scores": stale}
    if not rows:
        return {"fold": fold.name, "scored": False}
    for r in rows:  # JSON null -> NaN for the statistics
        for k, v in list(r.items()):
            if v is None:
                r[k] = float("nan")
    legacy = sorted({r["model"] for r in rows if r.get("input") == "real" and "n_cues" not in r})
    if legacy:
        log(f"!!! WARNING {fold.name}: scores of {legacy} predate the cue nulls (no null_all_mean / "
            f"null_lenmatched_mean / own_rank); their summary has the shuffled-trial null only. Re-run --stage score.")
    write_rows_csv(dirs.results / "per_trial.csv", rows)
    models = {}
    for m in model_names(cfg):
        mp = dirs.model_path(m, cfg)
        models[m] = {"path": _rel(mp), "sha256": _sha256(mp) if mp.is_file() else None, "inputs": inputs_for(m, cfg),
                     "predictions": {}}
        for inp in inputs_for(m, cfg):
            pp = dirs.predictions(m, inp)
            if pp.is_file():
                header, _ = read_prediction_header(pp)
                models[m]["predictions"][inp] = {"path": _rel(pp), "sha256": _sha256(pp),
                                                 "model_sha256": header.get("model_sha256")}
    summary = {
        "fold": fold.name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config_file": args.config,
        "code_version": prov["code_version"],
        "config_sha256": prov["config_sha256"],
        "git_commit": version_label(prov["code_version"]),  # kept for continuity with older summaries
        "seed": cfg["seed"],
        "design": fold.info,
        "reorder_channels": bool(e["reorder_channels"]),
        "models": models,
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
            "The single shuffled-trial partner is a lenient null and partly a sentence-length effect (few speech "
            "frames score higher against shorter references); the length-matched cue null is primary "
            "(experiment.summary.primary_null). Outputs with little speech (< ~40 frames) cannot be identified "
            "even for real attempted speech: see the min_speech_s sensitivity analysis.",
            "leaky_imagined sets: their cues were trained on (attempted); not a clean generalization test.",
        ],
        "results": summarize(rows, fold, cfg),
        "timing_s": timing,
        "outputs": {"per_trial_csv": _rel(dirs.results / "per_trial.csv"), "audio": _rel(dirs.audio),
                    "predictions": _rel(dirs.pred), "checkpoints": _rel(dirs.ckpt),
                    "score_matrices": {m: _rel(dirs.score_matrix(m)) for m in model_names(cfg)
                                       if dirs.score_matrix(m).is_file()}},
        "runtime": ctx.get("runtime"),
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
    ap.add_argument("--retrain", action="store_true",
                    help="train even if an exported model exists (then re-decode and rescore that variant)")
    ap.add_argument("--overwrite-predictions", action="store_true",
                    help="decode even if up-to-date predictions exist (then rescore)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: name <name>_smoke, 1 epoch, few trials (not a result)")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None,
                    help="override runtime.device (where the run goes, not what it computes; recorded per fold)")
    args = ap.parse_args(argv)

    over: dict = {"experiment": {}}
    if args.device:
        over["runtime"] = {"device": args.device}
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
        "code_version": _code_version(),
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
        f"epochs {cfg['training']['epochs']}; code {version_label(ctx['code_version'])}; runtime {ctx['runtime']}")
    t0 = time.time()
    data = load_data(cfg)
    for block in cfg["experiment"]["folds"]:
        run_fold(int(block), data, cfg, args, ctx)
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
