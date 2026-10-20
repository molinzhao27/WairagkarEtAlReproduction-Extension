#!/usr/bin/env python
"""Reproduce their offline inference on the held-out session and score the synthesized audio.

For every trial of ``inference.session``: predict LPCNet features with their pretrained ``.h5``
model (``wvx.inference``), synthesize predicted and target audio with LPCNet (``wvx.synthesis``;
target = LPCNet resynthesis of their stored target features over the same frames), and score the
pair with ``wvx.metrics.evaluate`` when that module exists (skipped with a warning otherwise).

Outputs:
    <paths.results>/<run_name>/per_trial_metrics.csv
    <paths.results>/<run_name>/summary.json      (config, versions, hashes, data checks, means)
    <paths.results>/<inference.audio_subdir>/<run_name>/   wavs + predictions.npz (gitignored)

Usage, on the GPU host inside the wvx container (via scripts/remote/gpu_job.sh):
    python -I scripts/run_inference.py --config configs/experiments/reproduce_inference.yaml
    python -I scripts/run_inference.py --config ... --max-trials 5      # smoke run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx import b2v  # noqa: E402  (must set up Keras 2 before TensorFlow is imported)
from wvx.config import REPO_ROOT, load_config  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/reproduce_inference.yaml"


def _rel(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.decode().strip()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return _rel(obj)
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:
        return obj.item()
    return obj


def _metrics_module():
    """``wvx.metrics`` if it exists (written separately); None, with a warning, if it does not."""
    try:
        return importlib.import_module("wvx.metrics")
    except ModuleNotFoundError as e:
        if e.name != "wvx.metrics":
            raise
        warnings.warn("wvx.metrics is not available: audio metrics skipped")
        return None


def _feature_scores(pred, target):
    """MSE and mean per-feature Pearson r between normalized predictions and aligned targets."""
    import numpy as np

    mse = float(np.mean((pred - target) ** 2))
    rs = []
    for k in range(pred.shape[1]):
        a, b = pred[:, k], target[:, k]
        if a.std() > 0 and b.std() > 0:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return {"feat_mse": mse, "feat_corr_mean": float(np.mean(rs)) if rs else float("nan")}


def _aggregate(rows: list[dict]) -> dict:
    import numpy as np

    out = {}
    keys = [k for k in rows[0] if k != "trial"] if rows else []
    for k in keys:
        vals = [r.get(k) for r in rows]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals if v is not None):
            continue
        x = np.array([v for v in vals if v is not None], dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            out[k] = {"mean": float(x.mean()), "sd": float(x.std()), "median": float(np.median(x)), "n": int(x.size)}
    return out


def run(cfg: dict, config_path: str | None = None) -> dict:
    t_start = time.time()
    b2v.use_keras2()
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]), deterministic_ops=bool(cfg["runtime"]["deterministic_ops"]))

    import numpy as np
    import tensorflow as tf

    from wvx import inference, synthesis, wairagkar_data

    icfg = cfg["inference"]
    run_name = cfg["run_name"]
    session = icfg["session"]
    data_dir = cfg["paths"]["wairagkar_neural_data"]
    out_dir = Path(cfg["paths"]["results"]) / run_name
    audio_dir = Path(cfg["paths"]["results"]) / icfg["audio_subdir"] / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    sr = int(cfg["synthesis"]["sample_rate"])
    caught: list[str] = []

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")

        source = icfg["target_audio_source"]
        if source not in ("raw", "decoder_target"):
            raise ValueError(f"inference.target_audio_source {source!r} not in (raw, decoder_target)")
        layer = wairagkar_data.layer_for_session(session, cfg)
        data_files = wairagkar_data.inspect_session_files(data_dir, session, cfg)
        sess = wairagkar_data.load_session(data_dir, session, layer, cfg)
        raw_targets = None
        if any(t.target is not None for t in sess.trials):
            raw_targets = wairagkar_data.load_raw_targets(data_dir, session, cfg)
        data_checks = wairagkar_data.describe_session(sess, cfg, raw_targets)
        model_path = Path(cfg["paths"]["pretrained_model"])
        model = inference.load_model(model_path, cfg)
        metrics = _metrics_module()
        if icfg["synthesize"]:
            synthesis.preflight(cfg)

        trials = list(enumerate(sess.trials))
        if icfg["max_trials"] is not None:
            trials = trials[: int(icfg["max_trials"])]

        rows, predictions = [], {}
        for idx, trial in trials:
            row: dict = {"trial": idx, "text": trial.text, "n_bins": trial.neural.shape[0]}
            t0 = time.time()
            pred = inference.predict_trial(model, trial.neural, cfg)
            row["n_frames"] = pred.shape[0]
            row["t_predict_s"] = time.time() - t0
            predictions[f"tr{idx}"] = pred
            if trial.target is not None:
                row.update(_feature_scores(pred, inference.align_to_predictions(trial.target, len(pred))))
            if icfg["synthesize"]:
                t0 = time.time()
                est = synthesis.decoder_output_to_waveform(pred, cfg)
                row["pred_wav"] = _rel(synthesis.write_wav(audio_dir / f"{session}_tr{idx}_pred.wav", est, cfg))
                ref = None
                if trial.target is not None:
                    if source == "raw":
                        raw = inference.align_to_predictions(raw_targets[idx], len(pred))
                        ref = synthesis.features_to_waveform(raw, cfg)
                    else:
                        norm = inference.align_to_predictions(trial.target, len(pred))
                        ref = synthesis.decoder_output_to_waveform(norm, cfg)
                    row["target_wav"] = _rel(
                        synthesis.write_wav(audio_dir / f"{session}_tr{idx}_target.wav", ref, cfg)
                    )
                row["t_synth_s"] = time.time() - t0
                if metrics is not None and ref is not None:
                    try:
                        scores = metrics.evaluate(ref, est, sr, cfg.get("metrics") or {})
                    except Exception as e:  # wvx.metrics edge cases (e.g. silent audio): record, go on
                        row["metrics_error"] = f"{type(e).__name__}: {e}"
                    else:
                        for k, v in scores.items():
                            if k in row:
                                raise KeyError(f"wvx.metrics returned {k!r}, which clashes with a column")
                            row[k] = v.item() if hasattr(v, "item") and getattr(v, "ndim", 1) == 0 else v
            rows.append(row)
            print(f"[run_inference] trial {idx}: {trial.text!r} bins={row['n_bins']} frames={row['n_frames']}", flush=True)

        caught = list(dict.fromkeys(f"{w.category.__name__}: {w.message}" for w in wlist))

    if icfg["save_predictions"]:
        np.savez_compressed(audio_dir / "predictions.npz", **predictions)

    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    csv_path = out_dir / "per_trial_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})

    lp = synthesis.lpcnet_binary(cfg)
    summary = {
        "run_name": run_name,
        "config_file": config_path,
        "git_commit": _git_commit(),
        "session": session,
        "layer": layer,
        "seed": cfg["seed"],
        "model": {"path": _rel(model_path), "sha256": _sha256(model_path)},
        "reorder_channels": bool(icfg["reorder_channels"]),
        "target_audio_source": icfg["target_audio_source"],
        "n_trials_in_session": len(sess.trials),
        "n_trials_run": len(rows),
        "metrics_available": metrics is not None,
        "n_metrics_errors": sum("metrics_error" in r for r in rows),
        "aggregate": _aggregate(rows),
        "data_files": data_files,
        "data_checks": data_checks,
        "lpcnet": {"path": _rel(lp), "sha256": _sha256(lp) if lp.is_file() else None},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "keras": sys.modules["keras"].__version__,
            "platform": platform.platform(),
            "devices": [d.name for d in tf.config.list_physical_devices()],
        },
        "warnings": caught,
        "outputs": {"csv": _rel(csv_path), "audio_dir": _rel(audio_dir)},
        "wall_time_s": time.time() - t_start,
        "config": _jsonable(cfg),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2, allow_nan=True)
    for msg in caught:
        print(f"[run_inference] warning: {msg}", file=sys.stderr)
    print(f"[run_inference] wrote {_rel(csv_path)} and {_rel(out_dir / 'summary.json')}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="experiment YAML merged onto default.yaml")
    ap.add_argument("--max-trials", type=int, default=None, help="override inference.max_trials")
    args = ap.parse_args(argv)
    overrides = {"inference": {"max_trials": args.max_trials}} if args.max_trials is not None else None
    cfg = load_config(args.config, overrides=overrides)
    run(cfg, config_path=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
