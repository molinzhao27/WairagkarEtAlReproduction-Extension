#!/usr/bin/env python
"""Host, project venv: offline decoder output of one benchmark cell, saved for the syllable segmenters (D-033).

    CUDA_VISIBLE_DEVICES= nice -n 24 .venv/bin/python -I scripts/syllable_decode.py --cell day195_m172
        [--config configs/experiments/syllable_benchmark.yaml] [--threads N] [--max-trials N] [--no-wav] [--overwrite]

A cell (``benchmark.cells``) is a session (Wairagkar ``session`` or Kunz ``kunz_condition``) and a released
decoder. Every trial is decoded as in scripts/validate_alignment.py (``wvx.inference.predict_trial``, channels
reordered for the day025-day195 models), and the prediction is laid on the neural timeline
(``wvx.targets.prediction_timeline``: silence frame for the first window_len - 1 bins). Writes, on the host only:

    <paths.syllables>/decoded/<cell>/predictions.npz   normalized predictions + header (model SHA-256, reorder, bins, cues)
    <paths.syllables>/decoded/<cell>/wav/NNNN.wav      LPCNet audio of each trial's timeline (16 kHz, PCM 16)

The audio is what the nuclei and Sylber segmenters read; the c0 method and the alignment read predictions.npz.
``--max-trials N`` writes to ``<cell>__first<N>`` (smoke runs). CPU only.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx import b2v  # noqa: E402  (Keras 2 before TensorFlow)
from wvx.config import resolve_path  # noqa: E402
from wvx.syllable_runtime import rel, runtime_record, sha256, start  # noqa: E402  (no numpy)


def cell_spec(cfg, cell: str) -> dict:
    cells = cfg["benchmark"]["cells"]
    if cell not in cells:
        raise SystemExit(f"unknown cell {cell!r}; configured: {sorted(cells)}")
    spec = dict(cells[cell])
    if "kunz_condition" in spec and resolve_path(spec["model"]) != resolve_path(cfg["kunz"]["decoder"]):
        raise SystemExit(f"cell {cell}: model {spec['model']} differs from kunz.decoder {cfg['kunz']['decoder']}")
    return spec


def reorder_for(model_path: Path, cfg) -> bool:
    """Their rule (validation.reorder_until_day): reorder for the released day025-day195 models."""
    day = int(re.search(r"day(\d+)", Path(model_path).name).group(1))
    return day <= int(cfg["validation"]["reorder_until_day"])


def corpus_name(spec: dict, cfg) -> str:
    return cfg["kunz"]["corpus"] if "kunz_condition" in spec else spec["session"]


def load_trials(spec: dict, cfg, with_targets: bool = False):
    """(session name, trials) of a cell; Wairagkar targets only if asked (their target key from the config)."""
    from wvx import wairagkar_data
    from wvx.config import deep_merge
    from wvx.session import SessionData

    if "kunz_condition" in spec:
        name = f"t15.2024.03.03_{spec['kunz_condition']}"
        return name, SessionData.load(Path(cfg["paths"]["kunz_processed"]) / f"{name}.npz").trials
    session = spec["session"]
    data_dir = resolve_path(cfg["validation"]["data_dir"])
    try:
        layer = wairagkar_data.layer_for_session(session, cfg)
    except KeyError:
        layer = -1                                   # informational only for inference
    over = {"wairagkar_data": {"targets": "required" if with_targets else "skip"}}
    return session, wairagkar_data.load_session(data_dir, session, layer, deep_merge(cfg, over)).trials


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/syllable_benchmark.yaml")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--threads", type=int, default=None, help="override runtime.threads (TF, BLAS) and n_workers")
    ap.add_argument("--max-trials", type=int, default=None, help="first N trials only, into <cell>__first<N>")
    ap.add_argument("--no-wav", action="store_true", help="predictions only (enough for method 'current')")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing predictions.npz")
    args = ap.parse_args(argv)

    cfg, env_set = start(args.config, args.threads)
    b2v.use_keras2()

    from concurrent.futures import ThreadPoolExecutor

    from wvx import inference, synthesis, syllables as sy, targets as tg
    from wvx.experiment_config import apply_tf_threads
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    tf_threads = apply_tf_threads(cfg)
    spec = cell_spec(cfg, args.cell)
    out_name = args.cell + (f"__first{args.max_trials}" if args.max_trials else "")
    out_dir = sy.decoded_dir(cfg, out_name)
    pred_file = out_dir / "predictions.npz"
    if pred_file.exists() and not args.overwrite:
        raise SystemExit(f"{rel(pred_file)} exists; pass --overwrite to decode again")

    t0 = time.time()
    model_path = resolve_path(spec["model"])
    reorder = reorder_for(model_path, cfg)
    session, trials = load_trials(spec, cfg)
    trials = trials[: args.max_trials] if args.max_trials else trials
    model = inference.load_model(model_path, cfg)
    synthesis.preflight(cfg)
    corpus = tg.TTSCorpus.load(tg.corpus_path(cfg, cfg["tts"]["voice"], corpus_name(spec, cfg)))
    sil20 = tg.silence_frame(corpus, cfg)               # the LPCNet code of digital silence (same toolchain)
    preds = []
    for i, t in enumerate(trials):
        preds.append(inference.predict_trial(model, t.neural, cfg, reorder=reorder))
        if (i + 1) % 25 == 0:
            print(f"[decode] {args.cell}: {i + 1}/{len(trials)} trials", flush=True)
    t_pred = time.time() - t0

    header = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "cell": args.cell, "cell_spec": spec, "session": session, "n_trials": len(trials),
        "model": {"path": rel(model_path), "sha256": sha256(model_path), "reorder_channels": reorder},
        "n_bins": [int(t.neural.shape[0]) for t in trials],
        "texts": [t.text for t in trials],
        "trial_meta": [{"block": t.block, "trial_index": t.meta.get("trial_index")} for t in trials],
        "silence_frame": sil20.tolist(),
        "wav": None if args.no_wav else {"dir": "wav", "sample_rate": int(cfg["synthesis"]["sample_rate"]),
                                          "subtype": cfg["synthesis"]["wav_subtype"]},
        "runtime": {**runtime_record(cfg, env_set), **tf_threads},
        "code_version": code_version(), "config_sha256": config_sha256(cfg),
    }

    if not args.no_wav:
        wav_dir = out_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        def synth(i):
            raw = tg.prediction_timeline(preds[i], trials[i].neural.shape[0], sil20, cfg)
            synthesis.write_wav(wav_dir / sy.trial_wav_name(i), synthesis.features_to_waveform(raw, cfg), cfg)

        with ThreadPoolExecutor(max_workers=int(cfg["benchmark"]["n_workers"])) as ex:
            list(ex.map(synth, range(len(trials))))
    header["wall_time_s"] = {"decode": t_pred, "total": time.time() - t0}
    sy.save_decoded(pred_file, preds, header)
    print(f"[decode] {args.cell}: {len(trials)} trials, model {rel(model_path)} (reorder {reorder}) -> {rel(out_dir)} "
          f"in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
