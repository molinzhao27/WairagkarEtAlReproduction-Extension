#!/usr/bin/env python
"""Host: aligned TTS targets for the Kunz T15 2024.03.03 trials (the paper's bootstrap method, D-020).

    python -I scripts/align_targets.py [--config configs/experiments/kunz_targets.yaml] [--max-trials N]

For each condition in ``kunz.conditions`` (attempted): load data/processed/kunz/<name>.npz
(scripts/prepare_kunz.py), run the pretrained decoder ``kunz.decoder`` on every trial, align the TTS of
its cue (data/processed/tts/<voice_slug>/<kunz.corpus>.npz, from scripts/make_tts.py) to the decoder
output (``wvx.alignment``), apply their target transform (``wvx.targets.decoder_target``) and write
the SessionData WITH targets to data/processed/kunz_targets/<name>.npz. ``Trial.meta["alignment"]``
keeps the mapped syllables and provenance. A small summary goes to
results/alignment_validation/kunz_<condition>_summary.json.

CPU only (run with CUDA_VISIBLE_DEVICES= ); LPCNet comes from paths.lpcnet.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx import b2v  # noqa: E402  (Keras 2 before TensorFlow)
from wvx.config import REPO_ROOT  # noqa: E402


def _rel(p) -> str:
    try:
        return str(Path(p).relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/kunz_targets.yaml")
    ap.add_argument("--max-trials", type=int, default=None, help="first N trials only (smoke run; not saved)")
    ap.add_argument("--reorder", choices=["true", "false"], default=None, help="override kunz.reorder_channels")
    ap.add_argument("--no-save", action="store_true", help="summary only (e.g. to compare channel orderings)")
    ap.add_argument("--tag", default=None, help="suffix of the summary file name")
    ap.add_argument("--decoder", default=None, help="override kunz.decoder (sensitivity checks; use with --no-save)")
    args = ap.parse_args(argv)

    b2v.use_keras2()
    import numpy as np

    from wvx import inference, synthesis, targets
    from wvx.seeding import set_seed
    from wvx.session import SessionData

    cfg = targets.load_targets_config(args.config)
    set_seed(int(cfg["seed"]))
    k = cfg["kunz"]
    voice = cfg["tts"]["voice"]
    if not voice:
        raise SystemExit("tts.voice is not set")
    dec = args.decoder or k["decoder"]
    model_path = Path(REPO_ROOT / dec) if not Path(dec).is_absolute() else Path(dec)
    model = inference.load_model(model_path, cfg)
    synthesis.preflight(cfg)
    corpus_file = targets.corpus_path(cfg, voice, k["corpus"])
    corpus = targets.TTSCorpus.load(corpus_file)

    for cond in k["conditions"]:
        t0 = time.time()
        name = f"t15.2024.03.03_{cond}"
        sess = SessionData.load(Path(cfg["paths"]["kunz_processed"]) / f"{name}.npz")
        trials = sess.trials[: args.max_trials] if args.max_trials else sess.trials
        reorder = bool(k["reorder_channels"]) if args.reorder is None else args.reorder == "true"
        preds, results = targets.align_trials(model, trials, corpus, cfg, reorder=reorder)
        n_syl, speech_frac, outside = [], [], []
        for t, res in zip(trials, results):
            t.target = targets.decoder_target(res.target_raw, cfg)
            t.meta["alignment"] = {
                "method": "wvx.alignment (D-020)",
                "voice": voice,
                "decoder": _rel(model_path),
                "n_pred_syllables": int(len(res.pred_segments)),
                "mapped_segments": res.mapped_segments.tolist(),
                "dtw_cost": res.dtw_cost,
            }
            n_syl.append(len(res.mapped_segments))
            speech_frac.append(float((res.mapped_segments[:, 1] - res.mapped_segments[:, 0]).sum() / t.neural.shape[0]))
            outside.append(float(np.mean(np.abs(t.target) > 1)))
        sess.validate(n_features=int(cfg["model"]["n_features"]), n_targets=int(cfg["model"]["n_targets"]))
        out = Path(cfg["paths"]["kunz_targets"]) / f"{name}.npz"
        save = args.max_trials is None and not args.no_save
        if save:
            SessionData(sess.name, sess.layer, trials).save(out)
        tgt = np.concatenate([t.target for t in trials])
        summary = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "session": name,
            "n_trials": len(trials),
            "saved": _rel(out) if save else None,
            "voice": voice,
            "tts_corpus": {"file": _rel(corpus_file), "meta": corpus.meta},
            "decoder": {"path": _rel(model_path), "sha256": _sha256(model_path),
                        "reorder_channels": reorder},
            "decoder_output_sanity": targets.decoder_output_sanity(trials, results, cfg),
            "syllables_per_trial": {"min": int(min(n_syl)), "median": float(np.median(n_syl)), "max": int(max(n_syl))},
            "speech_fraction_per_trial": {"min": float(min(speech_frac)), "median": float(np.median(speech_frac)),
                                          "max": float(max(speech_frac))},
            "pred_syllables_zero": int(sum(int(t.meta["alignment"]["n_pred_syllables"] == 0) for t in trials)),
            "target_fraction_outside_unit_range": float(np.mean(outside)),
            "target_mean": tgt.mean(0).tolist(),
            "target_sd": tgt.std(0).tolist(),
            "alignment_config": cfg["alignment"],
            "wall_time_s": time.time() - t0,
        }
        sp = Path(cfg["paths"]["alignment_validation"]) / (f"kunz_{cond}_summary" + (f"_{args.tag}" if args.tag else "") + ".json")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2))
        print(f"[align_targets] {name}: {len(trials)} trials, syllables/trial median {np.median(n_syl):.0f}, "
              f"speech fraction median {np.median(speech_frac):.2f} -> {summary['saved']}, {_rel(sp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
