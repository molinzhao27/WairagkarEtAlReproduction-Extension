#!/usr/bin/env python
"""Host: aligned TTS targets for the Kunz T15 2024.03.03 trials (the paper's bootstrap method, D-020).

    python -I scripts/align_targets.py [--config configs/experiments/kunz_targets.yaml] [--max-trials N]
    # v2 (D-033): forced-aligned TTS syllables + the benchmark's winning decoder-side method, from the saved
    # decode of the kunz_attempted cell (scripts/syllable_decode.py, scripts/syllable_segment.py):
    python -I scripts/align_targets.py --config configs/experiments/syllable_benchmark.yaml --arm A|B --method M \
        --decoded kunz_attempted

For each condition in ``kunz.conditions`` (attempted): load data/processed/kunz/<name>.npz
(scripts/prepare_kunz.py), run the pretrained decoder ``kunz.decoder`` on every trial (or, with ``--decoded
CELL``, take that cell's saved predictions, made with the same model file), align the TTS of its cue
(data/processed/tts/<voice_slug>/<kunz.corpus>.npz, from scripts/make_tts.py) to the decoder output
(``wvx.alignment``), apply their target transform (``wvx.targets.decoder_target``) and write the SessionData
WITH targets to <paths.kunz_targets>/<name>.npz (data/processed/kunz_targets with kunz_targets.yaml,
data/processed/kunz_targets_v2 with syllable_benchmark.yaml). An existing file is never replaced without
``--overwrite``. ``Trial.meta["alignment"]`` keeps the mapped syllables, the syllable method and provenance.
A small summary goes to <paths.alignment_validation>/kunz_<condition>_summary.json.

Syllables (``syllables`` section, overridable here): ``tts_side`` envelope (D-027) | forced (MFA + maximal
onset, scripts/syllable_tts.py), ``method`` current | nuclei | sylber (decoder side; nuclei and sylber need
``--decoded``), ``count_constraint``, and the mapping ``arm``: A = envelope DTW + snap (D-027), B = the decoder
syllables define the spans (D-033, settings ``syllables.segment_mapping`` in syllable_benchmark.yaml). The
defaults of kunz_targets.yaml reproduce D-027 exactly.

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
    ap.add_argument("--method", choices=["current", "nuclei", "sylber"], default=None, help="override syllables.method")
    ap.add_argument("--tts-side", choices=["envelope", "forced"], default=None, help="override syllables.tts_side")
    ap.add_argument("--count-constraint", action="store_true", help="set syllables.count_constraint")
    ap.add_argument("--arm", choices=["A", "B"], default=None, help="override syllables.arm (mapping)")
    ap.add_argument("--decoded", default=None, help="benchmark cell whose saved predictions (and segments) to use")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing targets file")
    args = ap.parse_args(argv)

    b2v.use_keras2()
    import numpy as np

    from wvx import alignment as al, inference, synthesis, syllables as sy, targets
    from wvx.config import deep_merge
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed
    from wvx.session import SessionData

    cfg = sy.load_any_targets_config(args.config)
    set_seed(int(cfg["seed"]))
    s = dict(cfg["syllables"])
    s.update({k: v for k, v in (("method", args.method), ("tts_side", args.tts_side), ("arm", args.arm)) if v is not None})
    s["count_constraint"] = bool(s["count_constraint"] or args.count_constraint)
    cfg = deep_merge(cfg, {"syllables": s})
    sy.settings(cfg)
    if s["method"] != "current" and not args.decoded:
        raise SystemExit(f"method {s['method']!r} reads a segmenter's output: pass --decoded CELL")
    if s["arm"] == "B" and "segment_mapping" not in s:
        raise SystemExit("arm B needs syllables.segment_mapping: use configs/experiments/syllable_benchmark.yaml")
    k = cfg["kunz"]
    voice = cfg["tts"]["voice"]
    if not voice:
        raise SystemExit("tts.voice is not set")
    dec = args.decoder or k["decoder"]
    model_path = Path(REPO_ROOT / dec) if not Path(dec).is_absolute() else Path(dec)
    decoded = None
    if args.decoded:
        decoded = sy.load_decoded(sy.decoded_dir(cfg, args.decoded) / "predictions.npz")
        if decoded[1]["model"]["sha256"] != _sha256(model_path):
            raise SystemExit(f"--decoded {args.decoded}: made with another model than {_rel(model_path)}")
        model = None
    else:
        model = inference.load_model(model_path, cfg)
    synthesis.preflight(cfg)
    corpus_file = targets.corpus_path(cfg, voice, k["corpus"])
    corpus = targets.TTSCorpus.load(corpus_file)
    sil20 = targets.silence_frame(corpus, cfg)
    tts_table = sy.load_tts_syllables(cfg, voice, k["corpus"]) if s["tts_side"] == "forced" else None
    records = sy.segment_records(cfg, args.decoded, s["method"]) if s["method"] != "current" else None
    if records is not None and records[1]["decoded"]["model"]["sha256"] != decoded[1]["model"]["sha256"]:
        raise SystemExit(f"segments of {args.decoded} {s['method']} do not belong to its predictions")

    for cond in k["conditions"]:
        t0 = time.time()
        name = f"t15.2024.03.03_{cond}"
        out = Path(cfg["paths"]["kunz_targets"]) / f"{name}.npz"
        save = args.max_trials is None and not args.no_save
        if save and out.exists() and not args.overwrite:
            raise SystemExit(f"{_rel(out)} exists; pass --overwrite to replace it (or use another paths.kunz_targets)")
        sess = SessionData.load(Path(cfg["paths"]["kunz_processed"]) / f"{name}.npz")
        trials = sess.trials[: args.max_trials] if args.max_trials else sess.trials
        reorder = bool(k["reorder_channels"]) if args.reorder is None else args.reorder == "true"
        preds = tts_segs = pred_segs = None
        if decoded is not None:
            h = decoded[1]
            if h["session"] != name or h["n_bins"][: len(trials)] != [int(t.neural.shape[0]) for t in trials] or \
                    h["model"]["reorder_channels"] != reorder:
                raise SystemExit(f"--decoded {args.decoded} does not match {name} (trials or channel order)")
            preds = decoded[0][: len(trials)]
        if tts_table is not None:
            tts_segs = [tts_table[targets.text_key(t.text)] for t in trials]
        if preds is not None and (s["method"] != "current" or s["count_constraint"] or tts_segs is not None
                                  or s["arm"] == "B"):
            pred_segs = []
            for i, t in enumerate(trials):
                env = al.envelope_db(targets.prediction_timeline(preds[i], t.neural.shape[0], sil20, cfg), cfg)
                segs = sy.decoder_segments(s["method"], env, cfg, records[0][i] if records else None)
                if s["count_constraint"]:
                    n_tts = len(tts_segs[i]) if tts_segs is not None else len(al.segment_syllables(
                        al.envelope_db(targets.reduce(corpus.lookup(t.text), cfg), cfg), cfg))
                    segs = sy.keep_strongest(segs, env, n_tts)
                pred_segs.append(segs)
        elif s["count_constraint"]:
            raise SystemExit("count_constraint needs --decoded (the constraint is applied to saved predictions)")
        preds, results = targets.align_trials(model, trials, corpus, cfg, reorder=reorder, preds=preds,
                                              tts_segments=tts_segs, pred_segments=pred_segs, arm=s["arm"])
        n_syl, speech_frac, outside = [], [], []
        for t, res in zip(trials, results):
            t.target = targets.decoder_target(res.target_raw, cfg)
            t.meta["alignment"] = {
                "method": "wvx.alignment (D-020)",
                "syllables": {"tts_side": s["tts_side"], "decoder_method": s["method"], "arm": s["arm"],
                              "count_constraint": s["count_constraint"], "decoded_cell": args.decoded},
                "mapping": res.info,
                "voice": voice,
                "decoder": _rel(model_path),
                "n_pred_syllables": int(len(res.pred_segments)),
                "n_tts_syllables": int(len(res.tts_segments)),
                "mapped_segments": res.mapped_segments.tolist(),
                "dtw_cost": res.dtw_cost,
            }
            n_syl.append(len(res.mapped_segments))
            speech_frac.append(float((res.mapped_segments[:, 1] - res.mapped_segments[:, 0]).sum() / t.neural.shape[0]))
            outside.append(float(np.mean(np.abs(t.target) > 1)))
        sess.validate(n_features=int(cfg["model"]["n_features"]), n_targets=int(cfg["model"]["n_targets"]))
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
            "syllables": {**{key: s[key] for key in ("tts_side", "method", "count_constraint", "arm")},
                          "segment_mapping": s.get("segment_mapping") if s["arm"] == "B" else None,
                          "decoded_cell": args.decoded,
                          "decoded_created": decoded[1].get("created") if decoded else None,
                          "segments": records[1] if records else None},
            "decoder_output_sanity": targets.decoder_output_sanity(trials, results, cfg),
            "syllables_per_trial": {"min": int(min(n_syl)), "median": float(np.median(n_syl)), "max": int(max(n_syl))},
            "speech_fraction_per_trial": {"min": float(min(speech_frac)), "median": float(np.median(speech_frac)),
                                          "max": float(max(speech_frac))},
            "pred_syllables_zero": int(sum(int(t.meta["alignment"]["n_pred_syllables"] == 0) for t in trials)),
            "arm_b": None if s["arm"] != "B" else {
                k: int(sum((r.info or {}).get(k, 0) for r in results))
                for k in ("n_decoder_syllables_without_tts", "n_tts_syllables_without_frames")},
            "target_fraction_outside_unit_range": float(np.mean(outside)),
            "target_mean": tgt.mean(0).tolist(),
            "target_sd": tgt.std(0).tolist(),
            "alignment_config": cfg["alignment"],
            "code_version": code_version(),
            "config_sha256": config_sha256(cfg),
            "wall_time_s": time.time() - t0,
        }
        sp = Path(cfg["paths"]["alignment_validation"]) / (f"kunz_{cond}_summary" + (f"_{args.tag}" if args.tag else "") + ".json")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, default=str))
        print(f"[align_targets] {name}: {len(trials)} trials (arm {s['arm']}, {s['tts_side']} TTS syllables, decoder side {s['method']}"
              f"{', count-constrained' if s['count_constraint'] else ''}), syllables/trial median {np.median(n_syl):.0f}, "
              f"speech fraction median {np.median(speech_frac):.2f} -> {summary['saved']}, {_rel(sp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
