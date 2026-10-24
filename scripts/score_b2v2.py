#!/usr/bin/env python
"""Score brain2voice-2.0-style decodes, or their old decoder's closed-loop audio, on the same trials (D-075).

    project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 10 .venv/bin/python -I scripts/score_b2v2.py --config configs/experiments/b2v2_all.yaml --preds test
    baseline (host: needs the raw session files):
    ... --closed-loop benchmark      (their live audio, brain2voice_pred_audio_16kHz, of the benchmark trials)

Per trial: Whisper WER of the synthesized audio at its own pace and sped up by the D-073 fixed factor (pitch kept;
``b2v2.score.speed_factor``), and for decodes also the phoneme error rate (greedy CTC of the per-bin argmax vs the
cue's label, SIL ignored) and the feature r vs the (unsmoothed) targets. Pooled WER = sum of word errors / sum of
reference words. Writes results/b2v2/<run_name>/scores_<name>.json and .md; audio to
results/audio/b2v2/<run_name>/<name>/ (gitignored).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[score_b2v2 {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preds", help="name of data/processed/b2v2_preds/<run_name>/<name>.npz")
    g.add_argument("--closed-loop", choices=["benchmark"], help="their live audio of these trials")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np
    import scipy.io

    from speed_asr import atempo
    from wvx import asr, b2v2_data as bd, kunz_experiment as kx, phonemes as ph, synthesis, wairagkar_data
    from wvx.provenance import code_version

    b, run = cfg["b2v2"], cfg["run_name"]
    sc = b["score"]
    data = bd.B2V2Data(resolve_path(b["data_dir"]))
    factor = float(sc["speed_factor"])
    acfg = sc["asr"]
    t0 = time.time()
    clips = {}                                                     # gid -> (wav, sr, extra row fields)
    if args.preds:
        name = args.preds
        with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            for gid in header["gids"]:
                t = data.by_gid[gid]
                cont, best = z[f"g{gid}_cont"], z[f"g{gid}_best"]
                _, target = data.trial_arrays(t)
                extra = {"feat_r": kx.feature_scores(cont, target)["feat_corr_mean"]}
                if t["label"] is not None:
                    e, n = ph.phoneme_error(t["label"], ph.ctc_greedy(best))
                    extra.update(phone_errors=e, ref_phones=n)
                clips[gid] = (synthesis.decoder_output_to_waveform(cont, cfg), int(cfg["synthesis"]["sample_rate"]), extra)
    else:
        name = f"closed_loop_{args.closed_loop}"
        header = {"source": "closed-loop audio", "key": b["closed_loop_key"]}
        want = [t for t in data.trials if t["benchmark"]]
        data_dir = resolve_path(cfg["paths"]["wairagkar_neural_data"])
        for sess in sorted({t["session"] for t in want}):
            path = wairagkar_data.session_files(data_dir, sess, cfg)[0]
            audio = np.atleast_1d(scipy.io.loadmat(str(path), squeeze_me=True, variable_names=[b["closed_loop_key"]])[
                b["closed_loop_key"]])
            for t in want:
                if t["session"] == sess:
                    clips[t["gid"]] = (np.asarray(audio[t["index"]], np.float32).ravel(),
                                       int(cfg["closed_loop"]["sample_rate"]), {})
        header["gids"] = sorted(clips)
    log(f"{name}: {len(clips)} clips")
    adir = resolve_path("results/audio/b2v2") / run / name
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for i, (gid, (wav, sr, extra)) in enumerate(sorted(clips.items())):
            t = data.by_gid[gid]
            synthesis.write_wav(adir / f"g{gid}.wav", wav, cfg, sr=sr)
            row = {"gid": gid, "session": t["session"], "index": t["index"], "benchmark": t["benchmark"],
                   "text": t["text"], **extra}
            for k, w in (("orig", wav), ("fixed", atempo(wav, sr, factor, Path(td)))):
                out = asr.evaluate_wav(w, sr, t["text"], acfg)
                row[k] = {"hyp": out.get("hyp_raw"), "errors": out.get("word_errors"), "words": out.get("ref_words"),
                          "wer": out.get("wer")}
            rows.append(row)
            if (i + 1) % 20 == 0:
                log(f"  {i + 1}/{len(clips)} ({time.time() - t0:.0f} s)")

    def pooled(rs, k):
        e = sum(float(r[k]["errors"]) for r in rs if r[k]["errors"] is not None)
        n = sum(float(r[k]["words"]) for r in rs if r[k]["words"] is not None)
        return e / n if n else float("nan")

    groups = {"all": rows}
    for s in sorted({r["benchmark"] for r in rows if r["benchmark"]}):
        groups[s] = [r for r in rows if r["benchmark"] == s]
    summary = {}
    for k, rs in groups.items():
        summary[k] = {"n": len(rs), "wer_orig": pooled(rs, "orig"), "wer_fixed": pooled(rs, "fixed")}
        pr = [r for r in rs if "ref_phones" in r]
        if pr:
            summary[k]["per"] = sum(r["phone_errors"] for r in pr) / max(1, sum(r["ref_phones"] for r in pr))
        fr = [r["feat_r"] for r in rs if "feat_r" in r and np.isfinite(r["feat_r"])]
        if fr:
            summary[k]["feat_r"] = float(np.mean(fr))
    out = {"run_name": run, "name": name, "decision": "D-075", "config": args.config, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "source": header,
           "speed_factor": factor, "summary": summary, "rows": rows, "wall_time_s": time.time() - t0}
    res = resolve_path("results/b2v2") / run
    res.mkdir(parents=True, exist_ok=True)
    (res / f"scores_{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = [f"# `{run}` / `{name}` (D-075)", "", f"Speed-up factor {factor} (D-073 fixed).", "",
             "| trials | n | WER | WER sped up | PER | feature r |", "|---|---|---|---|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {v['n']} | {100 * v['wer_orig']:.1f}% | {100 * v['wer_fixed']:.1f}% | "
                     + (f"{100 * v['per']:.1f}%" if "per" in v else "–") + " | "
                     + (f"{v['feat_r']:.3f}" if "feat_r" in v else "–") + " |")
    (res / f"scores_{name}.md").write_text("\n".join(lines) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/b2v2/{run}/scores_{name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
