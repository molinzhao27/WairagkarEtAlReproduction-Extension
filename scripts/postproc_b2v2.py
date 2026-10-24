#!/usr/bin/env python
"""D-078: post-processing of decoded voice (denoiser, +2 semitones, feature sharpening), scored by Whisper WER.

    PC, project venv, CPU:
    PATH=~/wvx/envs/wvx-tools/bin:$PATH CUDA_VISIBLE_DEVICES= nice -n 10 .venv/bin/python -I scripts/postproc_b2v2.py --config configs/experiments/b2v2_postproc.yaml

Per set (``b2v2.postproc.sets``: decode runs + prediction name) and trial, every variant in ``variants``, applied
uniformly (``wvx.postproc``): ``orig`` = the saved decoder audio (results/audio/b2v2/<run>/<name>/g<gid>.wav);
``denoise`` / ``pitch2`` = that audio processed; ``sharpen`` = the saved features with each cepstral deviation scaled
by ``sharpen_alpha``, re-vocoded; ``all`` = sharpen -> denoise -> pitch2. Whisper WER at the original speed and sped up
(D-073 factor). Writes results/b2v2/postproc/<set>.json / .md (pooled WER, paired per-trial tests vs ``orig``) and the
processed audio to results/audio/b2v2_postproc/<set>/<variant>/ (gitignored).
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

CFG = None


def _init(config_path):
    global CFG
    CFG = load_experiment_config(config_path)


def one(job):
    """All variants of one trial (runs in a worker process)."""
    import numpy as np
    import soundfile as sf

    from speed_asr import atempo
    from wvx import asr, postproc as pp, synthesis

    cfg = CFG
    b, p = cfg["b2v2"], cfg["b2v2"]["postproc"]
    run, name, gid, text, set_name = job
    sr = int(cfg["synthesis"]["sample_rate"])
    base, _ = sf.read(str(resolve_path("results/audio/b2v2") / run / name / f"g{gid}.wav"), dtype="float32")
    with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False) as z:
        cont = z[f"g{gid}_cont"]
    dn = p["denoise"]

    def denoise(w):
        return pp.spectral_gate(w, sr, **dn)

    def pitch(w):
        return pp.pitch_shift(w, sr, float(p["pitch_semitones"]))

    sharp = None
    out = {"gid": gid, "run": run, "text": text}
    for v in p["variants"]:
        if v == "orig":
            w = base
        elif v == "denoise":
            w = denoise(base)
        elif v == "pitch2":
            w = pitch(base)
        elif v in ("sharpen", "all"):
            if sharp is None:
                sharp = synthesis.decoder_output_to_waveform(pp.sharpen(cont, float(p["sharpen_alpha"])), cfg)
            w = sharp if v == "sharpen" else pitch(denoise(sharp))
        else:
            raise ValueError(f"unknown variant {v}")
        synthesis.write_wav(resolve_path("results/audio/b2v2_postproc") / set_name / v / f"g{gid}.wav", w, cfg, sr=sr)
        rec = {}
        with tempfile.TemporaryDirectory() as td:
            for k, ww in (("orig_speed", w), ("sped_up", atempo(w, sr, float(b["score"]["speed_factor"]), Path(td)))):
                r = asr.evaluate_wav(ww, sr, text, b["score"]["asr"])
                rec[k] = {"hyp": r.get("hyp_raw"), "errors": r.get("word_errors"), "words": r.get("ref_words"),
                          "wer": r.get("wer")}
        out[v] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_postproc.yaml")
    ap.add_argument("--limit", type=int, default=0, help="first N trials per set; writes *_smoke results (a code check)")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from scipy.stats import wilcoxon

    from wvx import b2v2_data as bd
    from wvx.provenance import code_version

    b, p = cfg["b2v2"], cfg["b2v2"]["postproc"]
    data = bd.B2V2Data(resolve_path(b["data_dir"]))
    res = resolve_path("results/b2v2/postproc")
    res.mkdir(parents=True, exist_ok=True)
    for set_name, s in p["sets"].items():
        t0 = time.time()
        jobs = []
        for run in s["runs"]:
            with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{s['name']}.npz", allow_pickle=False) as z:
                for g in json.loads(str(z["header"]))["gids"]:
                    jobs.append((run, s["name"], g, data.by_gid[g]["text"], set_name))
        if args.limit:
            jobs, set_name = jobs[: args.limit], f"{set_name}_smoke"
            jobs = [j[:4] + (set_name,) for j in jobs]
        with ProcessPoolExecutor(int(p["workers"]), initializer=_init, initargs=(args.config,)) as ex:
            rows = list(ex.map(one, jobs))

        def pooled(v, k):
            e = sum(r[v][k]["errors"] for r in rows if r[v][k]["errors"] is not None)
            n = sum(r[v][k]["words"] for r in rows if r[v][k]["words"] is not None)
            return e / n if n else float("nan")

        summary, alpha = {}, float(p["rule"]["alpha"])
        for v in p["variants"]:
            summary[v] = {k: pooled(v, k) for k in ("orig_speed", "sped_up")}
            if v != "orig":
                for k in ("orig_speed", "sped_up"):
                    d = np.array([r[v][k]["wer"] - r["orig"][k]["wer"] for r in rows
                                  if r[v][k]["wer"] is not None and r["orig"][k]["wer"] is not None])
                    summary[v][f"{k}_paired"] = {"mean_diff": float(d.mean()), "n": int(len(d)),
                                                 "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0}
                pr = summary[v]["orig_speed_paired"]
                summary[v]["helps"] = bool(summary["orig"]["orig_speed"] - summary[v]["orig_speed"]
                                           >= float(p["rule"]["min_points"]) / 100 and pr["p"] < alpha
                                           and pr["mean_diff"] < 0)
        out = {"set": set_name, "decision": "D-078", "config": args.config, "code_version": code_version(),
               "created": datetime.datetime.now().isoformat(timespec="seconds"), "settings": p, "n": len(rows),
               "summary": summary, "rows": rows, "wall_time_s": time.time() - t0}
        (res / f"{set_name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        lines = [f"# D-078 post-processing: `{set_name}` ({len(rows)} trials)", "",
                 "| variant | WER | vs orig (p) | WER sped up | vs orig (p) | rule |", "|---|---|---|---|---|---|"]
        for v, x in summary.items():
            a = x.get("orig_speed_paired")
            bb = x.get("sped_up_paired")
            lines.append(f"| {v} | {100 * x['orig_speed']:.1f}% | " +
                         (f"{100 * a['mean_diff']:+.1f} ({a['p']:.2g})" if a else "–") +
                         f" | {100 * x['sped_up']:.1f}% | " + (f"{100 * bb['mean_diff']:+.1f} ({bb['p']:.2g})" if bb else "–") +
                         " | " + ("helps" if x.get("helps") else ("–" if v == "orig" else "no")) + " |")
        (res / f"{set_name}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
