#!/usr/bin/env python
"""Speed-normalized ASR (D-073; user idea): transcribe decoder audio sped up to natural speaking rate, pitch kept.

    PC:   PATH=~/wvx/envs/wvx-tools/bin:$PATH .venv/bin/python -I scripts/speed_asr.py --config configs/experiments/speed_asr.yaml [--validate-only]

The decoders speak at T15's slow attempted pace; Whisper is trained on natural-rate speech. Every clip is the SAME
saved wav the dev test transcribed (results/audio/<run>/<model>/trNNN.wav; target audio in .../target/), so the
original and sped-up transcripts share the clip (no new vocoder draw). Speed-up: ffmpeg ``atempo`` (time scale
modification, pitch unchanged), chained in steps <= 2.
- ``fixed`` (primary): one factor for the session, the median over the dev TRAINING trials of (their target audio's
  speech seconds / the Alex TTS's speech seconds of the same cue); never uses the outputs being scored.
- ``per_trial`` (secondary): each clip's speech seconds / its cue's Alex speech seconds, clipped to [1, max_factor].
Speech seconds: the metrics' silence detector (``wvx.metrics.speech_frame_mask``, the TTS metric settings).
Also transcribed: the Alex TTS of each test cue (the natural-rate ceiling of that voice) and the target audio.
Writes results/speed_asr/<name>.json (per clip) and .md (pooled WER per model and paired comparisons).
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[speed_asr {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def atempo(wav, sr: int, factor: float, tmp: Path):
    """``wav`` sped up by ``factor`` (> 1 = faster) with ffmpeg atempo, pitch unchanged."""
    import numpy as np
    import soundfile as sf

    if abs(factor - 1.0) < 1e-6:
        return np.asarray(wav, np.float32)
    steps, f = [], float(factor)
    while f > 2.0:
        steps.append(2.0)
        f /= 2.0
    steps.append(f)
    src, dst = tmp / "in.wav", tmp / "out.wav"
    sf.write(str(src), np.asarray(wav, np.float32), sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter:a",
                    ",".join(f"atempo={s:.6f}" for s in steps), str(dst)], check=True)
    y, _ = sf.read(str(dst), dtype="float32")
    return y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/speed_asr.yaml")
    ap.add_argument("--validate-only", action="store_true", help="target audio, Alex and the first run's zero-shot only")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np
    import soundfile as sf
    from scipy.stats import wilcoxon

    from wvx import asr, metrics, synthesis, targets as tg, wairagkar_data
    from wvx.provenance import code_version

    c = cfg["speed_asr"]
    t0 = time.time()
    sr = int(cfg["synthesis"]["sample_rate"])
    mcfg = c["speech_metrics"]
    acfg = cfg["sanity"]["evaluation"]["asr"]["cfg"]
    hop = float(metrics._cfg(mcfg)["stft"]["hop_ms"]) / 1000.0
    split = json.loads(resolve_path(c["split_from"]).read_text())
    train, test = split["train_trials"], split["test_trials"]
    sess = wairagkar_data.load_session(resolve_path(cfg["sanity"]["data_dir"]), cfg["sanity"]["session"], 0, cfg)
    raw = wairagkar_data.load_raw_targets(resolve_path(cfg["sanity"]["data_dir"]), cfg["sanity"]["session"], cfg)
    cfg_t = load_experiment_config(c["voice_config"])
    alex = tg.TTSCorpus.load(tg.corpus_path(cfg_t, cfg_t["tts"]["voice"], cfg["sanity"]["session"]))

    def speech_s(w):
        return float(np.sum(metrics.speech_frame_mask(w, sr, mcfg))) * hop

    alex_wav = {i: synthesis.features_to_waveform(tg.reduce(alex.lookup(sess.trials[i].text), cfg), cfg)
                for i in train + test}
    alex_s = {i: speech_s(w) for i, w in alex_wav.items()}
    # fixed factor from the TRAINING trials: their (stored, full-trial) target audio vs Alex
    ratios = []
    for i in train:
        tw = synthesis.features_to_waveform(raw[i], cfg)
        if alex_s[i] > 0:
            ratios.append(speech_s(tw) / alex_s[i])
    fixed = float(np.clip(np.median(ratios), 1.0, float(c["max_factor"])))
    log(f"fixed factor (median over {len(ratios)} training trials, their targets / Alex): {fixed:.2f} "
        f"(IQR {np.percentile(ratios, 25):.2f}-{np.percentile(ratios, 75):.2f})")

    def transcribe(w, text):
        out = asr.evaluate_wav(w, sr, text, acfg)
        return {"wer": out.get("wer"), "hyp": out.get("hyp_raw"),
                "words": out.get("ref_words"), "errors": out.get("word_errors")}

    clips = [("alex_tts", None)] + [("target_audio", None)]
    runs = c["runs"][:1] if args.validate_only else c["runs"]
    for r in runs:
        models = r["models"][:1] if args.validate_only else r["models"]
        clips += [(r["name"], m) for m in models]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for run, model in clips:
            tm = time.time()
            for i in test:
                text = sess.trials[i].text
                if run == "alex_tts":
                    w = alex_wav[i]
                elif run == "target_audio":
                    w, _ = sf.read(str(resolve_path(f"results/audio/{c['runs'][0]['name']}/target/tr{i:03d}.wav")), dtype="float32")
                else:
                    w, _ = sf.read(str(resolve_path(f"results/audio/{run}/{model}/tr{i:03d}.wav")), dtype="float32")
                s = speech_s(w)
                per = float(np.clip(s / alex_s[i], 1.0, float(c["max_factor"]))) if alex_s[i] > 0 else 1.0
                row = {"run": run, "model": model or run, "trial_index": i, "text": text, "speech_s": s,
                       "alex_speech_s": alex_s[i], "per_trial_factor": per}
                row["orig"] = transcribe(w, text)
                if run != "alex_tts":
                    row["fixed"] = transcribe(atempo(w, sr, fixed, tmp), text)
                    row["per_trial"] = transcribe(atempo(w, sr, per, tmp), text)
                rows.append(row)
            log(f"{run}/{model or ''}: {len(test)} clips in {time.time() - tm:.0f} s")

    def pooled(rs, key):
        e = sum(float(x[key]["errors"]) for x in rs if x.get(key) and x[key]["errors"] is not None)
        n = sum(float(x[key]["words"]) for x in rs if x.get(key) and x[key]["words"] is not None)
        return e / n if n else float("nan")

    summary, by = {}, {}
    for x in rows:
        by.setdefault((x["run"], x["model"]), []).append(x)
    for (run, model), rs in by.items():
        summary[f"{run}/{model}"] = {k: pooled(rs, k) for k in ("orig", "fixed", "per_trial") if k in rs[0]}
    comps = {}
    # zero-shot audio is byte-identical in every dev run (checked before this run), so the first run's is the
    # reference for every model
    z = {x["trial_index"]: x for x in by.get((runs[0]["name"], "zero_shot"), [])}
    for r in runs:
        for m in r["models"]:
            if m == "zero_shot" or (r["name"], m) not in by:
                continue
            rs = by[(r["name"], m)]
            if not z:
                continue
            for k in ("orig", "fixed", "per_trial"):
                d = [x[k]["wer"] - z[x["trial_index"]][k]["wer"] for x in rs
                     if x[k]["wer"] is not None and z[x["trial_index"]][k]["wer"] is not None]
                comps.setdefault(f"{r['name']}/{m}", {})[k] = {
                    "mean_diff": float(np.mean(d)), "p": float(wilcoxon(d).pvalue) if np.any(np.array(d) != 0) else 1.0}
    out = {"name": c["name"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "code_version": code_version(), "settings": c, "fixed_factor": fixed,
           "fixed_factor_iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
           "summary_pooled_wer": summary, "vs_zero_shot": comps, "rows": rows, "wall_time_s": time.time() - t0}
    res = resolve_path("results/speed_asr")
    res.mkdir(parents=True, exist_ok=True)
    name = c["name"] + ("_validate" if args.validate_only else "")
    (res / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = [f"# Speed-normalized ASR `{name}` (D-073): pooled WER, {len(test)} day195 test trials", "",
             f"Fixed factor {fixed:.2f} (their target audio / Alex, median of {len(ratios)} training trials).", "",
             "| clips | original | fixed speed-up | per-trial speed-up |", "|---|---|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {100 * v['orig']:.1f}% | " + (f"{100 * v['fixed']:.1f}%" if "fixed" in v else "–") + " | "
                     + (f"{100 * v['per_trial']:.1f}%" if "per_trial" in v else "–") + " |")
    if comps:
        lines += ["", "| model vs its run's zero-shot | original Δ (p) | fixed Δ (p) | per-trial Δ (p) |", "|---|---|---|---|"]
        for k, v in comps.items():
            lines.append(f"| {k} | " + " | ".join(f"{100 * v[s]['mean_diff']:+.1f} ({v[s]['p']:.2g})"
                                                   for s in ("orig", "fixed", "per_trial")) + " |")
    (res / f"{name}.md").write_text("\n".join(lines) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/speed_asr/{name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
