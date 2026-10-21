"""Compare our offline inference with the audio their decoder produced live, per trial type.

    python -I scripts/compare_closed_loop.py [--config configs/experiments/reproduce_inference.yaml]

Run after scripts/run_inference.py (same config). For every trial with live (closed-loop) audio it
scores our offline audio against the live audio (do we reproduce what was heard?), and both against
the target audio. All metrics are also summarized per trial type (`trial_description`), since the
paper reports evaluation trials. Writes closed_loop_comparison.csv and closed_loop_summary.json next
to run_inference's outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.dont_write_bytecode = True

from wvx import metrics  # noqa: E402
from wvx.config import load_config, load_yaml, resolve_path  # noqa: E402


def _summary(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "sd": round(float(v.std()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/reproduce_inference.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cl, session = cfg["closed_loop"], cfg["inference"]["session"]
    mcfg = cfg.get("metrics") or load_yaml(resolve_path("configs/metrics.yaml"))
    run_dir = resolve_path(cfg["paths"]["results"]) / cfg["run_name"]
    neural = resolve_path(cfg["paths"]["wairagkar_neural_data"]) / session / cfg["wairagkar_data"]["neural_file"].format(session=session)
    m = scipy.io.loadmat(neural, variable_names=[cl["audio_key"], cl["description_key"]])
    live = m[cl["audio_key"]].ravel()
    desc = [str(np.asarray(d).squeeze()).strip() for d in np.asarray(m[cl["description_key"]]).ravel()]

    with open(run_dir / "per_trial_metrics.csv") as f:
        ours = {int(r["trial"]): r for r in csv.DictReader(f)}

    rows, by_type = [], defaultdict(lambda: defaultdict(list))
    for i, r in sorted(ours.items()):
        row = {"trial": i, "type": desc[i], "text": r["text"], "ours_vs_target_mel_corr": float(r["mel_corr"])}
        by_type[desc[i]]["ours_vs_target_mel_corr"].append(row["ours_vs_target_mel_corr"])
        audio = np.asarray(live[i], float).ravel()
        if audio.size:
            pred, _ = sf.read(resolve_path(r["pred_wav"]))
            tgt, _ = sf.read(resolve_path(r["target_wav"]))
            sr = cl["sample_rate"]
            # The live stream starts at trial onset; offline predictions start once the first full window
            # exists (window_len - 1 bins later). Drop that lead-in so both cover the same frames.
            lead = (cfg["model"]["window_len"] - 1) * cl["bin_ms"] * sr // 1000
            audio = audio[lead:]
            agree = metrics.evaluate(audio, pred, sr, mcfg)
            live_q = metrics.evaluate(tgt, audio, sr, mcfg)
            row.update({
                "ours_vs_live_mel_corr": agree["mel_corr"],
                "ours_vs_live_mel_corr_framewise": agree["mel_corr_framewise"],
                "live_vs_target_mel_corr": live_q["mel_corr"],
                "live_s": round(audio.size / sr, 2), "ours_s": round(pred.size / sr, 2),
            })
            for k in ("ours_vs_live_mel_corr", "ours_vs_live_mel_corr_framewise", "live_vs_target_mel_corr"):
                by_type[desc[i]][k].append(row[k])
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))   # first-seen order
    with open(run_dir / "closed_loop_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = {t: {k: _summary(v) for k, v in d.items()} for t, d in sorted(by_type.items())}
    (run_dir / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2))
    for t, d in summary.items():
        print(t, json.dumps(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
