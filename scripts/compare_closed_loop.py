"""Compare our offline inference with the audio their decoder produced live, per trial type.

    python -I scripts/compare_closed_loop.py [--config configs/experiments/reproduce_inference.yaml]

Run after scripts/run_inference.py (same config). For every trial with live (closed-loop) audio it
scores our offline audio against the live audio (do we reproduce what was heard?), and both against
the target audio. All metrics are also summarized per trial type (`trial_description`), since the
paper reports evaluation trials. Writes closed_loop_comparison.csv and closed_loop_summary.json next
to run_inference's outputs.

Lead-in and lag search. The live stream starts at trial onset; offline predictions start once the first full
window exists, (window_len - 1) bins later, so the live audio is trimmed by that lead-in before comparing. These
configured-trim columns are the primary result and are computed exactly as before. Because a trim that is off by a
few frames lowers the frame-wise agreement, the script also scans trims of lead-in + lag frames for lag in
[-N, N] (N = closed_loop.lag_search_frames). The best lag maximizes the mean frame-wise mel_corr of ours vs live
over all trials with live audio (ties: smaller |lag|, then the earlier trim). Columns ``*_best_lag`` hold every
metric at that one lag. ``best_lag_frames_trial`` is each trial's own best lag (reported, never used for a metric).
The best lag is chosen on these same trials, so its metrics are an in-sample upper bound for a fixed trim.

closed_loop_summary.json: one entry per trial type (as before), plus ``_lag_search`` (configured trim, scanned
lags, mean frame-wise mel_corr per lag, best lag) and ``_provenance`` (code version, config hash, input hashes).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.dont_write_bytecode = True

from wvx import metrics  # noqa: E402
from wvx.config import REPO_ROOT, deep_merge, load_config, resolve_path  # noqa: E402
from wvx.provenance import code_version, config_sha256  # noqa: E402
from wvx.seeding import set_seed  # noqa: E402

PRIMARY = ("ours_vs_live_mel_corr", "ours_vs_live_mel_corr_framewise", "live_vs_target_mel_corr")


def _summary(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "sd": round(float(v.std()), 4)}


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _rel(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def framewise_cfg(mcfg: dict) -> dict:
    """Metric settings that make ``metrics.mel_spectrogram_correlation`` the frame-wise mel_corr of
    ``metrics.evaluate`` (no silence removal, no DTW, truncate to the shorter signal)."""
    return deep_merge(mcfg, {"mel_corr": {"remove_silence": False, "dtw": False},
                             "alignment": {"without_dtw": "truncate"}})


def lag_scan(live: np.ndarray, pred: np.ndarray, sr: int, lead: int, hop: int, lags, fcfg: dict) -> dict:
    """{lag: frame-wise mel_corr(live[lead + lag * hop:], pred)} for every lag whose trim lies inside ``live``."""
    out = {}
    for lag in lags:
        start = lead + lag * hop
        if 0 <= start < live.size:
            out[int(lag)] = float(metrics.mel_spectrogram_correlation(live[start:], pred, sr, fcfg))
    return out


def best_lag(score_by_lag: dict):
    """Lag with the highest finite score (ties: smaller |lag|, then smaller lag); None if there is none."""
    ok = {lag: v for lag, v in score_by_lag.items() if v is not None and np.isfinite(v)}
    if not ok:
        return None
    return min(ok, key=lambda lag: (-ok[lag], abs(lag), lag))


def mean_by_lag(curves: list) -> dict:
    """{lag: mean score} over the lags scored (finite) in every curve."""
    if not curves:
        return {}
    lags = sorted(set.intersection(*(set(lag for lag, v in c.items() if np.isfinite(v)) for c in curves)))
    return {lag: float(np.mean([c[lag] for c in curves])) for lag in lags}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/reproduce_inference.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    cl, session = cfg["closed_loop"], cfg["inference"]["session"]
    # The metrics section itself (configs/metrics.yaml + the experiment's overrides): the same settings as before.
    mcfg = metrics.load_metrics_config(overrides=cfg.get("metrics") or None)
    fcfg = framewise_cfg(mcfg)
    run_dir = resolve_path(cfg["paths"]["results"]) / cfg["run_name"]
    neural = resolve_path(cfg["paths"]["wairagkar_neural_data"]) / session / cfg["wairagkar_data"]["neural_file"].format(session=session)
    m = scipy.io.loadmat(neural, variable_names=[cl["audio_key"], cl["description_key"]])
    live = m[cl["audio_key"]].ravel()
    desc = [str(np.asarray(d).squeeze()).strip() for d in np.asarray(m[cl["description_key"]]).ravel()]

    per_trial_csv = run_dir / "per_trial_metrics.csv"
    with open(per_trial_csv) as f:
        ours = {int(r["trial"]): r for r in csv.DictReader(f)}

    sr = int(cl["sample_rate"])
    hop = int(cl["bin_ms"]) * sr // 1000                      # one decoder frame (10 ms bin) in samples
    lead_frames = int(cfg["model"]["window_len"]) - 1
    lead = lead_frames * hop
    n_lag = int(cl["lag_search_frames"])
    lags = list(range(-n_lag, n_lag + 1))

    def clip(i):  # (live audio, our offline audio, target audio) of trial i; re-read when needed, not kept
        return (np.asarray(live[i], float).ravel(), sf.read(resolve_path(ours[i]["pred_wav"]))[0],
                sf.read(resolve_path(ours[i]["target_wav"]))[0])

    rows, by_type, curves = [], defaultdict(lambda: defaultdict(list)), {}
    for i, r in sorted(ours.items()):
        row = {"trial": i, "type": desc[i], "text": r["text"], "ours_vs_target_mel_corr": float(r["mel_corr"])}
        by_type[desc[i]]["ours_vs_target_mel_corr"].append(row["ours_vs_target_mel_corr"])
        if np.asarray(live[i]).size:
            audio, pred, tgt = clip(i)
            # The live stream starts at trial onset; offline predictions start once the first full window
            # exists (window_len - 1 bins later). Drop that lead-in so both cover the same frames.
            trimmed = audio[lead:]
            agree = metrics.evaluate(trimmed, pred, sr, mcfg)
            live_q = metrics.evaluate(tgt, trimmed, sr, mcfg)
            row.update({
                "ours_vs_live_mel_corr": agree["mel_corr"],
                "ours_vs_live_mel_corr_framewise": agree["mel_corr_framewise"],
                "live_vs_target_mel_corr": live_q["mel_corr"],
                "live_s": round(trimmed.size / sr, 2), "ours_s": round(pred.size / sr, 2),
            })
            for k in PRIMARY:
                by_type[desc[i]][k].append(row[k])
            curves[i] = lag_scan(audio, pred, sr, lead, hop, lags, fcfg)
            b = best_lag(curves[i])
            row["best_lag_frames_trial"] = b
            row["ours_vs_live_mel_corr_framewise_trial_best"] = curves[i][b] if b is not None else float("nan")
        rows.append(row)

    # One lag for all trials: the best mean frame-wise agreement; every metric again at that trim.
    lag_means = mean_by_lag(list(curves.values()))
    lag = best_lag(lag_means)
    if lag is not None:
        for row in rows:
            i = row["trial"]
            if i not in curves:
                continue
            audio, pred, tgt = clip(i)
            shifted = audio[lead + lag * hop:]
            agree = metrics.evaluate(shifted, pred, sr, mcfg)
            live_q = metrics.evaluate(tgt, shifted, sr, mcfg)
            row.update({
                "ours_vs_live_mel_corr_best_lag": agree["mel_corr"],
                "ours_vs_live_mel_corr_framewise_best_lag": agree["mel_corr_framewise"],
                "live_vs_target_mel_corr_best_lag": live_q["mel_corr"],
                "live_s_best_lag": round(shifted.size / sr, 2),
            })
            for k in PRIMARY:
                by_type[row["type"]][f"{k}_best_lag"].append(row[f"{k}_best_lag"])
            by_type[row["type"]]["best_lag_frames_trial"].append(float(row["best_lag_frames_trial"])
                                                                 if row["best_lag_frames_trial"] is not None else None)

    keys = list(dict.fromkeys(k for r in rows for k in r))   # first-seen order
    with open(run_dir / "closed_loop_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = {t: {k: _summary(v) for k, v in d.items()} for t, d in sorted(by_type.items())}
    summary["_lag_search"] = {
        "configured_trim_frames": lead_frames, "frame_ms": int(cl["bin_ms"]), "max_lag_frames": n_lag,
        "criterion": "mean frame-wise mel_corr, ours vs live, over all trials with live audio (in-sample)",
        "n_trials": len(curves), "best_lag_frames": lag,
        "best_trim_frames": None if lag is None else lead_frames + lag,
        "best_lag_at_search_edge": None if lag is None else abs(lag) == n_lag,
        "mean_framewise_by_lag": {str(k): round(v, 4) for k, v in lag_means.items()},
        "per_trial_best_lag_counts": {str(k): v for k, v in sorted(Counter(
            r["best_lag_frames_trial"] for r in rows if r.get("best_lag_frames_trial") is not None).items())},
    }
    run_summary = run_dir / "summary.json"
    summary["_provenance"] = {
        "code_version": code_version(), "config_file": args.config, "config_sha256": config_sha256(cfg),
        "inputs": {"per_trial_metrics_csv": {"path": _rel(per_trial_csv), "sha256": _sha256(per_trial_csv)},
                   "run_inference_code_version": (json.loads(run_summary.read_text()).get("code_version")
                                                  if run_summary.is_file() else None),
                   "neural_file": _rel(neural)},
    }
    (run_dir / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2))
    for t, d in summary.items():
        if not t.startswith("_"):
            print(t, json.dumps(d))
    ls = summary["_lag_search"]
    print(f"lag search: best lag {ls['best_lag_frames']} frames (trim {ls['best_trim_frames']} vs configured "
          f"{lead_frames}); mean frame-wise by lag {json.dumps(ls['mean_framewise_by_lag'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
