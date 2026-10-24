#!/usr/bin/env python
"""Drift map of T15's neural features across sessions and Kunz (D-071; descriptive).

Host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/drift_map.py --config configs/experiments/drift_map.yaml

Inputs in the stored channel order ([spike-band power | threshold crossings], 256 each; arrays = channel blocks
0-63 v6v, 64-127 M1, 128-191 55b, 192-255 d6v; Kunz has the same order, D-024). Every trial pooled to 50 ms frames.
Per session, kept: per feature type and array, the channel-correlation matrix; per array (both feature types,
128 columns) and overall (512), the top-k principal axes; per array, the mean skewness of the threshold-crossing
features (zero-inflation: higher = fewer crossings; the rolling z-score keeps the shape, not the level).
Per (reference, later session) pair: correlation similarity (Pearson r of the matrices' upper triangles) and
subspace overlap (mean squared cosine of the principal angles). Writes results/drift_map/<name>.json and .md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402

ARRAYS = ("v6v", "M1", "55b", "d6v")


def log(msg: str) -> None:
    print(f"[drift_map {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/drift_map.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np
    from scipy.stats import skew

    from wvx import neural_match as nm, wairagkar_data
    from wvx.provenance import code_version
    from wvx.session import SessionData

    d = cfg["drift_map"]
    k, pool = int(d["n_components"]), int(d["pool_bins"])
    t0 = time.time()

    def axes(x, kk):
        xc = x - x.mean(0)
        w, v = np.linalg.eigh(xc.T @ xc / (len(xc) - 1))
        return v[:, np.argsort(w)[::-1][:kk]]

    def summarize(label, trials):
        x = np.concatenate([nm.pool_frames(np.asarray(t, np.float64), pool) for t in trials])
        s = {"label": label, "n_frames": int(len(x)), "corr": {}, "axes": {}, "tx_skew": {}}
        for ft, off in (("sbp", 0), ("tx", 256)):
            for a, name in enumerate(ARRAYS):
                cols = x[:, off + 64 * a: off + 64 * (a + 1)]
                s["corr"][f"{ft}_{name}"] = np.corrcoef(cols, rowvar=False)
            s["corr"][f"{ft}_all"] = np.corrcoef(x[:, off: off + 256], rowvar=False)
        for a, name in enumerate(ARRAYS):
            idx = np.r_[64 * a: 64 * (a + 1), 256 + 64 * a: 256 + 64 * (a + 1)]
            s["axes"][name] = axes(x[:, idx], k)
            s["tx_skew"][name] = float(np.nanmean(skew(x[:, 256 + 64 * a: 256 + 64 * (a + 1)], axis=0)))
        s["axes"]["all"] = axes(x, k)
        return s

    sessions = {}
    data_dir = resolve_path(cfg["sanity"]["data_dir"])
    for name in d["sessions"]:
        sess = wairagkar_data.load_session(data_dir, name, 0, cfg)
        sessions[name] = summarize(name, [t.neural for t in sess.trials]) | {"day": int(name.split("day")[1])}
        del sess
        log(f"{name}: {sessions[name]['n_frames']} frames")
    kproc = resolve_path(d["kunz_processed"])
    for cond in d["kunz_conditions"]:
        sess = SessionData.load(kproc / f"{d['kunz_session']}_{cond}.npz")
        key = f"kunz_{cond}"
        sessions[key] = summarize(key, [t.neural for t in sess.trials]) | {"day": float(d["kunz_day"]),
                                                                        "day_range": d["kunz_day_range"]}
        del sess
        log(f"{key}: {sessions[key]['n_frames']} frames")

    def tri(m):
        return m[np.triu_indices(m.shape[0], 1)]

    def corr_sim(a, b):
        u, v = tri(a), tri(b)
        ok = np.isfinite(u) & np.isfinite(v)
        return float(np.corrcoef(u[ok], v[ok])[0, 1])

    def overlap(U, V):
        return float(np.sum((U.T @ V) ** 2) / U.shape[1])

    pairs = []
    names = list(sessions)
    for r in d["references"]:
        R = sessions[r]
        for s in names:
            S = sessions[s]
            if s == r or (not s.startswith("kunz") and S["day"] <= R["day"]):
                continue
            row = {"reference": r, "session": s, "gap_days": float(S["day"] - R["day"]),
                   "corr_similarity": {key: corr_sim(R["corr"][key], S["corr"][key]) for key in R["corr"]},
                   "subspace_overlap": {key: overlap(R["axes"][key], S["axes"][key]) for key in R["axes"]}}
            pairs.append(row)
    # within-session noise floor: odd vs even trials of the references
    floors = {}
    for r in d["references"]:
        sess = wairagkar_data.load_session(data_dir, r, 0, cfg)
        A = summarize(f"{r} odd", [t.neural for t in sess.trials[0::2]])
        B = summarize(f"{r} even", [t.neural for t in sess.trials[1::2]])
        floors[r] = {"corr_similarity": {key: corr_sim(A["corr"][key], B["corr"][key]) for key in A["corr"]},
                     "subspace_overlap": {key: overlap(A["axes"][key], B["axes"][key]) for key in A["axes"]}}
        del sess
    out = {"name": d["name"], "config": args.config, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "code_version": code_version(), "settings": d, "pairs": pairs, "within_session": floors,
           "tx_skew": {s: v["tx_skew"] for s, v in sessions.items()},
           "days": {s: v["day"] for s, v in sessions.items()}, "wall_time_s": time.time() - t0}
    res = resolve_path("results/drift_map")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{d['name']}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = [f"# Drift map `{d['name']}` (D-071)", "",
             "Subspace overlap (top-%d PCs, 1 = same) and correlation similarity (SBP, all channels) vs the reference;" % k,
             f"Kunz placed at day {d['kunz_day']} (range {d['kunz_day_range']}).", "",
             "| reference | session | gap (d) | overlap all | v6v | M1 | 55b | d6v | corr sim SBP | corr sim TX |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        o, c = p["subspace_overlap"], p["corr_similarity"]
        lines.append(f"| {p['reference']} | {p['session']} | {p['gap_days']:.0f} | {o['all']:.2f} | {o['v6v']:.2f} | "
                     f"{o['M1']:.2f} | {o['55b']:.2f} | {o['d6v']:.2f} | {c['sbp_all']:.2f} | {c['tx_all']:.2f} |")
    lines += ["", "Within-session (odd vs even trials): " + "; ".join(
        f"{r}: overlap {v['subspace_overlap']['all']:.2f}, corr sim SBP {v['corr_similarity']['sbp_all']:.2f}"
        for r, v in floors.items())]
    (res / f"{d['name']}.md").write_text("\n".join(lines) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/drift_map/{d['name']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
