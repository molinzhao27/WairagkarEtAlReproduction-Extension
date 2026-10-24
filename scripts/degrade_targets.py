#!/usr/bin/env python
"""Their real day195 training targets, degraded in controlled ways (D-072), as target files for the dev test.

    PC:   .venv/bin/python -I scripts/degrade_targets.py --config configs/experiments/target_sens_day195.yaml

Targets are their decoder targets (normalized, smoothed; ``wvx.wairagkar_data``) of the D-049 training trials. Per
feature, standardized over all training frames (mean mu, s.d. sd); then
- ``mix_<r>``: z' = z @ ((1 - lam) I + lam Q), Q one fixed seeded random orthogonal 20 x 20 matrix (the same for every
  trial: a systematic, voice-like change), re-standardized per feature; lam by bisection so that the mean per-trial
  feature r vs the real targets is <r>;
- ``noise_<r>``: z' = z + s * eta, eta Gaussian noise smoothed over time like their targets (sigma 2 frames), drawn per
  trial (unbiased: its expectation is the real target); s by bisection so the mean per-trial r is <r>;
then t' = z' * sd + mu. Writes <out_dir>/day195_<variant>.npz (keys trNNN, JSON header) and
results/target_sensitivity/<name>.json (lam / s and the achieved r per variant).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[degrade_targets {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/target_sens_day195.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    from wvx import kunz_experiment as kx, wairagkar_data
    from wvx.provenance import code_version

    d = cfg["degrade_targets"]
    train = json.loads(resolve_path(d["train_trials_from"]).read_text())["train_trials"]
    sess = wairagkar_data.load_session(resolve_path(cfg["sanity"]["data_dir"]), cfg["sanity"]["session"], 0, cfg)
    T = {i: np.asarray(sess.trials[i].target, np.float64) for i in train}
    allf = np.concatenate(list(T.values()))
    mu, sd = allf.mean(0), np.where(allf.std(0) > 1e-9, allf.std(0), 1.0)
    Z = {i: (t - mu) / sd for i, t in T.items()}
    rng = np.random.default_rng(int(d["seed"]))
    Q = np.linalg.qr(rng.normal(size=(20, 20)))[0]
    noise = {i: gaussian_filter1d(rng.normal(size=z.shape), float(d["noise_smooth_frames"]), axis=0) for i, z in Z.items()}

    def make(kind, a):
        out = {}
        if kind == "mix":
            M = (1 - a) * np.eye(20) + a * Q
            zz = {i: z @ M for i, z in Z.items()}
            cat = np.concatenate(list(zz.values()))
            m2, s2 = cat.mean(0), np.where(cat.std(0) > 1e-9, cat.std(0), 1.0)
            for i, z in zz.items():
                out[i] = ((z - m2) / s2 * sd + mu).astype(np.float32)
        else:
            for i, z in Z.items():
                out[i] = ((z + a * noise[i]) * sd + mu).astype(np.float32)
        return out

    def mean_r(out):
        return float(np.mean([kx.feature_scores(out[i], T[i])["feat_corr_mean"] for i in train]))

    out_dir = resolve_path(d["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"name": d["name"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
              "code_version": code_version(), "n_train": len(train), "variants": {}}
    for v in d["variants"]:
        kind, goal = v["kind"], float(v["r"])
        lo, hi = 0.0, float(v["upper"])
        for _ in range(40):                                   # r falls monotonically with the parameter here
            mid = (lo + hi) / 2
            if mean_r(make(kind, mid)) > goal:
                lo = mid
            else:
                hi = mid
        a = (lo + hi) / 2
        out = make(kind, a)
        r = mean_r(out)
        name = v["name"]
        header = {"variant": name, "kind": kind, "parameter": a, "achieved_mean_r": r, "goal_r": goal,
                  "decision": "D-072", "seed": int(d["seed"]), "code_version": code_version()}
        path = out_dir / f"day195_{name}.npz"
        tmp = path.with_name(path.stem + ".partial.npz")
        np.savez_compressed(tmp, header=np.array(json.dumps(header)), **{f"tr{i:03d}": out[i] for i in train})
        tmp.replace(path)
        report["variants"][name] = {k: header[k] for k in ("kind", "parameter", "achieved_mean_r", "goal_r")}
        log(f"{name}: parameter {a:.4f}, mean r vs real {r:.3f} -> {path}")
    res = resolve_path("results/target_sensitivity")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{d['name']}.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
