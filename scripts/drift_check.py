#!/usr/bin/env python
"""Premise check for target-free adaptation (D-063): does the distance between a session's neural data and a
decoder's own session, measured where the decoder's frozen core reads it, predict how well the decoder works there?

Host, project venv, CPU only:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/drift_check.py --config configs/experiments/drift_check.yaml

Each released Wairagkar model (``drift_check.models``: inference .h5 = avg-pool -> ONE day layer -> base) keeps the
day layer of its own last training session (``ref_session``; their ``save_inference_model(day_embed_layer=
session_layers[-1])``). For every later session in ``sessions`` (never the model's training data), and for the
reference session itself (split in two halves of its trials: the noise floor):

- raw input (512 features after channel reorder, pooled in pairs of bins as the model does): per-channel mean shift
  and log s.d. ratio vs the reference session; Frechet distance of the Gaussian fits;
- embeddings z = day_layer(pooled input) (128 per step, what the frozen core reads): Frechet distance, MMD^2
  (Gaussian kernel, bandwidth = median distance within the reference sample), share of all-zero (ReLU) units;
- decoding: the model's mean per-trial feature r (and MSE) vs their targets on ``n_decode`` seeded trials.

t15_day195 contributes ONLY the dev training trials (``train_trials_from``); its test trials are never read here.
Writes results/drift_check/<name>.json and .md (no trial data).
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


def log(msg: str) -> None:
    print(f"[drift_check {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def pooled(x):
    """(T, F) -> (T // 2, F): the model's avg_pool1d(ksize=2, strides=2) applied along a whole trial."""
    import numpy as np

    n = (x.shape[0] // 2) * 2
    return x[:n].reshape(-1, 2, x.shape[1]).mean(axis=1)


def embed(day_layer, xp, steps: int):
    """Pooled input (N, 512) -> day-layer embeddings (N', 128), fed in chunks of the layer's ``steps`` time steps."""
    import numpy as np

    n = (xp.shape[0] // steps) * steps
    if n == 0:
        return np.zeros((0, int(day_layer.output_shape[-1])), np.float32)
    z = day_layer(xp[:n].reshape(-1, steps, xp.shape[1]).astype(np.float32), training=False).numpy()
    return z.reshape(-1, z.shape[-1])


def frechet(a, b) -> float:
    import numpy as np

    mu = a.mean(0) - b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # Tr((ca cb)^(1/2)) = sum of the square roots of the eigenvalues of ca cb (real, >= 0 for PSD ca, cb)
    ev = np.clip(np.real(np.linalg.eigvals(ca @ cb)), 0, None)
    return float(mu @ mu + np.trace(ca) + np.trace(cb) - 2 * np.sqrt(ev).sum())


def mmd2(a, b, sigma: float) -> float:
    """Unbiased MMD^2 with a Gaussian kernel exp(-|x-y|^2 / (2 sigma^2))."""
    import numpy as np

    def k(x, y):
        d = (x * x).sum(1)[:, None] + (y * y).sum(1)[None] - 2 * x @ y.T
        return np.exp(-np.maximum(d, 0) / (2 * sigma ** 2))

    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    n, m = len(a), len(b)
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
                 - 2 * kab.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/drift_check.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    from wvx import b2v

    b2v.use_keras2()
    import numpy as np
    from scipy.stats import spearmanr

    from wvx import inference, kunz_experiment as kx, session_layers as sl, wairagkar_data
    from wvx.provenance import code_version

    d = cfg["drift_check"]
    rng_seed = int(d["seed"])
    data_dir = resolve_path(cfg["sanity"]["data_dir"])
    dev_train = set(json.loads(resolve_path(d["train_trials_from"]).read_text())["train_trials"])
    t0 = time.time()
    cache = {}

    def session(name):
        if name not in cache:
            sess = wairagkar_data.load_session(data_dir, name, 0, cfg)
            idx = [i for i in range(len(sess.trials)) if name != d["dev_session"] or i in dev_train]
            cache[name] = (sess, idx)
            log(f"{name}: {len(idx)} trials used (of {len(sess.trials)})")
        return cache[name]

    def raw_pooled(name, idx):
        sess, _ = session(name)
        return [pooled(kx.prepare_neural(sess.trials[i].neural, cfg)) for i in idx]

    def sub(x, n, key):
        r = np.random.default_rng([rng_seed, key])
        return x[r.choice(len(x), size=min(n, len(x)), replace=False)]

    rows = []
    for m in d["models"]:
        path = resolve_path(m["path"])
        day_layer, _ = sl.read_inference_h5(path)
        steps = int(day_layer.input_shape[1])
        model = inference.load_model(path, cfg)
        ref = m["ref_session"]
        _, ref_idx = session(ref)
        half = [ref_idx[0::2], ref_idx[1::2]]
        ref_raw = [np.concatenate(raw_pooled(ref, h)) for h in half]
        ref_z = [embed(day_layer, x, steps) for x in ref_raw]
        ref_all_raw, ref_all_z = np.concatenate(ref_raw), np.concatenate(ref_z)
        zs_ref = sub(ref_all_z, int(d["n_mmd"]), 1)
        dd = np.sqrt(np.maximum((zs_ref * zs_ref).sum(1)[:, None] + (zs_ref * zs_ref).sum(1)[None]
                                - 2 * zs_ref @ zs_ref.T, 0))
        sigma = float(np.median(dd[np.triu_indices(len(zs_ref), 1)]))
        mu_r, sd_r = ref_all_raw.mean(0), ref_all_raw.std(0)
        targets = [(ref + " (half B vs half A)", ref, half[1], ref_raw[1], ref_z[1], ref_raw[0], ref_z[0])]
        for s in m["sessions"]:
            _, idx = session(s)
            xr = np.concatenate(raw_pooled(s, idx))
            targets.append((s, s, idx, xr, embed(day_layer, xr, steps), ref_all_raw, ref_all_z))
        for label, s, idx, xr, z, rr, rz in targets:
            sess, _ = session(s)
            dec_idx = sorted(np.random.default_rng([rng_seed, int(s.split("day")[1])]).choice(
                idx, size=min(int(d["n_decode"]), len(idx)), replace=False).tolist())
            fr, mse = [], []
            for i in dec_idx:
                t = sess.trials[i]
                pred = kx.decode(model, kx.prepare_neural(t.neural, cfg), cfg)
                sc = kx.feature_scores(pred, inference.align_to_predictions(t.target, len(pred)))
                fr.append(sc["feat_corr_mean"]), mse.append(sc["feat_mse"])
            gap = int(s.split("day")[1]) - int(ref.split("day")[1])
            row = {"model": path.name, "ref_session": ref, "session": label, "gap_days": gap, "n_trials": len(idx),
                   "raw_mean_shift": float(np.mean(np.abs(xr.mean(0) - rr.mean(0)))),
                   "raw_log_sd_ratio": float(np.median(np.abs(np.log(xr.std(0) / rr.std(0))))),
                   "raw_frechet": frechet(xr, rr), "z_frechet": frechet(z, rz),
                   "z_mmd2": mmd2(sub(z, int(d["n_mmd"]), 2), sub(rz, int(d["n_mmd"]), 3), sigma),
                   "z_dead_fraction": float((z.max(0) <= 0).mean()), "mmd_sigma": sigma,
                   "n_decoded": len(dec_idx), "feat_r": float(np.nanmean(fr)), "feat_mse": float(np.mean(mse)),
                   "raw_channel_mean_abs": float(np.mean(np.abs(xr.mean(0)))), "raw_channel_sd_median": float(np.median(xr.std(0)))}
            rows.append(row)
            log(f"{path.name} on {label}: gap {gap} d, z Frechet {row['z_frechet']:.2f}, MMD2 {row['z_mmd2']:.4f}, "
                f"raw Frechet {row['raw_frechet']:.1f}, mean shift {row['raw_mean_shift']:.3f}, feat r {row['feat_r']:.3f}")
    # premise: across held-out (model, session) pairs, distance vs feature r, and with the gap partialled out
    held = [r for r in rows if "half" not in r["session"]]
    stats = {}
    for k in ("z_frechet", "z_mmd2", "raw_frechet", "raw_mean_shift", "gap_days"):
        rho = spearmanr([r[k] for r in held], [r["feat_r"] for r in held])
        stats[k] = {"spearman_vs_feat_r": float(rho.statistic), "p": float(rho.pvalue)}
    gaps = np.array([r["gap_days"] for r in held], float)

    def resid(v):
        v = np.asarray(v, float)
        A = np.vstack([gaps, np.ones_like(gaps)]).T
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]

    for k in ("z_frechet", "z_mmd2", "raw_frechet"):
        rho = spearmanr(resid([r[k] for r in held]), resid([r["feat_r"] for r in held]))
        stats[k]["partial_given_gap"] = {"spearman": float(rho.statistic), "p": float(rho.pvalue)}
    out = {"name": d["name"], "config": args.config, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "code_version": code_version(), "rows": rows, "premise": stats, "n_pairs": len(held),
           "wall_time_s": time.time() - t0}
    res = resolve_path("results/drift_check")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{d['name']}.json").write_text(json.dumps(kx.portable(out), indent=1, default=float) + "\n")
    lines = [f"# Drift check `{d['name']}` (D-063)", "",
             "| model | session | gap (d) | z Frechet | z MMD² | raw Frechet | raw mean shift | feat r |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['session']} | {r['gap_days']} | {r['z_frechet']:.2f} | {r['z_mmd2']:.4f} | "
              f"{r['raw_frechet']:.1f} | {r['raw_mean_shift']:.3f} | {r['feat_r']:.3f} |" for r in rows]
    lines += ["", "Spearman vs feature r over held-out pairs (n = %d):" % len(held), ""]
    lines += [f"- {k}: rho {v['spearman_vs_feat_r']:+.2f} (p {v['p']:.2g})"
              + (f"; given the gap: rho {v['partial_given_gap']['spearman']:+.2f} (p {v['partial_given_gap']['p']:.2g})"
                 if "partial_given_gap" in v else "") for k, v in stats.items()]
    (res / f"{d['name']}.md").write_text("\n".join(lines) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/drift_check/{d['name']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
