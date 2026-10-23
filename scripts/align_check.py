#!/usr/bin/env python
"""Validation of target-free alignment (D-063, wvx.neural_align) on sessions OTHER than the dev test session.

Host, project venv, CPU only:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/align_check.py --config configs/experiments/align_check.yaml

One released model (its day layer belongs to ``ref_session``). For every session in ``sessions`` (later sessions,
never t15_day195): the trials are split in two seeded halves; each variant is FITTED on the neural data of the fit
half only (no targets) and EVALUATED on the other half: feature r vs their targets per decoded trial, paired with
zero-shot (Wilcoxon), and the embedding Frechet distance to the reference session after alignment.

Positive control (``control``): the reference session itself, halves likewise, with a known synthetic drift applied
to both halves (per-channel gain and offset, plus channel mixing); the undrifted score is the ceiling. Variants that
cannot undo a drift they are designed for, or that hurt an undrifted session, are broken.
Writes results/align_check/<name>.json and .md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[align_check {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/align_check.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    from wvx import b2v

    b2v.use_keras2()
    import numpy as np

    from drift_check import embed, frechet
    from wvx import inference, kunz_experiment as kx, neural_align as na, session_layers as sl, wairagkar_data
    from wvx.provenance import code_version

    a = cfg["align_check"]
    seed = int(a["seed"])
    data_dir = resolve_path(cfg["sanity"]["data_dir"])
    path = resolve_path(a["model"])
    day_layer, _ = sl.read_inference_h5(path)
    steps = int(day_layer.input_shape[1])
    model = inference.load_model(path, cfg)
    w_orig = [np.array(w) for w in day_layer.get_weights()]
    t0 = time.time()

    ref = wairagkar_data.load_session(data_dir, a["ref_session"], 0, cfg)
    ref_x = [kx.prepare_neural(t.neural, cfg) for t in ref.trials]
    log(f"reference {a['ref_session']}: {len(ref_x)} trials")

    def halves(n, key):
        p = np.random.default_rng([seed, key]).permutation(n)
        return sorted(p[: n // 2].tolist()), sorted(p[n // 2:].tolist())

    def run(label, xs, targets, fit_idx, ev_idx, ref_frames, ceiling=None):
        """Fit every variant on xs[fit_idx] (neural only), score on xs[ev_idx]."""
        ev = sorted(np.random.default_rng([seed, 7]).choice(ev_idx, size=min(int(a["n_eval"]), len(ev_idx)),
                                                            replace=False).tolist())
        fit_frames = na.frames([xs[i] for i in fit_idx])
        ev_frames = na.frames([xs[i] for i in ev])
        z_ref = embed(day_layer, ref_frames, steps)
        out = {"label": label, "n_fit": len(fit_idx), "n_eval": len(ev), "variants": {}}
        maps = {"zero_shot": None}
        for v in a["variants"]:
            if v["kind"] == "standardize":
                maps[v["name"]] = na.fit_standardize(fit_frames, ref_frames)
            elif v["kind"] == "coral":
                maps[v["name"]] = na.fit_coral(fit_frames, ref_frames, float(v["shrink"]))
            elif v["kind"] == "mmd":
                w, info = na.fit_mmd_layer(day_layer, fit_frames.astype(np.float32), ref_frames.astype(np.float32),
                                           v, log=None)
                maps[v["name"]] = ("layer", w, info)
        base = None
        for name, m in maps.items():
            if isinstance(m, tuple):
                na.set_day_layer(model, m[1])
                day_layer.set_weights(m[1])
                xin = lambda x: x  # noqa: E731
                minfo = {k: v for k, v in m[2].items() if k != "history"} | {"mmd2_first_last": [m[2]["history"][0]["mmd2"], m[2]["history"][-1]["mmd2"]]}
            else:
                xin = (lambda x: x) if m is None else m
                minfo = {} if m is None else m.info()
            fr = []
            for i in ev:
                pred = kx.decode(model, np.ascontiguousarray(xin(xs[i]), dtype=np.float32), cfg)
                fr.append(kx.feature_scores(pred, inference.align_to_predictions(targets[i], len(pred)))["feat_corr_mean"])
            z_ev = embed(day_layer, na.frames([xin(xs[i]) for i in ev]) if m is not None and not isinstance(m, tuple)
                         else ev_frames, steps)
            row = {"feat_r": float(np.mean(fr)), "per_trial": fr, "z_frechet_eval": frechet(z_ev, z_ref), **minfo}
            if base is None:
                base = fr
            else:
                row["vs_zero_shot"] = kx.paired(fr, base)
            if ceiling is not None:
                row["ceiling_feat_r"] = ceiling
            out["variants"][name] = row
            na.set_day_layer(model, w_orig)
            day_layer.set_weights(w_orig)
            log(f"{label}: {name:14s} feat r {row['feat_r']:.3f}"
                + (f" ({100 * row['vs_zero_shot']['mean']:+.1f} pts r x100, p {row['vs_zero_shot']['wilcoxon_p']:.2g})"
                   if 'vs_zero_shot' in row else "") + f"; z Frechet {row['z_frechet_eval']:.2f}")
        return out

    results = []
    # positive control: synthetic drift on the reference session itself
    c = a["control"]
    r = np.random.default_rng([seed, 99])
    F = ref_x[0].shape[1]
    gain = np.exp(r.normal(0, float(c["log_gain_sd"]), F))
    sd = np.concatenate([x for x in ref_x]).std(0)
    offset = r.normal(0, float(c["offset_sd"]), F) * sd
    mix = np.eye(F) + float(c["mix_sd"]) * r.normal(size=(F, F)) / np.sqrt(F)
    drift = lambda x: ((np.asarray(x, np.float64) @ mix) * gain + offset).astype(np.float32)  # noqa: E731
    fi, ei = halves(len(ref_x), 0)
    ref_targets = [t.target for t in ref.trials]
    ref_frames_fit = na.frames([ref_x[i] for i in fi])  # the "reference" = the fit half, undrifted
    und = run(f"{a['ref_session']} undrifted (identity check)", ref_x, ref_targets, ei, ei, ref_frames_fit)
    ceil = und["variants"]["zero_shot"]["feat_r"]
    results.append(und)
    results.append(run(f"{a['ref_session']} + synthetic drift", [drift(x) for x in ref_x], ref_targets, ei, ei,
                       ref_frames_fit, ceiling=ceil))
    ref_frames_all = na.frames(ref_x)
    del ref
    for s in a["sessions"]:
        if s == a.get("dev_session"):
            raise SystemExit(f"{s} is the dev test session; it must not be used here")
        sess = wairagkar_data.load_session(data_dir, s, 0, cfg)
        xs = [kx.prepare_neural(t.neural, cfg) for t in sess.trials]
        fi, ei = halves(len(xs), int(s.split("day")[1]))
        results.append(run(s, xs, [t.target for t in sess.trials], fi, ei, ref_frames_all))
        del sess, xs
    out = {"name": a["name"], "config": args.config, "model": path.name, "ref_session": a["ref_session"],
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
           "settings": a, "results": results, "wall_time_s": time.time() - t0}
    res = resolve_path("results/align_check")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{a['name']}.json").write_text(json.dumps(kx.portable(out), indent=1, default=float) + "\n")
    names = list(results[0]["variants"])
    lines = [f"# Alignment check `{a['name']}` (D-063): feature r on held-out halves", "",
             "| session | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for rr in results:
        cells = []
        for n in names:
            v = rr["variants"][n]
            d = v.get("vs_zero_shot")
            cells.append(f"{v['feat_r']:.3f}" + (f" ({100 * d['mean']:+.1f}, p {d['wilcoxon_p']:.2g})" if d else ""))
        lines.append(f"| {rr['label']} | " + " | ".join(cells) + " |")
    (res / f"{a['name']}.md").write_text("\n".join(lines) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/align_check/{a['name']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
