#!/usr/bin/env python
"""Converted copies of the Kunz imagined data for the full TTS-referenced scoring (D-069).

Host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/inner_transform_data.py --config configs/experiments/inner_transform_data.yaml

For each transform (``wvx.inner_conversion``): every imagined block of a pair is converted by the transform fitted on
the OTHER three pairs (as in D-067), and the novel-cue eval block (E14) by the transform fitted on all four pairs.
Writes <out_dir>/<session>_imagined_<transform>.npz and <session>_imagined_eval_<transform>.npz (stored channel
order; the pipeline reorders as usual) and results/inner_transform_data/<name>.json (fit info, checks).

Check: every transform fitted in the stored channel order and then reordered equals the one fitted after
reordering (the order D-067 used), on one trial per transform and fold.
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
    print(f"[inner_transform_data {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/inner_transform_data.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import dataclasses

    import numpy as np

    from wvx import inner_conversion as ic, kunz_experiment as kx
    from wvx.inference import reorder_channels
    from wvx.provenance import code_version
    from wvx.session import SessionData

    c = cfg["inner_transform_data"]
    t0 = time.time()
    proc = resolve_path(cfg["paths"]["kunz_processed"])
    name = cfg["experiment"]["session"]
    att = SessionData.load(proc / f"{name}_attempted.npz")
    im = SessionData.load(proc / f"{name}_imagined.npz")
    ev = SessionData.load(proc / f"{name}_imagined_eval.npz")
    pairs = {int(a): int(i) for a, i in c["pairs"].items()}

    def blocks(sess):
        out = {}
        for t in sess.trials:
            out.setdefault(int(t.block), []).append(t)
        return out

    A, I = blocks(att), blocks(im)
    mapping = cfg["paths"]["electrode_mapping"]

    def reordered(trials):
        return [dataclasses.replace(t, neural=reorder_channels(t.neural, mapping).astype(np.float32)) for t in trials]

    out_dir = resolve_path(c["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"name": c["name"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
              "code_version": code_version(), "settings": c, "folds": {}, "eval": {}, "checks": []}
    converted = {n: {} for n in c["transforms"]}
    for a_out, i_out in list(pairs.items()) + [(None, None)]:
        fit_pairs = [(A[a], I[i]) for a, i in pairs.items() if a != a_out]
        key = a_out if a_out is not None else 0
        maps, info = ic.fit(fit_pairs, c, key, c["transforms"])
        # check: fitting in the stored order, then reordering == reordering, then fitting (D-067's order)
        maps_r, _ = ic.fit([(reordered(x), reordered(y)) for x, y in fit_pairs], c, key, c["transforms"])
        probe = I[i_out][0] if i_out is not None else ev.trials[0]
        for n in c["transforms"]:
            lhs = reorder_channels(maps[n](probe.neural), mapping)
            rhs = maps_r[n](reorder_channels(probe.neural, mapping))
            d = float(np.max(np.abs(lhs - rhs)))
            report["checks"].append({"fold": f"A{a_out}" if a_out else "all", "transform": n, "max_abs_diff": d})
            if d > float(c["check_tol"]):
                raise SystemExit(f"{n}: stored-order and reordered fits differ by {d} (fold {a_out})")
        target = I[i_out] if i_out is not None else ev.trials
        for n in c["transforms"]:
            for t in target:
                converted[n][id(t)] = maps[n](t.neural).astype(np.float32)
        slot = report["folds"].setdefault(f"A{a_out}/I{i_out}", {}) if a_out else report["eval"]
        slot.update({n: {k: v for k, v in info.get(n, {}).items() if k in ("scale", "mean_diff_share_in_subspace",
                                                                          "kind", "n_pairs")}
                     | ({"canonical_r_top5": info[n]["canonical_correlations"][:5]} if "canonical_correlations"
                        in info.get(n, {}) else {}) for n in c["transforms"]})
        log(f"fitted {'without A%d/I%d' % (a_out, i_out) if a_out else 'on all four pairs'}: "
            + ", ".join(f"{n} ok" for n in c["transforms"]))
    for n in c["transforms"]:
        for sess, suffix in ((im, "imagined"), (ev, "imagined_eval")):
            trials = [dataclasses.replace(t, neural=converted[n][id(t)],
                                          meta=dict(t.meta, inner_conversion={"transform": n, "decision": "D-069"}))
                      for t in sess.trials]
            out = SessionData(sess.name, sess.layer, trials)
            out.validate(n_features=int(cfg["model"]["n_features"]), n_targets=int(cfg["model"]["n_targets"]))
            path = out_dir / f"{name}_{suffix}_{n}.npz"
            out.save(path)
            log(f"{n}: {len(trials)} {suffix} trials -> {path}")
    report["max_check_diff"] = max(x["max_abs_diff"] for x in report["checks"])
    report["wall_time_s"] = time.time() - t0
    res = resolve_path("results/inner_transform_data")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{c['name']}.json").write_text(json.dumps(kx.portable(report), indent=1, default=float) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min; order check max |diff| {report['max_check_diff']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
