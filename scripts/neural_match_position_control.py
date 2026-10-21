#!/usr/bin/env python
"""Post-hoc control for the same-day neural matching (D-032; not pre-registered): is matching driven by trial POSITION?

Paired attempted/imagined blocks present the same sentences in the same order, so the own attempted trial is
also the same-position trial. Non-paired blocks share positions but not sentences: if slow drift within blocks
drove the match, same-position trials would match there too. For every (imagined block, attempted block)
combination this ranks the same-position attempted trial among all 50 (DTW trajectory similarity, the primary
measure; PCA fitted on the two pairs not involved) and reports mean similarity by |position difference|.

    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/neural_match_position_control.py \\
        --config configs/experiments/neural_match.yaml      ->  results/neural_match/position_control.{json,md}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402

DISTANCE_BINS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 9), (10, None))   # |position difference| groups (labels)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/neural_match.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    nice = cfg["runtime"].get("nice")
    if nice is not None and os.nice(0) < int(nice):
        os.nice(int(nice) - os.nice(0))
    sys.modules.setdefault("tensorflow", None)   # no TensorFlow needed (see scripts/neural_match.py)

    import numpy as np

    from wvx import neural_match as nm
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed
    from wvx.session import SessionData

    set_seed(int(cfg["seed"]))
    s = cfg["neural_match"]
    fmt = {k: str(v) for k, v in cfg["paths"].items()}
    sess = {c: SessionData.load(resolve_path(s[f"{c}_file"].format(**fmt))) for c in ("attempted", "imagined")}
    pool, drop, pc = int(s["pool_bins"]), bool(s["drop_partial_window"]), s["pca"]

    def blocks(ss):
        out: dict = {}
        for t in ss.trials:
            go = int(t.meta.get("go_offset", 0))
            out.setdefault(int(t.block), []).append((int(t.meta["trial_index"]), nm.pool_frames(t.neural[go:], pool, drop)))
        return {b: [x for _, x in sorted(v, key=lambda z: z[0])] for b, v in out.items()}

    A, I = blocks(sess["attempted"]), blocks(sess["imagined"])
    pairs = {int(a): int(i) for a, i in s["pairs"].items()}
    inv = {i: a for a, i in pairs.items()}
    combos, lines = [], []
    for ib in sorted(I):
        for ab in sorted(A):
            excl = {ab, inv[ib]}
            train = [x for b, xs in A.items() if b not in excl for x in xs]
            train += [x for b, xs in I.items() if inv[b] not in excl for x in xs]
            model = nm.fit_pca(np.concatenate(train), int(pc["n_components"]), bool(pc["standardize"]), float(pc["min_std"]))
            za, zi = [nm.apply_pca(model, x) for x in A[ab]], [nm.apply_pca(model, x) for x in I[ib]]
            S = np.array([[nm.dtw_similarity(x, y, s["dtw"]) for y in za] for x in zi])
            n = len(zi)
            ranks = np.array([nm.own_rank(S[r], np.arange(n), r)[0] for r in range(n)])
            d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
            by = {}
            for lo, hi in DISTANCE_BINS:
                m = (d >= lo) if hi is None else ((d >= lo) & (d <= hi))
                by[f"{lo}" if lo == hi else (f">={lo}" if hi is None else f"{lo}-{hi}")] = float(S[m].mean())
            rec = {"imagined_block": ib, "attempted_block": ab, "paired": pairs[ab] == ib, "n": n,
                   "same_position_mean_rank": float(ranks.mean()), "same_position_top1": float(np.mean(ranks == 1)),
                   "chance_mean_rank": (n + 1) / 2, "chance_top1": 1 / n, "mean_similarity_by_abs_position_diff": by}
            combos.append(rec)
            lines.append(f"| I{ib:02d} | A{ab:02d} | {'paired' if rec['paired'] else 'control'} | {ranks.mean():.2f} | "
                         f"{np.mean(ranks == 1):.2f} | " + " | ".join(f"{v:.3f}" for v in by.values()) + " |")
            print(lines[-1], flush=True)
    ctrl = [c for c in combos if not c["paired"]]
    summary = {"analysis": "neural_match_position_control", "post_hoc": True, "of": "notes/DECISIONS.md D-032",
               "combos": combos,
               "control_pooled": {"n_combos": len(ctrl),
                                  "mean_rank": float(np.mean([c["same_position_mean_rank"] for c in ctrl])),
                                  "top1": float(np.mean([c["same_position_top1"] for c in ctrl]))},
               "code_version": code_version(), "config_sha256": config_sha256(cfg)}
    out = resolve_path(s["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "position_control.json").write_text(json.dumps(summary, indent=1) + "\n")
    hdr = ("| Imagined | Attempted | Kind | Same-position mean rank (chance 25.5) | Top-1 (chance 0.02) | "
           + " | ".join(f"sim, abs dpos {k}" for k in combos[0]["mean_similarity_by_abs_position_diff"]) + " |")
    md = ["# Neural matching: position control (post hoc)", "", __doc__.split("\n\n")[1], "", hdr,
          "|" + "---|" * (5 + len(DISTANCE_BINS)), *lines, "",
          f"Control pooled (non-paired combos): mean rank {summary['control_pooled']['mean_rank']:.2f}, "
          f"top-1 {summary['control_pooled']['top1']:.3f} over {len(ctrl)} combos.", ""]
    (out / "position_control.md").write_text("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
