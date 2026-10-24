#!/usr/bin/env python
"""Word identification for the Kunz isolated verbal behaviors (D-079).

    any env with NumPy, CPU:
    .venv/bin/python -I scripts/identify_iso.py --config configs/experiments/b2v2_iso_base.yaml \
        --preds b2v2_iso_aall:listening b2v2_iso_aall:reading ... --name aall

For every word trial (DO_NOTHING excluded) in the prediction files (chunked decodes: ``g<gid>_lp50``): the CTC
log-likelihood of each of the seven words (word + SIL) over the trial; the true word's rank (1 = best; ties half).
Per behavior: top-1 (chance 1/7), mean rank (chance 4.0), the confusion counts, and a permutation null (word labels
shuffled across that behavior's trials, ``--perms`` times; one-sided p for top-1 and mean rank). Writes
results/b2v2/iso/<name>.json and .md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="RUN:NAME of data/processed/b2v2_preds/RUN/NAME.npz")
    ap.add_argument("--name", required=True)
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np

    from wvx import b2v2_data as bd, phonemes as ph
    from wvx.provenance import code_version

    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    words = sorted({t["text"] for t in data.trials if t["text"]})
    labels = {w: next(t["label"] for t in data.trials if t["text"] == w) for w in words}
    rows = []
    for spec in args.preds:
        run, name = spec.split(":")
        with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False) as z:
            for g in json.loads(str(z["header"]))["gids"]:
                t = data.by_gid[g]
                if not t["text"]:
                    continue
                lp = z[f"g{g}_lp50"].astype(np.float64)
                ll = [ph.ctc_log_likelihood(lp, labels[w]) for w in words]
                rows.append({"gid": g, "behavior": t["session"], "block": t["block"], "word": t["text"],
                             "run": run, "pred": name, "ll": ll, "best": words[int(np.argmax(ll))]})
    rng = np.random.default_rng(args.seed)
    summary = {}
    for beh in sorted({r["behavior"] for r in rows}):
        rs = [r for r in rows if r["behavior"] == beh]
        L = np.array([r["ll"] for r in rs])                                   # trials x words
        y = np.array([words.index(r["word"]) for r in rs])
        order = (-L).argsort(1)
        rank_of = np.empty_like(order)
        rank_of[np.arange(len(L))[:, None], order] = np.arange(1, len(words) + 1)   # rank of each word per trial

        def stats(lab):
            r = rank_of[np.arange(len(L)), lab]
            return float(np.mean(r == 1)), float(np.mean(r))

        top1, mrank = stats(y)
        perm = np.array([stats(rng.permutation(y)) for _ in range(args.perms)])
        conf = {w: {v: int(np.sum((y == i) & (np.argmax(L, 1) == j))) for j, v in enumerate(words)}
                for i, w in enumerate(words)}
        summary[beh] = {"n": len(rs), "top1": top1, "mean_rank": mrank,
                        "null_top1_mean": float(perm[:, 0].mean()), "null_mean_rank_mean": float(perm[:, 1].mean()),
                        "p_top1": float((1 + np.sum(perm[:, 0] >= top1)) / (1 + args.perms)),
                        "p_mean_rank": float((1 + np.sum(perm[:, 1] <= mrank)) / (1 + args.perms)),
                        "runs": sorted({r["run"] for r in rs}), "confusion": conf}
    out = {"name": args.name, "decision": "D-079", "preds": args.preds, "words": words, "perms": args.perms,
           "code_version": code_version(), "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "summary": summary, "rows": rows}
    res = resolve_path("results/b2v2/iso")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{args.name}.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [f"# Word identification `{args.name}` (D-079): 7 words, chance top-1 14.3%, mean rank 4.0", "",
             "| behavior | n | top-1 | p (perm) | mean rank | p (perm) | null top-1 |", "|---|---|---|---|---|---|---|"]
    for beh, v in summary.items():
        lines.append(f"| {beh} | {v['n']} | {100 * v['top1']:.1f}% | {v['p_top1']:.2g} | {v['mean_rank']:.2f} | "
                     f"{v['p_mean_rank']:.2g} | {100 * v['null_top1_mean']:.1f}% |")
    (res / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
