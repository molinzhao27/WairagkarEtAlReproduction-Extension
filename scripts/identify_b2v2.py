#!/usr/bin/env python
"""Sentence identification and phoneme error rate from b2v2 phoneme outputs (D-076).

    any env with NumPy/SciPy, CPU:
    .venv/bin/python -I scripts/identify_b2v2.py --config configs/experiments/b2v2_kunz_base.yaml \
        --preds b2v2_kunz_f03:heldout b2v2_kunz_f06:heldout ... --name folds_attempted

For each trial in the prediction files (chunked decodes: ``g<gid>_lp50``, the phase-0 50 ms log-probabilities): the
CTC log-likelihood (``wvx.phonemes.ctc_log_likelihood``) of every sentence of its (session, block); the true
sentence's rank (1 = best; ties count half); the null rank: the true sentence's rank under the output of the trial of
the same block closest in length (another trial of the same prediction file, so the same model); the PER (greedy CTC
of ``g<gid>_best50``, SIL ignored). Writes results/b2v2/identify/<name>.json and .md (per session; Wilcoxon of true
vs null rank).
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


def rank_of(ll, true: int) -> float:
    others = [v for k, v in enumerate(ll) if k != true]
    return 1.0 + sum(v > ll[true] for v in others) + 0.5 * sum(v == ll[true] for v in others)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="for b2v2.data_dir")
    ap.add_argument("--preds", nargs="+", required=True, help="RUN:NAME of data/processed/b2v2_preds/RUN/NAME.npz")
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    from scipy.stats import wilcoxon

    from wvx import b2v2_data as bd, phonemes as ph
    from wvx.provenance import code_version

    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    blocks = {}
    for t in data.trials:
        if t["label"] is not None:
            blocks.setdefault((t["session"], t.get("block")), []).append(t)
    rows = []
    for spec in args.preds:
        run, name = spec.split(":")
        with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False) as z:
            gids = json.loads(str(z["header"]))["gids"]
            lp = {g: z[f"g{g}_lp50"].astype(np.float64) for g in gids}
            best = {g: z[f"g{g}_best50"] for g in gids}
        for g in gids:
            t = data.by_gid[g]
            if t["label"] is None:
                continue
            cands = blocks[(t["session"], t.get("block"))]
            true = next(k for k, c in enumerate(cands) if c["gid"] == g)
            ll = [ph.ctc_log_likelihood(lp[g], c["label"]) for c in cands]
            same = [x for x in cands if x["gid"] != g and x["gid"] in lp]
            row = {"gid": g, "session": t["session"], "block": t.get("block"), "text": t["text"], "run": run,
                   "pred": name, "n_candidates": len(cands), "rank": rank_of(ll, true)}
            if same:
                j = min(same, key=lambda x: (abs(x["length"] - t["length"]), x["gid"]))
                row["null_gid"] = j["gid"]
                row["null_rank"] = rank_of([ph.ctc_log_likelihood(lp[j["gid"]], c["label"]) for c in cands], true)
            e, n = ph.phoneme_error(t["label"], ph.ctc_greedy(best[g]))
            row.update(phone_errors=e, ref_phones=n)
            rows.append(row)
    summary = {}
    for s in sorted({r["session"] for r in rows}):
        rs = [r for r in rows if r["session"] == s]
        rk = np.array([r["rank"] for r in rs])
        v = {"n": len(rs), "n_candidates": int(np.median([r["n_candidates"] for r in rs])),
             "mean_rank": float(rk.mean()), "median_rank": float(np.median(rk)), "top1": float(np.mean(rk == 1)),
             "top5": float(np.mean(rk <= 5)),
             "per": sum(r["phone_errors"] for r in rs) / max(1, sum(r["ref_phones"] for r in rs))}
        pr = [r for r in rs if "null_rank" in r]
        if pr:
            d = np.array([r["rank"] - r["null_rank"] for r in pr])
            v.update(null_mean_rank=float(np.mean([r["null_rank"] for r in pr])), n_paired=len(pr),
                     true_minus_null=float(d.mean()), p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        summary[s] = v
    out = {"name": args.name, "decision": "D-076", "preds": args.preds, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "summary": summary, "rows": rows}
    res = resolve_path("results/b2v2/identify")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{args.name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = [f"# Identification `{args.name}` (D-076)", "", f"Predictions: {', '.join(args.preds)}", "",
             "| trials | n | candidates | mean rank | top-1 | top-5 | null mean rank | true − null (p) | PER |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s, v in summary.items():
        lines.append(f"| {s} | {v['n']} | {v['n_candidates']} | {v['mean_rank']:.1f} | {100 * v['top1']:.1f}% | "
                     f"{100 * v['top5']:.1f}% | " + (f"{v['null_mean_rank']:.1f} | {v['true_minus_null']:+.1f} "
                                                      f"({v['p']:.2g})" if "p" in v else "– | –") +
                     f" | {100 * v['per']:.1f}% |")
    (res / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
