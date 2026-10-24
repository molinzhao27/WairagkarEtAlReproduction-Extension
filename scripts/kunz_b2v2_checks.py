#!/usr/bin/env python
"""Post-hoc checks of the D-076 inner-speech identification (exploratory; run after the pre-registered tests).

    PC/host, project venv, CPU:
    .venv/bin/python -I scripts/kunz_b2v2_checks.py --config configs/experiments/b2v2_kunz_base.yaml

1. Leakage: the sentences each fold adapted on (split.json train + validation) vs its test blocks (held-out attempted
   block, paired imagined block); the all-blocks model vs block 14.
2. Length: identification of the imagined trials (fold models) among only the sentences of their block with the
   same number of words.
3. Order: the same identification with each trial's phoneme stream time-reversed (keeps every trial-level statistic,
   destroys the phoneme order).
Writes results/b2v2/identify/checks.json and .md.
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

FOLDS = {3: 4, 6: 7, 11: 9, 13: 12}          # held-out attempted block -> imagined block with the same sentences


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_kunz_base.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    from scipy.stats import wilcoxon

    from wvx import b2v2_data as bd, phonemes as ph
    from wvx.provenance import code_version

    d = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    res_b2v2 = resolve_path("results/b2v2")
    leak = {}
    for k, ib in FOLDS.items():
        sp = json.loads((res_b2v2 / f"b2v2_kunz_f{k:02d}" / "split.json").read_text())
        trained = {d.by_gid[g]["text"] for g in sp["train"] + sp["val"]}
        test = {t["text"] for t in d.trials if (t["session"], t["block"]) in (("attempted", k), ("imagined", ib))}
        leak[f"fold_{k}"] = {"adaptation_sentences": len(trained), "overlap_with_test": len(trained & test)}
    sp = json.loads((res_b2v2 / "b2v2_kunz_all" / "split.json").read_text())
    leak["all_vs_block14"] = len({d.by_gid[g]["text"] for g in sp["train"] + sp["val"]}
                                 & {t["text"] for t in d.trials if t["session"] == "imagined_eval"})

    def n_words(t):
        return len(t["text"].split())

    rows = []
    for k in FOLDS:
        with np.load(resolve_path("data/processed/b2v2_preds") / f"b2v2_kunz_f{k:02d}" / "imagined.npz") as z:
            gids = json.loads(str(z["header"]))["gids"]
            lps = {g: z[f"g{g}_lp50"].astype(np.float64) for g in gids}
        cands = [d.by_gid[g] for g in gids]
        for g in gids:
            t, lp = d.by_gid[g], lps[g]
            ll = {c["gid"]: ph.ctc_log_likelihood(lp, c["label"]) for c in cands}
            rev = {c["gid"]: ph.ctc_log_likelihood(lp[::-1], c["label"]) for c in cands}
            same = [c["gid"] for c in cands if n_words(c) == n_words(t)]
            rows.append({"gid": g, "fold": k,
                         "rank": 1 + sum(v > ll[g] for c, v in ll.items() if c != g),
                         "rank_reversed": 1 + sum(v > rev[g] for c, v in rev.items() if c != g),
                         "rank_same_words": 1 + sum(ll[c] > ll[g] for c in same if c != g),
                         "n_same_words": len(same)})
    r = np.array([x["rank"] for x in rows])
    rr = np.array([x["rank_reversed"] for x in rows])
    rs = np.array([x["rank_same_words"] for x in rows])
    ns = np.array([x["n_same_words"] for x in rows])
    m = ns > 1
    summary = {
        "leakage": leak,
        "same_word_count": {"n": int(m.sum()), "median_candidates": float(np.median(ns[m])),
                            "top1": float(np.mean(rs[m] == 1)), "chance_top1": float(np.mean(1 / ns[m])),
                            "rank_fraction": float(np.mean(rs[m] / ns[m])),
                            "chance_rank_fraction": float(np.mean((ns[m] + 1) / 2 / ns[m]))},
        "time_reversed": {"mean_rank": float(rr.mean()), "mean_rank_true": float(r.mean()),
                          "top1": float(np.mean(rr == 1)), "p_true_vs_reversed": float(wilcoxon(r - rr).pvalue)},
    }
    out = {"decision": "D-076 (post-hoc checks)", "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "summary": summary, "rows": rows}
    res = res_b2v2 / "identify"
    res.mkdir(parents=True, exist_ok=True)
    (res / "checks.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    s, w, tr = summary, summary["same_word_count"], summary["time_reversed"]
    lines = ["# D-076 post-hoc checks (imagined trials, fold models)", "",
             f"- Leakage: overlap of adaptation sentences with test blocks per fold "
             f"{[v['overlap_with_test'] for v in list(s['leakage'].values())[:4]]}; all-blocks model vs block 14: "
             f"{s['leakage']['all_vs_block14']}.",
             f"- Same word count only ({w['n']} trials, median {w['median_candidates']:.0f} candidates): top-1 "
             f"{100 * w['top1']:.1f}% (chance {100 * w['chance_top1']:.1f}%); rank fraction {w['rank_fraction']:.3f} "
             f"(chance {w['chance_rank_fraction']:.3f}).",
             f"- Time-reversed phoneme stream: mean rank {tr['mean_rank']:.1f} (true order {tr['mean_rank_true']:.1f}), "
             f"top-1 {100 * tr['top1']:.1f}%; paired p {tr['p_true_vs_reversed']:.2g}."]
    (res / "checks.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
