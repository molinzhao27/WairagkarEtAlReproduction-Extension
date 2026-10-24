#!/usr/bin/env python
"""D-082: does the decoder's phoneme stream carry T15's silent counting?

    any env with NumPy/SciPy, CPU:
    .venv/bin/python -I scripts/count_analysis.py --config configs/experiments/b2v2_count_base.yaml \
        --go RUN:count_go --delay RUN:count_delay --control-config configs/experiments/b2v2_ov_base.yaml \
        --control RUN:d1215_eval_name --name NAME

A (sanity, the count said aloud): CTC log-likelihood of each number word ten..twenty (word + SIL) over the go-epoch
stream; top-1 and mean rank of the true count; null: counts permuted across trials.
B (primary, silent counting): per delay epoch, d = LL("one two ... n", in order) - mean LL(the same words in
``--scrambles`` random orders); also d_rev = LL(in order) - LL(reversed). Same words, same phonemes, same trial: only
the order differs. Wilcoxon one-sided (d > 0).
C (control): the same d on same-day inner-speech sentences decoded by the same model (no counting), each given an n
drawn from the counting trials; B vs C Mann-Whitney one-sided. Writes results/b2v2/count/<name>.json and .md.
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
    ap.add_argument("--go", required=True)
    ap.add_argument("--delay", required=True)
    ap.add_argument("--control-config", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--scrambles", type=int, default=50)
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args(argv)
    import numpy as np
    from scipy.stats import mannwhitneyu, wilcoxon

    from wvx import b2v2_data as bd, phonemes as ph
    from wvx.provenance import code_version

    cfg = load_experiment_config(args.config)
    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    nums = data.meta["numbers"]
    words = list(nums)                                                   # one .. twenty, in order
    rng = np.random.default_rng(args.seed)

    def load(spec):
        run, name = spec.split(":")
        with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False) as z:
            gids = json.loads(str(z["header"]))["gids"]
            return {g: z[f"g{g}_lp50"].astype(np.float64) for g in gids}

    def seq(ws):
        return ph.label([nums[w] for w in ws])

    # A: the spoken count
    go = load(args.go)
    cands = words[9:]                                                    # ten .. twenty
    L = np.array([[ph.ctc_log_likelihood(go[g], seq([w])) for w in cands] for g in go])
    y = np.array([cands.index(data.by_gid[g]["text"]) for g in go])
    order = (-L).argsort(1)
    rank_of = np.empty_like(order)
    rank_of[np.arange(len(L))[:, None], order] = np.arange(1, len(cands) + 1)

    def stats(lab):
        r = rank_of[np.arange(len(L)), lab]
        return float(np.mean(r == 1)), float(np.mean(r))

    top1, mrank = stats(y)
    perm = np.array([stats(rng.permutation(y)) for _ in range(args.perms)])
    A = {"n": len(y), "top1": top1, "mean_rank": mrank, "chance_top1": 1 / len(cands),
         "p_top1": float((1 + np.sum(perm[:, 0] >= top1)) / (1 + args.perms)),
         "p_mean_rank": float((1 + np.sum(perm[:, 1] <= mrank)) / (1 + args.perms)),
         "predicted": [cands[i] for i in L.argmax(1)], "true": [cands[i] for i in y]}

    def order_effect(lp, n):
        ws = words[:n]
        ll = ph.ctc_log_likelihood(lp, seq(ws))
        scr = []
        while len(scr) < args.scrambles:
            p = rng.permutation(n)
            if np.any(p != np.arange(n)):
                scr.append(ph.ctc_log_likelihood(lp, seq([ws[i] for i in p])))
        return ll - float(np.mean(scr)), ll - ph.ctc_log_likelihood(lp, seq(ws[::-1]))

    # B: silent counting
    delay = load(args.delay)
    B = [dict(zip(("d", "d_rev"), order_effect(delay[g], data.by_gid[g]["n"])), gid=g, n=data.by_gid[g]["n"])
         for g in delay]
    # C: control, same-day inner-speech sentences, same model
    ctrl = load(args.control)
    ns = [r["n"] for r in B]
    C = [dict(zip(("d", "d_rev"), order_effect(lp, int(ns[i % len(ns)]))), gid=g, n=int(ns[i % len(ns)]))
         for i, (g, lp) in enumerate(sorted(ctrl.items()))]
    dB, dC = np.array([r["d"] for r in B]), np.array([r["d"] for r in C])
    summary = {
        "A_spoken_count": {k: v for k, v in A.items() if k not in ("predicted", "true")},
        "B_silent_counting": {"n": len(dB), "mean_d": float(dB.mean()), "median_d": float(np.median(dB)),
                              "frac_positive": float(np.mean(dB > 0)),
                              "p_wilcoxon_greater": float(wilcoxon(dB, alternative="greater").pvalue),
                              "mean_d_rev": float(np.mean([r["d_rev"] for r in B])),
                              "p_rev_greater": float(wilcoxon([r["d_rev"] for r in B], alternative="greater").pvalue)},
        "C_control": {"n": len(dC), "mean_d": float(dC.mean()), "median_d": float(np.median(dC)),
                      "p_wilcoxon_greater": float(wilcoxon(dC, alternative="greater").pvalue)},
        "B_vs_C_mannwhitney_greater": float(mannwhitneyu(dB, dC, alternative="greater").pvalue),
    }
    out = {"name": args.name, "decision": "D-082", "go": args.go, "delay": args.delay, "control": args.control,
           "scrambles": args.scrambles, "perms": args.perms, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "summary": summary, "A": A, "B": B,
           "C": C}
    res = resolve_path("results/b2v2/count")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{args.name}.json").write_text(json.dumps(out, indent=1) + "\n")
    a, bb, c = summary["A_spoken_count"], summary["B_silent_counting"], summary["C_control"]
    lines = [f"# D-082 `{args.name}`", "",
             f"- A, spoken count (go epoch, 11 numbers, chance {100 * a['chance_top1']:.1f}%): top-1 {100 * a['top1']:.1f}% "
             f"(p {a['p_top1']:.2g}), mean rank {a['mean_rank']:.2f} (p {a['p_mean_rank']:.2g}).",
             f"- B, silent counting (in-order minus scrambled log-likelihood): mean {bb['mean_d']:+.2f}, median "
             f"{bb['median_d']:+.2f}, {100 * bb['frac_positive']:.0f}% of {bb['n']} trials positive, p {bb['p_wilcoxon_greater']:.2g}; "
             f"in-order minus reversed {bb['mean_d_rev']:+.2f} (p {bb['p_rev_greater']:.2g}).",
             f"- C, control (inner-speech sentences, no counting): mean {c['mean_d']:+.2f}, median {c['median_d']:+.2f} "
             f"(n {c['n']}, p {c['p_wilcoxon_greater']:.2g}); B vs C p {summary['B_vs_C_mannwhitney_greater']:.2g}."]
    (res / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
