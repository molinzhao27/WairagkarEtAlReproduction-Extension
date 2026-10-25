#!/usr/bin/env python
"""D-086 summary: every arm on the same 200 held-out inner-speech trials (folds as D-077), against the baseline.

    PC, project venv, CPU:
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/d086_summary.py --config configs/experiments/b2v2_kunz_d086.yaml

Per arm (``b2v2.d086.arms``: ``preds`` = RUN:NAME decodes, ``identify`` = an identification result name, and either
``scores`` (the runs' ``scores_<NAME>.json``) or ``postproc`` {set, variant} for a post-processed variant, with
``boost: true`` when the movement measure should see the boosted features):
- Whisper WER, pooled (original speed; sped up secondary), per-trial paired Wilcoxon vs ``baseline`` and vs
  ``secondary_vs``; the pre-registered rule for each ``primary`` arm: pooled WER >= ``rule.min_points`` lower than
  the baseline with p < ``rule.alpha``;
- missing words: Whisper word deletions per reference word (immediate repeats in the transcript collapsed, D-084);
- movement inside words (D-084): sound islands (c0 smoothed over ``smooth`` frames above ``threshold``, at least
  ``min_frames``, gaps under ``bridge`` frames bridged); per island the s.d. over time of the cepstra ``cepstra``,
  in units of the attempted held-out decodes' (``reference``) per-feature island s.d.; median over islands;
- identification top-1 / mean rank and PER (from ``identify``).
Writes results/b2v2/d086/summary.json / .md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def words(s: str) -> list:
    return re.sub(r"[^a-z' ]", " ", (s or "").lower().replace("-", " ")).split()


def deletions(ref: list, hyp: list) -> int:
    """Word deletions in a minimum-edit alignment of ``hyp`` to ``ref``."""
    import numpy as np

    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0], D[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, dl = n, m, 0
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            dl, i = dl + 1, i - 1
        else:
            j -= 1
    return dl


def islands(c0, isl):
    import numpy as np

    k = int(isl["smooth"])
    on = np.convolve(c0, np.ones(k) / k, "same") > float(isl["threshold"])
    idx = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    out = []
    for s, e in zip(idx[::2], idx[1::2]):
        if out and s - out[-1][1] < int(isl["bridge"]):
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return [(s, e) for s, e in out if e - s >= int(isl["min_frames"])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    from scipy.stats import wilcoxon

    from wvx import postproc as pp
    from wvx.provenance import code_version

    S = cfg["b2v2"]["d086"]
    isl = S["islands"]
    lo, hi = (int(x) for x in isl["cepstra"])
    bo = cfg["b2v2"]["postproc"]["boost"]

    def conts(specs, boost=False):
        out = {}
        for spec in specs:
            run, name = spec.split(":")
            with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False) as z:
                for g in json.loads(str(z["header"]))["gids"]:
                    c = z[f"g{g}_cont"]
                    out[int(g)] = pp.movement_boost(c, float(bo["beta"]), int(bo["window"])) if boost else c
        return out

    ref = [c[s:e] for c in conts(S["reference"]).values() for s, e in islands(c[:, 0], isl)]
    sd = np.concatenate(ref)[:, lo:hi].std(0) + 1e-6

    def movement(cs):
        v = [float(np.mean(c[s:e, lo:hi].std(0) / sd)) for c in cs.values() for s, e in islands(c[:, 0], isl)]
        return float(np.median(v)), len(v)

    def whisper_rows(arm):
        rows = {}
        if "postproc" in arm:
            pj = json.loads((resolve_path("results/b2v2/postproc") / f"{arm['postproc']['set']}.json").read_text())
            for r in pj["rows"]:
                v = r[arm["postproc"]["variant"]]
                rows[int(r["gid"])] = (r["text"], v["orig_speed"], v["sped_up"])
        else:
            for spec in arm["preds"]:
                run, name = spec.split(":")
                sj = json.loads((resolve_path("results/b2v2") / run / f"scores_{name}.json").read_text())
                for r in sj["rows"]:
                    rows[int(r["gid"])] = (r["text"], r["orig"], r["fixed"])
        return rows

    res = {}
    for an, arm in S["arms"].items():
        W = whisper_rows(arm)
        idj = json.loads((resolve_path("results/b2v2/identify") / f"{arm['identify']}.json").read_text())
        ids = {int(r["gid"]): r for r in idj["rows"]}
        nerr = sum(w[1]["errors"] for w in W.values())
        nw = sum(w[1]["words"] for w in W.values())
        nerr_f = sum(w[2]["errors"] for w in W.values())
        dl = 0
        for text, o, _ in W.values():
            h = words(o["hyp"])
            dl += deletions(words(text), [x for i, x in enumerate(h) if i == 0 or x != h[i - 1]])
        mv, n_isl = movement(conts(arm["preds"], bool(arm.get("boost"))))
        res[an] = {"n": len(W), "wer": nerr / nw, "wer_sped_up": nerr_f / nw, "missing_words": dl / nw,
                   "movement": mv, "n_islands": n_isl,
                   "top1": float(np.mean([ids[g]["rank"] == 1 for g in W])),
                   "mean_rank": float(np.mean([ids[g]["rank"] for g in W])),
                   "per": sum(ids[g]["phone_errors"] for g in W) / sum(ids[g]["ref_phones"] for g in W),
                   "_trial_wer": {g: w[1]["wer"] for g, w in W.items()}}

    def paired(a, b):
        ga = res[a]["_trial_wer"]
        gb = res[b]["_trial_wer"]
        g = sorted(set(ga) & set(gb))
        d = np.array([ga[x] - gb[x] for x in g])
        return {"n": len(g), "mean_diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0}

    base, sec = S["baseline"], S["secondary_vs"]
    rule = S["rule"]
    for an in res:
        if an != base:
            res[an]["vs_baseline"] = paired(an, base)
        if an not in (base, sec):
            res[an]["vs_" + sec] = paired(an, sec)
        if an in S["primary"]:
            pb = res[an]["vs_baseline"]
            res[an]["rule_met"] = bool(res[base]["wer"] - res[an]["wer"] >= float(rule["min_points"]) / 100
                                       and pb["p"] < float(rule["alpha"]) and pb["mean_diff"] < 0)
    for an in res:
        res[an].pop("_trial_wer")
    out = {"decision": "D-086", "config": args.config, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "settings": S, "arms": res}
    dst = resolve_path("results/b2v2/d086")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["# D-086: making inner-speech voice less muddy (200 held-out imagined trials)", "",
             f"Baseline `{base}`. Rule (primary arms): pooled WER >= {rule['min_points']} points lower, paired "
             f"p < {rule['alpha']}. Movement inside words: attempted held-out = 1.00 by construction of the units "
             "(median over islands differs).", "",
             "| arm | WER | vs baseline (p) | vs " + sec + " (p) | WER sped up | missing words | movement | top-1 | "
             "mean rank | PER | rule |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for an, r in res.items():
        vb, vs = r.get("vs_baseline"), r.get("vs_" + sec)
        lines.append(
            f"| {an} | {100 * r['wer']:.1f}% | " + (f"{100 * vb['mean_diff']:+.1f} ({vb['p']:.2g})" if vb else "–")
            + " | " + (f"{100 * vs['mean_diff']:+.1f} ({vs['p']:.2g})" if vs else "–")
            + f" | {100 * r['wer_sped_up']:.1f}% | {100 * r['missing_words']:.1f}% | {r['movement']:.2f} | "
            f"{100 * r['top1']:.1f}% | {r['mean_rank']:.1f} | {100 * r['per']:.1f}% | "
            + ("met" if r.get("rule_met") else ("not met" if an in S["primary"] else "–")) + " |")
    (dst / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
