#!/usr/bin/env python
"""D-095: T15's voice decoder for other participants (T12, T16): stage A (attempted) and stage B (inner speech).

    PC, project venv, CPU (a gq job):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/d095_generalize.py --config configs/experiments/b2v2_d095.yaml

Per participant (``b2v2.d095.participants``), from the identification results (results/b2v2/identify/<name>.json;
each trial's true-sentence rank among its block's sentences and the length-matched null rank) and the decodes
(scored by ``d093_factorial.per_trial``, which stops on a decoded trial of the run's own training or validation split):
- stage A, held-out attempted blocks: `transfer` (T15's decoder + a new input) vs `random` (a random decoder + a new
  input): paired per-trial rank (Wilcoxon, two-sided); each vs its null; top-1, mean rank, PER, words read right;
  Whisper word errors of `transfer`'s voice (results/b2v2/<run>/scores_heldout.json, pooled). The gate for stage B:
  `transfer` above its null with p < ``alpha_gate``.
- stage B (if its results exist), held-out imagined blocks: adapted vs its null (primary), vs stage A's zero-shot
  decoding of the same trials (paired rank), top-1, PER, words read right, Whisper word errors.
Writes ``out``/generalize.json / .md.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("d093_factorial", Path(__file__).resolve().parent / "d093_factorial.py")
_d093 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d093)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    from scipy.stats import wilcoxon

    from wvx import b2v2_data as bd
    from wvx.provenance import code_version

    S = cfg["b2v2"]["d095"]

    def wx(d):
        d = np.asarray(d, float)
        return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0

    def rows(name):
        f = resolve_path("results/b2v2/identify") / f"{name}.json"
        return {int(r["gid"]): r for r in json.loads(f.read_text())["rows"]} if f.is_file() else None

    def whisper(runs, pred):
        e = w = e2 = 0
        for run in runs:
            f = resolve_path("results/b2v2") / run / f"scores_{pred}.json"
            if not f.is_file():
                return None
            for r in json.loads(f.read_text())["rows"]:
                e, w, e2 = e + r["orig"]["errors"], w + r["orig"]["words"], e2 + r["fixed"]["errors"]
        return {"wer": e / max(1, w), "wer_sped_up": e2 / max(1, w), "words": w}

    def arm(p, name, runs, pred, data):
        r = rows(name)
        if r is None:
            return None
        pt = _d093.per_trial({"runs": runs, "identify": name, "pred": pred}, data)
        g = sorted(pt)
        if len(g) != len(r):
            raise SystemExit(f"{name}: {len(g)} decoded trials vs {len(r)} identified")
        rk = np.array([r[x]["rank"] for x in g])
        nl = np.array([r[x]["null_rank"] for x in g])
        return {"gids": g, "rank": rk, "null": nl, "n_candidates": int(np.median([r[x]["n_candidates"] for x in g])),
                "summary": {"n": len(g), "top1": float(np.mean(rk == 1)), "mean_rank": float(rk.mean()),
                            "null_mean_rank": float(nl.mean()), "p_vs_null": wx(rk - nl),
                            "per": sum(pt[x][3] for x in g) / sum(pt[x][4] for x in g),
                            "words_read_right": float(np.mean([pt[x][1] for x in g]))}}

    res = {}
    for p, P in S["participants"].items():
        data = bd.B2V2Data(resolve_path(P["data_dir"]))
        folds = [f"a{b:02d}" for b in P["folds"]]
        R = {}
        tr = arm(p, f"d095_{p}_transfer_attempted", [f"b2v2_d095_{p}_transfer_{f}" for f in folds], "heldout", data)
        rn = arm(p, f"d095_{p}_random_attempted", [f"b2v2_d095_{p}_random_{f}" for f in folds], "heldout", data)
        z0 = arm(p, f"d095_{p}_transfer_inner0", [f"b2v2_d095_{p}_transfer_{f}" for f in folds], "inner0", data)
        if tr and rn:
            if tr["gids"] != rn["gids"]:
                raise SystemExit(f"{p}: transfer and random scored different trials")
            R["A"] = {"transfer": tr["summary"], "random": rn["summary"], "n_candidates": tr["n_candidates"],
                      "transfer_minus_random_rank": float((tr["rank"] - rn["rank"]).mean()),
                      "p_transfer_vs_random": wx(tr["rank"] - rn["rank"]),
                      "whisper_transfer": whisper([f"b2v2_d095_{p}_transfer_{f}" for f in folds], "heldout")}
            R["A"]["primary_met"] = bool(R["A"]["transfer_minus_random_rank"] < 0
                                         and R["A"]["p_transfer_vs_random"] < float(S["alpha_a"]))
            R["A"]["gate_met"] = bool(tr["summary"]["mean_rank"] < tr["summary"]["null_mean_rank"]
                                      and tr["summary"]["p_vs_null"] < float(S["alpha_gate"]))
        if z0:
            R["inner_zero_shot"] = z0["summary"]
        ib = arm(p, f"d095_{p}_inner", [f"b2v2_d095_{p}_inner_{f}" for f in folds], "imagined", data)
        if ib:
            R["B"] = {"adapted": ib["summary"], "whisper": whisper([f"b2v2_d095_{p}_inner_{f}" for f in folds],
                                                                   "imagined")}
            R["B"]["primary_met"] = bool(ib["summary"]["mean_rank"] < ib["summary"]["null_mean_rank"]
                                         and ib["summary"]["p_vs_null"] < float(S["alpha_b"]))
            if z0 and z0["gids"] == ib["gids"]:
                R["B"]["adapted_minus_zero_shot_rank"] = float((ib["rank"] - z0["rank"]).mean())
                R["B"]["p_vs_zero_shot"] = wx(ib["rank"] - z0["rank"])
        res[p] = R
    out = {"decision": "D-095", "config": args.config, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "settings": S, "results": res}
    dst = resolve_path(S["out"])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "generalize.json").write_text(json.dumps(out, indent=1, default=float) + "\n")

    def line(tag, s):
        return (f"| {tag} | {s['n']} | {100 * s['top1']:.1f}% | {s['mean_rank']:.1f} | {s['null_mean_rank']:.1f} | "
                f"{s['p_vs_null']:.2g} | {100 * s['per']:.1f}% | {100 * s['words_read_right']:.1f}% |")
    L = ["# D-095: T15's voice decoder for other participants (text-only adaptation)", "",
         "| participant / stage / arm | trials | top-1 | mean rank | null mean rank | p vs null | PER | words read right |",
         "|---|---|---|---|---|---|---|---|"]
    for p, R in res.items():
        if "A" in R:
            L += [line(f"{p} A transfer (attempted)", R["A"]["transfer"]), line(f"{p} A random (attempted)", R["A"]["random"])]
        if "inner_zero_shot" in R:
            L.append(line(f"{p} inner, zero-shot", R["inner_zero_shot"]))
        if "B" in R:
            L.append(line(f"{p} B inner, adapted", R["B"]["adapted"]))
    L.append("")
    for p, R in res.items():
        if "A" in R:
            a, w = R["A"], R["A"]["whisper_transfer"]
            L.append(f"- {p} stage A ({a['n_candidates']} candidates per block): transfer - random mean rank "
                     f"{a['transfer_minus_random_rank']:+.1f} (p {a['p_transfer_vs_random']:.2g}): primary "
                     f"{'met' if a['primary_met'] else 'not met'}; gate for stage B {'met' if a['gate_met'] else 'not met'}"
                     + (f"; Whisper word errors of the voice {100 * w['wer']:.1f}% (sped up {100 * w['wer_sped_up']:.1f}%)"
                        if w else ""))
        if "B" in R:
            b, w = R["B"], R["B"]["whisper"]
            L.append(f"- {p} stage B: primary (adapted inner above its null, p < {S['alpha_b']}) "
                     f"{'met' if b['primary_met'] else 'not met'}"
                     + (f"; vs zero-shot mean rank {b['adapted_minus_zero_shot_rank']:+.1f} (p {b['p_vs_zero_shot']:.2g})"
                        if "p_vs_zero_shot" in b else "")
                     + (f"; Whisper word errors {100 * w['wer']:.1f}% (sped up {100 * w['wer_sped_up']:.1f}%)" if w else ""))
    (dst / "generalize.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
