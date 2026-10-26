#!/usr/bin/env python
"""D-096: strengthening the other participants' inner-speech adaptation (T12, T16): a 2 x 2 of more data (their
attempted sentences too) x more capacity (the first decoder layer trained too) around D-095's stage B.

    PC, project venv, CPU (a gq job):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/d096_strengthen.py --config configs/experiments/b2v2_d096.yaml

Per arm and participant: the held-out imagined blocks' PER (pooled), words read right, top-1 of 40 and mean rank
(``d093_factorial.per_trial``, with its leak guard, and the identification rows). Primary: each arm vs ``baseline``
(D-095 stage B), per trial over both participants' 320 trials (paired Wilcoxon): it counts if its pooled PER is
at least ``min_gain`` points lower with p < ``alpha``. Secondary: per participant; ranks; the 2 x 2 effects.
Writes ``out``/strengthen.json / .md.
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

    S = cfg["b2v2"]["d096"]
    parts = S["participants"]
    data = {p: bd.B2V2Data(resolve_path(v["data_dir"])) for p, v in parts.items()}
    arms = [S["baseline"]] + S["arms"]

    def wx(d):
        d = np.asarray(d, float)
        return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0

    T = {}                                                # arm -> participant -> {gid: (per, wrr, top1, err, ref, rank)}
    for a in arms:
        T[a] = {}
        for p, v in parts.items():
            folds = [f"a{b:02d}" for b in v["folds"]]
            runs = [S["run_pattern"][a].format(p=p, f=f) for f in folds]
            name = S["identify_pattern"][a].format(p=p)
            pt = _d093.per_trial({"runs": runs, "identify": name, "pred": "imagined"}, data[p])
            ids = {int(r["gid"]): r for r in json.loads(
                (resolve_path("results/b2v2/identify") / f"{name}.json").read_text())["rows"]}
            if set(pt) != set(ids) or len(pt) != int(v["n_trials"]):
                raise SystemExit(f"{a}/{p}: {len(pt)} decoded, {len(ids)} identified, expected {v['n_trials']}")
            T[a][p] = {g: pt[g] + (ids[g]["rank"],) for g in pt}
    keys = [(p, g) for p in parts for g in sorted(T[S["baseline"]][p])]

    def col(a, j, ps=None):
        return np.array([T[a][p][g][j] for p, g in keys if ps is None or p in ps])

    def pooled_per(a, ps=None):
        return float(col(a, 3, ps).sum() / col(a, 4, ps).sum())

    res = {"arms": {}, "vs_baseline": {}, "per_participant": {}, "effects": {}}
    for a in arms:
        res["arms"][a] = {"per": pooled_per(a), "words_read_right": float(col(a, 1).mean()),
                          "top1": float(col(a, 2).mean()), "mean_rank": float(col(a, 5).mean()),
                          **{f"{p}_{k}": v for p in parts for k, v in (
                              ("per", pooled_per(a, [p])), ("top1", float(col(a, 2, [p]).mean())),
                              ("mean_rank", float(col(a, 5, [p]).mean())))}}
    b = S["baseline"]
    for a in S["arms"]:
        d_per = col(a, 0) - col(b, 0)
        res["vs_baseline"][a] = {"per_points": 100 * (pooled_per(a) - pooled_per(b)), "p_per": wx(d_per),
                                 "rank_diff": float((col(a, 5) - col(b, 5)).mean()), "p_rank": wx(col(a, 5) - col(b, 5)),
                                 "wrr_points": 100 * float((col(a, 1) - col(b, 1)).mean()),
                                 "p_wrr": wx(col(a, 1) - col(b, 1))}
        v = res["vs_baseline"][a]
        v["primary_met"] = bool(v["per_points"] <= -float(S["min_gain"]) and v["p_per"] < float(S["alpha"]))
        for p in parts:
            res["per_participant"].setdefault(a, {})[p] = {
                "per_points": 100 * (pooled_per(a, [p]) - pooled_per(b, [p])),
                "p_per": wx(col(a, 0, [p]) - col(b, 0, [p])),
                "rank_diff": float((col(a, 5, [p]) - col(b, 5, [p])).mean()), "p_rank": wx(col(a, 5, [p]) - col(b, 5, [p]))}
    F = S.get("factors")                                # D-096's 2 x 2 (optional: D-098 has plain arms)
    m = {a: col(a, 0) for a in arms}
    res["effects"] = {} if not F else {"data": wx((m[F["data_yes"][0]] + m[F["data_yes"][1]]) - (m[F["data_no"][0]] + m[F["data_no"][1]])),
                      "data_points": 50 * float(((m[F["data_yes"][0]] + m[F["data_yes"][1]]) - (m[F["data_no"][0]] + m[F["data_no"][1]])).mean()),
                      "capacity": wx((m[F["cap_yes"][0]] + m[F["cap_yes"][1]]) - (m[F["cap_no"][0]] + m[F["cap_no"][1]])),
                      "capacity_points": 50 * float(((m[F["cap_yes"][0]] + m[F["cap_yes"][1]]) - (m[F["cap_no"][0]] + m[F["cap_no"][1]])).mean())}
    out = {"decision": S.get("decision", "D-096"), "config": args.config, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "n_trials": len(keys), "settings": S,
           "results": res}
    dst = resolve_path(S["out"])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "strengthen.json").write_text(json.dumps(out, indent=1) + "\n")
    ps = list(parts)
    title = S.get("title", "D-096: strengthening the other participants' inner-speech adaptation (held-out imagined blocks")
    L = [f"# {title}; "
         f"{len(keys)} trials: " + ", ".join(f"{p} {parts[p]['n_trials']}" for p in ps) + ")", "",
         "| arm | PER | words read right | top-1 of 40 | mean rank | " + " | ".join(f"{p} PER / top-1" for p in ps)
         + " | PER vs baseline (p) | rank vs baseline (p) | primary |", "|---|" + "---|" * (7 + len(ps))]
    for a in arms:
        r = res["arms"][a]
        vb = res["vs_baseline"].get(a)
        L.append(f"| {a} | {100 * r['per']:.1f}% | {100 * r['words_read_right']:.1f}% | {100 * r['top1']:.1f}% | "
                 f"{r['mean_rank']:.1f} | " + " | ".join(f"{100 * r[p + '_per']:.1f}% / {100 * r[p + '_top1']:.1f}%" for p in ps)
                 + (f" | {vb['per_points']:+.1f} ({vb['p_per']:.2g}) | {vb['rank_diff']:+.1f} ({vb['p_rank']:.2g}) | "
                    f"{'met' if vb['primary_met'] else 'not met'} |" if vb else " | – | – | – |"))
    e = res["effects"]
    if e:
        L += ["", f"2 x 2 (PER points, per trial): more data {e['data_points']:+.2f} (p {e['data']:.2g}); "
              f"more capacity {e['capacity_points']:+.2f} (p {e['capacity']:.2g})."]
    L += ["", f"Primary (each arm vs `{b}`: pooled PER >= {S['min_gain']} points lower, paired p < {S['alpha']})."]
    (dst / "strengthen.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
