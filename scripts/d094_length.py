#!/usr/bin/env python
"""D-094: the inner-speech read-out vs training length (3,000 / 6,000 / 10,000 steps; no adapter; 2 seeds each).

    PC, project venv, CPU (a gq job):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/d094_length.py --config configs/experiments/b2v2_kunz_d094.yaml

``b2v2.d094.cells``: per length and seed, the decode runs and the identification result (as D-093; scored by
``d093_factorial.per_trial``, which stops on a decoded trial of a run's own training split). Per trial, the PER
(phoneme errors / reference phonemes) and words read right are averaged over the seeds; each contrast (``a`` minus
``b``) is a paired Wilcoxon over the 200 held-out trials. Primary contrasts count if negative (fewer errors) with
p < ``alpha``. Also each length vs the baselines and, per cell, the step the validation picked. Writes
``out``/length.json / .md.
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

    S = cfg["b2v2"]["d094"]
    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    raw = {k: _d093.per_trial(v, data) for k, v in S["cells"].items()}
    bases = {n: _d093.per_trial(v, data) for n, v in S["baselines"].items()}
    gids = sorted(set.intersection(*[set(v) for v in [*raw.values(), *bases.values()]]))
    if len(gids) != int(S["n_trials"]):
        raise SystemExit(f"{len(gids)} trials common to every cell and baseline, expected {S['n_trials']}")
    lengths = S["lengths"]

    def mean_over_seeds(cell, j):
        return np.array([np.mean([raw[f"{cell}_{s}"][g][j] for s in S["seeds"]]) for g in gids])

    def test(d):
        return {"mean_diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0, "n": int(len(d))}

    def best_step(run):
        h = [r for r in json.loads((resolve_path("results/b2v2") / run / "summary.json").read_text())["history"]
             if "val_ctc" in r]
        return int(min(h, key=lambda r: r["val_ctc"])["step"])

    res = {"cells": {}, "contrasts": {}, "vs_baseline": {}, "baselines": {}}
    for c in lengths:
        r = {}
        for s in S["seeds"]:
            x = raw[f"{c}_{s}"]
            r[s] = {"per": sum(x[g][3] for g in gids) / sum(x[g][4] for g in gids),
                    "words_read_right": float(np.mean([x[g][1] for g in gids])),
                    "top1": float(np.mean([x[g][2] for g in gids])),
                    "best_steps": [best_step(run) for run in S["cells"][f"{c}_{s}"]["runs"]]}
        r["seed_mean"] = {k: float(np.mean([r[s][k] for s in S["seeds"]])) for k in ("per", "words_read_right", "top1")}
        res["cells"][c] = r
    for n, x in bases.items():
        res["baselines"][n] = {"per": sum(x[g][3] for g in gids) / sum(x[g][4] for g in gids),
                               "words_read_right": float(np.mean([x[g][1] for g in gids]))}
    for j, metric in ((0, "per"), (1, "words_read_right")):
        m = {c: mean_over_seeds(c, j) for c in lengths}
        res["contrasts"][metric] = {f"{k['a']}-{k['b']}": test(m[k["a"]] - m[k["b"]])
                                    for k in S["primary"] + S["secondary"]}
        res["vs_baseline"][metric] = {n: {c: test(m[c] - np.array([x[g][j] for g in gids])) for c in lengths}
                                      for n, x in bases.items()}
    alpha = float(S["alpha"])
    res["primary"] = {f"{k['a']}-{k['b']}": bool(res["contrasts"]["per"][f"{k['a']}-{k['b']}"]["mean_diff"] < 0
                                                  and res["contrasts"]["per"][f"{k['a']}-{k['b']}"]["p"] < alpha)
                      for k in S["primary"]}
    out = {"decision": "D-094", "config": args.config, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "n_trials": len(gids), "settings": S,
           "results": res}
    dst = resolve_path(S["out"])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "length.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# D-094: the inner-speech read-out vs training length (no adapter; 200 held-out March trials, 2 seeds)", "",
         "Baselines: " + "; ".join(f"`{n}` PER {100 * b['per']:.1f}%" for n, b in res["baselines"].items()) + ".", "",
         "| steps | PER " + " | PER ".join(S["seeds"]) + " | PER mean | words read right | top-1 | validation's step (per fold) | "
         + " | ".join(f"PER vs {n} (p)" for n in bases) + " |", "|---|" + "---|" * (len(S["seeds"]) + 4 + len(bases))]
    for c in lengths:
        r = res["cells"][c]
        L.append(f"| {c} | " + " | ".join(f"{100 * r[s]['per']:.1f}%" for s in S["seeds"])
                 + f" | {100 * r['seed_mean']['per']:.1f}% | {100 * r['seed_mean']['words_read_right']:.1f}% | "
                 f"{100 * r['seed_mean']['top1']:.1f}% | "
                 + "; ".join(",".join(str(x) for x in r[s]["best_steps"]) for s in S["seeds"]) + " | "
                 + " | ".join(f"{100 * res['vs_baseline']['per'][n][c]['mean_diff']:+.1f} "
                              f"({res['vs_baseline']['per'][n][c]['p']:.2g})" for n in bases) + " |")
    L += ["", "Contrasts per trial (seed-averaged; PER in points, negative = fewer errors; words read right in points):", ""]
    for name in res["contrasts"]["per"]:
        cp, cw = res["contrasts"]["per"][name], res["contrasts"]["words_read_right"][name]
        tag = " (primary)" if name in res["primary"] else ""
        L.append(f"- {name}{tag}: PER {100 * cp['mean_diff']:+.2f} (p {cp['p']:.2g}); words read right "
                 f"{100 * cw['mean_diff']:+.2f} (p {cw['p']:.2g})")
    L += ["", f"Primary (each counts if PER is lower with p < {alpha}): "
          + "; ".join(f"{k} {'met' if v else 'not met'}" for k, v in res["primary"].items())]
    (dst / "length.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
