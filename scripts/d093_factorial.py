#!/usr/bin/env python
"""D-093: the 2 x 2 of the March-only shared adapter x training length (2 seeds each), on the read-out.

    PC, project venv, CPU (a gq job):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/d093_factorial.py --config configs/experiments/b2v2_kunz_d093.yaml

``b2v2.d093.cells``: per cell and seed, the decode runs (``imagined``) and the identification result
(results/b2v2/identify/<identify>.json, which holds each trial's phoneme errors and rank). Per trial and cell, the PER
(phoneme errors / reference phonemes) and the share of words read right (every phoneme of the word right in the greedy
50 ms read-out) are averaged over the seeds. Effects per trial (paired Wilcoxon, two-sided, over the 200 held-out
imagined trials): the adapter (mean of the adapter cells minus mean of the no-adapter cells), the training length
(long minus short), their interaction ((ad long - no long) - (ad short - no short)), and each cell vs each of
``baselines`` (name -> runs + identification, like a cell).
Writes ``out``/factorial.json / .md.
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

_spec = importlib.util.spec_from_file_location("d086_summary", Path(__file__).resolve().parent / "d086_summary.py")
_d086 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d086)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    from scipy.stats import wilcoxon

    from wvx import b2v2_data as bd, phonemes as ph
    from wvx.provenance import code_version

    S = cfg["b2v2"]["d093"]
    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    SIL = ph.ID["SIL"]

    def per_trial(spec):
        """{gid: (per, words_read_right, top1)} for one cell x seed (or the baseline)."""
        ids = {int(r["gid"]): r for r in json.loads(
            (resolve_path("results/b2v2/identify") / f"{spec['identify']}.json").read_text())["rows"]}
        out = {}
        for run in spec["runs"]:
            sp = resolve_path("results/b2v2") / run / "split.json"
            if sp.is_file():                            # never score a trial the run trained or validated on
                sj = json.loads(sp.read_text())
                seen = set(sj.get("val", [])) | set(sj.get("train", []))
                for d in (sj.get("days") or {}).values():
                    seen |= set(d.get("inner", [])) | set(d.get("attempted", []))
            else:
                seen = None
            with np.load(resolve_path("data/processed/b2v2_preds") / run / "imagined.npz", allow_pickle=False) as z:
                gids_run = [int(g) for g in json.loads(str(z["header"]))["gids"]]
                if seen is None and "d093" in run:
                    raise SystemExit(f"{run}: no split.json to check the decoded trials against")
                if seen and set(gids_run) & seen:
                    raise SystemExit(f"{run}: {len(set(gids_run) & seen)} decoded trials are in its training split")
                for g in gids_run:
                    lab = data.by_gid[int(g)]["label"]
                    hyp = [x for x in ph.ctc_greedy(z[f"g{g}_lp50"].astype(np.float64).argmax(1)) if x != SIL]
                    ref = [x for x in lab if x != SIL]
                    wid = np.cumsum([0] + [x == SIL for x in lab])[:-1]
                    ok = {}
                    for w_, op in zip([int(wid[i]) for i, x in enumerate(lab) if x != SIL], _d086.phone_ops(ref, hyp)):
                        ok[w_] = ok.get(w_, True) and op == "hit"
                    r = ids[int(g)]
                    out[int(g)] = (r["phone_errors"] / max(1, r["ref_phones"]), sum(ok.values()) / max(1, len(ok)),
                                   float(r["rank"] == 1), r["phone_errors"], r["ref_phones"])
        return out

    cells = sorted({k.rsplit("_", 1)[0] for k in S["cells"]})
    raw = {k: per_trial(v) for k, v in S["cells"].items()}
    bases = {n: per_trial(v) for n, v in S["baselines"].items()}
    gids = sorted(set.intersection(*[set(v) for v in [*raw.values(), *bases.values()]]))
    if S.get("n_trials") and len(gids) != int(S["n_trials"]):
        raise SystemExit(f"{len(gids)} trials common to every cell and baseline, expected {S['n_trials']}")

    def cell_mean(cell, j):          # per trial, the seed average of metric j
        return np.array([np.mean([raw[f"{cell}_{s}"][g][j] for s in S["seeds"]]) for g in gids])

    def test(d):
        return {"mean_diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0, "n": int(len(d))}

    res = {"cells": {}, "effects": {}, "baselines": {}, "vs_baseline": {n: {} for n in bases}}
    for c in cells:
        res["cells"][c] = {s: {"per": sum(raw[f"{c}_{s}"][g][3] for g in gids) / sum(raw[f"{c}_{s}"][g][4] for g in gids),
                               "words_read_right": float(np.mean([raw[f"{c}_{s}"][g][1] for g in gids])),
                               "top1": float(np.mean([raw[f"{c}_{s}"][g][2] for g in gids]))} for s in S["seeds"]}
        res["cells"][c]["seed_mean"] = {k: float(np.mean([res["cells"][c][s][k] for s in S["seeds"]]))
                                        for k in ("per", "words_read_right", "top1")}
    for n, base in bases.items():
        res["baselines"][n] = {"per": sum(base[g][3] for g in gids) / sum(base[g][4] for g in gids),
                               "words_read_right": float(np.mean([base[g][1] for g in gids])),
                               "top1": float(np.mean([base[g][2] for g in gids]))}
    F = S["factors"]
    for j, metric in ((0, "per"), (1, "words_read_right")):
        m = {c: cell_mean(c, j) for c in cells}
        avg = lambda names: np.mean([m[c] for c in names], 0)  # noqa: E731
        res["effects"][metric] = {
            "adapter": test(avg(F["adapter"]) - avg(F["no_adapter"])),
            "length": test(avg(F["long"]) - avg(F["short"])),
            "interaction": test((m[F["long"][1]] - m[F["long"][0]]) - (m[F["short"][1]] - m[F["short"][0]])),
        }
        for n, base in bases.items():
            b = np.array([base[g][j] for g in gids])
            res["vs_baseline"][n][metric] = {c: test(m[c] - b) for c in cells}
    a = res["effects"]["per"]["adapter"]
    res["primary_adapter_lowers_per"] = bool(a["mean_diff"] < 0 and a["p"] < float(S["alpha"]))
    out = {"decision": "D-093", "config": args.config, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "n_trials": len(gids), "settings": S,
           "results": res}
    dst = resolve_path(S["out"])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "factorial.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# D-093: the March-only shared adapter x training length (200 held-out March trials, 2 seeds per cell)", ""]
    for n, rb in res["baselines"].items():
        L.append(f"Baseline `{n}` ({S['baselines'][n]['identify']}): PER {100 * rb['per']:.1f}%, words read right "
                 f"{100 * rb['words_read_right']:.1f}%, top-1 {100 * rb['top1']:.1f}%.")
    L += ["", "| cell | PER " + " | PER ".join(S["seeds"]) + " | PER mean | words read right (mean) | top-1 (mean) | "
          + " | ".join(f"PER vs {n} (p)" for n in bases) + " |", "|---|" + "---|" * (len(S["seeds"]) + 3 + len(bases))]
    for c in cells:
        r = res["cells"][c]
        L.append(f"| {c} | " + " | ".join(f"{100 * r[s]['per']:.1f}%" for s in S["seeds"])
                 + f" | {100 * r['seed_mean']['per']:.1f}% | {100 * r['seed_mean']['words_read_right']:.1f}% | "
                 f"{100 * r['seed_mean']['top1']:.1f}% | " + " | ".join(
                     f"{100 * vb['mean_diff']:+.1f} ({vb['p']:.2g})" for vb in
                     (res["vs_baseline"][n]["per"][c] for n in bases)) + " |")
    L += ["", "Effects per trial (seed-averaged; PER in points, negative = fewer errors):", ""]
    for metric in ("per", "words_read_right"):
        e = res["effects"][metric]
        L.append(f"- {metric}: adapter {100 * e['adapter']['mean_diff']:+.2f} (p {e['adapter']['p']:.2g}); "
                 f"length {100 * e['length']['mean_diff']:+.2f} (p {e['length']['p']:.2g}); interaction "
                 f"{100 * e['interaction']['mean_diff']:+.2f} (p {e['interaction']['p']:.2g})")
    L += ["", f"Primary (the adapter lowers PER, p < {S['alpha']}): "
          + ("met" if res["primary_adapter_lowers_per"] else "not met")]
    (dst / "factorial.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
