#!/usr/bin/env python
"""D-099: does D-086's voice target, added to T15's best inner recipe (D-097 `unfreeze`), make the voice intelligible?

    PC, project venv, CPU (a gq job):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/d099_voice.py --config configs/experiments/b2v2_kunz_d099.yaml

Per cell (``b2v2.d099.cells``: seeds x runs, the Whisper scores results/b2v2/<run>/scores_imagined.json and the
identification), per trial the seed-averaged Whisper word error rate (its own pace and sped up) and PER
(``d093_factorial.per_trial``, with the leak guard). Primary: `voice` vs `base`: pooled WER (word errors / reference
words, mean over seeds) at least ``min_gain`` points lower and the per-trial paired Wilcoxon p < ``alpha``.
Secondary: sped-up WER, PER, words read right, top-1. Writes ``out``/voice.json / .md.
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

    S = cfg["b2v2"]["d099"]
    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))

    def wx(d):
        d = np.asarray(d, float)
        return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0

    cells = {}
    for c, seeds in S["cells"].items():
        per_seed = []
        for spec in seeds:
            w = {}
            for run in spec["runs"]:
                for r in json.loads((resolve_path("results/b2v2") / run / f"scores_{spec.get('score', 'imagined')}.json").read_text())["rows"]:
                    w[int(r["gid"])] = (r["orig"]["errors"], r["fixed"]["errors"], r["orig"]["words"])
            pt = _d093.per_trial({"runs": spec["runs"], "identify": spec["identify"], "pred": spec.get("pred", "imagined")}, data)
            if set(w) != set(pt):
                raise SystemExit(f"{c}: Whisper and decodes cover different trials")
            per_seed.append({g: w[g] + pt[g] for g in pt})
        cells[c] = per_seed
    gids = sorted(set.intersection(*[set(s) for v in cells.values() for s in v]))
    if len(gids) != int(S["n_trials"]):
        raise SystemExit(f"{len(gids)} trials, expected {S['n_trials']}")

    def stat(c):
        ss = cells[c]
        wer = float(np.mean([sum(s[g][0] for g in gids) / sum(s[g][2] for g in gids) for s in ss]))
        wer_f = float(np.mean([sum(s[g][1] for g in gids) / sum(s[g][2] for g in gids) for s in ss]))
        per = float(np.mean([sum(s[g][6] for g in gids) / sum(s[g][7] for g in gids) for s in ss]))
        return {"wer": wer, "wer_sped_up": wer_f, "per": per,
                "words_read_right": float(np.mean([s[g][4] for s in ss for g in gids])),
                "top1": float(np.mean([s[g][5] for s in ss for g in gids]))}

    def trial(c, j, words=False):                      # seed-averaged per trial; j: 0 errors, 1 sped-up errors, 3 PER
        return np.array([np.mean([(s[g][j] / max(1, s[g][2])) if words else s[g][j] for s in cells[c]]) for g in gids])

    res = {"cells": {c: stat(c) for c in cells}}
    a, b = S["arm"], S["baseline"]
    res["contrast"] = {"wer_points": 100 * (res["cells"][a]["wer"] - res["cells"][b]["wer"]),
                       "p_wer": wx(trial(a, 0, True) - trial(b, 0, True)),
                       "wer_sped_up_points": 100 * (res["cells"][a]["wer_sped_up"] - res["cells"][b]["wer_sped_up"]),
                       "p_wer_sped_up": wx(trial(a, 1, True) - trial(b, 1, True)),
                       "per_points": 100 * (res["cells"][a]["per"] - res["cells"][b]["per"]),
                       "p_per": wx(trial(a, 3) - trial(b, 3))}
    k = res["contrast"]
    res["primary_met"] = bool(k["wer_points"] <= -float(S["min_gain"]) and k["p_wer"] < float(S["alpha"]))
    out = {"decision": S.get("decision", "D-099"), "config": args.config, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "n_trials": len(gids), "settings": S,
           "results": res}
    dst = resolve_path(S["out"])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "voice.json").write_text(json.dumps(out, indent=1) + "\n")
    L = [f"# {S.get('title', 'D-099: T15 inner-speech voice with the D-086 voice target')} ({len(gids)} held-out trials, "
         f"{len(cells[a])} seeds)", "", "| cell | Whisper WER | WER sped up | PER | words read right | top-1 of 50 |",
         "|---|---|---|---|---|---|"]
    for c, r in res["cells"].items():
        L.append(f"| {c} | {100 * r['wer']:.1f}% | {100 * r['wer_sped_up']:.1f}% | {100 * r['per']:.1f}% | "
                 f"{100 * r['words_read_right']:.1f}% | {100 * r['top1']:.1f}% |")
    L += ["", f"`{a}` vs `{b}`: WER {k['wer_points']:+.1f} points (p {k['p_wer']:.2g}); sped up {k['wer_sped_up_points']:+.1f} "
          f"(p {k['p_wer_sped_up']:.2g}); PER {k['per_points']:+.1f} (p {k['p_per']:.2g}).", "",
          f"Primary (WER at least {S['min_gain']} points lower, p < {S['alpha']}): "
          + ("met" if res["primary_met"] else "not met")]
    (dst / "voice.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
