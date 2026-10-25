#!/usr/bin/env python
"""D-086 self-distillation targets: the decoder's own voice for ATTEMPTED speech of each sentence, time-warped onto
the inner-speech trials of the same sentence (``wvx.distill``).

    PC, any env with NumPy, CPU:
    .venv/bin/python -I scripts/build_distill_targets.py --config configs/experiments/b2v2_kunz_d086_distill_k03.yaml

``b2v2.adapt.distill``: ``attempted`` = RUN:NAME of a decode of the attempted blocks whose sentences the training
imagined blocks repeat (continuous features and phase-0 phoneme log-probabilities); ``inner`` = RUN:NAME of a decode
of those imagined trials (only its phoneme stream is used, for the alignment); ``targets`` = the npz written here
(``g<gid>``: (T, 20) per imagined trial, 10 ms frames, the trial's length; header: provenance and anchor counts).
Each imagined trial's partner is the attempted trial of its ``paired_block`` with the same text.
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
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np

    from wvx import b2v2_data as bd, distill
    from wvx.provenance import code_version

    b = cfg["b2v2"]
    d = b["adapt"]["distill"]
    pool = int(b["data"]["pool"])
    data = bd.B2V2Data(resolve_path(b["data_dir"]))

    def load(spec):
        run, name = spec.split(":")
        z = np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False)
        return z, json.loads(str(z["header"]))["gids"]

    za, ga = load(d["attempted"])
    zi, gi = load(d["inner"])
    partner = {(data.by_gid[g]["block"], data.by_gid[g]["text"]): g for g in ga}
    out, info, missing = {}, {}, []
    for g in gi:
        tr = data.by_gid[g]
        p = partner.get((tr["paired_block"], tr["text"]))
        if p is None:
            missing.append(g)
            continue
        att = za[f"g{p}_cont"].astype(np.float32)
        f = distill.warp_map(zi[f"g{g}_lp50"].astype(np.float64), za[f"g{p}_lp50"].astype(np.float64), tr["label"],
                             int(tr["length"]), len(att), pool)
        out[f"g{g}"] = distill.warp_features(att, f)
        info[int(g)] = {"partner": int(p), "inner_s": int(tr["length"]) / 100, "attempted_s": len(att) / 100}
    dst = resolve_path(d["targets"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    header = {"config": args.config, "decision": "D-086", "attempted": d["attempted"], "inner": d["inner"],
              "n": len(out), "missing_partner": missing, "trials": info, "code_version": code_version(),
              "created": datetime.datetime.now().isoformat(timespec="seconds")}
    np.savez(dst, header=json.dumps(header), **out)
    print(f"{dst}: {len(out)} targets, {len(missing)} without a partner")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
