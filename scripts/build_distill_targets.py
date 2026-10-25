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
``kind: hidden`` (D-089): instead of the decoded voice, the TEACHER's encoder output on the attempted trial
(``teacher``: a checkpoint, run on the attempted neural data exactly as decoding runs, all 50 ms phases;
``wvx.b2v2_model.decode_hidden``), warped the same way and stored as float16 (T, d). Needs PyTorch (GPU if present).
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
    teacher = None
    if d.get("kind", "voice") == "hidden":
        import torch

        from wvx import b2v2_model as bm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(resolve_path(d["teacher"]), map_location=device)
        teacher = bm.B2V2(ck["model_cfg"]).to(device)
        teacher.load_state_dict(ck["state"])
        t_in = int(ck.get("input_id", 0))
        tt = b["adapt"]["training"]
    partner = {(data.by_gid[g]["block"], data.by_gid[g]["text"]): g for g in ga}
    out, info, missing = {}, {}, []
    for g in gi:
        tr = data.by_gid[g]
        p = partner.get((tr["paired_block"], tr["text"]))
        if p is None:
            missing.append(g)
            continue
        att = za[f"g{p}_cont"].astype(np.float32)
        if teacher is not None:
            x, _ = data.trial_arrays(data.by_gid[p])
            att = bm.decode_hidden(teacher, x, pool, int(tt["ctx"]), int(tt["hop"]), input_id=t_in)
        f = distill.warp_map(zi[f"g{g}_lp50"].astype(np.float64), za[f"g{p}_lp50"].astype(np.float64), tr["label"],
                             int(tr["length"]), len(att), pool)
        out[f"g{g}"] = distill.warp_features(att, f).astype(np.float16 if teacher is not None else np.float32)
        info[int(g)] = {"partner": int(p), "inner_s": int(tr["length"]) / 100, "attempted_s": len(att) / 100}
    dst = resolve_path(d["targets"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    header = {"config": args.config, "decision": b["adapt"].get("decision", "D-086"), "kind": d.get("kind", "voice"),
              "teacher": d.get("teacher"), "attempted": d["attempted"], "inner": d["inner"],
              "n": len(out), "missing_partner": missing, "trials": info, "code_version": code_version(),
              "created": datetime.datetime.now().isoformat(timespec="seconds")}
    np.savez(dst, header=json.dumps(header), **out)
    print(f"{dst}: {len(out)} targets, {len(missing)} without a partner")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
