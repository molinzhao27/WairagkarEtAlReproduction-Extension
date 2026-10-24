#!/usr/bin/env python
"""Simulated live decoding with a trained brain2voice-2.0-style decoder (D-075).

    PyTorch env (GPU if present):
    ~/wvx/envs/wvx-f5/bin/python -I scripts/decode_b2v2.py --config configs/experiments/b2v2_all.yaml --trials test

Every 10 ms bin of each trial: the last ``b2v2.decode.window_bins`` (paper: 800 ms) as 50 ms tokens ending there,
the last output kept (``wvx.b2v2_model.decode_live``). ``--trials``: ``test`` / ``val`` / ``train`` (the run's
split.json), ``benchmark``, or ``session:<name>:<split json>:<key>`` (trial indices in that session, e.g. the D-049
day195 test trials; output name ``<name>_<key>``), or ``sel:<session>[:<blocks>]`` (e.g. ``sel:attempted:3,6``).
``--mode chunked``: causal long-context decoding instead (``wvx.b2v2_model.decode_chunked``). Writes
data/processed/b2v2_preds/<run_name>/<name>.npz: per trial ``g<gid>_cont`` (T, 20) float32 and ``g<gid>_best`` (T,)
uint8 (argmax phoneme per bin); chunked decodes add ``g<gid>_best50`` / ``g<gid>_lp50`` (the phase-0 50 ms argmax
and log-probabilities); and a JSON header.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def select(data, spec: str, run_dir: Path) -> list[int]:
    if spec in ("test", "val", "train"):
        return json.loads((run_dir / "split.json").read_text())[spec]
    if spec == "benchmark":
        return [t["gid"] for t in data.trials if t["benchmark"]]
    if spec.startswith("sel:"):                          # sel:<session>[:<block>,<block>...] (Kunz, D-076)
        parts = spec.split(":")
        blocks = set(int(k) for k in parts[2].split(",")) if len(parts) > 2 and parts[2] else None
        return [t["gid"] for t in data.trials if t["session"] == parts[1] and (blocks is None or t.get("block") in blocks)]
    kind, name, path, key = spec.split(":")
    if kind != "session":
        raise SystemExit(f"unknown --trials {spec!r}")
    idx = set(int(i) for i in json.loads(resolve_path(path).read_text())[key])
    return [t["gid"] for t in data.trials if t["session"] == name and t["index"] in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="final", help="final | best | a path")
    ap.add_argument("--trials", required=True)
    ap.add_argument("--name", default=None, help="output name (default: the --trials value, ':' -> '_')")
    ap.add_argument("--window-bins", type=int, default=None, help="override b2v2.decode.window_bins")
    ap.add_argument("--mode", choices=["live", "chunked"], default="live",
                    help="live = the paper's sliding window (primary); chunked = causal long context "
                         "(wvx.b2v2_model.decode_chunked, a D-075 debugging check)")
    ap.add_argument("--data-dir", default=None, help="override b2v2.data_dir (e.g. projected Kunz data, D-077)")
    ap.add_argument("--ctx", type=int, default=128, help="chunked: tokens per window")
    ap.add_argument("--hop", type=int, default=64, help="chunked: kept outputs per window after the first")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    import torch

    from wvx import b2v2_data as bd, b2v2_model as bm
    from wvx.provenance import code_version

    b, run = cfg["b2v2"], cfg["run_name"]
    ckpt = resolve_path(cfg["paths"]["checkpoints"]) / run / f"{args.checkpoint}.pt" \
        if args.checkpoint in ("final", "best") else resolve_path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device)
    model = bm.B2V2(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["state"])
    input_id = int(ck.get("input_id", 0))
    data = bd.B2V2Data(resolve_path(args.data_dir or b["data_dir"]))
    gids = select(data, args.trials, resolve_path("results/b2v2") / run)
    window = int(args.window_bins or b["decode"]["window_bins"])
    pool = int(b["data"]["pool"])
    parts = args.trials.split(":")
    name = args.name or (f"{parts[1]}_{parts[3]}" if parts[0] == "session" else args.trials)
    t0, out = time.time(), {}
    for g in gids:
        x, _ = data.trial_arrays(data.by_gid[g])
        if args.mode == "live":
            c, lg = bm.decode_live(model, x, window, pool, input_id=input_id, device=device)
        else:
            c, lg, ph0 = bm.decode_chunked(model, x, pool, args.ctx, args.hop, input_id=input_id, device=device)
            out[f"g{g}_best50"] = ph0.argmax(-1).astype(np.uint8)
            lp = ph0 - ph0.max(-1, keepdims=True)
            out[f"g{g}_lp50"] = (lp - np.log(np.exp(lp).sum(-1, keepdims=True))).astype(np.float16)   # log-softmax
        out[f"g{g}_cont"] = c.astype(np.float32)
        out[f"g{g}_best"] = lg.argmax(-1).astype(np.uint8)
    header = {"run_name": run, "checkpoint": str(ckpt), "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
              "input_id": input_id, "trials": args.trials, "gids": gids, "mode": args.mode,
              "data_dir": str(resolve_path(args.data_dir or b["data_dir"])),
              "window_bins": window if args.mode == "live" else None,
              "chunked": {"ctx": args.ctx, "hop": args.hop} if args.mode == "chunked" else None, "pool": pool,
              "device": str(device), "code_version": code_version(), "decision": "D-075",
              "created": datetime.datetime.now().isoformat(timespec="seconds")}
    path = resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, header=np.array(json.dumps(header)), **out)
    tmp.replace(path)
    print(f"{len(gids)} trials decoded in {time.time() - t0:.0f} s ({device}, {args.mode}) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
