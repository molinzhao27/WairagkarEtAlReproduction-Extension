#!/usr/bin/env python
"""A randomly initialized b2v2 decoder with a trained one's architecture (D-095's control: does T15's training matter?).

    any env with PyTorch, CPU:
    python -I scripts/make_random_b2v2.py --like data/checkpoints/b2v2_all_gan/final.pt --seed 0 \
        --out data/checkpoints/b2v2_random_s0/final.pt

Builds ``wvx.b2v2_model.B2V2(model_cfg)`` of ``--like`` under ``torch.manual_seed(--seed)`` (PyTorch's default
initialization) and saves it in the trainers' checkpoint layout, so an adaptation can start from it as from a trained
base (``adapt.init``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--like", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import torch

    from wvx import b2v2_model as bm
    from wvx.provenance import code_version

    out = resolve_path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    like = torch.load(resolve_path(args.like), map_location="cpu", weights_only=False)
    cfg = dict(like["model_cfg"])
    torch.manual_seed(int(args.seed))
    model = bm.B2V2(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "step": 0, "best": None, "model_cfg": model.cfg, "input_id": 0,
                "config": None, "run_name": out.parent.name, "random_init": {"like": args.like, "seed": args.seed},
                "code_version": code_version()}, out)
    print(f"random b2v2 ({sum(p.numel() for p in model.parameters()):,} values, seed {args.seed}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
