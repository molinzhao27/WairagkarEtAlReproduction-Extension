#!/usr/bin/env python
"""D-200: apply a trained post-net (``wvx.postnet``) to a decode that stores 10 ms phoneme log-probabilities.

    any env with PyTorch, CPU:
    python -I scripts/apply_postnet.py --postnet data/checkpoints/b2v2_postnet/postnet.pt \
        --run b2v2_kunz_d097_unfreeze_s0_k03 --pred imagined_l10 --out imagined_postnet

Reads data/processed/b2v2_preds/<run>/<pred>.npz (``g<gid>_cont`` and ``g<gid>_lp10``; ``decode_b2v2.py
--save-logits``) and writes <out>.npz next to it: the same arrays with each ``g<gid>_cont`` replaced by the post-net's
output (the phoneme read-out is unchanged), and the header noting the post-net.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--postnet", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import numpy as np
    import torch

    from wvx import postnet as pn

    torch.set_num_threads(4)
    ck = torch.load(resolve_path(args.postnet), map_location="cpu", weights_only=False)
    c = ck["cfg"]
    model = pn.PostNet(c["n_cont"], c["n_phon"], c["d"], c["dilations"], c["k"])
    model.load_state_dict(ck["state"])
    src = resolve_path("data/processed/b2v2_preds") / args.run / f"{args.pred}.npz"
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    header = json.loads(str(arrays.pop("header")))
    for g in header["gids"]:
        arrays[f"g{g}_cont"] = pn.run(model, arrays[f"g{g}_cont"], arrays[f"g{g}_lp10"].astype(np.float32))
    header["postnet"] = {"checkpoint": args.postnet, "sha256": hashlib.sha256(resolve_path(args.postnet).read_bytes()).hexdigest(),
                         "source": f"{args.run}/{args.pred}.npz"}
    dst = src.with_name(f"{args.out}.npz")
    tmp = dst.with_name(dst.stem + ".partial.npz")
    np.savez_compressed(tmp, header=np.array(json.dumps(header)), **arrays)
    tmp.replace(dst)
    print(f"{len(header['gids'])} trials -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
