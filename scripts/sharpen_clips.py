#!/usr/bin/env python
"""Sharpened versions of decoded clips for listening (D-078 ``sharpen``; user: show original and sharpened).

    PC/host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/sharpen_clips.py --config configs/experiments/b2v2_kunz_base.yaml \
        --items b2v2_kunz_im03:imagined:216 b2v2_kunz_z:imagined_eval:401 ...

Each item is RUN:NAME:GID of a saved decode (``data/processed/b2v2_preds/RUN/NAME.npz``, ``g<gid>_cont``): the
features' cepstral deviations are scaled by ``--alpha`` (``wvx.postproc.sharpen``) and re-vocoded to
results/audio/b2v2_sharpen/RUN/NAME/g<gid>.wav (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_kunz_base.yaml")
    ap.add_argument("--items", nargs="+", required=True, help="RUN:NAME:GID")
    ap.add_argument("--alpha", type=float, default=1.5)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np

    from wvx import postproc as pp, synthesis

    for item in args.items:
        run, name, gid = item.split(":")
        with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False) as z:
            cont = z[f"g{gid}_cont"]
        wav = synthesis.decoder_output_to_waveform(pp.sharpen(cont, args.alpha), cfg)
        out = synthesis.write_wav(resolve_path("results/audio/b2v2_sharpen") / run / name / f"g{gid}.wav", wav, cfg)
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
