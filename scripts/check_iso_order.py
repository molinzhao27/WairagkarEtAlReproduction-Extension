#!/usr/bin/env python
"""Channel order of the Kunz isolated-verbal-behaviors files (D-079), by D-024's test: channel-correlation-matrix
similarity to data whose order is known.

    host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/check_iso_order.py --config configs/experiments/reproduce_inference.yaml \
        --file data/raw/kunz/verbalBehaviors/isolatedVerbalBehaviors/t15.2024.04.07_attempted_raw.mat

Spike-band-power channel-correlation matrices (50 ms frames, all 256 channels) of the file, preprocessed as
``wvx.kunz.load_isolated`` (configs/data/kunz.yaml), vs (a) Kunz T15 2024.03.03 attempted (D-024: release order)
and (b) Wairagkar t15_day195: as-is, under their ``reorder_channels`` mapping, and under 200 random within-array
shuffles (99th percentile). Prints the similarities (Pearson r of the upper triangles).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx import b2v  # noqa: E402  (before TensorFlow: reorder_channels uses their data_utils)

b2v.use_keras2()

from wvx.config import load_yaml, resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/reproduce_inference.yaml")
    ap.add_argument("--file", required=True)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np

    from wvx import inference, kunz, neural_match as nm, wairagkar_data
    from wvx.session import SessionData

    adapter = load_yaml(resolve_path("configs/data/kunz.yaml"))["adapter"]
    iso = kunz.load_isolated(resolve_path(args.file), "iso", 0, adapter)
    ref_k = SessionData.load(resolve_path("data/processed/kunz/t15.2024.03.03_attempted.npz"))
    ref_w = wairagkar_data.load_session(resolve_path(cfg["paths"]["wairagkar_neural_data"]), "t15_day195", 0, cfg)

    def frames(trials):
        return np.concatenate([nm.pool_frames(np.asarray(t.neural, np.float64), 5) for t in trials])[:, :256]

    def tri(c):
        return c[np.triu_indices(256, 1)]

    def sim(a, b):
        u, v = tri(np.corrcoef(a, rowvar=False)), tri(np.corrcoef(b, rowvar=False))
        ok = np.isfinite(u) & np.isfinite(v)
        return float(np.corrcoef(u[ok], v[ok])[0, 1])

    x = frames(iso.trials)
    rk, rw = frames(ref_k.trials), frames(ref_w.trials)
    reord = inference.reorder_channels(np.concatenate([x, x], 1), cfg["paths"]["electrode_mapping"])[:, :256]
    rng = np.random.default_rng(0)
    shuf_k, shuf_w = [], []
    for _ in range(200):
        perm = np.concatenate([a * 64 + rng.permutation(64) for a in range(4)])
        shuf_k.append(sim(x[:, perm], rk))
        shuf_w.append(sim(x[:, perm], rw))
    print(f"{Path(args.file).name}: {len(iso.trials)} trials, {len(x)} frames")
    print(f"  vs Kunz 2024.03.03 attempted: as-is {sim(x, rk):.3f}; their reorder mapping {sim(reord, rk):.3f}; "
          f"within-array shuffles 99th pct {np.percentile(shuf_k, 99):.3f}")
    print(f"  vs Wairagkar t15_day195:      as-is {sim(x, rw):.3f}; their reorder mapping {sim(reord, rw):.3f}; "
          f"within-array shuffles 99th pct {np.percentile(shuf_w, 99):.3f}")
    print(f"  (reference: Kunz 2024.03.03 attempted vs day195 as-is {sim(rk, rw):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
