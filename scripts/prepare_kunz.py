"""Convert the Kunz T15 2024.03.03 files into decoder-layout SessionData (.npz). Runs on the GPU host.

    python -I scripts/prepare_kunz.py [--config configs/data/kunz.yaml]

Reads data/raw/kunz/sentenceData/*.mat (see configs/data/kunz.yaml), writes one .npz per condition to
adapter.processed_dir and prints a summary. Targets are not generated here (Phase 2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import load_yaml, resolve_path  # noqa: E402
from wvx.kunz import load_condition  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/kunz.yaml")
    args = ap.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    adapter = cfg["adapter"]
    raw = resolve_path(cfg["output_dir"])
    out = resolve_path(adapter["processed_dir"])
    for cond, spec in adapter["conditions"].items():
        name = f"t15.{cfg['core_experiment']['session']}_{cond}"
        s = load_condition(raw / spec["file"], name, spec["layer"], adapter)
        s.validate(n_features=adapter["n_features"], n_targets=0)
        lengths = np.array([t.neural.shape[0] for t in s.trials])
        x = np.concatenate([t.neural for t in s.trials])
        s.save(out / f"{name}.npz")
        print(f"{name}: {len(s.trials)} trials, bins/trial {lengths.min()}-{lengths.max()} "
              f"(median {int(np.median(lengths))}), features {x.shape[1]}, "
              f"mean {x.mean():.3g}, sd {x.std():.3g} -> {(out / f'{name}.npz').relative_to(resolve_path('.'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
