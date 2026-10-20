"""Train the brain-to-voice decoder on prepared sessions (GPU host, or the Mac's CPU).

    python -I scripts/train.py --config configs/experiments/reproduce_training.yaml

Loads one SessionData file per entry of ``training.sessions`` from
``paths.training_sessions/<session>.npz``. Each session gets the ``session_layer`` from the config
(when one is given there), never its list position. Sessions in ``training.held_out_sessions`` are
refused. Then calls ``wvx.training.train``. Outputs go to ``<paths.checkpoints>/<run_name>/`` and
``<paths.results>/<run_name>/``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; keep third_party/ free of __pycache__

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx import b2v  # noqa: E402  (must come before TensorFlow)

b2v.use_keras2()

from wvx.config import load_config  # noqa: E402
from wvx.seeding import set_seed  # noqa: E402
from wvx.session import SessionData  # noqa: E402

log = logging.getLogger("train")


def load_sessions(cfg: dict) -> list[SessionData]:
    t = cfg["training"]
    held_out = {h["session"] for h in t.get("held_out_sessions") or []}
    src = Path(cfg["paths"]["training_sessions"])
    sessions = []
    for entry in t["sessions"]:
        name = entry["session"]
        if name in held_out:
            raise ValueError(f"{name} is listed both for training and as held out")
        s = SessionData.load(src / f"{name}.npz")
        if s.name != name:
            raise ValueError(f"{src / name}.npz holds session {s.name!r}")
        layer = entry.get("session_layer")
        if layer is not None and layer != s.layer:
            log.info("%s: using configured layer %d (file says %d)", name, layer, s.layer)
            s = dataclasses.replace(s, layer=layer)
        sessions.append(s)
        log.info("%s: %d trials, layer %d", name, len(s.trials), s.layer)
    return sessions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment YAML, merged over configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg["runtime"].get("deterministic_ops", False))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from wvx.training import train

    result = train(load_sessions(cfg), cfg)
    final = result.history.get(cfg["training"]["monitor"], [])
    print(f"run {result.run_name}: {len(final)} epochs, last {cfg['training']['monitor']} "
          f"{final[-1] if final else float('nan'):.4f}")
    print(f"checkpoints: {result.checkpoint_dir}")
    print(f"history:     {result.history_csv}")
    for layer, path in result.inference_models.items():
        print(f"inference layer {layer}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
