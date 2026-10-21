"""Time one training epoch on real trials and project the Kunz experiment's epoch time.

    python -I scripts/time_training.py [--config configs/experiments/time_training.yaml]

Runs wvx.training.train for one epoch per variant (frozen vs fine-tuned base) on a slice of a
Wairagkar session, starting from a released decoder. Prints windows/s and the projected epoch
time for the configured Kunz training set. Uses whatever device TensorFlow sees (CPU if
CUDA_VISIBLE_DEVICES is empty). Outputs go to scratch run directories under the configured paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.dont_write_bytecode = True

from wvx import b2v  # noqa: E402,F401  (must precede TensorFlow)
from wvx.config import deep_merge, load_config, load_yaml, resolve_path  # noqa: E402
from wvx.provenance import code_version, config_sha256  # noqa: E402
from wvx.seeding import set_seed  # noqa: E402
from wvx.session import SessionData  # noqa: E402
from wvx.training import n_windows, train  # noqa: E402
from wvx.wairagkar_data import load_session  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/time_training.yaml")
    args = ap.parse_args()
    cfg = load_config("configs/experiments/reproduce_training.yaml",
                      overrides=load_yaml(resolve_path(args.config)))
    set_seed(cfg["seed"])
    b = cfg["benchmark"]
    m = cfg["model"]
    icfg = load_config("configs/experiments/reproduce_inference.yaml")   # owns the data layout
    full = load_session(icfg["paths"]["wairagkar_neural_data"], b["session"], b["layer"], icfg)
    sess = SessionData(full.name, b["layer"], full.trials[: b["n_trials"]])
    cfg["training"]["sessions"] = [{"session": sess.name, "session_layer": b["layer"]}]
    n_aug = cfg["training"]["n_augmentations"]

    report = {"device": None, "variants": {}, "code_version": code_version(), "config_sha256": config_sha256(cfg)}
    for variant in b["variants"]:
        vcfg = deep_merge(cfg, {
            "run_name": f"{cfg['run_name']}_{variant}",
            "training": {"init": {"freeze_base": variant == "frozen_base"}},
        })
        t0 = time.time()
        result = train([sess], vcfg)
        wall = time.time() - t0
        split = json.loads(Path(result.split_json).read_text()) if getattr(result, "split_json", None) else {}
        train_trials = [t for i, t in enumerate(sess.trials)
                        if i not in set(split.get("val", {}).get(sess.name, []))] or sess.trials
        win = sum(n_windows(len(t.neural), m["window_len"], m["step_size"]) for t in train_trials)
        win_total = win * (1 + n_aug)
        rate = win_total / wall
        proj_win = b["project_n_trials"] * (b["project_mean_bins"] - m["window_len"] + 1) * (1 + n_aug)
        report["variants"][variant] = {
            "wall_s": round(wall, 1), "train_windows_incl_aug": win_total,
            "windows_per_s": round(rate, 1),
            "projected_kunz_epoch_min": round(proj_win / rate / 60, 1),
            "projected_kunz_windows_per_epoch": proj_win,
        }
        print(variant, report["variants"][variant], flush=True)

    import tensorflow as tf
    report["device"] = [d.name for d in tf.config.list_logical_devices()]
    out = resolve_path(cfg["paths"]["results"]) / cfg["run_name"] / "timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(resolve_path('.'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
