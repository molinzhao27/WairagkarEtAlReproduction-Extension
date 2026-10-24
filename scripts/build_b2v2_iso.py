#!/usr/bin/env python
"""Kunz isolated verbal behaviors (T15) in the b2v2 dataset format (D-079).

    host, project venv, CPU (after data/processed/b2v2_iso/phonemes.json exists):
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/build_b2v2_iso.py --config configs/experiments/b2v2_iso_base.yaml

Each behavior file is preprocessed by ``wvx.kunz.load_isolated`` (D-023, configs/data/kunz.yaml) and each trial is
cut to its D-079 window: the go epoch for the active behaviors, delay + go for the passive ones (listening, reading).
Writes ``<b2v2.data_dir>/neural.f16``, ``target.f32`` (zeros) and ``meta.json``: per trial its gid, behavior (as
``session``), block, word (``text``, "" for DO_NOTHING), phoneme label (word + SIL; [] for DO_NOTHING), window.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import load_yaml, resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_iso_base.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np

    from wvx import kunz, phonemes as ph
    from wvx.provenance import code_version

    b = cfg["b2v2"]
    iso = b["iso"]
    out_dir = resolve_path(b["data_dir"])
    table = json.loads((out_dir / "phonemes.json").read_text())["cues"]
    adapter = load_yaml(resolve_path(iso["adapter_config"]))["adapter"]
    fn, ft = open(out_dir / "neural.f16.partial", "wb"), open(out_dir / "target.f32.partial", "wb")
    trials, pos, gid, counts = [], 0, 0, {}
    for beh in iso["behaviors"]:
        s = kunz.load_isolated(resolve_path(iso["raw_dir"]) / f"{iso['session']}_{beh}_raw.mat", beh, 0, adapter)
        window = "delay+go" if beh in iso["passive"] else "go"
        for t in s.trials:
            x = t.neural if window == "delay+go" else t.neural[t.meta["go_offset"]:]
            n = x.shape[0]
            fn.write(np.asarray(x, np.float16).tobytes())
            ft.write(np.zeros((n, int(b["model"]["n_targets"])), np.float32).tobytes())
            label = ph.label(table[t.text]["phones"]) if t.text else []
            trials.append({"gid": gid, "session": beh, "day": None, "index": t.meta["trial_index"], "block": t.block,
                           "start": pos, "length": n, "text": t.text, "type": beh, "window": window,
                           "cue": t.meta["cue"], "benchmark": None, "closed_loop_audio": False, "label": label,
                           "words": None, "word_labels": None, "speech_span": [0, n]})
            counts[beh] = counts.get(beh, 0) + 1
            gid += 1
            pos += n
    fn.close()
    ft.close()
    (out_dir / "neural.f16.partial").replace(out_dir / "neural.f16")
    (out_dir / "target.f32.partial").replace(out_dir / "target.f32")
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
            "decision": "D-079", "n_bins": pos, "n_features": 512, "n_targets": int(b["model"]["n_targets"]),
            "neural_dtype": "float16", "target": "none (zeros)", "session": iso["session"], "trial_counts": counts,
            "trials": trials}
    (out_dir / "meta.json").write_text(json.dumps(meta) + "\n")
    print(f"{gid} trials, {pos} bins; {counts} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
