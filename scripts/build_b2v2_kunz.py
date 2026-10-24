#!/usr/bin/env python
"""Kunz T15 2024.03.03 in the b2v2 dataset format (D-076), after scripts/b2v2_phonemes.py with the same config.

    PC or host, project venv, CPU:
    .venv/bin/python -I scripts/build_b2v2_kunz.py --config configs/experiments/b2v2_kunz_base.yaml

Reads ``<b2v2.kunz.processed>/<session>_<condition>.npz`` (``wvx.session.SessionData``; neural features as processed
for the old pipeline, D-023, release channel order, D-024; every trial is its go window) and writes
``<b2v2.data_dir>/neural.f16``, ``target.f32`` (zeros: Kunz has no target audio) and ``meta.json``: per trial its gid,
condition (as ``session``), block, the attempted block holding the same sentences (``paired_block``), cue, phoneme
label (``phonemes.json``), ``speech_span`` = the whole trial; no word timings.
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
    ap.add_argument("--config", default="configs/experiments/b2v2_kunz_base.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np

    from wvx import phonemes as ph, targets as tg
    from wvx.provenance import code_version
    from wvx.session import SessionData

    b = cfg["b2v2"]
    k = b["kunz"]
    out_dir = resolve_path(b["data_dir"])
    table = json.loads((out_dir / "phonemes.json").read_text())["cues"]
    sess = {c: SessionData.load(resolve_path(k["processed"]) / f"{k['session']}_{c}.npz") for c in k["conditions"]}
    attempted = {}
    for t in sess["attempted"].trials:
        attempted.setdefault(t.block, set()).add(tg.text_key(t.text))
    fn, ft = open(out_dir / "neural.f16.partial", "wb"), open(out_dir / "target.f32.partial", "wb")
    trials, pos, gid, missing = [], 0, 0, []
    for c, s in sess.items():
        for i, t in enumerate(s.trials):
            n = t.neural.shape[0]
            fn.write(np.asarray(t.neural, np.float16).tobytes())
            ft.write(np.zeros((n, int(b["model"]["n_targets"])), np.float32).tobytes())
            key = tg.text_key(t.text)
            entry = table.get(key)
            prons = entry["phones"] if entry and all(p is not None for p in entry["phones"]) else None
            if prons is None:
                missing.append(key)
            paired = [ab for ab, cues in attempted.items() if key in cues]
            trials.append({"gid": gid, "session": c, "day": None, "index": i, "block": int(t.block), "start": pos,
                           "length": n, "text": key, "type": c, "paired_block": int(paired[0]) if paired else None,
                           "benchmark": None, "closed_loop_audio": False,
                           "label": ph.label(prons) if prons else None, "words": None, "word_labels": None,
                           "speech_span": [0, n]})
            gid += 1
            pos += n
    fn.close()
    ft.close()
    (out_dir / "neural.f16.partial").replace(out_dir / "neural.f16")
    (out_dir / "target.f32.partial").replace(out_dir / "target.f32")
    pairs = {}
    for t in trials:
        if t["session"] != "attempted" and t["paired_block"] is not None:
            pairs.setdefault(t["block"], set()).add(t["paired_block"])
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
            "decision": "D-076", "n_bins": pos, "n_features": int(sess["attempted"].trials[0].neural.shape[1]),
            "n_targets": int(b["model"]["n_targets"]), "neural_dtype": "float16", "target": "none (zeros)",
            "source": {c: str(resolve_path(k["processed"]) / f"{k['session']}_{c}.npz") for c in k["conditions"]},
            "block_pairs": {str(bk): sorted(v) for bk, v in pairs.items()}, "cues_without_label": missing,
            "trials": trials}
    (out_dir / "meta.json").write_text(json.dumps(meta) + "\n")
    print(f"{gid} trials, {pos} bins; pairs {meta['block_pairs']}; {len(missing)} cues without a label -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
