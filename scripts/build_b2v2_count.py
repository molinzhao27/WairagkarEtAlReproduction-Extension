#!/usr/bin/env python
"""Kunz conjunctive counting (T15 2024-12-15) in the b2v2 dataset format (D-082).

    host, two steps:
    1. /workspace/envs/wvx-syl/bin/python -I scripts/build_b2v2_count.py --config configs/experiments/b2v2_count_base.yaml --phonemes
    2. CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/build_b2v2_count.py --config configs/experiments/b2v2_count_base.yaml

``wvx.kunz.load_counting`` (D-023 preprocessing, release channel order). Two trials per count: ``count_delay`` (the
silent counting, delay epoch; label = "one two ... n", each word + SIL) and ``count_go`` (the count said aloud, go
epoch; label = the number word + SIL). Number words: first CMUdict pronunciations (step 1, ``phonemes.json``).
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

NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_count_base.yaml")
    ap.add_argument("--phonemes", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np

    b = cfg["b2v2"]
    out_dir = resolve_path(b["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.phonemes:
        import cmudict

        d = cmudict.dict()
        table = {w: [p.rstrip("012") for p in d[w][0]] for w in NUMBERS}
        (out_dir / "phonemes.json").write_text(json.dumps({"source": "cmudict first pronunciation (D-082)",
                                                           "numbers": table}, indent=1) + "\n")
        print(table)
        return 0

    from wvx import kunz, phonemes as ph
    from wvx.provenance import code_version

    nums = json.loads((out_dir / "phonemes.json").read_text())["numbers"]
    adapter = load_yaml(resolve_path(b["count"]["adapter_config"]))["adapter"]
    s = kunz.load_counting(resolve_path(b["count"]["file"]), "count", adapter)
    fn, ft = open(out_dir / "neural.f16.partial", "wb"), open(out_dir / "target.f32.partial", "wb")
    trials, pos, gid = [], 0, 0
    for t in s.trials:
        n, g = t.meta["n"], t.meta["go_offset"]
        for kind, x, label, text in (("count_delay", t.neural[:g], ph.label([nums[w] for w in NUMBERS[:n]]),
                                      " ".join(NUMBERS[:n])),
                                     ("count_go", t.neural[g:], ph.label([nums[NUMBERS[n - 1]]]), NUMBERS[n - 1])):
            fn.write(np.asarray(x, np.float16).tobytes())
            ft.write(np.zeros((len(x), int(b["model"]["n_targets"])), np.float32).tobytes())
            trials.append({"gid": gid, "session": kind, "day": None, "index": t.meta["trial_index"], "block": t.block,
                           "start": pos, "length": int(len(x)), "text": text, "n": n, "type": kind, "benchmark": None,
                           "closed_loop_audio": False, "label": label, "words": None, "word_labels": None,
                           "speech_span": [0, int(len(x))]})
            gid += 1
            pos += len(x)
    fn.close()
    ft.close()
    (out_dir / "neural.f16.partial").replace(out_dir / "neural.f16")
    (out_dir / "target.f32.partial").replace(out_dir / "target.f32")
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
            "decision": "D-082", "n_bins": pos, "n_features": 512, "n_targets": int(b["model"]["n_targets"]),
            "neural_dtype": "float16", "target": "none (zeros)", "numbers": nums, "trials": trials}
    (out_dir / "meta.json").write_text(json.dumps(meta) + "\n")
    print(f"{gid} trials ({len(s.trials)} counts), {pos} bins -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
