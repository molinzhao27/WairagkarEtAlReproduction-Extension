#!/usr/bin/env python
"""Decoder-output parity between two prediction files of the SAME model and inputs (e.g. the spare PC's GPU decode
vs the host's CPU decode; scripts/kunz_experiment.py writes results/audio/<run>/<fold>/predictions/<model>__<input>.npz).

    .venv/bin/python -I scripts/compare_predictions.py REFERENCE.npz OTHER.npz [--max-rel 1e-3] [--min-r 0.9999] \
        [--require-other-device gpu]

Checks that both headers name the same model file (SHA-256) and decode settings (older files: model, input, channel
reorder and fold), then compares every trial both files hold: max |other - reference| / max |reference| and the
correlation of the flattened features. Exits 1 if the two are the same file (path or bytes), a header differs, a
reference trial is missing from OTHER, a shape differs, a value is not finite, any trial is outside the tolerances,
or OTHER's header does not name --require-other-device (``runtime_device``, written by kunz_experiment.py).
Small, numpy only. Two CPU decodes on the host (12 vs 8 threads) are bit-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np


def load(path: str) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["header"])), {k: z[k] for k in z.files if k != "header"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("other")
    ap.add_argument("--max-rel", type=float, default=1e-3, help="per trial: max|other - ref| / max|ref|")
    ap.add_argument("--min-r", type=float, default=0.9999, help="per trial: correlation of the flattened features")
    ap.add_argument("--require-other-device", default=None, help="OTHER must have been decoded on this runtime.device")
    args = ap.parse_args(argv)

    if os.path.realpath(args.reference) == os.path.realpath(args.other):
        raise SystemExit("REFERENCE and OTHER are the same file")
    digest = [hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (args.reference, args.other)]
    if digest[0] == digest[1]:
        raise SystemExit("REFERENCE and OTHER have identical bytes (a copy, not a second decode)")
    ha, a = load(args.reference)
    hb, b = load(args.other)
    # Files written before decode_settings existed (e.g. the D-030 run) carry only model, input and channel reorder.
    keys = ("model_sha256", "decode_settings") if "decode_settings" in ha and "decode_settings" in hb else \
        ("model_sha256", "input", "reorder_channels", "fold")
    problems = [f"header {k}: {ha.get(k)!r} != {hb.get(k)!r}" for k in keys if ha.get(k) != hb.get(k)]
    if args.require_other_device and hb.get("runtime_device") != args.require_other_device:
        problems.append(f"OTHER was decoded on runtime_device {hb.get('runtime_device')!r}, not {args.require_other_device!r}")
    shared = sorted(set(a) & set(b))
    if not shared:
        problems.append("no trial in both files")
    if set(a) - set(b):
        problems.append(f"{len(set(a) - set(b))} reference trials missing from OTHER")
    rel, r = [], []
    for k in shared:
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        if x.shape != y.shape:
            problems.append(f"{k}: shape {x.shape} != {y.shape}")
            continue
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            problems.append(f"{k}: non-finite values")
        rel.append(float(np.abs(y - x).max() / max(np.abs(x).max(), 1e-12)))
        r.append(float(np.corrcoef(x.ravel(), y.ravel())[0, 1]))
    rel_a, r_a = np.array(rel), np.array(r)
    out = {"reference": args.reference, "other": args.other, "model_sha256": ha.get("model_sha256"),
           "runtime_devices": [ha.get("runtime_device"), hb.get("runtime_device")],
           "header_keys_compared": list(keys), "code_versions": [ha.get("code_version"), hb.get("code_version")],
           "n_shared": len(shared), "n_only_reference": len(set(a) - set(b)), "n_only_other": len(set(b) - set(a))}
    if len(rel_a):
        out |= {"max_rel_diff": {"max": float(rel_a.max()), "median": float(np.median(rel_a))},
                "r": {"min": float(r_a.min()), "median": float(np.median(r_a))},
                "n_outside": int((~(rel_a <= args.max_rel) | ~(r_a >= args.min_r)).sum()),   # NaN counts as outside
                "tolerance": {"max_rel": args.max_rel, "min_r": args.min_r}}
        if out["n_outside"]:
            problems.append(f"{out['n_outside']} of {len(rel_a)} trials outside the tolerances")
    out["problems"] = problems
    out["pass"] = not problems
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
