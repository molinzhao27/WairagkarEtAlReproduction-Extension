#!/usr/bin/env python
"""Kunz T15 open-vocabulary (125k-word) inner-speech sentences in the b2v2 dataset format (D-081).

Two steps, host, CPU:
    1. phonemes (the syllable env has cmudict):
       /workspace/envs/wvx-syl/bin/python -I scripts/build_b2v2_openvocab.py --config configs/experiments/b2v2_ov_base.yaml --phonemes
    2. data (project venv):
       CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/build_b2v2_openvocab.py --config configs/experiments/b2v2_ov_base.yaml

Files in ``b2v2.openvocab.files`` (``name: path``), preprocessed by ``wvx.kunz.load_condition`` (D-023, release
channel order D-024; every trial = its go window). Phonemes: the first CMUdict pronunciation of every word (lower-case,
letters and apostrophes), each word followed by SIL; a sentence with a word outside CMUdict gets no label. Writes
``<b2v2.data_dir>/phonemes.json`` (step 1), then ``neural.f16``, ``target.f32`` (zeros), ``meta.json`` (step 2): per
trial gid, file name (as ``session``), block, cue, label, and their closed-loop decoded text (``decoded_text``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import load_yaml, resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_ov_base.yaml")
    ap.add_argument("--phonemes", action="store_true", help="step 1: write phonemes.json (needs cmudict)")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    import scipy.io

    b = cfg["b2v2"]
    ov = b["openvocab"]
    out_dir = resolve_path(b["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.phonemes:
        import cmudict

        cmu = cmudict.dict()
        cues = {}
        for name, path in ov["files"].items():
            m = scipy.io.loadmat(str(resolve_path(path)), squeeze_me=True, variable_names=["sentenceText"])
            for s in np.atleast_1d(m["sentenceText"]):
                text = " ".join(str(s).split())
                ws = words_of(text)
                prons = [[p.rstrip("012") for p in cmu[w][0]] if w in cmu else None for w in ws]
                cues[text] = {"words": ws, "phones": prons, "sources": ["cmudict" if p else "unknown" for p in prons]}
        (out_dir / "phonemes.json").write_text(json.dumps({"source": "cmudict first pronunciation (D-081)",
                                                           "cues": cues}, indent=1) + "\n")
        print(f"{len(cues)} sentences; {sum(any(p is None for p in c['phones']) for c in cues.values())} with a word "
              f"outside CMUdict -> {out_dir / 'phonemes.json'}")
        return 0

    from wvx import kunz, phonemes as ph, targets as tg
    from wvx.provenance import code_version

    table = json.loads((out_dir / "phonemes.json").read_text())["cues"]
    adapter = load_yaml(resolve_path(ov["adapter_config"]))["adapter"]
    fn, ft = open(out_dir / "neural.f16.partial", "wb"), open(out_dir / "target.f32.partial", "wb")
    trials, pos, gid, counts = [], 0, 0, {}
    for name, path in ov["files"].items():
        s = kunz.load_condition(resolve_path(path), name, 0, adapter)
        for t in s.trials:
            n = t.neural.shape[0]
            fn.write(np.asarray(t.neural, np.float16).tobytes())
            ft.write(np.zeros((n, int(b["model"]["n_targets"])), np.float32).tobytes())
            key = tg.text_key(t.text)
            entry = table.get(key)
            prons = entry["phones"] if entry and all(p is not None for p in entry["phones"]) else None
            trials.append({"gid": gid, "session": name, "day": None, "index": t.meta["trial_index"],
                           "block": int(t.block), "start": pos, "length": n, "text": key, "type": name,
                           "decoded_text": t.meta.get("decoded_text", ""), "benchmark": None,
                           "closed_loop_audio": False, "label": ph.label(prons) if prons else None, "words": None,
                           "word_labels": None, "speech_span": [0, n]})
            counts[name] = counts.get(name, 0) + 1
            gid += 1
            pos += n
    fn.close()
    ft.close()
    (out_dir / "neural.f16.partial").replace(out_dir / "neural.f16")
    (out_dir / "target.f32.partial").replace(out_dir / "target.f32")
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
            "decision": "D-081", "n_bins": pos, "n_features": 512, "n_targets": int(b["model"]["n_targets"]),
            "neural_dtype": "float16", "target": "none (zeros)", "trial_counts": counts,
            "unlabeled": sum(t["label"] is None for t in trials), "trials": trials}
    (out_dir / "meta.json").write_text(json.dumps(meta) + "\n")
    print(f"{gid} trials, {pos} bins; {counts}; {meta['unlabeled']} without a label -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
