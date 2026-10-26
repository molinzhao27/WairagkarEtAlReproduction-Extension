#!/usr/bin/env python
"""Another Kunz participant's 50-word sentence session in the b2v2 dataset format (D-095: T15's decoder for others).

    GPU host (the raw files are there), the syllable env (it has ``cmudict``), CPU:
    nice -n 10 /workspace/envs/wvx-syl/bin/python -I scripts/build_b2v2_kunz_other.py --session t12.2023.11.28 \
        --out data/processed/b2v2_t12

Reads ``<raw>/<session>_{attempted_50words,imagined_50words,imagined_50words_eval}.mat`` (Kunz et al., Dryad
doi:10.5061/dryad.gf1vhhn1j) with T15's preprocessing (``wvx.kunz.load_condition``, configs/data/kunz.yaml: log,
rolling 10 s z-score, causal smoothing; all its parameters are in ms, so each file is processed at its own bin size,
20 ms for T12/T16, 33 ms for T16's eval block). Then, per trial (the go window), the features are brought to T15's
10 ms bins (bin i <- the source bin covering its start, i.e. repetition for 20 ms) and each feature type is zero-padded
to 256 channels (T12 has 128: [spike-band power, 0, threshold crossings, 0]), so the 512-wide input of T15's decoder
takes them; a new, freshly initialized input projection learns the mapping (D-095). Channels do not correspond to
T15's. Labels: each word's phonemes as in T15's table (``--lexicon``; all CMUdict first pronunciations), else its
CMUdict first pronunciation (stress removed). Writes ``neural.f16``, ``target.f32`` (zeros), ``meta.json`` (trials as
scripts/build_b2v2_kunz.py writes them, plus the source bin size) and ``phonemes.json`` (the cues' words and phones).
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

CONDITIONS = {"attempted": "attempted_50words", "imagined": "imagined_50words", "imagined_eval": "imagined_50words_eval"}


def to_10ms(x, bin_ms: float):
    """(T, F) at ``bin_ms`` -> (round(T * bin_ms / 10), F): each 10 ms bin takes the source bin covering its start."""
    import numpy as np

    n = int(round(x.shape[0] * bin_ms / 10))
    idx = np.minimum((np.arange(n) * 10) // bin_ms, x.shape[0] - 1).astype(int)
    return x[idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="file prefix, e.g. t12.2023.11.28")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="data/raw/kunz/sentenceData_all/sentenceData")
    ap.add_argument("--data-config", default="configs/data/kunz.yaml")
    ap.add_argument("--lexicon", default="data/processed/b2v2_kunz/phonemes.json")
    ap.add_argument("--channels", type=int, default=256, help="per feature type, after zero-padding")
    args = ap.parse_args(argv)
    import cmudict
    import numpy as np
    import scipy.io

    from wvx import kunz, phonemes as ph, targets as tg
    from wvx.provenance import code_version

    adapter = load_yaml(resolve_path(args.data_config))["adapter"]
    out_dir = resolve_path(args.out)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists")
    lex = {}
    for c in json.loads(resolve_path(args.lexicon).read_text())["cues"].values():
        for w, p in zip(c["words"], c["phones"]):
            lex.setdefault(w, p)
    cmu = {w: prons[0] for w, prons in cmudict.dict().items() if prons}
    sources = {}

    def pron(w):
        if w in lex:
            sources["t15_table"] = sources.get("t15_table", 0) + 1
            return lex[w]
        if w in cmu:
            sources["cmudict"] = sources.get("cmudict", 0) + 1
            return [re.sub(r"\d", "", p) for p in cmu[w]]
        return None

    sess, bins = {}, {}
    for cond, suffix in CONDITIONS.items():
        path = resolve_path(args.raw) / f"{args.session}_{suffix}.mat"
        bms = int(np.asarray(scipy.io.loadmat(path, variable_names=["binSize"])["binSize"]).squeeze())
        sess[cond] = kunz.load_condition(path, f"{args.session}_{cond}", 1, dict(adapter, bin_ms=bms))
        bins[cond] = bms
    attempted = {}
    for t in sess["attempted"].trials:
        attempted.setdefault(t.block, set()).add(tg.text_key(t.text))
    out_dir.mkdir(parents=True)
    C = int(args.channels)
    fn, ft = open(out_dir / "neural.f16.partial", "wb"), open(out_dir / "target.f32.partial", "wb")
    trials, table, pos, gid, missing = [], {}, 0, 0, []
    for cond, s in sess.items():
        for i, t in enumerate(s.trials):
            x = np.asarray(t.neural, np.float32)
            f = x.shape[1] // 2                            # [spike-band power, threshold crossings]
            if f > C:
                raise SystemExit(f"{f} channels > {C}")
            pad = np.zeros((x.shape[0], C - f), np.float32)
            x = to_10ms(np.concatenate([x[:, :f], pad, x[:, f:], pad], axis=1), bins[cond])
            n = x.shape[0]
            fn.write(x.astype(np.float16).tobytes())
            ft.write(np.zeros((n, 20), np.float32).tobytes())
            key = tg.text_key(t.text)
            words = re.findall(r"[a-z']+", key.lower())
            prons = [pron(w) for w in words]
            if any(p is None for p in prons) or not words:
                missing.append(key)
                prons = None
            else:
                table[key] = {"words": words, "phones": prons}
                if not all(p in ph.ID for q in prons for p in q):
                    raise SystemExit(f"unknown phone in {key}: {prons}")
            paired = [ab for ab, cues in attempted.items() if key in cues]
            trials.append({"gid": gid, "session": cond, "day": None, "index": i, "block": int(t.block), "start": pos,
                           "length": n, "text": key, "type": cond, "paired_block": int(paired[0]) if paired else None,
                           "benchmark": None, "closed_loop_audio": False,
                           "label": ph.label(prons) if prons else None, "words": None, "word_labels": None,
                           "speech_span": [0, n], "source_bin_ms": bins[cond]})
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
            "decision": "D-095", "participant_session": args.session, "n_bins": pos, "n_features": 2 * C,
            "n_targets": 20, "neural_dtype": "float16", "target": "none (zeros)", "bin_ms_source": bins,
            "channels_source": f, "source": {c: f"{args.raw}/{args.session}_{s}.mat" for c, s in CONDITIONS.items()},
            "block_pairs": {str(bk): sorted(v) for bk, v in pairs.items()}, "cues_without_label": missing,
            "label_sources": sources, "trials": trials}
    (out_dir / "meta.json").write_text(json.dumps(meta) + "\n")
    (out_dir / "phonemes.json").write_text(json.dumps({"created": meta["created"], "lexicon": args.lexicon,
                                                       "sources": sources, "cues": table}, indent=1) + "\n")
    print(f"{args.session}: {gid} trials, {pos} bins (10 ms), bins {bins}, {f} channels -> {2 * C} features; "
          f"pairs {meta['block_pairs']}; {len(missing)} cues without a label; labels {sources} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
