#!/usr/bin/env python
"""Dataset for the brain2voice-2.0-style decoder (D-075): every trial of all 37 Wairagkar sessions, one directory.

Host, project venv, CPU, after scripts/b2v2_phonemes.py:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/build_b2v2_data.py --config configs/experiments/b2v2_base.yaml

Sessions in their order (training_data_args.yaml, then the held-out t15_day195): this order is their global trial
indexing (``test_idx.mat``); checked: day165-day193 are [5770, 8184) (their training notebook) and 8,489 trials in
all. Per trial: neural features as released ([spike-band power | threshold crossings], 10 ms, float16), their LPCNet
targets scaled to [-1, 1] WITHOUT smoothing (``wairagkar_data.target_smoothing: false``), the cue's phoneme label
(``phonemes.json``) and its word timings (``word_indices``, 0-based [onset, offset) bins): per-word labels when the
timings are well-formed and as many as the cue's words; else only the speech span (well-formed, other count); else
none. Also the benchmark set (``random`` / ``video``) and whether the session file holds the closed-loop audio.
Writes ``<b2v2.data_dir>/neural.f16``, ``target.f32``, ``meta.json`` (``wvx.b2v2_data.B2V2Data``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[build_b2v2 {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def wellformed(w, n_bins: int) -> bool:
    import numpy as np

    w = np.asarray(w).reshape(-1, 2)
    return bool(len(w)) and bool((w[:, 0] >= 0).all() and (w[:, 1] <= n_bins).all() and (w[:, 0] < w[:, 1]).all()
                                 and (w[1:, 0] >= w[:-1, 1]).all())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_base.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np
    import scipy.io
    import yaml

    from wvx import phonemes as ph, targets as tg, wairagkar_data
    from wvx.provenance import code_version

    b = cfg["b2v2"]
    if cfg["wairagkar_data"]["target_smoothing"]:
        raise SystemExit("b2v2 data uses unsmoothed targets (wairagkar_data.target_smoothing: false)")
    t0 = time.time()
    data_dir = resolve_path(cfg["paths"]["wairagkar_neural_data"])
    out_dir = resolve_path(b["data_dir"])
    table = json.loads((out_dir / "phonemes.json").read_text())["cues"]
    sessions = [e["session"] for e in yaml.safe_load(open(resolve_path(cfg["paths"]["b2v_session_layers"])))["sessions"]]
    sessions += list(cfg["wairagkar_data"]["held_out_session_layers"])
    bm = scipy.io.loadmat(str(resolve_path(b["benchmark_file"])), squeeze_me=True)
    bench = {int(g): k.replace("_test_idx", "") for k in b["benchmark_keys"] for g in np.atleast_1d(bm[k])}

    fn, ft = open(out_dir / "neural.f16.partial", "wb"), open(out_dir / "target.f32.partial", "wb")
    trials, ranges, gid, pos = [], {}, 0, 0
    counts = {"per_word": 0, "span_only": 0, "no_timing": 0, "no_label": 0}
    c0, c0_in, c0_out, nf = [], [], [], None
    for name in sessions:
        sess = wairagkar_data.load_session(data_dir, name, 0, cfg)
        neural_path = wairagkar_data.session_files(data_dir, name, cfg)[0]
        names = {v[0] for v in scipy.io.whosmat(str(neural_path))}
        desc = None
        if b["description_key"] in names:
            desc = [str(x).strip() for x in np.atleast_1d(scipy.io.loadmat(
                str(neural_path), squeeze_me=True, variable_names=[b["description_key"]])[b["description_key"]])]
        si = data_dir / name / b["speech_indices_file"].format(session=name)
        wi = [np.asarray(x, np.int64).reshape(-1, 2) for x in np.atleast_1d(
            scipy.io.loadmat(str(si), squeeze_me=True, variable_names=[b["word_key"]])[b["word_key"]])]
        if len(wi) != len(sess.trials):
            raise ValueError(f"{name}: {len(wi)} word timings for {len(sess.trials)} trials")
        ranges[name] = [gid, gid + len(sess.trials)]
        for i, t in enumerate(sess.trials):
            n, nf = t.neural.shape[0], t.neural.shape[1]
            fn.write(np.asarray(t.neural, np.float16).tobytes())
            ft.write(np.asarray(t.target, np.float32).tobytes())
            c0.append(np.asarray(t.target[::10, 0], np.float32))
            entry = table.get(tg.text_key(t.text))
            prons = entry["phones"] if entry and all(p is not None for p in entry["phones"]) else None
            rec = {"gid": gid, "session": name, "day": int(name.split("day")[1]), "index": i, "start": pos,
                   "length": n, "text": tg.text_key(t.text), "type": desc[i] if desc else None,
                   "benchmark": bench.get(gid), "closed_loop_audio": b["closed_loop_key"] in names,
                   "label": ph.label(prons) if prons else None, "words": None, "word_labels": None,
                   "speech_span": None}
            w = wi[i]
            if prons is None:
                counts["no_label"] += 1
            elif wellformed(w, n) and len(w) == len(prons):
                rec["words"], rec["word_labels"] = w.tolist(), [ph.label([p]) for p in prons]
                rec["speech_span"] = [int(w[0, 0]), int(w[-1, 1])]
                counts["per_word"] += 1
                inw = np.zeros(n, bool)
                for on, off in w:
                    inw[on:off] = True
                c0_in.append(np.asarray(t.target[inw, 0][::10], np.float32))
                c0_out.append(np.asarray(t.target[~inw, 0][::10], np.float32))
            elif wellformed(w, n):
                rec["speech_span"] = [int(w[0, 0]), int(w[-1, 1])]
                counts["span_only"] += 1
            else:
                counts["no_timing"] += 1
            trials.append(rec)
            gid += 1
            pos += n
        log(f"{name}: {len(sess.trials)} trials, gid [{ranges[name][0]}, {ranges[name][1]}), "
            f"{sum(len(x.neural) for x in sess.trials)} bins")
        del sess
    fn.close()
    ft.close()
    # their indexing: day165-day193 are [5770, 8184) (training.ipynb), 8,489 trials in all
    i165, i193 = sessions.index("t15_day165"), sessions.index("t15_day193")
    if (ranges["t15_day165"][0], ranges["t15_day193"][1]) != (5770, 8184) or gid != 8489 or i193 < i165:
        raise SystemExit(f"global trial index check failed: day165 starts {ranges['t15_day165'][0]}, day193 ends "
                         f"{ranges['t15_day193'][1]}, {gid} trials")
    missing = sorted(set(bench) - {t["gid"] for t in trials})
    if missing:
        raise SystemExit(f"benchmark trials not found: {missing[:10]}")
    (out_dir / "neural.f16.partial").replace(out_dir / "neural.f16")
    (out_dir / "target.f32.partial").replace(out_dir / "target.f32")
    c0, c0_in, c0_out = np.concatenate(c0), np.concatenate(c0_in), np.concatenate(c0_out)
    qs = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    lengths = np.array([t["length"] for t in trials])
    bt = [t for t in trials if t["benchmark"]]
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
            "decision": "D-075", "n_bins": int(pos), "n_features": int(nf), "n_targets": 20,
            "neural_dtype": "float16", "target": "their LPCNet targets scaled to [-1, 1], unsmoothed",
            "sessions": ranges, "label_counts": counts,
            "benchmark": {"n": len(bt), "by_set": {k: sum(t["benchmark"] == k for t in bt) for k in ("random", "video")},
                          "with_closed_loop_audio": sum(t["closed_loop_audio"] for t in bt),
                          "with_label": sum(t["label"] is not None for t in bt)},
            "target_c0_percentiles": {k: {str(q): float(np.percentile(v, q)) for q in qs}
                                      for k, v in (("all", c0), ("inside_words", c0_in), ("outside_words", c0_out))},
            "trial_length_bins": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                                  "max": int(lengths.max())},
            "wall_time_s": time.time() - t0, "trials": trials}
    (out_dir / "meta.json").write_text(json.dumps(meta) + "\n")
    log(f"done: {gid} trials, {pos} bins, labels {counts}, benchmark {meta['benchmark']}, "
        f"{(time.time() - t0) / 60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
