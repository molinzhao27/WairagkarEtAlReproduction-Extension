#!/usr/bin/env python
"""Same-day neural matching, imagined -> attempted, no decoder (pre-registered, notes/DECISIONS.md D-032). Host, CPU.

    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/neural_match.py \\
        [--config configs/experiments/neural_match.yaml]

Reads the processed Kunz T15 2024.03.03 attempted and imagined inputs (``SessionData`` .npz from
scripts/prepare_kunz.py; trials start at the go cue) and the cue TTS LPCNet features (scripts/make_tts.py),
runs ``wvx.neural_match.analyze`` and writes ``<neural_match.output_dir>/``:

- ``per_trial.csv``: one row per imagined trial (own attempted trial, cue TTS duration, lengths, the
  similarity to the own trial and its rank per measure and candidate set, the length-matched candidates);
- ``summary.json``: settings, deviations from and additions to the pre-registration (the block-normalized
  length baseline, added 2026-10-21 before any run), pooled and per-pair statistics, PCA,
  limitations, code version (``wvx.provenance``), config SHA-256, input SHA-256s, runtime;
- ``summary.md``: the same, readable.

Thread caps (``runtime.threads``) are applied before numpy loads and the process lowers its own priority to
``runtime.nice``. No TensorFlow (its import is blocked), no LPCNet, no GPU.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import REPO_ROOT, resolve_path  # noqa: E402  (yaml only)
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402  (no numpy)

DEFAULT_CONFIG = "configs/experiments/neural_match.yaml"
QUESTION = ("On the same day, does the neural activity of an IMAGINED trial resemble the ATTEMPTED trial of "
            "the same sentence more than attempted trials of other sentences?")


def log(msg: str) -> None:
    print(f"[neural_match {datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return _rel(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and obj != obj:
        return None  # NaN -> null
    return obj


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def cue_durations(texts, cfg) -> tuple[dict, dict]:
    """{text_key: TTS speech seconds} for the given cues, and a record of where they came from."""
    from wvx import neural_match as nm
    from wvx import targets

    cd = cfg["neural_match"]["cue_duration"]
    voice = cd["tts_voice"] or cfg["tts"]["voice"]
    corpus_name = cd["tts_corpus"] or cfg["kunz"]["corpus"]
    path = targets.corpus_path(cfg, voice, corpus_name)
    corpus = targets.TTSCorpus.load(path)
    missing = sorted({t for t in texts if t not in corpus})
    if missing:
        raise KeyError(f"{len(missing)} cues have no TTS features in {_rel(path)}, e.g. {missing[:3]}")
    out = {targets.text_key(t): nm.tts_speech_seconds(corpus.lookup(t), cfg, cd["method"], float(cd["threshold_db"]))
           for t in sorted(set(texts))}
    rec = {"method": cd["method"], "threshold_db": cd["threshold_db"], "voice": voice, "corpus": corpus_name,
           "file": _rel(path), "sha256": _sha256(path), "n_cues": len(out)}
    return out, rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)

    cfg = load_experiment_config(args.config)
    env_set = apply_thread_env(cfg)  # before numpy
    nice = cfg["runtime"].get("nice")
    if nice is not None and os.nice(0) < int(nice):  # absolute niceness (only ever raised)
        os.nice(int(nice) - os.nice(0))

    # Nothing here uses TensorFlow. Block its import so wvx.seeding.set_seed seeds Python and numpy only and
    # this process never loads TensorFlow or starts its thread pools on the shared host.
    sys.modules.setdefault("tensorflow", None)

    import numpy as np

    from wvx import neural_match as nm
    from wvx.kunz_experiment import describe
    from wvx.metrics import normalize_text
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed
    from wvx.session import SessionData
    from wvx.targets import text_key

    seed = int(cfg["seed"])
    set_seed(seed)
    s = cfg["neural_match"]
    t0 = time.time()
    fmt = {k: str(v) for k, v in cfg["paths"].items()}
    files = {c: resolve_path(s[f"{c}_file"].format(**fmt)) for c in ("attempted", "imagined")}
    sessions = {}
    for c, p in files.items():
        sessions[c] = SessionData.load(p)
        log(f"loaded {_rel(p)}: {sessions[c].name}, {len(sessions[c].trials)} trials")

    texts = [t.text for sess in sessions.values() for t in sess.trials]
    durations, dur_rec = cue_durations(texts, cfg)
    vals = np.array(list(durations.values()))
    dur_rec.update(seconds=describe(vals), min_s=float(vals.min()), max_s=float(vals.max()))
    log(f"cue TTS speech durations ({dur_rec['method']}): {len(vals)} cues, {vals.min():.2f}-{vals.max():.2f} s")

    def match(t):
        go = int(t.meta.get("go_offset", 0))       # trials start at the go cue (pre_go_bins 0); kept general
        return nm.MatchTrial(x=t.neural[go:], key=normalize_text(t.text), text=t.text, block=int(t.block),
                             trial_index=int(t.meta["trial_index"]), cue_s=durations[text_key(t.text)])

    att = [match(t) for t in sessions["attempted"].trials]
    img = [match(t) for t in sessions["imagined"].trials]
    rows, results = nm.analyze(att, img, s["pairs"], s, seed, log=log)

    settings = {k: v for k, v in s.items() if k != "preregistered"}
    summary = {
        "analysis": "neural_match",
        "question": QUESTION,
        "preregistration": "notes/DECISIONS.md D-032",
        "direction": "imagined -> attempted",
        "seed": seed,
        "settings": settings,
        "deviations_from_preregistration": nm.deviations(settings, s["preregistered"]),
        "additions_to_preregistration": nm.additions(settings),
        "implementation_choices": [
            "Go window = every bin from the go cue (meta go_offset) to the trial end.",
            f"Trailing partial pooling window dropped: {s['drop_partial_window']}.",
            f"Cue TTS speech duration: {dur_rec['method']} on the TTS LPCNet c0 (threshold {dur_rec['threshold_db']} "
            "dB below the loudest frame); no LPCNet synthesis.",
            "Length-matched candidates: ties in |TTS duration difference| go to the earlier attempted trial.",
            "Secondary similarity uses the unpooled go-window bins; length-only baseline uses go-window bins.",
            "Ranks: 1 = most similar; ties share the mid-rank; top-1 = rank exactly 1 (a tie for best is not "
            "top-1). Candidates with an undefined similarity would be left out (counted as n_undefined).",
            "p_perm = (1 + #{null mean rank <= observed}) / (1 + n_permutations), one-sided; the same draws give "
            "pooled and per-pair p; each measure x candidate set has its own seeded stream. p_exact: the same "
            "null by convolution (check).",
            "PCA axes from the covariance eigendecomposition of the standardized training frames; the projection "
            "uses the training mean and s.d. (no re-standardization of the components).",
        ],
        "cue_duration": dur_rec,
        "data": {
            "attempted_trials": len(att), "imagined_trials": len(img),
            "go_bins": {"attempted": describe([len(t.x) for t in att]), "imagined": describe([len(t.x) for t in img])},
        },
        "results": results,
        "limitations": nm.LIMITATIONS,
        "interpretation_rule": nm.INTERPRETATION,
        "inputs": {c: {"file": _rel(p), "sha256": _sha256(p), "name": sessions[c].name} for c, p in files.items()},
        "code_version": code_version(),
        "config_file": args.config,
        "config_sha256": config_sha256(cfg),
        "runtime": {"threads": cfg["runtime"].get("threads"), "env": env_set, "nice": os.nice(0),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tensorflow_loaded": sys.modules.get("tensorflow") is not None,
                    "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                    "elapsed_s": round(time.time() - t0, 1)},
    }
    summary = _jsonable(summary)
    out = resolve_path(s["output_dir"])
    _write_csv(out / "per_trial.csv", rows)
    _write_text(out / "summary.json", json.dumps(summary, indent=2))
    _write_text(out / "summary.md", nm.summary_markdown(summary))
    p = results["primary"]["pooled"]
    log(f"PRIMARY (DTW, length-matched): mean rank {p['mean_rank']:.2f} (chance {p['chance_mean_rank']:.2f}), "
        f"top-1 {p['top1_rate']:.3f} (chance {p['top1_chance']:.3f}), p_perm {p['p_perm']:.3g}, "
        f"p_exact {p['p_exact']:.3g}")
    for m in results["measures"]:
        q = results[m]["length_matched"]["pooled"]
        log(f"{nm.MEASURE_LABELS[m]} (length-matched): mean rank {q['mean_rank']:.2f} "
            f"(chance {q['chance_mean_rank']:.2f}), p_perm {q['p_perm']:.3g}")
    if summary["deviations_from_preregistration"]:
        log(f"DEVIATIONS from the pre-registration: {summary['deviations_from_preregistration']}")
    log(f"wrote {_rel(out)}/per_trial.csv, summary.json, summary.md in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
