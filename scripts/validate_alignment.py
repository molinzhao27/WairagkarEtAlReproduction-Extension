#!/usr/bin/env python
"""Host: validate the target alignment on Wairagkar sessions against their syllable timings and targets.

    python -I scripts/validate_alignment.py --inspect --session t15_dayXXX      # file layout only
    python -I scripts/validate_alignment.py --session t15_dayXXX [--voice V] [--max-trials N]
        [--override '{"alignment": {"envelope": {"source": "c0"}}}'] [--tag NAME]

Evaluation sessions of the release hold ``brain2voice_t15_dayPREV.h5``, the closed-loop decoder trained
on all earlier sessions (the decoder the paper's alignment used for that session), its closed-loop
audio for the evaluation trials (``brain2voice_pred_audio_16kHz``), their syllable/word timings
(``<session>_speech_indices.mat``) and their aligned targets (``target_speech_feats_syl``/``_word``).

Two sources of "decoder output" are aligned, each with our TTS of the cue (``wvx.alignment``):
- ``offline_decoder``: their ``.h5`` run offline on every trial (``wvx.inference``, channels reordered
  for the day025-day195 models, as their notebook does);
- ``closed_loop_audio``: their closed-loop audio (evaluation trials only): tests our segmentation and
  alignment independently of our inference.

For each: (i) syllables detected in the decoder output and our mapped TTS syllables vs their
``syl_indices`` (signed error to the nearest of ours, per reference boundary, ms); (ii) our decoder
target vs theirs (both through their transform: Pearson r per feature over the whole trial) and the
speech onset/offset of each (c0 within ``validation.speech_mask_db`` of the trial's loudest frame).

Reference points, same transform and metric: ``span_uniform`` (TTS speech stretched uniformly over the
speech span detected in our decoder output: no syllables, no DTW), ``oracle_span`` (uniform stretch
over THEIR speech span), ``oracle_syllables`` (our TTS syllables placed on THEIR syllables, trials with
equal counts only: what our voice/stretch can reach with their timing) and ``chance`` (our target of
trial i vs their target of trial i + shift). ``voice_check`` compares speech-frame statistics of their
targets with every TTS corpus of this session on the host.

Writes results/alignment_validation/<session>_<voice_slug>[_<tag>].json. CPU only.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx import b2v  # noqa: E402  (Keras 2 before TensorFlow)
from wvx.config import REPO_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import scipy.io  # noqa: E402


def _rel(p) -> str:
    try:
        return str(Path(p).relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return _rel(x)
    if isinstance(x, (np.generic, np.ndarray)):
        return _jsonable(x.tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ---------------------------------------------------------------------------------- inspection


def _describe(v, depth=0):
    if depth > 4:
        return "..."
    if isinstance(v, np.ndarray):
        d = {"type": "ndarray", "dtype": str(v.dtype), "shape": list(v.shape)}
        if v.dtype == object and v.size:
            d["first"] = _describe(v.ravel()[0], depth + 1)
        elif v.dtype.names:
            d["fields"] = {n: _describe(v[n].ravel()[0] if v.size else None, depth + 1) for n in v.dtype.names}
        elif v.size and v.dtype.kind in "iuf":
            d["head"] = v.ravel()[:8].tolist()
        elif v.size and v.dtype.kind in "US":
            d["head"] = [str(s) for s in v.ravel()[:3]]
        return d
    if hasattr(v, "_fieldnames"):
        return {"type": "struct", "fields": {n: _describe(getattr(v, n), depth + 1) for n in v._fieldnames}}
    if isinstance(v, (str, bytes)):
        return {"type": type(v).__name__, "value": str(v)[:80]}
    if isinstance(v, (int, float, np.generic)):
        return {"type": type(v).__name__, "value": float(v)}
    return {"type": type(v).__name__}


def inspect(session_dir: Path) -> dict:
    out = {}
    for f in sorted(session_dir.iterdir()):
        info: dict = {"bytes": f.stat().st_size}
        if f.suffix == ".mat":
            info["variables"] = [{"name": n, "shape": list(s), "class": c} for n, s, c in scipy.io.whosmat(str(f))]
            small = [n for n, s, c in scipy.io.whosmat(str(f)) if "neural" not in f.name or n not in
                     ("spikepow_trials", "threshcross_trials")]
            m = scipy.io.loadmat(str(f), squeeze_me=True, struct_as_record=False, variable_names=small)
            info["content"] = {k: _describe(v) for k, v in m.items() if not k.startswith("__")}
        elif f.suffix in (".txt", ".md"):
            info["text"] = f.read_text(errors="replace")[:4000]
        out[f.name] = info
    return out


# --------------------------------------------------------------------------------- their timings


def load_intervals(path: Path, key: str, n_trials: int) -> list[np.ndarray]:
    """Per trial (K, 2) int64 [onset, offset) bins from ``<session>_speech_indices.mat``."""
    m = scipy.io.loadmat(str(path), squeeze_me=True, variable_names=[key])[key]
    out = [np.asarray(x, dtype=np.int64).reshape(-1, 2) for x in np.atleast_1d(m)]
    if len(out) != n_trials:
        raise ValueError(f"{path.name}: {len(out)} trials in {key}, neural file has {n_trials}")
    return out


# ------------------------------------------------------------------------------------- metrics


def feature_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per column (NaN where a column is constant)."""
    a, b = a - a.mean(0), b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(0) / den, np.nan)


def boundary_stats(errors_frames: list, frame_ms: float, tolerances) -> dict:
    e = np.concatenate([np.asarray(x, float).ravel() for x in errors_frames]) if errors_frames else np.array([])
    e = e[np.isfinite(e)] * frame_ms
    if not e.size:
        return {"n": 0}
    return {
        "n": int(e.size),
        "median_abs_ms": float(np.median(np.abs(e))),
        "mean_abs_ms": float(np.mean(np.abs(e))),
        "mean_signed_ms": float(np.mean(e)),
        **{f"within_{int(t)}ms": float(np.mean(np.abs(e) <= t)) for t in tolerances},
    }


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Shift d (frames) maximizing corr(a[t], b[t + d]): positive d means b lags a."""
    best, best_r = 0, -np.inf
    for d in range(-max_lag, max_lag + 1):
        x, y = (a[: len(a) - d], b[d:]) if d >= 0 else (a[-d:], b[: len(b) + d])
        if len(x) > 2 and x.std() > 0 and y.std() > 0:
            r = np.corrcoef(x, y)[0, 1]
            if r > best_r:
                best, best_r = d, r
    return int(best)


def summarize(values) -> dict:
    x = np.asarray([v for v in values if v is not None], dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std()), "median": float(np.median(x)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75))}


def speech_frames(raw20: np.ndarray, cfg, mask_db: float):
    """Frames of a raw (n, 20) array within mask_db of its loudest frame (c0 power)."""
    from wvx import alignment as al

    db = al.c0_to_db(raw20[:, 0], cfg)
    return raw20[db > db.max() - mask_db]


def voice_check(their_raw: list, corpora: dict, cfg) -> dict:
    """Their targets' speech-frame statistics vs each TTS voice's, on the same sentences."""
    from wvx import targets as tg

    mask_db = float(cfg["validation"]["speech_mask_db"])
    theirs = np.concatenate([speech_frames(r, cfg, mask_db) for r in their_raw])
    m_t = tg.moments(theirs)
    idx = np.asarray(cfg["voice_id"]["distance_features"], int)
    out = {"theirs": {"median_voiced_f0_hz": float(np.median(tg.voiced_f0_hz(theirs, cfg))),
                      "c0_speech_mean": float(m_t.mean[0])}}
    for voice, corpus in corpora.items():
        ours = np.concatenate([speech_frames(tg.reduce(f, cfg), cfg, mask_db) for f in corpus.frames])
        m_o = tg.moments(ours)
        z = (m_o.mean - m_t.mean) / m_t.std
        r = np.log(m_o.std / m_t.std)
        out[voice] = {"distance": float(np.mean(z[idx] ** 2 + r[idx] ** 2)),
                      "median_voiced_f0_hz": float(np.median(tg.voiced_f0_hz(ours, cfg))),
                      "c0_speech_mean": float(m_o.mean[0]),
                      "mean_z": z.tolist()}
    return out


def score_alignment(res, theirs_norm, their_raw, their_syl, cfg) -> tuple[dict, dict]:
    """Per-trial numbers for one alignment: boundary errors (frames) and target agreement."""
    from wvx import alignment as al
    from wvx import targets as tg

    mask_db = float(cfg["validation"]["speech_mask_db"])
    ours = tg.decoder_target(res.target_raw, cfg)
    r = feature_r(ours, theirs_norm)
    row = {"r": float(np.nanmean(r)), "n_detected": int(len(res.pred_segments)),
           "n_mapped": int(len(res.mapped_segments))}
    err = {}
    if len(their_syl):
        err = {
            "detected_on": al.boundary_errors(res.pred_segments[:, 0], their_syl[:, 0]),
            "detected_off": al.boundary_errors(res.pred_segments[:, 1], their_syl[:, 1]),
            "mapped_on": al.boundary_errors(res.mapped_segments[:, 0], their_syl[:, 0]),
            "mapped_off": al.boundary_errors(res.mapped_segments[:, 1], their_syl[:, 1]),
        }
    b_t = al.speech_bounds(their_raw, cfg, mask_db)
    b_o = al.speech_bounds(res.target_raw, cfg, mask_db)
    if b_t and b_o:
        err["span_on"], err["span_off"] = np.array([b_o[0] - b_t[0]]), np.array([b_o[1] - b_t[1]])
    return row, {"r": r, **err}


def add_variant(row: dict, acc: dict, name: str, res, theirs_norm, their_raw, their_syl, cfg) -> None:
    """Score one alignment of a trial into ``row[name]`` and the accumulator ``acc[name]``."""
    sc, er = score_alignment(res, theirs_norm, their_raw, their_syl, cfg)
    row[name] = sc
    acc[name]["r"].append(er.pop("r"))
    for k, e in er.items():
        acc[name]["errs"].setdefault(k, []).append(e)


REFERENCES = ("span_uniform", "oracle_span", "oracle_syllables", "chance")


def reference_r(i: int, trials, res, tts, their_raw_i, syl_i, sil20, cfg) -> dict:
    """Per-feature r of the reference placements of trial ``i`` (module docstring), when defined."""
    from wvx import alignment as al
    from wvx import targets as tg

    v = cfg["validation"]
    T = trials[i].neural.shape[0]
    theirs = trials[i].target

    def place(src_segs, dst_segs):
        return tg.decoder_target(al.stretch_features(tts, src_segs, dst_segs, T, sil20, cfg), cfg)

    out = {}
    tts_span = np.array([[res.tts_segments[0, 0], res.tts_segments[-1, 1]]])
    if len(res.pred_segments):
        dspan = np.array([[res.pred_segments[0, 0], res.pred_segments[-1, 1]]])
        out["span_uniform"] = feature_r(place(tts_span, dspan), theirs)
    b_t = al.speech_bounds(their_raw_i, cfg, float(v["speech_mask_db"]))
    if b_t:
        out["oracle_span"] = feature_r(place(tts_span, np.array([b_t])), theirs)
    if len(syl_i) == len(res.tts_segments):
        out["oracle_syllables"] = feature_r(place(res.tts_segments, syl_i), theirs)
    n = len(trials)
    j = (i + int(v["chance_shift"])) % n
    if j != i:
        L = min(T, trials[j].target.shape[0])
        out["chance"] = feature_r(tg.decoder_target(res.target_raw, cfg)[:L], trials[j].target[:L])
    return out


def variant_summary(rows: list, acc: dict, name: str, cfg) -> dict:
    from wvx import alignment as al

    fms, tol = al.frame_ms(cfg), cfg["validation"]["tolerances_ms"]
    a = acc[name]
    sub = [r for r in rows if name in r]
    return {
        "n_trials": len(sub),
        "target_r_mean_over_features": summarize(r[name]["r"] for r in sub),
        "target_r_per_feature": np.nanmean(np.stack(a["r"]), axis=0).tolist() if a["r"] else None,
        "detected_syllables_minus_theirs": summarize(r[name]["n_detected"] - r["n_their_syllables"] for r in sub),
        "fraction_detected_count_equal": float(np.mean([r[name]["n_detected"] == r["n_their_syllables"] for r in sub])) if sub else None,
        "boundaries_vs_syl_indices": {k: boundary_stats(a["errs"].get(k, []), fms, tol) for k in
                                      ("detected_on", "detected_off", "mapped_on", "mapped_off")},
        "speech_span_vs_their_target": {k: boundary_stats(a["errs"].get(k, []), fms, tol) for k in ("span_on", "span_off")},
    }


# ----------------------------------------------------------------------------------------- run


def run(cfg, session: str, voice: str, args) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from wvx import alignment as al
    from wvx import inference, synthesis, targets as tg, wairagkar_data
    from wvx.config import deep_merge

    t_start = time.time()
    v = cfg["validation"]
    cfg = deep_merge(cfg, {"wairagkar_data": {"keys": {"target": v["their_target_key"]}}})
    w = int(cfg["model"]["window_len"])
    data_dir = Path(REPO_ROOT / v["data_dir"])
    sdir = data_dir / session
    try:
        layer = wairagkar_data.layer_for_session(session, cfg)
    except KeyError:
        layer = -1                                               # informational only for inference
    sess = wairagkar_data.load_session(data_dir, session, layer, cfg)
    their_raw_all = wairagkar_data.load_raw_targets(data_dir, session, cfg)
    neural_file = wairagkar_data.session_files(data_dir, session, cfg)[0]
    extra = scipy.io.loadmat(str(neural_file), squeeze_me=True,
                             variable_names=["trial_description", v["closed_loop_audio_key"]])
    desc_all = [str(x).strip() for x in np.atleast_1d(extra["trial_description"])]
    audio_all = [np.asarray(a, dtype=np.float64).ravel() for a in np.atleast_1d(extra[v["closed_loop_audio_key"]])]
    si_path = sdir / v["speech_indices_file"].format(session=session)
    syl_all = load_intervals(si_path, v["syllable_key"], len(sess.trials))
    word_all = load_intervals(si_path, v["word_key"], len(sess.trials))

    a, b = (int(x) for x in args.trials.split(":")) if args.trials else (0, len(sess.trials))
    if args.max_trials is not None:
        b = min(b, a + args.max_trials)
    sel = slice(a, min(b, len(sess.trials)))
    trials = sess.trials[sel]
    n = len(trials)
    their_raw, desc, audio, syl, words = (x[sel] for x in (their_raw_all, desc_all, audio_all, syl_all, word_all))

    model_path = Path(args.model) if args.model else next(iter(sorted(sdir.glob("brain2voice_*.h5"))))
    day = int(re.search(r"day(\d+)", model_path.name).group(1))
    rflag = str(args.reorder or v["reorder_channels"]).lower()
    reorder = {"true": True, "false": False}.get(rflag, day <= int(v["reorder_until_day"]))
    model = inference.load_model(model_path, cfg)
    synthesis.preflight(cfg)
    corpus = tg.TTSCorpus.load(tg.corpus_path(cfg, voice, args.corpus or session))
    sil20 = tg.silence_frame(corpus, cfg)
    preds, results = tg.align_trials(model, trials, corpus, cfg, reorder=reorder)

    # their closed-loop audio (evaluation trials): its own envelope, segmentation and alignment
    full_scale = float(v["pcm_full_scale"])
    cl_idx = [i for i in range(n) if audio[i].size == trials[i].neural.shape[0] * int(cfg["synthesis"]["samples_per_frame"])]

    synth, encode = tg.lpcnet_callables(cfg)

    def closed_loop(i):
        T = trials[i].neural.shape[0]
        env = al.envelope_from_audio(audio[i] / full_scale, T, cfg)
        tts = tg.reduce(corpus.lookup(trials[i].text), cfg)
        return al.align_trial(np.zeros((T, 20), np.float32), tts, sil20, cfg, synth=synth, encode=encode, pred_env=env)

    with ThreadPoolExecutor(max_workers=int(cfg["alignment"]["n_workers"])) as ex:
        cl_results = dict(zip(cl_idx, ex.map(closed_loop, cl_idx)))

    rows = []
    acc = {k: {"errs": {}, "r": []} for k in ("offline", "closed_loop")}
    base_r = {k: [] for k in REFERENCES}
    for i, (t, res, pred) in enumerate(zip(trials, results, preds)):
        T = t.neural.shape[0]
        theirs = t.target
        tts = tg.reduce(corpus.lookup(t.text), cfg)
        row = {"trial": i, "category": desc[i], "n_bins": T, "text": t.text,
               "n_their_syllables": int(len(syl[i])), "n_their_words": int(len(words[i])),
               "n_tts_syllables": int(len(res.tts_segments)), "dtw_cost": res.dtw_cost,
               "decoder_r": float(np.nanmean(feature_r(pred, theirs[w - 1:])))}
        variants = [("offline", res)] + ([("closed_loop", cl_results[i])] if i in cl_results else [])
        for name, rr in variants:
            add_variant(row, acc, name, rr, theirs, their_raw[i], syl[i], cfg)
        if i in cl_results:
            a_env, b_env = res.pred_env[w - 1:], cl_results[i].pred_env[w - 1:]   # frames with predictions
            row["env_r_offline_vs_closed_loop"] = float(np.corrcoef(a_env, b_env)[0, 1])
            row["env_lag_closed_loop_frames"] = best_lag(a_env, b_env, int(v["max_lag_frames"]))
        for k, r in reference_r(i, trials, res, tts, their_raw[i], syl[i], sil20, cfg).items():
            base_r[k].append(r)
            row[f"{k}_r"] = float(np.nanmean(r))
        rows.append(row)

    corpora = {voice: corpus}
    for p in sorted(Path(cfg["paths"]["tts_features"]).glob(f"*/{session}.npz")):
        c = tg.TTSCorpus.load(p)
        corpora.setdefault(c.voice, c)
    return {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "session": session,
        "n_trials": n,
        "trial_range": [sel.start, sel.stop],
        "categories": {c: desc.count(c) for c in sorted(set(desc))},
        "voice": voice,
        "tag": args.tag,
        "their_target_key": v["their_target_key"],
        "decoder": {"path": _rel(model_path), "sha256": _sha256(model_path), "reorder_channels": reorder},
        "tts_meta": corpus.meta,
        "decoder_vs_their_target_r": summarize(r["decoder_r"] for r in rows),
        "decoder_output_sanity": tg.decoder_output_sanity(trials, results, cfg),
        "their_syllables_per_trial": summarize(r["n_their_syllables"] for r in rows),
        "tts_syllables_per_trial": summarize(r["n_tts_syllables"] for r in rows),
        "fraction_tts_count_equal_theirs": float(np.mean([r["n_tts_syllables"] == r["n_their_syllables"] for r in rows])),
        "offline_decoder": variant_summary(rows, acc, "offline", cfg),
        "closed_loop_audio": variant_summary(rows, acc, "closed_loop", cfg),
        "envelope_r_offline_vs_closed_loop": summarize(r.get("env_r_offline_vs_closed_loop") for r in rows),
        "envelope_lag_closed_loop_minus_offline_frames": summarize(r.get("env_lag_closed_loop_frames") for r in rows),
        "reference_target_r": {k: {"summary": summarize(r.get(f"{k}_r") for r in rows),
                                   "per_feature": np.nanmean(np.stack(vv), axis=0).tolist() if vv else None}
                               for k, vv in base_r.items()},
        "voice_check": voice_check(their_raw, corpora, cfg),
        "alignment_config": cfg["alignment"],
        "per_trial": rows,
        "wall_time_s": time.time() - t_start,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/kunz_targets.yaml")
    ap.add_argument("--session", action="append", required=True)
    ap.add_argument("--voice", default=None, help="default: tts.voice")
    ap.add_argument("--model", default=None, help="default: the session folder's brain2voice_*.h5")
    ap.add_argument("--reorder", default=None, choices=["auto", "true", "false"])
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--trials", default=None, help="A:B, trials A..B-1 only (smoke runs)")
    ap.add_argument("--override", default=None, help="JSON merged into the config (sensitivity checks)")
    ap.add_argument("--tag", default=None, help="suffix of the output file name")
    ap.add_argument("--corpus", default=None, help="TTS corpus name (default: the session name)")
    ap.add_argument("--inspect", action="store_true", help="print the session folder's file layout and exit")
    args = ap.parse_args(argv)

    b2v.use_keras2()
    from wvx import targets as tg
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    from wvx.syllables import load_any_targets_config      # kunz_targets.yaml: same as tg.load_targets_config

    cfg = load_any_targets_config(args.config, overrides=json.loads(args.override) if args.override else None)
    set_seed(int(cfg["seed"]))
    for session in args.session:
        if args.inspect:
            print(json.dumps(_jsonable(inspect(Path(REPO_ROOT / cfg["validation"]["data_dir"]) / session)), indent=1))
            continue
        voice = args.voice or cfg["tts"]["voice"]
        out = run(cfg, session, voice, args)
        out["code_version"], out["config_sha256"] = code_version(), config_sha256(cfg)
        name = f"{session}_{tg.voice_slug(voice)}" + (f"_{args.tag}" if args.tag else "") + ".json"
        path = Path(cfg["paths"]["alignment_validation"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(out), indent=1, allow_nan=False))
        print(f"[validate] {session}: wrote {_rel(path)}")
        for k in ("decoder_vs_their_target_r", "decoder_output_sanity", "their_syllables_per_trial", "tts_syllables_per_trial",
                  "offline_decoder", "closed_loop_audio", "envelope_r_offline_vs_closed_loop"):
            print(k, json.dumps(_jsonable(out[k])))
        print("reference_target_r", json.dumps(_jsonable({k: x["summary"] for k, x in out["reference_target_r"].items()})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
