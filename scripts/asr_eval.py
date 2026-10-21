"""ASR scoring (Whisper via faster-whisper, CPU int8) of the synthesized audio in a results directory.

    # GPU host, inside the container, CPU only:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/asr_eval.py \\
        --results-dir results/reproduce_inference_t15_day195 \\
        --experiment configs/experiments/reproduce_inference.yaml --sources target,live,pred
    # Sensitivity variants (configs/asr.yaml asr.variants; outputs get the suffix _<variant>):
    ... scripts/asr_eval.py --results-dir ... --variant fallback      # library-default temperature fallback
    # Re-derive the closed 50-word vocabulary from the Kunz cue sentences and check configs/asr.yaml:
    .venv/bin/python -I scripts/asr_eval.py --derive-vocabulary

Input: <results-dir>/<asr.eval.per_trial_csv> (scripts/run_inference.py writes per_trial_metrics.csv) with a
trial column, the reference text and one repo-relative wav path column per source (asr.eval.wav_columns:
target = LPCNet resynthesis of the target features, pred = our offline decoder audio). Source "live" is the
audio their decoder produced in closed loop, read from the session's neural .mat (experiment config sections
`closed_loop`, `inference.session`, `paths.wairagkar_neural_data`); it exists for evaluation trials only.
Trial type: the CSV's type column if present, else the .mat trial_description, else "all".

Output (next to the input; small files): asr_per_trial.csv (one row per trial x source, written as it goes)
and asr_summary.json (per source and trial type: n, mean, median, sd of per-trial rates, and pooled rates =
sum of errors / sum of reference units; constrained rates only over trials whose reference is in-vocabulary).
The summary records the effective settings (``effective_settings``: wvx.asr.effective_settings plus sources, type
filter, max_trials and live trim), the code version and the config hash. Existing outputs are replaced only when
their summary records the same effective settings; otherwise the script stops before decoding anything (pass
--overwrite to replace them, or use --variant/--tag for a new file name).
Every setting is in configs/asr.yaml; wvx.asr documents the definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.dont_write_bytecode = True

from wvx import asr  # noqa: E402
from wvx.config import deep_merge, load_config, resolve_path  # noqa: E402
from wvx.provenance import code_version, config_sha256  # noqa: E402
from wvx.seeding import set_seed  # noqa: E402

RATES = ("wer", "cer", "per", "wer_constrained", "cer_constrained", "per_constrained")
POOL = {  # rate -> (errors key, reference-units key)
    "wer": ("word_errors", "ref_words"), "cer": ("char_errors", "ref_chars"), "per": ("phone_errors", "ref_phones"),
    "wer_constrained": ("word_errors_constrained", "ref_words"),
    "cer_constrained": ("char_errors_constrained", "ref_chars"),
    "per_constrained": ("phone_errors_constrained", "ref_phones"),
}
FIELDS = ["trial", "type", "source", "text", "hyp_raw", "hyp_raw_prompted", "ref", "hyp", "hyp_constrained",
          *RATES, "ref_words", "word_errors", "word_errors_constrained", "ref_chars", "char_errors",
          "char_errors_constrained", "ref_phones", "phone_errors", "phone_errors_constrained", "ref_oov_words",
          "hyp_words", "hyp_empty", "n_segments", "avg_logprob", "no_speech_prob", "max_compression_ratio",
          "max_temperature", "skipped_silent", "speech_frames", "asr_seed", "asr_sampled", "asr_sampled_prompted",
          "audio_s", "asr_s", "wav"]
NOT_RECORDED = "<not recorded>"


def _finite(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _stats(rows, key):
    v = np.asarray([r[key] for r in rows if _finite(r[key])], float)
    if not len(v):
        return {"n": 0}
    err, tot = POOL[key]
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "sd": round(float(v.std()), 4),
            "pooled": round(asr.pooled_rate([r[err] for r in rows if _finite(r[key])],
                                            [r[tot] for r in rows if _finite(r[key])]), 4)}


def summarize(rows, cr_threshold):
    out = {}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["source"]][r["type"]].append(r)
        by[r["source"]]["ALL"].append(r)
    for src, types in by.items():
        out[src] = {}
        for t, rs in sorted(types.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
            out[src][t] = {"n_trials": len(rs), **{k: _stats(rs, k) for k in RATES},
                           "empty_hypothesis_fraction": round(float(np.mean([bool(r["hyp_empty"]) for r in rs])), 4),
                           "ref_in_vocabulary_fraction": round(float(np.mean([r["ref_oov_words"] == 0 for r in rs])), 4),
                           "repetition_flag_fraction": round(float(np.mean(
                               [r["max_compression_ratio"] > cr_threshold for r in rs])), 4),
                           "fallback_used_fraction": round(float(np.mean([r["max_temperature"] > 0 for r in rs])), 4),
                           "sampled_fraction": round(float(np.mean([bool(r.get("asr_sampled")) for r in rs])), 4),
                           "skipped_silent_fraction": round(float(np.mean([bool(r.get("skipped_silent"))
                                                                           for r in rs])), 4),
                           "asr_s_mean": round(float(np.mean([r["asr_s"] for r in rs])), 3),
                           "audio_s_mean": round(float(np.mean([r["audio_s"] for r in rs])), 3)}
    return out


def run_settings(acfg: dict, sources, keep_types, max_trials) -> dict:
    """Everything that determines the outputs of one run (JSON-normalized, so it compares equal to a reloaded copy)."""
    s = {"asr": asr.effective_settings(acfg), "sources": list(sources),
         "types_filter": sorted(keep_types) if keep_types else None, "max_trials": max_trials,
         "live_trim_lead_in": acfg["eval"]["live_trim_lead_in"]}
    return json.loads(json.dumps(s))


def legacy_settings(summary: dict) -> dict:
    """The parts of ``run_settings`` that a summary written before ``effective_settings`` existed recorded (for the
    message only: such a summary never counts as matching, since silent-clip skipping and per-clip seeding are new)."""
    a = summary.get("asr") or {}
    m = a.get("model") or {}
    return {"asr": {"engine": a.get("engine"), "model": {k: m.get(k) for k in ("name", "repo_id", "revision", "sha256")},
                    "device": a.get("device"), "compute_type": a.get("compute_type"), "audio": a.get("audio"),
                    "decode": a.get("decode"), "text": a.get("text"), "constrained": summary.get("constrained"),
                    "per": summary.get("per")},
            "sources": summary.get("sources"), "types_filter": summary.get("types_filter"),
            "max_trials": summary.get("max_trials"), "live_trim_lead_in": summary.get("live_trim_lead_in")}


def _flatten(d, prefix=""):
    if isinstance(d, dict) and d:
        out = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def settings_diff(old: dict, new: dict) -> list:
    """'key: old -> new' for every changed setting; keys present on one side only are listed by group at the end."""
    fo, fn = _flatten(old), _flatten(new)
    changed, one_sided = [], set()
    for k in sorted(set(fo) | set(fn)):
        if k not in fo or k not in fn:
            one_sided.add(".".join(k.split(".")[:2]))
        elif fo[k] != fn[k]:
            changed.append(f"{k}: {fo[k]!r} -> {fn[k]!r}")
    if one_sided:
        changed.append(f"{NOT_RECORDED} on one side: " + ", ".join(sorted(one_sided)))
    return changed


def check_existing(out_csv: Path, out_json: Path, settings: dict, overwrite: bool):
    """None if writing is fine; else why not. Fine = no previous outputs, or a previous summary with the same
    effective settings (a faithful re-run), or ``overwrite``."""
    if overwrite or not (out_csv.exists() or out_json.exists()):
        return None
    if not out_json.exists():
        return (f"{out_csv.name} exists without {out_json.name} (interrupted run?), so its settings are unknown. "
                f"Pass --overwrite to replace it, or --tag NAME to write new files.")
    old = json.loads(out_json.read_text())
    recorded = old.get("effective_settings")
    if recorded == settings:
        return None
    if recorded is None:
        why = (f"{out_json.name} predates effective-settings recording (no silent-clip skip, per-process seeding); "
               "differences in what it recorded:")
        diff = settings_diff(legacy_settings(old), settings)
    else:
        why = f"{out_json.name} was produced with different effective settings:"
        diff = settings_diff(recorded, settings)
    return "\n".join([f"refusing to overwrite {out_csv.name} / {out_json.name}. {why}", *(f"  {d}" for d in diff[:30]),
                      *(["  ..."] if len(diff) > 30 else []),
                      "Pass --overwrite to replace them, or --variant/--tag to write new files."])


def derive_vocabulary(acfg: dict) -> int:
    import scipy.io

    src = acfg["constrained"]["vocabulary_source"]
    union = set()
    for name in src["files"]:
        m = scipy.io.loadmat(resolve_path(src["dir"]) / name, variable_names=[src["key"]])
        texts = [str(np.asarray(c).squeeze()) if np.asarray(c).size else "" for c in m[src["key"]].ravel()]
        words = {w for t in texts for w in asr.normalize(t, acfg).split()}
        print(f"{name}: {len(texts)} sentences, {len(words)} words")
        union |= words
    derived, configured = sorted(union), sorted(acfg["constrained"]["vocabulary"])
    print(f"union: {len(derived)} words\n{derived}")
    if derived != configured:
        print("MISMATCH with configs/asr.yaml constrained.vocabulary:",
              "missing", sorted(set(derived) - set(configured)), "extra", sorted(set(configured) - set(derived)))
        return 1
    print("matches configs/asr.yaml constrained.vocabulary")
    return 0


def load_live(exp_cfg: dict):
    import scipy.io

    cl, session = exp_cfg["closed_loop"], exp_cfg["inference"]["session"]
    neural = resolve_path(exp_cfg["paths"]["wairagkar_neural_data"]) / session / \
        exp_cfg["wairagkar_data"]["neural_file"].format(session=session)
    m = scipy.io.loadmat(neural, variable_names=[cl["audio_key"], cl["description_key"]])
    audio = [np.asarray(a).ravel() for a in m[cl["audio_key"]].ravel()]
    desc = [str(np.asarray(d).squeeze()).strip() for d in np.asarray(m[cl["description_key"]]).ravel()]
    lead = (exp_cfg["model"]["window_len"] - 1) * cl["bin_ms"] * cl["sample_rate"] // 1000
    return audio, desc, int(cl["sample_rate"]), int(lead), str(neural)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", help="directory holding the per-trial CSV (repo-relative or absolute)")
    ap.add_argument("--asr-config", default=None, help="default: configs/asr.yaml")
    ap.add_argument("--experiment", default=None, help="experiment YAML with `closed_loop` (needed for source live)")
    ap.add_argument("--sources", default=None, help="comma list (default: every wav column, plus live if --experiment)")
    ap.add_argument("--types", default=None, help="comma list of trial types to keep (default: all)")
    ap.add_argument("--max-trials", type=int, default=None, help="first N kept trials only (smoke runs)")
    ap.add_argument("--variant", default=None, help="name in asr.variants, merged over the config (tags outputs)")
    ap.add_argument("--tag", default=None, help="suffix for the output file names (default: the variant name)")
    ap.add_argument("--derive-vocabulary", action="store_true", help="print the Kunz vocabulary and check the config")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing outputs even if they were produced with different effective settings")
    args = ap.parse_args()

    acfg = asr.load_asr_config(args.asr_config)
    if args.variant:
        acfg = deep_merge(acfg, acfg["variants"][args.variant])
    tag = args.tag if args.tag is not None else (args.variant or "")
    set_seed(load_config()["seed"])
    if args.derive_vocabulary:
        return derive_vocabulary(acfg)
    if not args.results_dir:
        ap.error("--results-dir is required")
    ev = acfg["eval"]
    run_dir = resolve_path(args.results_dir)
    with open(run_dir / ev["per_trial_csv"]) as f:
        trials = list(csv.DictReader(f))

    exp_cfg = load_config(args.experiment) if args.experiment else None
    live = load_live(exp_cfg) if exp_cfg is not None and "closed_loop" in exp_cfg else None
    sources = args.sources.split(",") if args.sources else \
        list(ev["wav_columns"]) + ([ev["live_source"]] if live else [])
    for s in sources:
        if s != ev["live_source"] and s not in ev["wav_columns"]:
            ap.error(f"unknown source {s!r}")
        if s == ev["live_source"] and live is None:
            ap.error("source live needs --experiment with a closed_loop section")
    keep_types = set(args.types.split(",")) if args.types else None

    def trial_type(r):
        if ev["type_column"] in r and r[ev["type_column"]]:
            return r[ev["type_column"]]
        return live[1][int(r[ev["trial_column"]])] if live else "all"

    todo = [r for r in trials if keep_types is None or trial_type(r) in keep_types]
    if args.max_trials is not None:
        todo = todo[: args.max_trials]

    sfx = f"_{tag}" if tag else ""
    out_csv = run_dir / ev["out_per_trial"].replace(".csv", f"{sfx}.csv")
    out_json = run_dir / ev["out_summary"].replace(".json", f"{sfx}.json")
    settings = run_settings(acfg, sources, keep_types, args.max_trials)
    refusal = check_existing(out_csv, out_json, settings, args.overwrite)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    model_path = asr.ensure_model(acfg)  # download (once) + sha256 check before the clock starts
    print(f"{len(todo)} trials x {sources}; model {model_path.name}, {acfg['compute_type']}, "
          f"{acfg['cpu_threads']} threads, constrained={acfg['constrained']['method']}", flush=True)

    rows, t_start, n_done = [], time.perf_counter(), 0
    n_total = sum(1 for r in todo for s in sources
                  if s != ev["live_source"] or live[0][int(r[ev["trial_column"]])].size)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in todo:
            i, text, ttype = int(r[ev["trial_column"]]), r[ev["text_column"]], trial_type(r)
            for s in sources:
                if s == ev["live_source"]:
                    audio, _, sr, lead, src_path = live
                    x = audio[i]
                    if not x.size:
                        continue  # no closed-loop audio for this trial
                    # float32 holding int16-scaled samples (notes/DATA_FORMATS.md) -> full scale 1.0
                    x = x.astype(np.float64) / acfg["audio"]["int_full_scale"]
                    if ev["live_trim_lead_in"]:
                        x = x[lead:]
                    wav = f"{Path(src_path).name}:{exp_cfg['closed_loop']['audio_key']}[{i}]"
                else:
                    wav = r[ev["wav_columns"][s]]
                    x, sr = sf.read(resolve_path(wav), dtype="float64")
                res = asr.evaluate_wav(x, sr, text, acfg)
                row = {"trial": i, "type": ttype, "source": s, "text": text, "wav": wav, **res}
                rows.append(row)
                w.writerow(row)
                f.flush()
                n_done += 1
                if n_done % ev["progress_every"] == 0 or n_done == n_total:
                    el = time.perf_counter() - t_start
                    print(f"{n_done}/{n_total} clips, {el:.0f} s elapsed, {el / n_done:.2f} s/clip, "
                          f"eta {el / n_done * (n_total - n_done):.0f} s", flush=True)

    total_s = time.perf_counter() - t_start
    summary = {
        "results_dir": str(Path(args.results_dir)), "variant": args.variant,
        "effective_settings": settings,
        "code_version": code_version(), "config_sha256": config_sha256(acfg),
        "sources": sources, "types_filter": sorted(keep_types) if keep_types else None,
        "max_trials": args.max_trials,
        "asr": {k: acfg[k] for k in ("engine", "model", "seed", "device", "compute_type", "cpu_threads", "audio",
                                     "decode", "silence", "text")},
        "constrained": {k: v for k, v in acfg["constrained"].items() if k != "vocabulary_source"},
        "per": {k: acfg["per"][k] for k in ("enabled", "engine", "strip_stress")},
        "live_trim_lead_in": ev["live_trim_lead_in"],
        "versions": {**asr.versions(), "python": platform.python_version()},
        "runtime": {"clips": len(rows), "total_s": round(total_s, 1),
                    "asr_s_per_clip_mean": round(float(np.mean([r["asr_s"] for r in rows])), 3) if rows else None,
                    "audio_s_total": round(float(sum(r["audio_s"] for r in rows)), 1),
                    "real_time_factor": round(float(sum(r["asr_s"] for r in rows) / sum(r["audio_s"] for r in rows)),
                                              4) if rows else None},
        "results": summarize(rows, acfg["decode"]["compression_ratio_threshold"]),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_csv.name} ({len(rows)} rows) and {out_json.name} in {total_s:.0f} s", flush=True)
    for src, types in summary["results"].items():
        for t, d in types.items():
            print(f"{src:7s} {t:32s} n={d['n_trials']:3d}  " + "  ".join(
                f"{k}={d[k].get('median', float('nan')):.3f}/{d[k].get('pooled', float('nan')):.3f}"
                for k in ("wer", "wer_constrained", "per")) + "   (median/pooled)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
