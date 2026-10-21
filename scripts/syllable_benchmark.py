#!/usr/bin/env python
"""Host, project venv: score the syllable-timing methods on Wairagkar sessions, then apply the choice rule (D-033).

    CUDA_VISIBLE_DEVICES= nice -n 24 .venv/bin/python -I scripts/syllable_benchmark.py evaluate --cell day195_m172
        [--methods current nuclei sylber] [--arms A B] [--no-count] [--no-reference] [--threads N] [--max-trials N]
        [--overwrite]
    .venv/bin/python -I scripts/syllable_benchmark.py summarize

``evaluate`` (one cell = session x decoder) loads the cell's saved decoder output (scripts/syllable_decode.py),
the TTS syllables of the session's cues (scripts/syllable_tts.py) and each method's decoder syllables
(``current`` from the envelope, ``nuclei`` / ``sylber`` from scripts/syllable_segment.py), maps them with each
arm (wvx.targets.align_trials: A = the unchanged D-027 mapping, B = segment-driven, wvx.syllables) and scores
every trial with the functions of scripts/validate_alignment.py against their ``syl_indices`` and ``_syl``
targets. Runs per cell (13):
  - every arm x method, free (the candidates) and count-constrained (secondary: the N strongest decoder
    syllables, N = TTS syllables), TTS side ``forced``;
  - reference: D-027's configuration (arm A, ``current``, envelope-dip TTS syllables), for continuity.
One JSON per run: results/syllable_benchmark/<cell>/<method>__<free|count>__tts-<side>__arm-<A|B>.json, with the
headline numbers, the full validate_alignment-style summaries, the reference placements (chance, their-timing
oracle) and per-trial rows.

``summarize`` collects every configured Wairagkar cell into results/syllable_benchmark/summary.json and
summary.md, and applies the pre-registered rule (``benchmark.choice``, wvx.syllables.choose_method: best
(arm, method) candidate; ties to arm A, then the simpler method). Every cell is reported, whatever the winner.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / "src"))
sys.path.insert(0, str(SCRIPTS))

from wvx import b2v  # noqa: E402  (Keras 2 before TensorFlow)
from wvx.config import resolve_path  # noqa: E402
from wvx.syllable_runtime import rel, runtime_record, sha256, start  # noqa: E402  (no numpy)


def run_name(method: str, count: bool, tts_side: str, arm: str) -> str:
    return f"{method}__{'count' if count else 'free'}__tts-{tts_side}__arm-{arm}"


def result_path(cfg, cell: str, method: str, count: bool, tts_side: str, arm: str) -> Path:
    return Path(cfg["paths"]["syllable_benchmark"]) / cell / f"{run_name(method, count, tts_side, arm)}.json"


def planned_runs(cfg, methods=None, arms=None, count: bool = True, reference: bool = True) -> list:
    """(method, count, tts_side, arm) of every run of a cell: arm x method x {free, count} on the forced TTS side,
    then the D-027 reference."""
    b = cfg["benchmark"]
    runs = [(m, c, "forced", a) for a in (arms or b["arms"]) for m in (methods or b["methods"])
            for c in ((False, True) if count else (False,))]
    if reference:
        ref = b["reference"]
        runs.append((ref["method"], False, ref["tts_side"], ref["arm"]))
    return runs


def headline(rows, acc, cfg, va, infos=None) -> dict:
    """The pre-registered numbers of one run (D-033) from validate_alignment-style accumulators."""
    import numpy as np

    vs = va.variant_summary(rows, acc, "offline", cfg)
    key = f"within_{int(cfg['benchmark']['onset_tolerance_ms'])}ms"
    oracle = [r for r in rows if "oracle_syllables_r" in r]
    return {
        "n_trials": len(rows),
        "onsets_within_tolerance": vs["boundaries_vs_syl_indices"]["mapped_on"].get(key),
        "detected_onsets_within_tolerance": vs["boundaries_vs_syl_indices"]["detected_on"].get(key),
        "onset_tolerance_ms": cfg["benchmark"]["onset_tolerance_ms"],
        "span_onset_error_median_ms": vs["speech_span_vs_their_target"]["span_on"].get("median_abs_ms"),
        "target_r_mean": vs["target_r_mean_over_features"].get("mean"),
        "chance_r_mean": va.summarize(r.get("chance_r") for r in rows).get("mean"),
        "oracle_r_mean": va.summarize(r.get("oracle_syllables_r") for r in rows).get("mean"),
        "oracle_n_trials": len(oracle),
        "target_r_mean_on_oracle_trials": float(np.mean([r["offline"]["r"] for r in oracle])) if oracle else None,
        "decoder_syllables_per_trial_median": float(np.median([r["offline"]["n_detected"] for r in rows])),
        "fraction_decoder_count_equal_theirs": vs["fraction_detected_count_equal"],
        "fraction_tts_count_equal_theirs": float(np.mean([r["n_tts_syllables"] == r["n_their_syllables"] for r in rows])),
        "tts_syllables_without_speech": int(sum(r["n_tts_syllables_without_speech"] for r in rows)),
        "tts_syllables_total": int(sum(r["n_tts_syllables"] for r in rows)),
        "decoder_syllables_total": int(sum(r["offline"]["n_detected"] for r in rows)),
        # arm B only (None for arm A): decoder syllables left silent, TTS syllables squeezed to no frames,
        # trials whose decoder output had no syllable (target all silence), trials paired one to one
        "decoder_syllables_without_tts": _sum_info(infos, "n_decoder_syllables_without_tts"),
        "tts_syllables_without_frames": _sum_info(infos, "n_tts_syllables_without_frames"),
        "trials_without_decoder_syllables": None if not infos else int(sum(i["n_decoder_syllables"] == 0 for i in infos)),
        "trials_paired_one_to_one": None if not infos else int(sum(i["pairing"] == "one_to_one" for i in infos)),
    }


def _sum_info(infos, key):
    return None if not infos else int(sum(i[key] for i in infos))


def score_run(method: str, count: bool, tts_side: str, trials, preds, corpus, cfg, their_raw, syl, words, desc,
              pred_env, tts_forced, records, reorder: bool, arm: str = "A") -> dict:
    """Align every trial with one (method, variant, TTS side, arm) and score it as scripts/validate_alignment.py does.

    ``trials``: objects with ``.neural`` (sets T), ``.text`` and ``.target`` (their decoder target); ``preds``: the
    saved normalized predictions; ``pred_env``: the decoder envelope per trial (dB, neural timeline);
    ``tts_forced``: per trial (K, 2) TTS frames (forced side); ``records``: the segmenter's per-trial records.
    """
    import numpy as np

    import validate_alignment as va
    from wvx import alignment as al, syllables as sy, targets as tg

    w = int(cfg["model"]["window_len"])
    sil20 = tg.silence_frame(corpus, cfg)
    tts_segs = tts_forced if tts_side == "forced" else None
    if count and tts_segs is None:
        raise ValueError("the count-constrained variant is defined for the forced TTS side")
    pred_segs, n_before = [], []
    for i in range(len(trials)):
        segs = sy.decoder_segments(method, pred_env[i], cfg, None if method == "current" else records[i])
        n_before.append(int(len(segs)))
        if count:
            segs = sy.keep_strongest(segs, pred_env[i], len(tts_segs[i]))
        pred_segs.append(segs)
    _, results = tg.align_trials(None, trials, corpus, cfg, reorder=reorder, log=None, preds=preds,
                                 tts_segments=tts_segs, pred_segments=pred_segs, arm=arm)
    rows, acc = [], {"offline": {"errs": {}, "r": []}}
    base_r = {k: [] for k in va.REFERENCES}
    for i, (t, res, pred) in enumerate(zip(trials, results, preds)):
        tts = tg.reduce(corpus.lookup(t.text), cfg)
        tts_speech = al.speech_mask(res.tts_env, cfg)       # the TTS frames speech-only DTW keeps
        row = {"trial": i, "category": desc[i], "n_bins": int(t.neural.shape[0]), "text": t.text,
               "n_their_syllables": int(len(syl[i])), "n_their_words": int(len(words[i])),
               "n_tts_syllables": int(len(res.tts_segments)), "n_decoder_syllables_before_constraint": n_before[i],
               "n_tts_syllables_without_speech": int(sum(not tts_speech[a:b].any() for a, b in res.tts_segments)),
               "dtw_cost": res.dtw_cost, "decoder_r": float(np.nanmean(va.feature_r(pred, t.target[w - 1:]))),
               "mapping": {k: v for k, v in (res.info or {}).items() if k != "assignment"} or None}
        va.add_variant(row, acc, "offline", res, t.target, their_raw[i], syl[i], cfg)
        for k, r in va.reference_r(i, trials, res, tts, their_raw[i], syl[i], sil20, cfg).items():
            base_r[k].append(r)
            row[f"{k}_r"] = float(np.nanmean(r))
        rows.append(row)
    infos = [res.info for res in results] if arm == "B" else None
    return {
        "headline": headline(rows, acc, cfg, va, infos),
        "decoder_vs_their_target_r": va.summarize(r["decoder_r"] for r in rows),
        "decoder_output_sanity": tg.decoder_output_sanity(trials, results, cfg),
        "offline_decoder": va.variant_summary(rows, acc, "offline", cfg),
        "reference_target_r": {k: {"summary": va.summarize(r.get(f"{k}_r") for r in rows),
                                   "per_feature": np.nanmean(np.stack(vv), axis=0).tolist() if vv else None}
                               for k, vv in base_r.items()},
        "per_trial": rows,
    }


def cmd_evaluate(cfg, env_set, args) -> int:
    b2v.use_keras2()
    import numpy as np
    import scipy.io

    import syllable_decode as sd
    import validate_alignment as va
    from wvx import alignment as al, syllables as sy, targets as tg, wairagkar_data
    from wvx.config import deep_merge
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    t_start = time.time()
    spec = sd.cell_spec(cfg, args.cell)
    if "session" not in spec:
        raise SystemExit(f"cell {args.cell} has no Wairagkar session (no ground truth to score against)")
    cell = args.cell + (f"__first{args.max_trials}" if args.max_trials else "")
    v = cfg["validation"]
    cfg = deep_merge(cfg, {"wairagkar_data": {"keys": {"target": v["their_target_key"]}}})
    session = spec["session"]
    voice = cfg["tts"]["voice"]
    runs = planned_runs(cfg, args.methods, args.arms, count=not args.no_count, reference=not args.no_reference)
    existing = [result_path(cfg, cell, *r) for r in runs if result_path(cfg, cell, *r).exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{len(existing)} results exist, e.g. {rel(existing[0])}; pass --overwrite to replace them")

    # their data, as in scripts/validate_alignment.py
    data_dir = resolve_path(v["data_dir"])
    _, trials = sd.load_trials(spec, cfg, with_targets=True)
    their_raw = wairagkar_data.load_raw_targets(data_dir, session, cfg)
    neural_file = wairagkar_data.session_files(data_dir, session, cfg)[0]
    desc = [str(x).strip() for x in np.atleast_1d(scipy.io.loadmat(str(neural_file), squeeze_me=True,
                                                                     variable_names=["trial_description"])["trial_description"])]
    si_path = data_dir / session / v["speech_indices_file"].format(session=session)
    syl = va.load_intervals(si_path, v["syllable_key"], len(trials))
    words = va.load_intervals(si_path, v["word_key"], len(trials))
    n = args.max_trials or len(trials)
    trials, their_raw, desc, syl, words = (x[:n] for x in (trials, their_raw, desc, syl, words))

    # the cell's saved decoder output: same model file, same trials
    model_path = resolve_path(spec["model"])
    preds, dec = sy.load_decoded(sy.decoded_dir(cfg, cell) / "predictions.npz")
    if dec["model"]["sha256"] != sha256(model_path):
        raise SystemExit(f"{cell}: predictions were made with another model file than {rel(model_path)}")
    if dec["n_bins"] != [int(t.neural.shape[0]) for t in trials] or \
            [tg.text_key(x) for x in dec["texts"]] != [tg.text_key(t.text) for t in trials]:
        raise SystemExit(f"{cell}: saved predictions do not match the session's trials")

    corpus = tg.TTSCorpus.load(tg.corpus_path(cfg, voice, session))
    sil20 = tg.silence_frame(corpus, cfg)
    pred_env = [al.envelope_db(tg.prediction_timeline(p, t.neural.shape[0], sil20, cfg), cfg) for p, t in zip(preds, trials)]
    tts_forced = None
    if any(side == "forced" for _, _, side, _ in runs):
        table = sy.load_tts_syllables(cfg, voice, session)
        tts_forced = [table[tg.text_key(t.text)] for t in trials]
    records, seg_headers = {}, {}
    for m in sorted({m for m, _, _, _ in runs if m != "current"}):
        recs, h = sy.segment_records(cfg, cell, m)
        if h["decoded"]["model"]["sha256"] != dec["model"]["sha256"] or len(recs) != n:
            raise SystemExit(f"{cell} {m}: segment file does not belong to these predictions")
        records[m], seg_headers[m] = recs, {k: h.get(k) for k in ("settings", "versions", "checkpoint_load", "code_version",
                                                                   "settings_differing_from_package_defaults", "created")}

    for method, count, side, arm in runs:
        t0 = time.time()
        scored = score_run(method, count, side, trials, preds, corpus, cfg, their_raw, syl, words, desc, pred_env,
                           tts_forced, records.get(method), dec["model"]["reorder_channels"], arm=arm)
        out = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "cell": cell, "cell_spec": spec, "session": session, "method": method, "arm": arm,
            "variant": "count_constrained" if count else "free", "tts_side": side,
            "role": "reference" if side != "forced" else ("secondary" if count else "candidate"),
            "headline": scored["headline"],
            "decoder": dec["model"], "decoded_created": dec.get("created"), "decoded_code_version": dec.get("code_version"),
            "segmenter": seg_headers.get(method), "voice": voice, "tts_meta": corpus.meta,
            "their_target_key": v["their_target_key"],
            **{k: scored[k] for k in ("decoder_vs_their_target_r", "decoder_output_sanity", "offline_decoder",
                                      "reference_target_r")},
            "settings": {"alignment": cfg["alignment"], "syllables": cfg["syllables"]},
            "runtime": runtime_record(cfg, env_set), "code_version": code_version(), "config_sha256": config_sha256(cfg),
            "per_trial": scored["per_trial"],
            "wall_time_s": time.time() - t0,
        }
        p = sy.save_json(result_path(cfg, cell, method, count, side, arm), va._jsonable(out))
        h = out["headline"]
        print(f"[evaluate] {cell} {run_name(method, count, side, arm)}: target r {h['target_r_mean']:.3f}, onsets within "
              f"{h['onset_tolerance_ms']} ms {h['onsets_within_tolerance']:.3f}, span onset error {h['span_onset_error_median_ms']} ms, "
              f"oracle {h['oracle_r_mean']} (n={h['oracle_n_trials']}) -> {rel(p)} ({out['wall_time_s']:.0f} s)", flush=True)
    print(f"[evaluate] {cell}: {len(runs)} runs in {time.time() - t_start:.0f} s")
    return 0


def cmd_summarize(cfg, env_set, args) -> int:
    import numpy as np

    from wvx import syllables as sy
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed
    from wvx.targets import voice_slug

    set_seed(int(cfg["seed"]))
    b = cfg["benchmark"]
    root = Path(cfg["paths"]["syllable_benchmark"])
    cells = {c: s for c, s in b["cells"].items() if "session" in s}
    table, missing = [], []
    for cell, spec in cells.items():
        for method, count, side, arm in planned_runs(cfg):
            p = result_path(cfg, cell, method, count, side, arm)
            if not p.is_file():
                missing.append(rel(p))
                continue
            d = sy.load_json(p)
            table.append({"cell": cell, "role": spec.get("role"), "age_days": spec.get("age_days"), "method": method,
                          "arm": arm, "variant": d["variant"], "tts_side": side, "run_role": d["role"], **d["headline"],
                          "code_version": d["code_version"], "file": rel(p)})
    target_r = {}
    for r in table:
        if r["variant"] == "free" and r["tts_side"] == "forced":
            target_r.setdefault(f"{r['arm']}/{r['method']}", {})[r["cell"]] = r["target_r_mean"]
    try:
        choice = sy.choose_method(target_r, b["choice"])
    except ValueError as e:
        choice = {"error": str(e)}
    voice = voice_slug(cfg["tts"]["voice"])
    corpora = sorted({s["session"] for s in cells.values()} | {cfg["kunz"]["corpus"]})
    tts = {c: {k: sy.load_json(p) for k, p in (("syllables", root / f"tts_syllables_{voice}_{c}.json"),
                                                 ("audio", root / f"tts_audio_{voice}_{c}.json")) if p.is_file()}
           for c in corpora}
    tts_short = {c: {"count_agreement_with_cmudict": t.get("syllables", {}).get("count_agreement_with_cmudict"),
                     "n_count_mismatches": t.get("syllables", {}).get("n_count_mismatches"),
                     "n_aligned": t.get("syllables", {}).get("n_aligned"),
                     "n_utterances": t.get("syllables", {}).get("n_utterances"),
                     "synthesis_delay_median_ms": t.get("audio", {}).get("lag", {}).get("median_ms")} for c, t in tts.items()}
    summary = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "choice": choice,
               "choice_rule": b["choice"], "table": table, "missing_results": missing, "tts": tts_short,
               "code_version": code_version(), "config_sha256": config_sha256(cfg)}
    sy.save_json(root / "summary.json", summary)

    def f(x, nd=3):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    lines = ["# Syllable-timing benchmark (D-033)", "",
             f"Choice rule: highest mean target r over {', '.join(b['choice']['primary_cells'])} (free, forced TTS), "
             f"over (arm, method); ties within {b['choice']['tie_tolerance']} go to the simpler candidate "
             f"({' < '.join(sy.candidates(b['choice']))}).", ""]
    if "winner" in choice:
        lines += [f"**Winner: arm {choice['winner_arm']}, {choice['winner_method']}** (means: " + ", ".join(f"{m} {v:.4f}" for m, v in choice["mean_target_r"].items())
                  + f"; tied within tolerance: {', '.join(choice['tied_within_tolerance'])})", ""]
    else:
        lines += [f"No choice yet: {choice['error']}", ""]
    tol = b["onset_tolerance_ms"]
    lines += [f"| cell | age (d) | arm | method | variant | TTS | onsets <= {tol} ms | span onset err (ms) | target r | chance r | oracle r (n) | r on oracle trials |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['cell']} | {r['age_days']} | {r['arm']} | {r['method']} | {r['variant']} | {r['tts_side']} | "
                     f"{f(r['onsets_within_tolerance'])} | {f(r['span_onset_error_median_ms'], 0)} | {f(r['target_r_mean'])} | "
                     f"{f(r['chance_r_mean'])} | {f(r['oracle_r_mean'])} ({r['oracle_n_trials']}) | {f(r['target_r_mean_on_oracle_trials'])} |")
    lines += ["", "TTS side (forced alignment + maximal onset): syllable count = CMUdict count in "
              + "; ".join(f"{c}: {f(t['count_agreement_with_cmudict'])} ({t['n_aligned']}/{t['n_utterances']} aligned)"
                          for c, t in tts_short.items()) + ".", ""]
    if missing:
        lines += [f"Missing results ({len(missing)}): " + ", ".join(missing), ""]
    (root / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if not missing and "winner" in choice else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["evaluate", "summarize"])
    ap.add_argument("--config", default="configs/experiments/syllable_benchmark.yaml")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--methods", nargs="+", choices=["current", "nuclei", "sylber"], default=None)
    ap.add_argument("--arms", nargs="+", choices=["A", "B"], default=None, help="default: benchmark.arms")
    ap.add_argument("--no-count", action="store_true", help="skip the count-constrained variants")
    ap.add_argument("--no-reference", action="store_true", help="skip the D-027 reference run")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-trials", type=int, default=None, help="score the <cell>__first<N> smoke cell")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.step == "evaluate" and not args.cell:
        ap.error("evaluate needs --cell")
    cfg, env_set = start(args.config, args.threads)
    return {"evaluate": cmd_evaluate, "summarize": cmd_summarize}[args.step](cfg, env_set, args)


if __name__ == "__main__":
    sys.exit(main())
