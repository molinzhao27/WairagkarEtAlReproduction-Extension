#!/usr/bin/env python
"""Word-splice targets (D-052) for the dev test: t15_day195 training trials, from their word recordings of other days.

    host: CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/word_splice.py --config configs/experiments/word_splice.yaml

``word_splice`` section: ``inventory_sessions`` (their 50-word trials, ``inventory_category``), ``eval_session``,
``train_trials_from`` (the dev run's data.json: the exact training trials), ``data_config`` (their targets as the dev
training loads them), ``placement_config`` (TTS + syllables + arm B, as tts_oracle/tts_auto), ``auto_cell`` (the
saved decode of the dev decoder on the eval session; scripts/syllable_decode.py).

Variants (``wvx.word_splice``; ``variants``, default the first two): ``splice_oracle`` on their true
``word_indices``; ``splice_auto`` on word spans from the arm-B mapped TTS syllables on the c0 dips of ``auto_cell``'s
output; ``splice_manual`` (D-061) on word spans from aggregated manual syllable answers (``manual``: the package
``name``, its ``config`` and optional ``label``; ``cell``), syllable -> word membership from the placement voice's
forced alignment; every training trial needs an answer. A trial with a word missing from the inventory, or whose
aligner words do not match its cue, keeps the TTS target of the same timing (counted); for ``splice_manual`` that is
the TTS placed one to one on the manual syllables.

Writes ``paths.word_splice``/<name>_<variant>.npz (keys trNNN, header info) and results/word_splice/<name>.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[word_splice {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def their_session(session: str, cfg_d, cfg_p):
    """(raw targets, word spans, syllable spans, cues, descriptions) of one Wairagkar session."""
    import numpy as np
    import scipy.io
    import validate_alignment as va
    from wvx import wairagkar_data

    data_dir = resolve_path(cfg_d["sanity"]["data_dir"])
    raw = wairagkar_data.load_raw_targets(data_dir, session, cfg_d)
    neural_file = wairagkar_data.session_files(data_dir, session, cfg_d)[0]
    keys = cfg_d["wairagkar_data"]["keys"]
    m = scipy.io.loadmat(str(neural_file), squeeze_me=True,
                         variable_names=[keys["cues"], keys["cues_fallback"], "trial_description"])
    cues = [str(x).strip() for x in np.atleast_1d(m[keys["cues"]] if keys["cues"] in m else m[keys["cues_fallback"]])]
    desc = [str(x).strip() for x in np.atleast_1d(m["trial_description"])]
    v = cfg_p["validation"]
    si = data_dir / session / v["speech_indices_file"].format(session=session)
    words = va.load_intervals(si, v["word_key"], len(raw))
    syl = va.load_intervals(si, v["syllable_key"], len(raw))
    return raw, words, syl, cues, desc


def manual_alignment_results(m, train, trials, corpus, preds, tts_segs, cfg_d, cfg_p):
    """Per training trial, the placement voice's TTS stretched one to one onto the aggregated manual syllable answers
    of package ``m["name"]`` (scripts/manual_alignment.py ingest; arm B, as its ``targets`` step), or None when the
    trial has no answer. Also checks that the key and answers belong to the same package version."""
    import numpy as np

    from wvx import syllables as sy, targets as tg
    from wvx.syllables import load_json

    cfg_m = load_experiment_config(m["config"])
    root = resolve_path(cfg_m["paths"]["manual_alignment"])
    sfx = f"_{m['label']}" if m.get("label") else ""
    key = load_json(root / f"{m['name']}_key.json")
    agg = load_json(root / f"{m['name']}_work{sfx}" / "aggregated.json")
    if agg["files_sha256"] != key["files_sha256"]:
        raise SystemExit("aggregated answers belong to another version of the package")
    if cfg_m["tts"]["voice"] != cfg_p["tts"]["voice"]:
        raise SystemExit(f"package voice {cfg_m['tts']['voice']!r} != placement voice {cfg_p['tts']['voice']!r}")
    t2id = {it["trial"]: iid for iid, it in key["items"].items() if it["cell"] == m["cell"]}
    have = [i for i in train if i in t2id and t2id[i] in agg["items"]]
    if len(have) < len(train):
        raise SystemExit(f"{len(train) - len(have)} of {len(train)} training trials have no aggregated answer in "
                         f"{m['name']}{sfx} (every trial needs one, D-061), e.g. {[i for i in train if i not in have][:5]}")
    res = [None] * len(train)
    if have:
        jj = [train.index(i) for i in have]
        segs = [sy.manual_segments(agg["items"][t2id[i]]["spans"], cfg_p, trials[j].neural.shape[0])
                for i, j in zip(have, jj)]
        _, r = tg.align_trials(None, [trials[j] for j in jj], corpus, cfg_p,
                               reorder=bool(cfg_d["experiment"]["reorder_channels"]), preds=[preds[i] for i in have],
                               tts_segments=[tts_segs[j] for j in jj], pred_segments=segs, arm="B", log=None)
        for j, rr, sg in zip(jj, r, segs):
            if (rr.info or {}).get("pairing") != "one_to_one" or not np.array_equal(rr.mapped_segments, sg):
                raise SystemExit(f"trial {train[j]}: manual spans were not used one to one")
            res[j] = rr
    n_ans = [agg["items"][t2id[i]]["n_answers"] for i in have]
    info = {"package": m["name"], "label": m.get("label"), "cell": m["cell"], "files_sha256": key["files_sha256"],
            "n_trials_with_answers": len(have), "n_trials": len(train),
            "answers_per_trial": {"min": min(n_ans), "median": float(np.median(n_ans)), "max": max(n_ans)} if n_ans else None}
    log(f"manual timing: {len(have)}/{len(train)} training trials answered ({m['name']}{sfx})")
    return res, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/word_splice.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np

    from wvx import alignment as al, kunz_experiment as kx, syllables as sy, targets as tg, voice_conversion as vc
    from wvx import wairagkar_data, word_splice as ws
    from wvx.provenance import code_version

    c = cfg["word_splice"]
    cfg_d, cfg_p = load_experiment_config(c["data_config"]), load_experiment_config(c["placement_config"])
    t0 = time.time()

    # inventory: their 50-word trials of the other sessions
    raws, spans, texts, srcs = [], [], [], []
    for s in c["inventory_sessions"]:
        raw, words, _, cues, desc = their_session(s, cfg_d, cfg_p)
        sel = [i for i, d in enumerate(desc) if d == c["inventory_category"]]
        raws += [raw[i] for i in sel]
        spans += [words[i] for i in sel]
        texts += [cues[i] for i in sel]
        srcs += [f"{s}:tr{i}" for i in sel]
        log(f"inventory: {s}: {len(sel)} trials")
    inv = ws.build_inventory(raws, spans, texts, srcs)
    log(f"inventory: {inv['n_trials']} trials used, {inv['n_skipped']} skipped, {len(inv['words'])} words, "
        f"{sum(len(v) for v in inv['words'].values())} exemplars")

    # the eval session's training trials
    ev = c["eval_session"]
    train = json.loads(resolve_path(c["train_trials_from"]).read_text())["train_trials"]
    data_dir = resolve_path(cfg_d["sanity"]["data_dir"])
    sess = wairagkar_data.load_session(data_dir, ev, 0, cfg_d)
    _, words_e, syl_e, cues_e, _ = their_session(ev, cfg_d, cfg_p)
    trials = [sess.trials[i] for i in train]

    # TTS timing variants (fallbacks, and the auto word spans)
    voice = cfg_p["tts"]["voice"]
    corpus = tg.TTSCorpus.load(tg.corpus_path(cfg_p, voice, ev))
    sil20 = tg.silence_frame(corpus, cfg_p)
    tdir = sy.tts_dir(cfg_p, voice, ev)
    js = sy.load_json(tdir / cfg_p["syllables"]["forced_alignment"]["syllables_file"])
    onsets = frozenset(tuple(o.split()) for o in js["legal_onsets"])
    tts = {}
    for u in js["utterances"]:
        w_, ph_ = sy.read_aligner_json(tdir / f"mfa_out_{ev}" / f"{u['name']}.json", cfg_p)
        syl_info = sy.tts_syllables(w_, ph_, onsets, cfg_p)
        tts[tg.text_key(u["text"])] = {"frames": np.asarray(u["frames"], np.int64).reshape(-1, 2),
                                      "words": syl_info["words"], "syllable_words": syl_info["syllable_words"]}
    variants = list(c.get("variants") or ["splice_oracle", "splice_auto"])
    preds, header = sy.load_decoded(sy.decoded_dir(cfg_p, c["auto_cell"]) / "predictions.npz")
    if [int(header["n_bins"][i]) for i in train] != [int(t.neural.shape[0]) for t in trials]:
        raise SystemExit(f"{c['auto_cell']}: saved predictions do not match {ev}'s trials")
    tts_segs = [tts[tg.text_key(t.text)]["frames"] for t in trials]
    auto_res = oracle_tts = manual_res = None
    if "splice_auto" in variants:
        auto_segs = [sy.decoder_segments("current", al.envelope_db(tg.prediction_timeline(preds[i], t.neural.shape[0],
                                                                                           sil20, cfg_p), cfg_p), cfg_p, None)
                     for i, t in zip(train, trials)]
        _, auto_res = tg.align_trials(None, trials, corpus, cfg_p, reorder=bool(cfg_d["experiment"]["reorder_channels"]),
                                      preds=[preds[i] for i in train], tts_segments=tts_segs, pred_segments=auto_segs,
                                      arm="B", log=None)
    if "splice_oracle" in variants:
        ok_syl = [bool(len(syl_e[i])) for i in train]
        oracle_tts = vc.oracle_targets(trials, [syl_e[i] for i in train], corpus, {tg.text_key(t.text): s for t, s in
                                                                                    zip(trials, tts_segs)}, cfg_p, valid=ok_syl)
    if "splice_manual" in variants:
        manual_res, manual_info = manual_alignment_results(c["manual"], train, trials, corpus, preds, tts_segs, cfg_d,
                                                          cfg_p)
        summary_manual = manual_info

    out, summary = {}, {"variants": {}}
    if manual_res is not None:
        summary["manual"] = summary_manual
    for variant in variants:
        tgt, info = {}, {"n_trials": len(train), "fallback_missing_word": 0, "fallback_word_mismatch": 0,
                         "fallback_no_timing": 0}
        for j, (i, t) in enumerate(zip(train, trials)):
            words = ws.words_of(t.text)
            T = t.neural.shape[0]
            if variant == "splice_oracle":
                spans_i = words_e[i] if len(words_e[i]) == len(words) else None
                fb = oracle_tts[j][0] if oracle_tts[j] is not None else None
            else:
                res_j = (auto_res if variant == "splice_auto" else manual_res)[j]
                info_t = tts[tg.text_key(t.text)]
                if res_j is None:
                    spans_i = None
                elif ([ws.words_of(w)[0] if ws.words_of(w) else w for w in info_t["words"]] != words
                        or len(info_t["syllable_words"]) != len(res_j.mapped_segments)):
                    spans_i = None
                else:
                    spans_i = ws.word_spans_from_syllables(res_j.mapped_segments, info_t["syllable_words"],
                                                           len(words))
                fb = tg.decoder_target(res_j.target_raw, cfg_d) if res_j is not None else None
            raw = None
            if spans_i is None:
                info["fallback_word_mismatch"] += 1
            else:
                raw, _ = ws.splice(words, spans_i, T, inv)
                if raw is None:
                    info["fallback_missing_word"] += 1
            if raw is not None:
                tgt[i] = tg.decoder_target(raw, cfg_d)
            elif fb is not None:
                tgt[i] = fb
            elif variant == "splice_manual":
                raise SystemExit(f"trial {i}: no manual answer and no spliced target")
            else:
                info["fallback_no_timing"] += 1
                tgt[i] = tg.decoder_target(auto_res[j].target_raw, cfg_d)
            if tgt[i].shape != t.target.shape:
                raise ValueError(f"trial {i}: {variant} target {tgt[i].shape} vs theirs {t.target.shape}")
        r = [vc.trial_r(tgt[i], sess.trials[i].target) for i in train]
        info["vs_theirs_word_target_r"] = kx.describe(r)
        spliced = [i for i in train]  # all trials, incl. fallbacks
        info["fraction_outside_unit_range"] = float(np.mean([float((abs(tgt[i]) > 1).mean()) for i in spliced]))
        log(f"{variant}: r vs their targets {info['vs_theirs_word_target_r']['mean']:.3f}; fallbacks "
            f"{ {k: v for k, v in info.items() if k.startswith('fallback')} }")
        path = resolve_path(cfg["paths"]["word_splice"]) / f"{c['name']}_{variant}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"variant": variant, "eval_session": ev, "inventory_sessions": c["inventory_sessions"], **info,
                "code_version": code_version()}
        tmp = path.with_name(path.stem + ".partial.npz")
        np.savez_compressed(tmp, header=np.array(json.dumps(meta, default=float)),
                            **{f"tr{i:03d}": tgt[i] for i in train})
        tmp.replace(path)
        summary["variants"][variant] = {**info, "file": str(path.relative_to(resolve_path(".")))}
        out[variant] = tgt
    summary.update({"name": c["name"], "config": args.config, "inventory": {
        "sessions": c["inventory_sessions"], "n_trials": inv["n_trials"], "n_skipped": inv["n_skipped"],
        "n_words": len(inv["words"]), "exemplars_per_word_min": min(len(v) for v in inv["words"].values())},
        "auto_cell": c["auto_cell"], "code_version": code_version(), "wall_time_s": time.time() - t0})
    res = resolve_path("results/word_splice")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{c['name']}.json").write_text(json.dumps(kx.portable(summary), indent=1, default=float) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/word_splice/{c['name']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
