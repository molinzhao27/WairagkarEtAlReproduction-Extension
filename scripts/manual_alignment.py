#!/usr/bin/env python
"""Manual syllable alignment by annotator agents: package, deliver, collect, ingest, score, targets (wvx.manual_alignment).

Host, project venv (CPU only, <= 4 threads):
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/manual_alignment.py package --name NAME [--salt HEX] [--overwrite]
    .venv/bin/python -I scripts/manual_alignment.py scan    --name NAME      # blindness scan of an existing package
    .venv/bin/python -I scripts/manual_alignment.py ingest  --name NAME [--snapshot DIR]
    .venv/bin/python -I scripts/manual_alignment.py score   --name NAME
    .venv/bin/python -I scripts/manual_alignment.py targets --name NAME [--overwrite]
Mac (the repo's environment; ssh + docker exec, as scripts/remote/sync.sh):
    python scripts/manual_alignment.py deliver --name NAME [--dest DIR]   # host package -> ~/manual_alignment/NAME/
    python scripts/manual_alignment.py collect --name NAME [--src DIR]    # ~/manual_alignment/NAME/answers/ -> host

Host layout under paths.manual_alignment (data/processed/manual_alignment/):
    NAME/                 the package (the only thing that reaches the Mac): items/, practice/, examples/, tools/, ...
    NAME_key.json         item id -> source (set, cell, session, trial), the salt; never leaves the host
    NAME_answers/collect_<UTC>/answers/   snapshots of the returned answers (collect)
    NAME_work/            ingest + score details with item ids (host only)
Small summaries WITHOUT item ids go to results/manual_alignment/NAME/ (results/ reaches the Mac and GitHub).

package   selects the blind calibration trials (seeded), the production trials, worked examples and practice items
          (disjoint, no shared cue), writes per item a PNG (the package's own tools/inspect_item.py render), the arrays
          (.npz) and the item JSON, the practice reference spans, settings, manifest (workers, batches with redundancy
          k, practice rounds), README/instructions/ORCHESTRATOR_PROMPT from configs/manual_alignment/, the tools, and
          files.sha256; then runs the blindness scan (with every calibration truth) and only then moves the package
          into place. Writes the key last.
deliver   runs the host scan, streams ONLY the package folder (tar through docker exec), refuses any member outside it
          or any key/answers/work file, verifies files.sha256 and re-scans locally before moving it into place.
collect   sends answers/ (JSON and Markdown only, no links) to a new NAME_answers/collect_<UTC>/ on the host.
ingest    validates every answers/<worker>/*.json of the latest snapshot (same rules as tools/inspect_item.py check),
          rejects with reasons (never fixes), aggregates (median per syllable start and end) and reports agreement.
score     calibration cells: manual (aggregated, and per worker) vs B/current on the SAME trials, with
          scripts/syllable_benchmark.score_run (identical code: onsets within 100 ms, span onset error, target r,
          chance, oracle), and the pre-registered adoption rule.
targets   Kunz targets from the aggregated production answers into paths.manual_targets (never v1/v2), one to one.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / "src"))
sys.path.insert(0, str(SCRIPTS))

from wvx.config import REPO_ROOT, resolve_path  # noqa: E402
from wvx.syllable_runtime import rel, runtime_record, sha256, start  # noqa: E402  (no numpy)

NAME_RE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"


# ------------------------------------------------------------------------------------------- helpers


def _paths(cfg, name: str, label: Optional[str] = None) -> dict:
    """``label`` (ingest/score/targets --label) keeps a second answer set of the same package apart: its own work
    folder, results subfolder and targets folder. No label = the first answer set, as before."""
    root = Path(cfg["paths"]["manual_alignment"])
    sfx = f"_{label}" if label else ""
    res = Path(cfg["paths"]["manual_alignment_results"]) / name
    return {"root": root, "pkg": root / name, "key": root / f"{name}_key.json", "answers": root / f"{name}_answers",
            "work": root / f"{name}_work{sfx}", "staging": root / f".{name}.partial",
            "results": res / label if label else res, "targets": Path(f"{cfg['paths']['manual_targets']}{sfx}")}


def _save(path: Path, obj) -> Path:
    import validate_alignment as va
    from wvx import syllables as sy

    return sy.save_json(path, va._jsonable(obj))


def _tts_info(cfg, corpus: str) -> dict:
    """{cue key: {"frames": (K, 2), "labels": [...]}} from the forced TTS syllables (scripts/syllable_tts.py), with the
    word and phones of every syllable re-derived from the aligner output and checked against the stored syllables."""
    import numpy as np

    from wvx import manual_alignment as ma, syllables as sy, targets as tg

    voice = cfg["tts"]["voice"]
    d = sy.tts_dir(cfg, voice, corpus)
    js = sy.load_json(d / cfg["syllables"]["forced_alignment"]["syllables_file"])
    onsets = frozenset(tuple(o.split()) for o in js["legal_onsets"])
    out = {}
    for u in js["utterances"]:
        words, phones = sy.read_aligner_json(d / f"mfa_out_{corpus}" / f"{u['name']}.json", cfg)
        syl = sy.tts_syllables(words, phones, onsets, cfg)
        if syl["words"] != u["words"] or not np.allclose(syl["syllables_s"], u["syllables_s"]):
            raise SystemExit(f"{corpus} {u['name']}: aligner output no longer matches syllables.json")
        labels = ma.syllable_labels(syl["words"], syl["syllable_words"], syl["syllable_phones"])
        frames = np.asarray(u["frames"], dtype=np.int64).reshape(-1, 2)
        if len(labels) != len(frames):
            raise SystemExit(f"{corpus} {u['name']}: {len(labels)} labels for {len(frames)} syllables")
        out[tg.text_key(u["text"])] = {"frames": frames, "labels": labels}
    return out


class Cells:
    """Per cell: decoded predictions (checked against the model and the trials), neural features, cues, wavs, their
    syllables (Wairagkar sessions), TTS syllables + labels. Sessions are loaded once."""

    def __init__(self, cfg, with_targets: bool = False):
        self.cfg, self.with_targets = cfg, with_targets
        self._sessions, self._tts, self._cells = {}, {}, {}

    def session(self, spec):
        import syllable_decode as sd

        key = spec.get("session") or spec["kunz_condition"]
        if key not in self._sessions:
            self._sessions[key] = sd.load_trials(spec, self.cfg, with_targets=self.with_targets)[1]
        return self._sessions[key]

    def tts(self, corpus):
        if corpus not in self._tts:
            self._tts[corpus] = _tts_info(self.cfg, corpus)
        return self._tts[corpus]

    def __getitem__(self, cell: str) -> dict:
        if cell in self._cells:
            return self._cells[cell]
        import numpy as np

        import syllable_decode as sd
        import validate_alignment as va
        from wvx import syllables as sy, targets as tg

        cfg = self.cfg
        spec = sd.cell_spec(cfg, cell)
        trials = self.session(spec)
        preds, h = sy.load_decoded(sy.decoded_dir(cfg, cell) / "predictions.npz")
        model = resolve_path(spec["model"])
        if h["model"]["sha256"] != sha256(model):
            raise SystemExit(f"{cell}: predictions were made with another model file than {rel(model)}")
        if h["n_bins"] != [int(t.neural.shape[0]) for t in trials] or \
                [tg.text_key(x) for x in h["texts"]] != [tg.text_key(t.text) for t in trials]:
            raise SystemExit(f"{cell}: saved predictions do not match the trials")
        corpus = sd.corpus_name(spec, cfg)
        tts = self.tts(corpus)
        missing = sorted({t.text for t in trials if tg.text_key(t.text) not in tts})
        if missing:
            raise SystemExit(f"{cell}: {len(missing)} cues without forced TTS syllables, e.g. {missing[:2]}")
        syl = None
        if "session" in spec:
            v = cfg["validation"]
            s = spec["session"]
            syl = va.load_intervals(resolve_path(v["data_dir"]) / s / v["speech_indices_file"].format(session=s),
                                    v["syllable_key"], len(trials))
        self._cells[cell] = {
            "spec": spec, "trials": trials, "preds": preds, "header": h, "corpus": corpus, "tts": tts, "syl": syl,
            "session": spec.get("session") or h["session"], "wav_dir": sy.decoded_dir(cfg, cell) / h["wav"]["dir"],
            "silence": np.asarray(h["silence_frame"], dtype=np.float32), "reorder": bool(h["model"]["reorder_channels"]),
        }
        return self._cells[cell]


# ------------------------------------------------------------------------------------------- package


def _auto_spans(cfg, cells: Cells, it: dict) -> list:
    """B/current spans of one item in seconds (only for display.show_auto_boundaries; default off)."""
    from wvx import alignment as al, manual_alignment as ma, syllables as sy, targets as tg

    c = cells[it["cell"]]
    t = c["trials"][it["trial"]]
    if "corpus_obj" not in c:
        c["corpus_obj"] = tg.TTSCorpus.load(tg.corpus_path(cfg, cfg["tts"]["voice"], c["corpus"]))
    pred_raw = tg.prediction_timeline(c["preds"][it["trial"]], t.neural.shape[0], c["silence"], cfg)
    env = al.envelope_db(pred_raw, cfg)
    res = sy.align_trial_segment_driven(pred_raw, tg.reduce(c["corpus_obj"].lookup(t.text), cfg), c["silence"], cfg,
                                        tts_segments=c["tts"][tg.text_key(t.text)]["frames"],
                                        pred_segments=sy.decoder_segments("current", env, cfg), pred_env=env)
    return ma.spans_seconds(res.mapped_segments, cfg)


def _cell_info(cfg, cells: Cells, cell: str) -> dict:
    """What wvx.manual_alignment.choose_items needs to know about a cell's trials."""
    from wvx import manual_alignment as ma, targets as tg

    c = cells[cell]
    tr = c["trials"]
    keys = [tg.text_key(t.text) for t in tr]
    return {"session": c["session"], "n": len(tr), "text": keys,
            "wellformed": [c["syl"] is not None and ma.wellformed(c["syl"][i], t.neural.shape[0]) for i, t in enumerate(tr)],
            "n_truth": [None if c["syl"] is None else len(c["syl"][i]) for i in range(len(tr))],
            "n_tts": [len(c["tts"][k]["frames"]) for k in keys],
            "duration_s": [t.neural.shape[0] * ma.frame_s(cfg) for t in tr]}


def _source(cfg, cells: Cells, show_auto: bool):
    """item -> the inputs of wvx.manual_alignment.write_item (truth only for practice items and examples)."""
    import soundfile as sf

    from wvx import syllables as sy, targets as tg

    def source(it):
        c = cells[it["cell"]]
        t = c["trials"][it["trial"]]
        pred_raw = tg.prediction_timeline(c["preds"][it["trial"]], t.neural.shape[0], c["silence"], cfg)
        wav, sr = sf.read(str(c["wav_dir"] / sy.trial_wav_name(it["trial"])), dtype="float64")
        if sr != int(cfg["synthesis"]["sample_rate"]):
            raise SystemExit(f"{it['cell']} trial {it['trial']}: audio at {sr} Hz")
        out = {"wav": wav, "pred_raw": pred_raw, "neural": t.neural, "text": t.text,
               "labels": c["tts"][tg.text_key(t.text)]["labels"],
               "auto": _auto_spans(cfg, cells, it) if show_auto and it["set"] in ("calibration", "production") else None}
        if it["set"] in ("practice", "example"):
            out["truth"] = c["syl"][it["trial"]]
        return out

    return source


def cmd_package(cfg, env_set, args) -> int:
    from wvx import b2v

    b2v.use_keras2()
    import secrets

    import numpy as np

    from wvx import manual_alignment as ma
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    t0 = time.time()
    s = ma.settings(cfg)
    P = _paths(cfg, args.name)
    if P["answers"].exists():
        raise SystemExit(f"{rel(P['answers'])} exists: answers belong to the current ids; make a package with a new name")
    if (P["pkg"].exists() or P["key"].exists()) and not args.overwrite:
        raise SystemExit(f"{rel(P['pkg'])} or its key exists; pass --overwrite to replace both")
    salt = args.salt or secrets.token_hex(int(s["ids"]["salt_bytes"]))
    if not re.fullmatch(r"[0-9a-f]{16,}", salt):
        raise SystemExit("--salt must be at least 16 lowercase hex characters")

    cells = Cells(cfg)
    names = sorted({c for g in s["calibration"]["groups"].values() if g["enabled"] for c in g["cells"]}
                   | set(s["production"]["cells"]) | {s["examples"]["cell"]}
                   | {c for g in s["practice"]["groups"].values() if g.get("enabled", True) for c in g["cells"]})
    info = {c: _cell_info(cfg, cells, c) for c in names}
    chosen = ma.choose_items(info, cfg, np.random.default_rng(int(cfg["seed"])))      # which trials: public seed
    stg = P["staging"]
    if stg.exists():
        shutil.rmtree(stg)
    stg.mkdir(parents=True)
    man = ma.build_package(stg, cfg, args.name, chosen, _source(cfg, cells, bool(s["display"]["show_auto_boundaries"])),
                           salt, log=lambda m: print(f"[package] {m} ({time.time() - t0:.0f} s)", flush=True))
    real = chosen["real"]
    truth = {r["id"]: cells[r["cell"]]["syl"][r["trial"]] for r in real if r["set"] == "calibration"}
    probs = ma.scan_package(stg, cfg, truth=truth, secrets=[salt])
    if probs:
        print("\n".join(f"[scan] {p}" for p in probs[:50]))
        raise SystemExit(f"blindness scan failed ({len(probs)} problems); the package stays in {rel(stg)}, not usable")
    if P["pkg"].exists():
        shutil.rmtree(P["pkg"])
    stg.rename(P["pkg"])
    key = {
        "name": args.name, "created": datetime.datetime.now().isoformat(timespec="seconds"), "salt": salt,
        "seed": int(cfg["seed"]), "files_sha256": sha256(P["pkg"] / "files.sha256"),
        "items": {r["id"]: {k: r[k] for k in ("set", "group", "cell", "session", "trial")}
                  | {"text": cells[r["cell"]]["trials"][r["trial"]].text} for r in real},
        "practice": {r["id"]: {k: r[k] for k in ("group", "cell", "session", "trial")} for r in chosen["practice"]},
        "examples": {r["id"]: {k: r[k] for k in ("cell", "session", "trial")} for r in chosen["examples"]},
        "settings": s, "code_version": code_version(), "config_sha256": config_sha256(cfg),
        "runtime": runtime_record(cfg, env_set),
    }
    _save(P["key"], key)
    sizes = ma.package_sizes(P["pkg"])
    summary = {"name": args.name, "created": key["created"], "counts": man["counts"],
               "sets": {k: sum(r["set"] == k for r in real) for k in ("calibration", "production")},
               "size_mb": sizes, "scan": "passed", "wall_time_s": time.time() - t0,
               "code_version": key["code_version"], "config_sha256": key["config_sha256"]}
    _save(P["results"] / "package_summary.json", summary)
    print(f"[package] {args.name}: {len(real)} items {summary['sets']}, {len(chosen['practice'])} practice, "
          f"{len(chosen['examples'])} examples, {man['counts']['batches']} batches for {man['counts']['workers']} workers "
          f"(k={man['counts']['redundancy_k']}); size {sizes} MB; scan passed -> {rel(P['pkg'])}, key {rel(P['key'])} "
          f"in {time.time() - t0:.0f} s")
    return 0


def _calibration_truth(cfg, key: dict) -> dict:
    """{real item id: their syllables (frames)} for the calibration items of a key (host only)."""
    v = cfg["validation"]
    cache, out = {}, {}
    for iid, it in key["items"].items():
        if it["set"] != "calibration":
            continue
        s = it["session"]
        if s not in cache:
            cache[s] = _intervals(resolve_path(v["data_dir"]) / s / v["speech_indices_file"].format(session=s),
                                  v["syllable_key"])
        out[iid] = cache[s][it["trial"]]
    return out


def _intervals(path: Path, key: str) -> list:
    """Per trial (K, 2) [onset, offset) bins from a speech_indices file (as validate_alignment.load_intervals)."""
    import numpy as np
    import scipy.io

    m = scipy.io.loadmat(str(path), squeeze_me=True, variable_names=[key])[key]
    return [np.asarray(x, dtype=np.int64).reshape(-1, 2) for x in np.atleast_1d(m)]


def cmd_scan(cfg, env_set, args) -> int:
    from wvx import manual_alignment as ma, syllables as sy

    P = _paths(cfg, args.name)
    key = sy.load_json(P["key"])
    probs = ma.scan_package(P["pkg"], cfg, truth=_calibration_truth(cfg, key), secrets=[key["salt"]])
    if sy.load_json(P["pkg"] / "manifest.json")["package"] != args.name:
        probs.append("manifest names another package")
    if probs:
        print("\n".join(f"[scan] {p}" for p in probs[:50]))
        print(f"[scan] {args.name}: FAILED ({len(probs)} problems)")
        return 1
    print(f"[scan] {args.name}: passed ({len(key['items'])} items checked against the true timings of "
          f"{sum(i['set'] == 'calibration' for i in key['items'].values())} blind items)")
    return 0


# -------------------------------------------------------------------------------------------- ingest


def _snapshot(P, chosen) -> Path:
    if chosen:
        return Path(chosen)
    snaps = sorted(P["answers"].glob("collect_*")) if P["answers"].is_dir() else []
    if not snaps:
        raise SystemExit(f"no answers under {rel(P['answers'])}: run collect on the Mac first")
    return snaps[-1]


def _package_items(pkg: Path) -> dict:
    from wvx import manual_alignment as ma, syllables as sy

    return {p.stem: ma.rule_meta(sy.load_json(p)) for p in sorted((pkg / "items").glob("*.json"))}


def cmd_ingest(cfg, env_set, args) -> int:
    from collections import Counter

    from wvx import manual_alignment as ma, syllables as sy
    from wvx.provenance import code_version, config_sha256

    P = _paths(cfg, args.name, args.label)
    snap = _snapshot(P, args.snapshot)
    adir = snap / "answers"
    man = sy.load_json(P["pkg"] / "manifest.json")
    key = sy.load_json(P["key"])
    items = _package_items(P["pkg"])
    files, ignored = [], []
    for p in sorted(adir.rglob("*")) if adir.is_dir() else []:
        if not p.is_file():
            continue
        r = p.relative_to(adir)
        if len(r.parts) == 2 and p.suffix == ".json" and (not args.workers or re.fullmatch(args.workers, r.parts[0])):
            try:
                files.append((p, r.parts[0], json.loads(p.read_text()), None))
            except (OSError, ValueError) as e:
                files.append((p, r.parts[0], None, f"not valid JSON: {e}"))
        else:
            ignored.append(r.as_posix())
    rules = ma.response_rules(cfg)
    res = ma.ingest(files, items, man, rules)
    agg = ma.aggregate(res["accepted"], items, cfg)
    agr = ma.agreement(res["accepted"], cfg)
    k = int(man["counts"]["redundancy_k"])
    n_ans = Counter(a["item_id"] for a in res["accepted"])
    cover = {}
    for sname in ("calibration", "production"):
        ids = [i for i, v in key["items"].items() if v["set"] == sname]
        cover[sname] = {"n_items": len(ids), "with_answer": sum(i in agg for i in ids),
                        f"with_{k}_or_more": sum(n_ans[i] >= k for i in ids)}
    codes = Counter(c for r in res["rejected"] for c, _ in r["errors"])
    per_worker = {}
    for a in res["accepted"]:
        per_worker.setdefault(a["worker_id"], {"accepted": 0, "rejected": 0})["accepted"] += 1
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    prov = {"name": args.name, "label": args.label, "workers_filter": args.workers, "created": stamp,
            "snapshot": rel(snap), "code_version": code_version(),
            "config_sha256": config_sha256(cfg), "files_sha256": key["files_sha256"]}
    w = P["work"]
    _save(w / "accepted.json", {**prov, "answers": res["accepted"]})
    _save(w / "rejected.json", {**prov, "rejected": res["rejected"], "skipped": res["skipped"], "ignored": ignored})
    _save(w / "aggregated.json", {**prov, "method": ma.settings(cfg)["aggregate"], "items": agg})
    _save(w / "agreement.json", {**prov, **agr})
    summary = {**prov, "n_files": len(files), "n_ignored_files": len(ignored), "n_practice_files": len(res["skipped"]),
               "n_answers_accepted": len(res["accepted"]), "n_rejections": len(res["rejected"]),
               "rejection_codes": dict(sorted(codes.items())), "coverage": cover,
               "answers_per_item": dict(sorted(Counter(n_ans.values()).items())),
               "agreement": {k2: agr[k2] for k2 in ("tolerance_ms", "flag_threshold_ms", "pooled", "n_items_with_2plus_answers",
                                                    "per_worker_median_abs_start_vs_others_ms")} | {"n_flagged": len(agr["flagged"])},
               "accepted_per_worker": {wk: v["accepted"] for wk, v in sorted(per_worker.items())}}
    _save(P["results"] / "ingest_summary.json", summary)
    print(f"[ingest] {args.name}: {len(files)} answer files ({len(ignored)} other files ignored, {len(res['skipped'])} practice), "
          f"{len(res['accepted'])} item answers accepted, {len(res['rejected'])} rejections {dict(codes)}; coverage {cover}; "
          f"agreement {agr['pooled']}, {len(agr['flagged'])} flagged -> {rel(w)}, {rel(P['results'] / 'ingest_summary.json')}")
    return 0


# --------------------------------------------------------------------------------------------- score


def _score_cell(cfg, cells: Cells, cell: str, trials_idx: list, spans: dict, sb, method: str, arm: str, tts_side: str):
    """scripts/syllable_benchmark.score_run on a subset of a cell's trials (method "manual": ``spans`` per trial)."""
    from wvx import alignment as al, targets as tg, wairagkar_data

    c = cells[cell]
    if "scoring" not in c:
        v = cfg["validation"]
        import scipy.io
        import numpy as np

        import validate_alignment as va

        data_dir = resolve_path(v["data_dir"])
        neural_file = wairagkar_data.session_files(data_dir, c["session"], cfg)[0]
        desc = [str(x).strip() for x in np.atleast_1d(scipy.io.loadmat(str(neural_file), squeeze_me=True,
                                                                         variable_names=["trial_description"])["trial_description"])]
        si = data_dir / c["session"] / v["speech_indices_file"].format(session=c["session"])
        corpus = tg.TTSCorpus.load(tg.corpus_path(cfg, cfg["tts"]["voice"], c["corpus"]))
        sil20 = tg.silence_frame(corpus, cfg)
        c["scoring"] = {
            "their_raw": wairagkar_data.load_raw_targets(data_dir, c["session"], cfg), "desc": desc,
            "words": va.load_intervals(si, v["word_key"], len(c["trials"])), "corpus": corpus,
            "pred_env": [al.envelope_db(tg.prediction_timeline(p, t.neural.shape[0], sil20, cfg), cfg)
                         for p, t in zip(c["preds"], c["trials"])],
            "tts_forced": [c["tts"][tg.text_key(t.text)]["frames"] for t in c["trials"]]}
    sc = c["scoring"]

    def sub(x):
        return [x[i] for i in trials_idx]

    records = [{"spans_s": spans[i]} for i in trials_idx] if method == "manual" else None
    out = sb.score_run(method, False, tts_side, sub(c["trials"]), sub(c["preds"]), sc["corpus"], cfg, sub(sc["their_raw"]),
                       sub(c["syl"]), sub(sc["words"]), sub(sc["desc"]), sub(sc["pred_env"]), sub(sc["tts_forced"]),
                       records, c["reorder"], arm=arm)
    if method == "manual":
        bad = [r["trial"] for r in out["per_trial"] if (r["mapping"] or {}).get("pairing") != "one_to_one"]
        if bad:
            raise SystemExit(f"{cell}: manual spans were not paired one to one in {len(bad)} trials (equal_counts setting?)")
    for r in out["per_trial"]:
        r["trial"] = trials_idx[r["trial"]]
    return out


def cmd_score(cfg, env_set, args) -> int:
    from wvx import b2v

    b2v.use_keras2()
    import syllable_benchmark as sb
    from wvx import manual_alignment as ma, syllables as sy
    from wvx.config import deep_merge
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    s = ma.settings(cfg)
    P = _paths(cfg, args.name, args.label)
    cfg = deep_merge(cfg, {"wairagkar_data": {"keys": {"target": cfg["validation"]["their_target_key"]}}})
    key = sy.load_json(P["key"])
    agg = sy.load_json(P["work"] / "aggregated.json")
    acc = sy.load_json(P["work"] / "accepted.json")
    if agg["files_sha256"] != key["files_sha256"]:
        raise SystemExit("aggregated answers belong to another version of the package")
    base = s["scoring"]["baseline"]
    cells = Cells(cfg, with_targets=True)
    by_cell: dict = {}
    for iid, it in key["items"].items():
        if it["set"] == "calibration":
            by_cell.setdefault(it["cell"], {})[it["trial"]] = iid
    workers = sorted({a["worker_id"] for a in acc["answers"]})
    results, details, coverage, manual_r, base_r = {}, {}, {}, {}, {}
    for cell in sorted(by_cell):
        t2id = by_cell[cell]
        sel = sorted(t2id)
        answered = [t for t in sel if t2id[t] in agg["items"]]
        coverage[cell] = len(answered) / len(sel)
        row = {"n_selected": len(sel), "n_answered": len(answered), "coverage": coverage[cell]}
        if answered:
            spans = {t: agg["items"][t2id[t]]["spans"] for t in answered}
            m = _score_cell(cfg, cells, cell, answered, spans, sb, "manual", "B", "forced")
            b = _score_cell(cfg, cells, cell, answered, {}, sb, base["method"], base["arm"], base["tts_side"])
            row["manual"], row["baseline_same_trials"] = m["headline"], b["headline"]
            manual_r[cell], base_r[cell] = m["headline"]["target_r_mean"], b["headline"]["target_r_mean"]
            details[cell] = {"manual": m, "baseline": b, "trial_to_item": {str(t): t2id[t] for t in answered}}
        ball = _score_cell(cfg, cells, cell, sel, {}, sb, base["method"], base["arm"], base["tts_side"])
        row["baseline_all_selected"] = ball["headline"]
        pw = {}
        for w in workers:
            mine = {a["item_id"]: a["spans"] for a in acc["answers"] if a["worker_id"] == w}
            tw = [t for t in sel if t2id[t] in mine]
            if not tw:
                continue
            mw = _score_cell(cfg, cells, cell, tw, {t: mine[t2id[t]] for t in tw}, sb, "manual", "B", "forced")
            bw = _score_cell(cfg, cells, cell, tw, {}, sb, base["method"], base["arm"], base["tts_side"])
            pw[w] = {"n_trials": len(tw), "manual": mw["headline"], "baseline_same_trials": bw["headline"]}
        row["per_worker"] = pw
        results[cell] = row
    decision = ma.adoption(manual_r, base_r, coverage, s["adoption"])
    prov = {"name": args.name, "label": args.label, "workers_filter": agg.get("workers_filter"),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "code_version": code_version(), "config_sha256": config_sha256(cfg), "files_sha256": key["files_sha256"],
            "aggregation": agg["method"], "baseline": base}
    _save(P["work"] / "score_details.json", {**prov, "cells": details})
    summary = {**prov, "adoption": decision, "cells": results}
    _save(P["results"] / "score_summary.json", summary)

    def f(x, nd=3):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    tol = cfg["benchmark"]["onset_tolerance_ms"]
    lines = [f"# Manual alignment `{args.name}`{f' ({args.label})' if args.label else ''}: blind calibration score", "",
             f"Adoption rule (pre-registered): manual is adopted iff its mean target r over {', '.join(s['adoption']['primary_cells'])} "
             f"beats {base['arm']}/{base['method']} on the same trials by >= {s['adoption']['min_improvement']} "
             f"(coverage >= {s['adoption']['min_coverage']}).", "",
             f"**{'ADOPTED' if decision['adopted'] else 'NOT ADOPTED'}**: manual {f(decision['manual_mean'])}, "
             f"baseline {f(decision['baseline_mean'])}, improvement {f(decision['improvement'])} ({decision.get('reason')})", "",
             f"| cell | source | n | onsets <= {tol} ms | span onset err (ms) | target r | chance r | oracle r (n) |",
             "|---|---|---|---|---|---|---|---|"]
    for cell, r in results.items():
        rows = [("manual (aggregated)", r.get("manual")), ("baseline, same trials", r.get("baseline_same_trials")),
                ("baseline, all selected", r["baseline_all_selected"])] + \
               [(f"worker {w}", v["manual"]) for w, v in r["per_worker"].items()] + \
               [(f"baseline on {w}'s trials", v["baseline_same_trials"]) for w, v in r["per_worker"].items()]
        for name, h in rows:
            if h:
                lines.append(f"| {cell} | {name} | {h['n_trials']} | {f(h['onsets_within_tolerance'])} | "
                             f"{f(h['span_onset_error_median_ms'], 0)} | {f(h['target_r_mean'])} | {f(h['chance_r_mean'])} | "
                             f"{f(h['oracle_r_mean'])} ({h['oracle_n_trials']}) |")
    (P["results"] / "score_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# ------------------------------------------------------------------------------------------- targets


def cmd_targets(cfg, env_set, args) -> int:
    from wvx import b2v

    b2v.use_keras2()
    import numpy as np

    from wvx import manual_alignment as ma, syllables as sy, targets as tg
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed
    from wvx.session import SessionData

    set_seed(int(cfg["seed"]))
    s = ma.settings(cfg)
    P = _paths(cfg, args.name, args.label)
    out_dir = P["targets"].resolve()
    protected = {resolve_path(p).resolve() for p in s["targets"]["protected"]}
    if out_dir in protected:
        raise SystemExit(f"{rel(out_dir)} is protected (v1/v2 targets); paths.manual_targets (+ _label) must be a new directory")
    key = sy.load_json(P["key"])
    agg = sy.load_json(P["work"] / "aggregated.json")
    if agg["files_sha256"] != key["files_sha256"]:
        raise SystemExit("aggregated answers belong to another version of the package")
    done = []
    for cell in s["production"]["cells"]:
        t2id = {it["trial"]: iid for iid, it in key["items"].items() if it["set"] == "production" and it["cell"] == cell}
        cells = Cells(cfg)
        c = cells[cell]
        if resolve_path(c["spec"]["model"]) != resolve_path(cfg["kunz"]["decoder"]):
            raise SystemExit(f"{cell}: not decoded with kunz.decoder")
        trials = c["trials"]
        if sorted(t2id) != list(range(len(trials))):
            raise SystemExit(f"{cell}: the key does not cover every trial")
        missing = [t2id[i] for i in range(len(trials)) if t2id[i] not in agg["items"]]
        if missing and s["targets"]["require_all_trials"]:
            raise SystemExit(f"{cell}: {len(missing)} of {len(trials)} trials have no aggregated answer, e.g. {missing[:5]}")
        name = c["header"]["session"]
        out = out_dir / f"{name}.npz"
        if out.exists() and not args.overwrite:
            raise SystemExit(f"{rel(out)} exists; pass --overwrite to replace it")
        corpus = tg.TTSCorpus.load(tg.corpus_path(cfg, cfg["tts"]["voice"], c["corpus"]))
        tts_segs = [c["tts"][tg.text_key(t.text)]["frames"] for t in trials]
        pred_segs = [sy.manual_segments(agg["items"][t2id[i]]["spans"], cfg, t.neural.shape[0]) for i, t in enumerate(trials)]
        _, results = tg.align_trials(None, trials, corpus, cfg, reorder=c["reorder"], log=None, preds=c["preds"],
                                     tts_segments=tts_segs, pred_segments=pred_segs, arm="B")
        cv = code_version()
        speech = []
        for i, (t, res) in enumerate(zip(trials, results)):
            if (res.info or {}).get("pairing") != "one_to_one" or not np.array_equal(res.mapped_segments, pred_segs[i]):
                raise SystemExit(f"{cell} trial {i}: manual spans were not used one to one")
            t.target = tg.decoder_target(res.target_raw, cfg)
            a = agg["items"][t2id[i]]
            t.meta["alignment"] = {
                "method": "manual (wvx.manual_alignment): TTS syllables stretched one to one onto aggregated answers",
                "package": args.name, "label": args.label, "item_id": t2id[i], "n_answers": a["n_answers"], "workers": a["workers"],
                "aggregation": agg["method"], "syllables": {"tts_side": "forced", "arm": "B", "decoder_method": "manual"},
                "voice": cfg["tts"]["voice"], "decoder": rel(resolve_path(c["spec"]["model"])),
                "mapped_segments": res.mapped_segments.tolist(), "code_version": cv, "files_sha256": key["files_sha256"]}
            speech.append(float((res.mapped_segments[:, 1] - res.mapped_segments[:, 0]).sum() / t.neural.shape[0]))
        sess = SessionData(name, -1, trials)
        sess.validate(n_features=int(cfg["model"]["n_features"]), n_targets=int(cfg["model"]["n_targets"]))
        SessionData(name, SessionData.load(Path(cfg["paths"]["kunz_processed"]) / f"{name}.npz").layer, trials).save(out)
        tgt = np.concatenate([t.target for t in trials])
        n_ans = [agg["items"][t2id[i]]["n_answers"] for i in range(len(trials))]
        summary = {"name": args.name, "label": args.label, "created": datetime.datetime.now().isoformat(timespec="seconds"), "session": name,
                   "saved": rel(out), "n_trials": len(trials), "answers_per_trial": {"min": min(n_ans), "median": float(np.median(n_ans)),
                                                                                 "max": max(n_ans)},
                   "speech_fraction_per_trial": {"min": min(speech), "median": float(np.median(speech)), "max": max(speech)},
                   "target_fraction_outside_unit_range": float(np.mean(np.abs(tgt) > 1)),
                   "all_finite": bool(np.all(np.isfinite(tgt))), "code_version": cv, "config_sha256": config_sha256(cfg),
                   "files_sha256": key["files_sha256"]}
        _save(P["results"] / f"targets_{name}_summary.json", summary)
        done.append(summary)
        print(f"[targets] {name}: {len(trials)} trials, answers per trial {summary['answers_per_trial']}, speech fraction "
              f"median {summary['speech_fraction_per_trial']['median']:.2f} -> {rel(out)}")
    return 0


# ------------------------------------------------------------------------------------ deliver, collect


def _host() -> str:
    h = os.environ.get("WVX_HOST")
    if not h:
        f = REPO_ROOT / "scripts" / "remote" / "host.env"
        if f.is_file():
            for line in f.read_text().splitlines():
                m = re.fullmatch(r"\s*(?:export\s+)?WVX_HOST=['\"]?([^'\"\s]+)['\"]?\s*", line)
                if m:
                    h = m.group(1)
    if not h:
        raise SystemExit("set WVX_HOST or create scripts/remote/host.env")
    return h


def _remote(d: dict, inner: str) -> list:
    """ssh command running ``inner`` (bash) in the repo inside the container."""
    import shlex

    return ["ssh", "-o", "BatchMode=yes", _host(), "docker", "exec", "-i", d["container"], "bash", "-c",
            shlex.quote(f"cd {shlex.quote(d['host_repo'])} && {inner}")]


def cmd_deliver(cfg, args) -> int:
    import shlex

    from wvx import manual_alignment as ma

    d = ma.settings(cfg)["delivery"]
    dest_root = Path(os.path.expanduser(args.dest or d["mac_root"])).resolve()
    dest = dest_root / args.name
    if dest.exists():
        raise SystemExit(f"{dest} exists; deliver never overwrites a package (answers may be in it)")
    r = subprocess.run(_remote(d, f"CUDA_VISIBLE_DEVICES= nice -n 19 {d['host_python']} -I scripts/manual_alignment.py "
                                  f"scan --name {shlex.quote(args.name)} --threads {int(d['scan_threads'])}"))
    if r.returncode != 0:
        raise SystemExit("the host's blindness scan failed: nothing delivered")
    stg = dest_root / f".{args.name}.partial"
    if stg.exists():
        shutil.rmtree(stg)
    stg.mkdir(parents=True)
    src_root = Path(cfg["paths"]["manual_alignment"]).relative_to(REPO_ROOT).as_posix()
    cmd = _remote(d, f"tar -C {shlex.quote(src_root)} -cf - {shlex.quote(args.name)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for m in tf:
                kind = "file" if m.isreg() else "dir" if m.isdir() else "link" if (m.issym() or m.islnk()) else "special"
                why = ma.tar_member_problem(m.name, kind, args.name)
                if why:
                    raise SystemExit(f"refused: {m.name}: {why}; nothing delivered")
                tf.extract(m, stg, set_attrs=False, filter="data")
                n += 1
        if proc.wait() != 0:
            raise SystemExit(f"remote tar failed ({proc.returncode}); nothing delivered")
        got = stg / args.name
        probs = ma.scan_package(got, cfg)
        if probs:
            raise SystemExit("local check failed: " + "; ".join(probs[:10]) + "; nothing delivered")
        dest_root.mkdir(parents=True, exist_ok=True)
        got.rename(dest)
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(stg, ignore_errors=True)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"[deliver] {args.name}: {n} archive members, files.sha256 verified, scan passed -> {dest} ({size / 1e6:.1f} MB)")
    return 0


def cmd_collect(cfg, args) -> int:
    import shlex

    from wvx import manual_alignment as ma

    d = ma.settings(cfg)["delivery"]
    src = Path(os.path.expanduser(args.src or d["mac_root"])).resolve() / args.name / "answers"
    if not src.is_dir():
        raise SystemExit(f"{src} not found")
    keep, skipped = [], []
    for p in sorted(src.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p}: symbolic link in answers/; nothing collected")
        if not p.is_file():
            continue
        r = p.relative_to(src)
        if p.suffix not in d["answer_suffixes"] or any(x.startswith(".") for x in r.parts):
            skipped.append(r.as_posix())
        elif p.stat().st_size > int(d["max_answer_file_bytes"]):
            raise SystemExit(f"{p}: larger than {d['max_answer_file_bytes']} bytes; nothing collected")
        else:
            keep.append((p, r))
    if len(keep) > int(d["max_answer_files"]):
        raise SystemExit(f"{len(keep)} files, more than {d['max_answer_files']}; nothing collected")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p, r in keep:
            data = p.read_bytes()
            info = tarfile.TarInfo(f"answers/{r.as_posix()}")
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tf.addfile(info, io.BytesIO(data))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(cfg["paths"]["manual_alignment"]).relative_to(REPO_ROOT).as_posix()
    target = f"{root}/{args.name}_answers/collect_{stamp}"
    inner = (f"test -d {shlex.quote(root + '/' + args.name)} || {{ echo 'no such package on the host' >&2; exit 3; }}; "
             f"mkdir -p {shlex.quote(root + '/' + args.name + '_answers')} && mkdir {shlex.quote(target)} && "
             f"tar -x --no-same-owner -C {shlex.quote(target)}")
    r = subprocess.run(_remote(d, inner), input=buf.getvalue())
    if r.returncode != 0:
        raise SystemExit(f"collect failed on the host ({r.returncode})")
    print(f"[collect] {args.name}: {len(keep)} files ({len(skipped)} other files left behind) -> host {target}")
    return 0


# ---------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["package", "scan", "ingest", "score", "targets", "deliver", "collect"])
    ap.add_argument("--name", required=True, help="package name (<batch_name>)")
    ap.add_argument("--config", default="configs/experiments/manual_alignment.yaml")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--salt", default=None, help="package: reuse a key's salt (reproduces its ids and batches)")
    ap.add_argument("--snapshot", default=None, help="ingest: an answers snapshot folder (default: the latest collect)")
    ap.add_argument("--dest", default=None, help="deliver: Mac folder (default: delivery.mac_root)")
    ap.add_argument("--src", default=None, help="collect: Mac folder (default: delivery.mac_root)")
    ap.add_argument("--label", default=None, help="ingest/score/targets: a separate answer set of the same package "
                                                    "(own work, results and targets folders)")
    ap.add_argument("--workers", default=None, help="ingest: only worker folders matching this regex (e.g. 'a[0-9]{2}')")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if not re.fullmatch(NAME_RE, args.name):
        ap.error(f"--name must match {NAME_RE}")
    if args.label is not None and not re.fullmatch(NAME_RE, args.label):
        ap.error(f"--label must match {NAME_RE}")
    if args.workers is not None:
        re.compile(args.workers)
    if args.step in ("deliver", "collect"):
        from wvx import manual_alignment as ma

        cfg = ma.load_config(args.config)
        return {"deliver": cmd_deliver, "collect": cmd_collect}[args.step](cfg, args)
    cfg, env_set = start(args.config, args.threads)
    return {"package": cmd_package, "scan": cmd_scan, "ingest": cmd_ingest, "score": cmd_score,
            "targets": cmd_targets}[args.step](cfg, env_set, args)


if __name__ == "__main__":
    sys.exit(main())
