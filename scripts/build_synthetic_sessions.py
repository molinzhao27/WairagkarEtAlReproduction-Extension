#!/usr/bin/env python
"""Training sessions with SYNTHETIC-voice targets (synthetic-voice decoder, user proposal 2026-10-23).

Host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/build_synthetic_sessions.py --config configs/experiments/synth_sessions.yaml

For every session in ``sessions`` (their training list with its ``session_layer``): each trial's neural features
(channel-reordered for their day <= 195 models, as every decode and training here) and, as the target, the voice's TTS
of the cue placed on THEIR true syllables (``wvx.voice_conversion.oracle_targets``: forced TTS syllables, arm B,
``wvx.targets.decoder_target``; the D-049 ``tts_oracle`` placement). A trial with missing or malformed true syllables, no
TTS, or a failed placement is dropped (counted by reason). Writes ``out_dir``/<session>.npz (``wvx.session.SessionData``, scripts/train.py's input) and
results/synth_sessions/<name>.json (per session: trials kept, target r vs their real targets).
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
    print(f"[synth_sessions {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def build_one(args):
    """One session (runs in a worker process)."""
    config, name, layer = args
    import dataclasses

    import numpy as np

    from word_splice import their_session
    from wvx import kunz_experiment as kx, syllables as sy, targets as tg, voice_conversion as vc, wairagkar_data
    from wvx.session import SessionData

    cfg = load_experiment_config(config)
    s = cfg["synth_sessions"]
    cfg_p = load_experiment_config(s["placement_config"])
    voice = cfg_p["tts"]["voice"]
    corpus = tg.TTSCorpus.load(tg.corpus_path(cfg_p, voice, s["corpus"]))
    table = sy.load_tts_syllables(cfg_p, voice, s["corpus"])
    sess = wairagkar_data.load_session(resolve_path(cfg["sanity"]["data_dir"]), name, layer, cfg)
    _, _, syl, _, _ = their_session(name, cfg, cfg_p)
    from wvx.manual_alignment import wellformed

    trials, r, dropped, reasons = [], [], 0, {"no_or_malformed_syllables": 0, "no_tts": 0, "placement_error": 0}
    for i, t in enumerate(sess.trials):
        if not (len(syl[i]) and wellformed(syl[i], t.neural.shape[0])):
            reasons["no_or_malformed_syllables"] += 1
            continue
        if tg.text_key(t.text) not in table:
            reasons["no_tts"] += 1
            continue
        try:
            p = vc.oracle_targets([t], [syl[i]], corpus, table, cfg_p)[0]
        except ValueError:
            reasons["placement_error"] += 1
            continue
        if p is None:
            reasons["placement_error"] += 1
            continue
        y = np.asarray(p[0], np.float32)
        r.append(kx.feature_scores(y, t.target)["feat_corr_mean"])
        trials.append(dataclasses.replace(
            t, neural=kx.prepare_neural(t.neural, cfg), target=y,
            meta=dict(t.meta, synthetic_target={"voice": voice, "placement": "tts_oracle (their syl_indices, arm B)",
                                                "pairing": p[1]}, channel_order="reordered")))
    out = SessionData(name, int(layer), trials)
    out.validate(n_features=int(cfg["model"]["n_features"]), n_targets=int(cfg["model"]["n_targets"]))
    path = resolve_path(s["out_dir"]) / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    dropped = sum(reasons.values())
    return name, {"layer": int(layer), "n_trials": len(trials), "dropped": dropped, "dropped_reasons": reasons,
                  "target_r_vs_theirs": float(np.nanmean(r)) if r else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/synth_sessions.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import yaml
    from concurrent.futures import ProcessPoolExecutor

    from wvx.provenance import code_version

    s = cfg["synth_sessions"]
    t0 = time.time()
    entries = yaml.safe_load(open(resolve_path(s["sessions_from"])))["sessions"]
    entries = [e for e in entries if int(e["session"].split("day")[1]) <= int(s["max_day"])]
    jobs = [(args.config, e["session"], int(e["session_layer"])) for e in entries]
    report = {}
    with ProcessPoolExecutor(int(s["workers"])) as ex:
        for name, info in ex.map(build_one, jobs):
            report[name] = info
            log(f"{name}: {info['n_trials']} trials (dropped {info['dropped']}), layer {info['layer']}, "
                f"target r vs theirs {info['target_r_vs_theirs']:.3f}")
    out = {"name": s["name"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "code_version": code_version(), "settings": s, "sessions": report,
           "n_trials": sum(v["n_trials"] for v in report.values()), "wall_time_s": time.time() - t0}
    res = resolve_path("results/synth_sessions")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{s['name']}.json").write_text(json.dumps(out, indent=1) + "\n")
    log(f"done: {out['n_trials']} trials in {len(report)} sessions, {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
