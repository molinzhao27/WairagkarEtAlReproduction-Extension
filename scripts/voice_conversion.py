#!/usr/bin/env python
"""Fit the TTS -> their-voice target conversion (D-051) on one Wairagkar session and check it on another.

    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/voice_conversion.py --config configs/experiments/voice_conversion.yaml

``voice_conversion`` section of the config:
- ``fit_session`` (t15_day193): every trial with well-formed ``syl_indices``; the Aman TTS placed on THEIR true
  syllables (``wvx.voice_conversion.oracle_targets``, settings of ``placement_config``) paired with their decoder
  targets (loaded as the dev training loads them, settings of ``data_config``: ``target_speech_feats_word``,
  scaled + smoothed); ``ks`` x ``alphas`` chosen by ``n_folds``-fold CV grouped by trial (``seed``); the final
  map is fit on all pairs and saved.
- ``check_session`` (t15_day195): never used to fit or choose. Target-level check only: per-trial r vs their targets
  of the TTS placed on true syllables, before and after conversion; and, when present, of the dev run's cached
  ``tts_oracle`` / ``tts_auto`` training targets (``check_cached``), plus an exact-equality check of
  ``oracle_targets`` against the cached ``tts_oracle`` targets.

Writes ``paths.voice_conversion``/<name>.npz (the map) and results/voice_conversion/<name>.json.
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
    print(f"[voice_conversion {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def valid_spans(segs, n_bins: int) -> bool:
    import numpy as np

    segs = np.asarray(segs)
    return bool(segs.size and segs.ndim == 2 and np.all(segs[:, 1] > segs[:, 0]) and np.all(segs[1:, 0] >= segs[:-1, 1])
                and segs[0, 0] >= 0 and segs[-1, 1] <= n_bins)


def session_pairs(session: str, cfg_d, cfg_p, log=log):
    """(trials, oracle placements or None, validity) of one session."""
    import validate_alignment as va
    from wvx import syllables as sy, targets as tg, voice_conversion as vc, wairagkar_data

    data_dir = resolve_path(cfg_d["sanity"]["data_dir"])
    sess = wairagkar_data.load_session(data_dir, session, 0, cfg_d)
    v = cfg_p["validation"]
    syl = va.load_intervals(data_dir / session / v["speech_indices_file"].format(session=session), v["syllable_key"],
                            len(sess.trials))
    ok = [valid_spans(s, t.neural.shape[0]) for s, t in zip(syl, sess.trials)]
    voice = cfg_p["tts"]["voice"]
    corpus = tg.TTSCorpus.load(tg.corpus_path(cfg_p, voice, session))
    table = sy.load_tts_syllables(cfg_p, voice, session)
    t0 = time.time()
    placed = vc.oracle_targets(sess.trials, syl, corpus, table, cfg_p, valid=ok)
    log(f"{session}: {sum(ok)} of {len(ok)} trials with well-formed syl_indices placed in {time.time() - t0:.0f} s")
    return sess, placed, ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/voice_conversion.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np

    from wvx import kunz_experiment as kx, voice_conversion as vc
    from wvx.provenance import code_version

    c = cfg["voice_conversion"]
    cfg_d, cfg_p = load_experiment_config(c["data_config"]), load_experiment_config(c["placement_config"])
    t_start = time.time()
    sess, placed, ok = session_pairs(c["fit_session"], cfg_d, cfg_p)
    pairs = [(p[0], t.target) for p, t in zip(placed, sess.trials) if p is not None]
    for (x, y) in pairs:
        if x.shape != y.shape:
            raise ValueError(f"placement {x.shape} vs their target {y.shape}")
    before = [vc.trial_r(x, y) for x, y in pairs]
    log(f"fit: {len(pairs)} pairs, {sum(len(x) for x, _ in pairs)} frames; r before conversion {np.nanmean(before):.3f}")
    rows, best = vc.cross_validate(pairs, c["ks"], c["alphas"], int(c["n_folds"]), int(cfg["seed"]))
    for r in rows:
        log(f"  CV k={r['k']:2d} alpha={r['alpha']:g}: r {r['r_mean']:.3f} (median {r['r_median']:.3f})")
    model = vc.fit(pairs, **best)
    cv_best = next(r for r in rows if r["k"] == best["k"] and r["alpha"] == best["alpha"])
    log(f"chosen k={best['k']} alpha={best['alpha']:g}: CV r {cv_best['r_mean']:.3f} (before {np.nanmean(before):.3f})")

    # check session: never used to fit or choose
    csess, cplaced, cok = session_pairs(c["check_session"], cfg_d, cfg_p)
    cpairs = [(p[0], t.target) for p, t in zip(cplaced, csess.trials) if p is not None]
    check = {"session": c["check_session"], "n_trials": len(cpairs),
             "oracle_r_before": kx.describe([vc.trial_r(x, y) for x, y in cpairs]),
             "oracle_r_after": kx.describe([vc.trial_r(vc.apply(model, x), y) for x, y in cpairs])}
    for variant, rel_path in (c.get("check_cached") or {}).items():
        path = resolve_path(rel_path)
        if not path.is_file():
            check[variant] = "not present"
            continue
        with np.load(path, allow_pickle=False) as z:
            keys = [k for k in z.files if k != "header"]
            cached = {int(k[2:]): z[k] for k in keys}
        idx = sorted(cached)
        entry = {"n_trials": len(idx),
                 "r_before": kx.describe([vc.trial_r(cached[i], csess.trials[i].target) for i in idx]),
                 "r_after": kx.describe([vc.trial_r(vc.apply(model, cached[i]), csess.trials[i].target) for i in idx])}
        if variant == "tts_oracle":  # must equal oracle_targets where syl_indices are well formed
            same = [bool(np.array_equal(cached[i], cplaced[i][0])) for i in idx if cplaced[i] is not None]
            entry["identical_to_oracle_targets"] = f"{sum(same)}/{len(same)}"
        check[variant] = entry
    for k2, v2 in check.items():
        if isinstance(v2, dict) and ("r_before" in v2 or "oracle_r_before" in v2):
            b, a = v2.get("r_before", v2.get("oracle_r_before")), v2.get("r_after", v2.get("oracle_r_after"))
            log(f"check {k2}: r {b.get('mean', float('nan')):.3f} -> {a.get('mean', float('nan')):.3f}")

    out_model = resolve_path(cfg["paths"]["voice_conversion"]) / f"{c['name']}.npz"
    meta = {"fit_session": c["fit_session"], "n_pairs": len(pairs), "voice": cfg_p["tts"]["voice"],
            "target_key": cfg_d["wairagkar_data"]["keys"]["target"], "code_version": code_version(),
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    vc.save(out_model, model, meta)
    summary = {"name": c["name"], "config": args.config, **meta, "model_file": str(out_model.relative_to(resolve_path("."))),
               "grid": rows, "chosen": best, "fit_r_before": kx.describe(before), "cv_r_chosen": cv_best,
               "fit_pairings": {str(k): [p[1] for p in placed if p is not None].count(k)
                                for k in set(p[1] for p in placed if p is not None)},
               "check": check, "wall_time_s": time.time() - t_start}
    res = resolve_path("results/voice_conversion")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{c['name']}.json").write_text(json.dumps(kx.portable(summary), indent=1, default=float) + "\n")
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {out_model}, results/voice_conversion/{c['name']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
