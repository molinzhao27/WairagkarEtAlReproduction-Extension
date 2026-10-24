#!/usr/bin/env python
"""D-074 check A1: did the voice transfer change the decoder's voice?

    PC, CPU, after the synth_transfer_day195 dev test has cached its tts_oracle targets:
    CUDA_VISIBLE_DEVICES= nice -n 10 .venv/bin/python -I scripts/voice_check.py --config configs/experiments/synth_transfer_day195.yaml

Decodes the day195 TRAINING trials zero-shot with the voice-transfer decoder (``training.init.path``) and with their
day-165 decoder (``--theirs``), and scores each output's feature r against their real targets and against the Alex
targets placed on their true syllables (the dev test's cached ``tts_oracle`` targets, ``data/processed/<name>/``).
If the core now speaks Alex, the new decoder's (r vs Alex) - (r vs theirs) is higher than their decoder's, paired per
trial. Writes results/voice_check/<name>.json and .md.
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

from wvx import b2v  # noqa: E402  (must come before TensorFlow, as in scripts/train.py)

b2v.use_keras2()

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402

THEIRS = "data/raw/wairagkar/t15_neural_data/t15_day172/brain2voice_t15_day165.h5"


def log(msg: str) -> None:
    print(f"[voice_check {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/synth_transfer_day195.yaml")
    ap.add_argument("--theirs", default=THEIRS, help="their decoder to compare with (same day-165 layer)")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np
    import tensorflow as tf
    from scipy.stats import wilcoxon

    import sanity_train_day195 as st
    from wvx import inference, kunz_experiment as kx
    from wvx.provenance import code_version

    t0 = time.time()
    name = cfg["sanity"]["name"]
    dirs = st.Dirs(cfg)
    if not (dirs.targets / "tts_oracle_targets.npz").is_file():
        raise SystemExit(f"no cached tts_oracle targets in {dirs.targets}; run the dev test first")
    sess, _, _, train, _, _ = st.load_day195(cfg)
    alex, alex_info = st.tts_targets("tts_oracle", sess, train, cfg, dirs)
    theirs = {i: sess.trials[i].target for i in train}
    ceiling = [kx.feature_scores(alex[i], theirs[i])["feat_corr_mean"] for i in train]

    models = {"voice_transfer": resolve_path(cfg["training"]["init"]["path"]), "theirs_day165": resolve_path(args.theirs)}
    r = {}
    for m, path in models.items():
        net = inference.load_model(path, cfg)
        rows = {}
        for i in train:
            pred = kx.decode(net, kx.prepare_neural(sess.trials[i].neural, cfg), cfg)
            rows[i] = {k: kx.feature_scores(pred, inference.align_to_predictions(t[i], len(pred)))["feat_corr_mean"]
                       for k, t in (("vs_theirs", theirs), ("vs_alex", alex))}
        r[m] = rows
        log(f"{m}: {len(rows)} trials decoded")
        del net
        tf.keras.backend.clear_session()

    def arr(m, k):
        return np.array([r[m][i][k] for i in train], float)

    summary = {m: {"r_vs_theirs": float(np.nanmean(arr(m, "vs_theirs"))), "r_vs_alex": float(np.nanmean(arr(m, "vs_alex"))),
                   "alex_minus_theirs": float(np.nanmean(arr(m, "vs_alex") - arr(m, "vs_theirs")))} for m in models}

    def paired(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]
        return {"mean_diff": float(d.mean()), "n": int(ok.sum()),
                "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}

    new, old = "voice_transfer", "theirs_day165"
    tests = {
        "shift_new_vs_theirs": paired(arr(new, "vs_alex") - arr(new, "vs_theirs"), arr(old, "vs_alex") - arr(old, "vs_theirs")),
        "r_vs_alex_new_vs_theirs": paired(arr(new, "vs_alex"), arr(old, "vs_alex")),
        "r_vs_theirs_new_vs_theirs": paired(arr(new, "vs_theirs"), arr(old, "vs_theirs")),
    }
    out = {"name": name, "decision": "D-074 A1", "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "code_version": code_version(), "models": {m: str(p) for m, p in models.items()},
           "model_sha256": {m: st._sha256(p) for m, p in models.items()}, "n_trials": len(train),
           "alex_targets": alex_info, "alex_vs_their_targets_r": float(np.nanmean(ceiling)),
           "summary": summary, "paired": tests, "per_trial": {m: {str(i): v for i, v in rr.items()} for m, rr in r.items()},
           "wall_time_s": time.time() - t0}
    res = resolve_path("results/voice_check")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    lines = [f"# Voice check `{name}` (D-074 A1): zero-shot output on {len(train)} day195 training trials", "",
             f"Alex targets (their true syllables) vs their targets: r {np.nanmean(ceiling):.3f}.", "",
             "| decoder | r vs their targets | r vs Alex targets | Alex − theirs |", "|---|---|---|---|"]
    for m, v in summary.items():
        lines.append(f"| {m} | {v['r_vs_theirs']:.3f} | {v['r_vs_alex']:.3f} | {v['alex_minus_theirs']:+.3f} |")
    lines += ["", "| paired (voice_transfer − theirs_day165) | mean diff | p (Wilcoxon) |", "|---|---|---|"]
    for k, v in tests.items():
        lines.append(f"| {k} | {v['mean_diff']:+.3f} | {v['p']:.2g} |")
    (res / f"{name}.md").write_text("\n".join(lines) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/voice_check/{name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
