#!/usr/bin/env python
"""Validation of same-word alignment (D-064, wvx.neural_align) on sessions OTHER than the dev test session.

    PC:   .venv/bin/python -I scripts/word_align_check.py --config configs/experiments/word_align_check.yaml --device gpu

Sentences almost never repeat across sessions (D-064), but words do. For each word, a template = the mean of its
occurrences in the reference session (the model's day-layer session; their true word spans, pooled by 2, resampled to
``length`` frames). In a new session, each word occurrence (resampled likewise) is paired with its word's template,
and a map is fitted so that the new occurrences match the templates: ``word_ridge`` (linear, shrunk toward the
identity), ``word_procrustes`` (rotation, no shrinkage), ``word_layer`` (a copy of the day layer whose embeddings of
the new frames match the original layer's embeddings of the template frames). No audio target is used; word identity
and word timing are.

Same protocol, halves, synthetic drift and evaluation trials as scripts/align_check.py (same seeds), so the two tables
compare row by row: fit on one half of a session (word occurrences of those trials, timing from THEIR word spans:
optimistic for Kunz, which has Astra timing), feature r vs their targets on 40 trials of the other half, paired with
zero-shot. Controls on the reference session: templates from half A; fit and score on half B, undrifted (must not
hurt) and with the synthetic drift (must recover it, mixing included).
Writes results/word_align_check/<name>.json and .md.
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
    print(f"[word_align_check {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="override runtime.device")
    ap.add_argument("--config", default="configs/experiments/word_align_check.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    if args.device:
        cfg["runtime"]["device"] = args.device
    apply_thread_env(cfg)
    from wvx import b2v

    b2v.use_keras2()
    from wvx.training import configure_device

    configure_device(cfg["runtime"].get("device", "auto"))
    import numpy as np

    from drift_check import embed, frechet
    from word_splice import their_session
    from wvx import inference, kunz_experiment as kx, neural_align as na, session_layers as sl, wairagkar_data
    from wvx import word_splice as ws
    from wvx.provenance import code_version

    a = cfg["word_align_check"]
    cfg_p = load_experiment_config(a["placement_config"])
    seed, L = int(a["seed"]), int(a["length"])
    data_dir = resolve_path(cfg["sanity"]["data_dir"])
    path = resolve_path(a["model"])
    day_layer, _ = sl.read_inference_h5(path)
    steps = int(day_layer.input_shape[1])
    model = inference.load_model(path, cfg)
    w_orig = [np.array(w) for w in day_layer.get_weights()]
    t0 = time.time()

    def load(s):
        sess = wairagkar_data.load_session(data_dir, s, 0, cfg)
        _, words, _, cues, _ = their_session(s, cfg, cfg_p)
        xs = [kx.prepare_neural(t.neural, cfg) for t in sess.trials]
        return xs, [t.target for t in sess.trials], words, [ws.words_of(c) for c in cues]

    def halves(n, key):
        p = np.random.default_rng([seed, key]).permutation(n)
        return sorted(p[: n // 2].tolist()), sorted(p[n // 2:].tolist())

    def occ_of(xs, spans, wl, idx):
        return na.word_occurrences([xs[i] for i in idx], [spans[i] if len(spans[i]) else None for i in idx],
                                   [wl[i] for i in idx], L, int(a["margin"]))

    def run(label, xs, targets, spans, wl, fit_idx, ev_idx, templates):
        ev = sorted(np.random.default_rng([seed, 7]).choice(ev_idx, size=min(int(a["n_eval"]), len(ev_idx)),
                                                            replace=False).tolist())
        X, Y, n_occ = na.paired_frames(occ_of(xs, spans, wl, fit_idx), templates)
        out = {"label": label, "n_fit_trials": len(fit_idx), "n_word_occurrences": n_occ, "n_eval": len(ev),
               "variants": {}}
        maps = {"zero_shot": None}
        for v in a["variants"]:
            if v["kind"] == "word_ridge":
                maps[v["name"]] = na.fit_ridge_to_identity(X, Y, float(v["lam_rel"]))
            elif v["kind"] == "word_procrustes":
                maps[v["name"]] = na.fit_procrustes(X, Y)
            elif v["kind"] == "word_layer":
                w, info = na.fit_paired_layer(day_layer, X.astype(np.float32), Y.astype(np.float32), v)
                maps[v["name"]] = ("layer", w, info)
        base = None
        for name, m in maps.items():
            if isinstance(m, tuple):
                na.set_day_layer(model, m[1])
                xin = lambda x: x  # noqa: E731
                minfo = {k: v for k, v in m[2].items() if k != "history"} | {
                    "mse_first_last": [m[2]["history"][0]["mse"], m[2]["history"][-1]["mse"]]}
            else:
                xin = (lambda x: x) if m is None else m
                minfo = {} if m is None else m.info()
            fr = []
            for i in ev:
                pred = kx.decode(model, np.ascontiguousarray(xin(xs[i]), dtype=np.float32), cfg)
                fr.append(kx.feature_scores(pred, inference.align_to_predictions(targets[i], len(pred)))["feat_corr_mean"])
            # fit quality on the paired word frames: how close the mapped occurrences get to their templates
            if isinstance(m, tuple):
                day_layer.set_weights(m[1])
                zx = embed(day_layer, X.astype(np.float32), steps)
                day_layer.set_weights(w_orig)
                zy = embed(day_layer, Y.astype(np.float32), steps)
                fit_err = float(np.mean((zx - zy) ** 2))
            else:
                xm = X if m is None else m(X)
                fit_err = float(np.mean((embed(day_layer, np.asarray(xm, np.float32), steps)
                                         - embed(day_layer, Y.astype(np.float32), steps)) ** 2))
            row = {"feat_r": float(np.mean(fr)), "per_trial": fr, "word_embedding_mse": fit_err, **minfo}
            if base is None:
                base = fr
            else:
                row["vs_zero_shot"] = kx.paired(fr, base)
            out["variants"][name] = row
            na.set_day_layer(model, w_orig)
            day_layer.set_weights(w_orig)
            log(f"{label}: {name:16s} feat r {row['feat_r']:.3f}"
                + (f" ({100 * row['vs_zero_shot']['mean']:+.1f} r x100, p {row['vs_zero_shot']['wilcoxon_p']:.2g})"
                   if "vs_zero_shot" in row else "") + f"; word-embedding MSE {fit_err:.4f}")
        return out

    results = []
    ref_xs, ref_t, ref_sp, ref_wl = load(a["ref_session"])
    log(f"reference {a['ref_session']}: {len(ref_xs)} trials")
    # controls (same halves and drift as align_check): templates from undrifted half A, fit + score on half B
    fi, ei = halves(len(ref_xs), 0)
    temp_A = na.word_templates(occ_of(ref_xs, ref_sp, ref_wl, fi), int(a["min_count"]))
    log(f"control templates (half A): {len(temp_A)} words")
    results.append(run(f"{a['ref_session']} undrifted (identity check)", ref_xs, ref_t, ref_sp, ref_wl, ei, ei, temp_A))
    c = a["control"]
    r = np.random.default_rng([seed, 99])
    F = ref_xs[0].shape[1]
    gain = np.exp(r.normal(0, float(c["log_gain_sd"]), F))
    sd = np.concatenate(ref_xs).std(0)
    offset = r.normal(0, float(c["offset_sd"]), F) * sd
    mix = np.eye(F) + float(c["mix_sd"]) * r.normal(size=(F, F)) / np.sqrt(F)
    drift = lambda x: ((np.asarray(x, np.float64) @ mix) * gain + offset).astype(np.float32)  # noqa: E731
    results.append(run(f"{a['ref_session']} + synthetic drift", [drift(x) for x in ref_xs], ref_t, ref_sp, ref_wl,
                       ei, ei, temp_A))
    templates = na.word_templates(occ_of(ref_xs, ref_sp, ref_wl, list(range(len(ref_xs)))), int(a["min_count"]))
    log(f"templates (all of {a['ref_session']}): {len(templates)} words")
    del ref_xs, ref_t
    for s in a["sessions"]:
        if s == a.get("dev_session"):
            raise SystemExit(f"{s} is the dev test session; it must not be used here")
        xs, tgt, sp, wl = load(s)
        fi, ei = halves(len(xs), int(s.split("day")[1]))
        results.append(run(s, xs, tgt, sp, wl, fi, ei, templates))
        del xs, tgt
    out = {"name": a["name"], "config": args.config, "model": path.name, "ref_session": a["ref_session"],
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
           "settings": a, "results": results, "wall_time_s": time.time() - t0}
    res = resolve_path("results/word_align_check")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{a['name']}.json").write_text(json.dumps(kx.portable(out), indent=1, default=float) + "\n")
    names = list(results[0]["variants"])
    lines = [f"# Same-word alignment check `{a['name']}` (D-064): feature r on held-out halves", "",
             "| session | word occurrences | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for rr in results:
        cells = []
        for n in names:
            v = rr["variants"][n]
            d = v.get("vs_zero_shot")
            cells.append(f"{v['feat_r']:.3f}" + (f" ({100 * d['mean']:+.1f}, p {d['wilcoxon_p']:.2g})" if d else ""))
        lines.append(f"| {rr['label']} | {rr['n_word_occurrences']} | " + " | ".join(cells) + " |")
    (res / f"{a['name']}.md").write_text("\n".join(lines) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/word_align_check/{a['name']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
