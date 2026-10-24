#!/usr/bin/env python
"""D-080 post-hoc: is the passive-listening word signal acoustic contamination? Two checks.

    host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/iso_contamination.py --config configs/experiments/b2v2_iso_base.yaml

(a) Cross-condition generalization: shrinkage LDA (sklearn lsqr/auto, Cholesky solve as scripts/iso_classifiers.py)
trained on one behavior's word trials, tested on another's: features = the first 1.0 s of the stimulus (passive:
delay start) or response (active: go start) window, 2 x 500 ms segment means of the 512 features (D-079 data).
Null: training labels permuted (``--perms``). Pairs: listening -> imaginedListening (no audio), the reverse, and
listening -> reading / imagined1stAuditory / attempted.
(b) Latency: features recomputed WITHOUT the 150 ms smoothing kernel (``wvx.kunz.load_isolated`` with a 1-bin
kernel), trials aligned at the delay start; leave-one-block-out LDA on 50 ms windows every 20 ms over 0-1500 ms, for
listening and reading; per-window null (labels permuted within block, ``--perms-latency``).
Writes results/b2v2/iso/contamination.json / .md.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import load_yaml, resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_iso_base.yaml")
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--perms-latency", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    import scipy.linalg
    from joblib import Parallel, delayed
    from sklearn.discriminant_analysis import _class_cov

    from wvx import b2v2_data as bd, kunz
    from wvx.provenance import code_version

    def lda_fit(X, y):
        classes = np.unique(y)
        priors = np.array([np.mean(y == c) for c in classes])
        cov = _class_cov(X, y, priors, shrinkage="auto")
        means = np.stack([X[y == c].mean(0) for c in classes])
        coef = scipy.linalg.solve(cov, means.T, assume_a="pos").T
        return classes, coef, -0.5 * np.sum(means * coef, 1) + np.log(priors)

    def lda_pred(model, X):
        classes, coef, b = model
        return classes[np.argmax(X @ coef.T + b, 1)]

    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    words = sorted({t["text"] for t in data.trials if t["text"]})
    out = {"decision": "D-080 post-hoc", "words": words, "code_version": code_version(),
           "created": datetime.datetime.now().isoformat(timespec="seconds")}

    # (a) cross-condition generalization, D-079 windows (passive: delay start; active: go start)
    def feats(beh):
        ts = [t for t in data.trials if t["session"] == beh and t["text"]]
        X = np.stack([np.concatenate([s.mean(0) for s in np.array_split(data.trial_arrays(t)[0][:100], 2)])
                      for t in ts])
        return X, np.array([words.index(t["text"]) for t in ts])

    F = {b: feats(b) for b in ("listening", "imaginedListening", "reading", "imagined1stAuditory", "attempted")}
    pairs = [("listening", "imaginedListening"), ("imaginedListening", "listening"), ("listening", "reading"),
             ("listening", "imagined1stAuditory"), ("listening", "attempted")]
    rng = np.random.default_rng(args.seed)
    gen = {}
    for a, b in pairs:
        (Xa, ya), (Xb, yb) = F[a], F[b]
        acc = float(np.mean(lda_pred(lda_fit(Xa, ya), Xb) == yb))
        seeds = rng.integers(0, 2 ** 31, args.perms)
        null = np.array(Parallel(n_jobs=args.jobs)(delayed(
            lambda s: float(np.mean(lda_pred(lda_fit(Xa, np.random.default_rng(s).permutation(ya)), Xb) == yb)))(s)
            for s in seeds))
        gen[f"{a}->{b}"] = {"accuracy": acc, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                            "p": float((1 + np.sum(null >= acc)) / (1 + args.perms))}
        print(f"(a) {a:20s} -> {b:20s} acc {100 * acc:5.1f}%  null {100 * null.mean():4.1f}%  p {gen[f'{a}->{b}']['p']:.3g}",
              flush=True)
    out["generalization"] = gen

    # (b) latency on unsmoothed features
    iso = cfg["b2v2"]["iso"]
    adapter = copy.deepcopy(load_yaml(resolve_path(iso["adapter_config"]))["adapter"])
    adapter["normalization"]["params"]["smoothing"].update(length_ms=adapter["bin_ms"], center_ms=0.0)
    lat = {}
    for beh in ("listening", "reading"):
        s = kunz.load_isolated(resolve_path(iso["raw_dir"]) / f"{iso['session']}_{beh}_raw.mat", beh, 0, adapter)
        ts = [t for t in s.trials if t.text]
        y = np.array([words.index(t.text) for t in ts])
        blocks = np.array([t.block for t in ts])
        starts = list(range(0, 146, 2))                                  # 0..1450 ms, 50 ms windows

        def acc_at(st, lab):
            X = np.stack([t.neural[st: st + 5].mean(0) for t in ts])
            c = 0
            for bl in np.unique(blocks):
                tr, te = blocks != bl, blocks == bl
                c += int(np.sum(lda_pred(lda_fit(X[tr], lab[tr]), X[te]) == lab[te]))
            return c / len(lab)

        def perm_within(lab, r):
            o = lab.copy()
            for bl in np.unique(blocks):
                i = np.where(blocks == bl)[0]
                o[i] = r.permutation(lab[i])
            return o

        curve = Parallel(n_jobs=args.jobs)(delayed(acc_at)(st, y) for st in starts)
        nulls = []
        for st in starts:
            sd = rng.integers(0, 2 ** 31, args.perms_latency)
            nulls.append(Parallel(n_jobs=args.jobs)(delayed(acc_at)(st, perm_within(y, np.random.default_rng(q)))
                                                    for q in sd))
        p95 = [float(np.percentile(n, 95)) for n in nulls]
        first = next((10 * st for st, a, q in zip(starts, curve, p95) if a > q), None)
        lat[beh] = {"window_ms": 50, "starts_ms": [10 * st for st in starts], "accuracy": curve, "null_p95": p95,
                    "first_above_null_ms": first, "peak_ms": 10 * starts[int(np.argmax(curve))],
                    "peak_accuracy": float(max(curve))}
        print(f"(b) {beh}: first window above the null's 95th percentile at {first} ms; peak {100 * max(curve):.1f}% at "
              f"{lat[beh]['peak_ms']} ms; first windows: " +
              ", ".join(f"{10 * st}ms {100 * a:.0f}%" for st, a in zip(starts[:8], curve[:8])), flush=True)
    out["latency"] = lat
    res = resolve_path("results/b2v2/iso")
    (res / "contamination.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["# D-080 post-hoc: acoustic-contamination checks", "", "(a) Cross-condition generalization (chance 14.3%):",
             "", "| train -> test | accuracy | null | p |", "|---|---|---|---|"]
    lines += [f"| {k} | {100 * v['accuracy']:.1f}% | {100 * v['null_mean']:.1f}% | {v['p']:.2g} |" for k, v in gen.items()]
    lines += ["", "(b) Latency (unsmoothed, 50 ms windows from the delay start):", ""]
    lines += [f"- {b}: first window above the null's 95th percentile at {v['first_above_null_ms']} ms; peak "
              f"{100 * v['peak_accuracy']:.1f}% at {v['peak_ms']} ms" for b, v in lat.items()]
    (res / "contamination.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
