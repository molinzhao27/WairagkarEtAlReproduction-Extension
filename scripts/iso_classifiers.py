#!/usr/bin/env python
"""D-080: is word identity decodable from each isolated verbal behavior, by a classifier trained on that behavior?

    host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/iso_classifiers.py --config configs/experiments/b2v2_iso_base.yaml

Trials and windows of ``data/processed/b2v2_iso`` (D-079), seven words (DO_NOTHING excluded). Features: the window in
4 equal segments, the mean of every feature per segment (all arrays: 4 x 512; one array: 4 x 128 = its 64 spike-band
power + 64 threshold-crossing channels). Shrinkage LDA (lsqr, Ledoit-Wolf), leave-one-block-out CV; null: labels
permuted within each block (``--perms``), the CV redone, one-sided p. (LDA = sklearn's lsqr/auto
shrinkage LDA solved by Cholesky, ``lda_predict``.) Writes results/b2v2/iso/classifiers.json / .md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402

ARRAYS = {"all": None, "v6v": 0, "M1": 1, "55b": 2, "d6v": 3}       # release order: 64-channel blocks per feature type


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_iso_base.yaml")
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    from joblib import Parallel, delayed
    import scipy.linalg
    from sklearn.discriminant_analysis import _class_cov

    from wvx import b2v2_data as bd
    from wvx.provenance import code_version

    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    words = sorted({t["text"] for t in data.trials if t["text"]})

    def features(t):
        x, _ = data.trial_arrays(t)
        segs = np.array_split(x, args.segments)
        return np.stack([s.mean(0) for s in segs])                      # (segments, 512)

    def cols(arr_idx):
        if arr_idx is None:
            return np.arange(512)
        return np.r_[64 * arr_idx: 64 * (arr_idx + 1), 256 + 64 * arr_idx: 256 + 64 * (arr_idx + 1)]

    def lda_predict(Xtr, ytr, Xte):
        """sklearn's LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), with its least-squares step replaced
        by a Cholesky solve (the shrunk covariance is positive definite, so the coefficients are the same; ~50x faster
        at 2048 features; checked against sklearn in tests/test_iso_classifiers.py)."""
        classes = np.unique(ytr)
        priors = np.array([np.mean(ytr == c) for c in classes])
        cov = _class_cov(Xtr, ytr, priors, shrinkage="auto")
        means = np.stack([Xtr[ytr == c].mean(0) for c in classes])
        coef = scipy.linalg.solve(cov, means.T, assume_a="pos").T
        intercept = -0.5 * np.sum(means * coef, 1) + np.log(priors)
        return classes[np.argmax(Xte @ coef.T + intercept, 1)]

    def cv_acc(X, y, blocks):
        correct = 0
        for b in np.unique(blocks):
            tr, te = blocks != b, blocks == b
            correct += int(np.sum(lda_predict(X[tr], y[tr], X[te]) == y[te]))
        return correct / len(y)

    def permute_within(y, blocks, rng):
        out = y.copy()
        for b in np.unique(blocks):
            idx = np.where(blocks == b)[0]
            out[idx] = rng.permutation(y[idx])
        return out

    results = {}
    for beh in sorted({t["session"] for t in data.trials}):
        ts = [t for t in data.trials if t["session"] == beh and t["text"]]
        F = np.stack([features(t) for t in ts])                         # (n, segments, 512)
        y = np.array([words.index(t["text"]) for t in ts])
        blocks = np.array([t["block"] for t in ts])
        for arr, ai in ARRAYS.items():
            X = F[:, :, cols(ai)].reshape(len(ts), -1)
            acc = cv_acc(X, y, blocks)
            seeds = np.random.default_rng([args.seed, len(results)]).integers(0, 2 ** 31, args.perms)
            null = np.array(Parallel(n_jobs=args.jobs)(
                delayed(cv_acc)(X, permute_within(y, blocks, np.random.default_rng(s)), blocks) for s in seeds))
            results[f"{beh}/{arr}"] = {"behavior": beh, "array": arr, "n": len(ts), "n_blocks": int(len(set(blocks))),
                                       "accuracy": acc, "null_mean": float(null.mean()),
                                       "null_p95": float(np.percentile(null, 95)),
                                       "p": float((1 + np.sum(null >= acc)) / (1 + args.perms))}
            print(f"{beh:20s} {arr:4s} acc {100 * acc:5.1f}%  null {100 * null.mean():4.1f}% "
                  f"(95th {100 * np.percentile(null, 95):4.1f}%)  p {results[f'{beh}/{arr}']['p']:.3g}", flush=True)
    out = {"decision": "D-080", "words": words, "perms": args.perms, "segments": args.segments,
           "code_version": code_version(), "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "results": results}
    res = resolve_path("results/b2v2/iso")
    res.mkdir(parents=True, exist_ok=True)
    (res / "classifiers.json").write_text(json.dumps(out, indent=1) + "\n")
    behs = sorted({v["behavior"] for v in results.values()})
    lines = ["# D-080: per-behavior word classifiers (shrinkage LDA, leave-one-block-out; chance 14.3%)", "",
             "Accuracy (permutation p) per array.", "",
             "| behavior | n | " + " | ".join(ARRAYS) + " |", "|---|---|" + "---|" * len(ARRAYS)]
    for beh in behs:
        cells = [f"{100 * results[f'{beh}/{a}']['accuracy']:.1f}% ({results[f'{beh}/{a}']['p']:.2g})" for a in ARRAYS]
        lines.append(f"| {beh} | {results[f'{beh}/all']['n']} | " + " | ".join(cells) + " |")
    (res / "classifiers.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
