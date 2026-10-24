#!/usr/bin/env python
"""D-077: inner -> attempted transforms in front of the b2v2 decoder. Fit per fold, write the projected imagined data.

    PC, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 10 .venv/bin/python -I scripts/kunz_b2v2_project.py --config configs/experiments/b2v2_kunz_project.yaml

For each fold k (test = the imagined block paired with attempted block k): ``wvx.inner_conversion.fit`` on the other
three block pairs (Kunz SessionData, the release's channel order), each transform applied to the test block's
imagined trials. Writes ``<b2v2.project.out_dir>/<transform>/`` (the ``wvx.b2v2_data.B2V2Data`` format: the 200
imagined trials, gids and labels as ``b2v2.data_dir``) and results/b2v2/project/fit.json / .md: transform info and the
inner-vs-attempted AUC per fold (logistic regression on standardized 50 ms frames, tested on the held-out pair):
``auc_fixed`` (trained on the untransformed fitting blocks) and ``auc_retrained`` (trained on the transformed ones).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_kunz_project.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wvx import b2v2_data as bd, inner_conversion as ic, neural_match as nm
    from wvx.provenance import code_version
    from wvx.session import SessionData

    b = cfg["b2v2"]
    k, p = b["kunz"], b["project"]
    pairs = {int(a): int(i) for a, i in p["pairs"].items()}
    names = [n for n in p["transforms"] if n != "none"]
    pool = int(p["settings"]["pool_bins"])
    rng = np.random.default_rng(int(p["settings"]["seed"]))
    S = {c: SessionData.load(resolve_path(k["processed"]) / f"{k['session']}_{c}.npz") for c in ("attempted", "imagined")}
    base = bd.B2V2Data(resolve_path(b["data_dir"]))
    gid_of = {("imagined", t["index"]): t["gid"] for t in base.trials if t["session"] == "imagined"}

    def block(c, blk):
        return [t for t in S[c].trials if t.block == blk]

    def frames(trials, fn=None):
        return np.concatenate([nm.pool_frames(np.asarray(fn(t.neural) if fn else t.neural), pool) for t in trials])

    def subsample(x, n):
        return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

    def discriminator(att, im):
        n = int(p["auc"]["max_frames"])
        xa, xi = subsample(att, n), subsample(im, n)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=float(p["auc"]["C"]), max_iter=2000))
        return clf.fit(np.concatenate([xa, xi]), np.r_[np.ones(len(xa)), np.zeros(len(xi))])

    def auc(clf, att, im):
        y = np.r_[np.ones(len(att)), np.zeros(len(im))]
        return float(roc_auc_score(y, clf.predict_proba(np.concatenate([att, im]))[:, 1]))

    index = {id(t): i for i, t in enumerate(S["imagined"].trials)}
    out = {n: {} for n in names}                                         # transform -> {gid: projected neural}
    report = {"folds": {}}
    for ak, ib in pairs.items():
        fit_pairs = [(block("attempted", a), block("imagined", i)) for a, i in pairs.items() if a != ak]
        maps, info = ic.fit(fit_pairs, p["settings"], seed_key=ak, names=names)
        fit_att = frames([t for atts, _ in fit_pairs for t in atts])
        fit_im = [t for _, ims in fit_pairs for t in ims]
        test_att, test_im = frames(block("attempted", ak)), block("imagined", ib)
        fixed = discriminator(fit_att, frames(fit_im))                     # trained on untransformed data
        fold = {"fit_blocks": [[a, i] for a, i in pairs.items() if a != ak], "test": [ak, ib], "info": info, "auc": {}}
        for n in ["none"] + names:
            fn = maps.get(n)
            te = frames(test_im, fn)
            fold["auc"][n] = {"auc_fixed": auc(fixed, test_att, te),
                              "auc_retrained": auc(discriminator(fit_att, frames(fit_im, fn)), test_att, te)}
            if fn is not None:
                for t in test_im:
                    out[n][gid_of[("imagined", index[id(t)])]] = fn(t.neural)
        report["folds"][str(ak)] = fold
        print(f"fold {ak}->{ib}: " + ", ".join(f"{n} AUC fixed {v['auc_fixed']:.3f} / retrained {v['auc_retrained']:.3f}"
                                               for n, v in fold["auc"].items()), flush=True)
    od = resolve_path(p["out_dir"])
    for n, arrs in out.items():
        d = od / n
        d.mkdir(parents=True, exist_ok=True)
        trials, pos = [], 0
        with open(d / "neural.f16", "wb") as fn_, open(d / "target.f32", "wb") as ft_:
            for g in sorted(arrs):
                x = arrs[g]
                fn_.write(np.asarray(x, np.float16).tobytes())
                ft_.write(np.zeros((len(x), int(b["model"]["n_targets"])), np.float32).tobytes())
                trials.append(dict(base.by_gid[g], start=pos, length=int(len(x))))
                pos += len(x)
        (d / "meta.json").write_text(json.dumps({"created": datetime.datetime.now().isoformat(timespec="seconds"),
                                                 "decision": "D-077", "transform": n, "n_bins": pos,
                                                 "n_features": int(base.meta["n_features"]),
                                                 "n_targets": int(b["model"]["n_targets"]), "trials": trials}) + "\n")
    summ = {n: {m: float(np.mean([f["auc"][n][m] for f in report["folds"].values()]))
                for m in ("auc_fixed", "auc_retrained")} for n in ["none"] + names}
    report.update(summary_auc=summ, settings=p, code_version=code_version(), decision="D-077",
                  created=datetime.datetime.now().isoformat(timespec="seconds"))
    res = resolve_path("results/b2v2/project")
    res.mkdir(parents=True, exist_ok=True)
    (res / "fit.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    lines = ["# D-077 transforms: inner-vs-attempted distinguishability (mean over 4 folds; 0.5 = indistinguishable)",
             "", "| transform | AUC, fixed discriminator | AUC, retrained discriminator |", "|---|---|---|"]
    lines += [f"| {n} | {v['auc_fixed']:.3f} | {v['auc_retrained']:.3f} |" for n, v in summ.items()]
    (res / "fit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
