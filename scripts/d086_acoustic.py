#!/usr/bin/env python
"""D-086 amendment: sound-only measures of the decoded inner-speech audio (no phonemes, no ASR).

    PC, project venv, CPU:
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I scripts/d086_acoustic.py --config configs/experiments/b2v2_kunz_d086.yaml

For every arm of ``b2v2.d086.arms`` (its audio: results/audio/b2v2/<run>/<name>/, or the post-processed variant) and
each of the 200 held-out imagined trials: log-mel spectrogram (``acoustic.mel``), silent frames dropped (below the
clip's peak minus ``trim_db``), MFCC 1..``n_mfcc - 1`` of the log-mel. References: ``attempted`` = the decoded
attempted audio of the same sentence (``acoustic.attempted``: RUN:NAME per fold, the trial of the imagined trial's
``paired_block`` with the same text); ``clean`` = a reading of each sentence in the decoder's target voice
(``acoustic.clean``: wav dir + texts json + QC json; failed sentences dropped).
- MCD (dB): DTW on the MFCCs (euclidean), mean over the path of (10 / ln 10) sqrt(2 sum diff^2);
- spectral correlation: along the same path, Pearson r per mel band across aligned frames, mean over bands;
- acoustic identification: MCD to each of the 50 references of the trial's block; top-1 / mean rank of its own
  sentence; null: a random candidate per trial as the true one (``perms`` draws); length-only baseline: rank by the
  difference in speech-frame count.
Writes ``b2v2.d086.out`` (default results/b2v2/d086)/acoustic.json / .md.
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

A = None


def _init(cfg_path):
    global A
    A = load_experiment_config(cfg_path)["b2v2"]["d086"]["acoustic"]


def features(path: str):
    """(log-mel (T, n_mels), MFCC 1.. (T, n_mfcc - 1)) of the speech frames of one wav."""
    import librosa
    import numpy as np
    import soundfile as sf

    m = A["mel"]
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != int(m["sr"]):
        w = librosa.resample(w, orig_sr=sr, target_sr=int(m["sr"]))
    S = librosa.feature.melspectrogram(y=w, sr=int(m["sr"]), n_fft=int(m["n_fft"]), hop_length=int(m["hop"]),
                                       n_mels=int(m["n_mels"]), fmax=float(m["fmax"]))
    db = 10 * np.log10(S + 1e-10)
    keep = db.max(0) > db.max() - float(A["trim_db"])
    lm = np.log(S[:, keep] + 1e-10)
    mf = librosa.feature.mfcc(S=lm, n_mfcc=int(A["n_mfcc"]))[1:]
    return lm.T.astype(np.float32), mf.T.astype(np.float64)


def mcd_path(a, b):
    """(MCD in dB along the DTW path, path) between MFCC sequences a (Ta, D) and b (Tb, D)."""
    import librosa
    import numpy as np

    _, wp = librosa.sequence.dtw(X=a.T, Y=b.T, metric="euclidean")
    d = a[wp[:, 0]] - b[wp[:, 1]]
    return float(np.mean(10 / np.log(10) * np.sqrt(2 * (d ** 2).sum(1)))), wp


def spec_corr(la, lb, wp):
    import numpy as np

    x, y = la[wp[:, 0]], lb[wp[:, 1]]
    x, y = x - x.mean(0), y - y.mean(0)
    r = (x * y).sum(0) / np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0) + 1e-12)
    return float(np.mean(r))


def feat_job(path):
    return path, features(path)


FEAT: dict = {}                                               # path -> features, inherited by forked workers


def trial_job(job):
    """All measures of one decoded clip (paths only; features from ``FEAT``): own references (MCD, r) and MCD to
    every candidate of its block."""
    key, path, own, cands = job
    la, ma = FEAT[path]
    out = {"key": key, "n_frames": len(ma)}
    for ref_name, ref in own.items():
        if ref is None:
            continue
        m, wp = mcd_path(ma, FEAT[ref][1])
        out[f"mcd_{ref_name}"], out[f"r_{ref_name}"] = m, spec_corr(la, FEAT[ref][0], wp)
    for ref_name, cl in cands.items():
        out[f"cand_{ref_name}"] = [(t, mcd_path(ma, FEAT[c][1])[0], abs(len(ma) - len(FEAT[c][1]))) for t, c in cl]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    _init(args.config)
    import multiprocessing as mp

    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from scipy.stats import wilcoxon

    from wvx import b2v2_data as bd
    from wvx.provenance import code_version

    S = cfg["b2v2"]["d086"]
    data = bd.B2V2Data(resolve_path(cfg["b2v2"]["data_dir"]))
    rng = np.random.default_rng(int(A["seed"]))

    def decode_gids(spec):
        run, name = spec.split(":")
        with np.load(resolve_path("data/processed/b2v2_preds") / run / f"{name}.npz", allow_pickle=False) as z:
            return json.loads(str(z["header"]))["gids"]

    # clips per arm
    clips = {}
    for an, arm in list(S["arms"].items()) + [("attempted", {"preds": A["attempted"]})]:
        c = {}
        for spec in arm["preds"]:
            run, name = spec.split(":")
            for g in decode_gids(spec):
                d = (resolve_path("results/audio/b2v2_postproc") / arm["postproc"]["set"] / arm["postproc"]["variant"]
                     if "postproc" in arm else resolve_path("results/audio/b2v2") / run / name)
                c[int(g)] = str(d / f"g{g}.wav")
        clips[an] = c
    att = clips["attempted"]                                  # attempted gid -> wav
    by_block = {}
    for g in att:
        by_block.setdefault(data.by_gid[g]["block"], []).append(g)
    cl = A["clean"]
    texts = json.loads(resolve_path(cl["texts"]).read_text())
    bad = set(int(i) for i in json.loads(resolve_path(cl["qc"]).read_text()).get("redo", []))
    clean = {t: str(resolve_path(cl["dir"]) / f"s{i:04d}.wav") for i, t in enumerate(texts) if i not in bad}

    paths = sorted({p for c in clips.values() for p in c.values()} | set(clean.values()))
    with ProcessPoolExecutor(int(A["workers"]), initializer=_init, initargs=(args.config,)) as ex:
        F = dict(ex.map(feat_job, paths, chunksize=8))
    FEAT.update(F)
    print(f"features of {len(F)} clips", flush=True)

    def refs_for(g):
        tr = data.by_gid[g]
        blk = tr.get("paired_block") or tr["block"]
        own_att = next((a for a in by_block[blk] if data.by_gid[a]["text"] == tr["text"]), None)
        cands_att = [(data.by_gid[a]["text"], att[a]) for a in by_block[blk]]
        cands_clean = [(data.by_gid[a]["text"], clean[data.by_gid[a]["text"]]) for a in by_block[blk]
                       if data.by_gid[a]["text"] in clean]
        return tr["text"], own_att, cands_att, cands_clean

    jobs = []
    for an, c in clips.items():
        for g, p in c.items():
            text, own_att, ca, cc = refs_for(g)
            own = {"attempted": att[own_att] if (own_att is not None and an != "attempted") else None,
                   "clean": clean.get(text)}
            cands = {"clean": cc} if an == "attempted" else {"attempted": ca, "clean": cc}
            jobs.append(((an, g, text), p, own, cands))
    with ProcessPoolExecutor(int(A["workers"]), mp_context=mp.get_context("fork"), initializer=_init,
                             initargs=(args.config,)) as ex:
        rows = list(ex.map(trial_job, jobs, chunksize=4))

    res = {}
    for an in clips:
        R = [r for r in rows if r["key"][0] == an]
        out = {"n": len(R)}
        for ref in ("attempted", "clean"):
            for m in ("mcd", "r"):
                v = [r[f"{m}_{ref}"] for r in R if f"{m}_{ref}" in r]
                if v:
                    out[f"{m}_{ref}"] = float(np.mean(v))
            ck = f"cand_{ref}"
            if any(ck in r for r in R):
                ranks, lranks, allr = [], [], []
                for r in R:
                    text = r["key"][2]
                    c = r[ck]
                    if text not in [t for t, _, _ in c]:
                        continue
                    d = np.array([x for _, x, _ in c])
                    ln = np.array([x for _, _, x in c])
                    i = [t for t, _, _ in c].index(text)
                    rk = 1 + (d < d[i]).sum() + 0.5 * ((d == d[i]).sum() - 1)
                    lrk = 1 + (ln < ln[i]).sum() + 0.5 * ((ln == ln[i]).sum() - 1)
                    ranks.append(rk)
                    lranks.append(lrk)
                    allr.append(np.argsort(np.argsort(d)) + 1)           # every candidate's rank (for the null)
                ranks = np.array(ranks)
                null = np.array([[a[rng.integers(len(a))] for a in allr] for _ in range(int(A["perms"]))])
                out[f"id_{ref}"] = {"n": len(ranks), "n_candidates": float(np.mean([len(a) for a in allr])),
                                    "top1": float(np.mean(ranks == 1)), "mean_rank": float(ranks.mean()),
                                    "p_top1": float((1 + np.sum((null == 1).mean(1) >= np.mean(ranks == 1)))
                                                    / (1 + len(null))),
                                    "p_mean_rank": float((1 + np.sum(null.mean(1) <= ranks.mean())) / (1 + len(null))),
                                    "length_only_top1": float(np.mean(np.array(lranks) == 1)),
                                    "length_only_mean_rank": float(np.mean(lranks))}
        out["_per_trial"] = {r["key"][1]: {k: r[k] for k in r if k.startswith(("mcd_", "r_"))} for r in R}
        res[an] = out
    base = S["baseline"]
    for an in res:
        if an in (base, "attempted"):
            continue
        for k in ("mcd_attempted", "r_attempted", "mcd_clean", "r_clean"):
            a, b = res[an]["_per_trial"], res[base]["_per_trial"]
            g = [x for x in a if x in b and k in a[x] and k in b[x]]
            d = np.array([a[x][k] - b[x][k] for x in g])
            if len(d):
                res[an][f"{k}_vs_baseline"] = {"mean_diff": float(d.mean()), "n": len(d),
                                               "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0}
    for an in res:
        res[an].pop("_per_trial")
    dst = resolve_path(S.get("out", "results/b2v2/d086"))
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "acoustic.json").write_text(json.dumps({
        "decision": S.get("decision", "D-086") + " (sound-only measures)", "config": args.config, "code_version": code_version(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"), "settings": A, "arms": res}, indent=1) + "\n")

    def f(x, k, fmt):
        return fmt.format(x[k]) if k in x else "–"

    def vs(x, k):
        v = x.get(f"{k}_vs_baseline")
        return f" ({v['mean_diff']:+.2f}, p {v['p']:.2g})" if v else ""

    lines = [f"# {S.get('decision', 'D-086')} sound-only measures (decoded audio; 200 held-out inner-speech trials per "
             "arm)", "",
             "MCD = mel-cepstral distortion (dB, lower = closer); r = spectral correlation (higher = closer); "
             "both DTW-aligned, speech frames only. In brackets: per-trial change vs the baseline (p). Acoustic "
             "identification: the own sentence's rank among the block's 50 by MCD (chance top-1 2%, mean rank 25.5).",
             "", "| arm | MCD to attempted | r to attempted | MCD to clean | r to clean | id. vs attempted: top-1 / "
             "mean rank (p) | id. vs clean: top-1 / mean rank (p) |", "|---|---|---|---|---|---|---|"]
    for an, x in res.items():
        ids = []
        for ref in ("attempted", "clean"):
            i = x.get(f"id_{ref}")
            ids.append(f"{100 * i['top1']:.1f}% / {i['mean_rank']:.1f} (p {i['p_mean_rank']:.2g}; length only "
                       f"{100 * i['length_only_top1']:.1f}% / {i['length_only_mean_rank']:.1f})" if i else "–")
        lines.append(f"| {an} | {f(x, 'mcd_attempted', '{:.2f}')}{vs(x, 'mcd_attempted')} | "
                     f"{f(x, 'r_attempted', '{:.3f}')}{vs(x, 'r_attempted')} | {f(x, 'mcd_clean', '{:.2f}')}"
                     f"{vs(x, 'mcd_clean')} | {f(x, 'r_clean', '{:.3f}')}{vs(x, 'r_clean')} | {ids[0]} | {ids[1]} |")
    (dst / "acoustic.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
