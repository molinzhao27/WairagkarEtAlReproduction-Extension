#!/usr/bin/env python
"""D-069 comparison of the kunz_inner_{none,gain,projector,paired} runs (Mac or host, pandas + scipy):

    python scripts/kunz_inner_compare.py > results/kunz_inner_compare.txt

Careful version: per trial, normalized rank q = (own_rank - 1) / (n_cues - 1) (0 = own cue first,
0.5 = chance), since silent outputs leave some cues unscored (n_cues < 50); pairs only trials ranked in both runs.
Also the raw own_rank (pre-registered metric) on the same trials, top-1, excess (mel_corr - length-matched null), WER."""
import glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

def load(run):
    fs = sorted(glob.glob(f"results/kunz_inner_{run}/fold_*/per_trial.csv"))
    if len(fs) < 4:
        return None
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in fs])
    df = df[(df["model"] == "zero_shot") & (df["input"] == "real")].copy()
    df["excess"] = df["tts_mel_corr"] - df["null_lenmatched_mean"]
    out = {}
    for role, g in df.groupby("role"):
        cols = ["own_rank", "n_cues", "excess", "asr_wer"]
        g = g.groupby("trial_index")[cols].mean() if role == "eval_imagined" else g.set_index("trial_index")[cols]
        g["q"] = (g["own_rank"] - 1) / (g["n_cues"] - 1)
        out[role] = g
    return out

def p(x):
    x = np.asarray(x, float)
    return wilcoxon(x).pvalue if np.any(x != 0) else 1.0

runs = {r: load(r) for r in ["none", "gain", "projector", "paired"]}
base = runs["none"]
for role in ["paired_imagined", "eval_imagined", "heldout_attempted"]:
    print(f"== {role}")
    for r, d in runs.items():
        if d is None:
            print(f"  {r:9s} not finished"); continue
        g = d[role]
        s = (f"  {r:9s} n={g['q'].notna().sum():3d}/{len(g)} q {g['q'].mean():.3f} rank {g['own_rank'].mean():5.2f} "
             f"(cues {g['n_cues'].mean():.1f}) top1 {(g['own_rank'] == 1).mean():.3f} excess {g['excess'].mean():+.4f} "
             f"WER {g['asr_wer'].mean():.2f}")
        if r != "none":
            b = base[role]
            both = g.index[g["q"].notna() & b.loc[g.index, "q"].notna()]
            dq = g.loc[both, "q"] - b.loc[both, "q"]
            dr = g.loc[both, "own_rank"] - b.loc[both, "own_rank"]
            de = (g["excess"] - b.loc[g.index, "excess"]).dropna()
            s += (f"\n            vs none on {len(both)} trials: Δq {dq.mean():+.3f} (p {p(dq):.2g}); Δrank {dr.mean():+.2f} "
                  f"(p {p(dr):.2g}); Δexcess {de.mean():+.4f} (p {p(de):.2g}); top1 {int(((g.loc[both,'own_rank']==1)&(b.loc[both,'own_rank']!=1)).sum())} gained / "
                  f"{int(((g.loc[both,'own_rank']!=1)&(b.loc[both,'own_rank']==1)).sum())} lost")
        print(s)
