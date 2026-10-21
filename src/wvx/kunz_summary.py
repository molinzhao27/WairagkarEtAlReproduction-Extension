"""Across-fold summary of a Kunz experiment run (``scripts/summarize_kunz.py``).

Reads the per-trial tables written by ``scripts/kunz_experiment.py`` (one per fold) and reports, per
model, per fold and pooled over folds:

- per set role (held-out attempted, paired imagined, ...): decoded speech (fraction of the output and
  seconds per trial) and silent outputs,
  ``mel_corr`` vs the cue's TTS, the shuffled-trial null, and the **excess** (real - null, paired per trial:
  same output, own cue vs another cue), plus pooled ASR rates for the real cue and the null cue;
- attempted vs imagined: the raw same-cue difference (paired on the cue within a fold), the difference of the
  excesses (Mann-Whitney U, unpaired: the null partners differ), and the same-cue speech-fraction difference.

The excess is the primary comparison: the null keeps the output fixed, so it controls for how speech-like
the output is (``mel_corr`` uses DTW, which raises the floor for unrelated speech; notes/METRICS.md 2.5).
Pure pandas/numpy/scipy; no TensorFlow.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from wvx import kunz_experiment as kx

# per_trial.csv columns (labels written by scripts/kunz_experiment.py, not tunables)
METRIC, NULL, SPEECH = "tts_mel_corr", "shuffled_trial_tts_mel_corr", "tts_est_speech_fraction"
ATTEMPTED, IMAGINED = "heldout_attempted", "paired_imagined"
_ASR = (("wer", "word_errors", "ref_words"), ("per", "phone_errors", "ref_phones"))


def pooled_rate(df, prefix: str, err: str, tot: str) -> float:
    """Sum of errors / sum of reference units over the trials where both are present."""
    if f"{prefix}{err}" not in df or f"{prefix}{tot}" not in df:
        return float("nan")
    e, n = df[f"{prefix}{err}"].astype(float), df[f"{prefix}{tot}"].astype(float)
    ok = e.notna() & n.notna()
    return float(e[ok].sum() / n[ok].sum()) if ok.any() and n[ok].sum() > 0 else float("nan")


def asr_block(df) -> dict:
    out = {}
    for name, err, tot in _ASR:
        for suf in ("", "_constrained"):
            out[f"{name}{suf}"] = pooled_rate(df, "asr_", err + suf, tot)
            out[f"{name}{suf}_null_shuffled_trial"] = pooled_rate(df, "shuffled_trial_asr_", err + suf, tot)
    return out


def speech_seconds(df, frame_s: float):
    """Decoded speech per trial in seconds: speech fraction x output frames x frame duration."""
    return df[SPEECH].astype(float) * df["n_frames"].astype(float) * frame_s


def role_block(df, frame_s: float) -> dict:
    sp = df[SPEECH].astype(float)
    return {
        "n": int(len(df)),
        "n_silent": int((sp == 0).sum()),
        "speech_fraction": kx.describe(sp),
        "speech_s": kx.describe(speech_seconds(df, frame_s)),
        "output_s": kx.describe(df["n_frames"].astype(float) * frame_s),
        "mel_corr_vs_tts": kx.describe(df[METRIC]),
        "null_shuffled_trial": kx.describe(df[NULL]),
        "excess": kx.paired(df[METRIC].astype(float), df[NULL].astype(float)),
        "asr": asr_block(df),
    }


def _cue_key(df):
    from wvx.metrics import normalize_text

    return list(zip(df["fold"], (normalize_text(t) for t in df["text"])))


def same_cue(a, b, col) -> dict:
    """a[col] - b[col] for trials with the same cue in the same fold (paired; Wilcoxon). ``col``: a column
    name or a function of the frame returning a series."""
    va, vb = (col(a), col(b)) if callable(col) else (a[col], b[col])
    idx = dict(zip(_cue_key(b), vb.astype(float)))
    pairs = [(x, idx[k]) for k, x in zip(_cue_key(a), va.astype(float)) if k in idx]
    if not pairs:
        return kx.paired([], [])
    x, y = zip(*pairs)
    return kx.paired(list(x), list(y))


def excess_difference(a, b) -> dict:
    """Attempted excess vs imagined excess: mean difference and two-sided Mann-Whitney U p (finite trials)."""
    ea = (a[METRIC].astype(float) - a[NULL].astype(float)).dropna().to_numpy()
    eb = (b[METRIC].astype(float) - b[NULL].astype(float)).dropna().to_numpy()
    out = {"n_attempted": int(ea.size), "n_imagined": int(eb.size), "mean_difference": float("nan"),
           "mannwhitney_p": float("nan")}
    if ea.size and eb.size:
        from scipy.stats import mannwhitneyu

        out["mean_difference"] = float(ea.mean() - eb.mean())
        out["mannwhitney_p"] = float(mannwhitneyu(ea, eb, alternative="two-sided").pvalue)
    return out


def scope_block(df, frame_s: float) -> dict:
    out = {"roles": {r: role_block(g, frame_s) for r, g in df.groupby("role", sort=False)}}
    a, b = df[df.role == ATTEMPTED], df[df.role == IMAGINED]
    if len(a) and len(b):
        out["attempted_vs_imagined"] = {
            "mel_corr_same_cue": same_cue(a, b, METRIC),
            "excess_difference": excess_difference(a, b),
            "speech_fraction_same_cue": same_cue(a, b, SPEECH),
            "speech_s_same_cue": same_cue(a, b, lambda d: speech_seconds(d, frame_s)),
        }
    return out


def summarize_run(df, frame_s: float) -> dict:
    """``df``: all folds' per-trial rows with a ``fold`` column; ``frame_s``: seconds per output frame.
    Real-input rows only are scored."""
    real = df[df["input"] == "real"]
    out = {}
    for model, m in real.groupby("model", sort=False):
        res = out[model] = {"folds": {f: scope_block(g, frame_s) for f, g in m.groupby("fold", sort=True)}}
        res["pooled"] = scope_block(m, frame_s)
        folds = [v for v in res["folds"].values() if "attempted_vs_imagined" in v]
        res["consistency"] = {
            "n_folds": len(folds),
            "attempted_excess_gt_imagined_excess": sum(
                v["attempted_vs_imagined"]["excess_difference"]["mean_difference"] > 0 for v in folds),
            "attempted_gt_imagined_same_cue": sum(
                v["attempted_vs_imagined"]["mel_corr_same_cue"].get("mean", float("nan")) > 0 for v in folds),
        }
        shuf = df[(df["input"] == "shuffled_time") & (df["model"] == model)]
        if len(shuf):
            res["shuffled_time_silent"] = {r: {"n": int(len(g)), "n_silent": int((g[SPEECH].astype(float) == 0).sum())}
                                           for r, g in shuf.groupby("role", sort=False)}
    return out


def _f(x, nd=3) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _p(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else (f"{x:.1e}" if x < 1e-3 else f"{x:.3f}")


def markdown(summary: dict, title: str) -> str:
    lines = [f"# {title}", "",
             "mel_corr vs the cue's TTS (LPCNet resynthesis); null = the same output vs another trial's cue; "
             "excess = real - null (paired, Wilcoxon p). Silent outputs (no speech frame) have no mel_corr and are "
             "counted separately. WER: faster-whisper small.en, unconstrained, pooled (null = same transcript vs "
             "another cue). Generated by scripts/summarize_kunz.py.", ""]
    for model, res in summary.items():
        lines += [f"## {model}", "",
                  "| scope | set | n (silent) | speech s / output s | mel_corr | null | excess (p) | WER (null) |",
                  "|---|---|---|---|---|---|---|---|"]
        scopes: Iterable = list(res["folds"].items()) + [("pooled", res["pooled"])]
        for name, sc in scopes:
            for role, r in sc["roles"].items():
                ex = r["excess"]
                lines.append(
                    f"| {name} | {role} | {r['n']} ({r['n_silent']}) | {_f(r['speech_s'].get('mean'), 2)} / "
                    f"{_f(r['output_s'].get('mean'), 1)} | "
                    f"{_f(r['mel_corr_vs_tts'].get('mean'))} | {_f(r['null_shuffled_trial'].get('mean'))} | "
                    f"{_f(ex.get('mean'))} ({_p(ex.get('wilcoxon_p'))}) | "
                    f"{_f(r['asr']['wer'], 2)} ({_f(r['asr']['wer_null_shuffled_trial'], 2)}) |")
        lines += ["", "| scope | attempted - imagined, same cue (p) | excess difference (Mann-Whitney p) | "
                      "speech s difference, same cue (p) |", "|---|---|---|---|"]
        for name, sc in scopes:
            c = sc.get("attempted_vs_imagined")
            if not c:
                continue
            s, e, sp = c["mel_corr_same_cue"], c["excess_difference"], c["speech_s_same_cue"]
            lines.append(f"| {name} | {_f(s.get('mean'))} ({_p(s.get('wilcoxon_p'))}, n={s['n']}) | "
                         f"{_f(e['mean_difference'])} ({_p(e['mannwhitney_p'])}) | "
                         f"{_f(sp.get('mean'), 2)} ({_p(sp.get('wilcoxon_p'))}) |")
        c = res["consistency"]
        lines += ["", f"Folds with attempted excess > imagined excess: {c['attempted_excess_gt_imagined_excess']}"
                      f"/{c['n_folds']}; with attempted > imagined (same cue): {c['attempted_gt_imagined_same_cue']}"
                      f"/{c['n_folds']}."]
        if "shuffled_time_silent" in res:
            st = ", ".join(f"{r} {v['n_silent']}/{v['n']}" for r, v in res["shuffled_time_silent"].items())
            lines += [f"Shuffled-time null: silent outputs {st}."]
        lines.append("")
    return "\n".join(lines)
