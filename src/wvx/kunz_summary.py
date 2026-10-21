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

**Cue nulls** (tables scored with them: columns ``null_lenmatched_mean``, ``null_all_mean``, ``own_rank``,
``n_cues``, ``est_speech_s``, ``early_speech_s``; ``wvx.kunz_experiment.cue_null_stats``). Then also, per role:
the excess under each null (length-matched other cues, all other cues, the shuffled-trial partner; the
configured primary null first), identification of the own cue (top-1 rate, median rank, exact rank-sum test
against uniform ranks) and early-window speech seconds; per scope the attempted-minus-imagined excess
difference under each null; per-fold p values Holm-adjusted across the folds; and a **sensitivity analysis**
restricted to outputs with at least ``min_speech_s`` of decoded speech. Tables without these columns (older
runs) give exactly the older summary.

Pure pandas/numpy/scipy; no TensorFlow.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np

from wvx import kunz_experiment as kx

# per_trial.csv columns (labels written by scripts/kunz_experiment.py, not tunables)
METRIC, NULL, SPEECH = "tts_mel_corr", "shuffled_trial_tts_mel_corr", "tts_est_speech_fraction"
ATTEMPTED, IMAGINED = "heldout_attempted", "paired_imagined"
CUE_NULL_COLUMNS = ("null_lenmatched_mean", "null_all_mean", "own_rank", "n_cues", "est_speech_s", "early_speech_s")
_ASR = (("wer", "word_errors", "ref_words"), ("per", "phone_errors", "ref_phones"))
# The base config of every Kunz experiment: settings used when a caller passes none (cf. configs/metrics.yaml
# for wvx.metrics). scripts/summarize_kunz.py always passes the run's own config.
BASE_CONFIG = "configs/experiments/kunz_full.yaml"
_SHORT = {"lenmatched": "length-matched", "all": "all cues", "shuffled_trial": "shuffled trial"}


def summary_settings(cfg: Mapping) -> dict:
    """The settings of the cue-null summaries, from a resolved experiment config."""
    e = cfg["experiment"]
    return {"primary_null": e["summary"]["primary_null"], "min_speech_s": float(e["summary"]["min_speech_s"]),
            "length_matched_k": int(e["nulls"]["length_matched_k"]),
            "early_window_s": float(e["evaluation"]["early_window_s"])}


def default_settings() -> dict:
    from wvx.experiment_config import load_experiment_config

    return summary_settings(load_experiment_config(BASE_CONFIG))


def has_cue_nulls(df) -> bool:
    """Every row scored with the cue nulls (``n_cues`` is 0, not missing, for a silent output)."""
    return len(df) > 0 and all(c in df for c in CUE_NULL_COLUMNS) and bool(df["n_cues"].notna().all())


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


def _null_excess(df, order) -> dict:
    """Excess of the own cue over each null (paired per output, Wilcoxon), in ``order``."""
    return {n: {"null": kx.describe(df[kx.NULL_COLUMNS[n]]),
                "excess": kx.paired(df[METRIC].astype(float), df[kx.NULL_COLUMNS[n]].astype(float))} for n in order}


def _identification(df) -> dict:
    return kx.identification(df["own_rank"].astype(float), df["n_cues"].astype(float))


def role_block(df, frame_s: float, order: Optional[list] = None) -> dict:
    sp = df[SPEECH].astype(float)
    out = {
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
    if order is not None:
        out["nulls"] = _null_excess(df, order)
        out["identification"] = _identification(df)
        out["early_speech_s"] = kx.describe(df["early_speech_s"])
        if "peak_dbfs" in df:
            out["peak_dbfs"] = kx.describe(df["peak_dbfs"])
    return out


def _cue_key(df):
    from wvx.metrics import normalize_text

    return list(zip(df["fold"], (normalize_text(t) for t in df["text"])))


def same_cue(a, b, col) -> dict:
    """a[col] - b[col] for trials with the same cue in the same fold (paired; Wilcoxon). ``col``: a column
    name or a function of the frame returning a series. A (fold, cue) repeated within ``a`` or ``b`` cannot
    be paired unambiguously and raises ValueError."""
    va, vb = (col(a), col(b)) if callable(col) else (a[col], b[col])
    ka = _cue_key(a)
    kx.index_unique(ka, "same_cue: (fold, cue) of the first rows")
    idx = kx.index_unique(_cue_key(b), "same_cue: (fold, cue) of the second rows")
    vb = list(vb.astype(float))
    pairs = [(x, vb[idx[k]]) for k, x in zip(ka, va.astype(float)) if k in idx]
    if not pairs:
        return kx.paired([], [])
    x, y = zip(*pairs)
    return kx.paired(list(x), list(y))


def excess_difference(a, b, null: str = NULL) -> dict:
    """Attempted excess vs imagined excess under the null column ``null``: mean difference and two-sided
    Mann-Whitney U p (finite trials)."""
    ea = (a[METRIC].astype(float) - a[null].astype(float)).dropna().to_numpy()
    eb = (b[METRIC].astype(float) - b[null].astype(float)).dropna().to_numpy()
    u = kx.unpaired(ea, eb)
    return {"n_attempted": u["n_a"], "n_imagined": u["n_b"], "mean_difference": u["mean_difference"],
            "mannwhitney_p": u["mannwhitney_p"]}


def sensitivity_block(df, order, min_speech_s: float) -> dict:
    """The cue-null tests restricted to outputs with >= ``min_speech_s`` of decoded speech (a sensitivity
    analysis: the selection depends on the output itself, so it is never the primary test)."""
    keep = df[df["est_speech_s"].astype(float) >= min_speech_s]
    out = {"analysis": "sensitivity analysis, not the primary test: only outputs with >= min_speech_s of decoded "
                       "speech (selected on the output itself)",
           "min_speech_s": min_speech_s,
           "roles": {r: {"n": int((df.role == r).sum()), "n_kept": int(len(g)),
                         "nulls": {n: v["excess"] for n, v in _null_excess(g, order).items()},
                         "identification": _identification(g)}
                     for r, g in keep.groupby("role", sort=False)}}
    a, b = keep[keep.role == ATTEMPTED], keep[keep.role == IMAGINED]
    if len(a) and len(b):
        out["attempted_vs_imagined"] = {"excess_difference_by_null": {n: excess_difference(a, b, kx.NULL_COLUMNS[n])
                                                                      for n in order}}
    return out


def scope_block(df, frame_s: float, settings: Optional[dict] = None) -> dict:
    order = kx.null_order(settings["primary_null"]) if settings else None
    out = {"roles": {r: role_block(g, frame_s, order) for r, g in df.groupby("role", sort=False)}}
    a, b = df[df.role == ATTEMPTED], df[df.role == IMAGINED]
    if len(a) and len(b):
        out["attempted_vs_imagined"] = {
            "mel_corr_same_cue": same_cue(a, b, METRIC),
            "excess_difference": excess_difference(a, b),
            "speech_fraction_same_cue": same_cue(a, b, SPEECH),
            "speech_s_same_cue": same_cue(a, b, lambda d: speech_seconds(d, frame_s)),
        }
        if order is not None:
            out["attempted_vs_imagined"]["excess_difference_by_null"] = {
                n: excess_difference(a, b, kx.NULL_COLUMNS[n]) for n in order}
            out["attempted_vs_imagined"]["early_speech_s_same_cue"] = same_cue(a, b, "early_speech_s")
    if order is not None:
        out["sensitivity"] = sensitivity_block(df, order, settings["min_speech_s"])
    return out


def _holm_across_folds(folds: dict, order) -> None:
    """Add Holm-adjusted p values, the family being the same test in every fold (one per block pair)."""
    names = list(folds)

    def adjust(get, key):
        entries = [get(folds[f]) for f in names]
        ps = [e.get(key, float("nan")) if e is not None else float("nan") for e in entries]
        for e, p in zip(entries, kx.holm(ps)):
            if e is not None:
                e[f"{key}_holm"] = p

    roles = sorted({r for f in names for r in folds[f]["roles"]})
    for r in roles:
        def role(sc, r=r):
            return sc["roles"].get(r)
        adjust(lambda sc: (role(sc) or {}).get("identification"), "uniform_rank_p")
        for n in order:
            adjust(lambda sc, n=n: ((role(sc) or {}).get("nulls") or {}).get(n, {}).get("excess"), "wilcoxon_p")
            adjust(lambda sc, n=n, r=r: sc["sensitivity"]["roles"].get(r, {}).get("nulls", {}).get(n), "wilcoxon_p")
    for n in order:
        adjust(lambda sc, n=n: sc.get("attempted_vs_imagined", {}).get("excess_difference_by_null", {}).get(n),
               "mannwhitney_p")
        adjust(lambda sc, n=n: sc["sensitivity"].get("attempted_vs_imagined", {})
               .get("excess_difference_by_null", {}).get(n), "mannwhitney_p")


def summarize_run(df, frame_s: float, settings: Optional[dict] = None) -> dict:
    """``df``: all folds' per-trial rows with a ``fold`` column; ``frame_s``: seconds per output frame;
    ``settings``: :func:`summary_settings` of the run's config (None: those of ``BASE_CONFIG``; only used
    when the table has the cue-null columns). Real-input rows only are scored."""
    real = df[df["input"] == "real"]
    out = {}
    for model, m in real.groupby("model", sort=False):
        st = None
        if has_cue_nulls(m):
            st = dict(settings or default_settings())
            st["null_order"] = kx.null_order(st["primary_null"])
        res = out[model] = {"folds": {f: scope_block(g, frame_s, st) for f, g in m.groupby("fold", sort=True)}}
        res["pooled"] = scope_block(m, frame_s, st)
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
        if st is not None:
            res["settings"] = st
            res["consistency_by_null"] = {
                n: sum(v["attempted_vs_imagined"]["excess_difference_by_null"][n]["mean_difference"] > 0
                       for v in folds) for n in st["null_order"]}
            _holm_across_folds(res["folds"], st["null_order"])
    return out


def _f(x, nd=3) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _p(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else (f"{x:.1e}" if x < 1e-3 else f"{x:.3f}")


def _ph(d: Mapping, key: str, fold: bool) -> str:
    """p, plus the Holm-adjusted p in brackets for a per-fold entry."""
    return _p(d.get(key)) + (f" [{_p(d.get(key + '_holm'))}]" if fold and f"{key}_holm" in d else "")


def _legacy_model_section(model: str, res: dict) -> list[str]:
    lines = [f"## {model}", "",
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
    return lines


def _ident_cells(i: Mapping, fold: bool) -> str:
    return (f"{_f(i.get('top1_rate'), 2)} ({_f(i.get('top1_chance'), 2)}) | "
            f"{_f(i.get('median_rank'), 1)} / {_f(i.get('median_n_cues'), 0)} ({_ph(i, 'uniform_rank_p', fold)})")


def _cue_model_section(model: str, res: dict) -> list[str]:
    st = res["settings"]
    order = st["null_order"]
    scopes: Iterable = [(f, sc, True) for f, sc in res["folds"].items()] + [("pooled", res["pooled"], False)]
    nh = " | ".join(f"excess {_SHORT[n]} (p)" for n in order)
    lines = [f"## {model}", "",
             f"| scope | set | n (silent) | speech s / output s | speech s in first {_f(st['early_window_s'], 0)} s | "
             f"mel_corr | {nh} | top-1 (chance) | median rank / cues (p) | WER (null) |",
             "|---" * (9 + len(order)) + "|"]
    for name, sc, fold in scopes:
        for role, r in sc["roles"].items():
            ex = " | ".join(f"{_f(r['nulls'][n]['excess'].get('mean'))} ({_ph(r['nulls'][n]['excess'], 'wilcoxon_p', fold)})"
                            for n in order)
            lines.append(
                f"| {name} | {role} | {r['n']} ({r['n_silent']}) | {_f(r['speech_s'].get('mean'), 2)} / "
                f"{_f(r['output_s'].get('mean'), 1)} | {_f(r['early_speech_s'].get('mean'), 2)} | "
                f"{_f(r['mel_corr_vs_tts'].get('mean'))} | {ex} | {_ident_cells(r['identification'], fold)} | "
                f"{_f(r['asr']['wer'], 2)} ({_f(r['asr']['wer_null_shuffled_trial'], 2)}) |")
    dh = " | ".join(f"excess difference {_SHORT[n]} (MW p)" for n in order)
    lines += ["", f"| scope | attempted - imagined, same cue (p) | {dh} | speech s difference, same cue (p) | "
                  f"early speech s difference, same cue (p) |", "|---" * (4 + len(order)) + "|"]
    for name, sc, fold in scopes:
        c = sc.get("attempted_vs_imagined")
        if not c:
            continue
        s, sp, es = c["mel_corr_same_cue"], c["speech_s_same_cue"], c["early_speech_s_same_cue"]
        d = " | ".join(f"{_f(c['excess_difference_by_null'][n]['mean_difference'])} "
                       f"({_ph(c['excess_difference_by_null'][n], 'mannwhitney_p', fold)})" for n in order)
        lines.append(f"| {name} | {_f(s.get('mean'))} ({_p(s.get('wilcoxon_p'))}, n={s['n']}) | {d} | "
                     f"{_f(sp.get('mean'), 2)} ({_p(sp.get('wilcoxon_p'))}) | "
                     f"{_f(es.get('mean'), 2)} ({_p(es.get('wilcoxon_p'))}) |")
    c, cn = res["consistency"], res["consistency_by_null"]
    lines += ["", "Folds with attempted excess > imagined excess: "
              + ", ".join(f"{_SHORT[n]} {cn[n]}/{c['n_folds']}" for n in order)
              + f"; with attempted > imagined (same cue): {c['attempted_gt_imagined_same_cue']}/{c['n_folds']}."]
    if "shuffled_time_silent" in res:
        sts = ", ".join(f"{r} {v['n_silent']}/{v['n']}" for r, v in res["shuffled_time_silent"].items())
        lines += [f"Shuffled-time null: silent outputs {sts}."]
    lines += ["", f"### {model}: sensitivity analysis (not the primary test)", "",
              f"Only outputs with at least {_f(st['min_speech_s'], 2)} s of decoded speech. The selection depends on "
              "the output itself (imagined outputs lose more trials), so this only checks whether the effects survive "
              "among outputs with enough speech to be scored at all.", "",
              f"| scope | set | n kept / n | {nh} | top-1 (chance) | median rank / cues (p) |",
              "|---" * (5 + len(order)) + "|"]
    for name, sc, fold in scopes:
        for role, r in sc["sensitivity"]["roles"].items():
            ex = " | ".join(f"{_f(r['nulls'][n].get('mean'))} ({_ph(r['nulls'][n], 'wilcoxon_p', fold)})" for n in order)
            lines.append(f"| {name} | {role} | {r['n_kept']} / {r['n']} | {ex} | "
                         f"{_ident_cells(r['identification'], False)} |")
    lines += ["", f"| scope | {dh} |", "|---" * (1 + len(order)) + "|"]
    for name, sc, fold in scopes:
        c = sc["sensitivity"].get("attempted_vs_imagined")
        if not c:
            continue
        d = " | ".join(f"{_f(c['excess_difference_by_null'][n]['mean_difference'])} "
                       f"({_ph(c['excess_difference_by_null'][n], 'mannwhitney_p', fold)})" for n in order)
        lines.append(f"| {name} | {d} |")
    lines.append("")
    return lines


def markdown(summary: dict, title: str) -> str:
    cue = [res["settings"] for res in summary.values() if "settings" in res]
    if not cue:  # tables without the cue nulls: the original report, unchanged
        lines = [f"# {title}", "",
                 "mel_corr vs the cue's TTS (LPCNet resynthesis); null = the same output vs another trial's cue; "
                 "excess = real - null (paired, Wilcoxon p). Silent outputs (no speech frame) have no mel_corr and are "
                 "counted separately. WER: faster-whisper small.en, unconstrained, pooled (null = same transcript vs "
                 "another cue). Generated by scripts/summarize_kunz.py.", ""]
        for model, res in summary.items():
            lines += _legacy_model_section(model, res)
        return "\n".join(lines)
    st = cue[0]
    lines = [f"# {title}", "",
             "mel_corr vs the cue's TTS (LPCNet resynthesis). Every null scores the SAME output against other cues of "
             f"its set: **length-matched** = mean over the {st['length_matched_k']} other cues closest in TTS speech "
             "duration; **all cues** = mean over all other cues; **shuffled trial** = one seeded other trial's cue "
             "(kept for continuity; lenient, and partly a sentence-length effect). "
             f"Primary null: **{_SHORT[st['primary_null']]}** (experiment.summary.primary_null), listed first. "
             "Excess = own cue - null, paired per output (Wilcoxon p). Identification: rank of the own cue among all "
             "cues of the set (1 = best); top-1 rate (chance = mean 1/cues), median rank / cues, p = exact rank-sum "
             "test against uniform ranks. Attempted vs imagined: difference of the excesses (Mann-Whitney U, "
             "unpaired). Per-fold p values: raw [Holm-adjusted across the folds]. Silent outputs (no speech frame) "
             f"have no mel_corr and are counted separately. Early speech: decoded speech seconds within the first "
             f"{_f(st['early_window_s'], 0)} s of the output. WER: faster-whisper small.en, unconstrained, pooled (null "
             "= same transcript vs the shuffled-trial cue). Generated by scripts/summarize_kunz.py.", ""]
    for model, res in summary.items():
        lines += _cue_model_section(model, res) if "settings" in res else _legacy_model_section(model, res)
    return "\n".join(lines)
