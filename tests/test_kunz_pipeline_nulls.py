"""Cue nulls, identification, resume fingerprints and the cue-null summaries (pure logic; CPU, no data).

wvx.kunz_experiment: set_cues, cue_null_stats, rank_sum_uniform_p, identification, holm, output_speech_stats,
training_fingerprint, prediction_staleness, index_unique; wvx.kunz_summary on tables with the cue-null columns.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from wvx import kunz_experiment as kx
from wvx import kunz_summary as ks
from wvx import metrics
from wvx.session import Trial

# ------------------------------------------------------------------------------- cue nulls and ranks


def _brute(scores, own, dur, k):
    others = [j for j in range(len(scores)) if j != own]
    fin = [j for j in others if np.isfinite(scores[j])]
    near = sorted(others, key=lambda j: (abs(dur[j] - dur[own]), j))[:k]
    near = [scores[j] for j in near if np.isfinite(scores[j])]
    rank = 1 + sum(scores[j] >= scores[own] for j in fin) if np.isfinite(scores[own]) else float("nan")
    return {"null_all_mean": np.mean([scores[j] for j in fin]) if fin else float("nan"),
            "null_lenmatched_mean": np.mean(near) if near else float("nan"), "own_rank": rank}


@pytest.mark.parametrize("seed", range(5))
def test_cue_null_stats_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 30))
    scores = rng.uniform(0.3, 0.8, n)
    scores[rng.integers(n)] = np.nan                       # one undefined cue
    dur = np.round(rng.uniform(1, 4, n), 1)               # rounded: ties in duration distance happen
    for own in range(n):
        if not np.isfinite(scores[own]):
            continue
        for k in (1, 3, n + 5):
            got, want = kx.cue_null_stats(scores, own, dur, k), _brute(scores, own, dur, k)
            for key in want:
                assert got[key] == pytest.approx(want[key], nan_ok=True), (own, k, key)
            assert got["n_cues"] == 1 + int(np.isfinite(np.delete(scores, own)).sum())
            assert got["cue_tts_speech_s"] == dur[own]
    k_all = kx.cue_null_stats(scores, 0 if np.isfinite(scores[0]) else 1, dur, n)   # k >= others: the same null
    assert k_all["null_lenmatched_mean"] == pytest.approx(k_all["null_all_mean"])


def test_rank_ties_count_against_own_and_undefined_output():
    s = [0.6, 0.6, 0.5, 0.7]
    assert kx.cue_null_stats(s, 0, [1, 2, 3, 4], 2)["own_rank"] == 3    # 0.7 and the tied 0.6 both rank above
    silent = kx.cue_null_stats([np.nan] * 4, 1, [1, 2, 3, 4], 2)
    assert math.isnan(silent["own_rank"]) and silent["n_cues"] == 0 and math.isnan(silent["null_all_mean"])
    with pytest.raises(ValueError):
        kx.cue_null_stats(s, 0, [1, 2, 3, 4], 0)


def test_lenmatched_picks_the_closest_durations():
    """Own cue 2 s: the length-matched null uses the cues of 1.9 and 2.2 s, not the short or long ones."""
    scores = [0.5, 0.9, 0.1, 0.2, 0.95]
    dur = [2.0, 0.5, 1.9, 2.2, 6.0]
    out = kx.cue_null_stats(scores, 0, dur, 2)
    assert out["null_lenmatched_mean"] == pytest.approx(0.15) and out["null_lenmatched_n"] == 2
    assert out["null_all_mean"] == pytest.approx((0.9 + 0.1 + 0.2 + 0.95) / 4)
    assert out["own_rank"] == 3


def test_set_cues_maps_trials_to_distinct_cue_columns():
    keys, reps, own = kx.set_cues(["Go home.", "Stay.", "go home", "Wait?"], normalize=metrics.normalize_text)
    assert keys == ["go home", "stay", "wait"] and reps == ["Go home.", "Stay.", "Wait?"] and own == [0, 1, 0, 2]


def _exact_two_sided(ranks, ns):
    sums = [sum(c) for c in itertools.product(*[range(1, n + 1) for n in ns])]
    s, tot = sum(ranks), len(sums)
    lo, hi = sum(x <= s for x in sums) / tot, sum(x >= s for x in sums) / tot
    return min(1.0, 2 * min(lo, hi))


@pytest.mark.parametrize("ranks,ns", [([1, 1, 1], [3, 4, 2]), ([2, 3, 1], [3, 4, 2]), ([3, 4, 2], [3, 4, 2]),
                                       ([1, 2, 2, 5], [5, 5, 3, 6])])
def test_rank_sum_p_is_exact(ranks, ns):
    assert kx.rank_sum_uniform_p(ranks, ns) == pytest.approx(_exact_two_sided(ranks, ns))


def test_rank_sum_p_is_calibrated_and_detects_identification():
    rng = np.random.default_rng(0)
    ps = [kx.rank_sum_uniform_p(rng.integers(1, 51, 50), [50] * 50) for _ in range(400)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.09          # ~5 % false positives under uniform ranks
    assert kx.rank_sum_uniform_p(rng.integers(1, 20, 50), [50] * 50) < 1e-6
    with pytest.raises(ValueError):
        kx.rank_sum_uniform_p([1.5], [3])


def test_identification_summary():
    out = kx.identification([1, 1, 2, np.nan], [4, 4, 4, 0])
    assert out["n"] == 3 and out["n_undefined"] == 1
    assert out["top1_rate"] == pytest.approx(2 / 3) and out["top1_chance"] == pytest.approx(0.25)
    assert out["median_rank"] == 1 and out["mean_normalized_rank"] == pytest.approx((1 / 3) / 3)
    assert 0 < out["uniform_rank_p"] <= 1
    assert kx.identification([np.nan], [0]) == {"n": 0, "n_undefined": 1}


def test_holm():
    out = kx.holm([0.01, 0.04, 0.03, np.nan])
    assert out[:3] == pytest.approx([0.03, 0.06, 0.06]) and math.isnan(out[3])
    assert kx.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_null_order_and_unpaired():
    assert kx.null_order("lenmatched") == ["lenmatched", "all", "shuffled_trial"]
    assert kx.null_order("all")[0] == "all"
    with pytest.raises(ValueError):
        kx.null_order("random")
    u = kx.unpaired([1.0, 2.0, 3.0, np.nan], [0.0, 0.5])
    assert (u["n_a"], u["n_b"]) == (3, 2) and u["mean_difference"] == pytest.approx(1.75)


def test_output_speech_stats_with_the_metrics_silence_detector():
    sr = 16000
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    wav = np.concatenate([np.zeros(sr), tone, tone, np.zeros(3 * sr), tone])    # speech at 1-3 s and 6-7 s
    mask = metrics.speech_frame_mask(wav, sr, {"evaluate": {"mel_corr_framewise": False}})
    out = kx.output_speech_stats(wav, mask, 0.01, 4.0)
    # the 50 ms analysis window widens each speech region by about 2 frames per edge
    assert out["est_speech_s"] == pytest.approx(3.0, abs=0.12)
    assert out["early_speech_s"] == pytest.approx(2.0, abs=0.08)
    assert out["est_speech_frames"] == int(mask.sum())
    assert out["peak_dbfs"] == pytest.approx(20 * math.log10(0.5), abs=0.01)
    silent = kx.output_speech_stats(np.zeros(sr), metrics.speech_frame_mask(np.zeros(sr), sr, None), 0.01, 4.0)
    assert silent["est_speech_frames"] == 0 and math.isnan(silent["peak_dbfs"])


def test_index_unique():
    assert kx.index_unique(["a", "b"], "x") == {"a": 0, "b": 1}
    with pytest.raises(ValueError, match="repeated key"):
        kx.index_unique(["a", "b", "a"], "x")


# ------------------------------------------------------------------------ resume fingerprints


def _cfg():
    return {"seed": 42, "runtime": {"deterministic_ops": True, "threads": {"omp": 12}},
            "model": {"window_len": 60},
            "training": {"epochs": 5, "verbose": 2, "log_every_batches": 50, "overwrite": False,
                         "init": {"source": "inference_h5", "path": "a/init.h5", "freeze_base": True}},
            "experiment": {"session": "s", "reorder_channels": True,
                           "variants": {"frozen": {"freeze_base": True}, "finetuned": {"freeze_base": False}}}}


def _trials():
    return [Trial(neural=np.zeros((2, 2)), target=None, text="a", block=6, meta={"trial_index": i}) for i in range(3)]


def test_training_fingerprint_tracks_what_changes_the_model():
    files = {"attempted_file": "aaa", "init": "iii"}
    base = kx.training_fingerprint(_cfg(), "frozen", _trials(), files)

    def fp(mut=None, variant="frozen", trials=None, f=None):
        c = _cfg()
        if mut:
            mut(c)
        return kx.training_fingerprint(c, variant, _trials() if trials is None else trials, f or files)

    same = [lambda c: c["training"].update(verbose=0, log_every_batches=1, overwrite=True),
            lambda c: c["training"]["init"].update(path="b/elsewhere.h5"),       # content counts, not the path
            lambda c: c["runtime"]["threads"].update(omp=1)]
    for mut in same:
        assert fp(mut) == base
    differ = [fp(lambda c: c["training"].update(epochs=6)), fp(lambda c: c.update(seed=1)),
              fp(lambda c: c["model"].update(window_len=50)), fp(lambda c: c["experiment"].update(reorder_channels=False)),
              fp(variant="finetuned"), fp(trials=_trials()[:2]), fp(f={**files, "attempted_file": "bbb"}),
              fp(lambda c: c["training"]["init"].update(freeze_base=False))]
    # (the base init.freeze_base is overridden by the variant, so only the variant's value matters)
    assert all(d != base for d in differ[:-1]) and differ[-1] == base
    assert kx.fingerprint_differences(base, differ[0]) == ["training.epochs"]


def test_prediction_staleness():
    keys = ["s__tr000", "s__tr001"]
    legacy = {"model_sha256": "m1", "input": "real", "reorder_channels": True, "seed": 42}
    real = {"input": "real", "reorder_channels": True, "window_len": 60}
    assert kx.prediction_staleness(legacy, keys, "m1", real, keys) == []        # unrecorded settings accepted
    assert "model" in kx.prediction_staleness(legacy, keys, "m2", real, keys)[0]
    assert kx.prediction_staleness(legacy, keys, "m1", {**real, "reorder_channels": False}, keys)
    assert kx.prediction_staleness(legacy, keys[:1], "m1", real, keys)[0].startswith("1 needed trial")
    assert kx.prediction_staleness({**legacy, "seed": 1}, keys, "m1", real, keys) == []   # seed: shuffled time only
    st = {"input": "shuffled_time", "reorder_channels": True, "seed": 42}
    assert kx.prediction_staleness({**legacy, "input": "shuffled_time", "seed": 1}, keys, "m1", st, keys)
    new = {"model_sha256": "m1", "decode_settings": {**real, "window_len": 50}}
    assert kx.prediction_staleness(new, keys, "m1", real, keys) == ["decode setting window_len was 50, now 60"]


# ------------------------------------------------------------------- cue-null summaries (cross-fold)


def _table(seed=0, n=20, cue_nulls=True):
    """Two folds; attempted identifies its cue (rank 1-3, excess ~0.1), imagined is at chance."""
    rng = np.random.default_rng(seed)
    rows = []
    for fold in ("fold_A03", "fold_A06"):
        for role, gain, speech in (("heldout_attempted", 0.10, 0.4), ("paired_imagined", 0.0, 0.1)):
            for i in range(n):
                null = rng.normal(0.5, 0.05)
                r = dict(model="zero_shot", input="real", fold=fold, role=role, trial_index=i, text=f"cue {i}",
                         n_frames=1000, tts_est_speech_fraction=speech, tts_mel_corr=null + gain + rng.normal(0, 0.01),
                         shuffled_trial_tts_mel_corr=null + rng.normal(0, 0.01))
                if cue_nulls:
                    r.update(null_all_mean=null + rng.normal(0, 0.01), null_lenmatched_mean=null + rng.normal(0, 0.01),
                             own_rank=float(rng.integers(1, 4) if gain else rng.integers(1, n + 1)), n_cues=n,
                             est_speech_s=speech * 10 * (i % 2), early_speech_s=speech * 4, peak_dbfs=-20.0)
                rows.append(r)
    return pd.DataFrame(rows)


SETTINGS = {"primary_null": "lenmatched", "min_speech_s": 0.4, "length_matched_k": 10, "early_window_s": 4.0}


def test_cue_null_summary_sections():
    out = ks.summarize_run(_table(), 0.01, SETTINGS)["zero_shot"]
    assert out["settings"]["null_order"] == ["lenmatched", "all", "shuffled_trial"]
    pooled = out["pooled"]
    att, img = pooled["roles"]["heldout_attempted"], pooled["roles"]["paired_imagined"]
    assert list(att["nulls"]) == ["lenmatched", "all", "shuffled_trial"]
    for n in att["nulls"]:
        assert att["nulls"][n]["excess"]["mean"] == pytest.approx(0.10, abs=0.01)
        assert abs(img["nulls"][n]["excess"]["mean"]) < 0.01
    assert att["identification"]["top1_rate"] > 0.1 and att["identification"]["uniform_rank_p"] < 1e-6
    assert att["early_speech_s"]["mean"] == pytest.approx(1.6)
    avi = pooled["attempted_vs_imagined"]
    assert list(avi["excess_difference_by_null"]) == ["lenmatched", "all", "shuffled_trial"]
    assert avi["excess_difference_by_null"]["lenmatched"]["mannwhitney_p"] < 1e-6
    assert avi["excess_difference"] == avi["excess_difference_by_null"]["shuffled_trial"]   # the old column
    assert avi["early_speech_s_same_cue"]["mean"] == pytest.approx(1.2)
    assert out["consistency_by_null"] == {"lenmatched": 2, "all": 2, "shuffled_trial": 2}
    # sensitivity: only outputs with >= 0.4 s speech (odd trials: 4 s attempted, 1 s imagined; even: 0 s)
    sens = pooled["sensitivity"]
    assert sens["roles"]["heldout_attempted"]["n_kept"] == 20 and sens["roles"]["heldout_attempted"]["n"] == 40
    assert "not the primary test" in sens["analysis"]
    # Holm across folds on the per-fold tests, never below the raw p
    for f in out["folds"].values():
        e = f["roles"]["paired_imagined"]["nulls"]["lenmatched"]["excess"]
        assert e["wilcoxon_p_holm"] >= e["wilcoxon_p"]
        d = f["attempted_vs_imagined"]["excess_difference_by_null"]["all"]
        assert d["mannwhitney_p_holm"] >= d["mannwhitney_p"]
    assert "wilcoxon_p_holm" not in pooled["roles"]["paired_imagined"]["nulls"]["lenmatched"]["excess"]

    md = ks.markdown({"zero_shot": out}, "t")
    assert "excess length-matched (p) | excess all cues (p) | excess shuffled trial (p)" in md
    assert "Primary null: **length-matched**" in md and "sensitivity analysis (not the primary test)" in md
    assert "| pooled | heldout_attempted |" in md and "]) |" in md  # Holm-adjusted p in brackets per fold
    for table in md.split("\n\n"):                                  # every table row has its header's width
        rows = [ln for ln in table.splitlines() if ln.startswith("|")]
        assert len({ln.count("|") for ln in rows}) <= 1, rows[:2]
    other = ks.summarize_run(_table(), 0.01, {**SETTINGS, "primary_null": "all"})["zero_shot"]
    assert list(other["pooled"]["roles"]["heldout_attempted"]["nulls"])[0] == "all"
    assert "excess all cues (p) | excess length-matched (p)" in ks.markdown({"zero_shot": other}, "t")


def test_cue_null_summary_without_settings_uses_the_base_config():
    out = ks.summarize_run(_table(), 0.01)["zero_shot"]
    assert out["settings"]["primary_null"] == ks.default_settings()["primary_null"]


def test_tables_without_cue_nulls_give_the_older_summary():
    out = ks.summarize_run(_table(cue_nulls=False), 0.01, SETTINGS)["zero_shot"]
    assert "settings" not in out and "consistency_by_null" not in out
    assert "nulls" not in out["pooled"]["roles"]["heldout_attempted"] and "sensitivity" not in out["pooled"]
    md = ks.markdown({"zero_shot": out}, "t")
    assert "| scope | set | n (silent) | speech s / output s | mel_corr | null | excess (p) | WER (null) |" in md
    assert "length-matched" not in md
