"""wvx.kunz_summary on a synthetic two-fold table (no TensorFlow, no data)."""

import numpy as np
import pandas as pd

from wvx import kunz_summary as ks


def _table(seed=42, n=20):
    rng = np.random.default_rng(seed)
    rows = []
    for fold in ("fold_A03", "fold_A06"):
        for role, gain, speech in (("heldout_attempted", 0.10, 0.4), ("paired_imagined", 0.0, 0.1)):
            for i in range(n):
                null = rng.normal(0.5, 0.05)
                rows.append(dict(model="zero_shot", input="real", fold=fold, role=role, trial_index=i,
                                 text=f"cue {i}", n_frames=1000, tts_est_speech_fraction=speech,
                                 tts_mel_corr=null + gain + rng.normal(0, 0.01),
                                 shuffled_trial_tts_mel_corr=null,
                                 asr_word_errors=4, asr_ref_words=5, shuffled_trial_asr_word_errors=6,
                                 shuffled_trial_asr_ref_words=5))
    return pd.DataFrame(rows)


def test_summarize_run_detects_attempted_excess():
    out = ks.summarize_run(_table(), frame_s=0.01)["zero_shot"]
    pooled = out["pooled"]
    att, img = pooled["roles"]["heldout_attempted"], pooled["roles"]["paired_imagined"]
    assert abs(att["excess"]["mean"] - 0.10) < 0.01 and abs(img["excess"]["mean"]) < 0.01
    assert abs(att["speech_s"]["mean"] - 4.0) < 1e-9 and abs(img["speech_s"]["mean"] - 1.0) < 1e-9
    c = pooled["attempted_vs_imagined"]
    assert c["mel_corr_same_cue"]["n"] == 40          # matched on (fold, cue)
    assert c["excess_difference"]["mannwhitney_p"] < 1e-6
    assert abs(c["speech_s_same_cue"]["mean"] - 3.0) < 1e-9
    assert att["asr"]["wer"] == 0.8 and att["asr"]["wer_null_shuffled_trial"] == 1.2
    assert out["consistency"] == {"n_folds": 2, "attempted_excess_gt_imagined_excess": 2,
                                  "attempted_gt_imagined_same_cue": 2}
    assert "| pooled | heldout_attempted |" in ks.markdown({"zero_shot": out}, "t")


def test_silent_outputs_counted_not_scored():
    df = _table()
    m = df.role == "paired_imagined"
    df.loc[m & (df.trial_index < 3), ["tts_est_speech_fraction", "tts_mel_corr", "shuffled_trial_tts_mel_corr"]] = \
        [0.0, np.nan, np.nan]
    img = ks.summarize_run(df, frame_s=0.01)["zero_shot"]["pooled"]["roles"]["paired_imagined"]
    assert img["n_silent"] == 6 and img["excess"]["n"] == 34


def test_set_decoded_in_every_fold_counts_once_when_pooled():
    """The eval block is the same trials in every fold: pooled, each trial counts once (mean over the folds'
    outputs), so four identical copies give exactly the one-fold statistics, not a p value from 4x the trials."""
    rng = np.random.default_rng(0)
    rows = []
    vals = [(rng.normal(0.5, 0.05), rng.normal(0.01, 0.05), int(rng.integers(1, 11))) for _ in range(30)]
    for fold in ("fold_A03", "fold_A06", "fold_A11", "fold_A13"):
        for i, (null, gain, rank) in enumerate(vals):
            rows.append(dict(model="zero_shot", input="real", fold=fold, role="eval_imagined", set="eval_imagined_E14",
                             trial_index=i, text=f"cue {i}", n_frames=1000, tts_est_speech_fraction=0.2,
                             tts_mel_corr=null + gain, shuffled_trial_tts_mel_corr=null, own_rank=rank, n_cues=10,
                             asr_word_errors=4, asr_ref_words=5, shuffled_trial_asr_word_errors=6,
                             shuffled_trial_asr_ref_words=5))
    df = pd.DataFrame(rows)
    one = ks.summarize_run(df[df.fold == "fold_A03"], frame_s=0.01)["zero_shot"]["pooled"]["roles"]["eval_imagined"]
    pooled = ks.summarize_run(df, frame_s=0.01)["zero_shot"]["pooled"]["roles"]["eval_imagined"]
    assert pooled["n"] == 30 and pooled["collapsed_over_folds"] == 4
    assert pooled["excess"]["wilcoxon_p"] == one["excess"]["wilcoxon_p"]
    assert abs(pooled["excess"]["mean"] - one["excess"]["mean"]) < 1e-12
    # the per-fold rows keep every fold's own outputs
    folds = ks.summarize_run(df, frame_s=0.01)["zero_shot"]["folds"]
    assert all(f["roles"]["eval_imagined"]["n"] == 30 for f in folds.values())
    # identification on the per-trial mean rank: top-1 share and a Wilcoxon test against chance (5.5 of 10)
    df2 = df.assign(null_lenmatched_mean=df.shuffled_trial_tts_mel_corr, null_all_mean=df.shuffled_trial_tts_mel_corr,
                    est_speech_s=2.0, early_speech_s=1.0)
    ident = ks._identification_mean(ks.collapse_repeated(df2)[0])
    assert ident["n"] == 30 and ident["rank_test"] == "wilcoxon_mean_rank_vs_chance"
    assert abs(ident["top1_rate"] - np.mean([r == 1 for _, _, r in vals])) < 1e-12
