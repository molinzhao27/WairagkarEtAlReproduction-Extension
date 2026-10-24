"""Phoneme labels and the brain2voice-2.0-style data pipeline (D-075): windows, batches, augmentation, live windows."""

import json

import numpy as np
import pytest

from wvx import b2v2_data as bd, phonemes as ph

CFG = {"data": {"pool": 5, "context_tokens": 8, "speech_feature": 0, "speech_threshold": 0.0},
       "augmentation": {"noise": 0.1, "offset": 0.05, "offset_tx_scale": 0.67, "drift": 0.005, "channel_mask": 0.1}}


# ---------------------------------------------------------------------------------------------- phonemes


def test_vocabulary_and_stress():
    assert ph.N_CLASSES == 41 and ph.ID[ph.BLANK] == 0 and ph.ID[ph.SIL] == 1
    assert ph.strip_stress("AH0") == "AH" and ph.strip_stress("ng") == "NG"
    with pytest.raises(ValueError):
        ph.strip_stress("spn")


def test_word_pronunciations_sources():
    words = [(0.0, 0.1, ""), (0.1, 0.5, "so"), (0.5, 0.9, "zorbly"), (0.9, 1.2, "blah")]
    phones = [(0.0, 0.1, "sil"), (0.1, 0.3, "S"), (0.3, 0.5, "OW1"), (0.5, 0.6, "Z"), (0.6, 0.75, "AO1"),
              (0.75, 0.9, "R"), (0.9, 1.2, "spn")]
    cmu = {"so": ["S", "OW1"]}
    out = ph.word_pronunciations(words, phones, cmu)
    assert [o["source"] for o in out] == ["cmudict", "aligner", "unknown"]
    assert out[0]["phones"] == ["S", "OW"] and out[1]["phones"] == ["Z", "AO", "R"] and out[2]["phones"] is None


def test_label_greedy_and_errors():
    assert ph.label([["S", "OW"], ["Y", "UW"]]) == [ph.ID["S"], ph.ID["OW"], 1, ph.ID["Y"], ph.ID["UW"], 1]
    assert ph.ctc_greedy([0, 2, 2, 0, 2, 3, 3, 1, 1, 0]) == [2, 2, 3, 1]
    assert ph.edit_distance("kitten", "sitting") == 3
    assert ph.phoneme_error([2, 1, 3, 1], [2, 3]) == (0, 2)


# ------------------------------------------------------------------------------------------------ windows


def _cut(words, s, e):
    return any(on < s < off or on < e < off for on, off in words)


def test_snap_window_never_cuts_a_word():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(50, 400))
        bounds = np.sort(rng.choice(np.arange(1, n), size=2 * int(rng.integers(1, 6)), replace=False))
        words = bounds.reshape(-1, 2).tolist()
        s0 = int(rng.integers(0, n - 40))
        s, e, inside = bd.snap_window(s0, s0 + 40, words, 5)
        if e > s:                                   # empty (s == e) when one word covers the whole window
            assert s0 <= s < e <= s0 + 40 and (e - s) % 5 == 0
            assert not _cut(words, s, e)
            assert inside == [k for k, (on, off) in enumerate(words) if on >= s and off <= e]


def test_sample_window_labels_match_words():
    t = {"length": 100, "words": [[10, 20], [30, 45], [60, 70]], "word_labels": [[5, 1], [6, 7, 1], [8, 1]],
         "label": [5, 1, 6, 7, 1, 8, 1], "speech_span": [10, 70]}
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = bd.sample_window(t, rng, 40, 5)
        assert w["ctc"] and (w["e"] - w["s"]) % 5 == 0 and w["e"] - w["s"] <= 40
        inside = [k for k, (on, off) in enumerate(t["words"]) if on >= w["s"] and off <= w["e"]]
        assert w["label"] == [i for k in inside for i in t["word_labels"][k]]


def test_sample_window_span_and_no_timing():
    rng = np.random.default_rng(2)
    t = {"length": 100, "words": None, "word_labels": None, "label": [5, 1], "speech_span": [40, 60]}
    for _ in range(50):
        w = bd.sample_window(t, rng, 40, 5)
        assert w["ctc"] and w["s"] <= 40 and w["e"] >= 60 and w["label"] == [5, 1]
    t2 = {"length": 100, "words": None, "word_labels": None, "label": None, "speech_span": None}
    w = bd.sample_window(t2, rng, 40, 5)
    assert not w["ctc"] and w["label"] == [] and w["e"] - w["s"] == 40


def _write_dataset(tmp_path, lengths, seed=0):
    rng = np.random.default_rng(seed)
    n = sum(lengths)
    neural = rng.normal(size=(n, 12)).astype(np.float16)
    target = rng.normal(size=(n, 3)).astype(np.float32)
    neural.tofile(tmp_path / "neural.f16")
    target.tofile(tmp_path / "target.f32")
    trials, a = [], 0
    for g, L in enumerate(lengths):
        trials.append({"gid": g, "start": a, "length": L, "words": None, "word_labels": None, "label": None,
                       "speech_span": None})
        a += L
    (tmp_path / "meta.json").write_text(json.dumps({"n_bins": n, "n_features": 12, "n_targets": 3, "trials": trials}))
    return bd.B2V2Data(tmp_path), neural, target


def test_make_batch_pools_and_takes_last_bin(tmp_path):
    data, neural, target = _write_dataset(tmp_path, [23, 61, 100])
    rng = np.random.default_rng(3)
    b = bd.make_batch(data, data.trials, rng, CFG)
    assert b["x"].shape[0] == 3 and b["x"].shape[1] <= 8 and b["y"].shape[2] == 3
    assert b["valid"].sum(1).tolist()[0] == 4                               # 23 bins -> 4 whole tokens
    # check the first trial exactly: its window ends at bin 23 - (23 - 20) ... reconstruct from valid tokens
    for i, t in enumerate(data.trials):
        n_tok = int(b["valid"][i].sum())
        x_trial = neural[t["start"]: t["start"] + t["length"]].astype(np.float32)
        y_trial = target[t["start"]: t["start"] + t["length"]]
        hits = [e for e in range(5 * n_tok, t["length"] + 1)
                if np.allclose(bd.pool_tokens(x_trial[e - 5 * n_tok: e], 5), b["x"][i, :n_tok], atol=1e-6)]
        assert hits, "window not found in its trial"
        e = hits[0]
        assert np.allclose(b["y"][i, :n_tok], y_trial[e - 5 * n_tok + 4: e: 5])
    assert not b["x"][~b["valid"]].any() and not b["ctc"].any()
    assert (b["speech"] == (b["valid"] & (b["y"][..., 0] > 0))).all()


def test_augment_keeps_padding_zero_and_masks_channels():
    rng = np.random.default_rng(4)
    x = np.ones((64, 8, 512), np.float32)
    valid = np.ones((64, 8), bool)
    valid[:, 6:] = False
    x[~valid] = 0
    y = bd.augment_batch(x, valid, rng, CFG["augmentation"])
    assert not y[~valid].any()
    zeroed = (y[:, :6] == 0).all(1).mean()
    assert 0.07 < zeroed < 0.13
    assert abs(float((y[:, :6] - 1)[(y[:, :6] != 0)].std()) - 0.1) < 0.03


def test_live_windows_end_at_each_step():
    x = np.random.default_rng(5).normal(size=(37, 4)).astype(np.float32)
    tok, n_tok, steps = bd.live_windows(x, 20, 5)
    assert steps[0] == 4 and steps[-1] == 36 and tok.shape == (33, 4, 4)
    assert n_tok[0] == 1 and n_tok.max() == 4
    for j, t in enumerate(steps):
        n = n_tok[j]
        assert np.allclose(tok[j, :n], bd.pool_tokens(x[t + 1 - 5 * n: t + 1], 5))
        assert not tok[j, n:].any()
