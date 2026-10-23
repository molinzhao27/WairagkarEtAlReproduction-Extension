"""wvx.word_splice on synthetic data (numpy only)."""

import numpy as np
import pytest

from wvx import word_splice as ws


def _inv():
    raw = np.zeros((30, 2), np.float32)
    raw[5:10] = 1.0                       # "hello": 5 frames of 1
    raw[15:25] = 2.0                      # "world": 10 frames of 2
    raw2 = np.zeros((20, 2), np.float32)
    raw2[2:4] = 3.0                       # "hello": 2 frames of 3
    return ws.build_inventory([raw, raw2, raw], [np.array([[5, 10], [15, 25]]), np.array([[2, 4]]), np.array([[5, 10]])],
                              ["Hello world.", "hello", "hello world"], ["a", "b", "c"])


def test_inventory_counts_words_and_skips_mismatched_trials():
    inv = _inv()
    assert inv["n_trials"] == 2 and inv["n_skipped"] == 1            # "hello world" with one span is skipped
    assert sorted(inv["words"]) == ["hello", "world"] and len(inv["words"]["hello"]) == 2
    assert np.allclose(inv["silence"], 0.0)


def test_splice_picks_the_closest_duration_and_resamples():
    inv = _inv()
    raw, used = ws.splice(["hello", "world"], np.array([[0, 3], [10, 20]]), 25, inv)
    assert raw.shape == (25, 2) and used[0] == "b:w0"                # the 2-frame exemplar is closer to 3 frames
    assert np.allclose(raw[0:3], 3.0) and np.allclose(raw[10:20], 2.0) and np.allclose(raw[3:10], 0.0)
    missing, _ = ws.splice(["hello", "moon"], np.array([[0, 3], [5, 8]]), 10, inv)
    assert missing is None
    with pytest.raises(ValueError):
        ws.splice(["hello"], np.array([[0, 3], [5, 8]]), 10, inv)


def test_resample_and_word_spans_from_syllables():
    seg = np.array([[0.0], [1.0]])
    assert np.allclose(ws.resample(seg, 3)[:, 0], [0.0, 0.5, 1.0])
    mapped = np.array([[2, 4], [4, 6], [8, 8], [9, 12]])
    spans = ws.word_spans_from_syllables(mapped, [0, 0, 1, 2], 3)
    assert spans.tolist() == [[2, 6], [0, 0], [9, 12]]                # word 1's syllable has no frames
