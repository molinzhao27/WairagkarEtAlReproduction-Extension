"""Kunz adapter on a synthetic file with the real files' keys, dtypes and 1-based inclusive epochs."""

import json

import numpy as np
import pytest
import scipy.io

from wvx.kunz import load_condition

C = 8          # channels per feature type (256 in the real files)
DELAY = 20     # delay-period bins (~400 in the real files)
GO = 30        # go-period bins


def _adapter(**kw):
    a = {
        "bin_ms": 10,
        "n_features": 2 * C,
        "pre_go_bins": 0,
        "clip_spikepow": None,
        "channel_permutation": None,
        "normalization": {"method": "none", "eps": 1e-6},
    }
    a.update(kw)
    return a


@pytest.fixture
def kunz_file(tmp_path):
    """3 blocks x 2 trials, laid out like the real files: delay then go, gaps between trials."""
    rng = np.random.default_rng(0)
    gap, blocks, go, delay, t = 5, [], [], [], 0
    for b in (3, 6, 11):
        start = t
        for _ in range(2):
            t += gap
            delay.append((t + 1, t + DELAY))           # 1-based inclusive
            go.append((t + DELAY + 1, t + DELAY + GO))
            t += DELAY + GO
        blocks += [b] * (t - start)
    T = t
    sp = rng.uniform(20, 5000, (T, C))
    tx = rng.integers(0, 5, (T, C)).astype(np.uint8)
    go, delay = np.array(go, np.int32), np.array(delay, np.int32)
    feats = np.empty((len(go), 1), object)
    for i, (g0, g1) in enumerate(go):
        feats[i, 0] = np.hstack([tx, sp])[g0 - 1:g1]  # TX first, as in the real sentenceFeatures
    text = np.empty((len(go), 1), object)
    decoded = np.empty((len(go), 1), object)
    for i in range(len(go)):
        text[i, 0] = f"sentence {i}."
        decoded[i, 0] = "" if i < 2 else f"sentence {i}"
    path = tmp_path / "t15.test.mat"
    scipy.io.savemat(path, {
        "spikePow": sp, "binnedTX": tx, "blockNum": np.array(blocks, np.uint8)[:, None],
        "goTrialEpochs": go, "delayTrialEpochs": delay, "sentenceText": text,
        "decodedSentenceText": decoded, "sentenceFeatures": feats, "binSize": np.int64(10),
    })
    return path, sp, tx, np.array(blocks), go, feats


def test_trials_match_sentence_features_with_spikepow_first(kunz_file):
    path, sp, tx, _, go, feats = kunz_file
    s = load_condition(path, "t", 0, _adapter())
    s.validate(n_features=2 * C, n_targets=20)
    assert len(s.trials) == len(go)
    for tr, f in zip(s.trials, feats[:, 0]):
        assert tr.neural.dtype == np.float32
        np.testing.assert_allclose(tr.neural, np.hstack([f[:, C:], f[:, :C]]).astype(np.float32))
        assert tr.neural.shape[0] == GO and tr.meta["go_offset"] == 0


def test_pre_go_context_and_block_boundary(kunz_file):
    path, *_ = kunz_file
    within = load_condition(path, "t", 0, _adapter(pre_go_bins=7))
    assert all(t.meta["go_offset"] == 7 for t in within.trials)
    huge = load_condition(path, "t", 0, _adapter(pre_go_bins=10_000))
    # The first trial of each block may only reach back to its own block's first bin.
    assert huge.trials[2].meta["go_offset"] == 5 + DELAY   # gap + delay of block 6's first trial
    assert huge.trials[0].meta["go_offset"] == 5 + DELAY


def test_metadata(kunz_file):
    path, *_ = kunz_file
    s = load_condition(path, "t", 3, _adapter())
    assert s.layer == 3
    assert [t.block for t in s.trials] == [3, 3, 6, 6, 11, 11]
    assert [t.meta["closed_loop"] for t in s.trials] == [False, False, True, True, True, True]
    assert s.trials[0].text == "sentence 0."


def test_zscore_block_and_permutation(kunz_file, tmp_path):
    path, sp, tx, blocks, go, _ = kunz_file
    perm = list(range(C))[::-1]
    pfile = tmp_path / "perm.json"
    pfile.write_text(json.dumps(perm))
    s = load_condition(path, "t", 0, _adapter(
        channel_permutation=str(pfile), normalization={"method": "zscore_block", "eps": 1e-6}))
    b = blocks == 3
    expect_sp = (sp[b][:, perm] - sp[b][:, perm].mean(0)) / (sp[b][:, perm].std(0) + 1e-6)
    expect_tx = (tx[b][:, perm] - tx[b][:, perm].mean(0)) / (tx[b][:, perm].std(0) + 1e-6)
    g0, g1 = go[0, 0] - 1, go[0, 1]
    start = int(np.flatnonzero(b)[0])
    np.testing.assert_allclose(s.trials[0].neural[:, :C], expect_sp[g0 - start:g1 - start], rtol=1e-5)
    np.testing.assert_allclose(s.trials[0].neural[:, C:], expect_tx[g0 - start:g1 - start], rtol=1e-5, atol=1e-6)


def test_bin_size_mismatch_raises(kunz_file):
    path, *_ = kunz_file
    with pytest.raises(ValueError):
        load_condition(path, "t", 0, _adapter(bin_ms=20))
