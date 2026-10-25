"""D-086: CTC forced alignment and the warp of attempted voice onto inner-speech timing."""

import itertools

import numpy as np

from wvx import distill, phonemes as ph


def _peaked(seq, C=6, sharp=8.0):
    """Log-probabilities strongly favouring ``seq`` (one class per frame)."""
    z = np.full((len(seq), C), -sharp)
    z[np.arange(len(seq)), seq] = 0.0
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def _path_score(logp, path, ext):
    return sum(logp[t, ext[s]] for t, s in enumerate(path))


def test_ctc_align_follows_a_clear_stream():
    lp = _peaked([0, 0, 2, 2, 0, 3, 0, 0, 4, 4, 4, 0])
    path = ph.ctc_align(lp, [2, 3, 4])
    ext = [0, 2, 0, 3, 0, 4, 0]
    assert [ext[s] for s in path] == [0, 0, 2, 2, 0, 3, 0, 0, 4, 4, 4, 0]
    assert list(distill.label_onsets(path, 3)) == [2, 5, 8]


def test_ctc_align_is_the_best_valid_path():
    rng = np.random.default_rng(0)
    lp = np.log(rng.dirichlet(np.ones(4), 6))
    label = [1, 1, 2]
    ext = [0, 1, 0, 1, 0, 2, 0]
    S = len(ext)
    best = -np.inf
    for p in itertools.product(range(S), repeat=6):            # brute force over every monotone CTC path
        if p[0] > 1 or p[-1] < S - 2:
            continue
        ok = all(b == a or b == a + 1 or (b == a + 2 and ext[b] != 0 and ext[b] != ext[a]) for a, b in zip(p, p[1:]))
        if ok:
            best = max(best, _path_score(lp, p, ext))
    path = ph.ctc_align(lp, label)
    assert np.isclose(_path_score(lp, path, ext), best)


def test_warp_identity_and_stretch():
    seq = [0, 1, 0, 0, 2, 0, 3, 0, 0, 0]
    lp = _peaked(seq)
    f = distill.warp_map(lp, lp, [1, 2, 3], 50, 50, 5)
    np.testing.assert_allclose(f, np.arange(50), atol=1e-9)
    slow = _peaked([0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0])   # the same, twice as slow
    f = distill.warp_map(slow, lp, [1, 2, 3], 100, 50, 5)
    assert np.all(np.diff(f) >= 0) and f[0] == 0 and f[-1] == 49
    # label 2: token 8 in the slow trial (frame 42) -> token 4 in the fast one (frame 22)
    assert abs(f[42] - 22) < 1e-9
    att = np.stack([np.arange(50, dtype=np.float32)] * 3, 1)
    w = distill.warp_features(att, f)
    np.testing.assert_allclose(w[:, 0], f, atol=1e-5)
