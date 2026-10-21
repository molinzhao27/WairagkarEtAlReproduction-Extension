"""wvx.preprocess: the causal log -> rolling z-score -> sigmoid smoothing of Wairagkar et al. 2025."""

import copy

import numpy as np
import pytest

from wvx.config import load_yaml, resolve_path
from wvx.preprocess import (
    block_runs,
    causal_smooth,
    rolling_zscore,
    sigmoid_kernel,
    wairagkar_block,
    wairagkar_transform,
)

SEED = 42
CFG = load_yaml(resolve_path("configs/data/kunz.yaml"))["adapter"]
BIN_MS = CFG["bin_ms"]


@pytest.fixture
def params():
    return copy.deepcopy(CFG["normalization"]["params"])


def _bins(ms):
    return int(round(ms / BIN_MS))


def _brute_zscore(x, window, include_current, ddof, min_std, zero_std_value):
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        end = t + 1 if include_current else t
        w = x[max(end - window, 0):end]
        if len(w) == 0:
            continue
        sd = w.std(0, ddof=ddof) if len(w) > ddof else np.zeros(x.shape[1])
        sd = np.where(sd <= min_std, zero_std_value, sd)
        out[t] = (x[t] - w.mean(0)) / sd
    return out


def test_config_defaults_are_the_documented_ones(params):
    assert CFG["normalization"]["method"] == "wairagkar"
    assert CFG["clip_spikepow"] == 50000
    assert _bins(params["rolling"]["window_ms"]) == 1000        # "past 10 s"
    assert _bins(params["smoothing"]["length_ms"]) == 15


def test_sigmoid_kernel_properties(params):
    sm = params["smoothing"]
    L, c, s = _bins(sm["length_ms"]), sm["center_ms"] / BIN_MS, sm["scale_ms"] / BIN_MS
    w = sigmoid_kernel(L, c, s)
    assert w.shape == (L,) and w.dtype == np.float64
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w > 0) and np.all(np.diff(w) < 0)           # most weight on the current bin, decaying
    k = np.arange(L)
    np.testing.assert_allclose(w / w[0], (1 + np.exp(-c / s)) / (1 + np.exp((k - c) / s)))
    # Half of the un-normalized peak weight at the center lag.
    ref = 1 / (1 + np.exp((k - c) / s))
    assert np.isclose(np.interp(c, k, ref), 0.5, atol=0.02)
    with pytest.raises(ValueError):
        sigmoid_kernel(0, 1.0, 1.0)
    with pytest.raises(ValueError):
        sigmoid_kernel(5, 1.0, 0.0)


@pytest.mark.parametrize("include_current", [True, False])
@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_zscore_matches_brute_force(include_current, ddof):
    rng = np.random.default_rng(SEED)
    x = rng.normal(5.0, 2.0, (130, 3))
    x[40:60, 1] = 7.0                                          # a constant stretch longer than the window
    got = rolling_zscore(x, 17, include_current, ddof, 1e-8, 1.0)
    want = _brute_zscore(x, 17, include_current, ddof, 1e-8, 1.0)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_rolling_zscore_constant_input_is_zero():
    z = rolling_zscore(np.full((50, 2), 3.7), 10, True, 0, 1e-8, 1.0)
    np.testing.assert_array_equal(z, 0.0)


def test_causal_smooth_matches_brute_force_with_renormalized_start():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(40, 2))
    w = sigmoid_kernel(6, 2.0, 1.0)
    got = causal_smooth(x, w)
    want = np.zeros_like(x)
    for t in range(40):
        ks = np.arange(min(6, t + 1))
        want[t] = (w[ks, None] * x[t - ks]).sum(0) / w[ks].sum()
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_shapes_dtypes_and_feature_check(params):
    rng = np.random.default_rng(SEED)
    sp = rng.uniform(20, 5000, (300, 4)).astype(np.float32)
    tx = rng.integers(0, 4, (300, 4)).astype(np.uint8)
    blocks = np.repeat([3, 6], 150)
    for x, feat in ((sp, "spikepow"), (tx, "threshcross")):
        y = wairagkar_transform(x, blocks, feat, params, BIN_MS)
        assert y.shape == x.shape and y.dtype == np.float64 and np.all(np.isfinite(y))
    with pytest.raises(ValueError):
        wairagkar_transform(sp, blocks, "lfp", params, BIN_MS)
    with pytest.raises(ValueError):
        wairagkar_transform(sp, blocks[:-1], "spikepow", params, BIN_MS)
    with pytest.raises(ValueError):                            # log of a non-positive value
        wairagkar_block(-sp, "spikepow", params, BIN_MS)
    bad = copy.deepcopy(params)
    bad["smoothing"]["length_ms"] = 155                        # not a whole number of 10 ms bins
    with pytest.raises(ValueError):
        wairagkar_block(sp, "spikepow", bad, BIN_MS)


def test_causality(params):
    """Changing bins from t0 on must leave every output before t0 untouched (also inside the warm-up)."""
    rng = np.random.default_rng(SEED)
    T = 2500
    x = rng.uniform(20, 5000, (T, 3))
    blocks = np.full(T, 4)
    base = wairagkar_transform(x, blocks, "spikepow", params, BIN_MS)
    for t0 in (5, 700, 1800):
        y = x.copy()
        y[t0:] = rng.uniform(20, 5000, (T - t0, 3))
        out = wairagkar_transform(y, blocks, "spikepow", params, BIN_MS)
        np.testing.assert_array_equal(out[:t0], base[:t0])
        assert not np.allclose(out[t0], base[t0])


def test_block_reset(params):
    """A block's output depends only on that block, so the previous block's data never leak in."""
    rng = np.random.default_rng(SEED)
    a = rng.poisson(0.3, (900, 3)).astype(float)
    b = rng.poisson(0.2, (1200, 3)).astype(float)
    blocks = np.r_[np.full(900, 3), np.full(1200, 6)]
    both = wairagkar_transform(np.r_[a, b], blocks, "threshcross", params, BIN_MS)
    alone = wairagkar_transform(b, np.full(1200, 6), "threshcross", params, BIN_MS)
    np.testing.assert_array_equal(both[900:], alone)
    other = wairagkar_transform(np.r_[a * 0 + 5, b], blocks, "threshcross", params, BIN_MS)
    np.testing.assert_array_equal(other[900:], both[900:])
    assert [(r.start, r.stop) for r in block_runs(blocks)] == [(0, 900), (900, 2100)]


def test_stationary_lognormal_input_is_standardized(params):
    """log(x) ~ N(m, s): after the 10 s warm-up the rolling z-score has mean ~0 and s.d. ~1, and the
    smoothed output keeps mean ~0 with s.d. sqrt(sum w^2) (white input)."""
    rng = np.random.default_rng(SEED)
    T, C = 6000, 8
    x = np.exp(rng.normal(6.5, 0.7, (T, C)))                    # spike-band-power-like, uV^2
    p = copy.deepcopy(params)
    roll, sm = p["rolling"], p["smoothing"]
    window = _bins(roll["window_ms"])
    z = rolling_zscore(np.log(x + p["log_offset"]["spikepow"]), window, roll["include_current"],
                       roll["ddof"], roll["min_std"], roll["zero_std_value"])[window:]
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03
    y = wairagkar_block(x, "spikepow", p, BIN_MS)[window:]
    w = sigmoid_kernel(_bins(sm["length_ms"]), sm["center_ms"] / BIN_MS, sm["scale_ms"] / BIN_MS)
    assert abs(y.mean()) < 0.03
    assert abs(y.std() / np.sqrt((w ** 2).sum()) - 1) < 0.05
    # Smoothing signature: lag-1 autocorrelation of the output equals the kernel's own (white input).
    yc = y - y.mean(0)
    ac1 = np.median((yc[1:] * yc[:-1]).sum(0) / (yc * yc).sum(0))
    assert abs(ac1 - (w[1:] * w[:-1]).sum() / (w ** 2).sum()) < 0.05


def test_isolated_count_response_has_the_kernel_shape(params):
    """One threshold crossing after a long silence: the output steps up by w_0 and decays as w_k."""
    T, t0 = 3000, 2500
    x = np.zeros((T, 1))
    x[::97] = 1.0                                                # sparse background so the s.d. is > 0
    x[t0 - 60:t0 + 60] = 0.0
    x[t0] = 1.0
    y = wairagkar_block(x, "threshcross", params, BIN_MS)[:, 0]
    sm = params["smoothing"]
    w = sigmoid_kernel(_bins(sm["length_ms"]), sm["center_ms"] / BIN_MS, sm["scale_ms"] / BIN_MS)
    r = (y[t0:t0 + len(w)] - y[t0 - 1]) / (y[t0] - y[t0 - 1])
    np.testing.assert_allclose(r, w / w[0], atol=0.01)           # small drift: the count enters the stats
