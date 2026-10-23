import numpy as np
import pytest

from wvx import neural_align as na


def _data(seed=0, n=4000, f=12):
    r = np.random.default_rng(seed)
    ref = r.normal(size=(n, f)) @ r.normal(size=(f, f)) * 0.5 + r.normal(size=f)
    mix = np.eye(f) + 0.3 * r.normal(size=(f, f))
    new = (ref[r.permutation(n)] - ref.mean(0)) @ mix * r.uniform(0.5, 2, size=f) + r.normal(size=f) * 2
    return new, ref


def test_pool_matches_avg_pool_pairs():
    x = np.arange(14, dtype=float).reshape(7, 2)
    assert np.allclose(na.pool(x), [[1, 2], [5, 6], [9, 10]])           # last odd bin dropped, as VALID pooling


def test_standardize_matches_channel_moments():
    new, ref = _data()
    m = na.fit_standardize(new, ref)
    y = m(new)
    assert np.allclose(y.mean(0), ref.mean(0), atol=1e-4) and np.allclose(y.std(0), ref.std(0), rtol=1e-4)
    assert np.allclose(np.diag(np.diag(m.M)), m.M)                       # per channel only


def test_coral_matches_covariance_without_shrinkage():
    new, ref = _data()
    y = na.fit_coral(new, ref, shrink=0.0)(new)
    assert np.allclose(y.mean(0), ref.mean(0), atol=1e-4)
    assert np.allclose(np.cov(y, rowvar=False), np.cov(ref, rowvar=False), atol=1e-3)


def test_linear_map_commutes_with_pooling():
    new, ref = _data()
    m = na.fit_coral(new, ref, shrink=0.1)
    x = new[:101]
    assert np.allclose(na.pool(m(x)), m(na.pool(x)), atol=1e-5)


def test_identity_when_new_equals_ref():
    _, ref = _data()
    for m in (na.fit_standardize(ref, ref), na.fit_coral(ref, ref, shrink=0.1)):
        assert np.allclose(m(ref), ref, atol=1e-4)


def test_mmd_layer_reduces_mmd_and_identity_stays():
    tf = pytest.importorskip("tensorflow")
    r = np.random.default_rng(0)
    layer = tf.keras.Sequential([tf.keras.Input((5, 16)), tf.keras.layers.Dense(16, activation="relu"),
                                 tf.keras.layers.Dense(128, activation="relu")])
    ref = r.normal(size=(3000, 16)).astype(np.float32)
    new = (ref @ (np.eye(16) + 0.5 * r.normal(size=(16, 16))) + 1.0).astype(np.float32)
    s = {"steps": 150, "batch": 250, "lr": 3e-3, "l2": 0.0, "sigma_scales": [0.5, 1, 2], "seed": 0, "log_every": 50}
    w, info = na.fit_mmd_layer(layer, new, ref, s)
    h = info["history"]
    assert h[-1]["mmd2"] < 0.5 * h[0]["mmd2"]
    w_same, info_same = na.fit_mmd_layer(layer, ref, ref, dict(s, steps=30, l2=1.0))
    assert info_same["weight_change_rel"] < 0.2
