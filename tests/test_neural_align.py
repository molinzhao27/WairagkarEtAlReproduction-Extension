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


def test_resample_endpoints_and_length():
    seg = np.arange(10, dtype=float)[:, None] * [1, 2]
    r = na.resample(seg, 4)
    assert r.shape == (4, 2) and np.allclose(r[0], seg[0]) and np.allclose(r[-1], seg[-1])


def test_word_maps_recover_a_known_mixing():
    rng = np.random.default_rng(3)
    F, L = 10, 8
    temps = {w: rng.normal(size=(L, F)) for w in "abcdef"}
    R = np.linalg.qr(rng.normal(size=(F, F)))[0]                    # a rotation (invisible to moment matching)
    occ = [(w, (temps[w] + 0.05 * rng.normal(size=(L, F))) @ R.T + 0.3) for w in "abcdef" for _ in range(6)]
    X, Y, n = na.paired_frames(occ, temps)
    for m in (na.fit_procrustes(X, Y), na.fit_ridge_to_identity(X, Y, 1e-3)):
        err = np.mean([np.abs(m(seg) - temps[w]).mean() for w, seg in occ])
        assert err < 0.1, (m.kind, err)
    assert na.fit_procrustes(Y, Y).info()["offdiag_rms"] < 1e-6     # identity when nothing changed


def test_converter_starts_as_identity_and_adversarial_runs():
    tf = pytest.importorskip("tensorflow")
    r = np.random.default_rng(0)
    layer = tf.keras.Sequential([tf.keras.Input((30, 16)), tf.keras.layers.Dense(16, activation="relu"),
                                 tf.keras.layers.Dense(128, activation="relu")])
    ref = [r.normal(size=(200, 16)).astype(np.float32) for _ in range(6)]
    new = [(x @ (np.eye(16) + 0.3 * r.normal(size=(16, 16))) + 0.5).astype(np.float32) for x in ref]
    s = {"hidden": 32, "window": 60, "batch": 16, "steps": 2, "n_critic": 2, "lr": 1e-4, "beta_1": 0.0,
         "beta_2": 0.9, "gp": 10.0, "lam_id": 1.0, "critic_channels": 16, "seed": 0, "log_every": 1}
    conv0 = na.Converter(np.ones((16, 4)), np.zeros(4), np.zeros((4, 16)), np.zeros(16))
    assert np.allclose(conv0(ref[0]), ref[0], atol=1e-6)
    conv, info = na.fit_adversarial(layer, new, ref, s)
    assert conv(new[0]).shape == new[0].shape and len(info["history"]) == 2
    w = na.sample_windows(ref, 5, 60, np.random.default_rng(1))
    assert w.shape == (5, 60, 16)


def test_projector_shifts_and_scales_only_inside_the_subspace():
    rng = np.random.default_rng(5)
    F, k = 12, 3
    V = np.linalg.qr(rng.normal(size=(F, F)))[0][:, :k].T
    lat = rng.normal(size=(4000, k))
    att = lat * 3 @ V + rng.normal(size=(4000, F)) * 0.5 + 2.0 * V[0]          # big subspace signal, offset on PC1
    im = lat * 1 @ V + rng.normal(size=(4000, F)) * 0.5                          # same signal, 3x weaker, no offset
    m, info = na.fit_projector(im, att, V)
    y = m(im)
    assert np.allclose((y.mean(0) - att.mean(0)) @ V.T, 0, atol=0.05)          # subspace mean -> attempted
    assert abs(np.std(y @ V.T) / np.std(att @ V.T) - 1) < 0.05                   # subspace magnitude -> attempted
    Q = np.eye(F) - V.T @ V
    assert np.allclose(y @ Q, im @ Q, atol=1e-4)                                 # outside: unchanged
    assert info["scale"] > 2.5


def test_paired_rotation_recovers_a_rotation_inside_the_subspace():
    rng = np.random.default_rng(6)
    F, k = 10, 4
    V = np.eye(F)[:k]
    R = np.linalg.qr(rng.normal(size=(k, k)))[0]
    a = rng.normal(size=(3000, k)) * [3, 2, 1.5, 1]
    att = np.c_[a, rng.normal(size=(3000, F - k))]
    im = np.c_[(a @ R) * 0.5 + 0.02 * rng.normal(size=(3000, k)), rng.normal(size=(3000, F - k))]
    m, info = na.fit_paired_rotation(im, att, V, 0.0, im.mean(0), att.mean(0))
    y = m(im)
    assert np.mean(np.abs(y[:, :k] - att[:, :k])) < 0.15 and min(info["canonical_correlations"]) > 0.9
