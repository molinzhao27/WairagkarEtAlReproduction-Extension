"""wvx.voice_conversion on synthetic data (numpy only)."""

import numpy as np

from wvx import voice_conversion as vc


def test_stack_context_shapes_and_edges():
    x = np.arange(12, dtype=float).reshape(6, 2)
    z = vc.stack_context(x, 2)
    assert z.shape == (6, 10)
    assert np.array_equal(z[:, 4:6], x)                 # the centre block is the frame itself
    assert np.array_equal(z[0, 0:2], x[0]) and np.array_equal(z[-1, 8:10], x[-1])   # edges padded
    assert vc.stack_context(x, 0) is not None and vc.stack_context(x, 0).shape == (6, 2)


def test_fit_recovers_a_linear_map_with_context():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(9, 3))
    pairs = []
    for _ in range(20):
        x = rng.normal(size=(50, 3))
        pairs.append((x, vc.stack_context(x, 1) @ A + 0.5))
    m = vc.fit(pairs, k=1, alpha=1e-6)
    x = rng.normal(size=(30, 3))
    assert np.allclose(vc.apply(m, x), vc.stack_context(x, 1) @ A + 0.5, atol=1e-4)


def test_cross_validation_prefers_the_true_context_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    pairs = []
    for _ in range(25):
        x = rng.normal(size=(40, 2))
        y = np.roll(x, 2, axis=0) + 0.05 * rng.normal(size=x.shape)   # needs 2 frames of context
        pairs.append((x, y))
    rows, best = vc.cross_validate(pairs, ks=[0, 2], alphas=[1.0], n_folds=5, seed=0)
    assert best["k"] == 2 and len(rows) == 2
    m = vc.fit(pairs, **best)
    path = vc.save(tmp_path / "m.npz", m, {"note": "test"})
    m2 = vc.load(path)
    x = rng.normal(size=(10, 2))
    assert np.array_equal(vc.apply(m, x), vc.apply(m2, x)) and m2["meta"] == {"note": "test"}
