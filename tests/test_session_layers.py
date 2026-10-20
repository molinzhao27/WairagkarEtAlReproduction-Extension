"""New session layers from pretrained weights: Dryad-style inference .h5 shim and our .tf checkpoints."""

import importlib.util

import numpy as np
import pytest

from wvx import b2v

pytestmark = [
    pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py"),
    # find_spec, not importorskip: importing TensorFlow before wvx.b2v would load Keras 3.
    pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="needs tensorflow"),
]

from wvx.config import load_config  # noqa: E402

EXPERIMENT = "configs/experiments/reproduce_training.yaml"


def tiny_cfg(tmp_path):
    return load_config(EXPERIMENT, overrides={
        "paths": {"checkpoints": str(tmp_path / "ckpt"), "results": str(tmp_path / "results")},
        "model": {"n_heads": 1, "n_transformer_blocks": 1, "mlp_units": [8]},
        "training": {"batch_size": 16},
    })


def _x(cfg, n=8, seed=0):
    m = cfg["model"]
    return np.random.default_rng(seed).standard_normal((n, m["window_len"], m["n_features"])).astype(np.float32)


def _sess(n, layer):
    return np.full((n, 1), layer, np.int32)


def _same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def _source_model(cfg, n_layers, seed=1):
    """Tiny Voiceformer whose input layers all differ (random weights per layer)."""
    from wvx import session_layers

    model = session_layers.build_voiceformer(cfg, n_layers)
    rng = np.random.default_rng(seed)
    for layer in model.day_embedding:
        layer.set_weights([(0.05 * rng.standard_normal(w.shape)).astype(np.float32) for w in layer.get_weights()])
    return model


def _export_h5(model, cfg, path, layer):
    from wvx import session_layers

    token_x, _, _ = session_layers.token_batch(cfg)
    b2v.load("lib.model_utils").save_inference_model(model, token_x, str(path), day_embed_layer=layer)
    return path.with_suffix(".h5")


def test_inference_h5_round_trip_predicts_identically(tmp_path):
    from wvx import session_layers
    import tensorflow as tf

    cfg = tiny_cfg(tmp_path)
    src = _source_model(cfg, 2)
    h5_path = _export_h5(src, cfg, tmp_path / "pre", layer=1)
    h5 = tf.keras.models.load_model(str(h5_path), compile=False)

    model = session_layers.from_inference_h5(h5_path, cfg, n_layers=3)
    assert model.n_sessions == 3
    assert _same(model.base_transformer.get_weights(), src.base_transformer.get_weights())
    x = _x(cfg)
    want = h5.predict(x, verbose=0)
    np.testing.assert_allclose(want, src.predict([x, _sess(len(x), 1)], verbose=0), atol=1e-6)
    for i in range(3):  # every new layer is a copy of the pretrained one
        assert _same(model.day_embedding[i].get_weights(), src.day_embedding[1].get_weights())
        np.testing.assert_allclose(model.predict([x, _sess(len(x), i)], verbose=0), want, atol=1e-6)


def test_frozen_base_is_unchanged_after_training_steps(tmp_path):
    from wvx import session_layers
    import tensorflow as tf

    cfg = tiny_cfg(tmp_path)
    h5_path = _export_h5(_source_model(cfg, 1), cfg, tmp_path / "pre", layer=0)
    model = session_layers.from_inference_h5(h5_path, cfg, n_layers=2, freeze_base=True)
    assert model.base_trans_trainable_vars == []  # captured in Voiceformer.__init__

    base0 = model.base_transformer.get_weights()
    day0, day1 = model.day_embedding[0].get_weights(), model.day_embedding[1].get_weights()
    rng = np.random.default_rng(3)
    n = cfg["training"]["batch_size"]
    x = _x(cfg, n=3 * n)
    y = rng.uniform(-1, 1, (3 * n, cfg["model"]["n_targets"])).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(((x, _sess(len(x), 1)), y)).batch(n)
    model.fit(ds, epochs=1, verbose=0)  # their Voiceformer.train_step

    assert _same(model.base_transformer.get_weights(), base0)
    assert not _same(model.day_embedding[1].get_weights(), day1)  # the trained layer moved
    assert _same(model.day_embedding[0].get_weights(), day0)


def test_checkpoint_adds_layers_through_their_load_model_weights(tmp_path):
    from wvx import session_layers

    cfg = tiny_cfg(tmp_path)
    src = _source_model(cfg, 2)
    path = tmp_path / "full.tf"
    src.save(str(path), save_format="tf")

    model = session_layers.from_checkpoint(path, cfg, n_layers=4)
    assert _same(model.base_transformer.get_weights(), src.base_transformer.get_weights())
    for i, from_layer in enumerate([0, 1, 1, 1]):  # new layers start from the last loaded layer
        assert _same(model.day_embedding[i].get_weights(), src.day_embedding[from_layer].get_weights())


def test_unexpected_h5_layout_is_rejected(tmp_path):
    from wvx import session_layers
    import tensorflow as tf

    cfg = tiny_cfg(tmp_path)
    plain = tf.keras.Sequential([tf.keras.Input((4,)), tf.keras.layers.Dense(2)])
    plain.save(str(tmp_path / "plain.h5"))
    with pytest.raises(ValueError, match="expected one Sequential input layer"):
        session_layers.from_inference_h5(tmp_path / "plain.h5", cfg, n_layers=1)
