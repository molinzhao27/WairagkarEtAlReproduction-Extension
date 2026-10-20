"""Build a Voiceformer, optionally with new session (input) layers taken from pretrained weights.

All model code is theirs (``models/brain2voice.py``, ``lib/model_utils.py``), imported unmodified
through :mod:`wvx.b2v`. This module only moves weights between their models
(``notes/WAIRAGKAR_CODE.md`` §4.4):

(a) :func:`from_inference_h5`: from a Dryad-style single-layer inference graph, as written by their
    ``model_utils.save_inference_model`` (Input -> avg_pool1d -> one input layer (``Sequential``) ->
    base transformer (``Functional``)). Upstream has no path for this. Every one of the K new input
    layers starts as a copy of the pretrained input layer, and the base gets the pretrained base weights.
(b) :func:`from_checkpoint`: from one of our own full Voiceformer SavedModels (``.tf``), through their
    ``model_utils.load_model_weights``. That copies the base and the existing input layers, and starts
    any extra layers as copies of the checkpoint's last layer.

Freezing the base: ``Voiceformer.__init__`` captures the trainable-variable lists once
(``brain2voice.py:210-216``), and their ``build_model`` forces ``base.trainable = True``
(``model_utils.py:49``). So for a frozen base we build with their classes ourselves, set
``trainable = False`` before the Voiceformer is constructed, and compile with the same settings as
``build_model``. Dropout in the base stays active during their ``train_step`` (``brain2voice.py:294``).

Channel order is the caller's concern: the Dryad models for t15_day025-t15_day195 expect the deprecated
electrode order (their ``reorder_channels``, §1c).

Settings come from the config: ``model`` (their architecture keys plus ``n_features``/``n_targets``)
and ``training`` (their optimizer/loss keys); window length from ``model.window_len``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

from wvx import b2v

b2v.use_keras2()  # their code needs Keras 2; raises if TensorFlow was already imported with Keras 3

import tensorflow as tf  # noqa: E402

PathLike = Union[str, Path]


def token_batch(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One all-zero sample ``(x, y, session)`` with the configured shapes.

    Their ``build_model`` uses its token batch only for shapes and a warm-up call
    (``model_utils.py:42,64``), and ``save_inference_model`` only for ``x.shape[1:]``.
    """
    m, t = cfg["model"], cfg["training"]
    x = np.zeros((1, m["window_len"], m["n_features"]), np.float32)
    y = np.zeros((1, m["n_targets"]), np.float32)
    s = np.zeros((1, 1), np.int32)
    return x, y, s


def build_voiceformer(cfg: dict, n_layers: int, *, freeze_base: bool = False) -> Any:
    """A compiled Voiceformer with ``n_layers`` input layers, built inside a MirroredStrategy scope.

    With a trainable base this is exactly their ``model_utils.build_model``. With ``freeze_base`` the base
    is frozen before the Voiceformer captures its variable lists, and the compile settings are the same
    as theirs (Adam(lr, beta_1, beta_2, epsilon), Huber(delta), Accuracy metric).
    """
    model_utils = b2v.load("lib.model_utils")
    x, y, s = token_batch(cfg)
    if not freeze_base:
        return model_utils.build_model(cfg, x, y, s, n_layers)

    brain2voice = b2v.load("models.brain2voice")
    m, t = cfg["model"], cfg["training"]
    with tf.distribute.MirroredStrategy().scope():
        base = brain2voice.base_transformer(
            transformer_input_shape=tuple(m["transformer_input_shape"]),
            output_dim=y.shape[-1],
            head_size=m["head_size"],
            n_heads=m["n_heads"],
            n_transformer_blocks=m["n_transformer_blocks"],
            mlp_units=m["mlp_units"],
            dropout=m["dropout"],
        )
        base.trainable = False
        model = brain2voice.Voiceformer(base, n_sessions=n_layers)
        model([x, s])  # warm-up call, as in build_model
        optimizer = tf.keras.optimizers.Adam(
            beta_1=t["adam_beta_1"],
            beta_2=t["adam_beta_2"],
            epsilon=t["adam_epsilon"],
            learning_rate=t["learning_rate"],
        )
        accuracy = tf.keras.metrics.Accuracy()
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=t["huber_delta"]),
        metrics=[accuracy],
    )
    return model


def read_inference_h5(path: PathLike) -> tuple[Any, Any]:
    """Load a single-layer inference ``.h5`` and return its ``(input_layer, base_transformer)``."""
    h = tf.keras.models.load_model(str(path), compile=False)
    days = [l for l in h.layers if isinstance(l, tf.keras.Sequential)]
    bases = [
        l for l in h.layers
        if isinstance(l, tf.keras.Model) and not isinstance(l, tf.keras.Sequential)
    ]
    if len(days) != 1 or len(bases) != 1:
        kinds = [f"{type(l).__name__}:{l.name}" for l in h.layers]
        raise ValueError(
            f"{path}: expected one Sequential input layer and one Functional base, got {kinds}"
        )
    return days[0], bases[0]


def from_inference_h5(
    path: PathLike, cfg: dict, n_layers: int, *, freeze_base: bool = False
) -> Any:
    """Voiceformer with ``n_layers`` new input layers, each a copy of the ``.h5``'s input layer.

    The base gets the ``.h5``'s base weights. ``set_weights`` raises if the configured architecture does
    not match the file.
    """
    day, base = read_inference_h5(path)
    model = build_voiceformer(cfg, n_layers, freeze_base=freeze_base)
    model.base_transformer.set_weights(base.get_weights())
    for layer in model.day_embedding:
        layer.set_weights(day.get_weights())
    return model


def from_checkpoint(
    path: PathLike, cfg: dict, n_layers: int, *, freeze_base: bool = False
) -> Any:
    """Voiceformer with ``n_layers`` input layers, weights from one of our full ``.tf`` checkpoints.

    Uses their ``load_model_weights``: layers that exist in the checkpoint are copied, and the extra
    ones start as copies of the checkpoint's last input layer.
    """
    model = build_voiceformer(cfg, n_layers, freeze_base=freeze_base)
    b2v.load("lib.model_utils").load_model_weights(model, str(path))
    return model
