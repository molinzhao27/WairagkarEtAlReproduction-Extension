"""Target-free adaptation (D-063): align a new session's neural input to the session a released decoder's day
layer belongs to, using neural data only (no audio targets).

The released inference model is avg_pool1d(2) -> one day layer (Dense 512 relu -> Dense 128 relu, per time step)
-> frozen base. Three variants, simplest first:

- ``standardize``: per channel, x' = (x - mean_new) / sd_new * sd_ref + mean_ref (512 gains and offsets);
- ``coral``: x' = (x - mean_new) C_new^(-1/2) C_ref^(1/2) + mean_ref, both covariances shrunk toward their mean
  variance (``shrink``) so the inverse square root is stable (Sun et al. 2016, CORAL);
- ``mmd``: a copy of the day layer trained so that the new session's embeddings match the reference session's,
  by a multi-bandwidth Gaussian MMD^2, plus ``l2`` x mean squared change of the weights (keeps it near the original).

The first two are linear maps of the input and commute with the model's pooling, so they are fitted on pooled
frames and applied to raw bins. All fits use neural data only; nothing here reads a target.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


def pool(x: np.ndarray) -> np.ndarray:
    """(T, F) -> (T // 2, F): the model's avg_pool1d(ksize=2, strides=2) along a whole trial."""
    n = (x.shape[0] // 2) * 2
    return x[:n].reshape(-1, 2, x.shape[1]).mean(axis=1)


def frames(trials) -> np.ndarray:
    """Pooled frames of a list of (T, F) inputs, stacked."""
    return np.concatenate([pool(np.asarray(x, np.float64)) for x in trials])


def _sym_pow(c: np.ndarray, p: float) -> np.ndarray:
    w, v = np.linalg.eigh((c + c.T) / 2)
    return (v * np.clip(w, 1e-12, None) ** p) @ v.T


def _shrunk_cov(x: np.ndarray, shrink: float) -> np.ndarray:
    c = np.cov(x, rowvar=False)
    return (1 - shrink) * c + shrink * np.trace(c) / c.shape[0] * np.eye(c.shape[0])


class LinearMap:
    """x' = (x - mu_new) @ M + mu_ref, applied per bin."""

    def __init__(self, mu_new: np.ndarray, M: np.ndarray, mu_ref: np.ndarray, kind: str):
        self.mu_new, self.M, self.mu_ref, self.kind = mu_new, M, mu_ref, kind

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        return ((x - self.mu_new) @ self.M + self.mu_ref).astype(np.float32)

    def info(self) -> dict:
        d = np.diag(self.M)
        off = self.M - np.diag(d)
        return {"kind": self.kind, "diag_median": float(np.median(d)), "diag_min": float(d.min()),
                "diag_max": float(d.max()), "offdiag_rms": float(np.sqrt((off ** 2).mean())),
                "mean_shift_abs_mean": float(np.abs(self.mu_ref - self.mu_new).mean())}


def fit_standardize(new: np.ndarray, ref: np.ndarray, min_sd: float = 1e-6) -> LinearMap:
    sd_n, sd_r = np.maximum(new.std(0), min_sd), np.maximum(ref.std(0), min_sd)
    return LinearMap(new.mean(0), np.diag(sd_r / sd_n), ref.mean(0), "standardize")


def fit_coral(new: np.ndarray, ref: np.ndarray, shrink: float) -> LinearMap:
    M = _sym_pow(_shrunk_cov(new, shrink), -0.5) @ _sym_pow(_shrunk_cov(ref, shrink), 0.5)
    return LinearMap(new.mean(0), M, ref.mean(0), "coral")


def mmd2_tf(a, b, sigmas):
    """Biased multi-bandwidth Gaussian MMD^2 between two batches (TensorFlow, differentiable)."""
    import tensorflow as tf

    def sq(x, y):
        return tf.maximum(tf.reduce_sum(x * x, 1)[:, None] + tf.reduce_sum(y * y, 1)[None] - 2 * tf.matmul(x, y, transpose_b=True), 0.0)

    daa, dbb, dab = sq(a, a), sq(b, b), sq(a, b)
    out = 0.0
    for s in sigmas:
        g = 1.0 / (2.0 * s * s)
        out += tf.reduce_mean(tf.exp(-g * daa)) + tf.reduce_mean(tf.exp(-g * dbb)) - 2 * tf.reduce_mean(tf.exp(-g * dab))
    return out / len(sigmas)


def fit_mmd_layer(day_layer, new_frames: np.ndarray, ref_frames: np.ndarray, settings: Mapping[str, Any],
                  log=None) -> tuple[list, dict]:
    """Train a copy of ``day_layer`` (Keras Sequential, input (steps, 512)) so that its embeddings of
    ``new_frames`` match the original layer's embeddings of ``ref_frames`` (fixed). Returns (weights, info).

    settings: steps, batch (frames per side), lr, l2, sigma_scales (x the median distance within the reference
    embeddings), seed, log_every."""
    import tensorflow as tf

    s = settings
    rng = np.random.default_rng(int(s["seed"]))
    tf.random.set_seed(int(s["seed"]))
    w0 = [np.array(w) for w in day_layer.get_weights()]
    steps = int(day_layer.input_shape[1])

    def emb(layer, x):
        n = (len(x) // steps) * steps
        return layer(x[:n].reshape(-1, steps, x.shape[1]).astype(np.float32), training=False).numpy().reshape(-1, 128)

    z_ref = emb(day_layer, ref_frames)
    sub = z_ref[rng.choice(len(z_ref), size=min(2000, len(z_ref)), replace=False)]
    d = np.sqrt(np.maximum((sub ** 2).sum(1)[:, None] + (sub ** 2).sum(1)[None] - 2 * sub @ sub.T, 0))
    med = float(np.median(d[np.triu_indices(len(sub), 1)]))
    sigmas = [float(k) * med for k in s["sigma_scales"]]
    layer = tf.keras.models.clone_model(day_layer)
    layer.set_weights(w0)
    layer.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=float(s["lr"]))
    w0_t = [tf.constant(w) for w in w0]
    b = int(s["batch"])
    # batches are whole chunks of `steps` pooled frames from the new session (the layer is per step anyway)
    n_chunks = len(new_frames) // steps
    new_chunks = new_frames[: n_chunks * steps].reshape(n_chunks, steps, new_frames.shape[1]).astype(np.float32)
    hist = []
    for it in range(int(s["steps"])):
        xc = new_chunks[rng.choice(n_chunks, size=max(1, b // steps), replace=False)]
        zr = tf.constant(z_ref[rng.choice(len(z_ref), size=b, replace=False)])
        with tf.GradientTape() as tape:
            zn = tf.reshape(layer(xc, training=False), (-1, 128))
            m = mmd2_tf(zn, zr, sigmas)
            reg = tf.add_n([tf.reduce_mean(tf.square(w - w0w)) for w, w0w in zip(layer.trainable_weights, w0_t)])
            loss = m + float(s["l2"]) * reg
        g = tape.gradient(loss, layer.trainable_weights)
        opt.apply_gradients(zip(g, layer.trainable_weights))
        if it % int(s["log_every"]) == 0 or it == int(s["steps"]) - 1:
            hist.append({"step": it, "mmd2": float(m), "reg": float(reg)})
            if log:
                log(f"  mmd step {it}: MMD2 {float(m):.4f}, weight change {float(reg):.2e}")
    w1 = layer.get_weights()
    info = {"kind": "mmd", "sigmas": sigmas, "median_ref_distance": med, "history": hist,
            "weight_change_rel": float(np.sqrt(sum(((a - c) ** 2).sum() for a, c in zip(w1, w0))
                                               / sum((c ** 2).sum() for c in w0)))}
    return w1, info


def set_day_layer(model, weights: list) -> Optional[list]:
    """Put ``weights`` into the single day layer (Sequential) of a released inference model; returns the old
    weights (to restore)."""
    import tensorflow as tf

    days = [l for l in model.layers if isinstance(l, tf.keras.Sequential)]
    if len(days) != 1:
        raise ValueError(f"expected one day layer, got {len(days)}")
    old = days[0].get_weights()
    days[0].set_weights(weights)
    return old
