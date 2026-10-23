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


# ------------------------------------------------------------------------- same-word alignment (D-064)

def resample(seg: np.ndarray, n: int) -> np.ndarray:
    """(T, F) -> (n, F) by linear interpolation along time (word occurrences to a common length)."""
    seg = np.asarray(seg, np.float64)
    if len(seg) == 1:
        return np.repeat(seg, n, 0)
    src = np.linspace(0, len(seg) - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(seg) - 1)
    w = (src - lo)[:, None]
    return seg[lo] * (1 - w) + seg[hi] * w


def word_occurrences(trials, word_spans, words, length: int, margin: int = 0) -> list:
    """[(word, (length, F) pooled frames)] for every word with a span: spans are [onset, offset) in bins of the raw
    (T, F) inputs; pooled by 2 as the model does, widened by ``margin`` pooled frames each side, then resampled."""
    out = []
    for x, spans, ws in zip(trials, word_spans, words):
        if spans is None or len(spans) != len(ws):
            continue
        p = pool(np.asarray(x, np.float64))
        for w, (a, b) in zip(ws, np.asarray(spans).reshape(-1, 2)):
            a2, b2 = max(int(a) // 2 - margin, 0), min(int(b) // 2 + margin, len(p))
            if b2 - a2 >= 2:
                out.append((w, resample(p[a2:b2], length)))
    return out


def word_templates(occ: list, min_count: int) -> dict:
    """{word: mean resampled occurrence} for words with at least ``min_count`` occurrences."""
    by: dict = {}
    for w, seg in occ:
        by.setdefault(w, []).append(seg)
    return {w: np.mean(v, axis=0) for w, v in by.items() if len(v) >= min_count}


def paired_frames(occ: list, templates: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """(X new frames, Y template frames, number of occurrences used): every occurrence of a word with a template."""
    xs, ys = [], []
    for w, seg in occ:
        if w in templates:
            xs.append(seg), ys.append(templates[w])
    if not xs:
        raise ValueError("no word occurrence has a template")
    return np.concatenate(xs), np.concatenate(ys), len(xs)


def fit_ridge_to_identity(X: np.ndarray, Y: np.ndarray, lam_rel: float) -> LinearMap:
    """x' = (x - mu_x) @ (I + D) + mu_y, D = ridge solution of (X - mu_x) D ~ (Y - mu_y) - (X - mu_x) with penalty
    lam_rel x mean eigenvalue of the X covariance (shrinks the map toward the identity, not toward zero)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc
    lam = float(lam_rel) * np.trace(G) / G.shape[0]
    D = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ (Yc - Xc))
    return LinearMap(mx, np.eye(G.shape[0]) + D, my, "word_ridge")


def fit_procrustes(X: np.ndarray, Y: np.ndarray) -> LinearMap:
    """x' = (x - mu_x) @ R + mu_y, R orthogonal minimizing |(X - mu_x) R - (Y - mu_y)| (no shrinkage)."""
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - mx).T @ (Y - my))
    return LinearMap(mx, U @ Vt, my, "word_procrustes")


def fit_paired_layer(day_layer, X: np.ndarray, Y: np.ndarray, settings: Mapping[str, Any], log=None) -> tuple[list, dict]:
    """A copy of ``day_layer`` trained so that layer_new(X) ~ layer_old(Y) frame by frame (MSE on the 128-d
    embeddings the frozen core reads), plus ``l2`` x mean squared weight change. settings: steps, batch, lr, l2,
    seed, log_every."""
    import tensorflow as tf

    s = settings
    rng = np.random.default_rng(int(s["seed"]))
    tf.random.set_seed(int(s["seed"]))
    w0 = [np.array(w) for w in day_layer.get_weights()]
    steps = int(day_layer.input_shape[1])

    def chunks(a):
        n = (len(a) // steps) * steps
        return a[:n].reshape(-1, steps, a.shape[1]).astype(np.float32)

    Xc, Yc = chunks(X), chunks(Y)
    Zy = day_layer(Yc, training=False).numpy()
    layer = tf.keras.models.clone_model(day_layer)
    layer.set_weights(w0)
    opt = tf.keras.optimizers.Adam(learning_rate=float(s["lr"]))
    w0_t = [tf.constant(w) for w in w0]
    nb = max(1, int(s["batch"]) // steps)
    hist = []
    for it in range(int(s["steps"])):
        idx = rng.choice(len(Xc), size=min(nb, len(Xc)), replace=False)
        with tf.GradientTape() as tape:
            zn = layer(Xc[idx], training=False)
            m = tf.reduce_mean(tf.square(zn - Zy[idx]))
            reg = tf.add_n([tf.reduce_mean(tf.square(w - w0w)) for w, w0w in zip(layer.trainable_weights, w0_t)])
            loss = m + float(s["l2"]) * reg
        g = tape.gradient(loss, layer.trainable_weights)
        opt.apply_gradients(zip(g, layer.trainable_weights))
        if it % int(s["log_every"]) == 0 or it == int(s["steps"]) - 1:
            hist.append({"step": it, "mse": float(m), "reg": float(reg)})
            if log:
                log(f"  paired step {it}: MSE {float(m):.4f}, weight change {float(reg):.2e}")
    w1 = layer.get_weights()
    return w1, {"kind": "word_layer", "history": hist, "n_frames": int(len(Xc) * steps),
                "weight_change_rel": float(np.sqrt(sum(((a - c) ** 2).sum() for a, c in zip(w1, w0))
                                                   / sum((c ** 2).sum() for c in w0)))}


# ---------------------------------------------------- adversarial alignment, ADAN-style (D-065)

class Converter:
    """x' = x + relu(x W1 + b1) W2 + b2 per 10 ms bin (512 features in and out); W2, b2 start at zero, so the converter
    starts as the identity. Applied in numpy after training."""

    def __init__(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = (np.asarray(a, np.float64) for a in (W1, b1, W2, b2))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        return (x + np.maximum(x @ self.W1 + self.b1, 0) @ self.W2 + self.b2).astype(np.float32)

    def info(self) -> dict:
        return {"kind": "adversarial", "W2_rms": float(np.sqrt((self.W2 ** 2).mean())),
                "b2_abs_mean": float(np.abs(self.b2).mean())}


def sample_windows(trials, n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` random windows of ``length`` bins from a list of (T, F) trials (trial chosen by its number of windows)."""
    sizes = np.array([max(len(x) - length + 1, 0) for x in trials], float)
    p = sizes / sizes.sum()
    out = np.empty((n, length, trials[0].shape[1]), np.float32)
    for k, i in enumerate(rng.choice(len(trials), size=n, p=p)):
        s = rng.integers(0, int(sizes[i]))
        out[k] = trials[i][s: s + length]
    return out


def fit_adversarial(day_layer, new_trials, ref_trials, settings: Mapping[str, Any], log=None) -> tuple[Converter, dict]:
    """WGAN-GP between converted new-session windows and reference-session windows, both read through the FROZEN
    original day layer (60-bin windows -> pooled -> (30, 128) embeddings, standardized with the reference
    embeddings' per-dimension mean and s.d.). Critic: three 1-D convolutions over time (kernel 3, LeakyReLU 0.2),
    mean over time, linear. Converter loss: -critic(converted) + ``lam_id`` x mean squared change of the input.
    settings: hidden, window, batch, steps (converter updates), n_critic, lr, beta_1, beta_2, gp, lam_id,
    critic_channels, seed, log_every."""
    import tensorflow as tf

    s = settings
    rng = np.random.default_rng(int(s["seed"]))
    tf.random.set_seed(int(s["seed"]))
    F, H, Wn, B = new_trials[0].shape[1], int(s["hidden"]), int(s["window"]), int(s["batch"])
    init = tf.keras.initializers.GlorotUniform(seed=int(s["seed"]))
    W1 = tf.Variable(init((F, H)))
    b1 = tf.Variable(tf.zeros(H))
    W2 = tf.Variable(tf.zeros((H, F)))
    b2 = tf.Variable(tf.zeros(F))
    gen_vars = [W1, b1, W2, b2]

    def convert(x):
        return x + tf.matmul(tf.nn.relu(tf.matmul(x, W1) + b1), W2) + b2

    ref_win = sample_windows(ref_trials, 2048, Wn, rng)
    z_ref = []
    for k in range(0, len(ref_win), 256):
        z_ref.append(day_layer(tf.nn.avg_pool1d(ref_win[k: k + 256], 2, 2, "VALID"), training=False).numpy())
    z_ref = np.concatenate(z_ref).reshape(-1, 128)
    mu = tf.constant(z_ref.mean(0), tf.float32)
    sd = tf.constant(np.where(z_ref.std(0) > 1e-6, z_ref.std(0), 1.0), tf.float32)

    def emb(x):
        return (day_layer(tf.nn.avg_pool1d(x, 2, 2, "VALID"), training=False) - mu) / sd

    C = int(s["critic_channels"])
    cinit = tf.keras.initializers.GlorotUniform(seed=int(s["seed"]) + 1)
    K1, K2, K3 = (tf.Variable(cinit((3, i, C))) for i in (128, C, C))
    c1, c2, c3 = (tf.Variable(tf.zeros(C)) for _ in range(3))
    Wo, bo = tf.Variable(cinit((C, 1))), tf.Variable(tf.zeros(1))
    crit_vars = [K1, c1, K2, c2, K3, c3, Wo, bo]

    def critic(z):
        h = tf.nn.leaky_relu(tf.nn.conv1d(z, K1, 1, "SAME") + c1, 0.2)
        h = tf.nn.leaky_relu(tf.nn.conv1d(h, K2, 1, "SAME") + c2, 0.2)
        h = tf.nn.leaky_relu(tf.nn.conv1d(h, K3, 1, "SAME") + c3, 0.2)
        return tf.matmul(tf.reduce_mean(h, axis=1), Wo) + bo

    opt_c = tf.keras.optimizers.Adam(float(s["lr"]), beta_1=float(s["beta_1"]), beta_2=float(s["beta_2"]))
    opt_g = tf.keras.optimizers.Adam(float(s["lr"]), beta_1=float(s["beta_1"]), beta_2=float(s["beta_2"]))

    @tf.function
    def critic_step(xn, xr, eps):
        with tf.GradientTape() as tape:
            zf, zr = emb(convert(xn)), emb(xr)
            zi = eps * zr + (1 - eps) * zf
            with tf.GradientTape() as gt:
                gt.watch(zi)
                di = critic(zi)
            gn = tf.sqrt(tf.reduce_sum(tf.square(gt.gradient(di, zi)), axis=[1, 2]) + 1e-12)
            w = tf.reduce_mean(critic(zr)) - tf.reduce_mean(critic(zf))
            loss = -w + float(s["gp"]) * tf.reduce_mean(tf.square(gn - 1.0))
        opt_c.apply_gradients(zip(tape.gradient(loss, crit_vars), crit_vars))
        return w

    @tf.function
    def gen_step(xn):
        with tf.GradientTape() as tape:
            xc = convert(xn)
            adv = -tf.reduce_mean(critic(emb(xc)))
            idp = tf.reduce_mean(tf.square(xc - xn))
            loss = adv + float(s["lam_id"]) * idp
        opt_g.apply_gradients(zip(tape.gradient(loss, gen_vars), gen_vars))
        return adv, idp

    hist = []
    for it in range(int(s["steps"])):
        for _ in range(int(s["n_critic"])):
            w = critic_step(tf.constant(sample_windows(new_trials, B, Wn, rng)),
                            tf.constant(sample_windows(ref_trials, B, Wn, rng)),
                            tf.constant(rng.uniform(size=(B, 1, 1)).astype(np.float32)))
        adv, idp = gen_step(tf.constant(sample_windows(new_trials, B, Wn, rng)))
        if it % int(s["log_every"]) == 0 or it == int(s["steps"]) - 1:
            hist.append({"step": it, "wasserstein": float(w), "identity_mse": float(idp)})
            if log:
                log(f"  adversarial step {it}: critic distance {float(w):.3f}, input change {float(idp):.4f}")
    conv = Converter(W1.numpy(), b1.numpy(), W2.numpy(), b2.numpy())
    return conv, {"kind": "adversarial", "history": hist, **conv.info()}


# --------------------------------------------- same-day inner -> attempted conversion (D-067)

def pca_basis(frames: np.ndarray, k: int) -> np.ndarray:
    """(k, F) top-k principal axes (unstandardized covariance) of the frames, by decreasing variance."""
    x = np.asarray(frames, np.float64)
    w, v = np.linalg.eigh(np.cov(x, rowvar=False))
    return v[:, np.argsort(w)[::-1][:k]].T


def fit_projector(im: np.ndarray, att: np.ndarray, V: np.ndarray) -> tuple[LinearMap, dict]:
    """Inside the subspace of ``V`` (k, F): shift imagined frames to the attempted mean and scale their deviations by
    s = sqrt(tr C_att / tr C_im) (both in the subspace); outside it, unchanged.
    x' = (x - mu_im) @ (I + (s - 1) P) + (I - P) mu_im + P mu_att, P = V^T V."""
    mu_i, mu_a = im.mean(0), att.mean(0)
    P = V.T @ V
    ci, ca = (im - mu_i) @ V.T, (att - mu_a) @ V.T
    s = float(np.sqrt((ca ** 2).sum() / len(ca) / ((ci ** 2).sum() / len(ci))))
    F = im.shape[1]
    d = mu_a - mu_i
    info = {"scale": s, "pc_sd_ratio_att_over_im": (ca.std(0) / ci.std(0)).tolist(),
            "mean_diff_norm": float(np.linalg.norm(d)),
            "mean_diff_share_in_subspace": float(np.linalg.norm(d @ V.T) ** 2 / max(np.linalg.norm(d) ** 2, 1e-12))}
    return LinearMap(mu_i, np.eye(F) + (s - 1) * P, (np.eye(F) - P) @ mu_i + P @ mu_a, "projector"), info


def fit_paired_rotation(im_pairs: np.ndarray, att_pairs: np.ndarray, V: np.ndarray, shrink: float,
                        mu_im: np.ndarray, mu_att: np.ndarray) -> tuple[LinearMap, dict]:
    """Sentence-paired map inside the subspace of ``V``: in PC coordinates, imagined frames are whitened, rotated by
    the orthogonal matrix that best matches them to their DTW-paired attempted frames (Procrustes in whitened
    space = CCA with all canonical correlations kept at full scale, so no shrinkage), and re-colored with the
    attempted covariance; outside the subspace, unchanged. Covariances shrunk ``shrink`` toward their diagonal."""
    ci = (im_pairs - mu_im) @ V.T
    ca = (att_pairs - mu_att) @ V.T

    def shrunk(c):
        C = np.cov(c, rowvar=False)
        return (1 - shrink) * C + shrink * np.diag(np.diag(C))

    Ci, Ca = shrunk(ci), shrunk(ca)
    Wi, Ca_h = _sym_pow(Ci, -0.5), _sym_pow(Ca, 0.5)
    U, S, Vt = np.linalg.svd((ci @ Wi).T @ (ca @ _sym_pow(Ca, -0.5)) / len(ci))
    M_pc = Wi @ (U @ Vt) @ Ca_h
    F = V.shape[1]
    P = V.T @ V
    info = {"canonical_correlations": S.tolist(), "n_pairs": int(len(ci))}
    return LinearMap(mu_im, np.eye(F) - P + V.T @ M_pc @ V, (np.eye(F) - P) @ mu_im + P @ mu_att, "paired"), info
