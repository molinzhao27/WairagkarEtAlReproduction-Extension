"""Voice conversion for decoder targets (D-051): map TTS-voice targets toward the participant's target voice.

Our Kunz targets are a TTS voice (Aman) stretched onto syllables. On their own data, training toward that voice
makes the adapted decoder worse than no adaptation, even with perfect timing (D-050). This module learns a
frame-wise **ridge regression** from TTS decoder targets (the 20 decoder-target features: cepstra + pitch, scaled
and smoothed as their loader does) with ±K frames of context to their real targets, on Wairagkar trials where the
TTS is placed on THEIR true syllables (``oracle_targets``), so the two are paired frame by frame.

    pairs = [(oracle_tts_target, their_target), ...]          # (T_i, 20) each
    table, best = cross_validate(pairs, ks=[0, 2, 5], alphas=[1, 10, 100], n_folds=5, seed=42)
    model = fit(pairs, **best)
    converted = apply(model, tts_target)                        # (T, 20)

Pure numpy; no TensorFlow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

SCHEMA = "wvx-voice-conversion/1"


def stack_context(x: np.ndarray, k: int) -> np.ndarray:
    """(T, F) -> (T, F * (2k + 1)): frames t-k .. t+k side by side, edges padded with the first/last frame."""
    x = np.asarray(x, dtype=np.float64)
    if k == 0:
        return x
    pad = np.concatenate([np.repeat(x[:1], k, 0), x, np.repeat(x[-1:], k, 0)])
    return np.concatenate([pad[j: j + x.shape[0]] for j in range(2 * k + 1)], axis=1)


def fit(pairs: Sequence[tuple[np.ndarray, np.ndarray]], k: int, alpha: float) -> dict:
    """Ridge map from stacked, standardized TTS frames to their frames (all frames of all pairs)."""
    xs = [stack_context(x, k) for x, _ in pairs]
    ys = [np.asarray(y, dtype=np.float64) for _, y in pairs]
    for (x, y), xx in zip(pairs, xs):
        if np.asarray(x).shape[0] != np.asarray(y).shape[0]:
            raise ValueError(f"pair lengths differ: {np.asarray(x).shape} vs {np.asarray(y).shape}")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    mx, sx = X.mean(0), X.std(0)
    sx[sx < 1e-12] = 1.0
    Z = (X - mx) / sx
    my = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + float(alpha) * np.eye(Z.shape[1]), Z.T @ (Y - my))
    return {"k": int(k), "alpha": float(alpha), "mean_x": mx, "std_x": sx, "W": W, "b": my,
            "n_frames": int(X.shape[0]), "n_pairs": len(pairs)}


def apply(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    """Converted targets (T, n_out), float32."""
    z = (stack_context(x, int(model["k"])) - model["mean_x"]) / model["std_x"]
    return (z @ model["W"] + model["b"]).astype(np.float32)


def trial_r(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over features of the per-feature Pearson r between two (T, F) arrays (constant features skipped),
    as ``wvx.kunz_experiment.feature_scores``' feat_corr_mean."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rs = []
    for j in range(a.shape[1]):
        if a[:, j].std() > 0 and b[:, j].std() > 0:
            rs.append(float(np.corrcoef(a[:, j], b[:, j])[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")


def cross_validate(pairs: Sequence[tuple[np.ndarray, np.ndarray]], ks: Sequence[int], alphas: Sequence[float],
                   n_folds: int, seed: int) -> tuple[list[dict], dict]:
    """K-fold CV grouped by pair (seeded): for every (k, alpha), the mean per-trial r of the converted held-out
    TTS targets vs their targets. Returns (rows, best {"k", "alpha"}), best = highest mean r (ties: smaller k,
    then larger alpha)."""
    n = len(pairs)
    if n < n_folds:
        raise ValueError(f"{n} pairs for {n_folds} folds")
    order = np.random.default_rng(seed).permutation(n)
    folds = [order[f::n_folds] for f in range(n_folds)]
    rows = []
    for k in ks:
        for alpha in alphas:
            rs = np.full(n, np.nan)
            for held in folds:
                hs = set(held.tolist())
                m = fit([pairs[i] for i in range(n) if i not in hs], k, alpha)
                for i in held:
                    rs[i] = trial_r(apply(m, pairs[i][0]), pairs[i][1])
            rows.append({"k": int(k), "alpha": float(alpha), "r_mean": float(np.nanmean(rs)),
                         "r_median": float(np.nanmedian(rs))})
    best = max(rows, key=lambda r: (r["r_mean"], -r["k"], r["alpha"]))
    return rows, {"k": best["k"], "alpha": best["alpha"]}


def save(path: Path, model: Mapping[str, Any], meta: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"schema": SCHEMA, **{k: model[k] for k in ("k", "alpha", "n_frames", "n_pairs")}, "meta": meta}
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(tmp, header=np.array(json.dumps(header, default=str)),
             **{k: np.asarray(model[k]) for k in ("mean_x", "std_x", "W", "b")})
    tmp.replace(path)
    return path


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        if header.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file")
        return {**{k: header[k] for k in ("k", "alpha", "n_frames", "n_pairs")}, "meta": header.get("meta"),
                **{k: z[k] for k in ("mean_x", "std_x", "W", "b")}}


def oracle_targets(trials: Sequence[Any], syllables: Sequence[np.ndarray], corpus, tts_table: Mapping[str, np.ndarray],
                   cfg: Mapping[str, Any], valid: Optional[Sequence[bool]] = None) -> list[Optional[tuple]]:
    """Per trial, the TTS placed on THEIR syllables, as decoder targets: (target (T, 20), pairing) or None when
    ``valid[i]`` is False. Same placement as ``tts_oracle`` (D-049): forced TTS syllables, arm B mapping onto
    ``syllables[i]``; no decoder output is involved (the decoder timeline is only its length T)."""
    from wvx import targets as tg
    from wvx.syllables import align_trial_segment_driven

    sil20 = tg.silence_frame(corpus, cfg)
    out = []
    for i, t in enumerate(trials):
        if valid is not None and not valid[i]:
            out.append(None)
            continue
        n_bins = int(t.neural.shape[0]) if getattr(t, "neural", None) is not None else int(t.target.shape[0])
        res = align_trial_segment_driven(np.repeat(sil20[None], n_bins, 0), tg.reduce(corpus.lookup(t.text), cfg), sil20,
                                         cfg, tts_segments=tts_table[tg.text_key(t.text)], pred_segments=syllables[i],
                                         pred_env=np.zeros(n_bins))
        out.append((tg.decoder_target(res.target_raw, cfg), (res.info or {}).get("pairing")))
    return out
