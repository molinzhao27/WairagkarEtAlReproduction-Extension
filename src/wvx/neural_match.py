"""Same-day neural matching, imagined -> attempted, with no decoder (pre-registered: notes/DECISIONS.md D-032).

Question: on Kunz T15 2024.03.03, does an IMAGINED trial's neural activity resemble the ATTEMPTED trial of
the same sentence more than attempted trials of other sentences? Paired blocks (A3/I4, A6/I7, A11/I9,
A13/I12) hold the same 50 cues, each exactly once per block, and no cue occurs in a non-paired block, so
for every imagined trial ``i_s`` there is exactly one own attempted trial ``a_s``. No decoder, no audio
targets and no day-to-day drift are involved.

Design (every number comes from ``configs/experiments/neural_match.yaml``, section ``neural_match``):

1. **Representation** (:func:`pool_frames`, :func:`fit_pca`, :func:`apply_pca`). The go window of each trial
   (decoder-input features, 512 per 10 ms bin) is average-pooled over non-overlapping ``pool_bins`` windows.
   For each pair, a PCA (``pca.n_components``) is fitted on the pooled frames of the OTHER pairs' trials,
   both conditions, no labels, after standardizing every feature by those same frames.
2. **Primary similarity** (:func:`dtw_similarity`): DTW between the imagined and attempted trajectories (cost
   ``dtw.metric``; scipy's ``cosine`` = 1 - cosine similarity; steps and band from ``dtw``), then the mean over
   components of the Pearson r between the aligned sequences: the ``mel_corr`` analogue, computed with
   ``wvx.metrics``' own DTW and row-wise Pearson helpers.
3. **Secondary similarity** (:func:`pattern_similarity`): Pearson r between the time-averaged 512-dim
   go-window patterns (no PCA, no DTW).
4. **Length-only baseline** (:func:`length_similarity`): ``-|log(attempted bins / imagined bins)|`` ("raw",
   pre-registered). ADDED 2026-10-21 before any run (``additional_baselines.length_block_normalized``,
   :data:`ADDITIONS`): the same score on durations divided by their block's median ("block-normalized").
5. **Identification** (:func:`own_rank`, :func:`length_matched_candidates`). Candidates for ``i_s``: ``a_s``
   plus the ``length_matched_k`` other attempted trials of the pair whose cues' TTS speech durations are
   closest to cue s (ties: earlier trial first), and, separately, all attempted trials of the pair. The rank
   of ``a_s`` among the candidates by similarity: 1 = best, ties get the mid-rank.
6. **Test** (:func:`null_rank_draws`, :func:`rank_stats`). Under H0 the own trial is a random candidate, so its
   rank is uniform over the candidates' ranks. Each of ``n_permutations`` seeded relabelings draws one
   candidate per trial as "own"; the one-sided p is ``(1 + #{null mean rank <= observed}) / (1 + n)``,
   pooled over pairs and per pair (the same draws). :func:`exact_mean_rank_p` gives the exact p of the same
   null by convolution, as a check on the Monte Carlo p.

Not available (stated in every output): a within-mode positive control. No sentence repeats within a
condition on this day, so there is no same-sentence ceiling for single-trial matching.

Pure NumPy/SciPy (+ librosa's DTW through ``wvx.metrics``); nothing is loaded at import, no TensorFlow.
``scripts/neural_match.py`` does the I/O; tests in ``tests/test_neural_match.py`` run on synthetic trials.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from wvx import metrics

# Labels (not tunables): similarity measures and candidate sets, primary first.
MEASURES = ("dtw", "pattern", "length", "length_norm")
PREREGISTERED_MEASURES = ("dtw", "pattern", "length")
MEASURE_LABELS = {
    "dtw": "DTW trajectory correlation (PCA, primary)",
    "pattern": "time-averaged pattern correlation (512-dim, secondary)",
    "length": "length-only baseline, -|log duration ratio|, raw (pre-registered)",
    "length_norm": "length-only baseline, -|log duration ratio|, block-normalized (added 2026-10-21 before any run)",
}
# Analyses added after the pre-registration (before any run): measure -> its switch in
# ``neural_match.additional_baselines`` and what it does. Reported as additions, never as the design.
ADDITIONS = {
    "length_norm": {
        "key": "additional_baselines.length_block_normalized",
        "label": "block-normalized (added 2026-10-21 before any run)",
        "description": "Length-only baseline on durations divided by the median go-window duration of the "
                       "trial's own block: -|log((attempted / median of its block) / (imagined / median of its "
                       "block))|; same rank statistics (length-matched and all). Imagined trials are ~1.5x longer "
                       "than attempted ones, so the raw ratio favours long attempted trials rather than the "
                       "proportionally matched one.",
    },
}
CANDIDATE_SETS = ("length_matched", "all")
PRIMARY = ("dtw", "length_matched")

LIMITATIONS = [
    "No within-mode positive control: no sentence repeats within a condition on this day, so the analysis "
    "has no same-sentence ceiling (how well a trial matches another trial of the same sentence in the same "
    "mode is unknown).",
    "At chance is NOT evidence of no shared code: single-trial matching may simply be too noisy.",
    "Direction imagined -> attempted only; trials are matched within a block pair (same day, same arrays).",
]
INTERPRETATION = (
    "Fixed in advance (D-032): above chance = imagined and attempted share sentence-specific neural structure "
    "on this day; at chance = not evidence of no shared code (no positive control)."
)


# ------------------------------------------------------------------------------------- representation


def pool_frames(x: np.ndarray, pool_bins: int, drop_partial: bool = True) -> np.ndarray:
    """(T, F) bins -> (T // pool_bins, F) means over non-overlapping windows (float64).

    A trailing partial window is dropped (``drop_partial``) or kept as the mean of its bins.
    """
    x = np.asarray(x)
    n = int(pool_bins)
    if x.ndim != 2 or n < 1:
        raise ValueError(f"need (T, F) features and pool_bins >= 1, got {x.shape} and {pool_bins}")
    m = x.shape[0] // n
    out = x[: m * n].reshape(m, n, x.shape[1]).mean(axis=1, dtype=np.float64)
    if not drop_partial and x.shape[0] % n:
        out = np.concatenate([out, x[m * n:].mean(axis=0, keepdims=True, dtype=np.float64)])
    if out.shape[0] == 0:
        raise ValueError(f"{x.shape[0]} bins give no pooled frame of {n} bins")
    return out


@dataclass
class PCAModel:
    mean: np.ndarray                      # (F,) training mean
    scale: np.ndarray                     # (F,) training s.d. (1 where not standardized or constant)
    components: np.ndarray                # (k, F) unit-norm axes, by decreasing variance
    explained_variance_ratio: np.ndarray  # (k,) of the standardized training frames' total variance
    n_frames: int

    def info(self) -> dict:
        evr = self.explained_variance_ratio
        return {"n_train_frames": int(self.n_frames), "n_components": int(evr.size),
                "explained_variance_total": float(evr.sum()),
                "explained_variance_ratio": [float(v) for v in evr]}


def fit_pca(frames: np.ndarray, n_components: int, standardize: bool = True, min_std: float = 0.0) -> PCAModel:
    """PCA of training frames (n, F), each feature standardized by these frames first (``standardize``).

    Features whose training s.d. is <= ``min_std`` are only centred. Axes come from the eigendecomposition of
    the covariance; each axis is signed so that its largest-magnitude loading is positive (deterministic).
    """
    z = np.array(frames, dtype=np.float64)          # own copy, standardized in place (memory)
    k = int(n_components)
    if z.ndim != 2 or z.shape[0] < 2 or not 1 <= k <= z.shape[1]:
        raise ValueError(f"cannot fit {n_components} components to frames of shape {z.shape}")
    mean = z.mean(axis=0)
    sd = z.std(axis=0) if standardize else np.ones(z.shape[1])
    scale = np.where(sd > float(min_std), sd, 1.0)
    z -= mean
    z /= scale
    cov = z.T @ z / (z.shape[0] - 1)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    order = np.argsort(w)[::-1][:k]
    comps = v[:, order].T
    lead = comps[np.arange(k), np.argmax(np.abs(comps), axis=1)]
    comps = comps * np.where(lead < 0, -1.0, 1.0)[:, None]
    total = float(w.sum())
    evr = w[order] / total if total > 0 else np.zeros(k)
    return PCAModel(mean=mean, scale=scale, components=comps, explained_variance_ratio=evr, n_frames=z.shape[0])


def apply_pca(model: PCAModel, x: np.ndarray) -> np.ndarray:
    """(T, F) -> (T, k): standardized by the training statistics, projected on the training axes."""
    return ((np.asarray(x, dtype=np.float64) - model.mean) / model.scale) @ model.components.T


# ----------------------------------------------------------------------------------------- similarity


def dtw_similarity(x: np.ndarray, y: np.ndarray, dtw_cfg: Mapping[str, Any]) -> float:
    """Mean over components of the Pearson r between two (T, k) trajectories after DTW alignment.

    ``dtw_cfg`` has the keys of ``configs/metrics.yaml`` ``alignment.dtw`` (metric, step_sizes, weights_add,
    weights_mul, global_constraints, band_rad); the path is ``wvx.metrics``' librosa DTW on the raw
    trajectories. Components constant along the path are skipped; NaN if all are.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"trajectories {x.shape} and {y.shape} must be (T, k) with the same k")
    if dtw_cfg["metric"] == "cosine" and (not np.all(np.any(x != 0, axis=1)) or not np.all(np.any(y != 0, axis=1))):
        raise ValueError("cosine DTW cost is undefined for an all-zero frame")
    ia, ib = metrics._align(x.T, y.T, True, {"alignment": {"dtw": dict(dtw_cfg)}}, "none")
    r = metrics._rowwise_pearson(x[ia].T, y[ib].T)
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")


def mean_pattern(x: np.ndarray) -> np.ndarray:
    """(T, F) trial -> (F,) time-averaged pattern (float64)."""
    return np.asarray(x).mean(axis=0, dtype=np.float64)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two vectors (NaN if either is constant)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"vectors {a.shape} and {b.shape} differ")
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(np.clip(a @ b / den, -1.0, 1.0)) if den > 0 else float("nan")


def pattern_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between the time-averaged feature patterns of two (T, F) trials (NaN if one is constant)."""
    return pearson(mean_pattern(x), mean_pattern(y))


def tts_speech_seconds(raw: np.ndarray, cfg: Mapping[str, Any], method: str, threshold_db: float) -> float:
    """Speech duration of one cue's TTS from its raw LPCNet frames (n, >= 1; c0 in column 0), no synthesis.

    - ``speech_span``: first to last frame whose c0 power is within ``threshold_db`` of the loudest frame
      (``wvx.alignment.speech_bounds``, the TTS speech span used to validate the alignment);
    - ``speech_frames``: frames in the alignment's speech mask (``wvx.alignment.speech_mask`` of the c0
      envelope; thresholds from ``alignment.segmentation``, ``threshold_db`` unused), pauses excluded.

    Seconds = frames x LPCNet frame duration (``synthesis`` section).
    """
    from wvx import alignment as al

    frame_s = al.frame_ms(cfg) / 1000.0
    if method == "speech_span":
        b = al.speech_bounds(raw, cfg, threshold_db)
        if b is None:
            raise ValueError("no speech frame in the TTS features")
        return (b[1] - b[0]) * frame_s
    if method == "speech_frames":
        return float(al.speech_mask(al.envelope_from_c0(raw, cfg), cfg).sum()) * frame_s
    raise ValueError(f"unknown cue_duration.method {method!r}")


def length_similarity(n_attempted: float, n_imagined: float) -> float:
    """Length-only baseline score: -|log(attempted duration / imagined duration)| (0 = equal length)."""
    if n_attempted <= 0 or n_imagined <= 0:
        raise ValueError("durations must be positive")
    return -abs(math.log(float(n_attempted) / float(n_imagined)))


def block_normalized_lengths(lengths: Sequence[float]) -> np.ndarray:
    """Durations of one block's trials divided by the block's median duration (added baseline)."""
    n = np.asarray(lengths, dtype=np.float64)
    if n.ndim != 1 or n.size == 0 or np.any(n <= 0):
        raise ValueError("need positive durations of at least one trial")
    return n / np.median(n)


def active_measures(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """The pre-registered measures plus every addition switched on in ``additional_baselines``."""
    extra = settings.get("additional_baselines") or {}
    on = {"length_norm": bool(extra.get("length_block_normalized", False))}
    return PREREGISTERED_MEASURES + tuple(m for m in ADDITIONS if on[m])


def additions(settings: Mapping[str, Any]) -> list[dict]:
    """The analyses run in addition to the pre-registered ones (for summary.json / summary.md)."""
    return [{"measure": m, **ADDITIONS[m]} for m in active_measures(settings) if m in ADDITIONS]


# ------------------------------------------------------------------------------- ranks and candidates


def midranks(scores: Sequence[float]) -> np.ndarray:
    """Rank of every score among all of them, 1 = highest; tied scores share the mean of their ranks."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or not np.all(np.isfinite(s)):
        raise ValueError("midranks needs a 1-D vector of finite scores")
    greater = (s[None, :] > s[:, None]).sum(axis=1)
    ties = (s[None, :] == s[:, None]).sum(axis=1) - 1
    return 1.0 + greater + 0.5 * ties


def own_rank(scores: Sequence[float], candidates: Sequence[int], own: int) -> tuple[float, np.ndarray]:
    """Mid-rank of candidate ``own`` among ``candidates`` by ``scores`` (indexed by candidate id).

    Candidates with a non-finite score are left out. Returns (rank of own, ranks of all kept candidates);
    (NaN, empty) if the own score is undefined.
    """
    c = np.asarray(candidates, dtype=int)
    if own not in c or len(set(c.tolist())) != c.size:
        raise ValueError("candidates must be distinct and include the own trial")
    s = np.asarray(scores, dtype=np.float64)[c]
    keep = np.isfinite(s)
    if not np.isfinite(s[c == own][0]):
        return float("nan"), np.empty(0)
    r = midranks(s[keep])
    return float(r[c[keep] == own][0]), r


def length_matched_candidates(cue_durations: Sequence[float], own: int, k: int) -> np.ndarray:
    """``own`` followed by the ``k`` other indices whose cue duration is closest to ``own``'s.

    Ties in |duration difference| go to the lower index (deterministic).
    """
    d = np.asarray(cue_durations, dtype=np.float64)
    if not 0 <= own < d.size:
        raise ValueError(f"own {own} out of range for {d.size} durations")
    if not 1 <= int(k) <= d.size - 1:
        raise ValueError(f"need 1 <= k <= {d.size - 1}, got {k}")
    if not np.all(np.isfinite(d)):
        raise ValueError("cue durations must be finite")
    others = np.array([j for j in range(d.size) if j != own], dtype=int)
    order = np.lexsort((others, np.abs(d[others] - d[own])))
    return np.concatenate([[own], others[order[: int(k)]]]).astype(int)


# ------------------------------------------------------------------------------------------- the test


def stream(seed: int, label: str) -> np.random.Generator:
    """Independent, reproducible random stream for one analysis, derived from the config seed."""
    return np.random.default_rng([int(seed), zlib.crc32(label.encode())])


def null_rank_draws(candidate_ranks: Sequence[np.ndarray], n_permutations: int, rng: np.random.Generator) -> np.ndarray:
    """(n_permutations, n_trials): per relabeling, the rank of one uniformly drawn candidate per trial."""
    n = np.array([len(c) for c in candidate_ranks], dtype=int)
    if n.size == 0 or np.any(n < 1):
        raise ValueError("every trial needs at least one candidate")
    pad = np.full((n.size, int(n.max())), np.nan)
    for t, c in enumerate(candidate_ranks):
        pad[t, : n[t]] = c
    j = np.floor(rng.random((int(n_permutations), n.size)) * n[None, :]).astype(int)
    return pad[np.arange(n.size)[None, :], j]


def _half_units(ranks) -> np.ndarray:
    h = 2.0 * np.asarray(ranks, dtype=np.float64)
    out = np.rint(h)
    if not np.allclose(h, out):
        raise ValueError("ranks must be multiples of 0.5 (mid-ranks)")
    return out.astype(np.int64)


def exact_mean_rank_p(candidate_ranks: Sequence[np.ndarray], observed: Sequence[float]) -> float:
    """Exact one-sided P(null mean rank <= observed mean rank), each trial's null uniform over its candidates'
    ranks, independently (convolution in units of half a rank)."""
    obs = _half_units(observed)
    if obs.size != len(candidate_ranks) or obs.size == 0:
        raise ValueError("one observed rank per trial needed")
    pmf, offset = np.ones(1), 0
    for c in candidate_ranks:
        h = _half_units(c)
        lo = int(h.min())
        pmf = np.convolve(pmf, np.bincount(h - lo) / h.size)
        offset += lo
    i = int(obs.sum()) - offset
    return float(min(1.0, max(0.0, pmf[: i + 1].sum()))) if i >= 0 else 0.0


def rank_stats(observed: Sequence[float], candidate_ranks: Sequence[np.ndarray], draws: np.ndarray) -> dict:
    """Mean rank, top-1 rate and their chance levels, one-sided Monte Carlo and exact p, for a set of trials.

    ``draws``: the matching columns of :func:`null_rank_draws`. Top-1 = own rank exactly 1 (a tie for best is
    not top-1); its chance level is the mean share of candidates ranked exactly 1.
    """
    r = np.asarray(observed, dtype=np.float64)
    n = np.array([len(c) for c in candidate_ranks], dtype=np.float64)
    if r.size == 0:
        return {"n_trials": 0}
    obs2 = int(_half_units(r).sum())
    null2 = np.rint(2.0 * draws.sum(axis=1)).astype(np.int64)
    return {
        "n_trials": int(r.size),
        "mean_n_candidates": float(n.mean()),
        "mean_rank": float(r.mean()),
        "chance_mean_rank": float(np.mean((n + 1) / 2)),
        "median_rank": float(np.median(r)),
        "top1_rate": float(np.mean(r == 1)),
        "top1_chance": float(np.mean([np.mean(np.asarray(c) == 1) for c in candidate_ranks])),
        "mean_normalized_rank": float(np.mean((r[n > 1] - 1) / (n[n > 1] - 1))) if np.any(n > 1) else float("nan"),
        "p_perm": float((1 + np.sum(null2 <= obs2)) / (1 + null2.size)),
        "p_exact": exact_mean_rank_p(candidate_ranks, r),
        "n_permutations": int(null2.size),
    }


# --------------------------------------------------------------------------------- the whole analysis


@dataclass
class MatchTrial:
    x: np.ndarray        # (T, F) go-window bins, decoder-input features
    key: str             # normalized cue (comparison key)
    text: str            # cue as given
    block: int
    trial_index: int
    cue_s: float         # TTS speech duration of the cue, seconds


def _by_block(trials: Sequence[MatchTrial]) -> dict[int, list[MatchTrial]]:
    out: dict[int, list[MatchTrial]] = {}
    for t in trials:
        out.setdefault(int(t.block), []).append(t)
    return {b: sorted(v, key=lambda t: t.trial_index) for b, v in out.items()}


def check_pairs(attempted: Sequence[MatchTrial], imagined: Sequence[MatchTrial],
                pairs: Mapping[int, int]) -> tuple[dict, dict]:
    """Trials by block, after checking the design on the data: the configured blocks exist, every cue occurs
    once per block, paired blocks hold the same cues, and no cue occurs in more than one pair."""
    a_blocks, i_blocks = _by_block(attempted), _by_block(imagined)
    pairs = {int(a): int(i) for a, i in pairs.items()}
    if set(a_blocks) != set(pairs) or set(i_blocks) != set(pairs.values()):
        raise ValueError(f"blocks in the data (A {sorted(a_blocks)}, I {sorted(i_blocks)}) do not match the "
                         f"configured pairs {pairs}")
    seen: dict[str, int] = {}
    for a, i in pairs.items():
        for b, ts in ((a, a_blocks[a]), (i, i_blocks[i])):
            keys = [t.key for t in ts]
            if len(set(keys)) != len(keys):
                raise ValueError(f"block {b}: a cue occurs more than once")
        ka, ki = {t.key for t in a_blocks[a]}, {t.key for t in i_blocks[i]}
        if ka != ki:
            raise ValueError(f"A{a} and I{i} do not hold the same cues ({len(ka ^ ki)} differ)")
        for key in ka:
            if key in seen:
                raise ValueError(f"cue {key!r} occurs in pairs A{seen[key]} and A{a}")
            seen[key] = a
    return a_blocks, i_blocks


def pair_label(a: int, i: int) -> str:
    return f"A{int(a):02d}-I{int(i):02d}"


def analyze(attempted: Sequence[MatchTrial], imagined: Sequence[MatchTrial], pairs: Mapping[int, int],
            settings: Mapping[str, Any], seed: int,
            log: Optional[Callable[[str], None]] = None) -> tuple[list[dict], dict]:
    """The pre-registered analysis (module docstring) on in-memory trials.

    ``settings``: the ``neural_match`` config section. Returns (one row per imagined trial, results), where
    results holds the per-pair PCA info and, per measure and candidate set, pooled and per-pair statistics.
    """
    from wvx.kunz_experiment import paired

    say = log or (lambda msg: None)
    a_blocks, i_blocks = check_pairs(attempted, imagined, pairs)
    pairs = {int(a): int(i) for a, i in pairs.items()}
    pool, drop = int(settings["pool_bins"]), bool(settings["drop_partial_window"])
    pc, dtw_cfg, k = settings["pca"], settings["dtw"], int(settings["length_matched_k"])
    pooled = {id(t): pool_frames(t.x, pool, drop) for t in [*attempted, *imagined]}
    patterns = {id(t): mean_pattern(t.x) for t in [*attempted, *imagined]}   # secondary: unpooled go window

    rows: list[dict] = []
    measures = active_measures(settings)
    per_trial: dict[tuple, list] = {(m, c): [] for m in measures for c in CANDIDATE_SETS}
    pca_info, block_median_bins = {}, {}
    for a, i in pairs.items():
        label = pair_label(a, i)
        others = [t for b, ts in a_blocks.items() if b != a for t in ts]
        others += [t for b, ts in i_blocks.items() if b != i for t in ts]
        model = fit_pca(np.concatenate([pooled[id(t)] for t in others]), int(pc["n_components"]),
                        bool(pc["standardize"]), float(pc["min_std"]))
        pca_info[label] = {**model.info(), "n_train_trials": len(others),
                           "train_blocks": sorted({int(t.block) for t in others})}
        A, I = a_blocks[a], i_blocks[i]
        za = [apply_pca(model, pooled[id(t)]) for t in A]
        zi = [apply_pca(model, pooled[id(t)]) for t in I]
        say(f"{label}: PCA on {model.n_frames} frames of {len(others)} trials (explained "
            f"{model.explained_variance_ratio.sum():.3f}); DTW {len(I)} x {len(A)}")
        S = {
            "dtw": np.array([[dtw_similarity(x, y, dtw_cfg) for y in za] for x in zi]),
            "pattern": np.array([[pearson(patterns[id(ti)], patterns[id(ta)]) for ta in A] for ti in I]),
            "length": np.array([[length_similarity(len(ta.x), len(ti.x)) for ta in A] for ti in I]),
        }
        if "length_norm" in measures:   # added baseline: durations relative to each block's median
            na, ni = block_normalized_lengths([len(t.x) for t in A]), block_normalized_lengths([len(t.x) for t in I])
            S["length_norm"] = np.array([[length_similarity(x, y) for x in na] for y in ni])
            block_median_bins[f"A{a:02d}"] = float(np.median([len(t.x) for t in A]))
            block_median_bins[f"I{i:02d}"] = float(np.median([len(t.x) for t in I]))
        a_index = {t.key: j for j, t in enumerate(A)}
        cue_s = np.array([t.cue_s for t in A], dtype=np.float64)
        for r, ti in enumerate(I):
            own = a_index[ti.key]
            cands = {"length_matched": length_matched_candidates(cue_s, own, k), "all": np.arange(len(A))}
            ta = A[own]
            row = {
                "pair": label, "attempted_block": int(a), "imagined_block": int(i),
                "imagined_trial_index": int(ti.trial_index), "attempted_trial_index": int(ta.trial_index),
                "text": ti.text, "cue_tts_speech_s": float(ti.cue_s),
                "imagined_bins": int(len(ti.x)), "attempted_bins": int(len(ta.x)),
                "imagined_frames": int(len(zi[r])), "attempted_frames": int(len(za[own])),
                "length_matched_candidates": ";".join(str(A[j].trial_index) for j in cands["length_matched"][1:]),
            }
            for m in measures:
                row[f"{m}_own_sim"] = float(S[m][r, own])
                for c in CANDIDATE_SETS:
                    rank, cranks = own_rank(S[m][r], cands[c], own)
                    oth = np.array([S[m][r, j] for j in cands[c] if j != own])
                    oth = oth[np.isfinite(oth)]
                    row[f"{m}_{c}_rank"] = rank
                    row[f"{m}_{c}_n"] = int(cranks.size)
                    row[f"{m}_{c}_others_mean"] = float(oth.mean()) if oth.size else float("nan")
                    per_trial[(m, c)].append((label, rank, cranks, row[f"{m}_own_sim"], row[f"{m}_{c}_others_mean"]))
            rows.append(row)

    n_perm = int(settings["n_permutations"])
    results: dict = {"pca": pca_info, "measures": list(measures)}
    if block_median_bins:
        results["block_median_go_bins"] = block_median_bins
    for m in measures:
        results[m] = {}
        for c in CANDIDATE_SETS:
            items = per_trial[(m, c)]
            ok = [it for it in items if np.isfinite(it[1])]
            labels = np.array([it[0] for it in ok])
            obs = np.array([it[1] for it in ok], dtype=np.float64)
            cr = [it[2] for it in ok]
            draws = null_rank_draws(cr, n_perm, stream(seed, f"{m}/{c}"))
            res = {"pooled": {**rank_stats(obs, cr, draws), "n_undefined": len(items) - len(ok)}, "per_pair": {}}
            for a, i in pairs.items():
                sel = np.flatnonzero(labels == pair_label(a, i))
                res["per_pair"][pair_label(a, i)] = rank_stats(obs[sel], [cr[j] for j in sel], draws[:, sel])
            # Descriptive, not part of the pre-registered test: own similarity minus the candidates' mean.
            res["own_minus_others_mean"] = paired([it[3] for it in items], [it[4] for it in items])
            results[m][c] = res
    results["primary"] = {"measure": PRIMARY[0], "candidates": PRIMARY[1], **results[PRIMARY[0]][PRIMARY[1]]}
    return rows, results


# ------------------------------------------------------------------------------------------ reporting


def deviations(settings: Mapping[str, Any], preregistered: Mapping[str, Any], prefix: str = "") -> list[dict]:
    """Every pre-registered setting whose value in ``settings`` differs: [{key, preregistered, used}]."""
    out = []
    for key, want in preregistered.items():
        got = settings.get(key, "<missing>") if isinstance(settings, Mapping) else "<missing>"
        if isinstance(want, Mapping):
            out += deviations(got if isinstance(got, Mapping) else {}, want, f"{prefix}{key}.")
        elif got != want:
            out.append({"key": f"{prefix}{key}", "preregistered": want, "used": got})
    return out


def _fmt_p(p) -> str:
    return "n/a" if p is None or (isinstance(p, float) and not math.isfinite(p)) else f"{p:.2g}"


def _stats_line(name: str, s: Mapping[str, Any]) -> str:
    if not s.get("n_trials"):
        return f"| {name} | 0 | | | | | |"
    return (f"| {name} | {s['n_trials']} | {s['mean_rank']:.2f} ({s['chance_mean_rank']:.2f}) | "
            f"{s['median_rank']:.1f} | {s['top1_rate']:.3f} ({s['top1_chance']:.3f}) | {_fmt_p(s['p_perm'])} | "
            f"{_fmt_p(s['p_exact'])} |")


def summary_markdown(summary: Mapping[str, Any]) -> str:
    """Human-readable report of ``scripts/neural_match.py``'s summary.json."""
    res = summary["results"]
    s = summary["settings"]
    lines = [
        "# Same-day neural matching, imagined -> attempted (pre-registered, D-032)",
        "",
        "Does an imagined trial's neural activity resemble the attempted trial of the same sentence (same day, "
        "paired blocks) more than attempted trials of other sentences? No decoder, no audio targets, no drift.",
        "",
        f"- Representation: {s['pool_bins']}-bin mean pooling, PCA to {s['pca']['n_components']} components "
        f"fitted on the other three pairs (standardized: {s['pca']['standardize']}).",
        f"- Primary: DTW (cost {s['dtw']['metric']}, steps {s['dtw']['step_sizes']}, band "
        f"{s['dtw']['band_rad'] if s['dtw']['global_constraints'] else 'none'}), mean Pearson r over components; "
        f"own attempted trial ranked among itself + {s['length_matched_k']} attempted trials closest in cue TTS "
        f"speech duration ({summary['cue_duration']['method']}).",
        f"- One-sided p: {s['n_permutations']} seeded relabelings (seed {summary['seed']}); exact p by "
        "convolution as a check. Ranks: 1 = best, ties mid-ranked; top-1 = rank exactly 1.",
        "",
    ]
    dev = summary.get("deviations_from_preregistration") or []
    lines.append("**Deviations from the pre-registration:** " + (
        "none." if not dev else "; ".join(f"{d['key']} = {d['used']} (pre-registered {d['preregistered']})"
                                         for d in dev)))
    add = summary.get("additions_to_preregistration") or []
    lines += ["", "**Additions to the pre-registration (reported next to it, not replacing it):** " + (
        "none." if not add else "; ".join(f"{d['label']}: {d['description']} (switch: {d['key']})" for d in add))]
    lines += ["", "Table cells: mean rank (chance), median rank, top-1 rate (chance), p_perm, p_exact.", ""]
    header = ["| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |",
              "|---|---|---|---|---|---|---|"]
    measures = res.get("measures") or [m for m in MEASURES if m in res]
    order = [PRIMARY] + [(m, c) for m in measures for c in CANDIDATE_SETS if (m, c) != PRIMARY]
    for m, c in order:
        r = res[m][c]
        tag = "PRIMARY: " if (m, c) == PRIMARY else ""
        lines += [f"## {tag}{MEASURE_LABELS[m]}, candidates: {c.replace('_', '-')}", ""] + header
        lines.append(_stats_line("pooled", r["pooled"]))
        lines += [_stats_line(k, v) for k, v in r["per_pair"].items()]
        d = r["own_minus_others_mean"]
        if d.get("n"):
            lines += ["", f"Descriptive: own similarity minus the other candidates' mean = {d['mean']:.4f} "
                          f"(own higher in {d['frac_a_gt_b']:.0%} of {d['n']} trials; Wilcoxon p "
                          f"{_fmt_p(d['wilcoxon_p'])})."]
        lines.append("")
    lines += ["## PCA (fitted on the other pairs)", "", "| Pair | Train trials | Train frames | Explained variance |",
              "|---|---|---|---|"]
    lines += [f"| {k} | {v['n_train_trials']} | {v['n_train_frames']} | {v['explained_variance_total']:.3f} |"
              for k, v in res["pca"].items()]
    cd = summary["cue_duration"]
    lines += ["", f"Cue TTS speech duration ({cd['method']}, voice {cd['voice']}): mean "
                  f"{cd['seconds']['mean']:.2f} s, range {cd['min_s']:.2f}-{cd['max_s']:.2f} s.", "",
              "## Limitations", ""]
    lines += [f"- {x}" for x in summary["limitations"]]
    lines += ["", f"Interpretation rule: {summary['interpretation_rule']}", "",
              f"Code {summary['code_version'].get('commit')} (dirty: {summary['code_version'].get('dirty')}), "
              f"config sha256 {summary['config_sha256'][:12]}.", ""]
    return "\n".join(lines)
