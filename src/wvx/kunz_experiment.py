"""Core experiment on Kunz T15 2024.03.03: fit on ATTEMPTED, test on IMAGINED (same day, same arrays).

Pure logic (no TensorFlow at import), used by ``scripts/kunz_experiment.py`` and unit-tested on
synthetic data in ``tests/test_kunz_experiment.py``.

**Folds** (``notes/PLAN.md`` Phase 3 c/d). Attempted blocks 3/6/11/13 pair one-to-one with imagined
blocks 4/7/9/12 (same 50 sentences each; ``notes/DATA_FORMATS.md`` §8). For a held-out attempted
block X the model trains on the other attempted blocks and is tested on:

- ``heldout_attempted``: block X (within mode, sentences unseen in training);
- ``paired_imagined``: the imagined block paired with X (same sentences as X, so also unseen in
  training) -- the key cross-mode test;
- ``eval_imagined``: block 14 (closed-loop imagined, 48/50 sentences novel; the few cues that also
  occur in training blocks are flagged per trial);
- ``leaky_imagined``: the other imagined blocks, whose sentences WERE trained on (attempted); reported
  separately and labelled sentence-leaky.

**Inputs.** Kunz columns are in Wairagkar's release order (D-024); their day <= 195 models expect the
old order, so :func:`prepare_neural` applies ``wvx.inference.reorder_channels`` (config
``experiment.reorder_channels``). The same function feeds training and decoding; decoding then calls
``predict_trial(..., reorder=False)`` so nothing is reordered twice (:func:`decode`).

**Nulls**, both seeded from the config seed and independent of the model and of the fold:

- shuffled trial (:func:`shuffled_trial_partner`): each output is scored against the TTS of another
  trial's cue from the same set (a derangement, also never the same text);
- shuffled time (:func:`shuffle_time`): the prepared input's time bins are permuted within the trial
  before decoding (key: block and trial index).

**Cue nulls and identification** (:func:`set_cues`, :func:`cue_null_stats`, :func:`identification`). Every
real output is also scored against EVERY distinct cue of its set (same output, so how speech-like it is
cancels). The single shuffled-trial partner proved too lenient and partly a sentence-length effect: with few
speech frames, ``mel_corr`` against unrelated speech rises as the reference gets shorter. So:

- ``null_all_mean``: mean over all other cues of the set;
- ``null_lenmatched_mean``: mean over the ``k`` other cues whose TTS speech duration is closest to the own
  cue's (``experiment.nulls.length_matched_k``);
- ``own_rank``: rank of the own cue among all cues (1 = best; ties count against the own cue), out of
  ``n_cues`` cues with a defined score. Under no information the rank is uniform on 1..n_cues;
  :func:`rank_sum_uniform_p` tests the rank sum exactly against that.

**Resume / provenance** (:func:`training_fingerprint`, :func:`prediction_staleness`): what a trained model
or a prediction file depends on, so ``scripts/kunz_experiment.py`` never reuses an output made under other
settings.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from wvx.config import REPO_ROOT, deep_merge
from wvx.session import SessionData, Trial

ROLES = ("heldout_attempted", "paired_imagined", "eval_imagined", "leaky_imagined")

# Independent random streams derived from the config seed (labels, not tunables).
_STREAM_SHUFFLED_TRIAL, _STREAM_SHUFFLED_TIME = 10, 11

# Per-trial columns of the three nulls on real outputs (labels written by scripts/kunz_experiment.py).
NULL_COLUMNS = {
    "lenmatched": "null_lenmatched_mean",        # k other cues closest in TTS speech duration
    "all": "null_all_mean",                      # every other cue of the set
    "shuffled_trial": "shuffled_trial_tts_mel_corr",   # one seeded other trial's cue (kept for continuity)
}
NULL_LABELS = {"lenmatched": "length-matched other cues", "all": "all other cues",
               "shuffled_trial": "one shuffled-trial partner"}

# training.* keys that only change logging or directory handling, never the trained weights.
TRAINING_KEYS_NOT_AFFECTING_MODEL = ("verbose", "log_every_batches", "overwrite")
# Prediction-header fields written before ``decode_settings`` existed (comparable when present).
_LEGACY_HEADER_SETTINGS = ("input", "reorder_channels", "seed")


# ------------------------------------------------------------------------------------- inputs


def prepare_neural(neural: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Decoder input for one Kunz trial: float32, reordered for the day <= 195 models if configured.

    The ONLY place inputs are transformed; used for training and for every decode.
    """
    x = np.asarray(neural, dtype=np.float32)
    if cfg["experiment"]["reorder_channels"]:
        from wvx.inference import reorder_channels

        x = reorder_channels(x, cfg["paths"]["electrode_mapping"]).astype(np.float32, copy=False)
    return np.ascontiguousarray(x)


def decode(model, prepared: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Normalized predictions for an input already passed through :func:`prepare_neural`."""
    from wvx.inference import predict_trial

    return predict_trial(model, prepared, cfg, reorder=False)  # never reorder twice


def _rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), *[int(k) for k in key]])


def _key(name: str) -> int:
    return zlib.crc32(name.encode())


def time_permutation(n_bins: int, seed: int, block: int, trial_index: int) -> np.ndarray:
    """Seeded permutation of a trial's time bins (shuffled-time null); never the identity for n > 1."""
    rng = _rng(seed, _STREAM_SHUFFLED_TIME, block, trial_index)
    perm = rng.permutation(n_bins)
    while n_bins > 1 and np.array_equal(perm, np.arange(n_bins)):
        perm = rng.permutation(n_bins)
    return perm


def shuffle_time(prepared: np.ndarray, seed: int, block: int, trial_index: int) -> np.ndarray:
    """The input with its rows (10 ms bins) permuted; each bin keeps all its 512 features."""
    return np.ascontiguousarray(prepared[time_permutation(len(prepared), seed, block, trial_index)])


def shuffled_trial_partner(texts: Sequence[str], seed: int, set_key: str, normalize=None) -> np.ndarray:
    """Seeded derangement ``p`` of a set: output i is scored against the cue of trial ``p[i]``.

    Uniform over permutations with no fixed point whose partner cue also differs from the trial's own
    cue (after ``normalize``, if given). ``set_key`` names the trials (e.g. ``EvalSet.key``, which does
    not depend on the fold), so the null is the same for every model and fold. Raises if impossible.
    """
    norm = normalize or (lambda s: s)
    keys = [norm(t) for t in texts]
    n = len(keys)
    if n < 2 or len(set(keys)) < 2:
        raise ValueError(f"{set_key}: need at least two different cues for a shuffled-trial null")
    rng = _rng(seed, _STREAM_SHUFFLED_TRIAL, _key(set_key))
    for _ in range(10_000):
        p = rng.permutation(n)
        if all(keys[p[i]] != keys[i] for i in range(n)):
            return p
    raise RuntimeError(f"{set_key}: no derangement with different cues found")


# -------------------------------------------------------------------------------------- folds


@dataclass
class EvalSet:
    name: str                 # e.g. "paired_imagined_I04"
    role: str                 # one of ROLES
    condition: str            # attempted | imagined | imagined_eval
    block: int
    trials: list[Trial]
    text_in_training: list[bool]   # cue (normalized) also cued in a training trial

    @property
    def texts(self) -> list[str]:
        return [t.text for t in self.trials]

    @property
    def key(self) -> str:
        """Fold-independent identity of the trials (the role of a block changes with the fold)."""
        return f"{self.condition}:B{self.block:02d}"


@dataclass
class Fold:
    heldout_block: int
    paired_block: int
    train_blocks: list[int]
    train_trials: list[Trial]          # attempted, with targets, NOT yet prepared
    sets: list[EvalSet]
    info: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"fold_A{self.heldout_block:02d}"


def _by_block(sess: SessionData) -> dict[int, list[Trial]]:
    out: dict[int, list[Trial]] = {}
    for t in sess.trials:
        out.setdefault(int(t.block), []).append(t)
    return out


def build_fold(
    attempted: SessionData,
    imagined: SessionData,
    imagined_eval: Optional[SessionData],
    heldout_block: int,
    cfg: Mapping[str, Any],
    normalize=None,
) -> Fold:
    """Training trials and evaluation sets for one held-out attempted block (see module docstring).

    Checks the pairing against the data (paired blocks must hold the same set of cues) and that no
    paired-imagined or held-out cue is in the training set. ``normalize`` maps a cue to its comparison
    key (default: ``wvx.metrics.normalize_text``).
    """
    if normalize is None:
        from wvx.metrics import normalize_text as normalize
    e = cfg["experiment"]
    pairs = {int(a): int(i) for a, i in e["pairs"].items()}
    if heldout_block not in pairs:
        raise ValueError(f"block {heldout_block} is not an attempted block of the pairing {pairs}")
    a_blocks, i_blocks = _by_block(attempted), _by_block(imagined)
    if set(a_blocks) != set(pairs) or set(i_blocks) != set(pairs.values()):
        raise ValueError(f"blocks in the data (A {sorted(a_blocks)}, I {sorted(i_blocks)}) do not match the "
                         f"configured pairing {pairs}")
    for a, i in pairs.items():
        ka = sorted(normalize(t.text) for t in a_blocks[a])
        ki = sorted(normalize(t.text) for t in i_blocks[i])
        if ka != ki:
            raise ValueError(f"A{a} and I{i} do not hold the same cues; the pairing is wrong")

    paired = pairs[heldout_block]
    train_blocks = [b for b in sorted(a_blocks) if b != heldout_block]
    train = [t for b in train_blocks for t in a_blocks[b]]
    train_keys = {normalize(t.text) for t in train}
    if any(t.target is None for t in train):
        raise ValueError("every training trial needs a target")

    def make(name, role, condition, block, trials):
        return EvalSet(name, role, condition, block, list(trials),
                       [normalize(t.text) in train_keys for t in trials])

    sets = [
        make(f"heldout_attempted_A{heldout_block:02d}", "heldout_attempted", "attempted", heldout_block,
             a_blocks[heldout_block]),
        make(f"paired_imagined_I{paired:02d}", "paired_imagined", "imagined", paired, i_blocks[paired]),
    ]
    if imagined_eval is not None and e.get("eval_block") is not None:
        eb = int(e["eval_block"])
        ev = _by_block(imagined_eval)
        if set(ev) != {eb}:
            raise ValueError(f"eval file holds blocks {sorted(ev)}, expected [{eb}]")
        sets.append(make(f"eval_imagined_E{eb:02d}", "eval_imagined", "imagined_eval", eb, ev[eb]))
    if e.get("include_leaky", True):
        for a in train_blocks:
            i = pairs[a]
            sets.append(make(f"leaky_imagined_I{i:02d}", "leaky_imagined", "imagined", i, i_blocks[i]))

    for s in sets:  # the design's guarantees, checked on the data
        if s.role in ("heldout_attempted", "paired_imagined") and any(s.text_in_training):
            raise ValueError(f"{s.name}: {sum(s.text_in_training)} cue(s) also occur in training")
        if s.role == "leaky_imagined" and not all(s.text_in_training):
            raise ValueError(f"{s.name}: expected every cue to occur in training")

    info = {
        "heldout_block": heldout_block,
        "paired_block": paired,
        "train_blocks": train_blocks,
        "n_train_pool_trials": len(train),
        "sets": {s.name: {"role": s.role, "condition": s.condition, "block": s.block, "n": len(s.trials),
                          "n_cues_in_training": int(sum(s.text_in_training))} for s in sets},
    }
    return Fold(heldout_block, paired, train_blocks, train, sets, info)


def training_session(fold: Fold, name: str, layer: int, cfg: Mapping[str, Any],
                     max_trials: Optional[int] = None) -> SessionData:
    """The fold's training trials as one SessionData with prepared inputs (``max_trials``: smoke runs)."""
    trials = fold.train_trials[:max_trials] if max_trials else fold.train_trials
    out = [Trial(neural=prepare_neural(t.neural, cfg), target=t.target, text=t.text, block=t.block,
                 meta=dict(t.meta)) for t in trials]
    return SessionData(name=name, layer=layer, trials=out)


# ---------------------------------------------------------------------------------- statistics


def feature_scores(pred: np.ndarray, target: np.ndarray) -> dict:
    """MSE and mean over features of the frame-wise Pearson r (normalized predictions vs targets)."""
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    if pred.shape != target.shape:
        raise ValueError(f"prediction {pred.shape} vs target {target.shape}")
    rs = [np.corrcoef(pred[:, k], target[:, k])[0, 1] for k in range(pred.shape[1])
          if pred[:, k].std() > 0 and target[:, k].std() > 0]
    return {"feat_mse": float(np.mean((pred - target) ** 2)),
            "feat_corr_mean": float(np.mean(rs)) if rs else float("nan")}


def describe(values: Sequence[float]) -> dict:
    """Mean, SD (ddof 1), median, quartiles and n of the finite values (n_nan counts the others)."""
    x = np.asarray([v for v in values if v is not None and v != ""], dtype=float)
    finite = x[np.isfinite(x)]
    out = {"n": int(finite.size), "n_nan": int(x.size - finite.size)}
    if finite.size:
        out.update(mean=float(finite.mean()), sd=float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                   median=float(np.median(finite)), q25=float(np.percentile(finite, 25)),
                   q75=float(np.percentile(finite, 75)))
    return out


def paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """a - b over pairs where both are finite: describe() of the differences, share a > b, Wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs equal-length inputs")
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    out = describe(d)
    out["frac_a_gt_b"] = float(np.mean(d > 0)) if d.size else float("nan")
    out["wilcoxon_p"] = float("nan")
    if d.size >= 2 and np.any(d != 0):
        from scipy.stats import wilcoxon

        out["wilcoxon_p"] = float(wilcoxon(d).pvalue)
    return out


def unpaired(a: Sequence[float], b: Sequence[float]) -> dict:
    """mean(a) - mean(b) over the finite values and the two-sided Mann-Whitney U p (unpaired)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {"n_a": int(a.size), "n_b": int(b.size), "mean_difference": float("nan"), "mannwhitney_p": float("nan")}
    if a.size and b.size:
        from scipy.stats import mannwhitneyu

        out["mean_difference"] = float(a.mean() - b.mean())
        out["mannwhitney_p"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return out


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p values (NaN entries stay NaN and do not count towards the family size)."""
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = idx.size
    running = 0.0
    for rank, i in enumerate(idx[np.argsort(p[idx], kind="stable")]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return [float(v) for v in out]


# ------------------------------------------------------------------------- cue nulls and ranks


def null_order(primary: str) -> list[str]:
    """The null names with the configured primary one first (``experiment.summary.primary_null``)."""
    if primary not in NULL_COLUMNS:
        raise ValueError(f"experiment.summary.primary_null must be one of {list(NULL_COLUMNS)}, got {primary!r}")
    return [primary] + [n for n in NULL_COLUMNS if n != primary]


def set_cues(texts: Sequence[str], normalize=None) -> tuple[list[str], list[str], list[int]]:
    """Distinct cues of a set in order of first appearance among its trials.

    Returns (keys, representative texts, own column per trial): ``keys[j]`` is the normalized cue of column
    j, ``texts[j]`` the raw text of its first trial (whose TTS is used), and ``own[i]`` the column of trial
    i's cue. Trials whose cues are equal after ``normalize`` share one column.
    """
    norm = normalize or (lambda s: s)
    keys: list[str] = []
    reps: list[str] = []
    own: list[int] = []
    pos: dict[str, int] = {}
    for t in texts:
        k = norm(t)
        if k not in pos:
            pos[k] = len(keys)
            keys.append(k)
            reps.append(t)
        own.append(pos[k])
    return keys, reps, own


def cue_null_stats(scores: Sequence[float], own: int, cue_speech_s: Sequence[float], k: int) -> dict:
    """Nulls and rank of one output from its scores against every cue of the set.

    ``scores[j]``: mel_corr of the output vs cue j's TTS (NaN = undefined); ``own``: the output's own cue;
    ``cue_speech_s[j]``: speech seconds of cue j's TTS; ``k``: how many length-matched cues.

    - ``null_all_mean``: mean of the defined scores of all other cues;
    - ``null_lenmatched_mean``: mean of the defined scores of the ``k`` other cues with the smallest
      |TTS speech duration - own cue's| (ties: lower column first); ``null_lenmatched_n`` of them defined;
    - ``own_rank``: 1 + number of other cues scoring >= the own cue (ties count against it), among
      ``n_cues`` cues with a defined score (own included). Undefined own score: rank NaN, n_cues 0.
    """
    s = np.asarray(scores, float)
    d = np.asarray(cue_speech_s, float)
    if s.ndim != 1 or s.shape != d.shape or not 0 <= own < s.size:
        raise ValueError(f"scores {s.shape}, durations {d.shape}, own {own}")
    if int(k) < 1:
        raise ValueError(f"length_matched_k must be >= 1, got {k}")
    others = np.array([j for j in range(s.size) if j != own], dtype=int)
    o = s[others]
    fin = np.isfinite(o)
    order = np.lexsort((others, np.abs(d[others] - d[own])))   # by duration distance, then column
    near = s[others[order[: int(k)]]]
    near = near[np.isfinite(near)]
    own_v = s[own]
    ok = bool(np.isfinite(own_v))
    return {
        "null_all_mean": float(o[fin].mean()) if fin.any() else float("nan"),
        "null_lenmatched_mean": float(near.mean()) if near.size else float("nan"),
        "null_lenmatched_n": int(near.size),
        "own_rank": float(1 + int(np.sum(o[fin] >= own_v))) if ok else float("nan"),
        "n_cues": int(1 + fin.sum()) if ok else 0,
        "cue_tts_speech_s": float(d[own]),
    }


def rank_sum_uniform_p(ranks: Sequence[float], n_cues: Sequence[float]) -> float:
    """Exact two-sided p of the rank sum if each rank were uniform on 1..n_cues, independently.

    Uses the trials with a defined rank and n_cues >= 1; ranks must be integers. The null distribution of
    the sum is the convolution of the discrete uniforms; p = min(1, 2 * min(P(S <= s), P(S >= s))).
    """
    r = np.asarray(ranks, float)
    n = np.asarray(n_cues, float)
    ok = np.isfinite(r) & np.isfinite(n) & (n >= 1)
    r, n = r[ok], n[ok].astype(int)
    if not r.size:
        return float("nan")
    if np.any(r != np.round(r)) or np.any(r < 1) or np.any(r > n):
        raise ValueError("ranks must be integers in 1..n_cues")
    pmf = np.ones(1)
    for size in n:
        pmf = np.convolve(pmf, np.full(size, 1.0 / size))
    i = int(r.sum()) - r.size          # pmf[0] is P(S = number of trials), every rank 1
    lower, upper = float(pmf[: i + 1].sum()), float(pmf[i:].sum())
    return float(min(1.0, 2.0 * min(lower, upper)))


def identification(ranks: Sequence[float], n_cues: Sequence[float]) -> dict:
    """How often the own cue is identified: top-1 rate (and its chance level, mean 1/n_cues), median rank,
    mean normalized rank (rank-1)/(n_cues-1) (0 = always best, 0.5 = chance), and the exact rank-sum test
    against uniform ranks (:func:`rank_sum_uniform_p`). Trials with an undefined rank are left out."""
    r = np.asarray(ranks, float)
    n = np.asarray(n_cues, float)
    ok = np.isfinite(r) & np.isfinite(n) & (n >= 1)
    r, n = r[ok], n[ok]
    out = {"n": int(r.size), "n_undefined": int(ok.size - r.size)}
    if r.size:
        multi = n > 1
        out.update(
            top1_rate=float(np.mean(r == 1)),
            top1_chance=float(np.mean(1.0 / n)),
            median_rank=float(np.median(r)),
            median_n_cues=float(np.median(n)),
            mean_normalized_rank=float(np.mean((r[multi] - 1) / (n[multi] - 1))) if multi.any() else float("nan"),
            uniform_rank_p=rank_sum_uniform_p(r, n),
        )
    return out


def output_speech_stats(wav: np.ndarray, speech: np.ndarray, hop_s: float, early_window_s: float) -> dict:
    """Speech content of one decoded output, from the metrics' own silence mask (``speech``, one bool per
    metric frame of ``hop_s`` seconds; frame t is centred at t * hop_s):

    - ``est_speech_frames`` / ``est_speech_s``: speech frames in the whole output;
    - ``early_speech_s``: speech seconds among the frames centred within the first ``early_window_s``;
    - ``peak_dbfs``: 20 log10 max|wav| (full scale 1.0); NaN for digital silence (peak 0).
    """
    speech = np.asarray(speech, bool)
    n_early = int(math.ceil(round(float(early_window_s) / float(hop_s), 6)))
    peak = float(np.max(np.abs(wav))) if np.size(wav) else 0.0
    return {
        "est_speech_frames": int(speech.sum()),
        "est_speech_s": float(speech.sum() * hop_s),
        "early_speech_s": float(speech[:n_early].sum() * hop_s),
        "peak_dbfs": 20.0 * math.log10(peak) if peak > 0 else float("nan"),
    }


# ------------------------------------------------------------------------- scoring in worker processes


@dataclass
class OutputJob:
    """One decoded output to vocode and score (:func:`score_output`); small enough to send to a worker process."""

    pred: np.ndarray                      # normalized decoder output (T, 20)
    text: str                             # own cue (a key of the TTS references)
    partner: Optional[tuple] = None       # (trial index, cue) of the shuffled-trial partner (real outputs)
    cues: Optional[list] = None           # every distinct cue of the set (real outputs: cue nulls)
    heldout: Optional[tuple] = None       # (trial index, training target) of a held-out attempted trial


_REFS: dict = {}


def init_output_scoring(cfg: Mapping[str, Any], tts: Mapping[str, np.ndarray], target: Mapping[int, np.ndarray],
                        sr: int, tts_metrics: Optional[Mapping], target_metrics: Optional[Mapping], hop_s: float,
                        early_window_s: float, blas_threads: Optional[int] = None) -> None:
    """Hold what :func:`score_output` needs in this process: the config (LPCNet), TTS audio per cue, aligned-target
    audio per held-out trial index, and the metric settings. A ``ProcessPoolExecutor`` initializer (so the
    references go to each worker once); ``blas_threads`` caps BLAS in a worker, as the workers run in parallel."""
    if blas_threads is not None:
        from threadpoolctl import threadpool_limits

        threadpool_limits(int(blas_threads))
    _REFS.clear()
    _REFS.update(cfg=cfg, tts=tts, target=target, sr=sr, tts_metrics=tts_metrics, target_metrics=target_metrics,
                 hop_s=hop_s, early_window_s=early_window_s)


def prefixed(prefix: str, scores: Mapping) -> dict:
    """Score names with ``prefix`` (unless they have it already), numpy scalars as Python scalars."""
    return {(k if k.startswith(prefix) else f"{prefix}{k}"): (v.item() if hasattr(v, "item") else v)
            for k, v in dict(scores).items()}


def score_output(job: OutputJob) -> tuple[np.ndarray, dict]:
    """LPCNet audio of one decoded output (``wvx.synthesis``; a failed synthesis raises) and its
    :func:`audio_scores`. Uses only ``job`` and what :func:`init_output_scoring` holds, so it gives the same result in
    any process. scripts/kunz_experiment.py runs it in worker processes: the LPCNet feature conversion and the
    metrics' DTW (librosa, numba without ``nogil``) hold the GIL, so threads could not run them in parallel."""
    from wvx import synthesis

    wav = synthesis.decoder_output_to_waveform(job.pred, _REFS["cfg"])
    return wav, audio_scores(wav, job)


def audio_scores(wav: np.ndarray, job: OutputJob) -> dict:
    """Audio scores of one decoded output's waveform, as per-trial fields in the per-trial table's order:

    ``tts_*`` vs its own cue's TTS (``wvx.metrics.evaluate``); its speech content (:func:`output_speech_stats`);
    ``shuffled_trial_partner`` and ``shuffled_trial_tts_*`` vs the partner's cue; ``_cue_scores``, the mel_corr vs
    EVERY cue of the set (NaN where undefined); for a held-out attempted trial the feature scores vs its target and
    ``target_*`` vs the aligned-target audio. A metric that fails is recorded as ``<prefix>error``, never fatal.
    """
    from wvx import metrics
    from wvx.inference import align_to_predictions

    ref = _REFS
    sr, tts_cfg = ref["sr"], ref["tts_metrics"]
    out: dict = {}

    def safe(prefix, fn):
        try:
            out.update(prefixed(prefix, fn()))
        except Exception as ex:  # record and go on (e.g. silent output -> metric undefined)
            out[f"{prefix}error"] = f"{type(ex).__name__}: {ex}"

    def mel(tts):
        try:
            return float(metrics.mel_spectrogram_correlation(tts, wav, sr, tts_cfg))
        except Exception:  # same policy as safe(): undefined, never fatal
            return float("nan")

    safe("tts_", lambda: metrics.evaluate(ref["tts"][job.text], wav, sr, tts_cfg))
    safe("", lambda: output_speech_stats(wav, metrics.speech_frame_mask(wav, sr, tts_cfg), ref["hop_s"],
                                         ref["early_window_s"]))
    if job.partner is not None:
        idx, cue = job.partner
        out["shuffled_trial_partner"] = int(idx)
        safe("shuffled_trial_tts_", lambda: metrics.evaluate(ref["tts"][cue], wav, sr, tts_cfg))
    if job.cues is not None:  # cue nulls: the same output vs EVERY distinct cue of its set
        out["_cue_scores"] = [mel(ref["tts"][x]) for x in job.cues]
    if job.heldout is not None:
        idx, target = job.heldout
        safe("", lambda: feature_scores(job.pred, align_to_predictions(target, len(job.pred))))
        safe("target_", lambda: metrics.evaluate(ref["target"][idx], wav, sr, ref["target_metrics"]))
    return out


def call(module: str, function: str, *args, **kwargs):
    """``module.function(*args, **kwargs)``, e.g. the configured ASR in a worker process of its own."""
    import importlib

    return getattr(importlib.import_module(module), function)(*args, **kwargs)


# ----------------------------------------------------------------------------- resume / provenance


def portable(obj: Any) -> Any:
    """JSON-ready copy for records and hashes: Paths become repo-relative strings (so a hash does not
    depend on where the repo is checked out), tuples become lists, numpy scalars Python scalars."""
    if isinstance(obj, Mapping):
        return {str(k): portable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [portable(v) for v in obj]
    if isinstance(obj, Path):
        try:
            return str(obj.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def training_fingerprint(cfg: Mapping[str, Any], variant: str, train_trials: Sequence[Trial],
                         input_files_sha256: Mapping[str, Optional[str]]) -> dict:
    """Everything a trained variant depends on: seed, deterministic ops, model, the effective training
    section (the variant's freeze_base applied; logging-only keys and the init path left out, the init file
    is identified by content), the session, the channel reorder, the training trials (block, trial index, in
    order) and the SHA-256 of the input files (training data with targets, electrode map, init model).
    Not included: run names, output paths, threads/device, the code version (recorded separately)."""
    e = cfg["experiment"]
    t = deep_merge(cfg["training"], {"init": {"freeze_base": bool(e["variants"][variant]["freeze_base"])}})
    t = {k: v for k, v in t.items() if k not in TRAINING_KEYS_NOT_AFFECTING_MODEL}
    t["init"] = {k: v for k, v in (t.get("init") or {}).items() if k != "path"}
    return portable({
        "seed": cfg["seed"],
        "deterministic_ops": bool(cfg["runtime"].get("deterministic_ops", False)),
        "model": cfg["model"],
        "training": t,
        "variant": variant,
        "session": e["session"],
        "reorder_channels": bool(e["reorder_channels"]),
        "train_trials": [[int(tr.block), int(tr.meta["trial_index"])] for tr in train_trials],
        "input_files_sha256": dict(input_files_sha256),
    })


def fingerprint_differences(old: Mapping[str, Any], new: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two fingerprints (for error messages)."""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "<absent>"), new.get(k, "<absent>")
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            out += fingerprint_differences(a, b, f"{prefix}{k}.")
        elif a != b:
            out.append(f"{prefix}{k}")
    return out


def prediction_staleness(header: Mapping[str, Any], keys: Sequence[str], model_sha256: Optional[str],
                         settings: Mapping[str, Any], expected_keys: Sequence[str]) -> list[str]:
    """Why a saved prediction file can NOT be reused (empty list: it can).

    - its header's ``model_sha256`` differs from the current model file's (``model_sha256``);
    - a decode setting differs: compared with the header's ``decode_settings``, or for files written before
      those were recorded, with the header's own ``input`` / ``reorder_channels`` / ``seed`` fields;
      settings recorded nowhere cannot be compared and are accepted;
    - a trial the current fold needs (``expected_keys``) is missing.
    """
    reasons = []
    if model_sha256 is not None and header.get("model_sha256") != model_sha256:
        reasons.append(f"made by model sha256 {str(header.get('model_sha256'))[:12]}, "
                       f"the current model is {model_sha256[:12]}")
    stored = dict(header.get("decode_settings") or {})
    for k, v in settings.items():
        if k in stored:
            old = stored[k]
        elif k in _LEGACY_HEADER_SETTINGS and k in header:
            old = header[k]
        else:
            continue
        if old != v:
            reasons.append(f"decode setting {k} was {old!r}, now {v!r}")
    missing = sorted(set(expected_keys) - set(keys))
    if missing:
        reasons.append(f"{len(missing)} needed trial(s) missing, e.g. {missing[:3]}")
    return reasons


def index_unique(keys: Sequence[Any], what: str) -> dict:
    """{key: position}; raises ValueError naming the repeated keys instead of silently keeping one."""
    out: dict = {}
    dup = []
    for i, k in enumerate(keys):
        if k in out:
            dup.append(k)
        out[k] = i
    if dup:
        raise ValueError(f"{what}: {len(dup)} repeated key(s), e.g. {dup[:3]}; pairing them would drop or "
                         f"double-count rows")
    return out
