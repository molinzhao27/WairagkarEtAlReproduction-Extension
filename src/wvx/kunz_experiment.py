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
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from wvx.session import SessionData, Trial

ROLES = ("heldout_attempted", "paired_imagined", "eval_imagined", "leaky_imagined")

# Independent random streams derived from the config seed (labels, not tunables).
_STREAM_SHUFFLED_TRIAL, _STREAM_SHUFFLED_TIME = 10, 11


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
