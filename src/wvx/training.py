"""Train the Wairagkar et al. 2025 brain-to-voice decoder (their Voiceformer) on SessionData.

Theirs, called unmodified through :mod:`wvx.b2v`:
- windowing: ``data_utils.get_sliding_windows``;
- augmentation: ``training_utils.augment_neural_data``;
- batch assembly: ``training_utils.concatenate_feature_windows``;
- model build and compile: ``model_utils.build_model`` (or :mod:`wvx.session_layers` for pretrained or
  frozen starts);
- the optimisation step: ``Voiceformer.train_step``, run by Keras ``fit``;
- checkpoint loading: ``model_utils.load_model_weights``;
- inference export: ``model_utils.save_inference_model``.

Ours, and why (``notes/WAIRAGKAR_CODE.md`` §1b, §3.4-3.7, §4.3):
- **Batching.** Every batch holds windows from ONE session, and every row is tagged with that
  session's ``SessionData.layer``. Their ``make_batches_sessionwise`` tags batches with the session's
  list position (``training_utils.py:129,160``), can let a batch run into the next session, and can
  raise ``KeyError`` at the end of the data. We never call it.
- **Train/validation split** from the actual trial counts: a seeded draw of ``n_trials`` (or a
  ``fraction``) from a pool of named sessions. Theirs samples from hard-coded index ranges
  (``config.yaml:23-27``).
- **Seeded augmentation.** ``augment_neural_data`` draws from an unseeded module-level generator
  (``training_utils._RNG``). For each call we swap in a generator seeded from (config seed, session,
  trial, copy) and restore the original afterwards. Copies are made when a batch needs them instead of
  being held in memory. With ``resample_augmentations_each_epoch: false`` they are the same every
  epoch, like their precomputed copies.
- **Pairing.** Window ``k`` covers bins ``[k*step, k*step + time_steps)`` and is paired with target
  frame ``k*step + time_steps - 1``. For ``step_size`` 1 this is their ``Y[-(T - 59):]``
  (``data_utils.py:200``). We window with ``get_sliding_windows`` directly instead of
  ``format_single_trial``, whose ``np.squeeze`` drops the window axis when ``T == time_steps``.
  Trials shorter than one window are excluded and listed in ``split.json``.

Target contract: ``Trial.target`` must already be the decoder target, i.e. the output of their
``get_speech_features_from_single_trial`` (18 cepstra + 2 pitch features, min-max scaled to [-1, 1]
and Gaussian-smoothed), one row per neural bin. ``train`` does not rescale or smooth it.

Outputs, for ``run_name`` R:
- ``<paths.checkpoints>/R/``: ``best.tf`` (best ``monitor``), ``final.tf``, and
  ``inference_layerNN.h5``, one per exported input layer.
- ``<paths.results>/R/``: ``history.csv``, ``split.json``, and ``config.yaml`` (the resolved config).
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import yaml

from wvx import b2v

b2v.use_keras2()  # their code needs Keras 2; raises if TensorFlow was already imported with Keras 3

import tensorflow as tf  # noqa: E402

from wvx import session_layers  # noqa: E402
from wvx.config import REPO_ROOT, resolve_path
from wvx.seeding import set_seed
from wvx.session import SessionData

log = logging.getLogger(__name__)

_data_utils = b2v.load("lib.data_utils")          # no TensorFlow import
_training_utils = b2v.load("lib.training_utils")  # no TensorFlow import

# Independent random streams derived from the config seed (labels, not tunables).
_STREAM_SPLIT, _STREAM_PLAN, _STREAM_AUG = 0, 1, 2

_AUG_LOCK = threading.Lock()


def _rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng([seed, *key])


def _name_key(name: str) -> int:
    """Stable integer for a session name (Python's ``hash`` is salted per process)."""
    return zlib.crc32(name.encode())


def n_windows(n_bins: int, time_steps: int, step_size: int) -> int:
    """Number of sliding windows in a trial of ``n_bins`` bins (0 if shorter than one window)."""
    return 0 if n_bins < time_steps else (n_bins - time_steps) // step_size + 1


# ----------------------------------------------------------------------------- split


@dataclass
class Split:
    """Trial indices (into ``SessionData.trials``) per session, in the order sessions were given."""

    sessions: list[str]
    layers: list[int]
    train: list[list[int]]
    validation: list[list[int]]
    excluded: list[list[int]]  # shorter than one window

    def to_json(self) -> dict:
        return {
            name: {"layer": layer, "train": tr, "validation": va, "excluded_too_short": ex}
            for name, layer, tr, va, ex in zip(
                self.sessions, self.layers, self.train, self.validation, self.excluded
            )
        }


def split_trials(sessions: Sequence[SessionData], cfg: dict) -> Split:
    """Seeded train/validation split computed from the sessions' actual trial counts.

    Config ``training.validation``: exactly one of ``n_trials`` (int) or ``fraction`` (0-1, at least
    one trial), drawn without replacement from the trials of the ``pool`` sessions (``null`` = all).
    The draw depends only on the seed and the (session name, trial index) pairs, not on the order in
    which sessions are passed.
    """
    t = cfg["training"]
    v = t["validation"]
    names = [s.name for s in sessions]
    pool_names = set(v.get("pool") or names)
    unknown = pool_names - set(names)
    if unknown:
        raise ValueError(f"validation pool names sessions not being trained: {sorted(unknown)}")

    eligible, excluded = [], []
    for s in sessions:
        ok = [n_windows(len(tr.neural), cfg["model"]["window_len"], cfg["model"]["step_size"]) > 0 for tr in s.trials]
        eligible.append([j for j, good in enumerate(ok) if good])
        excluded.append([j for j, good in enumerate(ok) if not good])
        if excluded[-1]:
            log.warning("%s: %d trial(s) shorter than one window excluded", s.name, len(excluded[-1]))

    pool = sorted(
        (s.name, j) for s, ids in zip(sessions, eligible) if s.name in pool_names for j in ids
    )
    n_trials, fraction = v.get("n_trials"), v.get("fraction")
    if (n_trials is None) == (fraction is None):
        raise ValueError("set exactly one of training.validation.n_trials / .fraction")
    if fraction is not None:
        if not 0 < fraction < 1:
            raise ValueError(f"validation fraction must be in (0, 1), got {fraction}")
        n_val = max(1, int(round(fraction * len(pool))))
    else:
        n_val = int(n_trials)
    if not 1 <= n_val <= len(pool):
        raise ValueError(f"need 1..{len(pool)} validation trials from the pool, asked for {n_val}")

    picks = _rng(cfg["seed"], _STREAM_SPLIT).choice(len(pool), size=n_val, replace=False)
    chosen = {pool[k] for k in picks}
    train = [[j for j in ids if (s.name, j) not in chosen] for s, ids in zip(sessions, eligible)]
    val = [[j for j in ids if (s.name, j) in chosen] for s, ids in zip(sessions, eligible)]
    return Split(names, [s.layer for s in sessions], train, val, excluded)


# ----------------------------------------------------------------------------- batching


@dataclass(frozen=True)
class Piece:
    """Windows ``[start, stop)`` of one copy of one trial."""

    trial: int  # index into SessionData.trials
    copy: int   # 0 = the recorded trial, k > 0 = k-th augmented copy
    start: int
    stop: int


@dataclass(frozen=True)
class BatchSpec:
    session: int  # index into SessionBatcher.sessions
    layer: int    # that session's SessionData.layer
    pieces: tuple[Piece, ...]

    @property
    def size(self) -> int:
        return sum(p.stop - p.start for p in self.pieces)


class SessionBatcher:
    """Session-pure batches of sliding windows, tagged with each session's configured layer.

    Within each session the trial copies (the recorded trial plus ``n_augmentations`` augmented copies)
    are put in a seeded random order (``shuffle``), their windows are laid end to end, and the stream is
    cut into batches of ``batch_size`` windows. A batch never crosses a session boundary, so a session's
    last batch can be smaller unless ``drop_remainder``. The batches of all sessions are then put in a
    seeded random order. That matches their batch composition (consecutive windows of consecutive
    trials of one session, ``training_utils.py:94-162``), without the overlap between neighbouring
    batches and without dropping windows past ``batch_size`` in long trials.
    """

    def __init__(
        self,
        sessions: Sequence[SessionData],
        trial_ids: Sequence[Sequence[int]],
        *,
        batch_size: int,
        time_steps: int,
        step_size: int,
        seed: int,
        n_augmentations: int = 0,
        augmentation_params: Optional[dict] = None,
        shuffle: bool = False,
        reshuffle_each_epoch: bool = False,
        resample_augmentations_each_epoch: bool = False,
        drop_remainder: bool = False,
    ):
        if len(sessions) != len(trial_ids):
            raise ValueError("need one list of trial ids per session")
        self.sessions = list(sessions)
        self.trial_ids = [list(ids) for ids in trial_ids]
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.step_size = step_size
        self.seed = seed
        self.n_augmentations = n_augmentations
        self.augmentation_params = dict(augmentation_params or {})
        self.shuffle = shuffle
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.resample_augmentations_each_epoch = resample_augmentations_each_epoch
        self.drop_remainder = drop_remainder
        self._keys = [_name_key(s.name) for s in self.sessions]
        self._cache: Optional[tuple[tuple, np.ndarray]] = None
        self._n_batches: Optional[int] = None
        for s, ids in zip(self.sessions, self.trial_ids):
            for j in ids:
                tr = s.trials[j]
                if tr.target is None:
                    raise ValueError(f"{s.name} trial {j} has no target")
                if self._n_win(tr) == 0:
                    raise ValueError(f"{s.name} trial {j} is shorter than one window")

    def _n_win(self, trial) -> int:
        return n_windows(len(trial.neural), self.time_steps, self.step_size)

    @property
    def n_batches(self) -> int:
        """Batches per epoch (the same every epoch: it depends only on window counts)."""
        if self._n_batches is None:
            self._n_batches = len(self.plan(0))
        return self._n_batches

    def plan(self, epoch: int = 0) -> list[BatchSpec]:
        rng = _rng(self.seed, _STREAM_PLAN, epoch if self.reshuffle_each_epoch else 0)
        batches: list[BatchSpec] = []
        for si, (sess, ids) in enumerate(zip(self.sessions, self.trial_ids)):
            units = [(j, c) for j in ids for c in range(self.n_augmentations + 1)]
            if self.shuffle:
                units = [units[k] for k in rng.permutation(len(units))]
            pieces: list[Piece] = []
            size = 0
            for j, c in units:
                n, start = self._n_win(sess.trials[j]), 0
                while start < n:
                    take = min(n - start, self.batch_size - size)
                    pieces.append(Piece(j, c, start, start + take))
                    size += take
                    start += take
                    if size == self.batch_size:
                        batches.append(BatchSpec(si, sess.layer, tuple(pieces)))
                        pieces, size = [], 0
            if pieces and not self.drop_remainder:
                batches.append(BatchSpec(si, sess.layer, tuple(pieces)))
        if self.shuffle:
            batches = [batches[k] for k in rng.permutation(len(batches))]
        return batches

    def neural(self, session: int, trial: int, copy: int, epoch: int = 0) -> np.ndarray:
        """The (T, n_features) neural data of one copy of one trial (copy 0 = as recorded)."""
        x = self.sessions[session].trials[trial].neural
        if copy == 0:
            # their get_sliding_windows asks for a writeable view, which fails on read-only arrays
            return x if x.flags.writeable else x.copy()
        key = (session, trial, copy, epoch if self.resample_augmentations_each_epoch else 0)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        rng = _rng(self.seed, _STREAM_AUG, self._keys[session], trial, copy, key[3])
        out = _augment(x, rng, self.augmentation_params)
        self._cache = (key, out)
        return out

    def batch(self, spec: BatchSpec, epoch: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Materialize one batch: ``x`` (B, time_steps, n_features), ``session`` (B, 1), ``y`` (B, n_targets)."""
        sess = self.sessions[spec.session]
        parts = []
        for p in spec.pieces:
            x = self.neural(spec.session, p.trial, p.copy, epoch)
            win = _data_utils.get_sliding_windows(x, self.time_steps, self.step_size)[:, 0]
            y = sess.trials[p.trial].target[self.time_steps - 1 :: self.step_size]
            parts.append({"X": win[p.start : p.stop], "Y": y[p.start : p.stop]})
        x, y = _training_utils.concatenate_feature_windows(parts)
        s = np.full((len(x), 1), spec.layer, dtype=np.int32)
        return x, s, y

    def epoch(self, epoch: int = 0) -> Iterator[tuple[BatchSpec, np.ndarray, np.ndarray, np.ndarray]]:
        for spec in self.plan(epoch):
            yield (spec, *self.batch(spec, epoch))

    def dataset(self, n_features: int, n_targets: int):
        """A ``tf.data.Dataset`` of ``((x, session), y)``: one pass = one epoch.

        Keras re-creates the iterator every epoch for a dataset of known size, which calls the generator
        again, so each pass gets the next epoch number.
        """
        counter = itertools.count()

        def gen():
            e = next(counter)
            for _, x, s, y in self.epoch(e):
                yield (x, s), y

        signature = (
            (
                tf.TensorSpec((None, self.time_steps, n_features), tf.float32),
                tf.TensorSpec((None, 1), tf.int32),
            ),
            tf.TensorSpec((None, n_targets), tf.float32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=signature)
        ds = ds.apply(tf.data.experimental.assert_cardinality(self.n_batches))
        return ds.prefetch(tf.data.AUTOTUNE)


def _augment(x: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Their ``augment_neural_data``, drawing from ``rng`` instead of their unseeded module generator."""
    with _AUG_LOCK:
        saved = _training_utils._RNG
        _training_utils._RNG = rng
        try:
            return _training_utils.augment_neural_data(x, **params)
        finally:
            _training_utils._RNG = saved


# ----------------------------------------------------------------------------- training


@dataclass
class TrainResult:
    run_name: str
    checkpoint_dir: Path
    results_dir: Path
    best_checkpoint: Optional[Path]   # SavedModel (.tf) with the best `monitor` value
    final_checkpoint: Optional[Path]  # SavedModel (.tf) after the last epoch
    inference_models: dict[int, Path]  # input-layer index -> single-layer inference .h5
    history: dict[str, list[float]]
    history_csv: Path
    split_json: Path
    split: Split
    model: Any = field(default=None, repr=False)  # the trained Voiceformer, in memory


def configure_device(device: str) -> None:
    """``runtime.device``: "auto" (TensorFlow's default placement), "cpu" (hide GPUs) or "gpu" (require one).

    Must run before TensorFlow initializes its devices.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        if gpus:
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError as e:
                raise RuntimeError("runtime.device=cpu must be applied before TensorFlow runs any op") from e
    elif device == "gpu":
        if not gpus:
            raise RuntimeError("runtime.device is 'gpu' but TensorFlow sees no GPU")
    elif device != "auto":
        raise ValueError(f"runtime.device must be auto, cpu or gpu, got {device!r}")


def n_session_layers(sessions: Sequence[SessionData], cfg: dict) -> int:
    """``model.n_session_layers``, or max(layer) + 1 over the sessions when it is null."""
    if not sessions:
        raise ValueError("no sessions to train on")
    n = cfg["model"].get("n_session_layers")
    return int(n) if n is not None else max(s.layer for s in sessions) + 1


def _check_sessions(sessions: Sequence[SessionData], cfg: dict, n_layers: int) -> None:
    m = cfg["model"]
    names = [s.name for s in sessions]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate session names: {names}")
    for s in sessions:
        if not 0 <= s.layer < n_layers:
            raise ValueError(f"{s.name}: layer {s.layer} outside 0..{n_layers - 1}")
        if any(t.target is None for t in s.trials):
            raise ValueError(f"{s.name}: every trial needs a target for training")
        s.validate(m["n_features"], m["n_targets"])


def _initial_model(cfg: dict, n_layers: int):
    init = cfg["training"].get("init") or {}
    source, freeze = init.get("source"), bool(init.get("freeze_base", False))
    if source is None:
        if freeze:
            raise ValueError("training.init.freeze_base needs pretrained weights (init.source)")
        return session_layers.build_voiceformer(cfg, n_layers)
    path = resolve_path(init["path"])
    if source == "inference_h5":
        return session_layers.from_inference_h5(path, cfg, n_layers, freeze_base=freeze)
    if source == "checkpoint":
        return session_layers.from_checkpoint(path, cfg, n_layers, freeze_base=freeze)
    raise ValueError(f"training.init.source must be null, inference_h5 or checkpoint, got {source!r}")


def _prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; choose another run_name or set training.overwrite")
    path.mkdir(parents=True, exist_ok=True)


def _portable(obj: Any) -> Any:
    """Config with Paths as repo-relative strings where possible (no machine paths in results/)."""
    if isinstance(obj, dict):
        return {k: _portable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_portable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj.relative_to(REPO_ROOT)) if obj.is_relative_to(REPO_ROOT) else str(obj)
    return obj


def train(sessions: list[SessionData], cfg: dict, extra_callbacks: Sequence[Any] = ()) -> TrainResult:
    """Train a Voiceformer on ``sessions``, each using its own ``SessionData.layer`` input layer.

    ``extra_callbacks`` (e.g. progress logging) are appended to the Keras callbacks.
    """
    configure_device(cfg["runtime"]["device"])
    set_seed(cfg["seed"], cfg["runtime"].get("deterministic_ops", False))
    model_utils = b2v.load("lib.model_utils")
    m, t = cfg["model"], cfg["training"]
    run = cfg["run_name"]

    n_layers = n_session_layers(sessions, cfg)
    _check_sessions(sessions, cfg, n_layers)
    if t["export_weights"] not in ("final", "best"):
        raise ValueError("training.export_weights must be 'final' or 'best'")
    if t["export_weights"] == "best" and not t["save_best_checkpoint"]:
        raise ValueError("export_weights: best needs save_best_checkpoint: true")

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / run
    res_dir = Path(cfg["paths"]["results"]) / run
    _prepare_dir(ckpt_dir, t["overwrite"])
    _prepare_dir(res_dir, t["overwrite"])

    split = split_trials(sessions, cfg)
    split_json = res_dir / "split.json"
    split_json.write_text(json.dumps({"seed": cfg["seed"], "sessions": split.to_json()}))
    (res_dir / "config.yaml").write_text(yaml.safe_dump(_portable(cfg), sort_keys=False))

    common = dict(
        batch_size=t["batch_size"], time_steps=m["window_len"], step_size=m["step_size"],
        seed=cfg["seed"], drop_remainder=t["drop_remainder"],
    )
    train_b = SessionBatcher(
        sessions, split.train, shuffle=True,
        n_augmentations=t["n_augmentations"],
        augmentation_params=t["augmentation_params"],
        reshuffle_each_epoch=t["reshuffle_each_epoch"],
        resample_augmentations_each_epoch=t["resample_augmentations_each_epoch"],
        **common,
    )
    val_b = SessionBatcher(sessions, split.validation, shuffle=False, **common)
    if train_b.n_batches == 0 or val_b.n_batches == 0:
        raise ValueError(f"empty data: {train_b.n_batches} train / {val_b.n_batches} validation batches")
    log.info(
        "%s: %d sessions, %d layers, %d train / %d validation trials, %d train / %d validation batches",
        run, len(sessions), n_layers, sum(map(len, split.train)), sum(map(len, split.validation)),
        train_b.n_batches, val_b.n_batches,
    )

    model = _initial_model(cfg, n_layers)

    history_csv = res_dir / "history.csv"
    best = ckpt_dir / "best.tf" if t["save_best_checkpoint"] else None
    callbacks = [
        tf.keras.callbacks.CSVLogger(str(history_csv)),
        tf.keras.callbacks.EarlyStopping(
            monitor=t["monitor"], mode=t["monitor_mode"], patience=t["early_stopping_patience"]
        ),
    ]
    if best is not None:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(best), monitor=t["monitor"], mode=t["monitor_mode"], save_best_only=True
        ))
    callbacks.extend(extra_callbacks)
    history = model.fit(
        train_b.dataset(m["n_features"], m["n_targets"]),
        epochs=t["epochs"],
        validation_data=val_b.dataset(m["n_features"], m["n_targets"]),
        callbacks=callbacks,
        verbose=t["verbose"],
    )

    final = None
    if t["save_final_checkpoint"]:
        final = ckpt_dir / "final.tf"
        model.save(str(final), save_format="tf")
    if t["export_weights"] == "best":
        if not best.exists():
            raise RuntimeError(f"no best checkpoint was written ({t['monitor']} never improved)")
        model_utils.load_model_weights(model, str(best))

    token_x, _, _ = session_layers.token_batch(cfg)
    layers = t.get("export_layers")
    layers = sorted({s.layer for s in sessions}) if layers is None else list(layers)
    exported = {}
    for layer in layers:
        base = ckpt_dir / f"inference_layer{layer:02d}"
        model_utils.save_inference_model(model, token_x, str(base), day_embed_layer=layer)
        exported[layer] = base.with_suffix(".h5")

    return TrainResult(
        run_name=run,
        checkpoint_dir=ckpt_dir,
        results_dir=res_dir,
        best_checkpoint=best if best is not None and best.exists() else None,
        final_checkpoint=final,
        inference_models=exported,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        history_csv=history_csv,
        split_json=split_json,
        split=split,
        model=model,
    )
