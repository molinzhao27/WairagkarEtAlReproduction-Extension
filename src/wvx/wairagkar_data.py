"""Load one session of the Wairagkar et al. 2025 Dryad release (``t15_neural_data``) as SessionData.

PROVISIONAL. This module was written from their code alone; no released file has been opened yet
(the data lives only on the GPU host). File names, ``.mat`` keys and array layout follow their
loaders exactly (``lib/dataloader.py:58-80``, ``lib/inference_utils.py:38-51``), and the target
preprocessing is their own ``data_utils.get_speech_features_from_single_trial``. Run
``inspect_session_files`` and ``describe_session`` on the first real session and check:

1. Folder ``<data_dir>/<session>/`` holds ``<session>_neural_data.mat`` and
   ``<session>_target_speech_feats.mat`` (also for the held-out t15_day195), and the pretrained
   ``brain2voice_t15_day193.h5`` in ``t15_day195/``.
2. Neural keys ``spikepow_trials``, ``threshcross_trials``, ``cues`` (or ``sentences``) and
   ``session`` exist; each trial is ``T x 256``; spikepow and threshcross have equal T per trial;
   the dtype (float32 vs float64) is recorded in ``Trial.meta``.
3. ``target_speech_feats_word`` has one array per trial with **exactly T rows** (their code assumes
   this without checking; we raise) and 20 or 36 columns (unknown until opened).
4. Targets are in LPCNet's raw range: after scaling with the JSON min/max almost all values lie in
   [-1, 1], and column 18 (pitch) stays within [-1.34, 3.10] (``describe_session`` reports both).
5. Neural features look already normalized (roughly zero mean, unit SD per channel), as the Dryad
   page says; their code applies no normalization (``describe_session`` reports the moments).
6. The ``session`` string inside the file matches the folder name (we warn otherwise).
7. Session -> input layer: day195 is not in their ``training_data_args.yaml``; its header comment
   assigns it ``session_layer 20`` (same week as day193). The pretrained ``.h5`` has one layer baked
   in, so this index is informational for inference.
8. Channel order: data are in the *current* ordering; the pretrained day025-day195 models need
   their ``reorder_channels`` (done in ``wvx.inference``, controlled by config). Confirm on real
   data that predictions with reordering beat predictions without.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import scipy.io

from wvx import b2v
from wvx.session import SessionData, Trial

PathLike = Union[str, Path]

_TARGET_MODES = ("required", "optional", "skip")


def layer_for_session(session: str, cfg: Mapping[str, Any]) -> int:
    """Input-layer index of ``session``: their ``training_data_args.yaml``, else the held-out map.

    Their YAML maps 36 training sessions onto 21 week-level layers (read with their
    ``dataloader.get_sessions``). The held-out t15_day195 is only named in a comment there, so its
    layer comes from ``cfg["wairagkar_data"]["held_out_session_layers"]``.
    """
    dataloader = b2v.load("lib.dataloader")
    sessions, layers = dataloader.get_sessions(str(cfg["paths"]["b2v_session_layers"]))
    if session in sessions:
        return int(layers[sessions.index(session)])
    held_out = cfg["wairagkar_data"].get("held_out_session_layers") or {}
    if session in held_out:
        return int(held_out[session])
    raise KeyError(f"no input layer known for session {session!r}")


def session_files(data_dir: PathLike, session: str, cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    """Paths of the neural and target ``.mat`` files of one session (dataloader.py:58-65)."""
    dcfg = cfg["wairagkar_data"]
    sess_dir = Path(data_dir) / session
    return (
        sess_dir / dcfg["neural_file"].format(session=session),
        sess_dir / dcfg["target_file"].format(session=session),
    )


def inspect_session_files(data_dir: PathLike, session: str, cfg: Mapping[str, Any]) -> dict:
    """Variable names, shapes and MATLAB classes of the session files, without loading the data."""
    out: dict = {}
    for kind, path in zip(("neural", "target"), session_files(data_dir, session, cfg)):
        if path.is_file():
            out[kind] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "variables": [
                    {"name": n, "shape": list(s), "class": c} for n, s, c in scipy.io.whosmat(str(path))
                ],
            }
        else:
            out[kind] = {"file": path.name, "missing": True}
    return out


def _per_trial(value: Any, what: str) -> list:
    """Split one ``loadmat(..., squeeze_me=True)`` value into per-trial items.

    A MATLAB cell array comes back as an object array (one element per trial). A char matrix of
    cues comes back as a string array padded with spaces. ``squeeze_me`` turns a 1x1 cell into its
    content, so a lone 2-D array or string is one trial (their loader would iterate its rows).
    A 3-D numeric array is taken as trials stacked on axis 0.
    """
    if isinstance(value, str):
        return [value]
    arr = np.asarray(value)
    if arr.dtype == object:
        return list(arr.ravel())
    if arr.dtype.kind in "US":
        return [str(s) for s in arr.ravel()]
    if arr.ndim == 3:
        return list(arr)
    if arr.ndim == 2:
        return [arr]
    raise ValueError(f"{what}: cannot split value of shape {arr.shape} and dtype {arr.dtype} into trials")


@lru_cache(maxsize=None)
def _lpc_norm(path: str) -> tuple[np.ndarray, np.ndarray]:
    return b2v.load("lib.data_utils").load_lpc_norm(path)


def lpc_norm(cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Their per-feature LPCNet (min, max), float32 arrays of length 20 (data_utils.load_lpc_norm)."""
    return _lpc_norm(str(cfg["paths"]["lpc_norm"]))


def _read_targets(target_path: Path, n_trials: int, cfg: Mapping[str, Any]) -> list[np.ndarray]:
    """Per-trial target arrays exactly as stored (dataloader.py:62-70), one per neural trial."""
    key = cfg["wairagkar_data"]["keys"]["target"]
    targets = [np.asarray(y) for y in _per_trial(scipy.io.loadmat(str(target_path), squeeze_me=True)[key], key)]
    if len(targets) != n_trials:
        raise ValueError(f"{target_path.name}: {len(targets)} target trials, {n_trials} neural trials")
    return targets


def _check_target(y: np.ndarray, n_bins: int, where: str, cfg: Mapping[str, Any]) -> None:
    n_targets = int(cfg["model"]["n_targets"])
    if y.ndim != 2 or y.shape[0] != n_bins or y.shape[1] < n_targets:
        # Their pairing Y[-(T-59):] (data_utils.py:200) would silently misalign this.
        raise ValueError(f"{where}: target shape {y.shape}, want ({n_bins}, >= {n_targets})")


def _reduce(y: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Stored target -> (T, 20) cepstra + pitch in LPCNet's range (their dimensionality reduction)."""
    sp = b2v.load("lib.speech_processor")
    n_ceps = int(cfg["wairagkar_data"]["n_cepstral_coeff"])
    return sp.lpcnet_features_dimensionality_reduction(y, n_cepstral_coeff=n_ceps).astype(np.float32)


def load_session(data_dir: PathLike, session: str, layer: int, cfg: Mapping[str, Any]) -> SessionData:
    """Load ``<data_dir>/<session>/`` into SessionData (PROVISIONAL, see module docstring).

    ``Trial.neural`` is ``[spikepow | threshcross]`` (T, 512) float32 in the release's channel
    order (dataloader.py:74). ``Trial.target`` is the decoder's training target (T, 20) float32 from
    their ``get_speech_features_from_single_trial`` (cepstra+pitch, scaled to [-1, 1], smoothed as
    configured). ``Trial.meta`` (JSON-safe) records source dtypes, cue key and target column count.
    The stored, unsmoothed target features are available from ``load_raw_targets``.
    """
    dcfg = cfg["wairagkar_data"]
    keys = dcfg["keys"]
    mode = dcfg["targets"]
    if mode not in _TARGET_MODES:
        raise ValueError(f"wairagkar_data.targets must be one of {_TARGET_MODES}, got {mode!r}")
    neural_path, target_path = session_files(data_dir, session, cfg)

    # Same call as theirs: scipy.io.loadmat(path, squeeze_me=True) (dataloader.py:58-61).
    neur = scipy.io.loadmat(str(neural_path), squeeze_me=True)
    spikepow = _per_trial(neur[keys["spikepow"]], keys["spikepow"])
    threshcross = _per_trial(neur[keys["threshcross"]], keys["threshcross"])
    cue_key = keys["cues"] if keys["cues"] in neur else keys["cues_fallback"]
    cues = _per_trial(neur[cue_key], cue_key)
    n = len(spikepow)
    if not (len(threshcross) == len(cues) == n):
        raise ValueError(
            f"{neural_path.name}: {n} spikepow, {len(threshcross)} threshcross, {len(cues)} cue trials"
        )
    mat_session = neur.get("session")
    if mat_session is not None and str(mat_session).strip() != session:
        warnings.warn(f"{neural_path.name}: 'session' is {str(mat_session)!r}, folder is {session!r}")

    targets = None
    if mode != "skip":
        if target_path.is_file():
            targets = _read_targets(target_path, n, cfg)
        elif mode == "required":
            raise FileNotFoundError(f"target file missing: {target_path}")
    if targets is not None:
        data_utils = b2v.load("lib.data_utils")
        lpc_min, lpc_max = lpc_norm(cfg)

    trials = []
    for i in range(n):
        sp, tc = np.asarray(spikepow[i]), np.asarray(threshcross[i])
        if sp.ndim != 2 or sp.shape != tc.shape:
            raise ValueError(f"{session} trial {i}: spikepow {sp.shape} vs threshcross {tc.shape}")
        meta: dict = {"trial_index": i, "neural_dtype": str(sp.dtype), "cue_key": cue_key}
        target = None
        if targets is not None:
            y = targets[i]
            _check_target(y, sp.shape[0], f"{session} trial {i}", cfg)
            meta["target_dtype"] = str(y.dtype)
            meta["target_columns"] = int(y.shape[1])
            target = data_utils.get_speech_features_from_single_trial(
                y,
                lpc_min,
                lpc_max,
                n_cepstral_coeff=int(dcfg["n_cepstral_coeff"]),
                smoothing=bool(dcfg["target_smoothing"]),
                smoothing_sigma=float(dcfg["target_smoothing_sigma"]),
            ).astype(np.float32)
        trials.append(
            Trial(
                neural=np.concatenate([sp, tc], axis=1).astype(np.float32),
                target=target,
                text=str(cues[i]).strip(),
                meta=meta,
            )
        )

    sess = SessionData(name=session, layer=int(layer), trials=trials)
    sess.validate(n_features=int(cfg["model"]["n_features"]), n_targets=int(cfg["model"]["n_targets"]))
    return sess


def load_raw_targets(data_dir: PathLike, session: str, cfg: Mapping[str, Any]) -> list[np.ndarray]:
    """Stored target features per trial as (T, 20) float32 in LPCNet's range: no scaling, no smoothing.

    For resynthesizing the target audio. Same file, keys and checks as ``load_session``.
    """
    neural_path, target_path = session_files(data_dir, session, cfg)
    if not target_path.is_file():
        raise FileNotFoundError(f"target file missing: {target_path}")
    key = cfg["wairagkar_data"]["keys"]["spikepow"]
    spikepow = scipy.io.loadmat(str(neural_path), squeeze_me=True, variable_names=[key])[key]
    n_bins = [np.asarray(x).shape[0] for x in _per_trial(spikepow, key)]
    targets = _read_targets(target_path, len(n_bins), cfg)
    for i, (y, t) in enumerate(zip(targets, n_bins)):
        _check_target(y, t, f"{session} trial {i}", cfg)
    return [_reduce(y, cfg) for y in targets]


def describe_session(
    sess: SessionData, cfg: Mapping[str, Any], raw_targets: Optional[list[np.ndarray]] = None
) -> dict:
    """Numbers to check the PROVISIONAL format assumptions against real data (module docstring)."""
    n_bins = np.array([t.neural.shape[0] for t in sess.trials])
    neural = np.concatenate([t.neural for t in sess.trials], axis=0)
    half = neural.shape[1] // 2
    out: dict = {
        "session": sess.name,
        "layer": sess.layer,
        "n_trials": len(sess.trials),
        "n_bins": {"min": int(n_bins.min()), "median": float(np.median(n_bins)), "max": int(n_bins.max())},
        "neural_dtypes": sorted({t.meta.get("neural_dtype", "?") for t in sess.trials}),
        "cue_keys": sorted({t.meta.get("cue_key", "?") for t in sess.trials}),
        "neural_moments": {
            block: {
                "mean_of_channel_means": float(x.mean(axis=0).mean()),
                "mean_of_channel_sds": float(x.std(axis=0).mean()),
                "min": float(x.min()),
                "max": float(x.max()),
            }
            for block, x in (("spikepow", neural[:, :half]), ("threshcross", neural[:, half:]))
        },
    }
    with_target = [t for t in sess.trials if t.target is not None]
    if with_target:
        norm = np.concatenate([t.target for t in with_target], axis=0)
        out["targets"] = {
            "n_trials": len(with_target),
            "dtypes": sorted({t.meta["target_dtype"] for t in with_target}),
            "columns": sorted({t.meta["target_columns"] for t in with_target}),
            "training_target_fraction_outside_unit_range": float(np.mean(np.abs(norm) > 1)),
        }
    if raw_targets:
        data_utils = b2v.load("lib.data_utils")
        lpc_min, lpc_max = lpc_norm(cfg)
        raw = np.concatenate(raw_targets, axis=0)
        scaled = data_utils.scale_between_minus_one_to_one(raw, lpc_min, lpc_max)
        out["raw_targets"] = {
            "min": raw.min(axis=0).tolist(),
            "max": raw.max(axis=0).tolist(),
            "fraction_outside_unit_range_after_scaling": float(np.mean(np.abs(scaled) > 1)),
        }
    return out
