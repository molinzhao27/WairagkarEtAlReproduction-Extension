"""Offline inference with their brain-to-voice ``.h5`` inference models (their inference.ipynb).

Per trial, as their ``inference_utils.run_inference_single_trial`` does: optional
``reorder_channels`` on each 256-wide feature block, ``[spikepow | threshcross]`` (T, 512),
60-bin windows with stride 1 from their ``data_utils.get_sliding_windows``, one prediction per
window. Output frame k belongs to bin ``k + window_len - 1`` (the window's last bin), the same
pairing their training uses (``Y[-(T-59):]``, data_utils.py:200), so a trial of T bins gives
T - 59 frames and the first 590 ms have none.

Differences from their function, none of which changes the numbers (tested against it):
- Returns the normalized [-1, 1] output; ``wvx.synthesis.denormalize`` applies their
  ``scale_to_original_range`` before vocoding.
- Windows are reshaped to (n, 60, 512) instead of ``np.squeeze``, which breaks when T == 60.
- Prediction runs in fixed-size batches (last one zero-padded, padding discarded) so memory is
  bounded and the graph is traced once; the model has no cross-sample ops.

Import ``wvx.b2v`` before TensorFlow: the model is loaded with their ``load_b2voice_model``, which
needs Keras 2 (``tf_keras``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from wvx import b2v
from wvx.session import SessionData

PathLike = Union[str, Path]


def load_model(path: PathLike, cfg: Optional[Mapping[str, Any]] = None):
    """Load an ``.h5`` inference model with their ``inference_utils.load_b2voice_model``.

    With ``cfg``, also check that its input is (window_len, n_features) and its output n_targets.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model not found: {path}")
    inference_utils = b2v.load("lib.inference_utils")  # sets up Keras 2 before TensorFlow loads
    model = inference_utils.load_b2voice_model(str(path))
    if cfg is not None:
        check_model(model, cfg)
    return model


def check_model(model, cfg: Mapping[str, Any]) -> None:
    m = cfg["model"]
    want_in = (int(m["window_len"]), int(m["n_features"]))
    want_out = int(m["n_targets"])
    got_in, got_out = tuple(model.input_shape[1:]), model.output_shape[-1]
    if got_in != want_in or got_out != want_out:
        raise ValueError(f"model maps {got_in} -> {got_out}, config expects {want_in} -> {want_out}")


def reorder_channels(neural: np.ndarray, mapping_path: PathLike) -> np.ndarray:
    """Their ``reorder_channels`` applied to each feature block separately, as their notebook does.

    Their function indexes columns with a 256-long permutation, so it must see one 256-wide block at
    a time: given the 512-wide array it would silently return 256 permuted spike-power columns. We
    split, reorder each block, and concatenate again (inference.ipynb cell 7; concat at
    inference_utils.py:103). Net effect per block: ``out[:, j] = in[:, m[m[j]]]``.
    """
    neural = np.asarray(neural)
    if neural.ndim != 2 or neural.shape[1] % 2:
        raise ValueError(f"neural shape {neural.shape}, want (T, 2 * n_channels)")
    half = neural.shape[1] // 2
    with open(mapping_path) as f:
        n_map = len(json.load(f)["electrode_mapping"])
    if n_map != half:
        raise ValueError(f"electrode mapping has {n_map} entries, each feature block has {half} columns")
    data_utils = b2v.load("lib.data_utils")
    spike_pow = data_utils.reorder_channels(neural[:, :half].copy(), str(mapping_path))
    thresh_cross = data_utils.reorder_channels(neural[:, half:].copy(), str(mapping_path))
    return np.concatenate([spike_pow, thresh_cross], axis=1)


def sliding_windows(neural: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """(T, F) -> (n_windows, window_len, F) with their ``get_sliding_windows``."""
    w, s = int(cfg["model"]["window_len"]), int(cfg["model"]["step_size"])
    if neural.ndim != 2 or neural.shape[0] < w:
        raise ValueError(f"trial of shape {neural.shape} is shorter than one {w}-bin window")
    win = b2v.load("lib.data_utils").get_sliding_windows(neural, window_len=w, step_size=s)
    return win.reshape(-1, w, neural.shape[1])


def align_to_predictions(frames: np.ndarray, n_predictions: int) -> np.ndarray:
    """The rows of a per-bin array (e.g. a target) that predictions refer to (data_utils.py:200).

    Their rule ``Y[-n:]``; it is the window-end pairing only for step_size 1 (their setting).
    """
    if not 0 < n_predictions <= len(frames):
        raise ValueError(f"cannot align {n_predictions} predictions to {len(frames)} frames")
    return frames[-n_predictions:]


def predict_trial(
    model, neural: np.ndarray, cfg: Mapping[str, Any], reorder: Optional[bool] = None
) -> np.ndarray:
    """(T, 512) neural features of one trial -> (T - window_len + 1, 20) normalized predictions.

    ``reorder`` defaults to ``cfg["inference"]["reorder_channels"]``.
    """
    if reorder is None:
        reorder = bool(cfg["inference"]["reorder_channels"])
    neural = np.asarray(neural, dtype=np.float32)
    if reorder:
        neural = reorder_channels(neural, cfg["paths"]["electrode_mapping"])
    win = sliding_windows(neural, cfg)
    batch = int(cfg["inference"]["batch_size"])
    out = []
    for start in range(0, win.shape[0], batch):
        chunk = np.ascontiguousarray(win[start : start + batch], dtype=np.float32)
        n = chunk.shape[0]
        if n < batch:
            chunk = np.concatenate([chunk, np.zeros((batch - n,) + chunk.shape[1:], np.float32)])
        out.append(np.asarray(model.predict_on_batch(chunk))[:n])
    return np.concatenate(out, axis=0).astype(np.float32)


def predict_session(model, sess: SessionData, cfg: Mapping[str, Any]) -> list[np.ndarray]:
    return [predict_trial(model, t.neural, cfg) for t in sess.trials]
