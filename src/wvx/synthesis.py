"""LPCNet vocoding around their ``speech_processor`` conversions, with safe subprocess calls.

Decode direction (their ``inference_utils.synthesize_speech``):
normalized decoder output (T, 20) -> ``data_utils.scale_to_original_range`` -> their
``speech_processor.reconstruct_lpcnet_features`` (T, 36) -> ``lpcnet_demo -synthesis`` -> 16-bit PCM,
160 samples per frame at 16 kHz -> float waveform.

Encode direction (their unused ``speech_processor.encode_lpcnet_features_from_wav``):
wav -> librosa.load(native rate, mono) -> temp PCM_24 wav -> ffmpeg to 16 kHz s16le ->
``lpcnet_demo -features`` -> (n_frames, 36) float32.

Their functions that call the binaries are not used: their retry loops never give up
(``speech_processor.py:64,75,87``), they build unquoted shell strings, and they write fixed temp
file names into a shared folder. Instead every call here has a pre-flight check of the binary, a
timeout, an argument list (no shell), and its own fresh temp directory. The numeric conversions
are theirs.

Differences from their path that do not change the audio:
- We read LPCNet's raw PCM directly and divide by 32768, which is what their ffmpeg->wav->
  ``librosa.load`` round trip yields for 16-bit audio; ffmpeg is only needed for encoding.
- With the LPCNet we build (xiph/LPCNet 7dc9942, ``END2END`` model), ``-synthesis`` reads 36 floats
  per frame but uses only the first 20; LPC comes from the network. Their Python LPC
  reconstruction (floats 20-35) is still computed, for fidelity, but has no effect on this build
  (tested in tests/test_synthesis.py). Their own LPCNet build date is unknown.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import numpy as np

from wvx import b2v
from wvx.config import resolve_path
from wvx.wairagkar_data import lpc_norm

PathLike = Union[str, Path]

# Full scale of signed 16-bit PCM (soundfile/librosa convert int16 to float by dividing by this).
PCM16_FULL_SCALE = float(-np.iinfo(np.int16).min)


class LPCNetError(RuntimeError):
    """An LPCNet or ffmpeg call failed, timed out, or produced output of the wrong size."""


# ---------------------------------------------------------------------------------------- tools


def lpcnet_binary(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["paths"]["lpcnet"]) / cfg["synthesis"]["lpcnet_binary"]


def ffmpeg_binary(cfg: Mapping[str, Any]) -> Path:
    """Configured ffmpeg. A bare name is looked up next to the running Python first, then on PATH."""
    name = str(cfg["synthesis"]["ffmpeg"])
    if os.sep in name or (os.altsep and os.altsep in name):
        return resolve_path(name)
    search = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    found = shutil.which(name, path=search)
    if found is None:
        raise LPCNetError(f"ffmpeg {name!r} not found next to {sys.executable} or on PATH")
    return Path(found)


def _env(cfg: Mapping[str, Any]) -> dict:
    env = os.environ.copy()
    if cfg["synthesis"]["strip_ld_library_path"]:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def _run(cmd: Sequence[PathLike], cfg: Mapping[str, Any], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    args = [str(c) for c in cmd]
    try:
        res = subprocess.run(
            args, cwd=cwd, env=_env(cfg), stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise LPCNetError(f"timed out after {timeout} s: {' '.join(args)}") from e
    except OSError as e:
        raise LPCNetError(f"cannot run {args[0]}: {e}") from e
    if res.returncode != 0:
        err = res.stderr.decode(errors="replace").strip()
        raise LPCNetError(f"exit {res.returncode}: {' '.join(args)}\n{err}")
    return res


@lru_cache(maxsize=None)
def _preflight_lpcnet(path: str, timeout: float) -> None:
    p = Path(path)
    if not p.is_file() or not os.access(p, os.X_OK):
        raise LPCNetError(f"LPCNet binary missing or not executable: {p} (run scripts/build_lpcnet.sh)")
    # Without arguments lpcnet_demo prints its usage and exits 1: proves the binary runs here.
    try:
        res = subprocess.run([path], stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise LPCNetError(f"LPCNet binary does not run: {p}: {e}") from e
    if b"-synthesis" not in res.stderr:
        raise LPCNetError(f"{p} did not print the lpcnet_demo usage; wrong binary?")


@lru_cache(maxsize=None)
def _preflight_ffmpeg(path: str, timeout: float) -> None:
    try:
        res = subprocess.run([path, "-version"], stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise LPCNetError(f"ffmpeg does not run: {path}: {e}") from e
    if res.returncode != 0:
        raise LPCNetError(f"`{path} -version` exited {res.returncode}")


def preflight(cfg: Mapping[str, Any], ffmpeg: bool = False) -> None:
    """Fail fast if the binaries are missing or don't run (instead of their infinite retries)."""
    t = float(cfg["synthesis"]["preflight_timeout_s"])
    _preflight_lpcnet(str(lpcnet_binary(cfg)), t)
    if ffmpeg:
        _preflight_ffmpeg(str(ffmpeg_binary(cfg)), t)


@contextlib.contextmanager
def _workdir(cfg: Mapping[str, Any]) -> Iterator[Path]:
    """A fresh temp dir per call, so parallel processes never share their fixed file names."""
    root = cfg["synthesis"]["tmp_root"]
    if root is not None:
        root = resolve_path(root)
        root.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=f"wvx_lpcnet_{os.getpid()}_", dir=root))
    try:
        yield d
    finally:
        if not cfg["synthesis"]["keep_tmp"]:
            shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------ conversions


def denormalize(features: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """[-1, 1] decoder features -> LPCNet's range, with their ``scale_to_original_range``."""
    lpc_min, lpc_max = lpc_norm(cfg)
    return b2v.load("lib.data_utils").scale_to_original_range(features, lpc_min, lpc_max)


def frames_from_features(raw: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """(T, 20) or (T, 36) features in LPCNet's range -> (T, 36) float32 frames for ``-synthesis``.

    Keeps 18 cepstra + 2 pitch (their ``lpcnet_features_dimensionality_reduction``), then rebuilds
    the 16 LPC floats with their ``reconstruct_lpcnet_features``, exactly as their synthesis does.
    """
    sp = b2v.load("lib.speech_processor")
    n_targets = int(cfg["model"]["n_targets"])
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < n_targets or raw.shape[0] == 0:
        raise ValueError(f"features shape {raw.shape}, want (T > 0, >= {n_targets})")
    if not np.all(np.isfinite(raw)):
        raise ValueError("features contain NaN or inf")
    if raw.shape[1] != n_targets:
        raw = sp.lpcnet_features_dimensionality_reduction(raw)
    frames = sp.reconstruct_lpcnet_features(raw.copy())
    want = (raw.shape[0], int(cfg["synthesis"]["n_frame_floats"]))
    if frames.shape != want:
        raise ValueError(f"reconstructed frames {frames.shape}, want {want}")
    return frames


def synthesize_frames(frames: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """(T, 36) LPCNet frames -> float32 waveform of T * 160 samples at 16 kHz, in [-1, 1)."""
    scfg = cfg["synthesis"]
    n_floats = int(scfg["n_frame_floats"])
    frames = np.ascontiguousarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != n_floats or frames.shape[0] == 0:
        raise ValueError(f"frames shape {frames.shape}, want (T > 0, {n_floats})")
    if not np.all(np.isfinite(frames)):
        raise ValueError("frames contain NaN or inf")
    preflight(cfg)
    with _workdir(cfg) as wd:
        f_in, f_out = wd / "features.f32", wd / "synth.pcm"
        frames.tofile(f_in)  # native-endian float32, as their decode writes it
        _run([lpcnet_binary(cfg), "-synthesis", f_in, f_out], cfg, wd, float(scfg["timeout_s"]))
        pcm = np.fromfile(f_out, dtype=np.int16)
    want = frames.shape[0] * int(scfg["samples_per_frame"])
    if pcm.size != want:
        raise LPCNetError(f"LPCNet wrote {pcm.size} samples for {frames.shape[0]} frames, want {want}")
    return pcm.astype(np.float32) / PCM16_FULL_SCALE


def features_to_waveform(raw: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """(T, 20|36) features in LPCNet's range (e.g. a stored target) -> waveform."""
    return synthesize_frames(frames_from_features(raw, cfg), cfg)


def decoder_output_to_waveform(pred: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Normalized decoder output (T, 20) in [-1, 1] -> waveform (their synthesis path)."""
    return features_to_waveform(denormalize(pred, cfg), cfg)


def encode_wav(wav_path: PathLike, cfg: Mapping[str, Any]) -> np.ndarray:
    """Wav file -> (n_frames, 36) float32 LPCNet features, one frame per 10 ms.

    Same steps as their ``encode_lpcnet_features_from_wav``. We also pass ``-ac 1`` to ffmpeg; their
    command lacks it, but librosa already downmixed to mono, so the output is the same.
    """
    import librosa
    import soundfile

    scfg = cfg["synthesis"]
    preflight(cfg, ffmpeg=True)
    sig, fs = librosa.load(str(wav_path), sr=None, mono=True)  # speech_processor.py:53-54
    if sig.size == 0:
        raise ValueError(f"empty audio: {wav_path}")
    timeout = float(scfg["timeout_s"])
    with _workdir(cfg) as wd:
        tmp_wav, pcm, f32 = wd / "input.wav", wd / "input.pcm", wd / "features.f32"
        soundfile.write(str(tmp_wav), sig, int(fs), subtype=scfg["encode_tmp_wav_subtype"])
        _run(
            [ffmpeg_binary(cfg), "-nostdin", "-y", "-loglevel", "error", "-i", tmp_wav,
             "-f", "s16le", "-ar", str(int(scfg["sample_rate"])), "-ac", "1", "-acodec", "pcm_s16le", pcm],
            cfg, wd, timeout,
        )
        _run([lpcnet_binary(cfg), "-features", pcm, f32], cfg, wd, timeout)
        flat = np.fromfile(f32, dtype=np.float32)
    n_floats = int(scfg["n_frame_floats"])
    if flat.size % n_floats:
        raise LPCNetError(f"LPCNet wrote {flat.size} floats, not a multiple of {n_floats}")
    return flat.reshape(-1, n_floats)


def write_wav(path: PathLike, wav: np.ndarray, cfg: Mapping[str, Any], sr: Optional[int] = None) -> Path:
    import soundfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = int(sr if sr is not None else cfg["synthesis"]["sample_rate"])
    soundfile.write(str(path), wav, rate, subtype=cfg["synthesis"]["wav_subtype"])
    return path
