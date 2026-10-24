"""Post-processing of decoded voice (D-078): spectral-gating denoiser, pitch shift, feature sharpening.

Every function is applied uniformly to every clip (no per-clip choice). ``spectral_gate`` follows the usual
stationary spectral-gating recipe (noise profile from the quietest frames, threshold in dB, smoothed soft mask);
``pitch_shift`` changes pitch with tempo kept (ffmpeg asetrate + aresample + atempo); ``sharpen`` counters the
over-smoothing of an MSE-trained decoder by scaling each cepstral feature's deviation from its trial mean.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def spectral_gate(wav: np.ndarray, sr: int, n_fft: int = 512, hop: int = 128, quiet_fraction: float = 0.2,
                  n_std: float = 1.5, reduction: float = 0.8, smooth_bins: int = 2, smooth_frames: int = 2) -> np.ndarray:
    """Denoise ``wav``: per frequency, the dB magnitude's mean and s.d. over the ``quiet_fraction`` lowest-energy STFT
    frames give the noise profile; bins above mean + ``n_std`` s.d. are kept, the rest attenuated by ``reduction``,
    through a mask smoothed over ±``smooth_bins`` bins and ±``smooth_frames`` frames."""
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    x = np.asarray(wav, np.float64)
    _, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
    mag = np.abs(Z)
    db = 20 * np.log10(mag + 1e-10)
    energy = (mag ** 2).sum(0)
    quiet = energy <= np.quantile(energy, quiet_fraction)
    if quiet.sum() < 2:
        return np.asarray(wav, np.float32)
    thr = db[:, quiet].mean(1, keepdims=True) + n_std * db[:, quiet].std(1, keepdims=True)
    mask = uniform_filter((db > thr).astype(np.float64), size=(2 * smooth_bins + 1, 2 * smooth_frames + 1))
    gain = mask + (1.0 - mask) * (1.0 - reduction)
    _, y = istft(Z * gain, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    return np.asarray(y[: len(x)], np.float32)


def pitch_shift(wav: np.ndarray, sr: int, semitones: float, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Shift pitch by ``semitones`` with the duration kept: resample-based shift (asetrate) back to ``sr``, then
    atempo by the inverse factor."""
    import soundfile as sf

    if semitones == 0:
        return np.asarray(wav, np.float32)
    f = 2.0 ** (semitones / 12.0)
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(src), np.asarray(wav, np.float32), sr, subtype="PCM_16")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-filter:a",
                        f"asetrate={sr * f:.3f},aresample={sr},atempo={1.0 / f:.6f}", str(dst)], check=True)
        y, _ = sf.read(str(dst), dtype="float32")
    n = len(wav)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def sharpen(pred: np.ndarray, alpha: float, n_cepstral: int = 18, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Scale the first ``n_cepstral`` features' deviations from their trial means by ``alpha`` (pitch features
    untouched), clipped to the decoder's output range [``lo``, ``hi``]."""
    out = np.array(pred, np.float32, copy=True)
    c = out[:, :n_cepstral]
    mu = c.mean(0, keepdims=True)
    out[:, :n_cepstral] = np.clip(mu + alpha * (c - mu), lo, hi)
    return out
