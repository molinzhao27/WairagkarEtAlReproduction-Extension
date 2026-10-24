"""D-078 post-processing: denoiser, pitch shift, feature sharpening."""

import shutil

import numpy as np
import pytest

from wvx import postproc as pp


def test_spectral_gate_cuts_noise_keeps_tone():
    sr = 16000
    t = np.arange(sr * 2) / sr
    tone = np.where((t > 0.5) & (t < 1.5), 0.5 * np.sin(2 * np.pi * 440 * t), 0.0)
    noise = np.random.default_rng(0).normal(0, 0.02, len(t))
    y = pp.spectral_gate(tone + noise, sr)
    assert y.shape == t.shape
    quiet = (t < 0.4) | (t > 1.6)
    assert np.std(y[quiet]) < 0.5 * np.std(noise[quiet])                  # noise in the gaps cut
    mid = (t > 0.7) & (t < 1.3)
    assert abs(np.corrcoef(y[mid], tone[mid])[0, 1]) > 0.98               # tone kept


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_pitch_shift_raises_frequency_keeps_length():
    sr = 16000
    t = np.arange(sr) / sr
    y = pp.pitch_shift(0.5 * np.sin(2 * np.pi * 200 * t), sr, 2.0)
    assert len(y) == sr
    spec = np.abs(np.fft.rfft(y[2000:14000] * np.hanning(12000)))
    f = np.fft.rfftfreq(12000, 1 / sr)[np.argmax(spec)]
    assert abs(f - 200 * 2 ** (2 / 12)) < 6


def test_sharpen_scales_cepstra_only():
    rng = np.random.default_rng(1)
    pred = np.clip(rng.normal(0, 0.1, (50, 20)), -1, 1).astype(np.float32)
    out = pp.sharpen(pred, 1.5)
    assert np.allclose(out[:, 18:], pred[:, 18:])
    assert np.allclose(out[:, :18].mean(0), pred[:, :18].mean(0), atol=1e-6)
    assert np.allclose(out[:, :18].std(0), 1.5 * pred[:, :18].std(0), rtol=1e-4)
    assert out.min() >= -1 and out.max() <= 1
