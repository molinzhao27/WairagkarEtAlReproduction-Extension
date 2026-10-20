"""wvx.synthesis: their conversions, safe subprocess handling, and a real LPCNet round trip."""

import os
import re
import stat

import numpy as np
import pytest

from wvx import b2v
from wvx.config import REPO_ROOT, deep_merge, load_config

pytestmark = pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py")

EXPERIMENT = "configs/experiments/reproduce_inference.yaml"
LPCNET_H = REPO_ROOT / "third_party" / "LPCNet" / "include" / "lpcnet.h"


@pytest.fixture
def cfg(tmp_path):
    return load_config(EXPERIMENT, overrides={"synthesis": {"tmp_root": str(tmp_path / "tmp")}})


def _lpcnet_runs(cfg) -> bool:
    from wvx import synthesis

    try:
        synthesis.preflight(cfg)
    except synthesis.LPCNetError:
        return False
    return True


def _ffmpeg_runs(cfg) -> bool:
    from wvx import synthesis

    try:
        synthesis.preflight(cfg, ffmpeg=True)
    except synthesis.LPCNetError:
        return False
    return True


def _norm(cfg):
    return b2v.load("lib.data_utils").load_lpc_norm(str(cfg["paths"]["lpc_norm"]))


def _realistic_features(cfg, n_frames, seed=0):
    """Raw-range (T, 20) features: smooth random walks inside their JSON min/max."""
    lo, hi = _norm(cfg)
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(scale=0.1, size=(n_frames, lo.size)), axis=0)
    u = (np.tanh(walk) + 1) / 2
    return (lo + u * (hi - lo)).astype(np.float32)


def _tone_wav(path, fs, f0, seconds):
    import soundfile

    t = np.arange(int(fs * seconds)) / fs
    harmonics = [k for k in range(1, 40) if k * f0 < 0.45 * fs]
    sig = sum((0.3 / k) * np.sin(2 * np.pi * k * f0 * t) for k in harmonics)
    sig *= np.hanning(sig.size) ** 0.25
    soundfile.write(str(path), sig.astype(np.float32), fs, subtype="PCM_16")


# ------------------------------------------------------------------------- config vs LPCNet


@pytest.mark.skipif(not LPCNET_H.is_file(), reason="LPCNet sources absent")
def test_config_constants_match_lpcnet_header(cfg):
    h = LPCNET_H.read_text()
    define = {m.group(1): int(m.group(2)) for m in re.finditer(r"#define (\w+) \(?(\d+)\)?", h)}
    assert cfg["synthesis"]["n_frame_floats"] == define["NB_TOTAL_FEATURES"]
    assert cfg["model"]["n_targets"] == define["NB_FEATURES"]
    assert cfg["synthesis"]["samples_per_frame"] == define["LPCNET_FRAME_SIZE"]
    # 10 ms frames at 16 kHz, matching the decoder's 10 ms bins.
    assert cfg["synthesis"]["sample_rate"] // define["LPCNET_FRAME_SIZE"] == 100


def test_json_pitch_range_is_lpcnet_encoder_clamp(cfg):
    # Encoder: features[18] = .01 * (clamp(2*period, 66, 510) - 200)  (LPCNet src/lpcnet_enc.c).
    lo, hi = _norm(cfg)
    assert lo[18] == pytest.approx(0.01 * (66 - 200), abs=1e-6)
    assert hi[18] == pytest.approx(0.01 * (510 - 200), abs=1e-6)


# -------------------------------------------------------------------------- conversions


def test_denormalize_inverts_their_scaling(cfg):
    from wvx import synthesis

    lo, hi = _norm(cfg)
    raw = _realistic_features(cfg, 30)
    norm = b2v.load("lib.data_utils").scale_between_minus_one_to_one(raw, lo, hi)
    np.testing.assert_allclose(synthesis.denormalize(norm, cfg), raw, rtol=1e-5, atol=1e-5)


def test_frames_are_their_reconstruction(cfg):
    from wvx import synthesis

    sp = b2v.load("lib.speech_processor")
    raw = _realistic_features(cfg, 12)
    frames = synthesis.frames_from_features(raw, cfg)
    np.testing.assert_array_equal(frames, sp.reconstruct_lpcnet_features(raw.copy()))
    assert frames.shape == (12, 36) and frames.dtype == np.float32
    # A 36-wide input (e.g. a stored target) is reduced to cepstra + pitch first.
    wide = np.concatenate([raw, np.ones((12, 16), np.float32)], axis=1)
    np.testing.assert_array_equal(synthesis.frames_from_features(wide, cfg), frames)


@pytest.mark.parametrize("bad", [np.zeros((0, 20)), np.zeros((5, 19)), np.full((5, 20), np.nan)])
def test_bad_features_rejected(cfg, bad):
    from wvx import synthesis

    with pytest.raises(ValueError):
        synthesis.frames_from_features(bad, cfg)


# --------------------------------------------------------- subprocess safety (fake binary)


def _fake_lpcnet(tmp_path, body):
    """A stand-in lpcnet_demo: prints the usage without args (pre-flight), else runs `body`."""
    p = tmp_path / "fake_lpcnet"
    p.write_text(
        "#!/bin/sh\n"
        'if [ "$#" -eq 0 ]; then echo "usage: lpcnet_demo -synthesis <f> <o>" >&2; exit 1; fi\n' + body + "\n"
    )
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def _cfg_with_binary(cfg, binary, **synth):
    return deep_merge(cfg, {"paths": {"lpcnet": binary.parent}, "synthesis": {"lpcnet_binary": binary.name, **synth}})


def test_missing_binary_fails_fast(cfg, tmp_path):
    from wvx import synthesis

    c = deep_merge(cfg, {"paths": {"lpcnet": tmp_path}, "synthesis": {"lpcnet_binary": "nope"}})
    with pytest.raises(synthesis.LPCNetError, match="missing"):
        synthesis.synthesize_frames(np.zeros((3, 36), np.float32), c)


def test_wrong_binary_fails_preflight(cfg, tmp_path):
    from wvx import synthesis

    p = tmp_path / "not_lpcnet"
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(synthesis.LPCNetError, match="usage"):
        synthesis.synthesize_frames(np.zeros((3, 36), np.float32), _cfg_with_binary(cfg, p))


def test_hanging_binary_times_out(cfg, tmp_path):
    from wvx import synthesis

    c = _cfg_with_binary(cfg, _fake_lpcnet(tmp_path, "exec sleep 10"), timeout_s=0.5)
    with pytest.raises(synthesis.LPCNetError, match="timed out"):
        synthesis.synthesize_frames(np.zeros((3, 36), np.float32), c)


def test_failing_binary_raises_with_stderr(cfg, tmp_path):
    from wvx import synthesis

    c = _cfg_with_binary(cfg, _fake_lpcnet(tmp_path, "echo boom >&2; exit 3"))
    with pytest.raises(synthesis.LPCNetError, match="boom"):
        synthesis.synthesize_frames(np.zeros((3, 36), np.float32), c)


def test_short_output_is_an_error_and_temp_dirs_are_removed(cfg, tmp_path):
    from wvx import synthesis

    # Writes 10 bytes (5 samples) instead of 3 frames x 160 samples.
    c = _cfg_with_binary(cfg, _fake_lpcnet(tmp_path, 'printf "0123456789" > "$3"'))
    with pytest.raises(synthesis.LPCNetError, match="samples"):
        synthesis.synthesize_frames(np.zeros((3, 36), np.float32), c)
    assert os.listdir(cfg["synthesis"]["tmp_root"]) == []


# ------------------------------------------------------------------------- real LPCNet


@pytest.fixture
def real(cfg):
    if not _lpcnet_runs(cfg):
        pytest.skip("LPCNet binary not built (scripts/build_lpcnet.sh)")
    return cfg


def test_synthesis_length_range_and_determinism(real):
    from wvx import synthesis

    raw = _realistic_features(real, 50)
    wav = synthesis.features_to_waveform(raw, real)
    assert wav.dtype == np.float32 and wav.shape == (50 * 160,)
    assert np.all(np.isfinite(wav)) and np.abs(wav).max() <= 1.0
    np.testing.assert_array_equal(wav, synthesis.features_to_waveform(raw, real))


def test_this_lpcnet_build_ignores_the_lpc_floats(real):
    # -synthesis reads 36 floats per frame but uses only the first 20 (END2END model): their
    # Python LPC reconstruction does not change the audio with this build.
    from wvx import synthesis

    frames = synthesis.frames_from_features(_realistic_features(real, 40, seed=1), real)
    zeroed = frames.copy()
    zeroed[:, 20:] = 0
    np.testing.assert_array_equal(synthesis.synthesize_frames(frames, real), synthesis.synthesize_frames(zeroed, real))


def test_round_trip_wav_features_wav(real, tmp_path):
    from wvx import synthesis

    if not _ffmpeg_runs(real):
        pytest.skip("ffmpeg not available")
    f0, seconds = 150.0, 1.0
    for fs in (16000, 22050):  # 22.05 kHz exercises the ffmpeg resampling to 16 kHz
        _tone_wav(tmp_path / f"tone{fs}.wav", fs, f0, seconds)
        feats = synthesis.encode_wav(tmp_path / f"tone{fs}.wav", real)
        assert feats.shape == (int(seconds * 100), 36) and feats.dtype == np.float32
        mid = feats[feats.shape[0] // 4 : 3 * feats.shape[0] // 4]
        # Pitch feature of a 150 Hz tone: .01 * (2 * 16000 / 150 - 200) = 0.133 (integer periods).
        assert np.median(mid[:, 18]) == pytest.approx(0.01 * (2 * 16000 / f0 - 200), abs=0.02)

        lo, hi = _norm(real)
        norm = b2v.load("lib.data_utils").scale_between_minus_one_to_one(feats[:, :20], lo, hi)
        wav = synthesis.decoder_output_to_waveform(norm, real)
        assert wav.shape == (feats.shape[0] * 160,)
        synthesis.write_wav(tmp_path / "resynth.wav", wav, real)
        again = synthesis.encode_wav(tmp_path / "resynth.wav", real)
        mid2 = again[again.shape[0] // 4 : 3 * again.shape[0] // 4]
        # LPCNet resynthesis keeps the pitch (within ~2 samples of period) and the energy contour.
        assert abs(np.median(mid2[:, 18]) - np.median(mid[:, 18])) < 0.05
        assert np.corrcoef(feats[:, 0], again[:, 0])[0, 1] > 0.5
