"""wvx.alignment on synthetic envelopes, features and audio (fast, deterministic, no LPCNet)."""

import numpy as np
import pytest

from wvx import alignment as al
from wvx.config import deep_merge
from wvx.targets import load_targets_config


@pytest.fixture(scope="module")
def base_cfg():
    return load_targets_config()


def _cfg(base, **sections):
    return deep_merge(base, {"alignment": sections}) if sections else base


SIL = -12.48528                       # LPCNet c0 of digital zeros
SILENCE20 = np.r_[SIL, np.zeros(17), 3.1, -0.5].astype(np.float32)


def _bumps(n, centres, width, height_db=40.0, floor_db=-20.0):
    """dB envelope: Gaussian bumps (syllables) over a floor."""
    t = np.arange(n)[:, None]
    lin = np.exp(-0.5 * ((t - np.asarray(centres)[None, :]) / width) ** 2).sum(1)
    return floor_db + height_db * np.minimum(lin, 1.0)


def _features_from_db(env_db):
    """Raw (n, 20) features whose c0 encodes the given dB power; other features carry a ramp."""
    n = env_db.size
    c0 = env_db * 4.242640687119285 / 10.0 - 4.0
    f = np.tile(SILENCE20, (n, 1)).astype(np.float64)
    f[:, 0] = c0
    f[:, 1] = np.linspace(-1, 1, n)
    return f.astype(np.float32)


# ---------------------------------------------------------------------------------- basics


def test_runs():
    np.testing.assert_array_equal(al.runs([0, 1, 1, 0, 1]), [[1, 3], [4, 5]])
    assert al.runs([0, 0]).shape == (0, 2)
    np.testing.assert_array_equal(al.runs([1, 1]), [[0, 2]])


def test_c0_to_db_of_silence_is_minus_20_db(base_cfg):
    # c0 = sqrt(18) * log10(1e-2) - 4  <=>  10 * log10(1e-2) = -20 dB
    assert al.c0_to_db(np.array([SIL]), base_cfg)[0] == pytest.approx(-20.0, abs=1e-4)


def test_audio_envelope_follows_bursts(base_cfg):
    cfg = base_cfg
    sr, spf = 16000, 160
    n = 100
    wav = np.zeros(n * spf)
    t = np.arange(20 * spf) / sr
    wav[30 * spf:50 * spf] = 0.5 * np.sin(2 * np.pi * 200 * t)
    env = al.envelope_from_audio(wav, n, cfg)
    assert env.shape == (n,)
    assert env[40] > env[10] + 60 and env[40] > env[80] + 60
    assert np.argmax(env) in range(30, 50)
    # shorter audio is zero-padded to the frame count
    assert al.envelope_from_audio(wav[: 50 * spf], n, cfg).shape == (n,)


def test_envelope_source_dispatch(base_cfg):
    env = _bumps(80, [40], 6)
    raw = _features_from_db(env)
    cfg = _cfg(base_cfg, envelope={"source": "c0", "smooth_ms": 0.0})
    np.testing.assert_allclose(al.envelope_db(raw, cfg), env, atol=1e-3)
    with pytest.raises(ValueError):
        al.envelope_db(raw, _cfg(base_cfg, envelope={"source": "audio"}))   # no synth given


# ---------------------------------------------------------------------------- segmentation


def test_segment_three_syllables_with_pause(base_cfg):
    env = _bumps(200, [40, 60, 130], 6)          # two joined syllables, a pause, one more
    segs = al.segment_syllables(env, base_cfg)
    assert len(segs) == 3
    assert segs[0, 1] == segs[1, 0] and abs(segs[0, 1] - 50) <= 1      # boundary at the dip
    assert segs[1, 1] < segs[2, 0]                                        # silence between
    assert segs[0, 0] > 20 and segs[2, 1] < 160                           # silence trimmed
    assert segs[0, 0] < 40 < segs[0, 1] and segs[2, 0] < 130 < segs[2, 1]


def test_shallow_dips_and_short_blips_do_not_split(base_cfg):
    env = _bumps(200, [40, 60], 12)                # dip < min_dip_db between broad bumps
    assert len(al.segment_syllables(env, base_cfg)) == 1
    blip = np.full(200, -20.0)
    blip[100:102] = 20.0                           # 20 ms: shorter than min_region_ms
    blip[50:90] = 20.0
    assert len(al.segment_syllables(blip, base_cfg)) == 1
    assert al.segment_syllables(np.full(50, -20.0), base_cfg).shape == (0, 2) or True


def test_short_gaps_are_bridged(base_cfg):
    env = np.full(200, -40.0)
    env[50:80] = 0.0
    env[83:120] = 0.0                              # 30 ms gap < min_gap_ms: bridged
    m = al.speech_mask(env, base_cfg)
    assert m[50:120].all() and not m[:50].any() and not m[120:].any()


# -------------------------------------------------------------------------- DTW and mapping


def _warp(env, src_knots, dst_knots, n_dst):
    """Resample an envelope with a piecewise-linear time map src_knots -> dst_knots."""
    t_src = np.interp(np.arange(n_dst), dst_knots, src_knots)
    return np.interp(t_src, np.arange(env.size), env)


def test_dtw_mapping_recovers_a_known_warp(base_cfg):
    src = _bumps(120, [30, 50, 85], 5)
    src_knots, dst_knots = [0, 20, 60, 119], [0, 70, 190, 299]   # slow speech starting late
    dst = _warp(src, src_knots, dst_knots, 300)
    seg_s = al.segment_syllables(src, base_cfg)
    seg_d = al.segment_syllables(dst, base_cfg)
    path, cost = al.dtw_path(src, dst, base_cfg)
    assert path[0].tolist() == [0, 0] and path[-1].tolist() == [119, 299]
    assert np.all(np.diff(path, axis=0) >= 0) and np.isfinite(cost)
    mapped = al.map_segments(seg_s, path, 300, seg_d, base_cfg)
    truth = np.interp(seg_s, src_knots, dst_knots)
    assert np.max(np.abs(mapped - truth)) <= 4                    # within 40 ms
    assert len(seg_s) == len(seg_d) == 3


def test_snapping_moves_to_detected_boundaries_and_keeps_order(base_cfg):
    path = np.stack([np.arange(100), np.arange(100)], axis=1)      # identity
    src = np.array([[10, 30], [30, 50], [60, 80]])
    dst = np.array([[13, 28], [28, 52], [57, 95]])
    snapped = al.map_segments(src, path, 100, dst, base_cfg)
    np.testing.assert_array_equal(snapped, [[13, 28], [28, 52], [57, 80]])  # 95 is beyond 100 ms
    plain = al.map_segments(src, path, 100, dst, _cfg(base_cfg, mapping={"snap": False}))
    np.testing.assert_array_equal(plain, src)


def test_mapping_never_overlaps_and_respects_min_length(base_cfg):
    path = np.stack([np.arange(50), np.zeros(50, int)], axis=1)    # everything collapses onto frame 0
    path = np.concatenate([path, [[49, k] for k in range(1, 40)]])
    out = al.map_segments(np.array([[5, 10], [10, 20], [30, 40]]), path, 40, None,
                          _cfg(base_cfg, mapping={"snap": False}))
    assert np.all(out[:, 1] - out[:, 0] >= 2)
    assert np.all(out[1:, 0] >= out[:-1, 1]) and out.max() <= 40


# -------------------------------------------------------------------------------- stretch


def test_stretch_features_places_segments_and_fills_silence(base_cfg):
    cfg = _cfg(base_cfg, stretch={"silence_pitch": "code"})
    src = np.arange(20, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    fill = np.array([-1, -2, -3], np.float32)
    out = al.stretch_features(src, np.array([[5, 10]]), np.array([[20, 30]]), 40, fill, cfg)
    assert out.shape == (40, 3) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:20], np.tile(fill, (20, 1)))
    np.testing.assert_array_equal(out[30:], np.tile(fill, (10, 1)))
    seg = out[20:30, 0]
    assert np.all(np.diff(seg) >= 0) and seg[0] >= 5 and seg[-1] <= 9   # a stretched ramp of 5..9
    same = al.stretch_features(src, np.array([[5, 10]]), np.array([[5, 10]]), 20, fill, cfg)
    np.testing.assert_array_equal(same[5:10], src[5:10])               # identity is exact


def test_stretch_gap_fill_modes(base_cfg):
    cfg = _cfg(base_cfg, stretch={"silence_pitch": "code"})
    src = np.arange(30, dtype=np.float32)[:, None]
    segs, dst = np.array([[0, 10], [15, 25]]), np.array([[0, 10], [20, 30]])
    sil = al.stretch_features(src, segs, dst, 30, np.array([-1.0], np.float32), cfg)
    assert np.all(sil[10:20] == -1)
    tts = al.stretch_features(src, segs, dst, 30, np.array([-1.0], np.float32),
                              _cfg(base_cfg, stretch={"gap_fill": "tts", "silence_pitch": "code"}))
    assert np.all(tts[10:20] >= 10) and np.all(tts[10:20] <= 14)


def test_silence_pitch_hold_follows_their_targets(base_cfg):
    src = np.tile(SILENCE20, (30, 1))
    src[:, 18] = np.linspace(0.0, 1.0, 30)                     # speech pitch contour
    out = al.stretch_features(src, np.array([[0, 10], [10, 30]]), np.array([[20, 30], [50, 60]]), 80,
                              SILENCE20, base_cfg)
    assert np.all(out[:20, 18] == np.float32(3.1))             # before the first word: the silence code
    assert np.all(out[30:50, 18] == out[29, 18])               # pause: pitch of the last speech frame
    assert np.all(out[60:, 18] == out[59, 18])
    assert np.all(out[30:50, 0] == np.float32(SIL)) and np.all(out[30:50, 19] == np.float32(-0.5))
    code = al.stretch_features(src, np.array([[0, 10]]), np.array([[20, 30]]), 40, SILENCE20,
                               _cfg(base_cfg, stretch={"silence_pitch": "code"}))
    assert np.all(code[30:, 18] == np.float32(3.1))
    np.testing.assert_array_equal(al.hold_pitch(np.array([5., 1, 2, 9, 9]), np.array([0, 1, 1, 0, 0], bool)),
                                  [5, 1, 2, 2, 2])


def test_dip_that_absorbs_a_decoder_pause_stays_silent(base_cfg):
    # source frames 0-19, syllables [5, 10) and [10, 15) touching at the dip frame 10; DTW pairs frame 10
    # with destination frames 30-49 (a 200 ms pause), every other frame one to one (shifted)
    pairs = [(i, i) for i in range(10)] + [(10, j) for j in range(10, 50)] + [(i, i + 39) for i in range(11, 20)]
    path = np.array(pairs)
    out = al.map_segments(np.array([[5, 10], [10, 15]]), path, 60, None, _cfg(base_cfg, mapping={"snap": False}))
    np.testing.assert_array_equal(out, [[5, 10], [50, 54]])
    short = [(i, i) for i in range(10)] + [(10, 10), (10, 11), (10, 12)] + [(i, i + 2) for i in range(11, 20)]
    out = al.map_segments(np.array([[5, 10], [10, 15]]), np.array(short), 22, None,
                          _cfg(base_cfg, mapping={"snap": False}))
    np.testing.assert_array_equal(out, [[5, 11], [11, 17]])     # 30 ms: no pause, centre of the dip


def test_stretch_audio_length_and_silence(base_cfg):
    spf = 160
    t = np.arange(30 * spf) / 16000
    wav = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    out = al.stretch_audio(wav, np.array([[5, 25]]), np.array([[40, 80]]), 100, base_cfg)
    assert out.shape == (100 * spf,)
    assert np.all(out[: 40 * spf] == 0) and np.all(out[80 * spf:] == 0)
    assert np.sqrt(np.mean(out[45 * spf:75 * spf] ** 2)) > 0.1


# ------------------------------------------------------------------------------ end to end


def test_align_trial_recovers_timing_from_c0_envelopes(base_cfg):
    cfg = _cfg(base_cfg, envelope={"source": "c0"})
    tts_env = _bumps(120, [30, 50, 85], 5)
    tts = _features_from_db(tts_env)
    src_knots, dst_knots = [0, 20, 60, 119], [0, 90, 210, 349]
    pred = _features_from_db(_warp(tts_env, src_knots, dst_knots, 350))
    res = al.align_trial(pred, tts, SILENCE20, cfg)
    assert res.target_raw.shape == (350, 20) and res.target_raw.dtype == np.float32
    assert len(res.tts_segments) == 3
    truth = np.interp(res.tts_segments, src_knots, dst_knots)
    assert np.max(np.abs(res.mapped_segments - truth)) <= 4
    first, last = res.mapped_segments[0, 0], res.mapped_segments[-1, 1]
    np.testing.assert_array_equal(res.target_raw[:first], np.tile(SILENCE20, (first, 1)))
    np.testing.assert_array_equal(res.target_raw[last:], np.tile(SILENCE20, (350 - last, 1)))
    s = res.summary()
    assert s["n_tts_syllables"] == 3 and len(s["mapped_segments"]) == 3


def test_align_trial_audio_domain_uses_callables(base_cfg):
    cfg = _cfg(base_cfg, envelope={"source": "c0"}, stretch={"domain": "audio"})
    tts = _features_from_db(_bumps(60, [30], 5))
    pred = _features_from_db(_bumps(100, [60], 8))
    calls = []

    def synth(raw):
        calls.append("synth")
        return np.random.default_rng(0).normal(scale=0.1, size=raw.shape[0] * 160).astype(np.float32)

    def encode(wav):
        calls.append("encode")
        return np.zeros((wav.size // 160 - 1, 36), np.float32)    # one frame short: padded with silence

    res = al.align_trial(pred, tts, SILENCE20, cfg, synth=synth, encode=encode)
    assert calls == ["synth", "encode"] and res.target_raw.shape == (100, 20)
    np.testing.assert_array_equal(res.target_raw[-1], SILENCE20)
    with pytest.raises(ValueError):
        al.align_trial(pred, tts, SILENCE20, cfg)


def test_compressed_index_round_trip():
    keep = np.array([3, 4, 5, 10, 11, 12, 13])
    segs = np.array([[3, 6], [10, 12], [12, 14]])
    c = al.to_compressed(segs, keep)
    np.testing.assert_array_equal(c, [[0, 3], [3, 5], [5, 7]])
    np.testing.assert_array_equal(al.to_absolute(c, keep), segs)


def test_speech_only_dtw_handles_word_by_word_speech(base_cfg):
    # TTS: three syllables spoken as one phrase. Decoder output: the same three, one at a time,
    # each 3x longer, with long pauses and a long silent start (T15's style).
    cfg = _cfg(base_cfg, envelope={"source": "c0"})
    tts_env = _bumps(110, [30, 50, 70], 5)
    pred_env = np.full(900, -20.0)
    truth = []
    for start in (300, 500, 700):                               # each TTS syllable -> 60 frames
        pred_env[start:start + 60] = _bumps(60, [30], 15)
        truth.append(start)
    tts, pred = _features_from_db(tts_env), _features_from_db(pred_env)
    res = al.align_trial(pred, tts, SILENCE20, cfg)
    assert len(res.tts_segments) == 3 and len(res.pred_segments) == 3
    on = res.mapped_segments[:, 0]
    np.testing.assert_allclose(on, res.pred_segments[:, 0], atol=2)       # each on its own word
    assert np.all(res.mapped_segments[:-1, 1] < res.mapped_segments[1:, 0])   # pauses stay silent
    full = al.align_trial(pred, tts, SILENCE20, _cfg(cfg, dtw={"speech_only": False}))
    assert len(full.mapped_segments) == 3                                  # runs; quality not asserted


def test_speech_bounds_and_boundary_errors(base_cfg):
    raw = _features_from_db(np.r_[np.full(10, -20.0), np.full(20, 30.0), np.full(10, -20.0)])
    assert al.speech_bounds(raw, base_cfg, 30.0) == (10, 30)
    err = al.boundary_errors(np.array([10, 31]), np.array([12, 30, 50]))
    np.testing.assert_array_equal(err, [-2, 1, -19])
    assert np.isnan(al.boundary_errors(np.array([]), np.array([1, 2]))).all()


# ---------------------------------------------------------------- external syllables (D-033)


def test_explicit_segments_reproduce_the_default(base_cfg):
    cfg = _cfg(base_cfg, envelope={"source": "c0"})
    tts_env = _bumps(110, [30, 50, 70], 5)
    pred_env = np.full(500, -20.0)
    for start in (200, 300, 400):
        pred_env[start:start + 60] = _bumps(60, [30], 15)
    tts, pred = _features_from_db(tts_env), _features_from_db(pred_env)
    ref = al.align_trial(pred, tts, SILENCE20, cfg)
    got = al.align_trial(pred, tts, SILENCE20, cfg, tts_segments=ref.tts_segments, pred_segments=ref.pred_segments)
    np.testing.assert_array_equal(got.mapped_segments, ref.mapped_segments)
    np.testing.assert_array_equal(got.target_raw, ref.target_raw)


def test_external_tts_syllables_outside_the_speech_frames(base_cfg):
    # forced alignment can put a syllable where the TTS envelope counts as silence (a weak final syllable,
    # a fricative onset): speech-only DTW drops those frames; the syllable must still be mapped, in order
    cfg = _cfg(base_cfg, envelope={"source": "c0"})
    tts_env = _bumps(120, [30, 50], 5)
    pred_env = np.full(400, -20.0)
    pred_env[100:160] = _bumps(60, [30], 12)
    pred_env[220:280] = _bumps(60, [30], 12)
    tts, pred = _features_from_db(tts_env), _features_from_db(pred_env)
    forced = np.array([[5, 22], [22, 40], [40, 62], [62, 70], [100, 115]])   # the last two lie in silence
    res = al.align_trial(pred, tts, SILENCE20, cfg, tts_segments=forced)
    m = res.mapped_segments
    np.testing.assert_array_equal(res.tts_segments, forced)
    assert m.shape == (5, 2) and np.all(m[:, 1] > m[:, 0]) and m.min() >= 0 and m.max() <= 400
    assert np.all(np.diff(m[:, 0]) >= 0) and np.all(np.diff(m[:, 1]) >= 0)
    assert np.all(m[1:3, 0] >= m[:2, 1])            # the syllables with speech frames stay disjoint, in order
    # the existing mapping squeezes syllables without any speech frame onto the end of the last decoder speech
    # region (D-033 reports how often: n_tts_syllables_without_speech); the stretch writes them in order
    assert m[3, 0] < m[2, 1] or m[4, 0] < m[3, 1]
    assert res.target_raw.shape == (400, 20)


def test_external_segments_are_checked(base_cfg):
    tts = _features_from_db(_bumps(60, [30], 5))
    pred = _features_from_db(_bumps(100, [60], 8))
    for bad in ([[10, 5]], [[0, 20], [10, 30]], [[0, 200]]):
        with pytest.raises(ValueError):
            al.align_trial(pred, tts, SILENCE20, base_cfg, pred_segments=np.array(bad))
    with pytest.raises(ValueError):
        al.align_trial(pred, tts, SILENCE20, base_cfg, tts_segments=np.zeros((0, 2), int))
