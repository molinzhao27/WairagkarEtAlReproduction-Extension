"""wvx.syllables: maximal-onset syllabification, forced-alignment output, decoder-side segments, the choice
rule and the benchmark config (pure NumPy; no Praat, PyTorch, MFA or data)."""

import json
import re

import numpy as np
import pytest

from wvx import alignment as al
from wvx import syllables as sy
from wvx.config import deep_merge


@pytest.fixture(scope="module")
def cfg():
    return sy.load_benchmark_config()


# ------------------------------------------------------------------------------- config


def test_benchmark_config_merges_the_whole_chain(cfg):
    assert cfg["synthesis"]["samples_per_frame"] == 160              # reproduce_inference.yaml
    assert cfg["alignment"]["segmentation"]["min_dip_db"] == 3.0      # kunz_targets.yaml, unchanged
    assert cfg["syllables"]["tts_side"] == "forced"                   # syllable_benchmark.yaml
    assert cfg["paths"]["kunz_targets"].name == "kunz_targets_v2" and cfg["paths"]["syllables"].is_absolute()
    sy.settings(cfg)


def test_benchmark_design_matches_the_preregistration(cfg):
    b = cfg["benchmark"]
    assert b["choice"]["primary_cells"] == ["day195_m172", "day195_m165"]
    assert b["choice"]["simplicity_order"] == list(sy.METHODS) == b["methods"]
    assert b["choice"]["arms"] == list(sy.ARMS) == b["arms"] and b["reference"]["arm"] == "A"
    assert cfg["syllables"]["segment_mapping"] == {"equal_counts": "one_to_one", "position_weight": 1.0,
                                                   "log_duration_weight": 1.0}
    assert b["choice"]["tie_tolerance"] == 0.01 and b["onset_tolerance_ms"] in cfg["validation"]["tolerances_ms"]
    cells = b["cells"]
    assert {c: (cells[c]["session"], cells[c]["age_days"]) for c in ("day195_m172", "day195_m165", "day195_m193", "day193_m165")} == {
        "day195_m172": ("t15_day195", 23), "day195_m165": ("t15_day195", 30),
        "day195_m193": ("t15_day195", 2), "day193_m165": ("t15_day193", 28)}
    assert cells["kunz_attempted"]["model"] == cfg["kunz"]["decoder"]
    for c in cells.values():                       # the decoder named in the file has the age the cell claims
        if "session" in c:
            model_day = int(re.search(r"brain2voice_t15_day(\d+)\.h5$", c["model"]).group(1))
            assert int(c["session"].split("day")[1]) - model_day == c["age_days"]
    n = cfg["syllables"]["segmenters"]["nuclei"]
    assert (n["ignorance_level_db"], n["min_dip_db"], n["intensity"]["min_pitch_hz"]) == (0.0, 2.0, 50.0)
    s = cfg["syllables"]["segmenters"]["sylber"]
    assert (s["encoding_layer"], s["merge_threshold"], s["norm_threshold"]) == (9, 0.8, 2.6)


def test_kunz_targets_defaults_are_d027(cfg):
    from wvx.targets import load_targets_config

    old = load_targets_config()
    assert old["syllables"] == {"tts_side": "envelope", "method": "current", "count_constraint": False, "arm": "A"}
    assert sy.load_any_targets_config("configs/experiments/kunz_targets.yaml") == old
    assert sy.load_any_targets_config("configs/experiments/syllable_benchmark.yaml")["syllables"]["tts_side"] == "forced"


def test_settings_rejects_unknown_values(cfg):
    with pytest.raises(ValueError):
        sy.settings(deep_merge(cfg, {"syllables": {"method": "dips"}}))
    with pytest.raises(ValueError):
        sy.settings(deep_merge(cfg, {"syllables": {"tts_side": "manual"}}))
    with pytest.raises(ValueError):
        sy.settings(deep_merge(cfg, {"syllables": {"arm": "C"}}))


# --------------------------------------------------------------------- maximal onset


ONSETS = frozenset({("S", "T", "R"), ("T", "R"), ("S", "T"), ("K", "R"), ("S",), ("T",), ("R",), ("K",), ("B",), ("L",),
                    ("P", "L"), ("N",), ("M",)})


def test_legal_onsets_counts_word_initial_clusters():
    prons = [["S", "T", "R", "IH1", "NG"], ["S", "T", "AA1", "P"], ["T", "R", "IY1"], ["AE1", "T"], ["S", "T", "EH1", "M"],
             ["HH", "M"]]                                      # no vowel: not an onset
    assert sy.legal_onsets(prons, 1) == {("S", "T", "R"), ("S", "T"), ("T", "R")}
    assert sy.legal_onsets(prons, 2) == {("S", "T")}


def test_syllabify_maximal_onset():
    # extra: EH1 K S T R AH0 -> EH K . S T R AH (S T R is a legal onset, K S T R is not)
    assert sy.syllabify(["EH1", "K", "S", "T", "R", "AH0"], ONSETS) == [(0, 2), (2, 6)]
    # without S T R, the longest legal suffix is T R
    assert sy.syllabify(["EH1", "K", "S", "T", "R", "AH0"], ONSETS - {("S", "T", "R")}) == [(0, 3), (3, 6)]
    # about: AH0 . B AW1 T ; hiatus (no consonant) and a cluster with no legal suffix (NG)
    assert sy.syllabify(["AH0", "B", "AW1", "T"], ONSETS) == [(0, 1), (1, 4)]
    assert sy.syllabify(["K", "EY1", "AA2", "S"], ONSETS) == [(0, 2), (2, 4)]
    assert sy.syllabify(["S", "IH1", "NG", "ER0"], ONSETS) == [(0, 3), (3, 4)]
    assert sy.syllabify(["S", "T", "R", "IH1", "NG", "Z"], ONSETS) == [(0, 6)]   # one vowel: the whole word
    assert sy.syllabify(["HH", "M"], ONSETS) == []
    for word in (["EH1", "K", "S", "T", "R", "AH0"], ["F", "AE1", "M", "AH0", "L", "IY0"]):
        parts = sy.syllabify(word, ONSETS)
        assert parts[0][0] == 0 and parts[-1][1] == len(word)
        assert all(a < b for a, b in parts) and all(p[1] == q[0] for p, q in zip(parts, parts[1:]))
        assert len(parts) == sum(sy.is_vowel(p) for p in word)


def test_dictionary_syllable_count():
    first = {"i": ["AY1"], "family": ["F", "AE1", "M", "AH0", "L", "IY0"], "hmm": ["HH", "M"]}
    assert sy.dictionary_syllable_count(["I", "family"], first) == 4
    assert sy.dictionary_syllable_count(["hmm"], first) == 0
    assert sy.dictionary_syllable_count(["family", "zork"], first) is None
    assert sy.cmudict_first_pronunciations({"a": [["AH0"], ["EY1"]], "b": []}) == {"a": ["AH0"]}


def _mfa_json(tmp_path, words, phones):
    d = {"start": 0, "end": 2.0, "tiers": {"words": {"type": "interval", "entries": words},
                                            "phones": {"type": "interval", "entries": phones}}}
    p = tmp_path / "u0000.json"
    p.write_text(json.dumps(d))
    return p


def test_tts_syllables_from_aligner_output(cfg, tmp_path):
    words = [[0.20, 0.37, "i"], [0.37, 0.60, "hope"], [0.80, 1.20, "extra"]]          # a pause before "extra"
    phones = [[0.20, 0.37, "AY1"], [0.37, 0.45, "HH"], [0.45, 0.54, "OW1"], [0.54, 0.60, "P"],
              [0.60, 0.80, "sil"],
              [0.80, 0.90, "EH1"], [0.90, 0.95, "K"], [0.95, 1.00, "S"], [1.00, 1.05, "T"], [1.05, 1.10, "R"],
              [1.10, 1.20, "AH0"]]
    w, p = sy.read_aligner_json(_mfa_json(tmp_path, words, phones), cfg)
    u = sy.tts_syllables(w, p, ONSETS, cfg)
    assert u["words"] == ["i", "hope", "extra"] and not u["unknown_words"]
    np.testing.assert_allclose(u["syllables_s"], [[0.20, 0.37], [0.37, 0.60], [0.80, 0.95], [0.95, 1.20]])
    unk = sy.tts_syllables([(0.2, 0.5, "<unk>")], [(0.2, 0.5, cfg["syllables"]["forced_alignment"]["unknown_label"])],
                           ONSETS, cfg)
    assert unk["unknown_words"] == ["<unk>"] and unk["syllables_s"] == [] and unk["words_without_vowel"] == ["<unk>"]


def test_seconds_to_frames_applies_the_synthesis_delay(cfg):
    lag = cfg["syllables"]["audio_lag_ms"]
    c = deep_merge(cfg, {"syllables": {"audio_lag_ms": 15.0}})
    # frame k's content sits at [10k + 15, 10k + 25) ms of audio
    np.testing.assert_array_equal(sy.seconds_to_frames([[0.215, 0.375]], c), [[20, 36]])
    np.testing.assert_array_equal(sy.seconds_to_points([0.2151, 0.2249], c, 100), [20, 20])
    np.testing.assert_array_equal(sy.seconds_to_points([0.215], c, 100, boundary=True), [20])
    # empty after rounding: dropped, unless keep_empty (TTS), which keeps count and order
    assert sy.seconds_to_frames([[0.2150, 0.2151]], c).shape == (0, 2)
    kept = sy.seconds_to_frames([[0.215, 0.2151], [0.2151, 0.305]], c, n_frames=50, keep_empty=True)
    np.testing.assert_array_equal(kept, [[20, 21], [21, 29]])
    np.testing.assert_array_equal(sy.seconds_to_frames([[0.0, 0.105]], c, n_frames=50), [[0, 9]])   # clipped at 0
    with pytest.raises(ValueError):
        sy.seconds_to_frames([[6.0, 9.0]], c, n_frames=50, keep_empty=True)
    assert lag == 15.0                                                          # the measured delay


# --------------------------------------------------------------------------- decoder side


def test_nuclei_to_segments():
    speech = np.zeros(100, bool)
    speech[10:40] = True
    speech[60:70] = True                    # a run without a nucleus: no syllable
    nuclei = [15, 25, 33, 85]               # three in the first run, one outside every run
    cuts = [20, 30, 60]                     # cut between nuclei i and i + 1
    np.testing.assert_array_equal(sy.nuclei_to_segments(nuclei, cuts, speech), [[10, 20], [20, 30], [30, 40]])
    assert sy.nuclei_to_segments([], [], speech).shape == (0, 2)
    np.testing.assert_array_equal(sy.nuclei_to_segments([15], [], speech), [[10, 40]])
    with pytest.raises(ValueError):
        sy.nuclei_to_segments([15, 25], [], speech)
    with pytest.raises(ValueError):
        sy.nuclei_to_segments([25, 15], [20], speech)


def test_keep_strongest():
    env = np.zeros(100)
    env[5], env[15], env[25], env[35] = 3.0, 9.0, 1.0, 9.0
    segs = np.array([[0, 10], [10, 20], [20, 30], [30, 40]])
    np.testing.assert_array_equal(sy.keep_strongest(segs, env, 2), [[10, 20], [30, 40]])
    np.testing.assert_array_equal(sy.keep_strongest(segs, env, 3), [[0, 10], [10, 20], [30, 40]])
    np.testing.assert_array_equal(sy.keep_strongest(segs, env, 9), segs)          # fewer than N: unchanged
    np.testing.assert_array_equal(sy.keep_strongest(segs, env, None), segs)
    tie = sy.keep_strongest(segs[1::2], env, 1)                                   # equal peaks: the earlier
    np.testing.assert_array_equal(tie, [[10, 20]])


def _bumps(n, centres, width, height_db=40.0, floor_db=-20.0):
    t = np.arange(n)[:, None]
    lin = np.exp(-0.5 * ((t - np.asarray(centres)[None, :]) / width) ** 2).sum(1)
    return floor_db + height_db * np.minimum(lin, 1.0)


def test_decoder_segments_dispatch(cfg):
    env = _bumps(300, [100, 118, 220], 6)            # two touching syllables, a pause, one more
    np.testing.assert_array_equal(sy.decoder_segments("current", env, cfg), al.segment_syllables(env, cfg))
    lag_s = cfg["syllables"]["audio_lag_ms"] / 1000
    rec = {"nuclei_s": [1.0 + lag_s, 1.18 + lag_s, 2.2 + lag_s], "cuts_s": [1.09 + lag_s, 1.7 + lag_s]}
    nuc = sy.decoder_segments("nuclei", env, cfg, rec)
    mask_runs = al.runs(al.speech_mask(env, cfg))
    assert len(mask_runs) == 2
    np.testing.assert_array_equal(nuc, [[mask_runs[0, 0], 109], [109, mask_runs[0, 1]], mask_runs[1]])
    syl = sy.decoder_segments("sylber", env, cfg, {"segments_s": [[0.9 + lag_s, 1.1 + lag_s], [1.1 + lag_s, 1.4 + lag_s]]})
    np.testing.assert_array_equal(syl, [[90, 110], [110, 140]])
    with pytest.raises(ValueError):
        sy.decoder_segments("nuclei", env, cfg)
    with pytest.raises(ValueError):
        sy.decoder_segments("dips", env, cfg, rec)


# ------------------------------------------------------------------------------- choice rule


RULE = {"primary_cells": ["a", "b"], "tie_tolerance": 0.01, "arms": ["A", "B"],
        "simplicity_order": ["current", "nuclei", "sylber"]}


def _r(**means):
    """Candidate means -> target_r table (same value in both primary cells)."""
    return {k.replace("_", "/"): {"a": v, "b": v} for k, v in means.items()}


def test_candidates_order_arm_first():
    assert sy.candidates(RULE) == ["A/current", "A/nuclei", "A/sylber", "B/current", "B/nuclei", "B/sylber"]


def test_choose_method_highest_mean_wins():
    r = _r(A_current=0.20, A_nuclei=0.30, A_sylber=0.25, B_current=0.21, B_nuclei=0.22, B_sylber=0.23)
    r["A/nuclei"] = {"a": 0.33, "b": 0.27}
    out = sy.choose_method(r, RULE)
    assert out["winner"] == "A/nuclei" and (out["winner_arm"], out["winner_method"]) == ("A", "nuclei")
    assert out["tied_within_tolerance"] == ["A/nuclei"] and out["mean_target_r"]["A/nuclei"] == pytest.approx(0.30)
    r["B/sylber"] = {"a": 0.40, "b": 0.40}
    assert sy.choose_method(r, RULE)["winner"] == "B/sylber"


def test_choose_method_ties_go_to_arm_a_then_the_simpler_method():
    # within 0.01 of the best (B/sylber 0.318): A/sylber and B/nuclei; arm A comes first
    r = _r(A_current=0.300, A_nuclei=0.300, A_sylber=0.309, B_current=0.300, B_nuclei=0.315, B_sylber=0.318)
    out = sy.choose_method(r, RULE)
    assert out["tied_within_tolerance"] == ["A/sylber", "B/nuclei", "B/sylber"] and out["winner"] == "A/sylber"
    r["A/sylber"] = {"a": 0.300, "b": 0.300}                     # now only arm B ties: the simpler method
    assert sy.choose_method(r, RULE)["winner"] == "B/nuclei"
    del r["B/current"]
    with pytest.raises(ValueError):
        sy.choose_method(r, RULE)


# ------------------------------------------------------------------- arm B: segment-driven


MAP = {"equal_counts": "one_to_one", "position_weight": 1.0, "log_duration_weight": 1.0}


def test_syllable_cost_uses_speech_time_shares():
    tts = np.array([[10, 20], [20, 40], [50, 60]])              # durations 10, 20, 10 (pause ignored)
    dec = 3 * tts + 100                                          # same structure, slower, shifted, pauses scaled
    c = sy.syllable_cost(tts, dec, MAP)
    assert c.shape == (3, 3) and np.allclose(np.diag(c), 0.0)
    assert np.all(c[~np.eye(3, dtype=bool)] > 0)
    onset, share = sy.speech_time_features(tts)
    np.testing.assert_allclose(onset, [0.0, 0.25, 0.75])
    np.testing.assert_allclose(share, [0.25, 0.5, 0.25])
    # one syllable's worth of misplacement costs about 1
    c2 = sy.syllable_cost(np.array([[0, 10], [10, 20]]), np.array([[0, 10], [10, 20]]), MAP)
    assert c2[0, 1] == pytest.approx(1.0)


def _check_path(path, K, J):
    assert path[0].tolist() == [0, 0] and path[-1].tolist() == [K - 1, J - 1]
    steps = np.diff(path, axis=0)
    assert all(tuple(st) in {(1, 1), (1, 0), (0, 1)} for st in steps)
    assert set(path[:, 0]) == set(range(K)) and set(path[:, 1]) == set(range(J))     # complete coverage


@pytest.mark.parametrize("K,J", [(3, 3), (5, 2), (2, 6), (1, 4), (4, 1), (7, 5)])
def test_syllable_path_and_assignment_are_monotone_and_complete(K, J):
    rng = np.random.default_rng(K * 10 + J)
    def segs(n):
        d = rng.integers(5, 30, n)
        on = np.cumsum(np.r_[0, d[:-1] + rng.integers(0, 20, n - 1)])
        return np.stack([on, on + d], axis=1)
    t, d = segs(K), segs(J)
    C = sy.syllable_cost(t, d, MAP)
    path = sy.syllable_path(C)
    _check_path(path, K, J)
    a = sy.assign_syllables(path, C)
    assert a.shape == (K,) and np.all(np.diff(a) >= 0) and a.min() >= 0 and a.max() < J
    for i in range(K):                                           # each TTS syllable goes to one of its path pairs
        assert a[i] in path[path[:, 0] == i, 1]
    np.testing.assert_array_equal(sy.syllable_path(C), path)     # deterministic


def test_equal_structure_gives_the_diagonal_even_by_dtw():
    t = np.array([[0, 10], [10, 30], [40, 50], [50, 55]])
    C = sy.syllable_cost(t, 2 * t + 7, MAP)
    np.testing.assert_array_equal(sy.syllable_path(C), np.stack([np.arange(4)] * 2, axis=1))


def test_segment_driven_spans_one_to_one_and_shared():
    tts = np.array([[0, 10], [10, 30], [30, 40]])
    dec = np.array([[100, 140], [200, 260]])
    np.testing.assert_array_equal(sy.segment_driven_spans(tts[:2], dec, np.array([0, 1])), dec)
    spans = sy.segment_driven_spans(tts, dec, np.array([0, 0, 1]))   # TTS 0 and 1 share decoder syllable 0
    np.testing.assert_array_equal(spans, [[100, 113], [113, 140], [200, 260]])   # split 10:20 of 40 frames
    left_silent = sy.segment_driven_spans(tts[:1], dec, np.array([1]))
    np.testing.assert_array_equal(left_silent, [[200, 260]])


SIL20 = np.r_[-12.48528, np.zeros(17), 3.1, -0.5].astype(np.float32)


def _feats(env_db, seed=0):
    """Raw (n, 20) frames: c0 from the dB envelope, other cepstra a seeded walk, voiced pitch."""
    f = np.tile(SIL20, (env_db.size, 1)).astype(np.float64)
    f[:, 0] = env_db * np.sqrt(18.0) / 10.0 - 4.0
    f[:, 1:18] = np.cumsum(np.random.default_rng(seed).normal(scale=0.1, size=(env_db.size, 17)), axis=0)
    f[:, 18], f[:, 19] = 0.5, 0.3
    return f.astype(np.float32)


def test_arm_b_equal_counts_one_to_one_and_spans_match(cfg):
    tts = _feats(_bumps(100, [30, 50, 70], 5))
    seg_t = np.array([[22, 40], [40, 60], [60, 80]])
    seg_p = np.array([[150, 190], [230, 300], [330, 360]])
    pred = _feats(np.full(400, -20.0), 1)
    res = sy.align_trial_segment_driven(pred, tts, SIL20, cfg, tts_segments=seg_t, pred_segments=seg_p)
    assert res.info["pairing"] == "one_to_one" and res.info["assignment"] == [0, 1, 2]
    np.testing.assert_array_equal(res.mapped_segments, seg_p)
    expect = al.stretch_features(tts, seg_t, seg_p, 400, SIL20, cfg)
    np.testing.assert_array_equal(res.target_raw, expect)
    for (a, b), (c, e) in zip(seg_t, seg_p):                     # each span holds its TTS syllable, stretched
        np.testing.assert_allclose(res.target_raw[c:e], al._resample(tts[a:b].astype(np.float64), e - c, "linear"),
                                   atol=1e-5)
    gaps = np.ones(400, bool)
    for c, e in seg_p:
        gaps[c:e] = False
    assert np.all(res.target_raw[gaps, 0] == np.float32(SIL20[0]))       # silence between and around
    np.testing.assert_array_equal(res.target_raw[190:230, 18], np.full(40, res.target_raw[189, 18]))  # pitch held


@pytest.mark.parametrize("seg_p", [np.array([[150, 260]]),                              # fewer decoder syllables
                                   np.array([[100, 120], [150, 190], [195, 230], [300, 330], [360, 380]])])  # more
def test_arm_b_unequal_counts_monotone_complete(cfg, seg_p):
    tts = _feats(_bumps(100, [30, 50, 70], 5))
    seg_t = np.array([[22, 40], [40, 60], [60, 80]])
    pred = _feats(np.full(400, -20.0), 1)
    res = sy.align_trial_segment_driven(pred, tts, SIL20, cfg, tts_segments=seg_t, pred_segments=seg_p)
    a = np.asarray(res.info["assignment"])
    m = res.mapped_segments
    assert res.info["pairing"] == "dtw" and m.shape == (3, 2) and np.all(np.diff(a) >= 0)
    assert np.all(m[:, 1] > m[:, 0]) and np.all(m[1:, 0] >= m[:-1, 1])           # every TTS syllable placed, in order
    for (c, e), j in zip(m, a):                                                    # inside its decoder syllable
        assert seg_p[j, 0] <= c and e <= seg_p[j, 1]
    for j in np.unique(a):                                                          # shared spans tile the syllable
        idx = np.flatnonzero(a == j)
        assert m[idx[0], 0] == seg_p[j, 0] and m[idx[-1], 1] == seg_p[j, 1]
    silent = [j for j in range(len(seg_p)) if j not in a]
    assert res.info["n_decoder_syllables_without_tts"] == len(silent)
    for j in silent:                                                                # unmatched: silence
        assert np.all(res.target_raw[seg_p[j, 0]:seg_p[j, 1], 0] == np.float32(SIL20[0]))


def test_arm_b_without_decoder_syllables_is_silence(cfg):
    tts = _feats(_bumps(100, [30, 50], 5))
    pred = _feats(np.full(300, -20.0), 1)
    res = sy.align_trial_segment_driven(pred, tts, SIL20, cfg, tts_segments=np.array([[20, 40], [40, 60]]),
                                        pred_segments=np.zeros((0, 2), int))
    assert res.info["n_decoder_syllables"] == 0 and np.all(res.target_raw[:, 0] == np.float32(SIL20[0]))


# ------------------------------------------------------------------------------- file formats


def test_decoded_round_trip(tmp_path):
    preds = [np.random.default_rng(0).normal(size=(n, 20)).astype(np.float32) for n in (3, 5)]
    path = sy.save_decoded(tmp_path / "c" / "predictions.npz", preds, {"n_bins": [62, 64], "cell": "x"})
    back, header = sy.load_decoded(path)
    assert header["cell"] == "x" and header["pred_lengths"] == [3, 5]
    for a, b in zip(preds, back):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        sy.save_decoded(tmp_path / "d.npz", preds, {"n_bins": [1]})


def test_save_json_refuses_nan(tmp_path):
    sy.save_json(tmp_path / "ok.json", {"a": [1.0]})
    assert sy.load_json(tmp_path / "ok.json") == {"a": [1.0]}
    with pytest.raises(ValueError):
        sy.save_json(tmp_path / "bad.json", {"a": float("nan")})
