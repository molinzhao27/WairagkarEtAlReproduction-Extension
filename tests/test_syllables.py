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
    assert old["syllables"] == {"tts_side": "envelope", "method": "current", "count_constraint": False}
    assert sy.load_any_targets_config("configs/experiments/kunz_targets.yaml") == old
    assert sy.load_any_targets_config("configs/experiments/syllable_benchmark.yaml")["syllables"]["tts_side"] == "forced"


def test_settings_rejects_unknown_values(cfg):
    with pytest.raises(ValueError):
        sy.settings(deep_merge(cfg, {"syllables": {"method": "dips"}}))
    with pytest.raises(ValueError):
        sy.settings(deep_merge(cfg, {"syllables": {"tts_side": "manual"}}))


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


RULE = {"primary_cells": ["a", "b"], "tie_tolerance": 0.01, "simplicity_order": ["current", "nuclei", "sylber"]}


def test_choose_method_highest_mean_wins():
    r = {"current": {"a": 0.20, "b": 0.22}, "nuclei": {"a": 0.30, "b": 0.24}, "sylber": {"a": 0.25, "b": 0.25}}
    out = sy.choose_method(r, RULE)
    assert out["winner"] == "nuclei" and out["tied_within_tolerance"] == ["nuclei"]
    assert out["mean_target_r"]["nuclei"] == pytest.approx(0.27)


def test_choose_method_ties_go_to_the_simpler_method():
    r = {"current": {"a": 0.300, "b": 0.300}, "nuclei": {"a": 0.309, "b": 0.309}, "sylber": {"a": 0.318, "b": 0.318}}
    out = sy.choose_method(r, RULE)        # within 0.01 of the best (0.318): nuclei and sylber; current is not
    assert out["tied_within_tolerance"] == ["nuclei", "sylber"] and out["winner"] == "nuclei"
    r["current"] = {"a": 0.3085, "b": 0.3085}
    assert sy.choose_method(r, RULE)["winner"] == "current"
    with pytest.raises(ValueError):
        sy.choose_method({"current": {"a": 0.3}, "nuclei": {"a": 0.3, "b": 0.3}, "sylber": {"a": 0.3, "b": 0.3}}, RULE)


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
