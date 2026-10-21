"""wvx.neural_match and scripts/neural_match.py on synthetic trials (CPU, no data, no TensorFlow).

Pooling, PCA fitted on training frames only, DTW similarity (speed invariance), mid-ranks with ties,
length-matched candidates, the permutation null (uniform, calibrated, equal to the exact null), a planted
signal recovered end to end and pure noise at chance, and the script on tiny synthetic input files.
"""

import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from wvx import neural_match as nm
from wvx.config import REPO_ROOT, deep_merge
from wvx.experiment_config import load_experiment_config

CONFIG = "configs/experiments/neural_match.yaml"
PAIRS = {3: 4, 6: 7, 11: 9, 13: 12}
D_LAT = 6          # latent dimensions of the synthetic "sentence" trajectories
IMAGINED_NOISE = 10.0   # per-bin noise s.d. of synthetic imagined trials (signal: 0.4 x the attempted one)


@pytest.fixture(scope="module")
def cfg():
    return load_experiment_config(CONFIG)


@pytest.fixture(scope="module")
def settings(cfg):
    return deep_merge(cfg["neural_match"], {"n_permutations": 2000})   # fewer relabelings: speed only


# ------------------------------------------------------------------------------------------- config


def test_config_is_the_preregistered_design(cfg):
    s = cfg["neural_match"]
    assert nm.deviations({k: v for k, v in s.items() if k != "preregistered"}, s["preregistered"]) == []
    assert s["pool_bins"] == 5 and s["pca"]["n_components"] == 32 and s["length_matched_k"] == 10
    assert s["n_permutations"] == 10000 and s["dtw"]["metric"] == "cosine"
    assert s["dtw"]["global_constraints"] is False
    assert 1 <= cfg["runtime"]["threads"]["omp"] <= 2       # shared host
    assert cfg["runtime"]["nice"] is not None
    assert "{kunz_processed}" in s["attempted_file"] and "{kunz_processed}" in s["imagined_file"]
    # The block-normalized length baseline is an ADDITION: switched on, absent from the frozen section.
    assert s["additional_baselines"]["length_block_normalized"] is True
    assert "additional_baselines" not in s["preregistered"]
    add = nm.additions(s)
    assert [a["measure"] for a in add] == ["length_norm"]
    assert add[0]["label"] == "block-normalized (added 2026-10-21 before any run)"
    assert nm.active_measures(s) == ("dtw", "pattern", "length", "length_norm")
    assert "raw (pre-registered)" in nm.MEASURE_LABELS["length"]


def test_deviations_are_reported(cfg):
    s = {k: v for k, v in cfg["neural_match"].items() if k != "preregistered"}
    changed = deep_merge(s, {"length_matched_k": 5, "dtw": {"global_constraints": True}})
    dev = nm.deviations(changed, cfg["neural_match"]["preregistered"])
    assert {d["key"] for d in dev} == {"length_matched_k", "dtw.global_constraints"}
    assert {"key": "length_matched_k", "preregistered": 10, "used": 5} in dev


def test_import_needs_no_tensorflow():
    code = "import sys, wvx.neural_match; print('tensorflow' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})
    assert out.stdout.strip() == "False"


# ----------------------------------------------------------------------------------- representation


def test_pool_frames_means_and_partial_window():
    x = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    p = nm.pool_frames(x, 5)
    assert p.shape == (2, 2)
    np.testing.assert_allclose(p[0], x[:5].mean(0))
    np.testing.assert_allclose(p[1], x[5:10].mean(0))
    k = nm.pool_frames(x, 5, drop_partial=False)
    assert k.shape == (3, 2)
    np.testing.assert_allclose(k[2], x[10:].mean(0))
    with pytest.raises(ValueError):
        nm.pool_frames(x[:4], 5)


def test_pca_uses_training_statistics_and_is_scale_invariant():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 2)) * [3.0, 1.0]
    train = z @ rng.normal(size=(2, 10)) + 0.05 * rng.normal(size=(500, 10)) + 7.0
    m = nm.fit_pca(train, 2)
    assert m.explained_variance_ratio.sum() > 0.95
    assert m.explained_variance_ratio[0] >= m.explained_variance_ratio[1]
    np.testing.assert_allclose(m.mean, train.mean(0))
    np.testing.assert_allclose(m.scale, train.std(0))
    np.testing.assert_allclose(m.components @ m.components.T, np.eye(2), atol=1e-10)
    proj = nm.apply_pca(m, train)
    np.testing.assert_allclose(proj.mean(0), 0.0, atol=1e-10)
    # New data is standardized with the TRAINING mean and s.d., not its own.
    test = train[:50] + 100.0
    np.testing.assert_allclose(nm.apply_pca(m, test), ((test - m.mean) / m.scale) @ m.components.T)
    assert np.abs(nm.apply_pca(m, test).mean(0)).max() > 1.0
    # Standardization makes the result independent of per-feature units.
    scales = rng.uniform(0.01, 100.0, 10)
    m2 = nm.fit_pca(train * scales, 2)
    np.testing.assert_allclose(nm.apply_pca(m2, test * scales), nm.apply_pca(m, test), atol=1e-8)
    # A constant feature is centred, not divided by zero.
    const = train.copy()
    const[:, 3] = 1.0
    assert np.all(np.isfinite(nm.apply_pca(nm.fit_pca(const, 2, min_std=1e-8), const)))
    with pytest.raises(ValueError):
        nm.fit_pca(train, 11)


# ----------------------------------------------------------------------------------------- similarity


def _smooth_function(rng, d=D_LAT, n_terms=3):
    """A random smooth trajectory u in [0, 1] -> (len(u), d)."""
    f = rng.uniform(0.5, 3.0, (n_terms, d))
    ph = rng.uniform(0, 2 * np.pi, (n_terms, d))
    a = rng.normal(size=(n_terms, d))
    return lambda u: (a[None] * np.sin(2 * np.pi * f[None] * np.asarray(u)[:, None, None] + ph[None])).sum(1)


def test_dtw_similarity_identical_trajectory_at_different_speeds(cfg):
    dtw = cfg["neural_match"]["dtw"]
    rng = np.random.default_rng(1)
    f = _smooth_function(rng, d=32)
    slow = f(np.linspace(0, 1, 230))
    fast = f(np.linspace(0, 1, 100))
    warped = f(np.linspace(0, 1, 160) ** 1.4)          # nonlinear time warp
    assert nm.dtw_similarity(slow, fast, dtw) > 0.97
    assert nm.dtw_similarity(fast, warped, dtw) > 0.97
    assert nm.dtw_similarity(slow, 0.3 * fast, dtw) == pytest.approx(nm.dtw_similarity(slow, fast, dtw))
    other = _smooth_function(rng, d=32)(np.linspace(0, 1, 100))
    assert nm.dtw_similarity(slow, other, dtw) < 0.8
    with pytest.raises(ValueError):
        nm.dtw_similarity(slow, np.zeros((10, 32)), dtw)


def test_block_normalized_lengths():
    np.testing.assert_allclose(nm.block_normalized_lengths([100, 200, 300]), [0.5, 1.0, 1.5])
    np.testing.assert_allclose(nm.block_normalized_lengths([150, 300, 450]), [0.5, 1.0, 1.5])   # scale-free
    with pytest.raises(ValueError):
        nm.block_normalized_lengths([100, 0])


def test_block_normalized_baseline_removes_a_condition_wide_slowdown(settings):
    """Imagined = exactly 1.5x its own attempted trial's duration: the block-normalized baseline finds the own
    trial every time; the raw (pre-registered) one prefers attempted trials 1.5x longer instead."""
    rng = np.random.default_rng(14)
    lengths = 2 * rng.choice(np.arange(50, 200), size=48, replace=False)    # distinct, even
    att, img = [], []
    for p, (a, i) in enumerate(PAIRS.items()):
        for s in range(12):
            idx = p * 12 + s
            text, n = f"Cue {p} {s}.", int(lengths[idx])
            att.append(nm.MatchTrial(rng.normal(size=(n, 40)).astype(np.float32), text, text, a, idx, n / 100))
            img.append(nm.MatchTrial(rng.normal(size=(n * 3 // 2, 40)).astype(np.float32), text, text, i, idx,
                                     n / 100))
    rows, res = nm.analyze(att, img, PAIRS, settings, seed=42)
    for c in nm.CANDIDATE_SETS:
        norm = res["length_norm"][c]["pooled"]
        assert norm["mean_rank"] == 1 and norm["top1_rate"] == 1 and norm["p_exact"] < 1e-10
    assert res["length"]["all"]["pooled"]["mean_rank"] > 3
    assert all(r["length_norm_all_rank"] == 1 for r in rows)
    assert res["block_median_go_bins"]["I04"] == pytest.approx(1.5 * res["block_median_go_bins"]["A03"])
    off = deep_merge(settings, {"additional_baselines": {"length_block_normalized": False}})
    rows_off, res_off = nm.analyze(att, img, PAIRS, off, seed=42)
    assert "length_norm" not in res_off and res_off["measures"] == ["dtw", "pattern", "length"]
    assert "length_norm_all_rank" not in rows_off[0] and nm.additions(off) == []
    assert res_off["length"] == res["length"]               # the pre-registered analyses are untouched


def test_pattern_and_length_similarity():
    rng = np.random.default_rng(2)
    p = rng.normal(size=8)
    x = p + 0.01 * rng.normal(size=(30, 8))
    y = 2 * p + 1 + 0.01 * rng.normal(size=(50, 8))
    assert nm.pattern_similarity(x, y) > 0.99
    assert np.isnan(nm.pattern_similarity(np.ones((5, 8)), y))
    assert nm.length_similarity(100, 100) == 0.0
    assert nm.length_similarity(100, 200) == pytest.approx(nm.length_similarity(200, 100))
    assert nm.length_similarity(100, 150) > nm.length_similarity(100, 300)


def test_tts_speech_seconds_from_c0(cfg):
    sil = -12.48528                                     # LPCNet c0 of digital silence (wvx.targets)
    rng = np.random.default_rng(3)
    raw = np.zeros((140, 36), np.float32)
    raw[:, 0] = sil
    raw[20:120, 0] = 2.0 + 0.3 * rng.normal(size=100)   # 100 frames of "speech"
    frame_s = cfg["synthesis"]["samples_per_frame"] / cfg["synthesis"]["sample_rate"]
    assert nm.tts_speech_seconds(raw, cfg, "speech_span", 30.0) == pytest.approx(100 * frame_s)
    assert nm.tts_speech_seconds(raw, cfg, "speech_frames", 30.0) == pytest.approx(100 * frame_s, abs=8 * frame_s)  # smoothing widens edges
    with pytest.raises(ValueError):
        nm.tts_speech_seconds(raw, cfg, "syllables", 30.0)


# --------------------------------------------------------------------------------- ranks, candidates


def test_midranks_with_ties():
    np.testing.assert_allclose(nm.midranks([0.9, 0.5, 0.9, 0.1]), [1.5, 3, 1.5, 4])
    np.testing.assert_allclose(nm.midranks([0.2, 0.2, 0.2]), [2, 2, 2])
    r = nm.midranks(np.random.default_rng(4).integers(0, 4, 11).astype(float))
    assert r.sum() == pytest.approx(11 * 12 / 2)      # mid-ranks always sum to n(n+1)/2


def test_own_rank_ties_and_undefined_scores():
    scores = np.array([0.3, 0.9, 0.9, np.nan, 0.1])
    rank, cr = nm.own_rank(scores, [0, 1, 2, 3, 4], own=1)
    assert rank == 1.5 and cr.size == 4                 # tied for best; the NaN candidate is left out
    assert nm.own_rank(scores, [4, 0], own=4)[0] == 2.0
    assert np.isnan(nm.own_rank(scores, [0, 3], own=3)[0])
    with pytest.raises(ValueError):
        nm.own_rank(scores, [0, 2], own=1)


def test_length_matched_candidates():
    d = [1.0, 2.0, 2.25, 1.75, 5.0, 2.0]
    np.testing.assert_array_equal(nm.length_matched_candidates(d, 1, 3), [1, 5, 2, 3])   # tie 2 vs 3: lower index
    np.testing.assert_array_equal(nm.length_matched_candidates(d, 4, 1), [4, 2])
    c = nm.length_matched_candidates(d, 0, 5)
    assert c[0] == 0 and sorted(c[1:]) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        nm.length_matched_candidates(d, 0, 6)
    rng = np.random.default_rng(5)
    dur = rng.uniform(1, 3, 50)
    for own in range(50):
        c = nm.length_matched_candidates(dur, own, 10)
        assert c[0] == own and own not in c[1:] and len(set(c)) == 11
        far = np.setdiff1d(np.arange(50), c)
        assert np.abs(dur[c[1:]] - dur[own]).max() <= np.abs(dur[far] - dur[own]).min()


# -------------------------------------------------------------------------------------- the null test


def test_permutation_null_is_uniform_over_the_candidates():
    rng = np.random.default_rng(6)
    draws = nm.null_rank_draws([np.arange(1.0, 12.0), np.array([1.5, 1.5, 3.0])], 110_000, rng)
    counts = np.bincount(draws[:, 0].astype(int), minlength=12)[1:]
    np.testing.assert_allclose(counts / draws.shape[0], 1 / 11, atol=0.004)
    assert draws[:, 1].mean() == pytest.approx(2.0, abs=0.01)
    assert set(np.unique(draws[:, 1])) == {1.5, 3.0}


def test_exact_p_matches_enumeration():
    cands = [np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 3.0])]
    combos = [a + b for a, b in itertools.product(*cands)]
    for obs in ([1.0, 1.5], [2.0, 1.5], [3.0, 3.0], [1.0, 3.0]):
        want = np.mean([c <= sum(obs) for c in combos])
        assert nm.exact_mean_rank_p(cands, obs) == pytest.approx(want)


def test_p_values_are_calibrated_under_h0():
    rng = np.random.default_rng(7)
    cand = [np.arange(1.0, 12.0)] * 50
    p_exact, p_perm = [], []
    for _ in range(300):
        obs = rng.integers(1, 12, 50).astype(float)   # own trial = a random candidate: H0
        draws = nm.null_rank_draws(cand, 1000, rng)
        s = nm.rank_stats(obs, cand, draws)
        p_exact.append(s["p_exact"])
        p_perm.append(s["p_perm"])
    p_exact, p_perm = np.array(p_exact), np.array(p_perm)
    assert 0.02 <= np.mean(p_exact <= 0.05) <= 0.09
    assert 0.40 <= np.mean(p_exact <= 0.5) <= 0.62
    assert np.abs(p_perm - p_exact).max() < 0.06       # Monte Carlo error with 1000 relabelings


def test_rank_stats_fields():
    cand = [np.arange(1.0, 12.0)] * 20
    obs = np.ones(20)
    s = nm.rank_stats(obs, cand, nm.null_rank_draws(cand, 10_000, np.random.default_rng(8)))
    assert s["mean_rank"] == 1 and s["chance_mean_rank"] == 6 and s["top1_rate"] == 1
    assert s["top1_chance"] == pytest.approx(1 / 11) and s["mean_normalized_rank"] == 0
    assert s["p_perm"] == pytest.approx(1 / 10_001)     # no relabeling reaches mean rank 1
    assert s["p_exact"] == pytest.approx(11.0 ** -20)
    tied = nm.rank_stats([1.5], [np.array([1.5, 1.5, 3.0])], np.full((10, 1), 1.5))
    assert tied["top1_rate"] == 0 and tied["top1_chance"] == 0


# ------------------------------------------------------------------------------------- end to end


def _synthetic(seed, n_sent, n_feat, planted):
    """Kunz-like trials: 4 block pairs, n_sent cues per pair. Attempted trial = the cue's latent trajectory
    through shared loadings + noise. Imagined = a noisy, nonlinearly time-warped, slower, scaled-down copy of
    it (planted) or pure noise of unrelated length (not planted)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(D_LAT, n_feat))
    att, img = [], []
    for p, (a, i) in enumerate(PAIRS.items()):
        for s in range(n_sent):
            text = f"Cue {p} {s}."
            cue_s = float(rng.uniform(0.8, 3.0))
            f = _smooth_function(rng)
            ta = int(150 * cue_s) + int(rng.integers(0, 60))
            xa = f(np.linspace(0, 1, ta)) @ w + 0.5 * rng.normal(size=(ta, n_feat))
            if planted:
                ti = int(ta * rng.uniform(1.2, 1.6))
                warp = np.linspace(0, 1, ti) ** rng.uniform(0.7, 1.4)
                xi = 0.4 * f(warp) @ w + IMAGINED_NOISE * rng.normal(size=(ti, n_feat))
            else:
                ti = int(rng.integers(150, 700))
                xi = IMAGINED_NOISE * rng.normal(size=(ti, n_feat))
            idx = p * n_sent + s
            att.append(nm.MatchTrial(xa.astype(np.float32), text.lower(), text, a, idx, cue_s))
            img.append(nm.MatchTrial(xi.astype(np.float32), text.lower(), text, i, idx, cue_s))
    return att, img


def _summary_for_markdown(results, settings):
    return {"results": results, "settings": settings, "seed": 42, "deviations_from_preregistration": [],
            "cue_duration": {"method": "speech_span", "voice": "v", "seconds": {"mean": 1.0}, "min_s": 0.5,
                             "max_s": 2.0},
            "limitations": nm.LIMITATIONS, "interpretation_rule": nm.INTERPRETATION,
            "code_version": {"commit": "abc", "dirty": False}, "config_sha256": "0" * 64}


def test_planted_signal_is_identified(settings):
    att, img = _synthetic(10, 20, 64, planted=True)
    rows, res = nm.analyze(att, img, PAIRS, settings, seed=42)
    assert len(rows) == 80
    assert all(r["attempted_trial_index"] == r["imagined_trial_index"] for r in rows)
    prim = res["primary"]["pooled"]
    assert prim["n_trials"] == 80 and prim["mean_n_candidates"] == 11 and prim["chance_mean_rank"] == 6
    assert prim["mean_rank"] < 2.0 and prim["top1_rate"] > 0.6
    assert prim["p_exact"] < 1e-12 and prim["p_perm"] == pytest.approx(1 / 2001)
    every = res["dtw"]["all"]["pooled"]
    assert every["mean_n_candidates"] == 20 and every["mean_rank"] < every["chance_mean_rank"] / 3
    for pair in res["primary"]["per_pair"].values():
        assert pair["n_trials"] == 20 and pair["mean_rank"] < 3 and pair["p_perm"] < 0.01
    assert set(res["pca"]) == {nm.pair_label(a, i) for a, i in PAIRS.items()}
    assert all(v["n_train_trials"] == 120 and v["n_components"] == 32 for v in res["pca"].values())
    md = nm.summary_markdown(_summary_for_markdown(res, settings))
    assert "PRIMARY" in md and "positive control" in md


def test_pure_noise_is_at_chance(settings):
    att, img = _synthetic(11, 20, 64, planted=False)
    rows, res = nm.analyze(att, img, PAIRS, settings, seed=42)
    assert res["measures"] == ["dtw", "pattern", "length", "length_norm"]
    for m in res["measures"]:
        for c in ("length_matched", "all"):
            s = res[m][c]["pooled"]
            assert abs(s["mean_rank"] - s["chance_mean_rank"]) < 0.2 * s["chance_mean_rank"], (m, c, s)
            assert s["p_exact"] > 0.01, (m, c, s)


def test_design_violations_are_refused(settings):
    att, img = _synthetic(12, 12, 40, planted=True)
    bad = list(img)
    bad[1] = nm.MatchTrial(bad[1].x, bad[0].key, bad[0].text, bad[1].block, bad[1].trial_index, bad[1].cue_s)
    with pytest.raises(ValueError, match="more than once"):
        nm.analyze(att, bad, PAIRS, settings, seed=42)
    moved = list(att)
    moved[0] = nm.MatchTrial(moved[0].x, att[12].key, att[12].text, 3, moved[0].trial_index, moved[0].cue_s)
    with pytest.raises(ValueError):
        nm.analyze(moved, img, PAIRS, settings, seed=42)
    with pytest.raises(ValueError, match="do not match"):
        nm.analyze(att, img, {3: 4, 6: 7, 11: 9}, settings, seed=42)


# ------------------------------------------------------------------------------------------- script


def test_script_on_synthetic_files(tmp_path, cfg):
    """scripts/neural_match.py end to end on tiny synthetic SessionData + TTS files (not real data)."""
    from wvx.session import SessionData, Trial
    from wvx.targets import TTSCorpus, corpus_path

    att, img = _synthetic(13, 12, 40, planted=True)
    proc = tmp_path / "processed"
    for cond, trials in (("attempted", att), ("imagined", img)):
        SessionData(f"syn_{cond}", 0, [Trial(neural=t.x, text=t.text, block=t.block,
                                             meta={"trial_index": t.trial_index, "go_offset": 0})
                                       for t in trials]).save(proc / f"{cond}.npz")
    frame_s = cfg["synthesis"]["samples_per_frame"] / cfg["synthesis"]["sample_rate"]
    frames = []
    for t in att:                       # TTS c0: silence, then cue_s seconds of "speech", then silence
        n = int(round(t.cue_s / frame_s))
        raw = np.zeros((n + 40, 36), np.float32)
        raw[:, 0] = -12.48528
        raw[20:20 + n, 0] = 2.0
        frames.append(raw)
    tts_root = tmp_path / "tts"
    TTSCorpus(cfg["tts"]["voice"], [t.text for t in att], frames, np.zeros(36, np.float32)).save(
        corpus_path({"paths": {"tts_features": tts_root}}, cfg["tts"]["voice"], cfg["kunz"]["corpus"]))
    out = tmp_path / "out"
    exp = {"base_experiment": CONFIG, "paths": {"tts_features": str(tts_root)},
           "runtime": {"nice": None},
           "neural_match": {"output_dir": str(out), "attempted_file": str(proc / "attempted.npz"),
                            "imagined_file": str(proc / "imagined.npz"), "n_permutations": 500}}
    cfg_file = tmp_path / "neural_match_test.yaml"
    cfg_file.write_text(yaml.safe_dump(exp))
    run = subprocess.run([sys.executable, "-I", str(REPO_ROOT / "scripts" / "neural_match.py"), "--config",
                          str(cfg_file)], capture_output=True, text=True, cwd=REPO_ROOT)
    assert run.returncode == 0, run.stderr[-3000:]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["deviations_from_preregistration"] == [
        {"key": "n_permutations", "preregistered": 10000, "used": 500}]
    assert [a["key"] for a in summary["additions_to_preregistration"]] == [
        "additional_baselines.length_block_normalized"]
    assert set(summary["results"]["length_norm"]) == {"length_matched", "all"}
    assert len(summary["config_sha256"]) == 64 and "commit" in summary["code_version"]
    assert summary["runtime"]["env"]["OMP_NUM_THREADS"] == str(cfg["runtime"]["threads"]["omp"])
    assert summary["runtime"]["tensorflow_loaded"] is False
    assert summary["cue_duration"]["n_cues"] == 48
    prim = summary["results"]["primary"]["pooled"]
    assert prim["n_trials"] == 48 and prim["mean_rank"] < prim["chance_mean_rank"]
    lines = (out / "per_trial.csv").read_text().splitlines()
    assert len(lines) == 49 and "dtw_length_matched_rank" in lines[0] and "length_norm_all_rank" in lines[0]
    md = (out / "summary.md").read_text()
    assert "n_permutations = 500 (pre-registered 10000)" in md and "positive control" in md
    assert "raw (pre-registered), candidates: length-matched" in md
    assert "block-normalized (added 2026-10-21 before any run), candidates: all" in md
    assert "Additions to the pre-registration" in md
