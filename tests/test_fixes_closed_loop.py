"""scripts/compare_closed_loop.py lag search (2026-10-21 audit: the configured lead-in trim looked 2 frames short).
Synthetic signals only; the real comparison is a host job."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from wvx import metrics
from wvx.config import REPO_ROOT, load_config

SR, HOP = 16000, 160


@pytest.fixture(scope="module")
def ccl():
    spec = importlib.util.spec_from_file_location("fixes_ccl", REPO_ROOT / "scripts" / "compare_closed_loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _speechlike(n_frames: int, seed: int) -> np.ndarray:
    """Noise with a random 10 ms-frame envelope: frame-wise mel_corr is sharply peaked at the true alignment."""
    rng = np.random.default_rng(seed)
    env = np.repeat(rng.random(n_frames) ** 3, HOP)
    return 0.3 * env * rng.standard_normal(env.size)


def test_framewise_cfg_reproduces_evaluate_framewise(ccl):
    mcfg = metrics.load_metrics_config()
    a, b = _speechlike(300, 1), _speechlike(280, 2)
    b[: a.size // 2] += 0.5 * a[: a.size // 2]
    want = metrics.evaluate(a, b, SR, mcfg)["mel_corr_framewise"]
    assert ccl.lag_scan(a, b, SR, 0, HOP, [0], ccl.framewise_cfg(mcfg))[0] == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("true_lag", [-3, 0, 2, 5])
def test_lag_scan_finds_a_known_offset(ccl, true_lag):
    lead_frames = 59
    pred = _speechlike(400, 3)
    # Live stream = (lead-in + true_lag frames of other audio) + the same audio our decoder produced.
    live = np.concatenate([_speechlike(lead_frames + true_lag, 4), pred])
    curve = ccl.lag_scan(live, pred, SR, lead_frames * HOP, HOP, range(-8, 9), ccl.framewise_cfg(
        metrics.load_metrics_config()))
    assert set(curve) == set(range(-8, 9))
    assert ccl.best_lag(curve) == true_lag
    assert curve[true_lag] > 0.99
    if true_lag:
        assert curve[true_lag] > curve[0]


def test_lag_scan_skips_trims_outside_the_clip(ccl):
    live = _speechlike(10, 5)
    curve = ccl.lag_scan(live, live, SR, 2 * HOP, HOP, range(-5, 6), ccl.framewise_cfg(metrics.load_metrics_config()))
    assert min(curve) == -2 and max(curve) == 5   # start >= 0 and < len(live)


def test_best_lag_tie_breaks_and_nan(ccl):
    assert ccl.best_lag({-1: 0.9, 1: 0.9, 0: 0.8}) == -1
    assert ccl.best_lag({0: 0.9, 2: 0.9, -2: 0.9}) == 0
    assert ccl.best_lag({0: float("nan"), 3: 0.1}) == 3
    assert ccl.best_lag({0: float("nan")}) is None and ccl.best_lag({}) is None


def test_mean_by_lag_uses_lags_scored_in_every_trial(ccl):
    curves = [{-1: 0.5, 0: 0.6, 1: 0.7}, {0: 0.8, 1: 0.9, 2: 1.0}, {0: 0.7, 1: float("nan")}]
    assert ccl.mean_by_lag(curves) == {0: pytest.approx(0.7)}
    assert ccl.mean_by_lag([]) == {}


def test_lag_search_is_configured():
    cl = load_config("configs/experiments/reproduce_inference.yaml")["closed_loop"]
    assert isinstance(cl["lag_search_frames"], int) and cl["lag_search_frames"] >= 2
    assert cl["bin_ms"] * cl["sample_rate"] // 1000 == HOP


def test_compare_closed_loop_end_to_end_on_synthetic_files(ccl, tmp_path, monkeypatch):
    """Tiny fake session: live audio = lead-in + 2 extra frames + exactly our offline audio. The configured-trim
    columns are computed as before; the lag search reports +2 and agreement ~1 there."""
    import csv
    import json
    import sys

    import scipy.io
    import soundfile as sf
    import yaml

    from wvx.config import deep_merge, load_yaml

    exp = load_yaml(REPO_ROOT / "configs" / "experiments" / "reproduce_inference.yaml")
    session, lead_frames = exp["inference"]["session"], 59
    exp = deep_merge(exp, {"run_name": "run", "paths": {"results": str(tmp_path / "results"),
                                                        "wairagkar_neural_data": str(tmp_path / "data")},
                           "closed_loop": {"lag_search_frames": 4}})
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(exp))
    run = tmp_path / "results" / "run"
    run.mkdir(parents=True)
    (tmp_path / "data" / session).mkdir(parents=True)

    live = np.empty((1, 3), dtype=object)
    rows = []
    for i in range(3):
        pred, tgt = _speechlike(300, 10 + i), _speechlike(300, 20 + i)
        sf.write(run / f"p{i}.wav", pred, SR, subtype="FLOAT")
        sf.write(run / f"t{i}.wav", tgt, SR, subtype="FLOAT")
        live[0, i] = np.zeros(0) if i == 0 else np.concatenate([_speechlike(lead_frames + 2, 30 + i), pred]) * 32768
        rows.append({"trial": i, "text": f"s{i}", "mel_corr": 0.5, "pred_wav": str(run / f"p{i}.wav"),
                     "target_wav": str(run / f"t{i}.wav")})
    with open(run / "per_trial_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cl = exp["closed_loop"]
    scipy.io.savemat(tmp_path / "data" / session / exp["wairagkar_data"]["neural_file"].format(session=session),
                     {cl["audio_key"]: live,
                      cl["description_key"]: np.array(["Training data   ", "Evaluation trial", "Evaluation trial"])})

    seeds = []
    monkeypatch.setattr(ccl, "set_seed", seeds.append)   # wvx.seeding would import TensorFlow (Keras 3) here
    monkeypatch.setattr(sys, "argv", ["compare_closed_loop.py", "--config", str(tmp_path / "exp.yaml")])
    assert ccl.main() == 0 and seeds == [42]

    with open(run / "closed_loop_comparison.csv") as f:
        out = {int(r["trial"]): r for r in csv.DictReader(f)}
    assert out[0]["ours_vs_live_mel_corr"] == "" and out[0]["best_lag_frames_trial"] == ""
    mcfg = metrics.load_metrics_config()
    for i in (1, 2):
        audio = np.asarray(live[0, i], float)
        pred = sf.read(run / f"p{i}.wav")[0]
        want = metrics.evaluate(audio[lead_frames * HOP:], pred, SR, mcfg)
        assert float(out[i]["ours_vs_live_mel_corr"]) == pytest.approx(want["mel_corr"], rel=1e-12)
        assert float(out[i]["ours_vs_live_mel_corr_framewise"]) == pytest.approx(want["mel_corr_framewise"], rel=1e-12)
        assert int(out[i]["best_lag_frames_trial"]) == 2
        assert float(out[i]["ours_vs_live_mel_corr_framewise_best_lag"]) > 0.99 > float(
            out[i]["ours_vs_live_mel_corr_framewise"])
    s = json.loads((run / "closed_loop_summary.json").read_text())
    assert s["_lag_search"]["best_lag_frames"] == 2 and s["_lag_search"]["best_trim_frames"] == 61
    assert s["_lag_search"]["configured_trim_frames"] == 59 and s["_lag_search"]["n_trials"] == 2
    assert set(s["_lag_search"]["mean_framewise_by_lag"]) == {str(k) for k in range(-4, 5)}
    assert s["Evaluation trial"]["ours_vs_live_mel_corr"]["n"] == 2
    assert s["Evaluation trial"]["ours_vs_live_mel_corr_framewise_best_lag"]["mean"] > 0.99
    assert s["Training data"] == {"ours_vs_target_mel_corr": {"n": 1, "mean": 0.5, "median": 0.5, "sd": 0.0}}
    p = s["_provenance"]
    assert p["code_version"]["source"] in ("git", "sync", None) and len(p["config_sha256"]) == 64
    assert len(p["inputs"]["per_trial_metrics_csv"]["sha256"]) == 64
