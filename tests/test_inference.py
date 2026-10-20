"""wvx.inference and scripts/run_inference.py with a tiny model built by THEIR code.

A tiny Voiceformer (their brain2voice, 1 block, 1 head) is exported with their
``model_utils.save_inference_model`` to ``.h5`` -- the format of the Dryad models -- then loaded
with our wrapper (their ``load_b2voice_model``) and checked against the Voiceformer itself and
against their ``run_inference_single_trial``.
"""

import importlib.util
import json
import sys

import numpy as np
import pytest

from wvx import b2v
from wvx.config import REPO_ROOT, deep_merge, load_config

pytestmark = [
    pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py"),
    # find_spec, not importorskip: importing TensorFlow before wvx.b2v would load Keras 3.
    pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed"),
]

EXPERIMENT = "configs/experiments/reproduce_inference.yaml"
TINY_LAYERS = 2  # day layers in the tiny Voiceformer
BAKED_LAYER = 1  # the one exported into the .h5


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """(Voiceformer, path of its exported .h5 inference model), built with their code."""
    brain2voice = b2v.load("models.brain2voice")
    model_utils = b2v.load("lib.model_utils")
    import tensorflow as tf

    cfg = load_config(EXPERIMENT)
    m = cfg["model"]
    tf.random.set_seed(cfg["seed"])
    # Their day layer is fixed at (30, 512) -> 128, so the base must take (30, 128) with head_size 128.
    base = brain2voice.base_transformer(
        transformer_input_shape=(m["window_len"] // 2, 128),
        output_dim=m["n_targets"],
        head_size=128,
        n_heads=1,
        n_transformer_blocks=1,
        mlp_units=[16],
        dropout=0.0,
    )
    vf = brain2voice.Voiceformer(base, n_sessions=TINY_LAYERS)
    x_tok = np.zeros((2, m["window_len"], m["n_features"]), np.float32)
    vf([x_tok, np.full((2, 1), BAKED_LAYER, np.int32)])  # warm-up build, as their build_model does
    d = tmp_path_factory.mktemp("model")
    model_utils.save_inference_model(vf, x_tok, str(d / "tiny.tf"), day_embed_layer=BAKED_LAYER)
    return vf, d / "tiny.h5"


@pytest.fixture
def cfg():
    return load_config(EXPERIMENT)


def _neural(n_bins, seed=0):
    return np.random.default_rng(seed).normal(size=(n_bins, 512)).astype(np.float32)


def _voiceformer_reference(vf, neural, cfg, layer):
    from wvx.inference import sliding_windows

    win = sliding_windows(neural, cfg)
    return vf([win, np.full((len(win), 1), layer, np.int32)], training=False).numpy()


def test_h5_round_trip_matches_voiceformer_layer(tiny, cfg):
    from wvx.inference import load_model, predict_trial

    vf, h5 = tiny
    model = load_model(h5, cfg)
    neural = _neural(83)
    pred = predict_trial(model, neural, cfg, reorder=False)
    assert pred.shape == (83 - 59, 20) and pred.dtype == np.float32
    np.testing.assert_allclose(pred, _voiceformer_reference(vf, neural, cfg, BAKED_LAYER), atol=1e-5)
    # ...and it is really the baked layer, not layer 0.
    assert np.abs(pred - _voiceformer_reference(vf, neural, cfg, 0)).max() > 1e-3


@pytest.mark.parametrize("reorder", [False, True])
def test_matches_their_inference(tiny, cfg, reorder):
    from wvx import synthesis
    from wvx.inference import load_model, predict_trial

    inference_utils = b2v.load("lib.inference_utils")
    data_utils = b2v.load("lib.data_utils")
    model = load_model(tiny[1], cfg)
    neural = _neural(97, seed=1)
    sp, tc = neural[:, :256], neural[:, 256:]
    mapping = str(cfg["paths"]["electrode_mapping"])
    if reorder:  # exactly their inference.ipynb cell 7
        sp = data_utils.reorder_channels(sp.copy(), mapping)
        tc = data_utils.reorder_channels(tc.copy(), mapping)
    lo, hi = data_utils.load_lpc_norm(str(cfg["paths"]["lpc_norm"]))
    theirs = inference_utils.run_inference_single_trial(model, sp, tc, cfg["model"]["window_len"], lo, hi)
    ours = synthesis.denormalize(predict_trial(model, neural, cfg, reorder=reorder), cfg)
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-5)


def test_reorder_is_double_gather_per_block(cfg):
    from wvx.inference import reorder_channels

    with open(cfg["paths"]["electrode_mapping"]) as f:
        m = np.array(json.load(f)["electrode_mapping"]) - 1
    x = np.tile(np.arange(512, dtype=np.float32), (4, 1))  # column j holds the value j
    out = reorder_channels(x, cfg["paths"]["electrode_mapping"])
    np.testing.assert_array_equal(out[0, :256], m[m])
    np.testing.assert_array_equal(out[0, 256:], 256 + m[m])
    # Pitfall their docstring doesn't mention: given all 512 columns their function silently
    # returns 256 spike-power columns. Our wrapper refuses widths that don't match the mapping.
    data_utils = b2v.load("lib.data_utils")
    assert data_utils.reorder_channels(x, str(cfg["paths"]["electrode_mapping"])).shape == (4, 256)
    with pytest.raises(ValueError, match="mapping"):
        reorder_channels(x[:, :384], cfg["paths"]["electrode_mapping"])


def test_window_edge_cases_and_batch_invariance(tiny, cfg):
    from wvx.inference import load_model, predict_trial

    model = load_model(tiny[1], cfg)
    one = predict_trial(model, _neural(60), cfg, reorder=False)  # their np.squeeze breaks here
    assert one.shape == (1, 20)
    with pytest.raises(ValueError, match="shorter"):
        predict_trial(model, _neural(59), cfg, reorder=False)
    neural = _neural(140, seed=2)
    a = predict_trial(model, neural, cfg, reorder=False)
    b = predict_trial(model, neural, deep_merge(cfg, {"inference": {"batch_size": 7}}), reorder=False)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_model_shape_check(tiny, cfg):
    from wvx.inference import load_model

    with pytest.raises(ValueError, match="config expects"):
        load_model(tiny[1], deep_merge(cfg, {"model": {"window_len": 30}}))
    with pytest.raises(FileNotFoundError):
        load_model(tiny[1].with_name("missing.h5"), cfg)


def test_align_to_predictions():
    from wvx.inference import align_to_predictions

    y = np.arange(100)[:, None]
    np.testing.assert_array_equal(align_to_predictions(y, 41)[:, 0], np.arange(59, 100))
    with pytest.raises(ValueError):
        align_to_predictions(y, 0)


# ------------------------------------------------------------------------- the script


def _script():
    spec = importlib.util.spec_from_file_location("run_inference", REPO_ROOT / "scripts" / "run_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _script_cfg(tmp_path, h5, synthesize):
    from test_wairagkar_data import write_session

    cfg = load_config(EXPERIMENT)
    write_session(tmp_path / "data", "t15_day195", cfg, n_bins=(75, 64, 90))
    over = {
        "run_name": "test_run",
        "paths": {
            "wairagkar_neural_data": tmp_path / "data",
            "pretrained_model": h5,
            "results": tmp_path / "results",
        },
        "inference": {"synthesize": synthesize},
        "synthesis": {"tmp_root": str(tmp_path / "tmp")},
    }
    return deep_merge(cfg, over)


def _lpcnet_runs(cfg):
    from wvx import synthesis

    try:
        synthesis.preflight(cfg)
    except synthesis.LPCNetError:
        return False
    return True


def test_run_inference_end_to_end(tiny, tmp_path, monkeypatch):
    import types

    import pandas as pd

    calls = []

    def evaluate(ref_wav, est_wav, sr, cfg_metrics):
        calls.append((ref_wav.shape, est_wav.shape, sr))
        return {"fake_score": float(np.mean(est_wav**2))}

    monkeypatch.setitem(sys.modules, "wvx.metrics", types.SimpleNamespace(evaluate=evaluate))
    cfg = _script_cfg(tmp_path, tiny[1], synthesize=True)
    if not _lpcnet_runs(cfg):
        cfg = deep_merge(cfg, {"inference": {"synthesize": False}})
    summary = _script().run(cfg, config_path=EXPERIMENT)

    out = tmp_path / "results" / "test_run"
    df = pd.read_csv(out / "per_trial_metrics.csv")
    assert list(df["trial"]) == [0, 1, 2]
    assert list(df["n_frames"]) == [75 - 59, 64 - 59, 90 - 59]
    assert df["feat_mse"].notna().all() and df["text"][0] == "sentence number 0"
    saved = json.loads((out / "summary.json").read_text())
    assert saved["layer"] == 20 and saved["n_trials_run"] == 3 and saved["reorder_channels"] is True
    assert saved["data_checks"]["n_trials"] == 3
    preds = np.load(tmp_path / "results" / "audio" / "test_run" / "predictions.npz")
    assert preds["tr2"].shape == (90 - 59, 20)
    if cfg["inference"]["synthesize"]:
        import soundfile

        assert summary["metrics_available"] and "fake_score" in df.columns
        assert calls == [((n * 160,), (n * 160,), 16000) for n in df["n_frames"]]
        wav, sr = soundfile.read(tmp_path / "results" / "audio" / "test_run" / "t15_day195_tr1_target.wav")
        assert sr == 16000 and wav.shape == (5 * 160,)


def test_run_inference_without_metrics_module(tiny, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wvx.metrics", None)  # import now raises ModuleNotFoundError
    cfg = _script_cfg(tmp_path, tiny[1], synthesize=False)
    summary = _script().run(deep_merge(cfg, {"inference": {"max_trials": 1}}))
    assert summary["metrics_available"] is False and summary["n_trials_run"] == 1
    assert any("wvx.metrics" in w for w in summary["warnings"])
