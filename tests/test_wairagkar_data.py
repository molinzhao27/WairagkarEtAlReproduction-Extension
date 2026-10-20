"""wvx.wairagkar_data on synthetic .mat files built to match what their loaders read.

The fixture mirrors their code (lib/dataloader.py:58-80, lib/inference_utils.py:38-51): per-session
folder, ``<s>_neural_data.mat`` with cell arrays ``spikepow_trials``/``threshcross_trials`` (T x 256
each), space-padded char-matrix ``cues``, a ``session`` string, and ``<s>_target_speech_feats.mat``
with a cell array ``target_speech_feats_word`` of T x {20, 36} LPCNet features.
"""

import re

import numpy as np
import pytest
import scipy.io

from wvx import b2v
from wvx.config import load_config

pytestmark = pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py")

EXPERIMENT = "configs/experiments/reproduce_inference.yaml"


def _cells(arrays):
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def write_session(
    root,
    session,
    cfg,
    n_bins=(75, 64, 90),
    target_cols=20,
    cue_key="cues",
    with_target=True,
    target_extra_rows=0,
    dtype=np.float64,
    seed=0,
):
    """Write one synthetic session in their release layout; return the arrays written."""
    rng = np.random.default_rng(seed)
    n_ch = cfg["model"]["n_features"] // 2
    keys = cfg["wairagkar_data"]["keys"]
    spk = [rng.normal(size=(t, n_ch)).astype(dtype) for t in n_bins]
    thr = [rng.normal(size=(t, n_ch)).astype(dtype) for t in n_bins]
    texts = [f"sentence number {i}" + "!" * i for i in range(len(n_bins))]
    d = root / session
    d.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        d / f"{session}_neural_data.mat",
        {keys["spikepow"]: _cells(spk), keys["threshcross"]: _cells(thr), cue_key: np.array(texts), "session": session},
    )
    lo, hi = b2v.load("lib.data_utils").load_lpc_norm(str(cfg["paths"]["lpc_norm"]))
    lpc_rng = np.random.default_rng([seed, target_cols])  # extra LPC columns don't shift `rng`
    tgt = []
    for t in n_bins:
        y = rng.uniform(lo, hi, size=(t + target_extra_rows, len(lo)))
        if target_cols > len(lo):
            y = np.concatenate([y, lpc_rng.normal(size=(y.shape[0], target_cols - len(lo)))], axis=1)
        tgt.append(y)
    if with_target:
        scipy.io.savemat(d / f"{session}_target_speech_feats.mat", {keys["target"]: _cells(tgt)})
    return {"spk": spk, "thr": thr, "texts": texts, "targets": tgt}


@pytest.fixture
def cfg():
    return load_config(EXPERIMENT)


def test_matches_their_load_neural_trials(tmp_path, cfg):
    from wvx.wairagkar_data import load_raw_targets, load_session

    write_session(tmp_path, "t15_day193", cfg)
    sess = load_session(tmp_path, "t15_day193", 20, cfg)
    raw = load_raw_targets(tmp_path, "t15_day193", cfg)

    dataloader = b2v.load("lib.dataloader")
    data_utils = b2v.load("lib.data_utils")
    lo, hi = data_utils.load_lpc_norm(str(cfg["paths"]["lpc_norm"]))
    theirs = dataloader.load_neural_trials(str(tmp_path), ["t15_day193"], [20])

    assert len(sess.trials) == len(theirs) == 3
    for ours, raw_y, th in zip(sess.trials, raw, theirs):
        np.testing.assert_array_equal(ours.neural, th["X"].astype(np.float32))
        assert ours.text == th["sentence"]
        np.testing.assert_array_equal(
            ours.target, data_utils.get_speech_features_from_single_trial(th["Y"], lo, hi).astype(np.float32)
        )
        np.testing.assert_array_equal(raw_y, th["Y"][:, :20].astype(np.float32))
        assert sess.layer == th["session_num"]
    assert sess.trials[0].neural.dtype == np.float32 and sess.trials[0].neural.shape == (75, 512)


def test_spikepow_block_comes_first(tmp_path, cfg):
    from wvx.wairagkar_data import load_session

    arrays = write_session(tmp_path, "t15_day193", cfg)
    t = load_session(tmp_path, "t15_day193", 20, cfg).trials[1]
    np.testing.assert_array_equal(t.neural[:, :256], arrays["spk"][1].astype(np.float32))
    np.testing.assert_array_equal(t.neural[:, 256:], arrays["thr"][1].astype(np.float32))


def test_session_layers_from_their_yaml(cfg):
    from wvx.wairagkar_data import layer_for_session

    assert layer_for_session("t15_day025", cfg) == 0
    assert layer_for_session("t15_day110", cfg) == 10
    assert layer_for_session("t15_day193", cfg) == 20
    assert layer_for_session("t15_day195", cfg) == 20  # held out: from our config
    with pytest.raises(KeyError):
        layer_for_session("t15_day999", cfg)


def test_held_out_layer_matches_their_comment(cfg):
    text = cfg["paths"]["b2v_session_layers"].read_text()
    m = re.search(r"(t15_day\d+) \(session_layer (\d+)\) is held out", text)
    assert m, "their training_data_args.yaml comment changed"
    assert cfg["wairagkar_data"]["held_out_session_layers"] == {m.group(1): int(m.group(2))}


def test_36_column_targets_give_the_same_20(tmp_path, cfg):
    from wvx.wairagkar_data import load_session

    write_session(tmp_path / "a", "t15_day195", cfg, target_cols=20, seed=3)
    write_session(tmp_path / "b", "t15_day195", cfg, target_cols=36, seed=3)
    a = load_session(tmp_path / "a", "t15_day195", 20, cfg)
    b = load_session(tmp_path / "b", "t15_day195", 20, cfg)
    for ta, tb in zip(a.trials, b.trials):
        np.testing.assert_array_equal(ta.target, tb.target)
        assert tb.meta["target_columns"] == 36 and ta.meta["target_columns"] == 20


def test_sentences_fallback_and_cue_padding_stripped(tmp_path, cfg):
    from wvx.wairagkar_data import load_session

    arrays = write_session(tmp_path, "t15_day195", cfg, cue_key="sentences")
    sess = load_session(tmp_path, "t15_day195", 20, cfg)
    assert [t.text for t in sess.trials] == arrays["texts"]
    raw = scipy.io.loadmat(tmp_path / "t15_day195" / "t15_day195_neural_data.mat", squeeze_me=True)
    assert raw["sentences"][0].endswith(" "), "fixture should store padded cues, like their release"


def test_target_length_mismatch_raises(tmp_path, cfg):
    from wvx.wairagkar_data import load_session

    write_session(tmp_path, "t15_day195", cfg, target_extra_rows=1)
    with pytest.raises(ValueError, match="target shape"):
        load_session(tmp_path, "t15_day195", 20, cfg)


def test_missing_target_file_modes(tmp_path, cfg):
    from wvx.config import deep_merge
    from wvx.wairagkar_data import load_session

    write_session(tmp_path, "t15_day195", cfg, with_target=False)
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path, "t15_day195", 20, cfg)
    for mode in ("optional", "skip"):
        c = deep_merge(cfg, {"wairagkar_data": {"targets": mode}})
        sess = load_session(tmp_path, "t15_day195", 20, c)
        assert all(t.target is None for t in sess.trials)


def test_single_trial_session(tmp_path, cfg):
    # scipy's squeeze_me turns a 1x1 cell into its content; still one (T, 512) trial.
    from wvx.wairagkar_data import load_session

    write_session(tmp_path, "t15_day195", cfg, n_bins=(70,))
    sess = load_session(tmp_path, "t15_day195", 20, cfg)
    assert len(sess.trials) == 1
    assert sess.trials[0].neural.shape == (70, 512) and sess.trials[0].target.shape == (70, 20)
    assert sess.trials[0].text == "sentence number 0"


def test_float32_source_and_target_scaling(tmp_path, cfg):
    from wvx.wairagkar_data import describe_session, load_raw_targets, load_session

    write_session(tmp_path, "t15_day195", cfg, dtype=np.float32)
    sess = load_session(tmp_path, "t15_day195", 20, cfg)
    assert sess.trials[0].meta["neural_dtype"] == "float32"
    d = describe_session(sess, cfg, load_raw_targets(tmp_path, "t15_day195", cfg))
    assert d["n_trials"] == 3 and d["targets"]["columns"] == [20]
    # Fixture targets are drawn inside the JSON min/max: after their scaling all lie in [-1, 1].
    assert d["raw_targets"]["fraction_outside_unit_range_after_scaling"] == 0.0
    assert d["targets"]["training_target_fraction_outside_unit_range"] == 0.0
    assert set(d["neural_moments"]) == {"spikepow", "threshcross"}


def test_inspect_session_files_lists_variables(tmp_path, cfg):
    from wvx.wairagkar_data import inspect_session_files

    write_session(tmp_path, "t15_day195", cfg)
    info = inspect_session_files(tmp_path, "t15_day195", cfg)
    names = {v["name"] for v in info["neural"]["variables"]}
    assert {"spikepow_trials", "threshcross_trials", "cues", "session"} <= names
    assert info["target"]["variables"][0]["name"] == "target_speech_feats_word"


def test_trial_count_mismatch_raises(tmp_path, cfg):
    from wvx.wairagkar_data import load_session

    write_session(tmp_path, "t15_day195", cfg)
    p = tmp_path / "t15_day195" / "t15_day195_neural_data.mat"
    m = scipy.io.loadmat(p)
    m["cues"] = m["cues"][:2]
    scipy.io.savemat(p, {k: v for k, v in m.items() if not k.startswith("__")})
    with pytest.raises(ValueError, match="cue trials"):
        load_session(tmp_path, "t15_day195", 20, cfg)


def test_session_survives_save_and_load(tmp_path, cfg):
    # SessionData.save writes Trial.meta as JSON, so the loader must keep meta JSON-safe.
    from wvx.session import SessionData
    from wvx.wairagkar_data import load_session

    write_session(tmp_path, "t15_day195", cfg)
    sess = load_session(tmp_path, "t15_day195", 20, cfg)
    sess.save(tmp_path / "s.npz")
    back = SessionData.load(tmp_path / "s.npz")
    assert back.layer == 20 and [t.text for t in back.trials] == [t.text for t in sess.trials]
    for a, b in zip(sess.trials, back.trials):
        np.testing.assert_array_equal(a.neural, b.neural)
        np.testing.assert_array_equal(a.target, b.target)
        assert a.meta == b.meta
