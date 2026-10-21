"""Core-experiment logic on tiny synthetic Kunz-like sessions (CPU, no TensorFlow).

Fold construction (pairing, no cue overlap between training and the paired imagined test), one
consistent channel reorder for training and decoding, deterministic nulls, the config chain.
"""

import json

import numpy as np
import pytest

from wvx import b2v
from wvx.config import REPO_ROOT
from wvx.experiment_config import load_experiment_config, resolve_chain
from wvx.session import SessionData, Trial

pytestmark = pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py")

from wvx import kunz_experiment as kx  # noqa: E402
from wvx.metrics import normalize_text  # noqa: E402

PILOT = "configs/experiments/kunz_pilot.yaml"
FULL = "configs/experiments/kunz_full.yaml"
PAIRS = {3: 4, 6: 7, 11: 9, 13: 12}
K = 4                 # trials per block
F, NT, W = 512, 20, 60


@pytest.fixture(scope="module")
def cfg():
    return load_experiment_config(PILOT)


def _trial(rng, text, block, idx, target=True):
    t = int(rng.integers(W + 5, W + 30))
    return Trial(neural=rng.standard_normal((t, F)).astype(np.float32),
                 target=rng.uniform(-1, 1, (t, NT)).astype(np.float32) if target else None,
                 text=text, block=block, meta={"trial_index": idx})


@pytest.fixture(scope="module")
def data():
    """Attempted (with targets), imagined (same cues per paired block, other order), eval (one cue reused)."""
    rng = np.random.default_rng(0)
    att, img, idx_a, idx_i = [], [], 0, 0
    for a, i in PAIRS.items():
        cues = [f"Sentence {a} number {j}." for j in range(K)]
        for c in cues:
            att.append(_trial(rng, c, a, idx_a))
            idx_a += 1
        for c in cues[::-1]:
            img.append(_trial(rng, c, i, idx_i, target=False))
            idx_i += 1
    ev = [_trial(rng, f"Novel cue {j}", 14, j, target=False) for j in range(K - 1)]
    ev.append(_trial(rng, "sentence 6 NUMBER 0", 14, K - 1, target=False))   # = an A6/I7 cue, normalized
    return (SessionData("att", 0, att), SessionData("img", 0, img), SessionData("ev", 0, ev))


def _fold(data, cfg, block):
    return kx.build_fold(*data, block, cfg, normalize=normalize_text)


# ------------------------------------------------------------------------------------- config


def test_pilot_and_full_configs(cfg):
    e, t = cfg["experiment"], cfg["training"]
    assert e["folds"] == [3] and e["run_variants"] == ["frozen"] and t["epochs"] == 5
    assert e["reorder_channels"] is True and {int(a): int(i) for a, i in e["pairs"].items()} == PAIRS
    assert e["variants"]["frozen"]["freeze_base"] is True
    assert cfg["runtime"]["device"] == "cpu"
    assert cfg["runtime"]["threads"] == {"intra_op": 12, "inter_op": 2, "omp": 12}
    assert cfg["runtime"]["nice"] == 19
    assert t["init"]["source"] == "inference_h5" and t["init"]["path"].endswith("brain2voice_t15_day193.h5")
    assert cfg["model"]["n_session_layers"] == 1
    assert t["batch_size"] == 1024 and t["n_augmentations"] == 3   # theirs, via reproduce_training.yaml
    assert cfg["paths"]["electrode_mapping"].is_file()               # via kunz_targets -> reproduce_inference
    assert cfg["tts"]["voice"] and cfg["kunz"]["corpus"] and "synthesis" in cfg
    assert t["validation"]["pool"] is None and t["sessions"] == []

    full = load_experiment_config(FULL)
    assert full["experiment"]["folds"] == [3, 6, 11, 13]
    assert full["experiment"]["run_variants"] == ["frozen", "finetuned"]
    assert full["experiment"]["variants"]["finetuned"]["freeze_base"] is False


def test_base_experiment_list_is_merged_in_order(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\ny: {p: 1, q: 1}\n")
    (tmp_path / "b.yaml").write_text(f"base_experiment: {tmp_path / 'a.yaml'}\ny: {{q: 2}}\nz: 2\n")
    (tmp_path / "c.yaml").write_text("x: 3\n")
    (tmp_path / "d.yaml").write_text(json.dumps({"base_experiment": [str(tmp_path / "b.yaml"), str(tmp_path / "c.yaml")],
                                                 "z": 4}))
    assert resolve_chain(tmp_path / "d.yaml") == {"x": 3, "y": {"p": 1, "q": 2}, "z": 4}


# -------------------------------------------------------------------------------------- folds


@pytest.mark.parametrize("block", sorted(PAIRS))
def test_fold_pairing_and_no_cue_leakage(data, cfg, block):
    fold = _fold(data, cfg, block)
    paired = PAIRS[block]
    assert fold.paired_block == paired
    assert fold.train_blocks == [b for b in sorted(PAIRS) if b != block]
    assert len(fold.train_trials) == 3 * K and {t.block for t in fold.train_trials} == set(fold.train_blocks)

    by_role = {}
    for s in fold.sets:
        by_role.setdefault(s.role, []).append(s)
    assert [len(by_role[r]) for r in kx.ROLES] == [1, 1, 1, 3]
    held, pair = by_role["heldout_attempted"][0], by_role["paired_imagined"][0]
    assert (held.block, pair.block, held.condition, pair.condition) == (block, paired, "attempted", "imagined")

    train_cues = {normalize_text(t.text) for t in fold.train_trials}
    assert not train_cues & {normalize_text(c) for c in pair.texts}      # the key test: unseen cues
    assert not train_cues & {normalize_text(c) for c in held.texts}
    assert sorted(held.texts) == sorted(pair.texts)                       # same sentences, both modes
    assert not any(pair.text_in_training) and not any(held.text_in_training)

    leaky = by_role["leaky_imagined"]
    assert sorted(s.block for s in leaky) == sorted(PAIRS[a] for a in fold.train_blocks)
    assert all(all(s.text_in_training) for s in leaky)                    # labelled sentence-leaky

    ev = by_role["eval_imagined"][0]
    assert ev.block == 14
    assert ev.text_in_training == [False] * (K - 1) + [block != 6]       # reused A6 cue flagged only if trained on


def test_wrong_pairing_or_blocks_raise(data, cfg):
    bad = {**cfg, "experiment": {**cfg["experiment"], "pairs": {3: 7, 6: 4, 11: 9, 13: 12}}}
    with pytest.raises(ValueError, match="pairing"):
        _fold(data, bad, 3)
    with pytest.raises(ValueError, match="not an attempted block"):
        _fold(data, cfg, 4)


def test_paired_cue_in_training_is_refused(data, cfg):
    """A cue repeated in another (consistent) block pair would leak into training: refused."""
    att, img, ev = data

    def retext(trials, old, new):
        return [Trial(neural=t.neural, target=t.target, text=new if t.text == old else t.text, block=t.block,
                      meta=t.meta) for t in trials]

    old, new = att.trials[K].text, att.trials[0].text          # an A6/I7 cue becomes an A3/I4 cue
    a2, i2 = SessionData("att", 0, retext(att.trials, old, new)), SessionData("img", 0, retext(img.trials, old, new))
    for block in (3, 6):
        with pytest.raises(ValueError, match="also occur in training"):
            _fold((a2, i2, ev), cfg, block)
    _fold((a2, i2, ev), cfg, 11)  # the repeated cue is only in training blocks here: allowed


# ----------------------------------------------------------------------------- channel reorder


class LastBinModel:
    """Stands in for the decoder: output = first 20 features of each window's last bin."""

    def predict_on_batch(self, x):
        return np.asarray(x)[:, -1, :NT]


@pytest.mark.parametrize("reorder", [True, False])
def test_one_reorder_for_training_and_decoding(data, cfg, reorder):
    from wvx.inference import reorder_channels

    c = {**cfg, "experiment": {**cfg["experiment"], "reorder_channels": reorder}}
    fold = _fold(data, c, 3)
    mapping = c["paths"]["electrode_mapping"]

    def expected(x):
        return reorder_channels(x, mapping) if reorder else x

    sess = kx.training_session(fold, "train", 0, c)
    for orig, tr in zip(fold.train_trials, sess.trials):
        np.testing.assert_array_equal(tr.neural, expected(orig.neural))
        np.testing.assert_array_equal(tr.target, orig.target)

    # Decoding feeds the same prepared input and never reorders a second time, whatever
    # inference.reorder_channels says.
    t = fold.sets[1].trials[0]
    for flag in (True, False):
        ci = {**c, "inference": {**c["inference"], "reorder_channels": flag}}
        out = kx.decode(LastBinModel(), kx.prepare_neural(t.neural, ci), ci)
        np.testing.assert_array_equal(out, expected(t.neural)[W - 1:, :NT])

    if reorder:  # the mapping really moves columns (otherwise this test would prove nothing)
        assert not np.array_equal(expected(t.neural), t.neural)


# -------------------------------------------------------------------------------------- nulls


def test_shuffled_trial_partner_is_a_seeded_derangement():
    texts = [f"cue {j}." for j in range(20)]
    p = kx.shuffled_trial_partner(texts, 42, "imagined:B04")
    assert sorted(p) == list(range(20)) and all(p[i] != i for i in range(20))
    np.testing.assert_array_equal(p, kx.shuffled_trial_partner(texts, 42, "imagined:B04"))
    assert not np.array_equal(p, kx.shuffled_trial_partner(texts, 43, "imagined:B04"))
    assert not np.array_equal(p, kx.shuffled_trial_partner(texts, 42, "imagined:B07"))

    dup = ["Go home.", "go home", "Stay.", "Wait?"]                 # equal after normalization
    for seed in range(20):
        q = kx.shuffled_trial_partner(dup, seed, "x", normalize=normalize_text)
        assert all(normalize_text(dup[q[i]]) != normalize_text(dup[i]) for i in range(4))
    with pytest.raises(ValueError):
        kx.shuffled_trial_partner(["a", "A."], 42, "x", normalize=normalize_text)


def test_shuffled_time_is_seeded_per_trial_and_keeps_bins():
    rng = np.random.default_rng(1)
    x1, x2 = rng.standard_normal((80, F)).astype(np.float32), rng.standard_normal((90, F)).astype(np.float32)
    later_first = kx.shuffle_time(x2, 42, 4, 2)
    a = kx.shuffle_time(x1, 42, 4, 1)
    np.testing.assert_array_equal(a, kx.shuffle_time(x1, 42, 4, 1))            # deterministic
    np.testing.assert_array_equal(later_first, kx.shuffle_time(x2, 42, 4, 2))  # independent of call order
    assert not np.array_equal(a, x1)
    perm = kx.time_permutation(80, 42, 4, 1)
    np.testing.assert_array_equal(a, x1[perm])                                 # whole bins move together
    assert sorted(perm) == list(range(80))
    assert not np.array_equal(perm, kx.time_permutation(80, 42, 4, 3))         # other trial, other order
    assert not np.array_equal(perm, kx.time_permutation(80, 43, 4, 1))         # other seed, other order


# --------------------------------------------------------------------------------- statistics


def test_describe_paired_and_feature_scores():
    d = kx.describe([1.0, 2.0, 3.0, float("nan")])
    assert (d["n"], d["n_nan"], d["mean"], d["sd"], d["median"]) == (3, 1, 2.0, 1.0, 2.0)
    p = kx.paired([2.0, 3.0, 4.0, 5.0, float("nan")], [1.0, 1.5, 2.0, 2.5, 0.0])
    assert p["n"] == 4 and p["frac_a_gt_b"] == 1.0 and p["mean"] == pytest.approx(1.75)
    assert 0 < p["wilcoxon_p"] <= 1
    y = np.random.default_rng(2).standard_normal((50, NT))
    s = kx.feature_scores(y, y)
    assert s["feat_corr_mean"] == pytest.approx(1.0) and s["feat_mse"] == 0.0
