"""Training on tiny synthetic sessions (CPU). Regression tests for their session-index bug.

Their batching tags each batch with the session's list position instead of its configured layer
(notes/WAIRAGKAR_CODE.md §1b). Here sessions are built so that list position != layer.
"""

import importlib.util
import json
from collections import Counter

import numpy as np
import pytest

from wvx import b2v

pytestmark = [
    pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py"),
    # find_spec, not importorskip: importing TensorFlow before wvx.b2v would load Keras 3.
    pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="needs tensorflow"),
]

from wvx.config import load_config  # noqa: E402
from wvx.session import SessionData, Trial  # noqa: E402

EXPERIMENT = "configs/experiments/reproduce_training.yaml"
N_FEATURES, N_TARGETS, WINDOW = 512, 20, 60
CODE_SCALE = 100.0  # session code is written into the neural data at this scale (>> augmentation noise)


def tiny_cfg(tmp_path, model=None, **training):
    over = {
        "run_name": "tiny",
        "paths": {"checkpoints": str(tmp_path / "ckpt"), "results": str(tmp_path / "results")},
        "model": {"n_heads": 1, "n_transformer_blocks": 1, "mlp_units": [8], **(model or {})},
        "training": {
            "batch_size": 32,
            "epochs": 2,
            "n_augmentations": 1,
            "verbose": 0,
            "validation": {"n_trials": 2, "fraction": None, "pool": None},
            **training,
        },
    }
    return load_config(EXPERIMENT, overrides=over)


def make_session(name, layer, n_trials, seed, code=0.0, lengths=(70, 100), short=()):
    """Neural = code * CODE_SCALE + noise; target columns 0/1/2 = code, trial index, bin index."""
    rng = np.random.default_rng(seed)
    trials = []
    for j in range(n_trials):
        t = WINDOW - 5 if j in short else int(rng.integers(*lengths))
        neural = (code * CODE_SCALE + 0.5 * rng.standard_normal((t, N_FEATURES))).astype(np.float32)
        neural[:, 0] = np.arange(t)
        target = rng.uniform(-1, 1, (t, N_TARGETS)).astype(np.float32)
        target[:, 0], target[:, 1], target[:, 2] = code, j, np.arange(t)
        trials.append(Trial(neural=neural, target=target))
    return SessionData(name, layer, trials)


def batcher(sessions, trial_ids, cfg, **kw):
    from wvx.training import SessionBatcher

    t = cfg["training"]
    return SessionBatcher(
        sessions, trial_ids, batch_size=t["batch_size"], time_steps=cfg["model"]["window_len"],
        step_size=cfg["model"]["step_size"], seed=cfg["seed"], **kw,
    )


def test_every_batch_is_one_session_with_its_configured_layer(tmp_path):
    from wvx.training import n_windows, split_trials

    cfg = tiny_cfg(tmp_path)
    t = cfg["training"]
    # list positions 0, 1, 2 -> layers 2, 0, 2 (two sessions share a weekly layer, as in theirs)
    sessions = [
        make_session("a", 2, 6, seed=1, code=1),
        make_session("b", 0, 5, seed=2, code=2),
        make_session("c", 2, 4, seed=3, code=3),
    ]
    split = split_trials(sessions, cfg)
    b = batcher(sessions, split.train, cfg, shuffle=True, n_augmentations=t["n_augmentations"],
                augmentation_params=t["augmentation_params"])

    seen = Counter()
    sizes = Counter()
    for spec, x, s, y in b.epoch(0):
        sess = sessions[spec.session]
        code = sess.trials[0].target[0, 0]
        assert np.all(s == sess.layer)
        assert np.all(y[:, 0] == code)                                          # targets: one session
        assert np.all(np.round(x[:, :, 1:].mean(axis=(1, 2)) / CODE_SCALE) == code)  # neural: same session
        assert set(y[:, 1].astype(int)) <= set(split.train[spec.session])      # no validation trials
        assert len(x) <= t["batch_size"]
        sizes[spec.session] += len(x) < t["batch_size"]
        for p in spec.pieces:
            seen[spec.session, p.trial] += p.stop - p.start

    assert max(sizes.values()) <= 1  # at most one short batch per session
    for si, ids in enumerate(split.train):
        for j in ids:
            n = n_windows(len(sessions[si].trials[j].neural), cfg["model"]["window_len"], cfg["model"]["step_size"])
            assert seen[si, j] == (t["n_augmentations"] + 1) * n  # every window of every copy, once

    # The same through the tf.data pipeline that Keras consumes.
    layer_of_code = {s.trials[0].target[0, 0]: s.layer for s in sessions}
    n = 0
    for (x, s), y in b.dataset(N_FEATURES, N_TARGETS):
        code = float(y[0, 0])
        assert np.all(y[:, 0].numpy() == code)
        assert np.all(s.numpy() == layer_of_code[code])
        n += 1
    assert n == b.n_batches


def test_windows_pair_with_last_bin_target_like_theirs(tmp_path):
    from wvx.training import split_trials

    cfg = tiny_cfg(tmp_path)
    s = make_session("a", 0, 4, seed=5)
    split = split_trials([s], cfg)
    b = batcher([s], [split.train[0] + split.validation[0]], cfg)
    for _, x, _, y in b.epoch(0):
        assert np.array_equal(x[:, -1, 0], y[:, 2])  # window ending at bin k <-> target frame k

    data_utils = b2v.load("lib.data_utils")
    tr = s.trials[0]
    T = len(tr.neural)
    theirs = data_utils.format_single_trial(
        {"X": tr.neural, "Y": None, "session": s.name, "session_num": s.layer},
        WINDOW, 1, pre_formatted_Y=tr.target[-(T - WINDOW + 1):],  # their pairing, data_utils.py:200
    )
    ours = batcher([s], [[0]], cfg, drop_remainder=False)
    xs, ys = zip(*[(x, y) for _, x, _, y in ours.epoch(0)])
    assert np.array_equal(np.concatenate(xs), theirs["X"])
    assert np.array_equal(np.concatenate(ys), theirs["Y"])


def test_split_is_seeded_and_counts_actual_trials(tmp_path):
    from wvx.training import split_trials

    def chosen(split):
        return {(n, j) for n, ids in zip(split.sessions, split.validation) for j in ids}

    cfg = tiny_cfg(tmp_path, validation={"n_trials": 5, "pool": ["a", "c"]})
    sessions = [
        make_session("a", 1, 12, seed=1, short=(3,)),
        make_session("b", 0, 9, seed=2),
        make_session("c", 1, 7, seed=3),
    ]
    first = split_trials(sessions, cfg)
    assert chosen(first) == chosen(split_trials(sessions, cfg))
    assert chosen(first) == chosen(split_trials(sessions[::-1], cfg))  # independent of list order
    assert chosen(first) != chosen(split_trials(sessions, {**cfg, "seed": cfg["seed"] + 1}))

    assert len(chosen(first)) == 5
    assert {n for n, _ in chosen(first)} <= {"a", "c"}  # only from the pool
    assert first.excluded == [[3], [], []]               # shorter than one window
    for s, tr, va, ex in zip(sessions, first.train, first.validation, first.excluded):
        assert sorted(tr + va + ex) == list(range(len(s.trials)))
        assert not set(tr) & set(va)

    frac = tiny_cfg(tmp_path, validation={"n_trials": None, "fraction": 0.25, "pool": None})
    assert len(chosen(split_trials(sessions, frac))) == round(0.25 * (12 - 1 + 9 + 7))

    with pytest.raises(ValueError):
        split_trials(sessions, tiny_cfg(tmp_path, validation={"fraction": 0.25}))  # both set
    with pytest.raises(ValueError):
        split_trials(sessions, tiny_cfg(tmp_path, validation={"pool": ["nope"]}))


def test_augmentation_uses_their_function_seeded_and_fixed_per_copy(tmp_path):
    training_utils = b2v.load("lib.training_utils")
    cfg = tiny_cfg(tmp_path, n_augmentations=2)
    t = cfg["training"]
    s = make_session("a", 0, 2, seed=4)
    kw = dict(n_augmentations=2, augmentation_params=t["augmentation_params"])
    their_rng = training_utils._RNG

    fixed = batcher([s], [[0, 1]], cfg, **kw)
    a1 = fixed.neural(0, 0, 1, epoch=0)
    assert training_utils._RNG is their_rng  # their generator restored after each call
    assert not np.array_equal(a1, s.trials[0].neural)
    assert not np.array_equal(a1, fixed.neural(0, 0, 2, epoch=0))  # copies differ
    assert np.array_equal(a1, fixed.neural(0, 0, 1, epoch=3))      # same copy every epoch
    assert np.array_equal(a1, batcher([s], [[0, 1]], cfg, **kw).neural(0, 0, 1))  # seeded

    other_seed = batcher([s], [[0, 1]], {**cfg, "seed": cfg["seed"] + 1}, **kw)
    assert not np.array_equal(a1, other_seed.neural(0, 0, 1))
    fresh = batcher([s], [[0, 1]], cfg, resample_augmentations_each_epoch=True, **kw)
    assert not np.array_equal(fresh.neural(0, 0, 1, epoch=0), fresh.neural(0, 0, 1, epoch=1))

    # Augmentation noise is their recipe: per-sample noise of SD sd_gaussian_noise dominates.
    diff = a1 - s.trials[0].neural
    assert abs(np.std(np.diff(diff, axis=0)) / np.sqrt(2) - t["augmentation_params"]["sd_gaussian_noise"]) < 0.1


def _day_weights(model, i):
    return model.day_embedding[i].get_weights()


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_train_end_to_end_updates_only_configured_layers(tmp_path):
    from wvx import session_layers
    from wvx.training import train

    # Pretrained single-layer model -> every new input layer starts identical, so untouched layers
    # stay equal to it. Sessions at list positions 0, 1 use layers 3, 0; their bug would train 0, 1.
    cfg0 = tiny_cfg(tmp_path)
    src = session_layers.build_voiceformer(cfg0, 1)
    token_x, _, _ = session_layers.token_batch(cfg0)
    b2v.load("lib.model_utils").save_inference_model(src, token_x, str(tmp_path / "pre"), day_embed_layer=0)
    start = _day_weights(src, 0)

    cfg = tiny_cfg(
        tmp_path, model={"n_session_layers": 5},
        init={"source": "inference_h5", "path": str(tmp_path / "pre.h5"), "freeze_base": False},
    )
    sessions = [make_session("x", 3, 6, seed=11), make_session("y", 0, 5, seed=12)]
    result = train(sessions, cfg)
    model = result.model

    assert not _same(_day_weights(model, 3), start)
    assert not _same(_day_weights(model, 0), start)
    for i in (1, 2, 4):
        assert _same(_day_weights(model, i), start), f"layer {i} must not be trained"

    # Outputs
    assert result.final_checkpoint.is_dir() and result.best_checkpoint.is_dir()
    assert result.checkpoint_dir == tmp_path / "ckpt" / "tiny"
    assert sorted(result.inference_models) == [0, 3]
    lines = result.history_csv.read_text().strip().splitlines()
    assert len(lines) == 1 + len(result.history["loss"])
    assert "val_loss" in lines[0]
    split = json.loads(result.split_json.read_text())
    assert split["seed"] == cfg["seed"] and split["sessions"]["x"]["layer"] == 3
    assert (result.results_dir / "config.yaml").is_file()

    # Exported single-layer .h5 == the trained model with that layer selected
    import tensorflow as tf

    x = np.random.default_rng(0).standard_normal((4, WINDOW, N_FEATURES)).astype(np.float32)
    h5 = tf.keras.models.load_model(str(result.inference_models[3]), compile=False)
    np.testing.assert_allclose(
        h5.predict(x, verbose=0), model.predict([x, np.full((4, 1), 3, np.int32)], verbose=0), atol=1e-5
    )

    with pytest.raises(FileExistsError):  # never overwrite a run silently
        train(sessions, cfg)


def test_train_is_reproducible_with_seed(tmp_path):
    from wvx.training import train

    sessions = [make_session("x", 1, 5, seed=21), make_session("y", 0, 4, seed=22)]
    runs = []
    for name in ("r1", "r2"):
        cfg = tiny_cfg(tmp_path, epochs=1, save_best_checkpoint=False, save_final_checkpoint=False)
        cfg["run_name"] = name
        runs.append(train(sessions, cfg))
    assert runs[0].history == runs[1].history
    assert all(np.array_equal(a, b) for a, b in zip(runs[0].model.get_weights(), runs[1].model.get_weights()))
