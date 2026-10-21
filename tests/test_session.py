import numpy as np
import pytest

from wvx.session import SessionData, Trial


def _trial(t=7, f=6, k=3, target=True):
    return Trial(
        neural=np.zeros((t, f), np.float32),
        target=np.zeros((t, k), np.float32) if target else None,
    )


def test_valid_session_passes():
    SessionData("s", 0, [_trial(), _trial(target=False)]).validate(n_features=6, n_targets=3)


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    trials = [
        Trial(neural=rng.random((5, 6), np.float32), target=rng.random((5, 3), np.float32),
              text="a b.", block=3, meta={"go_offset": np.int64(2), "closed_loop": True}),
        Trial(neural=rng.random((4, 6), np.float32), text="c?", block=6),
    ]
    SessionData("s", 2, trials).save(tmp_path / "s.npz")
    back = SessionData.load(tmp_path / "s.npz")
    assert (back.name, back.layer, len(back.trials)) == ("s", 2, 2)
    for a, b in zip(trials, back.trials):
        np.testing.assert_array_equal(a.neural, b.neural)
        assert (a.target is None) == (b.target is None)
        if a.target is not None:
            np.testing.assert_array_equal(a.target, b.target)
        assert (a.text, a.block, a.meta) == (b.text, b.block, b.meta)


@pytest.mark.parametrize(
    "trial",
    [
        Trial(neural=np.zeros((7, 5), np.float32)),                       # wrong n_features
        Trial(neural=np.zeros((7, 6), np.float64)),                       # wrong dtype
        Trial(neural=np.zeros((7, 6), np.float32), target=np.zeros((8, 3), np.float32)),  # T mismatch
    ],
)
def test_bad_trials_raise(trial):
    with pytest.raises(ValueError):
        SessionData("s", 0, [trial]).validate(n_features=6, n_targets=3)
