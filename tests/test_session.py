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
