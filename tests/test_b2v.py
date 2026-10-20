import importlib.util
import sys

import pytest

from wvx import b2v

pytestmark = pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py")


def test_loads_their_modules_from_the_fetched_copy():
    data_utils = b2v.load("lib.data_utils")
    assert data_utils.__file__ == str(b2v.B2V_ROOT / "lib" / "data_utils.py")
    assert callable(data_utils.reorder_channels)


def test_keras_resolves_to_keras2():
    # find_spec, not importorskip: importing TensorFlow before wvx.b2v would load Keras 3.
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("tensorflow not installed")
    b2v.load("models.brain2voice")
    import tensorflow as tf
    import tf_keras

    assert sys.modules["keras"] is tf_keras
    assert b2v.keras2_active(tf)


def test_rejects_non_b2v_modules():
    with pytest.raises(ValueError):
        b2v.load("os")
