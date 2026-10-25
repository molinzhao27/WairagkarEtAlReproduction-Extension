"""D-086 amendment: the DTW mel-cepstral distortion and spectral correlation of scripts/d086_acoustic.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")
_spec = importlib.util.spec_from_file_location(
    "d086_acoustic", Path(__file__).resolve().parents[1] / "scripts" / "d086_acoustic.py")
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)


def test_mcd_zero_for_a_time_stretched_copy_and_positive_otherwise():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 24))
    b = np.repeat(a, 2, axis=0)                                  # the same sounds, twice as slow
    m, wp = ac.mcd_path(a, b)
    assert m == pytest.approx(0.0, abs=1e-9)
    assert ac.mcd_path(a, a + 1.0)[0] == pytest.approx(10 / np.log(10) * np.sqrt(2 * 24), rel=1e-6)
    la = rng.normal(size=(40, 8))
    lb = np.repeat(la, 2, axis=0)
    assert ac.spec_corr(la, lb, wp) == pytest.approx(1.0, abs=1e-9)
    assert ac.spec_corr(la, -lb, wp) == pytest.approx(-1.0, abs=1e-9)
