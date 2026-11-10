"""wvx.postnet (D-200): identity at initialization, causality, the degradation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # not in the TF env ``wvx``; as tests/test_b2v2_model.py

from wvx import postnet as pn  # noqa: E402


def test_identity_at_init_and_causal():
    torch.manual_seed(0)
    m = pn.PostNet(d=32, dilations=(1, 2, 4))
    c, l = torch.randn(2, 50, 20), torch.log_softmax(torch.randn(2, 50, 41), -1)
    assert torch.equal(m(c, l), c)                          # zero-initialized output: the voice passes through
    torch.nn.init.normal_(m.out.weight)
    y = m(c, l)
    c2, l2 = c.clone(), l.clone()
    c2[:, 30:] += 5.0
    l2[:, 30:] = torch.log_softmax(torch.randn(2, 20, 41), -1)
    y2 = m(c2, l2)
    assert torch.allclose(y[:, :30], y2[:, :30], atol=1e-6) and not torch.allclose(y[:, 30:], y2[:, 30:])
    assert m.receptive_field == 1 + 2 * (1 + 2 + 4)


def test_run_keeps_nan_rows_and_degrade_flattens():
    m = pn.PostNet(d=16, dilations=(1,))
    c = np.random.default_rng(0).normal(size=(40, 20)).astype(np.float32)
    l = np.log(np.full((40, 41), 1 / 41, np.float32))
    c[:4] = np.nan
    y = pn.run(m, c, l)
    assert np.isnan(y[:4]).all() and np.allclose(y[4:], c[4:])
    rng = np.random.default_rng(1)
    lp = np.log(np.eye(41, dtype=np.float32)[np.arange(40) % 41] * 0.98 + 0.02 / 41)
    cd, ld = pn.degrade(c[4:], lp[4:], rng, {"p": 1.0, "max_smooth": 9, "shrink": [0.6, 0.6], "temperature": [3.0, 3.0]})
    assert cd.std(0).mean() < c[4:].std(0).mean()          # flatter voice
    assert np.allclose(np.exp(ld).sum(-1), 1, atol=1e-4) and np.exp(ld).max() < np.exp(lp).max()   # flatter read-out
