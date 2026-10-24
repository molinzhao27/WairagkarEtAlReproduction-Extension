"""The brain2voice-2.0-style decoder (D-075): causality, input projections, losses, schedule, live decoding.

Needs PyTorch (not in the TF env ``wvx``): run with ``~/wvx_f5/venv/bin/python -m pytest tests/test_b2v2_model.py``.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wvx import b2v2_data as bd, b2v2_model as bm  # noqa: E402

M = {"n_features": 12, "d_model": 32, "n_heads": 4, "d_ffn": 64, "n_layers": 2, "dropout": 0.0, "n_targets": 3,
     "n_phonemes": 41, "max_tokens": 64, "n_inputs": 1}


def _model(seed=0):
    torch.manual_seed(seed)
    return bm.B2V2(M).eval()


def test_causal():
    m = _model()
    x = torch.randn(2, 10, 12)
    c1, p1 = m(x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 12) * 5
    c2, p2 = m(x2)
    assert torch.allclose(c1[:, :6], c2[:, :6], atol=1e-5) and torch.allclose(p1[:, :6], p2[:, :6], atol=1e-5)
    assert not torch.allclose(c1[:, 6:], c2[:, 6:])


def test_add_input_copies_then_is_independent():
    m = _model()
    k = m.add_input(0)
    x = torch.randn(1, 5, 12)
    assert k == 1 and torch.allclose(m(x, 0)[0], m(x, 1)[0])
    with torch.no_grad():
        m.inputs[1].weight.add_(torch.randn_like(m.inputs[1].weight))  # (a constant shift is removed by LayerNorm)
    assert not torch.allclose(m(x, 0)[0], m(x, 1)[0])


def test_losses():
    y = torch.randn(2, 6, 3)
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[1, 4:] = False
    b = {"y": y, "valid": valid, "speech": valid & (y[..., 0] > 0), "labels": torch.tensor([[2, 1, 0], [0, 0, 0]]),
         "label_len": torch.tensor([2, 0]), "ctc": torch.tensor([True, False])}
    out = bm.losses(y.clone(), torch.randn(2, 6, 41), b, 5.0, 0.2)
    assert float(out["cont"]) == 0.0 and np.isfinite(float(out["ctc"])) and float(out["ctc"]) > 0
    b["ctc"] = torch.tensor([False, False])
    assert float(bm.losses(y + 1, torch.randn(2, 6, 41), b, 1.0, 1.0)["ctc"]) == 0.0
    assert abs(float(bm.losses(y + 1, torch.randn(2, 6, 41), b, 1.0, 1.0)["cont"]) - 1.0) < 1e-6
    # an empty label (window of silence) is a valid CTC target
    b["ctc"] = torch.tensor([False, True])
    assert np.isfinite(float(bm.losses(y, torch.randn(2, 6, 41), b, 1.0, 1.0)["ctc"]))


def test_schedule_and_weights():
    t = {"steps": 1000, "lr": 1e-3, "min_lr": 1e-5, "warmup": 0.1, "weight_ramp": 0.3, "speech_weight": 5.0,
         "silence_weight": 0.2}
    assert bm.lr_at(0, t) == pytest.approx(1e-5) and bm.lr_at(99, t) == pytest.approx(1e-3)
    assert bm.lr_at(999, t) == pytest.approx(1e-5, rel=1e-2)
    assert bm.speech_weights(0, 1000, t) == (1.0, 1.0)
    assert bm.speech_weights(300, 1000, t) == pytest.approx((5.0, 0.2))
    assert bm.speech_weights(150, 1000, t) == pytest.approx((3.0, 0.6))


def test_decode_live_matches_direct_windows():
    m = _model(1)
    x = np.random.default_rng(0).normal(size=(53, 12)).astype(np.float32)
    c, lg = bm.decode_live(m, x, window_bins=20, pool=5, chunk=7)
    assert c.shape == (53, 3) and lg.shape == (53, 41)
    for t in (4, 9, 23, 52):
        n = min(4, (t + 1) // 5)
        tok = torch.as_tensor(bd.pool_tokens(x[t + 1 - 5 * n: t + 1], 5))[None]
        assert np.allclose(m(tok)[0][0, -1].detach().numpy(), c[t], atol=1e-5)
    assert np.allclose(c[0], c[4])


def test_decode_chunked_each_output_sees_its_window():
    m = _model(2)
    pool, ctx, hop = 5, 8, 4
    x = np.random.default_rng(1).normal(size=(97, 12)).astype(np.float32)
    c, lg, ph0 = bm.decode_chunked(m, x, pool, ctx, hop)
    assert c.shape == (97, 3) and lg.shape == (97, 41) and not np.isnan(c).any()
    for p in range(pool):
        n_tok = (97 - p) // pool
        tok = bd.pool_tokens(x[p: p + pool * n_tok], pool)
        for k in range(n_tok):
            if k < ctx:
                s = 0
            else:                                  # the window whose kept part (last hop) holds k
                s = hop * ((k - ctx) // hop + 1)
            want = m(torch.as_tensor(tok[s: k + 1])[None])[0][0, -1].detach().numpy()
            assert np.allclose(c[p + pool * k + pool - 1], want, atol=1e-5), (p, k)
    assert ph0.shape == (97 // pool, 41)
