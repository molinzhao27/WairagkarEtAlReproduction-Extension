"""Brain2voice-2.0-style decoder (D-075; Wairagkar et al. 2026 bioRxiv 10.64898/2026.06.30.735633), PyTorch.

Built from the paper's appendix (its code is unreleased): a linear input projection (one per input "day"; the paper
has one), sinusoidal positions, an 8-layer causal Transformer encoder (384 wide, 6 heads, FFN 1536, dropout 0.1),
and two of its four heads: continuous LPCNet features (20) and phonemes (41 classes, CTC). Not built yet: the token
head, the self-supervised head and the adversarial discriminator. Our choices where the paper is silent: pre-norm
layers with GELU and a final LayerNorm; head hidden width 384.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def _head(d: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_out))


class B2V2(nn.Module):
    def __init__(self, m: Mapping[str, Any]):
        super().__init__()
        d = int(m["d_model"])
        self.cfg = dict(m)
        self.inputs = nn.ModuleList([nn.Linear(int(m["n_features"]), d) for _ in range(int(m.get("n_inputs", 1)))])
        self.register_buffer("pe", sinusoid(int(m["max_tokens"]), d), persistent=False)
        layer = nn.TransformerEncoderLayer(d, int(m["n_heads"]), int(m["d_ffn"]), float(m["dropout"]),
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, int(m["n_layers"]), norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.cont = _head(d, int(m["n_targets"]))
        self.phon = _head(d, int(m["n_phonemes"]))

    def add_input(self, copy_from: int = 0) -> int:
        """A new input projection (a new day), initialized as ``copy_from``; returns its index."""
        new = nn.Linear(self.inputs[copy_from].in_features, self.inputs[copy_from].out_features)
        new.load_state_dict(self.inputs[copy_from].state_dict())
        self.inputs.append(new.to(self.inputs[copy_from].weight.device))
        self.cfg["n_inputs"] = len(self.inputs)
        return len(self.inputs) - 1

    def forward(self, x: torch.Tensor, input_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, L, F) -> (continuous (B, L, 20), phoneme logits (B, L, 41)). Causal: position i sees 0..i only, so
        right padding never reaches a real token."""
        L = x.shape[1]
        h = self.inputs[input_id](x) + self.pe[:L]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        h = self.encoder(h, mask=causal, is_causal=True)
        return self.cont(h), self.phon(h)


# ------------------------------------------------------------------------------------------------ losses


def speech_weights(step: int, total: int, t: Mapping[str, Any]) -> tuple[float, float]:
    """(speech, silence) loss weights: from 1:1 linearly to ``speech_weight``:``silence_weight`` over the first
    ``weight_ramp`` of training (paper: 5:0.2 over 30%)."""
    f = min(1.0, step / max(1.0, float(t["weight_ramp"]) * total))
    return 1.0 + f * (float(t["speech_weight"]) - 1.0), 1.0 + f * (float(t["silence_weight"]) - 1.0)


def losses(cont: torch.Tensor, logits: torch.Tensor, batch: Mapping[str, torch.Tensor], w_speech: float,
           w_silence: float) -> dict:
    """``cont``: speech-weighted MSE (mean over features, sum of weights / number of valid tokens); ``ctc``: CTC per
    window divided by its label length (at least 1), mean over windows with a label (0 if none)."""
    valid, speech = batch["valid"], batch["speech"]
    w = torch.where(speech, w_speech, w_silence) * valid
    se = ((cont.float() - batch["y"]) ** 2).mean(-1)
    out = {"cont": (w * se).sum() / valid.sum().clamp_min(1)}
    on = batch["ctc"]
    if bool(on.any()):
        lp = F.log_softmax(logits.float(), -1).transpose(0, 1)                     # (L, B, C)
        n_in = valid.sum(1)
        per = F.ctc_loss(lp, batch["labels"], n_in, batch["label_len"], blank=0, reduction="none",
                         zero_infinity=True)
        per = per / batch["label_len"].clamp_min(1)
        out["ctc"] = per[on].mean()
    else:
        out["ctc"] = cont.new_zeros(())
    return out


def lr_at(step: int, t: Mapping[str, Any]) -> float:
    """Linear warm-up over ``warmup`` of the steps, then cosine from ``lr`` to ``min_lr``."""
    total, lr, lo = int(t["steps"]), float(t["lr"]), float(t["min_lr"])
    warm = max(1, int(float(t["warmup"]) * total))
    if step < warm:
        return lr * (step + 1) / warm
    f = (step - warm) / max(1, total - warm)
    return lo + 0.5 * (lr - lo) * (1 + math.cos(math.pi * min(1.0, f)))


def to_torch(batch: Mapping[str, np.ndarray], device) -> dict:
    return {k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in batch.items()}


# ---------------------------------------------------------------------------------------------- decoding


@torch.no_grad()
def decode_live(model: B2V2, x: np.ndarray, window_bins: int, pool: int, input_id: int = 0, chunk: int = 512,
                device: Optional[torch.device] = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulated live decoding of one trial (x: (T, F) 10 ms bins): at every bin t, the last ``window_bins`` bins as
    tokens ending at t, keep the last output. Returns continuous (T, 20) and phoneme logits (T, 41); the first
    ``pool - 1`` bins (no full token yet) repeat the first output."""
    from wvx.b2v2_data import live_windows

    device = device or next(model.parameters()).device
    model.eval()
    conts, logs = [], []
    T = x.shape[0]
    for a in range(0, T, chunk):
        tok, n_tok, steps = live_windows(x, window_bins, pool, start=a, stop=a + chunk)
        if not len(steps):
            continue
        c, lg = model(torch.as_tensor(tok, device=device), input_id)
        idx = torch.as_tensor(n_tok - 1, device=device)
        rows = torch.arange(len(steps), device=device)
        conts.append(c[rows, idx].float().cpu().numpy())
        logs.append(lg[rows, idx].float().cpu().numpy())
    c, lg = np.concatenate(conts), np.concatenate(logs)
    lead = T - len(c)
    return np.concatenate([np.repeat(c[:1], lead, 0), c]), np.concatenate([np.repeat(lg[:1], lead, 0), lg])
