"""D-200: a causal post-net that turns the b2v2 decoder's 10 ms output (its continuous voice features and its
phoneme log-probabilities) into voice features: "voice from the read-out".

Trained on T15's attempted speech, where real voice targets exist (``scripts/train_postnet.py``); applied unchanged to
any decode that stores ``g<gid>_lp10`` (``scripts/decode_b2v2.py --save-logits``; ``scripts/apply_postnet.py``).
Causal: every output frame sees only that frame and the past (a stack of dilated causal convolutions, receptive
field ``1 + 2 * sum(dilations)`` frames). The output is ``cont + delta`` and ``delta`` starts at exactly 0, so an
untrained post-net is the identity on the voice.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _Block(nn.Module):
    def __init__(self, d: int, k: int, dil: int):
        super().__init__()
        self.pad = (k - 1) * dil
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, 2 * d, k, dilation=dil)
        self.proj = nn.Conv1d(d, d, 1)

    def forward(self, x):                                   # x (B, d, T)
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.conv(F.pad(h, (self.pad, 0)))              # left padding only: causal
        a, g = h.chunk(2, dim=1)
        return x + self.proj(torch.tanh(a) * torch.sigmoid(g))


class PostNet(nn.Module):
    def __init__(self, n_cont: int = 20, n_phon: int = 41, d: int = 256, dilations=(1, 2, 4, 8, 16, 32), k: int = 3):
        super().__init__()
        self.cfg = {"n_cont": n_cont, "n_phon": n_phon, "d": d, "dilations": list(dilations), "k": k}
        self.inp = nn.Conv1d(n_cont + n_phon, d, 1)
        self.blocks = nn.ModuleList([_Block(d, k, int(x)) for x in dilations])
        self.out = nn.Conv1d(d, n_cont, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.cfg["k"] - 1) * sum(self.cfg["dilations"])

    def forward(self, cont: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
        """cont (B, T, 20), lp (B, T, 41) log-probabilities -> voice features (B, T, 20)."""
        x = self.inp(torch.cat([cont, lp], -1).transpose(1, 2))
        for b in self.blocks:
            x = b(x)
        return cont + self.out(x).transpose(1, 2)


def run(model: PostNet, cont: np.ndarray, lp: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
    """One trial: (T, 20), (T, 41) -> (T, 20). NaN rows (the decoder's first ``pool - 1`` bins) are filled from the
    first finite row before the post-net and kept as they were in the output."""
    device = device or next(model.parameters()).device
    c, l = np.array(cont, np.float32), np.array(lp, np.float32)
    bad = ~np.isfinite(c).all(1) | ~np.isfinite(l).all(1)
    if bad.all():
        return c
    first = int(np.argmax(~bad))
    c[bad], l[bad] = c[first], l[first]
    model.eval()
    with torch.no_grad():
        y = model(torch.as_tensor(c[None], device=device), torch.as_tensor(l[None], device=device))[0].cpu().numpy()
    y[bad] = np.asarray(cont, np.float32)[bad]
    return y


def degrade(cont: np.ndarray, lp: np.ndarray, rng: np.random.Generator, aug: dict) -> tuple[np.ndarray, np.ndarray]:
    """Training augmentation toward what inner speech does to the decoder's output (D-084: the voice holds a flat,
    averaged sound; the read-out is less sure): a causal moving average of the voice over ``w`` frames, its swing
    shrunk toward the sequence mean by ``s``, and the phoneme distribution flattened by a temperature ``tau``."""
    if rng.random() >= float(aug["p"]):
        return cont, lp
    w = int(rng.integers(1, int(aug["max_smooth"]) + 1))
    c = cont
    if w > 1:
        cs = np.cumsum(np.vstack([np.zeros((1, c.shape[1]), c.dtype), c]), 0)
        idx = np.arange(1, len(c) + 1)
        lo = np.maximum(idx - w, 0)
        c = (cs[idx] - cs[lo]) / (idx - lo)[:, None]
    s = rng.uniform(float(aug["shrink"][0]), float(aug["shrink"][1]))
    m = c.mean(0, keepdims=True)
    c = m + s * (c - m)
    tau = rng.uniform(float(aug["temperature"][0]), float(aug["temperature"][1]))
    z = lp / tau
    z = z - z.max(-1, keepdims=True)
    return c.astype(np.float32), (z - np.log(np.exp(z).sum(-1, keepdims=True))).astype(np.float32)
