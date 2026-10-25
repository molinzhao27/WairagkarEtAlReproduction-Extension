#!/usr/bin/env python
"""D-042-style data-free parity smoke for the b2v2 decoder on a new machine (a gq job): the ``b2v2.model`` network
with every parameter and one input batch drawn from a fixed NumPy seed (the same numbers on every machine and torch
version), one forward and one backward pass in float32 (TF32 off) on the GPU and on the CPU. Writes the largest
|GPU - CPU| differences and a fingerprint (output and gradient sums) to results/parity/b2v2_smoke_<name>.json, so
machines can be compared with each other as well.

    gq submit --on vast3 --project wvx --name parity_b2v2 --gpu --est 5m -- \\
        /workspace/envs/wvx-torch/bin/python -I scripts/parity_smoke_b2v2.py --name vast3
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_base.yaml")
    ap.add_argument("--name", required=True, help="machine name for the output file")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-3, help="largest allowed |GPU - CPU| (outputs; gradients relative)")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    import torch

    from wvx import b2v2_model as bm

    m, L = cfg["b2v2"]["model"], int(cfg["b2v2"]["data"]["context_tokens"])
    rng = np.random.default_rng(int(cfg["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = bm.B2V2(m)
    with torch.no_grad():
        for name, p in sorted(ref.named_parameters()):
            v = rng.normal(0.0, 0.02, tuple(p.shape))
            if name.endswith(".weight") and "norm" in name:
                v = 1.0 + v
            p.copy_(torch.as_tensor(v, dtype=torch.float32))
    x = torch.as_tensor(rng.normal(0.0, 1.0, (args.batch, L, int(m["n_features"]))), dtype=torch.float32)

    def run(device):
        model = bm.B2V2(m).to(device)
        model.load_state_dict(ref.state_dict())
        model.eval()                                            # no dropout: deterministic
        c, lg = model(x.to(device))
        loss = (c ** 2).mean() + torch.logsumexp(lg, -1).mean()
        loss.backward()
        grads = {n: p.grad.detach().cpu().double() for n, p in model.named_parameters()}
        return c.detach().cpu().double(), lg.detach().cpu().double(), float(loss), grads

    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    c0, l0, s0, g0 = run(torch.device("cpu"))
    out = {"name": args.name, "decision": "D-042 (parity smoke)", "torch": torch.__version__,
           "cuda": torch.version.cuda, "python": platform.python_version(),
           "device": torch.cuda.get_device_name(0) if gpu else None,
           "capability": list(torch.cuda.get_device_capability(0)) if gpu else None,
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "fingerprint_cpu": {"loss": s0, "cont_sum": float(c0.sum()), "logit_sum": float(l0.sum()),
                               "grad_abs_sum": float(sum(g.abs().sum() for g in g0.values()))}}
    if gpu:
        c1, l1, s1, g1 = run(gpu)
        out["fingerprint_gpu"] = {"loss": s1, "cont_sum": float(c1.sum()), "logit_sum": float(l1.sum()),
                                  "grad_abs_sum": float(sum(g.abs().sum() for g in g1.values()))}
        out["max_abs_diff"] = {"cont": float((c1 - c0).abs().max()), "logits": float((l1 - l0).abs().max()),
                               "loss": abs(s1 - s0),
                               "grad_rel": float(max(((g1[n] - g0[n]).abs().max() / (g0[n].abs().max() + 1e-12))
                                                     for n in g0))}
    dst = resolve_path("results/parity")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"b2v2_smoke_{args.name}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    ok = (gpu is None) or all(out["max_abs_diff"][k] < args.tol for k in ("cont", "logits", "grad_rel"))
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
