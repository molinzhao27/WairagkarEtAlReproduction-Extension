#!/usr/bin/env python
"""D-200: train the causal post-net (``wvx.postnet``) on T15's attempted speech with real voice targets.

    PC (gq; GPU if present, else CPU): /home/molin/wvx/envs/wvx-f5/bin/python -I scripts/train_postnet.py \
        --config configs/experiments/b2v2_postnet.yaml

``b2v2.postnet``: the decoder (``base``: T15's adversarial base, input 0), its split (``split``: the base's
results/b2v2/<run>/split.json), ``n_train`` training trials (a seeded draw from the base's training trials), the
augmentation (``wvx.postnet.degrade``), the model and training settings. The decoder runs causally over every chosen
trial (``decode_chunked``, as decoding does); the post-net learns base output (continuous features + phoneme
log-probabilities) -> the trial's real target features, with the base's own speech-weighted squared error, on random
windows. Validation: the base's validation trials (unaugmented); also the benchmark test trials (held out from the
base too), raw vs post-net. Writes data/checkpoints/<run_name>/postnet.pt and results/b2v2/<run_name>/summary.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[train_postnet {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="N trials / steps on CPU, nothing written")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    import torch

    from wvx import b2v2_data as bd, b2v2_model as bm, postnet as pn
    from wvx.provenance import code_version

    b, P = cfg["b2v2"], cfg["b2v2"]["postnet"]
    run = cfg["run_name"]
    T = P["training"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    torch.set_num_threads(int(P.get("threads", 8)))
    torch.manual_seed(int(T["seed"]))
    rng = np.random.default_rng(int(T["seed"]))
    ck = torch.load(resolve_path(P["base"]), map_location=device, weights_only=False)
    base = bm.B2V2(ck["model_cfg"]).to(device)
    base.load_state_dict(ck["state"])
    iid = int(ck.get("input_id", 0))
    data = bd.B2V2Data(resolve_path(b["data_dir"]))
    split = json.loads(resolve_path(P["split"]).read_text())
    n_tr = int(args.smoke or P["n_train"])
    tr_g = sorted(rng.choice(split["train"], size=n_tr, replace=False).tolist())
    va_g = split["val"][: (2 if args.smoke else None)]
    te_g = split["test"][: (2 if args.smoke else None)]
    pool, ctx, hop = int(b["data"]["pool"]), int(P["ctx"]), int(P["hop"])
    c0i, thr = int(b["data"]["speech_feature"]), float(b["data"]["speech_threshold"])
    w_sp, w_sil = float(b["training"]["speech_weight"]), float(b["training"]["silence_weight"])

    def decode(gids):
        out = {}
        for g in gids:
            x, y = data.trial_arrays(data.by_gid[g])
            c, lg, _ = bm.decode_chunked(base, x, pool, ctx, hop, input_id=iid, device=device)
            l = lg - np.nanmax(lg, -1, keepdims=True)
            l = l - np.log(np.nansum(np.exp(l), -1, keepdims=True))
            ok = np.isfinite(c).all(1) & np.isfinite(l).all(1)
            out[g] = (c[ok].astype(np.float16), l[ok].astype(np.float16), y[ok].astype(np.float16))
        return out

    t0 = time.time()
    D_tr, D_va, D_te = decode(tr_g), decode(va_g), decode(te_g)
    log(f"{run}: decoded {len(D_tr)} training, {len(D_va)} validation, {len(D_te)} test trials in "
        f"{time.time() - t0:.0f} s ({device})")
    model = pn.PostNet(d=int(P["model"]["d"]), dilations=P["model"]["dilations"], k=int(P["model"]["k"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(T["lr"]), weight_decay=float(T["weight_decay"]))
    W, B, steps = int(T["window"]), int(T["batch_size"]), int(args.smoke or T["steps"])

    def wmse(pred, y, mask=None):
        w = torch.where(y[..., c0i] > thr, w_sp, w_sil)
        if mask is not None:
            w = w * mask
            return (w * ((pred - y) ** 2).mean(-1)).sum() / mask.sum().clamp(min=1)
        return (w * ((pred - y) ** 2).mean(-1)).mean()

    def evaluate(D):
        model.eval()
        raw = net = 0.0
        with torch.no_grad():
            for c, l, y in D.values():
                ct, lt, yt = (torch.as_tensor(a.astype(np.float32), device=device)[None] for a in (c, l, y))
                raw += float(wmse(ct, yt)) / len(D)
                net += float(wmse(model(ct, lt), yt)) / len(D)
        model.train()
        return raw, net

    keys = list(D_tr)
    hist, best, t1 = [], None, time.time()
    out_dir = resolve_path(cfg["paths"]["checkpoints"]) / run
    for step in range(steps):
        lr = float(T["lr"]) * min(1.0, (step + 1) / max(1, int(0.05 * steps)))
        lr *= 0.5 * (1 + np.cos(np.pi * min(1.0, step / steps)))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        cs, ls, ys, ms = [], [], [], []
        for _ in range(B):
            c, l, y = D_tr[keys[int(rng.integers(0, len(keys)))]]
            s0 = int(rng.integers(0, max(1, len(c) - W)))
            c, l, y = (a[s0: s0 + W].astype(np.float32) for a in (c, l, y))
            c, l = pn.degrade(c, l, rng, P["augmentation"])
            pad = W - len(c)
            m = np.ones(W, np.float32)
            if pad > 0:                                      # repeat the last frame; masked out of the loss
                c, l, y = (np.vstack([a, np.repeat(a[-1:], pad, 0)]) for a in (c, l, y))
                m[W - pad:] = 0
            cs.append(c), ls.append(l), ys.append(y), ms.append(m)
        ct, lt, yt, mt = (torch.as_tensor(np.stack(a), device=device) for a in (cs, ls, ys, ms))
        loss = wmse(model(ct, lt), yt, mt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % int(T["val_every"]) == 0 or step + 1 == steps:
            raw, net = evaluate(D_va)
            rec = {"step": step + 1, "train": float(loss), "val_raw": raw, "val_postnet": net, "s": time.time() - t1}
            hist.append(rec)
            log(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}))
            if not args.smoke and (best is None or net < best):
                best = net
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state": model.state_dict(), "cfg": model.cfg, "step": step + 1, "val": net,
                            "config": args.config, "run_name": run, "code_version": code_version()}, out_dir / "postnet.pt")
    if args.smoke:
        log(f"smoke test: {steps} steps; nothing written")
        return 0
    model.load_state_dict(torch.load(out_dir / "postnet.pt", map_location=device, weights_only=False)["state"])
    te_raw, te_net = evaluate(D_te)
    res = resolve_path("results/b2v2") / run
    res.mkdir(parents=True, exist_ok=True)
    summary = {"run_name": run, "config": args.config, "decision": "D-200", "code_version": code_version(),
               "created": datetime.datetime.now().isoformat(timespec="seconds"), "n_train": len(D_tr),
               "n_val": len(D_va), "n_test": len(D_te), "receptive_field_frames": model.receptive_field,
               "n_params": sum(p.numel() for p in model.parameters()), "best_val_postnet": best,
               "test_attempted_raw": te_raw, "test_attempted_postnet": te_net, "history": hist,
               "wall_time_s": time.time() - t0, "device": str(device)}
    (res / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"done: best val {best:.4f}; benchmark test raw {te_raw:.4f} -> post-net {te_net:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
