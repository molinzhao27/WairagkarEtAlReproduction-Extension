#!/usr/bin/env python
"""D-091: pooled multi-day inner-speech adaptation of the b2v2 decoder (the user's "option 3").

    vast-3 / PC, a PyTorch env, GPU (a gq job):
    /workspace/envs/wvx-torch/bin/python -I scripts/train_b2v2_pool.py --config configs/experiments/b2v2_kunz_d091_pool_k03.yaml
    smoke test (CPU, nothing written): ... --smoke 3

``b2v2.pool``: start from ``init`` (a trained model; everything in it stays frozen) and give each of ``days`` its own
new input projection (initialized as ``copy_from``), trained on that day's whole trials with whole-trial CTC (as
D-076). ``inner`` trials (inner speech, any strategy) also pass the optional shared ``adapter`` (a residual low-rank
map on the input embedding, ``wvx.b2v2_model.SharedAdapter``, identity at start), which every day trains together:
the learned "inner -> attempted" correction. ``attempted`` trials (if any) train their day's input only, without the
adapter. Each step draws ``batch_size`` trials, each from a day chosen uniformly (``sampling: day_balanced``) and a
trial chosen uniformly within it; one random 50 ms phase per trial; the paper's augmentations from ``augment_from``.
Validation: ``n_validation`` inner trials of the ``decode_day`` drawn from its ``validation_blocks`` exactly as
``train_b2v2.split_trials`` draws them (so the same held-out trials as the single-day adaptation with this seed),
CTC at phase 0 with the adapter; best.pt = the lowest. best.pt decodes the decode day's inner speech: ``input_id`` and
``decode_adapter`` are stored in it for ``decode_b2v2.py``.
Writes data/checkpoints/<run_name>/{best,final}.pt and results/b2v2/<run_name>/{split.json, log.jsonl, summary.json}.
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
    print(f"[train_b2v2_pool {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def select(data, sels) -> list:
    """gids of the trials matching any of ``sels`` ({session, blocks?}), with a CTC label, in data order."""
    out = []
    for s in sels or []:
        blocks = set(int(b) for b in s["blocks"]) if s.get("blocks") is not None else None
        out += [t["gid"] for t in data.trials if t["session"] == s["session"] and t["label"] is not None
                and (blocks is None or t.get("block") in blocks)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="N steps on CPU, nothing written (a code check)")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    import torch
    from torch.nn import functional as F

    from wvx import b2v2_data as bd, b2v2_model as bm
    from wvx.provenance import code_version

    b = cfg["b2v2"]
    P = b["pool"]
    t = dict(b["training"], **(P.get("training") or {}))
    run = cfg["run_name"]
    steps = int(args.smoke or t["steps"])
    B, pool, ctx, hop = int(t["batch_size"]), int(b["data"]["pool"]), int(t["ctx"]), int(t["hop"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    torch.manual_seed(int(t["seed"]))
    rng = np.random.default_rng(int(t["seed"]) + 2)
    bf16 = bool(t["bf16"]) and device.type == "cuda"

    init = torch.load(resolve_path(P["init"]), map_location=device)
    model = bm.B2V2(init["model_cfg"]).to(device)
    model.load_state_dict(init["state"])
    for p in model.parameters():
        p.requires_grad_(False)
    cache, days = {}, []
    for d in P["days"]:
        dd = resolve_path(d["data_dir"])
        if dd not in cache:
            cache[dd] = bd.B2V2Data(dd)
        data = cache[dd]
        iid = model.add_input(int(d.get("copy_from", 0)))
        days.append({"name": d["name"], "data": data, "input_id": iid,
                     "inner": select(data, d.get("inner")), "attempted": select(data, d.get("attempted"))})
    ad = P.get("adapter")
    if ad:
        model.add_adapter(ad["name"], int(ad["rank"]))
    for d in days:
        for p in model.inputs[d["input_id"]].parameters():
            p.requires_grad_(True)
    if ad:
        for p in model.adapters[ad["name"]].parameters():
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))

    # validation: the decode day's held-out inner trials, drawn as train_b2v2.split_trials draws them
    dec = next(d for d in days if d["name"] == P["decode_day"])
    vpool = select(dec["data"], P["validation_from"])
    vrng = np.random.default_rng(int(t["seed"]))
    val = sorted(int(g) for g in vrng.choice(vpool, size=int(P["n_validation"]), replace=False))
    dec["inner"] = [g for g in dec["inner"] if g not in set(val)]
    for d in days:
        d["items"] = [(g, True) for g in d["inner"]] + [(g, False) for g in d["attempted"]]
        log(f"{run}: day {d['name']}: input {d['input_id']}, {len(d['inner'])} inner + {len(d['attempted'])} "
            f"attempted training trials")
    log(f"{run}: {len(val)} validation trials ({P['decode_day']}); adapter {ad}; {len(params)} trainable tensors, "
        f"{sum(p.numel() for p in params):,} values; device {device}")
    aug_from = int(float(t["augment_from"]) * steps)
    aname = ad["name"] if ad else None

    def trial_loss(day, gid, inner, phase, augment=False):
        tr = day["data"].by_gid[gid]
        x, _ = day["data"].trial_arrays(tr)
        n_tok = (x.shape[0] - phase) // pool
        tk = bd.pool_tokens(x[phase: phase + pool * n_tok], pool)
        if augment:
            tk = bd.augment_batch(tk[None], np.ones((1, n_tok), bool), rng, b["augmentation"])[0]
        tok = torch.as_tensor(tk, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            _, lg = bm.chunked_forward(model, tok, ctx, hop, day["input_id"], adapter=aname if inner else None)
        lp = F.log_softmax(lg.float(), -1)[:, None]
        lab = torch.as_tensor([tr["label"]], dtype=torch.long, device=device).reshape(1, -1)
        loss = F.ctc_loss(lp, lab, torch.tensor([n_tok]), torch.tensor([len(tr["label"])]), blank=0,
                          reduction="sum", zero_infinity=True)
        return loss / max(1, len(tr["label"]))

    def validate() -> float:
        model.eval()
        with torch.no_grad():
            v = float(np.mean([float(trial_loss(dec, g, True, 0)) for g in val]))
        model.train()
        return v

    ckpt_dir, res_dir = resolve_path(cfg["paths"]["checkpoints"]) / run, resolve_path("results/b2v2") / run

    def save(name, step, best):
        torch.save({"state": model.state_dict(), "step": step, "best": best, "model_cfg": model.cfg,
                    "input_id": dec["input_id"], "decode_adapter": aname,
                    "day_inputs": {d["name"]: d["input_id"] for d in days}, "config": args.config, "run_name": run,
                    "code_version": code_version()}, ckpt_dir / name)

    logf = None
    if not args.smoke:
        if (ckpt_dir / "final.pt").exists():
            raise SystemExit(f"{ckpt_dir}/final.pt exists; choose another run_name")
        res_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "split.json").write_text(json.dumps(
            {"val": val, "days": {d["name"]: {"inner": d["inner"], "attempted": d["attempted"]} for d in days}}) + "\n")
        logf = open(res_dir / "log.jsonl", "a")
    t0, hist = time.time(), []
    best = validate()
    log(f"{run}: validation CTC before adaptation {best:.4f}")
    hist.append({"step": 0, "val_ctc": best})
    if not args.smoke:
        save("best.pt", 0, best)
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = bm.lr_at(step, dict(t, steps=steps))
        loss = 0.0
        for _ in range(B):
            if P.get("sampling", "day_balanced") == "day_balanced":
                d = days[int(rng.integers(0, len(days)))]
            else:
                raise ValueError(f"unknown sampling {P.get('sampling')}")
            gid, inner = d["items"][int(rng.integers(0, len(d["items"])))]
            loss = loss + trial_loss(d, gid, inner, int(rng.integers(0, pool)), augment=step >= aug_from)
        loss = loss / B
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(t["grad_clip"]))
        opt.step()
        last = step + 1 == steps
        if (step + 1) % int(t["val_every"]) == 0 or last:
            v = validate()
            rec = {"step": step + 1, "train_ctc": float(loss), "val_ctc": v, "lr": bm.lr_at(step, dict(t, steps=steps)),
                   "s": time.time() - t0}
            if ad:
                rec["adapter_norm"] = float(model.adapters[aname].U.norm() * model.adapters[aname].V.norm())
            hist.append(rec)
            log(json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in rec.items()}))
            if logf:
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            if v < best:
                best = v
                if not args.smoke:
                    save("best.pt", step + 1, best)
    if args.smoke:
        log(f"smoke test: {steps} steps in {time.time() - t0:.0f} s; nothing written")
        return 0
    save("final.pt", steps, best)
    summary = {"run_name": run, "config": args.config, "decision": P.get("decision", "D-091"),
               "code_version": code_version(), "created": datetime.datetime.now().isoformat(timespec="seconds"),
               "steps": steps, "days": {d["name"]: {"input_id": d["input_id"], "n_inner": len(d["inner"]),
                                                    "n_attempted": len(d["attempted"])} for d in days},
               "adapter": ad, "n_val": len(val), "best_val_ctc": best, "history": hist,
               "n_trainable": sum(p.numel() for p in params), "wall_time_s": time.time() - t0, "device": str(device)}
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"done: {steps} steps in {(time.time() - t0) / 60:.1f} min, best validation CTC {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
