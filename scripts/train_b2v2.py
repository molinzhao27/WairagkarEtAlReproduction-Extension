#!/usr/bin/env python
"""Train (or adapt) the brain2voice-2.0-style decoder (D-075).

    PC, the PyTorch env, GPU (under the wvx lock):
    ~/wvx/envs/wvx-f5/bin/python -I scripts/train_b2v2.py --config configs/experiments/b2v2_all.yaml [--resume]
    smoke test (any machine with the data, CPU): ... --smoke 20

Split (``b2v2.split``): training trials = every trial of the sessions with day <= ``max_day`` (default all), minus
the benchmark trials when ``exclude_benchmark``, minus ``exclude_sessions``; ``n_validation`` of them (seeded) are
held out for the validation loss. ``b2v2.adapt`` (optional) instead fine-tunes a trained model on one session: a new
input projection initialized from the trained one, everything else frozen, trained on that session's training trials
(``train_trials_from``: their indices in the session) with the ``loss`` named there (``ctc`` = text only,
``cont`` = their real targets, ``both``). Kunz (D-076): ``train_blocks`` selects whole blocks of ``session`` instead
of a trial file, and ``full_trial: true`` trains on whole trials through the long-context chunks
(``adapt_full_trials``).

Writes data/checkpoints/<run_name>/{final,best,last}.pt and results/b2v2/<run_name>/{split.json, log.jsonl,
summary.json}.
"""

from __future__ import annotations

import argparse
import datetime
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[train_b2v2 {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def split_trials(data, cfg) -> dict:
    """{"train", "val", "test"} lists of gids (see the module docstring)."""
    import numpy as np

    b = cfg["b2v2"]
    s, t = b.get("split") or {}, b["training"]
    rng = np.random.default_rng(int(t["seed"]))
    a = b.get("adapt")
    if a and a.get("train_blocks") is not None:            # Kunz: whole blocks of one condition (D-076)
        blocks = set(int(k) for k in a["train_blocks"])
        pool = [x["gid"] for x in data.trials if x["session"] == a["session"] and x.get("block") in blocks]
        test = []
        n_val = int(a["n_validation"])
    elif a:
        tr = json.loads(resolve_path(a["train_trials_from"]).read_text())
        idx = set(int(i) for i in tr[a.get("train_key", "train_trials")])
        pool = [x["gid"] for x in data.trials if x["session"] == a["session"] and x["index"] in idx]
        test = [x["gid"] for x in data.trials if x["session"] == a["session"]
                and x["index"] in set(int(i) for i in tr[a.get("test_key", "test_trials")])]
        n_val = int(a["n_validation"])
    if a and a.get("full_trial"):
        pool = [g for g in pool if data.by_gid[g]["label"] is not None]
    else:
        max_day = int(s.get("max_day") or 10 ** 6)
        drop = set(s.get("exclude_sessions") or [])
        test = [x["gid"] for x in data.trials if x["benchmark"]] if s.get("exclude_benchmark") else []
        pool = [x["gid"] for x in data.trials if x["day"] <= max_day and x["session"] not in drop
                and x["gid"] not in set(test)]
        n_val = int(t["n_validation"])
    val = sorted(int(g) for g in rng.choice(pool, size=n_val, replace=False))
    train = [g for g in pool if g not in set(val)]
    return {"train": train, "val": val, "test": test}


def adapt_full_trials(model, input_id, params, opt, data, split, b, t, run, ckpt_dir, res_dir, args, device) -> int:
    """Text-only (CTC) adaptation on WHOLE trials (``adapt.full_trial``, D-076): each step ``batch_size`` trials, one
    random 50 ms phase each, through ``wvx.b2v2_model.chunked_forward`` (the long-context chunks the decoder runs),
    CTC against the full sentence divided by its length. Validation: the same loss at phase 0 on the held-out trials;
    best.pt = the lowest validation loss."""
    import numpy as np
    import torch
    from torch.nn import functional as F

    from wvx import b2v2_data as bd, b2v2_model as bm
    from wvx.provenance import code_version

    pool, ctx, hop = int(b["data"]["pool"]), int(t["ctx"]), int(t["hop"])
    steps, B = int(t["steps"]), int(t["batch_size"])
    rng = np.random.default_rng(int(t["seed"]) + 2)
    trials = [data.by_gid[g] for g in split["train"]]
    bf16 = bool(t["bf16"]) and device.type == "cuda"

    def trial_loss(tr, phase):
        x, _ = data.trial_arrays(tr)
        n_tok = (x.shape[0] - phase) // pool
        tok = torch.as_tensor(bd.pool_tokens(x[phase: phase + pool * n_tok], pool), device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            _, lg = bm.chunked_forward(model, tok, ctx, hop, input_id)
        lp = F.log_softmax(lg.float(), -1)[:, None]
        lab = torch.as_tensor([tr["label"]], dtype=torch.long, device=device)
        loss = F.ctc_loss(lp, lab, torch.tensor([n_tok]), torch.tensor([len(tr["label"])]), blank=0,
                          reduction="sum", zero_infinity=True)
        return loss / max(1, len(tr["label"]))

    def validate() -> float:
        model.eval()
        with torch.no_grad():
            v = float(np.mean([float(trial_loss(data.by_gid[g], 0)) for g in split["val"]]))
        model.train()
        return v

    def save(name, step, best):
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "model_cfg": model.cfg, "input_id": input_id, "config": args.config, "run_name": run,
                    "code_version": code_version()}, ckpt_dir / name)

    logf = None
    if not args.smoke:
        res_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "split.json").write_text(json.dumps(split) + "\n")
        logf = open(res_dir / "log.jsonl", "a")
    t0, hist = time.time(), []
    best = validate()
    log(f"{run}: whole-trial CTC adaptation, {len(trials)} trials ({len(split['val'])} validation); "
        f"validation CTC before adaptation {best:.4f}")
    hist.append({"step": 0, "val_ctc": best})
    if not args.smoke:
        save("best.pt", 0, best)
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = bm.lr_at(step, t)
        pick = rng.integers(0, len(trials), B)
        loss = sum(trial_loss(trials[i], int(rng.integers(0, pool))) for i in pick) / B
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(t["grad_clip"]))
        opt.step()
        last = step + 1 == steps
        if (step + 1) % int(t["val_every"]) == 0 or last:
            v = validate()
            rec = {"step": step + 1, "train_ctc": float(loss), "val_ctc": v, "lr": bm.lr_at(step, t),
                   "s": time.time() - t0}
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
    summary = {"run_name": run, "config": args.config, "decision": "D-076", "code_version": code_version(),
               "created": datetime.datetime.now().isoformat(timespec="seconds"), "steps": steps, "mode": "full_trial",
               "n_train": len(trials), "n_val": len(split["val"]), "best_val_ctc": best, "history": hist,
               "n_trainable": sum(p.numel() for p in params), "wall_time_s": time.time() - t0, "device": str(device)}
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"done: {steps} steps in {(time.time() - t0) / 60:.1f} min, best validation CTC {best:.4f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    ap.add_argument("--smoke", type=int, default=0, help="N steps on CPU, nothing written (a code check)")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import numpy as np
    import torch

    from wvx import b2v2_data as bd, b2v2_model as bm
    from wvx.provenance import code_version

    b = cfg["b2v2"]
    a = b.get("adapt")
    t = dict(b["training"], **(a.get("training") or {})) if a else b["training"]
    run = cfg["run_name"]
    steps = int(args.smoke or t["steps"])
    t["steps"] = steps
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    torch.manual_seed(int(t["seed"]))
    data = bd.B2V2Data(resolve_path(b["data_dir"]))
    split = split_trials(data, cfg)
    log(f"{run}: {len(split['train'])} training, {len(split['val'])} validation, {len(split['test'])} test trials; "
        f"device {device}")
    ckpt_dir, res_dir = resolve_path(cfg["paths"]["checkpoints"]) / run, resolve_path("results/b2v2") / run

    model = bm.B2V2(b["model"]).to(device)
    input_id = 0
    if a:
        init = torch.load(resolve_path(a["init"]), map_location=device)
        model = bm.B2V2(init["model_cfg"]).to(device)
        model.load_state_dict(init["state"])
        input_id = model.add_input(0)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.inputs[input_id].parameters():
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    start, best, hist = 0, float("inf"), []
    if args.resume and (ckpt_dir / "last.pt").is_file():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        start, best = int(ck["step"]), float(ck["best"])
        log(f"resumed at step {start}")
    elif not args.smoke and (ckpt_dir / "final.pt").exists():
        raise SystemExit(f"{ckpt_dir}/final.pt exists; choose another run_name")
    if a and a.get("full_trial"):
        return adapt_full_trials(model, input_id, params, opt, data, split, b, t, run, ckpt_dir, res_dir, args, device)

    trials = [data.by_gid[g] for g in split["train"]]
    loss_kind = (a or {}).get("loss", "both")
    use_cont, use_ctc = loss_kind in ("cont", "both"), loss_kind in ("ctc", "both")
    bf16 = bool(t["bf16"]) and device.type == "cuda"
    aug_from = int(float(t["augment_from"]) * steps)

    # validation: fixed seeded windows, no augmentation
    vrng = np.random.default_rng(int(t["seed"]) + 1)
    vtrials = [data.by_gid[g] for g in split["val"]] * int(t["val_windows_per_trial"])
    vbatches = [bd.make_batch(data, vtrials[i:i + int(t["batch_size"])], vrng, b)
                for i in range(0, len(vtrials), int(t["batch_size"]))]

    def validate() -> dict:
        model.eval()
        tot = {"cont": 0.0, "ctc": 0.0}
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            for vb in vbatches:
                bt = bm.to_torch(vb, device)
                c, lg = model(bt["x"], input_id)
                out = bm.losses(c, lg, bt, 1.0, 1.0)
                for k in tot:
                    tot[k] += float(out[k]) / len(vbatches)
        model.train()
        return tot

    # batches built in a background thread (NumPy releases the GIL for the copies)
    q: queue.Queue = queue.Queue(maxsize=8)
    rng = np.random.default_rng(int(t["seed"]) + 2 + start)

    def producer():
        for step in range(start, steps):
            pick = rng.integers(0, len(trials), int(t["batch_size"]))
            q.put(bd.make_batch(data, [trials[i] for i in pick], rng, b, augment=step >= aug_from))

    threading.Thread(target=producer, daemon=True).start()
    model.train()
    t0, logf = time.time(), None
    if not args.smoke:
        res_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "split.json").write_text(json.dumps(split) + "\n")
        logf = open(res_dir / "log.jsonl", "a")

    def save(name: str, step: int):
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "model_cfg": model.cfg, "input_id": input_id, "config": args.config, "run_name": run,
                    "code_version": code_version()}, ckpt_dir / name)

    run_sum = {"loss": 0.0, "cont": 0.0, "ctc": 0.0, "speech": 0.0, "ctc_on": 0.0, "n": 0}
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = bm.lr_at(step, t)
        bt = bm.to_torch(q.get(), device)
        ws, wsil = bm.speech_weights(step, steps, t)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            c, lg = model(bt["x"], input_id)
        out = bm.losses(c, lg, bt, ws, wsil)
        loss = (out["cont"] if use_cont else 0.0) + (float(t["ctc_weight"]) * out["ctc"] if use_ctc else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(t["grad_clip"]))
        opt.step()
        for k, v in (("loss", float(loss)), ("cont", float(out["cont"])), ("ctc", float(out["ctc"])),
                     ("speech", float(bt["speech"].sum() / bt["valid"].sum())), ("ctc_on", float(bt["ctc"].float().mean()))):
            run_sum[k] += v
        run_sum["n"] += 1
        last = step + 1 == steps
        if (step + 1) % 100 == 0 or last:
            rec = {k: run_sum[k] / run_sum["n"] for k in ("loss", "cont", "ctc", "speech", "ctc_on")}
            rec.update(step=step + 1, lr=bm.lr_at(step, t), s=time.time() - t0)
            run_sum = dict.fromkeys(run_sum, 0.0)
            run_sum["n"] = 0
            if (step + 1) % int(t["val_every"]) == 0 or last:
                rec["val"] = validate()
                score = rec["val"]["cont"] if use_cont else rec["val"]["ctc"]
                if score < best and not args.smoke:
                    best = score
                    save("best.pt", step + 1)
            hist.append(rec)
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
            if logf:
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
        if not args.smoke and ((step + 1) % int(t["checkpoint_every"]) == 0 or last):
            save("last.pt", step + 1)
    if args.smoke:
        log(f"smoke test: {steps} steps in {time.time() - t0:.0f} s; nothing written")
        return 0
    save("final.pt", steps)
    summary = {"run_name": run, "config": args.config, "decision": "D-075", "code_version": code_version(),
               "created": datetime.datetime.now().isoformat(timespec="seconds"), "steps": steps,
               "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"]),
               "adapt": a, "loss": loss_kind, "best_val": best, "final": hist[-1] if hist else None,
               "n_params": sum(p.numel() for p in model.parameters()),
               "n_trainable": sum(p.numel() for p in params), "wall_time_s": time.time() - t0,
               "device": str(device), "torch": torch.__version__}
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"done: {steps} steps in {(time.time() - t0) / 60:.1f} min -> {ckpt_dir}/final.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
