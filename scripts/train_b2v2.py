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
(``adapt_full_trials``; D-086 adds ``adapt.distill`` and ``adapt.adversarial`` there). ``training.adversarial`` (D-086)
adds the b2v2 paper's multiscale discriminator to training (adversarial and feature-matching losses after a warm-up).

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
    extra = []
    if a and a.get("train_blocks") is not None:            # Kunz: whole blocks of one condition (D-076)
        blocks = set(int(k) for k in a["train_blocks"])
        pool = [x["gid"] for x in data.trials if x["session"] == a["session"] and x.get("block") in blocks]
        for ex in a.get("extra") or []:                    # D-085: more training trials from other conditions,
            eb = set(int(k) for k in ex["blocks"])         # never in validation (that stays the main condition)
            extra += [x["gid"] for x in data.trials if x["session"] == ex["session"] and x.get("block") in eb]
        test = []
        n_val = int(a["n_validation"])
    elif a:
        tr = json.loads(resolve_path(a["train_trials_from"]).read_text())
        idx = set(int(i) for i in tr[a.get("train_key", "train_trials")])
        pool = [x["gid"] for x in data.trials if x["session"] == a["session"] and x["index"] in idx]
        test = [x["gid"] for x in data.trials if x["session"] == a["session"]
                and x["index"] in set(int(i) for i in tr[a.get("test_key", "test_trials")])]
        n_val = int(a["n_validation"])
    else:
        max_day = int(s.get("max_day") or 10 ** 6)
        drop = set(s.get("exclude_sessions") or [])
        test = [x["gid"] for x in data.trials if x["benchmark"]] if s.get("exclude_benchmark") else []
        pool = [x["gid"] for x in data.trials if x["day"] <= max_day and x["session"] not in drop
                and x["gid"] not in set(test)]
        n_val = int(t["n_validation"])
    if a and a.get("full_trial"):
        pool = [g for g in pool if data.by_gid[g]["label"] is not None]
        extra = [g for g in extra if data.by_gid[g]["label"] is not None]
    val = sorted(int(g) for g in rng.choice(pool, size=n_val, replace=False))
    train = [g for g in pool if g not in set(val)] + extra
    return {"train": train, "val": val, "test": test}


def adapt_full_trials(model, input_id, params, opt, data, split, b, t, run, ckpt_dir, res_dir, args, device) -> int:
    """Text-only (CTC) adaptation on WHOLE trials (``adapt.full_trial``, D-076): each step ``batch_size`` trials, one
    random 50 ms phase each, through ``wvx.b2v2_model.chunked_forward`` (the long-context chunks the decoder runs),
    CTC against the full sentence divided by its length. Validation: the same loss at phase 0 on the held-out trials;
    best.pt = the lowest validation loss.

    D-086 options. ``adapt.distill`` {targets, weight}: trials with a target (an npz of 10 ms (T, 20) arrays,
    ``g<gid>``; ``scripts/build_distill_targets.py``) add ``weight`` x the speech-weighted squared error of the
    continuous output against it (``speech_weight`` / ``silence_weight``; speech = target c0 above the threshold); it
    also enters the validation loss. With ``kind: hidden`` (D-089) the targets are (T, d) encoder outputs of a
    teacher and the term is ``weight`` x the mean cosine dissimilarity of this model's encoder output to them. ``adapt.adversarial`` {real, start, ramp, w_from, w_to, d_lr, d_weight_decay,
    d_every, scales, channels, kernel}: a multiscale discriminator (``wvx.b2v2_model.MultiScaleDiscriminator``)
    learns to tell the continuous output on these trials from ``real`` sequences (the decoder's attempted-speech
    voice: decode files ``RUN:NAME``, read at a random 50 ms phase), updated every ``d_every`` steps; from step
    ``start`` the generator loss (weight ramped ``w_from`` -> ``w_to`` over ``ramp`` steps) pushes the output toward
    them. Unpaired, so no feature matching; not in the validation loss."""
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
    aug_from = int(float(t["augment_from"]) * steps)            # D-085: the paper's augmentations on the 50 ms tokens
    a = b["adapt"]
    dcfg, acfg = a.get("distill"), a.get("adversarial")
    targets = {}
    if dcfg:
        with np.load(resolve_path(dcfg["targets"]), allow_pickle=False) as z:
            targets = {int(k[1:]): z[k] for k in z.files if k.startswith("g")}
        log(f"{run}: distillation targets for {sum(g in targets for g in split['train'])} of {len(trials)} training "
            f"and {sum(g in targets for g in split['val'])} of {len(split['val'])} validation trials")
    c0i, thr = int(b["data"]["speech_feature"]), float(b["data"]["speech_threshold"])
    w_sp, w_sil = float(t["speech_weight"]), float(t["silence_weight"])
    disc = dopt = None
    reals = []
    if acfg:
        for spec in acfg["real"]:
            r_run, r_name = spec.split(":")
            with np.load(resolve_path("data/processed/b2v2_preds") / r_run / f"{r_name}.npz", allow_pickle=False) as z:
                reals += [z[f"g{g}_cont"].astype(np.float32) for g in json.loads(str(z["header"]))["gids"]]
        disc = bm.MultiScaleDiscriminator(int(model.cfg["n_targets"]), acfg["scales"], acfg["channels"],
                                          int(acfg["kernel"])).to(device)
        dopt = torch.optim.AdamW(disc.parameters(), lr=float(acfg["d_lr"]), weight_decay=float(acfg["d_weight_decay"]))
        log(f"{run}: adversarial, {len(reals)} real sequences")

    def trial_forward(tr, phase, augment=False):
        """(supervised loss, continuous output (n_tok, 20))."""
        x, _ = data.trial_arrays(tr)
        n_tok = (x.shape[0] - phase) // pool
        tk = bd.pool_tokens(x[phase: phase + pool * n_tok], pool)
        if augment:
            tk = bd.augment_batch(tk[None], np.ones((1, n_tok), bool), rng, b["augmentation"])[0]
        tok = torch.as_tensor(tk, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            c, lg, h = bm.chunked_forward(model, tok, ctx, hop, input_id, return_hidden=True)
        lp = F.log_softmax(lg.float(), -1)[:, None]
        lab = torch.as_tensor([tr["label"]], dtype=torch.long, device=device)
        loss = F.ctc_loss(lp, lab, torch.tensor([n_tok]), torch.tensor([len(tr["label"])]), blank=0,
                          reduction="sum", zero_infinity=True) / max(1, len(tr["label"]))
        y = targets.get(int(tr["gid"]))
        if y is not None:
            frames = phase + pool * np.arange(n_tok) + pool - 1
            yt = torch.as_tensor(y[np.minimum(frames, len(y) - 1)].astype(np.float32), device=device)
            if dcfg.get("kind", "voice") == "hidden":            # D-089: the teacher's encoder output, cosine
                loss = loss + float(dcfg["weight"]) * (1 - F.cosine_similarity(h.float(), yt, dim=-1)).mean()
            else:
                w = torch.where(yt[:, c0i] > thr, w_sp, w_sil)
                loss = loss + float(dcfg["weight"]) * (w * ((c.float() - yt) ** 2).mean(-1)).mean()
        return loss, c

    def real_batch(n):
        out = []
        for i in rng.integers(0, len(reals), n):
            r, ph = reals[i], int(rng.integers(0, pool))
            out.append(torch.as_tensor(r[ph + pool - 1:: pool], device=device))
        return out

    def disc_on(seqs):
        y = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        m = torch.zeros(y.shape[:2], dtype=torch.bool, device=device)
        for i, q in enumerate(seqs):
            m[i, :len(q)] = True
        return disc(y, m)

    def validate() -> float:
        model.eval()
        with torch.no_grad():
            v = float(np.mean([float(trial_forward(data.by_gid[g], 0)[0]) for g in split["val"]]))
        model.train()
        return v

    def save(name, step, best):
        ck = {"state": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
              "model_cfg": model.cfg, "input_id": input_id, "config": args.config, "run_name": run,
              "code_version": code_version()}
        if disc is not None:
            ck["disc"] = disc.state_dict()
        torch.save(ck, ckpt_dir / name)

    logf = None
    if not args.smoke:
        res_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "split.json").write_text(json.dumps(split) + "\n")
        logf = open(res_dir / "log.jsonl", "w")            # no resume here: a restarted run starts its log over
    t0, hist = time.time(), []
    best = validate()
    log(f"{run}: whole-trial CTC adaptation, {len(trials)} trials ({len(split['val'])} validation); "
        f"validation loss before adaptation {best:.4f}")
    hist.append({"step": 0, "val_ctc": best})
    if not args.smoke:
        save("best.pt", 0, best)
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = bm.lr_at(step, t) * g.get("scale", 1.0)
        pick = rng.integers(0, len(trials), B)
        outs = [trial_forward(trials[i], int(rng.integers(0, pool)), augment=step >= aug_from) for i in pick]
        loss = sum(o[0] for o in outs) / B
        rec_extra = {}
        if disc is not None:
            fakes = [o[1].float() for o in outs]
            if step % int(acfg["d_every"]) == 0:
                dl = bm.disc_loss(disc_on(real_batch(B)), disc_on([f.detach() for f in fakes]))
                dopt.zero_grad(set_to_none=True)
                dl.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), float(t["grad_clip"]))
                dopt.step()
                rec_extra["d_loss"] = float(dl)
            wa = bm.adversarial_weight(step, acfg)
            if wa > 0:
                for p_ in disc.parameters():
                    p_.requires_grad_(False)
                gl = bm.gen_loss(disc_on(fakes))
                for p_ in disc.parameters():
                    p_.requires_grad_(True)
                loss = loss + wa * gl
                rec_extra.update(g_loss=float(gl), w_adv=wa)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(t["grad_clip"]))
        opt.step()
        last = step + 1 == steps
        if (step + 1) % int(t["val_every"]) == 0 or last:
            v = validate()
            rec = {"step": step + 1, "train_loss": float(loss), "val_ctc": v, "lr": bm.lr_at(step, t),
                   "s": time.time() - t0, **rec_extra}
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
    summary = {"run_name": run, "config": args.config, "decision": a.get("decision", "D-076"),
               "code_version": code_version(), "created": datetime.datetime.now().isoformat(timespec="seconds"),
               "steps": steps, "mode": "full_trial", "n_train": len(trials), "n_val": len(split["val"]),
               "best_val_ctc": best, "history": hist, "n_trainable": sum(p.numel() for p in params),
               "wall_time_s": time.time() - t0, "device": str(device)}
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"done: {steps} steps in {(time.time() - t0) / 60:.1f} min, best validation loss {best:.4f}")
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
        input_id = model.add_input(int(a.get("copy_from", 0)), a.get("rank"))
        if a.get("input_init") == "fresh":              # D-095: another participant's channels map to nothing of T15's
            if a.get("rank") is not None:
                raise SystemExit("input_init: fresh needs a full-rank input")
            model.inputs[input_id].reset_parameters()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.trainable_input_parameters(input_id):
            p.requires_grad_(True)
    eset = set()
    for i in (a or {}).get("train_encoder_layers") or []:   # D-098: encoder layers trained too, at encoder_lr_scale
        for p in model.encoder.layers[int(i)].parameters():
            p.requires_grad_(True)
            eset.add(id(p))
    params = [p for p in model.parameters() if p.requires_grad]
    if eset:
        groups = [{"params": [p for p in params if id(p) not in eset], "scale": 1.0},
                  {"params": [p for p in params if id(p) in eset], "scale": float(a.get("encoder_lr_scale", 1.0))}]
        opt = torch.optim.AdamW(groups, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    else:
        opt = torch.optim.AdamW(params, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    adv = None if (a and a.get("full_trial")) else t.get("adversarial")
    disc = dopt = None
    if adv:                                                     # D-086: the paper's multiscale discriminator
        disc = bm.MultiScaleDiscriminator(int(model.cfg["n_targets"]), adv["scales"], adv["channels"],
                                          int(adv["kernel"])).to(device)
        dopt = torch.optim.AdamW(disc.parameters(), lr=float(adv["d_lr"]), weight_decay=float(adv["d_weight_decay"]))
    start, best, hist = 0, float("inf"), []
    if args.resume and (ckpt_dir / "last.pt").is_file():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        if disc is not None and "disc" in ck:
            disc.load_state_dict(ck["disc"])
            dopt.load_state_dict(ck["dopt"])
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
        logf = open(res_dir / "log.jsonl", "a" if start else "w")   # append only when resuming

    def save(name: str, step: int):
        ck = {"state": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
              "model_cfg": model.cfg, "input_id": input_id, "config": args.config, "run_name": run,
              "code_version": code_version()}
        if disc is not None:
            ck.update(disc=disc.state_dict(), dopt=dopt.state_dict())
        torch.save(ck, ckpt_dir / name)

    run_sum = {"loss": 0.0, "cont": 0.0, "ctc": 0.0, "speech": 0.0, "ctc_on": 0.0, "n": 0}
    if disc is not None:
        run_sum.update(d_loss=0.0, g_loss=0.0, fm=0.0)
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = bm.lr_at(step, t) * g.get("scale", 1.0)
        bt = bm.to_torch(q.get(), device)
        ws, wsil = bm.speech_weights(step, steps, t)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            c, lg = model(bt["x"], input_id)
        out = bm.losses(c, lg, bt, ws, wsil)
        loss = (out["cont"] if use_cont else 0.0) + (float(t["ctc_weight"]) * out["ctc"] if use_ctc else 0.0)
        if disc is not None and step >= int(adv["start"]):
            # the paper: after a warm-up, the discriminator (every d_every batches) tells the targets from the
            # predictions; the model adds the generator loss (weight ramped) and feature matching (w_fm)
            yt, vm = bt["y"], bt["valid"]
            if (step - int(adv["start"])) % int(adv["d_every"]) == 0:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                    dl = bm.disc_loss(disc(yt, vm), disc(c.detach(), vm))
                dopt.zero_grad(set_to_none=True)
                dl.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), float(t["grad_clip"]))
                dopt.step()
                run_sum["d_loss"] += float(dl) * int(adv["d_every"])
            for p_ in disc.parameters():
                p_.requires_grad_(False)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                real, fake = disc(yt, vm), disc(c, vm)
                gl, fm = bm.gen_loss(fake), bm.feature_matching(real, fake)
            for p_ in disc.parameters():
                p_.requires_grad_(True)
            loss = loss + bm.adversarial_weight(step, adv) * gl + float(adv["w_fm"]) * fm
            run_sum["g_loss"] += float(gl)
            run_sum["fm"] += float(fm)
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
            rec = {k: run_sum[k] / run_sum["n"] for k in run_sum if k != "n"}
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
    summary = {"run_name": run, "config": args.config, "decision": b.get("decision", "D-075"),
               "code_version": code_version(),
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
