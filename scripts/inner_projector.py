#!/usr/bin/env python
"""Same-day conversion of inner-speech activity toward attempted-speech activity, then decoding (D-067).

Host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/inner_projector.py --config configs/experiments/inner_projector.yaml

Kunz t15.2024.03.03: attempted block a and imagined block i of a pair hold the same 50 cues. Leave one pair out: the
transforms (``wvx.neural_align``: gain, projector, paired, paired_shuffled) are fitted on the go windows of the other
three pairs (50 ms frames) and applied to the held-out pair's imagined trials; every trial is decoded with the
zero-shot day-193 model. Score: each imagined decode's DTW similarity (cosine, decoded features 0-17, 50 ms) to the
50 attempted decodes of its pair; own-sentence mid-rank among 50; top-1; speech fraction of the decode (c0 envelope
> the F5 clone silence frame's + 15 dB). Paired statistics over the 200 held-out imagined trials.
Writes results/inner_projector/<name>.json and .md.
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
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[inner_projector {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help="override runtime.device")
    ap.add_argument("--config", default="configs/experiments/inner_projector.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    if args.device:
        cfg["runtime"]["device"] = args.device
    apply_thread_env(cfg)
    from wvx import b2v

    b2v.use_keras2()
    from wvx.training import configure_device

    configure_device(cfg["runtime"].get("device", "auto"))
    import numpy as np

    from wvx import alignment as al, inference, kunz_experiment as kx, metrics, neural_align as na
    from wvx import neural_match as nm, targets as tg
    from wvx.provenance import code_version
    from wvx.session import SessionData

    c = cfg["inner_projector"]
    t0 = time.time()
    k, pool, seed = int(c["n_components"]), int(c["pool_bins"]), int(c["seed"])
    dtw_cfg = dict(c["dtw"])
    proc = resolve_path(cfg["paths"]["kunz_processed"])
    sess_name = cfg["experiment"]["session"]
    att = SessionData.load(proc / f"{sess_name}_attempted.npz")
    im = SessionData.load(proc / f"{sess_name}_imagined.npz")
    pairs = {int(a): int(i) for a, i in c["pairs"].items()}
    model = inference.load_model(resolve_path(cfg["kunz"]["decoder"]), cfg)
    cfg_t = load_experiment_config(c["silence_config"])
    sil20 = tg.silence_frame(tg.TTSCorpus.load(tg.corpus_path(cfg_t, cfg_t["tts"]["voice"], c["silence_corpus"])), cfg_t)
    sil_db = float(np.mean(al.envelope_db(np.repeat(sil20[None], 50, 0).astype(np.float64), cfg)))

    def blocks(sess):
        out = {}
        for t in sess.trials:
            out.setdefault(int(t.block), []).append(t)
        return out

    A, I = blocks(att), blocks(im)
    for a, i in pairs.items():
        if sorted(tg.text_key(t.text) for t in A[a]) != sorted(tg.text_key(t.text) for t in I[i]):
            raise SystemExit(f"A{a} and I{i} do not hold the same cues")
    X = {id(t): kx.prepare_neural(t.neural, cfg) for t in att.trials + im.trials}

    def go(t):
        return X[id(t)][int(t.meta["go_offset"]):]

    def frames(trials):
        return np.concatenate([nm.pool_frames(go(t), pool) for t in trials])

    def decode(x):
        return kx.decode(model, np.ascontiguousarray(x, np.float32), cfg)

    def speech_fraction(pred, n_bins):
        raw = tg.prediction_timeline(pred, n_bins, sil20, cfg)
        return float(np.mean(al.envelope_db(np.asarray(raw, np.float64), cfg) > sil_db + float(c["speech_db"])))

    def sim_seq(pred):
        return nm.pool_frames(pred[:, :18], pool)

    # attempted references: zero-shot decodes of every attempted trial (untransformed)
    ref = {}
    for t in att.trials:
        p = decode(X[id(t)])
        ref[id(t)] = (sim_seq(p), speech_fraction(p, X[id(t)].shape[0]))
    log(f"decoded {len(ref)} attempted references; attempted speech fraction "
        f"{np.mean([v[1] for v in ref.values()]):.3f}")

    names = ["none", "gain", "projector", "paired", "paired_shuffled"]
    rows, fold_info = [], {}
    for a_out, i_out in pairs.items():
        tr = [(a, i) for a, i in pairs.items() if a != a_out]
        im_fr = frames([t for _, i in tr for t in I[i]])
        att_fr = frames([t for a, _ in tr for t in A[a]])
        V = na.pca_basis(np.concatenate([im_fr, att_fr]), k)
        maps, info = {"none": None}, {}
        maps["gain"] = na.fit_standardize(im_fr, att_fr)
        maps["projector"], info["projector"] = na.fit_projector(im_fr, att_fr, V)
        mu_i, mu_a = im_fr.mean(0), att_fr.mean(0)
        rng = np.random.default_rng([seed, a_out])
        for name in ("paired", "paired_shuffled"):
            pi, pa = [], []
            for a, i in tr:
                by_text = {tg.text_key(t.text): t for t in A[a]}
                ims = I[i]
                partners = [by_text[tg.text_key(t.text)] for t in ims]
                if name == "paired_shuffled":                      # seeded derangement within the pair
                    while True:
                        perm = rng.permutation(len(partners))
                        if not np.any(perm == np.arange(len(partners))):
                            break
                    partners = [partners[j] for j in perm]
                for ti, ta in zip(ims, partners):
                    fi, fa = nm.pool_frames(go(ti), pool), nm.pool_frames(go(ta), pool)
                    ia, ib = metrics._align(((fi - mu_i) @ V.T).T, ((fa - mu_a) @ V.T).T, True,
                                            {"alignment": {"dtw": dtw_cfg}}, "none")
                    pi.append(fi[ia]), pa.append(fa[ib])
            maps[name], info[name] = na.fit_paired_rotation(np.concatenate(pi), np.concatenate(pa), V,
                                                            float(c["shrink"]), mu_i, mu_a)
        fold_info[f"A{a_out}/I{i_out}"] = {"projector": info["projector"],
                                           "paired_canonical_r": info["paired"]["canonical_correlations"][:5],
                                           "shuffled_canonical_r": info["paired_shuffled"]["canonical_correlations"][:5],
                                           "gain": maps["gain"].info()}
        refs = A[a_out]
        cand = list(range(len(refs)))
        for ti in I[i_out]:
            own = next(j for j, ta in enumerate(refs) if tg.text_key(ta.text) == tg.text_key(ti.text))
            row = {"pair": f"A{a_out}/I{i_out}", "trial_index": int(ti.meta["trial_index"]), "text": ti.text,
                   "own": own, "len_imagined": int(X[id(ti)].shape[0]),
                   "len_attempted": [int(X[id(ta)].shape[0]) for ta in refs]}
            for name in names:
                m = maps[name]
                x = X[id(ti)] if m is None else m(X[id(ti)])
                p = decode(x)
                q = sim_seq(p)
                scores = [nm.dtw_similarity(q, ref[id(ta)][0], dtw_cfg) for ta in refs]
                r, _ = nm.own_rank(scores, cand, own)
                row[name] = {"rank": r, "top1": bool(r == 1.0), "speech_fraction": speech_fraction(p, x.shape[0]),
                             "own_similarity": float(scores[own]), "scores": [float(v) for v in scores]}
            rows.append(row)
        log(f"A{a_out}/I{i_out}: " + ", ".join(
            f"{n} rank {np.mean([r[n]['rank'] for r in rows if r['pair'] == f'A{a_out}/I{i_out}']):.1f}" for n in names)
            + f"; projector scale {info['projector']['scale']:.2f}")

    summary = {}
    for n in names:
        rk = [r[n]["rank"] for r in rows]
        t1 = [r[n]["top1"] for r in rows]
        summary[n] = {"mean_rank": float(np.mean(rk)), "top1": float(np.mean(t1)),
                      "top5": float(np.mean([x <= 5 for x in rk])),
                      "speech_fraction": float(np.mean([r[n]["speech_fraction"] for r in rows]))}
        if n != "none":
            summary[n]["vs_none_rank"] = kx.paired(rk, [r["none"]["rank"] for r in rows])
            b = sum(1 for r in rows if r[n]["top1"] and not r["none"]["top1"])
            cc = sum(1 for r in rows if r["none"]["top1"] and not r[n]["top1"])
            summary[n]["top1_gained_lost"] = [b, cc]
    for r in rows:
        cand_lm = nm.length_matched_candidates(r["len_attempted"], r["own"], int(c["length_matched_k"])).tolist()
        r["length_matched_candidates"] = cand_lm
        r["length_only_rank"] = nm.own_rank([-abs(r["len_imagined"] - L) for L in r["len_attempted"]],
                                            list(range(len(r["len_attempted"]))), r["own"])[0]
        for n in names:
            r[n]["rank_length_matched"] = nm.own_rank(r[n]["scores"], cand_lm, r["own"])[0]
    for n in names:
        lm = [r[n]["rank_length_matched"] for r in rows]
        summary[n]["length_matched_mean_rank"] = float(np.mean(lm))
        summary[n]["length_matched_top1"] = float(np.mean([x == 1.0 for x in lm]))
        if n != "none":
            summary[n]["length_matched_vs_none"] = kx.paired(lm, [r["none"]["rank_length_matched"] for r in rows])
    summary["length_only"] = {"mean_rank": float(np.mean([r["length_only_rank"] for r in rows])),
                              "top1": float(np.mean([r["length_only_rank"] == 1.0 for r in rows]))}
    summary["paired"]["length_matched_vs_shuffled"] = kx.paired([r["paired"]["rank_length_matched"] for r in rows],
                                                                [r["paired_shuffled"]["rank_length_matched"] for r in rows])
    summary["paired"]["vs_shuffled_rank"] = kx.paired([r["paired"]["rank"] for r in rows],
                                                      [r["paired_shuffled"]["rank"] for r in rows])
    out = {"name": c["name"], "config": args.config, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "code_version": code_version(), "settings": c, "n_trials": len(rows),
           "attempted_speech_fraction": float(np.mean([v[1] for v in ref.values()])),
           "summary": summary, "folds": fold_info, "rows": rows, "wall_time_s": time.time() - t0}
    res = resolve_path("results/inner_projector")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{c['name']}.json").write_text(json.dumps(kx.portable(out), indent=1, default=float) + "\n")
    lines = [f"# Inner -> attempted conversion `{c['name']}` (D-067): {len(rows)} held-out imagined trials", "",
             f"Attempted decodes' speech fraction: {out['attempted_speech_fraction']:.3f}", "",
             "| transform | mean own rank / 50 | Δ vs none (p) | top-1 | top-5 | speech fraction |",
             "|---|---|---|---|---|---|"]
    for n in names:
        s = summary[n]
        d = s.get("vs_none_rank")
        lines.append(f"| {n} | {s['mean_rank']:.2f} | " + (f"{d['mean']:+.2f} ({d['wilcoxon_p']:.2g})" if d else "–")
                     + f" | {100 * s['top1']:.1f}% | {100 * s['top5']:.1f}% | {s['speech_fraction']:.3f} |")
    k_lm = int(c["length_matched_k"])
    lines += ["", f"Post hoc (not pre-registered) length control: rank among the own + {k_lm} attempted trials closest in "
              f"length to the own attempted trial (chance {(k_lm + 2) / 2:.1f}); length-only baseline over all 50: mean rank "
              f"{summary['length_only']['mean_rank']:.2f}, top-1 {100 * summary['length_only']['top1']:.1f}%", "",
              "| transform | length-matched mean rank | Δ vs none (p) | length-matched top-1 |", "|---|---|---|---|"]
    for n in names:
        s_ = summary[n]
        d = s_.get("length_matched_vs_none")
        lines.append(f"| {n} | {s_['length_matched_mean_rank']:.2f} | "
                     + (f"{d['mean']:+.2f} ({d['wilcoxon_p']:.2g})" if d else "–")
                     + f" | {100 * s_['length_matched_top1']:.1f}% |")
    lm_vs = summary["paired"]["length_matched_vs_shuffled"]
    lines += ["", f"paired vs paired_shuffled, length-matched: {lm_vs['mean']:+.2f} (p {lm_vs['wilcoxon_p']:.2g})"]
    vs = summary["paired"]["vs_shuffled_rank"]
    lines += ["", f"paired vs paired_shuffled: rank {vs['mean']:+.2f} (p {vs['wilcoxon_p']:.2g})", "",
              "Per fold (projector scale; share of the attempted − imagined mean difference inside the subspace):", ""]
    lines += [f"- {f}: scale {v['projector']['scale']:.2f}, mean-difference share "
              f"{v['projector']['mean_diff_share_in_subspace']:.2f}, paired canonical r {[round(x, 2) for x in v['paired_canonical_r']]}, "
              f"shuffled {[round(x, 2) for x in v['shuffled_canonical_r']]}" for f, v in fold_info.items()]
    (res / f"{c['name']}.md").write_text("\n".join(lines) + "\n")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> results/inner_projector/{c['name']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
