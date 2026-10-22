#!/usr/bin/env python
"""Own-cue identification on Wairagkar data, scored exactly like the Kunz runs: a reference point for the Kunz
identification numbers (D-046), from the same decoder family on its own patient's recent data.

Host, project venv, CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python -I scripts/cue_identification.py \\
        --cells day195_m193 day195_m172 day195_m165 [--category "Training data (50-word vocab)"]

For each cell (a session decoded by one released model; ``syllable_benchmark.cells``, decoder outputs saved by
scripts/syllable_decode.py): the trials of ``--category`` form one set; every decoded output is synthesized with
LPCNet and scored (DTW mel_corr, ``experiment.evaluation.tts_metrics`` of kunz_full.yaml) against the LPCNet
resynthesis of the TTS of EVERY distinct cue of the set (voice ``tts.voice``, corpus = the session); then
``wvx.kunz_experiment.cue_null_stats`` (length-matched k from kunz_full.yaml) and ``identification``, the same
functions as scripts/kunz_experiment.py. No targets, no ASR. These models were never trained on the session.

Writes results/cue_identification/<cell>.json (per-trial rows and the summary) and summary.md. ``--summary-only``
rebuilds summary.md from those files.

Every 50-word-vocabulary trial of t15_day195 has its own sentence (146 trials, 146 cues), while a Kunz block has
50. summary.md therefore also gives identification **among 50 candidates**: per trial, the expected top-1 and rank
if the other 49 candidates were a random draw from the other cues (exact, hypergeometric, from the own cue's rank
among all of them: P(top-1) = C(N - K, 49) / C(N, 49), with N other cues and K of them scoring at least as high).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402

OUT = "results/cue_identification"


def among(rows, m: int = 50) -> dict:
    """Expected identification among ``m`` candidates (the own cue + m - 1 drawn at random from the others)."""
    import numpy as np
    from scipy.special import comb

    top1, rank = [], []
    for r in rows:
        if not np.isfinite(r["own_rank"]):
            continue
        n_other, k = int(r["n_cues"]) - 1, int(r["own_rank"]) - 1
        if n_other < m - 1:
            raise ValueError(f"only {n_other + 1} cues, fewer than {m}")
        top1.append(comb(n_other - k, m - 1, exact=True) / comb(n_other, m - 1, exact=True))
        rank.append(1 + k * (m - 1) / n_other)
    return {"m": m, "n": len(top1), "top1_rate": float(np.mean(top1)), "top1_chance": 1.0 / m,
            "median_expected_rank": float(np.median(rank)), "mean_expected_rank": float(np.mean(rank))}


def write_summary(out_dir: Path, summaries: dict) -> None:
    """summary.md from every <cell>.json in ``out_dir`` (cells may run as separate processes)."""
    rows_by = {}
    for f in sorted(out_dir.glob("*.json")):
        d = json.loads(f.read_text())
        summaries.setdefault(f.stem, d["summary"])
        rows_by[f.stem] = d["trials"]
    s0 = next(iter(summaries.values()))
    lines = ["# Own-cue identification on Wairagkar data (Kunz scoring)", "",
             f"Trials: `{s0['category']}`. Each decoded output vs the TTS of every distinct cue of the set; DTW "
             "mel_corr; rank of the own cue (1 = best); excess over the 10 length-matched cues. Same functions as the "
             "Kunz runs (scripts/cue_identification.py). The models were never trained on these sessions. Every trial "
             "here has its own sentence, so the last two columns rescale to 50 candidates, as in a Kunz block "
             "(expected over random draws of the other 49).", "",
             "| cell | decoder age (days) | n | cues | top-1 (chance) | median rank (p) | excess length-matched (p) | "
             "top-1 among 50 (chance 0.02) | median rank / 50 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cell, s in sorted(summaries.items(), key=lambda kv: kv[1]["age_days"] or 0):
        i_, ex, a = s["identification"], s["excess_lenmatched"], among(rows_by[cell])
        lines.append(f"| {cell} | {s['age_days']} | {s['n_trials']} | {s['n_distinct_cues']} | "
                     f"{i_['top1_rate']:.3f} ({i_['top1_chance']:.3f}) | {i_['median_rank']:.1f} ({i_['uniform_rank_p']:.1e}) | "
                     f"{ex['mean']:.3f} ({ex['wilcoxon_p']:.1e}) | {a['top1_rate']:.3f} | {a['median_expected_rank']:.1f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--category", default="Training data (50-word vocab)")
    ap.add_argument("--cells-config", default="configs/experiments/syllable_benchmark.yaml")
    ap.add_argument("--scoring-config", default="configs/experiments/kunz_full.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true", help="rebuild summary.md from the existing <cell>.json")
    args = ap.parse_args(argv)
    if args.summary_only:
        write_summary(resolve_path(OUT), {})
        return 0
    cfg_c = load_experiment_config(args.cells_config)
    cfg = load_experiment_config(args.scoring_config)
    apply_thread_env(cfg)

    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import scipy.io

    from wvx import kunz_experiment as kx, metrics, synthesis, syllables as sy, targets, wairagkar_data
    from wvx.metrics import normalize_text
    from wvx.provenance import code_version, config_sha256

    ev = cfg["experiment"]["evaluation"]
    tts_cfg = ev["tts_metrics"]
    k = int(cfg["experiment"]["nulls"]["length_matched_k"])
    sr = int(cfg["synthesis"]["sample_rate"])
    o = dict(tts_cfg or {})  # as metric_hop_s in scripts/kunz_experiment.py
    hop_s = float(metrics.load_metrics_config(overrides=o.get("metrics", o) or None)["stft"]["hop_ms"]) / 1000.0
    voice = ev["tts_voice"] or cfg["tts"]["voice"]
    keys = cfg_c["wairagkar_data"]["keys"]
    data_dir = resolve_path(cfg_c["validation"]["data_dir"])
    out_dir = resolve_path(OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for cell in args.cells:
            spec = cfg_c["benchmark"]["cells"][cell]
            session = spec["session"]
            preds, header = sy.load_decoded(sy.decoded_dir(cfg_c, cell) / "predictions.npz")
            neural_file = wairagkar_data.session_files(data_dir, session, cfg_c)[0]
            mat = scipy.io.loadmat(str(neural_file), squeeze_me=True)
            cue_key = keys["cues"] if keys["cues"] in mat else keys["cues_fallback"]
            texts = [str(x).strip() for x in np.atleast_1d(mat[cue_key])]
            desc = [str(x).strip() for x in np.atleast_1d(mat["trial_description"])]
            if not (len(texts) == len(desc) == len(preds)):
                raise SystemExit(f"{cell}: {len(texts)} cues, {len(desc)} descriptions, {len(preds)} predictions")
            sel = [i for i, d in enumerate(desc) if d == args.category]
            corpus = targets.TTSCorpus.load(targets.corpus_path(cfg, voice, session))
            _, cue_texts, own = kx.set_cues([texts[i] for i in sel], normalize_text)
            t0 = time.time()
            tts = dict(pool.map(lambda x: (x, synthesis.features_to_waveform(corpus.lookup(x), cfg)), cue_texts))
            cue_s = np.array([float(np.sum(metrics.speech_frame_mask(tts[x], sr, tts_cfg))) * hop_s for x in cue_texts])

            def mel(ref, wav):
                try:
                    return float(metrics.mel_spectrogram_correlation(ref, wav, sr, tts_cfg))
                except Exception:  # undefined (e.g. silent output), as in scripts/kunz_experiment.py
                    return float("nan")

            def one(j):
                i = sel[j]
                wav = synthesis.decoder_output_to_waveform(preds[i], cfg)
                return [mel(tts[x], wav) for x in cue_texts]

            rows = []
            for j, scores in enumerate(pool.map(one, range(len(sel)))):
                r = {"trial": sel[j], "text": texts[sel[j]], "own_cue": own[j], "own_score": scores[own[j]]}
                r.update(kx.cue_null_stats(scores, own[j], cue_s, k))
                rows.append(r)
            ident = kx.identification([r["own_rank"] for r in rows], [r["n_cues"] for r in rows])
            ex = kx.paired([r["own_score"] for r in rows], [r["null_lenmatched_mean"] for r in rows])
            summaries[cell] = {"session": session, "model": spec["model"], "age_days": spec.get("age_days"),
                               "category": args.category, "n_trials": len(sel), "n_distinct_cues": len(cue_texts),
                               "identification": ident, "excess_lenmatched": ex,
                               "own_mel_corr": kx.describe([r["own_score"] for r in rows])}
            prov = {"code_version": code_version(), "scoring_config": args.scoring_config,
                    "scoring_config_sha256": config_sha256(kx.portable(cfg)), "voice": voice, "length_matched_k": k,
                    "predictions_header": {kk: header.get(kk) for kk in ("model", "session") if kk in header}}
            (out_dir / f"{cell}.json").write_text(json.dumps({**prov, "summary": summaries[cell], "trials": rows},
                                                             indent=1, default=float) + "\n")
            i_ = ident
            print(f"[cue_identification] {cell}: {len(sel)} trials, {len(cue_texts)} cues, top-1 {i_['top1_rate']:.3f} "
                  f"(chance {i_['top1_chance']:.3f}), median rank {i_['median_rank']:.1f}, excess {ex['mean']:.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    write_summary(out_dir, summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
