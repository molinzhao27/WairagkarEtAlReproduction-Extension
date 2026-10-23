#!/usr/bin/env python
"""kNN-VC voice conversion of the TTS toward their voice (D-054): preparation, re-encoding and the target check.

Host, project venv (CPU; ffmpeg from the tools env on PATH); the conversion itself is scripts/knnvc_convert.py under
an interpreter with PyTorch:

    .venv/bin/python -I scripts/knnvc.py prep   --config configs/experiments/knnvc.yaml
    /opt/worm-train/bin/python scripts/knnvc_convert.py --knnvc <models> --matching-dir <work>/matching \\
        --source-dir <work>/source/<corpus> --out-dir <work>/converted/<corpus>          # per corpus
    .venv/bin/python -I scripts/knnvc.py encode --config configs/experiments/knnvc.yaml
    .venv/bin/python -I scripts/knnvc.py check  --config configs/experiments/knnvc.yaml

prep    matching set: LPCNet audio of their raw word targets, each trial cut to its speech (word spans ± margin),
        50-word trials of ``matching_sessions`` (never the evaluated session); sources: the LPCNet audio of the
        stored TTS features of every cue of each corpus in ``corpora``.
encode  converted audio -> LPCNet features; the integer frame lag of the resynthesis/re-encoding chain is measured on
        the unconverted sources (c0 cross-correlation with the stored TTS features, median over cues) and removed,
        so the converted features share the TTS frame timeline; saved as a TTS corpus of voice ``voice``; the source
        voice's forced syllables are copied for it (kNN-VC keeps the frame timing).
check   per-trial r vs their targets of the TTS placed on their true syllables (oracle) and on the c0 dips of the
        dev decoder's output (auto), for the dev training trials of ``check_session``: source voice vs converted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[knnvc {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def _cfgs(cfg):
    k = cfg["knnvc"]
    return k, load_experiment_config(k["data_config"]), load_experiment_config(k["placement_config"])


def cmd_prep(cfg) -> None:
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    from word_splice import their_session
    from wvx import synthesis, targets as tg

    k, cfg_d, cfg_p = _cfgs(cfg)
    work = resolve_path(k["work_dir"])
    sr, spf = int(cfg_p["synthesis"]["sample_rate"]), int(cfg_p["synthesis"]["samples_per_frame"])
    margin = int(k["matching_margin_frames"])
    jobs = []
    for s in k["matching_sessions"]:
        raw, words, _, _, desc = their_session(s, cfg_d, cfg_p)
        for i, d in enumerate(desc):
            if d == k["matching_category"] and len(words[i]):
                a, b = max(int(words[i][0, 0]) - margin, 0), min(int(words[i][-1, 1]) + margin, raw[i].shape[0])
                jobs.append((work / "matching" / f"{s}_tr{i:03d}.wav", raw[i], a, b))

    def one(job):
        path, raw, a, b = job
        wav = synthesis.features_to_waveform(raw, cfg_p)
        synthesis.write_wav(path, wav[a * spf: b * spf], cfg_p)

    with ThreadPoolExecutor(int(k["workers"])) as ex:
        list(ex.map(one, jobs))
    log(f"matching set: {len(jobs)} wavs -> {work / 'matching'}")
    voice = k["source_voice"]
    for corpus_name in k["corpora"]:
        corpus = tg.TTSCorpus.load(tg.corpus_path(cfg_p, voice, corpus_name))
        out = work / "source" / corpus_name

        def src(j):
            synthesis.write_wav(out / f"s{j:04d}.wav", synthesis.features_to_waveform(corpus.frames[j], cfg_p), cfg_p)

        with ThreadPoolExecutor(int(k["workers"])) as ex:
            list(ex.map(src, range(len(corpus))))
        log(f"sources: {corpus_name}: {len(corpus)} wavs -> {out}")
    _ = np  # numpy kept for symmetry with the other steps


def best_lag(ref_c0, enc_c0, max_lag: int) -> int:
    """Integer frame shift L maximizing corr(ref_c0[t], enc_c0[t + L])."""
    import numpy as np

    best, best_r = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        a = ref_c0[max(0, -lag): len(ref_c0) - max(0, lag)]
        b = enc_c0[max(0, lag): max(0, lag) + len(a)]
        n = min(len(a), len(b))
        if n > 10 and np.std(a[:n]) > 0 and np.std(b[:n]) > 0:
            r = float(np.corrcoef(a[:n], b[:n])[0, 1])
            if r > best_r:
                best, best_r = lag, r
    return best


def cmd_encode(cfg) -> None:
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    from wvx import synthesis, syllables as sy, targets as tg

    k, cfg_d, cfg_p = _cfgs(cfg)
    work = resolve_path(k["work_dir"])
    src_voice, voice = k["source_voice"], k["voice"]
    meta_all = {}
    for corpus_name in k["corpora"]:
        corpus = tg.TTSCorpus.load(tg.corpus_path(cfg_p, src_voice, corpus_name))
        n = len(corpus)
        enc = lambda d, j: synthesis.encode_wav(work / d / corpus_name / f"s{j:04d}.wav", cfg_p)  # noqa: E731
        with ThreadPoolExecutor(int(k["workers"])) as ex:
            src_enc = list(ex.map(lambda j: enc("source", j), range(n)))
            conv_enc = list(ex.map(lambda j: enc("converted", j), range(n)))
        lags = [best_lag(corpus.frames[j][:, 0], src_enc[j][:, 0], int(k["max_lag_frames"])) for j in range(n)]
        lag = int(np.median(lags))
        frames = []
        for j in range(n):
            f = conv_enc[j][lag:] if lag >= 0 else np.concatenate([np.repeat(corpus.silence[None], -lag, 0), conv_enc[j]])
            L = corpus.frames[j].shape[0]
            f = f[:L] if f.shape[0] >= L else np.concatenate([f, np.repeat(corpus.silence[None], L - f.shape[0], 0)])
            frames.append(f.astype(np.float32))
        manifest = json.loads((work / "converted" / corpus_name / "manifest.json").read_text())
        meta = {"source_voice": src_voice, "method": "kNN-VC (D-054)", "frame_lag_removed": lag,
                "lag_distribution": {str(v): lags.count(v) for v in sorted(set(lags))}, "conversion": manifest,
                "created": datetime.datetime.now().isoformat(timespec="seconds")}
        out = tg.TTSCorpus(voice=voice, texts=list(corpus.texts), frames=frames, silence=corpus.silence, meta=meta)
        path = out.save(tg.corpus_path(cfg_p, voice, corpus_name))
        src_dir, dst_dir = sy.tts_dir(cfg_p, src_voice, corpus_name), sy.tts_dir(cfg_p, voice, corpus_name)
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)  # same frame timing: the source voice's forced syllables apply unchanged
        log(f"{corpus_name}: {n} converted cues, lag {lag} frames ({meta['lag_distribution']}) -> {path}")
        meta_all[corpus_name] = {k2: v for k2, v in meta.items() if k2 != "conversion"}
    (work / "encode_summary.json").write_text(json.dumps(meta_all, indent=1) + "\n")


def cmd_check(cfg) -> None:
    import numpy as np

    from word_splice import their_session
    from wvx import alignment as al, kunz_experiment as kx, syllables as sy, targets as tg, voice_conversion as vc
    from wvx import wairagkar_data
    from wvx.provenance import code_version

    k, cfg_d, cfg_p = _cfgs(cfg)
    ev = k["check_session"]
    train = json.loads(resolve_path(k["train_trials_from"]).read_text())["train_trials"]
    sess = wairagkar_data.load_session(resolve_path(cfg_d["sanity"]["data_dir"]), ev, 0, cfg_d)
    _, _, syl_e, _, _ = their_session(ev, cfg_d, cfg_p)
    trials = [sess.trials[i] for i in train]
    preds, header = sy.load_decoded(sy.decoded_dir(cfg_p, k["auto_cell"]) / "predictions.npz")
    out = {"session": ev, "n_trials": len(train), "code_version": code_version()}
    for voice in (k["source_voice"], k["voice"]):
        corpus = tg.TTSCorpus.load(tg.corpus_path(cfg_p, voice, ev))
        sil20 = tg.silence_frame(corpus, cfg_p)
        table = sy.load_tts_syllables(cfg_p, voice, ev)
        tts_segs = [table[tg.text_key(t.text)] for t in trials]
        ok = [bool(len(syl_e[i])) for i in train]
        oracle = vc.oracle_targets(trials, [syl_e[i] for i in train], corpus, table, cfg_p, valid=ok)
        auto_segs = [sy.decoder_segments("current", al.envelope_db(tg.prediction_timeline(preds[i], t.neural.shape[0],
                                                                                          sil20, cfg_p), cfg_p), cfg_p, None)
                     for i, t in zip(train, trials)]
        _, auto = tg.align_trials(None, trials, corpus, cfg_p, reorder=True, preds=[preds[i] for i in train],
                                  tts_segments=tts_segs, pred_segments=auto_segs, arm="B", log=None)
        r_o = [vc.trial_r(o[0], t.target) for o, t in zip(oracle, trials) if o is not None]
        r_a = [vc.trial_r(tg.decoder_target(a.target_raw, cfg_p), t.target) for a, t in zip(auto, trials)]
        out[voice] = {"oracle_r": kx.describe(r_o), "auto_r": kx.describe(r_a)}
        log(f"{voice}: oracle r {np.nanmean(r_o):.3f}, auto r {np.nanmean(r_a):.3f}")
    res = resolve_path("results/knnvc")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{k['name']}_check.json").write_text(json.dumps(kx.portable(out), indent=1, default=float) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "encode", "check"])
    ap.add_argument("--config", default="configs/experiments/knnvc.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    t0 = time.time()
    {"prep": cmd_prep, "encode": cmd_encode, "check": cmd_check}[args.step](cfg)
    log(f"{args.step} done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
