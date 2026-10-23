#!/usr/bin/env python
"""Voice-cloning TTS corpus in their voice (D-055): reference clip, and the generated audio -> a TTS corpus.

Host, project venv (CPU; ffmpeg from the tools env on PATH). Generation runs in the F5 env (scripts/f5_generate.py):

    .venv/bin/python -I scripts/f5_clone.py ref    --config configs/experiments/f5clone.yaml
    <f5-env>/bin/python scripts/f5_generate.py --ref-wav <work>/ref.wav --ref-text "$(cat <work>/ref.txt)" \\
        --texts <work>/texts_<corpus>.json --out-dir <work>/generated/<corpus> --hf-cache <models>   # per corpus
    .venv/bin/python -I scripts/f5_clone.py encode --config configs/experiments/f5clone.yaml
    # then forced syllables for the new voice (scripts/syllable_tts.py audio|align|syllabify --voice "<voice>"),
    # and the target check: scripts/knnvc.py check --config configs/experiments/f5clone.yaml

ref     their voice: ``ref_n_sentences`` 50-word trials of ``ref_session`` (never the evaluated session), each rebuilt
        from its own word segments (LPCNet audio of their raw word targets, ± 2 frames) joined by ``word_gap_s``,
        sentences joined by ``sentence_gap_s`` (their targets follow the participant's slow speech; the pauses are
        removed so the clone hears fluent speech), up to ``ref_max_s``; ref.txt = the sentences' cues.
        Also writes texts_<corpus>.json: the cues of each corpus in the source voice's corpus order.
qc      transcribes every generated cue; the ones not word-perfect are regenerated with a new seed (F5 sometimes
        speaks part of the reference text before a short cue; f5_generate.py --only <redo> --attempt N, then
        qc --recheck), up to ``qc_max_attempts``; the rest are reported.
encode  generated 24 kHz audio -> 16 kHz -> LPCNet features -> a TTS corpus of voice ``voice`` (same cue order).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import apply_thread_env, load_experiment_config  # noqa: E402


def log(msg: str) -> None:
    print(f"[f5_clone {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def cmd_ref(cfg) -> None:
    import numpy as np

    from word_splice import their_session
    from wvx import synthesis, targets as tg
    from wvx.word_splice import words_of

    f = cfg["f5clone"]
    cfg_d, cfg_p = load_experiment_config(f["data_config"]), load_experiment_config(f["placement_config"])
    work = resolve_path(f["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    sr, spf = int(cfg_p["synthesis"]["sample_rate"]), int(cfg_p["synthesis"]["samples_per_frame"])
    raw, words, _, cues, desc = their_session(f["ref_session"], cfg_d, cfg_p)
    pieces, texts, total = [], [], 0.0
    for i, d in enumerate(desc):
        if d != f["ref_category"] or not len(words[i]) or len(words[i]) != len(words_of(cues[i])):
            continue
        audio = synthesis.features_to_waveform(raw[i], cfg_p)
        sent = []
        for a, b in words[i]:
            sent.append(audio[max(int(a) - 2, 0) * spf: (int(b) + 2) * spf])
            sent.append(np.zeros(int(f["word_gap_s"] * sr), np.float32))
        sent = np.concatenate(sent[:-1])
        if total + sent.size / sr > float(f["ref_max_s"]):
            continue
        pieces += [sent, np.zeros(int(f["sentence_gap_s"] * sr), np.float32)]
        texts.append(cues[i])
        total += sent.size / sr + f["sentence_gap_s"]
        if len(texts) >= int(f["ref_n_sentences"]):
            break
    ref = np.concatenate(pieces[:-1])
    synthesis.write_wav(work / "ref.wav", ref, cfg_p)
    (work / "ref.txt").write_text(" ".join(texts) + "\n")
    log(f"reference: {len(texts)} sentences of {f['ref_session']}, {ref.size / sr:.1f} s: {' | '.join(texts)}")
    for corpus_name in f["corpora"]:
        corpus = tg.TTSCorpus.load(tg.corpus_path(cfg_p, f["source_voice"], corpus_name))
        (work / f"texts_{corpus_name}.json").write_text(json.dumps(list(corpus.texts), indent=0) + "\n")
        log(f"texts: {corpus_name}: {len(corpus)} cues")


def cmd_qc(cfg, recheck: bool = False) -> None:
    """Transcribe every generated cue (wvx.asr, as the scoring does); qc_<corpus>.json lists the cues whose transcript
    is not word-perfect (WER > 0), to regenerate with scripts/f5_generate.py --only ... --attempt N. ``recheck``:
    transcribe only the cues of the existing qc file's redo list (just regenerated) and keep the other rows."""
    import librosa

    from wvx import asr

    f = cfg["f5clone"]
    work = resolve_path(f["work_dir"])
    acfg = {"asr": {"cpu_threads": int(f.get("qc_cpu_threads", 4)), "model": {"allow_download": False}}}
    for corpus_name in f["corpora"]:
        texts = json.loads((work / f"texts_{corpus_name}.json").read_text())
        qpath = work / f"qc_{corpus_name}.json"
        prev = json.loads(qpath.read_text()) if recheck else None
        rows = prev["rows"] if prev else [None] * len(texts)
        for j, text in enumerate(texts):
            if prev and j not in prev["redo"]:
                continue
            y, _ = librosa.load(str(work / "generated" / corpus_name / f"s{j:04d}.wav"), sr=16000)
            r = asr.evaluate_wav(y, 16000, text, acfg)
            n_checks = rows[j]["n_checks"] + 1 if prev else 1
            rows[j] = {"cue": j, "text": text, "hyp": r.get("hyp_raw"), "wer": r.get("wer"), "n_checks": n_checks}
        bad = [r["cue"] for r in rows if not (r["wer"] == 0)]
        qpath.write_text(json.dumps({"n": len(rows), "n_not_perfect": len(bad), "redo": bad, "rows": rows},
                                    indent=1) + "\n")
        log(f"qc {corpus_name}: {len(rows) - len(bad)}/{len(rows)} word-perfect; redo {bad[:20]}{'...' if len(bad) > 20 else ''}")


def cmd_encode(cfg) -> None:
    import librosa
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    from wvx import synthesis, targets as tg

    f = cfg["f5clone"]
    cfg_p = load_experiment_config(f["placement_config"])
    work = resolve_path(f["work_dir"])
    sr = int(cfg_p["synthesis"]["sample_rate"])
    for corpus_name in f["corpora"]:
        src = tg.TTSCorpus.load(tg.corpus_path(cfg_p, f["source_voice"], corpus_name))
        gen = work / "generated" / corpus_name
        manifest = json.loads((gen / "manifest.json").read_text())
        if manifest["n"] != len(src):
            raise SystemExit(f"{corpus_name}: {manifest['n']} generated, {len(src)} cues")

        def one(j):
            y, _ = librosa.load(str(gen / f"s{j:04d}.wav"), sr=sr, mono=True)
            p = synthesis.write_wav(work / "wav16k" / corpus_name / f"s{j:04d}.wav", y, cfg_p)
            return synthesis.encode_wav(p, cfg_p).astype(np.float32)

        with ThreadPoolExecutor(int(f["workers"])) as ex:
            frames = list(ex.map(one, range(len(src))))
        meta = {"method": "F5-TTS zero-shot clone (D-055)", "ref_text": (work / "ref.txt").read_text().strip(),
                "generation": {k: v for k, v in manifest.items() if k != "durations_s"},
                "created": datetime.datetime.now().isoformat(timespec="seconds")}
        out = tg.TTSCorpus(voice=f["voice"], texts=list(src.texts), frames=frames, silence=src.silence, meta=meta)
        path = out.save(tg.corpus_path(cfg_p, f["voice"], corpus_name))
        log(f"{corpus_name}: {len(frames)} cues, {sum(len(x) for x in frames) / 100 / 60:.1f} min -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ref", "qc", "encode"])
    ap.add_argument("--config", default="configs/experiments/f5clone.yaml")
    ap.add_argument("--recheck", action="store_true", help="qc: only the cues of the existing redo list")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    apply_thread_env(cfg)
    t0 = time.time()
    if args.step == "qc":
        cmd_qc(cfg, recheck=args.recheck)
    else:
        {"ref": cmd_ref, "encode": cmd_encode}[args.step](cfg)
    log(f"{args.step} done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
