#!/usr/bin/env python
"""Host: TTS syllables by forced alignment + maximal onset (D-033, TTS side of every method).

    # project .venv (LPCNet): TTS audio for the aligner, and the LPCNet synthesis delay re-measured
    .venv/bin/python -I scripts/syllable_tts.py audio --corpus t15_day195
    # any Python (runs MFA from its env): phone/word alignment
    <syl-venv>/bin/python -I scripts/syllable_tts.py align --corpus t15_day195 --mfa-bin <mfa-env>/bin
    # syllable venv (cmudict): syllables per cue on the TTS feature timeline + count check vs CMUdict
    <syl-venv>/bin/python -I scripts/syllable_tts.py syllabify --corpus t15_day195

Common options: --config (default configs/experiments/syllable_benchmark.yaml), --voice (default tts.voice),
--threads N, --overwrite. Files (host only) under <paths.syllables>/tts/<voice_slug>/<corpus>/:

    mfa_in_<corpus>/uNNNN.wav|.lab   LPCNet resynthesis of the stored TTS features + the cue text (audio)
    index.json                       utterance names, cues, frame counts, per-utterance synthesis delay
    mfa_out_<corpus>/uNNNN.json      the aligner's words and phones (MFA --output_format json)
    mfa_run.json                     MFA command, version, model checksums, wall time
    syllables.json                   per cue: words, syllables (audio s) and frames (TTS feature timeline)

Results (small JSON) in <paths.syllable_benchmark>/: tts_audio_<voice>_<corpus>.json (delay check) and
tts_syllables_<voice>_<corpus>.json (alignment coverage, unknown words, syllable count vs CMUdict count).
The audio is resynthesized from the same LPCNet frames the targets are cut from, so a syllable's frames are
its audio times minus the measured synthesis delay (``syllables.audio_lag_ms``).
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.syllable_runtime import rel, runtime_record, sha256, start  # noqa: E402  (no numpy)


def _paths(cfg, voice, corpus):
    from wvx import syllables as sy

    d = sy.tts_dir(cfg, voice, corpus)
    return d, d / f"mfa_in_{corpus}", d / f"mfa_out_{corpus}"


def _guard(path: Path, overwrite: bool) -> None:
    """Refuse to replace an existing output (a non-empty directory or a file) unless ``overwrite``."""
    exists = any(path.iterdir()) if path.is_dir() else path.exists()
    if not exists:
        return
    if not overwrite:
        raise SystemExit(f"{rel(path)} exists; pass --overwrite to replace it")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


# -------------------------------------------------------------------------------------- audio


def cmd_audio(cfg, env_set, args) -> int:
    from wvx import b2v

    b2v.use_keras2()
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from wvx import synthesis, syllables as sy, targets as tg
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    synthesis.preflight(cfg)
    voice = args.voice or cfg["tts"]["voice"]
    cfile = tg.corpus_path(cfg, voice, args.corpus)
    corpus = tg.TTSCorpus.load(cfile)
    d, cdir, _ = _paths(cfg, voice, args.corpus)
    _guard(cdir, args.overwrite)
    cdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def one(i):
        raw = tg.reduce(corpus.frames[i], cfg)
        wav = synthesis.features_to_waveform(raw, cfg)
        name = sy.utterance_name(i)
        synthesis.write_wav(cdir / f"{name}.wav", wav, cfg)
        (cdir / f"{name}.lab").write_text(tg.text_key(corpus.texts[i]) + "\n")
        return {"name": name, "text": corpus.texts[i], "n_frames": int(raw.shape[0]),
                "lag_ms": sy.audio_feature_lag_ms(raw, wav, cfg)}

    with ThreadPoolExecutor(max_workers=int(cfg["benchmark"]["n_workers"])) as ex:
        utts = list(ex.map(one, range(len(corpus))))
    lags = np.array([u["lag_ms"] for u in utts])
    configured, tol = float(cfg["syllables"]["audio_lag_ms"]), float(cfg["syllables"]["lag_check"]["tolerance_ms"])
    lag = {"configured_ms": configured, "median_ms": float(np.median(lags)), "mean_ms": float(lags.mean()),
           "q25_ms": float(np.percentile(lags, 25)), "q75_ms": float(np.percentile(lags, 75)), "n": int(lags.size),
           "agrees": bool(abs(np.median(lags) - configured) <= tol), "tolerance_ms": tol}
    prov = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "voice": voice, "corpus": args.corpus,
            "corpus_file": rel(cfile), "corpus_sha256": sha256(cfile), "tts_meta": corpus.meta,
            "runtime": runtime_record(cfg, env_set), "code_version": code_version(), "config_sha256": config_sha256(cfg)}
    sy.save_json(d / "index.json", {**prov, "audio_dir": cdir.name, "lag": lag, "utterances": utts})
    out = Path(cfg["paths"]["syllable_benchmark"]) / f"tts_audio_{tg.voice_slug(voice)}_{args.corpus}.json"
    sy.save_json(out, {**prov, "n_utterances": len(utts), "lag": lag, "wall_time_s": time.time() - t0})
    print(f"[tts audio] {args.corpus}: {len(utts)} utterances -> {rel(cdir)}; synthesis delay median "
          f"{lag['median_ms']:.1f} ms (configured {configured:.1f}){'' if lag['agrees'] else '  !!! DISAGREES'}")
    return 0 if lag["agrees"] else 2


# -------------------------------------------------------------------------------------- align


def cmd_align(cfg, env_set, args) -> int:
    from wvx import syllables as sy
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    fa = cfg["syllables"]["forced_alignment"]
    voice = args.voice or cfg["tts"]["voice"]
    d, cdir, odir = _paths(cfg, voice, args.corpus)
    if not (d / "index.json").is_file():
        raise SystemExit(f"{rel(d / 'index.json')} missing: run the 'audio' step first")
    mfa_bin = Path(args.mfa_bin).resolve()
    mfa = mfa_bin / "mfa"
    root = Path(cfg["paths"]["mfa_root"])
    models = {
        "acoustic": (root / "pretrained_models" / "acoustic" / f"{fa['acoustic_model']}.zip",
                     fa["acoustic_model"], fa["acoustic_model_version"], fa["acoustic_model_sha256"]),
        "dictionary": (root / "pretrained_models" / "dictionary" / f"{fa['dictionary']}.dict",
                       fa["dictionary"], fa["dictionary_version"], fa["dictionary_sha256"]),
    }
    for kind, (p, name, ver, want) in models.items():
        if not p.is_file():
            raise SystemExit(f"{rel(p)} missing: MFA_ROOT_DIR={rel(root)} mfa model download {kind} {name} --version {ver}")
        if sha256(p) != want:
            raise SystemExit(f"{rel(p)}: SHA-256 differs from the configured one")
    _guard(odir, args.overwrite)
    env = {**os.environ, "MFA_ROOT_DIR": str(root), "PATH": f"{mfa_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
    version = subprocess.run([str(mfa), "version"], env=env, capture_output=True, text=True, check=True).stdout.strip()
    if version != str(fa["version"]):
        raise SystemExit(f"MFA {version} found, the config pins {fa['version']}")
    cmd = [str(mfa), "align", *[str(a) for a in fa["args"]], "-j", str(int(args.jobs or fa["num_jobs"])),
           str(cdir), fa["acoustic_model"], fa["dictionary"], str(odir)]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    n_out = len(list(odir.glob("*.json"))) if odir.is_dir() else 0
    rec = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "command": [rel(c) if os.sep in c else c for c in cmd],
           "mfa_version": version, "returncode": proc.returncode, "n_outputs": n_out,
           "models": {k: {"file": rel(p), "name": nm, "version": v, "sha256": w} for k, (p, nm, v, w) in models.items()},
           "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:], "wall_time_s": time.time() - t0,
           "runtime": runtime_record(cfg, env_set), "code_version": code_version(), "config_sha256": config_sha256(cfg)}
    sy.save_json(d / "mfa_run.json", rec)
    print(f"[tts align] {args.corpus}: MFA {version} exit {proc.returncode}, {n_out} alignments -> {rel(odir)} "
          f"in {rec['wall_time_s']:.0f} s")
    return proc.returncode


# ---------------------------------------------------------------------------------- syllabify


def cmd_syllabify(cfg, env_set, args) -> int:
    import importlib.metadata

    import cmudict
    import numpy as np

    from wvx import syllables as sy, targets as tg
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    s = cfg["syllables"]
    voice = args.voice or cfg["tts"]["voice"]
    d, _, odir = _paths(cfg, voice, args.corpus)
    out_file = d / s["forced_alignment"]["syllables_file"]
    _guard(out_file, args.overwrite)
    index = sy.load_json(d / "index.json")
    first = sy.cmudict_first_pronunciations(cmudict.dict())
    onsets = sy.legal_onsets(first.values(), int(s["syllabification"]["min_onset_words"]))
    utts, missing, mismatches = [], [], []
    for u in index["utterances"]:
        js = odir / f"{u['name']}.json"
        if not js.is_file():
            missing.append(u["text"])
            continue
        words, phones = sy.read_aligner_json(js, cfg)
        syl = sy.tts_syllables(words, phones, onsets, cfg)
        frames = sy.seconds_to_frames(syl["syllables_s"], cfg, u["n_frames"], keep_empty=True)
        n_dict = sy.dictionary_syllable_count(syl["words"], first)
        rec = {"name": u["name"], "text": u["text"], **syl, "frames": frames.tolist(), "n_syllables": int(len(frames)),
               "n_dictionary": n_dict}
        if n_dict != len(frames):
            mismatches.append({k: rec[k] for k in ("text", "words", "n_syllables", "n_dictionary")})
        utts.append(rec)
    n = np.array([u["n_syllables"] for u in utts])
    dur = np.concatenate([np.diff(np.asarray(u["frames"]), axis=1).ravel() for u in utts]) * sy.frame_ms(cfg) if utts else np.zeros(0)
    versions = {"cmudict": importlib.metadata.version("cmudict")}
    prov = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "voice": voice, "corpus": args.corpus,
            "settings": {"syllabification": s["syllabification"], "audio_lag_ms": s["audio_lag_ms"],
                         "forced_alignment": s["forced_alignment"]},
            "versions": versions, "n_legal_onsets": len(onsets),
            "runtime": runtime_record(cfg, env_set), "code_version": code_version(), "config_sha256": config_sha256(cfg)}
    sy.save_json(out_file, {**prov, "legal_onsets": sorted(" ".join(o) for o in onsets), "utterances": utts})
    summary = {
        **prov,
        "n_utterances": len(index["utterances"]), "n_aligned": len(utts), "missing_alignments": missing,
        "unknown_words": sorted({w for u in utts for w in u["unknown_words"]}),
        "words_without_vowel": sorted({w for u in utts for w in u["words_without_vowel"]}),
        "count_agreement_with_cmudict": float(np.mean([u["n_syllables"] == u["n_dictionary"] for u in utts])) if utts else None,
        "n_count_mismatches": len(mismatches), "count_mismatches": mismatches,
        "syllables_per_utterance": {"min": int(n.min()), "median": float(np.median(n)), "max": int(n.max())} if n.size else None,
        "syllable_ms": {"min": float(dur.min()), "median": float(np.median(dur)), "max": float(dur.max())} if dur.size else None,
    }
    out = Path(cfg["paths"]["syllable_benchmark"]) / f"tts_syllables_{tg.voice_slug(voice)}_{args.corpus}.json"
    sy.save_json(out, summary)
    print(f"[tts syllabify] {args.corpus}: {len(utts)}/{len(index['utterances'])} aligned, syllable count = CMUdict "
          f"count in {summary['count_agreement_with_cmudict']:.3f} of cues ({len(mismatches)} mismatches), "
          f"{len(summary['unknown_words'])} unknown words -> {rel(out_file)}, {rel(out)}")
    return 1 if missing or summary["unknown_words"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["audio", "align", "syllabify"])
    ap.add_argument("--config", default="configs/experiments/syllable_benchmark.yaml")
    ap.add_argument("--corpus", required=True, help="TTS corpus name, e.g. t15_day195, t15_day193, kunz_50words")
    ap.add_argument("--voice", default=None, help="default: tts.voice")
    ap.add_argument("--mfa-bin", default=None, help="align: bin directory of the MFA environment")
    ap.add_argument("--jobs", type=int, default=None, help="align: MFA -j (default syllables.forced_alignment.num_jobs)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.step == "align" and not args.mfa_bin:
        ap.error("align needs --mfa-bin")
    cfg, env_set = start(args.config, args.threads)
    return {"audio": cmd_audio, "align": cmd_align, "syllabify": cmd_syllabify}[args.step](cfg, env_set, args)


if __name__ == "__main__":
    sys.exit(main())
