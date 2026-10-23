#!/usr/bin/env python
"""Mac only: macOS ``say`` TTS -> LPCNet features; rank voices against their norm JSON; push features.

Audio exists only inside a temporary directory and is deleted as soon as it is encoded. What is kept
is the raw LPCNet frames per sentence (``wvx.targets.TTSCorpus``), which ``generate --push`` sends to
the host under data/processed/tts/<voice_slug>/<corpus>.npz.

    python scripts/make_tts.py voices
    python scripts/make_tts.py rank --sentences S.json
    python scripts/make_tts.py generate --sentences S.json --corpus kunz_50words [--voice V] [--push]

Sentences: a JSON list of strings, a JSON object of lists (their union, first occurrence order), or a
text file with one sentence per line. Duplicates (after whitespace normalization) are dropped.

Per sentence (``tts`` section of configs/experiments/kunz_targets.yaml): ``say -v VOICE -f text.txt
--file-format=WAVE --data-format=LEI16 -o x.wav`` at the voice's native rate, ``pad_ms`` of digital
silence added on both sides, then ``wvx.synthesis.encode_wav`` (their encode steps: 16 kHz s16le with
ffmpeg, ``lpcnet_demo -features``) -> (n, 36) float32. The silence frame is the encoding of
``silence_ms`` of zeros.

``rank`` (voice identification): every installed English voice except the novelty voices in
``voice_id.exclude`` speaks the corpus; ``wvx.targets.rank_voice`` compares its LPCNet statistics with
the mean/s.d. in their norm JSON (silence share fitted per voice) and reports the median voiced F0.
The best voice with a male-range F0 is the suggested choice. Writes
results/alignment_validation/voice_ranking.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import REPO_ROOT  # noqa: E402
from wvx.provenance import code_version, config_sha256  # noqa: E402
from wvx.seeding import set_seed  # noqa: E402
from wvx import synthesis, targets  # noqa: E402

VOICE_LINE = re.compile(r"^(?P<name>.+?)\s+(?P<locale>[a-z]{2,3}_[A-Za-z0-9]{2,4})\s+#")


# ------------------------------------------------------------------------------------ helpers


def mac_version() -> dict:
    out = {}
    for key in ("productVersion", "buildVersion"):
        res = subprocess.run(["sw_vers", f"-{key}"], capture_output=True, text=True)
        out[key] = res.stdout.strip()
    return out


def installed_voices(cfg) -> list[dict]:
    res = subprocess.run([cfg["tts"]["say_binary"], "-v", "?"], capture_output=True, text=True, check=True)
    voices = []
    for line in res.stdout.splitlines():
        m = VOICE_LINE.match(line)
        if m:
            voices.append({"name": m.group("name").strip(), "locale": m.group("locale")})
    counts: dict = {}
    for v in voices:
        counts[v["name"]] = counts.get(v["name"], 0) + 1
    for v in voices:
        v["listed_twice"] = counts[v["name"]] > 1   # by name, `say -v` then uses the compact voice
    return voices


def candidate_voices(cfg) -> tuple[list[str], list[dict]]:
    vcfg = cfg["voice_id"]
    exclude = set(vcfg["exclude"])
    chosen, skipped, seen = [], [], set()
    for v in installed_voices(cfg):
        base = v["name"].split(" (")[0]
        if not v["locale"].startswith(vcfg["locale_prefix"]) or v["name"] in seen:
            continue
        seen.add(v["name"])
        if base in exclude:
            skipped.append({**v, "reason": "novelty voice"})
        else:
            chosen.append(v["name"])   # a name listed twice resolves to one voice (checked: the compact one)
    return chosen, skipped


def voice_genders() -> dict:
    """(base name, locale) -> AVFoundation gender (0 unspecified, 1 male, 2 female), via osascript."""
    js = ('ObjC.import("AVFoundation"); var v = $.AVSpeechSynthesisVoice.speechVoices; var o = [];'
          'for (var i = 0; i < v.count; i++) { var x = v.objectAtIndex(i);'
          'o.push(ObjC.unwrap(x.name) + "\\t" + ObjC.unwrap(x.language) + "\\t" + x.gender); } o.join("\\n")')
    res = subprocess.run(["osascript", "-l", "JavaScript", "-e", js], capture_output=True, text=True)
    out: dict = {}
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            out.setdefault((parts[0], parts[1].replace("-", "_")), int(parts[2]))
    return out


def check_voice(voice: str, cfg) -> None:
    """`say` silently falls back to the default voice for unknown names, so check the name first."""
    names = [v["name"] for v in installed_voices(cfg)]
    if voice not in names:
        raise SystemExit(f"voice {voice!r} is not listed by `say -v ?`")


def read_sentences(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        items = data if isinstance(data, list) else [s for v in data.values() for s in v]
    else:
        items = path.read_text().splitlines()
    out, seen = [], set()
    for s in items:
        k = targets.text_key(s)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def corpus_digest(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


# ---------------------------------------------------------------------------------- TTS + LPCNet


def say_to_frames(text: str, voice: str, workdir: Path, cfg) -> np.ndarray:
    """One sentence -> (n, 36) raw LPCNet frames. All files are deleted before returning."""
    import soundfile

    t = cfg["tts"]
    d = Path(tempfile.mkdtemp(dir=workdir))
    try:
        spoken = text
        if t["word_pause_ms"]:   # `say` embedded command: silence between words (T15 spoke word by word)
            spoken = f" [[slnc {int(t['word_pause_ms'])}]] ".join(text.split())
        (d / "text.txt").write_text(spoken)
        cmd = [t["say_binary"], "-v", voice, "-f", str(d / "text.txt"), f"--file-format={t['file_format']}",
               f"--data-format={t['data_format']}", "-o", str(d / "say.wav")]
        if t["rate_wpm"] is not None:
            cmd[1:1] = ["-r", str(t["rate_wpm"])]
        for attempt in range(int(t["retries"]) + 1):   # `say` occasionally hangs under load
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=float(t["timeout_s"]))
                break
            except subprocess.TimeoutExpired:
                if attempt == int(t["retries"]):
                    raise
        x, fs = soundfile.read(str(d / "say.wav"), dtype="float32", always_2d=False)
        if x.ndim != 1 or x.size == 0:
            raise RuntimeError(f"say produced {x.shape} samples for {text!r}")
        pad = np.zeros(int(round(fs * float(t["pad_ms"]) / 1000)), np.float32)
        soundfile.write(str(d / "padded.wav"), np.concatenate([pad, x, pad]), fs, subtype="PCM_16")
        return synthesis.encode_wav(d / "padded.wav", cfg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def silence_frames(workdir: Path, cfg) -> np.ndarray:
    import soundfile

    d = Path(tempfile.mkdtemp(dir=workdir))
    try:
        sr = int(cfg["synthesis"]["sample_rate"])
        n = int(round(sr * float(cfg["tts"]["silence_ms"]) / 1000))
        soundfile.write(str(d / "zeros.wav"), np.zeros(n, np.float32), sr, subtype="PCM_16")
        f = synthesis.encode_wav(d / "zeros.wav", cfg)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if not np.all(f == f[0]):
        raise RuntimeError("LPCNet frames of digital silence are not constant")
    return f[0]


def build_corpus(voice: str, texts: list[str], workdir: Path, cfg) -> targets.TTSCorpus:
    with ThreadPoolExecutor(max_workers=int(cfg["tts"]["n_workers"])) as ex:
        frames = list(ex.map(lambda s: say_to_frames(s, voice, workdir, cfg), texts))
    meta = {
        "engine": cfg["tts"]["engine"],
        "macos": mac_version(),
        "say": {k: cfg["tts"][k] for k in ("rate_wpm", "file_format", "data_format", "pad_ms", "word_pause_ms")},
        "lpcnet_binary_sha256": hashlib.sha256(synthesis.lpcnet_binary(cfg).read_bytes()).hexdigest(),
        "corpus_sha256": corpus_digest(texts),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "frame_layout": "raw LPCNet frames (n, 36) from lpcnet_demo -features; first pad_ms and last pad_ms are digital silence",
    }
    return targets.TTSCorpus(voice=voice, texts=texts, frames=frames, silence=silence_frames(workdir, cfg), meta=meta)


# ----------------------------------------------------------------------------------- commands


def cmd_voices(cfg, args) -> int:
    chosen, skipped = candidate_voices(cfg)
    print("candidates:", *chosen, sep="\n  ")
    print("skipped:", *[f"{s['name']} ({s['reason']})" for s in skipped], sep="\n  ")
    return 0


def cmd_rank(cfg, args) -> int:
    texts = read_sentences(Path(args.sentences))
    k = int(cfg["voice_id"]["n_sentences"])
    if k < len(texts):   # seeded subset, kept in corpus order
        keep = np.sort(np.random.default_rng(int(cfg["seed"])).choice(len(texts), size=k, replace=False))
        texts = [texts[i] for i in keep]
    voices = args.voices or candidate_voices(cfg)[0]
    _, skipped = candidate_voices(cfg)
    vcfg = cfg["voice_id"]
    ref = json.loads(Path(args.reference).read_text())["pooled"] if args.reference else None
    genders = voice_genders()
    locale = {v["name"]: v["locale"] for v in installed_voices(cfg)}
    rows = []
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as td:
        for v in voices:
            check_voice(v, cfg)
            c = build_corpus(v, texts, Path(td), cfg)
            r = {"voice": v, "json_fit": targets.rank_voice(c.frames, c.silence, cfg),
                 "speech_moments": targets.speech_moments(c.frames, cfg, float(vcfg["speech_mask_db"]))}
            if ref is not None:
                r["vs_their_targets"] = targets.reference_distance(r["speech_moments"], ref, vcfg["reference_features"])
            base = v.split(" (")[0]
            r["gender"] = genders.get((base, locale.get(v, "")), None)
            r["male"] = r["gender"] == 1 or (r["gender"] == 0 and base in vcfg["male_voices_unlabelled"])
            rows.append(r)
            vt = r.get("vs_their_targets", {})
            print(f"[rank] {v:28s} json {r['json_fit']['distance']:.4f} (silence {r['json_fit']['silence_share']:.3f}) "
                  f"targets {vt.get('distance', float('nan')):.4f} F0 {r['json_fit']['median_voiced_f0_hz']:.1f} Hz "
                  f"({vt.get('median_f0_semitones_vs_ref', float('nan')):+.1f} st) male={r['male']}", flush=True)
    for key in ("json_fit", "vs_their_targets"):
        ranked = sorted((r for r in rows if key in r), key=lambda r: r[key]["distance"])
        for k, r in enumerate(ranked):
            r[f"rank_{key}"] = k + 1
    order = "vs_their_targets" if ref is not None else "json_fit"
    rows.sort(key=lambda r: r[order]["distance"])
    male = [r for r in rows if r["male"]]
    out = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "macos": mac_version(),
        "note": "Their TTS ran on macOS 13.5.1; voices on this macOS may differ from that release.",
        "corpus": {"file": Path(args.sentences).name, "n_sentences": len(texts), "sha256": corpus_digest(texts),
                   "texts": texts},
        "method": {
            "json_fit": "wvx.targets.rank_voice: mean over voice_id.distance_features of z(mean)^2 + "
                        "log(sd ratio)^2 against their norm JSON, silence share fitted per voice",
            "vs_their_targets": "wvx.targets.reference_distance: same distance on speech-frame moments "
                                "(voice_id.reference_features) against their released targets",
            "order": order,
        },
        "reference": args.reference,
        "suggested_voice": male[0]["voice"] if male else None,
        "skipped_voices": skipped,
        "ranking": rows,
        "voice_id_config": cfg["voice_id"],
        "code_version": code_version(),
        "config_sha256": config_sha256(cfg),
    }
    path = Path(args.out).resolve() if args.out else Path(cfg["paths"]["alignment_validation"]) / "voice_ranking.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[rank] suggested: {out['suggested_voice']}  -> {path.relative_to(REPO_ROOT)}")
    return 0


def cmd_generate(cfg, args) -> int:
    voice = args.voice or cfg["tts"]["voice"]
    if not voice:
        raise SystemExit("no voice: pass --voice or set tts.voice in the config")
    check_voice(voice, cfg)
    texts = read_sentences(Path(args.sentences))
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as td:
        work, out_root = Path(td) / "work", Path(td) / "tts"
        work.mkdir()
        c = build_corpus(voice, texts, work, cfg)
        path = c.save(out_root / targets.voice_slug(voice) / f"{args.corpus}.npz")
        n = np.array([f.shape[0] for f in c.frames])
        print(f"[generate] {voice}: {len(c)} sentences, frames/sentence {n.min()}-{n.max()} "
              f"(median {int(np.median(n))}), {path.stat().st_size / 1e6:.1f} MB")
        if args.push:
            subprocess.run([str(REPO_ROOT / "scripts" / "remote" / "sync.sh"), "data", f"{out_root}/",
                            "processed/tts/"], check=True)
            print(f"[generate] pushed to data/processed/tts/{targets.voice_slug(voice)}/{args.corpus}.npz")
    # the TemporaryDirectory (frames included) is gone here; nothing stays on the Mac
    return 0


def main(argv=None) -> int:
    if platform.system() != "Darwin":
        raise SystemExit("make_tts.py needs macOS `say`")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=targets.DEFAULT_TARGETS_CONFIG)
    ap.add_argument("--tmp-dir", default=None, help="parent of the temporary directory (default: system temp)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("voices")
    r = sub.add_parser("rank")
    r.add_argument("--sentences", required=True)
    r.add_argument("--voices", nargs="*", default=None, help="default: every candidate voice")
    r.add_argument("--out", default=None, help="output JSON (default: results/alignment_validation/voice_ranking.json)")
    r.add_argument("--reference", default=None,
                   help="their_target_speech_stats.json (speech-frame moments of their released targets)")
    g = sub.add_parser("generate")
    g.add_argument("--sentences", required=True)
    g.add_argument("--corpus", required=True, help="output name, e.g. kunz_50words")
    g.add_argument("--voice", default=None, help="default: tts.voice from the config")
    g.add_argument("--push", action="store_true", help="send to the host's data/processed/tts/")
    g.add_argument("--word-pause-ms", type=int, default=None, help="override tts.word_pause_ms")
    args = ap.parse_args(argv)
    over = {"tts": {"word_pause_ms": args.word_pause_ms}} if getattr(args, "word_pause_ms", None) else None
    cfg = targets.load_targets_config(args.config, overrides=over)
    cfg["synthesis"]["tmp_root"] = args.tmp_dir   # Mac: encode temp files in our temp dir, never under data/
    set_seed(int(cfg["seed"]))
    return {"voices": cmd_voices, "rank": cmd_rank, "generate": cmd_generate}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
