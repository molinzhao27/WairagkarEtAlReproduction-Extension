#!/usr/bin/env python
"""Zero-shot voice-cloning TTS with F5-TTS (D-055), run with the F5 environment's interpreter (host:
/workspace/envs/wvx-f5/bin/python). No wvx imports.

    <f5-env>/bin/python scripts/f5_generate.py --ref-wav REF.wav --ref-text "..." --texts TEXTS.json --out-dir DIR \\
        [--hf-cache DIR] [--seed 42] [--nfe 32] [--threads 8] [--limit N]

TEXTS.json: a list of cue strings; output DIR/sNNNN.wav (24 kHz, F5's vocoder rate) in list order, plus
DIR/manifest.json (model, checkpoint sha256, settings, per-cue durations). Existing outputs are kept (resumable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-wav", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="regenerate just these cue indices (overwriting), e.g. after a failed transcription check")
    ap.add_argument("--attempt", type=int, default=0, help="seed = seed + 1000 * attempt + index")
    args = ap.parse_args(argv)

    import soundfile as sf
    import torch

    torch.set_num_threads(args.threads)
    from f5_tts.api import F5TTS

    t0 = time.time()
    f5 = F5TTS(model="F5TTS_v1_Base", device=args.device, hf_cache_dir=args.hf_cache)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = json.loads(Path(args.texts).read_text())[: args.limit]
    durations, seeds = {}, {}
    mpath = out / "manifest.json"
    if mpath.exists():
        seeds = json.loads(mpath.read_text()).get("seeds", {})
    redo = set(args.only or [])
    for j, text in enumerate(texts):
        path = out / f"s{j:04d}.wav"
        if not path.exists() or j in redo:
            seed = args.seed + 1000 * args.attempt + j
            wav, sr, _ = f5.infer(args.ref_wav, args.ref_text, text, nfe_step=args.nfe, seed=seed,
                                  remove_silence=False, show_info=lambda *a, **k: None)
            seeds[f"s{j:04d}"] = seed
            tmp = path.with_suffix(".partial.wav")
            sf.write(str(tmp), wav, sr, subtype="PCM_16")
            tmp.replace(path)
        info = sf.info(str(path))
        durations[f"s{j:04d}"] = info.frames / info.samplerate
        if (j + 1) % 25 == 0:
            print(f"[f5] {j + 1}/{len(texts)} ({time.time() - t0:.0f} s)", flush=True)
    ckpts = sorted(Path(args.hf_cache).rglob("model_1250000.safetensors")) if args.hf_cache else []
    manifest = {"model": "F5TTS_v1_Base", "f5_tts": __import__("f5_tts").__dict__.get("__version__", "1.1.22"),
                "checkpoint_sha256": sha256(ckpts[0]) if ckpts else None, "torch": torch.__version__,
                "ref_wav": args.ref_wav, "ref_text": args.ref_text, "seed": args.seed, "nfe_step": args.nfe,
                "device": args.device, "n": len(texts), "seeds": seeds, "durations_s": durations,
                "wall_time_s": time.time() - t0}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[f5] {len(texts)} cues in {time.time() - t0:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
