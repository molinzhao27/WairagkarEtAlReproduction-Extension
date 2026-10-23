#!/usr/bin/env python
"""kNN-VC conversion (D-054), run with an interpreter that has PyTorch (on the host: /opt/worm-train/bin/python).

kNN-VC (Baas, van Niekerk & Kamper 2023; github.com/bshall/knn-vc, MIT): every 20 ms WavLM-Large layer-6 frame of
the source is replaced by the mean of its k nearest (cosine) frames in the target speaker's matching set, then
vocoded by their prematched HiFi-GAN. This script loads both models from local files (no download) and does not
need torchaudio (16 kHz mono 16-bit wavs via the stdlib ``wave``; no VAD: the matching wavs are speech segments).

    /opt/worm-train/bin/python scripts/knnvc_convert.py --knnvc DIR --matching-dir M --source-dir S --out-dir O \\
        [--topk 4] [--threads 8]

DIR holds code/ (the repo at a pinned commit), WavLM-Large.pt and prematch_g_02500000.pt. Every output has the
source's length (vocoder output padded or cut) and the matching set's median RMS. Writes O/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR = 16000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16 kHz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knnvc", required=True)
    ap.add_argument("--matching-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    root = Path(args.knnvc)
    sys.path.insert(0, str(root / "code"))
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    from wavlm.WavLM import WavLM, WavLMConfig

    t0 = time.time()
    ck = torch.load(root / "WavLM-Large.pt", map_location="cpu", weights_only=False)
    wavlm = WavLM(WavLMConfig(ck["cfg"]))
    wavlm.load_state_dict(ck["model"])
    wavlm.eval()
    h = AttrDict(json.loads((root / "code" / "hifigan" / "config_v1_wavlm.json").read_text()))
    gen = Generator(h)
    gen.load_state_dict(torch.load(root / "prematch_g_02500000.pt", map_location="cpu", weights_only=False)["generator"])
    gen.eval()
    gen.remove_weight_norm()

    @torch.inference_mode()
    def feats(x: np.ndarray):
        return wavlm.extract_features(torch.from_numpy(x)[None], output_layer=6, ret_layer_results=False)[0].squeeze(0)

    m_paths = sorted(Path(args.matching_dir).glob("*.wav"))
    m_wavs = [read_wav(p) for p in m_paths]
    matching = torch.cat([feats(x) for x in m_wavs], dim=0)
    target_rms = float(np.median([rms(x) for x in m_wavs]))
    m_norm = torch.linalg.vector_norm(matching, dim=-1)
    print(f"[knnvc] matching set: {len(m_paths)} wavs, {matching.shape[0]} frames "
          f"({matching.shape[0] * 0.02 / 60:.1f} min), median RMS {target_rms:.4f}; {time.time() - t0:.0f} s", flush=True)

    out_dir = Path(args.out_dir)
    s_paths = sorted(Path(args.source_dir).glob("*.wav"))
    t1 = time.time()
    for n, p in enumerate(s_paths):
        x = read_wav(p)
        with torch.inference_mode():
            q = feats(x)
            q_norm = torch.linalg.vector_norm(q, dim=-1)
            cos = (q @ matching.T) / (q_norm[:, None] * m_norm[None])          # cosine similarity
            idx = cos.topk(k=args.topk, dim=-1, largest=True).indices
            y = gen(matching[idx].mean(dim=1)[None]).squeeze().numpy()
        y = np.pad(y, (0, max(0, x.size - y.size)))[: x.size]
        r = rms(y)
        if r > 0:
            y = y * (target_rms / r)
        write_wav(out_dir / p.name, y)
        if (n + 1) % 50 == 0:
            print(f"[knnvc] converted {n + 1}/{len(s_paths)} ({time.time() - t1:.0f} s)", flush=True)
    manifest = {"method": "kNN-VC (Baas et al. 2023), WavLM-Large layer 6, prematched HiFi-GAN, no torchaudio/VAD",
                "topk": args.topk, "n_matching": len(m_paths), "matching_frames": int(matching.shape[0]),
                "target_rms": target_rms, "n_converted": len(s_paths), "torch": torch.__version__,
                "sha256": {f: sha256(root / f) for f in ("WavLM-Large.pt", "prematch_g_02500000.pt")},
                "wall_time_s": time.time() - t0}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[knnvc] {len(s_paths)} converted in {time.time() - t1:.0f} s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
