#!/usr/bin/env python
"""Host, syllable venv: decoder-side syllable segmenters on a cell's LPCNet audio (D-033 methods nuclei, sylber).

    <syl-venv>/bin/python -I scripts/syllable_segment.py fetch                       # pinned Sylber files (once)
    CUDA_VISIBLE_DEVICES= nice -n 24 <syl-venv>/bin/python -I scripts/syllable_segment.py run --cell day195_m172 \\
        --method nuclei|sylber [--threads N] [--max-trials N] [--overwrite]

``run`` reads <paths.syllables>/decoded/<cell>/ (scripts/syllable_decode.py) and writes
<paths.syllables>/segments/<cell>__<method>.json: per trial the segmenter's raw output in seconds of audio
(nuclei: nucleus times, the cuts between them and the script's counts; sylber: segments), with the settings,
package versions, checkpoint checksum and the decoded cell's header (model SHA-256). Conversion to frames
on the neural timeline happens at alignment time (wvx.syllables.decoder_segments). Method ``current`` needs
no file. Settings: syllables.segmenters (published defaults; wvx.syllable_segmenters has the references).
``--max-trials N`` reads/writes the ``<cell>__first<N>`` smoke cell. CPU only.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.syllable_runtime import rel, runtime_record, start  # noqa: E402  (no numpy)


def cmd_fetch(cfg, env_set, args) -> int:
    from wvx import syllable_segmenters as seg

    got = seg.fetch_sylber(cfg["syllables"]["segmenters"]["sylber"], cfg["paths"]["sylber_models"])
    for k, v in got.items():
        print(f"[fetch] {k} @ {v['revision'][:12]}: {v['bytes']:,} B, sha256 {v['sha256']} (verified)")
    return 0


def cmd_run(cfg, env_set, args) -> int:
    from wvx import syllable_segmenters as seg, syllables as sy
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    cell = args.cell + (f"__first{args.max_trials}" if args.max_trials else "")
    ddir = sy.decoded_dir(cfg, cell)
    _, header = sy.load_decoded(ddir / "predictions.npz")
    if header.get("wav") is None:
        raise SystemExit(f"{rel(ddir)} has no audio (decoded with --no-wav)")
    wav_dir = ddir / header["wav"]["dir"]
    wavs = [wav_dir / sy.trial_wav_name(i) for i in range(header["n_trials"])]
    absent = [w for w in wavs if not w.is_file()]
    if absent:
        raise SystemExit(f"{len(absent)} of {len(wavs)} audio files missing in {rel(wav_dir)}, e.g. {absent[0].name}")
    out = sy.segments_path(cfg, cell, args.method)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{rel(out)} exists; pass --overwrite to segment again")

    scfg = cfg["syllables"]["segmenters"][args.method]
    threads = int(cfg["runtime"]["threads"]["intra_op"])
    t0 = time.time()
    if args.method == "nuclei":
        run_one, extra = (lambda p: seg.praat_nuclei(p, scfg)), {"versions": seg.parselmouth_versions()}
    else:
        import torch

        torch.manual_seed(int(cfg["seed"]))
        model = seg.SylberSegmenter(scfg, cfg["paths"]["sylber_models"], threads)
        run_one = model
        extra = {"versions": model.versions, "checkpoint_load": model.load_report,
                 "settings_differing_from_package_defaults": model.differs_from_package_defaults}
    t_load = time.time() - t0
    trials = []
    for i, w in enumerate(wavs):
        trials.append({"trial": i, "file": w.name, **run_one(w)})
        if (i + 1) % 50 == 0:
            print(f"[segment] {cell} {args.method}: {i + 1}/{len(wavs)}", flush=True)
    rec = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "cell": cell, "method": args.method, "settings": scfg, **extra,
        "decoded": {k: header.get(k) for k in ("cell", "session", "model", "n_trials", "created", "code_version")},
        "units": "seconds of audio (subtract syllables.audio_lag_ms for the neural timeline)",
        "wall_time_s": {"load": t_load, "total": time.time() - t0},
        "runtime": runtime_record(cfg, env_set), "code_version": code_version(), "config_sha256": config_sha256(cfg),
        "trials": trials,
    }
    sy.save_json(out, rec)
    print(f"[segment] {cell} {args.method}: {len(trials)} trials -> {rel(out)} in {time.time() - t0:.0f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "run"])
    ap.add_argument("--config", default="configs/experiments/syllable_benchmark.yaml")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--method", choices=["nuclei", "sylber"], default=None)
    ap.add_argument("--threads", type=int, default=None, help="override runtime.threads (torch, BLAS)")
    ap.add_argument("--max-trials", type=int, default=None, help="use the <cell>__first<N> smoke cell")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.step == "run" and not (args.cell and args.method):
        ap.error("run needs --cell and --method")
    cfg, env_set = start(args.config, args.threads)
    return {"fetch": cmd_fetch, "run": cmd_run}[args.step](cfg, env_set, args)


if __name__ == "__main__":
    sys.exit(main())
