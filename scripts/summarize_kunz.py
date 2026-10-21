#!/usr/bin/env python
"""Across-fold summary of a Kunz experiment run (small CSV/JSON inputs; runs on the Mac or the host).

    python scripts/summarize_kunz.py --config configs/experiments/kunz_zeroshot.yaml

Reads results/<experiment.name>/fold_*/per_trial.csv (scripts/kunz_experiment.py) and writes
results/<experiment.name>/cross_fold_summary.json and .md (``wvx.kunz_summary``). The cue-null sections
(primary null, length-matched k, early window, sensitivity threshold) follow the run's own config
(``experiment.summary``, ``experiment.nulls.length_matched_k``, ``experiment.evaluation.early_window_s``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402

OUT_STEM = "cross_fold_summary"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)

    import pandas as pd

    from wvx import kunz_experiment as kx
    from wvx import kunz_summary
    from wvx.provenance import code_version, config_sha256
    from wvx.seeding import set_seed

    set_seed(int(cfg["seed"]))
    ex = cfg["experiment"]
    run_dir = resolve_path(ex["results_root"]) / ex["name"]
    files = sorted(run_dir.glob("fold_*/per_trial.csv"))
    if not files:
        raise SystemExit(f"no per_trial.csv under {run_dir}")
    syn = cfg["synthesis"]
    frame_s = float(syn["samples_per_frame"]) / float(syn["sample_rate"])  # one LPCNet frame per output bin
    df = pd.concat([pd.read_csv(f).assign(fold=f.parent.name) for f in files], ignore_index=True)
    settings = kunz_summary.summary_settings(cfg)
    summary = {"run": ex["name"], "config_file": args.config, "folds": [f.parent.name for f in files],
               "code_version": code_version(), "config_sha256": config_sha256(kx.portable(cfg)),
               "summary_settings": settings,
               "per_model": kunz_summary.summarize_run(df, frame_s, settings)}
    (run_dir / f"{OUT_STEM}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    md = kunz_summary.markdown(summary["per_model"], f"{ex['name']}: across folds")
    (run_dir / f"{OUT_STEM}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
