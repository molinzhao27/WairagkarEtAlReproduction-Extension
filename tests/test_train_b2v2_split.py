"""scripts/train_b2v2.py split_trials: each mode takes its own trials (a mis-nested else once made a non-whole-trial
adaptation fall into the from-scratch branch; caught before any run used that path)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from wvx import b2v2_data as bd

SPEC = importlib.util.spec_from_file_location("train_b2v2", Path(__file__).resolve().parents[1] / "scripts/train_b2v2.py")
train_b2v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_b2v2)


def _data(tmp_path):
    trials, a = [], 0
    for g in range(60):
        sess, blk = ("A", 1) if g < 20 else ("A", 2) if g < 40 else ("B", 9)
        trials.append({"gid": g, "session": sess, "block": blk, "index": g % 20, "day": 100 + g, "benchmark": None,
                       "start": a, "length": 10, "label": [2, 1] if g % 7 else None})
        a += 10
    np.zeros((a, 4), np.float16).tofile(tmp_path / "neural.f16")
    np.zeros((a, 2), np.float32).tofile(tmp_path / "target.f32")
    (tmp_path / "meta.json").write_text(json.dumps({"n_bins": a, "n_features": 4, "n_targets": 2, "trials": trials}))
    return bd.B2V2Data(tmp_path)


def test_split_modes(tmp_path):
    data = _data(tmp_path)
    t = {"seed": 0, "n_validation": 5}
    full = {"b2v2": {"training": t, "adapt": {"session": "A", "train_blocks": [1], "full_trial": True, "n_validation": 3,
                                              "extra": [{"session": "B", "blocks": [9]}]}}}
    sp = train_b2v2.split_trials(data, full)
    assert set(sp["val"]) <= set(range(20)) and len(sp["val"]) == 3
    assert {g for g in sp["train"] if g < 40} <= set(range(20)) and any(g >= 40 for g in sp["train"])
    assert all(data.by_gid[g]["label"] is not None for g in sp["train"] + sp["val"])
    windowed = {"b2v2": {"training": t, "adapt": {"session": "A", "train_blocks": [2], "n_validation": 2}}}
    sp = train_b2v2.split_trials(data, windowed)
    assert set(sp["train"] + sp["val"]) == set(range(20, 40)) and sp["test"] == []
    scratch = {"b2v2": {"training": t, "split": {"max_day": 129}}}
    sp = train_b2v2.split_trials(data, scratch)
    assert set(sp["train"] + sp["val"]) == set(range(30)) and len(sp["val"]) == 5
