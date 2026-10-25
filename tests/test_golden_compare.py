"""perf/golden/compare.py: pair differences, the floor from cross-machine pairs, and the verdict."""

import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("golden_compare", Path(__file__).resolve().parents[1] / "perf/golden/compare.py")
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def _run(d: Path, machine, w, best, rank, total=10.0):
    d.mkdir(parents=True)
    np.savez_compressed(d / "golden.npz", **{"base/w/x": np.asarray(w, np.float32),
                                             "dec_ref/cont/g1": np.asarray([1.0, np.nan], np.float32),
                                             "disc/dec_ref/best50/g1": np.asarray(best, np.int16)})
    (d / "golden.json").write_text(json.dumps({"meta": {"machine": machine, "created": "t", "code_version": {"c": 1},
                                                        "env": {}, "total_s": total, "timings_s": {"pool": total}},
                                               "discrete": {"id_ref": {"1": {"rank": rank}}}}))
    return d


def test_pair_identical_is_zero(tmp_path):
    a = _run(tmp_path / "a", "pc", [1.0, 2.0], [0, 3, 3, 0, 5], 1)
    b = _run(tmp_path / "b", "pc", [1.0, 2.0], [0, 3, 3, 0, 5], 1, total=20.0)
    r = gc.pair(a, b)
    assert r["continuous"]["base/w"]["max_abs"] == 0 and r["continuous"]["dec_ref/cont"]["finite_mismatch"] == 0
    assert r["discrete"]["disc/dec_ref/best50"]["mismatches"] == 0 and r["discrete"]["id_ref/rank"]["mismatches"] == 0
    assert r["timings_s"]["pool"]["ratio"] == 0.5


def test_floor_and_verdict(tmp_path):
    p1 = _run(tmp_path / "p1", "pc", [1.0, 2.0], [0, 3, 3, 0, 5], 1)
    v1 = _run(tmp_path / "v1", "vast3", [1.0, 2.001], [0, 3, 0, 0, 5], 1)
    doc = gc.build_floor([p1, v1], tmp_path / "floor.json")
    f = doc["floor"]
    assert doc["floor_from"].startswith("cross-machine pairs")
    assert f["continuous"]["base/w"]["source"] == "cross-machine"
    assert abs(f["continuous"]["base/w"]["max_abs"] - 0.001) < 1e-5
    assert f["discrete"]["disc/dec_ref/best50"]["mismatches"] == 1
    assert f["discrete"]["disc/dec_ref/best50"]["sequences_differing"] == 0      # (3, 5) either way
    ok = _run(tmp_path / "ok", "pc", [1.0, 2.0005], [0, 3, 3, 0, 5], 1)
    assert gc.verdict(gc.pair(ok, p1), doc)["pass"]
    big = _run(tmp_path / "big", "pc", [1.0, 2.01], [0, 3, 3, 0, 5], 2)
    v = gc.verdict(gc.pair(big, p1), doc)
    assert not v["pass"] and any("base/w" in x for x in v["reasons"]) and any("id_ref/rank" in x for x in v["reasons"])


def test_floor_falls_back_to_repeat_runs_for_one_machine_outputs(tmp_path):
    p1 = _run(tmp_path / "p1", "pc", [1.0, 2.0], [0, 3], 1)
    p2 = _run(tmp_path / "p2", "pc", [1.0, 2.0], [0, 3], 1)
    v1 = _run(tmp_path / "v1", "vast3", [1.0, 2.001], [0, 3], 1)
    for d, wer in ((p1, 0.5), (p2, 0.25)):             # only the PC scores with Whisper
        z = dict(np.load(d / "golden.npz"))
        z["asr_ref/wer_orig"] = np.asarray([wer], np.float32)
        np.savez_compressed(d / "golden.npz", **z)
    f = gc.build_floor([p1, p2, v1], tmp_path / "floor.json")["floor"]["continuous"]
    assert f["base/w"]["source"] == "cross-machine" and abs(f["base/w"]["max_abs"] - 0.001) < 1e-5
    assert f["asr_ref/wer_orig"]["source"] == "same-machine" and f["asr_ref/wer_orig"]["max_abs"] == 0.25
