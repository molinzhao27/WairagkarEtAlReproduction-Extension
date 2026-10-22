"""Manual syllable alignment (wvx.manual_alignment, scripts/manual_alignment.py, configs/manual_alignment/): a synthetic
package (no data) built by the real packaging code, its blindness, the answer rules (host and standalone copies agree),
ingestion, aggregation, agreement, the standalone tools, scoring (manual = their timing gives the oracle) and the
one-to-one targets."""

import ast
import copy
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wvx import b2v
from wvx import manual_alignment as ma
from wvx import syllables as sy
from wvx.config import REPO_ROOT

SILENCE36 = np.r_[-12.48528, np.zeros(17), 3.1, -0.5, np.zeros(16)].astype(np.float32)
WORDS = ["i", "hope", "it", "is", "thirsty"]
SYL_WORDS = [0, 1, 2, 3, 4, 4]
PHONES = [["AY1"], ["HH", "OW1", "P"], ["IH1", "T"], ["IH1", "Z"], ["TH", "ER1"], ["S", "T", "IY0"]]
SALT = "00112233445566778899aabbccddeeff"
SMALL = {"manual_alignment": {"workers": {"n": 4}, "batching": {"redundancy_k": 2, "items_per_batch": 3},
                              "practice": {"items_per_round": 1, "max_rounds": 2}}}
needs_b2v = pytest.mark.skipif(not b2v.B2V_ROOT.is_dir(), reason="run scripts/fetch_third_party.py")


@pytest.fixture(scope="module")
def cfg():
    return ma.load_config(overrides=SMALL)


def labels():
    return ma.syllable_labels(WORDS, SYL_WORDS, PHONES)


def truth_frames(T, seed, n=6):
    rng = np.random.default_rng(seed)
    starts = np.linspace(80, T - 80, n + 1)[:-1].astype(int) + rng.integers(0, 10, n)
    return np.stack([starts, starts + 30 + rng.integers(0, 15, n)], axis=1).astype(np.int64)


def synth_source(T, truth, seed):
    """Decoder output with a c0 hump per true syllable, matching audio and neural activity."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    env = np.full(T, -20.0)
    for a, b in truth:
        env = np.maximum(env, -20 + 40 * np.exp(-0.5 * ((t - (a + b) / 2) / ((b - a) / 2.5)) ** 2))
    raw = np.tile(SILENCE36[:20], (T, 1)).astype(np.float64)
    raw[:, 0] = env * np.sqrt(18.0) / 10.0 - 4.0
    raw[:, 1:18] = np.cumsum(rng.normal(scale=0.1, size=(T, 17)), axis=0)
    raw[:, 18], raw[:, 19] = 0.5, 0.3
    ts = np.arange(T * 160) / 16000
    amp = np.interp(np.arange(T * 160) / 160, t, 10 ** (env / 20) / 100)
    wav = 0.3 * amp * sum(np.sin(2 * np.pi * 120 * k * ts) / k for k in range(1, 6)) + 1e-3 * rng.normal(size=ts.size)
    neural = rng.normal(size=(T, 512)) + np.interp(t + 15, t, (env + 20) / 40)[:, None]
    return {"wav": wav, "pred_raw": raw.astype(np.float32), "neural": neural.astype(np.float32),
            "text": "I hope it is thirsty.", "labels": labels(), "truth": truth}


LENGTHS = {("A", 0): 1300, ("B", 0): 1300, ("A", 1): 900, ("A", 2): 950, ("A", 3): 900, ("K", 0): 800, ("K", 1): 700,
           ("K", 2): 2200}


def _chosen():
    return {"real": [{"set": "calibration", "group": "g", "cell": c, "session": "S", "trial": 0} for c in ("A", "B")]
            + [{"set": "production", "group": "production", "cell": "K", "session": "K", "trial": i} for i in range(3)],
            "practice": [{"set": "practice", "group": "p", "cell": "A", "session": "S", "trial": i} for i in (1, 2)],
            "examples": [{"set": "example", "group": "examples", "cell": "A", "session": "S", "trial": 3}]}


def _truth(it):
    return truth_frames(LENGTHS[(it["cell"] if it["cell"] != "B" else "A", it["trial"])], it["trial"])


def _source(it):
    T = LENGTHS[(it["cell"], it["trial"])]
    src = synth_source(T, _truth(it), 7 + it["trial"] + (100 if it["cell"] == "K" else 0))
    if it["set"] not in ("practice", "example"):
        del src["truth"]
    return src


@pytest.fixture(scope="module")
def built(cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("pkg") / "testpkg"
    root.mkdir()
    chosen = _chosen()
    man = ma.build_package(root, cfg, "testpkg", chosen, _source, SALT)
    truth = {r["id"]: _truth(r) for r in chosen["real"] if r["set"] == "calibration"}
    return SimpleNamespace(root=root, chosen=chosen, man=man, truth=truth)


def run_tool(root, *args):
    """A packaged tool as the annotators run it: isolated mode, from the package folder, no repo on the path."""
    return subprocess.run([sys.executable, "-I", *args], cwd=root, capture_output=True, text=True, timeout=120)


def tool_module(root, name):
    sys.path.insert(0, str(Path(root) / "tools"))
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location(name, Path(root) / "tools" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = old


def copy_pkg(built, tmp_path):
    dst = tmp_path / "testpkg"
    shutil.copytree(built.root, dst)
    return dst


# ------------------------------------------------------------------------------------------- config


def test_config_preregistration(cfg):
    s = ma.settings(cfg)
    a = s["adoption"]
    assert a["primary_cells"] == ["day195_m172", "day195_m165"] and a["min_improvement"] == 0.02
    assert a["metric"] == "target_r_mean" and s["scoring"]["baseline"] == {"method": "current", "arm": "B", "tts_side": "forced"}
    assert s["calibration"]["groups"]["day195"]["cells"] == ["day195_m172", "day195_m165"]
    assert cfg["syllables"]["segment_mapping"]["equal_counts"] == "one_to_one"      # manual spans pair one to one
    assert s["display"]["show_auto_boundaries"] is False
    out = cfg["paths"]["manual_targets"]
    assert out.name == "kunz_targets_manual" and all(out != REPO_ROOT / p for p in s["targets"]["protected"])
    assert {"data/processed/kunz_targets", "data/processed/kunz_targets_v2"} == set(s["targets"]["protected"])
    for c in s["calibration"]["groups"]["day195"]["cells"] + s["production"]["cells"]:
        assert c in cfg["benchmark"]["cells"]
    assert cfg["runtime"]["threads"]["intra_op"] <= 4 and cfg["alignment"]["n_workers"] <= 4


# ---------------------------------------------------------------------------------------- selection


def _info():
    def cell(session, n, texts, malformed=()):
        return {"session": session, "n": n, "text": texts, "wellformed": [i not in malformed for i in range(n)],
                "n_truth": [4 + i % 3 for i in range(n)], "n_tts": [4 + (i % 2) for i in range(n)],
                "duration_s": [5.0 + (i % 12) for i in range(n)]}

    t195 = [f"s{i}" if i % 3 else f"k{i % 20}" for i in range(80)]           # some cues shared with the Kunz set
    t193 = [f"u{i}" for i in range(50)]
    return {"day195_m172": cell("S195", 80, t195, malformed={5, 9}), "day195_m165": cell("S195", 80, t195, malformed={5, 9}),
            "day193_m165": cell("S193", 50, t193, malformed={1}),
            "kunz_attempted": {"session": "K", "n": 20, "text": [f"k{i}" for i in range(20)], "wellformed": [False] * 20,
                               "n_truth": [None] * 20, "n_tts": [5] * 20, "duration_s": [7.0] * 20}}


def test_choose_items_sets_are_disjoint(cfg):
    c = ma.load_config(overrides={"manual_alignment": {
        "calibration": {"groups": {"day195": {"n_trials": 12}, "day193": {"n_trials": 6}}},
        "examples": {"n": 2}, "practice": {"groups": {"day195": {"n_items": 8}, "day193": {"n_items": 3}}}}})
    info = _info()
    got = ma.choose_items(info, c, np.random.default_rng(42))
    assert got == ma.choose_items(info, c, np.random.default_rng(42))            # seeded
    cal = [r for r in got["real"] if r["set"] == "calibration"]
    by_cell = {cell: sorted(r["trial"] for r in cal if r["cell"] == cell) for cell in ("day195_m172", "day195_m165", "day193_m165")}
    assert by_cell["day195_m172"] == by_cell["day195_m165"] and len(by_cell["day195_m172"]) == 12   # same trials, both cells
    assert len(by_cell["day193_m165"]) == 6 and not {5, 9} & set(by_cell["day195_m172"]) and 1 not in by_cell["day193_m165"]
    assert sorted(r["trial"] for r in got["real"] if r["set"] == "production") == list(range(20))
    keys = {name: {(r["session"], r["trial"]) for r in rows} for name, rows in
            (("calibration", cal), ("production", [r for r in got["real"] if r["set"] == "production"]),
             ("practice", got["practice"]), ("examples", got["examples"]))}
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not keys[names[i]] & keys[names[j]], (names[i], names[j])
    real_texts = {info[r["cell"]]["text"][r["trial"]] for r in got["real"]}
    for r in got["practice"] + got["examples"]:
        assert info[r["cell"]]["text"][r["trial"]] not in real_texts and info[r["cell"]]["wellformed"][r["trial"]]
    for r in got["examples"]:
        i = info[r["cell"]]
        assert i["n_truth"][r["trial"]] == i["n_tts"][r["trial"]] and i["duration_s"][r["trial"]] <= 12.0
    assert {r["cell"] for r in got["practice"] if r["group"] == "day195"} <= {"day195_m172", "day195_m165"}
    assert len(got["practice"]) == 11


def test_select_refuses_too_few():
    with pytest.raises(ValueError):
        ma.select([1, 2], 3, np.random.default_rng(0), "x")


def test_assign_work_redundancy_and_siblings():
    ids = [f"{i:04x}" for i in range(40)]
    sib = {i: f"t{n // 2}" for n, i in enumerate(ids[:20])}                       # 10 trials shown twice
    workers, batches = ma.assign_work(ids, sib, 6, 3, 4, np.random.default_rng(1), "w{:02d}")
    seen: dict = {}
    for b, v in batches.items():
        assert len(v["items"]) <= 4 and re.fullmatch(r"batch_\d{3}", b)
        for i in v["items"]:
            seen.setdefault(i, []).append(v["worker_id"])
    assert all(len(w) == 3 and len(set(w)) == 3 for w in seen.values()) and set(seen) == set(ids)
    for w in workers:
        mine = [i for b in workers[w] for i in batches[b]["items"]]
        groups = [sib.get(i, i) for i in mine]
        assert len(groups) == len(set(groups))                                   # no worker sees a trial twice
    pb = ma.assign_practice([f"p{i}" for i in range(10)], list(workers), 2, 3, np.random.default_rng(2))
    for v in pb.values():
        flat = [i for r in v["rounds"] for i in r]
        assert len(v["rounds"]) == 3 and len(flat) == len(set(flat)) == 6
    with pytest.raises(ValueError):
        ma.assign_work(ids, {}, 2, 3, 4, np.random.default_rng(1), "w{:02d}")


def test_ids_are_salted_and_namespaced():
    a = ma.make_ids(["A|0", "B|0"], SALT, "real", 10)
    assert a != ma.make_ids(["A|0", "B|0"], "f" * 32, "real", 10)
    p = ma.make_ids(["A|0"], SALT, "practice", 10, "p")
    assert all(re.fullmatch(r"[0-9a-f]{10}", v) for v in a.values()) and p["A|0"].startswith("p") and p["A|0"][1:] != a["A|0"]


# ----------------------------------------------------------------------------------------- blindness


def test_package_passes_the_scan_and_holds_no_source(built, cfg):
    assert ma.scan_package(built.root, cfg, truth=built.truth, secrets=[SALT]) == []
    texts = "\n".join(p.read_text() for p in built.root.rglob("*") if p.suffix in (".json", ".md", ".py", ".txt"))
    for word in ("calibration", "production", "day195", "day193", "kunz", "session", SALT, "\"set\"", "\"cell\""):
        assert word not in texts.lower(), word
    real = [r for r in built.chosen["real"]]
    shapes = set()
    for r in real:
        d = json.loads((built.root / "items" / f"{r['id']}.json").read_text())
        assert d["kind"] == "item" and set(d) == ma.ITEM_FILE_KEYS
        shapes.add((tuple(d), tuple(d["syllables"][0]), tuple(d["files"])))
        with np.load(built.root / "items" / f"{r['id']}.npz") as z:
            assert set(z.files) == set(ma.ARRAY_NAMES) and z["mel_db"].dtype == np.float16
    assert len(shapes) == 1                                                      # calibration items look like production
    m = built.man
    assert set(m) == {"schema", "package", "counts", "answer_paths", "workers", "batches", "practice_batches", "items",
                      "practice_items", "examples"}
    assert sorted(m["items"]) == m["items"] and set(m["items"]) == {r["id"] for r in real}
    sib = [r["id"] for r in real if r["set"] == "calibration"]
    for w, v in m["workers"].items():                                             # one trial, two decoders: two workers
        mine = {i for b in v["batches"] for i in m["batches"][b]["items"]}
        assert len(mine & set(sib)) <= 1
    assert not list(built.root.rglob("__pycache__")) and "{{" not in texts


def _rehash(root):
    ma.write_hash_list(root)


@pytest.mark.parametrize("leak", ["key", "truth_numbers", "array", "file", "png_text", "mask", "hash", "salt", "text"])
def test_scan_catches_leaks(built, cfg, tmp_path, leak):
    root = copy_pkg(built, tmp_path)
    rid = next(r["id"] for r in built.chosen["real"] if r["set"] == "calibration")
    item = root / "items" / f"{rid}.json"
    if leak == "key":
        d = json.loads(item.read_text())
        d["session"] = "S"
        item.write_text(json.dumps(d))
    elif leak == "truth_numbers":
        st = json.loads((root / "settings.json").read_text())
        st["values"]["bands_hz"] = (built.truth[rid] / 100.0).tolist()           # benign key, true times
        (root / "settings.json").write_text(json.dumps(st))
    elif leak == "array":
        with np.load(root / "items" / f"{rid}.npz") as z:
            arr = {k: z[k] for k in z.files}
        arr["spans"] = built.truth[rid].astype(np.float32)
        np.savez_compressed(root / "items" / f"{rid}.npz", **arr)
    elif leak == "file":
        (root / "testpkg_key.json").write_text("{}")
    elif leak == "png_text":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig.savefig(root / "items" / f"{rid}.png", metadata={"Comment": "x"})
        plt.close(fig)
    elif leak == "mask":
        with np.load(root / "items" / f"{rid}.npz") as z:
            arr = {k: z[k] for k in z.files}
        mask = np.zeros(arr["t_s"].size, np.float16)
        for a, b in built.truth[rid]:
            mask[a:b] = 1
        arr["neural"] = mask
        np.savez_compressed(root / "items" / f"{rid}.npz", **arr)
    elif leak == "salt":
        (root / "README.md").write_text((root / "README.md").read_text() + SALT)
    elif leak == "text":
        (root / "instructions.md").write_text((root / "instructions.md").read_text() + "\nsome calibration items\n")
    if leak != "hash":
        _rehash(root)
    else:
        (root / "README.md").write_text("changed")
    probs = ma.scan_package(root, cfg, truth=built.truth, secrets=[SALT])
    want = {"key": "forbidden key", "truth_numbers": "true boundaries", "array": "arrays", "file": "unexpected file",
            "png_text": "PNG metadata", "mask": "speech mask", "hash": "files.sha256", "salt": "secret",
            "text": "forbidden text"}[leak]
    assert any(want in p for p in probs), (leak, probs)


def test_tar_members_outside_the_package_are_refused():
    assert ma.tar_member_problem("pkg/items/a.png", "file", "pkg") is None
    assert ma.tar_member_problem("pkg", "dir", "pkg") is None
    for name, kind in (("other/x", "file"), ("pkg/../pkg_key.json", "file"), ("/etc/x", "file"), ("pkg/x", "link"),
                       ("pkg/pkg_key.json", "file"), ("pkg/pkg_answers/a.json", "file")):
        assert ma.tar_member_problem(name, kind, "pkg") is not None, name


# ------------------------------------------------------------------------------------- answer rules


def _answer(built, iid, shift=0.0, batch=None, worker=None):
    d = json.loads(next(built.root.glob(f"*/{iid}.json") if (built.root / "items" / f"{iid}.json").exists()
                        else built.root.glob(f"**/{iid}.json")).read_text())
    t = copy.deepcopy(d["response_template"])
    tr = np.linspace(1.0, d["duration_s"] - 1.5, d["n_syllables"] + 1)
    for k, s in enumerate(t["syllables"]):
        s["start_s"], s["end_s"] = round(tr[k] + shift, 2), round(tr[k] + 0.2 + shift, 2)
    return t


def _file(built, batch, worker, entries):
    return {"schema": ma.settings(ma.load_config(overrides=SMALL))["schema"]["response"], "batch": batch,
            "worker_id": worker, "items": entries}


def _cases(built):
    m = built.man
    b, v = next(iter(m["batches"].items()))
    w, iid = v["worker_id"], v["items"][0]
    other = next(i for i in m["items"] if i not in v["items"])
    good = _answer(built, iid)

    def mod(f):
        e = copy.deepcopy(good)
        f(e)
        return _file(built, b, w, [e])

    def syl(k, **kw):
        return lambda e: e["syllables"][k].update(kw)

    dur = json.loads((built.root / "items" / f"{iid}.json").read_text())["duration_s"]
    return {
        "valid": (_file(built, b, w, [good]), set()),
        "valid_confidence": (mod(lambda e: (e.update(confidence="low", notes="n", unassigned_bursts=[[0.1, 0.2]]),
                                            e["syllables"][0].update(confidence="medium"))), set()),
        "extra_batch": (_file(built, "extra", w, [_answer(built, other)]), set()),
        "touching": (mod(lambda e: e["syllables"][1].update(start_s=e["syllables"][0]["end_s"])), set()),
        "int_seconds": (mod(lambda e: e["syllables"][-1].update(end_s=int(dur))), set()),
        "schema": ({**_file(built, b, w, [good]), "schema": "x"}, {"E_SCHEMA"}),
        "batch": (_file(built, "batch_999", w, [good]), {"E_BATCH"}),
        "worker": (_file(built, b, "w 1", [good]), {"E_WORKER"}),
        "top_key": ({**_file(built, b, w, [good]), "who": 1}, {"E_TOP"}),
        "not_object": ([good], {"E_TOP"}),
        "empty": (_file(built, b, w, []), {"E_ITEMS"}),
        "unknown_item": (mod(lambda e: e.update(item_id="ffffffffff")), {"E_ITEM_ID"}),
        "not_in_batch": (_file(built, b, w, [_answer(built, other)]), {"E_NOT_IN_BATCH"}),
        "duplicate": (_file(built, b, w, [good, copy.deepcopy(good)]), {"E_DUP_ITEM"}),
        "count": (mod(lambda e: e["syllables"].pop()), {"E_COUNT"}),
        "label": (mod(syl(0, label="I")), {"E_LABEL"}),
        "index": (mod(syl(1, index=0)), {"E_INDEX"}),
        "string_number": (mod(syl(0, start_s="1.0")), {"E_NUMBER"}),
        "null": (mod(syl(0, start_s=None)), {"E_NUMBER"}),
        "bool": (mod(syl(0, start_s=True)), {"E_NUMBER"}),
        "nan": (mod(syl(0, start_s=float("nan"))), {"E_FINITE"}),
        "inf": (mod(syl(0, end_s=float("inf"))), {"E_FINITE"}),
        "reversed": (mod(lambda e: e["syllables"][0].update(start_s=e["syllables"][0]["end_s"] + 0.05)), {"E_SPAN"}),
        "short": (mod(lambda e: e["syllables"][0].update(end_s=e["syllables"][0]["start_s"] + 0.005)), {"E_SHORT"}),
        "negative": (mod(syl(0, start_s=-0.1)), {"E_BOUNDS"}),
        "after_end": (mod(syl(-1, end_s=dur + 0.5)), {"E_BOUNDS"}),
        "overlap": (mod(lambda e: e["syllables"][1].update(start_s=e["syllables"][0]["end_s"] - 0.05)), {"E_ORDER"}),
        "confidence": (mod(syl(0, confidence="sure")), {"E_CONFIDENCE"}),
        "syl_key": (mod(syl(0, start=1.0)), {"E_SYL_KEYS"}),
        "missing_end": (mod(lambda e: e["syllables"][0].pop("end_s")), {"E_SYL_KEYS"}),
        "item_key": (mod(lambda e: e.update(spans=[])), {"E_ITEM_KEYS"}),
        "unassigned": (mod(lambda e: e.update(unassigned_bursts=[[2.0, 1.0]])), {"E_UNASSIGNED"}),
        "syllables_type": (mod(lambda e: e.update(syllables={})), {"E_SYLLABLES"}),
    }


def _codes(res):
    return {c for c, _ in res["file_errors"]} | {c for r in res["items"] for c, _ in r["errors"]}


def test_answer_rules_host_and_tool_agree(built, cfg):
    tool = tool_module(built.root, "inspect_item")
    st = json.loads((built.root / "settings.json").read_text())
    items_tool = tool.all_items(built.root)
    items_host = {k: v for k, v in items_tool.items()}
    rules_h, rules_t = ma.response_rules(cfg), tool.response_rules(st)
    bs = ma.batch_sets(built.man)
    assert bs == tool.batch_sets(built.man)
    for name, (obj, want) in _cases(built).items():
        h = ma.validate_response(obj, items_host, bs, rules_h, built.man["items"])
        t = tool.validate_response(obj, items_tool, bs, rules_t, built.man["items"])
        assert _codes(h) == _codes(t) == want or (want and want <= _codes(h) == _codes(t)), (name, _codes(h), _codes(t))
        acc_h = [r["item_id"] for r in h["items"] if not r["errors"]] if not h["file_errors"] else []
        assert acc_h == [r["item_id"] for r in tool.accepted(t)], name
        assert bool(want) == (not acc_h or bool(h["file_errors"])), name
    # where an answer file lives
    b, v = next(iter(built.man["batches"].items()))
    obj = _file(built, b, v["worker_id"], [])
    for wdir, fname, want in ((v["worker_id"], f"{b}.json", set()), ("w99", f"{b}.json", {"E_WORKER_DIR"}),
                              (v["worker_id"], "batch_000.json", {"E_FILENAME"}), (v["worker_id"], "mine.json", {"E_FILENAME"})):
        host = {c for c, _ in ma.path_errors(wdir, fname, obj, rules_h, built.man["practice_batches"])}
        tl = {c for c, _ in tool.path_errors(built.root / "answers" / wdir / fname, built.root, obj, rules_t,
                                             built.man["practice_batches"])}
        assert host == tl == want, (wdir, fname)


def test_check_cli(built, tmp_path):
    cases = _cases(built)
    for name in ("valid", "count", "overlap", "nan"):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(cases[name][0]))
        r = run_tool(built.root, "tools/inspect_item.py", "check", str(p))
        assert (r.returncode == 0) == (name == "valid"), (name, r.stdout)
        assert r.stdout.startswith("VALID" if name == "valid" else "INVALID")
        for code in cases[name][1]:
            assert f"[{code}]" in r.stdout
    single = tmp_path / "one.json"
    single.write_text(json.dumps(cases["valid"][0]["items"][0]))
    assert run_tool(built.root, "tools/inspect_item.py", "check", str(single)).returncode == 0
    (tmp_path / "bad.json").write_text("{not json")
    assert run_tool(built.root, "tools/inspect_item.py", "check", str(tmp_path / "bad.json")).returncode == 1


# ---------------------------------------------------------------------------------- ingest, aggregate


def _worker_files(built, shifts):
    """{worker: answer file} for every batch, each worker's starts shifted by shifts[worker]."""
    out = {}
    for b, v in built.man["batches"].items():
        w = v["worker_id"]
        out[(w, b)] = _file(built, b, w, [_answer(built, i, shift=shifts.get(w, 0.0)) for i in v["items"]])
    return out


def test_ingest_aggregate_and_agreement(built, cfg, tmp_path):
    shifts = {w: 0.01 * n for n, w in enumerate(sorted(built.man["workers"]))}
    files = [(f"answers/{w}/{b}.json", w, obj, None) for (w, b), obj in _worker_files(built, shifts).items()]
    pw = next(iter(built.man["practice_batches"]))
    files.append(("answers/w01/practice.json", "w01", {"batch": pw}, None))
    files.append(("answers/w01/broken.json", "w01", None, "not valid JSON"))
    items = {k: ma.rule_meta(json.loads((built.root / "items" / f"{k}.json").read_text())) for k in built.man["items"]}
    res = ma.ingest(files, items, built.man, ma.response_rules(cfg))
    k = built.man["counts"]["redundancy_k"]
    assert len(res["accepted"]) == k * len(built.man["items"]) and len(res["skipped"]) == 1
    assert [r["errors"][0][0] for r in res["rejected"]] == ["E_JSON"]
    agg = ma.aggregate(res["accepted"], items, cfg)
    for iid, a in agg.items():
        spans = np.asarray([x["spans"] for x in res["accepted"] if x["item_id"] == iid])
        np.testing.assert_allclose(a["spans"], np.median(spans, axis=0), atol=1e-6)
        assert a["n_answers"] == k and ma.validate_item_answer(
            {"item_id": iid, "syllables": [{"index": j, "label": items[iid]["labels"][j], "start_s": s, "end_s": e}
                                           for j, (s, e) in enumerate(a["spans"])]}, items[iid], ma.response_rules(cfg)) == []
    agr = ma.agreement(res["accepted"], cfg)
    for iid, r in agr["per_item"].items():
        ws = r["workers"]
        d = abs(shifts[ws[0]] - shifts[ws[1]]) * 1000
        assert r["median_abs_start_diff_ms"] == pytest.approx(d) and r["start_within_tol"] == (1.0 if d <= 100 else 0.0)
    # one worker answering an item twice: every copy rejected
    (w, b), obj = next(iter(_worker_files(built, {}).items()))
    dup = files + [(f"answers/{w}/extra.json", w, _file(built, "extra", w, [obj["items"][0]]), None)]
    res2 = ma.ingest(dup, items, built.man, ma.response_rules(cfg))
    assert sum(c == "E_DUP_WORKER" for r in res2["rejected"] for c, _ in r["errors"]) == 2
    assert len(res2["accepted"]) == len(res["accepted"]) - 1


def test_aggregate_median_of_valid_answers_is_valid(cfg):
    rng = np.random.default_rng(0)
    meta = {"x": {"n_syllables": 5, "labels": list("abcde"), "duration_s": 10.0, "kind": "item"}}
    acc = []
    for w in range(4):
        cuts = np.sort(rng.uniform(0.5, 9.5, 10))
        acc.append({"item_id": "x", "worker_id": f"w{w}", "spans": cuts.reshape(5, 2).round(2).tolist()})
    agg = ma.aggregate(acc, meta, cfg)["x"]
    np.testing.assert_allclose(agg["spans"], np.median([a["spans"] for a in acc], axis=0))
    assert agg["n_answers"] == 4


# ----------------------------------------------------------------------------------------------- tools


def test_tools_are_standalone(built):
    allowed = {"argparse", "json", "math", "re", "sys", "textwrap", "pathlib", "hashlib", "numpy", "matplotlib",
               "__future__", "inspect_item"}
    for f in ma.TOOL_FILES:
        tree = ast.parse((built.root / "tools" / f).read_text())
        mods = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert mods <= allowed, (f, mods - allowed)
        r = run_tool(built.root, f"tools/{f}", "--help")
        assert r.returncode == 0 and "usage" in r.stdout


def test_requirements_are_pinned_for_every_python():
    packaging = pytest.importorskip("packaging.requirements")
    lines = [ln for ln in (REPO_ROOT / "configs/experiments/manual_alignment.yaml").read_text().splitlines()]
    reqs = [packaging.Requirement(r) for r in ma.load_config()["manual_alignment"]["local_setup"]["requirements"]]
    assert lines and all(len(r.specifier) == 1 and next(iter(r.specifier)).operator == "==" for r in reqs)
    for minor in range(9, 15):
        env = {"python_version": f"3.{minor}"}
        active = sorted(r.name for r in reqs if r.marker is None or r.marker.evaluate(env))
        assert active == ["matplotlib", "numpy"], minor


def test_render_overlay_values_peaks(built, tmp_path):
    rid = next(r["id"] for r in built.chosen["real"] if r["set"] == "production" and r["trial"] == 2)   # 22 s: 3 rows
    tool = tool_module(built.root, "inspect_item")
    meta = json.loads((built.root / "items" / f"{rid}.json").read_text())
    a = tool.render(rid, tmp_path / "a.png", built.root)
    tool.render(rid, tmp_path / "b.png", built.root)
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes() and len(a["rows"]) == 3
    assert a["px_per_second"] == 100 and a["rows"][-1][1] == 22.0
    z = tool.render(rid, tmp_path / "z.png", built.root, start=2.0, end=3.0, width_px=1000)
    assert z["px_per_second"] == 1000 and tool.tick_steps(z["px_per_second"], json.loads(
        (built.root / "settings.json").read_text())["display"]) == (0.01, 0.1)
    draft = _answer(built, rid)
    (tmp_path / "draft.json").write_text(json.dumps(draft))
    o = tool.render(rid, tmp_path / "o.png", built.root, answer=[[s["start_s"], s["end_s"]] for s in draft["syllables"]])
    assert o["n_spans_drawn"] == meta["n_syllables"] == 6
    r = run_tool(built.root, "tools/inspect_item.py", "overlay", rid, str(tmp_path / "draft.json"), "--out", str(tmp_path / "c.png"))
    assert r.returncode == 0 and f"drew {meta['n_syllables']} of {meta['n_syllables']} spans" in r.stdout
    r = run_tool(built.root, "tools/inspect_item.py", "render", rid, "--start", "1", "--end", "2")
    assert r.returncode == 0 and (built.root / "answers" / "_views" / f"{rid}_1-2.png").is_file()
    shutil.rmtree(built.root / "answers" / "_views")
    r = run_tool(built.root, "tools/inspect_item.py", "values", rid, "--start", "1.0", "--end", "1.5", "--step", "0.05", "--bands")
    rows = [ln for ln in r.stdout.splitlines() if re.match(r"\s+\d", ln)]
    assert r.returncode == 0 and len(rows) == 11 and len(rows[0].split()) == 6
    truth = truth_frames(2200, 2)
    r = run_tool(built.root, "tools/inspect_item.py", "peaks", rid)
    peaks = [float(m) for m in re.findall(r"PEAK\s+t=\s*([\d.]+)", r.stdout)]
    assert len(peaks) == len(truth) and r.stdout.count("DIP") == len(truth) - 1
    np.testing.assert_allclose(peaks, truth.mean(axis=1) / 100 + 0.005, atol=0.03)
    assert run_tool(built.root, "tools/inspect_item.py", "render", "nosuchitem").returncode == 2


# --------------------------------------------------------------------------------------- practice


def test_score_practice(built, tmp_path):
    pid = built.man["practice_items"][0]
    ref = json.loads((built.root / "practice" / "truth" / f"{pid}.json").read_text())
    meta = json.loads((built.root / "practice" / "items" / f"{pid}.json").read_text())
    perfect = copy.deepcopy(meta["response_template"])
    for s, r in zip(perfect["syllables"], ref["syllables"]):
        s["start_s"], s["end_s"] = r["start_s"], r["end_s"]
    late = copy.deepcopy(perfect)
    for s in late["syllables"]:
        s["start_s"], s["end_s"] = s["start_s"] + 0.2, s["end_s"] + 0.2
    pb = next(b for b, v in built.man["practice_batches"].items() if pid in v["rounds"][0] + v["rounds"][1])
    w = built.man["practice_batches"][pb]["worker_id"]
    f_ok = tmp_path / "round1.json"
    f_ok.write_text(json.dumps(_file(built, pb, w, [perfect])))
    r = run_tool(built.root, "tools/score_practice.py", str(f_ok), "--overlay", str(tmp_path / "ov"), "--json", str(tmp_path / "s.json"))
    assert r.returncode == 0, r.stdout
    s = json.loads((tmp_path / "s.json").read_text())["summary"]
    assert s["starts_within_tol"] == 1.0 and s["median_abs_start_error_ms"] == 0 and s["count_agreement"] == 1.0
    assert (tmp_path / "ov" / f"{pid}_score.png").is_file() and "<-- start off" not in r.stdout
    f_late = tmp_path / "late.json"
    f_late.write_text(json.dumps(late))
    r = run_tool(built.root, "tools/score_practice.py", str(f_late))
    assert r.returncode == 0 and "within 100 ms 0/6" in r.stdout and "+200ms" in r.stdout
    # refuses real items, and truth that does not belong to its item
    real = _answer(built, built.man["items"][0])
    (tmp_path / "real.json").write_text(json.dumps(real))
    r = run_tool(built.root, "tools/score_practice.py", str(tmp_path / "real.json"))
    assert r.returncode == 3 and r.stdout.startswith("REFUSED")
    ex = json.loads((built.root / "examples" / "example_1_answer.json").read_text())
    (tmp_path / "ex.json").write_text(json.dumps(ex))
    assert run_tool(built.root, "tools/score_practice.py", str(tmp_path / "ex.json")).returncode == 3
    root = copy_pkg(built, tmp_path)
    p = root / "practice" / "items" / f"{pid}.json"
    p.write_text(p.read_text().replace('"kind": "practice"', '"kind": "practice" '))
    r = run_tool(root, "tools/score_practice.py", str(f_ok))
    assert r.returncode == 3 and "does not belong" in r.stdout
    bad = copy.deepcopy(perfect)
    bad["syllables"].pop()
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert run_tool(built.root, "tools/score_practice.py", str(tmp_path / "bad.json")).returncode == 1


def test_agreement_tool_matches_host(built, cfg, tmp_path):
    root = copy_pkg(built, tmp_path)
    workers = sorted(built.man["workers"])
    shifts = {w: (0.5 if w == workers[0] else 0.01 * n) for n, w in enumerate(workers)}
    for (w, b), obj in _worker_files(built, shifts).items():
        (root / "answers" / w).mkdir(parents=True, exist_ok=True)
        (root / "answers" / w / f"{b}.json").write_text(json.dumps(obj))
    (root / "answers" / workers[1] / "practice").mkdir()
    (root / "answers" / workers[1] / "practice" / "round1.json").write_text("{}")
    (root / "answers" / workers[1] / "technique.md").write_text("notes")
    r = run_tool(root, "tools/agreement.py", "--json", str(tmp_path / "agr.json"))
    assert r.returncode == 0, r.stdout
    got = json.loads((tmp_path / "agr.json").read_text())
    files = [(p, p.parent.name, json.loads(p.read_text()), None) for p in sorted((root / "answers").glob("*/*.json"))]
    items = {k: ma.rule_meta(json.loads((root / "items" / f"{k}.json").read_text())) for k in built.man["items"]}
    host = ma.agreement(ma.ingest(files, items, built.man, ma.response_rules(cfg))["accepted"], cfg)
    for iid, h in host["per_item"].items():
        t = got["items"][iid]
        for k in ("n_answers", "median_abs_start_diff_ms", "start_within_tol", "median_abs_end_diff_ms", "workers", "flagged"):
            assert t[k] == pytest.approx(h[k]) if isinstance(h[k], float) else t[k] == h[k], (iid, k)
    flagged = {i for i, a in host["per_item"].items() if workers[0] in a["workers"]}
    assert set(got["flagged"]) == set(host["flagged"]) == flagged and got["missing"] == []
    assert got["summary"]["median_abs_start_diff_ms"] == pytest.approx(host["pooled"]["median_abs_start_diff_ms"])


# ---------------------------------------------------------------------------- prompt and instructions


def _help(root, tool, sub=None):
    r = run_tool(root, f"tools/{tool}", *([sub] if sub else []), "--help")
    assert r.returncode == 0, (tool, sub, r.stderr)
    return r.stdout


def test_documents_name_only_existing_tools_and_flags(built):
    subs = {"render", "overlay", "values", "peaks", "check"}
    seen = set()
    for doc in ("ORCHESTRATOR_PROMPT.md", "instructions.md", "README.md"):
        text = (built.root / doc).read_text()
        assert "{{" not in text and "}}" not in text
        for span in re.findall(r"`([^`\n]+)`", text):
            m = re.search(r"tools/(\w+\.py)(.*)", span)
            if not m:
                continue
            tool, rest = m.group(1), shlex.split(re.sub(r"<[^>]*>", "X", m.group(2)))
            assert (built.root / "tools" / tool).is_file(), (doc, span)
            sub = None
            if tool == "inspect_item.py" and rest and not rest[0].startswith("-"):
                sub = rest[0]
                assert sub in subs, (doc, span)
            helptext = _help(built.root, tool, sub)
            for flag in [t for t in rest if t.startswith("--")]:
                assert re.search(rf"(^|\s|\[){re.escape(flag)}(\s|$|=|\])", helptext, re.M), (doc, span, flag)
            seen.add((tool, sub))
    assert {("inspect_item.py", s) for s in subs} <= seen
    assert ("score_practice.py", None) in seen and ("agreement.py", None) in seen
    prompt = (built.root / "ORCHESTRATOR_PROMPT.md").read_text()
    for b in built.man["batches"]:
        assert b in prompt
    assert f"{built.man['counts']['workers']} worker agents" in prompt and "~/manual_alignment/testpkg" in prompt


def test_fill_template_refuses_missing_values():
    assert ma.fill_template("a {{X}} b", {"X": 1, "Y": 2}) == "a 1 b"
    with pytest.raises(KeyError):
        ma.fill_template("{{MISSING}}", {})


# ---------------------------------------------------------------------- manual spans, scoring, targets


def test_manual_segments(cfg):
    np.testing.assert_array_equal(sy.manual_segments([[0.125, 0.30], [0.30, 0.444]], cfg, 100), [[13, 30], [30, 44]])
    np.testing.assert_array_equal(sy.decoder_segments("manual", np.zeros(100), cfg, {"spans_s": [[0.1, 0.2]]}), [[10, 20]])
    for bad in ([[0.3, 0.2]], [[0.1, 0.3], [0.25, 0.4]], [[0.1, 0.101]], [[0.1, 2.0]], [[float("nan"), 0.2]]):
        with pytest.raises(ValueError):
            sy.manual_segments(bad, cfg, 100)
    assert "manual" not in sy.METHODS
    seconds = [[a / 100, b / 100] for a, b in [[175, 330], [422, 525], [583, 649]]]
    np.testing.assert_array_equal(sy.manual_segments(seconds, cfg, 1000), [[175, 330], [422, 525], [583, 649]])


def test_adoption_rule(cfg):
    rule = ma.settings(cfg)["adoption"]
    full = {"day195_m172": 1.0, "day195_m165": 1.0}
    assert ma.adoption({"day195_m172": 0.34, "day195_m165": 0.34}, {"day195_m172": 0.32, "day195_m165": 0.32}, full, rule)["adopted"]
    assert not ma.adoption({"day195_m172": 0.335, "day195_m165": 0.335}, {"day195_m172": 0.32, "day195_m165": 0.32}, full, rule)["adopted"]
    low = ma.adoption({"day195_m172": 0.5, "day195_m165": 0.5}, {"day195_m172": 0.3, "day195_m165": 0.3},
                      {"day195_m172": 1.0, "day195_m165": 0.5}, rule)
    assert not low["adopted"] and "coverage" in low["reason"]
    assert not ma.adoption({"day195_m172": 0.5}, {"day195_m172": 0.3}, full, rule)["adopted"]


def _bumps(n, centres, width):
    t = np.arange(n)[:, None]
    return -20.0 + 40.0 * np.minimum(np.exp(-0.5 * ((t - np.asarray(centres)[None, :]) / width) ** 2).sum(1), 1.0)


def _frames36(env_db, seed):
    rng = np.random.default_rng(seed)
    f = np.tile(SILENCE36, (env_db.size, 1)).astype(np.float64)
    f[:, 0] = env_db * np.sqrt(18.0) / 10.0 - 4.0
    f[:, 1:18] = np.cumsum(rng.normal(scale=0.1, size=(env_db.size, 17)), axis=0)
    f[:, 18], f[:, 19] = 0.5, 0.3
    return f.astype(np.float32)


@pytest.fixture(scope="module")
def world(cfg):
    """Trials whose 'their' targets put each TTS syllable at a known place; the decoder output is unrelated (flat)."""
    from wvx import alignment as al, targets as tg

    w, T = cfg["model"]["window_len"], 420
    texts = ["You up?", "I hope it."]
    corpus = tg.TTSCorpus("V", texts, [_frames36(_bumps(100, c, 5), k) for k, c in enumerate([[30, 50], [30, 50, 70]])],
                          SILENCE36)
    sil20 = tg.silence_frame(corpus, cfg)
    their = [np.array([[150, 200], [260, 310]]), np.array([[120, 160], [200, 240], [300, 350]])]
    trials, preds, raw_t, tts_f = [], [], [], []
    for k, text in enumerate(texts):
        tts = tg.reduce(corpus.lookup(text), cfg)
        segs = al.segment_syllables(al.envelope_db(tts, cfg), cfg)
        raw = al.stretch_features(tts, segs, their[k], T, sil20, cfg)
        raw_t.append(raw)
        trials.append(SimpleNamespace(neural=np.zeros((T, cfg["model"]["n_features"]), np.float32), text=text,
                                      target=tg.decoder_target(raw, cfg)))
        flat = np.tile(sil20, (T, 1))
        flat[:, 0] += np.linspace(0, 1, T)                    # a decoder output with no syllable structure
        preds.append(tg.scale(flat[w - 1:], cfg))
        tts_f.append(segs)
    env = [al.envelope_db(tg.prediction_timeline(p, T, sil20, cfg), cfg) for p in preds]
    return dict(trials=trials, preds=preds, corpus=corpus, raw=raw_t, syl=their, env=env, tts=tts_f, sil20=sil20)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("syllable_benchmark", REPO_ROOT / "scripts" / "syllable_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec.loader.exec_module(mod)
    return mod


@needs_b2v
def test_manual_answers_equal_to_their_timing_score_the_oracle(cfg, world, bench):
    w = world
    records = [{"spans_s": (s / 100.0).tolist()} for s in w["syl"]]
    out = bench.score_run("manual", False, "forced", w["trials"], w["preds"], w["corpus"], cfg, w["raw"], w["syl"], w["syl"],
                          ["synthetic"] * 2, w["env"], w["tts"], records, False, arm="B")
    h = out["headline"]
    for row in out["per_trial"]:
        assert row["mapping"]["pairing"] == "one_to_one"
        assert row["offline"]["r"] == pytest.approx(row["oracle_syllables_r"], abs=1e-12)
    assert h["target_r_mean"] == pytest.approx(h["oracle_r_mean"], abs=1e-12) and h["oracle_n_trials"] == 2
    assert h["onsets_within_tolerance"] == 1.0 and h["span_onset_error_median_ms"] == 0
    base = bench.score_run("current", False, "forced", w["trials"], w["preds"], w["corpus"], cfg, w["raw"], w["syl"], w["syl"],
                           ["synthetic"] * 2, w["env"], w["tts"], None, False, arm="B")
    assert base["headline"]["target_r_mean"] < h["target_r_mean"] - 0.3        # the flat decoder output carries no timing


@needs_b2v
def test_targets_from_manual_spans_are_one_to_one(cfg, world):
    from wvx import targets as tg

    w = world
    spans = [[[1.0, 1.3], [2.2, 2.9]], [[1.1, 1.5], [1.5, 2.0], [3.0, 3.4]]]
    pred_segs = [sy.manual_segments(s, cfg, t.neural.shape[0]) for s, t in zip(spans, w["trials"])]
    _, res = tg.align_trials(None, w["trials"], w["corpus"], cfg, reorder=False, log=None, preds=w["preds"],
                             tts_segments=w["tts"], pred_segments=pred_segs, arm="B")
    for r, p, t in zip(res, pred_segs, w["tts"]):
        assert r.info["pairing"] == "one_to_one" and r.info["n_decoder_syllables_without_tts"] == 0
        np.testing.assert_array_equal(r.mapped_segments, p)
        assert len(r.mapped_segments) == len(t)
        target = tg.decoder_target(r.target_raw, cfg)
        assert target.shape == (420, 20) and np.all(np.isfinite(target))
        speech = np.zeros(420, bool)
        for a, b in p:
            speech[a:b] = True
        sil = np.all(np.isclose(r.target_raw[:, :18], w["sil20"][None, :18]), axis=1)
        assert not sil[speech].any() and sil[~speech].all()                     # speech exactly in the answered spans


# ------------------------------------------------------------------------------------ script glue


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("manual_alignment_script", REPO_ROOT / "scripts" / "manual_alignment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_host(tmp_path, built, cfg):
    """A host repo folder with the package where the config puts it (relative to the repo)."""
    host = tmp_path / "host"
    rel = Path(cfg["paths"]["manual_alignment"]).relative_to(REPO_ROOT)
    (host / rel).mkdir(parents=True)
    shutil.copytree(built.root, host / rel / "testpkg")
    return host, host / rel


def _local_remote(host):
    def remote(d, inner):
        if "manual_alignment.py scan" in inner:
            return ["true"]
        # the host runs GNU tar; macOS bsdtar would add AppleDouble members (which deliver refuses) without this
        return ["bash", "-c", f"export COPYFILE_DISABLE=1; cd {shlex.quote(str(host))} && {inner}"]
    return remote


def test_tools_leave_the_package_unchanged(built):
    assert ma.read_hash_list(built.root) == ma.file_hashes(built.root)
    assert not list(built.root.rglob("__pycache__")) and not any((built.root / "answers").iterdir())


def test_deliver_and_collect(script, built, cfg, tmp_path, monkeypatch):
    host, root = _fake_host(tmp_path, built, cfg)
    (root / "testpkg_key.json").write_text('{"salt": "x"}')                    # next to the package: never delivered
    monkeypatch.setattr(script, "_remote", _local_remote(host))
    mac = tmp_path / "mac"
    args = SimpleNamespace(name="testpkg", dest=str(mac), src=str(mac))
    assert script.cmd_deliver(cfg, args) == 0
    got = mac / "testpkg"
    assert ma.read_hash_list(got) == ma.file_hashes(got) == ma.file_hashes(built.root)
    assert not list(mac.rglob("*key*")) and not (mac / ".testpkg.partial").exists()
    with pytest.raises(SystemExit):                                              # never over an existing package
        script.cmd_deliver(cfg, args)
    # answers come back as a new snapshot; only JSON and Markdown
    w = built.man["batches"][next(iter(built.man["batches"]))]["worker_id"]
    (got / "answers" / w / "views").mkdir(parents=True)
    (got / "answers" / w / "batch_001.json").write_text("{}")
    (got / "answers" / w / "technique.md").write_text("notes")
    (got / "answers" / w / "views" / "x.png").write_bytes(b"png")
    assert script.cmd_collect(cfg, args) == 0
    snap = sorted((root / "testpkg_answers").glob("collect_*"))
    assert len(snap) == 1
    files = sorted(p.relative_to(snap[0]).as_posix() for p in snap[0].rglob("*") if p.is_file())
    assert files == [f"answers/{w}/batch_001.json", f"answers/{w}/technique.md"]


def test_deliver_refuses_extra_members(script, built, cfg, tmp_path, monkeypatch):
    host, root = _fake_host(tmp_path, built, cfg)
    (root / "testpkg" / "items" / "link.json").symlink_to(root / "testpkg" / "manifest.json")
    monkeypatch.setattr(script, "_remote", _local_remote(host))
    mac = tmp_path / "mac"
    with pytest.raises(SystemExit, match="link not allowed"):
        script.cmd_deliver(cfg, SimpleNamespace(name="testpkg", dest=str(mac)))
    assert not (mac / "testpkg").exists()
    (root / "testpkg" / "items" / "link.json").unlink()
    (root / "testpkg" / "items" / "stray.txt").write_text("x")                 # not in files.sha256
    with pytest.raises(SystemExit, match="local check failed"):
        script.cmd_deliver(cfg, SimpleNamespace(name="testpkg", dest=str(mac)))
    assert not (mac / "testpkg").exists()


def test_ingest_on_a_fake_host(script, built, cfg, tmp_path):
    from wvx.config import deep_merge

    data, res_dir = tmp_path / "ma", tmp_path / "results"
    c = deep_merge(cfg, {"paths": {"manual_alignment": data, "manual_alignment_results": res_dir}})
    shutil.copytree(built.root, data / "testpkg")
    key = {"items": {r["id"]: {"set": r["set"], "cell": r["cell"], "session": r["session"], "trial": r["trial"]}
                     for r in built.chosen["real"]},
           "files_sha256": "f" * 64}
    (data / "testpkg_key.json").write_text(json.dumps(key))
    snap = data / "testpkg_answers" / "collect_20261021T000000Z" / "answers"
    for (w, b), obj in _worker_files(built, {}).items():
        (snap / w).mkdir(parents=True, exist_ok=True)
        (snap / w / f"{b}.json").write_text(json.dumps(obj))
    (snap / w / "notes.md").write_text("x")
    args = SimpleNamespace(name="testpkg", snapshot=None, label=None, workers=None)
    assert script.cmd_ingest(c, {}, args) == 0
    agg = json.loads((data / "testpkg_work" / "aggregated.json").read_text())
    assert set(agg["items"]) == set(built.man["items"]) and agg["files_sha256"] == "f" * 64
    summary = (res_dir / "testpkg" / "ingest_summary.json").read_text()
    assert not any(i in summary for i in built.man["items"])                   # results/ never names an item
    s = json.loads(summary)
    k = built.man["counts"]["redundancy_k"]
    assert s["coverage"]["calibration"] == {"n_items": 2, "with_answer": 2, f"with_{k}_or_more": 2}
    assert s["n_rejections"] == 0 and s["n_ignored_files"] == 1

    # A second answer set of the same package (--label, --workers): its own work and results folders; the first
    # set's outputs stay as they were. Only the matching worker folders are ingested.
    before = (data / "testpkg_work" / "aggregated.json").read_text()
    first = sorted(p.name for p in snap.iterdir())[0]
    args2 = SimpleNamespace(name="testpkg", snapshot=None, label="second", workers=re.escape(first))
    assert script.cmd_ingest(c, {}, args2) == 0
    assert (data / "testpkg_work" / "aggregated.json").read_text() == before
    acc2 = json.loads((data / "testpkg_work_second" / "accepted.json").read_text())
    assert acc2["answers"] and {a["worker_id"] for a in acc2["answers"]} == {first}
    assert acc2["label"] == "second" and acc2["workers_filter"] == re.escape(first)
    s2 = json.loads((res_dir / "testpkg" / "second" / "ingest_summary.json").read_text())
    assert s2["n_ignored_files"] > s["n_ignored_files"]
    assert script._paths(c, "testpkg", "second")["targets"].name.endswith("_second")
