"""Manual syllable alignment by external annotator agents: blind task packages, answers, aggregation, the blindness
scan, and the adoption rule. Settings: ``manual_alignment`` in configs/experiments/manual_alignment.yaml; entry point
scripts/manual_alignment.py; package templates and the standalone tools in configs/manual_alignment/.

**Why.** The automatic timing (D-037: arm B, c0 dips, forced TTS syllables) reaches target r 0.316 against their
timed targets, the oracle 0.46. Annotator agents mark, for every TTS syllable of the cue, the burst of the decoder
output where T15 attempted it; the TTS syllables are then stretched onto those spans one to one (arm B with
``wvx.syllables.manual_segments``), by the unchanged scoring and target code.

**Items.**
- Blind calibration: a seeded subset of trials of a Wairagkar session, the SAME trials in every cell of a group
  (day195 decoded by the 23- and 30-day-old models; day193 by the 28-day-old one, confirmation). Scored on the host
  against their ``syl_indices`` and ``_syl`` targets.
- Production: every Kunz attempted trial. Their answers make the new targets.
- Practice (with reference spans, for local feedback) and worked examples: other trials of the same sessions,
  disjoint from both, and (``exclude_shared_text``) with no cue that a real item uses.

**Blindness.** Calibration and production items are mixed under opaque ids (``make_ids``: salted SHA-256; the salt
lives only in the key on the host). Nothing in the package names a session, cell, decoder or set, and no real item's
true timing is in it: ``scan_package`` enforces a file allowlist, forbidden key/text patterns, PNGs without text
chunks, fixed npz array names, and searches every JSON number list and array for the boundary sequence of every
calibration item (seconds and frames). Items that share a trial (one trial, two decoders) never go to the same worker.

**Time axis.** The neural timeline: frame k (10 ms bin k) centred at (k + 0.5) * 10 ms. The loudness and neural
traces are per frame; the spectrogram is cut from the decoder's LPCNet audio at the same centres plus the measured
synthesis lag (``syllables.audio_lag_ms``), so all three panels share one axis and an answer in seconds maps to bins
by rounding (``wvx.syllables.manual_segments``).

**Answers** (instructions.md has the format): one file per worker and batch. ``validate_response`` is the same rule
set as the package's tools/inspect_item.py (a standalone copy; tests/test_manual_alignment.py runs both on the same
cases). Bad answers are rejected with reasons, never fixed. Several answers per item are aggregated by the median of
each start and each end (still ordered and valid: medians preserve pointwise order), with pairwise agreement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]

DEFAULT_CONFIG = "configs/experiments/manual_alignment.yaml"
TOOL_FILES = ("inspect_item.py", "score_practice.py", "agreement.py")
ARRAY_NAMES = ("t_s", "mel_db", "mel_hz", "loudness_db", "neural", "duration_s")
ROOT_FILES = ("README.md", "ORCHESTRATOR_PROMPT.md", "instructions.md", "manifest.json", "settings.json", "files.sha256")
PACKAGE_DIRS = ("items", "practice", "practice/items", "practice/truth", "examples", "tools", "answers")

# The answer format; identical in tools/inspect_item.py.
RESPONSE_KEYS = ("schema", "batch", "worker_id", "items")
RESPONSE_OPTIONAL = ("notes",)
ITEM_KEYS = ("item_id", "syllables")
ITEM_OPTIONAL = ("confidence", "notes", "unassigned_bursts")
SYLLABLE_KEYS = ("index", "label", "start_s", "end_s")
SYLLABLE_OPTIONAL = ("confidence",)

# Keys of the package's own files (blindness: nothing else may appear in them).
ITEM_FILE_KEYS = {"schema", "item_id", "kind", "text", "duration_s", "n_frames", "frame_s", "n_syllables", "syllables",
                  "files", "response_template"}


def settings(cfg: Mapping[str, Any]) -> dict:
    return cfg["manual_alignment"]


def frame_s(cfg: Mapping[str, Any]) -> float:
    s = cfg["synthesis"]
    return float(s["samples_per_frame"]) / float(s["sample_rate"])


def load_config(path: PathLike = DEFAULT_CONFIG, overrides: Optional[Mapping] = None) -> dict:
    from wvx.experiment_config import load_experiment_config

    return load_experiment_config(path, overrides)


# ------------------------------------------------------------------------------------ selection + ids


def wellformed(segs: np.ndarray, n_bins: int) -> bool:
    """Their syllables of a trial are usable as truth: at least one, non-empty, ordered, without overlap, inside."""
    s = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    return bool(len(s) and np.all(s[:, 1] > s[:, 0]) and np.all(s[1:, 0] >= s[:-1, 1]) and s.min() >= 0
                and s.max() <= n_bins)


def select(candidates: Sequence[int], n: int, rng: np.random.Generator, what: str) -> list[int]:
    """``n`` of ``candidates`` without replacement, sorted (raises if there are fewer)."""
    c = np.asarray(sorted(set(int(x) for x in candidates)), dtype=np.int64)
    if n > c.size:
        raise ValueError(f"{what}: {n} wanted, only {c.size} eligible")
    return sorted(int(x) for x in rng.choice(c, size=int(n), replace=False))


def choose_items(info: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any], rng: np.random.Generator) -> dict:
    """Which (cell, trial) pairs become calibration, production, example and practice items.

    ``info[cell]``: {"session", "n", "wellformed" (their syllables usable, per trial; None for Kunz), "text" (cue
    keys), "n_truth", "n_tts", "duration_s"} per trial. Calibration: per group, ``n_trials`` of the well-formed trials,
    the same trials in every cell of the group. Production: every trial. Examples, then practice: well-formed trials
    of their sessions that no earlier set uses, without a cue of any real item (``exclude_shared_text``); examples
    also need as many true as TTS syllables and a short duration; each practice trial is shown with one of its
    group's cells. Every set is disjoint from every other by (session, trial); checked. Returns {"real", "examples",
    "practice"}: lists of {"set", "group", "cell", "session", "trial"}.
    """
    s = settings(cfg)

    def session_of(cells):
        ss = {info[c]["session"] for c in cells}
        if len(ss) != 1:
            raise ValueError(f"cells {list(cells)} belong to different sessions")
        return ss.pop()

    def item(set_, group, cell, trial):
        return {"set": set_, "group": group, "cell": cell, "session": info[cell]["session"], "trial": int(trial)}

    used: dict = {}
    calib, prod = [], []
    for gname, g in s["calibration"]["groups"].items():
        if not g["enabled"]:
            continue
        sess, c0 = session_of(g["cells"]), info[g["cells"][0]]
        sel = select([i for i in range(c0["n"]) if c0["wellformed"][i]], int(g["n_trials"]), rng, f"calibration {gname}")
        used.setdefault(sess, set()).update(sel)
        calib += [item("calibration", gname, c, i) for c in g["cells"] for i in sel]
    for c in s["production"]["cells"]:
        prod += [item("production", "production", c, i) for i in range(info[c]["n"])]
    real = calib + prod
    real_texts = {info[r["cell"]]["text"][r["trial"]] for r in real}

    def eligible(cell, extra=lambda c, i: True):
        c = info[cell]
        return [i for i in range(c["n"]) if c["wellformed"][i] and i not in used.get(c["session"], set())
                and not (s["exclude_shared_text"] and c["text"][i] in real_texts) and extra(c, i)]

    e = s["examples"]
    ex_sel = select(eligible(e["cell"], lambda c, i: (c["n_truth"][i] == c["n_tts"][i] or not e["require_equal_counts"])
                             and c["duration_s"][i] <= float(e["max_duration_s"])), int(e["n"]), rng, "examples")
    examples = [item("example", "examples", e["cell"], i) for i in ex_sel]
    used.setdefault(info[e["cell"]]["session"], set()).update(ex_sel)
    practice = []
    for gname, g in s["practice"]["groups"].items():
        sess = session_of(g["cells"])
        sel = select(eligible(g["cells"][0]), int(g["n_items"]), rng, f"practice {gname}")
        which = [g["cells"][j % len(g["cells"])] for j in range(len(sel))]
        which = [which[j] for j in rng.permutation(len(which))]
        practice += [item("practice", gname, c, i) for c, i in zip(which, sel)]
        used.setdefault(sess, set()).update(sel)
    sets = {"calibration": calib, "production": prod, "examples": examples, "practice": practice}
    keys = {k: {(x["session"], x["trial"]) for x in v} for k, v in sets.items()}
    names = list(keys)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if keys[names[a]] & keys[names[b]]:
                raise ValueError(f"{names[a]} and {names[b]} share trials")
    return {"real": real, "examples": examples, "practice": practice}


def make_ids(keys: Sequence[str], salt: str, namespace: str, n_hex: int, prefix: str = "") -> dict:
    """{key: prefix + n_hex hex characters of SHA-256(namespace|salt|key)}; unique or raises."""
    out = {k: prefix + hashlib.sha256(f"{namespace}|{salt}|{k}".encode()).hexdigest()[: int(n_hex)] for k in keys}
    if len(set(out.values())) != len(out):
        raise ValueError(f"id collision in namespace {namespace!r}: use more hex characters")
    return out


def assign_work(item_ids: Sequence[str], group_of: Mapping[str, str], n_workers: int, k: int, per_batch: int,
                rng: np.random.Generator, id_format: str) -> tuple[dict, dict]:
    """Batches of real items for a flat team: every item goes to ``k`` different workers.

    Workers are split into k groups (seeded); group r gets every item once, balanced over its workers, and items with
    the same ``group_of`` key (the same trial under two decoders) never go to the same worker. Each worker's items are
    cut into batches of at most ``per_batch``; batch numbers are shuffled. Returns ({worker: [batch ids]},
    {batch id: {"worker_id", "items"}}).
    """
    workers = [id_format.format(i + 1) for i in range(int(n_workers))]
    if not 1 <= int(k) <= len(workers):
        raise ValueError(f"redundancy k={k} needs 1 <= k <= {len(workers)} workers")
    order = [workers[i] for i in rng.permutation(len(workers))]
    groups = [order[r::int(k)] for r in range(int(k))]
    made = []
    for grp in groups:
        cap = math.ceil(len(item_ids) / len(grp))
        load: dict = {w: [] for w in grp}
        taken: dict = {w: set() for w in grp}
        for it in [item_ids[i] for i in rng.permutation(len(item_ids))]:
            g = group_of.get(it, it)
            free = [w for w in grp if g not in taken[w]]
            if not free:
                raise ValueError(f"no worker of a group of {len(grp)} can take item {it} without its sibling")
            room = [w for w in free if len(load[w]) < cap] or free
            w = min(room, key=lambda x: (len(load[x]), grp.index(x)))
            load[w].append(it)
            taken[w].add(g)
        for w in grp:
            its = sorted(load[w])
            nb = math.ceil(len(its) / int(per_batch))
            made += [(w, sorted(c.tolist())) for c in np.array_split(np.asarray(its, dtype=object), nb) if len(c)] if nb else []
    if len(made) > 999:
        raise ValueError(f"{len(made)} batches: batch ids have three digits")
    batches = {f"batch_{n + 1:03d}": {"worker_id": made[i][0], "items": made[i][1]}
               for n, i in enumerate(rng.permutation(len(made)))}
    return {w: sorted(b for b, v in batches.items() if v["worker_id"] == w) for w in workers}, batches


def assign_practice(practice_ids: Sequence[str], workers: Sequence[str], per_round: int, rounds: int,
                    rng: np.random.Generator) -> dict:
    """{practice batch: {"worker_id", "rounds": [[ids] * per_round] * rounds}}: every worker gets per_round * rounds
    distinct practice items (fresh ones each round), spread evenly over the practice set."""
    order = [practice_ids[i] for i in rng.permutation(len(practice_ids))]
    need = int(per_round) * int(rounds)
    if need > len(order):
        raise ValueError(f"{need} practice items per worker, only {len(order)} exist")
    out = {}
    for n, w in enumerate(workers):
        seq = [order[(n * need + j) % len(order)] for j in range(need)]
        out[f"practice_{n + 1:02d}"] = {"worker_id": w,
                                        "rounds": [seq[r * per_round:(r + 1) * per_round] for r in range(int(rounds))]}
    return out


# ------------------------------------------------------------------------------------ item content


def syllable_labels(words: Sequence[str], syllable_words: Sequence[int], syllable_phones: Sequence[Sequence[str]]) -> list:
    """Per TTS syllable: index, label ("hope", or "thirsty 1/2" for the first of two), word, its position, phones."""
    counts = Counter(int(w) for w in syllable_words)
    pos: Counter = Counter()
    out = []
    for k, (wi, ph) in enumerate(zip(syllable_words, syllable_phones)):
        wi = int(wi)
        pos[wi] += 1
        n, w = counts[wi], str(words[wi])
        out.append({"index": k, "label": w if n == 1 else f"{w} {pos[wi]}/{n}", "word": w, "word_index": wi,
                    "syllable_in_word": pos[wi], "syllables_in_word": n, "phones": " ".join(ph)})
    return out


def time_axis(n_frames: int, cfg: Mapping[str, Any]) -> np.ndarray:
    """Centre of every frame (s) on the neural timeline."""
    return ((np.arange(int(n_frames)) + 0.5) * frame_s(cfg)).astype(settings(cfg)["arrays"]["time_dtype"])


def mel_frequencies(cfg: Mapping[str, Any]) -> np.ndarray:
    import librosa

    m = settings(cfg)["mel"]
    return librosa.mel_frequencies(int(m["n_mels"]) + 2, fmin=float(m["fmin_hz"]), fmax=float(m["fmax_hz"]),
                                   htk=bool(m["htk"]))[1:-1].astype(np.float32)


def mel_db(wav: np.ndarray, n_frames: int, cfg: Mapping[str, Any]) -> np.ndarray:
    """(n_mels, n_frames) mel power in dB re the item's loudest bin, floored at -top_db. Frame k's window is centred
    at audio sample (k + 0.5) * samples_per_frame + the synthesis lag: the audio of feature frame k."""
    import librosa
    import scipy.signal

    m = settings(cfg)["mel"]
    sr, spf = int(cfg["synthesis"]["sample_rate"]), int(cfg["synthesis"]["samples_per_frame"])
    x = np.asarray(wav, dtype=np.float64).ravel()
    if x.size != int(n_frames) * spf:
        raise ValueError(f"audio has {x.size} samples, {n_frames} frames need {int(n_frames) * spf}")
    n_fft, win = int(m["n_fft"]), int(m["win_length"])
    lag = int(round(float(cfg["syllables"]["audio_lag_ms"]) * sr / 1000.0))
    centres = np.arange(int(n_frames)) * spf + spf // 2 + lag
    xp = np.concatenate([np.zeros(n_fft), x, np.zeros(n_fft)])
    w = np.zeros(n_fft)
    off = (n_fft - win) // 2
    w[off:off + win] = scipy.signal.get_window(m["window"], win, fftbins=True)
    frames = xp[(centres - n_fft // 2 + n_fft)[:, None] + np.arange(n_fft)[None, :]] * w
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=int(m["n_mels"]), fmin=float(m["fmin_hz"]),
                             fmax=float(m["fmax_hz"]), htk=bool(m["htk"]), norm=m["norm"])
    db = 10.0 * np.log10(np.maximum(fb @ power.T, float(m["amin"])))
    return np.maximum(db - db.max(), -float(m["top_db"]))


def loudness_db(pred_raw: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """The smoothed c0 envelope the c0-dip method segments (``wvx.alignment.envelope_db``), in dB re its maximum."""
    from wvx import alignment as al

    env = al.envelope_db(np.asarray(pred_raw, dtype=np.float32), cfg)
    if settings(cfg)["loudness"]["reference"] != "item_max":
        raise ValueError("loudness.reference must be item_max")
    return env - env.max()


def neural_activity(neural: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Mean over the chosen decoder-input feature columns per bin, Gaussian-smoothed, z-scored within the trial."""
    from scipy.ndimage import gaussian_filter1d

    n = settings(cfg)["neural"]
    x = np.asarray(neural, dtype=np.float64)
    half = int(cfg["model"]["n_features"]) // 2
    if x.ndim != 2 or x.shape[1] != 2 * half:
        raise ValueError(f"neural features {x.shape}, want (T, {2 * half})")
    cols = {"spikepow": slice(0, half), "threshcross": slice(half, None), "both": slice(None)}[n["features"]]
    y = gaussian_filter1d(x[:, cols].mean(axis=1), float(n["smooth_ms"]) / (1000.0 * frame_s(cfg)), mode="nearest")
    if n["normalize"] != "zscore":
        raise ValueError("neural.normalize must be zscore")
    sd = y.std()
    return (y - y.mean()) / sd if sd > 0 else np.zeros_like(y)


def item_arrays(wav: np.ndarray, pred_raw: np.ndarray, neural: Optional[np.ndarray], cfg: Mapping[str, Any]) -> dict:
    """The arrays of one item (``ARRAY_NAMES``): nothing but the signals and their axes."""
    a = settings(cfg)["arrays"]
    T = int(np.asarray(pred_raw).shape[0])
    out = {"t_s": time_axis(T, cfg), "mel_db": mel_db(wav, T, cfg).astype(a["dtype"]),
           "mel_hz": mel_frequencies(cfg), "loudness_db": loudness_db(pred_raw, cfg).astype(a["dtype"]),
           "duration_s": np.float64(round(T * frame_s(cfg), 6))}
    if settings(cfg)["neural"]["enabled"]:
        if neural is None or np.asarray(neural).shape[0] != T:
            raise ValueError("neural features missing or of another length than the decoder output")
        out["neural"] = neural_activity(neural, cfg).astype(a["dtype"])
    for k, v in out.items():
        if not np.all(np.isfinite(np.asarray(v, dtype=np.float64))):
            raise ValueError(f"item array {k} is not finite")
    return out


KIND_DIRS = {"item": "items", "practice": "practice/items", "example": "examples"}


def item_record(item_id: str, kind: str, text: str, labels: Sequence[dict], n_frames: int, cfg: Mapping[str, Any]) -> dict:
    """The item JSON the annotators read (and the answer rules use)."""
    d = KIND_DIRS[kind]
    return {
        "schema": settings(cfg)["schema"]["item"], "item_id": item_id, "kind": kind, "text": text,
        "duration_s": round(int(n_frames) * frame_s(cfg), 6), "n_frames": int(n_frames), "frame_s": frame_s(cfg),
        "n_syllables": len(labels), "syllables": [dict(s) for s in labels],
        "files": {"image": f"{d}/{item_id}.png", "arrays": f"{d}/{item_id}.npz"},
        "response_template": {"item_id": item_id, "syllables": [{"index": s["index"], "label": s["label"],
                                                                 "start_s": None, "end_s": None} for s in labels]},
    }


def spans_seconds(segs: np.ndarray, cfg: Mapping[str, Any]) -> list:
    """[onset, offset) frames -> [[start_s, end_s]] (frame boundaries)."""
    fs = frame_s(cfg)
    return [[round(int(a) * fs, 6), round(int(b) * fs, 6)] for a, b in np.asarray(segs, dtype=np.int64).reshape(-1, 2)]


def reference_record(item_id: str, item_sha256: str, segs: np.ndarray, cfg: Mapping[str, Any]) -> dict:
    """practice/truth/<id>.json: the reference spans of one PRACTICE item, bound to its item file by hash."""
    return {"schema": settings(cfg)["schema"]["reference"], "item_id": item_id, "item_sha256": item_sha256,
            "n_syllables": int(len(segs)),
            "syllables": [{"index": j, "start_s": a, "end_s": b} for j, (a, b) in enumerate(spans_seconds(segs, cfg))]}


def example_answer(item_id: str, labels: Sequence[dict], segs: np.ndarray, cfg: Mapping[str, Any]) -> dict:
    """examples/<id>_answer.json: the reference spans as an item answer (needs as many spans as syllables)."""
    sp = spans_seconds(segs, cfg)
    if len(sp) != len(labels):
        raise ValueError("an example needs as many reference syllables as TTS syllables")
    return {"item_id": item_id, "syllables": [{"index": s["index"], "label": s["label"], "start_s": a, "end_s": b}
                                              for s, (a, b) in zip(labels, sp)]}


def json_bytes(obj: Any) -> bytes:
    """The exact bytes every package JSON is written with (hashes are taken over these)."""
    return (json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")


def fill_template(text: str, values: Mapping[str, Any]) -> str:
    """Replace every {{KEY}} of a package template; raises if a placeholder has no value."""
    def sub(m):
        k = m.group(1)
        if k not in values:
            raise KeyError(f"template placeholder {{{{{k}}}}} has no value")
        return str(values[k])

    return re.sub(r"\{\{([A-Z0-9_]+)\}\}", sub, text)


# ------------------------------------------------------------------------------------------ packaging


def templates_dir(cfg: Mapping[str, Any]) -> Path:
    from wvx.config import resolve_path

    return resolve_path(settings(cfg)["templates"])


def load_tool(cfg: Mapping[str, Any], name: str = "inspect_item"):
    """The package's own tool module (configs/manual_alignment/tools/), so the images are drawn by the same code the
    annotators run. No bytecode is written next to the templates."""
    import importlib.util
    import sys

    p = templates_dir(cfg) / "tools" / f"{name}.py"
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location(f"wvx_ma_tool_{name}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def package_settings(cfg: Mapping[str, Any]) -> dict:
    """settings.json: everything the tools need, nothing about where items come from."""
    s = settings(cfg)
    return {"schema": s["schema"]["settings"], "frame_s": frame_s(cfg),
            "schemas": {k: s["schema"][k] for k in ("item", "response", "reference", "manifest")},
            "display": {k: v for k, v in s["display"].items() if k != "show_auto_boundaries"},
            "peaks": s["peaks"], "values": s["values"],
            "response": {**s["response"], "extra_batch_id": s["batching"]["extra_batch_id"]},
            "agreement": s["agreement"], "practice_scoring": s["practice_scoring"],
            "ids": {"practice_prefix": s["ids"]["practice_prefix"], "example_prefix": s["ids"]["example_prefix"]}}


def template_values(cfg: Mapping[str, Any], name: str, man: Mapping[str, Any], examples_md: str, workers_md: str) -> dict:
    s = settings(cfg)
    c = man["counts"]
    o = s["orchestrator"]
    return {
        "NAME": name, "PACKAGE_DIR": f"{s['delivery']['mac_root'].rstrip('/')}/{name}", "N_ITEMS": c["items"],
        "N_PRACTICE": c["practice_items"], "N_EXAMPLES": c["examples"], "N_WORKERS": c["workers"], "K": c["redundancy_k"],
        "N_BATCHES": c["batches"], "ITEMS_PER_BATCH": c["items_per_batch"], "PRACTICE_ROUNDS": s["practice"]["max_rounds"],
        "ITEMS_PER_ROUND": s["practice"]["items_per_round"], "RESPONSE_SCHEMA": s["schema"]["response"],
        "MIN_SPAN_S": s["response"]["min_span_s"],
        "CONFIDENCE_VALUES": ", ".join(f"`{v}`" for v in s["response"]["confidence_values"]),
        "EXTRA_BATCH_ID": s["batching"]["extra_batch_id"], "TOLERANCE_MS": s["practice_scoring"]["tolerance_ms"],
        "AGREEMENT_THRESHOLD_MS": s["agreement"]["flag_median_abs_start_ms"], "PX_PER_SECOND": s["display"]["px_per_second"],
        "SECONDS_PER_ROW": s["display"]["seconds_per_row"], "ROW_OVERLAP_S": s["display"]["row_overlap_s"],
        "PRACTICE_PREFIX": s["ids"]["practice_prefix"], "EXAMPLE_LIST": examples_md, "WORKER_TABLE": workers_md,
        "WORKER_MODEL": o["worker_model"], "WORKER_MODEL_ID": o["worker_model_id"] or f"<{o['worker_model']} model id>",
    }


def write_item(stg: Path, cfg: Mapping[str, Any], tool, st: Mapping[str, Any], item_id: str, kind: str,
               src: Mapping[str, Any]) -> bytes:
    """Arrays (.npz), item JSON and overview PNG of one item. ``src``: {"wav", "pred_raw", "neural", "text",
    "labels", "auto" (optional faint spans)}. Returns the item JSON bytes."""
    d = Path(stg) / KIND_DIRS[kind]
    d.mkdir(parents=True, exist_ok=True)
    pred_raw = np.asarray(src["pred_raw"])
    np.savez_compressed(d / f"{item_id}.npz", **item_arrays(src["wav"], pred_raw, src.get("neural"), cfg))
    data = json_bytes(item_record(item_id, kind, src["text"], src["labels"], pred_raw.shape[0], cfg))
    (d / f"{item_id}.json").write_bytes(data)
    tool.render(item_id, d / f"{item_id}.png", stg, auto=src.get("auto"), settings=st)
    return data


def build_package(stg: PathLike, cfg: Mapping[str, Any], name: str, chosen: Mapping[str, list], source, salt: str,
                  log=None) -> dict:
    """Write a whole package into the empty folder ``stg``; returns the manifest.

    ``chosen``: ``choose_items`` output (every dict gets its "id"); ``source(item)`` -> the ``write_item`` inputs plus
    "truth" ((J, 2) frames, practice and examples). Ids, item order, worker groups and batches come from the secret
    ``salt``; nothing that names a source enters the package. The blindness scan is the caller's next step.
    """
    s = settings(cfg)
    stg = Path(stg)
    rng = np.random.default_rng(int(salt, 16))
    real, practice, examples = chosen["real"], chosen["practice"], chosen["examples"]
    rid = make_ids([f"{r['cell']}|{r['trial']}" for r in real], salt, "real", s["ids"]["hex_chars"])
    pid = make_ids([f"{r['cell']}|{r['trial']}" for r in practice], salt, "practice", s["ids"]["hex_chars"],
                   s["ids"]["practice_prefix"])
    for r in real:
        r["id"] = rid[f"{r['cell']}|{r['trial']}"]
    for r in practice:
        r["id"] = pid[f"{r['cell']}|{r['trial']}"]
    for n, r in enumerate(examples):
        r["id"] = f"{s['ids']['example_prefix']}{n + 1}"
    sibling = {r["id"]: f"{r['session']}|{r['trial']}" for r in real}
    workers, batches = assign_work(sorted(rid.values()), sibling, int(s["workers"]["n"]), int(s["batching"]["redundancy_k"]),
                                   int(s["batching"]["items_per_batch"]), rng, s["workers"]["id_format"])
    pbatches = assign_practice(sorted(pid.values()), list(workers), int(s["practice"]["items_per_round"]),
                               int(s["practice"]["max_rounds"]), rng)
    wmap = {w: {"practice_batch": next(b for b, v in pbatches.items() if v["worker_id"] == w), "batches": bs}
            for w, bs in workers.items()}

    (stg / "answers").mkdir(parents=True)
    st = package_settings(cfg)
    (stg / "settings.json").write_bytes(json_bytes(st))
    tool = load_tool(cfg)
    for n, r in enumerate(sorted(real, key=lambda x: x["id"])):
        write_item(stg, cfg, tool, st, r["id"], "item", source(r))
        if log and (n + 1) % 25 == 0:
            log(f"{n + 1}/{len(real)} items")
    (stg / "practice" / "truth").mkdir(parents=True)
    for r in sorted(practice, key=lambda x: x["id"]):
        src = source(r)
        data = write_item(stg, cfg, tool, st, r["id"], "practice", {**src, "auto": None})
        ref = reference_record(r["id"], hashlib.sha256(data).hexdigest(), src["truth"], cfg)
        (stg / "practice" / "truth" / f"{r['id']}.json").write_bytes(json_bytes(ref))
    ex_lines = []
    for r in examples:
        src = source(r)
        write_item(stg, cfg, tool, st, r["id"], "example", {**src, "auto": None})
        ans = example_answer(r["id"], src["labels"], src["truth"], cfg)
        (stg / "examples" / f"{r['id']}_answer.json").write_bytes(json_bytes(ans))
        tool.render(r["id"], stg / "examples" / f"{r['id']}_overlay.png", stg,
                    reference=[[x["start_s"], x["end_s"]] for x in ans["syllables"]], settings=st)
        ex_lines.append(f"- `examples/{r['id']}.png`, `examples/{r['id']}_answer.json`, `examples/{r['id']}_overlay.png`: "
                        f"{len(src['labels'])} syllables, {np.asarray(src['pred_raw']).shape[0] * frame_s(cfg):.2f} s")
    (stg / "tools").mkdir()
    for f in TOOL_FILES:
        (stg / "tools" / f).write_bytes((templates_dir(cfg) / "tools" / f).read_bytes())
    (stg / "tools" / "requirements.txt").write_text("\n".join(s["local_setup"]["requirements"]) + "\n")
    man = {
        "schema": s["schema"]["manifest"], "package": name,
        "counts": {"items": len(real), "practice_items": len(practice), "examples": len(examples), "workers": len(workers),
                   "batches": len(batches), "redundancy_k": int(s["batching"]["redundancy_k"]),
                   "items_per_batch": int(s["batching"]["items_per_batch"])},
        "answer_paths": {"batch": "answers/<worker_id>/<batch_id>.json", "extra": "answers/<worker_id>/extra.json",
                         "practice": "answers/<worker_id>/practice/round<r>.json"},
        "workers": wmap, "batches": batches, "practice_batches": pbatches,
        "items": sorted(rid.values()), "practice_items": sorted(pid.values()), "examples": [r["id"] for r in examples],
    }
    (stg / "manifest.json").write_bytes(json_bytes(man))
    wt = ["| worker | practice batch | batches (items) |", "|---|---|---|"]
    for w, v in wmap.items():
        wt.append(f"| {w} | {v['practice_batch']} | " + ", ".join(f"{b} ({len(batches[b]['items'])})" for b in v["batches"]) + " |")
    vals = template_values(cfg, name, man, "\n".join(ex_lines), "\n".join(wt))
    for f in ("README.md", "instructions.md", "ORCHESTRATOR_PROMPT.md"):
        (stg / f).write_text(fill_template((templates_dir(cfg) / f).read_text(), vals))
    write_hash_list(stg)
    return man


def package_sizes(pkg: PathLike) -> dict:
    """MB per file type and in total."""
    out: dict = {}
    for p in Path(pkg).rglob("*"):
        if p.is_file():
            out[p.suffix or p.name] = out.get(p.suffix or p.name, 0) + p.stat().st_size
    out["total"] = sum(out.values())
    return {k: round(v / 1e6, 2) for k, v in sorted(out.items())}


# ------------------------------------------------------------------------------------ answer rules


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _key_errors(obj: dict, required, optional, code: str, where: str) -> list:
    errs = []
    missing = [k for k in required if k not in obj]
    unknown = sorted(str(k) for k in obj if k not in required and k not in optional)
    if missing:
        errs.append((code, f"{where}: missing {', '.join(missing)}"))
    if unknown:
        errs.append((code, f"{where}: unknown key(s) {', '.join(unknown)}"))
    return errs


def rule_meta(item: Mapping[str, Any]) -> dict:
    """What the answer rules need from an item file."""
    return {"n_syllables": int(item["n_syllables"]), "labels": [s["label"] for s in item["syllables"]],
            "duration_s": float(item["duration_s"]), "kind": item["kind"]}


def response_rules(cfg: Mapping[str, Any]) -> dict:
    s = settings(cfg)
    return {**s["response"], "schema": s["schema"]["response"], "extra_batch_id": s["batching"]["extra_batch_id"]}


def validate_item_answer(entry: Any, meta: Optional[Mapping[str, Any]], rules: Mapping[str, Any], where: str = "item") -> list:
    """[(code, message)] for one item answer (``meta``: ``rule_meta`` of the item, None if unknown)."""
    tol, conf = float(rules["tolerance_s"]), list(rules["confidence_values"])
    if not isinstance(entry, dict):
        return [("E_ITEM_TYPE", f"{where}: not an object")]
    errs = _key_errors(entry, ITEM_KEYS, ITEM_OPTIONAL, "E_ITEM_KEYS", where)
    iid = entry.get("item_id")
    if not isinstance(iid, str):
        return errs + [("E_ITEM_ID", f"{where}: item_id is not a string")]
    where = f"item {iid}"
    if meta is None:
        return errs + [("E_ITEM_ID", f"{where}: unknown item")]
    dur = float(meta["duration_s"])
    if "confidence" in entry and entry["confidence"] not in conf:
        errs.append(("E_CONFIDENCE", f"{where}: confidence not in {conf}"))
    if "notes" in entry and not isinstance(entry["notes"], str):
        errs.append(("E_NOTES", f"{where}: notes is not a string"))
    if "unassigned_bursts" in entry:
        ub = entry["unassigned_bursts"]
        if not (isinstance(ub, list) and all(isinstance(p, list) and len(p) == 2 and all(_is_number(v) and math.isfinite(v) for v in p)
                                             and -tol <= p[0] < p[1] <= dur + tol for p in ub)):
            errs.append(("E_UNASSIGNED", f"{where}: unassigned_bursts must be [start_s, end_s] pairs inside the item"))
    if "syllables" not in entry:
        return errs
    syl = entry["syllables"]
    if not isinstance(syl, list):
        return errs + [("E_SYLLABLES", f"{where}: syllables is not a list")]
    if len(syl) != int(meta["n_syllables"]):
        return errs + [("E_COUNT", f"{where}: {len(syl)} syllables, expected exactly {meta['n_syllables']}")]
    prev_end = None
    for k, s in enumerate(syl):
        w = f"{where} syllable {k}"
        if not isinstance(s, dict):
            errs.append(("E_SYL_TYPE", f"{w}: not an object"))
            prev_end = None
            continue
        errs += _key_errors(s, SYLLABLE_KEYS, SYLLABLE_OPTIONAL, "E_SYL_KEYS", w)
        if "index" in s and (not isinstance(s["index"], int) or isinstance(s["index"], bool) or s["index"] != k):
            errs.append(("E_INDEX", f"{w}: index is not {k}"))
        if "label" in s and s["label"] != meta["labels"][k]:
            errs.append(("E_LABEL", f"{w}: label {s['label']!r} is not {meta['labels'][k]!r}"))
        if "confidence" in s and s["confidence"] not in conf:
            errs.append(("E_CONFIDENCE", f"{w}: confidence not in {conf}"))
        ok = True
        for name in ("start_s", "end_s"):
            if name not in s:
                ok = False
            elif not _is_number(s[name]):
                errs.append(("E_NUMBER", f"{w}: {name} is not a number"))
                ok = False
            elif not math.isfinite(s[name]):
                errs.append(("E_FINITE", f"{w}: {name} is not finite"))
                ok = False
        if not ok:
            prev_end = None
            continue
        a, b = float(s["start_s"]), float(s["end_s"])
        if not a < b:
            errs.append(("E_SPAN", f"{w}: start {a} is not before end {b}"))
        elif b - a < float(rules["min_span_s"]) - tol:
            errs.append(("E_SHORT", f"{w}: shorter than {rules['min_span_s']} s"))
        if a < -tol or b > dur + tol:
            errs.append(("E_BOUNDS", f"{w}: [{a}, {b}] outside 0..{dur} s"))
        if prev_end is not None and a < prev_end - tol:
            errs.append(("E_ORDER", f"{w}: starts at {a} s before the previous syllable ends ({prev_end} s)"))
        prev_end = b
    return errs


def validate_response(obj: Any, items: Mapping[str, Mapping], batches: Mapping[str, set], rules: Mapping[str, Any],
                      real_ids: Iterable[str]) -> dict:
    """A whole answer file; same result structure as tools/inspect_item.py: {"file_errors", "items": [{"position",
    "item_id", "errors", "spans"}], "batch", "worker_id"}. An item is accepted iff neither it nor the file has errors."""
    out: dict = {"file_errors": [], "items": [], "batch": None, "worker_id": None}
    fe = out["file_errors"]
    if not isinstance(obj, dict):
        fe.append(("E_TOP", "answer file is not a JSON object"))
        return out
    fe += _key_errors(obj, RESPONSE_KEYS, RESPONSE_OPTIONAL, "E_TOP", "answer file")
    if "schema" in obj and obj["schema"] != rules["schema"]:
        fe.append(("E_SCHEMA", f"schema is not {rules['schema']!r}"))
    if "worker_id" in obj:
        wid = obj["worker_id"]
        if isinstance(wid, str) and re.fullmatch(rules["worker_id_pattern"], wid):
            out["worker_id"] = wid
        else:
            fe.append(("E_WORKER", "bad worker_id"))
    allowed = None
    if "batch" in obj:
        b = obj["batch"]
        if b == rules["extra_batch_id"]:
            allowed, out["batch"] = set(real_ids), b
        elif isinstance(b, str) and b in batches:
            allowed, out["batch"] = set(batches[b]), b
        else:
            fe.append(("E_BATCH", f"unknown batch {b!r}"))
    if "notes" in obj and not isinstance(obj["notes"], str):
        fe.append(("E_NOTES", "notes is not a string"))
    entries = obj.get("items", [])
    if not isinstance(entries, list):
        fe.append(("E_ITEMS", "items is not a list"))
        entries = []
    elif "items" in obj and not entries:
        fe.append(("E_ITEMS", "items is empty"))
    ids = [e.get("item_id") if isinstance(e, dict) and isinstance(e.get("item_id"), str) else None for e in entries]
    counts = Counter(i for i in ids if i is not None)
    for p, (e, iid) in enumerate(zip(entries, ids)):
        meta = items.get(iid) if iid is not None else None
        errs = validate_item_answer(e, meta, rules, where=f"items[{p}]")
        if meta is not None and allowed is not None and iid not in allowed:
            errs.append(("E_NOT_IN_BATCH", f"item {iid} is not in batch {out['batch']!r}"))
        if iid is not None and counts[iid] > 1:
            errs.append(("E_DUP_ITEM", f"item {iid} appears {counts[iid]} times"))
        spans = [[float(s["start_s"]), float(s["end_s"])] for s in e["syllables"]] if not errs else None
        out["items"].append({"position": p, "item_id": iid, "errors": errs, "spans": spans})
    return out


def path_errors(worker_dir: str, file_name: str, obj: Any, rules: Mapping[str, Any], practice_batches) -> list:
    """answers/<worker_id>/<batch_id>.json or extra*.json: the folder names the worker, the file names the batch
    (same rule as tools/inspect_item.path_errors for a file directly in a worker folder)."""
    if not isinstance(obj, dict):
        return []
    errs = []
    if obj.get("worker_id") != worker_dir:
        errs.append(("E_WORKER_DIR", f"worker_id {obj.get('worker_id')!r} differs from the folder {worker_dir}"))
    stem, batch = Path(file_name).stem, obj.get("batch")
    if re.fullmatch(rules["batch_file_pattern"], stem):
        if batch != stem:
            errs.append(("E_FILENAME", f"{file_name} holds batch {batch!r}"))
    elif re.fullmatch(rules["extra_file_pattern"], stem):
        if batch != rules["extra_batch_id"]:
            errs.append(("E_FILENAME", f"{file_name} must hold batch {rules['extra_batch_id']!r}"))
    elif batch not in practice_batches:
        errs.append(("E_FILENAME", f"{file_name}: answer files are named <batch_id>.json or extra.json"))
    return errs


def batch_sets(manifest: Mapping[str, Any]) -> dict:
    out = {b: set(v["items"]) for b, v in manifest["batches"].items()}
    for b, v in manifest["practice_batches"].items():
        out[b] = {i for r in v["rounds"] for i in r}
    return out


def ingest(files: Sequence[tuple], items: Mapping[str, Mapping], manifest: Mapping[str, Any], rules: Mapping[str, Any]) -> dict:
    """Validate answer files. ``files``: (path, worker folder name, parsed object or None, read error or None) of
    every answers/<worker>/*.json. Practice files are skipped. Returns {"accepted": [{"item_id", "worker_id", "batch",
    "file", "spans"}], "rejected": [{"file", "item_id", "errors"}], "skipped": [...]}. If one worker answered an item
    in two places, every copy is rejected (``E_DUP_WORKER``); nothing is ever fixed or chosen silently."""
    acc, rej, skipped = [], [], []
    batches = batch_sets(manifest)
    for path, wdir, obj, err in files:
        if err is not None:
            rej.append({"file": str(path), "item_id": None, "errors": [("E_JSON", err)]})
            continue
        if isinstance(obj, dict) and obj.get("batch") in manifest["practice_batches"]:
            skipped.append({"file": str(path), "reason": "practice answers"})
            continue
        res = validate_response(obj, items, batches, rules, manifest["items"])
        res["file_errors"] += path_errors(wdir, Path(path).name, obj, rules, manifest["practice_batches"])
        if res["file_errors"]:
            rej.append({"file": str(path), "item_id": None, "errors": res["file_errors"]})
            for r in res["items"]:
                rej.append({"file": str(path), "item_id": r["item_id"],
                            "errors": r["errors"] + [("E_FILE", "the file was rejected")]})
            continue
        for r in res["items"]:
            if r["errors"]:
                rej.append({"file": str(path), "item_id": r["item_id"], "errors": r["errors"]})
            else:
                acc.append({"item_id": r["item_id"], "worker_id": res["worker_id"], "batch": res["batch"],
                            "file": str(path), "spans": r["spans"]})
    n = Counter((a["item_id"], a["worker_id"]) for a in acc)
    dup = {k for k, c in n.items() if c > 1}
    keep = []
    for a in acc:
        if (a["item_id"], a["worker_id"]) in dup:
            rej.append({"file": a["file"], "item_id": a["item_id"],
                        "errors": [("E_DUP_WORKER", f"worker {a['worker_id']} answered item {a['item_id']} "
                                                    f"{n[(a['item_id'], a['worker_id'])]} times")]})
        else:
            keep.append(a)
    return {"accepted": keep, "rejected": rej, "skipped": skipped}


# --------------------------------------------------------------------------- aggregation + agreement


def aggregate(accepted: Sequence[Mapping], items: Mapping[str, Mapping], cfg: Mapping[str, Any]) -> dict:
    """{item: {"spans", "n_answers", "workers"}}: per syllable the median start and the median end over the item's
    answers (``aggregate.method``). Medians keep every pointwise order of the answers, so the result is valid; it is
    checked with the same rules anyway."""
    s = settings(cfg)
    if s["aggregate"]["method"] != "median":
        raise ValueError("aggregate.method must be median")
    rules = response_rules(cfg)
    by: dict = {}
    for a in accepted:
        by.setdefault(a["item_id"], []).append(a)
    out = {}
    for iid in sorted(by):
        ans = sorted(by[iid], key=lambda a: a["worker_id"])
        if len(ans) < int(s["aggregate"]["min_answers"]):
            continue
        med = np.median(np.asarray([a["spans"] for a in ans], dtype=np.float64), axis=0)
        spans = [[round(float(x), 6) for x in p] for p in med]
        entry = {"item_id": iid, "syllables": [{"index": k, "label": items[iid]["labels"][k], "start_s": a, "end_s": b}
                                               for k, (a, b) in enumerate(spans)]}
        errs = validate_item_answer(entry, items[iid], rules)
        if errs:
            raise AssertionError(f"median answer of {iid} breaks the rules: {errs}")
        out[iid] = {"spans": spans, "n_answers": len(ans), "workers": [a["worker_id"] for a in ans]}
    return out


def item_agreement(spans_list: Sequence, tol_ms: float) -> dict:
    """Pairwise agreement of an item's answers (same definitions as tools/agreement.py)."""
    n = len(spans_list)
    out = {"n_answers": n, "n_pairs": 0, "median_abs_start_diff_ms": None, "start_within_tol": None,
           "median_abs_end_diff_ms": None, "end_within_tol": None}
    if n < 2:
        return out
    a = np.asarray(spans_list, dtype=np.float64)
    i, j = np.triu_indices(n, k=1)
    ds = 1000.0 * np.abs(a[i, :, 0] - a[j, :, 0]).ravel()
    de = 1000.0 * np.abs(a[i, :, 1] - a[j, :, 1]).ravel()
    out.update({"n_pairs": int(i.size), "median_abs_start_diff_ms": float(np.median(ds)),
                "start_within_tol": float(np.mean(ds <= tol_ms + 1e-9)),
                "median_abs_end_diff_ms": float(np.median(de)), "end_within_tol": float(np.mean(de <= tol_ms + 1e-9))})
    return out


def agreement(accepted: Sequence[Mapping], cfg: Mapping[str, Any]) -> dict:
    """Inter-annotator agreement: per item, pooled, flagged items, and each worker's median |start - median of the
    other answers| (items with >= 2 other answers)."""
    ag = settings(cfg)["agreement"]
    tol, thr = float(ag["tolerance_ms"]), float(ag["flag_median_abs_start_ms"])
    by: dict = {}
    for a in accepted:
        by.setdefault(a["item_id"], {})[a["worker_id"]] = a["spans"]
    per_item, pooled, dev = {}, [], {}
    for iid in sorted(by):
        ws = sorted(by[iid])
        sp = [by[iid][w] for w in ws]
        r = item_agreement(sp, tol)
        r["workers"] = ws
        r["flagged"] = bool(r["n_answers"] >= int(ag["min_answers"]) and r["median_abs_start_diff_ms"] > thr)
        per_item[iid] = r
        if len(sp) >= 2:
            x = np.asarray(sp, dtype=np.float64)
            i, j = np.triu_indices(len(sp), k=1)
            pooled.append(1000.0 * np.abs(x[i, :, 0] - x[j, :, 0]).ravel())
        if len(sp) >= 3:
            x = np.asarray(sp, dtype=np.float64)
            for n, w in enumerate(ws):
                others = np.median(np.delete(x, n, axis=0)[:, :, 0], axis=0)
                dev.setdefault(w, []).append(1000.0 * np.abs(x[n, :, 0] - others))
    d = np.concatenate(pooled) if pooled else np.zeros(0)
    return {
        "tolerance_ms": tol, "flag_threshold_ms": thr,
        "pooled": {"n_start_pairs": int(d.size), "median_abs_start_diff_ms": float(np.median(d)) if d.size else None,
                   "start_within_tol": float(np.mean(d <= tol + 1e-9)) if d.size else None},
        "n_items_with_2plus_answers": int(sum(r["n_answers"] >= 2 for r in per_item.values())),
        "flagged": [i for i, r in per_item.items() if r["flagged"]],
        "per_worker_median_abs_start_vs_others_ms": {w: float(np.median(np.concatenate(v))) for w, v in sorted(dev.items())},
        "per_item": per_item,
    }


# --------------------------------------------------------------------------------------- adoption


def adoption(manual: Mapping[str, float], baseline: Mapping[str, float], coverage: Mapping[str, float],
             rule: Mapping[str, Any]) -> dict:
    """The pre-registered rule: adopt manual iff mean over the primary cells of its mean target r exceeds the
    baseline's (same trials) by >= min_improvement and every primary cell is covered >= min_coverage."""
    cells = list(rule["primary_cells"])
    missing = [c for c in cells if manual.get(c) is None or baseline.get(c) is None]
    out = {"rule": dict(rule), "manual_mean": None, "baseline_mean": None, "improvement": None, "adopted": False}
    if missing:
        out["reason"] = f"no result for {missing}"
        return out
    m, b = float(np.mean([manual[c] for c in cells])), float(np.mean([baseline[c] for c in cells]))
    low = [c for c in cells if coverage.get(c, 0.0) < float(rule["min_coverage"])]
    out.update({"manual_mean": m, "baseline_mean": b, "improvement": m - b,
                "coverage": {c: coverage.get(c) for c in cells}})
    if low:
        out["reason"] = f"coverage below {rule['min_coverage']} in {low}"
    elif m - b >= float(rule["min_improvement"]) - 1e-12:
        out.update({"adopted": True, "reason": f"improvement {m - b:+.4f} >= {rule['min_improvement']}"})
    else:
        out["reason"] = f"improvement {m - b:+.4f} < {rule['min_improvement']}"
    return out


# ---------------------------------------------------------------------------------- blindness scan


def sha256_file(path: PathLike) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def file_hashes(pkg: PathLike) -> dict:
    """{relative path: SHA-256} of every regular file except files.sha256."""
    pkg = Path(pkg)
    return {p.relative_to(pkg).as_posix(): sha256_file(p) for p in sorted(pkg.rglob("*"))
            if p.is_file() and not p.is_symlink() and p.name != "files.sha256"}


def write_hash_list(pkg: PathLike) -> Path:
    p = Path(pkg) / "files.sha256"
    p.write_text("".join(f"{h}  {f}\n" for f, h in file_hashes(pkg).items()))
    return p


def read_hash_list(pkg: PathLike) -> dict:
    out = {}
    for line in (Path(pkg) / "files.sha256").read_text().splitlines():
        h, f = line.split("  ", 1)
        out[f] = h
    return out


def allowed_file_patterns(cfg: Mapping[str, Any]) -> list:
    i = settings(cfg)["ids"]
    hx = f"[0-9a-f]{{{int(i['hex_chars'])}}}"
    px = f"{re.escape(i['practice_prefix'])}[0-9a-f]{{{int(i['hex_chars'])}}}"
    ex = f"{re.escape(i['example_prefix'])}[0-9]+"
    return [re.escape(f) for f in ROOT_FILES] + [
        f"items/{hx}\\.(png|json|npz)", f"practice/items/{px}\\.(png|json|npz)", f"practice/truth/{px}\\.json",
        f"examples/{ex}(\\.png|\\.json|\\.npz|_answer\\.json|_overlay\\.png)",
        "tools/(" + "|".join(re.escape(t) for t in TOOL_FILES) + ")", "tools/requirements\\.txt"]


def _json_walk(x, keys: list, strings: list, numbers: list, key: Optional[str] = None) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            keys.append(str(k))
            _json_walk(v, keys, strings, numbers, str(k))
    elif isinstance(x, list):
        for v in x:
            _json_walk(v, keys, strings, numbers, key)
    elif isinstance(x, str):
        strings.append((key, x))
    elif _is_number(x):
        numbers.append(float(x))


def contains_sequence(values: np.ndarray, seq: np.ndarray, tol: float) -> bool:
    """Every value of ``seq`` occurs in ``values`` (within tol), in this order (not necessarily adjacent)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    pos = 0
    for s in np.asarray(seq, dtype=np.float64).ravel():
        hits = np.flatnonzero(np.abs(v[pos:] - s) <= tol)
        if not hits.size:
            return False
        pos += int(hits[0]) + 1
    return True


def _png_text_chunks(path: Path) -> list:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return ["not a PNG"]
    out, i = [], 8
    while i + 8 <= len(data):
        n = int.from_bytes(data[i:i + 4], "big")
        t = data[i + 4:i + 8].decode("latin-1")
        if t in ("tEXt", "zTXt", "iTXt", "eXIf"):
            out.append(t)
        i += 12 + n
    return out


def scan_package(pkg: PathLike, cfg: Mapping[str, Any], truth: Optional[Mapping[str, np.ndarray]] = None,
                 secrets: Sequence[str] = ()) -> list:
    """Blindness scan of a package folder; returns the problems (empty = pass).

    Layout: only the allowlisted files and folders, no links; ids in items/ = the manifest's, practice truth only for
    practice items; files.sha256 lists every file with its hash. Names: no JSON key matches a forbidden pattern; npz
    files hold exactly the allowed arrays (axes recomputable from the item); PNGs carry no text chunks. Text: no
    forbidden pattern in any text (the cue-derived fields of item files exempt), no secret (the salt). Values (with
    ``truth`` = {real item id: (J, 2) true frames}, on the host only): no JSON number list and no array holds the
    boundary sequence of any blind item, in seconds or in frames, and no per-frame trace of an item equals its
    true speech mask.
    """
    pkg = Path(pkg)
    s = settings(cfg)
    b = s["blindness"]
    probs: list = []
    key_rx = [re.compile(p, re.I) for p in b["forbidden_key_patterns"]]
    text_rx = [re.compile(p, re.I) for p in b["forbidden_text_patterns"]]
    cue_fields, tol = set(b["cue_fields"]), float(b["value_tolerance"])
    pats = [re.compile(p) for p in allowed_file_patterns(cfg)]
    try:
        man = json.loads((pkg / "manifest.json").read_text())
        real, prac, ex = set(man["items"]), set(man["practice_items"]), set(man["examples"])
    except (OSError, ValueError, KeyError) as e:
        return [f"manifest.json unreadable: {e}"]
    files = []
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(pkg).as_posix()
        if p.is_symlink():
            probs.append(f"{rel}: symbolic link")
        elif p.is_dir():
            if rel not in PACKAGE_DIRS:
                probs.append(f"{rel}/: unexpected folder")
        elif not p.is_file():
            probs.append(f"{rel}: not a regular file")
        elif not any(rx.fullmatch(rel) for rx in pats):
            probs.append(f"{rel}: unexpected file")
        else:
            files.append(p)
    answers = pkg / "answers"
    if not answers.is_dir() or any(answers.iterdir()):
        probs.append("answers/ must exist and be empty in a fresh package")

    def ids_in(d, ext):
        return {p.stem for p in (pkg / d).glob(f"*{ext}")} if (pkg / d).is_dir() else set()

    for d, want in (("items", real), ("practice/items", prac)):
        for ext in (".json", ".npz", ".png"):
            if ids_in(d, ext) != want:
                probs.append(f"{d}/*{ext} do not match the manifest's ids")
    if ids_in("practice/truth", ".json") != prac:
        probs.append("practice/truth/ must hold exactly the practice items")
    if ids_in("examples", ".npz") != ex:
        probs.append("examples/ do not match the manifest")
    if real & (prac | ex) or prac & ex:
        probs.append("real, practice and example ids overlap")

    # hashes
    if (pkg / "files.sha256").is_file():
        listed, actual = read_hash_list(pkg), file_hashes(pkg)
        if listed != actual:
            diff = sorted(set(listed) ^ set(actual)) + sorted(f for f in set(listed) & set(actual) if listed[f] != actual[f])
            probs.append(f"files.sha256 does not match the files: {diff[:5]}")
    else:
        probs.append("files.sha256 missing")

    seqs = []
    for iid, segs in (truth or {}).items():
        f = np.asarray(segs, dtype=np.float64).reshape(-1, 2)
        if f.size >= int(b["min_truth_values"]):
            seqs.append((iid, f.ravel(), f.ravel() * frame_s(cfg)))

    def value_leak(rel, values):
        if not seqs or not np.asarray(values).size:
            return
        u = np.unique(np.round(np.asarray(values, dtype=np.float64).ravel() / tol))
        for iid, fr, sec in seqs:
            for seq in (fr, sec):
                if np.all(np.isin(np.round(seq / tol), u)) and contains_sequence(values, seq, tol):
                    probs.append(f"{rel}: holds the true boundaries of blind item {iid}")

    for p in files:
        rel = p.relative_to(pkg).as_posix()
        if p.suffix in (".md", ".py", ".txt", ".sha256", ".json"):
            text = p.read_text(encoding="utf-8")
            for sec in secrets:
                if sec and sec in text:
                    probs.append(f"{rel}: contains a secret of the key")
            if p.suffix == ".json":
                try:
                    obj = json.loads(text)
                except ValueError:
                    probs.append(f"{rel}: invalid JSON")
                    continue
                keys, strings, numbers = [], [], []
                _json_walk(obj, keys, strings, numbers)
                for k in set(keys):
                    if any(rx.search(k) for rx in key_rx):
                        probs.append(f"{rel}: forbidden key {k!r}")
                for k, v in strings:
                    if k in cue_fields and rel.startswith(("items/", "practice/items/", "examples/")):
                        continue
                    for rx in text_rx:
                        if rx.search(v):
                            probs.append(f"{rel}: forbidden text {rx.pattern!r} in {k!r}")
                if rel.startswith(("items/", "practice/items/", "examples/")) and not rel.endswith("_answer.json"):
                    extra = set(obj) - ITEM_FILE_KEYS
                    if extra:
                        probs.append(f"{rel}: unexpected keys {sorted(extra)}")
                value_leak(rel, numbers)
            else:
                for rx in text_rx:
                    if rx.search(text):
                        probs.append(f"{rel}: forbidden text {rx.pattern!r}")
        elif p.suffix == ".npz":
            try:
                with np.load(p, allow_pickle=False) as z:
                    arrs = {k: z[k] for k in z.files}
            except (OSError, ValueError) as e:
                probs.append(f"{rel}: unreadable ({e})")
                continue
            if not set(arrs) <= set(ARRAY_NAMES) or not {"t_s", "mel_db", "mel_hz", "loudness_db", "duration_s"} <= set(arrs):
                probs.append(f"{rel}: arrays {sorted(arrs)}, allowed {list(ARRAY_NAMES)}")
                continue
            T = arrs["t_s"].size
            if not np.array_equal(arrs["t_s"], time_axis(T, cfg)) or float(arrs["duration_s"]) != round(T * frame_s(cfg), 6):
                probs.append(f"{rel}: time axis or duration is not the standard one")
            for k, v in arrs.items():
                if v.dtype.kind not in "f":
                    probs.append(f"{rel}: array {k} has dtype {v.dtype}")
                value_leak(f"{rel}:{k}", v)
            iid = p.stem
            if truth and iid in truth:
                mask = np.zeros(T, dtype=bool)
                for a, c in np.asarray(truth[iid], dtype=np.int64).reshape(-1, 2):
                    mask[a:c] = True
                for k in ("loudness_db", "neural"):
                    if k in arrs and arrs[k].shape == (T,) and np.array_equal(arrs[k] > arrs[k].min(), mask):
                        probs.append(f"{rel}:{k} equals the true speech mask of {iid}")
        elif p.suffix == ".png":
            chunks = _png_text_chunks(p)
            if chunks:
                probs.append(f"{rel}: PNG metadata chunks {chunks}")
    return probs


# ------------------------------------------------------------------------------------------ delivery


def tar_member_problem(name: str, kind: str, package: str) -> Optional[str]:
    """Why a member of the delivered archive may not land on the Mac (None = fine): only regular files and folders
    inside the package folder; never the key, answers or work folders of the host."""
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        return "unsafe path"
    if not p.parts or p.parts[0] != package:
        return "outside the package folder"
    if kind not in ("file", "dir"):
        return f"{kind} not allowed (regular files and folders only)"
    if re.search(r"_key\.json$|_answers(/|$)|_work(/|$)", name):
        return "a key, answers or work file"
    return None
