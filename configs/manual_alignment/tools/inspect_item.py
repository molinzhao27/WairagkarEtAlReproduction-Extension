#!/usr/bin/env python3
"""Look at one item of this syllable-timing package, measure it, and check an answer before you submit it.

Needs Python >= 3.9 with numpy and matplotlib only (tools/requirements.txt). Run it from the package folder:

  python tools/inspect_item.py render  ITEM [--start S --end E] [--out FILE.png] [--width-px N]
  python tools/inspect_item.py overlay ITEM DRAFT.json [--start S --end E] [--out FILE.png] [--width-px N]
  python tools/inspect_item.py values  ITEM --start S --end E [--step 0.01] [--bands]
  python tools/inspect_item.py peaks   ITEM [--start S --end E] [--min-prominence DB]
  python tools/inspect_item.py check   DRAFT.json

ITEM is an item id from items/, practice/items/ or examples/. Every time is in seconds on the item's time axis
(the axis of the images). Output is deterministic: the same command gives the same text and the same image.

  render   the three panels (spectrogram, loudness, neural activity) of the whole item, or of [S, E] zoomed in;
           the grid gets finer as you zoom (down to 10 ms).
  overlay  the same view with the syllable spans of a draft drawn and labelled, to check and adjust your answer.
           DRAFT.json is one item answer {"item_id": ..., "syllables": [...]} or a whole answer file.
  values   a table of time, loudness (dB) and neural activity (z) every STEP seconds; --bands adds the
           spectral energy (dB) in a few frequency bands.
  peaks    the loudness peaks and the dips between them, with times, heights and dip depths. Report only.
  check    validates a draft (one item answer or a whole answer file) against the answer format: syllable count,
           labels, order, no overlap, inside the item. Exit code 0 = valid.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = TOOLS_DIR.parent

ARRAY_NAMES = ("t_s", "mel_db", "mel_hz", "loudness_db", "neural", "duration_s")

# The answer format (instructions.md). The same rules as the host-side ingestion (wvx.manual_alignment).
RESPONSE_KEYS = ("schema", "batch", "worker_id", "items")
RESPONSE_OPTIONAL = ("notes",)
ITEM_KEYS = ("item_id", "syllables")
ITEM_OPTIONAL = ("confidence", "notes", "unassigned_bursts")
SYLLABLE_KEYS = ("index", "label", "start_s", "end_s")
SYLLABLE_OPTIONAL = ("confidence",)


# ------------------------------------------------------------------------------------------ package


class PackageError(Exception):
    pass


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_settings(root=DEFAULT_ROOT) -> dict:
    p = Path(root) / "settings.json"
    if not p.is_file():
        raise PackageError(f"{p} not found: run the tools from inside the package")
    return load_json(p)


def load_manifest(root=DEFAULT_ROOT) -> dict:
    p = Path(root) / "manifest.json"
    if not p.is_file():
        raise PackageError(f"{p} not found: run the tools from inside the package")
    return load_json(p)


def item_dirs(root=DEFAULT_ROOT):
    root = Path(root)
    return [root / "items", root / "practice" / "items", root / "examples"]


def find_item(item_id: str, root=DEFAULT_ROOT) -> Path:
    """Path of ITEM's JSON file (items/, practice/items/ or examples/)."""
    if not isinstance(item_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+", item_id):
        raise PackageError(f"not an item id: {item_id!r}")
    for d in item_dirs(root):
        p = d / f"{item_id}.json"
        if p.is_file():
            return p
    raise PackageError(f"no item {item_id!r} in items/, practice/items/ or examples/")


def load_item(item_id: str, root=DEFAULT_ROOT) -> tuple:
    """(item JSON, arrays dict) of one item."""
    p = find_item(item_id, root)
    meta = load_json(p)
    with np.load(p.with_suffix(".npz"), allow_pickle=False) as z:
        arr = {k: np.asarray(z[k]) for k in z.files}
    missing = [k for k in ARRAY_NAMES if k not in arr and k != "neural"]
    if missing:
        raise PackageError(f"{p.with_suffix('.npz').name} lacks {missing}")
    arr["t_s"] = arr["t_s"].astype(np.float64)
    for k in ("mel_db", "mel_hz", "loudness_db", "neural"):
        if k in arr:
            arr[k] = arr[k].astype(np.float64)
    arr["duration_s"] = float(arr["duration_s"])
    return meta, arr


def item_rule_meta(meta: dict) -> dict:
    """What the answer rules need from an item file."""
    return {"n_syllables": int(meta["n_syllables"]), "labels": [s["label"] for s in meta["syllables"]],
            "duration_s": float(meta["duration_s"]), "kind": meta["kind"]}


def all_items(root=DEFAULT_ROOT) -> dict:
    """{item id: rule metadata} of every item, practice item and example in the package."""
    out = {}
    for d in item_dirs(root):
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.with_suffix(".npz").is_file():
                    out[p.stem] = item_rule_meta(load_json(p))
    return out


def batch_sets(manifest: dict) -> dict:
    """{batch id: set of item ids} for real and practice batches."""
    out = {b: set(v["items"]) for b, v in manifest["batches"].items()}
    for b, v in manifest["practice_batches"].items():
        out[b] = {i for r in v["rounds"] for i in r}
    return out


def response_rules(settings: dict) -> dict:
    r = dict(settings["response"])
    r["schema"] = settings["schemas"]["response"]
    return r


# --------------------------------------------------------------------------------------- validation


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _keys(obj: dict, required, optional, code: str, where: str) -> list:
    errs = []
    missing = [k for k in required if k not in obj]
    unknown = sorted(str(k) for k in obj if k not in required and k not in optional)
    if missing:
        errs.append((code, f"{where}: missing {', '.join(missing)}"))
    if unknown:
        errs.append((code, f"{where}: unknown key(s) {', '.join(unknown)} (allowed: {', '.join(tuple(required) + tuple(optional))})"))
    return errs


def validate_item_answer(entry, meta, rules: dict, where: str = "item") -> list:
    """[(code, message)] for one item answer; ``meta`` = item_rule_meta of the item (None: unknown item)."""
    tol = float(rules["tolerance_s"])
    conf = list(rules["confidence_values"])
    if not isinstance(entry, dict):
        return [("E_ITEM_TYPE", f"{where}: must be an object with item_id and syllables")]
    errs = _keys(entry, ITEM_KEYS, ITEM_OPTIONAL, "E_ITEM_KEYS", where)
    iid = entry.get("item_id")
    if not isinstance(iid, str):
        return errs + [("E_ITEM_ID", f"{where}: item_id must be a string")]
    where = f"item {iid}"
    if meta is None:
        return errs + [("E_ITEM_ID", f"{where}: no such item in this package")]
    dur = float(meta["duration_s"])
    if "confidence" in entry and entry["confidence"] not in conf:
        errs.append(("E_CONFIDENCE", f"{where}: confidence must be one of {conf}"))
    if "notes" in entry and not isinstance(entry["notes"], str):
        errs.append(("E_NOTES", f"{where}: notes must be a string"))
    if "unassigned_bursts" in entry:
        ub = entry["unassigned_bursts"]
        good = isinstance(ub, list) and all(
            isinstance(p, list) and len(p) == 2 and all(_is_number(v) and math.isfinite(v) for v in p)
            and -tol <= p[0] < p[1] <= dur + tol for p in ub)
        if not good:
            errs.append(("E_UNASSIGNED", f"{where}: unassigned_bursts must be a list of [start_s, end_s] pairs inside the item"))
    if "syllables" not in entry:
        return errs
    syl = entry["syllables"]
    if not isinstance(syl, list):
        return errs + [("E_SYLLABLES", f"{where}: syllables must be a list")]
    n = int(meta["n_syllables"])
    if len(syl) != n:
        return errs + [("E_COUNT", f"{where}: {len(syl)} syllables given, this item has exactly {n}")]
    labels, min_span = meta["labels"], float(rules["min_span_s"])
    prev_end = None
    for k, s in enumerate(syl):
        w = f"{where} syllable {k}"
        if not isinstance(s, dict):
            errs.append(("E_SYL_TYPE", f"{w}: must be an object"))
            prev_end = None
            continue
        errs += _keys(s, SYLLABLE_KEYS, SYLLABLE_OPTIONAL, "E_SYL_KEYS", w)
        if "index" in s and (not isinstance(s["index"], int) or isinstance(s["index"], bool) or s["index"] != k):
            errs.append(("E_INDEX", f"{w}: index must be {k} (syllables are listed in order)"))
        if "label" in s and s["label"] != labels[k]:
            errs.append(("E_LABEL", f"{w}: label must be {labels[k]!r}, got {s['label']!r}"))
        if "confidence" in s and s["confidence"] not in conf:
            errs.append(("E_CONFIDENCE", f"{w}: confidence must be one of {conf}"))
        ok = True
        for name in ("start_s", "end_s"):
            if name not in s:
                ok = False
            elif not _is_number(s[name]):
                errs.append(("E_NUMBER", f"{w}: {name} must be a number of seconds, got {s[name]!r}"))
                ok = False
            elif not math.isfinite(s[name]):
                errs.append(("E_FINITE", f"{w}: {name} must be finite"))
                ok = False
        if not ok:
            prev_end = None
            continue
        a, b = float(s["start_s"]), float(s["end_s"])
        if not a < b:
            errs.append(("E_SPAN", f"{w}: start_s {a} must be before end_s {b}"))
        elif b - a < min_span - tol:
            errs.append(("E_SHORT", f"{w}: {b - a:.4f} s long, the minimum is {min_span} s"))
        if a < -tol or b > dur + tol:
            errs.append(("E_BOUNDS", f"{w}: [{a}, {b}] must lie inside the item (0 to {dur} s)"))
        if prev_end is not None and a < prev_end - tol:
            errs.append(("E_ORDER", f"{w}: starts at {a} s, before syllable {k - 1} ends ({prev_end} s); "
                                    "syllables must be in order and must not overlap"))
        prev_end = b
    return errs


def answer_spans(entry: dict) -> list:
    return [[float(s["start_s"]), float(s["end_s"])] for s in entry["syllables"]]


def validate_response(obj, items: dict, batches: dict, rules: dict, real_ids) -> dict:
    """Check a whole answer file. ``items``: {id: item_rule_meta}; ``batches``: {batch id: set of ids};
    ``real_ids``: the items an ``extra`` answer may cover. Returns {"file_errors", "items": [{"position", "item_id",
    "errors", "spans"}], "batch", "worker_id"}; an item is accepted iff the file and the item have no error."""
    out = {"file_errors": [], "items": [], "batch": None, "worker_id": None}
    fe = out["file_errors"]
    if not isinstance(obj, dict):
        fe.append(("E_TOP", "an answer file must be a JSON object with schema, batch, worker_id and items"))
        return out
    fe += _keys(obj, RESPONSE_KEYS, RESPONSE_OPTIONAL, "E_TOP", "answer file")
    if "schema" in obj and obj["schema"] != rules["schema"]:
        fe.append(("E_SCHEMA", f"schema must be {rules['schema']!r}"))
    wid = obj.get("worker_id")
    if "worker_id" in obj:
        if isinstance(wid, str) and re.fullmatch(rules["worker_id_pattern"], wid):
            out["worker_id"] = wid
        else:
            fe.append(("E_WORKER", f"worker_id must match {rules['worker_id_pattern']}"))
    allowed = None
    if "batch" in obj:
        b = obj["batch"]
        if b == rules["extra_batch_id"]:
            allowed, out["batch"] = set(real_ids), b
        elif isinstance(b, str) and b in batches:
            allowed, out["batch"] = batches[b], b
        else:
            fe.append(("E_BATCH", f"unknown batch {b!r}"))
    if "notes" in obj and not isinstance(obj["notes"], str):
        fe.append(("E_NOTES", "notes must be a string"))
    entries = obj.get("items", [])
    if not isinstance(entries, list):
        fe.append(("E_ITEMS", "items must be a list"))
        entries = []
    elif "items" in obj and not entries:
        fe.append(("E_ITEMS", "items is empty"))
    ids = [e.get("item_id") if isinstance(e, dict) and isinstance(e.get("item_id"), str) else None for e in entries]
    for p, (e, iid) in enumerate(zip(entries, ids)):
        meta = items.get(iid) if iid is not None else None
        errs = validate_item_answer(e, meta, rules, where=f"items[{p}]")
        if meta is not None and allowed is not None and iid not in allowed:
            errs.append(("E_NOT_IN_BATCH", f"item {iid}: not part of batch {out['batch']!r}"))
        if iid is not None and ids.count(iid) > 1:
            errs.append(("E_DUP_ITEM", f"item {iid}: answered {ids.count(iid)} times in this file (every copy is rejected)"))
        out["items"].append({"position": p, "item_id": iid, "errors": errs,
                             "spans": answer_spans(e) if not errs else None})
    return out


def accepted(result: dict) -> list:
    """Item results of a validated file that are accepted (no file error, no item error)."""
    return [] if result["file_errors"] else [r for r in result["items"] if not r["errors"]]


def path_errors(path, root, obj, rules: dict, practice_batches) -> list:
    """Where an answer file lives: answers/<worker_id>/<batch_id>.json (or extra*.json), and practice answers in
    answers/<worker_id>/practice/. Files elsewhere get no path rules."""
    try:
        rel = Path(path).resolve().relative_to(Path(root).resolve() / "answers")
    except ValueError:
        return []
    parts = rel.parts
    if len(parts) < 2 or not isinstance(obj, dict):
        return []
    errs = []
    if obj.get("worker_id") != parts[0]:
        errs.append(("E_WORKER_DIR", f"worker_id {obj.get('worker_id')!r} differs from the folder answers/{parts[0]}/"))
    batch = obj.get("batch")
    if len(parts) == 2:
        stem = Path(parts[1]).stem
        if re.fullmatch(rules["batch_file_pattern"], stem):
            if batch != stem:
                errs.append(("E_FILENAME", f"file {parts[1]} must hold batch {stem!r}, not {batch!r}"))
        elif re.fullmatch(rules["extra_file_pattern"], stem):
            if batch != rules["extra_batch_id"]:
                errs.append(("E_FILENAME", f"file {parts[1]} must hold batch {rules['extra_batch_id']!r}"))
        elif batch not in practice_batches:
            errs.append(("E_FILENAME", f"answer files are named <batch_id>.json or extra.json, not {parts[1]}"))
    elif len(parts) == 3 and parts[1] == "practice" and batch not in practice_batches:
        errs.append(("E_FILENAME", f"answers/{parts[0]}/practice/ holds practice answers only"))
    return errs


def read_draft(path) -> tuple:
    """(object, None) or (None, error) for a JSON draft file."""
    try:
        with open(path) as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return None, f"cannot read {path} as JSON: {e}"


def draft_items(obj) -> list:
    """Item answers in a draft: a whole answer file (its items) or one item answer."""
    if isinstance(obj, dict) and "items" in obj:
        return [e for e in obj["items"] if isinstance(e, dict)] if isinstance(obj["items"], list) else []
    return [obj] if isinstance(obj, dict) else []


# ------------------------------------------------------------------------------------------ drawing


def tick_steps(px_per_s: float, display: dict) -> tuple:
    """(minor, major) grid steps in seconds: the finest minor step with >= min_minor_px between lines."""
    steps = [tuple(map(float, s)) for s in display["tick_steps_s"]]
    for minor, major in steps:
        if minor * px_per_s >= float(display["min_minor_px"]):
            return minor, major
    return steps[-1]


def _decimals(step: float) -> int:
    return max(0, -int(math.floor(math.log10(step) + 1e-9)))


def overview_rows(duration: float, display: dict) -> list:
    """[(lo, hi)] of the overview rows: row r covers [r * S, (r + 1) * S) (S = seconds_per_row) plus row_overlap_s of
    its neighbours; the last row ends at the next whole second after the end of the item (same scale, narrower)."""
    S, ov = float(display["seconds_per_row"]), float(display["row_overlap_s"])
    n = max(1, int(math.ceil(duration / S - 1e-9)))
    rows = [(r * S - (ov if r else 0.0), (r + 1) * S + (ov if r < n - 1 else 0.0)) for r in range(n)]
    rows[-1] = (rows[-1][0], min(rows[-1][1], float(math.ceil(duration - 1e-9))))
    return rows


def render(item_id: str, out, root=DEFAULT_ROOT, start=None, end=None, width_px=None, answer=None, reference=None,
           auto=None, settings=None) -> dict:
    """Draw ITEM (whole, or [start, end]) to ``out``. ``answer`` / ``reference``: lists of [start_s, end_s] spans
    (drawn in two lanes above the spectrogram, labelled with the item's syllables when the count matches);
    ``auto``: optional faint marks. Returns {"rows", "n_spans_drawn", "n_reference_drawn", "px_per_second"}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    st = settings or load_settings(root)
    d = st["display"]
    meta, arr = load_item(item_id, root)
    t, dur = arr["t_s"], arr["duration_s"]
    fs = float(st["frame_s"])
    if start is None and end is None:
        rows = overview_rows(dur, d)
        pps = float(d["px_per_second"])
    else:
        lo = 0.0 if start is None else float(start)
        hi = dur if end is None else float(end)
        if not hi > lo:
            raise PackageError(f"--start ({lo}) must be before --end ({hi})")
        rows = [(lo, hi)]
        pps = float(width_px or d["zoom_width_px"]) / (hi - lo)
    minor, major = tick_steps(pps, d)
    labels = [s["label"] for s in meta["syllables"]]
    lanes = (answer is not None) or (reference is not None)
    panels = [("spec", d["panel_heights_px"][0]), ("loud", d["panel_heights_px"][1])]
    if "neural" in arr:
        panels.append(("neural", d["panel_heights_px"][2]))
    if lanes:
        panels.insert(0, ("lanes", d["strip_height_px"]))
    m = d["margin_px"]
    title = (f"{meta['item_id']}   {meta['n_syllables']} syllables   {dur:.2f} s   "
             f"grid {minor:g} s (major {major:g} s)")
    sylline = "syllables: " + "  ".join(f"{k}:{lab}" for k, lab in enumerate(labels))
    tlines = [title] + textwrap.wrap(f"cue: {meta['text']}", int(d["title_chars_per_line"])) \
        + textwrap.wrap(sylline, int(d["title_chars_per_line"]))
    top = m["top"] + m["title_line"] * len(tlines)
    widths = [(hi - lo) * pps for lo, hi in rows]
    row_h = sum(h for _, h in panels) + d["panel_gap_px"] * (len(panels) - 1) + d["row_gap_px"]
    W = m["left"] + max(widths) + m["right"]
    H = top + row_h * len(rows) + m["bottom"]
    dpi = float(d["dpi"])
    plt.rcParams.update({"font.size": d["font_size"], "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    for i, line in enumerate(tlines):
        fig.text(m["left"] / W, 1 - (m["top"] + m["title_line"] * (i + 0.8)) / H, line, ha="left", va="baseline",
                 fontsize=d["font_size"] + (1 if i == 0 else 0), fontweight="bold" if i == 0 else "normal")

    mel, mel_hz = arr["mel_db"], arr["mel_hz"]
    top_db = float(np.nanmax(-mel)) if mel.size else 1.0
    loud = arr["loudness_db"]
    l_lo, l_hi = float(np.min(loud)) - d["loudness_headroom_db"], float(np.max(loud)) + d["loudness_headroom_db"]
    neu = arr.get("neural")
    n_lo, n_hi = (float(np.min(neu)) - 0.5, float(np.max(neu)) + 0.5) if neu is not None else (0, 1)
    fticks = [f for f in d["freq_ticks_hz"] if mel_hz[0] <= f <= mel_hz[-1]]
    fpos = np.interp(fticks, mel_hz, np.arange(mel_hz.size))
    drawn, ref_drawn = set(), set()
    alpha = float(d["span_alpha"])
    colors = ("tab:blue", "tab:orange")
    for r, (lo, hi) in enumerate(rows):
        y = top + r * row_h
        axes = {}
        for name, h in panels:
            ax = fig.add_axes([m["left"] / W, 1 - (y + h) / H, widths[r] / W, h / H])
            axes[name] = ax
            y += h + d["panel_gap_px"]
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_locator(MultipleLocator(major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
            ax.grid(True, which="major", axis="x", color="0.25", lw=0.8, alpha=0.7)
            ax.grid(True, which="minor", axis="x", color="0.5", lw=0.4, alpha=0.45)
            ax.tick_params(axis="x", which="both", labelbottom=False)
            if hi > dur:
                ax.axvspan(dur, hi, color="0.85", zorder=0)
            if lo < 0:
                ax.axvspan(lo, 0, color="0.85", zorder=0)
        bottom = axes[panels[-1][0]]
        bottom.tick_params(axis="x", which="major", labelbottom=True)
        bottom.xaxis.set_major_formatter(FormatStrFormatter(f"%.{_decimals(major)}f"))
        bottom.set_xlabel("time (s)", labelpad=1)
        ax = axes["spec"]
        ax.set_axisbelow(False)                                   # grid over the image
        ax.imshow(mel, origin="lower", aspect="auto", cmap=d["spectrogram_cmap"], vmin=-top_db, vmax=0,
                  interpolation="nearest", extent=(t[0] - fs / 2, t[-1] + fs / 2, -0.5, mel.shape[0] - 0.5))
        ax.set_xlim(lo, hi)
        ax.set_yticks(fpos)
        ax.set_yticklabels([f"{f / 1000:g}k" if f >= 1000 else f"{f:g}" for f in fticks])
        ax.set_ylabel("Hz")
        ax = axes["loud"]
        ax.plot(t, loud, color="black", lw=1.0)
        ax.set_ylim(l_lo, l_hi)
        ax.set_ylabel("loudness\n(dB)")
        ax.grid(True, which="major", axis="y", color="0.6", lw=0.4, alpha=0.5)
        if neu is not None:
            ax = axes["neural"]
            ax.plot(t, neu, color="tab:green", lw=1.0)
            ax.set_ylim(n_lo, n_hi)
            ax.set_ylabel("neural\n(z)")
        for name, ax in axes.items():
            if name == "lanes":
                continue
            if hi >= dur >= lo:
                ax.axvline(dur, color="black", lw=1.2)
            for a, b in auto or []:
                for x in (a, b):
                    if lo <= x <= hi:
                        ax.axvline(x, color="0.4", lw=0.7, ls=":", alpha=0.6)
            for k, (a, b) in enumerate(answer or []):
                if b > lo and a < hi:
                    ax.axvspan(a, b, color=colors[k % 2], alpha=alpha, lw=0)
            for a, b in reference or []:
                for x in (a, b):
                    if lo <= x <= hi:
                        ax.axvline(x, color="tab:green", lw=1.0, ls="--")
        if lanes:
            ax = axes["lanes"]
            ax.set_ylim(0, 2)
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(["reference", "answer"], fontsize=d["font_size"] - 1)
            ax.tick_params(axis="y", length=0)
            for lane, spans, done in ((1, answer, drawn), (0, reference, ref_drawn)):
                for k, (a, b) in enumerate(spans or []):
                    if not (b > lo and a < hi):
                        continue
                    col = "tab:green" if lane == 0 else colors[k % 2]
                    ax.add_patch(matplotlib.patches.Rectangle((a, lane + 0.08), b - a, 0.84, color=col,
                                                              alpha=0.55 if lane else 0.45, lw=0))
                    lab = f"{k}:{labels[k]}" if len(spans) == len(labels) else str(k)
                    x0, x1 = max(a, lo), min(b, hi)
                    if (x1 - x0) * pps < len(lab) * float(d["label_px_per_char"]):
                        lab = str(k)                              # too narrow: the index (the title lists the labels)
                    ax.text((x0 + x1) / 2, lane + 0.5, lab, ha="center", va="center", fontsize=d["font_size"] - 1,
                            clip_on=True)
                    done.add(k)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, metadata={"Software": None})
    plt.close(fig)
    return {"rows": [list(r) for r in rows], "n_spans_drawn": len(drawn), "n_reference_drawn": len(ref_drawn),
            "px_per_second": pps}


# ----------------------------------------------------------------------------------------- measuring


def frame_range(t: np.ndarray, start, end) -> np.ndarray:
    lo = -np.inf if start is None else float(start)
    hi = np.inf if end is None else float(end)
    return np.flatnonzero((t >= lo) & (t <= hi))


def band_db(mel_db: np.ndarray, mel_hz: np.ndarray, bands) -> np.ndarray:
    """(n_bands, T) energy in dB (re the item's loudest mel bin) of the mel bins whose centre lies in each band."""
    p = 10.0 ** (mel_db / 10.0)
    out = []
    for lo, hi in bands:
        sel = (mel_hz >= lo) & (mel_hz < hi)
        out.append(10.0 * np.log10(np.maximum(p[sel].sum(axis=0), 1e-12)) if sel.any() else np.full(p.shape[1], np.nan))
    return np.asarray(out)


def values_table(item_id: str, start: float, end: float, step=None, bands: bool = False, root=DEFAULT_ROOT) -> str:
    st = load_settings(root)
    meta, arr = load_item(item_id, root)
    t = arr["t_s"]
    fs = float(st["frame_s"])
    step = float(st["values"]["default_step_s"] if step is None else step)
    if not end > start:
        raise PackageError("--start must be before --end")
    if step <= 0:
        raise PackageError("--step must be positive")
    times = np.arange(start, end + step / 2, step)
    idx = np.clip(np.floor((times - t[0]) / fs + 0.5).astype(int), 0, t.size - 1)
    cols = ["time_s", "loudness_db"] + (["neural_z"] if "neural" in arr else [])
    bdb = None
    if bands:
        bl = st["values"]["bands_hz"]
        bdb = band_db(arr["mel_db"], arr["mel_hz"], bl)
        cols += [f"band_{int(a)}-{int(b)}Hz_db" for a, b in bl]
    lines = [f"# item {meta['item_id']}: values at the nearest 10 ms frame, every {step:g} s from {start:g} to {end:g} s",
             "  ".join(f"{c:>14s}" for c in cols)]
    for tt, i in zip(times, idx):
        row = [f"{tt:14.3f}", f"{arr['loudness_db'][i]:14.2f}"]
        if "neural" in arr:
            row.append(f"{arr['neural'][i]:14.2f}")
        if bdb is not None:
            row += [f"{v:14.2f}" for v in bdb[:, i]]
        lines.append("  ".join(row))
    return "\n".join(lines)


def prominences(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Topographic prominence of each local maximum (as scipy.signal.peak_prominences, whole signal)."""
    out = np.empty(peaks.size)
    for j, p in enumerate(peaks):
        left = x[:p][::-1]
        hl = np.flatnonzero(left > x[p])
        lmin = left[: hl[0]].min() if hl.size else (left.min() if left.size else x[p])
        right = x[p + 1:]
        hr = np.flatnonzero(right > x[p])
        rmin = right[: hr[0]].min() if hr.size else (right.min() if right.size else x[p])
        out[j] = x[p] - max(lmin, rmin)
    return out


def peaks_report(item_id: str, start=None, end=None, min_prominence=None, root=DEFAULT_ROOT) -> str:
    st = load_settings(root)
    meta, arr = load_item(item_id, root)
    t, x = arr["t_s"], arr["loudness_db"]
    minp = float(st["peaks"]["min_prominence_db"] if min_prominence is None else min_prominence)
    i = np.arange(1, x.size - 1)
    cand = i[(x[i] > x[i - 1]) & (x[i] >= x[i + 1])]
    prom = prominences(x, cand)
    keep = cand[prom >= minp]
    kprom = prom[prom >= minp]
    sel = frame_range(t, start, end)
    inside = np.isin(keep, sel)
    keep, kprom = keep[inside], kprom[inside]
    lo = t[sel[0]] if sel.size else float("nan")
    hi = t[sel[-1]] if sel.size else float("nan")
    lines = [f"# item {meta['item_id']}: loudness peaks with prominence >= {minp:g} dB between {lo:.3f} and {hi:.3f} s,",
             "# and the lowest point between each pair of neighbouring peaks (depth = how far it lies below each peak).",
             "# Report only: it does not decide where syllables are."]
    for n, (p, pr) in enumerate(zip(keep, kprom)):
        lines.append(f"PEAK  t={t[p]:8.3f} s  loudness={x[p]:7.2f} dB  prominence={pr:6.2f} dB")
        if n + 1 < keep.size:
            q = keep[n + 1]
            v = p + int(np.argmin(x[p:q + 1]))
            dl, dr = x[p] - x[v], x[q] - x[v]
            lines.append(f"DIP   t={t[v]:8.3f} s  loudness={x[v]:7.2f} dB  depth={min(dl, dr):6.2f} dB "
                         f"(below left peak {dl:.2f}, below right peak {dr:.2f})")
    if not keep.size:
        lines.append("no peak in this range")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- CLI


def check_report(path, root=DEFAULT_ROOT) -> tuple:
    """(exit code, report text) of ``check``."""
    obj, err = read_draft(path)
    if err:
        return 1, f"INVALID  {err}"
    st = load_settings(root)
    rules = response_rules(st)
    items = all_items(root)
    lines = []
    if isinstance(obj, dict) and "items" in obj:
        man = load_manifest(root)
        res = validate_response(obj, items, batch_sets(man), rules, man["items"])
        res["file_errors"] += path_errors(path, root, obj, rules, man["practice_batches"])
        for c, msg in res["file_errors"]:
            lines.append(f"ERROR [{c}] {msg}")
        for r in res["items"]:
            for c, msg in r["errors"]:
                lines.append(f"ERROR [{c}] {msg}")
        n_ok = len(accepted(res))
        ok = not lines
        head = (f"VALID  {n_ok} item answer(s), batch {res['batch']}, worker {res['worker_id']}" if ok else
                f"INVALID  {len(lines)} problem(s); {n_ok} of {len(res['items'])} item answers would be accepted")
    else:
        iid = obj.get("item_id") if isinstance(obj, dict) else None
        errs = validate_item_answer(obj, items.get(iid) if isinstance(iid, str) else None, rules)
        lines = [f"ERROR [{c}] {msg}" for c, msg in errs]
        ok = not errs
        head = f"VALID  item {iid}" if ok else f"INVALID  {len(errs)} problem(s)"
    return (0 if ok else 1), "\n".join([head] + lines)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="inspect_item.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", default=str(DEFAULT_ROOT), help="package folder (default: the folder above tools/)")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{render,overlay,values,peaks,check}")

    def view_args(p):
        p.add_argument("--start", type=float, default=None, help="zoom: first second shown")
        p.add_argument("--end", type=float, default=None, help="zoom: last second shown")
        p.add_argument("--out", default=None, help="PNG to write (default: answers/_views/<item>[_<start>-<end>].png)")
        p.add_argument("--width-px", type=int, default=None, help="zoomed image width in pixels (default from settings.json)")

    p = sub.add_parser("render", help="draw the three panels (whole item or zoomed)",
                       description="Draw the spectrogram, loudness and neural-activity panels of ITEM on one time axis.")
    p.add_argument("item")
    view_args(p)
    p = sub.add_parser("overlay", help="draw a draft's syllable spans on the item",
                       description="Like render, with the syllable spans of DRAFT drawn and labelled.")
    p.add_argument("item")
    p.add_argument("draft", help="one item answer, or a whole answer file that contains ITEM")
    view_args(p)
    p = sub.add_parser("values", help="table of loudness / neural activity / band energy",
                       description="Print time, loudness (dB) and neural activity (z) from START to END every STEP seconds.")
    p.add_argument("item")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--step", type=float, default=None, help="seconds between rows (default 0.01 = every frame)")
    p.add_argument("--bands", action="store_true", help="add spectral energy (dB) in the bands of settings.json")
    p = sub.add_parser("peaks", help="list loudness peaks and dips (report only)",
                       description="List loudness peaks (prominence >= the minimum) and the dips between them.")
    p.add_argument("item")
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--min-prominence", type=float, default=None, help="dB (default from settings.json)")
    p = sub.add_parser("check", help="validate a draft or an answer file",
                       description="Validate one item answer or a whole answer file against the answer format.")
    p.add_argument("draft")
    return ap


def _default_out(root: Path, item: str, start, end, suffix: str = "") -> Path:
    rng = "" if start is None and end is None else \
        f"_{0 if start is None else start:g}-{'end' if end is None else format(end, 'g')}"
    return root / "answers" / "_views" / f"{item}{rng}{suffix}.png"


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.package).resolve()
    try:
        if args.cmd in ("render", "overlay"):
            spans = None
            if args.cmd == "overlay":
                obj, err = read_draft(args.draft)
                if err:
                    print(f"ERROR {err}")
                    return 1
                mine = [e for e in draft_items(obj) if e.get("item_id") == args.item]
                if not mine:
                    print(f"ERROR the draft has no answer for item {args.item}")
                    return 1
                try:
                    spans = [[float(s["start_s"]), float(s["end_s"])] for s in mine[0]["syllables"]]
                except (KeyError, TypeError, ValueError):
                    print("ERROR every syllable of the draft needs numeric start_s and end_s; run check for details")
                    return 1
            out = Path(args.out) if args.out else _default_out(root, args.item, args.start, args.end,
                                                                "_overlay" if spans is not None else "")
            info = render(args.item, out, root, args.start, args.end, args.width_px, answer=spans)
            msg = f"wrote {out} ({len(info['rows'])} row(s), {info['px_per_second']:.0f} px/s"
            if spans is not None:
                msg += f", drew {info['n_spans_drawn']} of {len(spans)} spans"
            print(msg + ")")
            return 0
        if args.cmd == "values":
            print(values_table(args.item, args.start, args.end, args.step, args.bands, root))
            return 0
        if args.cmd == "peaks":
            print(peaks_report(args.item, args.start, args.end, args.min_prominence, root))
            return 0
        code, text = check_report(args.draft, root)
        print(text)
        return code
    except PackageError as e:
        print(f"ERROR {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
