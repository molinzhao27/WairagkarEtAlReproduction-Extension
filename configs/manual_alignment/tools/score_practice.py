#!/usr/bin/env python3
"""Score your answers for PRACTICE items against their reference syllable spans (practice/truth/).

Needs Python >= 3.9 with numpy (matplotlib too for --overlay). Run it from the package folder:

  python tools/score_practice.py ANSWERS.json [--overlay DIR] [--json OUT.json]

ANSWERS.json is a practice answer file ({"schema", "batch": "practice_NN", "worker_id", "items": [...]}) or one
item answer. Only practice items (ids starting with "p", files in practice/items/) can be scored: any other item
is refused, because reference spans exist for practice items only.

Per item and over all items it reports:
  - starts within T ms: the share of reference syllable starts that have one of your starts within T ms
    (T from settings.json, 100 ms);
  - median |start error|: over the reference syllables, the distance to your nearest start (ms);
  - count agreement: whether the reference has as many syllables as the item lists (when it does not, the
    participant merged or split syllables; your answer still lists every requested syllable);
  - span start / end error: your first start minus the reference first start, your last end minus the
    reference last end (ms; positive = late).
A per-syllable table follows: when the counts agree, reference syllable k is paired with your syllable k (start
and end errors); otherwise each reference syllable is paired with your nearest start.
--overlay DIR writes DIR/<item>_score.png: the item with your spans (answer lane) and the reference spans
(reference lane, green dashed lines), so you can see where you were off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                     # leave tools/ exactly as delivered (no __pycache__)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import inspect_item as ii  # noqa: E402  (the sibling tool: package loading and the answer rules)


class Refused(Exception):
    pass


def reference(item_id: str, root) -> dict:
    """The reference spans of a PRACTICE item; refuses any other item."""
    root = Path(root)
    st = ii.load_settings(root)
    prefix = st["ids"]["practice_prefix"]
    item_file = root / "practice" / "items" / f"{item_id}.json"
    ref_file = root / "practice" / "truth" / f"{item_id}.json"
    if not (isinstance(item_id, str) and item_id.startswith(prefix) and item_file.is_file() and ref_file.is_file()):
        raise Refused(f"item {item_id!r} is not a practice item: only items in practice/items/ can be scored")
    ref = ii.load_json(ref_file)
    digest = hashlib.sha256(item_file.read_bytes()).hexdigest()
    if ref.get("schema") != st["schemas"]["reference"] or ref.get("item_id") != item_id or ref.get("item_sha256") != digest:
        raise Refused(f"practice/truth/{item_id}.json does not belong to practice/items/{item_id}.json")
    return ref


def nearest_errors(ours: np.ndarray, theirs: np.ndarray) -> tuple:
    """For every reference time, (signed distance to the nearest of ours, index of that one)."""
    d = ours[None, :] - theirs[:, None]
    j = np.argmin(np.abs(d), axis=1)
    return d[np.arange(theirs.size), j], j


def score_item(spans, ref: dict, tol_ms: float) -> dict:
    a = np.asarray(spans, dtype=float).reshape(-1, 2)
    r = np.asarray([[s["start_s"], s["end_s"]] for s in ref["syllables"]], dtype=float).reshape(-1, 2)
    err, j = nearest_errors(a[:, 0], r[:, 0])
    err_ms = 1000.0 * err
    paired = None
    if len(a) == len(r):
        paired = {"start_ms": (1000.0 * (a[:, 0] - r[:, 0])).tolist(), "end_ms": (1000.0 * (a[:, 1] - r[:, 1])).tolist()}
    return {
        "n_requested": int(len(a)), "n_reference": int(len(r)), "count_agrees": bool(len(a) == len(r)),
        "starts_within_tol": float(np.mean(np.abs(err_ms) <= tol_ms + 1e-9)),
        "n_starts_within_tol": int(np.sum(np.abs(err_ms) <= tol_ms + 1e-9)),
        "median_abs_start_error_ms": float(np.median(np.abs(err_ms))),
        "span_start_error_ms": float(1000.0 * (a[0, 0] - r[0, 0])),
        "span_end_error_ms": float(1000.0 * (a[-1, 1] - r[-1, 1])),
        "nearest": {"start_error_ms": err_ms.tolist(), "answer_index": j.tolist()},
        "paired": paired,
        "reference_spans": r.tolist(), "answer_spans": a.tolist(),
    }


def item_table(item_id: str, labels, sc: dict, tol_ms: float) -> list:
    lines = [f"== {item_id}: starts within {tol_ms:g} ms {sc['n_starts_within_tol']}/{sc['n_reference']}, "
             f"median |start error| {sc['median_abs_start_error_ms']:.0f} ms, "
             f"count {'agrees' if sc['count_agrees'] else 'differs'} (reference {sc['n_reference']}, requested "
             f"{sc['n_requested']}), span start {sc['span_start_error_ms']:+.0f} ms, span end {sc['span_end_error_ms']:+.0f} ms"]
    r, a = sc["reference_spans"], sc["answer_spans"]
    if sc["paired"] is not None:
        lines.append(f"   {'k':>3s} {'label':<16s} {'ref start':>9s} {'ref end':>9s} {'your start':>10s} {'your end':>9s} "
                     f"{'start err':>10s} {'end err':>9s}")
        for k, lab in enumerate(labels):
            es, ee = sc["paired"]["start_ms"][k], sc["paired"]["end_ms"][k]
            flag = "" if abs(es) <= tol_ms + 1e-9 else "   <-- start off"
            lines.append(f"   {k:3d} {lab:<16s} {r[k][0]:9.2f} {r[k][1]:9.2f} {a[k][0]:10.2f} {a[k][1]:9.2f} "
                         f"{es:+8.0f}ms {ee:+7.0f}ms{flag}")
    else:
        lines.append(f"   reference has {sc['n_reference']} syllables, the item lists {sc['n_requested']}: each reference "
                     "syllable is paired with your nearest start")
        lines.append(f"   {'ref':>3s} {'ref start':>9s} {'ref end':>9s} {'nearest yours':>14s} {'start err':>10s}")
        for k, (e, j) in enumerate(zip(sc["nearest"]["start_error_ms"], sc["nearest"]["answer_index"])):
            flag = "" if abs(e) <= tol_ms + 1e-9 else "   <-- start off"
            lines.append(f"   {k:3d} {r[k][0]:9.2f} {r[k][1]:9.2f} {j:5d}:{labels[j]:<8s} {e:+8.0f}ms{flag}")
    return lines


def run(path, root, overlay=None) -> tuple:
    """(exit code, text, results dict)."""
    root = Path(root)
    st = ii.load_settings(root)
    tol = float(st["practice_scoring"]["tolerance_ms"])
    obj, err = ii.read_draft(path)
    if err:
        return 1, f"ERROR {err}", None
    entries = ii.draft_items(obj)
    ids = [e.get("item_id") for e in entries]
    refs = {}
    for iid in ids:
        refs[iid] = reference(iid, root)              # raises Refused before anything is scored
    rules = ii.response_rules(st)
    items = ii.all_items(root)
    if isinstance(obj, dict) and "items" in obj:
        man = ii.load_manifest(root)
        res = ii.validate_response(obj, items, ii.batch_sets(man), rules, man["items"])
        problems = [f"ERROR [{c}] {m}" for c, m in res["file_errors"]] + [
            f"ERROR [{c}] {m}" for r in res["items"] for c, m in r["errors"]]
    else:
        problems = [f"ERROR [{c}] {m}" for c, m in ii.validate_item_answer(obj, items.get(ids[0]) if ids else None, rules)]
    if problems:
        return 1, "\n".join(["INVALID answer file (fix it with inspect_item.py check, then score again):"] + problems), None
    lines, results = [], {}
    for e in entries:
        iid = e["item_id"]
        meta = ii.load_json(ii.find_item(iid, root))
        sc = score_item(ii.answer_spans(e), refs[iid], tol)
        results[iid] = sc
        lines += item_table(iid, [s["label"] for s in meta["syllables"]], sc, tol)
        if overlay:
            out = Path(overlay) / f"{iid}_score.png"
            ii.render(iid, out, root, answer=sc["answer_spans"], reference=sc["reference_spans"], settings=st)
            lines.append(f"   overlay: {out}")
    n_ref = sum(s["n_reference"] for s in results.values())
    within = sum(s["n_starts_within_tol"] for s in results.values())
    all_err = np.concatenate([np.abs(s["nearest"]["start_error_ms"]) for s in results.values()])
    summary = {"n_items": len(results), "starts_within_tol": within / n_ref, "tolerance_ms": tol,
               "median_abs_start_error_ms": float(np.median(all_err)),
               "count_agreement": float(np.mean([s["count_agrees"] for s in results.values()])),
               "median_abs_span_start_error_ms": float(np.median([abs(s["span_start_error_ms"]) for s in results.values()]))}
    head = (f"PRACTICE SCORE  {summary['n_items']} item(s): starts within {tol:g} ms {within}/{n_ref} "
            f"({100 * summary['starts_within_tol']:.0f}%), median |start error| {summary['median_abs_start_error_ms']:.0f} ms, "
            f"count agreement {100 * summary['count_agreement']:.0f}%, median |span start error| "
            f"{summary['median_abs_span_start_error_ms']:.0f} ms")
    return 0, "\n".join([head] + lines), {"summary": summary, "items": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="score_practice.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", help="practice answer file, or one practice item answer")
    ap.add_argument("--overlay", default=None, metavar="DIR", help="write DIR/<item>_score.png (your spans vs the reference)")
    ap.add_argument("--json", default=None, metavar="OUT", help="also write the scores as JSON")
    ap.add_argument("--package", default=str(ii.DEFAULT_ROOT), help="package folder (default: the folder above tools/)")
    args = ap.parse_args(argv)
    try:
        code, text, res = run(args.answers, args.package, args.overlay)
    except Refused as e:
        print(f"REFUSED {e}")
        return 3
    except ii.PackageError as e:
        print(f"ERROR {e}")
        return 2
    print(text)
    if res is not None and args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
