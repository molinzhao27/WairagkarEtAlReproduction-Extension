#!/usr/bin/env python3
"""Compare the workers' answers per item: how far apart their syllable starts are, and which items to re-assign.

Needs Python >= 3.9 with numpy. Run it from the package folder:

  python tools/agreement.py [--answers DIR] [--threshold-ms MS] [--json OUT.json]

It reads every answer file answers/<worker_id>/<batch_id>.json and answers/<worker_id>/extra*.json (practice
answers and anything in subfolders are ignored), checks each with the same rules as inspect_item.py check, and
uses the valid item answers only. If one worker answered the same item twice, every copy is set aside.

Per item, over every pair of workers and every syllable: the median |start difference| (ms) and the share of
start differences within the tolerance (settings.json, 100 ms); the same for ends. An item is FLAGGED when it
has at least two answers and its median |start difference| exceeds the threshold (settings.json, or
--threshold-ms): give it to one more worker who has not seen it (the table lists who has). Items with fewer
valid answers than the planned redundancy are listed as MISSING. No reference timing is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                     # leave tools/ exactly as delivered (no __pycache__)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import inspect_item as ii  # noqa: E402  (the sibling tool: package loading and the answer rules)


def item_agreement(spans_list, tol_ms: float) -> dict:
    """Pairwise agreement of several answers (each a list of [start_s, end_s], same length) for one item."""
    n = len(spans_list)
    out = {"n_answers": n, "n_pairs": 0, "median_abs_start_diff_ms": None, "start_within_tol": None,
           "median_abs_end_diff_ms": None, "end_within_tol": None}
    if n < 2:
        return out
    a = np.asarray(spans_list, dtype=float)                       # (n, K, 2)
    i, j = np.triu_indices(n, k=1)
    ds = 1000.0 * np.abs(a[i, :, 0] - a[j, :, 0]).ravel()
    de = 1000.0 * np.abs(a[i, :, 1] - a[j, :, 1]).ravel()
    out.update({"n_pairs": int(i.size), "median_abs_start_diff_ms": float(np.median(ds)),
                "start_within_tol": float(np.mean(ds <= tol_ms + 1e-9)),
                "median_abs_end_diff_ms": float(np.median(de)), "end_within_tol": float(np.mean(de <= tol_ms + 1e-9))})
    return out


def pooled(spans_by_item: dict, tol_ms: float) -> dict:
    ds = []
    for spans_list in spans_by_item.values():
        if len(spans_list) >= 2:
            a = np.asarray(spans_list, dtype=float)
            i, j = np.triu_indices(len(spans_list), k=1)
            ds.append(1000.0 * np.abs(a[i, :, 0] - a[j, :, 0]).ravel())
    if not ds:
        return {"n_start_pairs": 0, "median_abs_start_diff_ms": None, "start_within_tol": None}
    d = np.concatenate(ds)
    return {"n_start_pairs": int(d.size), "median_abs_start_diff_ms": float(np.median(d)),
            "start_within_tol": float(np.mean(d <= tol_ms + 1e-9))}


def collect(answers_dir, root) -> tuple:
    """({item: {worker: spans}}, problems): the valid answers of every real answer file under answers_dir."""
    root = Path(root)
    st = ii.load_settings(root)
    rules = ii.response_rules(st)
    man = ii.load_manifest(root)
    items = ii.all_items(root)
    batches = ii.batch_sets(man)
    found, problems = {}, []
    answers_dir = Path(answers_dir)
    for wdir in sorted(p for p in answers_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))) \
            if answers_dir.is_dir() else []:
        for f in sorted(wdir.glob("*.json")):
            obj, err = ii.read_draft(f)
            if err:
                problems.append(f"{f}: {err}")
                continue
            if isinstance(obj, dict) and obj.get("batch") in man["practice_batches"]:
                continue
            res = ii.validate_response(obj, items, batches, rules, man["items"])
            res["file_errors"] += ii.path_errors(f, root, obj, rules, man["practice_batches"])
            if res["file_errors"]:
                problems.append(f"{f}: rejected ({'; '.join(m for _, m in res['file_errors'])})")
                continue
            for r in res["items"]:
                if r["errors"]:
                    problems.append(f"{f}: item {r['item_id']} rejected ({'; '.join(m for _, m in r['errors'])})")
                else:
                    found.setdefault(r["item_id"], {}).setdefault(res["worker_id"], []).append((str(f), r["spans"]))
    answers = {}
    for iid, by_worker in found.items():
        for w, copies in by_worker.items():
            if len(copies) > 1:
                problems.append(f"item {iid}: worker {w} answered it {len(copies)} times "
                                f"({', '.join(c[0] for c in copies)}); every copy is set aside")
            else:
                answers.setdefault(iid, {})[w] = copies[0][1]
    return answers, problems


def report(answers_dir, root, threshold_ms=None) -> tuple:
    """(text, results dict)."""
    root = Path(root)
    st = ii.load_settings(root)
    man = ii.load_manifest(root)
    ag = st["agreement"]
    tol = float(ag["tolerance_ms"])
    thr = float(ag["flag_median_abs_start_ms"] if threshold_ms is None else threshold_ms)
    k = int(man["counts"]["redundancy_k"])
    answers, problems = collect(answers_dir, root)
    per_item, flagged, missing = {}, [], []
    for iid in sorted(man["items"]):
        by_w = answers.get(iid, {})
        workers = sorted(by_w)
        a = item_agreement([by_w[w] for w in workers], tol)
        a["workers"] = workers
        a["flagged"] = bool(a["n_answers"] >= int(ag["min_answers"]) and a["median_abs_start_diff_ms"] > thr)
        per_item[iid] = a
        if a["flagged"]:
            flagged.append(iid)
        if a["n_answers"] < k:
            missing.append(iid)
    pool = pooled({i: [answers[i][w] for w in sorted(answers[i])] for i in answers}, tol)
    summary = {"n_items": len(man["items"]), "n_items_with_2plus_answers": sum(a["n_answers"] >= 2 for a in per_item.values()),
               "n_flagged": len(flagged), "n_missing": len(missing), "redundancy_k": k, "tolerance_ms": tol,
               "threshold_ms": thr, **pool}
    f = lambda x, s="{:.0f}": "-" if x is None else s.format(x)    # noqa: E731
    lines = [f"AGREEMENT  {summary['n_items']} items, {summary['n_items_with_2plus_answers']} with >= 2 valid answers; "
             f"pooled median |start difference| {f(pool['median_abs_start_diff_ms'])} ms, "
             f"{f(None if pool['start_within_tol'] is None else 100 * pool['start_within_tol'])}% within {tol:g} ms; "
             f"{len(flagged)} flagged (median > {thr:g} ms), {len(missing)} with fewer than {k} answers",
             f"{'item':<12s} {'n':>2s} {'med|dstart|ms':>13s} {'within%':>8s} {'med|dend|ms':>11s}  workers"]
    for iid, a in per_item.items():
        lines.append(f"{iid:<12s} {a['n_answers']:2d} {f(a['median_abs_start_diff_ms']):>13s} "
                     f"{f(None if a['start_within_tol'] is None else 100 * a['start_within_tol']):>8s} "
                     f"{f(a['median_abs_end_diff_ms']):>11s}  {','.join(a['workers'])}{'   FLAGGED' if a['flagged'] else ''}")
    lines.append(f"FLAGGED ({len(flagged)}; give each to one more worker who is not listed): " + (" ".join(flagged) or "none"))
    lines.append(f"MISSING ({len(missing)}; fewer than {k} valid answers): " + (" ".join(missing) or "none"))
    if problems:
        lines.append(f"PROBLEMS ({len(problems)}):")
        lines += [f"  {p}" for p in problems]
    return "\n".join(lines), {"summary": summary, "items": per_item, "flagged": flagged, "missing": missing,
                              "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agreement.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", default=None, metavar="DIR", help="answers folder (default: answers/ of the package)")
    ap.add_argument("--threshold-ms", type=float, default=None, help="flag above this median |start difference| (ms)")
    ap.add_argument("--json", default=None, metavar="OUT", help="also write the results as JSON")
    ap.add_argument("--package", default=str(ii.DEFAULT_ROOT), help="package folder (default: the folder above tools/)")
    args = ap.parse_args(argv)
    root = Path(args.package)
    try:
        text, res = report(args.answers or root / "answers", root, args.threshold_ms)
    except ii.PackageError as e:
        print(f"ERROR {e}")
        return 2
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
