#!/usr/bin/env python
"""Generic MATLAB .mat dumper (v4/v5/v7 via scipy.io, v7.3/HDF5 via h5py).

Recurses into structs and cell arrays and prints, for every variable/leaf:
key path, shape, dtype/class and small value summaries (min/max/mean/std,
NaN count, first few elements, unique values for small integer/string arrays).
For 2-D numeric arrays it also summarises per-column means/stds (useful to see
whether features are z-scored). Cell arrays with many elements are summarised
(element shapes/dtypes aggregated) and only the first few elements are expanded.

Usage:
    python -I scripts/inspect_mat.py FILE.mat [FILE2.mat ...] [--json OUT.json]
        [--max-elems 3] [--max-preview 8] [--max-unique 50]

With several input files and --json, OUT may contain "{stem}" which is replaced
by each file's stem; otherwise a single JSON with one entry per file is written.

Dependencies: numpy, scipy, h5py (only for v7.3 files) + stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import load_config  # noqa: E402
from wvx.provenance import code_version, config_sha256  # noqa: E402
from wvx.seeding import set_seed  # noqa: E402


# ----------------------------------------------------------------------------- helpers

def _py(v):
    """Make numpy scalars JSON-serialisable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if np.isfinite(f) else str(f)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, float) and not np.isfinite(v):
        return str(v)
    return v


def _trunc_str(s: str, n: int = 200) -> str:
    return s if len(s) <= n else s[:n] + f"...<+{len(s) - n} chars>"


def numeric_summary(a: np.ndarray, opts) -> dict:
    out: dict = {}
    n = a.size
    if n == 0:
        out["empty"] = True
        return out
    if a.dtype == bool:
        out["n_true"] = int(a.sum())
        out["n_false"] = int(n - a.sum())
        return out
    if np.iscomplexobj(a):
        out["complex"] = True
        a = np.abs(a)
    af = a.astype(np.float64, copy=False) if a.dtype.kind in "iuf" else a
    if af.dtype.kind == "f":
        finite = np.isfinite(af)
        n_nan = int(np.isnan(af).sum())
        n_inf = int((~finite).sum()) - n_nan
        out["n_nan"] = n_nan
        if n_inf:
            out["n_inf"] = n_inf
        fv = af[finite] if (n_nan or n_inf) else af
        if fv.size:
            out["min"] = _py(fv.min())
            out["max"] = _py(fv.max())
            out["mean"] = _py(fv.mean())
            out["std"] = _py(fv.std())
            # integer-valued floats?
            out["all_integer_valued"] = bool(np.all(np.equal(np.mod(fv, 1), 0)))
            out["frac_zero"] = _py((fv == 0).mean())
    flat = a.reshape(-1, order="F")  # MATLAB (column-major) order
    out["first_elems_colmajor"] = [_py(x) for x in flat[: opts.max_preview]]
    is_intlike = a.dtype.kind in "iub" or out.get("all_integer_valued", False)
    if is_intlike and n <= 5_000_000:
        u, c = np.unique(a, return_counts=True)
        out["n_unique"] = int(u.size)
        if u.size <= opts.max_unique:
            out["unique_counts"] = {str(_py(k)): int(v) for k, v in zip(u, c)}
    elif n <= 200_000:
        out["n_unique"] = int(np.unique(a).size)
    # per-column stats for 2-D data (rows = time / samples, columns = channels)
    if a.ndim == 2 and a.shape[0] > a.shape[1] >= 2 and a.shape[1] <= 4096 and af.dtype.kind == "f":
        with np.errstate(all="ignore"):
            cm = np.nanmean(af, axis=0)
            cs = np.nanstd(af, axis=0)
        out["per_column"] = {
            "n_columns": int(a.shape[1]),
            "mean_of_col_means": _py(np.nanmean(cm)),
            "col_mean_min_median_max": [_py(np.nanmin(cm)), _py(np.nanmedian(cm)), _py(np.nanmax(cm))],
            "col_std_min_median_max": [_py(np.nanmin(cs)), _py(np.nanmedian(cs)), _py(np.nanmax(cs))],
            "n_constant_columns": int(np.sum(cs == 0)),
        }
    return out


def string_list_summary(strings: list[str], opts) -> dict:
    cnt = Counter(strings)
    return {
        "n_strings": len(strings),
        "n_unique": len(cnt),
        "n_empty": sum(1 for s in strings if s.strip() == ""),
        "first_values": [_trunc_str(s) for s in strings[: opts.max_preview]],
        "most_common": [[_trunc_str(k, 120), v] for k, v in cnt.most_common(5)],
        "length_min_max": [min(map(len, strings)), max(map(len, strings))] if strings else None,
    }


def aggregate_shapes(shapes: list[tuple]) -> dict:
    ndims = Counter(len(s) for s in shapes)
    agg: dict = {"ndim_counts": {str(k): v for k, v in ndims.items()}}
    if len(ndims) == 1:
        nd = next(iter(ndims))
        for d in range(nd):
            vals = np.array([s[d] for s in shapes])
            agg[f"dim{d}"] = {"min": int(vals.min()), "max": int(vals.max()),
                              "mean": float(vals.mean()), "n_unique": int(np.unique(vals).size)}
    return agg


def print_node(node: dict, path: str, indent: int, out=sys.stdout) -> None:
    pad = "  " * indent
    kind = node.get("kind")
    head = f"{pad}{path}: kind={kind}"
    if "matlab_shape" in node:
        head += f" matlab_shape={node['matlab_shape']}"
    if "shape" in node:
        head += f" shape={node['shape']}"
    if "dtype" in node:
        head += f" dtype={node['dtype']}"
    if "matlab_class" in node:
        head += f" class={node['matlab_class']}"
    print(head, file=out)
    for k in ("summary", "value", "strings", "element_summary", "element_shapes", "note"):
        if k in node:
            v = node[k]
            s = json.dumps(v, default=str)
            if len(s) > 1500:
                s = s[:1500] + "...<truncated>"
            print(f"{pad}  {k}: {s}", file=out)
    for ck, cv in node.get("fields", {}).items():
        print_node(cv, f"{path}.{ck}", indent + 1, out)
    for ek, ev in node.get("elements", {}).items():
        print_node(ev, f"{path}{ek}", indent + 1, out)


# ----------------------------------------------------------------------------- scipy (v5/v7)

def describe_scipy(obj, opts, depth=0) -> dict:
    import scipy.io.matlab as sm

    if depth > opts.max_depth:
        return {"kind": "truncated_depth"}
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:  # struct array
            node = {"kind": "struct", "shape": list(obj.shape), "fieldnames": list(obj.dtype.names)}
            if obj.size == 1:
                el = obj.reshape(-1)[0]
                node["fields"] = {f: describe_scipy(el[f], opts, depth + 1) for f in obj.dtype.names}
            else:
                node["note"] = f"struct array with {obj.size} elements; first element expanded"
                el = obj.reshape(-1, order="F")[0]
                node["fields"] = {f: describe_scipy(el[f], opts, depth + 1) for f in obj.dtype.names}
            return node
        if obj.dtype == object:  # cell array
            flat = obj.reshape(-1, order="F")
            node = {"kind": "cell", "shape": list(obj.shape), "n_elements": int(flat.size)}
            # string cells
            if flat.size and all(isinstance(x, np.ndarray) and x.dtype.kind == "U" for x in flat):
                strs = ["".join(x.reshape(-1).tolist()) if x.size else "" for x in flat]
                node["element_summary"] = {"all_char": True}
                node["strings"] = string_list_summary(strs, opts)
                return node
            kinds = Counter()
            dtypes = Counter()
            shapes = []
            for x in flat:
                if isinstance(x, np.ndarray):
                    kinds["struct" if x.dtype.names else ("cell" if x.dtype == object else "array")] += 1
                    dtypes[str(x.dtype)] += 1
                    shapes.append(tuple(x.shape))
                else:
                    kinds[type(x).__name__] += 1
            node["element_summary"] = {"kinds": dict(kinds), "dtypes": dict(dtypes)}
            if shapes:
                node["element_shapes"] = aggregate_shapes(shapes)
            node["elements"] = {
                f"{{{i}}}": describe_scipy(flat[i], opts, depth + 1) for i in range(min(opts.max_elems, flat.size))
            }
            return node
        if obj.dtype.kind == "U":
            strs = ["".join(r) for r in obj.reshape(obj.shape[0], -1)] if obj.ndim >= 2 else [str(x) for x in obj.reshape(-1)]
            if obj.ndim == 1:
                strs = [str(x) for x in obj]
            return {"kind": "char", "shape": list(obj.shape), "dtype": str(obj.dtype),
                    "strings": string_list_summary(strs, opts)}
        node = {"kind": "numeric" if obj.dtype != bool else "logical", "shape": list(obj.shape),
                "dtype": str(obj.dtype)}
        node["summary"] = numeric_summary(obj, opts)
        return node
    if isinstance(obj, sm.MatlabOpaque):
        return {"kind": "matlab_opaque", "note": "MATLAB opaque object (e.g. string/table/function handle); not decodable by scipy",
                "dtype": str(getattr(obj, "dtype", type(obj)))}
    if isinstance(obj, sm.MatlabFunction):
        return {"kind": "matlab_function"}
    if isinstance(obj, str):
        return {"kind": "str", "value": _trunc_str(obj)}
    return {"kind": type(obj).__name__, "value": _trunc_str(repr(obj))}


def v5_container_info(path: Path) -> dict:
    """Read the 128-byte header and the type of the first data element (15 = miCOMPRESSED => v7)."""
    with open(path, "rb") as f:
        hdr = f.read(128)
        nxt = f.read(8)
    text = hdr[:116].decode("ascii", "replace").rstrip()
    version = struct.unpack("<H", hdr[124:126])[0] if hdr[126:128] == b"IM" else struct.unpack(">H", hdr[124:126])[0]
    endian = hdr[126:128].decode("ascii", "replace")
    info = {"header_text": text, "header_version_field": hex(version), "endian_indicator": endian}
    if len(nxt) == 8:
        fmt = "<I" if endian == "IM" else ">I"
        dtype_tag = struct.unpack(fmt, nxt[:4])[0]
        info["first_element_type"] = int(dtype_tag)
        info["first_element_compressed"] = dtype_tag == 15
    return info


def inspect_scipy(path: Path, opts) -> dict:
    import scipy.io

    result: dict = {"container": v5_container_info(path)}
    who = scipy.io.whosmat(str(path))
    result["whosmat"] = [{"name": n, "shape": list(s), "class": c} for n, s, c in who]
    data = scipy.io.loadmat(str(path), squeeze_me=False, struct_as_record=True, mat_dtype=False)
    result["variables"] = {}
    for k in sorted(data):
        if k.startswith("__"):
            v = data[k]
            result.setdefault("mat_meta", {})[k] = _py(v) if not isinstance(v, (list, np.ndarray)) else str(v)
            continue
        result["variables"][k] = describe_scipy(data[k], opts)
    return result


# ----------------------------------------------------------------------------- h5py (v7.3)

def describe_h5(f, obj, opts, depth=0) -> dict:
    import h5py

    if depth > opts.max_depth:
        return {"kind": "truncated_depth"}
    mclass = obj.attrs.get("MATLAB_class", b"")
    mclass = mclass.decode() if isinstance(mclass, bytes) else str(mclass)
    if isinstance(obj, h5py.Group):
        node = {"kind": "struct" if mclass == "struct" else "group", "matlab_class": mclass or None}
        node["fields"] = {k: describe_h5(f, obj[k], opts, depth + 1) for k in obj.keys()}
        return node
    shape = tuple(obj.shape)
    node = {"kind": None, "shape": list(shape), "matlab_shape": list(shape[::-1]), "dtype": str(obj.dtype),
            "matlab_class": mclass or None}
    if obj.attrs.get("MATLAB_empty", 0):
        node["kind"] = "empty"
        return node
    if obj.dtype == h5py.ref_dtype or obj.dtype.kind == "O":
        node["kind"] = "cell"
        refs = obj[()].reshape(-1)  # h5py C-order == MATLAB column-major order of the transposed shape
        node["n_elements"] = int(refs.size)
        children = [f[r] for r in refs]
        if children and all(c.attrs.get("MATLAB_class", b"") in (b"char", "char") for c in children):
            strs = []
            for c in children:
                if c.attrs.get("MATLAB_empty", 0):
                    strs.append("")
                else:
                    arr = c[()]
                    strs.append("".join(chr(x) for x in arr.T.reshape(-1)))
            node["element_summary"] = {"all_char": True}
            node["strings"] = string_list_summary(strs, opts)
            return node
        shapes = [tuple(c.shape[::-1]) for c in children if hasattr(c, "shape")]
        node["element_summary"] = {"classes": dict(Counter(str(c.attrs.get("MATLAB_class", b"")) for c in children))}
        if shapes:
            node["element_shapes"] = aggregate_shapes(shapes)
        node["elements"] = {f"{{{i}}}": describe_h5(f, children[i], opts, depth + 1)
                            for i in range(min(opts.max_elems, len(children)))}
        return node
    data = obj[()]
    if mclass == "char":
        node["kind"] = "char"
        arr = np.asarray(data)
        if arr.ndim == 2:
            strs = ["".join(chr(x) for x in row) for row in arr.T]
        else:
            strs = ["".join(chr(x) for x in arr.reshape(-1))]
        node["strings"] = string_list_summary(strs, opts)
        return node
    node["kind"] = "logical" if mclass == "logical" else "numeric"
    arr = np.asarray(data).T  # back to MATLAB orientation
    node["summary"] = numeric_summary(arr, opts)
    return node


def inspect_h5(path: Path, opts) -> dict:
    import h5py

    with open(path, "rb") as fh:
        hdr = fh.read(128)
    result: dict = {"container": {"header_text": hdr[:116].decode("ascii", "replace").rstrip()}}
    result["variables"] = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if k.startswith("#"):
                continue
            result["variables"][k] = describe_h5(f, f[k], opts)
    return result


# ----------------------------------------------------------------------------- main

def inspect_file(path: Path, opts) -> dict:
    import h5py
    import scipy.io.matlab as sm

    with open(path, "rb") as fh:
        try:
            major, minor = sm.matfile_version(fh)
        except Exception as e:  # noqa: BLE001
            major, minor = None, None
            err = str(e)
        else:
            err = None
    entry = {"file": path.name, "bytes": path.stat().st_size}
    if major == 2 or (major is None and h5py.is_hdf5(str(path))):
        entry["format"] = "MATLAB v7.3 (HDF5)"
        entry.update(inspect_h5(path, opts))
    elif major in (0, 1):
        if major == 0:
            entry["format"] = "MATLAB v4"
        else:
            entry["format"] = "MATLAB v5-family (v6/v7); see container.first_element_compressed (True => v7 compressed)"
        entry["scipy_matfile_version"] = [major, minor]
        entry.update(inspect_scipy(path, opts))
    else:
        entry["format"] = f"UNKNOWN ({err})"
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write machine-readable summary here")
    ap.add_argument("--max-elems", type=int, default=3, help="cell elements to expand")
    ap.add_argument("--max-preview", type=int, default=8, help="preview items per array/string list")
    ap.add_argument("--max-unique", type=int, default=50, help="max unique values to list")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--quiet", action="store_true", help="do not print the text dump")
    opts = ap.parse_args(argv)
    set_seed(int(load_config()["seed"]))  # nothing here is random; AGENTS.md rule 7
    # Provenance for the JSON output: code version and the hash of the dump options (no YAML config here).
    provenance = {"code_version": code_version(),
                  "config_sha256": config_sha256({k: v for k, v in vars(opts).items() if k not in ("files", "json")})}

    entries = []
    for p in opts.files:
        entry = inspect_file(p, opts)
        entry["provenance"] = provenance
        entries.append(entry)
        if not opts.quiet:
            print(f"===== {entry['file']} ({entry['bytes']} bytes)")
            print(f"format: {entry['format']}")
            if "container" in entry:
                print(f"container: {json.dumps(entry['container'])}")
            for w in entry.get("whosmat", []):
                print(f"whosmat: {w}")
            for k, node in entry.get("variables", {}).items():
                print_node(node, k, 0)
            print()
        if opts.json is not None and "{stem}" in str(opts.json):
            out = Path(str(opts.json).replace("{stem}", p.stem))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(entry, indent=1, default=str), encoding="utf-8")
            print(f"[json] {out} ({os.path.getsize(out)} bytes)", file=sys.stderr)
    if opts.json is not None and "{stem}" not in str(opts.json):
        opts.json.parent.mkdir(parents=True, exist_ok=True)
        payload = entries[0] if len(entries) == 1 else entries
        opts.json.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        print(f"[json] {opts.json} ({os.path.getsize(opts.json)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
