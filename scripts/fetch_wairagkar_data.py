#!/usr/bin/env python3
"""Fetch, verify and extract the Wairagkar et al. 2025 Dryad dataset.

SOURCE
    Wairagkar, M., Card, N.S., Singer-Clark, T., et al. "An instantaneous
    voice-synthesis neuroprosthesis". Nature 644, 145-152 (2025).
    Paper DOI : 10.1038/s41586-025-09127-3  (https://doi.org/10.1038/s41586-025-09127-3)
    Data      : "Data for: An instantaneous voice synthesis neuroprosthesis"
    Dryad DOI : 10.5061/dryad.2280gb64f
    Landing   : https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f
    License   : CC0-1.0 (public domain dedication)

FILES (the authoritative sizes, digests, URLs and target paths live in
configs/data/wairagkar.yaml, which was filled from the Dryad API v2 on
2026-10-20 for dataset version 7; this script reads that file and hard-codes
none of them. The table below is documentation only -- if it ever disagrees
with the YAML, the YAML wins.)
    README.md                    4,635 bytes (4.6 KB)         dataset README
        sha-256 f9d05e8a8390579e025489146ecc4d2334c42a531c1e66fd222dffe147f69d70
    t15_control_experiment.zip   710,843,261 bytes (711 MB)   control-experiment neural data
        sha-256 c1ed4ea95f552901cb3bfe9220600a0657d0bba81caed57950c49ed5479f09bc
    t15_neural_data.zip          26,158,469,306 bytes (26.2 GB)
        sha-256 26e40f7aa693345a181ad175d9f0b3dbadadc45a82a9461711da2fdc0e1292dc
        neural-feature trials + pre-trained brain-to-voice decoders
    Total 26,869,317,202 bytes (26.87 GB, 25.0 GiB) as zips; the extracted size
    is not exposed by the Dryad API (the config records it as null).

TARGET LAYOUT (relative to the repository root, i.e. the parent of scripts/)
    data/raw/wairagkar/README.md
    data/raw/wairagkar/zips/t15_neural_data.zip
    data/raw/wairagkar/zips/t15_control_experiment.zip
    data/raw/wairagkar/t15_neural_data/          <- after --extract
    data/raw/wairagkar/t15_control_experiment/   <- after --extract
    A zip whose members are already wrapped in a top-level folder of that name
    is extracted into data/raw/wairagkar/; otherwise the script creates the
    folder itself, so the final layout is the same either way.

USAGE
    python scripts/fetch_wairagkar_data.py                   # dry run (default)
    python scripts/fetch_wairagkar_data.py --verify          # sha-256 of files already on disk
    python scripts/fetch_wairagkar_data.py --download t15_control_experiment.zip
    python scripts/fetch_wairagkar_data.py --download t15_neural_data.zip \\
            --i-confirm-large-download
    python scripts/fetch_wairagkar_data.py --extract t15_control_experiment.zip

    Dry run: lists each file, size, target path, whether it is present /
    partially downloaded / extracted, and checks free disk space against
    zip size + an extraction estimate. Nothing is written or downloaded.

    --download NAME [NAME ...]   streaming, resumable (HTTP Range) download to
        "<file>.part", then size + checksum verification, then rename. Re-run
        the same command to resume after an interruption. Any file larger than
        1 GB also needs --i-confirm-large-download, otherwise the script
        refuses. "all" selects every file. Existing files are never
        overwritten.
    --verify [NAME ...]          checksum files already present (e.g. after a
        manual browser download); no names means every file found.
    --extract NAME [NAME ...]    zip-slip-safe extraction (members that would
        land outside the target folder abort the run before anything is
        written); __MACOSX/ and .DS_Store entries are skipped. The zip is
        checksum-verified first (--skip-verify to skip) and is never deleted.
    Other options: --config PATH, --zips-dir PATH, --extract-dir PATH (put the
        zips and/or the extracted data on another volume, e.g. an external
        drive), --ignore-disk-check, --extract-size-factor X.

EXIT CODES
    0 success | 1 operation failed (network, checksum mismatch, bad zip, ...)
    2 usage / configuration error (including missing PyYAML) | 3 refused
    (large download not confirmed, or not enough disk space) | 130 interrupted

DISK SPACE
    The 26.2 GB t15_neural_data.zip plus its extracted copy needs roughly
    twice the zip size at peak (extracted size is unknown until the zip is
    present; the dry run assumes 1.0x the zip size, tune with
    --extract-size-factor). With ~47 GB free the zip itself fits, but zip +
    extraction does not, so put one of them on another volume with
    --zips-dir / --extract-dir, or free up space first.

AUTHENTICATION (checked 2026-10-20)
    Dryad's file-download endpoint (/api/v2/files/{id}/download) answers
    HTTP 401 "must have current bearer token" to anonymous requests, so the
    script needs a Dryad API token. Metadata endpoints remain anonymous.
    1. Sign in to https://datadryad.org with ORCID, open "My account", and
       create an API account (you get an application id and a secret).
    2. EITHER export a token (valid ~10 hours):
         export DRYAD_API_TOKEN="$(curl -s -X POST https://datadryad.org/oauth/token \\
             -d client_id=<application-id> -d client_secret=<secret> \\
             -d grant_type=client_credentials | python3 -c \\
             'import json,sys; print(json.load(sys.stdin)["access_token"])')"
       OR export DRYAD_CLIENT_ID and DRYAD_CLIENT_SECRET and let the script
       request (and, on expiry, renew) the token itself.
    The token is only sent over https (or loopback) and is dropped when Dryad
    redirects the download to another host (e.g. object storage). It is never
    printed. Do not commit credentials to the repository.
    API docs: https://github.com/datadryad/dryad-app/blob/main/documentation/apis/api_accounts.md

MANUAL DOWNLOAD IN A BROWSER (no token needed)
    1. Open https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f
    2. Download each file from the "Data files" list individually (the page's
       download links pass Dryad's browser check, which scripts cannot).
    3. Save the zips into data/raw/wairagkar/zips/ (create the folder) and
       README.md into data/raw/wairagkar/.
    4. Run "--verify" to compare the SHA-256 digests with the config, then
       "--extract NAME" for each zip.
    Alternatively, download elsewhere and point the script at it with
    --zips-dir PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import socket
import ssl
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

try:
    import yaml
except ImportError:
    sys.stderr.write("error: PyYAML is required (pip install pyyaml)\n")
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "wairagkar.yaml"

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_REFUSED, EXIT_INTERRUPTED = 0, 1, 2, 3, 130

DEFAULT_LARGE_BYTES = 1_000_000_000  # files strictly larger need explicit confirmation
CHUNK = 4 * 1024 * 1024
DISK_MARGIN = 2 * 1024**3  # headroom kept free on top of what we need
MAX_STALLED_RETRIES = 6  # consecutive failed attempts that made no progress
HASH_ALGOS = {"sha-256": "sha256", "sha256": "sha256", "sha-1": "sha1", "sha1": "sha1",
              "sha-512": "sha512", "sha512": "sha512", "md5": "md5"}
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
USER_AGENT = "fetch_wairagkar_data.py (research data fetch; python-urllib)"


class FetchError(Exception):
    """A clean, user-facing failure; `code` is the process exit status."""

    def __init__(self, message: str, code: int = EXIT_FAIL):
        super().__init__(message)
        self.code = code


class _Transient(Exception):
    """A network hiccup that is worth retrying (resuming)."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def fmt_size(n: float) -> str:
    n = float(n)
    if n >= 1e8:
        return f"{n / 1e9:.2f} GB ({n / 1024**3:.2f} GiB)"
    if n >= 1e5:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{int(n)} B"


def fmt_exact(n: int) -> str:
    return f"{fmt_size(n)} [{n:,} bytes]"


def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def show(p: Path) -> str:
    """Path for display: relative to the repo root when inside it."""
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def abspath(p: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(p))))


def nearest_existing(p: Path) -> Path:
    p = abspath(p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_bytes(p: Path) -> int:
    return shutil.disk_usage(nearest_existing(p)).free


def same_device(a: Path, b: Path) -> bool:
    return os.stat(nearest_existing(a)).st_dev == os.stat(nearest_existing(b)).st_dev


class Progress:
    """Throttled progress line on stderr (\\r-updated on a tty, sparse lines otherwise)."""

    def __init__(self, label: str, total: int, done: int = 0):
        self.label, self.total, self.base = label, max(total, 1), done
        self.t0 = self.last = time.monotonic()
        self.tty = sys.stderr.isatty()
        self.interval = 0.5 if self.tty else 30.0
        self.shown = False

    def update(self, done: int) -> None:
        now = time.monotonic()
        if now - self.last < self.interval:
            return
        self.last = now
        rate = (done - self.base) / max(now - self.t0, 1e-6)
        eta = (self.total - done) / rate if rate > 0 else 0
        line = (f"  {self.label}: {done / self.total:6.1%}  {done / 1e9:.2f}/{self.total / 1e9:.2f} GB"
                f"  {rate / 1e6:.1f} MB/s  ETA {fmt_eta(eta)}")
        sys.stderr.write(("\r" + line + "\033[K") if self.tty else line + "\n")
        sys.stderr.flush()
        self.shown = True

    def finish(self) -> None:
        if self.tty and self.shown:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()


def new_hasher(digest_type: str):
    algo = HASH_ALGOS.get(str(digest_type).lower())
    if algo is None:
        raise FetchError(f"unsupported digest type {digest_type!r} in config "
                         f"(supported: {', '.join(sorted(HASH_ALGOS))})", EXIT_USAGE)
    return hashlib.new(algo)


def hash_file(path: Path, digest_type: str, label: str | None = None, hasher=None) -> str:
    """Hex digest of `path` (streamed). Pass `hasher` to continue an existing one."""
    h = hasher or new_hasher(digest_type)
    total = path.stat().st_size
    prog = Progress(label, total) if (label and total > 256 * 1024**2) else None
    done = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if prog:
                prog.update(done)
    if prog:
        prog.finish()
    return h.hexdigest()


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class Entry:
    name: str
    description: str
    kind: str  # "archive" or "document"
    size: int
    digest: str
    digest_type: str
    url: str
    local: Path  # where the file itself lives
    extract: bool
    extract_root: Path | None  # folder the archive is extracted into
    extracted_dir: Path | None  # <extract_root>/<top-level folder name>
    extracted_size: int | None
    part: Path = field(init=False)

    def __post_init__(self):
        self.part = self.local.with_name(self.local.name + ".part")

    @property
    def stem(self) -> str:
        return self.extracted_dir.name if self.extracted_dir else self.local.stem


@dataclass
class Config:
    raw: dict
    entries: list[Entry]
    large_bytes: int
    path: Path

    @property
    def title(self) -> str:
        d = self.raw["dataset"]["dryad"]
        return (f"Wairagkar et al. 2025 -- Dryad doi:{d['doi']} (version {d['version_number']}, "
                f"{d['version_publication_date']}, {d['license']})")


def _safe_rel(value: object, what: str) -> PurePosixPath:
    p = PurePosixPath(str(value))
    if p.is_absolute() or ".." in p.parts or str(p) in ("", "."):
        raise FetchError(f"config: {what} must be a relative path without '..' (got {value!r})", EXIT_USAGE)
    return p


def load_config(path: Path, zips_dir: Path | None, extract_dir: Path | None) -> Config:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise FetchError(f"cannot read config {path}: {e}", EXIT_USAGE)
    except yaml.YAMLError as e:
        raise FetchError(f"config {path} is not valid YAML: {e}", EXIT_USAGE)
    try:
        _ = raw["dataset"]["dryad"]["doi"]  # presence check
        files = raw["files"]
        large = int(raw.get("download", {}).get("large_file_threshold_bytes", DEFAULT_LARGE_BYTES))
        entries = []
        for f in files:
            name = str(f["name"])
            if "/" in name or name in ("", ".", ".."):
                raise FetchError(f"config: bad file name {name!r}", EXIT_USAGE)
            kind = f.get("kind") or ("archive" if name.endswith(".zip") else "document")
            target = _safe_rel(f["target_path"], f"target_path of {name}")
            if kind == "archive":
                local = zips_dir / target.name if zips_dir else REPO_ROOT / target
            else:
                local = extract_dir / target.name if extract_dir else REPO_ROOT / target
            extract = bool(f.get("extract", False))
            extract_root = extracted = None
            if extract:
                extract_root = extract_dir or REPO_ROOT / _safe_rel(f["extract_to"], f"extract_to of {name}")
                extracted = extract_root / _safe_rel(f["extracted_dir"], f"extracted_dir of {name}").name
            esize = f.get("extracted_size_bytes")
            entries.append(Entry(
                name=name, description=" ".join(str(f.get("description", "")).split()), kind=kind,
                size=int(f["size_bytes"]), digest=str(f["digest"]).lower(), digest_type=str(f["digest_type"]),
                url=str(f["download_url"]), local=local, extract=extract, extract_root=extract_root,
                extracted_dir=extracted, extracted_size=None if esize is None else int(esize)))
            new_hasher(entries[-1].digest_type)  # fail early on unsupported digest types
    except (KeyError, TypeError, ValueError) as e:
        raise FetchError(f"config {path} is missing or has a malformed field: {e!r}", EXIT_USAGE)
    if not entries:
        raise FetchError(f"config {path} lists no files", EXIT_USAGE)
    return Config(raw=raw, entries=entries, large_bytes=large, path=path)


def select(cfg: Config, names: list[str]) -> list[Entry]:
    if "all" in names:
        return list(cfg.entries)
    chosen: list[Entry] = []
    for n in names:
        hits = [e for e in cfg.entries if e.name == n] or \
               [e for e in cfg.entries if Path(e.name).stem == n]
        if len(hits) != 1:
            valid = ", ".join(e.name for e in cfg.entries)
            raise FetchError(f"unknown file name {n!r}; valid names: {valid} (or 'all')", EXIT_USAGE)
        if hits[0] not in chosen:
            chosen.append(hits[0])
    return chosen


# --------------------------------------------------------------------------
# on-disk state and disk-space planning
# --------------------------------------------------------------------------
def junk_member(parts: list[str]) -> bool:
    return "__MACOSX" in parts or parts[-1] == ".DS_Store"


def exact_extracted_size(zip_path: Path) -> int | None:
    """Uncompressed size of the useful members, read from the zip directory."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return sum(i.file_size for i in zf.infolist()
                       if not i.is_dir() and not junk_member(i.filename.replace("\\", "/").split("/")))
    except (zipfile.BadZipFile, OSError):
        return None


def remaining_download(e: Entry) -> int:
    if e.local.exists():
        return 0
    return max(e.size - (e.part.stat().st_size if e.part.exists() else 0), 0)


def estimated_extraction(e: Entry, factor: float) -> tuple[int, str]:
    if e.extracted_size is not None:
        return e.extracted_size, "from config"
    if e.local.exists() and e.local.stat().st_size == e.size:
        exact = exact_extracted_size(e.local)
        if exact is not None:
            return exact, "exact, from the zip directory"
    return int(e.size * factor), f"estimate: {factor:g}x zip size (real size unknown until the zip is present)"


def disk_report(entries: list[Entry], factor: float) -> bool:
    """Print the disk-space check; returns True when everything fits."""
    dl = sum(remaining_download(e) for e in entries)
    ext_list = [(e, *estimated_extraction(e, factor)) for e in entries
                if e.extract and not (e.extracted_dir and e.extracted_dir.exists())]
    ex = sum(x[1] for x in ext_list)
    zdirs = sorted({e.local.parent for e in entries if e.kind == "archive"} or {e.local.parent for e in entries})
    xdirs = sorted({e.extract_root for e in entries if e.extract_root})
    zdir, xdir = zdirs[0], (xdirs[0] if xdirs else zdirs[0])
    free_z, free_x = free_bytes(zdir), free_bytes(xdir)
    together = same_device(zdir, xdir)
    print("Disk space check")
    if together:
        print(f"  free at {show(nearest_existing(xdir))}: {fmt_size(free_z)}")
    else:
        print(f"  free for zips    ({show(nearest_existing(zdir))}): {fmt_size(free_z)}")
        print(f"  free for extract ({show(nearest_existing(xdir))}): {fmt_size(free_x)}")
    print(f"  still to download : {fmt_size(dl)}")
    for e, n, how in ext_list:
        print(f"  extracting {e.name}: {fmt_size(n)}  ({how})")
    print(f"  safety margin     : {fmt_size(DISK_MARGIN)}")
    ok = True
    if together:
        need = dl + ex + DISK_MARGIN
        fits = need <= free_z
        verdict = "fits" if fits else f"DOES NOT FIT (short by {fmt_size(need - free_z)})"
        print(f"  download + extract (zips kept): {fmt_size(dl + ex)} + margin = {fmt_size(need)} -> {verdict}")
        ok &= fits
        if len(ext_list) > 1:
            for e, n, _ in ext_list:
                one = remaining_download(e) + n + DISK_MARGIN
                v = "fits" if one <= free_z else f"DOES NOT FIT (short by {fmt_size(one - free_z)})"
                print(f"    {e.name} alone: {fmt_size(remaining_download(e) + n)} + margin -> {v}")
        zips_only = dl + DISK_MARGIN <= free_z
        if not ok:
            print()
            print("  *** WARNING: NOT ENOUGH DISK SPACE for download + extraction on this volume. ***")
            if zips_only and dl:
                print(f"  The downloads alone need {fmt_size(dl)} and would still fit, but you could not extract them here.")
            elif not zips_only:
                print("  Even the downloads alone do not fit with the safety margin.")
            print("  Options: free up space; or keep zips and/or extracted data on a bigger volume with")
            print("  --zips-dir PATH / --extract-dir PATH (e.g. an external drive). This script never deletes zips.")
    else:
        fit_z = dl + DISK_MARGIN <= free_z
        fit_x = ex + DISK_MARGIN <= free_x
        print(f"  zips volume   : need {fmt_size(dl + DISK_MARGIN)} -> {'fits' if fit_z else 'DOES NOT FIT'}")
        print(f"  extract volume: need {fmt_size(ex + DISK_MARGIN)} -> {'fits' if fit_x else 'DOES NOT FIT'}")
        ok = fit_z and fit_x
        if not ok:
            print("  *** WARNING: NOT ENOUGH DISK SPACE on at least one volume. ***")
    return ok


def dry_run(cfg: Config, args) -> int:
    print(cfg.title)
    print(f"Config: {show(cfg.path)}")
    print("Mode: DRY RUN -- nothing is downloaded, written or deleted. "
          "Use --download NAME ... to fetch, --verify to checksum, --extract NAME to unpack.")
    token = bool(os.environ.get("DRYAD_API_TOKEN") or
                 (os.environ.get("DRYAD_CLIENT_ID") and os.environ.get("DRYAD_CLIENT_SECRET")))
    print("Dryad API credentials in environment: " + ("yes" if token else
          "NO (downloads need a token -- see the script docstring; or download in a browser)"))
    print()
    for i, e in enumerate(cfg.entries, 1):
        flag = "  [LARGE: needs --i-confirm-large-download]" if e.size > cfg.large_bytes else ""
        print(f"[{i}/{len(cfg.entries)}] {e.name}{flag}")
        print(f"    size    : {fmt_exact(e.size)}")
        print(f"    digest  : {e.digest_type} {e.digest}")
        print(f"    source  : {e.url}")
        print(f"    target  : {show(e.local)}")
        if e.local.exists():
            have = e.local.stat().st_size
            if have == e.size:
                status = "PRESENT, size matches (checksum not computed here; run --verify)"
            else:
                status = f"PRESENT BUT SIZE DIFFERS ({have:,} vs expected {e.size:,} bytes) -- do not use; re-fetch"
        elif e.part.exists():
            have = e.part.stat().st_size
            status = f"PARTIAL download {have:,}/{e.size:,} bytes ({have / e.size:.1%}); --download resumes it"
        else:
            status = "MISSING (not downloaded)"
        print(f"    status  : {status}")
        if e.extract:
            n, how = estimated_extraction(e, args.extract_size_factor)
            done = "present" if e.extracted_dir.exists() else "not extracted"
            print(f"    extracts: {show(e.extracted_dir)}/  ({done}); size {fmt_size(n)} [{how}]")
        print(f"    note    : {e.description}")
    print()
    disk_report(cfg.entries, args.extract_size_factor)
    return EXIT_OK


# --------------------------------------------------------------------------
# authentication and HTTP
# --------------------------------------------------------------------------
def _may_send_token(url: str) -> bool:
    u = urllib.parse.urlsplit(url)
    return u.scheme == "https" or (u.hostname or "") in LOOPBACK


class _RedirectKeepAuthSameHost(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never forward the Authorization header to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            same = urllib.parse.urlsplit(req.full_url).netloc.lower() == urllib.parse.urlsplit(newurl).netloc.lower()
            if not same or not _may_send_token(newurl):
                for k in list(new.headers):
                    if k.lower() == "authorization":
                        del new.headers[k]
        return new


OPENER = urllib.request.build_opener(_RedirectKeepAuthSameHost)


class TokenProvider:
    """Bearer token from DRYAD_API_TOKEN, or minted from DRYAD_CLIENT_ID / DRYAD_CLIENT_SECRET."""

    def __init__(self, cfg: Config):
        d = cfg.raw.get("download", {})
        self.token_url = d.get("token_url", "https://datadryad.org/oauth/token")
        self.token = os.environ.get(d.get("token_env_var", "DRYAD_API_TOKEN"), "").strip() or None
        self.cid = os.environ.get(d.get("client_id_env_var", "DRYAD_CLIENT_ID"), "").strip()
        self.secret = os.environ.get(d.get("client_secret_env_var", "DRYAD_CLIENT_SECRET"), "").strip()
        self.minted = False
        if not self.token and self.cid and self.secret:
            self.refresh()

    @property
    def can_refresh(self) -> bool:
        return bool(self.cid and self.secret)

    def refresh(self) -> bool:
        if not self.can_refresh:
            return False
        body = urllib.parse.urlencode({"client_id": self.cid, "client_secret": self.secret,
                                       "grant_type": "client_credentials"}).encode()
        req = urllib.request.Request(self.token_url, data=body, headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8", "User-Agent": USER_AGENT})
        try:
            with OPENER.open(req, timeout=60) as r:
                self.token = json.load(r)["access_token"]
        except urllib.error.HTTPError as e:
            raise FetchError(f"could not obtain a Dryad token (HTTP {e.code}); check DRYAD_CLIENT_ID / "
                             f"DRYAD_CLIENT_SECRET", EXIT_FAIL)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            raise FetchError(f"could not obtain a Dryad token: {e}", EXIT_FAIL)
        self.minted = True
        return True


AUTH_HELP = (
    "Dryad file downloads need an API token (anonymous requests get HTTP 401).\n"
    "  Set DRYAD_API_TOKEN (a bearer token, valid ~10 h), or DRYAD_CLIENT_ID + DRYAD_CLIENT_SECRET.\n"
    "  How to get them: see the docstring of this script / Dryad's api_accounts.md.\n"
    "  Or download the file in a browser from the dataset landing page, put it at the target path,\n"
    "  and run --verify."
)


def open_stream(url: str, tokens: TokenProvider, start: int):
    """GET `url` (optionally from byte `start`); returns the open response."""
    for attempt in (1, 2):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*",
                                                   "Accept-Encoding": "identity"})
        if start:
            req.add_header("Range", f"bytes={start}-")
        if tokens.token and _may_send_token(url):
            req.add_header("Authorization", f"Bearer {tokens.token}")
        try:
            return OPENER.open(req, timeout=60)
        except urllib.error.HTTPError as e:
            code = e.code
            retry_after = e.headers.get("Retry-After") if e.headers else None
            e.close()
            if code == 401 and attempt == 1 and tokens.can_refresh and tokens.refresh():
                continue  # token expired -> renewed, retry once
            if code in (401, 403):
                have = "the token in use was rejected or expired" if tokens.token else "no token was supplied"
                raise FetchError(f"Dryad refused the download (HTTP {code}; {have}).\n{AUTH_HELP}", EXIT_FAIL)
            if code == 404:
                raise FetchError(f"HTTP 404 for {url}: the file is no longer at this URL; re-query the Dryad API "
                                 f"and update the config", EXIT_FAIL)
            if code == 416:
                raise FetchError(f"HTTP 416: the partial file is not a prefix of the file on Dryad; delete the "
                                 f".part file and re-run", EXIT_FAIL)
            if code == 429 or code >= 500:
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 120))
                raise _Transient(f"HTTP {code}")
            raise FetchError(f"unexpected HTTP {code} from {url}", EXIT_FAIL)
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                raise FetchError(f"TLS certificate verification failed for {url}: {e.reason}", EXIT_FAIL)
            raise _Transient(str(e.reason))
        except (TimeoutError, socket.timeout, ConnectionError, ssl.SSLError, http.client.HTTPException) as e:
            raise _Transient(f"{type(e).__name__}: {e}")
    raise FetchError("authentication failed", EXIT_FAIL)  # pragma: no cover


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def check_file(e: Entry, label: bool = True) -> tuple[str, str]:
    """Returns (status, detail); status in OK / MISSING / SIZE / MISMATCH."""
    if not e.local.exists():
        return "MISSING", ""
    have = e.local.stat().st_size
    if have != e.size:
        return "SIZE", f"{have:,} bytes on disk, expected {e.size:,}"
    got = hash_file(e.local, e.digest_type, f"hashing {e.name}" if label else None)
    if got != e.digest:
        return "MISMATCH", f"{e.digest_type} {got} != expected {e.digest}"
    return "OK", f"{e.digest_type} {got}"


def do_verify(cfg: Config, names: list[str]) -> int:
    explicit = bool(names)
    entries = select(cfg, names) if explicit else cfg.entries
    bad = missing = good = 0
    for e in entries:
        status, detail = check_file(e)
        if status == "OK":
            good += 1
            print(f"OK        {e.name}  ({e.size:,} bytes, {detail})")
        elif status == "MISSING":
            missing += 1
            extra = f" (partial download exists: {e.part.stat().st_size:,} bytes)" if e.part.exists() else ""
            print(f"MISSING   {e.name}  not at {show(e.local)}{extra}")
        else:
            bad += 1
            print(f"{'BAD SIZE' if status == 'SIZE' else 'CHECKSUM MISMATCH':<9} {e.name}  {detail}")
    print(f"\nverified OK: {good}, missing: {missing}, bad: {bad}")
    if bad:
        print("Do not use files marked bad; delete them and fetch again.", file=sys.stderr)
        return EXIT_FAIL
    if good == 0:
        print("error: nothing to verify -- none of the selected files is present.", file=sys.stderr)
        return EXIT_FAIL
    if explicit and missing:
        return EXIT_FAIL
    return EXIT_OK


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
class _PartState:
    """Running hash of the bytes in the .part file, kept across retries."""

    def __init__(self, digest_type: str):
        self.digest_type = digest_type
        self.reset()

    def reset(self):
        self.hasher = new_hasher(self.digest_type)
        self.hashed = 0


def _attempt(e: Entry, tokens: TokenProvider, st: _PartState) -> None:
    existing = e.part.stat().st_size if e.part.exists() else 0
    if existing > e.size:
        print(f"  {e.part.name} is larger than expected ({existing:,} > {e.size:,}); restarting")
        e.part.unlink()
        existing = 0
    if existing != st.hashed:  # first attempt of this run, or state lost: re-hash what is on disk
        st.reset()
        if existing:
            hash_file(e.part, e.digest_type, f"re-reading partial {e.name}", st.hasher)
            st.hashed = existing
    if existing < e.size:
        resp = open_stream(e.url, tokens, existing)
        with resp:
            status = resp.status
            if existing and status == 200:
                print("  server ignored the Range request; restarting from byte 0")
                existing = 0
                st.reset()
            elif existing and status == 206:
                cr = resp.headers.get("Content-Range", "")
                if not cr.startswith(f"bytes {existing}-"):
                    raise FetchError(f"server returned unexpected Content-Range {cr!r} when resuming at {existing}")
            elif status not in (200, 206):
                raise FetchError(f"unexpected HTTP status {status}")
            clen = resp.headers.get("Content-Length")
            if clen and clen.isdigit() and int(clen) != e.size - existing:
                raise FetchError(
                    f"server will send {int(clen):,} bytes but {e.size - existing:,} were expected for {e.name}; "
                    f"the file on Dryad may have changed -- re-query the API and update the config")
            if existing:
                print(f"  resuming at byte {existing:,} ({existing / e.size:.1%})")
            prog = Progress(e.name, e.size, existing)
            e.part.parent.mkdir(parents=True, exist_ok=True)  # only once Dryad has agreed to send data
            try:
                with open(e.part, "ab" if existing else "wb") as out:
                    done = existing
                    while True:
                        try:
                            chunk = resp.read(CHUNK)
                        except (TimeoutError, socket.timeout, ConnectionError, ssl.SSLError,
                                http.client.HTTPException) as err:
                            raise _Transient(f"{type(err).__name__}: {err}")
                        if not chunk:
                            break
                        out.write(chunk)
                        st.hasher.update(chunk)
                        st.hashed += len(chunk)
                        done += len(chunk)
                        prog.update(done)
            finally:
                prog.finish()
    size = e.part.stat().st_size
    if size < e.size:
        raise _Transient(f"connection ended early at {size:,}/{e.size:,} bytes")
    if size > e.size:
        e.part.unlink()
        raise FetchError(f"{e.name}: received {size:,} bytes, expected {e.size:,}; partial file deleted, re-run")


def download_one(e: Entry, tokens: TokenProvider) -> None:
    print(f"\n== {e.name}  ({fmt_exact(e.size)})")
    print(f"   -> {show(e.local)}")
    if e.local.exists():
        status, detail = check_file(e)
        if status == "OK":
            print("   already present and checksum-verified; nothing to do")
            return
        raise FetchError(f"{show(e.local)} already exists but does not match the expected size/checksum "
                         f"({detail}). Move or delete it, then re-run.")
    st = _PartState(e.digest_type)
    stalled = 0
    while True:
        before = e.part.stat().st_size if e.part.exists() else 0
        try:
            _attempt(e, tokens, st)
            break
        except _Transient as err:
            after = e.part.stat().st_size if e.part.exists() else 0
            stalled = 0 if after > before else stalled + 1
            if stalled >= MAX_STALLED_RETRIES:
                raise FetchError(f"{e.name}: giving up after {stalled} attempts without progress ({err}). "
                                 f"The partial file is kept; re-run to resume.")
            wait = min(2 ** stalled, 60)
            print(f"  network problem ({err}); retrying in {wait}s (resumes from {after:,} bytes)", file=sys.stderr)
            time.sleep(wait)
        except OSError as err:
            raise FetchError(f"local I/O error while writing {show(e.part)}: {err}")
    got = st.hasher.hexdigest()
    if got != e.digest:
        e.part.unlink()
        raise FetchError(f"{e.name}: CHECKSUM MISMATCH after download ({e.digest_type} {got}, expected {e.digest}). "
                         f"The corrupt .part file was deleted; re-run to download again.")
    os.replace(e.part, e.local)
    print(f"   downloaded and verified ({e.digest_type} {got})")


def do_download(cfg: Config, names: list[str], args) -> int:
    chosen = select(cfg, names)
    large = [e for e in chosen if e.size > cfg.large_bytes]
    if large and not args.i_confirm_large_download:
        print("REFUSED: these files are larger than 1 GB:", file=sys.stderr)
        for e in large:
            print(f"  {e.name}: {fmt_size(e.size)}", file=sys.stderr)
        print("Re-run with --i-confirm-large-download to proceed (see the dry run for the disk-space check).",
              file=sys.stderr)
        return EXIT_REFUSED
    need = sum(remaining_download(e) for e in chosen)
    if need:
        free = free_bytes(chosen[0].local.parent)
        if need + DISK_MARGIN > free and not args.ignore_disk_check:
            print(f"REFUSED: need {fmt_size(need)} (+ {fmt_size(DISK_MARGIN)} margin) but only {fmt_size(free)} "
                  f"is free at {show(nearest_existing(chosen[0].local.parent))}. Free space, use --zips-dir PATH "
                  f"on a larger volume, or pass --ignore-disk-check.", file=sys.stderr)
            return EXIT_REFUSED
    tokens = TokenProvider(cfg)
    if not tokens.token:
        print("note: no Dryad token in the environment; trying anonymously (Dryad currently answers 401).",
              file=sys.stderr)
    for e in chosen:
        download_one(e, tokens)
    unextracted = [e for e in chosen if e.extract and not e.extracted_dir.exists()]
    if unextracted:
        print()
        disk_report(unextracted, args.extract_size_factor)
        print("Next: " + " ; ".join(f"--extract {e.name}" for e in unextracted))
    return EXIT_OK


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------
def _member_parts(name: str) -> list[str]:
    """Split a zip member name into safe path parts or raise FetchError."""
    if "\x00" in name:
        raise FetchError(f"unsafe zip member name (NUL byte): {name!r}")
    norm = name.replace("\\", "/")
    if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
        raise FetchError(f"unsafe zip member (absolute path): {name!r}")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if ".." in parts:
        raise FetchError(f"unsafe zip member (path traversal): {name!r}")
    return parts


def extract_one(e: Entry, args) -> None:
    print(f"\n== extracting {e.name}")
    if not e.extract:
        raise FetchError(f"{e.name} is not an archive (extract: false in the config)", EXIT_USAGE)
    if not e.local.exists():
        raise FetchError(f"{show(e.local)} is not present; --download it (or save it there manually) first")
    if not args.skip_verify:
        status, detail = check_file(e)
        if status != "OK":
            raise FetchError(f"{e.name} failed verification ({status}: {detail}); not extracting. "
                             f"Use --skip-verify to override.")
        print("   checksum OK")
    try:
        zf = zipfile.ZipFile(e.local)
    except (zipfile.BadZipFile, OSError) as err:
        raise FetchError(f"{show(e.local)} is not a readable zip file: {err}")
    with zf:
        members = []  # (info, parts)
        for info in zf.infolist():
            parts = _member_parts(info.filename)
            if not parts or junk_member(parts):
                continue
            if (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
                raise FetchError(f"zip member {info.filename!r} is a symbolic link; refusing to extract")
            members.append((info, parts))
        if not members:
            raise FetchError(f"{e.name} contains no extractable files")
        tops = {parts[0] for _, parts in members}
        wrapped = tops == {e.stem} and any(len(parts) > 1 for _, parts in members)
        base = e.extract_root if wrapped else e.extracted_dir
        print("   layout: " + (f"zip already has the top-level folder {e.stem!r}" if wrapped
                               else f"zip has no single top-level {e.stem!r} folder; creating it"))
        print(f"   into  : {show(e.extracted_dir)}/")
        base_real = os.path.realpath(base)
        plan = []
        for info, parts in members:
            dest = base.joinpath(*parts)
            real = os.path.realpath(dest)
            if os.path.commonpath([real, base_real]) != base_real and real != base_real:
                raise FetchError(f"unsafe zip member (escapes the target directory): {info.filename!r}")
            plan.append((info, dest))
        total = sum(i.file_size for i, _ in plan if not i.is_dir())
        free = free_bytes(base)
        print(f"   {sum(1 for i, _ in plan if not i.is_dir()):,} files, {fmt_size(total)} uncompressed; "
              f"{fmt_size(free)} free")
        if total + DISK_MARGIN > free and not args.ignore_disk_check:
            raise FetchError(f"not enough disk space to extract: need {fmt_size(total)} (+ {fmt_size(DISK_MARGIN)} "
                             f"margin), free {fmt_size(free)}. Free space, use --extract-dir PATH on a larger "
                             f"volume, or pass --ignore-disk-check.", EXIT_REFUSED)
        base.mkdir(parents=True, exist_ok=True)
        prog = Progress(f"extract {e.name}", total)
        done = wrote = skipped = 0
        try:
            for info, dest in plan:
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                if dest.is_file() and dest.stat().st_size == info.file_size:
                    skipped += 1
                    done += info.file_size
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".partial")
                try:
                    with zf.open(info) as src, open(tmp, "wb") as out:
                        while True:
                            chunk = src.read(CHUNK)
                            if not chunk:
                                break
                            out.write(chunk)
                            done += len(chunk)
                            prog.update(done)
                    os.replace(tmp, dest)
                except (zipfile.BadZipFile, OSError) as err:
                    if tmp.exists():
                        tmp.unlink()
                    raise FetchError(f"failed while extracting {info.filename!r}: {err}")
                wrote += 1
        finally:
            prog.finish()
    print(f"   done: {wrote:,} files written, {skipped:,} already present (same size) -> {show(e.extracted_dir)}/")


def do_extract(cfg: Config, names: list[str], args) -> int:
    for e in select(cfg, names):
        extract_one(e, args)
    return EXIT_OK


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fetch_wairagkar_data.py", allow_abbrev=False,
        description="Fetch/verify/extract the Wairagkar et al. 2025 Dryad dataset (doi:10.5061/dryad.2280gb64f). "
                    "With no action flag it performs a dry run. See the module docstring for details.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--download", nargs="+", metavar="NAME", help="download these files ('all' = every file)")
    g.add_argument("--verify", nargs="*", metavar="NAME", help="verify checksums of files already on disk")
    g.add_argument("--extract", nargs="+", metavar="NAME", help="extract these zip files")
    p.add_argument("--i-confirm-large-download", action="store_true",
                   help="required to download any file larger than 1 GB")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML manifest (default: %(default)s)")
    p.add_argument("--zips-dir", type=Path, help="store/read the zip files here instead of the config's path")
    p.add_argument("--extract-dir", type=Path, help="extract into (and keep README in) this folder instead")
    p.add_argument("--ignore-disk-check", action="store_true", help="do not refuse when free space looks too low")
    p.add_argument("--skip-verify", action="store_true", help="--extract: skip the checksum check of the zip")
    p.add_argument("--extract-size-factor", type=float, default=1.0,
                   help="assumed extracted/zip size ratio when the real size is unknown (default: %(default)s)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.extract_size_factor <= 0:
            raise FetchError("--extract-size-factor must be > 0", EXIT_USAGE)
        zips = abspath(args.zips_dir) if args.zips_dir else None
        xdir = abspath(args.extract_dir) if args.extract_dir else None
        cfg = load_config(abspath(args.config), zips, xdir)
        if args.download is not None:
            return do_download(cfg, args.download, args)
        if args.verify is not None:
            return do_verify(cfg, args.verify)
        if args.extract is not None:
            return do_extract(cfg, args.extract, args)
        return dry_run(cfg, args)
    except FetchError as err:
        print(f"error: {err}", file=sys.stderr)
        return err.code
    except KeyboardInterrupt:
        print("\ninterrupted -- partial downloads are kept; re-run the same command to resume.", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
