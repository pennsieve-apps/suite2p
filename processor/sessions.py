"""
Finding the movies under INPUT_DIR and naming their sessions (conventions.md, "Sessions").

Inputs arrive as symlinks, possibly in port subfolders or merged at the root, so files are found
recursively, following symlinks, by name and content, never by fixed paths.

A movie ("unit") is one of:
- a TIFF (.tif/.tiff), or a series of TIFFs in one folder that differ only in a trailing
  `_<number>` (ScanImage writes `<name>_00001_00001.tif`, `<name>_00001_00002.tif`, ...);
- an HDF5 file (.h5/.hdf5) with a 3-D (frames x Ly x Lx) dataset (`data` or the largest one);
- an NWB file with a TwoPhotonSeries in `acquisition`;
- a Bruker folder (a PrairieView .xml next to .ome.tif files).
NWB files without a movie are companions: their identity (session_start_time, identifier,
subject, ...) is carried into session.json of the movie with the same session_id.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

TIFF_EXT = (".tif", ".tiff")
H5_EXT = (".h5", ".hdf5")
NWB_EXT = (".nwb",)
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
SUBSES_RE = re.compile(r"sub-([A-Za-z0-9-]+?)_ses-([A-Za-z0-9-]+)")
SERIES_RE = re.compile(r"^(.*?)(_\d+)+$")


@dataclass
class Unit:
    kind: str                       # tiff | h5 | nwb | bruker
    files: list[Path]               # the movie's files in order (a folder for bruker)
    name: str                       # stem used for the session id and the timestamps csv
    session_id: str = ""
    sheet: dict = field(default_factory=dict)   # the sessions.csv row (minus session_id/path)
    companion: Path | None = None   # an NWB without a movie for the same session
    h5_key: str | None = None
    nwb_series: str | None = None

    @property
    def path(self) -> Path:
        return self.files[0]


def walk(root: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith("."):
                out.append(Path(dirpath) / f)
    return out


def sanitise(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", s).strip("-") or "session"


def session_from_name(path: Path, stem: str | None = None) -> str:
    """sub-<a>_ses-<b> from the file name, else a parent folder, else the sanitised stem."""
    for part in [path.name] + [p.name for p in path.parents][:4]:
        m = SUBSES_RE.search(part)
        if m:
            ses = m.group(2)
            if UUID_RE.match(ses):
                ses = ses[:8]
            return f"sub-{m.group(1)}_ses-{ses}"
    return sanitise(stem if stem is not None else _stem(path))


def _stem(path: Path) -> str:
    name = path.name
    for ext in (".ome.tiff", ".ome.tif") + TIFF_EXT + H5_EXT + NWB_EXT:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _is_nwb_movie(path: Path) -> str | None:
    """Name of the first 3-D TwoPhotonSeries in acquisition, else None."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            acq = f.get("acquisition")
            if acq is None:
                return None
            for name in acq:
                g = acq[name]
                if not isinstance(g, h5py.Group):
                    continue
                nt = g.attrs.get("neurodata_type", b"")
                nt = nt.decode() if isinstance(nt, bytes) else str(nt)
                if nt == "TwoPhotonSeries" and "data" in g and g["data"].ndim >= 3:
                    return name
    except OSError:
        return None
    return None


def _h5_key(path: Path) -> str | None:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            if "nwb_version" in f.attrs or "nwb_version" in f:
                return None
            if "data" in f and isinstance(f["data"], h5py.Dataset) and f["data"].ndim >= 3:
                return "data"
            best, size = None, 0
            def visit(name, obj):
                nonlocal best, size
                if isinstance(obj, h5py.Dataset) and obj.ndim in (3, 4) and obj.size > size:
                    best, size = name, obj.size
            f.visititems(visit)
            return best
    except OSError:
        return None


def read_sheet(files: list[Path]) -> list[dict]:
    rows = []
    for p in files:
        if p.name == "sessions.csv":
            try:
                with open(p, newline="", encoding="utf-8-sig") as f:
                    for r in csv.DictReader(f):
                        if r.get("session_id") and r.get("path"):
                            rows.append({k: (v or "").strip() for k, v in r.items() if k})
            except (OSError, csv.Error):
                continue
    return rows


def _sheet_match(rows: list[dict], path: Path) -> dict | None:
    """The sessions.csv row whose `path` names this file or one of its folders."""
    parts = {path.name} | {p.name for p in path.parents}
    best = None
    for r in rows:
        rp = r["path"].replace("\\", "/").rstrip("/")
        if not rp:
            continue
        if str(path).replace("\\", "/").endswith("/" + rp) or Path(rp).name in parts:
            # prefer the most specific (longest) match
            if best is None or len(rp) > len(best["path"]):
                best = r
    return best


def discover(root: Path) -> tuple[list[Unit], list[str]]:
    """The movies under root, with their session ids, and warnings."""
    files = walk(root)
    rows = read_sheet(files)
    warnings: list[str] = []
    units: list[Unit] = []
    companions: list[Path] = []

    bruker_dirs = set()
    for p in files:
        if p.suffix.lower() == ".xml" and any(q.name.lower().endswith((".ome.tif", ".ome.tiff"))
                                              for q in p.parent.iterdir()):
            try:
                head = p.read_text(errors="ignore")[:2000]
            except OSError:
                head = ""
            if "PVScan" in head or "PVStateShard" in head or "Prairie" in head:
                bruker_dirs.add(p.parent)
    for d in sorted(bruker_dirs):
        units.append(Unit("bruker", [d], d.name))

    tiff_groups: dict[tuple[Path, str], list[Path]] = {}
    for p in files:
        low = p.name.lower()
        if p.parent in bruker_dirs:
            continue
        if low.endswith(TIFF_EXT):
            stem = _stem(p)
            m = SERIES_RE.match(stem)
            key = (p.parent, m.group(1) if m and m.group(1) else stem)
            tiff_groups.setdefault(key, []).append(p)
        elif low.endswith(H5_EXT):
            key = _h5_key(p)
            if key:
                units.append(Unit("h5", [p], _stem(p), h5_key=key))
        elif low.endswith(NWB_EXT):
            series = _is_nwb_movie(p)
            if series:
                units.append(Unit("nwb", [p], _stem(p), nwb_series=series))
            else:
                companions.append(p)
    from natsort import natsorted
    for (folder, base), group in sorted(tiff_groups.items()):
        group = natsorted(group, key=lambda q: q.name)
        name = base if len(group) > 1 else _stem(group[0])
        units.append(Unit("tiff", group, name))

    for u in units:
        row = _sheet_match(rows, u.path)
        if row:
            u.session_id = sanitise(row["session_id"])
            u.sheet = {k: v for k, v in row.items() if k not in ("session_id", "path")}
        else:
            u.session_id = session_from_name(u.path, u.name)

    # disambiguate duplicates (two movies naming the same session): suffix with the file stem
    seen: dict[str, int] = {}
    for u in units:
        seen[u.session_id] = seen.get(u.session_id, 0) + 1
    for u in units:
        if seen[u.session_id] > 1:
            new = f"{u.session_id}_{sanitise(u.name)}"
            warnings.append(f"{len([v for v in units if v.session_id == u.session_id])} movies share "
                            f"session_id {u.session_id}; this one is {new}")
            u.session_id = new

    for c in companions:
        row = _sheet_match(rows, c)
        sid = sanitise(row["session_id"]) if row else session_from_name(c)
        for u in units:
            if u.session_id == sid and u.companion is None:
                u.companion = c
    units.sort(key=lambda u: u.session_id)
    return units, warnings


def find_timestamps_csv(unit: Unit) -> Path | None:
    """`<stem>_timestamps.csv` next to the movie (or inside a Bruker folder)."""
    folder = unit.files[0] if unit.kind == "bruker" else unit.files[0].parent
    names = [f"{unit.name}_timestamps.csv", f"{_stem(unit.files[0])}_timestamps.csv"]
    for n in names:
        p = folder / n
        if p.is_file():
            return p
    if unit.kind == "bruker":
        p = folder / "timestamps.csv"
        if p.is_file():
            return p
    return None


def read_timestamps_csv(path: Path):
    """Times (s) from a `frame, time` (or `time`, or `timestamp`) CSV, in frame order."""
    import numpy as np
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (r.fieldnames or [])}
        tcol = next((cols[c] for c in ("time", "timestamp", "timestamps", "t") if c in cols), None)
        if tcol is None:
            raise ValueError(f"{path.name}: no time column (expected frame,time)")
        fcol = cols.get("frame")
        rows = [(int(float(row[fcol])) if fcol else i, float(row[tcol])) for i, row in enumerate(r)]
    rows.sort()
    return np.array([t for _, t in rows], float)
