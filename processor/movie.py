"""
Reading two-photon movies in batches and writing Suite2p's per-plane binary files.

Suite2p 1.1.0's own NWB and sbx readers still have the pre-1.0 call signature (they fail when
run_s2p calls them), and its TIFF reader cannot read a time window, so this app reads TIFF,
HDF5 and NWB itself (tifffile / h5py, in batches; never the whole movie) and writes the same
int16 binary Suite2p writes (`data.bin`, frames x Ly x Lx; uint16 data are halved, as Suite2p
does). Bruker folders go through Suite2p's own `ome_to_binary`.

Frame rate, planes, channels and frame times come from (in order) the parameters, a
`<stem>_timestamps.csv`, the NWB TwoPhotonSeries (timestamps or starting_time + rate), the
ScanImage metadata of the TIFF, other TIFF metadata (ImageJ `finterval`, a JSON description with
`frame_rate_hz` / `fps`), or HDF5 attributes (`fs`, `frame_rate`, `rate`).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from processor.sessions import Unit, find_timestamps_csv, read_timestamps_csv

log = logging.getLogger("processor")


class MovieError(RuntimeError):
    pass


@dataclass
class MovieInfo:
    n_pages: int                       # frames as stored (planes and channels interleaved)
    Ly: int
    Lx: int
    dtype: str
    nplanes: int = 1
    nchannels: int = 1
    flyback: int = 0                   # ScanImage flyback frames per volume (not processed)
    fs_volume: float | None = None     # volumes (= frames per plane) per second
    fs_source: str = ""
    page_times: np.ndarray | None = None     # one time per stored page, or per volume
    times_per: str = ""                # "page" | "volume" | ""
    t0: float = 0.0
    time_source: str = "frames/fs"
    identity: dict = field(default_factory=dict)  # NWB session identity, when the movie is NWB
    indicator: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# sources


class TiffSource:
    def __init__(self, files: list[Path]):
        import tifffile
        self.files = files
        self.counts = []
        self._tifs = []
        for f in files:
            t = tifffile.TiffFile(f)
            self._tifs.append(t)
            n = len(t.pages)
            if n == 1:
                shp = t.pages[0].shape
                if len(shp) == 3 and t.pages[0].samplesperpixel == 1:
                    n = shp[0]                       # a single page holding a stack
            self.counts.append(n)
        p0 = self._tifs[0].pages[0]
        shape = p0.shape[-2:]
        self.Ly, self.Lx = int(shape[0]), int(shape[1])
        self.dtype = str(p0.dtype)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self.n_pages = int(self.offsets[-1])

    def read(self, a: int, b: int) -> np.ndarray:
        out = []
        for i, t in enumerate(self._tifs):
            lo, hi = self.offsets[i], self.offsets[i + 1]
            s, e = max(a, lo), min(b, hi)
            if s >= e:
                continue
            if len(t.pages) == 1 and self.counts[i] > 1:
                arr = t.pages[0].asarray()[s - lo:e - lo]
            else:
                arr = t.asarray(key=range(int(s - lo), int(e - lo)))
            if arr.ndim == 2:
                arr = arr[None]
            out.append(arr.reshape(-1, self.Ly, self.Lx))
        return np.concatenate(out, axis=0)

    def metadata(self, info: MovieInfo) -> None:
        t = self._tifs[0]
        si = None
        try:
            si = t.scanimage_metadata
        except Exception:
            si = None
        if si:
            parse_scanimage(si.get("FrameData", {}) or {}, info)
            return
        desc = ""
        try:
            desc = t.pages[0].description or ""
        except Exception:
            pass
        if desc.strip().startswith("{"):
            try:
                d = json.loads(desc)
                for k in ("frame_rate_hz", "fps", "frame_rate", "fs", "rate"):
                    if k in d and float(d[k]) > 0:
                        info.fs_volume, info.fs_source = float(d[k]), f"TIFF description ({k})"
                        break
            except (ValueError, TypeError):
                pass
        if info.fs_volume is None and t.imagej_metadata:
            ij = t.imagej_metadata
            if ij.get("finterval"):
                info.fs_volume, info.fs_source = 1.0 / float(ij["finterval"]), "ImageJ finterval"
            elif ij.get("fps"):
                info.fs_volume, info.fs_source = float(ij["fps"]), "ImageJ fps"
            if ij.get("channels") and int(ij["channels"]) > 1:
                info.nchannels = int(ij["channels"])
            if ij.get("slices") and int(ij["slices"]) > 1 and ij.get("frames", 1) > 1:
                info.nplanes = int(ij["slices"])

    def close(self):
        for t in self._tifs:
            t.close()


def parse_scanimage(fd: dict, info: MovieInfo) -> None:
    """Planes, channels, flyback frames and frame rate from ScanImage's FrameData."""
    def g(k, default=None):
        return fd.get(k, default)

    def on(v):
        return v in (True, 1, "true", "1")
    if on(g("SI.hRoiManager.mroiEnable")):
        info.notes.append("ScanImage multi-ROI (mesoscope) recordings are processed as one field of view")
    chans = g("SI.hChannels.channelSave", 1)
    nchan = len(chans) if isinstance(chans, (list, tuple)) else 1
    fastz = on(g("SI.hFastZ.enable")) or on(g("SI.hStackManager.enable"))
    nslices = int(g("SI.hStackManager.actualNumSlices", g("SI.hStackManager.numSlices", 1)) or 1)
    flyback = int(g("SI.hFastZ.numDiscardFlybackFrames", 0) or 0) if fastz else 0
    nplanes = nslices if fastz and nslices > 1 else 1
    info.nchannels, info.nplanes = nchan, nplanes
    info.flyback = flyback if nplanes > 1 else 0
    vol = g("SI.hRoiManager.scanVolumeRate")
    frm = g("SI.hRoiManager.scanFrameRate")
    rate = vol if (nplanes > 1 and vol) else frm
    if rate:
        info.fs_volume, info.fs_source = float(rate), "ScanImage metadata"


class H5Source:
    def __init__(self, path: Path, key: str):
        import h5py
        self._f = h5py.File(path, "r")
        self.ds = self._f[key]
        shp = self.ds.shape
        if self.ds.ndim == 4:      # frames x planes (or channels) x Ly x Lx
            self.n_pages = shp[0] * shp[1]
        else:
            self.n_pages = shp[0]
        self.Ly, self.Lx = int(shp[-2]), int(shp[-1])
        self.dtype = str(self.ds.dtype)
        self.key = key

    def read(self, a, b):
        if self.ds.ndim == 4:
            k = self.ds.shape[1]
            v0, v1 = a // k, -(-b // k)
            arr = self.ds[v0:v1].reshape(-1, self.Ly, self.Lx)
            return arr[a - v0 * k: b - v0 * k]
        return self.ds[a:b]

    def metadata(self, info: MovieInfo):
        for obj in (self.ds, self._f):
            for k in ("fs", "frame_rate", "frame_rate_hz", "rate", "fps"):
                if k in obj.attrs:
                    try:
                        v = float(np.asarray(obj.attrs[k]).ravel()[0])
                    except (TypeError, ValueError):
                        continue
                    if v > 0:
                        info.fs_volume, info.fs_source = v, f"HDF5 attribute {k}"
                        return

    def close(self):
        self._f.close()


def _s(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.shape == ():
        return _s(v[()])
    return v if isinstance(v, str) else str(v)


def nwb_identity(f) -> dict:
    """session_start_time, identifier, session_description, timestamps_reference_time, session_id,
    subject (dict), and the imaging plane's indicator, from an open h5py NWB file."""
    ident = {}
    for k in ("session_start_time", "identifier", "session_description", "timestamps_reference_time"):
        if k in f:
            ident[k] = _s(f[k][()])
    g = f.get("general")
    if g is not None:
        for k in ("session_id", "lab", "institution", "experimenter", "experiment_description"):
            if k in g:
                v = g[k][()]
                ident[k] = [_s(x) for x in v] if isinstance(v, np.ndarray) and v.ndim else _s(v)
        if "subject" in g:
            ident["subject"] = {k: _s(g["subject"][k][()]) for k in g["subject"]
                                if hasattr(g["subject"][k], "shape") and g["subject"][k].shape == ()}
        op = g.get("optophysiology")
        if op is not None:
            for name in op:
                pl = op[name]
                if hasattr(pl, "keys") and "indicator" in pl:
                    ident["indicator"] = _s(pl["indicator"][()])
                    if "location" in pl:
                        ident["location"] = _s(pl["location"][()])
                    if "imaging_rate" in pl:
                        try:
                            ident["imaging_rate"] = float(pl["imaging_rate"][()])
                        except (TypeError, ValueError):
                            pass
                    break
    return ident


class NwbSource:
    def __init__(self, path: Path, series: str):
        import h5py
        self._f = h5py.File(path, "r")
        self.g = self._f["acquisition"][series]
        self.ds = self.g["data"]
        if self.ds.ndim != 3:
            raise MovieError(f"{path.name}: TwoPhotonSeries {series} has {self.ds.ndim} dimensions; "
                             "only single-plane (frames x Ly x Lx) series are supported")
        self.n_pages = self.ds.shape[0]
        self.Ly, self.Lx = int(self.ds.shape[1]), int(self.ds.shape[2])
        self.dtype = str(self.ds.dtype)
        self.series = series

    def read(self, a, b):
        return self.ds[a:b]

    def metadata(self, info: MovieInfo):
        info.identity = nwb_identity(self._f)
        info.indicator = info.identity.get("indicator")
        if "timestamps" in self.g:
            ts = np.asarray(self.g["timestamps"][:], float)
            if len(ts) == self.n_pages:
                info.page_times, info.times_per = ts, "page"
                info.time_source = f"NWB {self.series}.timestamps"
                return
        if "starting_time" in self.g:
            st = self.g["starting_time"]
            info.t0 = float(st[()])
            rate = st.attrs.get("rate")
            if rate:
                info.fs_volume, info.fs_source = float(rate), f"NWB {self.series}.rate"
                info.time_source = f"NWB {self.series}.starting_time + frame / rate"

    def close(self):
        self._f.close()


def open_source(unit: Unit):
    if unit.kind == "tiff":
        return TiffSource(unit.files)
    if unit.kind == "h5":
        return H5Source(unit.files[0], unit.h5_key)
    if unit.kind == "nwb":
        return NwbSource(unit.files[0], unit.nwb_series)
    raise MovieError(f"no reader for {unit.kind}")


# ---------------------------------------------------------------------------------------------
# metadata


def describe(unit: Unit, src, fs_param: float, nplanes_param: int, nchannels_param: int) -> MovieInfo:
    info = MovieInfo(n_pages=src.n_pages, Ly=src.Ly, Lx=src.Lx, dtype=src.dtype)
    src.metadata(info)
    if nplanes_param > 0:
        info.nplanes, info.flyback = nplanes_param, 0
    if nchannels_param > 0:
        info.nchannels = nchannels_param
    if unit.kind == "nwb" and (info.nplanes > 1 or info.nchannels > 1):
        info.notes.append("NWB TwoPhotonSeries are read as one plane and one channel")
        info.nplanes, info.nchannels = 1, 1
    csv = find_timestamps_csv(unit)
    per_vol = info.nplanes + info.flyback
    n_vol = info.n_pages // (per_vol * info.nchannels)
    if csv is not None:
        ts = read_timestamps_csv(csv)
        if len(ts) == info.n_pages:
            info.page_times, info.times_per = ts, "page"
        elif len(ts) == n_vol:
            info.page_times, info.times_per = ts, "volume"
        elif info.nchannels > 1 and len(ts) == info.n_pages // info.nchannels:
            info.page_times = np.repeat(ts, info.nchannels)
            info.times_per = "page"
        else:
            raise MovieError(f"{csv.name} has {len(ts)} times but the movie has {info.n_pages} frames "
                             f"({n_vol} volumes)")
        info.time_source = csv.name
    if info.page_times is not None:
        vt = volume_times(info)
        if len(vt) > 1:
            fs_ts = 1.0 / float(np.median(np.diff(vt)))
            if fs_param <= 0:
                info.fs_volume, info.fs_source = fs_ts, f"timestamps ({info.time_source})"
            info.t0 = float(vt[0])
    if fs_param > 0:
        if info.fs_volume and abs(info.fs_volume - fs_param) / fs_param > 0.01:
            info.notes.append(f"FS={fs_param:g} differs from the file's {info.fs_volume:.4g} Hz "
                              f"({info.fs_source}); using FS")
        info.fs_volume, info.fs_source = float(fs_param), "FS parameter"
    if not info.fs_volume:
        raise MovieError("the frame rate is not in the file's metadata and there is no "
                         f"{unit.name}_timestamps.csv: set FS (frames per second per plane)")
    if info.n_pages % (per_vol * info.nchannels):
        info.notes.append(f"{info.n_pages % (per_vol * info.nchannels)} trailing frames do not fill a "
                          "volume and are dropped")
    return info


def volume_times(info: MovieInfo) -> np.ndarray:
    """Time of each volume (its first plane), session clock."""
    per_vol = (info.nplanes + info.flyback) * info.nchannels
    n_vol = info.n_pages // per_vol
    if info.page_times is not None:
        if info.times_per == "page":
            return np.asarray(info.page_times[: n_vol * per_vol: per_vol], float)
        return np.asarray(info.page_times[:n_vol], float)
    return info.t0 + np.arange(n_vol) / info.fs_volume


def plane_times(info: MovieInfo, plane: int, v0: int, v1: int) -> np.ndarray:
    """Times of plane `plane` for volumes v0..v1-1."""
    per_vol = (info.nplanes + info.flyback) * info.nchannels
    if info.page_times is not None and info.times_per == "page":
        idx = np.arange(v0, v1) * per_vol + plane * info.nchannels
        return np.asarray(info.page_times[idx], float)
    vt = volume_times(info)[v0:v1]
    if info.nplanes > 1 and info.page_times is None:
        vt = vt + plane / (info.fs_volume * (info.nplanes + info.flyback))
    return vt


def window(info: MovieInfo, t_start: float | None, t_stop: float | None) -> tuple[int, int]:
    """Volumes [v0, v1) whose times fall in [t_start, t_stop)."""
    vt = volume_times(info)
    v0, v1 = 0, len(vt)
    if t_start is not None:
        v0 = int(np.searchsorted(vt, t_start, side="left"))
    if t_stop is not None:
        v1 = int(np.searchsorted(vt, t_stop, side="left"))
    return v0, v1


# ---------------------------------------------------------------------------------------------
# writing


def to_int16(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.int16:
        return a
    if a.dtype == np.uint16 or a.dtype == np.int32 or a.dtype == np.uint32:
        return (a // 2).astype(np.int16)
    if a.dtype.kind == "f":
        return np.clip(np.rint(a), -32768, 32767).astype(np.int16)
    return a.astype(np.int16)


def write_binaries(src, info: MovieInfo, dbs: list[dict], v0: int, v1: int, functional_chan: int,
                   batch_volumes: int = 200, progress=None) -> None:
    """De-interleave volumes v0..v1-1 into each plane's data.bin (and data_chan2.bin)."""
    P, F, C = info.nplanes, info.flyback, info.nchannels
    per_vol = (P + F) * C
    fchan = functional_chan - 1 if C > 1 else 0
    files = [open(db["reg_file"], "wb") for db in dbs]
    files2 = [open(db["reg_file_chan2"], "wb") for db in dbs] if C > 1 else None
    means = [np.zeros((info.Ly, info.Lx)) for _ in dbs]
    means2 = [np.zeros((info.Ly, info.Lx)) for _ in dbs] if C > 1 else None
    t_last = time.time()
    try:
        for a in range(v0, v1, batch_volumes):
            b = min(a + batch_volumes, v1)
            pages = src.read(a * per_vol, b * per_vol)
            pages = pages.reshape(b - a, P + F, C, info.Ly, info.Lx)
            for k in range(P):
                fr = to_int16(pages[:, k, fchan])
                files[k].write(np.ascontiguousarray(fr).tobytes())
                means[k] += fr.sum(axis=0, dtype=np.float64)
                if C > 1:
                    fr2 = to_int16(pages[:, k, 1 - fchan])
                    files2[k].write(np.ascontiguousarray(fr2).tobytes())
                    means2[k] += fr2.sum(axis=0, dtype=np.float64)
            if progress and time.time() - t_last > 30:
                progress(b - v0, v1 - v0)
                t_last = time.time()
    finally:
        for f in files + (files2 or []):
            f.close()
    n = v1 - v0
    for k, db in enumerate(dbs):
        db["Ly"], db["Lx"], db["nframes"] = info.Ly, info.Lx, n
        db["frames_per_file"] = np.array([n])
        db["frames_per_folder"] = np.array([n])
        db["meanImg"] = means[k] / max(n, 1)
        if C > 1:
            db["meanImg_chan2"] = means2[k] / max(n, 1)
