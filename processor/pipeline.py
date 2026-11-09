"""
One session through Suite2p: movie -> per-plane binary -> registration, detection, extraction,
deconvolution and classification (`suite2p.run_s2p.run_plane`), then timestamps, session.json
and the QC figures.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np

from processor import figures
from processor.movie import (MovieError, MovieInfo, describe, open_source, plane_times, volume_times,
                             window, write_binaries)
from processor.params import Params
from processor.sessions import Unit, find_timestamps_csv, read_timestamps_csv

log = logging.getLogger("processor")
MIN_FRAMES = 50
REGMETRIC_FRAMES = 500   # Suite2p samples 2000-5000 frames (4+ GB for 512x512); see reg_metrics()


BLOCKS = (512, 384, 256, 192, 160, 128, 96, 80, 64, 48, 40, 32, 24, 16)


def fit_block_size(requested: int, Ly: int, Lx: int) -> int:
    """The largest Suite2p-friendly block size <= BLOCK_SIZE that gives at least two blocks per
    side (Suite2p 1.1.0's non-rigid step fails on a field of view with one block)."""
    limit = min(requested, min(Ly, Lx) // 2)
    return next((b for b in BLOCKS if b <= limit), 16)


def suite2p_settings(p: Params, fs: float, device: str, cellpose_model: Path | None,
                     block_size: int | None = None) -> dict:
    import suite2p
    s = suite2p.default_settings()
    s["torch_device"] = device
    s["fs"] = float(fs)
    s["tau"] = float(p.tau)
    d = p.diameter if p.diameter > 0 else 12.0
    s["diameter"] = [float(d), float(d)]
    s["run"]["do_registration"] = 1 if p.do_registration else 0
    s["run"]["do_regmetrics"] = False       # computed by reg_metrics() on fewer frames
    s["io"]["save_NWB"] = False
    s["io"]["save_mat"] = False
    s["io"]["delete_bin"] = False          # handled here (KEEP_BINARY)
    s["io"]["combined"] = False
    s["registration"]["nonrigid"] = bool(p.nonrigid)
    bs = int(block_size or p.block_size)
    s["registration"]["block_size"] = (bs, bs)
    s["registration"]["batch_size"] = int(p.batch_size)
    det = s["detection"]
    det["algorithm"] = {"functional": "sparsery", "sourcery": "sourcery", "anatomical": "cellpose"}[p.detection]
    det["threshold_scaling"] = float(p.threshold_scaling)
    if p.diameter > 0:
        # sparsery's spatial scale: 1 = 6 px, 2 = 12 px, 3 = 24 px, 4 = 48 px cells
        det["sparsery_settings"]["spatial_scale"] = int(min(4, max(1, round(math.log2(p.diameter / 3)))))
    if p.detection == "anatomical":
        det["cellpose_settings"]["cellpose_model"] = str(cellpose_model)
    s["classification"]["use_builtin_classifier"] = True
    s["classification"]["classifier_path"] = None
    return s


def make_dbs(unit: Unit, info: MovieInfo, sdir: Path, scratch: Path) -> list[dict]:
    """Per-plane db dicts, as suite2p.io.init_dbs makes them, with our folders."""
    dbs = []
    for k in range(info.nplanes):
        save_path = sdir / f"plane{k}"
        fast = scratch / f"plane{k}"
        save_path.mkdir(parents=True, exist_ok=True)
        fast.mkdir(parents=True, exist_ok=True)
        db = {
            "data_path": [str(unit.files[0].parent if unit.kind != "bruker" else unit.files[0])],
            "file_list": [str(f) for f in unit.files], "input_format": unit.kind,
            "keep_movie_raw": False, "nplanes": info.nplanes, "nchannels": info.nchannels,
            "nrois": 1, "functional_chan": 1, "save_path0": str(sdir.parent), "save_folder": sdir.name,
            "fast_disk": str(fast), "save_path": str(save_path),
            "settings_path": str(save_path / "settings.npy"), "db_path": str(save_path / "db.npy"),
            "reg_file": str(fast / "data.bin"), "iplane": k, "batch_size": 500,
        }
        if info.nchannels > 1:
            db["reg_file_chan2"] = str(fast / "data_chan2.bin")
        dbs.append(db)
    return dbs


def _bruker_info(unit: Unit, p: Params) -> MovieInfo:
    """Frame rate and frame times from the PrairieView xml."""
    import xml.etree.ElementTree as ET
    xml = next(f for f in sorted(unit.files[0].iterdir()) if f.suffix.lower() == ".xml")
    root = ET.parse(xml).getroot()
    period = None
    for el in root.iter("PVStateValue"):
        if el.get("key") == "framePeriod":
            period = float(el.get("value"))
            break
    times = [float(fr.get("relativeTime")) for fr in root.iter("Frame") if fr.get("relativeTime")]
    info = MovieInfo(n_pages=0, Ly=0, Lx=0, dtype="uint16", nplanes=max(p.nplanes, 1),
                     nchannels=max(p.nchannels, 1))
    if p.fs > 0:
        info.fs_volume, info.fs_source = p.fs, "FS parameter"
    elif period:
        info.fs_volume, info.fs_source = 1.0 / (period * info.nplanes), "PrairieView framePeriod"
    else:
        raise MovieError("no framePeriod in the PrairieView xml: set FS")
    if times:
        info.page_times, info.times_per, info.time_source = np.array(times), "page", f"{xml.name} relativeTime"
    csv = find_timestamps_csv(unit)
    if csv:
        info.page_times, info.times_per, info.time_source = read_timestamps_csv(csv), "page", csv.name
    return info


def run_session(unit: Unit, p: Params, out_root: Path, device: str, cellpose_model: Path | None) -> dict:
    """Process one movie; returns the session's summary (raises on failure)."""
    from suite2p.run_s2p import run_plane
    t0 = time.time()
    sid = unit.session_id
    sdir = out_root / sid
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    import tempfile
    # Suite2p's binary (as large as the raw movie, int16) is scratch: $TMPDIR (Lambda-safe)
    scratch = Path(tempfile.mkdtemp(prefix=f"suite2p-{sid}-"))
    notes: list[str] = []

    if unit.kind == "bruker":
        import suite2p.io as s2io
        info = _bruker_info(unit, p)
        dbs = make_dbs(unit, info, sdir, scratch)
        dbs[0]["file_list"] = sorted(str(f) for f in unit.files[0].iterdir()
                                     if f.name.lower().endswith((".ome.tif", ".ome.tiff")))
        for db in dbs:
            db["file_list"] = dbs[0]["file_list"]
            db["first_files"] = np.zeros(len(db["file_list"]), bool)
            db["first_files"][0] = True
        if p.t_start is not None or p.t_stop is not None:
            notes.append("T_START/T_STOP are not applied to Bruker folders (Suite2p's reader converts all frames)")
        files = [open(db["reg_file"], "wb") for db in dbs]
        files2 = [open(db["reg_file_chan2"], "wb") for db in dbs] if info.nchannels > 1 else None
        try:
            dbs = s2io.ome_to_binary(dbs, {}, files, files2)
        finally:
            for f in files + (files2 or []):
                f.close()
        info.Ly, info.Lx = dbs[0]["Ly"], dbs[0]["Lx"]
        n_vol = dbs[0]["nframes"]
        info.n_pages = n_vol * info.nplanes * info.nchannels
        if info.page_times is not None and len(info.page_times) not in (info.n_pages, n_vol):
            notes.append("frame times in the xml do not match the frame count; using frame / fs")
            info.page_times = None
        if info.page_times is not None and len(info.page_times) == n_vol:
            info.times_per = "volume"
        v0, v1 = 0, n_vol
        src = None
    else:
        src = open_source(unit)
        try:
            info = describe(unit, src, p.fs, p.nplanes, p.nchannels)
            v0, v1 = window(info, p.t_start, p.t_stop)
            if v1 - v0 < MIN_FRAMES:
                raise MovieError(f"only {v1 - v0} frames per plane in the time window (need >= {MIN_FRAMES})")
            if p.functional_chan > info.nchannels:
                raise MovieError(f"FUNCTIONAL_CHAN={p.functional_chan} but the movie has {info.nchannels} channel(s)")
            dbs = make_dbs(unit, info, sdir, scratch)
            log.info(f"{sid}: {unit.kind} {', '.join(f.name for f in unit.files[:3])}"
                     f"{' ...' if len(unit.files) > 3 else ''}: {info.n_pages} frames {info.Ly}x{info.Lx} "
                     f"{info.dtype}, {info.nplanes} plane(s), {info.nchannels} channel(s), "
                     f"{info.fs_volume:.4g} Hz per plane ({info.fs_source}); volumes {v0}..{v1}")
            write_binaries(src, info, dbs, v0, v1, p.functional_chan,
                           progress=lambda d, n: log.info(f"{sid}: converted {d}/{n} frames"))
        finally:
            src.close()
    notes += info.notes
    log.info(f"{sid}: binary written in {time.time() - t0:.1f} s")

    block = fit_block_size(p.block_size, info.Ly, info.Lx)
    if block != p.block_size and p.nonrigid:
        notes.append(f"BLOCK_SIZE {p.block_size} -> {block} px: the {info.Ly}x{info.Lx} field of view "
                     "needs at least two blocks per side for non-rigid registration")
    settings = suite2p_settings(p, info.fs_volume, device, cellpose_model, block)
    planes = []
    for k, db in enumerate(dbs):
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)
        tp = time.time()
        out = run_plane(db=db, settings=dict(settings), db_path=db["db_path"])
        reg, det, stat, F, Fneu, *_rest = out
        spks, iscell = _rest[2], _rest[3]
        pdir = Path(db["save_path"])
        n_rois = 0 if stat is None else len(stat)
        if iscell is not None and len(iscell):
            iscell = np.asarray(iscell, float)
            iscell[:, 0] = (iscell[:, 1] >= p.classifier_threshold).astype(float)
            np.save(pdir / "iscell.npy", iscell)
        n_cells = int(iscell[:, 0].sum()) if iscell is not None and len(iscell) else 0
        times = plane_times(info, k, v0, v1)
        np.save(pdir / "timestamps.npy", times)
        ops = np.load(pdir / "ops.npy", allow_pickle=True).item()
        if p.do_registration:
            try:
                ops.update(reg_metrics(db, ops, settings, device))
                np.save(pdir / "ops.npy", ops)
            except Exception as exc:  # metrics are QC only
                notes.append(f"plane{k}: registration metrics failed ({exc})")
        figures_k = figures.plane_figures(sid, k, info.nplanes, ops, stat, iscell, times, sdir)
        ref_cmp = None
        if unit.companion is not None and info.nplanes == 1 and stat is not None and len(stat):
            from processor import reference
            try:
                ref_cmp = reference.compare(unit.companion, stat, iscell, info.Ly, info.Lx)
                if ref_cmp:
                    bg = figures._full(ops.get("max_proj"), ops)
                    figures_k.append(reference.overlay_figure(sid, sdir, ref_cmp, stat, iscell,
                                                              bg if bg is not None else ops["meanImg"],
                                                              info.Ly, info.Lx))
                    ref_cmp = {k_: v for k_, v in ref_cmp.items() if not k_.startswith("_")}
                    log.info(f"{sid}: {ref_cmp['n_matched']}/{ref_cmp['n_reference_rois']} reference ROIs "
                             f"matched by our cells (IoU > 0.5) in {unit.companion.name}")
            except Exception as exc:
                notes.append(f"comparison with the reference ROIs failed: {exc}")
                ref_cmp = None
        planes.append({"plane": k, "n_rois": n_rois, "n_cells": n_cells, "reference_roi_match": ref_cmp,
                       "seconds": round(time.time() - tp, 1), "figures": figures_k,
                       **_reg_numbers(ops)})
        log.info(f"{sid} plane{k}: {n_rois} ROIs, {n_cells} classified as cells "
                 f"(prob >= {p.classifier_threshold:g}), {time.time() - tp:.1f} s")
        # files we do not ship (intermediate copies of what ops.npy holds)
        for extra in ("reg_outputs.npy", "detect_outputs.npy", "zcorr.npy"):
            (pdir / extra).unlink(missing_ok=True)
        if p.keep_binary:
            shutil.move(db["reg_file"], pdir / "data.bin")
            if "reg_file_chan2" in db:
                shutil.move(db["reg_file_chan2"], pdir / "data_chan2.bin")
    shutil.rmtree(scratch, ignore_errors=True)

    vt = volume_times(info)[v0:v1]
    np.save(sdir / "timestamps.npy", vt)
    fs_eff = (len(vt) - 1) / (vt[-1] - vt[0]) if len(vt) > 1 and vt[-1] > vt[0] else info.fs_volume
    session = {
        "session_id": sid,
        "source": [str(f.name) for f in unit.files],
        "format": unit.kind,
        "frame_rate": round(float(info.fs_volume), 6),
        "frame_rate_source": info.fs_source,
        "effective_frame_rate": round(float(fs_eff), 6),
        "n_frames": int(v1 - v0),
        "n_planes": info.nplanes,
        "n_channels": info.nchannels,
        "functional_chan": p.functional_chan if info.nchannels > 1 else 1,
        "Lx": info.Lx, "Ly": info.Ly,
        "timestamps_source": info.time_source,
        "t_start": round(float(vt[0]), 6), "t_stop": round(float(vt[-1]), 6),
        "first_frame": int(v0),
        "intensity_scale": 0.5 if info.dtype in ("uint16", "int32", "uint32") else 1.0,
        "tau": p.tau,
        "block_size": block,
        "detection": p.detection,
        **({"nwb_series": unit.nwb_series} if unit.nwb_series else {}),
        **({"h5_key": unit.h5_key} if unit.h5_key else {}),
    }
    ident = dict(info.identity)
    if unit.companion is not None:
        import h5py
        from processor.movie import nwb_identity
        try:
            with h5py.File(unit.companion, "r") as f:
                comp = nwb_identity(f)
            if not ident:
                ident = comp
            session["companion_nwb"] = unit.companion.name
        except OSError as exc:
            notes.append(f"companion NWB {unit.companion.name} unreadable: {exc}")
    if ident:
        session["source_nwb"] = ident
    if unit.sheet:
        session["sheet"] = unit.sheet
    if notes:
        session["notes"] = notes
    if abs(fs_eff - info.fs_volume) / info.fs_volume > 0.001:
        session["frame_rate_drift_percent"] = round(100 * (fs_eff - info.fs_volume) / info.fs_volume, 3)
    (sdir / "session.json").write_text(json.dumps(session, indent=2, default=str))

    return {
        "session": session, "planes": planes, "seconds": round(time.time() - t0, 1),
        "n_rois": sum(pl["n_rois"] for pl in planes), "n_cells": sum(pl["n_cells"] for pl in planes),
        "notes": notes,
    }


def reg_metrics(db: dict, ops: dict, settings: dict, device: str) -> dict:
    """Suite2p's PC-based registration metrics (tPC, regPC, regDX) on REGMETRIC_FRAMES frames
    spread evenly over the registered movie. Suite2p itself samples 2000 frames (5000 for long
    recordings), whose PCA needs over 4 GB for a 512 x 512 movie; 500 frames give the same metric
    (the residual shift between the frames at the two ends of each principal component)."""
    import torch
    from suite2p.registration.metrics import get_pc_metrics
    n, Ly, Lx = int(db["nframes"]), int(db["Ly"]), int(db["Lx"])
    if n < 300:
        return {}
    mm = np.memmap(db["reg_file"], dtype=np.int16, mode="r", shape=(n, Ly, Lx))
    idx = np.linspace(0, n - 1, min(REGMETRIC_FRAMES, n)).astype(int)
    mov = np.asarray(mm[idx])
    del mm
    tPC, regPC, regDX = get_pc_metrics(mov, yrange=ops.get("yrange"), xrange=ops.get("xrange"),
                                       settings=settings["registration"], device=torch.device(device))
    return {"tPC": tPC, "regPC": regPC, "regDX": regDX, "regmetric_frames": len(idx)}


def _reg_numbers(ops: dict) -> dict:
    out = {}
    if "yoff" in ops and len(ops["yoff"]):
        y, x = np.asarray(ops["yoff"], float), np.asarray(ops["xoff"], float)
        out["rigid_shift_max_px"] = round(float(np.max(np.hypot(y, x))), 2)
        out["rigid_shift_sd_px"] = round(float(np.std(np.hypot(y - y.mean(), x - x.mean()))), 3)
    if "regDX" in ops and ops["regDX"] is not None and len(ops["regDX"]):
        out["regDX_max_px"] = round(float(np.max(ops["regDX"])), 3)
    return out
