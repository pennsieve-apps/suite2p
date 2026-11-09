"""
suite2p: motion correction, ROI detection, cell classification and fluorescence/neuropil
extraction of two-photon movies with Suite2p 1.x, one session at a time.

INPUT_DIR: movies (TIFF/BigTIFF incl. ScanImage, HDF5, NWB TwoPhotonSeries, Bruker), optional
sessions.csv, optional `<stem>_timestamps.csv`, optional companion NWB (identity), optional
Cellpose model (DETECTION=anatomical). OUTPUT_DIR/suite2p/<session_id>/plane<k>/ holds Suite2p's
standard outputs; see README.md.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from processor.params import BadParameter, read_params
from processor.resources import cgroup_memory_peak_mb, choose_device, resolve_threads

APP = "suite2p"
CITATIONS = ["10.64898/2026.02.04.703741", "10.1101/061507", "10.1371/journal.pcbi.1005423"]
log = logging.getLogger("processor")


def _version() -> str:
    for p in (Path(__file__).resolve().parent.parent / "VERSION", Path("/processor/VERSION")):
        if p.is_file():
            return p.read_text().strip()
    return "0.0.0"


class _DropNoise(logging.Filter):
    """Suite2p warns at import that pynwb is missing; this app reads NWB itself."""
    def filter(self, record):
        return "pynwb not installed" not in record.getMessage()


def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    h.addFilter(_DropNoise())
    for name in ("processor", "suite2p"):
        lg = logging.getLogger(name)
        lg.handlers = [h]
        lg.setLevel(logging.INFO)
        lg.propagate = False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru = ru / 1024 if sys.platform != "darwin" else ru / 2**20
    cg = cgroup_memory_peak_mb()
    return round(max(ru, cg or 0), 1)


def find_cellpose_model(root: Path) -> Path | None:
    from processor.sessions import walk
    cands = []
    for p in walk(root):
        low = p.name.lower()
        if "cellpose-model" in [q.name for q in p.parents] or low.endswith((".pth", ".pt")) or low.startswith("cpsam"):
            if not low.endswith((".tif", ".tiff", ".nwb", ".h5", ".hdf5", ".csv", ".json", ".xml", ".md", ".txt")):
                cands.append(p)
    return cands[0] if cands else None


def _tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("suite2p", "torch", "numpy", "scipy", "tifffile", "h5py", "cellpose", "scanimage-tiff-reader"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def main() -> int:
    _setup_logging()
    started, t0 = _now(), time.time()
    version = _version()
    try:
        p = read_params()
    except BadParameter as exc:
        log.error(f"bad parameter: {exc}")
        return 2
    in_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    n_threads, src = resolve_threads(p.threads)
    log.info(f"{APP} {version}: INPUT_DIR={in_dir} OUTPUT_DIR={out_dir}, {n_threads} threads ({src})")

    import numpy as np
    import torch
    torch.set_num_threads(n_threads)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)
    dev = choose_device(p.use_gpu)
    log.info(dev["reason"])
    device = "cuda" if dev["accelerator"] == "gpu" else "cpu"

    cellpose_model = None
    if p.detection == "anatomical":
        cellpose_model = find_cellpose_model(in_dir)
        if cellpose_model is None:
            log.error("DETECTION=anatomical needs a Cellpose model file in the cellpose-model input "
                      "(the Cellpose-SAM weights are not bundled: see README, Licences)")
            return 2
        log.info(f"anatomical detection with Cellpose model {cellpose_model.name}")

    from processor.pipeline import run_session
    from processor.sessions import discover

    if not in_dir.is_dir():
        log.error(f"INPUT_DIR {in_dir} does not exist")
        return 1
    units, warnings = discover(in_dir)
    for w in warnings:
        log.warning(w)
    if not units:
        log.error("no movies found (TIFF, HDF5, NWB with a TwoPhotonSeries, or a Bruker folder)")
        return 1
    log.info(f"{len(units)} session(s): {', '.join(u.session_id for u in units)}")

    top = out_dir / APP
    top.mkdir(parents=True, exist_ok=True)
    results, statuses, figures_all = {}, {}, []
    for u in units:
        ts = time.time()
        try:
            r = run_session(u, p, top, device, cellpose_model)
            results[u.session_id] = r
            statuses[u.session_id] = {"status": "ok", "seconds": round(time.time() - ts, 1)}
            for pl in r["planes"]:
                for f in pl["figures"]:
                    figures_all.append({"path": f"{u.session_id}/{f['path']}", "caption": f["caption"]})
            s = r["session"]
            log.info(f"{u.session_id}: ok, {s['n_frames']} frames x {s['n_planes']} plane(s) at "
                     f"{s['frame_rate']:.4g} Hz, {r['n_rois']} ROIs, {r['n_cells']} cells, {time.time() - ts:.1f} s")
            for n in r["notes"]:
                warnings.append(f"{u.session_id}: {n}")
        except Exception as exc:  # one failed session must not take the others down
            log.error(f"{u.session_id}: failed: {exc}")
            log.debug(traceback.format_exc())
            traceback.print_exc(file=sys.stdout)
            import glob
            import shutil
            import tempfile
            for d in glob.glob(os.path.join(tempfile.gettempdir(), f"suite2p-{u.session_id}-*")):
                shutil.rmtree(d, ignore_errors=True)
            statuses[u.session_id] = {"status": "failed", "seconds": round(time.time() - ts, 1), "error": str(exc)}
            warnings.append(f"{u.session_id}: failed: {exc}")

    # sessions.csv (first app of the chain)
    extra_cols = []
    for u in units:
        for k in u.sheet:
            if k not in ("subject_id", "condition", "group") and k not in extra_cols:
                extra_cols.append(k)
    cols = ["session_id", "path", "subject_id", "condition", "group", "status", "n_frames", "n_planes",
            "frame_rate", "duration_s", "n_rois", "n_cells"] + extra_cols
    with open(top / "sessions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for u in units:
            r = results.get(u.session_id)
            s = r["session"] if r else {}
            subj = u.sheet.get("subject_id") or (s.get("source_nwb", {}).get("subject", {}) or {}).get("subject_id", "")
            row = {"session_id": u.session_id, "path": str(u.path.relative_to(in_dir)) if u.path.is_relative_to(in_dir) else u.path.name,
                   "subject_id": subj, "condition": u.sheet.get("condition", ""), "group": u.sheet.get("group", ""),
                   "status": statuses[u.session_id]["status"]}
            if r:
                row.update(n_frames=s["n_frames"], n_planes=s["n_planes"], frame_rate=s["frame_rate"],
                           duration_s=round(s["t_stop"] - s["t_start"] + 1 / s["frame_rate"], 3),
                           n_rois=r["n_rois"], n_cells=r["n_cells"])
            row.update({k: u.sheet.get(k, "") for k in extra_cols})
            w.writerow(row)

    summary_sessions = {}
    for u in units:
        r = results.get(u.session_id)
        entry = {"status": statuses[u.session_id]["status"]}
        entry.update({k: u.sheet[k] for k in ("subject_id", "condition", "group") if u.sheet.get(k)})
        if r:
            s = r["session"]
            entry.update(frame_rate=s["frame_rate"], n_frames=s["n_frames"], n_planes=s["n_planes"],
                         duration_s=round(s["t_stop"] - s["t_start"] + 1 / s["frame_rate"], 3),
                         n_rois=r["n_rois"], n_cells=r["n_cells"], time_source=s["timestamps_source"],
                         seconds=r["seconds"])
            if "frame_rate_drift_percent" in s:
                entry["frame_rate_drift_percent"] = s["frame_rate_drift_percent"]
            for pl in r["planes"]:
                if pl.get("reference_roi_match"):
                    entry["reference_roi_match"] = pl["reference_roi_match"]
                for k in ("rigid_shift_max_px", "rigid_shift_sd_px", "regDX_max_px"):
                    if k in pl:
                        entry[k if s["n_planes"] == 1 else f"plane{pl['plane']}_{k}"] = pl[k]
        else:
            entry["error"] = statuses[u.session_id].get("error", "")
        summary_sessions[u.session_id] = entry
    headline = {sid: {"fraction_reference_rois_matched": e["reference_roi_match"]["fraction_reference_matched"],
                      "n_matched": e["reference_roi_match"]["n_matched"],
                      "n_reference_rois": e["reference_roi_match"]["n_reference_rois"],
                      "reference": e["reference_roi_match"]["reference"]}
                for sid, e in summary_sessions.items() if e.get("reference_roi_match")}
    summary = {"app": APP, "version": version, "reference_roi_match": headline,
               "sessions": summary_sessions, "figures": figures_all,
               "tables": [{"path": "sessions.csv", "caption": "One row per session: status, frames, planes, frame rate, ROIs and cells."}],
               "warnings": warnings}
    (top / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    gpu = None
    if device == "cuda":
        import torch
        gpu = {"name": torch.cuda.get_device_name(0),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    variant = os.environ.get("APP_VARIANT") or ("gpu" if os.environ.get("TORCH_FLAVOUR") == "cuda" else "cpu")
    prov = {"app": APP, "version": version, "variant": variant, "tools": _tool_versions(),
            "parameters": p.as_env(),
            "inputs": sorted({str(f.relative_to(in_dir)) if f.is_relative_to(in_dir) else f.name
                              for u in units for f in u.files}),
            "started": started, "finished": _now(), "wall_seconds": round(time.time() - t0, 1),
            "peak_rss_mb": _peak_rss_mb(), "gpu": gpu, "device": dev["reason"],
            "sessions": statuses, "citations": CITATIONS}
    (top / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    n_ok = sum(1 for s in statuses.values() if s["status"] == "ok")
    log.info(f"done: {n_ok}/{len(units)} session(s) ok in {time.time() - t0:.1f} s, peak memory {prov['peak_rss_mb']} MB")
    return 0 if n_ok else 1
