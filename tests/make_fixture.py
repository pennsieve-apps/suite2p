"""
A small synthetic two-photon input with planted cells, for the unit tests, `make run` and the
review run:

  python -m tests.make_fixture data/fixture [--layout linked|flat|both] [--seed 7]

Two sessions:

- `movies/sub-synth01_ses-day1_2p.tif`: a 700-frame, 128 x 128 uint16 multi-page TIFF at 10 Hz
  with 24 planted somata (Gaussian-edged discs of about 12 px diameter), each with its own
  Poisson spike train convolved with a GCaMP6f-like kernel (rise 50 ms, decay 0.7 s), on a
  structured, slowly fluctuating neuropil background, with shot and read noise and a planted
  rigid motion (a slow drift plus small jitter, up to about 3 px). Frame times are in
  `sub-synth01_ses-day1_2p_timestamps.csv` (`frame, time`), starting at 12.5 s on the session
  clock, so the app's time handling is exercised (no re-zeroing).
- `movies/sub-synth02_ses-day1_ophys.nwb`: an NWB file with a 400-frame TwoPhotonSeries
  (`starting_time` 3.0 s, `rate` 15 Hz) of a different 96 x 96 field with 12 cells, its
  imaging plane (indicator GCaMP6f) and a subject, so the NWB reader and the session identity
  are exercised.

Without tifffile or pynwb (the platform's review generator image has numpy, scipy and h5py), both
movies are written as HDF5 (`data`, attribute `fs`) instead.

`truth/<session_id>_truth.npz` (centres, masks, spike trains, shifts) is written next to the
input (outside `linked/` and `flat/`), for the tests only.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SESSIONS = {
    "sub-synth01_ses-day1": dict(file="sub-synth01_ses-day1_2p.tif", ly=128, lx=128, n_frames=700,
                                 fs=10.0, t0=12.5, n_cells=24, max_shift=3.0),
    "sub-synth02_ses-day1": dict(file="sub-synth02_ses-day1_ophys.nwb", ly=96, lx=96, n_frames=400,
                                 fs=15.0, t0=3.0, n_cells=12, max_shift=1.0),
}
CELL_RADIUS = 5.5
TAU_DECAY = 0.7
TAU_RISE = 0.05


def _place_cells(rng, ly, lx, n, min_dist=13.0, margin=12):
    centres = []
    tries = 0
    while len(centres) < n and tries < 20000:
        tries += 1
        c = rng.uniform([margin, margin], [ly - margin, lx - margin])
        if all(np.hypot(*(c - d)) >= min_dist for d in centres):
            centres.append(c)
    return np.array(centres)


def _kernel(fs):
    t = np.arange(int(5 * TAU_DECAY * fs)) / fs
    k = (1 - np.exp(-t / TAU_RISE)) * np.exp(-t / TAU_DECAY)
    return k / k.max()


def simulate(cfg: dict, seed: int) -> dict:
    """The movie (uint16, T x Ly x Lx) and the truth."""
    rng = np.random.default_rng(seed)
    ly, lx, n_frames, fs, n_cells = cfg["ly"], cfg["lx"], cfg["n_frames"], cfg["fs"], cfg["n_cells"]
    pad = 8
    Ly, Lx = ly + 2 * pad, lx + 2 * pad          # simulate larger, crop after shifting
    centres = _place_cells(rng, ly, lx, n_cells) + pad
    yy, xx = np.mgrid[:Ly, :Lx]
    footprints = np.zeros((len(centres), Ly, Lx), np.float32)
    for i, (cy, cx) in enumerate(centres):
        r = np.hypot(yy - cy, xx - cx)
        fp = 1.0 / (1.0 + np.exp((r - CELL_RADIUS) / 0.8))       # soft disc
        ring = np.exp(-((r - CELL_RADIUS * 0.6) ** 2) / 6.0) * 0.3  # brighter cytoplasm ring
        footprints[i] = fp * (0.7 + ring)
    # spikes: Poisson at 0.2-1.5 Hz with a few bursts, convolved with the GCaMP6f-like kernel
    rates = rng.uniform(0.2, 1.5, len(centres))
    spikes = rng.poisson(rates[:, None] / fs, (len(centres), n_frames)).astype(np.float32)
    for i in range(len(centres)):
        for _ in range(3):
            t = rng.integers(0, n_frames - 5)
            spikes[i, t:t + 3] += rng.integers(1, 3)
    k = _kernel(fs)
    calcium = np.array([np.convolve(s, k)[:n_frames] for s in spikes], np.float32)
    # neuropil: smooth spatial texture with a slow common fluctuation
    tex = rng.normal(0, 1, (Ly // 8 + 2, Lx // 8 + 2))
    from scipy.ndimage import gaussian_filter, shift as nd_shift, zoom
    tex = zoom(gaussian_filter(tex, 1.0), 8, order=1)[:Ly, :Lx]
    neuropil_img = 180 + 40 * (tex - tex.min()) / (np.ptp(tex) + 1e-9)
    slow = 1 + 0.05 * np.sin(2 * np.pi * np.arange(n_frames) / (n_frames / 2.3))
    baseline = 220 + 60 * rng.uniform(0, 1, len(centres))      # resting brightness of each soma
    gain = 60.0      # one spike: about 0.15-0.2 dF/F after neuropil correction (GCaMP6f-like)
    # motion: slow drift + jitter, rigid
    drift = np.cumsum(rng.normal(0, 0.15, (n_frames, 2)), axis=0)
    drift -= drift.mean(axis=0)
    shifts = np.clip(drift + rng.normal(0, 0.3, (n_frames, 2)), -cfg["max_shift"], cfg["max_shift"])
    movie = np.empty((n_frames, ly, lx), np.uint16)
    fp_flat = footprints.reshape(len(centres), -1)
    for t in range(n_frames):
        img = neuropil_img * slow[t]
        img = img + ((baseline + gain * calcium[:, t]) @ fp_flat).reshape(Ly, Lx)
        img = nd_shift(img, shifts[t], order=1, mode="nearest")
        img = rng.poisson(np.clip(img, 0, None)).astype(np.float32) + rng.normal(0, 8, img.shape)
        movie[t] = np.clip(img[pad:pad + ly, pad:pad + lx], 0, 65535).astype(np.uint16)
    masks = footprints[:, pad:pad + ly, pad:pad + lx] > 0.5
    return {"movie": movie, "centres": centres - pad, "masks": masks, "spikes": spikes,
            "calcium": calcium, "shifts": shifts, "times": cfg["t0"] + np.arange(n_frames) / fs}


def write_h5(path: Path, movie: np.ndarray, fs: float) -> Path:
    """Fallback for the review generator image (numpy, scipy, h5py; no tifffile or pynwb)."""
    import h5py
    path = path.with_suffix(".h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=movie, chunks=(50,) + movie.shape[1:])
        f.attrs["fs"] = fs
    return path


def write_tiff(path: Path, movie: np.ndarray, times: np.ndarray) -> None:
    try:
        import tifffile
        tifffile.imwrite(path, movie, photometric="minisblack", metadata={"axes": "TYX"})
    except ImportError:
        path = write_h5(path, movie, 1 / float(np.median(np.diff(times))))
    with open(path.with_name(path.stem + "_timestamps.csv"), "w") as f:
        f.write("frame,time\n")
        for i, t in enumerate(times):
            f.write(f"{i},{t:.6f}\n")


def write_nwb(path: Path, movie: np.ndarray, cfg: dict, session_id: str) -> None:
    try:
        import pynwb  # noqa: F401
    except ImportError:
        write_h5(path, movie, cfg["fs"])
        return
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.file import Subject
    from pynwb.ophys import OpticalChannel, TwoPhotonSeries
    nwb = NWBFile(session_description="synthetic two-photon session with planted cells",
                  identifier=f"synthetic-{session_id}",
                  session_start_time=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
                  session_id=session_id.split("_ses-")[1], lab="synthetic", institution="Pennsieve")
    nwb.subject = Subject(subject_id="synth02", species="Mus musculus", sex="F", age="P90D")
    dev = nwb.create_device(name="Microscope")
    ch = OpticalChannel(name="green", description="GCaMP emission", emission_lambda=510.0)
    plane = nwb.create_imaging_plane(name="ImagingPlane", optical_channel=ch, imaging_rate=cfg["fs"],
                                     description="synthetic plane", device=dev, excitation_lambda=920.0,
                                     indicator="GCaMP6f", location="VISp")
    nwb.add_acquisition(TwoPhotonSeries(name="TwoPhotonSeries", data=movie, imaging_plane=plane,
                                        rate=cfg["fs"], starting_time=cfg["t0"], unit="n.a."))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def make(out: Path, layout: str = "both", seed: int = 7) -> None:
    out = Path(out)
    layouts = ["linked", "flat"] if layout == "both" else [layout]
    (out / "truth").mkdir(parents=True, exist_ok=True)
    built = {}
    for i, (sid, cfg) in enumerate(SESSIONS.items()):
        sim = simulate(cfg, seed + i)
        np.savez_compressed(out / "truth" / f"{sid}_truth.npz", centres=sim["centres"], masks=sim["masks"],
                            spikes=sim["spikes"], shifts=sim["shifts"], times=sim["times"], fs=cfg["fs"])
        built[sid] = sim
    for lay in layouts:
        root = out / lay
        if root.exists():
            shutil.rmtree(root)
        movies = root / "movies" if lay == "linked" else root
        movies.mkdir(parents=True)
        for sid, cfg in SESSIONS.items():
            path = movies / cfg["file"]
            if path.suffix == ".tif":
                write_tiff(path, built[sid]["movie"], built[sid]["times"])
            else:
                write_nwb(path, built[sid]["movie"], cfg, sid)
    print(f"fixture written to {out} ({', '.join(layouts)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--layout", choices=["linked", "flat", "both"], default="both")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    make(Path(a.out), a.layout, a.seed)


if __name__ == "__main__":
    main()
