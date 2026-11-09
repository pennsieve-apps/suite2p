"""
The synthetic fixture (tests/make_fixture.py: two sessions with planted cells and planted motion)
through the processor, without Docker. About a minute on a laptop CPU.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.make_fixture import make

SESSIONS = ("sub-synth01_ses-day1", "sub-synth02_ses-day1")


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("fixture")
    make(root, "both")
    return root


def _run(monkeypatch, in_dir: Path, out_dir: Path, **env) -> int:
    from processor.main import main
    base = {"INPUT_DIR": str(in_dir), "OUTPUT_DIR": str(out_dir), "USE_GPU": "false", "THREADS": "2",
            "HOME": str(out_dir.parent / "home")}
    for k in ("FS", "TAU", "NPLANES", "NCHANNELS", "T_START", "T_STOP", "DETECTION", "DIAMETER",
              "BLOCK_SIZE", "KEEP_BINARY", "CLASSIFIER_THRESHOLD", "NONRIGID", "DO_REGISTRATION"):
        monkeypatch.delenv(k, raising=False)
    for k, v in {**base, **env}.items():
        monkeypatch.setenv(k, str(v))
    return main()


@pytest.fixture(scope="session")
def linked_run(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out-linked")
    mp = pytest.MonkeyPatch()
    try:
        code = _run(mp, fixture_dir / "linked", out)
    finally:
        mp.undo()
    return code, out / "suite2p"


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_exit_code_and_layout(linked_run):
    code, top = linked_run
    assert code == 0
    for f in ("summary.json", "provenance.json", "sessions.csv"):
        assert (top / f).is_file()
    for sid in SESSIONS:
        d = top / sid
        for f in ("session.json", "timestamps.npy", f"{sid}_mean_image.png", f"{sid}_rois.png",
                  f"{sid}_registration.png"):
            assert (d / f).is_file(), f
        for f in ("ops.npy", "stat.npy", "F.npy", "Fneu.npy", "spks.npy", "iscell.npy", "timestamps.npy"):
            assert (d / "plane0" / f).is_file(), f
        assert not (d / "plane0" / "data.bin").exists()          # KEEP_BINARY=false
        assert not (d / ".binary").exists()


@pytest.mark.parametrize("sid", SESSIONS)
def test_planted_cells_are_detected(linked_run, fixture_dir, sid):
    _, top = linked_run
    truth = np.load(fixture_dir / "truth" / f"{sid}_truth.npz")
    stat = np.load(top / sid / "plane0" / "stat.npy", allow_pickle=True)
    iscell = np.load(top / sid / "plane0" / "iscell.npy")
    masks = truth["masks"]
    Ly, Lx = masks.shape[1:]
    rois = []
    for s in stat:
        m = np.zeros((Ly, Lx), bool)
        m[s["ypix"], s["xpix"]] = True
        rois.append(m)
    acc = iscell[:, 0] > 0
    found = [max((_iou(t, r) for r, a in zip(rois, acc) if a), default=0) for t in masks]
    recall = np.mean(np.array(found) > 0.25)
    assert recall >= 0.8, f"only {recall:.0%} of planted cells found among accepted ROIs"
    # accepted ROIs that match no planted cell (false positives) stay few
    fp = sum(1 for r, a in zip(rois, acc) if a and max(_iou(t, r) for t in masks) < 0.1)
    assert fp <= max(2, 0.2 * len(masks)), f"{fp} accepted ROIs match no planted cell"


@pytest.mark.parametrize("sid", SESSIONS)
def test_registration_recovers_planted_motion(linked_run, fixture_dir, sid):
    _, top = linked_run
    truth = np.load(fixture_dir / "truth" / f"{sid}_truth.npz")
    ops = np.load(top / sid / "plane0" / "ops.npy", allow_pickle=True).item()
    shifts = truth["shifts"]
    if np.std(shifts[:, 0]) < 0.5:
        pytest.skip("planted motion too small to correlate")
    assert np.corrcoef(ops["yoff"], shifts[:, 0])[0, 1] > 0.8
    assert np.corrcoef(ops["xoff"], shifts[:, 1])[0, 1] > 0.8
    assert "regDX" in ops


def test_times_on_session_clock(linked_run, fixture_dir):
    _, top = linked_run
    t1 = np.load(top / SESSIONS[0] / "timestamps.npy")
    assert t1[0] == pytest.approx(12.5) and np.median(np.diff(t1)) == pytest.approx(0.1)
    t2 = np.load(top / SESSIONS[1] / "plane0" / "timestamps.npy")
    assert t2[0] == pytest.approx(3.0) and np.median(np.diff(t2)) == pytest.approx(1 / 15)
    s1 = json.loads((top / SESSIONS[0] / "session.json").read_text())
    assert s1["frame_rate"] == pytest.approx(10.0)
    assert s1["timestamps_source"].endswith("_timestamps.csv")
    assert s1["n_frames"] == 700 and s1["Ly"] == 128 and s1["Lx"] == 128


def test_nwb_identity_and_summary(linked_run):
    _, top = linked_run
    s2 = json.loads((top / SESSIONS[1] / "session.json").read_text())
    ident = s2["source_nwb"]
    assert ident["session_start_time"].startswith("2026-01-15T10:30:00")
    assert ident["identifier"] == f"synthetic-{SESSIONS[1]}"
    assert ident["subject"]["subject_id"] == "synth02"
    assert ident["indicator"] == "GCaMP6f"
    summary = json.loads((top / "summary.json").read_text())
    assert summary["app"] == "suite2p"
    assert set(summary["sessions"]) == set(SESSIONS)
    assert all(s["status"] == "ok" and s["n_cells"] > 5 for s in summary["sessions"].values())
    assert all(f["caption"] and (top / f["path"]).is_file() for f in summary["figures"])
    prov = json.loads((top / "provenance.json").read_text())
    for k in ("app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
              "wall_seconds", "peak_rss_mb", "gpu", "sessions", "citations"):
        assert k in prov
    with open(top / "sessions.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["session_id"] for r in rows] == list(SESSIONS)
    assert all(r["status"] == "ok" for r in rows)


def test_flat_layout_and_time_window(fixture_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    code = _run(monkeypatch, fixture_dir / "flat", out, T_START=30, T_STOP=60)
    assert code == 0
    top = out / "suite2p"
    s1 = json.loads((top / SESSIONS[0] / "session.json").read_text())
    assert s1["n_frames"] == 300 and s1["t_start"] == pytest.approx(30.0)
    t = np.load(top / SESSIONS[0] / "plane0" / "timestamps.npy")
    assert len(t) == 300 and t[0] == pytest.approx(30.0)   # never re-zeroed
    F = np.load(top / SESSIONS[0] / "plane0" / "F.npy")
    assert F.shape[1] == 300
    # session 2 (3.0-29.6 s) has only 0 frames in the window: it fails, the step does not
    summary = json.loads((top / "summary.json").read_text())
    assert summary["sessions"][SESSIONS[1]]["status"] == "failed"


def test_keep_binary(fixture_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert _run(monkeypatch, fixture_dir / "flat", out, KEEP_BINARY="true", T_START=12, T_STOP=40) == 0
    b = out / "suite2p" / SESSIONS[0] / "plane0" / "data.bin"
    assert b.is_file() and b.stat().st_size == 275 * 128 * 128 * 2


def test_bad_parameters_and_no_input(fixture_dir, tmp_path, monkeypatch):
    assert _run(monkeypatch, fixture_dir / "linked", tmp_path / "a", DETECTION="cellpose") == 2
    assert _run(monkeypatch, fixture_dir / "linked", tmp_path / "b", TAU="-1") == 2
    assert _run(monkeypatch, fixture_dir / "linked", tmp_path / "c", T_START=10, T_STOP=5) == 2
    # anatomical detection needs a Cellpose model file (none is bundled)
    assert _run(monkeypatch, fixture_dir / "linked", tmp_path / "d", DETECTION="anatomical") == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run(monkeypatch, empty, tmp_path / "e") == 1


# --- units --------------------------------------------------------------------------------------

def test_session_ids(tmp_path):
    from processor.sessions import discover, session_from_name
    p = Path("/x/sub-NR-0031_ses-caa5dddc-9290-4e27-9f5e-575ba3598614_ophys.nwb")
    assert session_from_name(p) == "sub-NR-0031_ses-caa5dddc"
    assert session_from_name(Path("/d/sub-496001363_ses-503109347-StimA_desc-dg1-bin2_2p.tif")) == \
        "sub-496001363_ses-503109347-StimA"
    assert session_from_name(Path("/d/my movie (1).tif")) == "my-movie--1"
    # ScanImage series are one movie; sessions.csv names the session
    import tifffile
    for i in (1, 2, 3):
        tifffile.imwrite(tmp_path / f"m1_00001_0000{i}.tif", np.zeros((5, 8, 8), np.uint16))
    tifffile.imwrite(tmp_path / "other.tif", np.zeros((5, 8, 8), np.uint16))
    (tmp_path / "sessions.csv").write_text("session_id,path,condition\nmouseA_day1,m1_00001_00001.tif,ctrl\n")
    units, _ = discover(tmp_path)
    by = {u.session_id: u for u in units}
    assert set(by) == {"mouseA_day1", "other"}
    assert len(by["mouseA_day1"].files) == 3 and by["mouseA_day1"].sheet["condition"] == "ctrl"


def test_scanimage_metadata_and_volume_times():
    from processor.movie import MovieInfo, parse_scanimage, plane_times, volume_times, window
    info = MovieInfo(n_pages=(3 + 1) * 2 * 10, Ly=8, Lx=8, dtype="uint16")
    parse_scanimage({"SI.hChannels.channelSave": [1, 2], "SI.hFastZ.enable": True,
                     "SI.hStackManager.actualNumSlices": 3, "SI.hFastZ.numDiscardFlybackFrames": 1,
                     "SI.hRoiManager.scanVolumeRate": 5.0, "SI.hRoiManager.scanFrameRate": 20.0}, info)
    assert (info.nplanes, info.nchannels, info.flyback, info.fs_volume) == (3, 2, 1, 5.0)
    info.t0 = 2.0
    vt = volume_times(info)
    assert len(vt) == 10 and vt[1] - vt[0] == pytest.approx(0.2)
    assert plane_times(info, 2, 0, 2)[0] == pytest.approx(2.0 + 2 / (5.0 * 4))
    assert window(info, 2.3, 3.0) == (2, 5)


def test_block_size_fits_small_fields():
    from processor.pipeline import fit_block_size
    assert fit_block_size(128, 512, 512) == 128
    assert fit_block_size(128, 128, 128) == 64
    assert fit_block_size(128, 96, 96) == 48


def test_reference_roi_matching():
    from processor.reference import match
    Ly = Lx = 40
    def disc(cy, cx, r=4):
        yy, xx = np.mgrid[:Ly, :Lx]
        return np.flatnonzero(np.hypot(yy - cy, xx - cx) <= r)
    ref = [disc(10, 10), disc(10, 30), disc(30, 10)]
    ours = [disc(10, 11), disc(30, 30), disc(10, 30, 2)]       # one good match, one miss, one too small
    r, c, iou = match(ref, ours, Ly * Lx)
    assert list(r) == [0] and list(c) == [0] and iou[0] > 0.5
