"""
Comparison of our ROIs with reference ROIs from a companion NWB of the same session (for example
the Allen Brain Observatory's own segmentation in its processed NWB): the evidence that the
segmentation is sound, since there is no independent recent segmentation benchmark.

Reference ROIs are read from processing/ophys/ImageSegmentation/<PlaneSegmentation> (pixel_mask or
image_mask). Masks are binarised (every pixel with weight > 0), ROIs are paired one-to-one by
maximising IoU (Hungarian assignment), and a pair counts as a match when IoU > 0.5. The headline is
the fraction of reference ROIs matched by one of our ROIs classified as a cell.

NWB's pixel_mask stores (x, y), but writers differ in which is the column; both orientations are
tried and the better one is used (reported as `orientation`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IOU_MATCH = 0.5


def read_reference_masks(nwb: Path, Ly: int, Lx: int) -> tuple[list[np.ndarray], str] | None:
    """Flat pixel indices of each reference ROI (both orientations), and the table's path."""
    import h5py
    with h5py.File(nwb, "r") as f:
        seg = f.get("processing/ophys/ImageSegmentation")
        if seg is None:
            return None
        for name in seg:
            ps = seg[name]
            if not hasattr(ps, "keys"):
                continue
            if "pixel_mask" in ps:
                pm = ps["pixel_mask"][:]
                idx = ps["pixel_mask_index"][:].astype(np.int64)
                starts = np.r_[0, idx[:-1]]
                a, b = pm["x"].astype(np.int64), pm["y"].astype(np.int64)
                w = pm["weight"] if "weight" in pm.dtype.names else np.ones(len(a))
                out = {}
                for orient, (r, c) in {"x=column": (b, a), "x=row": (a, b)}.items():
                    if r.max() >= Ly or c.max() >= Lx:
                        continue
                    flat = r * Lx + c
                    out[orient] = [np.unique(flat[s:e][w[s:e] > 0]) for s, e in zip(starts, idx)]
                return out, f"processing/ophys/ImageSegmentation/{name}"
            if "image_mask" in ps:
                im = ps["image_mask"]
                if im.shape[1:] != (Ly, Lx):
                    return None
                return {"image_mask": [np.flatnonzero(im[i] > 0) for i in range(im.shape[0])]}, \
                    f"processing/ophys/ImageSegmentation/{name}"
    return None


def _sparse(masks: list[np.ndarray], n_pix: int):
    from scipy.sparse import csr_matrix
    rows = np.concatenate([np.full(len(m), i) for i, m in enumerate(masks)]) if masks else np.array([], int)
    cols = np.concatenate(masks) if masks else np.array([], int)
    return csr_matrix((np.ones(len(cols), np.float32), (rows, cols)), shape=(len(masks), n_pix))


def iou_matrix(ref: list[np.ndarray], ours: list[np.ndarray], n_pix: int) -> np.ndarray:
    A, B = _sparse(ref, n_pix), _sparse(ours, n_pix)
    inter = (A @ B.T).toarray()
    na = np.array([len(m) for m in ref], float)[:, None]
    nb = np.array([len(m) for m in ours], float)[None, :]
    return inter / np.maximum(na + nb - inter, 1)


def match(ref: list[np.ndarray], ours: list[np.ndarray], n_pix: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(reference index, our index, IoU) of the one-to-one pairs with IoU > IOU_MATCH."""
    from scipy.optimize import linear_sum_assignment
    if not ref or not ours:
        return np.array([], int), np.array([], int), np.array([])
    iou = iou_matrix(ref, ours, n_pix)
    r, c = linear_sum_assignment(-iou)
    keep = iou[r, c] > IOU_MATCH
    return r[keep], c[keep], iou[r, c][keep]


def compare(nwb: Path, stat, iscell, Ly: int, Lx: int, t_window: tuple[float, float] | None = None) -> dict | None:
    got = read_reference_masks(nwb, Ly, Lx)
    if got is None:
        return None
    variants, table = got
    ours_all = [np.unique(np.asarray(s["ypix"]) * Lx + np.asarray(s["xpix"])) for s in stat]
    cells = np.asarray(iscell[:, 0] > 0) if iscell is not None and len(iscell) else np.ones(len(stat), bool)
    ours_cells = [m for m, c in zip(ours_all, cells) if c]
    cell_idx = np.flatnonzero(cells)
    best = None
    for orient, ref in variants.items():
        r, c, iou = match(ref, ours_cells, Ly * Lx)
        if best is None or len(r) > best[1]:
            best = (orient, len(r), ref, r, c, iou)
    orient, n_match, ref, r, c, iou = best
    r_all, _, _ = match(ref, ours_all, Ly * Lx)
    return {
        "reference": f"{nwb.name}:{table}",
        "orientation": orient,
        "iou_threshold": IOU_MATCH,
        "n_reference_rois": len(ref),
        "n_our_rois": len(ours_all),
        "n_our_cells": len(ours_cells),
        "n_matched": int(n_match),
        "fraction_reference_matched": round(n_match / max(len(ref), 1), 4),
        "fraction_our_cells_matched": round(n_match / max(len(ours_cells), 1), 4),
        "fraction_reference_matched_any_roi": round(len(r_all) / max(len(ref), 1), 4),
        "median_iou_of_matches": round(float(np.median(iou)), 4) if len(iou) else None,
        "_pairs": (r, cell_idx[c]),
        "_ref_masks": ref,
    }


def overlay_figure(sid: str, out: Path, cmp: dict, stat, iscell, bg: np.ndarray, Ly: int, Lx: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion
    ref = cmp["_ref_masks"]
    r_m, o_m = cmp["_pairs"]
    cells = iscell[:, 0] > 0 if iscell is not None and len(iscell) else np.ones(len(stat), bool)
    rgba = np.zeros((Ly, Lx, 4))

    def edge(flat):
        m = np.zeros(Ly * Lx, bool)
        m[flat] = True
        m = m.reshape(Ly, Lx)
        return m & ~binary_erosion(m)
    matched_ref = set(r_m.tolist())
    matched_ours = set(o_m.tolist())
    for i, m in enumerate(ref):
        rgba[edge(m)] = (1, 0.6, 0, 1) if i in matched_ref else (1, 0.15, 0.15, 1)
    for j, s in enumerate(stat):
        if not cells[j]:
            continue
        e = edge(np.unique(np.asarray(s["ypix"]) * Lx + np.asarray(s["xpix"])))
        rgba[e] = (0.2, 0.8, 1, 1) if j in matched_ours else (0.2, 0.4, 1, 1)
    fig, ax = plt.subplots(figsize=(6.6, 7.0))
    a, b = np.nanpercentile(bg, [1, 99.5])
    ax.imshow(np.clip((bg - a) / (b - a + 1e-9), 0, 1), cmap="gray")
    ax.imshow(rgba, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"{sid}: {cmp['n_matched']} of {cmp['n_reference_rois']} reference ROIs matched "
                 f"({100 * cmp['fraction_reference_matched']:.0f} %, IoU > {IOU_MATCH})", fontsize=9)
    fig.tight_layout()
    p = out / f"{sid}_reference_rois.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return {"path": p.name, "caption": (
        f"Our cells compared with the reference segmentation in {cmp['reference'].split(':')[0]} (for the "
        "Allen demo: the Allen Brain Observatory's own ROIs of this session). Orange: reference ROIs that "
        "one of our cells matches (intersection over union > 0.5); red: reference ROIs we missed; cyan: "
        "our matching cells; blue: our cells with no reference partner (cells the reference left out, or "
        f"false positives). {cmp['n_matched']} of {cmp['n_reference_rois']} reference ROIs are matched "
        f"({100 * cmp['fraction_reference_matched']:.0f} %); {100 * cmp['fraction_our_cells_matched']:.0f} % of "
        "our cells have a reference partner. A shorter recording than the reference's finds fewer cells "
        "(cells that never fire in the window cannot be detected functionally).")}
