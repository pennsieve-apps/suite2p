"""
QC figures per plane (PNG, 150 dpi, session id in the title), each with the caption the report
shows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DPI = 150


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _norm(img: np.ndarray, lo=1, hi=99.5) -> np.ndarray:
    img = np.asarray(img, float)
    a, b = np.nanpercentile(img, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-12), 0, 1)


def _full(img, ops, fill=0.0):
    """A cropped (yrange x xrange) image placed back into the full frame."""
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    if img is None:
        return None
    img = np.asarray(img)
    if img.shape == (Ly, Lx):
        return img
    out = np.full((Ly, Lx), fill, float)
    y0, y1 = ops.get("yrange", [0, Ly])
    x0, x1 = ops.get("xrange", [0, Lx])
    h, w = min(img.shape[0], y1 - y0), min(img.shape[1], x1 - x0)
    out[y0:y0 + h, x0:x0 + w] = img[:h, :w]
    return out


def _name(sid, plane, nplanes, what):
    return f"{sid}_{what}.png" if nplanes == 1 else f"{sid}_plane{plane}_{what}.png"


def plane_figures(sid: str, plane: int, nplanes: int, ops: dict, stat, iscell, times, sdir: Path) -> list[dict]:
    plt = _plt()
    figs = []
    label = sid if nplanes == 1 else f"{sid}, plane {plane}"
    mean = ops.get("meanImg")
    maxp = _full(ops.get("max_proj"), ops)
    vcorr = _full(ops.get("Vcorr"), ops)

    # 1. mean / max / correlation images
    panels = [(mean, "mean image (registered)"), (maxp, "max projection"), (vcorr, "correlation map")]
    panels = [(im, t) for im, t in panels if im is not None]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), squeeze=False)
    for ax, (im, t) in zip(axs[0], panels):
        ax.imshow(_norm(im), cmap="gray", interpolation="nearest")
        ax.set_title(t, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"{label}: field of view", fontsize=11)
    fig.tight_layout()
    p = sdir / _name(sid, plane, nplanes, "mean_image")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    figs.append({"path": p.name, "caption": (
        f"Field of view of {label}. Left: the mean of all registered frames, which shows the "
        "anatomy (cell bodies as rings or blobs, blood vessels as dark lines). Middle: the maximum "
        "projection of the binned movie used for detection; bright, compact spots are cells that "
        "were active at least once. Right: Suite2p's correlation map; cells stand out where "
        "neighbouring pixels fluctuate together. A sharp mean image means registration worked; a "
        "blurred one means residual motion.")})

    # 2. ROI overlay
    bg = maxp if maxp is not None else mean
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ax.imshow(_norm(bg), cmap="gray", interpolation="nearest")
    n_acc = n_rej = 0
    if stat is not None and len(stat):
        Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
        acc_rgba = np.zeros((Ly, Lx, 4))
        rej_rgba = np.zeros((Ly, Lx, 4))
        rng = np.random.default_rng(0)
        is_cell = iscell[:, 0] > 0 if iscell is not None and len(iscell) else np.ones(len(stat), bool)
        for s, ok in zip(stat, is_cell):
            yp, xp = s["ypix"][~s["overlap"]] if "overlap" in s else s["ypix"], \
                     s["xpix"][~s["overlap"]] if "overlap" in s else s["xpix"]
            if ok:
                acc_rgba[yp, xp, :3] = plt.cm.hsv(rng.uniform(0, 1))[:3]
                acc_rgba[yp, xp, 3] = 0.55
                n_acc += 1
            else:
                rej_rgba[yp, xp] = (0.6, 0.6, 0.6, 0.35)
                n_rej += 1
        ax.imshow(rej_rgba, interpolation="nearest")
        ax.imshow(acc_rgba, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"{label}: {n_acc} cells (colour), {n_rej} other ROIs (grey)", fontsize=10)
    fig.tight_layout()
    p = sdir / _name(sid, plane, nplanes, "rois")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    figs.append({"path": p.name, "caption": (
        f"Regions of interest (ROIs) found in {label}, drawn on the max projection. Coloured: ROIs "
        "Suite2p's built-in classifier calls cells (probability at or above CLASSIFIER_THRESHOLD); "
        "grey: the other ROIs (dendrites, out-of-focus cells, noise), kept in the output but not "
        "analysed downstream. Look for coloured ROIs on every visible cell body and few on blood "
        "vessels or the image edge.")})

    # 3. registration offsets
    if "yoff" in ops and len(ops["yoff"]):
        t = np.asarray(times, float)
        n = min(len(t), len(ops["yoff"]))
        t = t[:n]
        rows = 3 if "yoff1" in ops and ops["yoff1"] is not None and np.size(ops["yoff1"]) else 2
        fig, axs = plt.subplots(rows, 1, figsize=(9, 2.2 * rows + 0.6), sharex=True)
        axs[0].plot(t, np.asarray(ops["yoff"])[:n], lw=0.6, label="y")
        axs[0].plot(t, np.asarray(ops["xoff"])[:n], lw=0.6, label="x")
        axs[0].set_ylabel("rigid shift (px)")
        axs[0].legend(loc="upper right", fontsize=8, frameon=False)
        r = 1
        if rows == 3:
            y1, x1 = np.asarray(ops["yoff1"])[:n], np.asarray(ops["xoff1"])[:n]
            nr = np.hypot(y1, x1)
            axs[1].plot(t, nr.max(axis=1), lw=0.6, color="tab:red", label="largest block")
            axs[1].plot(t, nr.mean(axis=1), lw=0.6, color="tab:purple", label="mean over blocks")
            axs[1].set_ylabel("non-rigid (px)")
            axs[1].legend(loc="upper right", fontsize=8, frameon=False)
            r = 2
        if "corrXY" in ops:
            axs[r].plot(t, np.asarray(ops["corrXY"])[:n], lw=0.6, color="k")
            axs[r].set_ylabel("corr. with ref.")
        axs[-1].set_xlabel("time (s, session clock)")
        extra = ""
        if "regDX" in ops and ops["regDX"] is not None and len(ops["regDX"]):
            extra = f"; PC drift after registration: max {float(np.max(ops['regDX'])):.2f} px"
        fig.suptitle(f"{label}: motion correction{extra}", fontsize=11)
        fig.tight_layout()
        p = sdir / _name(sid, plane, nplanes, "registration")
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        figs.append({"path": p.name, "caption": (
            f"Motion correction of {label}. Top: the rigid shift of each frame (pixels) that "
            "registration removed. Middle: the extra non-rigid (piecewise) shift, largest and mean over "
            "blocks of the image. Bottom: how well each registered frame matches the reference image; "
            "dips are frames with large or out-of-plane motion (e.g. during running or licking). Shifts "
            "of a few pixels are normal; steps or slow drifts of many pixels suggest z-drift, which "
            "registration cannot fix. A movie that was motion-corrected before (as Allen's) shows "
            "near-zero shifts.")})
    return figs
