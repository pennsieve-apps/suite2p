# suite2p

This app runs **[Suite2p](https://github.com/MouseLand/suite2p)**, developed by **Carsen Stringer
and Marius Pachitariu** with Chris Ki, Nicholas Del Grosso, Paul LaFosse and Qingqing Zhang at
**HHMI Janelia Research Campus** (Stringer et al. 2026, *Extracting large-scale neural activity with
Suite2p*, doi:10.64898/2026.02.04.703741; the original method: Pachitariu et al. 2016,
doi:10.1101/061507). Pennsieve packages Suite2p as a processor for its Two-photon Calcium Imaging
workflow; it did not develop the method, and the credit is the Suite2p team's. The demo data are
from the **Allen Brain Observatory** (Allen Institute, Visual Coding - Optical Physiology, DANDI:000728,
doi:10.48324/dandi.000728/0.240827.1809; de Vries et al. 2020, doi:10.1038/s41593-019-0550-9).

The first step of the **Two-photon Calcium Imaging** workflow:

- `suite2p → ophys-traces → ophys-activity → neuro-report`

A two-photon movie is gigabytes of frames in which a few hundred neurons flicker. Suite2p turns it
into a list of cells and their fluorescence over time: it removes the brain's motion (registration),
finds the pixels that belong together because they light up together (cell detection), decides
which of those regions look like cell bodies (a built-in classifier), and measures each cell's
fluorescence and that of the neuropil around it. The next app, `ophys-traces`, turns those traces
into ΔF/F and spike rates.

For the computational analyst: Suite2p 1.1.0 (PyTorch; GPU when available), per plane:
reference image → rigid + non-rigid (block-wise) phase-correlation registration → `sparsery`
functional detection on the time-binned movie (bins of `TAU × FS` frames) → overlap removal →
ROI and neuropil extraction (`neuropil_coefficient` 0.7, inner radius 2 px, ≥ 350 neuropil pixels)
→ OASIS deconvolution (`spks.npy`, unscaled) → the built-in classifier (`classifier.npy` shipped in
the wheel). The app writes Suite2p's own folder layout, so the output also opens in Suite2p's GUI.

## Ports

| Port | Direction | Content |
|---|---|---|
| `movies` | input (subfolder, required) | Movies, one session each: multi-page TIFF/BigTIFF/OME-TIFF (a ScanImage series `<name>_00001.tif`, `<name>_00002.tif`, ... is one movie), HDF5 (`data` or the largest 3-D dataset), or NWB with a `TwoPhotonSeries` in `acquisition`. Port media types: `image/tiff`, `image/tiff; profile=ome-tiff`, `application/x-hdf5`, `application/x-hdf5; profile=nwb`, `text/csv`. A Bruker PrairieView folder is also read (through Suite2p's own reader) but, untested on real data, is not declared on the port (`application/vnd.pennsieve.bruker-tiff`); Micro-Manager NDTiff is not supported. Optional: `<movie stem>_timestamps.csv` (`frame, time`), `sessions.csv`, a companion NWB of the same session (identity, and reference ROIs to compare with). |
| `cellpose-model` | input (subfolder, optional) | A Cellpose model file, only for `DETECTION=anatomical`. |
| `suite2p` | output | `suite2p/<session_id>/...` (below), `suite2p/sessions.csv`, `summary.json`, `provenance.json`. |

Inputs are found by name and content anywhere under `INPUT_DIR` (following symlinks), so they may
also be merged at its root.

**Sessions.** One movie = one session. `session_id` comes from `sessions.csv` (a row whose `path`
names the file or folder), else `sub-<a>_ses-<b>` in the file or folder name (a UUID session label is
cut to 8 characters), else the file stem. An NWB without a movie whose session id matches is the
movie's companion: its `session_start_time`, `identifier`, subject, indicator and so on go into
`session.json`, and its ROIs (if any) are compared with ours.

**Time.** Frame times, in seconds on the session clock, come from (in order) `<stem>_timestamps.csv`
(one row per frame as stored, or per volume), the NWB `TwoPhotonSeries` timestamps or
`starting_time + i / rate`, else `i / FS` from 0. `FS` = 0 takes the frame rate from those times or
from the file's metadata (ScanImage `scanFrameRate`/`scanVolumeRate`, ImageJ `finterval`, a JSON
description with `frame_rate_hz`, HDF5 attributes `fs`/`frame_rate`/`rate`). A time window
(`T_START`, `T_STOP`) keeps the times; nothing is re-zeroed.

## Outputs

`suite2p/<session_id>/`:

| File | Content |
|---|---|
| `plane<k>/ops.npy, stat.npy, F.npy, Fneu.npy, spks.npy, iscell.npy` | Suite2p's standard outputs (ROIs × frames). `iscell.npy` column 0 is 1 when the classifier probability (column 1) ≥ `CLASSIFIER_THRESHOLD`. `ops.npy` also holds the registration offsets (`yoff`, `xoff`, `yoff1`, `xoff1`, `corrXY`), summary images (`meanImg`, `max_proj`, `Vcorr`) and registration metrics (`regDX`). |
| `plane<k>/timestamps.npy` | Time of each frame of this plane (s, session clock). |
| `plane<k>/data.bin` | Only with `KEEP_BINARY=true`: the registered movie (int16). |
| `timestamps.npy` | Time of each volume (plane 0), session clock. |
| `session.json` | `frame_rate`, `frame_rate_source`, `effective_frame_rate`, `n_frames`, `n_planes`, `n_channels`, `Lx`, `Ly`, `timestamps_source`, `t_start`, `t_stop`, `first_frame`, `intensity_scale` (0.5 for uint16 movies: Suite2p halves them to int16), `tau`, `block_size`, `detection`, `source_nwb` (identity of the NWB movie or companion), `notes`. |
| `<session_id>_mean_image.png`, `_rois.png`, `_registration.png`, `_reference_rois.png` | QC figures (captions in `summary.json`). |

`suite2p/sessions.csv`: `session_id, path, subject_id, condition, group, status, n_frames,
n_planes, frame_rate, duration_s, n_rois, n_cells` and the sheet's other columns.
`summary.json` per conventions.md plus `reference_roi_match` (below); `provenance.json` with the
exact keys of conventions.md.

**Reference ROIs.** When a companion NWB holds a segmentation (`processing/ophys/ImageSegmentation`,
as Allen's processed files do), our cells are paired one-to-one with its ROIs by maximal
intersection-over-union (Hungarian assignment); a pair with IoU > 0.5 is a match. `summary.json` →
`reference_roi_match` gives, per session, the fraction of reference ROIs matched by our cells (the
headline), the fraction of our cells that have a reference partner, and the median IoU. There is no
independent, recent benchmark of two-photon segmentation, so agreement with the dataset's own
curated segmentation of the same session is the evidence we show.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `FS` | `0` | Frame rate per plane (Hz); 0 = from the frame times or metadata (the run fails with a clear message when there is neither). |
| `TAU` | `0.7` | Indicator decay time (s): 0.7 GCaMP6f, 1.0 GCaMP6m, 1.25-1.5 GCaMP6s, about 0.3-0.5 GCaMP8f/m (Suite2p's documentation). It sets the time binning for detection and Suite2p's own deconvolution. |
| `NPLANES`, `NCHANNELS` | `0` | Interleaved planes / channels; 0 = from ScanImage metadata (fast-z slices, saved channels, flyback frames dropped), else 1. |
| `FUNCTIONAL_CHAN` | `1` | With two channels, the one with the indicator. |
| `T_START`, `T_STOP` | empty | Optional time window (s, session clock). |
| `DO_REGISTRATION` | `true` | Motion correction (turn off for movies registered before). |
| `NONRIGID` | `true` | Non-rigid registration after the rigid step. |
| `BLOCK_SIZE` | `128` | Non-rigid block size (px); lowered automatically so a side has at least two blocks. |
| `BATCH_SIZE` | `25` (cpu), `100` (gpu) | Frames registered at a time; memory grows with it. |
| `DETECTION` | `functional` | `functional` (Suite2p's `sparsery`, the default and what the Suite2p paper evaluates), `sourcery`, or `anatomical` (Cellpose on the mean/max image; needs the `cellpose-model` input). |
| `DIAMETER` | `0` | Expected cell diameter (px); 0 = automatic. For `sparsery` it sets the spatial scale (6/12/24/48 px). |
| `THRESHOLD_SCALING` | `1.0` | Higher = fewer, stronger ROIs. |
| `CLASSIFIER_THRESHOLD` | `0.5` | Classifier probability above which an ROI is a cell. |
| `KEEP_BINARY` | `false` | Keep the registered movie. |
| `SEED`, `THREADS`, `USE_GPU` | `42`, `0`, `false` (cpu) / `auto` (gpu) | Reproducibility; CPU threads (0 = the container's CPU quota); device. |

## Variants

| Variant | Compute | Image | When |
|---|---|---|---|
| `gpu` (default) | NVIDIA T4 or newer, 15 GB RAM | CUDA 12.6 PyTorch | Suite2p 1.x is PyTorch end to end; registration, detection and extraction run on the GPU. Falls back to CPU (logged `USE_GPU=auto: cuda not available, using cpu`) when no GPU is visible. `BATCH_SIZE=100` fits a T4's 16 GB; raise it on a larger card. |
| `cpu` | 4 vCPU, 8 GB | CPU-only PyTorch | No GPU available; about 2.5-3 min per 1,000 frames of 512 × 512 (below). |

## Runtime

Measured on the Allen demo (512 × 512, GCaMP6f, 15.04 Hz) with the `cpu` image in amd64 emulation
on an Apple M-series Mac (Docker, 4 vCPU), so real amd64 CPUs are faster:

RUNTIME_TABLE

GPU numbers: to be measured on the dev compute node. The research notes estimate 15-25 min for
30,000 frames on one T4-class GPU.

Scratch: Suite2p's int16 binary (as large as the raw movie: 2 bytes × frames × Ly × Lx, e.g. 4.9 GB
for the full demo) is written to `$TMPDIR` and deleted at the end (or moved into the output with
`KEEP_BINARY=true`). Within Lambda's 10 GB and 15 min limits only short movies fit; the app is
sized for standard/GPU tasks.

## Results on the demo

DEMO_RESULTS

## Limitations

- Functional detection finds cells that are **active** in the recording; silent cells are missed
  (a short window finds fewer cells). Anatomical detection (Cellpose) needs a model you provide.
- The classifier is Suite2p's generic one, trained on mouse cortex somata; dendrites, unusual cell
  types or other species may be misclassified. ROIs below the threshold are kept, not deleted.
- uint16 movies are halved to int16, as Suite2p does (`intensity_scale` 0.5): ΔF/F is unaffected,
  absolute F values are half the camera's.
- NWB movies are read as one plane and one channel; ScanImage multi-ROI (mesoscope) recordings are
  read as one field of view; Bruker folders go through Suite2p's reader without a time window
  (Bruker support is untested on real data). Suite2p 1.1.0's sbx and nd2 readers still have the
  pre-1.0 interface, so those formats are not supported.
- Suite2p's registration metrics sample 2,000-5,000 frames (over 4 GB for 512 × 512); this app
  computes them on 500 frames spread over the recording.
- A movie registered before (as Allen's) shows near-zero shifts: that is expected.

## Licences

GPL-3.0-or-later (`LICENSE`), because the app imports Suite2p (GPL-3.0) in-process; see
`THIRD_PARTY.md` for every bundled component. Suite2p's built-in classifier is part of the Suite2p
wheel. **Cellpose-SAM weights are not bundled**: Cellpose's README says all Cellpose models are
trained on CC-BY-NC data, which does not clearly allow redistribution for every use.

## Citations

If you use this app, cite Suite2p and the works below (also in `CITATION.cff`):

- Stringer C, Ki C, Del Grosso N, LaFosse P, Zhang Q, Pachitariu M (2026). Extracting large-scale neural activity with Suite2p. *bioRxiv*. doi:10.64898/2026.02.04.703741 (the paper Suite2p asks you to cite)
- Pachitariu M, Stringer C, Dipoppa M, Schröder S, Rossi LF, Dalgleish H, Carandini M, Harris KD (2016). Suite2p: beyond 10,000 neurons with standard two-photon microscopy. *bioRxiv*. doi:10.1101/061507
- Friedrich J, Zhou P, Paninski L (2017). Fast online deconvolution of calcium imaging data. *PLOS Computational Biology* 13:e1005423. doi:10.1371/journal.pcbi.1005423, PMID: 28291787 (OASIS, Suite2p's `spks.npy`)
- With `DETECTION=anatomical`: Stringer C, Wang T, Michaelos M, Pachitariu M (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods* 18:100-106. doi:10.1038/s41592-020-01018-x; and Pachitariu M, Rariden M, Stringer C (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. *bioRxiv*. doi:10.1101/2025.04.28.651001
- Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L, Svoboda K, Frank L, Bouchard KE (2022). The Neurodata Without Borders ecosystem for neurophysiological data science. *eLife* 11:e78362. doi:10.7554/eLife.78362, PMID: 36193886
- Demo data: Baker C (2024). Allen Institute - Visual Coding - Optical Physiology. DANDI Archive. doi:10.48324/dandi.000728/0.240827.1809 (CC-BY-4.0); and de Vries SEJ, Lecoq JA, Buice MA, Groblewski PA, Ocker GK, Oliver M, Feng D, Cain N, Ledochowitsch P, Millman D, Roll K, Garrett M, Keenan T, Kuan L, Mihalas S, Olsen S, Thompson C, Wakeman W, Waters J, Williams D, Barber C, Berbesque N, Blanchard B, Bowles N, Caldejon SD, Casal L, Cho A, Cross S, Dang C, Dolbeare T, Edwards M, Galbraith J, Gaudreault N, Gilbert TL, Griffin F, Hargrave P, Howard R, Huang L, Jewell S, Keller N, Knoblich U, Larkin JD, Larsen R, Lau C, Lee E, Lee F, Leon A, Li L, Long F, Luviano J, Mace K, Nguyen T, Perkins J, Robertson M, Seid S, Shea-Brown E, Shi J, Sjoquist N, Slaughterbeck C, Sullivan D, Valenza R, White C, Williford A, Witten DM, Zhuang J, Zeng H, Farrell C, Ng L, Bernard A, Phillips JW, Reid RC, Koch C (2020). A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. *Nature Neuroscience* 23:138-151. doi:10.1038/s41593-019-0550-9, PMID: 31844315
