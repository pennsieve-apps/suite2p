# sample-input

The input contract is in `workflow-systems-neuroscience/docs/conventions.md` ("Calcium imaging").
Put one or more movies in the `movies` input; anything else (sessions.csv, `<stem>_timestamps.csv`,
a companion NWB) is optional.

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/suite2p/
```

Two sessions with planted cells and planted motion: a 700-frame 128 x 128 TIFF at 10 Hz (24 cells,
frame times from 12.5 s in `<stem>_timestamps.csv`) and a 400-frame 96 x 96 NWB TwoPhotonSeries at
15 Hz (12 cells, subject and indicator in the file). `data/fixture/truth/` holds the planted masks,
spikes and shifts.

## Real data: Allen Brain Observatory (the demo)

DANDI:000728 (Allen Institute, Visual Coding - Optical Physiology, CC-BY-4.0,
doi:10.48324/dandi.000728/0.240827.1809), session 503109347 (StimA), mouse 496001363, VISp L2/3,
GCaMP6f. The workflow repository stages it (`scripts/stage-demo-data.py allen-2p`):

- `sub-496001363_ses-503109347-StimA_desc-dg1-bin2_2p.tif`: frames 0-18,809 of the raw NWB's
  `MotionCorrectedTwoPhotonSeries` (asset `c2e07e77-8317-4a3e-b0b4-8d25dfb4988b`, 42.9 GB, streamed),
  binned 2:1 in time: 9,405 frames, 512 x 512, 15.04 Hz, 4.93 GB, BigTIFF.
- `..._timestamps.csv`: frame times on the session clock (5.74-631.01 s).
- `sub-496001363/sub-496001363_ses-503109347-StimA_behavior+image+ophys.nwb` (asset
  `d8e398ec-9b42-4369-a232-8d1ac7a0f80d`, 834 MB): the companion NWB, with Allen's 353 ROIs, which
  this app compares its cells with.

Link all of them into `movies`. The movie is already motion-corrected by Allen, so registration
shows near-zero shifts. Use the default `TAU=0.7` (GCaMP6f).

Any NWB with a raw `TwoPhotonSeries` on DANDI works too, e.g. DANDI:000347 (macaque V1/V2, 7 Hz,
GCaMP6s: set `TAU=1.25`).
