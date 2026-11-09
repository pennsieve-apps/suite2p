# Third-party software

**This repository is GPL-3.0-or-later** (see `LICENSE`): the app imports Suite2p (GPL-3.0) in the
same process (`import suite2p`, `suite2p.run_s2p.run_plane`), so the app is a work based on Suite2p
and is licensed compatibly. Our code calls Suite2p as a library; no Suite2p code is copied here. The
image contains the following third-party components, installed from PyPI at the pinned versions in
`requirements.txt`; their licences apply to the image. Two images are built (`TORCH_FLAVOUR`): the
**cpu** variant has the CPU-only PyTorch wheel and no NVIDIA software; the **gpu** variant has the
CUDA 12.6 build of PyTorch with the **NVIDIA CUDA libraries**, proprietary but redistributable
under the NVIDIA licence terms shipped in each wheel.

| Component | What | Licence | Source |
|---|---|---|---|
| Suite2p 1.1.0 (Carsen Stringer, Marius Pachitariu and contributors; © Howard Hughes Medical Institute) | Registration, ROI detection, extraction, deconvolution, classification; called as a library. | GPL-3.0 | https://github.com/MouseLand/suite2p |
| Suite2p built-in ROI classifier (`suite2p/classifiers/classifier.npy`, inside the suite2p wheel) | The cell/non-cell classifier applied to every ROI (`use_builtin_classifier`). | GPL-3.0 (part of Suite2p) | https://github.com/MouseLand/suite2p |
| Cellpose 4.2.1.1 (Stringer, Pachitariu et al.) | A dependency of Suite2p, used only with DETECTION=anatomical. **No Cellpose model weights are bundled** (see below). | BSD-3-Clause | https://github.com/MouseLand/cellpose |
| segment-anything 1.0 (Meta AI) | Imported by Cellpose 4 (its ViT backbone code); no weights. | Apache-2.0 | https://github.com/facebookresearch/segment-anything |
| PyTorch 2.14.1 (`+cpu` or `+cu126`), torchvision 0.29.1; triton 3.8.0 (gpu) | Tensor computation on CPU or GPU. | BSD-3-Clause style (torch's licence bundle); triton MIT | https://pytorch.org |
| nvidia-*-cu12, cuda-toolkit 12.6.3, cuda-bindings, cuda-pathfinder (gpu variant only; versions in `requirements-cuda.txt`) | CUDA runtime libraries. | NVIDIA proprietary, redistributable (licence in each wheel); cuda-bindings/pathfinder Apache-2.0 | https://developer.nvidia.com/cuda-toolkit |
| scanimage-tiff-reader 1.4.1.4 (© MBF Bioscience) | Suite2p's ScanImage TIFF reader (installed with Suite2p). | Apache-2.0 | https://pypi.org/project/scanimage-tiff-reader/ |
| tifffile 2026.11.8, imagecodecs 2026.10.10, roifile 2026.11.6 (Christoph Gohlke) | Reading TIFF/BigTIFF and its ScanImage/ImageJ metadata. | BSD-3-Clause | https://github.com/cgohlke |
| h5py 3.16.0 | Reading HDF5 and NWB movies. | BSD-3-Clause | https://www.h5py.org |
| NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, numba 0.68.1, llvmlite 0.50.0 | Arrays, filtering, PCA (registration metrics), JIT. | BSD-3-Clause / BSD-2-Clause (llvmlite, with LLVM's Apache-2.0 WITH LLVM-exception) | PyPI |
| opencv-python-headless 5.0.0.93, fastremap 1.21.0, fill_voids 2.1.2 | Image operations for Suite2p/Cellpose. | Apache-2.0 (OpenCV), LGPL-3.0 (fastremap, fill_voids; used unmodified as libraries) | PyPI |
| Matplotlib 3.11.2, Pillow 13.0.0 | QC figures. | Matplotlib licence (PSF-based); MIT-CMU | https://matplotlib.org |
| natsort, tqdm, joblib, threadpoolctl, importlib_metadata, zipp, contourpy, cycler, fonttools, kiwisolver, packaging, pyparsing, python-dateutil, cloudpickle, narwhals, filelock, fsspec, Jinja2, MarkupSafe, mpmath, networkx, sympy, setuptools, typing_extensions, tzdata | Dependencies of the above (versions in `requirements.txt`). | MIT / BSD / Apache-2.0 / PSF / MPL-2.0+MIT (tqdm) | PyPI |
| python:3.12.16-slim-bookworm | Base image (Debian 12 + CPython 3.12). | PSF; Debian packages under their own free licences | https://hub.docker.com/_/python |

## Not bundled

- **Cellpose-SAM model weights.** Hugging Face (`mouseland/cellpose-sam`) tags the weights BSD-3-Clause,
  but Cellpose's README states that "All Cellpose models are trained on data that is licensed under
  CC-BY-NC". Because that does not clearly allow redistribution for every use, the weights are not
  in the image. DETECTION=anatomical uses a model file the lab supplies in the `cellpose-model` input.
- **Demo data** (not in the image or the repository): DANDI:000728, Allen Institute, Visual Coding -
  Optical Physiology (CC-BY-4.0), doi:10.48324/dandi.000728/0.240827.1809.
