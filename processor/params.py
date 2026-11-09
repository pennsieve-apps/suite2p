"""
Parameters from the environment (names as in app.yml), validated. A bad value raises
BadParameter, which main() turns into exit code 2.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

DETECTION = ("functional", "sourcery", "anatomical")


class BadParameter(ValueError):
    pass


def _get(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


def _bool(name: str, default: bool) -> bool:
    v = _get(name, "true" if default else "false").lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise BadParameter(f"{name}={v!r}: expected true or false")


def _num(name: str, default: float, lo: float | None = None, hi: float | None = None,
         integer: bool = False, allow_empty: bool = False):
    raw = _get(name, "" if default is None else str(default))
    if raw == "" and allow_empty:
        return None
    try:
        v = float(raw)
    except ValueError:
        raise BadParameter(f"{name}={raw!r}: expected a number") from None
    if not math.isfinite(v):
        raise BadParameter(f"{name}={raw!r}: expected a finite number")
    if integer:
        if v != int(v):
            raise BadParameter(f"{name}={raw!r}: expected a whole number")
        v = int(v)
    if lo is not None and v < lo:
        raise BadParameter(f"{name}={raw!r}: must be >= {lo:g}")
    if hi is not None and v > hi:
        raise BadParameter(f"{name}={raw!r}: must be <= {hi:g}")
    return v


@dataclass
class Params:
    fs: float               # 0 = from timestamps / metadata
    tau: float
    nplanes: int            # 0 = from ScanImage metadata, else 1
    nchannels: int          # 0 = from ScanImage metadata, else 1
    functional_chan: int
    t_start: float | None
    t_stop: float | None
    do_registration: bool
    nonrigid: bool
    block_size: int
    batch_size: int
    detection: str
    diameter: float         # 0 = auto
    threshold_scaling: float
    nbins: int
    classifier_threshold: float
    keep_binary: bool
    use_gpu: str
    threads: int
    seed: int

    def as_env(self) -> dict:
        names = {"fs": "FS", "tau": "TAU", "nplanes": "NPLANES", "nchannels": "NCHANNELS",
                 "functional_chan": "FUNCTIONAL_CHAN", "t_start": "T_START", "t_stop": "T_STOP",
                 "do_registration": "DO_REGISTRATION", "nonrigid": "NONRIGID", "block_size": "BLOCK_SIZE",
                 "batch_size": "BATCH_SIZE", "detection": "DETECTION", "diameter": "DIAMETER",
                 "threshold_scaling": "THRESHOLD_SCALING", "nbins": "NBINS", "classifier_threshold": "CLASSIFIER_THRESHOLD",
                 "keep_binary": "KEEP_BINARY", "use_gpu": "USE_GPU", "threads": "THREADS", "seed": "SEED"}
        return {names[k]: v for k, v in asdict(self).items()}


def read_params() -> Params:
    detection = _get("DETECTION", "functional").lower()
    if detection not in DETECTION:
        raise BadParameter(f"DETECTION={detection!r}: expected one of {', '.join(DETECTION)}")
    use_gpu = _get("USE_GPU", "auto").lower()
    if use_gpu in ("1", "yes", "cuda", "gpu", "on"):
        use_gpu = "true"
    elif use_gpu in ("0", "no", "cpu", "off"):
        use_gpu = "false"
    if use_gpu not in ("auto", "true", "false"):
        raise BadParameter(f"USE_GPU={use_gpu!r}: expected auto, true or false")
    p = Params(
        fs=_num("FS", 0, lo=0),
        tau=_num("TAU", 0.7, lo=0.01, hi=10),
        nplanes=_num("NPLANES", 0, lo=0, hi=100, integer=True),
        nchannels=_num("NCHANNELS", 0, lo=0, hi=2, integer=True),
        functional_chan=_num("FUNCTIONAL_CHAN", 1, lo=1, hi=2, integer=True),
        t_start=_num("T_START", None, allow_empty=True),
        t_stop=_num("T_STOP", None, allow_empty=True),
        do_registration=_bool("DO_REGISTRATION", True),
        nonrigid=_bool("NONRIGID", True),
        block_size=_num("BLOCK_SIZE", 128, lo=16, hi=1024, integer=True),
        batch_size=_num("BATCH_SIZE", 25, lo=10, hi=5000, integer=True),
        detection=detection,
        diameter=_num("DIAMETER", 0, lo=0, hi=200),
        threshold_scaling=_num("THRESHOLD_SCALING", 1.0, lo=0.05, hi=20),
        nbins=_num("NBINS", 5000, lo=100, hi=50000, integer=True),
        classifier_threshold=_num("CLASSIFIER_THRESHOLD", 0.5, lo=0, hi=1),
        keep_binary=_bool("KEEP_BINARY", False),
        use_gpu=use_gpu,
        threads=_num("THREADS", 0, lo=0, integer=True),
        seed=_num("SEED", 42, lo=0, integer=True),
    )
    if p.t_start is not None and p.t_stop is not None and p.t_stop <= p.t_start:
        raise BadParameter(f"T_STOP={p.t_stop} must be greater than T_START={p.t_start}")
    if p.functional_chan == 2 and p.nchannels == 1:
        raise BadParameter("FUNCTIONAL_CHAN=2 needs NCHANNELS=2 (or 0 = from the file's metadata)")
    return p
