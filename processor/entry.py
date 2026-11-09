"""
Container entry point. Before numpy, torch or matplotlib are imported it

- points every writable cache at $TMPDIR (Lambda-safe: the file system is read-only except
  INPUT_DIR, OUTPUT_DIR, HOME and TMPDIR, which the runtime sets; nothing the image put in /tmp
  or HOME exists there), and
- fixes the thread count from THREADS (or the cgroup CPU quota, never the host's core count),

then runs main() once.
"""
import os
import sys
import tempfile

# the runtime sets HOME and TMPDIR but may not create them
for _v in ("HOME", "TMPDIR"):
    if os.environ.get(_v):
        os.makedirs(os.environ[_v], exist_ok=True)
_tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
for _var, _sub in (("MPLCONFIGDIR", "mpl"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "cache"),
                   ("TORCH_HOME", "torch")):
    _path = os.path.join(_tmp, _sub)
    os.makedirs(_path, exist_ok=True)
    os.environ[_var] = _path
os.environ.setdefault("USER", "app")          # getpass() under an arbitrary uid
os.environ.setdefault("MPLBACKEND", "Agg")

from processor.resources import resolve_threads, set_thread_env  # noqa: E402

_raw = os.environ.get("THREADS", "").strip()
_n, _source = resolve_threads(int(_raw) if _raw.isdigit() else 0)
set_thread_env(_n)

from processor.main import main  # noqa: E402

sys.exit(main())
