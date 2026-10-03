"""
Demo snapshot (review P4): precomputed results for the presets, so the hosted app answers instantly.

- `scripts/build_snapshot.py` runs the app on every preset with the default settings and records the downloads and
  the result of every heavy cached function in data/snapshot/snapshot.pkl.gz, with the date of the last price.
- In "Demo snapshot" mode (the default when the file is present), downloads come from the snapshot, so the heavy
  functions see the same inputs and their stored results are reused. A ticker or setting that is not in the snapshot
  is downloaded and computed live as usual.
- Heavy results are keyed on a hash of their inputs, not on the mode, so a stored result is only ever returned for
  exactly the inputs it was computed from.

The file is a pickle written by our own build script and shipped in the repository; it is never loaded from
anywhere else. It is ignored (live mode) if it is missing, or was built with a different pandas or numpy major
version, or if RISK_TOOL_NO_SNAPSHOT is set (the test suite sets it).
"""

import copy
import functools
import gzip
import hashlib
import inspect
import os
import pickle
import platform
import threading
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parent.parent / "data" / "snapshot" / "snapshot.pkl.gz"
SNAPSHOT, LIVE = "snapshot", "live"
MODE_KEY = "data_mode"  # session-state key of the sidebar choice
RECORD_ENV, DISABLE_ENV = "RISK_TOOL_SNAPSHOT_RECORD", "RISK_TOOL_NO_SNAPSHOT"

_lock = threading.Lock()
_store = None  # {"meta": {...}, "data": {key: result}}, loaded once per process
_recorded = {}
_misses = []


# ---------------------------------------------------------------
# Keys
# ---------------------------------------------------------------

def _feed(h, obj) -> None:
    """Feed a canonical byte form of `obj` into the hash: equal inputs give equal keys in any process."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        names = obj.columns if isinstance(obj, pd.DataFrame) else [obj.name]
        h.update(f"{type(obj).__name__}|{list(map(str, names))}|"
                 f"{list(map(str, (obj.dtypes if isinstance(obj, pd.DataFrame) else [obj.dtype])))}|{obj.shape}".encode())
        try:
            h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        except TypeError:  # unhashable cells (e.g. dicts): fall back to the pickle of the values
            h.update(pickle.dumps(obj, protocol=4))
    elif isinstance(obj, np.ndarray):
        h.update(f"nd|{obj.dtype}|{obj.shape}".encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        h.update(f"dict|{len(obj)}".encode())
        for k in sorted(obj, key=repr):
            _feed(h, k)
            _feed(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(f"{type(obj).__name__}|{len(obj)}".encode())
        for item in obj:
            _feed(h, item)
    else:
        h.update(f"{type(obj).__name__}|{obj!r}".encode())


def key(name: str, args: tuple, kwargs: dict) -> str:
    h = hashlib.sha256(name.encode())
    _feed(h, list(args))
    _feed(h, kwargs)
    return h.hexdigest()


# ---------------------------------------------------------------
# The store
# ---------------------------------------------------------------

def versions() -> dict:
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}


def compatible(meta: dict) -> bool:
    """Same pandas version and numpy major version as the build (pickles of pandas objects need the former)."""
    now = versions()
    return meta.get("pandas") == now["pandas"] and str(meta.get("numpy", "")).split(".")[0] == now["numpy"].split(".")[0]


def recording() -> bool:
    return bool(os.environ.get(RECORD_ENV))


def load(path: Path = None):
    """The snapshot, or None when it is missing, disabled or incompatible (then the app runs live)."""
    global _store
    if os.environ.get(DISABLE_ENV) or recording():
        return None
    path = Path(path or PATH)  # looked up at call time (tests point PATH elsewhere)
    with _lock:
        if _store is None:
            store = {}
            if path.exists():
                try:
                    with gzip.open(path, "rb") as f:
                        candidate = pickle.load(f)
                    store = candidate if compatible(candidate.get("meta", {})) else {}
                except Exception:
                    store = {}
            _store = store
    return _store or None


def reset() -> None:
    """Forget the loaded snapshot and the recording (tests)."""
    global _store
    with _lock:
        _store = None
        _recorded.clear()
        _misses.clear()


def meta():
    store = load()
    return store["meta"] if store else None


def mode() -> str:
    """This session's choice: the snapshot when it is available and not switched off in the sidebar."""
    if load() is None:
        return LIVE
    try:
        import streamlit as st
        return st.session_state.get(MODE_KEY, SNAPSHOT)
    except Exception:
        return SNAPSHOT


def lookup(name: str, args: tuple, kwargs: dict):
    """(True, result) if the snapshot holds this call, else (False, None)."""
    store = load()
    if store is None:
        return False, None
    k = key(name, args, kwargs)
    if k in store["data"]:
        return True, store["data"][k]
    _misses.append(name)
    return False, None


def record(name: str, args: tuple, kwargs: dict, result) -> None:
    if recording():
        _recorded[key(name, args, kwargs)] = result


def misses() -> list:
    return list(_misses)


def save(path: Path = None, prices_as_of=None) -> dict:
    """Write everything recorded in this process; returns the meta."""
    import datetime
    info = {**versions(), "built": datetime.datetime.now().isoformat(timespec="seconds"),
            "prices_as_of": None if prices_as_of is None else pd.Timestamp(prices_as_of).isoformat(),
            "entries": len(_recorded)}
    path = Path(path or PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=9) as f:
        pickle.dump({"meta": info, "data": dict(_recorded)}, f, protocol=4)
    return info


# ---------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------

def _bound(fn, args, kwargs) -> dict:
    """Arguments by name with defaults filled in, so f(x, "max") and f(x, period="max") share a key."""
    bound = inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def heavy(fn):
    """
    For a cached calculation: reuse the snapshot's result for exactly these inputs; record it when building.
    Used under st.cache_data, which hands each caller its own copy of the result.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        named = _bound(fn, args, kwargs)
        hit, result = lookup(fn.__name__, (), named)
        if hit:
            return result
        result = fn(*args, **kwargs)
        record(fn.__name__, (), named, result)
        return result
    return wrapper


def download(live_fn):
    """
    For a download: in snapshot mode, the stored download (prices to the snapshot date); otherwise, or for a call the
    snapshot does not hold, the live (Streamlit-cached) one. Not itself cached, so the mode is read on every call.
    """
    name = live_fn.__name__.lstrip("_")

    @functools.wraps(live_fn)
    def wrapper(*args, **kwargs):
        named = _bound(live_fn, args, kwargs)
        if mode() == SNAPSHOT:
            hit, result = lookup(name, (), named)
            if hit:
                return copy.deepcopy(result)  # callers may modify what they get; the store must not change
        result = live_fn(*args, **kwargs)
        record(name, (), named, result)
        return result
    return wrapper
