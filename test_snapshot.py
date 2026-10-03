"""
Demo snapshot (ui/snapshot.py, review P4): stable keys, hits and misses, version checks, and the bundled snapshot
running the whole app offline.
"""

import gzip
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ui import snapshot

ROOT = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A small snapshot file in tmp_path, switched on for this test only."""
    monkeypatch.delenv(snapshot.DISABLE_ENV, raising=False)
    monkeypatch.delenv(snapshot.RECORD_ENV, raising=False)
    path = tmp_path / "snapshot.pkl.gz"
    monkeypatch.setattr(snapshot, "PATH", path)
    snapshot.reset()

    def write(data: dict, meta: dict = None):
        with gzip.open(path, "wb") as f:
            pickle.dump({"meta": meta or {**snapshot.versions(), "prices_as_of": "2026-09-30", "built": "2026-10-03"},
                         "data": data}, f)
        snapshot.reset()
    yield write
    snapshot.reset()


def test_key_is_canonical_and_sensitive():
    s = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2026-01-01", periods=3))
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    k = snapshot.key("f", (s, frame), {"c": {"y": 2, "x": 1}, "n": None})
    assert k == snapshot.key("f", (s.copy(), frame.copy()), {"n": None, "c": {"x": 1, "y": 2}})  # dict order ignored
    assert k != snapshot.key("f", (s * 1.000001, frame), {"c": {"y": 2, "x": 1}, "n": None})   # a changed value
    assert k != snapshot.key("g", (s, frame), {"c": {"y": 2, "x": 1}, "n": None})              # another function
    assert k != snapshot.key("f", (s.rename("r"), frame), {"c": {"y": 2, "x": 1}, "n": None})  # a renamed series
    assert snapshot.key("f", (np.array([1.0, np.nan]),), {}) == snapshot.key("f", (np.array([1.0, np.nan]),), {})


def test_key_is_the_same_in_another_process():
    code = ("import pandas as pd; from ui import snapshot; "
            "print(snapshot.key('f', (pd.Series([1.5, 2.5], index=pd.date_range('2026-01-01', periods=2)),), {'p': 'max'}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert out == snapshot.key("f", (pd.Series([1.5, 2.5], index=pd.date_range("2026-01-01", periods=2)),), {"p": "max"})


def test_heavy_returns_the_stored_result_and_computes_a_miss(store):
    calls = []

    def calc(x, scale=2.0):
        calls.append(x)
        return x * scale

    named = {"x": 3.0, "scale": 2.0}
    store({snapshot.key("calc", (), named): "from the snapshot"})
    wrapped = snapshot.heavy(calc)
    assert wrapped(3.0) == "from the snapshot" and wrapped(x=3.0, scale=2.0) == "from the snapshot"  # same key
    assert calls == []
    assert wrapped(4.0) == 8.0 and calls == [4.0]  # not in the snapshot: computed live
    assert snapshot.misses() == ["calc"]


def test_download_follows_the_mode_and_returns_a_copy(store, monkeypatch):
    def _fetch(ticker, period="2y"):
        return {"source": "live", "df": pd.DataFrame({"Close": [1.0]})}

    stored = {"source": "snapshot", "df": pd.DataFrame({"Close": [9.0]})}
    store({snapshot.key("fetch", (), {"ticker": "ABC.NS", "period": "2y"}): stored})
    fetch = snapshot.download(_fetch)
    first = fetch("ABC.NS")
    assert first["source"] == "snapshot"
    first["df"].loc[0, "Close"] = -1.0  # a caller changing its copy must not change the store
    assert fetch("ABC.NS")["df"].loc[0, "Close"] == 9.0
    assert fetch("NEW.NS")["source"] == "live"
    monkeypatch.setattr(snapshot, "mode", lambda: snapshot.LIVE)
    assert fetch("ABC.NS")["source"] == "live"


def test_incompatible_or_disabled_snapshot_is_ignored(store, monkeypatch):
    store({"k": 1}, meta={"pandas": "0.0.1", "numpy": "1.0", "prices_as_of": "2026-09-30"})
    assert snapshot.load() is None and snapshot.mode() == snapshot.LIVE
    store({"k": 1})
    assert snapshot.load() is not None
    monkeypatch.setenv(snapshot.DISABLE_ENV, "1")
    snapshot.reset()
    assert snapshot.load() is None


def test_compatible_checks_pandas_and_numpy_major():
    now = snapshot.versions()
    assert snapshot.compatible({"pandas": now["pandas"], "numpy": now["numpy"]})
    assert snapshot.compatible({"pandas": now["pandas"], "numpy": now["numpy"].split(".")[0] + ".99.0"})
    assert not snapshot.compatible({"pandas": "1.5.3", "numpy": now["numpy"]})


BUNDLED = os.path.join(ROOT, "data", "snapshot", "snapshot.pkl.gz")


@pytest.mark.skipif(not os.path.exists(BUNDLED), reason="no bundled snapshot (scripts/build_snapshot.py)")
def test_bundled_snapshot_runs_every_page_offline(monkeypatch):
    """The demo snapshot alone (network blocked by conftest) renders every page of the default portfolio."""
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    from ui.pages import PAGE_KEY, PAGES
    monkeypatch.delenv(snapshot.DISABLE_ENV, raising=False)
    monkeypatch.setattr(snapshot, "PATH", __import__("pathlib").Path(BUNDLED))
    snapshot.reset()
    st.cache_data.clear()
    if snapshot.load() is None:
        pytest.skip("bundled snapshot built with another pandas/numpy version")
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=600)
    at.run()
    next(r for r in at.sidebar.radio if r.label == "Analysis Mode").set_value("Portfolio")
    at.run()
    for page in PAGES:
        at.radio(key=PAGE_KEY).set_value(page)
        at.run()
        assert not at.exception, f"{page}: {at.exception[0].value}"
        assert not at.error, f"{page}: {at.error[0].value}"
    # Every result from the snapshot when running on the versions it was built with (elsewhere, e.g. CI on another
    # numpy, a last-digit difference in a derived input can send a calculation live, which is allowed)
    built = snapshot.meta()
    if all(built.get(k) == v for k, v in snapshot.versions().items()):
        assert snapshot.misses() == [], f"computed live instead of from the snapshot: {sorted(set(snapshot.misses()))}"
    snapshot.reset()
    st.cache_data.clear()
