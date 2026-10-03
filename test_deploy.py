"""
Deployment check (review P4): the app, started cold in demo-snapshot mode with the network blocked, renders every
page of the default portfolio within Streamlit Community Cloud's smallest memory allocation.

Streamlit's documentation ("Manage your app", resource limits) gives 690 MB minimum and 2.7 GB maximum memory per
app; the check uses the minimum, so the app fits whatever the platform allocates.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(ROOT, "data", "snapshot", "snapshot.pkl.gz")
CLOUD_MIN_MEMORY_MB = 690


@pytest.mark.skipif(not os.path.exists(SNAPSHOT), reason="no bundled snapshot (scripts/build_snapshot.py)")
def test_cold_start_fits_the_cloud_memory_minimum():
    pytest.importorskip("psutil")
    env = {k: v for k, v in os.environ.items() if k != "RISK_TOOL_NO_SNAPSHOT"}
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "measure_app.py")], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=1800)
    lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
    assert lines, out.stdout[-2000:] + out.stderr[-2000:]
    result = json.loads(lines[-1])
    print(result)
    assert result["errors"] == [], result["errors"]
    if result.get("same_versions"):
        assert result["snapshot_misses"] == [], f"computed live: {result['snapshot_misses']}"
    assert result["peak_mb"] < CLOUD_MIN_MEMORY_MB, f"peak {result['peak_mb']} MB"
