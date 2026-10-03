"""
Build the demo snapshot (ui/snapshot.py): run the app on every preset with the default settings, visit every page,
and record the downloads and every heavy cached result in data/snapshot/snapshot.pkl.gz.

    python scripts/build_snapshot.py

Needs internet (Yahoo Finance). Then check it offline: `pytest test_snapshot.py`. Rebuild after changing any
calculation, or to move the snapshot date forward.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["RISK_TOOL_SNAPSHOT_RECORD"] = "1"  # before the app's modules are imported
os.environ.pop("RISK_TOOL_NO_SNAPSHOT", None)

import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from ui import snapshot  # noqa: E402
from ui.pages import PAGE_KEY, PAGES  # noqa: E402

PRESETS = ["ASIANPAINT.NS", "BRITANNIA.NS", "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "AAPL", "MSFT", "TSLA"]
PORTFOLIO = "Portfolio"


def visit(target: str) -> list:
    """Run the app for one preset (a ticker, or the default portfolio) and render every page; returns any errors."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=1800)
    at.run()
    if target == PORTFOLIO:
        next(r for r in at.sidebar.radio if r.label == "Analysis Mode").set_value("Portfolio")
    else:
        at.sidebar.text_input[0].set_value(target)
    at.run()
    problems = []
    for page in PAGES:
        at.radio(key=PAGE_KEY).set_value(page)
        at.run()
        problems += [f"{target} / {page}: {e.value}" for e in list(at.exception) + list(at.error)]
    return problems


def holdings_as_of(recorded: dict):
    """The latest date of the presets' own prices (not of FX or futures series, which trade at weekends)."""
    wanted = set(PRESETS) | {"^NSEI", "^GSPC"}
    return max(v["df"]["Date"].max() for v in recorded.values()
               if isinstance(v, dict) and v.get("symbol") in wanted and isinstance(v.get("df"), pd.DataFrame)
               and not v["df"].empty)


def main() -> int:
    started = time.time()
    problems = []
    for target in PRESETS + [PORTFOLIO]:
        problems += visit(target)
        print(f"{target}: done ({time.time() - started:.0f}s, {len(snapshot._recorded)} entries)", flush=True)
    if problems:
        print("Not saved; the app reported errors:\n" + "\n".join(problems))
        return 1
    meta = snapshot.save(prices_as_of=holdings_as_of(snapshot._recorded))
    size = snapshot.PATH.stat().st_size / 1e6
    print(f"saved {snapshot.PATH} ({size:.1f} MB): {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
