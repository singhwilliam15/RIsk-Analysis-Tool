"""
Run the whole app once, as Streamlit Cloud would on a cold start, and report the peak memory and the time taken.

    python scripts/measure_app.py                 # demo snapshot, network blocked (what test_deploy.py checks)
    python scripts/measure_app.py --live          # live downloads and calculations, for comparison (needs internet)
    python scripts/measure_app.py --ticker TCS.NS # one stock instead of the default portfolio

Prints one JSON line: peak resident memory (MB), seconds to the first page and to all pages, and any errors.
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()
    if args.live:
        os.environ["RISK_TOOL_NO_SNAPSHOT"] = "1"
    else:
        os.environ.pop("RISK_TOOL_NO_SNAPSHOT", None)
        real_connect = socket.socket.connect

        def guarded(sock, address, *a, **k):
            if isinstance(address, tuple) and str(address[0]) not in ("127.0.0.1", "::1", "localhost"):
                raise RuntimeError(f"network blocked in snapshot mode (tried {address})")
            return real_connect(sock, address, *a, **k)
        socket.socket.connect = guarded

    import psutil
    from streamlit.testing.v1 import AppTest
    from ui import snapshot
    from ui.pages import PAGE_KEY, PAGES

    proc = psutil.Process()
    peak = proc.memory_info().rss
    started = time.time()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=1800)
    at.run()
    if args.ticker:
        at.sidebar.text_input[0].set_value(args.ticker)
    else:
        next(r for r in at.sidebar.radio if r.label == "Analysis Mode").set_value("Portfolio")
    at.run()
    first = time.time() - started
    errors = []
    for page in PAGES:
        at.radio(key=PAGE_KEY).set_value(page)
        at.run()
        errors += [f"{page}: {e.value}"[:300] for e in list(at.exception) + list(at.error)]
        peak = max(peak, proc.memory_info().rss)
    info = proc.memory_info()
    peak = max(peak, getattr(info, "peak_wset", 0) or 0, info.rss)
    print(json.dumps({"mode": "live" if args.live else "snapshot", "target": args.ticker or "default portfolio",
                      "peak_mb": round(peak / 1e6), "first_page_s": round(first, 1),
                      "all_pages_s": round(time.time() - started, 1), "snapshot_misses": sorted(set(snapshot.misses())),
                      "same_versions": bool(snapshot.meta()) and all(snapshot.meta().get(k) == v
                                                                      for k, v in snapshot.versions().items()),
                      "errors": errors}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
