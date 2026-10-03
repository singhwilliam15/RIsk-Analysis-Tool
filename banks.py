"""
Banks and NBFCs (credit pillar): the metrics that matter for lenders, RBI's Prompt Corrective Action (PCA) bands
with the distance to each trigger, early-warning rules and five-year trends. Merton, Altman and leverage ratios
do not apply to them: deposits and borrowings are their business.

Data: one long CSV per entity in data/banks/<SYMBOL>.csv (template below). Rows come from NSE results XBRL
(scripts/fetch_bank_results.py) or are transcribed from annual reports with their page; a row is used only once
its filing date has passed (point in time). Thresholds: config/pca_thresholds.json, with the RBI circulars.

Methods: docs/methodology.md section 10.6.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BANKS_DIR = Path(__file__).parent / "data" / "banks"
PCA_PATH = Path(__file__).parent / "config" / "pca_thresholds.json"
COLUMNS = ["symbol", "entity_type", "d_sib", "period_end", "metric", "value_pct", "filing_date", "source"]
METRICS = {
    "gnpa": "Gross NPA %", "nnpa": "Net NPA %", "pcr": "Provision coverage %", "crar": "CRAR %", "cet1": "CET1 ratio %",
    "tier1": "Tier-1 capital ratio %", "leverage": "Tier-1 leverage ratio %", "casa": "CASA %",
    "cd_ratio": "Credit-deposit ratio %", "lcr": "LCR %", "nim": "Net interest margin %", "roa": "ROA %",
    "slippage": "Slippage ratio %",
}
BANDS = ("none", "RT1", "RT2", "RT3")
# Early-warning buffers (assumptions): how close to a PCA trigger counts as a warning
EARLY_WARNING = {"capital_buffer_pp": 1.0, "nnpa_buffer_pp": 1.0, "gnpa_rise_pp": 1.0}


def load_pca(path: Path = PCA_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def template() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load_bank(symbol: str, directory: Path = BANKS_DIR) -> pd.DataFrame:
    """The metrics file for `symbol` (without the .NS suffix), typed; empty if there is none."""
    path = Path(directory) / f"{symbol}.csv"
    if not path.exists():
        return template()
    frame = pd.read_csv(path, dtype={"source": str})
    unknown = set(frame["metric"]) - set(METRICS)
    if unknown:
        raise ValueError(f"{path.name}: unknown metric(s) {sorted(unknown)}")
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    frame["value_pct"] = pd.to_numeric(frame["value_pct"], errors="coerce")
    frame["d_sib"] = frame["d_sib"].astype(str).str.lower().isin(("true", "1", "yes"))
    return frame


def public(frame: pd.DataFrame, as_of) -> pd.DataFrame:
    """Rows public by `as_of` (filing date on or before it)."""
    if frame.empty or as_of is None:
        return frame
    return frame[frame["filing_date"] <= pd.Timestamp(as_of)]


def latest(frame: pd.DataFrame, as_of=None) -> dict:
    """Latest public value of each metric: {metric: (value, period_end, source)}. Later filings of the same
    period (e.g. an annual report after the results) win."""
    rows = public(frame, as_of).dropna(subset=["value_pct"])
    if rows.empty:
        return {}
    rows = rows.sort_values(["period_end", "filing_date"])
    return {m: (float(r["value_pct"]), r["period_end"], r["source"]) for m, r in rows.groupby("metric").tail(1).set_index("metric").iterrows()}


def band(value: float, spec: dict, d_sib: bool = False) -> tuple:
    """(band, distance to the RT1 trigger in pp). Distance is the headroom: positive = safe, negative = breached."""
    edges = spec.get("edges_dsib") if d_sib and "edges_dsib" in spec else spec["edges"]
    if value is None or not np.isfinite(value):
        return "not available", np.nan
    direction = spec["direction"]
    if direction == "below":
        hits = [value < e for e in edges]
        distance = value - edges[0]
    elif direction == "at_or_above":
        hits = [value >= e for e in edges]
        distance = edges[0] - value
    else:  # strictly above
        hits = [value > e for e in edges]
        distance = edges[0] - value
    level = sum(hits)  # edges are nested, so the count of breached edges is the band
    return BANDS[level], float(distance)


def pca_status(values: dict, entity_type: str, d_sib: bool = False, config: dict = None) -> dict:
    """PCA band and distance to the RT1 trigger for every indicator of `entity_type` ('bank' or 'nbfc')."""
    config = config or load_pca()
    spec = config[entity_type]
    rows = []
    for key, ind in spec["indicators"].items():
        value = values.get(key, (np.nan,))[0] if key in values else np.nan
        b, distance = band(value, ind, d_sib)
        edges = ind.get("edges_dsib") if d_sib and "edges_dsib" in ind else ind["edges"]
        rows.append({"Indicator": ind["label"], "Key": key, "Value": value, "Band": b, "Distance to trigger (pp)": distance,
                     "RT1 trigger": edges[0], "Rule": ind["basis"]})
    table = pd.DataFrame(rows)
    known = table[table["Band"] != "not available"]
    worst = max(known["Band"], key=BANDS.index) if len(known) else "not available"
    return {"table": table, "worst": worst, "circular": spec["circular"], "url": spec["url"],
            "min_headroom": float(known["Distance to trigger (pp)"].min()) if len(known) else np.nan,
            "coverage": f"{len(known)} of {len(table)} indicators"}


def early_warnings(frame: pd.DataFrame, entity_type: str, d_sib: bool = False, as_of=None, config: dict = None) -> list:
    """Rules of thumb short of a PCA breach (assumptions, EARLY_WARNING), plus the LCR minimum and a loss."""
    config = config or load_pca()
    values = latest(frame, as_of)
    status = pca_status(values, entity_type, d_sib, config)
    out = []
    for r in status["table"].to_dict("records"):
        if r["Band"] in ("none",) and np.isfinite(r["Distance to trigger (pp)"]):
            buffer = EARLY_WARNING["nnpa_buffer_pp"] if r["Key"] == "nnpa" else EARLY_WARNING["capital_buffer_pp"]
            if r["Distance to trigger (pp)"] < buffer:
                out.append(f"{r['Indicator']} {r['Value']:.2f}% is within {buffer:g} pp of its PCA trigger ({r['RT1 trigger']:g}%)")
    trend = history(frame, "gnpa", as_of)
    if len(trend) >= 2:
        # Compare with the period about a year earlier
        last = trend.iloc[-1]
        prior = trend[trend["period_end"] <= last["period_end"] - pd.Timedelta(days=330)]
        if len(prior) and last["value_pct"] - prior.iloc[-1]["value_pct"] >= EARLY_WARNING["gnpa_rise_pp"]:
            out.append(f"Gross NPA up {last['value_pct'] - prior.iloc[-1]['value_pct']:.2f} pp in a year "
                       f"({prior.iloc[-1]['value_pct']:.2f}% → {last['value_pct']:.2f}%)")
    if "roa" in values and values["roa"][0] < 0:
        out.append(f"Loss-making: ROA {values['roa'][0]:.2f}%")
    if "lcr" in values and values["lcr"][0] < config["lcr_minimum"]["value"]:
        out.append(f"LCR {values['lcr'][0]:.0f}% is below the {config['lcr_minimum']['value']:.0f}% minimum")
    return out


def history(frame: pd.DataFrame, metric: str, as_of=None) -> pd.DataFrame:
    """One value per period for `metric` (the latest filing of each period), oldest first."""
    rows = public(frame, as_of)
    rows = rows[(rows["metric"] == metric)].dropna(subset=["value_pct"])
    return rows.sort_values(["period_end", "filing_date"]).groupby("period_end").tail(1).reset_index(drop=True)


def trend_table(frame: pd.DataFrame, as_of=None, years: int = 5) -> pd.DataFrame:
    """Fiscal-year-end values (31 March) of every metric for the last `years` years: metrics as rows."""
    rows = public(frame, as_of)
    annual = rows[rows["period_end"].dt.month == 3]
    if annual.empty:
        return pd.DataFrame()
    annual = annual.sort_values("filing_date").groupby(["metric", "period_end"]).tail(1)
    wide = annual.pivot_table(index="metric", columns="period_end", values="value_pct", aggfunc="last")
    wide = wide[sorted(wide.columns)[-years:]]
    wide.index = [METRICS.get(m, m) for m in wide.index]
    wide.columns = [f"FY{c.year}" for c in wide.columns]
    return wide


def assess(symbol: str, as_of=None, directory: Path = BANKS_DIR, config: dict = None) -> dict:
    """Everything the Credit page needs for one bank or NBFC; {'available': False} without a file."""
    frame = load_bank(symbol, directory)
    rows = public(frame, as_of)
    if rows.empty:
        return {"available": False, "symbol": symbol}
    entity_type = rows["entity_type"].iloc[-1]
    d_sib = bool(rows["d_sib"].iloc[-1])
    config = config or load_pca()
    values = latest(frame, as_of)
    return {"available": True, "symbol": symbol, "entity_type": entity_type, "d_sib": d_sib,
            "values": values, "pca": pca_status(values, entity_type, d_sib, config),
            "warnings": early_warnings(frame, entity_type, d_sib, as_of, config),
            "trend": trend_table(frame, as_of), "latest_period": rows["period_end"].max()}
