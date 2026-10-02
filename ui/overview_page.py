"""
Overview page. Until the CRO dashboard (roadmap Phase 6), it shows the market-risk summary cards,
the positions, the data-quality score of each holding, the disclosure files loaded, and each
holding's fundamentals with the source of every figure.
"""

import io

import pandas as pd
import streamlit as st

from disclosures import DATASETS, template
from fundamentals import FIELDS_BY_NAME, PROFILE_FIELDS, STATEMENTS, fundamentals_template, statement_table
from ui.foundations import OVERRIDES_KEY
from ui.overview import render_overview
from ui.pages import BUILT, COMING_NEXT, PAGES

STATEMENT_TITLES = {"income": "Income statement", "balance": "Balance sheet", "cashflow": "Cash flow"}


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def render(ctx):
    render_overview(ctx)
    _positions(ctx)
    _data_quality(ctx)
    _fundamentals(ctx)
    _disclosures(ctx)
    _pillar_status()


def _positions(ctx):
    st.markdown("### 💼 Positions")
    table = ctx.positions.rename(columns={"Price": f"Price ({ctx.currency})", "Value": f"Value ({ctx.currency})"})
    st.dataframe(table.style.format({"Quantity": "{:,.2f}", f"Price ({ctx.currency})": "{:,.2f}",
                                     f"Value ({ctx.currency})": "{:,.0f}", "Weight": "{:.1%}"}),
                 hide_index=True, width="stretch")
    entered = "the weights entered" if ctx.entry_mode == "Weight" else f"the {ctx.entry_mode.lower()} entered"
    st.caption(f"Converted from {entered} at the latest close ({ctx.prices_as_of:%d %b %Y}). "
               "Sector is Yahoo Finance's classification, where it reports one.")


def _data_quality(ctx):
    st.markdown("### 🧪 Data quality")
    rows = [{"Ticker": t, "Score": q["score"], "Volume From": " + ".join(ctx.volume_sources.get(t, [t])),
             "Issues": "; ".join(q["reasons"]) or "no issues found"}
            for t, q in ctx.quality.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d")})
    st.caption("100 minus penalties for reversing spikes, stale prices, zero-volume days, gaps, short history and "
               "missing fundamentals. The thresholds are assumptions (docs/methodology.md, section 7). The lowest score "
               "feeds every trust grade. Volume for Indian stocks is NSE + BSE where Yahoo has both histories; "
               "Yahoo often has no usable BSE history for large caps, and then NSE volume is used alone.")
    with st.expander("Every check, per holding"):
        for ticker, q in ctx.quality.items():
            st.markdown(f"**{ticker}**: {q['score']}/100")
            st.dataframe(q["checks"], hide_index=True, width="stretch")


def _fundamentals(ctx):
    st.markdown("### 📑 Fundamentals and sources")
    tickers = list(ctx.fundamentals_by_ticker)
    ticker = st.selectbox("Holding", tickers, key="fundamentals_ticker") if len(tickers) > 1 else tickers[0]
    fund = ctx.fundamentals_by_ticker[ticker]

    if fund["errors"] and not len(fund["table"]):
        st.warning("Yahoo Finance returned no fundamentals for this holding (" + "; ".join(fund["errors"]) + "). "
                   "Upload them with the CSV template below.")
    profile = pd.DataFrame([{"Field": PROFILE_FIELDS[name][1], "Value": "not available" if p["value"] is None else
                             (f"{p['value']:,.0f}" if isinstance(p["value"], float) else p["value"]), "Source": p["source"]}
                            for name, p in fund["profile"].items()])
    st.dataframe(profile, hide_index=True, width="stretch")

    for statement in STATEMENTS:
        wide = statement_table(fund, statement)
        st.markdown(f"**{STATEMENT_TITLES[statement]}** (fiscal years ending)")
        if wide.empty:
            st.caption("not available")
        else:
            st.dataframe(wide.style.format("{:,.0f}", na_rep="not available"), width="stretch")

    if len(fund["table"]):
        with st.expander("Source of every figure"):
            sources = fund["table"].assign(Field=fund["table"]["Field"].map(lambda f: FIELDS_BY_NAME[f].label))
            st.dataframe(sources.style.format({"Value": "{:,.0f}", "Period End": "{:%Y-%m-%d}",
                                               "Filing Date": lambda d: "unknown" if pd.isna(d) else f"{d:%Y-%m-%d}"}),
                         hide_index=True, width="stretch")
    missing = [FIELDS_BY_NAME[f].label if f in FIELDS_BY_NAME else PROFILE_FIELDS[f][1] for f in fund["missing_fields"]]
    st.caption("Not available: " + (", ".join(missing) if missing else "none") +
               ". Missing figures are never estimated; later pillars skip what they cannot compute.")

    st.markdown("**Override figures with your own (CSV)**")
    st.caption("Download the template, fill in only the rows you want to replace (one row per field and fiscal year, "
               "values in full currency units, with the filing date if known and where the figure came from), and upload it. "
               "Uploaded figures replace Yahoo's field by field for every holding whose ticker matches.")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download fundamentals template", _csv(fundamentals_template(ticker)),
                           file_name="fundamentals_template.csv", mime="text/csv")
    with col2:
        upload = st.file_uploader("Upload overrides", type="csv", key="fundamentals_upload", label_visibility="collapsed")
    file_id = getattr(upload, "file_id", None)
    if file_id != st.session_state.get("fundamentals_upload_id"):
        st.session_state["fundamentals_upload_id"] = file_id
        st.session_state[OVERRIDES_KEY] = pd.read_csv(io.BytesIO(upload.getvalue()), dtype=str) if upload else None
        st.rerun()
    if ctx.override_errors:
        st.warning("Some uploaded rows were rejected:\n\n" + "\n".join(f"- {e}" for e in ctx.override_errors[:20]))


def _disclosures(ctx):
    st.markdown("### 🗂️ Disclosure data (India)")
    disclosures = ctx.disclosures
    if disclosures["files"].empty:
        st.info("No disclosure files are loaded yet. Pledges, surveillance lists, price bands, the F&O ban list, rating "
                "actions and auditor events come only from files you download from NSE, BSE and the rating agencies; "
                "the tool never fills them in. Save each file in data/disclosures/ and list it in manifest.json "
                "(see data/disclosures/README.md).")
    else:
        st.dataframe(disclosures["files"], hide_index=True, width="stretch")
    if disclosures["issues"]:
        st.warning("Disclosure files with problems:\n\n" + "\n".join(f"- {i}" for i in disclosures["issues"][:20]))
    with st.expander("Where each dataset comes from, and templates"):
        name = st.selectbox("Dataset", list(DATASETS), format_func=lambda n: DATASETS[n].title, key="disclosure_dataset")
        spec = DATASETS[name]
        st.markdown(f"**Official source:** {spec.official_source}")
        st.markdown(f"**How to download:** {spec.download_steps}")
        st.markdown("**Columns:** " + ", ".join(f"`{c.name}`" + ("" if c.required else " (optional)") for c in spec.columns))
        st.download_button(f"Download {spec.title.lower()} template", _csv(template(name)),
                           file_name=f"{name}_template.csv", mime="text/csv")


def _pillar_status():
    st.markdown("### 🧭 Pillars")
    rows = [{"Page": page, "Status": "built" if page in BUILT else f"coming next (Phase {COMING_NEXT[page][0]})"}
            for page in PAGES]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
