"""Shared inputs for every pillar: fundamentals (with CSV overrides), disclosures and data-quality scores."""

import streamlit as st

from data_quality import assess_holding
from disclosures import load_disclosures
from fundamentals import apply_overrides
from ui.cache import cached_fundamentals
from ui.context import export

OVERRIDES_KEY = "fundamental_overrides"  # session state: the uploaded override table, if any


def load_foundations(ctx):
    """Fundamentals, disclosures and data-quality scores for every holding, computed once for all pages."""
    price_frames = ctx.price_frames

    override_table = st.session_state.get(OVERRIDES_KEY)
    fundamentals_by_ticker, override_errors = {}, []
    with st.spinner("Loading fundamentals from Yahoo Finance..."):
        for ticker in price_frames:
            fund = cached_fundamentals(ticker)
            if override_table is not None:
                fund, errors = apply_overrides(fund, override_table)
                override_errors = errors  # the same for every ticker: they come from validating the whole upload
            fundamentals_by_ticker[ticker] = fund

    quality = {t: assess_holding(df, fundamentals_by_ticker.get(t)) for t, df in price_frames.items()}
    disclosures = load_disclosures()
    export(ctx, locals())
