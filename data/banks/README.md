# Bank and NBFC metrics

One CSV per entity, `<SYMBOL>.csv`, used by `banks.py` (RBI Prompt Corrective Action bands, early warnings, trends;
docs/methodology.md §10.6).

| Column | Meaning |
| --- | --- |
| symbol | NSE symbol |
| entity_type | `bank` or `nbfc` |
| d_sib | `True` for a Domestic Systemically Important Bank (Tier-1 leverage minimum 4% instead of 3.5%) |
| period_end | the balance-sheet date the value refers to |
| metric | one of gnpa, nnpa, pcr, crar, cet1, tier1, leverage, casa, cd_ratio, lcr, nim, roa, slippage |
| value_pct | the value in percent |
| filing_date | when it became public (point in time) |
| source | the XBRL URL, or "Annual report (URL), PDF page N" with any note |

**Bundled:**
- **HDFC Bank:**
  - XBRL NPA ratios and ROA, Jun 2018 – Dec 2024;
  - FY2022–FY2026 annual reports, for capital, LCR, CASA, NIM, provision coverage and the derived credit-deposit ratio.
- **Yes Bank:**
  - XBRL Jun 2018 – Jun 2020, plus Dec 2021. Later filings failed the scale check (see the methodology).
  - FY2018–FY2019 annual reports.

**Refresh the XBRL rows:** `python scripts/fetch_bank_results.py HDFCBANK:bank:dsib YESBANK:bank`. Rows transcribed from
annual reports are kept.

**Add an entity:** copy the columns above, one row per metric and period, and give the source and page for every
figure. Nothing is estimated: a metric that is not printed stays missing.
