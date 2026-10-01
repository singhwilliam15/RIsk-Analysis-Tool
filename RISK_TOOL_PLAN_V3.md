# Risk Analysis Tool — Plan v3: full risk coverage, integrated and honest

This replaces both RISK_TOOL_UPGRADE_PLAN.md and RISK_TOOL_PLAN_V2.md. Use only this file.

## Design principles

The tool covers every major risk type, but avoids the two weaknesses of most risk tools:

1. **Silos.** Most tools report market, credit and liquidity risk on separate pages, as if they were independent. In a real crisis they hit together: prices fall → volume dries up (liquidity) → leverage and distance to default worsen (credit) → pledged promoter shares hit margin calls and get sold (event risk) → prices fall further. This tool links the pillars through one **integrated stress engine**.
2. **False precision.** Every headline number carries a **confidence range and an A–D trust grade**, and every pillar has a "what this metric misses" panel.

## Structure

| Layer | Contents |
| --- | --- |
| Pillar 1 · Market risk | Existing engine (8 models, backtests, Euler decomposition, stress replay) + crisis correlation |
| Pillar 2 · Liquidity risk | SEBI/AMFI days-to-liquidate, stressed volume, LVaR, Amihud, circuit-lock (India) |
| Pillar 3 · Credit risk | Merton DD/PD, Altman Z / Z'', credit ratios with red flags, rating actions, portfolio expected loss |
| Pillar 4 · Concentration & factor risk | HHI, sector risk, factor betas (Indian/US factor data), PCA, effective number of bets |
| Pillar 5 · Event & governance risk | Promoter pledges, ASM/GSM surveillance, auditor events, F&O ban, jump-risk overlay |
| Trust layer (cross-cutting) | Confidence ranges, model risk, data quality and A–D grades for every pillar |
| Integration & decisions | Linked multi-pillar stress, reverse stress test, risk-change explanation, best risk-reducing trades, limits, CRO dashboard and memo |
| Evidence | Case studies on real Indian blow-ups, with hits, misses and false positives |
| *(optional)* FX risk | Mixed INR/USD portfolios |

**Out of scope (Quant Risk Terminal):** options pricing, Greeks, optimisation, risk parity.

## Master prompt

```text
Read RISK_TOOL_PLAN_V3.md in this repo. It turns this VaR tool into a full Risk Analysis Tool
with five risk pillars, a cross-cutting trust layer, an integrated stress and decision layer,
and evidence from case studies. Work through Phases 0-8 in order (Phase 8 is optional).

Rules for the whole job:
- Create a branch `full-risk-tool`, run `pytest` and confirm everything passes first.
- ONE phase at a time. Use plan mode: show me the plan for the phase, wait for approval,
  implement, run pytest, start the app briefly in Single Stock and Portfolio mode, commit, and
  summarise with REAL outputs for RELIANCE.NS, HDFCBANK.NS, ASIANPAINT.NS, one small-cap and
  the default 5-stock portfolio. Then STOP and wait for me. Never push.
- Architecture: one top-level module per pillar (liquidity.py, credit.py, concentration.py,
  events.py), plus trust.py, integration.py and decisions.py; UI pages in ui/; heavy work cached
  in ui/cache.py; results on ctx. Keep every existing test, the Excel export and the live
  Streamlit deployment working (pinned requirements; only packages with wheels for 3.11/3.12).
- Every pillar page has the same layout: headline metrics (with trust grade) → detail →
  "What this metric misses" panel → methodology link. Every pillar adds a sheet to the Excel
  report.
- Every calculation gets offline tests (network mocked) against an independent reference:
  a closed form, scipy, a simulation with known truth, or a hand-worked or published example.
- POINT-IN-TIME DISCIPLINE: analysis "as of" a date may only use data public on that date
  (filings by filing date; balance sheets no earlier than fiscal year end + 60 days if the
  filing date is unknown). Test this wherever dates matter.
- NEVER invent data. Fundamentals come from yfinance or my CSV uploads. Indian disclosure data
  (pledges, surveillance lists, price bands, rating actions, auditor events) come from official
  sources (NSE, BSE, rating agencies) that I download, or from a refresh script only if the
  site allows it; otherwise give me a CSV template and exact download steps. Show source and
  as-of date for every external figure. Missing fields show "not available", never a guess.
- Assumptions (jump sizes, event probabilities, impact coefficients, thresholds) are labelled
  "assumption" in the UI, editable, and justified in docs/methodology.md. Formula constants
  (Altman coefficients, zone cut-offs) cite their published source.
- Update README.md and docs/methodology.md at the end of every phase.

Start with Phase 0.
```

## Phase 0 — Foundations

```text
1. POSITIONING AND NAVIGATION
   - Rename the product to "Risk Analysis Tool" in the app, README, Excel report and docs.
     Do not rename the GitHub repo or Streamlit app; list those steps for me.
   - Navigation: Overview | Market | Liquidity | Credit | Concentration & Factors |
     Event & Governance | Integrated Stress | Decisions | Trust. Existing tabs move under
     Market, unchanged. Unbuilt pages show "coming next". Shared sidebar inputs and cached
     data must stay consistent across pages (st.navigation or a sidebar radio; explain choice).

2. POSITIONS
   - Holdings can be entered as shares or as value. Store {ticker, quantity, price, value,
     weight, sector}. Keep weight entry working.

3. DATA LAYER
   - Prices: keep Open, High, Low, Close, Volume. For Indian stocks, fetch .NS and .BO volume
     and sum them where both exist.
   - fetch_fundamentals(ticker): annual income statement, balance sheet, cash flow (last 4-5
     years), shares outstanding, market cap, sector, industry; row names normalised through a
     mapping table; returns a "missing_fields" list instead of raising. CSV upload with a
     downloadable template overrides Yahoo field by field; the UI shows each figure's source.
   - disclosures.py: loaders, templates and validators for (a) promoter holding and
     pledged/encumbered % by quarter, (b) ASM/GSM surveillance lists with dates, (c) price bands,
     (d) F&O ban list, (e) credit rating actions, (f) auditor resignations/qualified opinions.
     Data in data/disclosures/ with a manifest (source URL, download date, coverage).

4. DATA-QUALITY SCORE per holding: spikes that reverse, stale prices, zero-volume days, gaps,
   short history, missing fundamentals. Feeds the Trust layer.

Tests: field mapping on saved yfinance-shaped frames (one Indian, one US, with missing rows);
disclosure parsers on sample official-format files; positions/weights conversion; data-quality
checks on synthetic data.
```

## Phase 1 — Trust layer (built once, used by every pillar)

```text
Phase 1 of RISK_TOOL_PLAN_V3.md. Create trust.py and ui/trust_page.py.

1. A reusable interface: any pillar metric can be wrapped as
   TrustedMetric(value, low, high, grade, reasons, sources, assumptions).
2. Market-risk uncertainty: 90% confidence intervals for each model's VaR and ES by stationary
   block bootstrap (Politis-Romano; block length by Politis-White with a fallback). For GARCH-t,
   also propagate parameter uncertainty from the asymptotic covariance.
3. Model risk: ES range across models that PASS the backtests; model-risk add-on = highest
   passing ES - recommended ES; lookback sensitivity (1y/2y/5y/max); a "ghost effect" detector
   for Historical VaR (big losses entering/leaving the window).
4. Grade A-D from explicit documented rules (CI width, model dispersion, backtest pass and test
   power, data quality, sample length, share of inputs that are assumptions). Show the reasons.
5. Every headline number in the app and the Excel report shows "value (90% range, grade)".
   Later pillars must use this interface for their headline numbers.

Tests: bootstrap CI coverage close to 90% on simulated i.i.d. normal data (fast version);
grade rules on constructed cases.
```

## Phase 2 — Liquidity risk pillar

```text
Phase 2 of RISK_TOOL_PLAN_V3.md. Create liquidity.py and ui/liquidity_page.py.

1. Trading capacity: 20/60-day average daily traded volume and value; days to liquidate each
   position at a participation rate (sidebar, default 20%); % of portfolio sellable in 1/5/10
   days; slowest holding.
2. SEBI/AMFI-style stress test (replicating the official disclosure method): days to liquidate
   25% and 50% of the portfolio pro-rata at 10% of 3-month average NSE+BSE volume, with the
   least liquid 20% of holdings excluded. Parameters editable; show with and without exclusion.
   VALIDATION: upload a fund's monthly portfolio disclosure and compare with the AMC's published
   figure for that month; report the gap and likely causes honestly.
3. Stressed liquidity: average volume in stress windows (stress_scenarios.csv) vs normal;
   days to liquidate under stressed volume.
4. Spread and cost: Corwin-Schultz high-low spread estimate (or user-entered spread);
   Bangia et al. exogenous spread cost; square-root-law impact using stressed volume.
   Liquidity-adjusted VaR waterfall: VaR → + spread → + impact → + circuit-lock.
5. Amihud illiquidity (rolling) and comparison across holdings.
6. Circuit-lock risk (India): price band from the uploaded file, or inferred from history
   (labelled "inferred"); count of past lower-circuit days and longest run; exit-freeze
   scenario of N consecutive lower circuits (default = longest historical run, at least 3)
   during which the position cannot be sold; circuit-lock loss per holding and portfolio.
7. "What this misses": e.g. block deals, the stock's free float, liability-side redemptions.

Tests: pro-rata calculation by hand (3 stocks); exclusion rule; Corwin-Schultz on simulated
prices with a known spread; circuit detection and compounding on synthetic series.
```

## Phase 3 — Credit risk pillar

```text
Phase 3 of RISK_TOOL_PLAN_V3.md. Create credit.py and ui/credit_page.py.

1. Merton structural model: solve E = V·N(d1) - D·e^(-rT)·N(d2) and σ_E·E = N(d1)·σ_V·V for V
   and σ_V (T = 1; default point D = short-term debt + 0.5 × long-term debt, KMV convention).
   Report DD = [ln(V/D) + (r - σ_V²/2)T] / (σ_V√T) and PD = N(-DD), labelled
   "model-implied risk-neutral PD, not an agency PD". Equity volatility input selectable
   (historical / EWMA / GARCH-t) with PD shown for each. Iterative KMV estimate as a
   cross-check. Rolling DD over time using point-in-time balance sheets.
2. Altman Z (listed manufacturers): 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5, zones > 2.99
   safe, 1.81-2.99 grey, < 1.81 distress. Altman Z'' (emerging markets / non-manufacturers,
   default for Indian firms): 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4 (X4 = book equity / total
   liabilities), zones > 2.60 safe, 1.10-2.60 grey, < 1.10 distress. Verify coefficients and
   cut-offs against Altman's papers and cite them. Never compute from partial inputs.
3. Credit ratios with 4-5 year trends and red flags (thresholds in one config): debt/equity,
   net debt/EBITDA, interest cover, current and quick ratio, CFO/EBITDA, negative CFO streaks.
4. Rating actions from the disclosure data: current rating, last action, direction.
5. Financial companies (banks, NBFCs, insurers): skip Merton/Altman/leverage ratios with an
   explanation; optional manual panel for GNPA, NNPA, CAR, CASA, NIM from annual reports.
6. Portfolio view: per-holding DD, PD, Z zone, red-flag count, rating; weighted PD;
   credit-implied expected loss = Σ PD_i × position_i (loss given default = 100% for equity
   holders — state this).
7. "What this misses": e.g. off-balance-sheet debt, group-company exposure, promoter-level
   leverage (links to Pillar 5).

Tests: Merton recovers known V and σ_V from generated E and σ_E; PD rises with debt and with
volatility; published or hand-worked Altman examples; point-in-time balance-sheet selection.
```

## Phase 4 — Concentration & factor risk pillar

```text
Phase 4 of RISK_TOOL_PLAN_V3.md. Create concentration.py and ui/concentration_page.py.

1. Factor data: daily Indian Fama-French + momentum factors from the IIM Ahmedabad data
   library, and US Fama-French 3 + momentum from the Kenneth French library, via
   scripts/refresh_factor_data.py into data/factors/ with source and end-date metadata. If a
   download fails, stop and tell me how to download manually. Never fabricate factor data.
2. Factor regression per holding (Newey-West standard errors): betas, alpha, R², factor vs
   specific variance; rolling 1-year betas. Fallback: single-index model on the benchmark.
3. Portfolio factor risk: weighted betas and an Euler decomposition of variance/parametric VaR
   into each factor + specific risk (contributions sum exactly to the total).
4. Concentration: HHI and effective number of holdings; sector weights and sector risk
   contributions; PCA share of the first component and effective number of independent bets.
5. Crisis correlation: correlations on the full sample, in stress windows and on the market's
   worst 10% days; portfolio ES under each; "diversification kept in a crisis" ratio.
6. "What this misses": e.g. common ownership/crowding, group-company links (e.g. holdings in
   the same business group).

Tests: simulated returns with known loadings recover betas and the variance split; Euler parts
sum to total; HHI by hand; identical holdings give one PCA component at 100%.
```

## Phase 5 — Event & governance risk pillar (India)

```text
Phase 5 of RISK_TOOL_PLAN_V3.md. Create events.py and ui/events_page.py.

1. Point-in-time early-warning panel per holding: promoter pledge % and 4-quarter change,
   ASM/GSM stage, F&O ban, recent rating downgrade, auditor resignation/qualification,
   Merton DD trend (from Pillar 3), circuit history (from Pillar 2), group-company flag
   (manual tag).
2. Event-risk tier (Low / Elevated / High) from explicit documented rules.
3. Pledge margin-call analysis: given the pledged quantity and an assumed loan cover ratio
   (editable assumption), the share-price fall at which lenders could invoke the pledge, and
   the selling that would follow as days of average volume.
4. Jump overlay: ES as a mixture of the fitted GARCH-t distribution and a jump with probability
   p and size J per tier (editable assumptions, defaults justified later by Phase 7). Report
   standard ES vs event-adjusted ES and which holdings drive the gap.
5. "What this misses": e.g. fraud not yet visible in any disclosure, regulatory action.

Tests: tier rules on constructed cases; margin-call trigger price by hand; mixture ES against a
large simulation; no data after the as-of date is used.
```

## Phase 6 — Integration and decisions

```text
Phase 6 of RISK_TOOL_PLAN_V3.md. Create integration.py, decisions.py and their UI pages.

1. INTEGRATED (LINKED) STRESS ENGINE: one scenario hits all pillars together.
   - For each stress scenario (historical windows and custom shocks), compute together:
     market loss (replay / beta proxy) → volume falls to its stressed level (Pillar 2: longer
     exit, higher impact cost, circuit-lock) → equity value and volatility change feed Merton
     (Pillar 3: new DD/PD) → pledge triggers checked at the stressed price (Pillar 5: forced
     selling adds to impact) → crisis correlation (Pillar 4).
   - Output one table per scenario: market loss, + liquidity cost, + credit deterioration,
     + event effects = total stressed loss, compared with the "siloed" sum from separate pages.
     Show the interaction effect explicitly; it is the tool's main message.
2. REVERSE STRESS TEST: the most plausible shock that produces a loss L.
   - Asset space: minimise xᵀΣ⁻¹x subject to wᵀx ≤ -L; closed form x* = -L·Σw/(wᵀΣw).
     Full-sample and crisis Σ; plausibility under the fitted multivariate Student-t.
   - Macro space: regress on Nifty 50, Bank Nifty, Nifty IT, USD/INR, Brent (plus an uploaded
     G-sec yield if available); report "the most plausible way to lose 15%: Nifty -11%,
     Bank Nifty -17%, INR -4%, crude +22%" and the nearest historical analogue.
   - Then run the linked engine on that scenario so the reverse stress includes liquidity,
     credit and event effects.
3. RISK-CHANGE EXPLANATION: compare two as-of dates or an uploaded JSON snapshot; Shapley
   attribution of the ES change to positions, volatility, correlation, data-window roll and
   model switch; waterfall chart.
4. BEST RISK-REDUCING TRADES: marginal ES per holding; top 3 trades by ES reduction, each with
   its effect on days-to-liquidate, weighted PD and event-risk exposure; ES-minimising Nifty
   futures hedge notional (measurement only).
5. LIMITS: editable JSON limits across pillars (ES %, max weight, max sector weight, days to
   liquidate 50%, max weighted PD, max High-tier holdings, minimum trust grade); utilisation
   and traffic lights (green < 80%, amber 80-100%, red > 100%).
6. OVERVIEW = CRO DASHBOARD: one row per pillar (headline, range, grade, status), integrated
   stress headline, limit status, top 3 risks and top 3 actions.
7. ONE-PAGE CRO MEMO: PDF (reportlab + matplotlib) and Markdown. Bottom line · Pillar summary
   · Integrated stress · What the numbers miss · Limits · Actions · Data sources and caveats.
   Rule-based text, no LLM calls.

Tests: closed-form reverse stress against a numerical optimiser; Shapley parts sum to the
total; linked engine with liquidity/credit/event effects switched off equals the plain market
stress; limits at boundaries.
```

## Phase 7 — Evidence: did it see it coming?

```text
Phase 7 of RISK_TOOL_PLAN_V3.md. Create case_studies/ and docs/case_studies.md.

1. Events: Yes Bank (2018-2020), DHFL (2019), Zee Entertainment (2019 pledge stress), Adani
   Enterprises (Jan-Feb 2023), plus 1-2 I approve. Control group: 10 large stable names.
2. Run the tool AS OF 12, 6, 3 and 1 month(s) before each collapse, point-in-time only. Record
   per pillar: what standard VaR/ES said and its grade; liquidity metrics; Merton DD and Altman
   zone; event flags and tier; the integrated stress loss; and the realised loss.
3. Control group: how often the same flags fired (false positives).
4. Give me a checklist of the exact official files to download for each case; one source link
   per row; no gap-filling.
5. Report hits, misses and false positives honestly in a table. Use the results to justify or
   revise the Phase 5 jump defaults.
6. README section "Did it see it coming?" with the table and one chart.
```

## Phase 8 (optional) — FX risk

```text
Phase 8 of RISK_TOOL_PLAN_V3.md. Base-currency selector (INR/USD); convert with daily FX
(e.g. USDINR=X): base return = (1 + local)(1 + FX) - 1; FX as a separate risk factor in the
Euler decomposition and the linked stress engine (e.g. INR -10%). Tests: a USD stock with zero
local volatility carries exactly the FX risk; Euler parts sum to the total.
```

## After all phases (manual steps for you)

1. Review, merge and push. Rename the repo (e.g. `Risk-Analysis-Tool`), check the Streamlit app still deploys, and update the live-demo link everywhere.
2. Screenshots: CRO dashboard, integrated stress table, liquidity waterfall, credit panel, case-study table.
3. A 2-minute interview story: siloed risk tools miss interactions → what you built → the integrated stress result for one case study → what the tool still cannot do.
