# Risk Analysis Tool

[![tests](https://github.com/singhwilliam15/VaR-Analysis-Tool/actions/workflows/tests.yml/badge.svg)](https://github.com/singhwilliam15/VaR-Analysis-Tool/actions/workflows/tests.yml)
![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**▶ Live demo: [var-analysis-tool-jb48f2b7syny4j2vzeartr.streamlit.app](https://var-analysis-tool-jb48f2b7syny4j2vzeartr.streamlit.app/)**. Try any NSE, BSE or US ticker, or switch to Portfolio mode. The app may take about 30 seconds to wake up if nobody has used it recently.

A risk dashboard for a single stock or a multi-stock portfolio on NSE, BSE or US markets. It is being extended from market risk into a full risk analysis tool with five pillars: market, liquidity, credit, concentration & factor, and event & governance risk.

The pillars are linked through one integrated stress engine, because in a real crisis they hit together. Every headline number carries a 90% range and an A–D trust grade with its reasons. The plan, phase by phase, is in [RISK_TOOL_PLAN_V3.md](RISK_TOOL_PLAN_V3.md).

**Market risk** is complete. It estimates Value at Risk (VaR) and Expected Shortfall (ES) with eight models, from historical simulation to GARCH(1,1) with Student-t errors. It then **tests which model can be trusted**:
- every model is backtested out of sample with the Kupiec, Christoffersen and McNeil-Frey (ES) tests;
- models are ranked on tick loss, not p-values;
- portfolio risk is split across holdings with an exact Euler allocation;
- crisis scenarios are measured from real index data and replayed through the position's actual returns.

The **shared data layer** is in place for the coming pillars:
- positions entered as weights, shares or values;
- OHLCV prices, with NSE and BSE volume combined;
- annual fundamentals with a source on every figure;
- loaders for official Indian disclosures (pledges, ASM/GSM, price bands, F&O ban, ratings, auditor events);
- a data-quality score for every holding.

The **trust layer** shows how far each number can be relied on:
- a 90% range for every model's VaR and ES, from a block bootstrap or GARCH parameter draws;
- an A–D grade from written rules, with the reasons for every deduction;
- the model-risk add-on across the models that pass the backtests;
- ES on 1-year, 2-year, 5-year and full-history windows;
- a "ghost effect" detector for Historical VaR.

**Liquidity risk** asks how fast the positions could be sold, and at what cost:
- days to liquidate at a participation rate;
- a SEBI/AMFI-style stress test (days to sell 25% and 50% pro rata);
- volume in past crises;
- a Corwin-Schultz spread estimate, Bangia spread cost and square-root-law impact, combined in a liquidity-adjusted VaR waterfall;
- Amihud illiquidity;
- circuit-lock risk for Indian stocks (the loss if a stock locks at its lower circuit for several days).

**Credit risk** looks at how close each company is to default, from its share price and from its accounts:
- Merton distance to default and a model-implied (risk-neutral) PD under three equity-volatility inputs;
- an iterative KMV cross-check;
- distance to default month by month, using only the balance sheets public at each date;
- Altman Z and Z'', never computed from partial inputs;
- credit ratios with red flags;
- rating actions;
- a separate panel for banks and insurers;
- the portfolio's weighted PD and credit-implied expected loss.

Results export to a formatted Excel report. Built in Python with Streamlit, and covered by 332 offline tests in CI.

| Page | Status |
| --- | --- |
| Overview | Built: summary cards with 90% ranges and grades, positions, data quality, fundamentals and their sources, disclosure files. Becomes the CRO dashboard in Phase 6. |
| Market | Built: every model (with 90% range and grade), backtest, portfolio, stress and export tab. |
| Liquidity | Built: capacity, AMFI-style stress test, crisis volume, spread and impact cost, LVaR waterfall, Amihud, circuit-lock. |
| Credit | Built: Merton DD/PD (three volatility inputs, KMV cross-check, month-end history), Altman Z/Z'', ratios and red flags, ratings, a bank panel, weighted PD and expected loss. |
| Trust | Built: ranges, grades, model risk, lookback sensitivity, ghost effect. |
| Concentration & Factors · Event & Governance | Coming next (Phases 4–5) |
| Integrated Stress · Decisions | Coming next (Phase 6) |

### How far to trust the numbers (live data to 30 Sep 2026, 95% 1-day ES on ₹10 lakh, 2-year lookback)

| Position | Headline model | ES (90% range) | Grade | Main deductions |
| --- | --- | --- | --- | --- |
| Reliance | EWMA | ₹25,613 (₹24,068–₹27,486) | B | models disagree by 17%; 499 returns; λ is assumed |
| HDFC Bank | EWMA | ₹26,835 (₹25,106–₹28,711) | D | EWMA is recommended on the VaR tests but fails the ES test; only FHS passes both |
| Asian Paints | GARCH(1,1)-t | ₹30,088 (₹25,319–₹33,781) | B | range 28% wide; models disagree by 24% |
| Jaiprakash Power | GARCH(1,1)-t (fit failed, shows EWMA) | ₹42,071 (₹34,497–₹49,313) | C | models disagree by 137% (Student-t ₹93,296 vs FHS ₹35,773) |
| 5-stock portfolio | EWMA | ₹16,125 (₹15,165–₹17,114) | B | 499 returns; λ is assumed; HDFC Bank's data score |

### Credit (live data and Yahoo statements to FY Mar 2026, as of 1 Oct 2026)

| Company | Distance to default (range across volatility inputs) | Merton PD | Altman (primary) | Red flags |
| --- | --- | --- | --- | --- |
| Reliance | 11.40 (11.25–11.74) | below 0.001% | Z'' 2.18, **grey** (Z 1.90, grey) | none |
| TCS | 17.03 (14.48–18.82) | below 0.001% | Z'' 8.46, safe | none |
| Asian Paints | 20.97 (20.73–26.56) | below 0.001% | Z'' 6.71, safe | none |
| Britannia | 24.32 (22.71–28.69) | below 0.001% | Z'' 5.44, safe | none |
| Jaiprakash Power | 4.23 (4.23–6.69; GARCH fit failed) | 0.001% | Z'' 3.75, safe, **but Z 1.77, distress** | none |
| HDFC Bank | not modelled (bank) | – | not applicable | – |

- **Large caps are far from default on Merton.** Their default points are small relative to market value, so PDs are around 10⁻³⁰ and mean nothing on their own. The tool shows "below 0.001%" and grades their uncertainty on the distance-to-default scale instead.
- **Accounts and market can disagree.** Reliance's market-based distance to default is 11.4, but its accounts put it in Altman's grey zone. Its working capital is only 2.4% of total assets, and EBIT is 6.8% of a very large asset base. Jaiprakash Power is safe on Z'' but in distress on the 1968 manufacturers' Z. The page shows both.
- **Banks are not forced into these models.** HDFC Bank (25% of the default portfolio) gets a manual panel instead, so the portfolio PD covers 75% of the value and says so.

### Liquidity (live data to 1 Oct 2026, 20% participation)

- **₹10 lakh is liquid everywhere.** Every position is under 0.11% of a day's volume, so spread cost dominates impact. For Reliance, VaR ₹20,858 + spread ₹1,601 + impact ₹104 gives a liquidity-adjusted VaR of ₹22,563.
- **Circuit-lock risk dominates for a small-cap.** Jaiprakash Power (₹15 a share) has an inferred 5% band. Three lower circuits in a row (the 3-day floor; its longest past run was 1) would cost ₹1,42,625. That lifts its liquidity-adjusted VaR to ₹1,45,714 from a VaR of ₹33,548. The grade flags that both inputs are assumptions.
- **₹50 crore makes the pillar bite.** The 5-stock portfolio would need 0.09 days to sell 50% pro rata under the AMFI convention, and 0.21 days without the exclusion (Asian Paints binds, at 4.2% of a day's volume). Spread and impact add ₹27 lakh to VaR, giving a liquidity-adjusted VaR of ₹89.8 lakh (90% range ₹71.6–₹103.6 lakh).
- **Crises raise large-cap volume.** Median crisis-window volume was 1.08–1.32× the preceding four months for Reliance, HDFC Bank, TCS and Asian Paints, and 0.84× for Britannia. The tool never assumes a crisis makes selling easier, so it caps the factor at 1.

## Key findings

From live Yahoo Finance data, 5-year lookback, run on 1 Oct 2026:

- **Model choice moves the number by 79%.** Apple's 1-day 99% VaR on US$1m ranges from $34,574 (EWMA) to $61,838 (Cornish-Fisher) across the eight models.
- **GARCH(1,1)-t wins for Apple.** It is recommended at both 99% and 97.5%: the lowest out-of-sample tick loss among the models that pass every VaR test.
- **The ES backtest catches what VaR tests miss.** At 97.5%, EWMA passes all three VaR tests on Apple, but its ES is rejected (p < 0.001). For both stocks at both levels, the ES test rejects exactly the two normal-tailed models, Normal and EWMA.
- **√t overstates Apple's 10-day risk by 30%.** √t gives $152,071 against $116,860 from actual 10-day returns; a GARCH-t Monte Carlo gives $109,129.
- **A single beta misses real crisis behaviour.** Replayed through actual prices, Asian Paints fell 25% in the 2008 crash (Nifty −60%) but 18% after demonetisation (Nifty −7%).
- **Diversification cuts tail risk by 37%** in a five-stock NSE portfolio: Historical ES ₹31,000 → ₹19,496.
- **The data are checked too.** Yahoo's full Reliance history contains a +337% one-day spike that reverses the next day; the app flags it as a likely data error.

<!--
Screenshots (add the files to docs/images/, then remove this comment wrapper):
![Model comparison](docs/images/model-comparison.png)
![Backtesting](docs/images/backtesting.png)
![Portfolio risk](docs/images/portfolio-risk.png)
-->

## What it does

| Area | What you get |
| --- | --- |
| **Models** | Historical, Normal, Student-t (maximum likelihood), Cornish-Fisher (with a validity check), EWMA (RiskMetrics), Filtered Historical Simulation, GARCH(1,1)-t, and Monte Carlo on simulated GARCH-t paths. VaR at 90 / 95 / 97.5 / 99% and ES for each. |
| **Horizons** | 1–30 days. √t only where appropriate: the parametric models use `z·σ·√t − μ·t`, GARCH uses its variance term structure, Monte Carlo simulates full paths. An empirical overlapping-window check is shown next to √t. |
| **Backtesting** | Out-of-sample forecasts for every model, with the Kupiec, Christoffersen independence and conditional-coverage tests, the Basel traffic light and the McNeil-Frey ES test. The **recommended model** has the lowest tick loss among the models that pass. Verdicts show `LOW POWER` when there are too few test days. |
| **Portfolio** | Daily rebalancing or buy-and-hold. Risk split on a Historical ES, Historical VaR or Parametric VaR basis, with components that add up exactly to the total. Standalone and incremental risk, diversification benefit, correlation heatmap, and a what-if panel to change a weight or add a ticker. |
| **Stress testing** | 7 Indian and 7 US crises from an editable CSV, with drawdowns and recovery times measured from index data. Historical replay of the position, or a downside-beta proxy when it has no prices; a custom market move; a volatility shock. |
| **Positions** | Holdings entered as weights, share counts or money values, each stored as quantity, price, value, weight and sector. |
| **Data layer** | Open, high, low, close and volume, with NSE and BSE volume summed where Yahoo has both. Annual fundamentals mapped to one set of field names, which a CSV upload can override field by field; each figure shows its source. Loaders and validators for Indian disclosure files, which you download from NSE, BSE and the rating agencies, listed in a manifest with source and date. |
| **Data quality** | Adjusted prices, exchange-timezone dates, a visible data source, and warnings for suspicious moves. Each holding gets a 0–100 data-quality score, from checks for reversing spikes, stale prices, zero-volume days, gaps, short history and missing fundamentals. Data are never altered silently. |
| **Trust** | 90% ranges for every model's VaR and ES: a stationary block bootstrap (Politis-White block length) for the unconditional models, a residual bootstrap with today's volatility fixed for EWMA and FHS, and asymptotic parameter draws for GARCH-t and Monte Carlo. A–D grades come from written rules: range width, model dispersion, backtest, data quality, sample length, and the share of inputs that are assumptions. Also the model-risk add-on, lookback sensitivity, and the ghost effect in Historical VaR. |
| **Liquidity** | ADV, days to liquidate and share sellable in 1/5/10 days; the SEBI/AMFI-style stress test (with a fund-validation upload); crisis-window volume; Corwin-Schultz spread, Bangia spread cost, square-root impact and the liquidity-adjusted VaR waterfall; Amihud illiquidity; price bands (official or inferred), lower-circuit history and the exit-freeze loss. Every assumption (participation rate, k, Y, freeze length) is editable in the sidebar. |
| **Credit** | Merton distance to default and risk-neutral PD (historical, EWMA and GARCH-t equity volatility; KMV default point; iterative KMV cross-check; month-end history using only balance sheets public at each date). Altman Z and Z'' with zones. Six credit ratios over 4–5 years with red flags (thresholds in `config/credit_thresholds.json`), rating actions, a bank/NBFC panel, weighted PD and credit-implied expected loss. |
| **Excel report** | Dashboard (with 90% ranges and grades), Portfolio Risk, Backtesting, Stress Testing, Positions & Data, Trust, Liquidity, Credit, and Raw Data sheets. |

Every formula, test and design choice is in **[docs/methodology.md](docs/methodology.md)**, with references.

## Example results

All figures are live data on 1 Oct 2026, for a US$1m / ₹10,00,000 position with a 5-year lookback, unless stated.

### Apple (AAPL): 1,253 daily returns, 1,003 out-of-sample test days

| Model | 99% VaR | 97.5% VaR | 97.5% ES |
| --- | --- | --- | --- |
| Historical | $48,091 | $36,685 | $47,596 |
| Parametric (Normal) | $40,277 | $33,799 | $40,480 |
| Student-t | $47,331 | $34,340 | $50,967 |
| Cornish-Fisher | $61,838 | $38,342 | $66,732 |
| EWMA (RiskMetrics) | $34,574 | $29,129 | $34,744 |
| FHS (EWMA-filtered) | $40,318 | $31,473 | $44,587 |
| GARCH(1,1)-t | $39,847 | $29,770 | $42,091 |
| Monte Carlo (GARCH-t) | $41,355 | $30,547 | $43,350 |

| Model | 99%: breaches (10.0 expected) | 99%: VaR tests | 99%: tick loss (bp) | 97.5%: breaches (25.1 expected) | 97.5%: VaR tests | 97.5%: ES test (p) |
| --- | --- | --- | --- | --- | --- | --- |
| Historical | 13 | PASS | 6.438 | 26 | FAIL | PASS (0.064) |
| Parametric (Normal) | 15 | PASS | 6.563 | 29 | FAIL | **FAIL (0.002)** |
| Student-t | 11 | PASS | 6.556 | 29 | FAIL | PASS (0.453) |
| Cornish-Fisher | 11 | PASS | 7.122 | 24 | PASS | PASS (0.621) |
| EWMA (RiskMetrics) | 19 | **FAIL** | 6.525 | 31 | PASS | **FAIL (0.000)** |
| FHS (EWMA-filtered) | 15 | PASS | 6.647 | 26 | FAIL | PASS (0.092) |
| GARCH(1,1)-t | 13 | PASS | **6.236** | 27 | PASS | PASS (0.112) |
| Monte Carlo (GARCH-t) | 12 | PASS | 6.191 | 27 | PASS | PASS (0.195) |

- **GARCH(1,1)-t is recommended at both levels.** Monte Carlo's slightly lower tick loss is simulation noise around the same model, which is why Monte Carlo is never recommended.
- **EWMA reacts to volatility but has normal tails.** It fails the 99% VaR tests and both ES tests.
- **Cornish-Fisher's VaR is far too conservative.** It ties with Student-t for the breach count closest to target at 99% (11 against 10), yet it has the highest tick loss. A breach count alone would not show that.

### Asian Paints (ASIANPAINT.NS): 1,240 daily returns, 990 out-of-sample test days

| Model | 99% VaR | 97.5% VaR | 97.5% ES |
| --- | --- | --- | --- |
| Historical | ₹39,632 | ₹29,346 | ₹41,595 |
| Parametric (Normal) | ₹32,502 | ₹27,397 | ₹32,662 |
| Student-t | ₹39,838 | ₹28,520 | ₹43,465 |
| Cornish-Fisher | ₹46,072 | ₹32,642 | ₹48,194 |
| EWMA (RiskMetrics) | ₹27,683 | ₹23,323 | ₹27,819 |
| FHS (EWMA-filtered) | ₹38,665 | ₹27,378 | ₹38,708 |
| GARCH(1,1)-t | ₹34,050 | ₹24,969 | ₹36,515 |
| Monte Carlo (GARCH-t) | ₹34,887 | ₹26,141 | ₹36,038 |

| Model | 99%: breaches (9.9 expected) | 99%: VaR tests | 99%: tick loss (bp) | 97.5%: breaches (24.8 expected) | 97.5%: VaR tests | 97.5%: ES test (p) |
| --- | --- | --- | --- | --- | --- | --- |
| Historical | 11 | PASS | **4.869** | 27 | FAIL | PASS (0.396) |
| Parametric (Normal) | 19 | **FAIL** | 5.100 | 31 | FAIL | **FAIL (0.000)** |
| Student-t | 11 | PASS | 4.965 | 29 | FAIL | PASS (0.705) |
| Cornish-Fisher | 11 | PASS | 5.418 | 22 | PASS | PASS (0.188) |
| EWMA (RiskMetrics) | 22 | **FAIL** | 5.273 | 41 | FAIL | **FAIL (0.000)** |
| FHS (EWMA-filtered) | 13 | PASS | 4.891 | 29 | FAIL | PASS (0.463) |
| GARCH(1,1)-t | 12 | PASS | 4.931 | 34 | FAIL | PASS (0.633) |
| Monte Carlo (GARCH-t) | 10 | PASS | 4.937 | 30 | PASS | PASS (0.389) |

- **At 99%, Historical is recommended**, with the lowest tick loss among the passing models.
- **At 97.5%, Asian Paints' breaches cluster**, so most models fail the independence test. Cornish-Fisher is the only eligible model that passes, and it is recommended despite its higher tick loss: the rule puts validity before accuracy.
- **At 95%, all eight models fail**, GARCH and FHS included. They get the breach count right (Kupiec p ≥ 0.10) but the breaches arrive in clusters, and the app says "No model passes" rather than naming a winner.

### Multi-day: Apple, 99% 10-day VaR

| Method | 10-day VaR |
| --- | --- |
| Actual overlapping 10-day returns (check) | $116,860 |
| Historical × √10 | $152,071 |
| GARCH(1,1)-t, variance term structure | $120,551 |
| **Monte Carlo, simulated GARCH-t paths** | **$109,129** |

√t overstates Apple's 10-day risk by 30%. For Asian Paints it understates it: ₹1,25,327 from √t against ₹1,35,912 actual.

### Crisis replay: Asian Paints through each Nifty 50 crisis

| Scenario | Peak → trough | Nifty 50 fall | Nifty recovery | Asian Paints (actual) |
| --- | --- | --- | --- | --- |
| Global Financial Crisis | 08 Jan 2008 → 27 Oct 2008 | −59.9% | 496 days | −25.2% |
| US credit downgrade | 07 Jul 2011 → 20 Dec 2011 | −20.7% | 192 days | −17.0% |
| Taper tantrum | 17 May 2013 → 28 Aug 2013 | −14.6% | 34 days | −17.9% |
| Demonetisation | 08 Nov 2016 → 26 Dec 2016 | −7.4% | 22 days | −17.9% |
| IL&FS crisis | 28 Aug 2018 → 26 Oct 2018 | −14.6% | 114 days | −15.4% |
| COVID-19 crash | 14 Jan 2020 → 23 Mar 2020 | −38.4% | 158 days | −17.3% |
| Rate hikes and Russia-Ukraine | 18 Oct 2021 → 17 Jun 2022 | −17.2% | 108 days | −19.8% |

Asian Paints fell under half as much as the market in the GFC and COVID, but more than twice as much after demonetisation, a consumption shock. A beta-scaled shock (0.87 × −7.4% = −6.4%) would have badly understated that loss.

### Portfolio: five NSE stocks, 95% 1-day, 2-year lookback, rebalanced daily

| Holding | Weight | Standalone ES | Component ES | Share of ES | Incremental ES |
| --- | --- | --- | --- | --- | --- |
| Reliance | 30% | ₹8,438 | ₹5,856 | 30.0% | ₹5,184 |
| HDFC Bank | 25% | ₹7,028 | ₹5,103 | 26.2% | ₹4,052 |
| TCS | 20% | ₹7,563 | ₹3,592 | 18.4% | ₹2,508 |
| Asian Paints | 15% | ₹4,912 | ₹3,002 | 15.4% | ₹2,368 |
| Britannia | 10% | ₹3,061 | ₹1,943 | 10.0% | ₹1,427 |
| **Total** | 100% | **₹31,000** | **₹19,496** | 100% | |

- **Diversification cuts ES by 37%.** The average pairwise correlation is 0.26.
- **VaR and ES can disagree on the same trade.** Cutting Reliance to 10%, with the others scaled up, lowers Historical VaR by 4.9% but raises Historical ES by 1.3%.

## Run it locally

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/singhwilliam15/VaR-Analysis-Tool.git
cd VaR-Analysis-Tool
pip install -r requirements.txt
streamlit run app.py
```

- **On Windows**, you can double-click `run_app.bat` instead.
- **In GitHub Codespaces**, the dev container installs everything and starts the app on port 8501.
- **Tickers** use Yahoo Finance symbols: `.NS` for NSE (`RELIANCE.NS`), `.BO` for BSE, and no suffix for US stocks (`AAPL`).
- **Pinned versions:** `requirements.txt` pins the exact versions the tests pass with. On Python 3.11, numpy and scipy use slightly older pins, because their newest releases no longer support it.
- **Caching:** heavy results (model fits, rolling forecasts, backtests, crisis replays) are cached. On a 2,600-day history the first run takes about 10 s, and changing the position size or confidence level takes 1–2 s.

## Deploy (Streamlit Community Cloud)

The entry point is `app.py`, the requirements are pinned and no secrets are needed.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns this repository.
2. Click **Create app**, then choose **Deploy a public app from GitHub**.
3. Pick `singhwilliam15/VaR-Analysis-Tool`, branch `main`, main file `app.py`.
4. Under **Advanced settings**, choose **Python 3.12**.
5. Click **Deploy**. The first build takes a few minutes.
6. Paste the app's URL into the **Live demo** line at the top of this README.

This project is deployed this way at the live-demo link above.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The 332 tests run on every push and pull request through GitHub Actions, on Python 3.11 and 3.12. They run **offline**: `conftest.py` blocks outbound connections, and market data comes from a deterministic synthetic generator or mocked Yahoo responses. Every calculation is checked against an independent reference: a closed form, a simulation, a hand-worked example or known true parameters. Highlights:
- **GARCH:** recovers the true parameters from simulated GARCH-t data.
- **Student-t ES:** matches 2 million simulated draws.
- **Portfolio:** components add up exactly on every basis, and match the normal Euler shares on 400,000 simulated days.
- **No look-ahead:** rolling forecasts never use future data.
- **Basel traffic light:** reproduces the regulatory table.
- **McNeil-Frey:** rejects an ES understated by 30%.
- **Fundamentals:** the field mapping is checked against real RELIANCE.NS and AAPL statements saved from yfinance, including the years and rows Yahoo leaves out.
- **Disclosures:** the parsers read sample files laid out like NSE's downloads, reject bad rows with their row number, and apply the point-in-time dates.
- **Trust:** the bootstrap ranges contain the true VaR and ES about 90% of the time on simulated normal returns. GARCH parameter draws reproduce the fitted covariance. Every grade rule is checked at its thresholds. The ghost detector finds a planted crash as it enters and leaves the window.
- **Liquidity:**
  - the AMFI-style test is checked by hand on three stocks, including a holding that is partly excluded;
  - Corwin-Schultz recovers a known 1% spread from simulated bid-ask trades;
  - spread cost and impact are checked by hand;
  - the crisis-volume ratio is checked on planted volume drops;
  - circuit detection, band inference and the compounded freeze loss are checked on synthetic series.
- **Credit:**
  - Merton recovers known asset values and volatilities from the equity they imply;
  - PD rises with debt and with volatility;
  - iterative KMV agrees with the direct solution on a simulated asset path;
  - Altman Z and Z'' are checked by hand, and at every zone boundary;
  - a balance sheet is never used before its public date;
  - red flags are checked on a constructed weak company.
- **Positions and data quality:** share, value and weight entry give the same portfolio, checked by hand. Each data-quality check is tested on synthetic histories with planted defects.
- **App smoke tests:** `test_app.py` drives the full Streamlit app in both modes and on every page, including the error paths.

## Project structure

```text
app.py                 Entry point: shared sidebar, data and calculations, then the selected page
ui/                    Streamlit UI: navigation, sidebar, data loading, calculations, pages, one module per tab, cache
var_calculator.py      VaR/ES models, rolling forecasts, backtests, statistics
garch.py               GARCH(1,1)-t fitting, filtering, simulation; FHS
portfolio.py           Positions, portfolio construction, risk decomposition, incremental VaR, what-if
stress.py              Measured crisis scenarios, historical replay, downside beta, volatility shock
stress_scenarios.csv   Editable crisis windows (Nifty 50 and S&P 500)
data_fetcher.py        Yahoo Finance OHLCV download with a direct-HTTP fallback; NSE + BSE volume; spike flags
fundamentals.py        Annual statements and profile, field mapping, CSV overrides, point-in-time dates
disclosures.py         Loaders, templates and validators for Indian disclosure files
data_quality.py        Data-quality score per holding
trust.py               TrustedMetric, 90% ranges, model risk, lookback sensitivity, ghost effect, A-D grades
liquidity.py           Capacity, AMFI-style stress test, crisis volume, spread/impact, LVaR, Amihud, circuit lock
credit.py              Merton/KMV, Altman Z and Z'', credit ratios and red flags, ratings, point-in-time statements
config/                Credit thresholds and sector lists (editable)
data/disclosures/      Your downloaded disclosure files, manifest.json, templates and download steps
excel_exporter.py      Formatted Excel report
docs/methodology.md    Formulas, tests, design choices and references
RISK_TOOL_PLAN_V3.md   The roadmap, phase by phase
test_*.py, conftest.py Tests, synthetic market data and the network guard
test_data/             Saved yfinance statements used as test fixtures
.github/workflows/     CI: pytest on Python 3.11 and 3.12
```

## Limitations

**Already addressed** by a structured code review, worked through in five phases:
- **Data:** the fallback source now uses adjusted prices, and dates use the exchange's timezone.
- **Multi-day scaling:** the mean now scales with t, not √t.
- **Model selection:** models are ranked by tick loss, not p-values, and verdicts show `LOW POWER` when there are too few test days.
- **Models:** GARCH-t and FHS were added, and Monte Carlo is no longer a copy of the Normal model.
- **ES:** an Expected Shortfall backtest and the 97.5% level were added.
- **Portfolio:** the risk decomposition is consistent with the headline figures.
- **Stress tests:** measured from data instead of hard-coded.
- **Engineering:** CI, pinned dependencies and a modular app.

**What remains:**
- **Portfolios** are long-only and in one currency (no FX conversion or short positions).
- **Portfolio crisis replay** needs every holding to have prices for the crisis. One recently listed holding (for example LICI.NS, listed May 2022) switches the whole portfolio to the β-proxy for older crises.
- **Risk decomposition** covers Historical ES, Historical VaR and Parametric VaR only. Under buy-and-hold it applies today's drifted weights, so its total can differ from the headline figure.
- **Correlations and betas** are estimated on the lookback window. In crises correlations usually rise, so the diversification benefit shrinks when it is needed most.
- **√t scaling** (Historical, FHS) assumes independent returns. Compare it with the overlapping check, or use Monte Carlo.
- **GARCH(1,1)-t** has a constant mean and a symmetric response to shocks (no GJR/EGARCH leverage term). Its multi-day formula overstates the t-day tail; Monte Carlo is the better multi-day estimate.
- **Rolling backtests** refit Student-t and GARCH every 20 days, not daily. The first full-history run of a long-listed stock (for example AAPL `max`, about 11,000 days) takes about a minute.
- **The Acerbi-Székely (2014) ES test** is not implemented.
- **Stress scenarios** are historical. β-proxy rows assume today's crisis sensitivity held in past crises.
- **The risk-free rate** is a user-set assumption, not a live rate.
- **BSE volume** is often missing on Yahoo for large caps. When it is, NSE volume is used alone, and the Overview page says so for each holding.
- **Holiday rows.** Yahoo fills some exchange holidays with the previous close and zero volume. They add zero returns to the sample, which slightly lowers volatility. They are counted in the data-quality check but not removed.
- **Trust ranges** cover estimation error only, not a change of regime. GARCH and Monte Carlo ranges cover parameter uncertainty only. Bootstrap ranges for Historical ES are too narrow at small tail sizes (84% coverage instead of 90% in a 500-day, 95% simulation). The grade thresholds are assumptions.
- **Liquidity:**
  - the Corwin-Schultz spread still overstates spreads for the most liquid stocks (Reliance 0.06% against a quoted spread of a few hundredths of a percent);
  - impact uses an assumed constant;
  - inferred price bands are guesses until you load NSE's file;
  - the AMFI test is our reading of the convention, not an official calculation;
  - block deals, free float and redemptions are not modelled.
- **Credit:**
  - Merton PD is risk-neutral and assumes a single debt maturity;
  - Yahoo gives annual statements only, so quarterly deterioration is missed;
  - Altman's zones were fitted on US firms;
  - banks are not modelled beyond the manual panel;
  - off-balance-sheet, group and promoter-level debt are not captured.
- **Disclosure layouts.** The parsers for official NSE files (except `fo_secban.csv`) match headers through an alias table that has not yet been checked against real downloads.

## License

[MIT](LICENSE)
