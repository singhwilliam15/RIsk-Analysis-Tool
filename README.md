# VaR Analysis Tool

[![tests](https://github.com/singhwilliam15/VaR-Analysis-Tool/actions/workflows/tests.yml/badge.svg)](https://github.com/singhwilliam15/VaR-Analysis-Tool/actions/workflows/tests.yml)
![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Live demo:** _coming soon_. The link will be added here once the app is deployed on Streamlit Community Cloud.

A market-risk dashboard for a single stock or a multi-stock portfolio on NSE, BSE or US markets. It estimates Value at Risk (VaR) and Expected Shortfall (ES) with eight models, from historical simulation to GARCH(1,1) with Student-t errors. It then **tests which model can be trusted**:
- every model is backtested out of sample with the Kupiec, Christoffersen and McNeil-Frey (ES) tests;
- models are ranked on tick loss, not p-values;
- portfolio risk is split across holdings with an exact Euler allocation;
- crisis scenarios are measured from real index data and replayed through the position's actual returns.

Results export to a formatted Excel report. Built in Python with Streamlit, and covered by 135 offline tests in CI.

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
| **Data quality** | Adjusted prices, exchange-timezone dates, a visible data source, and warnings for suspicious moves. Data are never altered silently. |
| **Excel report** | Dashboard, Portfolio Risk, Backtesting, Stress Testing and Raw Data sheets. |

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

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The 135 tests run on every push and pull request through GitHub Actions, on Python 3.11 and 3.12. They run **offline**: `conftest.py` blocks outbound connections, and market data comes from a deterministic synthetic generator or mocked Yahoo responses. Every calculation is checked against an independent reference: a closed form, a simulation, a hand-worked example or known true parameters. Highlights:
- **GARCH:** recovers the true parameters from simulated GARCH-t data.
- **Student-t ES:** matches 2 million simulated draws.
- **Portfolio:** components add up exactly on every basis, and match the normal Euler shares on 400,000 simulated days.
- **No look-ahead:** rolling forecasts never use future data.
- **Basel traffic light:** reproduces the regulatory table.
- **McNeil-Frey:** rejects an ES understated by 30%.
- **App smoke tests:** `test_app.py` drives the full Streamlit app in both modes, including the error paths.

## Project structure

```text
app.py                 Entry point: wires the UI sections together
ui/                    Streamlit UI: sidebar, data loading, calculations, overview, one module per tab, cache
var_calculator.py      VaR/ES models, rolling forecasts, backtests, statistics
garch.py               GARCH(1,1)-t fitting, filtering, simulation; FHS
portfolio.py           Portfolio construction, risk decomposition, incremental VaR, what-if
stress.py              Measured crisis scenarios, historical replay, downside beta, volatility shock
stress_scenarios.csv   Editable crisis windows (Nifty 50 and S&P 500)
data_fetcher.py        Yahoo Finance download with a direct-HTTP fallback; data-quality checks
excel_exporter.py      Formatted Excel report
docs/methodology.md    Formulas, tests, design choices and references
test_*.py, conftest.py Tests, synthetic market data and the network guard
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

## License

[MIT](LICENSE)
