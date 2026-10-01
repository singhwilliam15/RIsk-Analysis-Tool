# VaR Analysis Tool

A Streamlit dashboard that measures the market risk of a single stock or a multi-stock portfolio and tests which risk model is reliable. It pulls daily prices from Yahoo Finance and estimates Value at Risk (VaR) and Expected Shortfall (ES) with eight models, from plain historical simulation up to GARCH(1,1)-t. Each model is backtested out of sample: VaR with the Kupiec and Christoffersen tests, ES with the McNeil-Frey test. Crisis scenarios are measured from real index data and replayed through the position's actual returns, and everything exports to a formatted Excel report.

It started as an Excel VaR workbook. This project rebuilds it in Python, so any NSE, BSE or US ticker can be analysed in seconds, and adds the model-validation layer a spreadsheet makes hard.

## Models

| Model | How it estimates the loss quantile | Captures fat tails? | Reacts to recent volatility? |
| --- | --- | --- | --- |
| **Historical** | Empirical percentile of past returns | Yes, if they are in the sample | No |
| **Parametric (Normal)** | `z·σ − μ` | No | No |
| **Student-t** | t-distribution fitted by maximum likelihood (`scipy.stats.t.fit`); method of moments as fallback | Yes | No |
| **Cornish-Fisher** | Normal quantile adjusted for skewness and kurtosis; flagged ⚠ outside its valid region | Yes | No |
| **EWMA (RiskMetrics)** | Normal quantile on an exponentially weighted volatility forecast, λ = 0.94 | No | Yes |
| **FHS (EWMA-filtered)** | Filtered Historical Simulation: empirical quantile of returns ÷ their EWMA volatility, rescaled by tomorrow's volatility | Yes | Yes |
| **GARCH(1,1)-t** | `σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁` with unit-variance Student-t errors, fitted with `arch` on returns × 100 | Yes | Yes |
| **Monte Carlo** | 1,000–10,000 simulated paths of the fitted GARCH(1,1)-t process | Yes | Yes |

Each model reports VaR at 90%, 95%, 97.5% and 99%, plus Expected Shortfall. ES uses closed forms for the Normal, Student-t, EWMA and GARCH-t models, the empirical tail average for Historical, FHS and Monte Carlo, and numerical integration for Cornish-Fisher. **97.5%** is the Basel FRTB Expected Shortfall level.

- **GARCH(1,1)-t** combines the two features the earlier results called for: volatility clustering, which EWMA has, and fat tails, which Student-t has. If the fit does not converge or is non-stationary (α + β ≥ 1), the app shows EWMA in its place and says so.
- **Cornish-Fisher** is only a valid quantile function when the expansion is increasing in z. The app checks the Maillard (2012) condition on the sample's skewness S and excess kurtosis K: with `a = K/8 − S²/6`, `b = S/3` and `c = 1 − K/8 + 5S²/36`, it needs `a ≥ 0` and `b² − 4ac ≤ 0`. Outside that region the model is marked ⚠ "treat with caution". This happens, for example, for AAPL's full history (excess kurtosis 18.7) and ADANIENT.NS over 5 years (20.8).
- **Monte Carlo** uses a seeded local random generator. The app warns when fewer than 50 simulated draws land in the tail, since ES is then very noisy (for example, 1,000 simulations at 99% leave only 10).

**Multi-day horizons (1–30 days)** follow one rule per model type:

| Model type | t-day VaR / ES |
| --- | --- |
| Parametric (Normal, Student-t, Cornish-Fisher, EWMA) | `z·σ·√t − μ·t`: volatility grows with √t, the mean with t |
| Historical, FHS | 1-day quantile × √t |
| GARCH(1,1)-t | GARCH variance term structure: `σ²(t days) = Σ E[σ²(T+h)]`, with `E[σ²(T+h)] = ω + (α+β)·E[σ²(T+h−1)]` |
| Monte Carlo | Simulated t-day GARCH-t paths, with volatility updating along each path (no √t) |

For Historical, the app also shows the **empirical t-day VaR from overlapping compounded t-day returns**, so the √t assumption can be checked against the data. Overlapping windows share days, so treat that figure as a sense check, not a precise estimate.

## Data and performance statistics

- **Prices:** split- and dividend-adjusted closes from yfinance. If yfinance fails, the app calls Yahoo's chart API directly and uses its adjusted closes (raw closes only if none are returned). The source and price basis are shown under the page title. Dates are converted with the exchange's timezone, so they don't depend on where the app is hosted. If company metadata can't be fetched, the price history is kept and the ticker is used as the name.
- **CAGR** is computed from the compounded return path. The arithmetic mean × 252 is reported separately as `ann_mean_return`.
- **Data-quality check:** any daily move larger than 25% is listed in a warning. A move that the next day reverses (the price ends up back near where it started) is marked as a likely data error. The data are never altered. Example: Yahoo's full RELIANCE.NS history contains a +337% day on 28 Jul 2005 followed by −77% the next day, which pushes the sample's excess kurtosis to about 3,600; choose a shorter lookback to exclude it.
- **Sharpe and Sortino** use an editable risk-free rate: 6.5% for INR and 4.0% for USD by default. These are **assumptions, not live rates**. Sortino divides by the downside deviation, `√mean(min(r − r_f, 0)²)`, taken over all days.

## Portfolio mode

Enter any number of tickers and weights (one currency), and choose **rebalanced daily** (weights reset to the targets each day) or **buy-and-hold** (shares bought once, weights drift with prices). The portfolio uses the dates on which every holding traded. The app names the holding that limits this shared history and how many days it cuts. All eight models, the backtests and the stress tests then run on the portfolio's return series.

Portfolios are **long-only**. Short positions would need borrow costs, margin and a gross/net exposure definition that the tool does not model, and the app says so next to the holdings table.

The **Portfolio Risk** tab splits risk across holdings on a basis you choose. On every basis, the components add up exactly to the total shown:

| Basis | Component of holding i | Adds up to |
| --- | --- | --- |
| **Historical ES** (default) | `wᵢ·E[−rᵢ | portfolio return ≤ its α-quantile]`: the exact tail-conditional Euler estimator | Portfolio historical ES |
| **Historical VaR** | `wᵢ·E[−rᵢ | portfolio return ≈ its α-quantile]`, averaged over the days ranked nearest the quantile (±0.25% of the sample, at least ±2 days), rescaled to the total | Portfolio historical VaR |
| **Parametric VaR** | `wᵢ·(z·(Σw)ᵢ/σₚ·√t − μᵢ·t)`: the normal Euler allocation | Portfolio parametric VaR |

The table also shows, on the same basis:
- **standalone risk**: each holding on its own;
- **incremental risk**: the portfolio with the holding minus without it, other positions unchanged;
- **risk / weight**: above 1× means the holding adds more risk than capital;
- the **diversification benefit**: sum of standalone − total;
- a correlation heatmap.

A **what-if panel** sets one holding to a new weight, or adds a new ticker, and shows the new Historical VaR, Historical ES and Parametric VaR against today's. The other holdings keep their relative sizes.

## Backtesting

Every model is tested **out of sample**: each day's VaR and ES are forecast from past data only and then compared with that day's actual return. All models are scored on the same days.

- **Historical, Normal, Cornish-Fisher and FHS** re-estimate daily on the previous 250 days.
- **Student-t** is refitted by maximum likelihood on the previous 250 days every 20 trading days.
- **GARCH(1,1)-t and Monte Carlo** are refitted every 20 trading days on all earlier data (at most 1,000 days, since GARCH needs more than 250 to estimate well), and volatility is filtered daily with the latest parameters. A fit that does not converge falls back to EWMA for that 20-day block. The app reports how many fits fell back: 0 of 51 for AAPL over 5 years, 27 of 289 for ASIANPAINT.NS over its full history.

| Test | Question it answers | Distribution |
| --- | --- | --- |
| **Kupiec POF** | Is the number of breaches right? | χ²(1) |
| **Christoffersen independence** | Do breaches cluster on consecutive days? | χ²(1) |
| **Conditional coverage** | Both together | χ²(2) |
| **Basel traffic light** | Regulatory zone from the binomial distribution of breaches | Green < 95% ≤ Yellow < 99.99% ≤ Red |
| **McNeil-Frey ES test** | On breach days, is the loss beyond VaR as large as the model's ES said? | Bootstrap, one-sided |

A model passes when all three p-values are at least 0.05. The traffic-light rule reproduces the regulatory 0–4 / 5–9 / 10+ zones at 99% over 250 days, and it is applied correctly at any confidence level and sample length.

**Choosing a model.** A higher p-value is not evidence of a better model, so the tool does not rank on p-values. It ranks on the **tick (quantile) loss** of each model's out-of-sample VaR:

`L = mean[(α − 1{r < −VaR})·(r + VaR)]`

The loss is lowest, on average, for the true quantile. The **recommended model** is the one with the lowest tick loss among the models that pass all three tests. If none pass, the app says so and shows the lowest-loss model with a warning. **Monte Carlo is scored but never recommended**: at a 1-day horizon it is the GARCH-t model plus simulation noise, so any edge over GARCH-t is luck. Its value is in multi-day paths.

**Expected Shortfall backtest.** Basel FRTB sets capital on 97.5% ES, so ES needs its own test. The McNeil-Frey (2000) test takes each breach day's residual `(loss − ES)/σ`, which should average zero if ES is right. A positive mean means losses beyond VaR are bigger than the model's ES. The p-value is one-sided and comes from bootstrapping the t-statistic of the centred residuals (10,000 resamples). It needs at least 5 breaches. The ES result is shown separately and does not affect the VaR verdict.

**Statistical power.** The estimation window is always 250 days. A verdict needs at least **250 test days at 99%** (2.5 expected breaches) and **100 at 90–95%**. With fewer, the verdict shows as `LOW POWER` and the app suggests a longer lookback. A 1-year lookback leaves almost no test days, so use 2y (95%) or 5y / max (99%).

## Stress testing

Nothing in the stress tests is hard-coded. The crisis windows live in [`stress_scenarios.csv`](stress_scenarios.csv): seven for the Nifty 50 (GFC 2008, 2011 US downgrade, 2013 taper tantrum, 2016 demonetisation, 2018 IL&FS, 2020 COVID, 2021–22 rate hikes and Russia–Ukraine) and seven US equivalents for the S&P 500.

- **Measured, not assumed.** Each scenario's market fall is the largest peak-to-trough drawdown inside its window, computed from downloaded index prices. **Market recovery** is the number of trading days from the trough until the index regained its peak. Windows the index history does not cover are flagged and left out. Each window was checked against the data. For example, the Nifty's GFC fall is 6,288 on 8 Jan 2008 to 2,524 on 27 Oct 2008 (−59.9%), and demonetisation is measured from the 8 Nov 2016 announcement close.
- **Historical replay first.** If the stock or portfolio has prices for the period, its result is its **actual** compounded return between the market's peak and trough dates, using the full price history regardless of the lookback window.
- **β-proxy only when there is no data.** For a stock listed after the crisis, the market fall is multiplied by the **downside beta** and labelled β-proxy. Downside beta is the stock's sensitivity on the market's worst 10% of days: `Σ r_s·r_m / Σ r_m²` on those days. The app shows normal and downside beta side by side.
- **Custom market move** from −60% to +40%: falls use the downside beta, rises the normal beta.
- **Volatility shock:** VaR and ES re-run with returns scaled around their mean, `r′ = μ + k·(r − μ)`, for k = 1, 2, 3.
- **Worst actual losses in the sample** over 1, 5, 10 and 21 days, for comparison against VaR.

## Example results

Live data, run on 1 Oct 2026.

**Apple (AAPL): 99% VaR, 5-year lookback, 1,003 out-of-sample test days (10 breaches expected)**

| Model | Breaches | Kupiec p | Indep. p | Cond. cov. p | VaR verdict | Tick loss (bp) | ES test (p) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Historical | 13 | 0.367 | 0.156 | 0.244 | PASS | 6.438 | PASS (0.105) |
| Parametric (Normal) | 15 | 0.142 | 0.218 | 0.159 | PASS | 6.563 | **FAIL (0.001)** |
| Student-t | 11 | 0.762 | 0.106 | 0.258 | PASS | 6.556 | PASS (0.394) |
| Cornish-Fisher | 11 | 0.762 | 0.106 | 0.258 | PASS | 7.122 | PASS (0.444) |
| EWMA (RiskMetrics) | 19 | 0.011 | 0.049 | 0.006 | **FAIL** | 6.525 | **FAIL (0.000)** |
| FHS (EWMA-filtered) | 15 | 0.142 | 0.500 | 0.271 | PASS | 6.647 | PASS (0.388) |
| GARCH(1,1)-t | 13 | 0.367 | 0.156 | 0.244 | PASS | **6.236** | PASS (0.113) |
| Monte Carlo (GARCH-t) | 12 | 0.544 | 0.130 | 0.264 | PASS | 6.191 | PASS (0.095) |

- **GARCH(1,1)-t is recommended**, with the lowest tick loss among eligible passing models. Monte Carlo's slightly lower loss is simulation noise; see above.
- **The ES test adds information the VaR tests miss.** The Normal model passes every VaR test, yet its ES is rejected (p = 0.001): when Apple breaks the 99% VaR, the losses are much bigger than a normal tail implies.
- **EWMA fails both tests.** It reacts to volatility but keeps normal tails.
- **FHS fixes EWMA's tails.** It uses the same volatility filter with an empirical tail, and passes everything.

**Asian Paints (ASIANPAINT.NS): 97.5% VaR (FRTB level), 5-year lookback, 990 test days (24.8 breaches expected)**

| Model | Breaches | Kupiec p | Indep. p | Cond. cov. p | VaR verdict | Tick loss (bp) | ES test (p) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Historical | 27 | 0.652 | 0.001 | 0.002 | FAIL | 9.632 | PASS (0.396) |
| Parametric (Normal) | 31 | 0.221 | 0.000 | 0.001 | FAIL | 9.575 | **FAIL (0.000)** |
| Student-t | 29 | 0.399 | 0.000 | 0.000 | FAIL | 9.626 | PASS (0.705) |
| Cornish-Fisher | 22 | 0.568 | 0.093 | 0.207 | PASS | **10.131** | PASS (0.188) |
| EWMA (RiskMetrics) | 41 | 0.002 | 0.001 | 0.000 | FAIL | 9.781 | **FAIL (0.000)** |
| FHS (EWMA-filtered) | 29 | 0.399 | 0.009 | 0.023 | FAIL | 9.764 | PASS (0.463) |
| GARCH(1,1)-t | 34 | 0.074 | 0.030 | 0.020 | FAIL | 9.457 | PASS (0.633) |
| Monte Carlo (GARCH-t) | 30 | 0.301 | 0.070 | 0.113 | PASS | 9.381 | PASS (0.389) |

Breaches cluster for Asian Paints, so most models fail the independence test. Cornish-Fisher passes and is recommended, even though its tick loss is the highest: it is the only eligible model whose breaches are statistically acceptable. Its very conservative VaR spaces breaches out. The rule deliberately puts validity before accuracy. Again the ES test rejects only the normal-tailed models.

**Asian Paints (ASIANPAINT.NS): 95% VaR, 5-year lookback, 990 test days.** Every model gets the breach **count** right (Kupiec p ≥ 0.10). But **all eight**, including GARCH and FHS, fail the **independence** test (p ≤ 0.025), with breaches arriving in clusters of 6–13 back-to-back pairs. The app reports "No model passes" and shows Parametric (Normal), the lowest tick loss, with a warning. A breach-count test alone would have passed all of them.

**Multi-day, 99% 10-day VaR, US$1m in AAPL (5y):**

| Method | 10-day VaR |
| --- | --- |
| Actual overlapping 10-day returns (Historical check) | $116,868 |
| Historical × √10 | $152,076 |
| GARCH(1,1)-t, variance term structure | $122,147 |
| **Monte Carlo, simulated GARCH-t paths** | **$106,051** |

√t overstates Apple's 10-day risk by 30%. The GARCH-based figures are much closer to what actually happened. Monte Carlo is lower than the GARCH formula because fat daily tails partly average out over 10 days: the simulation captures this, while applying a 1-day t-quantile to 10-day volatility does not. For ASIANPAINT.NS over 5 years, √t goes the other way: ₹1,25,327 against ₹1,35,912 actual.

**Asian Paints (ASIANPAINT.NS): ₹10,00,000 position, 1-day horizon, 2-year lookback**

| Model | 95% VaR | 97.5% VaR | 99% VaR | 97.5% ES |
| --- | --- | --- | --- | --- |
| Historical | ₹24,420 | ₹29,219 | ₹34,732 | ₹38,485 |
| Parametric (Normal) | ₹23,847 | ₹28,324 | ₹33,531 | ₹33,693 |
| Student-t | ₹22,582 | ₹29,981 | ₹41,824 | ₹45,767 |
| Cornish-Fisher | ₹23,271 | ₹31,999 | ₹44,991 | ₹47,129 |
| EWMA (RiskMetrics) | ₹19,573 | ₹23,323 | ₹27,683 | ₹27,819 |
| FHS (EWMA-filtered) | ₹22,201 | ₹25,763 | ₹31,112 | ₹32,996 |
| GARCH(1,1)-t | ₹19,593 | ₹25,473 | ₹34,467 | ₹36,922 |
| Monte Carlo (GARCH-t, 5,000 sims) | ₹20,264 | ₹26,690 | ₹35,291 | ₹36,451 |

Excess kurtosis is 3.19 and the Jarque-Bera test rejects normality (p < 0.0001). This is why the fat-tailed models give the highest 99% VaR; the Student-t maximum-likelihood fit gives ν = 3.2. The volatility models (EWMA, FHS, GARCH) sit lowest at 95%, because volatility today is below its 2-year average. Its beta to the Nifty 50 is 0.87, with a downside beta of 0.99.

**Crisis replay, ASIANPAINT.NS (actual returns over each Nifty 50 peak-to-trough)**

| Scenario | Peak → trough | Nifty 50 fall | Nifty recovery | Asian Paints |
| --- | --- | --- | --- | --- |
| Global Financial Crisis | 08 Jan 2008 → 27 Oct 2008 | −59.9% | 496 days | −25.2% |
| US credit downgrade | 07 Jul 2011 → 20 Dec 2011 | −20.7% | 192 days | −17.0% |
| Taper tantrum | 17 May 2013 → 28 Aug 2013 | −14.6% | 34 days | −17.9% |
| Demonetisation | 08 Nov 2016 → 26 Dec 2016 | −7.4% | 22 days | −17.9% |
| IL&FS crisis | 28 Aug 2018 → 26 Oct 2018 | −14.6% | 114 days | −15.4% |
| COVID-19 crash | 14 Jan 2020 → 23 Mar 2020 | −38.4% | 158 days | −17.3% |
| Rate hikes and Russia-Ukraine | 18 Oct 2021 → 17 Jun 2022 | −17.2% | 108 days | −19.8% |

Replay shows what a single beta cannot. Asian Paints fell well under half as much as the market in the GFC and COVID, but more than twice as much in demonetisation, a consumption shock that hit consumer stocks directly. A beta-scaled shock (0.87 × −7.4% = −6.4%) would have badly understated that loss.

**Five-stock NSE portfolio: ₹10,00,000, 95% 1-day, 2-year lookback, rebalanced daily**

| Holding | Weight | Standalone ES | Component ES | Share of ES | Risk / weight | Incremental ES |
| --- | --- | --- | --- | --- | --- | --- |
| Reliance | 30% | ₹8,438 | ₹5,856 | 30.0% | 1.00× | ₹5,184 |
| HDFC Bank | 25% | ₹7,028 | ₹5,103 | 26.2% | 1.05× | ₹4,052 |
| TCS | 20% | ₹7,563 | ₹3,592 | 18.4% | 0.92× | ₹2,508 |
| Asian Paints | 15% | ₹4,912 | ₹3,002 | 15.4% | 1.03× | ₹2,368 |
| Britannia | 10% | ₹3,061 | ₹1,943 | 10.0% | 1.00× | ₹1,427 |
| **Total** | 100% | **₹31,000** | **₹19,496** | 100% | | |

- Diversification cuts Historical ES from ₹31,000 (positions added up separately) to **₹19,496**, a 37% reduction. The average correlation is 0.26.
- On the ES basis, risk shares sit close to capital weights. On the parametric basis, Reliance carries 33.8% of the risk from a 30% weight.
- The Historical VaR basis gives Asian Paints only 9.7%, against 15.4% on the ES basis. VaR-based allocations depend on the handful of days nearest the quantile, which is why Historical ES is the default.
- **What-if:** cutting Reliance to 10%, with the other four scaled up, lowers Historical VaR by 4.9% but *raises* Historical ES by 1.3%. VaR and ES can disagree on the same trade because they measure different parts of the tail; adding INFY.NS at 15% raises both.

## Excel report

One click exports a `.xlsx` workbook:

- **Dashboard:** position settings, beta, risk-free assumption and data source, plus VaR at 90/95/97.5/99% and ES for every model, with each model's multi-day rule
- **Portfolio Risk** (portfolio mode only): the decomposition on the chosen basis (standalone, component and incremental risk, with live `SUM` totals), the diversification summary, the holding limiting the sample, and the correlation matrix
- **Backtesting:** the recommended model, and the full model-comparison table with tick loss, PASS / FAIL / LOW POWER and the McNeil-Frey ES test
- **Stress Testing:** measured crisis scenarios (peak, trough, market fall, recovery, replay or β-proxy, P&L), the volatility shock and worst actual losses
- **Raw Data:** prices, simple and log returns, rolling volatility

## Run it locally

Requires Python 3.11 or newer.

```bash
git clone https://github.com/singhwilliam15/VaR-Analysis-Tool.git
cd VaR-Analysis-Tool
pip install -r requirements.txt
streamlit run app.py
```

On Windows you can double-click `run_app.bat` instead. The project also opens directly in GitHub Codespaces, which starts the app for you.

Use Yahoo Finance symbols: `.NS` for NSE (`RELIANCE.NS`), `.BO` for BSE, and no suffix for US stocks (`AAPL`).

## Tests

```bash
pytest
```

The 125 tests check each calculation against an independent reference. They need no network: market data is mocked.

- GARCH(1,1)-t recovering the true parameters from simulated GARCH-t data; the variance filter, term structure and simulated paths checked against hand-written recursions; rolling GARCH forecasts never using future data; fallbacks counted
- the unit-variance t quantile and ES against 3 million simulated draws
- GARCH Monte Carlo matching the analytic GARCH-t VaR/ES at 1 day; FHS worked by hand
- Student-t maximum likelihood recovering known parameters, with a fallback when the optimiser fails
- the Cornish-Fisher validity condition against a brute-force monotonicity check
- the McNeil-Frey test passing a correct model and rejecting ES understated by 30%
- Student-t VaR and ES against 2 million simulated draws
- t-day parametric VaR equal to `z·σ·√t − μ·t`, and overlapping t-day VaR worked by hand
- CAGR from the compounded path, and Sortino downside deviation worked by hand
- tick loss worked by hand, and lowest at the true quantile on simulated data
- the recommended-model rule and the low-power threshold
- portfolio components adding up to the total on every basis and horizon, matching the headline Historical/Parametric figures, agreeing with the normal Euler shares on 400,000 simulated days, and component ES worked by hand
- incremental VaR, what-if weights, buy-and-hold values and the limiting-ticker report
- stress testing: drawdown and recovery by hand, replay equal to the compounded actual return, proxy rows using downside beta, downside beta recovering a known crisis beta (and staying stable), scenario config well-formed, volatility shock scaling exactly with k
- the data fetcher (mocked yfinance and Yahoo JSON): adjusted prices, a metadata failure keeping prices, exchange-timezone dates, and the suspicious-move check
- the EWMA recursion worked by hand
- Cornish-Fisher reducing to the normal model when skew and kurtosis are zero
- the Kupiec and Christoffersen statistics against known values
- the Basel zones against the regulatory table
- that rolling forecasts never use future data
- beta recovery on synthetic data
- component VaR summing to portfolio VaR
- marginal VaR against a finite-difference derivative
- zero diversification benefit for perfectly correlated assets
- the Excel report's contents

## Project structure

```text
app.py                 Streamlit dashboard
var_calculator.py      VaR/ES models, rolling forecasts, backtests, beta and stress testing
garch.py               GARCH(1,1)-t fitting, filtering, simulation; FHS
portfolio.py           Portfolio construction, risk decomposition, incremental VaR, what-if
stress.py              Measured crisis scenarios, historical replay, downside beta, volatility shock
stress_scenarios.csv   Editable crisis windows (Nifty 50 and S&P 500)
data_fetcher.py        Yahoo Finance download, with a direct-HTTP fallback
excel_exporter.py      Formatted Excel report
test_*.py              Unit tests
```

## Limitations

- **Portfolios are long-only and in a single currency.** There is no FX conversion and no short positions.
- **Risk decomposition** covers Historical ES, Historical VaR and Parametric VaR. The other models (GARCH, FHS, …) report a portfolio total only. Under buy-and-hold, the decomposition applies today's drifted weights to the history, so its total can differ from the headline figure, which follows the buy-and-hold path.
- **Portfolio crisis replay needs every holding to have prices for the crisis.** One recently listed holding (for example LICI.NS, listed May 2022) switches the whole portfolio to β-proxy for older crises; the app says which holding causes it. Replaying holdings with data and proxying only the missing one would be more accurate.
- **Correlations** are full-sample estimates. In a crisis, correlations usually rise, so the diversification benefit shrinks when it is needed most.
- **Multi-day VaR** uses √t for volatility, which assumes independent returns. Volatility clusters, so long-horizon figures are approximate. The overlapping-window check shows by how much.
- **The risk-free rate** is a user-set assumption, not a live market rate.
- **Cornish-Fisher** is an approximation. Even inside its valid region it can overstate the 99% tail and understate the 90% one.
- **GARCH(1,1)-t** has a constant mean and symmetric response to shocks (no leverage term such as GJR or EGARCH). Its multi-day formula applies a 1-day t-quantile to the summed variance, which overstates the t-day tail; Monte Carlo is the better multi-day estimate.
- **Rolling backtest shortcuts for speed:** Student-t and GARCH are refitted every 20 days rather than daily, and GARCH uses at most 1,000 past days. A full-history run (for example AAPL `max`, about 11,000 days) takes about a minute.
- **The Acerbi-Székely (2014) ES test** (the plan's stretch goal) is not implemented; McNeil-Frey is the only ES backtest.
- **Monte Carlo at 1 day** adds nothing beyond GARCH-t, so it is excluded from the recommendation.
- **Stress scenarios** are historical. They cannot capture a crisis unlike past ones, and β-proxy rows assume the stock's crisis sensitivity in the lookback window would have held in earlier crises.

## Built with

Python, pandas, NumPy, SciPy, arch, Streamlit, Plotly, openpyxl, yfinance, pytest
