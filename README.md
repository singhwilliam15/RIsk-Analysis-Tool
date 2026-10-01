# VaR Analysis Tool

A Streamlit dashboard that measures the market risk of a single stock or a multi-stock portfolio and tests which risk model is reliable. It pulls daily prices from Yahoo Finance and estimates Value at Risk (VaR) and Expected Shortfall (ES) with six models. Each model is backtested out of sample with the Kupiec and Christoffersen tests. Beta-adjusted crisis scenarios are run, and everything exports to a formatted Excel report.

It started as an Excel VaR workbook. This project rebuilds it in Python, so any NSE, BSE or US ticker can be analysed in seconds, and adds the model-validation layer a spreadsheet makes hard.

## Models

| Model | How it estimates the loss quantile | Captures fat tails? | Reacts to recent volatility? |
| --- | --- | --- | --- |
| **Historical** | Empirical percentile of past returns | Yes, if they are in the sample | No |
| **Parametric (Normal)** | `z·σ − μ` | No | No |
| **Student-t** | t-distribution; degrees of freedom from excess kurtosis (`ν = 4 + 6/K`) | Yes | No |
| **Cornish-Fisher** | Normal quantile adjusted for skewness and kurtosis | Yes | No |
| **EWMA (RiskMetrics)** | Normal quantile on an exponentially weighted volatility forecast, λ = 0.94 | No | Yes |
| **Monte Carlo** | 1,000–10,000 simulated returns from a fitted normal | No | No |

Each model reports VaR at 90%, 95% and 99%, plus Expected Shortfall: closed form for the Normal, Student-t and EWMA models, the empirical tail average for Historical and Monte Carlo, and numerical integration for Cornish-Fisher. Monte Carlo uses a seeded local random generator. The app warns when fewer than 50 simulated draws land in the tail, since ES is then very noisy (for example, 1,000 simulations at 99% leave only 10).

**Multi-day horizons (1–30 days)** follow one rule per model type:

| Model type | t-day VaR / ES |
| --- | --- |
| Parametric (Normal, Student-t, Cornish-Fisher, EWMA) | `z·σ·√t − μ·t`: volatility grows with √t, the mean with t |
| Historical, Monte Carlo | 1-day quantile × √t |

For Historical, the app also shows the **empirical t-day VaR from overlapping compounded t-day returns**, so the √t assumption can be checked against the data. Overlapping windows share days, so treat that figure as a sense check, not a precise estimate.

## Data and performance statistics

- **Prices:** split- and dividend-adjusted closes from yfinance. If yfinance fails, the app calls Yahoo's chart API directly and uses its adjusted closes (raw closes only if none are returned). The source and price basis are shown under the page title. Dates are converted with the exchange's timezone, so they don't depend on where the app is hosted. If company metadata can't be fetched, the price history is kept and the ticker is used as the name.
- **CAGR** is computed from the compounded return path. The arithmetic mean × 252 is reported separately as `ann_mean_return`.
- **Sharpe and Sortino** use an editable risk-free rate: 6.5% for INR and 4.0% for USD by default. These are **assumptions, not live rates**. Sortino divides by the downside deviation, `√mean(min(r − r_f, 0)²)`, taken over all days.

## Portfolio mode

Enter any number of tickers and weights (one currency). The portfolio is held at constant weights, rebalanced daily, on the dates every holding traded. All six models, the backtests and the stress tests then run on the portfolio's return series. A **Portfolio Risk** tab shows where the risk comes from:

| Measure | Definition |
| --- | --- |
| **Standalone VaR** | Historical VaR of each holding on its own |
| **Diversification benefit** | Sum of standalone VaRs − portfolio VaR: the loss diversification removes |
| **Marginal VaR** | `∂VaR/∂wᵢ = z·(Σw)ᵢ/σₚ − μᵢ` (1-day): extra VaR per unit of extra weight |
| **Component VaR** | `wᵢ·(z·(Σw)ᵢ/σₚ·√t − μᵢ·t)` (Euler allocation); the components add up exactly to the portfolio's parametric t-day VaR |
| **Risk / weight** | Share of risk ÷ share of capital; above 1× means the holding adds more risk than capital |
| **Correlation matrix** | Pairwise correlation of daily returns, as a heatmap |

## Backtesting

Every model is tested **out of sample**: each day's VaR is estimated only from the previous 250 trading days and then compared with that day's actual return. All models are scored on the same days.

| Test | Question it answers | Distribution |
| --- | --- | --- |
| **Kupiec POF** | Is the number of breaches right? | χ²(1) |
| **Christoffersen independence** | Do breaches cluster on consecutive days? | χ²(1) |
| **Conditional coverage** | Both together | χ²(2) |
| **Basel traffic light** | Regulatory zone from the binomial distribution of breaches | Green < 95% ≤ Yellow < 99.99% ≤ Red |

A model passes when all three p-values are at least 0.05. The traffic-light rule reproduces the regulatory 0–4 / 5–9 / 10+ zones at 99% over 250 days, and it is applied correctly at any confidence level and sample length.

**Choosing a model.** A higher p-value is not evidence of a better model, so the tool does not rank on p-values. It ranks on the **tick (quantile) loss** of each model's out-of-sample VaR:

`L = mean[(α − 1{r < −VaR})·(r + VaR)]`

The loss is lowest, on average, for the true quantile. The **recommended model** is the one with the lowest tick loss among the models that pass all three tests. If none pass, the app says so and shows the lowest-loss model with a warning.

**Statistical power.** The estimation window is always 250 days. A verdict needs at least **250 test days at 99%** (2.5 expected breaches) and **100 at 90–95%**. With fewer, the verdict shows as `LOW POWER` and the app suggests a longer lookback. A 1-year lookback leaves almost no test days, so use 2y (95%) or 5y / max (99%).

## Stress testing

- **Nine historical crises** (GFC 2008, COVID-19 2020, dot-com 2000–02, Black Monday 1987, etc.). Each market drawdown is scaled by the stock's **beta** to the Nifty 50 (Indian tickers) or S&P 500 (US tickers).
- **A custom market-shock slider**, beta-adjusted in the same way.
- **Worst actual losses in the sample** over 1, 5, 10 and 21 days, for comparison against VaR.

## Example results

Live data, run on 1 Oct 2026.

**Apple (AAPL): 99% VaR, 5-year lookback, 1,003 out-of-sample test days (10 breaches expected)**

| Model | Breaches | Kupiec p | Independence p | Cond. coverage p | Basel | Verdict | Tick loss (bp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Historical | 13 | 0.367 | 0.156 | 0.244 | 🟢 Green | PASS | **6.438** |
| Parametric (Normal) | 15 | 0.142 | 0.218 | 0.159 | 🟡 Yellow | PASS | 6.563 |
| Student-t | 12 | 0.544 | 0.130 | 0.264 | 🟢 Green | PASS | 6.495 |
| Cornish-Fisher | 11 | 0.762 | 0.106 | 0.258 | 🟢 Green | PASS | 7.122 |
| EWMA (RiskMetrics) | 19 | 0.011 | 0.049 | 0.006 | 🟡 Yellow | **FAIL** | 6.525 |

At 99%, the normal-based models breach more often than they should; Apple's returns have fatter tails than a normal curve. EWMA adapts quickly to volatility, but it still assumes normal tails, so it fails at 99%, and its breaches cluster (2 back-to-back). Among the four passing models, **Historical** has the lowest tick loss and is recommended. Cornish-Fisher has the breach count closest to target (11), but the highest tick loss: its VaR is often far too conservative. A breach count alone doesn't capture that.

**Asian Paints (ASIANPAINT.NS): 95% VaR, 5-year lookback, 990 test days.** Every model gets the breach **count** right: 47–58 against 49.5 expected, with Kupiec p from 0.23 to 0.94. But every model fails the **independence** test (p ≤ 0.025), because breaches arrive in clusters (6–10 back-to-back pairs). The app reports "No model passes" and shows Student-t (lowest tick loss) with a warning, rather than naming a "best" model. A breach-count test alone would have passed all five.

**Multi-day check, 99% 10-day Historical VaR, US$1m in AAPL:** √t scaling gives $152,075; the actual overlapping 10-day returns give $116,868. For ASIANPAINT.NS over 5 years it goes the other way: ₹1,25,327 from √t against ₹1,35,912 actual. √t is a rough approximation in both directions.

**Asian Paints (ASIANPAINT.NS): ₹10,00,000 position, 1-day horizon, 2-year lookback**

| Model | 95% VaR | 99% VaR | 95% ES |
| --- | --- | --- | --- |
| Historical | ₹24,420 | ₹34,733 | ₹32,744 |
| Parametric (Normal) | ₹23,847 | ₹33,531 | ₹29,784 |
| Student-t | ₹22,985 | ₹36,999 | ₹31,964 |
| Cornish-Fisher | ₹23,271 | ₹44,991 | ₹37,090 |
| EWMA (RiskMetrics) | ₹19,573 | ₹27,683 | ₹24,546 |
| Monte Carlo (5,000 sims) | ₹24,439 | ₹33,788 | ₹30,185 |

Excess kurtosis is 3.19 and the Jarque-Bera test rejects normality (p < 0.0001). This is why the fat-tailed models give the highest 99% VaR. The beta to the Nifty 50 is 0.87, so a 20% market fall maps to a 17.3% fall in the stock.

**Five-stock NSE portfolio: ₹10,00,000, 95% 1-day VaR, 2-year lookback (500 common days)**

| Holding | Weight | Standalone VaR | Component VaR | Share of risk | Risk / weight |
| --- | --- | --- | --- | --- | --- |
| Reliance | 30% | ₹6,236 | ₹5,150 | 33.8% | 1.13× |
| HDFC Bank | 25% | ₹5,038 | ₹3,620 | 23.7% | 0.95× |
| TCS | 20% | ₹5,155 | ₹3,243 | 21.3% | 1.06× |
| Asian Paints | 15% | ₹3,663 | ₹2,211 | 14.5% | 0.97× |
| Britannia | 10% | ₹2,093 | ₹1,023 | 6.7% | 0.67× |
| **Total** | 100% | **₹22,184** | **₹15,247** | 100% | |

The portfolio's Historical VaR is ₹15,601, against ₹22,184 if each position's risk is added up separately. Diversification removes **₹6,583 (30%)** of the risk; the average correlation is only 0.26. Reliance contributes more risk than its weight because it is the most correlated with the others (0.41 with HDFC Bank). Britannia is the best diversifier.

## Excel report

One click exports a `.xlsx` workbook:

- **Dashboard:** position settings, beta, risk-free assumption and data source, plus VaR at 90/95/99% and ES for every model, with each model's multi-day rule
- **Portfolio Risk** (portfolio mode only): diversification summary, the component-VaR table with live `SUM` totals, and the correlation matrix
- **Backtesting:** the recommended model, and the full model-comparison table with tick loss and PASS / FAIL / LOW POWER
- **Stress Testing:** beta-adjusted scenarios and worst actual losses
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

The 64 tests check each calculation against an independent reference. They need no network: market data is mocked.

- Student-t VaR and ES against 2 million simulated draws
- t-day parametric VaR equal to `z·σ·√t − μ·t`, and overlapping t-day VaR worked by hand
- CAGR from the compounded path, and Sortino downside deviation worked by hand
- tick loss worked by hand, and lowest at the true quantile on simulated data
- the recommended-model rule and the low-power threshold
- the data fetcher (mocked yfinance and Yahoo JSON): adjusted prices, a metadata failure keeping prices, exchange-timezone dates
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
var_calculator.py      VaR/ES models, backtests, beta and stress testing
portfolio.py           Portfolio construction, component VaR, diversification
data_fetcher.py        Yahoo Finance download, with a direct-HTTP fallback
excel_exporter.py      Formatted Excel report
test_*.py              Unit tests
```

## Limitations

- **Portfolios are long-only, in a single currency, at constant weights.** There is no FX conversion and no short positions.
- **Component VaR** decomposes the parametric (normal) VaR. The other models report a portfolio total only.
- **Correlations** are full-sample estimates. In a crisis, correlations usually rise, so the diversification benefit shrinks when it is needed most.
- **Multi-day VaR** uses √t for volatility, which assumes independent returns. Volatility clusters, so long-horizon figures are approximate. The overlapping-window check shows by how much.
- **The risk-free rate** is a user-set assumption, not a live market rate.
- **Cornish-Fisher** is an approximation. With very high kurtosis it can overstate the 99% tail and understate the 90% one, as the Apple figures show.
- **Stress scenarios** use approximate index drawdowns scaled by a single beta. Real crisis betas are usually higher than normal-period betas.

## Built with

Python, pandas, NumPy, SciPy, Streamlit, Plotly, openpyxl, yfinance, pytest
