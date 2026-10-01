# VaR Analysis Tool

A Streamlit dashboard that measures the market risk of a single stock position and tests which risk model is reliable. It pulls daily prices from Yahoo Finance and estimates Value at Risk (VaR) and Expected Shortfall (ES) with six models. Each model is backtested out of sample with the Kupiec and Christoffersen tests. Beta-adjusted crisis scenarios are run, and everything exports to a formatted Excel report.

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

Each model reports VaR at 90%, 95% and 99%, plus Expected Shortfall: closed form for the Normal, Student-t and EWMA models, the empirical tail average for Historical and Monte Carlo, and numerical integration for Cornish-Fisher. Multi-day horizons (1–30 days) are scaled by √t.

## Backtesting

Every model is tested **out of sample**: each day's VaR is estimated only from the previous 250 trading days and then compared with that day's actual return. All models are scored on the same days.

| Test | Question it answers | Distribution |
| --- | --- | --- |
| **Kupiec POF** | Is the number of breaches right? | χ²(1) |
| **Christoffersen independence** | Do breaches cluster on consecutive days? | χ²(1) |
| **Conditional coverage** | Both together | χ²(2) |
| **Basel traffic light** | Regulatory zone from the binomial distribution of breaches | Green < 95% ≤ Yellow < 99.99% ≤ Red |

A model passes when all three p-values are at least 0.05. The traffic-light rule reproduces the regulatory 0–4 / 5–9 / 10+ zones at 99% over 250 days, and it is applied correctly at any confidence level and sample length.

## Stress testing

- **Nine historical crises** (GFC 2008, COVID-19 2020, dot-com 2000–02, Black Monday 1987, etc.). Each market drawdown is scaled by the stock's **beta** to the Nifty 50 (Indian tickers) or S&P 500 (US tickers).
- **A custom market-shock slider**, beta-adjusted in the same way.
- **Worst actual losses in the sample** over 1, 5, 10 and 21 days, for comparison against VaR.

## Example results

Live data, run on 1 Oct 2026.

**Apple (AAPL): 99% VaR, 5-year lookback, 1,003 out-of-sample test days (10 breaches expected)**

| Model | Breaches | Kupiec p | Independence p | Cond. coverage p | Basel | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Historical | 13 | 0.367 | 0.156 | 0.244 | 🟢 Green | PASS |
| Parametric (Normal) | 15 | 0.142 | 0.218 | 0.159 | 🟡 Yellow | PASS |
| Student-t | 12 | 0.544 | 0.130 | 0.264 | 🟢 Green | PASS |
| Cornish-Fisher | 11 | 0.762 | 0.106 | 0.258 | 🟢 Green | PASS |
| EWMA (RiskMetrics) | 19 | 0.011 | 0.049 | 0.006 | 🟡 Yellow | **FAIL** |

At 99%, the normal-based models breach more often than they should; Apple's returns have fatter tails than a normal curve. The fat-tailed Student-t and Cornish-Fisher models come closest to the target. EWMA adapts quickly to volatility, but it still assumes normal tails, so it fails at 99%, and its breaches cluster (2 back-to-back).

**Asian Paints (ASIANPAINT.NS): ₹10,00,000 position, 1-day horizon, 2-year lookback**

| Model | 95% VaR | 99% VaR | 95% ES |
| --- | --- | --- | --- |
| Historical | ₹24,420 | ₹34,732 | ₹32,744 |
| Parametric (Normal) | ₹23,847 | ₹33,531 | ₹29,784 |
| Student-t | ₹22,985 | ₹36,999 | ₹31,964 |
| Cornish-Fisher | ₹23,271 | ₹44,991 | ₹37,090 |
| EWMA (RiskMetrics) | ₹19,573 | ₹27,683 | ₹24,546 |
| Monte Carlo | ₹23,403 | ₹34,052 | ₹29,889 |

Excess kurtosis is 3.19 and the Jarque-Bera test rejects normality (p < 0.0001). This is why the fat-tailed models give the highest 99% VaR. The beta to the Nifty 50 is 0.87, so a 20% market fall maps to a 17.3% fall in the stock.

## Excel report

One click exports a `.xlsx` workbook with four sheets:

- **Dashboard:** position settings and beta, plus VaR at 90/95/99% and ES for every model
- **Backtesting:** the full model-comparison table, with PASS/FAIL highlighted
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

The 27 tests check each model against an independent reference:

- Student-t VaR and ES against 2 million simulated draws
- the EWMA recursion worked by hand
- Cornish-Fisher reducing to the normal model when skew and kurtosis are zero
- the Kupiec and Christoffersen statistics against known values
- the Basel zones against the regulatory table
- that rolling forecasts never use future data
- beta recovery on synthetic data
- the Excel report's contents

## Project structure

```text
app.py                 Streamlit dashboard
var_calculator.py      VaR/ES models, backtests, beta and stress testing
data_fetcher.py        Yahoo Finance download, with a direct-HTTP fallback
excel_exporter.py      Formatted Excel report
test_var_calculator.py Unit tests
```

## Limitations

- **Single-asset positions only.** There is no correlation between holdings.
- **√t scaling** for multi-day VaR assumes independent returns. EWMA shows that volatility clusters, so long-horizon figures are approximate.
- **Cornish-Fisher** is an approximation. With very high kurtosis it can overstate the 99% tail and understate the 90% one, as the Apple figures show.
- **Stress scenarios** use approximate index drawdowns scaled by a single beta. Real crisis betas are usually higher than normal-period betas.

## Built with

Python, pandas, NumPy, SciPy, Streamlit, Plotly, openpyxl, yfinance, pytest
