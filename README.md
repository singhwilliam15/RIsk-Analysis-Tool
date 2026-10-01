# VaR Analysis Tool

A Streamlit dashboard that measures the market risk of a single stock position. It pulls daily prices from Yahoo Finance, estimates Value at Risk (VaR) and Expected Shortfall (ES) with three methods, backtests the model out of sample, runs historical stress scenarios, and exports the results to a formatted Excel report.

It started as an Excel VaR workbook. This project rebuilds that workbook in Python so any NSE, BSE or US ticker can be analysed in seconds, and adds an out-of-sample backtest.

## What it calculates

| Area | Method |
| --- | --- |
| **Historical VaR & ES** | Empirical percentile of daily returns; ES is the average loss beyond VaR |
| **Parametric VaR & ES** | Variance-covariance under a normal distribution: `VaR = z·σ − μ`, `ES = φ(z)/α·σ − μ` |
| **Monte Carlo VaR & ES** | 1,000–10,000 simulated returns from N(μ, σ), fixed seed for reproducibility |
| **Holding period** | 1–30 days, scaled by √t |
| **Backtesting** | Rolling 250-day Historical VaR, so each day's forecast uses only past data. The Kupiec proportion-of-failures test gives an LR statistic and p-value; the Basel traffic light is set by the binomial cumulative-probability rule (green < 95%, red ≥ 99.99%), which reproduces the regulatory 0–4 / 5–9 / 10+ zones at 99% over 250 days |
| **Stress testing** | Nine historical crises (GFC 2008, COVID-19 2020, dot-com 2000–02, etc.) plus a custom shock slider |
| **Performance stats** | Annualised return and volatility, Sharpe, Sortino, maximum drawdown, skewness, kurtosis |

## Example output

Live run on 1 Oct 2026 with the app's default settings: ₹10,00,000 position, 1-day horizon, 2-year lookback.

**Asian Paints (ASIANPAINT.NS)**

| Method | 95% VaR | 99% VaR | 95% ES |
| --- | --- | --- | --- |
| Historical | ₹24,420 (2.44%) | ₹34,733 (3.47%) | ₹32,744 (3.27%) |
| Parametric | ₹23,863 (2.39%) | ₹33,548 (3.35%) | ₹29,801 (2.98%) |
| Monte Carlo | ₹23,419 (2.34%) | ₹34,069 (3.41%) | ₹29,905 (2.99%) |

Historical ES is about 10% higher than the normal-model ES: the empirical return distribution has a fatter left tail than the normal curve assumes.

**Backtest:** 95% Historical VaR over 250 out-of-sample days gave 16 breaches against 12.5 expected (Kupiec p = 0.33). The model is not rejected.

## Excel report

One click exports a `.xlsx` workbook with four sheets:

- **Dashboard:** position settings and VaR/ES by method at 90%, 95% and 99%
- **Raw Data:** prices, simple and log returns, 30-day rolling volatility
- **Backtesting:** Kupiec test and Basel traffic light
- **Stress Testing:** loss and post-shock value for each scenario

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

The tests check the parametric VaR against its closed form, that ES always exceeds VaR, that the three methods agree on normally distributed data, square-root-of-time scaling, that rolling VaR never uses future data, a known Kupiec value, and that the traffic light reproduces the Basel table.

## Project structure

```text
app.py                 Streamlit dashboard
var_calculator.py      VaR, ES, backtesting and stress-testing logic
data_fetcher.py        Yahoo Finance download, with a direct-HTTP fallback
excel_exporter.py      Formatted Excel report
test_var_calculator.py Unit tests
```

## Limitations

- Single-asset positions only. There is no correlation between holdings.
- Parametric and Monte Carlo VaR assume normal returns, so they understate tail risk for fat-tailed stocks.
- √t scaling assumes independent, identically distributed returns. It ignores volatility clustering.
- Stress scenarios apply a fixed historical index drawdown, not a stock-specific beta.

## Built with

Python, pandas, NumPy, SciPy, Streamlit, Plotly, openpyxl, yfinance
