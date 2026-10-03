# Methodology

This document describes every calculation in the Risk Analysis Tool: the eight market-risk models, the multi-day rules, the backtests, the model-selection rule, the portfolio risk decomposition, the stress tests, and the data layer that every risk pillar shares (positions, prices and volume, fundamentals, disclosures and data quality), the trust layer (90% ranges, model risk and A–D grades), the liquidity, credit, concentration and factor, and event and governance pillars, the integrated stress engine and the decision layer. Section numbers match the code modules named in each heading. The pillars still to come (liquidity, credit, concentration, event risk, integration) are planned in `RISK_TOOL_PLAN_V3.md` and will be documented here as they are built.

**Notation.** `r_t` is the simple daily return on day t. α = 1 − confidence level is the tail probability (α = 0.01 for 99%). Losses are positive numbers, so VaR and ES are reported as positive amounts. `z = Φ⁻¹(1 − α)` is the standard normal quantile and φ is the normal density. μ and σ are the sample mean and standard deviation of daily returns. `t` (in multi-day formulas) is the holding period in days.

---

## 1. VaR and ES models (`var_calculator.py`, `garch.py`)

VaR at confidence 1 − α is the loss exceeded with probability α. Expected Shortfall (ES) is the average loss given that the loss exceeds VaR.

### 1.1 Historical simulation
```
VaR = −q_α(r)                ES = −mean(r | r ≤ q_α(r))
```
`q_α` is the empirical α-quantile, interpolated linearly between order statistics. No distribution is assumed, so the tails are only as fat as the sample.

### 1.2 Parametric (normal)
```
VaR = z·σ − μ                ES = φ(z)/α · σ − μ
```

### 1.3 Student-t
The degrees of freedom ν, location m and scale s are fitted by maximum likelihood (`scipy.stats.t.fit`), starting from the method-of-moments estimate `ν₀ = 4 + 6/K`, where K is the sample excess kurtosis. If the optimiser fails or returns ν ≤ 2.05, the method-of-moments fit is used instead, with the scale set to match the sample variance. With `q = t_ν⁻¹(1 − α)`:
```
VaR = s·q − m                ES = s · f_ν(q)/α · (ν + q²)/(ν − 1) − m
```
`f_ν` is the Student-t density. The ES formula is the closed form for the t-distribution.

### 1.4 Cornish-Fisher (modified VaR)
The normal quantile `z_α = Φ⁻¹(α)` is adjusted for skewness S and excess kurtosis K:
```
z_CF = z + (z² − 1)·S/6 + (z³ − 3z)·K/24 − (2z³ − 5z)·S²/36
VaR  = −(μ + z_CF·σ)
```
ES is the average Cornish-Fisher loss over the tail, `−(μ + σ·mean z_CF(u))` for u in (0, α). It is computed by the midpoint rule on 2,000 points.

**Validity check.** The expansion is a valid quantile function only if it is increasing in z. Its derivative is `a·z² + b·z + c`, with `a = K/8 − S²/6`, `b = S/3` and `c = 1 − K/8 + 5S²/36`. This is non-negative for every z if and only if `a ≥ 0` and `b² − 4ac ≤ 0` (Maillard, 2012). Outside that region the app marks the model ⚠.

### 1.5 EWMA (RiskMetrics)
```
σ²_t = λ·σ²_{t−1} + (1 − λ)·r²_{t−1},   λ = 0.94
VaR = z·σ_{T+1}              ES = φ(z)/α · σ_{T+1}
```
The mean is taken as zero (J.P. Morgan/Reuters, 1996). The recursion starts from the variance of the first 30 returns.

### 1.6 Filtered Historical Simulation (FHS)
Each return is divided by its EWMA volatility forecast, `z_t = r_t / σ_t`. Tomorrow's figures rescale the empirical tail of these standardised returns by tomorrow's volatility:
```
VaR = −q_α(z)·σ_{T+1}        ES = −mean(z | z ≤ q_α(z))·σ_{T+1}
```
FHS keeps the empirical tail shape and reacts to current volatility (Barone-Adesi, Giannopoulos and Vosper, 1999).

### 1.7 GARCH(1,1) with Student-t errors
```
r_t = μ + ε_t,   ε_t = σ_t·η_t,   η_t ~ unit-variance Student-t(ν)
σ²_t = ω + α₁·ε²_{t−1} + β₁·σ²_{t−1}
```
This is Bollerslev (1986), with the Student-t errors of Bollerslev (1987). The model is fitted by maximum likelihood with the `arch` package, on returns × 100 for numerical stability. A fit counts as valid only if the optimiser converged and the model is stationary: ω > 0, α₁, β₁ ≥ 0, α₁ + β₁ < 1 and ν > 2.05. Otherwise the app falls back to EWMA and says so. With `q̃ = t_ν⁻¹(α)·√((ν−2)/ν)`, the unit-variance t quantile:
```
VaR = −(μ + σ_{T+1}·q̃)        ES = σ_{T+1}·ES̃_ν(α) − μ
ES̃_ν(α) = f_ν(q)/α · (ν + q²)/(ν − 1) · √((ν−2)/ν),   q = t_ν⁻¹(1 − α)
```

### 1.8 Monte Carlo (GARCH-t paths)
The app simulates N paths (1,000–10,000) of the fitted GARCH(1,1)-t process from tomorrow's variance, updating volatility along each path. The returns are compounded over the holding period. VaR and ES are the empirical α-quantile and tail mean of the simulated returns. A seeded local random generator makes the results reproducible. The app warns when `N·α < 50`, i.e. too few tail draws for a stable ES. If GARCH does not converge, Monte Carlo samples a fitted normal distribution instead.

## 2. Multi-day horizons

| Model | t-day VaR (ES analogous) |
| --- | --- |
| Normal, Student-t, Cornish-Fisher, EWMA | `(VaR₁ + μ)·√t − μ·t`, i.e. `z·σ·√t − μ·t`: volatility scales with √t, the mean with t (EWMA has μ = 0) |
| Historical, FHS | `VaR₁·√t` |
| GARCH(1,1)-t | `−(μ·t + √(Σ_h E[σ²_{T+h}])·q̃)`, with `E[σ²_{T+h}] = ω + (α₁ + β₁)·E[σ²_{T+h−1}]` |
| Monte Carlo | Empirical quantile of simulated compounded t-day returns |

For Historical, the app also reports the empirical t-day VaR from overlapping compounded t-day returns, as a check on √t. Overlapping windows share days, so this figure is a sense check, not an independent estimate.

## 3. Backtesting (`var_calculator.py`)

### 3.1 Out-of-sample forecasts
Every model forecasts day t from data before t only, and all models are scored on the same days. Tests check that changing a future return never changes an earlier forecast.

| Model | Estimation for the rolling forecasts |
| --- | --- |
| Historical, Normal, Cornish-Fisher, FHS | Previous 250 days, updated daily |
| Student-t | Maximum likelihood on the previous 250 days, refitted every 20 days |
| GARCH(1,1)-t, Monte Carlo | All earlier data, up to 1,000 days, refitted every 20 days; σ filtered daily with the latest parameters; a non-converged fit uses EWMA for that block (counted and reported) |

A **breach** is a day with `r_t < −VaR_t`. With T test days, N breaches and `p̂ = N/T`:

### 3.2 Kupiec proportion-of-failures test (Kupiec, 1995)
```
LR_POF = −2·ln[ α^N (1−α)^(T−N) / ( p̂^N (1−p̂)^(T−N) ) ]  ~ χ²(1)
```
`0·ln 0 = 0`, so zero breaches still give a valid statistic. LR is floored at 0, so an exactly calibrated model cannot get a NaN p-value from rounding.

### 3.3 Christoffersen independence and conditional coverage (Christoffersen, 1998)
`n_ij` counts days in state j after a day in state i (1 = breach), with `π₀₁ = n₀₁/(n₀₀ + n₀₁)`, `π₁₁ = n₁₁/(n₁₀ + n₁₁)` and `π = (n₀₁ + n₁₁)/(n₀₀ + n₀₁ + n₁₀ + n₁₁)`:
```
LR_ind = −2·ln[ (1−π)^(n₀₀+n₁₀) π^(n₀₁+n₁₁) / ( (1−π₀₁)^n₀₀ π₀₁^n₀₁ (1−π₁₁)^n₁₀ π₁₁^n₁₁ ) ]  ~ χ²(1)
LR_cc  = LR_POF + LR_ind  ~ χ²(2)
```
A model **passes** when all three p-values are at least 0.05.

### 3.4 Basel traffic light (Basel Committee, 1996)
The zone comes from the binomial probability of seeing at most N breaches when the true rate is α: green if `P(X ≤ N) < 95%`, red if it is `≥ 99.99%`, yellow in between. At 99% over 250 days, this reproduces the regulatory zones of 0–4 (green), 5–9 (yellow) and 10+ (red). The same rule is then applied at any confidence level and sample length. The binomial CDF is computed in log space.

### 3.5 Statistical power
The estimation window stays at 250 days. A verdict needs at least 250 test days at 97.5% and 99%, and 100 at 90% and 95%. With fewer, it is reported as `LOW POWER` rather than PASS or FAIL.

### 3.6 Expected Shortfall test (McNeil and Frey, 2000)
On breach days, the standardised exceedance residual is `e_t = (L_t − ES_t)/σ_t`, where `L_t = −r_t` and σ_t is the model's own volatility forecast. If ES is correct, `E[e] = 0`. The one-sided test of `H₁: E[e] > 0` (ES too small) bootstraps the t-statistic of the residuals after centring them at zero (10,000 resamples, fixed seed), and needs at least 5 breaches. Basel's FRTB sets market-risk capital on 97.5% ES (Basel Committee, 2019), so the app offers 97.5% as a confidence level.

The Acerbi and Székely (2014) Z₂ test was considered and is **not implemented**.

### 3.7 Choosing a model: tick loss
A higher p-value is not evidence of a better model. Models are ranked by the average **quantile (tick) loss** of their out-of-sample VaR, with `q_t = −VaR_t`:
```
L = mean[ (α − 1{r_t < q_t}) · (r_t − q_t) ]
```
The tick loss is the scoring function of quantile regression (Koenker and Bassett, 1978). Its expectation is minimised by the true conditional quantile, which makes it a consistent way to compare quantile forecasts (Giacomini and Komunjer, 2005).
- **Recommended model:** the lowest tick loss among models that pass all three VaR tests **and do not fail the ES test** (§3.6). An ES test that could not run (too few breaches, low power) counts as not tested, not as a failure.
- **If every VaR-passing model fails the ES test:** the lowest-loss VaR-passing model is shown as the headline with a warning ("passes the VaR tests, fails the ES test"); it is not called recommended.
- **If none pass the VaR tests:** the app says so and shows the lowest-loss model with a warning.
- **Monte Carlo** is scored but excluded from the recommendation: at one day it is the GARCH-t model plus simulation noise.

## 4. Portfolio risk (`portfolio.py`)

The portfolio return is `r_p = Σ wᵢ·rᵢ` on the dates every holding traded.
- **Rebalanced daily:** constant weights.
- **Buy-and-hold:** weights drift as `w_{i,t} ∝ w_{i,0}·Π(1 + r_i)`.

### 4.1 Euler decomposition
VaR and ES are homogeneous of degree one in the weights, so Euler's theorem splits them exactly into contributions `wᵢ·∂ρ/∂wᵢ` that add up to the total (Tasche, 1999). The app offers three bases, and the components always sum to the total shown:

| Basis | Component of holding i |
| --- | --- |
| Historical ES | `wᵢ·E[−rᵢ \| r_p ≤ q_α(r_p)]`: the empirical tail average, exact |
| Historical VaR | `wᵢ·E[−rᵢ \| r_p ≈ q_α(r_p)]`, averaged over the days ranked within ±0.25% of the sample (at least ±2 days) of the quantile, then rescaled to the portfolio VaR |
| Parametric VaR | `wᵢ·( z·(Σw)ᵢ/σ_p·√t − μᵢ·t )` |

Historical components use √t for multi-day horizons, matching the Historical model.
- **Incremental risk** is the portfolio's risk with the holding minus its risk without it, other positions unchanged.
- The **diversification benefit** is the sum of standalone risks minus the portfolio risk.
- The **what-if** panel sets one weight (or adds a ticker) and scales the other holdings proportionally.

## 5. Stress testing (`stress.py`, `stress_scenarios.csv`)

1. **Scenario windows** are listed in `stress_scenarios.csv`: seven for the Nifty 50 and seven for the S&P 500. Each window is deliberately wider than the event.
2. **Market drawdown.** Inside each window, the largest peak-to-trough fall of the downloaded index price is `min_t (P_t / max_{s≤t} P_s) − 1`. **Recovery** is the number of trading days from the trough until the index regained the peak level. Windows outside the index history are flagged and excluded.
3. **Historical replay.** If the position has prices across the market's peak and trough dates, its scenario return is its actual compounded return between them.
4. **β-proxy**, used only without price data: market drawdown × downside beta, capped at −100%.
5. **Downside beta** is the sensitivity on the market's worst 10% of days, `β⁻ = Σ r_s·r_m / Σ r_m²` over those days (regression through the origin). An OLS slope with an intercept on the same days had a 2–4× larger bootstrap standard error on real NSE and US stocks, because the truncated market returns span a narrow range.
6. **Volatility shock.** VaR and ES are recomputed on `r′ = μ + k·(r − μ)` for k = 1, 2, 3.

## 6. Data and statistics (`data_fetcher.py`, `var_calculator.py`)

- **Prices** are split- and dividend-adjusted closes. Fallback timestamps are converted with the exchange's timezone.
- **Daily moves larger than 25%** are flagged; a move undone by the next day is labelled a likely data error. Data are never altered.
- **Statistics:**
  - CAGR = `(Π(1 + r))^(252/n) − 1`
  - Sharpe = `(mean(r) − r_f)/σ·√252`
  - Sortino = `(mean(r) − r_f) / √mean(min(r − r_f, 0)²) · √252`, with the daily risk-free rate `r_f` from a user-set annual assumption
  - Jarque-Bera = `n/6·(S² + K²/4) ~ χ²(2)` (Jarque and Bera, 1980)

## 7. Data layer (`data_fetcher.py`, `portfolio.py`, `fundamentals.py`, `disclosures.py`, `data_quality.py`)

Every pillar uses this layer. It never invents a figure: a value the source does not provide is shown as "not available".

### 7.1 Positions

Holdings can be entered as weights, share counts or money values. Each holding is stored as `{ticker, quantity, price, value, weight, sector}`, converted at the latest close `P_i`:

| Entry | Value `V_i` | Quantity `q_i` | Weight `w_i` |
| --- | --- | --- | --- |
| Weight `a_i` | `a_i / Σa · investment` | `V_i / P_i` | `V_i / ΣV` |
| Shares `q_i` | `q_i · P_i` | as entered | `V_i / ΣV` |
| Value `V_i` | as entered | `V_i / P_i` | `V_i / ΣV` |

- **Investment amount.** With share or value entry, the portfolio's total value `ΣV` replaces the sidebar investment amount.
- **Fractional quantities.** Weight entry can give fractional share counts.
- **Sector** is Yahoo Finance's classification, or "not available".
- **Buy-and-hold.** Weights are still treated as the weights at the start of the lookback window. With share entry they are today's weights, so a buy-and-hold history is an approximation.

### 7.2 Prices and volume

- **Columns.** Open, High, Low, Close and Volume are kept. Only a missing close drops a day; a day with no volume or no high/low stays, with that field missing.
- **Adjustment.** On the fallback source (the chart API), open, high and low are multiplied by the same adjustment factor as the close (`adjusted close / raw close`), so daily ranges stay consistent. Volume is not price-adjusted.
- **NSE + BSE volume.** For an Indian stock, the other listing (`.NS` ↔ `.BO`) is also downloaded. Its volume is added on the dates where both listings report volume. On other dates the primary listing's volume is used alone, and prices always come from the primary listing.
- **BSE coverage on Yahoo is patchy.** On 2 Oct 2026, Yahoo returned no usable 2-year `.BO` history for Reliance, HDFC Bank, Asian Paints and Britannia, but did for TCS and Jaiprakash Power. The Overview page shows which exchanges each holding's volume came from.

### 7.3 Fundamentals

- **What is fetched.** Annual income statement, balance sheet and cash flow (the latest 4–5 fiscal years that have data), plus shares outstanding, market cap, sector, industry and reporting currency, from yfinance.
- **Field mapping.** Yahoo's row names are mapped to canonical fields through the table `fundamentals.FIELDS`. Each canonical field has a list of aliases tried in order (for example total liabilities = "Total Liabilities Net Minority Interest", else "Total Liabilities"). Names are compared after removing case and punctuation, because yfinance has used both "TotalRevenue" and "Total Revenue".
- **Missing fields.** A field with no value in any year is listed in `missing_fields`; nothing is derived from other rows.
- **Overrides.** A CSV upload (template downloadable on the Overview page) replaces Yahoo's figures field by field and fiscal year by fiscal year, for holdings whose ticker matches without the exchange suffix.
  - Every figure keeps its source: "Yahoo Finance (yfinance): <Yahoo row name>", or "CSV upload: <your note>".
  - Rejected rows are listed with their spreadsheet row number.
- **Point in time.** A figure may be used "as of" a date only once it was public:
  - on its filing date, if known;
  - otherwise at **fiscal year end + 60 days** (`fundamentals.public_date`).

  Yahoo does not report filing dates, so Yahoo figures always use the 60-day rule. The rule is an assumption from the plan: Indian listed companies must publish audited annual results within 60 days of year end (SEBI LODR Reg. 33), though the full annual report comes later.

### 7.4 Disclosures (India)

- **Datasets.** Promoter pledges, ASM/GSM surveillance, price bands, the F&O ban list, rating actions and auditor events, from official sources only; `data/disclosures/README.md` gives the source of each.
- **Bundled snapshot (since October 2026).** `scripts/fetch_disclosures.py` reads NSE's public endpoints and writes a dated snapshot:
  - **Price bands** (`sec_list.csv`) and **F&O ban** files are saved unchanged.
  - **ASM/GSM** lists (JSON) go to the surveillance template. The lists show today's stage only, so `date_in` is the list date. The GSM stage is read from NSE's description ("GSM Stage 0", "Graded Surveillance Measure - Stage VI").
  - **Pledges** come from each quarter's shareholding-pattern XBRL (SEBI LODR Reg. 31): promoter holding, shares pledged as % of the promoter holding and of total shares, and the number pledged. Shares under non-disposal undertakings are not pledges and are left out. The public date is NSE's broadcast date. NSE's own pledge-data API returns nothing, and the shareholding history reaches back only about 20 quarters.
  - **Ratings** come from NSE's credit-rating disclosures (SEBI LODR Reg. 30), swept month by month for 24 months and kept for covered issuers, matched by the issuer code inside the rated ISIN. The action maps as: New → assigned; Reaffirm → reaffirmed; an explicit upgrade, downgrade, withdrawal or watch as stated; otherwise the change against the earlier rating in the same filing. A record whose action cannot be read is skipped, not guessed.
  - **Covered stocks:** the presets, Jaiprakash Power, and the five non-financial Nifty Midcap 150 / Smallcap 250 stocks with the highest share of total shares pledged, picked by a scan of every constituent's latest filing.
  - **Data as of:** each dataset's as-of date is shown on the Overview and Events pages.
- **Manifest.** Every file must be listed in `manifest.json` with its source URL, download date and coverage. Unlisted files are not loaded.
- **Validation.** Each row is checked for:
  - required columns and readable dates;
  - percentages between 0 and 100;
  - allowed values (surveillance measure, rating action, auditor event type; price band 2/5/10/20% or No Band);
  - consistency (pledged % of total shares ≤ promoter holding %; a surveillance exit on or after its entry).

  Rejected rows are reported with their row number, and the valid rows are kept.
- **Point in time.** Each dataset has a public date: the disclosure, entry, effective, trade, action or event date. A pledge row without a disclosure date becomes public at **quarter end + 21 days**, the filing deadline for the quarterly shareholding pattern (SEBI LODR Reg. 31(1)(b)).
- **Layouts.** NSE's `fo_secban.csv` is parsed structurally (its trade date comes from the first line), and `sec_list.csv` loads through the alias table (checked on a real download; it carries a 40% band, which is allowed). The JSON and XBRL sources are converted by `nse_sources.py`, which is tested on trimmed real downloads in `test_data/nse/`.

### 7.5 Data-quality score

Each holding scores 100 minus penalties, floored at 0. The thresholds and penalties are **assumptions**, set so that a clean, liquid NSE large-cap with enough history scores 100, and so that each defect that would distort a risk figure costs points:

| Check | Rule | Penalty | Why |
| --- | --- | --- | --- |
| Reversing spikes | daily move > 25% that the next day undoes by at least half (§6) | 15 each, max 30 | Almost always a bad price; it inflates historical VaR and ES. |
| Large moves, not reversed | daily move > 25% that lasts | 0 (reported) | Usually a real event. |
| Stale prices | days in runs of ≥ 3 identical closes | 10 if > 1% of days, 20 if > 5% | Repeated closes understate volatility. |
| Zero volume, unchanged close | volume 0 and close equal to the day before | 10 if > 2% of days, 20 if > 5% | On liquid NSE stocks these are mostly exchange holidays the source fills with the previous close (6 in 2 years for Reliance, HDFC Bank, Asian Paints and Britannia), so a few are tolerated. Many mean days without any trade. |
| Zero volume, price moved | volume 0 but the close changed | 5 if any, 10 if > 1% | Price and volume contradict each other. |
| No volume data | no volume at all | 5 | The liquidity pillar cannot run. |
| Gaps | gaps between trading days longer than 7 calendar days | 5 each, max 15 | Longer than any normal exchange holiday. |
| History length | fewer than 250 returns / fewer than 500 | 25 / 10 | 250 is one estimation window. 500 is that window plus the 250 out-of-sample days a 99% backtest needs (§3.5). A 2-year lookback gives 499 returns, one short, matching the Backtesting tab's `LOW POWER` flag at 99%. |
| Missing fundamentals | share of 10 key fields missing (revenue, EBIT, total assets and liabilities, current assets and liabilities, retained earnings, equity, cash from operations, shares outstanding) | up to 15, `round(15 × missing / 10)` | Later pillars (Altman Z, Merton) cannot run without them. Banks do not report EBIT or current assets, so they lose a few points by construction. |

The data are never altered. The lowest score among the holdings feeds the trust grade (§8.3).

**Known effect on market risk.** Holiday filler rows add zero returns to the sample, which slightly lowers volatility and VaR. They are kept, because the tool does not alter the downloaded data, and they are now counted on the Overview page.

## 8. Trust layer (`trust.py`)

Every headline number is reported as **value (90% range, grade)**, a `TrustedMetric(value, low, high, grade, reasons, sources, assumptions)`. Later pillars use the same interface.

### 8.1 90% ranges for VaR and ES

Ranges are the 5th and 95th percentiles of the estimate recomputed many times. Each model's uncertainty is generated in the way that fits how the model works:

| Models | How the estimate is recomputed | Count |
| --- | --- | --- |
| Historical, Normal, Student-t, Cornish-Fisher | Stationary block bootstrap of the returns (Politis and Romano, 1994). The model is re-estimated on each resample (Student-t by maximum likelihood). | 500 (Student-t: 200) |
| EWMA, FHS | Tomorrow's EWMA volatility σ̂ is held fixed. The volatility-standardised returns `z_t = r_t/σ_t` are block-bootstrapped. FHS takes the empirical tail of the resampled `z`. EWMA multiplies its figure by `rms(z*)/rms(z)`, the sampling error in the residual scale. | 500 |
| GARCH(1,1)-t | Parameter uncertainty only. 1,000 draws of (μ, ω, α, β, ν) come from a normal with the robust asymptotic covariance of the fit, truncated to valid parameters (ω > 0, α, β ≥ 0, α + β < 1, ν > 2.05). Each draw refilters the returns for tomorrow's σ and recomputes the closed-form t VaR and ES. | 1,000 |
| Monte Carlo | 200 of the GARCH parameter draws, each simulated with 2,000 one-day shocks, so the range includes simulation noise. | 200 |

- **Why EWMA, FHS and GARCH are not bootstrapped directly:** resampling raw returns scrambles the volatility clustering these models forecast from, and a GARCH refit per resample would add about a minute.
- **Rejected draws:** the share of GARCH draws rejected is reported. Above 50% the range is flagged as unreliable. On real NSE stocks 17–34% were rejected, because α sits near its zero bound.
- **Block length:** the Politis and White (2004) automatic choice, with the Patton, Politis and White (2009) correction (`arch.bootstrap.optimal_block_length`), applied to the returns and to the squared returns. The longer of the two is used, so volatility clustering is kept, clipped to [1, n/10]. If the estimate fails, the block length is `n^(1/3)`.
- **Horizon:** ranges are computed for 1-day VaR and ES at the selected confidence level. Multi-day bounds are multiplied by each model's own ratio of t-day to 1-day figure, so they follow the model's scaling rule (§2).
- **Coverage check:** on 300 samples of 500 i.i.d. normal days at 95%, the ranges contained the true value:
  - 89% of the time for Historical VaR;
  - 88% for Normal VaR and 88% for Normal ES;
  - **only 84% for Historical ES.** The percentile bootstrap undercovers ES when the tail holds only about 25 observations. The Trust page says so.

### 8.2 Model risk

- **Passing models:** those with a VaR-test verdict of PASS and an ES test that did not FAIL.
- **ES range across passing models**, and the **model-risk add-on** = highest passing ES − the recommended model's ES.
- **When there's no clean recommendation:** if no model passes every backtest, the headline model fails one, so the add-on is not defined and the range is taken across all models.

**Lookback sensitivity.** Every model's ES (except Monte Carlo) on the last 252, 504 and 1,260 trading days and on the full history. These are slices of the long price history already downloaded for the crisis replay.
- A window longer than the history is reported as not available.
- A window containing reversing spikes (§7.5) is flagged.
- Cornish-Fisher outside its valid region is noted, as on the Market page, and blanked when its ES is not positive. On Reliance's full history the +337% data-error spike makes Cornish-Fisher's ES −₹56 lakh, which is meaningless.

**Ghost effect.** Historical VaR jumps when a large loss enters the window, and again when it leaves n days later, although nothing new happened.
- **Ahead:** the tail losses (returns ≤ −VaR) among the oldest 21 days of today's window, and Historical VaR on the window without those 21 days.
- **Past:** the rolling Historical VaR over the full history, with today's window length. A day-to-day change above **5%** (an assumption) is attributed by these rules:
  - **exit**, if the return leaving the window was in the previous window's tail and the new return was not;
  - **entry**, in the opposite case;
  - **both**, or **reordering**, otherwise.
- **Why 5%, not 10%:** at 95% with a 499-day window, no day-to-day move in Reliance's 30-year history exceeded 11%, so a 10% threshold found almost nothing.

### 8.3 Grade A–D

Each rule deducts 0, 1 or 2 points. A missing input deducts 1 point. All thresholds are assumptions, chosen so that a well-backtested model on clean data with two or more years of history earns A or B.

| Rule | 0 | 1 | 2 |
| --- | --- | --- | --- |
| ES 90% range width ÷ ES | ≤ 20% | ≤ 40% | more |
| ES spread across passing models ÷ recommended ES (§8.2) | ≤ 15% | ≤ 30% | more |
| Backtest of this model | pass | low power or not tested | fail (VaR tests or ES test) |
| Lowest data-quality score among holdings (§7.5) | ≥ 90 | ≥ 75 | lower |
| Daily returns | ≥ 500 | ≥ 250 | fewer |
| Share of inputs that are assumptions | ≤ 25% | ≤ 50% | more |

- **Bands:** a total of 0–1 gives **A**, 2–3 **B**, 4–5 **C**, 6+ **D**.
- **Caps:** a model that fails its backtest is graded at best C, and a data-quality score below 50 gives D.
- **Inputs that count as assumptions:** EWMA's and FHS's λ = 0.94 (fixed, not estimated), and √t scaling for Historical and FHS at horizons over one day.
- **Headline grade:** the headline model's (the recommended model, or the best-scoring fallback of §3.7; Historical if there is too little data).
- **Recommendation and grade agree:** the recommendation (§3.7) and the backtest rule here use the same tests, so a model called "recommended" never has a failed backtest. Until the review fixes of October 2026 the recommendation used the VaR tests only, and HDFC Bank's EWMA was recommended while failing the ES test (grade D).

## 9. Liquidity risk (`liquidity.py`)

Positions are the share counts from §7.1. Volume is NSE + BSE where Yahoo has both (§7.2). Every parameter below that is not estimated from data is an **assumption**: it can be changed in the sidebar's "Liquidity assumptions" and is listed in the trust grade.

### 9.1 Trading capacity

- **Averages:** ADV and average traded value (volume × close) over 20 and 60 days.
- **Days to liquidate** a holding = shares ÷ (participation rate × 60-day ADV). The participation rate defaults to **20%**, an assumption: about the most one seller can take of a day's volume without dominating it.
- **Share sellable in h days** = Σ value × min(1, h ÷ days to liquidate) ÷ total value, with every holding sold at its own pace.

### 9.2 SEBI/AMFI-style stress test

- **Method:** days to sell 25% and 50% of the portfolio pro rata, at **10%** of 3-month (63-day) average volume, with the least liquid **20%** of the portfolio excluded. These follow the convention of the liquidity stress tests Indian equity funds disclose.
- **Exclusion (our reading):** holdings are ranked by days to liquidate the whole position and removed, least liquid first, until 20% of the value is gone. The holding on the boundary is removed only in part, and its remaining part is sold pro rata. Days = the slowest remaining holding.
- **Figures shown:** results are shown with and without the exclusion.
- **Validation:** the page can run the same test on a fund's uploaded monthly portfolio and compare it with the AMC's published figure. This is not an official calculation.

### 9.3 Stressed volume

- **Crisis ratio:** for each window in `stress_scenarios.csv`, average volume in the window ÷ average volume over the 120 trading days before it, on the primary listing's full history. Comparing with the months just before cancels decades of volume growth and old splits.
- **Which windows count:** a window needs at least 10 days of volume inside it and 60 before it.
- **Measured factor:** the median ratio.
- **Applied factor:** capped at **1**, an assumption. Large caps usually trade *more* in a sell-off (median ratios of 1.08–1.32× for Reliance, HDFC Bank, TCS and Asian Paints on 2 Oct 2026), but that volume belongs to other sellers too, so a crisis is never taken to make selling easier. Britannia's 0.84× is applied as measured.
- **Use:** stressed days to liquidate and market impact both use ADV × the applied factor.

### 9.4 Spread, impact and liquidity-adjusted VaR

- **Spread:** the Corwin and Schultz (2012) high-low estimator, with the paper's adjustment for overnight gaps.
  - From two consecutive days: β = Σ ln(Hₜ/Lₜ)², γ = ln(max H / min L)² over both days, α = (√(2β) − √β)/(3 − 2√2) − √(γ/(3 − 2√2)), S = 2(e^α − 1)/(1 + e^α).
  - The two-day estimates are averaged within each month, keeping negative values, and only the monthly average is floored at 0. Flooring each two-day estimate first, as is common, biases the spread upwards: on simulated prices with no spread it gave 0.39%, against 0.12% for monthly flooring. With a true spread of 1%, the estimator gave 0.97–1.07%.
  - Even so, it still overstates spreads for the most liquid stocks: Reliance's estimate is 0.06% against an actual NSE quoted spread of a few hundredths of a percent.
  - You can enter a known spread instead.
- **Spread cost** (Bangia et al., 1999) = ½ × value × (mean + k × standard deviation of the last 12 monthly spreads). k = **3** is an assumption; Bangia et al. chose the multiplier from the spread distribution's tails.
- **Market impact** (square-root law; Almgren et al., 2005; Tóth et al., 2011) = Y × daily σ × √(shares ÷ stressed daily volume) × value, capped at the whole value. Y = **1** is an assumption; empirical estimates are of order one. σ is the standard deviation of the last 60 daily returns.
- **Liquidity-adjusted VaR waterfall:** the headline VaR (the recommended model, §8) + spread cost + impact + circuit-lock add-on.
  - The add-on is max(0, total circuit-lock loss − VaR): the loss beyond VaR if every banded holding is frozen for its exit-freeze scenario (§9.6).
  - It is a stress add-on, not a probability-based quantity, so the total is labelled as such.

### 9.5 Amihud illiquidity

Amihud (2002): the 60-day rolling mean of |daily return| ÷ daily traded value, in basis points of price move per ₹1 crore traded. Each value uses data up to its own date only.

### 9.6 Circuit-lock risk (India)

- **Band:** from the official price-band file (§7.4) when loaded. Otherwise it is **inferred**, and labelled so: the band (2/5/10/20%) hit most often, counting days that close within 0.1% of the low with a return of −band ± 0.1 percentage points, or at the high with +band. At least 3 hits are needed; otherwise "no fixed band found" (F&O stocks have none).
- **Past lock-downs:** lower-circuit days and the longest consecutive run, in the lookback window.
- **Exit freeze:** N consecutive lower circuits during which the holding cannot be sold. By default N is the longest past run, but at least **3** (an assumption). The loss is 1 − (1 − band)^N of the holding's value.

### 9.7 Ranges and grades

- **Days to liquidate 50% and share sellable in 5 days:** the 5th–95th percentiles of the same figure recomputed with each day's rolling ADV over the past year. Today's figure can lie outside the range when recent volume is unusual.
- **Liquidity-adjusted VaR:** the VaR range, plus the 5th–95th percentiles of the spread cost (bootstrapping the monthly spreads), plus the impact over the past year's volume. Adding the bounds assumes they move together, which errs towards a wider range.
- **Circuit-lock loss:** a scenario with no range, so that rule is skipped.
- **Grades:** §8.3's rules, skipping model dispersion and the backtest, which don't apply here. Sample length is the number of days with volume data. Shares of inputs that are assumptions:
  - share sellable: participation rate, 1 of 2;
  - LVaR: k and Y, 2 of 5;
  - circuit-lock: an inferred band and a freeze length set by the floor or your override, up to 2 of 2;
  - AMFI test: 0, since its parameters are a disclosure convention.

## 10. Credit risk (`credit.py`)

Inputs are the annual statements from §7.3 (Yahoo Finance, or your CSV overrides). Only statements public by the analysis date are used: a fiscal year is public on its filing date if known, otherwise at fiscal year end + 60 days. A year with a late-filed figure becomes public only once all its figures are. Settings are under "Credit assumptions" in the sidebar, and red-flag thresholds are in `config/credit_thresholds.json`.

### 10.1 Merton distance to default (Merton, 1974; KMV conventions)

- **Inputs:**
  - E = latest close × shares outstanding;
  - σ_E = equity volatility;
  - D = default point = short-term debt + **0.5** × long-term debt (the KMV convention; Crosbie and Bohn, 2003);
  - T = **1** year;
  - r = the sidebar risk-free rate.

  D, T and r are assumptions.
- **Missing debt:** if either debt line is missing, Merton is not computed.
- **Solving for V and σ_V:** solve E = V·N(d₁) − D·e^(−rT)·N(d₂) and σ_E·E = N(d₁)·σ_V·V. The solver works in logs so both stay positive, with a residual tolerance of 10⁻⁸.
- **Outputs:** DD = [ln(V/D) + (r − σ_V²/2)T] / (σ_V√T), and **PD = N(−DD)**. This is a model-implied, *risk-neutral* probability: the assets drift at r, not at their real expected return, so it usually overstates real-world default frequencies. It is not an agency PD.
- **Three equity volatilities, with PD shown under each:**
  - last year's daily returns, annualised;
  - EWMA today, annualised;
  - GARCH(1,1)-t's average variance over the next 252 days, from its term structure (§1.7), so it matches a one-year horizon.

  The sidebar picks the headline one. The spread across the three is the PD's range ("range across equity-volatility inputs", not a confidence interval).
- **Iterative KMV cross-check** (Vassalou and Xing, 2004):
  - start from σ_V = σ_E·E/(E + D);
  - back out each day's V from that day's equity value over the last year;
  - re-estimate σ_V from the daily log changes in V;
  - repeat until σ_V changes by less than 0.0001.

  On simulated data it matched the two-equation DD to within 0.15.
- **DD over time:** at each month end in the lookback window, using only the balance sheet public by then:
  - E = that day's close × that balance sheet's shares in issue;
  - σ_E = the last year of daily returns.

  Closes are dividend-adjusted, which slightly understates past market values.

### 10.2 Altman Z-scores

| Model | Formula | Zones |
| --- | --- | --- |
| Z (Altman, 1968; listed manufacturers) | 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5, X4 = market value of equity / total liabilities | > 2.99 safe, 1.81–2.99 grey, < 1.81 distress |
| Z'' (Altman, 2000; Altman et al., 2017; non-manufacturers and emerging markets) | 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4, X4 = book equity / total liabilities | > 2.60 safe, 1.10–2.60 grey, < 1.10 distress |

- **Ratios:** X1 = working capital / total assets (current assets − current liabilities, or the reported working-capital line); X2 = retained earnings / TA; X3 = EBIT / TA; X5 = sales / TA.
- **Z'' variant:** the version used is the one without the +3.25 constant that the emerging-market bond score adds; its zones are the ones above.
- **Sources:** coefficients and cut-offs follow the cited papers. They were checked on 2 Oct 2026 against a secondary reproduction (Wikipedia, citing Altman, 1968, and Altman et al., 2017), because Altman's own copies on the NYU site returned server errors.
- **Which is primary:** Z'' for Indian firms and US non-manufacturers; Z for US firms in the manufacturing sectors listed in the config. Both are shown whenever the inputs exist.
- **Never from partial inputs:** if any input is missing, the score is "not available" and the missing items are named.

### 10.3 Credit ratios and red flags

Each fiscal year's ratios, trended over 4–5 years:

| Ratio | Formula | Red flag on the latest year (assumption) |
| --- | --- | --- |
| Debt / equity | total debt / equity | > 2, or equity ≤ 0 |
| Net debt / EBITDA | (total debt − cash) / EBITDA | > 4, or EBITDA ≤ 0 |
| Interest cover | EBIT / interest expense | < 1.5 |
| Current ratio | current assets / current liabilities | < 1 |
| Quick ratio | (current assets − inventory) / current liabilities | < 0.7 |
| Cash from operations / EBITDA | | < 0.5 |
| Negative cash from operations | consecutive latest years | 2 or more |

The thresholds are common credit-analysis rules of thumb, not regulatory limits.

### 10.4 Ratings and financial companies

- **Ratings:** the latest long-term rating action public by the analysis date (§7.4) gives the current rating, last action and direction. Short-term ratings (A1+ … A4) are on a different scale and are no longer misread as an "A" grade.
- **Agency default rates.** The latest long-term rating's category (AA+, AA and AA− → AA) is mapped to the agency's published average 1-year default rate, in `data/credit/rating_default_rates.csv`:
  - **CRISIL:** Annual Default and Ratings Transition Study FY2025, Table 1 (page 10): corporate issuers, long-term ratings, monthly static pools, FY2015–FY2025. AAA 0.00%, AA 0.05%, A 0.07%, BBB 0.46%, BB 2.86%, B 8.40%, C 24.98%.
  - **ICRA:** FY2025 Rating Transition and Default Study, page 36: all ratings excluding structured finance, 10-year average CDR-1. AAA 0.1%, AA 0.1%, A 0.2%, BBB 1.0%, BB 3.6%, B 6.0%, C 24.9% (printed to one decimal place).
  - For other agencies (CARE, India Ratings, Acuité and others), CRISIL's table is used and flagged. The scales are comparable but not identical. 'D' (in default) is 100%.
  - These are **real-world, historical** frequencies. Merton's PD is **risk-neutral and model-implied**. The Credit page shows both, side by side.
- **DD percentile (since October 2026).** For strong firms Merton's PD is around 10⁻³⁰, which means nothing on its own. So the Credit page leads with the distance to default and its percentile in a reference universe: `data/credit/dd_universe.csv`, built by `scripts/build_dd_universe.py` with the app's own code path and default inputs, from the covered stocks of §7.4. Percentile = share of the universe with a lower DD (ties count half).
- **Financial companies:** banks, NBFCs and insurers are identified by Yahoo sector "Financial Services" or industry keywords. Merton, Altman and leverage ratios are skipped for them: deposits and policyholder liabilities are their business, not debt in the Merton sense.
- **Banks and NBFCs with a metrics file** are assessed against RBI's PCA thresholds (§10.6). Without a file, a fallback panel lets you type in GNPA, NNPA, CAR, CASA and NIM; CAR is compared with the RBI minimum of 11.5% (9% CRAR plus the 2.5% capital conservation buffer).

### 10.5 Portfolio view, ranges and grades

- **Weighted PD** = Σ PDᵢ·valueᵢ / Σ valueᵢ over the modelled holdings. The unmodelled share (financials, missing data) is stated.
- **Credit-implied expected loss** = Σ PDᵢ × valueᵢ with **loss given default = 100%**: equity holders are last in line and usually recover nothing.
- **Ranges** span the three equity volatilities.
- **Grade adjustments for PD and expected loss:**
  - **Range width** is measured on the distance-to-default scale, DD = −Φ⁻¹(PD). For strong firms PDs are around 10⁻³⁰, where a relative width in PD means nothing.
  - **Sample length** is years of statements: ≥ 4 for 0 points, ≥ 3 for 1.
  - **Assumptions:** 3 of 6 Merton inputs (default-point weight, T, r).
  - **Extra deductions:** a balance sheet older than 15 months, or any unmodelled value.
- **Altman score:** no range, so that rule is skipped.

### 10.6 Banks and NBFCs: RBI Prompt Corrective Action (`banks.py`, `config/pca_thresholds.json`)

Merton, Altman and leverage ratios do not apply to lenders, whose deposits and borrowings are the business. Banks and NBFCs are instead measured against RBI's own supervisory triggers.

- **Metrics** (`data/banks/<SYMBOL>.csv`, long format, one row per metric and period, with the filing date and source):
  - **asset quality:** gross NPA %, net NPA %, provision coverage (excluding written-off assets, as printed);
  - **capital:** CRAR, CET1, Tier-1 capital ratio, Tier-1 leverage ratio;
  - **funding and liquidity:** CASA %, credit-deposit ratio, LCR;
  - **profitability:** NIM, ROA.
  - **Sources:**
    - NPA ratios and ROA come from NSE results XBRL (quarterly and annual; `scripts/fetch_bank_results.py`).
    - The rest is transcribed from annual reports, each value with its PDF page.
    - The credit-deposit ratio is derived from the printed balance-sheet advances and deposits (labelled).
    - Slippage is not printed in the reports used, so it is missing rather than estimated.
- **Point in time:** a value is used only once its filing date has passed. Results XBRL is dated by NSE's broadcast date, annual reports by their NSE filing date. When the same period is filed twice, the later filing wins.
- **XBRL checks** (on the filing's own figures):
  - **NPA ratios:** gross NPA amount ÷ gross NPA ratio = implied gross advances, and the quarter's interest on advances × 4 ÷ that = implied yield. Outside 2–25%, the ratios are dropped. Several Yes Bank filings from 2020 onwards entered the ratios 100× too small and rounded them to four decimals, so they cannot be recovered.
  - **ROA:** it must agree in sign, and within a factor of two, with the filing's profit ÷ total assets. Yes Bank's FY2020 filing says +0.05% in a year with a ₹16,418 crore loss (−6.4%), so it is dropped.
  - Annual-report net NPA matches the XBRL to 0.00 pp in every period where both exist (HDFC Bank FY2022–24, Yes Bank FY2019).
- **PCA bands.**
  - **Banks:** RBI/2021-22/118, 2 Nov 2021, effective 1 Jan 2022; scheduled commercial banks excluding small finance, payment and regional rural banks.

    | Indicator | Risk threshold 1 | Risk threshold 2 | Risk threshold 3 |
    | --- | --- | --- | --- |
    | CRAR (9% + 2.5% CCB = 11.5%) | < 11.5% | < 9.0% | < 7.5% |
    | CET1 (pre-specified trigger 6.125% + 2.5% CCB = 8.625%) | < 8.625% | < 7.0% | < 5.5% |
    | Net NPA ratio | ≥ 6% | ≥ 9% | ≥ 12% |
    | Tier-1 leverage ratio (minimum 4% for D-SIBs, 3.5% for others; RBI/2018-19/225, 28 Jun 2019) | up to 50 bps below | 50–100 bps below | more than 100 bps below |

  - **NBFCs:** RBI/2021-22/139, 14 Dec 2021, effective 1 Oct 2022. It covers deposit-taking NBFCs and middle/upper/top-layer non-deposit-taking NBFCs, excluding housing finance companies, government companies and primary dealers.

    | Indicator | Risk threshold 1 | Risk threshold 2 | Risk threshold 3 |
    | --- | --- | --- | --- |
    | CRAR | < 15% | < 12% | < 9% |
    | Tier-1 | < 10% | < 8% | < 6% |
    | Net NPA (incl. NPIs) | > 6% | > 9% | > 12% |

  - DHFL was a housing finance company, outside this framework.
- **Distance to trigger** = headroom to risk threshold 1 in percentage points (negative = breached).
- **Early warnings** (assumptions, `banks.EARLY_WARNING`):
  - a capital indicator within 1 pp of its trigger;
  - net NPA within 1 pp of 6%;
  - gross NPA up 1 pp or more in a year;
  - ROA < 0;
  - LCR below the 100% minimum.
- **Event tier:** for a bank or NBFC the credit signal is its PCA band: risk threshold 2 or 3 → High; risk threshold 1 or any early warning → Elevated.
- **Limitation:** the PCA thresholds are those of the 2021–22 circulars. Applied to earlier dates (the Yes Bank case study), they are a benchmark, not the rules then in force. The circulars may have been amended since; check RBI.

### 10.7 Debt positions (`debt.py`, `config/debt_assumptions.json`)

- **Input:** bonds, NCDs or loans in the sidebar, each with face value, coupon, maturity, rating, seniority, spread over the risk-free rate, and optionally the issuer's ticker.
- **Pricing:** clean price with annual coupons on calendar coupon dates, at y = risk-free + spread; modified (= spread) duration and convexity analytically (checked against a numerical derivative).
- **Expected loss over one year** = market value × PD × LGD.
  - PD is the rating's published 1-year default rate (§10.4).
  - LGD by seniority: senior unsecured 45% and subordinated 75% (Basel II foundation IRB, paras 287–288); secured 25% (an assumption). All are editable.
- **Stress (Integrated Stress page, credit link):** loss = full repricing at y + Δs, plus the one-year expected loss. The duration–convexity approximation is accurate to third order in Δs (tested against full repricing: within 0.1% at 100 bp).
  - **Issuer held as equity with Merton inputs:** Δs = s(PD stressed) − s(PD today), with s = −ln(1 − PD·LGD)/T and the risk-neutral Merton PD re-solved at the issuer's linked-stress price. This is the right measure for a spread, and it carries the equity feedback (pledge selling, circuits) into the bond: a cross-pillar interaction.
  - **Otherwise:** an assumed widening by rating category for a 50% market fall (AAA 150 bp … C 2,500 bp), scaled by the scenario's fall and capped at 1.5×.
- **Not modelled:** interest-rate risk on the bonds (the risk-free curve is held fixed), floating-rate and callable structures, and accrued interest. The VaR/ES pillars remain equity-only.

## 11. Concentration and factor risk (`concentration.py`, `factor_data.py`)

### 11.1 Factor data

- **India:** the IIM Ahmedabad Indian Fama-French-Momentum library (Agarwalla, Jacob and Varma, 2013). Daily market excess return (MF), SMB, HML, WML (momentum) and the risk-free rate, survivorship-bias adjusted.
- **US:** the Kenneth R. French library. Daily Fama-French 3 factors plus the momentum factor.
- **Refreshing:** `scripts/refresh_factor_data.py --iima-release YYYY-MM` downloads both (each site's robots.txt allows it) and writes `data/factors/<region>_daily.csv` in decimals, plus `metadata.json` (source URL, download date, coverage, citation).
  - A failed download writes nothing for that region and prints manual steps.
  - French's −99.99 / −999 codes become missing values; no day is filled in.
- **Latest release:** the 2 Oct 2026 download covers India to **31 Dec 2025** and the US to 31 Aug 2026.
- **Which file:** a portfolio uses the Indian file if every holding is listed on NSE/BSE, and the US file if none is. A mix has no single factor set without FX conversion.

### 11.2 Factor regressions

- **Model:** rᵢ − r_f = αᵢ + βᵢ'f + εᵢ on Mkt−RF, SMB, HML and momentum, over the days the stock and the factor file both cover.
- **Standard errors:** Newey and West (1987), Bartlett weights, with ⌊4(n/100)^(2/9)⌋ lags (Newey and West, 1994).
- **Outputs:** betas with t-statistics, annualised α (× 252), R², and factor variance β'Σ_fβ against specific variance σ²(ε).
- **Rolling betas:** 252-day windows, re-estimated every 21 days, each using only data up to its end date.
- **Fallback:** with fewer than **120** overlapping days for any holding, *every* holding falls back to the single-index model on the benchmark (Nifty 50 or S&P 500). The portfolio split needs one model for all holdings, and the fallback costs a grade point.

### 11.3 Portfolio factor risk

- **Portfolio betas:** b = Bᵀw.
- **Variance:** σ² = bᵀΣ_f b + Σ wᵢ²σ²(εᵢ), with Σ_f the factors' covariance over the regression days.
- **Euler split:** factor k contributes b_k(Σ_f b)_k, and specific risk contributes Σ wᵢ²σ²(εᵢ); the parts sum to σ² exactly. A negative part is a hedge.
- **Parametric VaR:** z·σ·investment (zero mean), split in the same proportions.
- **Specific risks are treated as uncorrelated**, the standard factor-model assumption, so σ² can differ slightly from the portfolio's sample variance.

### 11.4 Concentration

- **Holdings:** HHI = Σwᵢ², and effective number of holdings = 1/HHI.
- **Risk shares:** Euler shares of portfolio variance by holding are wᵢ(Σw)ᵢ / wᵀΣw, using the lookback window's sample covariance. Summed by sector (Yahoo's classification), they give sector risk shares.
- **Principal components:** eigenvalues λₖ and eigenvectors eₖ of Σ. Reported:
  - the first component's share of total variance, λ₁/Σλ;
  - Meucci's (2009) **effective number of bets**: the portfolio's variance splits across the uncorrelated components as pₖ = λₖ(eₖᵀw)² / wᵀΣw, and ENB = exp(−Σ pₖ ln pₖ). It is 1 for one bet and N for N independent equal-risk bets.

  For a long-only equity portfolio the market component usually dominates: the default five-stock portfolio had 1.24 effective bets.

### 11.5 Crisis correlation

- **Three samples:** correlations on the full common long history of the holdings, on the days inside the windows of `stress_scenarios.csv` (pooled), and on the market's worst 10% of days. Each needs at least 30 days.
- **ES under each:** zero-mean normal ES uses that sample's correlations and the *full-sample* volatilities, so only correlation changes.
- **Benefit kept** = (Σ standalone ES − portfolio ES) ÷ the same quantity under full-sample correlation.
- **Selection bias:** choosing days by the size of the market's fall narrows the market's range inside the sample, which *lowers* measured correlation (Boyer, Gibson and Loretan, 1999; Forbes and Rigobon, 2002). For the default portfolio, worst-day correlation was 0.19 against 0.18 overall, while the crisis windows showed 0.37. The headline is therefore the lower of the two crisis measures, and the spread between them is its range.

### 11.6 Ranges and grades

- **Effective bets and factor share:** 90% ranges by stationary block bootstrap (§8.1), with 200 resamples, recomputing the PCA and re-running every regression.
- **Benefit kept:** its range is "across crisis definitions", not a confidence interval.
- **First-component share:** shown without a range.
- **Grade deductions:**
  - factor data ending more than 92 days before the latest price;
  - falling back to the single-index model.
- **Assumptions:** the choice of factor model (1 of 3 inputs); the crisis windows and the normal ES (2 of 4).

## 12. Event and governance risk (`events.py`, `config/event_rules.json`)

### 12.1 Signals, point in time

Each holding gets eight signals. Every one uses only data public by the analysis date (§7.4 public dates). A signal whose data are not loaded is **not available**: it never counts as "no risk".

| Signal | Rule input | Source |
| --- | --- | --- |
| Promoter pledge | pledged % of promoter holding in the latest public quarter, and its change from the quarter about a year earlier | `pledges` file |
| ASM / GSM | the most severe measure in force (GSM > long-term ASM > short-term ASM > ESM; entered on or before the date, not yet exited) | `surveillance` file |
| F&O ban | a ban trade date within the last 30 calendar days | `fo_ban` file |
| Rating actions | downgrades and negative watches in the last 12 months; any downgrade below BBB− | `ratings` file |
| Auditor events | event types in the last 24 months | `auditor_events` file |
| Merton DD trend | DD at the latest month end, and its fall over 6 calendar months | credit pillar (§10.1) |
| Circuit history | lower-circuit days and the longest run in the lookback | liquidity pillar (§9.6) |
| Group tag | how many holdings share the tag you enter | sidebar |

Rating grades are read from the agencies' text (e.g. "CRISIL AA+", "[ICRA]A (Stable)", "CARE BBB- (CE)") on the scale AAA … D.

### 12.2 Tier rules (assumptions)

**High** if any of the following, otherwise **Elevated** if any of its rules fire, otherwise **Low**:

| | High | Elevated |
| --- | --- | --- |
| Pledge | ≥ 50% of promoter holding | ≥ 20%, or up ≥ 10 pp in 4 quarters |
| Surveillance | GSM (any stage), or long-term ASM stage ≥ III | any ASM stage |
| F&O ban | | within 30 days |
| Ratings (12 months) | ≥ 2 downgrades, or a downgrade below BBB− | a downgrade, or a negative watch |
| Auditor (24 months) | resignation, adverse opinion, disclaimer | qualified opinion |
| Merton DD | < 1.5 | < 3, or down ≥ 30% in 6 months |
| Circuit | | ≥ 3 lower-circuit days |
| Group | | ≥ 2 holdings with the same tag |

- **Reporting:** every rule that fired is listed, and the tier states its basis ("based on n of 8 signals").
- **Thresholds:** in `config/event_rules.json`. They are judgement calls, to be checked against the Phase 7 case studies.

### 12.3 Pledge margin calls

- **Loan:** sized at today's price at initial cover C₀ (pledged value ÷ loan; default **2.0×**, i.e. 50% loan-to-value). Lenders can invoke the pledge when cover falls to C_t (default **1.5×**). Both are assumptions.
- **Trigger:** a price fall of 1 − C_t/C₀ (25% at the defaults).
- **Selling** at the trigger price P_t is shown two ways:
  - all pledged shares, in days of 60-day ADV;
  - just enough to restore C₀: x = L(C₀ − C_t) / (P_t(C₀ − 1)).
- **Pledged quantity:** from the pledge file, or pledged % of total × shares outstanding.

### 12.4 Jump overlay on ES

- **The model:** each holding can jump on a given day by J with probability p, both set by its tier. A daily p compounds: the chance of at least one jump in a year of 252 trading days is 1 − (1 − p)^252, shown next to every daily p in the app.
- **Defaults, recalibrated in October 2026** on every NSE stock from 2016 to 2024 (`universe.jump_base_rates`; [case_studies.md](case_studies.md#jump-defaults-recalibrated-on-nse-wide-base-rates-review-m2)):

  | Tier | p a day (a year), before | p a day, now | J, now | Basis of the new value |
  | --- | --- | --- | --- | --- |
  | Low | 0 | 0 | – | – |
  | Elevated | 0.1% (22%) | **0** | −17% | 3+ lower-circuit days: excess rate of ≥ 10% falls −0.087% a day (90% range −0.131% to −0.050%); 20,574 falls |
  | High | 0.5% (72%) | **0** | −34% | series BE/BZ: excess rate of ≥ 20% falls −0.001% a day (−0.034% to +0.030%); 2,080 falls |

  - **A jump** is a one-day fall of at least |J| on a day the Nifty fell less than 5%: market-wide crashes belong to the base distribution.
  - **The excess** is the rate over the 63 trading days after each month-end where the proxy holds, minus the stock's own rate over the window the base model is fitted on. The base model already carries the stock's own history of falls, so only an excess belongs in the overlay. A probability cannot be negative, so the default is the excess floored at 0.
  - **The 90% range** comes from resampling stocks, because one stock's monthly windows overlap.
  - **J** is the mean size of the qualifying falls.
  - **The old values** were checked only on crash-selected case dates, which is circular (review M2).
  - **Limit:** the proxies are price-based, the only tier signals measurable point in time for every NSE stock. Disclosure-driven tiers (pledges, downgrades, auditor events) for stocks with calm prices could carry jump risk their own history does not show. That base rate is not measurable from public archives (NSE's pledge history starts in September 2021), so the default is zero rather than a guess, and the sensitivity table shows what any assumed p and J would add.
- **ES sensitivity table** (Events page): event-adjusted ES for p ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 1}% a day × J ∈ {−10, −20, −30, −50}%. It applies to the holdings in Elevated or High, or to the largest holding (labelled hypothetical) if none is. Each cell is the same closed form below.
- **Base distribution:** the one-day return distribution of the stock or portfolio is loc + scale·T_ν. This is the fitted GARCH(1,1)-t (σ̂·√((ν − 2)/ν) as the scale), or the Student-t fit if GARCH did not converge.
- **Portfolio:** a jump in holding i shifts the portfolio return by wᵢJᵢ. Every combination of jumps is enumerated exactly up to 12 jumping holdings; above that, up to two jumps.
- **Calculation:**
  - VaR solves Σ πₖ F_ν((x − loc − sₖ)/scale) = α;
  - ES uses the Student-t partial expectation E[T·1{T ≤ z}] = −(ν + z²)/(ν − 1)·f_ν(z);
  - there is no simulation, so no noise. On a 2-million-draw simulation the result agreed within 0.5%.
- **Shares of the gap:** each holding's jump switched on alone, scaled so the shares add to the total gap.
- **How much the assumptions matter:** on Reliance's real one-day distribution (2 Oct 2026), the old High-tier assumption (0.5% a day, −20%) would lift 95% ES from ₹29,295 to ₹47,136, and 99% ES from ₹42,872 to ₹1,24,904. With the calibrated defaults it adds nothing.

### 12.5 Ranges and grades

- **Event-adjusted ES and its gap:** the range runs from ½× to 2× every jump probability ("range across jump assumptions").
- **Assumptions counted:** only the jump settings of tiers that some holding actually falls in.
- **Missing data:** each disclosure dataset not loaded costs a point, capped at 2.
- **Value in Elevated/High and the number of High holdings:** no range.

## 13. Integrated stress (`integration.py`)

### 13.1 Linked stress engine

Each scenario hits every pillar together, holding by holding.

- **Scenarios:**
  - **Historical:** every window in `stress_scenarios.csv` that the benchmark covers. The holding's return is its actual return between the market's peak and trough (replay), or downside beta × the market's drawdown if it had no prices then. Volatility is the holding's own daily volatility inside the window, or today's volatility × the market's crisis-to-normal volatility ratio.
  - **Custom:** market −10%, −20% and −30%. Downside beta sets each holding's return, and volatility is today's × the median crisis-to-normal ratio of the market across the windows.
- **A price fixed point (since the review fixes of October 2026).** Before, the links were added as separate cost lines that never moved the price, so nothing could feed back. Now, for each holding, with x the price as a share of today's price P₀, x_m = 1 + r the market shock, Q our shares and S the pledged shares:
  1. **Events:** once the cover S·P/L falls to the margin-call cover C_t (loan L = S·P₀/C₀, sized today), lenders sell enough to restore C₀: x_f = (C₀·L − S·P)/(P·(C₀ − 1)), capped at S. At the trigger price this is §12.3's shares to restore; at a deeper fall it is more. *(The old engine sized the loan at the stressed price and sold only the trigger-price amount, which understated deep falls.)*
  2. **Liquidity, price impact:** all selling, x_f + Q (our own exit when liquidity is on), moves the price by the permanent part of square-root impact: x_perm = x_m·(1 − θ·I), I = min(1, Y·σ·√(sold/ADV)), at the crisis's volume (never above normal; §9.3) and volatility.
  3. **Liquidity, circuits:** if the fall 1 − x_perm reaches the band, the stock locks for its exit-freeze scenario: x = x_perm·(1 − band)^N_extra. In a **historical replay**, lower-circuit days already inside the replayed path (returns at or below −band + tolerance, from the return alone since the path has no high/low) are already in the market loss, so N_extra = max(0, N − days already locked).
  4. Repeat from 1 at the new price until it moves less than 0.1% in a round (at most 20 rounds; the number of rounds and whether it settled are reported).
  - **Convergence:** each round can only lower x (more forced selling, a lock that stays on), x ≥ 0, and forced selling is capped at S, so the sequence converges.
  - **Our loss** = Q·P₀ − proceeds, with proceeds = Q·P₀·x\*·(1 − (1 − θ)·I\* − spread rate) when we exit (Bangia spread on the final value), or Q·P₀·x\* with the liquidity link off. The permanent part of impact is in x\*; the temporary part is our execution cost. With liquidity alone this equals the old market + spread + impact up to second order (tested), and with every link off it is exactly the plain market loss (tested).
  - **θ = 2/3, the permanent share of impact (assumption, editable).** After a metaorder ends, its price impact decays to about two-thirds of its peak: Farmer, Gerig, Lillo and Waelbroeck (2013, *Quantitative Finance*) derive 2/3 from a "fair pricing" condition, and Bershova and Rakhlin (2013, *Quantitative Finance*) measure a similar level in institutional orders. θ = 0 turns the price feedback off; θ = 1 makes all impact permanent.
  - **Limits:** feedback runs within each stock. Contagion between holdings, other holders' fire sales and fund redemptions are not modelled.
- **Credit (C2 of the review):** for an equity holder the Merton PD is implied from the equity price, which the scenario already marks down, so adding (PD_stressed − PD) × value counted the default risk twice. Credit now adds **no loss** to equity. Merton is re-solved at equity × x\* with the scenario volatility, and the stressed DD and PD are reported as a signal. If the stressed DD falls below a threshold (1.5, an assumption, editable), a separate **jump-to-default** scenario shows the loss if the equity goes to a 10% or 0% recovery (an assumption), next to the risk-neutral, model-implied PD. It is never added to the linked total. A credit loss returns for bonds and loans (Phase D).
- **Crisis correlation:** this changes the next day's tail rather than the scenario loss itself, so it is reported alongside: crisis-window ES against normal ES (§11.5).
- **Interaction and the split.** v(S) is the loss with only the players in S switched on (market off means r = 0, with the crisis volume and volatility kept as part of liquidity).
  - **Interaction = v(all) − [v(market) + Σ over links (v(market + link) − v(market))]**: the cross effect only. It is zero when at most one link can act (tested).
  - **Exact Shapley values** over market, liquidity, credit and events (16 coalitions per holding) split v(all) so that the parts add up to it (tested against a hand calculation).
  - The old comparison with a "siloed sum" (the market loss plus each page's standalone headline today) mixed scenario and unconditional figures, so its difference was not an interaction. It is removed.

**What the live data show (2 Oct 2026).**
- **With the NSE snapshot loaded (3 Oct 2026), the links interact.** The 5-stock portfolio's worst scenario (Global Financial Crisis) shows +₹5,145: Asian Paints' pledge (5.0% of the company) is sold once its fall passes the trigger, over 4 rounds. Jaiprakash Power (17.5% of its shares pledged, official 20% band) in a −20% market loses ₹6.44 lakh against ₹2.17 lakh for the move alone, with an interaction of +₹43,432. For Afcons and Cohance the interaction is negative (−₹91,353 and −₹204,059 in the COVID replay): each link alone already drives the price through the band and towards the impact cap, so the links overlap.
- **Before the snapshot (Phase B results),** the interaction was zero for every preset, because only the liquidity link could act. The old engine's −₹1,206 (portfolio) and +₹1,27,008 (Jaiprakash Power) were not interactions (§13.1).
- **Heavy pledges push the square-root law far outside its measured range.** Lenders selling 17–54% of a company is hundreds of days' volume, and the impact cap of 100% binds. Read those rows as severe scenarios.
- **Feedback from your own exit is small for large caps:** it lowers prices by 0.01–0.02% at ₹10 lakh and 0.2–0.4% at ₹50 crore (1–2 rounds).
- **Under the inferred 5% band (Phase B),** Jaiprakash Power's −10% market loss was ₹2.38 lakh against ₹1.09 lakh for the move alone, the freeze being in the liquidity share. NSE's official file gives a 20% band, so a −10% move no longer breaches it (₹1.11 lakh now).
- **Phase B check with a hypothetical pledge** (before real pledges were loaded; 5% of shares, not data, under the inferred 5% band). In a −20% market, the lock takes the price through the 25% margin-call trigger, the lenders' sale pushes it 16.8% below the market move after 4 rounds, and the interaction is +₹28,513 (+₹59,668 at a 20% pledge).
- **The results are only as reliable as their inputs.** For Jaiprakash Power these are the inferred band, the 3-day floor and θ.
- **Replays (fix of October 2026).** Jaiprakash Power's historical replays already contain 1–29 lower-circuit days. Counting only the extra freeze days lowered its linked losses by 3% (Global Financial Crisis, 29 locked days, no extra) to 25% (US credit downgrade). Custom shocks were unchanged.
- **Credit (fix of October 2026).** Removing the double-counted credit add-on lowers Jaiprakash Power's linked losses by up to ₹30,522 (taper tantrum). Its stressed DD falls to 1.46 (taper tantrum) and 1.45 (COVID crash), below the 1.5 threshold, so a jump-to-default scenario is shown for those two.

### 13.2 Reverse stress test

- **Asset space:** the most plausible return vector that loses L over one month (Σ = 21 × the daily covariance) minimises xᵀΣ⁻¹x subject to wᵀx = −L. The closed form is **x\* = −L·Σw / (wᵀΣw)**, with squared Mahalanobis distance d² = L²/(wᵀΣw); it is tested against a numerical optimiser.
- **Crisis version:** the same using crisis-window correlations with full-sample volatilities.
- **Probability of the loss itself:** P(wᵀx ≤ −L), with σ_p = √(wᵀΣw) and d = L/σ_p (which is also the Mahalanobis distance of x\*):
  - normal: **Φ(−d)**;
  - multivariate Student-t with covariance Σ: the portfolio return is univariate t with scale σ_p·√((ν−2)/ν), so the probability is **T_ν(−d·√(ν/(ν−2)))**;
  - shown per month and as "about once every 1/(12p) years". Both are tested against two-million-draw simulations.
  - **Assumption:** ν is fitted to daily returns and applied to the one-month sum. Monthly sums are closer to normal than daily returns, so the Student-t figure is an upper-end estimate and the normal figure a lower-end one.
- **Share of outcomes at least this extreme in any direction:** P(D² ≥ d²), χ²ₖ under a normal and, under the t, (D²·ν/(ν−2))/k ~ F(k, ν) (d² is measured with the covariance, so it is rescaled to the t's dispersion matrix). This counts gains and every other direction too, so it is **not** the probability of the loss; it is shown only as a measure of how unusual the scenario is. Before October 2026 this share was presented as the probability of the loss, and the F version omitted the ν/(ν−2) rescaling.
- **Macro space:**
  - Portfolio daily returns are regressed on Nifty 50, Bank Nifty, Nifty IT, USD/INR and Brent (S&P 500, Nasdaq 100, the dollar index and Brent for US portfolios) over the last 504 common days.
  - The most plausible one-month macro move that loses L through those betas is y\* = −L·Σ_m b / (bᵀΣ_m b).
  - The **nearest historical analogue** is the past 21-day window, over the whole common history, whose compounded macro moves are closest to y\* in Mahalanobis distance. The portfolio's actual return over that window is reported.
  - A G-sec yield series is not included; Yahoo does not provide one.
- **Linked:** the asset-space scenario is also run through the linked engine.

## 14. Decisions (`decisions.py`, `memo.py`, `config/limits.json`)

### 14.1 Risk-change explanation (Shapley)

- **What is compared:** 1-day ES at the selected level, between today and a snapshot. The snapshot is either the same holdings N months ago (a window of the same length ending then) or an uploaded JSON snapshot saved earlier from this page.
- **ES as a function of five inputs:** ES = f(positions, volatility, correlation, window, model). The window's returns are standardised, whitened with their own correlation (Cholesky), re-correlated with the chosen correlation and rescaled by the chosen volatilities.
  - This reproduces the window exactly when its own moments are used (tested to 10⁻¹²).
  - "Window" therefore carries everything about the sample's shape (tails, clustering) that volatility and correlation do not.
- **Models:** Historical, Normal or Student-t.
- **Shapley attribution:** each input gets the average of its marginal effect over all 5! orderings, computed exactly from the 32 coalitions. The parts add up to the change exactly, and an input that did not change gets exactly zero (both tested).
- **Common tickers:** only tickers in both snapshots are compared.

### 14.2 Trades and hedge

- **ES by holding:** historical ES split into Euler (tail-conditional) contributions wᵢ·E[−rᵢ | portfolio ≤ its α-quantile]; marginal ES = contribution ÷ weight.
- **Candidate trades:** every sale of 5% of the portfolio into cash, and every 5% switch from one holding to another, re-evaluated on the same returns.
- **Shown:** the best sale and the two best switches, each with its effect on:
  - days to liquidate 50% (AMFI method);
  - the weighted Merton PD;
  - the share of value in Elevated or High event tiers.

  Selling into cash always cuts ES roughly in proportion, so switches are shown to reveal where diversification helps.
- **Hedge:** the short index-futures notional h ∈ [0, 2] × value that minimises the historical ES of r_p − h·r_m. This is measurement only: no basis, margin, roll costs or lot sizes.

### 14.3 Limits

`config/limits.json` (editable on the Decisions page for the session) holds example limits:

| Limit | Default |
| --- | --- |
| ES as % of value | 4% |
| Largest holding | 30% |
| Largest sector | 40% |
| Days to liquidate 50% | 5 |
| Weighted PD | 2% |
| High-tier holdings | 1 |
| Worst headline trust grade | C |

- **Utilisation** = measure ÷ limit. For the grade limit it is the worst grade's rank ÷ the limit's rank (A = 1 … D = 4).
- **Traffic lights:** green below 80%, amber from 80% to 100%, red above 100%.
- **Single stocks:** the largest-holding and largest-sector limits apply to portfolios only.

### 14.4 CRO dashboard and memo

- **Dashboard (Overview page):** one row per pillar (headline, range, grade, limit status), the worst linked stress, top risks and top actions.
- **Top risks:** ranked by explicit scores:

  | Situation | Score |
  | --- | --- |
  | Red limit | 100 + utilisation |
  | Worst linked loss | 60 + loss % |
  | High-tier holding | 55 |
  | Amber limit | 50 + utilisation/2 |
  | Grade D | 45 |
  | Holding with ES share ≥ 1.25 × its weight | 40 + excess |
  | Grade C | 25 |
  | Missing disclosure data | 20 |

- **Top actions**, in order: the best ES-cutting trades, the hedge if it cuts ES by more than 10%, red limits to fix, and data to load.
- **The one-page memo** (PDF via reportlab with a matplotlib chart, and Markdown) has these sections: Bottom line · Pillar summary · Integrated stress · What the numbers miss · Limits · Actions · Data sources and caveats. Its text is assembled from rules and the computed figures, with no language model.

## 15. Evidence: baselines and the wider test (`case_studies/engine.py`, `universe.py`)

### 15.1 Naive baselines (review M1)

Three rules from prices up to the as-of date only, fixed before the test:
- (a) close ÷ close 126 trading days earlier − 1 ≤ −30%;
- (b) today's 60-day standard deviation of daily returns at or above the 90th percentile of its own rolling 60-day values over the last 1,260 days (at least 500 needed);
- (c) close below the mean of the last 200 closes.

"Any baseline" is (a) or (b) or (c). On the 220 case and control dates each is tallied like a pillar (hit rate, false-positive rate) and given a lead time: for each case, the most months before the event at which it warned.

### 15.2 Wider test

- **Universe.** Every company share (ISIN INE…, series EQ/BE/BZ) in NSE's daily bhavcopies, 2014–2025.
  - Prices are back-adjusted: every price before an ex-date is multiplied by the action's factor. "Bonus a:b" is b/(a + b); a face-value split from F₁ to F₂ is F₂/F₁; both together multiply.
  - Renamed symbols are chained using NSE's symbol-change file.
  - A stock counts on a month-end if it has 250 prior trading days and traded in the last 5 market days.
- **Outcome.** The lowest close over the next 252 Nifty trading days is at least 50% below the as-of close.
  - "Observed path" uses the closes there are.
  - The upper bound also counts every stock that stopped trading inside the window and never returned. That includes mergers and buyouts, so it overstates the events.
- **Rules.** The tool's price flags through the same function as the case studies (`engine.price_flags`), and the three baselines.
- **Measures.** Precision P(event | flag), recall P(flag | event), base rate P(event), and lift = precision ÷ base rate; overall, for the liquid subset (median 60-day traded value ≥ ₹1 crore), by year, and without the symbols that have unexplained moves beyond ±40%.
- **What it cannot test.** Credit (no point-in-time statements for the universe) and events (no point-in-time disclosures). Results are in [case_studies.md](case_studies.md#wider-test-every-nse-stock-20162024-review-m1).

### 15.3 Tests

- **Bhavcopies:** both NSE formats parse to the same columns, ETFs and bonds are dropped, and Reliance's real 1:1 bonuses (7 Sep 2017, 28 Oct 2024) become −0.56% and +0.49% days, not −50%.
- **Rules:** every corporate-action wording seen in NSE's list parses to the right factor, and symbol chains follow A → B → C without relabelling a later, unrelated use of A.
- **Flags and metrics:** the universe's price flags equal the case-study engine's on the same window, the baselines and metrics are checked by hand, and the jump base rates recover a planted 1% daily jump rate (with zero excess over the stock's own history).

## 16. Demo snapshot (`ui/snapshot.py`, `scripts/build_snapshot.py`)

- **Built by running the app.** The build runs every preset (5 NSE stocks, 3 US stocks and the default portfolio) on every page with the default settings. It records each download and each heavy cached result in `data/snapshot/snapshot.pkl.gz`, keyed by a SHA-256 of its arguments:
  - frames and series through `pd.util.hash_pandas_object` with their column names, dtypes and shape;
  - dicts in sorted key order;
  - arguments bound to the function's signature, so positional and keyword calls share a key.
- **Downloads come from the snapshot in snapshot mode.** The heavy functions therefore see identical inputs and reuse their stored results. Anything else (a new ticker, a changed setting) misses and is computed live.
- **Results are keyed on inputs only, never on the mode.** A stored result is only ever returned for exactly the inputs it was computed from.
- **When the snapshot is ignored:** if it was built with another pandas version or numpy major version, or if `RISK_TOOL_NO_SNAPSHOT` is set (the test suite sets it).
- **Tests:**
  - keys are stable across processes, ignore dict order and change with any value;
  - hits, misses and the live switch work as intended, and a caller's changes to a snapshot download do not reach the store;
  - the bundled snapshot renders every page of the default portfolio with the network blocked and no live calculation;
  - `test_deploy.py` checks the cold-start peak memory against Streamlit Community Cloud's 690 MB minimum.

## References

- Acerbi, C. and Székely, B. (2014). Backtesting expected shortfall. *Risk*, December 2014.
- Agarwalla, S. K., Jacob, J. and Varma, J. R. (2013). Four factor model in Indian equities market. Working Paper W.P. No. 2013-09-05, Indian Institute of Management, Ahmedabad.
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589–609.
- Altman, E. I. (2000). Predicting financial distress of companies: revisiting the Z-score and ZETA models. Working paper, Stern School of Business, New York University.
- Altman, E. I., Iwanicz-Drozdowska, M., Laitinen, E. K. and Suvas, A. (2017). Financial distress prediction in an international context: a review and empirical analysis of Altman's Z-score model. *Journal of International Financial Management and Accounting*, 28(2), 131–171.
- Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). Direct estimation of equity market impact. *Risk*, July 2005.
- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.
- Bangia, A., Diebold, F. X., Schuermann, T. and Stroughair, J. D. (1999). Modeling liquidity risk, with implications for traditional market risk measurement and management. Wharton Financial Institutions Center working paper 99-06.
- Corwin, S. A. and Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance*, 67(2), 719–760.
- Crosbie, P. and Bohn, J. (2003). *Modeling default risk.* Moody's KMV.
- Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583–602.
- Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements.* Bank for International Settlements.
- Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk.* Bank for International Settlements.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Bollerslev, T. (1987). A conditionally heteroskedastic time series model for speculative prices and rates of return. *Review of Economics and Statistics*, 69(3), 542–547.
- Boyer, B. H., Gibson, M. S. and Loretan, M. (1999). Pitfalls in tests for changes in correlations. International Finance Discussion Paper 597, Board of Governors of the Federal Reserve System.
- Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Fama, E. F. and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
- Forbes, K. J. and Rigobon, R. (2002). No contagion, only interdependence: measuring stock market comovements. *Journal of Finance*, 57(5), 2223–2261.
- French, K. R. Data Library. Tuck School of Business at Dartmouth.
- Giacomini, R. and Komunjer, I. (2005). Evaluation and combination of conditional quantile forecasts. *Journal of Business & Economic Statistics*, 23(4), 416–431.
- Merton, R. C. (1974). On the pricing of corporate debt: the risk structure of interest rates. *Journal of Finance*, 29(2), 449–470.
- Meucci, A. (2009). Managing diversification. *Risk*, May 2009, 74–79.
- Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631–653.
- Patton, A., Politis, D. N. and White, H. (2009). Correction to "Automatic block-length selection for the dependent bootstrap". *Econometric Reviews*, 28(4), 372–375.
- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Politis, D. N. and White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews*, 23(1), 53–70.
- J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th edition.
- Jarque, C. M. and Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. *Economics Letters*, 6(3), 255–259.
- Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Maillard, D. (2012). A user's guide to the Cornish Fisher expansion. SSRN working paper.
- McNeil, A. J. and Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *Journal of Empirical Finance*, 7(3–4), 271–300.
- Securities and Exchange Board of India (2015). *SEBI (Listing Obligations and Disclosure Requirements) Regulations, 2015*, Regulations 31 (shareholding pattern) and 33 (financial results), as amended.
- Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn and A. W. Tucker (eds.), *Contributions to the Theory of Games II*, 307–317. Princeton University Press.
- Reserve Bank of India. *Master Circular – Basel III Capital Regulations* (current edition).
- Tasche, D. (1999). Risk contributions and performance measurement. Working paper, Technische Universität München.
- Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J. and Bouchaud, J.-P. (2011). Anomalous price impact and the critical nature of liquidity in financial markets. *Physical Review X*, 1(2), 021006.
- Vassalou, M. and Xing, Y. (2004). Default risk in equity returns. *Journal of Finance*, 59(2), 831–868.
