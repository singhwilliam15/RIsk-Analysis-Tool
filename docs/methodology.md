# Methodology

This document describes every calculation in the Risk Analysis Tool: the eight market-risk models, the multi-day rules, the backtests, the model-selection rule, the portfolio risk decomposition, the stress tests, and the data layer that every risk pillar shares (positions, prices and volume, fundamentals, disclosures and data quality), the trust layer (90% ranges, model risk and A–D grades), and the liquidity and credit pillars. Section numbers match the code modules named in each heading. The pillars still to come (liquidity, credit, concentration, event risk, integration) are planned in `RISK_TOOL_PLAN_V3.md` and will be documented here as they are built.

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
- **Recommended model:** the lowest tick loss among models that pass all three VaR tests.
- **If none pass:** the app says so and shows the lowest-loss model with a warning.
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

- **Datasets.** Promoter pledges, ASM/GSM surveillance, price bands, the F&O ban list, rating actions and auditor events. All of them come only from files you download from NSE, BSE and the rating agencies; `data/disclosures/README.md` gives the source and steps for each.
- **Manifest.** Every file must be listed in `manifest.json` with its source URL, download date and coverage. Unlisted files are not loaded.
- **Validation.** Each row is checked for:
  - required columns and readable dates;
  - percentages between 0 and 100;
  - allowed values (surveillance measure, rating action, auditor event type; price band 2/5/10/20% or No Band);
  - consistency (pledged % of total shares ≤ promoter holding %; a surveillance exit on or after its entry).

  Rejected rows are reported with their row number, and the valid rows are kept.
- **Point in time.** Each dataset has a public date: the disclosure, entry, effective, trade, action or event date. A pledge row without a disclosure date becomes public at **quarter end + 21 days**, the filing deadline for the quarterly shareholding pattern (SEBI LODR Reg. 31(1)(b)).
- **Layouts not yet verified.** Only NSE's `fo_secban.csv` layout is parsed structurally (its trade date comes from the first line). For the other official files, headers are matched through an alias table written without a real download to check against.

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
- **When there's no clean recommendation:** if no model passes, or the recommended model itself fails the ES test, the add-on is not defined, and the range is taken across all models.

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
- **Headline grade:** the recommended model's (Historical if there is none).
- **Not graded:** the recommendation rule (§3.7) still uses the three VaR tests only. The grade also counts the ES test, so a recommended model can be graded D, as HDFC Bank's EWMA was on 2 Oct 2026. The Trust page says so.

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

- **Ratings:** the latest rating action public by the analysis date in your rating-action file (§7.4) gives the current rating, last action and direction.
- **Financial companies:** banks, NBFCs and insurers are identified by Yahoo sector "Financial Services" or industry keywords. Merton, Altman and leverage ratios are skipped for them: deposits and policyholder liabilities are their business, not debt in the Merton sense.
- **Manual panel for financials:** GNPA, NNPA, capital adequacy (CAR), CASA and net interest margin, typed in from the annual report. CAR is compared with the RBI minimum of 11.5% (9% CRAR plus the 2.5% capital conservation buffer; RBI Master Circular on Basel III capital regulations).

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

## References

- Acerbi, C. and Székely, B. (2014). Backtesting expected shortfall. *Risk*, December 2014.
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
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Giacomini, R. and Komunjer, I. (2005). Evaluation and combination of conditional quantile forecasts. *Journal of Business & Economic Statistics*, 23(4), 416–431.
- Merton, R. C. (1974). On the pricing of corporate debt: the risk structure of interest rates. *Journal of Finance*, 29(2), 449–470.
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
- Reserve Bank of India. *Master Circular – Basel III Capital Regulations* (current edition).
- Tasche, D. (1999). Risk contributions and performance measurement. Working paper, Technische Universität München.
- Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J. and Bouchaud, J.-P. (2011). Anomalous price impact and the critical nature of liquidity in financial markets. *Physical Review X*, 1(2), 021006.
- Vassalou, M. and Xing, Y. (2004). Default risk in equity returns. *Journal of Finance*, 59(2), 831–868.
