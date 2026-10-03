# Risk Analysis Tool

[![tests](https://github.com/singhwilliam15/Risk-Analysis-Tool/actions/workflows/tests.yml/badge.svg)](https://github.com/singhwilliam15/Risk-Analysis-Tool/actions/workflows/tests.yml)
![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Market, liquidity, credit, concentration and event risk for an Indian or US stock or portfolio**, run through one
linked stress engine. Every headline number has a 90% range and an A–D trust grade. The tool was tested against
real Indian collapses and against every NSE stock from 2016 to 2024, and it reports where it does not beat simpler
rules.

**▶ Live demo: [var-analysis-tool-jb48f2b7syny4j2vzeartr.streamlit.app](https://var-analysis-tool-jb48f2b7syny4j2vzeartr.streamlit.app/)**.
It opens on a demo snapshot of the presets (prices to 2 Oct 2026), so they load at once. Switch **Data → Live** in
the sidebar for today's prices, or type any NSE, BSE or US ticker.

| CRO dashboard | Integrated stress | Bank credit (RBI PCA) |
| --- | --- | --- |
| [![CRO dashboard](docs/images/cro-dashboard.png)](docs/images/cro-dashboard.png) | [![Integrated stress](docs/images/integrated-stress.png)](docs/images/integrated-stress.png) | [![HDFC Bank against RBI's PCA triggers](docs/images/credit-panel.png)](docs/images/credit-panel.png) |

## Five findings, including the negative ones

1. **On 20 dates before five Indian collapses, it warned on 15, with false alarms on 9.5% of 200 control dates.**
   But "price below its 200-day average" also warned on 15. It did so with 27.5% false alarms, and earlier on Zee
   and Future Retail. The tool's edge on these dates is precision, not catching more.
   [Case studies](docs/case_studies.md#does-it-beat-a-naive-rule-review-m1)
2. **On every NSE stock from 2016 to 2024 (2,312 stocks, delisted ones included), the price flags barely beat
   chance.** Their lift is 1.13 for a 50% fall within a year. "Down 30% in six months" does twice as well (2.25).
   The cause is one threshold: the 5% daily-ES floor, set with large caps in mind, fires on 72% of all NSE
   stock-dates. It is reported, not re-tuned after the fact. [Wider test](docs/case_studies.md#wider-test-every-nse-stock-20162024-review-m1)
3. **Cross-pillar effects appear only where the data create them.** Pledge selling and circuit locks move the price
   round after round, and the loss is split by exact Shapley values.
   - With real NSE pledge data, the 5-stock portfolio's Global Financial Crisis replay gains a +₹5,145 interaction
     over 4 rounds (₹4.89 lakh loss on ₹10 lakh).
   - Jaiprakash Power, with 73% of promoter shares pledged, loses ₹6.44 lakh on a −20% market move, against
     ₹2.17 lakh from the market alone.
   - Without pledges the interaction is zero, and the page says so.
4. **Banks are measured against RBI's own triggers.**
   - HDFC Bank sits 5.62 pp clear of every Prompt Corrective Action threshold.
   - Yes Bank was in risk threshold 1 (CET1 8.4%) six months before its 2020 moratorium, using the 2021
     thresholds as a benchmark.
   - The data come from NSE results XBRL and annual reports, cited by page.
5. **A key assumption failed its own test and was replaced.**
   - The event-jump overlay assumed a 0.5% daily jump chance for High-risk stocks, which is 72% a year.
   - Measured on NSE-wide base rates, price-based risk proxies showed no more stock-specific crashes than the stock's
     own history, which the base model already uses. The default is now 0.
   - A grid shows what any assumed jump would add. [Methodology §12.4](docs/methodology.md#124-jump-overlay-on-es)

## What it does

| Pillar | Highlights |
| --- | --- |
| **Market** | 8 VaR/ES models, from historical to GARCH(1,1)-t and Monte Carlo. Out-of-sample Kupiec, Christoffersen and McNeil-Frey ES tests; the recommended model must pass the ES test too. Euler risk split; measured crisis replays. |
| **Trust** | A 90% range for every model (block bootstrap, GARCH parameter draws) and an A–D grade from written rules, with reasons. Model risk, lookback sensitivity, ghost effect. |
| **Liquidity** | Days to liquidate, AMFI-style stress test, Corwin-Schultz spread, square-root impact, liquidity-adjusted VaR, official NSE price bands and circuit-lock freezes. |
| **Credit** | Merton distance to default with its percentile, agency default rates (CRISIL, ICRA studies), Altman Z/Z''. Banks and NBFCs against RBI PCA thresholds; bonds and loans with expected loss and spread stress. |
| **Concentration** | Fama-French-Carhart factors (IIMA India, Kenneth French US) with Newey-West errors, factor/specific risk split, effective number of bets, crisis correlation. |
| **Event & governance** | Point-in-time NSE data: promoter pledges, ASM/GSM, F&O bans, price bands, rating actions. Low/Elevated/High tier, pledge margin-call trigger, jump overlay with a sensitivity grid. |
| **Integration & decisions** | Linked stress as a price fixed point with a Shapley split and interaction, jump to default, reverse stress (stock and macro space), risk-change attribution, risk-reducing trades, limits, CRO dashboard and one-page memo. |

More: [features and live results](docs/features.md) · [methodology and references](docs/methodology.md) ·
[case studies and the wider test](docs/case_studies.md) · [deploying](docs/deploy.md) ·
[the 2-minute story](docs/interview_story.md)

## Run it

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/singhwilliam15/Risk-Analysis-Tool.git
cd Risk-Analysis-Tool
pip install -r requirements.txt
streamlit run app.py
```

Tickers use Yahoo Finance symbols: `RELIANCE.NS` (NSE), `.BO` (BSE), `AAPL` (US). On Windows you can double-click
`run_app.bat`.

| Run | First page | All 9 pages | Peak memory |
| --- | --- | --- | --- |
| Demo snapshot (default portfolio) | 9 s | 28 s | 299 MB |
| Live downloads and calculations | 85 s | 119 s | 326 MB |

Measured cold on a laptop with `scripts/measure_app.py`. Streamlit Community Cloud allows at least 690 MB, and
`test_deploy.py` checks the snapshot run stays under it.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

There are 584 tests, run offline in CI on Python 3.11 and 3.12. The network is blocked, so market data come from a
synthetic generator, saved real files or the bundled snapshot. Each calculation is checked against an independent
reference: a closed form, a large simulation, a hand calculation or a known true parameter.
[What they check](docs/features.md#tests).

**Data refresh scripts** (in `scripts/`):

| Script | Refreshes |
| --- | --- |
| `fetch_disclosures.py` | NSE pledges, bands, surveillance and ratings |
| `fetch_bank_results.py` | Bank results XBRL |
| `refresh_factor_data.py` | Factor files |
| `fetch_bhavcopy.py` + `run_universe.py` | The wider test |
| `build_snapshot.py` | The demo snapshot |

## Limits, in short

- Long-only, single-currency portfolios.
- Merton PDs are risk-neutral.
- Price bands and pledges are a dated snapshot of NSE files.
- The event tiers' jump risk cannot be calibrated on disclosure data (NSE's pledge history starts in 2021).
- The wider test covers price-based flags only.

The full list is in [features.md](docs/features.md#limitations).

## License

[MIT](LICENSE)
