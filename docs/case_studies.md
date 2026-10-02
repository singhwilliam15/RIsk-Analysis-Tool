# Did it see it coming? Case studies on Indian collapses

The tool was run **as of 12, 6, 3 and 1 month(s) before** each collapse. Each run used only data public on that date: prices up to the date, statements public by then, and disclosures by their public dates. The results are compared with what happened next, and with a control group of 10 large, stable NSE stocks at the same dates. Run it with `python case_studies/run.py`; results go to `case_studies/results/`.

**Coverage today (2 Oct 2026).** Only price-based pillars could be assessed: market, liquidity and integrated stress. Credit needs statements for the years before each event, and Yahoo only has FY2023 onwards. Events need disclosure files (pledges, ratings, surveillance, F&O ban, auditor events). None of these is loaded yet, so those pillars show **not available**, never "no risk". The checklist at the end lists every file needed.

## Cases

| Case | Event date | Event | Prices |
| --- | --- | --- | --- |
| Yes Bank | 5 Mar 2020 | RBI places Yes Bank under a moratorium and supersedes its board | Yahoo |
| DHFL | 4 Jun 2019 | DHFL misses interest payments on its bonds | **not available** (delisted; Yahoo returns 404). Needs an NSE price file |
| Zee Entertainment | 25 Jan 2019 | Share price collapses on promoter-pledge and lender concerns | Yahoo |
| Adani Enterprises | 24 Jan 2023 | Hindenburg Research publishes its short-seller report | Yahoo |
| *Candidates, not run until approved* | | Jet Airways (suspends flights, 17 Apr 2019); Future Retail (business sold to Reliance after defaults, 29 Aug 2020) | Yahoo has both |

**Control group:** Reliance, TCS, HDFC Bank, Infosys, Hindustan Unilever, ITC, Kotak Mahindra Bank, Asian Paints, Nestlé India and Britannia, each at every case's four dates.

## What counts as a warning (assumptions)

At each as-of date, using the last 500 trading days up to it:

| Pillar | Flag |
| --- | --- |
| Market | EWMA volatility ≥ **1.5×** the past year's volatility, **or** 1-day 95% historical ES ≥ **5%** |
| Liquidity | ≥ 3 lower-circuit days in the window (the Elevated event rule, §12.2 of the methodology) |
| Credit | Merton DD < 3, or Altman Z'' in the distress zone (needs statements; banks and NBFCs are not modelled) |
| Events | event tier Elevated or High (needs disclosure files) |
| Integrated stress | a −20% market move, run through the linked engine (downside beta, volatility doubled, liquidity, circuit freeze), loses ≥ **30%** of a ₹1 crore position |

**Warning** = any flag. **Realised loss** = the fall from the as-of close to the lowest close between the as-of date and 63 trading days after the event.

## Results

| Case | As of | Market ES (headline model, grade) | Flags | Warning | Realised loss |
| --- | --- | --- | --- | --- | --- |
| Yes Bank | 12m (5 Mar 2019) | 11.1% (EWMA, C) | market | **yes** | −93% |
| Yes Bank | 6m (5 Sep 2019) | 12.1% (EWMA, D) | market, stress | **yes** | −74% |
| Yes Bank | 3m (5 Dec 2019) | 19.3% (Cornish-Fisher, C) | market, stress | **yes** | −74% |
| Yes Bank | 1m (5 Feb 2020) | 18.8% (Cornish-Fisher, C) | market, stress | **yes** | −57% |
| Zee Entertainment | 12m (25 Jan 2018) | 3.7% (Cornish-Fisher, B) | none | no | −46% |
| Zee Entertainment | 6m (25 Jul 2018) | 3.5% (Cornish-Fisher, A) | none | no | −39% |
| Zee Entertainment | 3m (25 Oct 2018) | 3.5% (Student-t, B) | none | no | −26% |
| Zee Entertainment | 1m (25 Dec 2018) | 3.2% (Normal, B) | market (volatility 1.63×) | **yes** | −27% |
| Adani Enterprises | 12m (24 Jan 2022) | 9.3% (GARCH-t, C) | market | **yes** | −30% |
| Adani Enterprises | 6m (24 Jul 2022) | 6.4% (Cornish-Fisher, B) | market | **yes** | −52% |
| Adani Enterprises | 3m (24 Oct 2022) | 6.4% (Student-t, B) | market | **yes** | −64% |
| Adani Enterprises | 1m (24 Dec 2022) | 6.6% (Student-t, B) | market | **yes** | −67% |
| DHFL | all | not available (no prices) | – | – | – |

ES is 1-day at 95% for the recommended model, with the Trust grade. The 90% ranges are in `case_studies/results/results.csv`.

![Realised loss after each as-of date, coloured by whether the tool warned](images/case_studies.png)

### Hits, misses and false positives

| Pillar | Case dates | Hits | Misses | Control dates | False positives |
| --- | --- | --- | --- | --- | --- |
| Market | 12 | 9 | 3 | 120 | 6 (5.0%) |
| Liquidity (circuits) | 12 | 0 | 12 | 120 | 0 |
| Credit | 0 | – | – | 0 | – (not available) |
| Events | 0 | – | – | 0 | – (not available) |
| Integrated stress | 12 | 3 | 9 | 120 | 2 (1.7%) |
| **Any warning** | **12** | **9 (75%)** | **3** | **120** | **8 (6.7%)** |

### What this shows, honestly

- **Yes Bank was visible in prices a year ahead.** It had already fallen hard in 2018–19, so its ES was 11–19% a day and every date warned. A price-based tool flags a stock that is already falling; that is not a forecast of the moratorium.
- **Zee was missed for a year.** Its volatility looked normal until a month before the collapse. The risk was in promoter pledges, which only the event pillar sees, and those files are not loaded. This is exactly the case the event pillar exists for. It cannot be credited until the pledge data are in.
- **Adani Enterprises "warned" because it was always volatile, not because of an early signal.** Its ES was above 5% at every date. The 12-month warning also sits on a 2-year window that includes the March 2020 crash. The same artefact is behind two of the eight false positives (Reliance and Kotak Mahindra Bank, January 2022).
- **False positives were 6.7% overall.** Most were market flags from volatility jumps in large caps (ITC in February 2020, Reliance and Asian Paints in October 2018) that were not followed by large losses. Two came from the integrated-stress rule on Reliance, whose high downside beta makes a −20% market move cost about 30%.
- **The liquidity flag never fired.** None of these names hit lower circuits in the windows; Yes Bank and Zee were F&O stocks, which have no fixed price band. Circuit risk matters for small-caps, not for these cases.
- **Small sample.** Twelve case dates from three stocks are not statistically meaningful; they illustrate what the tool can and cannot see.

### Phase 5 jump defaults: kept, not revised

Observed one-day falls over the 63 trading days after each as-of date:

| Dates | Days | Falls ≥ 10% | Falls ≥ 20% | P(fall ≥ 10%) a day | P(fall ≥ 20%) a day |
| --- | --- | --- | --- | --- | --- |
| Case, warning | 567 | 21 | 6 | 3.7% | 1.06% |
| Case, no warning | 189 | 1 | 1 | 0.53% | 0.53% |
| Control, warning | 504 | 2 | 0 | 0.40% | 0 |
| Control, no warning | 7,056 | 11 | 0 | 0.16% | 0 |

- **On every warned date (cases and controls together),** falls of 20% or more happened on 6 of 1,071 days (0.56% a day). That is close to the High-tier default of 0.5% a day with J = −20%.
- **But the sample is dominated by Yes Bank,** with 6 of the 7 falls of 20% or more. The warnings here are price flags, not the disclosure-based tiers the jump overlay uses.
- **The Elevated default** (0.1% a day, −10%) cannot be tested until disclosure files give real tiers.
- **Decision:** the defaults stay, marked "assumption", and are to be re-run once the checklist files are loaded.

## Checklist of official files to download

Put each file where the last column says, add disclosure files to `case_studies/data/disclosures/manifest.json` (source URL and download date), and rerun. Each link is the official landing page: search the company and period there. The deep links for specific filings were not verified, so none is given.

| Case | File | Period | Official source | Save as |
| --- | --- | --- | --- | --- |
| DHFL | Daily prices and volume (security-wise price-volume archive) | Jan 2016 – Sep 2019 | [NSE historical data](https://www.nseindia.com/report-detail/eq_security) | `case_studies/data/prices/DHFL.csv` |
| DHFL | Promoter pledged data | Mar 2017 – Jun 2019 quarters | [NSE pledged data](https://www.nseindia.com/companies-listing/corporate-filings-pledged-data) | `case_studies/data/disclosures/` (pledges) |
| DHFL | Credit rating rationales (all actions) | 2018 – 2019 | [CARE Ratings](https://www.careratings.com) | `case_studies/data/disclosures/` (ratings) |
| DHFL | Credit rating rationales | 2018 – 2019 | [ICRA](https://www.icra.in) | `case_studies/data/disclosures/` (ratings) |
| DHFL | Credit rating rationales | 2018 – 2019 | [CRISIL Ratings](https://www.crisilratings.com) | `case_studies/data/disclosures/` (ratings) |
| DHFL | Auditor announcements | 2017 – 2019 | [NSE corporate announcements](https://www.nseindia.com/companies-listing/corporate-filings-announcements) | `case_studies/data/disclosures/` (auditor events) |
| Yes Bank | Promoter pledged data | Mar 2018 – Dec 2019 quarters | [NSE pledged data](https://www.nseindia.com/companies-listing/corporate-filings-pledged-data) | `case_studies/data/disclosures/` (pledges) |
| Yes Bank | F&O ban files (daily) | Mar 2019 – Mar 2020 | [NSE derivatives reports](https://www.nseindia.com/all-reports-derivatives) | `case_studies/data/disclosures/` (fo_ban) |
| Yes Bank | Credit rating rationales | 2018 – 2020 | [ICRA](https://www.icra.in) | `case_studies/data/disclosures/` (ratings) |
| Yes Bank | Credit rating rationales | 2018 – 2020 | [CARE Ratings](https://www.careratings.com) | `case_studies/data/disclosures/` (ratings) |
| Yes Bank | Credit rating rationales | 2018 – 2020 | [India Ratings](https://www.indiaratings.co.in) | `case_studies/data/disclosures/` (ratings) |
| Yes Bank | Annual reports (GNPA, NNPA, CAR for the bank panel) | FY2018 – FY2019 | [NSE annual reports](https://www.nseindia.com/companies-listing/corporate-filings-annual-reports) | values for the Credit page's bank panel |
| Zee Entertainment | Promoter pledged data | Dec 2016 – Dec 2018 quarters | [NSE pledged data](https://www.nseindia.com/companies-listing/corporate-filings-pledged-data) | `case_studies/data/disclosures/` (pledges) |
| Zee Entertainment | F&O ban files (daily) | Jan 2018 – Jan 2019 | [NSE derivatives reports](https://www.nseindia.com/all-reports-derivatives) | `case_studies/data/disclosures/` (fo_ban) |
| Zee Entertainment | Annual statements (consolidated) | FY2016 – FY2018 | [NSE annual reports](https://www.nseindia.com/companies-listing/corporate-filings-annual-reports) | `case_studies/data/fundamentals/ZEEL.NS.csv` |
| Zee Entertainment | Credit rating rationales | 2018 – 2019 | [CARE Ratings](https://www.careratings.com) | `case_studies/data/disclosures/` (ratings) |
| Adani Enterprises | Promoter pledged data | Dec 2020 – Dec 2022 quarters | [NSE pledged data](https://www.nseindia.com/companies-listing/corporate-filings-pledged-data) | `case_studies/data/disclosures/` (pledges) |
| Adani Enterprises | ASM / GSM lists in force | Jan 2022 – Jan 2023 | [NSE surveillance](https://www.nseindia.com/reports/asm) | `case_studies/data/disclosures/` (surveillance) |
| Adani Enterprises | Annual statements (consolidated) | FY2020 – FY2022 | [NSE annual reports](https://www.nseindia.com/companies-listing/corporate-filings-annual-reports) | `case_studies/data/fundamentals/ADANIENT.NS.csv` |
| Adani Enterprises | Credit rating rationales | 2021 – 2023 | [CARE Ratings](https://www.careratings.com) | `case_studies/data/disclosures/` (ratings) |
| Adani Enterprises | Auditor announcements | 2021 – 2023 | [NSE corporate announcements](https://www.nseindia.com/companies-listing/corporate-filings-announcements) | `case_studies/data/disclosures/` (auditor events) |
| Control group | Annual statements (consolidated) | three years before each case's dates | [NSE annual reports](https://www.nseindia.com/companies-listing/corporate-filings-annual-reports) | `case_studies/data/fundamentals/<TICKER>.csv` |
| Control group | Promoter pledged data | same quarters as the cases | [NSE pledged data](https://www.nseindia.com/companies-listing/corporate-filings-pledged-data) | `case_studies/data/disclosures/` (pledges) |

**Data rules:**
- Statements go in the app's fundamentals template, in full currency units, with `shares_issued`, both debt lines and the filing date where known.
- Disclosures use the templates in `data/disclosures/templates/`.
- No row is to be filled in from memory or estimates: a figure that cannot be found stays missing.
