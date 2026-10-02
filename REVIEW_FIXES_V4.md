# Risk Analysis Tool — Review Fixes (v4)

A review of `singhwilliam15/RIsk-Analysis-Tool` at commit 35a0578 (all 434 tests pass). The
engineering and the honesty discipline are strong. The fixes below are about whether the
**headline claims** hold up: the linked stress engine, the credit add-on, the reverse-stress
probability, and the evidence behind the event and credit pillars.

## Findings

### Critical: these undermine headline claims
| ID | Finding | Evidence in the code |
|---|---|---|
| C1 | The "linked" stress engine is additive. There is no feedback: forced selling and impact are added as costs but never move the price, so they never re-trigger margin calls, circuits or Merton. The "interaction" compares scenario losses with *unconditional* standalone headlines, so it measures a definitional difference rather than an interaction. | `integration.py` `linked_stress`, `siloed_sum`; `ui/integration_layer.py` `standalone`. README: large-cap interaction ≈ −₹1,206; Jaiprakash Power's +₹1.27 lakh is the circuit-freeze scenario itself. |
| C2 | The credit add-on double-counts for equity holders. Merton PD is *implied from the equity price*, so the scenario equity loss already prices the higher default risk; adding ΔPD × value counts it twice. | `linked_stress`: `cred = max(0, PD_stressed − PD) × after` |
| C3 | The reverse-stress plausibility is misstated. χ²(k) and F(k, ν) give the probability of a move *anywhere* outside the Mahalanobis ellipsoid in k dimensions, not the probability of losing L. The README's "1.8% under a normal" corresponds to a one-sided probability of about 0.01%, roughly 160× smaller. | `integration.py` `plausibility`; README reverse-stress bullet |
| C4 | The two differentiating pillars run on no real data. No disclosure files are loaded, so the event pillar shows "not available" in the live demo, and the case studies test only the price-based pillars. | `data/disclosures/manifest.json` and `case_studies/data/*` are empty |

### Major
| ID | Finding |
|---|---|
| M1 | The case studies select on outcome (only collapses) and have no naive baseline. Without beating simple rules ("already down 30%", "volatility in top decile"), a 75% hit rate shows little. Yahoo also drops delisted stocks (DHFL returns 404), which creates survivorship bias in any wider test. |
| M2 | The jump-overlay defaults are implausible and calibrated circularly. 0.5% a day is about a 72% chance per year of a ≥20% one-day crash. The defaults were "confirmed" using warned days in a sample chosen for its crashes. |
| M3 | The recommended model ignores the ES backtest. HDFC Bank's headline model (EWMA) fails the ES test, but is still the headline, with grade D. |
| M4 | Merton PDs for large caps are about 10⁻³⁰ and mean nothing. Yahoo statements start in FY2023, which blocks DD history and the credit case studies. |
| M5 | Banks and NBFCs, a quarter of the default portfolio, plus Yes Bank and DHFL, get only a manual panel. For a banking-focused tool this is the biggest content gap. |
| M6 | Possible double count: the circuit-freeze loss is added on top of *historical replay* returns, which may already include those locked days. |

### Presentation
| ID | Finding |
|---|---|
| P1 | The repo name has a typo ("RIsk-Analysis-Tool"), and the live-demo URL still says `var-analysis-tool`. |
| P2 | The README is about 500 lines and partly stale: "is being extended", "Market risk is complete", and a status row "Integrated Stress · Decisions — Coming next (Phase 6)". |
| P3 | The interview story leads with "measures those links", which C1 contradicts. |
| P4 | Deployment of the renamed repo, with the much heavier app, is unverified on Streamlit Cloud (memory, cold-start time). |

---

## Master prompt (paste into Claude Code)

```
Read REVIEW_FIXES_V4.md in this repo. It is an external review with findings C1–C4, M1–M6 and
P1–P4, and fix phases A–E below. Work ONE phase at a time in order: plan mode → my approval →
implement → pytest → run the app briefly → commit → summarise with real before/after numbers
→ STOP. Branch: `review-fixes-v4`. Never push.

Keep all existing rules from RISK_TOOL_PLAN_V3.md: point-in-time data only, never invent data,
label assumptions, offline tests against independent references, and update README and
methodology after each phase.

If you disagree with a finding after reading the code, say so with evidence before changing
anything. Do not "fix" something that is not broken.

Start with Phase A.
```

---

## Phase A — Statistical corrections (C3, M3, M6)

```
Phase A of REVIEW_FIXES_V4.md.

1. C3 Reverse-stress plausibility. Replace the "probability" shown with the probability of the
   LOSS itself: P(wᵀx ≤ −L). Under a normal this is Φ(−L/σ_p); under the multivariate Student-t
   it is the univariate t tail of the portfolio at the same distance (scaled correctly for the
   t's variance). Keep the Mahalanobis distance and rename the χ²/F figure "share of
   scenarios at least this extreme in any direction" (or drop it). Also show the loss
   probability per month and as "once every N years". Tests: the closed form against a large
   simulation for both distributions; README numbers updated.
2. M3 Recommended model: when the ES backtest is available, require it to pass too.
   Order: pass all VaR tests and the ES test → lowest tick loss; else pass the VaR tests only →
   lowest tick loss, with a warning; else the current fallback. Re-check the trust-grade rules
   so a headline model that fails a test is never shown as "recommended". Tests for every branch.
3. M6 Circuit freeze in historical replay: only add the freeze loss when the replayed path
   does NOT already contain lower-circuit days for that holding in the window (or add only the
   additional days beyond those already in the replay). Test on a synthetic replay that
   already contains a lock.
```

---

## Phase B — A real linked engine (C1, C2)

```
Phase B of REVIEW_FIXES_V4.md. Rework integration.py.

1. C1 Feedback loop (fire-sale spiral). Within a scenario, iterate to a fixed point:
     price_k → (a) is the pledge margin-call trigger breached? → forced sale quantity
             → (b) the square-root-law PRICE impact of forced selling (and of our own exit)
                   lowers the price, not just a cost line
             → (c) circuit band reached? → lock days and frozen exit
             → (d) Merton re-solved at the new equity value and volatility (signal only; see 2)
     → price_{k+1}. Stop when the change in price is < 0.1% or after 20 rounds; report rounds.
   Impact parameters and the share of impact that is permanent (vs temporary) are labelled
   assumptions, with defaults justified in methodology.md.
2. C2 Credit for equity holders: remove ΔPD × value from the additive loss. Instead:
   - report the stressed DD/PD as a signal next to the scenario;
   - add an explicit JUMP-TO-DEFAULT scenario per non-financial holding: if the stressed DD
     falls below a threshold (assumption, editable), show the loss if equity goes to a
     recovery value (assumption, default 0–10%), with the model probability next to it.
   Additive credit loss is kept only for debt positions (see Phase D).
3. New interaction measure: interaction = full linked loss − (market loss + Σ each link
   switched on ALONE under the same scenario). This is the cross effect only. Also show a
   Shapley split of the full loss across market, liquidity, credit and events. Remove the
   "siloed sum of page headlines" comparison, or keep it clearly labelled "naive page sum"
   and not called interaction.
4. Tests: with links off, the result equals the plain market stress; a single link alone has
   zero interaction; a constructed pledge + thin-volume case shows a positive interaction that
   matches a hand calculation; convergence on a case where feedback is strong; Shapley parts
   sum to the total.
5. Re-run the README's integrated-stress table (large caps, the 5-stock portfolio,
   Jaiprakash Power, a small-cap with high pledging if data are loaded) and report honestly,
   including when interaction stays near zero.
```

---

## Phase C — Real data for the event and credit pillars (C4, M4)

```
Phase C of REVIEW_FIXES_V4.md. This phase needs files I download. First give me a precise
checklist, then wait while I download, then build.

1. Current data for the live demo: for every preset ticker plus 5 small/mid-caps with known
   high promoter pledging (pick them from NSE's pledged-data page once I've downloaded it),
   the latest promoter pledge data (last 8 quarters), current ASM/GSM lists, the price-band
   file and recent rating actions. Bundle them in data/disclosures/ with the manifest, so the
   live event pillar shows real tiers. Add a visible "Data as of" stamp.
2. Rating-implied default rates: map each company's long-term rating to the 1-year default
   rate in the rating agencies' published default studies (CRISIL, ICRA or CARE annual
   default studies). Store the table with source, year and agency, and show it next to the
   Merton PD.
3. Merton presentation: show DD and its percentile among the preset universe, instead of
   leading with PDs of 10⁻³⁰. Keep the PD in the detail with its "risk-neutral, model-implied"
   label.
4. Older statements: CSV uploads for FY2016–FY2022 for the case-study companies (I provide
   them from annual reports), so the credit pillar can run in the case studies.
```

---

## Phase D — Banking credit module and debt positions (M5, C2 follow-up)

```
Phase D of REVIEW_FIXES_V4.md.

1. Bank/NBFC credit module, replacing the manual panel: GNPA %, NNPA %, provision coverage,
   CRAR, CET1, Tier-1 leverage ratio, CASA %, credit-deposit ratio, LCR, NIM, ROA, slippage
   ratio. Data from annual reports and Basel III Pillar 3 disclosures via a CSV template
   (point-in-time by publication date). Trends over 5 years.
2. Early-warning rules mapped to RBI's Prompt Corrective Action (PCA) framework risk
   thresholds for capital, asset quality and leverage. Take the exact thresholds from the
   current RBI PCA circular, cite it, and show "distance to PCA trigger" for each metric.
   For NBFCs, use RBI's scale-based regulation and the NBFC PCA framework if applicable
   (verify and cite; if unsure, say so and leave it out).
3. Debt positions: allow holdings of corporate bonds/NCDs and loans with face value, coupon,
   maturity, rating, seniority and spread. Credit loss = EAD × PD × LGD (LGD by seniority,
   an assumption), plus spread-widening loss = spread duration × Δspread in stress scenarios.
   Debt positions join the linked engine (equity distress → spread widening → loss).
4. Tests: PCA distance on constructed banks; EL by hand; spread-duration loss against a
   full repricing of a simple bond.
```

---

## Phase E — Evidence and presentation (M1, M2, P1–P4)

```
Phase E of REVIEW_FIXES_V4.md.

1. M1 Baselines: on the same 220 case/control dates, compute naive rules: (a) down ≥ 30% over
   the previous 6 months, (b) 60-day volatility in the top decile of the stock's own 5-year
   history, (c) below its 200-day average. Report hit rate, false-positive rate and lead time
   for each next to the tool's flags. State plainly where the tool does not beat a baseline.
2. M1 Wider test (optional, if time allows): a point-in-time universe of NSE stocks with
   delisted names from NSE bhavcopy archives I download (Yahoo drops them). Precision/recall
   for a ≥50% drawdown over the next 12 months. If the universe is survivorship-biased, say so
   on the page.
3. M2 Jump defaults: express every jump probability as an annual probability as well as a
   daily one; recalibrate from base rates (Phase E2 universe, or published studies), not from
   a crash-selected sample; show an ES sensitivity table over p and J.
4. P2 README rewrite: one screen at the top (pitch, live demo, 3 screenshots, 5 key findings
   including the honest negatives), then a short feature table. Move detail to docs/.
   Remove stale lines ("being extended", "Coming next (Phase 6)").
5. P3 Update docs/interview_story.md to match the post-fix results.
6. P4 Deployment: add a lightweight "demo snapshot" mode (precomputed results for the presets,
   loaded instantly, with an "as of" stamp) so the live app is fast on Streamlit Cloud; heavy
   computation runs only for new tickers. Test that the app starts within the cloud memory limit.
7. P1 List the manual steps for me: rename the repo to `Risk-Analysis-Tool` (fixing "RIsk"),
   set the Streamlit app's custom subdomain (e.g. risk-analysis-tool), and update every link
   (README badges, profile README, resume, LinkedIn).
```
