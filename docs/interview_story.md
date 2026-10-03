# The 2-minute story

A script to say out loud: about 330 words, roughly two minutes at a natural pace. Every number is from the runs of
2–3 Oct 2026 (README, `docs/case_studies.md`). It was rewritten after an external review that found the first
version over-claimed; the review and its fixes are in `REVIEW_FIXES_V4.md`.

---

**1. The problem (20 seconds)**

Most risk tools put market, liquidity and credit risk on separate pages. In an Indian crisis they arrive together:
prices fall, the stock locks at its lower circuit, and lenders sell pledged promoter shares, which pushes the price
down further. I wanted a tool that models that spiral, and that says how far each number can be trusted.

**2. What I built (40 seconds)**

It's a risk tool for Indian and US stocks and portfolios, in Python and Streamlit, with 584 offline tests.

- **Five pillars** (market, liquidity, credit, concentration, and event risk), plus a check of banks against RBI's
  Prompt Corrective Action triggers.
- **A trust layer:** every headline number has a 90% range and an A-to-D grade.
- **A linked stress engine.** Forced pledge selling and circuit locks move the price round after round until it
  settles, and an exact Shapley split shows how much each pillar adds. With real NSE pledge data, Jaiprakash
  Power loses 64% on a 20% market fall, three times the market move alone. Without pledges the cross effect is
  zero, and the tool says so.

**3. Testing it honestly (40 seconds)**

I ran it before five Indian collapses using only data public at the time. It warned on 15 of 20 dates. Yes Bank was
inside RBI's first PCA threshold six months before its moratorium.

Then I checked it against simple rules. "Price below its 200-day average" also caught 15 of 20, with three times
the false alarms, so my edge there is precision. On every NSE stock from 2016 to 2024, delisted ones included, my
price flags barely beat chance: one tail-risk threshold, set with large caps in mind, fires on 72% of all NSE stock-days. "Down 30% in six months"
did twice as well. I reported that rather than re-tuning after the fact.

**4. What I changed because of the evidence (20 seconds)**

The event overlay assumed a high-risk stock had a 72% yearly chance of a 20% crash. NSE-wide base rates didn't
support it, so the default is now zero, with a grid showing what any assumption would add. The next step is a
size-aware volatility rule, tested out of sample.

---

**Likely follow-up questions, and the short answers**

- *Why Shapley?* It splits a loss exactly across market, liquidity, credit and events, whatever the order in which
  they act, and the interaction is what is left when they act together.
- *Why didn't credit add a loss for shareholders?* The Merton PD is implied from the share price, which the scenario
  already marks down, so adding it would count the same risk twice. Credit is a signal there, and a loss only for
  bonds and loans.
- *How do you avoid survivorship bias?* The wider test uses NSE's daily files, which keep delisted stocks. Prices are
  adjusted for bonuses and splits with NSE's corporate-action list, because NSE's previous close is not adjusted.
- *Why grades?* A VaR with a ±40% range should not look as solid as one with ±10%.
