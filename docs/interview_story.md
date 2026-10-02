# The 2-minute story

A script to say out loud: about 300 words, roughly two minutes at a natural pace. Every number is from the live runs on 2 Oct 2026 (README and `docs/case_studies.md`).

---

**1. The problem: siloed risk tools (20 seconds)**

Most risk tools put market, liquidity and credit risk on separate pages, as if they were independent. In a real crisis they arrive together. Prices fall, volume dries up, leverage worsens, and pledged promoter shares get sold, which pushes prices down further. I wanted a tool that measures those links, and that is honest about how far each number can be trusted.

**2. What I built (40 seconds)**

It's a risk analysis tool for Indian and US stocks and portfolios, built in Python and Streamlit, with 434 offline tests.

- **Five pillars:** market, liquidity, credit, concentration and factors, and event and governance risk.
- **A trust layer:** every headline number carries a 90% range and an A-to-D grade from written rules, so a fragile number looks fragile.
- **A linked stress engine:** it runs one crisis through every pillar at once and compares the result with simply adding up the separate pages.
- **A decision layer:** a reverse stress test, limits, risk-reducing trades, and a one-page memo for a chief risk officer, written by rules, not a language model.

**3. A result from a real collapse (40 seconds)**

I ran it as of 12, 6, 3 and 1 month before five Indian collapses, using only data public at the time.

Take Future Retail. Three months before the August 2020 sale, a standard model showed a volatile stock: 9% daily Expected Shortfall. The liquidity pillar showed what really mattered. The stock had been locked at its 5% lower circuit on 26 days, 18 of them in a row, so a holder could not sell. The linked stress put the loss from a 20% market fall at 70%.

Across the cases it warned on 15 of 20 dates, with false alarms on about one in ten control dates.

**4. What it still cannot do (20 seconds)**

- **Zee and Future Retail were missed at 12 months,** because their risk sat in promoter pledges and group debt, which need disclosure files I have not loaded yet.
- **For liquid large-caps, the pillars barely interact.** The tool says so rather than inventing an effect.
- **The jump sizes are assumptions** until those files can test them.

---

**Likely follow-up questions, and the short answers**

- *Why Shapley for risk change?* It splits a change exactly, whatever the order in which the factors moved.
- *Why grades?* A VaR with a ±40% range should not look as solid as one with ±10%.
- *What would you add next?* The disclosure data for the case studies, then FX for mixed INR/USD portfolios.
