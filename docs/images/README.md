# Screenshots

Images used in the README. They were captured on 3 Oct 2026 from the running app (`streamlit run app.py
--theme.base dark`) in demo-snapshot mode (prices to 2 Oct 2026). A headless browser (playwright, Microsoft Edge) at
1,700 px width captured the main area; empty space below the content is trimmed.

| File | What it shows |
| --- | --- |
| `cro-dashboard.png` | Overview, default 5-stock portfolio at ₹10 lakh: the CRO dashboard (pillar rows, bottom line, top risks and actions) |
| `integrated-stress.png` | Integrated Stress, same portfolio: the linked loss per scenario split by Shapley values, interaction, feedback rounds, reverse stress |
| `credit-panel.png` | Credit, HDFC Bank: RBI PCA headline, the PCA table with distance to each trigger, and the five-year trend |
| `case_studies.png` | Case studies: realised loss after each as-of date, red where the tool warned (written by `case_studies/run.py`) |
| `liquidity-waterfall.png` | Liquidity: VaR → spread → impact → circuit-lock waterfall (captured 2 Oct 2026; not used in the README now) |

To refresh them, run the app with the same settings and capture the same pages; rerun `case_studies/run.py` for the
case-study chart.
