# Screenshots

Images used in the main README. They were captured from the running app (`streamlit run app.py`) with the default 5-stock portfolio at ₹10 lakh, using a headless browser at 1700 px width, dark theme.

| File | What it shows |
| --- | --- |
| `cro-dashboard.png` | Overview: CRO dashboard (pillar rows, bottom line, top risks and actions) |
| `integrated-stress.png` | Integrated Stress: linked vs siloed loss per scenario (chart and table) |
| `liquidity-waterfall.png` | Liquidity: VaR → spread → impact → circuit-lock waterfall |
| `credit-panel.png` | Credit: headline PD/EL/DD/Altman and the by-holding table |
| `case_studies.png` | Case studies: realised loss after each as-of date, red where the tool warned (written by `case_studies/run.py`) |

To refresh them, rerun the app with the same settings and recapture the same sections, then rerun `case_studies/run.py` for the last one.
