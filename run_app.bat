@echo off
title VaR Risk Management Tool Launcher
echo Starting VaR Automated Analysis Tool...
"C:\Users\singh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true
pause
