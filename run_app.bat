@echo off
title VaR Risk Management Tool Launcher
cd /d "%~dp0"
echo Starting VaR Automated Analysis Tool...
python -m streamlit run app.py --server.port 8501
pause
