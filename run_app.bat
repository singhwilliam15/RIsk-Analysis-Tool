@echo off
title Risk Analysis Tool Launcher
cd /d "%~dp0"
echo Starting Risk Analysis Tool...
python -m streamlit run app.py --server.port 8501
pause
