@echo off
cd /d "%~dp0"
echo Starting TradingAgents Web UI...
echo.
echo Open your browser at: http://localhost:8501
echo.
python -m streamlit run webui\app.py --server.headless false
pause
