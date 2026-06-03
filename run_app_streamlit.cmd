@echo off
chcp 65001 >nul
cd /d "%~dp0"

set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set PATH=C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3\Library\bin;C:\Windows\System32;C:\Windows

"C:\ProgramData\Anaconda3\python.exe" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
