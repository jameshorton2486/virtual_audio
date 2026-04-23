@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Missing .venv. Run:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -r requirements.txt
  exit /b 1
)
echo Launching Virtual Audio Control...
echo Fixed input: CABLE Output (VB-Audio Virtual Cable)
.venv\Scripts\python.exe app.py
