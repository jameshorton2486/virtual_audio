@echo off
echo Building VirtualAudioControl.exe
if not exist .venv\Scripts\python.exe (
  echo Missing .venv. Run: python -m venv .venv
  exit /b 1
)
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\pyinstaller.exe --onefile --noconsole --name VirtualAudioControl app.py
if exist dist\VirtualAudioControl.exe (
  copy /Y nircmd.exe dist\ >nul
  copy /Y config.json dist\ >nul
  echo.
  echo Build complete. Distribute the files in dist\
) else (
  echo Build failed.
)
