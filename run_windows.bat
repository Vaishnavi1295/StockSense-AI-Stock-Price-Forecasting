@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo   StockSense AI - Windows setup and run
echo ==========================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher was not found.
  echo Install Python 3.11 64-bit from https://www.python.org/downloads/release/python-3119/
  echo During installation, select "Add Python to PATH".
  pause
  exit /b 1
)
py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 was not found.
  echo Please install Python 3.11 64-bit, then run this file again.
  echo Newer Python versions may not have compatible TensorFlow wheels yet.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 goto error
)
echo Installing or updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
echo Starting StockSense AI...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto error
goto end
:error
echo.
echo Setup or launch failed. Check the error above and README.md.
pause
exit /b 1
:end
endlocal
