@echo off
cd /d %~dp0
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (python --version >nul 2>nul && set PY=python)
if not defined PY (
  echo.
  echo  Python is not installed. Install Python 3.12 from https://www.python.org/downloads/
  echo  and tick "Add python.exe to PATH" on the first installer screen, then run this file again.
  echo.
  pause
  exit /b 1
)
if not exist .venv (%PY% -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
start "" http://127.0.0.1:8000
python server.py
pause
