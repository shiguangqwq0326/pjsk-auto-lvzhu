@echo off
setlocal DisableDelayedExpansion

set "LVZHU_PYTHON=%~dp0.venv_local\Scripts\python.exe"
if not exist "%LVZHU_PYTHON%" set "LVZHU_PYTHON=%~dp0.venv312\Scripts\python.exe"
if not exist "%LVZHU_PYTHON%" (
    echo Python environment not found. See README.md.
    if not "%LVZHU_NO_PAUSE%"=="1" pause
    exit /b 1
)

if "%~1"=="" (
    "%LVZHU_PYTHON%" "%~dp0run_one.py"
) else (
    "%LVZHU_PYTHON%" "%~dp0run_one.py" "%~1"
)
set "LVZHU_EXIT=%ERRORLEVEL%"
if not "%LVZHU_NO_PAUSE%"=="1" pause
exit /b %LVZHU_EXIT%
