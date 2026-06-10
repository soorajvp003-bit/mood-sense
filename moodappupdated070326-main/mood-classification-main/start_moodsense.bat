@echo off
setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
set "APP_DIR=%ROOT_DIR%moodapp-newfile-main\moodapp-newfile-main"
set "VENV_BASE=%USERPROFILE%\.venvs"
set "VENV_DIR=%VENV_BASE%\moodsense-py311"
set "BOOTSTRAP_PY="
set "USE_PY_LAUNCHER=0"

if not exist "%APP_DIR%\api\requirements.txt" (
  echo Canonical app not found at "%APP_DIR%".
  exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  if not exist "%VENV_BASE%" mkdir "%VENV_BASE%"
  where py >nul 2>nul
  if not errorlevel 1 (
    set "USE_PY_LAUNCHER=1"
  ) else (
    where python >nul 2>nul
    if not errorlevel 1 (
      set "BOOTSTRAP_PY=python"
    ) else if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
      set "BOOTSTRAP_PY=%LocalAppData%\Programs\Python\Python311\python.exe"
    )
  )

  if "!USE_PY_LAUNCHER!"=="0" if not defined BOOTSTRAP_PY (
    echo Python 3.11 bootstrap executable not found.
    exit /b 1
  )

  echo Creating Python 3.11 environment in "%VENV_DIR%"...
  if "!USE_PY_LAUNCHER!"=="1" (
    py -3.11 -m venv "%VENV_DIR%"
  ) else (
    "!BOOTSTRAP_PY!" -m venv "%VENV_DIR%"
  )
  if errorlevel 1 (
    echo Python 3.11 is required.
    exit /b 1
  )

  "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 exit /b 1

  "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%APP_DIR%\api\requirements.txt"
  if errorlevel 1 exit /b 1
)

if not exist "%APP_DIR%\api\data" mkdir "%APP_DIR%\api\data"

pushd "%APP_DIR%"
"%VENV_DIR%\Scripts\python.exe" api\app.py
popd
