@echo off
REM Activates the conda environment and launches the main pose-estimation pipeline GUI.
REM Change ENV_NAME below if your environment has a different name.
set "ENV_NAME=sam3d"
title Biomechanical Pose Pipeline
echo =============================================
echo   Biomechanical Pose Estimation Pipeline
echo =============================================
echo.

set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set "CONDA_ROOT=C:\ProgramData\anaconda3"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\AppData\Local\anaconda3"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\AppData\Local\miniconda3"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set "CONDA_ROOT=C:\ProgramData\miniconda3"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set "CONDA_ROOT=C:\miniconda3"

if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
    echo [ERROR] Could not find Anaconda or Miniconda.
    echo Edit CONDA_ROOT in this file to point to your conda install folder.
    pause
    exit /b 1
)

echo Found conda at: %CONDA_ROOT%

call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate %ENV_NAME%

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate the '%ENV_NAME%' environment.
    echo Run:  conda env list   to verify it exists.
    pause
    exit /b 1
)

echo Activated: %ENV_NAME% environment
echo.

echo Launching pipeline GUI...
cd /d "%~dp0"
python main_gui.py

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline exited with an error. See the log above.
)
pause
