@echo off
title LEGAT 2.0 - One-Click Launcher
echo ==============================================================================
echo                      LEGAT 2.0 JUDICIAL INTELLIGENCE
echo                Temporal Knowledge Graph and Precedent Engine
echo ==============================================================================
echo.

set PYTHON_BIN=C:\Users\onkar\Documents\python.exe
if not exist "%PYTHON_BIN%" (
    set PYTHON_BIN=python
)

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 using %PYTHON_BIN%...
start "LEGAT 2.0 - FastAPI Backend Server" cmd /k ""%PYTHON_BIN%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting Vite Frontend on http://localhost:5173...
start "LEGAT 2.0 - React Vite Frontend" cmd /k "npm run dev"

echo.
echo ==============================================================================
echo Both services launched successfully!
echo   * Backend API:  http://127.0.0.1:8000  (Docs: http://127.0.0.1:8000/docs)
echo   * Frontend App: http://localhost:5173
echo ==============================================================================
echo Waiting for servers to initialize...
ping 127.0.0.1 -n 4 >nul
start http://localhost:5173
