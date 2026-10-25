@echo off
cd /d "%~dp0"
where node >nul 2>nul
if %errorlevel% equ 0 (
  node server.mjs
  pause
  exit /b
)
where py >nul 2>nul
if %errorlevel% equ 0 (
  start "" http://127.0.0.1:4173
  py -3 -m http.server 4173 --bind 127.0.0.1
  pause
  exit /b
)
echo Please install Node.js 18+ once before going offline.
pause
