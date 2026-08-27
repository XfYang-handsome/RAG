@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
where python >nul 2>nul && (python start.py %*) || (py start.py %*)
pause