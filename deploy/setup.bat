@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
where python >nul 2>nul && (python setup.py %*) || (py setup.py %*)
if errorlevel 1 (
    echo.
    echo 初始化未完成，请查看上方错误信息。
    pause
)