@echo off
REM ============================================================
REM DroneVision 打包脚本（自动激活 Conda 环境版本）
REM 适用于尚未激活环境的普通终端/cmd
REM ============================================================

setlocal

set "ENV_NAME=yolov8_pyqt5"

REM 从 PATH 查找 Conda，避免写死本机安装目录。
REM 若普通 cmd 找不到 conda.bat，请先使用 Anaconda Prompt，或将其加入 PATH。
set "CONDA_BAT="
for /f "delims=" %%I in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%I"
if not defined CONDA_BAT (
    echo [错误] PATH 中未找到 conda.bat。请从 Anaconda Prompt 运行本脚本。
    pause
    exit /b 1
)

call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
    echo [错误] 无法激活 Conda 环境 %ENV_NAME%
    pause
    exit /b 1
)

REM 切换到项目根目录
cd /d "%~dp0.."

echo 已激活环境：%ENV_NAME%
echo 当前目录：%CD%
echo.

echo [1/3] 清理上次构建产物...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo [2/3] 开始打包...
python -m PyInstaller DroneVision.spec --noconfirm --clean --log-level WARN

if errorlevel 1 (
    echo.
    echo [错误] 打包失败。
    pause
    exit /b 1
)

echo [3/3] 完成！
echo   EXE：%CD%\dist\DroneVision\DroneVision.exe
echo.
pause
