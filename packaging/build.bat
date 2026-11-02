@echo off
REM ============================================================
REM DroneVision Windows EXE 打包脚本
REM 使用前提：已激活 yolov8_pyqt5 Conda 环境
REM 运行方法：双击本脚本，或在终端中执行
REM 输出位置：dist\DroneVision\DroneVision.exe
REM ============================================================

setlocal

REM 切换到项目根目录（脚本位于 packaging/ 子目录）
cd /d "%~dp0.."

echo [1/3] 清理上次构建产物...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo [2/3] 开始打包（onedir 模式，无控制台）...
python -m PyInstaller DroneVision.spec ^
    --noconfirm ^
    --clean ^
    --log-level WARN

if errorlevel 1 (
    echo.
    echo [错误] 打包失败，请查看上方错误信息。
    pause
    exit /b 1
)

echo [3/3] 打包完成！
echo.
echo   EXE 位置：%CD%\dist\DroneVision\DroneVision.exe
echo.
echo 提示：
echo   - 权重文件位于 dist\DroneVision\_internal\models\（由 spec 自动复制）
echo   - 检测结果默认保存至 %%USERPROFILE%%\Documents\DroneVision\outputs\
echo.
pause
