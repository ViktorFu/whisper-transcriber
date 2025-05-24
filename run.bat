@echo off
echo Whisper Transcriber 启动器
echo ========================

rem 切换到脚本所在目录
cd /d "%~dp0"

rem 运行Python程序
python -m src.main %*

rem 如果上面失败，尝试直接运行脚本
if errorlevel 1 (
    echo.
    echo 尝试备用启动方式...
    python run.py %*
)

pause 