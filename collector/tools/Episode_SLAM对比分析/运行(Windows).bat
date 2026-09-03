@echo off
rem ============================================================
rem  Episode SLAM 轨迹对比分析 - 一键运行脚本 (Windows)
rem  需要已安装 Python 3; 首次运行自动安装 numpy/matplotlib。
rem  用法: 双击后输入 episode 目录, 或拖拽目录到本文件上。
rem ============================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python, 请先安装 Python 3 并勾选 Add to PATH
    pause
    exit /b 1
)

echo [init] 检查/安装依赖 ...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [错误] 依赖安装失败, 请检查网络
    pause
    exit /b 1
)

if "%~1"=="" (
    echo 用法: python analyze_episode_crf.py ^<episode目录1^> [^<episode目录2^> ...] [选项]
    echo 示例: python analyze_episode_crf.py D:\data\episode_00009.zip.new D:\data\episode_00009_crf30.zip.new --plot
    echo.
    echo 直接输入要对比的 episode 目录(用空格隔开), 或拖拽目录到本 bat 上:
    set /p ARGS=episode 目录:
) else (
    set ARGS=%*
)

python analyze_episode_crf.py %ARGS%
echo.
pause
