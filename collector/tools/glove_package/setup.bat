@echo off
chcp 65001 >nul
echo ============================================
echo  手套识别 + 训练环境 一键安装
echo ============================================
echo.

cd /d "%~dp0"

REM 1. 创建虚拟环境
if not exist ".venv\Scripts\python.exe" (
    echo [1/5] 创建虚拟环境 .venv
    python -m venv .venv
    if errorlevel 1 (
        echo 失败：找不到系统 Python，请先安装 Python 3.10+ 并勾选 Add to PATH
        pause
        exit /b 1
    )
) else (
    echo [1/5] 虚拟环境已存在，跳过
)

REM 2. 升级 pip
echo [2/5] 升级 pip
".venv\Scripts\python.exe" -m pip install --upgrade pip -q

REM 3. 安装 CUDA 版 PyTorch（NVIDIA GPU 机器）
echo [3/5] 安装 CUDA 版 PyTorch（约 2.5GB，请耐心等待）
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo PyTorch CUDA 版安装失败，尝试 CPU 版...
    ".venv\Scripts\python.exe" -m pip install torch torchvision
)

REM 4. 安装其余依赖
echo [4/5] 安装其余依赖
".venv\Scripts\python.exe" -m pip install -r requirements.txt

REM 5. 打 clip 兼容补丁
echo [5/5] 修复 openai-clip 兼容性
".venv\Scripts\python.exe" fix_clip.py

echo.
echo ============================================
echo  安装完成！
echo.
echo  启动识别:
echo    .venv\Scripts\python.exe hand_demo_mmpose.py --prompt glove
echo.
echo  首次运行会自动下载模型（约 450MB，需要联网）:
echo    - YOLO-World 检测器 yolov8m-worldv2.pt (~55MB)
echo    - RTMPose 关键点模型 (~56MB)
echo    - CLIP 文本编码权重 (~340MB)
echo ============================================
pause
