#!/usr/bin/env bash
# ============================================================
#  DAQ 数据采集系统 —— Linux 一键部署脚本（与 Windows start.bat 行为一致）
#
#  用法:
#    ./start.sh               部署(按需) + 启动主程序
#    ./start.sh reinstall     删除 venv 强制重装（出问题首选）
#    ./start.sh extras        追加安装 mediapipe / pyrealsense2
#    ./start.sh extras-torch  追加安装 CPU 版 torch
#    ./start.sh help          打开 使用说明.md
#
#  依赖安装顺序: 离线 wheels/ 包 → 阿里云镜像 → 清华镜像 → 官方源
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

MODE=run
FORCE=0
case "${1:-}" in
    reinstall)    FORCE=1 ;;
    extras)       MODE=extras ;;
    extras-torch) MODE=extras-torch ;;
    help|guide)   MODE=help ;;
esac
# 非子命令参数原样透传给 main.py

if [ "$MODE" = help ]; then
    [ -f "使用说明.md" ] && xdg-open "使用说明.md" 2>/dev/null || true
    echo "常用命令: ./start.sh [reinstall|extras|extras-torch|help]"
    echo "English guide: 使用说明_EN.md"
    exit 0
fi

# ── [G] 解压层次自检 ──
if [ ! -f main.py ] || [ ! -f requirements.txt ]; then
    echo "[错误 G] 未找到 main.py —— 目录层次不对。start.sh 与 main.py 必须在同一目录。"
    exit 1
fi

# ── [1/6] 定位 Python（>= 3.10，推荐 3.12）──
PY=""
find_python() {
    for c in python3.12 python3.11 python3.10 python3; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PY="$(command -v "$c")"
            return 0
        fi
    done
    return 1
}
if [ -x venv/bin/python ]; then
    PY="venv/bin/python"
elif ! find_python; then
    echo "[错误 A] 未找到 Python >= 3.10。请先安装:"
    echo "  Ubuntu/Debian:  sudo apt install python3.12 python3.12-venv"
    echo "  CentOS/RHEL:    sudo dnf install python3.12"
    echo "  conda:          conda create -n daq python=3.12"
    exit 1
fi
echo "[1/6] 使用 Python: $PY"

# ── [2/6] venv ──
if [ -x venv/bin/python ] && [ "$FORCE" = 1 ]; then
    echo "[2/6] reinstall: 删除旧 venv ..."
    rm -rf venv
fi
if [ ! -x venv/bin/python ]; then
    echo "[2/6] 创建虚拟环境 venv（首次约 1 分钟）..."
    "$PY" -m venv venv
fi
VPY="venv/bin/python"

# ── [3/6] 依赖 ──
HASH="$(md5sum requirements.txt | cut -d' ' -f1)"
NEED_INSTALL=0
if [ "$FORCE" = 1 ]; then
    NEED_INSTALL=1
elif [ ! -f venv/.deps-ok ] || [ "$(cat venv/.deps-ok)" != "$HASH" ]; then
    NEED_INSTALL=1
fi

install_req() {
    local req="$1"
    if [ -d wheels ] && ls wheels/*.whl >/dev/null 2>&1; then
        echo "[3/6] 检测到 wheels/ 离线包，优先离线安装 ..."
        "$VPY" -m pip install --no-index --find-links wheels -r "$req" && return 0
        echo "[3/6] 离线包安装失败，转在线安装 ..."
    fi
    "$VPY" -m pip install -r "$req" -i https://mirrors.aliyun.com/pypi/simple/ && return 0
    "$VPY" -m pip install -r "$req" -i https://pypi.tuna.tsinghua.edu.cn/simple && return 0
    "$VPY" -m pip install -r "$req"
}

if [ "$NEED_INSTALL" = 1 ]; then
    echo "[3/6] 安装依赖（首次约 3-10 分钟，之后启动秒开）..."
    "$VPY" -m pip install --upgrade pip >/dev/null 2>&1 || true
    if ! install_req requirements.txt; then
        echo "[错误 D] 依赖下载/安装失败: 检查网络；内网环境请用 scripts/pack_wheels.py 生成 wheels/ 离线包。"
        exit 1
    fi
    echo "$HASH" > venv/.deps-ok
fi

# ── [4/6] 冒烟自检 ──
echo "[4/6] 依赖自检 ..."
if ! "$VPY" -c "import main" >/dev/null 2>&1; then
    echo "[错误 E] 依赖自检失败。查看具体原因:"
    echo "  venv/bin/python -c \"import main\""
    echo "重装: ./start.sh reinstall"
    exit 1
fi
echo "[4/6] 依赖自检通过"

# ── [5/6] 可选功能 ──
install_pkg() {
    if [ -d wheels ] && ls wheels/*.whl >/dev/null 2>&1; then
        "$VPY" -m pip install --no-index --find-links wheels "$1" && return 0
    fi
    "$VPY" -m pip install "$1" -i https://mirrors.aliyun.com/pypi/simple/ && return 0
    "$VPY" -m pip install "$1" -i https://pypi.tuna.tsinghua.edu.cn/simple && return 0
    "$VPY" -m pip install "$1"
}

case "$MODE" in
    extras)
        [ -f venv/.extras-ok ] || {
            echo "[5/6] 安装可选功能: mediapipe + pyrealsense2 ..."
            install_pkg mediapipe || echo "  [警告] mediapipe 安装失败（主程序不受影响）"
            install_pkg pyrealsense2 || echo "  [警告] pyrealsense2 安装失败（主程序不受影响）"
            echo done > venv/.extras-ok
        }
        echo "[5/6] 可选功能安装完成。运行 ./start.sh 启动主程序。"
        exit 0 ;;
    extras-torch)
        [ -f venv/.torch-ok ] || {
            echo "[5/6] 安装可选功能: torch CPU 版 ..."
            install_torch() {
                if [ -d wheels ] && ls wheels/*.whl >/dev/null 2>&1; then
                    "$VPY" -m pip install --no-index --find-links wheels torch && return 0
                fi
                "$VPY" -m pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu/ && return 0
                "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
            }
            install_torch || echo "  [警告] torch 安装失败（主程序不受影响）"
            echo done > venv/.torch-ok
        }
        echo "[5/6] 可选功能安装完成。运行 ./start.sh 启动主程序。"
        exit 0 ;;
esac

# ── [6/6] 启动 ──
echo "[6/6] 启动主程序 ..."
echo
echo "【操作指引】"
echo "  · 设备面板: 相机插入后约 2 秒自动出现，点击即可预览"
echo "  · 录制: 每路相机独立的 开始/停止；正常停止=保存，异常停止=丢弃"
echo "  · 任务/回放/上传: 见主界面与 使用说明.md"
echo
# 标记本次为 start.sh 启动 → 主程序弹出使用步骤窗口（可勾选不再显示）
DAQ_SHOW_GUIDE=1 exec "$VPY" main.py "$@"
