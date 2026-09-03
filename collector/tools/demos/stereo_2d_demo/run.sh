#!/bin/bash
# stereo_2d_demo 一键运行脚本（单目或双目）
set -e
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
    echo "用法: ./run.sh <视频1> [视频2] [-o 输出.mp4] [--no-smooth] [--freq-min 15] [--beta 0.6]"
    echo "示例:"
    echo "  单目: ./run.sh demo.mp4"
    echo "  双目: ./run.sh left.mp4 right.mp4"
    echo "  双目+跟手调优: ./run.sh left.mp4 right.mp4 --freq-min 15 --beta 0.6"
    echo "  关闭平滑:   ./run.sh left.mp4 right.mp4 --no-smooth"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] 找不到 python3，请先安装 Python 3.9+"
    exit 1
fi

# 优先用当前目录的 venv（如部署方自建），否则用系统 python3
if [ -x "venv/bin/python" ]; then
    PY=venv/bin/python
    echo "使用本地 venv: $PY"
else
    PY=python3
fi

echo "运行: $PY stereo_2d_demo.py $*"
exec "$PY" stereo_2d_demo.py "$@"
