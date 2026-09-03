#!/usr/bin/env bash
# ============================================================
#  Episode SLAM 轨迹对比分析 - 一键运行脚本 (Linux / macOS)
#  首次运行自动创建 .venv 并安装依赖, 之后直接可用。
# ============================================================
set -e
cd "$(dirname "$0")"

PY=python3
VENV=.venv

if [ ! -d "$VENV" ]; then
    echo "[init] 创建虚拟环境 $VENV ..."
    $PY -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip -q
    "$VENV/bin/pip" install -r requirements.txt -q
    echo "[init] 依赖安装完成"
fi

if [ $# -lt 1 ]; then
    echo "用法: ./run.sh <episode目录1> [<episode目录2> ...] [选项]"
    echo
    echo "示例:"
    echo "  ./run.sh /data/episode_00009.zip.new /data/episode_00009_crf30.zip.new --plot"
    echo "  ./run.sh A B C --ref A --plot        # 指定基准"
    echo "  ./run.sh A B --outdir 输出报告 --plot # 自定义输出目录"
    echo
    echo "全部选项见 使用说明.md"
    exit 1
fi

"$VENV/bin/python" analyze_episode_crf.py "$@"
