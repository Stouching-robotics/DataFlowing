#!/usr/bin/env bash
# DAQ 程序启动脚本 —— 使用项目自带 venv 运行
# 用法: ./run.sh
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    echo "[错误] 未找到 venv，请先创建虚拟环境:"
    echo "  python -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

exec venv/bin/python main.py "$@"
