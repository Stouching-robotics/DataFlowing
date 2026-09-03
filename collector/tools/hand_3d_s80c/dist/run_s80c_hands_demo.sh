#!/usr/bin/env bash
# S80C 双目手部关键点实时 demo 启动器（自包含分发包）。
#   ./run_s80c_hands_demo.sh [参数...]
# venv 在包内 ./venv 时自动使用；其他位置用：
#   VENV_PY=/path/to/venv/bin/python ./run_s80c_hands_demo.sh [参数...]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${VENV_PY:-$SCRIPT_DIR/venv/bin/python}"
if [ ! -x "$VENV_PY" ]; then
    VENV_PY="$(command -v python3 || true)"
    [ -n "$VENV_PY" ] || { echo "错误: 未找到 python3。请先建 venv 并安装依赖（见 使用说明.md）" >&2; exit 1; }
fi
# SDK 深度引擎 .so 靠运行时符号表解析 SDK 自带 OpenCV 4.2，预载其 lib406
export LD_LIBRARY_PATH="$SCRIPT_DIR/tools/hand_3d_s80c/third_party/opencv4.2/lib406:${LD_LIBRARY_PATH:-}"
cd "$SCRIPT_DIR"
exec "$VENV_PY" tools/hand_3d_s80c/live_demo_s80c.py "$@"
