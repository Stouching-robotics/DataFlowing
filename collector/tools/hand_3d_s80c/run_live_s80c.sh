#!/usr/bin/env bash
# S80C 双目鱼眼实时 3D 手部关键点 demo 启动器。
# 用法:
#   ./tools/hand_3d_s80c/run_live_s80c.sh                  # 直连相机
#   ./tools/hand_3d_s80c/run_live_s80c.sh --no-window --stats
#   ./tools/hand_3d_s80c/run_live_s80c.sh --glove          # 黑手套模式
#
# 深度引擎 .so 无 DT_NEEDED、靠运行时符号表解析 SDK 自带 OpenCV 4.2，
# 但 worker 子进程已用 ctypes RTLD_GLOBAL 预载（见 s80c_depth_worker.py），
# 主进程只需 venv python。LD_LIBRARY_PATH 保留为双保险（对 vikit 库的
# 间接依赖生效）。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="${VENV_PY:-$REPO_ROOT/venv/bin/python}"

export LD_LIBRARY_PATH="$SCRIPT_DIR/third_party/opencv4.2/lib406:${LD_LIBRARY_PATH:-}"

cd "$REPO_ROOT"
exec "$VENV_PY" tools/hand_3d_s80c/live_demo_s80c.py "$@"
