#!/usr/bin/env bash
# D435 实时 3D 手部关键点 demo 启动器（collector/venv，含 pyrealsense2==2.58.3）。
# 用法:
#   ./tools/hand_3d_d435/run_live_d435.sh                    # 直连相机
#   ./tools/hand_3d_d435/run_live_d435.sh --replay data/recordings/222/222_000011
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="${VENV_PY:-$REPO_ROOT/venv/bin/python}"

cd "$REPO_ROOT"
exec "$VENV_PY" tools/hand_3d_d435/live_demo.py "$@"
