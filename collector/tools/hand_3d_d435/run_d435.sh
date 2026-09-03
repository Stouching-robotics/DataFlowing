#!/usr/bin/env bash
# hand_3d_d435 D435 RGB-D 3D 手部管线启动器 —— 用 collector/venv
# （mediapipe + cv2 + scipy + pyarrow + pyrealsense2）。
# 用法: ./tools/hand_3d_d435/run_d435.sh <session_dir> [run_pipeline_d435.py 选项...]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="${VENV_PY:-$REPO_ROOT/venv/bin/python}"

cd "$REPO_ROOT"
exec "$VENV_PY" tools/hand_3d_d435/run_pipeline_d435.py "$@"
