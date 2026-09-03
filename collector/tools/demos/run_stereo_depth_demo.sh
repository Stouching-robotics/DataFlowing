#!/bin/bash
#
# S80M 双目深度 Demo 启动器
#
# 深度引擎 libfayssense_aikit_depth.so 按 SDK 自带 OpenCV 4.2.0 编译,
# 系统 OpenCV 4.6/4.13 不兼容 (cv::Exception: Unknown/unsupported array type).
# 通过 LD_LIBRARY_PATH 让 SDK 库绑定到自带 4.2 (lib406 目录).
#
# 用法:
#   ./run_stereo_depth_demo.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="${SDK_DIR:-${FAYSSENSE_SDK_DIR:-}}"
if [ -z "$SDK_DIR" ]; then
    echo "[ERROR] 请设置 SDK_DIR 或 FAYSSENSE_SDK_DIR=<FaysSense VI Kit Release 目录>" >&2
    exit 1
fi
OPENCV406_DIR="$SDK_DIR/thirdparty/opencv-4.2.0-linux-x86_64/lib406"

if [ ! -d "$OPENCV406_DIR" ]; then
    echo "[ERROR] 找不到自带 OpenCV 4.2 目录: $OPENCV406_DIR"
    echo "请先在 SDK 中构建 lib406 (符号链接 4.2 库为 .so.406 命名 + 外部依赖)" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$OPENCV406_DIR:$LD_LIBRARY_PATH"
exec python3 -u "$SCRIPT_DIR/test_stereo_depth_calib.py" "$@"
