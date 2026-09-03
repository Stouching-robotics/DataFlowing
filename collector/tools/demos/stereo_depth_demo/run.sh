#!/bin/bash
#
# 运行 S80M 双目深度 Demo (严格按 SDK 实现)
#
# LD_LIBRARY_PATH 按官方 GUI 相同方式设置:
#   - SDK 自带 OpenCV 4.2 的 lib406 兼容目录 (深度引擎必须绑定 4.2)
#   - SDK 官方库目录 (libfays_vikit.so / libfayssense_aikit_depth.so)
#
# 用法:
#   ./run.sh [viKitConfig] [depthConfig]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="${SDK_DIR:-${FAYSSENSE_SDK_DIR:-}}"
if [ -z "$SDK_DIR" ]; then
    echo "[ERROR] 请设置 SDK_DIR 或 FAYSSENSE_SDK_DIR=<FaysSense VI Kit Release 目录>" >&2
    exit 1
fi

LIB406_DIR="$SDK_DIR/thirdparty/opencv-4.2.0-linux-x86_64/lib406"
SDK_LIB_DIR="$SDK_DIR/lib/fays_atrak/$(uname -m)/Release"

if [ ! -d "$LIB406_DIR" ]; then
    echo "[ERROR] 找不到 SDK 自带 OpenCV 4.2 目录: $LIB406_DIR" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$LIB406_DIR:$SDK_LIB_DIR:$LD_LIBRARY_PATH"
exec "$SCRIPT_DIR/build/stereo_depth_demo" "$@"
