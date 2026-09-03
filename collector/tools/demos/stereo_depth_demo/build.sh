#!/bin/bash
#
# 构建 S80M 双目深度 Demo (严格按 SDK 实现)
#
# 与官方 stereo_depth_gui/build.sh 相同的构建方式, 额外:
#   -DOpenCV_DIR 指向 SDK 自带 OpenCV 4.2.0 (深度引擎按其编译, 必须对接 4.2)
#
# 用法:
#   ./build.sh                    # 自动检测 SDK (需设置 SDK_DIR 或 FAYSSENSE_SDK_DIR)
#   SDK_DIR=/path/to/sdk ./build.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

if [ -z "$SDK_DIR" ]; then
    SDK_DIR="${FAYSSENSE_SDK_DIR:-}"
fi
if [ -z "$SDK_DIR" ]; then
    echo "ERROR: 请设置 SDK_DIR 或 FAYSSENSE_SDK_DIR=<FaysSense VI Kit Release 目录>" >&2
    exit 1
fi
OPENCV42_DIR="$SDK_DIR/thirdparty/opencv-4.2.0-linux-x86_64"

if [ ! -d "$SDK_DIR/include/fays_atrak" ]; then
    echo "ERROR: SDK not found at $SDK_DIR (set SDK_DIR)" >&2
    exit 1
fi
if [ ! -f "$OPENCV42_DIR/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
    echo "ERROR: SDK bundled OpenCV 4.2 not found at $OPENCV42_DIR" >&2
    exit 1
fi

echo "SDK_DIR: $SDK_DIR"
echo "OpenCV : $OPENCV42_DIR (SDK bundled 4.2)"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSDK_DIR="$SDK_DIR" \
    -DOpenCV_DIR="$OPENCV42_DIR/lib/cmake/opencv4" \
    "$@"

make -j"$(nproc)"

echo ""
echo "===== 构建完成 ====="
echo "二进制: $BUILD_DIR/stereo_depth_demo"
echo ""
echo "运行 (自动设置 LD_LIBRARY_PATH 为 SDK 自带 OpenCV 4.2):"
echo "  ./run.sh"
