#!/usr/bin/env bash
# 一键构建两个示例: read_depth (CPU 算法库) + engine_depth (SDK 引擎)
# 用法: ./build.sh            # 编译
#       ./build.sh run-cpu    # 编译并运行方式A示例 (处理 data/sample_stacked.bmp)
#       ./build.sh run-engine # 编译并运行方式B示例
set -e

SDK_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SDK_ROOT/build"

echo "== configure =="
cmake -S "$SDK_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
echo "== build =="
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "构建完成:"
echo "  $BUILD_DIR/read_depth    (方式A: CPU 算法库, 依赖系统 OpenCV 4.x)"
echo "  $BUILD_DIR/engine_depth  (方式B: SDK 引擎, 依赖随包 .so, 无需设置 LD_LIBRARY_PATH)"
echo
echo "快速运行:"
echo "  $BUILD_DIR/read_depth   data/sample_stacked.bmp calib/calib.yaml 3 out/v3"
echo "  $BUILD_DIR/engine_depth data/sample_stacked.bmp config/stereo_depth.yaml out/engine"
echo

case "$1" in
    run-cpu)
        echo "== run read_depth (variant 3) =="
        mkdir -p "$SDK_ROOT/out"
        "$BUILD_DIR/read_depth" "$SDK_ROOT/data/sample_stacked.bmp" \
            "$SDK_ROOT/calib/calib.yaml" 3 "$SDK_ROOT/out/v3"
        ;;
    run-engine)
        echo "== run engine_depth =="
        mkdir -p "$SDK_ROOT/out"
        "$BUILD_DIR/engine_depth" "$SDK_ROOT/data/sample_stacked.bmp" \
            "$SDK_ROOT/config/stereo_depth.yaml" "$SDK_ROOT/out/engine"
        ;;
esac
