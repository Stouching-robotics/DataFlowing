#!/usr/bin/env bash
# S80M 双目 RGB 相机读取 —— 独立运行包装（自带 SDK 运行时库）
#
# 用法:
#   ./run.sh                          # 显示模式（需图形环境, Q/Esc 退出, S 截图, R 旋转）
#   ./run.sh --pipe -                 # 帧流模式: 二进制协议写 stdout（print 走 stderr）
#   ./run.sh --pipe out.bin           # 帧流模式: 写文件
#   IMU_HZ=200 ./run.sh --pipe -      # 指定 IMU 采样率（默认 200, 0=关闭）
#
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SDK 动态依赖优先用包内运行时库：
#   runtime/lib            → libft602.so（FTDI 桥接）
#   runtime/opencv4.2/*    → OpenCV 4.2 完整运行库 + tbb/webp ABI shims
#                            （缺 libtbb.so.2/libwebp.so.6 会 OSError 崩溃）
export LD_LIBRARY_PATH="$DIR/runtime/opencv4.2/lib406:$DIR/runtime/opencv4.2/lib:$DIR/runtime/opencv4.2/shims:$DIR/runtime/lib:${LD_LIBRARY_PATH:-}"
exec python3 "$DIR/read_stereo_rgb.py" "$@"
