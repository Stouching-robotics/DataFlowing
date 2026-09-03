#!/bin/bash
# build.sh — 编译 libfays_cb_bridge.so（回调桥接，ctypes 用 C ABI）。
#
# 不链接 libfays_vikit.so：注册函数运行时经 dlsym(RTLD_DEFAULT) 解析
# （worker 已 RTLD_GLOBAL 加载 vikit）→ SDK 版本无关，换 SDK 无需重编。
# 仅依赖 vendor 头 fays_atrak_types.h（AtrakImage 布局，随本目录携带）。
set -e
cd "$(dirname "$0")"
g++ -O2 -shared -fPIC -std=c++17 -o libfays_cb_bridge.so \
    fays_cb_bridge.cpp -Iinclude -ldl
echo "OK: $(pwd)/libfays_cb_bridge.so"
