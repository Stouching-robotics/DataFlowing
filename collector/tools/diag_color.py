#!/usr/bin/env python3
"""诊断: S80M 子进程 pipe 输出的一帧，颜色通道是否正确。

抓一帧同时存两份 PNG:
  /tmp/s80m_left_BGR.png  — 解码后原样 (BGR 布局)
  /tmp/s80m_left_RGB.png  — 转成 RGB 布局

用图片查看器打开，看哪张颜色正常 (如红色物体显示为红色)。
"""
import ctypes
import struct
import subprocess
import sys
import numpy as np
import cv2
import os

SDK_DIR = os.environ.get("FAYSSENSE_SDK_DIR", "")
STERE_DEMO = os.path.join(SDK_DIR, "read_stereo_rgb.py") if SDK_DIR else ""
if not SDK_DIR or not os.path.isfile(STERE_DEMO):
    print("[ERROR] 请设置 FAYSSENSE_SDK_DIR=<FaysSense VI Kit Release 目录>")
    sys.exit(1)
VENV_PY = sys.executable  # 本脚本须在 DAQ venv 下运行

proc = subprocess.Popen(
    [VENV_PY, STERE_DEMO, "--pipe", "-"],
    cwd=SDK_DIR,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)

try:
    fifo = proc.stdout
    # 等第一帧
    for i in range(50):
        header = fifo.read(4)
        if len(header) < 4:
            print(f"[{i}] 无数据, 子进程已退出 code={proc.poll()}")
            sys.exit(1)
        left_len = struct.unpack(">I", header)[0]
        left_data = fifo.read(left_len)
        if len(left_data) < left_len:
            print("帧不完整")
            sys.exit(1)

        header = fifo.read(4)
        if len(header) < 4:
            print("右目头缺失")
            sys.exit(1)
        right_len = struct.unpack(">I", header)[0]
        right_data = fifo.read(right_len)

        left = cv2.imdecode(np.frombuffer(left_data, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(right_data, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            continue

        # 分割上下半 (与 GUI 一致)
        if left.shape[0] > left.shape[1]:
            mid = left.shape[0] // 2
            l_top, l_bot = left[:mid], left[mid:]
        else:
            l_top, l_bot = left, None

        # BGR 原样保存
        cv2.imwrite("/tmp/s80m_left_BGR.png", l_top)
        # RGB 转换保存
        cv2.imwrite("/tmp/s80m_left_RGB.png", cv2.cvtColor(l_top, cv2.COLOR_BGR2RGB))

        print("已保存:")
        print("  /tmp/s80m_left_BGR.png  (OpenCV 原样, BGR 布局)")
        print("  /tmp/s80m_left_RGB.png  (BGR→RGB 转换后)")
        print("请用图片查看器打开两张，看哪张颜色正常。")
        break
finally:
    proc.terminate()
