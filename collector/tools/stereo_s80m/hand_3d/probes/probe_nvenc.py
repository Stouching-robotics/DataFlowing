#!/usr/bin/env python3
"""probe_nvenc: 管道写器冒烟 —— nvenc/libx264 直出 + BrokenPipe latch + 可解码性。

只写 /tmp，不落仓库。用法: python probes/probe_nvenc.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)  # stereo_s80m/hand_detection 已并入 tools/ 命名空间

from stereo_s80m.hand_3d.video_writer import (  # noqa: E402
    create_video_sink, find_ffmpeg, has_nvenc,
)

W, H, FPS, N = 640, 400, 25, 30
frames = [(np.random.default_rng(i).integers(0, 255, (H, W, 3), np.uint8))
          for i in range(N)]

print(f"find_ffmpeg() = {find_ffmpeg()}")
print(f"has_nvenc()   = {has_nvenc()}")

with tempfile.TemporaryDirectory() as td:
    for enc in ("auto", "nvenc", "libx264", "mp4v"):
        out = os.path.join(td, f"t_{enc}.mp4")
        sink = create_video_sink(out, FPS, W, H, encoder=enc)
        ok = sum(1 for f in frames if sink.write(f))
        final = sink.close()
        print(f"{enc:8s}: wrote {ok}/{N} frames, final={final}")
        if final and os.path.isfile(final):
            cap = cv2.VideoCapture(final)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, f0 = cap.read()
            cap.release()
            d = int(np.abs(f0.astype(int) - frames[0].astype(int)).mean()) if ret else -1
            print(f"        → 可解码 {n} 帧, 首帧均差 {d} (>0 有损压缩正常)")
        else:
            print("        → [FAIL] 无输出文件")

    # BrokenPipe latch：非法编码器让 ffmpeg 启动即退出 → write 应 latch 失败
    from stereo_s80m.hand_3d.video_writer import PipeVideoWriter
    sink = PipeVideoWriter(os.path.join(td, "bad.mp4"), FPS, W, H,
                           "h264_bogus_codec", ["-crf", "23"])
    print("BrokenPipe 测试（非法编码器）: ", end="")
    for i in range(5):
        sink.write(frames[i])
    final = sink.close()
    print(f"latch={sink.failed}, final={final}（应为 latch=True, final=None）")
