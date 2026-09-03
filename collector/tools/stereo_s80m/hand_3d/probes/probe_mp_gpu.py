#!/usr/bin/env python3
"""probe_mp_gpu: GPU delegate 冒烟 + 单目计时 + CPU/GPU 关键点差值。

只读不落盘。用法（venv）:
    python probes/probe_mp_gpu.py [video_path]
"""
import os
import sys
import time

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)  # stereo_s80m/hand_detection 已并入 tools/ 命名空间

MODEL = os.path.join(_REPO_ROOT, "tools", "models", "hand_landmarker.task")
VIDEO = (sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(_REPO_ROOT, "data", "recordings", "222", "222_000008",
                      "videos", "stereo_left", "chunk-0000", "stereo_left.mp4"))

from stereo_s80m.hand_3d.mp_gpu import FastHandLandmarker, smoke_test_gpu  # noqa: E402

print(f"smoke_test_gpu → {smoke_test_gpu(MODEL)}")

cap = cv2.VideoCapture(VIDEO)
frames = []
for _ in range(20):
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print(f"frames: {len(frames)} {frames[0].shape[:2]}")

det_g = FastHandLandmarker(MODEL, num_hands=2, delegate="gpu", smooth=False)
det_c = FastHandLandmarker(MODEL, num_hands=2, delegate="cpu", smooth=False)

# 预热 + 计时（各 10 帧）
for det, tag in ((det_g, "gpu"), (det_c, "cpu")):
    for f in frames[:3]:
        det.detect(f)
    t0 = time.perf_counter()
    n_hands = 0
    for f in frames:
        n_hands += len(det.detect(f))
    dt = (time.perf_counter() - t0) / len(frames) * 1000
    print(f"{tag}: {dt:.2f} ms/帧  ({n_hands} 手/20帧)")

# CPU vs GPU 关键点差
diffs = []
for f in frames:
    hg = det_g.detect(f)
    hc = det_c.detect(f)
    for a, b in zip(hg, hc):
        if a.label == b.label:
            diffs.append(float(np.abs(a.landmarks - b.landmarks).max()))
print(f"CPU vs GPU 关键点 max|Δ|: 中位={np.median(diffs):.2f}px  "
      f"p90={np.percentile(diffs, 90):.2f}px  max={max(diffs):.2f}px  (n={len(diffs)})")
det_g.close()
det_c.close()
