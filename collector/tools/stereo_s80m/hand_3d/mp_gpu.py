#!/usr/bin/env python3
"""
MediaPipe GPU delegate 直连封装 —— stage-1 检测提速。

hand_detection.hand_pipeline_mediapipe 写死 CPU delegate（共享模块不改），
本模块自建 vision.HandLandmarker 支持 BaseOptions.Delegate.GPU，输出
DetectedHand 结构与 MediaPipeDetector 完全一致（几何层零改动）。

实测（2026-08-17，RTX 5090 + mediapipe 1.0.1）：
- GPU delegate 3.0ms/帧 vs CPU 7.8ms/帧（单目）；创建 ~1.1s 一次性成本；
- GPU delegate 双线程反而慢（GL 上下文竞争 0.84×）→ GPU 时两目必须顺序；
- 与 CPU 结果关键点最大差 2.77px（fp16 数值差异，已知漂移记录在案）。

时间戳：RunningMode.VIDEO 要求单调递增，用自增帧号（1.0.1 实测）。

GPU delegate 可能 SIGSEGV（进程内 try/except 拦不住）→ smoke_test_gpu()
用子进程冒烟，通过才切 GPU。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hand_detection.hand_pipeline_mediapipe import OneEuroFilter2D  # noqa: E402
from stereo_s80m.hand_3d.detector import DetectedHand  # noqa: E402

_SMOKE_CODE = r"""
import sys
import numpy as np
sys.path.insert(0, {tools_dir!r})
from stereo_s80m.hand_3d.mp_gpu import FastHandLandmarker
det = FastHandLandmarker(model_path=sys.argv[1], num_hands=2,
                         det_conf=0.5, track_conf=0.5, delegate="gpu", smooth=False)
frame = np.zeros((400, 640, 3), np.uint8)
for _ in range(3):
    hands = det.detect(frame)
det.close()
print("GPU_SMOKE_OK", len(hands))
"""


def smoke_test_gpu(model_path: str, timeout: float = 90.0) -> bool:
    """子进程冒烟：GPU delegate 创建 + 3 帧推理。返回是否可用。

    GPU delegate 初始化失败可能 SIGSEGV——进程内 try/except 拦不住，
    必须子进程隔离。成功输出 GPU_SMOKE_OK 才算通过。
    """
    code = _SMOKE_CODE.format(tools_dir=os.path.join(_REPO_ROOT, "tools"))
    try:
        r = subprocess.run([sys.executable, "-c", code, model_path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        return False
    ok = r.returncode == 0 and "GPU_SMOKE_OK" in r.stdout
    if not ok and r.stderr.strip():
        print(f"  [GPU 冒烟失败，回退 CPU] {r.stderr.strip().splitlines()[-1][:160]}")
    return ok


class FastHandLandmarker:
    """vision.HandLandmarker 直连（可 delegate/阈值），float 亚像素输出。

    2D One-Euro 平滑逻辑与 hand_pipeline_mediapipe 一致（归一化坐标域，
    按 (手序号, 点序号) 维护滤波器状态）。
    """

    def __init__(self, model_path: str, num_hands: int = 2,
                 det_conf: float = 0.5, track_conf: float = 0.5,
                 delegate: str = "cpu",
                 smooth: bool = True, freq_min: float = 5.0,
                 beta: float = 0.05, dcutoff: float = 1.0):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        delegates = {"cpu": mp_python.BaseOptions.Delegate.CPU,
                     "gpu": mp_python.BaseOptions.Delegate.GPU}
        if delegate not in delegates:
            raise ValueError(f"未知 delegate: {delegate}")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path, delegate=delegates[delegate]),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=track_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        self._ts = 0                # 自增帧号时间戳（VIDEO 模式要求单调）
        self._smooth = smooth
        self._freq_min, self._beta, self._dcutoff = freq_min, beta, dcutoff
        self._filters = {}          # (hand_i, kpt_j) → OneEuroFilter2D
        self._t0 = None             # One-Euro 时间基准（真实毫秒，滤波 dt 用）

    def detect(self, frame_bgr: np.ndarray) -> list:
        """一帧 BGR → [DetectedHand]（float 亚像素，MediaPipe 拓扑序）。"""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._ts)
        self._ts += 1
        ts_ms = (time.perf_counter() - self._t0) * 1000.0 if self._t0 is not None \
            else self._set_t0()

        out = []
        for i, lms in enumerate(result.hand_landmarks):
            pts = []
            for j, lm in enumerate(lms):
                x, y = lm.x, lm.y
                if self._smooth:
                    key = (i, j)
                    f = self._filters.get(key)
                    if f is None:
                        f = OneEuroFilter2D(self._freq_min, self._beta, self._dcutoff)
                        self._filters[key] = f
                    x, y = f(x, y, ts_ms)
                pts.append((x * w, y * h))
            label = result.handedness[i][0].category_name if result.handedness else "Hand"
            score = float(result.handedness[i][0].score) if result.handedness else 0.0
            out.append(DetectedHand(landmarks=np.asarray(pts, np.float32),
                                    label=label, score=score, index=i))
        return out

    def _set_t0(self) -> float:
        self._t0 = time.perf_counter()
        return 0.0

    def reset(self) -> None:
        self._filters.clear()
        self._ts = 0
        self._t0 = None

    def close(self) -> None:
        self._landmarker.close()
