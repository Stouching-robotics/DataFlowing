#!/usr/bin/env python3
"""
3D 域时序平滑 —— 三角化之后的最终抖动抑制。

对 (2,21,3) 左目相机系 3D 关键点逐点跑 One-Euro 自适应低通（复用
hand_detection.hand_pipeline_mediapipe.OneEuroFilter3D）。平滑放在三角化之后
（米制 3D 域），不在 2D 上做双重平滑：精修 2D 与粗 2D 来源不同，在最终 3D
上统一平滑才能消除两种来源切换时的跳变。

防污染：手槽位 label 变化或"空→有"跳变时重置该槽滤波器，
避免上一只手的状态污染下一只。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hand_detection.hand_pipeline_mediapipe import OneEuroFilter3D  # noqa: E402


class Hand3DSmoother:
    """(2,21,3) 手部 3D 关键点时序 One-Euro 平滑。"""

    def __init__(self, freq_min: float = 3.0, beta: float = 0.3, dcutoff: float = 1.0):
        self.freq_min = freq_min
        self.beta = beta
        self.dcutoff = dcutoff
        self._filters = {}                 # (slot, kpt) → OneEuroFilter3D
        self._prev_labels = [None, None]
        self._prev_present = [False, False]
        self._t0 = time.perf_counter()

    def update(self, hands3d: np.ndarray, labels: list, valids=None) -> np.ndarray:
        """一帧平滑。valids: 每槽有效点数（≥8 视为有手），None 时不校验。

        返回 (2,21,3) float32；无效点保持 NaN（数据诚实，渲染层自行处理）。
        """
        pts = np.asarray(hands3d, dtype=np.float64).reshape(2, 21, 3)
        out = np.full((2, 21, 3), np.nan, dtype=np.float32)
        ts = (time.perf_counter() - self._t0) * 1000.0

        for slot in range(2):
            present = True if valids is None else (valids[slot] if slot < len(valids) else 0) >= 8
            label = labels[slot] if slot < len(labels) else ""
            if label != self._prev_labels[slot] or (present and not self._prev_present[slot]):
                for k in range(21):                       # 防跨手污染
                    self._filters.pop((slot, k), None)
            self._prev_labels[slot] = label
            self._prev_present[slot] = present
            if not present:
                continue
            for k in range(21):
                p = pts[slot, k]
                if not np.all(np.isfinite(p)):
                    continue                              # 无效点不喂滤波器，输出 NaN
                key = (slot, k)
                f = self._filters.get(key)
                if f is None:
                    f = OneEuroFilter3D(self.freq_min, self.beta, self.dcutoff)
                    self._filters[key] = f
                out[slot, k] = f(p[0], p[1], p[2], ts)
        return out
