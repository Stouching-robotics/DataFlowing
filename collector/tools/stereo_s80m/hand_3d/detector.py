#!/usr/bin/env python3
"""
2D 关键点检测抽象层 —— 几何层唯一依赖的接口。

`KeypointDetector.detect()` 返回 21 点像素关键点（MediaPipe 拓扑序），
几何三角化、两阶段精修、渲染全部只依赖这个接口，因此 HaMeR 等
GPU 神经检测器（gpu_hamer.py）可以作为可替换后端接入。

MediaPipeDetector 包装 hand_detection 现有管线，但关键点用
norm_landmarks × 帧尺寸重算为 **float 亚像素坐标**（现有管线
landmarks 是 int 截断，0.5px 量化对 ~3.8px 级重投影误差是可见
噪声源，对裁剪图精修（放大回全图）尤其重要）。
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hand_detection.hand_pipeline_mediapipe import MediaPipeHandPipeline  # noqa: E402


@dataclass
class DetectedHand:
    """单只手的 2D 检测结果（几何层消费的最小结构）。"""

    landmarks: np.ndarray          # (21,2) float32 亚像素像素坐标
    label: str = "Hand"            # "Left" / "Right"
    score: float = 0.0
    index: int = 0
    conf: np.ndarray | None = None  # (21,) 逐点置信度（批量检测器才有；MediaPipe 为 None）

    @classmethod
    def from_hand_result(cls, hr, frame_w: int, frame_h: int, index: int = 0) -> "DetectedHand":
        """从 MediaPipe HandResult 转换：norm_landmarks × 尺寸 → float 亚像素。"""
        pts = np.asarray(hr.norm_landmarks, dtype=np.float64).reshape(-1, 2)
        return cls(landmarks=(pts * [frame_w, frame_h]).astype(np.float32),
                   label=hr.label, score=hr.score, index=index)


class KeypointDetector(ABC):
    """每帧每手 21×2 像素关键点检测器接口。"""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list:
        """处理一帧 BGR 图，返回 DetectedHand 列表（按检测顺序）。"""
        ...

    def reset(self) -> None:
        """切换视频源/画面跳变时重置追踪状态。"""

    def close(self) -> None:
        """释放资源。"""


class MediaPipeDetector(KeypointDetector):
    """MediaPipe HandLandmarker 检测器（float 亚像素输出）。

    delegate="cpu" 走共享管线 MediaPipeHandPipeline（原路径）；
    delegate="gpu" 走 hand_3d.mp_gpu.FastHandLandmarker（GPU delegate，
    3.0ms/帧 vs CPU 7.8ms；注意 GPU 时两目检测必须同线程顺序，
    且输出与 CPU 有 ~2.8px 级 fp16 数值漂移）。
    """

    def __init__(self, model_path: str = None, num_hands: int = 2,
                 mirror: bool = False, smooth: bool = True,
                 freq_min: float = 5.0, beta: float = 0.05, dcutoff: float = 1.0,
                 det_conf: float = 0.5, track_conf: float = 0.5,
                 delegate: str = "cpu"):
        if model_path is None:
            model_path = os.path.join(_REPO_ROOT, "tools", "models",
                                      "hand_landmarker.task")
        self.num_hands = num_hands
        self.delegate = delegate
        if delegate == "gpu":
            # 惰性 import：mp_gpu 每次调用都新建 mediapipe 实例，venv 无
            # mediapipe 的解释器只有真用 GPU 时才炸
            from stereo_s80m.hand_3d.mp_gpu import FastHandLandmarker
            self._gpu = FastHandLandmarker(
                model_path=model_path, num_hands=num_hands,
                det_conf=det_conf, track_conf=track_conf, delegate="gpu",
                smooth=smooth, freq_min=freq_min, beta=beta, dcutoff=dcutoff)
            self._pipe = None
        else:
            self._pipe = MediaPipeHandPipeline(
                model_path=model_path, num_hands=num_hands,
                det_conf=det_conf, track_conf=track_conf,
                mirror=mirror, smooth=smooth,
                freq_min=freq_min, beta=beta, dcutoff=dcutoff)
            self._gpu = None

    def detect(self, frame_bgr: np.ndarray) -> list:
        if self._gpu is not None:
            return self._gpu.detect(frame_bgr)
        h, w = frame_bgr.shape[:2]
        result = self._pipe.process(frame_bgr)
        out = []
        for i, hr in enumerate(result.hands[: self.num_hands]):
            out.append(DetectedHand.from_hand_result(hr, w, h, index=i))
        return out

    def reset(self) -> None:
        if self._gpu is not None:
            self._gpu.reset()
        else:
            self._pipe.reset()

    def close(self) -> None:
        if self._gpu is not None:
            self._gpu.close()
        else:
            self._pipe.close()
