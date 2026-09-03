"""姿态关键点后端（黑手套链）：RTMPose hand5 / MediaPipe HandLandmarker。

统一契约（GloveDetector 只依赖这一份契约，后端可热切换）：
    backend(frame_bgr, bboxes=None)
        -> (kpts, scores)
        kpts   : (M, 21, 2) float32 全图像素坐标（21 点 MediaPipe 拓扑，
                 0=腕，与 RTMPose hand5 同序——_handedness 几何判手性直接
                 可用）；全局失败（模型异常）返回 None。
        scores : (M, 21) float32 或 None。RTMPose=SimCC 逐点响应均值；
                 MediaPipe=逐点 visibility（0-1）。下游只取每手均值对照
                 pose_conf_thr 门，语义近似即可。
        bboxes : [(x1,y1,x2,y2)]，省略则整图。
    backend.close()   释放底层会话（重建前必调；重复 close 安全）。
    backend.device    实际推理设备（"cuda"/"cpu"，初始化回退后可能变）。

MediaPipe 后端要点：
  - venv 的 mediapipe 1.0.0 无 legacy mp.solutions（模块与 TFLite 图均
    移除），必须走 Tasks API：`import mediapipe` 后
    `from mediapipe.tasks import python as mp_python`（mediapipe/tasks/
    __init__ 是空壳，`from mediapipe.tasks import BaseOptions` 会失败）。
  - 模型 = 仓库 models/hand_landmarker.task（Tasks API 需显式 .task）。
  - 整图 num_hands=2 检测 + 质心就近关联到 bboxes：MediaPipe 掌部
    检测器需要整图上下文——实测（000005 f18/20/43/45/52）框外扩
    1.25 裁剪喂进去 0/5 检出、整图 5/5 检出且 21/21 点落在高置信
    框内。输出仍是每 bbox 一行（顺序与入参一致），未关联到的框吐
    零行；HandTracker 语义（track_id 身份/运动门控/匹配）完全不变。
  - 零行必中 _degenerate 钳边/聚团 → 冻结兜底（全 NaN 反而会绕过
    unique/span 检查——NaN 互相不相等）。
  - 已知极限：MediaPipe 掌部检测器对黑手套检出率低（000005 整图
    5/60 帧命中 vs world 每帧出框），黑手套场景预期差于 RTMPose；
    本后端主要供裸手/效果对比用。
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

_RTMPOSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                "onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-"
                "74fb594_20230320.zip")
_DEFAULT_TASK = os.path.join(_REPO_ROOT, "tools", "models", "hand_landmarker.task")

_MP_PAD = 1.25      # 裁剪框外扩（与 rtmpose bbox_xyxy2cs padding 同口径）
_MP_MIN_CONF = 0.3  # HandLandmarker 检测/存在阈值（默认 0.5；放宽后由
                    # 下游退化过滤 + pose_conf 门兜底，保召回）


class RtmposePoseBackend:
    """RTMPose hand5（onnxruntime，SIMCC 256x256）。

    首次构造若模型未缓存会从 openmmlab 下载（~几十 MB）。CUDA EP 失败
    自动回退 CPU（与旧 GloveDetector 内联逻辑同语义）。
    """

    name = "rtmpose"

    def __init__(self, device: str = "cpu"):
        try:
            from rtmlib import RTMPose
            self._pose = RTMPose(_RTMPOSE_URL, model_input_size=(256, 256),
                                 backend="onnxruntime", device=device)
        except Exception as e:      # ORT CUDA EP 失败等 → CPU 兜底
            print(f"RTMPose {device} 初始化失败（{e}），回退 CPU")
            from rtmlib import RTMPose
            self._pose = RTMPose(_RTMPOSE_URL, model_input_size=(256, 256),
                                 backend="onnxruntime", device="cpu")
            device = "cpu"
        self.device = device

    def __call__(self, frame_bgr, bboxes=None
                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._pose(frame_bgr, bboxes=bboxes)

    def close(self):
        self._pose = None


class MediaPipePoseBackend:
    """MediaPipe HandLandmarker（Tasks API，整图 num_hands=2 + 框关联）。"""

    name = "mediapipe"

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cpu"):
        # 惰性 import：rtmpose 默认路径不承担 mediapipe 的导入开销。
        # Tasks API 导入模式见模块 docstring（venv mediapipe 1.0.0 特有）。
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        self._mp_python = mp_python
        self._vision = vision

        self.model_path = model_path or _DEFAULT_TASK
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)

        def _make(delegate):
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=self.model_path, delegate=delegate),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=_MP_MIN_CONF,
                min_hand_presence_confidence=_MP_MIN_CONF,
                min_tracking_confidence=_MP_MIN_CONF,
            )
            return vision.HandLandmarker.create_from_options(options)

        self._lm = _make(mp_python.BaseOptions.Delegate.CPU)
        if device == "cuda":
            # GPU delegate 在无可用 OpenGL 上下文时往往不是构造期报错而是
            # 首次推理期报错 → 冒烟检测兜底（同 --mp-delegate 口径）。
            try:
                lm = _make(mp_python.BaseOptions.Delegate.GPU)
                _smoke(lm, self._mp)
                self._lm.close()
                self._lm = lm
            except Exception as e:
                print(f"HandLandmarker GPU 初始化失败（{e}），回退 CPU")
                device = "cpu"
        self.device = device

    def __call__(self, frame_bgr, bboxes=None
                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        h, w = frame_bgr.shape[:2]
        if not bboxes:
            bboxes = [[0.0, 0.0, float(w), float(h)]]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect(self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb))
        kpts = np.zeros((len(bboxes), 21, 2), np.float32)
        scores = np.zeros((len(bboxes), 21), np.float32)
        if not res.hand_landmarks:
            return kpts, scores     # 全零行：下游 _degenerate → 冻结兜底

        # 整图检出手（归一化坐标）→ 全图像素 + 质心
        mp_hands = []
        for lm in res.hand_landmarks:
            pts = np.array([[float(lm[j].x) * w, float(lm[j].y) * h]
                            for j in range(21)], np.float32)
            vis = np.array([float(lm[j].visibility)
                            if lm[j].visibility is not None
                            else float(lm[j].presence)
                            if lm[j].presence is not None else 1.0
                            for j in range(21)], np.float32)
            mp_hands.append((pts, vis))
        used = [False] * len(mp_hands)

        # 质心就近关联（贪心）：每框找质心落在框外扩 _MP_PAD 内的最近
        # 未占用手；不匹配 → 该框零行。掌检器整图上下文召回远高于裁剪
        # （000005 实测 0/5 vs 5/5），关联后输出契约与 RTMPose 完全一致。
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            x1, y1 = float(x1), float(y1)
            x2, y2 = float(x2), float(y2)
            bw, bh = x2 - x1, y2 - y1
            pad = max(bw, bh) * (_MP_PAD - 1.0) / 2.0   # 每侧外扩（同旧裁剪口径）
            best_j, best_d = -1, float("inf")
            for j, (pts, _vis) in enumerate(mp_hands):
                if used[j]:
                    continue
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                if cx < x1 - pad or cx > x2 + pad \
                        or cy < y1 - pad or cy > y2 + pad:
                    continue       # 质心不在框外扩范围内 → 不属此框
                d = (cx - (x1 + x2) / 2.0) ** 2 + (cy - (y1 + y2) / 2.0) ** 2
                if d < best_d:
                    best_j, best_d = j, d
            if best_j >= 0:
                used[best_j] = True
                kpts[i] = mp_hands[best_j][0]
                scores[i] = mp_hands[best_j][1]
        return kpts, scores

    def close(self):
        if self._lm is not None:
            try:
                self._lm.close()
            except Exception:
                pass    # mediapipe 1.0.0 关闭期已知噪声（NoneType
                        # dispatcher），热切换时不允许中断
        self._lm = None


def _smoke(lm, mp) -> None:
    """GPU delegate 冒烟：跑一帧极小图，异常即抛出（调用方回退 CPU）。"""
    dummy = np.zeros((64, 64, 3), np.uint8)
    lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy))
