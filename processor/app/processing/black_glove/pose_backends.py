"""Local pose backends used by the bundled black-glove detector.

This is the self-contained equivalent of the hand_3d_d435 pose backend
contract.  It deliberately resolves the RTMPose model and MediaPipe task
from the runtime/package itself; it never imports the legacy tools directory.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
_RTMPOSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                "onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-"
                "74fb594_20230320.zip")
_DEFAULT_TASK = os.path.join(_PACKAGE_ROOT, "hand_landmarker.task")
_MP_PAD = 1.25
_MP_MIN_CONF = 0.3


class RtmposePoseBackend:
    """RTMPose hand5 SIMCC backend with the canonical call contract."""

    name = "rtmpose"

    def __init__(self, device: str = "cpu"):
        try:
            from rtmlib import RTMPose
            self._pose = RTMPose(_RTMPOSE_URL, model_input_size=(256, 256),
                                 backend="onnxruntime", device=device)
        except Exception as exc:
            print(f"RTMPose {device} 初始化失败（{exc}），回退 CPU")
            from rtmlib import RTMPose
            self._pose = RTMPose(_RTMPOSE_URL, model_input_size=(256, 256),
                                 backend="onnxruntime", device="cpu")
            device = "cpu"
        self.device = device

    def __call__(self, frame_bgr, bboxes=None):
        return self._pose(frame_bgr, bboxes=bboxes)

    def close(self):
        self._pose = None


class MediaPipePoseBackend:
    """MediaPipe Tasks hand landmarker with bbox-to-hand association."""

    name = "mediapipe"

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cpu"):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        self.model_path = model_path or _DEFAULT_TASK
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)

        def make(delegate):
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

        self._lm = make(mp_python.BaseOptions.Delegate.CPU)
        if device == "cuda":
            try:
                candidate = make(mp_python.BaseOptions.Delegate.GPU)
                _smoke(candidate, mp)
                self._lm.close()
                self._lm = candidate
            except Exception as exc:
                print(f"HandLandmarker GPU 初始化失败（{exc}），回退 CPU")
                device = "cpu"
        self.device = device

    def __call__(self, frame_bgr, bboxes=None):
        h, w = frame_bgr.shape[:2]
        if not bboxes:
            bboxes = [[0.0, 0.0, float(w), float(h)]]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._lm.detect(self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb))
        kpts = np.zeros((len(bboxes), 21, 2), np.float32)
        scores = np.zeros((len(bboxes), 21), np.float32)
        if not result.hand_landmarks:
            return kpts, scores

        detected = []
        for landmarks in result.hand_landmarks:
            pts = np.array([[float(landmarks[j].x) * w,
                             float(landmarks[j].y) * h]
                            for j in range(21)], np.float32)
            vis = np.array([
                float(landmarks[j].visibility)
                if landmarks[j].visibility is not None
                else float(landmarks[j].presence)
                if landmarks[j].presence is not None else 1.0
                for j in range(21)
            ], np.float32)
            detected.append((pts, vis))

        used = [False] * len(detected)
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad = max(bw, bh) * (_MP_PAD - 1.0) / 2.0
            best_j, best_d = -1, float("inf")
            for j, (pts, _vis) in enumerate(detected):
                if used[j]:
                    continue
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                if (cx < x1 - pad or cx > x2 + pad
                        or cy < y1 - pad or cy > y2 + pad):
                    continue
                d = ((cx - (x1 + x2) / 2.0) ** 2
                     + (cy - (y1 + y2) / 2.0) ** 2)
                if d < best_d:
                    best_j, best_d = j, d
            if best_j >= 0:
                used[best_j] = True
                kpts[i] = detected[best_j][0]
                scores[i] = detected[best_j][1]
        return kpts, scores

    def close(self):
        if self._lm is not None:
            try:
                self._lm.close()
            except Exception:
                pass
        self._lm = None


def _smoke(landmarker, mp) -> None:
    dummy = np.zeros((64, 64, 3), np.uint8)
    landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=dummy))
