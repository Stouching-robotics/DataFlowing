"""MediaPipe 裸手关键点检测管线 —— 供外部程序调用的统一入口。

基于 MediaPipe HandLandmarker (Tasks API)，输出 21 个关键点 + 3D 世界坐标 +
左右手判定 + 关节角度。

依赖: mediapipe, opencv-python, numpy

用法:
    from hand_pipeline_mediapipe import MediaPipeHandPipeline

    pipe = MediaPipeHandPipeline(model_path="tools/models/hand_landmarker.task",
                                  num_hands=2)

    for frame in video_frames:
        result = pipe.process(frame)
        # result.hands: [{id, label, score, landmarks, world_landmarks, angles, extended}]
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── 关键点定义 ──────────────────────────────────────────

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

FINGERS = {
    "Thumb":  ([1, 2, 3, 4],     (255, 128, 0)),
    "Index":  ([5, 6, 7, 8],     (0, 255, 0)),
    "Middle": ([9, 10, 11, 12],  (0, 255, 255)),
    "Ring":   ([13, 14, 15, 16], (255, 0, 255)),
    "Pinky":  ([17, 18, 19, 20], (0, 128, 255)),
}


def _joint_specs():
    specs = []
    for finger, (ids, color) in FINGERS.items():
        a, b, c, d = ids
        names = ("CMC", "MCP", "IP") if finger == "Thumb" else ("MCP", "PIP", "DIP")
        specs.append((finger, names[0], a, 0, b, color))
        specs.append((finger, names[1], b, a, c, color))
        specs.append((finger, names[2], c, b, d, color))
    return specs


JOINT_SPECS = _joint_specs()


def _angle_between(p_prev, p_vertex, p_next):
    v1 = np.array(p_prev, dtype=np.float64) - np.array(p_vertex, dtype=np.float64)
    v2 = np.array(p_next, dtype=np.float64) - np.array(p_vertex, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


# ── One-Euro 滤波器（关键点抖动平滑）─────────────────────

class OneEuroFilter:
    """单值 One-Euro 自适应低通滤波器。

    低速运动时强平滑消除抖动，高速运动时自动放宽以保持响应速度。

    参数
    ----
    freq_min : float
        最低截止频率（Hz）。静止时的平滑强度，越小越平滑。建议 0.5 ~ 2.0。
    beta : float
        速度系数。控制对快速运动的响应速度。建议 0.005 ~ 0.015。
    dcutoff : float
        速度估计的低通截止频率（Hz）。建议 1.0。
    """

    def __init__(self, freq_min=1.0, beta=0.007, dcutoff=1.0):
        self.freq_min = freq_min
        self.beta = beta
        self.dcutoff = dcutoff
        self.reset()

    def reset(self):
        """清空滤波器内部状态。"""
        self._prev_x: Optional[float] = None   # 上一帧滤波值
        self._prev_dx: Optional[float] = None  # 上一帧速度估计
        self._prev_ts: Optional[float] = None  # 上一帧时间戳 (ms)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """根据截止频率和时间步长计算平滑系数 α。"""
        tau = 1.0 / (2.0 * math.pi * cutoff) if cutoff > 1e-9 else 0.0
        return dt / (dt + tau) if tau > 0 else 1.0

    def __call__(self, x: float, ts_ms: float) -> float:
        """对当前帧的值做一次滤波。

        参数
        ----
        x : float
            当前帧原始值。
        ts_ms : float
            当前帧时间戳（毫秒）。

        返回
        ----
        float
            滤波后的值。
        """
        # 首帧直接返回
        if self._prev_x is None or self._prev_ts is None:
            self._prev_x = x
            self._prev_dx = 0.0
            self._prev_ts = ts_ms
            return x

        dt = (ts_ms - self._prev_ts) / 1000.0  # 秒
        if dt <= 1e-9:
            return self._prev_x

        # 第 1 阶段：平滑导数（速度估计）
        dx = (x - self._prev_x) / dt
        alpha_d = self._alpha(self.dcutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._prev_dx

        # 第 2 阶段：自适应截止频率
        fc = self.freq_min + self.beta * abs(dx_hat)
        alpha = self._alpha(fc, dt)
        x_hat = alpha * x + (1.0 - alpha) * self._prev_x

        self._prev_x = x_hat
        self._prev_dx = dx_hat
        self._prev_ts = ts_ms

        return x_hat


class OneEuroFilter2D:
    """2D One-Euro 滤波器，对 (x, y) 分量独立滤波。"""

    def __init__(self, freq_min=1.0, beta=0.007, dcutoff=1.0):
        self._fx = OneEuroFilter(freq_min, beta, dcutoff)
        self._fy = OneEuroFilter(freq_min, beta, dcutoff)

    def reset(self):
        self._fx.reset()
        self._fy.reset()

    def __call__(self, x: float, y: float, ts_ms: float) -> Tuple[float, float]:
        return self._fx(x, ts_ms), self._fy(y, ts_ms)


class OneEuroFilter3D:
    """3D One-Euro 滤波器，对 (x, y, z) 分量独立滤波。"""

    def __init__(self, freq_min=1.0, beta=0.007, dcutoff=1.0):
        self._fx = OneEuroFilter(freq_min, beta, dcutoff)
        self._fy = OneEuroFilter(freq_min, beta, dcutoff)
        self._fz = OneEuroFilter(freq_min, beta, dcutoff)

    def reset(self):
        self._fx.reset()
        self._fy.reset()
        self._fz.reset()

    def __call__(self, x: float, y: float, z: float, ts_ms: float) -> Tuple[float, float, float]:
        return self._fx(x, ts_ms), self._fy(y, ts_ms), self._fz(z, ts_ms)


# ── 预处理 ──────────────────────────────────────────────

class _Preprocessor:
    """可选的图像预处理（灰化、伽马、CLAHE），帮助深色手套场景。"""

    def __init__(self, gamma=0.4, clahe_clip=3.0, clahe_grid=8):
        self.gamma = gamma
        self._lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))

    def apply(self, bgr, mode="none"):
        if mode == "none":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if "gamma" in mode:
            g = cv2.LUT(g, self._lut)
        if "clahe" in mode:
            g = self._clahe.apply(g)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


# ── 输出结构 ────────────────────────────────────────────

class HandResult:
    """单只手的检测结果。"""

    def __init__(self, index: int):
        self.index = index           # 手在 MediaPipe 结果中的序号
        self.label: str = "Hand"     # Left / Right
        self.score: float = 0.0      # 分类置信度
        self.landmarks: np.ndarray = np.zeros((21, 2), dtype=np.float32)   # 像素坐标
        self.norm_landmarks: List[Tuple[float, float]] = []                 # 归一化 0-1
        self.world_landmarks: List[Tuple[float, float, float]] = []         # 3D 世界坐标
        self.angles: Dict[Tuple[str, str], float] = {}                      # 关节角度
        self.extended: List[str] = []                                       # 伸直的手指名


class FrameResult:
    """一帧的完整检测结果。"""

    def __init__(self):
        self.hands: List[HandResult] = []
        self.raw_landmarks: List = []    # MediaPipe 原始 landmarks
        self.raw_world: List = []        # MediaPipe 原始 world_landmarks
        self.raw_handedness: List = []   # MediaPipe 原始 handedness


# ── 管线 ────────────────────────────────────────────────

class MediaPipeHandPipeline:
    """MediaPipe 手部关键点检测管线。

    参数
    ----
    model_path : str
        hand_landmarker.task 模型文件路径。
    num_hands : int
        最多检测手数（默认 2）。
    det_conf : float
        手掌检测器置信度阈值（默认 0.5）。
    track_conf : float
        跟踪置信度阈值（默认 0.5）。
    preprocess_mode : str
        预处理方案: "none" / "gray" / "gray+clahe" / "gray+gamma+clahe"（默认 none）。
    mirror : bool
        是否做左右镜像（默认 True，适配自拍视角）。
    smooth : bool
        是否开启关键点抖动平滑（默认 True）。
    freq_min : float
        One-Euro 滤波器最低截止频率（Hz）。值越大响应越快、平滑越弱。
        静止时 alpha ≈ freq_min/(freq_min + fps*2π)。5.0 时每帧约 50% 新值。
        建议 3.0 ~ 10.0，默认 5.0。
    beta : float
        One-Euro 滤波器速度系数。高速运动时自动提高截止频率= freq_min + beta·|v|。
        注意：2D 坐标速度在归一化空间 (0~1/s)，beta 需较大才有明显效果。
        建议 0.02 ~ 1.0，默认 0.05。
    dcutoff : float
        One-Euro 滤波器速度估计的低通截止频率（Hz），默认 1.0。
    """

    def __init__(
        self,
        model_path: str = os.path.join(os.path.dirname(__file__), "..",
                                       "models", "hand_landmarker.task"),
        num_hands: int = 2,
        det_conf: float = 0.5,
        track_conf: float = 0.5,
        preprocess_mode: str = "none",
        mirror: bool = True,
        smooth: bool = True,
        freq_min: float = 5.0,
        beta: float = 0.05,
        dcutoff: float = 1.0,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe 模型不存在: {model_path}\n"
                "下载: curl -L -o tools/models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task")

        self.mirror = mirror
        self.preprocess_mode = preprocess_mode
        self.smooth = smooth
        self._freq_min = freq_min
        self._beta = beta
        self._dcutoff = dcutoff

        # One-Euro 滤波器状态（按 (手序号, 关键点序号) 索引）
        self._pixel_filters: Dict[Tuple[int, int], OneEuroFilter2D] = {}
        self._world_filters: Dict[Tuple[int, int], OneEuroFilter3D] = {}

        # 构建 landmarker
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=track_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._preprocessor = _Preprocessor()
        self._t0 = time.perf_counter()

    # ── 公共 API ──────────────────────────────────────────

    def process(self, frame: np.ndarray) -> FrameResult:
        """处理一帧 BGR 图像，返回 FrameResult。

        参数
        ----
        frame : np.ndarray
            BGR 图像 (H, W, 3)。

        返回
        ----
        FrameResult
            .hands: [HandResult, ...]  每只手的完整数据
        """
        h, w = frame.shape[:2]

        # 镜像
        if self.mirror:
            frame = cv2.flip(frame, 1)

        # 预处理
        rgb = self._preprocessor.apply(frame, self.preprocess_mode)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - self._t0) * 1000)

        # 推理
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        # 组装结果
        out = FrameResult()
        out.raw_landmarks = result.hand_landmarks
        out.raw_world = result.hand_world_landmarks
        out.raw_handedness = result.handedness

        for i, lms in enumerate(result.hand_landmarks):
            hand = HandResult(i)

            # ── 像素坐标 & 归一化坐标（应用 One-Euro 平滑）──
            if self.smooth:
                px_coords = []
                for j, lm in enumerate(lms):
                    key = (i, j)
                    if key not in self._pixel_filters:
                        self._pixel_filters[key] = OneEuroFilter2D(
                            self._freq_min, self._beta, self._dcutoff)
                    fx, fy = self._pixel_filters[key](lm.x, lm.y, ts_ms)
                    px_coords.append((int(fx * w), int(fy * h)))
                    hand.norm_landmarks.append((fx, fy))
                hand.landmarks = np.array(px_coords, dtype=np.float32)
            else:
                hand.landmarks = np.array(
                    [(int(lm.x * w), int(lm.y * h)) for lm in lms],
                    dtype=np.float32)
                hand.norm_landmarks = [(lm.x, lm.y) for lm in lms]

            # ── 3D 世界坐标（应用 One-Euro 平滑）──
            if i < len(result.hand_world_landmarks):
                if self.smooth:
                    wl_coords = []
                    for j, p in enumerate(result.hand_world_landmarks[i]):
                        key = (i, j)
                        if key not in self._world_filters:
                            self._world_filters[key] = OneEuroFilter3D(
                                self._freq_min, self._beta, self._dcutoff)
                        wl_coords.append(
                            self._world_filters[key](p.x, p.y, p.z, ts_ms))
                    hand.world_landmarks = wl_coords
                else:
                    hand.world_landmarks = [
                        (p.x, p.y, p.z)
                        for p in result.hand_world_landmarks[i]]

            # 左右手
            if i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                hand.label = cat.category_name
                hand.score = cat.score
                if self.mirror:
                    hand.label = {"Left": "Right", "Right": "Left"}.get(
                        hand.label, hand.label)

            # 关节角度（基于 3D 世界坐标，比 2D 图像坐标更稳定）
            if hand.world_landmarks:
                hand.angles = {}
                for finger, joint, vertex, prev_id, next_id, _ in JOINT_SPECS:
                    hand.angles[(finger, joint)] = _angle_between(
                        hand.world_landmarks[prev_id],
                        hand.world_landmarks[vertex],
                        hand.world_landmarks[next_id])

            # 伸直判断
            hand.extended = []
            for finger in FINGERS:
                if finger == "Thumb":
                    ok = (hand.angles.get((finger, "MCP"), 0) > 145
                          and hand.angles.get((finger, "IP"), 0) > 150)
                else:
                    ok = (hand.angles.get((finger, "PIP"), 0) > 150
                          and hand.angles.get((finger, "DIP"), 0) > 140)
                if ok:
                    hand.extended.append(finger)

            out.hands.append(hand)

        return out

    def reset(self) -> None:
        """重置追踪状态（切换视频源时调用）。"""
        self._landmarker.close()
        self._t0 = time.perf_counter()
        # 清空滤波器状态
        for f in self._pixel_filters.values():
            f.reset()
        for f in self._world_filters.values():
            f.reset()

    def close(self) -> None:
        """释放资源。"""
        self._landmarker.close()

    def __del__(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
