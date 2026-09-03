#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d435_hands_demo.py —— D435 录制会话 3D 手部关键点后处理 Demo（自包含单文件）。

对主程序 D435 录制会话（RGB 视频 + 深度 PNG + 标定 JSON）离线批处理，
输出三路处理视频：

  1_rgb_2d_overlay.mp4    RGB + 2D 五指分色骨架 + 每点深度 mm 标注 + 手标签 HUD
                          （--depth-overlay 时叠 300-1200mm 深度伪彩）
  2_hand_3d.mp4           3D 关键点可视化：静态正面视角（视角全程不动，能看全
                          手部关键点），含地面网格/坐标轴/腕部深度标注
  3_depth_colormap.mp4    对齐到彩色视口的深度图（0.3-1.5m JET 伪彩）

处理链与开发环境 D435 实时/离线 demo 完全一致：MediaPipe 21 点检测（CPU）
→ 手性投票 → 深度前向对齐 + 空穴回填 → 单目抬升（深度带采样 + 缺深补点）
→ 双手槽位分配（标签+几何+互斥+复活）→ 时序一致性门 + wholesale 两帧确认
→ αβ 槽位跟踪 + 遮挡预测 → One-Euro 3D 平滑 → 质心锚定（整手共模跳抑制）
→ 世界锚点锁定（3D 视角恒定）→ 三路渲染。

输入（主程序录制会话目录布局）：
    <会话目录>/videos/d435_rgb/chunk-0000/d435_rgb.mp4     （兜底 videos/d435_rgb.mp4）
    <会话目录>/depth/d435_depth/NNNNNN.png                 （1-based，16bit 毫米）
    <会话目录>/calibration/head_stereo.json                （录制期深度内参，权威）

用法：
    python d435_hands_demo.py <会话目录>
    python d435_hands_demo.py <会话目录> --out-dir result --depth-overlay
    python d435_hands_demo.py <会话目录> --calib 设备标定.json --fill 3

依赖（仅三个包，CPU 即可）：mediapipe / opencv-python / numpy
（requirements.txt 已锁定版本范围）。可选 ffmpeg：用于把输出转成 H.264
编码；没有 ffmpeg 也能运行，输出 mp4v 编码（VLC/PotPlayer 可直接播放）。

零仓库依赖：检测/对齐/抬升/分配/跟踪/平滑/渲染/转码全部内联在本文件。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ═══════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_SCRIPT_DIR, "hand_landmarker.task")

RENDER_SIZE = (1280, 720)
_ROT_TOTAL = 360            # 渲染器总帧：revolutions=1.0 时 frame_idx ≈ 方位角度数
_VIEW_YAW0 = math.pi        # 静态视角方位角（rad）：π = 正面看手掌
_VIEW_ELEV0 = 25.0          # 静态视角俯仰角（deg）
_GATE_FORGIVE = 5           # M6：逐点连续被门控帧数上限，超限采信新观测（防锁死）
# yaw 反解为渲染器 frame_idx：θ = 2π·1.0·frame_idx/(360−1) = π → 179.5 精确正面
_STATIC_FRAME_IDX = (_VIEW_YAW0 / (2.0 * math.pi * 1.0)) * (_ROT_TOTAL - 1)

_MAX_DEPTH_MM = 8000.0      # 远背景离群剔除上限（实测有效值 p95≈1.1m）
_FX_REL_TOL = 0.01          # 深度内参交叉核对容差（>1% 告警，可能换机）

# 手性投票（identity.py）
VOTE_WINDOW = 7
MIN_VOTE_SCORE = 0.7
ASSOC_GATE = 0.12
OVERLAP_PX = 0.05
MAX_TRACKS = 2

# 单目槽位分配（mono_assign.py）
UNRELIABLE_GATE = 0.15      # 3D 质心距槽预测的门限（米）
WRIST_MUTEX = 0.10          # 互斥守卫：两槽腕距小于此（米）触发排列比较
SWAP_MARGIN = 0.005         # 互斥换位需严格优于当前 ≥5mm（防逐帧抖动）
MIN_VALID_PTS = 4           # 3D 质心判据最少有效点数

# 深度带采样 / 时序门（lift3d.py）
BAND_HALF_M = 0.12          # 手深带半宽（米）
BAND_MIN_VALID = 4
GATE_M = 0.15               # 时序一致性门（米/帧）

# 内嵌出厂标定（示例回退值；生产环境应使用录制会话中的标定文件）
# head_stereo.json 交叉核对零警告）。仅供无 --calib、无脚本旁 JSON 时
# 回退；深度内参权威来源是录制会话的 head_stereo.json。
_EMBEDDED_CALIB = {
    "serial": "sample-d435",
    "color_intrinsics": {
        "width": 1280, "height": 720,
        "fx": 912.2550048828125, "fy": 910.3241577148438,
        "cx": 647.90576171875, "cy": 377.0218505859375,
        "model": "distortion.inverse_brown_conrady",
        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "depth_intrinsics": {
        "width": 848, "height": 480,
        "fx": 429.4732666015625, "fy": 429.4732666015625,
        "cx": 420.45916748046875, "cy": 231.80844116210938,
        "model": "distortion.brown_conrady",
        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "depth_to_color": {
        "rotation": [
            [0.999994158744812, -0.0013020101469010115, -0.0031666778959333897],
            [0.0013023947831243277, 0.9999991655349731, 0.00011940528202103451],
            [0.0031665198039263487, -0.0001235288509633392, 0.9999949932098389],
        ],
        "translation": [0.01472442876547575, 7.46611985960044e-05,
                        0.00021084764739498496],
    },
}

# ═══════════════════════════════════════════════════════════════════
# 关键点定义（hand_pipeline_mediapipe.py）
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# One-Euro 滤波器（hand_pipeline_mediapipe.py）
# ═══════════════════════════════════════════════════════════════════

class OneEuroFilter:
    """单值 One-Euro 自适应低通滤波器。

    低速运动时强平滑消除抖动，高速运动时自动放宽以保持响应速度。
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


# ═══════════════════════════════════════════════════════════════════
# 绘制共用定义（render_overlay.py 与 renderer_3d.py 原各有一份，字节
# 相同，此处共享一份）
# ═══════════════════════════════════════════════════════════════════

# 掌心连接（MediaPipe 21 点拓扑的腕-掌子集）
PALM_CONNECTIONS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

# 每指画法：拇指 1→2→3→4，其余指 0→(掌根)→MCP→PIP→DIP→指尖
FINGER_CHAINS = {name: (ids if name == "Thumb" else [0] + ids)
                 for name, (ids, _) in FINGERS.items()}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════════════════
# MediaPipe 检测管线（hand_pipeline_mediapipe.py）
# ═══════════════════════════════════════════════════════════════════

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


class MediaPipeHandPipeline:
    """MediaPipe 手部关键点检测管线（Tasks API，VIDEO 模式）。

    model_path: hand_landmarker.task 模型文件路径（默认脚本同目录）。
    num_hands: 最多检测手数（默认 2）。
    det_conf: 手掌检测器置信度阈值；track_conf: 跟踪置信度阈值。
    mirror: 是否左右镜像（默认 True，适配自拍视角；本 demo 检测器
            显式传 False——主程序录制画面未镜像）。
    smooth: 2D 关键点 One-Euro 平滑开关（默认 True）。
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
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
                f"请确认 hand_landmarker.task 与本脚本在同一目录，"
                f"或用 --model 指定路径。模型可从官方地址下载（约 7.8MB）:\n"
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

    def process(self, frame: np.ndarray) -> FrameResult:
        """处理一帧 BGR 图像，返回 FrameResult（.hands: [HandResult, ...]）。"""
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


# ═══════════════════════════════════════════════════════════════════
# 2D 检测抽象层（本文件内联，仅 CPU 路径）
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DetectedHand:
    """单只手的 2D 检测结果（几何层消费的最小结构）。"""

    landmarks: np.ndarray          # (21,2) float32 亚像素像素坐标
    label: str = "Hand"            # "Left" / "Right"
    score: float = 0.0
    index: int = 0
    conf: np.ndarray | None = None  # (21,) 逐点置信度（mmpose 才有；MediaPipe 为 None）

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
    """MediaPipe HandLandmarker 检测器（float 亚像素输出，仅 CPU）。"""

    def __init__(self, model_path: str = DEFAULT_MODEL, num_hands: int = 2,
                 mirror: bool = False, smooth: bool = True,
                 freq_min: float = 5.0, beta: float = 0.05, dcutoff: float = 1.0,
                 det_conf: float = 0.5, track_conf: float = 0.5):
        self.num_hands = num_hands
        self._pipe = MediaPipeHandPipeline(
            model_path=model_path, num_hands=num_hands,
            det_conf=det_conf, track_conf=track_conf,
            mirror=mirror, smooth=smooth,
            freq_min=freq_min, beta=beta, dcutoff=dcutoff)

    def detect(self, frame_bgr: np.ndarray) -> list:
        h, w = frame_bgr.shape[:2]
        result = self._pipe.process(frame_bgr)
        out = []
        for i, hr in enumerate(result.hands[: self.num_hands]):
            out.append(DetectedHand.from_hand_result(hr, w, h, index=i))
        return out

    def reset(self) -> None:
        self._pipe.reset()

    def close(self) -> None:
        self._pipe.close()


# ═══════════════════════════════════════════════════════════════════
# 手性投票（本文件内联，调试钩子已删）
# ═══════════════════════════════════════════════════════════════════

class HandednessVoter:
    """单目实例。update(hands) 原地覆盖每只 DetectedHand 的 label。

    轨迹 = {pos: 最近质心, votes: deque(label), last: 最近稳定 label,
    idle: 未关联帧数}。每帧贪心分配当前手→最近未占用轨迹（门限内）；
    未分配轨迹冻结（票仓保留，手暂离帧再回来继续投票）；新轨迹超
    MAX_TRACKS 时替换最旧轨迹。双手交叠时不做表决（原始 label 直通）
    并把两轨迹票仓重播种，防交叉关联污染票仓。
    """

    def __init__(self, window: int = VOTE_WINDOW,
                 min_score: float = MIN_VOTE_SCORE):
        self.window = window
        self.min_score = min_score
        self._tracks = []   # [{"pos": ndarray, "votes": deque, "last": str, "idle": int}]

    @staticmethod
    def _centroid(h) -> np.ndarray | None:
        pts = np.asarray(h.landmarks, np.float64).reshape(-1, 2)
        ok = np.isfinite(pts).all(axis=1)
        if ok.sum() < 3:
            return None
        return np.median(pts[ok], axis=0)

    def update(self, hands: list, frame_w: int = 1280, frame_h: int = 800,
               frame: int | None = None, cam: str = "?") -> None:
        if not hands:
            self._tracks = []          # 空帧：轨迹全清（重新开始）
            return
        gate = ASSOC_GATE * max(frame_w, frame_h)
        cents = [self._centroid(h) for h in hands]
        overlap = (len(cents) == 2 and cents[0] is not None
                   and cents[1] is not None
                   and float(np.linalg.norm(cents[0] - cents[1]))
                   <= OVERLAP_PX * max(frame_w, frame_h))

        # ── 贪心分配：当前手 → 最近未占用轨迹（门限内）──
        used, assigned = set(), {}     # hand_idx → track_idx
        for i, c in enumerate(cents):
            if c is None:
                continue
            best_j, best_d = -1, np.inf
            for j, tr in enumerate(self._tracks):
                if j in used:
                    continue
                dd = float(np.linalg.norm(c - tr["pos"]))
                if dd < best_d:
                    best_j, best_d = j, dd
            if best_j >= 0 and best_d <= gate:
                assigned[i] = best_j
                used.add(best_j)

        # ── 未分配的手 → 新轨迹（超限替换最旧）──
        for i, c in enumerate(cents):
            if c is None or i in assigned:
                continue
            tr = {"pos": c.copy(), "votes": deque(maxlen=self.window),
                  "last": "", "idle": 0}
            if len(self._tracks) >= MAX_TRACKS:
                j = max(range(len(self._tracks)),
                        key=lambda j: self._tracks[j]["idle"])
                self._tracks[j] = tr
                assigned[i] = j
            else:
                self._tracks.append(tr)
                assigned[i] = len(self._tracks) - 1

        if overlap:
            # 双手交叠：关联不可靠 → 标签/位置全冻结为交叠前的稳定值。
            for i, h in enumerate(hands):
                if i in assigned:
                    tr = self._tracks[assigned[i]]
                    if tr["last"]:
                        h.label = tr["last"]     # 冻结：交叠前的稳定标签
                    tr["idle"] = 0
                    # 位置不更新：交叠期质心不可靠，冻结供分离后回关联
            for j, tr in enumerate(self._tracks):
                if j not in used:
                    tr["idle"] += 1
            return

        # ── 逐手表决（轨迹票 + 原始票，严格多数；平票沿用轨迹稳定 label）──
        for i, h in enumerate(hands):
            if i in assigned:
                tr = self._tracks[assigned[i]]
                votes = list(tr["votes"])
                if h.score >= self.min_score and h.label:
                    votes.append(h.label)       # 当前原始票
                if votes:
                    majority = max(set(votes), key=votes.count)
                    if votes.count(majority) > len(votes) / 2:
                        h.label = majority
                    elif tr["last"]:
                        h.label = tr["last"]    # 平票稳定优先（重播种后防闪烁）
                elif tr["last"]:
                    h.label = tr["last"]        # 无任何票（新手轨迹+空原始 label）
                # MediaPipe 偶发 label=""（无手性输出）→ votes 全空，
                # 上方 if votes 兜底，空 label 不入票仓
            if i in assigned:
                tr = self._tracks[assigned[i]]
                if cents[i] is not None:
                    tr["pos"] = cents[i]
                if h.label:
                    tr["votes"].append(h.label)
                tr["last"] = h.label
                tr["idle"] = 0
        for j, tr in enumerate(self._tracks):
            if j not in used:
                tr["idle"] += 1


# ═══════════════════════════════════════════════════════════════════
# 三角化结果载体 + 槽位跟踪（本文件内联，仅保留单目管线所需部分）
# ═══════════════════════════════════════════════════════════════════

class TriangulationResult:
    """一次左右目点对三角化的结果（单目管线中作伪 pair 载体复用）。"""

    def __init__(self, points_3d: np.ndarray, reproj_error: np.ndarray):
        self.points_3d = points_3d          # (N,3) float64, 无效点 NaN, 相机系米制
        self.reproj_error = reproj_error    # (N,) float64, 无效点 inf
        self.valid = np.isfinite(reproj_error)          # (N,) bool
        self.valid_count = int(np.count_nonzero(self.valid))
        v = reproj_error[self.valid]
        self.mean_error = float(v.mean()) if v.size else float("inf")

    @property
    def z(self) -> np.ndarray:
        return self.points_3d[:, 2]


@dataclass
class PseudoHandPair:
    """丢失槽位的传播重检载体（接口与 HandPair 平替，供 refine_batch 消费）。

    result = TriangulationResult(预测 3D, 全 inf err)：valid_count=0 →
    采纳判据退化为"找到任何几何一致结果就采纳"；全失败时返回该预测
    结果本身（兜底传播）。
    """

    result: TriangulationResult
    left_label: str = ""
    l_idx: int = -1
    r_idx: int = -1


class HandSlotTracker:
    """两槽位（hand_0/hand_1）αβ 跟踪器，槽位顺序不重排。

    槽位 label 变化 → 槽位重置（换手了，旧状态污染无效）。
    """

    def __init__(self, max_lost: int = 15, alpha: float = 0.5, beta: float = 0.1,
                 debug_log: str = None):
        self.max_lost = max_lost
        self.alpha = alpha
        self.beta = beta
        self.slots = [{"label": None, "x": None, "v": None,
                       "last_t": None, "lost": 0} for _ in range(2)]
        self._dbg = open(debug_log, "w", newline="", encoding="utf-8") \
            if debug_log else None
        if self._dbg:
            self._dbg.write("frame,slot,event,label,lost\n")

    def debug(self, event: str, slot: int, t: int):
        if self._dbg:
            s = self.slots[slot]
            self._dbg.write(f"{t},{slot},{event},{s['label'] or ''},{s['lost']}\n")

    def slot_label(self, slot: int) -> str:
        return self.slots[slot]["label"] or ""

    def observe_slot(self, slot: int, label: str, pts3d: np.ndarray, t: int):
        """真实检测回写（真 pair 或救援成功）。label 变化 → 槽位重置。"""
        s = self.slots[slot]
        x_meas = np.asarray(pts3d, np.float64).reshape(-1, 3)
        if s["label"] is not None and s["label"] != label:
            s.update(label=None, x=None, v=None, last_t=None, lost=0)
            self.debug("reset", slot, t)
        s["label"] = label
        if s["x"] is None:
            s["x"] = x_meas.copy()
            s["v"] = np.zeros_like(x_meas)
            s["last_t"] = t
            s["lost"] = 0
            self.debug("observe-init", slot, t)
            return
        dt = max(float(t - s["last_t"]), 1e-3)
        if dt > self.max_lost:
            # 长缺口后的观测：αβ 恒速外推 v·dt 已不可信 → 重初始化，
            # 直接采信本次观测。
            s["x"] = x_meas.copy()
            s["v"] = np.zeros_like(x_meas)
            s["last_t"], s["lost"] = t, 0
            self.debug("observe-reinit", slot, t)
            return
        # αβ：预测 → 修正。NaN 点（该点本次无效）保持纯预测
        ok = np.isfinite(x_meas).all(axis=1)
        x_pred = s["x"] + s["v"] * dt
        x_new = np.where(ok[:, None],
                         self.alpha * x_meas + (1.0 - self.alpha) * x_pred,
                         x_pred)
        v_new = np.where(ok[:, None],
                         self.beta * (x_new - s["x"]) / dt
                         + (1.0 - self.beta) * s["v"],
                         s["v"])
        s["x"], s["v"], s["last_t"], s["lost"] = x_new, v_new, t, 0
        self.debug("observe", slot, t)

    def mark_lost(self, slot: int, t: int):
        """救援失败：计数丢失（超 max_lost 后 predict 返回 None，幻觉硬顶）。"""
        self.slots[slot]["lost"] += 1
        self.debug("mark-lost", slot, t)

    def predict(self, slot: int, t: int) -> np.ndarray | None:
        """恒速外推 x + v·(t − last_t)。从未见过 / 丢失超限 → None。"""
        s = self.slots[slot]
        if s["x"] is None or s["lost"] > self.max_lost:
            return None
        dt = max(float(t - s["last_t"]), 0.0)
        return s["x"] + s["v"] * dt

    def close(self):
        if self._dbg:
            self._dbg.close()
            self._dbg = None


def make_pseudo_pair(pred: np.ndarray, label: str) -> PseudoHandPair:
    """预测 3D → PseudoHandPair（err 全 inf：valid_count=0 让判据退化）。"""
    res = TriangulationResult(np.asarray(pred, np.float64).reshape(-1, 3),
                              np.full((21,), np.inf, np.float64))
    return PseudoHandPair(result=res, left_label=label)


# ═══════════════════════════════════════════════════════════════════
# 3D 域时序平滑（本文件内联）
# ═══════════════════════════════════════════════════════════════════

class Hand3DSmoother:
    """(2,21,3) 手部 3D 关键点时序 One-Euro 平滑。

    防污染：手槽位 label 变化或"空→有"跳变时重置该槽滤波器，
    避免上一只手的状态污染下一只。
    """

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


# ═══════════════════════════════════════════════════════════════════
# 深度→彩色对齐 + 关键点深度采样（hand_3d_d435/depth_align.py）
# ═══════════════════════════════════════════════════════════════════

def load_session_depth_intr(session_dir: str) -> dict | None:
    """录制期 head_stereo.json → 深度相机内参 dict（fx/fy/cx/cy/width/height）。"""
    path = os.path.join(session_dir, "calibration", "head_stereo.json")
    try:
        with open(path, encoding="utf-8") as f:
            head = json.load(f)
        dc = head["depth_camera"]
        return {
            "fx": float(dc["intrinsic"][0]), "fy": float(dc["intrinsic"][1]),
            "cx": float(dc["intrinsic"][2]), "cy": float(dc["intrinsic"][3]),
            "width": int(dc.get("resolution", [848, 480])[0]),
            "height": int(dc.get("resolution", [848, 480])[1]),
        }
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return None


class DepthAligner:
    """深度图 → 彩色视口 aligned 深度图（毫米，0=无效）。"""

    def __init__(self, color_intr: dict, depth_to_color: dict,
                 depth_intr: dict):
        self.fx_c = float(color_intr["fx"])
        self.fy_c = float(color_intr["fy"])
        self.cx_c = float(color_intr["cx"])
        self.cy_c = float(color_intr["cy"])
        self.cw = int(color_intr.get("width", 1280))
        self.ch = int(color_intr.get("height", 720))
        fxd = float(depth_intr["fx"])
        fyd = float(depth_intr["fy"])
        cxd = float(depth_intr["cx"])
        cyd = float(depth_intr["cy"])
        self.dw = int(depth_intr.get("width", 848))
        self.dh = int(depth_intr.get("height", 480))
        uu, vv = np.meshgrid(np.arange(self.dw, dtype=np.float32),
                             np.arange(self.dh, dtype=np.float32))
        # (dh,dw,3) 深度相机系单位射线（标定不变，构造时预计算一次）
        self._ray = np.stack([(uu - cxd) / fxd, (vv - cyd) / fyd,
                              np.ones_like(uu)], axis=-1).astype(np.float32)
        self._R = np.asarray(depth_to_color["rotation"], np.float64)
        self._t_mm = (np.asarray(depth_to_color["translation"], np.float64)
                      * 1000.0)

    def align_depth_to_color(self, depth_mm: np.ndarray) -> np.ndarray:
        """(dh,dw) uint16/float 毫米深度 → (ch,cw) float32 aligned 深度（0=无效）。"""
        z = np.asarray(depth_mm, np.float32)
        valid = (z > 0) & (z <= _MAX_DEPTH_MM)
        p_d = self._ray * z[..., None]                    # 深度相机系, mm
        p_c = p_d[valid] @ self._R.T + self._t_mm         # 彩色相机系, mm
        with np.errstate(divide="ignore", invalid="ignore"):
            u_c = self.fx_c * p_c[:, 0] / p_c[:, 2] + self.cx_c
            v_c = self.fy_c * p_c[:, 1] / p_c[:, 2] + self.cy_c
            z_c = p_c[:, 2]
        iu = np.rint(u_c).astype(np.int32)
        iv = np.rint(v_c).astype(np.int32)
        inside = ((iu >= 0) & (iu < self.cw) & (iv >= 0) & (iv < self.ch)
                  & (z_c > 0))
        # z-buffer：初始化为 +inf（0 是无效哨兵，用 0 初始化会让
        # minimum.at 的 min(0, z)=0 永远写不进任何正深度）
        aligned = np.full((self.ch, self.cw), np.inf, np.float32)
        np.minimum.at(aligned, (iv[inside], iu[inside]),
                      z_c[inside].astype(np.float32))     # 最近保留
        aligned[~np.isfinite(aligned)] = 0.0
        return self._fill_holes(aligned)

    def _fill_holes(self, aligned: np.ndarray, passes: int = 3) -> np.ndarray:
        """空穴回填：每轮对 0 像素取 3×3 有效邻域最小值（最近表面语义）。

        与开发环境 scipy.ndimage.minimum_filter(size=3, mode="constant",
        cval=inf) 逐像素等价（纯 numpy 实现，省去 scipy 依赖）。前向投影
        是 848×480 → 1280×720 的 ~2.12× 上采样：z-buffer 后 rint 跳列/
        跳行留下 ~50% 空穴（覆盖仅 ~19%）。缺口 ≤3px，3 轮即可补满；
        值只写空穴，已有值不碰（不腐蚀有效区）。
        """
        h, w = aligned.shape
        for _ in range(passes):
            holes = aligned == 0
            if not holes.any():
                break
            work = np.where(holes, np.inf, aligned)
            pad = np.pad(work, 1, mode="constant", constant_values=np.inf)
            nb = np.minimum.reduce([pad[i:i + h, j:j + w]
                                    for i in range(3) for j in range(3)])
            ok = holes & np.isfinite(nb)
            if not ok.any():
                break
            aligned[ok] = nb[ok]
        return aligned

    def sample_points(self, aligned: np.ndarray, uv, band=None) -> np.ndarray:
        """(N,2) 像素坐标（亚像素取最近整像素）→ (N,) 深度 mm，无效 NaN。

        3×3 窗口剔除 0/非有限后取中位，有效数 ≥2 才出数——中位天然抗
        边缘混入背景（手 ~445mm vs 背景 >1000mm，窗口内少数背景点不赢中位）。
        band=(z_lo, z_hi) 时窗口像素先按深度带过滤再取中位（手缘点窗口
        混入背景时背景像素被剔除，中位必落手上；带内有效 <2 → NaN，
        由上层 tracker 预测补全）。band 单位与 aligned 一致（mm）。
        """
        uv = np.asarray(uv, np.float32).reshape(-1, 2)
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        out = np.full(len(uv), np.nan, np.float32)
        for k in range(len(uv)):
            i0, i1 = max(v[k] - 1, 0), min(v[k] + 2, self.ch)
            j0, j1 = max(u[k] - 1, 0), min(u[k] + 2, self.cw)
            w = aligned[i0:i1, j0:j1]
            w = w[(w > 0) & np.isfinite(w)]
            if band is not None:
                w = w[(w >= band[0]) & (w <= band[1])]
            if w.size >= 2:
                out[k] = float(np.median(w))
        return out


class LiveAligner(DepthAligner):
    """DepthAligner + 可调填洞轮数（默认 1，与实时 demo 预算一致）。

    align_depth_to_color 内部调用 self._fill_holes(aligned)——覆写该方法
    注入轮数即可，对齐数学零改动。
    """

    def __init__(self, *args, fill_passes: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.fill_passes = fill_passes

    def _fill_holes(self, aligned, passes=None):
        return super()._fill_holes(aligned, passes or self.fill_passes)


# ═══════════════════════════════════════════════════════════════════
# 单目抬升 + 深度带采样 + 时序门（hand_3d_d435/lift3d.py）
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LiftResult:
    """mimic TriangulationResult 的最小接口。"""

    points_3d: np.ndarray                      # (21,3) 米，无效点 NaN
    mean_error: float = float("nan")           # 单目无重投影概念 → NaN
    valid_count: int = 0                       # 有效点数（槽位分配判可靠度用）


@dataclass
class D435Pair:
    """mimic RefinedPair：2D 检测 + 抬升 3D 的载体。"""

    result: LiftResult
    left_label: str = ""
    used: bool = False
    hand2d: np.ndarray | None = None           # (21,2) 原始 2D（2D 判据/叠显用）
    n_valid: int = 0                           # 深度采样有效点数
    det: object = None                         # 原始 DetectedHand
    measured: np.ndarray | None = None         # (21,) bool：点 z 来自实测（False=补点）


def _lift_z(aligner: DepthAligner, aligned: np.ndarray,
            pts2d: np.ndarray, band: bool = True) -> tuple[np.ndarray, float | None]:
    """21 点深度采样（米）→ (带内测量 z, 手深中位 zc 或 None)。

    band=True 两遍：先无约束取手深中位 zc，再只从 [zc±BAND_HALF_M] 带内
    窗口像素取中位（背景像素剔除）。第一遍有效点 <BAND_MIN_VALID →
    zc=None，退回无约束单遍结果（假检测/稀疏深度时不可信）。"""
    z_mm = aligner.sample_points(aligned, pts2d) * 0.001
    zc = None
    if band:
        ok1 = np.isfinite(z_mm)
        if ok1.sum() >= BAND_MIN_VALID:
            zc = float(np.median(z_mm[ok1]))
            z_mm = aligner.sample_points(
                aligned, pts2d,
                band=((zc - BAND_HALF_M) * 1000.0,
                      (zc + BAND_HALF_M) * 1000.0)) * 0.001
    return z_mm, zc


def lift_hand(hand, aligner: DepthAligner, aligned: np.ndarray,
              band: bool = True, complete: bool = True) -> D435Pair:
    """DetectedHand + aligned 深度图 → D435Pair（3D 在彩色相机系，米）。

    band=True（默认）：深度带约束采样（见 _lift_z）——手缘点的 3×3 窗口
    混入背景深度时背景像素被带滤剔除，中位必落手上。

    complete=True（默认）：深度缺失点补到手深中位 zc、x,y 由 2D 关键点
    反投影——参考单目动捕"保持 z 不变、调 x,y 使投影与 2D 关键点一致"
    的经验；zc 是 D435 实测真值（非模型值），补后投影与 2D 天然一致。
    测量有效点 <BAND_MIN_VALID 的检测（含假检测）无 zc → 不补，保持
    NaN（宁缺勿错）。valid_count 仍只计实测点数。
    """
    pts2d = np.asarray(hand.landmarks, np.float32).reshape(-1, 2)
    z_m, zc = _lift_z(aligner, aligned, pts2d, band=band)
    measured = np.isfinite(z_m)                     # z 来自实测（False=补点）
    n_valid = int(measured.sum())
    if complete and zc is not None:
        z_use = np.where(measured, z_m, zc)
    else:
        z_use = z_m
    ok = np.isfinite(z_use)
    xyz = np.full((21, 3), np.nan, np.float64)
    xyz[ok, 2] = z_use[ok]
    xyz[ok, 0] = (pts2d[ok, 0] - aligner.cx_c) * z_use[ok] / aligner.fx_c
    xyz[ok, 1] = (pts2d[ok, 1] - aligner.cy_c) * z_use[ok] / aligner.fy_c
    return D435Pair(result=LiftResult(xyz, float("nan"), n_valid),
                    left_label=hand.label, hand2d=pts2d, n_valid=n_valid,
                    det=hand, measured=measured)


def apply_slot_zc(pair: D435Pair, zc: float, aligner: DepthAligner) -> None:
    """M5：补点深度锚定到槽级稳定 zc（保持 z、调 x,y 反投影与 2D 一致）。

    原地修改 pair.result.points_3d。只作用于补点（measured=False）：
    实测点不动；zc 由调用方做时域稳定（逐帧独立中位是整手共模跳的
    最大来源）。无补点/无 hand2d 时零操作。
    """
    if pair.measured is None or pair.hand2d is None:
        return
    pts = np.asarray(pair.result.points_3d, np.float64).reshape(21, 3)
    comp = np.isfinite(pts).all(axis=1) & ~pair.measured
    if not comp.any():
        return
    u = pair.hand2d[comp, 0]
    v = pair.hand2d[comp, 1]
    pts[comp, 2] = zc
    pts[comp, 0] = (u - aligner.cx_c) * zc / aligner.fx_c
    pts[comp, 1] = (v - aligner.cy_c) * zc / aligner.fy_c
    pair.result.points_3d = pts


def gate_observations(pts: np.ndarray, pred: np.ndarray | None,
                      gate: float = GATE_M,
                      wholesale_frac: float = 0.6) -> tuple[np.ndarray, bool]:
    """时序一致性门：观测 3D 与槽位预测差 >gate 的点判可疑 → 置 NaN。

    交给 tracker.observe_slot 后，NaN 点走纯预测——可疑观测不写入状态，
    下一帧真实值可追回。翻面事件（手 0.35m ↔ 背景 1.4m）由此在写入前
    拦截。pred 为 None（槽未见过）不门控。

    **整手级逃逸**：可疑点 ≥ 有限点的 wholesale_frac → 返回 (原样, True)。
    单点翻面只打 1-3 个点；整手级不匹配说明槽状态过时。调用方见 True
    应触发槽位重置后采信原观测。
    """
    pts = np.asarray(pts, np.float64).reshape(21, 3).copy()
    if pred is None:
        return pts, False
    pred = np.asarray(pred, np.float64).reshape(21, 3)
    d = np.linalg.norm(pts - pred, axis=1)
    fin = np.isfinite(pts).all(axis=1)
    suspect = fin & (d > gate)
    if fin.sum() >= BAND_MIN_VALID and suspect.sum() >= wholesale_frac * fin.sum():
        return pts, True
    pts[suspect] = np.nan
    return pts, False


# ═══════════════════════════════════════════════════════════════════
# 单目双手槽位分配（hand_3d_d435/mono_assign.py，调试钩子已删）
# ═══════════════════════════════════════════════════════════════════

def _centroid3(pair) -> np.ndarray | None:
    pts = np.asarray(pair.result.points_3d, np.float64).reshape(-1, 3)
    ok = np.isfinite(pts).all(axis=1)
    if ok.sum() < MIN_VALID_PTS:
        return None
    return np.median(pts[ok], axis=0)


def _wrist(pair) -> np.ndarray | None:
    w = np.asarray(pair.result.points_3d, np.float64).reshape(-1, 3)[0]
    return w if np.isfinite(w).all() else None


def _cost(pair, slot_pred, color_intr):
    """3D 质心距槽预测质心（米）；质心不可靠退 2D 判据（预测投影 vs 2D 质心）。

    注意：槽预测是 (21,3) 且带 NaN 洞，必须取预测自身有效点中位做质心，
    不能整阵相减（NaN 污染范数 → cost 恒 NaN → 一切分配被拒）。
    """
    if slot_pred is None:
        return np.inf
    pred = np.asarray(slot_pred, np.float64).reshape(-1, 3)
    pok = np.isfinite(pred).all(axis=1)
    if pok.sum() < MIN_VALID_PTS:
        return np.inf
    p3 = np.median(pred[pok], axis=0)
    if p3[2] <= 0:
        return np.inf
    c3 = _centroid3(pair)
    if c3 is not None:
        return float(np.linalg.norm(c3 - p3))
    # 2D 退路：预测质心投影回图像 vs 该手 2D 质心（像素距 × Z/fx → 米）
    u = color_intr[0] * p3[0] / p3[2] + color_intr[2]
    v = color_intr[1] * p3[1] / p3[2] + color_intr[3]
    pts2d = np.asarray(pair.hand2d, np.float64).reshape(-1, 2)
    ok2 = np.isfinite(pts2d).all(axis=1)
    if ok2.sum() < MIN_VALID_PTS:
        return np.inf
    c2 = np.median(pts2d[ok2], axis=0)
    dist_px = float(np.linalg.norm(c2 - [u, v]))
    return dist_px * float(p3[2]) / color_intr[0]      # 像素 → 米（Z/fx）


def _in_out(pair, out) -> bool:
    return any(o is pair for o in out)


def _lab_ok(slot: int, pair, tracker: HandSlotTracker) -> bool:
    """pair 入 slot 是否 label 兼容（槽无标签 / 手无标签 / 相同）。"""
    sl = tracker.slot_label(slot)
    return sl == "" or not pair.left_label or sl == pair.left_label


def assign_mono_slots(pairs, tracker: HandSlotTracker, n: int,
                      color_intr=(917.0, 917.0, 640.0, 360.0),
                      lost_counts=(0, 0)) -> list:
    """pairs: list[D435Pair]（≤2，voter 已稳定 label）
    → [slot0_pair|None, slot1_pair|None]（None = 本帧该槽无真手）。

    决策层级：
      1. 冷启动：两槽从未见过 → 标签惯例 Left→slot0 / Right→slot1，
         无标签按检测序号；
      2. 标签唯一命中存活槽 + 几何门（≤UNRELIABLE_GATE）；
      2b. 标签唯一命中困境槽（上帧无真手，lost_counts≥1）→ 不设几何门
          （恒速外推预测在手离开期间会漂移，label 是唯一可靠信号）；
      3. 贪心几何：剩余手入最近未占用存活槽（门限内 + 标签守卫）；
      4. 互斥守卫：两槽双真实且腕距 <WRIST_MUTEX → 两种排列取总 cost
         更小者（防交叠期交叉串槽）；
      5. 未见槽冷启：有标签手按标签惯例入空死槽，无标签手唯一空死槽兜底。

    lost_counts：主循环维护的各槽连续丢失帧数（无真手帧计数）。
    """
    out: list = [None, None]
    if not pairs:
        return out
    pred = [tracker.predict(s, n) for s in range(2)]
    labels = [p.left_label for p in pairs]

    # 1) 冷启动：标签惯例（Left→0/Right→1），无标签按序号
    if (tracker.slot_label(0) == "" and tracker.slot_label(1) == ""
            and pred[0] is None and pred[1] is None):
        assigned = set()
        for s in (0, 1):
            for i, lab in enumerate(labels):
                if i in assigned:
                    continue
                if lab in ("Left", "Right") and lab == ("Left" if s == 0
                                                        else "Right"):
                    out[s] = pairs[i]
                    assigned.add(i)
        for i in [j for j in range(len(pairs)) if j not in assigned]:
            for s in (0, 1):
                if out[s] is None:
                    out[s] = pairs[i]
                    break
        return out

    # 2) 标签唯一命中存活槽 + 几何门
    for i, lab in enumerate(labels):
        if not lab or _in_out(pairs[i], out):
            continue
        match = [s for s in (0, 1)
                 if out[s] is None and tracker.slot_label(s) == lab
                 and pred[s] is not None]
        if len(match) == 1 and _cost(pairs[i], pred[match[0]],
                                     color_intr) <= UNRELIABLE_GATE:
            out[match[0]] = pairs[i]

    # 2b) 标签唯一命中困境槽（上帧无真手）→ 不设几何门
    for i, lab in enumerate(labels):
        if not lab or _in_out(pairs[i], out):
            continue
        match = [s for s in (0, 1)
                 if out[s] is None and tracker.slot_label(s) == lab
                 and lost_counts[s] >= 1]
        if len(match) == 1:
            out[match[0]] = pairs[i]

    # 3) 贪心几何：剩余手 → 最近未占用存活槽（门限内 + 标签守卫）
    free_slots = [s for s in (0, 1) if out[s] is None and pred[s] is not None]
    rest = [i for i in range(len(pairs)) if not _in_out(pairs[i], out)]
    rest.sort(key=lambda i: min((_cost(pairs[i], pred[s], color_intr)
                                 for s in free_slots), default=np.inf))
    for i in rest:
        if not free_slots:
            break
        cand = [s for s in free_slots
                if labels[i] == "" or tracker.slot_label(s) == ""
                or tracker.slot_label(s) == labels[i]]
        if not cand:
            continue
        best = min(cand, key=lambda s: _cost(pairs[i], pred[s], color_intr))
        if _cost(pairs[i], pred[best], color_intr) <= UNRELIABLE_GATE:
            out[best] = pairs[i]
            free_slots.remove(best)

    # 4) 互斥守卫：双真实且腕距 <WRIST_MUTEX → 两种排列取总 cost 更小者
    if all(out[s] is not None for s in (0, 1)):
        w0, w1 = _wrist(out[0]), _wrist(out[1])
        if w0 is not None and w1 is not None \
                and float(np.linalg.norm(w0 - w1)) < WRIST_MUTEX:
            cur = sum(_cost(out[s], pred[s], color_intr) for s in (0, 1))
            swp = sum(_cost(out[1 - s], pred[s], color_intr) for s in (0, 1))
            cur_ok = all(_lab_ok(s, out[s], tracker) for s in (0, 1))
            swp_ok = all(_lab_ok(s, out[1 - s], tracker) for s in (0, 1))
            if swp_ok and (not cur_ok or swp < cur - SWAP_MARGIN):
                out[0], out[1] = out[1], out[0]

    # 5) 未见槽冷启。丢失超过 max_lost 后 pred=None；如果本帧有多只
    # 未标注手，旧逻辑只有在 free_dead==1 时才会接回，导致两个槽永久
    # absent。相同数量的未标注检测与死槽可以安全按检测顺序重建，随后
    # HandSlotTracker.observe_slot 会按 dt>max_lost 走 re-init 分支。
    for i, lab in enumerate(labels):
        if _in_out(pairs[i], out):
            continue
        free_dead = [s for s in (0, 1) if out[s] is None and pred[s] is None]
        if not free_dead:
            continue
        if lab in ("Left", "Right"):
            want = 0 if lab == "Left" else 1
            if out[want] is None and pred[want] is None:
                out[want] = pairs[i]
        elif not lab and len(free_dead) == 1:
            out[free_dead[0]] = pairs[i]                # 无标签单死槽

    unlabeled = [i for i, lab in enumerate(labels)
                 if not lab and not _in_out(pairs[i], out)]
    free_dead = [s for s in (0, 1)
                 if out[s] is None and pred[s] is None]
    if len(unlabeled) == len(free_dead) and len(unlabeled) > 1:
        for s, i in zip(free_dead, unlabeled):
            out[s] = pairs[i]

    return out


# ═══════════════════════════════════════════════════════════════════
# RGB 2D 叠加渲染（hand_3d_d435/render_overlay.py）
# ═══════════════════════════════════════════════════════════════════

def _draw_hand(img: np.ndarray, pts2d: np.ndarray, depth_mm: np.ndarray):
    """pts2d (21,2) px（NaN 点跳过）+ 逐点深度标注（mm 取整）。"""
    p = []
    for k in range(21):
        x, y = float(pts2d[k, 0]), float(pts2d[k, 1])
        p.append((int(x), int(y)) if np.isfinite(x) and np.isfinite(y)
                 else None)
    for a, b in PALM_CONNECTIONS:
        if p[a] is not None and p[b] is not None:
            cv2.line(img, p[a], p[b], (200, 200, 200), 2, cv2.LINE_AA)
    for name, (ids, color) in FINGERS.items():
        chain = FINGER_CHAINS[name]
        for i in range(len(chain) - 1):
            if p[chain[i]] is not None and p[chain[i + 1]] is not None:
                cv2.line(img, p[chain[i]], p[chain[i + 1]], color, 3,
                         cv2.LINE_AA)
        for idx in ids:
            if p[idx] is not None:
                r = 6 if idx == ids[-1] else 4
                cv2.circle(img, p[idx], r, color, -1, cv2.LINE_AA)
                cv2.circle(img, p[idx], r, (30, 30, 30), 1, cv2.LINE_AA)
    if p[0] is not None:
        cv2.circle(img, p[0], 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p[0], 8, (40, 40, 40), 2, cv2.LINE_AA)
    for k in range(21):
        if p[k] is not None and np.isfinite(depth_mm[k]):
            cv2.putText(img, f"{depth_mm[k]:.0f}",
                        (p[k][0] + 5, p[k][1] - 5), FONT, 0.32,
                        (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(rgb: np.ndarray, hands2d: np.ndarray, hands3d: np.ndarray,
                 labels, propagated, presents, frame_idx: int, total: int,
                 title: str = "D435 3D hand keypoints") -> np.ndarray:
    """rgb (H,W,3) BGR；hands2d (2,21,2) px；hands3d (2,21,3) 米（平滑后）。

    返回叠加帧（拷贝，不修改原图）。
    """
    img = rgb.copy()
    depth_mm = []
    for s in range(2):
        z = np.asarray(hands3d[s], np.float64).reshape(21, 3)[:, 2] * 1000.0
        depth_mm.append(z if presents[s] else np.full(21, np.nan))
    for s in range(2):
        if not presents[s]:
            continue
        _draw_hand(img, np.asarray(hands2d[s], np.float64).reshape(21, 2),
                   depth_mm[s])

    # HUD
    y = 26
    cv2.putText(img, f"{title}  frame {frame_idx}/{total}", (12, y), FONT,
                0.55, (235, 235, 235), 1, cv2.LINE_AA)
    y += 28
    for s, name in enumerate(("L", "R")):
        if presents[s]:
            lab = labels[s] or "-"
            prop = " [PROP]" if propagated[s] else ""
            z0 = depth_mm[s][0]
            txt = (f"{name}: {lab}{prop}  wrist={z0:.0f}mm"
                   if np.isfinite(z0) else f"{name}: {lab}{prop}")
            cv2.putText(img, txt, (12, y), FONT, 0.5, (120, 220, 120), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(img, f"{name}: absent", (12, y), FONT, 0.5,
                        (90, 90, 200), 1, cv2.LINE_AA)
        y += 22
    return img


def blend_depth(rgb: np.ndarray, aligned_mm: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
    """伪彩深度叠层：300-1200mm 归一 JET（无效=不叠），α 混合回 BGR。"""
    m = np.where((aligned_mm > 0) & np.isfinite(aligned_mm),
                 np.clip((aligned_mm - 300.0) / 900.0, 0.0, 1.0), 0.0)
    colored = cv2.applyColorMap((m * 255).astype(np.uint8),
                                cv2.COLORMAP_JET)
    mask = (m > 0).astype(np.float32)[..., None]
    return (rgb.astype(np.float32) * (1.0 - alpha * mask)
            + colored.astype(np.float32) * (alpha * mask)).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════
# 3D 骨架渲染器（本文件内联）
# ═══════════════════════════════════════════════════════════════════

class RotatingSkeletonRenderer:
    """3D 视角骨架渲染器 —— numpy + cv2 自写透视投影（零额外依赖）。

    虚拟相机绕手质心放置（默认整段视频转 2 圈，仰角 25°），五指分色
    骨架 + 掌心灰连接 + 腕部白圆 + 地面网格 + 腕部深度标注 + 相机系
    坐标轴 + HUD。本 demo 用静态视角：frame_idx=179.5（θ=π 正面）+
    elevation=25°，相机位姿全程恒定。

    坐标系：彩色相机系（OpenCV 约定，+X 右 / +Y 下 / +Z 前，米）。
    """

    def __init__(self, width: int = 1280, height: int = 720, fov_deg: float = 45.0,
                 revolutions: float = 2.0, elevation_deg: float = 25.0,
                 ground_grid: bool = True, depth_labels: bool = True,
                 bg_color=(28, 28, 30), text_color=(235, 235, 235)):
        self.width, self.height = width, height
        self.cx, self.cy = width / 2.0, height / 2.0
        self.fov_rad = math.radians(fov_deg)
        self.f = (height / 2.0) / math.tan(self.fov_rad / 2.0)
        self.revolutions = revolutions
        self.elevation = math.radians(elevation_deg)
        self.ground_grid = ground_grid
        self.depth_labels = depth_labels
        self.bg_color = bg_color
        self.text_color = text_color

    # ── 虚拟相机 ──────────────────────────────────────────────

    def _look_at(self, eye: np.ndarray, target: np.ndarray):
        """返回相机基 (right, up, fwd)。相机系 Y 向下 → 世界"上"= -Y。"""
        fwd = target - eye
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        up_world = np.array([0.0, -1.0, 0.0])
        right = np.cross(up_world, fwd)                       # 保证画面右 = 世界 +X
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(fwd, right)
        return right, up, fwd

    def _project(self, pts3d: np.ndarray, right, up, fwd, eye):
        """(N,3) → ((N,2) 像素, (N,) 视线深度)。"""
        d = pts3d - eye
        z = d @ fwd
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f * (d @ right) / z
            v = self.cy - self.f * (d @ up) / z
        return np.column_stack([u, v]), z

    # ── 元素绘制 ──────────────────────────────────────────────

    def _seg(self, img, p, a, b, color, thick):
        if p[a] is not None and p[b] is not None:
            cv2.line(img, p[a], p[b], color, thick, cv2.LINE_AA)

    def _draw_hand(self, img, proj, fin, label, err):
        p = [(int(x), int(y)) if ok else None for (x, y), ok in zip(proj, fin)]
        for a, b in PALM_CONNECTIONS:
            self._seg(img, p, a, b, (200, 200, 200), 2)
        for name, (ids, color) in FINGERS.items():
            chain = FINGER_CHAINS[name]
            for i in range(len(chain) - 1):
                self._seg(img, p, chain[i], chain[i + 1], color, 3)
            for idx in ids:
                if p[idx] is not None:
                    r = 7 if idx == ids[-1] else 5      # 指尖大点/关节小点
                    cv2.circle(img, p[idx], r, color, -1, cv2.LINE_AA)
                    cv2.circle(img, p[idx], r, (30, 30, 30), 1, cv2.LINE_AA)
        if p[0] is not None:
            cv2.circle(img, p[0], 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p[0], 9, (40, 40, 40), 2, cv2.LINE_AA)

    def _draw_grid(self, img, right, up, fwd, eye, y_grid, z_center, x_center=0.0):
        col = (58, 58, 58)
        for x in np.arange(x_center - 0.4, x_center + 0.401, 0.05):
            seg3d = np.array([[x, y_grid, z_center - 0.45], [x, y_grid, z_center + 0.45]])
            px, z = self._project(seg3d, right, up, fwd, eye)
            if np.all(np.isfinite(px)) and np.all(z > 0.05):
                cv2.line(img, tuple(px[0].astype(int)), tuple(px[1].astype(int)), col, 1)
        for z in np.arange(z_center - 0.45, z_center + 0.451, 0.05):
            seg3d = np.array([[x_center - 0.4, y_grid, z], [x_center + 0.4, y_grid, z]])
            px, zz = self._project(seg3d, right, up, fwd, eye)
            if np.all(np.isfinite(px)) and np.all(zz > 0.05):
                cv2.line(img, tuple(px[0].astype(int)), tuple(px[1].astype(int)), col, 1)

    def _draw_axes(self, img, right, up, fwd, eye):
        o = np.array([[0.0, 0.0, 0.0]])
        for vec, color, name in (((0.1, 0.0, 0.0), (60, 60, 255), "X"),
                                 ((0.0, 0.1, 0.0), (60, 200, 60), "Y"),
                                 ((0.0, 0.0, 0.1), (255, 120, 60), "Z")):
            seg3d = np.vstack([o, o + np.array(vec)])
            px, z = self._project(seg3d, right, up, fwd, eye)
            if np.all(np.isfinite(px)) and np.all(z > 0.05):
                p0, p1 = tuple(px[0].astype(int)), tuple(px[1].astype(int))
                cv2.line(img, p0, p1, color, 1)
                cv2.putText(img, name, p1, FONT, 0.4, color, 1, cv2.LINE_AA)

    # ── 整帧渲染 ──────────────────────────────────────────────

    def render(self, hands3d: np.ndarray, labels=("", ""), errs=(np.nan, np.nan),
               frame_idx: int = 0, total: int = 1,
               title: str = "3D hand keypoints (left-cam frame, meters)") -> np.ndarray:
        img = np.full((self.height, self.width, 3), self.bg_color, np.uint8)
        pts = np.asarray(hands3d, dtype=np.float64).reshape(2, 21, 3)
        finite = np.isfinite(pts).all(axis=2)
        valid_all = pts[finite]
        if valid_all.size == 0:
            cv2.putText(img, "no valid 3D hand keypoints", (60, self.height // 2),
                        FONT, 0.9, self.text_color, 2, cv2.LINE_AA)
            return img

        centroid = valid_all.mean(axis=0)
        # span = np.max−np.min（原开发环境 np.ptp；numpy 2.0-2.4 已移除
        # np.ptp，为兼容客户环境显式改写）
        span = float((np.max(valid_all, axis=0) - np.min(valid_all, axis=0)).max())
        half = span / 2.0 if span > 1e-6 else 0.3
        dist = float(np.clip(2.2 * half / math.tan(self.fov_rad / 2.0), 0.2, 1.5))

        theta = 2.0 * math.pi * self.revolutions * frame_idx / max(total - 1, 1)
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        eye = centroid + dist * np.array([math.sin(theta) * ce, -se, math.cos(theta) * ce])
        right, up, fwd = self._look_at(eye, centroid)

        # 地面网格（最下手部点下方 0.05m 平面）
        if self.ground_grid:
            self._draw_grid(img, right, up, fwd, eye,
                            float(valid_all[:, 1].max()) + 0.05,
                            float(centroid[2]), float(centroid[0]))
        # 相机系原点坐标轴
        self._draw_axes(img, right, up, fwd, eye)

        # painter 算法：远手先画
        order = []
        for slot in range(2):
            fin = finite[slot]
            if not fin.any():
                continue
            order.append((float(((pts[slot, fin] - eye) @ fwd).mean()), slot))
        order.sort(key=lambda t: -t[0])

        proj_all, _ = self._project(pts.reshape(-1, 3), right, up, fwd, eye)
        proj_all = proj_all.reshape(2, 21, 2)
        for _, slot in order:
            self._draw_hand(img, proj_all[slot], finite[slot], labels[slot], errs[slot])

        # 腕部深度标注
        if self.depth_labels:
            for slot in range(2):
                if finite[slot, 0]:
                    px = proj_all[slot, 0]
                    if 0 <= px[0] < self.width and 0 <= px[1] < self.height:
                        col = (0, 255, 0) if labels[slot] == "Right" else (255, 200, 0)
                        txt = f"{labels[slot]}  z={pts[slot, 0, 2]:.2f}m"
                        if np.isfinite(errs[slot]):
                            txt += f"  err={errs[slot]:.1f}px"
                        cv2.putText(img, txt, (int(px[0]) + 12, int(px[1]) - 12),
                                    FONT, 0.5, col, 2, cv2.LINE_AA)

        # HUD
        cv2.putText(img, f"frame {frame_idx}/{total}", (16, 30),
                    FONT, 0.7, self.text_color, 2, cv2.LINE_AA)
        cv2.putText(img, title, (16, self.height - 14),
                    FONT, 0.5, self.text_color, 1, cv2.LINE_AA)
        return img


# ═══════════════════════════════════════════════════════════════════
# 输入源 / 渲染输入链稳定性（live_demo.py 的 ReplaySource + M1/M3 组件）
# ═══════════════════════════════════════════════════════════════════

def find_video(session_path: str, cam: str) -> str | None:
    """定位录制视频:videos/<cam>/chunk-0000/<cam>.mp4(chunk_0000 下划线
    变体一并兼容,采集端两代命名),回退 videos/<cam>.mp4。"""
    for chunk in ("chunk-0000", "chunk_0000"):
        p = os.path.join(session_path, "videos", cam, chunk, f"{cam}.mp4")
        if os.path.isfile(p):
            return p
    p = os.path.join(session_path, "videos", f"{cam}.mp4")
    if os.path.isfile(p):
        return p
    return None


class ReplaySource:
    """录制会话回放：帧 n ↔ {n+1:06d}.png（1-based）。

    pace=True 按录制 fps 步调（处理快时 sleep 补齐）；pace=0 不限速
    （批处理后处理用）。
    """

    def __init__(self, session: str, pace: float = 30.0):
        video = find_video(session, "d435_rgb")
        if not video:
            sys.exit(f"错误: 找不到 RGB 视频: {session}/videos/d435_rgb/")
        self._cap = cv2.VideoCapture(video)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        depth_dir = os.path.join(session, "depth", "d435_depth")
        self._depth_files = {}
        if os.path.isdir(depth_dir):
            for p in glob.glob(os.path.join(depth_dir, "*.png")):
                try:
                    self._depth_files[int(os.path.basename(p).split(".")[0])] \
                        = p
                except ValueError:
                    pass
        self._pace = float(pace)      # 0 = 不限速
        self._t0 = time.perf_counter()
        self.n = -1

    def next(self):
        ok, rgb = self._cap.read()
        if not ok:
            return None, None
        self.n += 1
        dp = self._depth_files.get(self.n + 1)
        if dp is None:
            d = None
        else:
            d = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if self._pace > 0:
            due = self._t0 + (self.n + 1) / self._pace
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(min(wait, 0.1))
        return rgb, d

    def close(self):
        self._cap.release()


def _nan_pair(label: str = "") -> D435Pair:
    return D435Pair(result=LiftResult(np.full((21, 3), np.nan, np.float64),
                                      float("nan"), 0), left_label=label)


def _pred_pair(pred: np.ndarray, label: str = "") -> D435Pair:
    return D435Pair(result=LiftResult(np.asarray(pred, np.float64)
                                      .reshape(21, 3), float("nan"), 0),
                    left_label=label)


def _ws_agree(prev, cur, tol: float = 0.30) -> bool:
    """M3② wholesale 两帧确认：相邻两帧被门控观测的质心距离 < tol 判互相一致。"""
    if prev is None:
        return False
    pa = np.asarray(prev, np.float64).reshape(21, 3)
    ca = np.asarray(cur, np.float64).reshape(21, 3)
    fa = np.isfinite(pa).all(axis=1)
    fb = np.isfinite(ca).all(axis=1)
    if fa.sum() < 4 or fb.sum() < 4:
        return False
    return bool(np.linalg.norm(np.median(pa[fa], axis=0)
                               - np.median(ca[fb], axis=0)) < tol)


class _SoftSmoother:
    """M3① 包装 Hand3DSmoother：镜像其 label 变化/"空→有"重建判定。

    重建帧且几何近（<0.1m，同一只手漏检回归/标签闪烁）时喂 0.5 混合
    输入（旧平滑输出 + 新观测）软衔接——否则 pop 滤波器后首帧输出=
    原始输入 → snap；几何远（真换手）不混，保持硬重置防跨手污染。
    """

    _MIN_PTS = 4          # 质心可靠下限
    _SOFT_DIST = 0.10     # 重建帧软衔接判距（米）

    def __init__(self, smoother):
        self._sm = smoother
        self._prev_out = None
        self._prev_labels = [None, None]
        self._prev_pres = [False, False]

    def update(self, h3, labels, valids):
        pres_flags = [v >= 8 for v in valids]
        if self._prev_out is not None:
            for s in range(2):
                if (labels[s] != self._prev_labels[s]
                        or (pres_flags[s] and not self._prev_pres[s])):
                    po = self._prev_out[s]
                    pofin = np.isfinite(po).all(axis=1)
                    nfin = np.isfinite(h3[s]).all(axis=1)
                    if pofin.sum() >= self._MIN_PTS \
                            and nfin.sum() >= self._MIN_PTS:
                        dc = float(np.linalg.norm(
                            np.median(po[pofin], axis=0)
                            - np.median(h3[s, nfin], axis=0)))
                        if dc < self._SOFT_DIST:
                            h3[s] = np.where(nfin[:, None],
                                             0.5 * po + 0.5 * h3[s], po)
        out = self._sm.update(h3, labels, valids)
        self._prev_out = out
        self._prev_labels = list(labels)
        self._prev_pres = pres_flags
        return out


class _CentroidAnchor:
    """M1 质心锚定：整手共模跳抑制（相机静止前提）。

    每槽有效点中位质心 c 走强 OneEuro（freq_min=3.0, beta=0.3,
    dcutoff=0.3m/s：静止抖动被强衰减，>0.3m/s 的真实手部运动快速通过），
    输出 = 输入 + (ĉ − c)——共模平移在质心层被吸收，手内形状与手势
    动力学原样保留。label 变化且几何近（<0.1m，同一只手重建）时软衔接
    （0.5 混合旧 ĉ 与新 c）防重置 snap；几何远（真换手）硬重置。
    """

    _MIN_PTS = 4          # 有效点下限（质心不可靠则跳过该槽）
    _SOFT_DIST = 0.10     # 重建帧软衔接判距（米）

    def __init__(self):
        self._filters = {}          # slot → OneEuroFilter3D
        self._prev_labels = [None, None]
        self._prev_c = [None, None]      # 上一帧输出质心（软衔接用）
        self._t0 = time.perf_counter()

    def apply(self, hands3d, labels) -> np.ndarray:
        pts = np.asarray(hands3d, np.float64).reshape(2, 21, 3)
        out = pts.copy()
        ts = (time.perf_counter() - self._t0) * 1000.0
        for s in range(2):
            fin = np.isfinite(pts[s]).all(axis=1)
            if fin.sum() < self._MIN_PTS:
                self._filters.pop(s, None)      # 空槽丢弃滤波器（重现时从观测起）
                self._prev_c[s] = None
                self._prev_labels[s] = labels[s]
                continue
            c = np.median(pts[s, fin], axis=0)
            if labels[s] != self._prev_labels[s]:
                if (self._filters.get(s) is not None
                        and self._prev_c[s] is not None
                        and np.linalg.norm(self._prev_c[s] - c)
                        < self._SOFT_DIST):
                    c = 0.5 * self._prev_c[s] + 0.5 * c    # 软衔接防 snap
                self._filters[s] = OneEuroFilter3D(3.0, 0.3, 0.3)
            if s not in self._filters:   # 空帧 pop 后同 label 重现：从观测起
                self._filters[s] = OneEuroFilter3D(3.0, 0.3, 0.3)
            c_hat = np.asarray(self._filters[s](c[0], c[1], c[2], ts),
                               np.float64)
            out[s, fin] = pts[s, fin] + (c_hat - c)
            self._prev_c[s] = c_hat
            self._prev_labels[s] = labels[s]
        return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 视频输出（mp4v 先写临时文件，收尾探测 ffmpeg 转 H.264+yuv420p）
# ═══════════════════════════════════════════════════════════════════

# ffmpeg 候选链：PATH 中查找 → 常见安装位置（Windows/Linux）
FFMPEG_CANDIDATES = list(dict.fromkeys(c for c in (
    shutil.which("ffmpeg"),
    shutil.which("ffmpeg.exe"),
    os.path.join(_SCRIPT_DIR, "ffmpeg.exe"),
    "C:\\ffmpeg\\bin\\ffmpeg.exe",
    "/usr/bin/ffmpeg",
) if c))


def _convert_to_h264(tmp_path, out_path):
    """把 mp4v 临时文件转码为 H.264+yuv420p（无 ffmpeg 时保留 mp4v）。"""
    print("  转码 H.264...")
    last_err = ""
    for ff in FFMPEG_CANDIDATES:
        try:
            ret = subprocess.run([
                ff, "-y",
                "-i", tmp_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                out_path,
            ], capture_output=True, text=True)
        except OSError:
            continue        # 该候选不存在/无法执行，试下一个
        if ret.returncode == 0:
            os.remove(tmp_path)
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  已保存: {out_path}  ({size_mb:.1f} MB)")
            return True
        last_err = ret.stderr[-500:] if ret.stderr else f"exit={ret.returncode}"
    print(f"[提示] 未找到可用的 ffmpeg，已保留 mp4v 编码视频（VLC/PotPlayer"
          f" 可直接播放）: {tmp_path}")
    if last_err:
        print(f"       (ffmpeg 输出: {last_err})")
    return False


class VideoSink:
    """一路输出视频：mp4v 先写临时文件，收尾转 H.264。"""

    def __init__(self, out_path: str, fps: float, size):
        self.out_path = out_path
        self.tmp_path = os.path.splitext(out_path)[0] + "_tmp.mp4"
        self.writer = cv2.VideoWriter(self.tmp_path,
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, size)
        if not self.writer.isOpened():
            sys.exit(f"[错误] 无法创建输出视频: {self.tmp_path}"
                     f"（请检查输出目录是否可写）")
        self.frames = 0

    def write(self, img: np.ndarray):
        self.writer.write(img)
        self.frames += 1

    def finish(self) -> str:
        """收尾：释放写入器并转 H.264，返回最终视频路径。"""
        self.writer.release()
        if _convert_to_h264(self.tmp_path, self.out_path):
            return self.out_path
        return self.tmp_path


# ═══════════════════════════════════════════════════════════════════
# 标定解析
# ═══════════════════════════════════════════════════════════════════

def _resolve_calib(calib_arg: str | None) -> dict:
    """标定解析顺序：--calib JSON > 脚本旁 d435_color_calib.json > 内嵌标定。

    --calib 显式指定时文件必须存在且可用，否则报错退出（不静默回退）。
    """
    if calib_arg:
        if not os.path.isfile(calib_arg):
            sys.exit(f"[错误] 标定文件不存在: {calib_arg}")
        try:
            with open(calib_arg, encoding="utf-8") as f:
                calib = json.load(f)
            _check_calib(calib, calib_arg)
            print(f"标定: --calib 指定: {calib_arg}")
            return calib
        except (OSError, json.JSONDecodeError, ValueError) as e:
            sys.exit(f"[错误] 标定文件不可用: {calib_arg}（{e}）")
    side = os.path.join(_SCRIPT_DIR, "d435_color_calib.json")
    if os.path.isfile(side):
        try:
            with open(side, encoding="utf-8") as f:
                calib = json.load(f)
            _check_calib(calib, side)
            print(f"标定: 脚本同目录 d435_color_calib.json: {side}")
            return calib
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"[警告] 脚本旁标定文件无法使用（{e}），改用内嵌标定")
    print("标定: 内嵌标定（本 demo 打包的开发机 D435 标定）")
    return json.loads(json.dumps(_EMBEDDED_CALIB))


def _check_calib(calib: dict, path: str) -> None:
    """校验标定 JSON 关键字段（内参或 R/t 缺失/为零 → ValueError）。"""
    for key in ("color_intrinsics", "depth_intrinsics", "depth_to_color"):
        if key not in calib:
            raise ValueError(f"缺少字段 {key}")
    ci = calib["color_intrinsics"]
    dtc = calib["depth_to_color"]
    if not all(ci.get(k) for k in ("fx", "fy", "cx", "cy")):
        raise ValueError("彩色内参缺失/为零")
    if "rotation" not in dtc or "translation" not in dtc:
        raise ValueError("depth_to_color 缺少 rotation/translation")


# ═══════════════════════════════════════════════════════════════════
# 主流程（处理链与 live_demo.py 逐行一致，仅替换批处理/输出段）
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="D435 录制会话 3D 手部关键点后处理 Demo（自包含单文件，"
                    "零仓库依赖，输出三路处理视频）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python d435_hands_demo.py data/recordings/222/222_000011\n"
            "  python d435_hands_demo.py data/recordings/222/222_000011 "
            "--out-dir result --depth-overlay\n"))
    ap.add_argument("session_dir", help="主程序录制会话目录（含 videos/d435_rgb、"
                                        "depth/d435_depth、calibration/head_stereo.json）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（默认 <会话目录>/d435_demo_output/）")
    ap.add_argument("--calib", default=None,
                    help="D435 彩色标定 JSON（默认脚本同目录 d435_color_calib.json，"
                         "再回退内嵌标定；深度内参以会话 head_stereo.json 为准）")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"hand_landmarker.task 路径（默认本脚本同目录）")
    ap.add_argument("--det-conf", type=float, default=0.4,
                    help="掌心检测置信度阈值（默认 0.4；动作快/丢手可再降到 0.3）")
    ap.add_argument("--track-conf", type=float, default=0.4,
                    help="手部跟踪置信度阈值（默认 0.4；丢手可再降到 0.3）")
    ap.add_argument("--propagate-max", type=int, default=15,
                    help="槽位丢失帧数硬顶（超限判 absent 不幻觉，默认 15）")
    ap.add_argument("--fill", type=int, default=1, choices=(1, 2, 3),
                    help="对齐空穴回填轮数（默认 1；深度空洞多可试 3）")
    ap.add_argument("--depth-overlay", action="store_true",
                    help="第 1 路视频叠 300-1200mm 深度伪彩（默认不叠）")
    args = ap.parse_args()

    session = args.session_dir.rstrip("/").rstrip("\\")
    if not os.path.isdir(session):
        sys.exit(f"错误: 会话目录不存在: {session}")
    video = find_video(session, "d435_rgb")
    if not video:
        sys.exit(f"错误: 找不到 RGB 视频: {session}/videos/d435_rgb/\n"
                 f"请确认会话目录布局为 主程序录制结构：\n"
                 f"  {session}/videos/d435_rgb/chunk-0000/d435_rgb.mp4\n"
                 f"  {session}/depth/d435_depth/NNNNNN.png\n"
                 f"  {session}/calibration/head_stereo.json")
    if not os.path.exists(args.model):
        sys.exit(f"[错误] 模型文件不存在: {args.model}\n"
                 f"请把 hand_landmarker.task 放到 {_SCRIPT_DIR} 目录，"
                 f"或用 --model 指定路径。")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"[错误] 无法打开视频: {video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in <= 0:
        fps_in = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = args.out_dir or os.path.join(session, "d435_demo_output")
    os.makedirs(out_dir, exist_ok=True)

    # ── 标定 + 对齐器 ──────────────────────────────────────
    calib = _resolve_calib(args.calib)
    session_depth = load_session_depth_intr(session)
    if session_depth is None:
        print("[警告] 录制 head_stereo.json 缺失，深度内参改用标定文件"
              "（或内嵌标定）值")
        session_depth = calib["depth_intrinsics"]
    else:
        # 交叉核对：会话深度内参与标定深度内参 fx/fy 偏差 >1% → 疑似换机
        ref = calib["depth_intrinsics"]
        if ref.get("fx") and ref.get("fy"):
            dfx = abs(session_depth["fx"] - ref["fx"]) / ref["fx"]
            dfy = abs(session_depth["fy"] - ref["fy"]) / ref["fy"]
            if max(dfx, dfy) > _FX_REL_TOL:
                print(f"[警告] 会话深度内参与标定深度内参偏差 "
                      f"{max(dfx, dfy) * 100:.1f}% (>1%)：疑似换机录制，"
                      f"建议提供该设备标定（--calib）")
    aligner = LiveAligner(calib["color_intrinsics"], calib["depth_to_color"],
                          session_depth, fill_passes=args.fill)
    color_intr = (aligner.fx_c, aligner.fy_c, aligner.cx_c, aligner.cy_c)

    # ── 检测/跟踪/平滑组件 ─────────────────────────────────
    det = MediaPipeDetector(num_hands=2, det_conf=args.det_conf,
                            track_conf=args.track_conf)
    voter = HandednessVoter()
    tracker = HandSlotTracker(max_lost=args.propagate_max)
    smoother = Hand3DSmoother()
    soft_smoother = _SoftSmoother(smoother)   # M3①：重建帧几何近时软衔接
    renderer = RotatingSkeletonRenderer(*RENDER_SIZE, revolutions=1.0)
    renderer.elevation = math.radians(_VIEW_ELEV0)   # 静态视角俯仰 25°

    # ── 输出三路视频 ───────────────────────────────────────
    sinks = [
        VideoSink(os.path.join(out_dir, "1_rgb_2d_overlay.mp4"), fps_in,
                  RENDER_SIZE),
        VideoSink(os.path.join(out_dir, "2_hand_3d.mp4"), fps_in, RENDER_SIZE),
        VideoSink(os.path.join(out_dir, "3_depth_colormap.mp4"), fps_in,
                  RENDER_SIZE),
    ]

    print(f"\n会话: {session}")
    print(f"视频: {video}  ({w_in}x{h_in} @ {fps_in:.0f} fps, "
          f"{total if total > 0 else '?'} 帧)")
    print(f"输出: {out_dir}/  （三路视频，1280×720）")
    print(f"检测: CPU, det/track conf {args.det_conf}/{args.track_conf}, "
          f"--fill {args.fill} 轮填洞, 传播上限 {args.propagate_max} 帧")
    print("处理中（批处理全速）...")

    source = ReplaySource(session, pace=0)

    lost_counts = [0, 0]     # 各槽连续丢失帧数（assigner 困境槽无门限救援用）
    zc_slot = [None, None]   # M5：槽级补点深度先验（EMA，换手/首帧取实测）
    ws_prev = [None, None]   # M3②：wholesale 两帧确认——上一帧被门控观测
    ws_streak = [0, 0]       # M3②：连续 wholesale 帧数（≥3 强制采信防死锁）
    gate_streak = [np.zeros(21, np.int64), np.zeros(21, np.int64)]
                             # M6：逐点连续被门控帧数（≥_GATE_FORGIVE 采信观测）
    centroid_anchor = _CentroidAnchor()   # M1：质心强平滑 + 共模平移校正
    slot_stats = {"real": [0, 0], "propagated": [0, 0], "absent": [0, 0]}
    view_anchor = None   # 3D 视图世界锚点：首帧有手时锁定（相机目标恒定，
                         # 视角不随手漂移；只平移不改手势）
    n = 0
    t0 = time.perf_counter()

    try:
        while True:
            rgb, d = source.next()
            if rgb is None:
                break

            if d is None or d.shape[:2] != (aligner.dh, aligner.dw):
                aligned = np.zeros((aligner.ch, aligner.cw), np.float32)
            else:
                aligned = aligner.align_depth_to_color(d)

            hands = det.detect(rgb)
            # 空帧不喂 voter：空帧会清空轨迹（scene reset 语义），短暂漏检
            # 会清票仓 → 重建期原始 label 闪烁 → 两手同 label。跳过空帧
            # 让轨迹 idle 保持。
            if hands:
                voter.update(hands, frame_w=rgb.shape[1],
                             frame_h=rgb.shape[0], frame=n, cam="d435")

            pairs = [lift_hand(hd, aligner, aligned) for hd in hands]
            # voter 重建期两手同 label：label 不可信，直接按 label 分配会
            # 有一手被标签守卫拒收。同 label 时先清空 label 走几何分配，
            # 观察时用槽自身 label（防 observe_slot 误判换手重置槽位）。
            same_lab = (len(pairs) == 2 and pairs[0].left_label
                        and pairs[0].left_label == pairs[1].left_label)
            if same_lab:
                for p in pairs:
                    p.left_label = ""
            out = assign_mono_slots(pairs, tracker, n, color_intr,
                                    lost_counts=tuple(lost_counts))
            if same_lab:
                for s in range(2):
                    if out[s] is not None:
                        sl = tracker.slot_label(s)
                        if sl:
                            out[s].left_label = sl

            slot_pairs, slot_dets, states = [], [], []
            for s in range(2):
                if out[s] is not None:
                    p = out[s]
                    # M5：补点深度锚定到槽级稳定 zc（逐帧独立中位是整手
                    # 共模跳的最大来源；实测点不动，补点 x,y 随 zc 反投影
                    # 保持与 2D 一致）。换手（观察 label ≠ 槽 label）取实测。
                    meas = getattr(p, "measured", None)
                    if meas is not None and meas.any():
                        pts3d = np.asarray(p.result.points_3d, np.float64) \
                            .reshape(21, 3)
                        zf = float(np.median(pts3d[meas, 2]))
                        if tracker.slot_label(s) != p.left_label \
                                or zc_slot[s] is None:
                            zc_slot[s] = zf
                        else:
                            zc_slot[s] = 0.5 * zc_slot[s] + 0.5 * zf
                        apply_slot_zc(p, zc_slot[s], aligner)
                    # 时序一致性门：与槽预测差 >150mm 的点判可疑置 NaN
                    gated, wholesale = gate_observations(
                        p.result.points_3d, tracker.predict(s, n))
                    # M6：门控锁死豁免 —— 关节被门控后 tracker 只走纯
                    # 预测（不更新），预测外推越走越远、|观测−预测|
                    # 永远 >150mm，关节点直到手离场重入（label 变化/
                    # 长缺口重初始化）才恢复。连续被门控 ≥_GATE_FORGIVE
                    # 帧且观测已恢复有限时采信观测：放行写入状态，αβ
                    # 每帧收敛一半，门控自然恢复后 streak 清零。换手帧
                    # （label 变化，旧状态对比无意义）不豁免。
                    if not wholesale:
                        if tracker.slot_label(s) != p.left_label:
                            gate_streak[s][:] = 0
                        meas3d = np.asarray(p.result.points_3d, np.float64) \
                            .reshape(21, 3)
                        g_fin = np.isfinite(
                            np.asarray(gated, np.float64)
                            .reshape(-1, 3)).all(axis=1)
                        m_fin = np.isfinite(meas3d).all(axis=1)
                        gs = gate_streak[s]
                        latched = ~g_fin & m_fin
                        gs[latched] += 1
                        gs[~latched] = 0
                        forgive = (gs >= _GATE_FORGIVE) & m_fin
                        if forgive.any():
                            gated[forgive] = meas3d[forgive]
                    if wholesale:
                        # M3②：整手级不匹配先两帧确认。连续两帧观测互相
                        # 一致才判槽状态过时 → 借 label 翻转触发槽位重置、
                        # 随即真观测干净初始化；单帧跳变/误检不采信不重置
                        # ——本帧走预测显示，状态不动。
                        if _ws_agree(ws_prev[s], gated) or ws_streak[s] >= 3:
                            tracker.observe_slot(s, "\x00reset",
                                                 np.full((21, 3), np.nan), n)
                            tracker.observe_slot(s, p.left_label, gated, n)
                            p.result.points_3d = gated
                            lost_counts[s] = 0
                            ws_prev[s] = None
                            ws_streak[s] = 0
                            gate_streak[s][:] = 0     # M6：状态重播种，streak 归零
                            slot_pairs.append(p)
                            slot_dets.append(p.det)
                            states.append("real")
                        else:
                            ws_prev[s] = gated
                            ws_streak[s] += 1
                            pred_now = tracker.predict(s, n)
                            if pred_now is not None:
                                slot_pairs.append(_pred_pair(
                                    pred_now, tracker.slot_label(s)))
                                slot_dets.append(None)
                                states.append("propagated")
                            else:
                                slot_pairs.append(_nan_pair(
                                    tracker.slot_label(s)))
                                slot_dets.append(None)
                                states.append("absent")
                    else:
                        tracker.observe_slot(s, p.left_label, gated, n)
                        p.result.points_3d = gated
                        lost_counts[s] = 0
                        ws_prev[s] = None
                        ws_streak[s] = 0
                        slot_pairs.append(p)
                        slot_dets.append(p.det)
                        states.append("real")
                else:
                    pred = tracker.predict(s, n)
                    tracker.mark_lost(s, n)
                    lost_counts[s] += 1
                    ws_prev[s] = None
                    ws_streak[s] = 0
                    if pred is not None:
                        slot_pairs.append(_pred_pair(pred,
                                                     tracker.slot_label(s)))
                        slot_dets.append(None)
                        states.append("propagated")
                    else:
                        slot_pairs.append(_nan_pair(tracker.slot_label(s)))
                        slot_dets.append(None)
                        states.append("absent")
            for s, st in enumerate(states):
                slot_stats[st][s] += 1

            presents = [st != "absent" for st in states]
            propagated = [st == "propagated" for st in states]
            labels = [slot_pairs[s].left_label if out[s] is not None
                      else tracker.slot_label(s) for s in range(2)]

            # (2,21,3) 槽位 3D（tracker αβ 已平滑）→ OneEuro 再平滑压静止
            # 抖动（M3① _SoftSmoother 包装：重建帧几何近时 0.5 混合软衔接）
            h3 = np.stack([np.asarray(p.result.points_3d, np.float64)
                           .reshape(21, 3) for p in slot_pairs])
            valids = [int(np.isfinite(h3[s]).all(axis=1).sum())
                      for s in range(2)]
            smoothed = soft_smoother.update(h3, labels, valids)
            # M1 质心锚定：质心强 OneEuro + 共模平移校正（仅展示路径，
            # 不回流 tracker）
            renderer_in = centroid_anchor.apply(smoothed, labels)

            # 2D：real 帧画检测骨架；propagated/absent 传 NaN 不画
            hands2d = np.stack([
                np.asarray(sd.landmarks, np.float32).reshape(21, 2)
                if sd is not None else np.full((21, 2), np.nan)
                for sd in slot_dets])

            # 3D 视角固定：渲染器每帧以输入有效点均值质心为相机目标、
            # 网格原点与缩放依据——把输入整体平移到世界锚点 view_anchor，
            # 目标即恒定：视角/网格/缩放完全静止，手的真实世界运动在
            # 固定网格中可见。锚点首帧有手时锁定（平移不改手势/缩放）。
            if view_anchor is None:
                fin_all = np.isfinite(renderer_in).all(axis=2)
                if fin_all.sum() >= 4:
                    view_anchor = renderer_in[fin_all].mean(axis=0)
            if view_anchor is not None:
                fin_all = np.isfinite(renderer_in).all(axis=2)
                if fin_all.sum() >= 4:
                    renderer_in = renderer_in - (
                        renderer_in[fin_all].mean(axis=0) - view_anchor)

            base = blend_depth(rgb, aligned, 0.35) if args.depth_overlay \
                else rgb
            ov = draw_overlay(base, hands2d, smoothed, labels, propagated,
                              presents, n + 1, max(total, n + 1),
                              "D435 offline 3D hands")
            # 静态正面视角：frame_idx=179.5（θ=π 精确正面，与开发环境
            # 实时 demo 手动复位视角一致），俯仰 25°
            rot = renderer.render(renderer_in, labels, (np.nan, np.nan),
                                  _STATIC_FRAME_IDX, _ROT_TOTAL,
                                  "D435 hand keypoints (color-cam, m)")
            # 深度图伪彩：aligned 深度（mm）→ 0.3-1.5m JET
            dimg = cv2.applyColorMap(
                (np.clip(aligned / 1000.0 - 0.3, 0.0, 1.2) / 1.2 * 255)
                .astype(np.uint8), cv2.COLORMAP_JET)
            cv2.putText(dimg, "aligned depth 0.3-1.5 m",
                        (12, dimg.shape[0] - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (80, 220, 255), 1, cv2.LINE_AA)

            sinks[0].write(ov)
            sinks[1].write(rot)
            sinks[2].write(dimg)
            n += 1

            if n % 30 == 0:
                elapsed = time.perf_counter() - t0
                pct = 100.0 * n / total if total > 0 else 0.0
                print(f"  帧 {n}/{total if total > 0 else '?'} "
                      f"({pct:.0f}%)  平均 {n / elapsed:.1f} fps")
    except KeyboardInterrupt:
        print("\n[中断] 用户中断，正在收尾（已处理帧照常输出）")
    finally:
        source.close()
        cap.release()
        det.close()

    # ── 收尾：转码 + 汇总 ──────────────────────────────────
    elapsed = time.perf_counter() - t0
    print(f"\n处理完成: {n} 帧, {elapsed:.1f}s, 平均 {n / elapsed:.1f} fps")
    for s in range(2):
        print(f"  slot{s} (hand_{s}): real {slot_stats['real'][s]} | "
              f"propagated {slot_stats['propagated'][s]} | "
              f"absent {slot_stats['absent'][s]}")
    print("")
    final_paths = []
    for sink in sinks:
        final_paths.append(sink.finish())
    print(f"\n输出目录: {out_dir}")
    for p in final_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
