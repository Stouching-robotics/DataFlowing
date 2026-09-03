#!/usr/bin/env python3
"""render_overlay.py —— D435 RGB 单视叠加渲染。

2D 骨架（分色，复用 hand_detection FINGERS 色表 + 自含 PALM_CONNECTIONS）
+ 逐关键点平滑深度 mm 标注 + 槽位 label/propagated HUD。
另含 blend_depth：伪彩深度叠层（探针与 --depth-overlay 共用）。
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from hand_detection.hand_pipeline_mediapipe import FINGERS  # noqa: E402

# 掌心连接（MediaPipe 21 点拓扑的腕-掌子集，与 renderer_3d 同）
PALM_CONNECTIONS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

# 每指画法：拇指 1→2→3→4，其余指 0→(掌根)→MCP→PIP→DIP→指尖
FINGER_CHAINS = {name: (ids if name == "Thumb" else [0] + ids)
                 for name, (ids, _) in FINGERS.items()}

FONT = cv2.FONT_HERSHEY_SIMPLEX


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


# ═══════════════════════════════════════════════════════
#  深度码值 JET —— 与主程序 core/depth_codec 口径一致
# ═══════════════════════════════════════════════════════
# 分发包（build_dist.sh）不含 core/，这里内置同参副本；改动须与
# core/depth_codec.py 两处同步。
_DEPTH_MIN_MM = 100.0
_DEPTH_MAX_MM = 5000.0
_DEPTH_QMAX = 4095
_LOG_LO = math.log(_DEPTH_MIN_MM)
_LOG_STEP = _DEPTH_QMAX / (math.log(_DEPTH_MAX_MM) - _LOG_LO)


def _depth_codes(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 毫米深度 → 12-bit 对数码（int32，无效=0，钳域）。"""
    mm = np.asarray(depth_mm, dtype=np.float64)
    valid = mm > 0
    codes = np.zeros(mm.shape, dtype=np.int32)
    if valid.any():
        codes[valid] = np.clip(
            np.rint((np.log(mm[valid]) - _LOG_LO) * _LOG_STEP),
            0, _DEPTH_QMAX).astype(np.int32)
    return codes


def depth_to_heatmap_bgr(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 毫米深度 → BGR JET（12-bit 对数码色标）。

    与主程序实时显示/存储文件回放同构（码值 → JET，log 域）。
    """
    codes = _depth_codes(depth_mm)
    c8 = (np.clip(codes, 0, _DEPTH_QMAX) * 255 // _DEPTH_QMAX).astype(np.uint8)
    return cv2.applyColorMap(c8, cv2.COLORMAP_JET)


def blend_depth(rgb: np.ndarray, aligned_mm: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
    """伪彩深度叠层：12-bit 对数码色标 JET（无效=不叠），α 混合回 BGR。

    色标与主程序/存储同构（近场 log 域区分度优于旧 300-1200mm 线性）。
    """
    codes = _depth_codes(aligned_mm)
    c8 = (np.clip(codes, 0, _DEPTH_QMAX) * 255 // _DEPTH_QMAX).astype(np.uint8)
    colored = cv2.applyColorMap(c8, cv2.COLORMAP_JET)
    mask = (codes > 0).astype(np.float32)[..., None]
    return (rgb.astype(np.float32) * (1.0 - alpha * mask)
            + colored.astype(np.float32) * (alpha * mask)).astype(np.uint8)
