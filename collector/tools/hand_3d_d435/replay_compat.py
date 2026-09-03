#!/usr/bin/env python3
"""replay_compat.py —— 录制会话布局/标定兼容层（live_demo --replay 用）。

新会话（槽名随 GUI 用户命名后）布局与大写槽名、双 RealSense 并存：
  videos/D435_depth_rgb/chunk-0000/D435_depth_rgb.mp4 + depth/D435_depth/
  calibration/head_stereo.json 可能是 D405 内参（fx≈645），D435 深度
  内参在 calibration/D435_depth_rgb_calibration.json（fx=429.47）。
旧会话（222 等）：videos/d435_rgb + depth/d435_depth + head_stereo.json
即 D435 内参、无 D435_depth_rgb_calibration.json。

本模块自动探测三种情形，旧会话行为与原来完全一致。
"""

from __future__ import annotations

import json
import os

from hand_3d_d435.depth_align import load_session_depth_intr

# 深度内参交叉核对容差（与 depth_align._FX_REL_TOL 一致）
_FX_REL_TOL = 0.01


def find_video_any(session: str) -> str | None:
    """依次尝试小写/大写槽名的 RGB 视频（chunk-0000 布局兼容）。"""
    for cam in ("d435_rgb", "D435_depth_rgb"):
        for p in (os.path.join(session, "videos", cam, "chunk-0000",
                               f"{cam}.mp4"),
                  os.path.join(session, "videos", f"{cam}.mp4")):
            if os.path.isfile(p):
                return p
    return None


def find_depth_dir(session: str) -> str | None:
    """依次尝试小写/大写槽名的深度 PNG 目录。"""
    for cam in ("d435_depth", "D435_depth"):
        p = os.path.join(session, "depth", cam)
        if os.path.isdir(p):
            return p
    return None


def load_session_depth_intr_any(session: str) -> dict | None:
    """head_stereo.json 深度内参，D405 误读时改读 D435_depth_rgb_calibration.json。

    双 RealSense 会话里 head_stereo.json 可能存的是 D405 内参（fx≈645），
    配到 848×480 的 D435 深度图上对齐全错；该会话的 D435 深度内参在
    同目录 D435_depth_rgb_calibration.json（depth_camera 段，fx=429.47）。
    判定：两者都存在且 fx 差 >1% 时，以 D435 文件为准（head_stereo 记录
    的"最后写入"设备不可靠，D435 文件按槽名命名、语义明确）。
    """
    head = load_session_depth_intr(session)
    d435_path = os.path.join(session, "calibration",
                             "D435_depth_rgb_calibration.json")
    if not os.path.isfile(d435_path):
        return head          # 旧会话：head_stereo 即 D435，行为不变
    try:
        with open(d435_path, encoding="utf-8") as f:
            dc = json.load(f)["depth_camera"]
        fx = float(dc["intrinsic"][0])
        d435 = {
            "fx": fx, "fy": float(dc["intrinsic"][1]),
            "cx": float(dc["intrinsic"][2]), "cy": float(dc["intrinsic"][3]),
            "width": int(dc.get("resolution", [848, 480])[0]),
            "height": int(dc.get("resolution", [848, 480])[1]),
        }
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return head
    if head is not None and abs(head["fx"] - fx) / fx <= _FX_REL_TOL:
        return head          # 一致：head_stereo 本来就是 D435
    if head is not None:
        print(f"警告: head_stereo.json 深度内参 fx={head['fx']:.1f} 与 "
              f"D435_depth_rgb_calibration.json fx={fx:.1f} 差 >1%"
              f"（head_stereo 疑为 D405 内参）→ 改用 D435 标定")
    return d435
