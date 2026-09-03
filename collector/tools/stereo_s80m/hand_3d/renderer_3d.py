#!/usr/bin/env python3
"""
3D 旋转视角骨架渲染器 —— numpy + cv2 自写透视投影（零额外依赖）。

虚拟相机绕双手质心匀速旋转（默认整段视频转 2 圈，仰角 25°），
五指分色骨架（与 2D demo 同一套 FINGERS 色表）+ 掌心灰连接 + 腕部白圆
+ 地面网格 + 腕部深度标注 + 相机系坐标轴 + HUD。

对比 matplotlib 3D：本渲染器 2-5ms/帧（matplotlib 60-150ms/帧）、
视角完全确定（无轴自动缩放抖动）、原生 BGR 直进 VideoWriter。
坐标系：左目相机系（OpenCV 约定，+X 右 / +Y 下 / +Z 前，米）。
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hand_detection.hand_pipeline_mediapipe import FINGERS  # noqa: E402

# 掌心连接（MediaPipe 21 点拓扑的腕-掌子集，本模块自含定义）
PALM_CONNECTIONS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

# 每指画法：拇指 1→2→3→4，其余指 0→(掌根)→MCP→PIP→DIP→指尖
FINGER_CHAINS = {name: (ids if name == "Thumb" else [0] + ids)
                 for name, (ids, _) in FINGERS.items()}

FONT = cv2.FONT_HERSHEY_SIMPLEX


class RotatingSkeletonRenderer:
    """离线 3D 旋转视角骨架渲染器（逐帧 render → VideoWriter）。"""

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
        """返回相机基 (right, up, fwd)。左目相机系 Y 向下 → 世界"上"= -Y。"""
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
                    r = 7 if idx == ids[-1] else 5      # 指尖大点/关节小点（与 draw_hand 同风格）
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

    def view_params(self, hands3d: np.ndarray):
        """(2,21,3) 米制点 → (质心, 相机距离, 网格 y)；无有效点时 None。

        render() 每帧据此定相机目标/缩放/网格位置；调用方想固定世界
        视角（相机不随手动）时可锁存首帧值经 fixed_view 传入。
        """
        pts = np.asarray(hands3d, dtype=np.float64).reshape(-1, 3)
        valid = pts[np.isfinite(pts).all(axis=1)]
        if valid.size == 0:
            return None
        centroid = valid.mean(axis=0)
        span = float(np.ptp(valid, axis=0).max())
        half = span / 2.0 if span > 1e-6 else 0.3
        dist = float(np.clip(2.2 * half / math.tan(self.fov_rad / 2.0), 0.2, 1.5))
        return centroid, dist, float(valid[:, 1].max()) + 0.05

    def render(self, hands3d: np.ndarray, labels=("", ""), errs=(np.nan, np.nan),
               frame_idx: int = 0, total: int = 1,
               title: str = "3D hand keypoints (left-cam frame, meters)",
               fixed_view=None) -> np.ndarray:
        img = np.full((self.height, self.width, 3), self.bg_color, np.uint8)
        pts = np.asarray(hands3d, dtype=np.float64).reshape(2, 21, 3)
        finite = np.isfinite(pts).all(axis=2)
        valid_all = pts[finite]

        # 视角参数：fixed_view=(质心, 距离, 网格 y) 时相机完全固定（世界
        # 视角，手在世界内自由运动）；默认 None = 逐帧随手（D435 行为）。
        if fixed_view is not None:
            centroid = np.asarray(fixed_view[0], dtype=np.float64)
            dist = float(fixed_view[1])
            grid_y = float(fixed_view[2])
        else:
            vp = self.view_params(hands3d)
            if vp is None:
                cv2.putText(img, "no valid 3D hand keypoints",
                            (60, self.height // 2),
                            FONT, 0.9, self.text_color, 2, cv2.LINE_AA)
                return img
            centroid, dist, grid_y = vp

        theta = 2.0 * math.pi * self.revolutions * frame_idx / max(total - 1, 1)
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        eye = centroid + dist * np.array([math.sin(theta) * ce, -se, math.cos(theta) * ce])
        right, up, fwd = self._look_at(eye, centroid)

        # 地面网格（最下手部点下方 0.05m 平面；fixed_view 时随锁定值不动）
        if self.ground_grid:
            self._draw_grid(img, right, up, fwd, eye,
                            grid_y, float(centroid[2]), float(centroid[0]))
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
