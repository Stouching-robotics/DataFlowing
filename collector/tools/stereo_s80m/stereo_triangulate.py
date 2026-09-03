#!/usr/bin/env python3
"""
S80M 双目三角化几何模块 —— 独立子模块，不依赖主程序 (core/ ui/ storage/)。

从标定 dict（config/s80m_stereo_calibration.json 或 episode calibration/head_stereo.json）
构建 StereoTriangulator：

  - distortion_model == "equidistant" → cv2.fisheye.stereoRectify + fisheye.undistortPoints
  - 其他模型                     → cv2.stereoRectify + cv2.undistortPoints

三角化输出的 3D 点在"左目（cam0）相机光学系"下（米制）：
  原点 = cam0 光心，+X 向右，+Y 向下，+Z 向前（OpenCV 约定）。
注意这是"物理位置"（与 MediaPipe 手局部系 world_landmarks 不同），
左右目同一时刻的关键点在此坐标系下统一。

用法（被 hand_triangulate.py 等独立脚本 import）::

    calib = load_stereo_calibration(session_path="/path/to/222_000002")
    tri   = StereoTriangulator(calib)            # 可选 swap_cams=True 处理左右目文件与 cam0/cam1 错位
    res   = tri.triangulate(lms_l, lms_r)        # lms 为原始(未矫正)图像素坐标 (21,2)
    res.points_3d   # (21,3) float64, 无效点为 NaN, 单位米
    res.reproj_error, res.valid_count, res.mean_error
    tri.rectified_image(frame, "left")           # 矫正后图像（可视化用）
    pairs = match_hands(left_hands, right_hands, tri)   # 跨目手匹配（几何主判据）

自检::

    python stereo_s80m/stereo_triangulate.py      # 3D→双图→三角化 往返误差测试
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

# ── 默认参数 ────────────────────────────────────────────────────
DEFAULT_MAX_REPROJ_ERR = 8.0     # 重投影误差阈值 (px)
DEFAULT_MAX_DEPTH = 3.0          # 有效深度上限 (米)
DEFAULT_MIN_DEPTH = 0.05         # 有效深度下限 (米)
MIN_VALID_POINTS = 8             # 判定"左右目拍到同一只手"的最少有效关键点数

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_CALIB_PATH = os.path.join(_REPO_ROOT, "config", "s80m_stereo_calibration.json")


# ── 标定加载 ────────────────────────────────────────────────────

def K_from_intrinsic(intr) -> np.ndarray:
    """[fx, fy, cx, cy] → 3×3 相机矩阵。"""
    fx, fy, cx, cy = (float(x) for x in intr)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def is_usable(calib: dict) -> bool:
    """标定是否可用于三角化：左右内参非零 + 外参 R/t 齐全。"""
    if not isinstance(calib, dict):
        return False
    try:
        for side in ("left_camera", "right_camera"):
            fx = calib[side]["intrinsic"][0]
            fy = calib[side]["intrinsic"][1]
            if not (fx > 0 and fy > 0):
                return False
        R = np.array(calib.get("rotation") or [], dtype=np.float64)
        t = np.array(calib.get("translation") or [], dtype=np.float64)
        if R.size != 9 or t.size != 3:
            return False
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            return False
        if np.linalg.norm(t) < 1e-4:          # 零基线无意义
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def load_stereo_calibration(session_path: str = None, calib_path: str = None) -> dict | None:
    """标定查找链：显式路径 → episode 级 → 设备级。

    1. calib_path 指定 → 直接加载
    2. <session>/calibration/head_stereo.json 且可用 → 用之
    3. <repo>/config/s80m_stereo_calibration.json 且可用 → 用之
    4. 都不可用 → None（调用方回退单目/报错）
    """
    candidates = []
    if calib_path:
        candidates.append(("--calib", calib_path))
    if session_path:
        candidates.append(("episode", os.path.join(session_path, "calibration", "head_stereo.json")))
    candidates.append(("device", DEVICE_CALIB_PATH))

    for label, path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                calib = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if is_usable(calib):
            return calib
        if label == "--calib":
            # 显式指定的标定必须可用，否则报错而不是静默回退
            raise ValueError(f"标定文件不可用（内参或 R/t 缺失/为零）: {path}")
    return None


# ── 结果数据结构 ────────────────────────────────────────────────

class TriangulationResult:
    """一次左右目点对三角化的结果。"""

    def __init__(self, points_3d: np.ndarray, reproj_error: np.ndarray):
        self.points_3d = points_3d          # (N,3) float64, 无效点 NaN, 左目相机系米制
        self.reproj_error = reproj_error    # (N,) float64, 无效点 inf
        self.valid = np.isfinite(reproj_error)          # (N,) bool
        self.valid_count = int(np.count_nonzero(self.valid))
        v = reproj_error[self.valid]
        self.mean_error = float(v.mean()) if v.size else float("inf")

    @property
    def z(self) -> np.ndarray:
        return self.points_3d[:, 2]


class HandPair:
    """左右目匹配出的同一只手。"""

    def __init__(self, l_idx: int, r_idx: int, left_label: str, result: TriangulationResult):
        self.l_idx = l_idx            # 左目 hands 列表下标
        self.r_idx = r_idx            # 右目 hands 列表下标
        self.left_label = left_label  # 左目 MediaPipe label ("Left"/"Right")
        self.result = result


# ── 三角化器 ────────────────────────────────────────────────────

class StereoTriangulator:
    """基于出厂标定的双目三角化器（fisheye / radtan 双路径）。"""

    def __init__(self, calib: dict, image_size: tuple = None, swap_cams: bool = False):
        """
        calib     : load_stereo_calibration() 返回的 dict
        image_size: (w, h)，缺省取 calib["resolution"]
        swap_cams : True 时把 "left_camera"/"right_camera" 与 R/t 互换处理——
                    用于实际文件名(left/right)与标定 cam0/cam1 错位的情况。
                    （S80M 有 stereo_swap_lr 配置，历史记忆与配置文件可能冲突，
                     hand_triangulate.py 会自动检测是否需要 swap）
        """
        self.calib = calib
        self.image_size = tuple(image_size) if image_size else tuple(calib["resolution"])
        w, h = self.image_size

        if swap_cams:
            lc, rc = calib["right_camera"], calib["left_camera"]
            R = np.array(calib["rotation"], dtype=np.float64).reshape(3, 3).T
            t = -R @ np.array(calib["translation"], dtype=np.float64)
        else:
            lc, rc = calib["left_camera"], calib["right_camera"]
            R = np.array(calib["rotation"], dtype=np.float64).reshape(3, 3)
            t = np.array(calib["translation"], dtype=np.float64).reshape(3, 1)

        # 内参按目标分辨率缩放（标定内参按 calib["resolution"] 给出）
        res = calib.get("resolution") or [w, h]
        s = w / float(res[0]) if res else 1.0
        self.K1, self.D1 = K_from_intrinsic([x * s for x in lc["intrinsic"]]), np.array(lc.get("distortion") or [], dtype=np.float64)
        self.K2, self.D2 = K_from_intrinsic([x * s for x in rc["intrinsic"]]), np.array(rc.get("distortion") or [], dtype=np.float64)
        self.model = str(lc.get("distortion_model", "")).lower()

        if self.D1.size > 4:      # SDK 8 系数只取 fisheye 前 4 个 (k1..k4)
            self.D1 = self.D1[:4]
        if self.D2.size > 4:
            self.D2 = self.D2[:4]

        # 双目矫正（equidistant → fisheye 路径）
        flags = cv2.CALIB_ZERO_DISPARITY
        if self.model == "equidistant":
            self.R1, self.R2, self.P1, self.P2, self.Q = \
                cv2.fisheye.stereoRectify(self.K1, self.D1, self.K2, self.D2,
                                          self.image_size, R, t, flags)
            self._init_undistort = cv2.fisheye.initUndistortRectifyMap
            self._undistort_pts = cv2.fisheye.undistortPoints
        else:
            self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = \
                cv2.stereoRectify(self.K1, self.D1, self.K2, self.D2,
                                  self.image_size, R, t, flags)
            self._init_undistort = cv2.initUndistortRectifyMap
            self._undistort_pts = cv2.undistortPoints

        # 矫正图像重映射表（可视化用）
        self._maps = {}
        for side, (K, D, Rr, P) in (("left", (self.K1, self.D1, self.R1, self.P1)),
                                    ("right", (self.K2, self.D2, self.R2, self.P2))):
            m1, m2 = self._init_undistort(K, D, Rr, P, self.image_size, cv2.CV_16SC2)
            self._maps[side] = (m1, m2)

        # 矫正焦距 / 基线（从矫正投影矩阵读，供校验与可视化）
        self.fx_rect = float(self.P1[0, 0])
        self.baseline_rect = abs(float(self.P1[0, 3]) / self.fx_rect) if abs(self.P1[0, 3]) > 0 else abs(float(self.P2[0, 3]) / self.fx_rect)

    # ── 像素坐标变换 ──────────────────────────────────────────

    def rectify_points(self, pts: np.ndarray, side: str) -> np.ndarray:
        """原始图像素 (N,2) → 矫正图像素 (N,2) float64。"""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if side == "left":
            return self._undistort_pts(pts, self.K1, self.D1, R=self.R1, P=self.P1).reshape(-1, 2)
        return self._undistort_pts(pts, self.K2, self.D2, R=self.R2, P=self.P2).reshape(-1, 2)

    def rectified_image(self, frame: np.ndarray, side: str) -> np.ndarray:
        """原始帧 → 矫正帧（可视化用）。"""
        m1, m2 = self._maps[side]
        return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)

    def project(self, xyz: np.ndarray, side: str) -> np.ndarray:
        """左目相机系 3D (N,3) → 矫正图像素 (N,2)。重投影验证/可视化用。"""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        n = xyz.shape[0]
        P = self.P1 if side == "left" else self.P2
        hom = P @ np.hstack([xyz, np.ones((n, 1))]).T      # 3×N
        with np.errstate(divide="ignore", invalid="ignore"):
            px = hom[:2] / hom[2]
        return px.T

    # ── 三角化 ────────────────────────────────────────────────

    def triangulate(self, pts_l: np.ndarray, pts_r: np.ndarray,
                    max_err: float = None, max_depth: float = None) -> TriangulationResult:
        """左右目原始(未矫正)图像素关键点 (N,2) → 左目相机系 3D (N,3) 米。

        逐点过滤：重投影误差 ≤ max_err 且 min_depth < z ≤ max_depth 且有限。
        无效点置 NaN（3D） / inf（误差），由调用方按列存放。
        """
        if max_err is None:
            max_err = DEFAULT_MAX_REPROJ_ERR
        if max_depth is None:
            max_depth = DEFAULT_MAX_DEPTH

        rl = self.rectify_points(pts_l, "left")
        rr = self.rectify_points(pts_r, "right")

        hom = cv2.triangulatePoints(self.P1, self.P2, rl.T, rr.T)   # 4×N 齐次
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = (hom[:3] / hom[3]).T                              # (N,3), 左目相机系米制

        # 逐点重投影误差 = 左右矫正图各自 |检测点 - 3D投影点| 的均值
        proj_l, proj_r = self.project(xyz, "left"), self.project(xyz, "right")
        err_l = np.linalg.norm(rl - proj_l, axis=1)
        err_r = np.linalg.norm(rr - proj_r, axis=1)
        err = 0.5 * (err_l + err_r)

        valid = (err <= max_err) & (xyz[:, 2] > DEFAULT_MIN_DEPTH) & (xyz[:, 2] <= max_depth)
        valid &= np.all(np.isfinite(xyz), axis=1)
        err[~valid] = np.inf
        xyz[~valid, :] = np.nan

        res = TriangulationResult(xyz, err)
        return res

    # ── 工具 ──────────────────────────────────────────────────

    def summarize(self) -> str:
        return (f"model={self.model} fx_rect={self.fx_rect:.1f} "
                f"baseline_rect={self.baseline_rect*1000:.1f}mm image={self.image_size}")


# ── 跨目手匹配 ──────────────────────────────────────────────────

def match_hands(left_hands, right_hands, tri: StereoTriangulator,
                max_err: float = None, max_depth: float = None,
                min_valid: int = MIN_VALID_POINTS) -> list:
    """左右目各自检测到的手列表 → 匹配出的同一只手（≤2×2 穷举 + 贪心去重）。

    评分（几何为主判据，label 只作加成，防左右手交换）::

        score = 有效点数 × 10 - 平均重投影误差 + (左右目 label 一致 + 5)

    返回按"物理左右"排序的 HandPair 列表：左目 label=="Left" 的手排前（hand_0），
    之后是 "Right"，其余按得分降序。左右目 label 冲突时以几何为准。
    """
    if max_err is None:
        max_err = DEFAULT_MAX_REPROJ_ERR
    if max_depth is None:
        max_depth = DEFAULT_MAX_DEPTH

    candidates = []
    for li, lh in enumerate(left_hands):
        for ri, rh in enumerate(right_hands):
            res = tri.triangulate(lh.landmarks, rh.landmarks, max_err, max_depth)
            if res.valid_count < min_valid:
                continue
            score = res.valid_count * 10.0 - res.mean_error
            if lh.label == rh.label:
                score += 5.0
            candidates.append((score, li, ri, lh.label, res))

    candidates.sort(key=lambda c: -c[0])
    used_l, used_r, out = set(), set(), []
    for score, li, ri, label, res in candidates:
        if li in used_l or ri in used_r:
            continue
        out.append(HandPair(li, ri, label, res))
        used_l.add(li)
        used_r.add(ri)
        if len(out) >= 2:
            break

    # 物理左右排序：左目 label "Left" 在前
    out.sort(key=lambda p: 0 if p.left_label == "Left" else 1)
    return out


# ── 自检 ────────────────────────────────────────────────────────

def _self_test() -> int:
    """用真实标定做 3D→双图投影→三角化 往返测试。"""
    calib = load_stereo_calibration()
    if calib is None:
        print("[FAIL] 找不到可用标定")
        return 1
    tri = StereoTriangulator(calib)
    print(f"三角化器: {tri.summarize()}")

    rng = np.random.default_rng(7)
    # 左目相机系随机 3D 点（手部典型范围）
    X = np.column_stack([rng.uniform(-0.25, 0.25, 10),
                         rng.uniform(-0.2, 0.2, 10),
                         rng.uniform(0.4, 1.5, 10)])

    # 投影到左右矫正图 → 加 0.5px 噪声（MediaPipe 关键点典型精度）
    # → 逆变换回原始(畸变)像素 → 三角化往返
    # 逆变换: 矫正像素 → 归一化矫正坐标(inv K_rect) → R^T 回到原始理想坐标 → 加畸变
    pl = tri.project(X, "left")
    pr = tri.project(X, "right")
    pl += rng.normal(0, 0.5, pl.shape)
    pr += rng.normal(0, 0.5, pr.shape)
    n = X.shape[0]
    _distort = cv2.fisheye.distortPoints if tri.model == "equidistant" else cv2.distortPoints

    def _rect_to_raw(px, Rr, K_rect, K, D):
        hom = np.hstack([px, np.ones((n, 1))]).T        # 3×N 像素齐次
        ideal = np.linalg.solve(K_rect, hom)            # 归一化矫正坐标
        orig_ideal = (Rr.T @ ideal)[:2].T               # 原始相机理想坐标 (N,2)
        return _distort(orig_ideal.reshape(-1, 1, 2), K, D).reshape(-1, 2)

    raw_l = _rect_to_raw(pl, tri.R1, tri.P1[:3, :3], tri.K1, tri.D1)
    raw_r = _rect_to_raw(pr, tri.R2, tri.P2[:3, :3], tri.K2, tri.D2)

    res = tri.triangulate(raw_l, raw_r)
    ok = res.valid_count == len(X)
    err3d = np.linalg.norm(res.points_3d - X, axis=1)
    e3 = float(err3d[res.valid].max())
    epx = float(res.reproj_error[res.valid].max())

    print(f"有效点: {res.valid_count}/{len(X)}   3D 最大误差: {e3*1000:.2f} mm   "
          f"重投影最大误差: {epx:.2f} px")
    print(f"深度范围: z ∈ [{res.z[res.valid].min():.3f}, {res.z[res.valid].max():.3f}] m")
    # 物理上限: 基线 8cm 的双目, 视差 1px 噪声在 1.5m 处 ≈ 80mm 深度误差
    # (dz ≈ z²·δd/(fx·b))。0.5px 噪声 → 允许 3D 误差 < 80mm 即自洽。
    if ok and e3 < 0.08 and epx < 2.0:
        print("[OK] 三角化往返自检通过")
        return 0
    print("[FAIL] 三角化往返自检未通过")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
