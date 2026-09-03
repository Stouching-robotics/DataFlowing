"""
双目深度计算模块 —— 基于 OpenCV StereoSGBM 从左右目图像生成视差/深度图。

输出 uint16 PNG，像素值 = 视差 × 16（OpenCV 标准亚像素格式），
单位可通过标定参数 (baseline × fx / disparity) 换算为毫米。

用法:
    from core.stereo_depth import StereoDepthComputer

    computer = StereoDepthComputer()
    disparity = computer.compute(left_frame, right_frame)  # uint16 (H, W)
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from config import settings


def depth_to_heatmap(depth_uint16: np.ndarray,
                     near_mm: float = 0.0,
                     far_mm: float = 0.0,
                     colormap: int = cv2.COLORMAP_JET,
                     smooth_k: int = 0) -> np.ndarray:
    """uint16 深度图（毫米）→ BGR 热力图。

    near_mm/far_mm 均 >0 时用**真固定色标**（与 hand_3d_d435 demo
    同口径）：整幅按 (depth-near)/(far-near)*255 做 0..255 裁剪——
    无效值 =0 与近于 near 的像素落入 JET(0) 深蓝，远于 far 的饱和为
    JET(255) 红，不置黑；帧间颜色完全一致，不随画面内容漂移。
    否则按帧内有效值 min/max 自适应归一化（S80M 传统行为），无效值
    同样显示为 JET(0) 深蓝而非黑。仅一侧给定时按窗口过滤 + 自适应。

    smooth_k>0 时先做 k×k 中值滤波（单像素椒盐噪点，仅可视化用）。

    Args:
        depth_uint16: uint16 深度图 (H, W)，单位毫米
        near_mm: 显示范围下限（毫米）；与 far_mm 同给 → 固定色标
        far_mm:  显示范围上限（毫米）
        colormap: OpenCV colormap 常量
        smooth_k: 中值滤波核（奇数，0/1 关闭）

    Returns:
        BGR 热力图 (H, W, 3) uint8
    """
    if smooth_k and smooth_k > 1:
        depth_uint16 = cv2.medianBlur(depth_uint16, smooth_k)

    fixed = near_mm > 0 and far_mm > 0
    if fixed:
        # 整幅 clip（demo 口径）：无效值(0) 与近于 near 的像素经负数
        # 裁剪落到 JET(0) 深蓝，远于 far 的饱和为 JET(255) 红——不置黑
        normed = np.clip((depth_uint16.astype(np.float32) - near_mm)
                         / (far_mm - near_mm) * 255.0,
                         0.0, 255.0).astype(np.uint8)
        return cv2.applyColorMap(normed, colormap)

    valid = depth_uint16 > 0
    if near_mm > 0 or far_mm > 0:
        lo = near_mm if near_mm > 0 else 0.0
        hi = far_mm if far_mm > 0 else float("inf")
        valid = valid & (depth_uint16 >= lo) & (depth_uint16 <= hi)

    normed = np.zeros_like(depth_uint16, dtype=np.uint8)
    if valid.any():
        vmin = depth_uint16[valid].min()
        vmax = depth_uint16[valid].max()
        if vmax > vmin:
            # 线性映射 vmin..vmax → 1..255（0 保留给无效值）
            normed[valid] = ((depth_uint16[valid].astype(np.float32) - vmin)
                             / (vmax - vmin) * 254 + 1).astype(np.uint8)
        else:
            normed[valid] = 128
    # 无效值保持 0 → JET(0) 深蓝（demo 口径，不再置黑）
    return cv2.applyColorMap(normed, colormap)


class DepthHeatmapSmoother:
    """深度热力图时域 EMA 平滑（仅可视化通道；无损深度流不受影响）。

    对 BGR 热力图做指数滑动平均，抑制帧间抖动噪点；无效区（JET(0)
    深蓝）随 EMA 自然淡入淡出，不会留下永久残影。
    """

    def __init__(self, alpha: float = 0.5):
        self._alpha = float(alpha)   # 上一帧权重（越大越平滑、拖影越重）
        self._prev: np.ndarray = None

    def update(self, bgr: np.ndarray) -> np.ndarray:
        if self._prev is None or self._prev.shape != bgr.shape:
            self._prev = bgr.copy()
            return bgr
        out = cv2.addWeighted(self._prev, self._alpha, bgr,
                              1.0 - self._alpha, 0)
        self._prev = out
        return out

    def reset(self):
        self._prev = None


class StereoDepthComputer:
    """基于 StereoSGBM 的双目深度计算器。

    为 S80M 双目摄像机 (800×640) 调优的默认参数。
    输出 uint16 视差图：像素值 = 视差 × 16（DISPVIS_MULT=16 时实际值需 /16）。
    """

    def __init__(self,
                 resolution: tuple = (800, 640),
                 num_disparities: int = 128,
                 block_size: int = 11):
        """
        Args:
            resolution: (width, height) of each camera frame
            num_disparities: 视差搜索范围，必须是 16 的倍数
            block_size: SAD 窗口大小，奇数，越大越平滑但丢失细节
        """
        self._width, self._height = resolution

        # StereoSGBM 参数
        self._num_disparities = max(16, num_disparities // 16 * 16)  # 强制 16 倍数
        self._block_size = max(5, block_size // 2 * 2 + 1)           # 强制奇数

        self._stereo: Optional[cv2.StereoSGBM] = None
        self._right_matcher: Optional[cv2.ximgproc.DisparityWLSFilter] = None
        self._wls_filter: Optional[cv2.ximgproc.DisparityWLSFilter] = None

        self._init_matchers()

    def _init_matchers(self):
        """初始化 StereoSGBM 匹配器 + WLS 滤波器（WLS 需要 opencv-contrib）。"""
        # 左目 → 右目 匹配
        self._left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self._num_disparities,
            blockSize=self._block_size,
            P1=8 * 3 * self._block_size ** 2,
            P2=32 * 3 * self._block_size ** 2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # 右目 → 左目 匹配（用于 WLS 左右一致性检查）
        self._right_matcher = cv2.ximgproc.createRightMatcher(self._left_matcher)

        # WLS 滤波器 —— 平滑视差图同时保留边缘
        self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(self._left_matcher)
        self._wls_filter.setLambda(8000.0)
        self._wls_filter.setSigmaColor(1.5)

    # ── 公开接口 ──────────────────────────────────────────

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """从左右目 BGR 帧计算视差图。

        Args:
            left:  左目 BGR 帧 (H, W, 3)
            right: 右目 BGR 帧 (H, W, 3)

        Returns:
            uint16 视差图 (H, W)，像素值 = 视差 × 16
            (eg. value=256 → disparity=16 pixels)
        """
        # 转灰度
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # 直方图均衡化 —— 改善纹理较弱区域的匹配
        gray_l = cv2.equalizeHist(gray_l)
        gray_r = cv2.equalizeHist(gray_r)

        # 计算左右视差
        disp_l = self._left_matcher.compute(gray_l, gray_r)   # int16
        disp_r = self._right_matcher.compute(gray_r, gray_l)  # int16

        # WLS 滤波（亚像素精度 float32）
        disp_filtered = self._wls_filter.filter(
            disp_l, gray_l, disparity_map_right=disp_r
        )  # float32

        # 转为 uint16，×16 缩放（OpenCV 标准格式）
        disp_filtered = np.maximum(disp_filtered, 0.0)
        disp_uint16 = (disp_filtered * 16.0).astype(np.uint16)

        return disp_uint16

    @property
    def num_disparities(self) -> int:
        return self._num_disparities

    @property
    def block_size(self) -> int:
        return self._block_size
