#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 数据四宫格演示播放器（单文件自包含，可直接发给客户）

用法:
    python hdf5_demo.py [data.h5]
    不带参数则启动后点"打开文件"选择

界面四宫格 + 底部 IMU:
    左上  相机原始视频: 双目并排 (stereo_left | stereo_right),
          单目单图, 或双目+单目同时显示(单目拼在下排)
    右上  渲染好的视频 (videos/hand_skeleton MP4 字节 或 images 内
          preview/keypoint/render/... 帧数组, 自动探测)
    左下  仿生手掌 (16x16 触觉矩阵按部位映射到手掌图形)
    右下  触觉传感器热力图 (16x16 压力矩阵, 左 | 右)
    底部  IMU 加速度/角速度波形 (observation/imu + imu_frame_index)

依赖 (见 requirements.txt):
    numpy, h5py, PyQt5, opencv-python

hdf5 数据格式 (本 demo 读取的数据集):
    <episode>/observation/images/stereo_left   (N, 800, 1280, 3) uint8 (RGB)
    <episode>/observation/images/stereo_right  (N, 800, 1280, 3) uint8 (RGB)
    <episode>/observation/images/<单目数据集>    (N, H, W, 3) uint8 (RGB)
          单目名: left/mono/monocular/rgb/color/cam0/camera/image/frame/
          left_camera/camera_left/head_left_rgb/head_rgb/front_rgb/front
          (与双目独立探测, 可同时存在)
    <episode>/videos/hand_skeleton             (nbytes,) uint8 · MP4 字节 (自动探测)
    <episode>/observation/images/<渲染视频>      (N, H, W, 3) uint8 (自动探测)
    <episode>/observation/tactile/left  (N, 16, 16) float 触觉矩阵
    <episode>/observation/tactile/right (N, 16, 16) float
    <episode>/observation/imu           (M, 6) float      变长聚合 (可选)
    <episode>/observation/imu_frame_index (M,) int       IMU 对应帧号 (可选)
"""

import sys
import os
import time
import argparse

import numpy as np
import h5py
import cv2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFileDialog, QSizePolicy,
)


# ══════════════════════════════════════════════════════════════════
# 1. 仿生手掌 —— 手掌图形锚点 (基于 1280x720 设计, 运行时等比缩放)
#    16x16 触觉矩阵按部位映射到手指关节/掌心
# ══════════════════════════════════════════════════════════════════
HAND_ANCHORS = {
    "thumb_tip": (350, 280), "thumb_joint": (420, 360), "thumb_base": (480, 420),
    "index_tip": (510, 120), "index_joint": (540, 240), "index_base": (560, 340),
    "middle_tip": (640, 80), "middle_joint": (640, 200), "middle_base": (640, 320),
    "ring_tip": (770, 120), "ring_joint": (740, 240), "ring_base": (720, 340),
    "pinky_tip": (930, 280), "pinky_joint": (860, 360), "pinky_base": (800, 420),
    "palm": (640, 480),
}
WRIST_ANCHOR = (640, 600)

# 16×16 传感器矩阵 → 手掌部位映射（行列号）
HAND_PART_CONFIG = {
    "thumb_joint":  {"rows": [0, 1, 2],  "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "index_joint":  {"rows": [3, 4, 5],  "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "middle_joint": {"rows": [6, 7, 8],  "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "ring_joint":   {"rows": [9, 10, 11], "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "pinky_joint":  {"rows": [12, 13, 14], "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "palm":         {"rows": list(range(15)), "cols": [10, 9, 8, 6, 4], "axis_order": "col_row"},
}

# Viridis 色表 (cv2 内置, 一次生成缓存)
_VIRIDIS_LUT = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_VIRIDIS)[:, 0]


def render_bionic_hand(mat, window_size=(640, 400), vmax=None):
    """仿生手掌: 2D 手掌骨架 + 触觉矩阵按部位映射 viridis 色块。

    Args:
        mat: 16×16 触觉矩阵
        window_size: 输出画布 (w, h)
        vmax: 色块归一化基准 (None = 用本帧最大值)
    """
    ww, wh = window_size
    sx, sy = ww / 1280.0, wh / 720.0
    frame = np.full((wh, ww, 3), 15, dtype=np.uint8)

    m = np.asarray(mat, dtype=np.float32)
    if m.ndim != 2:
        m = np.zeros((16, 16), np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    if vmax is None or vmax <= 0:
        vmax = float(m.max()) if m.max() > 0 else 1.0

    def _sp(x, y):
        return (int(x * sx), int(y * sy))

    wrist = _sp(*WRIST_ANCHOR)
    anchors = {k: _sp(*v) for k, v in HAND_ANCHORS.items()}
    _lw = lambda v: max(1, int(v * min(sx, sy)))

    # ── 手掌骨骼框架 ──
    line_c, joint_c = (60, 60, 60), (80, 80, 80)
    cv2.line(frame, wrist, anchors["palm"], line_c, _lw(6), cv2.LINE_AA)
    for fg in ("thumb", "index", "middle", "ring", "pinky"):
        b, j, t = anchors[f"{fg}_base"], anchors[f"{fg}_joint"], anchors[f"{fg}_tip"]
        cv2.line(frame, anchors["palm"], b, line_c, _lw(5), cv2.LINE_AA)
        cv2.line(frame, b, j, line_c, _lw(4), cv2.LINE_AA)
        cv2.line(frame, j, t, line_c, _lw(3), cv2.LINE_AA)
        for p, r in ((b, 10), (j, 8), (t, 6)):
            cv2.circle(frame, p, _lw(r), joint_c, -1, cv2.LINE_AA)
    cv2.circle(frame, anchors["palm"], _lw(16), joint_c, -1, cv2.LINE_AA)

    # ── 触觉色块映射 ──
    CELL = max(1, int(14 * min(sx, sy)))
    for part, cfg in HAND_PART_CONFIG.items():
        ax, ay = anchors[part]
        rows, cols = cfg["rows"], cfg["cols"]
        order = cfg.get("axis_order", "row_col")
        nrr, ncc = len(rows), len(cols)
        w = ncc * CELL if order == "row_col" else nrr * CELL
        h = nrr * CELL if order == "row_col" else ncc * CELL
        cx, cy = ax - w // 2, ay - h // 2
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if r >= 16 or c >= 16:
                    continue
                val = m[r, c]
                x1 = cx + (j if order == "row_col" else i) * CELL
                y1 = cy + (i if order == "row_col" else j) * CELL
                if val > 0:
                    idx = int(min(255, (val / vmax) * 255))
                    b, g, rr = _VIRIDIS_LUT[idx]
                    cv2.rectangle(frame, (x1, y1), (x1 + CELL, y1 + CELL),
                                  (int(b), int(g), int(rr)), -1)
                cv2.rectangle(frame, (x1, y1), (x1 + CELL, y1 + CELL),
                              (100, 100, 100), 1)
    return frame


# ══════════════════════════════════════════════════════════════════
# 2. 工具
# ══════════════════════════════════════════════════════════════════
def make_heatmap(mat, size=300, cmap=cv2.COLORMAP_JET, vmax=None):
    """16x16 压力矩阵 → 热力图 BGR 图像 (size x size)。"""
    m = np.asarray(mat, dtype=np.float32)
    if m.ndim != 2:
        m = np.zeros((16, 16), np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    if vmax is None or vmax <= 0:
        vmax = float(m.max()) if m.max() > 0 else 1.0
    norm = np.clip(m / vmax, 0.0, 1.0)
    big = cv2.resize(norm, (size, size), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap((big * 255).astype(np.uint8), cmap)
    n = m.shape[0]
    step = size // n
    for i in range(1, n):
        cv2.line(heat, (i * step, 0), (i * step, size), (60, 60, 60), 1)
        cv2.line(heat, (0, i * step), (size, i * step), (60, 60, 60), 1)
    return heat


def put_label(img, text, pos, color=(255, 255, 255)):
    """叠加标签文字（带黑底）。无效输入(object/空/非3通道)直接忽略。"""
    if (not isinstance(img, np.ndarray) or img.dtype == np.object_
            or img.size == 0 or img.ndim != 3 or img.shape[2] != 3):
        return
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = pos
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════
# 2.5 IMU 可视化 (observation/imu 变长聚合: 平铺样本 + imu_frame_index)
# ══════════════════════════════════════════════════════════════════
_IMU_COLORS = [(255, 90, 90), (90, 255, 90), (90, 170, 255),
               (255, 210, 90), (255, 110, 255), (90, 255, 255)]
_IMU_LABELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")


def render_imu(seq, counts, idx, window=240, w=880, h=185):
    """IMU 6 通道折线(acc 3 轴 / gyr 3 轴)+ 当前帧竖线 + 数值文本。

    Args:
        seq: (T, 6) 每帧首个 IMU 样本
        counts: (T,) 每帧 IMU 样本数
        idx: 当前视频帧号
    Returns:
        BGR 画布(顶部标题区, 右侧数值区, 左侧波形区)
    """
    canvas = np.full((h, w, 3), 18, np.uint8)
    n = len(seq)
    start = max(0, idx - window + 1)
    x0, x1 = 14, w - 128   # 波形区 | 右侧文本区
    y0, mid, y1 = 30, h // 2 + 2, h - 8

    cv2.putText(canvas, "IMU (Accel / Gyro)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # 分上下两图: 上 = acc(0..2), 下 = gyr(3..5)
    for part, (ya, yb) in enumerate(((y0, mid - 4), (mid + 4, y1))):
        cols = list(range(3 * part, 3 * part + 3))
        vals = seq[start:idx + 1, cols]
        if vals.size:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        else:
            lo, hi = -1.0, 1.0
        if hi - lo < 1e-6:
            hi = lo + 1.0
        cv2.putText(canvas, "acc" if part == 0 else "gyr",
                    (x0 + 2, (ya + yb) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)
        t_start = x0 + 34
        for c in cols:
            pts = []
            for t in range(start, idx + 1):
                v = seq[t, c]
                if v != v:  # NaN(该帧无样本)
                    continue
                px = t_start + (t - start) / max(1, window - 1) * (x1 - t_start - 10)
                py = ya + (yb - ya) * (1 - (v - lo) / (hi - lo))
                pts.append((int(px), int(py)))
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts, np.int32)], False,
                              _IMU_COLORS[c], 1, cv2.LINE_AA)
        # 当前帧竖线
        if start <= idx:
            px = t_start + (idx - start) / max(1, window - 1) * (x1 - t_start - 10)
            cv2.line(canvas, (int(px), ya), (int(px), yb), (255, 255, 255), 1)

    # 右侧: 当前帧数值
    tx = x1 + 8
    cv2.putText(canvas, f"frame {idx}", (tx, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    ns = int(counts[idx]) if idx < len(counts) else 0
    cv2.putText(canvas, f"samples {ns}", (tx, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    for c in range(6):
        v = seq[idx, c] if idx < len(seq) else float("nan")
        cv2.putText(canvas, f"{_IMU_LABELS[c]} {v:+.2f}", (tx, y0 + 54 + c * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _IMU_COLORS[c], 1, cv2.LINE_AA)
    return canvas


# ══════════════════════════════════════════════════════════════════
# 3. HDF5 数据加载
# ══════════════════════════════════════════════════════════════════
# 渲染视频数据集名匹配关键字（自动探测 h5 中已渲染好的视频）
RENDERED_KEYWORDS = ("preview", "keypoint", "annot", "render", "overlay",
                     "visual", "vis", "kp", "skeleton", "slam")


class RenderedVideo:
    """h5 内 <episode>/videos/hand_skeleton 的 MP4 原始字节 → 按帧解码。

    手部骨骼渲染视频以 MP4 字节存于 videos/ 组(attrs 标 format=mp4),
    用临时文件 + cv2.VideoCapture 解码,支持 __len__/__getitem__,
    与 images 组帧数组用法一致。

    性能: 逐帧 seek(CAP_PROP_POS_FRAMES)对 h264 要回关键帧重解, 2560×800
    大帧实测 ~70ms/帧 → 播放会掉帧成慢动作。因此连续访问(播放)走顺序
    read(~5ms/帧), 只有跳转(拖进度条/单帧步进)才 seek, 且跳转帧才缓存
    (缓存上限 _CACHE_MAX, 防 2560×800 帧撑爆内存)。
    """
    _CACHE_MAX = 30

    def __init__(self, ds):
        import tempfile
        data = ds[...].tobytes()
        self._tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self._tmp.write(data)
        self._tmp.close()
        self.cap = cv2.VideoCapture(self._tmp.name)
        # 打不开/损坏的 mp4 会返回负帧数, 规整为 0 (len() 必须 >= 0)
        self.n = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._cache = {}
        self._seq_next = 0   # 顺序读指针(播放加速用)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        i = int(i)
        if i in self._cache:
            # 缓存命中必须同时同步解码器指针: 只更新 _seq_next 会让 cap
            # 停在旧位置, 回卷后从 EOF 顺序读返回 None(右上角窗口失效),
            # 播放中拖到已缓存帧则会读到错帧。set+read 把 cap 定位到 i+1。
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.cap.read()
            self._seq_next = i + 1
            return self._cache[i]
        if i == self._seq_next:
            ok, frame = self.cap.read()          # 顺序播放: 直接读, 快
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 跳转: seek
            ok, frame = self.cap.read()
            if ok:
                self._cache[i] = frame           # 只缓存跳转帧
                if len(self._cache) > self._CACHE_MAX:
                    self._cache.pop(next(iter(self._cache)))
        self._seq_next = i + 1
        return frame if ok else None

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass
        self._cache.clear()


class H5Data:
    """读取 hdf5 中的数据（惰性访问，按帧索引取）。"""

    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, "r")
        ep_groups = [k for k in self.f.keys() if str(k).startswith("episode")]
        if not ep_groups:
            raise ValueError(f"no episode_* group in hdf5: {list(self.f.keys())}")
        self.ep = str(ep_groups[0])
        obs = self.f[f"{self.ep}/observation"]
        if "images" not in obs:
            raise ValueError("missing observation/images")

        # 视频: 双目与单目独立探测、不互斥 —— 同一个 h5 里混有
        # 双目(S80M)和单目(头戴相机)时, 两组都读进来同时显示;
        # 都找不到则报错并列出 images 组内容便于排查
        self.imgs = {}
        for side in ("stereo_left", "stereo_right"):
            key = f"{self.ep}/observation/images/{side}"
            if key in self.f:
                self.imgs[side] = self.f[key]
        # 单目: 常见数据集名逐个尝试 (与双目同时存在也读)
        for name in ("left", "mono", "monocular", "rgb", "color", "cam0",
                     "camera", "image", "frame", "left_camera", "camera_left",
                     "head_left_rgb", "head_rgb", "front_rgb", "front"):
            if name in self.imgs:
                continue
            key = f"{self.ep}/observation/images/{name}"
            if key in self.f:
                self.imgs["mono"] = self.f[key]
                break
        if not self.imgs:
            grp = self.f[f"{self.ep}/observation/images"]
            raise ValueError(
                "no stereo_left/stereo_right or mono video dataset; "
                f"images group contains: {list(grp.keys())}")
        self.n = len(next(iter(self.imgs.values())))

        # 渲染好的视频（自动探测）
        self.rendered = self._find_rendered()

        # 触觉矩阵
        self.tactile = {}
        for side in ("left", "right"):
            key = f"{self.ep}/observation/tactile/{side}"
            self.tactile[side] = self.f[key] if key in self.f else None

        # 播放帧率: 优先 meta/info attrs["capture_fps"](可选采集速度标记),
        # 无标记 → 按文件 fps 原样播放。
        # 说明: 视频文件本身按录制节奏编码(实测 25fps,墙钟时间戳验证
        # 播放时长≈真实录制时长), 因此文件 fps 即真实速度, 直接按它播
        # 即为实时。capture_fps 仅为将来"采集节拍与文件 fps 不一致"的
        # 批次预留的可选标记, 本批数据无此问题。
        self.fps = 25.0
        self.capture_fps = None
        info = self.f.get("meta/info")
        if info is not None:
            cf = info.attrs.get("capture_fps")
            if cf:
                try:
                    self.capture_fps = float(cf)
                except (TypeError, ValueError):
                    pass
            f_ = info.attrs.get("fps")
            if f_:
                try:
                    self.fps = float(f_)
                except (TypeError, ValueError):
                    pass
        if self.capture_fps is None:
            f_ = next(iter(self.imgs.values())).attrs.get("fps")
            if f_:
                try:
                    self.fps = float(f_)
                except (TypeError, ValueError):
                    pass

        # 预扫触觉全局最大值（热力图/手掌归一化基准）
        self.tactile_vmax = 0.0
        for ds in self.tactile.values():
            if ds is not None and len(ds):
                try:
                    self.tactile_vmax = max(self.tactile_vmax,
                                            float(np.nanmax(ds[:])))
                except Exception:
                    pass

        # IMU（变长聚合：平铺样本 + imu_frame_index 帧号列）
        self.imu_seq = None
        self.imu_counts = None
        imu_key = f"{self.ep}/observation/imu"
        if imu_key in self.f:
            imu = np.asarray(self.f[imu_key][:], dtype=np.float32)
            fidx_key = f"{self.ep}/observation/imu_frame_index"
            if fidx_key in self.f:
                fidx = np.asarray(self.f[fidx_key][:], dtype=np.int64)
                # 按帧号排序后, 取每帧第一个样本作图 + 统计每帧样本数
                order = np.argsort(fidx)
                sfidx = fidx[order]
                simu = imu[order]
                first = np.searchsorted(sfidx, np.arange(self.n), side="left")
                counts = np.bincount(sfidx, minlength=self.n)
                seq = np.full((self.n, 6), np.nan, dtype=np.float32)
                valid = first < len(simu)
                seq[valid] = simu[first[valid]]
                self.imu_seq, self.imu_counts = seq, counts

    def _find_rendered(self):
        """渲染视频自动探测：优先 videos/hand_skeleton（MP4 字节），
        再回退 images 组内名称含渲染关键字的帧数组。"""
        vkey = f"{self.ep}/videos/hand_skeleton"
        if vkey in self.f:
            try:
                return RenderedVideo(self.f[vkey])
            except Exception:
                pass
        grp = self.f[f"{self.ep}/observation/images"]
        for name in grp:
            if name in ("stereo_left", "stereo_right"):
                continue
            low = name.lower()
            if any(k in low for k in RENDERED_KEYWORDS):
                return grp[name]
        return None

    def close(self):
        if self.rendered is not None and hasattr(self.rendered, "close"):
            try:
                self.rendered.close()
            except Exception:
                pass
        try:
            self.f.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# 4. 主窗口
# ══════════════════════════════════════════════════════════════════
class DemoWindow(QMainWindow):
    def __init__(self, h5_path=None):
        super().__init__()
        self.setWindowTitle("HDF5 Data Viewer")
        self.resize(1440, 900)
        self.data = None
        self.idx = 0
        self.playing = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.lbl_top_left = QLabel("Top-left: Stereo Cameras (L | R)")
        self.lbl_top_right = QLabel("Top-right: Rendered Video")
        self.lbl_hand = QLabel("Bottom-left: Bionic Hand (L | R)")
        self.lbl_heat = QLabel("Bottom-right: Tactile Heatmap (L | R)")
        self.lbl_imu = QLabel("Bottom: IMU (Accel / Gyro)")
        for lb in (self.lbl_top_left, self.lbl_top_right,
                   self.lbl_hand, self.lbl_heat, self.lbl_imu):
            lb.setAlignment(Qt.AlignCenter)
            lb.setMinimumSize(1, 1)
            lb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lb.setStyleSheet("background:#111; color:#888; font-size:14px;")

        grid = QGridLayout()
        grid.addWidget(self.lbl_top_left, 0, 0)
        grid.addWidget(self.lbl_top_right, 0, 1)
        grid.addWidget(self.lbl_hand, 1, 0)
        grid.addWidget(self.lbl_heat, 1, 1)
        grid.addWidget(self.lbl_imu, 2, 0, 1, 2)   # IMU 横跨底部
        # 行权重: 视频行(0)与手掌行(1)平分剩余空间; IMU 行(2)固定高度。
        # 检测到无传感器时对应行 stretch 归零(load 时), 不会在中间留空。
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 0)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)
        self.grid = grid
        self.lbl_imu.setFixedHeight(240)

        # 控制条
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Open h5")
        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.lbl_frame = QLabel("- / -")
        self.btn_prev = QPushButton("|<")   # ASCII only, safe on Windows fonts
        self.btn_next = QPushButton(">|")
        for b in (self.btn_prev, self.btn_next):
            b.setEnabled(False)
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_play)
        bar.addWidget(self.btn_prev)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.btn_next)
        bar.addWidget(self.lbl_frame)
        root.addLayout(bar)

        self.btn_open.clicked.connect(self.open_file_dialog)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_prev.clicked.connect(lambda: self.seek(self.idx - 1))
        self.btn_next.clicked.connect(lambda: self.seek(self.idx + 1))
        self.slider.sliderMoved.connect(self.seek)

        self.timer = QTimer(self)
        self.timer.setInterval(40)  # 25fps
        self.timer.timeout.connect(self.next_frame)

        if h5_path:
            self.load(h5_path)

    # ── 文件 ──
    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HDF5 data file", "",
            "HDF5 (*.h5 *.hdf5);;All files (*)")
        if path:
            self.load(path)

    def load(self, path):
        try:
            if self.data:
                self.data.close()
            self.data = H5Data(path)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to open: {e}", 8000)
            return
        self.n = self.data.n
        self.idx = 0
        # Sensor detection: tactile panes are shown only when the tactile
        # datasets exist AND contain non-zero samples (tactile_vmax > 0;
        # all-zero data means the sensor was never touched during recording);
        # the IMU pane is shown only when imu data exists
        self.has_tactile = self.data.tactile_vmax > 0
        self.has_imu = self.data.imu_seq is not None
        self.lbl_hand.setVisible(self.has_tactile)
        self.lbl_heat.setVisible(self.has_tactile)
        self.lbl_imu.setVisible(self.has_imu)
        # 无传感器时该行不参与空间分配(否则隐藏行仍按 stretch 分到高度,
        # 导致视频与 IMU 之间留一块空)
        self.grid.setRowStretch(1, 1 if self.has_tactile else 0)
        # 播放时钟按帧率设置: capture_fps 优先(恢复采集实时速度),
        # 否则按文件 fps(原速还原)
        self.play_fps = self.data.capture_fps or self.data.fps
        self.timer.setInterval(max(1, int(round(1000 / self.play_fps))))
        self.slider.setRange(0, max(0, self.n - 1))
        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.setWindowTitle(
            f"HDF5 Data Viewer - {os.path.basename(path)} ({self.n} frames @ {self.play_fps:g}fps)")
        self.render_frame(0)
        rn = "yes" if self.data.rendered is not None else "no"
        speed_note = (f"capture speed {self.play_fps:g}fps" if self.data.capture_fps
                      else f"file speed {self.play_fps:g}fps")
        self.statusBar().showMessage(
            f"Loaded {os.path.basename(path)}: {self.n} frames | {speed_note} | "
            f"rendered video: {rn}", 6000)

    # ── 播放 ──
    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def next_frame(self):
        if self.data is None:
            return
        self.idx = 0 if self.idx + 1 >= self.n else self.idx + 1
        t0 = time.perf_counter()
        self.render_frame(self.idx)
        # 自调节: 渲染慢于播放时钟时自动降到渲染速率(稳定不掉帧),
        # 渲染快时保持时钟速度(30fps → 33ms)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.timer.setInterval(
            max(int(round(1000 / self.play_fps)), int(elapsed_ms)))

    def seek(self, i):
        if self.data is None:
            return
        self.idx = max(0, min(self.n - 1, int(i)))
        self.render_frame(self.idx)

    # ── 渲染 ──
    def render_frame(self, idx):
        self.slider.setValue(idx)
        self.lbl_frame.setText(f"{idx + 1} / {self.n}")

        # 左上: 双目并排 + 单目; 无立体对时单目单图
        # 注意: h5 内视频帧以 RGB 存储(导出时 BGR→RGB 写入),
        # 这里转回 BGR 给 _show_image(cv2 惯例是 BGR)。
        imgs = self.data.imgs
        l_key = "stereo_left" if "stereo_left" in imgs else "mono"
        l = cv2.cvtColor(np.asarray(imgs[l_key][idx]), cv2.COLOR_RGB2BGR)
        if "stereo_right" in imgs:
            r = cv2.cvtColor(np.asarray(imgs["stereo_right"][idx]),
                             cv2.COLOR_RGB2BGR)
            top = np.hstack([l, r])
            cam_label = "Stereo Cameras (L | R)"
        elif "stereo_left" in imgs:
            top = l
            cam_label = "Camera"
        else:
            top = l
            cam_label = "Mono Camera"
        put_label(top, cam_label, (12, 26), (0, 255, 255))
        # 单目与双目同时存在: 单目等宽缩放后拼在下方, 两组都可见
        mono_ds = imgs.get("mono")
        if mono_ds is not None and "stereo_left" in imgs:
            m = cv2.cvtColor(np.asarray(mono_ds[idx]), cv2.COLOR_RGB2BGR)
            h, w = top.shape[:2]
            mh = max(1, int(round(m.shape[0] * w / m.shape[1])))
            m = cv2.resize(m, (w, mh), interpolation=cv2.INTER_AREA)
            put_label(m, "Mono Camera", (12, 26), (255, 255, 0))
            top = np.vstack([top, m])
        self._show_image(self.lbl_top_left, top)

        # 右上: 渲染好的视频（h5 内已有, 不自行渲染）
        if self.data.rendered is not None:
            n = len(self.data.rendered)
            if idx < n:
                rv = self.data.rendered[idx]
                # 解码失败(RenderedVideo read 返回 None)或数据集为 object
                # dtype 时 rv 会是 object 数组, cv2 无法处理 → 提示而非崩溃
                ok_frame = (isinstance(rv, np.ndarray)
                            and rv.dtype != np.object_
                            and rv.size > 0)
                if ok_frame:
                    rv = np.asarray(rv)
                    # RenderedVideo 解码出 BGR 直接用; images 组帧数组是 RGB 需转
                    if not isinstance(self.data.rendered, RenderedVideo):
                        rv = cv2.cvtColor(rv, cv2.COLOR_RGB2BGR)
                    put_label(rv, "Rendered Video", (12, 26), (0, 255, 255))
                    self._show_image(self.lbl_top_right, rv)
                else:
                    self.lbl_top_right.setText(
                        "Rendered video: frame decode failed")
            else:
                self.lbl_top_right.setText(
                    f"Rendered video too short ({n} < {idx + 1})")
        else:
            self.lbl_top_right.setText(
                "No rendered video found in h5\n"
                "(auto-detect: videos/hand_skeleton\n"
                "or images with preview/keypoint/render/...)")

        # 左下: 仿生手掌（触觉矩阵映射）— hidden when no tactile data
        if self.has_tactile:
            mats = []
            for side in ("left", "right"):
                ds = self.data.tactile[side]
                mats.append(np.asarray(ds[idx], np.float32)
                            if ds is not None
                            else np.zeros((16, 16), np.float32))
            w = self.lbl_hand.width() or 620
            h = self.lbl_hand.height() or 400
            bh = render_bionic_hand(mats[0], (w, h), self.data.tactile_vmax)
            br = render_bionic_hand(mats[1], (w, h), self.data.tactile_vmax)
            pad = np.full((h, 16, 3), 15, np.uint8)
            bhv = np.hstack([bh, pad, br])
            put_label(bhv, "Bionic Hand (L | R)", (12, 26), (0, 255, 255))
            self._show_image(self.lbl_hand, bhv)

            # 右下: 触觉热力图
            hm = [make_heatmap(m, size=280, vmax=self.data.tactile_vmax)
                  for m in mats]
            pad2 = np.full((280, 24, 3), 20, np.uint8)
            out = np.hstack([hm[0], pad2, hm[1]])
            put_label(out, "Tactile Heatmap (L | R)", (12, 26), (0, 255, 255))
            self._show_image(self.lbl_heat, out)

        # 底部: IMU 波形（最近 window 帧, 当前帧竖线 + 数值）— hidden when no imu
        if self.has_imu:
            w = self.lbl_imu.width() or 880
            h = self.lbl_imu.height() or 185
            imu = render_imu(self.data.imu_seq, self.data.imu_counts, idx,
                             window=240, w=max(w, 400), h=max(h, 120))
            self._show_image(self.lbl_imu, imu)

    @staticmethod
    def _show_image(label, bgr):
        if (not isinstance(bgr, np.ndarray) or bgr.dtype == np.object_
                or bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3):
            return
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ══════════════════════════════════════════════════════════════════
# 5. 入口
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="HDF5 data viewer (stereo video / "
                                             "rendered video / tactile / imu)")
    ap.add_argument("h5", nargs="?", help="hdf5 file path (optional, use the UI)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = DemoWindow(args.h5)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
