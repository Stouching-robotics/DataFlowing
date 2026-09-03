#!/usr/bin/env python3
"""live_demo.py —— D435 3D 手部关键点实时 demo（独立模块，不动主程序）。

链路（与离线 run_pipeline_d435.py 同源，仅替换批处理段）：
  实时 D435（rgb8 1280×720@30 + z16 848×480@30，与主程序 core/d435_camera.py
  同参；设备独占，主程序预览开启时 EBUSY）或 --replay 录制回放
    → MediaPipe 2D（GPU 冒烟成功则 GPU，否则 CPU）+ 手性投票（空帧跳过守卫）
    → 深度前向对齐（填洞轮数 --fill，默认 1 轮：覆盖 91.6% vs 3 轮 94.0%，
      省 20ms/帧，3×3 中位采样几乎不受影响）
    → 单目抬升 → 槽位分配（互斥+复活，与离线同决策层级）
    → HandSlotTracker αβ 在线平滑（替代离线 fill_gaps/offline_smooth）
    → Hand3DSmoother OneEuro 再平滑（压静止抖动，防跨手污染；label 变化/
      空→有重建帧几何近时 0.5 混合软衔接，防重置 snap）
    → 渲染输入链整手稳定性（仅展示路径，不回灌 tracker）：
        M5 补点深度锚定槽级 zc EMA（zc 逐帧独立中位是整手共模跳最大来源；
          实测点不动，补点 x,y 随 zc 反投影保持 2D 一致）
        M3 wholesale 两帧确认（单帧跳变/误检不采信不重置，预测桥接显示）
        M1 质心锚定（有效点中位质心强 OneEuro + 共模平移校正，替代原 EMA；
          手内形状原样保留，相机静止前提成立）
    → 三窗口：RGB 叠加（骨架+深度标注+HUD）+ 3D 骨架（手动视角：左键
      拖拽环绕/俯仰、r 复位，不拖拽视角完全静止——输入平移到首帧锁定的
      世界锚点，渲染器相机目标恒定，手的真实世界运动在固定网格中可见）
      + 深度图实时显示（aligned 深度 0.3-1.5m 伪彩）。

用法（venv，已含 pyrealsense2==2.58.3）:
    ./tools/hand_3d_d435/run_live_d435.sh                     # 直连相机
    ./tools/hand_3d_d435/run_live_d435.sh --replay data/recordings/222/222_000011
    ./tools/hand_3d_d435/run_live_d435.sh --uvc 2             # 其他相机（设备号）
    ./tools/hand_3d_d435/run_live_d435.sh --uvc rtsp://…      # 或路径/URL/本地视频
        （--uvc：该相机无深度 → 2D-only，仅骨架叠加渲染单窗口，不生成
        深度图/3D 关键点；--export 只产 keypoints_2d.parquet +
        rgb_overlay.mp4）

按键：q/ESC 退出 | s 截图（keypoints_output/live_d435/）| d 深度伪彩叠层切换
      | g 裸手/黑手套模式切换（--glove 启动即手套模式）
      | b 姿态后端切换 RTMPose↔MediaPipe（仅手套模式，track/滤波状态保留）
      | r 重置 3D 视角（方位/俯仰复位；世界锚点保持首帧锁定）。3D 窗口
      左键拖拽手动旋转视角（横向=环绕，纵向=俯仰）。
      （--uvc 2D-only 窗口无 d/r 键：没有深度图/3D 视图。）
"""

from __future__ import annotations

import argparse
import collections
import faulthandler
import glob
import math
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from stereo_s80m.hand_3d.detector import MediaPipeDetector        # noqa: E402
from stereo_s80m.hand_3d.identity import HandednessVoter          # noqa: E402
from stereo_s80m.hand_3d.track3d import HandSlotTracker           # noqa: E402
from stereo_s80m.hand_3d.smoother import Hand3DSmoother           # noqa: E402
from stereo_s80m.hand_3d.renderer_3d import RotatingSkeletonRenderer  # noqa: E402
from stereo_s80m.hand_3d.video_writer import create_video_sink    # noqa: E402
from stereo_s80m.hand_3d import io                                # noqa: E402

from hand_detection.hand_pipeline_mediapipe import (OneEuroFilter2D,  # noqa: E402
                                                    OneEuroFilter3D)

from hand_3d_d435.depth_align import (DepthAligner, load_calib,       # noqa: E402
                                      load_session_depth_shape,
                                      load_session_depth_files,
                                      load_depth_frame)
from hand_3d_d435.lift3d import (D435Pair, LiftResult, lift_hand,     # noqa: E402
                                 gate_observations, apply_slot_zc)
from hand_3d_d435.mono_assign import assign_mono_slots             # noqa: E402
from hand_3d_d435.render_overlay import (blend_depth, draw_overlay,  # noqa: E402
                                         _draw_hand, depth_to_heatmap_bgr)
from hand_3d_d435 import replay_compat                             # noqa: E402
from hand_3d_d435.glove_detector import (GloveDetector,           # noqa: E402
                                          resolve_glove_weights)

RENDER_SIZE = (1280, 720)
_ROT_TOTAL = 360            # 渲染器总帧：revolutions=1.0 时 frame_idx ≈ 方位角度数
_SHOT_DIR = os.path.join(_REPO_ROOT, "keypoints_output", "live_d435")
_FX_REL_TOL = 0.01
_VIEW_YAW0 = math.pi        # 初始视角方位角（rad）：π = 正面看手掌
_VIEW_ELEV0 = 25.0          # 初始俯仰角（deg）
_GATE_FORGIVE = 5           # M6：逐点连续被门控帧数上限，超限采信新观测（防锁死）


class LiveAligner(DepthAligner):
    """DepthAligner + 可调填洞轮数（默认 1，实时预算用）。

    align_depth_to_color 内部调用 self._fill_holes(aligned)——覆写该方法
    注入轮数即可，对齐数学零改动。
    """

    def __init__(self, *args, fill_passes: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.fill_passes = fill_passes

    def _fill_holes(self, aligned, passes=None):
        return super()._fill_holes(aligned, passes or self.fill_passes)


# ── 输入源 ────────────────────────────────────────────────────

class LiveD435Source:
    """pyrealsense2 直开 D435/D435I：rgb8 1280×720@30 + z16 848×480@30。

    设备独占：主程序 D435 预览开启时 pipe.start 报 EBUSY。
    对齐参数取自**本机实时内参/外参**（self.live_calib）——换设备（如
    D435→D435I）自动自洽，固化标定 JSON 只用于 --replay 回放会话。
    serial 与标定不符仅提醒（录制数据与当前设备不同源）。
    """

    def __init__(self, calib: dict, exposure: float = 0.0, serial: str = None):
        import pyrealsense2 as rs   # 惰性导入：--replay 路径无需此包
        self._rs = rs
        ctx = rs.context()
        # pipe.start 无设备时默认阻塞等待插入（不报错）→ 先枚举，无设备直接退出
        devs = [d for d in ctx.query_devices()]
        if len(devs) == 0:
            raise RuntimeError(
                "未枚举到 RealSense 相机（设备未插/驱动未加载）。\n"
                "  请插上相机后重跑；无相机验证请用 --replay <session_dir>。")
        if serial is not None:
            hit = [d for d in devs
                   if d.get_info(rs.camera_info.serial_number) == serial]
            if not hit:
                listed = "、".join(
                    f"{d.get_info(rs.camera_info.name)} "
                    f"{d.get_info(rs.camera_info.serial_number)}"
                    for d in devs)
                raise RuntimeError(
                    f"未找到 serial={serial} 的设备。当前枚举到: {listed}。\n"
                    f"  请核对 --rs-serial，或去掉该参数用枚举到的第一台。")
        # 流组合回退：D435 = depth 848×480 + color 1280×720；D405 深度/
        # 彩色须同分辨率（实测混合组合 "Couldn't resolve requests"）→
        # 自动回退 1280×720 双流、再 848×480@60 双流。对齐链内参驱动
        # 自适应，depth 分辨率变化无需其他改动。
        combos = [
            (rs.stream.color, 1280, 720, rs.format.rgb8, 30,
             rs.stream.depth, 848, 480, rs.format.z16, 30),
            (rs.stream.color, 1280, 720, rs.format.rgb8, 30,
             rs.stream.depth, 1280, 720, rs.format.z16, 30),
            (rs.stream.color, 848, 480, rs.format.rgb8, 60,
             rs.stream.depth, 848, 480, rs.format.z16, 60),
        ]
        prof = None
        last_err = None
        for i, (cs, cw, ch_, cfmt, cfps, ds, dw, dh, dfmt, dfps) \
                in enumerate(combos):
            self._pipe = rs.pipeline(ctx)   # 失败的 pipeline 不复用
            cfg = rs.config()
            if serial is not None:
                cfg.enable_device(serial)   # 多设备时锁定指定 serial
            cfg.enable_stream(cs, cw, ch_, cfmt, cfps)
            cfg.enable_stream(ds, dw, dh, dfmt, dfps)
            try:
                prof = self._pipe.start(cfg)
                if i:
                    print(f"⚠ 默认流组合不可用（{last_err}），已回退 "
                          f"RGB {cw}×{ch_}@{cfps} + 深度 {dw}×{dh}@{dfps}")
                break
            except RuntimeError as e2:
                last_err = str(e2)
                if "resolve" not in last_err.lower():
                    raise RuntimeError(
                        f"打开 D435 失败（{e2}）。\n"
                        "  若提示 Device or resource busy：主程序正在使用 "
                        "D435，请先关闭主程序的 D435 预览再运行本 demo。"
                    ) from None
        if prof is None:
            raise RuntimeError(
                f"打开相机失败：默认与回退流组合均无法 resolve"
                f"（{last_err}）")

        dev = prof.get_device()
        serial = dev.get_info(rs.camera_info.serial_number)
        cal_serial = calib.get("serial", "")
        if cal_serial and serial != cal_serial:
            print(f"⚠ 注意: 当前设备 serial={serial} ≠ 标定/录制设备 "
                  f"{cal_serial}")
            print("  对齐改用本机实时内参/外参（实时 demo 自洽）；"
                  "固化标定仍用于 --replay 回放会话。")
        else:
            print(f"✓ 设备 serial={serial} 与固化标定一致")

        # 实时内参/外参：彩色内参 + 深度内参 + depth→color 外参
        # （P_color = R·P_depth + t，与 rs2.align 同源语义）。键名/单位
        # 与固化标定 JSON 一致（rotation/translation，t 米——DepthAligner
        # 内部才 ×1000 转毫米）。
        cp = prof.get_stream(rs.stream.color).as_video_stream_profile()
        ci = cp.get_intrinsics()
        dp = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        di = dp.get_intrinsics()
        ext = dp.get_extrinsics_to(prof.get_stream(rs.stream.color))
        self.live_calib = {
            "color_intrinsics": {"fx": float(ci.fx), "fy": float(ci.fy),
                                 "cx": float(ci.ppx), "cy": float(ci.ppy),
                                 "width": int(ci.width),
                                 "height": int(ci.height)},
            "depth_intrinsics": {"fx": float(di.fx), "fy": float(di.fy),
                                 "cx": float(di.ppx), "cy": float(di.ppy),
                                 "width": int(di.width),
                                 "height": int(di.height)},
            "depth_to_color": {
                "rotation": np.asarray(ext.rotation, np.float64)
                .reshape(3, 3).tolist(),
                "translation": [float(x) for x in ext.translation]},
        }
        print(f"✓ 实时内参: 彩色 fx={ci.fx:.1f} fy={ci.fy:.1f} "
              f"深度 fx={di.fx:.1f} fy={di.fy:.1f} "
              f"t=[{ext.translation[0] * 1000:.2f}, "
              f"{ext.translation[1] * 1000:.2f}, "
              f"{ext.translation[2] * 1000:.2f}]mm")

        # 固定曝光（--exposure）：运动模糊是检测丢手的常见原因，
        # 室内 30fps 建议 4000-10000µs（默认 0 = 自动曝光）
        if exposure > 0:
            for s in dev.query_sensors():
                if s.get_info(rs.camera_info.name) == "RGB Camera":
                    s.set_option(rs.option.enable_auto_exposure, 0)
                    s.set_option(rs.option.exposure, float(exposure))
                    print(f"✓ 彩色曝光固定 {exposure:g}µs（自动曝光关）")
                    break

        self._scale = float(dev.first_depth_sensor().get_depth_scale())
        self._cap_color = None

    def next(self):
        """→ (bgr, depth_mm)；流断返回 (None, None)。"""
        fs = self._pipe.wait_for_frames(timeout_ms=1000)
        cf, df = fs.get_color_frame(), fs.get_depth_frame()
        if cf is None or df is None:
            return None, None
        # rs2 缓冲在 frameset 释放后失效 → 到手即拷贝
        rgb = np.asanyarray(cf.get_data()).copy()
        raw = np.asanyarray(df.get_data()).copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)      # 离线录制是 BGR24
        mm = raw.astype(np.float32) * self._scale * 1000.0
        return bgr, mm

    def close(self):
        self._pipe.stop()


class ReplaySource:
    """录制会话回放（无相机验证/演示用）：帧 n ↔ {n+1:06d}.png（1-based，
    uint16 毫米 PNG；v1.0.11 窗口 raw16 bin 由 load_depth_frame 回退兼容）。

    pace=True 按录制 fps 步调（处理快时 sleep 补齐）；pace=0 不限速。
    """

    def __init__(self, session: str, pace: float = 30.0, video: str = None,
                 depth_dir: str = None):
        video = video or io.find_video(session, "d435_rgb")
        if not video:
            sys.exit(f"错误: 找不到 RGB 视频: {session}/videos/d435_rgb/")
        self._cap = cv2.VideoCapture(video)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        depth_dir = depth_dir or os.path.join(session, "depth", "d435_depth")
        self._depth_files = {}
        self._depth_shape = load_session_depth_shape(session)
        if os.path.isdir(depth_dir):
            self._depth_files = load_session_depth_files(depth_dir)
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
            d = load_depth_frame(dp, self._depth_shape)
        if self._pace > 0:
            due = self._t0 + (self.n + 1) / self._pace
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(min(wait, 0.1))
        return rgb, d

    def close(self):
        self._cap.release()


class UvcSource:
    """其他相机（OpenCV 打开）：数字=设备号（/dev/videoN），或路径/URL
    （如 rtsp://…，本地视频文件也可，便于无相机验证）。无深度输出 →
    next() 恒返回 (bgr, None)，2D-only 渲染（不产深度图/3D 关键点）。"""

    def __init__(self, spec: str):
        dev: int | str = int(spec) if spec.isdigit() else spec
        self._cap = cv2.VideoCapture(dev)
        if not self._cap.isOpened():
            videos = [os.path.basename(p) for p in glob.glob("/dev/video*")]
            msg = f"打开相机失败: {spec}"
            if videos:
                msg += (f"\n  本机 /dev/video*: {', '.join(sorted(videos))}"
                        "\n  数字传设备号（--uvc 2 = /dev/video2）")
            msg += "\n  也可传路径或 URL（如 rtsp://…）；设备被占用先关其他程序。"
            raise RuntimeError(msg)
        # 实时流缓冲缩到 1 帧：RTSP/UVC 长开后 backlog 会让渲染滞后越积越多
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"✓ 相机已打开: {spec}（无深度 → 2D-only 关键点渲染）")

    def next(self):
        """→ (bgr, None)；流断/文件播放完返回 (None, None)。"""
        ok, bgr = self._cap.read()
        if not ok:
            return None, None
        return bgr, None

    def close(self):
        self._cap.release()


# ── 相机枚举 + 热切换 ──────────────────────────────────────────

# ── 小工具（照搬离线）──────────────────────────────────────────

def _nan_pair(label: str = "") -> D435Pair:
    return D435Pair(result=LiftResult(np.full((21, 3), np.nan, np.float64),
                                      float("nan"), 0), left_label=label)


def _pred_pair(pred: np.ndarray, label: str = "") -> D435Pair:
    return D435Pair(result=LiftResult(np.asarray(pred, np.float64)
                                      .reshape(21, 3), float("nan"), 0),
                    left_label=label)


def _exp_meta(rows: list) -> dict:
    """parquet 公共列：frame/slot/label/state（帧×槽每行一条）。"""
    import pyarrow as pa
    return {
        "frame": pa.array([r[0] for r in rows], pa.int32()),
        "slot": pa.array([r[1] for r in rows], pa.int8()),
        "label": pa.array([r[2] for r in rows], pa.string()),
        "state": pa.array([r[3] for r in rows], pa.string()),
    }


def _write_export(export_dir: str, rows2d: list, rows3d: list,
                  sink3d, sink2d, n: int) -> None:
    """--export 落盘：两个 parquet（二进制 zstd，xy/xyz 为定长 float32
    数组列，NaN=无观测）+ render.mp4 / rgb_overlay.mp4（H.264+faststart，
    帧循环已用共享 sink 写入，此处 close 触发 ffmpeg 收尾）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(export_dir, exist_ok=True)
    p2 = os.path.join(export_dir, "keypoints_2d.parquet")
    p3 = os.path.join(export_dir, "keypoints_3d.parquet")
    pq.write_table(pa.table({**_exp_meta(rows2d),
                             "xy": pa.array([r[4:] for r in rows2d],
                                            pa.list_(pa.float32(), 42))}),
                   p2, compression="zstd")
    pq.write_table(pa.table({**_exp_meta(rows3d),
                             "xyz": pa.array([r[4:] for r in rows3d],
                                             pa.list_(pa.float32(), 63))}),
                   p3, compression="zstd")
    print(f"\n导出: {p2}\n      {p3}")
    for name, sink in (("render.mp4", sink3d), ("rgb_overlay.mp4", sink2d)):
        if sink is None:
            continue
        final = sink.close()   # None = 编码失败已放弃（半成品 mp4 已删）
        if final:
            print(f"      {final}（{n} 帧，30fps，H.264）")
        else:
            print(f"      [警告] {name} 编码失败已放弃（parquet 不受影响）")


def _write_export_2d(export_dir: str, rows2d: list, sink2d, n: int) -> None:
    """--export（--uvc 2D-only）：keypoints_2d.parquet（二进制 zstd，xy 定长
    float32[42]，NaN=无观测）+ rgb_overlay.mp4。无深度 → 不产 3D 产物。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(export_dir, exist_ok=True)
    p2 = os.path.join(export_dir, "keypoints_2d.parquet")
    pq.write_table(pa.table({**_exp_meta(rows2d),
                             "xy": pa.array([r[4:] for r in rows2d],
                                            pa.list_(pa.float32(), 42))}),
                   p2, compression="zstd")
    print(f"\n导出: {p2}（2D-only 相机，无深度 → 无 keypoints_3d.parquet/"
          f"render.mp4）")
    if sink2d is not None:
        final = sink2d.close()   # None = 编码失败已放弃（半成品 mp4 已删）
        if final:
            print(f"      {final}（{n} 帧，30fps，H.264）")
        else:
            print(f"      [警告] rgb_overlay.mp4 编码失败已放弃"
                  f"（parquet 不受影响）")


def _resolve_delegate(want: str) -> str:
    """auto → GPU 子进程冒烟（init 可能 SIGSEGV，进程内拦不住），失败回 CPU。"""
    if want != "auto":
        return want
    model_path = os.path.join(_REPO_ROOT, "tools", "models",
                              "hand_landmarker.task")
    if not os.path.isfile(model_path):
        print("  [模型缺失，用 CPU]")
        return "cpu"
    from stereo_s80m.hand_3d.mp_gpu import smoke_test_gpu
    if smoke_test_gpu(model_path):
        return "gpu"
    return "cpu"


# ── 3D 视角手动控制 ───────────────────────────────────────────

class _OrbitControl:
    """3D 窗口手动轨道控制：左键拖拽环绕/俯仰，r 复位（默认正面视角）。

    不动渲染器文件：renderer 的相机方位角由 frame_idx 驱动
    （theta = 2π·revolutions·frame_idx/(total-1)），把 yaw 反解成
    frame_idx 传入即可得到任意静态视角；俯仰用渲染器构造参数 elevation
    （实例属性，运行时赋值）。revolutions=1.0、total=360 时 HUD 的
    "frame x/360" 即当前方位角度数。
    """

    def __init__(self, yaw: float = _VIEW_YAW0, elev: float = _VIEW_ELEV0):
        self.yaw = float(yaw)          # 相机方位角（rad，π = 正面）
        self.elev = float(elev)        # 俯仰角（deg，正 = 俯视）
        self._dragging = False
        self._last = None

    def frame_idx(self, total: int, revolutions: float = 1.0) -> float:
        """方位角 → renderer 的 frame_idx（float，HUD 显示度数）。"""
        return self.yaw / (2.0 * math.pi * revolutions) * (total - 1)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging, self._last = True, (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            dx = x - self._last[0]
            dy = y - self._last[1]
            self._last = (x, y)
            self.yaw -= dx * 0.01                 # 右拖 = 相机向左环绕
            self.elev = float(np.clip(self.elev - dy * 0.15, -70.0, 80.0))

    def reset(self):
        self.yaw, self.elev = _VIEW_YAW0, _VIEW_ELEV0


# ── 渲染输入链稳定性（M1/M3/M5，仅展示路径，不回灌 tracker） ──────────

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
    """M3① 包装 Hand3DSmoother：镜像其 label 变化/"空→有"重建判定
    （smoother.py 只读，重置语义在自有包装类实现）。重建帧且几何近
    （<0.1m，同一只手漏检回归/标签闪烁）时喂 0.5 混合输入（旧平滑
    输出 + 新观测）软衔接——否则 pop 滤波器后首帧输出=原始输入 →
    snap；几何远（真换手）不混，保持硬重置防跨手污染。
    absent 帧输出全 NaN 时不覆盖 prev_out：重捕捉帧"空→有"触发器
    命中时 po 取丢失前最后一帧有限输出，软衔接才判得出（此前 NaN
    覆盖 → pofin≥4 永远失败 → 重捕捉必硬重置 snap = 闪烁主因之一）。
    """

    _MIN_PTS = 4          # 质心可靠下限
    _SOFT_DIST = 0.10     # 重建帧软衔接判距（米）

    def __init__(self, smoother):
        self._sm = smoother
        self._prev_out = [None, None]
        self._prev_labels = [None, None]
        self._prev_pres = [False, False]

    def update(self, h3, labels, valids):
        pres_flags = [v >= 8 for v in valids]
        for s in range(2):
            if (labels[s] != self._prev_labels[s]
                    or (pres_flags[s] and not self._prev_pres[s])):
                po = self._prev_out[s]
                if po is None:
                    continue
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
        for s in range(2):
            if np.isfinite(out[s]).all(axis=1).sum() >= self._MIN_PTS:
                self._prev_out[s] = out[s]    # absent 帧保留上次有限输出
        self._prev_labels = list(labels)
        self._prev_pres = pres_flags
        return out


class _OneEuro2DSmoother:
    """2D 关键点逐点 OneEuro 平滑（--smooth-2d，默认关）。

    3D 有 tracker αβ + OneEuro + 质心锚，2D 却画原始检测 landmark。
    按"槽/手"身份键管理 21 点滤波器组：键变化（换手）即重置该组防跨
    手污染；NaN 点跳过不更新。

    ★ 实测教训（2026-08-24 S80C）：freq_min=3Hz 在窗口链 20-45ms/帧
    下 alpha≈0.3-0.5 → 输出滞后 ~2-3 帧（~90ms），手一移动骨架明显
    落后于手——用户报"关键点不在手上、偏移"。默认已关闭；重开须把
    freq_min 降 ~1Hz 并按实际链帧率评估滞后。
    """

    def __init__(self, n_groups: int = 2,
                 freq_min: float = 3.0, beta: float = 0.3,
                 dcutoff: float = 1.0):
        self._groups = [[OneEuroFilter2D(freq_min, beta, dcutoff)
                         for _ in range(21)] for _ in range(n_groups)]
        self._keys = [None] * n_groups

    def reset(self, g: int):
        for f in self._groups[g]:
            f.reset()
        self._keys[g] = None

    def update(self, g: int, pts: np.ndarray, key, ts_ms: float) -> np.ndarray:
        """pts (21,2)（NaN 点跳过）；返回平滑后同形。key 变化自动重置。"""
        if self._keys[g] != key:
            self.reset(g)
            self._keys[g] = key
        out = np.full((21, 2), np.nan, np.float32)
        for k in range(21):
            x, y = float(pts[k, 0]), float(pts[k, 1])
            if np.isfinite(x) and np.isfinite(y):
                out[k, 0], out[k, 1] = self._groups[g][k](x, y, ts_ms)
        return out


class _CentroidAnchor:
    """M1 质心锚定：整手共模跳抑制（相机静止前提，参考单目动捕"对质心
    轨迹做很强的平滑"经验）。

    每槽有效点中位质心 c 走强 OneEuro（默认 freq_min=3.0, beta=0.3,
    dcutoff=0.3m/s：静止抖动被强衰减，>0.3m/s 的真实手部运动快速通过），
    输出 = 输入 + (ĉ − c)——共模平移在质心层被吸收，手内形状与手势
    动力学原样保留（比逐点再滤波更保形状；逐点独立平滑正是质心每帧
    微跳的来源）。label 变化且几何近（<0.1m，同一只手重建）时软衔接
    （0.5 混合旧 ĉ 与新 c）防重置 snap；几何远（真换手）硬重置。
    absent→real 重入同样软衔接：absent 帧只记无手、滤波器与 prev_c
    跨缺口保留（此前 pop → 重现必从原始观测起 → 质心 OneEuro 冷启动
    = 整手 3D 重捕捉前几帧跳 = 闪烁主因之一）；大 dt 时 OneEuro
    α→1，保留滤波器无风险。
    """

    _MIN_PTS = 4          # 有效点下限（质心不可靠则跳过该槽）
    _SOFT_DIST = 0.10     # 重建帧软衔接判距（米）

    def __init__(self, freq_min: float = 3.0, beta: float = 0.3,
                 dcutoff: float = 0.3):
        # 默认参数 = D435 深度口径；深度更噪的源（S80C/S80M：~20fps
        # 更新、有效率低）由调用方调低 freq_min 加强静止平滑（滞后换稳定）。
        self._freq_min = freq_min
        self._beta = beta
        self._dcutoff = dcutoff
        self._filters = {}          # slot → OneEuroFilter3D
        self._prev_labels = [None, None]
        self._prev_c = [None, None]      # 上一帧输出质心（软衔接用）
        self._prev_fin = [False, False]  # 上一帧是否有手（absent→real 判定）
        self._t0 = time.perf_counter()

    def apply(self, hands3d, labels) -> np.ndarray:
        pts = np.asarray(hands3d, np.float64).reshape(2, 21, 3)
        out = pts.copy()
        ts = (time.perf_counter() - self._t0) * 1000.0
        for s in range(2):
            fin = np.isfinite(pts[s]).all(axis=1)
            if fin.sum() < self._MIN_PTS:
                self._prev_fin[s] = False
                self._prev_labels[s] = labels[s]
                continue
            c = np.median(pts[s, fin], axis=0)
            reentered = (not self._prev_fin[s]
                         and self._filters.get(s) is not None)
            if labels[s] != self._prev_labels[s]:
                if (self._filters.get(s) is not None
                        and self._prev_c[s] is not None
                        and np.linalg.norm(self._prev_c[s] - c)
                        < self._SOFT_DIST):
                    c = 0.5 * self._prev_c[s] + 0.5 * c    # 软衔接防 snap
                self._filters[s] = OneEuroFilter3D(self._freq_min, self._beta, self._dcutoff)
                reentered = False
            elif reentered:
                # absent→real 重入：几何近 0.5 混合软衔接（镜像 label
                # 软衔接），几何远（手在丢失期间真移动）重建滤波器
                if (self._prev_c[s] is not None
                        and np.linalg.norm(self._prev_c[s] - c)
                        < self._SOFT_DIST):
                    c = 0.5 * self._prev_c[s] + 0.5 * c
                else:
                    self._filters[s] = OneEuroFilter3D(self._freq_min, self._beta, self._dcutoff)
            if s not in self._filters:   # 从未见过该槽：从观测起
                self._filters[s] = OneEuroFilter3D(self._freq_min, self._beta, self._dcutoff)
            c_hat = np.asarray(self._filters[s](c[0], c[1], c[2], ts),
                               np.float64)
            out[s, fin] = pts[s, fin] + (c_hat - c)
            self._prev_c[s] = c_hat
            self._prev_fin[s] = True
            self._prev_labels[s] = labels[s]
        return out.astype(np.float32)


# ── UVC 2D-only 循环 ──────────────────────────────────────────

def _glove_kwargs(args):
    """GloveDetector 构造公共参数（--glove-* 调参透传到全部构造点）。"""
    return dict(imgsz=args.glove_imgsz,
                det_conf=args.glove_det_conf,
                pose_conf_thr=args.glove_pose_conf,
                hold_max=args.glove_hold_max,
                nms_iou=args.glove_nms_iou,
                lost_timeout=args.glove_lost_timeout,
                box_alpha=args.glove_box_alpha,
                freeze_max=args.glove_freeze_max,
                pose_box_raw=(args.glove_pose_box == "raw"),
                new_track_conf=args.glove_new_track_conf,
                pose_backend=args.glove_pose_backend)


def _run_uvc_2d(args):
    """--uvc：其他相机 2D-only 实时关键点渲染。

    无深度 → 不走对齐/抬升/槽位/3D 平滑链（3D 链对深度有硬依赖），仅
    检测 → 2D 叠加渲染（单窗口）+ 可选 --export（2D parquet + 叠加视频）。
    裸手/手套双模式与 g 键切换与 3D 路径同语义。
    """
    source = UvcSource(args.uvc)
    delegate = _resolve_delegate(args.delegate)
    print(f"检测 delegate: {delegate}（det/track conf "
          f"{args.det_conf}/{args.track_conf}）")
    det = MediaPipeDetector(num_hands=2, delegate=delegate,
                            det_conf=args.det_conf,
                            track_conf=args.track_conf)
    # 黑手套模式：与 3D 路径同语义（惰性——未 --glove 时首次按 g 才加载）
    glove_choice = args.glove_detector
    glove_weights = resolve_glove_weights(glove_choice, args.glove_weights)
    glove_det = None
    glove_mode = args.glove
    if glove_mode:
        try:
            glove_det = GloveDetector(weights=glove_weights,
                                      **_glove_kwargs(args))
            print(f"黑手套模式: {glove_weights}（{glove_det.backend} "
                  f"{glove_det.device}，pose {glove_det.pose_backend}/"
                  f"{glove_det.pose_device}，"
                  f"conf {glove_det.det_conf}；按 g 切回裸手，"
                  f"b 切姿态后端，v 换检测器）")
        except FileNotFoundError:
            sys.exit(f"错误: 手套权重不存在: {glove_weights}")
    # 裸手 label 稳定靠 voter（与 3D 路径同守卫：空帧跳过防清轨迹）；
    # 手套身份由 GloveDetector per-track 锁存承担，跳过 voter（同 3D 路径）
    voter = HandednessVoter()
    win_title = "UVC live: 2D hands"

    if not args.no_window:
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, 1024, 576)
    shot_n = 0
    stats = {"n": 0, "det0": 0, "det1": 0, "det2": 0,
             "glove_n": 0, "glove_box": 0}
    export_dir = args.export
    exp2d_rows: list = []      # (frame, slot, label, state, x0..x20, y0..y20)
    exp_sink2d = None          # 原视频叠 2D 关键点 sink（首帧尺寸确定后惰性建）
    n = 0
    t_det = 0.0
    t0 = time.perf_counter()
    fps_win_t, fps_win_n = t0, 0
    fps = 0.0
    try:
        while True:
            rgb, _d = source.next()
            if rgb is None:
                break

            t1 = time.perf_counter()
            hands = (glove_det if glove_mode else det).detect(rgb)
            stats["n"] += 1
            stats[f"det{min(len(hands), 2)}"] += 1
            if glove_mode:
                stats["glove_n"] += 1
                stats["glove_box"] += 1 if glove_det.last_boxes else 0
            if hands and not glove_mode:
                voter.update(hands, frame_w=rgb.shape[1],
                             frame_h=rgb.shape[0], frame=n, cam="uvc")
            t_det += time.perf_counter() - t1

            # 2D-only：无槽位分配，直接按检测顺序装 (2,21,2)，不足补 NaN
            hands2d = np.full((2, 21, 2), np.nan, np.float32)
            labels = ["", ""]
            presents = [False, False]
            for s in range(min(len(hands), 2)):
                hands2d[s] = np.asarray(hands[s].landmarks,
                                        np.float32).reshape(21, 2)
                labels[s] = hands[s].label
                presents[s] = True
            # 无深度 → 3D 全 NaN：draw_overlay 逐点深度标注与 HUD wrist
            # 深度有 NaN 守卫，自动跳过（只画骨架 + 手性 label）
            hands3d = np.full((2, 21, 3), np.nan, np.float32)

            if export_dir is not None:
                for s in range(2):
                    exp2d_rows.append(
                        [n, s, labels[s], "real" if presents[s] else "absent"]
                        + [float(v) if np.isfinite(v) else float("nan")
                           for v in hands2d[s].flatten()])
                ov_exp = draw_overlay(rgb.copy(), hands2d, hands3d, labels,
                                      [False, False], presents, n, n + 1,
                                      win_title + (" [GLOVE]"
                                               if glove_mode else ""))
                if glove_mode and glove_det is not None:
                    for bx1, by1, bx2, by2, bconf in glove_det.last_boxes:
                        cv2.rectangle(ov_exp, (int(bx1), int(by1)),
                                      (int(bx2), int(by2)), (0, 255, 0), 2)
                        cv2.putText(ov_exp, f"{bconf:.2f}",
                                    (int(bx1), max(16, int(by1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                if exp_sink2d is None:
                    # 共享 sink（stereo_s80m.hand_3d.video_writer）：
                    # nvenc→libx264 管道→mp4v 两段式逐级回退，H.264+
                    # faststart 直出（venv cv2 只能写 mp4v，播放器打不开）
                    os.makedirs(export_dir, exist_ok=True)
                    exp_sink2d = create_video_sink(
                        os.path.join(export_dir, "rgb_overlay.mp4"),
                        30, ov_exp.shape[1], ov_exp.shape[0])
                exp_sink2d.write(np.ascontiguousarray(ov_exp))

            if not args.no_window:
                ov = draw_overlay(rgb, hands2d, hands3d, labels,
                                  [False, False], presents, n, n + 1,
                                  win_title + (" [GLOVE]"
                                               if glove_mode else ""))
                if glove_mode and glove_det is not None:
                    for bx1, by1, bx2, by2, bconf in glove_det.last_boxes:
                        cv2.rectangle(ov, (int(bx1), int(by1)),
                                      (int(bx2), int(by2)), (0, 255, 0), 2)
                        cv2.putText(ov, f"{bconf:.2f}",
                                    (int(bx1), max(16, int(by1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                now = time.perf_counter()
                fps_win_n += 1
                if now - fps_win_t >= 1.0:
                    fps = fps_win_n / (now - fps_win_t)
                    fps_win_t, fps_win_n = now, 0
                cv2.putText(ov, f"{fps:5.1f} fps  [q]uit [s]hot [g]love "
                                f"[b]ackend [2D-only, no depth]",
                            (12, ov.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (80, 220, 255), 1, cv2.LINE_AA)
                cv2.imshow(win_title, ov)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    os.makedirs(_SHOT_DIR, exist_ok=True)
                    p1 = os.path.join(_SHOT_DIR,
                                      f"shot_{shot_n:03d}_overlay.png")
                    cv2.imwrite(p1, ov)
                    print(f"  截图: {p1}")
                    shot_n += 1
                if key == ord("b"):
                    # 姿态后端热切换：只换 GloveDetector._pose（tracker/
                    # OneEuro/手性锁存状态保留）；非手套模式仅提示
                    if glove_mode and glove_det is not None:
                        new_backend = ("mediapipe"
                                       if glove_det.pose_backend == "rtmpose"
                                       else "rtmpose")
                        glove_det.set_pose_backend(new_backend)
                        print(f"[b] 姿态后端 → {new_backend}"
                              f"（{glove_det.pose_device}，"
                              f"track/滤波状态保留）")
                    else:
                        print("[b] 仅黑手套模式可用（按 g 进入）")
                if key == ord("v"):
                    # 检测器热切换：world ↔ det 重建 GloveDetector
                    # （track/OneEuro/手性锁存重置——同 g 键语义）
                    if args.glove_weights:
                        print("[v] --glove-weights 已显式指定权重，"
                              "v 键不生效")
                    elif glove_mode:
                        new_choice = ("det" if glove_choice == "world"
                                      else "world")
                        new_weights = resolve_glove_weights(
                            new_choice, args.glove_weights)
                        try:
                            _gd = GloveDetector(weights=new_weights,
                                                **_glove_kwargs(args))
                        except FileNotFoundError:
                            print(f"[v] 权重不存在: {new_weights}，切换取消")
                        else:
                            if glove_det is not None:
                                glove_det.close()
                            glove_det = _gd
                            glove_choice = new_choice
                            print(f"[v] 检测器 → {glove_det.backend} "
                                  f"{new_weights}（conf {glove_det.det_conf}"
                                  f"；track/滤波状态重置）")
                    else:
                        print("[v] 仅黑手套模式可用（按 g 进入）")
                if key == ord("g"):
                    glove_mode = not glove_mode
                    if glove_mode:
                        if glove_det is None:
                            try:
                                glove_det = GloveDetector(
                                    weights=glove_weights,
                                    **_glove_kwargs(args))
                            except FileNotFoundError:
                                print(f"[g] 手套权重不存在: {glove_weights}，"
                                      f"切换取消")
                                glove_mode = False
                        if glove_mode:
                            print(f"[g] 黑手套模式（{glove_det.backend} "
                                  f"{glove_det.device}）")
                    else:
                        # 切回裸手清追踪态：不能调 det.reset()（只
                        # landmarker.close() 不重建，实机踩过）→ 重建实例
                        if glove_det is not None:
                            glove_det.close()
                            glove_det = None
                        det.close()
                        det = MediaPipeDetector(num_hands=2, delegate=delegate,
                                                det_conf=args.det_conf,
                                                track_conf=args.track_conf)
                        print("[g] 裸手模式")
            else:
                now = time.perf_counter()
                fps_win_n += 1
                if now - fps_win_t >= 1.0:
                    fps = fps_win_n / (now - fps_win_t)
                    fps_win_t, fps_win_n = now, 0
                    print(f"  frame {n}: {fps:.1f} fps  "
                          f"det {t_det / max(n, 1) * 1000:.1f}ms")
            n += 1
    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        source.close()
        det.close()
        if glove_det is not None:
            glove_det.close()
        cv2.destroyAllWindows()

    if n:
        t_total = time.perf_counter() - t0
        print(f"\n── {n} 帧, {t_total:.1f}s, 平均 {n / t_total:.1f} fps "
              f"（det 均 {t_det / n * 1000:.1f}ms）──")
    if export_dir is not None:
        _write_export_2d(export_dir, exp2d_rows, exp_sink2d, n)
    if args.stats and stats["n"]:
        nn = stats["n"]
        print("── 诊断（--stats）──")
        print(f"  检测手数: 0 手 {stats['det0']} 帧 ({stats['det0'] / nn * 100:.1f}%)"
              f" | 1 手 {stats['det1']} 帧 ({stats['det1'] / nn * 100:.1f}%)"
              f" | 2 手 {stats['det2']} 帧 ({stats['det2'] / nn * 100:.1f}%)")
        if stats["glove_n"]:
            print(f"  手套模式: {stats['glove_n']} 帧 | 出框 "
                  f"{stats['glove_box']} 帧"
                  f" ({stats['glove_box'] / stats['glove_n'] * 100:.1f}%)")


# ── 主循环 ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", help="回放录制会话目录（无相机验证/演示）")
    ap.add_argument("--replay-pace", type=float, default=30.0,
                    help="回放步调 fps（0=不限速，默认 30）")
    ap.add_argument("--uvc", metavar="SPEC",
                    help="接入其他相机（OpenCV 打开：数字=设备号 /dev/videoN，"
                         "或路径/URL 如 rtsp://…、本地视频文件）。该相机无深度"
                         "输出 → 仅 2D 关键点叠加渲染（单窗口），不生成深度图/"
                         "3D 关键点；--export 只产 keypoints_2d.parquet + "
                         "rgb_overlay.mp4；与 --replay 互斥")
    ap.add_argument("--calib", help="标定 JSON（默认 hand_3d_d435/calibration/）")
    ap.add_argument("--delegate", default="auto", choices=("cpu", "gpu", "auto"),
                    help="MediaPipe delegate（auto=GPU 冒烟成功则 GPU）")
    ap.add_argument("--fill", type=int, default=1, choices=(0, 1, 2, 3),
                    help="对齐空穴回填轮数（默认 1，实时预算；0=不填洞"
                         "最快——S80C 恒等对齐 1:1 无上采样空穴用；"
                         "离线验收用 3）")
    ap.add_argument("--propagate-max", type=int, default=15,
                    help="槽位丢失帧数硬顶（超限 absent 不幻觉）")
    ap.add_argument("--det-conf", type=float, default=0.4,
                    help="掌心检测置信度阈值（默认 0.4；动作快/丢手可再降到 0.3）")
    ap.add_argument("--track-conf", type=float, default=0.4,
                    help="手部跟踪置信度阈值（默认 0.4；丢手可再降到 0.3）")
    ap.add_argument("--smooth-2d", action="store_true",
                    help="2D 关键点逐点 OneEuro 平滑（默认关——平滑引入"
                         "~2-3 帧跟随滞后，快动时骨架明显落后于手；D435/"
                         "S80C 均默认关，闪动严重再开）")
    ap.add_argument("--det-scale", type=float, default=1.0,
                    help="裸手检测输入缩放比（默认 1.0 全分辨率；S80C 默认"
                         "0.5——CPU XNNPACK 检测耗时 ~2 倍于分辨率差，半分辨率"
                         "省 ~7ms/帧，MediaPipe 尺度容忍）")
    ap.add_argument("--raw-2d", action="store_true",
                    help="2D 显示用原始检测直绘（与 3D 槽位链解耦；D435 "
                         "默认关，S80C 默认开——槽位门控下骨架消失/停摆是"
                         "闪烁/偏移观感来源之一）")
    ap.add_argument("--extrap-2d", action="store_true",
                    help="det_async 时 2D 显示关键点按框中心速度外推平移"
                         "（默认关；S80C 默认开——异步检测结果来自 1-3 帧"
                         "前图像，直接画当前帧=快动手时骨架落后于手）")
    ap.add_argument("--tear-probe", action="store_true",
                    help="保存最近 96 帧内部显示缓冲（½ 尺寸），按 q 退出"
                         "自动导出 tear_exit_*（t 键可随时手动导出）到 "
                         "keypoints_output/ 做撕裂诊断（默认关；屏幕合成"
                         "撕裂不会出现在内部帧里，帧内有水平缝=数据/相机"
                         "侧撕裂）")
    ap.add_argument("--glove", action="store_true",
                    help="启动即黑手套模式：glove_package YOLO-World 框 + "
                         "关键点（后端见 --glove-pose-backend；运行中按 g 切换）")
    ap.add_argument("--glove-detector", choices=("world", "det"),
                    default="world",
                    help="黑手套检测器选择：world=yolov8m-worldv2.pt（开放"
                         "词汇，提示词 hand/glove，默认）；det=glove_package"
                         "/runs/hand_det/weights/best.pt（yolo11n 单类 hand "
                         "训练产物，自动走 YOLO 后端）。运行中按 v 键热切换"
                         "（track/滤波状态重置）")
    ap.add_argument("--glove-imgsz", type=int, default=320,
                    help="world 检测输入边长（仅 world 后端；D435 默认 "
                         "320——40 张近景实测最优；S80C 默认 640——远手/"
                         "小面积手在 320 下等效 4× 缩小掉出模型有效尺度，"
                         "640 放大 2 倍找回，代价 GPU ~23→31ms/帧）")
    ap.add_argument("--glove-weights",
                    help="黑手套检测权重显式路径（优先于 --glove-detector；"
                         "按文件名是否含 world 自动判后端）")
    ap.add_argument("--glove-pose-backend", choices=("rtmpose", "mediapipe"),
                    default="rtmpose",
                    help="黑手套关键点后端：rtmpose=RTMPose hand5（ONNX "
                         "SIMCC 256x256，默认）；mediapipe=MediaPipe "
                         "HandLandmarker（tools/models/hand_landmarker.task，框内"
                         "裁剪检测，21 点同拓扑）。运行中按 b 键热切换"
                         "（track 追踪/逐点 OneEuro/手性锁存状态保留）。"
                         "注意：MediaPipe 掌部检测对黑手套不友好（历史 "
                         "4/68 检出率），预期明显差于 RTMPose，主要供裸手/"
                         "效果对比场景")
    ap.add_argument("--glove-det-conf", type=float, default=None,
                    help="黑手套模式检测框阈值（默认 world 0.05 / best.pt 0.3）")
    ap.add_argument("--glove-pose-conf", type=float, default=0.3,
                    help="RTMPose 逐点置信均值门（0=关）：低于门（背面/侧视骨架/"
                         "快动模糊）持出上次输出并按平滑框位移平移补偿（骨架随框"
                         "滑动不钉死），置信恢复自动续上；连续低置信满 "
                         "--glove-hold-max 帧则放行本轮骨架（持续低置信=真实新"
                         "姿势如握拳/抓取，不无限冻结）")
    ap.add_argument("--glove-hold-max", type=int, default=12,
                    help="黑手套低置信 hold 逃逸帧数（0=立即放行，-1=无限 hold"
                         "旧行为）：pose 置信连续低于 --glove-pose-conf 满该帧"
                         "数即放行本轮低置信骨架——握拳/抓取等黑手套姿势逐点"
                         "置信天然偏低，无限 hold 会把骨架永久冻在旧姿势；短于"
                         "该帧数的瞬时低置信（运动模糊，实测短段 2-7 帧）仍持"
                         "旧点防抖")
    ap.add_argument("--glove-nms-iou", type=float, default=0.6,
                    help="world 检测器 NMS IoU（双手重叠框频繁合并→手数闪变时"
                         "降 0.45）")
    ap.add_argument("--glove-lost-timeout", type=int, default=3,
                    help="track 丢框容忍帧数。双手交叉/重叠导致框合并又拆开时"
                         "可调 8：000004 实测 2 手 85.3%%→88.3%%、0 手 4.2%%→1.7%%；"
                         "但单手进出场景（Test_Data_000003）调 8 会把静止段 p95 "
                         "12.4→25.1mm 恶化，按会话取舍")
    ap.add_argument("--glove-new-track-conf", type=float, default=0.25,
                    help="新建 track 的最低框置信度（双阈值，0=关）：匹配"
                         "已有 track 不受限，低于此值的检测框不新建 track"
                         "——快动/出画时 world 的低置信闪框会在两手位置间"
                         "来回跳，每帧新建 track 导致滤波重置、骨架瞬移"
                         "（000005 实测 f161-200 连续 15 帧新建 track）。"
                         "只影响新手进入门槛，不影响已跟踪手的保持")
    ap.add_argument("--glove-box-alpha", type=float, default=0.7,
                    help="track 框 EMA 平滑系数（身份匹配/显示用；RTMPose "
                         "裁剪框来源见 --glove-pose-box）")
    ap.add_argument("--glove-pose-box", choices=("raw", "smooth"),
                    default="smooth",
                    help="RTMPose 裁剪框来源：smooth=EMA 平滑框（默认，链稳定；"
                         "3D 下游 wholesale/渲染指标好）；raw=原始检测框（消除"
                         "平滑框 ~2-3 帧稳态滞后，但框抖动直通下游——实测 "
                         "Test_Data_000003 wholesale 19 vs 6、renderer_in p95 "
                         "34.4 vs 11.8mm，仅极端快动且指标可接受时试）")
    ap.add_argument("--glove-freeze-max", type=int, default=15,
                    help="连续退化冻结输出上限帧数")
    ap.add_argument("--rs-serial",
                    help="指定 RealSense 序列号（多台共存时锁定设备）；"
                         "不指定=枚举第一台")
    ap.add_argument("--exposure", type=float, default=0.0,
                    help="固定彩色曝光 µs（0=自动；运动模糊丢手时试 4000-10000）")
    ap.add_argument("--stats", action="store_true",
                    help="统计诊断：退出时打印单手帧/同标签帧/wholesale 等计数")
    ap.add_argument("--depth-overlay", action="store_true",
                    help="启动即开深度伪彩叠层（运行中按 d 切换）")
    ap.add_argument("--no-window", action="store_true",
                    help="不开窗口只跑处理链并打印 fps（验证用）")
    ap.add_argument("--export", help="导出目录：keypoints_2d.parquet（槽位 2D "
                                    "关键点，像素）、keypoints_3d.parquet（质心锚定 "
                                    "3D，相机系米）——二进制 zstd，xy/xyz 定长 "
                                    "float32 数组列；render.mp4（3D 旋转渲染）、"
                                    "rgb_overlay.mp4（原视频叠 2D 关键点）；窗口/"
                                    "无窗口均可，帧号从 0 起，NaN=该帧该槽无观测")
    args = ap.parse_args()

    if args.uvc is not None and args.replay:
        ap.error("--uvc 与 --replay 互斥（--uvc 是独立 2D-only 模式）")
    if args.uvc is not None:
        try:
            _run_uvc_2d(args)
        except RuntimeError as e:
            print(f"错误: {e}")
            sys.exit(1)
        return

    try:
        calib = load_calib(args.calib)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(2)

    if args.replay:
        session = args.replay.rstrip("/")
        if not os.path.isdir(session):
            sys.exit(f"错误: 会话目录不存在: {session}")
        rvideo = replay_compat.find_video_any(session)
        if rvideo is None:
            sys.exit(f"错误: 找不到 RGB 视频: {session}/videos/")
        source = ReplaySource(session, pace=args.replay_pace, video=rvideo,
                              depth_dir=replay_compat.find_depth_dir(session))
        session_depth = replay_compat.load_session_depth_intr_any(session)
        if session_depth is None:
            print("警告: 录制 head_stereo.json 缺失，改用固化标定深度内参")
            session_depth = calib["depth_intrinsics"]
        print(f"回放: {session}（fps 步调 {args.replay_pace}）")
        align_calib = {"color_intrinsics": calib["color_intrinsics"],
                       "depth_to_color": calib["depth_to_color"],
                       "depth_intrinsics": session_depth}
    else:
        try:
            source = LiveD435Source(calib, exposure=args.exposure,
                                    serial=args.rs_serial)
        except RuntimeError as e:
            print(f"错误: {e}")
            sys.exit(1)
        align_calib = source.live_calib    # 本机实时内参/外参（换设备自洽）
    _run_3d_chain(args, source, align_calib)


def _shift_boxes_disparity(boxes, depth, fx, baseline, w, h):
    """左目平滑框按视差平移到右目坐标（手套模式右目共享框路径）。

    双目已极线行对齐（P0/P1），x_r = x_l − fx·B/z，z 取框内中位有效
    深度；无有效深度时原样返回（典型 1-2m 视差 20-40px，世界框有
    余量可容忍）。越界裁剪到 [0,w)×[0,h)，宽 <8px 的框丢弃。
    """
    out = []
    if depth is None or baseline <= 0:
        return [list(b) for b in boxes]
    dh, dw = depth.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = b
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(dw - 1, int(x2) + 1), min(dh - 1, int(y2) + 1)
        zs = depth[iy1:iy2 + 1, ix1:ix2 + 1]
        zs = zs[(zs > 0) & np.isfinite(zs)]
        d = 0.0
        if zs.size:
            d = fx * baseline / float(np.median(zs))
        x1 = max(0.0, min(x1 - d, w - 1.0))
        x2 = max(0.0, min(x2 - d, w - 1.0))
        if x2 - x1 < 8.0:
            continue
        out.append([x1, y1, x2, y2])
    return out


def _box_centers_moved(prev, cur, thresh=3.0):
    """两帧平滑框中心位移是否超阈值（右目手套 pose 运动门控，同
    HandTracker movement_thresh 口径）。prev=None/长度不同 → True。"""
    if prev is None or len(prev) != len(cur):
        return True
    for a, b in zip(prev, cur):
        if math.hypot((a[0] + a[2]) / 2.0 - (b[0] + b[2]) / 2.0,
                      (a[1] + a[3]) / 2.0 - (b[1] + b[3]) / 2.0) > thresh:
            return True
    return False


# 骨骼长度约束（S80C/S80M 深度噪声大时的异常骨长截断）：
# 以腕(0)为根的有向树，按父→子顺序逐关节钳制——子关节到父关节距离
# 超 max_bone_len 时沿原方向缩回球面，NaN 父/子不处理。正常手骨最长
# 腕→MCP ~0.12m，0.15m 上限只截深度噪声离群点，不扭曲真实手势。
# 父序构造为全部小于子索引（腕-掌边 0-17 不参与，17 父取 13）。
_BONE_PARENTS = (None, 0, 1, 2, 3,     # 拇指 1-4（0 腕→1 MCP→…→4 尖）
                 0, 5, 6, 7,           # 食指 5-8
                 5, 9, 10, 11,         # 中指 9-12
                 9, 13, 14, 15,        # 无名指 13-16
                 13, 17, 18, 19)       # 小指 17-20


def _clamp_bone_lengths(h3: np.ndarray, max_bone_len: float) -> np.ndarray:
    """层级骨长钳制（展示路径，不回改 tracker 状态）。"""
    out = np.array(h3, copy=True)
    for s in range(2):
        for j in range(21):
            pj = _BONE_PARENTS[j]
            if pj is None:
                continue
            p, c = out[s, pj], out[s, j]
            if not (np.isfinite(p).all() and np.isfinite(c).all()):
                continue
            d = c - p
            l = float(np.linalg.norm(d))
            if l > max_bone_len:
                out[s, j] = p + d * (max_bone_len / l)
    return out


_DISP_STALE_MAX = 10   # 异步模式：无新检测结果可复用显示缓存的最多帧数


class _DetWorker:
    """异步检测线程（det_async，S80C 默认）：latest-result 语义。

    主线程显示循环每帧 offer 最新 (rgb, rgb_r, n)，后台线程忙则丢旧帧
    （检测 ~23ms/帧是显示帧率的主瓶颈——解耦后显示走相机 50fps 全帧
    直推，与主程序 S80C 相机显示同口径）；完成后发布
    (hands, hands2d_r, det_ms, frame_n)，try_latest() 非阻塞取走。
    裸手专用（手套 CUDA 链走同步路径）。

    线程约束：det/det_r/voter/smo2d_right 全部只在本线程调用——
    HandLandmarker 是 VIDEO 模式、跟踪状态跨调用保留，单线程顺序
    调用是前提（左右目也必须各自实例，见链内 det_r 注释）。
    """

    def __init__(self, det, voter, det_r_factory, det_scale=1.0):
        self._det = det
        self._voter = voter
        self._det_r_factory = det_r_factory
        self._det_scale = det_scale
        self._det_r = None
        self._smo2d_right = _OneEuro2DSmoother(2)
        # Condition：worker 空等必须 wait()（释放锁休眠）。此前 Lock +
        # 持锁 sleep(0.001) 自旋，worker 释放后立刻重抢、主线程 offer
        # 长期被饿死 100-350ms（faulthandler 实测 26/39 次 dump 主线程
        # 卡 offer@with _lock——即实机"局部卡顿"真凶）。
        self._cond = threading.Condition()
        self._frame = None          # (rgb, rgb_r, n) 最新待检帧（忙则被覆盖）
        self._result = None         # (hands, hands2d_r, det_ms, frame_n)
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="det-worker")
        self._thread.start()

    def offer(self, rgb, rgb_r, n):
        """主线程每显示帧投喂最新帧；后台忙则丢旧帧（latest-frame 语义）。"""
        with self._cond:
            self._frame = (rgb, rgb_r, n)
            self._cond.notify()

    def try_latest(self):
        """非阻塞取最新完成结果；无新结果返回 None。"""
        with self._cond:
            r, self._result = self._result, None
        return r

    def stop(self):
        """停线程并释放右目检测器（det 由链的 finally 统一 close）。"""
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        if self._det_r is not None:
            try:
                self._det_r.close()
            except Exception:
                pass

    def _run(self):
        while not self._stop:
            try:
                self._detect_once()
            except Exception:
                # 线程内异常不杀整个 demo：打印后继续（0.5s 间隔防忙循环）
                traceback.print_exc(file=sys.stderr)
                time.sleep(0.5)

    def _detect_once(self):
        with self._cond:
            if self._frame is None:
                # 空等：wait 释放锁休眠直至 offer 唤醒（timeout 仅防
                # 漏 notify）。持锁 sleep 自旋会让主线程 offer 饿死。
                self._cond.wait(timeout=0.01)
                return
            rgb, rgb_r, n = self._frame
            self._frame = None
        t1 = time.perf_counter()
        if self._det_scale != 1.0:
            _dr = cv2.resize(
                rgb, (max(1, int(rgb.shape[1] * self._det_scale)),
                      max(1, int(rgb.shape[0] * self._det_scale))))
            hands = self._det.detect(_dr)
            for _hd in hands:
                _hd.landmarks = (np.asarray(_hd.landmarks, np.float32)
                                 / self._det_scale)
        else:
            hands = self._det.detect(rgb)
        if hands:
            self._voter.update(hands, frame_w=rgb.shape[1],
                               frame_h=rgb.shape[0], frame=n, cam="d435")
        # 右目独立检测器（与同步路径同款逻辑：半分辨率检测 + OneEuro）
        hands2d_r = []
        if rgb_r is not None:
            if self._det_r is None:
                self._det_r = self._det_r_factory()
            _now_ms = time.perf_counter() * 1000.0
            _rr = cv2.resize(rgb_r, (rgb_r.shape[1] // 2,
                                     rgb_r.shape[0] // 2))
            hands_r = self._det_r.detect(_rr)
            _rs = [rgb_r.shape[1] / _rr.shape[1],
                   rgb_r.shape[0] / _rr.shape[0]]
            for _i, _hd in enumerate(hands_r):
                _key = _hd.label if _hd.label not in ("", "Hand") \
                    else f"i{_i}"
                _pts = self._smo2d_right.update(
                    _i, np.asarray(_hd.landmarks, np.float32) * _rs,
                    _key, _now_ms)
                if np.isfinite(_pts).any():
                    hands2d_r.append(_pts)
            if not hands_r:
                for _i in range(2):
                    self._smo2d_right.reset(_i)
        with self._cond:
            self._result = (hands, hands2d_r,
                            time.perf_counter() - t1, n)


class _DispExtrap2D:
    """det_async 关键点显示外推（--extrap-2d，S80C 默认开）。

    异步检测的关键点来自 1-3 帧前图像，直接画在当前帧上=骨架落后于
    手（快动时肉眼可见 ~40-60ms 拖尾）。显示层按槽位把检测点整体平移
    到当前时刻：位移 = v × (t_now − t_obs)，v 由相邻两次检测框中心
    速度 EMA 估计；位移钳到框宽 0.5 与 120px 上限（防换手/误检飞点），
    >250ms 无新检测 v 归零（手停/丢手不漂）。仅平移、姿态形状仍是最新
    检测（形状滞后比位置滞后观感弱得多）。只作用于显示路径——3D 链/
    槽位/--export 全用原始检测值，不受污染。"""

    HORIZON = 0.25          # s：最长外推（超时 v 归零）
    MAX_SHIFT = 120.0       # px：单次外推位移硬顶

    def __init__(self):
        self._st = {}       # slot key → vx/vy/t_obs/t_prev/c_prev

    @staticmethod
    def _key(label, idx):
        return label if label not in ("", "Hand") else f"i{idx}"

    def observe(self, label, idx, pts, t_obs):
        """新检测到达：更新框中心速度 EMA 与观测时间戳。"""
        fin = pts[np.isfinite(pts).all(axis=1)]
        if len(fin) < 3:
            return
        c = ((fin.min(axis=0) + fin.max(axis=0)) * 0.5)   # bbox 中心
        k = self._key(label, idx)
        st = self._st.get(k)
        if st is None:
            st = {"vx": 0.0, "vy": 0.0, "t_prev": None, "c_prev": None}
        if st["t_prev"] is not None:
            dt = t_obs - st["t_prev"]
            if 0.005 < dt < 0.25:
                st["vx"] = 0.5 * (c[0] - st["c_prev"][0]) / dt \
                    + 0.5 * st["vx"]
                st["vy"] = 0.5 * (c[1] - st["c_prev"][1]) / dt \
                    + 0.5 * st["vy"]
        st["t_prev"], st["c_prev"] = t_obs, c
        st["t_obs"] = t_obs
        self._st[k] = st

    def apply(self, pts, label, idx, now):
        """按槽位速度把检测点平移到当前时刻。无观测历史则原样返回。"""
        st = self._st.get(self._key(label, idx))
        if st is None:
            return pts
        age = now - st["t_obs"]
        if age > self.HORIZON:
            st["vx"] = st["vy"] = 0.0
        age = min(age, self.HORIZON)
        dx, dy = st["vx"] * age, st["vy"] * age
        d = math.hypot(dx, dy)
        if d > self.MAX_SHIFT:
            dx *= self.MAX_SHIFT / d
            dy *= self.MAX_SHIFT / d
        out = np.array(pts, np.float32, copy=True)
        ok = np.isfinite(out).all(axis=1)
        out[ok] += (dx, dy)
        return out


def _write_tear_dump(ring, tdir):
    """把撕裂环写盘（后台线程调用；环帧只读，名含帧号）。

    JPEG Q90 而非 PNG：PNG 编码 CPU 高，后台线程会抢 XNNPACK 检测
    核（实测 det 15.5→24.4ms）；Q90 保留结构不糊缝，帧内水平缝
    （内容硬断）离线判别不受影响。"""
    try:
        # 每次导出先清空目标目录：退出 dump 固定写 tear_exit_000，
        # 不清的话多次运行帧号混叠，离线判侧会看错帧。
        import shutil as _shutil
        _shutil.rmtree(tdir, ignore_errors=True)
        os.makedirs(tdir, exist_ok=True)
        for _fn, _f in ring:
            _ok, _buf = cv2.imencode(
                ".jpg", _f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if _ok:
                with open(os.path.join(tdir, f"frame_{_fn:04d}.jpg"),
                          "wb") as _fo:
                    _fo.write(_buf.tobytes())
    except Exception as _e:                       # dump 失败不拖垮 demo
        print(f"  撕裂 dump 写盘失败: {_e}")


def _run_3d_chain(args, source, align_calib, win_title: str = "D435 live",
                  smooth3d_cfg=None, fixed_view3d: bool = False,
                  max_bone_len=None, det_async: bool = False,
                  win23_every: int = 1, extrap2d: bool = False,
                  tear_probe: bool = False):
    """3D 链主循环：对齐 → 检测（裸手/手套）→ 抬升 → 槽位 → 平滑 →
    三窗口渲染 + 可选 --export。

    win_title: 窗口名/渲染标题前缀（其他相机 demo 可传自定义名）。
    smooth3d_cfg: 3D 平滑参数覆盖 dict（可选）。键 freq_min/beta/dcutoff
        作用于逐点 OneEuro（Hand3DSmoother）；centroid_freq_min/
        centroid_beta/centroid_dcutoff 作用于 M1 质心锚定（centroid_*
        缺省时 freq_min/beta 跟随点级同名键、dcutoff 用质心自有默认）。
        None/缺省 = 本文件默认值（D435 深度口径），行为零变化——
    fixed_view3d: True 时 3D 视图用固定世界相机——首帧有手时锁存
        目标/缩放/网格（r 键重锁），之后相机完全静止、手在世界内自由
        运动；False（默认）= 既往行为（输入平移使质心恒定，相机目标
        不随手漂移，D435 口径零变化）。
    max_bone_len: 米制骨长上限（None=关，默认）。>0 时对 h3 做层级
        骨长钳制（噪声离群点缩回父关节球面），仅展示路径不回 tracker。
        D435 默认 None 行为零变化。
        深度更噪的源（S80C/S80M）由调用方传更低 freq_min 加强平滑。
    det_async: True 时裸手检测在后台线程跑（latest-result 语义）——
        显示循环每帧直推相机画面不等待检测（主程序 S80C 相机直推
        口径，显示 ~50fps、帧率不再被检测 ~23ms/帧 拖住）；关键点
        按检测速率更新，无新结果帧 2D 显示复用最近一次检测（防骨架
        闪没）、3D 槽位走传播。手套模式自动回同步路径（CUDA 检测器
        不进线程）。False（默认）= 既往同步行为（D435 口径零变化）。
    win23_every: win2 3D / win3 深度两辅助窗口每 N 帧 imshow 一次
        （N=1 = 逐帧，D435 默认行为零变化）。>1 时两窗口内容本身只
        以检测(~25fps)/深度(~20fps)速率变化，逐帧 X11 重传未变内容
        纯属浪费；省下的每帧时间让显示主循环回到相机 20ms 帧预算内
        （50fps 直推不丢帧），消除消费跟不上产帧导致的丢帧节拍跳动
        （即观感"卡顿"——平均 fps 40 但间隔 20/20/40ms 错拍）。
    extrap2d: True 时 2D 显示关键点做速度外推平移（_DispExtrap2D，
        仅 det_async + raw_2d 显示路径生效；S80C 默认开）。异步检测
        结果是 1-3 帧前图像的关键点，直接画在当前帧上=快动手时骨架
        落后于手；外推把显示点按框中心速度投影到当前时刻（仅平移，
        3D 链/槽位/--export 全用原始检测值）。False（默认）= D435
        口径零变化。
    tear_probe: True 时保存最近 96 帧（≈2.7s@35fps）内部显示缓冲
        （½ 尺寸），退出时自动导出 tear_exit_*（t 键手动导出保留）——
        看到撕裂按 q 退出即可，离线判别：显示侧撕裂只存在于屏幕合成
        输出（内部帧干净）；若导出帧内有水平缝则缝在数据/相机侧。
        False（默认）= 零开销零变化。
    """
    aligner = LiveAligner(align_calib["color_intrinsics"],
                              align_calib["depth_to_color"],
                          align_calib["depth_intrinsics"],
                          fill_passes=args.fill)
    color_intr = (aligner.fx_c, aligner.fy_c, aligner.cx_c, aligner.cy_c)

    delegate = _resolve_delegate(args.delegate)
    print(f"检测 delegate: {delegate}（--fill {args.fill} 轮填洞，"
          f"det/track conf {args.det_conf}/{args.track_conf}）")
    det = MediaPipeDetector(num_hands=2, delegate=delegate,
                            det_conf=args.det_conf,
                            track_conf=args.track_conf)
    # 右目独立检测器（惰性）：HandLandmarker 是 VIDEO 模式——跟踪状态
    # 跨调用保留（detect_for_video）。左右目共用同一实例时，右目半分辨率
    # 调用的跟踪裁剪框会继承左目的（坐标系/尺度混串），下一帧左目又继承
    # 右目的 → 双边关键点偏移+闪烁（2026-08-24 S80C 实测教训）。左右目
    # 必须各自实例、各自跟踪状态。D435 无右目流，零开销。
    det_r = None
    # 黑手套模式：YOLO-World 出框 + RTMPose（惰性——裸手启动零开销，
    # 未 --glove 时首次按 g 才加载，含一次性 ~1s CUDA 预热）
    glove_choice = args.glove_detector
    glove_weights = resolve_glove_weights(glove_choice, args.glove_weights)
    glove_det = None
    glove_mode = args.glove
    if glove_mode:
        try:
            glove_det = GloveDetector(weights=glove_weights,
                                      **_glove_kwargs(args))
            print(f"黑手套模式: {glove_weights}（{glove_det.backend} "
                  f"{glove_det.device}，pose {glove_det.pose_backend}/"
                  f"{glove_det.pose_device}，"
                  f"conf {glove_det.det_conf}；按 g 切回裸手，"
                  f"b 切姿态后端，v 换检测器）")
        except FileNotFoundError:
            sys.exit(f"错误: 手套权重不存在: {glove_weights}")
    voter = HandednessVoter()
    tracker = HandSlotTracker(max_lost=args.propagate_max)
    _sm3d = dict(smooth3d_cfg or {})
    smoother = Hand3DSmoother(freq_min=_sm3d.get("freq_min", 3.0),
                              beta=_sm3d.get("beta", 0.3),
                              dcutoff=_sm3d.get("dcutoff", 1.0))
    soft_smoother = _SoftSmoother(smoother)   # M3①：重建帧几何近时软衔接
    smo2d_left = _OneEuro2DSmoother(2)        # 左目 2D 平滑（按槽位身份）
    smo2d_right = _OneEuro2DSmoother(2)       # 右目 2D 平滑（按手身份，
                                              # 仅显示用）
    smo2d_left_glove = _OneEuro2DSmoother(2)  # 左目手套显示平滑（同右目
                                              # 口径：原始 pose + 轻平滑）
    renderer = RotatingSkeletonRenderer(*RENDER_SIZE, revolutions=1.0)

    win1 = f"{win_title}: RGB overlay"
    win2 = f"{win_title}: 3D view"
    win3 = f"{win_title}: depth"
    orbit = _OrbitControl()          # 手动视角（不再自动旋转）
    if not args.no_window:
        cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win3, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win1, 1024, 576)
        cv2.resizeWindow(win2, 1024, 576)
        cv2.resizeWindow(win3, 1024, 576)
        cv2.setMouseCallback(win2, orbit.on_mouse)
    depth_on = args.depth_overlay
    view_anchor = None   # 3D 视图锚点：首帧有手时锁定。fixed_view3d=True
                         # 时存 (质心, 距离, 网格 y)——固定世界相机，r 键
                         # 重锁；False（D435 默认）时存质心，输入平移使
                         # 相机目标恒定（既往行为，只平移不改手势）
    shot_n = 0
    _glove_r_boxes = None    # 右目手套 pose 运动门控：上次推理的平移后框
    _glove_r_skip = 0        # 连续未推理帧数（≥10 强制刷新，同 skip_timeout）
    _glove_r_disp = []       # 右目手套显示点：门控跳过帧复用（防骨架闪没）
    _glove_l_disp = [None, None]  # 左目手套显示点（槽位序）：本帧无原始
                                  # 点时复用上次平滑输出（防闪没）

    lost_counts = [0, 0]     # 各槽连续丢失帧数（assigner 困境槽无门限救援用）
    zc_slot = [None, None]   # M5：槽级补点深度先验（EMA，换手/首帧取实测）
    ws_prev = [None, None]   # M3②：wholesale 两帧确认——上一帧被门控观测
    ws_streak = [0, 0]       # M3②：连续 wholesale 帧数（≥3 强制采信防死锁）
    gate_streak = [np.zeros(21, np.int64), np.zeros(21, np.int64)]
                             # M6：逐点连续被门控帧数（≥_GATE_FORGIVE 采信观测）
    centroid_anchor = _CentroidAnchor(   # M1：质心强平滑 + 共模平移校正
        freq_min=_sm3d.get("centroid_freq_min", _sm3d.get("freq_min", 3.0)),
        beta=_sm3d.get("centroid_beta", _sm3d.get("beta", 0.3)),
        dcutoff=_sm3d.get("centroid_dcutoff", 0.3))
    stats = {"n": 0, "det0": 0, "det1": 0, "det2": 0, "same_label": 0,
             "glove_n": 0, "glove_box": 0,
             "wholesale": 0, "ws_skip": 0, "gate_pts": 0,
             "forgive_pts": 0}      # --stats 诊断计数
    diag_cens = []     # --stats 静止段诊断：(h3, smoothed, renderer_in) 序列
    export_dir = args.export     # --export：parquet + 两个 mp4 落盘
    exp2d_rows: list = []        # (frame, slot, label, state, x0..x20, y0..y20)
    exp3d_rows: list = []
    exp_sink = None              # 3D 旋转渲染 sink（首帧尺寸确定后惰性建）
    exp_sink2d = None            # 原视频叠 2D 关键点 sink
    n = 0
    n_det = 0                 # 异步模式实际检测帧数（det 均耗时口径）
    det_worker = None         # 异步检测线程（det_async；手套模式暂停）
    _hands_disp = []          # 2D 显示缓存：最近一次检测（stale 帧复用）
    _extrap2d = _DispExtrap2D() if extrap2d else None
    # 撕裂诊断：内部帧环（96 帧 ≈ 2.7s@35fps）+ 退出时自动导出
    # （免掐时机按 t）——离线判别相机侧/显示侧。持续录制（每 100 帧
    # 滚动落盘）用户已叫停：后台编码抢 CPU，且退出 dump 已够判侧。
    _tear_ring = collections.deque(maxlen=96 if tear_probe else 1)
    if tear_probe:
        print("  撕裂自动捕获已开：看到画面撕裂直接按 q 退出即可，"
              "最近 ~2.7s 内部帧自动导出 keypoints_output/ 下 "
              "tear_exit_*（t 键可随时手动导出，不持续写盘）")
    _hands2d_r_disp = []      # 右目 2D 显示缓存（同上）
    _disp_age = 0             # 距最近检测结果的显示帧数（超限清缓存防幽灵）
    t_det = t_align = 0.0
    t0 = time.perf_counter()
    fps_win_t, fps_win_n = t0, 0
    fps = 0.0
    # 帧间隔探针（HAND3D_WIN_PROF=1 环境变量门控，默认零开销零行为
    # 变化）：每 2s 打印显示循环实际帧间隔分布——"卡顿"的本质是间隔
    # 不均（丢帧节拍跳动），平均 fps 看不出，用 p50/p95/max 诊断。
    _prof = os.environ.get("HAND3D_WIN_PROF") == "1"
    _prof_ivals = []
    _prof_works = []
    _prof_secs = {"align": [], "det": [], "chain": [],
                  "rot": [], "ov": [], "dimg": [], "show": []}
    _prof_prev = None
    _prof_w1 = None
    _prof_p1 = None
    _prof_t = t0
    try:
        while True:
            if _prof:
                _now = time.perf_counter()
                if _prof_prev is not None:
                    _prof_ivals.append((_now - _prof_prev) * 1000.0)
                _prof_prev = _now
                if _now - _prof_t >= 2.0:
                    _prof_t = _now
                    if _prof_ivals:
                        _sec = " ".join(
                            f"{k}max={max(v):.1f}" for k, v in _prof_secs.items() if v)
                        print(f"[win-prof] n={n} 最近2s {len(_prof_ivals) / 2.0:.1f} fps "
                              f"帧间隔 p50={np.percentile(_prof_ivals, 50):.1f} "
                              f"p95={np.percentile(_prof_ivals, 95):.1f} "
                              f"max={max(_prof_ivals):.1f} ms "
                              f"work p50={np.percentile(_prof_works, 50):.1f} "
                              f"p95={np.percentile(_prof_works, 95):.1f} "
                              f"max={max(_prof_works):.1f} ms | {_sec}", flush=True)
                    _prof_ivals = []
                    _prof_works = []
                    for _v in _prof_secs.values():
                        _v.clear()
            rgb, d = source.next()
            if _prof:
                _prof_w1 = time.perf_counter()
                # 看门狗：本帧工作 >80ms 时 dump 全线程栈（诊断停顿点）
                faulthandler.dump_traceback_later(0.08, exit=False)
            if rgb is None:
                break

            # 右目帧（S80C --stereo-view；source 无 right_frame → None，
            # 链行为与 D435 完全一致）。右目仅显示用：独立检测渲染，
            # 不入槽位/3D。
            _rf_fn = getattr(source, "right_frame", None)
            rgb_r = _rf_fn() if _rf_fn is not None else None
            if rgb_r is not None and rgb_r.shape[:2] != rgb.shape[:2]:
                rgb_r = None

            t1 = time.perf_counter()
            if d is None or d.shape[:2] != (aligner.dh, aligner.dw):
                aligned = np.zeros((aligner.ch, aligner.cw), np.float32)
            else:
                aligned = aligner.align_depth_to_color(d)
            t_align += time.perf_counter() - t1
            if _prof:
                _prof_secs["align"].append(
                    (time.perf_counter() - t1) * 1000.0)

            # ── 检测段 ──
            # det_async（S80C 默认）：检测在后台线程跑 latest-result 语义，
            # 显示循环全帧直推不等待（主程序 S80C 相机直推口径）；无新
            # 结果帧 hands=[] 走槽位传播，2D 显示复用最近一次检测防闪没。
            # 手套模式自动回同步路径（CUDA 检测器不进线程）。
            # det_async=False（D435 默认）：下方同步检测代码原样，行为零变化。
            if _prof:
                _prof_p1 = time.perf_counter()
            if det_async and not glove_mode:
                if det_worker is None:
                    det_worker = _DetWorker(
                        det, voter,
                        lambda: MediaPipeDetector(
                            num_hands=2, delegate=delegate,
                            det_conf=args.det_conf,
                            track_conf=args.track_conf),
                        det_scale=args.det_scale)
                det_worker.offer(rgb, rgb_r, n)
                _res = det_worker.try_latest()
                if _res is not None:
                    hands, hands2d_r, _det_ms = _res[:3]
                    stats["n"] += 1
                    stats[f"det{min(len(hands), 2)}"] += 1
                    t_det += _det_ms
                    n_det += 1
                    # 2D 显示缓存：无新结果帧复用；检测到空手即清
                    # （不画幽灵骨架）
                    _hands_disp = list(hands)
                    _hands2d_r_disp = list(hands2d_r)
                    _disp_age = 0
                    # 显示外推观测：记录本结果的检测时刻与框中心速度
                    # （det_ms 结束于发布时刻，t_obs ≈ 检测开始时刻）
                    if _extrap2d is not None:
                        _t_obs = time.perf_counter() - _det_ms
                        for _i, _hd in enumerate(hands[:2]):
                            _extrap2d.observe(
                                _hd.label, _i,
                                np.asarray(_hd.landmarks,
                                           np.float32).reshape(21, 2),
                                _t_obs)
                else:
                    hands = []
                    hands2d_r = _hands2d_r_disp
                    _disp_age += 1
                    if _disp_age > _DISP_STALE_MAX:
                        _hands_disp = []
                        _hands2d_r_disp = []
            else:
                t1 = time.perf_counter()
                # --det-scale：裸手检测在缩小图上跑（CPU XNNPACK 下省一半
                # 时间），landmark 按比例回全分辨率坐标；手套路径用自己的
                # 裁剪框，不缩放。
                if not glove_mode and args.det_scale != 1.0:
                    _dr = cv2.resize(
                        rgb, (max(1, int(rgb.shape[1] * args.det_scale)),
                              max(1, int(rgb.shape[0] * args.det_scale))))
                    hands = det.detect(_dr)
                    for _hd in hands:
                        _hd.landmarks = (np.asarray(_hd.landmarks, np.float32)
                                         / args.det_scale)
                else:
                    hands = (glove_det if glove_mode else det).detect(rgb)
                stats["n"] += 1
                stats[f"det{min(len(hands), 2)}"] += 1
                if glove_mode:
                    stats["glove_n"] += 1
                    stats["glove_box"] += 1 if glove_det.last_boxes else 0
                # 空帧不喂 voter：identity.py 空帧会清空轨迹（scene reset 语义），
                # 短暂漏检会清票仓 → 重建期原始 label 闪烁 → 两手同 label
                # （离线 222_000011 f372-373 实踩）。跳过空帧让轨迹 idle 保持。
                # 手套模式整体跳过 voter：identity.py 新轨迹替换路径有同帧
                # 双分配缺陷（两手同时超关联门限 → 共用一个票仓 → 同 label
                # 锁死；Project_Test10 f79 起实测 358/508 帧，探针+逻辑回放
                # 双重实证），且 voter 前提是 MediaPipe 稳定手性；手套侧身份
                # 由 GloveDetector per-track 锁存票仓承担，同 label 守卫+几何
                # 分配兜底。裸手路径不变。
                if hands and not glove_mode:
                    voter.update(hands, frame_w=rgb.shape[1],
                                 frame_h=rgb.shape[0], frame=n, cam="d435")
                t_det += time.perf_counter() - t1
            # 2D 显示数据源：异步模式 stale 帧用检测缓存（显示与 3D 槽位
            # 链解耦）；同步模式即当前帧检测（原行为）
            if _prof and _prof_p1 is not None:
                _prof_secs["det"].append(
                    (time.perf_counter() - _prof_p1) * 1000.0)
                _prof_p1 = time.perf_counter()
            hands_disp_src = _hands_disp if (det_async and not glove_mode) \
                else hands

            # 左目手套显示点：与右目同口径——原始 pose + 轻平滑。
            # detect() 稳定层输出（逐点置信加权 + 随框平移/持出）在
            # S80C 上会随框漂移/持旧点变形（左目比右目差的根因）；
            # last_raw_pose() 是 detect() 本帧推理的原始点（含低置信
            # 持出帧），零额外推理。3D 槽位链仍用稳定层 hands 不变。
            # 门控跳过/退化帧（条目为 None）复用上次平滑输出——同右目
            # _glove_r_disp 语义：tracker 框位移 <3px 门控推理（10 帧
            # 强刷），手静止/慢动时多数帧无新原始点，清空会让骨架每
            # 几帧消失一次（左目闪烁根因）；track 消失（槽位超出
            # _raw_l 长度）才清空 + reset。
            if glove_mode and glove_det is not None and args.raw_2d:
                _now_l = time.perf_counter() * 1000.0
                _raw_l = glove_det.last_raw_pose()
                for _i in range(2):
                    if _i < len(_raw_l) and _raw_l[_i] is not None:
                        _sm = smo2d_left_glove.update(
                            _i, _raw_l[_i], f"i{_i}", _now_l)
                        if np.isfinite(_sm).any():
                            _glove_l_disp[_i] = _sm
                    elif _i >= len(_raw_l):
                        _glove_l_disp[_i] = None
                        smo2d_left_glove.reset(_i)

            # 右目独立检测 + 2D 平滑（仅显示，不入 voter/槽位/3D）。
            # 裸手：独立 MediaPipe 实例半分辨率检测——HandLandmarker 是
            # VIDEO 模式，跨流共享实例会污染跟踪状态（det_r 教训）。
            # 手套：MediaPipe 对黑手套是死路（实测 4/68），改共享左目
            # GloveDetector 的平滑框——同场景双目、极线行对齐，框按视差
            # 平移（x_r = x_l − fx·B/z，框内中位深度）到右目坐标后，用
            # 同一 pose 后端在右帧裁剪推理（RTMPose stateless、mediapipe
            # 后端 IMAGE 模式均无跨调用状态，共享实例安全；b 键热切换
            # 后端自动跟随）。右目 pose 按框位移门控（同 tracker 口径：
            # 中心位移 <3px 不推理、10 帧强制刷新）——静止段省推理且
            # smo2d_right 保持收敛不抖。
            hands_r = []
            if not (det_async and not glove_mode):
                hands2d_r = []   # 异步路径已由检测结果/缓存设定
            if rgb_r is not None and not (det_async and not glove_mode):
                _now_ms = time.perf_counter() * 1000.0
                t1 = time.perf_counter()
                if glove_mode and glove_det is not None:
                    _boxes_r = _shift_boxes_disparity(
                        glove_det.track_boxes(), aligned, color_intr[0],
                        getattr(source, "baseline_mm", 0.0),
                        rgb_r.shape[1], rgb_r.shape[0])
                    if not _boxes_r:
                        # 左目无活跃框 → 右半边同步清空
                        for _i in range(2):
                            smo2d_right.reset(_i)
                        _glove_r_boxes = []
                        _glove_r_skip = 0
                        t_det += time.perf_counter() - t1
                    else:
                        _moved = _box_centers_moved(_glove_r_boxes, _boxes_r)
                        _glove_r_skip = 0 if _moved else _glove_r_skip + 1
                        _pts_r = None   # 门控跳过帧不更新平滑器
                        if _moved or _glove_r_skip >= 10:
                            _glove_r_skip = 0
                            _glove_r_boxes = _boxes_r
                            _pts_r = glove_det.pose_on_boxes(
                                rgb_r, _boxes_r, keep_degenerate=True)
                        t_det += time.perf_counter() - t1
                        if _pts_r is not None:
                            _glove_r_disp = []
                            for _i, _pts in enumerate(_pts_r):
                                _sm = smo2d_right.update(
                                    _i, _pts, f"i{_i}", _now_ms)
                                if np.isfinite(_sm).any():
                                    _glove_r_disp.append(_sm)
                            if not _pts_r:
                                for _i in range(2):
                                    smo2d_right.reset(_i)
                        hands2d_r = _glove_r_disp   # 门控跳过帧复用上次平滑输出
                else:
                    if det_r is None:
                        det_r = MediaPipeDetector(num_hands=2,
                                                  delegate=delegate,
                                                  det_conf=args.det_conf,
                                                  track_conf=args.track_conf)
                    _rr = cv2.resize(rgb_r, (rgb_r.shape[1] // 2,
                                             rgb_r.shape[0] // 2))
                    hands_r = det_r.detect(_rr)
                    t_det += time.perf_counter() - t1
                    _rs = [rgb_r.shape[1] / _rr.shape[1],
                           rgb_r.shape[0] / _rr.shape[0]]
                    for _i, _hd in enumerate(hands_r):
                        _key = _hd.label if _hd.label not in ("", "Hand") \
                            else f"i{_i}"
                        _pts = smo2d_right.update(
                            _i, np.asarray(_hd.landmarks, np.float32) * _rs,
                            _key, _now_ms)
                        if np.isfinite(_pts).any():
                            hands2d_r.append(_pts)
                    if not hands_r:
                        for _i in range(2):
                            smo2d_right.reset(_i)

            pairs = [lift_hand(hd, aligner, aligned) for hd in hands]
            # voter 重建期两手同 label（离线 222 f372 教训）：label 不可信，
            # 直接按 label 分配会有一手被标签守卫拒收（"经常少一只手"的
            # 分配层来源之一）。同 label 时先清空 label 走几何分配，观察
            # 时用槽自身 label（防 observe_slot 误判换手重置槽位）。
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
                stats["same_label"] += 1

            slot_pairs, slot_dets, states = [], [], []
            for s in range(2):
                if out[s] is not None:
                    p = out[s]
                    # M5：补点深度锚定到槽级稳定 zc。zc 逐帧独立中位是整手
                    # 共模跳的最大来源（有效点集合一变，全部补点+整手质心
                    # 共模突跳）——槽级 EMA 吸收之；实测点不动，补点 x,y
                    # 随 zc 反投影保持与 2D 一致（"保持 z、调 x,y"经验）。
                    # 换手（观察 label ≠ 槽 label）取实测，防跨手污染。
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
                    # （tracker 对 NaN 点保持纯预测，翻面观测不入状态）
                    n_fin_before = int(np.isfinite(
                        np.asarray(p.result.points_3d, np.float64)
                        .reshape(-1, 3)).all(axis=1).sum())
                    gated, wholesale = gate_observations(
                        p.result.points_3d, tracker.predict(s, n))
                    stats["gate_pts"] += n_fin_before - int(np.isfinite(
                        np.asarray(gated, np.float64)
                        .reshape(-1, 3)).all(axis=1).sum())
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
                            stats["forgive_pts"] += int(forgive.sum())
                    if wholesale:
                        stats["wholesale"] += 1
                        # M3②：整手级不匹配先两帧确认。连续两帧观测互相
                        # 一致才判槽状态过时 → 借 label 翻转触发槽位重置、
                        # 随即真观测干净初始化（否则假检测钉背景死锁）；
                        # 单帧跳变/误检不采信不重置——本帧走预测显示，
                        # 状态不动（此前单帧即重置 → 显示瞬间贴到错误
                        # 观测、下一帧又跳回 = 整手跳的来源之一）。
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
                            stats["ws_skip"] += 1
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

            presents = [st != "absent" for st in states]
            propagated = [st == "propagated" for st in states]
            labels = [slot_pairs[s].left_label if out[s] is not None
                      else tracker.slot_label(s) for s in range(2)]

            # (2,21,3) 槽位 3D（tracker αβ 已平滑）→ OneEuro 再平滑压静止抖动
            # （M3① _SoftSmoother 包装：重建帧几何近时 0.5 混合软衔接，
            #   防 pop 滤波器后首帧输出=原始输入的 snap）
            h3 = np.stack([np.asarray(p.result.points_3d, np.float64)
                           .reshape(21, 3) for p in slot_pairs])
            # 骨长钳制（max_bone_len>0 时，S80C 默认 0.15m）：噪声离群
            # 关节缩回父关节球面，下游平滑/质心锚定不再被荒诞骨长毒化
            if max_bone_len is not None and max_bone_len > 0:
                h3 = _clamp_bone_lengths(h3, max_bone_len)
            valids = [int(np.isfinite(h3[s]).all(axis=1).sum())
                      for s in range(2)]
            smoothed = soft_smoother.update(h3, labels, valids)
            # M1 质心锚定（替代原 EMA）：质心强 OneEuro + 共模平移校正——
            # 逐点独立平滑下整手平移不受约束，质心每帧微跳；把质心单独
            # 强平滑、手内形状原样平移，整手共模跳被质心层吸收（仅展示
            # 路径，不回流 tracker）。
            renderer_in = centroid_anchor.apply(smoothed, labels)
            if args.stats:
                diag_cens.append((h3, smoothed, renderer_in))

            # 2D：real 帧画检测骨架；propagated/absent 传 NaN 不画
            hands2d = np.stack([
                np.asarray(sd.landmarks, np.float32).reshape(21, 2)
                if sd is not None else np.full((21, 2), np.nan)
                for sd in slot_dets])
            # --smooth-2d：逐点 OneEuro 压检测抖动（默认关，见
            # _OneEuro2DSmoother 滞后教训）。按槽位身份键管理（换手
            # label 变化自动重置）；absent 重置（重回时冷启动防陈旧状态
            # 跨缺口强拉），propagated 不更新保持。raw-2d 显示不画
            # slot 点，平滑无显示效果，跳过。
            if args.smooth_2d and not args.raw_2d:
                _now_ms = time.perf_counter() * 1000.0
                for s in range(2):
                    if states[s] == "absent":
                        smo2d_left.reset(s)
                    elif np.isfinite(hands2d[s]).any():
                        hands2d[s] = smo2d_left.update(
                            s, hands2d[s], labels[s], _now_ms)

            # --raw-2d（S80C 默认开）：2D 显示直接用当前帧原始检测，
            # 与 3D 槽位链彻底解耦——槽位 propagated/absent 时骨架消失
            # （闪烁）、wholesale 门控拒收时画不上（手在动骨架停着=偏移
            # 观感）。显示路径不应受 3D 时序门/深度稀疏影响；3D 窗口、
            # 槽位状态与 --export 仍走槽位链不变（D435 默认关，行为
            # 与既往完全一致）。裸手/手套模式同样生效：手套模式左目
            # 画与右目同口径的原始 pose（_glove_l_disp，见左目手套块
            # ——detect() 稳定层输出在 S80C 上随框漂移/持旧点变形），
            # 3D 槽位链仍用稳定层 hands 不变。
            hands2d_disp = hands2d
            labels_disp = list(labels)
            if args.raw_2d:
                hands2d_disp = np.full((2, 21, 2), np.nan, np.float32)
                labels_disp = ["", ""]
                if glove_mode and glove_det is not None:
                    for _i in range(2):
                        if _glove_l_disp[_i] is not None:
                            hands2d_disp[_i] = _glove_l_disp[_i]
                            if _i < len(hands):
                                labels_disp[_i] = hands[_i].label
                else:
                    for _i in range(min(len(hands_disp_src), 2)):
                        hands2d_disp[_i] = np.asarray(
                            hands_disp_src[_i].landmarks, np.float32).reshape(21, 2)
                        labels_disp[_i] = hands_disp_src[_i].label

            # --extrap-2d：把显示点按框中心速度外推平移到当前时刻
            # （仅 det_async + raw_2d 显示路径；见 _DispExtrap2D）。
            # 3D 链/槽位/--export 的 hands2d/smoothed 全用原始检测值。
            if (_extrap2d is not None and det_async and not glove_mode
                    and args.raw_2d):
                _now = time.perf_counter()
                for _i in range(2):
                    _p = hands2d_disp[_i]
                    if np.isfinite(_p).any():
                        hands2d_disp[_i] = _extrap2d.apply(
                            _p, labels_disp[_i], _i, _now)

            # --export：逐帧 2D（槽位像素）与 3D（质心锚定相机系 mm，视图
            # 平移前的原始值）累积；NaN=无观测。窗口/无窗口同路径。
            if export_dir is not None:
                for s in range(2):
                    exp2d_rows.append([n, s, labels[s], states[s]] + [
                        float(v) if np.isfinite(v) else float("nan")
                        for v in hands2d[s].flatten()])
                    exp3d_rows.append([n, s, labels[s], states[s]] + [
                        float(v) if np.isfinite(v) else float("nan")
                        for v in np.asarray(renderer_in[s],
                                            np.float64).flatten()])

            # 旋转渲染帧：窗口显示与 --export mp4 共用（无窗口纯导出也要出图）
            if _prof and _prof_p1 is not None:
                _prof_secs["chain"].append(
                    (time.perf_counter() - _prof_p1) * 1000.0)
                _prof_p1 = None
            if not args.no_window or export_dir is not None:
                # 视角锚定：
                #  fixed_view3d=True（S80C 默认）：世界固定相机——首帧
                #  有手时锁存目标/缩放/网格（view_params），之后相机完全
                #  静止、手在世界内自由运动（走渲染器 fixed_view 参数，
                #  默认 None 时渲染器行为与既往一致）；r 键重锁。
                #  fixed_view3d=False（D435 默认）：把输入整体平移到世界
                #  锚点 view_anchor 使质心恒定（相机目标不变），与既往
                #  行为完全一致。
                if fixed_view3d:
                    if view_anchor is None:
                        view_anchor = renderer.view_params(renderer_in)
                    view_in = renderer_in
                    fixed_vp = view_anchor
                else:
                    if view_anchor is None:
                        fin_all = np.isfinite(renderer_in).all(axis=2)
                        if fin_all.sum() >= 4:
                            view_anchor = renderer_in[fin_all].mean(axis=0)
                    view_in = renderer_in
                    if view_anchor is not None:
                        fin_all = np.isfinite(view_in).all(axis=2)
                        if fin_all.sum() >= 4:
                            view_in = view_in - (
                                view_in[fin_all].mean(axis=0) - view_anchor)
                    fixed_vp = None
                # 手动视角：yaw 反解为 frame_idx（相机真轨道，网格/手固定），
                # 俯仰经构造参数注入（运行时改实例属性，不动渲染器文件）
                renderer.elevation = math.radians(orbit.elev)
                if _prof:
                    _prof_p1 = time.perf_counter()
                rot = renderer.render(view_in, labels, (np.nan, np.nan),
                                      orbit.frame_idx(_ROT_TOTAL), _ROT_TOTAL,
                                      f"{win_title} hand keypoints "
                                      "(color-cam, m)", fixed_view=fixed_vp)
                if _prof and _prof_p1 is not None:
                    _prof_secs["rot"].append(
                        (time.perf_counter() - _prof_p1) * 1000.0)
                if export_dir is not None:
                    # 2D 叠加帧：原视频 + 2D 关键点骨架（导出用，拷贝
                    # 防污染窗口分支的 rgb；不含深度叠层/交互 HUD）
                    ov_exp = draw_overlay(rgb.copy(), hands2d, smoothed,
                                          labels, propagated, presents,
                                          n, n + 1,
                                          f"{win_title} 2D hands"
                                          + (" [GLOVE]" if glove_mode else ""))
                    if glove_mode and glove_det is not None:
                        for bx1, by1, bx2, by2, bconf in glove_det.last_boxes:
                            cv2.rectangle(ov_exp, (int(bx1), int(by1)),
                                          (int(bx2), int(by2)), (0, 255, 0), 2)
                            cv2.putText(ov_exp, f"{bconf:.2f}",
                                        (int(bx1), max(16, int(by1) - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 1, cv2.LINE_AA)
                    if exp_sink is None:
                        # 共享 sink（stereo_s80m.hand_3d.video_writer）：
                        # nvenc→libx264 管道→mp4v 两段式逐级回退，H.264+
                        # faststart 直出。venv cv2 只能写 mp4v，用户播放器
                        # 打不开（系统录像是 H.264），不能用 cv2.VideoWriter。
                        os.makedirs(export_dir, exist_ok=True)
                        exp_sink = create_video_sink(
                            os.path.join(export_dir, "render.mp4"),
                            30, rot.shape[1], rot.shape[0])
                        exp_sink2d = create_video_sink(
                            os.path.join(export_dir, "rgb_overlay.mp4"),
                            30, ov_exp.shape[1], ov_exp.shape[0])
                    exp_sink.write(np.ascontiguousarray(rot))
                    exp_sink2d.write(np.ascontiguousarray(ov_exp))

            if not args.no_window:
                # 显示基底：source 可选提供显示帧（S80C --stereo-view
                # 左右并排——右目仅显示用，检测/深度/3D 仍基于 rgb 左
                # 目帧）。D435 source 无此方法 → disp=rgb 行为不变。
                if _prof:
                    _prof_p1 = time.perf_counter()
                _disp_fn = getattr(source, "display_frame", None)
                disp = _disp_fn(rgb) if _disp_fn is not None else rgb
                if tear_probe:
                    # 撕裂诊断环：保存最近 ~2.7s 内部显示缓冲（½ 尺寸）。
                    # 显示侧撕裂只存在于屏幕合成输出，内部帧干净；导出
                    # 帧里有水平缝=缝在数据/相机侧。看到撕裂直接按 q
                    # 退出即可（无需掐时机）——退出时自动导出 tear_exit_*，
                    # t 键可随时手动导出（不持续写盘，用户 2026-08-25 已
                    # 叫停每 100 帧滚动落盘：后台编码抢 CPU）
                    # （t 键手动导出保留）。
                    _tear_ring.append((
                        n, cv2.resize(disp, (disp.shape[1] // 2,
                                             disp.shape[0] // 2))))
                if depth_on:
                    if disp.shape[:2] == aligned.shape[:2]:
                        base = blend_depth(disp, aligned, 0.35)
                    else:
                        # 并排视图：深度叠层只画左半（aligned 尺寸）
                        base = disp.copy()
                        _hw = aligned.shape[1]
                        base[:, :_hw] = blend_depth(
                            disp[:, :_hw], aligned, 0.35)
                else:
                    base = disp
                if args.raw_2d:
                    # 深度标注取槽位 3D（仅 real 帧可信；否则 NaN 不标）。
                    _h3 = np.array(smoothed, dtype=np.float64, copy=True)
                    for _s in range(2):
                        if states[_s] != "real":
                            _h3[_s] = np.nan
                    ov = draw_overlay(base, hands2d_disp, _h3, labels_disp,
                                      [False, False],
                                      [np.isfinite(hands2d_disp[_s]).any()
                                       for _s in range(2)],
                                      n, n + 1,
                                      f"{win_title} 3D hands")
                else:
                    ov = draw_overlay(base, hands2d, smoothed, labels,
                                      propagated, presents, n, n + 1,
                                      f"{win_title} 3D hands"
                                      + (" [GLOVE]" if glove_mode else ""))
                # 右目 2D 关键点（S80C --stereo-view；并排视图右半边）。
                # 仅在显示基底确有右半时画；深度标注传 NaN——右目空间无
                # 深度，_draw_hand 只画骨架不标深度数字。
                if rgb_r is not None and hands2d_r \
                        and disp.shape[1] > rgb.shape[1]:
                    _x_off = float(disp.shape[1] - rgb.shape[1])
                    for _pts in hands2d_r:
                        _draw_hand(ov, _pts + [_x_off, 0.0],
                                   np.full(21, np.nan))
                if glove_mode and glove_det is not None:
                    for bx1, by1, bx2, by2, bconf in glove_det.last_boxes:
                        cv2.rectangle(ov, (int(bx1), int(by1)),
                                      (int(bx2), int(by2)), (0, 255, 0), 2)
                        cv2.putText(ov, f"{bconf:.2f}",
                                    (int(bx1), max(16, int(by1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                # 深度图实时显示：aligned 深度（mm）→ 12-bit 码值 JET
                # （与主程序显示/存储文件同构）
                if _prof and _prof_p1 is not None:
                    _prof_secs["ov"].append(
                        (time.perf_counter() - _prof_p1) * 1000.0)
                    _prof_p1 = time.perf_counter()
                dimg = depth_to_heatmap_bgr(aligned)

                # 并排视图 imshow 显示副本降半：2560×800 直接走 X11
                # 传输 ~6MB/帧是窗口模式帧率主瓶颈之一（窗口本身仅
                # 1024×576，屏幕有效分辨率不变）；ov/dimg 保留原尺寸
                # 供截图。D435（disp==rgb）不触发，行为零变化。
                if disp.shape[1] > rgb.shape[1]:
                    ov_disp = cv2.resize(ov, (ov.shape[1] // 2,
                                              ov.shape[0] // 2))
                    dimg_disp = cv2.resize(dimg, (dimg.shape[1] // 2,
                                                  dimg.shape[0] // 2))
                else:
                    ov_disp, dimg_disp = ov, dimg
                if _prof and _prof_p1 is not None:
                    _prof_secs["dimg"].append(
                        (time.perf_counter() - _prof_p1) * 1000.0)
                    _prof_p1 = time.perf_counter()

                now = time.perf_counter()
                fps_win_n += 1
                if now - fps_win_t >= 1.0:
                    fps = fps_win_n / (now - fps_win_t)
                    fps_win_t, fps_win_n = now, 0
                _hint = ("[q]uit [s]hot"
                         + (" [t]ear-dump" if tear_probe else "")
                         + " [d]epth [g]love [b]ackend [r]eset view")
                for img in (ov_disp, rot, dimg_disp):
                    cv2.putText(img, f"{fps:5.1f} fps  {_hint}",
                                (12, img.shape[0] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (80, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(dimg_disp, "aligned depth 0.3-1.5 m",
                            (12, dimg_disp.shape[0] - 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (80, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(rot, "drag: orbit   r: reset view"
                            + (" + relock" if fixed_view3d else ""),
                            (12, rot.shape[0] - 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (80, 220, 255), 1, cv2.LINE_AA)
                cv2.imshow(win1, ov_disp)
                # win23_every>1：win2/win3 降频更新（见签名 docstring）。
                # 默认 1=D435 逐帧行为零变化。
                if n % win23_every == 0:
                    cv2.imshow(win2, rot)
                    cv2.imshow(win3, dimg_disp)
                key = cv2.waitKey(1) & 0xFF
                if _prof and _prof_p1 is not None:
                    _prof_secs["show"].append(
                        (time.perf_counter() - _prof_p1) * 1000.0)
                    _prof_p1 = None
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    orbit.reset()
                    if fixed_view3d:
                        view_anchor = None   # 固定视角重锁：下一帧有手时
                                             # 在当前手位置重建世界锚点
                if key == ord("s"):
                    os.makedirs(_SHOT_DIR, exist_ok=True)
                    p1 = os.path.join(_SHOT_DIR, f"shot_{shot_n:03d}_overlay.png")
                    p2 = os.path.join(_SHOT_DIR, f"shot_{shot_n:03d}_rot.png")
                    p3 = os.path.join(_SHOT_DIR, f"shot_{shot_n:03d}_depth.png")
                    cv2.imwrite(p1, ov)
                    cv2.imwrite(p2, rot)
                    cv2.imwrite(p3, dimg)
                    print(f"  截图: {p1} / {p2} / {p3}")
                    shot_n += 1
                if key == ord("t") and tear_probe:
                    # 手动导出当前环（自动检测/退出自动导出已覆盖，
                    # t 仅作补充）。显示侧撕裂只存在于屏幕合成输出——
                    # dump 帧全干净即显示侧；帧内有水平缝即数据/相机侧
                    # （缝两侧内容水平错位、位置不定）。
                    _tdir = os.path.join(_SHOT_DIR,
                                         f"tear_dump_{shot_n:03d}")
                    _write_tear_dump(list(_tear_ring), _tdir)
                    print(f"  撕裂 dump: {_tdir} ({len(_tear_ring)} 帧 "
                          f"内部缓冲)")
                if key == ord("d"):
                    depth_on = not depth_on
                if key == ord("b"):
                    # 姿态后端热切换：只换 GloveDetector._pose（tracker/
                    # OneEuro/手性锁存状态保留）；非手套模式仅提示
                    if glove_mode and glove_det is not None:
                        new_backend = ("mediapipe"
                                       if glove_det.pose_backend == "rtmpose"
                                       else "rtmpose")
                        glove_det.set_pose_backend(new_backend)
                        print(f"[b] 姿态后端 → {new_backend}"
                              f"（{glove_det.pose_device}，"
                              f"track/滤波状态保留）")
                    else:
                        print("[b] 仅黑手套模式可用（按 g 进入）")
                if key == ord("v"):
                    # 检测器热切换：world ↔ det 重建 GloveDetector
                    # （track/OneEuro/手性锁存重置——同 g 键语义）
                    if args.glove_weights:
                        print("[v] --glove-weights 已显式指定权重，"
                              "v 键不生效")
                    elif glove_mode:
                        new_choice = ("det" if glove_choice == "world"
                                      else "world")
                        new_weights = resolve_glove_weights(
                            new_choice, args.glove_weights)
                        try:
                            _gd = GloveDetector(weights=new_weights,
                                                **_glove_kwargs(args))
                        except FileNotFoundError:
                            print(f"[v] 权重不存在: {new_weights}，切换取消")
                        else:
                            if glove_det is not None:
                                glove_det.close()
                            glove_det = _gd
                            glove_choice = new_choice
                            print(f"[v] 检测器 → {glove_det.backend} "
                                  f"{new_weights}（conf {glove_det.det_conf}"
                                  f"；track/滤波状态重置）")
                    else:
                        print("[v] 仅黑手套模式可用（按 g 进入）")
                if key == ord("g"):
                    glove_mode = not glove_mode
                    if glove_mode:
                        if glove_det is None:
                            try:
                                glove_det = GloveDetector(
                                    weights=glove_weights,
                                    **_glove_kwargs(args))
                            except FileNotFoundError:
                                print(f"[g] 手套权重不存在: {glove_weights}，"
                                      f"切换取消")
                                glove_mode = False
                        if glove_mode:
                            # 异步检测线程随手套模式暂停：手套 CUDA 链走
                            # 同步路径；切回裸手时循环顶部按需重建
                            if det_worker is not None:
                                det_worker.stop()
                                det_worker = None
                            print(f"[g] 黑手套模式（{glove_det.backend} "
                                  f"{glove_det.device}）")
                    else:
                        # 切回裸手清追踪态防陈旧跟踪。不能用 det.reset()：
                        # MediaPipeHandPipeline.reset() 只 landmarker.close()
                        # 不重建（换视频源语义），调用后裸手永久不检测（实机踩）
                        # → 直接重建新实例（模型已缓存，~百 ms 一次性）。
                        # 手套侧同样 close 置 None：再切回走惰性新建，天然干净
                        if glove_det is not None:
                            glove_det.close()
                            glove_det = None
                        det.close()
                        det = MediaPipeDetector(num_hands=2, delegate=delegate,
                                                det_conf=args.det_conf,
                                                track_conf=args.track_conf)
                        print("[g] 裸手模式")
            else:
                now = time.perf_counter()
                fps_win_n += 1
                if now - fps_win_t >= 1.0:
                    fps = fps_win_n / (now - fps_win_t)
                    fps_win_t, fps_win_n = now, 0
                    _dn = max(n_det, 1) if det_async else max(n, 1)
                    print(f"  frame {n}: {fps:.1f} fps  "
                          f"det {t_det / _dn * 1000:.1f}ms"
                          + ("（检测帧）" if det_async else "") + "  "
                          f"align {t_align / max(n, 1) * 1000:.1f}ms")
            n += 1
            if _prof and _prof_w1 is not None:
                _prof_works.append((time.perf_counter() - _prof_w1) * 1000.0)
                faulthandler.cancel_dump_traceback_later()
    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        if _prof:
            faulthandler.cancel_dump_traceback_later()   # break 路径兜底撤防
        if det_worker is not None:
            det_worker.stop()     # 先停检测线程再 close det（线程正在用它）
        source.close()
        det.close()
        if glove_det is not None:
            glove_det.close()
        cv2.destroyAllWindows()
        # 撕裂诊断收尾：退出前同步导出当前环（tear_probe 时）。
        # 看到撕裂按 q 退出后，最近 ~2.7s 内部帧在此。
        if tear_probe and len(_tear_ring):
            _tdir = os.path.join(_SHOT_DIR, "tear_exit_000")
            _write_tear_dump(list(_tear_ring), _tdir)
            print(f"  撕裂环已导出: {_tdir} ({len(_tear_ring)} 帧内部缓冲)"
                  f"——看到撕裂退出即在此")

    if n:
        t_total = time.perf_counter() - t0
        _dn = max(n_det, 1) if det_async else max(n, 1)
        print(f"\n── {n} 帧, {t_total:.1f}s, 平均 {n / t_total:.1f} fps "
              f"（det 均 {t_det / _dn * 1000:.1f}ms"
              + ("，检测帧" if det_async else "") + " "
              f"align 均 {t_align / n * 1000:.1f}ms）──")
    if export_dir is not None:
        _write_export(export_dir, exp2d_rows, exp3d_rows,
                      exp_sink, exp_sink2d, n)
    if args.stats and stats["n"]:
        nn = stats["n"]
        print("── 诊断（--stats）──")
        print(f"  检测手数: 0 手 {stats['det0']} 帧 ({stats['det0'] / nn * 100:.1f}%)"
              f" | 1 手 {stats['det1']} 帧 ({stats['det1'] / nn * 100:.1f}%)"
              f" | 2 手 {stats['det2']} 帧 ({stats['det2'] / nn * 100:.1f}%)")
        if stats["glove_n"]:
            print(f"  手套模式: {stats['glove_n']} 帧 | 出框 "
                  f"{stats['glove_box']} 帧"
                  f" ({stats['glove_box'] / stats['glove_n'] * 100:.1f}%)")
        print(f"  两手同 label（voter 重建期，已几何兜底）: {stats['same_label']} 帧")
        print(f"  wholesale: {stats['wholesale']} 起（两帧确认跳过 "
              f"{stats['ws_skip']} 帧单帧跳变） | 门控点数: {stats['gate_pts']}"
              f" | 豁免点数: {stats['forgive_pts']}")
        if len(diag_cens) > 1:
            # 静止段（αβ 质心帧间位移 <50mm，排除真实运动/手入场阶跃）
            # 各级平滑链质心帧间位移 p95：量化 αβ / OneEuro / 质心锚定
            # 逐级贡献（"整手跳"的直接指标；入场瞬态是阶跃响应非跳变，
            # 不计入）
            print("  静止段质心帧间位移 p95（αβ 位移 <50mm 帧-槽对）:")
            for name, idx in (("αβ 槽状态 h3", 0),
                              ("OneEuro smoothed", 1),
                              ("质心锚定 renderer_in", 2)):
                disps = []
                for i in range(1, len(diag_cens)):
                    p_cur = diag_cens[i][idx]
                    p_prv = diag_cens[i - 1][idx]
                    h_cur = diag_cens[i][0]
                    h_prv = diag_cens[i - 1][0]
                    for s in range(2):
                        fc = np.isfinite(p_cur[s]).all(axis=1)
                        fp = np.isfinite(p_prv[s]).all(axis=1)
                        hc = np.isfinite(h_cur[s]).all(axis=1)
                        hp = np.isfinite(h_prv[s]).all(axis=1)
                        if fc.sum() < 8 or fp.sum() < 8 \
                                or hc.sum() < 8 or hp.sum() < 8:
                            continue
                        dh = float(np.linalg.norm(
                            np.median(h_cur[s, hc], axis=0)
                            - np.median(h_prv[s, hp], axis=0)))
                        if dh > 0.05:
                            continue
                        disps.append(float(np.linalg.norm(
                            np.median(p_cur[s, fc], axis=0)
                            - np.median(p_prv[s, fp], axis=0))))
                if disps:
                    print(f"    {name}: {np.percentile(disps, 95) * 1000:.1f}mm"
                          f"（静止帧-槽对 n={len(disps)}）")
        if stats["det1"]:
            print("  提示: 1 手帧占比高 → 先查检测（--det-conf/--track-conf "
                  "降 0.3、--exposure 4000-10000 压运动模糊）；")
            print("        若检测到 2 手但显示 1 手 → 查 wholesale/分配")


if __name__ == "__main__":
    main()
