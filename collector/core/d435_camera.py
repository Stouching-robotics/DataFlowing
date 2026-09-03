"""
Intel RealSense D400 系列（D435/D435I/D405…）深度双目采集模块 —— 进程内 QObject worker。

D400 系列是稳定 UVC 用户态设备（pyrealsense2 自带 librealsense2 .so + libusb 后端），
无需像 S80M 那样隔离到子进程。后台线程跑 rs2.pipeline，对外输出两路
（RGB 彩色 / 深度），经 Qt 信号跨线程发回主线程，复用
MainWindow 双目帧消费模式（显示 + 录制外部源链路）。
左红外流仅内部启用（时间戳元数据），不对外输出；右红外不再开流
（基线改从 depth sensor 的 stereo_baseline 选项直读）。

多设备支持：worker 按 serial 锁定具体设备（config.enable_device），
面板点击哪台就开哪台；serial 为空时自动取第一台 D400。

信号签名与 MainWindow.stereo_frame_ready 一致:
    frames_ready(str slot_id, np.ndarray frame, int hardware_ns, list imu_samples)
        - "d435_rgb": BGR 三通道彩色图
        - "d435_depth": uint16 毫米深度图（D435/D405 无 IMU，imu_samples 恒为 []）

左红外流仅内部启用（时间戳元数据），不对外输出、不显示、不落盘；
右红外不再开流——其唯一用途是算基线，现改从 depth sensor 的
stereo_baseline 选项直读。双 RealSense 同 hub 共享带宽时多一条
IR2 会让 D405 掉到 ~19fps（实测，与开启顺序无关），去掉后两台满帧。
深度本身由红外双目内部计算，无需暴露红外画面。

用法:
    from core.d435_camera import D435Worker, d435_available, list_d400_devices
    if d435_available():
        worker = D435Worker(width=848, height=480, fps=30, serial="…")
        worker.frames_ready.connect(on_frames)
        worker.start()
        ...
        worker.stop()

亦可独立运行做无头自检:
    python core/d435_camera.py
"""

from __future__ import annotations
import os
import sys

# 脚本直跑(如 python core/d435_camera.py)时 sys.path[0] 是 core/,
# 需把项目根插入 sys.path 才能 import config —— 仅在直跑场景生效
if os.path.dirname(os.path.abspath(__file__)) == sys.path[0]:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
from collections import deque
from typing import Optional, List

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QMutex, QMutexLocker, pyqtSignal

from config import settings

# pyrealsense2 延迟失败:未安装时本模块仍可 import(d435_available 返回 False)
try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


# d435_available 结果缓存(探测一次,避免重复打开设备上下文)
_d435_avail_cache: Optional[bool] = None


def d435_available(force: bool = False) -> bool:
    """检测 pyrealsense2 可用且存在 D400 系列设备。结果模块级缓存。

    force=True 绕过缓存（热插拔后点击路径的活体复查）。
    """
    global _d435_avail_cache
    if not force and _d435_avail_cache is not None:
        return _d435_avail_cache
    _d435_avail_cache = bool(list_d400_devices())
    return _d435_avail_cache


def list_d400_devices() -> List[tuple]:
    """返回已连接的 D400 系列设备 [(name, serial), …]（无设备/未安装 → []）。"""
    out: List[tuple] = []
    if rs is None:
        return out
    try:
        ctx = rs.context()
        for d in ctx.query_devices():
            try:
                if "D400" not in d.get_info(rs.camera_info.product_line):
                    continue
                out.append((d.get_info(rs.camera_info.name),
                            d.get_info(rs.camera_info.serial_number)))
            except Exception:
                continue
    except Exception:
        pass
    return out


def _frameset_timestamp_us(frames, rs_mod) -> float:
    """帧时间戳逐级回退（µs）：
      1) 深度帧 frame_timestamp 元数据（D435 老固件行为）
      2) 左红外帧 frame_timestamp 元数据
      3) 主机时间戳 frames.get_timestamp() × 1000

    D405 + pyrealsense2 2.58.3 的深度/红外流不支持该元数据
    （supports_frame_metadata=False，无条件读即 RuntimeError → 外层误判
    设备异常 → 2s 重连死循环）。每级先 supports_frame_metadata 探测再读，
    读失败继续回退；D405 实测走第 3 级——会话内仍单调递增，
    hardware_ns 语义不变，精度从设备计数器降为主机时钟。
    """
    for getter in (
            lambda: frames.get_depth_frame(),
            lambda: frames.get_infrared_frame(1),
    ):
        try:
            f = getter()
            if (f is not None and f.supports_frame_metadata(
                    rs_mod.frame_metadata_value.frame_timestamp)):
                meta = f.get_frame_metadata(
                    rs_mod.frame_metadata_value.frame_timestamp)
                if meta is not None:
                    return meta
        except Exception:
            continue
    return frames.get_timestamp() * 1000


def _apply_color_exposure(sensor, auto: bool, value: float) -> bool:
    """对 color 传感器应用曝光（µs）：auto=True 只开自动曝光；
    否则先关自动再写手动值。异常返回 False（不打断采集线程）。"""
    try:
        sensor.set_option(rs.option.enable_auto_exposure, 1 if auto else 0)
        if not auto:
            sensor.set_option(rs.option.exposure, float(value))
        return True
    except Exception:
        return False


class D435Worker(QObject):
    """D435 采集 worker:后台线程 rs2.pipeline 三路流,信号回主线程。"""

    # (slot_id, frame, hw_ns, imu_samples)
    # ★ hw_ns 必须用 object:PyQt5 队列信号把 Python int 按 C++ qint32
    #   封送,超过 2^31(≈2.1s 纳秒)即静默截断为负数(S80M 已有数据受害)
    frames_ready = pyqtSignal(str, np.ndarray, object, list)
    error_occurred = pyqtSignal(str)                         # 错误消息
    status_changed = pyqtSignal(str)                         # "running" / "stopped"

    def __init__(self, width: int = None, height: int = None,
                 fps: int = None, parent=None, ts_log: str = None,
                 rgb_width: int = None, rgb_height: int = None,
                 serial: str = None, model_name: str = None,
                 rgb_slot: str = None, depth_slot: str = None,
                 exposure: dict = None):
        super().__init__(parent)
        self._width = width if width is not None else settings.D435_RESOLUTION[0]
        self._height = height if height is not None else settings.D435_RESOLUTION[1]
        self._fps = fps if fps is not None else settings.D435_FPS
        self._rgb_width = (rgb_width if rgb_width is not None
                           else settings.D435_RGB_RESOLUTION[0])
        self._rgb_height = (rgb_height if rgb_height is not None
                            else settings.D435_RGB_RESOLUTION[1])
        self._serial = serial            # 目标设备序列号（None → 第一台 D400）
        self._model_name = model_name    # 仅用于日志/诊断
        # 槽位名（多设备时由 MainWindow 消歧分配；None 回落 settings 常量）
        self._rgb_slot = rgb_slot or settings.D435_SLOT_RGB
        self._depth_slot = depth_slot or settings.D435_SLOT_DEPTH
        self._depth_unit_factor = 1.0    # 设备深度单位 → mm 换算系数（流启动时探测）
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._calibration = None        # StereoCalibration,首帧后可用
        self._frame_count = 0
        self._t0_us: Optional[int] = None   # 首个帧计数(µs,重连时重置)
        self._last_us: int = 0              # 上一帧原始计数(解绕用)
        self._wrap_us: int = 0              # 累计回绕补偿(µs)
        self._ts_log = ts_log               # 诊断 CSV: i,ft_ms,meta_us(仅调试)
        self._ts_fh = None                  # 首启失败时 _run 尾部清理仍需存在
        # 曝光设置（主线程只写 pending；采集线程在流启动后应用，
        # 每次重连新建管道须重放最近一次设置）
        self._exp_lock = QMutex()
        self._exp_auto: Optional[bool] = None
        self._exp_value: Optional[float] = None
        self._exp_pending = False
        self._exp_range: Optional[tuple] = None   # (min, max) µs，流启动后缓存
        # 「最一开始」曝光基线（开流时读回硬件，应用任何设置之前）
        self._original_exp: Optional[tuple] = None
        if exposure:
            self._exp_auto = bool(exposure.get("auto", True))
            self._exp_value = float(exposure.get("value", 0.0))
            self._exp_pending = True

    # ── 生命周期 ────────────────────────────────────────

    def start(self):
        """启动后台采集线程(幂等)。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="d435-capture")
        self._thread.start()

    def stop(self):
        """停止采集线程(阻塞至多约 3s;wait_for_frames 1s 超时保证退出)。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.status_changed.emit("stopped")

    # ── 标定(录制开始时由主线程读取) ────────────────────

    def get_calibration(self):
        """返回首帧后提取的 StereoCalibration;尚未就绪时为 None。"""
        return self._calibration

    # ── 曝光调节（主线程只写 pending；采集线程应用并缓存量程） ──

    def set_exposure(self, auto: bool, value: float):
        """设置 color 传感器曝光：auto=True 开自动；否则手动值（µs）。"""
        with QMutexLocker(self._exp_lock):
            self._exp_auto = bool(auto)
            self._exp_value = float(value)
            self._exp_pending = True

    def exposure_info(self) -> tuple:
        """返回 ((min,max)µs 或 None, auto, value)；流未启动时量程 None。"""
        with QMutexLocker(self._exp_lock):
            return (self._exp_range, self._exp_auto, self._exp_value)

    def original_exposure(self) -> tuple:
        """「最一开始」曝光基线 (auto, value)；未捕获返回 None。"""
        with QMutexLocker(self._exp_lock):
            return self._original_exp

    # ── 后台线程 ────────────────────────────────────────

    def _run(self):
        if rs is None:
            self.error_occurred.emit("pyrealsense2 未安装")
            return

        while not self._stop_event.is_set():
            # 每次重连新建管道:复用同一 rs.pipeline 在"流停滞"后 stop→start
            # 无法复位 UVC 通道,同一对象会无限循环"无帧→重连"(实测停滞机
            # 新管道立即恢复出帧——新管道等价于新进程打开设备)
            pipe = rs.pipeline()
            try:
                config = rs.config()
                # 多设备时按序列号锁定目标设备；不指定则用第一台 D400
                if self._serial:
                    config.enable_device(self._serial)
                # 对外: RGB 彩色 + 深度两路
                config.enable_stream(rs.stream.color,
                                     self._rgb_width, self._rgb_height,
                                     rs.format.rgb8, self._fps)
                config.enable_stream(rs.stream.depth,
                                     self._width, self._height,
                                     rs.format.z16, self._fps)
                # 内部: 左红外仅用于时间戳元数据,不对外输出。右红外不开流:
                # 其唯一用途是算基线,现从 stereo_baseline 选项直读——
                # 双 RealSense 同 hub 时多一条 IR2 会把 D405 挤到 ~19fps
                config.enable_stream(rs.stream.infrared, 1,
                                     self._width, self._height,
                                     rs.format.y8, self._fps)
                profile = pipe.start(config)
                # 设备深度单位探测（D435=1mm，D405=0.1mm）→ mm 换算系数。
                # 帧统一归一化到 mm 输出：全链路（热力图/无损深度流/metadata
                # "unit:mm"/depth_scale 0.001）都按 mm 约定。
                try:
                    dev_unit = float(profile.get_device()
                                     .first_depth_sensor().get_depth_scale())
                except Exception:
                    dev_unit = float(settings.DEPTH_SCALE)
                self._depth_unit_factor = dev_unit / float(settings.DEPTH_SCALE)
                # 曝光：流启动后缓存 color 传感器量程并应用最近设置。
                # 重连 = 新管道 = 新传感器句柄，相机端设置会丢，须每次重放；
                # 消费 pending 同样在本线程完成（主线程只写 pending）。
                try:
                    color = profile.get_device().first_color_sensor()
                    rng = color.get_option_range(rs.option.exposure)
                    with QMutexLocker(self._exp_lock):
                        self._exp_range = (float(rng.min), float(rng.max))
                        # 「最一开始」曝光基线：应用任何设置前读回硬件
                        # 当前值（重连 = 新管道，读的是本次开流的硬件状态）
                        try:
                            oa = float(color.get_option(
                                rs.option.enable_auto_exposure))
                            ov = float(color.get_option(rs.option.exposure))
                            self._original_exp = (oa > 0.5, ov)
                        except Exception:
                            self._original_exp = None
                        auto, value = self._exp_auto, self._exp_value
                        self._exp_pending = False
                    if auto is not None:
                        _apply_color_exposure(color, auto,
                                              value if value is not None else 0.0)
                except Exception:
                    pass
                if self._calibration is None:
                    self._calibration = _build_calibration(
                        profile, self._fps, (self._width, self._height))
                    # 帧已归一化为 mm，标定里 depth_scale 写输出格式刻度
                    # （设备原生单位仅影响内部换算，不进入落盘标定）
                    self._calibration.depth_scale = float(settings.DEPTH_SCALE)
                self._t0_us = None  # 设备计数随流启动重置,重连后重新归零
                self._last_us = 0
                self._wrap_us = 0
                self._ts_fh = open(self._ts_log, "w") if self._ts_log else None
                if self._ts_fh:
                    self._ts_fh.write("i,ft_ms,meta_us\n")
                self.status_changed.emit("running")

                # 看门狗一: 停滞——连续无帧秒数(双 RealSense 第二台开流偶发
                # 静默停滞——管道已启动但不再出帧,wait_for_frames 持续超时,
                # 无异常可捕获;累计超过阈值则抛给外层走重连)
                # 看门狗二: 帧率——半死状态(零星出帧不报错但画面极卡),
                # 10s 窗口内帧数低于期望一半 → 同样抛给外层走重连
                stall_s = 0
                fps_win: deque = deque()          # 帧到达时刻(monotonic 秒)
                fps_win_start: Optional[float] = None   # 重连后首个帧时刻
                while not self._stop_event.is_set():
                    try:
                        frames = pipe.wait_for_frames(timeout_ms=1000)
                    except RuntimeError:
                        # 超时无帧(拔线初期/流停滞) → 计数,超阈值触发重连
                        stall_s += 1
                        if stall_s >= settings.D435_STALL_TIMEOUT_S:
                            raise RuntimeError(
                                f"流停滞({stall_s}s 无帧),触发重连")
                        continue
                    stall_s = 0
                    # 消费 pending 曝光设置（主线程 set_exposure 下发）
                    with QMutexLocker(self._exp_lock):
                        if self._exp_pending and self._exp_auto is not None:
                            auto, value = self._exp_auto, self._exp_value
                            self._exp_pending = False
                        else:
                            auto = None
                    if auto is not None:
                        try:
                            color = profile.get_device().first_color_sensor()
                            _apply_color_exposure(
                                color, auto, value if value is not None else 0.0)
                        except Exception:
                            pass
                    now_s = time.monotonic()
                    fps_win.append(now_s)
                    while (fps_win
                           and now_s - fps_win[0] > settings.D435_LOW_FPS_WINDOW_S):
                        fps_win.popleft()
                    if fps_win_start is None:
                        fps_win_start = now_s
                    elif (now_s - fps_win_start >= settings.D435_LOW_FPS_WINDOW_S
                          and len(fps_win) < (settings.D435_LOW_FPS_FRACTION
                                              * self._fps
                                              * settings.D435_LOW_FPS_WINDOW_S)):
                        raise RuntimeError(
                            f"帧率过低({len(fps_win)}帧/"
                            f"{settings.D435_LOW_FPS_WINDOW_S}s,期望≥"
                            f"{settings.D435_LOW_FPS_FRACTION * self._fps:.0f}fps),"
                            f"触发重连")

                    # 帧时间戳逐级回退（深度→红外→主机时钟，见
                    # _frameset_timestamp_us）。设备计数为 32 位 µs 计数器
                    # (2^32µs ≈ 71.6min 才回绕),软件解绕 + 首帧归零 → 会话内
                    # 单调递增 hardware_ns(与 S80M 会话相对时钟语义一致)。
                    # 注: 排查记录里疑似"每 2.1s 回绕"实为 Qt 信号 qint32
                    # 截断假象(>2^31ns 翻转,已改用 object 参数修复)。
                    meta_us = _frameset_timestamp_us(frames, rs)
                    raw_us = int(meta_us)
                    if self._t0_us is None:
                        self._t0_us = raw_us
                        self._last_us = raw_us
                        hw_ns = 0
                    else:
                        # 解绕: 下跌超过半程(2^31µs) → 补一个整程(2^32µs)
                        if raw_us < self._last_us - (1 << 31):
                            self._wrap_us += 1 << 32
                        self._last_us = raw_us
                        hw_ns = (raw_us + self._wrap_us - self._t0_us) * 1000
                    if self._ts_fh:
                        self._ts_fh.write(
                            f"{self._frame_count},{frames.get_timestamp()},"
                            f"{int(meta_us)}\n")
                        self._ts_fh.flush()

                    # 显式 copy:rs2 帧缓冲由 C++ 持有,frameset 一释放即失效;
                    # 深度帧经排队信号延迟消费,必须脱离底层缓冲(848×480×2≈0.8MB)
                    depth = np.asanyarray(
                        frames.get_depth_frame().get_data()).copy()  # 设备原生单位
                    if self._depth_unit_factor != 1.0:
                        # 归一化到 mm（D405 原生 0.1mm/单位；×0.1 不会溢出 uint16）
                        depth = (depth.astype(np.float32)
                                 * self._depth_unit_factor + 0.5).astype(np.uint16)

                    # RGB8 → BGR 三通道(显示 + MP4 录制链路要求 BGR24)
                    rgb = np.asanyarray(frames.get_color_frame().get_data())
                    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    self.frames_ready.emit(self._rgb_slot,
                                           rgb_bgr, hw_ns, [])
                    self.frames_ready.emit(self._depth_slot,
                                           depth, hw_ns, [])
                    self._frame_count += 1

            except RuntimeError as e:
                # 拔线 / 设备异常 → 每 2s 重连(仿 CameraWorker 重连策略)
                if not self._stop_event.is_set():
                    self.error_occurred.emit(
                        f"RealSense {self._model_name or ''} "
                        f"{self._serial or ''} 设备异常,2s 后重连: {e}".replace("  ", " "))
                    self.status_changed.emit("error")
                try:
                    pipe.stop()
                except Exception:
                    pass
                self._stop_event.wait(settings.CAMERA_RECONNECT_INTERVAL_MS / 1000.0)
            except Exception as e:
                if not self._stop_event.is_set():
                    self.error_occurred.emit(f"RealSense 采集异常: {e}")
                try:
                    pipe.stop()
                except Exception:
                    pass
                self._stop_event.wait(settings.CAMERA_RECONNECT_INTERVAL_MS / 1000.0)

        try:
            pipe.stop()
        except Exception:
            pass
        if self._ts_fh:
            try:
                self._ts_fh.close()
            except Exception:
                pass
            self._ts_fh = None


# ── 标定提取 ───────────────────────────────────────────

def _build_calibration(profile, fps: int, resolution: tuple):
    """从 rs2 pipeline profile 提取左右红外内参/深度内参/基线 → StereoCalibration。"""
    from core.calibration import StereoCalibration, CameraIntrinsics

    def _ir_intrinsics(idx: int) -> CameraIntrinsics:
        try:
            prof = profile.get_stream(rs.stream.infrared, idx)
        except RuntimeError:
            prof = None   # 未启用的流: get_stream 抛异常而非返回 None
        if prof is None:
            # IR2 已不开流:右目与左目是同一传感器模组、几何一致,
            # 内参镜像左目(基线另从 stereo_baseline 选项读取)
            return _ir_intrinsics(1)
        intr = prof.as_video_stream_profile().get_intrinsics()
        return CameraIntrinsics(
            intrinsic=[intr.fx, intr.fy, intr.ppx, intr.ppy],
            distortion=list(intr.coeffs),   # Inverse Brown Conrady 系数,原样存储
        )

    def _depth_intrinsics() -> CameraIntrinsics:
        prof = profile.get_stream(rs.stream.depth)
        intr = prof.as_video_stream_profile().get_intrinsics()
        return CameraIntrinsics(
            intrinsic=[intr.fx, intr.fy, intr.ppx, intr.ppy],
            distortion=list(intr.coeffs),
        )

    def _baseline() -> float:
        # 直读 depth sensor 的 stereo_baseline 选项(单位 mm→m),无需 IR2 流。
        # 与旧 IR1→IR2 外参算法实测一致(D435 50.15mm / D405 17.98mm)
        try:
            return float(profile.get_device().first_depth_sensor()
                         .get_option(rs.option.stereo_baseline)) / 1000.0
        except Exception:
            # 无此选项的罕见设备:IR2 若仍在流中则回落外参算法
            # (未启用时 get_stream 抛异常而非返回 None)
            try:
                ir2 = profile.get_stream(rs.stream.infrared, 2)
            except RuntimeError:
                ir2 = None
            if ir2 is not None:
                ir1 = profile.get_stream(
                    rs.stream.infrared, 1).as_video_stream_profile()
                trans = ir1.get_extrinsics_to(
                    ir2.as_video_stream_profile()).translation
                return float(np.linalg.norm(trans))   # D435 约 0.050 m
            return float(settings.STEREO_BASELINE)

    def _depth_scale() -> float:
        try:
            sensor = profile.get_device().first_depth_sensor()
            return float(sensor.get_depth_scale())
        except Exception:
            return float(settings.DEPTH_SCALE)

    return StereoCalibration(
        resolution=[resolution[0], resolution[1]],
        fps=float(fps),
        baseline=_baseline(),
        left_camera=_ir_intrinsics(1),
        right_camera=_ir_intrinsics(2),
        depth_camera=_depth_intrinsics(),
        depth_scale=_depth_scale(),
        cam_imu_timeshift=0.0,   # D435 无 IMU
    )


# ── 无头自检 ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    # 脚本直跑时把项目根加入 sys.path(config/core 均为包)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from PyQt5.QtCore import QCoreApplication

    app = QCoreApplication(sys.argv)

    if not d435_available():
        print("未检测到 D435 设备(pyrealsense2 或设备缺失)")
        sys.exit(1)

    worker = D435Worker()
    stats = {"t0": time.time(), "count": 0, "saved": False,
             "sample_dir": "/tmp/d435_selftest"}

    def on_frames(slot_id, frame, hw_ns, imu_samples):
        stats["count"] += 1
        # 无头自检不弹窗口(cv2.imshow 与 PyQt5 同进程会冲突);
        # 各抽一帧存 /tmp 供人工查看
        if not stats["saved"]:
            os.makedirs(stats["sample_dir"], exist_ok=True)
            if slot_id == settings.D435_SLOT_DEPTH:
                cv2.imwrite(os.path.join(stats["sample_dir"], "depth.png"), frame)
            else:
                cv2.imwrite(os.path.join(stats["sample_dir"],
                                         f"{slot_id}.png"), frame)
            if os.path.isfile(os.path.join(stats["sample_dir"], "depth.png")) \
                    and os.path.isfile(os.path.join(stats["sample_dir"],
                                                    f"{settings.D435_SLOT_RGB}.png")):
                stats["saved"] = True

    worker.frames_ready.connect(on_frames)
    worker.error_occurred.connect(lambda m: print("[error]", m))
    worker.start()

    print(f"D435 自检: RGB {worker._rgb_width}x{worker._rgb_height} + "
          f"深度 {worker._width}x{worker._height}@{worker._fps}, 运行 5s...")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)

    elapsed = time.time() - stats["t0"]
    print(f"  帧率: {stats['count'] / max(elapsed, 1e-6):.1f} 信号/秒")
    print(f"  抽样帧: {stats['sample_dir']}/ (depth.png, d435_rgb.png)")

    calib = worker.get_calibration()
    if calib is not None:
        print("── 标定 ──")
        print(f"  baseline     = {calib.baseline:.6f} m")
        print(f"  depth_scale  = {calib.depth_scale}")
        print(f"  left  IR 内参 = {calib.left_camera.intrinsic}")
        print(f"  right IR 内参 = {calib.right_camera.intrinsic}")
        print(f"  depth 内参   = {calib.depth_camera.intrinsic}")

    worker.stop()
