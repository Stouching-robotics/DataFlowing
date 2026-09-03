"""
摄像机采集模块 —— 后台线程采集帧，通过 Qt 信号发送到主线程。
所有 VideoCapture 操作在采集线程内执行，确保线程安全。
"""

from __future__ import annotations
import os
import re
import time
import threading
import queue
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

from config import settings


class CameraState:
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    RECORDING = "recording"
    ERROR = "error"


if os.name == "nt":
    _BACKENDS = [
        ("DShow", cv2.CAP_DSHOW),
        ("MSMF",  cv2.CAP_MSMF),
        ("ANY",   cv2.CAP_ANY),
    ]
else:
    _BACKENDS = [
        ("V4L2",   cv2.CAP_V4L2),
        ("FFMPEG", cv2.CAP_FFMPEG),   # 兜底：OpenCV 5.0 V4L2 的 obsensor bug 可能导致 UVC 相机无法打开
        ("ANY",    cv2.CAP_ANY),
    ]


class _Quiet:
    def __enter__(self):
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        self._saved_fd = os.dup(2)
        os.dup2(self._devnull_fd, 2)
        os.close(self._devnull_fd)
        return self
    def __exit__(self, *_):
        os.dup2(self._saved_fd, 2)
        os.close(self._saved_fd)
        return False


def _quiet():
    return _Quiet()


def _find_persistent_v4l_path(index: int) -> Optional[str]:
    """查找指向 /dev/video{index} 的 by-id 永久路径。

    返回 /dev/v4l/by-id/xxx 路径，该路径在设备重新插拔后保持不变，
    即使 /dev/videoN 编号发生变化。
    """
    target = f"../../video{index}"
    by_id_dir = "/dev/v4l/by-id"
    if not os.path.isdir(by_id_dir):
        return None
    try:
        for entry in os.listdir(by_id_dir):
            entry_path = os.path.join(by_id_dir, entry)
            try:
                if os.readlink(entry_path) == target:
                    return entry_path
            except OSError:
                pass
    except OSError:
        pass
    return None


def _scan_all_v4l_by_id_paths() -> list:
    """扫描 /dev/v4l/by-id/ 下所有设备路径，返回按优先级排列的路径列表。"""
    paths = []
    by_id_dir = "/dev/v4l/by-id"
    if not os.path.isdir(by_id_dir):
        return paths
    try:
        for entry in sorted(os.listdir(by_id_dir)):
            entry_path = os.path.join(by_id_dir, entry)
            if os.path.islink(entry_path):
                paths.append(entry_path)
    except OSError:
        pass
    return paths


def _resolve_v4l_index(index: int) -> list:
    """将 /dev/videoN 索引解析为应尝试的设备路径列表。

    优先级:
    1. /dev/v4l/by-id/xxx → /dev/video{index} 的永久路径（不随重新插拔改变）
    2. 数字索引 /dev/video{index}
    3. 所有 by-id 路径（仅作为重连兜底：设备可能换了 /dev/videoN 编号）
    """
    paths = []
    persistent = _find_persistent_v4l_path(index)
    if persistent:
        paths.append(persistent)
    paths.append(index)
    return paths


def _try_open_camera(index: int, test_read: bool = False,
                     fallback_all_by_id: bool = False) -> Tuple[Optional[cv2.VideoCapture], str]:
    """尝试打开摄像机。

    Args:
        index: /dev/videoN 数字索引
        test_read: 是否测试读取第一帧
        fallback_all_by_id: 索引失败后是否尝试所有 by-id 路径（用于重连场景）
    """
    # 跳过 FTDI 双目相机设备（需专用 SDK，OpenCV 无法正确捕获）
    if _is_sdk_device(index):
        return None, ""
    # 跳过 RealSense D435 的 UVC 节点（深度走 pyrealsense2 专用通道）
    if _is_realsense_node(index):
        return None, ""
    for device in _resolve_v4l_index(index):
        for name, backend in _BACKENDS:
            try:
                cap = cv2.VideoCapture(device, backend)
                if cap.isOpened():
                    if test_read:
                        ok, frame = cap.read()
                        if not (ok and frame is not None and frame.size > 0):
                            cap.release()
                            continue
                    return cap, name
            except Exception:
                pass
    # 重连兜底：扫描所有 by-id 路径（设备可能换了 /dev/videoN 编号）
    if fallback_all_by_id:
        for p in _scan_all_v4l_by_id_paths():
            # 跳过需专用 SDK / RealSense 的节点，避免 UVC 断线时误开
            try:
                target = os.readlink(p)
                if target.startswith("../../video"):
                    n = int(target.split("video")[-1])
                    if _is_sdk_device(n) or _is_realsense_node(n):
                        continue
            except (OSError, ValueError):
                pass
            for name, backend in _BACKENDS:
                try:
                    cap = cv2.VideoCapture(p, backend)
                    if cap.isOpened():
                        if test_read:
                            ok, frame = cap.read()
                            if not (ok and frame is not None and frame.size > 0):
                                cap.release()
                                continue
                        return cap, name
                except Exception:
                    pass
    return None, ""


def _get_usb_camera_indices() -> set:
    """用 DirectShow 获取 USB 摄像机的索引集合。"""
    usb_indices = set()
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
        for i, name in enumerate(devices):
            nl = name.lower()
            builtin_kw = ["integrated", "built-in", "builtin", "laptop",
                          "thinkpad", "ideapad", "hp ", "dell ", "front",
                          "rear", "depth", "ir camera", "3d",
                          "user facing", "acer", "lenovo", "asus", "internal"]
            is_builtin = any(kw in nl for kw in builtin_kw)
            is_usb = "usb" in nl
            if is_usb or (not is_builtin):
                usb_indices.add(i)
    except Exception:
        pass
    return usb_indices


def _is_sdk_device(index: int) -> bool:
    """检查 V4L2 设备是否需专用 SDK（FTDI 双目相机），非标准 UVC。"""
    name_path = f"/sys/class/video4linux/video{index}/name"
    try:
        with open(name_path, "r") as f:
            return "FTDI" in f.read()
    except Exception:
        return False


def _usb_vid_pid(video_index: int) -> Optional[Tuple[str, str]]:
    """沿 /sys/class/video4linux/videoN 设备树向上（≤6 层）查找 idVendor/idProduct。

    返回 (vid, pid) 小写十六进制串；找不到返回 None。sysfs 只读，无副作用。
    """
    node = os.path.realpath(f"/sys/class/video4linux/video{video_index}")
    for _ in range(6):
        try:
            with open(os.path.join(node, "idVendor")) as f:
                vid = f.read().strip()
            with open(os.path.join(node, "idProduct")) as f:
                pid = f.read().strip()
            if vid and pid:
                return vid.lower(), pid.lower()
        except OSError:
            pass
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent
    return None


def _is_realsense_node(index: int) -> bool:
    """检查 V4L2 设备是否为 Intel RealSense 的 UVC 节点。

    RealSense D400 全家族由 pyrealsense2 专用通道输出，其 UVC 节点不能
    被 OpenCV 当普通相机使用（否则会与 D435 模式重复列出/抢占设备）。
    识别: vendor 8086 且（PID 命中 D435 的 0b07 或驱动 name 含
    "RealSense"）——D435i 是 0b3a、其余型号 PID 各异，用 name 兜底。
    """
    vp = _usb_vid_pid(index)
    if vp is None or vp[0] != settings.REALSENSE_VID:
        return False
    if vp[1] == settings.REALSENSE_PID:
        return True
    try:
        with open(f"/sys/class/video4linux/video{index}/name") as f:
            return "realsense" in f.read().lower()
    except OSError:
        return False


def _by_id_device_streams() -> dict:
    """解析 /dev/v4l/by-id 下所有条目，返回 {设备前缀: [video_index, ...]}。

    entry 形如 usb-X_Y_Z-serial-video-index0，去掉末尾 "-video-indexN"
    后作为物理设备前缀，同一物理设备的多个流聚到同一键下。
    """
    device_streams: dict = {}
    by_id_dir = "/dev/v4l/by-id"
    if not os.path.isdir(by_id_dir):
        return device_streams
    try:
        for entry in os.listdir(by_id_dir):
            entry_path = os.path.join(by_id_dir, entry)
            try:
                target = os.readlink(entry_path)
                if target.startswith("../../video"):
                    idx = int(target.split("video")[-1])
                    # 提取 USB 设备前缀，去掉末尾的 "-video-indexN"
                    parts = entry.rsplit("-video-index", 1)
                    prefix = parts[0] if len(parts) == 2 else entry
                    device_streams.setdefault(prefix, []).append(idx)
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return device_streams


def _discover_indices(max_index: int = 8) -> list:
    """发现应尝试打开的设备索引，包含通过 by-id 路径发现的额外索引。

    对于同一 USB 物理设备（如 DECXIN 相机的 video0 和 video1 两个流），
    只保留主视频流（-video-index0），过滤掉辅助流（-video-index1+）。
    避免同一台相机在页面上出现多个窗口。
    """
    # 基础扫描范围：所有 /dev/videoN（0..max_index-1）
    all_indices = set(range(max_index))

    device_streams = _by_id_device_streams()
    # 每个物理设备只保留主视频流（-video-index0，即最小索引）
    # 但仅当该设备的多个流 BOTH 都在扫描范围内时才排除辅助流
    for prefix, idx_list in device_streams.items():
        idx_list.sort()
        primary_idx = idx_list[0]
        # 确保主视频流在扫描范围内
        all_indices.add(primary_idx)
        # 只在超过 1 个流时排除辅助流
        if len(idx_list) > 1:
            for idx in idx_list[1:]:
                all_indices.discard(idx)

    return sorted(all_indices)


def list_v4l_devices(max_index: int = 16) -> list:
    """sysfs 只读枚举 V4L2 设备（不 open、不 test_read，轮询安全）。

    以 /dev/v4l/by-id 分组识别物理设备，每个物理设备只保留主视频流
    （最小索引）。返回列表每项:
      {video_index, name, serial, by_id_path, vid, pid, is_sdk}
    name 从 by-id 前缀解码（去 "usb-" 前缀、字段分隔符转空格）;
    serial 启发式取前缀最后一段（仅纯字母数字且 ≥8 字符才认作序号）。
    被占用的设备不会被打开，仍会稳定出现在列表中。
    """
    devices = []
    for prefix, idx_list in sorted(_by_id_device_streams().items()):
        idx_list = sorted(idx_list)
        idx = idx_list[0]           # 主视频流
        if idx >= max_index:
            continue
        if prefix.startswith("usb-"):
            # usb-<vendor>_<model>_<serial> → 分段转空格
            segments = [s for s in re.split(r"[-_]", prefix[len("usb-"):]) if s]
            serial = ""
            if segments and segments[-1].isalnum() and len(segments[-1]) >= 8:
                serial = segments[-1]   # 序号单独显示，不进名称
                segments = segments[:-1]
            name = " ".join(segments).strip()
        else:
            name, serial = "", ""
        if not name:
            # 非 USB 前缀无法解码，回退 sysfs name 文件
            try:
                with open(f"/sys/class/video4linux/video{idx}/name") as f:
                    name = f.read().strip()
            except OSError:
                name = f"video{idx}"
        vp = _usb_vid_pid(idx)
        devices.append({
            "video_index": idx,
            "name": name,
            "serial": serial,
            "by_id_path": _find_persistent_v4l_path(idx),
            "vid": vp[0] if vp else None,
            "pid": vp[1] if vp else None,
            "is_sdk": _is_sdk_device(idx),
            "is_realsense": _is_realsense_node(idx),
        })
    return devices


def detect_cameras(max_index: int = 8) -> list:
    usb_indices = _get_usb_camera_indices()
    with _quiet():
        found = []
        seen_indices = set()
        for idx in _discover_indices(max_index):
            if idx >= max_index * 2:  # 只扩展到合理范围
                continue
            if usb_indices and idx not in usb_indices:
                continue
            if _is_sdk_device(idx):
                continue  # 跳过 FTDI 双目相机（需专用 SDK）
            if _is_realsense_node(idx):
                continue  # 跳过 RealSense UVC 节点（D435 走 SDK 专用通道）
            if idx in seen_indices:
                continue
            cap, backend = _try_open_camera(idx, test_read=True)
            if cap is not None:
                cap.release()
                found.append((idx, backend))
                seen_indices.add(idx)
    return found


class CameraWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    state_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    camera_opened = pyqtSignal(int, int, str)

    _RECONNECT_BASE_DELAY = 2.0
    _RECONNECT_MAX_DELAY = 30.0
    _READ_FAIL_THRESHOLD = 120
    _INITIAL_FAIL_THRESHOLD = 30
    _INITIAL_GRACE_PERIOD = 90

    def __init__(self, camera_index: int = 0, resolution: tuple = None,
                 record_queue: queue.Queue = None, on_queue_full=None):
        """Args:
            on_queue_full: 录制队列满丢帧回调（无参；v1.0.9 丢帧统计用，
                采集线程内调用，必须轻量线程安全）
        """
        super().__init__()
        self.camera_index = camera_index
        # 默认满分辨率 1280×960@30（需 MJPG，见 _open_camera）
        self._requested_res = resolution if resolution is not None else settings.DEFAULT_RESOLUTION
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._paused = False
        self._mutex = QMutex()
        self._fps_counter = _FPSCounter()
        self._state = CameraState.DISCONNECTED
        self._actual_res: Tuple[int, int] = (0, 0)
        self._backend_name = ""
        self._thread: Optional[threading.Thread] = None
        # 帧采集时间戳（微秒），在 capture_loop 中随帧一起更新
        self._latest_capture_ts_us: int = 0
        self._ts_lock = QMutex()
        # 录制帧队列 —— B+ 方案：采集线程直接写入，绕过 Qt 信号/主线程
        self._record_queue = record_queue
        self._on_queue_full = on_queue_full   # v1.0.9 丢帧统计回调
        # 曝光：UVC 固定自动曝光（正常亮度 + 30fps；实测该相机手动模式
        # 帧率上限 ~26fps）。开启/重连后由采集循环在首帧读出成功后
        # 强制应用（见 _open_camera/_capture_loop）
        self._exp_lock = QMutex()
        self._exp_pending = False

    @property
    def state(self) -> str:
        return self._state
    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
    @property
    def resolution(self) -> Tuple[int, int]:
        return self._actual_res if self._actual_res[0] > 0 else settings.DEFAULT_RESOLUTION
    @property
    def latest_capture_ts_us(self) -> int:
        """最新帧的采集时间戳（Unix 微秒），线程安全。"""
        with QMutexLocker(self._ts_lock):
            return self._latest_capture_ts_us

    def start(self):
        with QMutexLocker(self._mutex):
            if self._running:
                return
            self._running = True
            self._paused = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"cam-{self.camera_index}")
        self._thread.start()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            self._thread = None
        self._release_camera()
        self._set_state(CameraState.DISCONNECTED)

    def pause(self): self._paused = True
    def resume(self): self._paused = False

    def _set_state(self, state: str):
        if self._state != state:
            self._state = state
            self.state_changed.emit(state)

    # ── 曝光（UVC 固定自动曝光：开启/重连后强制应用） ──

    @staticmethod
    def _apply_exposure_to(cap, auto: bool, value: float,
                           backend: str = "") -> tuple:
        """把曝光参数写进 VideoCapture。V4L2 下须先切自动/手动再写值，
        否则手动值被自动模式覆盖。返回 (auto 设置成功, 值设置成功)。

        V4L2 后端（OpenCV 5 起）对 CAP_PROP_AUTO_EXPOSURE 是菜单值直通：
        标准相机 0=自动 1=手动，但部分相机（如 DECXIN）菜单布局不同
        （1=手动、3=光圈优先即自动、无 0）。因此按候选顺序尝试，
        以 set 返回 True 且 get 读回一致为准。Windows 按 OpenCV MSMF
        约定 1=自动/0=手动 写入；DShow/FFMPEG 不实现该属性（写入无副作用）。
        """
        auto_prop = getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", 21)
        if backend in ("V4L2", "ANY") and os.name != "nt":
            # 自动：0 标准自动 → 3 光圈优先 → 2 快门优先；手动：1 标准手动 → 0 兜底
            candidates = (0, 3, 2) if auto else (1, 0)
            ok_auto = False
            for cand in candidates:
                if cap.set(auto_prop, cand) and cap.get(auto_prop) == cand:
                    ok_auto = True
                    break
        else:
            # Windows：OpenCV MSMF 约定 1=自动 0=手动（写 0 会把相机锁进
            # 手动曝光，亮度不可控）；DShow/FFMPEG 不实现该属性，无副作用。
            ok_auto = cap.set(auto_prop, 1 if auto else 0)
        if auto:
            return ok_auto, True
        ok_val = cap.set(cv2.CAP_PROP_EXPOSURE, max(1.0, float(value)))
        return ok_auto, ok_val

    def _open_camera(self) -> bool:
        self._release_camera()
        with _quiet():
            cap, backend = _try_open_camera(self.camera_index, fallback_all_by_id=True)
        if cap is None:
            self._set_state(CameraState.ERROR)
            self.error_occurred.emit(f"Camera {self.camera_index}: cannot open")
            return False
        self._cap = cap
        self._backend_name = backend
        # UVC 满分辨率 30fps 必须走 MJPG：YUYV 格式在 1280×960 硬限 5fps，
        # 且驱动默认格式就是 YUYV（只设分辨率会静默掉帧率）。读回不一致
        # 说明该相机不支持 MJPG → 继续用默认格式。
        # FFMPEG 兜底后端不识别 FOURCC/FPS 属性（写了也无效），跳过。
        # 设置顺序按后端拆分（两者要求相反）：
        #  - Windows DShow/MSMF：每次 SetFormat 都以引脚当前/首选媒体类型为
        #    基础重建，最后一次设置决定最终格式 → FPS → 分辨率 → FOURCC 收尾，
        #    之后不再设任何属性。旧顺序（先 FOURCC 再补 FPS）恰好把 MJPG 打回
        #    YUYV（960p 只剩 5fps 档）；插上 D435 后驱动协商行为变化才暴露。
        #  - Linux V4L2：FOURCC 须在分辨率之前设置，帧率放最后。
        tunable = backend in ("V4L2", "DShow", "MSMF", "ANY")
        want_fourcc = (cv2.VideoWriter_fourcc(*settings.UVC_FOURCC)
                       if settings.UVC_FOURCC else 0)

        def _set_fourcc(quiet: bool = False):
            if not (tunable and want_fourcc):
                return
            try:
                self._cap.set(cv2.CAP_PROP_FOURCC, want_fourcc)
                if not quiet and int(self._cap.get(cv2.CAP_PROP_FOURCC)) != want_fourcc:
                    print(f"Camera {self.camera_index}: FOURCC "
                          f"{settings.UVC_FOURCC} 不支持，回退默认格式",
                          flush=True)
            except Exception:
                pass

        # 显式请求采集帧率（V4L2 按离散间隔就近选择；DShow/MSMF 设媒体类型
        # 帧率）。不设时驱动取默认间隔——本相机 MJPG 默认 120fps，会白烧
        # CPU 且录制丢帧。
        def _set_fps():
            if not tunable:
                return
            try:
                self._cap.set(cv2.CAP_PROP_FPS, settings.DEFAULT_FPS)
            except Exception:
                pass

        def _set_resolution():
            if not self._requested_res:
                return
            w, h = self._requested_res
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if backend in ("DShow", "MSMF"):
            # Windows：FOURCC 收尾，之后不再设任何属性（见上方注释）
            _set_fps()
            _set_resolution()
            _set_fourcc()
        else:
            # Linux V4L2 / ANY 兜底：FOURCC 在前，帧率最后（见上方注释）
            _set_fourcc()
            _set_resolution()
            _set_fps()
        # UVC 固定自动曝光：开启/重连一律回到自动（正常亮度 + 30fps；
        # 实测该相机手动模式帧率上限 ~26fps）。不在此直接写：部分 UVC
        # 相机在 STREAMON（首次 read）时重置/忽略 AE 相关控制，开流前的
        # 写入会丢失 → 标记 pending，由采集循环首帧读出成功后下发。
        with QMutexLocker(self._exp_lock):
            self._exp_pending = True
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w <= 0 or actual_h <= 0:
            actual_w, actual_h = 640, 480
        self._actual_res = (actual_w, actual_h)
        # 诊断：打印后端协商结果（Windows 5fps 排查用，控制台可见）
        try:
            fc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            fc_disp = "".join(chr(c) if 32 <= c < 127 else "?"
                              for c in ((fc >> 24) & 0xFF, (fc >> 16) & 0xFF,
                                        (fc >> 8) & 0xFF, fc & 0xFF))
            fps_rb = float(self._cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            fc_disp, fps_rb = "?", float("nan")
        print(f"Camera {self.camera_index}: backend={backend} "
              f"fourcc={fc_disp} res={actual_w}x{actual_h} "
              f"fps={fps_rb:.1f}", flush=True)
        self._set_state(CameraState.IDLE)
        self.camera_opened.emit(actual_w, actual_h, backend)
        return True

    def _release_camera(self):
        with QMutexLocker(self._mutex):
            if self._cap is not None:
                try: self._cap.release()
                except Exception: pass
                self._cap = None

    def _capture_loop(self):
        self._open_camera()
        display_interval = 1.0 / max(settings.DISPLAY_FPS_LIMIT, 1)
        last_emit = 0.0
        consecutive_failures = 0
        consecutive_open_failures = 0
        iteration_count = 0
        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
                cap = self._cap  # 局部引用，防止 stop() 并发释放
            if cap is None or not cap.isOpened():
                self._set_state(CameraState.ERROR)
                delay = self._reconnect_delay(consecutive_open_failures)
                time.sleep(delay)
                with QMutexLocker(self._mutex):
                    if not self._running: break
                if self._open_camera():
                    consecutive_open_failures = 0
                    consecutive_failures = 0
                    iteration_count = 0
                    time.sleep(0.1)
                else:
                    consecutive_open_failures += 1
                continue
            iteration_count += 1
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                consecutive_failures += 1
                threshold = self._INITIAL_FAIL_THRESHOLD if iteration_count <= self._INITIAL_GRACE_PERIOD else self._READ_FAIL_THRESHOLD
                if consecutive_failures >= threshold:
                    self._set_state(CameraState.ERROR)
                    self.error_occurred.emit(f"Camera {self.camera_index}: consecutive read failures, reconnecting")
                    self._release_camera()
                    consecutive_failures = 0
                    iteration_count = 0
                time.sleep(0.03)
                continue
            # 强制应用自动曝光。放在首帧读出成功之后：STREAMON 后相机才
            # 可靠接受 AE 控制，开流前的写入会被部分 UVC 相机重置/忽略
            # （重连后画面发暗）。仅在开启/重连后首帧执行一次。
            with QMutexLocker(self._exp_lock):
                pending = self._exp_pending
                self._exp_pending = False
            if pending:
                self._apply_exposure_to(cap, True, 0.0,
                                        backend=self._backend_name)
            # 帧采集成功，立即记录时间戳（微秒）
            capture_ts_us = int(time.time() * 1_000_000)
            with QMutexLocker(self._ts_lock):
                self._latest_capture_ts_us = capture_ts_us
            consecutive_failures = 0

            # B+ 方案: 将帧副本直接推入录制队列（采集线程 → 写入线程，无需主线程中转）
            if self._record_queue is not None:
                try:
                    self._record_queue.put_nowait(frame.copy())
                except queue.Full:
                    # 写入线程消费不及，丢弃旧帧，队列自然保持最新
                    if self._on_queue_full is not None:
                        try:
                            self._on_queue_full()
                        except Exception:
                            pass  # 统计回调失败不影响采集

            if self._state == CameraState.ERROR:
                self._set_state(CameraState.IDLE)
            self._fps_counter.tick()
            now = time.perf_counter()
            # 阈值留 10% 余量：采集帧率≈显示上限（30fps 采集 vs 30fps 上限）
            # 时，严格等号比较会因帧间隔抖动系统性跳过约一半帧（实测
            # 满分辨率 30fps 采集下显示仅 ~18fps）
            if now - last_emit >= display_interval * 0.9 and not self._paused:
                self.frame_ready.emit(frame.copy())
                self.fps_updated.emit(self._fps_counter.fps)
                last_emit = now
        self._release_camera()

    @staticmethod
    def _reconnect_delay(consecutive_failures: int) -> float:
        delay = CameraWorker._RECONNECT_BASE_DELAY * (2 ** consecutive_failures)
        return min(delay, CameraWorker._RECONNECT_MAX_DELAY)


class _FPSCounter:
    def __init__(self, window: int = 30):
        self._ticks = []; self._window = window; self.fps = 0.0
    def tick(self):
        now = time.perf_counter(); self._ticks.append(now)
        if len(self._ticks) > self._window: self._ticks.pop(0)
        if len(self._ticks) >= 2:
            elapsed = self._ticks[-1] - self._ticks[0]
            self.fps = (len(self._ticks) - 1) / elapsed if elapsed > 0 else 0.0
