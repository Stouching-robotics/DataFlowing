"""
设备检测模块 —— 统一枚举已连接设备（UVC + RealSense D400 + S80M + 蓝牙）。

列表轮询走 sysfs 只读扫描（不 open 设备、不 test_read）——轮询 <5ms、
录制中安全、被占用的相机不会从列表闪烁消失；open 测试只发生在开关打开后
（CameraWorker / D435Worker / S80M 子进程 / SensorBLEEngine 各自带重连）。

蓝牙两个来源：bluetoothctl 已配对列表（快、只读、每轮可用）+ bleak 主动
发现（慢 ~5s，节流缓存；手套连接中建议 set_ble_scan_suppressed(True) 抑制
扫描，避免挤占 BLE 数据吞吐）。

用法:
    from core.device_detector import DeviceScanner, detect_devices

    scanner = DeviceScanner()
    scanner.scan_finished.connect(on_devices)   # list[DeviceInfo]
    scanner.request_scan()                      # 后台线程扫描
"""

from __future__ import annotations
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List

from PyQt5.QtCore import QObject, pyqtSignal

from config import settings
from core.camera import list_v4l_devices, _is_sdk_device


@dataclass
class DeviceInfo:
    """已连接设备的一条描述（跨线程经信号传回主线程）。"""
    key: str                          # "uvc:{by-id前缀或索引}" | "d435:{rs serial}" | "s80m:ftdi" | "ble:{MAC}"
    kind: str                         # "uvc" | "d435" | "s80m" | "data_ble" | "ble"
    display_name: str                 # 设备内部命名（by-id 解码 / rs 权威名 / 蓝牙广播名）
    serial: str = ""                  # 设备序号（无则为空串）
    video_index: int = -1             # UVC 设备的 /dev/videoN 索引（其它为 -1）
    by_id_path: Optional[str] = None  # /dev/v4l/by-id 永久路径（有则填）
    backend: str = ""                 # 打开后端（开关打开后由 CameraWorker 决定，枚举时为空）
    address: str = ""                 # BLE MAC 地址（大写、冒号分隔；非 BLE 为空串）
    rssi: int = 0                     # BLE 信号强度（排序用）
    user_name: str = ""               # 用户命名（枚举后由 MainWindow 从 device_names.json 填充）

    @property
    def stable_key(self) -> str:
        """持久化用 key：跨插拔稳定（=key）。"""
        return self.key

    @property
    def group(self) -> str:
        """面板分组: "camera" | "glove" | "other_ble"。"""
        if self.kind == "data_ble":
            return "glove"
        if self.kind == "ble":
            return "other_ble"
        return "camera"

    @property
    def label(self) -> str:
        """列表/画面叠加显示名：用户命名优先，回落内部命名。"""
        return self.user_name or self.display_name


def _parse_by_id_entry(entry: str) -> Optional[dict]:
    """解析 /dev/v4l/by-id 条目名 → {prefix, serial, index}。

    形如 usb-X_Y_Z-serial-video-index0；serial 为启发式（最后一段、
    纯字母数字且 ≥8 字符），无法解析返回 None。
    """
    if not entry:
        return None
    parts = entry.rsplit("-video-index", 1)
    prefix = parts[0] if len(parts) == 2 else entry
    index = int(parts[1]) if (len(parts) == 2 and parts[1].isdigit()) else None
    serial = ""
    if prefix.startswith("usb-"):
        segments = [s for s in re.split(r"[-_]", prefix[len("usb-"):]) if s]
        if segments and segments[-1].isalnum() and len(segments[-1]) >= 8:
            serial = segments[-1]
    return {"prefix": prefix, "serial": serial, "index": index}


_REALSENSE_NAME_KW = ("realsense",)


def _is_realsense_name(name: str) -> bool:
    """Windows DShow 设备名命中 RealSense 关键词（D400 全家族走
    pyrealsense2 专用通道，其 UVC 节点不能当普通相机列出）。"""
    nl = (name or "").lower()
    return any(kw in nl for kw in _REALSENSE_NAME_KW)


def _list_uvc_devices_windows(max_index: int) -> List[DeviceInfo]:
    """Windows UVC 枚举：pygrabber DirectShow 设备名。

    索引与 OpenCV CAP_DSHOW 顺序一致（两者都枚举同一个 DirectShow
    视频输入设备类），面板 video_index 可直接用于开关打开。
    pygrabber 缺失时退化按索引 DShow open 探测（只验证可打开不读帧）。
    注意：Windows 无 by-id/sysfs，key 用索引（重插拔索引可能变化）。
    """
    infos: List[DeviceInfo] = []
    names = None
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
    except Exception:
        names = None
    if names is not None:
        for i, name in enumerate(names):
            if i >= max_index:
                break
            if _is_realsense_name(name):
                continue
            if not (name or "").strip():
                name = f"USB Camera {i}"
            infos.append(DeviceInfo(
                key=f"uvc:{i}",
                kind="uvc",
                display_name=name.strip(),
                video_index=i,
            ))
        return infos
    # 兜底：pygrabber 未安装 → DShow 索引探测（只开不读，快速释放）
    import cv2
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok = cap.isOpened()
            cap.release()
        except Exception:
            ok = False
        if ok:
            infos.append(DeviceInfo(
                key=f"uvc:{i}",
                kind="uvc",
                display_name=f"USB Camera {i}",
                video_index=i,
            ))
    return infos


def _list_uvc_devices(max_index: int = settings.DEVICE_SCAN_MAX_INDEX) -> List[DeviceInfo]:
    """UVC 网络摄像头（排除 RealSense UVC 节点与 FTDI SDK 设备）。

    Linux: sysfs 只读枚举（/dev/v4l/by-id 分组，轮询安全）。
    Windows: pygrabber DirectShow 枚举（见 _list_uvc_devices_windows）。
    """
    if os.name == "nt":
        return _list_uvc_devices_windows(max_index)
    infos: List[DeviceInfo] = []
    for d in list_v4l_devices(max_index):
        if d.get("is_sdk"):
            continue
        if d.get("is_realsense"):
            continue
        # key 用 by-id 前缀（跨插拔稳定）；无 by-id 时退化为索引
        prefix = str(d["video_index"])
        if d.get("by_id_path"):
            prefix = os.path.basename(d["by_id_path"]).rsplit("-video-index", 1)[0]
        infos.append(DeviceInfo(
            key=f"uvc:{prefix}",
            kind="uvc",
            display_name=d["name"] or f"USB Camera {d['video_index']}",
            serial=d.get("serial", ""),
            video_index=d["video_index"],
            by_id_path=d.get("by_id_path"),
        ))
    return infos


def _list_d435_devices() -> List[DeviceInfo]:
    """pyrealsense2 权威枚举（名称 + 序列号）；无设备/未安装返回 []。"""
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        infos: List[DeviceInfo] = []
        for dev in ctx.query_devices():
            try:
                if "D400" not in dev.get_info(rs.camera_info.product_line):
                    continue
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                infos.append(DeviceInfo(
                    key=f"d435:{serial}",
                    kind="d435",
                    display_name=name or "Intel RealSense D435",
                    serial=serial,
                ))
            except Exception:
                continue
        return infos
    except Exception:
        return []


def _list_s80m_devices(max_index: int = settings.DEVICE_SCAN_MAX_INDEX) -> List[DeviceInfo]:
    """FTDI 命中一次即返回单条 S80M 条目（SDK 配置写死 video0/video2）。

    序号尝试从 by-id 条目解析（FT602 通常无序号 → 空串）。
    """
    for i in range(max_index):
        if not _is_sdk_device(i):
            continue
        serial, by_id_path = "", None
        by_id_dir = "/dev/v4l/by-id"
        if os.path.isdir(by_id_dir):
            for entry in sorted(os.listdir(by_id_dir)):
                p = os.path.join(by_id_dir, entry)
                try:
                    if os.readlink(p) == f"../../video{i}":
                        parsed = _parse_by_id_entry(entry)
                        if parsed:
                            serial = parsed.get("serial", "")
                        by_id_path = p
                        break
                except OSError:
                    continue
        return [DeviceInfo(
            key="s80m:ftdi",
            kind="s80m",
            display_name="FaysSense S80M",
            serial=serial,
            video_index=i,
            by_id_path=by_id_path,
        )]
    return []


# ── 蓝牙枚举 ──────────────────────────────────────────
BLE_DISCOVERY_INTERVAL_S = 20.0        # bleak 主动发现节流间隔
BLE_DISCOVERY_TIMEOUT_S = 5.0          # 单次 discover 超时
_ble_discovery_cache = {"ts": 0.0, "devices": []}   # [(name, mac, rssi)]
_ble_discovery_lock = threading.Lock()
_ble_scan_suppressed = False            # 手套连接中由 MainWindow 置 True


def set_ble_scan_suppressed(on: bool):
    """抑制 bleak 主动发现（手套连接中防扫描挤占数据吞吐）。

    只影响主动发现；bluetoothctl 已配对列表是只读系统调用，不受影响。
    """
    global _ble_scan_suppressed
    _ble_scan_suppressed = bool(on)


def _mac_norm(mac: str) -> str:
    """MAC 归一化：大写、冒号分隔（bluetoothctl 与 bleak 格式统一）。"""
    return re.sub(r"[^0-9A-Fa-f]", ":", mac or "").replace("::", ":").strip(":").upper()


def _ble_discover() -> List[tuple]:
    """bleak 主动发现（节流缓存；线程内调用，5s 阻塞）。"""
    now = time.monotonic()
    with _ble_discovery_lock:
        if now - _ble_discovery_cache["ts"] <= BLE_DISCOVERY_INTERVAL_S:
            return list(_ble_discovery_cache["devices"])
        _ble_discovery_cache["ts"] = now
    result: List[tuple] = []
    try:
        import asyncio
        from bleak import BleakScanner
        loop = asyncio.new_event_loop()
        try:
            found = loop.run_until_complete(
                BleakScanner.discover(timeout=BLE_DISCOVERY_TIMEOUT_S))
            for d in found:
                rssi = None
                try:
                    rssi = d.rssi
                except Exception:
                    rssi = getattr(getattr(d, "details", None), "rssi", None)
                result.append((d.name or "", _mac_norm(d.address),
                               int(rssi) if rssi is not None else -999))
        finally:
            loop.close()
    except Exception:
        result = []
    with _ble_discovery_lock:
        _ble_discovery_cache["devices"] = result
    return list(result)


def _bluetoothctl_paired() -> dict:
    """bluetoothctl devices 已配对列表（只读；不可用返回 {}）。"""
    paired: dict = {}
    try:
        out = subprocess.run(["bluetoothctl", "devices"], capture_output=True,
                             text=True, timeout=5)
        for line in out.stdout.splitlines():
            m = re.match(r"Device\s+(\S+)\s+(.+)$", line.strip())
            if m:
                paired[_mac_norm(m.group(1))] = m.group(2).strip()
    except Exception:
        pass
    return paired


_GLOVE_NAMES = {"l", "r", "left", "right", "l_glove", "r_glove",
                "left_glove", "right_glove"}


def _is_glove_name(name: str) -> bool:
    """广播名判手套：历史 "Matrix…" 或单字母 L/R（现役手套固件）。"""
    n = (name or "").strip().lower()
    return "matrix" in n or n in _GLOVE_NAMES


def _list_ble_devices() -> List[DeviceInfo]:
    """蓝牙设备：bluetoothctl 已配对 + bleak 主动发现，按 MAC 去重合并。

    广播名含 "Matrix"/"L"/"R" 判为手套（data_ble），其余为普通 BLE
    （other_ble）；device_names.json 里已绑定 sensor 列的 MAC 永远按手套
    对待（连过一次即持久化，改名/空名不丢）。手套连接中
    （_ble_scan_suppressed）跳过主动发现，防扫描挤占数据吞吐。
    """
    merged: dict = {}
    for mac, name in _bluetoothctl_paired().items():
        merged[_mac_norm(mac)] = {"name": name, "rssi": 0}
    if not _ble_scan_suppressed:
        for name, mac, rssi in _ble_discover():
            mac = _mac_norm(mac)
            if mac in merged:
                # 配对名优先；无配对名时用广播名
                if not merged[mac]["name"]:
                    merged[mac]["name"] = name
            else:
                merged[mac] = {"name": name, "rssi": rssi}
    infos: List[DeviceInfo] = []
    for mac, d in merged.items():
        name = d["name"] or "BLE Device"
        is_glove = _is_glove_name(name) or bool(
            settings.device_sensor_role(f"ble:{mac}"))
        infos.append(DeviceInfo(
            key=f"ble:{mac}",
            kind="data_ble" if is_glove else "ble",
            display_name=name,
            serial=mac,
            address=mac,
            rssi=d["rssi"],
        ))
    return infos


def detect_devices(max_index: int = settings.DEVICE_SCAN_MAX_INDEX) -> List[DeviceInfo]:
    """四段枚举（UVC + D435 + S80M + BLE），各自容错，整体不崩。"""
    devices: List[DeviceInfo] = []
    try:
        devices += _list_uvc_devices(max_index)
    except Exception:
        pass
    try:
        devices += _list_d435_devices()
    except Exception:
        pass
    try:
        devices += _list_s80m_devices(max_index)
    except Exception:
        pass
    try:
        devices += _list_ble_devices()
    except Exception:
        pass
    return devices


class DeviceScanner(QObject):
    """后台线程扫描设备（sysfs 只读），经排队信号回主线程。

    带 _busy 守卫：扫描未完成时 request_scan 直接返回，防止轮询堆积。
    """

    scan_finished = pyqtSignal(list)   # list[DeviceInfo]

    def __init__(self, parent=None, max_index: int = None):
        super().__init__(parent)
        self._max_index = (max_index if max_index is not None
                           else settings.DEVICE_SCAN_MAX_INDEX)
        self._busy = False
        self._busy_lock = threading.Lock()
        self._stop = False

    def request_scan(self):
        """请求一次扫描；进行中则忽略（防轮询堆积）。"""
        with self._busy_lock:
            if self._busy or self._stop:
                return
            self._busy = True
        threading.Thread(target=self._run_scan, daemon=True,
                         name="device-scan").start()

    def _run_scan(self):
        try:
            devices = detect_devices(self._max_index)
            if not self._stop:
                self.scan_finished.emit(devices)
        finally:
            with self._busy_lock:
                self._busy = False

    def stop(self):
        """停止接受新扫描（在途扫描完成后自然退出）。"""
        with self._busy_lock:
            self._stop = True
