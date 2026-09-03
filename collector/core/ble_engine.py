"""
BLE 传感器数据引擎 —— 蓝牙连接、数据接收、解析、降噪。

作为 QObject 运行，在独立线程中处理 BLE 异步通信，
通过 Qt 信号将处理后的数据帧发送到 UI 线程。
"""

import asyncio
import sys
import time
import threading
import logging
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from bleak import BleakClient, BleakScanner

# ── 配置常量 ──────────────────────────────────────────
TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
MATRIX_ROWS = 16
MATRIX_COLS = 16
CALIBRATION_FRAMES = 100
TARGET_FPS = 100

logger = logging.getLogger("SensorBLE")


class SensorBLEEngine(QObject):
    """
    BLE 传感器数据采集引擎。

    信号
    ----
    device_found(str, str, int)  — 设备名, 地址, RSSI
    scan_complete(list)          — 扫描完成，设备列表
    connected(str)               — 已连接，设备地址
    disconnected()               — 已断开
    data_ready(ndarray)          — 处理后的 16×16 数据帧
    fps_updated(float)           — 硬件 FPS
    error_occurred(str)          — 错误消息
    calibration_progress(int)    — 校准进度 (0-100)
    """

    device_found = pyqtSignal(str, str, int)
    scan_complete = pyqtSignal(list)
    connected = pyqtSignal(str)
    disconnected = pyqtSignal()
    data_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    calibration_progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.data_array = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype=np.float32)
        self.data_lock = threading.Lock()

        self.raw_buffer = bytearray()
        self.buffer_lock = threading.Lock()

        self._running = False
        self._scanning = False
        self._mutex = QMutex()

        # 降噪参数
        self.base_noise_gate = 500
        self.dynamic_noise_ratio = 0.0
        self.temporal_smooth = 0.15
        self.spatial_filter_enabled = True
        self.history_buffer = None
        self.drift_baseline_val = 0

        # 校准
        self.is_calibrating = True
        self.calibration_buffer = []
        self.baseline_map = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype=np.float32)

        # FPS
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.hardware_fps = 0.0

        # 最新传感器数据的采集时间戳（微秒），由解析线程写入，主线程读取
        self.latest_data_ts_us: int = 0
        self._ts_lock = threading.Lock()

    # ── 公开接口 ──────────────────────────────────────

    def start_scan(self):
        """开始扫描 BLE 设备。"""
        with QMutexLocker(self._mutex):
            if self._scanning:
                return
            self._scanning = True
        threading.Thread(target=self._scan_loop, daemon=True).start()

    def connect_device(self, address: str):
        """连接到指定地址的 BLE 设备。"""
        with QMutexLocker(self._mutex):
            self._running = True
        self._target_mac = address
        threading.Thread(target=self._ble_loop, daemon=True).start()
        threading.Thread(target=self._parse_loop, daemon=True).start()

    def disconnect(self):
        """断开 BLE 连接。"""
        with QMutexLocker(self._mutex):
            self._running = False

    def start_calibration(self):
        """开始校准。"""
        with self.data_lock:
            self.is_calibrating = True
            self.calibration_buffer = []

    def set_noise_gate(self, value: int):
        self.base_noise_gate = value

    def set_dynamic_ratio(self, value: float):
        self.dynamic_noise_ratio = value

    def set_spatial_filter(self, enabled: bool):
        self.spatial_filter_enabled = enabled

    def set_temporal_smooth(self, value: float):
        self.temporal_smooth = value

    # ── BLE 扫描 ──────────────────────────────────────

    def _scan_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            devices = loop.run_until_complete(BleakScanner.discover(timeout=5.0))
            devices.sort(
                key=lambda d: ("Matrix" in (d.name or ""), bool((d.name or "").strip()),
                               getattr(d.details, "rssi", -999)),
                reverse=True,
            )
            self.scan_complete.emit(devices)
        except Exception as e:
            self.error_occurred.emit(f"BLE 扫描失败: {e}")
        finally:
            loop.close()
            with QMutexLocker(self._mutex):
                self._scanning = False

    # ── BLE 通信 ──────────────────────────────────────

    def _ble_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ble_main_async())

    async def _ble_main_async(self):
        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
            try:
                async with BleakClient(self._target_mac) as client:
                    self.connected.emit(self._target_mac)
                    await client.start_notify(TX_CHAR_UUID, self._ble_notify_callback)
                    while client.is_connected:
                        with QMutexLocker(self._mutex):
                            if not self._running:
                                break
                        await asyncio.sleep(1)
            except Exception as e:
                # 连接/通知失败原因上抛 UI（否则界面停在"连接中"无从排查）
                self.error_occurred.emit(f"连接失败: {e}")
            self.disconnected.emit()
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
            await asyncio.sleep(3)

    def _ble_notify_callback(self, sender, data):
        with self.buffer_lock:
            self.raw_buffer.extend(data)

    # ── 数据解析 ──────────────────────────────────────

    def _parse_loop(self):
        frame_header = b"\xAA\x55"
        frame_tail = b"\xFB\x03"
        data_size = 512
        frame_size = len(frame_header) + data_size + 1 + len(frame_tail)

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break

            with self.buffer_lock:
                buffer_len = len(self.raw_buffer)

            if buffer_len < frame_size:
                time.sleep(0.001)
                continue

            with self.buffer_lock:
                h_idx = self.raw_buffer.find(frame_header)
                if h_idx == -1:
                    del self.raw_buffer[:-1]
                    continue
                if h_idx > 0:
                    del self.raw_buffer[:h_idx]
                if len(self.raw_buffer) < frame_size:
                    continue

                frame = self.raw_buffer[:frame_size]
                if frame[-2:] == frame_tail:
                    d_bytes = frame[2 : 2 + data_size]
                    chksum = frame[2 + data_size]
                    calc_chk = 0
                    for b in d_bytes:
                        calc_chk ^= b
                    if calc_chk == chksum:
                        new_data = (
                            np.frombuffer(d_bytes, dtype=np.uint16)
                            .reshape((16, 16))
                            .astype(np.float32)
                        )
                        with self.data_lock:
                            self.data_array[:] = new_data
                        # 记录数据采集时间戳（微秒）
                        with self._ts_lock:
                            self.latest_data_ts_us = int(time.time() * 1_000_000)
                        self.frame_count += 1
                    del self.raw_buffer[:frame_size]
                else:
                    del self.raw_buffer[:2]

            curr_time = time.time()
            if curr_time - self.last_fps_time >= 1.0:
                self.hardware_fps = self.frame_count / (
                    curr_time - self.last_fps_time
                )
                self.fps_updated.emit(self.hardware_fps)
                self.frame_count = 0
                self.last_fps_time = curr_time

    # ── 数据处理 ──────────────────────────────────────

    def process_frame(self) -> tuple:
        """
        处理一帧数据：校准、降噪、滤波。
        返回 (processed_data, max_signal)。
        """
        with self.data_lock:
            raw_data = self.data_array.copy()

        # 校准中
        if self.is_calibrating:
            self.calibration_buffer.append(raw_data)
            progress = min(100, len(self.calibration_buffer) * 100 // CALIBRATION_FRAMES)
            self.calibration_progress.emit(progress)
            if len(self.calibration_buffer) >= CALIBRATION_FRAMES:
                self.baseline_map = np.max(np.stack(self.calibration_buffer), axis=0) + 10
                self.is_calibrating = False
            return None, 0

        # 时域平滑
        if self.history_buffer is None:
            self.history_buffer = raw_data.copy()
        self.history_buffer = (
            self.history_buffer * self.temporal_smooth
            + raw_data * (1.0 - self.temporal_smooth)
        )
        smoothed = self.history_buffer.copy()

        # 去基线
        processed = np.maximum(0, smoothed - self.baseline_map)

        # 漂移补偿
        self.drift_baseline_val = np.percentile(processed, 40)
        if self.drift_baseline_val > 100:
            processed = np.maximum(0, processed - self.drift_baseline_val * 0.8)

        # 动态噪声门
        current_max = processed.max()
        dynamic_gate = min(current_max * self.dynamic_noise_ratio, 250)
        final_gate = max(self.base_noise_gate, dynamic_gate)
        processed[processed < final_gate] = 0

        # 空间滤波
        if self.spatial_filter_enabled:
            import cv2
            mask = (processed > 0).astype(np.uint8)
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
            neighbors = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            noise_floor = np.maximum(self.baseline_map * 2.5, 180)
            isolated_noise = (
                (mask > 0) & (neighbors == 0) & (processed < noise_floor)
            )
            processed[isolated_noise] = 0

        max_signal = processed.max()
        return processed, max_signal
