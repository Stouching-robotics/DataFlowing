"""手套传感器控件 —— 仿生手掌画面，直接嵌入主网格（统一设备体系）。

替代旧底部传感器 dock：面板开关打开手套 → 主网格出现仿生手掌渲染，
录制时数据经 pipeline.write_sensor 写入 parquet 对应传感器列。
"""

from __future__ import annotations
from typing import Optional
import time

import numpy as np
from PyQt5.QtCore import QTimer

from config.i18n import tr
from core.ble_engine import SensorBLEEngine
from core.render_engine import render_hand
from core.sensor_hand_config import load_sensor_hand_config
from ui.camera_widget import CameraWidget


class GloveWidget(CameraWidget):
    """仿生手掌实时画面（固定 hand 渲染模式，复用 CameraWidget 覆盖条）。"""

    # 固定渲染画布尺寸（仿生手掌锚点基于 1280×720 设计，与旧面板一致）
    _RENDER_W = 1280
    _RENDER_H = 720

    def __init__(self, slot_id: str, address: str, sensor_column: str,
                 label: str = "", parent=None):
        super().__init__(slot_id, label or sensor_column, parent)
        self.address = address
        self.sensor_column = sensor_column

        self._engine: Optional[SensorBLEEngine] = None
        self._running = False
        self._pipeline = None
        self._current_vmax = 5000.0

        # 左/右手套使用不同仿生手掌映射配置（口径在
        # core.sensor_hand_config，与回放对话框共用同一份）
        self.hand_config = load_sensor_hand_config(sensor_column)

        # 渲染定时器（30ms ≈ 30fps；未连接时 tick 直接返回）
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_tick)
        self._render_timer.start(30)

    # ── 连接控制 ──────────────────────────────────────

    def start(self, address: str = ""):
        """连接 BLE 设备并开始渲染。"""
        if address:
            self.address = address
        self.video_widget.set_status_text(tr("连接中…"))
        if self._engine is None:
            self._engine = SensorBLEEngine()
            self._engine.connected.connect(self._on_connected)
            self._engine.disconnected.connect(self._on_disconnected)
            self._engine.fps_updated.connect(self._on_fps)
            self._engine.calibration_progress.connect(self._on_calib_progress)
            self._engine.error_occurred.connect(self._on_error)
        self._engine.connect_device(self.address)

    def stop(self):
        """断开连接、停止渲染。"""
        if self._engine:
            self._engine.disconnect()
        self._running = False
        self.video_widget.set_status_text(tr("已断开"))

    def set_pipeline(self, pipeline):
        """由主窗口注入/清除当前录制管线引用。"""
        self._pipeline = pipeline

    # ── 引擎信号 ──────────────────────────────────────

    def _on_connected(self, addr: str):
        self._running = True
        self.video_widget.set_status_text(tr("已连接: {}…", addr[:12]))
        if self._pipeline:
            self._pipeline.record_event(self.sensor_column, "connected")

    def _on_disconnected(self):
        self._running = False
        self.video_widget.set_status_text(tr("已断开"))
        if self._pipeline:
            self._pipeline.record_event(self.sensor_column, "disconnected")

    def _on_error(self, msg: str):
        """引擎错误显示到画面状态栏（失败原因不再只有控制台可见）。"""
        self.video_widget.set_status_text(tr("⚠ {}", msg))

    def _on_fps(self, fps: float):
        self.fps_label.setText(f"HW: {fps:.0f}")

    def _on_calib_progress(self, progress: int):
        self.video_widget.set_status_text(
            tr("校准完成") if progress >= 100 else tr("校准中… {}%", progress))

    # ── 渲染循环（仿生手掌，固定模式） ──────────────────

    def _render_tick(self):
        """定时器触发：处理一帧并更新画面（逻辑平移自 SensorPanel）。"""
        if self._engine is None or not self._running:
            return

        processed, max_signal = self._engine.process_frame()

        if processed is None:
            if self._engine.is_calibrating:
                self.video_widget.set_status_text(tr("校准中…"))
            return

        # 写入共享录制会话（→ parquet observation.<sensor_column>）
        if self._pipeline is not None and self._pipeline.is_recording:
            capture_ts = self._engine.latest_data_ts_us
            self._pipeline.write_sensor(processed, capture_ts,
                                        sensor_name=self.sensor_column)

        fps = self._engine.hardware_fps
        gate = self._engine.base_noise_gate
        dyn = self._engine.dynamic_noise_ratio
        spatial = self._engine.spatial_filter_enabled

        try:
            frame, self._current_vmax = render_hand(
                processed, max_signal, self.hand_config,
                (self._RENDER_W, self._RENDER_H),
                self._current_vmax, fps, gate, dyn, spatial,
                self._engine.drift_baseline_val,
            )
        except Exception:
            import traceback
            print(f"[{self.sensor_column}] render error:")
            traceback.print_exc()
            return

        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        """渲染好的 BGR 帧 → 画面（叠加传感器名 + 数据帧龄诊断）。"""
        import cv2
        now_us = int(time.time() * 1_000_000)
        age_ms = (now_us - self._engine.latest_data_ts_us) / 1000.0
        hw_fps = self._engine.hardware_fps
        cv2.rectangle(frame, (0, 0), (260, 52), (0, 0, 0), -1)
        cv2.putText(frame, self.sensor_column, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'HW: {hw_fps:.0f} fps  Age: {age_ms:.0f} ms',
                    (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if age_ms < 100 else
                    (0, 255, 255) if age_ms < 300 else (0, 0, 255), 1)
        self.video_widget.set_frame(frame)
