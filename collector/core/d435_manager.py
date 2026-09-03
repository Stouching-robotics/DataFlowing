"""
D435 深度双目设备管理器 —— worker 生命周期与帧处理口径
（QtCore QObject + 信号，不依赖 QtWidgets）。

  - D435DeviceManager  每台一个条目：D435Worker 创建/接线/停止 +
                       帧处理口径（calib 首帧注入、深度热力图/EMA、
                       录制写入）
  - 帧经 frames_ready(slot, ndarray, object, object, str) 交主线程

worker 类由调用方传入（离线测试在 ui.main_window 模块 patch D435Worker
替身仍生效）；hw_ns 用 object 封装（PyQt5 qint32 封送截断坑，同
core.s80m_manager 口径）。
"""

from __future__ import annotations

from functools import partial

import cv2
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal

from config.i18n import tr
from core.depth_codec import depth_to_heatmap_bgr
from core.stereo_depth import DepthHeatmapSmoother


class D435DeviceManager(QObject):
    """D435 深度双目采集 worker 管理（每台一个条目）。

    信号
    ----
    frames_ready(str, np.ndarray, object, object, str)
        — (slot_id, frame, hw_ns, imu_samples, dev_key)，采集线程 → 主线程
    log(str)
        — 跨线程日志
    """

    frames_ready = pyqtSignal(str, np.ndarray, object, object, str)
    log = pyqtSignal(str)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline

    @staticmethod
    def new_entry(label: str, serial: str, rgb_slot: str, depth_slot: str,
                  near_mm: float, far_mm: float, smooth_k: int,
                  temporal_alpha: float) -> dict:
        """注册表条目骨架（worker 由 spawn 填充，形状与旧主窗口口径一致）。"""
        return {
            "kind": "d435",
            "slots": [rgb_slot, depth_slot],
            "label": label,
            "serial": serial,
            "rgb_slot": rgb_slot,
            "depth_slot": depth_slot,
            "worker": None,
            "calib_sent": False,
            "near_mm": near_mm,
            "far_mm": far_mm,
            "smooth_k": smooth_k,
            "temporal_alpha": temporal_alpha,
            "heat_smoother": None,
        }

    def spawn(self, dev_key: str, entry: dict, model_name: str,
              profile: dict, exposure, worker_cls) -> None:
        """创建 D435Worker 并接线（帧/错误信号 → manager 信号 → 主窗口）。

        进程内采集 worker（无 IMU；深度与左红外同 imager 出厂对齐）。
        持久化曝光随 worker 启动应用（流启动后由采集线程消费）。
        信号连接在 worker.start() 之前完成。
        """
        worker = worker_cls(
            width=profile["depth_resolution"][0],
            height=profile["depth_resolution"][1],
            fps=profile["fps"],
            rgb_width=profile["rgb_resolution"][0],
            rgb_height=profile["rgb_resolution"][1],
            serial=entry["serial"], model_name=model_name,
            parent=self,
            rgb_slot=entry["rgb_slot"], depth_slot=entry["depth_slot"],
            exposure=exposure)
        worker.frames_ready.connect(
            partial(self._relay_frames, dev_key=dev_key))
        worker.error_occurred.connect(
            lambda m: self.log.emit(tr("[RealSense 错误] {}", m)))
        worker.start()
        entry["worker"] = worker

    def _relay_frames(self, slot_id: str, frame, hardware_ns=0,
                      imu_samples=None, dev_key: str = None):
        self.frames_ready.emit(slot_id, frame, hardware_ns,
                               imu_samples, dev_key)

    def process_frame(self, entry: dict, slot_id: str, frame,
                      hardware_ns: int = 0, dev_key: str = None) -> tuple:
        """D435 帧处理口径 → 返回 (display_frame, is_depth)。

        calib 首帧注入（录制开始时写 calibration/head_stereo.json）；
        深度槽：规范码值 JET（core.depth_codec 量化域色标，与存储
        12-bit 灰度视频同构——near/far 线性色标已废弃；EMA 按型号
        配置，仅显示降噪，存储码值不受影响）+ 录制写原始 uint16
        毫米深度（12-bit 灰度 MP4：gray12le 对数码，管线内量化）；
        RGB 槽：原帧 + 录制写外部帧源。UI 侧只做 set_frame 与 FPS
        计数。
        """
        # 首帧后把标定送入管线（录制开始时写 calibration/head_stereo.json）
        if not entry["calib_sent"]:
            calib = entry["worker"].get_calibration()
            if calib is not None:
                self._pipeline.set_device_calibration(dev_key, calib)
                entry["calib_sent"] = True

        if slot_id == entry["depth_slot"]:
            # 规范码值 JET（与存储 12-bit 灰度视频同构，见 depth_codec）；
            # 可选中值预滤（仅显示降噪，不进入存储码值）
            disp = frame
            if entry["smooth_k"] and entry["smooth_k"] > 1:
                disp = cv2.medianBlur(frame, entry["smooth_k"])
            heat = depth_to_heatmap_bgr(disp)
            # 时域 EMA（按型号配置；仅可视化，存储码值不受影响）
            if entry["temporal_alpha"] > 0:
                if entry["heat_smoother"] is None:
                    entry["heat_smoother"] = DepthHeatmapSmoother(
                        entry["temporal_alpha"])
                heat = entry["heat_smoother"].update(heat)
            # 录制：原始 uint16 毫米深度 → 12-bit 灰度 MP4（gray12le
            # 对数码，量化在管线内完成）
            if self._pipeline.is_recording:
                self._pipeline.write_depth(frame, depth_slot=slot_id)
            return heat, True
        if self._pipeline.is_recording:
            self._pipeline.write_external_frame(
                slot_id, frame.copy(),
                hardware_ns=hardware_ns, imu_samples=None)
        return frame, False

    def close(self, entry: dict):
        """停止 D435 worker（信号先断，防迟到信号进槽）。"""
        worker = entry["worker"]
        try:
            worker.frames_ready.disconnect()
        except Exception:
            pass
        worker.stop()
        worker.deleteLater()
