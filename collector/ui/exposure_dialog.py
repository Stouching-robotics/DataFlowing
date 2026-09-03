"""
每相机曝光设置对话框 —— 自动曝光开关 + 曝光值滑块。

范围/值语义由调用方（MainWindow）按设备类型注入：
  UVC:  V4L2 原始曝光值（量程随相机，通常 1..10000）
  D435/D405: µs（流启动后读 rs.option.exposure 量程）
  S80M: SDK 曝光值 1.0~885.0（与 yaml stereo_init_exposure 同单位）

交互：拖动滑块即时下发（apply_requested），勾选自动曝光立即生效；
数值标签随滑块实时更新。对话框为模态 exec_，只发参数不改持久化
（持久化由 MainWindow 统一处理）。

original 参数：设备「最一开始」的曝光基线 (auto, value)（开启设备时
应用任何设置之前读回）。传入后显示「恢复默认」按钮——无论怎么调整，
点击即可回到该基线（控件静默复位后下发一次，MainWindow 照常持久化）。
"""

from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout,
)

from config.i18n import tr


class ExposureDialog(QDialog):
    # 参数变化：apply(auto, value)（auto=True 时 value 忽略）
    apply_requested = pyqtSignal(bool, float)

    def __init__(self, parent, title: str, vmin: float, vmax: float,
                 value: float, auto: bool, decimals: int = 0,
                 original: Optional[tuple] = None):
        super().__init__(parent)
        self.setWindowTitle(tr("曝光设置") + " — " + title)
        self.setModal(True)
        self.setMinimumWidth(320)
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._decimals = int(decimals)
        self._auto = bool(auto)
        self._value = float(value)
        self._original = (bool(original[0]), float(original[1])) \
            if original is not None else None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._auto_cb = QCheckBox(tr("自动曝光"))
        self._auto_cb.setChecked(self._auto)
        self._auto_cb.toggled.connect(self._on_auto_toggled)
        layout.addWidget(self._auto_cb)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(self._to_ticks(self._value))
        self._slider.setEnabled(not self._auto)
        self._slider.sliderMoved.connect(self._on_moved)
        self._slider.sliderReleased.connect(self._on_released)
        row.addWidget(self._slider, 1)
        self._value_label = QLabel(self._fmt(self._value))
        self._value_label.setMinimumWidth(70)
        self._value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self._value_label)
        layout.addLayout(row)

        hint = QLabel(tr("拖动滑块即时生效"))
        hint.setStyleSheet("color:#888888; font-size:10px;")
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        self._reset_btn = QPushButton(tr("恢复默认"))
        self._reset_btn.setToolTip(tr("回到相机最一开始的曝光设置"))
        self._reset_btn.setVisible(self._original is not None)
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)
        close_btn = QPushButton(tr("关闭"))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ── 内部映射与事件 ────────────────────────────────

    def _to_ticks(self, value: float) -> int:
        span = self._vmax - self._vmin
        if span <= 0:
            return 0
        return int(round((float(value) - self._vmin) / span * 1000))

    def _from_ticks(self, ticks: int) -> float:
        return self._vmin + (self._vmax - self._vmin) * ticks / 1000.0

    def _fmt(self, value: float) -> str:
        return f"{value:.{self._decimals}f}"

    def _on_auto_toggled(self, checked: bool):
        self._auto = checked
        self._slider.setEnabled(not checked)
        self.apply_requested.emit(checked, self._value)

    def _on_moved(self, ticks: int):
        if self._auto:
            return
        self._value = self._from_ticks(ticks)
        self._value_label.setText(self._fmt(self._value))
        self.apply_requested.emit(False, self._value)

    def _on_released(self):
        if self._auto:
            return
        self._value = self._from_ticks(self._slider.value())
        self._value_label.setText(self._fmt(self._value))
        self.apply_requested.emit(False, self._value)

    def _on_reset(self):
        """恢复默认：控件静默回到「最一开始」的曝光基线并下发一次。"""
        if self._original is None:
            return
        oa, ov = self._original
        self._auto = oa
        self._value = ov
        # 静默更新控件（toggled/slider 信号会重复下发）
        self._auto_cb.blockSignals(True)
        self._auto_cb.setChecked(oa)
        self._auto_cb.blockSignals(False)
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_ticks(ov))
        self._slider.blockSignals(False)
        self._slider.setEnabled(not oa)
        self._value_label.setText(self._fmt(ov))
        self.apply_requested.emit(oa, ov)
