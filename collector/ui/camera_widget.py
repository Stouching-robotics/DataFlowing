"""
摄像机画面控件 —— 显示一路摄像机的实时画面和录制控制按钮。

画面支持鼠标操作:
  - 滚轮缩放（以画面中心为基准）
  - 左键拖拽平移
  - 双击还原（重置缩放和平移）


"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter,
)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QToolButton,
)

from config import settings
from config.i18n import tr, lang_manager
from core.helpers import format_duration


# ═══════════════════════════════════════════════════════
#  可缩放拖拽的视频显示控件
# ═══════════════════════════════════════════════════════

class ZoomableVideoWidget(QWidget):
    """
    支持鼠标交互的视频画面控件。

    操作方式:
      - 滚轮  → 以画面中心为基准缩放（1.1× / 0.9× 每级）
      - 拖拽  → 平移画面
      - 双击  → 还原到适应窗口大小

    缩放范围: 0.1× ~ 10×
    """

    ZOOM_MIN = 0.25
    ZOOM_MAX = 8.0
    ZOOM_STEP = 1.08                # 每级滚轮倍率（更平滑）

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._dragging = False
        self._last_mouse_pos = QPointF()

        self._status_text = tr("无信号")
        self._has_frame = False

        # 缩放电量提示（右下角短暂显示）
        self._show_zoom_hint = False
        self._zoom_hint_timer = 0

        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#000000; border:none;")

    # ── 公开接口 ──────────────────────────────────────

    def reset_view(self):
        """Reset zoom and pan to default."""
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self.update()

    def set_frame(self, bgr_frame: np.ndarray, flip_vertical: bool = False):
        """传入 BGR 帧并触发重绘。

        flip_vertical: 是否上下翻转画面（摄像机视频需要，传感器渲染不需要）。
        """
        try:
            if flip_vertical:
                bgr_frame = cv2.flip(bgr_frame, 0)
            # BGR → RGB 后用 Format_RGB888（Format_BGR888 在 Qt x86 上红蓝互换）
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, w * ch, QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
            self._has_frame = True
        except Exception:
            pass
        if self._show_zoom_hint:
            self._zoom_hint_timer -= 1
            if self._zoom_hint_timer <= 0:
                self._show_zoom_hint = False
        self.update()

    def set_status_text(self, text: str):
        """设置无画面时显示的文字。"""
        self._status_text = text
        self._has_frame = False
        self._pixmap = None
        self.update()

    def reset_view(self):
        """还原缩放和平移到初始状态。"""
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self.update()

    # ── 绘制 ──────────────────────────────────────────

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        ww = self.width()
        wh = self.height()

        if self._pixmap is None or not self._has_frame:
            painter.fillRect(self.rect(), Qt.black)
            painter.setPen(QColor("#888888"))
            painter.setFont(QFont("Microsoft YaHei", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, self._status_text)
            return

        pw = self._pixmap.width()
        ph = self._pixmap.height()

        base_scale = min(ww / pw, wh / ph) if pw > 0 and ph > 0 else 1.0
        scale = base_scale * self._zoom

        sw = pw * scale
        sh = ph * scale

        cx = (ww - sw) / 2.0 + self._offset.x()
        cy = (wh - sh) / 2.0 + self._offset.y()

        target = QRectF(cx, cy, sw, sh)
        source = QRectF(0, 0, pw, ph)

        painter.drawPixmap(target, self._pixmap, source)

        # 短暂的缩放百分比提示（右下角）
        if self._show_zoom_hint:
            pct = int(self._zoom * 100)
            hint = f"{pct}%"
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(hint) + 16
            th = fm.height() + 8
            painter.drawRoundedRect(
                QRectF(ww - tw - 12, wh - th - 12, tw, th), 6, 6
            )
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.drawText(
                QRectF(ww - tw - 12, wh - th - 12, tw, th),
                Qt.AlignCenter, hint
            )

    # ── 鼠标事件 ──────────────────────────────────────

    def wheelEvent(self, event):
        """滚轮缩放 —— 以鼠标位置为基准点缩放。"""
        if self._pixmap is None or not self._has_frame:
            return

        mouse_pos = QPointF(event.pos())
        old_zoom = self._zoom

        if event.angleDelta().y() > 0:
            self._zoom = min(self.ZOOM_MAX, self._zoom * self.ZOOM_STEP)
        else:
            self._zoom = max(self.ZOOM_MIN, self._zoom / self.ZOOM_STEP)

        if abs(self._zoom - old_zoom) < 1e-6:
            return

        # 以鼠标位置为基准调整偏移
        ratio = self._zoom / old_zoom
        self._offset = mouse_pos - ratio * (mouse_pos - self._offset)

        # 显示缩放百分比提示（持续约 1 秒）
        self._show_zoom_hint = True
        self._zoom_hint_timer = 30   # ~30 帧 ≈ 1 秒

        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_mouse_pos = QPointF(event.pos())
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = QPointF(event.pos()) - self._last_mouse_pos
            self._offset += delta
            self._last_mouse_pos = QPointF(event.pos())
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.OpenHandCursor)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """双击还原缩放和平移。"""
        if event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()


# ═══════════════════════════════════════════════════════
#  摄像机面板（组合视频控件 + 录制控制栏）
# ═══════════════════════════════════════════════════════

class CameraWidget(QFrame):
    """
    单路摄像机面板：视频画面 + 状态覆盖条。

    信号
    ----
    remove_requested(str)   — 请求移除此摄像机
    exposure_clicked(str)   — 点击曝光按钮（主窗口按槽位所属设备弹对话框）
    """

    remove_requested = pyqtSignal(str)
    exposure_clicked = pyqtSignal(str)

    def __init__(self, slot_id: str, camera_name: str = "",
                 parent: QWidget = None):
        super().__init__(parent)
        self.slot_id = slot_id
        self.camera_name = camera_name or f"Camera {slot_id}"

        self._fps = 0.0

        self._setup_ui()

        # 监听语言切换
        lang_manager.language_changed.connect(self._refresh_texts)

    # ═══════════════════════════════════════════════════
    #  UI 构建
    # ═══════════════════════════════════════════════════

    def _setup_ui(self):
        self.setMinimumSize(settings.FEED_MIN_WIDTH, settings.FEED_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:{settings.COLOR_BG_WIDGET}; "
                           f"border:1px solid {settings.COLOR_BORDER};")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(0)

        # ── 视频显示区域（可缩放拖拽） ────────────────
        self.video_widget = ZoomableVideoWidget()
        main_layout.addWidget(self.video_widget, 1)

        # ── 覆盖条（视频画面顶部，始终固定在左上角） ───
        self.overlay = QFrame(self.video_widget)
        self.overlay.setStyleSheet(
            "background:rgba(0,0,0,160); border-radius:4px;"
        )
        self.overlay.setFixedHeight(28)
        # 注意：覆盖条绝不能设 WA_TransparentForMouseEvents——Qt 语义是
        # 「该控件及其全部子控件不接收鼠标」，会连 ☀ 按钮一起屏蔽。
        # 信息条拖拽由 CameraGrid 的事件过滤器统一拦截，无需穿透。
        overlay_layout = QHBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(8, 0, 8, 0)
        overlay_layout.setSpacing(12)

        self.name_label = QLabel(self.camera_name)
        self.name_label.setStyleSheet(
            "color:#FFFFFF; font-weight:bold; font-size:11px; background:transparent;"
        )
        overlay_layout.addWidget(self.name_label)

        # ☀ 曝光按钮（默认隐藏；主窗口只对设备的"主槽位"显示。
        # _no_drag 属性让网格拖拽过滤器放行，按下不会触发画面拖拽/双击还原）
        self.exposure_btn = QToolButton()
        self.exposure_btn.setText("☀")
        self.exposure_btn.setFixedSize(24, 22)
        self.exposure_btn.setCursor(Qt.PointingHandCursor)
        self.exposure_btn.setToolTip(tr("曝光设置"))
        self.exposure_btn._no_drag = True
        self.exposure_btn.setStyleSheet(
            "QToolButton{background:transparent; color:#FFD54F; border:none;"
            "font-size:13px; padding:0;}"
            "QToolButton:hover{background:rgba(255,255,255,40); border-radius:3px;}"
            "QToolButton:disabled{color:#666666;}"
        )
        self.exposure_btn.clicked.connect(
            lambda: self.exposure_clicked.emit(self.slot_id))
        self.exposure_btn.setVisible(False)
        overlay_layout.addWidget(self.exposure_btn)

        overlay_layout.addStretch()

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet(
            "color:#AAAAAA; font-size:10px; background:transparent;"
        )
        overlay_layout.addWidget(self.fps_label)

        self.state_dot = QLabel("●")
        self.state_dot.setStyleSheet(
            f"color:{settings.COLOR_STOPPED}; font-size:10px; background:transparent;"
        )
        overlay_layout.addWidget(self.state_dot)

        self.overlay.move(4, 4)
        self.overlay.setFixedWidth(260)

    # ═══════════════════════════════════════════════════
    #  公开接口
    # ═══════════════════════════════════════════════════

    def update_frame(self, bgr_frame: np.ndarray):
        """用相机采集的 BGR 帧更新视频显示（上下翻转适配摄像头安装方向）。"""
        self.video_widget.set_frame(bgr_frame, flip_vertical=True)

    def update_fps(self, fps: float):
        """更新 FPS 显示。"""
        self._fps = fps
        self.fps_label.setText(f"FPS: {fps:.0f}")

    def set_frame_number(self, n: int):
        """回放用：在信息条右侧显示当前帧号（复用 FPS 标签位）。"""
        self.fps_label.setText(f"#{n}")

    def set_camera_state(self, state: str):
        """根据相机状态更新指示灯颜色和视频区域文字。"""
        if state == "recording":
            self.state_dot.setStyleSheet(
                f"color:{settings.COLOR_RECORDING}; font-size:10px; background:transparent;"
            )
        elif state == "error":
            self.state_dot.setStyleSheet(
                f"color:{settings.COLOR_ABNORMAL}; font-size:10px; background:transparent;"
            )
        elif state == "disconnected":
            self.state_dot.setStyleSheet("color:#BDBDBD; font-size:10px; background:transparent;")
            self.video_widget.set_status_text(tr("已断开"))
        else:
            self.state_dot.setStyleSheet(
                f"color:{settings.COLOR_STOPPED}; font-size:10px; background:transparent;"
            )
            if state == "idle":
                self.video_widget.set_status_text(tr("等待中…"))

    # ── 曝光入口（☀ 按钮） ─────────────────────────────

    def set_exposure_button_visible(self, visible: bool):
        """显示/隐藏曝光按钮（只对设备的"主槽位"显示）。"""
        self.exposure_btn.setVisible(visible)

    def set_exposure_enabled(self, enabled: bool):
        """启用/禁用曝光按钮（录制中禁用）。"""
        self.exposure_btn.setEnabled(enabled)
        self.exposure_btn.setToolTip(
            tr("曝光设置") if enabled else tr("录制中不可调整曝光"))

    def _refresh_texts(self, _lang: str = ""):
        """语言切换时刷新文字。"""
        self.exposure_btn.setToolTip(
            tr("曝光设置") if self.exposure_btn.isEnabled()
            else tr("录制中不可调整曝光"))

    # ═══════════════════════════════════════════════════
    #  事件重写
    # ═══════════════════════════════════════════════════

    def resizeEvent(self, event):
        """窗口大小变化时调整覆盖条位置和宽度。"""
        super().resizeEvent(event)
        self.overlay.setFixedWidth(min(self.video_widget.width() - 8, 300))
        self.overlay.move(4, 4)
