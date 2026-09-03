"""
摄像机网格布局 —— 基于嵌套 QSplitter 的可拖拽多画面布局。

布局策略：
  1 路  → 填满区域
  2 路  → [A | B]  水平分割
  3 路  → [A | B] / [C]  左边上下两路，右边一路
  4 路  → [A | B] / [C | D]  两行两列
  5+ 路 → 按 sqrt 自动计算行列数，嵌套分割
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QScrollArea, QLabel, QApplication,
)

from config.i18n import tr
from ui.camera_widget import CameraWidget


# 每个画面顶部用作拖拽手柄的信息条高度（像素）
DRAG_STRIP_H = 32
# 拖拽悬停目标的边框高亮色
DRAG_TARGET_BORDER = "border:2px solid #4FC3F7;"
# 分割条宽度（像素）：画面间调大小的拖拽手柄，太细难以抓住
SPLITTER_HANDLE_WIDTH = 8
# 分割条样式：常显灰条 + 悬停/按住高亮，否则暗色主题下几乎不可见
SPLITTER_HANDLE_QSS = (
    "QSplitter::handle { background:#4A4A4A; }"
    "QSplitter::handle:hover { background:#4FC3F7; }"
    "QSplitter::handle:pressed { background:#29B6F6; }"
)


class CameraGrid(QScrollArea):
    """
    容纳并排列 CameraWidget 的可调网格容器。

    公开方法
    --------
    add_camera(slot_id, name)   → CameraWidget
    remove_camera(slot_id)
    camera_widget(slot_id)      → CameraWidget | None
    clear()

    拖拽调位：按住任一画面顶部信息条（高 DRAG_STRIP_H）拖到另一
    画面上松开，两画面位置互换（事件过滤器 + 鼠标抓取实现）。
    """

    def __init__(self, parent: QWidget = None, empty_text: Optional[str] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 样式由 Qt-Material 暗色主题接管
        self._container = QWidget()
        self.setWidget(self._container)

        # 空状态提示文案（默认主窗口摄像机提示；回放等场景可覆盖）
        self._empty_text = empty_text

        self._widgets: Dict[str, QWidget] = {}
        self._splitter: Optional[QSplitter] = None

        # ── 拖拽调位状态 ──────────────────────────────
        self._drag_src: Optional[QWidget] = None     # 正在拖拽的源画面
        self._drag_hover: Optional[QWidget] = None   # 当前悬停高亮目标
        self._cursor_overridden = False              # 是否已设置拖拽光标

        self._rebuild_layout()

    # ── 公开接口 ──────────────────────────────────────

    def add_camera(self, slot_id: str, camera_name: str = "") -> CameraWidget:
        """添加一个摄像机控件到网格中。"""
        return self.add_widget(slot_id, CameraWidget(slot_id, camera_name))

    def add_widget(self, slot_id: str, widget: QWidget) -> QWidget:
        """添加任意控件到网格（相机 / 手套仿生手掌 / 无数据占位等）。"""
        if slot_id in self._widgets:
            return self._widgets[slot_id]
        self._widgets[slot_id] = widget
        self._install_drag_filter(widget)
        self._rebuild_layout()
        return widget

    def remove_camera(self, slot_id: str):
        """从网格中移除指定摄像机控件。"""
        w = self._widgets.pop(slot_id, None)
        if w:
            self._end_drag()    # 移除画面时先复位拖拽状态（防抓取/光标残留）
            self._remove_drag_filter(w)
            w.setParent(None)
            w.deleteLater()
            self._rebuild_layout()

    def camera_widget(self, slot_id: str) -> Optional[CameraWidget]:
        """根据 slot_id 获取对应的 CameraWidget。"""
        return self._widgets.get(slot_id)

    def clear(self):
        """移除所有摄像机。"""
        for sid in list(self._widgets.keys()):
            self.remove_camera(sid)

    def widget_count(self) -> int:
        """当前摄像机数量。"""
        return len(self._widgets)

    def slot_ids(self) -> List[str]:
        """返回所有 slot_id 列表。"""
        return list(self._widgets.keys())

    # ── 拖拽调位 ──────────────────────────────────────

    def _install_drag_filter(self, w: QWidget):
        """给画面控件及其子控件安装事件过滤器（子控件接收鼠标事件）。"""
        w.installEventFilter(self)
        for child in w.findChildren(QWidget):
            child.installEventFilter(self)

    def _remove_drag_filter(self, w: QWidget):
        w.removeEventFilter(self)
        for child in w.findChildren(QWidget):
            child.removeEventFilter(self)

    def eventFilter(self, obj, ev):
        """拖拽手柄事件：顶部信息条按下 → 抓取鼠标拖拽 → 悬停高亮 → 松开调位。"""
        # 信息条内交互控件（曝光按钮等）标记 _no_drag，按下直接放行，
        # 不触发拖拽抓取/双击还原
        if getattr(obj, "_no_drag", False):
            return super().eventFilter(obj, ev)
        t = ev.type()
        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            src = self._root_widget(obj)
            if (src is not None and len(self._widgets) > 1
                    and self._in_drag_strip(src, ev.globalPos())):
                self._drag_src = src
                src.grabMouse()
                if not self._cursor_overridden:
                    QApplication.setOverrideCursor(Qt.ClosedHandCursor)
                    self._cursor_overridden = True
                return True   # 消费按下事件：顶部信息条内不触发缩放平移
        elif t == QEvent.MouseButtonDblClick and ev.button() == Qt.LeftButton:
            src = self._root_widget(obj)
            if src is not None and self._in_drag_strip(src, ev.globalPos()):
                return True   # 信息条内双击也不复位缩放
        elif t == QEvent.MouseMove and self._drag_src is not None:
            self._set_drag_hover(self._widget_at_global(ev.globalPos()))
        elif t == QEvent.MouseButtonRelease and self._drag_src is not None:
            src = self._drag_src
            target = self._widget_at_global(ev.globalPos())
            self._end_drag()
            if target is not None and target is not src:
                self._swap_widgets(src, target)
            return True
        return super().eventFilter(obj, ev)

    def _root_widget(self, obj) -> Optional[QWidget]:
        """事件来源控件 → 所属网格画面（沿父链找注册表中的控件）。"""
        w = obj
        while w is not None:
            if w in self._widgets.values():
                return w
            w = w.parentWidget()
        return None

    def _in_drag_strip(self, src: QWidget, gpos) -> bool:
        """全局坐标是否落在 src 画面顶部信息条内。"""
        return src.mapFromGlobal(gpos).y() < DRAG_STRIP_H

    def _widget_at_global(self, gpos) -> Optional[QWidget]:
        """全局坐标命中的画面控件（无则 None）。"""
        for w in self._widgets.values():
            if w.rect().contains(w.mapFromGlobal(gpos)):
                return w
        return None

    def _set_drag_hover(self, target: Optional[QWidget]):
        """高亮拖拽悬停目标（恢复旧目标样式）。"""
        if self._drag_hover is target:
            return
        if self._drag_hover is not None:
            self._drag_hover.setStyleSheet("")
        self._drag_hover = target
        if target is not None:
            target.setStyleSheet(DRAG_TARGET_BORDER)

    def _swap_widgets(self, src: QWidget, target: QWidget):
        """调换两画面在网格中的位置（字典顺序即布局顺序）。"""
        items = list(self._widgets.items())
        si = next(i for i, kv in enumerate(items) if kv[1] is src)
        ti = next(i for i, kv in enumerate(items) if kv[1] is target)
        items[si], items[ti] = items[ti], items[si]
        self._widgets = dict(items)
        self._rebuild_layout()

    def _end_drag(self):
        """复位拖拽状态（释放当前拖拽源的抓取/光标/高亮）。"""
        if self._drag_src is None:
            return
        try:
            self._drag_src.releaseMouse()
        except Exception:
            pass
        self._drag_src = None
        self._set_drag_hover(None)
        if self._cursor_overridden:
            QApplication.restoreOverrideCursor()
            self._cursor_overridden = False

    # ── 布局重建 ──────────────────────────────────────

    def _rebuild_layout(self):
        """根据当前摄像机数量重建分割器布局。"""

        # 移除旧的分割器
        if self._splitter:
            self._splitter.setParent(None)
            self._splitter.deleteLater()
            self._splitter = None

        # 移除容器上的旧布局（避免 QLayout 重复添加警告）
        old_layout = self._container.layout()
        if old_layout is not None:
            tmp = QWidget()
            tmp.setLayout(old_layout)
            tmp.deleteLater()

        n = len(self._widgets)

        if n == 0:
            layout = QVBoxLayout(self._container)
            layout.setContentsMargins(0, 0, 0, 0)
            hint = QLabel(self._empty_text if self._empty_text is not None
                          else tr('尚未检测到摄像机。\n请从左侧设备面板勾选要开启的设备。'))
            hint.setAlignment(Qt.AlignCenter)
            hint.setWordWrap(True)
            hint.setStyleSheet(
                "font-size:14px; background:transparent; border:none;"
            )
            layout.addWidget(hint)
            return

        widgets = list(self._widgets.values())
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        if rows == 1 and cols == 1:
            layout = QVBoxLayout(self._container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widgets[0])
            return

        # 垂直分割器：每行是一个水平分割器
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.setChildrenCollapsible(False)
        v_splitter.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        v_splitter.setStyleSheet(SPLITTER_HANDLE_QSS)

        idx = 0
        for row in range(rows):
            row_widgets = []
            for col in range(cols):
                if idx < n:
                    row_widgets.append(widgets[idx])
                    idx += 1

            if len(row_widgets) == 1:
                v_splitter.addWidget(row_widgets[0])
            else:
                h_splitter = QSplitter(Qt.Horizontal)
                h_splitter.setChildrenCollapsible(False)
                h_splitter.setHandleWidth(SPLITTER_HANDLE_WIDTH)
                h_splitter.setStyleSheet(SPLITTER_HANDLE_QSS)
                for w in row_widgets:
                    h_splitter.addWidget(w)
                v_splitter.addWidget(h_splitter)

        # 初始等分
        v_splitter.setSizes([1] * v_splitter.count())

        self._splitter = v_splitter
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(v_splitter)
