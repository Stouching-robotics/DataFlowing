"""
设备检测面板 —— 已连接设备统一列表（UVC + D435 + S80M + 蓝牙）。

DeviceScanner 每 2s 扫描 → MainWindow 调 set_devices 重建列表。三组展示
（📷 相机 / 🧤 手套 / 🎧 其他蓝牙），"其他蓝牙"默认折叠、空组隐藏；每台
设备带独立开关（QCheckBox），勾选发射 device_toggled(DeviceInfo, bool)，
由 MainWindow 路由到主网格显示；双击条目弹命名框（持久化到
device_names.json），此后每次连接到该设备都显示用户命名；set_active_keys
多高亮当前显示中的设备；set_locked 在录制中禁用全部开关。
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QInputDialog, QAbstractItemView,
)

from config import settings
from config.i18n import tr
from config.settings import save_device_name

_ICON = {"uvc": "📹", "d435": "🔭", "s80m": "👁",
         "data_ble": "🧤", "ble": "🎧"}
_GROUP_ORDER = ["camera", "glove", "other_ble"]
_GROUP_TITLE = {"camera": "📷 相机", "glove": "🧤 手套",
                "other_ble": "🎧 其他蓝牙"}


class DevicePanel(QWidget):
    """左侧停靠的设备列表面板（分组 + 开关 + 双击命名）。"""

    device_toggled = pyqtSignal(object, bool)   # (DeviceInfo, checked)
    device_renamed = pyqtSignal(object, str)    # (DeviceInfo, new_name)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._hint = QLabel(tr("开关设备以显示画面"))
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            f"color:{settings.COLOR_TEXT_SECONDARY}; font-size:11px;")
        layout.addWidget(self._hint)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.setRootIsDecorated(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.itemExpanded.connect(
            lambda item: self._record_expansion(item, True))
        self._tree.itemCollapsed.connect(
            lambda item: self._record_expansion(item, False))
        layout.addWidget(self._tree, 1)

        self._refresh_hint = QLabel(tr("双击设备可重命名"))
        self._refresh_hint.setStyleSheet(
            f"color:{settings.COLOR_TEXT_HINT}; font-size:10px;")
        layout.addWidget(self._refresh_hint)

        self._items: dict = {}            # device key → QTreeWidgetItem
        self._last_check: dict = {}       # device key → 上次勾选状态（挡文字类变化）
        self._group_expanded = {"camera": True, "glove": True,
                                "other_ble": False}   # 组展开状态（重建保留）
        self._locked = False              # 录制中锁死开关

    # ── 对外接口 ─────────────────────────────────────

    def set_devices(self, devices: list):
        """blockSignals 下清空重建（勾选状态由 set_checked_keys 随后恢复）。"""
        self._tree.blockSignals(True)
        self._tree.clear()
        self._items = {}
        self._last_check = {}

        if not devices:
            placeholder = QTreeWidgetItem([tr("未检测到设备")])
            placeholder.setFlags(Qt.NoItemFlags)
            self._tree.addTopLevelItem(placeholder)
            self._tree.blockSignals(False)
            return

        by_group: dict = {}
        for dev in devices:
            by_group.setdefault(dev.group, []).append(dev)

        for group in _GROUP_ORDER:
            devs = by_group.get(group, [])
            if not devs:
                continue
            gitem = QTreeWidgetItem([tr(_GROUP_TITLE[group])])
            gitem.setFlags(Qt.ItemIsEnabled)
            gitem.setData(0, Qt.UserRole, ("group", group))
            self._tree.addTopLevelItem(gitem)
            for dev in devs:
                item = QTreeWidgetItem([self._row_text(dev)])
                item.setData(0, Qt.UserRole, dev)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable
                              | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Unchecked)
                item.setToolTip(0, dev.key)
                gitem.addChild(item)
                self._items[dev.key] = item
                self._last_check[dev.key] = False
            gitem.setExpanded(self._group_expanded.get(group, True))

        self._tree.blockSignals(False)
        if self._locked:
            self._apply_lock()

    def set_checked_keys(self, keys: set):
        """按 key 集合勾选设备（不发信号；设备不在列表中的 key 自然忽略）。"""
        self._tree.blockSignals(True)
        for key, item in self._items.items():
            checked = key in keys
            item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
            self._last_check[key] = checked
        self._tree.blockSignals(False)

    def set_checked(self, key: str, on: bool):
        """程序化勾选单个设备（不发信号，供失败回退）。"""
        item = self._items.get(key)
        if item is not None:
            self._tree.blockSignals(True)
            item.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            self._last_check[key] = on
            self._tree.blockSignals(False)

    def checked_keys(self) -> set:
        """当前勾选的设备 key 集合（开关状态 = UI 状态源）。"""
        return {key for key, item in self._items.items()
                if item.checkState(0) == Qt.Checked}

    def set_active_keys(self, keys: set):
        """高亮正在显示的设备（绿色前景），其余还原。空集合全还原。"""
        self._tree.blockSignals(True)
        for key, item in self._items.items():
            if key in keys:
                item.setForeground(0, QColor(settings.COLOR_STOPPED))
            else:
                item.setForeground(0, QColor(settings.COLOR_TEXT_PRIMARY))
        self._tree.blockSignals(False)

    def set_active_key(self, key):
        """兼容旧调用：单 key 高亮（None 全还原）。"""
        self.set_active_keys({key} if key else set())

    def set_locked(self, on: bool):
        """录制中锁死全部开关（灰显不可点，命名同步禁用）。"""
        self._locked = bool(on)
        self._apply_lock()
        self._tree.setToolTip(tr("录制中不可更改设备") if self._locked else "")

    def key_for_kind(self, kind: str):
        """返回列表中第一个 kind 匹配的设备 key（无则 None）。"""
        for key, item in self._items.items():
            dev = item.data(0, Qt.UserRole)
            if dev is not None and dev.kind == kind:
                return key
        return None

    def device_for_key(self, key: str):
        """返回列表中 key 对应的 DeviceInfo（无则 None）。"""
        item = self._items.get(key)
        return item.data(0, Qt.UserRole) if item else None

    def key_for_serial(self, serial: str):
        """按序列号找面板条目 key（多台 RealSense 时精确高亮；无则 None）。"""
        for key, item in self._items.items():
            dev = item.data(0, Qt.UserRole)
            if dev is not None and dev.serial and dev.serial == serial:
                return key
        return None

    def key_for_video_index(self, video_index: int):
        """返回列表中第一个 video_index 匹配的设备 key（无则 None）。"""
        for key, item in self._items.items():
            dev = item.data(0, Qt.UserRole)
            if dev is not None and getattr(dev, "video_index", -1) == video_index:
                return key
        return None

    def refresh_texts(self):
        """语言切换刷新提示文字、组标题与占位项。"""
        self._hint.setText(tr("开关设备以显示画面"))
        self._refresh_hint.setText(tr("双击设备可重命名"))
        self._tree.blockSignals(True)
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            if isinstance(data, tuple) and data[0] == "group":
                item.setText(0, tr(_GROUP_TITLE[data[1]]))
            elif data is None:
                item.setText(0, tr("未检测到设备"))
        self._tree.blockSignals(False)

    # ── 内部 ─────────────────────────────────────────

    def _row_text(self, dev) -> str:
        icon = _ICON.get(dev.kind, "📹")
        text = f"{icon} {dev.label}"
        if dev.serial:
            text += f" — {dev.serial}"
        return text

    def _record_expansion(self, item: QTreeWidgetItem, expanded: bool):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, tuple) and data[0] == "group":
            self._group_expanded[data[1]] = expanded

    def _apply_lock(self):
        """按 _locked 刷新所有子项 flags（锁 = 去 ItemIsEnabled）。"""
        for item in self._items.values():
            flags = (Qt.ItemIsEnabled | Qt.ItemIsSelectable
                     | Qt.ItemIsUserCheckable)
            if self._locked:
                flags &= ~Qt.ItemIsEnabled
            item.setFlags(flags)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """勾选状态变化 → device_toggled（文字变化时 checkState 不变，被挡掉）。"""
        if column != 0:
            return
        dev = item.data(0, Qt.UserRole)
        if dev is None or isinstance(dev, tuple):   # 组标题条目无开关
            return
        new = item.checkState(0) == Qt.Checked
        if self._last_check.get(dev.key) == new:
            return
        self._last_check[dev.key] = new
        self.device_toggled.emit(dev, new)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int):
        """双击条目 → 命名框 → 持久化 + 更新行文本 + device_renamed。"""
        dev = item.data(0, Qt.UserRole)
        # 组标题条目 UserRole 是 ("group", group) 元组，直接跳过；录制中禁用
        if dev is None or isinstance(dev, tuple) or self._locked:
            return
        current = dev.user_name or dev.display_name
        text, ok = QInputDialog.getText(
            self, tr("重命名设备"), tr("设备名称:"), text=current)
        if not ok or not text.strip():
            return
        name = text.strip()
        save_device_name(dev.stable_key, name)
        # 对话框打开期间 2s 轮询可能已重建列表、旧 item 被销毁 —— 必须用
        # 当前列表里的条目更新；设备已消失则仅持久化（重连后生效）
        cur = self.device_for_key(dev.key)
        if cur is None:
            return
        cur.user_name = name
        self._tree.blockSignals(True)
        self._items.get(dev.key).setText(0, self._row_text(cur))
        self._tree.blockSignals(False)
        self.device_renamed.emit(cur, name)
