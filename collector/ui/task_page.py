"""
Task selection page - shown on startup. User selects a task, then clicks
"Enter Collection" to switch to the data collection page.

All task data reads from data/tasks.json via storage.task_record.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
import os as _os
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QMessageBox, QMenu,
)

from config import settings
from config.i18n import tr, lang_manager
from core.task_record import (
    load_tasks, merge_backend_tasks, refresh_progress, mark_hidden,
    filter_by_identity, _tid,
)

_COL_NAME = 0
_COL_STATUS = 1
_COL_PROGRESS = 2
_COL_DATE = 3

# Status display config
_STATUS_CONFIG = {
    "pending":     {"label_key": "待采集",   "color": "#757575", "square": "■"},
    "in_progress": {"label_key": "采集中",   "color": "#42A5F5", "square": "■"},
    "completed":   {"label_key": "采集完成", "color": "#66BB6A", "square": "■"},
}


def _compute_status(task: dict) -> str:
    total = task.get("total_required", 0)
    completed = task.get("completed_count", 0)
    if total <= 0:
        return "pending"
    if completed >= total:
        return "completed"
    if completed > 0:
        return "in_progress"
    return "pending"


class TaskSelectionPage(QWidget):
    """Task selection page.

    Signals:
      task_selected(dict) — emitted when user clicks "Enter Collection".
      refresh_requested() — emitted when user clicks refresh.
    """

    task_selected = pyqtSignal(dict)
    refresh_requested = pyqtSignal()
    switch_account_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: list[dict] = []
        self._connected = False
        self._current_row = -1
        # 启动未选身份时按游客视图显示（仅公共任务）；登录/切换后由 set_identity 更新
        self._identity: str | None = "guest"

        self._setup_ui()

        self._tasks = filter_by_identity(load_tasks(), self._identity_scope())
        for t in self._tasks:
            t["status"] = _compute_status(t)
        self._rebuild_table()
        self._update_connection_ui()

        lang_manager.language_changed.connect(self._on_language_changed)

    # ── UI setup ───────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 28, 40, 20)
        root.setSpacing(12)

        # ── Header ──
        header = QHBoxLayout()
        logo = QLabel()
        logo_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "tools", "gongsitubiao.png")
        if _os.path.isfile(logo_path):
            pix = QPixmap(logo_path).scaledToHeight(36, Qt.SmoothTransformation)
            logo.setPixmap(pix)
            logo.setStyleSheet("background:transparent; border:none;")
            header.addWidget(logo)
            header.addSpacing(12)
        self._title_label = QLabel(settings.APP_NAME)
        self._title_label.setStyleSheet(
            f"font-size:22px; font-weight:bold; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; background:transparent; border:none;"
        )
        header.addWidget(self._title_label)
        header.addStretch()
        root.addLayout(header)

        # ── 服务器 / 身份栏 ──
        url_bar = QHBoxLayout()
        url_bar.setSpacing(6)

        self._server_label = QLabel(tr("服务器地址:"))
        self._server_label.setStyleSheet(
            f"color:{settings.COLOR_TEXT_SECONDARY}; font-size:11px; "
            f"background:transparent; border:none;"
        )
        url_bar.addWidget(self._server_label)

        self._server_host_label = QLabel(settings.load_server_url())
        self._server_host_label.setStyleSheet(
            f"color:{settings.COLOR_TEXT_PRIMARY}; font-size:11px; "
            f"background:transparent; border:none;"
        )
        url_bar.addWidget(self._server_host_label)

        url_bar.addSpacing(16)

        self._identity_caption_label = QLabel(tr("当前身份:"))
        self._identity_caption_label.setStyleSheet(
            f"color:{settings.COLOR_TEXT_SECONDARY}; font-size:11px; "
            f"background:transparent; border:none;"
        )
        url_bar.addWidget(self._identity_caption_label)

        self._identity_label = QLabel(tr("游客"))
        self._identity_label.setStyleSheet(
            f"color:{settings.COLOR_TEXT_PRIMARY}; font-weight:bold; font-size:11px; "
            f"background:transparent; border:none;"
        )
        url_bar.addWidget(self._identity_label)

        self._switch_btn = QPushButton(tr("切换账号"))
        self._switch_btn.setFixedHeight(28)
        self._switch_btn.setCursor(Qt.PointingHandCursor)
        self._switch_btn.setStyleSheet(
            f"QPushButton {{ background:{settings.COLOR_BTN_DEFAULT_BG}; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; "
            f"border:1px solid {settings.COLOR_BORDER}; "
            f"border-radius:4px; padding:4px 14px; font-size:11px; }}"
            f"QPushButton:hover {{ background:{settings.COLOR_BTN_HOVER}; }}"
        )
        self._switch_btn.clicked.connect(self._on_switch_clicked)
        url_bar.addWidget(self._switch_btn)

        url_bar.addStretch(1)

        self._conn_dot = QLabel("●")
        self._conn_dot.setFixedSize(14, 14)
        self._conn_dot.setAlignment(Qt.AlignCenter)
        url_bar.addWidget(self._conn_dot)

        self._conn_label = QLabel(tr("连接中…"))
        self._conn_label.setStyleSheet(
            f"color:{settings.COLOR_TEXT_SECONDARY}; font-size:11px; "
            f"background:transparent; border:none;"
        )
        url_bar.addWidget(self._conn_label)

        self._refresh_btn = QPushButton(tr("🔄 刷新"))
        self._refresh_btn.setFixedHeight(28)
        self._refresh_btn.setStyleSheet(
            f"QPushButton {{ background:{settings.COLOR_BTN_DEFAULT_BG}; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; "
            f"border:1px solid {settings.COLOR_BORDER}; "
            f"border-radius:4px; padding:4px 14px; font-size:11px; }}"
            f"QPushButton:hover {{ background:{settings.COLOR_BTN_HOVER}; }}"
        )
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        url_bar.addWidget(self._refresh_btn)

        root.addLayout(url_bar)

        # ── Task table (4 columns) ──
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels([
            tr("任务名称"), tr("状态"), tr("进度"), tr("发布时间"),
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setFocusPolicy(Qt.NoFocus)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)

        hdr = self._table.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(_COL_NAME, QHeaderView.Stretch)
        hdr.setSectionResizeMode(_COL_STATUS, QHeaderView.Fixed)
        hdr.setSectionResizeMode(_COL_PROGRESS, QHeaderView.Fixed)
        hdr.setSectionResizeMode(_COL_DATE, QHeaderView.Fixed)
        self._table.setColumnWidth(_COL_STATUS, 150)
        self._table.setColumnWidth(_COL_PROGRESS, 100)
        self._table.setColumnWidth(_COL_DATE, 110)

        self._table.setStyleSheet(
            f"QTableWidget {{ "
            f"background:{settings.COLOR_BG_PANEL}; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; "
            f"border:1px solid {settings.COLOR_BORDER}; border-radius:6px; "
            f"gridline-color:transparent; "
            f"selection-background-color:{settings.COLOR_BG_WIDGET}; }}"
            f"QTableWidget::item {{ "
            f"padding:8px 10px; "
            f"border-bottom:1px solid {settings.COLOR_BORDER}; }}"
            f"QTableWidget::item:selected {{ "
            f"background:{settings.COLOR_BG_WIDGET}; "
            f"border-left:3px solid #4CAF50; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; }}"
            f"QHeaderView::section {{ "
            f"background:{settings.COLOR_BG_MAIN}; "
            f"color:{settings.COLOR_TEXT_SECONDARY}; "
            f"font-size:11px; font-weight:bold; padding:8px 10px; "
            f"border:none; border-bottom:2px solid {settings.COLOR_BORDER_STRONG}; }}"
        )
        self._table.currentCellChanged.connect(self._on_cell_changed)
        self._table.cellDoubleClicked.connect(self._on_double_clicked)
        root.addWidget(self._table, 1)

        # ── Footer ──
        footer = QHBoxLayout()
        footer.setSpacing(16)

        self._selection_summary = QLabel("")
        self._selection_summary.setStyleSheet(
            f"color:{settings.COLOR_TEXT_SECONDARY}; font-size:12px; "
            f"background:transparent; border:none;"
        )
        footer.addWidget(self._selection_summary)
        footer.addStretch()

        self._enter_btn = QPushButton(tr("→ 进入采集"))
        self._enter_btn.setEnabled(False)
        self._enter_btn.setMinimumSize(150, 40)
        self._enter_btn.setCursor(Qt.PointingHandCursor)
        self._enter_btn.setStyleSheet(
            f"QPushButton {{ background:{settings.COLOR_BTN_START}; "
            f"color:white; font-weight:bold; font-size:14px; "
            f"border-radius:6px; padding:8px 28px; border:none; }}"
            f"QPushButton:hover {{ background:#388E3C; }}"
            f"QPushButton:disabled {{ "
            f"background:{settings.COLOR_BTN_DISABLED_BG}; "
            f"color:{settings.COLOR_BTN_DISABLED_TEXT}; }}"
        )
        self._enter_btn.clicked.connect(self._on_enter_clicked)
        footer.addWidget(self._enter_btn)

        self._delete_btn = QPushButton(tr("🗑 删除"))
        self._delete_btn.setEnabled(False)
        self._delete_btn.setMinimumSize(90, 40)
        self._delete_btn.setCursor(Qt.PointingHandCursor)
        self._delete_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:#D32F2F; "
            f"font-weight:bold; font-size:13px; "
            f"border:1px solid #D32F2F; border-radius:6px; padding:8px 16px; }}"
            f"QPushButton:hover {{ background:#3A1515; }}"
            f"QPushButton:disabled {{ "
            f"color:{settings.COLOR_BTN_DISABLED_TEXT}; "
            f"border-color:{settings.COLOR_BORDER}; }}"
        )
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        footer.addWidget(self._delete_btn)

        root.addLayout(footer)

    # ── Public API ─────────────────────────────────────

    def update_tasks(self, tasks: list[dict]):
        if tasks:
            merged = merge_backend_tasks(tasks, view_scope=self._identity_scope())
        else:
            merged = load_tasks()
        # 显示前按身份兜底过滤（决策：后端过滤为主，客户端防御性再滤一遍）
        self._tasks = filter_by_identity(merged, self._identity_scope())
        for t in self._tasks:
            t["status"] = _compute_status(t)
        self._rebuild_table()
        self._check_current_row()

    def update_task_progress(self, task_id: str, completed_count: int):
        self._tasks = filter_by_identity(refresh_progress(), self._identity_scope())
        for row, task in enumerate(self._tasks):
            if _tid(task) == task_id:
                task["status"] = _compute_status(task)
                self._update_row(row, task)
                if row == self._current_row:
                    name = task.get("name", task.get("task_name", ""))
                    self._selection_summary.setText(tr("已选择: {}", name))
                break

    def set_identity(self, identity: str | None):
        """更新当前身份（"guest" 或用户名）并即时按新身份过滤显示。"""
        self._identity = identity or "guest"
        self._update_identity_label()
        self._refilter()

    def set_server_display(self, url: str):
        """更新顶栏服务器地址显示（登录对话框改地址后由主窗口调用）。"""
        self._server_host_label.setText(url)
        self._server_host_label.setToolTip(url)

    def refresh_from_disk(self):
        """本地计数被进度上报回写更新后重读刷新（复用 _refilter）。

        不触发 merge/轮询 —— 走 update_tasks 会误触撤销清除逻辑，不可取。
        """
        self._refilter()

    def _identity_scope(self) -> str | None:
        """传给 task_record 的身份 scope：游客返回 "guest"，其余原样。"""
        return self._identity or None

    def _refilter(self):
        """本地全量条目按当前身份重滤重建表格（切换账号即时刷新，不等轮询）。"""
        self._tasks = filter_by_identity(load_tasks(), self._identity_scope())
        for t in self._tasks:
            t["status"] = _compute_status(t)
        self._rebuild_table()
        self._check_current_row()

    def _update_identity_label(self):
        if self._identity and self._identity != "guest":
            self._identity_label.setText(self._identity)
        else:
            self._identity_label.setText(tr("游客"))

    def set_connection_status(self, connected: bool):
        self._connected = connected
        self._update_connection_ui()

    def on_login_result(self, ok: bool, msg: str):
        """登陆结果反馈 — 更新状态提示。"""
        if ok:
            self._conn_label.setText(tr("登陆成功"))
        else:
            self._conn_label.setText(tr("登陆失败: {}", msg[:30]))

    def current_task(self) -> dict | None:
        if 0 <= self._current_row < len(self._tasks):
            return self._tasks[self._current_row]
        return None

    # ── Internal ───────────────────────────────────────

    def _update_connection_ui(self):
        if self._connected:
            self._conn_dot.setStyleSheet(
                f"color:{settings.COLOR_STOPPED}; font-size:14px; "
                f"background:transparent; border:none;")
            self._conn_label.setText(tr("后端已连接"))
        else:
            self._conn_dot.setStyleSheet(
                f"color:{settings.COLOR_TEXT_HINT}; font-size:14px; "
                f"background:transparent; border:none;")
            self._conn_label.setText(tr("后端未连接"))

    def _rebuild_table(self):
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._tasks))
        for row, task in enumerate(self._tasks):
            self._update_row(row, task)

    def _update_row(self, row: int, task: dict):
        if row >= self._table.rowCount():
            return

        # Col 0: name
        name = task.get("name", task.get("task_name", tr("未命名任务")))
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.UserRole, task)
        name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        f = name_item.font(); f.setPointSize(10); f.setBold(True); name_item.setFont(f)
        self._table.setItem(row, _COL_NAME, name_item)

        # Col 1: status badge (colored square + text)
        status = task.get("status", _compute_status(task))
        cfg = _STATUS_CONFIG.get(status, _STATUS_CONFIG["pending"])
        color = cfg["color"]
        label = tr(cfg["label_key"])
        html = '<span style="color:{}; font-size:14px;">{}</span>  {}'.format(
            color, cfg["square"], label)
        status_widget = QLabel(html)
        status_widget.setAlignment(Qt.AlignCenter)
        status_widget.setStyleSheet(
            "font-size:11px; font-weight:bold; "
            "color:{}; background:transparent; border:none; padding:2px 6px;".format(
                settings.COLOR_TEXT_PRIMARY)
        )
        self._table.setCellWidget(row, _COL_STATUS, status_widget)
        dummy = QTableWidgetItem()
        dummy.setFlags(Qt.NoItemFlags)
        self._table.setItem(row, _COL_STATUS, dummy)

        # Col 2: progress
        total = task.get("total_required", 0)
        completed = task.get("completed_count", 0)
        progress_item = QTableWidgetItem(f"{completed}/{total}")
        progress_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        progress_item.setFlags(progress_item.flags() & ~Qt.ItemIsEditable)
        if total <= 0:
            progress_item.setForeground(Qt.gray)
        elif completed >= total:
            progress_item.setForeground(Qt.green)
        elif completed > 0:
            progress_item.setForeground(QColor("#42A5F5"))
        else:
            progress_item.setForeground(Qt.gray)
        pf = progress_item.font(); pf.setBold(True); progress_item.setFont(pf)
        self._table.setItem(row, _COL_PROGRESS, progress_item)

        # Col 3: date
        created = task.get("assigned_at", task.get("created_at", ""))
        date_item = QTableWidgetItem(_format_date(created))
        date_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        date_item.setFlags(date_item.flags() & ~Qt.ItemIsEditable)
        date_item.setForeground(Qt.gray)
        self._table.setItem(row, _COL_DATE, date_item)

        self._table.setRowHeight(row, 40)

    def _on_cell_changed(self, row: int, _col: int, _pr: int, _pc: int):
        self._current_row = row
        if row < 0 or row >= len(self._tasks):
            self._enter_btn.setEnabled(False)
            self._selection_summary.setText("")
            return
        task = self._tasks[row]
        status = task.get("status", _compute_status(task))
        if status == "completed":
            self._enter_btn.setEnabled(False)
            self._enter_btn.setToolTip(tr("该任务已采集完成，请更换其他任务"))
        else:
            self._enter_btn.setEnabled(True)
            self._enter_btn.setToolTip("")
        name = task.get("name", task.get("task_name", ""))
        self._selection_summary.setText(tr("已选择: {}", name))

    def _on_double_clicked(self, _row: int, _col: int):
        if self._enter_btn.isEnabled():
            self._on_enter_clicked()

    def _on_enter_clicked(self):
        row = self._table.currentRow()
        if 0 <= row < len(self._tasks):
            task = self._tasks[row]
            status = task.get("status", _compute_status(task))
            if status == "completed":
                QMessageBox.warning(
                    self, tr("提示"),
                    tr("该任务已采集完成，请更换其他任务"),
                )
                return
            self.task_selected.emit(task)

    def _do_delete_task(self):
        """Execute task deletion (called by both button and right-click)."""
        row = self._table.currentRow()
        if row < 0 or row >= len(self._tasks):
            return
        task = self._tasks[row]
        name = task.get("name", task.get("task_name", ""))
        tid = _tid(task)
        reply = QMessageBox.question(
            self, tr("确认"),
            tr("确定要删除任务 \"{}\" 吗？\n删除后可在 tasks.json 中恢复。", name),
        )
        if reply == QMessageBox.Yes:
            mark_hidden(tid)
            self._tasks = filter_by_identity(load_tasks(), self._identity_scope())
            for t in self._tasks:
                t["status"] = _compute_status(t)
            self._rebuild_table()
            self._current_row = -1
            self._enter_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            self._selection_summary.setText("")

    def _on_delete_clicked(self):
        self._do_delete_task()

    def _on_context_menu(self, pos):
        """Right-click context menu."""
        row = self._table.rowAt(pos.y())
        if row < 0 or row >= len(self._tasks):
            return
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background:{settings.COLOR_BG_WIDGET}; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; border:1px solid {settings.COLOR_BORDER}; }}"
            f"QMenu::item:selected {{ background:{settings.COLOR_BTN_HOVER}; }}"
        )
        delete_action = menu.addAction(tr("🗑 删除任务"))
        action = menu.exec_(self._table.viewport().mapToGlobal(pos))
        if action == delete_action:
            self._do_delete_task()

    def _on_switch_clicked(self):
        self.switch_account_requested.emit()

    def _on_refresh_clicked(self):
        self._refresh_btn.setText(tr("刷新中…"))
        self._refresh_btn.setEnabled(False)
        self.refresh_requested.emit()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(1500, lambda: (
            self._refresh_btn.setText(tr("🔄 刷新")),
            self._refresh_btn.setEnabled(True),
        ))

    def _check_current_row(self):
        if self._current_row >= len(self._tasks):
            self._current_row = -1
            self._enter_btn.setEnabled(False)
            self._selection_summary.setText("")

    # ── Language ───────────────────────────────────────

    def _on_language_changed(self, lang: str):
        self._table.setHorizontalHeaderLabels([
            tr("任务名称"), tr("状态"), tr("进度"), tr("发布时间"),
        ])
        self._server_label.setText(tr("服务器地址:"))
        self._identity_caption_label.setText(tr("当前身份:"))
        self._update_identity_label()
        self._switch_btn.setText(tr("切换账号"))
        self._refresh_btn.setText(tr("🔄 刷新"))
        self._enter_btn.setText(tr("→ 进入采集"))
        self._delete_btn.setText(tr("🗑 删除"))
        self._update_connection_ui()
        # Rebuild to refresh status labels
        self._rebuild_table()
        if 0 <= self._current_row < len(self._tasks):
            name = self._tasks[self._current_row].get("name", "")
            self._selection_summary.setText(tr("已选择: {}", name))


# ── Helpers ────────────────────────────────────────────

def _tid(task: dict) -> str:
    return task.get("id", task.get("task_id", ""))


def _format_date(iso_str: str) -> str:
    if not iso_str:
        return ""
    return iso_str[:10]
