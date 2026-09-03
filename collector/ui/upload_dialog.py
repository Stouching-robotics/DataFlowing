"""
上传对话框 —— 选择录制会话 → 一键打包上传到服务器。
"""

from __future__ import annotations
import os
import shutil
import threading
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QTreeWidget, QTreeWidgetItem,
    QProgressBar, QCheckBox, QMessageBox, QComboBox,
)

from config import settings
from config.i18n import tr
from core.uploader import UploadManager
from core.recording_repository import RecordingRepo
from core.session_catalog import list_recordings
from core.helpers import episode_file_suffix


class UploadDialog(QDialog):
    """录制数据一键上传对话框。"""

    # 后台删除线程 → 主线程（队列投递）
    _session_deleted = pyqtSignal(str, str, int)   # (session_path, error, episode_index)
    # 后台拉取项目列表 → 主线程
    _projects_loaded = pyqtSignal(list)       # [(project_id, project_name), ...]
    _projects_failed = pyqtSignal(str)        # error

    def __init__(self, parent=None, data_dir: str = "", session=None):
        super().__init__(parent)
        self.setWindowTitle(tr("上传录制数据"))
        self.resize(680, 560)
        # 普通窗口类型 + 最大化/最小化按钮：Dialog 类型在 GNOME/Mutter 等
        # WM 下按钮显示但点击无效（WM_TYPE=Dialog 禁最大/最小化）。
        # exec_() 的模态由 WA_ShowModal 提供，与窗口类型无关。
        self.setWindowFlags((self.windowFlags() & ~Qt.WindowType_Mask)
                            | Qt.Window
                            | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)
        self._data_dir = data_dir or settings.RECORDING_DIR
        self._shared_session = session  # 复用已认证的 requests.Session

        # 上传管理器（延迟初始化）
        self._manager: Optional[UploadManager] = None
        self._total_tasks = 0
        self._done_tasks = 0
        self._task_path_map: dict = {}   # task_id → session_path（删除用）
        self._session_deleted.connect(self._on_session_deleted)
        self._projects_loaded.connect(self._on_projects_loaded)
        self._projects_failed.connect(self._on_projects_failed)

        self._setup_ui()
        self._refresh_list()

        QTimer.singleShot(50, self._init_manager)
        QTimer.singleShot(200, self._refresh_projects)

    def _init_manager(self):
        url = self._url_edit.text().strip()
        self._manager = UploadManager(url, session=self._shared_session)
        self._manager.task_status.connect(self._on_status)
        self._manager.task_progress.connect(self._on_progress)
        self._manager.task_completed.connect(self._on_task_done)
        self._manager.task_failed.connect(self._on_task_failed)
        self._manager.all_completed.connect(self._on_all_done)

    # ═══════════════════════════════════════════════════
    #  UI
    # ═══════════════════════════════════════════════════

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── 服务器地址 ──────────────────────────────────
        server_group = QGroupBox(tr("服务器配置"))
        sg = QVBoxLayout(server_group)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("服务器地址:")))
        self._url_edit = QLineEdit(settings.load_server_url())
        self._url_edit.setPlaceholderText(settings.SERVER_URL)
        self._url_edit.editingFinished.connect(self._refresh_projects)
        row.addWidget(self._url_edit, 1)

        self._test_btn = QPushButton(tr("测试连接"))
        self._test_btn.clicked.connect(self._test_connection)
        row.addWidget(self._test_btn)
        sg.addLayout(row)

        # ── 目标项目（服务器按会话名自动匹配可能撞名 409，可显式指定）──
        row2 = QHBoxLayout()
        row2.addWidget(QLabel(tr("目标项目:")))
        self._project_combo = QComboBox()
        self._project_combo.addItem(tr("自动（服务器按名称匹配）"), None)
        self._project_combo.currentIndexChanged.connect(self._on_project_changed)
        row2.addWidget(self._project_combo, 1)

        self._project_refresh_btn = QPushButton(tr("刷新项目"))
        self._project_refresh_btn.clicked.connect(self._refresh_projects)
        row2.addWidget(self._project_refresh_btn)
        sg.addLayout(row2)

        layout.addWidget(server_group)

        # ── 会话列表 ────────────────────────────────────
        list_group = QGroupBox(tr("选择要上传的会话"))
        lg = QVBoxLayout(list_group)

        ctrl_bar = QHBoxLayout()
        self._select_all_cb = QCheckBox(tr("全选"))
        self._select_all_cb.toggled.connect(self._toggle_all)
        ctrl_bar.addWidget(self._select_all_cb)

        self._refresh_btn = QPushButton(tr("刷新"))
        self._refresh_btn.clicked.connect(self._refresh_list)
        ctrl_bar.addWidget(self._refresh_btn)

        ctrl_bar.addStretch()

        self._filter_up_btn = QPushButton(tr("⬆ 选中已上传"))
        self._filter_up_btn.clicked.connect(lambda: self._filter_uploaded(True))
        ctrl_bar.addWidget(self._filter_up_btn)

        self._filter_un_btn = QPushButton(tr("⬇ 选中未上传"))
        self._filter_un_btn.clicked.connect(lambda: self._filter_uploaded(False))
        ctrl_bar.addWidget(self._filter_un_btn)

        lg.addLayout(ctrl_bar)

        self._list = QTreeWidget()
        self._list.setHeaderHidden(True)
        self._list.setSelectionMode(QTreeWidget.ExtendedSelection)
        self._list.setStyleSheet("""
            QTreeWidget::indicator {
                width:18px; height:18px;
                border:2px solid #757575; border-radius:3px;
                background:#1E1E1E;
            }
            QTreeWidget::indicator:checked {
                background:#4CAF50;
                border-color:#4CAF50;
            }
            QTreeWidget::indicator:hover {
                border-color:#BDBDBD;
            }
        """)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemChanged.connect(self._on_item_changed)
        lg.addWidget(self._list, 1)
        layout.addWidget(list_group, 1)

        # ── 进度 ────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel(tr("就绪"))
        self._status_label.setStyleSheet("color:#757575; font-size:11px;")
        layout.addWidget(self._status_label)

        # ── 按钮 ────────────────────────────────────────
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self._upload_btn = QPushButton(tr("⬆ 一键上传"))
        self._upload_btn.setStyleSheet(
            f"QPushButton {{ background:{settings.COLOR_BTN_START}; "
            f"color:white; font-weight:bold; }}"
        )
        self._upload_btn.clicked.connect(self._start_upload)
        btn_bar.addWidget(self._upload_btn)

        close_btn = QPushButton(tr("关闭"))
        close_btn.clicked.connect(self.close)
        btn_bar.addWidget(close_btn)

        layout.addLayout(btn_bar)

    # ═══════════════════════════════════════════════════
    #  会话列表
    # ═══════════════════════════════════════════════════

    def _refresh_list(self):
        """两级树：顶层 = 任务目录名，子项 = episode-xxx（仅显示 episode
        文件名，不再重复长任务名）。"""
        self._list.clear()
        by_task = {}
        for s in list_recordings(self._data_dir):
            by_task.setdefault(s["path"], []).append(s)
        for task_dir, items in sorted(by_task.items(), reverse=True):
            parent = QTreeWidgetItem([os.path.basename(task_dir)])
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable
                            | Qt.ItemIsAutoTristate)
            self._list.addTopLevelItem(parent)
            for s in sorted(items, key=lambda x: x["episode_index"],
                            reverse=True):
                n = s.get("episode_index", 0)
                status = UploadManager.get_upload_status(s["path"], n)
                icon = {"completed": "✅", "failed": "❌",
                        "pending": "⬜"}.get(status, "⬜")
                child = QTreeWidgetItem(
                    [f"{icon} episode-{episode_file_suffix(n):03d}"])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setData(0, Qt.UserRole, s)
                child.setData(0, Qt.UserRole + 1, status)
                child.setCheckState(0, Qt.Checked
                                    if status != "completed" else Qt.Unchecked)
                parent.addChild(child)
        # blockSignals 防止 setChecked 触发 _toggle_all 把所有项取消勾选
        self._select_all_cb.blockSignals(True)
        self._select_all_cb.setChecked(False)
        self._select_all_cb.blockSignals(False)

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int = 0):
        """点击任务目录行 = 展开/收起该目录下的数据；点击 episode 行 =
        切换勾选状态（不必精确点复选框）。"""
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())
            return
        new_state = Qt.Unchecked if item.checkState(0) == Qt.Checked else Qt.Checked
        item.setCheckState(0, new_state)

    def _on_item_changed(self, item: QTreeWidgetItem, col: int):
        """父项勾选联动：勾/取任务目录 → 全选/全不选其下 episode；
        子项变化由 ItemIsAutoTristate 反向汇总父项（PartiallyChecked
        直接返回，不反向联动）。"""
        if col != 0 or item.childCount() == 0:
            return
        state = item.checkState(0)
        if state == Qt.PartiallyChecked:
            return
        self._list.blockSignals(True)
        try:
            for j in range(item.childCount()):
                item.child(j).setCheckState(0, state)
        finally:
            self._list.blockSignals(False)

    def _toggle_all(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self._list.topLevelItemCount()):
            parent = self._list.topLevelItem(i)
            for j in range(parent.childCount()):
                parent.child(j).setCheckState(0, state)

    def _filter_uploaded(self, uploaded: bool):
        """根据上传状态勾选各 episode（父项三态自动汇总）。"""
        target = "completed" if uploaded else "pending"
        for i in range(self._list.topLevelItemCount()):
            parent = self._list.topLevelItem(i)
            for j in range(parent.childCount()):
                item = parent.child(j)
                s = item.data(0, Qt.UserRole)
                current = UploadManager.get_upload_status(
                    s["path"], s.get("episode_index", 0))
                item.setCheckState(0, Qt.Checked
                                   if current == target else Qt.Unchecked)

    # ═══════════════════════════════════════════════════
    #  操作
    # ═══════════════════════════════════════════════════

    def _test_connection(self):
        url = self._url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, tr("提示"), tr("请输入服务器地址"))
            return

        from core.api_client import APIClient
        client = APIClient(url)
        if client.health_check():
            QMessageBox.information(self, tr("成功"), tr("服务器连接正常 ✅"))
        else:
            QMessageBox.warning(self, tr("失败"), tr("无法连接到服务器 ❌"))
        client.close()

    # ═══════════════════════════════════════════════════
    #  目标项目
    # ═══════════════════════════════════════════════════

    def _refresh_projects(self):
        """后台拉取服务器项目列表（复用已认证 session），经信号回主线程。"""
        url = self._url_edit.text().strip()
        if not url:
            return

        def _run():
            from core.api_client import APIClient
            client = APIClient(url, session=self._shared_session)
            try:
                projects = client.get_projects()
            except Exception as e:   # noqa: BLE001 — 后台线程兜底上报
                self._projects_failed.emit(str(e)[:200])
                return
            finally:
                client.close()
            self._projects_loaded.emit(
                [(p.get("id", ""), p.get("name", "")) for p in projects])
        threading.Thread(target=_run, daemon=True).start()

    def _on_projects_loaded(self, projects: list):
        """重建目标项目下拉框，恢复已保存的选择。"""
        saved = settings.load_upload_project_id()
        self._project_combo.blockSignals(True)
        try:
            self._project_combo.clear()
            self._project_combo.addItem(tr("自动（服务器按名称匹配）"), None)
            found = saved == ""
            for pid, pname in projects:
                if not pid:
                    continue
                self._project_combo.addItem(
                    f"{pname}  [{pid[:8]}…]" if pname else pid, pid)
                if pid == saved:
                    self._project_combo.setCurrentIndex(self._project_combo.count() - 1)
                    found = True
            if not found:
                # 已保存的项目不在列表里（可能被删）——保留一项让用户知情
                self._project_combo.addItem(
                    tr("⚠ 已配置项目不在列表中: {}…", saved[:12]), saved)
                self._project_combo.setCurrentIndex(self._project_combo.count() - 1)
        finally:
            self._project_combo.blockSignals(False)
        # 重建后 manager 同步一次（列表为空时保持原值即可）
        if projects and self._manager:
            self._manager.project_id = settings.load_upload_project_id()

    def _on_projects_failed(self, err: str):
        self._status_label.setText(
            tr("[提示] 拉取服务器项目列表失败: {}", err))

    def _on_project_changed(self, _idx: int):
        pid = self._project_combo.currentData() or ""
        settings.save_upload_project_id(pid)
        if self._manager:
            self._manager.project_id = pid

    def _start_upload(self):
        if not self._manager:
            return

        url = self._url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, tr("提示"), tr("请输入服务器地址"))
            return
        self._manager.server_url = url

        selected = []
        for i in range(self._list.topLevelItemCount()):
            parent = self._list.topLevelItem(i)
            for j in range(parent.childCount()):
                item = parent.child(j)
                if item.checkState(0) == Qt.Checked:
                    s = item.data(0, Qt.UserRole)
                    selected.append((s["path"], s.get("episode_index", 0)))

        if not selected:
            QMessageBox.information(self, tr("提示"), tr("请先勾选要上传的会话"))
            return

        # 入队并记录 task_id → (session_path, episode_index)
        # （上传成功且自动删除开关开启时删除本地 episode 文件组）
        valid = [(p, n) for (p, n) in selected if os.path.isdir(p)]
        self._task_path_map.update(
            dict(zip(self._manager.add_tasks(valid), valid)))
        self._manager.start()
        self._total_tasks = len(selected)
        self._done_tasks = 0
        self._upload_btn.setEnabled(False)
        self._status_label.setText(tr("已入队 {} 个任务，开始处理…", len(selected)))
        self._progress_bar.setValue(0)

    def _on_status(self, task_id: str, msg: str):
        """串行处理时状态栏显示：第几条 + 会话名 + 当前正在打包/上传什么。"""
        pair = self._task_path_map.get(task_id, ("", 0))
        name = self._pair_name(pair)
        if self._total_tasks > 1:
            cur = min(self._done_tasks + 1, self._total_tasks)
            head = tr("第 {}/{} 条", cur, self._total_tasks)
        else:
            head = ""
        if name:
            head = f"{head} [{name}]" if head else f"[{name}]"
        self._status_label.setText(f"{head} {msg}".strip() if head else msg)

    @staticmethod
    def _pair_name(pair) -> str:
        """(path, episode_index) → 显示名（任务目录/episode-xxx，对齐本地文件）。"""
        path, n = pair
        if not path:
            return ""
        name = os.path.basename(path)
        return f"{name}/episode-{episode_file_suffix(n):03d}" if n > 0 else name

    def _on_progress(self, task_id: str, ratio: float):
        if self._total_tasks > 0:
            overall = int((self._done_tasks + ratio) / self._total_tasks * 100)
            self._progress_bar.setValue(min(overall, 100))

    def _on_task_done(self, task_id: str):
        self._done_tasks += 1
        self._refresh_list()
        # 手动上传同样遵循"上传后自动删除"开关（与主窗口自动上传一致）；
        # 不删除时把录制行标为「已上传」（本地保留），历史面板可见
        pair = self._task_path_map.pop(task_id, ("", 0))
        path, n = pair
        name = self._pair_name(pair) or task_id
        if self._total_tasks > 1:
            self._status_label.setText(
                tr("✅ 第 {}/{} 条上传完成: {}", self._done_tasks,
                   self._total_tasks, name))
        else:
            self._status_label.setText(tr("✅ 上传完成: {}", name))
        if path and settings.UPLOAD_DELETE_AFTER:
            self._delete_after_upload(path, n)
        else:
            RecordingRepo.mark_uploaded(path, episode_index=n)
            self._parent_log(tr("☁ 上传完成（本地保留）: {}", name))

    def _delete_after_upload(self, session_path: str, episode_index: int):
        """后台线程删除本地上传件：池化 = 彻底删除该 episode 文件组
        （不走 _trash），旧格式会话目录 = 整目录删除。完成后经信号回
        主线程收尾。"""
        def _run():
            err = ""
            try:
                if episode_index > 0:
                    from core.helpers import delete_pooled_episode
                    delete_pooled_episode(session_path, episode_index)
                else:
                    shutil.rmtree(session_path)
            except Exception as e:   # noqa: BLE001 — 收尾线程兜底上报
                err = str(e)
            self._session_deleted.emit(session_path, err, episode_index)
        threading.Thread(target=_run, daemon=True).start()

    def _on_session_deleted(self, session_path: str, err: str, episode_index: int):
        """删除收尾：失败只提示；成功标记录制记录并刷新列表。"""
        name = self._pair_name((session_path, episode_index))
        if err:
            self._status_label.setText(
                tr("[错误] 上传后自动删除失败: {}", f"{name} ({err})"))
            return
        RecordingRepo.mark_uploaded_deleted(
            session_path, episode_index=episode_index)
        self._refresh_list()
        self._status_label.setText(
            tr("☁ 上传完成，本地文件已删除: {}", name))
        self._parent_log(tr("☁ 上传完成，本地文件已删除: {}", name))

    def _parent_log(self, msg: str):
        """把上传结果写进主窗口日志面板（手动上传与自动上传同一口径）。"""
        log = getattr(self.parent(), "_log", None)
        if callable(log):
            log(msg)

    def _on_task_failed(self, task_id: str, error: str):
        name = self._pair_name(self._task_path_map.get(task_id, ("", 0))) or task_id
        self._status_label.setText(tr("❌ 上传失败: {}", f"{name}（{error}）"))
        self._parent_log(tr("[上传失败] {}: {}", name, error))
        self._refresh_list()

    def _on_all_done(self):
        self._upload_btn.setEnabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText(tr("全部完成"))
        self._refresh_list()

    def closeEvent(self, event):
        if self._manager:
            self._manager.stop()
        event.accept()
