"""
本地录制回放对话框 —— 选择录制会话，同步播放全部摄像机视频 + 传感器数据。
时间戳对齐：通过 Parquet 统一时间线中的 frame_idx 和 timestamp_us。

会话扫描 / 元数据 / 主时钟帧率 / 后台加载器等算法代码位于 core/：
  core.session_catalog    get_effective_fps / load_session_meta / list_sessions
  core.session_loader     SessionLoader（worker 线程 parquet 读取）
  core.sensor_hand_config load_sensor_hand_config / valid_sensor_names
本文件只保留 Qt 窗口与播放控制逻辑。
"""

import os
import json
import time
import shutil
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QSlider, QSplitter, QFileDialog,
    QWidget, QFrame, QComboBox, QMessageBox, QSizePolicy,
    QCheckBox, QShortcut, QStyle, QStyleOptionSlider,
)

from config import settings
from config.i18n import tr
from core.helpers import (video_mp4_path, hand_kpts_parquet_path,
                          egodata_video_path, episode_video_files,
                          episode_file_suffix, delete_pooled_episode)
from ui.camera_widget import ZoomableVideoWidget
from ui.camera_grid import CameraGrid, SPLITTER_HANDLE_WIDTH, SPLITTER_HANDLE_QSS
from core.render_engine import (
    render_heatmap, render_trace, render_grid, render_hand, render_deform_mesh,
    DeformMeshState, clear_trace_canvas,
)
from core.session_timeline import SensorTimeline
from core.session_catalog import get_effective_fps, load_session_meta, list_sessions
from core.session_loader import SessionLoader
from core.sensor_hand_config import load_sensor_hand_config, valid_sensor_names
from core.depth_reader import Gray12DepthVideo

# 手部关键点叠加（可选模块；draw_kpts_overlay 在 core.hand_tracking）
try:
    from core.hand_tracking import load_hand_kpts, draw_kpts_overlay
    _HAND_KPTS_AVAILABLE = True
except ImportError:
    load_hand_kpts = None
    draw_kpts_overlay = None
    _HAND_KPTS_AVAILABLE = False

# 旧名兼容（tests/test_playback_multifps.py 等仍按旧名引用）
_get_effective_fps = get_effective_fps


class _SeekSlider(QSlider):
    """进度条：点击滑槽直接跳到对应位置（Qt 默认点击只翻一页）。
    点在手柄上仍走默认拖拽。"""

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        handle = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
        if handle.contains(event.pos()):
            super().mousePressEvent(event)   # 手柄：默认拖拽
            return
        groove = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        if groove.width() <= 0:
            super().mousePressEvent(event)
            return
        ratio = (event.pos().x() - groove.left()) / groove.width()
        ratio = max(0.0, min(1.0, ratio))
        self.setValue(self.minimum()
                      + round(ratio * (self.maximum() - self.minimum())))
        # 走既有链路：pressed（停表）→ released（单次 seek，原在播则恢复）
        self.sliderPressed.emit()
        self.sliderReleased.emit()
        event.accept()


class PlaybackSensorWidget(QFrame):
    """单路传感器回放格：标题 + 模式选择 + 画面（默认仿生手掌）+ TS 标签。

    与摄像机格同入 CameraGrid：顶部标题行即拖拽手柄（事件过滤器拦截
    <32px 区域）；mode_combo / ts_label 标记 _no_drag，按下直接放行
    不触发拖拽。
    """

    def __init__(self, sensor_name: str, title: str = "", parent=None):
        super().__init__(parent)
        self.sensor_name = sensor_name
        self.setMinimumSize(160, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:{settings.COLOR_BG_WIDGET}; "
                           f"border:1px solid {settings.COLOR_BORDER};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        # 头部行（≈32px，与 CameraGrid 拖拽手柄区对齐）
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title_label = QLabel(
            f"{title} ({sensor_name})" if title else sensor_name)
        self.title_label.setStyleSheet(
            "font-weight:bold; font-size:11px; background:transparent; border:none;")
        head.addWidget(self.title_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            tr("🔥 热力图 (Heatmap)"), tr("📝 轨迹 (Trace)"),
            tr("📊 网格 (Grid)"), tr("🦾 仿生手掌 (Hand)"),
            tr("🕸 形变网格 (Deform)"),
        ])
        # 默认仿生手掌（连接信号前设置，无需 blockSignals）
        self.mode_combo.setCurrentIndex(3)
        self.mode_combo.setFixedHeight(24)
        self.mode_combo._no_drag = True   # 网格拖拽过滤器放行点击
        head.addWidget(self.mode_combo)

        head.addStretch(1)
        self.ts_label = QLabel("")
        self.ts_label._no_drag = True
        self.ts_label.setStyleSheet(
            "font-size:10px; color:#AAAAAA; background:transparent; border:none;")
        head.addWidget(self.ts_label)
        lay.addLayout(head)

        self.video_widget = ZoomableVideoWidget()
        self.video_widget.setMinimumSize(160, 120)
        lay.addWidget(self.video_widget, 1)


class PlaybackDialog(QDialog):
    """录制回放对话框。"""

    def __init__(self, parent=None, data_dir: str = None):
        super().__init__(parent)
        self.setWindowTitle(tr("录制回放"))
        self.resize(1400, 800)
        # 普通窗口类型 + 最大化/最小化按钮：Dialog 类型在 GNOME/Mutter 等
        # WM 下按钮显示但点击无效（WM_TYPE=Dialog 禁最大/最小化）。
        # exec_() 的模态由 WA_ShowModal 提供，与窗口类型无关。
        # F11 全屏切换见 _toggle_fullscreen
        self.setWindowFlags((self.windowFlags() & ~Qt.WindowType_Mask)
                            | Qt.Window
                            | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)
        self._data_dir = data_dir or settings.RECORDING_DIR

        # 状态
        self._session_path = ""
        self._episode_index = 0                 # 池化布局的 episode 序号（>0 = pooled）
        self._timeline: SensorTimeline = None   # 统一时间线（后台加载）
        self._caps = {}             # slot_id → cv2.VideoCapture
        self._depth_videos = {}     # slot_id → Gray12DepthVideo（12-bit 灰 MP4）
        self._camera_ids = []       # 摄像机 ID 列表
        self._cam_fps = {}          # slot_id → 每路实际帧率（主时钟 = max）
        self._cam_total = {}        # slot_id → 每路总帧数
        self._last_read_frame = {}  # slot_id → 最近读出帧号（顺序读判定）
        self._slot_names = {}       # slot_id → 用户命名（叠加显示）
        self._sensor_titles = {}    # 传感器列名 → 用户命名
        self._sensor_names = []     # 传感器列名（info["sensors"] 动态）
        self._total_frames = 0
        self._play_idx = 0
        self._playing = False
        self._fps = 30
        self._speed = 1.0
        self._hand_kpts = {}        # frame_index → {hand_data, num_hands, track_ids}
        self._show_hand_kpts = True # 是否显示手部关键点叠加
        self._hand_processor = None # SessionHandProcessor（主线程创建，复用）

        # 后台加载与竞态防护
        self._loader = SessionLoader(self)
        self._loader.finished.connect(self._on_session_loaded)
        self._loader.failed.connect(self._on_session_load_failed)
        self._load_gen = 0          # 加载代号，过期结果丢弃
        self._loading_path = ""     # 正在加载的会话路径
        self._closing = False       # 对话框关闭中（丢弃迟到的回调）

        # 播放状态
        self._last_tick_ts = 0.0    # 上次 tick 时间（追赶用）
        self._slider_dragging = False
        self._pending_seek = -1
        self._slider_was_playing = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._setup_ui()
        self._refresh_list()

        # F11 全屏切换（WindowShortcut 上下文：对话框内任意子控件聚焦时均可用）
        self._fs_shortcut = QShortcut(Qt.Key_F11, self)
        self._fs_shortcut.activated.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        """F11 切换全屏。"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── 顶部：目录浏览 ────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel(tr("录制目录:")))
        self._dir_btn = QPushButton(self._data_dir)
        self._dir_btn.clicked.connect(self._browse_dir)
        top.addWidget(self._dir_btn, 1)
        self._refresh_btn = QPushButton(tr("刷新"))
        self._refresh_btn.clicked.connect(self._refresh_list)
        top.addWidget(self._refresh_btn)
        layout.addLayout(top)

        # ── 中部：会话列表 + 画面 ─────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # 左：会话列表
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel(tr("录制会话:")))

        # 会话列表（两级树：顶层任务目录 → 子项 episode-xxx；
        # 子项可勾选，用于批量删除）
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
        self._list.itemClicked.connect(self._on_session_clicked)
        self._list.itemChanged.connect(self._on_item_changed)
        left_layout.addWidget(self._list)

        # 筛选按钮行
        filter_bar = QHBoxLayout()
        self._filter_uploaded_btn = QPushButton(tr("⬆ 选中已上传"))
        self._filter_uploaded_btn.clicked.connect(lambda: self._filter_by_upload(True))
        self._filter_uploaded_btn.setToolTip(tr("仅勾选已上传的会话"))
        filter_bar.addWidget(self._filter_uploaded_btn)

        self._filter_unuploaded_btn = QPushButton(tr("⬇ 选中未上传"))
        self._filter_unuploaded_btn.clicked.connect(lambda: self._filter_by_upload(False))
        self._filter_unuploaded_btn.setToolTip(tr("仅勾选未上传的会话"))
        filter_bar.addWidget(self._filter_unuploaded_btn)

        self._delete_btn = QPushButton(tr("🗑 删除选中"))
        self._delete_btn.setStyleSheet(
            "QPushButton { color:#D32F2F; font-weight:bold; }"
        )
        self._delete_btn.clicked.connect(self._delete_selected)
        self._delete_btn.setToolTip(tr("永久删除选中的会话"))
        filter_bar.addStretch()
        filter_bar.addWidget(self._delete_btn)
        left_layout.addLayout(filter_bar)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("font-size:10px; padding:4px;")
        left_layout.addWidget(self._info_label)
        splitter.addWidget(left_panel)

        # 右：视频网格 + 传感器
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # 统一画面网格：摄像机 + 传感器同格（可拖拽调位 / 分割条调大小）
        self.grid = CameraGrid(empty_text=tr("选择会话开始回放"))
        right_layout.addWidget(self.grid, 1)

        # ── 传感器面板状态（按 info["sensors"] 动态创建，格子建在 _rebuild_grid） ──
        self._sensor_modes = []       # 每个传感器当前模式
        self._sensor_vmax_list = []   # 每个传感器当前 vmax
        self._sensor_mesh_states = [] # 每个传感器 mesh state
        self._sensor_widgets = []     # ZoomableVideoWidget 列表
        self._sensor_ts_labels = []   # TS 标签列表
        self._sensor_hand_configs = []  # 每个传感器的仿生手掌配置
        self._sensor_cells = []       # PlaybackSensorWidget 列表

        self._sensor_config = {
            "rows": list(range(16)), "cols": list(range(16)), "axis_order": "row_col",
        }

        splitter.addWidget(right_panel)
        splitter.setSizes([250, 1150])
        # 会话列表 ↔ 画面区之间的分割条同样加宽可见（与画面网格一致）
        splitter.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        splitter.setStyleSheet(SPLITTER_HANDLE_QSS)
        layout.addWidget(splitter, 1)

        # ── 底部：播放控制 ────────────────────────────
        ctrl = QHBoxLayout()

        self._play_btn = QPushButton(tr("▶ 播放"))
        self._play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._play_btn)

        self._prev_btn = QPushButton("⏮")
        self._prev_btn.setFixedWidth(40)
        self._prev_btn.clicked.connect(self._prev_frame)
        ctrl.addWidget(self._prev_btn)

        self._next_btn = QPushButton("⏭")
        self._next_btn.setFixedWidth(40)
        self._next_btn.clicked.connect(self._next_frame)
        ctrl.addWidget(self._next_btn)

        self._slider = _SeekSlider(Qt.Horizontal)
        # 拖动中不 seek（每秒数十个 move 事件 × 每次 90-180ms 会卡死），
        # 松手时单次 seek；拖动中只刷新时间标签。点击滑槽直接跳帧
        # （见 _SeekSlider.mousePressEvent）。
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._on_slider_released)
        ctrl.addWidget(self._slider, 1)

        self._time_label = QLabel("00:00 / 00:00")
        ctrl.addWidget(self._time_label)

        self._speed_btn = QPushButton("1×")
        self._speed_btn.clicked.connect(self._cycle_speed)
        self._speed_btn.setFixedWidth(56)   # 0.25× 等长文本不截断
        self._speed_btn.setCursor(Qt.PointingHandCursor)
        self._speed_btn.setToolTip(tr("循环切换播放速度 (0.25×-4×)"))
        self._speed_btn.setStyleSheet(
            f"QPushButton {{ background:{settings.COLOR_BTN_DEFAULT_BG}; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; font-weight:bold; "
            f"font-size:13px; border:1px solid {settings.COLOR_BORDER}; "
            f"border-radius:4px; padding:4px 8px; }}"
            f"QPushButton:hover {{ background:{settings.COLOR_BTN_HOVER}; }}")
        ctrl.addWidget(self._speed_btn)

        # ── 手部关键点叠加开关 ────────────────────────
        self._hand_overlay_cb = QCheckBox(tr("✋ 手部叠加"))
        self._hand_overlay_cb.setChecked(True)
        self._hand_overlay_cb.toggled.connect(self._toggle_hand_overlay)
        ctrl.addWidget(self._hand_overlay_cb)

        # ── 追踪模式选择 ──────────────────────────────
        self._hand_mode_combo = QComboBox()
        self._hand_mode_combo.addItems([tr("🧤 手套追踪"), tr("🖐 裸手追踪")])
        self._hand_mode_combo.setToolTip(tr("选择手部追踪模式"))
        self._hand_mode_combo.setMaximumWidth(120)
        self._hand_mode_combo.setStyleSheet("QComboBox { padding:2px 4px; }")
        ctrl.addWidget(self._hand_mode_combo)

        # ── 处理手部关键点按钮 ────────────────────────
        self._process_kpts_btn = QPushButton(tr("🔄 提取关键点"))
        self._process_kpts_btn.clicked.connect(self._process_current_kpts)
        self._process_kpts_btn.setToolTip(tr("后台处理当前视频，提取手部关键点"))
        ctrl.addWidget(self._process_kpts_btn)

        layout.addLayout(ctrl)

    # ── 传感器面板（动态） ──────────────────────────────

    # 旧名兼容（tests 与旧调用方按 PlaybackDialog._valid_sensor_names 引用）
    _load_sensor_hand_config = staticmethod(load_sensor_hand_config)
    _valid_sensor_names = staticmethod(valid_sensor_names)

    def _rebuild_grid(self):
        """按当前会话重建统一网格：先摄像机（双目四路保持 2×2 顺序），
        后传感器格（slot_id = "sensor:{name}"，与主窗口手套格约定一致）。

        摄像机格复用 CameraWidget（信息条含命名与帧号），传感器格用
        PlaybackSensorWidget（默认仿生手掌）。网格拖拽调位 / 分割条
        调大小由 CameraGrid 统一提供。
        """
        self.grid.clear()   # 逐格 remove → _end_drag() 复位拖拽状态

        # ── 摄像机格（顺序即布局顺序；保留双目四路特殊排列） ──
        _STEREO_IDS = {"stereo_left", "stereo_right",
                       "stereo_left_aux", "stereo_right_aux"}
        if _STEREO_IDS.issubset(set(self._camera_ids)):
            cam_list = [cid for cid in
                        ["stereo_left", "stereo_right",
                         "stereo_left_aux", "stereo_right_aux"]
                        if cid in self._camera_ids]
        else:
            cam_list = list(self._camera_ids)
        for slot_id in cam_list:
            name = self._slot_names.get(slot_id, "")
            title = f"{name} ({slot_id})" if name else slot_id
            w = self.grid.add_camera(slot_id, title)
            w.state_dot.setVisible(False)   # 回放无录制状态灯

        # ── 传感器格（保留原 _rebuild_sensor_panels 的列表语义） ──
        self._sensor_modes = []
        self._sensor_vmax_list = []
        self._sensor_mesh_states = []
        self._sensor_widgets = []
        self._sensor_ts_labels = []
        self._sensor_hand_configs = []
        self._sensor_cells = []
        for idx, sensor_name in enumerate(self._sensor_names):
            cell = PlaybackSensorWidget(
                sensor_name, self._sensor_titles.get(sensor_name))
            self.grid.add_widget(f"sensor:{sensor_name}", cell)
            cell.mode_combo.currentIndexChanged.connect(
                lambda i, idx=idx: self._on_sensor_mode_changed(idx, i))
            self._sensor_cells.append(cell)
            self._sensor_widgets.append(cell.video_widget)
            self._sensor_ts_labels.append(cell.ts_label)
            self._sensor_modes.append("hand")
            self._sensor_vmax_list.append(5000.0)
            self._sensor_mesh_states.append(DeformMeshState())
            self._sensor_hand_configs.append(
                self._load_sensor_hand_config(sensor_name))

    # ── 浏览 ──────────────────────────────────────────

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("选择录制目录"), self._data_dir)
        if d:
            self._data_dir = d
            self._dir_btn.setText(d)
            self._refresh_list()

    def _refresh_list(self):
        """扫描录制根目录，两级树展示：顶层 = 任务目录名，子项 =
        episode-xxx（仅显示 episode 文件名，不再重复长任务名）。"""
        self._stop()
        self._list.clear()
        from core.uploader import UploadManager
        by_task = {}
        for s in list_sessions(self._data_dir):
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
                child.setData(0, Qt.UserRole + 1, status)  # 存上传状态
                child.setCheckState(0, Qt.Unchecked)
                parent.addChild(child)

    def _on_session_clicked(self, item: QTreeWidgetItem, _col: int = 0):
        """点击任务目录行 = 展开/收起该目录下的数据；点击 episode 行 =
        加载回放。"""
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())
            return
        session = item.data(0, Qt.UserRole)
        self._load_session(session["path"],
                           episode_index=session.get("episode_index", 0))

    def _filter_by_upload(self, uploaded: bool):
        """根据上传状态勾选/取消勾选各 episode（父项三态自动汇总）。"""
        target_status = "completed" if uploaded else "pending"
        for i in range(self._list.topLevelItemCount()):
            parent = self._list.topLevelItem(i)
            for j in range(parent.childCount()):
                item = parent.child(j)
                status = item.data(0, Qt.UserRole + 1)
                item.setCheckState(0, Qt.Checked
                                   if status == target_status
                                   else Qt.Unchecked)

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

    def _delete_selected(self):
        """永久删除选中的会话（池化 episode 文件组 / 旧格式会话目录）。

        删除前必须释放所有占用的视频文件句柄，否则 Windows 会阻止删除。
        """
        # ── 释放视频文件句柄 ──
        self._stop()
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()
        for dv in self._depth_videos.values():
            try:
                dv.close()
            except Exception:
                pass
        self._depth_videos.clear()
        # 强制 GC 确保句柄立即释放
        import gc
        gc.collect()

        selected = []
        for i in range(self._list.topLevelItemCount()):
            parent = self._list.topLevelItem(i)
            for j in range(parent.childCount()):
                item = parent.child(j)
                if item.checkState(0) == Qt.Checked:
                    selected.append(item.data(0, Qt.UserRole))

        if not selected:
            QMessageBox.information(self, tr("提示"), tr("请先勾选要删除的会话"))
            return

        names = "\n".join(
            f'  {os.path.basename(s["path"])}/'
            f'episode-{episode_file_suffix(s["episode_index"]):03d}'
            for s in selected)
        reply = QMessageBox.question(
            self, tr("确认删除"),
            tr("确定要永久删除以下会话？\n\n{}\n\n此操作不可逆。", names),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        from core.recording_repository import RecordingRepo
        from core.uploader import UploadManager
        deleted = 0
        for s in selected:
            episode_index = s.get("episode_index", 0)
            if episode_index > 0:
                # 池化布局：彻底删除该 episode 文件组
                # （不触碰其他 episode 与任务级 meta）
                ok = delete_pooled_episode(s["path"], episode_index)
            else:
                try:
                    shutil.rmtree(s["path"])
                    ok = True
                except OSError:
                    ok = False
            if ok:
                deleted += 1
                # 成功上传过的会话 → 历史标「已上传，本地已删」（与自动上传
                # 后删除口径一致）；未上传/上传失败 → 标「已删除，未上传」
                try:
                    if UploadManager.get_upload_status(
                            s["path"], episode_index) == "completed":
                        RecordingRepo.mark_uploaded_deleted(
                            s["path"], episode_index=episode_index)
                    else:
                        RecordingRepo.mark_deleted(
                            s["path"], episode_index=episode_index)
                except Exception:
                    pass

        self._refresh_list()
        QMessageBox.information(
            self, tr("完成"),
            tr("已永久删除 {}/{} 个会话。", deleted, len(selected)),
        )

    # ── 加载会话 ──────────────────────────────────────

    def _load_session(self, session_dir: str, episode_index: int = 0):
        """加载会话：元数据在主线程同步读（极快），parquet/关键点放后台线程。

        episode_index > 0 → 池化布局（task_dir + N 文件组）。
        """
        self._stop()
        self._session_path = session_dir
        self._episode_index = episode_index

        # 防重入：同一会话正在加载时忽略
        if self._loader.is_running() and self._loading_path == session_dir:
            return
        self._loading_path = session_dir

        # 加载期间置 0，_seek 的 total_frames<=0 提前返回天然防误操作
        self._total_frames = 0
        self._timeline = None

        try:
            # 格式检测 + 元数据 + 主时钟帧率 + 传感器列名（core.session_catalog）
            meta = load_session_meta(session_dir, episode_index)
            self._fps = meta["fps"]
            self._sensor_names = meta["sensor_names"]

            display = (f"{os.path.basename(session_dir)}/"
                       f"episode-{episode_file_suffix(episode_index):03d}"
                       if episode_index > 0 else os.path.basename(session_dir))
            self._info_label.setText(tr("⏳ 正在加载: {} …", display))

            # parquet 合并 + 手部关键点 → 后台线程
            self._load_gen += 1
            self._loader.start(self._load_gen, session_dir,
                               self._sensor_names, load_kpts=True,
                               episode_index=episode_index)
        except Exception as e:
            self._info_label.setText(tr("加载失败: {}", e))

    def _on_session_loaded(self, gen: int, payload: dict):
        """后台加载完成（Qt 主线程槽）。视频打开与首帧 seek 全部在此执行。"""
        if self._closing:
            return
        if gen != self._load_gen:
            return  # 过期结果（加载期间点了别的会话）
        session_dir = self._session_path
        episode_index = getattr(self, "_episode_index", 0)
        timeline = payload.get("timeline")
        self._hand_kpts = payload.get("hand_kpts", {})
        if timeline is None:
            # kpts-only 重载（手部处理完成后）：只换关键点，不重建播放器
            self._seek(self._play_idx)  # 刷新当前帧
            return

        try:
            self._timeline = timeline

            # 过滤幽灵传感器列：无 BLE 会话的 features 回退会把
            # observation.imu 等非手套特征误判成传感器（见 _load_session），
            # 这里按时间线实际列宽（16×16=256）只留真手套数据
            self._sensor_names = self._valid_sensor_names(
                timeline, self._sensor_names)

            # 打开所有摄像机视频（深度槽位 12-bit 灰 MP4 经访问器解码）
            for cap in self._caps.values():
                try: cap.release()
                except Exception: pass
            self._caps.clear()
            for dv in self._depth_videos.values():
                try: dv.close()
                except Exception: pass
            self._depth_videos.clear()
            meta = load_session_meta(session_dir, episode_index)
            fmt = meta["fmt"]
            info = meta["info"]
            cameras = info.get("cameras", {})
            # 用户命名叠加：devices 段（slot/sensor_column → name）
            # + device_names（槽位 → 用户命名）。egodata 会话的 devices
            # 全量段在 metadata.json，device_names 在 meta/info.json，
            # 两者合并读。
            slot_names = {}
            sensor_titles = {}
            name_sources = [info]
            if fmt == "egodata":
                info_path = os.path.join(session_dir, "meta", "info.json")
                if os.path.isfile(info_path):
                    try:
                        with open(info_path, "r", encoding="utf-8") as f:
                            name_sources.append(json.load(f))
                    except Exception:
                        pass
            for _i in name_sources:
                dn = _i.get("device_names")
                if isinstance(dn, dict):
                    for k, v in dn.items():
                        if isinstance(v, str) and v:
                            slot_names.setdefault(k, v)
                for d in _i.get("devices") or []:
                    if not isinstance(d, dict) or not d.get("name"):
                        continue
                    for s in d.get("slots") or []:
                        slot_names.setdefault(s, d["name"])
                    if d.get("sensor_column"):
                        sensor_titles.setdefault(d["sensor_column"], d["name"])
            self._slot_names = slot_names
            self._sensor_titles = sensor_titles
            # 过滤：只保留有实际视频文件的摄像机。深度槽位（D435 等）的
            # 新格式 = 12-bit 灰度 MP4（v1.1.2；gray12le 对数码，回放走
            # Gray12DepthVideo 解码+规范 JET 着色）；旧格式 = 双流 MKV
            # （v1.0.14，cv2 默认读流0 即热力图画面）或 8-bit 热力图
            # MP4（v1.0.13 及以前）。按文件存在性加入回放列表。
            _video_ids = []
            _depth_ids = []
            if episode_index > 0:
                # 池化布局：以本 episode 实际文件为准（key 可跨 episode
                # 改名，不能按任务级 cameras 推断——见 episode_video_files）
                files = episode_video_files(session_dir, episode_index)
                ext_map = info.get("video_extensions", {}) or {}
                for key in sorted(files):
                    is_depth = (ext_map.get(key) == "mkv"
                                or key.lower().endswith("_depth"))
                    (_depth_ids if is_depth else _video_ids).append(key)
                self._depth_ids = _depth_ids
                self._camera_ids = _video_ids + _depth_ids
                for slot_id in self._camera_ids:
                    mp4 = files[slot_id]
                    if slot_id in self._depth_ids:
                        acc = Gray12DepthVideo.from_path(mp4)
                        if acc is not None:
                            self._depth_videos[slot_id] = acc
                            continue
                    cap = cv2.VideoCapture(mp4)
                    if cap.isOpened():
                        self._caps[slot_id] = cap
                    else:
                        cap.release()
            else:
                for slot_id in list(cameras.keys()):
                    cam_info = cameras.get(slot_id, {})
                    is_depth = ((isinstance(cam_info, dict)
                                 and cam_info.get("type") == "depth")
                                or slot_id.lower().endswith("_depth"))
                    if is_depth:
                        _depth_ids.append(slot_id)
                    else:
                        _video_ids.append(slot_id)
                self._depth_ids = _depth_ids
                self._camera_ids = _video_ids + [
                    sid for sid in _depth_ids
                    if os.path.isfile(os.path.join(
                        session_dir, "depth", sid, f"{sid}.mkv"))
                    or os.path.isfile(os.path.join(
                        session_dir, "depth", sid, f"{sid}.mp4"))
                ]
                for slot_id in self._camera_ids:
                    if slot_id in self._depth_ids:
                        mp4 = os.path.join(session_dir, "depth", slot_id,
                                           f"{slot_id}.mkv")
                        if not os.path.isfile(mp4):
                            # 新（v1.1.2 12-bit 灰）或旧（v1.0.13 及以前
                            # 8-bit 热力图）MP4，由 gray12 探测区分
                            mp4 = os.path.join(session_dir, "depth", slot_id,
                                               f"{slot_id}.mp4")
                        acc = Gray12DepthVideo.from_path(mp4)
                        if acc is not None:
                            self._depth_videos[slot_id] = acc
                            continue
                    elif fmt == "egodata":
                        mp4 = egodata_video_path(session_dir, slot_id)
                    else:
                        mp4 = video_mp4_path(session_dir, slot_id)
                    if not os.path.isfile(mp4):
                        # 兼容旧格式: videos/<slot>/chunk_000000.mp4
                        mp4 = os.path.join(session_dir, "videos", slot_id, "chunk_000000.mp4")
                    if os.path.isfile(mp4):
                        cap = cv2.VideoCapture(mp4)
                        if cap.isOpened():
                            self._caps[slot_id] = cap
                        else:
                            cap.release()

            # 重建网格（在 _caps 检查之前：无视频会话也要清掉上一会话画面）
            self._rebuild_grid()

            if not self._caps and not self._depth_videos:
                self._info_label.setText(tr("没有找到视频文件"))
                return

            # ── 主时钟 + 每路独立帧率 ──
            # 每路 fps：info cameras fps → 视频文件实际 fps → 全局兜底
            self._cam_fps = {}
            for slot_id, cam_info in cameras.items():
                if (isinstance(cam_info, dict) and cam_info.get("fps")
                        and cam_info["fps"] > 0):
                    self._cam_fps[slot_id] = float(cam_info["fps"])
            for slot_id in self._camera_ids:
                dv = self._depth_videos.get(slot_id)
                if dv is not None:
                    if dv.fps > 0:
                        self._cam_fps[slot_id] = dv.fps
                    continue
                cap = self._caps.get(slot_id)
                if cap is not None:
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    if video_fps > 0:
                        self._cam_fps[slot_id] = float(video_fps)
            fallback_fps = max(self._fps, 1.0)
            for slot_id in self._camera_ids:
                self._cam_fps.setdefault(slot_id, fallback_fps)
            self._fps = (max(self._cam_fps.values())
                         if self._cam_fps else fallback_fps)

            # 每路总帧数；主时钟总帧数 = max(帧数_i / fps_i) × 主时钟
            # （逐路 seek 时按各自总帧数截断，最短路提前停画面）
            self._cam_total = {}
            duration_s = 0.0
            for slot_id in self._camera_ids:
                dv = self._depth_videos.get(slot_id)
                if dv is not None:
                    n = dv.total
                else:
                    cap = self._caps.get(slot_id)
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            ) if cap is not None else 1
                n = max(n, 1)
                self._cam_total[slot_id] = n
                duration_s = max(duration_s, n / self._cam_fps[slot_id])
            self._total_frames = int(round(duration_s * self._fps))
            if self._total_frames <= 0:
                self._total_frames = 100
            self._play_idx = 0
            self._last_read_frame = {}

            has_kpts = "✋" if self._hand_kpts else ""
            duration = self._total_frames / max(self._fps, 1)
            task_name = info.get("task_name", "")
            task_line = tr("📋 任务: {}", task_name) + "\n" if task_name else ""
            sig_rows = timeline.signal_count if timeline else 0
            n_ble = sum(1 for d in info.get("devices") or []
                        if isinstance(d, dict) and d.get("kind") == "ble")
            ble_line = (tr("🎧 其他蓝牙: {} 台", n_ble) + "\n") if n_ble else ""
            self._info_label.setText(
                f"{task_line}"
                f"📹 {len(self._camera_ids)}{tr('路摄像机')}\n"
                f"📊 {sig_rows}{tr('传感器行')}\n"
                f"{has_kpts} {tr('手部关键点')}: {len(self._hand_kpts)}{tr('帧')}\n"
                f"{ble_line}"
                f"⏱ {duration:.1f}s | FPS:{self._fps}"
            )

            self._slider.setRange(0, max(self._total_frames - 1, 0))
            self._seek(0)
        except Exception as e:
            self._info_label.setText(tr("加载失败: {}", e))

    def _on_session_load_failed(self, gen: int, error: str):
        """后台加载失败（Qt 主线程槽）。"""
        if self._closing or gen != self._load_gen:
            return
        self._info_label.setText(tr("加载失败: {}", error))

    # ── 播放控制 ──────────────────────────────────────

    def _toggle_play(self):
        if self._playing:
            self._stop()
        else:
            # 播完后再点播放 = 从头重播
            if self._play_idx >= self._total_frames - 1:
                self._seek(0)
            self._playing = True
            self._last_tick_ts = 0.0  # 首 tick 只进 1 帧
            interval = int(1000 / max(self._fps * self._speed, 0.1))
            self._timer.start(interval)
            self._play_btn.setText(tr("⏸ 暂停"))

    def _stop(self):
        self._playing = False
        self._timer.stop()
        self._play_btn.setText(tr("▶ 播放"))

    def _tick(self):
        if self._slider_dragging:
            return  # 拖动中不推进播放
        if self._play_idx >= self._total_frames - 1:
            self._stop()
            return
        # 追赶：主线程被 seek/渲染拖慢时按真实时间补帧，保持原速
        now = time.monotonic()
        elapsed = now - self._last_tick_ts if self._last_tick_ts > 0 else 0.0
        self._last_tick_ts = now
        steps = max(1, round(elapsed * self._fps * self._speed))
        self._seek(min(self._total_frames - 1, self._play_idx + steps))

    def _prev_frame(self):
        self._seek(max(0, self._play_idx - 1))

    def _next_frame(self):
        self._seek(min(self._total_frames - 1, self._play_idx + 1))

    def _seek(self, idx: int):
        if self._total_frames <= 0:
            return
        idx = max(0, min(self._total_frames - 1, idx))
        self._play_idx = idx
        self._slider.setValue(idx)

        # 主时钟时间 → 每路帧号（逐路独立，低帧率路按比例抽帧）
        t_s = idx / max(self._fps, 1.0)

        for slot_id in self._camera_ids:
            cap = self._caps.get(slot_id)
            dv = self._depth_videos.get(slot_id)
            if (cap is None and dv is None) or self.grid.camera_widget(slot_id) is None:
                continue
            fps_i = self._cam_fps.get(slot_id, self._fps)
            frame_i = int(t_s * fps_i + 0.5)   # 四舍五入（half-up）
            frame_i = max(0, min(self._cam_total.get(slot_id, 1) - 1, frame_i))
            last = self._last_read_frame.get(slot_id)
            if last is not None and frame_i == last:
                continue  # 低帧率路重复帧：画面与帧号标题保持
            if dv is not None:
                # 12-bit 灰 MP4：访问器内部顺序续读（小步）/-ss 快进
                # （大跳，最坏从最近关键帧解到目标帧，亚秒级）
                frame = dv.read(frame_i)
                ok = frame is not None
            else:
                # 小步前进（≤5 帧，定时器抖动/追赶所致）顺序 read 到底
                # （每帧 1-2ms，3 帧实测 5.7ms）；随机大跳（拖进度条）才
                # set+read seek（43-89ms/路）。小步也走 seek 的话，seek 耗时
                # 会把下一拍的 steps 推得更大 → 死亡螺旋，全程跳帧卡顿
                # （HEVC B 帧让 seek 比 x264 更贵，螺旋更易触发）。
                forward = ((last is not None and 0 < frame_i - last <= 5)
                           or (last is None and frame_i == 0))
                if forward:
                    reads = (frame_i - last) if last is not None else 1
                    ok, frame = True, None
                    for _ in range(reads):
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                    ok, frame = cap.read()
            self._last_read_frame[slot_id] = frame_i
            if ok and frame is not None:
                self._set_video_frame(slot_id, frame, frame_i)

        self._update_sensor(idx, t_s)
        self._update_time_label(idx)

    def _update_time_label(self, idx: int):
        t_cur = idx / max(self._fps, 1)
        t_total = (self._total_frames / max(self._fps, 1))
        self._time_label.setText(
            f"{int(t_cur//60):02d}:{int(t_cur%60):02d} / "
            f"{int(t_total//60):02d}:{int(t_total%60):02d}"
        )

    # ── 滑块拖动 ──────────────────────────────────────

    def _on_slider_pressed(self):
        """开始拖动：停表、只记位置不 seek（seek 留到松手）。"""
        self._slider_dragging = True
        self._slider_was_playing = self._playing
        self._pending_seek = -1
        if self._playing:
            self._stop()

    def _on_slider_moved(self, value: int):
        if not self._slider_dragging:
            return
        self._pending_seek = value
        self._update_time_label(value)

    def _on_slider_released(self):
        """松手：单次 seek；原在播则恢复播放。"""
        if not self._slider_dragging:
            return
        self._slider_dragging = False
        target = self._pending_seek
        if target < 0:
            target = self._slider.value()  # 点击滑槽未拖动：跳到点击处
        if target != self._play_idx:
            self._seek(target)
        self._pending_seek = -1
        if self._slider_was_playing and not self._playing:
            self._playing = True
            self._last_tick_ts = 0.0
            interval = int(1000 / max(self._fps * self._speed, 0.1))
            self._timer.start(interval)
            self._play_btn.setText(tr("⏸ 暂停"))

    def _set_video_frame(self, slot_id: str, frame: np.ndarray, frame_num: int):
        """将一帧显示到对应摄像机格。

        注意: 录制时 lerobot_writer 已做 np.flip(frame, axis=0)，
        所以 MP4 里的帧方向是正的，回放不需要再次翻转。
        """
        w = self.grid.camera_widget(slot_id)
        if w is None:
            return
        try:
            # ── 手部关键点叠加（如果可用且启用） ──────
            if (self._show_hand_kpts and self._hand_kpts
                    and _HAND_KPTS_AVAILABLE and draw_kpts_overlay):
                kpt = self._hand_kpts.get(frame_num)
                if kpt is not None:
                    if frame_num % 100 == 0:
                        print(f"[Playback] 叠加手部关键点 frame={frame_num}, hands={kpt['num_hands']}")
                    frame = draw_kpts_overlay(
                        frame, kpt["hand_data"], kpt.get("track_ids"))

            # MP4 视频帧已在录制时翻转过了，不需要再次 flip
            # （ZoomableVideoWidget.set_frame 内部完成 BGR→RGB 转换与缩放）
            w.video_widget.set_frame(frame)
            w.set_frame_number(frame_num)
        except Exception:
            import traceback
            traceback.print_exc()

    def _on_sensor_mode_changed(self, sensor_idx: int, mode_idx: int):
        """切换指定传感器的可视化模式。"""
        modes = ["heatmap", "trace", "grid", "hand", "deform"]
        self._sensor_modes[sensor_idx] = modes[mode_idx]
        self._sensor_widgets[sensor_idx].setVisible(True)
        self._sensor_widgets[sensor_idx].reset_view()
        if self._sensor_modes[sensor_idx] == "trace":
            clear_trace_canvas()
        self._sensor_vmax_list[sensor_idx] = 5000.0

    def _update_sensor(self, frame_idx: int, t_s: float = 0.0):
        """渲染所有传感器的数据到各自的显示组件。

        面板按 info["sensors"] 动态创建、直映射（不反序）。多帧率混合
        会话按主时钟时间二分（nearest_for_column_time）；帧率一致会话
        仍走帧号二分（时间戳含暂停负跳变时帧号更稳）。
        """
        tl = self._timeline
        if not tl or tl.signal_count == 0:
            return

        sensor_names = getattr(self, "_sensor_names", [])
        if not sensor_names:
            sensor_names = list(settings.SENSOR_NAMES)
        if not sensor_names:
            return

        if not self._sensor_widgets:
            return   # 无传感器格（无传感器会话），旧版靠 try/except 吞 IndexError

        fps_vals = {v for v in self._cam_fps.values()} if self._cam_fps else set()
        use_time = len(fps_vals) > 1

        size = (640, 400)  # 回放面板较小
        fps_val = 30
        gate, dyn, spatial = 0, 0.0, True
        cam_ts = float(frame_idx) / max(self._fps, 1.0)

        for idx, sensor_name in enumerate(sensor_names):
            try:
                col = f"observation.{sensor_name}"
                if use_time:
                    row, _ = tl.nearest_for_column_time(col, cam_ts)
                else:
                    row, _ = tl.nearest_for_column(col, frame_idx)
                state_row = None if row is None else tl.obs[col][row]
                if state_row is None:
                    if idx == 0 and "observation.state" in tl.obs:
                        if use_time:
                            row, _ = tl.nearest_for_column_time(
                                "observation.state", cam_ts)
                        else:
                            row, _ = tl.nearest_for_column(
                                "observation.state", frame_idx)
                        if row is not None:
                            state_row = tl.obs["observation.state"][row]
                    if state_row is None:
                        self._sensor_widgets[idx].set_status_text(tr("无信号"))
                        continue

                mat = np.asarray(state_row, dtype=np.float32).reshape(16, 16)
                mode = self._sensor_modes[idx]
                max_signal = mat.max()
                vmax = self._sensor_vmax_list[idx]
                mesh_state = self._sensor_mesh_states[idx]

                if mode == "heatmap":
                    frame, vmax = render_heatmap(
                        mat, max_signal, self._sensor_config, size,
                        vmax, fps_val, gate, dyn, spatial,
                    )
                elif mode == "trace":
                    frame, vmax, _ = render_trace(
                        mat, max_signal, self._sensor_config, size,
                        vmax, fps_val, gate, dyn, spatial,
                    )
                elif mode == "grid":
                    frame = render_grid(mat, max_signal, self._sensor_config, size, fps_val)
                elif mode == "hand":
                    frame, vmax = render_hand(
                        mat, max_signal, self._sensor_hand_configs[idx], size,
                        vmax, fps_val, gate, dyn, spatial, 0,
                    )
                elif mode == "deform":
                    frame, vmax = render_deform_mesh(
                        mat, max_signal, self._sensor_config, size,
                        vmax, fps_val, mesh_state,
                    )
                else:
                    continue

                self._sensor_vmax_list[idx] = vmax
                self._sensor_widgets[idx].set_frame(frame)
                dist = abs(tl.timestamps[row] - cam_ts)
                df = abs(int(tl.frame_indices[row]) - int(frame_idx))
                ts_str = f"{tl.timestamps[row]:.3f}s"
                d_str = f"{dist * 1000:.0f}ms/{df}f"
                self._sensor_ts_labels[idx].setText(
                    tr("传感器 TS: {}  Δ={}", ts_str, d_str))
            except Exception:
                pass

    def _cycle_speed(self):
        speeds = [0.25, 0.5, 1, 2, 4]
        try:
            idx = speeds.index(self._speed)
            self._speed = speeds[(idx + 1) % len(speeds)]
        except ValueError:
            self._speed = 1
        self._speed_btn.setText(f"{self._speed}×")
        if self._playing:
            self._timer.setInterval(int(1000 / max(self._fps * self._speed, 0.1)))

    # ── 手部关键点叠加控制 ──────────────────────────

    def _toggle_hand_overlay(self, checked: bool):
        """切换手部关键点叠加显示。"""
        self._show_hand_kpts = checked
        self._seek(self._play_idx)  # 刷新当前帧

    def _process_current_kpts(self):
        """对当前选中的会话后台提取手部关键点。"""
        if not self._session_path:
            QMessageBox.information(self, tr("提示"), tr("请先选择一个录制会话。"))
            return

        if not _HAND_KPTS_AVAILABLE:
            QMessageBox.information(self, tr("提示"), tr("手部关键点模块不可用。"))
            return

        episode_index = getattr(self, "_episode_index", 0)
        if episode_index > 0:
            # 池化布局的关键点提取在下一阶段（回填链）接入；
            # 现有 processor 仍按旧会话目录键控
            QMessageBox.information(
                self, tr("提示"),
                tr("池化布局的关键点提取即将支持，暂不可用。"))
            return

        # 如果正在处理中，提示用户
        if self._hand_processor is not None:
            if getattr(self._hand_processor, '_running', False):
                QMessageBox.information(self, tr("提示"), tr("处理正在进行中，请等待完成。"))
                return

        # 延迟创建 Processor（在主线程中，确保 Qt 信号正常投递）
        if self._hand_processor is None:
            from core.hand_processor import SessionHandProcessor
            self._hand_processor = SessionHandProcessor()
            self._hand_processor.finished.connect(self._on_hand_proc_finished)

        # 检查是否已有处理结果
        kpts_path = hand_kpts_parquet_path(self._session_path)
        if os.path.isfile(kpts_path):
            reply = QMessageBox.question(
                self, tr("确认"), tr("该录制已有手部关键点数据，要重新处理吗？"))
            if reply != QMessageBox.Yes:
                return

        self._process_kpts_btn.setEnabled(False)
        self._process_kpts_btn.setText(tr("处理中…"))
        mode = "bare" if self._hand_mode_combo.currentIndex() == 1 else "glove"
        self._hand_processor.process_session(self._session_path, mode=mode)

    def _on_hand_proc_finished(self, session_path: str, error: str):
        """手部关键点处理完成（Qt 主线程回调）。"""
        if error:
            self._process_kpts_btn.setText(tr("❌ 失败"))
        else:
            self._process_kpts_btn.setText(tr("✅ 完成"))
            if _HAND_KPTS_AVAILABLE and load_hand_kpts:
                # 后台重载 kpts（大会话 parquet 读放 worker 线程）
                self._load_gen += 1
                self._loader.start(
                    self._load_gen, session_path,
                    getattr(self, "_sensor_names", []),
                    load_kpts=True, load_timeline=False)
        self._process_kpts_btn.setEnabled(True)
        self._seek(self._play_idx)  # 刷新当前帧

    # ── 清理 ──────────────────────────────────────────

    def closeEvent(self, event):
        self._closing = True  # 迟到的 loader 回调按此丢弃
        if self._hand_processor:
            self._hand_processor.cancel()
        self._stop()
        for cap in self._caps.values():
            try: cap.release()
            except Exception: pass
        for dv in self._depth_videos.values():
            try: dv.close()
            except Exception: pass
        event.accept()
