"""
主窗口 —— 组合摄像机网格、工具栏、日志面板和录制历史面板。

负责：
  - 扫描设备列表（接入统一走设备面板开关）
  - 连接管线信号到 UI 控件
  - 录制状态管理和历史记录
  - 中英文语言切换
"""

from __future__ import annotations
import os, threading, time, json
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSettings
from PyQt5.QtGui import QFont, QColor, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QDockWidget, QTextEdit,
    QTableWidget, QTableWidgetItem, QMessageBox, QLabel,
    QAbstractItemView, QToolBar, QAction, QStackedWidget, QComboBox,
    QApplication, QDesktopWidget,
)

from config import settings
from config.i18n import tr, lang_manager
from core.database import db
from core.recording_repository import RecordingRepo
from core.recording_record import RecordingRecord
from core.task_record import load_tasks, increment_task_completed
from core.pipeline import CameraPipeline
from core.camera import CameraState
from core.device_detector import DeviceScanner
from ui.camera_grid import CameraGrid
from ui.device_panel import DevicePanel
from ui.exposure_dialog import ExposureDialog
from ui.playback_dialog import PlaybackDialog
from ui.upload_dialog import UploadDialog
from core.uploader import UploadManager
from ui.task_page import TaskSelectionPage
from ui.login_dialog import LoginDialog
from core.task_service import TaskService
from core.helpers import (format_duration, format_size_mb,
                          hand_kpts_parquet_path, session_summary,
                          episode_file_suffix)

# 双目相机 —— 子进程/管道/曝光/抽帧口径已入 core/s80m_manager；
# 路径常量 re-export 供本窗口可用性检查与离线测试 patch
import shutil
from core.s80m_manager import (
    S80MDeviceManager, frame_record_decision, s80m_depth_available,
    s80m_drop_watch, STEREO_DROP_ALERT_RATE,
    STEREO_AVAILABLE as _STEREO_AVAILABLE)
from core.stereo_depth import depth_to_heatmap

# D435 深度双目 —— 进程内 pyrealsense2 worker
# （worker 生命周期/帧处理口径在 core.d435_manager；本窗口保留
#   D435Worker/list_d400_devices 模块全局供离线测试 patch）
try:
    from core.d435_camera import (D435Worker, d435_available,
                                  list_d400_devices)
    _D435_AVAILABLE = d435_available()
except Exception:
    D435Worker = None
    d435_available = None
    list_d400_devices = None
    _D435_AVAILABLE = False
from core.d435_manager import D435DeviceManager

# 统一设备注册表 + 面板开关分派口径（core.device_manager）；
# 开启/关闭的具体动作留在本窗口回调，分派/元数据/抽帧状态重置在 core
from core.device_manager import DeviceManager, dispatch_toggle


# 设备命名/曝光基线归一化 —— 算法已入 core/device_naming.py；
# 模块级名字保留（re-export）供离线测试与旧引用兼容
from core.device_naming import (
    realsense_short, slot_base, normalize_original, allocate_slot_names)
from core.exposure_controller import apply_exposure, exposure_dialog_params
_slot_base = slot_base          # 旧名兼容（tests/device_panel_gui_smoke_test）
_realsense_short = realsense_short
_normalize_original = normalize_original

# 旧录制记录的退化相机名（含 i18n 默认英文值），用于一次性自愈补算
_LEGACY_CAM_NAMES = ("多路录制", "Multi-Camera", "")

# 手部关键点后处理是可选手部追踪子系统（依赖外部 RTMPose/YOLO/torch）
try:
    from core.hand_processor import SessionHandProcessor
    from core.auto_labeler import AutoLabeler
    _HAND_PROC_AVAILABLE = True
except ImportError:
    SessionHandProcessor = None
    AutoLabeler = None
    _HAND_PROC_AVAILABLE = False

# 手套控件是可选模块（依赖 bleak / core.ble_engine 子系统）
try:
    from ui.glove_widget import GloveWidget
    _GLOVE_AVAILABLE = True
except ImportError:
    GloveWidget = None
    _GLOVE_AVAILABLE = False


class MainWindow(QMainWindow):
    """应用程序主窗口。"""

    # 双目摄像机帧信号（从 FIFO 读取线程 → 主线程）
    # 参数: slot_id, frame, hardware_ns (SDK 硬件纳秒), imu_samples
    # ★ hardware_ns 必须用 object:PyQt5 队列信号把 Python int 按 C++
    #   qint32 封送,超过 2^31(≈2.1s 纳秒)即静默截断为负数,录制数据受害
    stereo_frame_ready = pyqtSignal(str, np.ndarray, object, list)
    # 跨线程日志信号（后台线程 → 主线程 QTextEdit.append）
    log_message = pyqtSignal(str)
    # 上传后自动删除结果信号（删除线程 → 主线程）：
    # (session_path, error_msg, episode_index)——episode_index 用 int 且恒 < 2^31
    _upload_session_deleted = pyqtSignal(str, str, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(settings.APP_NAME)
        self.resize(settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT)

        # ── 内部状态 ──────────────────────────────────
        self._pipeline = CameraPipeline()
        self._next_camera_index = 0
        self._current_task: dict | None = None           # 当前选中的任务
        self._shutting_down = False                      # 关闭窗口标志，防止线程发信号崩溃

        # ── 设备 worker 注册表（多路并发核心）─────────
        # key → {"kind": "uvc"|"d435"|"s80m"|"data_ble"|"ble",
        #        "slots": [grid 槽位], "label": 日志标签, 及 kind 专属字段:
        #        d435: worker/rgb_slot/depth_slot/near_mm/far_mm/smooth_k/
        #              temporal_alpha/heat_smoother/calib_sent/serial
        #        s80m: proc/watchdog/reader_thread/stderr_file/depth_active
        #        uvc:  (无专属，槽位即状态)
        # 注册表本体/查询/元数据口径在 core.device_manager（_workers 直接
        # 引用同一 dict，离线测试注入假条目沿用同一形状）
        self._device_manager = DeviceManager()
        self._workers = self._device_manager.entries
        # ── 面板开关分派表（kind → 具体开启/关闭动作；路由口径在
        #    core.device_manager.dispatch_toggle）──
        self._open_fns = {"uvc": self._open_uvc, "d435": self._open_d435,
                          "s80m": self._open_s80m, "data_ble": self._open_glove,
                          "ble": self._open_ble_placeholder}
        self._close_fns = {"uvc": self._close_uvc, "d435": self._close_d435,
                           "s80m": self._close_s80m, "data_ble": self._close_glove,
                           "ble": self._close_ble_placeholder}

        # ── 设备检测面板状态 ──────────────────────────
        self._device_scanner = DeviceScanner(parent=self)
        self._device_scanner.scan_finished.connect(self._on_devices_scanned)
        self._device_timer = QTimer(self)
        self._device_timer.setInterval(settings.DEVICE_POLL_INTERVAL_MS)
        self._device_timer.timeout.connect(self._device_scanner.request_scan)
        self._active_device_keys: set = set()   # 面板勾选集合（开关状态 = UI 状态源）
        self._lost_device_keys: set = set()     # 录制中拔线暂存 key（重插恢复勾选）
        self._last_device_keys: dict = {}       # key → 标签（插拔 diff 日志）
        self._last_device_kinds: dict = {}      # key → kind（蓝牙刷屏过滤）

        # ── 实时 FPS 显示（D435/S80M 无 CameraWorker 测量，主窗口统一计数）──
        self._fps_ring: dict = {}   # slot_id → deque(帧到达时刻, monotonic 秒)
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(self._update_fps_labels)
        self._fps_timer.start()

        # ── S80M 双目管理器（子进程/管道/曝光口径在 core.s80m_manager）──
        # frame_ready 信号链到 stereo_frame_ready → _on_stereo_frame；
        # log 链到 _log；device_closed → UI 侧注销（拆槽/清标定）
        self._s80m_manager = S80MDeviceManager(self._pipeline, parent=self)
        self._s80m_manager.frame_ready.connect(self.stereo_frame_ready)
        self._s80m_manager.depth_ready.connect(self._on_s80m_depth)
        self._s80m_manager.log.connect(self._log)
        self._s80m_manager.device_closed.connect(self._on_s80m_closed)

        # ── D435 管理器（worker 生命周期/帧口径在 core.d435_manager）──
        self._d435_manager = D435DeviceManager(self._pipeline, parent=self)
        self._d435_manager.frames_ready.connect(self._on_d435_frames)
        self._d435_manager.log.connect(self._log)

        # 连接双目帧信号（后台线程安全 → 主线程 UI 更新）
        self.stereo_frame_ready.connect(self._on_stereo_frame)

        # ── 手部关键点后处理 ──────────────────────────
        self._hand_processor = None  # SessionHandProcessor（延迟初始化）
        self._auto_labeler = None     # AutoLabeler（延迟初始化）

        # ── 初始化数据库 ──────────────────────────────
        db.init_schema()

        # ── 构建界面 ──────────────────────────────────
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._connect_pipeline_signals()

        # ── 任务轮询服务 ──────────────────────────────
        self._task_service = TaskService(settings.load_server_url())
        self._connect_task_service()

        # ── 上传管理器（录制完成后自动上传）───────────
        self._upload_manager = UploadManager(
            settings.load_server_url(),
            session=getattr(self._task_service, "_session", None),
        )
        self._upload_task_map: dict = {}   # task_id → session_path（日志用）
        self._upload_manager.task_completed.connect(self._on_upload_task_done)
        self._upload_manager.task_failed.connect(self._on_upload_task_failed)
        self._upload_manager.start()
        self._upload_session_deleted.connect(self._on_upload_session_deleted)

        # ── 监听语言切换 ──────────────────────────────
        lang_manager.language_changed.connect(self._on_language_changed)

        # ★ 初始化页面状态：确保任务选择页隐藏所有数据采集 UI
        self._on_page_changed(0)

        # 摄像机自动扫描推迟到用户确认任务后执行
        self._log(tr("DAQ 视频管线已启动。"))
        self._log(tr("请在任务选择页面选择一个任务。"))

    # ═══════════════════════════════════════════════════
    #  UI 构建
    # ═══════════════════════════════════════════════════

    def _setup_ui(self):
        # ── 页面栈（任务选择 / 数据采集） ──────────
        self._page_stack = QStackedWidget()
        self.setCentralWidget(self._page_stack)

        # ── Page 0: 任务选择页面 ──────────────────
        self._task_page = TaskSelectionPage()
        self._task_page.task_selected.connect(self._on_task_confirmed)
        self._page_stack.addWidget(self._task_page)  # index 0

        # ── Page 1: 数据采集页面 ──────────────────
        collection_page = QWidget()
        root_layout = QVBoxLayout(collection_page)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(2)

        self.grid = CameraGrid()
        root_layout.addWidget(self.grid, 1)
        self._page_stack.addWidget(collection_page)  # index 1

        # 默认显示任务选择页面
        self._page_stack.setCurrentIndex(0)
        self._page_stack.currentChanged.connect(self._on_page_changed)

        # ── 日志面板（底部停靠） ───────────────────────
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setFont(QFont("Consolas", 9))
        # 样式由 Qt-Material 暗色主题接管
        self._log_widget.document().setMaximumBlockCount(2000)
        # 跨线程日志信号（必须在 _log_widget 创建之后连接）
        self.log_message.connect(self._log_widget.append)

        self._log_dock = QDockWidget(tr("日志"), self)
        self._log_dock.setWidget(self._log_widget)
        self._log_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)

        # ── 录制历史面板（右侧停靠） ──────────────────
        self._history_table = QTableWidget(0, 6)
        self._history_table.setHorizontalHeaderLabels([
            tr("摄像机"), tr("文件"), tr("时长"), tr("大小"), tr("编码"),
            tr("状态"),
        ])
        self._history_table.horizontalHeader().setStretchLastSection(True)
        self._history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 样式由 Qt-Material 暗色主题接管
        self._history_table.setAlternatingRowColors(True)
        self._history_table.verticalHeader().setVisible(False)
        self._history_table.setColumnWidth(0, 170)
        self._history_table.setColumnWidth(1, 150)
        self._history_table.setColumnWidth(2, 70)
        self._history_table.setColumnWidth(3, 60)
        self._history_table.setColumnWidth(4, 130)

        self._history_dock = QDockWidget(tr("录制历史"), self)
        self._history_dock.setWidget(self._history_table)
        self._history_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self._history_dock)
        self._refresh_history()  # 启动即加载历史记录（旧记录在此惰性补算）

        # 设置停靠面板的初始大小，避免过度挤压视频区域
        self.resizeDocks([self._log_dock], [100], Qt.Vertical)
        self.resizeDocks([self._history_dock], [260], Qt.Horizontal)

        # ── 设备检测面板（左侧停靠：已连接设备统一列表，分组 + 开关 + 命名） ──
        # 手套等蓝牙设备不再有底部 dock：开关打开后画面直接进主网格
        # （GloveWidget），录制数据经 pipeline.write_sensor 落盘。
        self._device_panel = DevicePanel()
        self._device_panel.device_toggled.connect(self._on_device_toggled)
        self._device_panel.device_renamed.connect(self._on_device_renamed)
        self._device_dock = QDockWidget(tr("📷 设备检测"), self)
        self._device_dock.setWidget(self._device_panel)
        self._device_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self._device_dock)
        self.resizeDocks([self._device_dock], [240], Qt.Horizontal)

        # ── 状态栏 ────────────────────────────────────
        self._status_label = QLabel(tr("就绪"))
        self._status_label.setStyleSheet("padding:0 8px;")
        self.statusBar().addWidget(self._status_label, 1)

        self._cam_count_label = QLabel(f"{tr('摄像机:')} 0")
        self.statusBar().addPermanentWidget(self._cam_count_label)

    def _setup_menu(self):
        mb = self.menuBar()

        # ── 文件菜单 ───────────────────────────────────
        self._file_menu = mb.addMenu(tr("文件(&F)"))
        self._scan_menu_action = self._file_menu.addAction(
            tr("扫描摄像机"), self._auto_detect_cameras, "Ctrl+R")
        self._file_menu.addSeparator()
        self._back_to_task_action = self._file_menu.addAction(
            tr("返回任务选择"), self._go_back_to_task_selection, "Ctrl+B")
        self._file_menu.addSeparator()
        self._exit_menu_action = self._file_menu.addAction(
            tr("退出(&X)"), self.close, "Ctrl+Q")

        # ── 视图菜单 ───────────────────────────────────
        self._view_menu = mb.addMenu(tr("视图(&V)"))
        self._clear_log_action = self._view_menu.addAction(
            tr("清空日志"), self._log_widget.clear)
        self._refresh_history_action = self._view_menu.addAction(
            tr("刷新历史记录"), self._refresh_history)
        self._view_menu.addSeparator()
        self._reset_size_action = self._view_menu.addAction(
            tr("重置画面大小"), self._reset_camera_sizes)

        # ── 语言菜单 ───────────────────────────────────
        self._lang_menu = mb.addMenu(tr("语言(&L)"))
        self._lang_action_zh = self._lang_menu.addAction(
            tr("中文"))
        self._lang_action_zh.setCheckable(True)
        self._lang_action_zh.setChecked(True)
        self._lang_action_zh.triggered.connect(
            lambda: lang_manager.set_language("zh"))

        self._lang_action_en = self._lang_menu.addAction(
            tr("English"))
        self._lang_action_en.setCheckable(True)
        self._lang_action_en.setChecked(False)
        self._lang_action_en.triggered.connect(
            lambda: lang_manager.set_language("en"))

        # ── 帮助菜单 ───────────────────────────────────
        self._help_menu = mb.addMenu(tr("帮助(&H)"))
        self._guide_action = self._help_menu.addAction(
            tr("使用说明"), self._show_guide)
        self._about_action = self._help_menu.addAction(
            tr("关于"), self._show_about)

    def _setup_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setStyleSheet(
            f"QToolBar {{ spacing:4px; padding:3px; }}"
        )
        self._toolbar = tb  # 保存引用，用于页面切换时控制可见性

        # Logo
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "gongsitubiao.png")
        if os.path.isfile(logo_path):
            logo_lbl = QLabel()
            pix = QPixmap(logo_path).scaledToHeight(28, Qt.SmoothTransformation)
            logo_lbl.setPixmap(pix)
            logo_lbl.setStyleSheet("background:transparent; border:none; padding:0 8px;")
            tb.addWidget(logo_lbl)
            tb.addSeparator()

        # 返回任务选择按钮
        self._back_action = QAction(tr("← 返回任务选择"), self)
        self._back_action.triggered.connect(self._go_back_to_task_selection)
        tb.addAction(self._back_action)
        tb.addSeparator()

        # 扫描按钮
        self._scan_action = QAction(tr("🔍 扫描"), self)
        self._scan_action.triggered.connect(self._auto_detect_cameras)
        tb.addAction(self._scan_action)

        tb.addSeparator()

        # 全部录制 / 全部停止
        self._rec_all_action = QAction(tr("⏺ 全部录制"), self)
        self._rec_all_action.triggered.connect(self._record_all)
        tb.addAction(self._rec_all_action)

        self._stop_all_action = QAction(tr("⏹ 完成录制"), self)
        self._stop_all_action.triggered.connect(self._stop_all)
        tb.addAction(self._stop_all_action)

        # 异常终止：停止录制并永久删除本次录制的文件
        self._abort_action = QAction(tr("⛔ 异常终止"), self)
        self._abort_action.triggered.connect(self._abort_recording)
        self._abort_action.setToolTip(tr("停止录制并永久删除本次录制的全部文件"))
        tb.addAction(self._abort_action)

        tb.addSeparator()

        # 全部移除
        self._clear_action = QAction(tr("✕ 全部移除"), self)
        self._clear_action.triggered.connect(self._remove_all_cameras)
        tb.addAction(self._clear_action)

        tb.addSeparator()

        # 回放按钮
        self._playback_action = QAction(tr("📂 回放"), self)
        self._playback_action.triggered.connect(self._open_playback)
        tb.addAction(self._playback_action)

        tb.addSeparator()

        # 上传按钮
        self._upload_action = QAction(tr("☁ 上传"), self)
        self._upload_action.triggered.connect(self._open_upload)
        tb.addAction(self._upload_action)

        # 自动上传开关（checkable QAction 按钮，开启绿色高亮 + 开/关文字，
        # 持久化到 server_config.json；关闭后录制完成不再自动上传）
        self._upload_auto_action = QAction(
            tr("☁ 自动上传: {}",
               tr("开") if settings.UPLOAD_AUTO_SYNC else tr("关")), self)
        self._upload_auto_action.setCheckable(True)
        self._upload_auto_action.setChecked(settings.UPLOAD_AUTO_SYNC)
        self._upload_auto_action.setToolTip(
            tr("录制完成后自动上传；关闭后需在 ☁ 上传对话框手动上传"))
        self._upload_auto_action.toggled.connect(self._on_upload_auto_toggled)
        tb.addAction(self._upload_auto_action)
        self._style_toolbar_toggle(self._upload_auto_action)

        # 上传后自动删除开关（同上样式；持久化到 server_config.json，重启保持）
        self._upload_delete_action = QAction(
            tr("🗑 上传后自动删除: {}",
               tr("开") if settings.UPLOAD_DELETE_AFTER else tr("关")), self)
        self._upload_delete_action.setCheckable(True)
        self._upload_delete_action.setChecked(settings.UPLOAD_DELETE_AFTER)
        self._upload_delete_action.setToolTip(tr("上传成功后自动删除该会话的本地文件"))
        self._upload_delete_action.toggled.connect(self._on_upload_delete_toggled)
        tb.addAction(self._upload_delete_action)
        self._style_toolbar_toggle(self._upload_delete_action)

        self.addToolBar(tb)

        # ── 第二排工具栏：手部追踪 ───────────────────
        if settings.HAND_TRACK_ENABLED and _HAND_PROC_AVAILABLE:
            tb2 = QToolBar("HandTracking")
            tb2.setMovable(False)
            tb2.setStyleSheet("QToolBar { spacing:4px; padding:1px 3px; }")

            self._hand_mode_combo = QComboBox()
            self._hand_mode_combo.addItems([tr("🧤 手套追踪"), tr("🖐 裸手追踪")])
            self._hand_mode_combo.setCurrentIndex(0 if settings.HAND_TRACK_MODE == "glove" else 1)
            self._hand_mode_combo.setToolTip(tr("选择手部追踪模式：黑色手套 / 裸手"))
            self._hand_mode_combo.setMaximumWidth(130)
            self._hand_mode_combo.setStyleSheet("QComboBox { padding:2px 4px; border-radius:3px; }")
            tb2.addWidget(self._hand_mode_combo)

            self._process_hand_action = QAction(tr("✋ 处理手部关键点"), self)
            self._process_hand_action.triggered.connect(self._process_hand_keypoints)
            self._process_hand_action.setToolTip(tr("对录制完成的视频后台提取手部关键点"))
            tb2.addAction(self._process_hand_action)

            self._auto_label_action = QAction(tr("📊 自动标注"), self)
            self._auto_label_action.triggered.connect(self._auto_label)
            self._auto_label_action.setToolTip(tr("基于手部关键点数据自动生成手势标签"))
            tb2.addAction(self._auto_label_action)

            self.addToolBarBreak()  # 换行
            self.addToolBar(tb2)
            self._hand_toolbar = tb2
        else:
            self._hand_mode_combo = None
            self._process_hand_action = None
            self._auto_label_action = None
            self._hand_toolbar = None
        # 初始化时在任务选择页面，隐藏工具栏
        tb.setVisible(False)

    # ═══════════════════════════════════════════════════
    #  管线信号连接
    # ═══════════════════════════════════════════════════

    def _connect_pipeline_signals(self):
        pip = self._pipeline
        pip.slot_added.connect(self._on_slot_added)
        pip.slot_removed.connect(self._on_slot_removed)
        # 录制信号直接从 pipeline 发出
        pip.recording_started.connect(self._on_recording_started)
        pip.recording_finished.connect(self._on_recording_finished)
        pip.recording_aborted.connect(self._on_recording_aborted)
        pip.duration_changed.connect(self._on_duration)
        pip.error_occurred.connect(lambda sid, msg: self._log(f"{tr('[错误]')} {sid}: {msg}"))
        pip.state_changed.connect(
            lambda sid, st: self._log(tr("[{}] 摄像机状态: {}", sid, st))
        )
        # v1.0.9 编码器探测等录制期日志（writer → pipeline → 主窗口日志）
        pip.recording_log.connect(self._log)

    # ═══════════════════════════════════════════════════
    #  任务服务 & 页面切换
    # ═══════════════════════════════════════════════════

    def _connect_task_service(self):
        """连接 TaskService 信号到 TaskSelectionPage。

        轮询在身份确定后（maybe_show_login 末尾）才 start()；
        身份切换经 identity_changed → 任务页即时重滤。
        """
        self._task_service.tasks_updated.connect(self._task_page.update_tasks)
        self._task_service.connection_status.connect(self._task_page.set_connection_status)
        self._task_service.login_result.connect(self._task_page.on_login_result)
        self._task_service.identity_changed.connect(self._task_page.set_identity)
        self._task_service.identity_expired.connect(self._on_identity_expired)
        self._task_service.progress_synced.connect(self._task_page.refresh_from_disk)
        self._task_page.refresh_requested.connect(self._task_service.poll_now)
        self._task_page.switch_account_requested.connect(self._on_switch_account)

        # 只在错误首次出现时记录一次，避免重复刷屏
        self._last_task_error = ""
        def _on_task_error(msg: str):
            if msg != self._last_task_error:
                self._last_task_error = msg
                self._log(f"{tr('[错误]')} 任务服务: {msg}")
        self._task_service.error_occurred.connect(_on_task_error)

    # ═══════════════════════════════════════════════════
    #  登录 / 身份
    # ═══════════════════════════════════════════════════

    def maybe_show_login(self):
        """启动入口（main.py 调用）：弹登录对话框确定会话身份。

        返回 False = 用户关闭登录窗口（启动首屏取消），应用应直接退出。
        """
        return self._show_login_flow()

    def _show_login_flow(self):
        """登录对话框：账号登录 / 游客登录，完成后启动轮询。

        启动首屏（主窗口未显示）点关闭 → 返回 False 由 main.py 退出应用；
        应用内重开（切换账号/登录过期）关闭 → 维持旧行为游客降级。
        """
        is_startup = not self.isVisible()
        dlg = LoginDialog(self)
        if is_startup:
            # 主窗口尚未显示时（启动首屏）对话框没有可见父窗，
            # WM 会把它当普通窗口放左上角 → 显式移到主屏中央
            dlg.adjustSize()
            geo = dlg.frameGeometry()
            geo.moveCenter(QDesktopWidget().availableGeometry().center())
            dlg.move(geo.topLeft())
        dlg.exec_()
        url = dlg.server_url()
        settings.save_server_url(url)

        if dlg.choice() == "cancel":
            if is_startup:
                self._log(tr("登录已取消，程序退出。"))
                return False
            # 应用内重开取消 → 下方游客降级（与旧行为一致，避免
            # 登录过期时 401 轮询反复弹窗）

        if dlg.choice() == "login" and dlg.cookies() is not None:
            # 对话框内已用独立 Session 校验成功，直接接管 cookie
            self._task_service.adopt_login(url, dlg.username(), dlg.cookies())
            settings.save_remembered_username(
                dlg.username() if dlg.remember_checked() else "")
            self._log(tr("已登录: {} ({})", dlg.username(), url))
        else:
            self._task_service.set_guest(url)
            self._log(tr("已进入游客模式（仅可见公共任务）。"))

        # 上传管理器同步服务器地址（session 复用，无感知）
        if hasattr(self, "_upload_manager"):
            self._upload_manager.server_url = url
        self._task_page.set_server_display(url)
        self._task_service.start()
        return True

    def _on_switch_account(self):
        """任务页「切换账号」按钮 → 重开登录对话框。"""
        self._show_login_flow()

    def _on_identity_expired(self):
        """登录态轮询遇 401/403（token 过期/账号被撤销）→ 提示并重登；
        关闭对话框（Esc）自动降级游客。"""
        QMessageBox.information(self, tr("提示"), tr("登录已过期，请重新登录。"))
        self._show_login_flow()

    def _on_task_confirmed(self, task: dict):
        """用户选中任务并点击「进入采集」——切换到数据采集页面。"""
        self._current_task = task

        # 任务名直接取自所选任务（无输入框）
        task_name = task.get("name", task.get("task_name", ""))

        # 切换到数据采集页面
        self._page_stack.setCurrentIndex(1)

        # 记录日志
        self._log(tr("已选择任务: {}", task_name))

        # 不再自动扫描摄像机 —— 用户需先选择相机模式（单目/双目），再点击扫描
        self._log(tr("请选择相机模式（单目/双目），然后点击 [扫描] 按钮。"))

        # 更新窗口标题
        self.setWindowTitle(f"{task_name} — {settings.APP_NAME}")

    def _on_page_changed(self, index: int):
        """页面切换时控制工具栏和 dock 的可见性。"""
        if index == 0:
            # 任务选择页面 — 隐藏数据采集专属 UI
            self._toolbar.setVisible(False)
            if self._hand_toolbar:
                self._hand_toolbar.setVisible(False)
            self._log_dock.setVisible(False)
            self._history_dock.setVisible(False)
            # 设备检测面板：任务选择页隐藏 + 停轮询
            self._device_dock.setVisible(False)
            self._device_timer.stop()
        else:
            # 数据采集页面 — 显示全部 UI
            self._toolbar.setVisible(True)
            if self._hand_toolbar:
                self._hand_toolbar.setVisible(True)
            self._log_dock.setVisible(True)
            self._history_dock.setVisible(True)
            # 设备检测面板：采集页显示 + 立即扫描 + 周期轮询
            self._device_dock.setVisible(True)
            self._device_scanner.request_scan()
            self._device_timer.start()

    def _go_back_to_task_selection(self):
        """从数据采集页面返回任务选择页面。"""
        # 安全防护：如果正在录制，弹窗确认
        if self._pipeline.is_recording:
            reply = QMessageBox.question(
                self, tr("确认"),
                tr("录制中，确定要返回任务列表？当前录制将被终止。"),
            )
            if reply != QMessageBox.Yes:
                return
            self._pipeline.abort_recording("")

        # 取消手部关键点处理 + 自动标注（如果有在运行的）
        if self._hand_processor:
            self._hand_processor.cancel()
        if self._auto_labeler:
            self._auto_labeler.cancel()

        # 停止并移除所有摄像机
        self._teardown_all_workers()
        self._pipeline.remove_all()
        for sid in list(self.grid.slot_ids()):
            self.grid.remove_camera(sid)

        # 清设备面板激活状态（防止下次进入采集页误高亮）
        self._active_device_keys = set()
        self._lost_device_keys = set()
        self._last_device_keys = {}
        self._last_device_kinds = {}
        self._device_panel.set_checked_keys(set())
        self._device_panel.set_active_keys(set())

        # 切回任务选择页面
        self._page_stack.setCurrentIndex(0)

        # 清空当前任务
        self._current_task = None

        # 主动刷新任务列表
        self._task_service.poll_now()

        # 更新标题
        self.setWindowTitle(settings.APP_NAME)

        self._log(tr("已返回任务选择页面。"))

    def _connect_slot_signals(self, slot_id: str):
        """将单个 CameraSlot 的信号连接到 UI 回调。"""
        slot = self._pipeline.get_slot(slot_id)
        if not slot:
            return
        w = self.grid.camera_widget(slot_id)
        if not w:
            return

        # 相机帧 → 画面显示
        slot.frame_ready.connect(lambda sid, frame: self._on_frame(sid, frame))
        slot.state_changed.connect(lambda sid, st: self._on_camera_state(sid, st))
        slot.error_occurred.connect(
            lambda sid, msg: self._log(f"{tr('[错误]')} {sid}: {msg}")
        )
        slot.camera.camera_opened.connect(
            lambda ww, h, bk: self._log(tr("[{}] 相机已打开: {}×{} @ {}", slot_id, ww, h, bk))
        )
        slot.camera.fps_updated.connect(
            lambda fps: self._on_camera_fps(slot_id, fps)
        )

        w.remove_requested.connect(lambda sid: self._remove_camera(sid))
        w.exposure_clicked.connect(self._open_exposure_dialog)

    # ═══════════════════════════════════════════════════
    #  摄像机管理
    # ═══════════════════════════════════════════════════

    def _auto_detect_cameras(self):
        """扫描按钮：立即刷新左侧设备面板（接入统一走面板开关，不自动打开）。"""
        self._log(tr("正在扫描设备…"))
        self._device_scanner.request_scan()

    def _teardown_all_workers(self):
        """关闭全部已开启设备（注册表遍历分类清理口径在 core.device_manager）。"""
        self._device_manager.teardown_all(self._close_fns)

    # ── S80M 双目（每台一个子进程条目） ──────────────

    def _open_s80m(self, dev) -> bool:
        """面板开关打开 S80M 双目：子进程运行 read_stereo_rgb.py，帧经
        stdout 管道传回（子进程/管道/曝光/抽帧口径在 core.s80m_manager）。

        S80M SDK 抢占 video0/video2：与 RealSense 同开会冲突，
        检测到已有 d435 worker 时弹窗拒绝（后开者失败），不静默启动。
        """
        if not _STEREO_AVAILABLE:
            self._log(tr("[错误] 双目 demo 脚本不存在"))
            return False
        if self._device_manager.has_kind("d435"):
            QMessageBox.warning(
                self, tr("设备冲突"),
                tr("S80M 与 RealSense 深度相机无法同时开启。\n"
                   "两者共用 UVC 设备节点 (video0/video2)。\n"
                   "请先关闭 RealSense 设备再开启 S80M。"))
            self._log(tr("[设备] S80M 与 D435 冲突，已拒绝开启: {}",
                         self._device_label(dev)))
            return False

        self._log(tr("正在启动双目摄像机 (S80M)…"))

        # ★ S80M 内参注入：SDK 运行时无标定输出，从静态标定文件读
        # StereoCalibration 注册进管线（修复落盘零内参问题）
        _s80m_calib_file = os.path.join(settings.BASE_DIR, "config",
                                        "s80m_stereo_calibration.json")
        if os.path.isfile(_s80m_calib_file):
            try:
                from core.calibration import StereoCalibration
                self._pipeline.set_device_calibration(
                    dev.key, StereoCalibration.load(_s80m_calib_file))
                self._log(tr("[双目] 已注入静态标定（内参非零）"))
            except Exception as e:
                self._log(tr("[双目] 标定注入失败: {}", e))

        # 创建左右目 CameraWidget（仅 UI，不创建 CameraWorker）
        # 双目帧通过 stdout 管道直接推送到 widget，不经过 CameraWorker/OpenCV，
        # 避免 cv2.VideoCapture 与 SDK 抢占 /dev/video0
        # 同时注册为外部帧源 → 录制时自动写入 MP4
        #
        # S80M 输出: 左右两路 JPEG，各为一个完整单目画面 (1280×800)
        #   stereo_left  ← 左镜头画面
        #   stereo_right ← 右镜头画面
        _STEREO_RES_COMBINED = (800, 1280)  # (height, width)
        for sid, label in [("stereo_left", "Left Lens (S80M)"),
                           ("stereo_right", "Right Lens (S80M)")]:
            self.grid.add_camera(sid, label)
            # 视频源按录制帧率注册（MP4 时间基准 30fps；显示不受限全帧直推）
            self._pipeline.register_external_source(sid, _STEREO_RES_COMBINED,
                                                         fps=settings.STEREO_RECORD_FPS)

        # S80C 深度：子进程 SDK 深度引擎输出 → depth_ready → 第三格热力图。
        # 深度是真伪相机（set_depth_camera，D435 同款管线）：录制走
        # write_depth 双流 MKV（热力图 + 无损深度）。深度源 ~20fps
        # 低于录制 30fps，MP4 节拍由管线补拍最近帧（时长与 RGB 对齐）。
        if s80m_depth_available():
            self.grid.add_camera(settings.S80M_DEPTH_SLOT, "Depth (S80M)")
            self._pipeline.set_depth_camera(
                settings.S80M_DEPTH_SLOT, _STEREO_RES_COMBINED,
                fps=settings.STEREO_RECORD_FPS, master_slot="stereo_left",
                heatmap_near_mm=settings.S80M_DEPTH_NEAR_MM,
                heatmap_far_mm=settings.S80M_DEPTH_FAR_MM,
                heatmap_smooth_k=settings.S80M_DEPTH_SMOOTH_K)

        # 注册表条目 + 子进程（temp 50fps yaml / stderr / watchdog / reader）
        entry = self._s80m_manager.new_entry(self._device_label(dev))
        self._workers[dev.key] = entry
        if not self._s80m_manager.spawn(dev.key, entry):
            self._workers.pop(dev.key, None)
            self._log(tr("[错误] 双目 demo 脚本不存在"))
            return False
        # 曝光入口（☀ 按钮）只放左目主槽；持久化曝光随开随应用
        self._show_exposure_button("stereo_left")
        exp = settings.device_exposure(dev.key)
        if exp:
            self._s80m_set_exposure(dev.key, exp["auto"], exp["value"])
        self._log(tr("双目摄像机已启动"))
        return True

    def _on_stereo_frame(self, slot_id: str, frame: np.ndarray,
                         hardware_ns: int = 0, imu_samples: list = None):
        """接收双目帧信号（主线程执行，线程安全）。

        将显示帧直接写入录制队列，保证保存的视频与实时画面完全一致。

        注意：S80M 镜头方向已由 SDK config (left_cam_rotate_180 /
        right_cam_rotate_180 / stereo_swap_lr) 处理，此处不再翻转。

        硬件时间戳与 IMU 样本经 pipeline 写入 data/imu/ parquet
        （IMU 仅随 stereo_left 落盘，避免重复行）。

        ★ 相机档 / 30fps 采集（settings.STEREO_CAM_FPS，默认 50；子进程
          回调取帧=官方 GUI 同款，50→30 桶抽帧才有真 30fps）：显示全帧
          直推；录制按 wall 时钟 1/30s 桶抽帧（每桶保留首帧，左右目同
          ts 天然同步；传感器 hw 时钟跳变不空桶；主进程卡顿期间的突发
          帧按 hw 桶补录；hw_ns==0 兜底全录）。被抽掉帧携带的 IMU 块
          累积到下一帧，不丢样本。
        """
        display_frame = frame

        self._note_frame_arrival(slot_id)

        w = self.grid.camera_widget(slot_id)
        if w:
            w.video_widget.set_frame(display_frame, flip_vertical=False)

        # ★ 保存显示帧 → 录制（50→30 抽帧，口径在 core.s80m_manager）
        if self._pipeline.is_recording:
            entry = next((e for e in self._workers.values()
                          if e["kind"] == "s80m" and slot_id in e["slots"]),
                         None)
            # 左右目同 hw ts 成对送达：右目共享左目的 wall 时刻做桶判定，
            # 防两目处理间隔（显示工作 1-5ms）把一对帧拆进相邻桶——拆对
            # 会造成行序 -20ms 伪影且一对只录一眼
            mono = time.monotonic_ns()
            pair = entry.get("_pair") if entry else None
            if slot_id == "stereo_right" and pair and pair[0] == hardware_ns:
                mono = pair[1]
            elif entry is not None and slot_id == "stereo_left":
                entry["_pair"] = (hardware_ns, mono)
            record, imu_batch = frame_record_decision(
                entry, slot_id, hardware_ns, mono, imu_samples)
            if record:
                self._pipeline.write_external_frame(
                    slot_id, display_frame.copy(),
                    hardware_ns=hardware_ns, imu_samples=imu_batch)
            # 空桶看门狗：stereo_left 主槽口径（wall 时钟计数，口径在
            # core.s80m_manager.s80m_drop_watch；超阈值弹一次告警）
            if slot_id == "stereo_left":
                self._s80m_drop_watch(entry)

    def _on_s80m_depth(self, slot_id: str, depth: np.ndarray,
                       hardware_ns: int = 0):
        """接收 S80C 深度帧（子进程 SDK 深度引擎，实测 ~27fps 突发产出；
        主线程执行）。

        uint16 毫米 → 固定色标热力图直推深度格（与 D435 同口径）；
        录制时经管线 write_depth 12-bit 灰度 MP4 落盘（keep-latest，
        见 pipeline.write_depth）。深度源低于录制帧率，MP4 节拍由管线
        补拍最近帧（时长与 RGB 对齐）。
        """
        self._note_frame_arrival(slot_id)

        w = self.grid.camera_widget(slot_id)
        if not w:
            return
        heat = depth_to_heatmap(depth,
                                near_mm=settings.S80M_DEPTH_NEAR_MM,
                                far_mm=settings.S80M_DEPTH_FAR_MM,
                                smooth_k=settings.S80M_DEPTH_SMOOTH_K)
        w.video_widget.set_frame(heat, flip_vertical=False)

        if self._pipeline.is_recording:
            self._pipeline.write_depth(depth, depth_slot=slot_id)

    def _close_s80m(self, dev_key: str):
        """停止 S80M 子进程并清理（进程侧在 core.s80m_manager）。

        进程/文件/线程清理完毕后 manager 发 device_closed → _on_s80m_closed
        做管线注销、拆槽与标定清除（顺序与原实现一致）。
        """
        entry = self._workers.pop(dev_key, None)
        if not entry or entry["kind"] != "s80m":
            return
        self._s80m_manager.close(dev_key)

    def _on_s80m_closed(self, dev_key: str):
        """S80M 进程清理完毕（manager 信号）：注销管线源、拆槽、清标定。"""
        # 注销管线外部帧源（左右镜头各一路拼合帧）+ 深度伪相机（幂等）
        for sid in ["stereo_left", "stereo_right"]:
            self._pipeline.unregister_external_source(sid)
        self._pipeline.clear_depth_camera(settings.S80M_DEPTH_SLOT)
        # 移除 UI 控件（深度格只在开过深度时存在，按 slot_ids 守护）
        for sid in ["stereo_left", "stereo_right", settings.S80M_DEPTH_SLOT]:
            if sid in self.grid.slot_ids():
                self.grid.remove_camera(sid)
            self._fps_ring.pop(sid, None)
        # 清除本设备标定（避免串到下一会话）
        self._pipeline.clear_device_calibration(dev_key)

    # ── D435 深度双目（进程内 worker） ──────────────────

    def _open_d435(self, dev) -> bool:
        """面板开关打开 RealSense 深度双目：RGB 彩色 + 深度热力图两路画面。

        按 serial 查重（已开则拒绝重复）；槽名优先取 GUI 用户命名
        （device_names.json 持久化，如 "D405_depth" → D405_depth_rgb /
        D405_depth），未命名回落型号名（d435_rgb/d435_depth 保持经典名，
        兼容旧会话/服务器）；同前缀多台编号追加（d435_depth_2…；"depth"
        后缀仅用于回放过滤，是否落盘由显式注册决定）。
        按型号套用采集配置（D405 用 1280×720 原生深度 + 0.1-1.0m
        色标，其余用 D435 默认）。
        左右红外仅用于深度计算与标定，不显示、不落盘（用户确认）。
        """
        # 面板条目来自 2s 轮询的活体枚举,点击瞬间设备可能已被拔走/未就绪;
        # 按 serial 复查 rs 上下文而非导入期的 _D435_AVAILABLE 缓存——开机时
        # 相机未枚举完成会把常量锁死为 False,出现"面板有设备却点不动"
        live = set()
        if list_d400_devices is not None:
            live = {s for _, s in list_d400_devices()}
        if not live or dev.serial not in live:
            self._log(tr("[错误] 未检测到 RealSense 设备"))
            return False
        if self._device_manager.has_serial("d435", dev.serial):
            self._log(tr("[设备] RealSense {} 已开启", dev.serial))
            return False
        # 反向冲突检测（正向在 _open_s80m）：S80M 已开时拒绝 D435
        if self._device_manager.has_kind("s80m"):
            QMessageBox.warning(
                self, tr("设备冲突"),
                tr("S80M 与 RealSense 深度相机无法同时开启。\n"
                   "两者共用 UVC 设备节点 (video0/video2)。\n"
                   "请先关闭 S80M 再开启 RealSense。"))
            self._log(tr("[设备] D435 与 S80M 冲突，已拒绝开启: {}",
                         self._device_label(dev)))
            return False

        serial = dev.serial
        model_name = dev.display_name
        # 槽名优先用 GUI 用户命名（设备面板双击命名，持久化 device_names.json），
        # 未命名回落型号名（d435…）；分配口径见 core.device_naming.allocate_slot_names
        rgb_slot, depth_slot = allocate_slot_names(
            settings.device_name(dev.key), model_name,
            [e["rgb_slot"] for e in self._workers.values()
             if e["kind"] == "d435"])

        prof = settings.realsense_profile(model_name)
        short = _realsense_short(model_name)
        fps = prof["fps"]
        rgb_w, rgb_h = prof["rgb_resolution"]
        depth_w, depth_h = prof["depth_resolution"]
        near_mm = prof["depth_near_mm"]
        far_mm = prof["depth_far_mm"]
        smooth_k = prof.get("heatmap_smooth_k", 0)
        temporal_alpha = prof.get("heatmap_temporal_alpha", 0.0)

        self._log(tr("正在启动深度双目摄像机 ({})…", f"{short} {serial}"))

        # RGB 注册为外部帧源（录制时自动写入 videos/<rgb_slot>/ MP4）
        self.grid.add_camera(rgb_slot, f"RGB ({short})")
        self._pipeline.register_external_source(rgb_slot,
                                                (rgb_h, rgb_w), fps=fps)

        # 深度伪相机：不建视频目录，进入 metadata + 双流 MKV（热力图+无损深度）
        # 热力图色标与实时显示一致（固定范围），防帧间自适应闪烁
        self.grid.add_camera(depth_slot, f"Depth ({short})")
        self._pipeline.set_depth_camera(depth_slot,
                                        (depth_h, depth_w), fps=fps,
                                        master_slot=rgb_slot,
                                        heatmap_near_mm=near_mm,
                                        heatmap_far_mm=far_mm,
                                        heatmap_smooth_k=smooth_k,
                                        heatmap_temporal_alpha=temporal_alpha)

        # 注册表条目 + 进程内采集 worker（无 IMU；深度与左红外同 imager
        # 出厂对齐；创建/接线/帧口径在 core.d435_manager.D435DeviceManager）
        entry = self._d435_manager.new_entry(
            self._device_label(dev), serial, rgb_slot, depth_slot,
            near_mm, far_mm, smooth_k, temporal_alpha)
        self._workers[dev.key] = entry
        self._d435_manager.spawn(dev.key, entry, model_name, prof,
                                 settings.device_exposure(dev.key),
                                 D435Worker)
        # 曝光入口（☀ 按钮）只放 RGB 主槽；深度槽不重复显示
        self._show_exposure_button(rgb_slot)
        self._log(tr("深度双目摄像机已启动: {}", f"{short} {serial}"))
        return True

    def _on_d435_frames(self, slot_id: str, frame: np.ndarray,
                        hardware_ns: int = 0, imu_samples: list = None,
                        dev_key: str = None):
        """接收 D435 帧信号（主线程执行，按 worker 注册表条目分派）。

        RGB 槽：显示 + 录制（外部帧源链路）；
        深度槽：固定范围热力图显示 + 录制时写原始 uint16 深度。
        帧处理口径（calib 首帧注入/热力图 EMA/录制写入）在
        core.d435_manager，本槽只做 widget 绘制与 FPS 计数。
        """
        entry = self._workers.get(dev_key) if dev_key else None
        w = self.grid.camera_widget(slot_id)
        if not w:
            return
        # 已关闭设备的迟到信号（拔线清理后队列残余）→ 丢弃
        if not entry or entry["kind"] != "d435":
            return
        self._note_frame_arrival(slot_id)

        display, _is_depth = self._d435_manager.process_frame(
            entry, slot_id, frame, hardware_ns, dev_key)
        w.video_widget.set_frame(display, flip_vertical=False)

    def _close_d435(self, dev_key: str):
        """停止 D435 worker 并清理槽位（worker 停止在 core.d435_manager）。"""
        entry = self._workers.pop(dev_key, None)
        if not entry or entry["kind"] != "d435":
            return
        self._d435_manager.close(entry)

        # 注销外部帧源 + 深度伪相机 + 本设备标定
        rgb_slot, depth_slot = entry["rgb_slot"], entry["depth_slot"]
        self._pipeline.unregister_external_source(rgb_slot)
        self._pipeline.clear_depth_camera(depth_slot)
        self._pipeline.clear_device_calibration(dev_key)

        for sid in [rgb_slot, depth_slot]:
            if sid in self.grid.slot_ids():
                self.grid.remove_camera(sid)
            self._fps_ring.pop(sid, None)

    def _add_camera_slot(self, slot_id: str, camera_index: int, backend: str = "",
                         label: str = ""):
        """内部方法：创建管线槽位 + 网格控件 + 连接信号。"""
        try:
            slot = self._pipeline.add_camera(slot_id, camera_index)
            w = self.grid.add_camera(slot_id, label or f"Camera {camera_index}")
            self._connect_slot_signals(slot_id)
            self._next_camera_index = max(self._next_camera_index, camera_index + 1)
            self._log(tr("摄像机 {} 已添加 ({}):", camera_index, slot_id))
            self._update_status()
        except Exception as e:
            self._log(tr("添加摄像机 {} 失败: {}", camera_index, e))

    def _remove_camera(self, slot_id: str):
        """移除指定摄像机（录制中先确认）。"""
        if self._pipeline.is_recording:
            reply = QMessageBox.question(
                self, tr("确认"),
                tr("摄像机 {} 正在录制中，确定要中止并移除？", slot_id),
            )
            if reply != QMessageBox.Yes:
                return
        self._remove_camera_slot_no_confirm(slot_id)

    def _remove_camera_slot_no_confirm(self, slot_id: str):
        """移除指定摄像机槽（不弹确认；调用方已确认或场景无录制）。"""
        self._pipeline.remove_camera(slot_id)
        self.grid.remove_camera(slot_id)
        self._log(tr("摄像机 '{}' 已移除。", slot_id))
        # 若该槽属于某个面板开启的设备，同步关闭其注册表条目并取消勾选
        for key, entry in list(self._workers.items()):
            if slot_id in entry.get("slots", []):
                if entry["kind"] == "uvc":
                    self._close_uvc(key)
                elif entry["kind"] == "d435":
                    # 移除 d435 双槽之一 → 整机关闭（深度依赖 RGB 主槽）
                    self._close_d435(key)
                elif entry["kind"] == "data_ble":
                    self._close_glove(key)
                elif entry["kind"] == "ble":
                    self._close_ble_placeholder(key)
                self._active_device_keys.discard(key)
                self._device_panel.set_checked_keys(self._active_device_keys)
                self._device_panel.set_active_keys(self._active_device_keys)
                break
        self._update_status()

    def _remove_all_cameras(self):
        """移除所有摄像机。"""
        reply = QMessageBox.question(self, tr("确认"), tr("移除所有摄像机？"))
        if reply == QMessageBox.Yes:
            # 先停所有设备 worker（S80M 子进程 / D435 worker / UVC 槽）
            self._teardown_all_workers()
            for sid in list(self.grid.slot_ids()):
                self._remove_camera(sid)   # 带录制中止确认（与旧行为一致）
            # 清设备面板高亮（设备仍在列表，只是不再显示）
            self._active_device_keys = set()
            self._lost_device_keys = set()
            self._device_panel.set_checked_keys(set())
            self._device_panel.set_active_keys(set())

    # ═══════════════════════════════════════════════════
    #  设备检测面板（左侧列表 → 主网格显示）
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _device_label(dev) -> str:
        """DeviceInfo → 日志用显示标签（用户命名优先）。"""
        return f"{dev.label}" + (f" — {dev.serial}" if dev.serial else "")

    def _on_devices_scanned(self, devices: list):
        """设备扫描结果 → 面板重建 + 插拔 diff 日志 + 激活设备丢失处理。"""
        # 枚举后统一填用户命名（device_names.json），面板行显示 user_name 优先
        names = settings.load_device_names()
        for dev in devices:
            entry = names.get(dev.stable_key, "")
            dev.user_name = entry["name"] if isinstance(entry, dict) else (entry or "")

        self._device_panel.set_devices(devices)
        self._device_panel.set_checked_keys(self._active_device_keys)
        self._device_panel.set_active_keys(self._active_device_keys)

        keys = {d.key for d in devices}
        # 插拔 diff 日志（防刷屏：只对变化打日志；普通蓝牙（耳机/陌生设备）
        # 随 20s 主动发现轮次时隐时现，diff 日志会刷屏 → 跳过，设备面板
        # 列表与手套连接不受影响）
        appeared = keys - set(self._last_device_keys)
        disappeared = set(self._last_device_keys) - keys
        for key in sorted(appeared):
            dev = next((d for d in devices if d.key == key), None)
            if dev and dev.kind != "ble":
                self._log(tr("[设备] 已连接: {}", self._device_label(dev)))
        for key in sorted(disappeared):
            if self._last_device_kinds.get(key) != "ble":
                self._log(tr("[设备] 已断开: {}", self._last_device_keys[key]))
        self._last_device_keys = {d.key: self._device_label(d) for d in devices}
        self._last_device_kinds = {d.key: d.kind for d in devices}

        # 录制中拔线暂存的设备重新插回 → 恢复勾选 + 高亮
        for key in list(self._lost_device_keys):
            if key in keys:
                self._lost_device_keys.discard(key)
                self._active_device_keys.add(key)
                self._device_panel.set_checked_keys(self._active_device_keys)
                self._device_panel.set_active_keys(self._active_device_keys)
        # 已开启但已不在枚举列表的设备 → 丢失处理（多路并存，逐个处理）
        for key in sorted(self._active_device_keys - keys):
            self._handle_active_device_lost(key)

    def _handle_active_device_lost(self, key: str):
        """已开启的设备被拔出（多路并存：只处理该设备）。"""
        self._active_device_keys.discard(key)
        self._device_panel.set_checked_keys(self._active_device_keys)
        self._device_panel.set_active_keys(self._active_device_keys)
        entry = self._workers.get(key)
        label = entry.get("label", key) if entry else key
        if self._pipeline.is_recording:
            # 录制中不自动移除（remove_camera 会 abort 录制）；CameraWorker 自带重连，
            # 设备重插后恢复高亮（key 暂存于 _lost_device_keys）
            self._lost_device_keys.add(key)
            self._log(tr("[设备] 录制中拔线，不自动移除（重插自动恢复）: {}", label))
            return
        if entry and entry["kind"] == "d435":
            self._close_d435(key)
        elif entry and entry["kind"] == "s80m":
            self._close_s80m(key)
        elif entry and entry["kind"] == "uvc":
            self._close_uvc(key)
        elif entry and entry["kind"] == "data_ble":
            self._close_glove(key)
        elif entry and entry["kind"] == "ble":
            self._close_ble_placeholder(key)
        self._lost_device_keys.discard(key)
        self._log(tr("[设备] {} 已移除。", label))
        self._update_status()

    def _open_uvc(self, dev) -> bool:
        """面板开关打开 UVC 设备 → 增量建槽（不拆任何既有槽）。"""
        slot_id = settings._camera_slot_name(dev.video_index)
        if slot_id not in self.grid.slot_ids():
            self._add_camera_slot(slot_id, dev.video_index, dev.backend,
                                  label=dev.display_name)
        if slot_id not in self.grid.slot_ids():
            return False   # 建槽失败（设备被占等）
        # 槽已存在（auto-detect 或同索引重复）→ 仅注册状态
        self._workers[dev.key] = self._device_manager.uvc_entry(
            [slot_id], self._device_label(dev))
        # UVC 无曝光入口：固定自动曝光（worker 开启/重连时自动应用）
        return True

    def _close_uvc(self, dev_key: str):
        """关闭面板开启的 UVC 槽（按注册表条目）。"""
        entry = self._workers.pop(dev_key, None)
        if not entry or entry["kind"] != "uvc":
            return
        for sid in entry["slots"]:
            if sid in self.grid.slot_ids():
                self._remove_camera_slot_no_confirm(sid)

    # ── 每设备曝光设置（☀ 按钮 → 对话框 → 按类型下发 + 持久化） ──

    def _show_exposure_button(self, slot_id: str):
        """在槽画面信息条显示曝光按钮（只对设备的"主槽位"调用）。"""
        w = self.grid.camera_widget(slot_id)
        if w is not None:
            w.set_exposure_button_visible(True)

    def _open_exposure_dialog(self, slot_id: str):
        """☀ 按钮入口：找到槽所属设备，按类型弹曝光设置对话框。

        量程/基线解析口径在 core.exposure_controller.exposure_dialog_params；
        对话框只做 0..1000 刻度映射与下发信号。
        """
        if self._pipeline.is_recording:
            return   # 录制中按钮已禁用，双保险
        for key, entry in self._workers.items():
            if slot_id not in entry.get("slots", []):
                continue
            p = exposure_dialog_params(entry["kind"], entry, key)
            if p is None:
                return   # UVC 无曝光入口（固定自动曝光）
            dlg = ExposureDialog(self, p["label"], p["rng"][0], p["rng"][1],
                                 p["value"], p["auto"],
                                 decimals=p["decimals"],
                                 original=p["original"])
            dlg.apply_requested.connect(
                lambda a, v, k=key, e=entry: self._apply_exposure(k, e, a, v))
            dlg.exec_()
            return

    def _apply_exposure(self, dev_key: str, entry: dict,
                        auto: bool, value: float):
        """按设备类型下发曝光并持久化（口径在 core.exposure_controller）。"""
        apply_exposure(dev_key, entry, auto, value,
                       send_s80m=self._s80m_manager.send_exposure)

    def _s80m_set_exposure(self, dev_key: str, auto: bool, value: float):
        """向 S80M 子进程下发曝光（stdin 行协议在 core.s80m_manager）。"""
        entry = self._workers.get(dev_key)
        if not entry or entry["kind"] != "s80m":
            return
        self._s80m_manager.send_exposure(entry, auto, value)

    def _set_exposure_buttons_enabled(self, enabled: bool):
        """录制锁：所有已显示的曝光入口随录制状态禁用/恢复。"""
        for sid in self.grid.slot_ids():
            w = self.grid.camera_widget(sid)
            # isHidden 语义（窗口未 show 时 isVisible 恒 False）
            if w is not None and not w.exposure_btn.isHidden():
                w.set_exposure_enabled(enabled)

    # ── 手套 / 其他蓝牙（统一体系：画面进主网格，录制走 write_sensor） ──

    def _open_glove(self, dev) -> bool:
        """面板开关打开手套 → 仿生手掌画面进主网格 + 注册传感器列。"""
        if not _GLOVE_AVAILABLE:
            self._log(tr("[错误] 手套控件不可用（依赖缺失）"))
            return False
        # 广播名 'L'/'R' → 优先绑定对应手（否则按开关先后顺序分配，左右错位）
        prefer = {"l": "left_glove", "r": "right_glove"}.get(
            (dev.display_name or "").strip().lower(), "")
        role = settings.assign_glove_sensor_role(dev.key, prefer)
        slot = f"sensor:{dev.key}"
        if slot not in self.grid.slot_ids():
            w = GloveWidget(slot, dev.address, role, self._device_label(dev))
            self.grid.add_widget(slot, w)
            w.set_pipeline(self._pipeline)
            w.start(dev.address)
        self._pipeline.register_sensor(role)
        self._workers[dev.key] = self._device_manager.glove_entry(
            slot, role, self.grid.camera_widget(slot), self._device_label(dev))
        return True

    def _close_glove(self, dev_key: str):
        """关闭手套：断开 BLE、撤画面、注销传感器列。"""
        entry = self._workers.pop(dev_key, None)
        if not entry or entry["kind"] != "data_ble":
            return
        w = entry.get("glove")
        if w is not None:
            w.stop()
            w.set_pipeline(None)
        for sid in entry.get("slots", []):
            if sid in self.grid.slot_ids():
                self.grid.remove_camera(sid)
        role = entry.get("sensor_column")
        if role:
            self._pipeline.unregister_sensor(role)

    def _open_ble_placeholder(self, dev) -> bool:
        """面板开关打开无数据蓝牙（耳机类）→ 主网格显示占位说明。"""
        slot = f"ble:{dev.key}"
        if slot not in self.grid.slot_ids():
            w = self.grid.add_camera(slot, self._device_label(dev))
            w.video_widget.set_status_text(tr("该设备无可视化数据"))
        self._workers[dev.key] = self._device_manager.ble_entry(
            slot, self._device_label(dev))
        return True

    def _close_ble_placeholder(self, dev_key: str):
        """关闭无数据蓝牙占位画面。"""
        entry = self._workers.pop(dev_key, None)
        if not entry or entry["kind"] != "ble":
            return
        for sid in entry.get("slots", []):
            if sid in self.grid.slot_ids():
                self.grid.remove_camera(sid)

    def _on_device_toggled(self, dev, on: bool):
        """面板开关 → 打开/关闭设备（多路并发：只动自己，不互拆）。

        相机类开关即建/拆画面槽；手套 → 仿生手掌进网格 + 传感器列注册；
        无数据蓝牙 → 占位画面。录制锁双保险与 kind 路由口径在
        core.device_manager.dispatch_toggle，具体动作在 _open_* / _close_*。
        """
        opened = dispatch_toggle(dev, on, self._pipeline.is_recording,
                                 self._open_fns, self._close_fns)
        if on:
            if not opened:
                # 打开失败（录制中/冲突/缺库）→ 回退勾选
                self._device_panel.set_checked(dev.key, False)
                return
            self._active_device_keys.add(dev.key)
            self._device_panel.set_active_keys(self._active_device_keys)
            self._log(tr("[设备] 已开启: {}", self._device_label(dev)))
        else:
            self._active_device_keys.discard(dev.key)
            self._device_panel.set_active_keys(self._active_device_keys)
            self._log(tr("[设备] 已关闭: {}", self._device_label(dev)))
        self._update_status()

    def _on_device_renamed(self, dev, name: str):
        """面板双击重命名 → 日志 + 插拔标签缓存刷新（持久化已在面板完成）。"""
        self._last_device_keys[dev.key] = self._device_label(dev)
        # 同步 worker 注册表标签（拔线日志用新名）
        entry = self._workers.get(dev.key)
        if entry:
            entry["label"] = self._device_label(dev)
        self._log(tr("[设备] 已重命名: {}", self._device_label(dev)))

    def _reset_s80m_record_state(self):
        """录制起止时重置 50→30 抽帧状态（口径在 core.device_manager）。"""
        self._device_manager.reset_s80m_record_state()

    def _s80m_drop_watch(self, entry):
        """录制期空桶统计 + 告警（计数口径在 core.s80m_manager）。

        wall 时钟计数与 data 帧时间戳同源：空桶会让数据时间轴长于
        视频 PTS（均匀 30fps），下游按时间戳对齐即错位。累计 ~3 秒
        样本后空桶率超 STEREO_DROP_ALERT_RATE 记一次日志告警
        （不弹窗打扰录制，日志面板可见）。3 秒窗避开录制启动期
        （编码器子进程启动/任务目录创建）的瞬时抖动。
        """
        if not entry:
            return
        interval_ns = int(settings.STEREO_RECORD_MIN_INTERVAL_S * 1e9)
        _, dropped, elapsed = s80m_drop_watch(
            entry, time.monotonic_ns(), interval_ns)
        w = entry.get("drop_watch")
        if not w or w["alerted"] or elapsed < 90:
            return
        rate = dropped / elapsed
        if rate < STEREO_DROP_ALERT_RATE:
            return
        w["alerted"] = True
        # 告警只进日志面板，不弹窗打扰录制
        self._log(tr("[双目告警] 帧送达不稳定：近 3 秒空桶率 {:.0f}%"
                     "（{} 空桶 / {} 桶）——视频 PTS 与数据时间轴将错位；"
                     "若持续偏高请检查 USB 带宽与系统负载",
                     rate * 100, dropped, elapsed))

    def _s80m_drop_summary(self):
        """录制结束汇总空桶统计（有丢才打日志，状态随 stop 重置前调用）。"""
        for e in self._workers.values():
            if e.get("kind") != "s80m":
                continue
            w = e.get("drop_watch")
            if w and w["dropped"] > 0 and w["elapsed"] > 0:
                self._log(tr("[双目] 本次录制空桶 {} 个（{:.0f}%），"
                             "视频 PTS 与数据时间轴存在错位",
                             w["dropped"],
                             w["dropped"] / w["elapsed"] * 100))

    def _build_device_meta(self) -> list:
        """按注册表构建录制设备信息（口径在 core.device_manager）。"""
        return self._device_manager.build_device_meta()

    def _record_all(self):
        """开始录制（全部摄像机），任务名取自所选任务。

        若所选任务进度已满（completed_count >= total_required），
        阻止录制并弹窗提示。
        """
        if not self._pipeline.is_recording and self.grid.slot_ids():
            # 任务名直接取自所选任务（任务栏不再有自由输入框）
            task_name = ""
            if self._current_task:
                task_name = str(
                    self._current_task.get("name",
                                           self._current_task.get("task_name", ""))
                    or "").strip()

            # ── 进度满检查：以所选任务为准（按 id 匹配，消除同名跨身份误判）──
            batch_index = 0
            if task_name:
                all_tasks = load_tasks()
                matched = None
                sel_id = self._current_task.get("id") if self._current_task else None
                if sel_id:
                    matched = next((t for t in all_tasks if t.get("id") == sel_id), None)
                if matched is None:
                    matched = next((t for t in all_tasks if t.get("name") == task_name), None)
                if matched:
                    total = matched.get("total_required", 0)
                    completed = matched.get("completed_count", 0)
                    if total > 0 and completed >= total:
                        QMessageBox.warning(
                            self, tr("提示"),
                            tr("该任务采集进度已满（{}/{}），无法开始新的采集。",
                               completed, total),
                        )
                        return
                    # 批次序号: 当前已完成数 + 1 = 本次录制序号
                    batch_index = completed + 1

            self._reset_s80m_record_state()
            self._pipeline.start_recording(self.grid.slot_ids()[0],
                                           task_name=task_name,
                                           batch_index=batch_index,
                                           device_meta=self._build_device_meta())

    def _stop_all(self):
        """停止录制（正常完成）。"""
        if self._pipeline.is_recording:
            self._s80m_drop_summary()
            self._pipeline.finish_recording("")
            self._reset_s80m_record_state()

    def _abort_recording(self):
        """异常终止：停止录制并永久删除本次录制的全部文件。"""
        if not self._pipeline.is_recording:
            return
        self._pipeline.abort_recording("")
        self._reset_s80m_record_state()
        self._log(tr("⛔ 录制已异常终止，文件已丢弃。"))
        self._update_status()

    # ═══════════════════════════════════════════════════
    #  信号处理器：管线 → 控件
    # ═══════════════════════════════════════════════════

    def _on_slot_added(self, _slot_id: str):
        pass  # 控件已在 _add_camera_slot 中创建

    def _on_slot_removed(self, _slot_id: str):
        pass  # 控件已移除

    def _on_frame(self, slot_id: str, frame):
        if settings.CAMERA_MIRROR_HORIZONTAL:
            frame = cv2.flip(frame, 1)  # 单目相机左右镜像
        w = self.grid.camera_widget(slot_id)
        if w:
            w.update_frame(frame)

    def _on_camera_state(self, slot_id: str, state: str):
        w = self.grid.camera_widget(slot_id)
        if w:
            w.set_camera_state(state)
        self._log(tr("[{}] 摄像机状态: {}", slot_id, state))
        # 标记录制数据中的连接状态
        if state == CameraState.DISCONNECTED:
            self._pipeline.record_event(slot_id, "disconnected")
        else:
            self._pipeline.record_event(slot_id, "connected")

    def _on_camera_fps(self, slot_id: str, fps: float):
        w = self.grid.camera_widget(slot_id)
        if w:
            w.update_fps(fps)

    def _note_frame_arrival(self, slot_id: str):
        """记录一帧到达（实时 FPS 显示用；UVC 由 CameraWorker 自带测量）。"""
        self._fps_ring.setdefault(slot_id, deque()).append(time.monotonic())

    def _update_fps_labels(self):
        """每秒刷新非 UVC 槽位的 FPS 标签（过去 1 秒到达帧数）。"""
        now = time.monotonic()
        for sid, ring in list(self._fps_ring.items()):
            while ring and now - ring[0] > 1.0:
                ring.popleft()
            w = self.grid.camera_widget(sid)
            if w:
                w.update_fps(len(ring))

    def _on_recording_started(self, slot_id: str):
        """录制开始——更新状态栏 + 锁死设备开关。"""
        self._device_panel.set_locked(True)
        # 录制期间停设备轮询：detect_devices() 全量枚举（DShow/S80M/BLE 主动
        # 发现，150-210ms）在扫描线程挤占采集线程，帧以「空洞+突发」到达、
        # 外部帧队列 (maxsize=2) 溢出 → 每 2 秒周期丢帧（实测约 13%）。
        # 面板已锁死、扫描无用，直接停掉；完成后在 _on_recording_finished /
        # _on_recording_aborted 恢复。
        self._device_timer.stop()
        self._set_exposure_buttons_enabled(False)
        codec = getattr(self._pipeline, '_codec_name', 'MP4')
        task = getattr(self._pipeline, '_task_name', '')
        msg = tr("[{}] ▶ 录制开始 — 编码: {}", slot_id, codec)
        if task:
            msg += tr(" | 任务: {}", task)
        self._log(msg)
        self._update_status()

    def _on_recording_finished(self, slot_id: str, session_path: str):
        """录制完成——保存历史记录并更新任务进度。"""
        self._device_panel.set_locked(False)
        # 恢复设备轮询（_on_recording_started 停掉的；request_scan 有
        # _busy 守卫，扫描在途时忽略，不会堆积）。若录制期间已返回任务
        # 选择页（中止后切页），_on_page_changed(0) 已停轮询，这里不抢
        # 跑——再进采集页时 _on_page_changed(1) 会恢复。
        if self._page_stack.currentIndex() != 0:
            self._device_scanner.request_scan()
            self._device_timer.start()
        self._set_exposure_buttons_enabled(True)
        if session_path:
            episode_index = getattr(
                self._pipeline, "last_episode_index", 0) or 0
            cam_list, duration, size_mb = session_summary(
                session_path, self._pipeline.last_recording_frames,
                episode_index=episode_index)
            rec = RecordingRecord(
                camera_index=0,
                camera_name=cam_list or tr("多路录制"),  # 兜底：全空才用旧文案
                file_path=session_path,
                episode_index=episode_index,
                file_size_mb=size_mb,
                duration_sec=duration,
                status="completed",
            )
            RecordingRepo.save(rec)
            self._refresh_history()

            # ── 更新任务进度（按录制完成次数持久化，删除文件不回退）──
            # 归属以实际录制用的任务名为准（pipeline._task_name = 录制启动
            # 时输入框内容），而非进入采集页时选中的任务；
            # 匹配身份取自所选任务本身（id + assigned_user），
            # 避免"登录用户录制公共任务却按用户名匹配"的错配。
            task_name = getattr(self._pipeline, "_task_name", "") or ""
            sel = self._current_task
            task = increment_task_completed(
                task_name,
                task_id=(sel.get("id") if sel else None),
                assigned_user=(sel.get("assigned_user") if sel else None),
            ) if task_name else None
            if task:
                tid = task.get("id", task.get("task_id", ""))
                completed = task.get("completed_count", 0)
                total = task.get("total_required", 0)
                if self._current_task and self._current_task.get("name") == task_name:
                    self._current_task["completed_count"] = completed
                self._task_page.update_task_progress(tid, completed)
                self._log(tr("任务进度已更新: {} ({}/{})",
                           task_name, completed, total))
                # 即时上报增量（其他机器秒级可见；本机显示不依赖，失败由轮询兜底）
                self._task_service.flush_now()

            # ── v1.0.9 丢帧统计与编码器建议（编码背压可见化）──
            drops = self._pipeline.last_drop_stats
            if drops:
                imu_ov = drops.get("imu_overflow", 0)
                frame_drops = {k: v for k, v in drops.items()
                               if k != "imu_overflow"}
                total = sum(frame_drops.values())
                if total or imu_ov:
                    self._log(tr("[录制] 丢帧统计: {} (IMU 溢出 {})",
                                 frame_drops, imu_ov))
                frames = sum(self._pipeline.last_recording_frames.values())
                if (total > settings.DROP_WARN_MIN_COUNT
                        or (frames and total / frames > settings.DROP_WARN_RATIO)):
                    self._log(tr(
                        "[警告] 录制丢帧 {} 帧——编码器可能跟不上，建议降低"
                        "分辨率/路数，或修改 settings.RECORD_VIDEO_ENCODER 换"
                        "编码器", total))

        self._log(tr("[{}] ■ 录制完成: {}", slot_id, session_path))
        self._update_status()

        # ── 自动上传（仅正常停止的录制；异常丢弃走 aborted 不触发本回调；
        #    需"自动上传"开关开启，否则提示用户到 ☁ 上传对话框手动上传）──
        if session_path and settings.UPLOAD_ENABLED:
            if not settings.UPLOAD_AUTO_SYNC:
                self._log(tr("☁ 自动上传未开启，请手动上传: {}",
                             os.path.basename(session_path)))
            # 防重守卫：该 episode 尚未上传过才入队
            elif UploadManager.get_upload_status(
                    session_path, episode_index) != "completed":
                # 目标项目可能已在上传对话框中修改，入队前同步最新配置
                self._upload_manager.project_id = settings.load_upload_project_id()
                task_id = self._upload_manager.add_task(
                    session_path, episode_index=episode_index)
                self._upload_task_map[task_id] = (session_path, episode_index)
                self._log(tr("☁ 已自动加入上传队列: {}",
                             os.path.basename(session_path)))

        # ── 自动后台处理手部关键点 ──────────────────
        if settings.HAND_TRACK_ENABLED and _HAND_PROC_AVAILABLE and session_path:
            QTimer.singleShot(500, lambda: self._process_hand_keypoints(
                session_path, silent=True, episode_index=episode_index))

    def _on_recording_aborted(self, slot_id: str):
        """异常停止。"""
        self._device_panel.set_locked(False)
        # 中止走独立信号 recording_aborted，不经过 _on_recording_finished——
        # 必须单独恢复轮询，否则设备面板轮询永久停止。但中止后可能已切回
        # 任务选择页（轮询应停），守卫同 _on_recording_finished。
        if self._page_stack.currentIndex() != 0:
            self._device_scanner.request_scan()
            self._device_timer.start()
        self._set_exposure_buttons_enabled(True)
        self._log(tr("[{}] ✕ 录制已丢弃。", slot_id))
        self._update_status()
        if self._hand_processor:
            self._hand_processor.cancel()
        if self._auto_labeler:
            self._auto_labeler.cancel()

    def _on_duration(self, _slot_id: str, seconds: float):
        """录制时长 → 状态栏。"""
        self._status_label.setText(f"⏱ {format_duration(seconds)}")

    # ═══════════════════════════════════════════════════
    #  录制历史面板
    # ═══════════════════════════════════════════════════

    def _refresh_history(self):
        """从数据库加载最近 100 条录制记录并更新表格。"""
        records = RecordingRepo.list_all(limit=100)
        self._history_table.setRowCount(len(records))
        for i, r in enumerate(records):
            # 旧记录自愈：退化值实时从磁盘补算并回写 DB（一次性，之后读到真实值）
            degenerate = (not r.file_size_mb
                          or r.camera_name in _LEGACY_CAM_NAMES
                          or not r.duration_sec)
            if degenerate and os.path.isdir(r.file_path or ""):
                cam_list, duration, size = session_summary(
                    r.file_path, {}, episode_index=r.episode_index)
                if cam_list and (size > 0 or duration > 0):
                    r.camera_name, r.duration_sec, r.file_size_mb = cam_list, duration, size
                    RecordingRepo.save(r)
            self._history_table.setItem(i, 0, QTableWidgetItem(r.camera_name))
            fname = os.path.basename(r.file_path) if r.file_path else "-"
            if r.episode_index > 0:
                fname = f"{fname}_ep{episode_file_suffix(r.episode_index):06d}"
            self._history_table.setItem(i, 1, QTableWidgetItem(fname))
            self._history_table.setItem(i, 2, QTableWidgetItem(format_duration(r.duration_sec)))
            self._history_table.setItem(i, 3, QTableWidgetItem(format_size_mb(r.file_size_mb)))
            self._history_table.setItem(
                i, 4, QTableWidgetItem(self._history_codec_label(
                    r.file_path, r.episode_index)))
            # 上传成功但本地保留：新写入行 status="uploaded"；v1.0.12 之前的
            # 旧行无此状态，按 upload_task 最新行自愈显示（completed 且目录
            # 仍在 = 传过没删）
            uploaded_kept = (r.status == "uploaded") or (
                r.status == "completed" and os.path.isdir(r.file_path or "")
                and UploadManager.get_upload_status(
                    r.file_path, r.episode_index) == "completed")
            if r.status == "uploaded_deleted":
                status_text = tr("已上传，本地已删")
            elif uploaded_kept:
                status_text = tr("已上传")
            elif r.status == "completed":
                status_text = tr("已完成")
            elif r.status == "deleted":
                status_text = tr("已删除（未上传）")
            else:
                status_text = tr("已丢弃")
            status_item = QTableWidgetItem(status_text)
            if r.status == "completed":
                status_item.setForeground(QColor(settings.COLOR_STOPPED))
            elif r.status == "uploaded_deleted" or uploaded_kept:
                status_item.setForeground(QColor(settings.COLOR_STOPPED))
            else:
                # deleted（已删除未上传）与 aborted（已丢弃）均橙色警示
                status_item.setForeground(QColor(settings.COLOR_ABNORMAL))
            self._history_table.setItem(i, 5, status_item)

    @staticmethod
    def _history_codec_label(session_path: str, episode_index: int = 0) -> str:
        """录制历史「编码」列：池化读 episodes 行 video_codec（JSON）；
        旧会话读 metadata.video_codec（v1.0.9+）；无该字段=H.264。"""
        vc = None
        if episode_index > 0:
            from core.helpers import episode_row
            row = episode_row(session_path, episode_index)
            raw = row.get("video_codec")
            if isinstance(raw, str):
                try:
                    vc = json.loads(raw)
                except Exception:
                    vc = None
        else:
            try:
                with open(os.path.join(session_path, "metadata.json"),
                          encoding="utf-8") as f:
                    meta = json.load(f)
                vc = meta.get("video_codec")
            except Exception:
                return "—"
        if not vc:
            return "H.264"
        label = vc.get("codec") or ""
        if vc.get("encoder"):
            label += f" ({vc['encoder']})"
        return label or "—"

    # ═══════════════════════════════════════════════════
    #  语言切换
    # ═══════════════════════════════════════════════════

    def _on_language_changed(self, lang: str):
        """语言切换时刷新所有界面文字。"""
        # 更新语言菜单选中状态
        self._lang_action_zh.setChecked(lang == "zh")
        self._lang_action_en.setChecked(lang == "en")

        # 更新语言菜单项文字
        self._lang_action_zh.setText(tr("中文"))
        self._lang_action_en.setText(tr("English"))

        # ── 菜单栏标题 ──────────────────────────────
        self._file_menu.setTitle(tr("文件(&F)"))
        self._view_menu.setTitle(tr("视图(&V)"))
        self._lang_menu.setTitle(tr("语言(&L)"))
        self._help_menu.setTitle(tr("帮助(&H)"))

        # ── 文件菜单项 ──────────────────────────────
        self._scan_menu_action.setText(tr("扫描摄像机"))
        self._exit_menu_action.setText(tr("退出(&X)"))

        # ── 视图菜单项 ──────────────────────────────
        self._clear_log_action.setText(tr("清空日志"))
        self._refresh_history_action.setText(tr("刷新历史记录"))
        self._reset_size_action.setText(tr("重置画面大小"))

        # ── 返回任务选择 ──────────────────────────
        self._back_to_task_action.setText(tr("返回任务选择"))

        # ── 帮助菜单项 ──────────────────────────────
        self._guide_action.setText(tr("使用说明"))
        self._about_action.setText(tr("关于"))

        # ── 停靠面板标题 ────────────────────────────
        self._log_dock.setWindowTitle(tr("日志"))
        self._history_dock.setWindowTitle(tr("录制历史"))
        self._device_dock.setWindowTitle(tr("📷 设备检测"))
        self._device_panel.refresh_texts()

        # ── 历史表格表头 ────────────────────────────
        self._history_table.setHorizontalHeaderLabels([
            tr("摄像机"), tr("文件"), tr("时长"), tr("大小"), tr("状态"),
        ])

        # ── 工具栏按钮 ───────────────────────────────
        self._scan_action.setText(tr("🔍 扫描"))
        self._rec_all_action.setText(tr("⏺ 全部录制"))
        self._stop_all_action.setText(tr("⏹ 完成录制"))
        self._abort_action.setText(tr("⛔ 异常终止"))
        self._clear_action.setText(tr("✕ 全部移除"))
        self._playback_action.setText(tr("📂 回放"))
        self._upload_action.setText(tr("☁ 上传"))
        self._upload_auto_action.setText(
            tr("☁ 自动上传: {}",
               tr("开") if self._upload_auto_action.isChecked() else tr("关")))
        self._upload_auto_action.setToolTip(
            tr("录制完成后自动上传；关闭后需在 ☁ 上传对话框手动上传"))
        self._upload_delete_action.setText(
            tr("🗑 上传后自动删除: {}",
               tr("开") if self._upload_delete_action.isChecked() else tr("关")))
        self._upload_delete_action.setToolTip(
            tr("上传成功后自动删除该会话的本地文件"))

        # ── 工具栏提示 ───────────────────────────────
        self._abort_action.setToolTip(
            tr("停止录制并永久删除本次录制的全部文件"))

        # ── 手部追踪模式选择器 ──────────────────────
        if hasattr(self, '_hand_mode_combo') and self._hand_mode_combo:
            self._hand_mode_combo.setItemText(0, tr("🧤 手套追踪"))
            if self._hand_mode_combo.count() > 1:
                self._hand_mode_combo.setItemText(1, tr("🖐 裸手追踪"))
            self._hand_mode_combo.setToolTip(tr("选择手部追踪模式：黑色手套 / 裸手"))

        # ── 返回任务选择 ───────────────────────────
        self._back_action.setText(tr("← 返回任务选择"))

        # ── 任务选择页面 ───────────────────────────
        self._task_page._on_language_changed(lang)

        # ── 手部关键点处理按钮 ──────────────────────
        if self._process_hand_action:
            self._process_hand_action.setText(tr("✋ 处理手部关键点"))
            self._process_hand_action.setToolTip(tr("对录制完成的视频后台提取手部关键点"))
        if self._auto_label_action:
            self._auto_label_action.setText(tr("📊 自动标注"))
            self._auto_label_action.setToolTip(tr("基于手部关键点数据自动生成手势标签"))

        # ── 状态栏 ───────────────────────────────────
        self._update_status()
        self._refresh_history()

    # ═══════════════════════════════════════════════════
    #  视图辅助
    # ═══════════════════════════════════════════════════

    def _reset_camera_sizes(self):
        """重置所有分割条到等分状态。"""
        self.grid._rebuild_layout()

    # ═══════════════════════════════════════════════════
    #  状态栏刷新
    # ═══════════════════════════════════════════════════

    def _update_status(self):
        # 摄像机计数只算相机画面（手套/蓝牙占位槽不计入）
        n = sum(1 for sid in self.grid.slot_ids()
                if not sid.startswith(("sensor:", "ble:")))
        recording_count = 1 if self._pipeline.is_recording else 0
        self._cam_count_label.setText(
            f"{tr('摄像机:')} {n}  |  {tr('录制中:')} {recording_count}"
        )
        if recording_count > 0:
            self._status_label.setStyleSheet(
                f"color:{settings.COLOR_RECORDING}; font-weight:bold; font-size:12px; padding:0 8px;"
            )
        else:
            self._status_label.setText(tr("就绪"))
            self._status_label.setStyleSheet("padding:0 8px;")

    # ═══════════════════════════════════════════════════
    #  手部关键点后处理
    # ═══════════════════════════════════════════════════

    def _process_hand_keypoints(self, session_path: str = "", silent: bool = False,
                                episode_index: int = 0):
        """对录制完成的视频后台提取手部关键点。

        Args:
            session_path: 任务目录（池化）或旧会话目录路径
            silent: True=自动模式（录制完成自动触发），跳过确认弹窗，已有数据则跳过
            episode_index: 池化布局的 episode 序号（>0 = pooled）
        """
        if not _HAND_PROC_AVAILABLE:
            if not silent:
                self._log(tr("[手部关键点] 模块不可用（缺少依赖）。"))
            return

        # 如果正在处理中，提示用户
        if self._hand_processor and hasattr(self._hand_processor, '_running') and self._hand_processor._running:
            if not silent:
                QMessageBox.information(self, tr("提示"), tr("手部关键点处理正在进行中，请等待完成。"))
            return

        # 如果没有传入路径，使用上次录制的路径
        if not session_path:
            records = RecordingRepo.list_all(limit=1)
            if records:
                session_path = records[0].file_path
                episode_index = records[0].episode_index
            else:
                if not silent:
                    self._log(tr("[手部关键点] 没有找到可处理的录制。"))
                return

        if episode_index > 0:
            # 池化布局的关键点提取在下一阶段（回填链）接入；
            # 现有 processor 仍按旧会话目录键控
            self._log(tr("[手部关键点] 池化布局处理将在下一阶段启用，跳过。"))
            return

        if not os.path.isdir(session_path):
            if not silent:
                self._log(tr("[手部关键点] 会话目录不存在: {}", session_path))
            return

        kpts_path = hand_kpts_parquet_path(session_path)
        if os.path.isfile(kpts_path):
            if silent:
                return
            reply = QMessageBox.question(
                self, tr("确认"),
                tr("该录制已有手部关键点数据，要重新处理吗？"),
            )
            if reply != QMessageBox.Yes:
                return

        # ── 启动处理 ──────────────────────────────────
        self._log(tr("[手部关键点] 开始处理: {}", os.path.basename(session_path)))

        if not self._hand_processor:
            self._hand_processor = SessionHandProcessor()
            self._hand_processor.progress.connect(self._on_hand_proc_progress)
            self._hand_processor.status_changed.connect(
                lambda s: self._log(tr("[手部关键点] {}", s)))
            self._hand_processor.finished.connect(self._on_hand_proc_finished)

        # 按钮切换为"处理中"
        if self._process_hand_action:
            self._process_hand_action.setEnabled(False)
            self._process_hand_action.setText(tr("⏳ 处理中…"))

        mode = "bare" if (self._hand_mode_combo and self._hand_mode_combo.currentIndex() == 1) else "glove"
        self._hand_processor.process_session(session_path, mode=mode)

    def _on_hand_proc_progress(self, current: int, total: int):
        """手部关键点处理进度 → 状态栏 + 按钮。"""
        pct = current / max(total, 1) * 100
        self._status_label.setText(
            tr("✋ 处理手部关键点: {}/{} ({:.0f}%)", current, total, pct))
        if self._process_hand_action:
            self._process_hand_action.setText(tr("⏳ {:.0f}%", pct))

    def _on_hand_proc_finished(self, session_path: str, error: str):
        """手部关键点处理完成 → 自动触发标注。"""
        # 恢复按钮
        if self._process_hand_action:
            self._process_hand_action.setEnabled(True)
            self._process_hand_action.setText(tr("✋ 处理手部关键点"))

        if error:
            self._log(tr("[手部关键点] ❌ 处理失败: {}", error))
        else:
            self._log(tr("[手部关键点] ✅ 处理完成: {}",
                         os.path.basename(session_path)))
            # 自动触发标注
            QTimer.singleShot(300, lambda: self._auto_label(session_path, silent=True))
        self._update_status()

    def _auto_label(self, session_path: str = "", silent: bool = False):
        """基于手部关键点数据自动生成手势标签。

        Args:
            session_path: 会话目录路径，为空则使用最新录制
            silent: True=自动模式，静默执行
        """
        if not _HAND_PROC_AVAILABLE or AutoLabeler is None:
            if not silent:
                self._log(tr("[自动标注] 模块不可用。"))
            return

        if not session_path:
            records = RecordingRepo.list_all(limit=1)
            if records:
                session_path = records[0].file_path
            else:
                if not silent:
                    self._log(tr("[自动标注] 没有找到可处理的录制。"))
                return

        if not os.path.isdir(session_path):
            if not silent:
                self._log(tr("[自动标注] 会话目录不存在: {}", session_path))
            return

        kpts_path = hand_kpts_parquet_path(session_path)
        if not os.path.isfile(kpts_path):
            if not silent:
                self._log(tr("[自动标注] 请先提取手部关键点。"))
            return

        self._log(tr("[自动标注] 开始标注: {}",
                     os.path.basename(session_path)))

        if self._auto_labeler is None:
            self._auto_labeler = AutoLabeler()
            self._auto_labeler.progress.connect(
                lambda c, t: self._status_label.setText(
                    tr("📊 自动标注: {}/{} ({:.0f}%)", c, t, c / max(t, 1) * 100)))
            self._auto_labeler.finished.connect(self._on_auto_label_finished)

        if self._auto_label_action:
            self._auto_label_action.setEnabled(False)
            self._auto_label_action.setText(tr("⏳ 标注中…"))

        self._auto_labeler.label_session(session_path)

    def _on_auto_label_finished(self, session_path: str, error: str):
        """自动标注完成。"""
        if self._auto_label_action:
            self._auto_label_action.setEnabled(True)
            self._auto_label_action.setText(tr("📊 自动标注"))

        if error:
            self._log(tr("[自动标注] ❌ 标注失败: {}", error))
        else:
            self._log(tr("[自动标注] ✅ 标注完成: {}",
                         os.path.basename(session_path)))
        self._update_status()

    # ═══════════════════════════════════════════════════
    #  日志
    # ═══════════════════════════════════════════════════

    def _log(self, msg: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        text = f"[{stamp}] {msg}"
        # 主线程直接写，后台线程通过信号安全投递
        if QThread.currentThread() is QApplication.instance().thread():
            self._log_widget.append(text)
        else:
            self.log_message.emit(text)

    # ═══════════════════════════════════════════════════
    #  对话框
    # ═══════════════════════════════════════════════════

    def _open_playback(self):
        """打开录制回放对话框。"""
        dlg = PlaybackDialog(self, settings.RECORDING_DIR)
        dlg.exec_()
        # 对话框内删除会话可能改变历史状态（如标「已上传，本地已删」）
        self._refresh_history()

    def _on_upload_task_done(self, task_id: str):
        """自动上传完成——记录日志；"上传后自动删除"开启时删除本地会话目录，
        否则把录制行标为「已上传」（本地保留）并刷新历史。"""
        pair = self._upload_task_map.pop(task_id, ("", 0))
        path = pair[0] if isinstance(pair, tuple) else pair
        episode_index = pair[1] if isinstance(pair, tuple) else 0
        name = os.path.basename(path) if path else task_id
        if episode_index > 0:
            name = f"{name}_ep{episode_file_suffix(episode_index):06d}"
        self._log(tr("☁ 上传完成: {}", name))
        if path and settings.UPLOAD_DELETE_AFTER:
            self._delete_session_after_upload(path, episode_index)
        else:
            RecordingRepo.mark_uploaded(path, episode_index=episode_index)
            self._refresh_history()

    def _delete_session_after_upload(self, session_path: str,
                                     episode_index: int = 0):
        """上传成功后删除本地会话目录（后台线程，防大目录阻塞 UI）。

        池化布局（episode_index > 0）只删该 episode 文件组
        （delete_pooled_episode → 彻底删除，不走 _trash）。
        结果经 _upload_session_deleted 信号回主线程：标记 recording 行、
        刷新历史面板并记日志。手部关键点后处理持有的已打开视频句柄在
        Linux 下不受 unlink 影响，输出镜像在 keypoints_output/ 不受牵连。
        """
        def _worker():
            err = ""
            if episode_index > 0:
                from core.helpers import delete_pooled_episode
                delete_pooled_episode(session_path, episode_index)
            elif not os.path.isdir(session_path):
                err = "会话目录不存在"
            else:
                try:
                    shutil.rmtree(session_path)
                except OSError as e:
                    err = str(e)
            self._upload_session_deleted.emit(session_path, err,
                                              episode_index)
        threading.Thread(target=_worker, daemon=True,
                         name="upload-auto-delete").start()

    def _on_upload_session_deleted(self, session_path: str, err: str,
                                   episode_index: int = 0):
        """后台删除完成（主线程）：recording 行标记已删并刷新历史。"""
        name = os.path.basename(session_path)
        if episode_index > 0:
            name = f"{name}_ep{episode_file_suffix(episode_index):06d}"
        if err:
            self._log(tr("[错误] 上传后自动删除失败: {}", f"{name} ({err})"))
            return
        # 历史行保留（用户要求），状态标记"已上传，本地已删"
        RecordingRepo.mark_uploaded_deleted(
            session_path, episode_index=episode_index)
        self._refresh_history()
        self._log(tr("☁ 上传完成，本地文件已删除: {}", name))

    def _style_toolbar_toggle(self, action: QAction):
        """工具栏开关按钮高亮样式：开启绿色填充白字加粗，关闭灰边普通按钮。"""
        btn = self._toolbar.widgetForAction(action)
        if btn is not None:
            btn.setStyleSheet(
                "QToolButton { border-radius:3px; padding:2px 8px;"
                " border:1px solid #555; }"
                "QToolButton:checked { background:#2E7D32; color:white;"
                " font-weight:bold; border-color:#2E7D32; }"
            )

    def _on_upload_auto_toggled(self, on: bool):
        """"自动上传"开关——持久化，重启后保持；按钮文字同步开/关状态。"""
        settings.save_upload_auto_sync(bool(on))
        settings.UPLOAD_AUTO_SYNC = bool(on)   # 运行时立即生效（录制完成回调读此值）
        self._upload_auto_action.setText(
            tr("☁ 自动上传: {}", tr("开") if on else tr("关")))

    def _on_upload_delete_toggled(self, on: bool):
        """"上传后自动删除"开关——持久化，重启后保持；按钮文字同步开/关状态。"""
        settings.save_upload_delete_after(bool(on))
        settings.UPLOAD_DELETE_AFTER = bool(on)   # 运行时立即生效（上传完成回调读此值）
        self._upload_delete_action.setText(
            tr("🗑 上传后自动删除: {}", tr("开") if on else tr("关")))

    def _on_upload_task_failed(self, task_id: str, error: str):
        """自动上传失败——记录日志（重试 3 次后仍失败才触发）。"""
        pair = self._upload_task_map.pop(task_id, ("", 0))
        path = pair[0] if isinstance(pair, tuple) else pair
        episode_index = pair[1] if isinstance(pair, tuple) else 0
        name = os.path.basename(path) if path else task_id
        if episode_index > 0:
            name = f"{name}_ep{episode_file_suffix(episode_index):06d}"
        self._log(tr("[上传失败] {}: {}", name, error))

    def _open_upload(self):
        """打开上传对话框。"""
        session = getattr(self._task_service, '_session', None)
        dlg = UploadDialog(self, settings.RECORDING_DIR, session=session)
        dlg.exec_()

    # ═══════════════════════════════════════════════════
    #  使用说明窗口
    # ═══════════════════════════════════════════════════

    def _guide_settings(self) -> QSettings:
        return QSettings("DAQ", settings.APP_NAME)

    def maybe_show_guide(self):
        """经 start.bat / start.sh 启动（DAQ_SHOW_GUIDE=1）时弹出使用步骤窗口，
        勾选"下次不再显示"后不再弹出；直接启动仅首次弹出。
        勾选可随时经 帮助→使用说明 打开本窗口取消。"""
        qs = self._guide_settings()
        if qs.value("guide/dont_show", False, type=bool):
            return
        forced = os.environ.get("DAQ_SHOW_GUIDE") == "1"
        if not forced and qs.value("guide/shown_once", False, type=bool):
            return
        self._show_guide(qs)

    def _show_guide(self, qs: QSettings | None = None):
        from ui.guide_dialog import GuideDialog
        qs = qs or self._guide_settings()
        dlg = GuideDialog(self)
        dlg.exec_()
        # 每次关闭都按当前勾选状态落盘：勾选=不再自动弹出，取消勾选=恢复自动弹出
        qs.setValue("guide/dont_show", dlg.dont_show_checked())
        qs.setValue("guide/shown_once", True)
        qs.sync()   # 立即写盘，避免 QSettings 实例缓存延迟

    def _show_about(self):
        QMessageBox.about(
            self, tr("关于 DAQ 视频管线"),
            f"<b>{settings.APP_NAME}</b> v{settings.APP_VERSION}<br><br>"
            f"{tr('多路摄像机实时监控与录制系统')}<br>"
            f"• {tr('实时摄像机预览')}<br>"
            f"• {tr('可拖拽调整的画面布局')}<br>"
            f"• {tr('支持正常完成和异常停止两种录制模式')}<br>"
            f"• {tr('录制历史记录追踪')}<br>"
            f"• {tr('中英文界面切换')}"
        )

    # ═══════════════════════════════════════════════════
    #  安全退出
    # ═══════════════════════════════════════════════════

    def closeEvent(self, event):
        """关闭窗口前停止录制、释放资源。"""
        self._shutting_down = True
        self._s80m_manager.shutting_down = True
        self._device_timer.stop()
        self._device_scanner.stop()
        if self._pipeline.is_recording:
            self._pipeline.abort_recording("")
        self._teardown_all_workers()
        self._pipeline.remove_all()
        if self._hand_processor:
            self._hand_processor.cancel()
        if self._auto_labeler:
            self._auto_labeler.cancel()
        if hasattr(self, '_task_service'):
            self._task_service.stop()
        if hasattr(self, '_upload_manager'):
            self._upload_manager.stop()
        self._log(tr("DAQ 视频管线已关闭。"))
        event.accept()
