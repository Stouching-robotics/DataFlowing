# ui/

## 定位

`ui/` 是 DAQ 桌面主程序的 PyQt5 界面层：主窗口把设备枚举面板、多画面网格、录制控制、录制历史、任务选择、上传与回放全部串成图形界面，后台数据链路（管线、数据库、网络服务）均由本层发起的对象持有并注入。界面层本身不写数据文件，持久化一律经 `config.settings` 与 `core.*` 完成。

被以下模块引用（证据）：

- `main.py` `from ui.main_window import MainWindow` —— 主入口（Qt-Material 暗色主题），构造并 `show()`。
- `tools/tests/` 多个回归/冒烟测试直接构造界面类：`exposure_control_test.py`、`device_panel_gui_smoke_test.py`、`grid_drag_fps_test.py`、`test_playback_multifps.py`（还 import `_get_effective_fps`）、`d435_playback_test.py`、`glove_widget_test.py` 等。
- 包内互引：`ui/main_window.py`（CameraGrid/DevicePanel/ExposureDialog/LoginDialog/GuideDialog/PlaybackDialog/UploadDialog/TaskSelectionPage）、`ui/playback_dialog.py`（ZoomableVideoWidget、CameraGrid 及两个分割条常量）、`ui/camera_grid.py` 与 `ui/glove_widget.py`（CameraWidget）、`ui/login_dialog.py`（复用 `ui/guide_dialog.py` 的 `VisibleCheckBox`）。

注：磁盘上存在 `ui/hand_model/` 目录，但其中只剩未跟踪的 `__pycache__/`（已删除 `__init__.py`、`state.py` 的编译残留），无任何源码 `.py`，故不列入文件清单（其源码已在开源准备提交 8ea2f44 的死代码清理中移除）。

## 文件清单

| 文件 | 一句话作用 |
| --- | --- |
| `ui/main_window.py` | 主窗口：任务/采集双页面栈、设备面板路由（UVC/D435/S80M/蓝牙统一开关）、录制控制、录制历史表、上传与回放入口 |
| `ui/camera_grid.py` | 嵌套 QSplitter 多画面网格容器：画面拖拽调位、分割条调大小、空状态提示 |
| `ui/camera_widget.py` | 单路画面控件：`ZoomableVideoWidget`（滚轮缩放/拖拽平移/双击还原）+ `CameraWidget`（信息条、FPS、状态灯、☀ 曝光按钮） |
| `ui/device_panel.py` | 左侧设备检测面板：已连接设备统一分组列表（UVC/D435/S80M/手套/其他蓝牙），勾选开关、双击命名 |
| `ui/exposure_dialog.py` | 每设备曝光设置对话框（自动曝光开关 + 滑块，量程/单位语义由调用方注入） |
| `ui/glove_widget.py` | BLE 手套仿生手掌画面控件，直接嵌入主网格，录制数据经 `pipeline.write_sensor` 落盘 |
| `ui/login_dialog.py` | 启动/切换账号登录对话框：账号登录（对话框内同步校验）或游客登录，关闭窗口等价游客；结果经 `choice()/server_url()/username()/cookies()` 等访问器读取 |
| `ui/guide_dialog.py` | 使用说明窗口（首次启动自动弹出、帮助菜单重开；中英 HTML 模板；「不再显示」持久化到 QSettings）+ `VisibleCheckBox` 复选组件 |
| `ui/playback_dialog.py` | 录制回放对话框：统一网格同步播放全部视频 + 传感器渲染，深度槽识别、手部关键点叠加、批量删除 |
| `ui/task_page.py` | 启动时的任务选择页：任务表、服务器地址/凭据输入、连接状态、进入采集 |
| `ui/upload_dialog.py` | 录制会话一键打包上传对话框 |
| `ui/__init__.py` | 空包标记（0 行） |

## 各文件详解

### ui/main_window.py

**作用**：应用程序主窗口，界面层中枢。组合页面栈（任务选择页 + 数据采集页）、`CameraGrid` 主网格、左侧 `DevicePanel` 设备面板、底部日志 dock、右侧录制历史 dock、工具栏与菜单；启动时经 `_show_login_flow` 弹 `LoginDialog` 确定账号/游客身份，首次启动经 `_show_guide` 弹 `GuideDialog` 使用说明；持有 `CameraPipeline`、`DeviceScanner`、`TaskService`、`UploadManager` 与 `core.device_manager.DeviceManager` 注册表（`_workers` 为其 `entries` 别名）等核心对象并负责信号接线。所有设备的开启/关闭都经左侧面板开关按 `core.device_manager.dispatch_toggle` 分派到 `_open_*`/`_close_*` 系列方法：S80M 双目子进程 + 管道解析在 `core.s80m_manager.S80MDeviceManager`，D435 worker 生命周期与帧口径在 `core.d435_manager.D435DeviceManager`，手套以 BLE 引擎 + 渲染定时器运行。

**模块级辅助与常量**（v1.0.3 后多为 core 模块的兼容 re-export，离线测试直接 patch 这些名字）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_realsense_short` / `_slot_base` / `_normalize_original` | — | `core.device_naming` 的 `realsense_short`/`slot_base`/`normalize_original` 旧名兼容别名 | 同 core 定义 |
| `_LEGACY_CAM_NAMES` | 常量 | 旧录制记录的退化相机名（`"多路录制"`/`"Multi-Camera"`/`""`），历史表自愈判据 | — |
| `_STEREO_AVAILABLE` | 常量 | `core.s80m_manager.STEREO_AVAILABLE` 别名（自包含双目模块 `tools/stereo_s80m/read_stereo_rgb.py`，内含 libfays_vikit.so 3.9.0 与 yaml，与外部 SDK 目录独立） | — |
| `_D435_AVAILABLE` / `D435Worker` / `list_d400_devices` | 常量 | 导入 `core.d435_camera` 并缓存 `d435_available()`；导入失败全置 None/False（模块全局保留供离线测试 patch） | — |
| `_HAND_PROC_AVAILABLE` | 常量 | `core.hand_processor.SessionHandProcessor` + `core.auto_labeler.AutoLabeler` 可选导入结果 | — |
| `_GLOVE_AVAILABLE` | 常量 | `ui.glove_widget.GloveWidget` 可选导入结果（依赖 bleak 子系统） | — |

**类/信号**（`class MainWindow(QMainWindow)`）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `stereo_frame_ready` | `pyqtSignal(str, np.ndarray, object, list)` | 双目帧信号（FIFO 读取线程 → 主线程），参数 `slot_id, frame, hardware_ns, imu_samples` | 连接到 `_on_stereo_frame`。★ 封送坑：`hardware_ns` 必须用 `object`——PyQt5 队列信号把 Python int 按 C++ `qint32` 封送，超过 2^31（约 2.1s 纳秒）会静默截断为负数，录制数据受害 |
| `log_message` | `pyqtSignal(str)` | 跨线程日志信号（后台线程 → 主线程 `QTextEdit.append`） | `_log()` 在非主线程时经此投递 |
| `_upload_session_deleted` | `pyqtSignal(str, str)` | 上传后自动删除线程 → 主线程，参数 `(session_path, error_msg)` | 连接到 `_on_upload_session_deleted` |

**相机管理**（设备开启/关闭全在此区）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_auto_detect_cameras` | `()` | 工具栏「🔍 扫描」：立即请求一次设备扫描（接入统一走面板开关，不自动打开画面） | 记日志；`DeviceScanner.request_scan()` |
| `_teardown_all_workers` | `()` | 委托 `core.device_manager.DeviceManager.teardown_all(close_fns)` 按 kind 分类关闭全部已开启设备 | 清空注册表 |
| `_open_uvc` / `_close_uvc` | `(dev) -> bool` / `(dev_key)` | UVC 面板开关：增量建槽（`settings._camera_slot_name(dev.video_index)`）/ 按注册表拆槽；条目经 `DeviceManager.uvc_entry` 创建 | bool 成功与否 |
| `_open_s80m` | `(dev) -> bool` | 面板开关打开 S80M 双目：先注入静态标定（`config/s80m_stereo_calibration.json` → `StereoCalibration.load`），创建 `stereo_left`/`stereo_right` 两槽并注册外部帧源（fps=`settings.STEREO_RECORD_FPS`=30）；`s80m_depth_available()` 时追加第三格 `stereo_depth`（`set_depth_camera`，PNG16，v1.0.11），随后子进程/Popen/管道/watchdog/读线程全部交 `core.s80m_manager.S80MDeviceManager.spawn` | bool；与 D435 互斥（共用 video0/video2，后开者弹窗拒绝） |
| `_on_stereo_frame` | `(slot_id, frame, hardware_ns=0, imu_samples=None)` | 主线程槽（`S80MDeviceManager.frame_ready` 转投）：显示帧直推画面；录制时按 `core.s80m_manager.frame_record_decision`（1/30s 桶）抽帧；IMU 仅随 `stereo_left` 落盘 | `pipeline.write_external_frame(...)` |
| `_on_s80m_depth` | `(slot_id, depth, hardware_ns=0)` | `S80MDeviceManager.depth_ready` 槽：`depth_to_heatmap`（固定色标 + 中值滤波）→ 深度格显示；录制中 `pipeline.write_depth` | — |
| `_close_s80m` | `(dev_key)` | 委托 `S80MDeviceManager.close`（停 watchdog → 关 stdin → terminate/kill → join 读线程 → 清理临时文件），再注销两路外部帧源、`clear_depth_camera(stereo_depth)`、移除 UI 控件、清除本设备标定 | — |
| `_open_d435` | `(dev) -> bool` | 面板开关打开 RealSense：按 serial 活体复查 `list_d400_devices()`（不信任导入期缓存）、按 serial 查重、与 S80M 反向冲突检查；槽名 = `core.device_naming.allocate_slot_names` 派生 `{base}_rgb` + `{base}_depth`，RGB 注册外部帧源、深度注册深度伪相机（`set_depth_camera`，热力图 MP4 + PNG16 双通道，`raw_depth=True`）；worker 创建/信号接线/启动交 `core.d435_manager.D435DeviceManager.spawn` | bool；S80M 已开时弹窗拒绝 |
| `_on_d435_frames` | `(slot_id, frame, hardware_ns=0, imu_samples=None, dev_key=None)` | 委托 `core.d435_manager.D435DeviceManager.process_frame`（深度槽 → 热力图显示 + `write_depth`；RGB 槽 → 显示 + `write_external_frame`；首帧后标定送进管线 `calibration/head_stereo.json`） | 已关闭设备的迟到信号直接丢弃 |
| `_close_d435` | `(dev_key)` | 委托 `D435DeviceManager.close`（断信号 → stop → deleteLater）→ 注销外部帧源 + 清除深度伪相机 + 清除标定 → 移除双槽 | — |
| `_add_camera_slot` | `(slot_id, camera_index, backend="", label="")` | UVC 建槽：管线 `add_camera` + 网格 `add_camera` + `_connect_slot_signals` | 失败仅记日志 |
| `_connect_slot_signals` | `(slot_id)` | 单个 `CameraSlot` 信号 → UI 回调：`frame_ready`/`state_changed`/`error_occurred`/`camera_opened`/`fps_updated`，以及 widget 的 `remove_requested`/`exposure_clicked` | — |
| `_remove_camera` / `_remove_camera_slot_no_confirm` | `(slot_id)` | 移除单槽（录制中先弹确认）/ 不确认版；若槽属于某面板开启设备则整机关闭（d435 移除双槽之一即整机关）并取消勾选 | 同步面板勾选与高亮 |
| `_remove_all_cameras` | `()` | 工具栏「✕ 全部移除」：确认后先 `_teardown_all_workers()` 再逐槽移除 | 清面板勾选/高亮集合 |

**设备面板路由与曝光**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_device_label` | `(dev)` staticmethod | `DeviceInfo` → 日志显示标签（用户命名优先，有 serial 时附 `— serial`） | str |
| `_on_devices_scanned` | `(devices)` | 扫描结果（每 2s 轮询，`settings.DEVICE_POLL_INTERVAL_MS`=2000）：填用户命名 → `DevicePanel.set_devices` + 恢复勾选/高亮 → 插拔 diff 日志（只记变化）→ 录制中拔线暂存 `_lost_device_keys` 重插恢复勾选 → 已开启但已消失的设备逐个 `_handle_active_device_lost` | — |
| `_handle_active_device_lost` | `(key)` | 已开启设备被拔出：录制中不自动移除（重插恢复），否则按 kind 关设备 | — |
| `_on_device_toggled` | `(dev, on)` | 面板 `device_toggled` 路由：委托 `core.device_manager.dispatch_toggle`（录制锁双保险 + kind 路由到 `_open_*`/`_close_*` 回调，多路并发，互不拆台）；打开失败回退勾选 | 维护 `_active_device_keys` |
| `_on_device_renamed` | `(dev, name)` | 面板双击命名后：刷新插拔标签缓存与 worker 注册表标签 | 持久化已在面板完成 |
| `_show_exposure_button` | `(slot_id)` | 在设备的「主槽位」（S80M 左目、D435 的 RGB 槽）信息条显示 ☀ 按钮 | — |
| `_open_exposure_dialog` | `(slot_id)` | ☀ 入口：对话框参数口径在 `core.exposure_controller.exposure_dialog_params`（D435：worker 读回 µs 量程；S80M：1.0~885.0，`decimals=1`；UVC 无曝光入口固定自动曝光）；按类型弹 `ExposureDialog` | 模态 `exec_()` |
| `_apply_exposure` | `(dev_key, entry, auto, value)` | 委托 `core.exposure_controller.apply_exposure`（d435 → `worker.set_exposure`；s80m → `_s80m_set_exposure` 回调）并持久化（下次开启自动应用） | — |
| `_s80m_set_exposure` | `(dev_key, auto, value)` | 委托 `S80MDeviceManager.send_exposure`（stdin 行协议 `"SET_EXPOSURE <float>"`，`-1.0`=自动曝光），SDK 运行时生效无需重启 | BrokenPipe 时记日志 |
| `_set_exposure_buttons_enabled` | `(enabled)` | 录制锁：所有已显示的 ☀ 按钮随录制状态禁用/恢复 | 用 `isHidden()` 判断（窗口未 show 时 `isVisible` 恒 False） |
| `_open_glove` / `_close_glove` | `(dev) -> bool` / `(dev_key)` | 手套开关：广播名 `L`/`R` → `settings.assign_glove_sensor_role` 优先绑左右手；创建 `GloveWidget`（slot=`sensor:{key}`）注入管线并 `start(address)`，`pipeline.register_sensor(role)`；关闭时停 BLE、撤画面、注销传感器列 | bool |
| `_open_ble_placeholder` / `_close_ble_placeholder` | `(dev) -> bool` / `(dev_key)` | 无数据蓝牙（耳机类）：主网格显示「该设备无可视化数据」占位画面 | bool |

**录制**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_reset_s80m_record_state` | `()` | 委托 `DeviceManager.reset_s80m_record_state`：录制起止时重置 S80M 50→30 抽帧状态（`last_bucket` 桶号 + `pending_imu` 缓冲） | — |
| `_build_device_meta` | `() -> list` | 委托 `DeviceManager.build_device_meta`：手套无画面槽附 `sensor_column`；uvc/d435/s80m 附 `slots`（及 `serial`）；无数据蓝牙不进 devices 段 | list[dict] |
| `_record_all` | `()` | 工具栏「⏺ 全部录制」：任务名取自所选任务；所选任务进度已满（`completed_count >= total_required`，本地 `load_tasks()` 计数）时弹窗阻止；否则 `pipeline.start_recording(首个槽, task_name=..., batch_index=已完成数+1, device_meta=...)` | — |
| `_stop_all` | `()` | 「⏹ 完成录制」：`pipeline.finish_recording("")` + 重置抽帧状态 | — |
| `_abort_recording` | `()` | 「⛔ 异常终止」：`pipeline.abort_recording("")`（永久删除本次文件） | 记日志、刷新状态栏 |

**信号处理器（管线 → 控件）**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_on_slot_added` / `_on_slot_removed` | `(_slot_id)` | 空实现（控件已在 `_add_camera_slot` 中创建/移除） | — |
| `_on_frame` | `(slot_id, frame)` | UVC 帧 → 画面（`settings.CAMERA_MIRROR_HORIZONTAL` 时先水平镜像） | `w.update_frame` |
| `_on_camera_state` | `(slot_id, state)` | 状态 → 指示灯 + 日志；`CameraState.DISCONNECTED` 时 `pipeline.record_event(slot_id, "disconnected")`，否则记 `"connected"` | — |
| `_on_camera_fps` | `(slot_id, fps)` | UVC 由 `CameraWorker` 自带的 fps 信号 → 标签 | — |
| `_note_frame_arrival` | `(slot_id)` | 记录一帧到达时刻（D435/S80M 无 CameraWorker，主窗口统一计 FPS） | 写入 `_fps_ring` |
| `_update_fps_labels` | `()` | 每秒刷新非 UVC 槽的 FPS（过去 1 秒到达帧数） | — |
| `_on_recording_started` | `(slot_id)` | 录制开始：锁死设备面板开关 + 禁用曝光按钮 + 日志（含编码与任务名） | — |
| `_on_recording_finished` | `(slot_id, session_path)` | 录制完成：解锁面板/曝光；`core.helpers.session_summary(session_path, pipeline.last_recording_frames)` 算摄像机列/时长/大小 → `RecordingRecord` 存库 → 刷新历史；`increment_task_completed(实际任务名)` 更新任务进度；自动上传入队（开关开启且该会话尚未 `completed` 时 `_upload_manager.add_task`）；`HAND_TRACK_ENABLED` 时延迟 500ms 自动处理手部关键点（silent） | — |
| `_on_recording_aborted` | `(slot_id)` | 异常停止：解锁面板/曝光，取消手部后处理与标注 | — |
| `_on_duration` | `(_slot_id, seconds)` | 录制时长 → 状态栏 | — |

**录制历史**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_refresh_history` | `()` | `RecordingRepo.list_all(limit=100)` → 表格 5 列：摄像机/文件/时长/大小/状态；旧记录自愈：退化值（无大小、`_LEGACY_CAM_NAMES` 相机名、无时长）实时从磁盘 `session_summary` 补算并回写 DB；状态列按 `completed`/`uploaded`/`uploaded_deleted` 绿色（`uploaded` 含自愈兜底：`completed` 且目录仍在、上传任务表记为 `completed` 的旧行也显示「已上传」）、其余（已丢弃）异常色 | 启动时即加载 |

**上传 / 回放入口**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_open_playback` | `()` | 「📂 回放」：`PlaybackDialog(self, settings.RECORDING_DIR)` 模态打开 | — |
| `_open_upload` | `()` | 「☁ 上传」：`UploadDialog(self, settings.RECORDING_DIR, session=TaskService._session)` 模态打开 | — |
| `_on_upload_task_done` | `(task_id)` | 自动上传完成：记日志；`UPLOAD_DELETE_AFTER` 开启时后台线程删本地目录，否则 `RecordingRepo.mark_uploaded` 标「已上传（本地保留）」并刷新历史 | — |
| `_delete_session_after_upload` | `(session_path)` | 后台线程 `shutil.rmtree`（防大目录阻塞 UI），结果经 `_upload_session_deleted` 回主线程 | — |
| `_on_upload_session_deleted` | `(session_path, err)` | 删除收尾：失败记日志；成功 `RecordingRepo.mark_uploaded_deleted` 保留历史行（状态「已上传，本地已删」）并刷新 | — |
| `_on_upload_task_failed` | `(task_id, error)` | 自动上传失败（重试 3 次后仍失败才触发）→ 日志 | — |
| `_on_upload_auto_toggled` / `_on_upload_delete_toggled` | `(on)` | 两个工具栏开关：`settings.save_upload_auto_sync` / `save_upload_delete_after` 持久化 + 运行时立即生效 + 按钮文字/样式刷新 | — |
| `_style_toolbar_toggle` | `(action)` | 开关类 QAction 按钮高亮样式（开启绿底白字加粗） | — |

**任务服务与页面切换**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_connect_task_service` | `()` | `TaskService` 信号 ↔ `TaskSelectionPage`：tasks_updated/connection_status/login_result/identity_changed/identity_expired；任务页 refresh_requested→`poll_now`、switch_account_requested→`_on_switch_account`；错误首次出现只记一次（`_last_task_error` 判重） | 轮询在身份确定后（`_show_login_flow` 末尾）才 `start()` |
| `maybe_show_login` / `_show_login_flow` | `()` | 启动入口（main.py 调用）→ 登录流程：`LoginDialog` exec 后 `save_server_url`；账号登录成功经 `TaskService.adopt_login` 接管 cookie（`save_remembered_username`），否则 `set_guest`；同步 `_upload_manager.server_url` 与任务页 `set_server_display` 后 `task_service.start()` | — |
| `_on_switch_account` | `()` | 任务页「切换账号」按钮 → 重开登录对话框 | — |
| `_on_identity_expired` | `()` | 登录态轮询遇 401/403 → 提示重登；关闭对话框（Esc）自动降级游客 | — |
| `_on_task_confirmed` | `(task)` | 选中任务进入采集页（index 1），更新窗口标题；不自动扫描，提示先选模式再点扫描 | — |
| `_on_page_changed` | `(index)` | 页面切换控制工具栏/dock 可见性；任务页停设备轮询，采集页立即扫描 + 启动 `_device_timer`（2s 轮询） | — |
| `_go_back_to_task_selection` | `()` | 返回任务选择：录制中先确认（同意则 abort）；取消手部后处理/标注；`_teardown_all_workers` + 清网格 + 清面板状态；切回 index 0 并 `poll_now()` 刷新任务列表 | — |

**其余（语言/视图/手部后处理/日志/退出）**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_on_language_changed` | `(lang)` | 中英文切换：菜单/工具栏/dock 标题/历史表头/手部模式下拉全部 `tr()` 重刷，并调 `_task_page._on_language_changed` | — |
| `_reset_camera_sizes` | `()` | 视图菜单「重置画面大小」：`grid._rebuild_layout()` 等分 | — |
| `_update_status` | `()` | 状态栏：摄像机数（排除 `sensor:`/`ble:` 占位槽）+ 录制中标记与颜色 | — |
| `_process_hand_keypoints` | `(session_path="", silent=False)` | 后台提取手部关键点（`SessionHandProcessor.process_session`，mode 由手套/裸手下拉决定）；已有结果时弹确认（silent 模式跳过）；无路径用最近一条录制 | — |
| `_on_hand_proc_progress` / `_on_hand_proc_finished` | `(current, total)` / `(session_path, error)` | 进度 → 状态栏+按钮；完成后 300ms 自动触发 `_auto_label`（silent） | — |
| `_auto_label` / `_on_auto_label_finished` | `(session_path="", silent=False)` / `(session_path, error)` | 基于关键点 parquet 自动生成手势标签（`AutoLabeler.label_session`） | — |
| `_log` | `(msg)` | 时间戳日志：主线程直接 `append`，后台线程经 `log_message` 信号安全投递 | — |
| `_show_about` | `()` | 关于对话框（应用名/版本/功能简介） | — |
| `closeEvent` | `(event)` | 安全退出：置 `_shutting_down` → 停轮询/扫描器 → 录制中 abort → 拆全部 worker → 移除全部槽 → 取消手部后处理 → 停 TaskService/UploadManager | accept |

**关键数据**：

- `_workers` 设备注册表：`DeviceManager.entries` 的别名，`key → {"kind": "uvc"|"d435"|"s80m"|"data_ble"|"ble", "slots": [网格槽位], "label": 日志标签}`，另含 kind 专属字段（d435：`worker`/`rgb_slot`/`depth_slot`/`near_mm`/`far_mm`/`smooth_k`/`temporal_alpha`/`heat_smoother`/`calib_sent`/`serial`；s80m：`proc`/`stdin`/`watchdog`/`reader_thread`/`stderr_file`/`config_file`/`original_exp`/`entry_depth`/`last_bucket`/`pending_imu`；data_ble：`sensor_column`/`glove`）。
- 设备面板状态：`_active_device_keys`（勾选集合=UI 状态源）、`_lost_device_keys`（录制中拔线暂存）、`_last_device_keys`（key→标签，插拔 diff 日志）。
- FPS：`_fps_ring`（slot_id → 到达时刻 deque），1s 定时刷新。
- 历史表列名：`摄像机 / 文件 / 时长 / 大小 / 状态`。
- S80M 管道帧格式与曝光行协议见 `core/s80m_manager.py`，50→30 抽帧桶长 `settings.STEREO_RECORD_MIN_INTERVAL_S`。
- D435 槽名约定：RGB 槽 = 前缀 + `_rgb`，深度槽 = 前缀本身若以 `_depth` 结尾否则补 `_depth`；`depth` 后缀仅用于回放过滤，是否落盘由显式注册决定。
- meta 写入点：`_build_device_meta()` 产出 `devices` 段（`key`/`kind`/`name`/`slots`/`serial`/`sensor_column`）。

**调用关系**：被 `main.py` 构造。创建并持有：`core.pipeline.CameraPipeline`、`core.device_detector.DeviceScanner`、`core.device_manager.DeviceManager`、`core.s80m_manager.S80MDeviceManager`、`core.d435_manager.D435DeviceManager`、`core.task_service.TaskService`、`core.uploader.UploadManager`、`core.recording_repository.RecordingRepo`（静态调用）、`ui` 内全部其余组件。调用 `core.d435_camera.D435Worker`、`core.stereo_depth.StereoDepthComputer/depth_to_heatmap/DepthHeatmapSmoother`、`core.calibration.StereoCalibration`、`core.task_record.load_tasks/increment_task_completed`、`core.helpers.session_summary/format_duration/format_size_mb/hand_kpts_parquet_path`、`core.device_naming`/`core.exposure_controller` 口径；可选手部子系统 `core.hand_processor.SessionHandProcessor`、`core.auto_labeler.AutoLabeler`。

### ui/camera_grid.py

**作用**：多画面网格容器 `CameraGrid`（继承 `QScrollArea`），把每路画面排进嵌套 `QSplitter` 布局；所有画面（相机 `CameraWidget`、回放传感器格、手套仿生手掌、占位画面）同格共存。提供两种交互：按住画面顶部信息条拖到另一画面上松手即互换位置（事件过滤器 + 鼠标抓取）；画面间的分割条加宽（8px）并带悬停/按住高亮，可拖动调大小。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `add_camera` | `(slot_id: str, camera_name: str = "")` | 添加一个 `CameraWidget` 到网格 | 返回 `CameraWidget` |
| `add_widget` | `(slot_id: str, widget: QWidget)` | 添加任意控件（手套仿生手掌/传感器格/占位等）；重复 slot_id 直接返回已有控件 | 返回 QWidget；触发重建布局 |
| `remove_camera` | `(slot_id: str)` | 移除指定控件：先 `_end_drag()` 复位拖拽状态（防抓取/光标残留），再 `deleteLater()` | 触发重建布局 |
| `camera_widget` | `(slot_id: str)` | 按 slot_id 取控件 | `Optional[CameraWidget]` |
| `clear` | `()` | 移除所有画面 | — |
| `widget_count` | `() -> int` | 当前画面数量 | int |
| `slot_ids` | `() -> List[str]` | 全部 slot_id 列表（字典顺序即布局顺序） | list |
| `eventFilter` | `(obj, ev)` | 拖拽手柄事件：顶部 `DRAG_STRIP_H`（32px）内按下左键→`grabMouse` 抓取 + 闭合手掌光标；移动→悬停目标高亮（`DRAG_TARGET_BORDER`）；松开→`_swap_widgets` 互换位置；双击信息条不触发缩放还原；带 `_no_drag` 属性的控件（曝光按钮、模式下拉等）直接放行 | 消费事件时返回 True |
| `_rebuild_layout` | `()` | 按画面数重建布局：0 路=空提示文案（默认 i18n「尚未检测到摄像机…」，可 `empty_text` 覆盖）；1 路=填满；2 路=[A\|B]；3 路=[A\|B]/[C]；4 路=2×2；5+ 路按 `ceil(sqrt(n))` 行列嵌套分割；初始等分 | — |
| `_install_drag_filter` / `_remove_drag_filter` | `(w)` | 给画面控件及其全部子控件装/卸事件过滤器（子控件才真正接收鼠标事件） | — |
| `_root_widget` / `_in_drag_strip` / `_widget_at_global` | 辅助 | 事件来源控件沿父链找所属画面 / 判断全局坐标是否落在信息条内 / 命中画面 | QWidget 或 None |
| `_set_drag_hover` / `_swap_widgets` / `_end_drag` | 辅助 | 高亮悬停目标（恢复旧样式）/ 互换字典顺序并重建 / 释放抓取、还原光标与高亮 | — |

**关键数据**：

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `DRAG_STRIP_H` | 32 | 每画面顶部拖拽手柄信息条高度（像素） |
| `SPLITTER_HANDLE_WIDTH` | 8 | 分割条宽度（被 `ui/playback_dialog.py:29` 复用） |
| `SPLITTER_HANDLE_QSS` | 灰条 + 悬停/按住高亮 | 暗色主题下分割条可见性（被 `ui/playback_dialog.py:29` 复用） |
| `DRAG_TARGET_BORDER` | `border:2px solid #4FC3F7` | 拖拽悬停目标边框高亮 |

**调用关系**：被 `ui/main_window.py`（主网格）、`ui/playback_dialog.py`（回放统一网格）使用；`tools/tests/exposure_control_test.py`、`tools/tests/grid_drag_fps_test.py` 直接构造。依赖 `ui/camera_widget.py` 的 `CameraWidget` 与 `config.i18n.tr`（空提示文案）。

### ui/camera_widget.py

**作用**：单路画面的两个控件。`ZoomableVideoWidget` 是纯视频显示画布：接收 BGR `np.ndarray` 帧，内部 BGR→RGB 转 `QImage.Format_RGB888`（Qt x86 上 `Format_BGR888` 会红蓝互换），支持滚轮缩放（以鼠标位置为基准）、左键拖拽平移、双击还原、缩放百分比提示。`CameraWidget` 把视频画布 + 顶部覆盖条（名称、☀ 曝光按钮、FPS 标签、状态灯）组合成一路「画面」，供主网格与回放共用。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `ZoomableVideoWidget.reset_view` | `()` | 还原缩放（1.0）与平移（0,0） | 重绘 |
| `ZoomableVideoWidget.set_frame` | `(bgr_frame: np.ndarray, flip_vertical: bool = False)` | 传入 BGR 帧：可选上下翻转（摄像机视频需要，传感器渲染不需要），BGR→RGB 后存为 `QPixmap` | 触发重绘 |
| `ZoomableVideoWidget.set_status_text` | `(text: str)` | 设置无画面时居中显示的文字（如「无信号」「已断开」） | 清空画面 |
| `ZoomableVideoWidget.wheelEvent` | `(event)` | 滚轮缩放：1.08×/级，范围 0.25×~8.0×，以鼠标位置为基准调偏移，右下角短暂显示百分比 | 重绘 |
| `ZoomableVideoWidget.mousePressEvent` / `mouseMoveEvent` / `mouseReleaseEvent` / `mouseDoubleClickEvent` | `(event)` | 左键拖拽平移（闭合手掌光标）；双击 `reset_view` | — |
| `CameraWidget`（信号） | `remove_requested(str)` / `exposure_clicked(str)` | 请求移除此摄像机 / 点击 ☀（主窗口按槽位所属设备弹曝光对话框） | 由 `MainWindow._connect_slot_signals` 连接 |
| `CameraWidget.update_frame` | `(bgr_frame)` | 相机采集帧更新显示（`flip_vertical=True` 适配安装方向） | — |
| `CameraWidget.update_fps` | `(fps: float)` | 更新 `FPS: n` 标签 | — |
| `CameraWidget.set_frame_number` | `(n: int)` | 回放用：信息条右侧显示当前帧号 `#n`（复用 FPS 标签位） | — |
| `CameraWidget.set_camera_state` | `(state: str)` | `recording`/`error`/`disconnected`/`idle` → 状态灯颜色与「已断开/等待中…」文字 | — |
| `CameraWidget.set_exposure_button_visible` / `set_exposure_enabled` | `(visible: bool)` / `(enabled: bool)` | 显示/隐藏 ☀（只对设备主槽位）；启用/禁用（录制中禁用，tooltip 随之切换） | — |
| `CameraWidget._refresh_texts` | `(_lang="")` | 语言切换刷新文案 | 连接 `lang_manager.language_changed` |
| `CameraWidget.resizeEvent` | `(event)` | 调整覆盖条宽度与位置 | — |

**关键数据**：

- `ZOOM_MIN = 0.25`、`ZOOM_MAX = 8.0`、`ZOOM_STEP = 1.08`。
- 覆盖条：固定高 28px、初始宽 260px（随 resize 收缩到 ≤300px），位于画面左上角；刻意不设 `WA_TransparentForMouseEvents`（会连 ☀ 按钮一起屏蔽），画面信息条拖拽由 `CameraGrid` 事件过滤器统一拦截。
- ☀ 按钮带 `_no_drag` 属性标记，`CameraGrid.eventFilter` 对按下直接放行（不触发拖拽/双击还原）。
- 注意：`reset_view` 在文件中定义了两次（第 74-78 行与第 109-113 行），后者覆盖前者，行为一致。

**调用关系**：被 `ui/camera_grid.py`、`ui/glove_widget.py`（`GloveWidget` 继承 `CameraWidget`）、`ui/playback_dialog.py`（`ZoomableVideoWidget`）使用。依赖 `config.settings`（`FEED_MIN_WIDTH` 等尺寸与颜色常量）、`config.i18n.tr/lang_manager`、`core.helpers.format_duration`（导入存在，当前代码未直接调用）。

### ui/device_panel.py

**作用**：左侧停靠的设备检测面板。`DeviceScanner`（`core/device_detector`）每 2s 扫描的结果由 `MainWindow._on_devices_scanned` 调 `set_devices` 重建列表；三组展示（📷 相机 / 🧤 手套 / 🎧 其他蓝牙），「其他蓝牙」默认折叠、空组隐藏；每台设备带独立 `QCheckBox`，勾选发射 `device_toggled(DeviceInfo, bool)` 由主窗口路由到主网格显示；双击条目弹命名框并持久化到 `device_names.json`（此后每次连接显示用户命名）。

**类/函数**（`class DevicePanel(QWidget)`）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `device_toggled` / `device_renamed`（信号） | `pyqtSignal(object, bool)` / `pyqtSignal(object, str)` | 勾选变化（DeviceInfo, checked）/ 双击命名完成（DeviceInfo, new_name） | 由 `MainWindow` 连接 |
| `set_devices` | `(devices: list)` | blockSignals 下清空重建分组树；勾选状态由随后的 `set_checked_keys` 恢复；组展开状态跨重建保留 | 空列表显示「未检测到设备」占位 |
| `set_checked_keys` | `(keys: set)` | 按 key 集合静默勾选（不发信号） | — |
| `set_checked` | `(key: str, on: bool)` | 程序化勾选单个（不发信号，供失败回退） | — |
| `checked_keys` | `() -> set` | 当前勾选的设备 key 集合 | set（开关状态=UI 状态源） |
| `set_active_keys` | `(keys: set)` | 高亮正在显示中的设备（绿色前景），其余还原；空集合全还原 | — |
| `set_active_key` | `(key)` | 兼容旧调用：单 key 高亮（None 全还原） | — |
| `set_locked` | `(on: bool)` | 录制中锁死全部开关（去 `Qt.ItemIsEnabled`，命名同步禁用） | — |
| `key_for_kind` | `(kind: str)` | 列表中第一个 kind 匹配的设备 key | str 或 None |
| `device_for_key` | `(key: str)` | key → `DeviceInfo`（存在 `Qt.UserRole`） | DeviceInfo 或 None |
| `key_for_serial` | `(serial: str)` | 按序列号找 key（多台 RealSense 时精确高亮） | str 或 None |
| `key_for_video_index` | `(video_index: int)` | 第一个 `video_index` 匹配的设备 key | str 或 None |
| `refresh_texts` | `()` | 语言切换刷新提示文字、组标题与占位项 | — |
| `_row_text` | `(dev) -> str` | 行文本 = 图标 + `label`（+ `— serial`） | — |
| `_on_item_changed` | `(item, column)` | 勾选状态真正变化（用 `_last_check` 挡文字类变化与组标题）才发射 `device_toggled` | — |
| `_on_item_double_clicked` | `(item, _column)` | 双击 → `QInputDialog` 命名 → `settings.save_device_name(dev.stable_key, name)` 持久化 → 更新行文本 + `device_renamed`；对话框打开期间列表可能已被 2s 轮询重建，必须用当前列表里的条目更新 | — |

**关键数据**：

- `_ICON = {"uvc": "📹", "d435": "🔭", "s80m": "👁", "data_ble": "🧤", "ble": "🎧"}`；`_GROUP_ORDER = ["camera", "glove", "other_ble"]`；`_GROUP_TITLE` 对应中文组标题。
- `_items`：device key → `QTreeWidgetItem`；`_last_check`：key → 上次勾选状态（防 `itemChanged` 误触发）；`_group_expanded`：组展开状态（初始 camera/glove 展开、other_ble 折叠）。
- 命名持久化：`config.settings.save_device_name`（device_names.json）。

**调用关系**：由 `ui/main_window.py` 构造并连接 `device_toggled`/`device_renamed`；`tools/tests/device_panel_gui_smoke_test.py` 直接构造。依赖 `config.settings.save_device_name`、`config.i18n.tr`。

### ui/exposure_dialog.py

**作用**：每设备曝光设置模态对话框：自动曝光开关 + 曝光值滑块（内部统一 0..1000 刻度映射），拖动即时下发 `apply_requested(auto, value)`，勾选自动曝光立即生效。范围/值语义由调用方（`MainWindow`）按设备类型注入：UVC 是 V4L2 原始曝光值、D435/D405 是 µs（流启动后读 `rs.option.exposure` 量程）、S80M 是 SDK 曝光值 1.0~885.0（与 yaml `stereo_init_exposure` 同单位）。对话框只发参数不改持久化（持久化由主窗口统一处理）。

**类/函数**（`class ExposureDialog(QDialog)`）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `apply_requested`（信号） | `pyqtSignal(bool, float)` | 参数变化：`apply(auto, value)`（`auto=True` 时 value 忽略） | 由 `MainWindow._open_exposure_dialog` 连接到 `_apply_exposure` |
| `__init__` | `(parent, title, vmin, vmax, value, auto, decimals=0, original=None)` | 构造：自动曝光勾选框 + 0..1000 滑块 + 数值标签 + 「恢复默认」按钮（仅传入 `original` 时显示）+ 关闭 | 模态 |
| `_to_ticks` / `_from_ticks` | `(value) -> int` / `(ticks) -> float` | 曝光值 ↔ 0..1000 刻度线性映射 | — |
| `_fmt` | `(value) -> str` | 按 `decimals` 格式化数值标签 | — |
| `_on_auto_toggled` | `(checked)` | 自动曝光开关：禁用/启用滑块并立即 `apply_requested.emit(checked, 当前值)` | — |
| `_on_moved` / `_on_released` | `(ticks)` / `()` | 拖动中/松手时更新值并下发（`auto` 时忽略） | — |
| `_on_reset` | `()` | 「恢复默认」：控件静默（blockSignals）回到「最一开始」曝光基线 `(auto, value)` 并下发一次 | — |

**关键数据**：`_original` = 设备首次开启、应用任何设置之前读回的 `(auto, value)` 基线（来源为 `settings.device_original` 或 worker 读回值，经 `core.device_naming.normalize_original` 归一化）。

**调用关系**：由 `ui/main_window.py`（`_open_exposure_dialog`）按设备类型构造并 `exec_()`；`tools/tests/exposure_control_test.py` 直接构造。依赖 `config.i18n.tr`。

### ui/glove_widget.py

**作用**：BLE 手套仿生手掌画面控件（继承 `CameraWidget`，复用其覆盖条），直接嵌入主网格，替代旧的底部传感器 dock。面板开关打开手套后由主窗口创建；内部持有 `core.ble_engine.SensorBLEEngine`，30ms 渲染定时器处理一帧数据并用 `core.render_engine.render_hand` 画成 BGR 帧显示；录制中数据经 `pipeline.write_sensor` 写入 parquet 对应传感器列。

**类/函数**（`class GloveWidget(CameraWidget)`）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `__init__` | `(slot_id, address, sensor_column, label="", parent=None)` | 固定 `_RENDER_W=1280`×`_RENDER_H=720` 画布；左/右手套配置由 `core.sensor_hand_config.load_sensor_hand_config(sensor_column)` 加载（左/右不同配置文件）；启动 30ms 渲染定时器 | — |
| `start` | `(address: str = "")` | 连接 BLE 设备（懒创建引擎并连接 `connected`/`disconnected`/`fps_updated`/`calibration_progress`/`error_occurred` 信号） | 画面显示「连接中…」 |
| `stop` | `()` | 断开连接、停止渲染 | 画面显示「已断开」 |
| `set_pipeline` | `(pipeline)` | 主窗口注入/清除当前录制管线引用 | — |
| `_on_connected` / `_on_disconnected` | `(addr)` / `()` | 连接状态 → 画面文字 + `pipeline.record_event(sensor_column, "connected"/"disconnected")` | — |
| `_on_error` | `(msg)` | 引擎错误显示到画面状态栏（失败原因不再只有控制台可见） | — |
| `_on_fps` | `(fps)` | 硬件 FPS 显示到信息条（`HW: n`） | — |
| `_on_calib_progress` | `(progress)` | 校准进度显示（`校准中… n%` / `校准完成`） | — |
| `_render_tick` | `()` | 定时器触发：`engine.process_frame()` 拿一帧；录制中 `pipeline.write_sensor(processed, capture_ts, sensor_name=sensor_column)`；`render_hand(...)` 渲染并传给画面（逻辑平移自旧 `SensorPanel`） | — |
| `_display_frame` | `(frame)` | 渲染帧叠加传感器名 + 硬件 FPS + 数据帧龄（`Age: n ms`，按 <100/<300ms 绿/黄/红） | — |

**关键数据**：

- 渲染配置常量 `CONFIG_FILE`/`CONFIG_FILE_LEFT` 定义在 `core.render_engine`（指向 `config/sensors/hand_ble_config.json` 与 `hand_ble_config_left.json`）；加载与合并口径在 `core.sensor_hand_config.load_sensor_hand_config`。
- `_current_vmax` 初始 5000.0，跨帧回传保持色标稳定。

**调用关系**：由 `ui/main_window.py`（try/except 可选导入）在 `_open_glove` 中创建。依赖 `core.ble_engine.SensorBLEEngine`、`core.render_engine`、`core.sensor_hand_config`、`ui.camera_widget.CameraWidget`；`tools/tests/glove_widget_test.py` import 本模块做自检。

### ui/login_dialog.py

**作用**：登录对话框——启动时（或任务页点「切换账号」）弹出，确定会话身份。账号登录在对话框内同步校验（`TaskService.verify_credentials`，独立 `requests.Session`，≤8s），失败就地提示可重试，成功后携带 auth cookie 返回；游客登录直接进入（仅见公共任务）；关闭窗口（Esc/X）等价游客——后端不可达时不把用户锁死在启动页。

**类/函数**（`class LoginDialog(QDialog)`）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `choice` | `() -> str` | 结果："login"（点登录且校验成功）或 "guest"（游客/关闭窗口） | str |
| `server_url` | `() -> str` | 地址输入框内容；空则回退 `settings.SERVER_URL` | str |
| `username` / `password` | `() -> str` | 账号/密码输入框内容 | str |
| `remember_checked` | `() -> bool` | 「记住账号」勾选状态 | bool |
| `cookies` | `()` | 登录成功后的 auth cookie（失败/游客为 None） | cookie 或 None |
| `_on_login_clicked` | `()` | 空用户名就地提示；同步校验（按钮置「登录中…」防误以为卡死），成功记 cookie 并 accept，失败 `_show_error` | — |
| `_on_guest_clicked` | `()` | choice="guest" 并 accept | — |

**关键数据**：表单回填——服务器地址（与出厂默认相同则不回填、以 placeholder 显示）、用户名（`settings.load_remembered_username`）；错误提示仅登录失败时显示。

**调用关系**：由 `ui/main_window.py`（`_show_login_flow`）构造。依赖 `config.settings`、`config.i18n.tr`、`core.task_service.TaskService.verify_credentials`、`ui.guide_dialog.VisibleCheckBox`。

### ui/guide_dialog.py

**作用**：使用说明窗口——首次启动自动弹出（`guide/shown_once` 未置位，或环境变量 `DAQ_SHOW_GUIDE=1` 强制），也可从菜单 帮助 → 使用说明 随时重开；「下次启动不再自动显示」勾选持久化到 QSettings（`guide/dont_show`，取消勾选即恢复自动弹出；每次关闭按当前勾选落盘并 `sync()` 立即写盘）。正文为整段 HTML 使用步骤（登录 → 设备 → 录制 → 回放 → 上传），按当前界面语言选择 `GUIDE_HTML_ZH`/`GUIDE_HTML_EN` 模板，与根目录 `使用说明.md` 内容一致。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `guide_html` | `() -> str` | 按 `lang_manager.current` 返回当前语言的 HTML 正文 | str |
| `VisibleCheckBox` | `(QCheckBox)` | 暗色主题下勾选指示不明显：指示框加显式描边/填充，勾选时自绘白色 ✓（`login_dialog.py` 复用） | — |
| `GuideDialog` | `(parent=None)` | 760×640 对话框：标题（APP_NAME + 版本）、`QTextBrowser` 渲染 HTML、底部 VisibleCheckBox + 「关闭」 | — |
| `dont_show_checked` | `() -> bool` | 供调用方持久化「下次不再显示」 | bool |

**调用关系**：由 `ui/main_window.py`（`_show_guide`，惰性 import）构造；`VisibleCheckBox` 被 `ui/login_dialog.py` 复用。依赖 `config.settings`、`config.i18n`。

### ui/playback_dialog.py

**作用**：本地录制回放对话框。左侧会话列表（多选、按上传状态筛选、批量移入回收站），右侧复用 `CameraGrid` 的统一网格：摄像机格与传感器格同格共存，可拖拽调位、分割条调大小；底部播放控制（播放/暂停、逐帧、进度条、倍速 0.25~4×、手部关键点叠加开关、追踪模式、提取关键点）。时间戳对齐经 Parquet 统一时间线（`frame_idx` 与 `timestamp_us`）；会话加载（parquet 合并 + 手部关键点）全部放后台线程，视频打开与播放留在主线程（OpenCV FFmpeg 后端不承诺跨线程安全）。

**模块级**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_get_effective_fps` | `(info: dict) -> float` | `core.session_catalog.get_effective_fps` 的兼容别名（离线测试仍按旧名 import） | float |

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `SessionLoader` | `(parent=None)` QObject | `core.session_loader.SessionLoader` re-export：后台加载器，parquet 读取全在 worker 线程，`loaded(str, object, int)` 携带 `{"timeline": SensorTimeline, "hand_kpts": dict}`；`load(session_dir, gen)` 已有加载在跑时排队（新请求覆盖旧请求，过期结果被 gen 丢弃）；`cancel()` 取消 | 信号 `loaded`/`failed(int, str)` |
| `PlaybackSensorWidget` | `(sensor_name, title="", parent=None)` QFrame | 单路传感器回放格：标题 + 模式下拉（🔥 热力图/📝 轨迹/📊 网格/🦾 仿生手掌/🕸 形变网格，默认仿生手掌）+ `ZoomableVideoWidget` + TS 标签；头部行 ≈32px 与网格拖拽手柄区对齐，`mode_combo`/`ts_label` 标 `_no_drag` 放行点击 | — |
| `PlaybackDialog._toggle_fullscreen` | `()` | F11 切换全屏（`QShortcut` WindowShortcut 上下文，任意子控件聚焦时可用） | — |
| `PlaybackDialog._rebuild_grid` | `()` | 按当前会话重建统一网格：先摄像机格（双目四路 `stereo_left/right` + `_aux` 保持 2×2 特殊排列），后传感器格（slot_id=`sensor:{name}`，与主窗口手套格约定一致）；摄像机格隐藏状态灯 | — |
| `_browse_dir` / `_refresh_list` | `()` | 选择录制目录 / 重建会话列表（`core.session_catalog.list_sessions`；每项含上传状态图标 ✅/❌/⬜、路数、FPS、时长、任务 tag） | — |
| `_on_session_clicked` | `(item)` | 点击会话 → `_load_session(path)` | — |
| `_filter_by_upload` | `(uploaded: bool)` | 「⬆ 选中已上传」/「⬇ 选中未上传」：按 `UploadManager.get_upload_status` 勾选 | — |
| `_delete_selected` | `()` | 勾选会话移入回收站：先停播并释放全部 `cv2.VideoCapture` 句柄（否则 Windows 阻止删除）+ 强制 GC；`core.helpers.send_to_recycle_bin` 逐个移入 | 可逆 |
| `_load_session` | `(session_dir)` | 加载会话：主线程同步读元数据（`core.session_catalog.load_session_meta`：格式探测 → EgoData `metadata.json` 或 `meta/info.json`；传感器列表读 `info["sensors"]`，无则从 `info["features"]` 的 `observation.*` 键推断，最旧格式回退 `["state"]`）；parquet 合并 + 关键点交 `SessionLoader` 后台加载 | — |
| `_on_session_loaded` | `(gen, payload)` | 主线程槽：过期 gen 丢弃；过滤幽灵传感器列（时间线中确有 16×16=256 宽压力矩阵的才留，防 `observation.imu` 等非手套特征被误判）；打开每路 MP4（深度槽取 `depth/<slot>/<slot>.mp4`）；用户命名叠加（`device_names` + `devices` 段 slots/sensor_column → name）；算主时钟（info fps → 视频实际 fps → 全局兜底）、每路总帧数；重建网格、`_seek(0)` | — |
| `_on_session_load_failed` | `(gen, error)` | 后台加载失败 → 信息标签 | — |
| `_toggle_play` / `_stop` | `()` | 播放/暂停切换；`_stop` 也用于切会话前收尾 | — |
| `_tick` | `()` | 定时器推进：按真实流逝时间补帧追赶（主线程被 seek/渲染拖慢时保持原速）；到末尾自动停 | — |
| `_prev_frame` / `_next_frame` | `()` | 逐帧进退 | — |
| `_seek` | `(idx: int)` | 主时钟帧号 → 每路帧号（`t_s × fps_i` 四舍五入，低帧率路按比例抽帧）；小步前进（≤5 帧，定时器抖动/追赶所致）顺序 `read()` 到底（每帧 1-2ms），随机大跳（拖进度条）才 `set+read` seek（43-89ms/路）——小步也 seek 会因 seek 耗时把追赶 steps 推大形成死亡螺旋（HEVC B 帧 seek 更贵）；随后 `_update_sensor` + 时间标签 | — |
| `_on_slider_pressed` / `_on_slider_moved` / `_on_slider_released` | `(value)` | 拖动中不 seek（每秒数十个 move 事件 × 单次 90-180ms 会卡死），松手单次 seek；点击滑槽未拖动也跳到点击处；原在播则恢复播放 | — |
| `_set_video_frame` | `(slot_id, frame, frame_num)` | 一帧显示到摄像机格 + 帧号 `#n`；可选手部关键点叠加（`draw_kpts_overlay`）；MP4 帧录制时已 `np.flip(axis=0)` 过，回放不再翻转 | — |
| `_on_sensor_mode_changed` | `(sensor_idx, mode_idx)` | 切换传感器可视化模式（heatmap/trace/grid/hand/deform），trace 时 `clear_trace_canvas()`，vmax 重置 5000.0 | — |
| `_update_sensor` | `(frame_idx, t_s=0.0)` | 渲染全部传感器：多帧率混合会话按主时钟时间二分（`nearest_for_column_time`），帧率一致会话走帧号二分（`nearest_for_column`，时间戳含暂停负跳变时更稳）；16×16 矩阵按模式调 `render_heatmap/trace/grid/hand/deform_mesh`；TS 标签显示传感器时间戳与 Δ（ms/帧） | — |
| `_cycle_speed` | `()` | 倍速循环 0.25/0.5/1/2/4×，播放中即时改定时器间隔 | — |
| `_toggle_hand_overlay` | `(checked)` | 手部关键点叠加开关 → 重刷当前帧 | — |
| `_process_current_kpts` / `_on_hand_proc_finished` | `()` / `(session_path, error)` | 对当前会话后台提取手部关键点（主线程延迟创建 `SessionHandProcessor` 保证信号正常）；完成后 kpts-only 重载（`load_timeline=False`） | — |
| `closeEvent` | `(event)` | 置 `_closing`（迟到 loader 回调丢弃）、取消手部处理、停播、释放全部 VideoCapture | — |

**关键数据**：

- 深度槽判定（`_on_session_loaded`）：`cameras[slot]` 的 `type == "depth"` 或槽名以 `_depth` 结尾（槽名约定与主窗口 `_open_d435` 一致）；深度槽仅当其热力图 MP4（`depth/<slot>/<slot>.mp4`）存在时加入回放列表。
- 双目四路特殊排序：`_STEREO_IDS = {"stereo_left", "stereo_right", "stereo_left_aux", "stereo_right_aux"}` 四路齐备时按固定顺序排列。
- 传感器状态列表：`_sensor_modes`/`_sensor_vmax_list`/`_sensor_mesh_states`/`_sensor_widgets`/`_sensor_ts_labels`/`_sensor_hand_configs`/`_sensor_cells` 按 `info["sensors"]` 动态创建。
- meta 字段读取点：`cameras[].fps`、`cameras[].type`、`sensors`、`features`、`device_names`、`devices[].slots/sensor_column/name/kind`、`task_name`、`fps`（`info["fps"]` 兜底）；EgoData 会话的 devices 全量段在 `metadata.json`、device_names 在 `meta/info.json`，两处合并读。
- 会话列表项把上传状态存在 `Qt.UserRole + 1`。

**调用关系**：由 `ui/main_window.py`（`_open_playback`）构造；`tools/tests/test_playback_multifps.py`、`tools/tests/d435_playback_test.py` 直接构造。依赖 `core.session_timeline.load_timeline/SensorTimeline`、`core.session_catalog`（fps/扫描/元数据）、`core.session_loader.SessionLoader`、`core.render_engine` 五种渲染器与 `CONFIG_FILE`/`CONFIG_FILE_LEFT`、`core.sensor_hand_config`、`core.helpers` 路径与扫描函数、`core.uploader.UploadManager.get_upload_status`；可选手部叠加来自 `core.hand_tracking`。

### ui/task_page.py

**作用**：启动时显示的任务选择页（页面栈 index 0）。顶栏只读显示服务器地址（`set_server_display`）与当前身份，附「切换账号」按钮（发射 `switch_account_requested`）；中部任务表（4 列），底部「→ 进入采集」与「🗑 删除」。任务数据来自 `data/tasks.json`（`core.task_record.load_tasks`），后端任务经 `merge_backend_tasks` 合并，并 `set_identity` 按身份过滤（游客仅见公共任务）；用户选中任务点「进入采集」发射 `task_selected(dict)`，主窗口切换到数据采集页。

**模块级**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_compute_status` | `(task: dict) -> str` | 按 `total_required`/`completed_count` 算状态：`pending`/`in_progress`/`completed` | str |
| `_tid` | `(task: dict) -> str` | 取任务 id（`id` 或 `task_id`） | str |
| `_format_date` | `(iso_str: str) -> str` | 日期截取前 10 字符 | str |

**类/函数**（`class TaskSelectionPage(QWidget)`）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `task_selected` / `refresh_requested` / `switch_account_requested`（信号） | `pyqtSignal(dict)` / `pyqtSignal()` / `pyqtSignal()` | 点击「进入采集」/ 点击「🔄 刷新」/ 点击「切换账号」 | 由 `MainWindow._connect_task_service` 连接 |
| `update_tasks` | `(tasks: list[dict])` | 后端任务列表到达：`merge_backend_tasks` 合并（空则回退本地 `load_tasks`）→ 重算状态 → 重建表 | — |
| `update_task_progress` | `(task_id: str, completed_count: int)` | 录制完成回写进度：`refresh_progress()` 后按 `_tid` 找行刷新 | — |
| `set_connection_status` | `(connected: bool)` | 更新连接圆点与「后端已连接/未连接」文案 | — |
| `on_login_result` | `(ok: bool, msg: str)` | 登录结果反馈（`登陆成功` / `登陆失败: msg[:30]`） | — |
| `set_identity` | `(identity: str \| None)` | 更新当前身份（"guest" 或用户名）并即时按新身份重滤任务表 | — |
| `set_server_display` | `(url)` | 更新顶栏服务器地址显示与 tooltip（登录对话框改地址后由主窗口调用） | — |
| `current_task` | `() -> dict \| None` | 当前选中行任务 | dict 或 None |
| `_on_cell_changed` | `(row, _col, _pr, _pc)` | 选行联动：completed 任务禁用「进入采集」（tooltip 提示换任务） | — |
| `_on_double_clicked` / `_on_enter_clicked` | `()` | 双击行=快捷进入；进入前 completed 任务弹警告 | 发射 `task_selected` |
| `_do_delete_task` | `()` | 删除任务（按钮与右键菜单共用）：确认后 `mark_hidden`（可在 tasks.json 中恢复）→ 重载列表 | — |
| `_on_context_menu` | `(pos)` | 右键菜单「🗑 删除任务」 | — |
| `_on_refresh_clicked` | `()` | 发射 `refresh_requested`，按钮 1.5s 内显示「刷新中…」禁用 | — |
| `_on_language_changed` | `(lang)` | 表头/占位符/按钮文字刷新并重建表 | — |

**关键数据**：

- 表格 4 列：`任务名称 / 状态 / 进度 / 发布时间`（`_COL_NAME=0` … `_COL_DATE=3`）；状态列用彩色方块+文字的 badge（`_STATUS_CONFIG`：pending 灰 `#757575`、in_progress 蓝 `#42A5F5`、completed 绿 `#66BB6A`）；进度列 `completed/total` 右对齐。
- 身份与地址：服务器地址/记住账号由 `LoginDialog` 校验后经 `MainWindow._show_login_flow` 写入（`settings.save_server_url` / `save_remembered_username`）；登录态 cookie 由 `TaskService.adopt_login` 接管；身份过滤 scope 经 `_identity_scope()` 传给 `core.task_record`。

**调用关系**：由 `ui/main_window.py` 构造；任务服务信号在 `_connect_task_service` 中双向接线。依赖 `core.task_record`（`load_tasks`/`merge_backend_tasks`/`refresh_progress`/`mark_hidden`）、`config.settings`、`config.i18n`。

### ui/upload_dialog.py

**作用**：录制数据一键上传对话框：顶部服务器地址 + 「测试连接」（`core.api_client.APIClient.health_check`），中部会话列表（多选、全选、按上传状态筛选），底部进度条与状态栏。「⬆ 一键上传」把勾选会话入队 `UploadManager`（复用已认证的 `requests.Session`）；每项成功后遵循「上传后自动删除」开关：开启时后台线程删本地目录并把 `recording` 记录标记为 `uploaded_deleted`，否则标 `uploaded`（本地保留）；成功/失败结果均经 `_parent_log` 写进主窗口日志面板（与自动上传同一口径）。

**模块级**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| （无） | — | 会话扫描用 `core.session_catalog.list_recordings`（轻量版，不读元数据） | 按名称倒序 |

**类/函数**（`class UploadDialog(QDialog)`）：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_session_deleted`（信号） | `pyqtSignal(str, str)` | 后台删除线程 → 主线程：`(session_path, error)` | 连接 `_on_session_deleted` |
| `__init__` | `(parent=None, data_dir="", session=None)` | `session` 为复用已认证的 `requests.Session`；`QTimer.singleShot(50, _init_manager)` 延迟建 `UploadManager`（读 `_url_edit` 当前值） | — |
| `_refresh_list` | `()` | 重建会话列表：每项图标按 `UploadManager.get_upload_status`（✅ completed / ❌ failed / ⬜ pending），默认勾选未完成项；「全选」用 blockSignals 复位防 `_toggle_all` 连锁 | — |
| `_on_item_clicked` | `(item)` | 点击整行切换勾选（不必精确点复选框） | — |
| `_toggle_all` | `(checked)` | 全选/全不选 | — |
| `_filter_uploaded` | `(uploaded)` | 「⬆ 选中已上传」/「⬇ 选中未上传」 | — |
| `_test_connection` | `()` | `APIClient(url).health_check()` 弹结果框 | — |
| `_start_upload` | `()` | 勾选会话（过滤不存在的目录）`_manager.add_tasks(valid)` 入队并 `start()`；`_task_path_map` 记 task_id → session_path；禁用上传按钮 | — |
| `_on_status` | `(task_id, msg)` | 串行处理时状态栏显示「第 x/y 条 [会话名] 当前动作」 | — |
| `_on_progress` | `(task_id, ratio)` | 总进度 =（已完成数 + 当前比例）/ 总数 | 进度条 |
| `_on_task_done` | `(task_id)` | 单条完成：刷新列表；`UPLOAD_DELETE_AFTER` 开启时 `_delete_after_upload`（与主窗口自动上传同规则），否则 `RecordingRepo.mark_uploaded` + 主窗口日志「☁ 上传完成（本地保留）」 | — |
| `_delete_after_upload` | `(session_path)` | 后台线程 `shutil.rmtree`，结果经 `_session_deleted` 回主线程 | — |
| `_on_session_deleted` | `(session_path, err)` | 删除收尾：失败仅提示；成功 `RecordingRepo.mark_uploaded_deleted` + 刷新列表 + 主窗口日志「☁ 上传完成，本地文件已删除」 | — |
| `_on_task_failed` / `_on_all_done` | `(task_id, error)` / `()` | 失败：状态栏「❌ 上传失败」+ 主窗口日志 + 刷新列表 / 全部完成：恢复按钮、进度条 100、状态「全部完成」 | — |
| `closeEvent` | `(event)` | 停止 `_manager` | — |

**关键数据**：

- 上传状态来源 `UploadManager.get_upload_status(path)`：`completed`/`failed`/`pending`。
- `_task_path_map`：task_id → session_path（删除与状态显示用）。
- 「上传后自动删除」开关 `settings.UPLOAD_DELETE_AFTER` 对自动与手动上传统一生效。

**调用关系**：由 `ui/main_window.py`（`_open_upload`）构造。依赖 `core.uploader.UploadManager`、`core.api_client.APIClient`（测试连接）、`core.recording_repository.RecordingRepo`、`core.session_catalog.list_recordings`、`config.settings`、`config.i18n.tr`。

### ui/__init__.py

**作用**：ui 包的空标记文件（0 行），仅使目录成为可导入包。无类、无函数、无数据。
