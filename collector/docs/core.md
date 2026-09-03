# core/ — SDK 核心

## 定位

`core/` 是 collector 的 SDK 核心：相机采集、设备管理器、录制落盘、SQLite
持久化、上传/后端通信、回放会话、手部关键点、BLE 手套传感与通用工具全部
平铺于此（v1.0.3 架构重组后由原 `core/` + `storage/` + `network/` + `utils/`
+ `tools/sensors/` 合并而成）。

**分层规则**（SDK 化后确立）：

- core 只依赖 PyQt5.**QtCore**（QObject/pyqtSignal/QTimer）与 `config/`，
  **禁止 import ui**；唯一例外 `core/sensor_config_dialogs.py`（传感器
  配置对话框继承 `QDialog`，需 `PyQt5.QtWidgets`）；
- 算法口径全部在 core（纯函数优先），ui 槽只做 widget 操作；
- 跨线程一律语义信号（帧/日志/关闭完成），主线程创建 QObject；
- 可能超过 32 位的信号参数（纳秒时间戳）用 `object` 封装
  （PyQt5 int 参数按 qint32 封送，>2^31 静默翻负）。

## 文件清单

| 文件 | 一句话作用 |
|---|---|
| `core/camera.py` | UVC 相机采集：V4L2 sysfs 枚举 + `CameraWorker` 后台线程采集（重连/曝光/FOURCC） |
| `core/d435_camera.py` | RealSense D400 系列采集：`D435Worker` 进程内 rs2.pipeline 双路输出 + 运行时标定提取 |
| `core/device_detector.py` | 统一设备枚举（UVC + D435 + S80M + BLE），`DeviceScanner` 后台扫描 |
| `core/pipeline.py` | `CameraPipeline`：多路相机 + 共享 EgoData 录制会话，独立写入线程 |
| `core/s80m_manager.py` | S80M 双目子进程生命周期：Popen/stdin 曝光通道/watchdog/帧管道读取 + 50→30 抽帧口径 |
| `core/d435_manager.py` | D435 worker 生命周期 + 帧处理口径（calib 首帧注入/热力图 EMA/录制写入） |
| `core/device_manager.py` | 统一设备 worker 注册表 + 面板开关分派口径 + 录制设备元数据 |
| `core/device_naming.py` | 槽名清洗/分配（`allocate_slot_names`）+ 曝光基线归一化（纯函数，零 Qt） |
| `core/exposure_controller.py` | 每设备曝光：对话框参数解析 + 下发/持久化口径（零 Qt） |
| `core/egodata_writer.py` | EgoData / LeRobot v3 格式录制器：MP4/深度/Parquet/元数据多模态落盘 |
| `core/calibration.py` | 标定数据模型：双目内外参 + 深度缩放的 dataclass 与 JSON 序列化 |
| `core/database.py` | 线程安全 SQLite 单例（`recording` 与 `upload_task` 两表 + 建表 SQL） |
| `core/recording_record.py` | `RecordingRecord` dataclass：录制历史一行记录的字段定义（原 storage/models） |
| `core/recording_repository.py` | `RecordingRepo`：录制记录 CRUD 仓库（原 storage/repository） |
| `core/task_record.py` | 任务进度持久化（`data/tasks.json` 唯一权威源，缺文件自动播种） |
| `core/api_client.py` | 后端 HTTP REST 客户端（健康检查、上传、查询、删除） |
| `core/task_service.py` | 后台任务轮询服务（QTimer + 后台线程，Cookie JWT 登录） |
| `core/uploader.py` | 上传队列管理器（视频预压缩 → 打包 zip → 上传 → SQLite 状态持久化） |
| `core/session_timeline.py` | 会话传感器时间线：向量化合并 per-sensor parquet + 帧号二分查询 |
| `core/session_catalog.py` | 会话扫描/元数据/帧率解析/录制列表（纯函数，零 Qt） |
| `core/session_loader.py` | 回放后台加载器（parquet 合并 + 关键点，gen 防过期协议） |
| `core/stereo_depth.py` | 手写 StereoSGBM + WLS 视差计算，深度热力图与 EMA 平滑 |
| `core/hand_tracking.py` | 手部关键点后处理纯函数 API（glove/bare 双模式、自动标注、parquet 读写） |
| `core/hand_processor.py` | `SessionHandProcessor`：`hand_tracking.process_session` 的 Qt 薄封装 |
| `core/auto_labeler.py` | `AutoLabeler`：`hand_tracking.label_session` 的 Qt 薄封装 |
| `core/ble_engine.py` | BLE 传感器引擎：扫描/连接/通知订阅/帧解析/降噪，QObject + Qt 信号 |
| `core/render_engine.py` | 五种触觉渲染模式 + 仿生手掌配置路径常量（config/sensors/） |
| `core/sensor_config_dialogs.py` | PyQt5 传感器配置对话框（矩阵行列 / 仿生手掌逐部位） |
| `core/sensor_hand_config.py` | 仿生手掌配置加载/传感器列有效性过滤（纯函数，零 Qt） |
| `core/helpers.py` | 通用工具函数（约 47 个：ID/时间/格式化/路径/会话扫描） |

（`core/__init__.py` 为空包标记。）

## 采集与设备

### core/camera.py

**作用**：UVC 摄像机采集模块。模块级函数负责只读 sysfs 枚举与打开
（`list_v4l_devices` 不 open 设备、轮询安全；`_try_open_camera` 支持
`/dev/v4l/by-id` 永久路径与重连兜底；跳过 FTDI SDK 设备与 RealSense UVC
节点）。`CameraWorker` 在后台线程采集帧、经 Qt 信号发主线程，含指数退避
重连、UVC 固定自动曝光（首帧读出成功后强制应用）、MJPG FOURCC 设置与
读回校验、录制帧队列直推。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `CameraState` | 类常量 | 状态枚举 | `DISCONNECTED/IDLE/RECORDING/ERROR` |
| `list_v4l_devices` | `(max_index=16)` | sysfs 只读枚举 V4L2 设备，by-id 分组、每物理设备只留主视频流 | 每项 `{video_index, name, serial, by_id_path, vid, pid, is_sdk, is_realsense}` |
| `detect_cameras` | `(max_index=8)` | 打开测试（test_read）枚举可用相机 | `[(idx, backend), ...]` |
| `_try_open_camera`（内部） | `(index, test_read=False, fallback_all_by_id=False)` | 按后端列表（Linux: V4L2/FFMPEG/ANY；Windows: DShow/MSMF/ANY）尝试打开 | `(VideoCapture, backend名)` 或 `(None, "")` |
| `_is_sdk_device` / `_is_realsense_node`（内部） | `(index)` | sysfs name/VID:PID 判 FTDI SDK 设备与 RealSense 节点 | `bool` |
| `_usb_vid_pid`（内部） | `(video_index)` | 沿 sysfs 设备树向上（≤6 层）找 `idVendor`/`idProduct` | `(vid, pid)` 小写十六进制或 `None` |
| `CameraWorker` | `(camera_index=0, resolution=None, record_queue=None)` | UVC 采集 worker；默认分辨率 `settings.DEFAULT_RESOLUTION`（1280×960@30，MJPG） | 信号：`frame_ready(ndarray)`、`state_changed(str)`、`error_occurred(str)`、`fps_updated(float)`、`camera_opened(int,int,str)` |
| `CameraWorker.start` / `stop` / `pause` / `resume` | 无参 | 生命周期控制（stop join ≤5s） | 无 |
| `CameraWorker.state` / `is_connected` / `resolution` / `latest_capture_ts_us` | property | 状态、连接、实际分辨率、最新帧采集时间戳（Unix 微秒，线程安全） | 对应值 |
| `CameraWorker._apply_exposure_to`（内部，static） | `(cap, auto, value, backend="")` | V4L2 按候选菜单顺序（自动 0→3→2；手动 1→0）写自动曝光，以 set 返回 True 且 get 读回一致为准 | `(auto成功, 值成功)` |
| `_FPSCounter` | `(window=30)` | 滑动窗口帧率计数 | `tick()` 更新 `.fps` |

**关键数据**：重连参数 `_RECONNECT_BASE_DELAY=2.0s`、`_RECONNECT_MAX_DELAY=30.0s`、`_READ_FAIL_THRESHOLD=120`、`_INITIAL_FAIL_THRESHOLD=30`、`_INITIAL_GRACE_PERIOD=90`（迭代数）；曝光为"开启/重连后首帧读出成功后强制回自动"策略（注释：UVC 手动模式实测帧率上限 ~26fps）；录制帧队列 `queue.Queue(maxsize=3)`，满则丢旧帧保持最新。

**调用关系**：被 `core/pipeline.py`（`CameraWorker`/`CameraState`）、`core/device_detector.py`（`list_v4l_devices`/`_is_sdk_device`）、`ui/main_window.py`（`_add_camera_slot`）、`tools/tests/exposure_control_test.py`、`tools/tests/test_device_detector.py`、`tools/tests/mono_regression.py` 引用。

### core/d435_camera.py

**作用**：Intel RealSense D400 系列（D435/D435I/D405 等）深度双目采集。D400
是稳定 UVC 用户态设备（pyrealsense2 自带 librealsense2 + libusb），无需像
S80M 那样隔离到子进程；`D435Worker` 后台线程跑 `rs2.pipeline`，对外输出
RGB（BGR 三通道）+ 深度（uint16，归一化毫米）两路信号。左红外流仅内部
启用（时间戳元数据），不对外输出；右红外不开流（基线改从 depth sensor 的
`stereo_baseline` 选项直读，双 RealSense 同 hub 时多一条 IR2 会把 D405 挤到
~19fps）。带停滞/低帧率双看门狗，重连每次新建管道。`frames_ready` 信号的
`hardware_ns` 用 `object` 参数（PyQt5 队列信号把 int 按 qint32 封送，>2^31
纳秒会静默翻负）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `d435_available` | `(force=False)` | pyrealsense2 可用且存在 D400；结果模块级缓存，`force=True` 绕过（热插拔后活体复查） | `bool` |
| `list_d400_devices` | 无参 | 枚举 D400 设备（product_line 含 "D400"） | `[(name, serial), ...]`，无则 `[]` |
| `_apply_color_exposure`（模块级） | `(sensor, auto, value)` | 对 color 传感器应用曝光（µs） | `bool`，异常返回 False |
| `D435Worker` | `(width, height, fps, parent, ts_log, rgb_width, rgb_height, serial, model_name, rgb_slot, depth_slot, exposure)` | 采集 worker；分辨率默认 `settings.D435_RESOLUTION`/`D435_RGB_RESOLUTION`，`serial` 锁定设备（None→第一台 D400），槽名默认 `settings.D435_SLOT_RGB`/`D435_SLOT_DEPTH` | 信号：`frames_ready(str, np.ndarray, object, list)`（slot_id, frame, hw_ns, imu_samples；D435/D405 无 IMU，恒 `[]`）、`error_occurred(str)`、`status_changed(str)` |
| `D435Worker.start` / `stop` | 无参 | 启动/停止后台线程（stop 阻塞 ≤3s） | 无 |
| `D435Worker.get_calibration` | 无参 | 首帧后提取的标定 | `StereoCalibration` 或 `None` |
| `D435Worker.set_exposure` | `(auto, value)` | 主线程只写 pending；采集线程在流启动后应用，重连重放最近设置 | 无 |
| `D435Worker.exposure_info` | 无参 | 返回量程/auto/value | `((min,max)µs 或 None, auto, value)` |
| `D435Worker.original_exposure` | 无参 | 开流时读回的"最一开始"曝光基线 | `(auto, value)` 或 `None` |
| `_build_calibration`（模块级） | `(profile, fps, resolution)` | 从 rs2 profile 提取左右红外内参/深度内参/基线 | `StereoCalibration`（right IR2 未开流时内参镜像左目） |

**关键数据**：输出深度统一归一化到毫米（`_depth_unit_factor = 设备深度单位 / settings.DEPTH_SCALE`，D405 原生 0.1mm/单位）；帧时间戳用左红外帧 `frame_timestamp` 元数据（32 位 µs 计数，软件解绕 + 首帧归零 → 会话内单调 `hardware_ns`，与 S80M 语义一致）；流组合 color(rgb8) + depth(z16) + infrared#1(y8，仅内部)；标定 `cam_imu_timeshift=0.0`（D435 无 IMU）。`__main__` 为无头自检（运行 5s、抽帧存 `/tmp/d435_selftest`、打印标定）。

**调用关系**：被 `ui/main_window.py`（`_open_d435`，worker 类经 `D435DeviceManager.spawn` 传入）、`tools/tests/d405_worker_test.py`、`tools/tests/d435_e2e_test.py`、`tools/tests/d435_gui_smoke_test.py`、`tools/tests/exposure_control_test.py` 引用；函数内延迟导入 `core/calibration.py`。

### core/device_detector.py

**作用**：统一枚举已连接设备（UVC + RealSense D400 + S80M + 蓝牙）。列表轮询
走 sysfs 只读扫描（不 open、不 test_read），轮询 <5ms、录制中安全；open 测试
只发生在开关打开后（各 worker 自带重连）。蓝牙两个来源：`bluetoothctl`
已配对列表（快、只读）+ bleak 主动发现（慢 ~5s，节流缓存 20s；手套连接中
可 `set_ble_scan_suppressed(True)` 抑制）。`DeviceScanner` 在后台线程扫描，
经排队信号回主线程。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `DeviceInfo` | dataclass | 单条设备描述 | 字段：`key/kind/display_name/serial/video_index/by_id_path/backend/address/rssi/user_name`；property `stable_key`、`group`（`camera|glove|other_ble`）、`label`（用户命名优先） |
| `detect_devices` | `(max_index=settings.DEVICE_SCAN_MAX_INDEX)` | 四段枚举（UVC+D435+S80M+BLE），各自容错整体不崩 | `List[DeviceInfo]` |
| `DeviceScanner` | `(parent=None, max_index=None)` | 后台线程扫描，`_busy` 守卫防轮询堆积 | 信号 `scan_finished(list)`；`request_scan()`/`stop()` |
| `set_ble_scan_suppressed` | `(on: bool)` | 抑制 bleak 主动发现（手套连接中防扫描挤占吞吐），不影响 bluetoothctl 列表 | 无 |
| `_list_uvc_devices` / `_list_d435_devices` / `_list_s80m_devices` / `_list_ble_devices`（内部） | 各自 max_index | 分段枚举 | `List[DeviceInfo]` |
| `_is_glove_name`（内部） | `(name)` | 广播名判手套：含 "matrix" 或单字母 `l/r/left/right/l_glove/r_glove/left_glove/right_glove` | `bool` |

**关键数据**：`DeviceInfo.key` 格式约定：`"uvc:{by-id前缀或索引}"`、`"d435:{serial}"`、`"s80m:ftdi"`、`"ble:{MAC}"`；`kind` 取值 `uvc | d435 | s80m | data_ble | ble`（`data_ble`=手套）。BLE 常量 `BLE_DISCOVERY_INTERVAL_S=20.0`、`BLE_DISCOVERY_TIMEOUT_S=5.0`。S80M：FTDI 命中一次即返回单条（SDK 配置写死 video0/video2）。RealSense UVC 节点判定：vendor 8086 且 PID 0b07（D435）或驱动 name 含 "RealSense"（D435i 为 0b3a、其余型号 PID 各异，name 兜底）。

**调用关系**：被 `ui/main_window.py`（`DeviceScanner`）、`ui/device_panel.py`、`tools/tests/device_panel_gui_smoke_test.py`（`DeviceInfo`/`detect_devices`）、`tools/tests/glove_widget_test.py`、`tools/tests/multi_device_registry_test.py` 引用；调用了 `core/camera.py`。

### core/pipeline.py

**作用**：摄像机管线——管理多路相机槽位与一个共享 EgoData 录制会话。
`CameraSlot` 封装单路 `CameraWorker` + 帧缓冲队列（maxsize=3，采集线程直写，
绕过 Qt 信号/主线程）。`CameraPipeline` 录制时创建 `EgoDataWriter`，独立写入
线程以 `settings.RECORDING_FPS`（30fps）精确节拍把各槽位帧、传感器数据、
外部帧源（双目子进程等）、深度帧统一写入同一会话（MP4 + Parquet）。深度
"伪相机"经 `set_depth_camera` 显式注册（D435 原生深度与 S80C 子进程深度
引擎多路并存，S80C 用槽 `settings.S80M_DEPTH_SLOT`；无注册槽时随
`stereo_left` 兜底落盘）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `CameraSlot` | `(slot_id, camera_index)` | 单相机槽位 | 信号 `frame_ready(str, object)`、`state_changed(str, str)`、`error_occurred(str, str)`；`start_camera()`/`stop_camera()`/`camera_state` |
| `CameraPipeline` | `(output_dir=None)`（默认 `settings.RECORDING_DIR`） | 多路管线 | 信号：`slot_added`、`slot_removed`、`session_changed(object)`、`recording_started(str)`、`recording_finished(str, str)`、`recording_aborted(str)`、`duration_changed(str, float)`、`error_occurred(str, str)`、`state_changed(str, str)` |
| `CameraPipeline.add_camera` / `remove_camera` / `get_slot` / `slot_ids` / `slot_count` / `remove_all` | `(slot_id, camera_index)` | 槽位管理（录制中移除会先 `abort_recording`） | 对应值/无 |
| `CameraPipeline.register_external_source` / `unregister_external_source` | `(slot_id, resolution, fps=25.0)` | 外部帧源注册（队列 maxsize=2，不建 CameraWorker） | 无 |
| `CameraPipeline.write_external_frame` | `(slot_id, frame, hardware_ns=0, imu_samples=None)` | 外部帧入录制队列（非阻塞，满则丢） | 无 |
| `CameraPipeline.write_depth` | `(depth_frame, depth_slot="stereo_left")` | 深度帧入录制队列（uint16 毫米：S80C SDK 深度引擎 / D435 原生，同口径）；单深度相机时自动回落 | 无 |
| `CameraPipeline.set_depth_camera` / `clear_depth_camera` | `(name, resolution, fps, master_slot, heatmap_near_mm, heatmap_far_mm, heatmap_smooth_k, heatmap_temporal_alpha, raw_depth)` | 声明/清除深度伪相机（多路并存，只进 metadata 不建 MP4；`raw_depth=True` 时原始 uint16 bin 随热力图 MP4 落盘） | 无 |
| `CameraPipeline.set_external_calibration` / `set_device_calibration` / `clear_device_calibration` | `(calib)` / `(device_key, calib)` / `(device_key)` | 注册外部帧源标定（单值路径存 `"_default"` 键 → `calibration/head_stereo.json`） | 无 |
| `CameraPipeline.register_sensors` / `register_sensor` / `unregister_sensor` | `(sensor_names)` / `(sensor_name)` / `(sensor_name)` | 传感器名称注册（决定 parquet `observation.<name>` 列），须在 `start_recording` 前调用 | 无 |
| `CameraPipeline.start_recording` | `(slot_id, task_name="", batch_index=0, device_meta=None)` | 全部相机开始录制同一会话（后台线程初始化） | 信号 `recording_started` |
| `CameraPipeline.finish_recording` / `abort_recording` | `(slot_id)` | 正常停止（`end_episode` 最终化）/ 异常停止（`abort_episode` 删目录） | 信号 `recording_finished`/`recording_aborted` |
| `CameraPipeline.write_sensor` | `(data, capture_ts_us=0, sensor_name="")` | 传感器数据入队列（16×16 float32 压力矩阵 ravel） | 无 |
| `CameraPipeline.record_event` | `(device, event_type, message="")` | 记录设备连接/断开事件 → 写入线程每帧写 `status.<device>` 列 | 无 |
| `CameraPipeline.is_recording` / `elapsed` / `last_recording_frames` | property | 录制状态/已录时长/上一轮每相机帧数快照 | 对应值 |

**关键数据**：写入线程节拍 `frame_interval = 1.0 / settings.RECORDING_FPS`（30fps），sleep 到目标前 ~1ms 后 busy-wait 补齐；每帧先排空传感器队列取最新数据，CameraSlot 队列"排空取最新帧"、外部帧源队列"取一帧不排空"（输出均匀无抖动）；单目路径写 MP4 时上下翻转、外部双目帧路径已在 `ui/main_window._on_stereo_frame` 翻转故 `flip_vertical=False`；IMU 样本只随 `stereo_left` 写入（左右目共享一份，避免 `data/imu/` 重复行）；可选 `settings.CAMERA_MIRROR_HORIZONTAL` 水平镜像。**深度 MP4 补拍**（v1.0.11）：`_write_one_frame` 每个 master 槽节拍至多消费一帧深度——队列有新帧则写热力图 MP4 +（raw_depth 时）PNG16 并缓存于 `_last_depth_frames`；队列空（S80C 深度引擎 ~20fps 低于录制 30fps）则重写缓存帧到热力图 MP4（不落 PNG、不推进序号），保证深度 MP4 时长与 RGB 对齐。

**调用关系**：被 `ui/main_window.py`（主程序）、`tools/tests/d435_e2e_test.py`、`tools/tests/device_panel_gui_smoke_test.py`、`tools/tests/exposure_control_test.py` 引用；调用了 `core/camera.py` 与 `core/egodata_writer.py`。

## 设备管理器与口径（v1.0.3 从 ui 抽出）

### core/s80m_manager.py

**作用**：S80M 双目子进程生命周期管理（每台一个条目）。自包含双目模块
`tools/stereo_s80m/` 内含 `read_stereo_rgb.py` +
`lib/fays_atrak/x86_64/Release/libfays_vikit.so`（SDK 动态库，注释标 3.9.0）+
`config/fays_vikit.yaml`，与外部 SDK 目录完全独立——外部目录 git 更新、跑官方
demo 均不影响 DAQ 程序。Popen 子进程 stdout 管道传帧（`--pipe -`，避免 FIFO
权限问题），stdin 管道是曝光控制通道（行协议 `"SET_EXPOSURE <float>"`，
-1.0=自动），watchdog 定时检查进程存活并回读 stderr 末尾，reader 线程按
`[4B len][8B ts_ns][jpg]` 帧格式解析左右目 JPEG 与 IMU 样本块（56B/样本）。

**管道协议扩展（v1.0.11）**：`s80m_depth_available()`（`settings.S80M_DEPTH_ENABLED` 且 `tools/hand_3d_s80c/third_party` 深度库/模型/OpenCV4.2 齐全）时 spawn 给子进程追加 `--depth-sdk-dir`，每帧 IMU 块之后、flush 之前必写深度块（引擎失败期恒写 0 长度，保证解析确定性）：
```
[>I depth_len]                              ← 0 = 本帧无新深度
if depth_len>0: [>Q depth_ts][>I w][>I h][w*h*2 字节 uint16 毫米 LE]
```
reader 校验 `depth_len == w*h*2` 且尺寸合法后 `depth_ready.emit(slot, uint16, ts)`；子进程内移植自 `tools/hand_3d_s80c/s80c_depth_worker.py` 的 SDK 深度引擎（~20fps @ 相机 50fps，~220MB 缓冲）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `SDK_ROOT` / `STEREO_DEMO` / `STEREO_AVAILABLE` / `STEREO_CFG_50` | 模块常量 | 锚定 `tools/stereo_s80m/` 路径（不随 cwd 漂移） | — |
| `load_default_exposure` | 无参（模块级） | 「恢复默认」基线：yaml `stereo_init_exposure`（<0=自动） | `(auto, value)`，兜底 `(False, 400.0)` |
| `frame_record_decision` | `(entry, slot_id, hardware_ns, imu_samples)`（模块级） | 50→30 抽帧纯口径：每 1/30s 桶保留首帧；被抽帧的 IMU 块累积到下一帧；hw_ns==0 兜底全录 | `(record, imu_batch)` |
| `S80MDeviceManager` | `(pipeline, parent=None)` | 子进程生命周期管理 | 信号：`frame_ready(str, np.ndarray, object, list)`、`depth_ready(str, np.ndarray, object)`、`device_closed(str)`、`log(str)` |
| `S80MDeviceManager.new_entry` | `(label)` staticmethod | 注册表条目骨架（kind/slots/label + 进程字段占位；深度可用时 slots 含 `settings.S80M_DEPTH_SLOT`） | `dict` |
| `S80MDeviceManager.spawn` | `(dev_key, entry)` | 临时 50fps yaml 副本 + Popen + watchdog + reader 线程；深度可用时追加 `--depth-sdk-dir` 并注入 `LD_LIBRARY_PATH` | `bool` |
| `S80MDeviceManager.read_pipe` | `(dev_key)` | reader 线程：解析双目帧 → `frame_ready` ×2 + 深度块 → `depth_ready` | 无 |
| `S80MDeviceManager.check_alive` | `(dev_key)` | watchdog 槽：进程退出时停表 + 回读 stderr 末尾 500 字符 | 无 |
| `S80MDeviceManager.close` | `(dev_key)` | 停 watchdog → 关 stdin → terminate/kill → 关 stdout → join reader → 清理临时文件 → `device_closed` | 无 |
| `S80MDeviceManager.send_exposure` | `(entry, auto, value)` | stdin 行协议下发曝光（-1.0=自动） | 无 |
| `S80MDeviceManager.shutting_down` | 属性 | 窗口关闭标记（read_pipe 主循环退出条件） | — |

**关键数据**：`settings.STEREO_RECORD_MIN_INTERVAL_S`（1/30s）为抽帧桶宽；
帧时间戳为 SDK 硬件纳秒时钟（帧/IMU 同源，SLAM 对齐用）；镜头方向已由 SDK
config（`left_cam_rotate_180`/`right_cam_rotate_180`/`stereo_swap_lr`）处理，
reader 不再翻转；深度由子进程 SDK 深度引擎计算（进程常驻 ~1 核 + ~220MB，
`S80M_DEPTH_ENABLED=False` 一键回退纯 RGB）；`depth_ready` 的 hw_ns 用
`object` 类型信号参数（PyQt5 qint32 封送会截断 >2^31 纳秒）。

**调用关系**：被 `ui/main_window.py`（`_open_s80m`/`_on_stereo_frame`/
`_on_s80m_depth`/`_close_s80m`/`_s80m_set_exposure`）实例化；`frame_record_decision` 经
MainWindow 集成路径被 `tools/tests/s80m_50fps_decimation_test.py` 覆盖
（测试构造假 50fps 帧流走 `_on_stereo_frame` 接线，不直接 import 该函数）。

### core/d435_manager.py

**作用**：D435 深度双目设备管理器（每台一个条目）：worker 生命周期
（创建/信号接线/停止）与帧处理口径（calib 首帧注入、深度热力图/EMA、
录制写入）。`spawn` 接收 worker 类参数（离线测试在 `ui.main_window` 模块
patch `D435Worker` 替身仍生效）；信号连接在 `worker.start()` 之前完成。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `D435DeviceManager` | `(pipeline, parent=None)` | worker 生命周期 + 帧口径 | 信号：`frames_ready(str, np.ndarray, object, object, str)`（slot, frame, hw_ns, imu, dev_key）、`log(str)` |
| `D435DeviceManager.new_entry` | `(label, serial, rgb_slot, depth_slot, near_mm, far_mm, smooth_k, temporal_alpha)` staticmethod | 注册表条目骨架（worker 占位 None） | `dict` |
| `D435DeviceManager.spawn` | `(dev_key, entry, model_name, profile, exposure, worker_cls)` | 创建 worker（分辨率/fps 取 profile）、连接 `frames_ready`→`_relay_frames`、start | 无 |
| `D435DeviceManager.process_frame` | `(entry, slot_id, frame, hardware_ns=0, dev_key=None)` | 帧处理口径 → `(display_frame, is_depth)`；深度槽：热力图 + EMA + `pipeline.write_depth`；RGB 槽：原帧 + `write_external_frame` | `tuple` |
| `D435DeviceManager.close` | `(entry)` | 断开信号 → stop → deleteLater | 无 |

**关键数据**：calib 首帧注入只发生一次（`entry["calib_sent"]`），标定写
`calibration/head_stereo.json`；热力图 EMA 仅可视化（录制原始 PNG16 不受
影响）；`hw_ns` 用 object 封装（同 S80M 口径）。

**调用关系**：被 `ui/main_window.py`（`_open_d435`/`_on_d435_frames`/
`_close_d435`）实例化；调用了 `core/stereo_depth.py`。

### core/device_manager.py

**作用**：统一设备 worker 注册表 + 面板开关分派 + 录制元数据口径（纯
Python，零 Qt）。具体开启/关闭动作（建槽、弹窗、widget 接线）留在 UI 侧
回调，core 只做注册表与分派口径——录制锁双保险、kind 路由、失败语义统一
于此。主窗口 `_workers` 直接引用 `DeviceManager.entries`（同一 dict，离线
测试注入假条目沿用同一形状）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `DeviceManager` | 无参 | 注册表容器 | 属性 `entries`（dev_key → entry dict） |
| `DeviceManager.uvc_entry` / `ble_entry` / `glove_entry` | `(slots, label)` / `(slot, label)` / `(slot, role, glove_widget, label)` staticmethod | uvc/占位蓝牙/手套条目骨架 | `dict` |
| `DeviceManager.get` / `has_kind` / `has_serial` | `(dev_key)` / `(kind)` / `(kind, serial)` | 注册表查询（serial 查重用于 d435 同机重复开关检测） | 对应值 |
| `DeviceManager.build_device_meta` / `reset_s80m_record_state` / `teardown_all` | `()` / `()` / `(close_fns)` | 录制设备元数据 / 抽帧状态重置 / 全部关闭 | `list` / 无 / 无 |
| `build_device_meta` | `(entries)`（模块级） | 注册表 → meta devices 段（手套附 `sensor_column`，无数据蓝牙不进） | `list` |
| `reset_s80m_record_state` | `(entries)`（模块级） | 录制起止重置 50→30 抽帧状态（桶号 + IMU 缓冲） | 无 |
| `teardown_all` | `(entries, close_fns)`（模块级） | 遍历注册表分类清理（未知 kind 兜底弹出） | 无 |
| `dispatch_toggle` | `(dev, on, is_recording, open_fns, close_fns)`（模块级） | 面板开关分派：录制锁双保险 + kind 路由 → open/close 回调 | `opened: bool` |

**调用关系**：被 `ui/main_window.py`（`_on_device_toggled`/
`_teardown_all_workers`/`_build_device_meta`/`_reset_s80m_record_state`/
`_open_uvc`/`_open_glove`/`_open_ble_placeholder` + 冲突检测）使用。

### core/device_naming.py

**作用**：槽名清洗/分配与曝光基线归一化（纯函数，零 Qt 依赖）。从
`ui/main_window.py` 抽出的命名口径，供主窗口与离线测试共用。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `realsense_short` | `(model_name)` | 型号名 → 短名（如 "Intel RealSense D405" → "D405"） | `str` |
| `slot_base` | `(name, fallback)` | 槽名清洗：非 `[0-9A-Za-z_一-鿿]` 字符 → `_` | `str` |
| `normalize_original` | `(orig)` | 曝光基线归一化 → `(auto, value)` 或 `None` | `tuple | None` |
| `allocate_slot_names` | `(user_name, model_name, occupied_rgb_slots)` | 槽名分配：用户命名优先 → 型号名回落（d435_rgb/d435_depth 经典名）；同前缀多台编号追加（`_2`…） | `(rgb_slot, depth_slot)` |

**调用关系**：被 `ui/main_window.py`（`_open_d435`）与 `core/exposure_controller.py` 使用。

### core/exposure_controller.py

**作用**：每设备曝光控制——对话框参数解析与下发/持久化口径（零 Qt 依赖）。
UVC 无曝光入口（固定自动曝光，返回 None）。「恢复默认」基线（original）：
优先持久化的首见基线（`settings.device_original`），无则用 worker 本次开流
读回的硬件状态，并在对话框弹出时落盘锁定。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `exposure_dialog_params` | `(kind, entry, dev_key)` | 按设备类型解析对话框上下文（量程/自动/当前值/基线 + 首见基线落盘）；D435 → µs（流启动后读真实量程，兜底 1~66000）；S80M → 1.0~885.0 | `dict | None` |
| `apply_exposure` | `(dev_key, entry, auto, value, send_s80m=None)` | 按类型下发（d435 → worker.set_exposure；s80m → send_s80m 回调）+ `settings.save_device_exposure` 持久化 | 无 |

**调用关系**：被 `ui/main_window.py`（`_open_exposure_dialog`/`_apply_exposure`）使用；`tools/tests/exposure_control_test.py` 经 MainWindow `_apply_exposure` 接线与 `CameraWorker._apply_exposure_to` 间接覆盖（测试不直接 import 本模块）。

## 录制落盘与持久化（原 storage/）

### core/egodata_writer.py

**作用**：EgoData 格式录制器，多模态机器人遥操作数据采集的落盘核心
（`QObject`，带 Qt 信号）。一个 episode 对应一个会话目录：RGB 视频经 ffmpeg
管道写成 MP4，深度写热力图 MP4 和/或原始 uint16 PNG（png16），触觉传感器/IMU/手部
姿态写 LeRobot v3 兼容 Parquet，同时生成 `metadata.json`、`timestamps.json`、
`meta/info.json`、`meta/tasks.jsonl`、`meta/stats.json`、`meta/episodes/` 等
元数据。`meta/info.json` 是服务器上传契约（字段名、类型、结构不可动）；
`_write_metadata()` 写的 `metadata.json` 含 `devices` 段（设备分组权威信息）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `_get_ffmpeg` | 无参（模块级） | v1.0.9 起薄封装 `core/encoder_probe.list_working_ffmpegs()`（imageio 静态二进制 → PATH，须 `-version` 自检） | 可执行文件路径或 `None` |
| `_build_data_schema` | `(sensor_names, sensor_dim=256, device_ids=None)`（模块级） | 构建 data parquet 的 schema | `pa.Schema` |
| `EgoDataWriter.start_episode` | `(output_dir, cameras, fps=30, sensors, task_name, device_ids, calibration, camera_fps, depth_enabled, raw_depth, heatmap_near_mm, heatmap_far_mm, heatmap_smooth_k, heatmap_temporal_alpha, depth_heatmaps, depth_slots, raw_depth_slots, devices, calibrations, batch_index=0)` | 建 episode 目录结构、写标定与 metadata.json、启动各路 ffmpeg | `bool`；发 `episode_started(str)` |
| `EgoDataWriter.write_video_frame` | `(camera_name, frame, flip_vertical=True)` | 一帧 BGR 写入对应 ffmpeg 管道 | 无；管道断裂发 `error_occurred` |
| `EgoDataWriter.write_depth_frame` | `(frame_index, depth_frame, depth_slot="")` | uint16 深度 → 热力图 → 深度 MP4（首帧惰性建目录/启动 ffmpeg） | 无；`settings.DEPTH_ENABLED` 关闭且无显式深度槽时跳过 |
| `EgoDataWriter.write_raw_depth_frame` | `(frame_index, depth_frame, depth_slot="")` | 原始 uint16 深度写 `depth/<slot>/<1-based 序号>.png`（16-bit grayscale，毫米，压缩级 `D435_PNG_COMPRESSION`；D435/D405/S80C 统一 png16） | 写 PNG16 |
| `EgoDataWriter.write_frame_row` | `(frame_index, timestamp_s, sensors, connection_status, hardware_ns=0, imu_samples)` | 追加一行到 data parquet 缓冲 + 记录 timestamps 条目 | 内存缓冲；`hardware_ns` 仅双目帧非零，`imu_samples` 仅随 `stereo_left` 携带 |
| `EgoDataWriter.end_episode` | 无参 | 关 ffmpeg，写 timestamps/data/episode/info/tasks/stats/compat 全部文件 | 发 `episode_finished(episode_dir)` |
| `EgoDataWriter.abort_episode` | 无参 | 关 ffmpeg 并整目录 `shutil.rmtree` 丢弃 | 发 `episode_finished("")` |
| `_write_metadata` | `(cameras, fps, sensors, task_name)`（内部） | 写根级 `metadata.json`（含 `devices` 段） | 写文件 |
| `_write_info_json`（内部） | 无参 | 写 LeRobot v3 精确兼容 `meta/info.json`（服务器上传契约，结构不可动） | 写文件 |
| `_write_tasks_jsonl`（内部） | 无参 | 写 `meta/tasks.jsonl`（服务器上传依赖其识别任务） | 写文件 |
| `_write_episode_parquet` / `_write_compat_parquet`（内部） | 无参 | 写 `meta/episodes/chunk-000/file-000.parquet` 并复制到兼容旧路径 `chunk_000000.parquet` | 写文件 |
| `_write_timestamps`（内部） | 无参 | 写 `timestamps.json`；含 `hardware_ns` 的行按 hw 稳定排序保证时间线单调 | 写文件 |
| `_write_data_parquet`（内部） | 无参 | 按传感器分别写 `data/<sensor>/chunk-0000/chunk_000000.parquet` + `data/imu/...`（zstd） | 写文件 |
| `_write_stats_json`（内部） | 无参 | 写 `meta/stats.json`（均值/标准差/极值，imu 为 6 轴样本级统计） | 写文件 |
| `_plan_calibration_files` / `_device_slot_prefix`（内部） | `(legacy_calib)` | 规划标定文件布局：首台双目型写 `calibration/head_stereo.json`（服务器/回放依赖），其余写 `calibration/{slot前缀}_calibration.json` | `{相对路径: StereoCalibration}` |

**关键数据**：
- 目录布局（docstring 即约定）：`<output_dir>/<task_tag>/<tag>_000001/` 下 `videos/<cam>/chunk-0000/<cam>.mp4`、`depth/<slot>/<slot>.mp4`（热力图）或 `depth/<slot>/000001.png`（png16 uint16 毫米）、`calibration/head_stereo.json`、`metadata.json`、`timestamps.json`、`data/<sensor>/chunk-0000/chunk_000000.parquet`、`meta/episodes/chunk-000/file-000.parquet`。
- 编码器选择（v1.0.9）：`start_episode` 内（启动 ffmpeg 之前）按本会话 RGB 流数/最大分辨率调 `core/encoder_probe.select_encoder`（auto：nvenc 探针 → x265 速度门槛 → x264 兜底；显式指定失败按 auto 链回退），结果进程内缓存并写 `metadata.video_codec`；RGB 与深度热力图 ffmpeg 命令共用该选择（`choice.encoder + choice.args`）。
- Parquet schema（代码中的 `pa.schema` 真实定义）：`_EPISODE_SCHEMA` = `episode_index/task_index/start_frame_index/end_frame_index/length` 均 int64；`_IMU_SCHEMA` = `episode_index, frame_index, timestamp(float32), task_index, hardware_ns(int64), imu_ts_ns(list<int64>), observation.imu(list<list<float32,6>>)`，IMU 样本 6 轴 = `[gx, gy, gz, ax, ay, az]`；`_build_data_schema` 在传感器列 `observation.<name>`（list<float32,256>）之外固定加 `observation.<HAND_POSE_LEFT>`/`observation.<HAND_POSE_RIGHT>`（list<float32,63>，值全 0 占位）、`action`（list<float32,1>，恒 `[0.0]`）、每设备 `status.<device_id>`（string，默认 `"connected"`）。
- `metadata.json` 顶层键：`format="egodata"`、`format_version="1.0"`、`episode_index`、`fps`、`task_name`、`cameras`（每相机 `{height, width, type: rgb|depth, unit:"mm", format:"png16", fps?, device_key?, device?}`）、`devices`（每设备 `{key, kind, name, slots, serial?, sensor_column?, resolution, fps, calibration}`）、`sensors`、`sensor_dim`、`created_at`、`codebase_version`（`settings.APP_VERSION`）、`video_codec`（v1.0.9 编码器选择 `{encoder, codec, crf, ffmpeg, selected_by, probe}`）、`drop_stats`（v1.0.9 丢帧统计，`end_episode` 回写）。
- `meta/info.json` 顶层键：`codebase_version="v3.0"`、`fps`、`video`、`task_name`、`features`（`observation.<sn>: {dtype:"float32", shape:[16,16]}`、`observation.imu: {dtype:"float32", shape:[6]}`、`action: {shape:[1]}`）、`cameras`（dict，非 list——注释强调）、`devices`（紧凑段）、`device_names`（槽位→用户命名）、`sensors`、`sensor_dim`、`created_at`。
- `meta/tasks.jsonl` 行：`{"task_index": 0, "task": task_name or "default recording"}`。
- `meta/stats.json`：`observation.<sn>` 与 `observation.imu` 各含 `mean/std/min/max`（list<float32>），另有 `action` 恒值段。
- `timestamps.json`：`{"timestamps": [{frame_index, timestamp, wall_time, hardware_ns?}], "total_frames"}`。
- 深度槽位：显式 `depth_slots` 注册驱动落盘（不用槽名是否含 "depth" 猜测，D435/D405 槽名可被消歧编号如 `d435_depth_2`；S80C 用 `settings.S80M_DEPTH_SLOT`=`"stereo_depth"`）；未注册时回退 S80M 传统单槽 `settings.CAMERA_DEPTH`（`"head_depth"`）。

**调用关系**：被 `core/pipeline.py`（录制主链路）、`tools/tests/test_meta_devices.py`、`tools/tests/multi_device_registry_test.py` 引用；调用了 `core/stereo_depth.py`（`depth_to_heatmap`/`DepthHeatmapSmoother`）、`core/calibration.py`、`core/helpers.py` 的路径函数。

### core/calibration.py

**作用**：定义 EgoData 标准的标定数据模型。`CameraIntrinsics` 表达单目内参
（OpenCV 3×3 矩阵 + 畸变系数），`StereoCalibration` 表达双目 + 深度相机完整
标定（含基线、depth_scale、相机-IMU 时间偏移），并提供 OpenCV 结果互转与
JSON 读写。docstring 写明其 JSON 布局即为 `calibration/head_stereo.json` 的
落盘格式。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `CameraIntrinsics` | `intrinsic: [fx, fy, cx, cy]`；`distortion` 默认 `[]` | 单目内参 dataclass | 无 |
| `CameraIntrinsics.from_matrix` | `(K: 3×3, dist=None)` | 从 OpenCV 标定矩阵构建 | `CameraIntrinsics` |
| `CameraIntrinsics.to_matrix` | 无参 | 还原 3×3 相机矩阵 | `np.ndarray (3,3)` |
| `CameraIntrinsics.to_dist_array` | 无参 | 还原畸变系数 | `np.ndarray`（空时返回 `zeros(4)`） |
| `StereoCalibration` | 字段见下表 | 双目 + 深度标定 dataclass | 无 |
| `StereoCalibration.to_dict` | 无参 | 序列化为 dict（`depth_camera` 非空才输出该键） | `dict` |
| `StereoCalibration.from_dict` | `(d: dict)` | 从 dict 反序列化（缺键用默认值） | `StereoCalibration` |
| `StereoCalibration.save` | `(path)` | 写 JSON 文件 | 磁盘文件 |
| `StereoCalibration.load` | `(path)` | 读 JSON 文件 | `StereoCalibration` |
| `StereoCalibration.from_opencv` | `(K_left, dist_left, K_right, dist_right, baseline=0.095, resolution=(1280,800), fps=25.0, depth_scale=0.001)` | 从 `cv2.stereoCalibrate` 结果构建 | `StereoCalibration` |

**关键数据**：`StereoCalibration` 字段与 JSON 键一一对应：`type`（默认 `"stereo_rgbd_camera"`）、`name`（默认 `"head_stereo"`）、`resolution`（默认 `[1280, 800]`）、`fps`（默认 `25.0`）、`baseline`（默认 `0.095`，米）、`left_camera`/`right_camera`（各含 `intrinsic` 与 `distortion`）、`depth_scale`（默认 `0.001`，像素值 × depth_scale = 米）、`cam_imu_timeshift`（默认 `-0.0019` 秒，来源注释注明 SDK calib.yaml 的 timeshift_cam_imu）、可选 `depth_camera`（D435 独有；注释注明 rs2 畸变系数为 Inverse Brown Conrady 模型，原样存储）。

**调用关系**：被 `core/egodata_writer.py`（写入 `calibration/head_stereo.json`）、`core/d435_camera.py`（D435 运行时从 rs2 profile 提取标定）、`ui/main_window.py`（`_open_s80m` 静态标定注入）、`tools/tests/test_meta_devices.py` 引用。

### core/database.py

**作用**：SQLite 数据库管理器。`Database` 类用 `threading.local` 为每线程维护
独立连接（线程安全），建库时开启 WAL 模式与外键约束；`SCHEMA_SQL` 定义
`recording`（录制历史）与 `upload_task`（上传任务）两张表及索引。模块底部
创建模块级单例 `db = Database()`，全项目共享。路径取自
`config.settings.DB_PATH`（`config/settings.py`：`data/pipeline.db`）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `Database.__init__` | `(db_path=DB_PATH)` | 保存路径 | 无 |
| `Database.conn`（property） | 无参 | 取当前线程连接，自动创建目录/连接、设 row_factory=Row、WAL、外键 | `sqlite3.Connection` |
| `Database.close` | 无参 | 关闭当前线程连接 | 无 |
| `Database.init_schema` | 无参 | 执行 `SCHEMA_SQL`（IF NOT EXISTS） | 建表并 commit |
| `db` | 模块级单例 | 全局共享实例 | `Database` 实例 |

**关键数据**（SQLite 表，`config.settings.DB_PATH` = `data/pipeline.db`）：
- `recording`：`id TEXT PK`、`camera_index INTEGER NOT NULL`、`camera_name TEXT NOT NULL`、`file_path TEXT`、`file_size_mb REAL`、`duration_sec REAL`、`resolution_w INTEGER`、`resolution_h INTEGER`、`status TEXT`（`completed | uploaded | aborted | deleted | uploaded_deleted`）、`started_at TEXT NOT NULL`、`finished_at TEXT NOT NULL`；索引 `idx_recording_camera(camera_index)`、`idx_recording_date(started_at)`。
- `upload_task`：`id TEXT PK`、`session_path TEXT`、`session_name TEXT`、`status TEXT`（`pending | uploading | completed | failed | skipped`）、`progress REAL`、`retry_count INTEGER`、`server_url TEXT`、`server_session_id TEXT`、`error_message TEXT`、`created_at TEXT`、`updated_at TEXT`；索引 `idx_upload_status(status)`、`idx_upload_session(session_path)`。

**调用关系**：被 `core/recording_repository.py`、`core/uploader.py`、`ui/main_window.py` 引用。

### core/recording_record.py

**作用**：定义录制历史记录的 dataclass `RecordingRecord`，字段与 `recording`
表列一一对应，`id`/`started_at`/`finished_at` 默认用 `core.helpers` 的
`new_id`/`utcnow` 生成。（原名 storage/models.py，v1.0.3 改名避免与
`tools/models/` 混淆。）

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `RecordingRecord` | 字段见下 | 一条录制记录 | 无 |
| `RecordingRecord.from_row` | `(row)` classmethod | 从 `sqlite3.Row` 构造 | `RecordingRecord` |

**关键数据**：字段 `id: str`、`camera_index: int = 0`、`camera_name: str = ""`、`file_path: str = ""`、`file_size_mb: float = 0.0`、`duration_sec: float = 0.0`、`resolution_w: int = 0`、`resolution_h: int = 0`、`status: str = "completed"`（`completed | uploaded | aborted | deleted | uploaded_deleted`）、`started_at: str`、`finished_at: str`。

**调用关系**：被 `core/recording_repository.py`、`ui/main_window.py` 引用。

### core/recording_repository.py

**作用**：录制记录的数据仓库层，`RecordingRepo` 全部为静态方法，直接操作
`core.database` 的单例 `db`。其中 `mark_uploaded` 支持"上传成功、本地保留"
的语义（行保留、状态改为 `uploaded`），`mark_uploaded_deleted` 支持"上传
成功、本地已删"（状态 `uploaded_deleted`），两者都保证历史可查。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `RecordingRepo.save` | `(r: RecordingRecord)` | INSERT OR REPLACE 一条记录 | 写入并 commit |
| `RecordingRepo.list_all` | `(limit=100)` | 按 `started_at` 倒序取最近 N 条 | `List[RecordingRecord]` |
| `RecordingRepo.list_by_camera` | `(camera_index, limit=50)` | 按摄像机索引筛选 | `List[RecordingRecord]` |
| `RecordingRepo.delete` | `(record_id)` | 按 id 删除 | 删除并 commit |
| `RecordingRepo.mark_uploaded` | `(file_path)` | 按路径把状态改为 `uploaded`（已上传、本地保留） | 受影响行数 `int` |
| `RecordingRepo.mark_uploaded_deleted` | `(file_path)` | 按路径把状态改为 `uploaded_deleted` | 受影响行数 `int` |

**调用关系**：被 `ui/upload_dialog.py`、`ui/main_window.py` 引用；调用了 `core/database.py` 与 `core/recording_record.py`。

### core/task_record.py

**作用**：任务记录持久化。`data/tasks.json` 是唯一权威源（`_TASKS_FILE =
settings.DATA_DIR/tasks.json`）。核心设计：进度按**分账模型**持久化——
`local_count`（本机累计完成段数，`increment_task_completed()` +1）、
`synced_count`（后端已确认的本机段数，水位只前进不后退）、`backend_count`
（最近一次已知后端全局数），派生显示值 `completed_count = backend_count +
(local_count - synced_count)` 由 `_recount()` 统一重算（字段名保留，UI 零
改动）；上传成功后本地文件被删也不影响进度。旧数据缺新字段时一次性迁移
回填（`local/synced/backend` 均取旧 `completed_count`，目录扫描兜底，
历史不补报、显示连续）。`_read_raw()` 在文件缺失或 JSON 解析失败时自动
播种内置 `_SEED_TASKS`（3 个默认任务）。`merge_backend_tasks()` 把后端
推送的任务按 id 合并进本地记录（本地独有任务保留、(name, assigned_user)
同名去重、被删任务以 `hidden` 墓碑防止复活、账号身份轮询下的撤销清除；
后端全局数权威，本机水位沿用）。多用户语义：任务可带 `assigned_user`
（null/缺失 = 公共任务），`load_tasks(identity)` 与 `filter_by_identity()`
按身份过滤可见范围（None=全部、`"guest"`=仅公共、用户名=公共+指派给该
用户）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `load_tasks` | `(identity: str \| None = None)` | 加载任务，过滤 `hidden`，自动计算 `status`，按 `assigned_at` 新→旧排序；`identity` 过滤可见范围（None=全部/`"guest"`=仅公共/用户名=公共+指派） | `list[dict]`；旧数据缺同步字段时迁移回填并写回文件 |
| `filter_by_identity` | `(tasks, identity: str \| None = None)` | 纯函数：按身份过滤任务列表（不修改原列表，供 UI 对已有列表即时过滤） | 新列表 |
| `save_tasks` | `(tasks)` | 持久化任务元数据（含分账字段） | 写文件（哑写，不迁移不重算） |
| `merge_backend_tasks` | `(backend_tasks, view_scope: str \| None = None)` | 后端任务合并进本地记录（id 合并/本地保留/(name, assigned_user) 同名去重墓碑/账号身份撤销清除/后端全局数权威+本机水位沿用） | 合并后列表（内部调用 `save_tasks` + `load_tasks`） |
| `mark_hidden` | `(task_id)` | 标记任务隐藏（UI 不再显示） | 写文件 |
| `refresh_progress` | 无参 | 用持久化计数重载任务列表 | `list[dict]` |
| `increment_task_completed` | `(task_name, task_id: str \| None = None, assigned_user: str \| None = None)` | 一次录制完成：匹配优先级 id → (name, assigned_user) → 唯一同名回退；`local_count` +1 并重算显示值 | 更新后的任务 dict，无匹配返回 `None`；写文件 |
| `pending_sync_tasks` | 无参 | 所有 `local_count > synced_count` 的任务快照（含 hidden，不按身份过滤）；纯读路径不触发迁移 | `list[dict]`（id/name/local_count/synced_count） |
| `mark_synced` | `(task_id, synced_count, backend_count)` | flush 成功后回写：水位只前进不后退，`backend_count` 用响应值无条件替换 | 更新后的任务 dict 或 `None`；写文件 |
| `task_by_id` | `(task_id)` | 按 id 查单个任务 | `dict` 或 `None` |
| `_read_raw` | 无参（内部） | 读 `tasks.json`；缺文件/解析失败时播种 | `list[dict]` |
| `_write` | `(data)`（内部） | 原子写（`.tmp` + `os.replace`） | 写文件 |

**关键数据**（`data/tasks.json` 结构）：顶层 `{"tasks": [...], "updated_at": iso字符串}`。任务条目字段：`id`、`name`、`description`、`status`（`load_tasks` 自动计算：`pending`(0%) / `in_progress`(>0%) / `completed`(100%)）、`total_required`、`assigned_at`（ISO 时间）、`assigned_user`（指派用户；null/空/缺失归一化为 None = 公共任务）、`params`（dict）、`completed_count`（派生显示值 = `backend_count + (local_count - synced_count)`）、`local_count`/`synced_count`/`backend_count`（进度分账三字段）、`hidden`（bool 墓碑）。内置播种任务：`task_001`（`cylinder_grasping`）、`task_002`（`cube_placement`）、`task_003`（`valve_rotation`）。`_count_sessions()` 按 `settings.RECORDING_DIR/<task_name>/<会话>/meta/info.json` 存在与否统计会话数，仅用于旧数据迁移回填。

**调用关系**：被 `ui/main_window.py`（`load_tasks`、`increment_task_completed`）、`ui/task_page.py`（`load_tasks`、`filter_by_identity`、`merge_backend_tasks`、`refresh_progress`、`mark_hidden`）引用。

## 上传与后端通信（原 network/）

### core/api_client.py

**作用**：对接 Data Acquisition 服务器的 HTTP REST 客户端。基于 `requests`，
支持复用外部已认证的 `requests.Session`（如 `TaskService` 的登录会话）；
上传接口用包装文件对象在读取时回调进度，避免把数 GB 的 zip 一次性读入
内存。所有网络异常均被吞掉并转为布尔/空值/错误字典返回，不会向上抛出。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `APIClient` | `(base_url: str, session: requests.Session = None)` | REST 客户端；`session` 缺省时自建并设置 `User-Agent: DAQ-SDK/1.0` | — |
| `APIClient.close` | `()` | 仅关闭自建 session | 复用外部 session 时不关闭 |
| `APIClient.health_check` | `() -> bool` | GET `/health` 健康检查（UI「测试连接」用） | 200 为 `True`，异常/非 200 为 `False` |
| `APIClient.upload_session_zip` | `(zip_path, session_name, progress_cb: Optional[Callable[[int, int], None]] = None, name: str = "", project_id: str = "") -> dict` | POST `/api/v1/session/upload` 上传 zip（`progress_cb` 收 `(uploaded_bytes, total_bytes)`）；`name`/`project_id` 为服务器目标项目表单字段（同名项目前缀歧义时服务器返回 409，须用 `project_id` 消歧） | `{"ok": True, "session_id": …, "response": …}` 或 `{"ok": False, "error": …, "ambiguous_project": True}`（409 项目歧义时置 `ambiguous_project`；错误文本截断 300 字符） |
| `APIClient.get_projects` | `(limit: int = 100) -> list[dict]` | GET `/api/v1/projects` 项目列表（UI 下拉框用） | 异常返回 `[]` |
| `APIClient.get_sessions` | `(limit: int = 50) -> list[dict]` | GET `/api/v1/sessions` 查询列表 | `data["sessions"]`；异常返回 `[]` |
| `APIClient.get_session` | `(session_id: str) -> Optional[dict]` | GET `/api/v1/session/{id}` 详情 | 异常返回 `None` |
| `APIClient.delete_session` | `(session_id: str) -> bool` | DELETE `/api/v1/session/{id}` | 200 为 `True` |
| `_ProgressReader` | 内部类（文件包装） | 上传读取时回调进度 | 用完 `close()` |

**关键数据**：

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `CONNECT_TIMEOUT` | int | 10 | 连接超时（秒） |
| `READ_TIMEOUT` | int | 1800 | 读超时（秒）；大会话上传+服务器解包入库可能耗时数分钟，设 30 分钟 |
| `User-Agent` | str | `"DAQ-SDK/1.0"` | 自建 session 时写入请求头 |
| API 端点 | — | `POST /api/v1/session/upload`、`GET /api/v1/sessions`、`GET /api/v1/session/{id}`、`DELETE /api/v1/session/{id}`、`GET /api/v1/video/{id}/{cam}/stream`、`GET /health` | 模块 docstring 中的真实 API 清单 |

**调用关系**：被 `core/uploader.py`、`ui/upload_dialog.py`（懒加载用于测试连接与项目列表）import。依赖 `requests`。

### core/task_service.py

**作用**：后台任务轮询服务。`QTimer` 定时触发，每次轮询在 `threading.Thread`
后台线程执行 HTTP 请求，不阻塞 UI；通过 Qt 信号把任务列表与连接状态通知
UI。支持 Cookie-based JWT 认证：先 POST `/api/v1/auth/login`
（`remember_me=True`），复用带 cookie 的 `requests.Session` 轮询
GET `/api/v1/device/tasks?device_name=…`。设计模式参照已移除的 `SyncService`。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_normalize_tasks` | `(raw: list[dict]) -> list[dict]`（模块级函数） | 后端字段 → 内部字段名映射 | `current_count`→`completed_count`，状态 `"active"`→`"in_progress"`；输出含 `id/name/description/status/total_required/completed_count/assigned_at/assigned_user（null/空→None）/params` |
| `TaskService` | `(server_url: str = "", parent=None)` | 任务轮询服务；URL 缺省取 `settings.SERVER_URL`，轮询间隔取 `settings.TASK_POLL_INTERVAL_MS`（缺省 30000） | 自建带 cookie 的 `requests.Session`，缓存最近一次任务列表；身份三态 `_identity`：None=未决 / `"guest"` / 用户名，`_epoch` 代次守卫丢弃过期身份响应 |
| `TaskService.verify_credentials` | `(url, username, password)`（staticmethod） | 独立 Session 同步校验登录凭据（供登录对话框用，不触碰实例状态） | `(ok: bool, msg: str, cookies)`，成功时 cookies 由 `adopt_login` 接管 |
| `TaskService.adopt_login` | `(url, username, cookies)` | 采纳已验证登录：cookie 拷入实例 Session，切换账号身份并立即轮询 | `identity_changed` 信号 |
| `TaskService.set_guest` | `(url: str \| None = None)` | 切换游客身份：清 cookie 并立即轮询（只应拉到公共任务） | `identity_changed` 信号 |
| `TaskService.current_identity` | `() -> str \| None` | 当前身份：None / `"guest"` / 用户名 | — |
| `TaskService.start` / `stop` | `()` | 启动 / 停止定时轮询 | `start` 只启动 QTimer 不触发拉取——首次拉取由身份确定路径（`adopt_login`/`set_guest`）触发，UI 在身份确定后才 `start()` |
| `TaskService.poll_now` | `()` | 手动触发一次拉取 | 不改变定时器状态 |
| `TaskService.set_server_url` | `(url: str)` | 修改服务器地址并立即重新轮询 | 清 cookie、置未登录 |
| `TaskService.set_credentials` | `(username, password)` | 更新登录凭据 | — |
| `TaskService.set_server_and_credentials` | `(url, username, password)` | 同时更新地址与凭据 | 清 cookie、置未登录并重新轮询 |
| `TaskService.set_interval` | `(ms: int)` | 动态修改轮询间隔 | — |
| `TaskService.cached_tasks` | `() -> list[dict]` | 返回缓存任务列表 | 不发网络请求 |
| `TaskService.flush_now` | `()` | 录制完成后的即时进度上报（后台线程） | 本机显示不依赖它，仅为其他机器秒级传播；轮询 tick 兜底 |
| 信号 | `tasks_updated(list)` / `connection_status(bool)` / `error_occurred(str)` / `login_result(bool, str)` / `identity_changed(str)` / `identity_expired()` / `progress_synced()` | 任务刷新 / 连接状态 / 出错 / 登录结果 / 身份确定或切换（`"guest"` 或用户名）/ 登录态轮询遇 401/403（会话过期，UI 重登或降级游客）/ 进度增量上报成功后本地计数已更新 | 登录态 401/403 → 身份置空 + `identity_expired`；游客态 401 只报错不断身份；网络错误与 5xx 只记错误、不改连接状态 |
| 内部 | `_trigger_poll` / `_trigger_login_and_poll` / `_login_and_poll` / `_do_login` / `_poll` / `_post_progress` / `_note_fail` / `_flush_progress` | 后台线程调度、登录、拉取、进度水位合并上报 | daemon 线程执行；`_flush_progress` 仅在 GET 200 后调用，`_flush_lock` 防并发重复 POST |

**关键数据**：

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `CONNECT_TIMEOUT` | int | 10 | HTTP 请求超时（秒） |
| 轮询间隔 | int | `settings.TASK_POLL_INTERVAL_MS`（默认 30000） | 读不到时 `getattr` 兜底 5000 毫秒（代码兜底值，非配置默认） |
| 设备认领名 | str | `settings.DEVICE_NAME` | 读不到时回退 `"EGO_001"`；作为 `device_name` 查询参数 |
| API 端点 | — | `POST /api/v1/auth/login`、`GET /api/v1/device/tasks`、`POST /api/v1/device/tasks/progress` | 登录 body 含 `username/password/remember_me`；进度上报 body 含 `task_id/session_id/increment/device_name`，`session_id = "{device}:{task_id}:{水位}"` 幂等（后端按 `(device_name, session_id)` 去重，重复请求返回当前全局数不重复加；游客上报公共任务应放行）；404 → `_progress_supported=False` 静默降级本地口径 |
| `User-Agent` | str | `"DAQ-SDK/1.0"` | — |

**调用关系**：被 `ui/main_window.py` 实例化。依赖 `requests` 与 `PyQt5.QtCore`。

### core/uploader.py

**作用**：上传队列管理器。工作流：①（可选）用 ffmpeg 把 session 内 videos
重编码到低码率（保持分辨率/帧率不变，帧级对齐不受影响）；②用 `zipfile`
打包 session 目录为 zip（`ZIP_STORED` 免二次压缩，EgoData 路径转 LeRobot v3
兼容路径）；③经 `APIClient` 上传；④状态持久化到 SQLite 表 `upload_task`，
全程通过 Qt 信号通知 UI。失败自动重试（最多 `settings.UPLOAD_RETRY_MAX`
次），临时 zip 与预压缩产物用完即删。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `UploadTask` | `(session_path: str, server_url: str)` | 单个上传任务数据对象（`__slots__` 11 字段） | 状态初值 `pending`，进度 0.0 |
| `UploadTask.to_dict` / `UploadTask.from_row` | `() -> dict` / `(row: dict)`（staticmethod） | 序列化 / 从数据库行还原 | — |
| `UploadManager` | `(server_url: str = "", session=None)` | 上传队列管理器；URL 缺省 `settings.SERVER_URL`，重试/并发取 `settings.UPLOAD_RETRY_MAX` / `settings.UPLOAD_MAX_CONCURRENT`；`session` 可复用已认证 `requests.Session` | 队列与活跃任务表由 `QMutex` 保护 |
| `UploadManager.server_url` | property（可写） | 当前服务器地址 | — |
| `UploadManager.add_task` | `(session_path: str) -> str` | 入队一个任务（先写库再入队） | 返回 `task_id` |
| `UploadManager.add_tasks` | `(session_paths: list[str]) -> list[str]` | 批量入队 | 只收存在的目录 |
| `UploadManager.start` / `stop` | `()` | 启动 / 停止 worker 线程 | `stop` 等线程最多 10 秒 |
| `UploadManager.pending_count` / `active_count` / `all_done` | `() -> int` / `() -> int` / `() -> bool` | 队列计数查询 | — |
| 信号 | `task_added(str)` / `task_started(str)` / `task_status(str, str)` / `task_progress(str, float)` / `task_completed(str)` / `task_failed(str, str)` / `all_completed()` | 入队 / 开始 / 状态文字 / 进度 0~1 / 完成 / 失败 / 全部完成 | — |
| `UploadManager.get_upload_status` | `(session_path: str) -> str`（staticmethod） | 查某会话最近一条上传状态 | 无记录返回 `"pending"` |
| `UploadManager.list_tasks` | `(session_path: str = "") -> list[dict]`（staticmethod） | 查上传记录（可按会话过滤） | 按 `created_at` 倒序 |
| 内部 | `_worker_loop` / `_upload_one` / `_zip_session` / `_find_working_ffmpeg` / `_precompress_videos` / `_save_to_db` | 并发调度、单任务全流程、打包、ffmpeg 探测、视频预压缩、写库 | 打包与预压缩的临时文件名均带 `task_id`，避免同进程并发任务互相覆盖 |

**关键数据**：

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| 并发 / 重试 / CRF | 常量 | 取自 `settings.UPLOAD_MAX_CONCURRENT`（默认 1）、`settings.UPLOAD_RETRY_MAX`（默认 3）、`settings.UPLOAD_VIDEO_CRF`（默认 30） | 串行=1 的原因：并发时预压缩临时文件互相覆盖致视频损坏 |
| 上传状态值 | str | `"pending"` / `"completed"` / `"failed"` | 重试期间回 `pending` 重新入队 |
| `upload_task` 表字段 | — | `id, session_path, session_name, status, progress, retry_count, server_url, server_session_id, error_message, created_at, updated_at` | `INSERT OR REPLACE` 写 `core/database.py` 的 `db` |
| 进度区间约定 | — | 预压缩 0~0.08，打包 0.08~0.12，上传 0.12~1.0 | `task_progress` 的 0~1 映射 |
| 路径转换 | — | EgoData `chunk-0000` → LeRobot v3 `chunk_0000`；`videos/stereo_left/chunk-0000/stereo_left_aux.mp4` → `videos/stereo_left/chunk_0000/stereo_left_aux.mp4`；旧扁平结构 `videos/<file>.mp4` → `videos/<cam>/chunk_0000/file-0000.mp4` | 打包时改写 arcname |
| ffmpeg 候选 | str 列表 | `~/miniconda3/envs/lerobot/bin/ffmpeg`、`~/anaconda3/envs/lerobot/bin/ffmpeg`、PATH 上的 `"ffmpeg"` | conda base 的 ffmpeg 因 openvino/tbb 库冲突不可用，故优先 lerobot 环境的独立 ffmpeg；找不到则跳过压缩 |
| 预压缩参数 | — | `libx265`、`-crf UPLOAD_VIDEO_CRF`（默认 30）、`-preset veryfast`、`-tag:v hvc1`、`-pix_fmt yuv420p`、单文件超时 3600 秒 | 仅处理 `videos/` 下 `.mp4/.avi/.mov`；v1.0.9 起先 `ffmpeg -i` 探测源编码，已是 HEVC（录制直出）的跳过预压，探测失败按未知照旧压缩 |

**调用关系**：被 `ui/main_window.py`、`ui/upload_dialog.py`、`ui/playback_dialog.py`（懒加载）import。自身调用：`config.settings`、`core.database.db`、`core.api_client.APIClient`。依赖 `PyQt5.QtCore`（`QObject`、`pyqtSignal`、`QMutex`、`QMutexLocker`）。

## 回放与会话

### core/session_timeline.py

**作用**：录制会话传感器时间线——向量化合并 per-sensor parquet + 帧号二分
查询。替代 `ui/playback_dialog.py` 中逐行 dict 合并（docstring 实测：60 分钟
会话 179464 行，32.5s → 约 1s），提供 O(log n) 的帧号→传感器行查询。纯
numpy/pyarrow 实现，无 Qt 依赖，可在后台线程运行。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `SensorTimeline` | `(frame_indices, timestamps, obs, signal_mask)` | 统一时间线（按 `(episode_index, frame_index)` 排序） | 属性：`frame_indices (N,) int64`、`timestamps (N,) float64`、`obs {col: (N,D) float32}`（缺值 NaN）、`signal_mask (N,) bool`、`signal_count int`；支持 `bool()`/`len()` |
| `SensorTimeline.nearest_for_column` | `(col, frame_idx)` | 该列中 `frame_index` 最接近的行 | `(行号, |Δ帧|)` 或 `(None, None)`，二分 |
| `SensorTimeline.nearest_for_column_time` | `(col, t_s)` | 该列中 `timestamp` 最接近 `t_s`（秒）的行；含暂停负跳变时退化为向量化 argmin | `(行号, |Δt|秒)` 或 `(None, None)` |
| `load_timeline` | `(session_dir, sensor_names)` | 加载会话传感器时间线（per-sensor 多文件或旧格式单表双路径） | `SensorTimeline` |
| `_merge_per_sensor_parquets`（内部） | `(paths, sensor_names)` | 向量化合并：按 `(episode_index, frame_index, round6(timestamp))` 分组，组内取最后一个非空观测值 | `SensorTimeline` |
| `_find_sensor_parquet` / `_find_all_sensor_parquets`（内部） | `(session_dir)` | 定位 `data/<sensor>/chunk-0000/chunk_000000.parquet` | 路径或 `[]` |
| `_column_to_matrix`（内部） | `(col, n_rows)` | pyarrow 列 → `(n_rows, D)` float32 矩阵（fixed_size_list 展平，失败返回 None） | `np.ndarray` 或 `None` |

**关键数据**：合并键列 `_KEY_COLS = ("episode_index", "frame_index", "timestamp")`（旧格式单表缺 `episode_index` 时填 0）；信号掩码语义 = 任一观测列该行 `|值|.sum > 0`；时间戳不保证严格单调（录制暂停点存在负跳变，查询用帧号）。

**调用关系**：被 `ui/playback_dialog.py`（`load_timeline`/`SensorTimeline`）、`tools/tests/test_playback_multifps.py` 引用。

### core/session_catalog.py

**作用**：录制会话目录——扫描、元数据读取与主时钟帧率解析（零 Qt 依赖，
纯函数）。供回放对话框、录制历史刷新与上传对话框共用。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `get_effective_fps` | `(info)` | 回放主时钟帧率：每路相机 fps 最大值；旧数据无 per-camera fps 时按双目命名回退 `settings.STEREO_FPS` | `float` |
| `list_sessions` | `(directory)` | 扫描全部会话（含元数据/时长/fps） | `[{name, path, tag, info, duration, fps}]` |
| `list_recordings` | `(base_dir)` | 轻量扫描全部完整 session（上传对话框口径） | `[{name, path, tag}]` |
| `load_session_meta` | `(session_dir)` | 同步读单会话元数据（格式探测 / info / fps / 传感器列名） | `{"fmt", "info", "fps", "sensor_names"}` |

**调用关系**：被 `ui/playback_dialog.py`、`ui/upload_dialog.py`（`list_recordings`）使用；调用了 `core/helpers.py`。

### core/session_loader.py

**作用**：回放后台加载器——把时间线与手部关键点加载放进后台线程，经
`finished(int, object)` / `failed(int, str)` 信号回主线程（payload 为
`{"timeline": SensorTimeline, "hand_kpts": dict}`，按引用传递）；`gen` 防过期
协议由调用方（`ui/playback_dialog.py`）维护——收到 gen 不一致的迟到结果即
丢弃。加载期间新请求排队（最新覆盖旧的，过期结果本就会被 gen 丢弃）。
VideoCapture 不在 worker 中创建/读取（OpenCV FFmpeg 后端不承诺跨线程安全），
视频打开与播放留在调用方主线程。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `SessionLoader` | `(parent=None)` | 后台加载器 | 信号 `finished(int, object)`（gen, payload）、`failed(int, str)`（gen, error） |
| `SessionLoader.start` | `(gen, session_dir, sensor_names, load_kpts=False, load_timeline=True)` | 后台线程加载；已有加载在跑时把请求排队（新覆盖旧） | 无 |
| `SessionLoader.is_running` | `()` | 是否有加载在跑 | `bool` |

**调用关系**：被 `ui/playback_dialog.py`（`.finished` 接线 + `.start(...)` 调用）使用；调用了 `core/session_timeline.py`（`load_timeline`）与 `core/hand_tracking.py`（`load_hand_kpts`，惰性导入，缺失时降级为不加载关键点）。

## 手部关键点

### core/hand_tracking.py

**作用**：手部关键点后处理独立模块，零 Qt 依赖纯函数式 API。两种追踪模式：
`"glove"`（YOLO 检测黑色手套 + RTMPose 关键点，仅 2D）与 `"bare"`（MediaPipe
裸手追踪，2D + 3D world_landmarks，供下游 3D 可视化与数据集导出使用）。
支持对已录制会话逐帧提取关键点写 parquet、手势自动标注
（`auto_labels.parquet` + `auto_actions.jsonl`）、以及关键点/标注的读取
（三层路径回退：`keypoints_output/` → `session/keypoints/` →
`session/annotations/`）。重型依赖（ultralytics/torch/rtmlib 与
`hand_pipeline`、`hand_pipeline_mediapipe`、`hand_common`）全部延迟导入。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `process_session` | `(session_path, mode="", detector="", det_device="cuda", pose_device="cuda", progress_cb, status_cb, cancel_check)` | 逐帧提取手部关键点写 parquet；bare 模式额外写 3D parquet；glove 模式无 CUDA 时设备自动降级 cpu | `{"success", "frames", "elapsed", "fps", "mode", "session_path"}` 或 `{"success": False, "error"}` |
| `label_session` | `(session_path, progress_cb, cancel_check)` | 基于手部关键点 + 传感器数据逐帧手势标注，写 `auto_labels.parquet` 与 `auto_actions.jsonl` | `{"success", "frames", "elapsed"}` 或错误 |
| `load_hand_kpts` | `(session_path)` | 读 2D 关键点 | `{frame_index: {hand_data, num_hands, track_ids}}` 或 `None` |
| `load_hand_3d` | `(session_path)` | 读 3D landmarks | `{frame_index: row}` 或 `None` |
| `load_auto_labels` | `(session_path)` | 读自动标注 | `{frame_index: row}` 或 `None` |
| `draw_kpts_overlay` | `(frame, data, track_ids=None)` | BGR 帧上叠加关键点 + 手 #id + 手势文本 | 叠加后帧 |
| `classify_gesture` | `(extended, thumb_tip_pt, index_tip_pt)` | 按伸直手指数 + 拇指-食指距离分类 | `{"gesture", "pinch", "fist", "extended_count"}` |
| `_pack_hand_data` / `_unpack_hand_data` / `_pack_hand_3d_data`（内部） | 见下 | 固定长度打包/解包 | `np.ndarray` / tuple |
| `_compute_contact`（内部） | `(motion_0, motion_1, sensors)` | 由手部运动 + 传感器峰值推接触状态 | `"grasping/holding/reaching/resting/none"` |
| `_segment_actions`（内部） | `(rows)` | 帧级手势 → 动作片段（<10 帧合并入前段） | `[{start_frame, end_frame, label, dominant_hand, confidence}]` |

**关键数据**：常量 `PER_HAND_DIM = 21*2 + 4 = 46`（21 点×2 坐标 + 4 框坐标）、`MAX_HANDS = 2`、`TOTAL_DIM = 92`、`PER_HAND_3D_DIM = 21*3 = 63`。Parquet schema（真实定义）：
- `_HAND_KPTS_SCHEMA`：`frame_index(int32)`、`num_hands(int32)`、`hand_data(list<float32,92>)`、`track_ids(list<int32,2>)`。
- `_HAND_3D_SCHEMA`：`frame_index(int32)`、`hand_0_present(bool)`、`hand_0_landmarks_3d(list<float32,63>)`、`hand_0_label(string)`、`hand_1_*` 同构。
- `_AUTO_LABELS_SCHEMA`：`frame_index(int32)`、每手 `hand_N_gesture(string)/hand_N_extended(list<string>)/hand_N_extended_count(int32)/hand_N_fist(bool)/hand_N_pinch(bool)/hand_N_center_x(float32)/hand_N_center_y(float32)/hand_N_motion(float32)`、`two_hand_distance(float32)`、`contact(string)`。
- 手势分类阈值：`PINCH_THRESH = 30.0`（像素）、`TWO_HAND_TOUCH_THRESH = 80.0`；`MIN_SEGMENT_FRAMES = 10`。
- MediaPipe 模型缓存 `~/.cache/hand_landmarker.task`（从 `settings.HAND_MEDIAPIPE_MODEL` 复制，大小不一致时更新）。

**调用关系**：被 `ui/playback_dialog.py`、`scripts/process_hands.py`、`tools/demos/demo_glove_kpts/demo_glove_video.py`、`core/hand_processor.py`、`core/auto_labeler.py` 引用。

### core/hand_processor.py

**作用**：`SessionHandProcessor`——对已录制会话逐帧跑手部关键点推理的 Qt 薄
封装，核心逻辑全在 `core.hand_tracking.process_session`。后台线程执行，经
Qt 信号回主线程。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `SessionHandProcessor` | `(parent=None)` | 后台处理器 | 信号 `progress(int, int)`、`finished(str, str)`（session_path, error_msg，空=成功）、`status_changed(str)` |
| `SessionHandProcessor.process_session` | `(session_path, detector="", det_device="cuda", pose_device="cuda", mode="")` | 启动后台处理（运行中再调被忽略） | 无 |
| `SessionHandProcessor.cancel` | 无参 | 取消当前任务 | 无 |

**调用关系**：被 `ui/main_window.py`、`ui/playback_dialog.py` 引用；调用了 `core/hand_tracking.py`。

### core/auto_labeler.py

**作用**：`AutoLabeler`——基于手部关键点数据的自动标注器，
`core.hand_tracking.label_session` 的 Qt 薄封装。后台线程执行，信号回主线程。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `AutoLabeler` | `(parent=None)` | 后台标注器 | 信号 `progress(int, int)`（已处理帧, 总帧数）、`finished(str, str)`（session_path, error_msg） |
| `AutoLabeler.label_session` | `(session_path)` | 启动后台标注（运行中再调被忽略） | 无 |
| `AutoLabeler.cancel` | 无参 | 取消（`_cancelled` 检查 `_running`） | 无 |

**调用关系**：被 `ui/main_window.py` 引用；调用了 `core/hand_tracking.py`。

## 手套传感（原 tools/sensors/）

### core/ble_engine.py

**作用**：BLE 传感器数据采集引擎。作为 `QObject` 运行，在独立线程中处理
BLE 异步通信（基于 `bleak`），通过 Qt 信号把连接状态、FPS、校准进度与错误
发送到 UI 线程。数据链路：`BleakClient` 订阅 TX 特征通知 → 原始字节缓冲 →
`_parse_loop` 按帧头/帧尾/校验和解析成 16×16 压力矩阵 → `process_frame()`
做校准、时域平滑、去基线、漂移补偿、动态噪声门与空间滤波，返回处理后的
矩阵供 UI 渲染（UI 侧是轮询式取帧，而非信号推帧）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `SensorBLEEngine` | `__init__(self)` | 初始化 16×16 数据数组、降噪参数（`base_noise_gate=500` 等）、校准状态与线程锁 | 无 |
| `start_scan` | `start_scan(self)` | 启动 daemon 线程执行 5 秒 BLE 扫描（`BleakScanner.discover`），按名称含 `"Matrix"`、非空名称、RSSI 排序 | 完成后发 `scan_complete`；失败发 `error_occurred` |
| `connect_device` | `connect_device(self, address: str)` | 记录目标地址，启动 `_ble_loop`（连接 + 订阅通知 + 断线重连）与 `_parse_loop`（帧解析）两个 daemon 线程 | 连接成功发 `connected`；失败发 `error_occurred` 与 `disconnected` |
| `disconnect` | `disconnect(self)` | 置 `_running=False`，结束通信与解析循环 | 循环退出时发 `disconnected` |
| `start_calibration` | `start_calibration(self)` | 清空校准缓冲并进入校准态 | 校准期间 `process_frame` 返回 `(None, 0)` 并逐帧发 `calibration_progress` |
| `set_noise_gate` | `set_noise_gate(self, value: int)` | 设置基础噪声门 | 无返回 |
| `set_dynamic_ratio` | `set_dynamic_ratio(self, value: float)` | 设置动态噪声门比例 | 无返回 |
| `set_spatial_filter` | `set_spatial_filter(self, enabled: bool)` | 开关空间滤波 | 无返回 |
| `set_temporal_smooth` | `set_temporal_smooth(self, value: float)` | 设置时域平滑系数 | 无返回 |
| `process_frame` | `process_frame(self) -> tuple` | 一帧完整处理：校准 → 时域平滑 → 去基线 → 漂移补偿 → 动态噪声门 → 空间滤波 | 返回 `(processed_data, max_signal)`；校准期间返回 `(None, 0)` |
| 信号 | `device_found(str, str, int)` / `scan_complete(list)` / `connected(str)` / `disconnected()` / `data_ready(ndarray)` / `fps_updated(float)` / `error_occurred(str)` / `calibration_progress(int)` | Qt 信号定义，UI 侧连接使用 | `device_found` 与 `data_ready` 在本文件内定义但未 emit；UI 实际通过轮询 `process_frame()` 取帧 |

**关键数据**：

- 传感器矩阵：`MATRIX_ROWS = 16`、`MATRIX_COLS = 16`，数据以 `np.uint16` 解析后转 `float32`。
- 帧协议：帧头 `b"\xAA\x55"` + 512 字节数据 + 1 字节 XOR 校验和 + 帧尾 `b"\xFB\x03"`；校验和不通过则整帧丢弃。512 字节 = 256 个 uint16 → reshape 为 16×16。
- BLE 特征：`TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"`（Nordic UART 服务 TX 特征），通知回调只做字节缓冲。
- 校准：`CALIBRATION_FRAMES = 100`，`baseline_map` = 校准帧逐格最大值 + 10；`TARGET_FPS = 100` 为常量定义。
- 降噪默认参数：`base_noise_gate=500`、`dynamic_noise_ratio=0.0`、`temporal_smooth=0.15`、`spatial_filter_enabled=True`。
- 漂移补偿：`processed` 的 40 分位数 > 100 时，减去其 0.8 倍（`drift_baseline_val` 同时供 UI 显示）。
- 动态噪声门：`min(current_max * dynamic_noise_ratio, 250)`，与基础门取较大者；低于门限的格点清零。
- 空间滤波：孤立点判据——邻域（3×3 无中心核）全空且值低于 `max(baseline_map * 2.5, 180)` 的格点清零。
- 时间戳：`latest_data_ts_us`（微秒）由解析线程写入、UI 线程读取，用于数据年龄判断。

**调用关系**：被 `ui/glove_widget.py` 导入并实例化（连接信号后调用 `connect_device` / `disconnect`，每帧轮询 `process_frame()` 与公开属性如 `is_calibrating`、`latest_data_ts_us`、`hardware_fps`）；被 `tools/tests/glove_widget_test.py` mock 替换。依赖 `bleak`、`PyQt5`、`numpy`（空间滤波分支依赖 `cv2`）。

### core/render_engine.py

**作用**：传感器数据渲染引擎，实现五种可视化模式：热力图（`render_heatmap`）、
轨迹（`render_trace`）、网格数值（`render_grid`）、仿生手掌（`render_hand`）、
拓扑形变（`render_deform_mesh`）。所有渲染函数接收处理后的 16×16 数据与
配置，返回 BGR 格式的 numpy 帧，由 UI 层转为 `QPixmap` 显示。本文件同时
集中定义仿生手掌映射配置路径常量（左右手各一套 JSON，位于 `config/sensors/`），
供 `ui/glove_widget.py` 与 `ui/playback_dialog.py` 复用——代码注释说明这些
常量原先定义在 `sensors.sensor_panel`（已删除），迁到 core 是为了让 UI 取
路径时不必引入 `bleak`。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `CONFIG_DIR` / `CONFIG_FILE` / `CONFIG_FILE_LEFT` | 模块常量 | 指向 `config/sensors/hand_ble_config.json` 与 `hand_ble_config_left.json`（基于 `settings.BASE_DIR`） | — |
| `_get_viridis_lut` | `_get_viridis_lut() -> np.ndarray` | 延迟初始化 256 级 Viridis 颜色查找表（`VIRIDIS_LUT`，首次调用构建） | 返回 `(256, 1, 3)` uint8 LUT |
| `_draw_hud` | `_draw_hud(frame, lines, color=(255,255,255))` | 帧左上角绘制半透明 HUD 信息栏 | 原地修改 `frame` |
| `_load_json` | `_load_json(path, default)` | 读取 JSON；`default` 为 dict 时浅合并（默认值兜底），异常静默回退 | 返回配置 dict |
| `render_heatmap` | `(processed, max_signal, config, window_size, current_vmax, fps, noise_gate, dyn_ratio, spatial_on)` | 模式 1：子矩阵提取 → subpixel 2× 超分 → 缩放/高斯模糊 → Viridis 伪彩 | 返回 `(frame, new_vmax)` |
| `render_trace` | `(processed, max_signal, config, window_size, current_vmax, fps, noise_gate, dyn_ratio, spatial_on)` | 模式 2：历史数据叠加显示（`_trace_canvas` 取逐格最大值），支持 `clear_trace_canvas` 清空 | 返回 `(frame, new_vmax, changed)` |
| `clear_trace_canvas` | `clear_trace_canvas()` | 清空轨迹画布 | 全局 `_trace_canvas` 置 `None` |
| `render_grid` | `(processed, max_signal, config, window_size, fps)` | 模式 3：每格显示数值，蓝色强度映射幅值 | 返回 `frame` |
| `render_hand` | `(processed, max_signal, config, window_size, current_vmax, fps, noise_gate, dyn_ratio, spatial_on, drift)` | 模式 4：仿生手掌映射——骨骼框架 + 各手指区域传感器格点上色 | 返回 `(frame, new_vmax)` |
| `DeformMeshState` | `__init__(self)` | 模式 5 交互状态：`holes` / `flip_x` / `flip_y` / `deform_strength` / 网格缓存 / 拖拽绘制状态 | 无 |
| `render_deform_mesh` | `(processed, max_signal, config, window_size, current_vmax, fps, state: DeformMeshState)` | 模式 5：网格 + 孔洞形变 + 数据点圆 | 返回 `(frame, new_vmax)` |

**关键数据**：

- `HAND_ANCHORS`：16 个手部锚点坐标（`thumb/index/middle/ring/pinky` 各自的 `tip/joint/base` + `palm`），基于 1280×720 画布设计，运行时按实际 `window_size` 等比缩放；`WRIST_ANCHOR = (640, 600)`。
- `DEFAULT_HAND`：默认 16×16 传感器到手部区域的映射，如 `thumb_joint: rows [0,1,2] cols [14,12,13,15]`、`palm: rows 0-14 cols [10,9,8,6,4]`，各手指关节区域共享 `axis_order: "col_row"`。
- 配置 JSON 字段（`config/sensors/hand_ble_config.json` 与 `hand_ble_config_left.json`）：16 个部位键（`thumb_tip` … `palm`），每项含 `name`（中文部位名，如 "大拇指-指肚"）、`rows`、`cols`、`axis_order`（`"row_col"` 或 `"col_row"`）。左右两文件键同值异（左手套关节锚点不同）。
- 热力图/轨迹增强参数：`subpixel`（默认 True，梯度 2× 超分）、`gamma`（默认 1.0）、`blur`（默认 5，奇数化）。
- vmax 自适应（热力图/轨迹模式）：候选值 `max(5000, p99.7)`，上浮时按 `current_vmax*0.95 + vmax_cand*0.05` 平滑；仿生手掌/形变模式改用 `max(max_signal, 5000)`。
- 形变网格：`CELL = 26`（形变模式）/ `CELL = int(14 * min(sx, sy))`（仿生手掌模式）。

**调用关系**：被 `ui/glove_widget.py`（`render_hand`、`DEFAULT_HAND`、`_load_json`、`CONFIG_FILE`、`CONFIG_FILE_LEFT`）与 `ui/playback_dialog.py`（五种模式、`DeformMeshState`、`clear_trace_canvas`）调用。依赖 `config.settings`（`BASE_DIR`）、`numpy`、`cv2`。

### core/sensor_config_dialogs.py

**作用**：PyQt5 配置对话框。`MatrixConfigDialog` 是通用 16×16 矩阵行列选择
界面（按钮点击选行/列，可撤销/清空，可选映射方向）；`HandConfigDialog` 是
仿生手掌模式配置界面，为每个手部部位（指尖/关节/指根/手掌）列出当前
rows/cols 映射，点击"选择"弹出子级 `MatrixConfigDialog` 逐部位编辑。两者
操作的数据结构与 `render_engine.py` 的配置 JSON 字段一致（`rows` / `cols` /
`axis_order`）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `MatrixConfigDialog` | `__init__(self, parent, title: str, config: dict)` | 构建行列选择对话框（16 个行按钮 + 16 个列按钮 + 映射方向单选） | 修改内部 `_rows/_cols/_order` |
| `_add_row` / `_undo_row` / `_clear_row` | `_add_row(self, val)` 等 | 行选择序列的追加/撤销/清空 | 更新标签显示 |
| `_add_col` / `_undo_col` / `_clear_col` | `_add_col(self, val)` 等 | 列选择序列的追加/撤销/清空 | 更新标签显示 |
| `get_config` | `get_config(self) -> dict` | 输出当前选择 | 返回 `{"rows", "cols", "axis_order"}` |
| `HandConfigDialog` | `__init__(self, parent, config: dict)` | 深拷贝配置并构建逐部位配置列表 | 无 |
| `_format_cfg` | `_format_cfg(self, item)` | 把部位配置格式化为单行摘要文本 | 返回字符串 |
| `_open_sub_config` | `_open_sub_config(self, key)` | 弹出 `MatrixConfigDialog` 编辑某部位 | 确认后写回 `item["rows"/"cols"/"axis_order"]` 并刷新标签 |
| `get_config` | `get_config(self) -> dict` | 输出全部部位配置 | 返回完整配置 dict |

**关键数据**：配置结构为 `{部位键: {"rows": [...], "cols": [...], "axis_order": "row_col"|"col_row", 可选 "name"}}`；`MatrixConfigDialog` 固定显示 16 行（00-15）与 16 列（00-15）按钮。

**调用关系**：`HandConfigDialog` 内部调用 `MatrixConfigDialog`（子对话框），与 `render_engine.py` 的配置口径保持一致。

### core/sensor_hand_config.py

**作用**：传感器（BLE 手套）仿生手掌配置与有效性过滤（零 Qt 依赖，纯函数）。
供 `ui/glove_widget.py` 与 `ui/playback_dialog.py` 共用同一套口径。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `load_sensor_hand_config` | `(sensor_name)` | 按传感器角色加载仿生手掌映射（左/右手套不同配置文件，`DEFAULT_HAND` 之外部位并入） | `dict` |
| `valid_sensor_names` | `(timeline, names)` | 只保留时间线中确实存在 16×16 压力矩阵（256 宽）的传感器列（防幽灵传感器格） | `list` |

**调用关系**：被 `ui/glove_widget.py`（hand_config 加载）、`ui/playback_dialog.py` 使用；调用了 `core/render_engine.py`。

## 通用工具

### core/helpers.py

**作用**：通用工具函数集合（原 utils/helpers.py），按主题分五块：①基础工具
（ID/时间/回收站/格式化）；②会话摘要（供录制历史面板"摄像机/时长/大小"
三列使用）；③命名与 LeRobot v3 分块路径；④关键点数据路径
（`keypoints_output/` 镜像结构 + 两级回退路径）；⑤EgoData 格式路径、会话
枚举与格式检测。无类，全部为模块级函数。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `new_id` | `() -> str` | 生成短 ID | 12 位十六进制（`uuid4().hex[:12]`） |
| `utcnow` | `() -> str` | UTC 时间戳 | ISO-8601 字符串 `YYYY-MM-DDTHH:MM:SSZ` |
| `send_to_recycle_bin` | `(path: str) -> bool` | 将文件/文件夹移入回收站 | Windows 走 `SHFileOperationW`(FOF_ALLOWUNDO)；Linux 优先 `gio trash`，无 gio 时手动移入 `~/.local/share/Trash`（写 `.trashinfo` 记录原路径）；成功 `True` |
| `format_duration` | `(seconds: float) -> str` | 秒数格式化 | `HH:MM:SS` 字符串 |
| `session_size_mb` | `(session_dir: str) -> float` | 累加会话目录全部文件大小 | MB；目录不存在/出错返回 0.0 |
| `read_metadata` | `(session_dir: str) -> dict` | 读会话 `metadata.json` | 任何失败返回 `{}` |
| `session_summary` | `(session_dir: str, frames_by_cam: dict) -> tuple` | 会话展示摘要（录制历史面板三列数据源） | 返回 `(camera_list, duration_sec, size_mb)`；时长=各相机 帧数/fps 的最大值，帧数全缺时降级 `timestamps.json` 最后一条相对时间，再降级 `total_frames / 最小 fps` |
| `format_size_mb` | `(mb: float) -> str` | 大小列文本 | `"12.3 MB"`；0/缺失显示 `"-"` |
| `timestamp_filename` | `(prefix: str = "rec", ext: str = ".avi") -> str` | 带时间戳文件名 | 如 `rec_20260804_143052.avi` |
| `task_tag` | `(task_name: str = "") -> str` | 任务分类标签（子文件夹名） | 清洗后标签，如 `"grasp_cup"`；空输入返回 `"session"` |
| `session_dirname` | `(task_name: str = "", batch_index: int = 0) -> str` | 会话目录名 | 如 `grasp_cup_20260805_195046`；`batch_index>0` 时插入 4 位批次号 |
| `chunk_dir` / `chunk_file` | `(chunk_index: int = 0) -> str` / `(chunk_index: int = 0, name: str = "file") -> str` | LeRobot v3 分块目录/文件名 | `"chunk-000"` / `"file-000"` |
| `data_parquet_path` | `(session_dir, chunk_index=0) -> str` | 数据 parquet 路径 | `data/chunk-000/file-000.parquet` |
| `episode_parquet_path` | `(session_dir, chunk_index=0) -> str` | episode 元数据路径 | `meta/episodes/chunk-000/file-000.parquet` |
| `video_mp4_path` | `(session_dir, camera_key, chunk_index=0) -> str` | 视频路径 | `videos/<camera_key>/chunk-000.mp4` |
| `tasks_jsonl_path` | `(session_dir) -> str` | 任务记录路径 | `meta/tasks.jsonl` |
| `keypoints_video_dir` | `(session_dir) -> str` | 关键点视频目录 | `keypoints_output/<项目>/<会话>/videos/` |
| `hand_kpts_parquet_path` | `(session_dir) -> str` | 手部 2D 关键点路径 | `keypoints_output/…/hand_pose/chunk-000.parquet` |
| `auto_labels_parquet_path` | `(session_dir) -> str` | 自动标注路径 | `keypoints_output/…/auto_labels/auto_labels.parquet` |
| `hand_3d_parquet_path` | `(session_dir) -> str` | 手部 3D 关键点路径 | `keypoints_output/…/hand_pose_3d/chunk-000.parquet` |
| `list_all_sessions` | `(base_dir: str) -> list` | 扫描全部会话目录（兼容 EgoData 与 LeRobot v3） | `[(ses_path, ses_name, tag_name), …]`，按名称倒序 |
| `episode_dirname` | `(episode_index: int = 1, task_name: str = "", digits: int = 6) -> str` | EgoData episode 目录名 | 有任务名 `Chew_gum_000001`，否则 `episode_000001` |
| `next_episode_index` | `(base_dir, task_name="", batch_index=0) -> int` | 下一个 episode 索引 | `max(batch_index, 目录扫描最大值+1)`；`batch_index<=0` 时纯目录扫描 |
| `egodata_video_path` / `egodata_video_dir` | `(episode_dir, camera_name) -> str` | 视频文件 / 目录路径 | `videos/<base_cam>/chunk-0000/<camera_name>.mp4`；`_aux` 后缀摄像头归入主摄像头目录 |
| `egodata_depth_path` | `(episode_dir, depth_name, frame_index) -> str` | 深度帧路径 | `depth/head_depth/000001.png`（6 位序号，png16） |
| `egodata_image_path` | `(episode_dir, camera_name, frame_index) -> str` | 图像帧路径 | `images/head_left_rgb/000001.jpg` |
| `egodata_calibration_path` | `(episode_dir, calib_name: str = "head_stereo") -> str` | 标定文件路径 | `calibration/head_stereo.json` |
| `egodata_metadata_path` | `(episode_dir) -> str` | 元数据路径 | `episode_dir/metadata.json` |
| `egodata_timestamps_path` | `(episode_dir) -> str` | 时间戳路径 | `episode_dir/timestamps.json` |
| `egodata_sensor_data_dir` / `egodata_sensor_parquet_path` | `(episode_dir, sensor_name) -> str` | 传感器数据目录 / parquet | `data/<sensor_name>/chunk-0000/` 与 `…/chunk_000000.parquet` |
| `detect_session_format` | `(session_dir: str) -> str` | 会话格式检测 | `"egodata"`（有 `metadata.json`）/ `"lerobot_v3"`（有 `meta/info.json`）/ `"unknown"` |
| `list_all_egodata_sessions` | `(base_dir: str) -> list` | 扫描全部 EgoData episode | `[(ep_path, ep_name, tag_name), …]`，按名称倒序 |
| 内部函数 | `_slot_base` / `_sub_name` / `_device_display_names` / `_timestamps_duration` / `_sanitize_tag` / `_keypoints_session_dir` | 流名→设备基名、槽名→子画面名、录制历史"摄像机"列文本、降级补算时长、标签清洗、session→镜像目录映射 | 供公开函数内部使用（见下方"会话摘要"说明） |
| 内部函数 | `_session_kpts_hand_kpts_path` / `_session_kpts_hand_3d_path` / `_session_kpts_auto_labels_path` / `_legacy_hand_kpts_path` / `_legacy_hand_3d_path` / `_legacy_auto_labels_path` | 回退路径 1：会话内 `keypoints/`；回退路径 2：最旧版 `annotations/` | 历史数据兼容定位 |

**关键数据**（路径/命名约定，均为代码中真实存在的硬编码结构）：

- 会话识别标志：EgoData 会话 = 目录内含 `metadata.json`；LeRobot v3 会话 = 目录内含 `meta/info.json`。
- 关键点输出镜像：`data/recordings/<项目>/<会话>` → `keypoints_output/<项目>/<会话>`（`_keypoints_session_dir` 基于 `settings.RECORDING_DIR` 与 `settings.KEYPOINTS_OUTPUT_DIR` 映射）。
- 关键点回退查找顺序：`keypoints_output/` 镜像 → 会话内 `keypoints/` → 会话内 `annotations/`（最旧版，自动标注为 `annotations/mmpose/auto_labels.parquet`）。
- 会话摘要约定（`session_summary`）：每相机 fps 优先取 `cameras[cam].fps`，缺省 `metadata["fps"]`，再缺省 `settings.RECORDING_FPS`；`camera_list` 由 `_device_display_names` 按设备聚合——D435 一台设备出 rgb+depth 两路子画面显示为 `"D435_depth (rgb, depth)"`，双目 S80M 显示为 `"FaysSense S80M (left, right)"`，单流设备只显示设备名；`devices` 段缺失（legacy 会话）时按 `_slot_base` 对 `cameras` 键分组。
- 命名清洗规则（`_sanitize_tag`）：保留字母、数字、下划线与中文（Unicode `一-鿿`），其余字符替换为 `_`。
- episode 索引兼容：新命名 `<tag>_NNNNNN` 与旧命名 `episode_NNNNNN`（位数取 `settings.EPISODE_DIGITS`）。

**调用关系**：被 `core/`（`recording_record.py`、`egodata_writer.py`、`hand_tracking.py`、`session_timeline.py`、`session_catalog.py`）、`ui/`（`main_window.py`、`camera_widget.py`、`playback_dialog.py`、`upload_dialog.py`）、`scripts/process_hands.py`、`tools/tests/` 各 import 点引用；其中 `session_summary` 供录制历史面板生成"摄像机"列摘要，`send_to_recycle_bin` 供回放对话框删除会话。自身 `from config import settings`。

### core/stereo_depth.py

**作用**：双目深度计算模块。`StereoDepthComputer` 基于手写 OpenCV
`StereoSGBM`（未用 SDK 深度引擎）+ `DisparityWLSFilter`（opencv-contrib），
从左右目 BGR 帧算视差图，输出 uint16（像素值 = 视差 × 16，OpenCV 标准亚
像素格式），单位可经标定参数（baseline × fx / disparity）换算为毫米。
`depth_to_heatmap` 与 `DepthHeatmapSmoother` 是深度可视化的共享工具（实时
显示与 `core/egodata_writer.py` 落盘热力图共用）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `depth_to_heatmap` | `(depth_uint16, near_mm=0.0, far_mm=0.0, colormap=cv2.COLORMAP_JET, smooth_k=0)` | uint16 毫米深度 → BGR 热力图；near/far 同给为真固定色标（整幅 clip 0..255，demo 口径：无效值/近端→JET(0) 深蓝、超远→红饱和，不置黑），否则帧内自适应（min/max→1..255，无效→JET(0) 深蓝）；`smooth_k>0` 先 k×k 中值滤波 | `np.ndarray (H, W, 3) uint8` |
| `DepthHeatmapSmoother` | `(alpha=0.5)` | 热力图时域 EMA（仅可视化通道，原始 PNG16 不受影响） | `update(bgr)` 返回平滑帧；`reset()` |
| `StereoDepthComputer` | `(resolution=(800, 640), num_disparities=128, block_size=11)` | S80M 双目调优的 SGBM 计算器；`num_disparities` 强制 16 倍数、`block_size` 强制奇数 | 无 |
| `StereoDepthComputer.compute` | `(left, right)`（BGR (H,W,3)） | 灰度化 + 直方图均衡 → 左/右视差 → WLS 滤波 → ×16 转 uint16 | `np.ndarray (H, W) uint16` |
| `StereoDepthComputer.num_disparities` / `block_size` | property | 参数查询 | `int` |

**关键数据**：SGBM 参数（代码硬编码）：`minDisparity=0`、`P1=8*3*block²`、`P2=32*3*block²`、`disp12MaxDiff=1`、`preFilterCap=63`、`uniquenessRatio=10`、`speckleWindowSize=100`、`speckleRange=32`、`mode=STEREO_SGBM_MODE_SGBM_3WAY`；WLS `lambda=8000.0`、`sigmaColor=1.5`。输出视差缩放：value=256 → 视差 16 像素。

**调用关系**：被 `ui/main_window.py`（`_on_s80m_depth` 用 `depth_to_heatmap` 渲染 S80C 深度格）、`core/d435_manager.py:24` 与 `core/egodata_writer.py:41`（`depth_to_heatmap`/`DepthHeatmapSmoother` 热力图生成/落盘）、`tools/tests/test_depth_heatmap.py:20` 引用。`StereoDepthComputer`（旧 S80M 视差路径）自 v1.0.11 起无调用方，仅为 demo 保留。

## 数据流

**采集链路**：`ui/main_window.py` 经 `core/device_detector.py`（复用
`core/camera.py` 的 sysfs 枚举）获得统一设备列表；面板开关经
`core/device_manager.dispatch_toggle` 分派到 `_open_*`。UVC 走
`core/camera.py` 的 `CameraWorker`；S80M 走 `core/s80m_manager.py` 的子进程
管道（帧 → `frame_ready` 信号 → `_on_stereo_frame` 抽帧口径）；D435 走
`core/d435_manager.py` + `core/d435_camera.py`。所有帧汇入
`core/pipeline.py` 的 `CameraPipeline` → 独立写入线程按 30fps 节拍调
`core/egodata_writer.py` 落盘（MP4 + Parquet + 深度热力图/PNG16 + 元数据）。
手套数据经 `core/ble_engine.py` 采集 → `pipeline.write_sensor` 入队。

**持久化链路**：会话结束后 `ui/main_window.py` 经
`core/recording_repository.py`（→ `core/database.py` 单例 `db`）把录制摘要写
进 `data/pipeline.db` 的 `recording` 表，上传任务另记 `upload_task` 表
（`core/uploader.py` 全程状态持久化）；任务进度独立走
`core/task_record.py` 的 `data/tasks.json`（录制完成 →
`increment_task_completed` 本机 `local_count` +1 → 显示值重算，与本地会话
文件是否被删无关），随后 `TaskService._flush_progress` 把本机未上报增量
以水位合并方式 POST `/api/v1/device/tasks/progress`（幂等键
`{device}:{task_id}:{水位}`，断网/失败由下个轮询 tick 兜底重试）。

**上传链路**：`ui/upload_dialog.py` 经 `core/uploader.py`
（`UploadManager`：ffmpeg 预压缩 → zip 打包 → `core/api_client.py` 上传 →
SQLite 状态）→ `POST /api/v1/session/upload`；`meta/info.json` 的结构即上传
服务器所依赖的契约。

**回放链路**：`ui/playback_dialog.py` 经 `core/session_catalog.py` 扫描/读
元数据，经 `core/session_loader.py` 后台加载（`core/session_timeline.py`
向量化合并 + `core/hand_tracking.py` 关键点），主线程只做 widget 操作。

**后处理链路**：已录制会话的手部关键点由 `core/hand_tracking.py`
（`scripts/process_hands.py` 或 UI 经 `core/hand_processor.py`/
`core/auto_labeler.py`）提取为 2D/3D parquet，供下游 3D 可视化与数据集导出
使用。
