# config/

## 定位

config/ 是应用全局配置、国际化与版本号所在的配置包。版本号唯一来源是 `config/__init__.py` 中的 `__version__ = "1.0.3"`，`config/settings.py` 以 `from config import __version__ as APP_VERSION` 从此取值，保证全系统版本号单一出处。

被几乎所有子系统 import：`core/pipeline.py`、`core/camera.py`、`core/hand_tracking.py`、`core/egodata_writer.py`、`core/render_engine.py`、`ui/main_window.py` 等均 `from config import settings`；`ui/camera_widget.py`、`ui/device_panel.py`、`ui/camera_grid.py`、`ui/glove_widget.py`、`ui/playback_dialog.py`、`ui/upload_dialog.py`、`ui/exposure_dialog.py`、`ui/task_page.py` 等 import `settings` 与 i18n 的 `tr`/`lang_manager`；`core/database.py` 直接 `from config.settings import DB_PATH, DATA_DIR`；`ui/device_panel.py` 直接 import `save_device_name`。

## 文件清单

| 文件 | 一句话作用 |
| --- | --- |
| `config/__init__.py` | 包入口，只定义版本号 `__version__`（全系统唯一来源） |
| `config/settings.py` | 全部全局配置常量，及 server_config / device_names / device_params 三类 JSON 持久化的读写函数 |
| `config/i18n.py` | 中英文翻译字典、`LanguageManager` 单例与翻译函数 `tr()` |
| `config/s80m_stereo_calibration.json` | S80M 双目标定参数副本（JSON 数据文件，非代码；由 `ui/main_window.py`（`_open_s80m`）拼接路径读取，供双目采集/显示使用） |
| `config/sensors/` | 手套仿生手掌映射配置（`hand_ble_config.json` 右手 / `hand_ble_config_left.json` 左手，各 16 部位；由 `core/render_engine.py` 常量指路，`core/sensor_hand_config.py` 加载） |

## 各文件详解

### config/__init__.py

**作用**：config 包的入口文件。仅一行真实代码 `__version__ = "1.0.3"`，是版本号的唯一定义点；`settings.py` 的 `APP_VERSION` 及其他模块均从这里取值（文件 docstring 明确说明"版本号集中定义于此"）。

**类/函数**：无。

**关键数据**：

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `__version__` | str | `"1.0.3"` | 版本号唯一来源 |

**调用关系**：被 `config/settings.py` import（`APP_VERSION`）；间接被所有 `from config import settings` 的模块触发。

### config/settings.py

**作用**：应用程序全局配置中心。集中定义路径、窗口尺寸、相机采集、录制、双目、D435/RealSense、设备检测、上传、任务服务、手部关键点与暗色主题颜色等全部常量；同时提供 `server_config.json`、`device_names.json`、`device_params.json` 三类配置文件的读写函数（均为 merge-write，保留未传入的旧字段）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_camera_slot_name` | `(index: int) -> str` | OpenCV 设备索引 → EgoData 槽位名 | 0→`head_left_rgb`，1→`head_right_rgb`，2+→`head_right_rgb_N` |
| `_camera_slot_name_legacy` | `(index: int) -> str` | 旧 LeRobot v3 会话命名兼容 | 0→`head_rgb`，其余→`head_rgb_N` |
| `realsense_profile` | `(model_name: str) -> dict` | 按型号名（子串匹配）查 `REALSENSE_PROFILES` | 返回该型号配置；未收录回落 D435 默认配置 |
| `_load_server_config` / `_save_server_config` | `() -> dict` / `(data: dict)` | 读 / merge-write `server_config.json` | 异常或不存在返回 `{}`；写盘保留旧字段 |
| `load_server_url` / `save_server_url` | `() -> str` / `(url: str)` | 读 / 存用户服务器地址 | 无保存值或空串时返回出厂默认 `SERVER_URL` |
| `load_credentials` / `save_credentials` | `() -> tuple[str, str]` / `(username, password)` | 读 / 存登录凭据 | 返回 `(username, password)` |
| `load_upload_auto_sync` / `save_upload_auto_sync` | `() -> bool` / `(on: bool)` | 读 / 存"录制完成后自动上传"开关 | 缺省默认 `True`（沿用历史行为） |
| `load_upload_delete_after` / `save_upload_delete_after` | `() -> bool` / `(on: bool)` | 读 / 存"上传成功后自动删除本地文件"开关 | 缺省默认 `False` |
| `load_device_names` / `save_device_name` / `remove_device_name` | `() -> dict` / `(key, name, sensor="")` / `(key)` | 设备命名表读 / merge-write / 删除 | `sensor` 仅 `data_ble` 用，绑定 parquet 列名；`""` 表示不动原值 |
| `device_name` / `device_sensor_role` | `(key: str) -> str` | 读单个设备名 / 读绑定列名 | 兼容 `{"name": …}` 结构与旧版纯字符串；无则空串 |
| `assign_glove_sensor_role` | `(key: str, prefer: str = "") -> str` | 为首次连接的手套分配空闲 parquet 列名并按 MAC 键持久化 | 返回 `left_glove`/`right_glove` 等；优先占 `prefer`，否则按 `SENSOR_NAMES` 顺序取空余名，无空余名兜底最后一个 |
| `load_device_params` / `save_device_exposure` | `() -> dict` / `(key, auto, value)` | 每设备参数表读 / 写曝光 | merge-write 保留其他条目与参数 |
| `device_exposure` | `(key: str) -> dict` | 读单设备曝光 | `{"auto": bool, "value": float}`，无则 `None` |
| `ensure_device_original` | `(key, auto, value) -> bool` | 记录设备"最一开始"的曝光基线（首次才写，之后永不覆盖） | 返回 `True` 表示本次为新写入 |
| `device_original` | `(key: str) -> dict` | 读原始曝光基线 | 结构同 `device_exposure`，无则 `None` |

**关键数据**（均为模块级常量，代码中真实存在）：

| 分组 | 常量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 路径 | `BASE_DIR` / `DATA_DIR` / `RECORDINGS_DIR` / `DB_PATH` / `KEYPOINTS_OUTPUT_DIR` | str | 仓库根 / `data` / `data/recordings` / `data/pipeline.db` / 根下 `keypoints_output` | 由 `os.path` 拼接 |
| 应用 | `APP_NAME` | str | `"EGO数据管线"` | 窗口标题等 |
| 应用 | `APP_VERSION` | str | 来自 `config/__init__.py` | 版本号唯一来源在此 |
| 应用 | `WINDOW_WIDTH` / `WINDOW_HEIGHT` | int | 1600 / 900 | 主窗口尺寸 |
| 相机 | `DEFAULT_FPS` / `DEFAULT_RESOLUTION` / `UVC_FOURCC` | int / tuple / str | 30 / `(1280, 960)` / `"MJPG"` | UVC 满分辨率须配合 MJPG 才能 30fps |
| 相机 | `CAMERA_RECONNECT_INTERVAL_MS` / `MAX_CAMERAS` | int | 2000 / 8 | 断线重试间隔 / 最大相机数 |
| 录制 | `RECORDING_FPS` / `RECORDING_DIR` / `DEPTH_ENABLED` | int / str / bool | 30 / `RECORDINGS_DIR` / `False` | `DEPTH_ENABLED` 为遗留开关（旧 S80M 视差录制路径，已被 `S80M_DEPTH_ENABLED` 取代，保持 `False` 勿启用） |
| 设备命名 | `CAMERA_LEFT` / `CAMERA_RIGHT` / `CAMERA_DEPTH` | str | `"head_left_rgb"` / `"head_right_rgb"` / `"head_depth"` | EgoData 命名约定 `<位置>_<模态>` |
| 设备命名 | `CAMERA_PRIMARY_LEGACY` | str | `"head_rgb"` | 旧 LeRobot v3 会话兼容 |
| EgoData 输出 | `EPISODE_PREFIX` / `EPISODE_DIGITS` / `DEPTH_FORMAT` / `DEPTH_SCALE` | str / int / str / float | `"episode"` / 6 / `"png16"` / 0.001 | 深度存 16-bit PNG（像素值 × `DEPTH_SCALE` = 米；v1.0.11 曾试 raw16 裸 bin，体积过大/上传失败后 v1.0.12 回退） |
| 双目标定 | `STEREO_BASELINE` / `STEREO_RESOLUTION` / `STEREO_FPS` | float / tuple / int | 0.095 / `(1280, 800)` / 30 | 基线（米）等 |
| 双目 | `STEREO_CAM_FPS` / `STEREO_RECORD_FPS` / `STEREO_RECORD_MIN_INTERVAL_S` | int / int / float | 50 / 30 / 1.0/30 | S80M 相机档 50fps（50→30 桶抽帧才有真 30fps；回调取帧=官方 GUI 同款组合，8/31 撕裂已根治；25 档保留为回退）、录制按 1/30s 桶抽帧 |
| 双目 | `STEREO_CB_BRIDGE` | bool | True | 取帧用官方 GUI 同款回调（SDK 装配完成才交付、帧必完整；shim 缺失/注册失败自动回退轮询） |
| S80C 深度 | `S80M_DEPTH_ENABLED` | bool | `True` | S80C 深度开关（关闭→子进程不加 `--depth-sdk-dir`，管道协议字节不变、零 CPU 开销） |
| S80C 深度 | `S80M_DEPTH_SLOT` | str | `"stereo_depth"` | 深度槽位（录制为 `depth/stereo_depth/…`） |
| S80C 深度 | `S80M_DEPTH_NEAR_MM` / `S80M_DEPTH_FAR_MM` / `S80M_DEPTH_SMOOTH_K` | int / int / int | 300 / 3000 / 3 | 热力图固定色标下限 / 上限（毫米）/ 中值滤波核 |
| D435 | `D435_SLOT_RGB` / `D435_SLOT_DEPTH` | str | `"d435_rgb"` / `"d435_depth"` | 录制槽位名 |
| D435 | `D435_RESOLUTION` / `D435_RGB_RESOLUTION` / `D435_FPS` | tuple / tuple / int | `(848, 480)` / `(1280, 720)` / 30 | 深度 / RGB 流配置 |
| D435 | `D435_DEPTH_NEAR_MM` / `D435_DEPTH_FAR_MM` | int | 300 / 4000 | 实时显示深度范围过滤（毫米） |
| D435 | `D435_PNG_COMPRESSION` / `D435_STALL_TIMEOUT_S` | int | 1 / 5 | [DEPRECATED v1.0.11] 深度已改存原始 uint16 二进制，压缩级不再生效 / 流停滞看门狗阈值（秒） |
| D435 | `D435_LOW_FPS_WINDOW_S` / `D435_LOW_FPS_FRACTION` | int / float | 10 / 0.5 | 帧率看门狗窗口与下限比例 |
| RealSense | `REALSENSE_PROFILES` | dict | 见下 | 按型号给采集配置 |
| RealSense | `REALSENSE_PROFILES["D405"]` | dict | `depth_resolution`/`rgb_resolution` `(1280,720)`、`fps` 30、`depth_near_mm` 100、`depth_far_mm` 1000、`heatmap_smooth_k` 3、`heatmap_temporal_alpha` 0.5 | D405 短距近景配置 |
| 设备检测 | `DEVICE_SCAN_MAX_INDEX` / `DEVICE_POLL_INTERVAL_MS` | int | 16 / 2000 | V4L2 枚举上限 / 轮询间隔（毫秒） |
| 设备检测 | `REALSENSE_VID` / `REALSENSE_PID` | str | `"8086"` / `"0b07"` | Intel vendor ID / D435 product ID（其他型号靠驱动 name 兜底） |
| 传感器 | `SENSOR_NAMES` | list | `["right_glove", "left_glove"]` | 写入 parquet observation 列名 |
| 手部 | `HAND_POSE_LEFT` / `HAND_POSE_RIGHT` / `HAND_POSE_DIM` / `SENSOR_DIM` | str / str / int / int | `"left_hand_pose"` / `"right_hand_pose"` / 63（21×3）/ 256 | 手部 63 维、触觉 16×16 展平 |
| 显示 | `CAMERA_MIRROR_HORIZONTAL` / `DISPLAY_FPS_LIMIT` / `FEED_MIN_WIDTH` / `FEED_MIN_HEIGHT` | bool / int / int / int | `True` / 30 / 320 / 240 | 镜像翻转（显示+录制）、GUI 帧率上限、最小画面 |
| 颜色 | `COLOR_RECORDING` / `COLOR_STOPPED` / `COLOR_ABNORMAL` | str | `"#EF5350"` / `"#66BB6A"` / `"#FFA726"` | 红=录制中 / 绿=空闲 / 橙=异常 |
| 颜色 | `COLOR_BG_MAIN` / `COLOR_BG_PANEL` / `COLOR_BG_WIDGET` / `COLOR_BG_CANVAS` | str | `"#1E1E1E"` / `"#252525"` / `"#2D2D2D"` / `"#1A1A1A"` | 暗色主题背景四层 |
| 颜色 | `COLOR_TEXT_PRIMARY` / `COLOR_TEXT_SECONDARY` / `COLOR_TEXT_HINT` | str | `"#E0E0E0"` / `"#9E9E9E"` / `"#616161"` | 文字三档 |
| 颜色 | `COLOR_BORDER` / `COLOR_BORDER_STRONG` | str | `"#424242"` / `"#616161"` | 边框与分割线 |
| 颜色 | `COLOR_BTN_START` / `COLOR_BTN_STOP` / `COLOR_BTN_ABORT` | str | `"#43A047"` / `"#E53935"` / `"#FF8F00"` | 开始 / 停止 / 异常停止按钮 |
| 颜色 | `COLOR_BTN_DEFAULT_BG` / `COLOR_BTN_HOVER` / `COLOR_BTN_DISABLED_BG` / `COLOR_BTN_DISABLED_TEXT` | str | `"#424242"` / `"#555555"` / `"#333333"` / `"#616161"` | 按钮常规四态 |
| 上传 | `SERVER_URL` | str | `"http://127.0.0.1:8000"` | 出厂默认占位地址；实际地址由用户在页面填写并持久化到 `server_config.json` |
| 上传 | `SERVER_CONFIG_FILE` / `DEVICE_NAMES_FILE` / `DEVICE_PARAMS_FILE` | str | `data/server_config.json` / `data/device_names.json` / `data/device_params.json` | 三个持久化文件路径 |
| 上传 | `UPLOAD_ENABLED` / `UPLOAD_MAX_CONCURRENT` / `UPLOAD_RETRY_MAX` | bool / int / int | `True` / 1 / 3 | 串行=1（并发时预压缩临时文件会互相覆盖致视频损坏） |
| 上传 | `UPLOAD_AUTO_SYNC` / `UPLOAD_DELETE_AFTER` | bool | 启动时从 `server_config.json` 读取（默认 `True` / `False`） | 录制后自动上传 / 上传后删本地 |
| 上传 | `UPLOAD_PRECOMPRESS_VIDEO` / `UPLOAD_VIDEO_CRF` | bool / int | `True` / 30 | 上传前视频重编码低码率 / CRF 档（v1.0.9 起 HEVC 录制件自动跳过） |
| 录制编码 | `RECORD_VIDEO_ENCODER` | str | `"auto"` | 录制编码器：`"auto"` 自动探测（nvenc→x265→x264）/ `"nvenc"` / `"x265"` / `"x264"` 显式指定 |
| 录制编码 | `RECORD_VIDEO_CRF` / `RECORD_VIDEO_X264_CRF` | int / int | 30 / 23 | HEVC 直出 CRF 档 / x264 回退 CRF 档（与 v1.0.8 现状一致） |
| 录制编码 | `ENCODER_PROBE_ENABLED` / `ENCODER_PROBE_FRAME_COUNT` / `ENCODER_PROBE_MAX_STREAMS` / `ENCODER_PROBE_TIMEOUT_S` | bool / int / int / int | `True` / 45 / 4 / 15 | 录前编码器速度探针开关 / 每探针合成帧数 / 并行流上限 / 单进程超时秒 |
| 录制编码 | `ENCODER_X265_MIN_FPS_RATIO` | float | 1.5 | x265 达标门槛 = 录制帧率 × 该比值（不达标回退 x264） |
| 录制编码 | `IMU_PENDING_MAX_SAMPLES` | int | 18000 | 双目 IMU 防丢缓冲上限（约 1 分钟 @300Hz；队列满时丢帧保 IMU） |
| 录制编码 | `DROP_WARN_RATIO` / `DROP_WARN_MIN_COUNT` | float / int | 0.01 / 30 | 录制结束丢帧告警阈值（占比 1% 或 30 帧） |
| 任务服务 | `TASK_POLL_INTERVAL_MS` / `TASK_API_URL` / `DEVICE_NAME` | int / str / str | 30000 / `SERVER_URL` / `"EGO_001"` | 轮询间隔 / API 地址 / 设备认领名 |
| 手部追踪 | `HAND_TRACK_ENABLED` / `HAND_TRACK_MODE` | bool / str | `False` / `"glove"` | 需 ultralytics；`"glove"` 黑色手套 / `"bare"` 裸手 |
| 手部追踪 | `HAND_DETECTION_DIR` / `HAND_DET_MODEL` / `HAND_MEDIAPIPE_MODEL` | str | 根下 `tools/hand_detection` / 其内 `best.pt` / `hand_landmarker.task` | YOLO 手套模型 / MediaPipe 裸手模型 |
| 手部追踪 | `HAND_DET_DEVICE` / `HAND_POSE_DEVICE` / `HAND_TRACK_MAX_HANDS` / `HAND_DATA_DIM` | str / str / int / int | `"cuda"` / `"cuda"` / 2 / 21×2×2+4×2+1（=93） | 设备、最多手数、展平维度 |

**JSON 字段约定**（由本模块的读写函数定义）：

- `data/server_config.json`：`server_url`（str）、`username`（str）、`password`（str）、`upload_auto_sync`（bool，缺省 `True`）、`upload_delete_after`（bool，缺省 `False`）。
- `data/device_names.json`：key = `DeviceInfo.stable_key`（形如 `"uvc:{by-id前缀}"` / `"d435:{serial}"` / `"ble:{MAC}"` 等），value = `{"name": str, "sensor"?: str}`（`sensor` 仅 `data_ble` 设备用，绑定 parquet 列名 `right_glove`/`left_glove`，按 MAC 持久化）；旧版纯字符串自动升级为 `{"name": …}`。
- `data/device_params.json`：key 同 `stable_key`，value = `{"exposure": {"auto": bool, "value": float}, "original": {同结构}}`；`auto=True` 时 `value` 忽略；`original` 首次看到才写入、之后永不覆盖，供"恢复默认"回到开机原厂曝光。

**调用关系**：被 `core/`（`pipeline.py`、`camera.py`、`hand_tracking.py`、`stereo_depth.py`、`device_detector.py`、`d435_camera.py`、`egodata_writer.py`、`task_record.py`、`render_engine.py`、`helpers.py`（`as _settings`）等）、`ui/`（`main_window.py`、`camera_widget.py`、`device_panel.py`、`playback_dialog.py`、`upload_dialog.py`、`task_page.py`）以及 `tools/tests/` 多个用例 import。自身仅依赖 `os` 与 `config.__version__`。

### config/i18n.py

**作用**：国际化模块，支持中英文界面切换。维护一张以中文原文为键的翻译字典 `_TRANSLATIONS`，通过模块级单例 `lang_manager` 记录当前语言，切换时发射 Qt 信号通知所有监听者刷新界面文字；`tr()` 负责查表翻译。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `LanguageManager` | `(QObject)` | 管理当前语言，切换时发信号 | 信号 `language_changed(str)`，参数 `"zh"` 或 `"en"` |
| `LanguageManager.current` | property | 当前语言 | 默认 `"en"` |
| `LanguageManager.set_language` | `(lang: str)` | 切换语言并通知监听者 | 非法值或相同语言时不动作 |
| `LanguageManager.toggle` | `()` | 中英文之间切换 | — |
| `lang_manager` | 全局单例 | 全应用共享的语言管理器 | 默认英文 |
| `tr` | `(text: str, *fmt_args) -> str` | 翻译函数 | 未收录条目原样返回；有 `fmt_args` 时 `str.format` 填充 |

**关键数据**：

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `_TRANSLATIONS` | dict | 键为中文原文，值为 `{"en": …, "zh": …}`；覆盖录制记录、菜单栏、工具栏、相机控件、状态栏、日志消息、双目 S80M、相机模式切换、RealSense、错误消息、对话框、面板标题、关于、空状态、传感器面板、设备检测、设备统一接入、每设备曝光、状态显示、回放、上传、上传开关、数据查看器、任务选择页面、手部关键点、自动标注等全部界面文案 |

**调用关系**：被 `ui/main_window.py`、`ui/camera_widget.py`、`ui/device_panel.py`、`ui/camera_grid.py`、`ui/glove_widget.py`、`ui/playback_dialog.py`、`ui/upload_dialog.py`、`ui/exposure_dialog.py`、`ui/task_page.py`、`tools/tests/device_panel_gui_smoke_test.py`、`tools/tests/glove_widget_test.py`、`tools/tests/exposure_control_test.py`、`tools/tests/test_playback_multifps.py` 等 import。依赖 `PyQt5.QtCore`（`QObject`、`pyqtSignal`）。
