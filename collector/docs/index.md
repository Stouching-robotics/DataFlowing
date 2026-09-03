# collector 仓库总览

collector 是一个基于 PyQt5 的数据采集 GUI 系统：多相机（UVC 摄像头、Intel
RealSense D435/D405、S80M 双目）与 BLE 数据手套的同步录制、回放与 HTTP 上传；
配套手部 3D 关键点处理链路（`tools/hand_detection/`、`tools/hand_3d_d435/`、
`tools/stereo_s80m/`）与离线 SLAM 数据集导出。采集输出为 EgoData / LeRobot v3
兼容格式，供机器人遥操作数据集使用。

## 快速开始

### 主程序

仓库自带虚拟环境 `venv/`（含 PyQt5、OpenCV、pyarrow、pyrealsense2 等），
依赖清单见 `requirements.txt`。

```bash
./run.sh                       # 等价于 venv/bin/python main.py
venv/bin/python main.py        # 直接启动
```

Windows 下用 `run.bat`（自动设置 Qt 平台插件路径后运行 `main.py`）。
若本机没有 `venv/`，先创建：

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
```

说明：`pyrealsense2`、`opencv-python`、`torch` 等未列入 `requirements.txt`
（按需在 venv 内安装；`main.py` 对 torch 做可选导入，缺失时自动跳过）。

首次运行无需手动建配置：`data/tasks.json` 在首次读取时由程序写入内置种子
任务；`data/server_config.json`、`data/device_names.json`、
`data/device_params.json` 在首次保存对应设置时自动创建（模板文件仅作结构
参考，不会自动复制为真实配置）。`data/pipeline.db` 在首次启动时建表。

### 主程序界面

`main.py` 启动的 `MainWindow`（`ui/main_window.py`）提供：

- 可拖拽调整大小的多画面网格布局（回放界面同样支持视频拖拽调位、分割条调大小）；
- 每路摄像机独立的录制控制——正常停止保存，异常停止丢弃（`abort_episode` 直接删除会话目录）；
- 录制历史记录追踪（SQLite）与回放（`ui/playback_dialog.py`）；
- 设备检测面板（统一枚举 UVC/D435/S80M 设备，2 秒轮询）、任务页、上传页；
- 中英文界面切换（`config/i18n.py`，`tr()` 翻译 + `lang_manager`）。

### 手部 3D 关键点（D435）

```bash
./tools/hand_3d_d435/run_live_d435.sh                    # 实时 demo（直连相机）
./tools/hand_3d_d435/run_live_d435.sh --replay <会话目录>  # 回放会话
./tools/hand_3d_d435/run_d435.sh <会话目录>                # 离线管线
```

两个 launcher 均通过 `VENV_PY` 环境变量选择解释器（默认仓库 `venv/bin/python`）。

### Demos

```bash
./tools/demos/run_stereo_depth_demo.sh        # S80M 深度引擎 demo（需 SDK，见下）
venv/bin/python tools/demos/test_stereo_depth_calib.py   # 双目深度标定自检（需 SDK）
venv/bin/python tools/hand_detection/demo_stereo_hands.py  # 双目 + MediaPipe 手部 demo
```

`tools/demos/stereo_2d_demo/` 是自带 README/requirements/run.sh 的独立小包；
`tools/demos/` 下另有 `demo_glove_kpts/`、`hdf5_demo_v1.4/`、`HSV_Visualizer_V4.4.1.py`。

### 测试

```bash
venv/bin/python tools/tests/test_device_detector.py   # 设备枚举
venv/bin/python tools/tests/test_depth_heatmap.py     # 深度热力图
# 真机相关测试需连接对应设备：d435_*、s80m_*、device_panel_gui_smoke_test 等
```

`tools/tests/` 内 `d405_worker_test.py`、`d435_e2e_test.py`、`s80m_50fps_decimation_test.py`
等覆盖 D405/D435 采集与 S80M 抽帧逻辑，`mono_regression.py`、
`s80m_signal_regression.py` 为回归测试。

## 环境变量约定

### `FAYSSENSE_SDK_DIR`（仅独立 S80M 工具必需）

FaysSense VI Kit SDK 的安装路径（Release 目录）。以下工具依赖它，
**未设置时直接报错退出**：

- `tools/demos/run_stereo_depth_demo.sh`（也接受 `SDK_DIR`；且要求 SDK 内已构建
  `thirdparty/opencv-4.2.0-linux-x86_64/lib406`）
- `tools/demos/stereo_depth_demo/run.sh`、`tools/demos/stereo_depth_demo/build.sh`
- `tools/demos/test_stereo_depth_calib.py`
- `tools/stereo_s80m/capture_calibration.py`
- `tools/diag_color.py`、`tools/diag_frame_layout.py`

示例：`export FAYSSENSE_SDK_DIR=/path/to/VIKitRelease`

主程序 S80C/S80M 采集链路不使用此变量——仓库自带 SDK 副本与全部依赖
（见下方硬件支持表），git 克隆后无需安装 SDK 即可运行。

### `FFMPEG_BIN`（可选）

ffmpeg 可执行文件覆盖。默认兜底
`~/miniconda3/envs/lerobot/bin/ffmpeg`（`~` 自动展开），候选列表逐个自检、
第一个可用者胜出。使用处：

- `tools/stereo_s80m/render_stereo.py`：候选顺序 `which ffmpeg` → `/usr/bin/ffmpeg`
  → `$FFMPEG_BIN`（未设则 lerobot 路径）
- `tools/stereo_s80m/hand_3d/video_writer.py`：`$FFMPEG_BIN`（未设则 lerobot 路径）
  优先，再 `which ffmpeg` / `/usr/bin/ffmpeg` 兜底
- `tools/hand_detection/demo_stereo_hands.py`：与 `render_stereo.py` 相同候选顺序

注意：录制器 `core/egodata_writer.py` 不走 `FFMPEG_BIN`——它优先使用
`imageio_ffmpeg` 自带的静态 ffmpeg，其次取 PATH 中通过 `-version` 自检的
`ffmpeg`（conda 版曾因 openvino/tbb 符号错误启动即崩，自检不通过即弃用）。

### `VENV_PY`（可选）

launcher 脚本使用的 Python 解释器，默认 `$REPO_ROOT/venv/bin/python`：

```bash
VENV_PY=/path/to/python ./tools/hand_3d_d435/run_live_d435.sh
```

使用处：`tools/hand_3d_d435/run_live_d435.sh`、`tools/hand_3d_d435/run_d435.sh`。

## 目录总览

| 目录 | 作用 | 文档 |
|---|---|---|
| `config/` | 全局配置：路径/相机/录制/上传常量（`settings.py`）、中英文文案（`i18n.py`）、S80M 标定副本与 `config/sensors/` 手套仿生手掌映射 | `docs/config.md` |
| `core/` | ★ SDK 核心（平铺，30 个模块）：采集管线（`pipeline.py`/`camera.py`/`d435_camera.py`/`device_detector.py`）、设备管理器（`device_manager.py`/`s80m_manager.py`/`d435_manager.py`/`device_naming.py`/`exposure_controller.py`）、录制落盘（`egodata_writer.py`/`calibration.py`/`database.py`/`recording_record.py`/`recording_repository.py`/`task_record.py`）、上传后端（`api_client.py`/`task_service.py`/`uploader.py`）、回放会话（`session_timeline.py`/`session_catalog.py`/`session_loader.py`）、手部（`hand_tracking.py`/`hand_processor.py`/`auto_labeler.py`）、手套传感（`ble_engine.py`/`render_engine.py`/`sensor_config_dialogs.py`/`sensor_hand_config.py`）、通用（`helpers.py`/`stereo_depth.py`） | `docs/core.md` |
| `data/` | 本地配置（仓库提供 `*.example.json` 模板与出厂默认 `device_params.json`；其余配置、录制数据、SQLite 均 gitignore） | `docs/data.md`（本文档集） |
| `tools/demos/` | 交付版 demo：S80M 深度引擎 demo、双目 2D demo、手套关键点 demo、hdf5 演示、标定自检脚本 | `docs/demos.md` |
| `tools/hand_detection/` | 手部检测：YOLO 手套检测（`best.pt`、`world_detector.py`）、MediaPipe 裸手管线、`demo_stereo_hands.py` | `docs/hand_detection.md` |
| `scripts/` | 录制后离线处理：手部关键点提取/标注/可视化导出（`process_hands.py`） | `docs/scripts.md` |
| `tools/stereo_s80m/` | S80M 双目工具：`read_stereo_rgb.py`、`stereo_triangulate.py`、`render_stereo.py`、`hand_3d/`、离线 SLAM 导出（`export_offline_slam_dataset.py`、`validate_offline_dataset.py`） | `docs/stereo_s80m.md` |
| `tools/` | 诊断与手部 3D 工具：`diag_*.py`、`hand_3d_d435/`（独立 D435 3D 手部管线模块）、`hand_3d_s80c/`（S80C 双目实时裸手/手套关键点 demo，含自包含 SDK）、`fayssense_depth_sdk/`（FaysSense VI Kit 深度引擎 SDK，专有）、`glove_package/`、`tests/`（回归/冒烟测试）、`weights/`（大权重）、`models/`（模型权重） | `docs/tools.md` |
| `ui/` | PyQt5 界面：`main_window.py`、`camera_grid.py`、`playback_dialog.py`（回放）、`upload_dialog.py`、`task_page.py`、`device_panel.py`（设备检测面板）等 | `docs/ui.md` |
| `tools/models/` | 模型权重：`hand_landmarker.task`（MediaPipe 裸手关键点，供 `tools/hand_detection/demo_stereo_hands.py` 等使用） | — |
| `keypoints_output/` | 录制后手部关键点输出（镜像录制目录结构，gitignore） | — |
| `dist/` | 自包含 demo 发布包（gitignore） | — |
| `venv/`、`tools/weights/` | 虚拟环境、CLIP 等大权重（均 gitignore） | — |
| `docs/` | 本文档目录 | — |

### 根目录文件

| 文件 | 说明 |
|---|---|
| `main.py` | 主入口：qt-material `dark_teal` 暗色主题；先做 torch 可选导入（避免 DLL 冲突），并修复 cv2 覆盖 `QT_QPA_PLATFORM_PLUGIN_PATH` 导致的 Qt 平台插件加载问题，然后启动 `ui.main_window.MainWindow` |
| `start.bat` / `start.sh` | Windows / Linux 一键部署：无 Python 时自动安装、建 venv、装依赖并启动主程序；子命令 `reinstall` / `extras` / `extras-torch` / `help` |
| `.gitattributes` | 换行符规范化：`*.bat` 强制 CRLF，保证客户从 GitLab 下载后双击可用 |
| `requirements.txt` | 核心依赖：PyQt5、numpy（<2）、pyarrow、imageio-ffmpeg、bleak、qt-material、h5py、requests、pygrabber、comtypes、pyvista、pyvistaqt |
| `.gitignore` | 排除录制数据（`data/recordings/`、`data/*.db`）、真实配置文件、venv、keypoints_output、大权重等 |
| `run.sh` / `run.bat` | Linux / Windows 启动脚本（使用 venv 解释器运行 `main.py`） |
| `README.md` | 仓库说明（开发者向，中英双语合并版：英文在前、中文在后，顶部章节导航表） |
| `使用说明.md` / `使用说明_EN.md` | 客户操作指引与错误码 A–G 排查（中英两版；`start.bat help` 与启动弹窗引用） |
| `tools/gongsitubiao.png` | 界面 logo（`ui/main_window.py`、`ui/task_page.py` 引用） |

## 数据链路速览

```
录制 ──► core/pipeline.py（相机帧 + BLE 手套数据汇聚）
         └─► core/egodata_writer.py（EgoDataWriter）
              └─► data/recordings/<任务>/<任务>_NNNNNN/（MP4 + Parquet + 元数据）
                       │
                       ├─► 回放：ui/playback_dialog.py（兼容 EgoData / LeRobot v3 会话）
                       ├─► 上传：core/uploader.py → POST /api/v1/session/upload
                       │        （服务器地址见 data/server_config.json，可自动同步/删本地）
                       └─► 离线处理：
                            tools/stereo_s80m/（三角化 stereo_triangulate.py、
                              渲染 render_stereo.py、离线 SLAM 数据集导出）
                            tools/hand_3d_d435/（D435 RGB-D 3D 手部关键点）
                            tools/hand_detection/ + scripts/（手套/裸手关键点）
                            └─► 输出到 keypoints_output/<任务>/<会话>/（不写回录制目录）
```

- 任务进度：`data/tasks.json` 由 `core/task_record.py` 维护，进度按分账模型
  持久化（`local_count` 本机完成数 / `synced_count` 已确认水位 /
  `backend_count` 后端全局数，显示值 `completed_count = backend_count +
  (local_count - synced_count)`），与本地文件是否被删除无关；多电脑协同
  采集时经 `POST /api/v1/device/tasks/progress` 水位合并上报，后端聚合全局数。
- 录制历史与上传队列：`data/pipeline.db`（表 `recording`、`upload_task`）。
- 目录结构与各元数据字段详见 `docs/data.md`。

## 开发约定

- **版本号**：唯一定义在 `config/__init__.py`（`__version__`），各模块经
  `config.settings.APP_VERSION` 引用，录制元数据写入 `codebase_version`。
- **设备命名**：EgoData 标准 `<位置>_<模态>`，如 `head_left_rgb` /
  `head_right_rgb` / `head_depth`；触觉传感器 `right_glove` / `left_glove`；
  手部关键点 `right_hand_pose` / `left_hand_pose`（见 `config/settings.py`）。
  UVC 槽位按索引映射：0 → `head_left_rgb`、1 → `head_right_rgb`、
  2+ → `head_right_rgb_N`。
- **PyQt5 信号参数**：可能超过 32 位整数范围的参数（如纳秒时间戳）一律用
  `object` 类型声明（`pyqtSignal(object)`）——`int` 参数会按 qint32 封送，
  超过 2^31 静默翻负。
- **SQLite**：`core/database.py` 线程本地连接 + WAL 模式；
  上传后删除本地文件的录制行保留为 `uploaded_deleted`、上传成功但本地保留
  标为 `uploaded`，均供历史可查。
- **任务进度**：`completed_count` 是持久化权威值，不依赖扫描录制目录
  （上传后自动删本地也不会回退）；`hidden` 是删除墓碑，防后端推送复活。
- **i18n**：界面文案经 `config/i18n.py` 的 `tr()` 翻译，勿在 UI 代码中硬编码
  中英文混用文案。

## 硬件支持

| 设备 | 接入方式 | 说明 |
|---|---|---|
| UVC 相机 | `/dev/videoN`（OpenCV V4L2） | MJPG 像素格式（`settings.UVC_FOURCC`）；单目镜像显示；最多 8 路 |
| Intel RealSense D435 / D405 | `pyrealsense2` | RGB + depth 双路槽位（如 `d435_rgb` / `d435_depth`）；深度原始 uint16 PNG16（毫米）+ 热力图 MP4；内置停滞/帧率看门狗自动重连；D405 有独立采集配置（近距） |
| S80M 双目 | FaysSense VI Kit SDK（仓库自带，含 FT602 桥驱动） | 无需安装 SDK / 设置 `FAYSSENSE_SDK_DIR`；深度引擎依赖仓库自带 OpenCV 4.2（`lib406` 目录）；相机档 `STEREO_CAM_FPS`（默认 50fps）+ 回调取帧（官方 GUI 同款）按 1/30s 桶抽帧录制 30fps；携带硬件纳秒时间戳与 IMU 样本；v1.0.11 起子进程内置 SDK 深度引擎（`tools/hand_3d_s80c/third_party`，~20fps）→ 第三格实时深度热力图 + 录制 raw16/热力图 MP4（`S80M_DEPTH_ENABLED=False` 关闭） |
| BLE 数据手套 | `bleak`（core/ble_engine.py） | 左右手各一只；parquet 列名绑定（`right_glove` / `left_glove`）按设备 MAC 持久化在 `data/device_names.json` 中 |

## 隐私与本地配置

- 真实配置文件**不进仓库**：`data/server_config.json`、`data/device_names.json`、
  `data/tasks.json` 以及 `data/*.db`、`data/recordings/` 均被 `.gitignore`
  排除；仓库只提供模板 `server_config.example.json`、
  `device_names.example.json`、`tasks.example.json`（结构见 `docs/data.md`）。
- `server_config.json` 可能包含服务器地址与登录凭据，`device_names.json`
  的 key 含设备序列号/MAC——这些文件务必保持本地私有。
- `SERVER_URL` 出厂默认值为 `http://127.0.0.1:8000`（`config/settings.py`），
  用户可在界面填写实际服务器地址并持久化。
- `data/device_params.json`（曝光设置）出厂为空对象 `{}`，仓库内默认文件
  本身不含敏感信息；但其 key 与 `device_names.json` 同为设备稳定标识，
  本地增长后可能含序列号/MAC，修改过的文件请勿提交。

## 相关文档

各模块细节见 `docs/` 下的分文档：

- `docs/data.md` — `data/` 配置与数据存储结构（本文档集）
- `docs/config.md`、`docs/core.md` — 配置、SDK 核心（采集/设备管理器/录制落盘/上传/回放/手套传感）
- `docs/ui.md`、`docs/scripts.md` — 界面、离线脚本
- `docs/demos.md`、`docs/stereo_s80m.md`、`docs/tools.md`、`docs/hand_detection.md` — demo、S80M、工具与手部检测
