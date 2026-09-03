# data/ —— 配置文件与数据存储结构

本文档描述 `data/` 目录下的配置文件格式、SQLite 表结构与录制数据的目录布局。
所有结构均从代码（`config/settings.py`、`core/task_record.py`、
`core/egodata_writer.py`、`core/database.py`、`core/recording_record.py`、
`core/helpers.py`）与仓库内模板文件归纳得出。

## 1. 总览：仓库提供什么，本地生成什么

| 文件/目录 | 归属 | 说明 |
|---|---|---|
| `server_config.example.json` | 仓库模板 | 服务器连接配置模板；本地真实配置写在 `server_config.json`（gitignore） |
| `device_names.example.json` | 仓库模板 | 设备命名表模板（空 `{}`）；本地真实配置写在 `device_names.json`（gitignore） |
| `tasks.example.json` | 仓库模板 | 任务列表模板（`{"tasks": []}`）；本地真实数据写在 `tasks.json`（gitignore） |
| `device_params.json` | 仓库内默认文件 | 每设备相机参数（曝光），出厂为空对象 `{}`；key 用设备稳定标识，增长后可能含设备标识，勿提交本地改动 |
| `recordings/` | 本地生成（gitignore） | 录制会话输出根目录 |
| `app.db`、`pipeline.db` | 本地生成（gitignore） | SQLite 数据库；当前代码使用 `pipeline.db`（`settings.DB_PATH`），`app.db` 未被现有代码引用，属历史遗留 |

本地真实文件可能包含服务器地址、凭据、MAC 等敏感信息，**永不提交**；
新环境首次运行由程序按需创建，结构见下文。

## 2. 配置文件

### 2.1 `server_config.json`（模板 `server_config.example.json`）

服务器连接与上传行为配置。字段由 `config/settings.py` 中的读写函数定义：

| 键 | 类型 | 含义 | 默认/回退 |
|---|---|---|---|
| `server_url` | string | 服务器地址；为空时回退出厂默认 `http://127.0.0.1:8000`（`settings.SERVER_URL`） | 空 |
| `username` | string | 上传登录用户名 | 空 |
| `password` | string | 上传登录密码 | 空 |
| `upload_auto_sync` | bool | 录制完成后是否自动上传 | `true` |
| `upload_delete_after` | bool | 上传成功后是否自动删除本地文件 | `false` |

写入采用合并式（merge-write）：`_save_server_config` 只更新传入键，保留其余字段。

### 2.2 `device_names.json`（模板 `device_names.example.json`）

设备命名表。顶层为 JSON 对象：

```json
{
  "<DeviceInfo.stable_key>": { "name": "<用户命名>", "sensor": "<parquet 列名>" }
}
```

- **key**：设备稳定标识 `DeviceInfo.stable_key`，形式如 `uvc:{by-id 前缀}`、
  `d435:{serial}`、`ble:{MAC}` 等（含设备序列号/MAC，属本地敏感信息，不入库）。
- **value**：规范形式为 `{"name": str, "sensor"?: str}`；旧版本可能是纯字符串，
  读取时自动兼容升级。
  - `name`：用户在界面中给设备的命名（槽名随 GUI 用户命名）。
  - `sensor`：仅 BLE 数据手套使用，把设备绑定到 parquet 列名
    （`right_glove` / `left_glove`）。按 MAC 持久化，重连保持；首次连接的未知手套
    由 `assign_glove_sensor_role` 按广播名偏好（`L` → `left_glove`、
    `R` → `right_glove`）自动分配一个空闲列名并写回。

### 2.3 `tasks.json`（模板 `tasks.example.json`）

采集任务列表，是任务进度的事实来源（single source of truth）。顶层结构：

```json
{
  "tasks": [ { <任务条目> }, ... ],
  "updated_at": "<ISO 时间戳>"
}
```

任务条目字段（由 `core/task_record.py` 归纳）：

| 键 | 类型 | 含义 |
|---|---|---|
| `id` | string | 任务唯一 ID（读取时也兼容 `task_id` 键名） |
| `name` | string | 任务名；录制时输入的 `task_name` 按 `name` 精确匹配来累计进度，同时是录制目录名的来源 |
| `description` | string | 任务描述 |
| `total_required` | int | 要求完成的录制总次数（`>0` 才视为有效任务） |
| `assigned_at` | string | 任务分配时间（ISO 8601，用于列表排序，新→旧；空/非法排最后） |
| `params` | object | 任务自定义参数（如对象、用手等） |
| `completed_count` | int | 录制完成次数（持久化权威值，每次录制完成 +1，与本地文件是否被删除无关） |
| `status` | string | `pending` / `in_progress` / `completed`，每次加载时由 `completed_count` 与 `total_required` 重新推算 |
| `hidden` | bool | 删除墓碑：用户删除过的任务标 `true`，防止后端再次推送时“复活” |

行为要点（均来自代码）：

- 首次运行若文件缺失或解析失败，写入内置示例种子任务（`task_001` 起 3 条）后重读。
- `load_tasks` 除旧数据回填外不写回文件；后端推送经 `merge_backend_tasks`
  按 `id` 合并，本地保留 `total_required`/`params` 覆盖字段与
  `completed_count` 权威值；同名多可见任务只保留 `assigned_at` 最新的一条，
  其余标 `hidden`。
- 旧数据缺少 `completed_count` 时，一次性按 `data/recordings/<name>/` 下含
  `meta/info.json` 的会话目录数回填初值并写回，此后以持久化值为准。

### 2.4 `device_params.json`

每设备相机参数（当前为曝光设置）持久化。顶层为 JSON 对象，key 与
`device_names.json` 同为 `DeviceInfo.stable_key`：

```json
{
  "<stable_key>": {
    "exposure": { "auto": true, "value": 0.0 },
    "original": { "auto": true, "value": 0.0 }
  }
}
```

- `exposure`：当前曝光（`auto=true` 时 `value` 被忽略）。
- `original`：设备首次开启时读回的原厂曝光基线，只写一次、永不覆盖，
  供“恢复默认”按钮使用。

仓库内默认文件为空对象 `{}`，随使用增长。注意：key 与 `device_names.json` 同为
`DeviceInfo.stable_key`，增长后可能含设备序列号/MAC 等标识，本地修改过的文件请勿提交。

## 3. SQLite 数据库（`data/pipeline.db`）

`core/database.py` 定义建表 SQL（`CREATE TABLE IF NOT EXISTS`），
连接为线程本地单例，启用 `journal_mode=WAL` 与 `foreign_keys=ON`。

### 表 `recording` —— 录制历史记录

| 列 | 类型 | 含义 |
|---|---|---|
| `id` | TEXT PK | 记录 ID |
| `camera_index` | INTEGER | 触发录制的摄像机槽位索引 |
| `camera_name` | TEXT | 摄像机名 |
| `file_path` | TEXT | 会话目录路径 |
| `file_size_mb` | REAL | 文件大小（MB） |
| `duration_sec` | REAL | 录制时长（秒） |
| `resolution_w` / `resolution_h` | INTEGER | 分辨率 |
| `status` | TEXT | `completed` / `uploaded`（已上传、本地保留）/ `aborted` / `deleted` / `uploaded_deleted`（已上传、本地已删，行保留供历史可查） |
| `started_at` / `finished_at` | TEXT | 起止时间 |

索引：`idx_recording_camera(camera_index)`、`idx_recording_date(started_at)`。
读写封装在 `core/recording_repository.py`（`RecordingRepo`）、数据类在
`core/recording_record.py`（`RecordingRecord`）。

### 表 `upload_task` —— 上传任务记录

| 列 | 类型 | 含义 |
|---|---|---|
| `id` | TEXT PK | 上传任务 ID |
| `session_path` | TEXT | 会话目录路径 |
| `session_name` | TEXT | 会话名 |
| `status` | TEXT | `pending` / `uploading` / `completed` / `failed` / `skipped` |
| `progress` | REAL | 进度（0.0–1.0） |
| `retry_count` | INTEGER | 重试次数 |
| `server_url` | TEXT | 目标服务器 |
| `server_session_id` | TEXT | 服务器返回的会话 ID |
| `error_message` | TEXT | 错误信息 |
| `created_at` / `updated_at` | TEXT | 创建/更新时间 |

索引：`idx_upload_status(status)`、`idx_upload_session(session_path)`。

## 4. 录制目录结构（`data/recordings/`）

录制由 `core/pipeline.py` 驱动 `core/egodata_writer.py`（`EgoDataWriter`）
写入。目录布局为 EgoData 标准，并兼容 LeRobot v3 格式消费方。

### 4.1 目录树

```
data/recordings/                          # 录制根目录（settings.RECORDING_DIR）
└── <task_tag>/                           # 任务目录：清洗后的任务名（task_name 为空时叫 "session"）
    └── <task_tag>_000001/                # 会话（episode）目录：任务名 + 6 位序号
                                          #   旧格式兼容：episode_000001/
        ├── metadata.json                 # EgoData 根级元数据（见 4.3）
        ├── timestamps.json               # 逐帧时间戳（见 4.4）
        ├── videos/                       # RGB 视频（每台相机一路 MP4）
        │   └── <camera_name>/            #   槽名，如 head_left_rgb；_aux 后缀相机归入主相机目录
        │       └── chunk-0000/
        │           └── <camera_name>.mp4 #   编码自适应（v1.0.9：HEVC CRF30 直出，性能不足回退 H.264 CRF23；实际见 metadata.video_codec）
        ├── depth/                        # 深度（仅启用深度时创建）
        │   └── <深度槽名>/               #   如 head_depth（S80M 兜底）/ d435_depth（D435）/ stereo_depth（S80C）
        │       ├── <深度槽名>.mp4        #   深度热力图视频（JET 伪彩，可视化用途）
        │       └── 000001.png …          #   原始 uint16 毫米 PNG（16-bit grayscale，
        │                                 #   raw_depth 槽位；v1.0.11 曾改 raw16 bin，
        │                                 #   体积过大且上传失败，v1.0.12 回退 PNG16）
        ├── calibration/                  # 标定（StereoCalibration）
        │   ├── head_stereo.json          #   首台双目型设备标定（服务器/回放/三角化依赖此路径）
        │   └── <槽名前缀>_calibration.json  # 其余设备的标定（槽名前缀去掉 "_N" 消歧编号）
        ├── data/                         # 传感器数据（LeRobot v3 兼容 Parquet）
        │   ├── <sensor>/                 #   每传感器一个目录，如 right_glove / left_glove
        │   │   └── chunk-0000/
        │   │       └── chunk_000000.parquet   # zstd 压缩，schema 见 4.5
        │   └── imu/                      #   双目 IMU（仅双目会话，每帧一行）
        │       └── chunk-0000/
        │           └── chunk_000000.parquet
        └── meta/                         # LeRobot v3 兼容元数据
            ├── info.json                 # 上传服务器严格依赖的字段（见 4.6）
            ├── stats.json                # 各特征 mean/std/min/max（归一化统计）
            ├── tasks.jsonl               # {"task_index": 0, "task": "<任务名，空时 'default recording'>"}，单行 JSON
            └── episodes/
                ├── chunk-000/
                │   └── file-000.parquet  # 每会话一行的 episode 表
                └── chunk_000000.parquet  # 前者的兼容旧路径副本
```

要点：

- **序号**：episode 序号 = `max(任务进度 batch_index, 目录扫描最大序号 + 1)`；
  `batch_index` 是“录制完成次数 + 1”（`core/task_record.py` 持久化），
  上传后自动删除本地文件也不会使序号回退。
- **视频/深度/标定**：深度槽位（`depth_slots`）不建视频目录也不走 MP4 合成；
  `start_episode` 中 `_aux` 后缀摄像头归入主摄像头目录。
- **旧格式兼容**：会话目录也可能以 `<tag>_YYYYMMDD_HHMMSS`（`session_dirname`）
  命名且内含 `meta/info.json`（LeRobot v3 旧会话）；`core/helpers.py` 的
  `list_all_sessions` / `detect_session_format` 通过 `metadata.json`（egodata）
  或 `meta/info.json`（lerobot_v3）识别会话，回放支持两种格式。
- **中止录制**：`abort_episode` 直接 `rmtree` 丢弃整个会话目录。
- **关键点数据不写在本目录**：手部关键点输出镜像到仓库根的
  `keypoints_output/<task>/<session>/`（见第 5 节）。

### 4.2 深度通道说明

- **S80M 传统路径**（无显式注册深度槽）：单槽兜底 `head_depth`，
  由 `write_depth_frame` 惰性创建目录并合成热力图 MP4，受
  `settings.DEPTH_ENABLED` 门控（默认关闭，遗留开关勿启用）。
- **D435/D405**（`core/pipeline.py` 经 `set_depth_camera` 显式注册槽位）：
  打开即录深度；槽位配置了 `raw_depth=True` 时额外写原始 uint16 PNG16
  （`egodata_depth_path`：`depth/<槽名>/000001.png`，文件名从 1 起 6 位，
  压缩级 `settings.D435_PNG_COMPRESSION`）。
  热力图支持固定色标（`near_mm`/`far_mm`）、3×3 中值与 EMA 时域平滑
  （仅作用于可视化通道，原始 PNG16 不经过）。
- **S80C**（v1.0.11）：`set_depth_camera(settings.S80M_DEPTH_SLOT,
  raw_depth=True)` 显式注册槽位 `stereo_depth`，深度由 `read_stereo_rgb.py`
  子进程内 SDK 深度引擎计算（~20fps），经管道深度块 → `depth_ready` 信号
  回主程序；录制时热力图 MP4 + PNG16 与 D435 同口径落盘。深度源低于
  录制帧率时热力图 MP4 重复最近帧补拍（时长与 RGB 对齐），PNG16 仅
  记新帧不重复。
- **读取**：PNG16 直接 `cv2.imread(path, cv2.IMREAD_UNCHANGED)`；
  v1.0.11 窗口会话存的是 raw16 bin（`np.fromfile(dtype=np.uint16)
  .reshape(h, w)`，`(h, w)` 取 `metadata.json` `cameras.<槽名>.height/width`），
  读取方已做扩展名回退兼容（`depth_align.load_depth_frame`）。

### 4.3 `metadata.json` 字段概览（EgoData 根级元数据）

| 键 | 含义 |
|---|---|
| `format` / `format_version` | 固定 `"egodata"` / `"1.0"` |
| `episode_index` | episode 序号 |
| `fps` | 默认录制帧率 |
| `task_name` | 任务名（可为空） |
| `cameras` | `{槽名: {height, width, ...}}`；深度槽带 `type:"depth"`、`unit:"mm"`、`format:"png16"`（v1.0.11 窗口会话为 `"raw16"`）；RGB 槽带独立帧率 `fps`（与默认不同时）；挂设备归属时带 `device_key`/`device` |
| `devices` | 设备段数组：`{key, kind, name, slots, serial?, sensor_column?, resolution, fps, calibration}`；`serial` 仅在设备提供时写入 |
| `sensors` | 传感器名列表（如 `["right_glove", "left_glove"]`） |
| `sensor_dim` | 传感器维度（256 = 16×16 触觉矩阵展平） |
| `created_at` | 创建时间（Unix 时间） |
| `codebase_version` | 程序版本（`config.__version__`） |
| `video_codec` | v1.0.9：本会话视频编码信息 `{encoder, codec, crf, ffmpeg, selected_by, probe}`；`selected_by` = `"auto"` 或显式指定名，`probe` = 录前速度探针结果（x265 路径） |
| `drop_stats` | v1.0.9：录制丢帧统计 `{队列键: 丢弃帧数}` + `imu_overflow`（IMU 防丢缓冲溢出次数）；无丢帧时全为 0 |

### 4.4 `timestamps.json`

```json
{
  "timestamps": [ { "frame_index": 0, "timestamp": 0.0, "wall_time": 1.7e9 }, ... ],
  "total_frames": 1234
}
```

- `timestamp`：会话相对时间（秒，与 parquet 的 `timestamp` 列同源）；`wall_time`：写入时刻的墙上时间。
- `hardware_ns`：仅双目相机帧携带（SDK 硬件纳秒时钟，与 IMU 同源）；含硬件
  时间戳的行在写出时按 `hardware_ns` 稳定排序，保证时间线单调。

### 4.5 Parquet schema（`data/` 与 `meta/episodes/`）

- `data/<sensor>/chunk-0000/chunk_000000.parquet`（每帧一行，zstd 压缩）：

  | 列 | 类型 | 含义 |
  |---|---|---|
  | `episode_index` | int64 | episode 序号 |
  | `frame_index` | int64 | 帧序号 |
  | `timestamp` | float32 | 会话时间（秒） |
  | `task_index` | int64 | 任务序号（当前固定 0） |
  | `observation.<sensor>` | list<float32, 256> | 传感器读数（16×16 展平；缺帧补零） |
  | `observation.left_hand_pose` / `observation.right_hand_pose` | list<float32, 63> | 手部关键点（21 关节 × xyz；录制时占位为零，后处理回填） |
  | `action` | list<float32, 1> | 动作（当前固定 `[0.0]`） |
  | `status.<device_id>` | string | 该设备在本帧的连接状态（默认 `"connected"`） |

- `data/imu/chunk-0000/chunk_000000.parquet`（双目 IMU，每帧一行）：

  | 列 | 类型 | 含义 |
  |---|---|---|
  | `episode_index` / `frame_index` / `timestamp` / `task_index` | 同上 | 同上 |
  | `hardware_ns` | int64 | 帧的 SDK 硬件纳秒时间戳 |
  | `imu_ts_ns` | list<int64> | 本帧窗口内 IMU 样本时间戳，与样本一一对应 |
  | `observation.imu` | list<list<float32, 6>> | 样本序列，每样本 `[gx, gy, gz, ax, ay, az]` |

- `meta/episodes/chunk-000/file-000.parquet`（每会话一行）：
  `episode_index`、`task_index`、`start_frame_index`、`end_frame_index`、
  `length`（均为 int64）。

### 4.6 `meta/info.json` 字段概览（LeRobot v3 兼容，上传服务器严格依赖）

| 键 | 含义 |
|---|---|
| `codebase_version` | 固定字符串 `"v3.0"` |
| `fps` | 默认录制帧率 |
| `video` | bool：是否有视频 |
| `task_name` | 任务名（空时 `""`） |
| `features` | `{observation.<sensor>: {dtype:"float32", shape:[16,16]}, observation.imu: {dtype:"float32", shape:[6]}, action: {dtype:"float32", shape:[1]}}`；shape 必须是 2D `[16,16]` 而非 1D `[256]` |
| `cameras` | dict 格式 `{槽名: {height, width, fps?}}`（深度槽不在此列） |
| `devices` | 紧凑设备段：`[{key, kind, name, slots}]`（无 serial 等敏感字段） |
| `device_names` | 槽位 → 用户命名映射 |
| `sensors` | 传感器名列表 |
| `sensor_dim` | 传感器维度 |
| `created_at` | 创建时间（Unix 时间） |

`stats.json` 对应 `features` 各键给出 `mean`/`std`/`min`/`max`（IMU 为 6 轴
样本级统计，`action` 为占位值），供归一化使用。

## 5. 相邻输出目录 `keypoints_output/`（不在 `data/` 下）

录制后的手部关键点处理结果**不写回** `data/recordings/`，而是镜像输出到
仓库根的 `keypoints_output/`（`settings.KEYPOINTS_OUTPUT_DIR`，gitignore）：

```
keypoints_output/<task>/<session>/
├── videos/                             # 关键点可视化视频
├── hand_pose/chunk-000.parquet         # 2D 手部关键点
├── hand_pose_3d/chunk-000.parquet      # 3D 手部关键点
└── auto_labels/auto_labels.parquet     # 自动标注
```

读取时有三级回退路径（`core/helpers.py`）：先查 `keypoints_output/` 镜像，
再查会话目录内 `keypoints/`，最后查旧版 `annotations/`（含 `annotations/mmpose/`）。
