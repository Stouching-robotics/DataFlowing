# 数据文件架构接口契约（v1.1.2）

> 本文是录制数据落盘格式的**规范性契约**，供一切读写该数据的程序严格对齐：
> 服务器导入器、离线处理工具、第三方数据读取器。凡本文标注 **MUST** 的条目，
> 写入方必须产出、读取方必须容忍；标注 SHOULD 为建议。歧义以 `core/helpers.py`
> 与 `core/egodata_writer.py` 的实现为准。

v1.1.0 起录制数据从「每会话一个子目录」改为**任务级池化布局**（LeRobot v3
命名）。v1.1.1 起 `stats.json` 自含累加器（每块加 `count`），`.stats_state.json`
边车已废除。**v1.1.2 起每段文件名前缀 `file-` 改为 `episode-`**（编号不变：
`file-011` → `episode-011`），应用内删除改为直接彻底删除（不再产生 `_trash/`）；
**v1.1.2 起深度存储改单流 12-bit 灰度 HEVC MP4**（gray12le 对数深度码；
x265 无 12-bit 灰度能力时回落 FFV1 gray16le MKV；旧双流 MKV / PNG16 仅
读侧容忍，见 §8）。
旧格式（v1.0 每会话目录）仅存在于历史数据，读侧工具内部处理，
**本契约只定义新格式**。

---

## 1. 顶层布局与格式判别

```
data/recordings/<task>/                        # <task> = 任务清洗名（可含中文）
├── videos/
│   └── chunk-{c:03d}/<image_key>/episode-{f:03d}.{ext}
├── data/
│   └── chunk-{c:03d}/episode-{f:03d}.parquet
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.jsonl
    ├── recycled_episode.json                  # 异常终止回退标记，见 §7.5
    └── episodes/
        ├── .lock                              # 运行时锁锚点，见 §7.4
        └── chunk-{c:03d}/episode-{f:03d}.parquet # 每 episode 一个文件（单行）
```

- **格式判别**：读 `meta/info.json` 的 `format` 字段，`"pooled_episodes_v1"`
  即本契约；`format` 缺失且存在 `<task>_NNNNNN` 子目录者为 v1.0 旧格式。
- 任务目录内**没有** `calibration/`、`metadata.json`、`timestamps.json`、
  每会话子目录（v1.0 遗留物）。
- `<task>` 即任务名；上传/项目语义中「会话名」概念废除，任务名即项目名。

## 2. 编号规则（MUST）

全局 episode 序号 **N 从 1 起递增**（跨该任务全部 episode；权威 = data/
videos 文件组扫描与 `tasks.json` 进度水位取 max）：

```
chunk = (N - 1) // 1000     # chunk-000 装 N=1..1000
fnum  = (N - 1) % 1000      # 三位零起：episode-000 .. episode-999
```

- `chunks_size = 1000`（info.json 声明，服务器/工具按此计算，勿写死）。
- 同一 episode 的**所有文件共用同一个 (chunk, fnum)**：视频各流、data
  parquet、episodes 每段文件（episode-005 = 第 6 个 episode，全部同名）。
- **界面/上传的任务名后缀 = episode 文件号（0 基）**：第 N 段显示
  `_ep{(N-1)%1000:06d}`（即 `_ep000011` = 本地 `episode-011`）。回放/上传
  界面显示名、zip 内 arcname、上传表单 `episode_index` 值**三处同一号码**
  （= 本地文件后缀）；真实全局序号 N 只出现在 parquet 的
  `episode_index` 列，不靠界面名反推。
- **异常终止不占号**：GUI「异常终止」丢弃的录制**不消耗序号**——中止时
  写下 `meta/recycled_episode.json` 标记（§7.5），下一次录制优先复用该号
  （即使进度水位已跑在前面也不跳号）；正常完成即清标记。**删除（含上传后
  自动删除）不回退**：已完成的 episode 序号永不复用（服务器已持有该号）。
  删除即彻底删除（v1.1.2 起无 `_trash/` 回收区）。
- **旧分片命名残留（读侧 MUST 容忍）**：v1.1.0 早期曾用「每 chunk 一个
  分片」，分片文件名 = chunk 自身编号（chunk-000 → file-000.parquet，
  一个文件装该 chunk 内最多 1000 个 episode 的行）。该布局已废除，新写入
  一律每段一文件；读侧工具（`episode_row`/上传/迁移）对旧分片按
  `episode_index` 列过滤做回退，**不靠文件名**。v1.1.2 起每段文件前缀
  `episode-`、分片保留 `file-` 前缀，两者路径不再重名。

## 3. meta/info.json（任务级，MUST 字段）

单 JSON 对象（可读缩进，无单行约束）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `format` | str | `"pooled_episodes_v1"`（格式判别键） |
| `chunks_size` | int | `1000` |
| `data_path` | str | 模板 `data/chunk-{c:03d}/episode-{f:03d}.parquet` |
| `video_path` | str | 模板 `videos/chunk-{c:03d}/{image_key}/episode-{f:03d}.{ext}` |
| `episodes_path` | str | 模板 `meta/episodes/chunk-{c:03d}/episode-{f:03d}.parquet` |
| `codebase_version` | str | 数据代码基准版本（如 `"v3.0"`，沿用服务器侧语义） |
| `app_version` | str | 主程序版本（如 `"1.1.0"`） |
| `fps` | float | 录制帧率（如 `30.0`） |
| `video` | bool | 是否含视频 |
| `task_name` | str | 任务名（= 目录名） |
| `features` | obj | **只声明实际存在的列**：`action` 恒在；`observation.<sn>` 每传感器一列；有 IMU 才有 `observation.imu`。形如 `{"action": {"dtype": "float32", "shape": [1]}, "observation.imu": {"dtype": "float32", "shape": [6]}, "observation.right_glove": {"dtype": "float32", "shape": [256]}}` |
| `cameras` | obj | RGB 相机键 → `{height, width}`（深度槽不入 cameras，见 `video_extensions`） |
| `devices` | list | 设备紧凑注册：`{key, kind, name, slots}`（kind: d435/s80m/uvc…；slots 为该设备的 image_key 列表） |
| `device_names` | obj | image_key → 设备显示名 |
| `sensors` | list | 任务级出现过的传感器名 |
| `sensor_dim` | int | 传感器观测维度（如 `256`） |
| `created_at` | float | 任务创建时间（Unix 秒） |
| `calibration` | obj | **任务级最新标定**（每设备一键：如 `head_stereo`，内容为 `StereoCalibration.to_dict()` 规范化结构：`type/name/resolution/fps/baseline/left_camera/right_camera/depth_scale/cam_imu_timeshift`，相机内参为 `{intrinsic: [fx,fy,cx,cy], distortion: [...]}`）。多设备时以设备名/前缀消歧编号（`d435_rgb_calibration` 等） |
| `total_episodes` | int | 当前 episode 数（上传删除不回退，见 §6） |
| `video_extensions` | obj | **每个 image_key** → 扩展名（RGB=`mp4`；深度=`mp4`=12-bit 灰度 HEVC、`mkv`=FFV1 回落）；读侧必须按此判文件，勿按扩展名硬猜格式 |

## 4. meta/stats.json（任务级全局统计，v1.1.1 契约）

单 JSON 对象。**每块 {count, mean, std, min, max}，列表维度 = 列维度**：

```json
{
  "observation.right_glove": {"count": 1196, "mean": [...256 个], "std": [...],
                              "min": [...], "max": [...]},
  "observation.imu":         {"count": 9922, "mean": [...6 个], "std": [...],
                              "min": [...], "max": [...]},
  "action": {"count": 0, "mean": [0.0], "std": [1.0], "min": [0.0], "max": [0.0]}
}
```

- **`count` 是累加器契约**：`sum = mean * count`、`sum_sq = (std² + mean²) * count`。
  写入方在每个 episode 结束后**读入现有 stats.json、按公式合并本 episode 统计、
  原子写回**；文件自身即持久累加器，**不存在任何边车**（`.stats_state.json`
  已废除，读到即旧版残留，忽略即可）。
- **MUST**：`count` 恒存在。无 `count` 的块 = v1.1.0 旧格式，写入方须全量重扫
  `data/` 重建一次再增量合并（参考 `core.helpers.recalc_stats`）。
- `action` 恒为占位块（`count:0`，仅声明列形状，不参与统计）。
- 只统计实际存在的列：帧级 `observation.<sn>`（逐行计数，含全零行）、样本级
  `observation.imu`（按样本计数）；`observation.*hand_pose` 为恒写占位零列
  （后处理回填），**不入统计**。
- 语义：删除 episode 不回退 stats（与任务进度水位一致）；stats.json 是全任务
  聚合快照，不是逐 episode 重建值。

## 5. meta/tasks.jsonl（任务级，单行 JSONL 是格式契约）

```jsonl
{"task_index": 0, "task": "D435-裸手_3D手部关键点识别_AI标注"}
```

- **MUST 每行一个完整 JSON 对象、且当前实现为单行**；读取方必须
  `json.loads(f.readline())` 逐行解析。**禁止美化/多行重排**——多行会静默破坏
  首行读取器（task_index 变 0）。写入方只在文件不存在时创建。

## 6. meta/episodes/（episode 元数据，每段一个文件）

### 6.1 文件与原子性

- **每 episode 一个文件** `chunk-{c:03d}/episode-{f:03d}.parquet`（与 data/videos
  同编号：episode-000 = episode 1），**恒 1 行**（10 列，见 §6.2）。
- episode 结束时**单行原子写**：临时文件 + `os.replace` 原子替换；同号重录 =
  覆盖该文件。读取方 MUST 容忍文件在任意时刻被原子替换（每次打开重新
  读取，勿缓存 fd）。
- 新布局写入**不需要 flock**（单文件原子替换即完整）；flock 锚点 `.lock`
  仅剩旧分片回退的读-改-写路径使用（见 §7.4）。
- **旧分片回退（读侧 MUST 容忍）**：v1.1.0 早期目录可能存在「每 chunk 一个
  分片」（文件名 = chunk 编号的 file-{c:03d}.parquet，≤1000 行，行序按
  `episode_index` 升序）。读侧定位行必须按 `episode_index` 列过滤、
  **不靠文件名**（v1.1.2 起分片保留 file- 前缀、每段文件改 episode-
  前缀，路径不再重名）。
  `scripts/migrate_pooled_storage.py --split-episodes` 可把旧分片原地拆成
  每段文件（幂等）。

### 6.2 列（10 列，MUST）

| 列 | 类型 | 说明 |
|---|---|---|
| `episode_index` | int64 | 全局序号 N（1 起） |
| `task_index` | int64 | 任务序号（现恒 0） |
| `start_frame_index` | int64 | 首帧 frame_index |
| `end_frame_index` | int64 | 末帧 frame_index |
| `length` | int64 | 帧数 = data parquet 行数 |
| `created_at` | float64 | episode 开始时间（Unix 秒） |
| `duration_sec` | float64 | 时长（秒） |
| `drop_stats` | str | **JSON 字符串**（如 `{"imu_overflow": 0}`），读取方须 `json.loads` |
| `video_codec` | str | **JSON 字符串**：本机原始编码信息（encoder/codec/crf/ffmpeg 路径/probe）；上传 zip 中的 mp4 可能已被再编码，见 §8 |
| `calibration` | str | **JSON 字符串**：本 episode 标定（结构与 info.json 的 calibration 同型），保每 episode 标定保真 |

## 7. data/（每 episode 一个 parquet）

`data/chunk-{c:03d}/episode-{f:03d}.parquet`，zstd 压缩，**稀疏列**（有数据才写该列）。

### 7.1 键列（恒有）

| 列 | 类型 | 说明 |
|---|---|---|
| `episode_index` | int64 | = N |
| `frame_index` | int64 | 帧内序号，0 起递增，无空洞（abort 不落盘） |
| `timestamp` | float32 | **墙钟相对秒**（episode 内相对，f32 精度够） |
| `task_index` | int64 | 现恒 0 |
| `wall_time` | float64 | **绝对 Unix 秒**（宿主时钟锚） |
| `hardware_ns` | int64 | 设备时钟硬件时间戳（ns，可绕回）；IMU 挂靠基准 |

### 7.2 观测列（稀疏）

| 列 | 类型 | 说明 |
|---|---|---|
| `observation.<sn>` | list<float32, 256> | 每帧一个传感器读数（维度 = info.json `sensor_dim`） |
| `imu_ts_ns` | list<int64> | 与 `observation.imu` **一一对应**的样本时间戳 |
| `observation.imu` | list<list<float32, 6>> | 挂在帧行上的**变长样本列表**（1000Hz/400Hz 只是 list 长短变化） |
| `observation.left_hand_pose` | list<float32, 63> | **恒写占位零**，后处理回填（裸手 63 维 world landmarks） |
| `observation.right_hand_pose` | list<float32, 63> | 同上 |
| `action` | list<float32, 1> | 恒写（现为 0） |
| `status.<did>` | str | 稀疏设备状态列（`"connected"`/`"disconnected"` 等） |

### 7.3 时间对齐模型（MUST 理解）

**表为帧行为主**：一行 = 一个 30fps 帧；视频帧/传感器读数/帧行共用同一
`frame_index`（写入线程同拍产出）。IMU 为变长 list 列挂在帧行上，录制时按
`hardware_ns` 设备时钟窗口挂靠。读取零重采样；宿主时间锚定靠 `timestamp` +
`wall_time`。客户 9 列平表由导出工具展开插值合成，非本契约内容。

### 7.4 meta/episodes/.lock（运行时辅助文件，非数据）

`meta/episodes/.lock` 是**空文件**，flock 锁锚点——旧分片回退的读-改-写用
`os.replace` 原子替换（inode 变化），锁必须挂在**路径不变的锚文件**上，故锚点
为空文件、锁在 fd 上而非文件内容。新布局的每段文件写入是单文件原子替换，
不走该锁。**其他程序 MUST 忽略该文件**（不解析、不打包、不告警）；上传 zip
不含它（dotfiles 一律不进包）。

### 7.5 meta/recycled_episode.json（异常终止回退标记，非数据）

内容为单 JSON 对象 `{"episode_index": <int>, "freed_at": "<ISO>"}`。录制被
「异常终止」时由写入器**原子写入**（临时件 + `os.replace`），标记该序号
可复用；下一次录制取号时若该号未被文件组占用则**优先复用**；
录制正常完成时清除，号已被占（跨机共享目录/遗留）时自动放弃。删除
（含上传后自动删除）**不写**该标记——已完成的序号永不复用（见 §2）。
**其他程序 MUST 忽略该文件**（不解析、不打包、不告警）；上传 zip 不含它。

## 8. videos/（每 episode 每流一文件）

- 路径 `videos/chunk-{c:03d}/<image_key>/episode-{f:03d}.{ext}`，`<image_key>` 为
  槽名（同任务内不同 episode 之间**允许槽改名**，如 ep1 用 `D435_depth_rgb`、
  ep2 用 `D435_head_rgb`——读侧必须按 episodes 行的 image_key 枚举，勿假设
  任务级固定集合）。
- 扩展名以 info.json `video_extensions` 为准：RGB=`mp4`（HEVC/h264），
  深度=`mp4`（12-bit 灰度 HEVC）或 `mkv`（FFV1 回落，仅 x265 无 12-bit
  灰度能力时产生）。
- **深度视频（v1.1.2 起，MUST）**：单流 12-bit 灰度 HEVC MP4——hevc Rext
  `gray12le` 单平面（对数深度码，qp=6、range=full、不带 profile；hvc1
  容器标记），解码后按对数码反量化回 uint16 毫米（实现口径在
  `core/depth_codec.py` / `core/depth_reader.py`）。读侧注意：cv2 直读
  gray12le 的 8-bit 转换不可靠，像素一律走 ffmpeg CLI 解 gray12le。
- **FFV1 回落（读侧 MUST 支持）**：x265 不可用时写 FFV1 `gray16le` 无损
  MKV（uint16 毫米原值），`video_extensions` 记 `mkv`。
- **旧深度布局（读侧 MUST 容忍，仅历史数据）**：
  - v1.0.14 双流 MKV：流0 = 热力图 H.264（默认播放画面），流1 = FFV1
    `gray16le` 无损 uint16 毫米深度（用 ffprobe/ffmpeg 的 stderr 探测
    FFV1 流号）；v1.0.14 迁移合成的单流 MKV 同款；
  - PNG16：v1.0.12 及更早的单帧 PNG 序列。
- RGB 的 `_aux` 类辅助流归主 key 同一目录，非主键可忽略。

## 9. 上传 zip 契约（给服务器导入器，端点不变）

`POST /api/v1/session/upload`，`name` 语义 = **任务名**，form 字段
`episode_index`（字符串化的 **episode 文件号**——0 基，与本 episode 的
`episode-{f:03d}` 完全一致，见 §2「任务名后缀」；真实全局序号 N 以包内
parquet 的 `episode_index` 列为准；旧服务器忽略未知字段）。

zip = **单 episode 切片**，包内 arcname 与原池化目录完全一致（相对任务目录）：

```
videos/chunk-{c:03d}/<key>/episode-{f:03d}.{ext}   # 仅本 episode 的各流文件
data/chunk-{c:03d}/episode-{f:03d}.parquet         # 本 episode 数据
meta/episodes/chunk-{c:03d}/episode-{f:03d}.parquet# 本 episode 单行文件直传
meta/info.json                                   # 任务级快照
meta/stats.json                                  # 任务级快照（v1.1.1 含 count）
meta/tasks.jsonl                                 # 任务级
```

- **MUST**：dotfiles（`.lock`、临时件）不进包；episodes 直传本 episode 的单行
  文件（旧分片回退时打成单行切片，arcname 同为新命名）。
- 导入器以 `info.json.format == "pooled_episodes_v1"` 识别新格式。
- **HEVC 预压缩**：zip 内 mp4 可能已被再编码为 HEVC（hvc1, CRF30）以缩小体积，
  与 episodes 行 `video_codec`（描述本机原始编码）**不必一致**；mkv 深度永不再编码。
- 上传删除 = 删该 episode 文件组（videos/data 各 key 本 episode 文件），
  episodes 行、stats、total_episodes **不回退**。

## 10. 关键点镜像布局（keypoints_output/）

后处理产物（手部关键点）按同一池化键镜像存放，与录制目录解耦：

```
keypoints_output/<task>/episode_{N:06d}/
├── hand_pose/chunk-000.parquet        # 2D 关键点（MediaPipe 后端）
├── auto_labels/auto_labels.parquet    # 自动标注
└── hand_pose_3d/chunk-000.parquet     # 3D 关键点（bare 63 维 world landmarks，
                                       #   回填 data parquet 两列的来源）
```

镜像目录内文件名不含 episode 号（父目录已键控），按帧 `frame_index` 与
data parquet 对齐。

## 11. 不属于架构的文件（其他程序须忽略/不得依赖）

| 文件/目录 | 说明 |
|---|---|
| `meta/episodes/.lock` | 运行时 flock 锚点，恒空 |
| `meta/recycled_episode.json` | 异常终止回退标记（§7.5），完成即删 |
| `meta/.stats_state.json` | v1.1.0 边车残留，**已废除**，见即删 |
| `_trash/` | v1.1.1 及更早应用内删除的回收区残留；v1.1.2 起删除即彻底删除，不再产生 |
| `.{name}.tmp` | 原子写临时件（`os.replace` 前身） |
| `_episodes_{task_id}_{N}.parquet` | 上传打包临时切片（仅旧分片回退路径产生），打完即删 |
| `_migrate_backup_*` | 一次性迁移备份目录 |

## 12. 对齐检查清单（新程序接入逐项验证）

1. 读 `info.json` 判 `format=="pooled_episodes_v1"`，未知 format 报错不猜。
2. 路径全部经模板 + (N→chunk,file) 计算，勿扫描拼接；`chunks_size` 读 info.json。
3. episodes 每段一个文件（episode- 编号 = data/videos 同编号）；旧目录的多行
   分片（file- 前缀）按 `episode_index` 列过滤回退，勿按文件名对齐。
4. stats.json 每块必有 `count`；无 count 的旧块触发全量重算或忽略并告警。
5. tasks.jsonl 逐行 `json.loads`，写回保持单行。
6. 深度文件按 `video_extensions` 判扩展名；12-bit 灰度 MP4 走 ffmpeg
   gray12le 解码 + 对数码反量化，FFV1 MKV 做流号探测（双流/单流都
   支持），旧 PNG16 按历史回退读取。
7. 视频槽名按 episodes 行枚举，容忍跨 episode 槽改名。
8. dotfiles 与临时件一律忽略；上传 zip arcname 与池化目录一致、episodes 带
   本 episode 单行文件、`name`=任务名 + form `episode_index`。
9. IMU 是变长 list 列，读取端展开时按 `imu_ts_ns` 对齐，勿假设固定采样率行宽。
10. 新布局 episodes 写入 = 单文件临时件 + os.replace 原子替换（无需 flock）；
    旧分片回退的读-改-写必须走 flock（POSIX）+ 临时文件 + os.replace。
