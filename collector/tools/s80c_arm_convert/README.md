# s80c_arm_convert — ARM 端 S80C 数据集 → 主程序会话格式

demo 转换器：把 ARM 板（RV1126B 等）采集的 S80C 数据集转成主程序录制会话格式
（LeRobot v3 布局），灰度 Bayer 视频转彩色。

## 输入格式约定（ARM 端，采集侧）

**真实 ARM 样本已 pin（2026-08-28，`/home/stouch/dataset`），以下布局已验证**：

```
dataset/
├── meta/info.json             # 采集侧元数据（字段均已实测对齐）:
│   ├── fps: 30.0
│   ├── cameras.left/right: {key, width:1280, height:800, codec:"hevc",
│   │        content:"raw_bayer_gray", video_path:"videos/<eye>/chunk-000/file-000.mp4"}
│   ├── imu: {rate_hz, path:"data/imu.bin", sample_struct:">Q6d",
│   │        sample_bytes:56, sample_count, gravity}
│   ├── timestamps_path: "data/timestamps.json"
│   └── device: {model:"S80M", bayer_pattern:"BG2BGR",  ← OpenCV 常量名,
│             color_mode:"raw"}
├── data/timestamps.json       # {"time_base":"ns", "fps", "frame_count",
│                              #  "frame_ts_ns": [每帧相机时间戳 ns]}
│                              #  实测: 50fps 相机被 30fps 采样 → 步进 20/40ms
├── data/imu.bin               # 每条 56B = >Q6d: 8B ts(ns) + 6×8B double(gyro×3+acc×3)
│                              #  实测 ~400Hz（info 声称 200 仅供参考，以 ts 为准）
└── videos/<left|right>/chunk-000/file-000.mp4   # HEVC 1280×800 30fps，
                                                 # 灰度 = 原始 Bayer 阵列
```

**回退路径**（真实布局之外的其他采集方式，尽力识别）：
- `data/*.parquet` 逐帧表（frame_index + timestamp/hardware_ns 列名回退）优先于
  timestamps.json；parquet 里已有 `imu_ts_ns`+`observation.imu` 时 IMU 直接用
- `meta/episodes/*.parquet` 的 episode_index 列 → 输出 episode 序号
- videos/ 下文件名含 left/right（或 cam0/cam1）按名分目；无名单文件按上下叠
  stacked 处理（上半左目/下半右目）；无名双文件按序当左右目

**Bayer 阵列**：优先读 info.json 的 `device.bayer_pattern`——采集端写的是
OpenCV 常量名（如 "BG2BGR"），本工具自动翻译（BG2BGR→真实 rggb），
`--bayer` 显式覆盖。**录制文件存储相位（2026-08-31 用户 A/B 裁决钉死）**：
右目 = info 声明的阵列（实测 rggb，OpenCV 名 BG2BGR）；左目 = 右目相位的
翻转（bggr）——采集端把左目物理倒装烘焙成 180° 几何旋转，Bayer 相位随之
rggb→bggr。旧「RGGB 三证」的相位配对测试只 pin 住 G 相位（(0,1)/(1,0)），
rggb 与 bggr 在此测试下不可分（R/B 盲区），被用户肉眼裁决推翻。SDK 路径
相位在 yaml rotate（0/1/0）；OpenCV 回退路径用 `--bayer-left/--bayer-right`
（默认左=右的相位翻转）。

## 用法

```bash
cd /home/stouch/collector
venv/bin/python tools/s80c_arm_convert/convert_arm_dataset.py \
    --input /home/stouch/dataset \
    --output data/recordings/<Task>/<Task>_0 \
    --task-name <Task>
```

输出目录结构（字段级对齐主程序 writer）：

```
<output>/
├── timestamps.json            # 每帧左右目两条（frame_index/timestamp/hardware_ns），按 hw 排序
├── data/imu/chunk-0000/chunk_000000.parquet   # 7 列 schema，zstd
├── meta/info.json             # codebase_version v3.0、cameras、features imu shape[6]
├── meta/stats.json            # IMU 真实 mean/std/min/max
├── meta/tasks.jsonl
├── meta/episodes/chunk_000000.parquet + chunk-000/file-000.parquet
├── metadata.json              # egodata 1.0
├── calibration/head_stereo.json   # 输入侧有 *calib*.json 自动带入，或 --calib 指定
└── videos/<slot>/chunk-0000/<slot>.mp4   # 彩色 MP4（x264 ultrafast crf23 yuv420p）
```

## 灰度 → 彩色 的关键假设（⚠ 重要）

S80C **没有硬件 ISP**：灰度流就是传感器 Bayer 直出，彩色化由 SDK 软 ISP 完成
（BLC → WB/AWB → demosaic → CCM → gamma）。本工具默认直接调 **SDK 的离线
彩色化接口**（`FAYS_VIK_Offline_*`，x86 3.9.0 库，与采集端
`stereo_color_mode=1` 的 RGB 模式**同一套参数源**——即 vikit yaml 里的
`stereo_awb` + `stereo_R/G/B_gain`）：

- SDK 离线 ISP 默认开：左右目分文件且 1280×800 时自动使用
  （`third_party/lib/libfays_vikit.so` + 随工具附带的
  `config/fays_vikit_stereo_rgb.yaml`，上色参数与 ARM64 驱动包
  `stereo_s80m/dist/s80m_stereo_camera_arm64_v1.0.zip` 的
  `config/fays_vikit.yaml` 一致，**rotate 按录制文件口径取 0/1/0**——
  驱动逐字 1/0/1 是原始传感器流的相位，见下）
- `--isp-yaml <path>`：采集端自己那份 vikit yaml 更准（采集端把它存进数据集
  就是逐设备精确还原）
- `--sdk-awb 0/1` / `--sdk-gains R,G,B`：运行时覆盖 yaml 的白平衡
- `--sdk-wb-auto`：SDK 上色后灰世界精修——采样帧统计全局 B/R 增益
  （目标 G/B=G/R=1，钳制 0.6-1.8）。**SDK 离线 AWB 对真实场景收敛不彻底
  （实测多帧一致残留约 11% 的 G/B 余差），输出仍微微偏暖/偏绿时开这个**；
  也可 `--sdk-wb-gain B,R` 手动给增益（两者互斥）
- `--no-sdk-isp`：回退纯 OpenCV 路径（去马赛克 + 白平衡 `--wb 1.2,1.0,1.5`
  + gamma `--gamma 2.2`，库缺失/非 1280×800 输入时也自动走这条）；录制文件
  分目相位：左目 `bggr`（RG2BGR）/ 右目 `rggb`（BG2BGR），可经
  `--bayer-left/--bayer-right` 覆盖

实测钉死（2026-08-28 合成色块 + 真实样本，2026-08-31 用户 A/B 裁决复核）：
- **离线管线读 yaml 的 awb/增益，不读 isp_param.ini**（改 ini 输出不变）
- **字节序按 SDK 输出 struct 的 encoding 字段直出，不翻转**：输出
  `AtrakImage.encoding` 声称 `AIE_BGR8`（枚举 0，SDK 头文件注释
  "BGR-packed if 3-channels"，实测运行时常量 0）——字节直出即 BGR，
  与写入端 `rawvideo bgr24` 严格一致；若某版 SDK 声称 `AIE_RGB8` 则
  自动换通道成 BGR。合成色块（FFV1 无损输入）双眼红→红、蓝→蓝 6/6
- **rotate_180 离线 = Bayer 相位开关**（合成色块实测：rotate=1 → 按 rggb
  相位读该目、rotate=0 → 按 bggr 相位读、R↔B 解释互换），不施加几何旋转。
  **录制文件存储相位：左目 bggr / 右目 rggr**（2026-08-31 用户 A/B 裁决
  B=红蓝互换为对钉死；旧「RGGB 三证」的相位配对只 pin 住 G 相位，rggb 与
  bggr 不可分）→ 默认 yaml rotate `0/1/0` 恰好双眼正确。驱动逐字 `1/0/1`
  只适用于**原始传感器流**（左目 rggb/右目 bggr，左目物理倒装未烘焙）：
  `--isp-yaml config/fays_vikit_raw_sensor.yaml` + `--rotate-left 180`。
  `stereo_swap_lr` 离线惰性；`stereo_init_exposure`/`stereo_gain_value`/
  端口字段离线无影响（逐字节相同）
- 合成测试输入勿用 mp4v/H.264：有损编码会把 Bayer 棋盘压平（实测
  (偶偶)-(奇奇) 相位差 200→0），须用 FFV1 无损 AVI 或直喂内存

**如果采集端已经把 Bayer 解成灰度再存（或做了不可逆的灰度化），颜色信息已丢失，
任何工具都无法还原**——此时只能用 `--already-demosaiced` 存灰度 BGR。

拿到真实样本后核对三点：Bayer 顺序（拍个红/绿/蓝色块对图）、上下叠 vs 分文件、
IMU bin 字节序（大端 `>Q6d` 是 8B ts + 6 double）。

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--fps` | 30 | 输出帧率；输入 info.json 有 fps 时优先 |
| `--bayer` | rggb | Bayer 阵列顺序（仅 OpenCV 回退路径；=右目相位，左目默认取相位翻转） |
| `--bayer-left` | 右目的翻转 | 左目 Bayer 阵列顺序（仅 OpenCV 回退路径；录制文件默认 bggr） |
| `--bayer-right` | 右目声明 | 右目 Bayer 阵列顺序（仅 OpenCV 回退路径；录制文件默认 rggb） |
| `--wb` | 1.2,1.0,1.5 | 白平衡增益 r,g,b（仅 OpenCV 回退路径） |
| `--gamma` | 2.2 | gamma（1.0 关闭；仅 OpenCV 回退路径） |
| `--no-sdk-isp` | off | 禁用 SDK 离线 ISP，强制 OpenCV demosaic |
| `--isp-yaml` | config/fays_vikit_stereo_rgb.yaml | SDK ISP 配置（采集端自带那份更准） |
| `--sdk-awb` | yaml 值 | 覆盖 yaml 的 stereo_awb（0/1） |
| `--sdk-gains` | yaml 值 | 覆盖 yaml 的 stereo_R/G/B_gain，如 1.0,0.6,1.3 |
| `--sdk-wb-auto` | off | SDK 输出后灰世界精修（去残余偏绿，推荐开） |
| `--sdk-wb-gain` | 无 | SDK 输出后 B,R 手动增益，如 1.08,1.11 |
| `--stacked` | 按文件名 | 单 MP4 上下叠双眼（无名文件自动判，见上） |
| `--already-demosaiced` | off | 采集端已解 Bayer（仅灰→灰） |
| `--resize` | 1280x800 | 去马赛克后统一缩放（`0` 不缩放） |
| `--rotate-left/right` | 0 | 单目旋转（0/180） |
| `--swap` | off | 交换左右目 |
| `--imu-endian` | `>` | IMU bin 字节序（`<` 小端） |
| `--calib` | 自动找 | 标定 JSON（写入 calibration/） |

## 依赖与 ffmpeg

- numpy / opencv-python / pyarrow（venv 已齐；**无 pandas**）
- ffmpeg：按 `imageio-ffmpeg` → 环境变量 `DAQ_FFMPEG` →
  `~/miniconda3/envs/lerobot/bin/ffmpeg` → PATH 顺序找。
  注意 conda base 的 ffmpeg 因 openvino/tbb 不可用。

## 已 pin 结论（真实样本 2026-08-28，15/15 验证通过）

1. **字段名**：全部对齐（info.json / timestamps.json / imu.bin / videos 布局，
   见上）；本工具仍保留回退识别，采集侧改 schema 也能跑
2. **颜色可还原** ✓：灰度域 Laplacian 方差 54490 → 确认原始 Bayer 马赛克；
   SDK 离线 ISP 彩色化后马赛克口径亮度相关 corr=0.9588/0.9583（f10/100/300，
   源 2×2 周期平均消棋盘）、合成色块（FFV1 无损）双眼红/蓝/灰 6/6 正确；
   2026-08-31 用户 A/B 裁决（B=红蓝互换为对）后重转：输出与 B 对照物
   色度相关 +0.91（左目）、与 A/B 静帧 B 半 +0.89（右目）；灰世界精修后
   全帧均值 BGR≈(121.8,125.3,121.9)（G/B=1.029、G/R=1.028）近中性，
   增益 B×1.081 R×1.114
3. **分文件**（videos/left|right/...）；**录制文件存储相位 左 bggr / 右
   rggb**（OpenCV 名 RG2BGR / BG2BGR）——info 声明的 "BG2BGR"=右目口径，
   左目=相位翻转（采集端烘焙左目倒装 180°，见「Bayer 阵列」节）
4. **IMU bin 大端 `>Q6d`** ✓（与 --pipe 协议同款）；IMU 与视频帧同源时钟
   （启动期 294.7ms 样本归首帧、晚于末帧 hw 的样本按 writer 口径丢弃）
5. 实测杂项：IMU 实际 ~400Hz（info 的 rate_hz=200 不准）；首帧间 1003ms
   启动空档 → 第 1 行 IMU 455 个样本属正常；左右目视频帧数可能差 1（440/439），
   取最小值并 WARN

## 色彩还原的置信度说明

默认走 **SDK 离线 ISP**，上色参数（awb=1、增益 1.0/0.6/1.3）与 ARM64 驱动包
`stereo_s80m/dist/s80m_stereo_camera_arm64_v1.0.zip` 内
`config/fays_vikit.yaml` 一致，rotate/swap 按**录制文件口径**取 `0/1/0`
（驱动逐字 `1/0/1` 只适用于原始传感器流，见 `fays_vikit_raw_sensor.yaml`）。
差异已实测钉死并按下述原则处理：

1. **通道序**：SDK 输出 struct 的 `encoding` 字段声称 `AIE_BGR8`（实测
   运行时常量 0，SDK 头文件注释 "BGR-packed if 3-channels"）——字节序
   以该字段为准直出（**不翻转**），写入端 `rawvideo bgr24` 与之严格
   一致；若某版 SDK 声称 `AIE_RGB8` 则自动换通道成 BGR。合成色块
   （FFV1 无损输入）双眼红→红、蓝→蓝 6/6
2. **rotate/swap 相位**：离线 `rotate_180` = Bayer 相位开关——rotate=1
   按 rggb 相位读该目、rotate=0 按 bggr 相位读（R↔B 解释互换、无几何
   旋转，合成色块实测）。**录制文件（几何已烘焙）存储相位 左 bggr /
   右 rggb**（2026-08-31 用户 A/B 裁决 B=红蓝互换为对；旧「RGGB 三证」
   相位配对只 pin 住 G 相位，R/B 盲区）→ 默认 yaml `0/1/0` 双眼正确。
   原始传感器流（未烘焙，左目物理倒装）存储相位 左 rggb / 右 bggr →
   驱动逐字 `1/0/1`，用 `--isp-yaml config/fays_vikit_raw_sensor.yaml`
   + `--rotate-left 180`。`stereo_swap_lr` 离线惰性
3. **AWB 收敛余差**：离线 AWB 对真实场景收敛不彻底（残留约 11% 通道余差，
   表现为微微偏暖/偏绿），开 `--sdk-wb-auto` 灰世界精修消除——真实样本
   实测增益 B×1.124 R×1.065，第 100 帧左目 BGR (117.5,123.1,123.0)、
   全帧均值 G/B=1.033 G/R=1.026 近中性；双眼色调一致差 3.5-4.8

用于训练/标注前建议肉眼核对输出视频色彩（`vlc` 或
`ffplay <输出>/videos/stereo_left/chunk-0000/stereo_left.mp4`）。
