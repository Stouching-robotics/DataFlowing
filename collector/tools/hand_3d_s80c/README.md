# hand_3d_s80c —— S80C 双目鱼眼实时裸手关键点 2D/3D demo

用 FaysSense VI Kit SDK 官方深度引擎（CPU SGBM）拿 S80C（FS-VI80-S80C）
的深度图，复用 [hand_3d_d435](../hand_3d_d435/) 的 `_run_3d_chain` 全链
（MediaPipe 检测 → 深度抬升 → 双槽分配 → 平滑 → 三窗口渲染/导出），
做实时 2D 骨架叠加 + 3D 旋转骨架 + 深度伪彩可视化。

## 快速开始

```bash
# 相机必须空闲（主程序/其他 SDK 进程占用时打不开）
./tools/hand_3d_s80c/run_live_s80c.sh

# 黑手套模式（glove_package YOLO-World + RTMPose；默认双目并排下
# 左右目都有手套关键点——右目共享左目平滑框按视差平移后
# 独立 RTMPose 推理，见下"黑手套模式"）
./tools/hand_3d_s80c/run_live_s80c.sh --glove

# 双目并排显示为默认（win1 显示 2560×800 矫正左右目；右目独立
# MediaPipe 检测渲染 2D 关键点，深度/3D 仍只用左目）；
# 不需要时 --no-stereo-view 关闭
./tools/hand_3d_s80c/run_live_s80c.sh --no-stereo-view

# 无窗口验证 / 导出
./tools/hand_3d_s80c/run_live_s80c.sh --no-window --stats
./tools/hand_3d_s80c/run_live_s80c.sh --export out/s80c_session
```

完整参数、依赖、worker 单跑见下两节。

## 依赖

### 硬件

| 项 | 要求 |
|----|------|
| 相机 | S80C（FS-VI80-S80C）双目鱼眼，FTDI USB 连接。**设备独占**：被主程序/其他 SDK 进程占用时打不开（worker 5s 看门狗退出，demo 重启一次后报错） |
| GPU | 建议但不必须：裸手模式纯 CPU 可跑；手套模式检测框 YOLO-World 走 torch CUDA（GPU ~12ms vs CPU ~85ms/帧）。RTMPose 走 onnxruntime CUDA EP |
| 显示 | 窗口模式需要 X11（如 `DISPLAY=:1`）；`--no-window` 不需要 |

### 运行环境（本机实测版本）

Python 3.12（本机 `venv/bin/python` = 3.12.3）。安装方式：

```bash
python3 -m venv venv && source venv/bin/activate
pip install -U pip
# 必装（裸手模式完整可用，CPU 即可）：
pip install numpy==2.5.2 opencv-python==5.0.0 mediapipe==1.0.0
# 手套模式（--glove）再加——YOLO-World 检测框 + RTMPose 关键点：
#   torch（NVIDIA GPU 见 https://pytorch.org/get-started/locally/ 装 CUDA 版；
#         CPU 版也能跑，检测框 ~85ms/帧 vs GPU ~12ms）
#   ultralytics==8.4.121  rtmlib  onnxruntime-gpu==1.29.0
# --export 导出再加：pyarrow==25.0.0（parquet）+ 系统 ffmpeg
#   （PATH 可找到即可；或 export FFMPEG_BIN=/path/to/ffmpeg 指定）
```

| 包 | 版本 | 必装？ | 用途 |
|----|------|--------|------|
| numpy | 2.5.2 | ✅ | 基础 |
| opencv-python | 5.0.0 | ✅ | demo 主进程 cv2（鱼眼 remap 在 worker 内） |
| mediapipe | 1.0.0 | ✅ | 裸手检测 + 手套 mediapipe 后端；**本机 venv 编译未开 GPU（"GPU processing is disabled in build flags"），恒 CPU XNNPACK**，`--delegate auto` 也落 CPU |
| torch / ultralytics | 2.13.0+cu130 / 8.4.121 | 手套模式 | YOLO-World 检测框（CUDA 实测可用；无 GPU 落 CPU） |
| onnxruntime-gpu | 1.29.0 | 手套模式 | RTMPose hand5 ONNX（CUDA EP；无 GPU 换 `onnxruntime`，初始化失败自动回退 CPU） |
| rtmlib | — | 手套模式 | RTMPose 封装；模型首次使用自动下载（见下） |
| pyarrow | 25.0.0 | `--export` | parquet 落盘 |
| ffmpeg | 系统包 | `--export` | render.mp4/rgb_overlay.mp4 编码（`FFMPEG_BIN` 环境变量或 PATH 探测） |

### 分发打包

`dist/s80c_hands_demo_v1.0/` 是自包含分发包：`third_party/`（SDK
3.9.0 + OpenCV 4.2，~52MB）+ 本模块 + 链上全部仓库内代码（
`hand_3d_d435/live_demo.py` 及其 8 个子模块、`stereo_s80m/hand_3d/`
7 个组件、`hand_detection/`、手套模式 `glove_package/` 3 文件 +
yolov8m-worldv2.pt、`models/hand_landmarker.task`、标定回退
`fayssense_depth_sdk/calib/calib.yaml`）+ 使用说明 + requirements。
发给他人的机器上只需 venv + pip 装依赖（见上），**不需要本仓库**。

重新打包（代码/依赖有改动后）：`./tools/hand_3d_s80c/build_dist.sh`
——从当前工作区复制（含未提交修改），覆盖 `dist/s80c_hands_demo_v1.0/`
并打 tar.gz。唯一对仓库文件的裁剪：`tools/stereo_s80m/hand_3d/__init__.py`
剥离 `run_pipeline` import（分发包不含该离线管线，S80C 链不引用）。

### SDK 与配置（默认路径，均可用参数覆盖）

**自包含**：SDK 库、OpenCV 4.2 预载、深度配置已整包复制进
`third_party/`（~52MB），demo 默认**零外部依赖**——外部
`FaysSense_VI_Kit_Release` 被改/删不影响本 demo（2026-08-24 起因：
用户在该外部目录自编译 stereo_depth_gui，其配置改动弄崩 demo）。

| 项 | 默认路径（仓库内） | 说明 |
|----|---------|------|
| SDK 库 | `third_party/lib/`（libfays_vikit.so + libfayssense_aikit_depth.so） | **3.9.0**（已验证）；3.9.1 有 RGB 失败段错误史，勿用。`--sdk-dir` 覆盖（可指回外部目录调试） |
| OpenCV 4.2 | `third_party/opencv4.2/lib406/` | SDK .so 符号解析用（RTLD_GLOBAL 预载）；`--opencv-dir` 覆盖 |
| 相机配置 | `tools/stereo_s80m/config/fays_vikit_50fps.yaml` | 与主程序同款的 50fps 副本（与 25fps 版唯一差异 `stereo_fps: 50`）；启动时按 sysfs 端口自动解析重写临时 yaml（FTDI 接口 1.0=双目/1.2=IMU，rgb 置 NULL——写死 /dev/videoN 会被 USB 插拔挤位，见 read_stereo_rgb.py 教训）；`--vikit-config` 覆盖 |
| 深度配置 | `third_party/config/stereo_depth.yaml` | depth_mode:1 CPU SGBM、async_mode:1；**worker 启动时重写 calib_path → 设备标定合成的临时 kalibr yaml（机型自动适配，见"已知边界"）、model_path → 仓库内 rknn**（临时副本，原文件不动）。**`input_stereo_width/height` 必须保持 1280×800**（与喂帧尺寸一致——改小后引擎内部缓冲不匹配，预热阶段直接段错误，2026-08-24 实测）；`--depth-config` 覆盖 |
| 标定回退 | `tools/fayssense_depth_sdk/calib/calib.yaml` | 仅 `GetCalibrationParam` 失败时用（引擎 calib_path 与 K1/D1 回退同源，cam0/cam1 equidistant KB4）；worker 单跑时 `--calib-yaml` 覆盖 |

`third_party/` 收录边界（取舍）：
- **收**：2 个 SDK .so；OpenCV 4.2 的 13 个模块 .so.4.2.0 + 逐模块
  `lib406/libopencv_*.so.406` 链（`libwebp.so.6→shims/libwebp.so.7`、
  `libtbb.so.2→shims/libtbb.so.12` 指回同目录，`libtiff.so.5` 指系统
  `/usr/lib/x86_64-linux-gnu/libtiff.so`）；webp/tbb 本体（来自 conda）；
  stereo_depth.yaml + rknn 模型。
- **不收**：videoio 及其 `libavcodec` 依赖链（conda 的 av* 牵连
  rsvg/gobject/x264/x265/aom 等二十余个库——demo 用不到
  cv::VideoCapture，相机走 ViKit、JPEG 模式走 venv cv2；worker 预载
  best-effort 自动跳过，无影响）；系统库（glibc/libtiff 等）按惯例
  留在 /usr/lib。

### 模型权重

| 用途 | 文件 | 说明 |
|------|------|------|
| 裸手 MediaPipe | `tools/models/hand_landmarker.task`（仓库内，~7.8MB） | HandLandmarker 21 点 |
| 手套检测框 | `tools/glove_package/yolov8m-worldv2.pt`（仓库内，~57MB） | YOLO-World + 提示词 `["hand","glove"]`，imgsz 320 / conf 0.05；权重名含 "world" 判 world 后端，`--glove-detector det` 切训练产物 `runs/hand_det/weights/best.pt`（yolo11n 单类 hand，普通 YOLO 后端 conf 0.3，运行中 v 键热切换） |
| 手套关键点（rtmpose 默认） | `~/.cache/rtmlib/hub/checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx` | hand5 SIMCC 256×256；rtmlib 首次使用自动从 openmmlab 下载（~几十 MB），本机已缓存 |
| 手套关键点（mediapipe 后端） | `tools/models/hand_landmarker.task` | 同裸手模型，框内裁剪检测 |

手套模式额外依赖 `tools/glove_package/world_detector.py`（只读复用，demo
自动加 sys.path，无需 cwd 在包内；注意 demo 只依赖这一个模块，不加载
glove_package 的 CLIP 标注工具链）。

## 使用方法

### 启动方式

```bash
# 推荐：启动器（自动补 SDK OpenCV lib406 到 LD_LIBRARY_PATH，双保险）
./tools/hand_3d_s80c/run_live_s80c.sh [参数...]

# 等价：直接 venv python 也可以——worker 启动时检测到
# LD_LIBRARY_PATH 缺 lib406 会 re-exec 自身补上（stdout 管道 fd 跨
# exec 保留），无需手动 export
DISPLAY=:1 venv/bin/python tools/hand_3d_s80c/live_demo_s80c.py [参数...]
```

### 典型场景

```bash
# 窗口实时演示（裸手）
./tools/hand_3d_s80c/run_live_s80c.sh

# 双目并排显示为默认（左右目各有一份 2D 关键点；深度/3D 只用左目）；
# 关闭用 --no-stereo-view
./tools/hand_3d_s80c/run_live_s80c.sh --no-stereo-view

# 黑手套模式（YOLO-World 框 + RTMPose）
./tools/hand_3d_s80c/run_live_s80c.sh --glove

# 黑手套 + 双目并排为默认（右目共享左目平滑框视差平移 + 独立 pose 推理）
./tools/hand_3d_s80c/run_live_s80c.sh --glove

# 手套关键点换 MediaPipe 后端（对比效果；运行中按 b 热切换）
./tools/hand_3d_s80c/run_live_s80c.sh --glove --glove-pose-backend mediapipe

# 无窗口验证（headless 跑处理链、打印 fps；q 或 Ctrl-C 退出）
./tools/hand_3d_s80c/run_live_s80c.sh --no-window --stats

# 录制导出（窗口/无窗口均可，退出后落盘）：
#   keypoints_2d.parquet（槽位 2D 关键点，像素）
#   keypoints_3d.parquet（质心锚定 3D，相机系米）
#   render.mp4（3D 旋转渲染）、rgb_overlay.mp4（原视频叠 2D 关键点）
./tools/hand_3d_s80c/run_live_s80c.sh --export out/s80c_session
./tools/hand_3d_s80c/run_live_s80c.sh --no-window --export out/s80c_headless

# 引擎矫正图对照（排查 remap 与引擎矫正不一致时；见"已知边界"）
./tools/hand_3d_s80c/run_live_s80c.sh --rect-mode sdk

# 丢手/远手调参
./tools/hand_3d_s80c/run_live_s80c.sh --det-conf 0.3 --track-conf 0.3 --det-scale 1.0

# 深度叠层启动即开（运行中按 d 切换）
./tools/hand_3d_s80c/run_live_s80c.sh --depth-overlay
```

### 全部命令行参数

**相机与 SDK：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--sdk-dir` | `third_party/lib/`（worker 默认） | 3.9.0 库目录（勿用 3.9.1） |
| `--vikit-config` | `tools/stereo_s80m/config/fays_vikit_50fps.yaml` | 相机配置模板（端口自动解析） |
| `--depth-config` | `third_party/config/stereo_depth.yaml`（worker 默认） | 深度引擎配置（calib_path/model_path 运行时重写为仓库内路径） |
| `--opencv-dir` | `third_party/opencv4.2/lib406`（worker 默认） | SDK 自带 OpenCV 4.2 |
| `--rect-mode` | `remap` | 2D 视图来源：`remap`=自身鱼眼矫正；`sdk`=引擎矫正图（对照用，尺寸可能非 1280×800 致 3D 失效） |

**显示与检测（裸手）：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--no-stereo-view` | （双目默认**开**） | 关掉 win1 左右目并排显示（右目独立检测/渲染；深度/3D 仍只用左目） |
| `--fixed-view` | （固定视角默认**关**） | 开启 3D 固定世界视角：首帧有手时锁定相机目标/缩放/网格，手在世界内自由运动（手移出视场就出画，`r` 在当前手位置重锁）。默认关=与 D435 相同，相机目标/缩放随手动 |
| `--max-bone-len` | `0.15` | 3D 单个关节（骨）长度上限，米。正常手骨最长腕→MCP ~0.12m，0.15 只截深度噪声离群点（沿父→子方向缩回，逐级级联保证所有显示骨长 ≤ 上限）；`0`=关 |
| `--depth-overlay` | 关 | 启动即开深度伪彩叠层（d 键切换） |
| `--delegate` | `auto` | MediaPipe delegate（venv GPU 不可用，恒落 CPU） |
| `--det-scale` | `0.5` | 检测输入缩放比（丢小/远手提回 1.0） |
| `--det-async` | （同步默认**关**） | 开异步检测：显示全帧直推 ~50fps 不等待检测（主程序 S80C 口径）；默认关=检测/显示同步、关键点与画面逐帧严格对应（D435 裸手同款，快动手不落后），窗口模式 ~30fps |
| `--no-extrap-2d` | （外推默认**开**） | 关掉关键点显示外推（仅 `--det-async` 时生效：把 1-3 帧前检测结果按框中心速度平移到当前时刻；同步模式不介入） |
| `--tear-probe` | （自动捕获默认**关**） | 开启撕裂自动捕获：排查画面撕裂时再开——后台保存最近 ~3s 内部显示缓冲，看到画面撕裂**直接按 q 退出**即自动导出 `keypoints_output/live_d435/tear_exit_*`（t 键可随时手动导出，不持续写盘）——离线判别：导出帧内有水平缝=数据/相机侧，全干净而画面撕裂=屏幕合成/远程桌面层。默认关=启动不保存帧缓冲、退出不导出 |
| `--det-conf` | `0.4` | 掌心检测置信度阈值 |
| `--track-conf` | `0.4` | 手部跟踪置信度阈值 |
| `--smooth-2d` | 关 | 2D 逐点 OneEuro 平滑（引入 ~2-3 帧滞后，快动慎开） |
| `--smooth3d-freq-min` | `1.5` | 3D 关键点 OneEuro freq_min（逐点 + M1 质心锚定共用）。**S80C/S80M 深度比 D435 噪**（~20fps 更新、有效率低），3D 抖动更大 → 默认在 D435 口径（3.0Hz）上加一倍静止平滑；`3.0`=D435 同款，越大越跟手但越抖、越小越稳但快动滞后越大 |
| `--no-raw-2d` | （raw-2d 默认**开**） | 关掉原始检测直绘、2D 显示回挂 3D 槽位链 |
| `--fill` | `0` | 对齐空穴回填轮数 0-3（恒等对齐 1:1 时 0 纯裁剪最快） |
| `--propagate-max` | `15` | 槽位丢失帧数硬顶（超限 absent 不幻觉） |

**手套模式（`--glove`）：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--glove` | 关 | 启动即黑手套模式（运行中按 g 切换） |
| `--glove-detector` | `world` | 检测器选择：`world`=yolov8m-worldv2.pt（开放词汇，提示词 hand/glove）；`det`=glove_package/runs/hand_det/weights/best.pt（yolo11n 单类 hand 训练产物，自动走 YOLO 后端）；v 键热切换 |
| `--glove-imgsz` | `640`（D435 320） | world 检测输入边长：S80C 默认 640——远手/小面积手在 320 下等效 4× 缩小掉出模型有效尺度，640 放大 2 倍找回（代价 GPU ~23→31ms/帧；近景大召回可能略降 320:40/40 vs 640:30/40） |
| `--glove-weights` | — | 检测框权重显式路径（优先于 `--glove-detector`；按文件名是否含 world 自动判后端） |
| `--glove-pose-backend` | `rtmpose` | 关键点后端：`rtmpose`（hand5 ONNX）/ `mediapipe`（框内裁剪）；b 键热切换 |
| `--glove-det-conf` | world 0.05 / best 0.3 | 检测框阈值 |
| `--glove-pose-conf` | `0.15`（D435 0.3） | 逐点置信均值门：低于持出上次输出并按平滑框位移平移补偿。S80C 默认更低——鱼眼矫正图置信系统性偏低，握拳/抓取在 0.3 门下一律被冻结无法捕捉 |
| `--glove-hold-max` | `12` | 低置信 hold 逃逸帧数（0=立即放行，-1=无限 hold）：连续低置信满该帧数即放行本轮骨架——持续低置信=真实新姿势（握拳/抓取黑手套逐点置信天然低），不无限冻结；瞬时低置信（运动模糊）仍持旧点防抖 |
| `--glove-nms-iou` | `0.6` | 框 NMS IoU（双手框频繁合并→手数闪变时降 0.45） |
| `--glove-lost-timeout` | `8`（D435 3） | track 丢框容忍帧数：S80C 远手/手背框闪烁时 3 帧即死会持续丢手，故默认 8（双手交叉/重叠可继续调高，按会话取舍） |
| `--glove-new-track-conf` | `0.1`（D435 0.25） | 新建 track 最低框置信度（双阈值）：world 框置信天然偏低（离线实测 93% < 0.25），0.25 门会挡死远手重捕获；flicker 框有碎片框拒绝兜底 |
| `--glove-box-alpha` | `0.7` | track 框 EMA 平滑系数 |
| `--glove-pose-box` | `smooth` | RTMPose 裁剪框来源：`smooth`=EMA 平滑框（链稳定）；`raw`=原始框（无稳态滞后但抖动直通下游） |
| `--glove-freeze-max` | `15` | 连续退化冻结输出上限帧数 |

**输出与诊断：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--no-window` | 关 | 不开窗口只跑处理链打印 fps（验证用） |
| `--stats` | 关 | 退出时打印统计诊断（单手帧/同标签帧/wholesale 等计数） |
| `--export DIR` | 无 | 导出 keypoints_2d/3d.parquet + render.mp4 + rgb_overlay.mp4 |

### 按键（与 D435 demo 一致）

| 键 | 作用 |
|----|------|
| `q` / `Esc` | 退出（worker 子进程随管道 EOF 优雅销毁 SDK 句柄；撕裂自动捕获默认关——排查撕裂时用 `--tear-probe` 启动，看到画面撕裂直接按 `q` 退出即自动导出最近 ~3s 内部帧 tear_exit_*） |
| `t` | 手动导出当前撕裂诊断环（仅 `--tear-probe` 时有效） |
| `d` | 深度伪彩叠层开关 |
| `g` | 黑手套模式切换（首次按会加载 YOLO-World，含 ~1s CUDA 预热） |
| `b` | 手套关键点后端热切换（rtmpose / mediapipe） |
| `v` | 手套检测器热切换（YOLO-World ↔ 训练 best.pt；track/滤波状态重置） |
| `s` | 截图 |
| `r` | 复位 3D 视角；`--fixed-view` 开时同时在当前手位置重锁相机（默认关时仅复位） |
| 鼠标左键拖拽 | 旋转 3D 视图（win2） |

### 环境变量

| 变量 | 作用 |
|------|------|
| `VENV_PY` | 启动器使用的 python（默认 `$REPO_ROOT/venv/bin/python`） |
| `DISPLAY` | 窗口模式需要 X11（如 `:1`）；`--no-window` 不需要 |
| `LD_LIBRARY_PATH` | 启动器自动补 SDK OpenCV lib406；直接 python 启动时 worker re-exec 自愈，无需手动设 |

### worker 单独跑（调试相机/深度引擎）

```bash
# stdout 是二进制管道（帧头+payload），勿直连终端；每秒深度统计在 stderr
venv/bin/python tools/hand_3d_s80c/s80c_depth_worker.py

# 抓管道离线回放（帧格式见"架构"节）
venv/bin/python tools/hand_3d_s80c/s80c_depth_worker.py \
    > /tmp/s80c_pipe.bin 2>/tmp/s80c_worker.log
```

worker 自有参数（demo 不全部暴露）：`--sdk-dir / --vikit-config /
--depth-config / --opencv-dir / --calib-yaml / --rect-mode / --stereo-view /
--pipe-format raw|jpeg`（默认 raw 零编解码；`jpeg` 仅调试回退——压缩噪声
会加重 MediaPipe 逐帧抖动）。用途：验证相机打开/端口解析/深度引擎
Bind、抓管道数据离线回放链路。SIGTERM 优雅退出、SDK 句柄释放无残留。

## 性能预期

- **检测/显示同步为默认（D435 裸手同款口径）**：每显示帧同步检测
  关键点与画面逐帧严格对应、快动手不落后（异步检测关键点滞后
  1-3 帧是"骨架跟不上手"的根因，2026-08-25 用户实测反馈后改默认）。
  0.5 缩放下 det 15.5ms/帧（左右目两趟），headless 实测 45.7fps、
  窗口模式 ~30fps（D435 同 30fps 量级）。`--det-async` 打开回
  50fps 全帧直推（latest-result 语义，显示不等待检测；关键点滞后
  靠 `--extrap-2d` 显示外推补偿，快动手仍不如同步跟手）。
- **2D 检测/叠加**：相机 50fps（1280×800 矫正左目，MediaPipe CPU
  XNNPACK）。无窗口全链消费 49-50fps 满帧；同步窗口模式实测
  ~30fps。worker 默认 raw BGR 管道（type=4/5，零 JPEG 编解码）+
  demo 默认 `--fill 0`（恒等对齐纯裁剪）。
- **显示与检测分辨率分离**：检测在 640×400 跑（`--det-scale 0.5`，
  landmark 按比例回全分辨率坐标，丢小/远手可提回 1.0）；并排视图
  2560×800 绘制仍全分辨率，仅 imshow 显示副本降半 1280×400（窗口
  仅 1024×576，屏幕有效分辨率不变；截图仍全分辨率）。
- **左右目各自独立检测器实例**：HandLandmarker 是 VIDEO 模式
  （跟踪状态跨帧保留），左右目共用同一实例会互相污染跟踪裁剪框
  （坐标系/尺度混串 → 关键点偏移+闪烁，2026-08-24 实测教训）。
  左目 `det`、右目 `det_r` 各跑各的跟踪状态。
- **2D 显示与 3D 槽位链解耦**：默认 `--raw-2d`——2D 骨架直接画当前
  帧原始检测，槽位 propagated/absent 或 wholesale 门控不影响 2D
  显示（3D 窗口/导出仍走槽位链；`--no-raw-2d` 回旧行为）。
- **深度更新**：CPU SGBM。SDK 文档标称 ~0.7s/帧（≈1.4fps）；**本机
  实测 ~20fps**（50fps 相机下每秒 ~18-22 张新深度，有效 ~40%，中位
  ~1.7m）。worker 只在有新深度时发送（demo 侧锁存最近一张），3D
  骨架由槽位 tracker 平滑衔接——即便深度慢到 1.4fps 也只表现为 3D
  更新点跳变、帧间平滑，不丢手不硬顶。
- 启动前 ~1-2s 为深度引擎预热（async_mode 需连续喂 ≥5 帧），期间
  仅 2D 渲染，预热完成后 3D 窗口自动出现。

## 黑手套模式（`--glove`，2026-08-24 修复两处 S80C 专属缺陷）

左目走 GloveDetector 全稳定层（world 框 + HandTracker + RTMPose +
逐点置信加权 + hold_translate），与 D435 同款。此前 S80C 上"效果很差"
的两处根因与修复：

1. **右目在手套模式下跑 MediaPipe**——黑手套对 MediaPipe 是死路
   （实测 4/68），右半边空/乱。现改为：共享左目 GloveDetector 的平滑
   框（`track_boxes()`），按视差平移到右目坐标（`x_r = x_l − fx·B/z`，
   框内中位深度，同场景双目、极线行对齐——fx=502.4、基线 79.7mm，
   1.2m 处视差 ~33px），再用**同一 pose 后端**在右帧裁剪推理
   （`pose_on_boxes()`：RTMPose stateless、mediapipe 后端 IMAGE 模式
   均无跨调用状态，共享实例安全——与裸手 det_r 的 VIDEO 模式教训
   不同；`b` 键热切换后端自动跟随）。右目 pose 按框位移门控（中心
   位移 <3px 不推理、10 帧强制刷新，同 tracker 口径），静止段省推理
   且骨架不抖；门控跳过帧复用上次平滑输出防闪没。
2. **手套模式被排除在 `--raw-2d` 之外**——2D 显示挂在 3D 槽位链上，
   S80C 深度稀疏（有效 ~40%、更新 ~20fps）导致槽位链频繁
   propagated/门控 → 手套骨架闪没/冻结（D435 深度密集无此问题）。
   现 raw-2d 对裸手/手套同样生效：2D 显示与 3D 槽位链解耦（3D
   窗口/导出仍走槽位链不变；D435 默认关 raw-2d，行为零变化）。
3. **左目关键点偏移/形状不对（比右目差）**——右目画 pose_on_boxes
   原始点（效果好），左目却画 detect() 稳定层输出：稳定层（逐点
   置信加权 + 随框平移/持出）在 S80C（world 框噪声 + 低置信段）
   会随框漂移/持旧点变形。现左目显示改与右目同口径——detect()
   本帧推理的原始 pose（`last_raw_pose()`，含低置信持出帧、零额外
   推理开销）+ 同款轻平滑（独立 `smo2d_left_glove` 实例）；本帧无
   原始点的帧（退化/门控跳过）复用上次平滑输出防闪没。3D 槽位链
   仍用稳定层 hands 不变。

## 架构

```
live_demo_s80c.py ──spawn──▶ s80c_depth_worker.py（SDK ctypes 子进程）
  S80CSource                     │ FAYS_VIK_GetStereoFrames（50fps 上下拼接帧）
  │ 读管道线程                    │ RGB→BGR → FAYS_ATRAK_D_FeedStereoImage
  │ 握手 JSON → align_calib      │ FAYS_ATRAK_D_GetDepthImage（米 → ×1000 mm）
  ▼                              │ cv2.fisheye remap 左右目 → P0/P1 矫正空间
_run_3d_chain（复用 D435 全链）◀─┼─ stdout 管道：[1B type][4B seq][8B ts]
  win_title="S80C live"          │   [4B w][4B h][4B len][payload]
                                 │   type=0 JSON 握手（P0 内参）/ type=4 raw
                                 │   BGR 矫正左目（默认，零编解码）/
                                 │   type=2 raw float32 mm 深度（仅新帧，
                                 │   demo 锁存）/ type=5 raw BGR 矫正右目
                                 │   （仅 --stereo-view，先于 type=4）；
                                 │   --pipe-format jpeg 回退 type=1/3
```

`--stereo-view` 时 win1 显示基底由 `S80CSource.display_frame()` 提供：
左右目并排拼接（右目同样经 K2/D2+R1/P1 鱼眼矫正到 P1 空间）。右目由
`_run_3d_chain` 独立跑一份 MediaPipe 检测（半分辨率 ~2.5ms）并渲染
2D 关键点到右半边（`_draw_hand` + x 偏移，经 `getattr(source,
"right_frame", None)` 配对）；深度/3D 仍只用左目。D435 source 无这两
个钩子 → 行为零变化。

**坐标空间**：rgb、2D 关键点、深度、3D 全部在引擎矫正左目 P0 空间
（1280×800，fx≈457）。worker 用 `cv2.fisheye.initUndistortRectifyMap`
(K1/D1 原始内参, R0/P0 引擎矫正矩阵) 把 raw 鱼眼左目 remap 到该空间，
`depth_to_color` 恒等变换（DepthAligner 自测证明恒等对齐逐点精确）——
无跨相机对齐误差。

**OpenCV 隔离**：SDK .so 无 DT_NEEDED、按 SDK 自带 OpenCV 4.2 编译
（cv::stereoRectify 传 MatExpr，系统 OpenCV 4.6+ 会崩），必须
RTLD_GLOBAL 预载 lib406。这部分只在 worker 子进程内做，demo 主进程
（venv cv2 5.0 + mediapipe）不受污染。lib406 内部依赖链用裸 SONAME
（libopencv_*.so.4.2、libwebp.so.6…）需 lib406 在动态链接器搜索
路径上——worker 启动时检测 LD_LIBRARY_PATH 缺失会 **re-exec 自身**
补上（stdout 管道 fd 跨 exec 保留），无论被 demo spawn 还是手动
单跑都自洽。

**相机内参分辨率**：SDK GetCalibrationParam 返回的标定分辨率是
640×400（fx=228.6），流是 1280×800 → worker 按流分辨率等比缩放内参
（fx 457.2 与 calib.yaml 1280×800 值完全吻合）；畸变系数为归一化量
不缩放。

## 已知边界

- **深度有效率**：鱼眼边缘纹理弱，深度有效率约 25%（中位 0.4m 场景量级）；
  无深度区域关键点走槽级 zc 补点/预测。
- **设备独占**：S80C 被主程序或其他 SDK 进程占用时，worker 初始化失败
  （5s 无帧看门狗自动退出，demo 重启一次后报错）。
- **`--rect-mode sdk`**：2D 视图改用引擎 GetRectifiedImage 左半（验证
  对照用）。若引擎矫正图尺寸非 1280×800，与深度网格不一致 → 3D 对齐
  失效（链的形状守卫会静默归零），仅用于对比 remap 与引擎矫正是否一致。
- 相机标定：worker 优先用 SDK `GetCalibrationParam` 的 cam0 内参，
  失败回退 `tools/fayssense_depth_sdk/calib/calib.yaml` cam0（equidistant KB4）。
  `--stereo-view` 另取 cam1 内参建右目矫正映射（K2/D2+R1/P1，yaml
  回退同构）；拿不到 cam1 时自动退回单目显示并在 stderr 告警。
- **FTDI 掉线/链路不稳**：相机偶发从 USB 总线整体消失（`lsusb` 无
  FTDI 设备，worker 报"未找到 FTDI 双目接口 1.0"）——重新插拔 USB
  恢复，软件无法自愈。连续多轮 init/teardown 后出现概率更高，排查先
  看 `lsusb`。链路不稳的另一个症状（2026-08-24 实测）：ViKit 枚举到
  "Number of cameras: 0" → 深度引擎 BindViKit 报 "no extrinsics
  found" 失败、或预热阶段偶发段错误——静置片刻/重插 USB 后自愈，
  与代码无关。
- **机型自动适配（2026-08-24）**：worker 启动时用
  `GetCalibrationParam` 读设备存储标定（K/D + T_cn_cnm1/T_cn_imu），
  **运行时合成引擎 kalibr yaml**（同 vendor dump 格式，内参 ×2 到
  1280×800）指给引擎——2D remap 与深度引擎永远同一台相机的标定，
  换相机自动跟随（S80C/S80M 均实测识别：fx≈457.2→S80C、464.8→S80M，
  日志 `[Calib] 设备标定合成引擎 yaml（机型 …）`）。只有
  `GetCalibrationParam` 失败（链路不稳）才回退 `--calib-yaml` 静态
  yaml。S80C 标定 dump 存档：
  `FaysSense_VI_Kit_Release/scripts/FS-VI80-S80C_3500000262190088_dump_calib.yaml`
  （fx=228.6@640×400，仅归档参考，运行时不读）。
- **MediaPipe GPU 不可用**：venv mediapipe 编译未开 GPU（冒烟实测
  "GPU processing is disabled in build flags"）→ `--delegate auto` 恒落
  CPU XNNPACK（~7ms/帧 @1280×800，50fps 预算内）；勿期待 GPU 加速。
- **裸手闪动**：主要由 raw 管道（零 JPEG 噪声）+ 50fps + 左右目
  独立检测器实例（见架构）解决。2D OneEuro 平滑默认**关**
  （`--smooth-2d` 开）——freq_min=3Hz 在窗口链 20-45ms/帧下滞后
  ~2-3 帧（~90ms），快动时骨架明显落后于手；要开须降 freq_min 并
  评估滞后。JPEG 管道（worker 单跑 `--pipe-format jpeg`）的压缩噪声
  会加重 MediaPipe 逐帧抖动，仅调试回退用。
- **固定世界视角（`--fixed-view`，2026-08-25 起默认关）**：3D 窗口
  默认与 D435 相同——相机目标/缩放随手动（质心锚定）；`--fixed-view`
  开启后首帧有手时锁存目标/缩放/网格（renderer `fixed_view` 参数），
  之后手在世界内自由运动；手走出锁定视场就出画（固定相机语义），按
  `r` 在当前手位置重锁。双手场景锁的是锁定时两手的共同质心；一只手
  退出/进入画面不再触发视角重居中。曾默认开启（2026-08-24），用户
  实机觉得视角不好已改回 D435 口径。D435 demo 不受影响（不传该参数，
  零变化）。
- **骨长约束（2026-08-24）**：`--max-bone-len 0.15` 默认开——以腕为
  根逐级钳制 3D 骨长，超限关节沿父→子方向缩回（父被钳后子边以钳后
  父为锚级联收紧，处理完所有显示骨长 ≤ 上限）。只作用于展示路径
  （平滑/质心锚定/渲染之前），不回改 tracker；正常手骨（最长腕→MCP
  ~0.12m）零改动。`--max-bone-len 0` 关。

## 画面撕裂排查（--tear-probe runbook）

**现状（2026-09-01）**：S80C/S80M 实时画面曾出现水平错位带；`--tear-probe`
导出帧**内有水平缝** → 撕裂在**数据/相机侧**（非显示层——本地显示器 +
同环境 D435 demo 不撕裂已排除显示栈）。worker/demo 管线自身防御到位
（单线程、返回即拷贝、管道保序、右先左后配对正确）。**主程序 v1.0.13
已按官方 GUI 同款回调取帧根治（25 档+回调用户确认不撕；50 档+回调
probe 机制全绿、目测待用户）**——demo 已移植同款全套并**默认开**
（回调取帧 + 迟到帧跳过 + 深度 feed A/B 双缓冲，见下「修复方案」）。
判据更新位置：本节「结论」小节。

**runbook**：

1. 同轮捕获：`./run_live_s80c.sh --tear-probe --raw-dump DIR --raw-ring
   160`（DIR 任意可写目录；raw 环默认仅 32 帧≈0.64s，加 `--raw-ring 160`
   使 raw 覆盖 tear 环全部 96 帧时段）——demo 导出**矫正**帧环
   （`tear_exit_*`），worker 退出时导出 **pre-remap 原始**帧环
   （`DIR/raw_<ts_ns>.jpg`，半尺寸）。
2. 保持手快速挥动 ≥60s，见撕裂 ≥2 次按 `q` 退出（无需掐时机）。
3. **t 键精确捕获（2026-08-31 起，推荐）**：见撕立即按 `t`——demo 导出
   `tear_dump_N`（显示侧帧环）并同步发 SIGUSR1 给 worker 导出
   `DIR/t_N/`（raw 环，与 tear 环同轮配对；S80C 缝细+突发稀疏，靠
   退出窗口采样撞大运不可靠，按时刻配对才是正路）。一次会话可多次
   按 t。S80C 缝比 S80M 细、半尺寸机器检测可能漏 → 加 `--raw-full`
   导出全尺寸（~300-500KB/帧）。
4. 浏览判定：`venv/bin/python tools/hand_3d_s80c/browse_tear_dump.py
   [目录]`——`d` diff 视图（缝=孤立水平亮带）、`h` 行差提示条
   （仅提示不做自动判定）、`m` 放大镜看像素级缝、`t` 标记证据帧
   （退出打印帧号清单）。
5. 证据矩阵：

| 导出帧观察 | 判定 |
|---|---|
| 全干净（diff 只显整帧均匀运动） | 显示/远程桌面层撕裂 |
| 帧内有水平缝（整条水平带与上下错位、带内是前一帧内容） | 数据/相机侧撕裂 |
| raw 有缝 | 相机/SDK 交付已撕裂（缝行位置固定=固件半帧拼装；随机=缓冲竞态） |
| raw 干净而矫正有缝 | remap 侧（对照 `--rect-mode sdk` 确认） |
| 斜向拉伸（随快速运动、静止无） | 滚动快门，相机固有 |

**修复方案（2026-09-01 起默认开）**：`--cb-bridge`（改用 SDK
RegisterStereoImageCallback 回调取帧=官方 stereo_depth_gui 同款路径，
SDK 装配线程写完帧才回调，绕过 GetStereoFrames 内部拷贝与装配的竞态；
经 third_party/cb_bridge/ C++ 桥接 std::function→C ABI）+ **迟到帧跳过**
（USB 重传批 120ms 回跳整 3 帧，回调流暴露、跳过保时间戳单调）+
**深度 feed A/B 双缓冲**（引擎异步读时单缓冲 memmove 覆盖嫌疑）——
三项全套=主程序 v1.0.13 已验证方案（用户确认 25+回调不撕、50+回调
probe 机制全绿；**demo 同款 2026-09-01 实机确认：回调不撕、轮询撕**）。
注册失败/桥接缺失自动回退轮询；`--no-cb-bridge` 手动回退对照（回退
即回到撕裂路径，仅诊断用）。

**诊断 flag（实验，默认全关）**：`--pipe-format jpeg`（排除管道层）、
`--race-probe`（每秒日志统计拷贝窗口内缓冲被改帧数——坐实/排除
SDK 写缓冲与拷贝竞态）、`--double-buffer`（交替 data 指针 A/B，仅
race-probe 出正信号后试；SDK 若缓存指针会引向旧缓冲更糟）、
`--settle-poll`（修复候选，**2026-08-31 实机已证无效**——带率
21.0%→18.2% 纹丝未动，仅留诊断）、`--raw-dump DIR`（pre-remap 原始
帧环，退出时导出 JPEG——判定"缝在 raw 还是 remap 后"的终局判据）、
`--vikit-config tools/stereo_s80m/config/fays_vikit.yaml`（25fps 对照，
时序敏感判据；**仅诊断**——用户已定不接受降帧运营）。

**结论**：
- **R1（2026-08-31，S80M，50fps 默认）**：用户标记 4 帧（21/41/65/85
  = frame_0607/0627/0651/0671），缝位置 正中/下方/下方/上方——
  **位置不定，非固定行半帧拼装**。定量：标记帧与邻帧存在两目共模的
  细全宽带（行 32-37、104-107、205-208、343-399 等）——**raw 侧
  坐实数据侧撕裂**（与用户判据一致）。严格位移台阶（跳变 ≥6px+持续
  2 带+两目共模）：raw/remap 均 0 处（缝处 dx 差弱或快速挥手噪声
  淹没）。
- **R1b（S80M，50fps，`--raw-ring 160 --race-probe`）**：用户 raw 帧
  标记 7 帧（3/7/30/34/67/112/140 = raw_501…/502…/507…/508…/515…/
  524…/530…），3,7 与 30,34 两两相距 ~0.1s——**缝持续 ~100ms 级即
  消、复发间隔 0.5-1s 级不规则**；机器共模细带率 30/143 帧对 =
  21.0%（行 23-134）。race-probe 统计随终端关闭丢失（不计）。
- **R1c（S80M，50fps，`--settle-poll`）**：机器共模细带率 24/132
  帧对 = 18.2%（行 47-286）——与基线 21.0% **无差异**；用户确认
  "还是有撕裂"。**"SDK 返回早于其写入完成"假设死亡：缝在 SDK 交付
  之前已在帧内形成**（SDK 内部拼装竞态或相机侧曝光/读出时序——两目
  同行共模=硬件同步、位置漂移=时序抖动，S80M 更重与固件差异一致）。
- **官方 GUI 对照（2026-08-31 用户观察）**：官方 stereo_depth_gui
  实时画面**无撕裂**。核实其源码：GUI 用
  **FAYS_VIK_RegisterStereoImageCallback 回调取帧**（SDK 装配线程写
  完帧才回调，live_sensor.cpp 回调内直接拷走），而 worker 用
  GetStereoFrames 轮询拷进自有缓冲——曾疑缝在轮询的内部拷贝与
  SDK 装配写帧的竞态。**但 config diff 显示 GUI 与我们在三轴不同：
  fps 25vs50、AE 自动曝光(-1)vs固定 400/6、color_mode 0vs1**——
  fps 因素当时未隔离。
- **R7 回调取帧（2026-08-31，S80C，50fps `--cb-bridge`）**：用户
  **仍有明显撕裂**。回调=SDK 推送（装配完成才回调），轮询=拉取
  内拷贝——两条投递路径同撕。**（2026-09-01 复评）**：R7/R8/R11
  当时的 cb 路径**尚无迟到帧跳过、无深度 feed A/B 双缓冲**（两项
  均为主程序 v1.0.13 同批移植）——"投递 API 无关、缝在 SDK 相机
  线程交付的帧本身"的结论仅适用于**裸回调**，对全套方案不成立。
- **R2 25fps（2026-08-31，S80C，3.9.0 轮询）**：`--vikit-config
  tools/stereo_s80m/config/fays_vikit.yaml`（与 50fps yaml 仅 fps
  不同）——用户观察**无撕裂**（"两个隔离实验都不撕裂"之一）。
- **GUI@50fps（2026-08-31，S80C，3.9.1）——已被源码推翻**：官方
  GUI 加载我们的 50fps yaml（端口改好、固定曝光 400/6、color_mode
  1）——用户观察**无撕裂无卡顿**（"这个不卡顿"）。**但 GUI 源码
  钉死：main_window.cpp:83-85 `mTimer = new QTimer(this); ...
  mTimer->start(40)`——显示循环硬编码 40ms=25fps**，与 yaml 无关。
  "GUI@50"实为"SDK 50fps 交付 + 25fps 消费"（回调照常 50 次/秒
  拷进 latestStereo，Qt 每 40ms 只取最新一帧显示）。**25fps 消费
  干净与 R2 完全一致——"50fps 干净对照组"从未存在**，fps 非唯一
  因子的判读作废。GUI 窗口 FPS 读数（grabFrame 计数）应显示
  ~25.0，可现场核实。
- **R8 SDK 版本对照（2026-08-31，S80C，50fps）**：demo + 外部
  3.9.1（`--sdk-dir .../Release`）+ `--cb-bridge`——用户**仍然撕裂**。
  → **SDK 版本排除**（3.9.0/3.9.1 同撕）。投递 API 排除（R7）。fps
  排除（GUI@50 不撕）。**现役矩阵**：
  | 实验 | fps | SDK | 取帧 | 消费侧 | 结果 |
  |---|---|---|---|---|---|
  | GUI 自有 config | 25 | 3.9.1 | 回调 | GUI C++ | 无撕 |
  | GUI + 我们 50 yaml | 50 | 3.9.1 | 回调 | GUI C++ | 无撕 |
  | demo 轮询 | 50 | 3.9.0 | 轮询 | worker | 撕 |
  | demo 回调 R7 | 50 | 3.9.0 | 回调 | worker | 撕 |
  | demo 回调 R8 | 50 | 3.9.1 | 回调 | worker | 撕 |
  | demo R2 | 25 | 3.9.0 | 轮询 | worker | 无撕 |
- **R11 OpenCV 配组对照（2026-08-31，S80C，50fps）**：demo +
  外部 3.9.1 + `--cb-bridge` + `--opencv-dir` 指向外部 SDK 自带
  OpenCV 4.2（GUI 精确同款配组）——用户**仍然撕裂**。→ **OpenCV
  运行时排除**。cb_bridge 回调体内仅 ~1ms memmove（非阻塞，槽满
  即覆写不阻塞 SDK 线程），GUI 回调体反而更重（cvtColor+copyTo
  数 ms）且干净——**回调内耗时不是触发点**。
  **现役矩阵**：
  | 实验 | SDK 交付 | 消费 fps | SDK | 取帧 | 消费侧 | 结果 |
  |---|---|---|---|---|---|---|
  | GUI 自有 config | 50 | **25**（QTimer 40ms） | 3.9.1 | 回调 | GUI C++ | 无撕 |
  | GUI + 我们 50 yaml | 50 | **25**（QTimer 40ms） | 3.9.1 | 回调 | GUI C++ | 无撕 |
  | demo 轮询 R1 | 50 | 50 | 3.9.0 | 轮询 | worker | 撕 |
  | demo 回调 R7 | 50 | 50 | 3.9.0 | 回调 | worker | 撕 |
  | demo 回调 R8 | 50 | 50 | 3.9.1 | 回调 | worker | 撕 |
  | demo 回调 R11 | 50 | 50 | 3.9.1 | 回调 | worker+外部 OpenCV | 撕 |
  | demo R2 | 25 | 25 | 3.9.0 | 轮询 | worker | 无撕 |
  | 主程序 25+全套 | 25 | 25 | 3.9.0 | 回调+迟到跳过+A/B feed | 轻子进程→管道→主程序 | **无撕（用户确认）** |
  | 主程序 50+全套 | 50 | 50 | 3.9.0 | 回调+迟到跳过+A/B feed | 轻子进程→管道→主程序 | 待目测（probe 机制全绿） |
  | demo 全套 R12 | 50 | 50 | 3.9.0 | 回调+迟到跳过+A/B feed | worker | **无撕（用户确认）** |
  | demo 轮询 R12 对照 | 50 | 50 | 3.9.0 | 轮询（--no-cb-bridge） | worker | 撕 |
  **2026-09-01 R12 实机裁决**：demo 全套 50fps **不撕**、`--no-cb-bridge`
  轮询同轮**撕**——A/B 干净。旧剩余嫌疑双结案：**(a)** 50fps 全帧率
  消费负载假说死亡（R12 全套=worker Python 重链 50fps 消费，与 GUI
  C++ 轻链同样不撕）；**(b)** worker 后处理链无责（全套与轮询同一
  条后处理链，仅取帧路径不同）。根因钉死=**轮询 GetStereoFrames
  内部拷贝与 SDK 装配写帧的竞态**，回调（装配完成才交付）+迟到跳过
  +A/B feed 全套根治。迟到跳过与 A/B feed 各自的独立贡献未隔离
  （不拆件验证——用户已确认全套效果）。t 键配对捕获保留为将来
  复发时的终局判据。
- **R10 负载梯度（2026-08-31，S80C，50fps `--rect-mode sdk`）**：
  用户观察**仍撕裂但频率下降**。判读谨慎：sdk 模式显示来自深度
  引擎矫正缓冲（更新率 ~20fps + 鱼眼→矩形几何翘曲会把水平缝带
  拉弯变淡）——"频率下降"含显示侧混淆（更少的不同帧/秒 + 缝带
  变难辨认），不能单独坐实负载假说；但缝在引擎产出的帧里仍存在
  与 R1 raw 证据一致（数据侧）。**GUI 源码核实**：info_bar 有
  "FPS: xx.x" 读数（grabFrame 计数，非相机流率）——GUI@50 再跑
  一次读该数字即可判定 (c)：~50=低负载消费侧 50fps 无撕（负载
  假说方向），~25=GUI 实为 25fps（50fps 必撕假说复活，指向厂商
  固件）。
- **R12 全套方案默认开（2026-09-01 实机确认）**：demo worker 移植
  主程序 v1.0.13 三项全套并**默认开**——cb-bridge 回调取帧 + 迟到帧
  跳过（USB 重传批 120ms 回跳整 3 帧，回调流暴露） + 深度 feed A/B
  双缓冲。桥接缺失自动回退轮询；`--no-cb-bridge` 手动回退对照。
  **实机 A/B 裁决（2026-09-01）**：`./run_live_s80c.sh`（默认全套
  50fps）**不撕**；`--no-cb-bridge` 同轮**撕**——根治成立，默认路径
  即修复。主程序 50+全套组合的录制目测仍待用户（probe 机制全绿 +
  demo 同款实机确认，预期同好）。
- **终局判据（t 键配对捕获，已实现）**：`./run_live_s80c.sh
  --cb-bridge --raw-dump DIR --tear-probe --raw-full`——见撕即按
  t：同一轮内导出 tear 环（显示侧）+ raw 环（pre-remap，全尺寸
  S80C 细缝可辨）。**raw 有缝=SDK 在 50fps 消费下交付撕裂帧**
  （负载触发 SDK/相机内部竞态，证据链完备→报厂商，软件侧穷尽）；
  **raw 干净=缝在我们后处理链（remap/管道/demo 合成）**，可修——
  把重链移出取帧进程（demo 侧处理或独立进程）保 50fps 取帧节奏。

## 验证记录

- worker 单跑（2026-08-24 实测）：`venv/bin/python tools/hand_3d_s80c/
  s80c_depth_worker.py` —— ViKit 3.9.0 打开、端口自动解析
  (stereo=/dev/video0)、深度引擎 Bind 成功、P0 fx'=502.4、握手 JSON
  发出、25fps 相机 / ~20fps 深度新帧 / 有效 26%（stderr 每秒统计；
  stdout 是二进制流，勿直接重定向到终端）。SIGTERM 优雅退出、SDK
  句柄释放无残留。
- demo 错误路径（相机未插）：worker BindViKit 失败退出 → demo 提前
  报错（exit code + 排查提示）→ 自动重试一次 → 干净退出，无僵尸进程。
- `--stereo-view` 离线验证（2026-08-24）：worker 单跑 25s 捕获管道
  （2.4GB）→ 用 demo 真实 reader 逻辑回放——type=3 右目帧到达、左右
  各 1280×800、`display_frame` 并排 2560×800、两半视差正确（右半近处
  物体相对左移）、深度 45% 有效。链路完整，实机窗口待相机插回目测。
- 50fps 满帧（2026-08-24 headless 实测）：`--no-window --stereo-view`
  跑 40s，worker 相机稳定 48.6-50.5fps（优化前 demo 消费 36-40fps、
  最初 25fps），深度新帧 18-22 张/秒、有效 ~40%。优化项：raw BGR
  管道（杀 JPEG 编解码）、深度仅新帧发送（demo 锁存）、右目半分辨率
  检测、`--fill 0` 恒等对齐纯裁剪。
- 实机窗口（2026-08-24 实测）：`--stereo-view` 多轮用户目测，每轮
  q 干净退出（相机 49-50fps 稳定）。迭代：①2D 平滑滞后致偏移→默认关；
  ②关键点偏移+闪烁根因=VIDEO 模式检测器实例跨流共享→右目独立实例
  （det 17.8→23.7ms，右目不再蹭左目跟踪状态、老实跑满检测）；
  ③2D 显示改原始检测直绘（--raw-2d）。窗口提速项 det-scale 0.5 +
  显示副本降半（链 22-28fps 带手）。最终验收（关键点贴合双手、
  无闪动）由用户确认。
- 黑手套模式修复（2026-08-24，用户报"效果很差"）：右目 MediaPipe
  死路 → 左目平滑框视差平移 + 共享 pose 后端（见上"黑手套模式"）；
  raw-2d 扩展到手套模式解耦 3D 槽位链。辅助函数（视差平移/运动门控）
  单测已过；右目实机已确认。
- 左目手套渲染修复（2026-08-24，用户报"左目比右目差很多=偏移/形状
  不对"）：左目 2D 显示改与右目同口径（detect() 原始 pose +
  轻平滑，见上"黑手套模式"第 3 点）。glove_detector.py 加
  `last_raw_pose()` 只读访问器（detect() 内原始点缓存，稳定层之前、
  与返回值同序），live_demo.py raw-2d 手套分支画 `_glove_l_disp`。
  py_compile + import 冒烟已过；实机验收待用户。
- 双目默认开 + 固定世界视角 + 骨长约束（2026-08-24，用户三项需求）：
  ①`--stereo-view` 反转默认——参数改 `--no-stereo-view`（store_false），
  headless 实测握手行 `stereo-view=on`、右目矫正映射照常建立；
  ②3D 固定世界视角——renderer_3d.py 加 `view_params()` +
  `render(fixed_view=…)` 可选参数（默认 None 行为不变，D435 零变化），
  live_demo `_run_3d_chain(..., fixed_view3d=)` 首帧锁存目标/缩放/网格、
  r 重锁；③骨长约束默认 0.15m——`_clamp_bone_lengths` 以腕为根层级
  钳制（父钳后子边级联，全链骨长 ≤ 上限），仅展示路径不回 tracker。
  单测：离群 0.3/0.5m → 0.15m 方向保持、正常手零改动、NaN 槽不动、
  级联不变量；headless 30s `--export` 实跑 render.mp4 695 帧 1280×720
  无错误。**2026-08-25 ②视角改回 D435 口径（用户实机觉得固定视角
  不好）**——S80C 固定视角默认关、参数改 `--fixed-view` 可选开，
  D435 不变。
