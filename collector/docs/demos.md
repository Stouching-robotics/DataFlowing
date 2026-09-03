# tools/demos/

## 定位

`tools/demos/` 汇集面向客户交付与开发者自检的演示程序：HDF5 录制回放查看器、UVC 相机 HSV/触觉可视化工具、双目 2D 手部关键点 demo、黑手套检测对比工具，以及 S80M 双目深度 demo（Python 与 C++ 两版）。

本目录独立于主程序运行。全库检索（排除 `tools/demos/` 自身与 `docs/`）仅 `tools/hand_3d_s80c/s80c_depth_worker.py:118` 注释引用 `tools/demos/run_stereo_depth_demo.sh`；主程序中的「双目 demo」字样（`ui/main_window.py:703,711,759`、`config/i18n.py:100`）实际指自包含模块 `tools/stereo_s80m/read_stereo_rgb.py`（路径常量 `STEREO_DEMO` 在 `core/s80m_manager.py:40`），与 `tools/demos/` 无关；`config/settings.py:86` 的深度显示下限注释（0.3m）参考的是本目录 S80M 深度 demo 的口径。其中 `tools/demos/demo_glove_kpts/` 反向复用了主程序的 `core/hand_tracking.py` 代码路径。

## 文件清单

| 文件/子目录 | 一句话作用 | 运行方式 |
|---|---|---|
| `tools/demos/demo_glove_kpts/compare_detectors.py` | 黑手套检测器 A/B 对比（对照检测器 vs 新训练模型），输出对比视频与指标 CSV | `python tools/demos/demo_glove_kpts/compare_detectors.py [--old world\|.pt] [--new .pt]` |
| `tools/demos/demo_glove_kpts/demo_glove_video.py` | 单视频黑手套关键点识别 + 渲染（复用主程序 core/ 代码路径） | `python tools/demos/demo_glove_kpts/demo_glove_video.py [--video PATH]` |
| `tools/demos/demo_glove_kpts/D435_depth_rgb_kpts.mp4` | 示例渲染输出视频（运行 `demo_glove_video.py` 后生成，约 10 MB；当前仓库内未保留） | — |
| `tools/demos/hdf5_demo_v1.4/hdf5_demo.py` | HDF5 录制四宫格回放查看器（双目视频/渲染视频/仿生手掌/触觉热力图/IMU，单文件自包含） | `python tools/demos/hdf5_demo_v1.4/hdf5_demo.py [data.h5]` |
| `tools/demos/hdf5_demo_v1.4/README.md`、`HELP.md` | 英文 / 中文使用说明 | — |
| `tools/demos/hdf5_demo_v1.4/requirements.txt` | 依赖：`numpy`、`h5py`、`PyQt5`、`opencv-python` | `pip install -r tools/demos/hdf5_demo_v1.4/requirements.txt` |
| `tools/demos/HSV_Visualizer_V4.4.1.py` | UVC 相机 HSV/RGB 通道可视化 + ROI 统计 + RGB/Hue 时序曲线 + 平面触觉热力图链路复刻 | `python tools/demos/HSV_Visualizer_V4.4.1.py` |
| `tools/demos/stereo_2d_demo/stereo_2d_demo.py` | 双目/单目视频 2D 手部关键点检测 + 渲染（单文件自包含） | `python tools/demos/stereo_2d_demo/stereo_2d_demo.py left.mp4 right.mp4 -o out.mp4` |
| `tools/demos/stereo_2d_demo/hand_landmarker.task` | MediaPipe 手部关键点模型权重（约 7.5 MB，须与脚本同目录） | `--model` 可改路径 |
| `tools/demos/stereo_2d_demo/run.sh`、`requirements.txt`、`README.md` | 一键运行脚本（自动检测本地 venv）/ 依赖 / 中文说明 | `tools/demos/stereo_2d_demo/run.sh left.mp4 right.mp4` |
| `tools/demos/test_stereo_depth_calib.py` | S80M 双目深度 demo（Python ctypes 调用 FaysSense SDK 深度引擎） | 必须经启动器：`./tools/demos/run_stereo_depth_demo.sh` |
| `tools/demos/run_stereo_depth_demo.sh` | 深度 demo 启动器（设置 `LD_LIBRARY_PATH` 绑定 SDK 自带 OpenCV 4.2） | `FAYSSENSE_SDK_DIR=<SDK目录> ./tools/demos/run_stereo_depth_demo.sh` |
| `tools/demos/stereo_depth_640x400.yaml` | 640×400 分辨率的双目深度引擎配置样例 | 参考用；运行时读取 SDK 目录内配置 |
| `tools/demos/stereo_depth_demo/` | C++ 版 S80M 双目深度 demo（`main.cpp` + `CMakeLists.txt` + `build.sh` + `run.sh`） | 目录内 `./build.sh && ./run.sh`（需 SDK） |

## 各文件详解

### `tools/demos/demo_glove_kpts/demo_glove_video.py`

**作用**：对单个视频逐帧跑黑手套检测管线（YOLO 检测黑手套 + RTMPose 21 点关键点），叠加框/Hand #ID/分色骨架/伸指信息后写输出视频，可选实时显示窗口。完全复用主程序两条现成代码路径：`core/hand_processor.py` 内部 `process_session()` 的 glove 分支与 `core/hand_tracking.draw_kpts_overlay()` 渲染，行为与 GUI 会话后处理一致。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `_find_ffmpeg` | `() -> str` | 按优先级找可用 ffmpeg（lerobot 环境 → PATH → miniconda base） | 返回路径，找不到返回 `""` |
| `build_pipeline` | `(det_device, pose_device, detector="")` | 构造 `HandPipeline`；CUDA 不可用时自动回退 CPU | 返回 `HandPipeline` 实例 |
| `main` | — | 解析参数、逐帧推理渲染、写临时 mpeg4 后用 ffmpeg 转 H.264 | 返回退出码；输出 `<源文件名>_kpts.mp4` |

**关键数据**：
- 输入：任意视频；默认 `data/recordings/Project_Test10/Project_Test10_000003/videos/D435_depth_rgb/chunk-0000/D435_depth_rgb.mp4`。
- 默认检测器权重：`tools/hand_detection/best.pt`（`--detector` 可改）。
- 命令行参数：`--video`（默认上述路径）、`--out`（默认写入本 demo 目录 `<源文件名>_kpts.mp4`）、`--detector`、`--start 0`、`--frames 0`（0=到结尾）、`--det-device cuda`、`--pose-device cuda`、`--no-display`、`--no-transcode`（跳过转码，直接输出 cv2 的 mpeg4）。
- 输出：cv2 先写 `<out>.mpeg4.tmp.mp4`（mp4v 编码），结束用 ffmpeg 转 `libx264 -crf 20 yuv420p +faststart`；无 ffmpeg 时保留 mpeg4。
- 显示窗口按键：`q`/`Esc` 退出，空格 暂停/继续；按源帧率限速。

**调用关系**：不被主程序调用（无外部引用）；导入 `core/hand_tracking`（`_lazy_import_pipeline`、`_pack_hand_data`、`draw_kpts_overlay`，`core/hand_tracking.py` 经 `core/hand_processor.py` 间接与主程序共用）。被同目录 `compare_detectors.py` 导入复用。

### `tools/demos/demo_glove_kpts/compare_detectors.py`

**作用**：黑手套检测器 A/B 对比工具。两个参数完全相同的 `HandPipeline`（仅检测器不同，`det_imgsz`/conf/追踪/姿态门/渲染链与主程序部署口径一致）对同一段视频逐帧各跑一遍，输出各自渲染视频、左右并排对比视频和指标 CSV。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `ModelStats` | `__init__(label, weights)`；`add(boxes, kpts, scores, track_ids)`；`summary()` | 单模型逐帧指标收集（基于 `process()` 门后输出，即用户实际看到的渲染结果） | `summary()` 返回指标 dict |
| `_as_pt` | `(path, out_dir)` | ultralytics 只认 `.pt` 后缀；非 `.pt` 权重复制成临时 `.pt` | 返回可用路径 |
| `_transcode` | `(tmp, out, ffmpeg)` | mpeg4 临时文件 → H.264（与 `demo_glove_video.py` 相同配方，`crf 20`） | 返回是否转码成功；无 ffmpeg 时原样改名 |
| `main` | — | 建两条管线、逐帧推理 + 并排渲染、转码、打印并写指标表 | 输出 `old.mp4`/`new.mp4`/`side_by_side.mp4`/`compare_report.csv` |

**关键数据**：
- `--old` 默认 `"world"`：内置 YOLO-World 零样本（prompt `["hand","glove"]`，imgsz 320，权重 `yolov8m-worldv2.pt`）；也可传 `.pt` 路径对比旧训练模型。
- `--new` 默认 `tools/glove_package/runs/hand_det/weights/best.pt`。
- `--out-dir` 默认 `keypoints_output/ab_compare`；`--video` 默认同 `demo_glove_video.py` 的 `DEFAULT_VIDEO`。
- 其余参数：`--start 0`、`--frames 0`、`--det-device cuda`、`--pose-device cuda`、`--display`（弹出实时窗口，`q` 退出）、`--no-transcode`。
- 并排视频：顶部 30px 标签条（`LABEL_H`，按偶数对齐保证 yuv420p），左对照右新，下方帧号。
- 指标口径（`compare_report.csv`）：检出帧率（boxes 非空帧占比）、单手率、kpt 均值中位数（被渲染手部 21 点置信度均值的中位数）、腕点（点 0）抖动 p50/p95/max（px）、ID 切换次数、平均 track 寿命（帧）。

**调用关系**：不被主程序调用；从同目录 `demo_glove_video` 导入 `build_pipeline`、`draw_kpts_overlay`、`_pack_hand_data`、`_find_ffmpeg`、`DEFAULT_VIDEO`；间接依赖 `core/hand_tracking` 与 `tools/glove_package` 训练产物。

### `tools/demos/hdf5_demo_v1.4/hdf5_demo.py`

**作用**：单文件自包含的 HDF5 数据四宫格演示播放器（可直接发给客户）。一个窗口同时显示：左上双目原始视频（可同时拼单目）、右上 h5 内已渲染好的视频、左下仿生手掌（16×16 触觉矩阵按部位映射）、右下触觉热力图、底部 IMU 加速度/角速度波形。触觉与 IMU 面板只在文件里确有该数据时显示。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `render_bionic_hand` | `(mat, window_size=(640,400), vmax=None)` | 2D 手掌骨架 + 16×16 触觉矩阵按部位映射 viridis 色块（锚点基于 1280×720 等比缩放） | 返回 BGR 画布 |
| `make_heatmap` | `(mat, size=300, cmap=COLORMAP_JET, vmax=None)` | 16×16 压力矩阵 → 带网格线的热力图 | 返回热力图 BGR |
| `put_label` | `(img, text, pos, color)` | 叠加带黑底标签文字；无效输入直接忽略 | 就地修改 |
| `render_imu` | `(seq, counts, idx, window=240, w=880, h=185)` | IMU 6 通道折线（上 acc/下 gyr）+ 当前帧竖线 + 右侧数值 | 返回 IMU 波形画布 |
| `RenderedVideo` | `__init__(ds)`、`__len__()`、`__getitem__(i)`、`close()` | h5 内 `videos/hand_skeleton` 的 MP4 原始字节 → 临时文件 + `cv2.VideoCapture` 按帧解码；顺序播放走顺序 read，跳转才 seek 且只缓存跳转帧（`_CACHE_MAX=30`） | 帧 BGR 数组或 `None` |
| `H5Data` | `__init__(path)`、`_find_rendered()`、`close()` | 惰性读取 hdf5，探测双目/单目视频、渲染视频、触觉、IMU、播放帧率 | 属性：`imgs`、`rendered`、`tactile`、`fps`、`capture_fps`、`tactile_vmax`、`imu_seq`/`imu_counts` |
| `DemoWindow` | `__init__(h5_path=None)`、`load(path)`、`open_file_dialog()`、`toggle_play()`、`next_frame()`、`seek(i)`、`render_frame(idx)`、`_show_image(label, bgr)`（静态） | PyQt5 主窗口：四宫格 + 控制条（Open h5/Play/Pause/单帧步进/滑块），`QTimer` 40ms（25fps），渲染慢于时钟时自动降速 | 渲染各面板像素图 |
| `main` | `h5` 为可选位置参数 | 启动 QApplication | — |

**关键数据**（读取的数据集，均在 `episode_*` 组内）：
- `observation/images/stereo_left`、`stereo_right`：`(N, 800, 1280, 3)` uint8 RGB（必需，缺失报错）。
- 单目数据集名探测（可与双目并存）：`left`/`mono`/`monocular`/`rgb`/`color`/`cam0`/`camera`/`image`/`frame`/`left_camera`/`camera_left`/`head_left_rgb`/`head_rgb`/`front_rgb`/`front`。
- `videos/hand_skeleton`：MP4 字节（优先作为渲染视频源）；兜底探测 `images` 组内名称含 `RENDERED_KEYWORDS`（`preview`/`keypoint`/`annot`/`render`/`overlay`/`visual`/`vis`/`kp`/`skeleton`/`slam`）的帧数组。
- `observation/tactile/left`、`right`：`(N, 16, 16)` float（全零视为无数据，面板隐藏）。
- `observation/imu`：`(M, 6)` float（acc xyz + gyr xyz）；`observation/imu_frame_index`：`(M,)` int 对应帧号；查看器自动按帧重新聚合（每帧取第一个样本作图 + 统计样本数）。
- 播放帧率：`meta/info` attrs `capture_fps` 优先，其次 `fps`，再其次数据集 attrs `fps`，默认 25。
- 运行：`python tools/demos/hdf5_demo_v1.4/hdf5_demo.py [data.h5]`，不带参数启动后点 Open h5 选择。

**调用关系**：不被任何其他代码调用（独立 GUI 单文件）；仅依赖 `numpy`/`h5py`/`cv2`/`PyQt5`。`HELP.md` 含详细使用说明与 FAQ。

### `tools/demos/HSV_Visualizer_V4.4.1.py`

**作用**：UVC 相机实时可视化工具。同一窗口 3 行拼接显示原图/H/S/V/R/G/B 七个通道与平面触觉热力图，底部 RGB+Hue 时序曲线；支持双 ROI 均值/方差统计、相机参数实时调节（滑块直写 `cv2.VideoCapture`）、传感器 ROI 裁切、热力图链路参数设置（复刻 `touch_sensor.py` 的 H 基线差分 → mask → 阈值 → 力标定链路）、断线自动重连。无命令行参数，全部经 GUI 操作。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `open_camera` | `(port)` | 打开摄像头；Linux 固定 V4L2 + MJPG 640×480@30fps，Windows DirectShow | 返回 `cv2.VideoCapture` |
| `ROIDialog` | `create_slider_with_label(...)`、`update_params()`、`reset_to_defaults()` | ROI 1（绿）/ROI 2（红）的 x/y/w/h 滑块弹窗，默认 r1=(275,320,50,55)、r2=(275,120,50,55) | 回写 `parent.roi_params` |
| `CameraParamsDialog` | `_on_slider`/`_on_spin`/`_write_camera(prop_id, value)`、`_on_auto_exp_toggled`、`_on_auto_wb_toggled`、`_reset_defaults` | 9 项相机参数滑块 + 自动曝光/白平衡复选框，拖动即时写入相机；Linux 曝光值 log2 秒 ↔ V4L2 100µs 互转，自动曝光极性 V4L2 与 Windows 相反 | 实时修改相机参数 |
| `HeatmapSettingsDialog` | `_on_scale_slider`/`_on_scale_spin`/`_reset_defaults`/`_confirm` | 热力图缩放/Mask B/Mask R/Hue 阈值/标定系数设置，默认 `scale=10.0`、`mask_b=50`、`mask_r=130`、`hue_threshold=5.0`、`fz_p1=1.0` | `_confirm` 回写主窗口 |
| `MainWindow` | `calibrate_baseline(num_frames=5)`、`compute_fz_matrix(frame)`、`update_frame()`、`draw_heatmap(fz_matrix, cell_w, cell_h)`、`draw_rgb_curve(cell_w, cell_h)`、`change_port()`、`reconnect_camera()`、`_do_retry_reconnect()`、`_setup_camera_defaults()`、`keyPressEvent(event)`、鼠标事件、`closeEvent(event)` | 主窗口：30ms 定时刷新；`compute_fz_matrix` 复刻 H 提取→基线差分→mask→阈值过滤→`×fz_p1` 链路；自动重连阈值 30 连败帧，重试间隔 67 帧（约 2s）；右键拖拽可在任一子图内画 ROI2 | 更新 UI；`R` 键触发重连 |

**关键数据**：
- 相机参数列表（`CAM_PARAMS`）：亮度 0-255、对比度 0-255、色调 0-180、饱和度 0-255、锐度 0-255、伽马 1-500、增益 16-248、曝光 -13 ~ -1、白平衡 2800-6500K。
- 主窗口默认 `roi_params`：`r1=(150,200,50,55)`、`r2=(275,120,200,200)`；`sensor_roi` 默认全帧 `(0,640,0,480)`。
- 打开相机后自动关闭自动曝光/自动白平衡并校准 H 基线；统计值每 10 帧（约每秒 3 次）刷新一次；R/G/B/H 均值历史最近 500 帧。
- 输出：无文件输出，纯实时显示。

**调用关系**：不被其他代码调用（独立 GUI）；依赖 `numpy`/`cv2`/`PyQt5`。文件内注释多处标注其热力图链路「复刻 `touch_sensor.py:516` / `touch_sensor.py:702`」及 `sightac_sdk` 逻辑——该模块属于外部 sightac SDK，本仓库内不存在同名文件，仅作为算法口径参考。

### `tools/demos/stereo_2d_demo/stereo_2d_demo.py`

**作用**：双目（或单目）视频 2D 手部关键点检测 + 渲染演示。自包含单文件：不依赖仓库任何其他模块，MediaPipe 检测、One-Euro 平滑、五指分色绘制、手势识别、视频合成与 H.264 转码全部在本文件内。模型文件 `hand_landmarker.task` 须与本文件同目录（缺失时报错并给出官方下载地址）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `_angle_between` | `(p_prev, p_vertex, p_next)` | 顶点处两骨段夹角（度，180°=伸直） | 返回角度 |
| `_compute_joint_angles` | `(pts)` | 21 点 `(21,2)` → 每关节角度 | 返回 `{(finger, joint): 度}` |
| `_count_extended_fingers` | `(angles)` | 角度 → 伸直手指名列表（空=握拳） | 返回手指名列表 |
| `_OneEuroFilter` | `__init__(freq_min=5.0, beta=0.05, dcutoff=1.0)`、`__call__(x, ts_ms)` | 单值 One-Euro 自适应低通 | 返回滤波值 |
| `_OneEuroFilter2D` | `__init__(freq_min=5.0, beta=0.05, dcutoff=1.0)`、`__call__(x, y, ts_ms)` | x/y 分量独立滤波 | 返回 `(x_hat, y_hat)` |
| `HandDetector` | `__init__(model_path=MODEL_PATH, num_hands=2, mirror=False, smooth=True, freq_min=15.0, beta=0.6, dcutoff=1.0)`；`process(frame_bgr)`；`close()` | MediaPipe Tasks API VIDEO 模式手部检测，置信度门限均为 0.5，可选 One-Euro 平滑 | `process` 返回 `[[(x,y)×21], ...]` 像素关键点 |
| `_draw_hand` | `(frame, pts)` | 五指分色骨架：掌心灰连接、腕白圆、指尖 7px/关节 5px + 深色描边 | 就地绘制 |
| `_draw_kpts_overlay` | `(frame, hands, cam_label="", frame_idx=0, total=0)` | 包围框 + 底衬标签 + 分色骨架 + 手势文本（`open: 手指名`/`fist`） | 返回叠加后的帧 |
| `_create_writer` | `(out_path, fps, width, height)` | 创建 mp4v 临时写器 | 返回 `(writer, tmp_path)` |
| `_find_working_ffmpeg` | `()` | 探测可运行 ffmpeg（存在但损坏的候选会被跳过） | 返回路径或 `None` |
| `_finalize` | `(writer, tmp_path, out_path)` | 释放写器后用 ffmpeg 转 `libx264 -crf 23 yuv420p`；无 ffmpeg 则 mp4v 原样交付 | 返回最终输出路径 |
| `process_video` | `(streams, output, detector, cam_labels)` | 多流逐帧检测 + 并排渲染（`hconcat`） | 写输出视频 |
| `main` | 位置参数 `videos`（1 或 2 个） | 解析参数、建检测器、跑 `process_video` | 退出码 |

**关键数据**：
- 输入：1 个视频=单目；2 个视频=左右目并排输出（最多 2 个，超出报错）。帧数取各流最小值。
- 输出：默认 `stereo_2d_output.mp4`；帧叠加：青绿包围框 + `Hand #0/#1` 标签 + 五指分色（拇指橙、食指绿、中指黄、无名指紫、小指蓝）+ 手势文本。
- 命令行参数：`-o/--output`（默认 `stereo_2d_output.mp4`）、`--model`（默认同目录 `hand_landmarker.task`）、`--freq-min 15.0`（越大越跟手越抖）、`--beta 0.6`（快速运动滞后就调大）、`--no-smooth`（关闭平滑）。
- 依赖：`mediapipe>=0.10`、`opencv-python>=4.5`、`numpy>=1.21`（`requirements.txt`）。
- `run.sh`：一键运行（自动检测本目录 `venv/bin/python`，否则系统 `python3`，要求 Python 3.9+）。

**调用关系**：不被其他代码调用（自包含单文件，`README.md` 明确「不依赖仓库里任何其他模块或目录」，面向解压即用的客户交付）。

### `tools/demos/test_stereo_depth_calib.py`

**作用**：S80M 双目深度 demo（Python 版）。用 `ctypes` 直接调用 FaysSense VI Kit SDK 官方深度引擎：`FAYS_VIK_CreateHandleWithConfig` 打开相机 → `FAYS_ATRAK_D_CreateHandleWithConfig` 创建深度引擎（`stereo_depth.yaml`，`depth_mode=1` CPU）→ `FAYS_ATRAK_D_BindViKit` 绑定（在线模式，标定由相机提供）→ 循环 `GetStereoFrames` / `FeedStereoImage` / `GetDepthImage` / `GetRectifiedImage`，三窗口实时显示深度热力图、原始/矫正双目、视差图；`P` 键导出点云 PLY。文件为顶层执行脚本（无 `main()`），并注意 S80M 固件的两个怪癖：实际拼接为 上=右目/下=左目（已由官方配置 `stereo_swap_lr: 1` 在传感器层解决，脚本默认不交换）、实际输出 RGB 通道序与头文件声称的 BGR 相反（`SWAP_RGB=True` 转回 BGR）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `_preload_opencv_dir` | `(libdir)` | `ctypes.CDLL(RTLD_GLOBAL)` 预载 OpenCV 4.2 模块及外部依赖（libwebp/libtiff/libtbb/dc1394/av* 等） | 加载动态库 |
| `img_to_numpy` | `(img: AtrakImage)` | uint8 `AtrakImage` → numpy（按 `step` 处理行填充，`as_strided`） | 返回副本数组 |
| `depth_to_numpy` | `(img: AtrakDepthImage)` | float32 深度/视差 → numpy 2D | 返回数组 |
| `save_pointcloud_ply` | `(path, pos, rgb)` | `pos: (N,3)` float32 + `rgb: (N,)` uint32 ARGB8 → 二进制 PLY（`binary_little_endian 1.0`） | 写文件 |

结构体（对应 SDK 头文件 `fays_atrak_types.h`）：`AtrakImage`、`AtrakDepthImage`（data 为 float，深度单位米）、`AtrakRectifyInfo`（矫正矩阵 R0/R1/P0/P1）、`AtrakMap`（点云 pos N×3 float、rgb N×1 ARGB8）、`AtrakIntrinsics`、`AtrakExtrinsics`、`AtrakCamParam`、`AtrakCamChainParam`、`AtrakImuParam`、`AtrakCalibrationParam`。

**关键数据**：
- 环境变量 `FAYSSENSE_SDK_DIR` 必需；未设置直接 `SystemExit` 报错退出。由此拼出库与配置路径：`lib/fays_atrak/x86_64/Release/libfays_vikit.so`、`libfayssense_aikit_depth.so`、`config/fays_vikit.yaml`、`config/perception/stereo_depth/stereo_depth.yaml`。
- OpenCV：优先预载 SDK 自带 4.2（`thirdparty/opencv-4.2.0-linux-x86_64/lib406`，深度引擎按其编译，系统 OpenCV 4.6/4.13 不兼容）；找不到才回退系统 4.6（引擎在其下会崩）。
- 输出缓冲必须按 SDK 头文件宏上限分配（如 `FAYS_ATRAK_IMG_MAX_BYTES = 3840*2160*3`、`FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM = 262144`），偏小会越界写堆导致随机段错误。
- 运行时打印标定摘要（双 Cam fx/fy/分辨率、Baseline mm）与矫正后 `fx'`/`baseline'`。
- 深度显示范围 `D_NEAR=0.3`、`D_FAR=4.0` m；按键：`Q`/`Esc` 退出、`S` 截图（`depth_heat_<ts>.png`、`disparity_<ts>.png`、`stereo_rectified_<ts>.png`、`depth_<ts>.raw` float32 米）、`P` 导出 `pointcloud_<ts>.ply`、`L` 切换左右顺序验证、`1`-`4` 调深度范围；每秒打印一次有效深度占比/中位/范围统计。

**调用关系**：必须经 `tools/demos/run_stereo_depth_demo.sh` 启动（见下），不可直接 `python` 运行（深度库加载失败时会提示用启动器）。不被主程序调用；与 C++ 版 `tools/demos/stereo_depth_demo/main.cpp` 管线一致、缓冲区口径一致。注：`tools/demos/stereo_depth_640x400.yaml` 文件头注明 "generated by test_stereo_depth_calib.py"，但当前脚本运行时读取的是 SDK 目录内的 `stereo_depth.yaml`（`DEPTH_CONFIG`），该 yaml 是 640×400 分辨率的配置样例（`depth_mode: 1`、`wls_filter: true`、`farthest_dist: 10.0`、`cloest_dist: 0.3`）。

### `tools/demos/run_stereo_depth_demo.sh`

**作用**：Python 深度 demo 启动器。深度引擎 `libfayssense_aikit_depth.so` 按 SDK 自带 OpenCV 4.2.0 编译，系统 OpenCV 4.6/4.13 不兼容（`cv::Exception: Unknown/unsupported array type`），启动器通过 `LD_LIBRARY_PATH` 让 SDK 库绑定自带 4.2（`lib406` 目录），然后 `exec python3 -u test_stereo_depth_calib.py "$@"`。SDK 目录由 `SDK_DIR` 或 `FAYSSENSE_SDK_DIR` 环境变量提供，未设置报错退出；`lib406` 目录缺失也会报错退出。

### `tools/demos/stereo_depth_demo/`（C++ 版深度 demo）

**作用**：S80M 双目深度 demo 的 C++ 版，严格按 FaysSense VI Kit SDK 官方实现编写（图像读取参考 SDK `example/fays_vikit_example.cpp`，深度引擎参考 `stereo_depth_gui/core/depth_engine.cpp`），管线与 Python 版一致（打开相机 → 创建深度引擎 → 绑定 → 取帧/深度/矫正 → 导出点云 PLY）。此处仅说明存在与用途，不展开 C++ 内部实现。

**构成与运行**：
- `main.cpp`：主程序（含 SDK 同款直方图均衡 + JET 深度着色）。
- `CMakeLists.txt`：CMake ≥ 3.12；`SDK_DIR` 未定义时默认 `/home/REDACTED/FaysSense_VI_Kit_Release`（占位符，实际构建由 `build.sh` 传 `-DSDK_DIR` 覆盖）；必须用 SDK 自带 OpenCV 4.2（`find_package(OpenCV)` 经 `-DOpenCV_DIR` 指向 SDK 目录）；链接顺序注释明确：SDK 库在前、OpenCV 4.2 模块其次、`lib406` 外部依赖 shim 最后。
- `build.sh`：自动检测 SDK（`SDK_DIR` 或 `FAYSSENSE_SDK_DIR`，未设置报错退出），校验 SDK 头文件与自带 OpenCV 4.2 的 cmake 配置存在，`cmake -DCMAKE_BUILD_TYPE=Release` + `make`，产物 `build/stereo_depth_demo`。
- `run.sh`：设置 `LD_LIBRARY_PATH`（SDK 自带 OpenCV 4.2 的 `lib406` 目录 + SDK 库目录）后运行 `build/stereo_depth_demo [viKitConfig] [depthConfig]`。
- 目录内 `depth_*.bin`、`depth_*_rect.png`、`depth_*_depth.png` 等为运行/调试时导出的示例产物（当前仓库内未保留），非源码。

**调用关系**：不被仓库其他代码调用；依赖外部 FaysSense VI Kit SDK（需 `FAYSSENSE_SDK_DIR` 指向 SDK Release 目录）。
