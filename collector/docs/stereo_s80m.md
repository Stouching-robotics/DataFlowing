# tools/stereo_s80m/

## 定位

`tools/stereo_s80m/` 是 S80M 双目相机（FaysSense VI Kit）的**全部处理代码**，自包含独立模块：只读复用 `tools/hand_detection/`（MediaPipe 检测管线），不修改主程序（`ui/`、`core/`）。覆盖完整链路：出厂标定捕获/导出 → 双目三角化 → 3D 手部两阶段管线（`hand_3d/` 子包）→ 骨架渲染 → 离线 ORB-SLAM 数据集导出与本地校验 → 相机 RGB 读取（`read_stereo_rgb.py`，含 `/dev/videoN` 端口自动解析）。

被哪些模块使用（引用证据）：

- 主程序以**子进程**方式运行 `tools/stereo_s80m/read_stereo_rgb.py`，帧经 stdout 管道传回：`core/s80m_manager.py:6`（模块注释注明目录内含 `libfays_vikit.so` (3.9.0) + `fays_vikit.yaml`）、`core/s80m_manager.py:37-42`（`STEREO_DEMO`/`STEREO_AVAILABLE` 路径常量，锚定仓库根不随 cwd 漂移）、`ui/main_window.py:703`（`_open_s80m`，面板开关启动子进程）。
- D435 工具包大量复用 `hand_3d/` 子包：
  - `tools/hand_3d_d435/live_demo.py:65-71`（`MediaPipeDetector`、`HandednessVoter`、`HandSlotTracker`、`Hand3DSmoother`、`RotatingSkeletonRenderer`、`create_video_sink`、`io`）
  - `tools/hand_3d_d435/run_pipeline_d435.py:37-43`（同上全套）
  - `tools/hand_3d_d435/mono_assign.py:34`、`fill_track.py:32`（`HandSlotTracker`）
  - `tools/hand_3d_d435/glove_detector.py:76`（`DetectedHand`）
  - `tools/hand_3d_d435/tools/render_keypoints_parquet.py:46-47`（`RotatingSkeletonRenderer`、`create_video_sink`）
  - `tools/hand_3d_d435/probes/probe_live_consistency.py:39-42`、`probe_align_overlay.py:57-58`
- 包内自用：`hand_3d/` 依赖本目录的 `stereo_triangulate.py`（三角化几何）与 `render_stereo.py`（叠加渲染）；`render_stereo.py:50`、`hand_benchmark.py:45`、`hand_triangulate.py:73` 等。

目录内另有非 Python 数据：`tools/stereo_s80m/lib/fays_atrak/x86_64/Release/libfays_vikit.so`（FaysSense SDK 动态库）与 `tools/stereo_s80m/config/fays_vikit.yaml`、`fays_vikit_50fps.yaml`（SDK 配置，`--config` 可切换 50fps 副本）。`tools/stereo_s80m/dist/` 为交付目录；`tools/stereo_s80m/offline_slam_output/` 为产物目录（首次运行 `export_offline_slam_dataset.py` 后生成，仓库内不保留），见文末说明。

## 文件清单

| 文件 | 一句话作用 |
| --- | --- |
| `tools/stereo_s80m/capture_calibration.py` | 从 SDK 出厂标定静态 yaml（`FAYSSENSE_SDK_DIR` 环境变量定位）生成设备级 `config/s80m_stereo_calibration.json`，可选 `--live` 连相机校验 |
| `tools/stereo_s80m/export_calibration.py` | 通过 ctypes 从设备 ROM 读取标定，写会话级 `calibration/head_stereo.json` + 设备级 JSON + DumpCalib yaml 备份 |
| `tools/stereo_s80m/read_stereo_rgb.py` | S80M 双目 RGB 读取：ctypes 调 SDK 取左右目帧/IMU，`--pipe` 二进制管道或窗口预览，含设备端口自动解析与曝光控制 |
| `tools/stereo_s80m/stereo_triangulate.py` | 双目三角化几何核心：标定加载、fisheye/radtan 双路径矫正、三角化、跨目手匹配、自检 |
| `tools/stereo_s80m/render_stereo.py` | 骨架叠加渲染模块：3D/2D 单帧叠加、视频写器与 H.264 转码、parquet 纯渲染重放 |
| `tools/stereo_s80m/hand_triangulate.py` | 单阶段后处理脚本：左右目视频跑 MediaPipe → 三角化 → LeRobot 风格 parquet + 可视化视频 |
| `tools/stereo_s80m/hand_benchmark.py` | 手势检测参数基准：分辨率 × 帧率 × 颜色 × 单双目扫描，输出稳定性指标 CSV |
| `tools/stereo_s80m/export_offline_slam_dataset.py` | EgoData 会话 → 离线双目惯性 SLAM 数据集（ORB-SLAM3 格式，矫正灰度 PNG + CSV + orb_calibration.yaml） |
| `tools/stereo_s80m/validate_offline_dataset.py` | 离线 SLAM 数据集本地验收：复刻客户侧 FileStorage 格式检查与官方 rectify 一致性校验 |
| `tools/stereo_s80m/hand_3d/__init__.py` | 子包入口：从 `run_pipeline` 导入 `main` 供包外引用（无 `__main__` 执行块，管线入口仍是 `run_pipeline.py`） |
| `tools/stereo_s80m/hand_3d/detector.py` | 2D 检测抽象层：`KeypointDetector` 接口 + `MediaPipeDetector`（float 亚像素，CPU/GPU 双后端） |
| `tools/stereo_s80m/hand_3d/identity.py` | 手性投票 `HandednessVoter`：轨迹票仓多数表决，压掉 MediaPipe handedness 逐帧闪烁 |
| `tools/stereo_s80m/hand_3d/track3d.py` | 槽位跟踪 + 遮挡传播：`HandSlotTracker` αβ 滤波预测 3D，`make_pseudo_pair` 生成伪 pair 救援 |
| `tools/stereo_s80m/hand_3d/smoother.py` | 3D 域因果时序平滑：`Hand3DSmoother` 逐点 One-Euro，防跨手状态污染 |
| `tools/stereo_s80m/hand_3d/perspective_crop.py` | 两阶段透视裁剪精修（Hur et al. 2025）：3D 投影取 ROI → 裁剪图重检测 → 二次三角化 |
| `tools/stereo_s80m/hand_3d/mp_gpu.py` | MediaPipe GPU delegate 直连封装 `FastHandLandmarker` + 子进程冒烟测试 |
| `tools/stereo_s80m/hand_3d/postprocess.py` | 离线后处理：`fill_gaps` 间隙插值 + `offline_smooth` 零相位速度自适应平滑 |
| `tools/stereo_s80m/hand_3d/renderer_3d.py` | 3D 旋转视角骨架渲染器 `RotatingSkeletonRenderer`（numpy+cv2 透视投影，零额外依赖） |
| `tools/stereo_s80m/hand_3d/io.py` | 数据 IO：会话元数据读取 + LeRobot 风格 parquet 打包/落盘 + meta 合并 |
| `tools/stereo_s80m/hand_3d/video_writer.py` | 管道视频写器：ffmpeg 子进程单段直出 H.264（nvenc → libx264 → mp4v 逐级回退） |
| `tools/stereo_s80m/hand_3d/run_pipeline.py` | 两阶段管线主流程 + CLI：检测 → 配对 → 三角化 → 精修 → 平滑 → 渲染 → parquet |
| `tools/stereo_s80m/hand_3d/probes/probe_compare3.py` | 探针：单阶段 vs 两阶段双配置硬对比（指标 CSV + 并排视频 montage） |
| `tools/stereo_s80m/hand_3d/probes/probe_jitter.py` | 探针：P5 抖动验收——三列 jitter 对比 + 快段保真 + 长缺口不幻觉检查 + 合成静止手测试 |
| `tools/stereo_s80m/hand_3d/probes/probe_mp_gpu.py` | 探针：GPU delegate 冒烟 + 单目计时 + CPU/GPU 关键点差值 |
| `tools/stereo_s80m/hand_3d/probes/probe_nvenc.py` | 探针：管道写器冒烟（各编码器直出 + BrokenPipe latch + 可解码性） |
| `tools/stereo_s80m/dist/s80m_stereo_camera/*.py` | 客服交付版自包含副本（3 个 .py，非本目录源码，见文末） |

## 各文件详解

### capture_calibration.py

**作用**：离线生成设备级标定。解析 FaysSense VI Kit SDK 发布目录下的出厂标定静态文件 `config/calib/calib.yaml`（默认路径由环境变量 `FAYS_CALIB_YAML` 或 `FAYSSENSE_SDK_DIR` 定位），提取左右目内参/畸变/外参 `T_cn_cnm1`，把内参从 SDK 原生 640×400 **等比缩放到录制分辨率 1280×800**，写入 `config/s80m_stereo_calibration.json`。`--live` 可选路径通过 ctypes 加载 `libfays_vikit.so` 只调 `CreateHandleWithConfig`/`GetCalibrationParam`/`DestroyHandle` 与静态文件对比基线/内参——注释明确**绝不调用 `GetStereoFrames`**（SDK 3.9.1 在 RGB 失败后有段错误风险）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `load_calib_yaml` | `(path)` | PyYAML 解析 SDK 标定 yaml，校验 `cam0`/`cam1` 存在 | dict |
| `_intrinsics_from_cam` | `(cam)` | 从 yaml 相机条目提取 K(3×3)、dist、模型、分辨率 | 4 元组 |
| `_T_cn_cnm1_from_cam` | `(cam)` | 从 cam1 的 `T_cn_cnm1` 4×4 齐次矩阵提取 R/t | (R 3×3, t 3×1) |
| `build_calibration` | `(yaml_path, target_resolution)` | 构建设备级标定 dict，内参按宽度比例缩放 | dict（写文件由 main 完成） |
| `verify_with_sdk` | `(calib, vikit_lib, config_path)` | 连相机读 ROM 标定对比：基线偏差 >2mm、fx 偏差 >1% 告警 | 打印对比结果 |
| `main` | `--calib-yaml / --output / --live` | CLI：构建 → 写 JSON → 打印摘要（含 R 正交性 det 检查）→ 可选在线校验 | 退出码 |

**关键数据**：输出 JSON 含平台字段（`type: "stereo_rgbd_camera"`、`name: "head_stereo"`、`resolution: [1280, 800]`、`fps: 25.0`、`baseline`（米，`t` 的模）、`left_camera`/`right_camera`（`intrinsic` 为 `[fx, fy, cx, cy]`、`distortion`、`distortion_model`）、`depth_scale: 0.001`、`cam_imu_timeshift`）+ **三角化扩展字段** `rotation`（行主序 9 元）、`translation`（3 元，米）、`distortion_model`（决定三角化走 fisheye 还是普通路径）。

**调用关系**：独立 CLI（无包内调用方）；其输出 JSON 被 `stereo_triangulate.load_stereo_calibration` 的查找链第 3 级消费。

### export_calibration.py

**作用**：采集端职责——从设备 ROM 读取出厂标定（后端二进制符号走查确认 `FAYS_VIK_GetCalibrationParam`/`FAYS_VIK_DumpCalib` 存在），补齐 SDK 示例没导出标定的链路。写 `<session>/calibration/head_stereo.json`（平台格式）+ `calibration/s80m_dump_calib.yaml`（DumpCalib 原始备份）+ `config/s80m_stereo_calibration.json`（设备级默认，三角化回退链用）。同样只调 Create/GetCalibrationParam/DumpCalib/Destroy，不碰 `GetStereoFrames`。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `AtrakIntrinsics` / `AtrakExtrinsics` / `AtrakCamParam` / `AtrakCamChainParam` / `AtrakImuParam` / `AtrakCalibrationParam` | ctypes `Structure` | 对应 `fays_atrak_types.h` 的 SDK 结构体 | — |
| `DISTORTION_MODEL_NAMES` | dict | SDK 畸变枚举（0-4）→ 字符串：`none`/`equidistant`(KB4)/`radtan`/`brown_conrady`/`cvbasic` | — |
| `_load_sdk` | — | 校验库/配置存在，预加载 OpenCV（4.2 或 4.6），绑定 mangled 符号 | 函数句柄组 |
| `calibration_to_dict` | `(cal: AtrakCalibrationParam)` | SDK 结构体 → 平台 JSON（含 `imu` 噪声参数与三角化扩展字段），内参 640×400→1280×800 缩放 | dict |
| `is_usable` | `(calib)` | 与 `stereo_triangulate.is_usable` 同逻辑（不依赖 cv2 的自包含版） | bool |
| `export` | `(calib_path, yaml_path, device_output)` | 完整导出流程，finally 中 `DestroyHandle` | (calib dict, yaml 路径) |
| `main` | `--session / --output` | CLI | 退出码 |

**关键数据**：JSON 与 `capture_calibration.py` 同 schema，另加 `"source": "fays_sdk_device_rom"` 与 `imu` 子对象（`accelerometer_noise_density`、`accelerometer_random_walk`、`gyroscope_noise_density`、`gyroscope_random_walk`、`update_rate`）。

**调用关系**：独立 CLI；输出被 `stereo_triangulate.py`、`export_offline_slam_dataset.py`（交叉校验源）消费。

### read_stereo_rgb.py

**作用**：S80M 双目 RGB 读取脚本，ctypes 调 `libfays_vikit.so`（mangled 符号 `FAYS_VIK_CreateHandleWithConfig`/`GetStereoFrames`/`GetImuData`/`GetVersion`）。两种模式：`--pipe <path|->` 以 JPEG 二进制协议把左右目帧 + IMU 样本写到管道（主程序经此收帧）；无 `--pipe` 时窗口实时预览（Q/Esc 退出、S 截图、R 切 180° 旋转）。启动时**按设备名（`FTDI Superspeed Video Bridge`）与 USB 接口号重写临时 yaml**：接口 `1.0` = 双目对、`1.2` = IMU/中置 RGB、`rgb_dev_port` 一律置 `NULL`——绕开 `/dev/videoN` 端口漂移（代码注释原文：DECXIN/RealSense 先枚举时，S80C 的 FTDI 桥被挤到 `video2+`，SDK 会拿别的相机当双目，现象 `RgbInit failed` + 主程序永远等不到帧）。解析失败保持原 yaml 不动。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `AtrakImage` / `AtrakIMU` | ctypes `Structure` | SDK 图像（`device_id/seq/timestamp/width/height/channel/encoding/step/bytes/data`）与 IMU 样本（纳秒时间戳 + gyro×3 + acc×3） | — |
| `_graceful_exit` | `(sig, frame)` | SIGTERM/SIGINT → `SystemExit`，走 finally 释放 SDK，避免 FT602 设备留坏状态 | 进程退出 |
| `_stdin_control` | — | 后台线程读 stdin：`SET_EXPOSURE <float>` 行协议（`-1.0`=自动，`1.0~885.0`=手动；符号缺失时静默降级） | 调用 `FAYS_VIK_SetStereoExposure` |
| `_imu_collector` / `_drain_imu` | — | IMU 高频轮询线程（`IMU_HZ` 环境变量，默认 200；轮询率 = 2.5× 采样率）+ 线程安全取走缓冲 | 列表 |
| 主循环 | — | `GetStereoFrames` → 上下拼接图切分（**上=左目、下=右目**）→ 通道序修正 → JPEG 编码写管道 | 帧/IMU 字节流 |

**关键数据**：

- 管道协议（大端）：`[4B left_len][8B left_ts_ns][left_jpg]` + `[4B right_len][8B right_ts_ns][right_jpg]` + `[4B imu_count][imu_count × (8B ts + 8B×3 gyro + 8B×3 acc)]`；时间戳为 SDK 硬件纳秒时钟（帧/IMU 同源）。
- JPEG 质量 85；预分配缓冲 `MAX_BYTES = 1280*800*2*3`。
- SDK 帧 `encoding` 声称 BGR8 但通道序实为 RGB → 统一 `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` 修正。
- `IMU_HZ` 环境变量（默认 200，0=关）；注释记录教训：2kHz 轮询每秒 ~2000 次 GIL 抢占把 Stereo FPS 从 25 拖到 6。
- 环境修复：设置 `QT_QPA_PLATFORM_PLUGIN_PATH`/`QT_QPA_PLATFORM=xcb` 指向 cv2 5.0.0 wheel 自带 Qt 插件。

**调用关系**：被 `ui/main_window.py:703`（`_open_s80m`）以子进程启动；`--pipe -` 时 fd 1 被重定向到 stderr 以拦截 C++ SDK 的 printf/cout。

### stereo_triangulate.py

**作用**：双目三角化几何核心，独立子模块不依赖主程序。从标定 dict 构建 `StereoTriangulator`：`distortion_model == "equidistant"` 走 `cv2.fisheye.stereoRectify` + `fisheye.undistortPoints`，其他模型走 `cv2.stereoRectify` + `cv2.undistortPoints`。三角化输出的 3D 点在**左目（cam0）相机光学系**下（米制）：原点 = cam0 光心，+X 右、+Y 下、+Z 前（OpenCV 约定）。还提供左右目手列表的跨目匹配（几何主判据）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `K_from_intrinsic` | `(intr)` | `[fx, fy, cx, cy]` → 3×3 相机矩阵 | np.ndarray |
| `is_usable` | `(calib)` | 左右内参非零 + R(9)/t(3) 齐全且有限 + 基线非零 | bool |
| `load_stereo_calibration` | `(session_path=None, calib_path=None)` | 查找链：`--calib` 显式路径 → `<session>/calibration/head_stereo.json` → `config/s80m_stereo_calibration.json`；显式指定不可用时**报错而非静默回退** | dict 或 None |
| `TriangulationResult` | `(points_3d, reproj_error)` | 结果容器：`points_3d` (N,3) 无效点 NaN、`reproj_error` (N,) 无效 inf、`valid`/`valid_count`/`mean_error`、`z` 属性 | — |
| `HandPair` | `(l_idx, r_idx, left_label, result)` | 左右目匹配出的同一只手 | — |
| `StereoTriangulator` | `(calib, image_size=None, swap_cams=False)` | 三角化器：内参按目标分辨率缩放、SDK 8 系数只取 fisheye 前 4 个（k1..k4）、矫正映射表、`swap_cams` 处理左右文件与 cam0/cam1 错位 | — |
| `StereoTriangulator.rectify_points` | `(pts, side)` | 原始图像素 (N,2) → 矫正图像素 (N,2) | np.ndarray |
| `StereoTriangulator.rectified_image` | `(frame, side)` | 原始帧 → 矫正帧（可视化用） | np.ndarray |
| `StereoTriangulator.project` | `(xyz, side)` | 左目相机系 3D → 矫正图像素（重投影验证/可视化） | (N,2) |
| `StereoTriangulator.triangulate` | `(pts_l, pts_r, max_err=None, max_depth=None)` | 原始像素关键点 (N,2) → 3D (N,3)；逐点过滤重投影误差 ≤ max_err、`min_depth < z ≤ max_depth`、有限 | `TriangulationResult` |
| `StereoTriangulator.summarize` | — | 单行摘要：模型/fx_rect/baseline_rect/分辨率 | str |
| `match_hands` | `(left_hands, right_hands, tri, max_err, max_depth, min_valid)` | ≤2×2 穷举 + 贪心去重；评分 = `有效点数×10 − 平均重投影误差 + (label 一致 +5)`；按物理左右排序（左目 label "Left" 在前） | `HandPair` 列表 |
| `_self_test` | — | 3D → 双图投影（+0.5px 噪声）→ 逆变换回原始像素 → 三角化往返误差测试 | 退出码 |

**关键数据**：阈值常量 `DEFAULT_MAX_REPROJ_ERR = 8.0` px、`DEFAULT_MAX_DEPTH = 3.0` m、`DEFAULT_MIN_DEPTH = 0.05` m、`MIN_VALID_POINTS = 8`。实例属性 `fx_rect`（矫正焦距，取自 P1）、`baseline_rect`（矫正基线）。`swap_cams=True` 时左右内参互换且外参取逆（`R^T`, `−R·t`）。自检允许 3D 误差 < 80mm、重投影 < 2.0px（注释：基线 8cm 双目 1px 视差噪声在 1.5m 处 ≈ 80mm 深度误差）。

**调用关系**：被 `render_stereo.py:50`、`hand_triangulate.py:73`、`hand_benchmark.py:45`、`hand_3d/run_pipeline.py:63`、`hand_3d/perspective_crop.py:37`、`hand_3d/track3d.py:34` 及 `tools/hand_3d_d435/*` 广泛复用。

### render_stereo.py

**作用**：独立于检测/三角化的可视化组件。两种模式：`3d`（默认）把三角化 3D 骨架（`observation.keypoints.hand_3d` 列）投影叠加到左右目**矫正图**；`2d` 把 MediaPipe 原始 2D 关键点（`stereo_left`/`stereo_right` 列，84 维）叠加到**原图**。既作为库被 `hand_triangulate.py` 逐帧调用，也可作为独立 CLI 从已落盘 parquet 纯渲染重放（不重新检测）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `HAND_EDGES` | 常量 | MediaPipe 21 点骨架连接拓扑（0-17 腕等 21 条边） | — |
| `draw_skeleton` | `(img, pts, color, radius, thickness)` | 按 `HAND_EDGES` 画单色骨架，非有限点跳过 | 原地画 |
| `_draw_hand_styled` | `(img, pts, label, box_color, pts_angle)` | demo 风格单手叠加：teal 边框 + 底衬标签 + 五指分色（复用 `tools/hand_detection/hand_common.draw_hand`）+ 手势文本；有效点 <8 整手跳过 | 原地画 |
| `overlay_view` | `(img, pairs, tri, side, frame_idx, total)` | 一张矫正图上叠加 3D 骨架（`HandPair` 列表投影回该视角） | 原地画 |
| `overlay_view_2d` | `(img, kpts, side, frame_idx, total)` | 原图上叠加 (2,21,2) 原始 2D 关键点，全零手跳过 | 原地画 |
| `create_video_writer` | `(out_path, fps, width, height)` | mp4v 临时写器（`*_tmp.avi`） | (writer, tmp_path) |
| `finalize_video` | `(writer, tmp_path, out_path)` | 释放写器并转码 H.264（libx264 crf 23）；无可用 ffmpeg 时保留 mp4v | 最终路径 str |
| `render_session_from_parquet` | `(session_dir, parquet_path, calib_path, out_dir, mode)` | 纯渲染重放：3d 消费 `hand_3d`（126=2×21×3）+ present/label/err 列；2d 消费 `stereo_left/right`（84）列 | 视频路径 |
| `render_video_2d` | `(session_dir, out_dir, scale, smooth)` | 直接从视频跑 2D 检测 + 渲染（左右目各自独立 MediaPipe 实例，共享会污染 VIDEO 模式追踪先验）；`scale<1` 在缩放图上检测 | 视频路径 |
| `main` | `session_dir [--parquet --calib --out --mode --detect --scale]` | CLI | 退出码 |

**关键数据**：骨架色——品红 = 三角化 3D 投影、青 = 原始 2D 关键点。ffmpeg 候选链：`shutil.which("ffmpeg")` → `/usr/bin/ffmpeg` → 环境变量 `FFMPEG_BIN`（默认兜底 `~/miniconda3/envs/lerobot/bin/ffmpeg`）；注释说明 conda base 的 ffmpeg 因 openvino/tbb 符号错误不可用（2026-08 实测）。

**调用关系**：被 `hand_triangulate.py:66`、`hand_3d/run_pipeline.py:59`、`hand_3d/video_writer.py:129`（旧两段式回退路径）调用。

### hand_triangulate.py

**作用**：单阶段（无裁剪精修）双目手部关键点后处理脚本。左右目视频各自跑 `MediaPipeHandPipeline`（独立实例）→ 方向自检 → `match_hands` 跨目配对 → 三角化 → 物理 3D 关键点按 LeRobot 风格并入 episode parquet；同时产出并排矫正图可视化视频。**只新增** `data/keypoints/` 与 meta 追加，不修改现有文件内容。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_find_video` | `(session_path, cam)` | 定位 `videos/<cam>/chunk-0000/<cam>.mp4`，回退 `videos/<cam>.mp4` | 路径或 None |
| `_load_episode_meta` | `(session_path)` | 读 `meta/episodes/*.parquet` 与 `meta/tasks.jsonl` | (episode_index, task_index) |
| `_load_timestamps` | `(session_path)` | `timestamps.json` → {frame_index: timestamp}，同帧多条取第一条 | dict |
| `_pack_2d` / `_pack_3d` / `_pack_errors` | `(hands)` / `(pairs)` / `(pairs)` | 打包 84 维 2D（缺手全零）/ 126 维 3D（无效 NaN）/ 每手平均重投影误差 | list |
| `_detect_orientation` | `(pipe_l, pipe_r, vc_l, vc_r, tri_normal, tri_swapped)` | 前 10 帧两种配对各三角化比平均重投影误差（小者几何自洽），True=需 swap | bool |
| `_merge_info_json` / `_merge_stats_json` | `(session_path)` / `(session_path, rows)` | 纯追加 `meta/info.json` features 与 `meta/stats.json` 统计（mean/std/min/max，NaN 忽略） | 写文件 |
| `main` | `session_dir [--calib --max-err --max-depth --every --no-video --freq-min --beta --no-smooth]` | 主循环 + parquet 落盘 + 副本拆分 + 视频转码 + 汇总 | 退出码 |

**关键数据**：

- parquet 列：`episode_index`/`frame_index`/`timestamp`/`task_index` + `observation.keypoints.stereo_left`（float32 list 84 = 2手×21点×(x,y)）、`stereo_right`（84）、`hand_3d`（126 = 2手×21点×(x,y,z)，米）、`reprojection_error`（2）、`hand_0/1_present`（bool）、`hand_0/1_label`（string）、`action`（占位 `[0.0]`）。`hand_3d` 无效点 NaN，2D 缺手全零。
- 输出：`<session>/data/keypoints/chunk-0000/chunk_000000.parquet`（zstd）；镜像目录 `keypoints_output/<tag>/<session>/` 下 `stereo_triangulate.mp4` + `hand_3d/chunk-000.parquet` + `hand_2d/chunk-000.parquet`（2D 副本为列子集）。
- One-Euro 平滑参数由 CLI 控制（`--freq-min` 默认 5.0、`--beta` 默认 0.05、`--no-smooth` 全关）。

**调用关系**：调用 `hand_detection.hand_pipeline_mediapipe`、`stereo_s80m.render_stereo`、`stereo_s80m.stereo_triangulate`；独立 CLI。

### hand_benchmark.py

**作用**：S80M 手势检测参数基准测试。对会话视频离线扫参数组合（4 分辨率 × 4 帧率隔帧 × rgb/gray × mono/stereo），`ProcessPoolExecutor` 多进程并行，输出稳定性指标矩阵 CSV + 终端表格。指标（smooth 关闭，测管线原始稳定性）：`det_rate`、`det2_rate`、`mean_score`、`max_miss`（最大连续未检出段）、`disp`（帧间关键点位移中位）、`jitter`（位移一阶差分中位）、`tri_rate`、`tri_err`、`ms_frame`。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `run_group` | `(session, calib_path, swap, (w,h), (fps,step), color, mode)` | worker 进程内跑一组参数（左右目独立 MediaPipe 实例），统计全指标 | dict |
| `_detect_swap` | `(session, calib_path, size)` | ORB 极线误差法自检：两种配对各自 |y差| 中位数（0.75 ratio 过滤），小者正确 | bool |
| `main` | `--session --calib --workers --out --modes` | 组装 4×4×2×2 组合 → 并行跑 → CSV + 结果矩阵表 | 退出码 |

**关键数据**：`RESOLUTIONS = [(1280,800), (960,600), (640,400), (480,300)]`；`FPS_STEPS = [(25,1), (12,2), (6,4), (3,8)]`（fps, 隔帧数）；灰度 = `GRAY2BGR` 三通道复制（信息等价）；双目三角化用 `image_size=(w,h)` 且 `swap_cams=swap`。CSV 默认落 `<脚本目录>/hand_benchmark/benchmark_<tag>.csv`（`os.makedirs` 按需创建；`--modes` 非 `all` 时文件名追加 `_<modes>` 后缀）。

**调用关系**：调用 `hand_detection.hand_pipeline_mediapipe` 与 `stereo_s80m.stereo_triangulate`；独立 CLI。

### export_offline_slam_dataset.py

**作用**：EgoData 录制会话 → 离线双目惯性 SLAM 数据集（ORB-SLAM3 输入格式；格式规范 `OFFLINE_DATA_FORMAT.md` / `OFFLINE_DATA_RECORDING.md` 为客户侧 offline_slam/ 交付文档，不在本仓库）。转换要点：1280×800 彩色视频 → 灰度 → 缩放标定原生 640×400 → 用工厂标定（equidistant/fisheye）双目矫正 → PNG；相机硬件 32 位纳秒计数器（每 2^32 ns ≈ 4.295s 回卷）逐帧回卷修正并对齐 IMU 时基、按规范应用一次 `timeshift_cam_imu`；IMU 从 parquet 逐帧批次展开为逐样本行，列序 `[gx,gy,gz,ax,ay,az] → [ax,ay,az,gx,gy,gz]`；两个 CSV 时间戳严格递增、IMU 覆盖完整相机时间范围。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `parse_calib_yaml` | `(path)` | 手写递归解析工厂 `calib.yaml`（避免 PyYAML 依赖，支持内联/块序列两种列表风格） | dict |
| `load_session` | `(session_dir)` | 读 `timestamps.json` + IMU parquet（`observation.imu`/`imu_ts_ns` 列展开）+ `metadata.json` | dict |
| `unwrap_camera_clock` | `(frames, imu_ref_ns, period_ns=40_000_000)` | 32 位回卷修正：用首条 IMU 样本推断回卷次数 k，逐帧增量累加，陈旧读帧放回标称网格保证严格递增 | 绝对时钟列表 |
| `build_rect_maps` | `(cal, size=(640,400))` | 工厂标定 → 双目矫正映射表（内参按分辨率比例缩放，与 ROM 原生 640×400 标定一致） | dict（map0/map1/P0/P1/R0/R1） |
| `check_orientation` | `(video_dir, cal, size, n_frames=6)` | ORB 极线对齐自检（注释：本会话实测 3px vs 57px），True=需交换左右 | bool |
| `extract_frames` | `(video_dir, ts_list, maps, out_dir, size)` | 左右视频 → 灰度 → 缩放 → 矫正 → PNG（PNG_COMPRESSION 6）；帧数必须等于时间戳数 | 每目写盘统计 |
| `write_csvs` | `(out_dir, ts_aligned, ts_raw, imu_rows, timeshift)` | 写 `images.csv` / `camera_timestamps.csv` / `imu.csv` | 写文件 |
| `_opencv_matrix` | `(name, rows, cols, dt, data)` | OpenCV FileStorage 矩阵块（数据单行，`dt` 显式指定） | str |
| `write_orb_calibration` | `(out_dir, cal, P0, P1, serial, imu_rate)` | 生成 `orb_calibration.yaml`（ORB-SLAM3 FileStorage 配置，六项修复 + 必需键清单） | 写文件 |
| `write_runtime_calibration` | `(out_dir, cal, serial, size)` | 生成 `s80m_runtime_calibration.yaml`（K1/D1/K2/D2/R/T/T_b_c1 + serial/image size/timeshift），供官方采集器一致性验收 | 写文件 |
| `verify_against_device_json` | `(cal, tol=1e-3)` | 工厂 yaml 与设备 ROM dump（`config/s80m_stereo_calibration.json`）交叉校验，防 yaml 过期/错设备 | 问题列表 |
| `write_metadata_yaml` | `(out_dir, session, stats, n_imu, imu_rate)` | 交付说明 `metadata.yaml` | 写文件 |
| `validate` | `(ts_aligned, imu_rows)` | 校验时间戳严格递增 + IMU 覆盖 | 问题列表 |
| `main` | `--session --out --calib --serial` | 全流程：校验 → 时钟对齐 → 方向自检 → 帧提取 → 写 CSV/标定 → 校验 → tar.gz 打包 | 退出码 |

**关键数据**：

- 输出目录 `session_YYYYMMDD_HHMMSS/`：`orb_calibration.yaml`、`images.csv`（`timestamp_ns,left_image,right_image`）、`imu.csv`、`camera_timestamps.csv`（审计：原始/对齐后/偏移）、`metadata.yaml`、`cam0/data/<ts>.png`、`cam1/data/<ts>.png`（已矫正 640×400 8-bit 灰度）；同名目录已存在时加 `_v2/_v3` 后缀，绝不覆盖。默认输出根 `tools/stereo_s80m/offline_slam_output/`（首次运行后生成，仓库内不保留）。
- `orb_calibration.yaml` 关键点（注释明确六项修复）：`%YAML:1.0` 必须首行；`Camera.width/height`（非 rows/cols）；`IMU.Frequency`（大写 F）；实数值必须带小数点（否则解析成 INT 被 `isReal()` 拒绝）；`Stereo.ThDepth` + 全部 10 个 `Viewer.*`；`IMU.GyroWalk/AccWalk` 非零（0 会导致预积分协方差退化 → Sophus 崩溃）；`IMU.T_b_c1` 必须 `dt: f`（32F，ORB-SLAM3 的 `toMatrix3d` 用 `.at<float>()` 读，写成 `dt: d` 会按 double 重解释 float → Sophus 崩溃）。IMU 噪声常量：`NoiseGyro 1.9e-05`、`NoiseAcc 1.22e-04`、`GyroWalk 0.0002`、`AccWalk 0.00086`；`Camera.bf = baseline × fx`。
- `_TIMESHIFT_APPLIED = True`：时间偏移已应用一次，回放时 `--camera-time-shift-s` 必须为 0（写进 yaml 注释与 metadata）。
- 方向约定（SDK `fays_atrak_types.h`："T_cn_imu // Bring points in {imu} frame to {camera n} frame"）：工厂 `T_cam_imu` = IMU系→左目系，即 ORB-SLAM3 的 `IMU.T_b_c1`，直接写入。
- `--serial`：设备序列号（用于元数据与校验；默认值写死在 argparse 中，本仓库文档不抄录序列号）。

**调用关系**：独立 CLI；与 `validate_offline_dataset.py` 配套（导出产物由其本地验收）。

### validate_offline_dataset.py

**作用**：离线 SLAM 数据集本地验收——**复刻客户侧两份检查**：① `cv::FileStorage` 格式检查（`离线ORB-SLAM报错修复记录.md` 六项修复 + 必需键）；② 官方采集器 `prepare_rectification_and_validate_settings` 复刻（`offline_slam/fayssense_offline_dataset_recorder.cc`）：用 `s80m_runtime_calibration.yaml` 的 K/D/R/T 在 640×400 重算 `cv2.fisheye.stereoRectify`，校验 orb yaml 一致性；③ 数据完整性（时间戳严格递增、IMU 覆盖、帧数、图像属性）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `REQUIRED_KEYS` / `REAL_KEYS` | 常量 | 必需键清单（`Camera.*`/`Camera1.*`/`Stereo.*`/`IMU.*`/`Viewer.*`）与必须是 REAL 节点的实数值键清单 | — |
| `check_orb_yaml` | `(session)` | `orb_calibration.yaml` 格式检查：首行 `%YAML:1.0`、键齐全、REAL 节点类型、`IMU.T_b_c1` 必须 32F 且 4×4 | 问题列表 |
| `check_runtime_consistency` | `(session, size=(640,400))` | 复刻官方验收：`Camera.type=="Rectified"`、`Camera1.fx/fy/cx/cy` 与重算 P1 差 ≤1e-4、`Stereo.b` 与 `P2[0,3]/P2[0,0]` 差 ≤1e-4、`IMU.T_b_c1` 与 runtime 差（Inf-norm）≤1e-5、fps/IMU 频率 >0 | 问题列表 |
| `check_data` | `(session)` | images.csv/imu.csv 时间戳严格递增、IMU 覆盖相机时间范围、抽 1 张图验证 640×400 8bit 灰度 | (issues, info) |
| `main` | `session_dir`（sys.argv） | 顺序跑三组检查，任一失败退出码 1 | 退出码 |

**调用关系**：独立 CLI，输入是 `export_offline_slam_dataset.py` 的产物目录。

### hand_3d/__init__.py

**作用**：子包入口。把仓库根插入 `sys.path` 后 `from stereo_s80m.hand_3d.run_pipeline import main`，`__all__ = ["main"]`——`python -m stereo_s80m.hand_3d` 只完成导入不执行管线；入口仍是 `python tools/stereo_s80m/hand_3d/run_pipeline.py <session>`（与模块 docstring 用法一致）。

### hand_3d/detector.py

**作用**：2D 关键点检测抽象层——几何层唯一依赖的接口。`KeypointDetector.detect()` 返回 21 点像素关键点（MediaPipe 拓扑序），三角化、精修、渲染全部只依赖该接口，因此 GPU 神经检测器可作为可替换后端接入。`MediaPipeDetector` 关键改进：用 `norm_landmarks × 帧尺寸` 重算 **float 亚像素坐标**（现有管线 landmarks 是 int 截断，0.5px 量化对 ~3.8px 级重投影误差是可见噪声源，对裁剪图放大精修尤其重要）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `DetectedHand` | dataclass：`landmarks (21,2)`、`label`、`score`、`index`、`conf (21,)` | 单只手的 2D 检测结果（几何层消费的最小结构） | — |
| `DetectedHand.from_hand_result` | `(hr, frame_w, frame_h, index=0)` | MediaPipe `HandResult` → 亚像素像素坐标 | `DetectedHand` |
| `KeypointDetector` | ABC | 抽象接口：`detect(frame_bgr)` / `reset()` / `close()` | — |
| `MediaPipeDetector` | `(model_path, num_hands=2, mirror, smooth, freq_min=5.0, beta=0.05, dcutoff=1.0, det_conf=0.5, track_conf=0.5, delegate="cpu")` | 包装 `MediaPipeHandPipeline`（CPU）或惰性接入 `hand_3d.mp_gpu.FastHandLandmarker`（GPU，3.0ms/帧 vs CPU 7.8ms；GPU 时两目必须同线程顺序，输出有 ~2.8px 级 fp16 数值漂移） | — |

**调用关系**：被 `run_pipeline.py:70`、`tools/hand_3d_d435/live_demo.py:65`、`run_pipeline_d435.py:37`、`glove_detector.py:76`（只用 `DetectedHand`）等使用；`mp_gpu.py:35` 反向 import `DetectedHand`。

### hand_3d/identity.py

**作用**：手性投票——修复 MediaPipe handedness 逐帧独立分类在遮挡/快动/对称手势下闪烁，导致 `match_hands` 槽位排序不稳、tracker 换手误重置、骨架身份对穿。每目一个 `HandednessVoter`：维护 ≤2 条手轨迹（质心最近贪心关联，ByteTrack 式），每条轨迹带 7 帧 label 票仓，当前帧 = 轨迹票 + 原始票**严格多数表决**。注释记录两代教训：v1 单次投票平票保原始 → 闪烁穿透；v2 双手交叠时贪心关联交叉 → 票仓自锁错 label → 增加**重叠守卫**（两质心 < `OVERLAP_PX` 时不表决，原始 label 直通并把轨迹票仓重播种；实际实现为交叠期 label 冻结为交叠前稳定值、位置不更新）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `HandednessVoter` | `(window=7, min_score=0.7)` | 单目实例；轨迹 = {pos, votes(deque), last, idle}，空帧全清、未分配轨迹冻结、超限替换最旧 | — |
| `HandednessVoter.update` | `(hands, frame_w=1280, frame_h=800, frame=None, cam="?")` | 贪心分配 → 重叠守卫 → 逐手表决（严格多数，平票沿用轨迹稳定 label）→ 轨迹更新 | 原地覆盖每只手 `label` |
| `_dump_debug` | — | `HAND3D_IDENTITY_DEBUG` 环境变量指向文件时，atexit 写 `frame,cam;pre,post,score,cx,cy` 调试行 | 写文件 |

**关键数据**：`VOTE_WINDOW=7`、`MIN_VOTE_SCORE=0.7`（低于此的原始检测不计票）、`ASSOC_GATE=0.12`（×max(宽,高)，1280×800 下 ≈154px）、`OVERLAP_PX=0.05`（≈64px）、`MAX_TRACKS=2`。

**调用关系**：被 `run_pipeline.py:74`（每目一个实例，在 `match_hands` 之前原地覆盖 label）、`tools/hand_3d_d435/live_demo.py:66` 等使用。

### hand_3d/track3d.py

**作用**：遮挡传播（两件套）：`HandSlotTracker` 为每个槽位（hand_0/hand_1，`match_hands` 已按 left_label 排序：Left 前）维护 αβ 滤波的 3D 位置/速度，缺手帧生成预测 3D；`PseudoHandPair` 把预测 3D 包成伪 pair（`l_idx/r_idx=-1`、`valid_count=0`、err=inf）并入同一次 refine 批处理——预测 3D → `tri.project` → crop 的现成路径做救援重检。幻觉防护：`max_lost` 硬顶（默认 15，长缺口保持 absent 不得幻觉）、`propagated` 列可过滤、`--propagate-max 0` 全关。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `PseudoHandPair` | dataclass：`result`、`left_label`、`l_idx=-1`、`r_idx=-1` | 丢失槽位的传播重检载体（接口与 `HandPair` 平替） | — |
| `HandSlotTracker` | `(max_lost=15, alpha=0.5, beta=0.1, debug_log=None)` | 两槽位 αβ 跟踪器；槽位 label 变化 → 槽位重置（换手防旧状态污染） | — |
| `HandSlotTracker.observe_slot` | `(slot, label, pts3d, t)` | 真实检测回写：label 变化重置；首见初始化；`dt > max_lost` 重初始化直接采信本次观测（防陈旧速度外推冲掉新鲜观测）；否则 αβ 预测-修正（NaN 点保持纯预测） | 更新槽位状态 |
| `HandSlotTracker.mark_lost` | `(slot, t)` | 救援失败计数丢失 | 递增 lost |
| `HandSlotTracker.predict` | `(slot, t)` | 恒速外推 `x + v·(t − last_t)`；从未见过或丢失超限 → None | np.ndarray 或 None |
| `HandSlotTracker.slot_label` / `debug` / `close` | — | 查询当前槽 label / 写事件日志 / 关闭 | — |
| `make_pseudo_pair` | `(pred, label)` | 预测 3D → `PseudoHandPair`（err 全 inf 使 `_adopt` 判据退化） | `PseudoHandPair` |

**关键数据**：槽位状态含 `label/x/v/last_t/lost`；debug CSV 头 `frame,slot,event,label,lost`。注释：基线会话 `222_000008` hand_1 检出 57%、5 段 ≤3 帧短缺口；长缺口 45/89 帧保持 absent。

**调用关系**：被 `run_pipeline.py:73`、`tools/hand_3d_d435/live_demo.py:67`、`mono_assign.py:34`、`fill_track.py:32` 使用。`track3d.py` 依赖 `stereo_triangulate.TriangulationResult`。

### hand_3d/smoother.py

**作用**：3D 域因果时序平滑——三角化之后的最终抖动抑制。对 (2,21,3) 左目相机系 3D 关键点逐点跑 One-Euro 自适应低通（复用 `hand_detection.hand_pipeline_mediapipe.OneEuroFilter3D`）。平滑放在三角化之后的米制 3D 域而非 2D：精修 2D 与粗 2D 来源不同，在最终 3D 上统一平滑才能消除两种来源切换时的跳变。防污染：槽位 label 变化或"空→有"跳变时重置该槽全部滤波器。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `Hand3DSmoother` | `(freq_min=3.0, beta=0.3, dcutoff=1.0)` | 每 (slot, kpt) 一个 `OneEuroFilter3D`，时间基准 perf_counter 毫秒 | — |
| `Hand3DSmoother.update` | `(hands3d, labels, valids=None)` | 一帧平滑：`valids`（每槽有效点数 ≥8 视为有手）；无效点保持 NaN 不喂滤波器（数据诚实，渲染层自行处理） | (2,21,3) float32 |

**调用关系**：被 `run_pipeline.py:72`（主循环因果平滑，渲染用）与 `tools/hand_3d_d435/live_demo.py:68`、`probe_live_consistency.py:42` 使用。

### hand_3d/perspective_crop.py

**作用**：两阶段透视裁剪精修——Hur et al. 2025（Eurographics）"Perspective Crop Based Egocentric Hand Pose Estimation via Fisheye Stereo Vision" 的落地实现。原理：鱼眼边缘手形畸变是 MediaPipe 全图检测误差主要来源；粗三角化给出 3D 位置后按 3D 投影在每只眼取手部 ROI，放大成 256² 裁剪图（默认取自矫正图 = 已去鱼眼畸变），裁剪图重检测 21 点（手更大、畸变更小 → 更准），再逆变换回原始图像素二次三角化。采纳判据：精修结果有效点数 ≥ 粗结果且平均重投影误差 < 粗结果，否则保留粗结果（**精修永不回退精度**）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_centroid3` | `(points)` | 3D 质心（≥ `MIN_VALID_POINTS` 个有效点的中位数） | np.ndarray 或 None |
| `_rescue_near` | `(res, coarse)` | 伪救援位置门限：精修 3D 质心须在预测质心 `RESCUE_MAX_DIST` 内 | bool |
| `RefinedPair` | dataclass：`pair`、`kpts_l_raw/kpts_r_raw`、`result`、`used`、`reason`、`conf_l/conf_r` | 一只手的粗匹配 + 精修结果；`left_label` 属性兼容 `overlay_view` 的 `HandPair` 接口 | — |
| `CropRefiner` | `(tri, crop_size=256, pad_ratio=0.5, crop_source="rect", max_err, max_depth, refine_det_l, refine_det_r, epi_y_align=False, kpt_soft_thr=0.15)` | 精修器；`refine_det_l/r` 为每目独立裁剪图检测器（num_hands=1, smooth=False，与全图管线隔离防追踪先验串扰） | — |
| `CropRefiner.rect_to_raw` | `(px_rect, tri, side)` | 矫正图像素 → 原始(畸变)图像素（inv(K_rect) → R^T → 加畸变，与 `_self_test` 同算法） | (N,2) |
| `CropRefiner._make_crop` | `(img, proj_pts)` | 按投影点取 ROI → pad 0.5 → 正方形原生分辨率 crop；有效点 <4 / 边长 < `MIN_CROP_BOX` 放弃 | (crop, x1, y1) 或 None |
| `CropRefiner._refine_side` | `(img, proj_pts, det, side, to_raw=True)` | 单侧裁剪检测（空检测重试一次）→ 关键点回全图像素（to_raw=False 返回矫正图像素供对极对齐） | (21,2) 或 None |
| `CropRefiner._adopt` | `(pair, kpts_raw, conf_raw, coarse_l, coarse_r)` | 候选（精修/置信度加权 blend）重三角化 → 真 pair 严格优于粗结果才采纳；伪 pair 走绝对质量门槛（≥8 点且 err ≤ `RESCUE_MAX_ERR` 且位置近）防重复救援；失败返回粗结果（伪 pair 标记 `propagated`） | `RefinedPair` |
| `CropRefiner._blend_w` | `(conf)` | 逐点 blend 权重 `w = 0.25 + 0.5·c ∈ [0.25, 0.75]`；无置信度标量 0.5；conf < `kpt_soft_thr` 的点 w=0（完全回退粗检） | 权重 |
| `CropRefiner.refine` | `(pair, rect_l, rect_r, raw_l, raw_r, coarse_l, coarse_r)` | 整手精修（MediaPipe 逐 crop 路径）：投影 → 裁剪 → 检测 → （可选对极 y 对齐）→ 转 raw → `_adopt` | `RefinedPair` |
| `CropRefiner.refine_batch` | `(pairs, rect_l, rect_r, raw_l, raw_r, coarse_l_src, coarse_r_src)` | 整帧批量精修：crop 收集 → 单次 `detect_batch` → 分发（为 `batch_capable` 检测器保留，当前无启用后端；检测器无 `detect_batch` 时逐手回退 `refine()`） | `RefinedPair` 列表 |

**关键数据**：

- 常量：`MIN_CROP_BOX=20`、`RESCUE_MAX_ERR=3.0`（px，伪救援采纳的绝对误差上限）、`RESCUE_MAX_DIST=0.15`（m，伪救援位置门限）。
- 对极 y 对齐（`epi_y_align`，构造默认关、`run_pipeline` MP 路径已接线开）：矫正后左右目同名关键点必共行，实测存在 ~+3px 系统性竖直视差——在 rect 空间强制 `y=(y_l+y_r)/2` 再转 raw。
- 检测分辨率 = 裁剪原生边长（256~512 之间，`detect_size = min(max(s, crop_size), crop_size*2)`），保留全部像素信息。
- `RefinedPair.reason` 取值：`ok` / `ok-refined` / `ok-blend` / `no-crop-det` / `not-better` / `coarse-invalid` / `propagated`。

**调用关系**：被 `run_pipeline.py:71`（构造 `CropRefiner`）使用；依赖 `stereo_s80m.stereo_triangulate`（`StereoTriangulator` 及默认阈值、`MIN_VALID_POINTS`）。

### hand_3d/mp_gpu.py

**作用**：MediaPipe GPU delegate 直连封装——stage-1 检测提速。`hand_detection.hand_pipeline_mediapipe` 写死 CPU delegate（共享模块不改），本模块自建 `vision.HandLandmarker` 支持 `BaseOptions.Delegate.GPU`，输出 `DetectedHand` 结构与 `MediaPipeDetector` 完全一致（几何层零改动）。GPU delegate 可能 SIGSEGV（进程内 try/except 拦不住）→ `smoke_test_gpu()` 用子进程冒烟，通过才切 GPU。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `smoke_test_gpu` | `(model_path, timeout=90.0)` | 子进程跑"GPU delegate 创建 + 3 帧推理"，stdout 含 `GPU_SMOKE_OK` 才算通过 | bool |
| `FastHandLandmarker` | `(model_path, num_hands=2, det_conf=0.5, track_conf=0.5, delegate="cpu", smooth=True, freq_min=5.0, beta=0.05, dcutoff=1.0)` | `RunningMode.VIDEO` + 自增帧号时间戳（1.0.1 要求单调）；2D One-Euro 平滑逻辑与共享管线一致（归一化坐标域，按 (手序号, 点序号) 维护滤波器） | — |
| `FastHandLandmarker.detect` | `(frame_bgr)` | BGR → SRGB mp.Image → `detect_for_video` → 亚像素 `DetectedHand` 列表（label 取 `handedness[0].category_name`） | list |
| `FastHandLandmarker.reset` / `close` | — | 清滤波器/时间戳；释放 landmarker | — |

**关键数据**（代码注释实测，2026-08-17，RTX 5090 + mediapipe 1.0.1）：GPU delegate 3.0ms/帧 vs CPU 7.8ms/帧（单目），创建 ~1.1s 一次性成本；GPU delegate 双线程反而慢（GL 上下文竞争 0.84×）→ GPU 时两目必须顺序；与 CPU 结果关键点最大差 2.77px（fp16 数值差异，已知漂移）。

**调用关系**：被 `detector.py:86`（`delegate="gpu"` 惰性 import）、`run_pipeline.py:389`（`--mp-delegate auto` 冒烟）、`tools/hand_3d_d435/live_demo.py:419` 使用；探针 `probe_mp_gpu.py:24` 直接测。

### hand_3d/postprocess.py

**作用**：离线后处理——间隙插值 + 零相位速度自适应平滑（"平衡"档）。两阶段都在主循环之后、parquet 落盘之前，只在 `--video-smooth offline` 跑：① `fill_gaps` 对 ≤max_gap 的缺手短间隙逐点逐轴插值（优先三次，上下文不足退二次/线性），**覆盖**传播外推值并置 `propagated=True`；长缺口不动（不幻觉）。② `offline_smooth` 逐槽位逐点逐轴按连续有效段 savgol(7,3) 零相位平滑 + 速度自适应混合。平滑结果写入**新列** `observation.keypoints.hand_3d_smoothed`（float32 [2,21,3]），`hand_3d` 列保持原始精修值语义不变。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `ReplayPair` | `(label, points_3d, mean_error, valid_count)` | 渲染用伪 pair（把 parquet 行还原成 `overlay_view` 可消费结构，同 `render_stereo._ReplayPair`） | — |
| `_runs` | `(mask)` | bool mask 的连续 False 段 → [(a,b)] 闭区间 | list |
| `fill_gaps` | `(rows, max_gap=15)` | 间隙插值：锚点 = 间隙两侧真实检测帧（present 且非 propagated），优先左右各 2 帧三次插值（<4 退线性）；回写 `hand_3d`/`present=True`/`propagated=True`/label 沿用 | 填充帧-槽数（原地改 rows） |
| `offline_smooth` | `(rows, sg_window=7, sg_poly=3, v0=0.08, fps=25.0, still_window=21)` | 逐槽逐点逐轴：连续有效段 savgol 零相位 + 速度自适应混合 `w=exp(-(v/v0)²)`（速度从 savgol 去噪输出测量）；静止 v_sg<20mm/s 用长窗 21、20-80mm/s 线性混入、>80mm/s 基准窗；跳变阻尼（帧间位移 >`JUMP_THR` 的突变点 ±still_window/2 内保 raw） | (N,2,21,3) float32 |
| `pairs_from_row` | `(row, key="observation.keypoints.hand_3d_smoothed", min_points=4)` | parquet 行 → `ReplayPair` 列表（第二遍渲染 pass 用）：按"有效点数 ≥ min_points"纳入渲染，保证间隙期骨架连续 | list |

**关键数据**：`JUMP_THR = 0.010`（m/帧 = 10mm/帧；注释：噪声 σ=1.2mm 下帧差 ~1.7mm，10mm≈6σ 不误触发）。注释记录跳变阻尼效果：长窗零相位对阶跃有预振铃（合成测试静止段边界 105-115 帧 wobble 3.11mm、峰值 68mm），阻尼后边界无预振铃、内部（30-100 帧）wobble 1.19→0.37mm。

**调用关系**：被 `run_pipeline.py:75`（`fill_gaps` + `offline_smooth`）、`tools/hand_3d_d435/run_pipeline_d435.py:40`、`tools/hand_3d_d435/tools/render_keypoints_parquet.py:173` 使用；`run_pipeline.py:291` 二次 import `pairs_from_row`。

### hand_3d/renderer_3d.py

**作用**：3D 旋转视角骨架渲染器——numpy + cv2 自写透视投影（零额外依赖）。虚拟相机绕双手质心匀速旋转（默认整段视频转 2 圈，仰角 25°），五指分色骨架（与 2D demo 同一套 `FINGERS` 色表）+ 掌心灰连接 + 腕部白圆 + 地面网格 + 腕部深度标注 + 相机系坐标轴 + HUD。对比 matplotlib 3D：本渲染器 2-5ms/帧（matplotlib 60-150ms/帧）、视角完全确定、原生 BGR 直进 VideoWriter。坐标系：左目相机系（OpenCV 约定，+X 右/+Y 下/+Z 前，米）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `PALM_CONNECTIONS` / `FINGER_CHAINS` | 常量 | 掌心连接（0-1/0-5/5-9/9-13/13-17/0-17）；每指画法链（拇指 1→2→3→4，其余 0→掌根→MCP→PIP→DIP→指尖） | — |
| `RotatingSkeletonRenderer` | `(width=1280, height=720, fov_deg=45.0, revolutions=2.0, elevation_deg=25.0, ground_grid=True, depth_labels=True, bg_color, text_color)` | 渲染器；焦距 `f = (h/2)/tan(fov/2)` | — |
| `RotatingSkeletonRenderer._look_at` | `(eye, target)` | 虚拟相机基 (right, up, fwd)；左目相机系 Y 向下 → 世界"上" = −Y | 基向量 |
| `RotatingSkeletonRenderer._project` | `(pts3d, right, up, fwd, eye)` | (N,3) → ((N,2) 像素, (N,) 视线深度) | 投影 |
| `RotatingSkeletonRenderer._draw_hand` | `(img, proj, fin, label, err)` | 五指分色 + 掌心 + 腕部白圆（指尖 7px/关节 5px/腕 9px + 深色描边，与 `draw_hand` 同风格） | 原地画 |
| `RotatingSkeletonRenderer._draw_grid` / `_draw_axes` | — | 地面网格（最下手部点下方 0.05m 平面，0.05m 间距）/ 相机系原点坐标轴（0.1m 长） | 原地画 |
| `RotatingSkeletonRenderer.render` | `(hands3d, labels=("",""), errs=(nan,nan), frame_idx=0, total=1, title=...)` | 整帧渲染：质心/跨度 → 相机距离 `clip(2.2·half/tan(fov/2), 0.2, 1.5)` → 旋转角 `2π·revolutions·frame/total` → painter 算法（远手先画）→ HUD | (H,W,3) BGR 帧 |

**关键数据**：默认画布 `RENDER_SIZE=(1280, 720)`（`run_pipeline.py:81` 定义）；腕部深度标注 `"Left/Right z=0.XXm err=0.0px"`；无有效点帧显示 "no valid 3D hand keypoints"。

**调用关系**：被 `run_pipeline.py:76`、`tools/hand_3d_d435/live_demo.py:69`、`run_pipeline_d435.py:41`、`tools/render_keypoints_parquet.py:46` 使用；复用 `hand_detection.hand_pipeline_mediapipe.FINGERS`。

### hand_3d/io.py

**作用**：数据 IO——会话元数据读取 + LeRobot 风格 parquet 打包/落盘。schema 与 `hand_triangulate.py` 完全一致（保证 `render_stereo` 重放路径兼容），新增三列：`observation.keypoints.stage2`（每手 bool，精修是否被采纳）、`observation.keypoints.propagated`（每手 bool，传播/插值标记）、`observation.keypoints.hand_3d_smoothed`（离线平滑结果）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `find_video` | `(session_path, cam)` | 同 `hand_triangulate._find_video`（chunk-0000 → 根目录回退） | 路径或 None |
| `load_episode_meta` | `(session_path)` | 读 `meta/episodes/*.parquet` + `meta/tasks.jsonl`，异常时默认 0 | (episode_index, task_index) |
| `load_timestamps` | `(session_path)` | `timestamps.json` → {frame_index: timestamp} | dict |
| `pack_2d` / `pack_3d` / `pack_errors` | `(hands)` / `(pairs)` / `(pairs)` | 84 维 2D（缺手全零）/ 126 维 3D（无效 NaN）/ 每手平均误差（`valid_count` 为 0 时 NaN） | list |
| `pack_stage2` | `(pairs)` | 每手精修是否被采纳（`p.used`）→ bool list [2] | list |
| `write_parquet` | `(rows, path, drop_keys=())` | dict 列表 → parquet（zstd）；`drop_keys` 跳过不存在的列（如 causal 模式不写 `hand_3d_smoothed`） | 写入路径 |
| `merge_info_json` | `(session_path, drop_keys=())` | 纯追加 `meta/info.json` features（含移除上次写入的 drop_keys） | 写文件 |
| `merge_stats_json` | `(session_path, rows)` | 纯追加 `meta/stats.json` 的 mean/std/min/max 统计 | 写文件 |

**关键数据**：`FEATURES_ADD` 注册 11 个 feature 键（dtype/shape 与 parquet 列一致，含 `stage2`、`propagated`、`hand_3d_smoothed` 三个扩展）；`DIM_2D=84`、`DIM_3D=126`；`action` 列仍为占位 `[0.0]`。

**调用关系**：被 `run_pipeline.py:78`、`tools/hand_3d_d435/live_demo.py:71`、`run_pipeline_d435.py:43`、`probe_align_overlay.py:57` 使用。

### hand_3d/video_writer.py

**作用**：管道视频写器——渲染段提速。旧路径（`render_stereo.create_video_writer/finalize_video`）两段式：cv2 mp4v 写临时 avi（CPU 编码 ~15ms/帧）+ 事后 libx264 转码（再花一遍全片时间）。本模块用 ffmpeg 管道单段直出：rawvideo bgr24 → stdin → H.264 → mp4，编码移出主线程（ffmpeg 子进程），主循环只剩帧拷贝。编码器逐级回退：nvenc（lerobot ffmpeg 带 `h264_nvenc`）→ libx264（管道）→ 旧两段式 mp4v。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `find_ffmpeg` | — | 找一个能跑起来的 ffmpeg（`-version` 退出码探测，结果缓存）：`FFMPEG_BIN`（默认 `~/miniconda3/envs/lerobot/bin/ffmpeg`）→ `shutil.which("ffmpeg")` → `/usr/bin/ffmpeg` | 路径或 None |
| `has_nvenc` | `(ffmpeg=None)` | 该 ffmpeg 是否带 `h264_nvenc`（`-encoders` 探测，结果缓存） | bool |
| `PipeVideoWriter` | `(out_path, fps, width, height, codec, codec_args)` | ffmpeg 管道写器：`-f rawvideo -pix_fmt bgr24 -i - ... -c:v codec ... -movflags +faststart` | — |
| `PipeVideoWriter.write` | `(frame_bgr)` | 写一帧；BrokenPipeError → latch 失败（调用方可忽略，parquet 数据不受影响） | bool |
| `PipeVideoWriter.close` | — | 关 stdin 等 ffmpeg 退出（60s 超时）；失败删半成品 mp4 | 最终路径或 None |
| `Mp4vWriter` | `(out_path, fps, width, height)` | 旧两段式路径包装（惰性 import `render_stereo`），统一 `.write/.close` 接口 | — |
| `create_video_sink` | `(out_path, fps, width, height, encoder="auto")` | 按 encoder 创建写器：auto=nvenc→libx264→mp4v 逐级回退；nvenc 参数 `-rc vbr -cq 23 -b:v 0`，libx264 参数 `-crf 23 -preset veryfast` | sink 对象 |

**调用关系**：被 `run_pipeline.py:77`、`tools/hand_3d_d435/*`、探针 `probe_nvenc.py:18`、`probe_compare3.py:30` 使用。

### hand_3d/run_pipeline.py

**作用**：两阶段管线主流程 + CLI，`tools/stereo_s80m/hand_3d/` 的入口。流程（Hur et al. 2025 落地）：左右目视频 → 每目独立 MediaPipe 2D 检测（stage-1，float 亚像素）→ 手性投票 → `match_hands` 跨目配对 → 鱼眼双目三角化（粗 3D）→ 透视裁剪精修（stage-2：3D 投影 → 手 ROI 256² 裁剪图 → 重检测 → 二次三角化）→ 3D 域平滑（默认离线零相位 + 第二遍渲染，`--video-smooth causal` 为因果 One-Euro）→ 3D 旋转视角渲染 + 矫正图并排叠加 → H.264 视频 + hand_3d parquet。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_detect_orientation` | `(det_l, det_r, vc_l, vc_r, tri_normal, tri_swapped)` | 方向自检（同 `hand_triangulate` 的重投影误差法），True=需 swap | bool |
| `_rescue_too_close` | `(pred, pairs, thresh)` | 伪 pair 救援互斥：预测 3D 与任一真实 pair 有效点质心距离 < thresh 则跳过救援（防"一只手进两个槽"；`--rescue-min-dist` 默认 0.10m，0=关） | bool |
| `_slot_debug` | `(line)` | `HAND3D_SLOT_DEBUG` 环境变量指向文件时写槽位规划事实（诊断用） | 追加写文件 |
| `_best_slot_for` | `(pair, tracker, n, exclude=None, bias=None)` | 真 pair 槽位归属：几何（粗 3D 质心按槽位粗→精偏移校正进精空间，与 tracker 预测比距离，≤`_UNRELIABLE_GATE` 达标；两槽相差 ≤`_AMBIGUITY_MARGIN` 时标签优先）→ 标签复活（死亡槽唯一匹配）→ 冷启动 Left→0/Right→1 → 全无返回 -1（丢给双槽伪救援） | 槽位号或 -1 |
| `_DispTracker` | — | 按槽位记录相邻帧 3D 位移中位与一阶差分（抖动，≥8 共同有效点才计） | — |
| `_render_offline_pass` | `(vp_l, vp_r, rows, tri, renderer, out_dir, out_names, fps, w, h, args)` | 第二遍渲染 pass：重读视频逐帧渲染平滑后数据（rot + rect + 可选 2d 三个视频 sink） | 视频路径 dict |
| `run_session` | `(session_dir, args)` | 主流程（详见下文"数据流"节）：检测器/投票器/方向自检/精修器/平滑器/tracker 装配 → 主循环 → 离线后处理 → parquet → 视频 | stats dict |
| `_summarize` | `(stats)` | 汇总打印：匹配帧/双手帧/stage-2 采纳率/reason 分布/精修前后重投影误差/三档抖动/各阶段耗时/深度范围 | 打印 |
| `_write_metrics_csv` | `(stats, path)` | 逐帧指标 CSV（与已落盘基线 parquet 对比）：`frame_index,n_pairs,err_ours,valid_ours,stage2,err_base,valid_base` | 写文件 |
| `main` | `session_dir [--calib --max-err --max-depth --no-refine --propagate-max --rescue-min-dist --track-debug --crop-size --crop-pad --crop-source --no-smooth3d --freq-min --beta --every --no-video --render-2d --no-parquet --write-episode --compare --out-dir --det-parallel --mp-delegate --det-conf --track-conf --video-encoder --video-smooth --sg-window --sg-v0 --write-smoothed]` | CLI，全参数见 `--help`；`--write-episode` 才写 `<session>/data/keypoints/` 并追加 meta（默认只写 `keypoints_output/`，不动会话目录） | 退出码 |

**关键数据**：

- 阈值：`_UNRELIABLE_GATE = 0.15`（m，粗 3D 与活轨迹预测最小距离 >150mm → pair 不可靠）、`_AMBIGUITY_MARGIN = 0.04`（m，两槽距离差 ≤40mm 视为平手 → 标签裁决）；`--max-err` 默认 8px、`--max-depth` 默认 3m；`--propagate-max` 默认 15；`--crop-size` 256、`--crop-pad` 0.5、`--crop-source rect`；3D One-Euro `--freq-min` 3.0、`--beta` 0.3；离线平滑 `--sg-window` 7、`--sg-v0` 0.08（m/s）。
- 输出（默认 `keypoints_output/<tag>/<session>/`）：`hand_3d_refined/chunk-000.parquet`（zstd）、`hand_3d_rotating.mp4`（1280×720）、`stereo_triangulate_refined.mp4`（矫正图并排）、可选 `stereo_2d_refined.mp4`（原图 2D 叠加）、`track_events.csv`（`--track-debug`）、`metrics.csv`（`--compare`）。parquet 先于视频写盘：视频渲染失败不影响数据。
- stage-1 双线程（仅 CPU delegate，MediaPipe 推理释放 GIL 实测 ~1.9×；GPU 双线程 0.84× → gpu 顺序，`--det-parallel auto`）。
- 槽位规划修复注释：pair 不再按列表顺序占槽（双手同 label 或单 pair 时列表顺序会串槽）。

**调用关系**：调用本子包除 `mp_gpu`（经 `detector` 间接）外的全部模块 + `stereo_s80m.render_stereo`/`stereo_triangulate` + `hand_detection.hand_pipeline_mediapipe`；被 `hand_3d/__init__.py:25` 导出为 `main`。

### hand_3d/probes/probe_compare3.py

**作用**：探针——双配置硬对比（纯 MP 单阶段 vs 纯 MP 两阶段）。读两个 run 目录（同一会话、同默认后处理，仅 stage-2 不同）的 parquet + 视频：① 指标表（打印 + `compare_report.csv`）：err mean/p95、双手帧、propagated 数、label 翻转次数、raw/offline 抖动中位；② 三个并排视频（`montage/`）：`rect_side_by_side.mp4`（0.5×）、`rect_zoom4x.mp4`（跟随 B 手部 ROI 4× 放大，亚像素精度差在此可见）、`rot_side_by_side.mp4`。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_jitter` | `(h3)` | 与 `_DispTracker` 同口径的抖动中位（帧间共同有效点位移中位的一阶差分中位，mm） | float |
| `compute_metrics` | `(run_dir)` | 读 `hand_3d_refined/chunk-000.parquet` 计算全指标（含 hand_0/hand_1 label 翻转次数） | dict |
| `_montage_rect` / `_montage_zoom` / `_montage_rot` | — | 并排视频生成；zoom 版 ROI 中心 = B 每帧 hand_0 的 stage-1 左目 2D 点中位（5 帧滑动中值防跳变；注释：3D 质心会被深度离群点拉偏，实测中心 vs 2D 质心 p90 差 179px） | 视频文件 |
| `main` | `--a <dir> --b <dir> [--montage <dir>]` | 指标对比 + montage 视频 | 退出码 |

**关键数据**：帧率栏来自 run 日志硬编码（`fps_log = {"a": "101.4", "b": "26.6"}`）；指标表会话为 222_000008。**调用关系**：使用 `hand_3d/video_writer.create_video_sink`；独立 CLI。

### hand_3d/probes/probe_jitter.py

**作用**：探针——P5 抖动验收：对 offline 模式跑出的 parquet（`hand_3d`=原始精修值、`hand_3d_smoothed`=离线零相位平滑值）复现因果 One-Euro（与主循环 `Hand3DSmoother` 同参数 freq_min=3.0/beta=0.3，ts 用标称 25fps 帧间隔）作第三列对比。指标与 `_DispTracker` 同口径：`disp(t)` = 相邻帧共同有效点位移中位、`jitter(t) = |disp(t) − disp(t−1)|`。速度分带（25fps）：低速 disp<2mm(<50mm/s)、中速 2-6mm、快段 >6mm(>150mm/s)。验收线：低速段 smoothed jitter 中位 <1.0mm；快段 |smoothed−raw| 中位 <1.5mm；长缺口（>15 帧）smoothed 保持 NaN 不幻觉。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `_load` | `(path)` | 读 parquet → raw/smoothed/propagated 三维数组 | 三元组 |
| `_causal` | `(raw, labels)` | 复现主循环因果 One-Euro（label 变化重置滤波器） | (N,2,21,3) |
| `_band_stats` | `(name, col, mask, raw, disp)` | 速度分带统计：jitter 中位/p90 + 对 raw 的保真中位 | 打印 |
| `main` | `parquet`（argv） | 三列分带对比 + 传播/长缺口检查 | 退出码 |
| `synth_still` | `(n=200, sigma_mm=1.2, seed=7)` | 合成静止手测试：噪声模型来自 P2 实测（GPU delegate 2D 抖动中位 0.72px（1280×800，fx_rect=362）→ z=0.5m 处 3D σ ≈ 0.72px×0.5m/362px×1.48 ≈ 1.2mm）；含快段台阶（帧 120-160 匀速 40mm/帧平移）验证零相位无滞后 | 退出码 |

**调用关系**：复用 `hand_detection.hand_pipeline_mediapipe.OneEuroFilter3D`；独立 CLI（`--synth` 走合成测试）。

### hand_3d/probes/probe_mp_gpu.py

**作用**：探针——GPU delegate 冒烟 + 单目计时 + CPU/GPU 关键点差值。只读不落盘。用法：`python probes/probe_mp_gpu.py [video_path]`（默认 `data/recordings/222/222_000008/videos/stereo_left/chunk-0000/stereo_left.mp4`）。

**类/函数**：脚本式（无函数导出）。流程：`smoke_test_gpu(MODEL)` → 读 20 帧 → CPU/GPU 各预热 3 帧 + 计时（ms/帧）→ 逐帧同 label 手对算关键点 `max|Δ|`（打印中位/p90/max）。

**调用关系**：使用 `hand_3d/mp_gpu.FastHandLandmarker` 与 `smoke_test_gpu`；独立 CLI。

### hand_3d/probes/probe_nvenc.py

**作用**：探针——管道写器冒烟：nvenc/libx264 直出 + BrokenPipe latch + 可解码性。只写 `/tmp`，不落仓库。用法：`python probes/probe_nvenc.py`。

**类/函数**：脚本式。流程：打印 `find_ffmpeg()`/`has_nvenc()` → 对 4 种 encoder（`auto/nvenc/libx264/mp4v`）各写 30 帧随机帧并回读验证可解码性与首帧均差 → BrokenPipe 测试（非法编码器 `h264_bogus_codec` 让 ffmpeg 启动即退出，断言 `latch=True, final=None`）。

**调用关系**：使用 `hand_3d/video_writer`（`create_video_sink`、`find_ffmpeg`、`has_nvenc`、`PipeVideoWriter`）；独立 CLI。

## 数据流

完整链路（标定 → 2D → 三角化 → 槽位跟踪 → 平滑 → 渲染/parquet）：

**0. 标定准备（采集前）**

- `capture_calibration.py`：SDK 出厂标定静态 yaml（`FAYSSENSE_SDK_DIR` 环境变量定位 `config/calib/calib.yaml`）→ 内参缩放到 1280×800 → `config/s80m_stereo_calibration.json`。
- `export_calibration.py`（可选，连相机）：设备 ROM `FAYS_VIK_GetCalibrationParam` → 会话级 `calibration/head_stereo.json` + 设备级 JSON + DumpCalib yaml 备份。
- 标定查找链（`stereo_triangulate.load_stereo_calibration`）：显式 `--calib` → episode `calibration/head_stereo.json` → 设备级 `config/s80m_stereo_calibration.json`。

**1. 采集**：主程序 `ui/main_window.py:703` 子进程运行 `read_stereo_rgb.py --pipe -`（启动时按 FTDI 设备名+USB 接口号自动解析端口重写临时 yaml），左右目 JPEG + IMU 二进制流经管道传回录制为 `videos/stereo_left|right/` + IMU parquet + `timestamps.json`。

**2. 离线 3D 手部管线**（`hand_3d/run_pipeline.py`，核心链路）：

1. **stage-1 检测**：每目独立 `MediaPipeDetector`（`--mp-delegate auto`：GPU 子进程冒烟通过才切，失败回退 CPU；CPU 时两目双线程并行）→ 左右各 ≤2 只 `DetectedHand`（float 亚像素 21 点）。
2. **手性投票**：每目一个 `HandednessVoter.update` 原地覆盖 label（7 帧票仓严格多数 + 交叠守卫）。
3. **跨目配对 + 粗三角化**：`match_hands`（几何主判据，≤2×2 穷举）→ `StereoTriangulator.triangulate` 粗 3D（左目相机系，米）；先做左右方向自检（两种配对比平均重投影误差决定 `swap_cams`）。
4. **槽位规划 + 遮挡传播**：真 pair 经 `_best_slot_for` 按轨迹归属槽位（几何 + 标签裁决）；缺手槽位 `HandSlotTracker.predict` 生成预测 3D → `make_pseudo_pair` 伪 pair（`max_lost=15` 幻觉硬顶，`--propagate-max 0` 全关）。
5. **stage-2 透视裁剪精修**：`CropRefiner.refine`——3D 投影 → 矫正图手 ROI 256² 裁剪图（`crop_source="rect"`）→ 每目独立 num_hands=1 检测器重检测 → 对极 y 对齐（左右 y 取平均）→ `rect_to_raw` 回原始像素 → 二次三角化；采纳判据"严格优于粗结果"（伪 pair 走绝对质量门槛）。
6. **tracker 回写**：真 pair / 救援成功 → `observe_slot`（αβ 更新，label 变化重置）；失败 → `mark_lost`。粗→精偏移对（`slot_bias`）持续维护供槽位几何校正。
7. **因果平滑（渲染用）**：`Hand3DSmoother.update` 对 (2,21,3) 逐点 One-Euro；parquet 存精修原始值。
8. **parquet 落盘**：`io.write_parquet` → `keypoints_output/<tag>/<session>/hand_3d_refined/chunk-000.parquet`（LeRobot 风格列 + `stage2`/`propagated`/`hand_3d_smoothed`）。
9. **离线后处理**（`--video-smooth offline`，默认）：`fill_gaps`（≤15 帧短间隙插值，覆盖传播值）→ `offline_smooth`（savgol 零相位 + 速度自适应混合 + 跳变阻尼）→ 写入 `hand_3d_smoothed` 新列。
10. **第二遍渲染 pass**：重读视频，`RotatingSkeletonRenderer.render`（3D 旋转视角）+ `overlay_view`（矫正图骨架叠加）+ 可选 `overlay_view_2d` → `create_video_sink`（nvenc→libx264→mp4v 回退）→ `hand_3d_rotating.mp4`、`stereo_triangulate_refined.mp4`、`stereo_2d_refined.mp4`。

**3. 旁路/下游**：

- **单阶段后处理**（旧路径，无裁剪精修）：`hand_triangulate.py` 直接写 episode `data/keypoints/` parquet；`hand_benchmark.py` 做参数扫描基准。
- **纯渲染重放**（不重新检测）：`render_stereo.render_session_from_parquet` 从已落盘 parquet 出视频。
- **离线 SLAM 数据集**：`export_offline_slam_dataset.py` 会话 → 矫正灰度 PNG + images/imu CSV + `orb_calibration.yaml`（ORB-SLAM3 FileStorage 格式）→ `validate_offline_dataset.py` 本地复刻客户侧验收（格式键 + rectify 一致性 + 数据完整性）。
- **D435 复用**：`tools/hand_3d_d435/` 复用本模块的 `detector`/`identity`/`track3d`/`smoother`/`renderer_3d`/`video_writer`/`io`（仅替换数据源与部分几何），与 S80M 管线共享全部跟踪/平滑/渲染逻辑。

## 产物与交付目录（非源码）

- `tools/stereo_s80m/offline_slam_output/`：`export_offline_slam_dataset.py` 的默认输出根（`session_*` 数据集目录与 `.tar.gz` 包），首次运行后生成、仓库内不保留，产物不参与代码。
- `tools/stereo_s80m/dist/s80m_stereo_camera/`：客服交付版自包含副本（`read_stereo_rgb.py` + `export_calibration.py` + `pipe_consumer.py` 管道协议消费示例），与仓库侧同名脚本同源，改本目录脚本时需同步更新交付版。
