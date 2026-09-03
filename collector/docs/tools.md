# tools/

## 定位

`tools/` 是独立工具包集合，与主程序（`ui/`、`core/`）完全解耦：仓库其余代码不 import 它（全仓 grep `import tools`/`from tools` 无命中），它只单向复用 `tools/stereo_s80m/hand_3d/` 与 `tools/hand_detection/` 的既有组件（只读，不改任何现有文件），输出统一落在 `keypoints_output/` 下。

每个入口脚本自带 `sys.path` shim——把仓库根与 `tools/` 插入 `sys.path`，从而能直接 `from hand_3d_d435.*`、`from stereo_s80m.hand_3d.*` 自导入，**无需安装、无需改环境变量**（唯一例外：`tools/glove_package/` 内脚本要求 cwd 在包目录内，见下）。

## 目录总览

| 子目录/文件 | 一句话作用 |
| --- | --- |
| `tools/diag_color.py` | S80M 子进程 pipe 输出一帧的颜色通道诊断（需 `FAYSSENSE_SDK_DIR`） |
| `tools/diag_frame_layout.py` | ctypes 直连 FaysSense SDK 打印标定/帧布局/双目相关性（需 `FAYSSENSE_SDK_DIR`） |
| `tools/hand_3d_d435/` | D435 RGB-D 单目 3D 手部关键点独立模块（实时 demo + 离线批处理 + 验收探针） |
| `tools/hand_3d_d435/probes/` | 6 个离线验收探针（对齐/一致性/完整性/标签/传播/骨长） |
| `tools/hand_3d_d435/tools/` | 3 个配套小工具（parquet 渲染、交付版 demo 导出、标定提取） |
| `tools/hand_3d_d435/run_live_d435.sh` | 实时 demo 启动器（venv 选择 + 参数透传） |
| `tools/hand_3d_d435/run_d435.sh` | 离线批处理启动器（venv 选择 + 参数透传） |
| `tools/hand_3d_d435/calibration/` | 固化标定 JSON（`.gitignore` 第 48 行忽略，需先跑提取脚本生成） |
| `tools/hand_3d_d435/新手套接入.md` | 新手套（不同纹路/图案）接入验证说明 |
| `tools/glove_package/` | YOLO-World + RTMPose 手套识别/标注/训练自包含工具包（11 个 .py） |
| `tools/glove_package/使用说明.md` | 工具包安装与用法总说明 |
| `tools/glove_package/yolov8m-worldv2.pt` | YOLO-World 检测器权重（默认，被 `hand_3d_d435` 复用） |
| `tools/glove_package/yolo11n.pt` | 检测器训练基座权重（`train_detector.py` 默认 `--model`） |
| `tools/demos/` | 交付版 demo 与自检脚本（S80M 深度引擎、双目 2D、手套关键点、hdf5） | `docs/demos.md` |
| `tools/hand_detection/` | 手部检测：YOLO 手套检测（`best.pt`、`world_detector.py`）+ MediaPipe 裸手管线 | `docs/hand_detection.md` |
| `tools/stereo_s80m/` | S80M 双目工具：标定/三角化/渲染/离线 SLAM 导出（主程序子进程） | `docs/stereo_s80m.md` |
| `tools/hand_3d_s80c/` | S80C 双目实时裸手/手套关键点 demo（含自包含 SDK `third_party/`、`build_dist.sh` 分发包） | 包内 `README.md` |
| `tools/fayssense_depth_sdk/` | FaysSense VI Kit 深度引擎 SDK（C++，专有；S80C demo 自包含用副本） | 包内 `README.md` |
| `tools/models/` | 模型权重：`hand_landmarker.task`（MediaPipe 裸手关键点） | — |
| `tools/tests/` | 16 个回归/冒烟测试（11 离线 + 5 真机，见文末） | `README.md` 测试节 |
| `tools/weights/clip/` | CLIP ViT-B-32 大权重（gitignore，不随仓库分发） | — |

本文档详解覆盖 33 个 `.py` 文件（2 诊断 + 11 手套包 + 11 D435 模块 + 6 探针 + 3 小工具）；`demos/`、`hand_detection/`、`stereo_s80m/`、`hand_3d_s80c/`、`tests/` 等目录的脚本见各自文档。

## hand_3d_d435 模块

D435 RGB-D 单目 3D 手部关键点独立模块（与主程序解耦）。链路两版同源：

- **离线**（`run_pipeline_d435.py`）：RGB(1280×720) MediaPipe 2D（+ 手性投票）→ 原生深度(848×480 mm PNG) 前向对齐抬升 3D（彩色相机系，米）→ 单目槽位分配 → `HandSlotTracker` 遮挡传播 → tracker 前向+后向填充 + `offline_smooth` → 旋转渲染 + RGB 叠显 + parquet。
- **实时**（`live_demo.py`）：同上，仅把批处理段换成 αβ + OneEuro 在线平滑链，并加 M1/M3/M5/M6 展示路径稳定性修正（见下）。

### 子模块文件详解

#### `__init__.py`

**作用**：包说明文档串，无代码。声明模块只复用 `tools/stereo_s80m/hand_3d/` 组件（detector/identity/track3d/postprocess/renderer_3d/video_writer/io），不改任何现有文件，输出落在 `keypoints_output/`。

#### `depth_align.py`

**作用**：深度→彩色离线前向对齐 + 关键点深度采样。采集链路（`core/d435_camera.py`）不做 `rs2.align`，本模块逐深度像素投影：`P_d = ray·Z_d` → `P_c = R·P_d + t` → 彩色投影，z-buffer 保最近（0=无效哨兵），再空穴回填（848×480→1280×720 是 ~2.12× 上采样，rint 跳列跳行留 ~50% 空穴，3×3 min 邻域 3 轮回填，只写空穴不腐蚀有效值）。

公开接口：

| 接口 | 说明 |
| --- | --- |
| `load_session_depth_intr(session_dir)` | 读录制期 `calibration/head_stereo.json` → 深度内参 dict（fx/fy/cx/cy/width/height），失败返回 None |
| `load_calib(calib_path=None)` | 读固化标定 JSON（默认 `tools/hand_3d_d435/calibration/d435_color_calib.json`），缺失抛 `FileNotFoundError` 并提示先跑提取脚本 |
| `DepthAligner(color_intr, depth_to_color, depth_intr)` | 构造时预计算深度侧单位射线与 R/t（t 米→毫米） |
| `DepthAligner.align_depth_to_color(depth_mm)` | (dh,dw) uint16 毫米深度 → (ch,cw) float32 aligned 深度（0=无效） |
| `DepthAligner.sample_points(aligned, uv, band=None)` | (N,2) 像素坐标 → (N,) 深度 mm；3×3 窗口取中位（有效 ≥2 才出数），`band=(z_lo,z_hi)` 时先按深度带过滤（手缘点抗背景混入） |

**关键数据**：`_MAX_DEPTH_MM = 8000.0`（远背景离群上限）、`_FX_REL_TOL = 0.01`（深度内参交叉核对容差）。`python depth_align.py` 可直接跑合成自测（恒等外参逐点精确、t_x=25mm 平移、中位抗离群、回填不腐蚀）。

**调用关系**：被 `lift3d.py`、`replay_compat.py`、`run_pipeline_d435.py`、`live_demo.py`、`probe_align_overlay.py`、`probe_live_consistency.py` 导入。

#### `lift3d.py`

**作用**：2D `DetectedHand` + aligned 深度 → 彩色相机系 3D（米）。`LiftResult`/`D435Pair` 两个 dataclass mimic 双目管线的 `RefinedPair` 接口，使 `io.pack_3d`/`pack_errors`/`pack_stage2` 零改动可消费。`mean_error` 恒 NaN（单目无重投影概念），真实质量信号走旁挂 metrics。

| 接口 | 说明 |
| --- | --- |
| `LiftResult` | `points_3d`(21,3 米，NaN=无效)、`mean_error`（恒 NaN）、`valid_count`（实测有效点数） |
| `D435Pair` | `result`、`left_label`、`used`、`hand2d`(21,2)、`n_valid`、`det`、`measured`(21, bool：z 来自实测，False=补点) |
| `lift_hand(hand, aligner, aligned, band=True, complete=True)` | 深度采样抬升；`band`=两遍带约束采样（手深中位 zc ± 0.12m 带内取中位）；`complete`=缺失点补到 zc、x,y 按 2D 反投影（"保持 z、调 x,y"） |
| `apply_slot_zc(pair, zc, aligner)` | M5：补点深度锚定到槽级稳定 zc，只动 `measured=False` 的补点，实测点不动 |
| `gate_observations(pts, pred, gate=0.15, wholesale_frac=0.6)` | 时序一致性门：|观测−预测| > 0.15m 的点置 NaN（tracker 对 NaN 走纯预测）；可疑点 ≥60% 时返回 `(原样, True)` 触发槽位重置 |

**关键数据**：`BAND_HALF_M = 0.12`、`BAND_MIN_VALID = 4`（第一遍有效点不足退无约束）、`GATE_M = 0.15`。

**调用关系**：被 `fill_track.py`（`gate_observations`）、`mono_assign.py`（自测）、`run_pipeline_d435.py`、`live_demo.py`、`probe_live_consistency.py` 导入。

#### `mono_assign.py`

**作用**：单目双手槽位分配（`hand_0`/`hand_1` 连续身份），仿双目 `_best_slot_for` 的决策层级：

1. 冷启动：标签惯例 Left→slot0 / Right→slot1，无标签按检测序号；
2. 标签唯一命中存活槽 + 几何门（≤0.15m）；
2b. 标签唯一命中困境槽（上帧无真手）→ 不设几何门（恒速外推会漂移，label 是唯一可靠信号）；
3. 贪心几何：剩余手入最近未占用存活槽（质心 ≥4 有效点，不足退 2D 判据；label 冲突守卫）；
4. 互斥守卫：双真实且腕距 <0.10m → 两种排列取总 cost 更小者（须严格优于 ≥5mm 防抖动）；
5. 未见槽冷启（label 惯例/无标签单死槽兜底）；
6. 兜底丢弃。

| 接口 | 说明 |
| --- | --- |
| `assign_mono_slots(pairs, tracker, n, color_intr, lost_counts=(0,0), debug=False)` | `list[D435Pair]`(≤2) → `[slot0_pair|None, slot1_pair|None]`；`lost_counts` 供 2b 规则用 |
| `_cost(pair, slot_pred, color_intr)` | 3D 质心距槽预测（米）；质心不可靠退 2D 判据（预测投影 vs 2D 质心，像素距 ×Z/fx） |
| `_dbg(n, labels, pred, out, evt)` | `HAND3D_SLOT_DEBUG` 环境变量指向文件时追加决策日志（`atexit` 落盘） |

**关键数据**：`UNRELIABLE_GATE = 0.15`、`WRIST_MUTEX = 0.10`、`SWAP_MARGIN = 0.005`、`MIN_VALID_PTS = 4`。`python tools/hand_3d_d435/mono_assign.py` 可跑 7 项构造轨迹自测。

**调用关系**：import `stereo_s80m.hand_3d.track3d.HandSlotTracker`；被 `run_pipeline_d435.py`、`live_demo.py`、`probe_live_consistency.py` 调用。

#### `fill_track.py`

**作用**：离线缺失帧槽的 tracker 前向+后向填充（替代 `fill_gaps` 短桥接）。用与实时版同源的 `HandSlotTracker`（αβ、逐点 NaN 保持纯预测）沿时间轴前向跑一遍（真实观测 observe、缺失帧 predict），再反向跑一遍补首观测前的头部段；合并时真实观测优先、前向预测次之、后向兜底。轨迹是恒速外推，比线性插值更物理；门控与实时版一致（`gate_observations`）。`fill_gaps` 语义保留：被填帧 `present=True`、`propagated=True`、label 取最近 present 帧；从未见过的槽不幻觉 present。

| 接口 | 说明 |
| --- | --- |
| `tracker_fill(rows, max_lost=15)` | 原地修改 rows（parquet 行字典列表），返回填充帧-槽数 |
| `_forward_pass(h3, present, prop, labels, max_lost)` | 单方向 tracker 填充 → (N,2,21,3) 状态轨迹 |

**调用关系**：被 `run_pipeline_d435.py` 调用（`--propagate-max` 透传 max_lost）；自测在 `__main__`。

#### `render_overlay.py`

**作用**：D435 RGB 单视叠加渲染：2D 骨架（分色，复用 `tools/hand_detection` 的 `FINGERS` 色表 + 自含 `PALM_CONNECTIONS`）+ 逐关键点平滑深度 mm 标注 + 槽位 label/propagated HUD；另含伪彩深度叠层。

| 接口 | 说明 |
| --- | --- |
| `draw_overlay(rgb, hands2d, hands3d, labels, propagated, presents, frame_idx, total, title=...)` | 返回叠加帧（拷贝，不改原图）；absent 槽不画 |
| `blend_depth(rgb, aligned_mm, alpha=0.4)` | 伪彩深度叠层：300-1200mm 归一 JET（无效=不叠），α 混合回 BGR |
| `PALM_CONNECTIONS` / `FINGER_CHAINS` | 掌心连接（腕-掌子集）与每指画法链（拇指 1→2→3→4，其余 0→MCP→…→指尖） |

**调用关系**：被 `run_pipeline_d435.py`、`live_demo.py`、`probe_align_overlay.py` 导入。

#### `replay_compat.py`

**作用**：录制会话布局/标定兼容层（`live_demo --replay` 用）。自动探测三类会话：新会话（槽名随 GUI 用户命名 → `videos/D435_depth_rgb/`、`depth/D435_depth/`，`head_stereo.json` 可能是 D405 内参）、旧会话（`videos/d435_rgb` + `depth/d435_depth` + `head_stereo.json` 即 D435 内参）、双 RealSense 并存会话。

| 接口 | 说明 |
| --- | --- |
| `find_video_any(session)` | 依次尝试小写/大写槽名的 RGB 视频（兼容 `chunk-0000/` 布局） |
| `find_depth_dir(session)` | 依次尝试小写/大写槽名的深度 PNG 目录 |
| `load_session_depth_intr_any(session)` | head_stereo 深度内参；两文件并存且 fx 差 >1% 时改读 `D435_depth_rgb_calibration.json`（head_stereo 记录的"最后写入"设备不可靠） |

#### `glove_detector.py`

**作用**：黑手套检测器（YOLO-World 手套框 + RTMPose hand5 21 点 + 帧间稳定层）。检测契约对齐 `DetectedHand`（landmarks (21,2)、MediaPipe 拓扑 0=腕、label、score=框 conf），下游 voter/抬升/槽位/平滑/渲染零改动。

| 接口 | 说明 |
| --- | --- |
| `GloveDetector(weights=None, det_conf=None, ...)` | 权重默认 `tools/glove_package/yolov8m-worldv2.pt`；文件名含 "world" 走 world 后端（conf 0.05），否则普通 YOLO（`best.pt` 回退开关，conf 0.3）；惰性加载（ultralytics/torch/rtmlib 首次构造才 import，裸手启动零开销） |
| `detect(frame_bgr)` | → `list[DetectedHand]`，按 track_id 稳定序输出 |
| `reset()` / `close()` | 清空全部帧间状态 / 释放模型 |

**关键数据（帧间稳定层）**：`HandTracker` 贪心 IoU 匹配 + 框 EMA(α=0.7) + 运动门控 3px（静止帧免 RTMPose 推理、0.33s 强制刷新）+ 丢框持 3 帧；逐点 `OneEuroFilter2D`（freq_min=5.0/beta=0.05/dcutoff=1.0，归一化坐标域，键 (track_id, 点序号)）；退化族过滤（框外点 ≥16 / 唯一点 <15 / span <0.2×框对角线）；per-track 手性锁存（连续 3 票相同锁死；双手框中心距 <0.05×max(w,h) 冻结，镜像 voter latch）；`pose_conf_thr`（默认 0.3）持出低置信骨架；复活继承 `_REVIVE_MAX=90` 帧 / `_REVIVE_DIST=0.3×max(w,h)`（死亡 track 的 OneEuro/锁存/冻结缓冲 re-key 给新 track，消重捕捉闪烁）。

**调用关系**：只读复用 `stereo_s80m.hand_3d.detector.DetectedHand`、`hand_detection.hand_pipeline_mediapipe.OneEuroFilter2D`、`glove_package` 的 `world_detector` 与 `hand_tracker`（shim 把 tools/、glove_package、tools/hand_detection 三目录都插入 sys.path）；被 `live_demo.py` 调用（按 g 键热切换）。

#### `pose_backends.py`

**作用**：姿态关键点后端（黑手套链），把 RTMPose / MediaPipe 两个后端统一到同一契约，`GloveDetector` 只依赖契约、后端可热切换。契约：`backend(frame_bgr, bboxes=None) -> (kpts, scores)`——`kpts` (M,21,2) 全图像素坐标（21 点 MediaPipe 拓扑 0=腕，与 RTMPose hand5 同序，手性几何判据直接可用），全局失败返回 None；`scores` (M,21) 或 None（RTMPose=SimCC 逐点响应、MediaPipe=逐点 visibility，下游只取每手均值对照 pose_conf 门）；`backend.close()` 释放（重建前必调，幂等）；`backend.device` 实际推理设备。

| 接口 | 说明 |
| --- | --- |
| `RtmposePoseBackend(device)` | RTMPose hand5（onnxruntime，SIMCC 256×256）；首次构造模型未缓存时自动从 openmmlab 下载；CUDA EP 初始化失败自动回退 CPU |
| `MediaPipePoseBackend(model_path=None, device)` | HandLandmarker Tasks API（venv mediapipe 1.0.0 无 legacy `mp.solutions`，必须 `from mediapipe.tasks import python`）；默认模型 `tools/models/hand_landmarker.task`；IMAGE 模式整图 `num_hands=2` 检测 + 质心就近关联到入参 bboxes——实测（000005）框外扩 1.25 裁剪喂入 0/5 检出、整图 5/5 检出，输出仍按入参框顺序每框一行，未关联框吐零行 |

**关键数据**：`_MP_PAD = 1.25`（裁剪外扩，与 RTMPose bbox padding 同口径）、`_MP_MIN_CONF = 0.3`（检测/存在阈值放宽，由下游退化过滤 + pose_conf 门兜底保召回）。

**调用关系**：被 `glove_detector.py` 使用（后端热切换）；MediaPipe 后端对黑手套检出率低（整图 5/60 帧 vs world 每帧出框），主要供裸手/效果对比。

#### `run_pipeline_d435.py`

**作用**：离线批处理入口。用法 `./tools/hand_3d_d435/run_d435.sh <session_dir> [选项]`；产物（`keypoints_output/<tag>/<session>/`）：`d435_hand_3d_rotating.mp4`、`d435_rgb_overlay.mp4`、`hand_3d_refined/chunk-000.parquet`、`d435_metrics.csv`（`--track-debug` 追加 `track_events.csv`）。

参数：`--out-dir`、`--calib`、`--mp-delegate`（cpu/gpu，默认 cpu）、`--det-conf`（0.5）、`--track-conf`（0.5）、`--propagate-max`（15，槽位丢失硬顶）、`--depth-overlay`（overlay 视频叠伪彩深度）、`--video-encoder`、`--no-video`、`--no-parquet`、`--track-debug`。

**关键流程**：空帧不喂 voter（identity.py 空帧会清轨迹 → 重建期同 label 闪烁）；`gate_observations` wholesale 时借 `"\x00reset"` 标签触发槽位重置；收尾 `tracker_fill` + `offline_smooth(sg_window=7, sg_poly=3, v0=0.08, fps, still_window=21)`；parquet 列 schema 与双目管线同构（`pack_3d`/`pack_errors`/`pack_stage2`）。汇总打印每槽 present/propagated 比例、label 翻转数、腕位移 p50/p95、腕→中指 MCP 骨长。

内部函数：`_load_episode_task`（pyarrow 直读 episodes parquet + tasks.jsonl）、`_pack_2d_slots`、`_render_videos`、`_summarize`、`_write_metrics`、`_nan_pair`/`_pred_pair` 占位构造。

**调用关系**：import `stereo_s80m.hand_3d` 的 detector/identity/track3d/postprocess/renderer_3d/video_writer/io + 本模块其余全部子模块。

#### `live_demo.py`

**作用**：实时 demo 入口（模块内 1415 行的主体）。三窗口：`D435 live: RGB overlay`（骨架+深度标注+HUD）、`D435 live: 3D view`（3D 骨架，左键拖拽环绕/俯仰、`r` 复位；输入平移到首帧锁定的世界锚点，不拖拽时视角/网格/缩放完全静止）、`D435 live: depth`（aligned 深度 0.3-1.5m 伪彩）。按键：`q`/ESC 退出、`s` 截图（`keypoints_output/live_d435/`，三窗口各一张）、`d` 深度伪彩叠层切换、`g` 裸手/黑手套热切换（切回裸手重建 MediaPipeDetector 实例——`det.reset()` 只 close landmarker 不重建，实机踩过）、`r` 复位视角。

输入源三类：`LiveD435Source`（pyrealsense2 直开，rgb8 1280×720@30 + z16 848×480@30；流组合失败自动回退 1280×720 双流、848×480@60 双流——D405 深度/彩色须同分辨率；对齐用**本机实时内参/外参**，换设备自洽；设备独占，主程序预览开启时 EBUSY）、`ReplaySource`（`--replay <session_dir>`，按录制 fps 步调回放，经 `replay_compat` 兼容新旧布局）、`UvcSource`（`--uvc`：设备号/路径/URL，无深度 → 2D-only 单窗口模式，只产 2D 关键点渲染）。

主要参数：`--replay` / `--replay-pace`（30）、`--uvc`、`--calib`、`--delegate`（auto=GPU 子进程冒烟成功则 GPU）、`--fill`（默认 1 轮填洞：覆盖 91.6% vs 3 轮 94.0%，省 20ms/帧）、`--propagate-max`（15）、`--det-conf`/`--track-conf`（0.4）、`--glove` 及 `--glove-weights/-det-conf/-pose-conf/-nms-iou/-lost-timeout/-box-alpha/-freeze-max`、`--rs-serial`、`--exposure`（0=自动，运动模糊丢手时试 4000-10000µs）、`--stats`、`--depth-overlay`、`--no-window`、`--export`。`--export` 落盘四件套：`keypoints_2d.parquet`（xy 定长 float32[42]，像素）、`keypoints_3d.parquet`（xyz float32[63]，质心锚定 3D 米）、`render.mp4`、`rgb_overlay.mp4`（H.264+faststart，venv cv2 只能写 mp4v 故走共享 sink）。

展示路径稳定性修正（不回灌 tracker）：**M5** 补点深度锚定槽级 zc EMA（换手/首帧取实测，否则 0.5 混合，`apply_slot_zc`）；**M3①** `_SoftSmoother` 包装 `Hand3DSmoother`（label 变化/空→有重建且几何近 <0.1m 时 0.5 混合软衔接防 snap）；**M3②** wholesale 两帧确认（`_ws_agree` tol=0.30，或连续 ≥3 帧强制采信，单帧跳变只走预测显示）；**M1** `_CentroidAnchor` 质心锚定（每槽有效点中位质心走强 OneEuroFilter3D(3.0, 0.3, 0.3)，输出=输入+(ĉ−c)，共模平移在质心层吸收，手内形状原样保留）；**M6** 门控锁死豁免（逐点连续被门控 ≥5 帧且观测恢复有限 → 采信观测，防"预测外推越走越远永不恢复"）。

其他守卫：空帧不喂 voter；两手同 label（voter 重建期）先清空 label 走几何分配；手套模式整体跳过 voter（identity.py 新轨迹同帧双分配缺陷，实测同 label 锁死），身份由 `GloveDetector` per-track 锁存承担。

内部类/函数：`LiveAligner`、`LiveD435Source`、`ReplaySource`、`UvcSource`、`_OrbitControl`、`_SoftSmoother`、`_CentroidAnchor`、`_run_uvc_2d`、`_run_3d_chain`、`_ws_agree`、`_resolve_delegate`、`_write_export`/`_write_export_2d`、`_exp_meta`。

#### `tools/` 子目录（parquet 渲染 / 交付版导出 / 标定提取）

- `tools/hand_3d_d435/tools/render_keypoints_parquet.py`：任意 21 点 3D 关键点 parquet（列名同 `io.pack_3d`：`observation.keypoints.hand_3d`/`hand_3d_smoothed`）→ 3D 视角渲染视频。零相机依赖（不 import pyrealsense2/mediapipe，仅 numpy+cv2+pyarrow，`--smooth` 才需 scipy）。选项：`--col`（hand_3d/hand_3d_smoothed）、`--view`（rotating/static）、`--revolutions`、`--yaw`/`--elev`、`--smooth`（渲染前跑 `offline_smooth` 压抖）、`--scale`/`--shift`（尺度修正/平移）、`--frame N`（单帧 PNG 预览）、`--fps`、`--encoder`；支持多文件通配拼接、(N,2,21,3) 与 (N,21,3) 两种展平、`frame_index` 乱序保险；启动时打印数据体检（腕位移/骨长，骨长明显偏小提示 `--scale`）。
- `tools/hand_3d_d435/tools/demo_export_parquet.py`：跑交付版 `dist/d435_hands_demo_v1.1/d435_hands_demo.py` 并以**钩子**方式抓取每帧数据落 parquet（不复制管线代码、不改交付目录）：`install_hooks` 在 `_SoftSmoother.update`（抓槽位原始 3D/平滑 3D）与 `draw_overlay`（抓 2D/labels/present，`frame_idx-1` 收口成行）上挂钩；`Collector` 攒行，schema 复用 `tools/stereo_s80m/hand_3d/io.py` 的 `write_parquet`，与 `run_pipeline_d435.py` 产物同构、现有 probes 可直接读。用法：`venv/bin/python tools/hand_3d_d435/tools/demo_export_parquet.py <会话目录> --out-dir <dir>`；输出 demo 原三路视频 + `hand_3d_refined/chunk-000.parquet`。
- `tools/hand_3d_d435/tools/extract_d435_color_calib.py`：从真机 D435 提取彩色内参 + 深度内参 + depth→color 外参，固化到 `tools/hand_3d_d435/calibration/d435_color_calib.json`（模块离线对齐的唯一标定来源；外参语义 `P_color = R·P_depth + t`，t 单位米，D435 典型 |t|≈25mm）。必须用 venv python（pyrealsense2 只装在 venv）；`--session` 时与录制 `head_stereo.json` 深度内参交叉核对（差 >1% 告警"可能不是同一台设备录制"）；自带防呆（t 范数不在 [10,60]mm 告警、设备被主程序独占时报 busy）。产物含 serial/firmware 字段（运行时值，文档不转载）。

### probes/ 探针（合表）

全部用 venv python 运行，输入 parquet 的用 `run_pipeline_d435.py` 产物。

| 文件名 | 用途 | 输入 | 输出 |
| --- | --- | --- | --- |
| `probe_align_overlay.py` | 深度↔RGB 对齐验收：覆盖率 + 手部深度一致性 A/B（正确 vs 反号外参）+ 边缘距离（仅信息，实测噪声底之上不作判据） | `--session` 会话目录（+ `--calib`、`--frames` 抽样帧号） | `probe_align/` 下 `frame_XXX_align.png`（伪彩叠层）/`frame_XXX_edges.png`（深度边叠层）+ 覆盖率中位 ≥60% 与 手散布 ≤200mm 且 ≤0.8×反号 的 PASS/FAIL |
| `probe_live_consistency.py` | 重跑 live_demo 在线链（1 轮填洞 + αβ + Hand3DSmoother）与离线 parquet 逐帧比对 | `--session`（离线产物需先存在） | 两者都有限的腕点（点 0）深度差：中位 <10mm、p95 <30mm 判 PASS/FAIL |
| `probe_3d_completeness.py` | 3D 输出完整性与翻面事件（旋转渲染质量直接判据） | `--parquet`（读 `hand_3d_smoothed`） | 整手 21 点全有限帧率 ≥50%、翻面事件（相邻帧 z 跳 >300mm）≤2 起判 PASS/FAIL；另报有效点中位、腕点有限率 |
| `probe_label_stability.py` | 逐槽 label 翻转计数 | `--parquet` | 全片 0 次翻转（连续 present 帧间 Left↔Right）判 PASS/FAIL；退出码 0/1 |
| `probe_propagation.py` | propagated 比例 + absent 缺口直方图 | `--parquet` | 每槽 propagated <15%、absent 缺口全 ≤15 帧判 PASS/FAIL；另报硬 absent（3D 全 NaN 未幻觉）帧数；退出码 0/1 |
| `probe_bone_lengths.py` | 腕→中指 MCP 骨长自检（标定无关硬判据，剔除 propagated 帧） | `--parquet`（读 `hand_3d` 原始值） | 每槽中位 ∈[72,95]mm、IQR<25mm、<5% 帧出 [55,115]mm、两槽中位差 <10mm 判 PASS/FAIL；退出码 0/1 |

### launcher 脚本说明

- `tools/hand_3d_d435/run_live_d435.sh`：实时 demo 启动器。`VENV_PY` 环境变量可覆盖解释器（默认 `$REPO_ROOT/venv/bin/python`，即 collector/venv，含 pyrealsense2 2.58.3）；脚本自动 `cd` 回仓库根后 `exec "$VENV_PY" tools/hand_3d_d435/live_demo.py "$@"`——全部参数原样透传给 `live_demo.py`。例：`./tools/hand_3d_d435/run_live_d435.sh`（直连相机）、`... --replay data/recordings/222/222_000011`（回放）。
- `tools/hand_3d_d435/run_d435.sh`：离线批处理启动器，同款 `VENV_PY`/`cd` 逻辑，`exec` 到 `run_pipeline_d435.py`。用法 `./tools/hand_3d_d435/run_d435.sh <session_dir> [选项...]`。

## glove_package 工具包

YOLO-World 检测 + RTMPose 姿态的自包含工具箱（11 个 .py），专为戴手套场景优化（MediaPipe 在手套上完全失效，本方案实测检出 40/40）。**独立运行特性**：

- **cwd 必须在包目录内**：包内脚本用顶层裸导入（`import hand_common` / `import world_detector` / `from world_detector import iou`）互引，不挂 sys.path shim；默认模型路径 `MODEL_PATH` 以包目录 `_HERE` 锚定（`yolov8m-worldv2.pt`）。从包外 `python tools/glove_package/xxx.py` 会 ImportError——需 `cd tools/glove_package` 后再运行（`hand_3d_d435/glove_detector.py` 则不受此限：它自己把 `glove_package` 插进 sys.path）。
- **自包含**：不依赖 mediapipe（`hand_common` 刻意不 import 父目录）；依赖见 `requirements.txt`（torch/torchvision 由 `setup.bat` 装 CUDA 版，numpy、opencv-python、onnxruntime、rtmlib、ultralytics、openai-clip）；首次运行联网下载模型约 450MB（YOLO-World 权重 ~55MB + RTMPose hand5 ~56MB 缓存到 `~/.cache/rtmlib/` + CLIP 文本编码权重约 338MB）。
- **训练产物在包内 `runs/`**：`train_detector.py` 把相对 `--project` 锚定到包目录（ultralytics 会把相对路径解析到全局 settings 的 runs_dir，产物会落到包外），产出 `runs/hand_det/weights/best.pt`。`dataset/`、`captures1/`、`to_label/`、`verify_pose_out/` 等数据目录也在包内。

### 各文件

#### `world_detector.py`

**作用**：YOLO-World 开放词汇检测器——`auto_label.py` 和 `infer.py` 的唯一真值来源（标注框与推理框同一套配置）。

| 接口 | 说明 |
| --- | --- |
| `WorldDetector(model, prompt, imgsz, device, nms_iou, use_onnx)` | `.pt` 走 `YOLOWorld` + `set_classes(prompt)`；`.onnx` 走 `YOLO`（ONNX GPU 不可用——缺系统 CUDA Toolkit，强制 CPU） |
| `WorldDetector.__call__(src, conf, max_boxes, wh, reuse_boxes)` | BGR 帧 → `(boxes_xyxy [N,4], confs [N,])`，已过滤 + 两级 NMS + Top-N；`reuse_boxes` 帧跳过缓存 |
| `WorldDetector.postprocess(boxes, confs, w, h, max_boxes, nms_iou)` | 裁画面 + 丢非法/过小框（8px）+ 两级 NMS（中心距 <框宽 15% 且 IoU>nms_iou 判同一只手；中心有偏移只用 >0.85 极高阈值——解决"两只手靠近误删一只"）+ 按置信度 Top-N |
| `iou(a, b)` / `_center_close(a, b)` | 框 IoU / 中心距离判据（被 `hand_tracker.py`、`glove_detector.py` 复用） |
| `add_args(ap, max_boxes_default)` / `from_args(args, device)` | 给 argparse 挂公共参数（`--weights/--prompt/--imgsz/--conf/--max-boxes/--det-skip`），保证各脚本命令行一致 |

**关键数据（默认值）**：`DEFAULT_MODEL = "yolov8m-worldv2.pt"`、`DEFAULT_PROMPT = ["hand","glove"]`、`DEFAULT_IMGSZ = 320`（手占比大时比 640 更准更快）、`DEFAULT_CONF = 0.05`、`DEFAULT_NMS_IOU = 0.6`。

#### `hand_tracker.py`

**作用**：手部身份追踪——贪心 IoU 跨帧框匹配 + 稳定 track_id，解决"框按置信度排序帧间交换 → 关键点粘连"、"丢一帧就清状态 → ID 跳变"。`HandTrack` dataclass 记录框（EMA 平滑）、`last_good_kpts` 逐点冻结缓存、skip/lost 计数。

| 接口 | 说明 |
| --- | --- |
| `HandTracker(max_hands=2, iou_match_thr=0.3, lost_timeout=3, movement_thresh=3.0, skip_timeout=10, box_smooth_alpha=0.7)` | 每帧先 `update_detections`（贪心匹配 + 老化 + 清理超时 track + 限活跃数） |
| `get_boxes_for_pose()` | 只在运动超阈值/新 track/skip 超时（静止 10 帧强制刷新）时返回需要推理的框 |
| `update_pose_results(indices, kpts, scores)` | RTMPose 结果写回对应 track，记推理时框中心 |
| `get_results()` | → (boxes, kpts, scores, track_ids)；未出过 pose 的 track 吐 (21,2) 零数组（下游注意：`hand_3d_d435/glove_detector.py` 因此额外维护 `_posed` 集合过滤占位） |
| `update_last_good(track_index, point_index, value)` / `get_last_good(...)` | 逐点冻结缓存读写；**注意 `track_index` 是活跃列表下标**（与 `get_boxes_for_pose` 返回的 `self.tracks` 下标不一致，`glove_detector.py` 因此直接改 `track.last_good_kpts` 绕开） |
| `clear()` / `hand_count` | 重置 / 活跃手数 |

#### `hand_common.py`

**作用**：RTMPose 方案共用的关键点定义与绘制（21 点与 MediaPipe 同序：0=腕、1-4 拇指、5-8 食指、9-12 中指、13-16 无名指、17-20 小指）。

| 接口 | 说明 |
| --- | --- |
| `FINGERS` / `PALM_CONNECTIONS` / `JOINT_SPECS` | 五指 id 与 BGR 色表 / 掌心连接 / 关节表（拇指 CMC/MCP/IP，其余 MCP/PIP/DIP） |
| `angle_between()` / `compute_joint_angles(pts)` | 关节角（度，图像平面算——手指透视缩短时偏小） |
| `count_extended_fingers(angles)` | 伸直判定：拇指 MCP>145° 且 IP>150°；其余 PIP>150° 且 DIP>140° |
| `draw_hand(frame, pts, angles, kpt_scores, thr)` | 骨架+关节角绘制，低于 `thr` 的点画空心 |
| `draw_panel(frame, x, y, lines, width)` | 半透明信息面板 |
| `build_pose(device)` | RTMPose hand5（onnxruntime，首次自动下载 ~56MB） |
| `auto_device()` | CUDA > MPS > CPU |
| `RSCapture(serial, width, height)` | pyrealsense2 颜色流包装，接口对齐 `cv2.VideoCapture`（懒加载） |
| `pose_is_glove(kpts, scores, box, ...)` | 误检抑制：kpt 均值 ≥0.45、≥0.3 的高置信点 ≥15、点团 span ≥0.3×框对角线，三条全过才算真手套 |

#### `hand_demo_mmpose.py`

**作用**：摄像头实时手指/关节识别完整 demo（YOLO-World + RTMPose + HandTracker + 逐点冻结 + 数据采集）。快捷键：`q`/ESC 退出、`a` 关节角度、`s` 骨架、`b` 检测框、`h` 帮助、`i` UI 面板、空格暂停、`c` 抓拍、`r` 连续采集、`m` 只存漏检帧（最需要的难样本）。采集目录 `captures1/<时间戳>/`（`session.json` + `frames/000001.jpg` + `labels.jsonl` 预标注——存的是镜像后未画骨架的干净原图）。UVC 源做左右镜像 + 上下翻转（倒装校正），`--realsense` 不翻。遮挡处理：低置信点占比 ≥90% 整手冻结，否则按手指粒度冻结（伸直指/腕/高置信点才更新 `last_good`，阈值 0.2）。检测器可选自训权重（`YOLOWrapper`）；`wd.add_args` 挂载 `--weights`（默认锚定包目录）/`--prompt`/`--imgsz`/`--conf`/`--max-boxes`/`--det-skip`。另有 `--output` 出视频、`--data-out` 出逐帧检测 JSONL。

#### `infer.py`

**作用**：RTMPose 实时推理精简版。`--source` 支持摄像头序号/图片目录/视频文件，`--realsense` 走 D435 颜色流；`--detector` 可选 `world`（默认）/`rtmdet`（官方 RTMDet-nano，黑手套上失效，仅基线对照）/自训 `.pt` 路径。含 `--det-skip` 帧跳过（复用上一帧框，CPU 加速）。同一套误检抑制（`pose_is_glove` 连续 2 帧确认）+ 遮挡/逐点冻结（阈值 0.3）。快捷键同 demo 子集：`q`/ESC、`a`、`s`、`b`、`h`、空格。

#### `annotate.py`

**作用**：极简手部框人工标注工具（不用装任何标注软件）。`--dir`（默认 `to_label`）下 `images/` → 同级 `labels/`，保持子目录结构，输出 YOLO 格式（单类别 class id=0）。操作：左键拖拽画框、`u` 撤销、`c` 清空、`d`/空格/→ 下一张、`a`/← 上一张、`g` 跳到第一张未标注的、`q`/ESC 退出（自动保存）。**无框也写空文件**——标记"这张我看过了"，`prepare_dataset.py --merge` 会跳过空文件，无手图不会被当负样本混进训练集。`--max-width` 限制显示宽度（超宽图缩放显示、坐标仍按原图写）。标注要点（文件头）：框住整只手（指尖到手腕含袖口），偏大 10~30% 没问题。

#### `auto_label.py`

**作用**：手部框自动标注（零样本 / 自举双模式）。零样本=YOLO-World 文本提示直接检（默认），自举=`--model best.pt` 用已训检测器标新数据。**每张图只取置信最高的 N 个框**（`--max-boxes`，默认 1）是过滤误检的关键（原始输出 2.6 框/图）。`--eval` 模式不写标注，拿现有 `labels/` 当 ground truth 打分（IoU≥0.5 命中率 + 匹配框平均 IoU + 最差几张），对比图到 `viz_eval/`（文件名以 IoU 开头，最差排前）；`--overwrite` 才覆盖已标注。自动标注完输出核对图 `viz_auto/`，并提示用 `annotate.py` 人工复核。首次运行联网下载 YOLO-World + CLIP 权重（约 400MB）。注意默认 `--device mps`（此文件与其他文件的 auto_device 不一致，非 Apple 机器需显式传）。

#### `prepare_dataset.py`

**作用**：采集数据 → YOLO 检测数据集（单类别 hand）。两阶段：

1. `--captures <dir>` 扫 `labels.jsonl` 采集会话：有检测结果的帧从 21 点直接算 bbox（白捡的标注，`--pad` 外扩 0.12 与人工框风格对齐）；漏检帧导出到 `--to-label` 供人工标（**空标签绝不进训练集**——YOLO 空标签表示"无目标"，会把戴手套的手教成背景）。切分按 session（避免相邻帧泄漏进验证集），只有一个 session 时按时间顺序取末尾做验证集。
2. `--merge <dir>` 把人工标好的合并进 `--out` 数据集（跳过空文件），固定种子随机切 val。

产物 `dataset/`：`images/{train,val}` + `labels/{train,val}` + `data.yaml` + `viz/` 核对图（从写入的 .txt 反解坐标画框——画的就是模型真正读到的框）。

#### `train_detector.py`

**作用**：训练单类别手部检测器（替换在黑手套上输出 0 框的 RTMDet）。只训检测不训关键点（实测 RTMPose 给定正确框误差 3.9px，瓶颈只在"找不到手"）。默认 `yolo11n.pt`、100 轮、imgsz 640、batch 16；`--project`（默认 `runs`）**锚定到包目录**、`--name`（默认 `hand_det`）→ 最佳权重 `runs/hand_det/weights/best.pt`；`--resume` 从 last.pt 续训；`--no-augment-hsv` 关色相/饱和度增强（黑手套没色彩信息，算力让给几何增强，保留 hsv_v=0.4）。训练完提示 `python infer.py --detector <best.pt>`。要求先跑 `prepare_dataset.py` 生成 `dataset/data.yaml`。

#### `verify_pose.py`

**作用**：拿人工标好的框检验 RTMPose 关键点模型在目标手套上到底行不行——**大批量标注前先跑**（整套"只标框不标点"方案的前提验证）。标 10~20 张即可：`--dir`（含 `images/`+`labels/`）→ 逐张画骨架到 `verify_pose_out/`（`--out`），打印关键点置信均值。结论判定**必须逐张打开图看骨架是否贴合手指**（置信度在空白背景上反而更高）；贴合 → 只标框方案成立，不贴合 → 需改方案（标关键点微调或换模型）。

#### `fix_clip.py`

**作用**：修复 openai-clip 在 Python 3.12+ 的兼容问题（`clip.py` 的 `from pkg_resources import packaging` 在新 setuptools 中已移除 → 替换为 `import packaging.version`）。`find_clip()` 在 site-packages 定位 `clip/clip.py`，幂等（已打过补丁或不需要时跳过）。`setup.bat` 装完依赖后自动运行。

## 诊断工具

两个 FaysSense SDK（S80M 双目）诊断脚本，都从 `FAYSSENSE_SDK_DIR` 环境变量定位 SDK（`<FaysSense VI Kit Release 目录>`），未设置直接报错退出。

#### `diag_color.py`

**作用**：验证 S80M 子进程 pipe 输出一帧的颜色通道是否正确。以子进程跑 SDK 的 `read_stereo_rgb.py --pipe -`（cwd 设为 SDK 目录），按 4 字节大端长度前缀协议逐帧解析左右目 JPEG，`cv2.imdecode` 后按上下半分割（与 GUI 一致，竖图上下半各一目），同时存两份 PNG：`/tmp/s80m_left_BGR.png`（OpenCV 原样 BGR 布局）与 `/tmp/s80m_left_RGB.png`（BGR→RGB 转换）——用图片查看器看哪张颜色正常（如红色物体显示为红色）。**须在 DAQ venv 下运行**（`VENV_PY = sys.executable` 直取当前解释器）。

#### `diag_frame_layout.py`

**作用**：ctypes 直连 SDK 动态库，验证帧布局 + 打印标定数据。加载 `$FAYSSENSE_SDK_DIR/lib/fays_atrak/x86_64/Release/libfays_vikit.so`，按 C++ 符号名（`_Z31FAYS_VIK_CreateHandleWithConfigPPvPKc` 等）声明 5 个接口：创建句柄（配置文件 `config/fays_vikit.yaml`）、取版本、取一帧、取标定参数。输出：每个相机的内参/畸变/外参 `T_cn_cnm1` 与基线（mm）；抓一帧后按 2 列×4 行（每行 400 高）统计 8 个子区 mean/std、左右半帧互相关、行 0-1 vs 行 2-3 差异，保存 `/tmp/diag_full_frame.png` 与 `/tmp/diag_grid.png`。启动时预加载 OpenCV `.so.406`/`.so.4.2` 共享库（`RTLD_GLOBAL`）并设置 `QT_QPA_PLATFORM_PLUGIN_PATH` 指向 cv2 自带的 qt plugins，规避 SDK 自带 OpenCV 与本环境 cv2 的冲突。

## tests/ 测试

16 个测试脚本（11 离线 + 5 真机），离线测试加 `QT_QPA_PLATFORM=offscreen` 无需任何硬件；真机测试需对应设备在线。运行命令见 `README.md` 测试节。

| 文件 | 类型 | 一句话 |
| --- | --- | --- |
| `test_playback_multifps.py` | 离线 | 多帧率会话回放 |
| `s80m_signal_regression.py` | 离线 | S80M 信号回归（object 参数大整数时间戳） |
| `s80m_50fps_decimation_test.py` | 离线 | S80M 50→30 抽帧录制决策 |
| `multi_device_registry_test.py` | 离线 | 多设备 worker 注册表 |
| `exposure_control_test.py` | 离线 | 曝光控制解析/下发 |
| `test_meta_devices.py` | 离线 | 会话元数据设备行 |
| `test_depth_heatmap.py` | 离线 | 深度热力图生成 |
| `glove_widget_test.py` | 离线 | 手套控件配置路由 |
| `grid_drag_fps_test.py` | 离线 | 网格拖拽/分割条 |
| `device_panel_gui_smoke_test.py` | 离线 | 设备面板 GUI 冒烟 |
| `test_device_detector.py` | 离线 | 设备枚举 |
| `d405_worker_test.py` | 真机 | D405 采集 worker |
| `d435_e2e_test.py` | 真机 | D435 端到端 |
| `d435_gui_smoke_test.py` | 真机 | D435 GUI 冒烟 |
| `d435_playback_test.py` | 真机 | D435 回放 |
| `mono_regression.py` | 真机 | 单目回归 |
