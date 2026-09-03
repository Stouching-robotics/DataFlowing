# tools/hand_detection/

## 定位

手部 2D 关键点检测共享库，向仓库内多套上层管线提供统一的手部检测能力。核心是两条路径：裸手路径 `MediaPipeHandPipeline`（MediaPipe HandLandmarker Tasks API，`hand_pipeline_mediapipe.py`），黑手套路径 `HandPipeline`（YOLO-World 检测 + RTMPose 关键点 + IoU 追踪，`hand_pipeline.py`）。目录无 `__init__.py`，以 Python 隐式命名空间包形式被导入；存在两种导入风格：

- 包限定导入 `from hand_detection.xxx import ...`（依赖仓库根在 `sys.path`），如 `tools/stereo_s80m/hand_3d/detector.py:28`、`tools/hand_3d_d435/live_demo.py:73`、`tools/hand_3d_d435/render_overlay.py:24`。
- 顶层导入 `import hand_common` / `from hand_pipeline import HandPipeline`（调用方先把本目录插入 `sys.path`），如 `core/hand_tracking.py:45-46,106,121,327`（主程序手部追踪模块）、`tools/stereo_s80m/render_stereo.py:57-62`（sys.path shim + `from hand_common import ...`，缺失回退内置简单画法）、`tools/hand_3d_d435/glove_detector.py:85-86`（sys.path shim 见 :66-74，顶层 `from glove_package.hand_tracker import HandTracker` / `from world_detector import ...`）。

被 import 的完整清单（package 限定）：`tools/stereo_s80m/hand_3d/detector.py:28`、`tools/stereo_s80m/hand_3d/mp_gpu.py:34`、`tools/stereo_s80m/hand_3d/probes/probe_jitter.py:34`、`tools/stereo_s80m/hand_3d/renderer_3d.py:27`、`tools/stereo_s80m/hand_3d/run_pipeline.py:61`、`tools/stereo_s80m/hand_3d/smoother.py:26`、`tools/stereo_s80m/hand_benchmark.py:44`、`tools/stereo_s80m/hand_triangulate.py:67`、`tools/stereo_s80m/render_stereo.py:346`、`tools/hand_3d_d435/glove_detector.py:84`、`tools/hand_3d_d435/live_demo.py:73`、`tools/hand_3d_d435/render_overlay.py:24`。注意：`tools/glove_package/` 内含 `hand_common.py`、`hand_tracker.py`、`world_detector.py` 的同源副本（该包自包含）；`tools/hand_3d_d435/glove_detector.py` 的 `hand_tracker` 为显式命名空间导入 `from glove_package.hand_tracker import HandTracker`（`hand_detection/` 下同源旧副本 sys.path 排前，裸导入会解析错、双阈值等新参数失效），`world_detector` 为顶层导入 `from world_detector import ...`（按 sys.path 顺序解析到本目录副本，见 :80-84 注释）。

MediaPipe 手部模型为仓库根 `tools/models/hand_landmarker.task`（HandLandmarker Tasks 模型）；`tools/hand_detection/` 内也有一份内容相同的副本（`tools/hand_detection/hand_landmarker.task`，md5 一致），`config/settings.py:423` 的 `HAND_MEDIAPIPE_MODEL` 指向目录内副本。

## 文件清单

| 文件 | 一句话作用 |
| --- | --- |
| `tools/hand_detection/hand_pipeline_mediapipe.py` | MediaPipe 裸手 21 关键点管线 + One-Euro 滤波（`OneEuroFilter`/`OneEuroFilter2D`/`OneEuroFilter3D`）。 |
| `tools/hand_detection/hand_pipeline.py` | 黑手套模式统一入口：YOLO-World 检测 + RTMPose 关键点 + 追踪 + 冻结/遮挡。 |
| `tools/hand_detection/hand_tracker.py` | 基于 IoU 的跨帧手部身份追踪（`HandTrack` / `HandTracker`）。 |
| `tools/hand_detection/world_detector.py` | YOLO-World 开放词汇检测器（提示词 `hand`/`glove`）+ 两级 NMS 后处理。 |
| `tools/hand_detection/hand_common.py` | RTMPose 方案共用的 21 点定义、关节角度与绘制工具（不依赖 mediapipe）。 |
| `tools/hand_detection/demo_stereo_hands.py` | 双目视频手部关键点演示（左右目各自独立管线，输出 `keypoints_output/`）。 |
| `tools/hand_detection/test_smoothing.py` | One-Euro 平滑效果对比脚本（左原始 / 右平滑，并排输出视频）。 |

附：目录内的非 `.py` 文件——`yolov8m-worldv2.pt` 是指向 `../glove_package/yolov8m-worldv2.pt` 的符号链接（YOLO-World 权重）；`best.pt` 是自训 YOLO 回退权重（`config/settings.py:422` 的 `HAND_DET_MODEL` 指向它）；`hand_landmarker.task` 为 MediaPipe 手部模型副本；`hand_pipeline_mediapipe.py.bak` 为旧版备份（非 `.py`，不参与导入）。

## 各文件详解

### tools/hand_detection/hand_pipeline_mediapipe.py

**作用**：基于 MediaPipe HandLandmarker（Tasks API）的裸手关键点检测管线，是外部程序调用的统一入口。输出每只手 21 个关键点的像素坐标、归一化坐标、3D 世界坐标、左右手判定（`Left`/`Right`）、关节角度与伸直手指列表；内置 One-Euro 自适应低通滤波对像素坐标与世界坐标做抖动平滑。默认模型路径 `tools/models/hand_landmarker.task`（仓库根），模型缺失时抛 `FileNotFoundError` 并附下载 URL。管线运行在 `RunningMode.VIDEO` 模式下，内部持有跨帧追踪先验——左右目须各自独立实例，共享实例交替喂帧会污染追踪状态（`demo_stereo_hands.py:144-145` 注释：双手检出率 99.7% → ~50%）。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `LANDMARK_NAMES` | 模块常量 | 21 点名称（`WRIST`、`THUMB_CMC`…`PINKY_TIP`） | — |
| `FINGERS` | 模块常量 | 五指 → (关键点下标 1-20, BGR 颜色)，如 `Thumb: [1,2,3,4]` | — |
| `JOINT_SPECS` | 由 `_joint_specs()` 生成 | 15 个关节三元组 `(手指, 关节名, 顶点, 前点, 后点, 颜色)` | — |
| `OneEuroFilter` | `__init__(freq_min=1.0, beta=0.007, dcutoff=1.0)` | 单值 One-Euro 自适应低通滤波 | `__call__(x, ts_ms) -> float` |
| `OneEuroFilter2D` | 同上参数 | 对 `(x, y)` 分量独立滤波 | `__call__(x, y, ts_ms) -> (float, float)` |
| `OneEuroFilter3D` | 同上参数 | 对 `(x, y, z)` 分量独立滤波 | `__call__(x, y, z, ts_ms) -> (float, float, float)` |
| `_Preprocessor` | `__init__(gamma=0.4, clahe_clip=3.0, clahe_grid=8)` | 可选预处理（灰化/伽马/CLAHE），帮助深色手套场景 | `apply(bgr, mode)` 返回 RGB |
| `HandResult` | `__init__(index: int)` | 单只手结果容器 | 字段见"关键数据" |
| `FrameResult` | `__init__()` | 一帧完整结果容器 | `hands` / `raw_landmarks` / `raw_world` / `raw_handedness` |
| `MediaPipeHandPipeline` | `__init__(model_path="tools/models/hand_landmarker.task", num_hands=2, det_conf=0.5, track_conf=0.5, preprocess_mode="none", mirror=True, smooth=True, freq_min=5.0, beta=0.05, dcutoff=1.0)` | 构建 HandLandmarker 与滤波器状态 | 无 |
| `process` | `process(self, frame: np.ndarray) -> FrameResult` | 镜像 → 预处理 → 推理 → 组装/平滑 | 返回 `FrameResult` |
| `reset` | `reset(self) -> None` | 关闭 landmarker、清空滤波器状态（切换视频源时调用） | 重建 `_landmarker` |
| `close` | `close(self) -> None` | 释放 MediaPipe 资源 | 关闭 `_landmarker` |

**关键数据**：

- 关键点维度：每只手 21 点；`landmarks` 为 `(21, 2)` float32 像素坐标，`norm_landmarks` 为 21 个归一化 `(x, y)`（0-1），`world_landmarks` 为 21 个 3D 世界坐标 `(x, y, z)`。
- `HandResult` 字段：`index`（结果序号）、`label`（`Left`/`Right`）、`score`、`landmarks`、`norm_landmarks`、`world_landmarks`、`angles`（`{(手指, 关节): 度数}`，基于 3D 世界坐标计算）、`extended`（伸直手指名列表）。
- 伸直判定阈值：拇指 `MCP > 145` 且 `IP > 150`；其余四指 `PIP > 150` 且 `DIP > 140`。
- 滤波默认参数：管线层 `freq_min=5.0`、`beta=0.05`、`dcutoff=1.0`（归一化空间速度量纲下 `beta` 需较大才有明显效果）；`OneEuroFilter` 类本身默认 `freq_min=1.0`、`beta=0.007`、`dcutoff=1.0`。滤波状态按 `(手序号, 关键点序号)` 键惰性创建。
- 镜像：`mirror=True` 时先 `cv2.flip(frame, 1)`，且左右手 label 在镜像后翻转。
- 固定项：`min_hand_presence_confidence=0.5`；时间戳 `ts_ms` 以构造时刻为 0 的 `perf_counter` 毫秒。

**调用关系**：被上文"定位"列出的 14 处 import；主要使用方为 `tools/stereo_s80m/hand_3d/`（双目 3D 三角化链）、`tools/hand_3d_d435/`（D435 单目 3D 链）、`core/hand_tracking.py:327`（主程序裸手模式，顶层导入）、`tools/hand_detection/demo_stereo_hands.py:25`、`tools/hand_detection/test_smoothing.py:21`。依赖 `mediapipe`、`opencv-python`、`numpy`。

### tools/hand_detection/hand_pipeline.py

**作用**：黑手套模式的手部检测 + 关键点推理统一入口 `HandPipeline`。流程：检测（`detector="world"` 走 `world_detector.WorldDetector`，否则用 `_YOLOWrapper` 加载自训 `.pt`）→ `HandTracker` 跨帧追踪 → 仅对需要推理的框跑 RTMPose 21 点 → `pose_is_glove` 无手套误检抑制 → 逐点置信度冻结 + 遮挡判断。默认检测器权重为 `MODEL_PATH = tools/hand_detection/yolov8m-worldv2.pt`（符号链接）。`det_imgsz` 默认 1280——注释说明 `best.pt` 对 640 缩放过敏感（同框 conf 掉到 0.4-0.65、位置漂移幻觉），1280 原生推理才能还原训练尺度下的检测质量。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `HandPipeline` | `__init__(detector="world", det_device="cpu", pose_device="cpu", max_hands=2, conf=0.05, det_imgsz=1280)` | 组装检测器（`WorldDetector` 或 `_YOLOWrapper`）、RTMPose 模型与 `HandTracker` | 无 |
| `process` | `process(frame: np.ndarray, apply_freeze: bool = True)` | 检测 → 追踪 → 关键点 → 误检抑制 → 冻结/遮挡 | 返回 `(boxes, kpts, scores, track_ids)`，`kpts` 为 `(N,21,2)` 或 `None` |
| `reset` | `reset(self) -> None` | 清空追踪状态（切换视频源时调用） | `_tracker.clear()` |
| `detector_name` | 属性 | 当前检测器名称（`WorldDetector` / `YOLO (best.pt)`） | 返回 str |
| `_apply_freeze` | `_apply_freeze(kpts, scores)` | 遮挡判定（低置信点占比 ≥ 阈值则整手冻结）+ 按手指粒度逐点冻结 | 原地修改 `kpts` |
| `_YOLOWrapper` | `__init__(weights, device, imgsz=1280)` | ultralytics YOLO 包装（回退权重路径） | `__call__(frame, conf)` 返回 `[[x1,y1,x2,y2,conf], ...]` |

**关键数据**：

- 检测参数：`conf=0.05`（world 后端）、`det_imgsz=1280`（YOLO 后端）、`max_hands=2`。
- 追踪参数（传给 `HandTracker`）：`iou_match_thr=0.3`、`lost_timeout=3`、`movement_thresh=3`、`skip_timeout=10`、`box_smooth_alpha=0.7`。
- 冻结/遮挡：`kpt_freeze_thr=0.2`（逐点低置信阈值）、`occlusion_ratio=0.9`（低置信点占比 ≥ 0.9 判整手遮挡）。
- 无手套误检抑制：`hc.pose_is_glove`（阈值见 `hand_common.py`），不通过则整帧清空输出。

**调用关系**：被 `core/hand_tracking.py:106` 以顶层 `from hand_pipeline import HandPipeline` 延迟导入（该文件已把 `tools/hand_detection/` 插入 `sys.path`；依赖 ultralytics/torch/rtmlib，缺失时抛带提示的 `ImportError`）；`tools/demos/demo_glove_kpts/demo_glove_video.py` 与 `compare_detectors.py` 使用本管线。内部：`import hand_common as hc`、`import world_detector as wd`、`from hand_tracker import HandTracker`（文件开头把自身目录插入 `sys.path`）。

### tools/hand_detection/hand_tracker.py

**作用**：手部身份追踪模块。用贪心 IoU 匹配做跨帧框关联，给每只手分配稳定 ID；每只手独立维护 `kpts`/`scores`/`last_good_kpts`/`skip_counter`/`lost_counter`，EMA 平滑检测框，短期记忆（连续丢失 ≤ `lost_timeout` 帧的 track 保持活跃但不跑推理）。设计动机（文件 docstring）：框按置信度排序会导致帧间交换、按数组下标操作冻结缓存会错乱、检测丢失一帧就清空状态会导致 ID 跳变——本模块按 track 对象而非数组下标管理状态，避免上述问题。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `HandTrack` | dataclass | 一只手跨帧状态 | 字段：`id`/`box`/`raw_box`/`prev_center`/`kpts`/`scores`/`last_good_kpts`/`skip_counter`/`lost_counter`/`active` |
| `HandTracker` | `__init__(max_hands=2, iou_match_thr=0.3, lost_timeout=3, movement_thresh=3.0, skip_timeout=10, box_smooth_alpha=0.7)` | 初始化 track 列表与 ID 计数器 | 无 |
| `update_detections` | `update_detections(boxes: List[List[float]]) -> None` | 每帧必须调用：贪心 IoU 匹配 → 创建/更新 track → 老化丢失 track → EMA 平滑 → 限制活跃数 ≤ `max_hands` | 修改内部 tracks |
| `get_boxes_for_pose` | `get_boxes_for_pose()` | 挑出需要跑关键点推理的框（新 track / 移动超阈值 / skip 超时；丢失中的不跑） | 返回 `(boxes, indices)` |
| `update_pose_results` | `update_pose_results(indices, new_kpts, new_scores)` | 把 RTMPose 结果写入对应 track，记录框中心、清零 `skip_counter` | 无返回 |
| `get_results` | `get_results()` | 取活跃 track 的框/关键点/置信度/ID（无关键点的补零矩阵） | 返回 `(boxes, kpts, scores, track_ids)` |
| `update_last_good` | `update_last_good(track_index, point_index, value)` | 更新某 track 某关键点的冻结缓存 | 惰性创建 `last_good_kpts` |
| `get_last_good` | `get_last_good(track_index, point_index)` | 读冻结缓存 | 返回 `np.ndarray` 或 `None` |
| `clear` | `clear()` | 重置所有 track | 清空 `tracks`、`_next_id=0` |
| `hand_count` | 属性 | 当前活跃手数 | 返回 int |
| `_needs_pose` | `_needs_pose(track, movement_thresh, skip_timeout)` | 判断是否需要推理（无数据/超时/位移超阈值） | 返回 bool |
| `_smooth_box` | `_smooth_box(old, new, alpha)` | EMA 平滑框（alpha=0 不平滑，1 完全不动） | 返回列表 |
| `_greedy_match` | `_greedy_match(prev_boxes, curr_boxes, iou_thr)` | 按 IoU 降序贪心配对（`max_hands≤2` 时等价匈牙利算法，不依赖 scipy） | 返回 `(matches, unmatched_curr, unmatched_prev)` |

**关键数据**：`kpts`/`scores` 维度 `(21, 2)` / `(21,)`；`last_good_kpts` 为逐点冻结缓存 `(21, 2)`。

**调用关系**：`from world_detector import iou`（顶层导入，依赖本目录在 `sys.path`）。被 `tools/hand_detection/hand_pipeline.py:35` 使用；`tools/glove_package/` 有同源副本（`tools/glove_package/hand_tracker.py`），`tools/hand_3d_d435/glove_detector.py:85` 为显式命名空间导入 `from glove_package.hand_tracker import HandTracker`（hand_detection 旧副本 sys.path 排前，裸导入会解析错）。

### tools/hand_detection/world_detector.py

**作用**：YOLO-World 开放词汇手部检测器，提示词驱动免训练，是 `tools/glove_package/auto_label.py` 与 `infer.py` 的唯一真值来源（原先两处各写一份配置导致标注框与推理框不一致，统一收敛到本文件）。默认 `yolov8m-worldv2.pt` @ `imgsz=320`——docstring 记录了 40 张真实手套数据的实测：`yolov8m-worldv2@320` 召回 40/40、平均 IoU 0.780、22.6ms，优于 640 与 `yolov8x-worldv2`。提示词 `["hand", "glove"]` 会让同一只手被检出两次，因此后处理带两级 NMS（区分"同手重复检出"与"两手靠近"）。`__call__` 接受 BGR numpy 帧，统一返回 `(boxes_xyxy [N,4], confs [N,])`。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| 常量 | `DEFAULT_MODEL="yolov8m-worldv2.pt"` / `DEFAULT_PROMPT=["hand","glove"]` / `DEFAULT_IMGSZ=320` / `DEFAULT_CONF=0.05` / `DEFAULT_NMS_IOU=0.6` | 检测器默认参数（标注与推理共用） | — |
| `WorldDetector` | `__init__(model=DEFAULT_MODEL, prompt=None, imgsz=DEFAULT_IMGSZ, device="cpu", nms_iou=DEFAULT_NMS_IOU, use_onnx=False)` | 按权重后缀选后端：`.pt` 走 `YOLOWorld`（GPU ~12ms/CPU ~85ms），`.onnx` 走 ultralytics YOLO ONNX（强制 CPU） | 无 |
| `__call__` | `__call__(src, conf=DEFAULT_CONF, max_boxes=1, wh=None, reuse_boxes=False)` | 推理 + 后处理；`reuse_boxes=True` 时复用上一帧结果（帧跳过） | 返回 `(boxes [N,4], confs [N,])` |
| `postprocess` | `postprocess(boxes, confs, w, h, max_boxes, nms_iou=DEFAULT_NMS_IOU)` 静态方法 | 裁入画面、丢弃 <8px 非法框、按置信度排序（不按面积）、两级 NMS、Top-N | 返回 `(boxes, confs)` |
| `postprocess_legacy` | `postprocess_legacy(...)` 静态方法 | 旧版纯 IoU 阈值 NMS，保留给兼容性测试 | 返回 `(boxes, confs)` |
| `iou` | `iou(a, b)` | 两框 IoU | 返回 float |
| `_center_close` | `_center_close(a, b, ratio=0.15)` | 中心距是否 < 较小框宽度的 15% | 返回 bool |
| `add_args` | `add_args(ap, max_boxes_default=1)` | 向 argparse 挂检测器公共参数（`--weights/--prompt/--imgsz/--conf/--max-boxes/--det-skip`） | 修改 `ap` |
| `from_args` | `from_args(args, device)` | 由解析结果构建 `WorldDetector` | 返回实例 |

**关键数据**：

- 两级 NMS 判据：中心距 < 框宽 15% 且 `IoU >= nms_iou`(0.6) → 视为同手重复，抑制；中心有偏移 → 视为两手靠近，仅 `IoU > 0.85` 才抑制。
- 后处理过滤：框宽/高 ≥ 8px、坐标非有限值剔除。
- 帧跳过缓存：`_last_boxes`，供 `reuse_boxes=True` 复用。

**调用关系**：被 `tools/hand_detection/hand_pipeline.py:34` 顶层导入；`tools/glove_package/auto_label.py:38`、`infer.py:29`、`hand_demo_mmpose.py:39` 顶层导入（`glove_package` 内自带副本 `world_detector.py`）；`tools/hand_3d_d435/glove_detector.py:86` 顶层导入（`from world_detector import WorldDetector, iou`，按 sys.path 顺序解析到本目录副本）。依赖 `ultralytics`（`YOLOWorld`）、`numpy`。

### tools/hand_detection/hand_common.py

**作用**：RTMPose 方案共用的手部关键点定义与绘制工具。关键点约定与 MediaPipe 完全一致（21 点同序：0 腕、1-4 拇指、5-8 食指、9-12 中指、13-16 无名指、17-20 小指），刻意不 import 父目录、不依赖 mediapipe（仅 cv2/numpy），使 `rtmpose`/glove 相关脚本可自包含运行。提供 2D 关节角度计算（图像平面，注意手指正对相机透视缩短时角度偏小，区别于 MediaPipe 管线的 3D 世界坐标角度）、五指分色绘制、RTMPose hand5 模型构建与"真手套"误检判定。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `FINGERS` | 模块常量 | 五指 → (下标, BGR 颜色)，与 MediaPipe 同序 | — |
| `PALM_CONNECTIONS` | 模块常量 | 掌心连接 `(0,1),(0,5),(5,9),(9,13),(13,17),(0,17)` | — |
| `JOINT_SPECS` | 由 `joint_specs()` 生成 | 15 个关节三元组 | — |
| `angle_between` | `angle_between(p_prev, p_vertex, p_next)` | 顶点处两条骨段夹角（度，180° = 伸直） | 返回 float |
| `compute_joint_angles` | `compute_joint_angles(pts)` | 由 21 点 2D 坐标算全部关节角度 | 返回 `{(手指, 关节): 度}` |
| `count_extended_fingers` | `count_extended_fingers(angles)` | 伸直手指判定（阈值同 MediaPipe 管线：拇指 MCP>145/IP>150，其余 PIP>150/DIP>140） | 返回手指名列表 |
| `draw_hand` | `draw_hand(frame, pts, angles=None, show_angles=True, kpt_scores=None, thr=0.3)` | 绘制 21 点骨架（掌心灰线 + 五指分色 + 腕部白圆 + 角度标注）；`kpt_scores` 低于 `thr` 的点画空心 | 原地修改 `frame` |
| `draw_panel` | `draw_panel(frame, x, y, lines, width=340)` | 半透明信息面板 | 原地修改 `frame` |
| `build_pose` | `build_pose(device="cpu")` | 构建 RTMPose hand5（21 点与 MediaPipe 同序，首次运行自动下载 ~56MB） | 返回 `RTMPose` 实例 |
| `auto_device` | `auto_device()` | 自动选择推理设备：CUDA > MPS > CPU | 返回 str |
| `pose_is_glove` | `pose_is_glove(kpts, scores, box, mean_thr=0.45, n_ok_thr=15, ok_thr=0.3, span_thr=0.3)` | 误检抑制：均值/高置信点数/点团 span 三条全过才判"真手套" | 返回 bool |

**关键数据**：

- `pose_is_glove` 三条判据（docstring 实测 2026-08-20）：误检框关键点均值 0.20-0.31、≥0.3 高置信点 ≤10、点团 span ≤0.23×框对角线；真手套 0.64-0.81、21/21 高置信、span 0.55-0.79。
- RTMPose 模型：`rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip`（OpenMMLab onnx SDK，输入 256×256，onnxruntime 后端）。

**调用关系**：被 `tools/hand_detection/hand_pipeline.py:33`（`import hand_common as hc`）、`tools/hand_detection/demo_stereo_hands.py:26`、`tools/hand_detection/test_smoothing.py:26`（包限定导入 `draw_hand`）、`core/hand_tracking.py:121`（顶层导入，主程序绘制复用）、`tools/stereo_s80m/render_stereo.py:59`（顶层导入，失败时回退本文件内置简单画法）。`tools/glove_package/hand_common.py` 为同源副本（`infer.py`、`verify_pose.py` 等顶层导入的是该副本）。

### tools/hand_detection/demo_stereo_hands.py

**作用**：双目手部关键点演示脚本。同时打开左右目视频，分别用两个独立的 `MediaPipeHandPipeline` 实例逐帧检测（`mirror=False`，头戴式双目相机无需镜像），`draw_hand` 绘制关键点与角度，左右并排合成输出 mp4，最后用 ffmpeg 转 H.264（libx264 crf 23）便于播放。左右目各自独立实例的原因见注释：共享实例交替喂帧会污染 VIDEO 模式追踪先验，双手检出率从 99.7% 塌陷到 ~50%。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `draw_header` | `draw_header(frame, text, color=(255,255,255))` | 顶部半透明标题栏（LEFT/RIGHT CAMERA） | 原地修改 `frame` |
| `draw_info_overlay` | `draw_info_overlay(frame, hand_results, fps_val)` | 底部叠加 FPS、每只手 label/score/伸直手指 | 原地修改 `frame` |
| `process_and_draw` | `process_and_draw(pipe, frame)` | 单目检测 + 绘制 | 返回 `(标注帧, hands)` |
| `main` | `main()` | 打开视频 → 双管线循环 → 写临时 mp4 → ffmpeg 转码 | 输出 `keypoints_output/stereo_hands.mp4` |

**关键数据**：

- 默认输入：`data/recordings/Test1/Test1_000020/videos/stereo_left/chunk-0000/stereo_left.mp4` 与同路径 `stereo_right.mp4`（示例会话不在仓库内，需自备或用 `--left`/`--right` 覆盖）。
- 输出：`keypoints_output/stereo_hands.mp4`（临时文件 `stereo_hands_tmp.mp4`，转码成功即删除）；`VIEW_WIDTH = 640`（每目宽度）。
- 管线参数：`num_hands=2`、`det_conf=0.5`、`track_conf=0.5`、`preprocess_mode="none"`、`mirror=False`、`smooth=True`、`freq_min=5.0`、`beta=0.05`；模型 `tools/models/hand_landmarker.task`（仓库根）。
- ffmpeg 候选：`FFMPEG_CANDIDATES` 依次尝试 `shutil.which("ffmpeg")`、`/usr/bin/ffmpeg`、环境变量 `FFMPEG_BIN` 指定的路径、`~/miniconda3/envs/lerobot/bin/ffmpeg`（注释说明 conda base 的 ffmpeg 因 openvino/tbb 符号错误不可用）；全部失败则保留原始 mp4v 临时文件。

**调用关系**：`from hand_detection.hand_pipeline_mediapipe import MediaPipeHandPipeline`（:25）、`from hand_detection.hand_common import draw_hand`（:26）。独立运行入口（`python tools/hand_detection/demo_stereo_hands.py`）。

### tools/hand_detection/test_smoothing.py

**作用**：One-Euro 平滑效果对比测试脚本。对同一段视频用单个 `MediaPipeHandPipeline`（`smooth=True`）处理，输出左右并排视频：左侧画原始关键点（取自 `FrameResult.raw_landmarks`，未滤波），右侧画平滑后的关键点 + 关节角度，直观对比滤波去抖效果。末尾用 ffmpeg 转 H.264。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
| --- | --- | --- | --- |
| `draw_landmarks_simple` | `draw_landmarks_simple(frame, pts, color=(0,255,0), label="")` | 简化画法：21 点 + 五指连线（无角度） | 原地修改 `frame` |
| `draw_info_bar` | `draw_info_bar(frame, frame_idx, total, fps, hand_count)` | 顶部信息栏（帧号/FPS/手数/左右说明） | 原地修改 `frame` |
| `raw_pixel_coords` | `raw_pixel_coords(lms, w, h)` | MediaPipe 原始归一化 landmark 转像素坐标（无滤波） | 返回 `(21, 2)` float32 |
| `main` | `main()` | 读视频 → 管线循环 → 并排合成 → 转码 | 输出 `keypoints_output/smoothing_comparison.mp4` |

**关键数据**：

- 默认输入：`data/recordings/Project_812/Project_812_000005/videos/head_left_rgb/chunk-0000/head_left_rgb.mp4`（示例会话不在仓库内，需自备或改 `main()` 中的路径）。
- 输出：`keypoints_output/smoothing_comparison.mp4`（临时 `smoothing_comparison_tmp.mp4`）；输出宽度 = 输入 2 倍（左右并排）。
- 管线参数与 `demo_stereo_hands.py` 一致：`num_hands=2`、`det_conf=0.5`、`track_conf=0.5`、`mirror=False`、`smooth=True`、`freq_min=5.0`、`beta=0.05`、`dcutoff=1.0`；模型 `tools/models/hand_landmarker.task`。
- ffmpeg：直接调用 `"ffmpeg"` 命令（libx264 crf 23），失败时保留 mp4v 临时文件并打印 stderr 尾部（本脚本未使用 `demo_stereo_hands.py` 的 `FFMPEG_BIN` 候选逻辑）。

**调用关系**：`from hand_detection.hand_pipeline_mediapipe import (MediaPipeHandPipeline, FINGERS, JOINT_SPECS)`（:21-25）、`from hand_detection.hand_common import draw_hand`（:26）。独立运行入口。
