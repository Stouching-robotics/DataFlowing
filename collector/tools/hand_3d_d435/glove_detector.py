"""黑手套检测器：YOLO-World 手套框 + 关键点后端（RTMPose/MediaPipe，
可热切换）+ 帧间稳定。

检测契约对齐 stereo_s80m.hand_3d.detector.DetectedHand：detect(frame_bgr)
-> list[DetectedHand]（landmarks (21,2) 像素、MediaPipe 拓扑 0=腕、
label "Left"/"Right"、score=检测框 conf）。下游（voter/抬升/槽位/平滑/
渲染）零改动——唯一注意 HandednessVoter 只数 score>=0.7 的票（identity.py
MIN_VOTE_SCORE=0.7），框 conf 不足 0.7 时该手 label 不进票仓。已核实
identity.py:139/156-158：voter 对此退化为 latch（票仓只累积已表决
label，轨迹存续期锁死首票）——本模块按 track_id 自建票仓与 voter 互补。

2026-08-20 重构（用户指令：切换 world 自带检测器 + 抑制抖动）：

检测框后端 = glove_package.world_detector.WorldDetector（YOLO-World
yolov8m-worldv2.pt + 提示词 ["hand","glove"]，imgsz 320、conf 0.05、
两级 NMS、按 conf Top-N——包内 40 张实测 40/40 召回）。权重名含
"world" 走 world 后端，否则普通 YOLO（best.pt 回退开关，conf 0.3）。
world_detector 只读复用，不改包内任何文件。

帧间稳定层（对齐裸手 MediaPipeDetector 内部机制，见方案对照表）：
1. HandTracker（hand_detection.hand_tracker 只读复用）：贪心 IoU 匹配
   + 框 EMA α=0.7（仅匹配/显示）+ 运动门控 3px（静止帧免 RTMPose 推理、
   0.33s 强制刷新）+ 丢框持 3 帧 + track_id 帧间身份。注意 _needs_pose
   要求 last_good_kpts 非 None 门控才生效（hand_tracker.py:293）——每
   track 首次合格 pose 后必须 update_last_good 初始化。
2. 逐点 OneEuroFilter2D（freq_min=5.0/beta=0.05/dcutoff=1.0，归一化
   坐标域，键 (track_id, 点序号)，单调钟）——裸手同款同参数；退化
   冻结帧不重复喂滤波器（防把点吸向陈旧值），低置信 hold 帧把**纯**
   旧点回喂保持滤波时间连续，运动补偿在滤波器**外**施加（喂补偿点
   会被滤波器按 alpha 打折，骨架只能跟到框速度的一半）。
3. 快动跟随（2026-08-24）：hold/退化冻结输出按平滑框中心位移平移补偿
   （hold_translate）——硬冻结在快动时骨架钉死原地、置信恢复瞬间跳回
   （不跟手+闪烁主因），平移后骨架随框滑动。补偿施加在滤波器外
   （_last_out 恒为纯滤波值，不重复累加——旧实现回写补偿点再加全量
   位移，逐帧累加致骨架飞出框外，2026-08-24 实机"更糟糕"根因）；恢复
   帧滤波从纯旧点收敛，有一次小幅回跳后跟上（远小于硬冻结的恢复
   snap）。裁剪框默认仍用平滑框（链稳定）；可选 raw 框（pose_box_raw）
   ——消除框 EMA 稳态滞后（~2.3 帧）但框抖动直通 RTMPose，实测
   Test_Data_000003 下游 wholesale 19 vs 6、renderer_in 34.4 vs
   11.8mm，仅极端快动时可尝试。
4. 退化族过滤"丢弃→冻结"：三条件参数不动（旧链 3D 毒化定案——框外
   点≥16 / 唯一点<15 / span<0.2×框对角线，对送入 RTMPose 的框判）。
   命中退化的新 pose 不写 tracker → 输出上次滤波值；连续无合格 pose 超
   freeze_max（默认 15，对齐 --propagate-max）帧不输出走传播。运动
   门控跳过推理的帧不算退化（免误伤健康静止手）。
5. 手性 per-track 锁存（连续 3 票相同 → 锁死不再变，镜像 voter latch；
   双手框重叠冻结，镜像 voter OVERLAP_PX）。实测否决 7 帧滑窗多数表决：
   手旋转时 cross_z 符号缓慢翻转，翻转过渡期两手滑窗多数会短暂同值 →
   下游同 label 守卫触发 358/359 帧（旧裸直通+voter latch 只 5 帧）。
   锁存后 label 永不变，慢翻转被完全吸收。手性本身几何合成：
   cross_z = axis.x*d.y - axis.y*d.x（axis=P9-P0，d=P5-P0），>0 →
   Left（掌心朝相机约定；手背朝相机会反转——下游 voter latch + 几何
   兜底吸收）。
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
_GLOVE_PKG = os.path.join(_TOOLS_DIR, "glove_package")
_HAND_DET = os.path.join(_REPO_ROOT, "tools", "hand_detection")
for _d in (_TOOLS_DIR, _GLOVE_PKG, _HAND_DET):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from stereo_s80m.hand_3d.detector import DetectedHand      # noqa: E402

# hand_tracker/world_detector 内部 `from world_detector import iou` 需要
# glove_package 在 sys.path（上面 shim 已做）；hand_pipeline_mediapipe 按
# live_demo.py:58 同款命名空间包路径导入。均只读复用。注意 hand_tracker
# 必须显式用 glove_package 命名空间：hand_detection/ 下有一份同源旧副本，
# sys.path 上 hand_detection 排在前面，裸 `from hand_tracker import` 会
# 解析到旧副本（双阈值等新参数即失效）。
from hand_detection.hand_pipeline_mediapipe import OneEuroFilter2D     # noqa: E402
from glove_package.hand_tracker import HandTracker                     # noqa: E402
from world_detector import WorldDetector, iou                          # noqa: E402
# 姿态后端（统一契约见 pose_backends.py docstring）：hand_3d_d435 命名
# 空间包路径（live_demo 同款），显式包路径防与旧副本混淆（同 HandTracker
# 教训——hand_detection 下旧副本在 sys.path 前面）。
from hand_3d_d435.pose_backends import (                               # noqa: E402
    MediaPipePoseBackend, RtmposePoseBackend)

_MP_TASK = os.path.join(_REPO_ROOT, "tools", "models", "hand_landmarker.task")

_WORLD_WEIGHTS = os.path.join(_GLOVE_PKG, "yolov8m-worldv2.pt")
_DET_WEIGHTS = os.path.join(_GLOVE_PKG, "runs", "hand_det", "weights", "best.pt")
_WORLD_PROMPT = ["hand", "glove"]
_WORLD_CONF = 0.05
_YOLO_CONF = 0.3
_OVERLAP_FRAC = 0.05       # 双手框中心距 < 0.05*max(w,h) → 手性冻结（镜像 voter）
_REVIVE_MAX = 90           # 复活窗口（帧，3s@30fps）：真实重捕捉常 >15 帧
                            # （手出画再入画——闪烁只在 live 出现即此故）；
                            # 几何门兜底防误继承
_REVIVE_DIST = 0.3         # 复活匹配：框中心距 < 0.3*max(w,h)（帧相对）


def resolve_glove_weights(choice="world", explicit=None):
    """检测器选择 → 权重路径。explicit（--glove-weights）优先于 choice；
    choice="world" → yolov8m-worldv2.pt（开放词汇 world 后端），
    choice="det" → glove_package 训练产物 best.pt（yolo11n 单类 hand，
    文件名不含 world → 自动走 ultralytics YOLO 后端）。"""
    if explicit:
        return explicit
    if choice == "det":
        return _DET_WEIGHTS
    return _WORLD_WEIGHTS


class GloveDetector:
    """YOLO-World/YOLO 出框 + HandTracker 追踪 + RTMPose/MediaPipe 关键点。

    惰性加载：ultralytics/torch/rtmlib/mediapipe 只在首次构造时 import
    （裸手启动零开销；首次切换有一次 ~1s 加载 + CUDA 预热）。
    """

    def __init__(self, weights=None, det_conf=None, num_hands=2,
                 device="auto", pose_device="auto",
                 pose_backend="rtmpose", pose_model=None,
                 prompt=None, imgsz=320, nms_iou=0.6,
                 use_tracker=True, movement_thresh=3.0, skip_timeout=10,
                 box_alpha=0.7, lost_timeout=3, iou_thr=0.3,
                 use_oe=True, oe_freq_min=5.0, oe_beta=0.05, oe_dcutoff=1.0,
                 use_vote=True, vote_window=3, freeze_max=15,
                 pose_conf_thr=0.3, hold_max=12, pose_box_raw=False,
                 hold_translate=True, new_track_conf=0.25,
                 spawn_confirm=2, match_contain_thr=0.7):
        """pose_conf_thr：RTMPose 逐点置信均值门（None 关闭）。低于门的骨架
        大概率是背面/侧视的"看似合理但贴错"解，只持出上次输出（不喂
        OneEuro、不更新锁存、不计退化），置信恢复后自动续上。

        hold_max：连续低置信 hold 帧数上限（None=无限 hold，旧行为）。
        持续低置信 = 大概率是真实新姿势（握拳/抓取等——黑手套无手指
        纹理，RTMPose 逐点置信天然偏低，实测离线数据 25% 帧 <0.3），
        无限 hold 会把骨架永久冻在旧姿势（"姿势无法捕捉"根因）。连续
        满 hold_max 帧即放行本轮低置信点（按正常新 pose 走 OneEuro
        收敛），瞬时低置信（快动模糊，实测短段 2-7 帧）仍持旧点防抖。

        pose_box_raw：RTMPose 裁剪框来源。默认 False 用平滑框（链稳定，
        3D 下游 wholesale/渲染指标好）；True 用原始检测框——平滑框对匀速
        运动有约 α/(1−α)≈2.3 帧稳态滞后，快动时手掌部分出框致 conf 下降
        触发 hold；但 raw 框帧间抖动直通 RTMPose，会扰动下游 3D 链
        （实测 Test_Data_000003 wholesale 19 vs 6、renderer_in p95
        34.4 vs 11.8mm）——仅极端快动且下游指标可接受时开。身份匹配/
        显示恒用平滑框。

        hold_translate：hold/冻结帧把旧骨架按平滑框中心位移平移（运动
        补偿）。硬冻结在快动时骨架钉死原地、置信恢复瞬间跳回（闪烁主因
        之一）；平移后骨架随框滑动。补偿在逐点 OneEuro **之外**施加
        （滤波器只回喂纯旧点）：喂补偿点会被滤波器按 alpha 打折、骨架
        只能跟到框速度的一半；全量叠加会重复累加致骨架飞出框外。
        prev_center=上次推理时的框中心，逐帧累积差即 hold 期间框走过
        的位移。恢复帧滤波从纯旧点收敛，一次小幅回跳后跟上（远小于
        硬冻结的恢复 snap）。

        new_track_conf：新建 track 的最低框置信度（双阈值，0=关）。
        匹配已有 track 不受限；快动模糊/手出画边缘时 world 常给出
        0.1-0.3 的闪框且位置在两手间来回跳，若不设门每帧新建 track
        → 逐点滤波/手性锁存清零 → 骨架瞬移（000005 实测 f161-200
        连续 15 帧新建 t10-t24，槽 0 全程 47 次 >200px 巨跳）。只抬
        新手进入门槛（实测健康段手置信 0.36-0.79），不损已跟踪手。

        spawn_confirm：新 track 候选时间确认（帧，0=关）：不与活跃
        track 框重叠的框须连续该帧数位置稳定才送进 tracker——背景
        假框/单帧闪框不再新建 track（低置信手背闪框死亡后 track 被
        背景物假框飘走的根因）。已匹配已有 track 的框直通不受限
        （同 new_track_conf 双阈值语义）；远手框持续存在则 1 帧延迟
        后正常新建。

        match_contain_thr：跨手框拒收占比（None 关）——透传
        HandTracker：某 track 匹配到的框若 ≥thr 面积落在另一 track
        本帧匹配框内即拒收（宁可 lost 持旧框），防 track 在另一只手
        的碎片框上续命、骨架跳手（"框飘到另一只手上"）。

        pose_backend：关键点后端 "rtmpose"（默认）/ "mediapipe"。
        RTMPose=hand5 SIMCC（onnxruntime，256x256）；MediaPipe=
        HandLandmarker（Tasks API，pose_model .task，整图检测+
        质心关联到 tracker 框，21 点同拓扑）。运行中热切换用
        set_pose_backend()——只换 _pose，tracker/逐点 OneEuro/
        手性锁存等帧间状态全部保留。MediaPipe 掌部检测对黑手套
        检出率低（000005 整图 5/60 vs world 每帧出框），预期差于
        RTMPose，主要供裸手/效果对比；无关联手 → 全零骨架 →
        退化冻结兜底（不毒化 tracker）。"""
        if weights is None:
            weights = _WORLD_WEIGHTS
        if not os.path.isfile(weights):
            raise FileNotFoundError(weights)
        self.backend = "world" if "world" in os.path.basename(weights).lower() \
            else "yolo"
        if det_conf is None:
            det_conf = _WORLD_CONF if self.backend == "world" else _YOLO_CONF
        self.det_conf = det_conf
        self.num_hands = num_hands
        self.last_boxes = []   # 本轮 raw 框 (x1,y1,x2,y2,conf)，供 overlay 画框

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        if self.backend == "world":
            self._world = WorldDetector(weights, prompt=prompt, imgsz=imgsz,
                                        device=self.device, nms_iou=nms_iou)
            self._model = None
        else:
            from ultralytics import YOLO
            self._world = None
            self._model = YOLO(weights)

        if pose_device == "auto":
            pose_device = "cuda" if device == "cuda" else "cpu"
        self.pose_backend = pose_backend
        self._pose_model = pose_model
        self._pose = self._build_pose(pose_backend, pose_device)
        self.pose_device = self._pose.device

        # 帧间稳定层
        self._use_oe = use_oe
        self._oe_freq_min, self._oe_beta, self._oe_dcutoff = \
            oe_freq_min, oe_beta, oe_dcutoff
        self._use_vote = use_vote
        self._vote_window = vote_window
        self._freeze_max = freeze_max
        self._pose_conf_thr = pose_conf_thr
        self._hold_max = None if (hold_max is not None and hold_max < 0) \
            else hold_max
        self._pose_box_raw = pose_box_raw
        self._hold_translate = hold_translate
        self._iou_thr = iou_thr
        self._spawn_confirm = spawn_confirm
        self._spawn_cand = {}   # 新 track 候选：(量化位置键 -> (框, 帧数))
        self._tracker = HandTracker(max_hands=num_hands, iou_match_thr=iou_thr,
                                    lost_timeout=lost_timeout,
                                    movement_thresh=movement_thresh,
                                    skip_timeout=skip_timeout,
                                    box_smooth_alpha=box_alpha,
                                    new_track_conf=new_track_conf,
                                    match_contain_thr=match_contain_thr) \
            if use_tracker else None
        self._filters = {}       # (track_id, 点序号) -> OneEuroFilter2D
        self._votes = {}         # track_id -> deque(label, maxlen=7)
        self._last_label = {}    # track_id -> 上次稳定 label
        self._last_out = {}      # track_id -> (21,2) 上次滤波输出（冻结帧复用）
        self._posed = {}         # track_id -> True（至少出过一次合格 pose）
        self._stale = {}         # track_id -> 连续退化帧数（门控跳过不计）
        self._conf_mem = {}      # track_id -> 上次框 conf
        self._held = set()       # 本轮被低置信门持出的 tid（探针/诊断）
        self._hold_count = {}    # tid -> 连续低置信 hold 帧数（hold_max 逃逸）
        self._last_pose_conf = {}  # tid -> 本轮 pose 逐点均值置信（探针/诊断）
        self._last_raw_pose = []   # 本帧原始 pose（与 detect() 返回同序，
                                   # 无新原始点的 track 为 None；左目 2D 显示用）
        self._t0 = time.perf_counter()
        # 复活继承：HandTracker 丢 >3 帧即删 track，重检测拿新 id →
        # per-track 稳定层（OneEuro/手性锁存/冻结缓冲）全清空 = 重捕捉
        # 闪烁主因。近期死亡且几何近的 track 状态按新 id re-key 继承。
        self._grave = {}         # 旧 track_id -> (死亡帧号, 最后平滑框, 最后输出,
                                 #                  votes, 锁存 label, 框 conf)
        self._last_tbox = {}     # track_id -> 本帧平滑框（死亡时入殓用）
        self._frame_n = 0

    # ── 主入口 ──────────────────────────────────────────────

    def _confirm_spawns(self, raw):
        """新框入场确认：与现有 track（含丢失未超时的短闪回）有重叠
        （IoU≥iou_thr）的框视为持续检测直接放行；其余须连续
        spawn_confirm 帧按最佳 IoU 链（≥0.5）延续才喂 tracker。只拦
        单帧/断续的背景误检（new_track_conf 放低后易被低门放行的碎片），
        跟踪中或只闪失 1-2 帧的手不受影响。"""
        if self._spawn_confirm <= 1:
            return raw
        refs = []                        # 放行基准：活跃 track 的平滑框+原始框
        for t in self._tracker.tracks:
            if t.active:
                refs.append(list(t.box))
                refs.append(list(t.raw_box))
        out, cand, used = [], {}, set()
        for b in raw:
            if refs and max(iou(b[:4], r) for r in refs) >= self._iou_thr:
                out.append(b)
                continue
            best_key, best_iou = None, 0.5
            for key, (pb, _n) in self._spawn_cand.items():
                if key in used:
                    continue
                v = iou(b[:4], pb[:4])
                if v > best_iou:
                    best_key, best_iou = key, v
            if best_key is not None:
                n = self._spawn_cand[best_key][1] + 1
                if n >= self._spawn_confirm:
                    out.append(b)        # 确认入场
                else:
                    cand[best_key] = (b, n)
                used.add(best_key)
            else:
                cand[(len(cand),)] = (b, 1)   # 新候选
        self._spawn_cand = cand
        return out

    def detect(self, frame_bgr):
        self._frame_n += 1
        self._held = set()               # 每帧重建（探针/诊断快照）
        self._last_pose_conf = {}
        self._last_raw_pose = []         # 每帧重建（含无 tracker/无手路径）
        h, w = frame_bgr.shape[:2]
        raw = self._detect_boxes(frame_bgr, h, w)
        self.last_boxes = raw
        if self._tracker is None:
            return self._detect_stateless(frame_bgr, raw, h, w)

        # 1) 追踪 + 入殓 + 复活继承 + 运动门控推理
        # 新框先过入场确认（spawn_confirm），再喂 tracker
        feed = (self._confirm_spawns(raw) if self._spawn_confirm > 1
                else raw)
        self._tracker.update_detections([list(b[:4]) for b in feed],
                                        [b[4] for b in feed])
        self._bury_dead()
        self._revive_tracks(h, w)
        boxes_for_pose, idxs = self._tracker.get_boxes_for_pose()
        if boxes_for_pose:
            # 裁剪来源统一口径（raw=快动跟随；平滑框只负责匹配/显示），
            # 再对两活跃框做重叠裁剪（见 _clip_overlap），最后取本轮
            # 需推理的子集。重叠裁剪必须在子集化**之前**做：另一只手
            # 本帧可能不需要推理，但它的框仍要参与裁剪——否则单框 crop
            # 吞掉另一只手时姿态跟错手（000005 f232-249 宽框教训）。
            crop_src = {}
            for t in self._tracker.tracks:
                if t.active and t.lost_counter == 0:
                    crop_src[t.id] = (list(t.raw_box) if self._pose_box_raw
                                      else list(t.box))
            if len(crop_src) == 2:
                ids2 = sorted(crop_src)
                crop_src[ids2[0]], crop_src[ids2[1]] = self._clip_overlap(
                    crop_src[ids2[0]], crop_src[ids2[1]])
            boxes_for_pose = [crop_src[self._tracker.tracks[i].id]
                              for i in idxs]
        new_pose = {}      # track_id -> (tracks 下标, (21,2) 合格原始点)
        raw_map = {}       # track_id -> (21,2) 原始点（含低置信持出帧，
                           # 仅退化/无输出不入；last_raw_pose 数据源）
        degraded = set()   # 本轮要求推理但退化/无输出的 track_id
        if boxes_for_pose:
            kpts, scores = self._pose(frame_bgr, bboxes=boxes_for_pose)
            if kpts is None:
                degraded.update(self._tracker.tracks[i].id for i in idxs)
            else:
                sc = (None if scores is None
                      else np.asarray(scores, np.float32))
                for j, (idx, k) in enumerate(zip(idxs, kpts)):
                    pts = np.asarray(k, np.float32).reshape(21, 2)
                    tid = self._tracker.tracks[idx].id
                    if sc is not None and j < len(sc):
                        self._last_pose_conf[tid] = float(
                            np.asarray(sc[j], np.float32).mean())
                    if self._degenerate(pts, boxes_for_pose[j]):
                        degraded.add(tid)      # 冻结：不写 tracker
                    elif self._pose_conf_thr is not None \
                            and tid in self._last_pose_conf \
                            and self._last_pose_conf[tid] < self._pose_conf_thr \
                            and tid in self._last_out:
                        # 低置信骨架：只对有旧输出的 track 持出（新 track
                        # 无旧点可持，低置信首帧宁可直出保存在性）。
                        # hold_max 逃逸：持续低置信 = 大概率真实新姿势
                        # （握拳/抓取——黑手套逐点置信天然偏低），无限
                        # hold 会把骨架永久冻在旧姿势（"姿势无法捕捉"根
                        # 因）。连续满 hold_max 帧放行本轮低置信点（走
                        # 正常新 pose 路径，OneEuro 从旧点收敛跟上）；
                        # 瞬时低置信（运动模糊）仍持旧点防抖。
                        if self._hold_max is not None:
                            self._hold_count[tid] = \
                                self._hold_count.get(tid, 0) + 1
                            if self._hold_count[tid] >= self._hold_max:
                                new_pose[tid] = (idx, pts)
                                self._hold_count[tid] = 0
                            else:
                                self._held.add(tid)
                        else:
                            self._held.add(tid)
                    else:
                        self._hold_count[tid] = 0
                        new_pose[tid] = (idx, pts)
                    # 原始点缓存（左目 2D 显示与右目同口径用）：所有
                    # 推理结果都记（含低置信持出帧与退化帧——手背视图
                    # hand5 塌缩被判退化时，2D 显示侧仍需有点跟随手而
                    # 非冻旧点；只做显示数据源，3D/槽位/voter/滤波状态
                    # 的 degenerate 保护完全不变）。只做数据源，不碰
                    # 任何跟踪/滤波状态。
                    raw_map[tid] = pts
        if new_pose:
            idxs2 = [v[0] for v in new_pose.values()]
            kp = np.stack([v[1] for v in new_pose.values()])
            self._tracker.update_pose_results(
                idxs2, kp, np.zeros((len(idxs2), 21), np.float32))
            for v in new_pose.values():
                # 门控生效前提：_needs_pose 要求 last_good_kpts 非 None
                # （hand_tracker.py:293）。首次合格 pose 后初始化一次。
                # 不用 update_last_good API——它按 active 列表下标取 track
                # （hand_tracker.py:270），与 get_boxes_for_pose 返回的
                # self.tracks 下标不一致（超 max_hands 被停用的 track 仍在
                # self.tracks 中，实机 world 假框触发 3 active → IndexError）。
                tr = self._tracker.tracks[v[0]]
                tr.last_good_kpts = tr.kpts.copy()

        # 2) 汇总输出（track_id 稳定序，防 top-2 conf 换位造成槽位换手）
        tboxes, tkpts, _, ids = self._tracker.get_results()
        if not ids:
            self._gc_dead(set())
            return []
        self._last_tbox = {int(tid): list(b)
                           for tid, b in zip(ids, tboxes)}
        ts = (time.perf_counter() - self._t0) * 1000.0
        confs = self._match_confs(tboxes, raw)
        t_by_id = {t.id: t for t in self._tracker.tracks}
        out = []
        raw_out = []        # 与 out 同序的原始点缓存（无新原始点为 None）
        for i in np.argsort(ids):
            tid = int(ids[i])
            if tid in new_pose:
                pts = new_pose[tid][1]
                self._posed[tid] = True
                self._stale[tid] = 0
                self._last_out[tid] = (self._smooth_points(tid, pts, w, h, ts)
                                       if self._use_oe
                                       else pts.astype(np.float32))
                pts_out = self._last_out[tid]
                raw_lab = self._handedness(pts)
                raw_out.append(raw_map.get(tid))
            elif tid not in self._posed:
                raw_out.append(None)
                continue       # 全零占位（get_results 对未出过 pose 的 track
                               # 吐 (21,2) 零数组，hand_tracker.py:231）→ 不输出
            else:
                raw_out.append(raw_map.get(tid))  # 持出帧=原始点；退化/门控帧=None
                # 冻结/持出帧：门控跳过推理不算退化（健康静止手）；只退化帧计数。
                # 低置信 hold（tid ∈ _held）也走这里——不进 degraded 不计
                # stale，只持出 _last_out 旧点、raw_lab=None 不搅动锁存。
                if tid in degraded:
                    self._stale[tid] = self._stale.get(tid, 0) + 1
                if self._stale.get(tid, 0) > self._freeze_max:
                    continue   # 幽灵手上限 → 本帧不出，走下游传播
                pts_out = self._last_out[tid]
                feed = (self._use_oe and tid in self._held
                        and tid in self._last_out)
                if feed:
                    # hold 期间把纯旧点回喂 OneEuro 保持滤波时间连续
                    # （不喂补偿点——补偿必须在滤波器**外**施加：喂补偿
                    # 点会被滤波器按 alpha 打折（30fps 下 alpha≈0.5），
                    # 骨架只能跟到框速度的一半，仍然滞后）
                    self._last_out[tid] = self._smooth_points(
                        tid, pts_out, w, h, ts)
                    pts_out = self._last_out[tid]
                # 运动补偿：旧骨架按平滑框中心位移平移（hold_translate）。
                # 快动时硬冻结=骨架钉死原地+恢复瞬间跳回（闪烁主因）；
                # 框中心至少以 EMA 速度跟随手，平移后骨架随框滑动。
                # prev_center 是上次推理时的框中心，总位移 = 当前框中心
                # − prev_center；该 track 出过 pose 才到此处，prev_center
                # 必非 None（新 track 无旧点可持，首帧低置信宁可直出）。
                # 补偿在滤波器外、_last_out 恒为纯滤波值 → 不会重复累加
                # （旧实现把补偿后的点回写 _last_out 再加全量位移，逐帧
                # 累加致骨架飞出框外再猛跳回——2026-08-24 实机"更糟糕"
                # 根因）。恢复帧滤波从纯旧点收敛，有一次小幅回跳后跟上。
                if self._hold_translate:
                    tr = t_by_id.get(tid)
                    if tr is not None and tr.prev_center is not None:
                        cx = (tr.box[0] + tr.box[2]) / 2.0
                        cy = (tr.box[1] + tr.box[3]) / 2.0
                        pts_out = pts_out + np.asarray(
                            [cx - tr.prev_center[0],
                             cy - tr.prev_center[1]], np.float32)
                raw_lab = None
            label = (self._vote_label(tid, raw_lab, w, h, tboxes, ids)
                     if self._use_vote
                     else (raw_lab or self._last_label.get(tid, "")))
            conf = confs[i]
            if conf is not None:
                self._conf_mem[tid] = conf
            else:
                conf = self._conf_mem.get(tid, 0.5)
            out.append(DetectedHand(landmarks=pts_out.copy(),
                                    label=label, score=float(conf), index=0))
        self._last_raw_pose = raw_out
        self._gc_dead(set(int(i) for i in ids))
        return out

    # ── 框源（world / yolo 后端） ───────────────────────────

    def _detect_boxes(self, frame_bgr, h, w):
        if self._world is not None:
            boxes, confs = self._world(frame_bgr, conf=self.det_conf,
                                       max_boxes=self.num_hands)
            # world postprocess 已做 clip/8px 过滤/两级 NMS/按 conf Top-N
            return [(float(x1), float(y1), float(x2), float(y2), float(c))
                    for (x1, y1, x2, y2), c in zip(boxes, confs)]
        r = self._model(frame_bgr, conf=self.det_conf, imgsz=640,
                        device=self.device, verbose=False)[0]
        boxes = []
        for xyxy, conf in zip(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.conf.cpu().numpy()):
            x1 = max(0.0, min(float(xyxy[0]), w - 1.0))
            y1 = max(0.0, min(float(xyxy[1]), h - 1.0))
            x2 = max(0.0, min(float(xyxy[2]), w - 1.0))
            y2 = max(0.0, min(float(xyxy[3]), h - 1.0))
            if x2 - x1 < 8 or y2 - y1 < 8:      # 同 glove_package 口径
                continue
            boxes.append((x1, y1, x2, y2, float(conf)))
        boxes.sort(key=lambda b: -b[4])
        return boxes[:self.num_hands]

    def _detect_stateless(self, frame_bgr, raw, h, w):
        """use_tracker=False 的保底路径（原无状态行为原样保留）。"""
        if not raw:
            return []
        kpts, scores = self._pose(frame_bgr, bboxes=[[b[0], b[1], b[2], b[3]]
                                                     for b in raw])
        hands = []
        if kpts is None:
            return hands
        sc = None if scores is None else np.asarray(scores, np.float32)
        for i, ((x1, y1, x2, y2, conf), k) in enumerate(zip(raw, kpts)):
            pts = np.asarray(k, np.float32).reshape(21, 2)
            if self._degenerate(pts, (x1, y1, x2, y2)):
                continue
            if sc is not None and i < len(sc) \
                    and self._pose_conf_thr is not None \
                    and float(np.asarray(sc[i], np.float32).mean()) \
                    < self._pose_conf_thr:
                continue       # 无状态路径：低置信骨架直接弃帧
            hands.append(DetectedHand(landmarks=pts,
                                      label=self._handedness(pts),
                                      score=conf, index=0))
        return hands

    # ── 稳定层组件 ─────────────────────────────────────────

    def _smooth_points(self, tid, k, w, h, ts_ms):
        """逐点 OneEuroFilter2D（归一化坐标域，与裸手同域同参数）。"""
        norm = np.asarray(k, np.float64) / [w, h]
        out = np.empty_like(norm)
        for j in range(21):
            key = (tid, j)
            f = self._filters.get(key)
            if f is None:
                f = self._filters[key] = OneEuroFilter2D(
                    self._oe_freq_min, self._oe_beta, self._oe_dcutoff)
            out[j, 0], out[j, 1] = f(norm[j, 0], norm[j, 1], ts_ms)
        return (out * [w, h]).astype(np.float32)

    def _vote_label(self, tid, raw_lab, w, h, tboxes, ids):
        """per-track 手性锁存：连续 vote_window 票相同 → 锁死不再变。

        未锁定期返回最近一票原始几何 label（= 旧裸直通行为，下游 voter
        latch + 同 label 守卫吸收）；锁存后 label 永不变（镜像 voter
        latch——它才是旧验收 5 帧同 label 的稳定来源，滑窗多数会重开
        已定问题，Project_Test10 实测 358/359 帧同 label）。
        """
        if len(tboxes) == 2:
            ca = ((tboxes[0][0] + tboxes[0][2]) / 2,
                  (tboxes[0][1] + tboxes[0][3]) / 2)
            cb = ((tboxes[1][0] + tboxes[1][2]) / 2,
                  (tboxes[1][1] + tboxes[1][3]) / 2)
            if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) \
                    < _OVERLAP_FRAC * max(w, h):
                return self._last_label.get(tid, "") or (raw_lab or "")
        dq = self._votes.setdefault(tid, deque(maxlen=self._vote_window))
        if raw_lab:
            dq.append(raw_lab)
        if self._last_label.get(tid):
            return self._last_label[tid]      # 已锁死：翻转不再重开
        votes = list(dq)
        if not votes:
            return ""
        if len(votes) == self._vote_window \
                and all(v == votes[0] for v in votes):
            self._last_label[tid] = votes[0]  # 连续 N 票一致 → 锁死
            return votes[0]
        return votes[-1]

    def _clip_overlap(self, box_a, box_b):
        """两活跃框重叠时按中垂面裁剪（主导轴）——各自 crop 只含自己的手。

        world 在手接近/交叠时会发宽框（000005 f232-249：左手框 x2 从
        ~900 膨胀到 ~1173，把右手整个吞进去，IoU≈0.4），RTMPose 在宽框
        里会跟到另一只手 → 骨架瞬移到对侧手。裁剪只作用于 pose 裁剪
        （tracker 平滑框/显示不受影响）：以两框中心的连线的中垂面为准，
        只在中心距更大的主导轴裁（dx≥dy 裁 x，否则裁 y）——副轴常完全
        重叠（双手同高度），裁副轴会把真手截半。
        不重叠/中心过近（<20px）→ 原样返回。
        """
        a, b = list(box_a), list(box_b)
        ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
        cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        dx, dy = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
        if max(dx, dy) < 20.0:
            return a, b
        if dx >= dy:
            if max(a[0], b[0]) >= min(a[2], b[2]):   # x 向不重叠
                return a, b
            mid = (ca[0] + cb[0]) / 2.0
            if ca[0] < cb[0]:
                a[2] = min(a[2], mid)
                b[0] = max(b[0], mid)
            else:
                a[0] = max(a[0], mid)
                b[2] = min(b[2], mid)
        else:
            if max(a[1], b[1]) >= min(a[3], b[3]):   # y 向不重叠
                return a, b
            mid = (ca[1] + cb[1]) / 2.0
            if ca[1] < cb[1]:
                a[3] = min(a[3], mid)
                b[1] = max(b[1], mid)
            else:
                a[1] = max(a[1], mid)
                b[3] = min(b[3], mid)
        return a, b

    def track_boxes(self):
        """活跃 track 的平滑框 [(x1,y1,x2,y2)]——detect() 内 pose 裁剪
        同源口径（active 且未丢失）。右目共享框路径用（同场景双目，
        框按视差平移到右目坐标后跑 pose，见 live_demo 右目手套块），
        只读不碰 tracker 状态。"""
        if self._tracker is None:
            return []
        return [list(t.box) for t in self._tracker.tracks
                if t.active and t.lost_counter == 0]

    def pose_on_boxes(self, frame_bgr, boxes, keep_degenerate=False):
        """在给定框上跑当前姿态后端（右目共享框路径，2D 显示用）。

        返回 (21,2) 像素点列表（退化框过滤后无输出；keep_degenerate=True
        时退化框的原始点也返回——手背视图 hand5 塌缩被判退化时，2D
        显示侧仍需点跟随手；本函数只供显示路径，3D/槽位不经过）。
        复用本实例 _pose：RTMPose 是 stateless ONNX；mediapipe 后端是
        IMAGE 模式（RunningMode.IMAGE，无跨调用跟踪状态）——均无跨流
        污染（与 VIDEO 模式 det_r 教训不同），b 键热切换后端自动跟随。
        退化过滤与 detect() 同口径（送进框的点按同框判）。"""
        if not boxes or self._pose is None:
            return []
        kpts, _ = self._pose(frame_bgr,
                             bboxes=[list(map(float, b)) for b in boxes])
        if kpts is None:
            return []
        out = []
        for k, b in zip(kpts, boxes):
            pts = np.asarray(k, np.float32).reshape(21, 2)
            if keep_degenerate or not self._degenerate(pts, b):
                out.append(pts)
        return out

    def last_raw_pose(self):
        """本帧 detect() 内实际推理的原始关键点（稳定层之前），与
        detect() 返回值同序（track_id 稳定序）；本帧无新原始点的 track
        对应 None（门控跳过/冻结帧；退化帧自 2026-08-25 起也记录——
        手背视图 2D 显示跟随需要，3D/槽位链保护不变）。

        S80C 左目 2D 显示用：右目画 pose_on_boxes 原始点，左目画
        detect() 稳定层输出——稳定层（逐点 OneEuro + hold 持出 +
        hold_translate 运动补偿）在 S80C（world 框噪声 + 低置信段）
        会随框漂移/持旧点变形（左目比右目差的根因）；改画同帧原始点
        后左右同口径。数据来自 detect() 本帧推理（含低置信持出帧），
        零额外推理开销；3D 槽位链仍用 detect() 稳定层输出不变。
        只读不碰任何状态。"""
        return self._last_raw_pose

    def _match_confs(self, tboxes, raw):
        """tracker 平滑框 ↔ raw 框（带 conf）mini-IoU 回填 conf（score 用）。"""
        confs = []
        for tb in tboxes:
            best = None
            for rb in raw:
                iou_v = iou(tb, rb[:4])
                if iou_v >= 0.1 and (best is None or iou_v > best[0]):
                    best = (iou_v, rb[4])
            confs.append(best[1] if best else None)
        return confs

    def _bury_dead(self):
        """把刚死的 track 的稳定层状态整体入殓（复活继承候选池）。

        必须在 get_results/_last_tbox 重建之前调用：_last_tbox 每帧
        重建为当前活跃框，死亡帧的最后平滑框只有这里拿得到（上一帧
        副本）。入殓后 per-track 状态立即弹出（防泄漏）；滤波器保留
        在 _filters 供复活 re-key，本帧未命中复活时由 _gc_dead 兜底。
        """
        live = {t.id for t in self._tracker.tracks}
        dead = [t for t in set().union(self._votes, self._last_label,
                                       self._last_out, self._posed,
                                       self._stale, self._conf_mem,
                                       self._hold_count)
                if t not in live]
        for tid in dead:
            if tid in self._posed and tid in self._last_tbox:
                self._grave[tid] = (self._frame_n,
                                    self._last_tbox[tid],
                                    self._last_out.get(tid),
                                    self._votes.get(tid),
                                    self._last_label.get(tid),
                                    self._conf_mem.get(tid))
            for d in (self._votes, self._last_label, self._last_out,
                      self._posed, self._stale, self._conf_mem,
                      self._hold_count):
                d.pop(tid, None)

    def _gc_dead(self, live_ids):
        """兜底清理：入殓已在 _bury_dead 完成，这里只删残余状态。

        滤波器保留到坟墓过期（复活窗口内 re-key 需要），过期才删。
        """
        graved = set(self._grave)
        for key in [k for k in self._filters
                    if k[0] not in live_ids and k[0] not in graved]:
            del self._filters[key]
        for tid in [t for t in set().union(self._votes, self._last_label,
                                           self._last_out, self._posed,
                                           self._stale, self._conf_mem,
                                           self._hold_count)
                    if t not in live_ids]:
            for d in (self._votes, self._last_label, self._last_out,
                      self._posed, self._stale, self._conf_mem,
                      self._hold_count):
                d.pop(tid, None)
        # 过期坟墓清理（超 _REVIVE_MAX 帧不再可复活，滤波器一并删）
        for gid in [g for g, v in self._grave.items()
                    if self._frame_n - v[0] > _REVIVE_MAX]:
            del self._grave[gid]
            for j in range(21):
                self._filters.pop((gid, j), None)

    def _revive_tracks(self, h, w):
        """复活继承：把坟墓里的稳定层状态 re-key 给新 track。

        新 track = tracker 活跃但不在任何 per-track 状态里的 id（被删
        track 的重建或全新手，包括同帧删旧建新的 churn）。一墓一手贪心
        最近匹配，框中心距 < _REVIVE_DIST*max(w,h) 才继承——手在丢失
        期间真移动了则几何不近，走冷启动（现行为）。命中后：手性锁存
        立即延续（无 label 翻转）、OneEuro 第 2 帧起有历史、首帧退化
        时可冻结到上次输出。
        """
        if not self._grave:
            return
        live = {t.id for t in self._tracker.tracks}
        fresh = [tid for tid in live
                 if tid not in self._posed and tid not in self._votes]
        if not fresh:
            return
        tbox = {t.id: t.box for t in self._tracker.tracks}
        dmax = _REVIVE_DIST * max(h, w)
        cand = {g: v for g, v in self._grave.items()
                if self._frame_n - v[0] <= _REVIVE_MAX}
        used = set()
        while True:
            best = None     # (dist, new_tid, grave_tid)
            for tid in fresh:
                b = tbox.get(tid)
                if b is None:
                    continue
                c = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                for gid, (_, gbox, _, _, _, _) in cand.items():
                    if gid in used:
                        continue
                    gc = ((gbox[0] + gbox[2]) / 2.0,
                          (gbox[1] + gbox[3]) / 2.0)
                    d = math.hypot(c[0] - gc[0], c[1] - gc[1])
                    if d < dmax and (best is None or d < best[0]):
                        best = (d, tid, gid)
            if best is None:
                break
            _, tid, gid = best
            _, _, gout, gvotes, glabel, gconf = self._grave.pop(gid)
            for j in range(21):
                f = self._filters.pop((gid, j), None)
                if f is not None:
                    self._filters[(tid, j)] = f
            if gvotes is not None:
                self._votes[tid] = gvotes
            if glabel:
                self._last_label[tid] = glabel
            if gconf is not None:
                self._conf_mem[tid] = gconf
            if gout is not None:
                self._last_out[tid] = gout
            self._posed[tid] = True    # 复活即视为出过 pose：首帧退化可冻结
            self._stale[tid] = 0
            fresh.remove(tid)
            used.add(gid)

    def _clear_state(self):
        self._filters.clear()
        self._votes.clear()
        self._last_label.clear()
        self._last_out.clear()
        self._posed.clear()
        self._stale.clear()
        self._conf_mem.clear()
        self._grave.clear()
        self._last_tbox.clear()
        self._held = set()
        self._spawn_cand = {}
        self._hold_count.clear()
        self._last_pose_conf = {}
        self._last_raw_pose = []
        self._frame_n = 0
        self._t0 = time.perf_counter()

    # ── 退化过滤 / 手性（旧链定案，不动） ───────────────────

    @staticmethod
    def _degenerate(pts, box):
        """退化族三条件（对 tracker 平滑框判）：钳边/聚团/出框任一命中即弃。"""
        x1, y1, x2, y2 = box
        outside = int(np.sum((pts[:, 0] < x1) | (pts[:, 0] > x2) |
                             (pts[:, 1] < y1) | (pts[:, 1] > y2)))
        if outside >= 16:
            return True
        uniq = len(np.unique(pts.round(0).astype(np.int64), axis=0))
        if uniq < 15:
            return True
        span = math.hypot(pts[:, 0].max() - pts[:, 0].min(),
                          pts[:, 1].max() - pts[:, 1].min())
        return span < 0.2 * math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _handedness(pts):
        """几何合成：axis=P9-P0 与 d=P5-P0 的 z 叉积符号定左右（掌心朝相机）。"""
        axis = pts[9] - pts[0]
        d = pts[5] - pts[0]
        cross = axis[0] * d[1] - axis[1] * d[0]
        return "Left" if cross > 0 else "Right"

    # ── 生命周期 ───────────────────────────────────────────

    def _build_pose(self, backend, pose_device):
        """按后端名构造关键点后端（统一契约见 pose_backends.py）。"""
        if backend == "mediapipe":
            return MediaPipePoseBackend(model_path=self._pose_model,
                                        device=pose_device)
        if backend != "rtmpose":
            raise ValueError(f"未知姿态后端: {backend}（rtmpose/mediapipe）")
        return RtmposePoseBackend(device=pose_device)

    def set_pose_backend(self, backend):
        """热切换姿态后端（live_demo b 键）：仅换 self._pose。

        tracker/逐点 OneEuro/手性锁存/冻结缓冲/时间基准 _t0 等帧间
        状态全部保留——track_id 连续、滤波不停摆（对比 g 键的 close
        +重建整实例：那会清空全部稳定层状态）。切换后首帧滤波点继承
        旧后端输出，OneEuro 平滑收敛（一次性小幅过渡）；MediaPipe
        加载 .task ~百 ms 级，会卡一帧。"""
        if backend == self.pose_backend:
            return
        self._pose.close()
        self._pose = self._build_pose(backend, self.pose_device)
        self.pose_backend = backend
        self.pose_device = self._pose.device

    def reset(self):
        """清空全部帧间状态（tracker/滤波器/票仓）。"""
        if self._tracker is not None:
            self._tracker.clear()
        self._clear_state()
        self.last_boxes = []

    def close(self):
        self._model = None
        self._world = None
        if self._pose is not None:
            self._pose.close()
        self._pose = None
        self._tracker = None
        self._clear_state()
