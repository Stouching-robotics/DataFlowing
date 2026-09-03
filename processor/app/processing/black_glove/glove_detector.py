"""黑手套检测器：YOLO-World 手套框 + RTMPose hand5 21 点 + 帧间稳定。

检测契约使用本包的 DetectedHand：detect(frame_bgr)
-> list[DetectedHand]（landmarks (21,2) 像素、MediaPipe 拓扑 0=腕、
label "Left"/"Right"、score=检测框 conf）。下游（voter/抬升/槽位/平滑/
渲染）零改动——唯一注意 HandednessVoter 只数 score>=0.7 的票（identity.py
MIN_VOTE_SCORE=0.7），框 conf 不足 0.7 时该手 label 不进票仓。已核实
identity.py:139/156-158：voter 对此退化为 latch（票仓只累积已表决
label，轨迹存续期锁死首票）——本模块按 track_id 自建票仓与 voter 互补。

2026-08-20 重构（用户指令：切换 world 自带检测器 + 抑制抖动）：

检测框后端 = 本包 glove_package.world_detector.WorldDetector（YOLO-World
yolov8m-worldv2.pt + 提示词 ["hand","glove"]，imgsz 320、conf 0.05、
两级 NMS、按 conf Top-N——包内 40 张实测 40/40 召回）。权重名含
"world" 走 world 后端，否则普通 YOLO（best.pt 回退开关，conf 0.3）。
world_detector 只读复用，不改包内任何文件。

帧间稳定层（对齐裸手 MediaPipeDetector 内部机制，见方案对照表）：
1. HandTracker（本包 glove_package.hand_tracker）：贪心 IoU 匹配
   + 框 EMA α=0.7 + 运动门控 3px（静止帧免 RTMPose 推理、0.33s 强制
   刷新）+ 丢框持 3 帧 + track_id 帧间身份。注意 _needs_pose 要求
   last_good_kpts 非 None 门控才生效（hand_tracker.py:293）——每 track
   首次合格 pose 后必须 update_last_good 初始化。
2. 逐点 OneEuroFilter2D（freq_min=5.0/beta=0.05/dcutoff=1.0，归一化
   坐标域，键 (track_id, 点序号)，单调钟）——裸手同款同参数；冻结帧
   不重复喂滤波器（防把点吸向陈旧值）。
3. 快动/低置信稳定：低置信结果冻结、框中心运动补偿、双手重叠裁剪，
   以及短时丢失后的轨迹状态复活，避免骨架钉死、跳手和左右身份重置。
4. 退化族过滤"丢弃→冻结"：三条件参数不动（旧链 3D 毒化定案——框外
   点≥16 / 唯一点<15 / span<0.2×框对角线，对 tracker 平滑框判）。命中
   退化的新 pose 不写 tracker → 输出上次滤波值；连续无合格 pose 超
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

from app.processing.black_glove.contracts import DetectedHand
from app.processing.black_glove.filters import OneEuroFilter2D
from app.processing.black_glove.glove_package.hand_tracker import HandTracker
from app.processing.black_glove.glove_package.world_detector import WorldDetector, iou
from app.processing.black_glove.pose_backends import (
    MediaPipePoseBackend, RtmposePoseBackend,
)

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
_WORLD_WEIGHTS = os.path.join(_PACKAGE_ROOT, "weights", "yolov8m-worldv2.pt")
_MP_TASK = os.path.join(_PACKAGE_ROOT, "hand_landmarker.task")
_WORLD_PROMPT = ["hand", "glove"]
_WORLD_CONF = 0.05
_YOLO_CONF = 0.3
_OVERLAP_FRAC = 0.05       # 双手框中心距 < 0.05*max(w,h) → 手性冻结（镜像 voter）
_REVIVE_MAX = 90           # 丢失后允许继承稳定状态的最长帧数
_REVIVE_DIST = 0.3         # 复活时框中心允许的最大相对距离


class GloveDetector:
    """YOLO-World/YOLO 出框 + HandTracker 追踪 + RTMPose 关键点。

    惰性加载：ultralytics/torch/rtmlib 只在首次构造时 import（裸手启动
    零开销；首次切换有一次 ~1s 加载 + CUDA 预热）。
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
        self.imgsz = max(160, int(imgsz))
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
        self._held = set()       # 本轮低置信度持出
        self._hold_count = {}    # tid -> 连续低置信 hold 帧数（hold_max 逃逸）
        self._last_pose_conf = {}  # track_id -> 姿态点均值置信度
        self._grave = {}         # 死亡轨迹的稳定状态，供短期复活
        self._last_tbox = {}     # 上一帧活跃轨迹框
        self._last_raw_pose = []
        self._frame_n = 0
        self._t0 = time.perf_counter()

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
        self._held = set()
        self._last_pose_conf = {}
        self._last_raw_pose = []
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
            # 裁剪来源统一口径：再对两活跃框做重叠裁剪，最后取本轮
            # 需要推理的子集。另一只手即使本帧不推理，也必须参与裁剪。
            crop_src = {}
            for track in self._tracker.tracks:
                if track.active and track.lost_counter == 0:
                    crop_src[track.id] = (
                        list(track.raw_box) if self._pose_box_raw
                        else list(track.box))
            # pose 裁剪边距：平滑框在快速运动时滞后于手（EMA α=0.7 稳态
            # 滞后 ~2.3 帧），手的前进边缘会冲出框外被裁掉 → RTMPose 在
            # 缺边的裁剪上估计 → 关键点被拉回框内 → 运动方向上出现
            # 肉眼可见的"延迟感"（实测 x 方向响应率仅 0.64、y 方向正常
            # 的不对称滞后）。按框尺寸外扩 15% 边距把滞后量包进来；
            # RTMPose 输入固定 256×256，边距只增加裁剪余量，不改变
            # tracker/匹配/显示任何状态。
            for _tid, _b in crop_src.items():
                _pad_x = 0.15 * max(1.0, _b[2] - _b[0])
                _pad_y = 0.15 * max(1.0, _b[3] - _b[1])
                _b[0] = max(0.0, _b[0] - _pad_x)
                _b[1] = max(0.0, _b[1] - _pad_y)
                _b[2] = min(float(w), _b[2] + _pad_x)
                _b[3] = min(float(h), _b[3] + _pad_y)
            if len(crop_src) == 2:
                ids2 = sorted(crop_src)
                crop_src[ids2[0]], crop_src[ids2[1]] = self._clip_overlap(
                    crop_src[ids2[0]], crop_src[ids2[1]])
            boxes_for_pose = [crop_src[self._tracker.tracks[i].id]
                              for i in idxs]
        new_pose = {}      # track_id -> (tracks 下标, (21,2) 合格原始点)
        raw_map = {}       # track_id -> (21,2) 原始点
        degraded = set()   # 本轮要求推理但退化/无输出的 track_id
        if boxes_for_pose:
            kpts, scores = self._pose(frame_bgr, bboxes=boxes_for_pose)
            if kpts is None:
                degraded.update(self._tracker.tracks[i].id for i in idxs)
            else:
                score_array = (None if scores is None
                               else np.asarray(scores, np.float32))
                for j, (idx, k) in enumerate(zip(idxs, kpts)):
                    pts = np.asarray(k, np.float32).reshape(21, 2)
                    tid = self._tracker.tracks[idx].id
                    if score_array is not None and j < len(score_array):
                        self._last_pose_conf[tid] = (
                            float(np.asarray(score_array[j], np.float32).mean())
                        )
                    if self._degenerate(pts, boxes_for_pose[j]):
                        degraded.add(tid)      # 冻结：不写 tracker
                    elif self._pose_conf_thr is not None \
                            and tid in self._last_pose_conf \
                            and self._last_pose_conf[tid] < self._pose_conf_thr \
                            and tid in self._last_out:
                        # hold_max 逃逸：持续低置信 = 大概率真实新姿势
                        # （手背/握拳——hand5 逐点置信天然偏低），无限
                        # hold 会把骨架永久冻在旧姿势。连续满 hold_max
                        # 帧放行本轮低置信点（走正常新 pose 路径收敛）；
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
                    if tid not in degraded:
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
        self._last_tbox = {int(tid): list(box)
                           for tid, box in zip(ids, tboxes)}
        ts = (time.perf_counter() - self._t0) * 1000.0
        confs = self._match_confs(tboxes, raw)
        t_by_id = {t.id: t for t in self._tracker.tracks}
        out = []
        raw_out = []
        for i in np.argsort(ids):
            tid = int(ids[i])
            if tid in new_pose:
                pts = new_pose[tid][1]
                self._posed[tid] = True
                self._stale[tid] = 0
                self._last_out[tid] = (self._smooth_points(tid, pts, w, h, ts)
                                       if self._use_oe
                                       else pts.astype(np.float32))
                raw_lab = self._handedness(pts)
                pts_out = self._last_out[tid]
                raw_out.append(raw_map.get(tid))
            elif tid not in self._posed:
                raw_out.append(None)
                continue       # 全零占位（get_results 对未出过 pose 的 track
                               # 吐 (21,2) 零数组，hand_tracker.py:231）→ 不输出
            else:
                raw_out.append(raw_map.get(tid))
                if tid in degraded:
                    self._stale[tid] = self._stale.get(tid, 0) + 1
                if self._stale.get(tid, 0) > self._freeze_max:
                    continue   # 幽灵手上限 → 本帧不出，走下游传播
                pts_out = self._last_out[tid]
                feed = (self._use_oe and tid in self._held
                        and tid in self._last_out)
                if feed:
                    self._last_out[tid] = self._smooth_points(
                        tid, pts_out, w, h, ts)
                    pts_out = self._last_out[tid]
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
        result = self._pose(frame_bgr, bboxes=[[b[0], b[1], b[2], b[3]]
                                               for b in raw])
        if isinstance(result, tuple):
            kpts, scores = result
        else:
            kpts, scores = result, None
        hands = []
        if kpts is None:
            return hands
        score_array = (None if scores is None
                       else np.asarray(scores, np.float32))
        for i, ((x1, y1, x2, y2, conf), k) in enumerate(zip(raw, kpts)):
            pts = np.asarray(k, np.float32).reshape(21, 2)
            if self._degenerate(pts, (x1, y1, x2, y2)):
                continue
            if (score_array is not None and i < len(score_array)
                    and self._pose_conf_thr is not None
                    and float(np.nanmean(score_array[i]))
                    < self._pose_conf_thr):
                continue
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

    def track_boxes(self):
        """返回当前活跃且未丢失轨迹的平滑框。"""
        if self._tracker is None:
            return []
        return [list(track.box) for track in self._tracker.tracks
                if track.active and track.lost_counter == 0]

    def pose_on_boxes(self, frame_bgr, boxes):
        """在指定框上调用当前姿态后端，不修改追踪器状态。"""
        if not boxes or self._pose is None:
            return []
        kpts, _ = self._pose(
            frame_bgr,
            bboxes=[list(map(float, box)) for box in boxes],
        )
        if kpts is None:
            return []
        out = []
        for k, box in zip(kpts, boxes):
            pts = np.asarray(k, np.float32).reshape(21, 2)
            if not self._degenerate(pts, box):
                out.append(pts)
        return out

    def last_raw_pose(self):
        """返回本帧实际推理得到的原始姿态，顺序与 detect() 一致。"""
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

    def _clip_overlap(self, box_a, box_b):
        """双手接近时沿主导轴裁开姿态 crop，避免 RTMPose 跟错手。"""
        a, b = list(box_a), list(box_b)
        ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
        cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        dx, dy = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
        if max(dx, dy) < 20.0:
            return a, b
        if dx >= dy:
            if max(a[0], b[0]) >= min(a[2], b[2]):
                return a, b
            mid = (ca[0] + cb[0]) / 2.0
            if ca[0] < cb[0]:
                a[2] = min(a[2], mid)
                b[0] = max(b[0], mid)
            else:
                a[0] = max(a[0], mid)
                b[2] = min(b[2], mid)
        else:
            if max(a[1], b[1]) >= min(a[3], b[3]):
                return a, b
            mid = (ca[1] + cb[1]) / 2.0
            if ca[1] < cb[1]:
                a[3] = min(a[3], mid)
                b[1] = max(b[1], mid)
            else:
                a[1] = max(a[1], mid)
                b[3] = min(b[3], mid)
        return a, b

    def _bury_dead(self):
        """保存刚被 tracker 删除的轨迹，供短时间内的新轨迹复活。"""
        live = {track.id for track in self._tracker.tracks}
        state_ids = set().union(self._votes, self._last_label, self._last_out,
                                self._posed, self._stale, self._conf_mem)
        for tid in state_ids - live:
            if tid in self._posed and tid in self._last_tbox:
                self._grave[tid] = (
                    self._frame_n,
                    self._last_tbox[tid],
                    self._last_out.get(tid),
                    self._votes.get(tid),
                    self._last_label.get(tid),
                    self._conf_mem.get(tid),
                )
            for state in (self._votes, self._last_label, self._last_out,
                          self._posed, self._stale, self._conf_mem):
                state.pop(tid, None)

    def _revive_tracks(self, h, w):
        """将位置接近的短期新轨迹接回旧轨迹的平滑和左右手状态。"""
        if not self._grave:
            return
        live = {track.id for track in self._tracker.tracks}
        fresh = [tid for tid in live
                 if tid not in self._posed and tid not in self._votes]
        if not fresh:
            return
        boxes = {track.id: track.box for track in self._tracker.tracks}
        max_dist = _REVIVE_DIST * max(h, w)
        candidates = {
            gid: value for gid, value in self._grave.items()
            if self._frame_n - value[0] <= _REVIVE_MAX
        }
        used = set()
        while True:
            best = None
            for tid in fresh:
                box = boxes.get(tid)
                if box is None:
                    continue
                center = ((box[0] + box[2]) / 2.0,
                          (box[1] + box[3]) / 2.0)
                for gid, value in candidates.items():
                    if gid in used:
                        continue
                    old_box = value[1]
                    old_center = ((old_box[0] + old_box[2]) / 2.0,
                                  (old_box[1] + old_box[3]) / 2.0)
                    distance = math.hypot(center[0] - old_center[0],
                                          center[1] - old_center[1])
                    if distance < max_dist and (best is None
                                                or distance < best[0]):
                        best = (distance, tid, gid)
            if best is None:
                break
            _, tid, gid = best
            _, _, old_out, old_votes, old_label, old_conf = \
                self._grave.pop(gid)
            for point_index in range(21):
                filt = self._filters.pop((gid, point_index), None)
                if filt is not None:
                    self._filters[(tid, point_index)] = filt
            if old_votes is not None:
                self._votes[tid] = old_votes
            if old_label:
                self._last_label[tid] = old_label
            if old_conf is not None:
                self._conf_mem[tid] = old_conf
            if old_out is not None:
                self._last_out[tid] = old_out
            self._posed[tid] = True
            self._stale[tid] = 0
            fresh.remove(tid)
            used.add(gid)

    def _gc_dead(self, live_ids):
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
        for gid in [gid for gid, value in self._grave.items()
                    if self._frame_n - value[0] > _REVIVE_MAX]:
            del self._grave[gid]
            for point_index in range(21):
                self._filters.pop((gid, point_index), None)

    def _clear_state(self):
        self._filters.clear()
        self._votes.clear()
        self._last_label.clear()
        self._last_out.clear()
        self._posed.clear()
        self._stale.clear()
        self._conf_mem.clear()
        self._held.clear()
        self._spawn_cand = {}
        self._hold_count.clear()
        self._last_pose_conf.clear()
        self._grave.clear()
        self._last_tbox.clear()
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
        if backend == "mediapipe":
            return MediaPipePoseBackend(model_path=self._pose_model,
                                        device=pose_device)
        if backend != "rtmpose":
            raise ValueError(f"未知姿态后端: {backend}（rtmpose/mediapipe）")
        return RtmposePoseBackend(device=pose_device)

    def set_pose_backend(self, backend):
        """只切换姿态后端，保留追踪、滤波和手性状态。"""
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
