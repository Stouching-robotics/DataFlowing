"""手部身份追踪模块 — 用 IoU 做跨帧框匹配，给每只手分配稳定 ID。

解决的问题（详见项目根因分析）：
  1. 框按置信度排序 → 帧间交换 → last_good_kpts 写入了错误的手 → 关键点粘连
  2. 运动门控 + 置信度冻结都按数组下标操作 → ID 交换后数据错乱
  3. 检测丢失一帧就清空所有状态 → 重新检测时 ID 跳变

核心思路：
  - 贪心 IoU 匹配（max_hands≤2 时等价于匈牙利算法且不依赖 scipy）
  - 每只手独立维护 kpts / scores / last_good_kpts / skip_counter / lost_counter
  - EMA 框平滑减少检测抖动
  - 短期记忆：连续丢失 ≤lost_timeout 帧的 track 保持活跃但不跑推理

用法：
    tracker = HandTracker(max_hands=2)
    while True:
        frame = cap.read()
        det_boxes = detector(frame)
        tracker.update_detections(det_boxes)

        boxes_for_pose, indices = tracker.get_boxes_for_pose()
        if boxes_for_pose:
            kpts, scores = pose(frame, bboxes=boxes_for_pose)
            tracker.update_pose_results(indices, kpts, scores)

        boxes, kpts, scores, ids = tracker.get_results()
        # 用 boxes/kpts/scores/ids 画图和做冻结逻辑
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# 复用 world_detector 里的 iou 函数，避免重复定义
from world_detector import iou


# ═══════════════════════════════════════════════════════════════════════
# 单条跟踪记录
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HandTrack:
    """一只手跨帧的状态。"""

    id: int                                   # 稳定 ID，分配后不变
    box: List[float]                          # EMA 平滑框 [x1, y1, x2, y2]
    raw_box: List[float]                      # 原始检测框（用于运动检测）
    prev_center: Optional[Tuple[float, float]] = None  # 上次推理时框中心
    kpts: Optional[np.ndarray] = None         # (21, 2)
    scores: Optional[np.ndarray] = None       # (21,)
    last_good_kpts: Optional[np.ndarray] = None  # (21, 2) 逐点冻结缓存
    skip_counter: int = 0                     # 连续跳过推理的帧数
    lost_counter: int = 0                     # 连续未匹配到检测的帧数
    active: bool = True                       # 当前帧是否活跃


# ═══════════════════════════════════════════════════════════════════════
# 追踪器
# ═══════════════════════════════════════════════════════════════════════

class HandTracker:
    """管理多只手的跨帧身份追踪。

    参数
    ----
    max_hands : int
        最多同时追踪的手数（默认 2）。
    iou_match_thr : float
        IoU 匹配阈值，≥ 此值认为两个框是同一只手（默认 0.3）。
    lost_timeout : int
        连续丢失超过此帧数后删除 track（默认 3）。
    movement_thresh : float
        框中心位移超过此像素数触发关键点推理（默认 3）。
    skip_timeout : int
        静止超过此帧数强制刷新关键点（默认 10）。
    box_smooth_alpha : float
        EMA 平滑系数，0=不平滑，1=完全不动（默认 0.7）。
    """

    def __init__(
        self,
        max_hands: int = 2,
        iou_match_thr: float = 0.3,
        lost_timeout: int = 3,
        movement_thresh: float = 3.0,
        skip_timeout: int = 10,
        box_smooth_alpha: float = 0.7,
    ):
        self.max_hands = max_hands
        self.iou_match_thr = iou_match_thr
        self.lost_timeout = lost_timeout
        self.movement_thresh = movement_thresh
        self.skip_timeout = skip_timeout
        self.box_smooth_alpha = box_smooth_alpha

        self.tracks: List[HandTrack] = []
        self._next_id: int = 0

    # ── 公共 API ──────────────────────────────────────────────────

    def update_detections(self, boxes: List[List[float]]) -> None:
        """每帧必须调用。传入当前帧的原始检测框（XYXY 格式）。

        内部做：贪心 IoU 匹配 → 创建/更新 track → 清理超时 track → EMA 框平滑。
        """
        # 仅保留前 max_hands 个框（与检测器的 Top-N 一致）
        curr_boxes = [list(b) for b in boxes[:self.max_hands]]
        prev_boxes = [t.box for t in self.tracks if t.active]

        # ── 贪心 IoU 匹配 ──────────────────────────────
        matches, unmatched_curr, unmatched_prev = _greedy_match(
            prev_boxes, curr_boxes, self.iou_match_thr
        )

        # 先把所有已有 track 标记为非活跃
        for t in self.tracks:
            t.active = False

        # ── 更新匹配到的 track ─────────────────────────
        for prev_idx, curr_idx in matches:
            track = self.tracks[prev_idx]
            track.raw_box = curr_boxes[curr_idx]
            track.box = _smooth_box(track.box, curr_boxes[curr_idx],
                                    self.box_smooth_alpha)
            track.lost_counter = 0
            track.active = True

        # ── 新建 track ─────────────────────────────────
        for curr_idx in unmatched_curr:
            new_track = HandTrack(
                id=self._next_id,
                box=curr_boxes[curr_idx],
                raw_box=curr_boxes[curr_idx],
            )
            self._next_id += 1
            self.tracks.append(new_track)

        # ── 老化未匹配的已有 track ─────────────────────
        for prev_idx in unmatched_prev:
            track = self.tracks[prev_idx]
            track.lost_counter += 1
            # 丢失但未超时 → 保持活跃（短期记忆）
            if track.lost_counter <= self.lost_timeout:
                track.active = True

        # ── 清理长期丢失的 track ───────────────────────
        self.tracks = [
            t for t in self.tracks
            if t.active or t.lost_counter <= self.lost_timeout
        ]

        # ── 限制活跃 track 数量 ≤ max_hands ────────────
        # 当旧 track 丢失 + 新检测不匹配时，可能同时存在 > max_hands 个活跃 track
        active = [t for t in self.tracks if t.active]
        if len(active) > self.max_hands:
            # 优先保留非丢失的（lost_counter==0），多余的关闭丢失最久的
            active.sort(key=lambda t: (t.lost_counter, -t.id))
            for t in active[self.max_hands:]:
                t.active = False

        # ── 对未跑推理的活跃 track 增加 skip_counter ──
        for track in self.tracks:
            if track.active and track.lost_counter == 0:
                if not _needs_pose(track, self.movement_thresh,
                                   self.skip_timeout):
                    track.skip_counter += 1

    def get_boxes_for_pose(self) -> Tuple[List[List[float]], List[int]]:
        """返回 (需要推理的框列表, 对应 track 在 self.tracks 里的下标)。

        只在运动超阈值 / 新 track / skip 超时的情况下才返回框。
        已丢失（lost_counter>0）的 track 不跑推理。
        """
        boxes_out: List[List[float]] = []
        indices_out: List[int] = []
        for i, track in enumerate(self.tracks):
            if not track.active or track.lost_counter > 0:
                continue
            if _needs_pose(track, self.movement_thresh, self.skip_timeout):
                boxes_out.append(track.box)
                indices_out.append(i)
        return boxes_out, indices_out

    def update_pose_results(
        self,
        indices: List[int],
        new_kpts: np.ndarray,      # (M, 21, 2)
        new_scores: np.ndarray,    # (M, 21)
    ) -> None:
        """把 RTMPose 推理结果写入对应 track。"""
        for idx, k, s in zip(indices, new_kpts, new_scores):
            track = self.tracks[idx]
            track.kpts = k.copy()
            track.scores = s.copy()
            # 记录本次推理时的框中心
            bx = track.box
            track.prev_center = ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)
            track.skip_counter = 0

    def get_results(self) -> Tuple[
        List[List[float]],
        Optional[np.ndarray],
        Optional[np.ndarray],
        List[int],
    ]:
        """返回 (boxes, kpts, scores, track_ids)。

        - boxes: 平滑后的框列表
        - kpts: (N, 21, 2) 或 None
        - scores: (N, 21) 或 None
        - track_ids: 稳定 ID 列表
        """
        active = [t for t in self.tracks if t.active]
        if not active:
            return [], None, None, []

        boxes = [t.box for t in active]
        ids = [t.id for t in active]

        kpts_list, scores_list = [], []
        for t in active:
            if t.kpts is not None:
                kpts_list.append(t.kpts.copy())
                scores_list.append(t.scores.copy())
            else:
                kpts_list.append(np.zeros((21, 2), dtype=np.float32))
                scores_list.append(np.zeros(21, dtype=np.float32))

        kpts = np.array(kpts_list, dtype=np.float32)
        scores = np.array(scores_list, dtype=np.float32)
        return boxes, kpts, scores, ids

    def update_last_good(
        self, track_index: int, point_index: int, value: np.ndarray
    ) -> None:
        """更新某个 track 某个关键点的冻结缓存。"""
        t = self._active_track(track_index)
        if t.last_good_kpts is None:
            t.last_good_kpts = (
                t.kpts.copy()
                if t.kpts is not None
                else np.zeros((21, 2), dtype=np.float32)
            )
        t.last_good_kpts[point_index] = value.copy()

    def get_last_good(
        self, track_index: int, point_index: int
    ) -> Optional[np.ndarray]:
        """读取某个 track 某个关键点的冻结缓存。"""
        t = self._active_track(track_index)
        if t.last_good_kpts is None:
            return None
        return t.last_good_kpts[point_index].copy()

    def clear(self) -> None:
        """重置所有 track。"""
        self.tracks.clear()
        self._next_id = 0

    # ── 内部 ──────────────────────────────────────────────────────

    def _active_track(self, index: int) -> HandTrack:
        """获取第 index 个活跃 track。"""
        active = [t for t in self.tracks if t.active]
        return active[index]

    @property
    def hand_count(self) -> int:
        return sum(1 for t in self.tracks if t.active)


# ═══════════════════════════════════════════════════════════════════════
# 内部辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _needs_pose(
    track: HandTrack,
    movement_thresh: float,
    skip_timeout: int,
) -> bool:
    """判断某只 track 是否需要运行关键点推理。

    条件（满足任一即返回 True）：
    1. 没有关键点数据（新 track）
    2. 连续跳过帧数超时（强制刷新）
    3. 框中心位移 ≥ movement_thresh
    """
    if track.kpts is None or track.last_good_kpts is None:
        return True
    if track.skip_counter >= skip_timeout:
        return True
    # 计算当前框中心与前次推理时中心的偏移
    bx = track.box
    cur_cx = (bx[0] + bx[2]) / 2.0
    cur_cy = (bx[1] + bx[3]) / 2.0
    if track.prev_center is not None:
        dx = cur_cx - track.prev_center[0]
        dy = cur_cy - track.prev_center[1]
        if math.hypot(dx, dy) >= movement_thresh:
            return True
    else:
        # 还没有 prev_center → 说明需要首次推理
        return True
    return False


def _smooth_box(
    old: Optional[List[float]],
    new: List[float],
    alpha: float,
) -> List[float]:
    """EMA 平滑框坐标。

    alpha=1.0 → 完全用旧值（不动）
    alpha=0.0 → 完全用新值（不平滑）
    """
    if old is None:
        return list(new)
    return [alpha * o + (1.0 - alpha) * n for o, n in zip(old, new)]


def _greedy_match(
    prev_boxes: List[List[float]],
    curr_boxes: List[List[float]],
    iou_thr: float,
) -> Tuple[
    List[Tuple[int, int]],  # matches: (prev_idx, curr_idx)
    List[int],               # unmatched_curr
    List[int],               # unmatched_prev
]:
    """贪心 IoU 匹配：按 IoU 降序逐一配对，已匹配的跳过。

    对于 max_hands≤2 的场景与匈牙利算法等价，且无需 scipy 依赖。
    """
    n_prev, n_curr = len(prev_boxes), len(curr_boxes)
    if n_prev == 0:
        return [], list(range(n_curr)), []
    if n_curr == 0:
        return [], [], list(range(n_prev))

    # 收集所有 (iou, prev_i, curr_j) ≥ 阈值的组合
    pairs: List[Tuple[float, int, int]] = []
    for i in range(n_prev):
        for j in range(n_curr):
            iou_val = iou(prev_boxes[i], curr_boxes[j])
            if iou_val >= iou_thr:
                pairs.append((iou_val, i, j))

    # 按 IoU 降序
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_prev: set = set()
    matched_curr: set = set()
    matches: List[Tuple[int, int]] = []

    for _iou_val, pi, cj in pairs:
        if pi not in matched_prev and cj not in matched_curr:
            matches.append((pi, cj))
            matched_prev.add(pi)
            matched_curr.add(cj)

    unmatched_curr = [j for j in range(n_curr) if j not in matched_curr]
    unmatched_prev = [i for i in range(n_prev) if i not in matched_prev]
    return matches, unmatched_curr, unmatched_prev
