#!/usr/bin/env python3
"""
手性投票：MediaPipe handedness 逐帧独立分类，遮挡/快动/对称手势时闪烁
（222_000009 实测 5 处），导致 match_hands 槽位排序不稳
（stereo_triangulate.py:312 按 Left 排序）、tracker 换手误重置、骨架身份
对穿。修复：每目一个 HandednessVoter，维护 ≤2 条手轨迹（质心最近贪心
关联，ByteTrack 式，防检测顺序变化串扰——区别于 MediaPipe 内置滤波按
列表下标索引的串位问题），每条轨迹带 7 帧 label 票仓，当前帧 = 轨迹票
+ 原始票严格多数表决输出稳定 label。

v1 教训（单次最近历史项投票 → 1:1 平票保原始 → 闪烁穿透，009 实测
16 次翻转）：必须按**轨迹**聚合窗口内全部历史票，7 票多数才压得住
单帧闪烁（6:1）。
v2 教训（双手交叠时贪心关联交叉 → 票仓自锁错 label，009 抓瓶处
293→294 翻转锁死 114 帧）：**重叠守卫**——两质心 <OVERLAP_PX 时
不做表决（原始 label 直通）并把两轨迹票仓重播种为原始票，防交叉
关联把对方轨迹的旧票带进自己的票仓。
"""

from __future__ import annotations

import atexit
import os
from collections import deque

import numpy as np

VOTE_WINDOW = 7           # 轨迹票仓帧数（手性闪烁是瞬态，7 帧多数即可压掉）
MIN_VOTE_SCORE = 0.7      # handedness 置信度低于此的原始检测不计票
ASSOC_GATE = 0.12         # 关联距离门限（×max(宽,高)：1280×800 下 ≈154px）
OVERLAP_PX = 0.05         # 重叠守卫门限（×max(宽,高)：1280×800 下 64px，
                          # 抓取等双手交叠时质心距远小于此）
MAX_TRACKS = 2            # MediaPipe 单目最多 2 手

_DEBUG_PATH = os.environ.get("HAND3D_IDENTITY_DEBUG")
_DEBUG_LINES: list = []   # "frame,cam;pre,post,score,cx,cy|..." 每帧每目一行


class HandednessVoter:
    """单目实例。update(hands) 原地覆盖每只 DetectedHand 的 label。

    轨迹 = {pos: 最近质心, votes: deque(label), last: 最近稳定 label,
    idle: 未关联帧数}。每帧贪心分配当前手→最近未占用轨迹（门限内）；
    未分配轨迹冻结（票仓保留，手暂离帧再回来继续投票）；新轨迹超
    MAX_TRACKS 时替换最旧轨迹。
    """

    def __init__(self, window: int = VOTE_WINDOW,
                 min_score: float = MIN_VOTE_SCORE):
        self.window = window
        self.min_score = min_score
        self._tracks = []   # [{"pos": ndarray, "votes": deque, "last": str, "idle": int}]
        if _DEBUG_PATH:
            atexit.register(self._dump_debug)

    def _dump_debug(self) -> None:
        try:
            with open(_DEBUG_PATH, "a") as f:
                f.write("\n".join(_DEBUG_LINES) + "\n")
        except OSError:
            pass

    @staticmethod
    def _centroid(h) -> np.ndarray | None:
        pts = np.asarray(h.landmarks, np.float64).reshape(-1, 2)
        ok = np.isfinite(pts).all(axis=1)
        if ok.sum() < 3:
            return None
        return np.median(pts[ok], axis=0)

    def update(self, hands: list, frame_w: int = 1280, frame_h: int = 800,
               frame: int | None = None, cam: str = "?") -> None:
        pre = [(h.label, float(h.score)) for h in hands]
        if not hands:
            self._tracks = []          # 空帧：轨迹全清（重新开始）
            self._debug(frame, cam, pre, [])
            return
        gate = ASSOC_GATE * max(frame_w, frame_h)
        cents = [self._centroid(h) for h in hands]
        overlap = (len(cents) == 2 and cents[0] is not None
                   and cents[1] is not None
                   and float(np.linalg.norm(cents[0] - cents[1]))
                   <= OVERLAP_PX * max(frame_w, frame_h))

        # ── 贪心分配：当前手 → 最近未占用轨迹（门限内）──
        used, assigned = set(), {}     # hand_idx → track_idx
        for i, c in enumerate(cents):
            if c is None:
                continue
            best_j, best_d = -1, np.inf
            for j, tr in enumerate(self._tracks):
                if j in used:
                    continue
                dd = float(np.linalg.norm(c - tr["pos"]))
                if dd < best_d:
                    best_j, best_d = j, dd
            if best_j >= 0 and best_d <= gate:
                assigned[i] = best_j
                used.add(best_j)

        # ── 未分配的手 → 新轨迹（超限替换最旧）──
        for i, c in enumerate(cents):
            if c is None or i in assigned:
                continue
            tr = {"pos": c.copy(), "votes": deque(maxlen=self.window),
                  "last": "", "idle": 0}
            if len(self._tracks) >= MAX_TRACKS:
                j = max(range(len(self._tracks)),
                        key=lambda j: self._tracks[j]["idle"])
                self._tracks[j] = tr
                assigned[i] = j
            else:
                self._tracks.append(tr)
                assigned[i] = len(self._tracks) - 1

        if overlap:
            # 双手交叠：关联不可靠 → 标签/位置全冻结为交叠前的稳定值。
            # v3 教训：原始直通会放行交叠期闪烁（009 抓瓶 169-176 实测），
            # 重播种原始票会把错误原始票自锁进票仓（293→294 锁 114 帧）；
            # 冻结则交叠期 label 恒定，分离后按冻结位置贪心回关联不断身份。
            for i, h in enumerate(hands):
                if i in assigned:
                    tr = self._tracks[assigned[i]]
                    if tr["last"]:
                        h.label = tr["last"]     # 冻结：交叠前的稳定标签
                    tr["idle"] = 0
                    # 位置不更新：交叠期质心不可靠，冻结供分离后回关联
            for j, tr in enumerate(self._tracks):
                if j not in used:
                    tr["idle"] += 1
            self._debug(frame, cam, pre, [(h.label, cents[i]) for i, h in enumerate(hands)])
            return

        # ── 逐手表决（轨迹票 + 原始票，严格多数；平票沿用轨迹稳定 label）──
        for i, h in enumerate(hands):
            if i in assigned:
                tr = self._tracks[assigned[i]]
                votes = list(tr["votes"])
                if h.score >= self.min_score and h.label:
                    votes.append(h.label)       # 当前原始票
                if votes:
                    majority = max(set(votes), key=votes.count)
                    if votes.count(majority) > len(votes) / 2:
                        h.label = majority
                    elif tr["last"]:
                        h.label = tr["last"]    # 平票稳定优先（重播种后防闪烁）
                elif tr["last"]:
                    h.label = tr["last"]        # 无任何票（新手轨迹+空原始 label）
                # 007 实测：MediaPipe 偶发 label=""（无手性输出）→ votes 全空
                # max() 崩溃——上方 if votes 兜底，空 label 不入票仓
            # 更新轨迹：位置跟随当前质心、票仓存稳定 label、未关联轨迹 idle+1
            if i in assigned:
                tr = self._tracks[assigned[i]]
                if cents[i] is not None:
                    tr["pos"] = cents[i]
                if h.label:
                    tr["votes"].append(h.label)
                tr["last"] = h.label
                tr["idle"] = 0
        for j, tr in enumerate(self._tracks):
            if j not in used:
                tr["idle"] += 1
        self._debug(frame, cam, pre, [(h.label, cents[i]) for i, h in enumerate(hands)])

    def _debug(self, frame, cam, pre, post) -> None:
        if _DEBUG_PATH and frame is not None:
            parts = []
            for i, h in enumerate(pre):
                lab, sc = h
                c = post[i][1] if i < len(post) else None
                pos = f"{c[0]:.0f},{c[1]:.0f}" if c is not None else "-, -"
                parts.append(f"{lab},{post[i][0]},{sc:.2f},{pos}")
            _DEBUG_LINES.append(f"{frame},{cam};" + "|".join(parts))
