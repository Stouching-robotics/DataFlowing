#!/usr/bin/env python3
"""
遮挡传播：槽位跟踪 + 预测 3D 的伪 pair 重检。

问题：stage-1 MediaPipe 偶发丢手（小幅度遮挡）→ 该帧无 pair → 无输出，
parquet 出现短缺口（基线 222_000008：hand_1 检出 57%，5 段 ≤3 帧短缺口）。

机制（两件套）：
1. HandSlotTracker —— 每槽位（hand_0/hand_1，match_hands 已按 left_label
   排序：Left 前）维护 αβ 滤波的 3D 位置/速度。缺手帧生成 **预测 3D**；
2. PseudoHandPair —— 预测 3D 包成伪 pair（l_idx/r_idx=-1，result 的
   valid_count=0、err=inf）并入**同一次** refine_batch：预测 3D →
   tri.project → _make_crop 现成路径，批量检测器一次批前向覆盖真手+救援
   crop，零逻辑复制。_adopt 判据对 valid_count=0 的粗结果自动退化：批量
   检测器找到任何几何一致结果就采纳（真救援，不标记 propagated）；
   全失败则返回预测 3D 兜底（reason 保持失败原因，parquet 标 propagated=True）。

幻觉防护：max_lost=15 硬顶（长缺口 45/89 帧保持 absent，不得幻觉）；
propagated 列可过滤；--propagate-max 0 全关。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stereo_s80m.stereo_triangulate import TriangulationResult  # noqa: E402


@dataclass
class PseudoHandPair:
    """丢失槽位的传播重检载体（接口与 HandPair 平替，供 refine_batch 消费）。

    result = TriangulationResult(预测 3D, 全 inf err)：valid_count=0 →
    _adopt 判据退化为"批量检测器找到任何几何一致结果就采纳"；全失败时
    _adopt 返回该预测结果本身（兜底传播）。
    """

    result: TriangulationResult
    left_label: str = ""
    l_idx: int = -1
    r_idx: int = -1


class HandSlotTracker:
    """两槽位（hand_0/hand_1）αβ 跟踪器，槽位顺序不重排（match_hands 已排序）。

    槽位 label 变化 → 槽位重置（换手了，旧状态污染无效）。
    """

    def __init__(self, max_lost: int = 15, alpha: float = 0.5, beta: float = 0.1,
                 debug_log: str = None):
        self.max_lost = max_lost
        self.alpha = alpha
        self.beta = beta
        self.slots = [{"label": None, "x": None, "v": None,
                       "last_t": None, "lost": 0} for _ in range(2)]
        self._dbg = open(debug_log, "w", newline="", encoding="utf-8") \
            if debug_log else None
        if self._dbg:
            self._dbg.write("frame,slot,event,label,lost\n")

    def debug(self, event: str, slot: int, t: int):
        if self._dbg:
            s = self.slots[slot]
            self._dbg.write(f"{t},{slot},{event},{s['label'] or ''},{s['lost']}\n")

    def slot_label(self, slot: int) -> str:
        return self.slots[slot]["label"] or ""

    def observe_slot(self, slot: int, label: str, pts3d: np.ndarray, t: int):
        """真实检测回写（真 pair 或救援成功）。label 变化 → 槽位重置。"""
        s = self.slots[slot]
        x_meas = np.asarray(pts3d, np.float64).reshape(-1, 3)
        if s["label"] is not None and s["label"] != label:
            s.update(label=None, x=None, v=None, last_t=None, lost=0)
            self.debug("reset", slot, t)
        s["label"] = label
        if s["x"] is None:
            s["x"] = x_meas.copy()
            s["v"] = np.zeros_like(x_meas)
            s["last_t"] = t
            s["lost"] = 0
            self.debug("observe-init", slot, t)
            return
        dt = max(float(t - s["last_t"]), 1e-3)
        if dt > self.max_lost:
            # 长缺口后的观测：αβ 恒速外推 v·dt 已不可信（009 409 实测
            # 116 帧陈旧速度外推把新鲜观测冲掉 50% → 状态中毒 → 后续
            # 16 帧真 pair 全被几何门控拒绝、输出漂移幽灵）。重初始化，
            # 直接采信本次观测。
            s["x"] = x_meas.copy()
            s["v"] = np.zeros_like(x_meas)
            s["last_t"], s["lost"] = t, 0
            self.debug("observe-reinit", slot, t)
            return
        # αβ：预测 → 修正。NaN 点（该点本次无效）保持纯预测
        ok = np.isfinite(x_meas).all(axis=1)
        x_pred = s["x"] + s["v"] * dt
        x_new = np.where(ok[:, None],
                         self.alpha * x_meas + (1.0 - self.alpha) * x_pred,
                         x_pred)
        v_new = np.where(ok[:, None],
                         self.beta * (x_new - s["x"]) / dt
                         + (1.0 - self.beta) * s["v"],
                         s["v"])
        s["x"], s["v"], s["last_t"], s["lost"] = x_new, v_new, t, 0
        self.debug("observe", slot, t)

    def mark_lost(self, slot: int, t: int):
        """救援失败：计数丢失（超 max_lost 后 predict 返回 None，幻觉硬顶）。"""
        self.slots[slot]["lost"] += 1
        self.debug("mark-lost", slot, t)

    def predict(self, slot: int, t: int) -> np.ndarray | None:
        """恒速外推 x + v·(t − last_t)。从未见过 / 丢失超限 → None。"""
        s = self.slots[slot]
        if s["x"] is None or s["lost"] > self.max_lost:
            return None
        dt = max(float(t - s["last_t"]), 0.0)
        return s["x"] + s["v"] * dt

    def close(self):
        if self._dbg:
            self._dbg.close()
            self._dbg = None


def make_pseudo_pair(pred: np.ndarray, label: str) -> PseudoHandPair:
    """预测 3D → PseudoHandPair（err 全 inf：valid_count=0 让 _adopt 判据退化）。"""
    res = TriangulationResult(np.asarray(pred, np.float64).reshape(-1, 3),
                              np.full((21,), np.inf, np.float64))
    return PseudoHandPair(result=res, left_label=label)
