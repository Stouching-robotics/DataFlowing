#!/usr/bin/env python3
"""fill_track.py —— 离线缺失帧槽的 tracker 前向+后向填充（替代 fill_gaps 短桥接）。

问题：fill_gaps 只桥 ≤max_gap 的短缺口（两侧都要真实锚点）。D435 实测
腕点 62% 帧无深度（桌沿深度阴影，7×7 窗口全空洞）、整手 21 点齐全帧
仅 3.3%——长缺口不桥 → 旋转渲染骨架支离破碎、有效点质心逐帧跳。

做法：用与实时版同源的 HandSlotTracker（αβ，逐点 NaN 保持纯预测）沿
时间轴前向跑一遍（真实观测 observe、缺失帧 predict），再反向跑一遍补
首观测之前的头部段；合并时真实观测优先、前向预测次之、后向预测兜底。
轨迹性质：恒速外推，比 fill_gaps 的线性插值更物理；门控与实时版一致
（gate_observations），翻面观测不入状态。fill_gaps 语义保留：
被填帧 present=True、propagated=True、label 取最近 present 帧。

不修改 stereo_s80m（fill_gaps 只被替换调用，文件不动）。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from stereo_s80m.hand_3d.track3d import HandSlotTracker          # noqa: E402
from hand_3d_d435.lift3d import gate_observations               # noqa: E402

N_HANDS, N_KPTS = 2, 21


def _forward_pass(h3: np.ndarray, present: np.ndarray,
                  prop: np.ndarray, labels: np.ndarray,
                  max_lost: int) -> np.ndarray:
    """前向 tracker 填充 → (N,2,21,3)：αβ 状态轨迹。

    真实观测帧存 observe 后的状态（NaN 点已由纯预测补上，与实时链路
    逐点语义一致）；缺失帧存 predict。返回的数组整列可当"该时刻 tracker
    状态"用：合并时观测优先、状态补洞。
    """
    n = len(h3)
    tracker = HandSlotTracker(max_lost=max_lost)
    out = np.full_like(h3, np.nan)
    lost = [0, 0]
    for i in range(n):
        for s in range(N_HANDS):
            if present[i, s] and not prop[i, s]:        # 真实检测
                pts, wholesale = gate_observations(h3[i, s],
                                                   tracker.predict(s, i))
                if wholesale:      # 状态过时 → 槽位重置后采信观测（同主循环）
                    tracker.observe_slot(s, "\x00reset",
                                         np.full((N_KPTS, 3), np.nan), i)
                tracker.observe_slot(s, labels[i, s] or "", pts, i)
                lost[s] = 0
                out[i, s] = tracker.predict(s, i)       # observe 后的状态（dt=0 → x）
            else:
                pred = tracker.predict(s, i)
                tracker.mark_lost(s, i)
                lost[s] += 1
                if pred is not None:
                    out[i, s] = pred
    return out


def tracker_fill(rows: list, max_lost: int = 15) -> int:
    """前向+后向 tracker 填充缺失帧槽（原地修改 rows，语义同 fill_gaps）。

    返回填充帧-槽数（absent → present 的数量）。
    """
    if not rows:
        return 0
    n = len(rows)
    h3 = np.stack([np.asarray(r["observation.keypoints.hand_3d"], np.float32)
                   .reshape(N_HANDS, N_KPTS, 3) for r in rows])
    present = np.stack([(r["observation.keypoints.hand_0_present"],
                         r["observation.keypoints.hand_1_present"])
                        for r in rows])
    prop = np.stack([np.asarray(r["observation.keypoints.propagated"], np.bool_)
                     for r in rows])
    labels = np.stack([(r["observation.keypoints.hand_0_label"],
                        r["observation.keypoints.hand_1_label"])
                       for r in rows])

    fwd = _forward_pass(h3, present, prop, labels, max_lost)
    bwd = _forward_pass(h3[::-1], present[::-1], prop[::-1], labels[::-1],
                        max_lost)[::-1]      # 反向：补首观测前的头部段

    filled = 0
    for i in range(n):
        row3d = h3[i].copy()                 # 整行 (2,21,3)：真实观测原位保留
        for s in range(N_HANDS):
            real = present[i, s] and not prop[i, s]
            # 观测优先、前向状态次之、后向兜底——逐点补洞（真实帧的 NaN
            # 点也补：观测在某点缺深度时用 tracker 状态，与实时链路一致；
            # 真实帧 fwd = observe 后状态，NaN 点已是纯预测值）
            miss = ~np.isfinite(h3[i, s]).all(axis=1)[:, None]
            fillv = np.where(
                np.isfinite(fwd[i, s]).all(axis=1)[:, None],
                fwd[i, s],
                np.where(np.isfinite(bwd[i, s]).all(axis=1)[:, None],
                         bwd[i, s], h3[i, s]))
            row3d[s] = np.where(miss, fillv, h3[i, s])
            if not present[i, s]:
                has = bool(np.isfinite(row3d[s]).all(axis=1).any())
                if not has:              # 从未见过的槽无预测 → 不幻觉
                    rows[i][f"observation.keypoints.hand_{s}_label"] = ""
                    if s == 0:
                        rows[i]["observation.keypoints.hand_0_present"] = False
                    else:
                        rows[i]["observation.keypoints.hand_1_present"] = False
                    rows[i]["observation.keypoints.propagated"][s] = False
                    continue
                # 取最近 present 帧的 label（左优先）
                lab = ""
                for j in range(i - 1, -1, -1):
                    if present[j, s]:
                        lab = labels[j, s]
                        break
                if not lab:
                    for j in range(i + 1, n):
                        if present[j, s]:
                            lab = labels[j, s]
                            break
                rows[i][f"observation.keypoints.hand_{s}_label"] = lab
                filled += 1
            rows[i]["observation.keypoints.propagated"][s] = \
                bool(prop[i, s]) or not real
            if s == 0:
                rows[i]["observation.keypoints.hand_0_present"] = True
            else:
                rows[i]["observation.keypoints.hand_1_present"] = True
        rows[i]["observation.keypoints.hand_3d"] = row3d.reshape(-1).tolist()
    return filled


if __name__ == "__main__":
    # 自测：构造 5 帧 2 槽，中段缺失 + 头部缺失，验证前向/后向/真实优先
    def _mk():
        r = []
        for i in range(6):
            r.append({
                "observation.keypoints.hand_3d": np.full(
                    (2, 21, 3), np.nan, np.float32).reshape(-1).tolist(),
                "observation.keypoints.propagated": [False, False],
                "observation.keypoints.hand_0_present": False,
                "observation.keypoints.hand_1_present": False,
                "observation.keypoints.hand_0_label": "",
                "observation.keypoints.hand_1_label": "",
            })
        return r
    rows = _mk()
    p = np.arange(21, dtype=np.float32).reshape(21, 1).repeat(3, 1) * 0.001
    for i, z in ((1, 0.5), (2, 0.6), (5, 0.9)):
        both = np.full((2, 21, 3), np.nan, np.float32)
        both[0] = p + [0, 0, z]
        rows[i]["observation.keypoints.hand_3d"] = \
            both.reshape(-1).tolist()
        rows[i]["observation.keypoints.hand_0_present"] = True
        rows[i]["observation.keypoints.hand_0_label"] = "Left"
    n_fill = tracker_fill(rows, max_lost=15)
    z = np.stack([np.asarray(r["observation.keypoints.hand_3d"])
                  .reshape(2, 21, 3) for r in rows])
    assert n_fill == 3, f"应填 3 帧-槽（slot0 f0/3/4），实 {n_fill}"
    assert np.isfinite(z[0, 0, 0, 2]), "后向应填头部 f0"       # 恒速外推值（≈0.62）
    assert 0.5 < z[3, 0, 0, 2] < 0.9 and 0.5 < z[4, 0, 0, 2] < 0.9, \
        "前向应填中段 f3/4"                                   # αβ 外推（≈0.55/0.56）
    assert np.isclose(z[1, 0, 0, 2], 0.5) and np.isclose(z[2, 0, 0, 2], 0.6), \
        "真实观测原位保留"
    assert rows[0]["observation.keypoints.hand_0_present"]
    assert rows[0]["observation.keypoints.propagated"][0]
    assert rows[0]["observation.keypoints.hand_0_label"] == "Left"
    assert not rows[0]["observation.keypoints.hand_1_present"], \
        "从未见过的槽不幻觉 present"
    assert not np.isfinite(z[0, 1]).any(), "未见过槽保持全 NaN"
    print("fill_track 自测通过")
