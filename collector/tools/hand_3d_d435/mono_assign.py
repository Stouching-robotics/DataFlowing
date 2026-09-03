#!/usr/bin/env python3
"""mono_assign.py —— 单目双手槽位分配（hand_0/hand_1 连续身份）。

仿 stereo run_pipeline._best_slot_for 的单目版决策层级（009 教训全部继承）：
  1. 冷启动：两槽从未见过（label 空且 predict None）→ 标签惯例
     Left→slot0 / Right→slot1，无标签按检测序号；
  2. 标签唯一命中存活槽 + 几何门（≤UNRELIABLE_GATE）：恰好一只手的 label
     唯一命中某槽且质心距预测在门限内 → 强制入槽（009：标签命中稳于噪声几何）；
  3. 贪心几何：剩余手入最近未占用存活槽（质心需 ≥MIN_VALID_PTS 有效点，
     <4 退 2D 判据——槽预测投影回图像比 2D 质心距离）；
  4. 互斥守卫：两槽双真实且腕距 <WRIST_MUTEX → 两种排列取总 cost 更小者
     （防交叠期交叉串槽，009 重复检测教训）；
  5. 复活：未分配手的 label 唯一命中死亡槽（predict None 且 label 相同）→ 复活；
  6. 兜底：丢弃该检测（主循环对空槽走 predict/mark_lost）。

独立自测（构造轨迹）：python tools/hand_3d_d435/mono_assign.py
"""

from __future__ import annotations

import atexit
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from stereo_s80m.hand_3d.track3d import HandSlotTracker  # noqa: E402

UNRELIABLE_GATE = 0.15      # 3D 质心距槽预测的门限（米，009 经验值）
WRIST_MUTEX = 0.10          # 互斥守卫：两槽腕距小于此（米）触发排列比较
SWAP_MARGIN = 0.005         # 互斥换位需严格优于当前 ≥5mm（防逐帧抖动）
MIN_VALID_PTS = 4           # 3D 质心判据最少有效点数

_DEBUG_PATH = os.environ.get("HAND3D_SLOT_DEBUG")
_DEBUG_LINES: list = []


def _centroid3(pair) -> np.ndarray | None:
    pts = np.asarray(pair.result.points_3d, np.float64).reshape(-1, 3)
    ok = np.isfinite(pts).all(axis=1)
    if ok.sum() < MIN_VALID_PTS:
        return None
    return np.median(pts[ok], axis=0)


def _wrist(pair) -> np.ndarray | None:
    w = np.asarray(pair.result.points_3d, np.float64).reshape(-1, 3)[0]
    return w if np.isfinite(w).all() else None


def _cost(pair, slot_pred, color_intr):
    """3D 质心距槽预测质心（米）；质心不可靠退 2D 判据（预测投影 vs 2D 质心）。

    注意：槽预测是 (21,3) 且带 NaN 洞，必须取预测自身有效点中位做质心，
    不能整阵相减（NaN 污染范数 → cost 恒 NaN → 一切分配被拒）。
    """
    if slot_pred is None:
        return np.inf
    pred = np.asarray(slot_pred, np.float64).reshape(-1, 3)
    pok = np.isfinite(pred).all(axis=1)
    if pok.sum() < MIN_VALID_PTS:
        return np.inf
    p3 = np.median(pred[pok], axis=0)
    if p3[2] <= 0:
        return np.inf
    c3 = _centroid3(pair)
    if c3 is not None:
        return float(np.linalg.norm(c3 - p3))
    # 2D 退路：预测质心投影回图像 vs 该手 2D 质心（像素距 × Z/fx → 米）
    u = color_intr[0] * p3[0] / p3[2] + color_intr[2]
    v = color_intr[1] * p3[1] / p3[2] + color_intr[3]
    pts2d = np.asarray(pair.hand2d, np.float64).reshape(-1, 2)
    ok2 = np.isfinite(pts2d).all(axis=1)
    if ok2.sum() < MIN_VALID_PTS:
        return np.inf
    c2 = np.median(pts2d[ok2], axis=0)
    dist_px = float(np.linalg.norm(c2 - [u, v]))
    return dist_px * float(p3[2]) / color_intr[0]      # 像素 → 米（Z/fx）


def _in_out(pair, out) -> bool:
    return any(o is pair for o in out)


def _lab_ok(slot: int, pair, tracker: HandSlotTracker) -> bool:
    """pair 入 slot 是否 label 兼容（槽无标签 / 手无标签 / 相同）。"""
    sl = tracker.slot_label(slot)
    return sl == "" or not pair.left_label or sl == pair.left_label


def assign_mono_slots(pairs, tracker: HandSlotTracker, n: int,
                      color_intr=(917.0, 917.0, 640.0, 360.0),
                      lost_counts=(0, 0), debug: bool = False) -> list:
    """pairs: list[D435Pair]（≤2，voter 已稳定 label）
    → [slot0_pair|None, slot1_pair|None]（None = 本帧该槽无真手）。

    lost_counts：主循环维护的各槽连续丢失帧数（无真手帧计数），用于
    "困境槽无门限救援"（规则 2b）——恒速外推预测在手离开期间会漂移，
    此时几何门不可靠，label 是唯一可靠信号（222_000011 f374 教训）。
    """
    out: list = [None, None]
    if not pairs:
        return out
    pred = [tracker.predict(s, n) for s in range(2)]
    labels = [p.left_label for p in pairs]

    # 1) 冷启动：标签惯例（Left→0/Right→1），无标签按序号
    if (tracker.slot_label(0) == "" and tracker.slot_label(1) == ""
            and pred[0] is None and pred[1] is None):
        assigned = set()
        for s in (0, 1):
            for i, lab in enumerate(labels):
                if i in assigned:
                    continue
                if lab in ("Left", "Right") and lab == ("Left" if s == 0
                                                        else "Right"):
                    out[s] = pairs[i]
                    assigned.add(i)
        for i in [j for j in range(len(pairs)) if j not in assigned]:
            for s in (0, 1):
                if out[s] is None:
                    out[s] = pairs[i]
                    break
        _dbg(n, labels, pred, out, "cold")
        return out

    # 2) 标签唯一命中存活槽 + 几何门（健康槽保留门限：009 教训防
    #    flicker 标签窃槽）
    for i, lab in enumerate(labels):
        if not lab or _in_out(pairs[i], out):
            continue
        match = [s for s in (0, 1)
                 if out[s] is None and tracker.slot_label(s) == lab
                 and pred[s] is not None]
        if len(match) == 1 and _cost(pairs[i], pred[match[0]],
                                     color_intr) <= UNRELIABLE_GATE:
            out[match[0]] = pairs[i]
            _dbg(n, labels, pred, out, f"label{i}({lab})->{match[0]}")

    # 2b) 标签唯一命中困境槽（上帧无真手）→ 不设几何门。
    #     恒速外推预测在手离开期间会漂移，几何判据不可靠；label 是唯一
    #     可靠信号（009 教训：标签命中稳于噪声几何）。此规则覆盖死亡槽
    #     label 复活（旧规则 5 的 revive 分支）。
    for i, lab in enumerate(labels):
        if not lab or _in_out(pairs[i], out):
            continue
        match = [s for s in (0, 1)
                 if out[s] is None and tracker.slot_label(s) == lab
                 and lost_counts[s] >= 1]
        if len(match) == 1:
            out[match[0]] = pairs[i]
            _dbg(n, labels, pred, out, f"label2b{i}({lab})->{match[0]}")

    # 3) 贪心几何：剩余手 → 最近未占用存活槽（门限内）。
    #    标签守卫：槽已有已知 label 且 ≠ 手 label → 不入。
    #    （222_000011 f374 教训：丢失期单手被贪心塞进 label 冲突的槽，
    #     槽标签被 observe_slot 覆盖翻转，之后真手回来无法回槽。）
    free_slots = [s for s in (0, 1) if out[s] is None and pred[s] is not None]
    rest = [i for i in range(len(pairs)) if not _in_out(pairs[i], out)]
    rest.sort(key=lambda i: min((_cost(pairs[i], pred[s], color_intr)
                                 for s in free_slots), default=np.inf))
    for i in rest:
        if not free_slots:
            break
        cand = [s for s in free_slots
                if labels[i] == "" or tracker.slot_label(s) == ""
                or tracker.slot_label(s) == labels[i]]
        if not cand:
            continue
        best = min(cand, key=lambda s: _cost(pairs[i], pred[s], color_intr))
        if _cost(pairs[i], pred[best], color_intr) <= UNRELIABLE_GATE:
            out[best] = pairs[i]
            free_slots.remove(best)
            _dbg(n, labels, pred, out, f"geom{i}->{best}")

    # 4) 互斥守卫：双真实且腕距 <WRIST_MUTEX → 两种排列取总 cost 更小者。
    #    标签一致性：换位不得破坏 label↔槽对应（除非换位前的排列本来就
    #    标签冲突——理论上规则 2/2b/3 已保证不会，此处兜底）。
    if all(out[s] is not None for s in (0, 1)):
        w0, w1 = _wrist(out[0]), _wrist(out[1])
        if w0 is not None and w1 is not None \
                and float(np.linalg.norm(w0 - w1)) < WRIST_MUTEX:
            cur = sum(_cost(out[s], pred[s], color_intr) for s in (0, 1))
            swp = sum(_cost(out[1 - s], pred[s], color_intr) for s in (0, 1))
            cur_ok = all(_lab_ok(s, out[s], tracker) for s in (0, 1))
            swp_ok = all(_lab_ok(s, out[1 - s], tracker) for s in (0, 1))
            if swp_ok and (not cur_ok or swp < cur - SWAP_MARGIN):
                out[0], out[1] = out[1], out[0]
                _dbg(n, labels, pred, out,
                     f"mutex-swap d={np.linalg.norm(w0 - w1) * 1000:.0f}mm")

    # 5) 未见槽冷启（label 复活已被 2b 覆盖）：
    #    - 有标签手：标签惯例 Left→0/Right→1（未见槽 slot_label 恒 ""）
    #    - 无标签手：唯一空死槽兜底
    for i, lab in enumerate(labels):
        if _in_out(pairs[i], out):
            continue
        free_dead = [s for s in (0, 1) if out[s] is None and pred[s] is None]
        if not free_dead:
            continue
        if lab in ("Left", "Right"):
            want = 0 if lab == "Left" else 1
            if out[want] is None and pred[want] is None:
                out[want] = pairs[i]
                _dbg(n, labels, pred, out, f"cold-slot{i}({lab})->{want}")
        elif not lab and len(free_dead) == 1:
            out[free_dead[0]] = pairs[i]                # 无标签单死槽
            _dbg(n, labels, pred, out, f"dead{i}(nolab)->{free_dead[0]}")

    _dbg(n, labels, pred, out, "done")
    return out


def _dbg(n, labels, pred, out, evt) -> None:
    if not _DEBUG_PATH:
        return
    ps = []
    for p in pred:
        if p is None:
            ps.append("-")
            continue
        a = np.asarray(p, np.float64).reshape(-1, 3)
        ok = np.isfinite(a).all(axis=1)
        m = np.median(a[ok], axis=0) if ok.any() else np.full(3, np.nan)
        ps.append(f"{m[0]:.3f},{m[1]:.3f},{m[2]:.3f}")
    os_ = ["-" if o is None else f"{o.left_label}" for o in out]
    _DEBUG_LINES.append(f"f{n} {evt} labels={labels} pred=[{';'.join(ps)}] "
                        f"out={os_}")


if _DEBUG_PATH:
    atexit.register(lambda: open(_DEBUG_PATH, "a").write(
        "\n".join(_DEBUG_LINES) + "\n"))


# ── 构造轨迹自测 ──────────────────────────────────────────────

def _mk_pair(x, y, z, label, n_valid=21) -> "object":
    from hand_3d_d435.lift3d import D435Pair, LiftResult
    pts = np.full((21, 3), np.nan, np.float64)
    pts[:, 0], pts[:, 1], pts[:, 2] = x, y, z
    h2 = np.column_stack([np.full(21, x * 1000.0), np.full(21, y * 1000.0)])
    return D435Pair(result=LiftResult(pts, float("nan"), n_valid),
                    left_label=label, hand2d=h2, n_valid=n_valid)


def _selftest():
    from hand_3d_d435.lift3d import D435Pair, LiftResult

    # 1) 冷启动：Left→0/Right→1，无标签按序号
    tr = HandSlotTracker()
    out = assign_mono_slots(
        [_mk_pair(-0.1, 0.0, 0.5, "Left"), _mk_pair(0.1, 0.0, 0.5, "Right")],
        tr, 0)
    assert out[0].left_label == "Left" and out[1].left_label == "Right", out
    for s, p in enumerate(out):
        tr.observe_slot(s, p.left_label, p.result.points_3d, 0)
    print("✓ selftest 1: 冷启动标签惯例")

    # 2) 标签唯一命中存活槽：Right 手几何上离 slot0 预测近，但 label=Right
    #    唯一命中 slot1 → 强制入 slot1（标签优先）
    out = assign_mono_slots([_mk_pair(0.11, 0.0, 0.5, "Right")], tr, 5)
    assert out[1] is not None and out[1].left_label == "Right", out
    assert out[0] is None, out
    print("✓ selftest 2: 标签唯一命中优先于几何")

    # 3) 贪心几何：同 label 双手 → 距离判据
    tr2 = HandSlotTracker()
    pL = _mk_pair(-0.1, 0.0, 0.5, "Left")
    pR = _mk_pair(0.1, 0.0, 0.5, "Right")
    o0 = assign_mono_slots([pL, pR], tr2, 0)
    for s, p in enumerate(o0):
        tr2.observe_slot(s, p.left_label, p.result.points_3d, 0)
    # 两手均无标签 → 冷启动按序号入槽（out=[pL, pR]），继续观察
    o5 = assign_mono_slots([_mk_pair(0.12, 0.0, 0.5, ""),
                            _mk_pair(-0.12, 0.0, 0.5, "")], tr2, 5)
    # 无标签走几何：离 slot0 预测（-0.1）近的 -0.12 手应入 slot0
    assert o5[0] is not None and abs(o5[0].result.points_3d[0, 0] + 0.12) < 1e-6, o5
    assert o5[1] is not None and abs(o5[1].result.points_3d[0, 0] - 0.12) < 1e-6, o5
    print("✓ selftest 3: 无标签手贪心几何分配")

    # 4) 互斥守卫：双真实腕距<100mm 且当前交叉串槽 → 换位
    tr3 = HandSlotTracker()
    pA = _mk_pair(-0.02, 0.0, 0.5, "")
    pB = _mk_pair(0.01, 0.0, 0.5, "")
    o0 = assign_mono_slots([pA, pB], tr3, 0)
    assert o0[0] is pA and o0[1] is pB, o0     # 冷启动按序号
    for s, p in enumerate(o0):
        tr3.observe_slot(s, p.left_label, p.result.points_3d, 0)
    # 下一帧：检测顺序颠倒（pB 先来），无标签 → 几何应把 pA 放回 slot0
    o5 = assign_mono_slots([pB, pA], tr3, 5)
    assert o5[0] is pA and o5[1] is pB, "互斥/几何未纠正顺序颠倒"
    print("✓ selftest 4: 交叠期几何分配纠正串槽")

    # 5) 复活：死亡槽 label 唯一命中 → 复活
    tr4 = HandSlotTracker()
    o0 = assign_mono_slots([_mk_pair(-0.1, 0.0, 0.5, "Left")], tr4, 0)
    tr4.observe_slot(0, "Left", o0[0].result.points_3d, 0)
    for t in range(1, 25):
        tr4.mark_lost(1, t)            # slot1 从未见过 → 始终死亡
    out = assign_mono_slots([_mk_pair(0.1, 0.0, 0.5, "Right")], tr4, 25)
    assert out[1] is not None and out[1].left_label == "Right", out
    print("✓ selftest 5: 死亡槽 label 复活")

    # 6) 困境槽无门限救援（2b）：slot1 上帧丢失（lost=3），右手回到
    #    0.4m 外（恒速外推漂移超 150mm 门限）→ 仍按 label 入 slot1
    tr5 = HandSlotTracker()
    o0 = assign_mono_slots([_mk_pair(-0.1, 0.0, 0.5, "Left"),
                            _mk_pair(0.1, 0.0, 0.5, "Right")], tr5, 0)
    for s, p in enumerate(o0):
        tr5.observe_slot(s, p.left_label, p.result.points_3d, 0)
    out = assign_mono_slots([_mk_pair(0.1, 0.0, 0.9, "Right")], tr5, 5,
                            lost_counts=(0, 3))
    assert out[1] is not None and out[1].left_label == "Right", out
    assert out[0] is None, "Left 槽不该被困境右手占用"
    print("✓ selftest 6: 困境槽无门限 label 救援")

    # 7) 贪心标签守卫：健康槽 label 冲突的手不得就近误配
    tr6 = HandSlotTracker()
    o0 = assign_mono_slots([_mk_pair(-0.1, 0.0, 0.5, "Left"),
                            _mk_pair(0.1, 0.0, 0.5, "Right")], tr6, 0)
    for s, p in enumerate(o0):
        tr6.observe_slot(s, p.left_label, p.result.points_3d, 0)
    # Right 手出现在 Left 槽预测附近（几何最近），slot1 门限外 → 应全拒
    out = assign_mono_slots([_mk_pair(-0.11, 0.0, 0.5, "Right")], tr6, 5)
    assert out[0] is None and out[1] is None, \
        f"label 冲突的手被贪心误配: {out}"
    print("✓ selftest 7: 贪心标签守卫拒绝冲突误配")

    print("全部自测通过")


if __name__ == "__main__":
    _selftest()
