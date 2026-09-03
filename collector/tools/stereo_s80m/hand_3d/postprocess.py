#!/usr/bin/env python3
"""
离线后处理 —— 间隙插值 + 零相位速度自适应平滑（"平衡"档）。

两阶段（都在主循环之后、parquet 落盘之前，只在 --video-smooth offline 跑）：

1. fill_gaps：≤max_gap 的缺手短间隙逐点逐轴插值（优先三次，上下文不足
   4 帧退二次/线性），**覆盖**传播外推值；置 propagated=True 标记非直接
   检测。长缺口不动（不幻觉）。
2. offline_smooth：逐槽位逐点逐轴按连续有效段分段 savgol(7,3) 零相位
   平滑 + 速度自适应混合 w=exp(-(v/v0)²)：静止（v≈0）全 SG（压 >8Hz
   抖动），快速动作（v≫30mm/s）保 raw（不损失 2-5Hz 手势动力学）。

平滑结果写入新列 observation.keypoints.hand_3d_smoothed（float32
[2,21,3]），hand_3d 列保持原始精修值（含间隙插值）语义不变。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

N_HANDS, N_KPTS = 2, 21

JUMP_THR = 0.010          # 位移突变门槛（m/帧=10mm/帧）：噪声 σ=1.2mm 下帧差 ~1.7mm，
                          # 10mm≈6σ 不会误触发；真实"静止→抓取"阶跃远超此值


# ── 渲染用伪 pair（第二遍渲染 pass 从 parquet 行重建）─────────

class ReplayPair:
    """把 parquet 行还原成 overlay_view 可消费的伪 HandPair
    （同 render_stereo._ReplayPair，避免 import 共享模块私有类）。"""

    def __init__(self, label: str, points_3d, mean_error, valid_count: int):
        self.left_label = label
        self.result = SimpleNamespace(points_3d=np.asarray(points_3d, np.float64),
                                      mean_error=mean_error,
                                      valid_count=int(valid_count))


# ── 间隙插值 ──────────────────────────────────────────────────

def _runs(mask: np.ndarray) -> list:
    """bool mask 的连续 False 段 → [(a, b)]（闭区间）。"""
    out, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


def fill_gaps(rows: list, max_gap: int = 15) -> int:
    """≤max_gap 的缺手短间隙逐点逐轴插值（原地修改 rows）。

    插值锚点 = 间隙两侧的**真实检测**帧（present 且非 propagated），
    优先取左右各 2 帧作三次插值（不足 4 帧退线性）。插值覆盖间隙内的
    传播外推值并置 propagated=True。返回填充帧-槽数。
    """
    if max_gap <= 0 or not rows:
        return 0
    n = len(rows)
    h3 = np.stack([np.asarray(r["observation.keypoints.hand_3d"], np.float32)
                   .reshape(N_HANDS, N_KPTS, 3) for r in rows])          # (N,2,21,3)
    prop = np.stack([np.asarray(r["observation.keypoints.propagated"], np.bool_)
                     for r in rows])                                     # (N,2)
    present = np.stack([(r["observation.keypoints.hand_0_present"],
                         r["observation.keypoints.hand_1_present"]) for r in rows])

    filled = 0
    for slot in range(N_HANDS):
        real = present[:, slot] & ~prop[:, slot]      # 真实检测帧（锚点候选）
        for a, b in _runs(real):
            if b - a + 1 > max_gap:
                continue                               # 长缺口不动（不幻觉）
            left, right = a - 1, b + 1
            if not (left >= 0 and right < n and real[left] and real[right]):
                continue                               # 无两侧锚点（冷启动等）
            idx = [i for i in (a - 2, a - 1, b + 1, b + 2)
                   if 0 <= i < n and real[i]]
            xs = np.arange(a, b + 1, dtype=np.float64)
            for k in range(N_KPTS):
                for ax in range(3):
                    ys = np.asarray([h3[i, slot, k, ax] for i in idx])
                    ok = np.isfinite(ys)
                    if ok.sum() < 2:
                        continue
                    ix, iy = np.asarray(idx)[ok], ys[ok]
                    # 三次（≥4 锚点）/二次（3）/线性（2）
                    deg = min(3, len(ix) - 1)
                    coef = np.polyfit(ix, iy, deg)
                    h3[a:b + 1, slot, k, ax] = np.polyval(coef, xs)
            # 回写：present=True、propagated=True（非直接检测）、label 沿用
            lbl = rows[left][f"observation.keypoints.hand_{slot}_label"]
            for i in range(a, b + 1):
                rows[i]["observation.keypoints.hand_3d"] = \
                    h3[i].reshape(-1).tolist()
                rows[i]["observation.keypoints.propagated"][slot] = True
                if slot == 0:
                    rows[i]["observation.keypoints.hand_0_present"] = True
                    rows[i]["observation.keypoints.hand_0_label"] = lbl
                else:
                    rows[i]["observation.keypoints.hand_1_present"] = True
                    rows[i]["observation.keypoints.hand_1_label"] = lbl
                filled += 1
    return filled


# ── 零相位速度自适应平滑 ──────────────────────────────────────

def offline_smooth(rows: list, sg_window: int = 7, sg_poly: int = 3,
                   v0: float = 0.08, fps: float = 25.0,
                   still_window: int = 21) -> np.ndarray:
    """逐槽位逐点逐轴：连续有效段 savgol 零相位 + 速度自适应混合。

    返回 (N,2,21,3) float32。w=exp(-(v_sg/v0)²)：**速度从 savgol 去噪输出
    测量**（旧版从含噪 raw 测：静止时噪声导数 46-55mm/s 越 v0 门限 → 权重≈0，
    越静越不滤——已修）。静止自适应长窗：v_sg<20mm/s 用 still_window(21)、
    20-80mm/s 线性混入基准窗、>80mm/s 基准窗（SG7 只压 >8Hz 抖动，
    不动 2-5Hz 手势动力学）。零相位无滞后；快段保 raw 保真。

    **跳变阻尼**：帧间位移 >JUMP_THR(10mm/帧) 的突变点（静止→抓取）两侧
    ±still_window/2 内强制保 raw——长窗零相位对阶跃有预振铃（合成测试
    静止段边界 105-115 帧 wobble 3.11mm、峰值 68mm），阻尼后边界无预振铃、
    内部（30-100 帧）wobble 1.19→0.37mm 不受影响。
    """
    from scipy.signal import savgol_filter

    n = len(rows)
    h3 = np.stack([np.asarray(r["observation.keypoints.hand_3d"], np.float32)
                   .reshape(N_HANDS, N_KPTS, 3) for r in rows])
    out = h3.copy()
    half = still_window // 2

    def _odd_window(base: int, max_len: int, poly: int) -> int:
        w = min(base, max_len)
        w = w - 1 if w % 2 == 0 else w        # 奇数窗口
        return w if w > poly else 0           # 太短不滤

    for slot in range(N_HANDS):
        for k in range(N_KPTS):
            for ax in range(3):
                x = h3[:, slot, k, ax]
                for a, b in _runs(~np.isfinite(x)):     # _runs 返回 mask 为 False 的段 = 有限值段
                    seg = x[a:b + 1]
                    w7 = _odd_window(sg_window, len(seg), sg_poly)
                    if w7 == 0:
                        continue
                    sg7 = savgol_filter(seg, w7, sg_poly, mode="interp")
                    w21 = _odd_window(still_window, len(seg), sg_poly)
                    if w21 > w7:
                        sg21 = savgol_filter(seg, w21, sg_poly, mode="interp")
                        # 静止度：v_sg<20mm/s 全用长窗、>80mm/s 全用基准窗，之间线性混合
                        vel = np.abs(np.gradient(sg21)) * float(fps)   # m/s
                        alpha = np.clip((0.08 - vel) / 0.06, 0.0, 1.0)
                        sg = alpha * sg21 + (1.0 - alpha) * sg7
                    else:
                        vel = np.abs(np.gradient(sg7)) * float(fps)    # 段太短只有基准窗
                        sg = sg7
                    wgt = np.exp(-(vel / v0) ** 2)

                    # 跳变阻尼：位移突变点 ±half 内保 raw（防长窗预振铃）
                    jd = np.abs(np.diff(seg))
                    jumps = np.where(jd > JUMP_THR)[0]
                    if jumps.size:
                        t = np.arange(len(seg))
                        near = np.zeros(len(seg), bool)
                        for j in jumps:
                            near |= (np.abs(t - j) <= half) | (np.abs(t - (j + 1)) <= half)
                        wgt = np.where(near, 0.0, wgt)

                    out[a:b + 1, slot, k, ax] = wgt * sg + (1.0 - wgt) * seg
    return out


def pairs_from_row(row: dict, key: str = "observation.keypoints.hand_3d_smoothed",
                   min_points: int = 4) -> list:
    """parquet 行 → [ReplayPair]（第二遍渲染 pass 用）。

    与 render_session_from_parquet 的过滤不同：传播/插值帧 err 为 NaN，
    但 3D 有限——按"有效点数 ≥ min_points"纳入渲染，保证间隙期骨架连续。
    """
    pts = np.asarray(row[key], np.float32).reshape(N_HANDS, N_KPTS, 3)
    errs = row["observation.keypoints.reprojection_error"]
    labels = [row["observation.keypoints.hand_0_label"],
              row["observation.keypoints.hand_1_label"]]
    presents = [row["observation.keypoints.hand_0_present"],
                row["observation.keypoints.hand_1_present"]]
    pairs = []
    for i in range(N_HANDS):
        if not presents[i]:
            continue
        valid = int(np.isfinite(pts[i]).all(axis=1).sum())
        if valid < min_points:
            continue
        e = errs[i]
        pairs.append(ReplayPair(labels[i], pts[i],
                                float(e) if np.isfinite(e) else float("inf"),
                                valid))
    return pairs
