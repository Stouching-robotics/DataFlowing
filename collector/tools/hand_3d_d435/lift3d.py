#!/usr/bin/env python3
"""lift3d.py —— 2D DetectedHand + aligned 深度 → 彩色相机系 3D（米）。

LiftResult / D435Pair 载体 mimic RefinedPair 接口（result.points_3d /
mean_error / valid_count + used + left_label），使 io.pack_3d / pack_errors /
pack_stage2 零改动可消费。mean_error 恒 NaN：单目无重投影误差概念，
真实质量信号（逐点 3×3 窗口有效数/MAD）走旁挂 metrics CSV。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_3d_d435.depth_align import DepthAligner


@dataclass
class LiftResult:
    """mimic TriangulationResult 的最小接口（io.pack_* 消费）。"""

    points_3d: np.ndarray                      # (21,3) 米，无效点 NaN
    mean_error: float = float("nan")           # 单目无重投影概念 → NaN
    valid_count: int = 0                       # 有效点数（槽位分配判可靠度用）


@dataclass
class D435Pair:
    """mimic RefinedPair：pack_3d/pack_errors/pack_stage2 直接可消费。"""

    result: LiftResult
    left_label: str = ""
    used: bool = False
    hand2d: np.ndarray | None = None           # (21,2) 原始 2D（2D 判据/叠显用）
    n_valid: int = 0                           # 深度采样有效点数（metrics）
    det: object = None                         # 原始 DetectedHand（打包 2D 列用）
    measured: np.ndarray | None = None         # (21,) bool：点 z 来自实测（False=补点）


# ── 深度带约束采样（防手缘点窗口混入背景 → z 翻面） ──────────────

BAND_HALF_M = 0.12       # 手深带半宽（米）：手厚 ~4cm + 倾斜余量，足够
BAND_MIN_VALID = 4       # 第一遍有效点不足此数 → 退回无约束（带不可靠）
GATE_M = 0.15            # 时序一致性门（米/帧）：|观测−预测| 超此值判可疑


def _lift_z(aligner: DepthAligner, aligned: np.ndarray,
            pts2d: np.ndarray, band: bool = True) -> tuple[np.ndarray, float | None]:
    """21 点深度采样（米）→ (带内测量 z, 手深中位 zc 或 None)。

    band=True 两遍：先无约束取手深中位 zc，再只从 [zc±BAND_HALF_M] 带内
    窗口像素取中位（背景像素剔除）。第一遍有效点 <BAND_MIN_VALID →
    zc=None，退回无约束单遍结果（假检测/稀疏深度时不可信）。"""
    z_mm = aligner.sample_points(aligned, pts2d) * 0.001
    zc = None
    if band:
        ok1 = np.isfinite(z_mm)
        if ok1.sum() >= BAND_MIN_VALID:
            zc = float(np.median(z_mm[ok1]))
            z_mm = aligner.sample_points(
                aligned, pts2d,
                band=((zc - BAND_HALF_M) * 1000.0,
                      (zc + BAND_HALF_M) * 1000.0)) * 0.001
    return z_mm, zc


def lift_hand(hand, aligner: DepthAligner, aligned: np.ndarray,
              band: bool = True, complete: bool = True) -> D435Pair:
    """DetectedHand + aligned 深度图 → D435Pair（3D 在彩色相机系，米）。

    band=True（默认）：深度带约束采样（见 _lift_z）——手缘点的 3×3 窗口
    混入背景深度时背景像素被带滤剔除，中位必落手上。

    complete=True（默认）：深度缺失点补到手深中位 zc、x,y 由 2D 关键点
    反投影——参考单目动捕"保持 z 不变、调 x,y 使投影与 2D 关键点一致"
    的经验（效果比插值好得多）；我们的 zc 是 D435 实测真值（非 solvePnP
    模型值），且补后投影与 2D 天然一致。腕点这类窗口全空洞的点（实测
    62% 帧缺失）从此不再缺。测量有效点 <BAND_MIN_VALID 的检测（含假
    检测）无 zc → 不补，保持 NaN（宁缺勿错）。valid_count 仍只计实测
    点数——槽位分配可靠度判据不变。
    """
    pts2d = np.asarray(hand.landmarks, np.float32).reshape(-1, 2)
    z_m, zc = _lift_z(aligner, aligned, pts2d, band=band)
    measured = np.isfinite(z_m)                     # z 来自实测（False=补点）
    n_valid = int(measured.sum())
    if complete and zc is not None:
        z_use = np.where(measured, z_m, zc)
    else:
        z_use = z_m
    ok = np.isfinite(z_use)
    xyz = np.full((21, 3), np.nan, np.float64)
    xyz[ok, 2] = z_use[ok]
    xyz[ok, 0] = (pts2d[ok, 0] - aligner.cx_c) * z_use[ok] / aligner.fx_c
    xyz[ok, 1] = (pts2d[ok, 1] - aligner.cy_c) * z_use[ok] / aligner.fy_c
    return D435Pair(result=LiftResult(xyz, float("nan"), n_valid),
                    left_label=hand.label, hand2d=pts2d, n_valid=n_valid,
                    det=hand, measured=measured)


def apply_slot_zc(pair: D435Pair, zc: float, aligner: DepthAligner) -> None:
    """M5：补点深度锚定到槽级稳定 zc（保持 z、调 x,y 反投影与 2D 一致）。

    原地修改 pair.result.points_3d。只作用于补点（measured=False）：
    实测点不动；zc 由调用方做时域稳定（逐帧独立中位是整手共模跳的
    最大来源）。无补点/无 hand2d 时零操作。
    """
    if pair.measured is None or pair.hand2d is None:
        return
    pts = np.asarray(pair.result.points_3d, np.float64).reshape(21, 3)
    comp = np.isfinite(pts).all(axis=1) & ~pair.measured
    if not comp.any():
        return
    u = pair.hand2d[comp, 0]
    v = pair.hand2d[comp, 1]
    pts[comp, 2] = zc
    pts[comp, 0] = (u - aligner.cx_c) * zc / aligner.fx_c
    pts[comp, 1] = (v - aligner.cy_c) * zc / aligner.fy_c
    pair.result.points_3d = pts


def gate_observations(pts: np.ndarray, pred: np.ndarray | None,
                      gate: float = GATE_M,
                      wholesale_frac: float = 0.6) -> tuple[np.ndarray, bool]:
    """时序一致性门：观测 3D 与槽位预测差 >gate 的点判可疑 → 置 NaN。

    交给 tracker.observe_slot 后，NaN 点走纯预测（track3d.py 已支持逐点
    NaN 保持预测）——可疑观测不写入状态，下一帧真实值可追回。翻面事件
    （手 0.35m ↔ 背景 1.4m）由此在写入前拦截；真手速 4.5m/s 以上的点
    会被滞后 1 帧（用预测过渡），可接受。pred 为 None（槽未见过）不门控。

    **整手级逃逸**：可疑点 ≥ 有限点的 wholesale_frac → 返回 (原样, True)。
    单点翻面只打 1-3 个点；整手级不匹配说明槽状态过时（实测：入场前假
    检测把状态钉在背景 1.6m，此后门控会把每帧真观测 0.43m 全判可疑 →
    死锁永不恢复）。调用方见 True 应触发槽位重置后采信原观测。
    """
    pts = np.asarray(pts, np.float64).reshape(21, 3).copy()
    if pred is None:
        return pts, False
    pred = np.asarray(pred, np.float64).reshape(21, 3)
    d = np.linalg.norm(pts - pred, axis=1)
    fin = np.isfinite(pts).all(axis=1)
    suspect = fin & (d > gate)
    if fin.sum() >= BAND_MIN_VALID and suspect.sum() >= wholesale_frac * fin.sum():
        return pts, True
    pts[suspect] = np.nan
    return pts, False
