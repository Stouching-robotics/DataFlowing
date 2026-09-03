#!/usr/bin/env python3
"""
两阶段透视裁剪精修 —— Hur et al. 2025 (Eurographics) "Perspective Crop Based
Egocentric Hand Pose Estimation via Fisheye Stereo Vision" 的落地实现。

原理：鱼眼边缘手形畸变是 MediaPipe 全图检测误差的主要来源。粗三角化给出
3D 位置后，按 3D 投影在每只眼画面里取手部 ROI，放大成 256² 的"透视裁剪图"
（默认取自矫正图 = 已去鱼眼畸变），在裁剪图上重新检测 21 点（手更大、
畸变更小 → 更准），再逆变换回原始图像素做二次三角化。

stage-2 检测器：MediaPipe（逐 crop 检测，refine()）。refine_batch()
批量路径为 batch_capable 检测器保留（当前无启用后端）。批量检测器可
额外输出逐点置信度 → 置信度加权 blend（替代盲 0.5 平均）+ 低置信否决。

对极 y 对齐（epi_y_align，构造参数默认关、run_pipeline MP 路径已接线开）：
矫正后左右目同名关键点必共行，实测存在 ~+3px 系统性竖直视差（signed Δy
两检测器一致为正）——在 rect 空间强制 y=(y_l+y_r)/2 再转 raw，消除竖直
系统残差+随机噪声。

采纳判据：精修结果有效点数 ≥ 粗结果 且 平均重投影误差 < 粗结果，
否则保留粗结果（保证精修永不回退精度）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stereo_s80m.stereo_triangulate import (  # noqa: E402
    StereoTriangulator,
    DEFAULT_MAX_REPROJ_ERR,
    DEFAULT_MAX_DEPTH,
    MIN_VALID_POINTS,
)

MIN_CROP_BOX = 20      # 裁剪框最小边长（像素），小于直接放弃该侧
RESCUE_MAX_ERR = 3.0   # 伪 pair 救援采纳的绝对重投影误差上限（px）。
                       # 伪 pair 粗结果 valid=0/err=inf，"严格优于粗"判据必然退化
                       # （0≥0、有限<inf 恒真）→ 会把另一槽位已认领的手重复救援。
                       # 真救援（007 实测 err 0.31/0.41px）远低于此，垃圾救援被拦。
RESCUE_MAX_DIST = 0.15  # 伪 pair 救援采纳的位置门限（m）：精修 3D 质心离
                        # 预测质心 ≤150mm 才信。009 295 实测 crop 在左槽预测处
                        # 检到右手（几何一致、身份错误）离预测 190mm——互斥
                        # 门槛（两槽 3D 距离）拦不住同帧已认领手，位置门限拦。
                        # 007 真救援是死槽复活（无伪 pair），不受此门影响。


def _centroid3(points) -> np.ndarray | None:
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    ok = np.isfinite(pts).all(axis=1)
    if ok.sum() < MIN_VALID_POINTS:
        return None
    return np.median(pts[ok], axis=0)


def _rescue_near(res, coarse) -> bool:
    """伪救援位置门限：精修 3D 质心须在预测质心 RESCUE_MAX_DIST 内。"""
    c_res, c_pred = _centroid3(res.points_3d), _centroid3(coarse.points_3d)
    if c_res is None or c_pred is None:
        return False
    return bool(np.linalg.norm(c_res - c_pred) <= RESCUE_MAX_DIST)


@dataclass
class RefinedPair:
    """一只手的粗匹配 + 精修结果。result 为最终采纳结果（精修或粗）。"""

    pair: object                              # 原始 HandPair（粗匹配）
    kpts_l_raw: np.ndarray | None             # (21,2) 精修左目原始图像素（失败 None）
    kpts_r_raw: np.ndarray | None
    result: object                            # 最终采纳的 TriangulationResult
    used: bool                                # 精修是否被采纳
    reason: str                               # ok / no-crop-det / not-better / coarse-invalid
    conf_l: np.ndarray | None = None          # (21,) 精修左目逐点置信度（批量检测器才有）
    conf_r: np.ndarray | None = None

    @property
    def left_label(self) -> str:
        """兼容 render_stereo.overlay_view 的 HandPair 接口。"""
        return self.pair.left_label


class CropRefiner:
    """两阶段精修器。refine_det_l/r 为每目独立的裁剪图检测器
    （num_hands=1, smooth=False，与全图管线隔离，避免追踪先验串扰；
    批量路径同一实例供两眼，批量前向）。"""

    def __init__(self, tri: StereoTriangulator,
                 crop_size: int = 256, pad_ratio: float = 0.5,
                 crop_source: str = "rect",
                 max_err: float = DEFAULT_MAX_REPROJ_ERR,
                 max_depth: float = DEFAULT_MAX_DEPTH,
                 refine_det_l=None, refine_det_r=None,
                 epi_y_align: bool = False,
                 kpt_soft_thr: float = 0.15):
        self.tri = tri
        self.crop_size = crop_size
        self.pad_ratio = pad_ratio
        self.crop_source = crop_source          # "rect" = 矫正图裁剪（默认） / "raw" = 原始畸变图裁剪
        self.max_err = max_err
        self.max_depth = max_depth
        self.refine_det_l = refine_det_l
        self.refine_det_r = refine_det_r
        self.epi_y_align = epi_y_align          # 对极 y 对齐（仅 crop_source="rect" 生效）
        self.kpt_soft_thr = kpt_soft_thr        # 逐点软阈值：conf 低于此值的点 blend 权重归零

    # ── 像素逆变换：矫正图像素 → 原始(畸变)图像素 ──────────────
    # 算法与 stereo_triangulate._self_test 的 _rect_to_raw 相同：
    # 矫正像素 → inv(K_rect) 归一化矫正坐标 → R^T 回原始理想坐标 → 加畸变

    @staticmethod
    def rect_to_raw(px_rect: np.ndarray, tri: StereoTriangulator, side: str) -> np.ndarray:
        """矫正图像素 (N,2) → 原始图像素 (N,2)。"""
        px = np.asarray(px_rect, dtype=np.float64).reshape(-1, 2)
        n = px.shape[0]
        P = tri.P1 if side == "left" else tri.P2
        Rr = tri.R1 if side == "left" else tri.R2
        K, D = (tri.K1, tri.D1) if side == "left" else (tri.K2, tri.D2)
        hom = np.hstack([px, np.ones((n, 1))]).T            # 3×N 像素齐次
        ideal = np.linalg.solve(P[:3, :3], hom)             # 归一化矫正坐标
        orig_ideal = (Rr.T @ ideal)[:2].T                   # 原始相机理想坐标 (N,2)
        _distort = cv2.fisheye.distortPoints if tri.model == "equidistant" else cv2.distortPoints
        return _distort(orig_ideal.reshape(-1, 1, 2), K, D).reshape(-1, 2)

    # ── crop 提取（检测器无关）──────────────────────────────────

    def _make_crop(self, img: np.ndarray, proj_pts: np.ndarray):
        """按投影点取手部 ROI → pad 0.5 → 正方形原生分辨率 crop。

        返回 (crop, x1, y1) 或 None（有效点太少 / 框太小）。
        """
        H, W = img.shape[:2]
        ok = np.isfinite(proj_pts).all(axis=1)
        if ok.sum() < 4:
            return None

        x1 = max(0, int(proj_pts[ok, 0].min()))
        y1 = max(0, int(proj_pts[ok, 1].min()))
        x2 = min(W, int(np.ceil(proj_pts[ok, 0].max())))
        y2 = min(H, int(np.ceil(proj_pts[ok, 1].max())))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        # pad 后扩成正方形（对称补足 + clamp 回图像内）
        pad = int(self.pad_ratio * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
        s = min(max(x2 - x1, y2 - y1), W, H)
        if s < MIN_CROP_BOX:
            return None
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        x1 = min(max(int(round(cx - s / 2.0)), 0), W - s)
        y1 = min(max(int(round(cy - s / 2.0)), 0), H - s)
        x2, y2 = x1 + s, y1 + s

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop, x1, y1

    # ── 单侧精修（MediaPipe 路径）──────────────────────────────

    def _refine_side(self, img: np.ndarray, proj_pts: np.ndarray, det, side: str,
                     to_raw: bool = True):
        """在 img 里按投影点取手部 ROI 裁剪 → 检测 → 返回关键点。

        to_raw=True 返回原始图像素 (21,2)；False 返回该侧全图像素
        （crop_source="rect" 时即矫正图像素，供对极 y 对齐用）。
        """
        cc = self._make_crop(img, proj_pts)
        if cc is None:
            return None
        crop, x1, y1 = cc
        # 检测分辨率：裁剪原生边长（256~512 之间），保留全部像素信息。
        # MediaPipe 内部统一缩到 ~224²，源图越大 → 亚像素定位越准（放大效应）。
        s = crop.shape[1]
        detect_size = min(max(s, self.crop_size), self.crop_size * 2)
        scale = detect_size / float(crop.shape[1])
        crop_rs = cv2.resize(crop, (detect_size, detect_size),
                             interpolation=cv2.INTER_LINEAR)
        hands = det.detect(crop_rs)
        if not hands:
            # 多 landmarker 实例并发下偶发空检测（同一图重跑即命中），重试一次
            hands = det.detect(crop_rs)
        if not hands:
            return None

        kpts_crop = np.asarray(hands[0].landmarks, np.float64).reshape(-1, 2)
        kpts_full = kpts_crop / scale + np.array([x1, y1], np.float64)   # 回到该侧全图像素

        if self.crop_source == "rect":
            if to_raw:
                return self.rect_to_raw(kpts_full, self.tri, side)       # 原始图像素
            return kpts_full.astype(np.float32)                          # 矫正图像素（对齐用）
        return kpts_full.astype(np.float32)                              # 本就原始图像素

    # ── 候选生成 + 采纳判据（refine/refine_batch 共用）──────────

    def _adopt(self, pair, kpts_raw: dict, conf_raw: dict,
               coarse_l: np.ndarray = None, coarse_r: np.ndarray = None) -> RefinedPair:
        """候选（精修/加权 blend）三角化 → 严格优于粗结果才采纳。

        conf_raw: {side: (21,) 置信度}。有置信度时 blend 权重 w=0.25+0.5·c
        （高置信精修点主导、低置信回退粗检）；无置信度退化为 0.5/0.5 平均。
        """
        coarse = pair.result
        is_pseudo = getattr(pair, "l_idx", 0) < 0    # 传播重检伪 pair
        if kpts_raw.get("left") is None or kpts_raw.get("right") is None:
            reason = "propagated" if is_pseudo else "no-crop-det"
            return RefinedPair(pair, kpts_raw.get("left"), kpts_raw.get("right"),
                               coarse, False, reason,
                               conf_l=conf_raw.get("left"), conf_r=conf_raw.get("right"))

        # 候选：精修重三角化 /（可选）置信度加权粗精 2D 平均后重三角化
        candidates = []
        res2 = self.tri.triangulate(kpts_raw["left"], kpts_raw["right"],
                                    self.max_err, self.max_depth)
        candidates.append(("refined", res2))
        if coarse_l is not None and coarse_r is not None:
            wl = self._blend_w(conf_raw.get("left"))
            wr = self._blend_w(conf_raw.get("right"))
            kl_b = (wl * kpts_raw["left"].astype(np.float64) +
                    (1.0 - wl) * np.asarray(coarse_l, np.float64).reshape(-1, 2))
            kr_b = (wr * kpts_raw["right"].astype(np.float64) +
                    (1.0 - wr) * np.asarray(coarse_r, np.float64).reshape(-1, 2))
            resB = self.tri.triangulate(kl_b, kr_b, self.max_err, self.max_depth)
            candidates.append(("blend", resB))

        # 采纳判据：真 pair 严格优于粗结果（永不回退精度）；伪 pair 的粗结果
        # valid=0/err=inf 使该判据退化（0≥0、有限<inf 恒真）→ 改绝对质量门槛
        # （≥MIN_VALID_POINTS 且 err ≤ RESCUE_MAX_ERR），拦掉"把另一只已认领
        # 的手救援进空槽"的重复检测。候选间仍取误差最低者。
        best, best_err, reason = None, coarse.mean_error, "not-better"
        for tag, res in candidates:
            if is_pseudo:
                if res.valid_count >= MIN_VALID_POINTS and \
                        res.mean_error <= RESCUE_MAX_ERR and res.mean_error < best_err \
                        and _rescue_near(res, coarse):
                    best, best_err = res, res.mean_error
                    reason = f"ok-{tag}"
            elif res.valid_count >= coarse.valid_count and res.mean_error < best_err:
                best, best_err = res, res.mean_error
                reason = f"ok-{tag}"
        if best is None:
            if is_pseudo:
                reason = "propagated"
            return RefinedPair(pair, kpts_raw["left"], kpts_raw["right"],
                               coarse, False, reason,
                               conf_l=conf_raw.get("left"), conf_r=conf_raw.get("right"))
        return RefinedPair(pair, kpts_raw["left"], kpts_raw["right"],
                           best, True, reason,
                           conf_l=conf_raw.get("left"), conf_r=conf_raw.get("right"))

    def _blend_w(self, conf):
        """逐点 blend 权重：w = 0.25 + 0.5·c ∈ [0.25, 0.75]；无置信度 → 标量 0.5。

        kpt_soft_thr > 0 时 conf 低于阈值的点 w=0（该点完全回退 stage-1
        粗检）——批量检测器对遮挡点会给出低分热斑，盲目加权会引入错误位置。
        """
        if conf is None:
            return 0.5
        c = np.clip(np.asarray(conf, np.float64).reshape(-1, 1), 0.0, 1.0)
        w = 0.25 + 0.5 * c
        if self.kpt_soft_thr > 0:
            w[c < self.kpt_soft_thr] = 0.0
        return w

    # ── 整手精修（MediaPipe 逐 crop 路径，语义与旧版一致）──────

    def refine(self, pair, rect_l: np.ndarray, rect_r: np.ndarray,
               raw_l: np.ndarray = None, raw_r: np.ndarray = None,
               coarse_l: np.ndarray = None, coarse_r: np.ndarray = None) -> RefinedPair:
        """pair: HandPair(粗匹配)。rect_*: 矫正帧。raw_*: 原始帧（crop_source="raw" 时用）。

        coarse_l/coarse_r: 粗匹配的原始图像素关键点 (21,2)，用于 blend 候选——
        两次检测（全图/裁剪图）是准独立估计，2D 平均后再三角化可降噪声（√2 效应）。
        未提供时只做"精修 vs 粗"二选一。
        """
        coarse = pair.result
        pts3d = np.asarray(coarse.points_3d, dtype=np.float64).reshape(-1, 3)
        valid = np.isfinite(pts3d).all(axis=1)
        if valid.sum() < MIN_VALID_POINTS:
            return RefinedPair(pair, None, None, coarse, False, "coarse-invalid")

        # 对极 y 对齐（仅矫正图裁剪）：先收集矫正图像素，对齐后再转 raw
        align_rect = self.epi_y_align and self.crop_source == "rect"
        kpts_side = {}
        for side in ("left", "right"):
            det = self.refine_det_l if side == "left" else self.refine_det_r
            if det is None:
                continue
            proj_rect = self.tri.project(pts3d[valid], side)      # 矫正图像素（永远可算）
            if self.crop_source == "rect":
                kpts_side[side] = self._refine_side(rect_l if side == "left" else rect_r,
                                                    proj_rect, det, side, to_raw=not align_rect)
            else:                                                 # raw: 投影点逆变换回原始图
                proj_raw = self.rect_to_raw(proj_rect, self.tri, side)
                kpts_side[side] = self._refine_side(raw_l if side == "left" else raw_r,
                                                    proj_raw, det, side)
        kpts_raw = {}
        if align_rect:
            kl = kpts_side.get("left")
            kr = kpts_side.get("right")
            if kl is not None and kr is not None:
                ym = 0.5 * (kl[:, 1] + kr[:, 1])                  # 矫正后同名点必共行
                kl[:, 1] = ym
                kr[:, 1] = ym
            for side in ("left", "right"):
                if kpts_side.get(side) is not None:
                    kpts_raw[side] = self.rect_to_raw(kpts_side[side], self.tri, side)
        else:
            kpts_raw = kpts_side
        return self._adopt(pair, kpts_raw, {}, coarse_l=coarse_l, coarse_r=coarse_r)

    # ── 整帧批量精修（batch_capable 检测器；当前无启用后端）──────

    def refine_batch(self, pairs, rect_l: np.ndarray, rect_r: np.ndarray,
                     raw_l: np.ndarray = None, raw_r: np.ndarray = None,
                     coarse_l_src: list = None, coarse_r_src: list = None) -> list:
        """一帧内全部手的 stage-2 精修：crop 收集 → 单次 detect_batch → 分发。

        coarse_l_src/coarse_r_src: stage-1 检测结果列表（DetectedHand），
        按 pairs[i].l_idx/r_idx 取粗关键点用于 blend。
        检测器无 detect_batch 时逐手回退 refine()（MediaPipe 兜底）。
        """
        det = self.refine_det_l or self.refine_det_r
        if det is None or not hasattr(det, "detect_batch"):
            return [self.refine(
                        p, rect_l, rect_r, raw_l=raw_l, raw_r=raw_r,
                        coarse_l=(coarse_l_src[p.l_idx].landmarks
                                  if coarse_l_src is not None and p.l_idx >= 0 else None),
                        coarse_r=(coarse_r_src[p.r_idx].landmarks
                                  if coarse_r_src is not None and p.r_idx >= 0 else None))
                    for p in pairs]

        n = len(pairs)
        coarse_pts = []            # 每 pair 的 (coarse_l, coarse_r)
        crop_list, crop_jobs = [], []   # crop_jobs 与 crop_list 对齐: (pair_idx, side, x1, y1)
        for pi, p in enumerate(pairs):
            coarse = p.result
            pts3d = np.asarray(coarse.points_3d, np.float64).reshape(-1, 3)
            valid = np.isfinite(pts3d).all(axis=1)
            # 护栏：伪 pair（传播重检）l_idx/r_idx=-1，无 stage-1 粗 2D，
            # blend 禁用（coarse_pts 传 None）；负下标会静默取列表尾元素
            coarse_pts.append(((coarse_l_src[p.l_idx].landmarks
                                if coarse_l_src is not None and p.l_idx >= 0 else None),
                               (coarse_r_src[p.r_idx].landmarks
                                if coarse_r_src is not None and p.r_idx >= 0 else None)))
            if valid.sum() < MIN_VALID_POINTS:
                continue
            for side in ("left", "right"):
                proj_rect = self.tri.project(pts3d[valid], side)
                if self.crop_source == "rect":
                    cc = self._make_crop(rect_l if side == "left" else rect_r, proj_rect)
                else:
                    proj_raw = self.rect_to_raw(proj_rect, self.tri, side)
                    cc = self._make_crop(raw_l if side == "left" else raw_r, proj_raw)
                if cc is None:
                    continue
                crop, x1, y1 = cc
                crop_list.append(crop)
                crop_jobs.append((pi, side, x1, y1))

        # 单次批量检测（crop 为原生分辨率，检测器内部自行 resize 256²）
        hands_lists = det.detect_batch(crop_list) if crop_list else []

        # 分发回各手各目（先留在全图像素空间，对极对齐后再转 raw）
        kpts_side = [{} for _ in range(n)]
        conf_raw = [{} for _ in range(n)]
        for (pi, side, x1, y1), hands in zip(crop_jobs, hands_lists):
            if not hands:
                continue
            h = hands[0]
            kpts_crop = np.asarray(h.landmarks, np.float64).reshape(-1, 2)
            kpts_side[pi][side] = kpts_crop + np.array([x1, y1], np.float64)   # scale=1：原生分辨率
            conf = getattr(h, "conf", None)
            if conf is not None:
                conf_raw[pi][side] = np.asarray(conf, np.float32).reshape(-1)

        # 对极 y 对齐（仅矫正图裁剪）：矫正后同名点必共行，左右目 y 取平均，
        # 消除系统性竖直视差（实测 signed Δy ≈ +3px）与随机 y 噪声。
        align_rect = self.epi_y_align and self.crop_source == "rect"
        kpts_raw = [{} for _ in range(n)]
        for pi in range(n):
            kl = kpts_side[pi].get("left")
            kr = kpts_side[pi].get("right")
            if align_rect and kl is not None and kr is not None:
                ym = 0.5 * (kl[:, 1] + kr[:, 1])
                kl[:, 1] = ym
                kr[:, 1] = ym
            for side in ("left", "right"):
                if kpts_side[pi].get(side) is not None:
                    if self.crop_source == "rect":
                        kpts_raw[pi][side] = self.rect_to_raw(kpts_side[pi][side],
                                                              self.tri, side)
                    else:
                        kpts_raw[pi][side] = kpts_side[pi][side].astype(np.float32)

        # 逐手采纳判据（与 refine() 同一 _adopt 逻辑）
        out = []
        for pi, p in enumerate(pairs):
            cl, cr = coarse_pts[pi]
            out.append(self._adopt(p, kpts_raw[pi], conf_raw[pi], coarse_l=cl, coarse_r=cr))
        return out
