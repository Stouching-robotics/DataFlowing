#!/usr/bin/env python3
"""probe_align_overlay.py —— D435 深度↔RGB 对齐验收探针。

对抽样帧输出：
  - 伪彩深度 α0.4 叠 RGB 的 PNG（frame_XXX_align.png）
  - 深度 Canny 边缘（绿）叠 RGB 的 PNG（frame_XXX_edges.png）
并统计覆盖率 + 手部深度一致性 A/B + 边缘距离（信息）。

判据（验收）：
  1. aligned 深度在彩色框内覆盖率 ≥60%（下限哨兵，防外参大错时深度
     整体落出画面/覆盖塌方）。深度视锥 HFOV≈89° ⊃ 彩色≈70°，减去
     原始深度 5-15% 无效点后，填洞覆盖率物理上就是 ~85-95%——原计划
     的 90% 上限是在未做空穴回填（覆盖仅 ~19%）时的判据，回填落地后
     上限不再成立，已删（回填正确性由 depth_align 自测 4 单独保证：
     只填空穴、不腐蚀有效值）。
  2. 手部深度一致性 A/B（方向性判据）：对抽样帧检测到的每只手，21 个
     2D 关键点在 aligned 深度上采样，统计有效点深度的 p90−p10 散布
     （手厚 ~100mm，正确对齐全点落在手上）：
       正确外参 散布中位 ≤ 200mm，且 ≤ 反号外参散布中位 × 0.8
     实测 222_000011（手距 0.7-1.6m）：正确 ~40mm、反号 ~90mm（方向
     一致 2×）；反号把关键点平移 17-38px，部分边界点采到背景/桌面。
     注：本场景手距较远，任何深度↔RGB 比对都只能给出 2× 方向信号——
     外参的正确性最终由构造保证：设备出厂标定 + API 语义（P_color =
     R·P_depth + t，与 rs2.align 同源）+ 录制同机交叉核对 + 骨长
     验收（标定无关硬判据，probe_bone_lengths）。

边缘距离为什么只作信息不作判据（实测结论）：848×480→1280×720 是
2.12× 上采样；采集链路对深度无任何后处理（core/d435_camera.py 裸
数据）→ 物体轮廓深度天然侵蚀 2-5px（深度分辨率）= 4-10px（彩色分辨
率）；RGB 纹理边（海报/衣服）深度看不到，远距 2-5m 量化阶跃在平滑
墙面 RGB 无对应边。实测：物体级深度边→强 RGB 边 6-28px 帧间抖动、
互相关平移峰无一致方向、反号 A/B 边缘距离不判胜负——边缘类判据在
本场景噪声底之上，任何绝对阈值都无法区分正确标定与 30-60px 的反号
错位，故不作判据。

用法（venv）: python tools/hand_3d_d435/probes/probe_align_overlay.py <session>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from stereo_s80m.hand_3d import io                                   # noqa: E402
from stereo_s80m.hand_3d.detector import MediaPipeDetector           # noqa: E402
from hand_3d_d435.depth_align import (DepthAligner, load_calib,      # noqa: E402
                                      load_session_depth_intr,
                                      load_session_depth_files,
                                      load_depth_frame)
from hand_3d_d435.render_overlay import blend_depth                 # noqa: E402

VALID_RANGE = (300.0, 3000.0)     # 深度边缘/覆盖率统计的有效范围（mm）
NEAR_RANGE = (300.0, 1500.0)      # 近距物体级（信息用边缘）
NEAR_STEP = 150.0                 # 物体级阶跃阈值（mm，手 445 vs 背景 >1000）
SPREAD_OK_MAX = 200.0             # 判据 2：正确外参手内散布上限（mm）
SPREAD_FLIP_RATIO = 0.8           # 判据 2：正确散布 ≤ 反号散布 × 此系数


def _depth_edges(aligned: np.ndarray) -> np.ndarray:
    """aligned mm → 归一 uint8 → Canny 边缘 bool（信息用）。"""
    m = np.where((aligned > 0) & np.isfinite(aligned),
                 np.clip((aligned - VALID_RANGE[0])
                         / (VALID_RANGE[1] - VALID_RANGE[0]), 0.0, 1.0), 0.0)
    dn = (m * 255).astype(np.uint8)
    return cv2.Canny(dn, 30, 90) > 0


def _near_object_edges(aligned: np.ndarray) -> np.ndarray:
    """近距物体级深度边缘：相邻阶跃 >150mm 且两侧均 ∈[300,1500]mm（信息用）。"""
    a = aligned
    step_h = np.abs(a[:, 1:] - a[:, :-1]) > NEAR_STEP   # (h, w-1)
    step_v = np.abs(a[1:, :] - a[:-1, :]) > NEAR_STEP   # (h-1, w)
    e = np.zeros_like(a, bool)
    e[:, 1:] |= step_h
    e[:, :-1] |= step_h
    e[1:, :] |= step_v
    e[:-1, :] |= step_v
    near = (a > NEAR_RANGE[0]) & (a < NEAR_RANGE[1])
    return e & near & np.roll(near, 1, 1) & np.roll(near, -1, 1) \
        & np.roll(near, 1, 0) & np.roll(near, -1, 0)


def _edge_dist(src_edges: np.ndarray,
               dst_edges: np.ndarray) -> float:
    """src 边缘每个像素到 dst 边缘最近距离的中位（px）。"""
    if not src_edges.any() or not dst_edges.any():
        return float("nan")
    dt = cv2.distanceTransform((~dst_edges).astype(np.uint8),
                               cv2.DIST_L2, 3)
    d = dt[src_edges]
    return float(np.median(d))


def _best_shift(src_edges: np.ndarray, dst_edges: np.ndarray):
    """互相关平移扫描（信息用）：dst 平移后与 src 重叠计数最大位置。"""
    if not src_edges.any() or not dst_edges.any():
        return 0, 0, 0.0
    s = src_edges.astype(np.float32)
    d = dst_edges.astype(np.float32)
    best, bx, by = -1.0, 0, 0
    for dy in range(-15, 16):
        for dx in range(-40, 41):
            c = float((np.roll(d, (dy, dx), (0, 1)) * s).sum())
            if c > best:
                best, bx, by = c, dx, dy
    return bx, by, best


def _hand_spread(aligner: DepthAligner, aligned: np.ndarray,
                 hands: list) -> list:
    """每只手 21 点采样深度的 p90−p10 散布（mm）。有效点 <5 → None。"""
    out = []
    for h in hands:
        # sample_points 返回 mm，勿再缩放
        z = aligner.sample_points(aligned,
                                  np.asarray(h.landmarks, np.float32))
        z = z[np.isfinite(z)]
        if z.size >= 5:
            out.append(float(np.percentile(z, 90) - np.percentile(z, 10)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", help="录制会话目录")
    ap.add_argument("--calib", help="标定 JSON（默认 hand_3d_d435/calibration/）")
    ap.add_argument("--frames", default="0,100,200,300,400",
                    help="抽样帧号（逗号分隔）")
    ap.add_argument("--out", help="PNG 输出目录（默认 keypoints_output/<tag>/<sess>/probe_align/）")
    args = ap.parse_args()

    session = args.session.rstrip("/")
    video = io.find_video(session, "d435_rgb")
    if not video:
        sys.exit(f"错误: 找不到 RGB 视频 {session}/videos/d435_rgb/")
    depth_dir = os.path.join(session, "depth", "d435_depth")
    if not os.path.isdir(depth_dir):
        sys.exit(f"错误: 找不到深度目录 {depth_dir}")

    try:
        calib = load_calib(args.calib)
    except FileNotFoundError as e:
        sys.exit(f"错误: {e}")
    sd = load_session_depth_intr(session)
    if sd is None:
        print("警告: head_stereo.json 缺失，用固化标定深度内参")
        sd = calib["depth_intrinsics"]
    aligner = DepthAligner(calib["color_intrinsics"],
                           calib["depth_to_color"], sd)
    flip = dict(calib["depth_to_color"])
    flip["translation"] = [-x for x in flip["translation"]]
    aligner_flip = DepthAligner(calib["color_intrinsics"], flip, sd)

    detector = MediaPipeDetector(num_hands=2, delegate="cpu")

    depth_files = load_session_depth_files(depth_dir)

    tag = os.path.basename(os.path.dirname(session))
    out = args.out or os.path.join(_REPO_ROOT, "keypoints_output", tag,
                                   os.path.basename(session), "probe_align")
    os.makedirs(out, exist_ok=True)

    frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    cap = cv2.VideoCapture(video)
    covs, spreads_ok, spreads_flip = [], [], []
    near_d2rs, raw_d2rs, raw_r2ds, shifts = [], [], [], []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, rgb = cap.read()
        if not ok:
            print(f"  帧 {f}: 读取失败")
            continue
        dp = depth_files.get(f + 1)
        if dp is None:
            print(f"  帧 {f}: 无对应深度 {f + 1:06d}.bin")
            continue
        d = load_depth_frame(dp, (aligner.dh, aligner.dw))
        if d is None:
            print(f"  帧 {f}: 深度 bin 读取失败")
            continue
        aligned = aligner.align_depth_to_color(d)

        valid = (aligned > 0) & np.isfinite(aligned)
        in_range = valid & (aligned >= VALID_RANGE[0]) \
            & (aligned <= VALID_RANGE[1])
        cov = float(valid.mean())
        cov_r = float(in_range.mean())
        covs.append(cov)

        # 手部深度一致性 A/B（判据 2）
        hands = detector.detect(rgb)
        aligned_flip = aligner_flip.align_depth_to_color(d)
        s_ok = _hand_spread(aligner, aligned, hands)
        s_fl = _hand_spread(aligner_flip, aligned_flip, hands)
        spreads_ok += s_ok
        spreads_flip += s_fl

        # 边缘类（信息用）
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        r_edges = cv2.Canny(gray, 50, 150) > 0
        r_strong = cv2.Canny(gray, 100, 220) > 0
        d_raw = _depth_edges(aligned)
        d_near = _near_object_edges(aligned)
        near_d2rs.append(_edge_dist(d_near, r_strong))
        raw_d2rs.append(_edge_dist(d_raw, r_edges))
        raw_r2ds.append(_edge_dist(r_edges, d_raw))
        shifts.append(_best_shift(d_near, r_strong))

        comp = blend_depth(rgb, aligned, 0.4)
        comp_edges = comp.copy()
        comp_edges[d_raw] = (0.25 * comp_edges[d_raw]
                             + 0.75 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(os.path.join(out, f"frame_{f:03d}_align.png"), comp)
        cv2.imwrite(os.path.join(out, f"frame_{f:03d}_edges.png"), comp_edges)

        print(f"  帧 {f}: 覆盖率 {cov * 100:.1f}% "
              f"(300-3000mm {cov_r * 100:.1f}%) "
              f"手散布 正确 "
              f"{np.median(s_ok):.0f}mm/反号 {np.median(s_fl):.0f}mm "
              f"(n={len(s_ok)})")

    cap.release()

    cov_med = float(np.median(covs))
    ok_cov = cov_med >= 0.60
    if len(spreads_ok) >= 3:
        s_ok_med = float(np.median(spreads_ok))
        s_fl_med = float(np.median(spreads_flip))
        ok_spread = (s_ok_med <= SPREAD_OK_MAX
                     and s_ok_med <= SPREAD_FLIP_RATIO * s_fl_med)
    else:
        s_ok_med = s_fl_med = float("nan")
        ok_spread = False
        print("  警告: 抽样帧检测到手 <3 只，判据 2 无法评估")
    ok = ok_cov and ok_spread

    print(f"\n整体: 覆盖率中位 {cov_med * 100:.1f}% (≥60%) "
          f"{'PASS' if ok_cov else 'FAIL'} | "
          f"手部深度散布 正确 {s_ok_med:.0f}mm (≤{SPREAD_OK_MAX:.0f}mm "
          f"且 ≤0.8×反号 {s_fl_med:.0f}mm) "
          f"{'PASS' if ok_spread else 'FAIL'}")
    print(f"信息: 物体边→强RGB边 中位 {np.nanmedian(near_d2rs):.2f}px | "
          f"原始边 d→r 中位 {np.nanmedian(raw_d2rs):.2f}px "
          f"r→d 中位 {np.nanmedian(raw_r2ds):.2f}px | "
          f"平移峰 {[(s[0], s[1]) for s in shifts]}")
    print(f"对齐判据: {'PASS' if ok else 'FAIL'}")
    print(f"PNG 输出: {out}")


if __name__ == "__main__":
    main()
