#!/usr/bin/env python3
"""
S80M 双目鱼眼 3D 手部关键点检测 + 3D 渲染 Demo —— 独立模块（不动主程序）。

流程（Hur et al. 2025 两阶段管线落地）::

    左右目视频 → 每目独立 MediaPipe 2D 检测（stage-1，float 亚像素）
    → match_hands 跨目配对 → 鱼眼双目三角化（粗 3D）
    → 透视裁剪精修（stage-2：3D 投影 → 手 ROI 256² 裁剪图 → 重检测 → 二次三角化）
    → 3D 域平滑（默认离线零相位+第二遍渲染，--video-smooth causal=因果 One-Euro）
    → 3D 旋转视角渲染 + 矫正图并排叠加
    → H.264 视频 + hand_3d parquet（LeRobot 风格 schema + stage2/propagated/
      hand_3d_smoothed 列）

只读复用 stereo_s80m/（三角化/渲染）与 hand_detection/（检测/平滑），
不修改任何现有文件；全部输出落在 keypoints_output/<tag>/<session>/ 下。

用法::

    python stereo_s80m/hand_3d/run_pipeline.py <session_dir> [选项]

    --no-refine        关闭两阶段精修（等价单阶段基线，用于对比）
    --crop-size N      精修裁剪图边长 px（默认 256）
    --crop-source      精修裁剪来源 rect|raw（默认 rect=矫正图）
    --no-smooth3d      关闭 3D 时序平滑
    --no-video         不生成视频（只出数据）
    --render-2d        额外输出原图 2D 叠加视频
    --every N          每隔 N 帧处理一次（快速测试）
    --mp-delegate      stage-1 delegate cpu|gpu|auto（默认 auto：GPU 子进程
                       冒烟通过才切，失败回退 CPU）
    --propagate-max N  遮挡传播最大丢失帧数（默认 15，0=全关）
    --video-smooth     视频平滑 offline|causal（默认 offline：循环后零相位
                       平滑+第二遍渲染，静止抖动 <1mm）
    --video-encoder    视频编码 auto|nvenc|libx264|mp4v（默认 auto 逐级回退）
    --write-episode    同时写入 <session>/data/keypoints/ 并追加 meta
                       （默认只写 keypoints_output/，不动会话目录）
    --compare          输出逐帧指标 CSV（与已落盘的基线 parquet 对比）
    --max-err/--max-depth   三角化过滤阈值（默认 8px / 3m）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)  # stereo_s80m/hand_detection 已并入 tools/ 命名空间

from hand_detection.hand_pipeline_mediapipe import MediaPipeHandPipeline  # noqa: E402
from stereo_s80m.render_stereo import overlay_view, overlay_view_2d      # noqa: E402
from stereo_s80m.stereo_triangulate import (                              # noqa: E402
    StereoTriangulator,
    load_stereo_calibration,
    match_hands,
    DEFAULT_MAX_REPROJ_ERR,
    DEFAULT_MAX_DEPTH,
)
from stereo_s80m.hand_3d.detector import MediaPipeDetector               # noqa: E402
from stereo_s80m.hand_3d.perspective_crop import CropRefiner, RefinedPair  # noqa: E402
from stereo_s80m.hand_3d.smoother import Hand3DSmoother                 # noqa: E402
from stereo_s80m.hand_3d.track3d import HandSlotTracker, make_pseudo_pair  # noqa: E402
from stereo_s80m.hand_3d.identity import HandednessVoter                 # noqa: E402
from stereo_s80m.hand_3d.postprocess import fill_gaps, offline_smooth     # noqa: E402
from stereo_s80m.hand_3d.renderer_3d import RotatingSkeletonRenderer    # noqa: E402
from stereo_s80m.hand_3d.video_writer import create_video_sink          # noqa: E402
from stereo_s80m.hand_3d import io                                      # noqa: E402

MODEL_PATH = os.path.join(_REPO_ROOT, "tools", "models", "hand_landmarker.task")
RENDER_SIZE = (1280, 720)


# ── 左右目方向自检（stereo_swap_lr 隐患）────────────────────────

def _detect_orientation(det_l, det_r, vc_l, vc_r, tri_normal, tri_swapped) -> bool:
    """用前几帧判断 stereo_left.mp4 对应标定 cam0 还是 cam1（True=需 swap）。"""
    err_n, err_s = [], []
    for _ in range(10):
        ok_l, fl = vc_l.read()
        ok_r, fr = vc_r.read()
        if not (ok_l and ok_r):
            break
        hl = det_l.detect(fl)
        hr = det_r.detect(fr)
        if not hl or not hr:
            continue
        for lh in hl[:1]:
            for rh in hr[:1]:
                rn = tri_normal.triangulate(lh.landmarks, rh.landmarks)
                if rn.valid_count >= 8:
                    err_n.append(rn.mean_error)
                rs = tri_swapped.triangulate(lh.landmarks, rh.landmarks)
                if rs.valid_count >= 8:
                    err_s.append(rs.mean_error)
        if len(err_n) >= 2 and len(err_s) >= 2:
            break
    if not err_s:
        return False
    if not err_n:
        return True
    print(f"  [方向自检] 常规配对平均误差 {np.mean(err_n):.2f}px (n={len(err_n)}), "
          f"交换配对 {np.mean(err_s):.2f}px (n={len(err_s)})")
    return np.mean(err_s) + 0.5 < np.mean(err_n)


# ── 伪 pair 救援互斥 ─────────────────────────────────────────

def _rescue_too_close(pred: np.ndarray, pairs: list, thresh: float) -> bool:
    """伪 pair 预测 3D 与任一真实 pair 的有效点质心距离 < thresh → True（跳过救援）。

    遮挡时单侧掉检测 → 空槽预测位置恰在另一只已认领手附近（222_000009 实测
    腕距 2-50mm），救援会把同一只手重复塞进两个槽；此检查保证救援只在预测
    位置远离已认领手时进行（真救援场景两手远距，不受影响）。
    """
    if thresh <= 0:
        return False
    pred = np.asarray(pred, np.float64).reshape(-1, 3)
    okp = np.isfinite(pred).all(axis=1)
    if okp.sum() < 2:
        return False
    cp = np.median(pred[okp], axis=0)
    for p in pairs:
        pts = np.asarray(p.result.points_3d, np.float64).reshape(-1, 3)
        ok = np.isfinite(pts).all(axis=1)
        if ok.sum() < 2:
            continue
        if np.linalg.norm(np.median(pts[ok], axis=0) - cp) < thresh:
            return True
    return False


# ── 槽位规划事实转储（HAND3D_SLOT_DEBUG 指向文件；诊断 009 抓瓶翻转用）──
_SLOT_DBG_PATH = os.environ.get("HAND3D_SLOT_DEBUG")


def _slot_debug(line: str) -> None:
    if _SLOT_DBG_PATH:
        try:
            with open(_SLOT_DBG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


_UNRELIABLE_GATE = 0.15   # 粗 3D 与活轨迹预测的最小距离 >150mm → pair 不可靠
_AMBIGUITY_MARGIN = 0.04  # 两槽几何距离相差 ≤40mm 视为平手 → 标签优先裁决


def _best_slot_for(pair, tracker, n: int, exclude: set | None = None,
                   bias: list | None = None) -> int:
    """真 pair 的槽位归属（-1 = 不可靠，丢弃走双槽伪救援路径）。

    决策层级：
    (a) 几何：pair 粗 3D 质心按每槽 (coarse, refined) 偏移对校正进精空间，
        与 tracker 预测比距离；最小距离 ≤150mm → 该槽。粗精空间系统性
        偏差 ~80mm 会让原始距离失真（009 抓瓶 294 实测 d0=87<d1=112
        把右手误分 slot0），校正后 d1≈6mm。两槽距离相差 ≤40mm 视为
        平手，pair 标签唯一命中某槽轨迹标签时标签槽优先（009 296
        右手 pair d=[93,116] 险胜 stale 槽0，被标签裁决拉回槽1）。
    (b) 复活：无槽几何达标时，pair 标签与**死亡槽**（predict None）标签
        唯一匹配 → 该槽复活（009 409 左手重现时 slot0 轨迹已死，
        标签是唯一线索；几何达标槽不参与，防垃圾 pair 冒领活槽）。
    (c) 全无 → -1：跨手误匹配的垃圾三角化（009 170 的 (98,54,123)）
        丢给双槽伪救援，误匹配帧两手都能被真实救援。
    冷启动（双槽从未见过）→ 标签惯例 Left→0 / Right→1。
    exclude：已被前面 pair 认领的槽位（双 pair 贪心分配用）。
    """
    pts = np.asarray(pair.result.points_3d, np.float64).reshape(-1, 3)
    ok = np.isfinite(pts).all(axis=1)
    preds = [None if (exclude and slot in exclude) else tracker.predict(slot, n)
             for slot in range(2)]
    lab = pair.left_label
    cold = all(p is None for p in preds)
    d: list = []
    if ok.sum() >= 4:
        c = np.median(pts[ok], axis=0)
        for slot in range(2):
            pr = preds[slot]
            if pr is None:
                d.append(float("inf"))
                continue
            pr = np.asarray(pr, np.float64).reshape(-1, 3)
            okr = np.isfinite(pr).all(axis=1)
            if okr.sum() < 4:
                d.append(float("inf"))
                continue
            cc = c
            if bias is not None and bias[slot] is not None:
                c_med, r_med = bias[slot]
                cc = c - (c_med - r_med)          # 粗→精空间校正
            # 质心距（形状无关：tracker 状态偶有 20 点精修结果）
            dd = float(np.linalg.norm(np.median(pr[okr], axis=0) - cc))
            d.append(dd if np.isfinite(dd) else float("inf"))
        fin = [x for x in d if np.isfinite(x)]
        if fin:
            best = int(np.argmin(d))
            if d[best] <= _UNRELIABLE_GATE:
                # 歧义平手裁决（009 296 实测 d=[93,116]：右手 pair 离 stale
                # 槽0 更近 → 右手抢占槽0、左手身份整段丢槽 → 409 后级联
                # 失联）：两槽距离相差 ≤40mm 且 pair 标签唯一命中某槽轨迹
                # 标签 → 标签槽优先（轨迹久不更新时几何噪声大，标签是更
                # 稳的线索；294 d=[96,59] 亦然，argmin 本就指向标签槽）。
                if lab and len(fin) == 2 and \
                        abs(d[0] - d[1]) <= _AMBIGUITY_MARGIN:
                    lab_match = [slot for slot in range(2)
                                 if tracker.slot_label(slot) == lab]
                    if len(lab_match) == 1 and d[lab_match[0]] <= _UNRELIABLE_GATE:
                        _slot_debug(f"f{n} best amb-label({lab}) "
                                    f"d={[round(x, 3) for x in d]}->{lab_match[0]}")
                        return lab_match[0]
                _slot_debug(f"f{n} best c={c.round(3).tolist()} "
                            f"d={[round(x, 3) for x in d]}->{best}")
                return best
    # (b) 标签复活：死亡槽唯一匹配
    if lab:
        dead_match = [slot for slot in range(2)
                      if preds[slot] is None
                      and tracker.slot_label(slot) == lab]
        if len(dead_match) == 1:
            _slot_debug(f"f{n} best revive-slot{dead_match[0]}({lab})")
            return dead_match[0]
    if cold:
        want = 0 if lab == "Left" else 1
        if exclude and want in exclude:
            want = 1 - want
        _slot_debug(f"f{n} best cold-label({lab})->{want}")
        return want
    _slot_debug(f"f{n} best drop({lab}) "
                f"d={[round(x, 3) for x in d]}")
    return -1


# 空槽占位（槽位对齐的 refine 列表里 None 槽 → io 打包 NaN / 渲染 absent）
_ABSENT_REFINED = RefinedPair(
    type("_AbsentPair", (), {"left_label": ""})(),
    None, None,
    type("_AbsentResult", (), {"points_3d": np.full((21, 3), np.nan, np.float64),
                               "valid_count": 0, "mean_error": float("inf")})(),
    False, "absent")


# ── 帧间稳定性指标（3D 域，对比平滑前后）───────────────────────

class _DispTracker:
    """按槽位记录相邻帧 3D 位移中位与一阶差分（抖动）。"""

    def __init__(self):
        self.prev = {}
        self.prev_disp = {}
        self.disp, self.jitter = [], []

    def add(self, slot, pts3d):
        ok = np.isfinite(pts3d).all(axis=1)
        if ok.sum() < 8:
            self.prev.pop(slot, None)
            self.prev_disp.pop(slot, None)
            return
        cur = np.full_like(pts3d, np.nan)          # NaN 填充保持形状一致
        cur[ok] = pts3d[ok]
        if slot in self.prev:
            both = ok & np.isfinite(self.prev[slot]).all(axis=1)
            if both.sum() >= 8:                    # 只在两帧都有效的点上量位移
                d = np.median(np.linalg.norm(cur[both] - self.prev[slot][both], axis=1))
                self.disp.append(d)
                if slot in self.prev_disp:
                    self.jitter.append(abs(d - self.prev_disp[slot]))
                self.prev_disp[slot] = d
        self.prev[slot] = cur


# ── 第二遍渲染 pass（--video-smooth offline）────────────────────

def _render_offline_pass(vp_l, vp_r, rows, tri, renderer, out_dir,
                         out_names, fps, w, h, args) -> dict:
    """重读视频逐帧渲染：rows 已是 fill_gaps + offline_smooth 后的最终数据。

    传播/插值帧 err 为 NaN 但 3D 有限——按有效点数纳入渲染，保证间隙期
    骨架连续（与 render_session_from_parquet 的 err 过滤不同）。
    """
    from stereo_s80m.hand_3d.postprocess import pairs_from_row

    writers = {
        "rot": create_video_sink(os.path.join(out_dir, out_names["rot"]),
                                 fps, *RENDER_SIZE, encoder=args.video_encoder),
        "rect": create_video_sink(os.path.join(out_dir, out_names["rect"]),
                                  fps, w * 2, h, encoder=args.video_encoder),
    }
    if args.render_2d:
        writers["2d"] = create_video_sink(
            os.path.join(out_dir, out_names["2d"]), fps, w * 2, h,
            encoder=args.video_encoder)

    vc_l, vc_r = cv2.VideoCapture(vp_l), cv2.VideoCapture(vp_r)
    n_total = len(rows)
    try:
        k_src = 0
        for row in rows:
            # frame_index 即原视频帧号（主循环 n 每消费一帧 +1，含 grab 跳过的帧）
            target = row["frame_index"]
            while k_src < target:
                if not (vc_l.grab() and vc_r.grab()):
                    raise RuntimeError("视频提前结束")
                k_src += 1
            ok_l, fl = vc_l.read()
            ok_r, fr = vc_r.read()
            k_src += 1
            if not (ok_l and ok_r):
                raise RuntimeError("视频提前结束")

            fi = row["frame_index"]
            hands3d = np.asarray(row["observation.keypoints.hand_3d_smoothed"],
                                 np.float32).reshape(2, 21, 3)
            labels = [row["observation.keypoints.hand_0_label"],
                      row["observation.keypoints.hand_1_label"]]
            errs = [float(e) if np.isfinite(e) else np.nan
                    for e in row["observation.keypoints.reprojection_error"]]
            if "rot" in writers:
                writers["rot"].write(renderer.render(hands3d, labels, errs, fi, n_total))
            if "rect" in writers:
                rect_l = tri.rectified_image(fl, "left")
                rect_r = tri.rectified_image(fr, "right")
                pairs = pairs_from_row(row)
                overlay_view(rect_l, pairs, tri, "left", fi, n_total)
                overlay_view(rect_r, pairs, tri, "right", fi, n_total)
                writers["rect"].write(cv2.hconcat([rect_l, rect_r]))
            if "2d" in writers:
                frame_l2, frame_r2 = fl.copy(), fr.copy()
                overlay_view_2d(frame_l2, np.asarray(
                    row["observation.keypoints.stereo_left"], np.float32).reshape(2, 21, 2),
                    "left", fi, n_total)
                overlay_view_2d(frame_r2, np.asarray(
                    row["observation.keypoints.stereo_right"], np.float32).reshape(2, 21, 2),
                    "right", fi, n_total)
                writers["2d"].write(cv2.hconcat([frame_l2, frame_r2]))
    finally:
        vc_l.release()
        vc_r.release()

    videos = {}
    for key, sink in writers.items():
        path = sink.close()
        if path:
            videos[key] = path
        else:
            print(f"  [警告] 视频 [{key}] 编码失败已跳过: {out_names[key]}")
    return videos


# ── 主流程 ─────────────────────────────────────────────────────

def run_session(session_dir: str, args: argparse.Namespace) -> dict:
    session = os.path.abspath(session_dir)
    vp_l, vp_r = io.find_video(session, "stereo_left"), io.find_video(session, "stereo_right")
    if not vp_l or not vp_r:
        raise FileNotFoundError(f"找不到双目视频: stereo_left={vp_l} stereo_right={vp_r}")

    calib = load_stereo_calibration(session, args.calib)
    if calib is None:
        raise RuntimeError("无可用的立体标定（查找链: episode calibration/ → "
                           "config/s80m_stereo_calibration.json）")
    print(f"标定: baseline={calib['baseline']*1000:.1f}mm "
          f"模型={calib['left_camera']['distortion_model']}")

    episode_index, task_index = io.load_episode_meta(session)
    timestamps = io.load_timestamps(session)

    vc_l, vc_r = cv2.VideoCapture(vp_l), cv2.VideoCapture(vp_r)
    n_total = min(int(vc_l.get(cv2.CAP_PROP_FRAME_COUNT)),
                  int(vc_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    w, h = int(vc_l.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vc_l.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"视频: {n_total} 帧  {w}x{h}@{fps:.0f}fps")

    # ── stage-1 检测器：每目独立实例（共享会污染 VIDEO 模式追踪先验）──
    mp_delegate = args.mp_delegate
    if mp_delegate == "auto":
        # GPU delegate 初始化可能 SIGSEGV → 子进程冒烟通过才切（mp_gpu.py）
        from stereo_s80m.hand_3d.mp_gpu import smoke_test_gpu
        mp_delegate = "gpu" if smoke_test_gpu(MODEL_PATH) else "cpu"
    print(f"  [stage-1] MediaPipe delegate: {mp_delegate}")
    det_l = MediaPipeDetector(model_path=MODEL_PATH, num_hands=2, mirror=False,
                              smooth=True, freq_min=5.0, beta=0.05,
                              det_conf=args.det_conf, track_conf=args.track_conf,
                              delegate=mp_delegate)
    det_r = MediaPipeDetector(model_path=MODEL_PATH, num_hands=2, mirror=False,
                              smooth=True, freq_min=5.0, beta=0.05,
                              det_conf=args.det_conf, track_conf=args.track_conf,
                              delegate=mp_delegate)

    # 手性投票（每目独立）：压掉 MediaPipe handedness 逐帧闪烁
    # （label 翻转 → match_hands 槽位排序/tracker 换手重置/骨架身份对穿）
    voter_l = HandednessVoter()
    voter_r = HandednessVoter()

    # ── 方向自检 ─────────────────────────────────────────────
    tri_normal = StereoTriangulator(calib)
    tri_swapped = StereoTriangulator(calib, swap_cams=True)
    use_swap = _detect_orientation(det_l, det_r, vc_l, vc_r, tri_normal, tri_swapped)
    print(f"  [方向自检] {'swap 交换配对' if use_swap else '常规配对'}")
    vc_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vc_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tri = tri_swapped if use_swap else tri_normal
    print(f"三角化器: {tri.summarize()}")

    # ── stage-2 精修器（独立检测实例，num_hands=1, smooth=False）──
    refiner = None
    if not args.no_refine:
        det_l2 = MediaPipeDetector(model_path=MODEL_PATH, num_hands=1, mirror=False,
                                   smooth=False)
        det_r2 = MediaPipeDetector(model_path=MODEL_PATH, num_hands=1, mirror=False,
                                   smooth=False)
        refiner = CropRefiner(tri, crop_size=args.crop_size, pad_ratio=args.crop_pad,
                              crop_source=args.crop_source, max_err=args.max_err,
                              max_depth=args.max_depth,
                              refine_det_l=det_l2, refine_det_r=det_r2,
                              # 对极 y 对齐原仅在 mmpose 分支接线（mmpose 已移除）；
                              # 系统 Δy 是几何性质、两检测器实测一致 → 接线到 MP
                              epi_y_align=True)

    smoother = None if args.no_smooth3d else Hand3DSmoother(args.freq_min, args.beta)

    # ── 输出 ─────────────────────────────────────────────────
    tag, name = os.path.basename(os.path.dirname(session)), os.path.basename(session)
    out_dir = args.out_dir or os.path.join(_REPO_ROOT, "keypoints_output", tag, name)
    os.makedirs(out_dir, exist_ok=True)

    # ── 遮挡传播 tracker（需 stage-2 精修器；propagate-max 0 = 全关）──
    tracker = None
    if refiner is not None and args.propagate_max > 0:
        tracker = HandSlotTracker(
            max_lost=args.propagate_max,
            debug_log=os.path.join(out_dir, "track_events.csv") if args.track_debug else None)

    writers = {}
    # causal 模式：主循环内逐帧写视频（现行为）
    # offline 模式（默认）：主循环只积累 rows，循环后离线平滑 + 第二遍渲染 pass
    if not args.no_video and args.video_smooth == "causal":
        # 管道写器（nvenc/libx264/mp4v 逐级回退，见 video_writer.py）：
        # 三个视频各自独立 ffmpeg 进程，编码移出主线程
        writers["rot"] = create_video_sink(
            os.path.join(out_dir, "hand_3d_rotating.mp4"), fps, *RENDER_SIZE,
            encoder=args.video_encoder)
        writers["rect"] = create_video_sink(
            os.path.join(out_dir, "stereo_triangulate_refined.mp4"), fps, w * 2, h,
            encoder=args.video_encoder)
        if args.render_2d:
            writers["2d"] = create_video_sink(
                os.path.join(out_dir, "stereo_2d_refined.mp4"), fps, w * 2, h,
                encoder=args.video_encoder)
    renderer = RotatingSkeletonRenderer(*RENDER_SIZE, revolutions=2.0)

    # det 双线程（仅 CPU delegate）：MediaPipe 推理释放 GIL，两目并行实测
    # ~1.9×；GPU delegate 双线程反而 0.84×（GL 上下文竞争）→ gpu 顺序
    det_parallel = {"auto": mp_delegate == "cpu",
                    "on": True, "off": False}[args.det_parallel]
    det_pool = ThreadPoolExecutor(max_workers=2) if det_parallel else None

    # ── 主循环 ───────────────────────────────────────────────
    rows = []
    reasons = Counter()
    n_pairs_frames = n_two = n_used = 0
    errs_coarse, errs_final = [], []
    confs = []          # 采纳手的 stage-2 逐点置信度（MediaPipe 恒无；列保留兼容）
    t_det = t_match = t_refine = t_smooth = t_render = 0.0
    disp_raw = _DispTracker()
    disp_sm = _DispTracker()
    slot_bias: list = [None, None]   # 每槽上次观测的 (粗质心, 精质心)，供
                                     # _best_slot_for 把 pair 粗 3D 校正进精空间
    t0 = time.time()
    n = 0

    while True:
        if args.every > 1 and n % args.every != 0:
            # 跳过帧只 grab 不解码（--every 提速）
            if not (vc_l.grab() and vc_r.grab()):
                break
            n += 1
            continue
        ok_l, fl = vc_l.read()
        ok_r, fr = vc_r.read()
        if not (ok_l and ok_r):
            break

        t = time.perf_counter()
        if det_pool is not None:
            fut_l = det_pool.submit(det_l.detect, fl)
            fut_r = det_pool.submit(det_r.detect, fr)
            hl = fut_l.result()
            hr = fut_r.result()
        else:
            hl = det_l.detect(fl)
            hr = det_r.detect(fr)
        t_det += time.perf_counter() - t

        # 手性投票（原地覆盖 label，match_hands/ tracker/平滑器全部自动受益）
        voter_l.update(hl, frame_w=w, frame_h=h, frame=n, cam="L")
        voter_r.update(hr, frame_w=w, frame_h=h, frame=n, cam="R")

        t = time.perf_counter()
        pairs = match_hands(hl, hr, tri, args.max_err, args.max_depth)
        t_match += time.perf_counter() - t

        rect_l = tri.rectified_image(fl, "left")
        rect_r = tri.rectified_image(fr, "right")

        # ── 槽位规划：真 pair 按轨迹连续归属槽位；缺手槽位生成伪 pair ──
        # 修复：pair 不再按列表顺序占槽——009 抓瓶期右目丢左手后，唯一
        # pair(右手) 若进 slot0 则槽位交换 + label 翻转锁死 114 帧；双手
        # 同 label（左目闪烁）时列表顺序同样会串槽（170 实测双 Right pair）。
        # 双 pair 也走轨迹最近贪心分配（先到先得，exclude 防同槽）。
        n_real = len(pairs)
        slot_pairs = [None, None]          # 槽位 → 真 pair（None=缺）
        if tracker is not None:
            taken: set = set()
            for p in pairs[:2]:
                s = _best_slot_for(p, tracker, n, exclude=taken, bias=slot_bias)
                if s < 0:
                    continue            # 不可靠 pair 丢弃（双槽伪救援接管）
                if s in taken:          # 防御（决策已含 exclude）
                    s = 1 - s
                slot_pairs[s] = p
                taken.add(s)
        else:
            slot_pairs = list(pairs[:2])
        # 事实转储：pairs 组成 + tracker 槽位状态（诊断用，正常关闭）
        if _SLOT_DBG_PATH:
            parts = []
            for pi, p in enumerate(pairs[:2]):
                pp = np.asarray(p.result.points_3d, np.float64).reshape(-1, 3)
                okk = np.isfinite(pp).all(axis=1)
                wrist3d = np.median(pp[okk], axis=0).round(3).tolist() \
                    if okk.sum() >= 4 else None
                rlab = hr[p.r_idx].label if 0 <= p.r_idx < len(hr) else "?"
                parts.append(f"p{pi}:L{p.l_idx}({p.left_label})/R{p.r_idx}({rlab})"
                             f" ok={int(okk.sum())} w={wrist3d}")
            tks = []
            for slot in range(2):
                st = tracker.slots[slot]
                pred = tracker.predict(slot, n)
                b = slot_bias[slot]
                bstr = "-" if b is None else (
                    f"{np.asarray(b[0]).round(3).tolist()}/"
                    f"{np.asarray(b[1]).round(3).tolist()}")
                tks.append(f"tk{slot}:{st['label'] or '-'} lost={st['lost']} "
                           f"x={st['x'][0].round(3).tolist() if st['x'] is not None else None}"
                           f" pw={pred[0].round(3).tolist() if pred is not None else None}"
                           f" bias={bstr}")
            _slot_debug(f"f{n};n={n_real};" + ";".join(parts) + ";" + ";".join(tks))

        refine_input, real_flags = [], []
        if tracker is not None:
            for slot in range(2):
                if slot_pairs[slot] is not None:
                    refine_input.append(slot_pairs[slot])
                    real_flags.append(True)
                else:
                    pred = tracker.predict(slot, n)
                    if pred is None:
                        refine_input.append(None)
                        real_flags.append(False)
                        continue
                    # 互斥：预测位置与已认领手重叠 → 跳过救援（宁缺毋滥，
                    # 防"一只手进两个槽"；--rescue-min-dist 0 关闭此检查）
                    if _rescue_too_close(pred,
                                         [p for p in slot_pairs if p is not None],
                                         args.rescue_min_dist):
                        tracker.debug("excluded", slot, n)
                        refine_input.append(None)
                        real_flags.append(False)
                        continue
                    refine_input.append(make_pseudo_pair(pred, tracker.slot_label(slot)))
                    real_flags.append(False)
                    tracker.debug("pseudo", slot, n)
        else:
            refine_input = list(pairs)
            real_flags = [True] * len(refine_input)

        # ── refine：refined[i] 与槽位 i 对齐（空槽 None）──
        refined = [None] * len(refine_input)
        live = [(i, p) for i, p in enumerate(refine_input) if p is not None]
        if live and refiner is not None:
            t = time.perf_counter()
            outs = [refiner.refine(
                        p, rect_l, rect_r, raw_l=fl, raw_r=fr,
                        coarse_l=(hl[p.l_idx].landmarks if p.l_idx >= 0 else None),
                        coarse_r=(hr[p.r_idx].landmarks if p.r_idx >= 0 else None))
                    for _, p in live]
            t_refine += time.perf_counter() - t
            for (i, _), o in zip(live, outs):
                refined[i] = o
        elif live:
            for i, p in live:
                refined[i] = RefinedPair(p, None, None, p.result, False, "refine-off")

        # ── tracker 回写：真 pair / 救援成功 = 真检测更新；失败 = mark_lost ──
        if tracker is not None:
            for i, p in enumerate(refined):
                if p is None:
                    continue
                if real_flags[i] or p.used:
                    pts = p.result.points_3d
                    if real_flags[i] and not p.used and slot_bias[i] is not None:
                        # 精修失败回退粗结果：按槽位粗→精偏移校正进精空间再
                        # 入 tracker——状态混存粗/精两套坐标会互相污染
                        # （009 349 起 17 帧 drop 循环：偏移对拿被粗观测
                        # 污染的 tracker 状态当"精参照"，校正越纠越偏）。
                        c_med, r_med = slot_bias[i]
                        pts = np.asarray(pts, np.float64) - (c_med - r_med)
                    tracker.observe_slot(i, p.left_label, pts, n)
                else:
                    tracker.mark_lost(i, n)

        # ── 粗→精偏移对维护（_best_slot_for 校正用）。只在"真 pair 且精修
        #    成功"帧更新：失败帧的粗结果已按旧偏移校正入 tracker（见回写段），
        #    伪救援帧的 pair.result=预测 3D 非粗观测——两者作参照都会污染
        #    偏移对（009 349 起 drop 循环的另一半根因）。──
        if tracker is not None:
            for i in range(2):
                p = refined[i] if i < len(refined) else None
                if p is None or not p.used or not real_flags[i]:
                    continue
                pp = np.asarray(p.pair.result.points_3d, np.float64).reshape(-1, 3)
                okk = np.isfinite(pp).all(axis=1)
                if okk.sum() < 4:
                    continue
                rp = np.asarray(p.result.points_3d, np.float64).reshape(-1, 3)
                okr = np.isfinite(rp).all(axis=1)
                if okr.sum() < 4:
                    continue
                slot_bias[i] = (np.median(pp[okk], axis=0),
                                np.median(rp[okr], axis=0))

        # 指标（伪 pair 传播帧不计入误差统计：err=inf 非测量值）
        if pairs:
            n_pairs_frames += 1
            if len(pairs) >= 2:
                n_two += 1
        for i, p in enumerate(refined):
            if p is None:
                continue
            if real_flags[i]:
                errs_coarse.append(p.pair.result.mean_error)
            if p.result.valid_count:
                errs_final.append(p.result.mean_error)
            if p.used:
                n_used += 1
                for c in (p.conf_l, p.conf_r):
                    if c is not None:
                        confs.append(np.asarray(c, np.float32))
            reasons[p.reason] += 1

        # 3D 平滑（渲染用；parquet 存精修原始值）
        refined_io = [p if p is not None else _ABSENT_REFINED for p in refined[:2]]
        hands3d_raw = np.asarray(io.pack_3d(refined_io), np.float32).reshape(2, 21, 3)
        labels = [p.left_label if p is not None else "" for p in refined[:2]]
        labels += [""] * (2 - len(labels))
        valids = [p.result.valid_count if p is not None else 0 for p in refined[:2]]
        valids += [0] * (2 - len(valids))
        t = time.perf_counter()
        hands3d_sm = smoother.update(hands3d_raw, labels, valids) if smoother else hands3d_raw
        t_smooth += time.perf_counter() - t
        # 传播帧 valid=0 不喂滤波器（无平滑状态）→ 渲染回退用预测值
        hands3d_sm = np.where(np.isnan(hands3d_sm), hands3d_raw, hands3d_sm)

        for slot in range(2):
            if valids[slot] >= 8:
                disp_raw.add(slot, hands3d_raw[slot])
                disp_sm.add(slot, hands3d_sm[slot])

        # parquet 行（propagated 列：该槽位是预测传播值而非直接检测）
        present0 = len(refined) > 0 and refined[0] is not None
        present1 = len(refined) > 1 and refined[1] is not None
        propagated = [False if p is None else (not real_flags[i] and not p.used)
                      for i, p in enumerate(refined[:2])]
        propagated += [False] * (2 - len(propagated))
        rows.append({
            "episode_index": episode_index,
            "frame_index": n,
            "timestamp": np.float32(timestamps.get(n, 0.0)),
            "task_index": task_index,
            "observation.keypoints.stereo_left": io.pack_2d(hl),
            "observation.keypoints.stereo_right": io.pack_2d(hr),
            "observation.keypoints.hand_3d": io.pack_3d(refined_io),
            "observation.keypoints.reprojection_error": io.pack_errors(refined_io),
            "observation.keypoints.hand_0_present": present0,
            "observation.keypoints.hand_1_present": present1,
            "observation.keypoints.hand_0_label": refined[0].left_label if present0 else "",
            "observation.keypoints.hand_1_label": refined[1].left_label if present1 else "",
            "observation.keypoints.stage2": io.pack_stage2(refined_io),
            "observation.keypoints.propagated": propagated,
            "action": [0.0],
        })

        # 渲染
        t = time.perf_counter()
        if "rot" in writers:
            errs = [(p.result.mean_error if p.result.valid_count else np.nan)
                    if p is not None else np.nan for p in refined[:2]]
            errs += [np.nan] * (2 - len(errs))
            writers["rot"].write(renderer.render(hands3d_sm, labels, errs, n, n_total))
        if "rect" in writers:
            overlay_view(rect_l, refined_io, tri, "left", n, n_total)
            overlay_view(rect_r, refined_io, tri, "right", n, n_total)
            writers["rect"].write(cv2.hconcat([rect_l, rect_r]))
        if "2d" in writers:
            frame_l2, frame_r2 = fl.copy(), fr.copy()
            overlay_view_2d(frame_l2, np.asarray(io.pack_2d(hl), np.float32).reshape(2, 21, 2),
                            "left", n, n_total)
            overlay_view_2d(frame_r2, np.asarray(io.pack_2d(hr), np.float32).reshape(2, 21, 2),
                            "right", n, n_total)
            writers["2d"].write(cv2.hconcat([frame_l2, frame_r2]))
        t_render += time.perf_counter() - t

        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"  {n}/{n_total} 帧  ({n/el:.1f} fps)  pairs={n_pairs_frames}  "
                  f"stage2采纳={n_used}", flush=True)

    vc_l.release()
    vc_r.release()
    det_l.close()
    det_r.close()
    if det_pool is not None:
        det_pool.shutdown(wait=True)
    if tracker is not None:
        tracker.close()
    if refiner is not None:
        refiner.refine_det_l.close()
        refiner.refine_det_r.close()

    if not rows:
        raise RuntimeError("未处理任何帧")

    # ── 收尾 ─────────────────────────────────────────────────
    out_names = {"rot": "hand_3d_rotating.mp4",
                 "rect": "stereo_triangulate_refined.mp4",
                 "2d": "stereo_2d_refined.mp4"}

    # 离线后处理（--video-smooth offline，默认）：间隙插值 + 零相位速度自适应平滑，
    # 写入 hand_3d_smoothed 新列；hand_3d 列保持原始精修值语义不变
    t_offline = 0.0
    n_filled = 0
    disp_off = _DispTracker()
    if args.video_smooth == "offline":
        t0 = time.perf_counter()
        n_filled = fill_gaps(rows, max_gap=args.propagate_max)
        smoothed = offline_smooth(rows, sg_window=args.sg_window,
                                  sg_poly=3, v0=args.sg_v0, fps=fps)
        t_offline = time.perf_counter() - t0
        for i, r in enumerate(rows):
            r["observation.keypoints.hand_3d_smoothed"] = \
                smoothed[i].reshape(-1).tolist()
            for slot in range(2):
                disp_off.add(slot, smoothed[i, slot])
        print(f"  离线后处理: 插值 {n_filled} 帧-槽  平滑 SG{args.sg_window} "
              f"v0={args.sg_v0*1000:.0f}mm/s  耗时 {t_offline:.1f}s")
    elif args.write_smoothed:
        # causal 模式 + --write-smoothed：列存在但全 NaN（无离线平滑）
        nan_col = [float("nan")] * (2 * 21 * 3)
        for r in rows:
            r["observation.keypoints.hand_3d_smoothed"] = nan_col

    # parquet 先于视频写盘：视频渲染失败不影响数据
    drop_keys = ()
    if not (args.video_smooth == "offline" or args.write_smoothed):
        drop_keys = ("observation.keypoints.hand_3d_smoothed",)
    parquet_path = None
    if not args.no_parquet:
        parquet_path = io.write_parquet(
            rows, os.path.join(out_dir, "hand_3d_refined", "chunk-000.parquet"),
            drop_keys=drop_keys)
    if args.write_episode:
        ep_path = io.write_parquet(rows, os.path.join(
            session, "data", "keypoints", "chunk-0000", "chunk_000000.parquet"),
            drop_keys=drop_keys)
        io.merge_info_json(session, drop_keys=drop_keys)
        io.merge_stats_json(session, rows)
        print(f"  ✓ episode parquet: {ep_path}")

    # 视频：causal = 主循环逐帧已写；offline = 第二遍渲染 pass（重读视频）
    videos = {}
    t_render_offline = 0.0
    if not args.no_video:
        if args.video_smooth == "offline":
            t0 = time.perf_counter()
            try:
                videos = _render_offline_pass(vp_l, vp_r, rows, tri, renderer,
                                              out_dir, out_names, fps, w, h, args)
            except Exception as e:
                print(f"  [警告] 离线渲染失败: {e}（可用 --video-smooth causal 重跑生成视频）")
            t_render_offline = time.perf_counter() - t0
        else:
            for key, sink in writers.items():
                path = sink.close()
                if path:
                    videos[key] = path
                else:
                    print(f"  [警告] 视频 [{key}] 编码失败已跳过: {out_names[key]}")

    stats = {
        "n": len(rows), "n_pairs_frames": n_pairs_frames, "n_two": n_two,
        "n_used": n_used, "reasons": reasons,
        "errs_coarse": np.array(errs_coarse), "errs_final": np.array(errs_final),
        "stage2_conf": np.concatenate(confs) if confs else np.zeros(0, np.float32),
        "stage2_vetoed": (getattr(refiner.refine_det_l, "n_vetoed", 0)
                          if refiner is not None else 0),
        "t_det": t_det, "t_match": t_match, "t_refine": t_refine,
        "t_smooth": t_smooth, "t_render": t_render,
        "t_render_offline": t_render_offline, "n_filled": n_filled,
        "disp_raw": disp_raw, "disp_sm": disp_sm, "disp_off": disp_off,
        "out_dir": out_dir, "videos": videos, "parquet_path": parquet_path,
        "rows": rows, "session": session,
    }
    return stats


# ── 汇总打印 + 指标 CSV ────────────────────────────────────────

def _summarize(stats: dict):
    n = stats["n"]
    print("\n── 3D 管线汇总 ──")
    print(f"  帧数: {n}   匹配帧: {stats['n_pairs_frames']} "
          f"({stats['n_pairs_frames']/n*100:.1f}%)   双手帧: {stats['n_two']} "
          f"({stats['n_two']/n*100:.1f}%)")
    if stats["reasons"]:
        total = sum(stats["reasons"].values())
        # 传播帧（伪 pair 救援失败）不是测量，不计入采纳率分母
        n_prop = stats["reasons"].get("propagated", 0)
        total_real = total - n_prop
        used_pct = stats["n_used"] / total_real * 100 if total_real else 0
        print(f"  stage-2 精修: 采纳 {stats['n_used']}/{total_real} ({used_pct:.1f}%)  "
              f"传播兜底 {n_prop}  原因分布 {dict(stats['reasons'])}")
    if stats["stage2_conf"].size:
        c = stats["stage2_conf"]
        print(f"  stage-2 逐点置信度(采纳): mean={c.mean():.3f}  "
              f"p10={np.percentile(c, 10):.3f}")
    if stats.get("stage2_vetoed"):
        print(f"  stage-2 低置信否决: {stats['stage2_vetoed']} crop")
    for tag, arr in (("精修前(粗)", stats["errs_coarse"]),
                     ("精修后(采纳)", stats["errs_final"])):
        if arr.size:
            print(f"  {tag} 重投影误差: mean={arr.mean():.2f}px  "
                  f"p95={np.percentile(arr, 95):.2f}px")
    for tag, tr in (("精修原始 3D", stats["disp_raw"]),
                    ("因果平滑 3D", stats["disp_sm"]),
                    ("离线平滑 3D", stats["disp_off"])):
        if tr.disp:
            d = np.median(tr.disp) * 1000
            j = np.median(tr.jitter) * 1000 if tr.jitter else float("nan")
            print(f"  {tag}: 帧间位移中位={d:.2f}mm  抖动中位={j:.2f}mm")
    ms = stats["n"]
    print(f"  耗时: det={stats['t_det']/ms*1000:.1f}ms/帧  "
          f"match+tri={stats['t_match']/ms*1000:.1f}ms/帧  "
          f"refine={stats['t_refine']/ms*1000:.1f}ms/帧  "
          f"smooth={stats['t_smooth']/ms*1000:.1f}ms/帧  "
          f"render={stats['t_render']/ms*1000:.1f}ms/帧")
    if stats.get("t_render_offline"):
        print(f"  离线第二遍渲染: {stats['t_render_offline']:.1f}s  "
              f"插值 {stats.get('n_filled', 0)} 帧-槽")
    zs = np.array([np.array(r["observation.keypoints.hand_3d"], np.float32)[2::3]
                   for r in stats["rows"]])
    zs = zs[np.isfinite(zs)]
    if zs.size:
        print(f"  物理 3D 深度: z ∈ [{zs.min():.3f}, {zs.max():.3f}] m")
    for k, v in stats["videos"].items():
        print(f"  ✓ 视频 [{k}]: {v}")
    if stats["parquet_path"]:
        print(f"  ✓ parquet: {stats['parquet_path']}")


def _write_metrics_csv(stats: dict, path: str):
    """逐帧指标 CSV（含基线 parquet 对比，若存在）。"""
    session = stats["session"]
    base_path = os.path.join(session, "data", "keypoints", "chunk-0000",
                             "chunk_000000.parquet")
    base = {}
    if os.path.isfile(base_path):
        try:
            import pyarrow.parquet as pq
            for r in pq.read_table(base_path).to_pylist():
                base[r["frame_index"]] = r
        except Exception:
            pass

    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        hdr = ["frame_index", "n_pairs", "err_ours", "valid_ours", "stage2",
               "err_base", "valid_base"]
        wr.writerow(hdr)
        for r in stats["rows"]:
            fi = r["frame_index"]
            err = r["observation.keypoints.reprojection_error"]
            ours = [e for e in err if np.isfinite(e)]
            valid_ours = len(ours)
            base_err = base.get(fi, {}).get("observation.keypoints.reprojection_error", [])
            b = [e for e in base_err if np.isfinite(e)]
            wr.writerow([fi, 1 if r["observation.keypoints.hand_0_present"] else 0,
                         f"{np.mean(ours):.2f}" if ours else "",
                         valid_ours,
                         int(any(r["observation.keypoints.stage2"])),
                         f"{np.mean(b):.2f}" if b else "",
                         len(b)])
    print(f"  ✓ 指标 CSV: {path}")


# ── CLI ────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", help="episode 目录，如 data/recordings/222/222_000008")
    ap.add_argument("--calib", default=None, help="指定标定 JSON（默认走查找链）")
    ap.add_argument("--max-err", type=float, default=DEFAULT_MAX_REPROJ_ERR)
    ap.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH)
    ap.add_argument("--no-refine", action="store_true", help="关闭两阶段精修（单阶段基线）")
    ap.add_argument("--propagate-max", type=int, default=15,
                    help="遮挡传播最大丢失帧数：缺手槽位用预测 3D 做 crop 重检，"
                         "失败则速度外推并标 propagated 列（0=关；长缺口不幻觉）")
    ap.add_argument("--rescue-min-dist", type=float, default=0.10,
                    help="伪 pair 救援互斥距离（米）：预测位置与已认领手的 3D 质心"
                         "距离小于此值则跳过救援（防一只手进两个槽；0=关闭）")
    ap.add_argument("--track-debug", action="store_true",
                    help="输出 tracker 事件 CSV（track_events.csv）")
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--crop-pad", type=float, default=0.5)
    ap.add_argument("--crop-source", default="rect", choices=["rect", "raw"])
    ap.add_argument("--no-smooth3d", action="store_true")
    ap.add_argument("--freq-min", type=float, default=3.0, help="3D One-Euro 最低截止频率 Hz")
    ap.add_argument("--beta", type=float, default=0.3, help="3D One-Euro 速度系数")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--render-2d", action="store_true", help="额外输出原图 2D 叠加视频")
    ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--write-episode", action="store_true",
                    help="同时写 <session>/data/keypoints/ 并追加 meta（默认不动会话目录）")
    ap.add_argument("--compare", action="store_true", help="输出逐帧指标 CSV")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--det-parallel", default="auto", choices=["auto", "on", "off"],
                    help="两目 stage-1 检测并行（CPU delegate 实测 ~1.9×；"
                         "auto=cpu 开 / gpu 关）")
    ap.add_argument("--mp-delegate", default="auto", choices=["cpu", "gpu", "auto"],
                    help="stage-1 MediaPipe delegate。auto=子进程冒烟通过用 GPU"
                         "（3ms/帧），失败回退 CPU。GPU 输出与 CPU 有 ~2.8px 级 "
                         "fp16 数值漂移（已知）")
    ap.add_argument("--det-conf", type=float, default=0.5,
                    help="stage-1 手掌检测置信度阈值（遮挡场景可降到 0.4）")
    ap.add_argument("--track-conf", type=float, default=0.5,
                    help="stage-1 手部跟踪置信度阈值（遮挡场景可降到 0.4）")
    ap.add_argument("--video-encoder", default="auto",
                    choices=["auto", "nvenc", "libx264", "mp4v"],
                    help="视频编码器：auto=nvenc→libx264→mp4v 逐级回退；"
                         "mp4v=旧两段式（零回归基线）")
    ap.add_argument("--video-smooth", default="offline", choices=["offline", "causal"],
                    help="视频平滑模式：offline=循环后零相位平滑+第二遍渲染（默认，"
                         "静止抖动 <1mm，代价 ~5s 重解码）；causal=旧因果 One-Euro 单遍")
    ap.add_argument("--sg-window", type=int, default=7,
                    help="离线平滑 savgol 窗口（奇数，默认 7；仅 offline 模式）")
    ap.add_argument("--sg-v0", type=float, default=0.08,
                    help="离线平滑速度混合阈值 m/s（默认 0.08=80mm/s：慢速段仍平滑、"
                         "快动保 raw；速度从 savgol 去噪输出测量）")
    ap.add_argument("--write-smoothed", dest="write_smoothed", action="store_true",
                    default=True,
                    help="parquet 写 hand_3d_smoothed 列（默认开）")
    ap.add_argument("--no-write-smoothed", dest="write_smoothed", action="store_false",
                    help="不写 hand_3d_smoothed 列")
    args = ap.parse_args(argv)

    session = os.path.abspath(args.session_dir)
    if not os.path.isdir(session):
        print(f"[ERROR] 会话目录不存在: {session}")
        return 1
    try:
        stats = run_session(session, args)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        return 1

    _summarize(stats)
    if args.compare:
        _write_metrics_csv(stats, os.path.join(stats["out_dir"], "metrics.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
