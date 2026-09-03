#!/usr/bin/env python3
"""D435 RGB-D 单目 3D 手部关键点管线主循环。

链路：RGB(1280×720) MediaPipe 2D（+ 手性投票）→ 原生深度(848×480 mm PNG)
离线对齐抬升 3D（彩色相机系）→ 单目槽位分配（互斥+复活）→ HandSlotTracker
遮挡传播 → tracker 前向+后向填充 + offline_smooth(fps=30) → 旋转渲染 +
RGB 叠显 + parquet。

用法（venv）:
    ./tools/hand_3d_d435/run_d435.sh <session_dir> [选项]

产物（keypoints_output/<tag>/<session>/）:
    d435_hand_3d_rotating.mp4 / d435_rgb_overlay.mp4 /
    hand_3d_refined/chunk-000.parquet + d435_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from stereo_s80m.hand_3d.detector import MediaPipeDetector        # noqa: E402
from stereo_s80m.hand_3d.identity import HandednessVoter          # noqa: E402
from stereo_s80m.hand_3d.track3d import HandSlotTracker           # noqa: E402
from stereo_s80m.hand_3d.postprocess import offline_smooth        # noqa: E402
from stereo_s80m.hand_3d.renderer_3d import RotatingSkeletonRenderer   # noqa: E402
from stereo_s80m.hand_3d.video_writer import create_video_sink    # noqa: E402
from stereo_s80m.hand_3d import io                                # noqa: E402

from hand_3d_d435.depth_align import (DepthAligner, load_calib,       # noqa: E402
                                      load_session_depth_intr,
                                      load_session_depth_files,
                                      load_depth_frame)
from hand_3d_d435.lift3d import (D435Pair, LiftResult, lift_hand,     # noqa: E402
                                 gate_observations)
from hand_3d_d435.fill_track import tracker_fill                  # noqa: E402
from hand_3d_d435.mono_assign import assign_mono_slots             # noqa: E402
from hand_3d_d435.render_overlay import blend_depth, draw_overlay  # noqa: E402

_FX_REL_TOL = 0.01
RENDER_SIZE = (1280, 720)


# ── 小工具 ────────────────────────────────────────────────────

def _load_episode_task(session: str) -> tuple:
    """meta/episodes parquet → (episode_index, task_index)。

    venv 无 pandas（io.load_episode_meta 会静默回退 0），
    用 pyarrow 直读。
    """
    episode_index = 0
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(os.path.join(session, "meta", "episodes",
                                       "chunk_000000.parquet"),
                          columns=["episode_index"])
        episode_index = int(t.column("episode_index")[0].as_py())
    except Exception:
        pass
    task_index = 0
    try:
        with open(os.path.join(session, "meta", "tasks.jsonl"),
                  encoding="utf-8") as f:
            task_index = int(json.loads(f.readline())["task_index"])
    except Exception:
        pass
    return episode_index, task_index


def _nan_pair(label: str = "") -> D435Pair:
    """缺槽占位（全 NaN，valid_count=0 → pack_errors 输出 NaN）。"""
    return D435Pair(result=LiftResult(np.full((21, 3), np.nan, np.float64),
                                      float("nan"), 0), left_label=label)


def _pred_pair(pred: np.ndarray, label: str = "") -> D435Pair:
    """传播帧：tracker 预测 3D 包成 pair（valid_count=0 不冒充真检测）。"""
    return D435Pair(result=LiftResult(np.asarray(pred, np.float64)
                                      .reshape(21, 3), float("nan"), 0),
                    left_label=label)


def _pack_2d_slots(slot_dets) -> list:
    """槽位序 DetectedHand|None → 84 维 [2手×21点×(x,y)]（None 安全）。"""
    arr = np.zeros((2, 21, 2), np.float32)
    for i, h in enumerate(slot_dets):
        if h is not None:
            arr[i] = np.asarray(h.landmarks, np.float32).reshape(21, 2)
    return arr.flatten().tolist()


# ── 渲染 pass ─────────────────────────────────────────────────

def _render_videos(args, rows, video, fps, w, h, out_dir, aligner,
                   depth_files) -> None:
    rot = create_video_sink(os.path.join(out_dir, "d435_hand_3d_rotating.mp4"),
                            fps, *RENDER_SIZE, encoder=args.video_encoder)
    ov = create_video_sink(os.path.join(out_dir, "d435_rgb_overlay.mp4"),
                           fps, w, h, encoder=args.video_encoder)
    renderer = RotatingSkeletonRenderer(*RENDER_SIZE, revolutions=2.0)
    cap = cv2.VideoCapture(video)
    for i, row in enumerate(rows):
        hands3d = np.asarray(row["observation.keypoints.hand_3d_smoothed"],
                             np.float32).reshape(2, 21, 3)
        labels = [row["observation.keypoints.hand_0_label"],
                  row["observation.keypoints.hand_1_label"]]
        rot.write(renderer.render(hands3d, labels, (np.nan, np.nan),
                                  i, len(rows),
                                  "D435 hand keypoints (color-cam frame, meters)"))
        ok, rgb = cap.read()
        if not ok:
            continue
        if args.depth_overlay:
            dp = depth_files.get(i + 1)
            if dp is not None:
                d = load_depth_frame(dp, (aligner.dh, aligner.dw))
                if d is not None:
                    rgb = blend_depth(rgb, aligner.align_depth_to_color(d),
                                      0.35)
        hands2d = np.asarray(row["observation.keypoints.stereo_left"],
                             np.float32).reshape(2, 21, 2)
        presents = [row["observation.keypoints.hand_0_present"],
                    row["observation.keypoints.hand_1_present"]]
        ov.write(draw_overlay(rgb, hands2d, hands3d, labels,
                              row["observation.keypoints.propagated"],
                              presents, i, len(rows)))
    cap.release()
    rp = rot.close()
    op = ov.close()
    if rp:
        print(f"  ✓ rotating: {rp}")
    if op:
        print(f"  ✓ overlay: {op}")


# ── 汇总 ──────────────────────────────────────────────────────

def _summarize(rows, out_dir, n_processed, t_total) -> None:
    h3 = np.stack([np.asarray(r["observation.keypoints.hand_3d_smoothed"],
                              np.float32).reshape(2, 21, 3)
                   for r in rows])                       # (N,2,21,3)
    prop = np.stack([r["observation.keypoints.propagated"] for r in rows])
    present = np.stack([(r["observation.keypoints.hand_0_present"],
                         r["observation.keypoints.hand_1_present"])
                        for r in rows])
    print(f"\n── 汇总（{n_processed} 帧, {t_total:.1f}s）──")
    flips = 0
    bones = []
    for s, name in enumerate(("slot0", "slot1")):
        n_pres = int(present[:, s].sum())
        n_prop = int((present[:, s] & prop[:, s]).sum())
        real = present[:, s] & ~prop[:, s]
        # label 翻转（连续 present 帧间 Left↔Right 切换）
        labs = [r[f"observation.keypoints.hand_{s}_label"] for r in rows]
        flips_s = 0
        prev = ""
        for i in range(1, len(rows)):
            if present[i, s] and present[i - 1, s] \
                    and labs[i] in ("Left", "Right") \
                    and labs[i - 1] in ("Left", "Right") \
                    and labs[i] != labs[i - 1]:
                flips_s += 1
        flips += flips_s
        # 腕部逐帧位移（平滑后）
        w = h3[:, s, 0, :]
        ok = np.isfinite(w).all(axis=1)
        disp = np.linalg.norm(np.diff(w[ok], axis=0), axis=1) * 1000.0
        # 骨长自检（腕→中指MCP，真检测帧）
        b = np.linalg.norm(h3[real, s, 0, :] - h3[real, s, 9, :],
                           axis=1) * 1000.0
        b = b[np.isfinite(b)]
        bones.append(b)
        print(f"  {name}: present {n_pres}/{n_processed} "
              f"propagated {n_prop} 翻转 {flips_s} "
              f"腕位移 p50={np.median(disp) if len(disp) else np.nan:.2f}mm "
              f"p95={np.percentile(disp, 95) if len(disp) else np.nan:.2f}mm "
              f"骨长 p50={np.median(b) if len(b) else np.nan:.1f}mm "
              f"(n={len(b)})")
    print(f"  全片 label 翻转: {flips}")
    print(f"  产物目录: {out_dir}")


def _write_metrics(metrics, path) -> None:
    if not metrics:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(metrics[0].keys())
        for m in metrics:
            wr.writerow(m.values())
    print(f"  ✓ metrics: {path}")


# ── 主循环 ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", help="录制会话目录（如 data/recordings/222/222_000011）")
    ap.add_argument("--out-dir", help="输出目录（默认 keypoints_output/<tag>/<session>/）")
    ap.add_argument("--calib", help="标定 JSON（默认 hand_3d_d435/calibration/d435_color_calib.json）")
    ap.add_argument("--mp-delegate", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--track-conf", type=float, default=0.5)
    ap.add_argument("--propagate-max", type=int, default=15,
                    help="槽位丢失帧数硬顶（超限 absent 不幻觉）")
    ap.add_argument("--depth-overlay", action="store_true",
                    help="overlay 视频叠伪彩深度层")
    ap.add_argument("--video-encoder", default="auto")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--track-debug", action="store_true",
                    help="写 track_events.csv（tracker 事件）")
    args = ap.parse_args()

    session = args.session_dir.rstrip("/")
    if not os.path.isdir(session):
        print(f"错误: 会话目录不存在: {session}")
        sys.exit(2)
    video = io.find_video(session, "d435_rgb")
    if not video:
        print(f"错误: 找不到 RGB 视频: {session}/videos/d435_rgb/")
        sys.exit(2)
    depth_dir = os.path.join(session, "depth", "d435_depth")
    if not os.path.isdir(depth_dir):
        print(f"错误: 找不到深度目录: {depth_dir}")
        sys.exit(2)

    # 标定（彩色内参/外参=固化 JSON；深度内参=录制 head_stereo.json 权威）
    try:
        calib = load_calib(args.calib)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(2)
    session_depth = load_session_depth_intr(session)
    if session_depth is None:
        print("警告: 录制 head_stereo.json 缺失，改用固化标定的深度内参")
        session_depth = calib["depth_intrinsics"]
    else:
        cd = calib.get("depth_intrinsics")
        if cd:
            for k in ("fx", "fy", "cx", "cy"):
                a, b = float(session_depth[k]), float(cd[k])
                rel = abs(a - b) / max(abs(a), 1e-9)
                if rel > _FX_REL_TOL:
                    print(f"警告: 录制深度内参 {k}={a:.2f} 与固化标定 {b:.2f} "
                          f"差 {rel * 100:.1f}%（可能非同一设备录制）")
    aligner = DepthAligner(calib["color_intrinsics"],
                           calib["depth_to_color"], session_depth)
    print(f"标定: serial={calib.get('serial', '?')} 彩色 "
          f"fx={aligner.fx_c:.2f} fy={aligner.fy_c:.2f} "
          f"cx={aligner.cx_c:.2f} cy={aligner.cy_c:.2f}")

    episode_index, task_index = _load_episode_task(session)
    timestamps = io.load_timestamps(session)
    tag = os.path.basename(os.path.dirname(session))
    name = os.path.basename(session)
    out_dir = args.out_dir or os.path.join(_REPO_ROOT, "keypoints_output",
                                           tag, name)
    os.makedirs(out_dir, exist_ok=True)

    depth_files = load_session_depth_files(depth_dir)
    if not depth_files:
        print(f"错误: 深度目录无 PNG16（或 v1.0.11 窗口 raw16 bin）: {depth_dir}")
        sys.exit(2)

    det = MediaPipeDetector(num_hands=2, delegate=args.mp_delegate,
                            det_conf=args.det_conf,
                            track_conf=args.track_conf)
    voter = HandednessVoter()
    tracker = HandSlotTracker(
        max_lost=args.propagate_max,
        debug_log=os.path.join(out_dir, "track_events.csv")
        if args.track_debug else None)

    cap = cv2.VideoCapture(video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows, metrics = [], []
    t0 = time.perf_counter()
    t_det = t_align = 0.0
    lost_counts = [0, 0]     # 各槽连续丢失帧数（assigner 困境槽无门限救援用）
    n = 0
    while True:
        ok, rgb = cap.read()
        if not ok:
            break
        dp = depth_files.get(n + 1)
        t1 = time.perf_counter()
        if dp is None:
            aligned = np.zeros((aligner.ch, aligner.cw), np.float32)
        else:
            d = load_depth_frame(dp, (aligner.dh, aligner.dw))
            if d is None:
                d = np.zeros((aligner.dh, aligner.dw), np.uint16)
            aligned = aligner.align_depth_to_color(d)
        t_align += time.perf_counter() - t1

        t1 = time.perf_counter()
        hands = det.detect(rgb)
        # 空帧不喂 voter：identity.py 空帧会清空轨迹（scene reset 语义），
        # 单目录制的短暂漏检（几帧无手）会清掉票仓 → 重建期原始 label
        # 闪烁穿透 → 两手同 label（实测 222_000011 f372-373 空帧后 slot1
        # 58 帧无法复活）。跳过空帧让轨迹 idle 保持（voter 设计本就支持
        # 手暂离再回来继续投票；长期不归的手会按 idle 被新轨迹顶替）。
        if hands:
            voter.update(hands, frame_w=w, frame_h=h, frame=n, cam="d435")
        t_det += time.perf_counter() - t1

        pairs = [lift_hand(hd, aligner, aligned) for hd in hands]
        out = assign_mono_slots(
            pairs, tracker, n,
            (aligner.fx_c, aligner.fy_c, aligner.cx_c, aligner.cy_c),
            lost_counts=tuple(lost_counts))

        slot_pairs, slot_dets = [], []
        states = []
        for s in range(2):
            if out[s] is not None:
                p = out[s]
                # 时序一致性门：与槽预测差 >150mm 的点判可疑置 NaN
                # （tracker 对 NaN 点保持纯预测，翻面观测不入状态）
                gated, wholesale = gate_observations(p.result.points_3d,
                                                     tracker.predict(s, n))
                if wholesale:
                    # 整手级不匹配：槽状态过时（入场前假检测钉在背景）→
                    # 借 label 翻转触发 track3d 槽位重置，随即真观测干净初始化
                    tracker.observe_slot(s, "\x00reset",
                                         np.full((21, 3), np.nan), n)
                tracker.observe_slot(s, p.left_label, gated, n)
                p.result.points_3d = gated
                lost_counts[s] = 0
                slot_pairs.append(p)
                slot_dets.append(p.det)
                states.append("real")
            else:
                pred = tracker.predict(s, n)
                tracker.mark_lost(s, n)
                lost_counts[s] += 1
                if pred is not None:
                    slot_pairs.append(_pred_pair(pred, tracker.slot_label(s)))
                    slot_dets.append(None)
                    states.append("propagated")
                else:
                    slot_pairs.append(_nan_pair(tracker.slot_label(s)))
                    slot_dets.append(None)
                    states.append("absent")

        presents = [st != "absent" for st in states]
        propagated = [st == "propagated" for st in states]
        labels = [slot_pairs[s].left_label if out[s] is not None
                  else tracker.slot_label(s) for s in range(2)]
        rows.append({
            "episode_index": episode_index,
            "frame_index": n,
            "timestamp": np.float32(timestamps.get(n, 0.0)),
            "task_index": task_index,
            "observation.keypoints.stereo_left": _pack_2d_slots(slot_dets),
            "observation.keypoints.stereo_right": _pack_2d_slots(slot_dets),
            "observation.keypoints.hand_3d": io.pack_3d(slot_pairs),
            "observation.keypoints.reprojection_error":
                io.pack_errors(slot_pairs),
            "observation.keypoints.hand_0_present": presents[0],
            "observation.keypoints.hand_1_present": presents[1],
            "observation.keypoints.hand_0_label": labels[0],
            "observation.keypoints.hand_1_label": labels[1],
            "observation.keypoints.stage2": io.pack_stage2(slot_pairs),
            "observation.keypoints.propagated": propagated,
        })
        # metrics：逐帧旁挂质量信号
        m = {"frame": n, "n_det": len(hands)}
        for s in range(2):
            p = slot_pairs[s]
            pts = np.asarray(p.result.points_3d, np.float64).reshape(21, 3)
            ok = np.isfinite(pts).all(axis=1)
            z = np.median(pts[ok, 2]) * 1000.0 if ok.any() else np.nan
            wz = pts[0, 2] * 1000.0 if np.isfinite(pts[0]).all() else np.nan
            m.update({
                f"s{s}_state": states[s],
                f"s{s}_label": labels[s],
                f"s{s}_nvalid": int(ok.sum()),
                f"s{s}_wrist_z_mm": round(float(wz), 1) if np.isfinite(wz) else "",
                f"s{s}_med_z_mm": round(float(z), 1) if np.isfinite(z) else "",
            })
        if all(presents):
            w0 = np.asarray(slot_pairs[0].result.points_3d)[0]
            w1 = np.asarray(slot_pairs[1].result.points_3d)[0]
            if np.isfinite(w0).all() and np.isfinite(w1).all():
                m["wrist_dist_mm"] = round(
                    float(np.linalg.norm(w0 - w1)) * 1000.0, 1)
        metrics.append(m)
        n += 1
    cap.release()
    print(f"主循环: {n} 帧  det {t_det:.1f}s  align {t_align:.1f}s")

    if not rows:
        print("错误: 未读到任何帧")
        sys.exit(2)

    # 收尾：tracker 前向+后向填充（替代 fill_gaps 短桥接：长缺口也补，
    # 腕点 62% 缺失等由恒速外推补全）+ 离线零相位平滑（D435 必须 fps=30）
    filled = tracker_fill(rows, max_lost=args.propagate_max)
    smoothed = offline_smooth(rows, sg_window=7, sg_poly=3, v0=0.08,
                              fps=fps, still_window=21)
    for i, r in enumerate(rows):
        r["observation.keypoints.hand_3d_smoothed"] = \
            smoothed[i].reshape(-1).tolist()
    print(f"tracker_fill 填充 {filled} 帧-槽；offline_smooth 完成")

    if not args.no_parquet:
        pq_path = io.write_parquet(
            rows, os.path.join(out_dir, "hand_3d_refined", "chunk-000.parquet"))
        print(f"  ✓ parquet: {pq_path}")
    _write_metrics(metrics, os.path.join(out_dir, "d435_metrics.csv"))

    if not args.no_video:
        _render_videos(args, rows, video, fps, w, h, out_dir, aligner,
                       depth_files)

    _summarize(rows, out_dir, n, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
