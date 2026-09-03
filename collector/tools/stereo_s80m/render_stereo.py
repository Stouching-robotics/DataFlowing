#!/usr/bin/env python3
"""
S80M 双目骨架渲染模块 —— 独立于检测/三角化的可视化组件。

职责：把关键点叠加到左右目画面，合成并排 H.264 视频。两种模式:

  3d (默认): 三角化 3D 骨架（observation.keypoints.hand_3d 列）投影叠加到
             左右目**矫正图**。只消费 hand_3d，不碰 2D 检测列。
  2d       : MediaPipe 原始 2D 关键点（observation.keypoints.stereo_left /
             stereo_right 列，84 维 [2手×21点×xy]）直接叠加到左右目**原图**
             （2D 检测在原图上跑，点坐标即原图像素空间）。

两种用法:

  1) 作为库被 hand_triangulate.py 调用（处理循环内逐帧渲染）
  2) 独立 CLI 纯渲染重放：从已落盘的 parquet 渲染视频，无需重新跑检测

API::

    overlay_view(img, pairs, tri, side, frame_idx, total)   # 3d 单帧叠加
    overlay_view_2d(img, kpts, side, frame_idx, total)      # 2d 单帧叠加
    create_video_writer(out_path, fps, w, h) -> (writer, tmp)  # 创建写器
    finalize_video(writer, tmp_path, out_path) -> str          # 收尾转码, 返回最终路径
    render_session_from_parquet(...) -> str                    # 纯渲染重放

CLI::

    python stereo_s80m/render_stereo.py <session_dir> [--parquet PATH]
        [--calib PATH] [--out DIR] [--mode 3d|2d]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
sys.path.insert(0, _TOOLS_DIR)      # stereo_s80m/hand_detection 已并入 tools/ 命名空间

from stereo_s80m.stereo_triangulate import StereoTriangulator, load_stereo_calibration  # noqa: E402

# ── 复用主程序 hand_detection 的绘制风格 ──────────────────────
# demo（core/hand_tracking.draw_kpts_overlay → hand_common.draw_hand）用
# 五指分色 + 掌心灰连接 + 腕部白圆 + 半径分级（腕9/指尖7/关节5）+ 深色描边，
# 可读性远好于单色连线。这里直接复用同一套绘制；hand_common 仅依赖 cv2/numpy，
# 缺失时退回本文件内的简单画法（单色 draw_skeleton）。
_HAND_DET_DIR = os.path.normpath(os.path.join(_REPO_ROOT, "tools", "hand_detection"))
if os.path.isdir(_HAND_DET_DIR) and _HAND_DET_DIR not in sys.path:
    sys.path.insert(0, _HAND_DET_DIR)
try:
    from hand_common import (draw_hand, compute_joint_angles,
                             count_extended_fingers)  # noqa: F401
    _STYLED_DRAW = True
except ImportError:
    _STYLED_DRAW = False

# ── 骨架拓扑（MediaPipe 21 点连接）─────────────────────────────
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
SKELETON_COLOR = (255, 0, 255)      # 品红: 三角化 3D 骨架投影
SKELETON_2D_COLOR = (255, 255, 0)   # 青: MediaPipe 原始 2D 关键点（原图空间）
RADIUS, THICKNESS = 3, 1


# ── 单帧叠加 ──────────────────────────────────────────────────

def draw_skeleton(img: np.ndarray, pts: np.ndarray, color=SKELETON_COLOR,
                  radius=RADIUS, thickness=THICKNESS):
    """把 (N,2) 像素点按 HAND_EDGES 连线画骨架。非有限点跳过。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    for a, b in HAND_EDGES:
        if a < len(pts) and b < len(pts):
            pa, pb = pts[a], pts[b]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), color, thickness)
    for p in pts:
        if np.all(np.isfinite(p)):
            cv2.circle(img, tuple(p.astype(int)), radius, color, -1)


def _kpts_bbox(pts: np.ndarray, pad: int = 12):
    """有限关键点的包围盒 (x1,y1,x2,y2)；无有效点返回 None。"""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    ok = np.isfinite(pts).all(axis=1)
    if not ok.any():
        return None
    return (int(pts[ok, 0].min()) - pad, int(pts[ok, 1].min()) - pad,
            int(pts[ok, 0].max()) + pad, int(pts[ok, 1].max()) + pad)


def _draw_hand_styled(img: np.ndarray, pts: np.ndarray, label: str,
                      box_color=(0, 220, 220), pts_angle=None):
    """demo 风格单手叠加：teal 边框 + 底衬标签 + 五指分色骨架 + 手势文本。

    学 core/hand_tracking.draw_kpts_overlay → hand_common.draw_hand：
    边框/标签提高可读性，五指分色+半径分级（腕9/指尖7/关节5）+深色描边
    让指骨走向一眼可辨。pts_angle 给出手势角度计算源（2D 或度量 3D 点），
    None 则不画手势文本。
    """
    pts = np.asarray(pts, np.float64).reshape(21, 2)
    ok = np.isfinite(pts).all(axis=1)
    if ok.sum() < 8:                     # 有效点太少, 整只手跳过
        return

    bbox = _kpts_bbox(pts)
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)

    if _STYLED_DRAW and ok.all():
        draw_hand(img, pts.astype(np.float32), angles=None, show_angles=False)
    else:
        draw_skeleton(img, pts)          # 部分无效退回单色(自动跳过非有限点)

    if bbox and pts_angle is not None and _STYLED_DRAW:
        try:
            angles = compute_joint_angles(pts_angle)
            extended = count_extended_fingers(angles)
            text = "open: " + ",".join(extended) if extended else "fist"
            cv2.putText(img, text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 120), 1, cv2.LINE_AA)
        except Exception:
            pass


def overlay_view(img: np.ndarray, pairs, tri: StereoTriangulator, side: str,
                 frame_idx: int, total: int):
    """一张矫正图上叠加三角化 3D 骨架（hand_3d 投影回该视角）。

    pairs : HandPair 列表（每只 .result.points_3d / .result.mean_error /
            .result.valid_count / .left_label）—— 与 hand_triangulate.py
            match_hands 的输出及 hand_3d 列数据同源。
    """
    for i, p in enumerate(pairs):
        proj = tri.project(p.result.points_3d, side)
        label = f"#{i} {p.left_label}"
        if p.result.valid_count:
            label += f" err={p.result.mean_error:.1f}px"
        # 手势角度用度量 3D 点算（投影回 2D 有透视缩短, 角度会失真）
        _draw_hand_styled(img, proj, label, pts_angle=p.result.points_3d)
    cv2.putText(img, f"{side} frame {frame_idx}/{total}  detected={len(pairs)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "COLORED: per-finger skeleton (3D triangulated, projected into this view)",
                (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


def overlay_view_2d(img: np.ndarray, kpts: np.ndarray, side: str,
                    frame_idx: int, total: int):
    """原图上叠加 MediaPipe 原始 2D 关键点（不透矫正/三角化）。

    kpts : (2, 21, 2) 像素坐标；该手全零 = 未检出，跳过。
    2D 检测在原图像素空间进行，因此叠加在原图（不矫正）上才对得上。
    """
    kpts = np.asarray(kpts, np.float32).reshape(2, 21, 2)
    for i in range(len(kpts)):
        pts = kpts[i]
        if np.all(pts == 0):      # 缺手全零占位
            continue
        _draw_hand_styled(img, pts, f"#{i}", pts_angle=pts)
    cv2.putText(img, f"{side} frame {frame_idx}/{total}  detected={int(np.any(kpts, axis=(1, 2)).sum())}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "COLORED: per-finger skeleton (2D MediaPipe keypoints, raw view)",
                (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


# ── 视频输出 ──────────────────────────────────────────────────

def create_video_writer(out_path: str, fps: float, width: int, height: int):
    """创建 mp4v 临时写器。返回 (writer, tmp_path)。"""
    tmp_path = os.path.splitext(out_path)[0] + "_tmp.avi"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写器: {tmp_path}")
    return writer, tmp_path


def finalize_video(writer, tmp_path: str, out_path: str) -> str:
    """释放写器并转码 H.264。返回最终视频路径（无可用 ffmpeg 时保留 mp4v）。"""
    writer.release()
    converted = False
    # 候选 ffmpeg：conda base 的 ffmpeg 因 openvino/tbb 符号错误不可用
    # （2026-08 实测），逐个尝试直到成功
    candidates = [shutil.which("ffmpeg"), "/usr/bin/ffmpeg",
                  os.environ.get(
                      "FFMPEG_BIN",
                      os.path.expanduser(
                          "~/miniconda3/envs/lerobot/bin/ffmpeg"))]
    for ff in dict.fromkeys(c for c in candidates if c):
        try:
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp_path,
                            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                            out_path], check=True)
            converted = True
            break
        except (OSError, subprocess.CalledProcessError):
            continue
    if converted:
        os.remove(tmp_path)
        return out_path
    return tmp_path


# ── 纯渲染重放（不重新检测，只渲染已落盘的 hand_3d）────────────

def _find_video(session_dir: str, cam: str) -> str | None:
    """定位左右目视频：videos/<cam>/chunk-0000/<cam>.mp4，回退 videos/<cam>.mp4。"""
    for p in (os.path.join(session_dir, "videos", cam, "chunk-0000", f"{cam}.mp4"),
              os.path.join(session_dir, "videos", f"{cam}.mp4")):
        if os.path.isfile(p):
            return p
    return None


class _ReplayPair:
    """把 parquet 行还原成 overlay_view 可消费的伪 HandPair。"""

    def __init__(self, label: str, points_3d, mean_error, valid_count):
        self.left_label = label
        self.result = SimpleNamespace(points_3d=points_3d,
                                      mean_error=mean_error,
                                      valid_count=valid_count)


def render_session_from_parquet(session_dir: str, parquet_path: str,
                                calib_path: str = None, out_dir: str = None,
                                mode: str = "3d") -> str:
    """从已落盘的关键点 parquet 渲染视频（不做任何检测/三角化）。

    mode="3d": 消费 observation.keypoints.hand_3d 列（126 = 2手×21点×3）+
        present/label/err 列，投影叠加到左右目**矫正图**。
    mode="2d": 消费 observation.keypoints.stereo_left/right 列（84 = 2手×21点×2），
        MediaPipe 原图空间关键点直接叠加到左右目**原图**，无需标定。
    返回视频路径。
    """
    import pyarrow.parquet as pq

    mode = mode.lower()
    if mode not in ("3d", "2d"):
        raise ValueError(f"未知渲染模式: {mode}（可选 3d|2d）")

    session = os.path.abspath(session_dir)
    vp_l, vp_r = _find_video(session, "stereo_left"), _find_video(session, "stereo_right")
    if not vp_l or not vp_r:
        raise FileNotFoundError(f"找不到双目视频: stereo_left={vp_l} stereo_right={vp_r}")

    calib = load_stereo_calibration(session, calib_path) if mode == "3d" else None
    if mode == "3d" and calib is None:
        raise RuntimeError("无可用的立体标定（查找链: episode calibration/ → config/s80m_stereo_calibration.json）")
    tri = StereoTriangulator(calib) if calib else None

    rows = pq.read_table(parquet_path).to_pylist()
    if not rows:
        raise RuntimeError(f"parquet 为空: {parquet_path}")
    if tri is not None:
        print(f"重放 {len(rows)} 行 ← {parquet_path}  ({tri.summarize()})")
    else:
        print(f"重放 {len(rows)} 行 ← {parquet_path}  (2d 模式, 无需标定)")

    if out_dir is None:
        out_dir = os.path.join(_REPO_ROOT, "keypoints_output",
                               os.path.basename(os.path.dirname(session)),
                               os.path.basename(session))
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, "stereo_triangulate.mp4" if mode == "3d"
                             else "stereo_2d.mp4")

    vc_l, vc_r = cv2.VideoCapture(vp_l), cv2.VideoCapture(vp_r)
    fps = vc_l.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(vc_l.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, tmp_path = create_video_writer(out_video, fps, w * 2, h)

    for k, row in enumerate(rows):
        fi = int(row["frame_index"])
        vc_l.set(cv2.CAP_PROP_POS_FRAMES, fi)
        vc_r.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok_l, fl = vc_l.read()
        ok_r, fr = vc_r.read()
        if not (ok_l and ok_r):
            continue

        if mode == "2d":
            # 2D 检测在原图空间 → 叠加到原图，不做矫正
            kpts_l = np.array(row["observation.keypoints.stereo_left"],
                              np.float32).reshape(2, 21, 2)
            kpts_r = np.array(row["observation.keypoints.stereo_right"],
                              np.float32).reshape(2, 21, 2)
            frame_l, frame_r = fl.copy(), fr.copy()
            overlay_view_2d(frame_l, kpts_l, "left", fi, len(rows))
            overlay_view_2d(frame_r, kpts_r, "right", fi, len(rows))
        else:
            pts = np.array(row["observation.keypoints.hand_3d"], np.float32).reshape(2, 21, 3)
            errs = row["observation.keypoints.reprojection_error"]
            labels = [row["observation.keypoints.hand_0_label"],
                      row["observation.keypoints.hand_1_label"]]
            presents = [row["observation.keypoints.hand_0_present"],
                        row["observation.keypoints.hand_1_present"]]
            pairs = []
            for i in range(2):
                if presents[i] and np.isfinite(errs[i]):
                    pairs.append(_ReplayPair(labels[i], pts[i], float(errs[i]), 21))
            frame_l = tri.rectified_image(fl, "left")
            frame_r = tri.rectified_image(fr, "right")
            overlay_view(frame_l, pairs, tri, "left", fi, len(rows))
            overlay_view(frame_r, pairs, tri, "right", fi, len(rows))
        writer.write(cv2.hconcat([frame_l, frame_r]))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(rows)} 帧")

    vc_l.release()
    vc_r.release()
    return finalize_video(writer, tmp_path, out_video)


def render_video_2d(session_dir: str, out_dir: str = None,
                    scale: float = 1.0, smooth: bool = True) -> str:
    """直接从视频跑 2D 检测 + 渲染（无需已落盘 parquet）。

    左右目各自独立 MediaPipe pipeline（共享实例交替喂帧会污染 VIDEO 模式
    追踪先验），检测在原图/缩放图像素空间 → overlay_view_2d 叠加 → 并排 H.264。
    scale < 1 时检测在缩放图上跑（基准结论: 640×400 与 1280×800 检出率
    等价且更快），关键点坐标按 1/scale 映射回原图渲染。
    返回最终视频路径。
    """
    from hand_detection.hand_pipeline_mediapipe import MediaPipeHandPipeline

    MODEL_PATH = os.path.join(_REPO_ROOT, "tools", "models", "hand_landmarker.task")
    session = os.path.abspath(session_dir)
    vp_l, vp_r = _find_video(session, "stereo_left"), _find_video(session, "stereo_right")
    if not vp_l or not vp_r:
        raise FileNotFoundError(f"找不到双目视频: stereo_left={vp_l} stereo_right={vp_r}")

    pipe_l = MediaPipeHandPipeline(model_path=MODEL_PATH, num_hands=2,
                                   mirror=False, smooth=smooth)   # 与 demo 一致:
    pipe_r = MediaPipeHandPipeline(model_path=MODEL_PATH, num_hands=2,
                                   mirror=False, smooth=smooth)   # 头戴双目不镜像
    vc_l, vc_r = cv2.VideoCapture(vp_l), cv2.VideoCapture(vp_r)
    fps = vc_l.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(vc_l.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(vc_l.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if scale != 1.0:
        dw, dh = int(w * scale), int(h * scale)

    if out_dir is None:
        out_dir = os.path.join(_REPO_ROOT, "keypoints_output",
                               os.path.basename(os.path.dirname(session)),
                               os.path.basename(session))
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, "stereo_2d.mp4")
    writer, tmp_path = create_video_writer(out_video, fps, w * 2, h)
    print(f"渲染 2D ← {os.path.basename(vp_l)} / {os.path.basename(vp_r)}  "
          f"{w}x{h}@{fps:.0f}fps {total}帧  scale={scale}  smooth={smooth}")

    kpts_l = kpts_r = None
    for i in range(total):
        ok_l, fl = vc_l.read()
        ok_r, fr = vc_r.read()
        if not (ok_l and ok_r):
            break
        dfl = cv2.resize(fl, (dw, dh), interpolation=cv2.INTER_AREA) if scale != 1.0 else fl
        dfr = cv2.resize(fr, (dw, dh), interpolation=cv2.INTER_AREA) if scale != 1.0 else fr
        kpts_l = np.zeros((2, 21, 2), np.float32)
        kpts_r = np.zeros((2, 21, 2), np.float32)
        for j, hh in enumerate(pipe_l.process(dfl).hands[:2]):
            kpts_l[j] = np.array(hh.landmarks, np.float32)
        for j, hh in enumerate(pipe_r.process(dfr).hands[:2]):
            kpts_r[j] = np.array(hh.landmarks, np.float32)
        if scale != 1.0:                     # 缩放图检测 → 原图像素空间
            kpts_l /= scale
            kpts_r /= scale
        frame_l, frame_r = fl.copy(), fr.copy()
        overlay_view_2d(frame_l, kpts_l, "left", i, total)
        overlay_view_2d(frame_r, kpts_r, "right", i, total)
        writer.write(cv2.hconcat([frame_l, frame_r]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{total} 帧", flush=True)

    vc_l.release()
    vc_r.release()
    pipe_l.close()
    pipe_r.close()
    return finalize_video(writer, tmp_path, out_video)


def main():
    ap = argparse.ArgumentParser(description="S80M 骨架渲染（parquet 重放或直接视频检测）")
    ap.add_argument("session_dir", help="episode 目录（提供左右目视频）")
    ap.add_argument("--parquet", default=None,
                    help="关键点 parquet（默认 <session>/data/keypoints/chunk-0000/chunk_000000.parquet）")
    ap.add_argument("--calib", default=None, help="标定 JSON（默认走查找链）")
    ap.add_argument("--out", default=None, help="输出目录（默认 keypoints_output/<tag>/<session>）")
    ap.add_argument("--mode", default="3d", choices=["3d", "2d"],
                    help="渲染模式: 3d=三角化骨架投影到矫正图(默认), 2d=MediaPipe 原始 2D 关键点叠加原图")
    ap.add_argument("--detect", action="store_true",
                    help="跳过 parquet：直接从视频跑 2D 检测 + 渲染（无需落盘数据）")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="检测缩放系数 (<1 更快; 基准: 0.5=640×400 与全分辨率检出等价; 需 --detect)")
    args = ap.parse_args()

    if args.detect:
        try:
            video = render_video_2d(args.session_dir, args.out, args.scale)
            print(f"✓ 视频: {video}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        return

    parquet = args.parquet or os.path.join(
        args.session_dir, "data", "keypoints", "chunk-0000", "chunk_000000.parquet")
    if not os.path.isfile(parquet):
        print(f"[ERROR] 找不到关键点 parquet: {parquet}")
        sys.exit(1)
    try:
        video = render_session_from_parquet(args.session_dir, parquet,
                                            args.calib, args.out, args.mode)
        print(f"✓ 视频: {video}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
