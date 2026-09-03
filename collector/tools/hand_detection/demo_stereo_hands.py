#!/usr/bin/env python3
"""Stereo hand keypoint detection demo.

Processes both left and right camera videos simultaneously using
MediaPipeHandPipeline + One-Euro smoothing, renders side-by-side output.

Usage:
    python hand_detection/demo_stereo_hands.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hand_detection.hand_pipeline_mediapipe import MediaPipeHandPipeline
from hand_detection.hand_common import draw_hand

# ── Config ─────────────────────────────────────────────────

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_DEMO_DIR, "..", ".."))

LEFT_VIDEO = os.path.join(
    _REPO_ROOT,
    "data/recordings/Test1/Test1_000020/videos/stereo_left/chunk-0000/stereo_left.mp4",
)
RIGHT_VIDEO = os.path.join(
    _REPO_ROOT,
    "data/recordings/Test1/Test1_000020/videos/stereo_right/chunk-0000/stereo_right.mp4",
)

OUTPUT_DIR = os.path.join(_REPO_ROOT, "keypoints_output")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "stereo_hands.mp4")
OUTPUT_VIDEO_TMP = os.path.join(OUTPUT_DIR, "stereo_hands_tmp.mp4")

MODEL_PATH = os.path.join(_REPO_ROOT, "tools", "models", "hand_landmarker.task")

# conda base 的 ffmpeg 因 openvino/tbb 符号错误不可用，逐个候选尝试
FFMPEG_CANDIDATES = [shutil.which("ffmpeg"), "/usr/bin/ffmpeg",
                     os.environ.get(
                         "FFMPEG_BIN",
                         os.path.expanduser(
                             "~/miniconda3/envs/lerobot/bin/ffmpeg"))]

# Output scale: each camera view resized to this width (height auto)
VIEW_WIDTH = 640

# ── Drawing ────────────────────────────────────────────────


def draw_header(frame, text, color=(255, 255, 255)):
    """Draw a semi-transparent header bar at the top."""
    h = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def draw_info_overlay(frame, hand_results, fps_val):
    """Draw hand count, extended fingers, and FPS on frame."""
    y = frame.shape[0] - 10
    # FPS
    cv2.putText(frame, f"{fps_val:.0f} fps", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    # Hands
    if not hand_results:
        cv2.putText(frame, "No hands", (10, y - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        for hi, hand in enumerate(hand_results):
            wrist = (int(hand.landmarks[0][0]), int(hand.landmarks[0][1]))
            ext_str = " ".join(hand.extended[:3]) if hand.extended else "—"
            label = f"{hand.label} ({hand.score:.2f})  [{ext_str}]"
            color = (0, 255, 0) if hand.label == "Right" else (255, 200, 0)
            cv2.putText(frame, label, (10, y - 24 - hi * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def process_and_draw(pipe: MediaPipeHandPipeline, frame: np.ndarray) -> np.ndarray:
    """Run hand detection on frame and draw results. Returns annotated frame."""
    out = frame.copy()
    result = pipe.process(frame)

    for hand in result.hands:
        draw_hand(out, hand.landmarks, angles=hand.angles, show_angles=True)

    return out, result.hands


# ── Main ───────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Stereo hand keypoint detection demo")
    ap.add_argument("--left", default=LEFT_VIDEO, help="左目视频路径")
    ap.add_argument("--right", default=RIGHT_VIDEO, help="右目视频路径")
    ap.add_argument("--out", default=OUTPUT_VIDEO, help="输出视频路径")
    ap.add_argument("--view-width", type=int, default=VIEW_WIDTH,
                    help="每目输出宽度 (默认 640)")
    args = ap.parse_args()
    out_video = os.path.abspath(args.out)
    out_video_tmp = os.path.splitext(out_video)[0] + "_tmp.mp4"
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)

    # ── Open videos ──────────────────────────────────────
    cap_left = cv2.VideoCapture(args.left)
    cap_right = cv2.VideoCapture(args.right)

    for name, cap in [("Left", cap_left), ("Right", cap_right)]:
        if not cap.isOpened():
            print(f"ERROR: Cannot open {name} video")
            sys.exit(1)

    fps_in = cap_left.get(cv2.CAP_PROP_FPS)
    w_in = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    # Verify right matches
    w_r = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_r = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Left  video: {w_in}x{h_in} @ {fps_in:.0f} fps, {total_frames} frames")
    print(f"Right video: {w_r}x{h_r} @ {fps_in:.0f} fps, {total_r} frames")

    # Scale dimensions
    view_w = args.view_width
    view_h = int(h_in * view_w / w_in)
    print(f"Output view: {view_w}x{view_h} each")

    # ── Init pipeline ────────────────────────────────────
    # 左右目各自独立 pipeline：共享实例交替喂帧会污染 VIDEO 模式追踪先验，
    # 导致双手检出率塌陷（基准实测 99.7% → ~50%）
    def _make_pipe():
        return MediaPipeHandPipeline(
            model_path=MODEL_PATH,
            num_hands=2,
            det_conf=0.5,
            track_conf=0.5,
            preprocess_mode="none",
            mirror=False,      # head-mounted stereo camera, no mirror
            smooth=True,
            freq_min=5.0,
            beta=0.05,
        )

    pipe_left = _make_pipe()
    pipe_right = _make_pipe()

    # ── Output video ─────────────────────────────────────
    # Layout: left view + right view side by side
    out_w = view_w * 2
    out_h = view_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_tmp, fourcc, fps_in, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video")
        sys.exit(1)

    print(f"Output: {out_video}  ({out_w}x{out_h})")
    print("Processing...")

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        if not ret_left or not ret_right:
            break

        # Resize for output
        view_left = cv2.resize(frame_left, (view_w, view_h))
        view_right = cv2.resize(frame_right, (view_w, view_h))

        # Detect + draw
        left_out, hands_left = process_and_draw(pipe_left, view_left)
        right_out, hands_right = process_and_draw(pipe_right, view_right)

        # Headers
        draw_header(left_out, "LEFT CAMERA  (Stereo Left)")
        draw_header(right_out, "RIGHT CAMERA  (Stereo Right)")

        # Info overlays
        elapsed = time.perf_counter() - t_start
        fps_proc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        draw_info_overlay(left_out, hands_left, fps_proc)
        draw_info_overlay(right_out, hands_right, fps_proc)

        # Combine side-by-side
        side_by_side = np.hstack([left_out, right_out])

        # Progress bar at bottom
        progress = int(out_w * frame_idx / max(total_frames, 1))
        cv2.line(side_by_side, (0, out_h - 2), (progress, out_h - 2),
                 (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(side_by_side)
        frame_idx += 1

        if frame_idx % 25 == 0:
            print(f"  Frame {frame_idx}/{max(total_frames, total_r)} | "
                  f"{fps_proc:.1f} fps | "
                  f"L:{len(hands_left)} R:{len(hands_right)} hands")

    # ── Cleanup ──────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    cap_left.release()
    cap_right.release()
    writer.release()
    pipe_left.close()
    pipe_right.close()

    print(f"\nDone! {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps avg)")

    # ── Convert to H.264 ─────────────────────────────────
    print("Converting to H.264...")
    converted = False
    for ff in dict.fromkeys(c for c in FFMPEG_CANDIDATES if c):
        try:
            ret = subprocess.run([
                ff, "-y",
                "-i", out_video_tmp,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                out_video,
            ], capture_output=True, text=True)
            if ret.returncode == 0:
                converted = True
                break
        except OSError:
            continue        # 候选不存在/无法执行，试下一个
    if converted:
        os.remove(out_video_tmp)
        size_mb = os.path.getsize(out_video) / 1024 / 1024
        print(f"Output saved: {out_video}  ({size_mb:.1f} MB)")
    else:
        print(f"ffmpeg failed, raw mp4v kept: {out_video_tmp}")
        if ret.stderr:
            print(ret.stderr[-500:])


if __name__ == "__main__":
    main()
