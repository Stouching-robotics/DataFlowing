#!/usr/bin/env python3
"""Test script: run MediaPipeHandPipeline on a video with smoothing,
render keypoints + angles, save output video.

Usage:
    python hand_detection/test_smoothing.py
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

# Allow importing from hand_detection/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hand_detection.hand_pipeline_mediapipe import (
    MediaPipeHandPipeline,
    FINGERS,
    JOINT_SPECS,
)
from hand_detection.hand_common import draw_hand

# ── Config ─────────────────────────────────────────────────

VIDEO_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data/recordings/Project_812/Project_812_000005/videos/head_left_rgb/chunk-0000/head_left_rgb.mp4",
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "keypoints_output")
OUTPUT_VIDEO_TMP = os.path.join(OUTPUT_DIR, "smoothing_comparison_tmp.mp4")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "smoothing_comparison.mp4")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models/hand_landmarker.task")

# ── Drawing helpers ────────────────────────────────────────


def draw_landmarks_simple(frame, pts, color=(0, 255, 0), label=""):
    """Draw 21 keypoints + finger connections with simple style (no angles)."""
    p = [(int(x), int(y)) for x, y in pts[:21]]

    # Palm connections
    palm_conn = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
    for a, b in palm_conn:
        cv2.line(frame, p[a], p[b], (180, 180, 180), 1, cv2.LINE_AA)

    # Finger connections
    finger_colors = {
        "Thumb": (255, 128, 0),
        "Index": (0, 255, 0),
        "Middle": (0, 255, 255),
        "Ring": (255, 0, 255),
        "Pinky": (0, 128, 255),
    }
    for finger, (ids, fcolor) in FINGERS.items():
        chain = ids if finger == "Thumb" else [0] + ids
        for i in range(len(chain) - 1):
            cv2.line(frame, p[chain[i]], p[chain[i + 1]], fcolor, 2, cv2.LINE_AA)
        for idx in ids:
            r = 5 if idx == ids[-1] else 3
            cv2.circle(frame, p[idx], r, fcolor, -1, cv2.LINE_AA)

    # Wrist
    cv2.circle(frame, p[0], 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, p[0], 7, (40, 40, 40), 1, cv2.LINE_AA)

    if label:
        cv2.putText(frame, label, p[0], cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)


def draw_info_bar(frame, frame_idx, total, fps, hand_count):
    """Draw top info bar."""
    h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    texts = [
        f"Frame: {frame_idx}/{total}",
        f"FPS: {fps:.1f}",
        f"Hands: {hand_count}",
        "LEFT=Raw  |  RIGHT=Smoothed",
    ]
    x = 10
    for txt in texts:
        cv2.putText(frame, txt, (x, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        x += 200


def raw_pixel_coords(lms, w, h):
    """Convert raw normalized landmarks to pixel coords (no filtering)."""
    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in lms], dtype=np.float32)


# ── Main ───────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w_in}x{h_in} @ {fps_in:.1f} fps, {total_frames} frames")

    # Init pipeline with smoothing
    pipe = MediaPipeHandPipeline(
        model_path=MODEL_PATH,
        num_hands=2,
        det_conf=0.5,
        track_conf=0.5,
        preprocess_mode="none",
        mirror=False,       # head-mounted camera, no mirror needed
        smooth=True,
        freq_min=5.0,
        beta=0.05,
        dcutoff=1.0,
    )

    # Output video: side-by-side (2x width)
    out_w = w_in * 2
    out_h = h_in
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_TMP, fourcc, fps_in, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video: {OUTPUT_VIDEO_TMP}")
        sys.exit(1)

    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Resolution: {out_w}x{out_h} (side-by-side)")
    print("Processing...")

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipe.process(frame)

        # Build side-by-side
        # Left: raw (from raw_landmarks, no filtering)
        # Right: smoothed (from result.hands)
        left = frame.copy()
        right = frame.copy()

        # Left side — raw landmarks
        for i, lms in enumerate(result.raw_landmarks):
            pts = raw_pixel_coords(lms, w_in, h_in)
            label = ""
            if i < len(result.raw_handedness) and result.raw_handedness[i]:
                label = result.raw_handedness[i][0].category_name
            draw_landmarks_simple(left, pts, label=f"RAW {label}")

        # Right side — smoothed landmarks + angles
        for hand in result.hands:
            draw_hand(right, hand.landmarks, angles=hand.angles, show_angles=True)
            # Draw hand label + extended fingers
            wrist = (int(hand.landmarks[0][0]), int(hand.landmarks[0][1]))
            info = f"{hand.label} ({hand.score:.2f})"
            if hand.extended:
                info += " | " + " ".join(hand.extended[:2])
            cv2.putText(right, info, (wrist[0], wrist[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Combine
        side_by_side = np.hstack([left, right])
        draw_info_bar(side_by_side, frame_idx, total_frames,
                      fps_in, len(result.hands))

        writer.write(side_by_side)
        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - t_start
            fps_proc = frame_idx / elapsed
            print(f"  Frame {frame_idx}/{total_frames} | "
                  f"{fps_proc:.1f} fps processing | "
                  f"hands: {len(result.hands)}")

    # Cleanup
    elapsed = time.perf_counter() - t_start
    cap.release()
    writer.release()
    pipe.close()

    print(f"\nDone! {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps avg)")

    # ── 转码为 H.264（浏览器/播放器兼容）──
    import subprocess
    print("Converting to H.264...")
    ret = subprocess.run([
        "ffmpeg", "-y",
        "-i", OUTPUT_VIDEO_TMP,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        OUTPUT_VIDEO,
    ], capture_output=True, text=True)
    if ret.returncode == 0:
        os.remove(OUTPUT_VIDEO_TMP)
        print(f"Output saved to: {OUTPUT_VIDEO}")
    else:
        # ffmpeg 失败时保留原始 mp4v 文件
        print(f"ffmpeg failed, keeping raw mp4v: {OUTPUT_VIDEO_TMP}")
        print(ret.stderr[-500:])
    print(f"  LEFT  side: raw (no smoothing)")
    print(f"  RIGHT side: One-Euro smoothed + joint angles")


if __name__ == "__main__":
    main()
