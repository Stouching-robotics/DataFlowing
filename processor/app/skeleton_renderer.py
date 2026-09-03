"""Hand skeleton video pre-rendering using Google MediaPipe drawing_utils.

Reads a video + parquet hand keypoints, renders skeletons onto every frame
using official MediaPipe styling, and writes a new MP4 alongside the original.

Usage::

    from app.skeleton_renderer import render_skeleton_video
    output = await render_skeleton_video(video_path, session_dir)
    # → video_path.parent / "video_skeleton.mp4"  (or None if no hand data)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _has_hand_keypoints(parquet_path: Path) -> bool:
    """Check whether a merged data parquet has hand keypoint columns with data."""
    try:
        df = pd.read_parquet(parquet_path, columns=["hand_0_keypoints", "hand_1_keypoints"])
    except (ValueError, Exception):
        return False

    for col in ("hand_0_keypoints", "hand_1_keypoints"):
        if col in df.columns:
            has_data = df[col].notna().any()
            if has_data:
                return True
    return False


def _find_data_parquet(session_dir: Path) -> Path | None:
    """Find the merged data parquet in a session directory."""
    for d in session_dir.rglob("data"):
        if not d.is_dir():
            continue
        parq_files = sorted(
            p for p in d.rglob("chunk_*.parquet")
            if "meta" not in str(p)
            and not p.name.startswith("auto_labels_")
            and not p.name.startswith("hand_kpts_")
        )
        if parq_files:
            return parq_files[0]
    return None


def _build_keypoint_lookup(parquet_path: Path) -> dict[int, dict]:
    """Build a frame_index → hand keypoints lookup dict.

    Returns dict like::

        {0: {"h0": [[x,y]*21], "h1": [[x,y]*21]}, 30: {"h0": [...], "h1": None}, ...}
    """
    df = pd.read_parquet(parquet_path)
    lookup: dict[int, dict] = {}

    has_h0 = "hand_0_keypoints" in df.columns
    has_h1 = "hand_1_keypoints" in df.columns

    if not has_h0 and not has_h1:
        return lookup

    for _, row in df.iterrows():
        fi = int(row["frame_index"])
        entry: dict = {}
        for hk, col in (("h0", "hand_0_keypoints"), ("h1", "hand_1_keypoints")):
            if col not in df.columns:
                continue
            val = row[col]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                entry[hk] = None
                continue
            kp_list = val.tolist() if hasattr(val, "tolist") else val
            if isinstance(kp_list, list) and len(kp_list) >= 21:
                entry[hk] = [[float(p[0]), float(p[1])] for p in kp_list[:21]]
            else:
                entry[hk] = None
        lookup[fi] = entry

    return lookup


# ═══════════════════════════════════════════════════════════════
# Drawing
# ═══════════════════════════════════════════════════════════════

def _get_drawing_func():
    """Return the best available drawing function.

    Tries Google MediaPipe official drawing_utils first,
    falls back to OpenCV-based rendering.
    """
    try:
        from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
        from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
        from mediapipe.tasks.python.components.containers import landmark as lm

        def _draw_mediapipe(frame_bgr, kp_lookup, frame_index):
            """Draw hand skeletons on a BGR frame using Google MediaPipe official API."""
            entry = kp_lookup.get(frame_index)
            if not entry:
                return

            W, H = 640, 480  # source resolution

            for hk in ("h0", "h1"):
                kp = entry.get(hk)
                if not kp or len(kp) < 21:
                    continue

                landmarks = []
                for pt in kp[:21]:
                    l = lm.NormalizedLandmark()
                    l.x = pt[0] / W
                    l.y = pt[1] / H
                    l.z = 0.0
                    landmarks.append(l)

                drawing_utils.draw_landmarks(
                    frame_bgr,
                    landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

        return _draw_mediapipe, "mediapipe"

    except ImportError:
        pass

    # ── OpenCV fallback ──
    def _draw_opencv(frame_bgr, kp_lookup, frame_index):
        """Draw hand skeletons using OpenCV (near-MediaPipe replica)."""
        entry = kp_lookup.get(frame_index)
        if not entry:
            return

        import cv2

        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
        ]
        # BGR colours per finger (matches MediaPipe style)
        finger_colors = [
            (255, 48, 48),   # thumb  → red
            (128, 64, 128),  # index  → purple
            (48, 255, 255),  # middle → yellow
            (48, 255, 48),   # ring   → green
            (48, 208, 255),  # pinky  → orange
        ]

        for hk in ("h0", "h1"):
            kp = entry.get(hk)
            if not kp or len(kp) < 21:
                continue
            pts = [(int(p[0]), int(p[1])) for p in kp[:21]]

            # Palm connections (gray)
            for i, j in [(0, 1), (0, 17), (1, 5), (5, 9), (9, 13), (13, 17)]:
                cv2.line(frame_bgr, pts[i], pts[j], (128, 128, 128), 3, cv2.LINE_AA)

            # Finger connections (colored)
            for f in range(5):
                c = finger_colors[f]
                base = f * 4 + 1
                for i in range(3):
                    cv2.line(frame_bgr, pts[base + i], pts[base + i + 1],
                             c, 2, cv2.LINE_AA)

            # Landmark dots: white border + colored fill
            radii = [7, 4, 4, 3, 2, 4, 4, 3, 2, 4, 4, 3, 2,
                     4, 4, 3, 2, 4, 4, 3, 2]
            for i, (x, y) in enumerate(pts):
                if i == 0:
                    fc = (255, 255, 255)  # wrist white
                else:
                    fc = finger_colors[(i - 1) // 4]
                r = radii[i]
                cv2.circle(frame_bgr, (x, y), r + 1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (x, y), max(1, r - 1), fc, -1, cv2.LINE_AA)

    return _draw_opencv, "opencv"


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

async def render_skeleton_video(
    video_path: Path,
    session_dir: Path,
) -> Path | None:
    """Render hand skeletons onto a video, returning the output path.

    Args:
        video_path: Path to the source MP4 file.
        session_dir: Session root directory (contains data/ with parquet).

    Returns:
        Path to the skeleton video, or ``None`` if no hand data exists.
    """
    import cv2

    # 1. Find and validate hand data
    parquet_path = _find_data_parquet(session_dir)
    if parquet_path is None:
        logger.info("No data parquet found in %s — skipping skeleton render", session_dir)
        return None

    if not _has_hand_keypoints(parquet_path):
        logger.info("No hand keypoint columns in %s — skipping skeleton render", parquet_path.name)
        return None

    # 2. Build keypoint lookup (frame_index → keypoints)
    kp_lookup = _build_keypoint_lookup(parquet_path)
    keypoint_frames = sum(1 for v in kp_lookup.values() if v.get("h0") or v.get("h1"))
    if keypoint_frames == 0:
        logger.info("No frames with hand keypoints — skipping skeleton render")
        return None

    logger.info(
        "Rendering skeleton video: %d / %d total rows have hand keypoints",
        keypoint_frames, len(kp_lookup),
    )

    # 3. Get drawing function
    draw_func, backend = _get_drawing_func()
    logger.info("Using drawing backend: %s", backend)

    # 4. Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("Video: %dx%d @ %.1f fps, %d frames", width, height, fps, total_frames)

    # 5. Output path: {stem}_skeleton{ext}
    output_path = video_path.parent / f"{video_path.stem}_skeleton{video_path.suffix}"

    # 6. FFmpeg H.264 encoder (browser-compatible)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = width * height * 3  # BGR24 = 3 bytes per pixel

    # 7. Process frames
    frame_idx = 0
    rendered_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_func(frame, kp_lookup, frame_idx)
        proc.stdin.write(frame.tobytes())

        if kp_lookup.get(frame_idx):
            rendered_frames += 1

        frame_idx += 1
        if frame_idx % 50 == 0:
            logger.debug("Skeleton render: %d / %d frames", frame_idx, total_frames)

    cap.release()
    proc.stdin.close()
    proc.wait()

    logger.info(
        "Skeleton video written: %s (%d frames rendered, %d total)",
        output_path.name, rendered_frames, frame_idx,
    )
    return output_path
