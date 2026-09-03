"""RGB_TO_2D_BareHand — RGB-only hand landmarks with a display-only 3D view.

This is a standalone workflow module for ordinary RGB recordings.  It does
not call the Hand Skeleton module and does not require a depth stream.  The
pipeline is:

    RGB frame -> MediaPipe HandLandmarker 2D landmarks
              -> RGBWorldTracker PnP/image-scale placement
              -> camera-relative estimated 3D preview artifact

The output follows the Hand Skeleton Hand3D parquet contract.  Because depth
is not observed, the result is explicitly labelled ``rgb_estimated_meters``
and must not be treated as metric depth ground truth.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.processing import ArtifactRef, JobContext, ProcessingModule, field
from app.processing.hand_render import draw_demo_style
from app.processing.hand_slot_assignment import (
    StableHandSlotAssigner,
    clip_normalized_xy,
)
from app.processing.registry import register
from app.processing.theme import HAND3D_COLOR
from app.processing.modules.rgb_world_tracking import RGBWorldTracker


_BLACK_GLOVE_ROOT = Path(__file__).resolve().parents[1] / "black_glove"


def _resolve_model() -> Path:
    candidates = (
        Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task",
        _BLACK_GLOVE_ROOT / "hand_landmarker.task",
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"MediaPipe hand model not found; searched: {searched}")


HAND_LANDMARKER_MODEL = _resolve_model()


def _angle(a, vertex, b) -> float:
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    v2 = np.asarray(b, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _gesture(local: np.ndarray) -> tuple[str, int]:
    """Return a stable lightweight finger-extension label and bitmask."""
    chains = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    bits = {name: 1 << i for i, name in enumerate(chains)}
    extended = []
    for name, ids in chains.items():
        if name == "thumb":
            ok = _angle(local[1], local[2], local[3]) > 145
            ok = ok and _angle(local[2], local[3], local[4]) > 150
        else:
            ok = _angle(local[ids[0]], local[ids[1]], local[ids[2]]) > 150
            ok = ok and _angle(local[ids[1]], local[ids[2]], local[ids[3]]) > 140
        if ok:
            extended.append(name)
    mask = sum(bits[name] for name in extended)
    return ("fist" if not extended else "open:" + ",".join(extended)), mask


class _PointSmoother:
    """Small per-slot 2D One-Euro smoother for the overlay/keypoint stream."""

    def __init__(self, freq_min: float, beta: float):
        self.freq_min = max(0.1, float(freq_min))
        self.beta = max(0.0, float(beta))
        self.prev: dict[tuple[int, int], np.ndarray] = {}
        self.prev_d: dict[tuple[int, int], np.ndarray] = {}
        # Each hand slot is updated in sequence for a frame.  Keeping one
        # timestamp for the whole smoother makes slot 1 see dt≈0 because
        # slot 0 has already advanced the timestamp; the second hand then
        # effectively freezes whenever smoothing is enabled.
        self.prev_ts: dict[int, float] = {}

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(1e-9, cutoff))
        return float(dt / (dt + tau))

    def update(self, slot: int, points: np.ndarray, timestamp: float) -> np.ndarray:
        previous_timestamp = self.prev_ts.get(slot, timestamp)
        dt = max(1e-5, timestamp - previous_timestamp)
        out = points.astype(np.float64, copy=True)
        for index, point in enumerate(out):
            key = (slot, index)
            old = self.prev.get(key)
            old_d = self.prev_d.get(key, np.zeros(3, dtype=np.float64))
            if old is None:
                self.prev[key] = point.copy()
                self.prev_d[key] = np.zeros(3, dtype=np.float64)
                continue
            derivative = (point - old) / dt
            alpha_d = self._alpha(1.0, dt)
            filtered_d = alpha_d * derivative + (1.0 - alpha_d) * old_d
            cutoff = self.freq_min + self.beta * np.abs(filtered_d)
            alpha = np.array([self._alpha(float(value), dt) for value in cutoff])
            filtered = alpha * point + (1.0 - alpha) * old
            out[index] = filtered
            self.prev[key] = filtered.copy()
            self.prev_d[key] = filtered_d
        self.prev_ts[slot] = timestamp
        return out.astype(np.float32)


def _create_landmarker(config: dict[str, Any]):
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    max_hands = max(1, min(2, int(config.get("max_hands", 2))))
    detection = float(config.get("min_detection_conf", 0.1))
    presence = float(config.get("min_presence_conf", 0.1))
    tracking = float(config.get("min_tracking_conf", 0.5))
    device = str(config.get("device", "auto"))
    options = dict(
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=detection,
        min_hand_presence_confidence=presence,
        min_tracking_confidence=tracking,
    )
    if device in ("auto", "cuda:0"):
        try:
            return vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(HAND_LANDMARKER_MODEL),
                        delegate=BaseOptions.Delegate.GPU),
                    **options))
        except Exception as exc:
            print(f"[rgb_hand_3d] GPU unavailable, fallback CPU: {exc}")
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
            **options))


def _write_h264(source: Path, destination: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return source
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-g", "25", "-keyint_min", "25", "-sc_threshold", "0",
             "-movflags", "+faststart", "-an", str(destination)],
            check=True, capture_output=True, timeout=3600,
        )
        if destination.exists() and destination.stat().st_size > 0:
            source.unlink(missing_ok=True)
            return destination
    except Exception:
        pass
    return source


def _flat(value, size: int) -> list[float] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array.tolist() if array.size == size else None


def _detect_video(video_path: Path, config: dict[str, Any], output_video: Path,
                  progress: Callable[[float], None] | None) -> dict[str, Any]:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create RGB Hand3D video: {output_video}")

    landmarker = _create_landmarker(config)
    preview_3d = bool(config.get("preview_3d", True))
    tracker = (RGBWorldTracker(
        width, height,
        freq_min=float(config.get("freq_min", 5.0)),
        beta=float(config.get("beta", 0.05)),
    ) if preview_3d else None)
    slot_assigner = StableHandSlotAssigner(2)
    smoother = (_PointSmoother(
        float(config.get("freq_min", 5.0)), float(config.get("beta", 0.05)))
        if bool(config.get("smooth", True)) else None)
    rows: list[dict[str, Any]] = []
    frame_index = 0
    timestamp_ms = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
            timestamp_ms += int(1000.0 / max(1.0, fps))
            row: dict[str, Any] = {"frame_index": frame_index}
            candidates = []
            for detected_index, landmarks in enumerate(result.hand_landmarks[:2]):
                raw_label = (
                    str(result.handedness[detected_index][0].category_name)
                    if detected_index < len(result.handedness)
                    and result.handedness[detected_index] else "")
                score = (float(result.handedness[detected_index][0].score)
                         if detected_index < len(result.handedness)
                         and result.handedness[detected_index] else 0.0)
                points = np.asarray(
                    [[float(p.x), float(p.y), float(p.z)] for p in landmarks[:21]],
                    dtype=np.float32)
                candidates.append({
                    "label": raw_label,
                    "center": np.mean(points[:, :2], axis=0),
                    "payload": (points, detected_index, raw_label, score),
                })
            ordered = slot_assigner.assign(candidates)
            for slot in range(2):
                candidate = ordered[slot]
                if candidate is None:
                    row.update({
                        f"hand_{slot}_keypoints": None,
                        f"hand_{slot}_world_landmarks": None,
                        f"hand_{slot}_landmarks_3d_local": None,
                        f"hand_{slot}_world_position": None,
                        f"hand_{slot}_coordinate_frame": None,
                        f"hand_{slot}_tracking_source": None,
                        f"hand_{slot}_depth_source": None,
                        f"hand_{slot}_reprojection_error": None,
                        f"hand_{slot}_handedness": None,
                        f"hand_{slot}_confidence": None,
                        f"hand_{slot}_gesture": "",
                        f"hand_{slot}_fingers": -1,
                    })
                    continue
                points, detected_index, raw_handedness, confidence = \
                    candidate["payload"]
                if smoother is not None:
                    points = smoother.update(
                        slot, points, frame_index / max(1.0, float(fps)))
                points = clip_normalized_xy(points)
                # Keep the semantic label latched to the tracked image slot;
                # a transient detector flip must not swap the two hands.
                handedness = slot_assigner.stable_label(
                    slot, str(candidate.get("label") or raw_handedness))
                # Match Black Hand RGB3D's stable RGB-only local model. Do
                # not mix MediaPipe world-landmark axes into this workflow;
                # they use a different orientation convention.
                local = (RGBWorldTracker.black_glove_reference_local(points)
                         if preview_3d else None)
                local = (RGBWorldTracker.orient_palm_facing(local, handedness)
                         if local is not None else None)
                # RGB has no measured depth. Keep the detected hand model's
                # local geometry for every frame; PnP is used only to estimate
                # the wrist's camera-relative position. Applying its per-frame
                # rotation makes the fixed-root preview visibly jump.
                preserve_model_geometry = True
                tracking = tracker.update(
                    f"slot_{slot}", points, local,
                    frame_index / max(1.0, float(fps)),
                    preserve_model_geometry=preserve_model_geometry
                ) if local is not None else None
                gesture, fingers = _gesture(local) if local is not None else ("", -1)
                row.update({
                    f"hand_{slot}_keypoints": points.tolist(),
                    f"hand_{slot}_world_landmarks": (
                        tracking["landmarks_3d"].tolist() if tracking else None),
                    f"hand_{slot}_landmarks_3d_local": (
                        tracking["landmarks_3d_local"].tolist() if tracking else
                        (local.tolist() if local is not None else None)),
                    f"hand_{slot}_world_position": (
                        tracking["position"].tolist() if tracking else None),
                    f"hand_{slot}_coordinate_frame": (
                        tracking["coordinate_frame"] if tracking else None),
                    f"hand_{slot}_tracking_source": (
                        tracking["tracking_source"] if tracking else None),
                    f"hand_{slot}_depth_source": "rgb_estimate" if tracking else None,
                    f"hand_{slot}_reprojection_error": (
                        tracking["reprojection_error"] if tracking else None),
                    f"hand_{slot}_handedness": handedness,
                    f"hand_{slot}_confidence": confidence,
                    f"hand_{slot}_gesture": gesture,
                    f"hand_{slot}_fingers": fingers,
                })
                pixels = (points[:, :2] * [width, height]).tolist()
                draw_demo_style(frame, pixels)
                cv2.putText(frame,
                            f"RGB {'3D' if preview_3d else '2D'} {handedness} {confidence:.2f}",
                            (max(0, int(points[:, 0].min() * width)),
                             max(18, int(points[:, 1].min() * height) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1,
                            cv2.LINE_AA)
            rows.append(row)
            writer.write(frame)
            frame_index += 1
            if progress and total:
                progress(min(0.98, frame_index / total))
    finally:
        cap.release()
        writer.release()
        landmarker.close()
    if not rows:
        raise RuntimeError(f"No frames in video: {video_path}")
    return {"rows": rows, "video": _write_h264(
        output_video, output_video.with_name(output_video.stem + "_h264.mp4")),
            "fps": fps, "width": width, "height": height}


def _canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = {"frame_index": int(source["frame_index"])}
        for slot in (0, 1):
            prefix = f"hand_{slot}"
            keypoints = _flat(source.get(f"{prefix}_keypoints"), 63)
            landmarks = _flat(source.get(f"{prefix}_world_landmarks"), 63)
            local = _flat(source.get(f"{prefix}_landmarks_3d_local"), 63)
            position = _flat(source.get(f"{prefix}_world_position"), 3)
            present = landmarks is not None
            row.update({
                f"{prefix}_present": present,
                f"{prefix}_2d_present": keypoints is not None,
                f"{prefix}_landmarks_3d": landmarks or [float("nan")] * 63,
                f"{prefix}_landmarks_3d_local": local or [float("nan")] * 63,
                f"{prefix}_world_position": position or [float("nan")] * 3,
                f"{prefix}_coordinate_frame": source.get(
                    f"{prefix}_coordinate_frame") or "",
                f"{prefix}_tracking_source": source.get(
                    f"{prefix}_tracking_source") or "",
                f"{prefix}_depth_source": source.get(
                    f"{prefix}_depth_source") or "",
                f"{prefix}_keypoints": keypoints or [float("nan")] * 63,
                f"{prefix}_label": source.get(f"{prefix}_handedness") or "",
                f"{prefix}_reprojection_error": source.get(
                    f"{prefix}_reprojection_error"),
                f"{prefix}_gesture": source.get(f"{prefix}_gesture") or "",
                f"{prefix}_fingers": int(source.get(f"{prefix}_fingers") or -1),
                f"{prefix}_confidence": float(
                    source.get(f"{prefix}_confidence") or float("nan")),
            })
        output.append(row)
    return output


@register
class RGBHand3DModule(ProcessingModule):
    slug = "rgb_to_2d_bare_hand"
    version = "1.0"
    category = "process"
    label = "RGB_TO_2D_BareHand"
    icon = "ant-design:deployment-unit-outlined"
    color = HAND3D_COLOR
    inputs = ({"key": "video", "label": "RGB Video"},)
    # RGB has no metric depth. The hand_3d parquet remains an internal
    # display artifact; the workflow contract exposes only 2D keypoints.
    outputs = ({"key": "hand_keypoints", "label": "Hand 2D"},)
    default_config = {
        "max_hands": 2, "min_detection_conf": 0.1,
        "min_presence_conf": 0.1, "min_tracking_conf": 0.5,
        "device": "auto", "smooth": True, "freq_min": 5.0, "beta": 0.05,
        "preview_3d": True,
    }
    config_schema = (
        field("max_hands", "number", "Max hands", 2, min=1, max=2),
        field("min_detection_conf", "number", "Detection confidence", 0.1,
              min=0, max=1, step=0.05),
        field("min_presence_conf", "number", "Presence confidence", 0.1,
              min=0, max=1, step=0.05),
        field("min_tracking_conf", "number", "Tracking confidence", 0.5,
              min=0, max=1, step=0.05),
        field("device", "select", "Device", "auto",
              options=["auto", "cpu", "cuda:0"]),
        field("smooth", "boolean", "One-Euro smoothing", True),
        field("preview_3d", "boolean", "Display-only 3D preview", True),
        field("freq_min", "number", "Smoothing cutoff Hz", 5.0,
              min=1, max=60, step=1),
        field("beta", "number", "Smoothing speed coefficient", 0.05,
              min=0, max=2, step=0.05),
    )
    execution_target = "worker"
    capabilities = ("hand_keypoints", "rgb", "rgb_preview_3d", "video_overlay",
                    "camera_relative", "no_depth")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        import pandas as pd

        video_refs = [ref for ref in ctx.incoming.values() if ref.kind == "video"]
        if not video_refs:
            ctx.skip("No RGB video input — skipped")
        video_refs.sort(key=lambda ref: (ref.source_key or "", ref.path or ""))
        output_root = ctx.output_root / "hand_3d"
        render_root = ctx.output_root / "skeleton"
        output_root.mkdir(parents=True, exist_ok=True)
        results = []
        for index, ref in enumerate(video_refs[:2]):
            path = ctx.resolve(ref)
            if not path or not path.exists():
                ctx.skip(f"RGB video artifact missing: {ref.path}")
            source = str(ref.source_key or path.stem)
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_"
                           for ch in source)
            render_path = render_root / f"rgb_hand_3d_{safe}.mp4"
            result = _detect_video(
                path, ctx.config, render_path,
                lambda value, i=index, n=min(2, len(video_refs)):
                ctx.progress((i + value) / n))
            canonical = _canonical_rows(result["rows"])
            if index == 0:
                parquet_path = output_root / "hand_world.parquet"
                manifest_path = output_root / "hand_world.manifest.json"
            else:
                right_root = ctx.output_root / "hand_3d_right"
                right_root.mkdir(parents=True, exist_ok=True)
                parquet_path = right_root / "hand_world_right.parquet"
                manifest_path = right_root / "hand_world_right.manifest.json"
            pd.DataFrame(canonical).to_parquet(parquet_path, index=False)
            preview_3d = bool(ctx.config.get("preview_3d", True))
            manifest = {
                "frames": len(canonical),
                "source_key": source,
                "views": min(2, len(video_refs)),
                "view": "left" if index == 0 else "right",
                "detector": "hand_landmarker",
                "mode": "rgb_estimated_3d" if preview_3d else "rgb_2d",
                "unit": "rgb_estimated_meters" if preview_3d else "image_normalized",
                "preview_3d": preview_3d,
                "coordinate_frame": "camera_relative" if preview_3d else "image_normalized",
                "metric_3d_available": False,
                "method": ("MediaPipe RGB landmarks + hand-model PnP/image-scale "
                           "wrist placement; stable local hand geometry"),
                "orientation_mode": "stable_local_model_wrist_pnp_only",
                "camera_model": "approximate_pinhole",
                "thresholds": {
                    "min_detection_conf": float(ctx.config.get("min_detection_conf", 0.1)),
                    "min_presence_conf": float(ctx.config.get("min_presence_conf", 0.1)),
                    "min_tracking_conf": float(ctx.config.get("min_tracking_conf", 0.5)),
                },
                "smoothing": {
                    "enabled": bool(ctx.config.get("smooth", True)),
                    "freq_min": float(ctx.config.get("freq_min", 5.0)),
                    "beta": float(ctx.config.get("beta", 0.05)),
                },
                "render_video": str(result["video"]),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            results.append((ref, parquet_path, result["video"], manifest))

        primary_ref, primary_parquet, primary_video, primary_manifest = results[0]
        if ctx.progress:
            ctx.progress(1.0)
        outputs = {
            "hand_keypoints": ctx.ref("hand_keypoints", primary_parquet,
                                       source_key=primary_ref.source_key,
                                       metadata={**primary_manifest,
                                                 "public_contract": "2d"}),
            "render_video": ctx.ref("video", primary_video,
                                    source_key=primary_ref.source_key,
                                    metadata={"skeleton": True,
                                              "rgb_estimated_3d": preview_3d}),
        }
        # Expose the second RGB view as a real workflow artifact.  The file
        # was already written above; without this handle, static export
        # discovery cannot include the right-camera feature group.
        if len(results) > 1:
            right_ref, right_parquet, _right_video, right_manifest = results[1]
            outputs["hand_keypoints#right"] = ctx.ref(
                "hand_keypoints", right_parquet,
                source_key=right_ref.source_key,
                metadata={**right_manifest, "public_contract": "2d"})
        return outputs
