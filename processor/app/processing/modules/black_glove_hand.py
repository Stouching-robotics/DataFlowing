"""Black-glove hand skeleton module.

This module is deliberately separate from ``RGB_TO_2D_BlackGlove``.  Every
video input is processed independently by the black-glove detector:

* mono: one RGB video -> one 2D keypoint parquet + skeleton video;
* stereo: left and right RGB videos -> two independent outputs and an
  optional side-by-side preview;
* depth, when compatible depth streams exist, is sampled from the matching
  RGB view and lifted into that camera's coordinates, with one
  ``hand_3d/<source>.parquet`` artifact per device.

This workflow is the depth-based ``RGB-D_3D_BlackGlove`` workflow. The
separate ``RGB_TO_2D_BlackGlove`` module owns RGB-only estimated preview data.

There is no stereo triangulation.  Left/right output slots are kept by the
detector track id and the D435 3D stage adds label-aware fixed-slot state;
there is no cross-camera matching.  The detector implementation and D435
tracking helper are bundled under ``app/processing/black_glove``; this
workflow adapter owns the artifact contract, video rendering, and depth path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.processing import ArtifactRef, JobContext, ProcessingModule, field
from app.processing.hand_render import draw_demo_style, hand_style_scale
from app.processing.black_glove.d435_tracking import (
    BAND_HALF_M, BAND_MIN_VALID, AlphaBetaHandSlots, gate_observations,
)
from app.lerobot_v21 import DepthVideoReader
from app.processing.registry import register
from app.processing.theme import HAND3D_COLOR


_BLACK_GLOVE_ROOT = Path(__file__).resolve().parents[1] / "black_glove"
_DEFAULT_WEIGHTS = _BLACK_GLOVE_ROOT / "weights" / "yolov8m-worldv2.pt"


def _safe_key(value: str | None, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or fallback))


def _load_glove_detector():
    """Load the bundled detector lazily while keeping worker startup light."""
    from app.processing.black_glove.glove_detector import GloveDetector
    return GloveDetector


def _detector_config(config: dict[str, Any]) -> dict[str, Any]:
    # Existing workflow graphs may already contain the first adapter's
    # defaults (0.5/1/0.2/10/0.2). Migrate that exact untouched bundle so a
    # user does not have to rebuild every saved workflow. A partially custom
    # configuration is respected field-by-field below.
    legacy_defaults = (
        float(config.get("movement_thresh", float("nan"))) == 0.5 and
        int(config.get("skip_timeout", -1)) == 1 and
        float(config.get("box_alpha", float("nan"))) == 0.2 and
        float(config.get("freq_min", float("nan"))) == 10.0 and
        float(config.get("beta", float("nan"))) == 0.2
    )
    if legacy_defaults:
        config = {
            **config,
            "movement_thresh": 1.5,
            "skip_timeout": 3,
            "box_alpha": 0.7,
            "freq_min": 5.0,
            "beta": 0.05,
        }
    # The previous stability profile was too aggressive for black gloves:
    # it rejected low-confidence but usable hands and froze valid fast
    # motions. Normalize that exact profile back to the high-recall settings,
    # including already-saved workflow graphs.
    over_filtered_profile = (
        abs(float(config.get("det_conf", 0.08)) - 0.08) < 1e-6
        and abs(float(config.get("movement_thresh", 2.0)) - 2.0) < 1e-6
        and int(config.get("skip_timeout", 4)) == 4
        and abs(float(config.get("pose_conf_thr", 0.4) or 0.4) - 0.4) < 1e-6
        and config.get("hold_translate", False) is False
    )
    if over_filtered_profile:
        config = {
            **config,
            "det_conf": 0.05,
            "movement_thresh": 1.5,
            "skip_timeout": 3,
            "freeze_max": 15,
            "pose_conf_thr": 0.15,
            "hold_translate": True,
        }
    # 2026-08-26 远距离/手背修复档：旧默认 bundle（imgsz 320 +
    # new_track_conf 0.25 + pose_conf_thr 0.3 且未设 lost_timeout）
    # 整体升到远距离档，与 tools(1)(1) hand_3d_s80c 实测参数一致；
    # 部分自定义的配置按字段保留不动。兼容两代旧快照：8-24 更早的
    # 保存配置没有 new_track_conf/pose_conf_thr 键（缺键=旧默认，
    # 直接升级）；8-26 前的完整默认 bundle 按显式旧值识别。
    legacy_far_defaults = (
        int(config.get("imgsz", 640)) == 320
        and config.get("lost_timeout") is None
        and (
            ("new_track_conf" not in config
             and "pose_conf_thr" not in config)
            or (abs(float(config.get("new_track_conf", 0.1)) - 0.25) < 1e-6
                and abs(float(config.get("pose_conf_thr", 0.15) or 0.15)
                        - 0.3) < 1e-6)
        )
    )
    if legacy_far_defaults:
        config = {
            **config,
            "imgsz": 640,
            "new_track_conf": 0.1,
            "pose_conf_thr": 0.15,
            "lost_timeout": 8,
        }
    # 运动门控旧默认对（3px/10 帧）：EMA 平滑框只跟手速 ~30%，3px 门控
    # 在快动时几乎不触发、慢动时又长期冻结，实测造成"骨架跟手延迟感"。
    # 旧默认组合整体收紧为 1.5px/3 帧（000003 快扫段实测有效）。
    legacy_gate_defaults = (
        abs(float(config.get("movement_thresh", 1.5)) - 3.0) < 1e-6
        and int(config.get("skip_timeout", 3)) == 10
    )
    if legacy_gate_defaults:
        config = {
            **config,
            "movement_thresh": 1.5,
            "skip_timeout": 3,
        }
    weights = Path(str(config.get("weights_path") or _DEFAULT_WEIGHTS))
    pose_conf_thr = config.get("pose_conf_thr", 0.15)
    match_contain_thr = config.get("match_contain_thr", 0.7)
    return {
        "weights": str(weights.expanduser()),
        "num_hands": max(1, min(2, int(config.get("max_hands", 2)))),
        "device": str(config.get("device", "auto")),
        "pose_device": str(config.get("pose_device", "auto")),
        "det_conf": float(config.get("det_conf", 0.05)),
        "imgsz": int(config.get("imgsz", 640)),
        "use_tracker": bool(config.get("use_tracker", True)),
        # Keep the detector defaults in sync with the validated standalone
        # black-glove pipeline.  Refreshing RTMPose on every tiny box wobble
        # makes the output noisier; the tracker should gate pose inference and
        # retain the last good pose briefly when a detector box is missed.
        "movement_thresh": max(0.0, float(config.get("movement_thresh", 1.5))),
        "skip_timeout": max(1, int(config.get("skip_timeout", 3))),
        "box_alpha": min(1.0, max(0.0, float(config.get("box_alpha", 0.7)))),
        "use_oe": bool(config.get("smooth", True)),
        "oe_freq_min": float(config.get("freq_min", 5.0)),
        "oe_beta": float(config.get("beta", 0.05)),
        "freeze_max": int(config.get("freeze_max", 15)),
        "pose_conf_thr": (None if pose_conf_thr is None
                           or float(pose_conf_thr) <= 0
                           else float(pose_conf_thr)),
        "pose_backend": str(config.get("pose_backend", "rtmpose")),
        "pose_model": (str(config.get("pose_model"))
                       if config.get("pose_model") else None),
        "pose_box_raw": bool(config.get("pose_box_raw", False)),
        "hold_translate": bool(config.get("hold_translate", True)),
        # 低置信闪框不能创建新轨迹；已有轨迹仍允许用低置信框维持。
        # 远距离档（S80C 实测）：0.1 而非 0.25——离线统计 93% 的
        # world 框 conf<0.25，旧门会挡死远手重捕获。
        "new_track_conf": max(0.0, float(config.get("new_track_conf", 0.1))),
        # 丢失容忍（远距离档 8 帧：远手/手背框闪烁时 3 帧即死会持续丢手）。
        "lost_timeout": max(1, int(config.get("lost_timeout", 8))),
        # 低置信 hold 放行上限（手背/握拳修复：无限 hold 会把骨架永久
        # 冻在旧姿势）。
        "hold_max": max(1, int(config.get("hold_max", 12))),
        # 新框入场确认帧数（拦单帧背景假框/闪框，低 conf 下必须）。
        "spawn_confirm": max(1, int(config.get("spawn_confirm", 2))),
        # 跨手碎片框拒收占比（0 关）：防"框飘到另一只手上"。
        "match_contain_thr": (
            None if match_contain_thr is None
            or float(match_contain_thr) <= 0
            else min(1.0, float(match_contain_thr))),
    }


def _make_detector(config: dict[str, Any]):
    cls = _load_glove_detector()
    cfg = _detector_config(config)
    return cls(**cfg)


def _transcode_to_h264(src: Path, dst: Path) -> bool:
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", str(dst)],
            check=True, capture_output=True, timeout=900,
        )
        return dst.exists() and dst.stat().st_size > 0
    except Exception as exc:
        print(f"[black_glove] H.264 transcode skipped: {exc}")
        return False


def _find_depth_sources(ctx: JobContext, video_refs: list,
                        config: dict[str, Any]):
    """Return one depth sampler source per RGB device.

    A multi-camera batch may contain different depth resolutions and different
    calibration files.  Never choose the first depth directory globally:
    pair each RGB ref through the same slot/token matcher used by Hand 3D.
    """
    import cv2

    try:
        from app.processing.modules.depth_hand_3d import (
            _find_device_pairs, _load_demo, _pair_calibration,
        )

        demo = _load_demo()
        if demo is None:
            return {}, ["black_glove: depth demo not found"]
        configured = str(config.get("depth_camera") or "").strip()
        requested_rgb = str(config.get("depth_source") or "").strip()
        sources: dict[str, tuple] = {}
        failures: list[str] = []
        pairs = _find_device_pairs(ctx, video_refs)
        for pair in pairs:
            source_key = str(pair.get("rgb_source") or "")
            depth_name = str(pair.get("depth_source") or "")
            if configured and depth_name != configured:
                continue
            if requested_rgb and source_key != requested_rgb:
                continue
            depth_dir = pair.get("depth_dir")
            depth_video = pair.get("depth_video")
            ref = pair.get("rgb_ref")
            video_path = ctx.resolve(ref)
            if (depth_dir is None and depth_video is None) or not depth_name:
                failures.append(f"{source_key}: depth pair missing")
                continue
            if video_path is None or not video_path.exists():
                failures.append(f"{source_key}: video artifact missing")
                continue
            depth_pngs = sorted(depth_dir.glob("*.png")) if depth_dir else []
            depth_reader = None
            if depth_video is not None:
                try:
                    depth_reader = DepthVideoReader(depth_video)
                    first = depth_reader.read()
                except (OSError, RuntimeError, ValueError) as exc:
                    failures.append(f"{source_key}: depth video unreadable ({exc})")
                    first = None
                finally:
                    if depth_reader is not None:
                        depth_reader.close()
            else:
                first = (cv2.imread(str(depth_pngs[0]), cv2.IMREAD_UNCHANGED)
                         if depth_pngs else None)
            if first is None:
                failures.append(f"{source_key}: depth frame unreadable")
                continue
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                failures.append(f"{source_key}: video cannot be opened")
                continue
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            cap.release()
            try:
                calib, calib_source = _pair_calibration(
                    pair, demo, (height, width), first.shape[:2])
                aligner = demo.LiveAligner(
                    calib["color_intrinsics"], calib["depth_to_color"],
                    calib["depth_intrinsics"],
                    fill_passes=max(1, min(3, int(config.get("depth_fill", 1)))),
                )
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                failures.append(f"{source_key}: calibration unavailable ({exc})")
                continue
            sources[source_key] = (depth_dir, depth_pngs, depth_video, aligner,
                                   depth_name, calib_source)
        return sources, failures
    except Exception as exc:
        print(f"[black_glove] depth path disabled: {exc}")
        return {}, [f"black_glove: depth path disabled ({exc})"]


class _DepthSampler:
    # Keep this in sync with the D435 Hand 3D lift path.  The aligned depth
    # image can contain background pixels at a glove landmark (especially on
    # the glove boundary), so a single unbounded sample can produce a valid
    # but completely wrong camera-space Z.
    _BAND_HALF_MM = BAND_HALF_M * 1000.0
    _BAND_MIN_VALID = BAND_MIN_VALID

    def __init__(self, source):
        if len(source) == 5:
            (self.depth_dir, self.depth_pngs, self.aligner,
             self.depth_name, self.calib_source) = source
            self.depth_video = None
        else:
            (self.depth_dir, self.depth_pngs, self.depth_video, self.aligner,
             self.depth_name, self.calib_source) = source
        self.depth_reader = (DepthVideoReader(self.depth_video)
                             if self.depth_video is not None else None)
        # Recordings in the current batches start at 000001, while some
        # exporters use 000000. Detect the convention once so frame 0 is not
        # silently paired with the wrong depth image.
        self.frame_offset = (
            0 if self.depth_reader is not None
            else 0 if (self.depth_dir / "000000.png").exists() else 1
        )

    def close(self) -> None:
        if self.depth_reader is not None:
            self.depth_reader.close()
            self.depth_reader = None

    def frame_xyz(self, frame_index: int, keypoints_px: list[list[float]],
                  video_w: int, video_h: int) -> np.ndarray:
        xyz, _measured, _zc = self.frame_xyz_with_meta(
            frame_index, keypoints_px, video_w, video_h)
        return xyz

    def frame_xyz_with_meta(self, frame_index: int,
                            keypoints_px: list[list[float]],
                            video_w: int, video_h: int):
        """Return xyz plus measured-point mask and raw hand-depth centre."""
        import cv2

        empty = np.full((len(keypoints_px), 3), np.nan, np.float32)
        empty_mask = np.zeros(len(keypoints_px), dtype=bool)

        if self.depth_reader is not None:
            depth = self.depth_reader.read()
        else:
            # Prefer the detected filename convention. Never blindly use a
            # positional fallback: one missing middle depth frame would shift
            # every later RGB/depth pair by one frame.
            expected = frame_index + self.frame_offset
            named = self.depth_dir / f"{expected:06d}.png"
            depth_path = named if named.exists() else None
            if depth_path is None and 0 <= frame_index < len(self.depth_pngs):
                candidate = self.depth_pngs[frame_index]
                if not candidate.stem.isdigit() or int(candidate.stem) == expected:
                    depth_path = candidate
            depth = (cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                     if depth_path is not None else None)
        if depth is None or depth.shape[:2] != (self.aligner.dh, self.aligner.dw):
            return empty, empty_mask, None
        aligned = self.aligner.align_depth_to_color(depth)
        scale_x = self.aligner.cw / max(1.0, float(video_w))
        scale_y = self.aligner.ch / max(1.0, float(video_h))
        uv = np.asarray(keypoints_px, np.float32).reshape(-1, 2)
        uv_aligned = uv * [scale_x, scale_y]
        # Match Hand 3D's two-pass lift: estimate the hand's depth centre
        # first, then resample only inside that depth band.  This removes a
        # background sample such as 0.8 m -> 2.7 m which otherwise becomes a
        # long 3D bone segment.
        z_mm = self.aligner.sample_points(aligned, uv_aligned)
        raw_valid = np.isfinite(z_mm) & (z_mm > 0)
        measured = raw_valid.copy()
        zc_mm = None
        if int(raw_valid.sum()) >= self._BAND_MIN_VALID:
            zc_mm = float(np.median(z_mm[raw_valid]))
            band = (zc_mm - self._BAND_HALF_MM,
                    zc_mm + self._BAND_HALF_MM)
            band_z = self.aligner.sample_points(
                aligned, uv_aligned, band=band)
            band_valid = np.isfinite(band_z) & (band_z > 0)
            measured = band_valid.copy()
            # Missing/out-of-band landmarks are completed at the measured
            # hand centre, as in Hand 3D's complete=True path.  X/Y still
            # come from the 2D landmark, so the projected skeleton remains
            # aligned with the RGB image.
            z_mm = np.where(band_valid, band_z, zc_mm).astype(np.float32)
        fx, fy, cx, cy = (self.aligner.fx_c, self.aligner.fy_c,
                           self.aligner.cx_c, self.aligner.cy_c)
        xyz = np.full((len(uv), 3), np.nan, np.float32)
        valid = np.isfinite(z_mm) & (z_mm > 0)
        z = z_mm[valid] / 1000.0
        u, v = uv_aligned[valid, 0], uv_aligned[valid, 1]
        xyz[valid, 0] = ((u - cx) / fx * z).astype(np.float32)
        xyz[valid, 1] = ((v - cy) / fy * z).astype(np.float32)
        xyz[valid, 2] = z.astype(np.float32)
        return xyz, measured & valid, (
            None if zc_mm is None else float(zc_mm) / 1000.0)


class _DepthTemporalSmoother:
    """Smooth lifted 3D points without inventing missing observations.

    The standalone D435 pipeline keeps a short-lived 3D slot state and
    reinitializes it after a long gap.  The workflow used to write every
    depth-lifted frame directly to parquet, so depth noise and a stale filter
    state could leak into the 3D preview.  This small, device-agnostic layer
    keeps the same semantics for any depth camera: valid observations are
    smoothed, missing frames stay missing, and a label/long-gap/large-jump
    transition starts a fresh filter.
    """

    def __init__(self, fps: float, config: dict[str, Any]):
        from app.processing.black_glove.filters import OneEuroFilter

        self._filter_type = OneEuroFilter
        self._fps = max(float(fps or 30.0), 1.0)
        self._freq_min = max(float(config.get("depth_freq_min", 3.0)), 0.01)
        self._beta = max(float(config.get("depth_beta", 0.3)), 0.0)
        self._dcutoff = max(float(config.get("depth_dcutoff", 1.0)), 0.01)
        self._max_lost = max(1, int(config.get("depth_max_lost", 15)))
        self._filters: list[list[list[OneEuroFilter]] | None] = [None, None]
        self._labels: list[str | None] = [None, None]
        self._last_frame: list[int | None] = [None, None]
        self._centres: list[np.ndarray | None] = [None, None]

    def _new_filters(self):
        return [
            [self._filter_type(self._freq_min, self._beta, self._dcutoff)
             for _ in range(3)]
            for _ in range(21)
        ]

    def _reset(self, slot: int, label: str, frame_index: int,
               centre: np.ndarray):
        self._filters[slot] = self._new_filters()
        self._labels[slot] = label or None
        self._last_frame[slot] = frame_index
        self._centres[slot] = centre.copy()

    def apply(self, rows: list[dict[str, Any]], fps: float | None = None):
        if fps:
            self._fps = max(float(fps), 1.0)
        for row in rows:
            frame_index = int(row.get("frame_index", 0))
            ts_ms = frame_index * 1000.0 / self._fps
            for slot in (0, 1):
                prefix = f"hand_{slot}"
                if not row.get(f"{prefix}_present"):
                    continue
                raw = row.get(f"{prefix}_landmarks_3d")
                if raw is None:
                    continue
                xyz = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
                if xyz.shape != (21, 3):
                    continue
                valid = np.isfinite(xyz).all(axis=1)
                if int(valid.sum()) < 4:
                    continue
                centre = np.nanmedian(np.where(valid[:, None], xyz, np.nan), axis=0)
                label = str(row.get(f"{prefix}_handedness") or "")
                last = self._last_frame[slot]
                jump = (self._centres[slot] is not None and
                        float(np.linalg.norm(centre - self._centres[slot])) > 0.35)
                must_reset = (
                    self._filters[slot] is None or
                    (self._labels[slot] not in (None, "", label)
                     and label) or
                    (last is not None and frame_index - last > self._max_lost) or
                    jump
                )
                if must_reset:
                    self._reset(slot, label, frame_index, centre)
                filters = self._filters[slot]
                if filters is None:
                    continue
                smoothed = xyz.copy()
                for point_index in np.flatnonzero(valid):
                    for axis in range(3):
                        smoothed[point_index, axis] = filters[point_index][axis](
                            float(xyz[point_index, axis]), ts_ms)
                row[f"{prefix}_landmarks_3d"] = smoothed.tolist()
                self._labels[slot] = label or self._labels[slot]
                self._last_frame[slot] = frame_index
                self._centres[slot] = np.nanmedian(
                    np.where(np.isfinite(smoothed), smoothed, np.nan), axis=0)


def _row_for_hands(frame_index: int, hands: list, width: int, height: int,
                   depth_sampler: _DepthSampler | None = None,
                   rgb_tracker=None, timestamp_s: float = 0.0) -> dict:
    row: dict[str, Any] = {"frame_index": int(frame_index)}
    for slot in range(2):
        hand = hands[slot] if slot < len(hands) else None
        if hand is None:
            row.update({
                f"hand_{slot}_present": False,
                f"hand_{slot}_2d_present": False,
                f"hand_{slot}_keypoints": None,
                f"hand_{slot}_handedness": None,
                f"hand_{slot}_confidence": None,
                f"hand_{slot}_depth_mm": None,
                f"hand_{slot}_landmarks_3d": None,
                f"hand_{slot}_landmarks_3d_local": None,
                f"hand_{slot}_world_position": None,
                f"hand_{slot}_coordinate_frame": None,
                f"hand_{slot}_tracking_source": None,
                f"hand_{slot}_depth_source": None,
                f"hand_{slot}_state": "absent",
                f"hand_{slot}_propagated": False,
                f"hand_{slot}_depth_measured": None,
                f"hand_{slot}_depth_center_m": None,
            })
            continue
        px = np.asarray(hand.landmarks, np.float32).reshape(21, 2)
        norm = np.column_stack((px[:, 0] / max(1, width),
                                px[:, 1] / max(1, height),
                                np.zeros(21, np.float32)))
        norm[:, :2] = np.clip(norm[:, :2], 0.0, 1.0)
        hand_label = str(getattr(hand, "label", "") or "")
        if rgb_tracker is not None:
            hand_label = rgb_tracker.stabilize_handedness(
                f"hand_{slot}", hand_label)
        rgb_estimate = None
        if depth_sampler:
            xyz, depth_measured, depth_center_m = \
                depth_sampler.frame_xyz_with_meta(
                    frame_index, px.tolist(), width, height)
        elif rgb_tracker is not None:
            local = rgb_tracker.local_from_image(norm)
            local = (rgb_tracker.orient_palm_facing(
                local, hand_label)
                     if local is not None else None)
            rgb_estimate = (rgb_tracker.update(
                f"hand_{slot}", norm, local, timestamp_s,
                preserve_model_geometry=True)
                if local is not None else None)
            xyz = (np.asarray(rgb_estimate["landmarks_3d"], np.float32)
                   if rgb_estimate is not None else
                   np.full((21, 3), np.nan, np.float32))
        else:
            xyz = np.full((21, 3), np.nan, np.float32)
        depth_mm = (xyz[:, 2] * 1000.0).tolist() if depth_sampler else None
        local_xyz = (np.asarray(
            rgb_estimate.get("landmarks_3d_local"), np.float32).tolist()
            if rgb_estimate is not None else None)
        world_position = (np.asarray(
            rgb_estimate.get("position"), np.float32).tolist()
            if rgb_estimate is not None else None)
        row.update({
            f"hand_{slot}_present": True,
            f"hand_{slot}_2d_present": True,
            f"hand_{slot}_keypoints": norm.tolist(),
            f"hand_{slot}_handedness": hand_label,
            f"hand_{slot}_confidence": float(getattr(hand, "score", 0.0)),
            f"hand_{slot}_depth_mm": depth_mm,
            f"hand_{slot}_landmarks_3d": xyz.tolist(),
            f"hand_{slot}_landmarks_3d_local": local_xyz,
            f"hand_{slot}_world_position": world_position,
            f"hand_{slot}_coordinate_frame": (
                str(rgb_estimate.get("coordinate_frame"))
                if rgb_estimate is not None else None),
            f"hand_{slot}_tracking_source": (
                str(rgb_estimate.get("tracking_source"))
                if rgb_estimate is not None else None),
            f"hand_{slot}_depth_source": "depth" if depth_sampler else None,
            f"hand_{slot}_state": "real",
            f"hand_{slot}_propagated": False,
            f"hand_{slot}_depth_measured": (
                depth_measured.tolist() if depth_sampler else None),
            f"hand_{slot}_depth_center_m": (
                depth_center_m if depth_sampler else None),
        })
    return row


def _same_wholesale_observation(previous, current, threshold: float = 0.08) -> bool:
    """Return whether two wholesale 3D observations are mutually consistent."""
    if previous is None:
        return False
    a = np.asarray(previous, np.float64).reshape(21, 3)
    b = np.asarray(current, np.float64).reshape(21, 3)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if int(ok.sum()) < 4:
        return False
    return float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1))) <= threshold


def _apply_d435_tracking(rows: list[dict], config: dict[str, Any]) -> dict[str, int]:
    """Add v1.0-style 3D gating, slot tracking and explicit frame states.

    The detector remains the existing YOLO/RTMPose detector.  This stage only
    operates on its 2D/depth output and never changes detector coordinates.
    ``real`` means the current frame supplied a usable observation;
    ``propagated`` means the 3D value is an alpha-beta prediction; ``absent``
    means there is no usable observation or bounded prediction.
    """
    tracker = AlphaBetaHandSlots(
        max_lost=max(1, int(config.get("propagate_max",
                                      config.get("depth_max_lost", 15)))),
        alpha=float(config.get("ab_alpha", 0.5)),
        beta=float(config.get("ab_beta", 0.1)),
    )
    wholesale_previous: list[np.ndarray | None] = [None, None]
    wholesale_streak = [0, 0]
    gate_streak = [np.zeros(21, dtype=np.int32),
                   np.zeros(21, dtype=np.int32)]
    gate_labels = ["", ""]
    slot_zc: list[float | None] = [None, None]
    slot_zc_labels = ["", ""]
    stats = {"real": 0, "propagated": 0, "absent": 0}
    # Keep immutable observations so a reverse pass can fill a leading gap
    # without treating a forward prediction as training data.
    observations = []
    for source in rows:
        frame_observations = []
        for slot in (0, 1):
            prefix = f"hand_{slot}"
            label = str(source.get(f"{prefix}_handedness") or "")
            detected = bool(source.get(f"{prefix}_2d_present"))
            points = None
            raw = source.get(f"{prefix}_landmarks_3d")
            if raw is not None:
                try:
                    candidate = np.asarray(raw, np.float64).reshape(21, 3)
                    if int(np.isfinite(candidate).all(axis=1).sum()) >= 4:
                        points = candidate.copy()
                except (TypeError, ValueError):
                    points = None
            # v1.0 M5: the per-frame depth median can jump when the set of
            # valid glove pixels changes. Stabilize only completed points at
            # a slot-level z centre; measured points remain untouched.
            measured = source.get(f"{prefix}_depth_measured")
            if points is not None and measured is not None:
                try:
                    measured_mask = np.asarray(measured, dtype=bool).reshape(21)
                except (TypeError, ValueError):
                    measured_mask = np.zeros(21, dtype=bool)
                measured_mask &= np.isfinite(points).all(axis=1)
                if int(measured_mask.sum()) >= BAND_MIN_VALID:
                    zf = source.get(f"{prefix}_depth_center_m")
                    if zf is None or not np.isfinite(float(zf)):
                        zf = float(np.median(points[measured_mask, 2]))
                    else:
                        zf = float(zf)
                    if slot_zc_labels[slot] != label or slot_zc[slot] is None:
                        slot_zc[slot] = zf
                    else:
                        slot_zc[slot] = 0.5 * slot_zc[slot] + 0.5 * zf
                    missing = np.isfinite(points).all(axis=1) & ~measured_mask
                    if missing.any() and slot_zc[slot] > 0:
                        old_z = points[missing, 2].copy()
                        scale = slot_zc[slot] / np.maximum(old_z, 1.0e-6)
                        points[missing, 0] *= scale
                        points[missing, 1] *= scale
                        points[missing, 2] = slot_zc[slot]
                    slot_zc_labels[slot] = label
            frame_observations.append((detected, points, label))
        observations.append(frame_observations)

    for row_index, row in enumerate(rows):
        frame_index = int(row.get("frame_index", 0))
        for slot in (0, 1):
            prefix = f"hand_{slot}"
            detected_2d, points, label = observations[row_index][slot]

            prediction = tracker.predict(slot, frame_index)
            state = "absent"
            output = None
            if detected_2d and points is not None:
                gated, wholesale = gate_observations(
                    points, prediction,
                    gate=float(config.get("depth_gate_m", 0.15)),
                )
                if wholesale:
                    if _same_wholesale_observation(
                            wholesale_previous[slot], gated) or wholesale_streak[slot] >= 2:
                        # The old slot state is stale. Re-seed only after a
                        # second agreeing frame so one bad depth frame cannot
                        # teleport the entire hand.
                        tracker.reset(slot, label)
                        tracker.observe(slot, label, points, frame_index)
                        wholesale_previous[slot] = None
                        wholesale_streak[slot] = 0
                        output = tracker.predict(slot, frame_index)
                        state = "real"
                    else:
                        wholesale_previous[slot] = gated
                        wholesale_streak[slot] += 1
                        output = prediction
                        state = "propagated" if output is not None else "absent"
                else:
                    if gate_labels[slot] != label:
                        gate_streak[slot][:] = 0
                        gate_labels[slot] = label
                    finite_points = np.isfinite(points).all(axis=1)
                    gated_finite = np.isfinite(gated).all(axis=1)
                    latched = finite_points & ~gated_finite
                    gate_streak[slot][latched] += 1
                    gate_streak[slot][~latched] = 0
                    forgive_after = max(1, int(config.get("gate_forgive", 5)))
                    forgive = (gate_streak[slot] >= forgive_after
                               ) & finite_points
                    observation_for_tracker = np.asarray(gated, np.float64).copy()
                    if forgive.any():
                        # Re-admit recovered joints instead of leaving a gate
                        # permanently locked.
                        observation_for_tracker[forgive] = points[forgive]
                        gate_streak[slot][forgive] = 0
                    tracker.observe(slot, label, observation_for_tracker,
                                    frame_index)
                    wholesale_previous[slot] = None
                    wholesale_streak[slot] = 0
                    output = tracker.predict(slot, frame_index)
                    state = "real"
            else:
                # No current usable observation: predict before incrementing
                # lost count, exactly as the v1.0 live path does.
                tracker.mark_lost(slot)
                output = prediction
                state = "propagated" if output is not None else "absent"
                wholesale_previous[slot] = None
                wholesale_streak[slot] = 0
                gate_streak[slot][:] = 0

            row[f"{prefix}_state"] = state
            row[f"{prefix}_propagated"] = state == "propagated"
            if state != "real":
                stable_label = tracker.slot_label(slot)
                if stable_label:
                    row[f"{prefix}_handedness"] = stable_label
            if state == "absent" or output is None:
                row[f"{prefix}_present"] = False
                row[f"{prefix}_landmarks_3d"] = None
                row[f"{prefix}_tracking_source"] = "none"
            else:
                row[f"{prefix}_present"] = True
                row[f"{prefix}_landmarks_3d"] = np.asarray(
                    output, np.float32).reshape(21, 3).tolist()
                row[f"{prefix}_tracking_source"] = (
                    "yolo_rtmpose_depth_real" if state == "real"
                    else "d435_alpha_beta_propagated")
    # Offline processing has the whole sequence available.  A reverse pass
    # fills a bounded leading portion of a gap that the forward-only pass
    # could not reach, while real observations always remain authoritative.
    reverse = AlphaBetaHandSlots(
        max_lost=tracker.max_lost, alpha=tracker.alpha, beta=tracker.beta)
    for row_index in range(len(rows) - 1, -1, -1):
        row = rows[row_index]
        frame_index = int(row.get("frame_index", row_index))
        for slot in (0, 1):
            prefix = f"hand_{slot}"
            detected, points, label = observations[row_index][slot]
            if detected and points is not None:
                gated, wholesale = gate_observations(
                    points, reverse.predict(slot, frame_index),
                    gate=float(config.get("depth_gate_m", 0.15)),
                )
                reverse.observe(slot, label, points if wholesale else gated,
                                frame_index)
                continue
            prediction = reverse.predict(slot, frame_index)
            reverse.mark_lost(slot)
            if (str(row.get(f"{prefix}_state") or "absent") == "absent"
                    and prediction is not None):
                row[f"{prefix}_present"] = True
                row[f"{prefix}_propagated"] = True
                row[f"{prefix}_state"] = "propagated"
                row[f"{prefix}_landmarks_3d"] = np.asarray(
                    prediction, np.float32).reshape(21, 3).tolist()
                stable_label = reverse.slot_label(slot)
                if stable_label:
                    row[f"{prefix}_handedness"] = stable_label
                row[f"{prefix}_tracking_source"] = "d435_alpha_beta_propagated"

    for row in rows:
        for slot in (0, 1):
            state = str(row.get(f"hand_{slot}_state") or "absent")
            stats[state] += 1
    return stats


def _flat63(value) -> list[float]:
    """Normalize one 21x3 value to the hand_3d parquet contract."""
    if value is None:
        return np.full(63, np.nan, np.float32).tolist()
    try:
        arr = np.asarray(value)
        if arr.dtype == object:
            arr = np.asarray([np.asarray(point, dtype=np.float32)
                              for point in value], dtype=np.float32)
        arr = arr.astype(np.float32).reshape(21, 3)
    except (TypeError, ValueError):
        return np.full(63, np.nan, np.float32).tolist()
    return arr.reshape(63).tolist()


def _hand_3d_rows(rows: list[dict], fps: float = 30.0,
                  config: dict[str, Any] | None = None) -> list[dict]:
    """Convert depth or RGB-estimated rows to the existing Hand 3D schema."""
    config = config or {}
    _apply_d435_tracking(rows, config)
    if bool(config.get("depth_smooth", True)):
        _DepthTemporalSmoother(fps, config).apply(rows, fps)
    output = []
    for source in rows:
        row = {"frame_index": int(source["frame_index"])}
        for slot in (0, 1):
            prefix = f"hand_{slot}"
            present = bool(source.get(f"{prefix}_present"))
            state = str(source.get(f"{prefix}_state") or
                        ("real" if present else "absent"))
            row.update({
                f"{prefix}_present": present,
                f"{prefix}_2d_present": bool(
                    source.get(f"{prefix}_2d_present", present)),
                f"{prefix}_landmarks_3d": _flat63(
                    source.get(f"{prefix}_landmarks_3d")),
                f"{prefix}_landmarks_3d_local": _flat63(
                    source.get(f"{prefix}_landmarks_3d_local")),
                f"{prefix}_world_position": (
                    list(source.get(f"{prefix}_world_position"))
                    if source.get(f"{prefix}_world_position") is not None
                    else None),
                f"{prefix}_coordinate_frame": source.get(
                    f"{prefix}_coordinate_frame"),
                f"{prefix}_tracking_source": source.get(
                    f"{prefix}_tracking_source"),
                f"{prefix}_depth_source": source.get(
                    f"{prefix}_depth_source"),
                f"{prefix}_keypoints": _flat63(
                    source.get(f"{prefix}_keypoints")),
                f"{prefix}_label": str(
                    source.get(f"{prefix}_handedness") or ""),
                f"{prefix}_confidence": float(
                    source.get(f"{prefix}_confidence") or 0.0),
                f"{prefix}_reprojection_error": float("nan"),
                f"{prefix}_gesture": "",
                f"{prefix}_fingers": -1,
                f"{prefix}_state": state,
                f"{prefix}_propagated": bool(
                    source.get(f"{prefix}_propagated", state == "propagated")),
            })
        output.append(row)
    return output


def _draw_hands(frame, hands) -> None:
    for hand in hands:
        points = np.asarray(hand.landmarks, np.float32).reshape(21, 2)
        draw_demo_style(frame, points.tolist())
        import cv2
        label = str(getattr(hand, "label", "") or "Hand")
        score = float(getattr(hand, "score", 0.0))
        x, y = int(points[:, 0].min()), int(points[:, 1].min())
        scale = hand_style_scale(points)
        cv2.putText(frame, f"glove {label} {score:.2f}",
                    (max(0, x), max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.28, 0.45 * scale),
                    (0, 220, 255), max(1, int(round(scale))),
                    cv2.LINE_AA)


def _stable_slot_hands(hands: list, slot_by_track: dict,
                       slot_last_seen: list[int], slot_last_centres: list,
                       frame_index: int,
                       slot_labels: list[str] | None = None,
                       slot_velocities: list | None = None) -> list:
    """Keep a disappearing hand in its output slot.

    ``GloveDetector`` supplies a useful tracker id in ``DetectedHand.index``,
    but that id can be recreated when a glove is briefly occluded.  Therefore
    it is only a soft hint.  Existing slots are assigned jointly by predicted
    image position first, then by track id and the latched handedness label.
    This is important when two hands cross: trusting the detector list or a
    newly-created id at that exact frame swaps the two output columns and
    consequently swaps the RGB 3D hands as well.
    """
    ordered = [None, None]
    if not hands:
        return ordered

    slot_labels = slot_labels if slot_labels is not None else ["", ""]
    if slot_velocities is None:
        slot_velocities = [None, None]

    from itertools import combinations, permutations

    def _centre(hand) -> np.ndarray:
        points = np.asarray(hand.landmarks, dtype=np.float32).reshape(21, 2)
        return np.nanmedian(points, axis=0).astype(np.float32)

    def _key(hand, fallback_index: int):
        raw_key = getattr(hand, "index", None)
        try:
            return int(raw_key)
        except (TypeError, ValueError):
            return ("position", fallback_index)

    candidates = []
    used_keys = set()
    for fallback_index, hand in enumerate(hands[:2]):
        key = _key(hand, fallback_index)
        if key in used_keys:
            key = ("position", fallback_index)
        used_keys.add(key)
        candidates.append((key, hand, _centre(hand),
                           str(getattr(hand, "label", "") or "")))

    active_slots = [s for s in (0, 1) if slot_last_centres[s] is not None]
    assignments: list[tuple[int, int]] = []  # (candidate index, slot)

    if active_slots:
        def _predicted(slot: int) -> np.ndarray:
            centre = np.asarray(slot_last_centres[slot], dtype=np.float32)
            velocity = slot_velocities[slot]
            if velocity is None:
                return centre
            gap = min(3, max(1, int(frame_index - slot_last_seen[slot])))
            return centre + np.asarray(velocity, dtype=np.float32) * gap

        def _cost(candidate_index: int, slot: int) -> float:
            key, _hand, centre, label = candidates[candidate_index]
            distance = float(np.linalg.norm(centre - _predicted(slot)))
            mapped = slot_by_track.get(key)
            # Track IDs and handedness are deliberately weak tie-breakers.
            # Position continuity must win when an ID is recreated or two
            # hands cross in the image.
            if mapped == slot:
                distance -= 30.0
            elif mapped in (0, 1):
                distance += 30.0
            stable = str(slot_labels[slot] or "")
            if label and stable and label != stable:
                distance += 20.0
            return distance

        count = min(len(candidates), len(active_slots))
        best = None
        for slot_subset in combinations(active_slots, count):
            for candidate_order in permutations(range(len(candidates)), count):
                total = sum(_cost(ci, slot)
                            for slot, ci in zip(slot_subset, candidate_order))
                # Prefer keeping more recently observed slots when costs tie.
                key = (total, tuple(-slot_last_seen[s] for s in slot_subset),
                       tuple(candidate_order))
                if best is None or key < best[0]:
                    best = (key, slot_subset, candidate_order)
        if best is not None:
            _, slots, candidate_order = best
            assignments.extend(zip(candidate_order, slots))

        used_candidates = {ci for ci, _slot in assignments}
        used_slots = {slot for _ci, slot in assignments}
        free_slots = [s for s in (0, 1) if s not in used_slots]
        for ci, (_key_value, _hand, _centre_value, _label) in enumerate(candidates):
            if ci in used_candidates or not free_slots:
                continue
            slot = free_slots.pop(0)
            assignments.append((ci, slot))
    else:
        # Initial order is kept for backward compatibility with existing
        # batches. Once initialized, the geometric assignment above takes
        # over and prevents later left/right swaps.
        assignments = [(ci, ci) for ci in range(min(2, len(candidates)))]

    for ci, slot in assignments:
        key, hand, centre, label = candidates[ci]
        old_centre = slot_last_centres[slot]
        old_seen = slot_last_seen[slot]
        if old_centre is not None and old_seen >= 0:
            gap = max(1, int(frame_index - old_seen))
            measured_velocity = (centre - old_centre) / float(gap)
            previous_velocity = slot_velocities[slot]
            slot_velocities[slot] = measured_velocity if previous_velocity is None else (
                0.5 * np.asarray(previous_velocity, dtype=np.float32)
                + 0.5 * measured_velocity)
        slot_last_seen[slot] = int(frame_index)
        slot_last_centres[slot] = centre
        # Latch handedness once per persistent output slot. The detector's
        # per-frame geometric label is allowed to flip during a palm/back
        # transition, but that must not rename a healthy 3D track.
        if not slot_labels[slot] and label:
            slot_labels[slot] = label
        stable = str(slot_labels[slot] or label or "")
        if stable:
            try:
                hand.label = stable
            except AttributeError:
                pass
        # Remove stale IDs that previously pointed at this slot. A recreated
        # detector ID is then associated with the slot without causing a
        # one-frame swap.
        for old_key, old_slot in list(slot_by_track.items()):
            if old_slot == slot and old_key != key:
                slot_by_track.pop(old_key, None)
        slot_by_track[key] = slot
        ordered[slot] = hand
    return ordered


def run_local(video_path: Path, output_dir: Path, skeleton_dir: Path,
              config: dict[str, Any], progress_callback: Callable[[float], None] | None,
              depth_sampler: _DepthSampler | None = None,
              keypoint_name: str = "hand_keypoints.parquet") -> dict[str, Any]:
    import cv2
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    detector = _make_detector(config)
    rgb_tracker = None
    if (depth_sampler is None
            and bool(config.get("preview_3d", True))):
        from app.processing.modules.rgb_world_tracking import RGBWorldTracker

        rgb_tracker = RGBWorldTracker(
            width, height,
            freq_min=float(config.get("freq_min", 5.0)),
            beta=float(config.get("beta", 0.05)),
        )
    raw_path = skeleton_dir / f"{Path(keypoint_name).stem}_skeleton.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        detector.close()
        cap.release()
        raise RuntimeError(f"Cannot create skeleton video: {raw_path}")
    rows: list[dict] = []
    slot_by_track: dict = {}
    slot_last_seen = [-1, -1]
    slot_last_centres = [None, None]
    slot_velocities = [None, None]
    slot_labels = ["", ""]
    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hands = detector.detect(frame)
            ordered_hands = _stable_slot_hands(
                hands, slot_by_track, slot_last_seen, slot_last_centres,
                frame_index, slot_labels, slot_velocities)
            rows.append(_row_for_hands(
                frame_index, ordered_hands, width, height, depth_sampler, rgb_tracker,
                frame_index / max(1.0, float(fps))))
            _draw_hands(frame, [hand for hand in ordered_hands if hand is not None])
            writer.write(frame)
            frame_index += 1
            if progress_callback and total:
                progress_callback(min(0.98, frame_index / total))
    finally:
        cap.release()
        writer.release()
        detector.close()
    if not rows:
        raise RuntimeError(f"No frames in video: {video_path}")
    parquet_path = output_dir / keypoint_name
    # Depth masks/centres are internal inputs to the 3D tracker, not part of
    # the public 2D keypoint artifact. Keep the 2D schema compact.
    public_rows = [
        {key: value for key, value in row.items()
         if not (key.endswith("_depth_measured")
                 or key.endswith("_depth_center_m"))}
        for row in rows
    ]
    pd.DataFrame(public_rows).to_parquet(parquet_path, index=False)
    h264_path = skeleton_dir / f"{Path(keypoint_name).stem}_skeleton_h264.mp4"
    video_path_out = h264_path if _transcode_to_h264(raw_path, h264_path) else raw_path
    if video_path_out != raw_path:
        raw_path.unlink(missing_ok=True)
    frames_with_hands = sum(bool(r["hand_0_present"] or r["hand_1_present"]) for r in rows)
    manifest = {
        "frames": len(rows),
        "frames_with_hands": frames_with_hands,
        "fps": fps,
        "width": width,
        "height": height,
        "detector": ("YOLO-World + "
                     + str(config.get("pose_backend", "rtmpose"))),
        "weights": str(_detector_config(config)["weights"]),
        "mode": "independent_2d_per_view",
        "triangulation": False,
        "depth_lift": bool(depth_sampler),
        "rgb_estimated_3d": (
            depth_sampler is None
            and bool(config.get("preview_3d", True))),
        "preview_3d": bool(config.get("preview_3d", True)),
        "depth_camera": depth_sampler.depth_name if depth_sampler else None,
        "skeleton_video": str(video_path_out.name),
    }
    if depth_sampler:
        manifest.update({
            "unit": "camera_meters",
            "coordinate_frame": "aligned_color_camera",
            "world_coordinates": True,
        })
    (output_dir / f"{Path(keypoint_name).stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(1.0)
    return {"parquet": parquet_path, "video": video_path_out, "manifest": manifest,
            "rows": rows}


@register
class BlackGloveHandModule(ProcessingModule):
    slug = "rgbd_to_3d_black_glove"
    version = "1.7"
    category = "process"
    label = "RGB-D_3D_BlackGlove"
    icon = "ant-design:aim-outlined"
    color = HAND3D_COLOR
    inputs = ({"key": "video", "label": "RGB Video"},
              {"key": "depth", "label": "Depth"})
    # 2D keypoints remain an internal artifact for rendering/tracking.  The
    # workflow graph exposes the single downstream contract: Hand 3D.
    outputs = ({"key": "hand_3d", "label": "Hand 3D"},)
    default_config = {
        "mode": "auto", "max_hands": 2, "det_conf": 0.05,
        "device": "auto", "pose_device": "auto", "imgsz": 640,
        "pose_backend": "rtmpose", "pose_model": "",
        "smooth": True, "freq_min": 5.0, "beta": 0.05,
        "use_tracker": True, "movement_thresh": 1.5, "skip_timeout": 3,
        "box_alpha": 0.7, "freeze_max": 15, "pose_conf_thr": 0.15,
        "pose_box_raw": False, "hold_translate": True, "new_track_conf": 0.1,
        "lost_timeout": 8, "hold_max": 12, "spawn_confirm": 2,
        "match_contain_thr": 0.7,
        "depth_camera": "",
        "depth_fill": 1, "depth_smooth": True, "depth_freq_min": 3.0,
        "depth_beta": 0.3, "depth_max_lost": 15, "propagate_max": 15,
        "depth_gate_m": 0.15, "gate_forgive": 5,
        "ab_alpha": 0.5, "ab_beta": 0.1,
    }
    config_schema = (
        field("mode", "select", "Input mode", "auto", options=["auto", "mono", "stereo"]),
        field("max_hands", "number", "Max hands", 2, min=1, max=2),
        field("det_conf", "number", "Glove detection confidence", 0.05, min=0.01, max=1, step=0.01),
        field("device", "select", "Detector device", "auto", options=["auto", "cpu", "cuda"]),
        field("pose_device", "select", "Pose device", "auto", options=["auto", "cpu", "cuda"]),
        field("pose_backend", "select", "Pose backend", "rtmpose",
              options=["rtmpose", "mediapipe"]),
        field("pose_model", "string", "MediaPipe task path (optional)", ""),
        field("imgsz", "number", "Detector image size", 640, min=160, max=1280, step=32),
        field("smooth", "boolean", "One-Euro smoothing", True),
        field("freq_min", "number", "Smoothing cutoff Hz", 5.0, min=1, max=60, step=1),
        field("beta", "number", "Smoothing speed coefficient", 0.05, min=0, max=2, step=0.05),
        field("use_tracker", "boolean", "Track hands between frames", True),
        field("movement_thresh", "number", "Pose refresh movement threshold (px)", 1.5,
              min=0, max=20, step=0.5),
        field("skip_timeout", "number", "Maximum skipped pose frames", 3,
              min=1, max=30, step=1),
        field("box_alpha", "number", "Tracking box smoothing", 0.7,
              min=0, max=1, step=0.05),
        field("pose_conf_thr", "number", "Pose confidence threshold", 0.15,
              min=0, max=1, step=0.05),
        field("pose_box_raw", "boolean", "Use raw box for pose crop", False),
        field("hold_translate", "boolean", "Compensate held pose motion", True),
        field("new_track_conf", "number", "New track confidence gate", 0.1,
              min=0, max=1, step=0.01),
        field("lost_timeout", "number", "Lost track timeout (frames)", 8,
              min=1, max=60, step=1),
        field("hold_max", "number", "Low-confidence hold limit (frames)", 12,
              min=1, max=120, step=1),
        field("spawn_confirm", "number", "New box confirm frames", 2,
              min=1, max=5, step=1),
        field("match_contain_thr", "number", "Cross-hand box reject ratio (0=off)", 0.7,
              min=0, max=1, step=0.05),
        field("depth_camera", "string", "Depth source override (optional)", ""),
        field("depth_source", "string", "RGB source to receive depth (optional)", ""),
        field("depth_fill", "number", "Depth hole-fill passes", 1, min=1, max=3),
        field("depth_smooth", "boolean", "Smooth lifted 3D points", True),
        field("depth_freq_min", "number", "3D smoothing cutoff Hz", 3.0,
              min=0.1, max=30, step=0.5),
        field("depth_beta", "number", "3D smoothing speed coefficient", 0.3,
              min=0, max=2, step=0.05),
        field("depth_max_lost", "number", "3D reset gap (frames)", 15,
              min=1, max=120, step=1),
        field("propagate_max", "number", "3D prediction limit (frames)", 15,
              min=1, max=60, step=1),
        field("depth_gate_m", "number", "3D jump gate (meters)", 0.15,
              min=0.03, max=1, step=0.01),
        field("gate_forgive", "number", "Gate recovery frames", 5,
              min=1, max=30, step=1),
        field("ab_alpha", "number", "3D alpha correction", 0.5,
              min=0.05, max=1, step=0.05),
        field("ab_beta", "number", "3D beta velocity correction", 0.1,
              min=0.01, max=1, step=0.01),
    )
    execution_target = "worker"
    capabilities = ("black_glove", "hand_keypoints", "hand_3d", "video_overlay",
                    "mono", "stereo", "depth_camera_meters",
                    "no_triangulation")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        video_refs = [r for key, r in ctx.incoming.items()
                      if r.kind == "video"
                      and (key == "video" or key.startswith("video#"))]
        if not video_refs:
            ctx.skip("RGB Video is not connected — skipped")
        depth_refs = [r for key, r in ctx.incoming.items()
                      if r.kind == "depth"
                      and (key == "depth" or key.startswith("depth#"))]
        if not depth_refs:
            ctx.skip("Depth is not connected — skipped")
        depth_ref = depth_refs[0]
        depth_path = ctx.resolve(depth_ref)
        if not depth_path or not depth_path.exists():
            ctx.skip("Connected Depth is unavailable — skipped")

        # Only the depth artifact connected to this card may select the depth
        # sidecar. This keeps a multi-camera batch from borrowing another
        # camera's depth directory during the legacy pairing scan.
        depth_metadata = getattr(depth_ref, "metadata", None) or {}
        connected_depth_key = str(
            depth_ref.source_key or depth_metadata.get("depth_source") or ""
        ).strip()
        depth_config = dict(ctx.config)
        # A stale manually entered override must not win over the port that is
        # actually connected in this run. An empty connected key deliberately
        # leaves pairing to the single-source legacy matcher.
        depth_config["depth_camera"] = connected_depth_key
        depth_config["depth_source"] = ""
        video_refs.sort(key=lambda r: (r.source_key or "", r.path or ""))
        mode = str(ctx.config.get("mode") or "auto")
        if mode == "mono":
            video_refs = video_refs[:1]
        elif mode == "stereo":
            video_refs = video_refs[:2]

        depth_sources, depth_failures = _find_depth_sources(
            ctx, video_refs, depth_config)
        if not depth_sources:
            ctx.skip("Black Glove Hand Skeleton requires a matching depth stream")
        # Keep only RGB views that have their own depth pair. Never fall back
        # to RGB-estimated 3D in this module; that is the separate
        # Black Hand RGB3D workflow.
        video_refs = [ref for ref in video_refs
                      if str(ref.source_key or "") in depth_sources]
        if not video_refs:
            ctx.skip("No RGB/depth pair available for Black Glove Hand Skeleton")
        output_kp = ctx.output_root / "hand_keypoints"
        output_video = ctx.output_root / "skeleton"
        out: dict[str, ArtifactRef] = {}
        results = []
        world_count = 0
        for index, ref in enumerate(video_refs):
            path = ctx.resolve(ref)
            if not path or not path.exists():
                ctx.skip(f"Video artifact missing — skipped: {ref.path}")
            key = _safe_key(ref.source_key, path.stem)
            source_key = str(ref.source_key or path.stem)
            depth_source = depth_sources.get(source_key)
            sampler = _DepthSampler(depth_source)
            try:
                result = run_local(path, output_kp, output_video, ctx.config,
                                   lambda p, i=index, n=len(video_refs):
                                   ctx.progress((i + p) / n),
                                   depth_sampler=sampler,
                                   keypoint_name=f"{key}.parquet")
            finally:
                sampler.close()
            # Keep the original artifact source in the manifest as well as in
            # ArtifactRef. The review resolver uses this when a source name
            # contains characters that were sanitized for the filename.
            result["manifest"]["source_key"] = source_key
            if depth_failures:
                result["manifest"]["processing_warnings"] = list(depth_failures)
            (result["parquet"].parent / f"{key}.manifest.json").write_text(
                json.dumps(result["manifest"], indent=2), encoding="utf-8")
            results.append((ref, key, result))
            keypoint_handle = "hand_keypoints" if index == 0 else f"hand_keypoints#{index + 1}"
            out[keypoint_handle] = ctx.ref(
                "hand_keypoints", result["parquet"], source_key=source_key,
                metadata=result["manifest"])
            skeleton_handle = "skeleton_video" if index == 0 else f"skeleton_video#{index + 1}"
            out[skeleton_handle] = ctx.ref(
                "video", result["video"], source_key=source_key,
                metadata={**result["manifest"], "skeleton": True})
            import pandas as pd

            world_dir = ctx.output_root / "hand_3d"
            world_dir.mkdir(parents=True, exist_ok=True)
            world_path = world_dir / f"{key}.parquet"
            world_rows = _hand_3d_rows(result["rows"], result["manifest"]["fps"],
                                        ctx.config)
            pd.DataFrame(world_rows).to_parquet(world_path, index=False)
            state_counts = {"real": 0, "propagated": 0, "absent": 0}
            for world_row in world_rows:
                for slot in (0, 1):
                    state = str(world_row.get(f"hand_{slot}_state") or "absent")
                    if state in state_counts:
                        state_counts[state] += 1
            world_manifest = {
                **result["manifest"],
                "mode": "black_glove_depth_lift",
                "unit": "camera_meters",
                "coordinate_frame": "aligned_color_camera",
                "world_coordinates": True,
                "metric_3d_available": True,
                "box_overlay": False,
                "prediction_for_preview_only": True,
                "real_only_export_default": True,
                "state_counts": state_counts,
                "calib_source": depth_source[5],
                "method": ("YOLO-World + "
                           + str(ctx.config.get("pose_backend", "rtmpose"))
                           + " 2D keypoints + "
                           "per-device depth alignment, 3x3 median depth-band lift, "
                           "hand-centre completion, depth jump gating, alpha-beta "
                           "slot prediction, and reset-aware temporal smoothing"),
            }
            (world_dir / f"{key}.manifest.json").write_text(
                json.dumps(world_manifest, indent=2), encoding="utf-8")
            hand_handle = "hand_3d" if world_count == 0 else "hand_3d#right"
            out[hand_handle] = ctx.ref(
                "hand_3d", world_path, source_key=source_key,
                metadata=world_manifest)
            world_count += 1

        if depth_failures:
            print("[black_glove] processing warnings: "
                  + "; ".join(depth_failures))

        # A stereo preview is only a display artifact. It does not combine
        # detections or create any cross-camera coordinates.
        if len(results) == 2:
            try:
                import cv2
                left, right = results[0][2]["video"], results[1][2]["video"]
                cap_l, cap_r = cv2.VideoCapture(str(left)), cv2.VideoCapture(str(right))
                w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                fps = cap_l.get(cv2.CAP_PROP_FPS) or 25.0
                raw = output_video / "black_glove_stereo_side_by_side.mp4"
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w * 2, h))
                while True:
                    ok_l, fl = cap_l.read(); ok_r, fr = cap_r.read()
                    if not ok_l or not ok_r:
                        break
                    writer.write(cv2.hconcat([fl, fr]))
                writer.release(); cap_l.release(); cap_r.release()
                h264 = output_video / "black_glove_stereo_side_by_side_h264.mp4"
                combined = h264 if _transcode_to_h264(raw, h264) else raw
                if combined != raw:
                    raw.unlink(missing_ok=True)
                out["stereo_preview"] = ctx.ref(
                    "video", combined, source_key="stereo",
                    metadata={"skeleton": True, "triangulation": False,
                              "independent_views": True})
            except Exception as exc:
                print(f"[black_glove] stereo preview skipped: {exc}")
        return out
