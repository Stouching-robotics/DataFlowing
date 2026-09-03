"""RGB_TO_2D_BlackGlove — black-glove RGB-only keypoint workflow.

This module deliberately never searches for or reads a depth stream.  It uses
the black-glove detector for 2D landmarks and the shared RGBWorldTracker to
place those landmarks in a camera-relative estimated 3D coordinate frame.
The depth-based counterpart is ``RGB-D_3D_BlackGlove``.
"""

from __future__ import annotations

import json
from typing import Any

from app.processing import ArtifactRef, JobContext, ProcessingModule, field
from app.processing.registry import register
from app.processing.theme import HAND3D_COLOR
from app.processing.modules.black_glove_hand import (
    _hand_3d_rows,
    _safe_key,
    run_local,
)


def _config_schema() -> tuple[dict[str, Any], ...]:
    return (
        field("mode", "select", "Input mode", "auto",
              options=["auto", "mono", "stereo"]),
        field("max_hands", "number", "Max hands", 2, min=1, max=2),
        field("det_conf", "number", "Glove detection confidence", 0.05,
              min=0.01, max=1, step=0.01),
        field("device", "select", "Detector device", "auto",
              options=["auto", "cpu", "cuda"]),
        field("pose_device", "select", "Pose device", "auto",
              options=["auto", "cpu", "cuda"]),
        field("pose_backend", "select", "Pose backend", "rtmpose",
              options=["rtmpose", "mediapipe"]),
        field("pose_model", "string", "MediaPipe task path (optional)", ""),
        field("imgsz", "number", "Detector image size", 640,
              min=160, max=1280, step=32),
        field("smooth", "boolean", "One-Euro smoothing", True),
        field("preview_3d", "boolean", "Display-only 3D preview", True),
        field("freq_min", "number", "Smoothing cutoff Hz", 5.0,
              min=1, max=60, step=1),
        field("beta", "number", "Smoothing speed coefficient", 0.05,
              min=0, max=2, step=0.05),
        field("use_tracker", "boolean", "Track hands between frames", True),
        field("movement_thresh", "number", "Pose refresh movement threshold (px)",
              1.5, min=0, max=20, step=0.5),
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
    )


@register
class BlackHandRGB3DModule(ProcessingModule):
    slug = "rgb_to_2d_black_glove"
    version = "1.3"
    category = "process"
    label = "RGB_TO_2D_BlackGlove"
    icon = "ant-design:deployment-unit-outlined"
    color = HAND3D_COLOR
    inputs = ({"key": "video", "label": "RGB Video"},)
    # The RGB-derived hand_3d parquet is retained only for the 3D preview.
    # Downstream workflow nodes receive the public 2D contract.
    outputs = ({"key": "hand_keypoints", "label": "Hand 2D"},)
    default_config = {
        "mode": "auto", "max_hands": 2, "det_conf": 0.05,
        "device": "auto", "pose_device": "auto", "imgsz": 640,
        "pose_backend": "rtmpose", "pose_model": "",
        "smooth": True, "freq_min": 5.0, "beta": 0.05,
        "preview_3d": True,
        "use_tracker": True, "movement_thresh": 1.5,
        "skip_timeout": 3, "box_alpha": 0.7, "pose_conf_thr": 0.15,
        "pose_box_raw": False, "hold_translate": True,
        "new_track_conf": 0.1,
        "lost_timeout": 8, "hold_max": 12, "spawn_confirm": 2,
        "match_contain_thr": 0.7,
    }
    config_schema = _config_schema()
    execution_target = "worker"
    capabilities = ("black_glove", "hand_keypoints", "rgb_preview_3d",
                    "camera_relative", "video_overlay",
                    "mono", "stereo", "no_depth")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        import pandas as pd

        video_refs = [ref for ref in ctx.incoming.values() if ref.kind == "video"]
        if not video_refs:
            ctx.skip("No RGB video input — skipped")
        video_refs.sort(key=lambda ref: (ref.source_key or "", ref.path or ""))
        mode = str(ctx.config.get("mode") or "auto")
        if mode == "mono":
            video_refs = video_refs[:1]
        elif mode == "stereo":
            video_refs = video_refs[:2]
        else:
            video_refs = video_refs[:2]

        output_kp = ctx.output_root / "hand_keypoints"
        output_video = ctx.output_root / "skeleton"
        output_3d = ctx.output_root / "hand_3d"
        preview_3d = bool(ctx.config.get("preview_3d", True))
        if preview_3d:
            output_3d.mkdir(parents=True, exist_ok=True)
        out: dict[str, ArtifactRef] = {}

        for index, ref in enumerate(video_refs):
            path = ctx.resolve(ref)
            if not path or not path.exists():
                ctx.skip(f"RGB video artifact missing: {ref.path}")
            source_key = str(ref.source_key or path.stem)
            key = _safe_key(ref.source_key, path.stem)
            # depth_sampler is intentionally omitted. run_local creates the
            # RGBWorldTracker path when this argument is None.
            result = run_local(
                path, output_kp, output_video, ctx.config,
                lambda value, i=index, n=len(video_refs):
                ctx.progress((i + value) / n),
                depth_sampler=None,
                keypoint_name=f"{key}.parquet",
            )
            preview_3d = bool(ctx.config.get("preview_3d", True))
            result["manifest"].update({
                "source_key": source_key,
                "mode": "black_glove_rgb_estimated_3d" if preview_3d else "black_glove_rgb_2d",
                "unit": "rgb_estimated_meters" if preview_3d else "image_normalized",
                "coordinate_frame": "camera_relative" if preview_3d else "image_normalized",
                "world_coordinates": False,
                "metric_3d_available": False,
                "preview_3d": preview_3d,
                "method": ("YOLO-World + "
                           + str(ctx.config.get("pose_backend", "rtmpose"))
                           + " 2D keypoints + "
                           "RGB hand-model PnP and image-scale fallback with "
                           "stable local hand geometry and One-Euro smoothing"),
                "orientation_mode": "stable_local_model_wrist_pnp_only",
            })
            (result["parquet"].parent / f"{key}.manifest.json").write_text(
                json.dumps(result["manifest"], indent=2), encoding="utf-8")

            keypoint_handle = "hand_keypoints" if index == 0 else f"hand_keypoints#{index + 1}"
            out[keypoint_handle] = ctx.ref(
                "hand_keypoints", result["parquet"], source_key=source_key,
                metadata=result["manifest"])
            skeleton_handle = "skeleton_video" if index == 0 else f"skeleton_video#{index + 1}"
            out[skeleton_handle] = ctx.ref(
                "video", result["video"], source_key=source_key,
                metadata={**result["manifest"], "skeleton": True})

            if preview_3d:
                world_path = output_3d / f"{key}.parquet"
                # run_local already applies the RGB One-Euro filter.  The
                # helper also supports depth lifting and would otherwise
                # apply the depth-only temporal smoother a second time to
                # RGB estimates.
                rgb_3d_config = {**ctx.config, "depth_smooth": False}
                world_rows = _hand_3d_rows(
                    result["rows"], result["manifest"]["fps"], rgb_3d_config)
                pd.DataFrame(world_rows).to_parquet(world_path, index=False)
                world_manifest = {
                    **result["manifest"],
                    "artifact": "black_hand_rgb_3d",
                    "views": len(video_refs),
                    "view": "left" if index == 0 else "right",
                    "preview_3d": True,
                }
                (output_3d / f"{key}.manifest.json").write_text(
                    json.dumps(world_manifest, indent=2), encoding="utf-8")
        if ctx.progress:
            ctx.progress(1.0)
        return out
