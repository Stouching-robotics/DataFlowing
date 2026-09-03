"""RGB-D Camera — generic input for a color stream with depth metadata.

The physical device category is intentionally separate from the uploaded
stream/source key.  A D435 or a future RGB-D camera can therefore use the
same workflow card while the selected source key continues to identify the
real files on disk.
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import INPUT_COLOR


@register
class RGBDCameraModule(ProcessingModule):
    slug = "rgbd_camera"
    version = "1.0"
    category = "input"
    label = "RGB-D Camera"
    icon = "ant-design:video-camera-outlined"
    color = INPUT_COLOR
    outputs = ({"key": "video", "label": "RGB Video"},
               {"key": "depth", "label": "Depth"})
    default_config = {"source_key": "", "fps": 30}
    config_schema = (
        field("source_key", "string",
              "Device name / source key (blank = auto-detect)", ""),
    )
    execution_target = "server"
    capabilities = ("video_input", "depth", "rgbd")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        videos = ctx.find_videos()
        if not videos:
            ctx.skip("Configured RGB-D video source is missing — skipped")
        from app.lerobot_v21 import canonical_source_key, source_key_from_video
        source_key = ctx.config.get("source_key") or None
        source_key = source_key or source_key_from_video(
            videos[0], ctx.input_root / "videos")
        source_key = canonical_source_key(source_key)
        video_ref = ctx.ref("video", videos[0], source_key=source_key or None)
        # Publish the actual depth stream as a typed artifact.  LeRobot v2.1
        # stores metric depth as a video; the legacy PNG directory remains a
        # read-only compatibility path for old batches.
        depth_ref = None
        try:
            from app.processing.modules.depth_hand_3d import _find_device_pairs
            pair = next(iter(_find_device_pairs(ctx, [video_ref])), None)
            depth_path = None
            if pair:
                depth_path = pair.get("depth_video") or pair.get("depth_dir")
            if depth_path is not None and depth_path.exists():
                depth_ref = ctx.ref(
                    "depth", depth_path, source_key=pair.get("depth_source"),
                    metadata={"rgb_source": source_key,
                              "depth_source": pair.get("depth_source")})
        except Exception as exc:
            # Pairing is validated again by the RGB-D processing module; an
            # unavailable sidecar must not make the input card crash.
            print(f"[rgbd_camera] depth artifact discovery skipped: {exc}")
        outputs = {"video": video_ref}
        if depth_ref is not None:
            outputs["depth"] = depth_ref
        return outputs
