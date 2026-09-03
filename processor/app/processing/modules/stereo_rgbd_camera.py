"""Stereo RGB-D Camera — two RGB views plus one paired depth stream.

This is a distinct input contract from ``stereo_camera``.  The latter only
publishes left/right RGB video, while this module also publishes a typed
``Depth`` output that can be connected to the RGB-D 3D hand modules.
The concrete camera name remains a source key in the run snapshot; the card
label is the stable device category used by the workflow editor.
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import INPUT_COLOR


@register
class StereoRGBDCameraModule(ProcessingModule):
    slug = "stereo_rgbd_camera"
    version = "1.0"
    category = "input"
    label = "Stereo RGB-D Camera"
    icon = "ant-design:video-camera-outlined"
    color = INPUT_COLOR
    outputs = (
        {"key": "video_left", "label": "Left RGB Video"},
        {"key": "video_right", "label": "Right RGB Video"},
        {"key": "depth", "label": "Depth"},
    )
    default_config = {
        "source_keys": "", "source_key": "", "position": "", "fps": 30,
    }
    config_schema = (
        field("source_keys", "string",
              "Device names (comma; blank = auto-detect)", ""),
        field("source_key", "string", "Source key", ""),
    )
    execution_target = "server"
    capabilities = ("video_input", "depth", "rgbd", "stereo")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        videos = ctx.find_videos()
        if len(videos) < 2:
            ctx.skip("Stereo RGB-D Camera requires left and right RGB video — skipped")
        from app.lerobot_v21 import canonical_source_key, source_key_from_video
        actual_keys = [source_key_from_video(path, ctx.input_root / "videos")
                       for path in videos[:2]]

        raw_keys = ctx.config.get("source_keys")
        if isinstance(raw_keys, str):
            keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        elif isinstance(raw_keys, (list, tuple)):
            keys = [str(key).strip() for key in raw_keys if str(key).strip()]
        else:
            keys = []
        keys = [canonical_source_key(key) for key in keys]

        refs = [
            ctx.ref("video", path,
                    source_key=(keys[index] if index < len(keys)
                                else actual_keys[index]))
            for index, path in enumerate(videos[:2])
        ]

        # Use the same physical-device pairing logic as RGB-D 3D processing.
        # A real v2.1 depth video (or a legacy PNG directory) can produce the
        # typed Depth output; this card never fabricates depth from RGB.
        from app.processing.modules.depth_hand_3d import _find_device_pairs

        try:
            pairs = _find_device_pairs(ctx, refs)
        except (OSError, RuntimeError, ValueError) as exc:
            ctx.skip(f"Stereo RGB-D Camera depth stream is unavailable ({exc}) — skipped")
        pair = next((item for item in pairs
                     if (item.get("depth_video") is not None
                         or item.get("depth_dir") is not None)
                     and item.get("depth_source")), None)
        if pair is None:
            return {"video_left": refs[0], "video_right": refs[1]}

        depth_path = pair.get("depth_video") or pair.get("depth_dir")
        depth_ref = ctx.ref(
            "depth", depth_path, source_key=pair.get("depth_source"),
            metadata={
                "rgb_source": pair.get("rgb_source"),
                "depth_source": pair.get("depth_source"),
                "device_key": (pair.get("device") or {}).get("key"),
                "device_name": (pair.get("device") or {}).get("name"),
            },
        )
        return {
            "video_left": refs[0],
            "video_right": refs[1],
            "depth": depth_ref,
        }
