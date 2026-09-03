"""Legacy RGB camera input alias.

New workflows use ``mono_camera`` (displayed as ``RGB Camera``). Keep this alias executable for historical
graphs, but leave its source empty so a newly created graph cannot inherit a
machine-specific stream name.
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import INPUT_COLOR


@register
class RGBCameraModule(ProcessingModule):
    slug = "rgb_camera"
    version = "1.0"
    category = "input"
    # Historical slug kept for saved workflows; new workflows use the
    # Historical slug kept for saved workflows; new UI name is standardized.
    label = "RGB Camera"
    icon = "ant-design:video-camera-outlined"
    color = INPUT_COLOR
    outputs = ({"key": "video", "label": "RGB Video"},)
    default_config = {"source_key": "", "position": "", "fps": 30}
    config_schema = (field("source_key", "string", "Source key", ""),)
    execution_target = "server"
    capabilities = ("video_input", "ego_camera")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        videos = ctx.find_videos()
        if not videos:
            ctx.skip("Configured video source is missing — skipped")
        from app.lerobot_v21 import canonical_source_key, source_key_from_video
        source_key = (ctx.config.get("source_key")
                      or source_key_from_video(videos[0], ctx.input_root / "videos"))
        source_key = canonical_source_key(source_key)
        return {"video": ctx.ref("video", videos[0],
                                 source_key=source_key or None)}
