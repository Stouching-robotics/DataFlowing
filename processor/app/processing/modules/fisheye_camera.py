"""Legacy fisheye camera input alias.

Fisheye is a lens attribute of the fixed ``RGB Camera`` category. Keep this
alias executable for historical graphs, without a machine-specific default.
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import INPUT_COLOR


@register
class FisheyeCameraModule(ProcessingModule):
    slug = "fisheye_camera"
    version = "1.0"
    category = "input"
    # Historical slug kept for saved workflows; fisheye is a lens attribute
    # of the generic RGB Camera input, not a separate device category.
    label = "RGB Camera"
    icon = "ant-design:video-camera-outlined"
    color = INPUT_COLOR
    outputs = ({"key": "video", "label": "RGB Video"},)
    default_config = {"source_key": "", "position": "", "fps": 30}
    config_schema = (field("source_key", "string", "Source key", ""),)
    execution_target = "server"
    capabilities = ("video_input", "calibration")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        videos = ctx.find_videos()
        if not videos:
            ctx.skip("Configured video source is missing — skipped")
        return {"video": ctx.ref("video", videos[0],
                                 source_key=ctx.config.get("source_key"))}
