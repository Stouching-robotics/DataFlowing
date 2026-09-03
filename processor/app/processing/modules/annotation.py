"""Annotation — 透传模块:保留上游 artifact,标记标注需求。"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import ANNOTATION_COLOR


@register
class AnnotationModule(ProcessingModule):
    slug = "annotation"
    version = "1.0"
    category = "process"
    label = "Human Annotation"
    icon = "ant-design:field-time-outlined"
    color = ANNOTATION_COLOR
    # Keep the persisted ``data`` handle for old workflows, but make the
    # semantic contract explicit: human annotation operates on RGB video.
    inputs = ({"key": "data", "label": "RGB Video"},)
    outputs = ({"key": "annotation", "label": "Annotation"},)
    default_config = {"type": "frame_level", "auto_label": True}
    execution_target = "server"
    capabilities = ("annotation",)

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        # The handle remains named ``data`` for backward compatibility. Do
        # not treat sensor/keypoint artifacts as an annotation video input.
        if not any(ref.kind == "video" for ref in ctx.incoming.values()):
            ctx.skip("No RGB video input — skipped")
        return dict(ctx.incoming)
