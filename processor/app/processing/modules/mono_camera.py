"""Mono Video — 输入源模块:按设备名称输出批次内单目主视频。

设备名称(source_key)留空 = 自动识别:find_videos 回退取批次全部主
RGB 视频(排除 depth/ 深度预览、_aux 辅助流、skeleton 骨骼视频)的第
一路。鱼眼(head_left_rgb)、D435 RGB(d435_rgb)等单目命名统一走
本卡片,无需按相机形态细分卡片。
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import INPUT_COLOR


@register
class MonoCameraModule(ProcessingModule):
    slug = "mono_camera"
    version = "1.0"
    category = "input"
    label = "RGB Camera"
    icon = "ant-design:video-camera-outlined"
    color = INPUT_COLOR
    outputs = ({"key": "video", "label": "RGB Video"},)
    default_config = {"source_key": "", "fps": 30}
    config_schema = (
        field("source_key", "string",
              "Device name / source key (blank = auto-detect)", ""),
    )
    execution_target = "server"
    capabilities = ("video_input", "mono")

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
