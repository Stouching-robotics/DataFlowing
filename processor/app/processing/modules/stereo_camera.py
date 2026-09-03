"""Stereo Camera — 输入源模块:输出 video_left / video_right 双端口。

找到 ≥2 路视频时输出左右两路,供两个 MediaPipe 节点分别处理;
只有 1 路时退化为单 video 输出。
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import INPUT_COLOR


@register
class StereoCameraModule(ProcessingModule):
    slug = "stereo_camera"
    version = "1.0"
    category = "input"
    label = "Stereo RGB Camera"
    icon = "ant-design:video-camera-outlined"
    color = INPUT_COLOR
    outputs = ({"key": "video_left", "label": "Left RGB Video"},
               {"key": "video_right", "label": "Right RGB Video"})
    # source_keys 为设备名称(逗号分隔);未配置时 find_videos 自动识别，
    # 显式设备缺失时只处理实际命中的视频，不回退复用其他设备。
    # A workflow is authored before an upload exists. Empty source keys mean
    # "auto-match the real stereo pair in this episode"; concrete stream names
    # are injected only into the per-run snapshot.
    default_config = {"source_keys": "", "source_key": "", "position": "", "fps": 30}
    config_schema = (
        field("source_keys", "string",
              "Device names (comma; blank = auto-detect)",
              ""),
        field("source_key", "string", "Source key", ""),
    )
    execution_target = "server"
    capabilities = ("video_input", "stereo")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        videos = ctx.find_videos()
        if not videos:
            ctx.skip("Configured video source is missing — skipped")
        from app.lerobot_v21 import canonical_source_key, source_key_from_video
        actual_keys = [source_key_from_video(path, ctx.input_root / "videos")
                       for path in videos[:2]]
        keys = []
        raw = ctx.config.get("source_keys")
        if isinstance(raw, str):
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        keys = [canonical_source_key(key) for key in keys]
        if len(videos) >= 2:
            # 双目:输出左右两路,供两个 MediaPipe 节点分别处理
            return {
                "video_left": ctx.ref("video", videos[0],
                                      source_key=keys[0] if keys else actual_keys[0]),
                "video_right": ctx.ref("video", videos[1],
                                       source_key=keys[1] if len(keys) > 1 else actual_keys[1]),
            }
        selected_key = ctx.config.get("source_key") or actual_keys[0]
        if keys:
            path_lower = videos[0].as_posix().lower()
            selected_key = next(
                (key for key in keys if key.lower() in path_lower),
                selected_key,
            )
        return {"video": ctx.ref("video", videos[0],
                                 source_key=selected_key or None)}
