"""Glove Sensor — 输入源模块:输出批次内手套传感器 parquet。"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import INPUT_COLOR


@register
class GloveSensorModule(ProcessingModule):
    slug = "glove_sensor"
    version = "1.0"
    category = "input"
    label = "Glove Sensor"
    icon = "ant-design:edit-outlined"
    color = INPUT_COLOR
    outputs = ({"key": "sensor_data", "label": "Glove Sensor Data"},)
    # The physical glove source is resolved from the uploaded episode. Keep
    # the workflow definition device-agnostic until that data exists. This
    # stream is intended for sensor quality review, not video annotation.
    default_config = {"source_key": "", "device": "SenseGlove",
                      "hand": "both", "fps": 60}
    config_schema = (field("source_key", "string", "Source key", ""),)
    execution_target = "server"
    capabilities = ("sensor_input", "glove")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        parquets = ctx.find_parquet()
        if not parquets:
            ctx.skip("No parquet found in batch")
        return {"sensor_data": ctx.ref("glove_sensor", parquets[0],
                                       source_key=ctx.config.get("source_key"))}
