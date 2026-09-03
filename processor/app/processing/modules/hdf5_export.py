"""HDF5 Export — 把批次构建为单一 .h5 数据集。

与 LeRobot Export 节点同源数据(视频/手部骨骼/传感器/标注),由
app.hdf5_export.build_hdf5_dataset 完成:视频逐帧抽帧 → uint8 数组
(gzip 压缩),表列按 LeRobot observation 语义组织成层级结构。
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import EXPORT_COLOR


@register
class HDF5ExportModule(ProcessingModule):
    slug = "hdf5_export"
    version = "1.1"
    category = "export"
    label = "HDF5 Export"
    icon = "ant-design:database-outlined"
    color = EXPORT_COLOR
    inputs = ({"key": "data", "label": "Exportable Data"},)
    outputs = ({"key": "dataset", "label": "Dataset"},)
    default_config = {"compression": "gzip", "level": 4}
    config_schema = (
        field("compression", "select", "Compression", "gzip", options=["gzip", "lzf"]),
        field("level", "number", "Compression level", 4, min=0, max=9),
    )
    execution_target = "worker"
    capabilities = ("dataset_export", "hdf5")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        if not ctx.incoming:
            ctx.skip("No upstream data — skipped")
        episode_id = str(ctx.job.get("episode_id") or "")
        if not episode_id:
            ctx.skip("No episode_id in job")

        from app.hdf5_export import build_hdf5_dataset
        compression = str(ctx.config.get("compression", "gzip"))
        level = int(ctx.config.get("level", 4) or 4)
        # 连接驱动:输入卡片(视频)连到本节点 → 抽帧该视频;没连 → 不抽帧。
        include_video_keys = [
            r.source_key for r in ctx.incoming.values()
            if r.kind == "video" and r.source_key
        ]
        hand_keypoints_paths = [
            str(p) for r in ctx.incoming.values()
            if r.kind == "hand_keypoints" and r.path
            for p in (ctx.resolve(r),) if p
        ]
        hand_3d_paths = [
            str(p) for r in ctx.incoming.values()
            if r.kind == "hand_3d" and r.path
            for p in (ctx.resolve(r),) if p
        ]
        hand_3d_unit = next(
            (r.metadata.get("unit") for r in ctx.incoming.values()
             if r.kind == "hand_3d" and getattr(r, "metadata", None)
             and r.metadata.get("unit")),
            None)
        out_path = build_hdf5_dataset(
            episode_id, [episode_id], ctx.output_root,
            include_video_keys=include_video_keys,
            hand_keypoints_paths=hand_keypoints_paths,
            hand_3d_paths=hand_3d_paths,
            hand_3d_unit=hand_3d_unit,
            compression=compression, level=level)
        if not out_path.exists():
            ctx.skip("HDF5 build produced no file")
        return {
            "dataset": ctx.ref("dataset", out_path, metadata={
                "root": str(out_path.relative_to(ctx.output_root)) if out_path != ctx.output_root else ".",
                "episode_id": episode_id,
            }),
        }
