"""Processing Module Framework — pluggable pipeline nodes.

每个功能(相机输入、骨骼识别、手套、标注、审核、导出)是一个独立模块:
一个文件 = 一个模块(元数据 + run()),放在 app/processing/modules/ 下,
自动发现注册。前端画布与 worker 执行都从注册表取模块定义。

新增自定义模块三步:
1. 复制 modules/example 模板为 app/processing/modules/<slug>.py
2. 填写 slug/label/category/inputs/outputs/config_schema 等元数据,实现 run(ctx)
3. 重启后端与 worker —— 前端面板自动出现可拖拽卡片,worker 自动执行
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable


def field(name: str, kind: str, label: str, default=None, **extra) -> dict:
    """config_schema 单字段定义: {"name","type","label","default",**extra}。"""
    value = {"name": name, "type": kind, "label": label, "default": default}
    value.update(extra)
    return value


@dataclass
class ModuleResult:
    """Returned by a ProcessingModule after execution (legacy, kept for compat)."""
    success: bool
    data: dict[str, Any] = dc_field(default_factory=dict)
    outputs: dict[str, "ArtifactRef"] = dc_field(default_factory=dict)
    metrics: dict[str, Any] = dc_field(default_factory=dict)
    error: str | None = None


@dataclass
class ArtifactRef:
    """Portable reference to a workflow input/output artifact."""

    kind: str
    path: str | None = None
    source_key: str | None = None
    schema_version: str = "1.0"
    metadata: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "source_key": self.source_key,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactRef":
        return cls(
            kind=str(value.get("kind", "unknown")),
            path=value.get("path"),
            source_key=value.get("source_key"),
            schema_version=str(value.get("schema_version", "1.0")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ModuleDescriptor:
    """Metadata shared by the backend module registry and card palette."""

    slug: str
    version: str
    category: str
    label: str
    icon: str
    color: str
    inputs: tuple[dict[str, str], ...] = ()
    outputs: tuple[dict[str, str], ...] = ()
    default_config: dict[str, Any] = dc_field(default_factory=dict)
    config_schema: tuple[dict[str, Any], ...] = ()
    execution_target: str = "worker"
    capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.slug,
            "slug": self.slug,
            "version": self.version,
            "category": self.category,
            "label": self.label,
            "icon": self.icon,
            "color": self.color,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "defaultConfig": self.default_config,
            "default_config": self.default_config,
            "configSchema": list(self.config_schema),
            "config_schema": list(self.config_schema),
            "executionTarget": self.execution_target,
            "execution_target": self.execution_target,
            "capabilities": list(self.capabilities),
        }


class ModuleSkip(Exception):
    """模块主动跳过(无有效输入等)—— 节点标记 skipped,不视为失败。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class JobContext:
    """Worker 与模块之间的统一执行上下文。"""

    node_id: str
    node_type: str
    config: dict[str, Any]
    job: dict[str, Any]
    input_root: Path                       # 批次解压目录(源文件扫描)
    output_root: Path                      # 输出根(outputs/)
    incoming: dict[str, ArtifactRef]       # targetHandle → 上游 ArtifactRef(源节点为空 dict)
    progress: Callable[[float], None]      # 本节点进度 0..1
    node_states: dict                      # 节点状态累加器

    @property
    def output_dir(self) -> Path:
        d = self.output_root / self.node_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ref(self, kind: str, path: Path, source_key: str | None = None,
            metadata: dict | None = None) -> ArtifactRef:
        """构造 ArtifactRef,路径自动相对化:input_root 下 → 相对 input_root,否则相对 output_root。"""
        try:
            rel = path.resolve().relative_to(self.input_root.resolve())
        except ValueError:
            rel = path.resolve().relative_to(self.output_root.resolve())
        return ArtifactRef(kind=kind, path=str(rel).replace("\\", "/"),
                           source_key=source_key, metadata=metadata or {})

    def resolve(self, ref: ArtifactRef) -> Path | None:
        """按 input_root/output_root 依次解析 ArtifactRef 路径。"""
        if not ref or not ref.path:
            return None
        p = self.input_root / ref.path
        if p.exists():
            return p
        p = self.output_root / ref.path
        return p if p.exists() else None

    def find_videos(self) -> list[Path]:
        """按 config 的 source_keys/source_key/position 匹配批次内视频(见 batch.py)。"""
        from app.processing.batch import find_videos
        return find_videos(self.input_root, self.config)

    def find_parquet(self) -> list[Path]:
        """批次内全部 parquet(排除 auto_labels/hand_kpts 等合并产物)。"""
        from app.processing.batch import find_parquet
        return find_parquet(self.input_root)

    def skip(self, reason: str):
        raise ModuleSkip(reason)


class ProcessingModule:
    """Base class for a pipeline processing node.

    Subclass, fill in the metadata class attributes, implement ``run()``,
    and decorate with ``@register`` from app.processing.registry — the
    module appears in the Workflow Studio palette and runs in the worker
    automatically.
    """

    slug: str = ""
    version: str = "1.0"
    category: str = "process"          # input | process | review | export
    label: str = ""
    icon: str = "ant-design:appstore-outlined"
    color: str = "#64748b"
    inputs: tuple[dict[str, str], ...] = ()
    outputs: tuple[dict[str, str], ...] = ()
    default_config: dict[str, Any] = dc_field(default_factory=dict)
    config_schema: tuple[dict[str, Any], ...] = ()
    execution_target: str = "worker"
    capabilities: tuple[str, ...] = ()

    @classmethod
    def descriptor(cls) -> ModuleDescriptor:
        return ModuleDescriptor(
            slug=cls.slug, version=cls.version, category=cls.category, label=cls.label,
            icon=cls.icon, color=cls.color, inputs=tuple(cls.inputs),
            outputs=tuple(cls.outputs), default_config=dict(cls.default_config),
            config_schema=tuple(cls.config_schema), execution_target=cls.execution_target,
            capabilities=tuple(cls.capabilities),
        )

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        """执行入口。返回输出 handle → ArtifactRef;None 或 raise ModuleSkip → 节点 skipped。"""
        raise NotImplementedError
