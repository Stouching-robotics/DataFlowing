"""Module registry — maps WorkflowNode.type to ProcessingModule instance.

模块在 app/processing/modules/ 下自动发现(见 __init__.py),按 slug 注册。
Worker 执行与前端画布目录都从本注册表取模块定义。
"""

from importlib import import_module

from app.processing import ProcessingModule, ModuleDescriptor
from app.workflow_types import canonical_node_type

_registry: dict[str, ProcessingModule] = {}


def register(cls):
    """Register a processing module by its slug (decorator)."""
    if not cls.slug:
        raise ValueError(f"Module {cls.__name__} has no slug")
    if cls.slug in _registry:
        raise ValueError(f"Duplicate module slug: {cls.slug}")
    _registry[cls.slug] = cls()
    return cls


def get(slug: str) -> ProcessingModule | None:
    """Look up a module by canonical slug or a historical alias."""
    return _registry.get(canonical_node_type(slug))


def all_modules() -> list[ProcessingModule]:
    """All registered modules."""
    return list(_registry.values())


def all_descriptors() -> list[ModuleDescriptor]:
    """All module descriptors (catalog source of truth)."""
    return [m.descriptor() for m in _registry.values()]


# Populate registry on import
# (modules register themselves via @register decorator on class definition)
import_module("app.processing.modules")
