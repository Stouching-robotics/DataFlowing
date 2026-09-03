"""Module catalog — auto-derived from the processing module registry.

新增模块只需在 app/processing/modules/ 放一个文件,这里自动出现;
catalog 按 (category, slug) 显式排序,保证输出稳定。
"""

from __future__ import annotations

from app.processing.registry import all_descriptors

_CATEGORY_ORDER = {"input": 0, "process": 1, "review": 2, "export": 3}


def module_catalog() -> list[dict]:
    items = sorted(all_descriptors(),
                   key=lambda d: (_CATEGORY_ORDER.get(d.category, 9), d.slug))
    return [item.to_dict() for item in items]


def get_descriptor(slug: str):
    from app.workflow_types import canonical_node_type
    wanted = canonical_node_type(slug)
    return next((d for d in all_descriptors() if d.slug == wanted), None)
