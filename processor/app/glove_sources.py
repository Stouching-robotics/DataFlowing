"""Helpers for treating the two SenseGlove channels as one input device.

The recorder stores the left and right channels separately because they are
two data streams.  That storage detail must not leak into the workflow UI:
one glove device can still carry both ``left_glove`` and ``right_glove``
source keys in its node configuration.
"""

from __future__ import annotations

import re


_SIDE_TOKEN = re.compile(r"(^|[_\-.])(left|right)(?=[_\-.]|$)", re.IGNORECASE)


def _glove_base(key: str) -> str | None:
    """Return a stable pair key for a left/right glove channel."""
    value = str(key or "").strip()
    lower = value.lower()
    if "glove" not in lower:
        return None
    match = _SIDE_TOKEN.search(value)
    if not match:
        return None
    # Remove the separator together with the side token.  For example,
    # left_glove_joint/right_glove_joint both become ``glove_joint``.
    base = _SIDE_TOKEN.sub("_", value, count=1).strip("_-. ").lower()
    return base or "glove"


def group_glove_source_keys(values) -> list[list[str]]:
    """Group paired glove channels while preserving singleton channels.

    The result is deterministic and ordered left, right.  Non-glove sensor
    names are returned as singleton groups so this helper is safe to use on
    a complete ``info.json`` sensor list.
    """
    unique: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        key = str(raw or "").strip()
        if key and key.lower() not in seen:
            unique.append(key)
            seen.add(key.lower())

    grouped: dict[str, dict[str, str]] = {}
    singleton_order: list[str] = []
    for key in unique:
        base = _glove_base(key)
        match = _SIDE_TOKEN.search(key) if base else None
        side = match.group(2).lower() if match else ""
        if not base or side not in {"left", "right"}:
            singleton_order.append(key)
            continue
        bucket = grouped.setdefault(base, {})
        bucket.setdefault(side, key)

    result: list[list[str]] = []
    result.extend([key] for key in singleton_order)

    # Pair ordering is based on the first appearance of each base, while the
    # channels inside a pair always use left then right.
    bases: list[str] = []
    for key in unique:
        base = _glove_base(key)
        if base and base not in bases:
            bases.append(base)
    paired: list[list[str]] = []
    for base in bases:
        bucket = grouped.get(base, {})
        keys = [bucket[side] for side in ("left", "right") if side in bucket]
        if keys:
            paired.append(keys)

    # Keep original singleton order, followed by grouped glove channels.  In
    # the normal recorder layout this gives left_glove,right_glove together.
    return result + paired


def is_paired_glove_source(keys: list[str]) -> bool:
    """Whether a group contains both left and right glove channels."""
    sides = {
        (match.group(2).lower() if match else "")
        for key in keys
        for match in [_SIDE_TOKEN.search(str(key))]
        if match and "glove" in str(key).lower()
    }
    return sides == {"left", "right"}


def merge_paired_glove_nodes(graph: dict) -> bool:
    """Collapse legacy left/right glove input nodes in a workflow graph.

    Older auto-generated workflows contain one node per channel.  The first
    node is kept, both source keys are written to its config, and outgoing
    edges from the removed node are redirected to it.  Returns whether the
    graph changed.
    """
    if not isinstance(graph, dict):
        return False
    nodes = list(graph.get("nodes") or [])
    by_base: dict[str, list[dict]] = {}
    for node in nodes:
        data = node.get("data") or {}
        if data.get("nodeType") != "glove_sensor":
            continue
        config = data.get("config") or {}
        raw = config.get("source_keys") or config.get("source_key") or config.get("position")
        values = raw.split(",") if isinstance(raw, str) else raw or []
        keys = [str(value).strip() for value in values if str(value).strip()]
        bases = {_glove_base(key) for key in keys}
        for base in bases:
            if base:
                by_base.setdefault(base, []).append(node)

    changed = False
    redirects: dict[str, str] = {}
    remove_ids: set[str] = set()
    for duplicate_nodes in by_base.values():
        if len(duplicate_nodes) < 2:
            continue
        keeper = duplicate_nodes[0]
        keeper_data = keeper.setdefault("data", {})
        keeper_config = keeper_data.setdefault("config", {})
        all_keys: list[str] = []
        for node in duplicate_nodes:
            config = (node.get("data") or {}).get("config") or {}
            raw = config.get("source_keys") or config.get("source_key") or config.get("position")
            values = raw.split(",") if isinstance(raw, str) else raw or []
            for key in values:
                key = str(key).strip()
                if key and key.lower() not in {item.lower() for item in all_keys}:
                    all_keys.append(key)
        merged = group_glove_source_keys(all_keys)
        paired = next((group for group in merged if is_paired_glove_source(group)), None)
        if not paired:
            continue
        keeper_config["source_key"] = paired[0]
        keeper_config["source_keys"] = ",".join(paired)
        keeper_config.pop("position", None)
        keeper_data["label"] = keeper_data.get("label") or "Glove Sensor"
        for node in duplicate_nodes[1:]:
            node_id = str(node.get("id") or "")
            if node_id:
                redirects[node_id] = str(keeper.get("id") or "")
                remove_ids.add(node_id)
        changed = True

    if not changed:
        return False
    graph["nodes"] = [node for node in nodes
                      if str(node.get("id") or "") not in remove_ids]
    seen_edges: set[tuple[str, str, str, str]] = set()
    rewritten: list[dict] = []
    for edge in graph.get("edges") or []:
        source = redirects.get(str(edge.get("source") or ""), edge.get("source"))
        target = redirects.get(str(edge.get("target") or ""), edge.get("target"))
        if not source or not target:
            continue
        edge = dict(edge)
        edge["source"] = source
        edge["target"] = target
        identity = (str(source), str(edge.get("sourceHandle") or ""),
                    str(target), str(edge.get("targetHandle") or ""))
        if identity in seen_edges:
            continue
        seen_edges.add(identity)
        edge["id"] = (
            f"xy-edge__{source}{edge.get('sourceHandle') or ''}-"
            f"{target}{edge.get('targetHandle') or ''}")
        rewritten.append(edge)
    graph["edges"] = rewritten
    return True
