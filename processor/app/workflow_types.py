"""Canonical workflow node types and backward-compatible graph migration.

The workflow JSON is an external contract between the editor, API and worker.
Keep the user-facing names and the persisted node types explicit here so a
module rename cannot silently break an old saved workflow.
"""

from __future__ import annotations

import copy
from typing import Any


# Historical slugs are aliases only. New workflow definitions use the
# canonical semantic names below.
LEGACY_TO_CANONICAL: dict[str, str] = {
    "rgb_hand_3d": "rgb_to_2d_bare_hand",
    "black_hand_rgb_3d": "rgb_to_2d_black_glove",
    "stereo_triangulate": "rgbd_to_3d_bare_hand",
    "black_glove_hand": "rgbd_to_3d_black_glove",
}

RGB_2D_TYPES = {
    "rgb_to_2d_bare_hand",
    "rgb_to_2d_black_glove",
}
RGBD_3D_TYPES = {
    "rgbd_to_3d_bare_hand",
    "rgbd_to_3d_black_glove",
}
HAND_PROCESS_TYPES = RGB_2D_TYPES | RGBD_3D_TYPES
ANNOTATION_TYPES = {"annotation", "ai_annotation"}

CANONICAL_LABELS: dict[str, str] = {
    "rgb_to_2d_bare_hand": "RGB_TO_2D_BareHand",
    "rgb_to_2d_black_glove": "RGB_TO_2D_BlackGlove",
    "rgbd_to_3d_bare_hand": "RGB-D_3D_BareHand",
    "rgbd_to_3d_black_glove": "RGB-D_3D_BlackGlove",
}


def canonical_node_type(value: Any) -> str:
    """Return the canonical slug while accepting historical aliases."""
    raw = str(value or "").strip()
    return LEGACY_TO_CANONICAL.get(raw.lower(), raw)


def _port(key: str, label: str) -> dict[str, str]:
    return {"key": key, "label": label}


def _migrate_hand_ports(node_type: str, data: dict[str, Any]) -> bool:
    """Upgrade stored hand-module port snapshots to the semantic contract."""
    changed = False
    if node_type in ANNOTATION_TYPES:
        # ``data`` is retained as the persisted handle so old edges continue
        # to load, while the visible contract clearly says RGB Video.
        old_inputs = data.get("inputs") or []
        inputs = [_port("data", "RGB Video")]
        if inputs != old_inputs:
            data["inputs"] = inputs
            changed = True
    if node_type == "rgbd_camera":
        old_outputs = data.get("outputs") or []
        outputs = [item for item in old_outputs if isinstance(item, dict)]
        keys = {str(item.get("key") or "") for item in outputs}
        if "video" not in keys:
            outputs.insert(0, _port("video", "RGB Video"))
        if "depth" not in keys:
            outputs.append(_port("depth", "Depth"))
        outputs = [
            _port(str(item.get("key") or ""),
                  "RGB Video" if str(item.get("key") or "") == "video"
                  else "Depth" if str(item.get("key") or "") == "depth"
                  else str(item.get("label") or ""))
            if isinstance(item, dict) else item
            for item in outputs
        ]
        if outputs != old_outputs:
            data["outputs"] = outputs
            changed = True
    if node_type == "stereo_rgbd_camera":
        old_outputs = data.get("outputs") or []
        outputs = [
            _port("video_left", "Left RGB Video"),
            _port("video_right", "Right RGB Video"),
            _port("depth", "Depth"),
        ]
        if outputs != old_outputs:
            data["outputs"] = outputs
            changed = True
    if node_type in RGB_2D_TYPES:
        old_outputs = data.get("outputs") or []
        # The public contract is intentionally one output. Multi-view files
        # remain internal artifacts and are not separate connection ports.
        outputs = [_port("hand_keypoints", "Hand 2D")]
        if outputs != old_outputs:
            data["outputs"] = outputs
            changed = True

    if node_type in RGBD_3D_TYPES:
        old_inputs = data.get("inputs") or []
        inputs = [item for item in old_inputs if isinstance(item, dict)]
        keys = {str(item.get("key") or "") for item in inputs}
        if "video" not in keys:
            inputs.insert(0, _port("video", "RGB Video"))
        if "depth" not in keys:
            inputs.append(_port("depth", "Depth"))
        inputs = [
            _port(str(item.get("key") or ""),
                  "RGB Video" if str(item.get("key") or "") == "video"
                  else "Depth" if str(item.get("key") or "") == "depth"
                  else str(item.get("label") or ""))
            if isinstance(item, dict) else item
            for item in inputs
        ]
        if inputs != old_inputs:
            data["inputs"] = inputs
            changed = True
        old_outputs = data.get("outputs") or []
        outputs = [_port("hand_3d", "Hand 3D")]
        if outputs != old_outputs:
            data["outputs"] = outputs
            changed = True

    return changed


def migrate_graph_types(graph: dict | None) -> tuple[dict, bool]:
    """Return a migrated graph and whether its persisted representation changed.

    The function is deliberately pure: callers can use it for API responses,
    worker snapshots and on-disk migration without mutating a cached graph.
    """
    result = copy.deepcopy(graph or {})
    nodes = result.get("nodes") or []
    node_types: dict[str, str] = {}
    changed = False

    for node in nodes:
        if not isinstance(node, dict):
            continue
        data = node.setdefault("data", {})
        if not isinstance(data, dict):
            continue
        raw_type = str(data.get("nodeType") or "")
        node_type = canonical_node_type(raw_type)
        if raw_type and node_type != raw_type:
            data["nodeType"] = node_type
            changed = True
        node_types[str(node.get("id") or "")] = node_type
        if node_type in CANONICAL_LABELS and data.get("label") != CANONICAL_LABELS[node_type]:
            data["label"] = CANONICAL_LABELS[node_type]
            changed = True
        if _migrate_hand_ports(node_type, data):
            changed = True

    # RGB-only nodes now expose 2D keypoints. Rewrite historical edge handles
    # so review/annotation/export continue receiving the actual public output.
    for edge in result.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source_type = node_types.get(str(edge.get("source") or ""))
        handle = str(edge.get("sourceHandle") or "")
        result_output = (
            "annotation" if source_type in {"annotation", "ai_annotation"}
            else "reviewed" if source_type in {"human_review", "ai_quality_review"}
            else ""
        )
        if result_output and handle == "result":
            edge["sourceHandle"] = result_output
            edge["id"] = (
                f"xy-edge__{edge.get('source') or ''}"
                f"{edge.get('sourceHandle') or ''}-"
                f"{edge.get('target') or ''}{edge.get('targetHandle') or ''}"
            )
            changed = True
            continue
        if source_type not in RGB_2D_TYPES:
            continue
        if handle == "hand_3d" or handle.startswith("hand_3d#"):
            suffix = handle[len("hand_3d"):]
            edge["sourceHandle"] = f"hand_keypoints{suffix}"
            edge["id"] = (
                f"xy-edge__{edge.get('source') or ''}"
                f"{edge.get('sourceHandle') or ''}-"
                f"{edge.get('target') or ''}{edge.get('targetHandle') or ''}"
            )
            changed = True

    # A few early black-glove snapshots exposed an intermediate 2D handle in
    # addition to Hand 3D. The depth-based card now has one public output.
    for edge in result.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        if node_types.get(str(edge.get("source") or "")) not in RGBD_3D_TYPES:
            continue
        handle = str(edge.get("sourceHandle") or "")
        if handle == "hand_keypoints" or handle.startswith("hand_keypoints#"):
            edge["sourceHandle"] = "hand_3d" + handle[len("hand_keypoints"):]
            edge["id"] = (
                f"xy-edge__{edge.get('source') or ''}"
                f"{edge.get('sourceHandle') or ''}-"
                f"{edge.get('target') or ''}{edge.get('targetHandle') or ''}"
            )
            changed = True

    # If an old workflow already used an RGB-D input card, complete its new
    # typed depth edge. Do not invent a depth source for old mono cards: those
    # remain executable through the processing module's legacy auto-pair path.
    existing_edges = {
        (str(edge.get("source") or ""), str(edge.get("target") or ""),
         str(edge.get("targetHandle") or ""))
        for edge in result.get("edges") or []
        if isinstance(edge, dict)
    }
    for edge in list(result.get("edges") or []):
        if not isinstance(edge, dict) or (edge.get("targetHandle") or "") != "video":
            continue
        source_id = str(edge.get("source") or "")
        target_id = str(edge.get("target") or "")
        if node_types.get(source_id) not in {"rgbd_camera", "stereo_rgbd_camera"}:
            continue
        if node_types.get(target_id) not in RGBD_3D_TYPES:
            continue
        identity = (source_id, target_id, "depth")
        if identity in existing_edges:
            continue
        depth_edge = dict(edge)
        depth_edge["sourceHandle"] = "depth"
        depth_edge["targetHandle"] = "depth"
        depth_edge["id"] = (
            f"xy-edge__{source_id}depth-{target_id}depth"
        )
        result.setdefault("edges", []).append(depth_edge)
        existing_edges.add(identity)
        changed = True

    result["nodes"] = nodes
    return result, changed


def migrate_workflow_records(records: list[dict] | None) -> tuple[list[dict], bool]:
    """Migrate all saved workflow definitions, preserving unrelated fields."""
    migrated: list[dict] = []
    changed = False
    for record in records or []:
        if not isinstance(record, dict):
            migrated.append(record)
            continue
        item = copy.deepcopy(record)
        item["graph"], graph_changed = migrate_graph_types(item.get("graph"))
        changed = changed or graph_changed
        migrated.append(item)
    return migrated, changed
