"""Template workflow instantiation — 上传数据自动补齐输入卡片并连线。

模板工作流 = graph 中只含处理节点(mediapipe_hand/annotation/human_review/
lerobot_export)和它们之间的边,输入卡片被删掉。约定式判定:存在"无
incoming 边"的 process/review/export 输入 handle 即视为模板。

上传数据时(session.py 钩子)识别输入源(相机名 + info.json sensors),
调用 ``complete_graph`` 补齐输入卡片并自动连线,生成完整工作流实例
(可编辑、非预设,``template_id`` 指向模板)绑定项目;后续上传出现新
输入源 → ``increment_instance`` 增量补全(只新增,不改既有节点/边)。

连线规则:
- 视频类输入(rgb/fisheye/stereo)→ 每路视频连一个 mediapipe_hand 的
  ``video`` 输入(mediapipe_hand.run 只处理第一个 video incoming):
  先复用模板中悬空的 mediapipe_hand,不够则深拷贝原型。
- 手套传感器(sensor_data)→ 绝不连 video 端口;第一个手套连到模板中
  悬空的 ``data`` 输入(如 annotation),后续手套/无 data 输入时照建
  不连线(worker 中 glove_sensor.run 自行 find_parquet)。
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from uuid import uuid4

from app.localstore import get_workflow, upsert_project, upsert_workflow
from app.glove_sources import (
    group_glove_source_keys,
    is_paired_glove_source,
    merge_paired_glove_nodes,
)
from app.processing.batch import _keys_of
from app.processing.catalog import get_descriptor
from app.workflow_types import HAND_PROCESS_TYPES, migrate_graph_types

# ``mono_camera`` is the canonical input module for newly detected single-view
# cameras. Keep the historical aliases so old graphs still participate in
# increment-instance de-duplication.
INPUT_MODULES = {
    "mono_camera", "rgbd_camera", "rgb_camera", "fisheye_camera",
    "stereo_camera", "stereo_rgbd_camera", "glove_sensor",
}
VIDEO_PROCESS_TYPES = {
    "mediapipe_hand", *HAND_PROCESS_TYPES,
}
MULTI_VIEW_VIDEO_TYPES = set(HAND_PROCESS_TYPES)
EDGE_STYLE = {"stroke": "#475569", "strokeWidth": 2}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 模板判定 ──────────────────────────────────────────

def _incoming_handles(edges: list[dict]) -> dict[str, list[str]]:
    """node_id → 已被喂入的 targetHandle key 列表(含缺省 "data")。"""
    incoming: dict[str, list[str]] = {}
    for e in edges:
        target = e.get("target")
        if target:
            incoming.setdefault(str(target), []).append(e.get("targetHandle") or "data")
    return incoming


def _dangling_inputs(nodes: list[dict], edges: list[dict]) -> dict[str, list[str]]:
    """每个节点的悬空输入 handle key 列表(仅 process/review/export)。

    input 类节点无 inputs,天然排除;已被入边喂满的输入端口不算悬空。
    """
    incoming = _incoming_handles(edges)
    out: dict[str, list[str]] = {}
    for n in nodes:
        data = n.get("data") or {}
        if data.get("category") == "input":
            continue
        fed = {h.lower() for h in incoming.get(str(n.get("id")), [])}
        dangling = [
            i.get("key") for i in (data.get("inputs") or [])
            if i.get("key") and i.get("key").lower() not in fed
        ]
        if dangling:
            out[str(n["id"])] = dangling
    return out


def is_template_workflow(workflow: dict) -> bool:
    """约定式模板判定:graph 中存在无 incoming 边的 process 输入 handle。"""
    graph = workflow.get("graph") or {}
    return bool(_dangling_inputs(graph.get("nodes") or [], graph.get("edges") or []))


# ── 基础构造工具 ──────────────────────────────────────

def _alloc_node_id(used: set[str]) -> str:
    """``node_<max+1>`` 递增分配,保证与现有节点不冲突。"""
    max_n = 0
    for nid in used:
        if isinstance(nid, str) and nid.startswith("node_"):
            try:
                max_n = max(max_n, int(nid[5:]))
            except ValueError:
                pass
    nid = f"node_{max_n + 1}"
    used.add(nid)
    return nid


def _edge_id(source: str, source_handle: str, target: str, target_handle: str) -> str:
    return f"xy-edge__{source}{source_handle}-{target}{target_handle}"


def _camera_module(camera: str) -> str:
    """相机名 → 输入模块映射(在剔除 _aux 之后调用)。"""
    c = str(camera).lower()
    if "stereo" in c and "depth" in c:
        return "stereo_rgbd_camera"
    if "stereo" in c:
        return "stereo_camera"
    if ("rgbd" in c or ("depth" in c and any(
            token in c for token in ("rgb", "color")))):
        return "rgbd_camera"
    # New instances use one generic single-view input.  Keep rgb_camera and
    # fisheye_camera in INPUT_MODULES so historical graphs remain valid, but
    # do not create new model/lens-specific cards.
    return "mono_camera"


def _paired_camera_key(camera: str, available: set[str]) -> str | None:
    """Find the matching left/right stream without relying on a model name."""
    value = str(camera)
    replacements = (
        ("_left_", "_right_"), ("_right_", "_left_"),
        ("_left", "_right"), ("_right", "_left"),
    )
    for old, new in replacements:
        if old not in value.lower():
            continue
        candidate = value.lower().replace(old, new)
        exact = next((item for item in available if item.lower() == candidate), None)
        if exact:
            return exact
    return None


def _camera_groups(cameras: list[str]) -> list[list[str]]:
    """Group generic left/right stream names into physical stereo cameras.

    This handles ``stereo_left/right`` as well as future names such as
    ``head_left_rgb/head_right_rgb``. Unpaired streams remain mono inputs.
    """
    values = sorted({str(camera).strip() for camera in cameras if str(camera).strip()})
    remaining = set(values)
    groups: list[list[str]] = []
    for camera in values:
        if camera not in remaining:
            continue
        remaining.remove(camera)
        pair = _paired_camera_key(camera, remaining)
        if pair:
            remaining.remove(pair)
            groups.append([camera, pair])
        else:
            groups.append([camera])
    return groups


def _input_node(node_type: str, source_keys: list[str], x: float, y: float,
                used: set[str]) -> dict:
    """构造完整 React Flow 输入节点(与现网手动搭建的节点同构)。

    config 取自模块 default_config,并把 source_key/source_keys 覆写为
    实际检测到的名字 —— worker 只读 node.data.config,这是唯一生效位。
    """
    desc = get_descriptor(node_type)
    if desc is None:
        raise ValueError(f"Unknown input module: {node_type}")
    d = desc.to_dict()
    config = dict(d.get("default_config") or {})
    if len(source_keys) > 1:
        config["source_keys"] = ",".join(source_keys)
        config["source_key"] = source_keys[0]
    else:
        config["source_key"] = source_keys[0]
        config.pop("source_keys", None)
    data = {
        "label": d.get("label"),
        "category": "input",
        "icon": d.get("icon"),
        "color": d.get("color"),
        "config": config,
        "nodeType": node_type,
        "inputs": list(d.get("inputs") or []),
        "outputs": list(d.get("outputs") or []),
        "configSchema": list(d.get("config_schema") or []),
        "executionTarget": d.get("execution_target") or "server",
    }
    return {
        "id": _alloc_node_id(used),
        "type": "workflowNode",
        "position": {"x": x, "y": y},
        "data": data,
        "measured": {"width": 280, "height": 86},
        "selected": False,
        "dragging": False,
    }


def _copy_video_node(proto: dict, copy_k: int, used: set[str]) -> dict:
    """深拷贝视频处理节点原型(新 id、原型正下方偏移)。

    拷贝不带任何下游边 —— 处理节点本身只返回临时结果，完成阶段合并到
    对应 episode parquet；
    双目新模块可通过同一 video handle 的重复边接收 video#1。
    """
    node = copy.deepcopy(proto)
    node["id"] = _alloc_node_id(used)
    pos = node.get("position") or {}
    node["position"] = {
        "x": pos.get("x", 0),
        "y": pos.get("y", 0) + (copy_k + 1) * 220,
    }
    node["selected"] = False
    node["dragging"] = False
    return node


def _make_edge(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {
        "style": dict(EDGE_STYLE),
        "source": source,
        "sourceHandle": source_handle,
        "target": target,
        "targetHandle": target_handle,
        "id": _edge_id(source, source_handle, target, target_handle),
    }


# ── 核心算法 ──────────────────────────────────────────

def complete_graph(graph: dict, camera_names: list[str], sensors: list[str]) -> tuple[list[dict], list[dict], dict]:
    """在任意 graph 上补齐输入子图(创建与增量共用)。

    返回 ``(nodes, edges, added)`` —— 普通输入只追加内容；左右手套
    传感器会合并到一个节点，兼容并修正历史重复节点。
    """
    migrated, _ = migrate_graph_types(graph)
    nodes = migrated.get("nodes") or []
    edges = migrated.get("edges") or []
    normalized_graph = {"nodes": nodes, "edges": edges}
    merge_paired_glove_nodes(normalized_graph)
    nodes = normalized_graph["nodes"]
    edges = normalized_graph["edges"]
    used = {str(n.get("id")) for n in nodes}
    existing_edge_ids = {e.get("id") for e in edges}
    added = {"video_nodes": 0, "mp_copies": 0,
             "processor_copies": 0, "glove_nodes": 0}

    # ── 1. 已有输入覆盖集(增量去重:nodeType + source_key)──
    covered: set[tuple[str, str]] = set()
    for n in nodes:
        data = n.get("data") or {}
        if data.get("nodeType") not in INPUT_MODULES:
            continue
        for k in _keys_of(data.get("config") or {}):
            covered.add((data["nodeType"], str(k).lower()))

    # ── 2. 期望输入源(相机剔除 _aux;传感器来自 info.json)──
    mains = [c for c in (camera_names or []) if not str(c).lower().endswith("_aux")]
    video_groups: list[tuple[str, list[str]]] = []
    existing_stereo_keys = {
        key for module, key in covered
        if module in {"stereo_camera", "stereo_rgbd_camera"}
    }
    for group in _camera_groups(mains):
        mod = "stereo_camera" if len(group) > 1 else _camera_module(group[0])
        if mod in {"stereo_camera", "stereo_rgbd_camera"}:
            # A complete existing pair is already covered. An unpaired
            # historical stereo node is allowed to receive a later pair in a
            # future migration rather than suppressing every new camera.
            if all(key.lower() in existing_stereo_keys for key in group):
                continue
            video_groups.append((mod, group))
        else:
            c = group[0]
            if (mod, c.lower()) in covered:
                continue
            video_groups.append((mod, group))

    # 存量工作流可能已有一个 left_glove 节点。发现成对传感器时直接把
    # 右手 source key 补到该节点，不再新建第二个 SenseGlove 节点。
    for glove_group in group_glove_source_keys(sensors):
        if not is_paired_glove_source(glove_group):
            continue
        group_lower = [str(key).lower() for key in glove_group]
        for node in nodes:
            data = node.get("data") or {}
            if data.get("nodeType") != "glove_sensor":
                continue
            config = data.setdefault("config", {})
            existing = _keys_of(config)
            if not existing or not any(
                    old.lower() in new or new in old.lower()
                    for old in existing for new in group_lower):
                continue
            config["source_key"] = glove_group[0]
            config["source_keys"] = ",".join(glove_group)
            config.pop("position", None)
            break

    desired_glove = [
        keys for keys in group_glove_source_keys(sensors)
        if not any(("glove_sensor", str(key).lower()) in covered for key in keys)
    ]

    # ── 3. 视频处理节点(模板复用池)+ 原型 ──
    # MediaPipe 保持旧的一路一节点语义；新的 Hand3D/黑手套模块可在
    # 同一个节点中接收 stereo_camera 的两路视频。
    video_fed = {str(e.get("target")) for e in edges
                 if (e.get("targetHandle") or "data") == "video"}
    video_processors: dict[str, list[dict]] = {}
    for n in nodes:
        node_type = (n.get("data") or {}).get("nodeType")
        if node_type in VIDEO_PROCESS_TYPES:
            video_processors.setdefault(node_type, []).append(n)
    dangling_processors = {
        node_type: [n for n in items
                    if str(n.get("id")) not in video_fed]
        for node_type, items in video_processors.items()
    }

    # ── 4. 布局:输入列在最左,复制的处理节点在原型正下方 ──
    proc_nodes = [n for n in nodes if (n.get("data") or {}).get("category") != "input"]
    min_x = min((n.get("position") or {}).get("x", 0) for n in proc_nodes) if proc_nodes else 0
    min_y = min((n.get("position") or {}).get("y", 0) for n in proc_nodes) if proc_nodes else 0
    input_x = min_x - 340
    row = 0
    copy_k = 0

    # ── 5. 视频输入节点:连接模板中的视频处理节点 ──
    for mod, keys in video_groups:
        if mod in {"stereo_camera", "stereo_rgbd_camera"}:
            handles = ["video_left"] if len(keys) < 2 else ["video_left", "video_right"]
        else:
            handles = ["video"]
        # 没有视频处理节点原型→ 不建孤立输入节点
        if not video_processors:
            continue
        node = _input_node(mod, keys, input_x, min_y + row * 180, used)
        nodes.append(node)
        added["video_nodes"] += 1
        row += 1
        for processor_type, prototypes in video_processors.items():
            prototype = prototypes[0]
            reusable = dangling_processors.get(processor_type) or []
            # Multi-view modules consume both refs in one run. Legacy
            # MediaPipe consumes only the first ref, so it gets one node/view.
            groups = [handles] if (
                mod in {"stereo_camera", "stereo_rgbd_camera"}
                and len(handles) > 1
                and processor_type in MULTI_VIEW_VIDEO_TYPES
            ) else [[h] for h in handles]
            for group in groups:
                if reusable:
                    processor = reusable.pop(0)
                else:
                    processor = _copy_video_node(prototype, copy_k, used)
                    nodes.append(processor)
                    added["mp_copies"] += 1
                    added["processor_copies"] += 1
                    copy_k += 1
                for h in group:
                    e = _make_edge(node["id"], h, processor["id"], "video")
                    if e["id"] not in existing_edge_ids:
                        edges.append(e)
                        existing_edge_ids.add(e["id"])
                # RGB-D processing has a typed depth input. The input card
                # publishes the paired depth directory; keep the historical
                # worker-side auto-pair fallback for old uploads that lack a
                # usable sidecar, but make the normal graph connection explicit.
                if (mod in {"rgbd_camera", "stereo_rgbd_camera"}
                        and processor_type in {
                            "rgbd_to_3d_bare_hand", "rgbd_to_3d_black_glove",
                        }):
                    e = _make_edge(node["id"], "depth", processor["id"], "depth")
                    if e["id"] not in existing_edge_ids:
                        edges.append(e)
                        existing_edge_ids.add(e["id"])

    # ── 6. 手套传感器:绝不连 video 端口 ──
    dangling_by_key: dict[str, str] = {}
    for nid, keys in _dangling_inputs(nodes, edges).items():
        for k in keys:
            dangling_by_key.setdefault(k, nid)   # 每个 handle key 只取第一个目标
    data_target = dangling_by_key.get("data")

    glove_wired = False
    for source_keys in desired_glove:
        g = _input_node("glove_sensor", [str(key) for key in source_keys],
                        input_x, min_y + row * 180, used)
        nodes.append(g)
        added["glove_nodes"] += 1
        row += 1
        if data_target and not glove_wired:
            e = _make_edge(g["id"], "sensor_data", data_target, "data")
            if e["id"] not in existing_edge_ids:
                edges.append(e)
                existing_edge_ids.add(e["id"])
                glove_wired = True

    return nodes, edges, added


# ── 实例生命周期 ──────────────────────────────────────

def build_instance(template: dict, camera_names: list[str], sensors: list[str]) -> dict:
    """由模板 + 输入源生成完整工作流实例(非预设、可编辑)。"""
    nodes, edges, _added = complete_graph(
        template.get("graph") or {}, camera_names, sensors)
    return {
        "id": str(uuid4()),
        "name": f"{template.get('name') or 'Workflow'} (auto)",
        "description": template.get("description"),
        "graph": {"nodes": nodes, "edges": edges},
        "node_configs": template.get("node_configs") or {},
        "status": "active",
        "is_preset": False,
        "template_id": template["id"],
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
    }


def increment_instance(instance: dict, camera_names: list[str], sensors: list[str]) -> bool:
    """增量补全:有新输入源 → 只新增节点/边,写回;无变化返回 False。"""
    nodes, edges, added = complete_graph(
        instance.get("graph") or {}, camera_names, sensors)
    if not any(added.values()):
        return False
    instance["graph"] = {"nodes": nodes, "edges": edges}
    instance["updated_at"] = _utcnow_iso()
    upsert_workflow(instance)
    return True


def _find_instance(project: dict, template_id: str) -> dict | None:
    """项目绑定中 ``template_id`` 匹配、非预设的最新实例。"""
    if not project:
        return None
    wf_ids = project.get("workflow_ids") or []
    if not isinstance(wf_ids, list):
        wf_ids = [wf_ids] if wf_ids else []
    candidates = [
        w for w in (get_workflow(i) for i in wf_ids)
        if w and w.get("template_id") == template_id and not w.get("is_preset")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda w: w.get("updated_at") or "")


def _bind_to_project(project: dict, workflow_id: str) -> None:
    """实例 id 去重追加到项目 workflow_ids 并写回。"""
    wf_ids = project.get("workflow_ids") or []
    if not isinstance(wf_ids, list):
        wf_ids = [wf_ids] if wf_ids else []
    if workflow_id not in wf_ids:
        wf_ids.append(workflow_id)
    project["workflow_ids"] = wf_ids
    upsert_project(project)


def batch_covers_instance(instance: dict, camera_names: list[str]) -> bool:
    """入队前判定:实例的 video 输入键被本批次全覆盖才允许入队。

    手套传感器不参与判定(glove_sensor.run 自行 find_parquet);
    stereo 部分覆盖放行(模块退化单路,安全);缺主目视频 → False。
    """
    graph = instance.get("graph") or {}
    have = {
        str(c).lower() for c in (camera_names or [])
        if not str(c).lower().endswith("_aux")
    }
    for n in graph.get("nodes", []):
        data = n.get("data") or {}
        if data.get("category") != "input" or data.get("nodeType") == "glove_sensor":
            continue
        keys = [str(k).lower() for k in _keys_of(data.get("config") or {})]
        if not keys:
            continue
        if data.get("nodeType") in {"stereo_camera", "stereo_rgbd_camera"}:
            if not any(k in have for k in keys):
                return False
        else:
            if not all(k in have for k in keys):
                return False
    return True
