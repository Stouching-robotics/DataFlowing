"""Project-level device naming bindings — 工作流全局共享,项目按卡片覆盖设备名。

主流方案:工作流 = 处理逻辑(共享一份);项目 = 配置。项目对某工作流的
输入卡片(source_key/source_keys)做本项目覆盖(如头戴式 head_fisheye_rgb
vs 胸戴式 chest_fisheye_rgb),互不污染,改逻辑一次全项目生效。

绑定存储:projects.json 项目对象 ``workflow_bindings``:
    {"<工作流id>": {"<node_id>": {"source_key": "chest_fisheye_rgb"}}}

两个纯函数:
- ``effective_input_keys(workflow, bindings)``:上传匹配用 —— 合并绑定后
  的工作流期望输入键(与 _workflow_input_source_keys 同规则)。
- ``apply_bindings(graph, bindings)``:入队快照用 —— **深拷贝** graph 并把
  绑定写进输入节点 data.config(worker 只读 node.data.config,不应用
  node_configs;必须深拷贝,否则污染共享工作流定义)。
"""

from __future__ import annotations

import copy


def input_keys_match(want: list[str], available) -> bool:
    """工作流期望键 vs 批次相机名的**双向子串**匹配。

    设备名("D435")配相机全名("D435_depth_rgb")、或工作流全名配批次
    别名均命中。此前调用方用 set 精确相等判断(`w in have_primary`),
    前缀型设备名永远不命中 → 上传/重跑误报 No match。
    """
    avail = [str(a).lower() for a in (available or [])]
    for w in want:
        wl = str(w).lower()
        for a in avail:
            if wl in a or a in wl:
                return True
    return False


def matched_input_keys(want: list[str], available) -> list[tuple[str, str]]:
    """Return one-to-one ``(workflow_key, available_key)`` matches.

    The boolean matcher above answers whether any input is compatible. This
    helper is used when a workflow has multiple input devices so one uploaded
    stream cannot silently satisfy two different requested devices.
    """
    available_values = []
    seen_available: set[str] = set()
    for value in available or []:
        original = str(value).strip()
        lowered = original.lower()
        if not original or lowered in seen_available:
            continue
        seen_available.add(lowered)
        available_values.append((original, lowered))

    matches: list[tuple[str, str]] = []
    used_available: set[str] = set()
    for value in want or []:
        requested = str(value).strip()
        requested_lower = requested.lower()
        if not requested:
            continue
        candidates = [
            (original, lowered) for original, lowered in available_values
            if lowered not in used_available
            and (requested_lower in lowered or lowered in requested_lower)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: (
            item[1] != requested_lower,
            abs(len(item[1]) - len(requested_lower)),
            item[1],
        ))
        selected = candidates[0]
        used_available.add(selected[1])
        matches.append((requested, selected[0]))
    return matches


def effective_input_keys(workflow: dict, bindings: dict | None = None) -> list[str]:
    """工作流期望的输入源键,叠加项目绑定覆盖(绑定优先于节点 config)。

    ``bindings``: 项目对该工作流的 ``{node_id: {"source_key"|"source_keys": ...}}``。
    读取规则与 app.api.workflows._workflow_input_source_keys 一致:
    source_keys(逗号拆分)→ source_key → position。
    """
    graph = workflow.get("graph") or {}
    overrides = workflow.get("node_configs") or {}
    keys: list[str] = []
    for node in graph.get("nodes", []):
        node_id = node.get("id")
        data = node.get("data") or {}
        config = dict(data.get("config") or {})
        override = overrides.get(node_id, {}) if isinstance(overrides, dict) else {}
        if isinstance(override, dict):
            config.update(override)
        binding = (bindings or {}).get(node_id)
        if isinstance(binding, dict):
            _apply_binding_value(config, binding)
        values = (config.get("source_keys") or config.get("source_key") or config.get("position"))
        if isinstance(values, str):
            values = [v.strip() for v in values.split(",") if v.strip()]
        if isinstance(values, (list, tuple, set)):
            keys.extend(str(value) for value in values if value)
    return keys


def _apply_binding_value(config: dict, binding: dict) -> None:
    """绑定语义 = 节点输入命名:同时覆盖 source_keys(多源逗号)与
    source_key(首源)。工作流匹配读取顺序是 source_keys 优先,只改
    source_key 对 stereo(有 source_keys)不生效 —— 两处必须一致。
    """
    bound = str(binding.get("source_keys") or binding.get("source_key") or "").strip()
    if not bound:
        return
    config["source_keys"] = bound
    config["source_key"] = bound.split(",")[0].strip()


def auto_bindings_for_batch(workflow: dict, camera_names: list[str]) -> dict | None:
    """类型兜底匹配:精确名不匹配时,按单目/双目类型自动对齐输入名。

    单目工作流(mono_camera/fisheye/rgb 输入节点)+ 批次单路主目 → 输入
    节点 source_key 覆盖为批次主目名;双目工作流(stereo_camera)+ 批次
    多路主目 → source_keys 覆盖为批次主目名。返回绑定 dict(供
    apply_bindings 注入入队快照,不改工作流定义);类型不匹配返回 None
    (仍走不兼容异常)。

    主目判定只数 RGB 相机:深度相机(d435_depth 等)不是视频流,不能
    参与单/双目类型判断 —— 单 D435 批次(d435_rgb + d435_depth)必须
    判为单目,否则单目工作流会被误判不兼容(failed)。
    """
    graph = workflow.get("graph") or {}
    # A depth device can legitimately publish a color video slot such as
    # ``D435_depth_rgb``.  Exclude only pure depth streams; do not discard a
    # real RGB video merely because its parent device name contains "depth".
    def is_primary_video(name: str) -> bool:
        low = str(name).lower()
        if low.endswith("_aux"):
            return False
        return not ("depth" in low and "rgb" not in low)

    mains = [str(c) for c in (camera_names or []) if is_primary_video(str(c))]
    if not mains:
        return None
    # Do not treat any two cameras as a stereo pair. Two independent mono
    # cameras are valid, too; stereo requires a conventional left/right pair
    # (or an explicit stereo stream name).
    def has_stereo_pair(values: list[str]) -> bool:
        available = {str(value).strip().lower() for value in values}
        for value in available:
            replacements = (
                ("_left_", "_right_"), ("_right_", "_left_"),
                ("_left", "_right"), ("_right", "_left"),
            )
            for old, new in replacements:
                if old in value and value.replace(old, new) in available:
                    return True
        return False

    is_stereo_batch = has_stereo_pair(mains) or any(
        "stereo" in str(c).lower() for c in mains
    )
    bindings: dict = {}
    for node in graph.get("nodes", []):
        data = node.get("data") or {}
        node_type = data.get("nodeType", "")
        if node_type not in ("mono_camera", "rgbd_camera", "fisheye_camera", "rgb_camera",
                             "stereo_camera", "stereo_rgbd_camera"):
            continue
        if node_type in ("stereo_camera", "stereo_rgbd_camera"):
            if not is_stereo_batch:
                return None
            bindings[node["id"]] = {"source_keys": ",".join(sorted(mains))}
        else:
            if is_stereo_batch:
                return None
            bindings[node["id"]] = {"source_key": mains[0]}
    return bindings or None


def apply_bindings(graph: dict | None, bindings: dict | None) -> dict:
    """深拷贝 graph,把项目绑定写进输入节点 ``data.config``。

    返回新 graph(原对象不动);无绑定 → 仍深拷贝(调用方入队快照需要隔离)。
    绑定值空字符串/None 视为清除(保留工作流默认值)。
    """
    g = copy.deepcopy(graph or {})
    nodes = g.get("nodes") or []
    for node in nodes:
        node_id = node.get("id")
        binding = (bindings or {}).get(node_id)
        if not isinstance(binding, dict):
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        config = dict(data.get("config") or {})
        # 只有输入源卡片(source_key/source_keys/legacy position)才有绑定语义.
        # Older RGB/fisheye graphs stored only ``position``; accepting it here
        # lets the new type-first auto-match upgrade their run snapshot too.
        if ("source_keys" not in config and "source_key" not in config
                and "position" not in config):
            continue
        bound = str(binding.get("source_keys") or binding.get("source_key") or "").strip()
        if not bound:
            continue
        # 与 effective_input_keys 一致:覆盖 source_keys(多源)与
        # source_key(首源);source_key 变更同步旧 position 字段
        _apply_binding_value(config, binding)
        config["position"] = config["source_key"]
        data["config"] = config
    return g
