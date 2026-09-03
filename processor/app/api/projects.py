"""Project APIs — 纯本地文件驱动(无数据库)。

项目定义存 data/state/projects.json,episode 从 data/sessions/ 扫描。
项目 → Episodes 两层(任务概念已移除),episode 按序号(Test005_000028 → 28)排序。
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from uuid import uuid4

from app.localstore import (
    scan_sessions, list_projects, save_projects, upsert_project, delete_project,
    list_workflows, get_workflow, upsert_workflow, read_episode_state,
    list_runs, list_exceptions, list_upload_history, list_upload_events,
)
from app.glove_sources import group_glove_source_keys, is_paired_glove_source
from app.device_naming import (
    decorate_device_sources,
    display_names_for_sources,
    is_depth_only_key,
)

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


def _upload_stats(project: dict, episodes: list[dict] | None = None) -> dict:
    """Build upload counters with unique batches separated from events.

    ``uploaded_total`` is a legacy mutable counter and used to increase on
    every replacement in older versions. It must never be used as the source
    of truth. The project count is the union of unique historical batch ids
    and currently present episode directories; upload events remain a
    separate audit metric.
    """
    name = str(project.get("name") or "")
    history = [h for h in list_upload_history()
               if h.get("project_name") == name]
    events = [e for e in list_upload_events()
               if e.get("project_name") == name]
    if episodes is None:
        episodes = scan_sessions()
    current_batch_ids = {
        str(e.get("id") or "")
        for e in episodes
        if e.get("project") == name and e.get("id")
    }
    historical_batch_ids = {
        str(h.get("episode_id") or "")
        for h in history
        if h.get("episode_id")
    }
    unique_batch_ids = historical_batch_ids | current_batch_ids
    unique_batches = len(unique_batch_ids)
    event_total = len(events)
    event_batch_ids = {
        str(e.get("episode_id") or "")
        for e in events
        if e.get("episode_id")
    }
    reuploads = sum(1 for e in events if e.get("classification") == "reupload")
    last_upload = max(
        [str(x.get("uploaded_at") or "") for x in history + events
         if x.get("uploaded_at")]
        or [""],
    )
    return {
        "uploaded_total": unique_batches,
        "batch_count": unique_batches,
        "new_batch_count": unique_batches,
        "reupload_count": reuploads,
        "upload_event_count": event_total,
        "unclassified_upload_count": len(unique_batch_ids - event_batch_ids),
        "last_upload_at": last_upload or None,
    }


def _set_workflow_project(workflow_id: str | None, project_id: str) -> None:
    """Persist the one-to-one project context on the workflow record."""
    if not workflow_id:
        return
    workflow = get_workflow(str(workflow_id))
    if workflow is None:
        return
    if workflow.get("project_id") == project_id:
        return
    workflow["project_id"] = project_id
    upsert_workflow(workflow)


def _clear_workflow_project(workflow_id: str | None, project_id: str) -> None:
    """Clear a previous relation only when it still points to this project."""
    if not workflow_id:
        return
    workflow = get_workflow(str(workflow_id))
    if workflow is None or workflow.get("project_id") != project_id:
        return
    workflow["project_id"] = None
    upsert_workflow(workflow)


def _device_input_sources(episodes: list[dict]) -> list[dict]:
    """Build selectable physical-device inputs from uploaded batch metadata.

    A physical device is the UI item (``devices[].name``); its video slots
    remain internal source keys used by processing.  Multiple devices are
    returned as separate entries, and a device with a left/right pair becomes
    one stereo input entry.
    """
    by_id: dict[str, dict] = {}
    known_camera_keys = {
        str(camera)
        for episode in episodes
        for camera in (episode.get("camera_names") or [])
        if not str(camera).lower().endswith("_aux")
    }
    for episode in episodes:
        names = episode.get("device_names") or {}
        for raw in episode.get("devices") or []:
            if not isinstance(raw, dict):
                continue
            device_name = str(raw.get("name") or "").strip()
            if not device_name:
                continue
            device_id = str(raw.get("key") or device_name).strip()
            item = by_id.setdefault(device_id, {
                "id": device_id,
                "name": device_name,
                "kind": str(raw.get("kind") or ""),
                "source_keys": [],
                "slots": [],
            })
            declared_slots = [
                *(raw.get("slots") or []),
                *(raw.get("depth_keys") or raw.get("depths") or []),
            ]
            for slot in declared_slots:
                slot = str(slot).strip()
                if slot and slot not in item["slots"]:
                    item["slots"].append(slot)
                if slot and slot in known_camera_keys and slot not in item["source_keys"]:
                    item["source_keys"].append(slot)
            # A collector may provide the mapping but omit slots in one batch.
            for stream, mapped_name in names.items():
                if str(mapped_name).strip() == device_name and str(stream) in known_camera_keys:
                    if str(stream) not in item["source_keys"]:
                        item["source_keys"].append(str(stream))

    # Older batches may have cameras but no devices[] declaration. Keep them
    # selectable as anonymous physical inputs for backward compatibility.
    declared_keys = {key for item in by_id.values() for key in item["source_keys"]}
    fallback_keys = known_camera_keys - declared_keys
    fallback_used: set[str] = set()
    for key in sorted(fallback_keys):
        if key in fallback_used:
            continue
        pair = None
        if key.endswith("_left"):
            candidate = f"{key[:-5]}_right"
            if candidate in fallback_keys:
                pair = [key, candidate]
        elif key.endswith("_right"):
            candidate = f"{key[:-6]}_left"
            if candidate in fallback_keys:
                pair = [candidate, key]
        keys = pair or [key]
        name = next((str(e.get("device_names", {}).get(key) or "").strip()
                     for e in episodes if key in (e.get("device_names") or {})), "") or key
        device_name = key.rsplit("_", 1)[0] if pair and key.endswith("_left") else name
        by_id[f"camera:{keys[0]}"] = {
            "id": f"camera:{keys[0]}", "name": device_name, "kind": "stereo" if pair else "camera",
            "source_keys": keys, "slots": keys,
        }
        fallback_used.update(keys)

    result: list[dict] = []
    for item in by_id.values():
        slots = sorted(set(item.pop("slots", [])))
        depth_keys = sorted({key for key in slots if is_depth_only_key(key)})
        # ``source_keys`` is the video binding contract. Keep pure depth
        # streams as metadata so RGB-D/stereo classification can see them,
        # but do not offer a depth frame as a normal video input.
        keys = sorted({key for key in item.pop("source_keys", [])
                       if key not in depth_keys})
        if not keys:
            continue
        low = [key.lower() for key in keys]
        kind = f"{item.get('kind') or ''} {item.get('name') or ''}".lower()
        left_right = (any(key.endswith("_left") or "_left_" in key for key in low)
                      and any(key.endswith("_right") or "_right_" in key for key in low))
        is_stereo = len(keys) >= 2 and (left_right or "stereo" in kind)
        has_depth = bool(depth_keys) or "depth" in kind or "rgbd" in kind
        item["source_keys"] = keys
        if depth_keys:
            item["depth_keys"] = depth_keys
        item["slots"] = slots
        item["input_type"] = (
            "stereo_rgbd_camera"
            if is_stereo and has_depth
            else "stereo_camera"
            if is_stereo
            else "rgbd_camera"
            if has_depth
            else "mono_camera"
        )
        item["source_key"] = keys[0]
        item["label"] = f"{item['name']} · {'Stereo' if is_stereo else 'Mono'}"
        result.append(item)

    # The recorder stores left/right as two channels, but they are one
    # physical SenseGlove input in the workflow. Keep both source keys on the
    # single card so processing/export can still access both streams.
    sensors = sorted({str(s) for e in episodes for s in (e.get("sensors") or [])})
    for keys in group_glove_source_keys(sensors):
        paired = is_paired_glove_source(keys)
        source_id = "glove:hands" if paired else f"sensor:{keys[0]}"
        name = "SenseGlove" if paired else keys[0]
        result.append({
            "id": source_id, "name": name, "kind": "sensor",
            "input_type": "glove_sensor", "source_key": keys[0],
            "source_keys": keys,
            "label": f"{name} · {'Both Hands' if paired else 'Sensor'}",
        })
    decorate_device_sources(result)
    return sorted(result, key=lambda item: (item.get("input_type") or "", item.get("display_name") or ""))


def _online_input_sources(devices: list[dict]) -> list[dict]:
    """Convert heartbeat capabilities into the same selectable-source shape."""
    result: list[dict] = []
    glove_groups: list[tuple[str, list[str]]] = []
    for device in devices or []:
        name = str(device.get("name") or device.get("device_id") or "").strip()
        if not name:
            continue
        keys = [str(key).strip() for key in (device.get("cameras") or []) if str(key).strip()]
        if keys:
            low = [key.lower() for key in keys]
            kind = f"{device.get('kind') or ''} {name}".lower()
            left_right = (any(key.endswith("_left") or "_left_" in key for key in low)
                          and any(key.endswith("_right") or "_right_" in key for key in low))
            depth_keys = [str(key).strip() for key in (
                device.get("depth_keys") or device.get("depths") or [])
                          if str(key).strip()]
            stereo = len(keys) >= 2 and (left_right or "stereo" in kind)
            has_depth = bool(depth_keys) or "depth" in kind or "rgbd" in kind
            result.append({
                "id": str(device.get("device_id") or name), "name": name,
                "kind": device.get("kind") or "camera",
                "input_type": (
                    "stereo_rgbd_camera" if stereo and has_depth
                    else "stereo_camera" if stereo
                    else "rgbd_camera" if has_depth
                    else "mono_camera"
                ),
                "source_key": keys[0], "source_keys": keys,
                "depth_keys": depth_keys,
                "label": f"{name} · {'Stereo' if stereo else 'Mono'}",
            })
        for keys in group_glove_source_keys(device.get("sensors") or []):
            glove_groups.append((name, keys))
    for name, keys in glove_groups:
        paired = is_paired_glove_source(keys)
        result.append({
            # Match the historical-source id so the project endpoint does not
            # append a second card when the same SenseGlove is online.
            "id": "glove:hands" if paired else f"{name}:sensor:{keys[0]}",
            "name": "SenseGlove" if paired else keys[0],
            "kind": "sensor", "input_type": "glove_sensor",
            "source_key": keys[0], "source_keys": keys,
            "label": f"{'SenseGlove' if paired else keys[0]} · {'Both Hands' if paired else 'Sensor'}",
        })
    decorate_device_sources(result)
    return result


def _wf_ids(project: dict) -> list[str]:
    """项目唯一绑定的工作流(兼容旧 workflow_ids 数组文件)。"""
    value = project.get("workflow_id")
    if not value:
        ids = project.get("workflow_ids") or []
        if not isinstance(ids, list):
            ids = [ids] if ids else []
        value = ids[0] if ids else None
    value = str(value or "")
    return [value] if value else []


def _wf_names(project: dict) -> list[str]:
    return [w.get("name") for w in (get_workflow(i) for i in _wf_ids(project)) if w]


def _wf_valid_ids(project: dict) -> list[str]:
    """仅返回仍存在的工作流 ID(悬空引用过滤),与 _wf_names 同序对齐。

    曾出现:项目绑定了一个已被删除的工作流,响应里 workflow_ids 含死引用
    而 workflow_names 已过滤 → 前端按 ids 遍历、按 names 索引,名字错位
    并回退显示假名 "Workflow"。所有响应点必须用本函数替代 _wf_ids。
    """
    return [i for i in _wf_ids(project) if get_workflow(i) is not None]


@router.get("/{project_id}/input-sources")
def project_input_sources(project_id: str):
    """项目实际检测到的采集输入源 → 驱动前端输入卡片。

    只返回该项目已上传批次中实际出现过的输入源。在线设备能力属于
    全局设备目录，不能并入项目响应，否则空项目会显示其他项目的设备。
    相机名剔除 _aux 辅助流;模块 source_key(s) 与 available 双向子串
    匹配(覆盖 left_glove_joint vs left_glove)。
    """
    from fastapi import HTTPException
    from app.processing.batch import match_input_modules

    project = next((p for p in list_projects() if p["id"] == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    episodes = [e for e in scan_sessions() if e.get("project") == project.get("name")]
    camera_names = sorted({c for e in episodes for c in (e.get("camera_names") or [])})
    device_sources = _device_input_sources(episodes)
    # Keep physical device names separate from stream/source keys.  For
    # example, D435_depth is the device name while D435_depth_rgb is the
    # actual video source used by processing.
    device_names: dict[str, str] = {}
    devices: dict[str, dict] = {}
    for episode in episodes:
        for stream, device_name in (episode.get("device_names") or {}).items():
            if stream and device_name:
                device_names[str(stream)] = str(device_name)
        for device in episode.get("devices") or []:
            if not isinstance(device, dict):
                continue
            key = str(device.get("key") or device.get("name") or "")
            if key:
                devices[key] = device
    # 项目输入源应反映“项目实际收到过什么”，不能反过来依赖工作流
    # 是否已经声明 glove_sensor，否则还没搭工作流时历史手套设备会消失。
    # 仍只保留真实压力数据，避免把手部骨骼列误识别成手套。
    declared = sorted({s for e in episodes for s in (e.get("sensors") or [])})
    sensors = [s for s in declared if _sensor_has_data(episodes, s)]
    # 主目集:只有 aux 没有主目时不算有该输入(与上传自动匹配一致)
    have_primary = {c.lower() for c in camera_names if not c.lower().endswith("_aux")}
    available = have_primary | {s.lower() for s in sensors}

    # Keep the raw ``device_names`` map for compatibility and expose a
    # separate standardized map for the Studio display.
    decorate_device_sources(device_sources)
    device_display_names = display_names_for_sources(device_sources)

    # This endpoint is project-scoped and read-only. Do not merge the global
    # heartbeat/device directory here; a project with no episodes must have no
    # concrete device options.
    observed = project.get("observed_devices") or {}
    if not isinstance(observed, dict):
        observed = {}

    return {
        "project_id": project_id,
        "project_name": project.get("name"),
        "has_episodes": bool(episodes),
        "has_online_devices": False,
        "camera_names": camera_names,
        "device_names": device_names,
        "device_display_names": device_display_names,
        "devices": list(devices.values()),
        "device_sources": device_sources,
        "sensors": sensors,
        "device_inputs": {"cameras": camera_names, "sensors": sensors},
        "observed_devices": observed,
        "modules": match_input_modules(available),
    }


@router.put("/{project_id}/workflow-inputs")
def put_project_workflow_inputs(project_id: str, body: dict):
    """持久化"项目 × 工作流"的可用输入源。

    Studio 以 ?project= 打开工作流并获取输入源时调用:首次构建记录
    当时的相机/传感器;后续再有新的传感器数据(上传新批次/采集端新
    设备)时**取并集并入**,形成该项目该工作流的历史输入源记录。
    存 projects.json 的 workflow_inputs[workflow_id] 字段。
    """
    from fastapi import HTTPException
    from app.localstore import utcnow_iso

    project = next((p for p in list_projects() if p["id"] == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    wf_id = str(body.get("workflow_id") or "")
    if not wf_id:
        raise HTTPException(status_code=400, detail="workflow_id required")

    stored = dict(project.get("workflow_inputs") or {})
    prev = stored.get(wf_id) or {}

    def _merge(key: str) -> list[str]:
        vals = set(str(v) for v in (prev.get(key) or []) if v)
        for v in body.get(key) or []:
            if v:
                vals.add(str(v))
        return sorted(vals)

    # 相机兼容两种字段名(批次侧 camera_names / 设备侧 cameras),取并集去重
    cam_set = {str(v) for v in (prev.get("cameras") or []) if v}
    for v in list(body.get("camera_names") or []) + list(body.get("cameras") or []):
        if v:
            cam_set.add(str(v))
    entry = {
        "cameras": sorted(cam_set),
        "sensors": _merge("sensors"),
        "device_inputs": {**(prev.get("device_inputs") or {}),
                          **(body.get("device_inputs") or {})},
        "updated_at": utcnow_iso(),
    }
    stored[wf_id] = entry
    project["workflow_inputs"] = stored
    upsert_project(project)
    return {"project_id": project_id, "workflow_id": wf_id,
            "workflow_inputs": stored}


def _sensor_has_data(episodes: list[dict], sensor: str) -> bool:
    """项目任一批次中该传感器列有真实压力数据(全文件均匀采样)。

    与 frames-data 同规则:全零列(手部骨骼识别参数占位)不算真实
    手套传感器;有非零压力的真实传感器即使首非零帧在后半段也检出。
    """
    from pathlib import Path
    import pyarrow.parquet as pq
    from app.routes.ingestion import _col_has_pressure
    col = f"observation.{sensor}"
    for e in episodes:
        batch = Path(e.get("path") or "")
        if not batch.is_dir():
            continue
        # 优先 data/<sensor>/ 目录(采集端规范布局)
        for parq in (batch / "data" / sensor).rglob("*.parquet"):
            if _col_has_pressure(parq, col):
                return True
        # 兜底:扫描所有 parquet 找该列
        for parq in batch.rglob("*.parquet"):
            try:
                if col in pq.ParquetFile(parq).schema_arrow.names and _col_has_pressure(parq, col):
                    return True
            except Exception:
                continue
    return False


def _episode_index(name: str) -> int | None:
    import re
    if re.search(r"_(\d{8})_\d{6}$", str(name)):
        return None
    m = re.search(r"_(\d{6})$", str(name))
    return int(m.group(1)) if m else None


@router.get("/summary")
def project_summary(
    status: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """按项目聚合 episode 状态分布(原 /api/v1/tasks,字段 task_name 兼容)。"""
    episodes = scan_sessions()
    by_project: dict[str, dict] = {}
    for e in episodes:
        pname = e.get("project") or "Uncategorized"
        st = e.get("status") or "unknown"
        if status and st != status:
            continue
        p = by_project.setdefault(pname, {
            "task_name": pname, "project_name": pname,
            "total_episodes": 0, "status_counts": {},
            "avg_frames": 0, "last_upload_at": None, "cleaning_failed": 0,
        })
        p["total_episodes"] += 1
        p["status_counts"][st] = p["status_counts"].get(st, 0) + 1
    projects = list(by_project.values())
    if search:
        low = search.lower()
        projects = [p for p in projects if low in p["task_name"].lower()]
    projects.sort(key=lambda p: p["total_episodes"], reverse=True)
    total = len(projects)
    return {"tasks": projects[offset:offset + limit], "total": total,
            "limit": limit, "offset": offset}


@router.get("/hierarchy")
def get_hierarchy(
    status: str | None = Query(None),
    search: str | None = Query(None),
):
    """Project → Episodes(无数据库,直接扫描 sessions 目录)。"""
    episodes = scan_sessions()
    deleted_ids = set()
    from app.localstore import list_deleted_episodes
    deleted_ids = {d["id"] for d in list_deleted_episodes()}
    episodes = [e for e in episodes if e["id"] not in deleted_ids]
    if status:
        if status in ("completed", "to_review"):
            episodes = [e for e in episodes if e.get("status") in ("completed", "to_review")]
        elif status == "reviewed":
            episodes = [e for e in episodes if e.get("status") in ("reviewed", "approved")]
        else:
            episodes = [e for e in episodes if e.get("status") == status]

    # 项目文件夹**全量显示**(含空项目/刚创建的项目):右侧层级列表先铺
    # 所有项目定义,有批次的再挂批次;Uncategorized 兜底(有批次无定义)。
    # 注意:批次目录名是 sanitize 后的(空格→_、去不安全字符),与项目原始
    # 名可能不同 —— 归一化匹配,避免同一项目显示成两个文件夹。
    try:
        from app.storage import sanitize_task_name
    except Exception:
        sanitize_task_name = None

    def _norm(name: str) -> str:
        return sanitize_task_name(name) if sanitize_task_name else name

    project_defs: dict[str, dict] = {}
    for p in list_projects():
        if p.get("name"):
            project_defs.setdefault(_norm(p["name"]), p)
    projects_map: dict[str, dict] = {}
    order: list[str] = []
    for p in project_defs.values():
        projects_map[p["name"]] = {"name": p["name"], "episodes": []}
        order.append(p["name"])
    for e in episodes:
        pname = e.get("project") or "Uncategorized"
        if pname not in projects_map:
            # 批次目录名可能已 sanitize(空格→_ 等)→ 归一化后归属项目定义
            # (所有项目定义已先铺进 projects_map,found 一定存在)
            found = project_defs.get(_norm(pname))
            if found:
                pname = found["name"]
        if pname not in projects_map:
            projects_map[pname] = {"name": pname, "episodes": []}
            order.append(pname)
        projects_map[pname]["episodes"].append(e)

    # 批次异常按 episode 建一次索引(避免每批次重复读文件)
    try:
        excs_by_ep: dict[str, list] = {}
        for x in list_exceptions():
            excs_by_ep.setdefault(x.get("episode_id") or "", []).append(x)
    except Exception:
        excs_by_ep = {}

    tree = []
    for pname in order:
        p_eps = projects_map[pname]["episodes"]
        p_eps.sort(key=lambda e: (
            e.get("episode_index") is None,
            e.get("episode_index") if e.get("episode_index") is not None else 10**9,
            e["name"],
        ))
        project_def = project_defs.get(_norm(pname))
        workflow_names = _wf_names(project_def) if project_def else []
        tree.append({
            "project": {
                "id": project_def["id"] if project_def else pname,
                "name": pname,
                "workflow_name": workflow_names[0] if workflow_names else None,
                "workflow_names": workflow_names,
                "status": (project_def or {}).get("status", "active"),
            },
            "episodes": [_ep_dict(e, excs_by_ep) for e in p_eps],
            "tasks": [],
        })

    if search:
        s = search.lower()
        tree = [n for n in tree
                if s in n["project"]["name"].lower()
                or any(s in e["name"].lower() or s in e["id"].lower() for e in n["episodes"])]

    return {"projects": tree, "total": sum(len(n["episodes"]) for n in tree)}


def _ep_dict(e: dict, excs_by_ep: dict | None = None) -> dict:
    state = read_episode_state(e["id"])
    # 该批次的异常(上传不匹配/运行失败)—— Review 页点击文件后详情显示
    excs = (excs_by_ep or {}).get(e["id"]) or []
    return {
        "id": e["id"],
        "name": e["name"],
        "status": e.get("status"),
        "episode_index": e.get("episode_index"),
        "camera_names": e.get("camera_names") or [],
        "camera_streams": e.get("camera_streams") or {},
        "device_names": e.get("device_names") or {},
        "devices": e.get("devices") or [],
        "camera_group": {"type": "single", "count": len(e.get("camera_names") or [])},
        "camera_groups": {},
        "has_skeleton": bool(e.get("has_skeleton")),
        "frame_count": e.get("frame_count") or 0,
        "fps": e.get("fps") or 30,
        "timestamp": e.get("timestamp", ""),
        "created_at": e.get("created_at"),
        "cleaning_report": state.get("cleaning_report"),
        "exceptions": [{
            "kind": x.get("kind"), "workflow_name": x.get("workflow_name"),
            "message": x.get("message"), "wanted": x.get("wanted") or [],
            "matched": x.get("matched") or [], "missing": x.get("missing") or [],
            "available": x.get("available") or [],
        } for x in excs],
    }


# ── 项目 CRUD(JSON)──────────────────────────────────

@router.get("")
def list_project_api(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
):
    projects = list_projects()
    # 这里只读项目列表，不能因为工作流缓存未加载、远程存储暂时不可用
    # 或服务启动顺序异常，就把已有的项目绑定写成空值。
    # 工作流绑定的清理必须放在明确的工作流删除/解绑流程中执行。
    if status:
        projects = [p for p in projects if p.get("status") == status]
    total = len(projects)
    rows = projects[offset:offset + limit]

    # ── 异常计数(上传不匹配落盘 + 该项目批次的 failed run)──
    # 只扫一次,避免每项目重复扫描(O(n²) 收敛)
    try:
        episodes = scan_sessions()
    except Exception:
        episodes = []
    ep_project = {e["id"]: e.get("project") for e in episodes}
    proj_id_by_name = {p.get("name"): p.get("id") for p in projects}
    mismatch_counts: dict[str, int] = {}
    try:
        for e in list_exceptions():
            if e.get("kind") in {"upload_mismatch", "input_missing"} and e.get("project_id"):
                mismatch_counts[e["project_id"]] = mismatch_counts.get(e["project_id"], 0) + 1
    except Exception:
        pass
    failed_counts: dict[str, int] = {}
    try:
        for r in list_runs():
            if not isinstance(r, dict) or r.get("status") != "failed":
                continue
            pid = proj_id_by_name.get(ep_project.get(r.get("episode_id")))
            if pid:
                failed_counts[pid] = failed_counts.get(pid, 0) + 1
    except Exception:
        pass
    out = []
    for p in rows:
        wf_ids = _wf_valid_ids(p)
        wf_names = _wf_names(p)   # 已删除的工作流引用 → 过滤后可能为空
        episode_count = sum(1 for e in episodes if e.get("project") == p.get("name"))
        upload_stats = _upload_stats(p, episodes)
        params = p.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        out.append({
            "id": p["id"], "name": p["name"], "description": p.get("description"),
            "workflow_ids": wf_ids,
            "workflow_id": wf_ids[0] if wf_ids else None,   # 兼容旧字段
            "workflow_name": wf_names[0] if wf_names else None,  # 兼容旧字段
            "workflow_names": wf_names,
            "device_type": p.get("device_type"),
            "params": params,
            "target_episodes": int(params.get("target_episodes") or 0),
            "status": p.get("status", "active"),
            "episode_count": episode_count,
            **upload_stats,
            "exception_count": mismatch_counts.get(p["id"], 0) + failed_counts.get(p["id"], 0),
            "observed_inputs": p.get("observed_inputs") or [],
            "created_at": p.get("created_at"), "updated_at": p.get("updated_at"),
        })
    return {"projects": out, "total": total}


@router.post("", status_code=201)
def create_project(body: dict):
    from app.localstore import utcnow_iso
    from fastapi import HTTPException
    requested_wf = body.get("workflow_id")
    if not requested_wf:
        legacy_ids = body.get("workflow_ids") or []
        if not isinstance(legacy_ids, list):
            legacy_ids = [legacy_ids] if legacy_ids else []
        requested_wf = legacy_ids[0] if legacy_ids else None
    wf_ids = [str(requested_wf)] if requested_wf else []
    # 只保留仍存在的工作流,拒绝悬空引用(创建时提交的 id 可能已删除)
    wf_ids = [str(i) for i in wf_ids if i and get_workflow(str(i)) is not None]
    project = {
        "id": str(uuid4()),
        "name": body.get("name"),
        "description": body.get("description"),
        "workflow_ids": wf_ids,
        "workflow_id": wf_ids[0] if wf_ids else None,
        "device_type": body.get("device_type"),
        "params": body.get("params") or {},
        "status": body.get("status", "active"),
        "created_at": utcnow_iso(),
        "updated_at": utcnow_iso(),
    }
    upsert_project(project)
    _set_workflow_project(project.get("workflow_id"), project["id"])
    params = project.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    return {
        **project,
        "workflow_ids": _wf_valid_ids(project),
        "params": params,
        "target_episodes": int(params.get("target_episodes") or 0),
        "workflow_names": _wf_names(project),
        "episode_count": 0,
    }


@router.get("/{project_id}")
def get_project(project_id: str):
    project = next((p for p in list_projects() if p["id"] == project_id), None)
    if project is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Project not found")
    episode_count = sum(1 for e in scan_sessions() if e.get("project") == project.get("name"))
    params = project.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    return {**project, **_upload_stats(project),
            "workflow_ids": _wf_valid_ids(project), "params": params,
            "target_episodes": int(params.get("target_episodes") or 0),
            "workflow_names": _wf_names(project),
            "episode_count": episode_count}


@router.get("/{project_id}/uploads")
def project_uploads(project_id: str, limit: int = Query(100, ge=1, le=1000)):
    """Return append-only upload events for project audit/review."""
    project = next((p for p in list_projects() if p["id"] == project_id), None)
    if project is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Project not found")
    events = [e for e in list_upload_events()
              if e.get("project_name") == project.get("name")]
    events.sort(key=lambda e: e.get("uploaded_at") or "", reverse=True)
    return {
        "project_id": project_id,
        "project_name": project.get("name"),
        "stats": _upload_stats(project),
        "events": events[:limit],
        "total_events": len(events),
    }


@router.get("/{project_id}/bindings")
def get_project_bindings(project_id: str):
    """项目对绑定工作流的设备命名映射(workflow_bindings)。"""
    from fastapi import HTTPException
    project = next((p for p in list_projects() if p["id"] == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    bindings = project.get("workflow_bindings") or {}
    if not isinstance(bindings, dict):
        bindings = {}
    return {"project_id": project_id, "workflow_bindings": bindings}


@router.put("/{project_id}/bindings")
def put_project_bindings(project_id: str, body: dict):
    """写入/清除单个节点绑定:{workflow_id, node_id, source_key}

    source_key 为空字符串/None → 删除该节点绑定(恢复工作流默认值)。
    """
    from fastapi import HTTPException
    project = next((p for p in list_projects() if p["id"] == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    workflow_id = str(body.get("workflow_id") or "")
    node_id = str(body.get("node_id") or "")
    if not workflow_id or not node_id:
        raise HTTPException(status_code=400, detail="workflow_id and node_id are required")
    from app.localstore import get_workflow as _get_wf
    wf = _get_wf(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    bindings = project.get("workflow_bindings") or {}
    if not isinstance(bindings, dict):
        bindings = {}
    wf_bindings = bindings.get(workflow_id) or {}
    if not isinstance(wf_bindings, dict):
        wf_bindings = {}
    raw = body.get("source_key")
    if raw is None or not str(raw).strip():
        # 清除绑定
        wf_bindings.pop(node_id, None)
        if not wf_bindings:
            bindings.pop(workflow_id, None)
    else:
        wf_bindings[node_id] = {"source_key": str(raw).strip()}
        bindings[workflow_id] = wf_bindings
    project["workflow_bindings"] = bindings
    upsert_project(project)
    _set_workflow_project(workflow_id, project_id)
    backfill = {"episodes": 0, "queued": 0, "already_scheduled": 0, "skipped": 0}
    try:
        from app.workflow_dispatch import backfill_project
        backfill = backfill_project(project, [workflow_id])
    except Exception as exc:
        print(f"[Projects] Binding backfill skipped: {exc}")
    return {"project_id": project_id, "workflow_bindings": bindings,
            "backfill": backfill}


@router.get("/{project_id}/tree")
def get_project_tree(project_id: str):
    """项目 → 批次(episode)树,Projects 页面展开用(任务概念已移除)。"""
    from fastapi import HTTPException
    project = next((p for p in list_projects() if p["id"] == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    episodes = [e for e in scan_sessions() if e.get("project") == project.get("name")]
    episodes.sort(key=lambda e: (
        e.get("episode_index") is None,
        e.get("episode_index") if e.get("episode_index") is not None else 10**9,
        e["name"],
    ))
    tasks = []
    for e in episodes:
        tasks.append({
            "id": e["id"],
            "name": e["id"],
            "original_archive": e["id"],
            "created_at": e.get("created_at"),
            "episode_count": 1,
            "episodes": [{
                "id": e["id"],
                "name": e["name"],
                "camera": (e.get("camera_names") or [None])[0],
                "status": e.get("status"),
                "frame_count": e.get("frame_count") or 0,
                "fps": e.get("fps") or 30,
            }],
        })
    # 已永久删除的上传历史(目录已删,记录保留):追加为 purged 条目
    try:
        existing_ids = {e["id"] for e in episodes}
        purged = [
            h for h in list_upload_history()
            if h.get("project_name") == project.get("name")
            and h.get("episode_id") not in existing_ids
        ]
        purged.sort(key=lambda h: h.get("uploaded_at") or "")
        for h in purged:
            tasks.append({
                "id": h["episode_id"],
                "name": h["episode_id"],
                "original_archive": h["episode_id"],
                "created_at": h.get("uploaded_at"),
                "episode_count": 1,
                "purged": True,
                "episodes": [{
                    "id": h["episode_id"],
                    "name": h["episode_id"],
                    "status": "purged",
                    "frame_count": 0,
                    "fps": 0,
                }],
            })
    except Exception as exc:
        print(f"[Projects] Failed to merge upload history: {exc}")
    return {
        "project": {**project, "workflow_ids": _wf_valid_ids(project), "workflow_names": _wf_names(project)},
        "tasks": tasks,
        "task_count": len(tasks),
        "episode_count": len(episodes),
    }


@router.put("/{project_id}")
def update_project(project_id: str, body: dict):
    from fastapi import HTTPException
    project = next((p for p in list_projects() if p["id"] == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    previous_workflows = set(_wf_ids(project))
    # 文件系统即数据库:批次目录名 = sanitize(项目名)。改名必须同步
    # 重命名 sessions/<旧名>/ → sessions/<新名>/,否则旧批次变成孤儿
    # (原项目在审核页"消失"),新项目名下 0 批次像"新项目"。
    if "name" in body and body.get("name") and body["name"] != project.get("name"):
        from app.storage import sanitize_task_name
        from app.localstore import SESSIONS_ROOT
        old_project_name = str(project.get("name") or "")
        old_dir = SESSIONS_ROOT / sanitize_task_name(old_project_name)
        new_dir = SESSIONS_ROOT / sanitize_task_name(str(body["name"]))
        if old_dir.is_dir():
            if new_dir.exists():
                raise HTTPException(
                    status_code=409,
                    detail="A sessions folder with the new name already exists; rename aborted.")
            old_dir.rename(new_dir)
            # 批次状态里保存的是项目原名；目录改名后直接遍历新目录并
            # 同步更新状态。不能依赖 scan_sessions 再按旧项目名筛选：
            # scan_sessions 看到新目录后会正确返回新项目名，旧筛选会把
            # 这些批次漏掉，下一次扫描又可能被旧状态影响归属。
            from app.localstore import write_episode_state
            for batch_dir in sorted(new_dir.iterdir()):
                if not batch_dir.is_dir():
                    continue
                state = read_episode_state(batch_dir.name)
                state["project"] = body["name"]
                write_episode_state(batch_dir.name, state)
        # 即使项目当前没有 sessions 目录,上传历史/异常/旧运行快照也
        # 可能保存了项目名,因此这些索引同步不能放在 is_dir 分支里。
        from app.localstore import rename_project_references
        rename_project_references(project.get("name"), body["name"])
        # 采集端可能在项目改名后仍继续发送旧任务前缀。保留别名，
        # 让后续上传仍归入同一项目，而不是创建新序列或 Uncategorized。
        aliases = {
            str(alias).strip()
            for alias in (project.get("name_aliases") or [])
            if str(alias).strip()
        }
        if old_project_name:
            aliases.add(old_project_name)
        project["name_aliases"] = sorted(aliases)
    for key, value in body.items():
        project[key] = value
    # 保存时过滤无效工作流引用(提交的 id 可能已被删除),避免悬空
    # 一个项目只允许一个工作流。workflow_id 是新字段；旧客户端提交
    # workflow_ids 时只取第一个，避免重跑时出现多个候选工作流。
    if "workflow_id" in body or "workflow_ids" in body:
        requested_wf = body.get("workflow_id")
        if not requested_wf:
            legacy_ids = body.get("workflow_ids") or []
            if not isinstance(legacy_ids, list):
                legacy_ids = [legacy_ids] if legacy_ids else []
            requested_wf = legacy_ids[0] if legacy_ids else None
        requested_wf = str(requested_wf) if requested_wf else None
        if requested_wf and get_workflow(requested_wf) is None:
            requested_wf = None
        project["workflow_id"] = requested_wf
        project["workflow_ids"] = [requested_wf] if requested_wf else []
    params = project.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    project["params"] = params
    upsert_project(project)
    for old_workflow in previous_workflows - set(_wf_ids(project)):
        _clear_workflow_project(old_workflow, project_id)
    for current_workflow in _wf_ids(project):
        _set_workflow_project(current_workflow, project_id)
    # 工作流首次绑定或项目输入绑定改变后，自动回填历史批次。派发器有
    # 版本指纹和原子去重，重复保存/刷新页面不会重复处理。
    backfill = {"episodes": 0, "queued": 0, "already_scheduled": 0, "skipped": 0}
    current_workflows = set(_wf_ids(project))
    if current_workflows - previous_workflows or project.get("status") == "active":
        try:
            from app.workflow_dispatch import backfill_project
            backfill = backfill_project(project)
        except Exception as exc:
            print(f"[Projects] Historical workflow backfill skipped: {exc}")
    return {**project,
            "workflow_ids": _wf_valid_ids(project),
            "workflow_names": _wf_names(project),
            "target_episodes": int(params.get("target_episodes") or 0),
            "episode_count": sum(1 for e in scan_sessions() if e.get("project") == project.get("name")),
            "backfill": backfill}


@router.delete("/{project_id}", status_code=204)
def delete_project_api(project_id: str):
    delete_project(project_id)
