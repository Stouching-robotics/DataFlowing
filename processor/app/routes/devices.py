"""Device heartbeat & polling API — 本地 JSON 存储(无数据库)。

设备存 data/state/devices.json;采集端轮询 /device/tasks 获取
active 项目列表(任务概念已移除,项目即采集任务)。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, Query

from app.localstore import (
    STATE_ROOT,
    cached_sessions_for_tasks,
    list_projects,
    scan_sessions,
)
from app.security import verify_api_key
from app.device_naming import decorate_device_sources, display_names_for_sources

router = APIRouter(prefix="/api/v1", tags=["devices"])

OFFLINE_THRESHOLD_SECONDS = 120
DEVICES_FILE = STATE_ROOT / "devices.json"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_devices() -> list[dict]:
    from app.localstore import _read_json
    return _read_json(DEVICES_FILE, [])


def _save_devices(devices: list[dict]) -> None:
    from app.localstore import _write_json
    _write_json(DEVICES_FILE, devices)


def _device_status(dev: dict) -> str:
    last = dev.get("last_seen_at")
    if not last:
        return "unknown"
    try:
        elapsed = (utcnow() - datetime.fromisoformat(last)).total_seconds()
    except Exception:
        return "unknown"
    return "online" if elapsed <= OFFLINE_THRESHOLD_SECONDS else "offline"


# ── Heartbeat ─────────────────────────────────────────

@router.post("/devices/heartbeat")
async def device_heartbeat(body: dict, _: str = Depends(verify_api_key)):
    """采集端心跳。可携带采集能力(capabilities)驱动前端输入卡片:

    {
      "device_name": "DAQ-01",
      "capabilities": {"cameras": ["stereo_left", "stereo_right", "head_fisheye_rgb"],
                       "sensors": ["left_glove", "right_glove"]}
    }

    capabilities 可选:带则更新(采集端当前实际连接的硬件),不带则保持旧值。
    心跳间隔必须 < 120s,否则设备判定 offline、输入卡片不再显示。
    """
    devices = _load_devices()
    incoming_id = str(body.get("device_id") or "").strip()
    incoming_name = str(body.get("device_name") or "").strip()
    if not incoming_id and not incoming_name:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="device_id or device_name required")
    # device_id 是稳定身份；旧采集端只有 name 时继续兼容按 name 识别。
    dev = next((d for d in devices if incoming_id and d.get("device_id") == incoming_id), None)
    if dev is None:
        dev = next((d for d in devices if not incoming_id and d.get("name") == incoming_name), None)
    now = utcnow()
    if not dev:
        dev = {"id": str(uuid4()), "device_id": incoming_id or str(uuid4()),
               "name": incoming_name or incoming_id,
               "meta": body.get("meta"), "first_seen_at": now.isoformat()}
        devices.append(dev)
    elif incoming_name:
        # 改名不产生第二台逻辑设备，历史能力与项目绑定继续归属于同一 id。
        dev["name"] = incoming_name
    dev.setdefault("device_id", dev.get("id") or str(uuid4()))
    dev["last_seen_at"] = now.isoformat()
    if body.get("meta"):
        dev["meta"] = body.get("meta")
    # "capabilities" in body 判断:允许空 dict({} 表示拔光硬件,清空能力)
    if "capabilities" in body and body.get("capabilities") is not None:
        caps = body["capabilities"]
        if isinstance(caps, dict):
            dev["capabilities"] = {
                "cameras": _norm_cap_list(caps.get("cameras")),
                "sensors": _norm_cap_list(caps.get("sensors")),
            }
        else:
            dev["capabilities"] = {"cameras": _norm_cap_list(caps), "sensors": []}
    dev["status"] = "online"
    _save_devices(devices)
    return {"ok": True, "device_id": dev["device_id"],
            "device_name": dev["name"], "status": "online"}


# ── Online device capabilities → Studio 输入卡片 ──────

def _norm_cap_list(value) -> list[str]:
    """归一化能力列表:list[str] 直接返回;逗号分隔字符串按逗号拆分
    (采集端可能上报字符串而非列表,避免逐字符迭代)。"""
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _online_device_capabilities() -> dict:
    """在线设备上报的采集能力 union(cameras + sensors)。"""
    cameras: set[str] = set()
    sensors: set[str] = set()
    devices: list[dict] = []
    for d in _load_devices():
        if _device_status(d) != "online":
            continue
        caps = d.get("capabilities")
        if isinstance(caps, dict):
            device_cameras = _norm_cap_list(caps.get("cameras"))
            device_sensors = _norm_cap_list(caps.get("sensors"))
            cameras.update(device_cameras)
            sensors.update(device_sensors)
        elif isinstance(caps, list):  # 兼容扁平列表
            device_cameras = _norm_cap_list(caps)
            device_sensors = []
            cameras.update(device_cameras)
        else:
            device_cameras, device_sensors = [], []
        devices.append({
            "device_id": d.get("device_id") or d.get("id"),
            "name": d.get("name"),
            "kind": d.get("kind") or (d.get("meta") or {}).get("kind"),
            "cameras": device_cameras,
            "sensors": device_sensors,
            "last_seen_at": d.get("last_seen_at"),
        })
    return {"cameras": sorted(cameras), "sensors": sorted(sensors), "devices": devices}


@router.get("/devices/input-sources")
def device_input_sources():
    """在线采集设备的可用输入源 → 驱动 Studio 输入面板(无 ?project= 时)。

    只聚合在线设备(120s 内心跳过)上报的 capabilities;模块匹配与
    项目 input-sources 共用 match_input_modules。
    """
    from app.processing.batch import match_input_modules
    from app.api.projects import _device_input_sources, _online_input_sources
    caps = _online_device_capabilities()
    episodes = scan_sessions()
    historical_sources = _device_input_sources(episodes)
    online_sources = _online_input_sources(caps.get("devices", []))
    sources_by_id = {str(source.get("id")): source for source in historical_sources}
    for source in online_sources:
        sources_by_id[str(source.get("id"))] = source
    device_sources = list(sources_by_id.values())
    decorate_device_sources(device_sources)
    camera_names = {
        str(camera).lower()
        for episode in episodes
        for camera in (episode.get("camera_names") or [])
        if not str(camera).lower().endswith("_aux")
    }
    sensors = {
        str(sensor).lower()
        for episode in episodes
        for sensor in (episode.get("sensors") or [])
    }
    available = camera_names | sensors
    available |= {c.lower() for c in caps["cameras"] if not c.lower().endswith("_aux")}
    available |= {s.lower() for s in caps["sensors"]}
    device_names = {
        str(stream): str(name)
        for episode in episodes
        for stream, name in (episode.get("device_names") or {}).items()
        if stream and name
    }
    device_display_names = display_names_for_sources(device_sources)
    devices_by_id = {
        str(device.get("key") or device.get("name")): device
        for episode in episodes
        for device in (episode.get("devices") or [])
        if isinstance(device, dict) and (device.get("key") or device.get("name"))
    }
    for device in caps.get("devices", []):
        devices_by_id[str(device.get("device_id") or device.get("name"))] = device
    return {
        "has_episodes": bool(episodes),
        "has_online_devices": bool(caps["cameras"] or caps["sensors"]),
        "cameras": caps["cameras"],
        "sensors": caps["sensors"],
        "device_names": device_names,
        "device_display_names": device_display_names,
        "devices": list(devices_by_id.values()),
        "device_sources": device_sources,
        "modules": match_input_modules(available),
    }


# ── Device list (Web UI) ──────────────────────────────

@router.get("/devices")
async def list_devices():
    devices = _load_devices()
    devices.sort(key=lambda d: d.get("last_seen_at") or "", reverse=True)
    return {"devices": [
        {
            "id": d.get("id"),
            "device_id": d.get("device_id") or d.get("id"),
            "name": d.get("name"),
            "status": _device_status(d),
            "meta": d.get("meta"),
            "first_seen_at": d.get("first_seen_at"),
            "last_seen_at": d.get("last_seen_at"),
            "capabilities": d.get("capabilities") or {"cameras": [], "sensors": []},
        }
        for d in devices
    ]}


# ── Device task polling(项目即采集任务)───────────────

@router.get("/device/tasks")
async def device_tasks(
    device_name: str = Query(..., min_length=1),
    _: str = Depends(verify_api_key),
):
    """采集端轮询此接口获取 active 项目(DAQ client 格式)。"""
    project_rows = await asyncio.to_thread(list_projects)
    projects = [p for p in project_rows if p.get("status", "active") == "active"]
    # Prefer the last complete snapshot.  A cache miss only happens during a
    # cold start, and even that first remote scan is kept off the event loop.
    episodes = cached_sessions_for_tasks()
    if episodes is None:
        episodes = await asyncio.to_thread(scan_sessions)

    tasks = []
    for p in projects:
        cur = sum(1 for e in episodes if e.get("project") == p.get("name"))
        params = p.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        tasks.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "description": p.get("description"),
            "status": p.get("status", "active"),
            "total_required": int(params.get("target_episodes") or 0),
            "current_count": cur,
            "assigned_at": p.get("created_at"),
            "params": params,
        })

    return {"tasks": tasks, "updated_at": utcnow().isoformat()}
