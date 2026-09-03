"""工作流自动派发与历史回填。

这里集中处理上传派发、项目绑定后的历史回填和幂等判断，避免上传路由、
手工重处理接口各自维护一套略有差异的匹配规则。工作流定义仍是共享的；
项目绑定只覆盖输入设备名，真正入队时才生成项目级 graph snapshot。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.localstore import (
    add_exception,
    delete_exception,
    get_workflow,
    list_runs,
    list_exceptions,
    read_episode_state,
    scan_sessions,
    save_run_if_absent,
    set_episode_status,
)
from app.workflow_bindings import (
    apply_bindings,
)
from app.device_naming import camera_profile, is_depth_only_key
from app.glove_sources import group_glove_source_keys
from app.lerobot_v21 import canonical_source_key


_CAMERA_INPUTS = {
    "mono_camera", "rgbd_camera", "fisheye_camera", "rgb_camera",
    "stereo_camera", "stereo_rgbd_camera",
}


def project_workflow_ids(project: dict | None) -> list[str]:
    """Return the one workflow bound to a project.

    ``workflow_ids`` remains readable for old project files, but the product
    model is one project → one workflow.  Prefer the canonical singular field
    when present and ignore legacy extra entries.
    """
    if not project:
        return []
    value = project.get("workflow_id")
    if not value:
        raw = project.get("workflow_ids") or []
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        value = raw[0] if raw else None
    value = str(value or "")
    return [value] if value else []


def workflow_revision(workflow: dict, bindings: dict | None = None) -> str:
    """生成稳定的工作流版本指纹。

    绑定后的 graph 也进入指纹，所以同一工作流在两个项目使用不同设备
    名称时不会互相去重。字段排序保证 JSON 文件重排不会导致重复运行。
    """
    payload = {
        "workflow_id": workflow.get("id"),
        "graph": apply_bindings(workflow.get("graph") or {}, bindings or {}),
        "node_configs": workflow.get("node_configs") or {},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _config_source_keys(config: dict) -> list[str]:
    values = config.get("source_keys") or config.get("source_key") or config.get("position")
    if isinstance(values, str):
        values = [v.strip() for v in values.split(",") if v.strip()]
    if not isinstance(values, (list, tuple, set)):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _node_source_groups(workflow: dict, bindings: dict | None) -> tuple[list[str], list[str]]:
    """Return source keys from connected input cards only.

    Input cards may be placed on a canvas before the upload is known. A card
    with no outgoing edge is design-time content and must not make dispatch
    reject an otherwise compatible upload.
    """
    graph = apply_bindings(workflow.get("graph") or {}, bindings or {})
    connected = {
        str(edge.get("source")) for edge in graph.get("edges", [])
        if edge.get("source")
    }
    camera_keys: list[str] = []
    sensor_keys: list[str] = []
    for node in graph.get("nodes", []):
        if str(node.get("id")) not in connected:
            continue
        data = node.get("data") or {}
        node_type = str(data.get("nodeType") or "")
        if node_type not in _CAMERA_INPUTS and node_type != "glove_sensor":
            continue
        values = _config_source_keys(data.get("config") or {})
        target = sensor_keys if node_type == "glove_sensor" else camera_keys
        target.extend(values)
    return camera_keys, sensor_keys


def _episode_sources(episode: dict) -> tuple[set[str], set[str]]:
    cameras = {
        str(value).lower()
        for value in (episode.get("camera_names") or [])
        if not str(value).lower().endswith("_aux")
    }
    # A workflow may be configured with the physical device name from
    # meta/info.json (for example D435_depth), while the uploaded video is
    # stored under its slot/source key (D435_depth_rgb).  Include aliases for
    # matching only; actual processing continues to use camera_names.
    for device_name in (episode.get("device_names") or {}).values():
        value = str(device_name).strip().lower()
        if value:
            cameras.add(value)
    sensors = {str(value).lower() for value in (episode.get("sensors") or [])}
    # 采集端有时没有把 sensors 写进 info.json，但会按 data/<sensor>/ 保存
    # parquet。只从目录名补充手套/触觉类，IMU/深度仍只是附属能力。
    batch_path = Path(str(episode.get("path") or ""))
    data_root = batch_path / "data"
    if data_root.is_dir():
        for child in data_root.iterdir():
            name = child.name.lower()
            if child.is_dir() and any(k in name for k in ("glove", "tactile", "sensor")):
                sensors.add(name)
    return cameras, sensors


def _episode_camera_values(episode: dict) -> list[str]:
    return [str(value) for value in (episode.get("camera_names") or [])
            if not str(value).lower().endswith("_aux")]


def _episode_input_groups(episode: dict) -> list[dict]:
    """Build physical input groups from the uploaded episode metadata.

    A group is the matching unit: one RGB stream, one Stereo pair, one RGB-D
    device, or one Stereo RGB-D device with an RGB pair plus depth metadata.
    The real source
    keys are kept for the worker snapshot, while ``input_type`` is the stable
    workflow contract.
    """
    metadata: dict = {}
    batch_path = Path(str(episode.get("path") or ""))
    for relative in ("meta/info.json", "metadata.json"):
        candidate = batch_path / relative
        if candidate.is_file():
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
                break
            except (OSError, ValueError):
                continue
    camera_values = _episode_camera_values(episode)
    if not camera_values:
        camera_values = [str(value) for value in (metadata.get("cameras") or {})
                         if not str(value).lower().endswith("_aux")]
    camera_lowers = {value.lower() for value in camera_values}
    device_names = episode.get("device_names") or metadata.get("device_names") or {}
    if not isinstance(device_names, dict):
        device_names = {}
    devices = episode.get("devices") or metadata.get("devices") or []
    groups: list[dict] = []
    used: set[str] = set()

    def add_group(name: str, kind: str, slots: list[str]) -> None:
        clean: list[str] = []
        seen: set[str] = set()
        for value in slots:
            item = str(value).strip()
            low = item.lower()
            # State camera_names usually lists only RGB video streams; the
            # physical device metadata may additionally list a pure depth
            # slot. Keep that slot for RGB-D classification and pairing.
            if (not item or low in seen
                    or (low not in camera_lowers and not is_depth_only_key(item))):
                continue
            seen.add(low)
            clean.append(item)
        if not clean:
            return
        if all(key.lower() in used for key in clean):
            return
        rgb_keys = [key for key in clean if not is_depth_only_key(key)]
        if not rgb_keys:
            return
        profile, _lens = camera_profile(kind, name, clean)
        groups.append({
            "input_type": profile,
            "source_keys": rgb_keys,
            "depth_keys": [key for key in clean if is_depth_only_key(key)],
            "aliases": [str(name).strip(), *clean],
        })
        used.update(key.lower() for key in clean)

    # New collector metadata provides the authoritative physical grouping.
    if isinstance(devices, list):
        for device in devices:
            if not isinstance(device, dict):
                continue
            name = str(device.get("name") or device.get("key") or "").strip()
            slots = [str(value) for value in (device.get("slots") or [])]
            if name:
                slots.extend(
                    str(stream) for stream, mapped in device_names.items()
                    if str(mapped).strip().lower() == name.lower()
                )
            add_group(name, str(device.get("kind") or ""), slots)

    # Compatibility path for older episode state that has only stream → name.
    by_device: dict[str, list[str]] = {}
    for stream, name in device_names.items():
        key = str(stream).strip()
        if (not key or (key.lower() not in camera_lowers
                        and not is_depth_only_key(key))):
            continue
        group_name = str(name).strip() or key
        by_device.setdefault(group_name.lower(), [group_name]).append(key)
    for values in by_device.values():
        add_group(values[0], "", values[1:])

    # Final fallback for old state without physical device metadata: pair
    # conventional left/right streams and classify the remaining stream.
    remaining = [key for key in camera_values if key.lower() not in used]
    remaining_lowers = {key.lower() for key in remaining}
    for key in sorted(remaining, key=str.lower):
        if key.lower() not in remaining_lowers:
            continue
        pair = None
        low = key.lower()
        replacements = (
            ("_left_", "_right_"), ("_right_", "_left_"),
            ("_left", "_right"), ("_right", "_left"),
        )
        for old, new in replacements:
            if old not in low:
                continue
            candidate = low.replace(old, new)
            pair = next((value for value in remaining
                          if value.lower() == candidate), None)
            if pair:
                break
        add_group("", "stereo" if pair else "", [key, pair] if pair else [key])
        remaining_lowers.discard(key.lower())
        if pair:
            remaining_lowers.discard(pair.lower())

    sensors = set(str(value) for value in (episode.get("sensors") or []))
    sensors.update(str(value) for value in (metadata.get("sensors") or []))
    # Keep the existing directory fallback for batches whose state file is
    # older than the uploaded metadata.
    sensors.update(_episode_sources(episode)[1])
    for keys in group_glove_source_keys(sorted(sensors)):
        groups.append({
            "input_type": "glove_sensor",
            "source_keys": [str(key) for key in keys],
            "depth_keys": [],
            "aliases": [str(key) for key in keys],
        })
    return groups


def _workflow_input_specs(workflow: dict, bindings: dict | None) -> list[dict]:
    """Return connected input cards with their fixed semantic categories."""
    graph = apply_bindings(workflow.get("graph") or {}, bindings or {})
    connected = {
        str(edge.get("source")) for edge in graph.get("edges", [])
        if edge.get("source")
    }
    specs: list[dict] = []
    for node in graph.get("nodes", []):
        node_id = str(node.get("id") or "")
        if not node_id or node_id not in connected:
            continue
        data = node.get("data") or {}
        node_type = str(data.get("nodeType") or "")
        if node_type not in _CAMERA_INPUTS and node_type != "glove_sensor":
            continue
        semantic = (
            "glove_sensor" if node_type == "glove_sensor"
            else "stereo_rgbd_camera" if node_type == "stereo_rgbd_camera"
            else "rgbd_camera" if node_type == "rgbd_camera"
            else "stereo_rgb" if node_type == "stereo_camera"
            else "mono_rgb"
        )
        specs.append({
            "node_id": node_id,
            "node_type": node_type,
            "semantic_type": semantic,
            "keys": _config_source_keys(data.get("config") or {}),
        })
    return specs


def _key_matches_group(key: str, group: dict) -> bool:
    wanted = str(key).strip().lower()
    if not wanted:
        return False
    canonical_wanted = canonical_source_key(key).lower()
    aliases = [str(alias).strip() for alias in group.get("aliases") or []
               if str(alias).strip()]
    if any(canonical_source_key(alias).lower() == canonical_wanted
           for alias in aliases):
        return True
    return any(
        wanted in alias.lower() or alias.lower() in wanted
        for alias in aliases
    )


def _auto_bindings_for_episode(workflow: dict, episode: dict,
                               bindings: dict | None) -> tuple[dict, set[str], list[dict], list[dict]]:
    """Match connected fixed cards to real input groups.

    A persisted source name is preferred only when it matches the uploaded
    group. A stale name from an older workflow is treated as a hint and is
    replaced by a same-category source in the run snapshot.
    """
    specs = _workflow_input_specs(workflow, bindings)
    groups = _episode_input_groups(episode)
    remaining = set(range(len(groups)))
    auto: dict = {}
    matched: set[str] = set()

    for spec in specs:
        candidates = [i for i in remaining
                      if groups[i].get("input_type") == spec["semantic_type"]]
        if not candidates:
            continue
        named = [i for i in candidates
                 if any(_key_matches_group(key, groups[i]) for key in spec["keys"])]
        selected = sorted(
            named or candidates,
            key=lambda i: str(groups[i].get("aliases") or ""),
        )[0]
        group = groups[selected]
        remaining.remove(selected)
        keys = [str(value) for value in group.get("source_keys") or []]
        if not keys:
            continue
        auto[spec["node_id"]] = (
            {"source_keys": ",".join(keys)}
            if len(keys) > 1 or spec["semantic_type"] == "glove_sensor"
            else {"source_key": keys[0]}
        )
        matched.add(spec["node_id"])

    missing = [spec for spec in specs if spec["node_id"] not in matched]
    return auto, matched, missing, groups


def _clear_unmatched_input_configs(graph: dict, matched: set[str]) -> None:
    """Remove stale source names from skipped input branches in a run copy."""
    for node in graph.get("nodes", []):
        node_id = str(node.get("id") or "")
        data = node.get("data") or {}
        node_type = str(data.get("nodeType") or "")
        if node_id in matched or (node_type not in _CAMERA_INPUTS and node_type != "glove_sensor"):
            continue
        config = data.get("config")
        if not isinstance(config, dict):
            continue
        for field in ("source_key", "source_keys", "position"):
            if field in config:
                config[field] = ""


def _record_mismatch(project: dict | None, episode: dict, workflow: dict,
                     wanted: list[str], available: set[str], message: str) -> None:
    try:
        add_exception({
            "id": str(uuid4()),
            "kind": "upload_mismatch",
            "project_id": project.get("id") if project else None,
            "project_name": project.get("name") if project else None,
            "episode_id": episode.get("id"),
            "workflow_id": workflow.get("id"),
            "workflow_name": workflow.get("name"),
            "wanted": list(wanted),
            "available": sorted(available),
            "message": message,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        print(f"[WorkflowDispatch] Failed to record mismatch: {exc}")


def _clear_episode_mismatches(episode_id: str) -> None:
    """Remove stale upload-mismatch records after a compatible run exists."""
    try:
        for exc in list_exceptions():
            if (exc.get("episode_id") == episode_id
                    and exc.get("kind") == "upload_mismatch"):
                delete_exception(exc.get("id"))
    except Exception as exc:
        print(f"[WorkflowDispatch] Failed to clear stale mismatches: {exc}")


def _record_mismatch_summary(project: dict | None, episode: dict,
                             mismatches: list[dict]) -> None:
    """Store one mismatch for the batch, not one for every candidate workflow."""
    if not mismatches:
        return
    available = mismatches[0].get("available") or []
    wanted: list[str] = []
    for item in mismatches:
        for value in item.get("wanted") or []:
            if value not in wanted:
                wanted.append(value)
    _clear_episode_mismatches(str(episode.get("id")))
    try:
        add_exception({
            "id": str(uuid4()),
            "kind": "upload_mismatch",
            "project_id": project.get("id") if project else None,
            "project_name": project.get("name") if project else None,
            "episode_id": episode.get("id"),
            "workflow_id": None,
            "workflow_name": "No compatible workflow",
            "wanted": wanted,
            "available": sorted(available),
            "message": (
                f"No bound workflow matches episode inputs "
                f"{sorted(available)} (checked {len(mismatches)} workflow(s))"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        print(f"[WorkflowDispatch] Failed to record mismatch summary: {exc}")


def _record_partial_input_warnings(project: dict | None, episode: dict,
                                   partials: list[dict]) -> None:
    """Persist missing inputs while still allowing matched inputs to run."""
    for item in partials:
        missing = [str(value) for value in item.get("missing") or [] if str(value).strip()]
        if not missing:
            continue
        available = [str(value) for value in item.get("available") or []]
        matched = [str(value) for value in item.get("matched") or []]
        try:
            add_exception({
                "id": str(uuid4()),
                "kind": "input_missing",
                "project_id": project.get("id") if project else None,
                "project_name": project.get("name") if project else None,
                "episode_id": episode.get("id"),
                "workflow_id": item.get("workflow_id"),
                "workflow_name": item.get("workflow_name"),
                "wanted": list(item.get("wanted") or []),
                "matched": matched,
                "missing": missing,
                "available": available,
                "message": (
                    f"Workflow partially matched; missing inputs: {missing}. "
                    f"Only available inputs will be processed: {matched or available}."
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            print(f"[WorkflowDispatch] Failed to record partial input warning: {exc}")


def _clear_episode_input_warnings(episode_id: str) -> None:
    """Clear partial-input warnings after a complete input match exists."""
    try:
        for exc in list_exceptions():
            if (exc.get("episode_id") == episode_id
                    and exc.get("kind") in {"upload_mismatch", "input_missing"}):
                delete_exception(exc.get("id"))
    except Exception as exc:
        print(f"[WorkflowDispatch] Failed to clear input warnings: {exc}")


def prepare_workflow_run(workflow: dict, episode: dict, project: dict | None = None,
                         bindings: dict | None = None,
                         record_mismatch: bool = True,
                         mismatch_records: list[dict] | None = None,
                         partial_records: list[dict] | None = None,
                         ) -> tuple[dict | None, str | None]:
    """Match real episode inputs and return an immutable run snapshot.

    Matching is type-first (RGB / Stereo RGB / RGB-D / Stereo RGB-D / Glove Sensor),
    then source-name-aware. This lets a workflow be assembled before upload
    and keeps stale source names from an older workflow from blocking a run.
    """
    bindings = bindings if isinstance(bindings, dict) else {}
    cameras, sensors = _episode_sources(episode)
    specs = _workflow_input_specs(workflow, bindings)
    if not specs:
        return None, "no_supported_inputs"

    auto, matched, missing, groups = _auto_bindings_for_episode(
        workflow, episode, bindings)
    if not matched:
        wanted = [spec["semantic_type"] for spec in specs]
        message = f"Workflow expects input types {wanted}, episode has {sorted(cameras | sensors)}"
        if mismatch_records is not None:
            mismatch_records.append({
                "wanted": wanted,
                "available": sorted(cameras | sensors),
            })
        elif record_mismatch:
            _record_mismatch(project, episode, workflow, wanted, cameras | sensors, message)
        return None, "input_type_mismatch"

    bindings = {**bindings, **auto}
    if partial_records is not None and missing:
        partial_records.append({
            "workflow_id": workflow.get("id"),
            "workflow_name": workflow.get("name"),
            "wanted": [spec["semantic_type"] for spec in specs],
            "matched": [spec["semantic_type"] for spec in specs
                        if spec["node_id"] in matched],
            "missing": [spec["semantic_type"] for spec in missing],
            "available": sorted(cameras | sensors),
        })

    bound = dict(workflow)
    bound["graph"] = apply_bindings(workflow.get("graph") or {}, bindings)
    _clear_unmatched_input_configs(bound["graph"], matched)
    return bound, None


def enqueue_workflow_once(workflow: dict, episode: dict, project: dict | None = None,
                          bindings: dict | None = None, trigger: str = "",
                          record_mismatch: bool = True,
                          mismatch_records: list[dict] | None = None,
                          partial_records: list[dict] | None = None,
                          force_rerun: bool = False) -> dict:
    """匹配并以工作流版本幂等入队。返回统一结果。"""
    bound, reason = prepare_workflow_run(
        workflow, episode, project, bindings,
        record_mismatch=record_mismatch,
        mismatch_records=mismatch_records,
        partial_records=partial_records,
    )
    if bound is None:
        return {"status": "skipped", "reason": reason, "workflow_id": workflow.get("id")}

    # A forced reprocess starts a new annotation generation.  Do not carry
    # old manual or AI segments into the new run.
    episode_id = str(episode["id"])
    if force_rerun:
        try:
            from app.ai_annotation import invalidate_ai_annotation_tasks
            invalidate_ai_annotation_tasks(episode_id)
        except Exception as exc:
            # Annotation cleanup remains valid even if the optional AI module
            # is unavailable during a worker-only dispatch.
            print(f"[WorkflowDispatch] AI task invalidation skipped: {exc}")
        from app.localstore import save_annotations
        save_annotations(episode_id, [])

    # 延迟导入避免 workflow API 与派发器互相导入。
    from app.api.workflows import _build_run_record

    # ``bound`` 已包含名称匹配产生的自动绑定；指纹必须基于最终 snapshot，
    # 否则同一工作流切换到另一台同类型相机时会被错误去重。
    revision = workflow_revision(bound, {})
    run = _build_run_record(bound, str(episode["id"]),
                            project_id=project.get("id") if project else None,
                            workflow_revision=revision, trigger=trigger)
    previous_status = read_episode_state(episode_id).get("status")
    # Mark processing before publishing the run record.  This closes the
    # race where a fast worker completed the run and the dispatcher then
    # wrote processing after the worker had already written to_review.
    if previous_status != "processing":
        set_episode_status(episode_id, "processing")
    try:
        saved, created = save_run_if_absent(
            run,
            allow_completed_rerun=force_rerun,
            supersede_active=force_rerun,
        )
    except Exception:
        # Do not leave a batch stuck in processing if enqueue itself fails.
        if read_episode_state(episode_id).get("status") == "processing":
            set_episode_status(episode_id, previous_status or "to_review")
        raise
    if not created and saved.get("status") in ("completed", "failed"):
        # A normal upload/backfill may find an already completed run. Restore
        # the old state; only a newly created run should move it to processing.
        if read_episode_state(episode_id).get("status") == "processing":
            set_episode_status(episode_id, previous_status or "to_review")
    if created:
        return {"status": "queued", "run": saved, "workflow_id": workflow.get("id")}
    return {"status": "already_scheduled", "run": saved, "workflow_id": workflow.get("id")}


def dispatch_project_episode(project: dict | None, episode: dict,
                             trigger: str = "upload",
                             workflow_ids: list[str] | None = None,
                             force_rerun: bool = False) -> dict:
    """派发一个项目批次的全部 active 工作流。"""
    if not project or project.get("status", "active") != "active":
        return {"queued": 0, "matched": 0, "skipped": 0}
    bindings_all = project.get("workflow_bindings") or {}
    if not isinstance(bindings_all, dict):
        bindings_all = {}
    stats = {"queued": 0, "matched": 0, "skipped": 0, "already_scheduled": 0}
    candidate_ids = project_workflow_ids(project)
    if workflow_ids is not None:
        allowed = {str(value) for value in workflow_ids}
        candidate_ids = [value for value in candidate_ids if value in allowed]
    mismatches: list[dict] = []
    partials: list[dict] = []
    for workflow_id in candidate_ids:
        workflow = get_workflow(workflow_id)
        if not workflow or workflow.get("status", "draft") != "active":
            stats["skipped"] += 1
            continue
        result = enqueue_workflow_once(
            workflow, episode, project,
            bindings_all.get(workflow_id) if isinstance(bindings_all.get(workflow_id), dict) else {},
            trigger=trigger,
            record_mismatch=False,
            mismatch_records=mismatches,
            partial_records=partials,
            force_rerun=force_rerun,
        )
        if result["status"] == "queued":
            stats["queued"] += 1
            stats["matched"] += 1
        elif result["status"] == "already_scheduled":
            stats["already_scheduled"] += 1
            stats["matched"] += 1
        else:
            stats["skipped"] += 1
    if stats["matched"]:
        # One valid workflow is enough for an episode. Other candidate
        # workflows are alternatives, not errors for this same batch. A
        # partial match is different: keep a visible warning for review.
        if partials:
            _clear_episode_mismatches(str(episode.get("id")))
            _record_partial_input_warnings(project, episode, partials)
        else:
            _clear_episode_input_warnings(str(episode.get("id")))
    elif mismatches:
        _record_mismatch_summary(project, episode, mismatches)
    return stats


def backfill_project(project: dict, workflow_ids: list[str] | None = None) -> dict:
    """工作流首次绑定/设备绑定改变后，回填项目内未完成的历史批次。"""
    wanted = workflow_ids or project_workflow_ids(project)
    wanted = {str(value) for value in wanted if value}
    stats = {"episodes": 0, "queued": 0, "already_scheduled": 0, "skipped": 0}
    # Backfill is for batches that have never entered this workflow.  A user
    # who wants to apply a changed workflow to an already processed batch uses
    # the explicit Reprocess action, which can force a new run safely.
    existing = {
        (str(run.get("workflow_id") or ""), str(run.get("episode_id") or ""))
        for run in list_runs()
        if run.get("workflow_id") and run.get("episode_id")
    }
    for episode in scan_sessions():
        if episode.get("project") != project.get("name"):
            continue
        stats["episodes"] += 1
        if any((workflow_id, str(episode.get("id") or "")) in existing
               for workflow_id in wanted):
            stats["skipped"] += 1
            continue
        result = dispatch_project_episode(
            project, episode, trigger="project_backfill", workflow_ids=list(wanted)
        )
        for key in ("queued", "already_scheduled", "skipped"):
            stats[key] += result.get(key, 0)
    return stats
