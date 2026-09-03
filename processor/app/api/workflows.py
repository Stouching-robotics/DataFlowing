"""Workflow CRUD, module catalog and async run queue — 本地 JSON(无数据库)。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import get_current_user
from app.localstore import (
    list_workflows, get_workflow, upsert_workflow, delete_workflow,
    list_runs, get_run, get_episode,
    scan_sessions, save_run_if_absent, read_episode_state, set_episode_status,
)
from app.processing.catalog import module_catalog
from app.glove_sources import merge_paired_glove_nodes
from app.workflow_types import migrate_graph_types

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workflow_project_id(workflow_id: str) -> str | None:
    """Return the owning project for legacy workflows without project_id.

    Older workflow records only existed on the project side.  Resolving that
    relation here keeps direct workflow URLs consistent while the next save
    persists the normalized field.
    """
    try:
        from app.localstore import list_projects
        for project in list_projects():
            workflow_ids = project.get("workflow_ids") or []
            if not isinstance(workflow_ids, list):
                workflow_ids = [workflow_ids] if workflow_ids else []
            if project.get("workflow_id"):
                workflow_ids = [project.get("workflow_id"), *workflow_ids]
            if workflow_id in {str(value) for value in workflow_ids if value}:
                return str(project.get("id")) if project.get("id") else None
    except Exception:
        return None
    return None


def _bind_workflow_to_project(workflow_id: str, project_id: str | None) -> None:
    """Keep the project-side one-workflow pointer in sync with the relation."""
    if not project_id:
        return
    try:
        from app.localstore import list_projects, upsert_project
        for project in list_projects():
            if str(project.get("id")) != str(project_id):
                continue
            old_ids = project.get("workflow_ids") or []
            if not isinstance(old_ids, list):
                old_ids = [old_ids] if old_ids else []
            old_id = project.get("workflow_id") or (old_ids[0] if old_ids else None)
            project["workflow_id"] = workflow_id
            project["workflow_ids"] = [workflow_id]
            upsert_project(project)
            if old_id and str(old_id) != str(workflow_id):
                old_workflow = get_workflow(str(old_id))
                if old_workflow and old_workflow.get("project_id") == str(project_id):
                    old_workflow["project_id"] = None
                    upsert_workflow(old_workflow)
            return
    except Exception as exc:
        print(f"[Workflows] Project relation sync skipped: {exc}")


def _wf_out(w: dict) -> dict:
    graph, _ = migrate_graph_types(w.get("graph") or {})
    # Normalize workflows generated before left/right glove channels were
    # represented by one physical device node.
    merge_paired_glove_nodes(graph)
    nodes = graph.get("nodes") or []
    # React Flow 节点必需 position:手写/模板 graph 可能缺失,按序号自动布局
    for i, node in enumerate(nodes):
        if not isinstance(node.get("position"), dict):
            node["position"] = {"x": 60 + (i % 3) * 280, "y": 40 + (i // 3) * 200}
    graph["nodes"] = nodes
    project_id = w.get("project_id") or _workflow_project_id(str(w.get("id")))
    return {
        "id": w["id"], "name": w["name"], "description": w.get("description"),
        "project_id": project_id,
        "graph": graph, "node_configs": w.get("node_configs") or {},
        "status": w.get("status", "draft"),
        "is_preset": bool(w.get("is_preset")),
        "node_count": len(nodes),   # 前端下拉 empty 标记
        "created_at": w.get("created_at"), "updated_at": w.get("updated_at"),
    }


def _run_out(r: dict) -> dict:
    return {
        "id": r["id"], "workflow_id": r.get("workflow_id"),
        "episode_id": r.get("episode_id"), "status": r.get("status", "queued"),
        "started_at": r.get("started_at"), "finished_at": r.get("finished_at"),
        "node_states": r.get("node_states") or {}, "error_log": r.get("error_log"),
        "worker_id": r.get("worker_id"), "attempt": r.get("attempt", 0),
        "progress": r.get("progress", 0.0), "outputs": r.get("outputs") or {},
        "created_at": r.get("created_at"),
    }


def _norm_source(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _workflow_input_source_keys(workflow: dict) -> list[str]:
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
        values = (config.get("source_keys") or config.get("source_key") or config.get("position"))
        if isinstance(values, str):
            # 逗号分隔的多源("stereo_left,stereo_right")要拆开,
            # 否则上传匹配时拿整串字符串比较,双目数据永远匹配不上。
            values = [v.strip() for v in values.split(",") if v.strip()]
        if isinstance(values, (list, tuple, set)):
            keys.extend(str(value) for value in values if value)
    return keys


@router.get("/modules")
def list_modules():
    """Workflow Studio 的模块/卡片目录。"""
    return module_catalog()


@router.get("")
def list_workflows_api(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
):
    workflows = list_workflows()
    if status:
        workflows = [w for w in workflows if w.get("status") == status]
    total = len(workflows)
    return {"workflows": [_wf_out(w) for w in workflows[offset:offset + limit]],
            "total": total, "limit": limit, "offset": offset}


@router.post("", status_code=201)
def create_workflow(body: dict):
    graph, _ = migrate_graph_types(body.get("graph") or {})
    wf = {
        "id": str(uuid4()),
        "name": body.get("name", "Untitled"),
        "project_id": str(body["project_id"]) if body.get("project_id") else None,
        "description": body.get("description"),
        "graph": graph,
        "node_configs": body.get("node_configs") or {},
        "status": body.get("status", "draft"),
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
    }
    merge_paired_glove_nodes(wf["graph"])
    upsert_workflow(wf)
    _bind_workflow_to_project(wf["id"], wf.get("project_id"))
    return _wf_out(wf)


@router.get("/{workflow_id}")
def get_workflow_api(workflow_id: str):
    wf = get_workflow(workflow_id)
    if wf is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Workflow not found")
    return _wf_out(wf)


def _require_preset_edit_perm(wf: dict, current_user: dict) -> None:
    """模板权限:非 admin 不能改/删模板,也不能设置或取消模板标记。"""
    if wf.get("is_preset") and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Template workflows can only be modified by an admin. Save a copy instead.",
        )


@router.put("/{workflow_id}")
def update_workflow_api(
    workflow_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    wf = get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    _require_preset_edit_perm(wf, current_user)
    # 新增/取消预设(提升为预设)仅限超管 —— 非 admin 提交 is_preset 一律拒绝
    if "is_preset" in body and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admins can create or remove preset workflows.",
        )
    for key in ("name", "project_id", "description", "graph", "node_configs", "status", "is_preset"):
        if key in body:
            wf[key] = str(body[key]) if key == "project_id" and body[key] else (None if key == "project_id" else body[key])
    wf["graph"], _ = migrate_graph_types(wf.get("graph") or {})
    merge_paired_glove_nodes(wf.get("graph") or {})
    wf["updated_at"] = _utcnow_iso()
    upsert_workflow(wf)
    _bind_workflow_to_project(wf["id"], wf.get("project_id"))
    # 保存 active 工作流后，已绑定项目中从未运行过的历史批次自动回填。
    # 运行版本指纹与 backfill 的历史运行过滤保证重复保存不会重复入队。
    if wf.get("status") == "active":
        try:
            from app.localstore import list_projects
            from app.workflow_dispatch import backfill_project, project_workflow_ids
            for project in list_projects():
                if workflow_id in project_workflow_ids(project):
                    backfill_project(project, [workflow_id])
        except Exception as exc:
            print(f"[Workflows] Publish backfill skipped: {exc}")
    return _wf_out(wf)


@router.delete("/{workflow_id}", status_code=204)
def delete_workflow_api(
    workflow_id: str,
    current_user: dict = Depends(get_current_user),
):
    wf = get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    _require_preset_edit_perm(wf, current_user)
    delete_workflow(workflow_id)


def _build_run_record(workflow: dict, episode_id: str, *, project_id: str | None = None,
                      workflow_revision: str | None = None,
                      trigger: str = "") -> dict:
    graph = workflow.get("graph") or {}
    states = {
        node.get("id"): {"type": node.get("data", {}).get("nodeType", ""), "status": "queued"}
        for node in graph.get("nodes", []) if node.get("id")
    }
    run = {
        "id": str(uuid4()),
        "workflow_id": workflow["id"],
        "episode_id": episode_id,
        "status": "queued",
        "node_states": states,
        "outputs": {},
        "attempt": 0,
        "progress": 0.0,
        # Worker 执行必需:graph + node_configs + 名称
        "workflow_name": workflow.get("name"),
        "graph": graph,
        "node_configs": workflow.get("node_configs") or {},
        "created_at": _utcnow_iso(),
    }
    if project_id:
        run["project_id"] = project_id
    if workflow_revision:
        run["workflow_revision"] = workflow_revision
    if trigger:
        run["trigger"] = trigger
    return run


def _enqueue_run(workflow: dict, episode_id: str, *, project_id: str | None = None,
                 workflow_revision: str | None = None,
                 trigger: str = "", force_rerun: bool = False) -> dict:
    """兼容旧调用的入队入口；新派发链路传入版本指纹做幂等。"""
    if not workflow_revision:
        from app.workflow_dispatch import workflow_revision as _revision
        workflow_revision = _revision(workflow, {})
    run = _build_run_record(
        workflow, episode_id, project_id=project_id,
        workflow_revision=workflow_revision, trigger=trigger,
    )
    if force_rerun:
        try:
            from app.ai_annotation import invalidate_ai_annotation_tasks
            invalidate_ai_annotation_tasks(episode_id)
        except Exception as exc:
            print(f"[Workflows] AI task invalidation skipped: {exc}")
        from app.localstore import save_annotations
        save_annotations(episode_id, [])
    previous_status = read_episode_state(episode_id).get("status")
    if previous_status != "processing":
        set_episode_status(episode_id, "processing")
    try:
        saved, created = save_run_if_absent(
            run,
            allow_completed_rerun=force_rerun,
            supersede_active=force_rerun,
        )
    except Exception:
        if read_episode_state(episode_id).get("status") == "processing":
            set_episode_status(episode_id, previous_status or "to_review")
        raise
    if (not created and saved.get("status") in ("completed", "failed")
            and read_episode_state(episode_id).get("status") == "processing"):
        set_episode_status(episode_id, previous_status or "to_review")
    return saved


@router.get("/{workflow_id}/usage")
def workflow_usage_api(workflow_id: str):
    """哪些项目绑定了该工作流(Studio 改全局时提示影响范围)。"""
    from fastapi import HTTPException
    from app.localstore import list_projects
    wf = get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    used_by = []
    for p in list_projects():
        ids = p.get("workflow_id")
        if ids:
            ids = [str(ids)]
        else:
            ids = p.get("workflow_ids") or []
            if not isinstance(ids, list):
                ids = [ids] if ids else []
            ids = [str(i) for i in ids if i]
        if workflow_id in ids:
            used_by.append({"id": p["id"], "name": p.get("name")})
    return {"workflow_id": workflow_id, "project_count": len(used_by), "projects": used_by}


@router.post("/{workflow_id}/run", status_code=202)
def run_workflow_api(
    workflow_id: str,
    episode_id: str | None = Query(None, description="批次名;省略用最新"),
):
    from fastapi import HTTPException
    wf = get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if episode_id:
        ep = get_episode(episode_id)
        if ep is None:
            raise HTTPException(status_code=404, detail="Episode not found")
    else:
        episodes = scan_sessions()
        if not episodes:
            raise HTTPException(status_code=404, detail="No episode available")
        ep = episodes[-1]
        episode_id = ep["id"]
    # Keep the manual Studio run on the same type-first resolver as upload and
    # Review reprocess. Otherwise this legacy endpoint could bypass the fixed
    # input-category contract and run a stale source_key from an old graph.
    from app.workflow_dispatch import enqueue_workflow_once
    result = enqueue_workflow_once(
        wf, ep, trigger="manual_run", force_rerun=True,
    )
    if result.get("status") == "skipped":
        raise HTTPException(
            status_code=409,
            detail="No connected workflow input matches this episode",
        )
    run = result["run"]
    return _run_out(run)


@router.get("/{workflow_id}/runs")
def list_runs_api(workflow_id: str, limit: int = Query(20, ge=1, le=100)):
    runs = [r for r in list_runs() if r.get("workflow_id") == workflow_id]
    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return [_run_out(r) for r in runs[:limit]]


@router.get("/runs/{run_id}")
def get_run_api(run_id: str):
    from fastapi import HTTPException
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    return _run_out(run)


@router.post("/runs/{run_id}/retry", status_code=202)
def retry_run_api(run_id: str):
    from fastapi import HTTPException
    previous = get_run(run_id)
    if previous is None:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    if previous.get("status") != "failed":
        raise HTTPException(status_code=409, detail="Only failed runs can be retried")
    wf = get_workflow(previous.get("workflow_id"))
    ep = get_episode(previous.get("episode_id"))
    if wf is None or ep is None:
        raise HTTPException(status_code=404, detail="Workflow or episode no longer exists")
    from app.workflow_dispatch import enqueue_workflow_once
    result = enqueue_workflow_once(
        wf, ep, trigger="run_retry", force_rerun=True,
    )
    if result.get("status") == "skipped":
        raise HTTPException(
            status_code=409,
            detail="No connected workflow input matches this episode",
        )
    return _run_out(result["run"])
