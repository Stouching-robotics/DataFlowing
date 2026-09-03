"""Exception aggregation — 输入异常(落盘)∪ 运行失败(聚合 runs 生成)。

upload_mismatch/input_missing:上传输入不匹配或部分缺失时写入 data/state/exceptions.json
(可删除);run_failed:从 data/state/runs/ 里 status=="failed" 的记录动态生成
(id="run-<run_id>", message=error_log),不落盘、不可删除。两者按 created_at
倒序合并展示,全部 try/except 容错,坏 JSON/已删 episode 不 500。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.localstore import (
    list_exceptions, delete_exception, clear_exceptions,
    list_runs, scan_sessions, list_projects,
)

router = APIRouter(prefix="/api/v1/exceptions", tags=["exceptions"])


def _aggregate_exceptions(project_id: str | None = None) -> list[dict]:
    """合并两类异常:落盘的 upload_mismatch + 动态生成的 run_failed。"""
    items: list[dict] = []
    try:
        for e in list_exceptions():
            if e.get("kind") in {"upload_mismatch", "input_missing"}:
                items.append(e)
    except Exception as exc:
        print(f"[Exceptions] Failed to read exceptions.json: {exc}")
    try:
        eps_by_id = {e["id"]: e for e in scan_sessions()}
        proj_by_name = {p.get("name"): p for p in list_projects()}
        for run in list_runs():
            if not isinstance(run, dict) or run.get("status") != "failed":
                continue
            ep = eps_by_id.get(run.get("episode_id")) or {}
            pname = ep.get("project")
            proj = proj_by_name.get(pname)
            items.append({
                "id": f"run-{run.get('id')}",
                "kind": "run_failed",
                "project_id": proj.get("id") if proj else None,
                "project_name": pname,
                "episode_id": run.get("episode_id"),
                "workflow_id": run.get("workflow_id"),
                "workflow_name": run.get("workflow_name"),
                "wanted": [],
                "available": [],
                "message": run.get("error_log") or "Run failed",
                "run_id": run.get("id"),
                "created_at": run.get("finished_at") or run.get("created_at"),
            })
    except Exception as exc:
        print(f"[Exceptions] Failed to aggregate run failures: {exc}")
    if project_id:
        items = [x for x in items if x.get("project_id") == project_id]
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items


@router.get("")
def list_exceptions_api(
    project_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    items = _aggregate_exceptions(project_id)
    return {"exceptions": items[offset:offset + limit], "total": len(items)}


@router.delete("/{exception_id}")
def delete_exception_api(exception_id: str):
    # run_failed 是聚合生成(无落盘),不能删除
    if exception_id.startswith("run-"):
        raise HTTPException(status_code=404, detail="Exception not found")
    if not delete_exception(exception_id):
        raise HTTPException(status_code=404, detail="Exception not found")
    return {"message": "Exception deleted"}


@router.delete("", status_code=200)
def clear_exceptions_api():
    """清空所有落盘的输入异常(不影响动态生成的 run_failed)。"""
    return {"deleted": clear_exceptions()}
