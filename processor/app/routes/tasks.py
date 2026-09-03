"""Projects API (原 tasks) — 按项目聚合 episode,列出状态分布。

任务概念已移除:episode 归属项目(Session.project_id → Project.name),
这里按项目名分组统计,供导航树和项目页面使用。
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Episode, Session as SessionModel, Project
from app.routes.ingestion import _ep_to_out

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.get("/summary")
async def project_summary(
    status: str | None = Query(default=None, description="Filter by episode status"),
    search: str | None = Query(default=None, description="Search project name"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_session),
):
    """按项目聚合 episode 状态分布(原 /api/v1/tasks)。

    返回字段保留 ``task_name``(= 项目名)以兼容旧导航树调用。
    """
    # 聚合:project → status → counts
    cols = [
        Project.name.label("project_name"),
        Episode.status,
        func.count(Episode.id).label("cnt"),
        func.avg(Episode.frame_count).label("avg_frames"),
        func.max(Episode.received_at).label("last_upload"),
    ]
    base = (
        select(*cols)
        .select_from(Episode)
        .join(SessionModel, Episode.session_id == SessionModel.id)
        .join(Project, SessionModel.project_id == Project.id)
        .where(Episode.deleted_at.is_(None))
        .group_by(Project.name, Episode.status)
    )
    if status:
        base = base.where(Episode.status == status)

    rows = (await db.execute(base)).all()

    # 无项目归属的 episode → Uncategorized
    orphan_base = (
        select(
            func.count(Episode.id).label("cnt"),
            Episode.status,
        )
        .select_from(Episode)
        .outerjoin(SessionModel, Episode.session_id == SessionModel.id)
        .where(Episode.deleted_at.is_(None), SessionModel.id.is_(None))
        .group_by(Episode.status)
    )
    if status:
        orphan_base = orphan_base.where(Episode.status == status)
    orphan_rows = (await db.execute(orphan_base)).all()

    projects_map: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    for pname, ep_status, cnt, avg_fr, last_up in rows:
        pname = pname or "unknown"
        if pname not in projects_map:
            projects_map[pname] = {"task_name": pname, "project_name": pname,
                                   "total_episodes": 0, "status_counts": {}}
            stats[pname] = {"avg_frames": 0, "last_upload_at": None, "failed_count": 0}
        projects_map[pname]["total_episodes"] += cnt
        projects_map[pname]["status_counts"][ep_status] = cnt
        if avg_fr:
            stats[pname]["avg_frames"] = round(float(avg_fr), 1)
        if last_up:
            stats[pname]["last_upload_at"] = last_up.isoformat()
        if ep_status == "failed":
            stats[pname]["failed_count"] += cnt

    for _cnt, ep_status in orphan_rows:
        pname = "Uncategorized"
        if pname not in projects_map:
            projects_map[pname] = {"task_name": pname, "project_name": pname,
                                   "total_episodes": 0, "status_counts": {}}
            stats[pname] = {"avg_frames": 0, "last_upload_at": None, "failed_count": 0}
        projects_map[pname]["total_episodes"] += _cnt
        projects_map[pname]["status_counts"][ep_status] = _cnt

    projects = list(projects_map.values())
    if search:
        low = search.lower()
        projects = [p for p in projects if low in p["task_name"].lower()]
    projects.sort(key=lambda p: p["total_episodes"], reverse=True)

    for p in projects:
        st = stats.get(p["task_name"], {})
        p["avg_frames"] = st.get("avg_frames", 0)
        p["last_upload_at"] = st.get("last_upload_at")
        p["cleaning_failed"] = st.get("failed_count", 0)

    total = len(projects)
    return {"tasks": projects[offset:offset + limit], "total": total,
            "limit": limit, "offset": offset}
