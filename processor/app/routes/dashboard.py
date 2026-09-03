"""Dashboard API — 文件系统扫描统计(无数据库)。"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Query

from app.localstore import (
    scan_sessions, list_deleted_episodes, read_episode_state, list_projects,
)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def _today_start() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _alive_episodes() -> list[dict]:
    deleted = {d["id"] for d in list_deleted_episodes()}
    return [e for e in scan_sessions() if e["id"] not in deleted]


def _status_of(e: dict) -> str:
    return e.get("status") or "unknown"


@router.get("/overview")
def dashboard_overview():
    """聚合统计 — 6 张统计卡。"""
    today = _today_start()
    eps = _alive_episodes()

    def _in_today(e: dict) -> bool:
        try:
            return datetime.fromisoformat(e.get("created_at") or "") >= today
        except Exception:
            return False

    total_all = len(eps)
    total_today = sum(1 for e in eps if _in_today(e))
    reviewing = [e for e in eps if _status_of(e) in ("completed", "to_review")]
    approved = [e for e in eps if _status_of(e) in ("reviewed", "approved")]
    failed = [e for e in eps if _status_of(e) == "failed"]

    return {
        "total": {"total": total_all, "today": total_today, "label": "All Episodes"},
        "reviewing": {"total": len(reviewing), "today": sum(1 for e in reviewing if _in_today(e)), "label": "To Review"},
        "approved": {"total": len(approved), "today": sum(1 for e in approved if _in_today(e)), "label": "Approved"},
        # 首页展示当前仍失败的批次，不把同一批次历史重试产生的多个
        # failed run 或异常详情重复累加。历史失败运行仍可在异常页追溯。
        "failed": {"total": len(failed), "today": sum(1 for e in failed if _in_today(e)), "label": "Failed"},
        "active_tasks": {"total": len(list_projects()), "today": 0, "label": "Active Tasks"},
        "datasets": {"total": 0, "today": 0, "label": "Datasets"},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/recent-episodes")
def recent_episodes(limit: int = Query(default=6, ge=1, le=20)):
    eps = sorted(_alive_episodes(), key=lambda e: e.get("created_at") or "", reverse=True)[:limit]
    episodes = []
    for e in eps:
        duration = 0.0
        fps = e.get("fps") or 30
        fc = e.get("frame_count") or 0
        if fps and fc:
            duration = round(fc / fps, 1)
        state = read_episode_state(e["id"])
        episodes.append({
            "id": e["id"],
            "task_name": e.get("project") or e.get("name") or e["id"],
            "status": e.get("status"),
            "received_at": e.get("created_at"),
            "duration_sec": duration,
            "frame_count": fc,
            "cleaning_passed": (state.get("cleaning_report") or {}).get("passed"),
        })
    return {"episodes": episodes}


@router.get("/pipeline")
def pipeline_status():
    eps = _alive_episodes()
    stages = [
        {"name": "Received", "count": sum(1 for e in eps if _status_of(e) == "received")},
        {"name": "Processing", "count": sum(1 for e in eps if _status_of(e) == "processing")},
        {"name": "Review", "count": sum(1 for e in eps if _status_of(e) in ("completed", "to_review"))},
        {"name": "Approved", "count": sum(1 for e in eps if _status_of(e) in ("reviewed", "approved"))},
        {"name": "Dataset", "count": 0},
    ]
    return {"stages": stages}


@router.get("/trend")
def trend(days: int = Query(default=30, ge=7, le=90)):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    day_map: dict[str, int] = {}
    for e in _alive_episodes():
        try:
            dt = datetime.fromisoformat(e.get("created_at") or "")
        except Exception:
            continue
        if dt >= cutoff:
            key = str(dt.date())
            day_map[key] = day_map.get(key, 0) + 1

    labels, counts = [], []
    for i in range(days - 1, -1, -1):
        d = datetime.now(timezone.utc) - timedelta(days=i)
        key = str(d.date())
        labels.append(d.strftime("%m-%d"))
        counts.append(day_map.get(key, 0))
    return {"labels": labels, "counts": counts}
