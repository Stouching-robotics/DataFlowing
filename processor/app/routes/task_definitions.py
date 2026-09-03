"""Task Definition CRUD API — pre-defined collection task management."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Episode, TaskDefinition
from app.schemas import (
    TaskDefinitionCreate, TaskDefinitionUpdate,
    TaskDefinitionOut, TaskDefinitionListOut,
)

router = APIRouter(prefix="/api/v1", tags=["task-definitions"])


# ── Helpers ──────────────────────────────────────────

async def _compute_progress(db: AsyncSession, defn: TaskDefinition) -> TaskDefinitionOut:
    """Compute episode progress for a task definition (batches = uploaded zips)."""
    row = (await db.execute(
        select(
            func.count(Episode.id),
            func.max(Episode.received_at),
        ).where(
            Episode.deleted_at.is_(None),
            Episode.task_description == defn.name,
        )
    )).one()
    cur_eps = int(row[0])
    last_up = row[1]

    ep_pct = min(100.0, round(100.0 * cur_eps / defn.target_episodes, 1)) if defn.target_episodes > 0 else 0.0
    is_complete = defn.target_episodes > 0 and cur_eps >= defn.target_episodes

    return TaskDefinitionOut(
        id=defn.id,
        name=defn.name,
        description=defn.description,
        claimer=defn.claimer,
        target_episodes=defn.target_episodes,
        params=defn.params,
        status=defn.status,
        current_episodes=cur_eps,
        episode_progress_pct=ep_pct,
        is_complete=is_complete,
        last_upload_at=last_up,
        created_at=defn.created_at,
        updated_at=defn.updated_at,
    )


# ── List ─────────────────────────────────────────────

@router.get("/task-definitions", response_model=TaskDefinitionListOut)
async def list_task_definitions(
    status: str = Query(None),
    search: str = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_session),
):
    q = select(TaskDefinition).order_by(TaskDefinition.created_at.desc())
    if status:
        q = q.where(TaskDefinition.status == status)

    rows = (await db.execute(q)).scalars().all()

    # Compute progress for each definition
    out = []
    for d in rows:
        out.append(await _compute_progress(db, d))

    # Search filter (applied after progress computation)
    if search:
        s = search.lower()
        out = [d for d in out if s in d.name.lower() or (d.description and s in d.description.lower())]

    total = len(out)
    out = out[offset:offset + limit]

    return TaskDefinitionListOut(definitions=out, total=total)


# ── Create ───────────────────────────────────────────

@router.post("/task-definitions", status_code=201, response_model=TaskDefinitionOut)
async def create_task_definition(
    body: TaskDefinitionCreate,
    db: AsyncSession = Depends(get_session),
):
    # Check duplicate name
    existing = (await db.execute(
        select(TaskDefinition).where(TaskDefinition.name == body.name)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail=f"Task name '{body.name}' already exists")

    defn = TaskDefinition(
        id=uuid4(),
        name=body.name,
        description=body.description,
        claimer=body.claimer,
        target_episodes=body.target_episodes,
        fps=30,
        params=body.params,
        status=body.status,
    )
    db.add(defn)
    await db.commit()
    await db.refresh(defn)
    return await _compute_progress(db, defn)


# ── Update ───────────────────────────────────────────

@router.put("/task-definitions/{definition_id}", response_model=TaskDefinitionOut)
async def update_task_definition(
    definition_id: UUID,
    body: TaskDefinitionUpdate,
    db: AsyncSession = Depends(get_session),
):
    defn = await db.get(TaskDefinition, definition_id)
    if not defn:
        raise HTTPException(status_code=404, detail="Task definition not found")

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(defn, field, value)

    defn.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(defn)
    return await _compute_progress(db, defn)


# ── Delete ───────────────────────────────────────────

@router.delete("/task-definitions/{definition_id}")
async def delete_task_definition(
    definition_id: UUID,
    db: AsyncSession = Depends(get_session),
):
    defn = await db.get(TaskDefinition, definition_id)
    if not defn:
        raise HTTPException(status_code=404, detail="Task definition not found")
    await db.delete(defn)
    await db.commit()
    return {"message": "Task definition deleted"}
