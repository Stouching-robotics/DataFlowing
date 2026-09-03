"""Annotation APIs — 本地 JSON 存储(无数据库)。

标注数据存 data/state/annotations/<episode_id>.json:
[{"id", "episode_id", "label", "start_frame_index", "end_frame_index",
  "color", "sort_order", "notes", "created_at", "updated_at",
    "source_scope": ["stereo_left", "head_depth"],
    "keyframes": [{"id", "frame_index", "event"}]}]
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.auth import decode_access_token
from app.localstore import (
    list_annotations,
    get_episode,
    mutate_annotations,
    mutate_annotation_by_id,
)

router = APIRouter(prefix="/api/v1", tags=["annotations"])

_PALETTE = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

# ── 数据集自动同步 ─────────────────────────────────────
# 手动改标注(增/删/改/关键帧)后,防抖触发一次最新 run 的
# LeRobot 数据集重建 —— 标注列同步进 parquet；导出数据集不再生成
# 独立的 meta/annotations.jsonl(与 AI 标注完成后的自动重建同一实现)。
_sync_pending: dict[str, asyncio.Task] = {}


def _schedule_dataset_sync(episode_id: str, delay: float = 4.0) -> None:
    async def _debounced() -> None:
        try:
            await asyncio.sleep(delay)
            _sync_pending.pop(episode_id, None)
            from app.ai_annotation import _rebuild_lerobot_after_ai
            await _rebuild_lerobot_after_ai(episode_id)
        except asyncio.CancelledError:
            raise

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = _sync_pending.get(episode_id)
    if task is not None and not task.done():
        task.cancel()
    _sync_pending[episode_id] = loop.create_task(_debounced())


# ── 实时同步(跨电脑标注变化通知)──────────────────

_episode_ws_clients: dict[str, set[WebSocket]] = {}


class _ConflictError(Exception):
    """Raised inside a mutator when optimistic version check fails."""

    def __init__(self, seg: dict):
        self.seg = seg


async def _broadcast_annotations(
    episode_id: str, action: str, annotation_id: str | None = None
) -> None:
    """Notify every browser watching this episode that annotations changed.

    WebSocket 只做变化通知;保存仍走 REST,并发控制留在后端原子写入里。
    """
    clients = _episode_ws_clients.get(episode_id)
    if not clients:
        return
    message = {
        "type": "annotations_changed",
        "episode": episode_id,
        "action": action,
        "annotation_id": annotation_id,
    }
    dead: list[WebSocket] = []
    for ws in list(clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)
    if not clients:
        _episode_ws_clients.pop(episode_id, None)


def notify_annotations_changed(episode_id: str, action: str, annotation_id: str | None = None) -> None:
    """Fire-and-forget broadcast for synchronous call sites (e.g. upload-time
    auto_actions import) that cannot await the async broadcaster."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_broadcast_annotations(episode_id, action, annotation_id))


@router.websocket("/annotations/ws")
async def annotation_ws(websocket: WebSocket):
    """Per-episode change channel. Kept separate from REST saves on purpose:
    a dropped socket must never lose data, only miss a refresh notification.
    """
    episode_id = websocket.query_params.get("episode", "")
    if not episode_id or get_episode(episode_id) is None:
        await websocket.close(code=4404)
        return
    # WebSocket requests bypass AuthMiddleware (starlette passes non-http
    # scopes straight through), so authenticate here: cookie or ?token=
    token = websocket.query_params.get("token") or websocket.cookies.get("auth_token")
    if not token or decode_access_token(token) is None:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    clients = _episode_ws_clients.setdefault(episode_id, set())
    clients.add(websocket)
    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except Exception:
                break
            if isinstance(msg, dict) and msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
        if not clients:
            _episode_ws_clients.pop(episode_id, None)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seg_to_out(seg: dict) -> dict:
    return {
        "id": seg["id"],
        "episode_id": seg.get("episode_id"),
        "label": seg.get("label", ""),
        "start_frame_index": seg.get("start_frame_index", 0),
        "end_frame_index": seg.get("end_frame_index", 0),
        "color": seg.get("color", "#3B82F6"),
        "sort_order": seg.get("sort_order", 0),
        "notes": seg.get("notes"),
        "source_scope": seg.get("source_scope") or ["episode"],
        "ai_media_sources": seg.get("ai_media_sources") or [],
        "created_at": seg.get("created_at"),
        "updated_at": seg.get("updated_at"),
        "keyframes": seg.get("keyframes") or [],
        # AI 候选段字段(老数据缺省 = 人工已确认)
        "status": seg.get("status") or "confirmed",
        "source": seg.get("source") or "manual",
        "ai_score": seg.get("ai_score"),
        "ai_reason": seg.get("ai_reason"),
        "ai_instruction": seg.get("ai_instruction"),
        "ai_confidence": seg.get("ai_confidence"),
    }


def _require_episode(episode_id: str) -> None:
    if get_episode(episode_id) is None:
        raise HTTPException(status_code=404, detail="Episode not found")


def _next_color(segs: list[dict]) -> str:
    used = {s.get("color") for s in segs if s.get("color")}
    for c in _PALETTE:
        if c not in used:
            return c
    return _PALETTE[len(segs) % len(_PALETTE)]


def _time_order(segs: list[dict]) -> list[dict]:
    """Sort markers in the same left-to-right order as the video timeline."""
    return sorted(
        segs,
        key=lambda seg: (
            int(seg.get("start_frame_index", 0) or 0),
            int(seg.get("end_frame_index", 0) or 0),
            int(seg.get("sort_order", 0) or 0),
            str(seg.get("created_at") or ""),
        ),
    )


@router.get("/episode/{episode_id}/annotations")
async def list_annotation_api(episode_id: str):
    segs = _time_order(list_annotations(episode_id))
    return {"annotations": [_seg_to_out(s) for s in segs], "total": len(segs)}


@router.post("/episode/{episode_id}/annotations", status_code=201)
async def create_annotation(episode_id: str, body: dict):
    _require_episode(episode_id)
    def mutator(segs: list[dict]) -> dict:
        now = _utcnow()
        source_scope = body.get("source_scope") or ["episode"]
        if isinstance(source_scope, str):
            source_scope = [source_scope]
        if not isinstance(source_scope, list):
            source_scope = ["episode"]
        seg = {
            "id": str(uuid4()),
            "episode_id": episode_id,
            "label": body.get("label") or "unlabeled",
            "start_frame_index": int(body.get("start_frame_index", 0)),
            "end_frame_index": int(body.get("end_frame_index", 0)),
            "color": _next_color(segs),
            "sort_order": int(body.get("sort_order", len(segs))),
            "notes": body.get("notes"),
            "source_scope": [str(item) for item in source_scope if item],
            "created_at": now,
            "updated_at": now,
            "keyframes": [],
        }
        segs.append(seg)
        return seg

    seg = mutate_annotations(episode_id, mutator)
    await _broadcast_annotations(episode_id, "create", seg["id"])
    _schedule_dataset_sync(episode_id)
    return _seg_to_out(seg)


@router.put("/annotation/{annotation_id}")
async def update_annotation(annotation_id: str, body: dict):
    def mutator(segs: list[dict], index: int) -> dict:
        seg = segs[index]
        # Optimistic concurrency: the caller sends the updated_at it saw when
        # the edit started; a mismatch means another device changed the slice
        # in between, so refuse instead of silently overwriting.
        expected = body.get("updated_at")
        if expected and seg.get("updated_at") and seg["updated_at"] != expected:
            raise _ConflictError(_seg_to_out(seg))
        for key in ("label", "start_frame_index", "end_frame_index", "color",
                    "sort_order", "notes", "source_scope", "status"):
            if key not in body:
                continue
            value = body[key]
            if key == "source_scope":
                value = [value] if isinstance(value, str) else value
                value = value if isinstance(value, list) else ["episode"]
                value = [str(item) for item in value if item]
            # 手动改过内容的 AI 段标记 user_edited:下次 AI 标注不再
            # 整体替换它(仅确认 status 变化不算内容修改)。
            if (key != "status" and seg.get("source") == "ai"
                    and seg.get(key) != value):
                seg["user_edited"] = True
            seg[key] = value
        seg["updated_at"] = _utcnow()
        return seg

    try:
        found = mutate_annotation_by_id(annotation_id, mutator)
    except _ConflictError as exc:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "Annotation was modified by another device",
                "annotation": exc.seg,
            },
        )
    if found is None:
        raise HTTPException(status_code=404, detail="Annotation not found")
    ep_id, seg = found
    await _broadcast_annotations(ep_id, "update", annotation_id)
    _schedule_dataset_sync(ep_id)
    return _seg_to_out(seg)


@router.delete("/annotation/{annotation_id}")
async def delete_annotation(annotation_id: str):
    def mutator(segs: list[dict], index: int) -> dict:
        return segs.pop(index)

    found = mutate_annotation_by_id(annotation_id, mutator)
    if found is None:
        raise HTTPException(status_code=404, detail="Annotation not found")
    ep_id, _ = found
    await _broadcast_annotations(ep_id, "delete", annotation_id)
    _schedule_dataset_sync(ep_id)
    return {"message": "Annotation deleted"}


@router.post("/annotation/{annotation_id}/keyframe", status_code=201)
async def add_keyframe(annotation_id: str, body: dict):
    def mutator(segs: list[dict], index: int) -> dict:
        seg = segs[index]
        kfs = seg.setdefault("keyframes", [])
        kfs.append({
            "id": str(uuid4()),
            "frame_index": int(body.get("frame_index", 0)),
            "event": body.get("event"),
        })
        seg["updated_at"] = _utcnow()
        return seg

    found = mutate_annotation_by_id(annotation_id, mutator)
    if found is None:
        raise HTTPException(status_code=404, detail="Annotation not found")
    ep_id, seg = found
    await _broadcast_annotations(ep_id, "keyframe", annotation_id)
    _schedule_dataset_sync(ep_id)
    return {"message": "Keyframe added", "keyframe": seg["keyframes"][-1]}


def _keyframe_owner(keyframe_id: str) -> str | None:
    """Locate the annotation that holds a keyframe (keyframes have their own ids)."""
    from app.localstore import ANNOTATIONS_DIR

    if not ANNOTATIONS_DIR.is_dir():
        return None
    for f in sorted(ANNOTATIONS_DIR.glob("*.json")):
        for seg in list_annotations(f.stem):
            if any(k.get("id") == keyframe_id for k in (seg.get("keyframes") or [])):
                return seg["id"]
    return None


@router.delete("/keyframe/{keyframe_id}")
async def delete_keyframe(keyframe_id: str):
    anno_id = _keyframe_owner(keyframe_id)
    if anno_id is None:
        raise HTTPException(status_code=404, detail="Keyframe not found")

    def mutator(segs: list[dict], index: int) -> dict:
        seg = segs[index]
        seg["keyframes"] = [
            k for k in (seg.get("keyframes") or []) if k.get("id") != keyframe_id
        ]
        seg["updated_at"] = _utcnow()
        return seg

    found = mutate_annotation_by_id(anno_id, mutator)
    if found is None:
        raise HTTPException(status_code=404, detail="Keyframe not found")
    ep_id, _ = found
    await _broadcast_annotations(ep_id, "keyframe_delete", anno_id)
    return {"message": "Keyframe deleted"}


@router.get("/episode/{episode_id}/annotations/per-frame")
async def annotations_per_frame(episode_id: str):
    """每帧标注映射: {frame_index: [labels]}。"""
    segs = list_annotations(episode_id)
    per_frame: dict[int, list[str]] = {}
    for s in segs:
        for f in range(int(s.get("start_frame_index", 0)), int(s.get("end_frame_index", 0)) + 1):
            per_frame.setdefault(f, []).append(s.get("label", ""))
    return {"frame_labels": {str(k): v for k, v in per_frame.items()}}
