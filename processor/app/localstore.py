"""Local file store — canonical project datasets plus system state.

Project data is authoritative under ``data/sessions/<project>`` and each
project is exactly a LeRobot v2.1-style ``data/meta/videos`` tree.  Queue,
workflow, review and audit state stays outside projects under ``data/state``.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

from app.config import settings
from app.workflow_types import canonical_node_type, migrate_workflow_records

# Session scans and state updates can be nested (for example a status update
# during a worker completion).  A re-entrant lock keeps the metadata cache
# coherent without forcing a full remote-directory rescan for every update.
_lock = threading.RLock()
_MutationResult = TypeVar("_MutationResult")

# 短时间内重复的列表请求共享扫描结果,降低远程存储的目录往返次数。
_sessions_cache: list[dict] | None = None
_sessions_cache_at = 0.0
# The task endpoint may need a snapshot while a new upload is replacing the
# structural session cache.  Keep the last complete snapshot as a stale-but-
# fast fallback; the ingest queue refreshes it after the batch is committed.
_task_sessions_cache: list[dict] | None = None
_SESSIONS_CACHE_TTL = 300.0
# 全量遍历远程挂载很慢:只允许单一线程执行(_scan_lock),遍历本身不持有
# _lock —— 缓存过期触发的重扫不再把状态读写请求拖在锁后面排队。
_scan_lock = threading.Lock()
_scan_generation = 0          # invalidate_session_cache 递增;并发扫描据此丢弃过期结果
_state_write_generation = 0   # write_episode_state 递增;扫描期间的状态写入触发重扫
_video_probe_cache: dict[str, tuple[int, float]] = {}
_episode_state_cache: dict[str, dict] = {}
_deleted_episodes_cache: list[dict] | None = None
_deleted_episodes_cache_at = 0.0
_DELETED_EPISODES_CACHE_TTL = 60.0
_workflows_cache: list[dict] | None = None
_workflows_cache_at = 0.0
_runs_cache: list[dict] | None = None
_runs_cache_at = 0.0
# Runs are written by the API process and read by separate worker processes.
# A ten-minute process-local cache can hide a newly queued upload/reprocess
# from workers. Keep the cache short while still coalescing rapid UI scans.
_STATE_LIST_CACHE_TTL = 2.0

STATE_ROOT = settings.storage_root / "state"
EPISODE_STATES_DIR = STATE_ROOT / "episode_states"
ANNOTATIONS_DIR = STATE_ROOT / "annotations"
SESSIONS_ROOT = settings.storage_root / "sessions"
PROJECTS_FILE = STATE_ROOT / "projects.json"
WORKFLOWS_FILE = STATE_ROOT / "workflows.json"
EXCEPTIONS_FILE = STATE_ROOT / "exceptions.json"

_status_order = ["received", "processing", "to_review", "completed",
                 "reviewed", "approved", "rejected", "failed"]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def migrate_workflow_definitions() -> bool:
    """Persist the canonical node types in the local workflow store once.

    Historical run snapshots are intentionally left untouched: they are an
    audit record of what actually ran. Only editable workflow definitions are
    migrated, and the operation is idempotent.
    """
    global _workflows_cache, _workflows_cache_at
    with _lock:
        records = _read_json(WORKFLOWS_FILE, [])
        migrated, changed = migrate_workflow_records(
            records if isinstance(records, list) else [])
        if not changed:
            return False
        _write_json(WORKFLOWS_FILE, migrated)
        _workflows_cache = None
        _workflows_cache_at = 0.0
        return True


# ── JSON helpers ─────────────────────────────────────

def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ── 批次(episode)扫描 ────────────────────────────────

def _episode_index_of(name: str) -> int | None:
    """批次名序号: Test005_000028 → 28;时间戳格式返回 None。"""
    if re.search(r"_(\d{8})_\d{6}$", str(name)):
        return None
    m = re.search(r"_(\d{6})$", str(name))
    return int(m.group(1)) if m else None


def invalidate_session_cache() -> None:
    """Invalidate session metadata after a structural filesystem change.

    This is intentionally reserved for uploads, deletes and renames.  A
    normal review/workflow status change is patched into the existing cache by
    ``write_episode_state`` below, so changing one card never re-probes every
    video on the remote storage mount.
    """
    global _sessions_cache, _sessions_cache_at, _scan_generation
    with _lock:
        _scan_generation += 1  # 并发中的扫描结果作废(其快照不含本次结构变化)
        _sessions_cache = None
        _sessions_cache_at = 0.0
        _video_probe_cache.clear()


def scan_sessions() -> list[dict]:
    """扫描 data/sessions/<项目>/<批次名>/ 生成批次列表(按序号升序)。

    返回 [{id, name, episode_index, project, status, fps, frame_count,
            camera_names, timestamp, created_at, path}]

    远程挂载上的全量遍历只允许单一线程执行(_scan_lock),遍历本身不持有
    _lock:缓存过期触发的重扫不再把状态读写等无关请求拖在锁后面排队。
    遍历期间发生结构变化(invalidate)/状态写入时,按代际计数丢弃过期结果
    重扫一次。
    """
    global _sessions_cache, _sessions_cache_at, _task_sessions_cache
    while True:
        with _lock:
            now = time.monotonic()
            if (_sessions_cache is not None
                    and now - _sessions_cache_at < _SESSIONS_CACHE_TTL):
                return copy.deepcopy(_sessions_cache)
            generation = _scan_generation
            writes_generation = _state_write_generation
        with _scan_lock:
            with _lock:
                now = time.monotonic()
                if (_sessions_cache is not None
                        and now - _sessions_cache_at < _SESSIONS_CACHE_TTL):
                    # 等待 _scan_lock 期间已有线程完成扫描
                    return copy.deepcopy(_sessions_cache)
            results = _build_sessions_snapshot()
            with _lock:
                if (_scan_generation == generation
                        and _state_write_generation == writes_generation):
                    _sessions_cache = results
                    _sessions_cache_at = time.monotonic()
                    _task_sessions_cache = copy.deepcopy(results)
                    return copy.deepcopy(results)
                # 本次遍历期间发生了结构/状态变化:结果可能过期,重扫一次


def cached_sessions_for_tasks() -> list[dict] | None:
    """Return the last complete snapshot without touching remote storage.

    Device polling must remain responsive while an upload invalidates and
    rebuilds the full SSHFS-backed session snapshot.  A short-lived stale
    count is preferable to making the collector wait for a remote directory
    scan; the ingest queue publishes a fresh snapshot after commit.
    """
    with _lock:
        if _task_sessions_cache is None:
            return None
        return copy.deepcopy(_task_sessions_cache)


def _build_sessions_snapshot() -> list[dict]:
    """全量遍历批次目录组装快照(慢,远程挂载)。调用方负责单飞与发布。"""
    results: list[dict] = []
    if not SESSIONS_ROOT.is_dir():
        return results
    for project_dir in sorted(SESSIONS_ROOT.iterdir()):
        if not project_dir.is_dir():
            continue
        project = project_dir.name
        if ((project_dir / "data").is_dir()
                and (project_dir / "meta" / "episodes").is_dir()):
            results.extend(_scan_project_dir(project_dir, project))
            continue
        # Old nested batches and top-level flat sessions are intentionally not
        # part of the active scan.  They must be migrated once into the
        # canonical project tree before they become visible again.
    results.sort(key=lambda e: (
        e.get("episode_index") is None,
        e.get("episode_index") if e.get("episode_index") is not None else 10**9,
        e["name"],
    ))
    return results


def _scan_project_dir(project_dir: Path, project: str) -> list[dict]:
    """Scan a project-level LeRobot tree and expose one row per episode."""
    from app.lerobot_v21 import is_depth_source
    from app.project_dataset import episode_files, project_episode_rows

    info = _read_json(project_dir / "meta" / "info.json", {})
    if not isinstance(info, dict):
        info = {}
    rows: list[dict] = []
    for row in project_episode_rows(project_dir):
        try:
            episode_index = int(row.get("episode_index"))
        except (TypeError, ValueError):
            continue
        episode_id = str(row.get("episode_id") or row.get("source_batch")
                         or f"{project}_{episode_index:06d}")
        files = episode_files(project_dir, episode_index)
        state = read_episode_state(episode_id)
        # Project-level info is an aggregate.  Do not let it make a new
        # one-camera episode display devices that belong to another episode.
        # Per-episode collector metadata is authoritative when present.
        # New projects inline collector metadata into the episode index row.
        episode_metadata = {}
        embedded_collector = row.get("collector")
        if isinstance(embedded_collector, dict):
            candidate = embedded_collector.get("metadata.json")
            if isinstance(candidate, dict):
                episode_metadata = candidate
        if not isinstance(episode_metadata, dict):
            episode_metadata = {}
        actual_sources = {
            str(source) for source, _path in files.get("videos") or []
            if str(source).strip()
        }
        episode_device_names: dict[str, str] = {}
        for stream, value in (episode_metadata.get("device_names") or {}).items():
            if str(stream).strip() and str(value).strip():
                episode_device_names[str(stream)] = str(value)
        for stream, value in (episode_metadata.get("cameras") or {}).items():
            if not isinstance(value, dict):
                continue
            display_name = str(value.get("device_name") or value.get("device") or "").strip()
            if display_name:
                episode_device_names.setdefault(str(stream), display_name)
        episode_devices: list[dict] = []
        for raw in episode_metadata.get("devices") or []:
            if not isinstance(raw, dict):
                continue
            episode_devices.append({
                "key": str(raw.get("key") or ""),
                "kind": str(raw.get("kind") or ""),
                "name": str(raw.get("name") or ""),
                "slots": [str(slot) for slot in (raw.get("slots") or [])
                          if str(slot).strip()],
            })
            if raw.get("name"):
                for slot in raw.get("slots") or []:
                    if str(slot).strip():
                        episode_device_names.setdefault(str(slot), str(raw["name"]))
        episode_sensors = [str(value) for value in (episode_metadata.get("sensors") or [])]
        declared_fps = float(episode_metadata.get("fps") or info.get("fps") or 30)
        try:
            state_fps = float(state.get("fps") or 0)
        except (TypeError, ValueError):
            state_fps = 0.0
        try:
            state_count = int(state.get("frame_count") or 0)
        except (TypeError, ValueError):
            state_count = 0
        cameras: dict[str, dict] = {}
        depth_sources: list[str] = []
        for source, video_path in files.get("videos") or []:
            if is_depth_source(source):
                depth_sources.append(source)
                continue
            try:
                relative = video_path.relative_to(settings.storage_root)
            except ValueError:
                relative = video_path
            cameras[source] = {
                "path": str(relative).replace("\\", "/"),
                "fps": state_fps or declared_fps,
                "frame_count": state_count or int(row.get("length") or 0),
            }
        frame_count = state_count or int(row.get("length") or 0)
        project_name = state.get("project") or project
        if project != "Uncategorized":
            try:
                from app.storage import sanitize_task_name
                if sanitize_task_name(str(project_name)) != project:
                    project_name = project
            except Exception:
                project_name = project
        rows.append({
            "id": episode_id,
            "name": episode_id,
            "episode_index": episode_index,
            "project": project_name,
            "status": state.get("status", "completed"),
            "fps": state_fps or declared_fps,
            "frame_count": frame_count,
            "camera_names": sorted(cameras),
            "camera_streams": cameras,
            "depth_sources": sorted(depth_sources),
            "devices": episode_devices,
            "device_names": episode_device_names,
            "sync": {
                "master_fps": state_fps or declared_fps,
                "master_frame_count": frame_count,
                "stream_fps": {key: value.get("fps") for key, value in cameras.items()},
                "stream_frame_counts": {key: value.get("frame_count") for key, value in cameras.items()},
                "source": "episode_metadata",
            },
            "timestamp": str(episode_metadata.get("timestamp") or info.get("timestamp") or ""),
            "created_at": state.get("created_at") or _dir_mtime(project_dir),
            "path": str(project_dir),
            "dataset_root": str(project_dir),
            "episode_data": [str(path) for path in files.get("data") or []],
            "episode_videos": [str(path) for _source, path in files.get("videos") or []],
            "episode_row": dict(row),
            "has_skeleton": False,
            "sensors": sorted(set(episode_sensors)),
        })
    return rows


def warm_metadata_caches() -> None:
    """Warm remote metadata once in a background thread after startup.

    The mounted NAS is the source of truth, but opening hundreds of small JSON
    files over SSHFS is expensive.  Warming the same short-lived caches used by
    the APIs keeps the first browser request from paying that cost while the
    backend event loop is serving the UI.
    """
    for loader in (scan_sessions, list_workflows, list_runs):
        try:
            loader()
        except Exception as exc:
            print(f"[Startup] Metadata warm-up failed for {loader.__name__}: {exc}")


def _scan_batch_dir(batch_dir: Path, project: str) -> dict | None:
    """扫描一个批次目录,组装 episode 记录。"""
    name = batch_dir.name
    # meta/info.json 或 metadata.json
    info = {}
    for cand in (batch_dir / "meta" / "info.json", batch_dir / "metadata.json"):
        if cand.is_file():
            data = _read_json(cand, {})
            if isinstance(data, dict):
                info = data
                break
    # ``devices[].name`` is the physical collector/device name.  The values
    # in ``slots`` (and the keys in ``device_names``) are stream names used by
    # the files under videos/.  Keep both: UI/device binding may show
    # ``D435_depth``, while processing must still open ``D435_depth_rgb``.
    declared_devices: list[dict] = []
    device_names: dict[str, str] = {}
    raw_devices = info.get("devices") or []
    if isinstance(raw_devices, list):
        for raw in raw_devices:
            if not isinstance(raw, dict):
                continue
            device = {
                "key": str(raw.get("key") or ""),
                "kind": str(raw.get("kind") or ""),
                "name": str(raw.get("name") or ""),
                "slots": [str(slot) for slot in (raw.get("slots") or [])
                          if str(slot).strip()],
            }
            declared_devices.append(device)
            if device["name"]:
                for slot in device["slots"]:
                    device_names.setdefault(slot, device["name"])
    # Explicit mapping is authoritative when supplied by the collector.
    declared_names = info.get("device_names") or {}
    if isinstance(declared_names, dict):
        for stream, device_name in declared_names.items():
            stream_name = str(stream).strip()
            display_name = str(device_name).strip()
            if stream_name and display_name:
                device_names[stream_name] = display_name
    state = read_episode_state(name)
    declared_fps = float(info.get("fps") or 30)
    try:
        state_fps = float(state.get("fps") or 0)
    except (TypeError, ValueError):
        state_fps = 0.0
    try:
        state_frame_count = int(state.get("frame_count") or info.get("frame_count") or 0)
    except (TypeError, ValueError):
        state_frame_count = 0
    cameras: dict[str, dict] = {}
    videos_root = batch_dir / "videos"
    if videos_root.is_dir():
        # LeRobot v2.1 stores streams below
        # videos/observation.images.<source>/chunk-000/episode_XXXXXX.mp4.
        # Resolve the source-aware path so ``chunk-000`` is never exposed as
        # a camera/device name.
        from app.lerobot_v21 import is_depth_source, iter_video_streams
        for source_key, video_path in iter_video_streams(videos_root):
            if is_depth_source(source_key):
                # Pure depth is a capability of the physical device, not a
                # standalone RGB camera in the session list.  It remains
                # available through devices/device_names and workflow pairing.
                continue
            cam = source_key
            if "_aux" in video_path.stem and "_aux" not in cam:
                cam = f"{cam}_aux"
            if cam in cameras:
                continue
            # Listing a project must not open every remote video.  The upload
            # state stores the authoritative shared timeline; use it when
            # present; otherwise probe the canonical container once.
            if state_frame_count > 0 and (state_fps > 0 or declared_fps > 0):
                frame_count = state_frame_count
                stream_fps = state_fps or declared_fps
            else:
                frame_count, stream_fps = _probe_video_cached(video_path)
            try:
                relative_path = video_path.relative_to(settings.storage_root)
            except ValueError:
                relative_path = video_path
            cameras[cam] = {
                "path": str(relative_path).replace("\\", "/"),
                "fps": stream_fps or declared_fps,
                "frame_count": frame_count,
            }
    camera_names = sorted(cameras.keys())
    for camera_name, stream in cameras.items():
        # ``device_name`` is presentation/metadata only.  The dictionary key
        # and ``path`` remain the real source stream used by processing.
        base_name = camera_name[:-5] if camera_name.endswith("_aux") else camera_name
        stream_name = device_names.get(camera_name) or device_names.get(base_name)
        if stream_name:
            stream["device_name"] = stream_name
    # 嵌套 sessions/<项目>/<批次> 布局中,父目录才是批次的物理归属。
    # 旧版本曾把上传时的 task_name(Test_Data) 写进 state.project,并在
    # 这里无条件优先使用它,于是同一项目下的批次被层级 API 拆成了一个
    # 名为 Test_Data 的“项目”。保留与目录名等价的原始项目名(例如项目
    # 名含空格时)以兼容项目改名逻辑;不等价的旧值一律视为过期元数据。
    stored_project = state.get("project") or project
    if project != "Uncategorized" and stored_project:
        try:
            from app.storage import sanitize_task_name
            same_project = sanitize_task_name(str(stored_project)) == project
        except Exception:
            same_project = str(stored_project) == project
        if not same_project:
            stored_project = project
    first_stream = next((c for c in cameras.values() if c.get("fps")), {})
    stream_fps = first_stream.get("fps") or declared_fps
    stream_frame_count = max(
        (int(c.get("frame_count") or 0) for c in cameras.values()),
        default=0,
    )
    # A previous scan may have persisted stale metadata (for example 30 FPS
    # while all four uploaded streams are actually 25 FPS).  A real container
    # probe wins; the state file is only a fallback when probing is unavailable.
    master_fps = stream_fps or state.get("fps") or declared_fps
    master_frame_count = stream_frame_count or state.get("frame_count") or info.get("frame_count") or 0
    sensors = [str(s) for s in (info.get("sensors") or [])]
    data_root = batch_dir / "data"
    if data_root.is_dir():
        for child in data_root.iterdir():
            lower = child.name.lower()
            if child.is_dir() and any(k in lower for k in ("glove", "tactile", "sensor")):
                if child.name not in sensors:
                    sensors.append(child.name)
    return {
        "id": name,
        "name": name,
        "episode_index": _episode_index_of(name),
        "project": stored_project,
        "status": state.get("status", "completed"),
        "fps": master_fps,
        "frame_count": master_frame_count,
        "camera_names": camera_names,
        "camera_streams": cameras,
        # Physical device metadata is per episode.  camera_names and
        # camera_streams remain the actual file/source keys.
        "devices": declared_devices,
        "device_names": device_names,
        "sync": {
            "master_fps": master_fps,
            "master_frame_count": master_frame_count,
            "stream_fps": {name: value.get("fps") for name, value in cameras.items()},
            "stream_frame_counts": {name: value.get("frame_count") for name, value in cameras.items()},
            "source": "video_probe" if stream_frame_count or stream_fps else "episode_state",
        },
        "timestamp": str(info.get("timestamp") or ""),
        "created_at": state.get("created_at") or _dir_mtime(batch_dir),
        "path": str(batch_dir),
        "has_skeleton": False,
        # info.json 声明的传感器(如 left_glove/right_glove),input-sources 推导用
        "sensors": sorted(sensors),
    }


def _dir_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return utcnow_iso()


def _probe_video(path: Path) -> tuple[int, float]:
    """Probe stream frame count/FPS; uploaded global metadata is only a fallback."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
        if count > 0 or fps > 0:
            return count, fps
    except Exception:
        pass
    return 0, 0.0


def _probe_video_cached(path: Path) -> tuple[int, float]:
    """Probe each video once until the session cache is invalidated.

    The cv2 probe is the slow part (remote mount) — it runs outside
    ``_lock``; only the dict access is synchronized.
    """
    key = str(path)
    with _lock:
        cached = _video_probe_cache.get(key)
    if cached is None:
        cached = _probe_video(path)
        with _lock:
            _video_probe_cache[key] = cached
    return cached


def get_episode(episode_id: str) -> dict | None:
    """按批次名查 episode(扫描目录)。"""
    name = str(episode_id)
    return next((episode for episode in scan_sessions()
                 if str(episode.get("id")) == name), None)


# ── 审核状态 ─────────────────────────────────────────

def read_episode_state(episode_id: str) -> dict:
    """Read state once and serve subsequent API reads from memory.

    State is written only through ``write_episode_state`` in this service, so
    that writer updates this cache synchronously. Returning copies prevents a
    caller from accidentally changing the cached source of truth.
    """
    key = str(episode_id)
    with _lock:
        cached = _episode_state_cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
    state = _read_json(EPISODE_STATES_DIR / f"{key}.json", {})
    with _lock:
        _episode_state_cache[key] = copy.deepcopy(state)
    return state


def write_episode_state(episode_id: str, state: dict) -> None:
    key = str(episode_id)
    _write_json(EPISODE_STATES_DIR / f"{key}.json", state)
    # Status writes are by far the most frequent metadata mutation (queue,
    # complete, fail, approve and unapprove).  The session cache already
    # holds immutable video metadata, so update only the state-derived fields
    # in place.  If this is a new upload and the episode is not cached yet,
    # fall back to a full refresh on the next read.
    state_fields = ("status", "created_at", "project", "fps", "frame_count", "has_skeleton")
    global _deleted_episodes_cache, _deleted_episodes_cache_at, _state_write_generation
    with _lock:
        _episode_state_cache[key] = copy.deepcopy(state)
        _state_write_generation += 1  # 并发中的扫描读到旧状态则作废重扫
        _deleted_episodes_cache = None
        _deleted_episodes_cache_at = 0.0
        if _sessions_cache is None:
            return
        for episode in _sessions_cache:
            if str(episode.get("id")) != key:
                continue
            for field in state_fields:
                if field in state and state[field] is not None:
                    episode[field] = state[field]
            return
    # The batch is not in the current snapshot (normally a newly uploaded
    # session), so the next list read must discover it from disk.
    invalidate_session_cache()


def set_episode_status(episode_id: str, status: str) -> dict:
    with _lock:
        state = read_episode_state(episode_id)
        state["status"] = status
        state["updated_at"] = utcnow_iso()
        if status in ("reviewed", "approved") and not state.get("approved_at"):
            state["approved_at"] = utcnow_iso()
        if status == "to_review":
            state["approved_at"] = None
        write_episode_state(episode_id, state)
        return state


def _resolve_episode_project_root(episode_id: str,
                                  project_root: str | Path | None = None
                                  ) -> Path | None:
    """Locate the flat-layout project dir owning an episode, else None.

    1. explicit project_root (caller already resolved the episode);
    2. the episode's own state file "project" key → SESSIONS_ROOT/<project>;
    3. the in-memory scan snapshot (_sessions_cache) → dataset_root.
    Returns None when the episode is not part of any canonical project
    (legacy nested batch or pre-migration session), so the caller falls
    back to the legacy directory sweep.
    """
    name = str(episode_id)
    if project_root:
        root = Path(project_root)
        if (root / "meta" / "episodes").is_dir():
            return root
    state = _read_json(EPISODE_STATES_DIR / f"{name}.json", {})
    if isinstance(state, dict) and state.get("project"):
        root = SESSIONS_ROOT / str(state["project"])
        if (root / "meta" / "episodes").is_dir():
            return root
    if _sessions_cache:
        for ep in _sessions_cache:
            if str(ep.get("id")) == name and ep.get("dataset_root"):
                root = Path(ep["dataset_root"])
                if (root / "meta" / "episodes").is_dir():
                    return root
    return None


def delete_episode(episode_id: str, permanent: bool = False,
                   project_root: str | Path | None = None) -> None:
    """软删除(状态标记 deleted)或永久删除(删文件+状态)。

    扁平 LeRobot 布局下 permanent 经
    project_dataset.delete_project_episode 完整移除该批次的数据/视频/索引行
    并重建项目聚合;解析不到扁平项目时回退旧嵌套布局的目录扫除。
    """
    name = str(episode_id)
    with _lock:
        if permanent:
            root = _resolve_episode_project_root(name, project_root)
            if root is not None:
                # LookupError 视为幂等 no-op(重试/崩溃残留),其余异常向上
                # 抛:删除失败时状态文件仍在,回收站条目保留,可重试。
                from app.project_dataset import delete_project_episode
                try:
                    delete_project_episode(root, name)
                except LookupError:
                    pass
            else:
                # 旧嵌套布局(或整项目/session 删除):保留原目录扫除语义
                for project_dir in SESSIONS_ROOT.iterdir() if SESSIONS_ROOT.is_dir() else []:
                    if project_dir.is_dir():
                        batch = project_dir / name
                        if batch.is_dir():
                            import shutil
                            shutil.rmtree(batch, ignore_errors=True)
                batch = SESSIONS_ROOT / name
                if batch.is_dir():
                    import shutil
                    shutil.rmtree(batch, ignore_errors=True)
            _remove_json(EPISODE_STATES_DIR / f"{name}.json")
            _remove_json(ANNOTATIONS_DIR / f"{name}.json")
            _episode_state_cache.pop(name, None)
            global _deleted_episodes_cache, _deleted_episodes_cache_at
            _deleted_episodes_cache = None
            _deleted_episodes_cache_at = 0.0
            # A permanent delete removes a session directory, so the next
            # scan must rebuild its structural view.
            invalidate_session_cache()
        else:
            state = read_episode_state(name)
            state["status"] = "deleted"
            state["deleted_at"] = utcnow_iso()
            write_episode_state(name, state)


def _remove_json(path: Path) -> None:
    try:
        path.unlink()
    except Exception:
        pass


def list_deleted_episodes() -> list[dict]:
    global _deleted_episodes_cache, _deleted_episodes_cache_at
    with _lock:
        now = time.monotonic()
        if (_deleted_episodes_cache is not None
                and now - _deleted_episodes_cache_at < _DELETED_EPISODES_CACHE_TTL):
            return copy.deepcopy(_deleted_episodes_cache)
    results = []
    if EPISODE_STATES_DIR.is_dir():
        for f in sorted(EPISODE_STATES_DIR.glob("*.json")):
            state = read_episode_state(f.stem)
            if state.get("status") == "deleted":
                results.append({"id": f.stem, "deleted_at": state.get("deleted_at")})
    with _lock:
        _deleted_episodes_cache = copy.deepcopy(results)
        _deleted_episodes_cache_at = time.monotonic()
    return results


def restore_episode(episode_id: str) -> None:
    with _lock:
        state = read_episode_state(episode_id)
        state["status"] = "to_review"
        state.pop("deleted_at", None)
        write_episode_state(episode_id, state)


# ── 项目 ─────────────────────────────────────────────

def list_projects() -> list[dict]:
    return _read_json(PROJECTS_FILE, [])


def save_projects(projects: list[dict]) -> None:
    _write_json(PROJECTS_FILE, projects)


def upsert_project(project: dict) -> list[dict]:
    with _lock:
        projects = list_projects()
        found = next((p for p in projects if p["id"] == project["id"]), None)
        if found:
            found.update(project)
        else:
            projects.append(project)
        save_projects(projects)
        return projects


def delete_project(project_id: str) -> None:
    with _lock:
        projects = [p for p in list_projects() if p["id"] != project_id]
        save_projects(projects)


# ── 上传历史(统计"上传过的所有数据":软删/永久删除后记录保留)──

UPLOAD_HISTORY_FILE = STATE_ROOT / "upload_history.json"
UPLOAD_EVENTS_FILE = STATE_ROOT / "upload_events.json"


def list_upload_history() -> list[dict]:
    return _read_json(UPLOAD_HISTORY_FILE, [])


def list_upload_events() -> list[dict]:
    """Return append-only upload events.

    ``upload_history.json`` intentionally remains one row per batch because
    the project tree uses it to resurrect purged batches.  This separate file
    keeps every upload attempt, including a collector re-upload that replaces
    an existing batch.
    """
    value = _read_json(UPLOAD_EVENTS_FILE, [])
    return value if isinstance(value, list) else []


def record_upload_event(
    episode_id: str,
    project_name: str,
    uploaded_at: str,
    *,
    classification: str,
    overwrote_existing: bool,
    archive_filename: str = "",
    archive_size_bytes: int = 0,
) -> dict:
    """Append one auditable upload event and return the stored record."""
    event = {
        "event_id": str(uuid4()),
        "episode_id": str(episode_id),
        "project_name": str(project_name),
        "uploaded_at": str(uploaded_at),
        "classification": (
            "reupload" if classification == "reupload" else "new_batch"),
        "overwrote_existing": bool(overwrote_existing),
        "archive_filename": str(archive_filename or ""),
        "archive_size_bytes": max(0, int(archive_size_bytes or 0)),
        "source": "session_upload",
    }
    with _lock:
        events = list_upload_events()
        events.append(event)
        _write_json(UPLOAD_EVENTS_FILE, events)
    return event


def upsert_upload_record(episode_id: str, project_name: str, uploaded_at: str) -> None:
    """记录一次上传(同批次重新上传 → 覆盖并复活)。"""
    with _lock:
        items = list_upload_history()
        found = next((x for x in items if x.get("episode_id") == episode_id), None)
        if found:
            found.update({"project_name": project_name, "uploaded_at": uploaded_at})
            found.pop("purged_at", None)
        else:
            items.append({
                "episode_id": episode_id,
                "project_name": project_name,
                "uploaded_at": uploaded_at,
                "purged_at": None,
            })
        _write_json(UPLOAD_HISTORY_FILE, items)


def mark_upload_purged(episode_id: str, purged_at: str | None = None) -> None:
    """永久删除:标记记录(统计不丢,历史可查)。"""
    from datetime import datetime, timezone
    with _lock:
        items = list_upload_history()
        found = next((x for x in items if x.get("episode_id") == episode_id), None)
        if found:
            found["purged_at"] = purged_at or datetime.now(timezone.utc).isoformat()
            _write_json(UPLOAD_HISTORY_FILE, items)


def rename_project_references(old_name: str, new_name: str) -> dict[str, int]:
    """同步项目改名后的持久化名称引用,但不改变任何稳定 ID。

    项目目录和 episode state 由项目 API 处理;这里负责那些不会随目录
    扫描自动修正的历史索引。工作流 ID、批次/运行 ID、标注 ID 都保持
    不变。运行记录中的 project_name 仅作为兼容快照,一并更新可避免
    异常/运行列表出现同一项目两个名称。
    """
    old_name = str(old_name or "")
    new_name = str(new_name or "")
    if not old_name or not new_name or old_name == new_name:
        return {"upload_history": 0, "upload_events": 0, "exceptions": 0, "runs": 0}

    changed = {"upload_history": 0, "upload_events": 0, "exceptions": 0, "runs": 0}
    with _lock:
        history = list_upload_history()
        for item in history:
            if item.get("project_name") == old_name:
                item["project_name"] = new_name
                changed["upload_history"] += 1
        if changed["upload_history"]:
            _write_json(UPLOAD_HISTORY_FILE, history)

        events = list_upload_events()
        for item in events:
            if item.get("project_name") == old_name:
                item["project_name"] = new_name
                changed["upload_events"] += 1
        if changed["upload_events"]:
            _write_json(UPLOAD_EVENTS_FILE, events)

        exceptions = list_exceptions()
        for item in exceptions:
            if item.get("project_name") == old_name:
                item["project_name"] = new_name
                changed["exceptions"] += 1
        if changed["exceptions"]:
            _write_json(EXCEPTIONS_FILE, exceptions)

        # 兼容早期 run 快照中冗余保存的 project_name。没有该字段的
        # 旧运行记录不需要改写; workflow_name 则不改,保留历史审计语义。
        runs_dir = globals().get("RUNS_DIR", STATE_ROOT / "runs")
        for path in Path(runs_dir).glob("*.json"):
            run = _read_json(path, {})
            if not isinstance(run, dict) or run.get("project_name") != old_name:
                continue
            run["project_name"] = new_name
            _write_json(path, run)
            changed["runs"] += 1
    return changed


# ── 上传输入异常(落盘;run_failed 由 API 聚合 runs 生成)──

def list_exceptions() -> list[dict]:
    return _read_json(EXCEPTIONS_FILE, [])


def add_exception(rec: dict) -> None:
    """追加一条异常记录;同 (episode_id, workflow_id) 去重(重复上传覆盖)。"""
    with _lock:
        items = list_exceptions()
        found = next(
            (x for x in items
             if x.get("episode_id") == rec.get("episode_id")
             and x.get("workflow_id") == rec.get("workflow_id")),
            None,
        )
        if found:
            for key in ("created_at", "message", "wanted", "matched", "missing", "available"):
                if key in rec:
                    found[key] = rec[key]
        else:
            items.append(rec)
        _write_json(EXCEPTIONS_FILE, items)


def delete_exception(exception_id: str) -> bool:
    """删除一条落盘异常;不存在返回 False。"""
    with _lock:
        items = list_exceptions()
        kept = [x for x in items if x.get("id") != exception_id]
        if len(kept) == len(items):
            return False
        _write_json(EXCEPTIONS_FILE, kept)
        return True


def clear_exceptions() -> int:
    with _lock:
        n = len(list_exceptions())
        _write_json(EXCEPTIONS_FILE, [])
        return n


# ── 传感器显示策略(工作流驱动)────────────────────────

def project_has_glove_sensor(project_name: str) -> bool:
    """项目绑定的任一工作流含 glove_sensor 输入节点 → True。

    手套传感器显示/匹配由工作流声明驱动(处理链声明了手套设备才
    显示手套数据);骨骼识别参数(hand_pose 等)列名不含 glove/sensor,
    天然不会作为手套传感器。
    """
    # episode.project 是 sanitize 后的目录名(空格→_、去不安全字符),
    # projects.json 存原始名 —— 两者都要比,否则特殊字符项目名匹配失败
    try:
        from app.storage import sanitize_task_name
    except Exception:
        sanitize_task_name = None
    for p in list_projects():
        pname = p.get("name") or ""
        if pname != project_name:
            if not (sanitize_task_name and sanitize_task_name(pname) == project_name):
                continue
        ids = p.get("workflow_ids") or []
        if not isinstance(ids, list):
            ids = [ids] if ids else []
        ids = [str(i) for i in ids if i]
        if not ids and p.get("workflow_id"):
            ids = [str(p["workflow_id"])]
        for wf_id in ids:
            wf = get_workflow(wf_id)
            if not wf:
                continue
            for node in (wf.get("graph") or {}).get("nodes", []):
                if (node.get("data") or {}).get("nodeType") == "glove_sensor":
                    return True
    return False


def _workflow_has_node(workflow: dict | None, node_type: str) -> bool:
    """Return whether a saved workflow graph declares ``node_type``."""
    if not isinstance(workflow, dict):
        return False
    for node in (workflow.get("graph") or {}).get("nodes", []):
        data = node.get("data") or {}
        if canonical_node_type(data.get("nodeType")) == canonical_node_type(node_type) \
                or canonical_node_type(node.get("type")) == canonical_node_type(node_type):
            return True
    return False


def _run_has_node(run: dict | None, node_type: str) -> bool:
    """Check both current node states and the persisted run graph.

    The graph is intentionally checked as a historical fallback: projects or
    workflow definitions can be renamed/deleted after a run has produced a
    valid data artifact.
    """
    if not isinstance(run, dict):
        return False
    for state in (run.get("node_states") or {}).values():
        if (isinstance(state, dict)
                and canonical_node_type(state.get("type")) == canonical_node_type(node_type)):
            return True
    for node in (run.get("graph") or {}).get("nodes", []):
        data = node.get("data") or {}
        if canonical_node_type(data.get("nodeType")) == canonical_node_type(node_type) \
                or canonical_node_type(node.get("type")) == canonical_node_type(node_type):
            return True
    return False


def episode_has_glove_sensor(episode: dict | str | None) -> bool:
    """Resolve whether an episode is allowed to expose glove sensor data.

    The current project workflow remains authoritative when it exists.  For
    historical batches whose project/workflow was renamed or removed, use the
    episode's declared sensors or its persisted workflow run as evidence.  The
    ingestion layer still verifies that pressure columns contain non-zero data
    before creating UI sources, so metadata alone cannot create a fake tile.
    """
    if isinstance(episode, str):
        episode = get_episode(episode)
    if not isinstance(episode, dict):
        return False

    project_name = str(episode.get("project") or "")
    matched_project = None
    try:
        from app.storage import sanitize_task_name
    except Exception:
        sanitize_task_name = None
    for project in list_projects():
        name = str(project.get("name") or "")
        if name == project_name or (
            sanitize_task_name and sanitize_task_name(name) == project_name
        ):
            matched_project = project
            break

    if matched_project is not None:
        ids = matched_project.get("workflow_ids") or []
        if not isinstance(ids, list):
            ids = [ids] if ids else []
        if not ids and matched_project.get("workflow_id"):
            ids = [matched_project["workflow_id"]]
        workflows = [get_workflow(str(wid)) for wid in ids if wid]
        workflows = [wf for wf in workflows if wf]
        # A valid current workflow explicitly controls visibility.
        if workflows:
            return any(_workflow_has_node(wf, "glove_sensor") for wf in workflows)
        # Dangling workflow IDs are treated like a deleted workflow and fall
        # through to historical evidence below.

    sensors = episode.get("sensors") or []
    if any(any(token in str(sensor).lower() for token in ("glove", "sensor", "tactile"))
           for sensor in sensors):
        return True

    episode_id = str(episode.get("id") or episode.get("name") or "")
    if episode_id:
        runs = [run for run in list_runs()
                if str(run.get("episode_id") or "") == episode_id]
        runs.sort(key=lambda run: str(run.get("created_at") or ""), reverse=True)
        if any(_run_has_node(run, "glove_sensor") for run in runs):
            return True
    return False


# ── 工作流 ───────────────────────────────────────────

def list_workflows() -> list[dict]:
    global _workflows_cache, _workflows_cache_at
    now = time.monotonic()
    if (_workflows_cache is not None
            and now - _workflows_cache_at < _STATE_LIST_CACHE_TTL):
        return copy.deepcopy(_workflows_cache)
    value = _read_json(WORKFLOWS_FILE, [])
    _workflows_cache = value if isinstance(value, list) else []
    _workflows_cache_at = now
    return copy.deepcopy(_workflows_cache)


def save_workflows(workflows: list[dict]) -> None:
    global _workflows_cache, _workflows_cache_at
    _write_json(WORKFLOWS_FILE, workflows)
    _workflows_cache = None
    _workflows_cache_at = 0.0


def get_workflow(workflow_id: str) -> dict | None:
    return next((w for w in list_workflows() if w["id"] == workflow_id), None)


def upsert_workflow(workflow: dict) -> None:
    with _lock:
        workflows = list_workflows()
        found = next((w for w in workflows if w["id"] == workflow["id"]), None)
        if found:
            found.update(workflow)
        else:
            workflows.append(workflow)
        save_workflows(workflows)


def delete_workflow(workflow_id: str) -> None:
    """删除工作流定义,并同步清理所有项目对该工作流的绑定引用。

    若只删定义不清理绑定,项目会留下悬空 workflow_ids —— 卡片渲染时
    ids/names 错位并显示假名(曾出现 'Workflow')。workflow_bindings
    (项目级设备命名映射)同样按工作流存储,一并移除。
    """
    with _lock:
        workflows = [w for w in list_workflows() if w["id"] != workflow_id]
        save_workflows(workflows)
        projects = list_projects()
        changed = False
        for p in projects:
            ids = p.get("workflow_ids") or []
            if not isinstance(ids, list):
                ids = [ids] if ids else []
            if workflow_id in ids:
                p["workflow_ids"] = [i for i in ids if i != workflow_id]
                changed = True
            if str(p.get("workflow_id") or "") == str(workflow_id):
                p["workflow_id"] = None
                changed = True
            bindings = p.get("workflow_bindings") or {}
            if isinstance(bindings, dict) and workflow_id in bindings:
                bindings.pop(workflow_id, None)
                p["workflow_bindings"] = bindings
                changed = True
            workflow_inputs = p.get("workflow_inputs") or {}
            if isinstance(workflow_inputs, dict) and workflow_id in workflow_inputs:
                workflow_inputs.pop(workflow_id, None)
                p["workflow_inputs"] = workflow_inputs
                changed = True
        if changed:
            save_projects(projects)


# ── 标注 ─────────────────────────────────────────────

def list_annotations(episode_id: str) -> list[dict]:
    return _read_json(ANNOTATIONS_DIR / f"{episode_id}.json", [])


def save_annotations(episode_id: str, annotations: list[dict]) -> None:
    _write_json(ANNOTATIONS_DIR / f"{episode_id}.json", annotations)


def mutate_annotations(
    episode_id: str,
    mutator: Callable[[list[dict]], _MutationResult],
) -> _MutationResult:
    """Atomically read, mutate, and persist one episode's annotations.

    Annotation files are whole-document JSON files. Keeping the complete
    read-modify-write cycle under the same lock prevents two browser requests
    from silently overwriting each other.
    """
    with _lock:
        path = ANNOTATIONS_DIR / f"{episode_id}.json"
        annotations = _read_json(path, [])
        if not isinstance(annotations, list):
            annotations = []
        result = mutator(annotations)
        _write_json(path, annotations)
        return result


def mutate_annotation_by_id(
    annotation_id: str,
    mutator: Callable[[list[dict], int], _MutationResult],
) -> tuple[str, _MutationResult] | None:
    """Atomically mutate an annotation while locating its episode file."""
    with _lock:
        if not ANNOTATIONS_DIR.is_dir():
            return None
        for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
            annotations = _read_json(path, [])
            if not isinstance(annotations, list):
                continue
            index = next(
                (i for i, item in enumerate(annotations)
                 if item.get("id") == annotation_id),
                None,
            )
            if index is None:
                continue
            result = mutator(annotations, index)
            _write_json(path, annotations)
            return path.stem, result
    return None


# ── 工作流运行队列 ───────────────────────────────────

RUNS_DIR = STATE_ROOT / "runs"


def list_runs() -> list[dict]:
    global _runs_cache, _runs_cache_at
    now = time.monotonic()
    if (_runs_cache is not None
            and now - _runs_cache_at < _STATE_LIST_CACHE_TTL):
        return copy.deepcopy(_runs_cache)
    if not RUNS_DIR.is_dir():
        _runs_cache = []
        _runs_cache_at = now
        return []
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        run = _read_json(f, None)
        if isinstance(run, dict):
            runs.append(run)
    _runs_cache = runs
    _runs_cache_at = time.monotonic()
    return copy.deepcopy(runs)


def get_run(run_id: str) -> dict | None:
    return _read_json(RUNS_DIR / f"{run_id}.json", None)


def save_run(run: dict) -> None:
    global _runs_cache, _runs_cache_at
    with _lock:
        _write_json(RUNS_DIR / f"{run.get('id')}.json", run)
        _runs_cache = None
        _runs_cache_at = 0.0


def update_run_if_owned(
    run_id: str,
    worker_id: str,
    lease_token: str,
    mutator: Callable[[dict], None],
) -> dict | None:
    """Atomically mutate a running job only for its current lease owner."""
    global _runs_cache, _runs_cache_at
    with _lock:
        path = RUNS_DIR / f"{run_id}.json"
        run = _read_json(path, None)
        if not isinstance(run, dict):
            return None
        if (run.get("status") != "running"
                or run.get("worker_id") != worker_id
                or run.get("lease_token") != lease_token):
            return None
        mutator(run)
        _write_json(path, run)
        _runs_cache = None
        _runs_cache_at = 0.0
        return run


def save_run_if_absent(
    run: dict,
    allow_completed_rerun: bool = False,
    supersede_active: bool = False,
) -> tuple[dict, bool]:
    """按(工作流,批次,工作流版本)原子去重后保存运行记录。

    返回 ``(existing_or_new_run, created)``。queued/running/completed 的
    同版本运行不会重复入队；手动 Reprocess 可通过
    ``allow_completed_rerun`` 重新创建 completed 运行；failed 允许重新入队。
    当数据被重传或用户明确重处理时，``supersede_active`` 会先让该批次
    同工作流的任何 queued/running 任务失效(包括旧版本)，避免旧 Worker
    把旧数据结果写回新批次或覆盖新版本产物。
    旧版本运行没有
    ``workflow_revision`` 时，queued/running 仍视为占用，completed 则
    允许第一次迁移到带版本的幂等记录。
    """
    global _runs_cache, _runs_cache_at
    with _lock:
        revision = run.get("workflow_revision")
        for existing in list_runs():
            if (existing.get("workflow_id") != run.get("workflow_id")
                    or existing.get("episode_id") != run.get("episode_id")):
                continue
            same_revision = existing.get("workflow_revision") == revision
            legacy_active = (not existing.get("workflow_revision")
                             and existing.get("status") in ("queued", "running"))
            if (supersede_active
                    and existing.get("status") in ("queued", "running")):
                existing["status"] = "superseded"
                existing["finished_at"] = utcnow_iso()
                existing["error_log"] = "Superseded by a newer upload or reprocess request"
                _write_json(RUNS_DIR / f"{existing.get('id')}.json", existing)
                _runs_cache = None
                _runs_cache_at = 0.0
                continue
            if same_revision or legacy_active:
                if existing.get("status") in ("queued", "running"):
                    return existing, False
                if existing.get("status") == "completed" and not allow_completed_rerun:
                    return existing, False
        _write_json(RUNS_DIR / f"{run.get('id')}.json", run)
        _runs_cache = None
        _runs_cache_at = 0.0
        return run, True


def delete_run(run_id: str) -> None:
    global _runs_cache, _runs_cache_at
    _remove_json(RUNS_DIR / f"{run_id}.json")
    _runs_cache = None
    _runs_cache_at = 0.0
