"""File storage utilities — streaming writes, SHA256, cleanup."""

import os
import json
import hashlib
import tempfile
import shutil
import asyncio
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from fastapi import UploadFile, HTTPException
from uuid import uuid4

from app.config import settings
from app.remote_storage import remote_storage

ALLOWED_EXTENSIONS = settings.ALLOWED_VIDEO_EXTENSIONS | settings.ALLOWED_IMAGE_EXTENSIONS


def _rel_path(subdir: str, filename: str) -> str:
    """Build a date-based relative path: <subdir>/YYYY/MM/DD/<filename>"""
    now = datetime.now(timezone.utc)
    date_path = now.strftime("%Y/%m/%d")
    return f"{subdir}/{date_path}/{filename}"


def _abs_path(rel_path: str) -> Path:
    return settings.storage_root / rel_path


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_extension(filename: str) -> str:
    """Validate and return the file extension (lowercase, no dot)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if not ext:
        raise HTTPException(status_code=400, detail="File has no extension")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Extension .{ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}")
    return ext


class FileSaveResult:
    def __init__(self, relative_path: str, size: int, sha256: str):
        self.relative_path = relative_path
        self.size = size
        self.sha256 = sha256


async def save_upload(file: UploadFile, subdir: str = "raw_images") -> FileSaveResult:
    """Stream an UploadFile to disk, compute SHA-256, return result."""
    ext = _validate_extension(file.filename or "unknown.bin")
    unique_name = f"{uuid4().hex}.{ext}"
    rel_path = _rel_path(subdir, unique_name)
    abs_path = _abs_path(rel_path)
    _ensure_dir(abs_path)

    sha = hashlib.sha256()
    size = 0

    # Write to temp file first, then atomic rename
    tmp = tempfile.NamedTemporaryFile(dir=abs_path.parent, delete=False)
    try:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            sha.update(chunk)
            tmp.write(chunk)
            size += len(chunk)
            if size > settings.MAX_UPLOAD_SIZE:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(status_code=413, detail="File too large")
    except HTTPException:
        raise
    except Exception as e:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    finally:
        tmp.close()

    os.replace(tmp.name, str(abs_path))
    remote = remote_storage()
    if remote is not None:
        await asyncio.to_thread(remote.upload_file, abs_path, rel_path)
    return FileSaveResult(relative_path=rel_path, size=size, sha256=sha.hexdigest())


def delete_file(relative_path: str) -> None:
    """Delete a file by its relative path."""
    abs_path = _abs_path(relative_path)
    if abs_path.exists():
        abs_path.unlink()
    remote = remote_storage()
    if remote is not None:
        remote.remove_file(relative_path)


def sync_tree_to_remote(local_root: Path, remote_relative: str | Path) -> None:
    """Mirror a local tree to SFTP; no-op for local storage."""
    remote = remote_storage()
    if remote is not None:
        remote.upload_tree(local_root, remote_relative)


def sync_file_to_remote(local_path: Path, remote_relative: str | Path) -> None:
    remote = remote_storage()
    if remote is not None:
        remote.upload_file(local_path, remote_relative)


async def sync_file_to_remote_async(local_path: Path, remote_relative: str | Path) -> None:
    await asyncio.to_thread(sync_file_to_remote, local_path, remote_relative)


async def ensure_local_file(relative_path: str | Path) -> Path | None:
    """Materialize one remote file into the local cache when needed."""
    raw = str(relative_path).replace("\\", "/")
    safe = PurePosixPath(raw)
    if safe.is_absolute() or ".." in safe.parts:
        raise ValueError(f"Unsafe storage path: {relative_path}")
    local_path = settings.storage_root / Path(*safe.parts)
    if local_path.exists():
        return local_path
    remote = remote_storage()
    if remote is None:
        return None
    await asyncio.to_thread(remote.download_file, relative_path, local_path)
    return local_path if local_path.exists() else None


async def sync_tree_to_remote_async(local_root: Path, remote_relative: str | Path) -> None:
    await asyncio.to_thread(sync_tree_to_remote, local_root, remote_relative)


def delete_tree(remote_relative: str | Path, local_root: Path | None = None) -> None:
    """Delete a session/result tree from cache and authoritative storage."""
    if local_root is not None and local_root.exists():
        shutil.rmtree(local_root, ignore_errors=True)
    remote = remote_storage()
    if remote is not None:
        remote.remove_tree(remote_relative)


def storage_ok() -> bool:
    """Check that the storage directory is writable."""
    try:
        remote = remote_storage()
        if remote is not None:
            return remote.check()
        probe = settings.storage_root / ".health_probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def scan_existing_mp4(episode_id: str, camera: str) -> str | None:
    """Find an existing MP4 video file for an episode/camera.

    Searches the raw_videos directory by episode_id prefix.
    """
    raw_dir = settings.storage_root / "videos"
    # Look for compiled MP4 or individual frames
    mp4_file = raw_dir / f"{episode_id}_{camera}.mp4"
    if mp4_file.exists():
        return str(mp4_file)

    # Check date-organized directories
    for root, _, files in os.walk(str(raw_dir)):
        for f in files:
            if f.startswith(episode_id[:8]) and camera in f and f.endswith(".mp4"):
                return str(Path(root) / f)
    return None


def find_session_dir(session_id: str, db=None) -> Path | None:
    """Find session directory under sessions/.

    - New: looks up Session.original_archive (e.g. 'test00/test00_20260806_102355')
    - Current layout: sessions/<project>/<task>/ — walk two levels
    - Old: walks for directory named {session_id}
    """
    sessions_root = settings.storage_root / "sessions"
    if not sessions_root.exists():
        return None
    sid = str(session_id)

    # Walk: project/task-organized structure
    for project_dir in sessions_root.iterdir():
        if project_dir.is_dir():
            candidate = project_dir / sid
            if candidate.is_dir():
                return candidate
            # Project-level LeRobot storage keeps all episodes directly under
            # sessions/<project>/{data,meta,videos}; resolve the episode ID
            # from the project metadata instead of expecting a batch folder.
            if (project_dir / "meta" / "episodes").is_dir():
                try:
                    from app.project_dataset import project_episode_rows
                    for row in project_episode_rows(project_dir):
                        if (str(row.get("episode_id") or row.get("source_batch") or "") == sid
                                or str(row.get("episode_index") or "") == sid):
                            return project_dir
                except (OSError, ValueError, TypeError):
                    pass
    # Old flat structure
    candidate = sessions_root / sid
    if candidate.is_dir():
        return candidate
    return None


# ── Session dir cache (avoids repeated filesystem walks) ──
_session_dir_cache: dict[str, Path | None] = {}
# None 结果的缓存时间戳 —— SFTP 下载可能稍后才完成(或暂时失败),
# 若把 None 永久缓存,下载完成后也永远读不到该 session。
_session_dir_cache_none_at: dict[str, float] = {}
_SESSION_DIR_NONE_TTL = 30.0


def _session_cache_ready(path: Path) -> bool:
    """A session cache is usable only when it contains downloaded files."""
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


async def find_session_dir_async(session_id: str, db) -> Path | None:
    """Async version: looks up Session.original_archive for the path.

    Result is cached — session directories don't move after creation.
    Negative (None) results expire after a short TTL so a later SFTP
    download of the session tree is picked up without a backend restart.
    """
    sid = str(session_id)
    if sid in _session_dir_cache:
        cached = _session_dir_cache[sid]
        if cached is not None:
            if _session_cache_ready(cached):
                return cached
            del _session_dir_cache[sid]  # stale entry
        else:
            # 负面缓存:TTL 内直接返回 None,超时后重新查找
            import time as _time
            cached_at = _session_dir_cache_none_at.get(sid, 0.0)
            if _time.monotonic() - cached_at < _SESSION_DIR_NONE_TTL:
                return None
            del _session_dir_cache[sid]
            _session_dir_cache_none_at.pop(sid, None)

    from uuid import UUID
    from app.models import Session
    sessions_root = settings.storage_root / "sessions"

    # Try DB lookup for stored path
    original_archive = None
    try:
        sess = await db.get(Session, UUID(sid))
        if sess:
            original_archive = sess.original_archive
            if original_archive:
                candidate = sessions_root / original_archive
                if _session_cache_ready(candidate):
                    _session_dir_cache[sid] = candidate
                    return candidate
    except Exception:
        pass

    remote = remote_storage()
    if remote is not None and original_archive:
        candidate = sessions_root / original_archive
        await asyncio.to_thread(
            remote.download_tree,
            Path("sessions") / original_archive,
            candidate,
        )
        if candidate.is_dir():
            _session_dir_cache[sid] = candidate
            return candidate

    # Fallback: walk filesystem (project/task two-level layout)
    for project_dir in sessions_root.iterdir():
        if project_dir.is_dir():
            candidate = project_dir / sid
            if _session_cache_ready(candidate):
                _session_dir_cache[sid] = candidate
                return candidate
            if (project_dir / "meta" / "episodes").is_dir():
                try:
                    from app.project_dataset import project_episode_rows
                    for row in project_episode_rows(project_dir):
                        if (str(row.get("episode_id") or row.get("source_batch") or "") == sid
                                or str(row.get("episode_index") or "") == sid):
                            _session_dir_cache[sid] = project_dir
                            return project_dir
                except (OSError, ValueError, TypeError):
                    pass
    candidate = sessions_root / sid
    if _session_cache_ready(candidate):
        _session_dir_cache[sid] = candidate
        return candidate
    import time as _time
    _session_dir_cache[sid] = None
    _session_dir_cache_none_at[sid] = _time.monotonic()
    return None


def sanitize_task_name(name: str) -> str:
    """Sanitize task name for folder use. Preserves Chinese, replaces unsafe chars."""
    unsafe = r'[\\/:*?"<>|]'
    import re
    name = re.sub(unsafe, '', name)
    name = name.replace(' ', '_')
    name = name.strip('. ')
    return name or 'default_recording'


def parse_zip_filename(filename: str) -> dict:
    """Parse zip filename like 'session_20260805_192922.zip'.

    Returns dict with:
      - prefix: task prefix ('session')
      - basename: filename without .zip ('session_20260805_192922')
      - date_str: '2026-08-05'
      - time_str: '19:29:22'
      - timestamp: '2026/8/5 19:29:22' (display format)
    """
    import re
    name = filename.rsplit('.', 1)[0]  # remove extension
    result = {'prefix': name, 'basename': name, 'timestamp': name}

    # Match pattern: {prefix}_{YYYYMMDD}_{HHMMSS}
    m = re.match(r'^(.+?)_(\d{8})_(\d{6})$', name)
    if m:
        prefix = m.group(1)
        date_part = m.group(2)  # 20260805
        time_part = m.group(3)  # 192922
        result['prefix'] = prefix
        result['basename'] = name
        try:
            y, mo, d = date_part[:4], date_part[4:6], date_part[6:8]
            h, mi, s = time_part[:2], time_part[2:4], time_part[4:6]
            result['date_str'] = f'{y}-{mo}-{d}'
            result['time_str'] = f'{h}:{mi}:{s}'
            result['timestamp'] = f'{int(y)}/{int(mo)}/{int(d)} {h}:{mi}:{s}'
        except (ValueError, IndexError):
            pass
    return result
