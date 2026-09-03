"""Durable single-flight upload ingestion queue.

The HTTP request only receives the archive into a local spool and records a
small job document.  Decompression, remote-storage writes, video probing and
workflow dispatch run in one dedicated thread so they cannot block Uvicorn's
event loop or the collector's task polling endpoint.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings


_JOBS_ROOT = settings.storage_root / "state" / "upload_jobs"
_jobs_lock = threading.RLock()
_pending: queue.Queue[str] = queue.Queue()
_stop_event = threading.Event()
_worker_thread: threading.Thread | None = None
_DUPLICATE_WINDOW_SECONDS = 15 * 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(upload_id: str) -> Path:
    return _JOBS_ROOT / f"{str(upload_id)}.json"


def _read_job(upload_id: str) -> dict | None:
    try:
        value = json.loads(_job_path(upload_id).read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _write_job(job: dict) -> None:
    _JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    path = _job_path(str(job["upload_id"]))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def create_job(filename: str, name: str, project_id: str) -> dict:
    upload_id = uuid4().hex
    spool = settings.upload_staging_root / f"{upload_id}.upload"
    job = {
        "upload_id": upload_id,
        "filename": str(filename),
        "name": str(name or ""),
        "project_id": str(project_id or ""),
        "staged_path": str(spool),
        "status": "receiving",
        "size_bytes": 0,
        "sha256": "",
        "created_at": _now(),
        "received_at": None,
        "started_at": None,
        "finished_at": None,
        "session_id": None,
        "result": None,
        "error": None,
        "duplicate_of": None,
    }
    with _jobs_lock:
        _write_job(job)
    return job


def set_received(upload_id: str, size_bytes: int, sha256: str) -> dict:
    with _jobs_lock:
        job = _read_job(upload_id)
        if job is None:
            raise FileNotFoundError(f"Upload job not found: {upload_id}")
        job.update({
            "status": "received",
            "size_bytes": max(0, int(size_bytes)),
            "sha256": str(sha256 or ""),
            "received_at": _now(),
        })
        _write_job(job)
        return job


def enqueue(upload_id: str) -> dict:
    with _jobs_lock:
        job = _read_job(upload_id)
        if job is None:
            raise FileNotFoundError(f"Upload job not found: {upload_id}")
        job.update({
            "status": "queued", "error": None,
            "started_at": None, "finished_at": None,
        })
        _write_job(job)
    _pending.put(upload_id)
    start()
    return job


def fail(upload_id: str, error: str) -> None:
    with _jobs_lock:
        job = _read_job(upload_id)
        if job is None:
            return
        job.update({"status": "failed", "finished_at": _now(),
                    "error": str(error)[:4000]})
        _write_job(job)


def mark_deduplicated(upload_id: str, duplicate_of: str) -> dict | None:
    with _jobs_lock:
        job = _read_job(upload_id)
        if job is None:
            return None
        job.update({"status": "deduplicated", "finished_at": _now(),
                    "duplicate_of": str(duplicate_of)})
        _write_job(job)
        return job


def _job_age_seconds(job: dict) -> float:
    try:
        created = datetime.fromisoformat(str(job.get("created_at")))
        return max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
    except Exception:
        return 0.0


def find_active_duplicate(sha256: str) -> dict | None:
    """Find a retry of the same archive while it is still recent/in flight."""
    if not sha256:
        return None
    with _jobs_lock:
        for path in sorted(_JOBS_ROOT.glob("*.json")) if _JOBS_ROOT.is_dir() else []:
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(job, dict) or job.get("sha256") != sha256:
                continue
            status = str(job.get("status") or "")
            if status in {"queued", "processing"}:
                return job
            if status == "completed" and _job_age_seconds(job) <= _DUPLICATE_WINDOW_SECONDS:
                # A completed upload is only a safe deduplication target while
                # its committed episode still exists.  The user may have
                # deleted/purged it after the upload; accepting a resend as a
                # duplicate in that case would silently lose the new archive.
                session_id = str(job.get("session_id") or "")
                if session_id:
                    try:
                        from app.localstore import get_episode
                        if get_episode(session_id) is not None:
                            return job
                    except Exception:
                        # If the storage check is unavailable, do not dedupe a
                        # new upload based on an unverified history record.
                        pass
    return None


def get_job(upload_id: str) -> dict | None:
    with _jobs_lock:
        return _read_job(upload_id)


def public_job(job: dict) -> dict:
    """Return status without exposing the local spool path or hash."""
    status = str(job.get("status") or "")
    staged = Path(str(job.get("staged_path") or ""))
    result = {
        key: job.get(key)
        for key in (
            "upload_id", "filename", "status", "size_bytes", "created_at",
            "received_at", "started_at", "finished_at", "session_id",
            "result", "error", "duplicate_of",
        )
    }
    result["recovery_available"] = status == "failed" and staged.is_file()
    return result


class _StagedUpload:
    """Small UploadFile-compatible adapter used by the legacy processor."""

    def __init__(self, path: Path, filename: str):
        self.filename = filename
        self._file = path.open("rb")

    async def read(self, size: int = -1) -> bytes:
        return self._file.read(size)

    def close(self) -> None:
        self._file.close()


def _process_one(upload_id: str) -> None:
    job = get_job(upload_id)
    if job is None or job.get("status") != "queued":
        return
    staged_path = Path(str(job.get("staged_path") or ""))
    if not staged_path.is_file():
        fail(upload_id, "Local upload spool is missing")
        return

    with _jobs_lock:
        latest = _read_job(upload_id)
        if latest is None or latest.get("status") != "queued":
            return
        latest.update({"status": "processing", "started_at": _now()})
        _write_job(latest)

    upload = _StagedUpload(staged_path, str(job.get("filename") or staged_path.name))
    try:
        # Import lazily: session.py imports this queue from the HTTP endpoint.
        from app.routes.session import _process_upload_job

        response = asyncio.run(_process_upload_job(
            upload,
            name=str(job.get("name") or ""),
            project_id=str(job.get("project_id") or ""),
            staged_path=staged_path,
        ))
        body = getattr(response, "body", b"")
        result: dict[str, Any] = json.loads(body.decode("utf-8")) if body else {}
        with _jobs_lock:
            latest = _read_job(upload_id) or job
            latest.update({
                "status": "completed",
                "finished_at": _now(),
                "session_id": result.get("session_id"),
                "result": result,
                "error": None,
            })
            _write_job(latest)
        # Refresh the stale-but-fast task snapshot outside Uvicorn's event loop.
        try:
            from app import localstore
            localstore.scan_sessions()
        except Exception as exc:
            print(f"[UploadQueue] Task snapshot refresh skipped: {exc}")
        print(f"[UploadQueue] Completed upload {upload_id}: {result.get('session_id')}")
    except Exception as exc:
        fail(upload_id, str(exc))
        print(f"[UploadQueue] Upload {upload_id} failed: {exc}")
    finally:
        upload.close()
        # A successful import has already been committed to the authoritative
        # storage and can release the local spool.  Keep failed jobs locally so
        # a transient NAS/SSHFS outage never destroys the only received copy.
        latest = get_job(upload_id)
        if latest and latest.get("status") in {"completed", "deduplicated"}:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                pass


def _run() -> None:
    while not _stop_event.is_set():
        try:
            upload_id = _pending.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            _process_one(upload_id)
        finally:
            _pending.task_done()


def _recover_jobs() -> None:
    """Resume queued jobs after a normal API process restart."""
    if not _JOBS_ROOT.is_dir():
        return
    for path in sorted(_JOBS_ROOT.glob("*.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "")
        upload_id = str(job.get("upload_id") or path.stem)
        staged = Path(str(job.get("staged_path") or ""))
        if status == "processing":
            # The old API process is gone; a durable spool can safely restart.
            job["status"] = "queued"
            job["error"] = None
            _write_job(job)
            status = "queued"
        if status == "queued" and staged.is_file():
            _pending.put(upload_id)
        elif status == "receiving":
            job.update({"status": "failed", "finished_at": _now(),
                        "error": "Upload interrupted before it was received"})
            _write_job(job)
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass


def start() -> None:
    global _worker_thread
    with _jobs_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _stop_event.clear()
        _JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        _recover_jobs()
        _worker_thread = threading.Thread(
            target=_run, name="egodata-upload-ingest", daemon=True,
        )
        _worker_thread.start()


def stop() -> None:
    global _worker_thread
    _stop_event.set()
    thread = _worker_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)
    _worker_thread = None
