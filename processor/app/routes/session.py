"""Session upload — normalize every archive into a canonical project dataset."""

import asyncio
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.config import settings
from app.security import verify_api_key

router = APIRouter(prefix="/api/v1", tags=["session"])


def _copy_upload_file(source, target: Path) -> tuple[int, str]:
    """Copy the already-received multipart file to local durable storage."""
    source.seek(0)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with target.open("wb") as output:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


@router.post("/session/upload")
async def session_upload(
    _: str = Depends(verify_api_key),
    file: UploadFile = File(...),
    name: str = Form(""),
    project_id: str = Form(""),
):
    """Accept an archive quickly; process it in the dedicated ingest queue.

    ``201`` is retained for old collectors that only accept the historical
    success status.  ``status=queued`` and ``upload_id`` tell upgraded clients
    that storage normalization/workflow dispatch continues in the background.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Never let an archive filename escape the local spool directory.
    filename = Path(str(file.filename).replace("\\", "/")).name
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name")

    from app import upload_queue
    job = await asyncio.to_thread(
        upload_queue.create_job, filename, name, project_id,
    )
    staged_path = Path(str(job["staged_path"]))
    try:
        # The multipart body is already received by FastAPI at this point.
        # Copying it in one thread keeps the event loop responsive while the
        # archive is spooled locally rather than onto SSHFS.
        size_bytes, sha256 = await asyncio.to_thread(
            _copy_upload_file, file.file, staged_path,
        )
        if size_bytes <= 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if size_bytes > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
        await asyncio.to_thread(
            upload_queue.set_received, job["upload_id"], size_bytes, sha256,
        )

        duplicate = await asyncio.to_thread(
            upload_queue.find_active_duplicate, sha256,
        )
        if duplicate is not None:
            await asyncio.to_thread(
                upload_queue.mark_deduplicated,
                job["upload_id"], duplicate["upload_id"],
            )
            staged_path.unlink(missing_ok=True)
            result = upload_queue.public_job(duplicate)
            result["deduplicated"] = True
            return JSONResponse(status_code=201, content=result)

        await asyncio.to_thread(upload_queue.enqueue, job["upload_id"])
    except HTTPException as exc:
        await asyncio.to_thread(
            upload_queue.fail, job["upload_id"], str(exc.detail),
        )
        staged_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        await asyncio.to_thread(upload_queue.fail, job["upload_id"], str(exc))
        staged_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to stage upload")

    return JSONResponse(status_code=201, content={
        "upload_id": job["upload_id"],
        "status": "queued",
        "accepted": True,
        "filename": filename,
        "session_id": None,
        "session_name": Path(filename).stem,
        "files_preserved": None,
        "imported": 0,
        "project_id": project_id or None,
        "project_name": None,
        "dispatch": None,
        "episodes": [],
    })


@router.get("/session/upload/{upload_id}")
async def session_upload_status(
    upload_id: str,
    _: str = Depends(verify_api_key),
):
    """Return the durable receive/ingest status for an upload."""
    from app import upload_queue
    job = await asyncio.to_thread(upload_queue.get_job, upload_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Upload job not found")
    return upload_queue.public_job(job)


@router.post("/session/upload/{upload_id}/retry")
async def retry_session_upload(
    upload_id: str,
    _: str = Depends(verify_api_key),
):
    """Retry a failed upload from its retained local receive spool."""
    from app import upload_queue
    job = await asyncio.to_thread(upload_queue.get_job, upload_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Upload job not found")
    if job.get("status") != "failed":
        raise HTTPException(status_code=409, detail="Only failed uploads can be retried")
    staged_path = Path(str(job.get("staged_path") or ""))
    if not staged_path.is_file():
        raise HTTPException(status_code=410, detail="Retained upload spool is missing")
    await asyncio.to_thread(upload_queue.enqueue, upload_id)
    latest = await asyncio.to_thread(upload_queue.get_job, upload_id)
    return upload_queue.public_job(latest or job)


async def _process_upload_job(
    file: UploadFile,
    name: str = "",
    project_id: str = "",
    staged_path: Path | None = None,
):
    """Normalize one locally-spooled archive into the authoritative dataset."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    filename = Path(str(file.filename).replace("\\", "/")).name
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name")

    source_path = Path(staged_path) if staged_path else None
    if source_path is not None:
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="Upload spool is missing")
        # Keep decompression and intermediate files local too.  Only the final
        # normalized batch and preserved original archive are written to NAS.
        tmpdir = Path(tempfile.mkdtemp(
            prefix="egodata-upload-", dir=str(source_path.parent),
        ))
        archive_path = source_path
    else:
        upload_tmp_root = settings.temp_root / "uploads"
        upload_tmp_root.mkdir(parents=True, exist_ok=True)
        tmpdir = Path(tempfile.mkdtemp(
            prefix="egodata-upload-", dir=str(upload_tmp_root),
        ))
        archive_path = None
    batch_dir = None
    staging_dir = None
    backup_dir = None
    committed = False
    rollback_failed = False
    old_state = {}
    old_annotations = []

    def _rollback_commit() -> None:
        """Restore the previous batch if anything fails after the swap."""
        nonlocal committed, rollback_failed
        if not committed:
            # Failed uploads are recorded in the upload/job state and logs;
            # the extracted staging tree is disposable and must not accumulate
            # hidden ``.upload-*`` directories below the sessions root.
            if staging_dir and staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
                print(f"[Upload] Removed failed staging tree: {staging_dir}")
            return
        # Project-level commits are already transactional inside
        # append_project_episode().  Once it has returned successfully there
        # is no batch directory to roll back here; keeping the new project is
        # safer than deleting the whole multi-episode dataset on a later
        # diagnostic/dispatch error.
        if not backup_dir or not backup_dir.exists():
            committed = False
            return
        try:
            if batch_dir and batch_dir.exists():
                shutil.rmtree(batch_dir, ignore_errors=True)
            if backup_dir and backup_dir.exists() and batch_dir:
                backup_dir.replace(batch_dir)
            from app.localstore import (
                invalidate_session_cache, save_annotations, write_episode_state,
            )
            if batch_dir:
                write_episode_state(batch_name, old_state)
                save_annotations(batch_name, old_annotations)
                # The previous directory has been restored.  Its streams and
                # metadata may differ from the failed replacement, so this is
                # a structural change rather than a cheap status update.
                invalidate_session_cache()
        except Exception as rollback_err:
            rollback_failed = True
            print(f"[Upload] Rollback failed: {rollback_err}")
        committed = False

    try:
        # ---- 兼容直接调用:正常 HTTP 路径已经在本地 spool 完成 ----
        if archive_path is None:
            archive_path = tmpdir / Path(str(file.filename).replace("\\", "/")).name
            with open(archive_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    f.write(chunk)

        # ---- 先探测压缩包格式(魔数优先,不盲信扩展名),再解压 ----
        kind = _detect_archive_kind(file.filename, archive_path)
        extract_tmp = tmpdir / "extract"
        extract_tmp.mkdir()
        if kind == "zip":
            _extract_zip_encoded(archive_path, extract_tmp)
        else:
            _extract_tar_encoded(archive_path, extract_tmp, kind)
        print(f"[Upload] Archive {kind} extracted ({file.filename})")

        # ---- Normalize layout: strip one redundant top-level directory ----
        # Collector archives sometimes carry a wrapper folder (e.g.
        # "episode_000015/…" or "dataset/…"), which would otherwise nest one
        # level too deep: sessions/<task>/<session>/<wrapper>/…. Expected
        # layout is sessions/<task>/<session>/timestamps.json (flat dataset).
        extract_items = list(extract_tmp.iterdir())
        if len(extract_items) == 1 and extract_items[0].is_dir():
            inner = extract_items[0]
            inner_names = {p.name.lower() for p in inner.iterdir()}
            if inner_names & {"data", "meta", "videos", "calibration",
                              "timestamps.json", "metadata.json", "dataset"}:
                for item in inner.iterdir():
                    shutil.move(str(item), str(extract_tmp / item.name))
                inner.rmdir()
                print(f"[Upload] Stripped wrapper directory from archive: {inner.name}")

        # ---- 读任务名 ----
        meta_root = _find_meta_dir(extract_tmp) or extract_tmp
        info = _read_json(meta_root / "meta" / "info.json")
        fps = info.get("fps", 30)
        codebase_version = info.get("codebase_version", "v3.0")

        tasks = {}
        tasks_json_path = meta_root / "meta" / "tasks.json"
        if tasks_json_path.is_file():
            try:
                task_value = json.loads(tasks_json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                task_value = []
            if isinstance(task_value, dict):
                task_value = task_value.get("tasks", [])
            if isinstance(task_value, list):
                for index, item in enumerate(task_value):
                    if isinstance(item, str):
                        tasks[index] = item
                    elif isinstance(item, dict):
                        tid = item.get("task_id", item.get("task_index", index))
                        desc = item.get("description", item.get("task", ""))
                        if str(desc).strip():
                            tasks[tid] = desc
        if not tasks:
            tasks_path = meta_root / "meta" / "tasks.jsonl"
            if tasks_path.exists():
                for line in _read_lines(tasks_path):
                    t = json.loads(line.strip())
                    tid = t.get("task_id", t.get("task_index", 0))
                    desc = t.get("description", t.get("task", ""))
                    tasks[tid] = desc

        task_desc = tasks.get(0, name or "default_recording")

        # ---- 从 zip 文件名解析前缀和时间戳 ----
        from app.storage import parse_zip_filename
        zip_info = parse_zip_filename(file.filename)

        # ---- 归属项目:zip 前缀(或 name 参数)匹配本地项目 ----
        # 先精确匹配,再前缀匹配 —— zip 名为 "Test005_000028" 应归入项目 "Test005"。
        from app.storage import sanitize_task_name
        from app.localstore import (
            list_projects, write_episode_state, list_upload_history,
        )
        requested_project_id = str(project_id or "").strip()
        project_name = (name or zip_info['prefix']).strip()
        # 项目可以先建好再搭工作流。只有 archived 项目不再接收新批次；
        # draft/active/paused 都保留数据归属，active 才决定是否自动派发。
        active_projects = [p for p in list_projects()
                           if p.get("status", "active") != "archived"]

        def _project_aliases(project: dict) -> set[str]:
            aliases = {str(project.get("name") or "").strip()}
            raw_aliases = project.get("name_aliases") or []
            if isinstance(raw_aliases, str):
                raw_aliases = [raw_aliases]
            aliases.update(
                str(alias).strip()
                for alias in raw_aliases
                if str(alias).strip()
            )
            # Backward compatibility for projects renamed before aliases
            # were introduced: existing batch prefixes are old collector
            # names and therefore valid aliases for this project.
            project_dir = settings.storage_root / "sessions" / sanitize_task_name(
                str(project.get("name") or ""))
            if project_dir.is_dir():
                for child in project_dir.iterdir():
                    if not child.is_dir():
                        continue
                    match = re.match(r"^(.+)_\d{6}$", child.name)
                    if match:
                        aliases.add(match.group(1))
            return {alias for alias in aliases if alias}

        aliases_by_project = {
            id(project): _project_aliases(project)
            for project in active_projects
        }
        matched_project = None
        if requested_project_id:
            # A client-provided project_id is authoritative. Never let a zip
            # prefix or a similarly named project override an explicit choice.
            matched_project = next(
                (p for p in active_projects
                 if str(p.get("id") or "") == requested_project_id),
                None,
            )
            if matched_project is None:
                raise HTTPException(status_code=400,
                                    detail=f"Unknown project_id: {requested_project_id}")
            if matched_project.get("status") == "archived":
                raise HTTPException(status_code=409,
                                    detail="Archived projects cannot receive new uploads")
            project_name = str(matched_project.get("name") or project_name).strip()
        else:
            # Legacy clients may still send only name. Keep compatibility, but
            # require a unique exact/prefix resolution when no project_id was
            # supplied; the selected project remains visible in the response.
            project_name_low = project_name.lower()
            exact_hits = [
                p for p in active_projects
                if project_name_low in {alias.lower() for alias in aliases_by_project[id(p)]}
            ]
            if len(exact_hits) == 1:
                matched_project = exact_hits[0]
            elif len(exact_hits) > 1:
                raise HTTPException(
                    status_code=409,
                    detail=(f"Ambiguous project name '{project_name}'; "
                            "provide project_id explicitly"),
                )
            else:
                prefix_hits = [
                    p for p in active_projects
                    if any(project_name_low.startswith(alias.lower())
                           for alias in aliases_by_project[id(p)])
                ]
                # Only use a prefix fallback when exactly one project wins.
                # Similar project names must be selected explicitly.
                if prefix_hits:
                    longest = max(
                        len(alias)
                        for p in prefix_hits
                        for alias in aliases_by_project[id(p)]
                        if project_name_low.startswith(alias.lower())
                    )
                    longest_hits = [
                        p for p in prefix_hits
                        if any(
                            project_name_low.startswith(alias.lower())
                            and len(alias) == longest
                            for alias in aliases_by_project[id(p)]
                        )
                    ]
                    if len(longest_hits) == 1:
                        matched_project = longest_hits[0]
                    else:
                        raise HTTPException(
                            status_code=409,
                            detail=(f"Ambiguous project prefix '{project_name}'; "
                                    "provide project_id explicitly"),
                        )
        project_folder = sanitize_task_name(matched_project["name"]) if matched_project else "Uncategorized"
        upload_history = list_upload_history()
        project_dir = settings.storage_root / "sessions" / project_folder
        from app.project_dataset import (
            allocate_project_episode_id, is_project_dataset,
            migrate_project_dataset, project_episode_rows,
        )
        # Existing projects may still contain the pre-project-level
        # ``<project>/<batch>/`` layout.  Migrate that project once before
        # appending the new episode so every new upload has one consistent
        # LeRobot 2.1 root and the append path never mixes two layouts.
        if project_dir.is_dir() and not is_project_dataset(project_dir):
            legacy_children = [child for child in project_dir.iterdir()
                               if child.is_dir() and not child.name.startswith(".")]
            if legacy_children:
                migrate_project_dataset(project_dir)
        incoming_name = str(zip_info.get("basename") or Path(filename).stem).strip()
        existing_ids = {
            str(row.get("episode_id") or row.get("source_batch") or "")
            for row in project_episode_rows(project_dir)
        }
        # A collector resend keeps its original episode ID; a new upload is
        # appended as another episode in the same project-level dataset.
        batch_name = allocate_project_episode_id(
            project_dir, incoming_name,
            matched_project["name"] if matched_project else project_folder,
        )
        batch_preexisted = batch_name in existing_ids
        is_reupload = batch_preexisted
        batch_dir = project_dir
        archive_size_bytes = archive_path.stat().st_size if archive_path.is_file() else 0

        # 新数据先落到项目目录旁的 staging 路径。合并并校验成功后,
        # 由 project_dataset 原子替换项目根目录。
        staging_dir = project_dir.parent / f".{project_folder}.{batch_name}.upload-{uuid4().hex}"
        backup_dir = None
        staging_dir.mkdir(parents=True, exist_ok=False)
        from app.localstore import read_episode_state, list_annotations
        old_state = read_episode_state(batch_name)
        old_annotations = list_annotations(batch_name)

        # ---- 移动解压文件到最终位置 ----
        for item in extract_tmp.iterdir():
            shutil.move(str(item), str(staging_dir / item.name))

        # ---- 先合并旧采集端的附加标注,再统一为 LeRobot v2.1 ----
        # v2.1 的 canonical payload 只有 data/meta/videos 三个根目录;
        # original、处理产物和审核状态不能混进这个数据集根目录。
        _normalize_episodes_layout(staging_dir)
        _merge_auto_labels(staging_dir)
        from app.lerobot_v21 import (
            is_depth_source, iter_video_streams, normalize_extracted_dataset,
        )
        normalize_extracted_dataset(staging_dir, batch_name)
        info = _read_json(staging_dir / "meta" / "info.json")

        # ---- 直接扫描规范化后的源视频,不把纯深度流当成 RGB 相机 ----
        video_records = []
        for source_key, video_file in iter_video_streams(staging_dir / "videos"):
            if is_depth_source(source_key):
                continue
            rel_path = str(video_file.relative_to(settings.storage_root)).replace("\\", "/")
            camera = source_key
            video_records.append({
                "camera": camera,
                "path": rel_path,
                "frame_count": _count_frames(video_file),
                "file": video_file.name,
            })

        camera_names = [item["camera"] for item in video_records]
        # Different streams can have a one-frame offset (for example, the
        # depth stream may contain one fewer frame than RGB).  Do not let the
        # lexicographically first camera decide the episode length.
        master_frame_count = max(
            (int(item.get("frame_count") or 0) for item in video_records),
            default=0,
        )

        # ---- 准备状态;提交到正式批次后才落盘 ----
        now = datetime.now(timezone.utc).isoformat()
        new_state = {
            "id": batch_name,
            "name": batch_name,
            "project": matched_project["name"] if matched_project else "Uncategorized",
            "status": "to_review",
            "fps": fps,
            "frame_count": master_frame_count,
            "camera_names": camera_names,
            "created_at": now,
            "received_at": now,
        }

        # ── 数据清理(文件级校验) ──
        if settings.CLEANING_ENABLED:
            try:
                from app.cleaning import validate_batch
                report = validate_batch(staging_dir)
                new_state["cleaning_report"] = report
                new_state["status"] = "to_review" if report.get("passed") else "failed"
                print(f"[Cleaning] Validated {batch_name}: passed={report.get('passed')}")
            except Exception as clean_err:
                print(f"[Cleaning] Non-fatal error during validation: {clean_err}")

        # ---- 合并到项目级 LeRobot 根目录并原子提交 ----
        from app.project_dataset import append_project_episode
        append_result = append_project_episode(
            project_dir, staging_dir, batch_name, replace=is_reupload,
        )
        committed = True
        # append_project_episode keeps its hard-link backup outside the active
        # dataset.  It is intentionally not removed by this request cleanup.
        print(
            f"[Upload] Appended project episode {batch_name} "
            f"chunk={append_result.get('chunk_index')}"
        )
        backup_dir = None

        # The uploaded archive is only an ingest transport.  It is not part of
        # the LeRobot dataset and is deliberately not copied into an archive
        # directory.  The temporary archive/extracted wrapper is removed in
        # ``finally`` only after this verified atomic commit; failed jobs keep
        # their local spool so they can be retried or inspected.
        # 只有正式提交后才替换同名 episode 的状态和标注,避免 staging
        # 阶段污染线上数据。新上传默认从空标注开始。
        from app.localstore import (
            invalidate_session_cache, save_annotations, write_episode_state,
        )
        write_episode_state(batch_name, new_state)
        save_annotations(batch_name, [])
        # A new upload or a same-name re-upload changes files/cameras, not
        # just review state.  Force the next session scan to discover the new
        # structure before a workflow or review request can use stale streams.
        invalidate_session_cache()
        # 同名重传会替换批次目录:媒体视图缓存与 worker 输入包缓存一并失效
        # (输入包由新鲜度戳兜底,这里主动清理避免留下无用的大文件)。
        try:
            from app.media_cache import invalidate_episode as _invalidate_media
            _invalidate_media(batch_name)
            from app.api.worker import clear_input_zip_cache
            clear_input_zip_cache(batch_name)
        except Exception:
            pass
        # The generic auto_actions file is in the staged upload metadata. It
        # is intentionally not copied into the project-level canonical meta
        # root, so import it before staging is cleaned up.
        _import_auto_actions_json(staging_dir, batch_name)

        # ── 自动入队:统一派发器(没有工作流时只入库)──
        dispatch_result = {"queued": 0, "matched": 0, "skipped": 0}
        try:
            from app.workflow_dispatch import (
                dispatch_project_episode, project_workflow_ids,
            )
            # Warm the Worker input archive before publishing a run.  The
            # batch itself is already committed, and this work stays in the
            # durable upload thread, so the collector has already received a
            # 201 while a cold NAS/SSHFS scan happens here.  The Worker route
            # uses the same lock and cache for manual reruns.
            if matched_project and project_workflow_ids(matched_project):
                try:
                    from app.api.worker import prepare_episode_input_cache
                    prepare_episode_input_cache(batch_name, batch_dir)
                    print(f"[Upload] Worker input cache ready: {batch_name}")
                except Exception as cache_err:
                    # The upload is still valid; the Worker endpoint will
                    # retry the cache build when it claims the run.
                    print(f"[Upload] Worker input cache warm-up skipped: {cache_err}")
            dispatch_result = dispatch_project_episode(matched_project, {
                "id": batch_name,
                "path": str(batch_dir),
                "camera_names": camera_names,
                # Preserve the collector's physical grouping for type-first
                # workflow matching (Mono / Stereo / RGB-D). The run snapshot
                # still stores source keys for actual file resolution.
                "device_names": info.get("device_names") or {},
                "devices": info.get("devices") or [],
                "sensors": [str(s) for s in (info.get("sensors") or [])],
            }, trigger="upload", force_rerun=is_reupload)
            print(
                f"[Upload] Auto-dispatch batch={batch_name} "
                f"reupload={is_reupload} result={dispatch_result}"
            )
        except Exception as auto_err:
            # 入库成功不应因为调度器故障回滚原始数据；后续项目绑定或
            # 手工重处理仍可补派。
            print(f"[Upload] Auto-dispatch skipped (data preserved): {auto_err}")

        # 计数和唯一批次索引只在新批次完整提交后更新。覆盖重传只
        # 追加审计事件，不增加项目的唯一批次数。
        if matched_project:
            try:
                primary = {c for c in camera_names if not c.lower().endswith("_aux")}
                declared = [str(s) for s in (info.get("sensors") or [])]
                current = set(matched_project.get("observed_inputs") or [])
                matched_project["observed_inputs"] = sorted(current | primary | set(declared))
                from app.localstore import upsert_project, upsert_upload_record
                upsert_upload_record(batch_name, matched_project["name"], now)
                # Recompute from unique committed ids instead of incrementing
                # a legacy counter. This also repairs old projects whose
                # uploaded_total was inflated by previous reuploads.
                history_ids = {
                    str(item.get("episode_id") or "")
                    for item in list_upload_history()
                    if item.get("project_name") == matched_project["name"]
                    and item.get("episode_id")
                }
                current_ids = {
                    str(row.get("episode_id") or row.get("source_batch") or "")
                    for row in project_episode_rows(project_dir)
                    if str(row.get("episode_id") or row.get("source_batch") or "")
                }
                matched_project["uploaded_total"] = len(history_ids | current_ids)
                upsert_project(matched_project)
            except Exception as obs_err:
                print(f"[Upload] Failed to record upload counters: {obs_err}")

        # 只有批次完整落盘并完成入库后才追加审计事件;中途解压/校验
        # 失败时不把失败尝试误记成一次成功上传。
        if matched_project:
            try:
                from app.localstore import record_upload_event
                record_upload_event(
                    batch_name,
                    matched_project["name"],
                    now,
                    classification="reupload" if is_reupload else "new_batch",
                    overwrote_existing=batch_preexisted,
                    archive_filename=file.filename,
                    archive_size_bytes=archive_size_bytes,
                )
            except Exception as event_err:
                print(f"[Upload] Failed to record upload event: {event_err}")

        return JSONResponse(status_code=201, content={
            "session_id": batch_name,
            "session_name": zip_info['prefix'],
            "files_preserved": str(batch_dir),
            "imported": 1 if video_records else 0,
            "project_id": matched_project.get("id") if matched_project else None,
            "project_name": matched_project.get("name") if matched_project else None,
            "dispatch": dispatch_result,
            "episodes": [{
                "episode_id": batch_name,
                "camera": video_records[0]["camera"] if video_records else None,
                "camera_names": camera_names,
                "frame_count": master_frame_count,
                "file": video_records[0]["file"] if video_records else None,
            }],
        })

    except HTTPException:
        _rollback_commit()
        raise
    except Exception as e:
        _rollback_commit()
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")
    finally:
        # Do not remove the extracted tree until append_project_episode has
        # completed its atomic swap and verification.  Once the request exits,
        # both success and failure paths must be free of hidden upload trees;
        # the durable upload queue keeps the original archive for retry.
        if staging_dir and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir and backup_dir.exists() and not rollback_failed:
            shutil.rmtree(backup_dir, ignore_errors=True)
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


@router.get("/sessions")
async def session_list():
    from app.localstore import scan_sessions
    rows = scan_sessions()
    return {"sessions": [{"id": r["id"], "name": r["name"],
                           "episode_count": 1,
                           "created_at": r.get("created_at", "")} for r in rows]}


@router.delete("/session/{session_id}")
async def session_delete(session_id: str, _: str = Depends(verify_api_key)):
    from app.localstore import delete_episode
    delete_episode(session_id, permanent=True)
    return {"message": "Session deleted"}


# ── helpers ──────────────────────────────────────────

# ── 压缩包处理:先探测格式 → 编码修复解压 → 原始压缩包单独保留 ──

def _detect_archive_kind(filename: str, path: Path) -> str:
    """探测压缩包格式,优先读文件头魔数,回退按扩展名。

    返回: zip | tar | tar.gz | tar.bz2
    避免"扩展名写了 .zip 实际是 tar"之类的错配解压。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except Exception:
        head = b""
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "tar.gz"
    if head.startswith(b"BZh"):
        return "tar.bz2"
    if head[257:262] == b"ustar":
        return "tar"
    fn = (filename or "").lower()
    if fn.endswith(".zip"):
        return "zip"
    if fn.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if fn.endswith(".tar.bz2"):
        return "tar.bz2"
    if fn.endswith(".tar"):
        return "tar"
    raise HTTPException(status_code=400, detail=f"Unsupported archive format: {filename}")


def _fix_zip_name(raw: str) -> str:
    """修复 zip 条目名编码:Windows 压缩工具(WinRAR/好压等)用 GBK 写文件名,
    Python zipfile 默认按 cp437 解码 → 中文名乱码。还原字节后按 GBK 重解。"""
    try:
        raw.encode("cp437")
    except UnicodeEncodeError:
        return raw  # 本身是合法 Unicode(UTF-8 flag 已置位),无需修复
    for enc in ("gbk", "utf-8"):
        try:
            return raw.encode("cp437").decode(enc)
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return raw


def _fix_tar_name(name: str) -> str:
    """修复 tar 成员名编码:tarfile 按 UTF-8 解码,GBK 名字以 surrogateescape
    残留形式出现(U+DC80–U+DCFF)。还原字节后按 GBK 重解。"""
    try:
        name.encode("utf-8")  # 可完整编码 → 名字本身合法,无需修复
        return name
    except UnicodeEncodeError:
        pass
    try:
        return name.encode("utf-8", "surrogateescape").decode("gbk")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def _safe_extract_target(dest: Path, name: str) -> Path:
    """防 zip-slip:目标路径必须落在 dest 内。"""
    root = dest.resolve()
    target = (dest / name).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Unsafe archive path: {name}")
    return target


def _extract_zip_encoded(archive_path: Path, dest: Path) -> int:
    """解压 zip,条目名做编码修复(GBK→UTF-8),避免中文文件名乱码。"""
    count = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            raw = info.filename
            if not (info.flag_bits & 0x800):  # 未标记 UTF-8 → 可能是 GBK
                raw = _fix_zip_name(raw)
            target = _safe_extract_target(dest, raw)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            count += 1
    return count


def _normalize_episodes_layout(batch_dir: Path) -> None:
    """Normalize an uploaded episode index to one Parquet file per episode."""
    episodes_dir = batch_dir / "meta" / "episodes"
    if not episodes_dir.is_dir():
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        files = sorted(episodes_dir.rglob("*.parquet"))
        rows: list[dict] = []
        for path in files:
            rows.extend(row for row in pq.read_table(path).to_pylist()
                        if isinstance(row, dict))
    except (ImportError, OSError, ValueError, TypeError) as exc:
        print(f"[Upload] episodes normalize skipped: {exc}")
        return
    if not rows:
        return
    rows.sort(key=lambda row: int(row.get("episode_index", 10**9)))
    for index, row in enumerate(rows):
        row.setdefault("episode_index", index)
    shutil.rmtree(episodes_dir)
    for row in rows:
        index = int(row["episode_index"])
        target = (episodes_dir / f"chunk-{index // 1000:03d}"
                  / f"episode_{index:06d}.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row]), target)


def _extract_tar_encoded(archive_path: Path, dest: Path, kind: str) -> int:
    """解压 tar/tar.gz/tar.bz2,成员名做编码修复。"""
    mode = {"tar.gz": "r:gz", "tar.bz2": "r:bz2"}.get(kind, "r")
    count = 0
    with tarfile.open(archive_path, mode) as tf:
        for member in tf.getmembers():
            target = _safe_extract_target(dest, _fix_tar_name(member.name))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            count += 1
    return count


def _merge_auto_labels(session_dir: Path) -> None:
    """Merge auto_labels + hand_kpts parquet files into the main data parquet.

    Collects:
      - auto_labels_chunk_*.parquet  (gesture labels, no coords)
      - hand_kpts_chunk_*.parquet    (raw hand_data flat array, 92 floats)

    Converts the flat ``hand_data`` array into ``hand_0_keypoints`` /
    ``hand_1_keypoints`` (21 × [x, y] lists each) and merges everything
    into ``chunk_*.parquet`` on ``frame_index``.

    After merge the separate auto / hand_kpts files are removed.
    """
    data_dir = None
    for d in session_dir.rglob("data"):
        if d.is_dir():
            data_dir = d
            break
    if data_dir is None:
        return

    # Both the legacy collector and LeRobot-style uploads may place these
    # files under data/chunk-XXX rather than directly under data/.
    auto_label_files = sorted(data_dir.rglob("auto_labels_chunk_*.parquet"))
    hand_kpts_files  = sorted(data_dir.rglob("hand_kpts_chunk_*.parquet"))

    if not auto_label_files and not hand_kpts_files:
        return

    import pandas as pd
    import numpy as np

    # ── Build a combined hand-data DataFrame ──
    # Start from auto_labels if available, otherwise hand_kpts
    hand_df = None

    if auto_label_files:
        for auto_path in auto_label_files:
            try:
                hand_df = pd.read_parquet(auto_path)
                break
            except Exception:
                continue

    if hand_kpts_files:
        for kpts_path in hand_kpts_files:
            try:
                kpts_df = pd.read_parquet(kpts_path)
                # Convert flat hand_data array → structured keypoint lists
                kpts_df = _parse_hand_kpts(kpts_df)
                if hand_df is not None:
                    hand_df = hand_df.merge(
                        kpts_df, on="frame_index", how="left", suffixes=("", "_kpts")
                    )
                else:
                    hand_df = kpts_df
                break
            except Exception as e:
                print(f"[AutoLabels] Failed to parse hand_kpts {kpts_path.name}: {e}")

    if hand_df is None:
        return

    # ── Merge into main data parquet ──
    # Exclude auto_labels / hand_kpts files from data file list
    data_files = sorted(
        p for p in data_dir.rglob("*.parquet")
        if not p.name.startswith("auto_labels_") and not p.name.startswith("hand_kpts_")
    )
    if not data_files:
        return
    data_path = data_files[0]

    try:
        data_df = pd.read_parquet(data_path)
        merged = data_df.merge(hand_df, on="frame_index", how="left")
        merged.to_parquet(data_path, index=False)

        # Remove the separate files after successful merge
        for p in auto_label_files:
            try: p.unlink()
            except Exception: pass
        for p in hand_kpts_files:
            try: p.unlink()
            except Exception: pass

        print(f"[AutoLabels] Merged {len(hand_df.columns)} hand cols "
              f"→ {data_path.name} ({len(merged)} rows)")
    except Exception as e:
        print(f"[AutoLabels] Failed to merge into {data_path.name}: {e}")


def _parse_hand_kpts(kpts_df: "pd.DataFrame") -> "pd.DataFrame":
    """Convert flat ``hand_data`` column into ``hand_0_keypoints`` / ``hand_1_keypoints``.

    ``hand_data`` is a flat float32 array of 92 values:
      - Floats  0–45  → hand 0: 23 (x,y) pairs  (21 landmarks + 2 bbox pts)
      - Floats 46–91  → hand 1: 23 (x,y) pairs

    We extract the first 21 pairs per hand as the standard hand landmarks.
    When ``num_hands == 0`` the array is all zeros — we return None.
    """
    import numpy as np

    def extract_hand(hd, offset, num_hands, hand_idx):
        """Extract 21 keypoints for one hand from the flat array."""
        if hd is None:
            return None
        # hd may be a numpy array or a list
        arr = np.asarray(hd, dtype=np.float32) if not isinstance(hd, np.ndarray) else hd
        if len(arr) < 92:
            return None
        # Check if this hand is active: first point has non-zero x
        first_x = float(arr[offset])
        if first_x == 0.0 and num_hands <= hand_idx:
            return None  # this hand slot is empty
        kp = []
        for i in range(21):  # first 21 landmarks
            idx = offset + i * 2
            x = float(arr[idx])
            y = float(arr[idx + 1])
            kp.append([x, y])
        # All zeros → no hand
        if all(abs(p[0]) < 0.5 and abs(p[1]) < 0.5 for p in kp):
            return None
        return kp

    kp_0_list = []
    kp_1_list = []
    for _, row in kpts_df.iterrows():
        hd = row.get("hand_data")
        nh = int(row.get("num_hands", 0))
        kp_0_list.append(extract_hand(hd, 0, nh, 0))
        kp_1_list.append(extract_hand(hd, 46, nh, 1))

    result = kpts_df[["frame_index"]].copy()
    # Ensure pure Python types (not numpy arrays) for JSON serialization
    result["hand_0_keypoints"] = [
        [[float(x), float(y)] for x, y in kp] if kp is not None else None
        for kp in kp_0_list
    ]
    result["hand_1_keypoints"] = [
        [[float(x), float(y)] for x, y in kp] if kp is not None else None
        for kp in kp_1_list
    ]
    return result


def _import_auto_actions_json(batch_dir: Path, batch_name: str) -> None:
    """Parse meta/auto_actions.jsonl → 写标注 JSON(localstore)。

    Each line describes an action segment:
      {"action": "reach", "start_frame": 0, "end_frame": 75, "confidence": 0.92}

    These become pre-annotations that human reviewers verify and correct.
    """
    actions_path = batch_dir / "meta" / "auto_actions.jsonl"
    if not actions_path.exists():
        return

    from app.localstore import list_annotations, save_annotations

    try:
        lines = actions_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return

        existing = list_annotations(batch_name)
        existing_ids = {a.get("id") for a in existing}
        sort_order = len(existing)
        created = 0
        for line in lines:
            try:
                act = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            label = act.get("action") or act.get("label", "")
            start_f = int(act.get("start_frame", 0))
            end_f = int(act.get("end_frame", 0))
            if not label or end_f <= start_f:
                continue
            seg_id = f"auto-{batch_name}-{sort_order}"
            if seg_id in existing_ids:
                continue
            confidence = act.get("confidence")
            existing.append({
                "id": seg_id,
                "episode_id": batch_name,
                "label": label,
                "start_frame_index": start_f,
                "end_frame_index": end_f,
                "color": "#3B82F6",
                "sort_order": sort_order,
                "notes": "auto-generated"
                         + (f" (confidence: {confidence:.2f})" if confidence is not None else ""),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            existing_ids.add(seg_id)
            sort_order += 1
            created += 1
        if created:
            save_annotations(batch_name, existing)
            print(f"[AutoActions] Created {created} pre-annotations for {batch_name}")
            # Notify browsers watching this episode that annotations appeared
            from app.routes.annotations import notify_annotations_changed
            notify_annotations_changed(batch_name, "auto_import")
    except Exception as e:
        print(f"[AutoActions] Failed to import auto_actions: {e}")

def _find_meta_dir(root: Path) -> Path | None:
    for d in root.rglob("meta"):
        if d.is_dir(): return d.parent
    return root if (root / "meta").is_dir() else None


def _read_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f: return json.load(f)
    return {}


def _read_lines(path: Path) -> list[str]:
    if path.exists():
        with open(path) as f: return f.readlines()
    return []


def _count_frames(video_path: Path) -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release(); return count
    except Exception:
        return 1


def _guess_camera(mp4_path: Path) -> str:
    """从路径中提取摄像头名,兼容新旧两种布局。

    旧格式: videos/<cam>/chunk_000000.mp4        → cam
    新格式: videos/<cam>/chunk-0000/<cam>.mp4    → cam
            videos/<cam>/chunk-0000/<cam>_aux.mp4 → <cam>_aux
    (新格式的 aux 辅助视频与主视频在同一个目录下,不能都叫 cam,
     否则 dict 里 aux 会覆盖主目;文件名带 _aux 且目录名不带 → 拆成
     <目录名>_aux。)
    """
    # Prefer the source-aware parser.  In LeRobot v2.1 the first directory
    # below videos/ is ``chunk-000``; using it as the camera name is the cause
    # of the historical "all devices become chunk-000" bug.
    try:
        from app.lerobot_v21 import source_key_from_video
        source = source_key_from_video(mp4_path)
        if source:
            if "_aux" in mp4_path.stem.lower() and "_aux" not in source.lower():
                return f"{source}_aux"
            return source
    except Exception:
        pass

    parts = mp4_path.parts
    cam_dir = None
    for i, p in enumerate(parts):
        if p == "videos" and i + 1 < len(parts):
            cam_dir = parts[i + 1]
            break
    if cam_dir is None:
        cam_dir = mp4_path.parent.name
    stem = mp4_path.stem.lower()
    if "_aux" in stem and "_aux" not in cam_dir.lower():
        return f"{cam_dir}_aux"
    return cam_dir


def _episode_index_of(basename: str) -> int | None:
    """任务内序号: Test005_000028 → 28。

    排除时间戳格式(session_20260805_192922 → None,结尾 6 位是时分秒,
    前面还有 8 位日期,不是任务内序号)。
    """
    import re
    if re.search(r"_(\d{8})_\d{6}$", str(basename)):
        return None
    m = re.search(r"_(\d{6})$", str(basename))
    return int(m.group(1)) if m else None


def _camera_role(camera: str) -> str:
    """Derive a display role without changing the source camera identifier."""
    normalized = camera.lower().replace("-", "_").replace(" ", "_")
    if "left" in normalized:
        return "left"
    if "right" in normalized:
        return "right"
    if "aux" in normalized:
        return "aux"
    if "depth" in normalized:
        return "depth"
    return "primary"
