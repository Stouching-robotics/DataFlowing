"""Worker queue + artifact transport — 本地 JSON 运行队列(无数据库)。

运行记录存系统级 ``data/state/runs/<run_id>.json``；输入 = 批次目录打包
zip；完成结果合并回项目的 ``data/meta/videos`` 三个 canonical 目录。
``state/runs`` 不会创建在任何项目目录内。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.config import settings
from app.localstore import (
    list_runs, get_run, save_run, get_episode,
    set_episode_status, list_exceptions, delete_exception,
    update_run_if_owned, save_annotations,
)
from app.security import verify_worker_api_key

router = APIRouter(prefix="/api/v1/worker", tags=["worker"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _lease_seconds() -> int:
    return max(15, settings.WORKER_LEASE_SECONDS)


def _worker_tmp_root() -> Path:
    # ZIP assembly and result upload are transient operations.  Keep them on
    # the local disk instead of the SSHFS/NAS-backed STORAGE_DIR; otherwise a
    # cold workflow pays remote filesystem latency before the first node runs.
    path = settings.upload_staging_root / "worker-api"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remove_later(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


@router.post("/jobs/claim")
async def claim_job(body: dict, _: str = Depends(verify_worker_api_key)):
    """Worker 轮询领取任务:queued 或租约过期的 running。"""
    now = _now()
    runs = list_runs()
    candidate = None
    for r in sorted(runs, key=lambda r: r.get("created_at") or ""):
        status = r.get("status")
        if status in ("queued", "pending"):
            candidate = r
            break
        if status == "running":
            try:
                lease = datetime.fromisoformat(r.get("lease_until") or "")
                if lease < now:
                    candidate = r
                    break
            except Exception:
                candidate = r
                break
    if candidate is None:
        return Response(status_code=204)

    run = candidate
    run["status"] = "running"
    run["worker_id"] = body.get("worker_id")
    run["lease_token"] = uuid.uuid4().hex
    run["attempt"] = (run.get("attempt") or 0) + 1
    run["started_at"] = run.get("started_at") or now.isoformat()
    run["heartbeat_at"] = now.isoformat()
    run["lease_until"] = (now + timedelta(seconds=_lease_seconds())).isoformat()
    save_run(run)

    ep = get_episode(run.get("episode_id"))
    if ep is None:
        run["status"] = "failed"
        run["error_log"] = "Episode no longer exists"
        save_run(run)
        raise HTTPException(status_code=409, detail=run["error_log"])

    return {
        "run_id": run["id"],
        "workflow_id": run.get("workflow_id"),
        "episode_id": run.get("episode_id"),
        "workflow_name": run.get("workflow_name"),
        "graph": run.get("graph") or {},
        "node_configs": run.get("node_configs") or {},
        "attempt": run.get("attempt", 0),
        "lease_token": run.get("lease_token"),
        "video_path": None,
        "video_paths": ep.get("camera_streams") or {},
        "cameras": ep.get("camera_names") or [],
        "device_names": ep.get("device_names") or {},
        "devices": ep.get("devices") or [],
        "camera": (ep.get("camera_names") or [None])[0] if ep.get("camera_names") else None,
        "fps": ep.get("fps") or 30,
        "input_url": f"/api/v1/worker/jobs/{run['id']}/input",
    }


_INPUT_ZIP_CACHE_KEEP = 3  # 本地磁盘上最多保留的批次输入包数量(最近使用优先)
_INPUT_ZIP_LOCK = threading.Lock()


def _input_zip_cache_dir() -> Path:
    """Worker 输入包缓存目录 —— 必须放本地临时目录(非远程挂载),复用才有意义。

    丢缓存只会导致重新打包,不影响正确性。
    """
    return Path(tempfile.gettempdir()) / "egodata-worker-input-cache"


def _input_zip_cache_paths(episode_id: str) -> tuple[Path, Path]:
    base = _input_zip_cache_dir() / str(episode_id)
    return base.with_suffix(".zip"), base.with_suffix(".stamp.json")


def _input_zip_stamp(batch_dir: Path) -> dict:
    """廉价新鲜度戳:上传总会重写 info.json,目录 inode 兜底重传改名。"""
    for rel in ("meta/info.json", "metadata.json"):
        cand = batch_dir / rel
        if cand.is_file():
            try:
                st = cand.stat()
                return {"kind": rel, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino}
            except OSError:
                pass
    try:
        st = batch_dir.stat()
        return {"kind": "dir", "mtime_ns": st.st_mtime_ns, "ino": st.st_ino}
    except OSError:
        return {"kind": "none"}


def _input_zip_stamp_matches(stamp_path: Path, batch_dir: Path) -> bool:
    try:
        stored = json.loads(stamp_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return stored == _input_zip_stamp(batch_dir)


def _evict_input_zip_cache() -> None:
    """只保留最近使用的 N 份输入包,防止本地磁盘被多批次 zip 占满。"""
    try:
        zips = sorted(_input_zip_cache_dir().glob("*.zip"),
                      key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for stale in zips[_INPUT_ZIP_CACHE_KEEP:]:
            try:
                stale.unlink(missing_ok=True)
                stale.with_suffix(".stamp.json").unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        pass


def clear_input_zip_cache(episode_id: str) -> None:
    """批次重传/永久删除后清除对应输入包(惰性清理,失败无害)。"""
    try:
        zip_path, stamp_path = _input_zip_cache_paths(episode_id)
        zip_path.unlink(missing_ok=True)
        stamp_path.unlink(missing_ok=True)
    except Exception:
        pass


def prepare_episode_input_cache(episode_id: str, batch_dir: Path | None = None) -> Path:
    """Build a local immutable input archive for a workflow batch.

    The authoritative batch may be on SSHFS/NAS.  A single lock prevents the
    upload queue and a worker from scanning the same remote directory twice at
    the same time.  The cache file is only published after it is complete.
    """
    if batch_dir is None:
        episode = get_episode(str(episode_id))
        if episode is None:
            raise FileNotFoundError("Episode no longer exists")
        batch_dir = Path(str(episode.get("path") or ""))
    else:
        batch_dir = Path(batch_dir)
    if not batch_dir.is_dir():
        raise FileNotFoundError("Episode data is missing")

    zip_path, stamp_path = _input_zip_cache_paths(str(episode_id))
    with _INPUT_ZIP_LOCK:
        if zip_path.is_file() and _input_zip_stamp_matches(stamp_path, batch_dir):
            return zip_path

        tmp = tempfile.NamedTemporaryFile(
            prefix=f"egodata-input-{episode_id}-", suffix=".zip",
            dir=str(_worker_tmp_root()), delete=False,
        )
        tmp.close()
        try:
            with zipfile.ZipFile(tmp.name, "w", allowZip64=True) as archive:
                canonical_episode = None
                if (batch_dir / "meta" / "episodes").is_dir():
                    from app.project_dataset import episode_row, episode_files
                    canonical_episode = episode_row(batch_dir, str(episode_id))
                if canonical_episode:
                    episode_index = int(canonical_episode.get("episode_index"))
                    selected = episode_files(batch_dir, episode_index)
                    files: list[tuple[Path, Path]] = []
                    files.extend((path, path.relative_to(batch_dir))
                                  for path in selected.get("data") or [])
                    files.extend((path, path.relative_to(batch_dir))
                                 for _source, path in selected.get("videos") or [])
                    for path in selected.get("meta") or []:
                        source = batch_dir / path
                        relative = path
                        # The processing modules expect one episode's depth
                        # and calibration directly below their traditional
                        # roots, while the project dataset namespaces those
                        # files by episode to avoid collisions.
                        for folder in ("depth", "calibration", "collector"):
                            prefix = Path("meta") / folder / str(episode_id)
                            if prefix in path.parents:
                                relative = Path("meta") / folder / path.relative_to(prefix)
                                break
                        files.append((source, relative))
                else:
                    files = []
                    for path in batch_dir.rglob("*"):
                        if not path.is_file():
                            continue
                        rel = path.relative_to(batch_dir)
                        if "processed" in rel.parts or "original" in rel.parts:
                            continue
                        files.append((path, rel))
                for path, arcname in files:
                    compression = (
                        zipfile.ZIP_STORED
                        if path.suffix.lower() in {".mp4", ".parquet", ".npz"}
                        else zipfile.ZIP_DEFLATED
                    )
                    archive.write(path, str(arcname), compress_type=compression)
        except Exception:
            _remove_later(tmp.name)
            raise

        try:
            _input_zip_cache_dir().mkdir(parents=True, exist_ok=True)
            shutil.move(tmp.name, str(zip_path))
            stamp_path.write_text(
                json.dumps(_input_zip_stamp(batch_dir)), encoding="utf-8"
            )
            _evict_input_zip_cache()
        except Exception:
            _remove_later(tmp.name)
            raise
        return zip_path


@router.get("/jobs/{run_id}/input")
def download_job_input(run_id: str, _: str = Depends(verify_worker_api_key)):
    """打包 canonical episode 为 zip 供 Worker 下载。

    项目只包含 data/meta/videos；后处理结果在完成阶段合并回 episode，
    因此按批次缓存打包结果并复用:重跑工作流不再反复全量遍历/读取远程挂载 ——
    这是审核页在 run 期间点开其他视频卡顿的主要来源之一。
    """
    run = get_run(run_id)
    if run is None or run.get("episode_id") is None:
        raise HTTPException(status_code=404, detail="Worker job not found")
    ep = get_episode(run["episode_id"])
    if ep is None:
        raise HTTPException(status_code=409, detail="Episode no longer exists")
    batch_dir = Path(ep["path"])
    if not batch_dir.is_dir():
        raise HTTPException(status_code=404, detail="Episode data is missing")

    try:
        zip_path = prepare_episode_input_cache(run["episode_id"], batch_dir)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to package job input")

    return FileResponse(str(zip_path), media_type="application/zip",
                        filename=f"egodata-{run_id}.zip")


@router.post("/jobs/{run_id}/heartbeat")
async def heartbeat_job(run_id: str, body: dict, _: str = Depends(verify_worker_api_key)):
    now = _now()
    worker_id = str(body.get("worker_id") or "")
    lease_token = str(body.get("lease_token") or "")

    def mutate(run: dict) -> None:
        run["heartbeat_at"] = now.isoformat()
        run["lease_until"] = (now + timedelta(seconds=_lease_seconds())).isoformat()
        run["progress"] = body.get("progress", run.get("progress", 0.0))
        if body.get("node_states"):
            run["node_states"] = body.get("node_states")

    run = update_run_if_owned(run_id, worker_id, lease_token, mutate)
    if run is None:
        raise HTTPException(status_code=409, detail="Worker job is no longer owned by this lease")
    return {"run_id": run_id, "status": run["status"], "progress": run.get("progress")}


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe path in worker result archive")
            target = (destination / relative).resolve()
            if root != target and root not in target.parents:
                raise ValueError("Unsafe path in worker result archive")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


@router.post("/jobs/{run_id}/complete")
async def complete_job(
    run_id: str,
    # WorkerClient sends these values as multipart form fields together with
    # result_zip.  Without Form(), FastAPI treats scalar parameters as query
    # parameters, so the lease check sees empty strings and rejects every
    # otherwise successful completion with 409 Conflict.
    worker_id: str = Form(""),
    lease_token: str = Form(""),
    node_states: str = Form(""),
    outputs: str = Form(""),
    result_zip: UploadFile = File(None),
    _: str = Depends(verify_worker_api_key),
):
    """Worker 上传处理结果 → 合并到项目 episode → completed。

    字段名与 worker/client.py 保持一致:node_states / outputs(JSON 字符串)
    + result_zip(结果压缩包)。
    """
    import json
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Worker job not found")
    if (run.get("status") != "running"
            or run.get("worker_id") != worker_id
            or run.get("lease_token") != lease_token):
        raise HTTPException(status_code=409, detail="Worker job is no longer owned by this lease")
    ep = get_episode(run.get("episode_id"))
    if ep is None:
        raise HTTPException(status_code=409, detail="Episode no longer exists")

    # The uploaded result is only a transport package.  Extract it into a
    # local temporary directory, publish its useful files into the canonical
    # episode, then remove the temporary directory.  This prevents
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f"egodata-result-{run_id}-", dir=str(_worker_tmp_root())))
    tmp = tempfile.NamedTemporaryFile(prefix=f"egodata-out-{run_id}-", suffix=".zip",
                                      dir=str(_worker_tmp_root()), delete=False)
    tmp.close()
    try:
        if result_zip:
            with open(tmp.name, "wb") as f:
                while chunk := await result_zip.read(1024 * 1024):
                    f.write(chunk)
        _safe_extract(Path(tmp.name), staging_dir)
    except Exception as e:
        _remove_later(tmp.name)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to store outputs: {e}")
    finally:
        _remove_later(tmp.name)

    latest = get_run(run_id)
    if (latest is None or latest.get("status") != "running"
            or latest.get("worker_id") != worker_id
            or latest.get("lease_token") != lease_token):
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=409, detail="Worker job lease changed before completion")

    finished_at = _now().isoformat()
    parsed_states = json.loads(node_states) if node_states else (latest.get("node_states") or {})
    parsed_outputs = json.loads(outputs) if outputs else (latest.get("outputs") or {})

    try:
        from app.project_dataset import publish_processing_result
        parsed_outputs = publish_processing_result(
            Path(ep["path"]), str(run.get("episode_id") or ""), run_id,
            staging_dir, parsed_outputs, parsed_states, finished_at,
        )
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=500,
                            detail=f"Failed to publish episode result: {exc}")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    def mark_completed(current: dict) -> None:
        current["status"] = "completed"
        current["finished_at"] = finished_at
        current["progress"] = 1.0
        current["node_states"] = parsed_states
        current["outputs"] = parsed_outputs

    run = update_run_if_owned(run_id, worker_id, lease_token, mark_completed)
    if run is None:
        raise HTTPException(status_code=409, detail="Worker job lease changed during completion")

    # 产物已整体替换:失效该批次的媒体视图缓存(media-groups/hand-3d),
    # 下次点击按新产物重新组装。
    try:
        from app.media_cache import invalidate_episode as _invalidate_media
        _invalidate_media(run["episode_id"])
    except Exception:
        pass

    # 审核状态回到待审核。failed → to_review 也要恢复:重试成功的批次
    # 若卡在 failed,前端默认列表(Reviewing)不显示,批次"消失"
    # (与下方"异常清理"同语义:用户重试成功即解除失败态)。
    # The episode snapshot was read before the dispatcher marked the run as
    # processing and can therefore still carry the old status. Always publish
    # the completed run as reviewable; checking the stale ``ep`` snapshot
    # leaves successfully reprocessed batches stuck at "processing".
    set_episode_status(run["episode_id"], "to_review")

    # A completed workflow run always starts from an empty annotation file.
    # This also covers runs queued through older/manual entry points.
    from app.ai_annotation import invalidate_ai_annotation_tasks
    invalidate_ai_annotation_tasks(run["episode_id"])
    save_annotations(run["episode_id"], [])

    # run 成功 → 该批次的上传不匹配异常已解决(用户重试成功),清理残留
    # 徽标;失败时异常保留(fail_job 会把批次置 failed,双通道可见)
    try:
        for exc in list_exceptions():
            if (exc.get("episode_id") == run.get("episode_id")
                    and exc.get("kind") == "upload_mismatch"):
                delete_exception(exc.get("id"))
    except Exception as exc:
        print(f"[Worker] Failed to clear episode exceptions: {exc}")

    # AI 标注联动:工作流含 ai_annotation 卡片 →
    # 批次处理完成后后台异步跑一次 AI 标注(不阻塞响应,失败可见)。
    # AI 节点本身就是启用标志,不再要求额外的 auto_suggest 开关。
    # local/api 仅决定实际使用的 VLM 供应商。
    try:
        from app.ai_annotation import (
            run_ai_annotation,
            ai_annotation_node_config,
            ai_quality_review_node_config,
            video_quality_review_node_config,
        )
        graph = run.get("graph") or {}
        node_configs = run.get("node_configs") or {}
        cfg = ai_annotation_node_config(graph, node_configs)
        quality_cfg = ai_quality_review_node_config(graph, node_configs)
        video_quality_cfg = video_quality_review_node_config(graph, node_configs)
        if cfg is not None and not run.get("ai_suggest_triggered"):
            run["ai_suggest_triggered"] = True
            save_run(run)
            # The editor exposes only zh/en. Normalize old or malformed
            # workflow snapshots to the historical default instead of letting
            # an arbitrary value silently select the English prompt branch.
            prompt_language = (
                "en" if str(cfg.get("prompt_language") or "zh").lower() in {"en", "english"}
                else "zh"
            )
            run_ai_annotation(
                run["episode_id"],
                str(cfg.get("mode") or "signal_vlm"),
                float(cfg.get("min_confidence", 0.7)),
                prompt_language,
                debounce_sec=float(cfg.get("debounce_sec") or 2.0),
                min_seg_sec=float(cfg.get("min_seg_sec") or 0.8),
                max_segments=int(cfg.get("max_segments") or 0),
                auto_confirm=True,
                quality_gate=quality_cfg is not None,
                video_quality_gate=video_quality_cfg is not None,
                vlm_cfg=cfg,  # vlm_provider/api_* 供应商配置(卡片)
            )
            print(f"[Worker] AI annotation triggered "
                  f"for {run['episode_id']} (mode={cfg.get('mode')})")
        elif (cfg is None and video_quality_cfg is not None
              and not run.get("video_quality_triggered")):
            # A media-only workflow can use the same quality card without an
            # AI Annotation node.  Run it after the worker result is stored;
            # the bounded decoder work is isolated from this completion call.
            run["video_quality_triggered"] = True
            save_run(run)
            from app.video_quality import run_video_quality_review
            asyncio.create_task(run_video_quality_review(
                run["episode_id"],
                ep["path"],
                int(ep.get("frame_count") or 0),
                float(ep.get("fps") or 0.0),
            ))
            print(f"[Worker] Video quality review triggered "
                  f"for {run['episode_id']}")
    except Exception as exc:
        print(f"[Worker] AI annotation trigger skipped: {exc}")

    # 归档联动:处理完成后后台把该批次增量推送到 NAS(只推不删;rsync
    # 逐块校验+断点续传,失败仅记日志不阻塞响应;每晚 cron 兜底补传)。
    # 脚本不存在(如 Windows 环境)时静默跳过。
    try:
        import subprocess
        import threading
        _archive_script = Path(__file__).resolve().parents[2] / "scripts" / "archive_sync.sh"
        if _archive_script.is_file():
            _batch_dir = str(Path(ep["path"]))
            threading.Thread(
                target=lambda: subprocess.run(
                    ["/bin/bash", str(_archive_script), _batch_dir],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                daemon=True).start()
    except Exception as _archive_err:
        print(f"[Worker] archive push trigger skipped: {_archive_err}")
    return {"run_id": run_id, "status": "completed"}


@router.post("/jobs/{run_id}/fail")
async def fail_job(run_id: str, body: dict, _: str = Depends(verify_worker_api_key)):
    worker_id = str(body.get("worker_id") or "")
    lease_token = str(body.get("lease_token") or "")
    finished_at = _now().isoformat()

    def mark_failed(run: dict) -> None:
        run["status"] = "failed"
        run["finished_at"] = finished_at
        run["error_log"] = body.get("error") or "Worker reported failure"

    run = update_run_if_owned(run_id, worker_id, lease_token, mark_failed)
    if run is None:
        raise HTTPException(status_code=409, detail="Worker job is no longer owned by this lease")
    # 工作流运行失败 → 批次进 Failed(Review 页 Failed 过滤可见,异常徽标
    # 由 /api/v1/exceptions 聚合 run_failed 展示)。只在 processing/to_review
    # 时覆盖 —— 已审核(reviewed/approved)的批次是历史失败,不动;
    # 用户 Reprocess 重试会先置 processing,成功后回 to_review。
    try:
        ep = get_episode(run.get("episode_id"))
        if ep is not None and ep.get("status") in ("processing", "to_review"):
            set_episode_status(run["episode_id"], "failed")
    except Exception as exc:
        print(f"[Worker] Failed to mark episode failed: {exc}")
    return {"run_id": run_id, "status": "failed"}
