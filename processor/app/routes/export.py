"""Export APIs — 本地文件打包(无数据库)。

导出任务存 data/state/export_tasks/<job_id>.json;打包 = 批次目录 zip。
"""

from __future__ import annotations

import tempfile
import zipfile
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.localstore import (
    STATE_ROOT, scan_sessions, get_episode, list_deleted_episodes, list_runs,
    get_workflow, list_projects, _read_json, _write_json, _remove_json,
)
from app.security import verify_api_key
from app.workflow_types import HAND_PROCESS_TYPES, RGBD_3D_TYPES, canonical_node_type
from app.project_dataset import (
    episode_chunk_for_index,
    episode_files,
    episode_row,
    is_project_dataset,
)

router = APIRouter(prefix="/api/v1/export", tags=["export"])

EXPORTS_DIR = STATE_ROOT / "exports"
EXPORT_TASKS_DIR = STATE_ROOT / "export_tasks"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tmp_root() -> Path:
    path = STATE_ROOT / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_download_stem(value: object, fallback: str = "egodata") -> str:
    """Keep project names readable in downloaded filenames.

    The old exporter used an ASCII-only whitelist, which silently removed
    Chinese project names and produced names such as ``D435-__3D_AI.zip``.
    Preserve Unicode while still preventing path separators and control
    characters from escaping the filename boundary.
    """
    name = str(value or "").strip()
    name = re.sub(r"[\x00-\x1f\x7f]+", "_", name)
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name or fallback


def _zip_batches(batch_dirs: list[Path], dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", allowZip64=True) as archive:
        for batch in batch_dirs:
            for path in batch.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(batch)
                if not rel.parts or rel.parts[0] not in {"data", "meta", "videos"}:
                    continue
                arcname = Path(batch.name) / rel
                compression = zipfile.ZIP_STORED if path.suffix.lower() in {".mp4", ".parquet", ".npz"} else zipfile.ZIP_DEFLATED
                archive.write(path, str(arcname), compress_type=compression)


def _zip_directory(root: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", allowZip64=True) as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, str(path.relative_to(root)),
                              compress_type=zipfile.ZIP_STORED if path.suffix.lower() in {".mp4", ".parquet"} else zipfile.ZIP_DEFLATED)


def _zip_lerobot(root: Path, dest: Path) -> None:
    """打包 LeRobot 标准结构(data/ + meta/ + videos/),不携带处理副产品
    (hand_3d/skeleton/calibration 等)。连接驱动导出带视频 feature 的
    数据集必须带上 videos/,否则查看器/官方加载器在归档里找不到视频
    文件(File not found in archive)。"""
    with zipfile.ZipFile(dest, "w", allowZip64=True) as archive:
        for sub in ("data", "meta", "videos"):
            sub_dir = root / sub
            if not sub_dir.is_dir():
                continue
            for path in sub_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, str(path.relative_to(root)),
                                  compress_type=zipfile.ZIP_STORED if path.suffix.lower() in {".mp4", ".parquet"} else zipfile.ZIP_DEFLATED)


def _cleanup_later(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _episode_dir(episode_id: str) -> Path:
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    return Path(ep["path"])


def _require_exportable_episode(episode_id: str) -> dict:
    """Keep Reviewing batches out of every download path, not only the UI."""
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    if ep.get("status") not in ("reviewed", "approved"):
        raise HTTPException(status_code=409, detail="Episode is still in Reviewing")
    return ep


def _build_zip(batches: list[Path], prefix: str, filename: str) -> FileResponse:
    tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".zip",
                                      dir=str(_tmp_root()), delete=False)
    tmp.close()
    try:
        _zip_batches(batches, Path(tmp.name))
    except Exception:
        _cleanup_later(tmp.name)
        raise HTTPException(status_code=500, detail="Failed to package episodes")
    return FileResponse(tmp.name, media_type="application/zip", filename=filename,
                        background=BackgroundTask(_cleanup_later, tmp.name))


def _zip_episode_selection(episode_ids: list[str], dest: Path) -> None:
    """Zip exactly the requested episodes from canonical project datasets.

    A project root can now contain hundreds of episodes.  Packaging the root
    directly would silently export every episode when the user requested one,
    and would also include unrelated episodes in a batch download.
    """
    groups: dict[Path, list[tuple[str, dict, dict]]] = {}
    for episode_id in episode_ids:
        ep = get_episode(str(episode_id))
        if ep is None:
            continue
        root = Path(ep["path"])
        if not is_project_dataset(root):
            continue
        row = episode_row(root, str(episode_id))
        if row is None:
            continue
        index = int(row.get("episode_index"))
        groups.setdefault(root, []).append(
            (str(episode_id), row, episode_files(root, index)))

    with zipfile.ZipFile(dest, "w", allowZip64=True) as archive:
        written: set[str] = set()

        def add_file(path: Path, arcname: str) -> None:
            if not path.is_file() or arcname in written:
                return
            compression = (zipfile.ZIP_STORED
                           if path.suffix.lower() in {".mp4", ".parquet", ".npz"}
                           else zipfile.ZIP_DEFLATED)
            archive.write(path, arcname, compress_type=compression)
            written.add(arcname)

        for root, entries in groups.items():
            selected_rows = [row for _episode_id, row, _files in entries]
            selected_indexes = {int(row.get("episode_index")) for row in selected_rows}
            for _episode_id, _row, files in entries:
                for path in files.get("data", []):
                    add_file(path, str(path.relative_to(root)))
                for _source, path in files.get("videos", []):
                    add_file(path, str(path.relative_to(root)))
                for path in files.get("meta", []):
                    meta_path = path if path.is_absolute() else root / path
                    rel = str(meta_path.relative_to(root))
                    if (rel not in {"meta/info.json", "meta/stats.json",
                                    "meta/tasks.json"}
                            and not rel.startswith("meta/episodes/")):
                        add_file(meta_path, rel)

            info = root / "meta" / "info.json"
            tasks = root / "meta" / "tasks.json"
            stats = root / "meta" / "stats.json"
            add_file(info, "meta/info.json")
            add_file(tasks, "meta/tasks.json")
            if selected_rows:
                import pyarrow as pa
                import pyarrow.parquet as pq
                rows_by_chunk: dict[int, list[dict]] = {}
                for row in selected_rows:
                    index = int(row.get("episode_index", 0))
                    rows_by_chunk.setdefault(episode_chunk_for_index(index), []).append(row)
                for chunk_index, chunk_rows in sorted(rows_by_chunk.items()):
                    for row in chunk_rows:
                        index = int(row.get("episode_index", 0))
                        sink = pa.BufferOutputStream()
                        pq.write_table(pa.Table.from_pylist([row]), sink)
                        archive.writestr(
                            f"meta/episodes/chunk-{chunk_index:03d}/"
                            f"episode_{index:06d}.parquet",
                            sink.getvalue().to_pybytes(),
                        )
            add_file(stats, "meta/stats.json")

        # Keep compatibility for pre-migration sessions.


def _build_episode_zip(episode_ids: list[str], prefix: str,
                       filename: str) -> FileResponse:
    tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".zip",
                                      dir=str(_tmp_root()), delete=False)
    tmp.close()
    try:
        _zip_episode_selection([str(value) for value in episode_ids], Path(tmp.name))
    except Exception:
        _cleanup_later(tmp.name)
        raise HTTPException(status_code=500, detail="Failed to package episodes")
    return FileResponse(tmp.name, media_type="application/zip", filename=filename,
                        background=BackgroundTask(_cleanup_later, tmp.name))


# ── 按工作流自动匹配导出格式 ─────────────────────────

def _episode_export_target(episode_id: str) -> dict | None:
    """Return the latest published export product for one episode.

    Export products live in the system-level export cache.  The project
    dataset itself remains only ``data/``, ``meta/`` and ``videos/``.
    """
    ep = get_episode(episode_id)
    if ep is None:
        return None
    runs = [r for r in list_runs() if r.get("episode_id") == episode_id]
    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    for run in runs:
        if run.get("status") != "completed":
            continue
        export_node = next(
            (n for n in (run.get("graph") or {}).get("nodes", [])
             if str((n.get("data") or {}).get("nodeType") or "").lower()
                in ("lerobot_export", "hdf5_export")),
            None,
        )
        if export_node is None:
            continue
        node_type = str((export_node.get("data") or {}).get("nodeType") or "").lower()
        cfg = (export_node.get("data") or {}).get("config") or {}

        # New runs publish export products to the system-level export cache
        # and store the published path in the run snapshot.  The source
        # project still contains only data/meta/videos.
        run_outputs = run.get("outputs") or {}
        published = run_outputs.get("artifacts", run_outputs)
        if isinstance(published, dict):
            for handles in published.values():
                if not isinstance(handles, dict):
                    continue
                for ref in handles.values():
                    if not isinstance(ref, dict):
                        continue
                    candidate = Path(str(ref.get("path") or ""))
                    if node_type == "hdf5_export":
                        if candidate.is_file() and candidate.suffix.lower() in {".h5", ".hdf5"}:
                            return {"kind": "hdf5", "version": None,
                                    "product": candidate, "root": candidate.parent}
                    elif (candidate / "meta" / "info.json").is_file():
                        return {"kind": "lerobot",
                                "version": str(cfg.get("version") or "v3.0"),
                                "product": candidate, "root": candidate}

    return None


def _workflow_export_version(episode_id: str) -> str | None:
    """Resolve the LeRobot version from the episode's bound workflow.

    Review-page exports intentionally do not have a separate format selector:
    the workflow's ``lerobot_export.version`` is the source of truth.  A
    completed run is used only as a fallback for older projects whose current
    binding no longer contains the export node.
    """
    episode = get_episode(str(episode_id))
    if episode is None:
        return None
    project = next((item for item in list_projects()
                    if item.get("name") == episode.get("project")), None)
    workflow_ids: list[str] = []
    if project:
        raw_ids = project.get("workflow_ids") or []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        if project.get("workflow_id"):
            raw_ids = [project.get("workflow_id"), *raw_ids]
        workflow_ids = [str(value) for value in raw_ids if str(value).strip()]
    for workflow_id in dict.fromkeys(workflow_ids):
        workflow = get_workflow(workflow_id)
        if not workflow:
            continue
        for node in (workflow.get("graph") or {}).get("nodes") or []:
            data = node.get("data") or {}
            if str(data.get("nodeType") or "").lower() != "lerobot_export":
                continue
            config = dict(data.get("config") or {})
            config.update((workflow.get("node_configs") or {}).get(
                node.get("id"), {}))
            version = str(config.get("version") or "v3.0").strip().lower()
            if version.startswith("v2"):
                return "v2.1"
            if version.startswith("v3"):
                return "v3.0"

    # Compatibility fallback: an older completed run still records the exact
    # export-node configuration used for that episode.
    target = _episode_export_target(str(episode_id))
    if target and target.get("kind") == "lerobot":
        version = str(target.get("version") or "").strip().lower()
        if version.startswith("v2"):
            return "v2.1"
        if version.startswith("v3"):
            return "v3.0"
    return None


@router.get("/episode-format/{episode_id}")
async def episode_export_format(episode_id: str):
    """前端按钮徽标:该批次下载时实际返回的格式。"""
    if get_episode(episode_id) is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    target = _episode_export_target(episode_id)
    if target is None:
        return {"episode_id": episode_id, "format": "raw", "version": None,
                "label": "Raw", "available": False}
    if target["kind"] == "hdf5":
        fmt, label = "hdf5", "HDF5"
    else:
        version = target["version"] or "v3.0"
        fmt = "lerobot_v2" if version.lower().startswith("v2") else "lerobot_v3"
        label = f"LeRobot {version}"
    return {"episode_id": episode_id, "format": fmt,
            "version": target["version"], "label": label, "available": True}


@router.get("/download-episode/{episode_id}")
async def download_single_episode(episode_id: str):
    """下载 = 按工作流导出格式返回最终数据集;无产物回退原始批次 zip。"""
    episode = _require_exportable_episode(episode_id)
    project_name = _safe_download_stem(episode.get("project"), episode_id)
    target = _episode_export_target(episode_id)
    if target is not None:
        if target["kind"] == "hdf5":
            return FileResponse(str(target["product"]),
                                media_type="application/x-hdf5",
                                filename=f"{project_name}.h5")
        tmp = tempfile.NamedTemporaryFile(prefix=f"egodata-{episode_id}-",
                                          suffix=".zip", dir=_tmp_root(),
                                          delete=False)
        tmp.close()
        _zip_lerobot(target["root"], Path(tmp.name))
        return FileResponse(tmp.name, media_type="application/zip",
                            filename=f"{project_name}.zip",
                            background=BackgroundTask(_cleanup_later, tmp.name))
    return _build_episode_zip([episode_id], f"egodata-{episode_id}-", f"{project_name}.zip")


@router.get("/download-reviewed")
async def download_reviewed_sessions():
    eps = [e for e in scan_sessions() if e.get("status") in ("reviewed", "approved")]
    deleted = {d["id"] for d in list_deleted_episodes()}
    eps = [e for e in eps if e["id"] not in deleted]
    if not eps:
        raise HTTPException(status_code=404, detail="No reviewed episodes")
    return _build_episode_zip([str(e["id"]) for e in eps], "egodata-reviewed-", "reviewed.zip")


@router.post("/batch-download")
async def batch_download_sessions(body: dict, _: str = Depends(verify_api_key)):
    ids = body.get("episode_ids") or []
    if not ids:
        raise HTTPException(status_code=400, detail="No episode_ids provided")
    episode_ids: list[str] = []
    project_names: set[str] = set()
    for ep_id in ids:
        episode = _require_exportable_episode(str(ep_id))
        episode_ids.append(str(ep_id))
        project = _safe_download_stem(episode.get("project"), "")
        if project:
            project_names.add(project)
    if not episode_ids:
        raise HTTPException(status_code=404, detail="No valid episodes")
    filename = f"{next(iter(project_names))}.zip" if len(project_names) == 1 else "egodata-batch.zip"
    return _build_episode_zip(episode_ids, "egodata-batch-", filename)


# ── 导出任务(JSON,后台打包)─────────────────────────

def _load_task(job_id: str) -> dict | None:
    return _read_json(EXPORT_TASKS_DIR / f"{job_id}.json", None)


def _save_task(task: dict) -> None:
    _write_json(EXPORT_TASKS_DIR / f"{task['id']}.json", task)


def _list_tasks() -> list[dict]:
    if not EXPORT_TASKS_DIR.is_dir():
        return []
    return [t for t in (_read_json(f, {}) for f in sorted(EXPORT_TASKS_DIR.glob("*.json"))) if t]


def _run_export_task(job_id: str, episode_ids: list[str], export_format: str,
                     dataset_name: str, split_ratio: float) -> None:
    try:
        blocked = [
            str(episode_id) for episode_id in episode_ids
            if (get_episode(str(episode_id)) or {}).get("status")
            not in ("reviewed", "approved")
        ]
        if blocked:
            raise RuntimeError("Some episodes are still in Reviewing")
        dest = EXPORTS_DIR / f"{job_id}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if export_format in {"lerobot_v2", "lerobot_v3"}:
            from app.lerobot_export import build_lerobot_dataset
            export_root = EXPORTS_DIR / f"{dataset_name}-{job_id[:8]}"
            version = "v2.1" if export_format == "lerobot_v2" else "v3.0"
            build_lerobot_dataset(dataset_name, episode_ids, export_root,
                                  split_ratio, version=version)
            _zip_directory(export_root, dest)
        else:
            _zip_episode_selection([str(e) for e in episode_ids], dest)
        task = _load_task(job_id)
        if task:
            task["status"] = "completed"
            task["output_dir"] = str(dest)
            task["finished_at"] = _utcnow()
            _save_task(task)
    except Exception as e:
        task = _load_task(job_id)
        if task:
            task["status"] = "failed"
            task["error"] = str(e)
            task["finished_at"] = _utcnow()
            _save_task(task)


@router.post("/start", status_code=202)
async def export_start(body: dict, background_tasks: BackgroundTasks,
                       _: str = Depends(verify_api_key)):
    requested_ids = body.get("episode_ids")
    if requested_ids:
        if not isinstance(requested_ids, list):
            raise HTTPException(status_code=400, detail="episode_ids must be a list")
        episode_ids = [str(episode_id) for episode_id in requested_ids]
        blocked = [
            episode_id for episode_id in episode_ids
            if (get_episode(episode_id) or {}).get("status")
            not in ("reviewed", "approved")
        ]
        if blocked:
            raise HTTPException(status_code=409,
                                detail="Some episodes are still in Reviewing")
    else:
        episode_ids = [
            e["id"] for e in scan_sessions()
            if e.get("status") in ("reviewed", "approved")
        ]
    if not episode_ids:
        raise HTTPException(status_code=404, detail="No reviewed episodes")
    requested_name = str(body.get("dataset_name") or "").strip()
    selected_projects = {
        _safe_download_stem((get_episode(episode_id) or {}).get("project"), "")
        for episode_id in episode_ids
    }
    selected_projects.discard("")
    # The review page does not choose a separate dataset name.  A single
    # project batch should download with that project's full Unicode name;
    # mixed-project selections keep a neutral batch name.
    if not requested_name or requested_name in {"egodata-batch", "egodata-export", "dataset"}:
        requested_name = (next(iter(selected_projects))
                          if len(selected_projects) == 1 else "egodata-batch")
    dataset_name = _safe_download_stem(requested_name, "egodata-export")
    requested_format = str(body.get("export_format") or "").strip().lower()
    if requested_format:
        export_format = requested_format
    else:
        # Review-page exports have no independent format selector. Resolve
        # each selected episode from its project's workflow export node.
        workflow_versions = {
            _workflow_export_version(episode_id)
            for episode_id in episode_ids
        }
        workflow_versions.discard(None)
        if len(workflow_versions) > 1:
            raise HTTPException(
                status_code=409,
                detail=("Selected episodes use different workflow export "
                        "versions; select episodes from one workflow/project."),
            )
        resolved_version = next(iter(workflow_versions), "v3.0")
        export_format = ("lerobot_v2" if resolved_version.startswith("v2")
                         else "lerobot_v3")
    if export_format not in {"lerobot_v2", "lerobot_v3"}:
        raise HTTPException(status_code=400,
                            detail="Unsupported export format")
    job = {
        "id": str(uuid4()),
        "dataset_name": dataset_name,
        "status": "running",
        "episode_ids": episode_ids,
        "split_ratio": body.get("split_ratio", 0.9),
        "progress": 0.0,
        "export_format": export_format,
        "error": None,
        "output_dir": None,
        "created_at": _utcnow(),
        "finished_at": None,
    }
    _save_task(job)
    background_tasks.add_task(
        _run_export_task,
        job["id"],
        episode_ids,
        job["export_format"],
        job["dataset_name"],
        float(job["split_ratio"] or 0.9),
    )
    return job


@router.get("/list")
async def export_list(limit: int = Query(20, ge=1, le=100), offset: int = 0):
    tasks = sorted(_list_tasks(), key=lambda t: t.get("created_at") or "", reverse=True)
    return tasks[offset:offset + limit]


@router.get("/{job_id}")
async def export_status(job_id: str):
    task = _load_task(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    return task


@router.get("/download/{job_id}")
async def download_export_job(job_id: str):
    task = _load_task(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    dest = EXPORTS_DIR / f"{job_id}.zip"
    if not dest.exists():
        raise HTTPException(status_code=404, detail="Export output not ready")
    return FileResponse(dest, media_type="application/zip",
                        filename=f"{task.get('dataset_name', 'dataset')}.zip")


@router.delete("/{job_id}")
async def delete_export_job(job_id: str, _: str = Depends(verify_api_key)):
    _remove_json(EXPORT_TASKS_DIR / f"{job_id}.json")
    try:
        (EXPORTS_DIR / f"{job_id}.zip").unlink(missing_ok=True)
    except Exception:
        pass
    return {"message": "Export job deleted"}


# ── Re-export:只重建导出,不重跑检测 ─────────────────────────
# 工作流改导出节点(配置/连线)后,本接口按"最新 completed run 的 canonical
# episode 数据 + 当前工作流的导出配置/连线"重建系统级导出缓存,并回写 run 快照
# 的导出节点配置
# (下载徽标/格式检测按快照驱动,回写后自动显示新格式)。

import asyncio

from app.localstore import save_run

_RE_TASKS: dict[str, dict] = {}
_RE_TASKS_DIR = STATE_ROOT / "re_export_tasks"

_PASSTHROUGH_TYPES = {
    "annotation", "human_review", "ai_annotation", "ai_quality_review",
}
_CAMERA_TYPES = {"rgb_camera", "fisheye_camera", "rgbd_camera", "stereo_camera",
                 "stereo_rgbd_camera", "mono_camera"}


def _re_set_task(task_id: str, **kw) -> None:
    t = _RE_TASKS.get(task_id)
    if t is None:
        return
    t.update(kw)
    try:
        ep = t.get("episode_id")
        if ep:
            _RE_TASKS_DIR.mkdir(parents=True, exist_ok=True)
            _write_json(_RE_TASKS_DIR / f"{ep}.json", t)
    except OSError:
        pass


def purge_episode_re_export_tasks(episode_id: str) -> None:
    """Cancel re-export jobs for a permanently deleted episode.

    Pop first so a still-running job's final _re_set_task call no-ops and
    cannot recreate the on-disk mirror; then remove the mirror file.
    """
    episode_id = str(episode_id)
    for task_id, task in list(_RE_TASKS.items()):
        if str(task.get("episode_id")) == episode_id:
            _RE_TASKS.pop(task_id, None)
    try:
        (_RE_TASKS_DIR / f"{episode_id}.json").unlink(missing_ok=True)
    except OSError:
        pass


def _export_spec_from_graph(graph: dict, node_configs: dict) -> dict | None:
    """从工作流图静态推导导出节点的输入范围(与 DAG 执行同语义):
    - video_keys:相机输入节点沿纯透传链(annotation/human_review/
      ai_annotation)连到导出节点 → 其 source_keys。处理节点
      (mediapipe 等)不透传视频,链式场景视频不进数据集(需求一致)
    - kinds:mediapipe_hand + RGB-only processors → hand_keypoints;
      RGB-D processors → hand_3d (+hand_3d#right for bare-hand multi-view);
      透传节点继续回溯
    图里没有 lerobot_export 节点返回 None。
    """
    nodes = {n.get("id"): n for n in graph.get("nodes") or []}
    edges = graph.get("edges") or []
    export_id = next((nid for nid, n in nodes.items()
                      if (n.get("data") or {}).get("nodeType") == "lerobot_export"),
                     None)
    if export_id is None:
        return None
    video_keys: list[str] = []
    kinds: set[str] = set()
    stack = [e.get("source") for e in edges if e.get("target") == export_id]
    visited: set[str] = set()
    while stack:
        nid = stack.pop()
        if not nid or nid in visited:
            continue
        visited.add(nid)
        node = nodes.get(nid)
        if node is None:
            continue
        t = canonical_node_type((node.get("data") or {}).get("nodeType") or "")
        cfg = dict((node.get("data") or {}).get("config") or {})
        cfg.update((node_configs or {}).get(nid, {}))
        if t in _CAMERA_TYPES:
            for k in (cfg.get("source_keys") or cfg.get("source_key") or "").split(","):
                k = k.strip()
                if k and k not in video_keys:
                    video_keys.append(k)
        elif t == "mediapipe_hand" or t in HAND_PROCESS_TYPES - RGBD_3D_TYPES:
            kinds.add("hand_keypoints")
        if t in RGBD_3D_TYPES:
            kinds.add("hand_3d")
            if t == "rgbd_to_3d_bare_hand":
                kinds.add("hand_3d#right")
        elif t in _PASSTHROUGH_TYPES:
            stack.extend(e.get("source") for e in edges
                         if e.get("target") == nid)
        # 其他节点类型不贡献导出输入,不回溯
    return {"video_keys": video_keys, "kinds": kinds}


async def _run_re_export(task_id: str, episode_id: str) -> None:
    from app.lerobot_export import build_lerobot_dataset
    try:
        _re_set_task(task_id, status="loading", detail="定位 run 与工作流")
        await asyncio.sleep(0)
        ep = get_episode(episode_id)
        if ep is None:
            raise RuntimeError("Episode not found")
        runs = [r for r in list_runs()
                if r.get("episode_id") == episode_id
                and r.get("status") == "completed" and r.get("created_at")]
        runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        run = runs[0] if runs else None
        if run is None:
            raise RuntimeError("无 completed run 记录(先跑一次工作流再重导出)")
        project_root = Path(ep["path"])
        source_row = episode_row(project_root, episode_id)
        if source_row is None:
            raise RuntimeError("canonical episode metadata 不存在")
        source_index = int(source_row.get("episode_index", 0))
        source_data = (project_root / "data"
                       / f"chunk-{episode_chunk_for_index(source_index):03d}"
                       / f"episode_{source_index:06d}.parquet")
        if not source_data.is_file():
            raise RuntimeError("canonical episode data parquet 不存在")
        # 当前工作流:优先项目**当前绑定**的工作流(用户改的连线/配置
        # 就生效在那里);run 的来源工作流仅作兜底(run 可能出自已被
        # 解绑的旧工作流,若优先它会拿到过期的连线/源键)。
        wf = None
        project = next((p for p in list_projects()
                        if p.get("name") == ep.get("project")), None)
        for wf_id in (project.get("workflow_ids") or []) if project else []:
            w = get_workflow(wf_id)
            if w is not None and _export_spec_from_graph(
                    w.get("graph") or {}, w.get("node_configs") or {}) is not None:
                wf = w
                break
        if wf is None:
            old_wf = get_workflow(run.get("workflow_id") or "")
            if old_wf is not None and _export_spec_from_graph(
                    old_wf.get("graph") or {},
                    old_wf.get("node_configs") or {}) is not None:
                wf = old_wf
        if wf is None:
            raise RuntimeError("项目绑定工作流里没有 LeRobot 导出节点")
        spec = _export_spec_from_graph(wf.get("graph") or {},
                                       wf.get("node_configs") or {})
        if spec is None:
            raise RuntimeError("当前工作流没有 LeRobot 导出节点")
        export_node = next(
            n for n in (wf.get("graph") or {}).get("nodes", [])
            if (n.get("data") or {}).get("nodeType") == "lerobot_export")
        cfg = dict((export_node.get("data") or {}).get("config") or {})
        cfg.update((wf.get("node_configs") or {}).get(export_node.get("id"), {}))
        version = str(cfg.get("version") or "v3.0")
        split_ratio = float(cfg.get("split_ratio", 0.9))
        # 检测结果已经合并进 canonical episode parquet,重导出不重跑检测。
        kinds = spec["kinds"]
        hand_keypoints_paths = ([str(source_data)]
                                if "hand_keypoints" in kinds else None)
        hand_3d_paths = ([str(source_data)] if "hand_3d" in kinds else None)
        hand_3d_right_paths = ([str(source_data)]
                               if "hand_3d#right" in kinds else None)
        include_video_keys = spec["video_keys"] or None  # None=全量(旧语义),[]=无视频
        _re_set_task(task_id, status="building",
                     detail=f"重建 LeRobot {version} 数据集中(复用检测产物,不重跑)")
        await asyncio.sleep(0)
        # 导出副本位于系统级缓存,不污染项目根目录。
        import shutil
        run_root = EXPORTS_DIR / str(run["id"]) / "dataset"
        shutil.rmtree(run_root, ignore_errors=True)
        run_root.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            build_lerobot_dataset, episode_id, [episode_id], run_root,
            split_ratio, include_video_keys,
            hand_keypoints_paths, hand_3d_paths, hand_3d_right_paths,
            version, None)
        # 回写 run 快照的导出节点配置 → 下载徽标/格式检测显示新配置。
        # 旧 run 快照可能没有导出节点(早期工作流无导出)→ 从当前工作流
        # 把导出节点(带新配置)补进快照图,保证 _episode_export_target
        # 能读到格式。
        patched = False
        run_graph = run.setdefault("graph", {})
        for n in (run_graph.get("nodes") or []):
            if (n.get("data") or {}).get("nodeType") == "lerobot_export":
                n.setdefault("data", {})["config"] = dict(
                    n.get("data", {}).get("config") or {})
                n["data"]["config"].update({
                    "version": version, "split_ratio": split_ratio})
                if (run.get("node_configs") or {}).get(n.get("id")) is not None:
                    run["node_configs"][n.get("id")].update({
                        "version": version, "split_ratio": split_ratio})
                patched = True
                break
        if not patched:
            node_copy = dict(export_node)
            node_copy["id"] = f"re_export_{node_copy.get('id')}"
            node_copy["data"] = dict(node_copy.get("data") or {})
            node_copy["data"]["config"] = dict(cfg)
            run_graph.setdefault("nodes", [])
            run_graph["nodes"].append(node_copy)
            export_output_node_id = node_copy["id"]
        else:
            export_output_node_id = export_node.get("id")
        run_outputs = run.setdefault("outputs", {})
        artifact_map = run_outputs.setdefault("artifacts", {})
        artifact_map.setdefault(export_output_node_id, {})["dataset"] = {
            "kind": "dataset",
            "path": str(run_root),
            "source_key": None,
            "schema_version": "1.0",
            "metadata": {"root": ".", "version": version},
        }
        save_run(run)
        _re_set_task(task_id, status="done",
                     detail=f"重建完成:LeRobot {version}"
                            + (f",视频:{len(spec['video_keys'])} 路(按连线)"
                               if spec["video_keys"]
                               else ",视频:自动识别全量")
                            + (f",骨骼列 {sorted(kinds)}" if kinds else ""))
    except Exception as exc:
        _re_set_task(task_id, status="failed", detail=str(exc))


@router.post("/re-export/{episode_id}")
async def re_export_episode(episode_id: str):
    """只重建导出:复用最新 completed run 的检测产物 + 当前工作流的
    导出节点配置/连线,原地重建 LeRobot 数据集并回写 run 快照。
    不重跑 mediapipe 等检测节点。"""
    if get_episode(episode_id) is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    task_id = str(uuid4())
    _RE_TASKS[task_id] = {"task_id": task_id, "episode_id": episode_id,
                          "status": "queued", "detail": ""}
    asyncio.get_running_loop().create_task(_run_re_export(task_id, episode_id))
    return {"task_id": task_id, "status": "queued"}


@router.get("/re-export/{episode_id}/status")
async def re_export_status(episode_id: str):
    for t in reversed(list(_RE_TASKS.values())):
        if t.get("episode_id") == episode_id:
            return t
    disk = _read_json(_RE_TASKS_DIR / f"{episode_id}.json", None)
    if disk is None:
        return {"episode_id": episode_id, "status": "idle"}
    if disk.get("status") in ("queued", "loading", "building"):
        disk["status"] = "interrupted"
        disk["detail"] = str(disk.get("detail") or "") + "(服务重启,任务中断,请重新触发)"
    return disk
