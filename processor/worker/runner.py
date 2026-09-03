from __future__ import annotations

import logging
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable

from app.config import settings
from app.processing import JobContext, ModuleSkip, ArtifactRef
from app.processing.registry import get as get_module
from app.workflow_types import migrate_graph_types
from worker.client import WorkerClient

logger = logging.getLogger("egodata.worker")


def _topo_sort(nodes: list[dict], edges: list[dict]) -> list[str]:
    degrees = {node.get("id"): 0 for node in nodes if node.get("id")}
    children = {node_id: [] for node_id in degrees}
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source in children and target in degrees:
            children[source].append(target)
            degrees[target] += 1
    ready = [node_id for node_id, degree in degrees.items() if degree == 0]
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for child in children[current]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    return order + [node_id for node_id in degrees if node_id not in order]


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe input archive path")
            target = (destination / relative).resolve()
            if root != target and root not in target.parents:
                raise ValueError("Unsafe input archive path")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)


def _zip_directory(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        for path in source.rglob("*"):
            if path.is_file():
                compression = zipfile.ZIP_STORED if path.suffix.lower() in {".mp4", ".parquet"} else zipfile.ZIP_DEFLATED
                archive.write(path, str(path.relative_to(source)), compress_type=compression)


def _incoming_artifacts(node_id: str, edges: list[dict], node_outputs: dict[str, dict[str, ArtifactRef]]) -> dict[str, ArtifactRef]:
    """Aggregate upstream artifacts for a node.

    三条规则(单边/单输出行为与旧实现逐位一致):
    1. 同一 targetHandle 多条边 → 后缀去重(video → video#1、video#2…),
       解决"双目左右目两条边指向同一识别节点,后边覆盖前边"的问题;
    2. 命中 sourceHandle 时,同源 "#" 兄弟键一并并入
       (mediapipe_hand 多路输出 hand_keypoints / hand_keypoints#1 的场景);
    3. sourceHandle 未命中(透传链,如 annotation/reviewed 在透传输出中
       不存在)→ 平铺 source 的全部输出,而不是只取第一个。
    """
    inputs: dict[str, ArtifactRef] = {}
    for edge in edges:
        if edge.get("target") != node_id:
            continue
        source_outputs = node_outputs.get(edge.get("source"), {})
        source_handle = edge.get("sourceHandle")
        target_handle = edge.get("targetHandle") or "data"
        if source_handle and source_handle in source_outputs:
            # 规则 2:主键 + "#" 兄弟键(如 hand_keypoints, hand_keypoints#1)
            refs = [source_outputs[source_handle]] + [
                r for k, r in source_outputs.items()
                if k.startswith(source_handle + "#")]
        elif len(source_outputs) == 1:
            refs = [next(iter(source_outputs.values()))]  # 现状行为
        elif source_outputs:
            # 规则 3:透传链平铺全部输出
            refs = list(source_outputs.values())
        else:
            continue
        seen: set[tuple] = set()
        for ref in refs:
            dedup_key = (ref.kind, str(ref.path), ref.source_key)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            # 规则 1:同 targetHandle 冲突 → 后缀去重,边序确定性
            key, n = target_handle, 1
            while key in inputs:
                key = f"{target_handle}#{n}"
                n += 1
            inputs[key] = ref
    return inputs


def execute_job(
    job: dict,
    input_root: Path,
    output_root: Path,
    heartbeat: Callable[[float, dict], None] | None = None,
) -> tuple[dict, dict]:
    graph, _ = migrate_graph_types(job.get("graph") or {})
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_map = {node.get("id"): node for node in nodes}
    node_outputs: dict[str, dict[str, ArtifactRef]] = {}
    node_states: dict[str, dict] = {}
    output_manifest: dict = {"artifacts": {}}
    order = _topo_sort(nodes, edges)

    for index, node_id in enumerate(order):
        node = node_map.get(node_id)
        if not node:
            continue
        data = node.get("data") or {}
        node_type = data.get("nodeType", "")
        config = dict(data.get("config") or {})
        incoming = _incoming_artifacts(node_id, edges, node_outputs)
        node_states[node_id] = {"type": node_type, "status": "running", "progress": 0.0}
        if heartbeat:
            heartbeat(index / max(1, len(order)), node_states)

        # 统一分发:按 node_type 查模块注册表执行(可插拔,新增模块无需改这里)
        module = get_module(node_type)
        if module is None:
            raise ValueError(f"Unsupported worker module: {node_type}")

        def node_progress(p: float):
            if heartbeat:
                heartbeat((index + p) / max(1, len(order)), node_states)

        ctx = JobContext(
            node_id=node_id, node_type=node_type, config=config, job=job,
            input_root=input_root, output_root=output_root,
            incoming=incoming, progress=node_progress, node_states=node_states,
        )
        try:
            outputs = module.run(ctx)
        except ModuleSkip as exc:
            # 模块主动跳过(无有效输入)—— 标记 skipped,不失败,继续后续节点
            node_states[node_id] = {"type": node_type, "status": "skipped",
                                    "note": exc.reason}
            continue
        if outputs is None:
            node_states[node_id] = {"type": node_type, "status": "skipped"}
            continue

        node_outputs[node_id] = outputs
        output_manifest["artifacts"][node_id] = {key: ref.to_dict() for key, ref in outputs.items()}
        warnings = []
        for ref in outputs.values():
            for warning in (ref.metadata or {}).get("processing_warnings") or []:
                if warning and warning not in warnings:
                    warnings.append(str(warning))
        state = {"type": node_type, "status": "completed", "progress": 1.0}
        if warnings:
            state["warnings"] = warnings
        node_states[node_id] = state
        if heartbeat:
            heartbeat((index + 1) / max(1, len(order)), node_states)

    return node_states, output_manifest


def process_one(client: WorkerClient, job: dict, worker_id: str, work_dir: Path) -> None:
    run_id = str(job["run_id"])
    lease_token = str(job.get("lease_token") or "")
    # ignore_cleanup_errors:Windows 上 OpenCV 句柄可能短暂占用文件,
    # 清理失败不应让已完成的 job 报错。
    with tempfile.TemporaryDirectory(prefix=f"egodata-{run_id}-", dir=str(work_dir),
                                     ignore_cleanup_errors=True) as temp:
        root = Path(temp)
        input_zip = root / "input.zip"
        input_root = root / "input"
        output_root = root / "outputs"
        client.download_input(run_id, input_zip)
        _safe_extract(input_zip, input_root)

        def heartbeat(progress: float, states: dict):
            # 心跳失败不中断处理(后端热加载/瞬时故障时继续跑),
            # 否则 MediaPipe 进度回调异常会中断整个 job。
            try:
                client.heartbeat(run_id, worker_id, lease_token,
                                 min(0.99, progress), states)
            except Exception as exc:
                # A 409 means this worker lost the lease to a newer attempt.
                # Continuing would waste GPU time and could let a stale worker
                # submit results after the newer attempt has started.
                if getattr(getattr(exc, "response", None), "status_code", None) == 409:
                    raise RuntimeError("Worker lease lost; aborting stale attempt") from exc
                logger.warning("Heartbeat failed (continuing): %s", exc)

        states, manifest = execute_job(job, input_root, output_root, heartbeat)
        result_zip = root / "outputs.zip"
        _zip_directory(output_root, result_zip)
        client.complete(run_id, worker_id, lease_token, result_zip, states, manifest)


def run_forever(server_url: str, api_key: str, worker_id: str, device: str = "auto", poll_seconds: float = 2.0, work_dir: Path | None = None):
    work_dir = work_dir or settings.temp_root / "worker"
    work_dir.mkdir(parents=True, exist_ok=True)
    client = WorkerClient(server_url, api_key)
    try:
        while True:
            job = None
            try:
                job = client.claim(worker_id, ["mediapipe", "hand_keypoints", "video_overlay"], device)
                if job is None:
                    time.sleep(poll_seconds)
                    continue
                logger.info("Claimed workflow run %s", job["run_id"])
                process_one(client, job, worker_id, work_dir)
                logger.info("Completed workflow run %s", job["run_id"])
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.exception("Worker job failed: %s", exc)
                if "job" in locals() and job:
                    try:
                        client.fail(str(job["run_id"]), worker_id,
                                    str(job.get("lease_token") or ""),
                                    str(exc), retry=True)
                    except Exception:
                        logger.exception("Failed to report worker failure")
                time.sleep(poll_seconds)
    finally:
        client.close()
