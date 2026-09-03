"""Project-level LeRobot 2.1 dataset storage.

The active dataset root is ``sessions/<project>``.  Individual uploads are
episodes in the same ``data/meta/videos`` tree; they are not nested under a
second batch directory.  Workflow outputs are merged into the corresponding
episode parquet files; preview overlays are rendered in the browser and are
never stored as a second video.
There is deliberately no ``processed/``, ``meta/processing/`` or
``state/runs/`` directory inside a project.  This module is shared by the
migration command and the upload path so both produce the same layout.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.lerobot_v21 import (
    DEPTH_MAX_MM,
    DEPTH_MIN_MM,
    DEPTH_QMAX,
    DEPTH_QP,
    DEPTH_VIDEO_ENCODING,
    _source_feature,
    encode_depth_png_sequence,
    iter_video_streams,
    normalize_metadata_sources,
    normalize_extracted_dataset,
    source_key_from_video,
)


_EPISODE_RE = re.compile(r"episode_(\d+)", re.IGNORECASE)
_CHUNK_RE = re.compile(r"chunk[-_](\d+)", re.IGNORECASE)
_CANONICAL_ROOTS = {"data", "meta", "videos"}
PROJECT_CHUNK_SIZE = 1000
EPISODES_METADATA_CHUNK_SIZE = 1000


def _episode_parquet_files(project_root: Path) -> list[Path]:
    """Return one canonical Parquet index file per episode."""
    root = Path(project_root) / "meta" / "episodes"
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.glob("chunk-*/episode_*.parquet")
        if path.is_file()
    )


def has_episode_index(project_root: Path) -> bool:
    """Whether a project has the canonical Parquet episode index."""
    root = Path(project_root)
    return bool(_episode_parquet_files(root))


def write_project_episode_index(project_root: Path,
                                rows: Iterable[dict[str, Any]]) -> None:
    """Write the canonical sharded Parquet episode index.

    Each Parquet file contains exactly one episode row.  Episode files are
    grouped into chunks of 1000 and use the same episode number as data/video.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    project_root = Path(project_root)
    normalized = []
    for row in rows:
        item = dict(row)
        # Arrow cannot serialize an empty struct.  Missing per-episode stats
        # are valid; the project-level stats.json writer will compute a
        # fallback from the frame parquet when needed.
        if item.get("stats") == {}:
            item.pop("stats", None)
        normalized.append(item)
    normalized.sort(key=lambda row: int(row.get("episode_index", 10**9)))
    episodes_root = project_root / "meta" / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    for path in sorted(episodes_root.rglob("*.parquet")):
        path.unlink(missing_ok=True)
    for path in sorted(episodes_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    for row in normalized:
        episode_index = int(row.get("episode_index", 0))
        chunk_index = episode_index // EPISODES_METADATA_CHUNK_SIZE
        target = (episodes_root / f"chunk-{chunk_index:03d}"
                  / f"episode_{episode_index:06d}.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row]), target)

    # The legacy flat JSONL index is intentionally not generated anymore.
    (project_root / "meta" / "episodes.jsonl").unlink(missing_ok=True)


def is_project_dataset(root: Path) -> bool:
    root = Path(root)
    return ((root / "data").is_dir()
            and has_episode_index(root)
            and (root / "videos").is_dir()
            and (root / "meta" / "info.json").is_file()
            and (root / "meta" / "stats.json").is_file()
            and (root / "meta" / "tasks.json").is_file())


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_tasks_json(path: Path) -> list[dict[str, Any]]:
    """Read the compact internal task index stored as one JSON document."""
    value = _read_json(path, None)
    if isinstance(value, dict):
        value = value.get("tasks", value)
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            rows.append({"task_index": index, "task": item})
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("task") or item.get("description") or "").strip()
        if not text:
            continue
        try:
            task_index = int(item.get("task_index", item.get("task_id", index)))
        except (TypeError, ValueError):
            task_index = index
        rows.append({"task_index": task_index, "task": text})
    return rows


def _write_tasks_json(project_root: Path,
                      rows: Iterable[dict[str, Any]]) -> None:
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = str(row.get("task") or row.get("description") or "").strip()
        if not text:
            continue
        try:
            task_index = int(row.get("task_index", row.get("task_id", index)))
        except (TypeError, ValueError):
            task_index = index
        normalized.append({"task_index": task_index, "task": text})
    normalized.sort(key=lambda row: int(row["task_index"]))
    path = Path(project_root) / "meta" / "tasks.json"
    _write_json(path, normalized)
    (Path(project_root) / "meta" / "tasks.jsonl").unlink(missing_ok=True)


def _summary_count(summary: dict[str, Any]) -> int:
    value = summary.get("count", 1)
    if isinstance(value, list):
        value = value[0] if value else 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _as_numeric_array(value: Any):
    import numpy as np
    return np.asarray(value, dtype=np.float64)


def _aggregate_episode_stats(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-episode summaries into LeRobot-style global stats."""
    import numpy as np
    accumulated: dict[str, dict[str, Any]] = {}
    for row in rows:
        summaries = row.get("stats") if isinstance(row, dict) else None
        if not isinstance(summaries, dict):
            continue
        for name, summary in summaries.items():
            if not isinstance(summary, dict):
                continue
            try:
                count = _summary_count(summary)
                minimum = _as_numeric_array(summary["min"])
                maximum = _as_numeric_array(summary["max"])
                mean = _as_numeric_array(summary["mean"])
                std = _as_numeric_array(summary.get("std", 0.0))
                if not (minimum.shape == maximum.shape == mean.shape == std.shape):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            item = accumulated.get(name)
            if item is None:
                accumulated[name] = {
                    "min": minimum.copy(), "max": maximum.copy(),
                    "sum": mean * count, "m2": (std ** 2) * count,
                    "count": count,
                }
                continue
            if item["sum"].shape != mean.shape:
                continue
            old_count = int(item["count"])
            total = old_count + count
            delta = mean - item["sum"] / old_count
            item["m2"] = (item["m2"] + (std ** 2) * count
                           + delta ** 2 * old_count * count / total)
            item["sum"] += mean * count
            item["min"] = np.minimum(item["min"], minimum)
            item["max"] = np.maximum(item["max"], maximum)
            item["count"] = total

    result: dict[str, dict[str, Any]] = {}
    for name, item in accumulated.items():
        count = int(item["count"])
        mean = item["sum"] / count
        variance = np.maximum(item["m2"] / count, 0.0)
        values = {
            "min": item["min"], "max": item["max"],
            "mean": mean, "std": np.sqrt(variance), "count": [count],
        }
        normalized_values = {}
        for key, value in values.items():
            if hasattr(value, "ndim"):
                normalized_values[key] = (
                    float(value) if value.ndim == 0 else value.tolist()
                )
            else:
                normalized_values[key] = value
        result[name] = normalized_values
    return result


def _compute_data_stats(project_root: Path) -> dict[str, dict[str, Any]]:
    """Compute fallback global statistics when old per-episode stats are absent."""
    import numpy as np
    import pandas as pd
    columns: dict[str, list[Any]] = {}
    for path in sorted((Path(project_root) / "data").rglob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        for name in frame.columns:
            try:
                array = np.asarray(frame[name].tolist(), dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if array.ndim == 0 or array.ndim > 2:
                continue
            columns.setdefault(str(name), []).append(array.reshape(len(frame), -1))
    result: dict[str, dict[str, Any]] = {}
    for name, arrays in columns.items():
        try:
            array = np.concatenate(arrays, axis=0)
            finite = np.where(np.isfinite(array), array, np.nan)
            if not np.isfinite(finite).any():
                continue
            stats = {
                "min": np.nanmin(finite, axis=0),
                "max": np.nanmax(finite, axis=0),
                "mean": np.nanmean(finite, axis=0),
                "std": np.nanstd(finite, axis=0),
                "count": [int(np.isfinite(finite).sum())],
            }
            normalized_stats = {}
            for key, value in stats.items():
                if hasattr(value, "size"):
                    normalized_stats[key] = (
                        float(value.reshape(-1)[0])
                        if value.size == 1 else value.tolist()
                    )
                else:
                    normalized_stats[key] = value
            result[name] = normalized_stats
        except (TypeError, ValueError):
            continue
    return result


def write_project_stats(project_root: Path,
                        rows: Iterable[dict[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    stats = _aggregate_episode_stats(rows)
    if not stats:
        stats = _compute_data_stats(project_root)
    _write_json(Path(project_root) / "meta" / "stats.json", stats)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _episode_stats_frame_count(row: dict[str, Any]) -> int | None:
    """Return the frame count advertised by one v2.1 stats row."""
    stats = row.get("stats")
    if not isinstance(stats, dict):
        return None
    frame_stats = stats.get("frame_index")
    if not isinstance(frame_stats, dict):
        return None
    count = frame_stats.get("count")
    if isinstance(count, list) and count:
        try:
            return int(count[0])
        except (TypeError, ValueError):
            return None
    return None


def _repair_episode_stats(stats_rows: list[dict[str, Any]],
                          episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Align v2.1 per-episode statistics with the canonical episode indexes.

    Older migrations occasionally preserved the old ``episode_index`` in a
    stats row, or omitted a row altogether.  The frame count is a stable
    second key, so use it to recover a unique match and create an empty stats
    row when the source did not provide statistics for an episode.
    """
    candidates = [dict(row) for row in stats_rows if isinstance(row, dict)]
    used: set[int] = set()
    repaired: list[dict[str, Any]] = []
    ordered_episodes = sorted(
        episode_rows,
        key=lambda row: int(row.get("episode_index", 10**9)),
    )
    for episode in ordered_episodes:
        target_index = int(episode["episode_index"])
        expected_length = int(episode.get("length") or 0)
        chosen_index: int | None = None

        # Prefer an already-correct row when its frame count agrees.
        for index, candidate in enumerate(candidates):
            if index in used or _episode_index(candidate.get("episode_index")) != target_index:
                continue
            count = _episode_stats_frame_count(candidate)
            if count is None or count == expected_length:
                chosen_index = index
                break

        # If the index is stale, recover by a unique frame-count match.
        if chosen_index is None and expected_length:
            matches = [
                index for index, candidate in enumerate(candidates)
                if index not in used
                and _episode_stats_frame_count(candidate) == expected_length
            ]
            if len(matches) == 1:
                chosen_index = matches[0]

        if chosen_index is None:
            repaired.append({"episode_index": target_index, "stats": {}})
            continue

        used.add(chosen_index)
        candidate = candidates[chosen_index]
        candidate["episode_index"] = target_index
        stats = candidate.get("stats")
        if isinstance(stats, dict) and isinstance(stats.get("episode_index"), dict):
            index_stats = stats["episode_index"]
            for key in ("min", "max", "mean"):
                if key in index_stats:
                    index_stats[key] = float(target_index)
        repaired.append(candidate)
    return repaired


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_file() and source.is_file() and target.stat().st_size == source.stat().st_size:
            return
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def clone_tree(source: Path, target: Path) -> None:
    """Clone a tree with hard links when possible.

    Hard links keep the migration recoverable without temporarily doubling a
    large depth-PNG dataset.  Filesystems without hard-link support fall back
    to normal copies.
    """
    source = Path(source)
    target = Path(target)
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        dest = target / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _link_or_copy(path, dest)


def _episode_index(value: Any) -> int | None:
    try:
        number = int(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def episode_index_from_name(name: str) -> int | None:
    match = re.search(r"_(\d{6})$", str(name or ""))
    return int(match.group(1)) if match else None


def project_episode_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _episode_parquet_files(project_root):
        try:
            import pyarrow.parquet as pq
            values = pq.read_table(path).to_pylist()
        except (ImportError, OSError, ValueError):
            values = []
        rows.extend(value for value in values if isinstance(value, dict))
    rows.sort(key=lambda row: int(row.get("episode_index", 10**9)))
    return rows


def _match_episode_row(rows: list[dict[str, Any]],
                       episode_id: str) -> dict[str, Any] | None:
    """Match an episode row by canonical id first, then trailing index.

    Same semantics as :func:`episode_row` but over a caller-supplied row
    list, so a second parquet read is avoided when rows are already loaded.
    """
    wanted = str(episode_id)
    for row in rows:
        if str(row.get("episode_id") or row.get("source_batch") or "") == wanted:
            return row
    index = episode_index_from_name(wanted)
    if index is not None:
        for row in rows:
            if _episode_index(row.get("episode_index")) == index:
                return row
    return None


def episode_row(project_root: Path, episode_id: str) -> dict[str, Any] | None:
    return _match_episode_row(project_episode_rows(project_root), episode_id)


def _episode_id_for_row(row: dict[str, Any], project_name: str) -> str:
    return str(
        row.get("episode_id")
        or row.get("source_batch")
        or f"{project_name}_{int(row.get('episode_index', 0)):06d}"
    )


def _chunk_index(value: Any) -> int:
    try:
        number = int(value)
        return max(0, number)
    except (TypeError, ValueError):
        return 0


def episode_chunk_for_index(episode_index: int,
                            chunks_size: int = PROJECT_CHUNK_SIZE) -> int:
    """Map the 0-based episode number to a LeRobot v2.1 chunk.

    Episodes 000000--000999 are stored in ``chunk-000``; 001000 starts
    ``chunk-001``.  The result is based on the stable dataset index, never
    on upload order or on the number of legacy directories found.
    """
    size = max(1, int(chunks_size or PROJECT_CHUNK_SIZE))
    return max(0, int(episode_index) // size)


def _episode_number_from_path(path: Path) -> int | None:
    match = _EPISODE_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _chunk_number_from_path(path: Path) -> int:
    for part in path.parts:
        match = _CHUNK_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return 0


def episode_files(project_root: Path, episode_index: int) -> dict[str, Any]:
    """Return only one episode's source files from a project dataset."""
    root = Path(project_root)
    data_files = [
        path for path in (root / "data").rglob("*.parquet")
        if path.is_file() and _episode_number_from_path(path) == episode_index
    ] if (root / "data").is_dir() else []
    videos: list[tuple[str, Path]] = []
    for source, path in iter_video_streams(root / "videos"):
        if _episode_number_from_path(path) == episode_index:
            videos.append((source, path))
    row = next((item for item in project_episode_rows(root)
                if _episode_index(item.get("episode_index")) == episode_index), {})
    episode_id = _episode_id_for_row(row, root.name)
    relative_meta: list[Path] = [Path("meta/info.json"), Path("meta/tasks.json")]
    return {
        "episode_index": episode_index,
        "episode_id": episode_id,
        "row": row,
        "data": data_files,
        "videos": videos,
        "meta": [path for path in relative_meta if (root / path).is_file()],
    }


def _processing_token(value: Any, fallback: str = "result") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return text or fallback


def _processing_artifacts(outputs: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Flatten the worker output manifest into ``(node, handle, ref)`` rows."""
    if not isinstance(outputs, dict):
        return []
    values = outputs.get("artifacts", outputs)
    if not isinstance(values, dict):
        return []
    result: list[tuple[str, str, dict[str, Any]]] = []
    for node_id, handles in values.items():
        if not isinstance(handles, dict):
            continue
        for handle, ref in handles.items():
            if isinstance(ref, dict):
                result.append((str(node_id), str(handle), dict(ref)))
    return result


def _processing_json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
    return value if value is not None else default


def _processing_ref_path(extracted_root: Path, ref: dict[str, Any]) -> Path | None:
    """Resolve a worker reference without allowing it to escape the result zip."""
    raw = str(ref.get("path") or "").strip()
    if not raw:
        return None
    root = Path(extracted_root).resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(frame: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_processing_frame(data_frame: Any, result_frame: Any,
                            column_names: list[str]) -> Any:
    """Merge a processing parquet onto one canonical episode by frame index."""
    import pandas as pd

    if "frame_index" not in data_frame.columns:
        raise ValueError("canonical episode has no frame_index column")
    if "frame_index" not in result_frame.columns:
        raise ValueError("processing result has no frame_index column")
    result = data_frame.copy()
    result_index = result["frame_index"]
    source = result_frame.drop_duplicates("frame_index").set_index("frame_index")
    lookup = source.reindex(result_index.tolist())
    for old_name, new_name in column_names:
        if old_name not in source.columns:
            continue
        # Keep the canonical row order and permit sparse detectors: missing
        # result frames become null rather than shifting another frame.
        values = lookup[old_name].reset_index(drop=True)
        result[new_name] = values
    return result


def _publish_export_artifact(project_root: Path, run_id: str,
                             ref_path: Path, ref: dict[str, Any]) -> Path:
    """Keep export products outside the three-directory dataset contract."""
    export_root = (Path(project_root).parent.parent / "state" / "exports"
                   / _processing_token(run_id, "run"))
    metadata = ref.get("metadata") or {}
    relative_root = str(metadata.get("root") or "").strip()
    if relative_root and relative_root != ".":
        candidate = ref_path.parent
        # The module stores ``root`` relative to its node output directory.
        # Walk up until the declared dataset root is visible.
        for parent in [ref_path.parent, *ref_path.parents]:
            if (parent / "meta" / "info.json").is_file():
                candidate = parent
                break
    elif (ref_path.parent / "data").is_dir() and (ref_path.parent / "meta").is_dir():
        candidate = ref_path.parent
    else:
        candidate = ref_path
    if candidate.is_dir():
        destination = export_root / "dataset"
        if destination.exists():
            shutil.rmtree(destination)
        clone_tree(candidate, destination)
        return destination
    destination = export_root / ref_path.name
    _atomic_copy(candidate, destination)
    return destination


def publish_processing_result(
    project_root: Path,
    episode_id: str,
    run_id: str,
    extracted_root: Path,
    outputs: dict[str, Any],
    node_states: dict[str, Any] | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Merge one completed worker result into the canonical episode.

    Processing parquet columns are written to ``data/.../episode_*.parquet``;
    skeleton/preview videos are deliberately not persisted; the episode-level
    run summary is written to its row in
    ``meta/episodes/.../episode_*.parquet``.  No project-local processing
    directory or processing manifest is created.
    """
    import pandas as pd

    root = Path(project_root)
    row = episode_row(root, str(episode_id))
    if row is None:
        raise RuntimeError(f"Cannot publish processing result: episode not found: {episode_id}")
    episode_index = int(row.get("episode_index", 0))
    chunk_index = episode_chunk_for_index(episode_index)
    data_path = (root / "data" / f"chunk-{chunk_index:03d}"
                 / f"episode_{episode_index:06d}.parquet")
    meta_path = (root / "meta" / "episodes" / f"chunk-{chunk_index:03d}"
                 / f"episode_{episode_index:06d}.parquet")
    if not data_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"Canonical episode files are incomplete: {episode_id}")

    data_frame = pd.read_parquet(data_path)
    old_columns = _processing_json(row.get("processing_columns"), [])
    if not isinstance(old_columns, list):
        old_columns = []
    data_frame = data_frame.drop(
        columns=[str(name) for name in old_columns if str(name) in data_frame.columns],
        errors="ignore",
    )

    records = _processing_artifacts(outputs)
    tabular_records = [item for item in records
                       if str(item[2].get("kind") or "") in
                       {"hand_keypoints", "hand_3d", "glove_sensor"}]
    # A reprocess is authoritative for its hand-3D sources.  Remove any
    # source-qualified hand-3D columns left by an earlier partial publish
    # (including legacy ``*_1`` collision columns) before merging the fresh
    # left/right results.  Other processing kinds remain untouched.
    if any(str(ref.get("kind") or "") == "hand_3d"
           for _node, _handle, ref in records):
        stale_hand3d = [str(name) for name in data_frame.columns
                        if str(name).startswith("processing.hand_3d.")]
        if stale_hand3d:
            data_frame = data_frame.drop(columns=stale_hand3d, errors="ignore")
    assigned_columns: set[str] = set(data_frame.columns)
    merged_columns: list[str] = []
    published = {"artifacts": {}}
    merged_hand3d_sources: set[str] = set()

    for node_id, handle, ref in records:
        source_path = _processing_ref_path(Path(extracted_root), ref)
        if source_path is None:
            # Input passthrough references point into the worker input cache and
            # are not part of result_zip.  They remain in the run state but
            # are not treated as newly published processing files.
            continue
        kind = str(ref.get("kind") or "")
        metadata = dict(ref.get("metadata") or {})
        published_ref = dict(ref)
        source = _processing_token(ref.get("source_key") or metadata.get("source_key")
                                   or handle, kind)
        # Review/export nodes pass the node-36 hand-3D refs through again.
        # They are the same source data, not a second measurement. Merge each
        # source once and keep the passthrough artifact only as an audit link.
        if kind == "hand_3d" and source in merged_hand3d_sources:
            published_ref["path"] = str(data_path.relative_to(root)).replace("\\", "/")
            published_ref["metadata"] = {
                **metadata, "merged_into": published_ref["path"],
                "deduplicated": True, "merged_columns": [],
            }
            published["artifacts"].setdefault(node_id, {})[handle] = published_ref
            continue
        if kind == "hand_3d":
            merged_hand3d_sources.add(source)
        if kind == "hand_3d" and not (
                bool(metadata.get("metric_3d_available"))
                or str(metadata.get("unit") or "") in {"camera_meters", "meters"}
                or str(metadata.get("mode") or "").lower()
                in {"depth_hand_3d", "d435_depth_lifted", "depth_lifted"}):
            # RGB_TO_3D/other RGB-only estimators may create a temporary
            # spatial preview artifact.  It is not real depth and must not be
            # persisted in the LeRobot source data.
            published_ref["path"] = None
            published_ref["metadata"] = {**metadata, "preview_only": True,
                                          "storage": "browser_overlay"}
            published["artifacts"].setdefault(node_id, {})[handle] = published_ref
            continue
        if kind in {"hand_keypoints", "hand_3d", "glove_sensor"}:
            result_frame = pd.read_parquet(source_path)
            raw_columns = [str(name) for name in result_frame.columns
                           if name not in {"frame_index", "episode_index"}]
            # A propagated canonical reference can already contain several
            # hand-3D source namespaces.  When that happens, publish only the
            # namespace belonging to this artifact ref and strip it back to
            # the raw fields before applying the canonical name below.
            source_columns: list[tuple[str, str]] = []
            if kind == "hand_3d":
                prefix = f"processing.hand_3d.{source}."
                namespaced = [name for name in raw_columns
                              if name.startswith(prefix)]
                raw_names = {name[len(prefix):] for name in namespaced}
                source_columns = [
                    (name, name[len(prefix):]) for name in namespaced
                    if not (name[len(prefix):].endswith("_1")
                            and name[len(prefix):-2] in raw_names)
                ]
            if not source_columns:
                source_columns = [(name, name) for name in raw_columns]
            column_names: list[tuple[str, str]] = []
            for source_name, raw_name in source_columns:
                # Hand-3D is produced once per RGB view.  Always retain the
                # source namespace, even for the first result, otherwise the
                # first stereo view stays as plain ``hand_0_*`` columns and
                # the second view is namespaced only because of a collision.
                # That makes it impossible for the review UI to choose the
                # view with more valid landmarks deterministically.
                public_name = (f"processing.{kind}.{source}.{raw_name}"
                               if kind == "hand_3d" else raw_name)
                if public_name in assigned_columns:
                    public_name = f"processing.{kind}.{source}.{raw_name}"
                while public_name in assigned_columns:
                    public_name = f"{public_name}_1"
                assigned_columns.add(public_name)
                column_names.append((source_name, public_name))
                merged_columns.append(public_name)
            data_frame = _merge_processing_frame(data_frame, result_frame, column_names)
            published_ref["path"] = str(data_path.relative_to(root)).replace("\\", "/")
            published_ref["metadata"] = {
                **metadata, "merged_into": published_ref["path"],
                "merged_columns": [name for _raw, name in column_names],
            }
        elif kind == "video" and (metadata.get("skeleton")
                                   or "skeleton" in handle.lower()
                                   or metadata.get("render_video")):
            # Skeleton videos are preview-only.  The browser overlays the
            # merged keypoints on the original RGB stream, so persisting a
            # second rendered video would duplicate large source files.
            published_ref["path"] = None
            published_ref["metadata"] = {**metadata, "preview_only": True,
                                          "storage": "browser_overlay"}
        elif kind in {"dataset", "hdf5", "export"}:
            target = _publish_export_artifact(root, run_id, source_path, ref)
            published_ref["path"] = str(target)
            published_ref["metadata"] = {**metadata, "published_path": str(target)}
        else:
            # Unknown artifacts are not copied into the dataset tree.  Keeping
            # their manifest entry is useful for diagnostics and preserves
            # forward compatibility with new modules.
            published_ref["path"] = str(ref.get("path") or "")
        published["artifacts"].setdefault(node_id, {})[handle] = published_ref

    _atomic_write_parquet(data_frame, data_path)
    row = dict(row)
    row.update({
        "processing_run_id": str(run_id),
        "processing_status": "completed",
        "processing_completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
        "processing_node_states": json.dumps(node_states or {}, ensure_ascii=False,
                                               separators=(",", ":")),
        "processing_outputs": json.dumps({
            "artifacts": published.get("artifacts") or {},
        }, ensure_ascii=False, separators=(",", ":")),
        "processing_columns": json.dumps(merged_columns, ensure_ascii=False,
                                          separators=(",", ":")),
    })
    rows = [row if int(item.get("episode_index", -1)) == episode_index else item
            for item in project_episode_rows(root)]
    write_project_episode_index(root, rows)
    return published


def _task_rows(root: Path, fallback: str) -> list[dict[str, Any]]:
    rows = _read_tasks_json(root / "meta" / "tasks.json")
    if not rows:
        rows = _read_jsonl(root / "meta" / "tasks.jsonl")
    if rows:
        return rows
    return [{"task_index": 0, "task": fallback or "default_recording"}]


def _task_text(row: dict[str, Any], fallback: str) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return str(row.get("task") or fallback or "default_recording")


def _merge_unique_dicts(existing: list[dict[str, Any]], additions: Any,
                       keys: tuple[str, ...]) -> list[dict[str, Any]]:
    result = [dict(item) for item in existing if isinstance(item, dict)]
    values = additions if isinstance(additions, list) else []
    seen = {tuple(str(item.get(key) or "") for key in keys) for item in result}
    for item in values:
        if not isinstance(item, dict):
            continue
        marker = tuple(str(item.get(key) or "") for key in keys)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(dict(item))
    return result


def _merge_info(project_root: Path, source_info: dict[str, Any], rows: list[dict[str, Any]],
                tasks: list[dict[str, Any]]) -> dict[str, Any]:
    target = _read_json(project_root / "meta" / "info.json", {})
    if not isinstance(target, dict):
        target = {}
    target = normalize_metadata_sources(target)
    source_info = normalize_metadata_sources(source_info)
    if not target:
        target = dict(source_info)
    features = dict(target.get("features") or {})
    features.update(dict(source_info.get("features") or {}))
    target.update({
        "format": "lerobot_v2.1",
        "codebase_version": "v2.1",
        "total_episodes": len(rows),
        "total_frames": sum(int(row.get("length") or 0) for row in rows),
        "total_tasks": len(tasks),
        "total_chunks": len({episode_chunk_for_index(
            int(row.get("episode_index"))) for row in rows
            if _episode_index(row.get("episode_index")) is not None}),
        "chunks_size": PROJECT_CHUNK_SIZE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.{ext}",
        "features": features,
        "extensions": {
            **(target.get("extensions") or {}),
            "episodes_file": "meta/episodes/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "stats_file": "meta/stats.json",
            "tasks_file": "meta/tasks.json",
            "project_storage": "sessions/<project>/{data,meta,videos}",
            "processing_storage": "data+meta/episodes+videos",
        },
    })
    target["extensions"].pop("calibration_root", None)
    target["extensions"].pop("episodes_stats_file", None)
    target["extensions"].pop("episodes_compat_file", None)
    if source_info.get("fps") and not target.get("fps"):
        target["fps"] = source_info["fps"]
    target["devices"] = _merge_unique_dicts(
        target.get("devices") or [], source_info.get("devices"), ("key",),
    )
    names = dict(target.get("device_names") or {})
    names.update(dict(source_info.get("device_names") or {}))
    target["device_names"] = names
    cameras = dict(target.get("cameras") or {})
    cameras.update(dict(source_info.get("cameras") or {}))
    target["cameras"] = cameras
    sensors = list(dict.fromkeys(
        [str(value) for value in (target.get("sensors") or [])]
        + [str(value) for value in (source_info.get("sensors") or [])]
    ))
    target["sensors"] = sensors
    return target


def _read_episode_sidecars(episode_root: Path, episode_id: str) -> dict[str, Any]:
    """Inline calibration and collector provenance into the episode row."""
    def drop_empty_maps(value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                cleaned = drop_empty_maps(child)
                if cleaned is not None:
                    result[key] = cleaned
            return result or None
        if isinstance(value, list):
            return [cleaned for child in value
                    if (cleaned := drop_empty_maps(child)) is not None]
        return value

    result: dict[str, Any] = {}
    for kind in ("calibration", "collector"):
        base = Path(episode_root) / "meta" / kind
        candidate = base / str(episode_id)
        if not candidate.is_dir():
            candidate = base
        if not candidate.is_dir():
            continue
        values: dict[str, Any] = {}
        for path in sorted(candidate.rglob("*.json")):
            value = _read_json(path, None)
            if value is not None:
                if isinstance(value, dict):
                    value = normalize_metadata_sources(value)
                value = drop_empty_maps(value)
                if value is not None:
                    values[str(path.relative_to(candidate))] = value
        if values:
            result[kind] = values
    return result


def _copy_data_with_episode_index(source: Path, target: Path,
                                  episode_index: int) -> None:
    """Copy a parquet and normalize its internal episode index if present."""
    try:
        import pandas as pd
        frame_table = pd.read_parquet(source)
        if "episode_index" not in frame_table.columns:
            _link_or_copy(source, target)
            return
        values = frame_table["episode_index"].dropna().unique().tolist()
        if len(values) == 1 and int(values[0]) == int(episode_index):
            _link_or_copy(source, target)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        frame_table["episode_index"] = int(episode_index)
        frame_table.to_parquet(target, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to normalize episode_index in {source.name}: {exc}"
        ) from exc


def merge_normalized_episode(project_root: Path, episode_root: Path,
                             episode_id: str, chunk_index: int,
                             replace: bool = False,
                             target_episode_index: int | None = None) -> dict[str, Any]:
    """Merge one normalized episode into a project-level dataset root."""
    project_root = Path(project_root)
    episode_root = Path(episode_root)
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "meta").mkdir(parents=True, exist_ok=True)
    source_info = _read_json(episode_root / "meta" / "info.json", {})
    source_rows = project_episode_rows(episode_root)
    source_row = source_rows[0] if source_rows else {}
    source_episode_index = _episode_index(source_row.get("episode_index"))
    if source_episode_index is None:
        source_episode_index = episode_index_from_name(episode_id)
    episode_index = (int(target_episode_index)
                     if target_episode_index is not None
                     else source_episode_index)
    if episode_index is None:
        raise RuntimeError(f"Cannot determine episode_index for {episode_id}")
    # Re-apply the canonical rule here even when a migration caller supplied
    # a temporary chunk number based on directory order.
    chunk_index = episode_chunk_for_index(episode_index)

    existing_rows = project_episode_rows(project_root)
    old = next((row for row in existing_rows
                if _episode_index(row.get("episode_index")) == episode_index
                or str(row.get("episode_id") or "") == str(episode_id)), None)
    if old is not None and not replace:
        raise RuntimeError(f"Episode already exists: {episode_id}")
    if old is not None:
        old_index = existing_rows.index(old)
        existing_rows.pop(old_index)
        old_chunk = _chunk_index(old.get("data/chunk_index"))
        old_data = project_root / "data" / f"chunk-{old_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        old_data.unlink(missing_ok=True)
        for path in (project_root / "videos").rglob(f"episode_{episode_index:06d}.*"):
            path.unlink(missing_ok=True)
        # Sidecar metadata is embedded into the episode row below, so a
        # replacement cannot leave an obsolete per-episode metadata folder.

    source_data = [path for path in (episode_root / "data").rglob("*.parquet")
                   if path.is_file()] if (episode_root / "data").is_dir() else []
    if not source_data:
        raise RuntimeError(f"Normalized episode has no data parquet: {episode_id}")
    source_data.sort()
    data_target = project_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
    _copy_data_with_episode_index(source_data[0], data_target, episode_index)

    source_videos = iter_video_streams(episode_root / "videos")
    for source, video_path in source_videos:
        destination = (project_root / "videos"
                       / f"observation.images.{source}"
                       / f"chunk-{chunk_index:03d}"
                       / f"episode_{episode_index:06d}{video_path.suffix.lower()}")
        _link_or_copy(video_path, destination)

    source_task_rows = _task_rows(episode_root, str(source_info.get("task_name") or episode_id))
    target_tasks = _task_rows(project_root, str(source_info.get("task_name") or episode_id))
    task_by_text = {_task_text(row, ""): int(row.get("task_index", index))
                    for index, row in enumerate(target_tasks)}
    next_task = max(task_by_text.values(), default=-1) + 1
    task_text = _task_text(source_row, str(source_info.get("task_name") or episode_id))
    if task_text not in task_by_text:
        task_by_text[task_text] = next_task
        next_task += 1
    target_tasks = [{"task_index": index, "task": text}
                    for text, index in sorted(task_by_text.items(), key=lambda item: item[1])]

    row = dict(source_row)
    if not isinstance(row.get("stats"), dict):
        source_stats = _read_jsonl(episode_root / "meta" / "episodes_stats.jsonl")
        source_stats_row = next(
            (item for item in source_stats
             if _episode_index(item.get("episode_index")) == source_episode_index),
            None,
        )
        if isinstance(source_stats_row, dict):
            row["stats"] = source_stats_row.get("stats") or {}
    row.update(_read_episode_sidecars(episode_root, str(episode_id)))
    row.update({
        "episode_index": episode_index,
        "episode_id": str(episode_id),
        "source_batch": str(episode_id),
        "data/chunk_index": int(chunk_index),
        "data/file_index": int(episode_index),
        "task_index": int(task_by_text[task_text]),
        "tasks": [task_text],
    })
    for key in list(row):
        if key.endswith("/chunk_index"):
            row[key] = int(chunk_index)
        elif key.endswith("/file_index"):
            row[key] = int(episode_index)
    previous_frames = sum(int(item.get("length") or 0) for item in existing_rows)
    length = int(row.get("length") or 0)
    row["dataset_from_index"] = previous_frames
    row["dataset_to_index"] = previous_frames + length
    existing_rows.append(row)
    existing_rows.sort(key=lambda item: int(item.get("episode_index", 10**9)))
    write_project_episode_index(project_root, existing_rows)

    write_project_stats(project_root, existing_rows)
    _write_tasks_json(project_root, target_tasks)
    info = _merge_info(project_root, source_info, existing_rows, target_tasks)
    _write_json(project_root / "meta" / "info.json", info)
    return {
        "episode_id": str(episode_id),
        "episode_index": episode_index,
        "chunk_index": int(chunk_index),
        "frame_count": length,
        "video_sources": sorted(source for source, _path in source_videos),
        "path": str(project_root),
    }


def verify_project_dataset(project_root: Path) -> dict[str, Any]:
    root = Path(project_root)
    rows = project_episode_rows(root)
    errors: list[str] = []
    unexpected = sorted(
        path.name for path in root.iterdir()
        if path.name not in _CANONICAL_ROOTS and not path.name.startswith(".")
    ) if root.is_dir() else []
    errors.extend(f"unexpected project root entry: {name}" for name in unexpected)
    expected_meta = {"info.json", "stats.json", "tasks.json", "episodes"}
    meta_root = root / "meta"
    if meta_root.is_dir():
        unexpected_meta = sorted(
            path.name for path in meta_root.iterdir()
            if path.name not in expected_meta and not path.name.startswith(".")
        )
        errors.extend(f"unexpected meta entry: {name}" for name in unexpected_meta)
    for row in rows:
        index = _episode_index(row.get("episode_index"))
        if index is None:
            errors.append("episode row without episode_index")
            continue
        chunk = _chunk_index(row.get("data/chunk_index"))
        expected_chunk = episode_chunk_for_index(index)
        if chunk != expected_chunk:
            errors.append(
                f"wrong chunk for episode {index:06d}: {chunk:03d} != {expected_chunk:03d}"
            )
        data_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{index:06d}.parquet"
        if not data_path.is_file():
            errors.append(f"missing data: {data_path.relative_to(root)}")
        if not any(_episode_number_from_path(path) == index
                   for path in (root / "videos").rglob("*") if path.is_file()):
            errors.append(f"missing video for episode {index:06d}")
    return {
        "passed": not errors and bool(rows),
        "episodes": len(rows),
        "data_files": sum(1 for path in (root / "data").rglob("*.parquet") if path.is_file())
                      if (root / "data").is_dir() else 0,
        "video_files": sum(1 for path in (root / "videos").rglob("*") if path.is_file())
                       if (root / "videos").is_dir() else 0,
        "errors": errors,
    }


def _prune_empty_artifact_dirs(project_root: Path) -> None:
    """Remove emptied chunk/camera directories under data/ and videos/.

    Never removes ``data``/``videos`` themselves; the project must remain a
    valid skeleton for future uploads.
    """
    root = Path(project_root)
    for base in ("data", "videos"):
        base_dir = root / base
        if not base_dir.is_dir():
            continue
        for path in sorted(base_dir.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()


def _reindex_dataset_ranges(rows: list[dict[str, Any]]) -> None:
    """Re-derive contiguous dataset_from_index/dataset_to_index.

    Official LeRobot v2.1 keeps a dense global frame order.  Episode indexes
    stay stable (files are named after them); only the dataset ranges are
    compacted after a deletion.
    """
    rows.sort(key=lambda row: int(row.get("episode_index", 10**9)))
    cursor = 0
    for row in rows:
        length = int(row.get("length") or 0)
        row["dataset_from_index"] = cursor
        cursor += length
        row["dataset_to_index"] = cursor


def _reconcile_task_rows(rows: list[dict[str, Any]],
                         existing_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild the tasks.json index from surviving rows.

    Keeps the old task_index↔text mapping for texts that still appear, so
    unrelated tasks never shift; drops tasks no surviving episode references
    (an empty project yields an empty tasks.json), and fixes each row's
    ``task_index`` in place before the index rewrite.
    """
    surviving_texts = {_task_text(row, "default_recording") for row in rows}
    text_index: dict[str, int] = {
        str(task["task"]): int(task["task_index"])
        for task in existing_tasks
        if isinstance(task, dict) and task.get("task")
        and str(task["task"]) in surviving_texts
    }
    next_index = max(text_index.values(), default=-1) + 1
    for row in rows:
        text = _task_text(row, "default_recording")
        if text not in text_index:
            text_index[text] = next_index
            next_index += 1
        row["task_index"] = text_index[text]
    return sorted(
        ({"task_index": index, "task": text} for text, index in text_index.items()),
        key=lambda item: int(item["task_index"]),
    )


def delete_project_episode(project_root: Path | str,
                           episode_id: str) -> dict[str, Any]:
    """Permanently remove one episode and every artifact from a project dataset.

    Removes its data parquet and all per-source videos, drops its row from
    the canonical meta/episodes index, recomputes contiguous dataset frame
    ranges, rebuilds meta/stats.json, meta/tasks.json and meta/info.json
    from the surviving rows, and prunes emptied chunk directories.

    Episode indexes stay stable (gaps are valid); only dataset ranges are
    compacted.  Idempotent: any artifact already gone is skipped.  Raises
    LookupError when the id matches neither a row nor an index-shaped name.
    """
    root = Path(project_root)
    info0 = _read_json(root / "meta" / "info.json", {})
    if not isinstance(info0, dict):
        info0 = {}
    tasks0 = _read_tasks_json(root / "meta" / "tasks.json")

    rows = project_episode_rows(root)
    row0 = _match_episode_row(rows, episode_id)
    index = _episode_index(row0.get("episode_index")) if row0 is not None else None
    if index is None:
        index = episode_index_from_name(episode_id)
    if index is None:
        raise LookupError(f"Episode not found in project dataset: {episode_id}")

    removed_data: list[str] = []
    if (root / "data").is_dir():
        for path in sorted((root / "data").rglob("*.parquet")):
            if path.is_file() and _episode_number_from_path(path) == index:
                path.unlink(missing_ok=True)
                removed_data.append(str(path))

    removed_videos: list[str] = []
    if (root / "videos").is_dir():
        for path in sorted((root / "videos").rglob(f"episode_{index:06d}.*")):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed_videos.append(str(path))

    _prune_empty_artifact_dirs(root)

    if row0 is not None:
        rows.remove(row0)
    _reindex_dataset_ranges(rows)
    tasks = _reconcile_task_rows(rows, tasks0)

    # Same rebuild tail as merge/repack: index → stats → tasks → info.
    write_project_episode_index(root, rows)
    write_project_stats(root, rows)
    _write_tasks_json(root, tasks)
    info = _merge_info(root, info0, rows, tasks)
    _write_json(root / "meta" / "info.json", info)

    verification = verify_project_dataset(root)
    # With zero surviving rows verify reports passed=False by design
    # (bool(rows)); an empty project is still a successful deletion.
    if rows and not verification["passed"]:
        raise RuntimeError(
            f"Episode delete verification failed: {verification['errors']}")

    return {
        "episode_id": str(episode_id),
        "episode_index": index,
        "row_removed": row0 is not None,
        "removed_data": removed_data,
        "removed_videos": removed_videos,
        "remaining": len(rows),
        "verification": verification,
    }


def repack_project_chunks(project_root: Path, *, keep_backup: bool = True) -> dict[str, Any]:
    """Rebuild canonical ``data`` and ``videos`` chunk directories.

    This is used for projects migrated before the 1000-episode rule was
    enforced.  Only the canonical payload and the three supported metadata
    files are rebuilt.  The old active tree is kept as a recoverable backup
    until the caller explicitly removes it.
    """
    project_root = Path(project_root)
    if not is_project_dataset(project_root):
        return {"changed": False, "project": project_root.name,
                "verification": verify_project_dataset(project_root)}
    rows = project_episode_rows(project_root)
    if not rows:
        return {"changed": False, "project": project_root.name,
                "verification": verify_project_dataset(project_root)}

    storage_root = project_root.parent.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temp_root = project_root.parent / f".{project_root.name}.repack-{stamp}"
    backup_root = (storage_root / ".backups" / "session-migrations"
                   / f"{project_root.name}-repack-{stamp}")
    shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=False)
    mappings: list[dict[str, Any]] = []
    try:
        for name in _CANONICAL_ROOTS:
            clone_tree(project_root / name, temp_root / name)

        # Remove canonical payload files before rebuilding them.  Metadata is
        # reduced to the active contract below; no collector/calibration/depth
        # sidecar directory survives in the new project root.
        for path in sorted((temp_root / "data").rglob("*.parquet"), reverse=True):
            path.unlink(missing_ok=True)
        for path in sorted((temp_root / "videos").rglob("*"), reverse=True):
            if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
                path.unlink(missing_ok=True)
        for root in (temp_root / "data", temp_root / "videos"):
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()

        rebuilt_rows: list[dict[str, Any]] = []
        actual_sources: set[str] = set()
        for new_index, row in enumerate(rows):
            old_index = _episode_index(row.get("episode_index"))
            if old_index is None:
                raise RuntimeError("Cannot repack an episode without episode_index")
            old_chunk = _chunk_index(row.get("data/chunk_index"))
            old_data = (project_root / "data" / f"chunk-{old_chunk:03d}"
                        / f"episode_{old_index:06d}.parquet")
            if not old_data.is_file():
                raise RuntimeError(f"Missing data parquet: {old_data}")
            new_chunk = episode_chunk_for_index(new_index)
            new_data = (temp_root / "data" / f"chunk-{new_chunk:03d}"
                        / f"episode_{new_index:06d}.parquet")
            _copy_data_with_episode_index(old_data, new_data, new_index)
            for video in (project_root / "videos").rglob("*"):
                if not video.is_file() or video.suffix.lower() not in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
                    continue
                if _episode_number_from_path(video) != old_index:
                    continue
                source = source_key_from_video(video, project_root / "videos")
                actual_sources.add(source)
                target = (temp_root / "videos"
                          / f"observation.images.{source}"
                          / f"chunk-{new_chunk:03d}"
                          / f"episode_{new_index:06d}{video.suffix.lower()}")
                _link_or_copy(video, target)
            updated = dict(row)
            updated["episode_index"] = new_index
            updated["data/chunk_index"] = new_chunk
            for key in list(updated):
                if key.endswith("/chunk_index"):
                    updated[key] = new_chunk
                elif key.endswith("/file_index"):
                    updated[key] = new_index
            rebuilt_rows.append(updated)
            mappings.append({"old_episode_index": old_index,
                             "episode_index": new_index,
                             "old_chunk": old_chunk,
                             "chunk_index": new_chunk})

        rebuilt_rows.sort(key=lambda item: int(item.get("episode_index", 10**9)))
        info = _read_json(temp_root / "meta" / "info.json", {})
        if not isinstance(info, dict):
            info = {}
        declared_sources = sorted(
            str(key)[len("observation.images."):]
            for key in (info.get("features") or {})
            if str(key).startswith("observation.images.")
        )
        source_aliases: dict[str, str] = {}
        if len(declared_sources) == len(actual_sources):
            source_aliases = dict(zip(declared_sources, sorted(actual_sources)))
        elif len(actual_sources) == 1:
            # Older migrations may have left both a stale declared key and
            # the real on-disk key in info.json.  The canonical video folder
            # is authoritative for the normalized dataset.
            actual = next(iter(actual_sources))
            source_aliases = {
                source: actual for source in declared_sources if source != actual
            }
        if source_aliases:
            def rename_keys(value: Any, *, feature_keys: bool = False) -> Any:
                if not isinstance(value, dict):
                    return value
                result: dict[Any, Any] = {}
                for key, item in value.items():
                    raw_key = str(key)
                    prefix = "observation.images."
                    if feature_keys and raw_key.startswith(prefix):
                        source = raw_key[len(prefix):]
                        new_key = prefix + source_aliases.get(source, source)
                    else:
                        new_key = source_aliases.get(raw_key, raw_key)
                    # If both the stale key and the real key are present,
                    # keep the real-key value and drop the duplicate.
                    if key in source_aliases and new_key in value:
                        continue
                    result[new_key] = item
                return result
            info["features"] = rename_keys(info.get("features") or {}, feature_keys=True)
            info["cameras"] = rename_keys(info.get("cameras") or {})
            info["device_names"] = rename_keys(info.get("device_names") or {})
            info["video_extensions"] = rename_keys(info.get("video_extensions") or {})
            for device in info.get("devices") or []:
                if isinstance(device, dict):
                    device["slots"] = list(dict.fromkeys(
                        source_aliases.get(str(slot), str(slot))
                        for slot in device.get("slots") or []
                    ))
                    device["resolution"] = rename_keys(device.get("resolution") or {})
                    device["fps"] = rename_keys(device.get("fps") or {})
            for row in rebuilt_rows:
                for key in list(row):
                    for old, new in source_aliases.items():
                        prefix = f"videos/observation.images.{old}/"
                        if key.startswith(prefix):
                            row[key.replace(prefix, f"videos/observation.images.{new}/", 1)] = row.pop(key)
                            break
            for path in (temp_root / "meta" / "collector").glob("*/metadata.json"):
                metadata = _read_json(path, {})
                if not metadata:
                    continue
                metadata["cameras"] = rename_keys(metadata.get("cameras") or {})
                # The collector may number its metadata from 1 while the
                # canonical LeRobot files use a 0-based episode index.
                collector_id = path.parent.name
                row_for_metadata = next(
                    (item for item in rebuilt_rows
                     if _episode_id_for_row(item, project_root.name) == collector_id),
                    None,
                )
                if row_for_metadata is not None:
                    metadata["episode_index"] = int(row_for_metadata["episode_index"])
                    metadata["chunk_index"] = int(
                        row_for_metadata.get("data/chunk_index") or 0
                    )
                    metadata["file_index"] = int(row_for_metadata["data/file_index"])
                for device in metadata.get("devices") or []:
                    if isinstance(device, dict):
                        device["slots"] = [source_aliases.get(str(slot), str(slot))
                                            for slot in device.get("slots") or []]
                        device["resolution"] = rename_keys(device.get("resolution") or {})
                        device["fps"] = rename_keys(device.get("fps") or {})
                _write_json(path, metadata)
        # A v2.1 reader must never advertise a stream that has no corresponding
        # file.  This handles older migrations where ``features`` retained a
        # stale ``head_right_rgb_4`` key after the actual folder became
        # ``head_right_rgb``.
        actual_sources = sorted(actual_sources)
        feature_map = info.get("features") or {}
        if isinstance(feature_map, dict):
            stream_features: dict[str, Any] = {}
            for key, value in feature_map.items():
                raw_key = str(key)
                if not raw_key.startswith("observation.images."):
                    stream_features[raw_key] = value
                    continue
                source = raw_key[len("observation.images."):]
                source = source_aliases.get(source, source)
                if source in actual_sources:
                    stream_features[f"observation.images.{source}"] = value
            info["features"] = stream_features
        for field in ("cameras", "device_names", "video_extensions"):
            value = info.get(field)
            if isinstance(value, dict):
                info[field] = {
                    source_aliases.get(str(key), str(key)): item
                    for key, item in value.items()
                    if source_aliases.get(str(key), str(key)) in actual_sources
                }
        for device in info.get("devices") or []:
            if isinstance(device, dict):
                device["slots"] = [
                    source_aliases.get(str(slot), str(slot))
                    for slot in device.get("slots") or []
                    if source_aliases.get(str(slot), str(slot)) in actual_sources
                ]
        write_project_episode_index(temp_root, rebuilt_rows)
        write_project_stats(temp_root, rebuilt_rows)
        # Replace the cloned info before merging so stale source aliases from
        # an earlier migration cannot be reintroduced by _merge_info().
        _write_json(temp_root / "meta" / "info.json", info)
        tasks = _task_rows(temp_root, project_root.name)
        info = _merge_info(temp_root, info,
                           rebuilt_rows, tasks)
        info["total_chunks"] = len({episode_chunk_for_index(
            int(row["episode_index"])) for row in rebuilt_rows})
        _write_json(temp_root / "meta" / "info.json", info)
        _write_tasks_json(temp_root, tasks)
        meta_root = temp_root / "meta"
        for item in list(meta_root.iterdir()):
            if item.name in {"info.json", "stats.json", "tasks.json", "episodes"}:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)
        verification = verify_project_dataset(temp_root)
        if not verification["passed"]:
            raise RuntimeError(f"Chunk repack verification failed: {verification['errors']}")

        backup_root.parent.mkdir(parents=True, exist_ok=True)
        project_root.rename(backup_root)
        try:
            temp_root.rename(project_root)
        except Exception:
            if backup_root.exists():
                backup_root.rename(project_root)
            raise
        if not keep_backup:
            shutil.rmtree(backup_root, ignore_errors=True)
        return {"changed": True, "project": project_root.name,
                "backup": str(backup_root) if keep_backup else None,
                "mappings": mappings,
                "verification": verify_project_dataset(project_root)}
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def migrate_project_depth_pngs(project_root: Path, *, keep_backup: bool = True) -> dict[str, Any]:
    """Convert an already-canonical project's PNG depth sidecars to HEVC.

    Older project migrations kept ``meta/depth/<episode_id>/<source>/*.png``
    for compatibility.  The current contract stores logarithmic depth codes
    beside RGB streams as CQP HEVC ``gray12le`` video.  Conversion is completed and
    verified for every sequence before the PNG tree is moved to a recoverable
    backup.
    """
    project_root = Path(project_root)
    depth_root = project_root / "meta" / "depth"
    if not depth_root.is_dir():
        return {"changed": False, "project": project_root.name,
                "reason": "no meta/depth directory"}
    if not is_project_dataset(project_root):
        raise RuntimeError(f"Not a canonical project dataset: {project_root}")

    rows = project_episode_rows(project_root)
    rows_by_id = {
        _episode_id_for_row(row, project_root.name): row for row in rows
    }
    rows_by_index = {
        int(row["episode_index"]): row for row in rows
        if _episode_index(row.get("episode_index")) is not None
    }

    sequences: list[tuple[str, Path, list[Path], dict[str, Any]]] = []

    def add_sequence(episode_id: str | None, source_dir: Path) -> None:
        frames = sorted(
            path for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
            and path.stem.isdigit()
        )
        if not frames:
            return
        numbers = [int(path.stem) for path in frames]
        expected = list(range(numbers[0], numbers[0] + len(numbers)))
        if numbers != expected:
            raise RuntimeError(
                f"Depth frame sequence has gaps for {source_dir}: "
                f"expected consecutive files, got {frames[0].name}..{frames[-1].name}"
            )
        row = rows_by_id.get(str(episode_id or "")) if episode_id else None
        if row is None and episode_id and str(episode_id).isdigit():
            row = rows_by_index.get(int(episode_id))
        if row is None and len(rows) == 1:
            row = rows[0]
        if row is None:
            raise RuntimeError(
                f"Cannot map depth directory {source_dir} to an episode row"
            )
        sequences.append((str(source_dir.name), source_dir, frames, row))

    # Canonical migrated data is namespaced per episode.  Also accept the
    # single-episode flat form produced by early v2.1 migrations.
    direct = [path for path in sorted(depth_root.iterdir()) if path.is_dir()
              and any(p.is_file() and p.suffix.lower() == ".png"
                      and p.stem.isdigit() for p in path.iterdir())]
    if direct:
        for source_dir in direct:
            add_sequence(None, source_dir)
    for episode_dir in sorted(depth_root.iterdir()):
        if not episode_dir.is_dir() or episode_dir in direct:
            continue
        for source_dir in sorted(episode_dir.iterdir()):
            if source_dir.is_dir():
                add_sequence(episode_dir.name, source_dir)
    if not sequences:
        return {"changed": False, "project": project_root.name,
                "reason": "no numeric PNG depth sequences"}

    # Duplicate source+episode pairs would overwrite one another and are not
    # safe to infer.  Stop before touching the active dataset in that case.
    markers = [(int(row["episode_index"]), source)
               for source, _directory, _frames, row in sequences]
    if len(markers) != len(set(markers)):
        raise RuntimeError("Duplicate depth source for the same episode")

    info_path = project_root / "meta" / "info.json"
    info = _read_json(info_path, {})
    features = dict(info.get("features") or {})
    extensions = dict(info.get("extensions") or {})
    video_extensions = dict(info.get("video_extensions") or {})
    output_records: list[dict[str, Any]] = []
    fps_default = float(info.get("fps") or 30.0)
    for source, depth_dir, frames, row in sequences:
        episode_index = int(row["episode_index"])
        chunk_index = episode_chunk_for_index(episode_index)
        destination = (project_root / "videos"
                       / f"observation.images.{source}"
                       / f"chunk-{chunk_index:03d}"
                       / f"episode_{episode_index:06d}.mp4")
        probe = encode_depth_png_sequence(
            depth_dir, frames, destination, fps=fps_default,
        )
        feature = _source_feature(info, source, probe, True)
        video_info = feature.setdefault("video_info", {})
        video_info.update({
            "video.channels": 1,
            "video.is_depth_map": True,
            "video.is_depth_visualization": False,
            "video.depth_scale": 0.001,
            "video.depth_encoding": DEPTH_VIDEO_ENCODING,
            "video.depth_min_mm": DEPTH_MIN_MM,
            "video.depth_max_mm": DEPTH_MAX_MM,
            "video.depth_qmax": DEPTH_QMAX,
            "video.depth_qp": DEPTH_QP,
            "video.depth_quantization": "log",
            "has_audio": False,
        })
        feature["shape"] = [int(probe["height"]), int(probe["width"]), 1]
        features[f"observation.images.{source}"] = feature
        video_extensions[source] = "mp4"
        key = f"videos/observation.images.{source}/"
        count = int(probe.get("frames") or len(frames))
        fps = float(probe.get("fps") or fps_default)
        row[f"videos/observation.images.{source}/chunk_index"] = chunk_index
        row[f"videos/observation.images.{source}/file_index"] = episode_index
        row[f"videos/observation.images.{source}/from_timestamp"] = 0.0
        row[f"videos/observation.images.{source}/to_timestamp"] = (
            count / fps if fps else 0.0
        )
        output_records.append({
            "episode_id": _episode_id_for_row(row, project_root.name),
            "episode_index": episode_index,
            "source": source,
            "frames": count,
            "path": str(destination.relative_to(project_root)),
            "codec": probe.get("codec"),
            "pix_fmt": probe.get("pix_fmt"),
        })

    # Only metadata after all output videos were verified.  The old PNG tree
    # is moved, not deleted, so rollback remains possible.
    info["features"] = features
    info["video_extensions"] = video_extensions
    info["format"] = "lerobot_v2.1"
    info["codebase_version"] = "v2.1"
    info["chunks_size"] = PROJECT_CHUNK_SIZE
    info["extensions"] = {
        **extensions,
        "depth_storage": "videos/observation.images.<source>/chunk-{episode_chunk:03d}/episode_<episode_index>.mp4",
        "depth_encoding": (
            "HEVC gray12le log codes, qp=6, range=full, "
            "depth_mm=[100,5000], qmax=4095, depth_scale=0.001"
        ),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (project_root.parent.parent / ".backups" / "session-migrations"
                   / f"{project_root.name}-depth-png-{timestamp}")
    if backup_root.exists():
        raise RuntimeError(f"Backup destination already exists: {backup_root}")
    write_project_episode_index(project_root, rows)
    _write_json(info_path, info)
    backup_depth = backup_root / "meta" / "depth"
    backup_depth.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(depth_root), str(backup_depth))
    if not keep_backup:
        shutil.rmtree(backup_root, ignore_errors=True)
    verification = verify_project_dataset(project_root)
    if not verification["passed"]:
        raise RuntimeError(f"Depth migration verification failed: {verification['errors']}")
    return {"changed": True, "project": project_root.name,
            "converted": output_records,
            "backup": str(backup_root) if keep_backup else None,
            "verification": verification}


def _source_names(project_root: Path) -> list[Path]:
    return sorted(
        path for path in Path(project_root).iterdir()
        if path.is_dir() and not path.name.startswith(".")
        and path.name not in _CANONICAL_ROOTS
    )


def _clone_source_payload(source: Path, target: Path) -> None:
    for name in ("data", "meta", "videos", "calibration", "depth"):
        candidate = source / name
        if candidate.is_dir():
            clone_tree(candidate, target / name)
    for name in ("metadata.json", "timestamps.json", "upload_manifest.json"):
        candidate = source / name
        if candidate.is_file():
            _link_or_copy(candidate, target / name)


def _preserve_sidecars(source: Path, storage_root: Path, project_name: str,
                       episode_id: str) -> None:
    for source_name, target_name in (("processed", "processed"),
                                     ("original", "archives")):
        source_dir = source / source_name
        if not source_dir.is_dir():
            continue
        clone_tree(source_dir, Path(storage_root) / target_name / project_name / episode_id)


def allocate_project_episode_id(project_root: Path, incoming_name: str,
                                project_name: str) -> str:
    """Choose a stable episode ID without creating a batch subdirectory."""
    incoming = str(incoming_name or "").strip()
    existing = project_episode_rows(project_root)
    existing_ids = {
        str(row.get("episode_id") or row.get("source_batch") or "")
        for row in existing
    }
    if incoming and incoming in existing_ids:
        return incoming
    if incoming and episode_index_from_name(incoming) is not None:
        return incoming
    used = [int(row.get("episode_index")) for row in existing
            if _episode_index(row.get("episode_index")) is not None]
    next_index = max(used, default=0) + 1
    prefix = str(project_name or project_root.name or "episode").strip()
    return f"{prefix}_{next_index:06d}"


def append_project_episode(project_root: Path, episode_root: Path,
                           episode_id: str, *, replace: bool = False) -> dict[str, Any]:
    """Atomically append/replace one episode in a project-level dataset."""
    project_root = Path(project_root)
    episode_root = Path(episode_root)
    parent = project_root.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    temp_root = parent / f".{project_root.name}.append-{stamp}"
    backup_root = (project_root.parent.parent / ".backups" / "session-uploads"
                   / f"{project_root.name}-{stamp}")
    shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=False)
    try:
        if project_root.is_dir() and not is_project_dataset(project_root):
            legacy = [path for path in project_root.iterdir()
                      if path.is_dir() and not path.name.startswith(".")
                      and path.name not in _CANONICAL_ROOTS]
            if legacy:
                raise RuntimeError(
                    f"Project still uses legacy batch directories; migrate first: {project_root}"
                )
        if project_root.is_dir():
            clone_tree(project_root, temp_root)
        rows = project_episode_rows(temp_root)
        existing = next((row for row in rows
                         if str(row.get("episode_id") or row.get("source_batch") or "") == str(episode_id)), None)
        if existing is None:
            target_episode_index = max(
                [_episode_index(row.get("episode_index")) for row in rows
                 if _episode_index(row.get("episode_index")) is not None],
                default=-1,
            ) + 1
            chunk_index = episode_chunk_for_index(target_episode_index)
        else:
            if not replace:
                raise RuntimeError(f"Episode already exists: {episode_id}")
            target_episode_index = _episode_index(existing.get("episode_index"))
            if target_episode_index is None:
                raise RuntimeError(f"Existing episode has no episode_index: {episode_id}")
            chunk_index = _chunk_index(existing.get("data/chunk_index"))
        result = merge_normalized_episode(
            temp_root, episode_root, episode_id, chunk_index, replace=replace,
            target_episode_index=target_episode_index,
        )
        verification = verify_project_dataset(temp_root)
        if not verification["passed"]:
            raise RuntimeError(f"Upload verification failed: {verification['errors']}")
        backup_root.parent.mkdir(parents=True, exist_ok=True)
        project_existed = project_root.is_dir()
        if project_existed:
            project_root.rename(backup_root)
        try:
            temp_root.rename(project_root)
        except Exception:
            if project_existed and backup_root.exists():
                backup_root.rename(project_root)
            raise
        result["verification"] = verify_project_dataset(project_root)
        result["backup"] = str(backup_root) if project_existed else None
        return result
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def migrate_project_dataset(project_root: Path, *, keep_backup: bool = True) -> dict[str, Any]:
    """Flatten legacy batch directories into one project-level dataset.

    The active project is atomically replaced only after all episodes and
    metadata validate.  The old tree is retained under ``data/.backups``.
    """
    project_root = Path(project_root)
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    batches = _source_names(project_root)
    if not batches:
        return {"changed": False, "project": project_root.name,
                "verification": verify_project_dataset(project_root)}

    storage_root = project_root.parent.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temp_root = project_root.parent / f".{project_root.name}.migrate-{stamp}"
    backup_root = (storage_root / ".backups" / "session-migrations"
                   / f"{project_root.name}-{stamp}")
    shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=False)
    mappings: list[dict[str, Any]] = []
    try:
        for target_episode_index, batch in enumerate(batches):
            source_stage = temp_root / f".source-{target_episode_index:03d}"
            _clone_source_payload(batch, source_stage)
            normalized = normalize_extracted_dataset(source_stage, batch.name)
            mappings.append(merge_normalized_episode(
                temp_root, source_stage, batch.name,
                episode_chunk_for_index(target_episode_index),
                target_episode_index=target_episode_index,
            ))
            _preserve_sidecars(batch, storage_root, project_root.name, batch.name)
            shutil.rmtree(source_stage, ignore_errors=True)

        verification = verify_project_dataset(temp_root)
        if not verification["passed"]:
            raise RuntimeError(f"Project migration verification failed: {verification['errors']}")

        backup_root.parent.mkdir(parents=True, exist_ok=True)
        if backup_root.exists():
            raise RuntimeError(f"Backup destination already exists: {backup_root}")
        project_root.rename(backup_root)
        try:
            temp_root.rename(project_root)
        except Exception:
            backup_root.rename(project_root)
            raise
        if not keep_backup:
            # This option is intentionally opt-in; callers should normally
            # retain the backup until the UI and worker are verified.
            shutil.rmtree(backup_root, ignore_errors=True)
        return {
            "changed": True,
            "project": project_root.name,
            "backup": str(backup_root) if keep_backup else None,
            "episodes": mappings,
            "verification": verify_project_dataset(project_root),
        }
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
