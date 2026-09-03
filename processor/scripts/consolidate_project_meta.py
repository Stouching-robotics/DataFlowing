#!/usr/bin/env python3
"""Consolidate active project metadata to the internal four-entry contract.

The active project metadata is intentionally small and predictable::

    meta/info.json
    meta/stats.json
    meta/tasks.json
    meta/episodes/chunk-XXX/episode_XXXXXX.parquet

Calibration and collector JSON is preserved inside the matching episode row
under ``calibration`` and ``collector``.  Unexpected legacy metadata is moved
to a timestamped sibling history directory after verification; data and video
files are not copied or removed by this command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _archive_path(history: Path, relative: Path) -> Path:
    target = history / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def consolidate(root: Path, history_root: Path) -> dict:
    from app.project_dataset import (
        _merge_info,
        _read_episode_sidecars,
        _read_json,
        _read_jsonl,
        _repair_episode_stats,
        _task_rows,
        _write_json,
        _write_tasks_json,
        episode_chunk_for_index,
        project_episode_rows,
        clone_tree,
        verify_project_dataset,
        write_project_episode_index,
        write_project_stats,
    )

    root = Path(root).resolve()
    meta = root / "meta"
    if not (root / "data").is_dir() or not (root / "videos").is_dir():
        raise RuntimeError(f"Not a project dataset: {root}")
    rows = project_episode_rows(root)
    if not rows:
        raise RuntimeError(f"No episode metadata found: {root}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history = Path(history_root).resolve() / stamp / root.name
    backup_meta = history / "meta-before"
    clone_tree(meta, backup_meta)

    try:
        old_stats = _read_jsonl(meta / "episodes_stats.jsonl")
        repaired_stats = _repair_episode_stats(old_stats, rows)
        stats_by_index = {
            int(item["episode_index"]): item.get("stats") or {}
            for item in repaired_stats
            if isinstance(item, dict) and item.get("episode_index") is not None
        }

        normalized_rows = []
        rows = sorted(rows, key=lambda item: int(item.get("episode_index", 0)))
        dataset_cursor = 0
        for row in rows:
            row = dict(row)
            index = int(row.get("episode_index", len(normalized_rows)))
            episode_id = str(
                row.get("episode_id") or row.get("source_batch")
                or f"{root.name}_{index:06d}"
            )
            row["episode_index"] = index
            row["episode_id"] = episode_id
            row.setdefault("source_batch", episode_id)
            if not isinstance(row.get("stats"), dict) or not row.get("stats"):
                stats = stats_by_index.get(index, {})
                if stats:
                    row["stats"] = stats
                else:
                    row.pop("stats", None)
            row.update(_read_episode_sidecars(root, episode_id))

            chunk = episode_chunk_for_index(index)
            row["data/chunk_index"] = chunk
            row["data/file_index"] = index
            for key in list(row):
                if key.endswith("/chunk_index"):
                    row[key] = chunk
                elif key.endswith("/file_index"):
                    row[key] = index

            length = int(row.get("length") or 0)
            row["dataset_from_index"] = dataset_cursor
            dataset_cursor += length
            row["dataset_to_index"] = dataset_cursor
            normalized_rows.append(row)

        tasks = _task_rows(root, root.name)
        write_project_episode_index(root, normalized_rows)
        write_project_stats(root, normalized_rows)
        _write_tasks_json(root, tasks)

        info = _read_json(meta / "info.json", {})
        info = _merge_info(root, info, normalized_rows, tasks)
        _write_json(meta / "info.json", info)

        verification = verify_project_dataset(root)
        allowed = {"info.json", "stats.json", "tasks.json", "episodes"}
        legacy_meta = history / "legacy-meta"
        for child in list(meta.iterdir()):
            if child.name in allowed:
                continue
            shutil.move(str(child), str(_archive_path(legacy_meta, Path(child.name))))
        actual = {child.name for child in meta.iterdir()}
        if not verification["passed"]:
            raise RuntimeError(f"dataset verification failed: {verification['errors']}")
        if actual != allowed:
            raise RuntimeError(f"metadata contract not ready: {sorted(actual)}")

        # The old files/directories are already absent from the active tree
        # only after the successful write above.  Keep them out of meta while
        # preserving a local rollback/history copy beside the sessions root.
        # (The backup contains the complete pre-consolidation meta tree.)
        return {
            "project": root.name,
            "episodes": len(normalized_rows),
            "chunks": len({episode_chunk_for_index(int(row["episode_index"]))
                           for row in normalized_rows}),
            "history": str(history),
            "verification": verification,
        }
    except Exception:
        # Restore the metadata tree only; source data and videos were never
        # changed.  This makes a failed project independently retryable.
        if meta.exists():
            shutil.rmtree(meta)
        clone_tree(backup_meta, meta)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument(
        "--history-root", type=Path,
        help="sibling directory for pre-consolidation metadata backups",
    )
    args = parser.parse_args()
    for root in args.roots:
        root = root.resolve()
        history_root = args.history_root or (root.parent.parent / ".meta-history")
        try:
            result = consolidate(root, history_root)
        except Exception as exc:
            print(json.dumps({"project": root.name, "error": str(exc)},
                             ensure_ascii=False, indent=2))
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
