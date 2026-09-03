"""Clean up active LeRobot v2.1 media without touching recoverable history.

The first D435 collectors used several names for the same two physical
streams.  This utility normalizes them to ``D435_rgb`` and ``D435_depth`` in
active datasets.  It also removes files advertised as pure depth when ffprobe
shows that they are colour/video pixels rather than metric
``gray12le``/``gray16le`` depth.  Every moved video and every rewritten
metadata file is preserved in ``data/.backups/session-migrations``.

Default mode is a read-only report.  Use ``--apply`` only after reviewing it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.lerobot_v21 import (  # noqa: E402
    VIDEO_SUFFIXES,
    canonical_source_key,
    is_depth_source,
    iter_video_streams,
)
from app.project_dataset import (  # noqa: E402
    episode_files,
    episode_row,
    project_episode_rows,
    write_project_episode_index,
)


EPISODE_RE = re.compile(r"episode_(\d+)$")
SOURCE_RENAMES = {
    "D435_depth_rgb": "D435_rgb",
    "D435_head_rgb": "D435_rgb",
    "D435_head_depth": "D435_depth",
}
DEPTH_PIX_FMTS = {"gray12le", "gray16le"}
DEVICE_NAME_ALIASES = {
    "d435_depth": "D435",
    "d435_head": "D435",
}


def _probe(path: Path) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt,width,height",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        streams = json.loads(result.stdout or "{}").get("streams") or []
        return {str(k): str(v or "") for k, v in (streams[0] if streams else {}).items()}
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return {}


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _backup_file(source: Path, project: Path, backup: Path) -> None:
    _copy_or_link(source, backup / project.name / source.relative_to(project))


def _backup_video(source: Path, project: Path, backup: Path) -> None:
    _copy_or_link(source, backup / project.name / source.relative_to(project))


def _metadata_files(project: Path) -> list[Path]:
    meta = project / "meta"
    if not meta.is_dir():
        return []
    return sorted(path for path in meta.rglob("*")
                  if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})


def _merge_values(current: Any, incoming: Any) -> Any:
    """Merge colliding source-key metadata while keeping canonical values."""
    if isinstance(current, dict) and isinstance(incoming, dict):
        merged = dict(current)
        for key, value in incoming.items():
            if key not in merged:
                merged[key] = value
            else:
                merged[key] = _merge_values(merged[key], value)
        return merged
    if isinstance(current, list) and isinstance(incoming, list):
        result = list(current)
        for value in incoming:
            if value not in result:
                result.append(value)
        return result
    return current


def _rename_source_value(value: Any, old: str, new: str) -> Any:
    old_feature = f"observation.images.{old}"
    new_feature = f"observation.images.{new}"
    if isinstance(value, str):
        return value.replace(old_feature, new_feature).replace(old, new)
    if isinstance(value, list):
        return [_rename_source_value(item, old, new) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, child in value.items():
        renamed_key = key.replace(old_feature, new_feature).replace(old, new)
        renamed_child = _rename_source_value(child, old, new)
        if renamed_key in result:
            result[renamed_key] = _merge_values(result[renamed_key], renamed_child)
        else:
            result[renamed_key] = renamed_child
    return result


def _rewrite_source_names(project: Path, old: str, new: str,
                          backup: Path | None) -> list[str]:
    changed: list[str] = []
    for path in _metadata_files(project):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        updated = text
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(text)
                updated = json.dumps(
                    _rename_source_value(value, old, new),
                    ensure_ascii=False, indent=2,
                )
            except (ValueError, TypeError):
                updated = text.replace(
                    f"observation.images.{old}",
                    f"observation.images.{new}",
                ).replace(old, new)
        elif path.suffix.lower() == ".jsonl":
            lines = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    lines.append(json.dumps(
                        _rename_source_value(json.loads(line), old, new),
                        ensure_ascii=False,
                    ))
                except (ValueError, TypeError):
                    lines.append(line.replace(
                        f"observation.images.{old}",
                        f"observation.images.{new}",
                    ).replace(old, new))
            updated = "\n".join(lines) + ("\n" if lines else "")
        else:
            updated = text.replace(
                f"observation.images.{old}",
                f"observation.images.{new}",
            ).replace(old, new)
        if updated == text:
            continue
        if backup is not None:
            _backup_file(path, project, backup)
        path.write_text(updated, encoding="utf-8")
        changed.append(str(path.relative_to(project)))
    return changed


def _canonical_device_name(value: Any, kind: Any = "", key: Any = "") -> Any:
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    context = f"{kind} {key}".lower()
    if low in DEVICE_NAME_ALIASES and "d435" in context:
        return DEVICE_NAME_ALIASES[low]
    return value


def _normalise_devices(items: Any) -> Any:
    if not isinstance(items, list):
        return items
    merged: list[Any] = []
    by_key: dict[str, int] = {}
    for raw in items:
        if not isinstance(raw, dict):
            merged.append(raw)
            continue
        item = _normalise_device_metadata(raw)
        key = str(item.get("key") or "").strip()
        if not key or key not in by_key:
            if key:
                by_key[key] = len(merged)
            merged.append(item)
            continue
        index = by_key[key]
        merged[index] = _merge_values(merged[index], item)
    return merged


def _normalise_device_metadata(value: Any, field: str = "",
                               kind: Any = "", key: Any = "") -> Any:
    """Normalize physical D435 names without changing stream keys."""
    if isinstance(value, list):
        return [_normalise_device_metadata(item, field, kind, key)
                for item in value]
    if not isinstance(value, dict):
        if field in {"name", "device", "device_name"}:
            return _canonical_device_name(value, kind, key)
        return value
    local_kind = value.get("kind") or kind
    local_key = value.get("key") or key
    result: dict[str, Any] = {}
    for name, child in value.items():
        if name == "devices":
            child = _normalise_devices(child)
        elif name == "device_names" and isinstance(child, dict):
            child = {
                stream: _canonical_device_name(display, "d435", stream)
                for stream, display in child.items()
            }
        elif name in {"name", "device", "device_name"}:
            child = _canonical_device_name(child, local_kind, local_key)
        else:
            child = _normalise_device_metadata(child, name, local_kind, local_key)
        result[name] = child
    return result


def _rewrite_device_metadata(project: Path, backup: Path | None) -> list[str]:
    changed: list[str] = []
    for path in _metadata_files(project):
        if path.suffix.lower() != ".json":
            continue
        try:
            before = path.read_text(encoding="utf-8")
            value = json.loads(before)
        except (OSError, UnicodeError, ValueError, TypeError):
            continue
        updated_value = _normalise_device_metadata(value)
        updated = json.dumps(updated_value, ensure_ascii=False, indent=2)
        if updated == before:
            continue
        if backup is not None:
            _backup_file(path, project, backup)
        path.write_text(updated, encoding="utf-8")
        changed.append(str(path.relative_to(project)))
    return changed


def _rewrite_video_extensions(project: Path, backup: Path | None) -> list[str]:
    """Make the v2.1 extension index agree with active video files."""
    info_path = project / "meta" / "info.json"
    if not info_path.is_file():
        return []
    actual = {
        source: path.suffix.lower().lstrip(".")
        for source, path in iter_video_streams(project / "videos")
    }
    if not actual:
        return []
    try:
        before = info_path.read_text(encoding="utf-8")
        info = json.loads(before)
    except (OSError, UnicodeError, ValueError, TypeError):
        return []
    if not isinstance(info, dict):
        return []
    current = info.get("video_extensions")
    if current == actual:
        return []
    info["video_extensions"] = dict(sorted(actual.items()))
    if backup is not None:
        _backup_file(info_path, project, backup)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return [str(info_path.relative_to(project))]


def _rewrite_json(path: Path, transform) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return False
    updated = transform(value)
    if updated == value:
        return False
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    os.replace(temporary, path)
    return True


def _remove_source_from_value(value: Any, source: str) -> Any:
    """Remove a source only from source-bearing metadata fields."""
    source_fields = {"cameras", "device_names"}
    list_fields = {"slots", "depth_sources", "metric_depth_sources", "camera_sources"}
    if isinstance(value, list):
        return [_remove_source_from_value(item, source) for item in value
                if item != source]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in source_fields and isinstance(item, dict):
            item = {name: child for name, child in item.items() if name != source}
        elif key in list_fields and isinstance(item, list):
            item = [child for child in item if child != source]
        result[key] = _remove_source_from_value(item, source)
    return result


def _remove_episode_source_metadata(project: Path, episode_id: str,
                                    source: str, backup: Path | None) -> list[str]:
    changed: list[str] = []
    collector = project / "meta" / "collector" / episode_id
    if not collector.is_dir():
        return changed
    for path in sorted(collector.glob("*.json")):
        before = path.read_text(encoding="utf-8", errors="ignore")
        if source not in before:
            continue
        if backup is not None:
            _backup_file(path, project, backup)
        if _rewrite_json(path, lambda value: _remove_source_from_value(value, source)):
            changed.append(str(path.relative_to(project)))
    return changed


def _remove_episode_row_source(project: Path, episode_index: int, source: str,
                               backup: Path | None) -> bool:
    rows = project_episode_rows(project)
    if not rows:
        return False
    changed = False
    prefix = f"videos/observation.images.{source}/"
    updated: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("episode_index", -1)) == episode_index:
            filtered = {key: value for key, value in row.items()
                        if not str(key).startswith(prefix)}
            changed |= filtered != row
            row = filtered
        updated.append(row)
    if not changed:
        return False
    if backup is not None:
        for path in sorted((project / "meta" / "episodes").rglob("*.parquet")):
            _backup_file(path, project, backup)
    write_project_episode_index(project, updated)
    return True


def _episode_id(project: Path, index: int) -> str:
    for row in project_episode_rows(project):
        if int(row.get("episode_index", -1)) == index:
            return str(row.get("episode_id") or row.get("source_batch") or "")
    return ""


def _active_video_paths(project: Path) -> list[Path]:
    videos = project / "videos"
    if not videos.is_dir():
        return []
    return sorted(path for path in videos.rglob("*.mp4")
                  if path.is_file() and any(part.startswith("chunk-") for part in path.parts))


def clean_project(project: Path, backup: Path | None, apply: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"project": project.name, "renamed": [],
                              "invalid_depth": [], "metadata": []}
    if not project.is_dir():
        return result

    # Normalize historical D435 source names in every chunk.  If the target
    # directory already exists, merge only non-colliding episode files into
    # it; the caller has already verified that episode numbers do not overlap.
    for old_source, new_source in SOURCE_RENAMES.items():
        old_dirs = sorted((project / "videos").rglob(
            f"observation.images.{old_source}"))
        for old_dir in old_dirs:
            if not old_dir.is_dir():
                continue
            new_dir = old_dir.with_name(f"observation.images.{new_source}")
            result["renamed"].append({
                "from": str(old_dir.relative_to(project)),
                "to": str(new_dir.relative_to(project)),
            })
            if not apply:
                continue
            if new_dir.exists():
                conflicts = sorted(
                    item.name for item in old_dir.iterdir()
                    if (new_dir / item.name).exists()
                )
                if conflicts:
                    raise RuntimeError(
                        f"Rename collision: {new_dir} ({', '.join(conflicts)})"
                    )
            if backup is not None:
                for path in old_dir.rglob("*"):
                    if path.is_file():
                        _backup_video(path, project, backup)
            new_dir.mkdir(parents=True, exist_ok=True)
            for item in sorted(old_dir.iterdir()):
                target = new_dir / item.name
                if target.exists():
                    raise RuntimeError(f"Rename collision: {target}")
                item.rename(target)
            old_dir.rmdir()
        if apply and old_dirs:
            result["metadata"].extend(_rewrite_source_names(
                project, old_source, new_source, backup))

    # Merge duplicate physical-device records left by historical uploads.
    # Stream keys remain untouched: D435_rgb and D435_depth are data sources,
    # while D435 is the single hardware name shown by the UI.
    if apply:
        result["metadata"].extend(_rewrite_device_metadata(project, backup))
        result["metadata"].extend(_rewrite_video_extensions(project, backup))

    # A metric depth stream must be lossless gray12/gray16.  Keep the valid
    # files and move only the colour-coded/ordinary-video members out of the
    # active dataset.  This handles mixed historical batches in one source.
    invalid: list[tuple[Path, str, int]] = []
    for path in _active_video_paths(project):
        relative_parts = path.relative_to(project).parts
        source_parts = [part for part in relative_parts
                        if part.startswith("observation.images.")]
        if not source_parts:
            continue
        source = canonical_source_key(source_parts[-1])
        if not is_depth_source(source):
            continue
        probe = _probe(path)
        if probe.get("pix_fmt") in DEPTH_PIX_FMTS:
            continue
        match = EPISODE_RE.search(path.stem)
        if not match:
            continue
        invalid.append((path, source, int(match.group(1))))
    for path, source, episode_index in invalid:
        result["invalid_depth"].append({
            "path": str(path.relative_to(project)),
            "source": source,
            "episode_index": episode_index,
            "probe": _probe(path),
        })
        if not apply:
            continue
        if backup is not None:
            _backup_video(path, project, backup)
        path.unlink()
        episode_id = _episode_id(project, episode_index)
        if episode_id:
            result["metadata"].extend(_remove_episode_source_metadata(
                project, episode_id, source, backup))
        if _remove_episode_row_source(project, episode_index, source, backup):
            result["metadata"].append("meta/episodes/")

    return result


def clean_duplicate_processed_depth(root: Path, backup: Path | None,
                                    apply: bool) -> list[dict[str, Any]]:
    """Move raw PNG sidecars when the same episode has canonical depth video."""
    processed = root / "processed"
    sessions = root / "sessions"
    result: list[dict[str, Any]] = []
    if not processed.is_dir():
        return result

    for raw_root in sorted(processed.rglob("raw_depth")):
        if not raw_root.is_dir():
            continue
        try:
            relative = raw_root.relative_to(processed)
            project_name, episode_name = relative.parts[0], relative.parts[1]
        except (ValueError, IndexError):
            continue
        project = sessions / project_name
        if not project.is_dir():
            continue
        row = episode_row(project, episode_name)
        if not row:
            continue
        try:
            episode_index = int(row["episode_index"])
        except (KeyError, TypeError, ValueError):
            continue
        canonical = {
            source: path for source, path in episode_files(project, episode_index)["videos"]
        }
        for source_dir in sorted(path for path in raw_root.rglob("*") if path.is_dir()):
            pngs = [path for path in source_dir.glob("*.png")
                    if path.is_file() and path.stem.isdigit()]
            if not pngs:
                continue
            source = canonical_source_key(source_dir.name)
            video = canonical.get(source)
            if video is None or not is_depth_source(source):
                continue
            probe = _probe(video)
            if probe.get("pix_fmt") not in DEPTH_PIX_FMTS:
                continue
            item = {
                "path": str(source_dir.relative_to(root)),
                "project": project_name,
                "episode_id": episode_name,
                "episode_index": episode_index,
                "source": source,
                "png_count": len(pngs),
                "canonical_video": str(video.relative_to(root)),
                "canonical_probe": probe,
            }
            result.append(item)
            if apply:
                if backup is None:
                    raise RuntimeError("backup path is required for cleanup")
                target = backup / "processed-legacy" / source_dir.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                original_parent = source_dir.parent
                shutil.move(str(source_dir), str(target))
                # Remove only the empty legacy raw_depth shell left behind in
                # the active processed run; never remove a non-empty run or
                # any sibling annotation output.
                parent = original_parent
                while parent != processed and parent != root:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="active data root containing sessions/")
    parser.add_argument("--project", action="append",
                        help="limit to one or more project directory names")
    parser.add_argument("--apply", action="store_true",
                        help="perform moves and metadata updates; default is dry-run")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    sessions = root / "sessions"
    if not sessions.is_dir():
        raise SystemExit(f"sessions directory not found: {sessions}")
    wanted = set(args.project or [])
    projects = [path for path in sorted(sessions.iterdir())
                if path.is_dir() and (not wanted or path.name in wanted)]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / ".backups" / "session-migrations" / f"media-cleanup-{stamp}" \
        if args.apply else None
    if backup is not None:
        backup.mkdir(parents=True, exist_ok=False)

    report = {"root": str(root), "apply": bool(args.apply),
              "backup": str(backup) if backup else None, "projects": [],
              "legacy_processed_depth": []}
    for project in projects:
        report["projects"].append(clean_project(project, backup, args.apply))
    report["legacy_processed_depth"] = clean_duplicate_processed_depth(
        root, backup, args.apply)

    if args.apply:
        for item in report["projects"]:
            project = sessions / item["project"]
            if backup is None:
                continue
            # Keep maintenance reports outside the canonical LeRobot meta
            # namespace.  ``meta`` should contain dataset metadata only.
            manifest = backup / "reports" / f"{item['project']}.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(item, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
