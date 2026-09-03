"""Filesystem artifact discovery shared by review APIs.

The active application reads only the canonical project-level
``data/meta/videos`` layout.  Old sidecar directories are intentionally not
resolved by the runtime.
"""

from __future__ import annotations

import re
from pathlib import Path


_SOURCE_SUFFIXES = ("_depth_rgb", "_depth", "_rgb")


def _episode_index_for_id(session_dir: Path, episode_id: str) -> int | None:
    try:
        from app.project_dataset import episode_row
        row = episode_row(Path(session_dir), str(episode_id))
        return int(row.get("episode_index")) if row else None
    except (TypeError, ValueError, OSError):
        return None


def _safe(value: str) -> str:
    return str(value or "").replace("/", "_").replace("\\", "_").strip()


def _source_variants(value: str) -> set[str]:
    raw = _safe(value).lower()
    if not raw:
        return set()
    variants = {raw, re.sub(r"[^a-z0-9_.-]+", "_", raw)}
    changed = True
    while changed:
        changed = False
        for item in tuple(variants):
            for suffix in _SOURCE_SUFFIXES:
                if item.endswith(suffix) and len(item) > len(suffix):
                    shortened = item[: -len(suffix)].rstrip("_")
                    if shortened not in variants:
                        variants.add(shortened)
                        changed = True
            # Collectors have used both ``rgb`` and ``head_rgb`` for the
            # same head-facing colour stream.  Keep this alias limited to
            # RGB keys; depth keys must never become interchangeable with an
            # RGB stream merely because they share a device prefix.
            if item == "head_rgb":
                variants.add("rgb")
            elif item == "rgb":
                variants.add("head_rgb")
            elif item.endswith("_head_rgb"):
                variants.add(item[:-len("_head_rgb")] + "_rgb")
            elif item.endswith("_rgb") and not item.endswith("_depth_rgb"):
                variants.add(item[:-len("_rgb")] + "_head_rgb")
    return variants


def source_matches(value: str | None, camera: str) -> bool:
    """Whether a manifest/file source refers to *camera*.

    ``D405`` and ``D405_depth_rgb`` intentionally match: the former is the
    keypoint artifact stem while the latter is the uploaded video source key.
    """

    return bool(_source_variants(str(value or "")) & _source_variants(camera))


def _unique(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def find_hand_keypoint_candidates(session_dir: Path, camera: str,
                                  episode_id: str | None = None) -> list[Path]:
    """Find MediaPipe 2D parquet files for a camera."""

    result: list[Path] = []
    # Current runs merge the detector columns into the canonical episode
    # parquet.  Returning that file keeps the review API independent of a
    # per-run output directory.
    canonical = _canonical_episode_data(session_dir, episode_id, "2d")
    if canonical is not None:
        result.append(canonical)
    return _unique(result)


def find_hand3d_candidates(session_dir: Path, camera: str,
                           episode_id: str | None = None) -> list[Path]:
    """Find 3D artifacts whose manifest/source maps to a camera."""

    result: list[Path] = []
    canonical = _canonical_episode_data(session_dir, episode_id, "3d")
    if canonical is not None:
        result.append(canonical)
    return _unique(result)


def _canonical_episode_data(session_dir: Path, episode_id: str | None,
                            dimension: str) -> Path | None:
    """Return the canonical episode parquet when it contains merged results."""
    if not episode_id:
        return None
    try:
        from app.project_dataset import episode_row, episode_chunk_for_index, is_project_dataset
        root = Path(session_dir)
        if not is_project_dataset(root):
            return None
        row = episode_row(root, str(episode_id))
        if row is None:
            return None
        index = int(row.get("episode_index", 0))
        path = (root / "data" / f"chunk-{episode_chunk_for_index(index):03d}"
                / f"episode_{index:06d}.parquet")
        if not path.is_file():
            return None
        import pandas as pd
        columns = {str(name) for name in pd.read_parquet(path, engine="pyarrow").columns}
        if dimension == "3d":
            marker = any(
                "landmarks_3d" in name or "world_position" in name
                or "hand_3d" in name
                or name.endswith("hand_left_3d")
                or name.endswith("hand_right_3d")
                for name in columns)
        else:
            marker = any("keypoints" in name or "2d_present" in name
                         or "hand_0_present" in name for name in columns)
        return path if marker else None
    except Exception:
        return None


def find_depth_video(session_dir: Path, source: str,
                     episode_id: str | None = None) -> Path | None:
    # LeRobot v2.1 is authoritative.  Active project datasets must resolve
    # depth only from their canonical metric video; stale PNG/raw-depth
    # sidecars are deliberately not allowed to become a preview source.
    try:
        from app.lerobot_v21 import is_depth_source, iter_video_streams
        episode_index = _episode_index_for_id(Path(session_dir), str(episode_id))
        for source_key, path in iter_video_streams(Path(session_dir) / "videos"):
            if (is_depth_source(source_key)
                    and source_matches(source_key, source)
                    and (episode_index is None
                         or f"episode_{episode_index:06d}" in path.stem)):
                return path
    except Exception:
        pass
    return None


def has_depth_source(session_dir: Path, episode_id: str | None = None) -> bool:
    """Whether an episode has a usable canonical depth video.

    PNG sequences can still be found by migration utilities, but they do not
    qualify as an active project's depth input or world-coordinate preview.
    """
    try:
        from app.lerobot_v21 import is_depth_source, iter_video_streams
        from app.project_dataset import episode_row
        wanted_index = None
        if episode_id and (Path(session_dir) / "meta" / "episodes").is_dir():
            row = episode_row(Path(session_dir), str(episode_id))
            wanted_index = int(row["episode_index"]) if row else None
        for source, path in iter_video_streams(Path(session_dir) / "videos"):
            if not is_depth_source(source):
                continue
            if wanted_index is None or path.stem == f"episode_{wanted_index:06d}":
                return True
    except (TypeError, ValueError, OSError):
        pass
    return False
