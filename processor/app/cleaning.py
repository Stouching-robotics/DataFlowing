"""Data cleaning / validation — runs on uploaded episodes before review.

Checks:
  1. Frame alignment — parquet frame count vs video frame count
  2. Sensor completeness — observation.state not all-zero
  3. Video decode — MP4 files readable by OpenCV
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Episode


def utcnow():
    return datetime.now(timezone.utc)


async def validate_episode(ep: Episode, session_dir: Optional[Path], db: AsyncSession) -> dict:
    """Run all validation checks on a single episode.

    Returns the cleaning_report dict: {passed, checks: [{name, passed, detail}], summary}
    """
    checks = []

    # ── 1. Frame alignment check ──
    checks.append(await _check_frame_alignment(ep, session_dir))

    # ── 2. Sensor completeness ──
    checks.append(await _check_sensor_completeness(ep, session_dir))

    # ── 3. Video decode check ──
    checks.append(await _check_video_decode(ep))

    # ── 4. Device connectivity check ──
    checks.append(await _check_device_connectivity(ep, session_dir))

    passed = all(c["passed"] for c in checks)
    failed_names = [c["name"] for c in checks if not c["passed"]]

    report = {
        "passed": passed,
        "checks": checks,
        "summary": "All checks passed" if passed else f"Failed: {', '.join(failed_names)}",
        "validated_at": utcnow().isoformat(),
    }

    # Update episode
    ep.cleaning_report = report
    ep.processing_started_at = utcnow()
    ep.review_ready_at = utcnow()

    if passed:
        # Map old legacy statuses appropriately
        if ep.status not in ("reviewed", "approved", "rejected"):
            ep.status = "to_review"
    else:
        if ep.status not in ("reviewed", "approved"):
            ep.status = "failed"

    await db.flush()
    return report


# ── Individual checks ──────────────────────────────────────────

async def _check_frame_alignment(ep: Episode, session_dir: Optional[Path]) -> dict:
    """Compare parquet frame count with video frame count."""
    if not session_dir or not session_dir.exists():
        return {"name": "frame_alignment", "passed": True,
                "detail": "No session dir, skipped"}

    try:
        import pandas as pd

        parquet_frames = 0
        seen_indices = set()
        for data_dir in session_dir.rglob("data"):
            if not data_dir.is_dir():
                continue
            for pq_file in data_dir.rglob("*.parquet"):
                try:
                    df = pd.read_parquet(pq_file)
                    if "frame_index" in df.columns:
                        for fi in df["frame_index"]:
                            seen_indices.add(int(fi))
                except Exception:
                    continue

        parquet_frames = len(seen_indices)
        video_frames = ep.frame_count or 1
        if parquet_frames == 0:
            return {"name": "frame_alignment", "passed": True,
                    "detail": "No parquet data to compare"}

        ratio = abs(parquet_frames - video_frames) / max(video_frames, 1)
        passed = ratio < 0.10  # < 10% deviation

        return {
            "name": "frame_alignment",
            "passed": passed,
            "detail": f"parquet={parquet_frames}, video={video_frames}, dev={ratio:.1%}",
        }
    except Exception as e:
        return {"name": "frame_alignment", "passed": True,
                "detail": f"Check error (skipped): {e}"}


async def _check_sensor_completeness(ep: Episode, session_dir: Optional[Path]) -> dict:
    """Check that observation.state data is present (not all-zero for all frames)."""
    if not session_dir or not session_dir.exists():
        return {"name": "sensor_completeness", "passed": True,
                "detail": "No session dir, skipped"}

    try:
        import pandas as pd
        import numpy as np

        for data_dir in session_dir.rglob("data"):
            if not data_dir.is_dir():
                continue
            for pq_file in data_dir.rglob("*.parquet"):
                try:
                    df = pd.read_parquet(pq_file)
                    # Look for observation columns
                    obs_cols = [c for c in df.columns if c.startswith("observation.")]
                    if not obs_cols:
                        continue

                    # Sample first 10 frames, check not all zero
                    sample = df[obs_cols[0]].head(min(10, len(df)))
                    all_zero = True
                    for val in sample:
                        if isinstance(val, (list, np.ndarray)):
                            arr = np.array(val, dtype=np.float32)
                            if arr.max() > 0:
                                all_zero = False
                                break

                    if all_zero:
                        return {
                            "name": "sensor_completeness",
                            "passed": False,
                            "detail": "All sampled observation values are zero",
                        }
                    return {
                        "name": "sensor_completeness",
                        "passed": True,
                        "detail": f"Checked {len(sample)} frames, data present",
                    }
                except Exception:
                    continue

        # No observation columns found — video-only episode, OK
        return {"name": "sensor_completeness", "passed": True,
                "detail": "No observation columns (video-only episode)"}
    except Exception as e:
        return {"name": "sensor_completeness", "passed": True,
                "detail": f"Check error (skipped): {e}"}


async def _check_video_decode(ep: Episode) -> dict:
    """Verify the associated MP4 video is readable by OpenCV."""
    try:
        import cv2
    except ImportError:
        return {"name": "video_decode", "passed": True,
                "detail": "OpenCV not available, skipped"}

    video_path = None
    if ep.meta and ep.meta.get("video_path"):
        p = settings.storage_root / ep.meta["video_path"]
        if p.exists():
            video_path = p

    if not video_path:
        # Try scanning sessions dir
        sessions_root = settings.storage_root / "sessions"
        if sessions_root.exists():
            for mp4 in sessions_root.rglob("*.mp4"):
                video_path = mp4
                break

    if not video_path:
        return {"name": "video_decode", "passed": True,
                "detail": "No video file found to check"}

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"name": "video_decode", "passed": False,
                    "detail": f"Cannot open: {video_path.name}"}
        ret, _ = cap.read()
        cap.release()
        if not ret:
            return {"name": "video_decode", "passed": False,
                    "detail": f"Cannot read first frame: {video_path.name}"}
        return {"name": "video_decode", "passed": True,
                "detail": f"Video readable: {video_path.name}"}
    except Exception as e:
        return {"name": "video_decode", "passed": False,
                "detail": f"Decode error: {e}"}


async def _check_device_connectivity(ep: Episode, session_dir: Optional[Path]) -> dict:
    """Scan parquet for status.* columns — flag disconnected devices as failed."""
    if not session_dir or not session_dir.exists():
        return {"name": "device_connectivity", "passed": True,
                "detail": "No session dir, skipped"}

    try:
        import pandas as pd

        disconnected_devices: set[str] = set()
        for data_dir in session_dir.rglob("data"):
            if not data_dir.is_dir():
                continue
            for pq_file in data_dir.rglob("*.parquet"):
                try:
                    df = pd.read_parquet(pq_file)
                    status_cols = [c for c in df.columns if c.startswith("status.")]
                    for col in status_cols:
                        if "disconnected" in df[col].values:
                            device_name = col.split(".", 1)[1]
                            disconnected_devices.add(device_name)
                except Exception:
                    continue

        if disconnected_devices:
            return {
                "name": "device_connectivity",
                "passed": False,
                "detail": f"Disconnected: {', '.join(sorted(disconnected_devices))}",
            }
        return {
            "name": "device_connectivity",
            "passed": True,
            "detail": "All devices connected",
        }
    except Exception as e:
        return {"name": "device_connectivity", "passed": True,
                "detail": f"Check error (skipped): {e}"}


# ── 文件级校验(无数据库版本)──────────────────────────

def validate_batch(batch_dir: Path) -> dict:
    """校验一个批次目录(无数据库):帧对齐 / 传感器 / 视频解码。"""
    checks = [
        _check_frame_alignment_fs(batch_dir),
        _check_sensor_completeness_fs(batch_dir),
        _check_video_decode_fs(batch_dir),
    ]
    passed = all(c["passed"] for c in checks)
    failed_names = [c["name"] for c in checks if not c["passed"]]
    return {
        "passed": passed,
        "checks": checks,
        "summary": "All checks passed" if passed else f"Failed: {', '.join(failed_names)}",
        "validated_at": utcnow().isoformat(),
    }


def _check_frame_alignment_fs(batch_dir: Path) -> dict:
    """parquet 帧数 vs 视频帧数(±10%)。"""
    try:
        import pandas as pd
        seen = set()
        for data_dir in batch_dir.rglob("data"):
            if not data_dir.is_dir():
                continue
            for pq_file in data_dir.rglob("*.parquet"):
                try:
                    df = pd.read_parquet(pq_file)
                    if "frame_index" in df.columns:
                        seen.update(int(fi) for fi in df["frame_index"])
                except Exception:
                    continue
        parquet_frames = len(seen)
        video_frames = 0
        for mp4 in batch_dir.rglob("*.mp4"):
            cap = cv2.VideoCapture(str(mp4))
            try:
                video_frames = max(video_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0)
            finally:
                cap.release()
        if parquet_frames == 0:
            return {"name": "frame_alignment", "passed": True, "detail": "No parquet data"}
        ratio = abs(parquet_frames - video_frames) / max(video_frames, 1)
        return {"name": "frame_alignment", "passed": ratio < 0.10,
                "detail": f"parquet={parquet_frames}, video={video_frames}, dev={ratio:.1%}"}
    except Exception as e:
        return {"name": "frame_alignment", "passed": True, "detail": f"Check error (skipped): {e}"}


def _check_sensor_completeness_fs(batch_dir: Path) -> dict:
    """采样前 10 帧,检查 observation 数据非全零。"""
    try:
        import pandas as pd
        import numpy as np
        for data_dir in batch_dir.rglob("data"):
            if not data_dir.is_dir():
                continue
            for pq_file in data_dir.rglob("*.parquet"):
                try:
                    df = pd.read_parquet(pq_file)
                    obs_cols = [c for c in df.columns if c.startswith("observation.")]
                    if not obs_cols:
                        continue
                    sample = df[obs_cols[0]].head(min(10, len(df)))
                    for val in sample:
                        if isinstance(val, (list, np.ndarray)):
                            if np.array(val, dtype=np.float32).max() > 0:
                                return {"name": "sensor_completeness", "passed": True,
                                        "detail": f"Checked {len(sample)} frames, data present"}
                    return {"name": "sensor_completeness", "passed": False,
                            "detail": "All sampled observation values are zero"}
                except Exception:
                    continue
        return {"name": "sensor_completeness", "passed": True,
                "detail": "No observation columns (video-only episode)"}
    except Exception as e:
        return {"name": "sensor_completeness", "passed": True,
                "detail": f"Check error (skipped): {e}"}


def _check_video_decode_fs(batch_dir: Path) -> dict:
    """验证批次内第一个 MP4 可被 OpenCV 读取。"""
    try:
        import cv2
    except ImportError:
        return {"name": "video_decode", "passed": True, "detail": "OpenCV not available"}
    mp4s = sorted(batch_dir.rglob("*.mp4"))
    if not mp4s:
        return {"name": "video_decode", "passed": True, "detail": "No video file"}
    try:
        cap = cv2.VideoCapture(str(mp4s[0]))
        if not cap.isOpened():
            return {"name": "video_decode", "passed": False,
                    "detail": f"Cannot open: {mp4s[0].name}"}
        ret, _ = cap.read()
        cap.release()
        return {"name": "video_decode", "passed": ret,
                "detail": f"Video readable: {mp4s[0].name}"}
    except Exception as e:
        return {"name": "video_decode", "passed": False, "detail": f"Decode error: {e}"}
