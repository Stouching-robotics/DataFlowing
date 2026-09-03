"""Post-processing video quality checks.

The checks in this module are deliberately media-only.  They do not try to
decide which hand is left/right or whether an action label is correct.  A
workflow can use them as a gate before the existing ``reviewed``/Approved
state:

* the file opens and sampled frames can be decoded;
* a stream is not materially shorter than the episode timeline;
* sampled frames are not almost entirely black;
* sampled frames do not remain identical for a configurable duration.

An encoded MP4 does not always retain the capture device's original frame
index.  Therefore the frame-count check detects a materially short stream,
while exact capture-time drops require source ``frame_index``/PTS metadata and
are reported separately by future collectors.  The result is fail-closed for
automatic approval, but the old UI still only exposes Reviewing/Approved.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.localstore import read_episode_state, write_episode_state


DEFAULT_SAMPLE_INTERVAL_SEC = 0.5
DEFAULT_MAX_SAMPLES = 240
DEFAULT_FREEZE_MIN_SEC = 2.0
DEFAULT_FREEZE_DIFF = 0.75
DEFAULT_BLACK_MEAN = 5.0
DEFAULT_BLACK_DARK_RATIO = 0.995
DEFAULT_BLACK_MIN_SEC = 0.5
DEFAULT_FRAME_TOLERANCE_RATIO = 0.005
DEFAULT_FRAME_TOLERANCE_MIN = 2
_VIDEO_QC_SEMAPHORE = asyncio.Semaphore(1)


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _raw_videos(batch_dir: Path) -> list[Path]:
    """Return input videos only; never inspect processed run outputs."""
    videos_root = batch_dir / "videos"
    if videos_root.is_dir():
        paths = [p for p in videos_root.rglob("*")
                 if p.is_file() and p.suffix.lower() == ".mp4"]
    else:
        paths = [p for p in batch_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() == ".mp4"
                 and "processed" not in p.parts
                 and "original" not in p.parts
                 and "tmp" not in p.parts]
    return sorted(paths, key=lambda p: str(p).lower())


def _sample_indices(frame_count: int, fps: float) -> tuple[list[int], float]:
    """Build approximately half-second sample positions with a hard cap."""
    count = max(0, int(frame_count))
    if count <= 0:
        return [], 0.0
    interval = _float_env(
        "EGODATA_VIDEO_QC_SAMPLE_INTERVAL_SEC",
        DEFAULT_SAMPLE_INTERVAL_SEC,
        minimum=0.05,
    )
    frame_step = max(1, int(round(max(1.0, fps) * interval)))
    indices = list(range(0, count, frame_step))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    max_samples = _int_env(
        "EGODATA_VIDEO_QC_MAX_SAMPLES", DEFAULT_MAX_SAMPLES, minimum=3)
    if len(indices) > max_samples:
        # Keep samples distributed across the whole stream.  This may miss a
        # very short freeze in a very long video, but it prevents a quality
        # check from turning into a full decode pass on remote storage.
        positions = [round(i * (len(indices) - 1) / (max_samples - 1))
                     for i in range(max_samples)]
        indices = [indices[position] for position in positions]
    return sorted(set(indices)), frame_step / max(1.0, fps)


def _add_freeze_range(ranges: list[list[float]], start: int, end: int,
                      fps: float) -> None:
    if end < start:
        return
    item = [round(start / max(1.0, fps), 3),
            round(end / max(1.0, fps), 3)]
    if ranges and item[0] <= ranges[-1][1] + 0.01:
        ranges[-1][1] = max(ranges[-1][1], item[1])
    else:
        ranges.append(item)


def _check_stream(path: Path, expected_frames: int = 0,
                  expected_fps: float = 0.0) -> dict:
    """Check one video without loading the complete stream into memory."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        return {
            "path": str(path),
            "passed": False,
            "reason": "opencv_unavailable",
            "error": str(exc),
        }

    report: dict = {
        "path": str(path),
        "passed": False,
        "frame_count": 0,
        "expected_frame_count": max(0, int(expected_frames or 0)),
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "sampled_frames": 0,
        "decode_error_count": 0,
        "decode_errors": [],
        "black_frame_count": 0,
        "black_frames": [],
        "black_ranges": [],
        "freeze_ranges": [],
        "dropped_frame_estimate": 0,
        "count_tolerance": 0,
    }
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        report["reason"] = "video_open_failed"
        cap.release()
        return report

    try:
        frame_count = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)))
        height = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)))
        report.update(frame_count=max(0, frame_count), fps=round(max(0.0, fps), 4),
                      width=max(0, width), height=max(0, height))
        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            report["reason"] = "video_metadata_invalid"
            return report

        indices, sample_interval = _sample_indices(frame_count, fps)
        report["sample_interval_sec"] = round(sample_interval, 3)
        black_mean = _float_env("EGODATA_VIDEO_QC_BLACK_MEAN",
                                DEFAULT_BLACK_MEAN, minimum=0.0)
        black_dark_ratio = min(
            1.0,
            _float_env("EGODATA_VIDEO_QC_BLACK_DARK_RATIO",
                       DEFAULT_BLACK_DARK_RATIO, minimum=0.0),
        )
        black_min_sec = _float_env("EGODATA_VIDEO_QC_BLACK_MIN_SEC",
                                  DEFAULT_BLACK_MIN_SEC, minimum=0.0)
        freeze_min_sec = _float_env("EGODATA_VIDEO_QC_FREEZE_MIN_SEC",
                                    DEFAULT_FREEZE_MIN_SEC, minimum=0.5)
        freeze_diff = _float_env("EGODATA_VIDEO_QC_FREEZE_DIFF",
                                 DEFAULT_FREEZE_DIFF, minimum=0.0)
        previous_small = None
        previous_index = None
        freeze_start = None
        freeze_last = None
        black_start = None
        black_last = None

        for index in indices:
            # Seeking to sparse positions avoids a full decode pass.  Each
            # seek is still followed by read(), so corrupt samples are caught.
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                report["decode_error_count"] += 1
                if len(report["decode_errors"]) < 20:
                    report["decode_errors"].append(index)
                previous_small = None
                previous_index = None
                freeze_start = None
                freeze_last = None
                black_start = None
                black_last = None
                continue

            report["sampled_frames"] += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dark_ratio = float(np.mean(gray <= black_mean))
            gray_mean = float(np.mean(gray))
            if gray_mean <= black_mean and dark_ratio >= black_dark_ratio:
                report["black_frame_count"] += 1
                if black_start is None:
                    black_start = index
                black_last = index
                if len(report["black_frames"]) < 20:
                    report["black_frames"].append({
                        "frame": index,
                        "time_sec": round(index / max(1.0, fps), 3),
                        "mean": round(gray_mean, 3),
                        "dark_ratio": round(dark_ratio, 6),
                    })
            elif black_start is not None:
                if ((black_last - black_start) / max(1.0, fps)
                        >= black_min_sec):
                    _add_freeze_range(report["black_ranges"],
                                      black_start, black_last, fps)
                black_start = None
                black_last = None

            small = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
            small = small.astype(np.float32)
            if previous_small is not None and previous_index is not None:
                difference = float(np.mean(np.abs(small - previous_small)))
                if difference <= freeze_diff:
                    if freeze_start is None:
                        freeze_start = previous_index
                    freeze_last = index
                    duration = (index - freeze_start) / max(1.0, fps)
                    if duration >= freeze_min_sec:
                        _add_freeze_range(report["freeze_ranges"],
                                          freeze_start, freeze_last, fps)
                else:
                    freeze_start = None
                    freeze_last = None
            previous_small = small
            previous_index = index

        if black_start is not None and ((black_last - black_start)
                                        / max(1.0, fps) >= black_min_sec):
            _add_freeze_range(report["black_ranges"],
                              black_start, black_last, fps)

        expected = max(0, int(expected_frames or 0))
        if expected > 0:
            tolerance = max(
                DEFAULT_FRAME_TOLERANCE_MIN,
                int(round(expected * _float_env(
                    "EGODATA_VIDEO_QC_FRAME_TOLERANCE_RATIO",
                    DEFAULT_FRAME_TOLERANCE_RATIO,
                    minimum=0.0,
                ))),
            )
            report["count_tolerance"] = tolerance
            deficit = max(0, expected - frame_count)
            report["dropped_frame_estimate"] = deficit
            report["frame_count_mismatch"] = bool(deficit > tolerance)
        else:
            report["frame_count_mismatch"] = False

        reasons: list[str] = []
        if report["decode_error_count"]:
            reasons.append("decode_error")
        if report["black_ranges"]:
            reasons.append("black_screen")
        if report["freeze_ranges"]:
            reasons.append("freeze_suspected")
        if report.get("frame_count_mismatch"):
            reasons.append("frame_count_short")
        report["passed"] = not reasons
        report["reason"] = "passed" if report["passed"] else ",".join(reasons)
        report["policy"] = {
            "black_mean": black_mean,
            "black_dark_ratio": black_dark_ratio,
            "black_min_sec": black_min_sec,
            "freeze_min_sec": freeze_min_sec,
            "freeze_diff": freeze_diff,
        }
        return report
    except Exception as exc:
        report["reason"] = "video_check_failed"
        report["error"] = str(exc)[:240]
        return report
    finally:
        cap.release()


def check_video_quality(batch_dir: str | Path, expected_frames: int = 0,
                        expected_fps: float = 0.0) -> dict:
    """Run the bounded post-processing video quality check for a batch."""
    root = Path(batch_dir)
    try:
        videos = _raw_videos(root)
    except Exception as exc:
        return {
            "passed": False,
            "reason": "video_scan_failed",
            "error": str(exc)[:240],
            "streams": [],
            "expected_frame_count": max(0, int(expected_frames or 0)),
        }
    if not videos:
        return {
            "passed": False,
            "reason": "no_input_video",
            "streams": [],
            "expected_frame_count": max(0, int(expected_frames or 0)),
        }

    streams = [_check_stream(path, expected_frames, expected_fps)
               for path in videos]
    passed = all(bool(stream.get("passed")) for stream in streams)
    return {
        "passed": passed,
        "reason": "passed" if passed else "video_quality_failed",
        "streams": streams,
        "stream_count": len(streams),
        "expected_frame_count": max(0, int(expected_frames or 0)),
        "expected_fps": round(max(0.0, float(expected_fps or 0.0)), 4),
    }


async def check_video_quality_async(batch_dir: str | Path,
                                   expected_frames: int = 0,
                                   expected_fps: float = 0.0) -> dict:
    """Serialize bounded media scans so multiple completions do not flood IO."""
    async with _VIDEO_QC_SEMAPHORE:
        return await asyncio.to_thread(
            check_video_quality, batch_dir, expected_frames, expected_fps)


def _apply_video_quality_result(episode_id: str, report: dict) -> None:
    """Persist the result while preserving the two-state review UI."""
    state = read_episode_state(episode_id)
    state["video_quality_status"] = "passed" if report.get("passed") else "failed"
    state["video_quality_report"] = report
    current = str(state.get("status") or "")
    if report.get("passed"):
        if current in {"to_review", "completed", "reviewed"}:
            state["status"] = "reviewed"
            if not state.get("approved_at"):
                state["approved_at"] = _utcnow()
    elif current in {"processing", "completed", "reviewed", "approved", "failed"}:
        state["status"] = "to_review"
        state["approved_at"] = None
    write_episode_state(episode_id, state)


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


async def run_video_quality_review(episode_id: str, batch_dir: str | Path,
                                  expected_frames: int = 0,
                                  expected_fps: float = 0.0) -> dict:
    """Run media checks off the event loop and persist the review decision."""
    try:
        report = await check_video_quality_async(
            batch_dir, expected_frames, expected_fps)
    except Exception as exc:
        # A background quality task must never become an unhandled asyncio
        # exception and accidentally leave an old Approved state in place.
        report = {
            "passed": False,
            "reason": "video_check_failed",
            "error": str(exc)[:240],
            "streams": [],
        }
    _apply_video_quality_result(episode_id, report)
    return report
