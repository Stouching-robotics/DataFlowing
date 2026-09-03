"""Browser-compatible video preview cache.

The collector and LeRobot dataset may use HEVC (including 12-bit depth
streams), but the Review page must not depend on the browser being able to
decode those codecs. This module creates a local, derived H.264 preview on
first use and reuses it for subsequent range requests.

The source file is never changed. The preview is a disposable local cache;
the authoritative recording and the lossless metric-depth asset remain in
the configured storage directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path

from app.config import settings


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _cache_root() -> Path:
    root = settings.upload_staging_root / "browser-preview"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(source: Path) -> str:
    stat = source.stat()
    fingerprint = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _is_h264_yuv420(source: Path) -> bool:
    """Avoid re-encoding an already browser-compatible source."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt",
             "-of", "csv=p=0", str(source)],
            check=True, capture_output=True, text=True, timeout=15,
        )
        codec, _, pix_fmt = result.stdout.strip().partition(",")
        return codec.lower() == "h264" and pix_fmt in {"yuv420p", "yuvj420p"}
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def ensure_h264_preview(source: Path) -> Path:
    """Return an H.264/yuv420p MP4 suitable for HTML5 playback.

    Callers should run this function via ``asyncio.to_thread`` so a cold
    transcode never blocks FastAPI's event loop.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Video source not found: {source}")
    if _is_h264_yuv420(source):
        return source

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to build browser video previews")

    key = _cache_key(source)
    destination = _cache_root() / f"{key}.mp4"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    with _lock_for(key):
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        # Keep the final .mp4 suffix so ffmpeg can infer the output muxer.
        temporary = destination.with_name(
            f".{destination.stem}.{os.getpid()}.part.mp4")
        temporary.unlink(missing_ok=True)
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
                 "-map", "0:v:0", "-an",
                 "-c:v", "libx264", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p",
                 # Short GOP keeps frame stepping and range seeks responsive.
                 "-g", "25", "-keyint_min", "25", "-sc_threshold", "0",
                 "-movflags", "+faststart", str(temporary)],
                check=True, capture_output=True, timeout=3600,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("ffmpeg produced an empty browser preview")
            os.replace(temporary, destination)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("H.264 preview transcode timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"H.264 preview transcode failed: {detail}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return destination
