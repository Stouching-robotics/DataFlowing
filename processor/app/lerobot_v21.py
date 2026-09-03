"""LeRobot v2.1 storage helpers used by the upload pipeline.

The collector currently sends a small v3-style archive for one episode.  The
server normalizes each upload into the project-level ``sessions/<project>``
dataset and appends it to the v2.1 layout:

    data/chunk-000/episode_XXXXXX.parquet
    meta/{info.json,stats.json,tasks.json,episodes/chunk-*/episode_*.parquet,...}
    videos/observation.images.<source>/chunk-000/episode_XXXXXX.mp4

Workflow results are merged into the corresponding episode parquet files
under the canonical ``data/`` and ``meta/`` directories.  Preview videos are
rendered in the browser and are never stored in the dataset.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from uuid import uuid4
from pathlib import Path
from typing import Any, Iterable


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}
DEPTH_PNG_SUFFIX = ".png"

# Canonical depth-video contract.  The stored gray12le samples are logarithmic
# depth codes, never JET/RGB pixels and never direct millimetres.
DEPTH_MIN_MM = 100.0
DEPTH_MAX_MM = 5000.0
DEPTH_QMAX = 4095
DEPTH_QP = 6
DEPTH_VIDEO_ENCODING = "depth_mm_log_to_gray12le"
# Kept as aliases for callers that imported the old names.
DEPTH_VIDEO_MAX_MM = DEPTH_QMAX
DEPTH_PREVIEW_MIN_MM = DEPTH_MIN_MM
DEPTH_PREVIEW_MAX_MM = DEPTH_MAX_MM
EPISODES_METADATA_CHUNK_SIZE = 1000

_LOG_LO = math.log(DEPTH_MIN_MM)
_LOG_SPAN = math.log(DEPTH_MAX_MM) - _LOG_LO
_LOG_STEP = DEPTH_QMAX / _LOG_SPAN

# Historical collectors used ``D435_depth_rgb`` for the colour stream of a
# D435.  It is not a depth stream (``is_depth_source`` already excludes it),
# but keeping that name in the canonical dataset makes the RGB/depth split
# ambiguous.  New uploads and legacy-path discovery use the clear key below.
SOURCE_ALIASES = {
    "d435_depth_rgb": "D435_rgb",
    "d435_head_rgb": "D435_rgb",
    "d435_head_depth": "D435_depth",
}
DEVICE_NAME_ALIASES = {
    "d435_depth": "D435",
    "d435_head": "D435",
}


def depth_video_encoder_args() -> list[str]:
    """Return the exact FFmpeg arguments for canonical depth video output.

    keyint=2:官方 pyav 解码按"最近关键帧向后 seek"取帧,大 GOP(默认
    250)时 seek 会越过目标帧落在下一个关键帧 → FrameTimestampError
    (实测 870 帧视频仅 4 个关键帧,viz/训练在关键帧前 1-2 帧崩溃)。
    官方录制管线 GOP=2,这里对齐。
    """
    return [
        "-pix_fmt", "gray12le",
        "-tag:v", "hvc1",
        "-x265-params", f"qp={DEPTH_QP}:range=full:keyint=2:min-keyint=2",
    ]


def quantize_depth(depth_mm):
    """Convert uint16 millimetres to the canonical 12-bit log depth code."""
    import numpy as np

    mm = np.asarray(depth_mm, dtype=np.float64)
    valid = mm > 0
    codes = np.zeros(mm.shape, dtype="<u2")
    if valid.any():
        codes[valid] = np.clip(
            np.rint((np.log(mm[valid]) - _LOG_LO) * _LOG_STEP),
            0, DEPTH_QMAX,
        ).astype("<u2")
    return codes


def dequantize_depth(codes):
    """Convert canonical 12-bit log codes to rounded uint16 millimetres."""
    import numpy as np

    c = np.asarray(codes, dtype=np.float64)
    return np.rint(np.exp(_LOG_LO + c / _LOG_STEP)).astype("<u2")


def codes_to_heatmap_bgr(codes):
    """Convert canonical codes to the exact OpenCV JET BGR display image."""
    import cv2
    import numpy as np

    c8 = ((np.clip(codes, 0, DEPTH_QMAX).astype(np.int32) * 255)
          // DEPTH_QMAX).astype(np.uint8)
    return cv2.applyColorMap(c8, cv2.COLORMAP_JET)


def depth_to_heatmap_bgr(depth_mm):
    """Convert millimetres to the canonical preview-only OpenCV JET image."""
    return codes_to_heatmap_bgr(quantize_depth(depth_mm))


def _depth_video_info_from_path(path: Path) -> dict[str, Any]:
    """Read depth metadata beside a canonical video when available."""
    path = Path(path)
    source = next(
        (part[len("observation.images."):]
         for part in path.parts if part.startswith("observation.images.")),
        path.parent.name,
    )
    for parent in (path.parent, *path.parents):
        info_path = parent / "meta" / "info.json"
        if not info_path.is_file():
            continue
        info = _read_json(info_path)
        feature = (info.get("features") or {}).get(
            f"observation.images.{source}", {})
        video_info = feature.get("video_info") if isinstance(feature, dict) else None
        return dict(video_info) if isinstance(video_info, dict) else {}
    return {}


def _depth_video_mode(video_info: dict[str, Any]) -> str:
    """Return ``log``, ``legacy_log`` or ``direct_mm`` for a stored stream."""
    if not isinstance(video_info, dict):
        video_info = {}
    encoding = str(video_info.get("video.depth_encoding") or "").lower()
    # Older uploads explicitly identify the gray12le samples as direct
    # millimetres.  This check must happen before the generic gray12le
    # fallback below; otherwise a direct-mm stream is interpreted as a log
    # code stream and values such as 500 mm are decoded to the wrong depth.
    if "uint16_mm" in encoding and "gray12le" in encoding:
        return "direct_mm"
    if encoding == DEPTH_VIDEO_ENCODING or "log_to_gray12le" in encoding:
        return "log"
    # The current collector contract stores the logarithmic 12-bit code
    # directly in gray12le.  Some older ``meta/info.json`` files describe the
    # transport as ``uint16_mm_to_gray12le_clipped_4095mm`` even though the
    # pixels are already codes, not millimetres.  A pure depth gray12le
    # stream is therefore log-coded by default; otherwise values such as
    # code=1511 are incorrectly treated as 1511 mm instead of ~420 mm.
    if (video_info.get("video.is_depth_map")
            and "gray12le" in encoding):
        return "log"
    if video_info.get("video.use_log") and (
            video_info.get("video.depth_min") is not None
            and video_info.get("video.depth_max") is not None):
        return "legacy_log"
    # Existing project videos were written as direct millimetres.  Keeping
    # this fallback prevents an old recording from being silently interpreted
    # as a new logarithmic code stream before it is explicitly migrated.
    return "direct_mm"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _safe_source_key(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("observation.images."):
        value = value[len("observation.images."):]
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "camera"


def canonical_source_key(value: str) -> str:
    """Return the stable source key used by the v2.1 dataset.

    This is deliberately a narrow compatibility map.  Generic ``*_rgb`` and
    ``*_depth`` names must remain distinct because they describe different
    physical streams and are used by workflow matching.
    """
    safe = _safe_source_key(value)
    return SOURCE_ALIASES.get(safe.lower(), safe)


def canonical_device_name(value: str, kind: str = "", key: str = "") -> str:
    """Return a stable hardware name without changing stream identifiers."""
    text = str(value or "")
    if (text.strip().lower() in DEVICE_NAME_ALIASES
            and "d435" in f"{kind} {key}".lower()):
        return DEVICE_NAME_ALIASES[text.strip().lower()]
    return text


def normalize_metadata_sources(document: Any) -> dict[str, Any]:
    """Normalize source aliases and duplicate physical-device records.

    This runs on each newly uploaded episode before it is merged into the
    project dataset, so an old collector cannot reintroduce ``D435_head_*``
    keys or a second device entry after a one-time migration.
    """
    if not isinstance(document, dict):
        return document
    result = dict(document)
    for field in ("features", "cameras", "device_names", "video_extensions"):
        value = result.get(field)
        if not isinstance(value, dict):
            continue
        mapped: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            prefix = "observation.images."
            source = key[len(prefix):] if field == "features" and key.startswith(prefix) else key
            canonical = canonical_source_key(source)
            new_key = prefix + canonical if field == "features" and key.startswith(prefix) else canonical
            # A canonical entry is authoritative if both old and new names
            # exist in one metadata document.
            if new_key not in mapped or source == canonical:
                mapped[new_key] = child
        result[field] = mapped

    devices = result.get("devices")
    if isinstance(devices, list):
        merged: list[dict[str, Any]] = []
        positions: dict[str, int] = {}
        for raw in devices:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            kind = str(item.get("kind") or "")
            device_key = str(item.get("key") or item.get("id") or "")
            if "name" in item:
                item["name"] = canonical_device_name(item["name"], kind, device_key)
            slots = [canonical_source_key(slot) for slot in item.get("slots") or []
                     if str(slot).strip()]
            item["slots"] = list(dict.fromkeys(slots))
            for field in ("resolution", "fps"):
                mapping = item.get(field)
                if isinstance(mapping, dict):
                    item[field] = {
                        canonical_source_key(k): v for k, v in mapping.items()
                    }
            if device_key and device_key in positions:
                current = merged[positions[device_key]]
                current["slots"] = list(dict.fromkeys(
                    [*(current.get("slots") or []), *(item.get("slots") or [])]
                ))
                for field in ("resolution", "fps"):
                    current.setdefault(field, {})
                    current[field].update(item.get(field) or {})
                for field, value in item.items():
                    if field not in {"slots", "resolution", "fps"} and not current.get(field):
                        current[field] = value
                continue
            if device_key:
                positions[device_key] = len(merged)
            merged.append(item)
        result["devices"] = merged

    device_names = result.get("device_names")
    if isinstance(device_names, dict):
        result["device_names"] = {
            canonical_source_key(stream): canonical_device_name(
                display, "d435", str(stream),
            )
            for stream, display in device_names.items()
        }
    cameras = result.get("cameras")
    if isinstance(cameras, dict):
        normalized_cameras = {}
        for stream, value in cameras.items():
            if isinstance(value, dict):
                value = dict(value)
                for field in ("device", "device_name"):
                    if field in value:
                        value[field] = canonical_device_name(
                            value[field], str(value.get("type") or "d435"), str(stream),
                        )
            normalized_cameras[canonical_source_key(stream)] = value
        result["cameras"] = normalized_cameras
    return result


def source_key_from_video(path: Path, videos_root: Path | None = None) -> str:
    """Extract the source key from legacy, v2.1 and collector-v3 paths."""
    path = Path(path)
    if videos_root is None:
        parts = path.parts
        try:
            index = max(i for i, value in enumerate(parts) if value == "videos")
            videos_root = Path(*parts[:index + 1])
        except ValueError:
            videos_root = path.parent

    try:
        relative = path.relative_to(videos_root)
        parts = relative.parts
    except ValueError:
        parts = path.parts

    # v2.1: videos/observation.images.left/chunk-000/episode_000000.mp4
    # v3.0: videos/chunk-000/observation.images.left/file-000.mp4
    for part in parts[:-1]:
        if part.startswith("observation.images."):
            return canonical_source_key(part)

    # Collector source directories may omit the observation.images prefix.
    # Keep parsing them for normalization, but all newly written output uses
    # one of the two canonical layouts above.
    if len(parts) >= 2:
        candidate = parts[-2]
        if candidate.startswith("chunk-") or candidate.startswith("chunk_"):
            candidate = parts[-3] if len(parts) >= 3 else path.stem
        return canonical_source_key(candidate)
    return canonical_source_key(path.stem)


def is_depth_source(source_key: str | None) -> bool:
    """Return true only for a pure depth stream, not ``*_depth_rgb``."""
    value = str(source_key or "").lower().strip()
    if not value or "depth" not in value:
        return False
    return not any(token in value for token in ("rgb", "color", "video"))


def iter_video_streams(videos_root: Path) -> list[tuple[str, Path]]:
    """Return one stable ``(source_key, path)`` entry per video stream.

    The normalizer writes one file per source for a single uploaded batch.  If
    a malformed archive contains multiple files for one source, the caller
    can detect the duplicate instead of silently mixing episodes.
    """
    if not videos_root.is_dir():
        return []
    result: list[tuple[str, Path]] = []
    for path in sorted(videos_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            result.append((source_key_from_video(path, videos_root), path))
    return result


def _episode_index_from_name(batch_name: str) -> int | None:
    match = re.search(r"_(\d{6})$", str(batch_name))
    return int(match.group(1)) if match else None


def _data_parquets(root: Path) -> list[Path]:
    data_root = root / "data"
    if not data_root.is_dir():
        return []
    return sorted(
        path for path in data_root.rglob("*.parquet")
        if path.is_file() and "meta" not in path.parts
    )


def _load_frame_table(paths: list[Path]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for LeRobot v2.1 normalization") from exc

    candidates = []
    for path in paths:
        try:
            table = pd.read_parquet(path)
        except Exception as exc:
            print(f"[LeRobot v2.1] skip unreadable parquet {path.name}: {exc}")
            continue
        if "frame_index" in table.columns:
            candidates.append((len(table), path, table))
    if not candidates:
        raise RuntimeError("No data parquet containing frame_index was found")

    # The main frame table is normally the largest one.  Additional parquet
    # files (for example a glove sensor stream) are merged by frame_index.
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    _, primary_path, frame_table = candidates[0]
    for _, extra_path, extra in reversed(candidates[1:]):
        if extra_path == primary_path:
            continue
        columns = [c for c in extra.columns
                   if c != "frame_index" and c not in frame_table.columns]
        if not columns:
            continue
        frame_table = frame_table.merge(
            extra[["frame_index", *columns]], on="frame_index", how="left",
        )
    return frame_table


def _legacy_frame_table(root: Path, info: dict[str, Any], batch_name: str):
    """Build a truthful frame table for older collector archives.

    Some early RGB-only uploads contain ``timestamps.json`` and episode
    summary parquet files, but no frame-level ``data`` parquet.  We preserve
    the available timeline instead of treating a processed hand-keypoint
    parquet as source data.  Missing action/IMU values are deliberately not
    fabricated.
    """
    import pandas as pd

    timestamp_paths = [root / "timestamps.json",
                       root / "meta" / "collector" / "timestamps.json"]
    timestamp_rows: list[dict[str, Any]] = []
    for path in timestamp_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        rows = payload.get("timestamps") if isinstance(payload, dict) else payload
        if isinstance(rows, list):
            timestamp_rows = [row for row in rows if isinstance(row, dict)]
            if timestamp_rows:
                break

    episode_index = _episode_index_from_name(batch_name)
    length = 0
    for path in sorted((root / "meta").rglob("*.parquet")) if (root / "meta").is_dir() else []:
        try:
            table = pd.read_parquet(path)
        except Exception:
            continue
        if "length" not in table.columns or table.empty:
            continue
        row = table.iloc[0]
        try:
            length = int(row.get("length") or 0)
        except (TypeError, ValueError):
            length = 0
        if episode_index is None:
            try:
                value = int(row.get("episode_index"))
                if value >= 0:
                    episode_index = value
            except (TypeError, ValueError):
                pass
        if length:
            break

    if timestamp_rows:
        frame_table = pd.DataFrame(timestamp_rows)
    else:
        if length <= 0:
            # Last-resort count from the actual source video.  This is still
            # source-derived and does not invent any sensor or pose values.
            length = sum(_probe_video(path, float(info.get("fps") or 30)).get("frames", 0)
                         for _source, path in iter_video_streams(root / "videos"))
        if length <= 0:
            raise RuntimeError("No frame-level data or timestamps found")
        fps = float(info.get("fps") or 30.0)
        frame_table = pd.DataFrame({
            "frame_index": list(range(length)),
            "timestamp": [index / fps for index in range(length)],
        })

    if "frame_index" not in frame_table.columns:
        frame_table.insert(0, "frame_index", range(len(frame_table)))
    if episode_index is None:
        episode_index = 0
    frame_table["episode_index"] = int(episode_index)
    return frame_table


def _infer_episode_index(root: Path, batch_name: str, frame_table) -> int:
    if "episode_index" in frame_table.columns and len(frame_table):
        try:
            value = int(frame_table["episode_index"].iloc[0])
            if value >= 0:
                return value
        except (TypeError, ValueError, IndexError):
            pass
    episode_index = _episode_index_from_name(batch_name)
    return episode_index if episode_index is not None else 0


def _probe_video(path: Path, fallback_fps: float = 30.0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frames": 0, "fps": float(fallback_fps or 30.0),
        "width": 0, "height": 0, "codec": "", "pix_fmt": "",
    }
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        result.update({
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or fallback_fps or 30.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        })
        cap.release()
    except Exception:
        pass
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt,width,height",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        streams = json.loads(completed.stdout or "{}").get("streams") or []
        if streams:
            stream = streams[0]
            result["codec"] = str(stream.get("codec_name") or "")
            result["pix_fmt"] = str(stream.get("pix_fmt") or "")
            result["width"] = int(stream.get("width") or result["width"] or 0)
            result["height"] = int(stream.get("height") or result["height"] or 0)
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        pass
    return result


def depth_png_directories(root: Path) -> list[tuple[str, Path, list[Path]]]:
    """Find numeric depth-frame directories in a legacy upload tree."""
    root = Path(root)
    if not root.is_dir():
        return []
    result: list[tuple[str, Path, list[Path]]] = []
    seen: set[str] = set()
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        frames = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == DEPTH_PNG_SUFFIX
            and path.stem.isdigit()
        )
        if not frames:
            continue
        source = _safe_source_key(directory.name)
        if source in seen:
            raise RuntimeError(
                f"Multiple depth frame directories use the same source key: {source}"
            )
        numbers = [int(path.stem) for path in frames]
        expected = list(range(numbers[0], numbers[0] + len(numbers)))
        if numbers != expected:
            raise RuntimeError(
                f"Depth frame sequence has gaps for {source}: "
                f"expected consecutive files, got {frames[0].name}..{frames[-1].name}"
            )
        seen.add(source)
        result.append((source, directory, frames))
    return result


def encode_depth_png_sequence(depth_dir: Path, frames: list[Path], destination: Path,
                              fps: float = 30.0) -> dict[str, Any]:
    """Encode uint16 millimetre PNGs using the native depth-video contract.

    The generated stream contains 12-bit logarithmic depth codes, not a
    colourized preview.  Quantization is explicit so FFmpeg cannot silently
    rescale uint16 PNG values.  CQP qp=6 and full range are part of the
    interchange contract; CRF and lossless mode are intentionally avoided.
    """
    import cv2
    import numpy as np

    if not frames:
        raise RuntimeError(f"No depth frames found in {depth_dir}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert depth PNGs to video")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    first_number = int(frames[0].stem)
    sample = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
    if sample is None or sample.ndim != 2 or sample.dtype != np.uint16:
        raise RuntimeError(
            f"Depth frame must be a single-channel uint16 PNG: {frames[0]}"
        )
    height, width = sample.shape
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{uuid4().hex}{destination.suffix or '.mp4'}"
    )
    process = None
    try:
        process = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "gray12le",
             "-s", f"{width}x{height}", "-r", str(float(fps or 30.0)),
             "-i", "pipe:0", "-frames:v", str(len(frames)),
             "-an", "-c:v", "libx265",
             *depth_video_encoder_args(), "-movflags", "+faststart",
             str(temporary)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        for frame_path in frames:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if frame is None or frame.ndim != 2 or frame.dtype != np.uint16:
                raise RuntimeError(
                    f"Depth frame must be a single-channel uint16 PNG: {frame_path}"
                )
            if frame.shape != sample.shape:
                raise RuntimeError(
                    f"Depth frame resolution mismatch: {frame_path} "
                    f"has {frame.shape}, expected {sample.shape}"
                )
            # Explicitly provide canonical 12-bit log samples. Passing the
            # PNG pattern to FFmpeg would silently rescale uint16 values.
            codes = quantize_depth(frame)
            process.stdin.write(np.ascontiguousarray(codes).tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait(timeout=1800)
        if return_code != 0 or not temporary.is_file() \
                or temporary.stat().st_size <= 0:
            detail = (stderr.decode(errors="replace")
                      or "ffmpeg returned no output").strip()
            raise RuntimeError(f"Depth video encoding failed: {detail[-1000:]}")
        probe = _probe_video(temporary, float(fps or 30.0))
        if probe.get("codec") not in {"hevc", "h265"} \
                or probe.get("pix_fmt") != "gray12le":
            raise RuntimeError(
                "Depth video verification failed: "
                f"codec={probe.get('codec')!r}, pix_fmt={probe.get('pix_fmt')!r}"
            )
        if int(probe.get("width") or 0) <= 0 or int(probe.get("height") or 0) <= 0:
            raise RuntimeError("Depth video verification failed: invalid dimensions")
        temporary.replace(destination)
        probe["frames"] = int(probe.get("frames") or len(frames))
        probe["fps"] = float(probe.get("fps") or fps or 30.0)
        probe["source_frame_start"] = first_number
        probe["source_frame_count"] = len(frames)
        probe["depth_encoding"] = DEPTH_VIDEO_ENCODING
        probe["depth_min_mm"] = DEPTH_MIN_MM
        probe["depth_max_mm"] = DEPTH_MAX_MM
        probe["depth_qmax"] = DEPTH_QMAX
        probe["depth_qp"] = DEPTH_QP
        return probe
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise RuntimeError(f"Depth video encoding failed: {exc}") from exc
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)


class DepthVideoReader:
    """Sequentially decode a depth stream as raw codes or metric millimetres."""

    def __init__(self, path: Path):
        self.path = Path(path)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to read metric depth video")
        probe = _probe_video(self.path)
        pix_fmt = str(probe.get("pix_fmt") or "")
        if pix_fmt not in {"gray12le", "gray16le"}:
            raise RuntimeError(
                f"Metric depth video must use gray12le/gray16le, got {pix_fmt!r}"
            )
        self.width = int(probe.get("width") or 0)
        self.height = int(probe.get("height") or 0)
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"Metric depth video has invalid dimensions: {self.path}")
        self.frame_count = int(probe.get("frames") or 0)
        self.fps = float(probe.get("fps") or 30.0)
        self.video_info = _depth_video_info_from_path(self.path)
        self.mode = _depth_video_mode(self.video_info)
        self._frame_bytes = self.width * self.height * 2
        self._process = subprocess.Popen(
            [ffmpeg, "-v", "error", "-i", str(self.path),
             "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def read_codes(self):
        """Read one stored little-endian uint16 frame without colorization."""
        import numpy as np

        if self._process.stdout is None:
            return None
        chunks: list[bytes] = []
        remaining = self._frame_bytes
        while remaining:
            data = self._process.stdout.read(remaining)
            if not data:
                return None
            chunks.append(data)
            remaining -= len(data)
        return np.frombuffer(b"".join(chunks), dtype="<u2").reshape(
            self.height, self.width,
        ).copy()

    def read_mm(self):
        """Read one frame as uint16 millimetres for metric processing."""
        import numpy as np

        codes = self.read_codes()
        if codes is None:
            return None
        return self._codes_to_mm(codes)

    def _codes_to_mm(self, codes):
        import numpy as np

        if self.mode == "log":
            depth = dequantize_depth(codes)
            # The canonical display maps code 0 to JET(0), but metric
            # processing must retain its invalid-pixel semantics.
            depth[codes == 0] = 0
            return depth
        if self.mode == "legacy_log":
            low = float(self.video_info["video.depth_min"])
            high = float(self.video_info["video.depth_max"])
            shift = float(self.video_info.get("video.shift") or 0.0)
            log_low = math.log(low + shift)
            log_span = math.log(high + shift) - log_low
            depth_m = np.exp(log_low + codes.astype(np.float64) / DEPTH_QMAX * log_span) - shift
            depth_m[codes == 0] = 0.0
            return np.rint(np.clip(depth_m * 1000.0, 0, 65535)).astype("<u2")
        return codes

    def read_canonical_codes(self):
        """Read a frame as canonical codes without creating a color image.

        New streams already contain canonical codes.  Known legacy direct-mm
        and legacy-log streams are converted in memory for compatibility; no
        converted frame is written back to the dataset.
        """
        codes = self.read_codes()
        if codes is None:
            return None
        if self.mode == "log":
            import numpy as np
            return np.clip(codes, 0, DEPTH_QMAX).astype("<u2", copy=False)
        return quantize_depth(self._codes_to_mm(codes))

    def read(self):
        """Backward-compatible metric read used by 3D processing modules."""
        return self.read_mm()

    def close(self) -> None:
        process = self._process
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()


def _to_mp4(source: Path, destination: Path) -> Path:
    """Put a video in the v2.1 MP4 path, preserving codec when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return source
    if source.suffix.lower() == ".mp4":
        shutil.copy2(source, destination)
        return destination
    try:
        completed = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
             "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart",
             str(destination)],
            capture_output=True, timeout=1800, check=False,
        )
        if completed.returncode == 0 and destination.is_file():
            return destination
    except (OSError, subprocess.TimeoutExpired):
        pass
    # Keep the original codec/container rather than discarding a depth stream.
    fallback = destination.with_suffix(source.suffix.lower())
    shutil.copy2(source, fallback)
    return fallback


def _canonicalize_depth_video(source: Path, destination: Path,
                              fps: float) -> dict[str, Any]:
    """Transcode a legacy gray depth video to canonical log depth codes."""
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to canonicalize depth video")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{uuid4().hex}{destination.suffix or '.mp4'}"
    )
    reader = DepthVideoReader(source)
    process = None
    written = 0
    try:
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray12le",
            "-s", f"{reader.width}x{reader.height}",
            "-r", str(float(fps or reader.fps or 30.0)), "-i", "pipe:0",
            "-an", "-c:v", "libx265", *depth_video_encoder_args(),
            "-movflags", "+faststart", str(temporary),
        ]
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        while True:
            codes = reader.read_canonical_codes()
            if codes is None:
                break
            process.stdin.write(np.ascontiguousarray(codes, dtype="<u2").tobytes())
            written += 1
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait(timeout=1800)
        if return_code != 0 or written <= 0 or not temporary.is_file():
            detail = (stderr.decode(errors="replace")
                      or "ffmpeg returned no output").strip()
            raise RuntimeError(f"Depth video canonicalization failed: {detail[-1000:]}")
        probe = _probe_video(temporary, float(fps or reader.fps or 30.0))
        if probe.get("codec") not in {"hevc", "h265"} \
                or probe.get("pix_fmt") != "gray12le":
            raise RuntimeError(
                "Depth video canonicalization verification failed: "
                f"codec={probe.get('codec')!r}, pix_fmt={probe.get('pix_fmt')!r}"
            )
        temporary.replace(destination)
        probe.update({
            "frames": int(probe.get("frames") or written),
            "fps": float(probe.get("fps") or fps or reader.fps or 30.0),
            "depth_encoding": DEPTH_VIDEO_ENCODING,
            "depth_min_mm": DEPTH_MIN_MM,
            "depth_max_mm": DEPTH_MAX_MM,
            "depth_qmax": DEPTH_QMAX,
            "depth_qp": DEPTH_QP,
        })
        return probe
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise RuntimeError(f"Depth video canonicalization failed: {exc}") from exc
    finally:
        reader.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)


def _source_feature(info: dict, source: str, probe: dict, metric_depth: bool) -> dict:
    key = f"observation.images.{source}"
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    original = features.get(key) if isinstance(features.get(key), dict) else {}
    original_video = original.get("video_info")
    video_info = dict(original_video) if isinstance(original_video, dict) else {}
    is_depth = bool(video_info.get("video.is_depth_map")) or metric_depth
    video_info.update({
        "video.fps": probe["fps"],
        "video.height": probe["height"],
        "video.width": probe["width"],
        "video.channels": 1 if is_depth else 3,
        "video.codec": probe.get("codec") or video_info.get("video.codec") or "",
        "video.pix_fmt": probe.get("pix_fmt") or video_info.get("video.pix_fmt") or "",
        "video.is_depth_map": is_depth,
        "video.is_depth_visualization": bool(is_depth is False and is_depth_source(source)),
        "has_audio": False,
    })
    if metric_depth:
        video_info.update({
            "video.depth_min_mm": DEPTH_MIN_MM,
            "video.depth_max_mm": DEPTH_MAX_MM,
            "video.depth_qmax": DEPTH_QMAX,
            "video.depth_qp": DEPTH_QP,
            "video.depth_quantization": "log",
            "video.depth_encoding": probe.get("depth_encoding") or DEPTH_VIDEO_ENCODING,
        })
    feature = dict(original)
    feature.update({
        "dtype": "video",
        "shape": [probe["height"], probe["width"], video_info["video.channels"]],
        "names": ["height", "width", "channel"],
        "video_info": video_info,
    })
    return feature


def _read_task_rows(meta_root: Path, fallback: str) -> tuple[dict[int, str], str]:
    tasks: dict[int, str] = {}
    json_path = meta_root / "tasks.json"
    json_value = _read_json(json_path)
    if isinstance(json_value, dict):
        json_value = json_value.get("tasks", json_value)
    if isinstance(json_value, list):
        for default_index, row in enumerate(json_value):
            if isinstance(row, str):
                index, text = default_index, row.strip()
            elif isinstance(row, dict):
                try:
                    index = int(row.get("task_index", row.get("task_id", default_index)))
                except (TypeError, ValueError):
                    index = default_index
                text = str(row.get("task") or row.get("description") or "").strip()
            else:
                continue
            if text:
                tasks[index] = text
    for filename in ("tasks.jsonl",):
        if tasks:
            break
        path = meta_root / filename
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(row, dict):
                continue
            try:
                index = int(row.get("task_index", row.get("task_id", 0)))
            except (TypeError, ValueError):
                index = 0
            text = str(row.get("task") or row.get("description") or "").strip()
            if text:
                tasks[index] = text
    if not tasks:
        tasks[0] = fallback or "default_recording"
    return tasks, tasks[min(tasks)]


def _episode_stats(frame_table) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for column in frame_table.columns:
        try:
            series = frame_table[column]
            if not getattr(series.dtype, "kind", "") in "biuf":
                continue
            values = series.to_numpy(dtype=float)
            if not len(values):
                continue
            import numpy as np
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            stats[column] = {
                "min": float(finite.min()), "max": float(finite.max()),
                "mean": float(finite.mean()), "std": float(finite.std()),
                "count": [int(len(finite))],
            }
        except Exception:
            continue
    return stats


def _write_episode_index(meta_root: Path,
                         rows: Iterable[dict[str, Any]]) -> None:
    """Write the canonical sharded Parquet episode index for one upload."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    episodes_root = Path(meta_root) / "episodes"
    if episodes_root.is_dir():
        shutil.rmtree(episodes_root)
    normalized = [dict(row) for row in rows]
    normalized.sort(key=lambda row: int(row.get("episode_index", 10**9)))
    for row in normalized:
        episode_index = int(row.get("episode_index", 0))
        chunk = episode_index // EPISODES_METADATA_CHUNK_SIZE
        target = (episodes_root / f"chunk-{chunk:03d}"
                  / f"episode_{episode_index:06d}.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist([row]),
            target,
        )


def normalize_extracted_dataset(root: Path, batch_name: str) -> dict[str, Any]:
    """Normalize one extracted collector archive to LeRobot v2.1 in-place."""
    root = Path(root)
    meta_root = root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    info = normalize_metadata_sources(_read_json(meta_root / "info.json"))
    fallback_task = str(info.get("task_name") or batch_name)
    data_paths = _data_parquets(root)
    try:
        frame_table = _load_frame_table(data_paths)
    except RuntimeError:
        frame_table = _legacy_frame_table(root, info, batch_name)
    episode_index = _infer_episode_index(root, batch_name, frame_table)
    frame_table = frame_table.copy()
    frame_table["episode_index"] = episode_index
    if "frame_index" in frame_table.columns:
        frame_table = frame_table.sort_values("frame_index", kind="stable")
    frame_count = int(len(frame_table))
    fps = float(info.get("fps") or 30.0)

    # Replace all source data parquet files with one v2.1 episode parquet.
    data_target = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    data_target.parent.mkdir(parents=True, exist_ok=True)
    frame_table.to_parquet(data_target, index=False)
    for path in data_paths:
        if path.resolve() != data_target.resolve():
            path.unlink(missing_ok=True)
    for directory in sorted((root / "data").glob("**/*"), reverse=True):
        if directory.is_dir() and directory != data_target.parent:
            try:
                directory.rmdir()
            except OSError:
                pass

    # Normalize one video per source to the v2.1 path template.
    source_videos = iter_video_streams(root / "videos")
    normalized_video_sources = list(source_videos)
    grouped: dict[str, list[Path]] = {}
    for source, path in source_videos:
        grouped.setdefault(source, []).append(path)
    video_features: dict[str, dict] = {}
    video_extensions: dict[str, str] = {}
    video_meta: dict[str, dict[str, Any]] = {}
    videos_root = root / "videos"
    for source, paths in sorted(grouped.items()):
        if len(paths) > 1:
            raise RuntimeError(
                f"Multiple video files found for source {source}; "
                "one uploaded batch must contain one file per source"
            )
        source_path = paths[0]
        probe = _probe_video(source_path, fps)
        # A pure depth stream is metric only when its metadata/pixel format
        # proves it.  A normal yuv420p/colourized depth preview is not 3D data.
        old_feature = (info.get("features") or {}).get(
            f"observation.images.{source}", {})
        old_video = old_feature.get("video_info") if isinstance(old_feature, dict) else {}
        metric_depth = bool(isinstance(old_video, dict)
                            and old_video.get("video.is_depth_map"))
        metric_depth = metric_depth or (
            is_depth_source(source)
            and probe.get("pix_fmt") in {"gray12le", "gray16le", "gray10le"}
        )
        destination = (videos_root / f"observation.images.{_safe_source_key(source)}"
                       / "chunk-000" / f"episode_{episode_index:06d}.mp4")
        if metric_depth and _depth_video_mode(old_video) != "log":
            # Legacy gray16/gray12 streams may carry direct millimetres.  Do
            # not relabel them as log codes without converting the samples.
            probe = _canonicalize_depth_video(source_path, destination, fps)
            final_path = destination
        else:
            final_path = _to_mp4(source_path, destination)
        video_extensions[source] = final_path.suffix.lstrip(".")
        video_meta[source] = {
            "frames": probe["frames"] or frame_count,
            "fps": probe["fps"],
            "metric_depth": metric_depth,
            "source": source,
        }
        video_features[f"observation.images.{source}"] = _source_feature(
            info, source, probe, metric_depth,
        )

    # Legacy collectors may upload metric depth as depth/<source>/*.png.
    # Convert it before the legacy tree is removed.  New projects therefore
    # keep only one independent depth video beside RGB streams, while the
    # resolver continues to read old meta/depth PNGs from already-migrated
    # projects.
    depth_root = root / "depth"
    depth_directories = depth_png_directories(depth_root)
    for source, depth_dir, frames in depth_directories:
        if source in video_meta:
            # An explicitly supplied depth video is authoritative.  The PNG
            # copy is redundant and is removed only after the whole batch has
            # passed the output verification below.
            continue
        destination = (videos_root / f"observation.images.{_safe_source_key(source)}"
                       / "chunk-000" / f"episode_{episode_index:06d}.mp4")
        depth_probe = encode_depth_png_sequence(
            depth_dir, frames, destination, fps=float(info.get("fps") or fps or 30.0),
        )
        depth_scale = 0.001
        depth_feature = _source_feature(info, source, depth_probe, True)
        depth_video_info = depth_feature.setdefault("video_info", {})
        depth_video_info.update({
            "video.channels": 1,
            "video.is_depth_map": True,
            "video.is_depth_visualization": False,
            "video.depth_scale": depth_scale,
            "video.depth_encoding": DEPTH_VIDEO_ENCODING,
            "video.depth_min_mm": DEPTH_MIN_MM,
            "video.depth_max_mm": DEPTH_MAX_MM,
            "video.depth_qmax": DEPTH_QMAX,
            "video.depth_qp": DEPTH_QP,
            "video.depth_quantization": "log",
            "has_audio": False,
        })
        depth_feature["shape"] = [
            int(depth_probe.get("height") or 0),
            int(depth_probe.get("width") or 0), 1,
        ]
        video_features[f"observation.images.{source}"] = depth_feature
        video_extensions[source] = "mp4"
        video_meta[source] = {
            "frames": int(depth_probe.get("frames") or len(frames)),
            "fps": float(depth_probe.get("fps") or fps or 30.0),
            "metric_depth": True,
            "source": source,
            "depth_scale": depth_scale,
        }
        normalized_video_sources.append((source, destination))

    # Do not retain raw depth PNGs in a newly normalized batch.  This is
    # reached only after every sequence has produced and verified its video;
    # if conversion fails, normalization aborts and the upload staging tree
    # remains available for retry/diagnosis.
    if depth_root.is_dir():
        shutil.rmtree(depth_root)

    # Remove any source/legacy video tree only after all files have been
    # copied.  The canonical v2.1 layout is videos/<video_key>/chunk-000/.
    for child in list(videos_root.iterdir()) if videos_root.is_dir() else []:
        if not child.name.startswith("observation.images."):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            continue
        for nested in list(child.iterdir()) if child.is_dir() else []:
            if nested.name != "chunk-000":
                if nested.is_dir():
                    shutil.rmtree(nested, ignore_errors=True)
                else:
                    nested.unlink(missing_ok=True)

    tasks, default_task = _read_task_rows(meta_root, fallback_task)
    # Keep an explicit task file, but remove v3-only pooled episode metadata.
    for stale in (meta_root / "tasks.parquet",):
        stale.unlink(missing_ok=True)
    episode_row: dict[str, Any] = {
        "episode_index": episode_index,
        "tasks": [default_task],
        "task_index": min(tasks),
        "length": frame_count,
        "data/chunk_index": 0,
        "data/file_index": episode_index,
        "dataset_from_index": 0,
        "dataset_to_index": frame_count,
    }
    for source, values in sorted(video_meta.items()):
        key = f"observation.images.{source}"
        count = int(values.get("frames") or frame_count)
        stream_fps = float(values.get("fps") or fps)
        episode_row[f"videos/{key}/chunk_index"] = 0
        episode_row[f"videos/{key}/file_index"] = episode_index
        episode_row[f"videos/{key}/from_timestamp"] = 0.0
        episode_row[f"videos/{key}/to_timestamp"] = count / stream_fps if stream_fps else 0.0
    episode_stats = _episode_stats(frame_table)
    episode_row["stats"] = episode_stats
    _write_episode_index(meta_root, [episode_row])
    (meta_root / "episodes_stats.jsonl").unlink(missing_ok=True)
    tasks_path = meta_root / "tasks.json"
    tasks_path.write_text(
        json.dumps([{"task_index": index, "task": text}
                    for index, text in sorted(tasks.items())],
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (meta_root / "tasks.jsonl").unlink(missing_ok=True)

    # Move collector-only metadata under meta/, keeping the dataset root at
    # exactly data/meta/videos.  Calibration is metadata, not a fourth root.
    calibration_root = root / "calibration"
    if calibration_root.is_dir():
        target = meta_root / "calibration"
        target.mkdir(parents=True, exist_ok=True)
        for item in calibration_root.iterdir():
            shutil.move(str(item), str(target / item.name))
        calibration_root.rmdir()
    for filename in ("metadata.json", "timestamps.json"):
        source = root / filename
        if source.is_file():
            target = meta_root / "collector" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))

    # Keep the dataset root limited to data/meta/videos without hiding raw
    # depth frames from the processing pipeline.  PNG depth is auxiliary
    # source material, so it lives below meta/depth while the synchronized
    # video stream remains under videos/observation.images.<source>/.
    for item in list(root.iterdir()):
        if item.name in {"data", "meta", "videos"}:
            continue
        if item.name == "depth":
            target = meta_root / "depth"
        else:
            target = meta_root / "collector" / item.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"Metadata destination already exists: {target}")
        shutil.move(str(item), str(target))

    # Collector metadata is part of the dataset contract.  Normalize it too,
    # otherwise a historical upload could reintroduce old source/device names
    # after the active videos were already written with canonical keys.
    for metadata_path in meta_root.rglob("*.json"):
        metadata = _read_json(metadata_path)
        if not metadata:
            continue
        normalized_metadata = normalize_metadata_sources(metadata)
        _write_json(metadata_path, normalized_metadata)
    info = normalize_metadata_sources(info)

    features = dict(info.get("features") or {}) if isinstance(info.get("features"), dict) else {}
    features.update(video_features)
    normalized = dict(info)
    normalized.update({
        "format": "lerobot_v2.1",
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_frames": frame_count,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.{ext}",
        "features": features,
        "video_extensions": video_extensions,
        "extensions": {
            **(info.get("extensions") or {}),
            "episodes_file": "meta/episodes/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "stats_file": "meta/stats.json",
            "tasks_file": "meta/tasks.json",
        },
    })
    _write_json(meta_root / "info.json", normalized)

    # A standalone upload_manifest is useful for diagnostics but belongs in
    # meta/ so it does not create a fourth dataset-root directory.
    return {
        "episode_index": episode_index,
        "frame_count": frame_count,
        "fps": fps,
        "video_sources": sorted(source for source, _ in normalized_video_sources),
        "camera_sources": sorted(source for source, _ in normalized_video_sources
                                  if not is_depth_source(source)),
        "depth_sources": sorted(source for source, _ in normalized_video_sources
                                 if is_depth_source(source)),
        "video_meta": video_meta,
        "info": normalized,
    }
