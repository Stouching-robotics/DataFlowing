"""Episode APIs — 纯本地文件驱动(无数据库)。

episode 的 ID 是批次目录名(如 Test005_000030),数据源为
data/sessions/<项目>/<批次名>/ + data/state/(审核状态、标注)。
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Episode, Frame
from app.localstore import (
    scan_sessions, get_episode, read_episode_state, write_episode_state,
    set_episode_status, delete_episode as ls_delete_episode,
    restore_episode as ls_restore_episode, list_deleted_episodes,
    list_exceptions, delete_exception,
    episode_has_glove_sensor,
)
from app.auth import require_roles

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


def _ep_to_out(ep: dict) -> dict:
    """localstore 扫描结果 → 与旧 EpisodeOut 兼容的响应结构。"""
    state = read_episode_state(ep["id"])
    return {
        "id": ep["id"],
        "session_id": None,
        "name": ep["name"],
        "task_description": ep.get("project"),
        "task_id": None,
        "status": ep.get("status"),
        "frame_count": ep.get("frame_count") or 0,
        "fps": ep.get("fps") or 30,
        "camera_names": ep.get("camera_names") or [],
        "meta": {
            "timestamp": ep.get("timestamp", ""),
            "camera_streams": ep.get("camera_streams") or {},
            "camera_group": {"type": "single", "count": len(ep.get("camera_names") or [])},
            "episode_index": ep.get("episode_index"),
            "cleaning_report": state.get("cleaning_report"),
        },
        "cleaning_report": state.get("cleaning_report"),
        "received_at": ep.get("created_at"),
        "approved_at": state.get("approved_at"),
        "created_at": ep.get("created_at"),
        "updated_at": state.get("updated_at"),
        "deleted_at": state.get("deleted_at"),
    }


def _ep_to_out_brief(ep: dict) -> dict:
    """轻量序列化(列表视图)。"""
    state = read_episode_state(ep["id"])
    return {
        "id": ep["id"],
        "session_id": None,
        "name": ep["name"],
        "task_description": ep.get("project"),
        "task_id": None,
        "status": ep.get("status"),
        "frame_count": ep.get("frame_count") or 0,
        "fps": ep.get("fps") or 30,
        "camera_names": ep.get("camera_names") or [],
        "meta": {"timestamp": ep.get("timestamp", "")},
        "cleaning_report": None,
        "received_at": ep.get("created_at"),
        "approved_at": state.get("approved_at"),
        "created_at": ep.get("created_at"),
        "updated_at": state.get("updated_at"),
        "deleted_at": state.get("deleted_at"),
    }


# ── 列表 / 详情 ───────────────────────────────────────

@router.get("/episodes")
def episode_list(status: str | None = None, limit: int = 50, offset: int = 0,
                       brief: bool = False):
    """文件系统扫描的 episode 列表:文件删了记录自然消失。"""
    episodes = scan_sessions()
    if status == "deleted":
        deleted_ids = {d["id"] for d in list_deleted_episodes()}
        episodes = [e for e in episodes if e["id"] in deleted_ids]
    else:
        deleted_ids = {d["id"] for d in list_deleted_episodes()}
        episodes = [e for e in episodes if e["id"] not in deleted_ids]
        if status:
            if status == "completed":
                episodes = [e for e in episodes if e.get("status") in ("completed", "to_review")]
            else:
                episodes = [e for e in episodes if e.get("status") == status]
    total = len(episodes)
    episodes = episodes[offset:offset + limit]
    serialize = _ep_to_out_brief if brief else _ep_to_out
    return {"episodes": [serialize(e) for e in episodes], "total": total,
            "limit": limit, "offset": offset}


@router.get("/episode/{episode_id}")
def episode_detail(episode_id: str):
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return _ep_to_out(ep)


# ── 帧数据(热力图/传感器)────────────────────────────

_frames_data_cache: dict[str, dict] = {}
_MAX_FRAMES_CACHE = 3


def _col_has_pressure(parquet_path, col: str, sample_rows: int = 256) -> bool:
    """列是否存在真实压力数据(全文件**均匀采样**,不只看头部)。

    热力图显示 = 工作流含 glove_sensor 节点 **且** 列有真实压力数据:
    全零列(骨骼识别参数占位)不驱动热力图;有非零压力的真实传感器
    即使首非零帧在文件后半段(如 right_glove 第 68 帧)也能检出
    (旧实现 head(64) 会误杀)。
    """
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=[col])
        total = len(df)
        if total == 0:
            return False
        step = max(1, total // max(1, sample_rows))
        for val in df[col].iloc[::step]:
            if _any_nonzero(val):
                return True
    except Exception:
        return True   # 读取失败不误杀,保留该列
    return False


def _any_nonzero(value) -> bool:
    """递归检查嵌套数组是否含非零元素。

    pandas 读回嵌套 parquet 列时,元素可能是 (n,) 的 object ndarray
    (每个元素又是数组)—— 直接 np.asarray 会抛 ValueError;递归展平
    兼容 list 嵌套 / object ndarray / None 缺失值。
    """
    import numpy as np
    a = np.asarray(value)
    if a.dtype == object:
        for v in a.ravel():
            if v is None:
                continue
            if _any_nonzero(v):
                return True
        return False
    return a.size > 0 and bool(np.count_nonzero(a))


_EPISODE_FPS_CACHE: dict[str, float] = {}


def _canonical_data_files(batch_dir: Path, episode_id: str) -> list[Path]:
    """Return only the active episode parquet from the project dataset."""
    try:
        from app.project_dataset import episode_files, episode_row, is_project_dataset
        root = Path(batch_dir)
        if not is_project_dataset(root):
            return []
        row = episode_row(root, str(episode_id))
        if row is None:
            return []
        return list(episode_files(root, int(row.get("episode_index", 0))).get("data", []))
    except (OSError, TypeError, ValueError):
        return []


def _episode_real_fps(episode_id: str, batch_dir: Path) -> float:
    """实测批次视频的真实 fps(采集元数据可能不准,如 30 vs 实际 25)。

    前端播放器以 fps 做帧↔时间换算,元数据偏差会导致跳帧/切片定位
    整体偏移。探测结果按批次缓存(fps 不可变);失败返回 0(调用方回退)。
    """
    if episode_id in _EPISODE_FPS_CACHE:
        return _EPISODE_FPS_CACHE[episode_id]
    fps = 0.0
    try:
        import cv2
        videos_root = batch_dir / "videos"
        from app.lerobot_v21 import is_depth_source, iter_video_streams
        streams = [(source, path) for source, path in iter_video_streams(videos_root)
                   if not is_depth_source(source)]
        for _source, mp4 in streams:
            cap = cv2.VideoCapture(str(mp4))
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            finally:
                cap.release()
            if fps > 0:
                break
    except Exception:
        fps = 0.0
    _EPISODE_FPS_CACHE[episode_id] = fps
    return fps


@router.get("/episode/{episode_id}/frames-data")
def episode_frames_data(episode_id: str):
    """返回帧 observation 数据 + 可用传感器列表(heatmap 用)。"""
    cache_key = episode_id
    if cache_key in _frames_data_cache:
        return _frames_data_cache[cache_key]

    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    from pathlib import Path
    batch_dir = Path(ep["path"])

    frames_data = []
    available_sensors: list[str] = []
    import pandas as pd
    import numpy as np

    # 手套传感器显示 = 项目工作流含 glove_sensor 节点 **且** 列有真实
    # 压力数据(全文件均匀采样,不只看头部 —— 压力可能从任意帧开始,
    # 头部采样会误杀真实传感器;全零列 = 骨骼识别参数占位,不驱动热力图)。
    # 骨骼识别参数列(hand_pose 等)列名不含 glove/sensor,天然不进候选。
    # Current project workflow is preferred; historical batches also work
    # after their project/workflow was renamed or removed.  The resolver only
    # enables scanning here; real pressure columns are still validated below.
    glove_enabled = episode_has_glove_sensor(ep)
    if glove_enabled:
        # 第一遍:聚合所有 parquet 的传感器列名 —— 左右手套可能分散在
        # data/left_glove/ 与 data/right_glove/ 两个 parquet,不能只扫第一个。
        try:
            import pyarrow.parquet as pq
            canonical_files = _canonical_data_files(batch_dir, episode_id)
            for _data_dir in (batch_dir / "data",):
                for parq_file in canonical_files:
                    try:
                        names = pq.ParquetFile(parq_file).schema_arrow.names
                    except Exception:
                        continue
                    for col in names:
                        if not col.startswith("observation."):
                            continue
                        sname = col.split(".", 1)[1]
                        lower = sname.lower()
                        # glove/sensor/tactile 关键字(触觉/压力列);骨骼参数
                        # 列(hand_pose 等)不含这些关键字,天然排除
                        if not ("sensor" in lower or "glove" in lower or "tactile" in lower):
                            continue
                        if sname in available_sensors:
                            continue
                        # 真实压力数据(全文件均匀采样)才驱动热力图
                        if not _col_has_pressure(parq_file, col):
                            continue
                        available_sensors.append(sname)
        except Exception:
            pass

    # 第二遍:构建帧数据。项目无手套设备(glove_enabled=False)→ 前端不
    # 显示任何传感器画面,跳过帧载荷构建(省去全量读 parquet + 逐行组装;
    # 前端仅消费 fps/sensors/frame_count,不依赖 frames 载荷)。
    if not glove_enabled:
        frames_data = []
    canonical_files = _canonical_data_files(batch_dir, episode_id) if glove_enabled else []
    for _data_dir in (batch_dir / "data",):
        for parq_file in canonical_files:
            try:
                df = pd.read_parquet(parq_file)
                obs_cols = [c for c in df.columns if c.startswith("observation.")]
                act_cols = [c for c in df.columns if c.lower().startswith("action")]
                if not obs_cols and not act_cols:
                    continue
                # 按 frame_index 去重 —— 采集端 parquet 可能每帧多行(重采样),
                # 帧数必须与视频帧数一致(139 帧),否则热图/进度会越界。
                frames_map: dict[int, dict] = {}
                for _, row in df.iterrows():
                    frame_idx = int(row.get("frame_index", len(frames_map)))
                    if frame_idx in frames_map:
                        continue
                    frame: dict = {"frame_index": frame_idx}
                    for col in obs_cols:
                        val = row[col]
                        if isinstance(val, (list, np.ndarray)):
                            # Some legacy S80C rows contain a nested object
                            # array for IMU samples. One malformed optional
                            # field must not discard the complete frame or
                            # hide valid glove columns from the review page.
                            try:
                                arr = np.array(val, dtype=np.float32).tolist()
                            except (TypeError, ValueError):
                                continue
                            if isinstance(arr, list):
                                sname = col.split(".", 1)[1]
                                key = f"observation_{sname}"
                                if arr and isinstance(arr[0], list):
                                    frame[key] = arr
                                elif len(arr) == 256:
                                    frame[key] = [arr[i*16:(i+1)*16] for i in range(16)]
                                else:
                                    frame[key] = arr
                    if obs_cols and available_sensors:
                        first = f"observation_{available_sensors[0]}"
                        if first in frame:
                            frame["observation_state"] = frame[first]
                    for col in act_cols:
                        val = row[col]
                        if isinstance(val, (list, float, int, np.floating, np.ndarray)):
                            if isinstance(val, np.ndarray):
                                val = val.tolist()
                            frame["action"] = val if isinstance(val, list) else [float(val)]
                            break
                    frames_map[frame_idx] = frame
                if frames_map:
                    frames_data = [frames_map[i] for i in sorted(frames_map)]
                    break
            except Exception:
                continue
        if frames_data:
            break

    result = {
        "episode_id": episode_id,
        "fps": _episode_real_fps(episode_id, batch_dir) or ep.get("fps") or 30,
        "frame_count": len(frames_data) or ep.get("frame_count") or 0,
        "sensors": available_sensors,
        "frames": frames_data,
    }
    if len(_frames_data_cache) >= _MAX_FRAMES_CACHE:
        _frames_data_cache.pop(next(iter(_frames_data_cache)))
    _frames_data_cache[cache_key] = result
    return result


def _invalidate_episode_api_caches(episode_id: str) -> None:
    """Evict in-memory frames/fps caches for a permanently deleted episode."""
    key = str(episode_id)
    # frames-data 缓存命中发生在 get_episode 404 检查之前,必须显式清除,
    # 否则删除后仍可能短暂返回旧帧数据。
    _frames_data_cache.pop(key, None)
    _EPISODE_FPS_CACHE.pop(key, None)


# ── Media groups(双目兼容 / 素材抽屉)────────────────

def _member_label(camera: str, role: str) -> str:
    label_map = {
        "left": "Left", "right": "Right", "primary": "Primary",
        "left_aux": "Left Aux", "right_aux": "Right Aux", "aux": "Aux",
    }
    return label_map.get(role, camera)


def _detect_glove_sources(batch_dir, ep_id_str: str, project_name: str = "") -> list[dict]:
    """Detect real glove/tactile columns in the canonical episode parquet."""
    if batch_dir is None:
        return []
    from app.localstore import episode_has_glove_sensor
    episode = get_episode(ep_id_str)
    if episode is None:
        episode = {"id": ep_id_str, "project": project_name or ""}
    if not episode_has_glove_sensor(episode):
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []
    sources: list[dict] = []
    for parq_file in _canonical_data_files(Path(batch_dir), ep_id_str):
        try:
            names = pq.ParquetFile(parq_file).schema_arrow.names
        except Exception:
            continue
        for col in names:
            if not col.startswith("observation."):
                continue
            source_key = col.split(".", 1)[1]
            lower = source_key.lower()
            if not any(token in lower for token in ("glove", "sensor", "tactile")):
                continue
            if not _col_has_pressure(parq_file, col):
                continue
            sources.append({
                "id": f"glove:{source_key}",
                "kind": "glove",
                "source_key": source_key,
                "label": source_key,
                "heatmap_url": f"/api/v1/video/{ep_id_str}/heatmap/{{frame}}?sensor={source_key}",
            })
    return sources


def _detect_depth_sources(batch_dir, ep_id_str: str, master_frame_count: int = 0,
                          master_fps: float = 0) -> list[dict]:
    """Detect canonical metric-depth video streams for the review page.

    Active project datasets use one HEVC ``gray12le`` depth video per source.
    PNG sequences remain migration/diagnostic material only and are not
    exposed as a review preview or as a fake depth video.
    """
    if batch_dir is None:
        return []
    from pathlib import Path
    batch_dir = Path(batch_dir)
    sources = []
    # 只从 canonical v2.1 HEVC depth videos识别深度源；PNG 和旧处理目录不参与
    # 活动项目的设备/输入匹配。
    try:
        from app.lerobot_v21 import is_depth_source, iter_video_streams
        import json
        info = {}
        info_path = batch_dir / "meta" / "info.json"
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                info = {}
        features = info.get("features") if isinstance(info, dict) else {}
        episode_index = None
        try:
            from app.project_dataset import episode_row
            row = episode_row(batch_dir, str(ep_id_str or ""))
            episode_index = int(row["episode_index"]) if row else None
        except (TypeError, ValueError, OSError, KeyError):
            episode_index = None
        for source_name, video_path in iter_video_streams(batch_dir / "videos"):
            if (not is_depth_source(source_name)
                    or (episode_index is not None
                        and video_path.stem != f"episode_{episode_index:06d}")):
                continue
            feature = features.get(f"observation.images.{source_name}", {})
            video_info = feature.get("video_info", {}) if isinstance(feature, dict) else {}
            metric_depth = bool(video_info.get("video.is_depth_map"))
            # Metadata is required for metric depth; a pure depth-looking
            # filename alone must not turn a colour video into 3D input.
            if not metric_depth:
                continue
            frames = 0
            fps = float(master_fps or info.get("fps") or 0)
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 0)
                cap.release()
            except Exception:
                pass
            try:
                depth_stat = video_path.stat()
                depth_cache_key = f"{depth_stat.st_size:x}-{depth_stat.st_mtime_ns:x}"
            except OSError:
                depth_cache_key = None
            sources.append({
                "id": f"depth:{source_name}",
                "kind": "depth",
                "source_key": source_name,
                "label": source_name,
                "frame_count": int(master_frame_count or frames),
                "available_frame_count": frames,
                "missing_frames": [],
                "fps": fps or None,
                "dtype": "video",
                "unit": "meter",
                "metric_depth": True,
                # Stable browser-cache revision. A workflow replacement of
                # the MP4 changes this key, so cached raw windows are never
                # reused for newly processed depth data.
                "depth_cache_key": depth_cache_key,
                "depth_url": None,
                "raw_depth_url": f"/api/v1/video/{ep_id_str}/depth-stream/{source_name}",
                "depth_codes_url": (
                    f"/api/v1/video/{ep_id_str}/depth-codes/{source_name}/{{frame}}"),
                "depth_codes_window_url": (
                    f"/api/v1/video/{ep_id_str}/depth-codes-window/"
                    f"{source_name}?start_frame={{start}}&end_frame={{end}}"),
                "depth_codes_full_url": (
                    f"/api/v1/video/{ep_id_str}/depth-codes-full/{source_name}"),
                # Deprecated alias retained for clients that used the old
                # frame URL; it now returns raw uint16 codes, never JET.
                "depth_preview_url": (
                    f"/api/v1/video/{ep_id_str}/depth-preview/{source_name}/{{frame}}"),
                # Browser-compatible grayscale clock only.  The frontend
                # never uses its pixels for depth colorization.
                "depth_video_url": (
                    f"/api/v1/video/{ep_id_str}/depth-preview-stream/{source_name}"),
            })
    except Exception as exc:
        print(f"[Ingestion] v2.1 depth video discovery skipped: {exc}")

    return sources


@router.get("/episode/{episode_id}/media-groups")
def episode_media_groups(episode_id: str):
    """组织好的媒体组结构(双目组/单目/辅助 + 可拖拽素材清单)。"""
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    # 组装成本在远程挂载上(递归扫 glove/depth 目录 + 读 parquet 采样),
    # 批次结构在两次 run/上传之间不变 —— 缓存响应,上传/reprocess/run
    # 完成时显式失效(media_cache.invalidate_episode)。
    from app.media_cache import get_media_groups
    cached = get_media_groups(episode_id)
    # The response shape is part of the browser transport contract.  Do not
    # serve an older in-memory payload after a frontend transport upgrade;
    # otherwise player.js silently falls back to one HTTP request per depth
    # frame and the window decoder is never used.
    cached_depth_sources = [
        source for source in (cached or {}).get("sources", [])
        if isinstance(source, dict) and source.get("kind") == "depth"
    ]
    cache_has_depth_window = (
        not cached_depth_sources
        or all(source.get("depth_codes_window_url") for source in cached_depth_sources)
    )
    cache_has_depth_full = (
        not cached_depth_sources
        or all(source.get("depth_codes_full_url") for source in cached_depth_sources)
    )
    cache_has_depth_revision = (
        not cached_depth_sources
        or all(source.get("depth_cache_key") for source in cached_depth_sources)
    )
    if (cached is not None and cache_has_depth_window and cache_has_depth_full
            and cache_has_depth_revision):
        return cached

    from pathlib import Path
    batch_dir = Path(ep["path"])
    streams = ep.get("camera_streams") or {}

    from app.media_groups import group_camera_streams, stereo_device_label
    grouped = group_camera_streams(streams)

    def _stream_entry(item: dict) -> dict:
        """单个原始视频条目;手部骨骼由前端 SVG 覆盖层显示。"""
        cam = item["source_key"]
        stream = streams.get(cam) or {}
        entry = {
            "source_key": cam,
            "role": item.get("role", "primary"),
            "label": _member_label(cam, item.get("role")),
            "kind": "video",
            "path": stream.get("path"),
            "frame_count": stream.get("frame_count") or ep.get("frame_count") or 0,
            "fps": stream.get("fps") or ep.get("fps") or 0,
            # The review page must not depend on browser HEVC support. The
            # raw endpoint remains available for download/diagnostics.
            "stream_url": f"/api/v1/video/{episode_id}/{cam}/preview-stream",
            "raw_stream_url": f"/api/v1/video/{episode_id}/{cam}/stream",
        }
        return entry

    def _as_source(entry: dict, group_id: str | None) -> dict:
        source = {
            "id": f"video:{entry['source_key']}",
            "kind": "video",
            "group_id": group_id,
            "source_key": entry["source_key"],
            "label": entry["label"],
            "stream_url": entry["stream_url"],
            "raw_stream_url": entry.get("raw_stream_url"),
            "frame_count": entry["frame_count"],
            "fps": entry["fps"],
        }
        return source

    result = {
        "episode_id": episode_id,
        "fps": ep.get("fps"),
        "frame_count": ep.get("frame_count") or 0,
        "sync": ep.get("sync") or {},
        "groups": [],
        "singles": [],
        "sources": [],
    }

    for grp in grouped.get("groups") or []:
        grp = dict(grp)
        if not grp.get("label") or grp["label"] == f"{grp['id']} 双目":
            grp["label"] = stereo_device_label(batch_dir, grp["id"], episode_id)
        grp["members"] = [_stream_entry(m) for m in grp.get("members") or []]
        grp["aux"] = [_stream_entry(m) for m in grp.get("aux") or []]
        result["groups"].append(grp)
        for m in grp["members"]:
            result["sources"].append(_as_source(m, grp["id"]))
        for m in grp["aux"]:
            result["sources"].append(_as_source(m, grp["id"]))

    for s in grouped.get("singles") or []:
        entry = _stream_entry(s)
        result["singles"].append(entry)
        result["sources"].append(_as_source(entry, None))

    result["sources"].extend(_detect_glove_sources(batch_dir, episode_id, ep.get("project") or ""))
    result["sources"].extend(_detect_depth_sources(
        batch_dir,
        episode_id,
        int(ep.get("frame_count") or 0),
        float(ep.get("fps") or 0),
    ))
    from app.media_cache import set_media_groups
    set_media_groups(episode_id, result)
    return result


# ── 审核 / 删除 ──────────────────────────────────────

@router.post("/episode/{episode_id}/review")
def episode_review(episode_id: str):
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    set_episode_status(episode_id, "reviewed")
    return {"message": "Reviewed"}


@router.post("/episode/{episode_id}/unreview")
def episode_unreview(episode_id: str):
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    if ep.get("status") not in ("reviewed", "approved"):
        raise HTTPException(status_code=400, detail=f"Episode status is '{ep.get('status')}', not reviewed")
    set_episode_status(episode_id, "completed")
    return {"message": "Unreviewed"}


@router.post("/episode/{episode_id}/retry")
def episode_retry(episode_id: str):
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    set_episode_status(episode_id, "to_review")
    return {"message": "Moved back to review queue"}


@router.post("/episode/{episode_id}/reprocess", status_code=202)
def episode_reprocess(episode_id: str,
                            _: dict = Depends(require_roles("admin", "engineer"))):
    """主动重新处理批次:按批次归属项目的绑定工作流重新入队。

    与上传自动入队同一套匹配逻辑(source_key 交集 + workflow_bindings 快照),
    但由用户显式触发(Review 页 Reprocess 按钮)。新 run 完成后自动只保留
    最新产物;run 失败则旧产物保留(不主动删)。
    """
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    if ep.get("status") not in ("completed", "to_review", "failed"):
        raise HTTPException(
            status_code=400,
            detail="Only reviewing or failed episodes can be reprocessed",
        )

    from app.localstore import list_projects
    from app.workflow_dispatch import dispatch_project_episode

    # 批次归属项目:data/sessions/<项目>/<批次>/ 第二级即项目名
    project_name = ep.get("project") or (
        str(ep.get("path") or "").replace("\\", "/").rstrip("/").split("/")[-2]
        if "/" in str(ep.get("path") or "").replace("\\", "/") else None)
    project = None
    if project_name:
        project = next((p for p in list_projects() if p.get("name") == project_name), None)
    if project is None:
        raise HTTPException(status_code=409,
                            detail=f"No project bound for episode '{episode_id}' (unmatched upload)")
    stats = dispatch_project_episode(
        project, ep, trigger="manual_reprocess", force_rerun=True
    )
    if not stats.get("queued") and not stats.get("already_scheduled"):
        raise HTTPException(status_code=409,
                            detail="No bound workflow matches this episode's inputs")
    # run 完成会整体替换产物:提前失效媒体视图缓存,避免下次点击
    # 拿到旧 hand_3d/深度 素材清单(重新组装一次的成本远低于误展示)。
    try:
        from app.media_cache import invalidate_episode as _invalidate_media
        _invalidate_media(episode_id)
    except Exception:
        pass
    return {"episode_id": episode_id, "enqueued": stats.get("queued", 0),
            "already_scheduled": stats.get("already_scheduled", 0),
            "status": read_episode_state(episode_id).get("status") or "processing"}


@router.post("/episode/{episode_id}/delete")
def episode_soft_delete(episode_id: str):
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    ls_delete_episode(episode_id, permanent=False)
    return {"message": "Moved to trash"}


@router.post("/episode/{episode_id}/restore")
def episode_restore(episode_id: str):
    ls_restore_episode(episode_id)
    return {"message": "Restored"}


@router.delete("/episode/{episode_id}/permanent")
def episode_permanent_delete(episode_id: str):
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    # 永久删除只作用于回收站里的批次,避免与处理中任务竞态。
    if read_episode_state(episode_id).get("status") != "deleted":
        raise HTTPException(status_code=409,
                            detail="Episode is not in trash; soft-delete first")
    # 先失效在途 AI 标注 / re-export 任务,再删文件。
    try:
        from app.ai_annotation import purge_episode_ai_annotation
        purge_episode_ai_annotation(episode_id)
    except Exception as exc:
        print(f"[Episode] Failed to purge AI annotation tasks: {exc}")
    try:
        from app.routes.export import purge_episode_re_export_tasks
        purge_episode_re_export_tasks(episode_id)
    except Exception as exc:
        print(f"[Episode] Failed to purge re-export tasks: {exc}")
    # ep 在删除前捕获,携带 artifact 路径供缓存精准失效;失败时异常向上
    # 抛(500),状态文件仍在,回收站条目保留,可重试。
    ls_delete_episode(episode_id, permanent=True,
                      project_root=Path(ep.get("dataset_root") or ep["path"]))
    # 上传历史标记(记录保留,统计"上传过的所有数据"不丢)
    try:
        from app.localstore import mark_upload_purged
        mark_upload_purged(episode_id)
    except Exception as exc:
        print(f"[Episode] Failed to mark upload history purged: {exc}")
    # 该批次的异常记录一并清理(批次已删除,异常徽标不再残留)
    try:
        for exc in list_exceptions():
            if exc.get("episode_id") == episode_id:
                delete_exception(exc.get("id"))
    except Exception as exc:
        print(f"[Episode] Failed to clear exceptions: {exc}")
    # 媒体视图缓存与 worker 输入包缓存一并清除
    try:
        from app.media_cache import invalidate_episode as _invalidate_media
        _invalidate_media(episode_id)
        from app.api.worker import clear_input_zip_cache
        clear_input_zip_cache(episode_id)
    except Exception as exc:
        print(f"[Episode] Failed to clear caches: {exc}")
    # 进程内帧数据 / 视频解析缓存
    _invalidate_episode_api_caches(episode_id)
    try:
        from app.routes.video import invalidate_episode_caches
        invalidate_episode_caches(
            episode_id,
            data_paths=ep.get("episode_data") or [],
            video_paths=ep.get("episode_videos") or [],
            episode_index=ep.get("episode_index"),
        )
    except Exception as exc:
        print(f"[Episode] Failed to evict video caches: {exc}")
    return {"message": "Permanently deleted"}


@router.post("/episodes/purge-trash")
def episode_purge_trash():
    deleted = list_deleted_episodes()
    for d in deleted:
        ep = get_episode(d["id"])  # 文件已缺的崩溃残留返回 None,仍清状态
        try:
            from app.ai_annotation import purge_episode_ai_annotation
            purge_episode_ai_annotation(d["id"])
        except Exception as exc:
            print(f"[Episode] Failed to purge AI annotation tasks: {exc}")
        try:
            from app.routes.export import purge_episode_re_export_tasks
            purge_episode_re_export_tasks(d["id"])
        except Exception as exc:
            print(f"[Episode] Failed to purge re-export tasks: {exc}")
        ls_delete_episode(
            d["id"], permanent=True,
            project_root=Path(ep["dataset_root"]) if ep else None)
        try:
            from app.localstore import mark_upload_purged
            mark_upload_purged(d["id"])
        except Exception:
            pass
        # 异常记录一并清理(批次已删除,不再残留徽标)
        try:
            for exc in list_exceptions():
                if exc.get("episode_id") == d["id"]:
                    delete_exception(exc.get("id"))
        except Exception as exc:
            print(f"[Episode] Failed to clear exceptions: {exc}")
        # 媒体视图缓存与 worker 输入包缓存一并清除
        try:
            from app.media_cache import invalidate_episode as _invalidate_media
            _invalidate_media(d["id"])
            from app.api.worker import clear_input_zip_cache
            clear_input_zip_cache(d["id"])
        except Exception as exc:
            print(f"[Episode] Failed to clear caches: {exc}")
        # 进程内帧数据 / 视频解析缓存
        _invalidate_episode_api_caches(d["id"])
        if ep is not None:
            try:
                from app.routes.video import invalidate_episode_caches
                invalidate_episode_caches(
                    d["id"],
                    data_paths=ep.get("episode_data") or [],
                    video_paths=ep.get("episode_videos") or [],
                    episode_index=ep.get("episode_index"),
                )
            except Exception as exc:
                print(f"[Episode] Failed to evict video caches: {exc}")
    return {"message": f"Purged {len(deleted)} episodes"}
