"""批次媒体视图缓存(media-groups / hand-3d 元信息响应)。

审核页每次点击视频,后端都要在批次目录(SSHFS 远程挂载)上做多次全目录
递归(glove parquet / 深度 PNG / hand_3d 产物)并读取 parquet 采样 —— 这是
"点击别的视频加载很慢"的主要来源。批次内容在两次上传 / 工作流 run 之间
不变,因此按批次缓存组装好的响应,并在结构变化时显式失效:

    - 上传(含同名重传)完成  → session.py 上传提交后
    - reprocess 入队         → routes/ingestion.py episode_reprocess
    - run 完成(产物已替换)   → api/worker.py complete_job

TTL 是兜底自愈:任何未接失效钩子的外部改动会在到期后自然重建。
"""

from __future__ import annotations

import copy
import threading
import time

_lock = threading.Lock()
_MEDIA_GROUPS: dict[str, tuple[float, dict]] = {}
_HAND3D_META: dict[str, tuple[float, dict]] = {}
_TTL = 600.0  # 兜底自愈;正常路径由 invalidate_episode 显式失效


def _fresh(entry: tuple[float, dict] | None, now: float) -> bool:
    return entry is not None and now - entry[0] < _TTL


def get_media_groups(episode_id: str) -> dict | None:
    with _lock:
        entry = _MEDIA_GROUPS.get(str(episode_id))
        if _fresh(entry, time.monotonic()):
            return copy.deepcopy(entry[1])
    return None


def set_media_groups(episode_id: str, payload: dict) -> None:
    with _lock:
        _MEDIA_GROUPS[str(episode_id)] = (time.monotonic(), copy.deepcopy(payload))


def get_hand3d_meta(episode_id: str) -> dict | None:
    with _lock:
        entry = _HAND3D_META.get(str(episode_id))
        if _fresh(entry, time.monotonic()):
            return copy.deepcopy(entry[1])
    return None


def set_hand3d_meta(episode_id: str, payload: dict) -> None:
    with _lock:
        _HAND3D_META[str(episode_id)] = (time.monotonic(), copy.deepcopy(payload))


def invalidate_episode(episode_id: str) -> None:
    """批次结构/产物变化(上传替换、reprocess 入队、run 完成)后调用。"""
    key = str(episode_id)
    with _lock:
        _MEDIA_GROUPS.pop(key, None)
        _HAND3D_META.pop(key, None)
