"""Camera/video grouping — 恢复双目(左右目)语义。

上传的 zip 里一台双目相机产生多个独立 mp4(videos/stereo_left/…、
videos/stereo_right/…),扁平 camera 列表会丢失"组"的语义,前端只能
把它们当成互不相关的视频。本模块按命名模式把扁平 camera 聚合为:

- 双目组(stereo):left + right 配对,``*_aux`` 为辅助流
- 单目组(single):无配对的独立 camera(如 head_left_rgb)

新旧数据兼容:新上传数据在 ``meta.camera_groups`` 里携带分组结果,
旧数据没有该字段,由 media-groups API 用本模块实时推导。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 后缀从长到短,避免 stereo_left_aux 被误配为 ("stereo_left", "aux")
_ROLE_SUFFIXES = (
    ("_left_aux", "left_aux"),
    ("_right_aux", "right_aux"),
    ("_aux_left", "left_aux"),
    ("_aux_right", "right_aux"),
    ("_left", "left"),
    ("_right", "right"),
    ("_aux", "aux"),
)


def _role_of(name: str) -> tuple[str, str]:
    """提取 (前缀, 角色): stereo_left_aux → ("stereo", "left_aux")"""
    normalized = str(name).strip().lower()
    for suffix, role in _ROLE_SUFFIXES:
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)].rstrip("_"), role
    return normalized, "primary"


def _member(name: str, role: str, stream: dict) -> dict:
    # 注意:role 必须最后赋值 —— stream 里可能带上游 _camera_role 的粗糙
    # role(如 stereo_left_aux → "left"),解包会错误覆盖推导出的 left_aux。
    return {"source_key": name, **(stream or {}), "role": role}


def group_camera_streams(camera_streams: dict[str, Any]) -> dict[str, Any]:
    """把扁平 ``{camera: stream}`` 聚合为 groups + singles。

    输入例如::

        {"stereo_left": {"path": "...", "frame_count": 258, "fps": 25},
         "stereo_right": {...}, "stereo_left_aux": {...}, "stereo_right_aux": {...}}

    返回::

        {"groups": [{"id": "stereo", "type": "stereo", "label": "stereo 双目",
                     "members": [left, right], "aux": [left_aux, right_aux]}],
         "singles": [{"source_key": "head_left_rgb", "role": "primary", ...}]}
    """
    parsed: dict[str, list[tuple[str, str, dict]]] = {}
    for name, stream in (camera_streams or {}).items():
        prefix, role = _role_of(name)
        parsed.setdefault(prefix, []).append((name, role, stream or {}))

    groups: list[dict] = []
    singles: list[dict] = []
    for prefix, members in parsed.items():
        roles = {role for _, role, _ in members}
        has_left = "left" in roles or "left_aux" in roles
        has_right = "right" in roles or "right_aux" in roles
        if has_left and has_right:
            group: dict[str, Any] = {
                "id": prefix,
                "type": "stereo",
                "label": prefix,
                "members": [],
                "aux": [],
            }
            for name, role, stream in members:
                entry = _member(name, role, stream)
                if role in ("left", "right"):
                    group["members"].append(entry)
                else:
                    group["aux"].append(entry)
            groups.append(group)
        else:
            for name, role, stream in members:
                singles.append(_member(name, role, stream))

    return {"groups": groups, "singles": singles}


def stereo_device_label(session_dir: Path | None, fallback: str,
                        episode_id: str | None = None) -> str:
    """从 ``calibration/*.json`` 找双目设备名作为组标签。

    ``head_stereo.json`` (type=stereo_rgbd_camera) → "head_stereo 双目",
    找不到则回退为前缀命名。
    """
    if session_dir is not None:
        calib_dir = Path(session_dir) / "calibration"
        if episode_id:
            namespaced = Path(session_dir) / "meta" / "calibration" / str(episode_id)
            if namespaced.is_dir():
                calib_dir = namespaced
        if not calib_dir.is_dir():
            calib_dir = Path(session_dir) / "meta" / "calibration"
        if calib_dir.is_dir():
            for f in sorted(calib_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(data.get("type", "")).startswith("stereo"):
                    name = data.get("name") or f.stem
                    if name:
                        return name
    return fallback


def has_skeleton_video(session_dir: Path | None, camera: str,
                       episode_id: str | None = None) -> bool:
    """探测批次内是否已有骨骼叠加视频(worker 输出)。

    手部叠加由前端将 parquet 关键点绘制到原始 RGB 视频上，因此这里不再
    探测或保留第二份骨骼视频。
    """
    if session_dir is None:
        return False
    return False


def video_dir_has_skeleton(video_root: Path) -> bool:
    """``videos/<camera>/`` 目录下是否有骨骼叠加视频。"""
    if not video_root.is_dir():
        return False
    return any("_skeleton" in p.name for p in video_root.glob("*_skeleton*.mp4"))
