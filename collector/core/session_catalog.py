"""
录制数据目录 —— 扫描、元数据读取与主时钟帧率解析（零 Qt 依赖，纯函数）。

供回放对话框与录制历史刷新共用：
  - get_effective_fps   任务元数据 → 回放主时钟帧率
  - list_sessions       扫描录制根目录，返回全部 episode（池化布局）
  - load_session_meta   任务元数据读取（格式探测 / info / fps / 传感器列名）

v1.1.0 起读侧只认任务级池化布局（meta/info.json format=="pooled_episodes_v1"），
键为 (task_dir, episode_index)；旧格式会话目录不再列出。
"""

from __future__ import annotations

import os
import json

from config import settings
from core.helpers import (
    episode_refs, detect_session_format,
    pooled_info_path, episode_row, episode_video_files,
)


def get_effective_fps(info: dict) -> float:
    """从任务元数据中提取回放主时钟帧率（每路摄像机帧率的最大值）。

    主时钟取最高帧率保证播放平滑；低帧率路按 _cam_fps 逐路独立
    seek 按比例抽帧。兼容旧数据：无 per-camera fps 时通过摄像机
    命名检测双目模式。
    """
    cameras = info.get("cameras", {})
    cam_fps_vals = []
    for cam_name, cam_info in cameras.items():
        if isinstance(cam_info, dict):
            f = cam_info.get("fps")
            if f and f > 0:
                cam_fps_vals.append(float(f))

    if cam_fps_vals:
        # 有 per-camera fps：取最大值作为主时钟
        return max(cam_fps_vals)

    # 兼容旧数据：无 per-camera fps 时，按摄像机名称推断
    cam_names = [k.lower() for k in cameras.keys()]
    is_stereo = any("stereo" in cn for cn in cam_names)
    if is_stereo:
        return float(settings.STEREO_FPS)

    global_fps = info.get("fps", 30)
    return float(global_fps) if global_fps > 0 else 30.0


def load_session_meta(task_dir: str, episode_index: int = 0) -> dict:
    """同步读取任务元数据（极快，主线程安全）。

    返回 {"fmt", "info", "fps", "sensor_names", "episode_index"}：
      - fmt          "pooled"（池化布局）或旧格式名
      - episode_index 传入的 episode 序号（0 表示未指定，仅任务级元数据）
      - sensor_names 从 info["sensors"] 读；旧格式从 features 推断；
                     最旧格式回退 ["state"]
    """
    fmt = detect_session_format(task_dir)

    if fmt == "pooled":
        mp = pooled_info_path(task_dir)
    else:
        mp = os.path.join(task_dir, "meta", "info.json")
    info = {}
    if os.path.isfile(mp):
        with open(mp, "r", encoding="utf-8") as f:
            info = json.load(f)

    sensor_names = info.get("sensors", [])
    if not sensor_names:
        # 兼容旧格式：尝试从 features 键推断
        features = info.get("features", {})
        sensor_names = [
            k.replace("observation.", "") for k in features
            if k.startswith("observation.")
        ]
    if not sensor_names:
        sensor_names = ["state"]  # 最旧格式回退

    return {
        "fmt": fmt,
        "info": info,
        "fps": get_effective_fps(info),
        "sensor_names": sensor_names,
        "episode_index": episode_index,
    }


def list_sessions(directory: str) -> list:
    """扫描录制根目录，返回全部 episode（按名称倒序）。

    每项 {"name", "path", "tag", "episode_index", "info", "duration", "fps"}：
      - path          = 任务目录（task_dir）
      - episode_index = episode 序号（池化文件组的 N）
      - duration      = episodes 行的 duration_sec（秒）
    """
    sessions = []
    for ref in episode_refs(directory):
        task_dir = ref["task_dir"]
        episode_index = ref["episode_index"]
        meta = load_session_meta(task_dir, episode_index)
        row = episode_row(task_dir, episode_index)
        sessions.append({
            "name": ref["name"], "path": task_dir, "tag": ref["task"],
            "episode_index": episode_index,
            "info": meta["info"],
            "duration": row.get("duration_sec", 0) or 0,
            "fps": meta["fps"],
            "cams": len(episode_video_files(task_dir, episode_index)),
        })
    return sessions


def list_recordings(base_dir: str) -> list[dict]:
    """扫描录制根目录，返回全部 episode（上传对话框口径）。

    每项 {"name", "path", "tag", "episode_index"}（轻量版 list_sessions，
    不读元数据）。
    """
    return [{"name": r["name"], "path": r["task_dir"], "tag": r["task"],
             "episode_index": r["episode_index"]}
            for r in episode_refs(base_dir)]
