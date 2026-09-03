"""Batch directory file discovery helpers — shared by modules and worker.

从批次解压目录(input_root)按 source_keys/source_key/position 匹配视频、
按位置匹配 parquet。匹配时排除辅助流(_aux)和骨骼视频。
"""

from __future__ import annotations

from pathlib import Path

from app.lerobot_v21 import (
    canonical_source_key,
    is_depth_source,
    iter_video_streams,
)


def _keys_of(config: dict) -> list[str]:
    keys: list[str] = []
    raw_keys = config.get("source_keys")
    if isinstance(raw_keys, str):
        keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    elif isinstance(raw_keys, (list, tuple)):
        keys = [str(k) for k in raw_keys if str(k).strip()]
    single = config.get("source_key") or config.get("position")
    if single and str(single).strip() and str(single) not in keys:
        keys.insert(0, str(single))
    return keys


# 相机名 → 模块类型推断(source_key 匹配不上时的兜底,保证项目上传过的
# 数据类型在输入面板可见)。旧卡规则保留(存量工作流兼容);新体系下
# stereo_camera/stereo_rgbd_camera 先认领 stereo_*,mono_camera 兜底认领剩余相机名。
_CAMERA_TYPE_RULES: dict[str, tuple[str, ...]] = {
    "stereo_camera": ("stereo",),
    "stereo_rgbd_camera": ("stereo",),
    "rgbd_camera": ("rgbd",),
    "fisheye_camera": ("fisheye",),
    "rgb_camera": ("rgb", "ego"),
}

# 匹配优先级(显式顺序,不依赖注册序):双目先认领,单目兜底
_MATCH_PRIORITY = ["stereo_rgbd_camera", "stereo_camera", "rgbd_camera", "mono_camera"]

# 传感器类输入源关键词:mono 兜底认领时跳过(不抢手套/IMU/深度等)
_SENSOR_NAME_KEYWORDS = ("glove", "imu", "action", "tactile", "sensor",
                         "depth", "hand")


def match_input_modules(available: set[str]) -> list[dict]:
    """输入模块与可用输入源的匹配结果(projects/devices input-sources 共用)。

    ``available``: 小写化的可用输入源集合(相机主目名 + 传感器名)。
    匹配分两步:
    1. 模块 source_key(s) 与 available **双向子串**匹配 —— 精确名
       (stereo_left vs stereo_left)与包含关系(left_glove_joint vs
       left_glove)都覆盖;
    2. source_key 未命中时按类型兜底:stereo_camera 认领含 stereo 的
       相机名;mono_camera 认领剩余的非传感器相机名(head_left_rgb /
       d435_rgb / ego_rgb 等任意单目命名,与设备名无关);旧 rgb/fisheye
       卡规则保留(存量工作流)。
    同一输入源只归属一个模块(优先子串匹配;兜底按 _MATCH_PRIORITY)。
    返回 [{type, label, matched, matched_keys}]。
    """
    from app.processing.catalog import module_catalog
    by_type = {
        d.get("type") or d.get("slug"): d
        for d in module_catalog()
        if d.get("category") == "input"
    }
    order = [t for t in _MATCH_PRIORITY if t in by_type]
    order += [t for t in by_type if t not in order]   # 其余按注册序

    modules = []
    used: set[str] = set()   # 已归属的输入源,避免重复匹配
    for node_type in order:
        d = by_type[node_type]
        keys = _keys_of(dict(d.get("default_config") or {}))
        hits: list[str] = []
        # 1) source_key 双向子串匹配
        for k in keys:
            kl = k.lower()
            for a in sorted(available):
                if a in used:
                    continue
                if kl in a or a in kl:
                    hits.append(a)
                    used.add(a)
                    break
        # 2) 类型兜底
        if not hits:
            if node_type == "mono_camera":
                # 单目 = 兜底认领剩余相机名(跳过传感器/深度类名称)
                for a in sorted(available):
                    if a in used:
                        continue
                    # A color slot published by a depth camera is still a
                    # normal video input (e.g. D435_depth_rgb).  Only pure
                    # depth streams should be excluded from Mono Video.
                    if (any(kw in a for kw in _SENSOR_NAME_KEYWORDS)
                            and not ("rgb" in a and "depth" in a)):
                        continue
                    hits.append(a)
                    used.add(a)
                    break
            elif node_type in {"rgbd_camera", "stereo_rgbd_camera"}:
                # RGB-D is identified from a color stream carrying a depth
                # marker. Pure depth streams are not normal video inputs.
                for a in sorted(available):
                    if a in used:
                        continue
                    if node_type == "stereo_rgbd_camera" and "stereo" not in a:
                        continue
                    if ("depth" in a or "rgbd" in a) and any(
                            token in a for token in ("rgb", "color")):
                        hits.append(a)
                        used.add(a)
                        break
            else:
                rules = _CAMERA_TYPE_RULES.get(node_type)
                if rules:
                    for a in sorted(available):
                        if a in used:
                            continue
                        # rgb 兜底不抢 stereo/fisheye 的相机名
                        if node_type == "rgb_camera" and any(
                                kw in a for kw in ("stereo", "fisheye")):
                            continue
                        if any(kw in a for kw in rules):
                            hits.append(a)
                            used.add(a)
                            break
        modules.append({
            "type": node_type,
            "label": d.get("label"),
            "matched": bool(hits),
            "matched_keys": hits,
        })
    return modules


def _is_main_video(path: Path, source_key: str | None = None) -> bool:
    """主 RGB 流判定:排除深度流、辅助流(_aux)和预览渲染流。"""
    p = path.as_posix().lower()
    return (not is_depth_source(source_key)
            and "/depth/" not in p and "/processed/" not in p
            and "_aux" not in p and "skeleton" not in p)


def find_videos(input_root: Path, config: dict) -> list[Path]:
    """按 source_keys(或 source_key/position)匹配视频列表,保持配置顺序。

    双目兼容:stereo_camera 可配置 ``source_keys: "stereo_left,stereo_right"``,
    每个 key 匹配一个视频;匹配时排除辅助流(_aux)、骨骼视频、深度预览
    与 processed 处理产物。

    自动识别:未配置 key → 返回 videos/ 目录下全部主 RGB 视频(按路径
    字典序稳定排序,left 在 right 前)。配置了显式 key 但未命中时返回空
    列表，让对应输入节点 skipped，避免把另一台相机误认成该设备。
    """
    expected_name = Path(str(config.get("expected_name") or "")).name
    videos_root = input_root / "videos"
    stream_entries = iter_video_streams(videos_root) if videos_root.is_dir() else []
    videos = [path for source, path in stream_entries
              if _is_main_video(path, source)]
    # Compatibility fallback for malformed/legacy uploads where videos are
    # not under a videos/ root.  The canonical v2.1 path always uses the
    # source-aware list above.
    if not videos:
        videos = sorted(input_root.rglob("*.mp4"))
    if expected_name:
        for path in videos:
            if path.name == expected_name and _is_main_video(path):
                return [path]

    keys = _keys_of(config)
    found: list[Path] = []
    used: set[Path] = set()
    for key in keys:
        key_l = key.lower()
        canonical_key_l = canonical_source_key(key).lower()
        # Match the source identity before falling back to path text.  This
        # keeps old workflows using D435_head_rgb/D435_depth_rgb compatible
        # with the canonical D435_rgb directory after v2.1 migration.
        candidates = [p for source, p in stream_entries
                      if p in videos and p not in used
                      and canonical_source_key(source).lower() == canonical_key_l]
        if not candidates:
            candidates = [p for source, p in stream_entries
                          if p in videos and p not in used
                          and (key_l in source.lower()
                               or source.lower() in key_l)]
        if not candidates:
            candidates = [p for p in videos
                          if key_l in p.as_posix().lower() and p not in used
                          and _is_main_video(p)]
        if not candidates:
            continue
        # 优先主目(左目在右目之前,字典序稳定)
        candidates.sort(key=lambda p: p.as_posix())
        found.append(candidates[0])
        used.add(candidates[0])
    if found:
        return found
    # An explicit source key is tied to one device. Do not auto-detect and
    # reuse another camera when that device is absent; the input module will
    # mark itself skipped and downstream nodes can use available inputs.
    if keys:
        return []
    # 自动识别回退:只扫 canonical videos/ 主目录树
    pool = videos
    mains = sorted([p for p in pool if _is_main_video(p)],
                   key=lambda p: p.as_posix())
    if mains:
        return mains
    # videos/ 无结果时兜底全树扫描(兼容不规范的旧批次布局)
    return sorted([p for p in videos if _is_main_video(p)],
                  key=lambda p: p.as_posix())


def find_video(input_root: Path, config: dict) -> Path:
    """按 source_key 匹配单个视频(兼容旧节点)。"""
    videos = find_videos(input_root, config)
    if not videos:
        raise FileNotFoundError("No MP4 video found in uploaded episode")
    return videos[0]


def find_parquet(input_root: Path) -> list[Path]:
    """批次内全部 parquet(排除 auto_labels/hand_kpts 合并产物与 meta 目录)。"""
    out = []
    for p in sorted(input_root.rglob("*.parquet")):
        lower = p.as_posix().lower()
        if "auto_labels" in lower or "hand_kpts" in lower:
            continue
        if "/meta/" in lower:
            continue
        out.append(p)
    return out
