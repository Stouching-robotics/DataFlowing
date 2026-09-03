"""HDF5 dataset export — 与 LeRobot 导出同源数据,写成单一 .h5 文件。

组织(与 LeRobot observation 语义对齐,一集一个 group):

    dataset.h5
    ├─ /meta/info                      attrs: format/version/fps/features
    ├─ /episode_000000
    │  ├─ /observation
    │  │  ├─ /images/<source_key>      (T, H, W, 3) uint8   视频抽帧(RGB)
    │  │  ├─ /state/hand_left_world    (T, 21, 3) float32   米制 3D 骨骼
    │  │  ├─ /state/hand_right_world   (T, 21, 3) float32
    │  │  ├─ /tactile/left             (T, 16, 16) float32  SenseGlove 压力阵列
    │  │  ├─ /tactile/right            (T, 16, 16) float32
    │  │  ├─ /hand_pose/left           (T, 63) float32      采集端姿态参数
    │  │  ├─ /hand_pose/right          (T, 63) float32
    │  │  ├─ /imu                      (T, 6) float32
    │  │  ├─ /hand_0_gesture ...       (T,) 变长字符串      手指伸展标签(角度判定)
    │  │  └─ /annotation ...           (T,) 变长字符串列
    │  ├─ /action                      (T, D) float32
    │  ├─ /videos/hand_skeleton        (nbytes,) uint8    手部骨骼渲染 MP4 原始字节
    │  └─ /meta                        attrs: episode_id/task/split/annotations

数值列统一 float32(缺帧 NaN 填充),字符串列变长 utf8;
视频逐帧流式写入(内存安全),gzip 压缩 + 每帧一个 chunk。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.localstore import get_episode
from app.lerobot_export import (
    _annotation_maps,
    _calibration_docs,
    _hand_columns,
    _infer_hand_3d_unit,
    _key_matches_any,
    _prepare_hand_export_inputs,
    _probe_video,
    _read_hand_rows,
    _read_hand_3d_rows_by_source,
    _device_hand_features,
    _device_metadata,
    _read_sensor_rows,
    _source_path,
)

# 采集端姿态参数(63)与压力阵列(256)的语义路径
_TACTILE_KEYS = ("observation.tactile.left", "observation.tactile.right")
_POSE_KEYS = ("observation.left_hand_pose", "observation.right_hand_pose")


def _read_timestamps(session_dir: Path):
    """读取采集端 timestamps.json(帧级时间戳,每帧 2 条 = 左右目各 1 条)。

    返回 (M, 4) float64 数组 [frame_index, timestamp(秒), wall_time(Unix 秒),
    hardware_ns], 保持原始顺序; 文件缺失/解析失败 → None(不影响导出)。
    """
    import numpy as np
    path = Path(session_dir) / "timestamps.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("timestamps") or []
        if not rows:
            return None
        return np.asarray(
            [[float(r.get("frame_index", 0)), float(r.get("timestamp", 0)),
              float(r.get("wall_time", 0)), float(r.get("hardware_ns", 0))]
             for r in rows], dtype=np.float64)
    except Exception:
        return None


def _find_hand_render_video(hand_3d_paths: list[str], session_dir: Path,
                            hand_keypoints_paths: list[str] | None = None) -> Path | None:
    """No rendered hand video is exported; previews are browser overlays."""
    return None


def _collect(columns: dict[str, dict[int, Any]], str_columns: set[str],
             key: str, value: Any, frame_index: int) -> None:
    """按 key + 帧号收集(字典天然对齐,缺帧在写入时 NaN/空串填充)。"""
    if isinstance(value, (str, bytes)):
        str_columns.add(key)
    columns.setdefault(key, {})[frame_index] = value


def _write_columns(obs_group: Any, columns: dict[str, dict[int, Any]],
                   str_columns: set[str], T: int) -> dict[str, dict]:
    """把收集好的列写成 h5py dataset,返回 {feature key: 描述}。"""
    import h5py
    import numpy as np
    features: dict[str, dict] = {}

    for key, by_frame in columns.items():
        if not by_frame:
            continue
        path = key[len("observation."):].replace(".", "/") \
            if key.startswith("observation.") else key

        # 字符串列:变长 utf8
        if key in str_columns:
            ds = obs_group.create_dataset(
                path, shape=(T,), dtype=h5py.string_dtype(encoding="utf-8"),
                compression="gzip")
            for i in range(T):
                ds[i] = str(by_frame.get(i, ""))
            features[key if key.startswith("observation.") else f"observation.{key}"] = {
                "dtype": "string", "shape": [1]}
            continue

        # 数值列:首值定 shape,缺帧 NaN
        first = next(value for value in by_frame.values() if value is not None)
        base_shape = np.asarray(first).shape
        is_tactile = key in _TACTILE_KEYS
        if is_tactile:
            base_shape = (16, 16)
        # 手部骨骼列与 LeRobot 一致走 63 位扁平存储,这里恢复 (21,3)
        is_hand_coordinate = any(
            key == f"observation.state.hand_{side}_{coord}{suffix}"
            for side in ("left", "right")
            for coord in ("world", "3d", "2d")
            for suffix in ("", "_rcam")
        ) or ("observation.state.devices." in key and any(
            f"hand_{side}_{coord}" in key
            for side in ("left", "right")
            for coord in ("world", "3d", "2d")
        ))
        if is_hand_coordinate:
            base_shape = (21, 3)
        # 跨帧形状不一致 = 变长聚合(如 IMU 每帧样本数不定)→ 平铺 + 帧索引,
        # 不丢数据;固定形状走常规 (T, ...) NaN 填充
        shapes = {np.asarray(v).shape for v in by_frame.values() if v is not None}
        if len(shapes) > 1:
            # dtype 保持源语义:整型(如 imu_ts_ns 的 ns 时间戳)用 int64,
            # 否则 float32 —— float32 只有 7 位有效数字,会丢 ns 精度
            kinds = {np.asarray(v).dtype.kind for v in by_frame.values() if v is not None}
            out_dtype = np.int64 if kinds <= {"i", "u"} else np.float32
            arrays = [np.asarray(v, dtype=out_dtype)
                      for v in by_frame.values() if v is not None]
            # 1D 变长列(如 imu_ts_ns)→ 列向量 (N,1) 再按帧拼接;
            # 2D+ 列(如 imu (N,6))保持原样(axis=0 按帧堆叠)
            arrays = [a if a.ndim >= 2 else a.reshape(-1, 1) for a in arrays]
            dim = arrays[0].shape[-1]
            flat = np.concatenate(arrays, axis=0)
            frame_ids = np.concatenate([
                np.full(len(a), fi, dtype=np.int32)
                for fi, a in zip(by_frame.keys(), arrays)])
            ds = obs_group.create_dataset(path, data=flat, compression="gzip")
            ds.attrs["note"] = (
                "变长聚合:每帧样本数不定(如 IMU 按帧聚合,200Hz/25fps≈8样本/帧);"
                f"按 {path}_frame_index 切片还原每帧样本")
            obs_group.create_dataset(path + "_frame_index", data=frame_ids,
                                     compression="gzip")
            features[key if key.startswith("observation.") else f"observation.{key}"] = {
                "dtype": str(np.dtype(out_dtype)), "shape": [dim],
                "note": "变长聚合(每帧样本数不定),配 {path}_frame_index 列切片还原",
            }
            continue
        stacked = np.full((T,) + base_shape, np.nan, dtype=np.float64)
        for i, value in by_frame.items():
            arr = np.asarray(value)
            if is_tactile:
                arr = arr.reshape(16, 16)
            elif is_hand_coordinate:
                arr = arr.reshape(21, 3)
            stacked[i] = arr
        ds = obs_group.create_dataset(path, data=stacked.astype(np.float32),
                                      compression="gzip")
        features[key if key.startswith("observation.") else f"observation.{key}"] = {
            "dtype": "float32",
            "shape": list(stacked.shape[1:]) or [1],
        }
    return features


def build_hdf5_dataset(dataset_name: str, episode_ids: list[str], output_dir: Path,
                       split_ratio: float = 0.9,
                       include_video_keys: list[str] | None = None,
                       hand_keypoints_paths: list[str] | None = None,
                       hand_3d_paths: list[str] | None = None,
                       hand_3d_unit: str | None = None,
                       compression: str = "gzip", level: int = 4) -> Path:
    """把批次构建为单一 .h5 数据集(与 LeRobot 导出同源数据)。

    ``include_video_keys``: 只抽帧这些 source_key 对应的视频(工作流连接
    驱动);None = 全量;空列表 = 不抽帧。

    ``hand_keypoints_paths`` / ``hand_3d_paths``: 手部骨骼/深度 3D 产物。
    RGB_TO_3D 的 hand_3d 仅用于前端预览,导出时自动读取其中的 2D 关键点;
    只有真实深度产物才写入 HDF5 的 world 3D 字段。
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 export") from exc
    import cv2
    import numpy as np

    episodes = [get_episode(str(episode_id)) for episode_id in episode_ids]
    episodes = [episode for episode in episodes if episode]
    if not episodes:
        raise RuntimeError("No valid episodes selected for export")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}.h5"

    (hand_keypoints_paths, hand_3d_paths, _hand_3d_right_paths,
     hand_3d_unit) = _prepare_hand_export_inputs(
         hand_keypoints_paths, hand_3d_paths, None, hand_3d_unit)
    hand_rows = _read_hand_rows(hand_keypoints_paths, hand_3d_paths)
    device_hand_rows = _read_hand_3d_rows_by_source(hand_3d_paths)
    hand_3d_unit = hand_3d_unit or _infer_hand_3d_unit(hand_3d_paths)
    coordinate_key = (
        "world" if hand_3d_unit in {"camera_meters", "meter", "meters"} else
        "3d" if hand_3d_paths else "2d"
    )
    is_3d = any(
        entry and entry.get("is_3d")
        for hand in hand_rows.values()
        for entry in hand.values()
    ) if hand_rows else False

    total_frames = 0
    features: dict[str, dict] = {}
    fps_first = float(episodes[0].get("fps") or 30)

    with h5py.File(str(out_path), "w") as f:
        meta = f.create_group("meta/info")

        for out_episode_index, episode in enumerate(episodes):
            episode_id = str(episode["id"])
            session_dir = Path(episode["path"])
            streams = episode.get("camera_streams") or {}
            sensor_rows = _read_sensor_rows(session_dir)
            ts_arr = _read_timestamps(session_dir)
            annotation_map, annotation_defs, _skipped_candidates = _annotation_maps(episode_id)
            episode_fps = float(episode.get("fps") or 30)

            # 连接驱动:只抽帧连到导出节点的视频源
            video_meta: list[tuple[str, Path, int, float, int, int]] = []
            for source_key, stream in sorted(streams.items()):
                if include_video_keys is not None and not _key_matches_any(source_key, include_video_keys):
                    continue
                source = _source_path(stream.get("path"))
                if source is None or not source.exists():
                    continue
                count, vfps, width, height = _probe_video(source)
                video_meta.append((source_key, source, count or 0, vfps or episode_fps,
                                   width, height))

            sensor_max = max(sensor_rows.keys(), default=-1) + 1
            hand_max = max(hand_rows.keys(), default=-1) + 1
            T = max([meta[2] for meta in video_meta] + [sensor_max, hand_max, 0])
            total_frames += T

            ep = f.create_group(f"episode_{out_episode_index:06d}")
            obs = ep.create_group("observation")

            # 1) 视频 → 逐帧抽帧写 uint8 数组(gzip + 每帧 chunk,流式省内存)
            for source_key, source, count, vfps, width, height in video_meta:
                if width <= 0 or height <= 0 or T <= 0:
                    continue
                ds = obs.create_dataset(
                    f"images/{source_key}", shape=(T, height, width, 3),
                    dtype="uint8", chunks=(1, height, width, 3),
                    compression=compression, compression_opts=level)
                ds.attrs["fps"] = vfps
                ds.attrs["source_key"] = source_key
                ds.attrs["source_episode"] = episode_id
                cap = cv2.VideoCapture(str(source))
                try:
                    i = 0
                    while i < T:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        ds[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        i += 1
                finally:
                    cap.release()
                features[f"observation.images.{source_key}"] = {
                    "dtype": "uint8", "shape": [T, height, width, 3],
                    "video_info": {"video.fps": vfps, "video.is_depth_map": False},
                }

            # 2) 表列:传感器 + 手部骨骼 + 标注(与 LeRobot 列同构,按帧号对齐)
            columns: dict[str, dict[int, Any]] = {}
            str_columns: set[str] = set()
            for frame_index in range(T):
                row: dict[str, Any] = {}
                row.update(sensor_rows.get(frame_index, {}))
                hand = hand_rows.get(frame_index)
                if hand:
                    row.update(_hand_columns(hand, coordinate_key=coordinate_key))
                for device_source, rows_by_frame in device_hand_rows.items():
                    device_hand = rows_by_frame.get(frame_index)
                    if device_hand:
                        row.update(_hand_columns(
                            device_hand, coordinate_key=coordinate_key,
                            device_namespace=device_source))
                anno, anno_index, _scope = annotation_map.get(
                    frame_index, (None, -1, ["episode"]))
                row["annotation"] = anno or ""
                row["annotation_index"] = anno_index
                for key, value in row.items():
                    _collect(columns, str_columns, key, value, frame_index)

            # 3) 动作列(独立 dataset,非 observation 子项)
            if "action" in columns and "action" not in str_columns:
                import numpy as np
                by_frame = columns["action"]
                first = next(value for value in by_frame.values()
                             if value is not None)
                act = np.full((T,) + np.asarray(first).shape, np.nan)
                for i, value in by_frame.items():
                    act[i] = np.asarray(value)
                if act.ndim == 1:
                    act = act.reshape(-1, 1)
                ep.create_dataset("action", data=act.astype(np.float32),
                                  compression="gzip")
                features["action"] = {"dtype": "float32",
                                      "shape": list(act.shape[1:]) or [1]}
                del columns["action"]

            col_features = _write_columns(obs, columns, str_columns, T)
            features.update(col_features)
            if device_hand_rows:
                device_unit = hand_3d_unit or (
                    "mediapipe_world_relative" if coordinate_key == "3d"
                    else "normalized_image_coords")
                device_note = (
                    "深度图抬升相机系米制 3D(每个物理设备独立坐标系,米)"
                    if coordinate_key == "world" else
                    "MediaPipe world_landmarks 相对 3D(设备独立)"
                    if coordinate_key == "3d" else
                    "MediaPipe 21 关键点归一化图像坐标(设备独立)")
                for device_source in device_hand_rows:
                    features.update(_device_hand_features(
                        device_source, coordinate_key, device_unit, device_note))

            # 2b) 采集端帧级时间戳(timestamps.json 原样保留):
            # 每帧 2 条 = 左右目各 1 条; timestamp(秒) / wall_time(Unix 秒) /
            # hardware_ns 三路时钟,客户可做帧级时间对齐分析。
            if ts_arr is not None and len(ts_arr):
                tds = obs.create_dataset("timestamps", data=ts_arr,
                                         compression="gzip")
                tds.attrs["note"] = (
                    "采集端帧时间戳(每帧2条=左右目各1条),列:"
                    "[frame_index, timestamp(秒), wall_time(Unix秒), hardware_ns];"
                    "timestamp/wall_time 交叉验证文件帧率与真实录制时长一致")
                features["observation.timestamps"] = {
                    "dtype": "float64", "shape": list(ts_arr.shape),
                    "note": "采集端帧时间戳,每帧2条(左右目),"
                            "列=[frame_index, timestamp, wall_time, hardware_ns]",
                }

            # 4) 手部骨骼渲染视频(MP4 原始字节)→ /videos/hand_skeleton。
            # 与 hand_3d parquet 同源交付;attrs 记录格式信息,客户可提取为
            # 独立 MP4 直接用播放器打开(无需 h5py 逐帧还原)。只存原始字节,
            # 体积小(约 10 MB),不重复抽帧。
            hand_video = _find_hand_render_video(
                hand_3d_paths or [], session_dir, hand_keypoints_paths)
            if hand_video is not None:
                vcount, vfps, vwidth, vheight = _probe_video(hand_video)
                mp4_bytes = np.frombuffer(hand_video.read_bytes(), dtype="uint8")
                vds = ep.create_group("videos").create_dataset(
                    "hand_skeleton", data=mp4_bytes)
                vds.attrs["format"] = "mp4"
                vds.attrs["fps"] = vfps
                vds.attrs["width"] = vwidth
                vds.attrs["height"] = vheight
                vds.attrs["frames"] = vcount
                vds.attrs["source"] = hand_video.name
                # 单目(mediapipe_hand 的 skeleton/ 产物)与双目(三角化
                # hand_3d/ 产物)渲染视频语义不同, description 区分开。
                mono = hand_video.parent.name == "skeleton"
                vds.attrs["description"] = (
                    "MediaPipe hand-skeleton render video "
                    "(single camera, skeleton overlay)" if mono
                    else "Stereo-triangulated hand-skeleton render video "
                         "(left/right side by side, skeleton overlay)")
                features["videos.hand_skeleton"] = {
                    "dtype": "uint8", "format": "mp4", "fps": vfps,
                    "width": vwidth, "height": vheight, "frames": vcount,
                    "note": "MP4 原始字节,提取后为独立 .mp4 文件",
                }

            # 5) episode meta
            ep_meta = ep.create_group("meta")
            ep_meta.attrs["episode_id"] = episode_id
            ep_meta.attrs["task_description"] = str(episode.get("project") or "")
            ep_meta.attrs["fps"] = episode_fps
            ep_meta.attrs["length"] = T
            ep_meta.attrs["devices"] = json.dumps(
                _device_metadata(session_dir), ensure_ascii=False)
            ep_meta.attrs["split"] = (
                "train" if out_episode_index < max(1, int(len(episodes) * split_ratio))
                else "val")
            if annotation_defs:
                ep_meta.attrs["annotations"] = json.dumps(
                    annotation_defs, ensure_ascii=False)

            # 6) 标定 → /meta/calibration/<episode>/<name> attrs(JSON 字符串)
            # 与 stereo_triangulate 一致:批次标定缺失/全零 → 回退内置默认标定
            # (否则 meta 里声明的是未生效的标定,3D 数据无法复现)。
            # 回退逻辑统一在 lerobot_export._calibration_docs(单点维护)。
            cal_group = meta.create_group(f"calibration/{episode_id}")
            cal_docs = _calibration_docs(session_dir)
            for name, value in cal_docs.items():
                cal_group.attrs[name] = json.dumps(value, ensure_ascii=False)

        # 手部骨骼 features 补充(unit 语义)
        if hand_rows:
            for side in ("left", "right"):
                key = f"observation.state.hand_{side}_{coordinate_key}"
                if key in features:
                    features[key].update({
                        "unit": hand_3d_unit or (
                            "mediapipe_world_relative" if coordinate_key == "3d"
                            else "normalized_image_coords"),
                        "note": (
                            "深度图抬升相机系 3D 世界坐标(米)"
                            if coordinate_key == "world" else
                            "MediaPipe 21 关键点相对 3D"
                            if coordinate_key == "3d" else
                            "MediaPipe 21 关键点归一化图像坐标"),
                    })

        meta.attrs["format"] = "egodata-hdf5"
        meta.attrs["version"] = "1.0"
        meta.attrs["total_episodes"] = len(episodes)
        meta.attrs["total_frames"] = total_frames
        meta.attrs["fps"] = fps_first
        meta.attrs["features"] = json.dumps(features, ensure_ascii=False)
        meta.attrs["calibration_root"] = "calibration/"

        # 内嵌说明文档(结构树/内容表/读取示例)—— 随 h5 交付,
        # 客户无需额外文档即可理解数据。在 f 关闭前生成(需要实际形状)。
        f.create_dataset("meta/README", data=_build_readme(dataset_name, f))

    return out_path


def _tree_lines(entries: list[tuple[list[str], str]], prefix: str = "") -> list[str]:
    """把 (路径段, 描述文本) 渲染为 ASCII 树行(递归,按段分组)。"""
    groups: dict[str, list[tuple[list[str], str]]] = {}
    for parts, note in entries:
        groups.setdefault(parts[0], []).append((parts[1:], note))

    lines: list[str] = []
    keys = sorted(groups)
    for i, key in enumerate(keys):
        sub = groups[key]
        is_last = i == len(keys) - 1
        branch = "└─ " if is_last else "├─ "
        if sub and sub[0][0]:  # 有子项 → 目录
            lines.append(f"{prefix}{branch}{key}/")
            lines.extend(_tree_lines(sub, prefix + ("   " if is_last else "│  ")))
        else:
            lines.append(f"{prefix}{branch}{key:<34} {sub[0][1]}")
    return lines


def _build_readme(dataset_name: str, h5: Any) -> str:
    """从 h5 文件实际结构生成内嵌说明(中文):结构树/内容表/读取示例。

    直接遍历 dataset(真实 shape),不用 features 语义表(那里 shape 是
    LeRobot 语义,如 [1] 表示每帧 1 值,对客户不直观)。
    """
    import h5py as _h5py

    # —— 结构树(遍历实际 dataset,跳过 meta/ 与 README 自身)——
    entries: list[tuple[list[str], str]] = []

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, _h5py.Dataset):
            return
        parts = name.split("/")
        if parts[0] == "meta":
            return
        shape = "(" + ", ".join(str(s) for s in obj.shape) + ")"
        note = f"{shape:<28} {str(obj.dtype)}"
        attrs = dict(obj.attrs)
        if attrs.get("format") == "mp4":
            note += f" · MP4({attrs.get('frames')}帧,可提取为视频)"
        entries.append((parts, note))

    h5.visititems(visit)
    root = "episode_000000"
    root_items = [e for e in entries if e[0][0] == root]
    others = [e for e in entries if e[0][0] != root]
    tree: list[str] = [f"{root}/"]
    tree.extend(_tree_lines([(parts[1:], note) for parts, note in root_items], ""))
    for parts, note in others:
        tree.append(f"{'/'.join(parts)}    {note}")
    tree_str = "\n".join(tree)

    # —— 内容表(按实际结构判断)——
    all_names = {"/".join(e[0]) for e in entries}
    rows: list[str] = []
    img_keys = sorted(n for n in all_names
                      if n.startswith(f"{root}/observation/images/"))
    if img_keys:
        rows.append(f"| **原视频** | {len(img_keys)} 路原始帧 | ✅ 全量 |")
    hand_coordinate = next(
        (coord for coord in ("world", "3d", "2d")
         if f"{root}/observation/state/hand_left_{coord}" in all_names),
        None,
    )
    if hand_coordinate:
        description = {
            "world": "21 个关键点 × X/Y/Z 深度相机米制坐标",
            "3d": "21 个关键点 × X/Y/Z MediaPipe 相对 3D 坐标",
            "2d": "21 个关键点 × X/Y/Z 归一化图像坐标",
        }[hand_coordinate]
        rows.append(
            f"| **手部 {hand_coordinate.upper()} 骨骼** | {description}"
            " | ✅ 见有效帧标记列 |")
    skel = f"{root}/videos/hand_skeleton"
    if skel in all_names:
        skel_attrs = h5[skel].attrs
        rows.append(
            f"| **手部骨骼渲染视频** | `videos/hand_skeleton`:MP4 原始字节"
            f"({skel_attrs['width']}×{skel_attrs['height']} 左右目并排,骨架叠加) | "
            f"✅ {skel_attrs['frames']} 帧,{skel_attrs['fps']} fps |")
    if any(n.startswith(f"{root}/observation/tactile/") for n in all_names):
        rows.append("| **手套压力** | 16×16 压力阵列(SenseGlove) | ✅ 全帧 |")
    if f"{root}/observation/imu" in all_names:
        rows.append("| **IMU** | 6 轴惯性数据,与视频帧对齐 | ✅ 变长聚合,"
                    "配 `imu_frame_index` 切片还原 |")
    if f"{root}/observation/timestamps" in all_names:
        rows.append("| **帧时间戳** | 采集端每帧 2 条(左右目):"
                    "`timestamp` / `wall_time` / `hardware_ns` 三路时钟 |"
                    "✅ 用于帧级时间对齐分析 |")
    content_table = "\n".join(rows)

    # 提取视频章节仅在文件确实含 videos/hand_skeleton 时输出
    video_extract = ""
    if skel in all_names:
        video_extract = """
### 提取手部骨骼渲染视频(MP4)

`videos/hand_skeleton` 存的是 MP4 文件的**原始字节**,提取后即为标准 MP4 文件:

```python
import h5py

with h5py.File("dataset.h5", "r") as h5:
    mp4_bytes = h5["episode_000000/videos/hand_skeleton"][...].tobytes()

with open("hand_skeleton.mp4", "wb") as f:
    f.write(mp4_bytes)
```
"""

    readme = f"""# {dataset_name} 数据文件说明(HDF5 格式)

> 本文件由 EgoData 数据采集系统工作流自动生成
> 帧数/fps 与内容以 meta/info attrs(features)为准

---

## 一、文件是什么

本文件是 **HDF5 格式(Hierarchical Data Format 5)** 的单个数据容器——
一个文件内以"文件夹树"结构同时存放视频、传感器数据、手部 3D 骨骼、
骨骼渲染视频、标注等多种数据,便于传输、归档和程序化读取。

一个批次(一次数据采集片段)= 文件中的一个 episode。

## 二、文件内部结构

```
{tree_str}
```

## 三、核心内容一览

| 内容 | 说明 | 状态 |
|---|---|---|
{content_table}

## 四、如何读取

```python
import h5py

    with h5py.File("dataset.h5", "r") as h5:
        # 读一帧左目图像
        img = h5["episode_000000/observation/images/stereo_left"][0]
    # 读第一帧右手骨骼(字段可能是 world / 3d / 2d)
    hand3d = h5["episode_000000/observation/state/hand_right_{hand_coordinate or '3d'}"][0]
    # 读全部标注
    labels = h5["episode_000000/observation/annotation"][:]
```
{video_extract}
其他工具:HDFView、Panoply、MATLAB(h5read)等支持 HDF5 的软件均可打开。

---

*本说明随数据一同交付。*
"""
    return readme
