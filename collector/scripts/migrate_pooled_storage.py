#!/usr/bin/env python
"""一次性迁移脚本：旧「每会话一个子目录」布局 → v1.1.0 任务级池化布局。

旧布局（egodata v1.0，每会话一个子目录）:
    <task>/<task>_NNNNNN/
    ├── metadata.json / timestamps.json / calibration/*.json
    ├── videos/<cam>/chunk-0000/<cam>.mp4
    ├── data/<sensor>/chunk-0000/chunk_000000.parquet（每传感器一文件）
    ├── depth/<slot>/<slot>.mkv | NNNNNN.png | NNNNNN.bin
    └── meta/{info.json, stats.json, tasks.jsonl,
               episodes/{chunk_000000.parquet, chunk-000/file-000.parquet}}

新布局（v1.1.0 任务级池化；v1.1.2 起文件前缀 file- → episode-）:
    <task>/
    ├── videos/chunk-NNN/<image_key>/episode-NNN.ext # 每 episode 一文件
    ├── data/chunk-NNN/episode-NNN.parquet           # 单文件稀疏列
    └── meta/{info.json, stats.json, tasks.jsonl,
              episodes/chunk-NNN/episode-NNN.parquet}

迁移内容（每会话）:
  1. 按旧 metadata.json episode_index/created_at 排序定新全局序号 N（
     已存在池化 episode 之后续排）
  2. 视频 move: videos/<cam>/chunk-0000/<cam>.mp4 → 池化路径
  3. 深度: mkv 直接 move；png16/bin 序列经 numpy+ffmpeg 无损合成
     单流 FFV1 gray16le mkv（数据无损；显示层热力图由读取端按需生成）
  4. data parquet 列合并: timestamps.json 为骨架（含 wall_time/
     hardware_ns）+ 各传感器/IMU 按 frame_index 对齐（同帧多槽行取首行）
     + action/手部占位零 + 旧 status.<did> 列（如有）
  5. episodes 分片 append 一行（created_at/video_codec/drop_stats 取自
     metadata.json，calibration 取 calibration/*.json 原文）
  6. 任务级 info.json（并集合并）/ tasks.jsonl / stats.json（v1.1.1
     自含 count 累加器，无 .stats_state 边车）
  7. pipeline.db 改写（recording.file_path → 任务目录 + episode_index；
     upload_task.session_path 同）；改前自动备份
  8. keypoints_output/<task>/<旧会话名>/ → <task>/episode_{N:06d}/
  9. 旧会话目录先**全量快照复制**到 data/recordings/_migrate_backup_<ts>/
     再移动媒体（快照在前是 restore 无损的前提），迁移完成删除残壳
     （--no-backup 跳过快照=显式破坏性模式；manifest 记录全部映射）

用法:
    venv/bin/python scripts/migrate_pooled_storage.py --dry-run
    venv/bin/python scripts/migrate_pooled_storage.py [--backup] [--task 名称]
    venv/bin/python scripts/migrate_pooled_storage.py --validate
    venv/bin/python scripts/migrate_pooled_storage.py --restore [--task 名称]

限制（一次性工具，已知可接受）:
  - stats 为累加器语义：--restore 不回退 stats 数值（若任务无剩余
    episode 则删除 stats 文件；有剩余则保留并含已迁移数值）
  - 旧格式行无视频帧-传感器一一对应元信息：传感器/IMU 按 frame_index
    挂到同帧首行（旧读取端同口径）
  - 深度合成单流 FFV1（非双流），显示端按需热力图
  - 无 ffmpeg 时跳过 png16/bin 深度会话并警告（视频/数据仍迁移）
  - 快照复制需要 2× 磁盘临时余量；确认备份无误后再删除备份区
"""
import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import settings
from core.helpers import (episode_chunk_file, POOLED_CHUNK_SIZE,
                          pooled_video_path,
                          pooled_data_parquet_path, pooled_episodes_path,
                          _legacy_episodes_shard_path,
                          pooled_info_path, pooled_stats_path,
                          pooled_tasks_jsonl_path,
                          load_stats_acc, merge_stat_block,
                          acc_to_stats_json, recalc_stats,
                          list_task_episodes)
from core.egodata_writer import (_episode_rows_table, _read_episode_rows,
                                 _atomic_write_parquet, _LockedFile)

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

try:
    import cv2  # png16 解码
except ImportError:
    cv2 = None

MANIFEST_VERSION = 1
MANIFEST_NAME = ".migrate_manifest.json"

RECORDINGS_DIR = os.path.join(settings.DATA_DIR, "recordings")
DB_PATH = os.path.join(settings.DATA_DIR, "pipeline.db")

DEFAULT_SENSOR_DIM = 256


# ── 日志 / 报告 ──────────────────────────────────────────────

class Report:
    """迁移执行报告（dry-run 只打印，实跑落到 backup 根目录）。"""

    def __init__(self):
        self.sessions = []
        self.warnings = []

    def add(self, task, session, n, kind, detail, created, db_rows, warns):
        self.sessions.append({
            "task": task, "session": session, "N": n, "kind": kind,
            "detail": detail, "created": created, "db_rows": db_rows,
        })
        self.warnings.extend(warns)

    def print(self, label):
        print(f"\n===== 迁移报告 [{label}] =====")
        for s in self.sessions:
            print(f"  [{s['kind']}] {s['task']} :: {s['session']} → "
                  f"episode {s['N']}  ({s['detail']})")
            for f in s["created"]:
                print(f"       + {f}")
            for d in s["db_rows"]:
                print(f"       db {d['table']} id={d['id']}")
        if self.warnings:
            print("  ⚠ 警告:")
            for w in self.warnings:
                print(f"       - {w}")
        if not self.sessions:
            print("  （无可迁移/已回滚的会话）")


# ── 会话发现 ────────────────────────────────────────────────

def task_dirs(root, task_filter):
    out = []
    for name in sorted(os.listdir(root)):
        if name.startswith("_") or name.startswith("."):
            continue
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if task_filter and name not in task_filter:
            continue
        out.append((name, d))
    return out


def legacy_sessions(task_dir, task):
    """任务目录下的旧会话子目录（有 metadata.json 且非池化结构目录）。"""
    out = []
    try:
        children = os.listdir(task_dir)
    except OSError:
        return out
    for name in children:
        d = os.path.join(task_dir, name)
        if not os.path.isdir(d):
            continue
        if name in ("videos", "data", "meta", "depth", "calibration") \
                or name.startswith("chunk-") or name.startswith("_"):
            continue
        if os.path.isfile(os.path.join(d, "metadata.json")):
            out.append(d)
    out.sort(key=lambda d: _session_sort_key(d, task))
    return out


def _session_sort_key(session_dir, task):
    """排序键：旧 metadata.json 的 episode_index，回退 created_at/名字。"""
    try:
        with open(os.path.join(session_dir, "metadata.json"),
                  "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        m = {}
    return (int(m.get("episode_index", 0) or 0),
            float(m.get("created_at", 0) or 0),
            os.path.basename(session_dir))


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


# ── 旧数据读取 ──────────────────────────────────────────────

def read_ts_rows(session_dir):
    """timestamps.json → ([{frame_index, timestamp, wall_time, hardware_ns}],
    total_frames)；缺失返回 (None, None)。"""
    ts = load_json(os.path.join(session_dir, "timestamps.json"))
    if not ts or not isinstance(ts.get("timestamps"), list):
        return None, None
    return ts["timestamps"], int(ts.get("total_frames") or 0)


def read_sensor_parquet(session_dir, sensor):
    """data/<sensor>/chunk-0000/chunk_000000.parquet → {frame_index: row}"""
    for rel in (f"data/{sensor}/chunk-0000/chunk_000000.parquet",
                f"data/{sensor}/chunk_000000.parquet",
                f"data/{sensor}/chunk-000/file-000.parquet"):
        p = os.path.join(session_dir, rel)
        if os.path.isfile(p):
            import pyarrow.parquet as pq
            t = pq.read_table(p)
            idx = {}
            for r in t.to_pylist():
                fi = int(r.get("frame_index", 0))
                if fi not in idx:      # 同帧多槽行取首行
                    idx[fi] = r
            return idx
    return None


def find_legacy_media(session_dir, metadata):
    """定位旧视频/深度文件。

    返回 ({cam: (src, kind)}, {depth_slot: (src, kind)})
    kind ∈ {move, png16, bin}
    """
    vids, depths = {}, {}
    for cam, info in (metadata.get("cameras") or {}).items():
        if not isinstance(info, dict):
            continue
        ctype = info.get("type", "rgb")
        if ctype == "depth":
            src = _find_depth_source(session_dir, cam)
            if src:
                depths[cam] = src
            continue
        # rgb: videos/<cam>/chunk-0000/<cam>.mp4（兼容无 chunk 层）
        for rel in (f"videos/{cam}/chunk-0000/{cam}.mp4",
                    f"videos/{cam}/{cam}.mp4"):
            p = os.path.join(session_dir, rel)
            if os.path.isfile(p):
                vids[cam] = (p, "move")
                break
        if cam not in vids:
            # 兜底：videos/<cam> 下任意 mp4（槽名可能与 metadata 名不同）
            d = os.path.join(session_dir, "videos", cam)
            if os.path.isdir(d):
                for root, _dirs, files in os.walk(d):
                    for fn in sorted(files):
                        if fn.endswith(".mp4"):
                            vids[cam] = (os.path.join(root, fn), "move")
                            break
                    if cam in vids:
                        break
    return vids, depths


def _find_depth_source(session_dir, slot):
    mkv = os.path.join(session_dir, "depth", slot, f"{slot}.mkv")
    if os.path.isfile(mkv):
        return (mkv, "move")
    d = os.path.join(session_dir, "depth", slot)
    if not os.path.isdir(d):
        return None
    pngs = sorted(f for f in os.listdir(d)
                  if f.endswith(".png") and f.split(".")[0].isdigit())
    if pngs:
        return (os.path.join(d, pngs[0]), "png16")
    bins = sorted(f for f in os.listdir(d)
                  if f.endswith(".bin") and f.split(".")[0].isdigit())
    if bins:
        return (os.path.join(d, bins[0]), "bin")
    return None


def depth_frame_files(session_dir, slot):
    """深度帧文件按数字文件名升序 → [(abs_path, frame_no)]"""
    d = os.path.join(session_dir, "depth", slot)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        stem = fn.rsplit(".", 1)[0]
        if stem.isdigit() and fn.rsplit(".", 1)[1] in ("png", "bin"):
            out.append((os.path.join(d, fn), int(stem)))
    out.sort(key=lambda x: x[1])
    return out


# ── 深度合成（png16 / bin → 单流 FFV1 gray16le mkv） ─────────

def find_ffmpeg(metadata):
    """ffmpeg 二进制：metadata 记录的 imageio_ffmpeg 路径 → PATH → None。"""
    vc = metadata.get("video_codec") or {}
    p = vc.get("ffmpeg")
    if p and os.path.isfile(p):
        return p
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def synthesize_depth_mkv(task_dir, session_dir, slot, metadata, out_path,
                         warn):
    """png16/bin 序列 → 无损单流 FFV1 gray16le mkv（out_path）。"""
    frames = depth_frame_files(session_dir, slot)
    if not frames:
        warn(f"{slot}: 深度帧目录为空，跳过")
        return False
    ffmpeg = find_ffmpeg(metadata)
    if not ffmpeg:
        warn(f"{slot}: 无 ffmpeg，跳过深度合成（{len(frames)} 帧未迁移）")
        return False
    cam = (metadata.get("cameras") or {}).get(slot) or {}
    fps = float(metadata.get("fps") or cam.get("fps") or 30.0)
    width = int(cam.get("width") or 0)
    height = int(cam.get("height") or 0)

    tmp_raw = os.path.join(task_dir, f".migrate_tmp_{slot}_N{int(time.time()*1000)%100000}.raw")
    try:
        # 逐帧解码/读入 memmap，避免整段驻留内存
        first = None
        count = 0
        ext = os.path.splitext(frames[0][0])[1].lower()
        if ext == ".png":
            if cv2 is None:
                warn(f"{slot}: 无 cv2（OpenCV），跳过 png16 深度合成")
                return False
            for path, _no in frames:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 3:
                    img = img[..., 0]
                if img.dtype != np.uint16:
                    warn(f"{slot}: png 非 uint16（{img.dtype}），跳过深度合成")
                    return False
                if first is None:
                    first = img
                    width, height = img.shape[1], img.shape[0]
                count += 1
        else:  # bin: raw uint16
            if width <= 0 or height <= 0:
                warn(f"{slot}: bin 序列缺分辨率（metadata 无 width/height），"
                     "跳过深度合成")
                return False
            first = np.zeros((height, width), dtype=np.uint16)
            count = len(frames)
        if first is None or count == 0:
            return False

        npx = width * height
        mm = np.memmap(tmp_raw, dtype=np.uint16, mode="w+",
                       shape=(count * npx,))
        i = 0
        if ext == ".png":
            for path, _no in frames:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None or img.ndim == 3:
                    img = img[..., 0] if img is not None and img.ndim == 3 \
                        else np.zeros((height, width), np.uint16)
                mm[i * npx:(i + 1) * npx] = img.reshape(-1)[:npx]
                i += 1
        else:
            for path, _no in frames:
                raw = np.fromfile(path, dtype=np.uint16)
                mm[i * npx:(i + 1) * npx] = raw[:npx]
                i += 1
        mm.flush()
        del mm

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "gray16le",
               "-s", f"{width}x{height}", "-framerate", f"{fps}",
               "-i", tmp_raw,
               "-c:v", "ffv1", "-pix_fmt", "gray16le", out_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            warn(f"{slot}: ffmpeg 合成失败: {r.stderr.strip()[:300]}")
            return False
        return True
    finally:
        try:
            os.remove(tmp_raw)
        except OSError:
            pass


# ── data parquet 列合并 ─────────────────────────────────────

def build_merged_table(meta, ts_rows, sensor_idx, imu_idx):
    """按新格式契约构建本 episode 的 data 表。

    ts_rows 为骨架（一行一帧槽位行）；sensor_idx/imu_idx 为
    {frame_index: 旧行}（同帧多槽行取首行）。
    """
    import pyarrow as pa
    n_ep = int(meta.get("episode_index", 0))  # 由调用方覆盖为新 N
    sensor_dim = int((meta.get("sensor_dim") or DEFAULT_SENSOR_DIM)
                     or DEFAULT_SENSOR_DIM)
    hand_dim = int(settings.HAND_POSE_DIM or 63)

    sensors = sorted(sensor_idx.keys())
    cols = {}
    cols["episode_index"] = pa.array([n_ep] * len(ts_rows), pa.int64())
    cols["frame_index"] = pa.array([int(r.get("frame_index", 0))
                                    for r in ts_rows], pa.int64())
    cols["timestamp"] = pa.array([float(r.get("timestamp", 0.0))
                                  for r in ts_rows], pa.float32())
    cols["task_index"] = pa.array([0] * len(ts_rows), pa.int64())
    cols["wall_time"] = pa.array([float(r.get("wall_time", 0.0))
                                  for r in ts_rows], pa.float64())
    cols["hardware_ns"] = pa.array([int(r.get("hardware_ns", 0) or 0)
                                    for r in ts_rows], pa.int64())
    cols["action"] = pa.array([[0.0]] * len(ts_rows),
                              pa.list_(pa.float32(), 1))

    for sn in sensors:
        idx = sensor_idx[sn]
        vals = []
        for r in ts_rows:
            fi = int(r.get("frame_index", 0))
            row = idx.get(fi)
            v = (row or {}).get(f"observation.{sn}")
            if v is None:
                v = [0.0] * sensor_dim
            v = [float(x) for x in v]
            if len(v) < sensor_dim:
                v = v + [0.0] * (sensor_dim - len(v))
            vals.append(v[:sensor_dim])
        cols[f"observation.{sn}"] = pa.array(
            vals, pa.list_(pa.float32(), sensor_dim))

    if imu_idx is not None:
        # IMU 只挂在每帧首行（旧格式=stereo_left 行，左右目共享样本）
        ts_list, imu_list = [], []
        seen = set()
        for r in ts_rows:
            fi = int(r.get("frame_index", 0))
            row = imu_idx.get(fi)
            if fi in seen or row is None:
                ts_list.append([])
                imu_list.append([])
                continue
            seen.add(fi)
            ts_list.append([int(x) for x in (row.get("imu_ts_ns") or [])])
            imu_list.append([[float(x) for x in s]
                             for s in (row.get("observation.imu") or [])])
        cols["imu_ts_ns"] = pa.array(ts_list, pa.list_(pa.int64()))
        cols["observation.imu"] = pa.array(
            imu_list, pa.list_(pa.list_(pa.float32(), 6)))

    for pose in (settings.HAND_POSE_LEFT, settings.HAND_POSE_RIGHT):
        cols[f"observation.{pose}"] = pa.array(
            [[0.0] * hand_dim] * len(ts_rows),
            pa.list_(pa.float32(), hand_dim))

    # status.<did>：旧 parquet 携带过设备状态列则保留（同帧取首行）
    dids = set()
    for idx in sensor_idx.values():
        for row in (idx or {}).values():
            for key in (row or {}).keys():
                if isinstance(key, str) and key.startswith("status."):
                    dids.add(key[len("status."):])
    for did in sorted(dids):
        vals = []
        for r in ts_rows:
            fi = int(r.get("frame_index", 0))
            row = next((idx.get(fi) for idx in sensor_idx.values()
                        if f"status.{did}" in (idx.get(fi) or {})), None)
            vals.append(str((row or {}).get(f"status.{did}", "connected")))
        cols[f"status.{did}"] = pa.array(vals, pa.string())

    return pa.table(cols), sensors


# ── 统计（stats.json 自含累加器，v1.1.1） ──────────────────

def episode_stat_blocks(table, sensors, has_imu):
    """本 episode 各列的统计块 {key: {count,sum,sum_sq,min,max}}。"""
    blocks = {}
    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    for sn in sensors:
        name = f"observation.{sn}"
        vals = [np.asarray(v, dtype=np.float64) for v in cols[name]
                if v and any(x != 0 for x in v)]
        if not vals:
            continue
        arr = np.stack(vals)
        blocks[name] = {
            "count": int(arr.shape[0]),
            "sum": arr.sum(axis=0).tolist(),
            "sum_sq": (arr * arr).sum(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
        }
    if has_imu:
        samples = [np.asarray(s, dtype=np.float64)
                   for v in cols["observation.imu"] for s in v]
        if samples:
            arr = np.stack(samples)
            blocks["observation.imu"] = {
                "count": int(arr.shape[0]),
                "sum": arr.sum(axis=0).tolist(),
                "sum_sq": (arr * arr).sum(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }
    return blocks


def merge_state_and_write(task_dir, blocks):
    """把各 episode 统计块合并进 stats.json（v1.1.1 自含累加器，同写入器）。"""
    acc, need_recalc = load_stats_acc(task_dir)
    if need_recalc:
        acc = recalc_stats(task_dir)
    for key, blk in blocks.items():
        dim = len(blk["sum"])
        merge_stat_block(acc, key, blk, dim)
    _atomic_write_json(pooled_stats_path(task_dir), acc_to_stats_json(acc))


def _atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── info.json 合成（任务级并集合并，同写入器语义） ──────────

def build_info(task_dir, task, sessions_meta):
    """sessions_meta: [(metadata, sensors, has_imu, calib_dict)] 按 N 升序。"""
    path = pooled_info_path(task_dir)
    base = load_json(path, {})
    if not isinstance(base, dict):
        base = {}

    merged_features = dict(base.get("features") or {})
    merged_cameras = dict(base.get("cameras") or {})
    merged_exts = dict(base.get("video_extensions") or {})
    merged_calib = dict(base.get("calibration") or {})
    sensors_union, has_imu_any = set(), False

    last = None
    for meta, sensors, has_imu, calib in sessions_meta:
        last = meta
        sensors_union |= set(sensors)
        has_imu_any = has_imu_any or has_imu
        for sn in sorted(sensors):
            merged_features[f"observation.{sn}"] = {
                "dtype": "float32", "shape": [16, 16]}
        if has_imu:
            merged_features["observation.imu"] = {
                "dtype": "float32", "shape": [6]}
        merged_features["action"] = {"dtype": "float32", "shape": [1]}
        for cam, info in (meta.get("cameras") or {}).items():
            if not isinstance(info, dict):
                continue
            if info.get("type") == "depth":
                merged_exts[cam] = "mkv"
                continue
            entry = {"height": int(info.get("height", 0)),
                     "width": int(info.get("width", 0))}
            cam_fps = float(info.get("fps") or meta.get("fps") or 0.0)
            if cam_fps and cam_fps != float(meta.get("fps") or 0.0):
                entry["fps"] = cam_fps
            merged_cameras[cam] = entry
        merged_exts.update({cam: "mp4" for cam in merged_cameras})
        merged_calib.update(calib or {})

    devices, device_names = [], {}
    if last:
        devices = [{"key": d.get("key", ""), "kind": d.get("kind", ""),
                    "name": d.get("name", ""),
                    "slots": list(d.get("slots") or [])}
                   for d in (last.get("devices") or [])]
        for d in (last.get("devices") or []):
            if d.get("name"):
                for slot in (d.get("slots") or []):
                    device_names[slot] = d["name"]

    info = dict(base)
    info.update({
        "format": "pooled_episodes_v1",
        "chunks_size": POOLED_CHUNK_SIZE,
        "data_path": "data/chunk-{c:03d}/episode-{f:03d}.parquet",
        "video_path": "videos/chunk-{c:03d}/{image_key}/episode-{f:03d}.{ext}",
        "episodes_path": "meta/episodes/chunk-{c:03d}/episode-{f:03d}.parquet",
        "codebase_version": (last or {}).get("codebase_version", "v3.0"),
        "app_version": f"{settings.APP_VERSION}-migrated",
        "fps": float((last or {}).get("fps") or 30.0),
        "video": len(merged_cameras) > 0,
        "task_name": task,
        "features": merged_features,
        "cameras": merged_cameras,
        "devices": devices,
        "device_names": device_names,
        "sensors": sorted(sensors_union),
        "sensor_dim": int((last or {}).get("sensor_dim")
                          or DEFAULT_SENSOR_DIM),
        "created_at": max([float((m or {}).get("created_at") or 0.0)
                           for m, *_ in sessions_meta] or [time.time()]),
        "calibration": merged_calib,
        "total_episodes": len(list_task_episodes(task_dir)),
        "video_extensions": merged_exts,
    })
    return info


# ── episodes 分片 append ───────────────────────────────────

def _task_lock(task_dir):
    lock_path = os.path.join(task_dir, "meta", "episodes", ".lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fh = open(lock_path, "a+")
    if _fcntl is not None:
        _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
    return _LockedFile(fh)


def append_episode_row(task_dir, n, row):
    """每段一个文件、单行原子写（与 data/videos 同编号）；旧分片里
    的同号行一并删除，避免双份。"""
    path = pooled_episodes_path(task_dir, n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _task_lock(task_dir):
        _atomic_write_parquet(_episode_rows_table([row]), path)
        cidx, _ = episode_chunk_file(n)
        legacy = _legacy_episodes_shard_path(task_dir, cidx)
        if legacy != path and os.path.isfile(legacy):
            rows = _read_episode_rows(legacy)
            kept = [r for r in rows if int(r.get("episode_index", 0)) != n]
            if len(kept) != len(rows):
                if kept:
                    _atomic_write_parquet(_episode_rows_table(kept), legacy)
                else:
                    os.remove(legacy)


def remove_episode_row(task_dir, n):
    """移除 episode N 的 episodes 元数据（每段文件 → os.remove；
    旧分片回退 → 读-改-写删行）。N=1 时分片与每段文件同路径，
    先按内容判每段文件才删，避免误删整个旧分片。"""
    path = pooled_episodes_path(task_dir, n)
    cidx, _ = episode_chunk_file(n)
    legacy = _legacy_episodes_shard_path(task_dir, cidx)
    with _task_lock(task_dir):
        removed_path = False
        if os.path.isfile(path):
            rows = _read_episode_rows(path)
            if len(rows) <= 1 and (not rows or
                                   int(rows[0].get("episode_index", 0)) == n):
                os.remove(path)  # 每段文件（或旧分片只剩该行）
                removed_path = True
        if os.path.isfile(legacy) and not (removed_path and legacy == path):
            rows = _read_episode_rows(legacy)
            kept = [r for r in rows if int(r.get("episode_index", 0)) != n]
            if len(kept) != len(rows):
                if kept:
                    _atomic_write_parquet(_episode_rows_table(kept), legacy)
                else:
                    os.remove(legacy)  # 分片空 → 删除文件本身


# ── DB 改写 ────────────────────────────────────────────────

def db_rows_for_path(db_path, col, old_path):
    """返回 {table: [(id, 现值)]}——只匹配旧会话目录绝对路径。"""
    if not os.path.isfile(db_path):
        return [], []
    con = sqlite3.connect(db_path)
    try:
        rec = con.execute(
            "SELECT id, file_path FROM recording "
            "WHERE file_path=? AND episode_index=0", (old_path,)).fetchall()
        up = con.execute(
            "SELECT id, session_path FROM upload_task "
            "WHERE session_path=?", (old_path,)).fetchall()
    except sqlite3.OperationalError:
        rec, up = [], []
    finally:
        con.close()
    return [{"id": r[0], "table": "recording", "old": r[1]}
            for r in rec], [{"id": r[0], "table": "upload_task", "old": r[1]}
                            for r in up]


def db_update_episode(db_path, rows, task_dir, n, task):
    con = sqlite3.connect(db_path)
    try:
        for r in rows:
            if r["table"] == "recording":
                con.execute("UPDATE recording SET file_path=?, "
                            "episode_index=? WHERE id=?",
                            (task_dir, n, r["id"]))
            else:
                con.execute("UPDATE upload_task SET session_path=?, "
                            "session_name=?, episode_index=? WHERE id=?",
                            (task_dir, task, n, r["id"]))
        con.commit()
    finally:
        con.close()


def db_restore_episode(db_path, rows):
    con = sqlite3.connect(db_path)
    try:
        for r in rows:
            if r["table"] == "recording":
                con.execute("UPDATE recording SET file_path=?, "
                            "episode_index=0 WHERE id=?",
                            (r["old"], r["id"]))
            else:
                con.execute("UPDATE upload_task SET session_path=?, "
                            "episode_index=0 WHERE id=?",
                            (r["old"], r["id"]))
        con.commit()
    finally:
        con.close()


# ── 单个会话迁移 ───────────────────────────────────────────

def migrate_session(task_dir, task, session_dir, n, backup_root, report,
                    dry_run, backup, db_path):
    warns = []
    meta = load_json(os.path.join(session_dir, "metadata.json"), {})
    if not meta:
        return "skip", warns
    ts_rows, total_frames = read_ts_rows(session_dir)
    if ts_rows is None or not ts_rows:
        # 无 timestamps.json：从传感器 parquet 并集重建骨架（wall_time 近似）
        import pyarrow.parquet as pq
        skeleton = {}
        for sub in os.listdir(os.path.join(session_dir, "data")):
            idx = read_sensor_parquet(session_dir, sub)
            for fi, r in (idx or {}).items():
                skeleton.setdefault(fi, {
                    "frame_index": fi,
                    "timestamp": float(r.get("timestamp", 0.0)),
                    "wall_time": float(meta.get("created_at", time.time()))
                    + float(r.get("timestamp", 0.0)),
                    "hardware_ns": int(r.get("hardware_ns", 0) or 0),
                })
        ts_rows = [skeleton[k] for k in sorted(skeleton)]
        warns.append("无 timestamps.json，骨架由 data parquet 重建"
                     "（wall_time 近似）")
    if not ts_rows:
        return "skip", ["无 timestamps.json 且无 data parquet，跳过"]

    vids, depths = find_legacy_media(session_dir, meta)

    # 防撞护栏：任何计划目标已存在 → 整会话跳过，绝不覆盖池化已有文件
    # （部分迁移/并发重跑时 N 可能撞号，文件级检查是最强不变量）
    existing = []
    for cam in sorted(vids):
        if os.path.exists(pooled_video_path(task_dir, cam, n, ext="mp4")):
            existing.append(os.path.relpath(
                pooled_video_path(task_dir, cam, n, ext="mp4"), task_dir))
    for slot in sorted(depths):
        if os.path.exists(pooled_video_path(task_dir, slot, n, ext="mkv")):
            existing.append(os.path.relpath(
                pooled_video_path(task_dir, slot, n, ext="mkv"), task_dir))
    if os.path.exists(pooled_data_parquet_path(task_dir, n)):
        existing.append(os.path.relpath(
            pooled_data_parquet_path(task_dir, n), task_dir))
    # episodes 行撞号（restore 半途失败残留行 → 追加会造重复行；
    # list_task_episodes 只扫 data/videos 文件，看不到 episodes 行）
    # 每段文件与旧分片两条路径都要查（N=1 时两者同路径，set 去重）
    import pyarrow.parquet as pq
    cidx, _ = episode_chunk_file(n)
    for cand in {pooled_episodes_path(task_dir, n),
                 _legacy_episodes_shard_path(task_dir, cidx)}:
        if os.path.isfile(cand):
            try:
                if any(r["episode_index"] == n for r in pq.read_table(
                        cand, columns=["episode_index"]).to_pylist()):
                    existing.append(f"meta/episodes 行 {n}")
            except Exception:
                pass
    if existing:
        return "skip", [f"防撞护栏：池化目标已存在，跳过本会话绝不覆盖："
                        f"{existing[:4]}"]

    # ① 完整快照先行（媒体 move 出目录之前），否则 restore 会丢数据
    backup_path = os.path.join(backup_root, task,
                               os.path.basename(session_dir))
    if not dry_run and backup:
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        try:
            shutil.copytree(session_dir, backup_path)
        except OSError as e:
            return "skip", [f"快照失败（{e}），本会话不动"]
        if not os.path.isfile(os.path.join(backup_path, "metadata.json")):
            shutil.rmtree(backup_path, ignore_errors=True)
            return "skip", ["快照缺 metadata.json（校验失败），本会话不动"]
    sensors_found = {}
    imu_idx = None
    data_dir = os.path.join(session_dir, "data")
    if os.path.isdir(data_dir):
        for sub in sorted(os.listdir(data_dir)):
            if sub == "imu":
                imu_idx = read_sensor_parquet(session_dir, "imu")
                continue
            idx = read_sensor_parquet(session_dir, sub)
            if idx:
                # 全零传感器视为从未出现（旧格式零填充=缺席）
                present = False
                for r in idx.values():
                    v = r.get(f"observation.{sub}")
                    if v and any(float(x) != 0 for x in v):
                        present = True
                        break
                if present:
                    sensors_found[sub] = idx
                else:
                    warns.append(f"传感器 {sub} 全零，视为缺席不写列")

    created = []

    def plan_create(rel):
        return os.path.join(task_dir, rel)

    # 视频 move
    for cam, (src, kind) in sorted(vids.items()):
        dst = pooled_video_path(task_dir, cam, n, ext="mp4")
        created.append(dst)
        if not dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _move_file(src, dst)
    for cam, info in (meta.get("cameras") or {}).items():
        if (isinstance(info, dict)
                and info.get("type", "rgb") != "depth"
                and cam not in vids):
            warns.append(f"摄像头 {cam} 无视频文件（旧目录缺 mp4）")

    # 深度 move / 合成
    depth_done = {}
    for slot, (src, kind) in sorted(depths.items()):
        dst = pooled_video_path(task_dir, slot, n, ext="mkv")
        created.append(dst)
        if kind == "move":
            if not dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                _move_file(src, dst)
            depth_done[slot] = "mkv"
        else:  # png16 / bin
            if dry_run:
                depth_done[slot] = f"{kind}→mkv"
                continue
            ok = synthesize_depth_mkv(task_dir, session_dir, slot, meta,
                                      dst, warns.append)
            depth_done[slot] = "mkv" if ok else "skipped"

    # data parquet
    import pyarrow as pa
    table, sensors = build_merged_table(meta, ts_rows, sensors_found,
                                        imu_idx)
    table = table.set_column(
        0, "episode_index",
        pa.array([n] * table.num_rows, pa.int64()))
    data_path = pooled_data_parquet_path(task_dir, n)
    created.append(data_path)
    if not dry_run:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        _atomic_write_parquet(table, data_path)

    # episodes 行
    calib = {}
    cal_dir = os.path.join(session_dir, "calibration")
    if os.path.isdir(cal_dir):
        for fn in sorted(os.listdir(cal_dir)):
            if fn.endswith(".json"):
                c = load_json(os.path.join(cal_dir, fn))
                if isinstance(c, dict):
                    calib[fn[:-5]] = c
    video_codec = dict(meta.get("video_codec") or {})
    video_codec.pop("ffmpeg", None)  # 机器相关绝对路径不入库
    fps = float(meta.get("fps") or 30.0)
    ts_vals = [float(r.get("timestamp", 0.0)) for r in ts_rows]
    ep_row = {
        "episode_index": n,
        "task_index": 0,
        "start_frame_index": int(min(r["frame_index"] for r in ts_rows)),
        "end_frame_index": int(max(r["frame_index"] for r in ts_rows)),
        "length": int(total_frames
                      or len({int(r["frame_index"]) for r in ts_rows})),
        "created_at": float(meta.get("created_at") or time.time()),
        # 真实墙钟时长（多槽行共享时间轴，不随行数翻倍）
        "duration_sec": float(max(0.0, max(ts_vals) - min(ts_vals))
                              + 1.0 / fps),
        "drop_stats": json.dumps(meta.get("drop_stats") or {},
                                 ensure_ascii=False),
        "video_codec": json.dumps(video_codec, ensure_ascii=False),
        "calibration": json.dumps(calib, ensure_ascii=False),
    }
    if not dry_run:
        append_episode_row(task_dir, n, ep_row)

    # DB 改写（先于目录移动，崩溃后重跑可从 manifest 续）
    db_rec, db_up = db_rows_for_path(db_path, "file_path", session_dir)
    if not dry_run and (db_rec or db_up):
        db_update_episode(db_path, db_rec + db_up, task_dir, n, task)

    # keypoints_output 重键
    kp_moves = []
    kp_base = os.path.join(settings.DATA_DIR, "keypoints_output", task)
    kp_old = os.path.join(kp_base, os.path.basename(session_dir))
    kp_new = os.path.join(kp_base, f"episode_{n:06d}")
    if os.path.isdir(kp_old):
        kp_moves.append((kp_old, kp_new))
        if not dry_run:
            os.makedirs(os.path.dirname(kp_new), exist_ok=True)
            _move_file(kp_old, kp_new)

    # ② 媒体已 move 入池化（快照在前，--no-backup 时无快照）；删除残壳
    # 防重跑重复发现（--no-backup 属显式破坏性模式）
    if not dry_run:
        shutil.rmtree(session_dir, ignore_errors=True)

    report.add(task, os.path.basename(session_dir), n,
               "migrated" if not dry_run else "planned",
               f"视频 {len(vids)} 路 / 深度 {list(depth_done.values())} / "
               f"数据 {table.num_rows} 行 x {len(sensors)} 传感器",
               created, db_rec + db_up, warns)
    return {
        "session": os.path.basename(session_dir),
        "N": n,
        "backup_path": backup_path,
        "created_files": created,
        "db_rows": db_rec + db_up,
        "keypoints": [{"from": a, "to": b} for a, b in kp_moves],
        "metadata": meta,
        "sensors": sensors,
        "has_imu": imu_idx is not None
        and any((r.get("observation.imu") or []) for r in imu_idx.values()),
        "calib": calib,
        "warnings": warns,
    }, warns


def _move_file(src, dst):
    try:
        os.rename(src, dst)
    except OSError:
        shutil.move(src, dst)


# ── 任务级收尾 ─────────────────────────────────────────────

def finalize_task(task_dir, task, session_entries, dry_run):
    """info.json / tasks.jsonl / stats 合成（实跑时）。"""
    metas = [(e["metadata"], e["sensors"], e["has_imu"], e["calib"])
             for e in session_entries if e.get("metadata")]
    info = build_info(task_dir, task, metas)
    blocks = {}
    if not dry_run:
        for e in session_entries:
            import pyarrow.parquet as pq
            p = pooled_data_parquet_path(task_dir, e["N"])
            if os.path.isfile(p):
                t = pq.read_table(p)
                blocks.update(episode_stat_blocks(
                    t, e["sensors"], e["has_imu"]))
        _atomic_write_json(pooled_info_path(task_dir), info)
        if not os.path.isfile(pooled_tasks_jsonl_path(task_dir)):
            with open(pooled_tasks_jsonl_path(task_dir), "w",
                      encoding="utf-8") as f:
                json.dump({"task_index": 0, "task": task}, f,
                          ensure_ascii=False)
                f.write("\n")
        merge_state_and_write(task_dir, blocks)
    return info


# ── manifest ───────────────────────────────────────────────

def manifest_path(task_dir):
    return os.path.join(task_dir, "meta", MANIFEST_NAME)


def load_manifest(task_dir):
    m = load_json(manifest_path(task_dir), {})
    if not isinstance(m, dict):
        m = {}
    return m


def save_manifest(task_dir, manifest):
    _atomic_write_json(manifest_path(task_dir), manifest)


# ── 回滚 ───────────────────────────────────────────────────

def restore_task(task_dir, task, manifest, dry_run, report):
    """按 manifest 回滚一个任务的全部迁移。"""
    entries = manifest.get("sessions") or []
    for e in reversed(entries):
        n = int(e["N"])
        for f in e.get("created_files") or []:
            if os.path.isfile(f):
                if not dry_run:
                    os.remove(f)
        remove_episode_row(task_dir, n) if not dry_run else None
        if not dry_run and (e.get("db_rows")):
            db_restore_episode(DB_PATH, e["db_rows"])
        for kp in e.get("keypoints") or []:
            if os.path.isdir(kp["to"]):
                if not dry_run:
                    os.makedirs(os.path.dirname(kp["from"]),
                                exist_ok=True)
                    _move_file(kp["to"], kp["from"])
        bp = e.get("backup_path")
        if bp and os.path.isdir(bp):
            if not dry_run:
                os.makedirs(task_dir, exist_ok=True)
                _move_file(bp, os.path.join(
                    task_dir, os.path.basename(bp)))
        report.add(task, e.get("session", "?"), n,
                   "restored" if not dry_run else "planned-restore",
                   f"backup={bp}", [], [], e.get("warnings") or [])
    # 备份根目录若已清空则一并回收
    if not dry_run:
        for entry in entries:
            bp = entry.get("backup_path")
            if bp:
                root = os.path.dirname(os.path.dirname(bp))
                if os.path.isdir(root) and not os.listdir(root):
                    _rmdir_empty(root)
    # 任务清空则回收 info/stats/tasks；有剩余 episode 则保留（stats 不回退）
    if not dry_run and not list_task_episodes(task_dir):
        for p in (pooled_info_path(task_dir), pooled_stats_path(task_dir),
                  pooled_tasks_jsonl_path(task_dir)):
            try:
                os.remove(p)
            except OSError:
                pass
    # 池化结构空目录自底向上回收（不动已恢复的旧会话目录）
    if not dry_run:
        for sub in ("videos", "data"):
            for root, dirs, _files in os.walk(
                    os.path.join(task_dir, sub), topdown=False):
                _rmdir_empty(root)
        ep_dir = os.path.join(task_dir, "meta", "episodes")
        for root, _dirs, _files in os.walk(ep_dir, topdown=False):
            # 注意：不能用 walk 的 dirs 判断——那是下探时快照，
            # 子目录在自底向上过程中已被删，快照仍非空会挡住清理
            leftovers = [e for e in os.listdir(root) if e != ".lock"]
            if not leftovers:
                try:
                    os.remove(os.path.join(root, ".lock"))
                except OSError:
                    pass
            _rmdir_empty(root)
        _rmdir_empty(os.path.join(task_dir, "meta"))
    if not dry_run:
        try:
            os.remove(manifest_path(task_dir))
        except OSError:
            pass
        _rmdir_empty(os.path.join(task_dir, "meta"))


def _rmdir_empty(path):
    try:
        os.rmdir(path)
    except OSError:
        pass


# ── 校验 ───────────────────────────────────────────────────

def _episode_video_files(task_dir, n):
    """episode N 的全部视频文件路径列表（任意 key/扩展名）。"""
    _, fidx = episode_chunk_file(n)
    out = []
    vroot = os.path.join(task_dir, "videos")
    if os.path.isdir(vroot):
        for cname in os.listdir(vroot):
            cdir = os.path.join(vroot, cname)
            if not os.path.isdir(cdir):
                continue
            for key in os.listdir(cdir):
                kd = os.path.join(cdir, key)
                if not os.path.isdir(kd):
                    continue
                for fn in os.listdir(kd):
                    # v1.1.2 起 episode- 前缀；file- 前缀为更早数据（容忍）
                    if (fn.startswith(f"episode-{fidx:03d}.")
                            or fn.startswith(f"file-{fidx:03d}.")):
                        out.append(os.path.join(kd, fn))
    return out


def validate_task(task_dir, task):
    """读取端自检：episodes 行 ↔ data parquet ↔ 视频文件组 ↔ info.json。"""
    print(f"── 校验 {task} ──")
    import pyarrow.parquet as pq
    ok = True
    info = load_json(pooled_info_path(task_dir), {})
    if info.get("format") != "pooled_episodes_v1":
        print("  ✗ info.json 缺失或非池化格式")
        return False
    episodes = []
    dupes = set()
    ep_dir = os.path.join(task_dir, "meta", "episodes")
    if not os.path.isdir(ep_dir):
        print("  ✗ meta/episodes 缺失")
        return False
    for c in sorted(os.listdir(ep_dir)):
        if not c.startswith("chunk-"):
            continue
        d = os.path.join(ep_dir, c)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".parquet"):
                continue
            p = os.path.join(d, f)
            rows = pq.read_table(p).to_pylist()
            # 每段文件不变量：单行且 episode-{f}（旧数据 file-{f}）与
            # episode_index 同编号
            if len(rows) == 1:
                n = int(rows[0]["episode_index"])
                fm = re.match(r"^(?:episode|file)-(\d+)\.parquet$", f)
                fidx = int(fm.group(1)) if fm else -1
                if fidx != (n - 1) % POOLED_CHUNK_SIZE:
                    print(f"  ✗ {os.path.relpath(p, task_dir)}: "
                          f"episode-{fidx:03d} 与 episode {n} 编号不符")
                    ok = False
            for r in rows:
                n = int(r["episode_index"])
                if any(int(x["episode_index"]) == n for x in episodes):
                    dupes.add(n)
                episodes.append(r)
    if dupes:
        print(f"  ✗ episode 行重复（新旧布局并存）: {sorted(dupes)}")
        ok = False
    print(f"  episodes 行: {len(episodes)}")
    for r in episodes:
        n = int(r["episode_index"])
        data = pooled_data_parquet_path(task_dir, n)
        if not os.path.isfile(data):
            print(f"  ✗ episode {n}: 缺 data parquet")
            ok = False
            continue
        t = pq.read_table(data)
        cols = set(t.column_names)
        need = {"episode_index", "frame_index", "timestamp", "task_index",
                "wall_time", "hardware_ns", "action"}
        miss = need - cols
        if miss:
            print(f"  ✗ episode {n}: 缺列 {sorted(miss)}")
            ok = False
        else:
            print(f"  ✓ episode {n}: {t.num_rows} 行, 列 "
                  f"{sorted(c for c in cols if c.startswith('observation') or c.startswith('status'))}")
        # 视频：episode 自己的文件组（key 可跨会话改名，不能要求
        # 每个 key 每个 episode 都有文件）
        if not _episode_video_files(task_dir, n):
            print(f"  ✗ episode {n}: 无任何视频文件")
            ok = False
    # 每个声明的 key 至少在某个 episode 有文件（任务级并集口径）
    for key, ext in (info.get("video_extensions") or {}).items():
        if not any(os.path.isfile(pooled_video_path(
                task_dir, key, int(r["episode_index"]), ext=ext))
                for r in episodes):
            print(f"  ✗ 视频 key {key} ({ext}) 全任务无文件")
            ok = False
    if os.path.isfile(pooled_stats_path(task_dir)):
        print(f"  ✓ stats.json 存在: {sorted(load_json(pooled_stats_path(task_dir), {}))}")
    return ok


def split_episodes(task_dir, task, dry_run):
    """旧分片（每 chunk 一个多行文件）→ 每段一个文件；幂等。

    - 多行文件按 episode_index 拆成每段单行文件（每段文件已存在
      则跳过，不覆盖——每段文件为准）。
    - v1.1.2 起每段文件前缀 episode-，与旧分片名 file-{chunk} 不再
      重名（v1.1.1 及更早两者同路径，写回源路径特判已无必要但保留）。
    - 拆完删除旧分片；单行文件不动。
    - 拆分过则刷新 info.json episodes_path 模板（旧任务写的是旧模板）。
    """
    ep_dir = os.path.join(task_dir, "meta", "episodes")
    if not os.path.isdir(ep_dir):
        print(f"  {task}: 无 meta/episodes，跳过")
        return
    changed = False
    for c in sorted(os.listdir(ep_dir)):
        if not c.startswith("chunk-"):
            continue
        d = os.path.join(ep_dir, c)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".parquet"):
                continue
            p = os.path.join(d, fn)
            rows = _read_episode_rows(p)
            if len(rows) <= 1:
                continue  # 每段新布局或只剩一行，不动
            dests = {}
            for r in rows:
                n = int(r.get("episode_index", 0))
                if n:
                    dests[pooled_episodes_path(task_dir, n)] = r
            wrote = 0
            for dst, r in dests.items():
                if dst == p:
                    continue  # 源路径自身最后写
                if os.path.isfile(dst):
                    continue  # 每段文件已存在（幂等，不覆盖）
                if dry_run:
                    wrote += 1
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                _atomic_write_parquet(_episode_rows_table([r]), dst)
                wrote += 1
            if dry_run:
                print(f"  [plan] {task}: {os.path.relpath(p, task_dir)} "
                      f"{len(rows)} 行 → {len(dests)} 个每段文件")
                continue
            if p in dests:
                # chunk-0：N=1 的行写回源路径（分片变成每段文件）
                _atomic_write_parquet(_episode_rows_table([dests[p]]), p)
            else:
                os.remove(p)
            changed = True
            print(f"  {task}: {os.path.relpath(p, task_dir)} {len(rows)} 行 "
                  f"→ {len(dests)} 个每段文件")
    if changed and not dry_run:
        info_path = pooled_info_path(task_dir)
        info = load_json(info_path, {})
        if isinstance(info, dict):
            info["episodes_path"] = ("meta/episodes/chunk-{c:03d}/"
                                     "episode-{f:03d}.parquet")
            _atomic_write_json(info_path, info)
    if dry_run:
        print(f"  [plan] {task}: 以上为计划，未落盘")


# ── 主流程 ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="旧会话 → v1.1.0 任务池化布局")
    ap.add_argument("--task", action="append", default=[],
                    help="只处理指定任务目录名（可多次）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印迁移计划，不落盘")
    ap.add_argument("--backup", action="store_true", default=True,
                    help="旧会话目录移入备份区（默认开）")
    ap.add_argument("--no-backup", dest="backup", action="store_false",
                    help="旧会话目录不备份（不推荐）")
    ap.add_argument("--force", action="store_true",
                    help="任务已有 manifest 时仍处理剩余旧会话")
    ap.add_argument("--restore", action="store_true",
                    help="按 manifest 回滚迁移")
    ap.add_argument("--validate", action="store_true",
                    help="读取端自检（迁移后校验报告）")
    ap.add_argument("--split-episodes", action="store_true",
                    help="旧 episodes 分片（每 chunk 多行）拆成每段一文件（幂等）")
    ap.add_argument("--recordings-root", default=None,
                    help="测试用：覆盖录制根目录（默认 data/recordings）")
    ap.add_argument("--db-path", default=None,
                    help="测试用：覆盖 pipeline.db 路径")
    args = ap.parse_args()
    global RECORDINGS_DIR, DB_PATH
    if args.recordings_root:
        RECORDINGS_DIR = args.recordings_root
    if args.db_path:
        DB_PATH = args.db_path

    report = Report()
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(RECORDINGS_DIR,
                               f"_migrate_backup_{run_ts}")
    db_backup_path = os.path.join(
        settings.DATA_DIR, f"pipeline.db.migrate_backup_{run_ts}")
    db_needs_backup = False

    if args.restore:
        for task, task_dir in task_dirs(RECORDINGS_DIR, args.task):
            manifest = load_manifest(task_dir)
            if manifest.get("sessions"):
                restore_task(task_dir, task, manifest, args.dry_run, report)
        report.print("restore")
        return 0

    if args.validate:
        all_ok = True
        for task, task_dir in task_dirs(RECORDINGS_DIR, args.task):
            if os.path.isdir(os.path.join(task_dir, "meta", "episodes")):
                all_ok = validate_task(task_dir, task) and all_ok
        print()
        print("PASS: 校验全部通过" if all_ok
              else "FAIL: 存在校验失败项")
        return 0 if all_ok else 1

    if args.split_episodes:
        for task, task_dir in task_dirs(RECORDINGS_DIR, args.task):
            split_episodes(task_dir, task, args.dry_run)
        print()
        print("PASS: 旧分片拆分完成（幂等，可重复跑）")
        return 0

    for task, task_dir in task_dirs(RECORDINGS_DIR, args.task):
        sessions = legacy_sessions(task_dir, task)
        if not sessions:
            continue
        manifest = load_manifest(task_dir)
        if manifest.get("sessions") and not args.force:
            report.warnings.append(
                f"{task}: 已有 manifest（迁移过），--force 才会重扫")
            continue
        done = {os.path.basename(e["backup_path"])
                for e in (manifest.get("sessions") or [])
                if e.get("backup_path")}
        todo = [s for s in sessions
                if os.path.basename(s) not in done]
        if not todo:
            continue

        n_next = max(list_task_episodes(task_dir) or [0]) + 1
        entries = manifest.get("sessions") or []
        for session_dir in todo:
            db_rec, db_up = db_rows_for_path(
                DB_PATH, "file_path", session_dir)
            if db_rec or db_up:
                db_needs_backup = True
                if not args.dry_run and not os.path.isfile(db_backup_path):
                    shutil.copy2(DB_PATH, db_backup_path)
                    print(f"pipeline.db 已备份: {db_backup_path}")
            entry, warns = migrate_session(
                task_dir, task, session_dir, n_next, backup_root, report,
                args.dry_run, args.backup, DB_PATH)
            if entry == "skip":
                report.warnings.extend(warns)
                continue
            entries.append(entry)
            n_next += 1
        finalize_task(task_dir, task, entries, args.dry_run)
        manifest = {
            "version": MANIFEST_VERSION,
            "run_ts": run_ts,
            "db_backup": db_backup_path if db_needs_backup else None,
            "sessions": entries,
        }
        if not args.dry_run:
            save_manifest(task_dir, manifest)

    report.print("dry-run" if args.dry_run else "migrate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
