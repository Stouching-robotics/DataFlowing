#!/usr/bin/env python3
"""
数据 IO —— 会话元数据读取 + LeRobot 风格 parquet 打包/落盘。

schema 与 hand_triangulate.py 完全一致（保证 render_stereo 重放路径兼容），
新增一列 observation.keypoints.stage2（每手 bool，精修是否被采纳）。
打包逻辑按本模块数据结构（DetectedHand / RefinedPair）微调后内联于此，
避免跨模块 import 私有函数。
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

N_HANDS, N_KPTS = 2, 21
DIM_2D = N_HANDS * N_KPTS * 2        # 84
DIM_3D = N_HANDS * N_KPTS * 3        # 126

FEATURES_ADD = {
    "observation.keypoints.stereo_left":        {"dtype": "float32", "shape": [2, 21, 2]},
    "observation.keypoints.stereo_right":       {"dtype": "float32", "shape": [2, 21, 2]},
    "observation.keypoints.hand_3d":            {"dtype": "float32", "shape": [2, 21, 3]},
    "observation.keypoints.reprojection_error": {"dtype": "float32", "shape": [2]},
    "observation.keypoints.hand_0_present":     {"dtype": "bool", "shape": [1]},
    "observation.keypoints.hand_1_present":     {"dtype": "bool", "shape": [1]},
    "observation.keypoints.hand_0_label":       {"dtype": "string", "shape": [1]},
    "observation.keypoints.hand_1_label":       {"dtype": "string", "shape": [1]},
    "observation.keypoints.stage2":             {"dtype": "bool", "shape": [2]},
    "observation.keypoints.propagated":         {"dtype": "bool", "shape": [2]},
    "observation.keypoints.hand_3d_smoothed":   {"dtype": "float32", "shape": [2, 21, 3]},
}


# ── 会话元数据 ──────────────────────────────────────────────────

def find_video(session_path: str, cam: str) -> str | None:
    """定位左右目视频：videos/<cam>/chunk-0000/<cam>.mp4，回退 videos/<cam>.mp4。"""
    for p in (os.path.join(session_path, "videos", cam, "chunk-0000", f"{cam}.mp4"),
              os.path.join(session_path, "videos", f"{cam}.mp4")):
        if os.path.isfile(p):
            return p
    return None


def load_episode_meta(session_path: str) -> tuple:
    episode_index, task_index = 0, 0
    try:
        import pandas as pd
        ep = pd.read_parquet(os.path.join(session_path, "meta", "episodes",
                                          "chunk_000000.parquet"))
        episode_index = int(ep["episode_index"].iloc[0])
    except Exception:
        pass
    try:
        with open(os.path.join(session_path, "meta", "tasks.jsonl"), encoding="utf-8") as f:
            task_index = int(json.loads(f.readline())["task_index"])
    except Exception:
        pass
    return episode_index, task_index


def load_timestamps(session_path: str) -> dict:
    """timestamps.json → {frame_index: timestamp}（同帧多条取第一条）。"""
    try:
        with open(os.path.join(session_path, "timestamps.json"), encoding="utf-8") as f:
            entries = json.load(f)["timestamps"]
    except Exception:
        return {}
    out = {}
    for e in entries:
        fi = e.get("frame_index")
        if fi is not None and fi not in out:
            out[fi] = float(e.get("timestamp", 0.0))
    return out


# ── 数据打包 ──────────────────────────────────────────────────

def pack_2d(hands: list) -> list:
    """DetectedHand 列表 → 84 维 [2手×21点×(x,y)]，缺手全零。"""
    arr = np.zeros((N_HANDS, N_KPTS, 2), np.float32)
    for i, h in enumerate(hands[:N_HANDS]):
        arr[i] = np.asarray(h.landmarks, np.float32).reshape(N_KPTS, 2)
    return arr.flatten().tolist()


def pack_3d(pairs: list) -> list:
    """RefinedPair 列表 → 126 维 [2手×21点×(x,y,z)]，无效/缺手 NaN。"""
    arr = np.full((N_HANDS, N_KPTS, 3), np.nan, np.float32)
    for i, p in enumerate(pairs[:N_HANDS]):
        arr[i] = np.asarray(p.result.points_3d, np.float32).reshape(N_KPTS, 3)
    return arr.flatten().tolist()


def pack_errors(pairs: list) -> list:
    err = np.full(N_HANDS, np.nan, np.float32)
    for i, p in enumerate(pairs[:N_HANDS]):
        if p.result.valid_count:
            err[i] = p.result.mean_error
    return err.tolist()


def pack_stage2(pairs: list) -> list:
    used = np.zeros(N_HANDS, np.bool_)
    for i, p in enumerate(pairs[:N_HANDS]):
        used[i] = bool(p.used)
    return used.tolist()


# ── parquet 落盘 ──────────────────────────────────────────────

def write_parquet(rows: list, path: str, drop_keys: tuple = ()) -> str:
    """rows（dict 列表）→ parquet（zstd）。schema 与 hand_triangulate 一致 + stage2。

    drop_keys：跳过不存在的列（如 causal 模式不写 hand_3d_smoothed 时）。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {
        "episode_index": pa.array([r["episode_index"] for r in rows], pa.int64()),
        "frame_index": pa.array([r["frame_index"] for r in rows], pa.int64()),
        "timestamp": pa.array([r["timestamp"] for r in rows], pa.float32()),
        "task_index": pa.array([r["task_index"] for r in rows], pa.int64()),
        "observation.keypoints.stereo_left": pa.array(
            [r["observation.keypoints.stereo_left"] for r in rows],
            pa.list_(pa.float32(), DIM_2D)),
        "observation.keypoints.stereo_right": pa.array(
            [r["observation.keypoints.stereo_right"] for r in rows],
            pa.list_(pa.float32(), DIM_2D)),
        "observation.keypoints.hand_3d": pa.array(
            [r["observation.keypoints.hand_3d"] for r in rows],
            pa.list_(pa.float32(), DIM_3D)),
        "observation.keypoints.reprojection_error": pa.array(
            [r["observation.keypoints.reprojection_error"] for r in rows],
            pa.list_(pa.float32(), N_HANDS)),
        "observation.keypoints.hand_0_present": pa.array(
            [r["observation.keypoints.hand_0_present"] for r in rows], pa.bool_()),
        "observation.keypoints.hand_1_present": pa.array(
            [r["observation.keypoints.hand_1_present"] for r in rows], pa.bool_()),
        "observation.keypoints.hand_0_label": pa.array(
            [r["observation.keypoints.hand_0_label"] for r in rows], pa.string()),
        "observation.keypoints.hand_1_label": pa.array(
            [r["observation.keypoints.hand_1_label"] for r in rows], pa.string()),
        "observation.keypoints.stage2": pa.array(
            [r["observation.keypoints.stage2"] for r in rows], pa.list_(pa.bool_(), N_HANDS)),
        "observation.keypoints.propagated": pa.array(
            [r["observation.keypoints.propagated"] for r in rows], pa.list_(pa.bool_(), N_HANDS)),
        "action": pa.array([[0.0]] * len(rows), pa.list_(pa.float32(), 1)),
    }
    if "observation.keypoints.hand_3d_smoothed" not in drop_keys:
        cols["observation.keypoints.hand_3d_smoothed"] = pa.array(
            [r["observation.keypoints.hand_3d_smoothed"] for r in rows],
            pa.list_(pa.float32(), DIM_3D))
    schema = pa.schema([(k, v.type) for k, v in cols.items()])
    table = pa.table(cols, schema=schema)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


# ── meta 合并（--write-episode 时用；纯追加，不碰其他键）──────

def merge_info_json(session_path: str, drop_keys: tuple = ()):
    path = os.path.join(session_path, "meta", "info.json")
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    features = info.setdefault("features", {})
    features.update({k: v for k, v in FEATURES_ADD.items() if k not in drop_keys})
    for k in drop_keys:                          # 移除上次运行已写入的键
        features.pop(k, None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  ✓ meta/info.json 追加 features: {len(FEATURES_ADD)} 个键")


def merge_stats_json(session_path: str, rows: list):
    path = os.path.join(session_path, "meta", "stats.json")
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)

    def _feature_stats(rows, key, skip_if):
        vals = [np.asarray(r[key], dtype=np.float64) for r in rows if not skip_if(r)]
        if not vals:
            return None
        m = np.stack(vals)
        with np.errstate(all="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return {"mean": np.nanmean(m, axis=0).tolist(),
                        "std": np.nanstd(m, axis=0).tolist(),
                        "min": np.nanmin(m, axis=0).tolist(),
                        "max": np.nanmax(m, axis=0).tolist()}

    no_hands = lambda r: not (r["observation.keypoints.hand_0_present"]
                              or r["observation.keypoints.hand_1_present"])
    for key, skip in {
            "observation.keypoints.stereo_left": no_hands,
            "observation.keypoints.stereo_right": no_hands,
            "observation.keypoints.hand_3d": no_hands,
            "observation.keypoints.reprojection_error": no_hands,
            "action": lambda r: False}.items():
        s = _feature_stats(rows, key, skip)
        if s is not None:
            stats[key] = s
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ meta/stats.json 追加统计")
