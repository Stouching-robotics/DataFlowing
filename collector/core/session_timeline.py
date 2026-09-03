"""
录制会话传感器时间线 —— 向量化合并 per-sensor parquet + 帧号二分查询。

替代 ui/playback_dialog.py 中逐行 dict 合并（60 分钟会话 179464 行实测
32.5s → 本实现约 1s），并提供 O(log n) 的帧号→传感器行查询。

纯 numpy/pyarrow 实现，无 Qt 依赖，可在后台线程运行。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# 合并键列（旧格式单表可能缺 episode_index，缺失时填 0）
_KEY_COLS = ("episode_index", "frame_index", "timestamp")


# ═══════════════════════════════════════════════════════
#  parquet 文件定位
# ═══════════════════════════════════════════════════════

def _find_sensor_parquet(session_dir: str) -> str:
    """查找新结构下的传感器 parquet 文件路径。

    新结构: data/<sensor>/chunk-0000/chunk_000000.parquet
    返回第一个找到的路径，无则返回空串。
    """
    data_dir = os.path.join(session_dir, "data")
    if os.path.isdir(data_dir):
        for entry in sorted(os.listdir(data_dir)):
            parquet = os.path.join(data_dir, entry, "chunk-0000", "chunk_000000.parquet")
            if os.path.isfile(parquet):
                return parquet
    return ""


def _find_all_sensor_parquets(session_dir: str) -> list[str]:
    """查找新结构下所有 per-sensor parquet 文件路径。

    新结构: data/<sensor>/chunk-0000/chunk_000000.parquet
    每个传感器独立一个 parquet，只含自己的 observation.<name> 列。
    返回所有找到的路径列表（按字母排序）。
    """
    data_dir = os.path.join(session_dir, "data")
    paths = []
    if os.path.isdir(data_dir):
        for entry in sorted(os.listdir(data_dir)):
            parquet = os.path.join(data_dir, entry, "chunk-0000", "chunk_000000.parquet")
            if os.path.isfile(parquet):
                paths.append(parquet)
    return paths


def _column_to_matrix(col, n_rows: int) -> Optional[np.ndarray]:
    """pyarrow 列 → (n_rows, D) float32 矩阵（缺值 NaN）。

    fixed_size_list / 标量数组均展平为二维；转换失败（如变长 list）返回 None。
    """
    try:
        arr = col.combine_chunks()
        if pa.types.is_fixed_size_list(arr.type):
            child = arr.values.to_numpy(zero_copy_only=False)
        else:
            child = arr.to_numpy(zero_copy_only=False)
        m = np.asarray(child)
        if m.ndim == 1:
            m = m.reshape(n_rows, -1)
        return m.astype(np.float32, copy=False)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
#  时间线
# ═══════════════════════════════════════════════════════

class SensorTimeline:
    """合并后的统一时间线（按 (episode_index, frame_index) 排序）。

    Attributes:
        frame_indices : (N,) int64    每行对应的视频帧号（非降，可二分）
        timestamps    : (N,) float64  每行时间戳（秒，不保证严格单调——
                       录制暂停点存在负跳变，查询请用帧号）
        obs           : {col: (N,D) float32}  观测列矩阵（缺值 NaN）
        signal_mask   : (N,) bool     该行任一观测列有非零信号
        signal_count  : int           有信号行总数
    """

    def __init__(self, frame_indices, timestamps, obs, signal_mask):
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.obs = obs
        self.signal_mask = np.asarray(signal_mask, dtype=bool)
        self.signal_count = int(self.signal_mask.sum())
        # 每列非空行的 (fi 子数组, 行号子数组) —— nearest_for_column 二分用
        self._col_index = {}
        for col, V in obs.items():
            valid = ~np.isnan(V).all(axis=1)
            rows = np.flatnonzero(valid)
            if rows.size:
                self._col_index[col] = (self.frame_indices[rows], rows)

    def __bool__(self) -> bool:
        return len(self.frame_indices) > 0

    def __len__(self) -> int:
        return len(self.frame_indices)

    def nearest_for_column(self, col: str, frame_idx: int
                           ) -> tuple[Optional[int], Optional[int]]:
        """返回该列中 frame_index 最接近 frame_idx 的 (行号, |Δ帧|)。

        列不存在或无有效行时返回 (None, None)。O(log n) 二分。
        假设：传感器 frame_index 与视频帧号同源同帧率空间
        （加载时已按 fi 排序保证非降；若未来出现不同源数据，
        在 load_timeline 处回退为 fi→ts 区间线性扫描）。
        """
        sub = self._col_index.get(col)
        if sub is None:
            return None, None
        fi_sub, rows_sub = sub
        i = int(np.searchsorted(fi_sub, frame_idx, side="left"))
        cands = []
        if i > 0:
            cands.append(i - 1)
        if i < len(fi_sub):
            cands.append(i)
        if not cands:
            return None, None
        best = min(cands, key=lambda k: abs(int(fi_sub[k]) - int(frame_idx)))
        return int(rows_sub[best]), int(abs(int(fi_sub[best]) - int(frame_idx)))

    def nearest_for_column_time(self, col: str, t_s: float
                                ) -> tuple[Optional[int], Optional[float]]:
        """返回该列中 timestamp 最接近 t_s（秒）的 (行号, |Δt|秒)。

        列不存在或无有效行时返回 (None, None)。
        时间戳单调（无暂停）时 O(log n) 二分；含暂停负跳变时退化为
        向量化 argmin（该列有效行上的最小 |Δt|）。

        多帧率回放用：主时钟时间 → 传感器时间线（传感器行按录制
        30fps 循环时间戳，与视频主时钟帧率可能不同源）。
        """
        sub = self._col_index.get(col)
        if sub is None:
            return None, None
        _, rows_sub = sub
        ts_sub = self.timestamps[rows_sub]
        if ts_sub.size == 0:
            return None, None
        if bool(np.all(np.diff(ts_sub) >= 0)):
            i = int(np.searchsorted(ts_sub, t_s, side="left"))
            cands = []
            if i > 0:
                cands.append(i - 1)
            if i < len(ts_sub):
                cands.append(i)
            if not cands:
                return None, None
            best = min(cands, key=lambda k: abs(float(ts_sub[k]) - t_s))
        else:
            best = int(np.argmin(np.abs(ts_sub - t_s)))
        return int(rows_sub[best]), float(abs(float(ts_sub[best]) - t_s))


# ═══════════════════════════════════════════════════════
#  加载与合并
# ═══════════════════════════════════════════════════════

def load_timeline(session_dir: str, sensor_names: list[str],
                  episode_index: int = 0) -> SensorTimeline:
    """加载会话传感器时间线。

    episode_index > 0（池化布局）→ 直接读本 episode 的单个 data parquet
    （data/chunk-{c:03d}/episode-{f:03d}.parquet）；
    episode_index == 0 → 旧格式双路径（per-sensor 多文件或单表回退）。
    """
    if episode_index > 0:
        from core.helpers import pooled_data_parquet_path
        tp = pooled_data_parquet_path(session_dir, episode_index)
        paths = [tp] if os.path.isfile(tp) else []
    else:
        paths = _find_all_sensor_parquets(session_dir)
        if not paths:
            # 旧格式回退：单个 data parquet
            from core.helpers import data_parquet_path
            tp = data_parquet_path(session_dir)
            if not os.path.isfile(tp):
                tp = os.path.join(session_dir, "data", "chunk_000000.parquet")
            if os.path.isfile(tp):
                paths = [tp]
    if not paths:
        return SensorTimeline([], [], {}, [])
    return _merge_per_sensor_parquets(paths, sensor_names)


def _merge_per_sensor_parquets(paths: list[str],
                               sensor_names: list[str]) -> SensorTimeline:
    """向量化合并多个 per-sensor parquet 为统一时间线。

    语义与旧逐行 dict 合并一致（已对真实会话逐项验证）：
    - 按 (episode_index, frame_index, round6(timestamp)) 分组合并
    - 每组输出首行的键值；观测列取组内最后一个非空值
    - 信号掩码 = 任一观测列该行 |值|.sum > 0
    """
    obs_cols = [f"observation.{sn}" for sn in sensor_names]

    ei_parts, fi_parts, ts_parts = [], [], []
    ranges = []                  # 每文件在全量行空间中的 (start, end)
    col_blocks = {c: [] for c in obs_cols}   # col → [(start, matrix)]

    pos = 0
    for p in paths:
        schema = pq.read_schema(p)
        keys = [c for c in _KEY_COLS if schema.get_field_index(c) >= 0]
        if "frame_index" not in keys:
            raise ValueError(f"parquet 缺少 frame_index 列: {p}")
        have_obs = [c for c in obs_cols if schema.get_field_index(c) >= 0]
        tbl = pq.read_table(p, columns=keys + have_obs)
        n = len(tbl)

        ei = (tbl.column("episode_index").to_numpy(zero_copy_only=False)
              if "episode_index" in keys else np.zeros(n, np.int64))
        fi = tbl.column("frame_index").to_numpy(zero_copy_only=False)
        ts = (tbl.column("timestamp").to_numpy(zero_copy_only=False)
              if "timestamp" in keys else np.zeros(n, np.float64))
        ei_parts.append(np.asarray(ei, np.int64))
        fi_parts.append(np.asarray(fi, np.int64))
        ts_parts.append(np.asarray(ts, np.float64))

        for c in have_obs:
            m = _column_to_matrix(tbl.column(c), n)
            if m is not None:
                col_blocks[c].append((pos, m))
        ranges.append((pos, pos + n))
        pos += n

    N = pos
    ei = np.concatenate(ei_parts)
    fi = np.concatenate(fi_parts)
    ts = np.concatenate(ts_parts)

    # 分组排序（ts6 只作组键，输出时间戳保留原始值）
    ts6 = np.round(ts, 6)
    order = np.lexsort((ts6, fi, ei))
    ei_s, fi_s, ts6_s = ei[order], fi[order], ts6[order]

    new = np.empty(N, dtype=bool)
    new[0] = True
    if N > 1:
        np.not_equal(ei_s[1:], ei_s[:-1], out=new[1:])
        new[1:] |= (fi_s[1:] != fi_s[:-1]) | (ts6_s[1:] != ts6_s[:-1])
    gstart = np.flatnonzero(new)
    gcount = len(gstart)

    obs = {}
    signal_mask = np.zeros(gcount, dtype=bool)
    for c in obs_cols:
        blocks = col_blocks.get(c)
        if not blocks:
            continue
        D = blocks[0][1].shape[1]
        V = np.full((N, D), np.nan, dtype=np.float32)
        for start, m in blocks:
            V[start:start + m.shape[0]] = m
        V = V[order]

        # 组内最后一个非空值（组内全缺值时置 NaN，不能取 V[-1]）
        valid = ~np.isnan(V).all(axis=1)
        last_valid = np.maximum.reduceat(
            np.where(valid, np.arange(N, dtype=np.int64), -1), gstart)
        idx = np.maximum(last_valid, 0)
        col_mat = np.where(last_valid[:, None] >= 0, V[idx], np.nan)
        obs[c] = col_mat
        signal_mask |= (np.abs(col_mat).sum(axis=1) > 0)

    return SensorTimeline(fi_s[gstart], ts[order][gstart], obs, signal_mask)
