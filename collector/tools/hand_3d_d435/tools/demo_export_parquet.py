#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑交付版 D435 demo 并把 2D/3D 关键点落成 parquet（不改交付件）。

交付版 `dist/d435_hands_demo_v1.1/d435_hands_demo.py` 只输出三路视频，
关键点用完即弃。本工具以**钩子**方式复用它的处理链——不复制管线代码，
也不改交付目录里的任何文件——在两个函数上挂钩抓取每帧数据：

    _SoftSmoother.update  → 入参 h3 = 槽位原始 3D，返回值 = 平滑 3D
    draw_overlay          → hands2d / labels / propagated / presents

两者在主循环同一次迭代内先后调用，凑齐一行；行 schema 复用
`tools/stereo_s80m/hand_3d/io.py` 的 write_parquet，与 run_pipeline_d435.py
产物同构，现有 probes 可直接读。

输出（沿用 keypoints_output/<tag>/<session>/ 约定）::

    <out_dir>/1_rgb_2d_overlay.mp4      demo 原三路视频
    <out_dir>/2_hand_3d.mp4
    <out_dir>/3_depth_colormap.mp4
    <out_dir>/hand_3d_refined/chunk-000.parquet   关键点

用法::

    venv/bin/python tools/hand_3d_d435/tools/demo_export_parquet.py <会话目录> \
        --out-dir keypoints_output/<tag>/<session>
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEMO_PY = os.path.join(_REPO, "dist", "d435_hands_demo_v1.1", "d435_hands_demo.py")
IO_PY = os.path.join(_REPO, "tools", "stereo_s80m", "hand_3d", "io.py")

N_HANDS, N_KPTS = 2, 21


def _load(name: str, path: str):
    """按文件路径加载模块（绕开 stereo_s80m.hand_3d 包 __init__ 的重依赖）。"""
    if not os.path.isfile(path):
        sys.exit(f"[错误] 找不到模块文件: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Collector:
    """逐帧攒行。_h3/_sm 由平滑钩子写入，draw_overlay 钩子收口成一行。"""

    def __init__(self, session: str, dio):
        self.rows: list = []
        self.ts = dio.load_timestamps(session)
        self.episode_index, self.task_index = dio.load_episode_meta(session)
        self._h3 = None      # (2,21,3) 槽位原始 3D（tracker 输出，软衔接前）
        self._sm = None      # (2,21,3) One-Euro 平滑后 3D

    def note_smooth(self, h3_in, sm_out) -> None:
        self._h3 = np.array(h3_in, np.float32).reshape(N_HANDS, N_KPTS, 3)
        self._sm = np.array(sm_out, np.float32).reshape(N_HANDS, N_KPTS, 3)

    def add(self, n: int, hands2d, labels, propagated, presents) -> None:
        nan3 = np.full((N_HANDS, N_KPTS, 3), np.nan, np.float32)
        h3 = self._h3 if self._h3 is not None else nan3
        sm = self._sm if self._sm is not None else nan3
        # 2D 缺手哨兵值按 io.py 约定为 0（3D 才用 NaN）
        kp2d = np.nan_to_num(
            np.asarray(hands2d, np.float32).reshape(N_HANDS, N_KPTS, 2),
            nan=0.0).reshape(-1).tolist()
        self.rows.append({
            "episode_index": self.episode_index,
            "frame_index": n,
            "timestamp": np.float32(self.ts.get(n, 0.0)),
            "task_index": self.task_index,
            "observation.keypoints.stereo_left": kp2d,
            # 单目 D435：无右目，写同一份 2D 保持 schema 兼容（与
            # run_pipeline_d435.py 的 _pack_2d_slots 处理一致）
            "observation.keypoints.stereo_right": kp2d,
            "observation.keypoints.hand_3d": h3.reshape(-1).tolist(),
            "observation.keypoints.hand_3d_smoothed": sm.reshape(-1).tolist(),
            # 单目无重投影概念；demo 不做 stage2 精修
            "observation.keypoints.reprojection_error": [float("nan")] * N_HANDS,
            "observation.keypoints.stage2": [False] * N_HANDS,
            "observation.keypoints.hand_0_present": bool(presents[0]),
            "observation.keypoints.hand_1_present": bool(presents[1]),
            "observation.keypoints.hand_0_label": labels[0] or "",
            "observation.keypoints.hand_1_label": labels[1] or "",
            "observation.keypoints.propagated": [bool(propagated[0]),
                                                 bool(propagated[1])],
        })
        self._h3 = self._sm = None


def install_hooks(demo, col: Collector) -> None:
    """在 demo 模块上挂钩子（只读旁路，不改变任何返回值/处理链）。"""
    orig_soft = demo._SoftSmoother.update

    def soft_update(self, h3, labels, valids):
        h3_raw = np.array(h3, np.float32)      # 拷贝：原函数会原地改 h3
        out = orig_soft(self, h3, labels, valids)
        col.note_smooth(h3_raw, out)
        return out

    demo._SoftSmoother.update = soft_update

    orig_draw = demo.draw_overlay

    def draw_overlay(rgb, hands2d, hands3d, labels, propagated, presents,
                     frame_idx, total, title="D435 3D hand keypoints"):
        col.add(frame_idx - 1, hands2d, labels, propagated, presents)
        return orig_draw(rgb, hands2d, hands3d, labels, propagated, presents,
                         frame_idx, total, title)

    demo.draw_overlay = draw_overlay


def main() -> None:
    ap = argparse.ArgumentParser(
        description="跑交付版 D435 demo 并导出 2D/3D 关键点 parquet")
    ap.add_argument("session_dir", help="主程序录制会话目录")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--det-conf", default="0.4")
    ap.add_argument("--track-conf", default="0.4")
    ap.add_argument("--fill", default="1")
    ap.add_argument("--propagate-max", default="15")
    ap.add_argument("--depth-overlay", action="store_true")
    args = ap.parse_args()

    session = args.session_dir.rstrip("/")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    demo = _load("d435_hands_demo", DEMO_PY)
    dio = _load("hand3d_io", IO_PY)

    col = Collector(session, dio)
    install_hooks(demo, col)

    argv = ["d435_hands_demo.py", session, "--out-dir", out_dir,
            "--det-conf", args.det_conf, "--track-conf", args.track_conf,
            "--fill", args.fill, "--propagate-max", args.propagate_max]
    if args.depth_overlay:
        argv.append("--depth-overlay")

    saved_argv = sys.argv
    sys.argv = argv
    try:
        demo.main()
    finally:
        sys.argv = saved_argv

    if not col.rows:
        sys.exit("[错误] 未采集到任何帧，parquet 未写出")

    pq_path = dio.write_parquet(
        col.rows, os.path.join(out_dir, "hand_3d_refined", "chunk-000.parquet"))
    size_mb = os.path.getsize(pq_path) / 1024 / 1024
    print(f"\n  ✓ 关键点 parquet: {pq_path}  ({size_mb:.2f} MB, "
          f"{len(col.rows)} 帧)")


if __name__ == "__main__":
    main()
