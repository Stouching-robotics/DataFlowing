#!/usr/bin/env python3
"""
S80M 双目手部关键点后处理 —— 独立模块（不动 ui/ 主程序）。

左右目视频各自跑 MediaPipe 2D 检测 → 出厂标定矫正 → 三角化 → 物理 3D 关键点，
按 LeRobot 风格并入 episode data/ parquet；另产出可视化视频。

用法::

    python stereo_s80m/hand_triangulate.py <session_dir> [--calib PATH]
        [--max-err 8.0] [--max-depth 3.0] [--no-video] [--every N]

输入::

    <session>/videos/stereo_left/chunk-0000/stereo_left.mp4
    <session>/videos/stereo_right/chunk-0000/stereo_right.mp4

标定查找链（stereo_triangulate.load_stereo_calibration）::

    --calib 指定 → <session>/calibration/head_stereo.json（需可用）
    → <repo>/config/s80m_stereo_calibration.json（设备级，M1 生成）

输出::

    <session>/data/keypoints/chunk-0000/chunk_000000.parquet   关键点数据（LeRobot 风格列，2D+3D 全量）
    <session>/meta/info.json                                   追加 features 注册
    <session>/meta/stats.json                                  追加统计（新增特征）
    <repo>/keypoints_output/<tag>/<session>/stereo_triangulate.mp4   可视化视频（左右矫正图并排）
    <repo>/keypoints_output/<tag>/<session>/hand_3d/chunk-000.parquet   3D 关键点副本
    <repo>/keypoints_output/<tag>/<session>/hand_2d/chunk-000.parquet   2D 关键点副本

parquet 列::

    episode_index / frame_index / timestamp / task_index
    observation.keypoints.stereo_left           list<float32> 84 = 2手×21点×(x,y)   左目 2D
    observation.keypoints.stereo_right          list<float32> 84                       右目 2D
    observation.keypoints.hand_3d               list<float32> 126 = 2手×21点×(x,y,z)  物理 3D(米)
    observation.keypoints.reprojection_error    list<float32> 2                       每手平均重投影误差(px)
    observation.keypoints.hand_0/1_present      bool
    observation.keypoints.hand_0/1_label        string      MediaPipe 解剖左右手标签
    action                                      list<float32> 1   （占位 0.0，与 info.json 注册一致）

hand_3d 无效点存 NaN（超出重投影误差/深度范围或未检测到），2D 未检测到的手全零。

不修改任何现有文件的内容：只新增 data/keypoints/、追加 info.json features 与 stats.json 键。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
sys.path.insert(0, _TOOLS_DIR)      # hand_detection/stereo_s80m 已并入 tools/ 命名空间

from hand_detection.hand_pipeline_mediapipe import MediaPipeHandPipeline  # noqa: E402
from stereo_s80m.render_stereo import (                                  # noqa: E402
    overlay_view,
    create_video_writer,
    finalize_video,
)
from stereo_s80m.stereo_triangulate import (                              # noqa: E402
    StereoTriangulator,
    load_stereo_calibration,
    match_hands,
    DEFAULT_MAX_REPROJ_ERR,
    DEFAULT_MAX_DEPTH,
)

# ── 常量 ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(_REPO_ROOT, "tools", "models", "hand_landmarker.task")
N_HANDS = 2
N_KPTS = 21
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
}


# ── 会话元数据读取 ──────────────────────────────────────────────

def _find_video(session_path: str, cam: str) -> str:
    """定位左右目视频：videos/<cam>/chunk-0000/<cam>.mp4，回退 videos/<cam>.mp4。"""
    for p in (os.path.join(session_path, "videos", cam, "chunk-0000", f"{cam}.mp4"),
              os.path.join(session_path, "videos", f"{cam}.mp4")):
        if os.path.isfile(p):
            return p
    return None


def _load_episode_meta(session_path: str) -> tuple:
    episode_index, task_index = 0, 0
    try:
        import pandas as pd
        ep = pd.read_parquet(os.path.join(session_path, "meta", "episodes", "chunk_000000.parquet"))
        episode_index = int(ep["episode_index"].iloc[0])
    except Exception:
        pass
    try:
        with open(os.path.join(session_path, "meta", "tasks.jsonl"), encoding="utf-8") as f:
            task_index = int(json.loads(f.readline())["task_index"])
    except Exception:
        pass
    return episode_index, task_index


def _load_timestamps(session_path: str) -> dict:
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


# ── 数据打包 ────────────────────────────────────────────────────

def _pack_2d(hands) -> list:
    """手列表 → 84 维 [2手×21点×(x,y)]，缺手全零。"""
    arr = np.zeros((N_HANDS, N_KPTS, 2), np.float32)
    for i, h in enumerate(hands[:N_HANDS]):
        arr[i] = h.landmarks[:N_KPTS]
    return arr.flatten().tolist()


def _pack_3d(pairs) -> list:
    """匹配结果 → 126 维 [2手×21点×(x,y,z)]，无效/缺手 NaN。"""
    arr = np.full((N_HANDS, N_KPTS, 3), np.nan, np.float32)
    for i, p in enumerate(pairs[:N_HANDS]):
        arr[i] = p.result.points_3d[:N_KPTS]
    return arr.flatten().tolist()


def _pack_errors(pairs) -> list:
    err = np.full(N_HANDS, np.nan, np.float32)
    for i, p in enumerate(pairs[:N_HANDS]):
        err[i] = p.result.mean_error if p.result.valid_count else np.nan
    return err.tolist()


# 渲染（骨架叠加 + 视频输出）已在独立模块 stereo_s80m/render_stereo.py


# ── 左右目方向自检（stereo_swap_lr 隐患）────────────────────────

def _detect_orientation(pipe_l, pipe_r, vc_l, vc_r,
                        tri_normal: StereoTriangulator, tri_swapped: StereoTriangulator) -> bool:
    """用前几帧判断 stereo_left.mp4 对应标定 cam0 还是 cam1。

    返回 True = 需要 swap（即 stereo_left 文件实际是 cam1 的画面）。
    判据：两种配对各自三角化的平均重投影误差，误差小的配对几何自洽。
    配对错误时重投影误差大 → 有效点不足 → 天然被过滤。
    """
    err_n, err_s = [], []
    for _ in range(10):
        ok_l, fl = vc_l.read()
        ok_r, fr = vc_r.read()
        if not (ok_l and ok_r):
            break
        hl = pipe_l.process(fl).hands
        hr = pipe_r.process(fr).hands
        if not hl or not hr:
            continue
        for lh in hl[:1]:
            for rh in hr[:1]:
                rn = tri_normal.triangulate(lh.landmarks, rh.landmarks)
                if rn.valid_count >= 8:
                    err_n.append(rn.mean_error)
                rs = tri_swapped.triangulate(lh.landmarks, rh.landmarks)
                if rs.valid_count >= 8:
                    err_s.append(rs.mean_error)
        if len(err_n) >= 2 and len(err_s) >= 2:
            break
    if not err_s:
        return False
    if not err_n:
        return True
    print(f"  [方向自检] 常规配对平均误差 {np.mean(err_n):.2f}px (n={len(err_n)}), "
          f"交换配对 {np.mean(err_s):.2f}px (n={len(err_s)})")
    return np.mean(err_s) + 0.5 < np.mean(err_n)


# ── meta 合并（纯追加，不碰其他字段）────────────────────────────

def _merge_info_json(session_path: str):
    path = os.path.join(session_path, "meta", "info.json")
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    features = info.setdefault("features", {})
    features.update(FEATURES_ADD)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  ✓ meta/info.json 追加 features: {len(FEATURES_ADD)} 个键")


def _feature_stats(rows: list, key: str, skip_if) -> dict | None:
    """按行收集列值 → {mean, std, min, max}（逐元素，NaN 忽略）。skip_if 过滤掉该行。"""
    vals = [np.asarray(r[key], dtype=np.float64) for r in rows if not skip_if(r)]
    if not vals:
        return None
    m = np.stack(vals)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # 全 NaN 列 (缺手位)
            out = {"mean": np.nanmean(m, axis=0).tolist(),
                   "std": np.nanstd(m, axis=0).tolist(),
                   "min": np.nanmin(m, axis=0).tolist(),
                   "max": np.nanmax(m, axis=0).tolist()}
    return out


def _merge_stats_json(session_path: str, rows: list):
    path = os.path.join(session_path, "meta", "stats.json")
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)

    no_hands = lambda r: not (r["observation.keypoints.hand_0_present"]
                              or r["observation.keypoints.hand_1_present"])
    specs = {
        "observation.keypoints.stereo_left": no_hands,
        "observation.keypoints.stereo_right": no_hands,
        "observation.keypoints.hand_3d": no_hands,
        "observation.keypoints.reprojection_error": no_hands,
        "action": lambda r: False,
    }
    for key, skip in specs.items():
        s = _feature_stats(rows, key, skip)
        if s is not None:
            stats[key] = s
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ meta/stats.json 追加统计: {len(specs)} 个特征")


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="S80M 双目手部关键点后处理（独立模块，不动主程序）")
    ap.add_argument("session_dir", help="episode 目录，如 data/recordings/222/222_000002")
    ap.add_argument("--calib", default=None, help="指定标定 JSON（默认走查找链）")
    ap.add_argument("--max-err", type=float, default=DEFAULT_MAX_REPROJ_ERR, help="重投影误差阈值 px")
    ap.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH, help="最大深度 m")
    ap.add_argument("--every", type=int, default=1, help="每隔 N 帧处理一次（快速测试用）")
    ap.add_argument("--no-video", action="store_true", help="不生成可视化视频")
    # One-Euro 平滑参数（关键点跟手性）：freq_min 越大越跟手（但越抖），
    # beta 越大快速运动响应越快；--no-smooth 完全关平滑（最跟手、最抖）
    ap.add_argument("--freq-min", type=float, default=5.0,
                    help="One-Euro 位置滤波最低截止频率 Hz（默认 5.0；跟手不足就调大，如 15）")
    ap.add_argument("--beta", type=float, default=0.05,
                    help="One-Euro 速度自适应系数（默认 0.05；快速运动滞后就调大，如 0.6）")
    ap.add_argument("--no-smooth", action="store_true",
                    help="关闭 One-Euro 平滑（最跟手，但抖动全部保留）")
    args = ap.parse_args()

    session = os.path.abspath(args.session_dir)
    if not os.path.isdir(session):
        print(f"[ERROR] 会话目录不存在: {session}")
        sys.exit(1)

    vp_l = _find_video(session, "stereo_left")
    vp_r = _find_video(session, "stereo_right")
    if not vp_l or not vp_r:
        print(f"[ERROR] 找不到双目视频: stereo_left={vp_l} stereo_right={vp_r}")
        sys.exit(1)

    calib = load_stereo_calibration(session, args.calib)
    if calib is None:
        print("[ERROR] 无可用的立体标定（查找链: episode calibration/ → config/s80m_stereo_calibration.json）")
        print("        先运行: python stereo_s80m/capture_calibration.py")
        sys.exit(1)
    calib_src = args.calib or "episode/device 查找链"
    print(f"标定: 来自 {calib_src}  baseline={calib['baseline']*1000:.1f}mm "
          f"模型={calib['left_camera']['distortion_model']}")

    episode_index, task_index = _load_episode_meta(session)
    timestamps = _load_timestamps(session)
    print(f"episode_index={episode_index} task_index={task_index} "
          f"时间戳帧数={len(timestamps)}")

    # ── 视频输入 ────────────────────────────────────────────
    vc_l = cv2.VideoCapture(vp_l)
    vc_r = cv2.VideoCapture(vp_r)
    n_total = int(vc_l.get(cv2.CAP_PROP_FRAME_COUNT))
    n_r = int(vc_r.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total != n_r:
        print(f"[WARN] 左右目帧数不一致: {n_total} vs {n_r}，将处理到 min")
    n_total = min(n_total, n_r)
    w, h = int(vc_l.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vc_l.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"视频: {vp_l} + {vp_r}  {n_total} 帧  {w}x{h}@{fps}")

    # ── 检测管线：每视角独立实例（One-Euro 平滑状态互不串扰）─
    # 平滑强度由 CLI 控制：freq_min/beta 越大越跟手（--no-smooth 完全关掉）
    pipe_l = MediaPipeHandPipeline(model_path=MODEL_PATH, num_hands=N_HANDS,
                                   mirror=False, smooth=not args.no_smooth,
                                   freq_min=args.freq_min, beta=args.beta,
                                   dcutoff=1.0)
    pipe_r = MediaPipeHandPipeline(model_path=MODEL_PATH, num_hands=N_HANDS,
                                   mirror=False, smooth=not args.no_smooth,
                                   freq_min=args.freq_min, beta=args.beta,
                                   dcutoff=1.0)

    # ── 左右目方向自检（stereo_swap_lr 配置与历史记录冲突隐患）─
    tri_normal = StereoTriangulator(calib)
    tri_swapped = StereoTriangulator(calib, swap_cams=True)
    use_swap = _detect_orientation(pipe_l, pipe_r, vc_l, vc_r, tri_normal, tri_swapped)
    if use_swap:
        print("  [方向自检] stereo_left 文件对应标定 cam1 → 交换配对后三角化")
    else:
        print("  [方向自检] stereo_left 文件对应标定 cam0（常规配对）")
    vc_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vc_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tri = tri_swapped if use_swap else tri_normal
    print(f"三角化器: {tri.summarize()}")

    # ── 输出（keypoints_output/<tag>/<session>/ 镜像目录）────
    # 视频 + hand_3d 数据副本都落在同一镜像目录下
    tag, name = os.path.basename(os.path.dirname(session)), os.path.basename(session)
    out_kpts_dir = os.path.join(_REPO_ROOT, "keypoints_output", tag, name)
    os.makedirs(out_kpts_dir, exist_ok=True)

    writer, tmp_video, out_video = None, None, None
    if not args.no_video:
        out_video = os.path.join(out_kpts_dir, "stereo_triangulate.mp4")
        writer, tmp_video = create_video_writer(out_video, fps, w * 2, h)

    # ── 主循环 ──────────────────────────────────────────────
    rows = []
    t0 = time.time()
    n = 0
    n_matched = n_two = 0
    errs_all = []
    while True:
        ok_l, fl = vc_l.read()
        ok_r, fr = vc_r.read()
        if not (ok_l and ok_r):
            break
        if args.every > 1 and n % args.every != 0:
            n += 1
            continue

        hl = pipe_l.process(fl).hands
        hr = pipe_r.process(fr).hands
        pairs = match_hands(hl, hr, tri, args.max_err, args.max_depth)

        present0 = len(pairs) > 0
        present1 = len(pairs) > 1
        if present0:
            n_matched += 1
            errs_all.append(pairs[0].result.mean_error)
        if present1:
            n_two += 1

        label0 = pairs[0].left_label if present0 else ""
        label1 = pairs[1].left_label if present1 else ""
        rows.append({
            "episode_index": episode_index,
            "frame_index": n,
            "timestamp": np.float32(timestamps.get(n, 0.0)),
            "task_index": task_index,
            "observation.keypoints.stereo_left": _pack_2d(hl),
            "observation.keypoints.stereo_right": _pack_2d(hr),
            "observation.keypoints.hand_3d": _pack_3d(pairs),
            "observation.keypoints.reprojection_error": _pack_errors(pairs),
            "observation.keypoints.hand_0_present": present0,
            "observation.keypoints.hand_1_present": present1,
            "observation.keypoints.hand_0_label": label0,
            "observation.keypoints.hand_1_label": label1,
            "action": [0.0],
        })

        if writer is not None:
            frame_l = tri.rectified_image(fl, "left")
            frame_r = tri.rectified_image(fr, "right")
            overlay_view(frame_l, pairs, tri, "left", n, n_total)
            overlay_view(frame_r, pairs, tri, "right", n, n_total)
            writer.write(cv2.hconcat([frame_l, frame_r]))

        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"  {n}/{n_total} 帧  ({n/el:.1f} fps, 检测+三角化)  "
                  f"pairs={n_matched}")
    vc_l.release()
    vc_r.release()

    if not rows:
        print("[ERROR] 未处理任何帧")
        sys.exit(1)

    # ── parquet 落盘 ────────────────────────────────────────
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
        "action": pa.array([[0.0]] * len(rows), pa.list_(pa.float32(), 1)),
    }
    schema = pa.schema([(k, v.type) for k, v in cols.items()])
    table = pa.table(cols, schema=schema)
    out_parquet = os.path.join(session, "data", "keypoints", "chunk-0000",
                               "chunk_000000.parquet")
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    pq.write_table(table, out_parquet, compression="zstd")
    print(f"✓ 关键点 parquet: {out_parquet}  ({len(rows)} 行)")

    # hand_3d / hand_2d 副本 → keypoints_output/<tag>/<session>/ 镜像目录
    # （episode data/ 那份保留全量；副本按数据类型拆分，与主程序 hand_pose/ 惯例一致）
    copy_3d = os.path.join(out_kpts_dir, "hand_3d", "chunk-000.parquet")
    os.makedirs(os.path.dirname(copy_3d), exist_ok=True)
    pq.write_table(table, copy_3d, compression="zstd")
    print(f"✓ hand_3d 副本: {copy_3d}")

    _HAND_2D_COLS = [
        "episode_index", "frame_index", "timestamp", "task_index",
        "observation.keypoints.stereo_left",
        "observation.keypoints.stereo_right",
        "observation.keypoints.hand_0_present",
        "observation.keypoints.hand_1_present",
        "observation.keypoints.hand_0_label",
        "observation.keypoints.hand_1_label",
    ]
    copy_2d = os.path.join(out_kpts_dir, "hand_2d", "chunk-000.parquet")
    os.makedirs(os.path.dirname(copy_2d), exist_ok=True)
    pq.write_table(table.select(_HAND_2D_COLS), copy_2d, compression="zstd")
    print(f"✓ hand_2d 副本: {copy_2d}")

    # ── meta 合并 ───────────────────────────────────────────
    _merge_info_json(session)
    _merge_stats_json(session, rows)

    # ── 视频转码 H.264（render_stereo 模块负责）──────────────
    if writer is not None:
        out_video = finalize_video(writer, tmp_video, out_video)
        print(f"✓ 可视化视频: {out_video}")

    # ── 汇总 ────────────────────────────────────────────────
    errs_all = np.array(errs_all)
    print("\n── 处理汇总 ──")
    print(f"  帧数: {len(rows)}   检测到手帧: {n_matched} ({n_matched/len(rows)*100:.0f}%)   "
          f"双手帧: {n_two}")
    print(f"  平均重投影误差: {errs_all.mean():.2f}px  p95: {np.percentile(errs_all, 95):.2f}px"
          f"  (阈值 {args.max_err}px)")
    zs = [np.array(r["observation.keypoints.hand_3d"], np.float32)[2::3] for r in rows]
    zs = np.concatenate(zs)
    zs = zs[np.isfinite(zs)]
    if zs.size:
        print(f"  物理 3D 深度: z ∈ [{zs.min():.3f}, {zs.max():.3f}] m  基线 {calib['baseline']*1000:.1f}mm")


if __name__ == "__main__":
    main()
