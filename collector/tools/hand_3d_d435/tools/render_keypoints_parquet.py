#!/usr/bin/env python3
"""render_keypoints_parquet.py —— 任意 21 点 3D 关键点 parquet → 3D 视角渲染视频。

给「没有 D435 相机、由外部管线解算 21 点 3D 关键点」的场景：只要数据装进
parquet（本仓库 io.pack_3d 同款列名，即 observation.keypoints.hand_3d /
hand_3d_smoothed，每手 21 点 × (x,y,z) 米、无效点 NaN），本工具直接吃，
用与 D435 管线相同的 RotatingSkeletonRenderer 出旋转视角视频或静态视角图。

对数据零相机依赖：不 import pyrealsense2 / mediapipe / 深度对齐。仅
numpy + cv2 + pyarrow（+ --smooth 时 scipy）。

用法（仓库 venv）:
    ./venv/bin/python tools/hand_3d_d435/tools/render_keypoints_parquet.py <parquet> [选项]

    # 旋转视角（默认转 2 圈）出视频
    python tools/hand_3d_d435/tools/render_keypoints_parquet.py chunk-000.parquet
    # 静态正面视角（yaw=180°）出视频
    python tools/hand_3d_d435/tools/render_keypoints_parquet.py chunk-000.parquet --view static
    # 单帧预览 PNG（不写视频）
    python tools/hand_3d_d435/tools/render_keypoints_parquet.py chunk-000.parquet --frame 100
    # 先零相位平滑再渲染（源数据无时域滤波时用，可明显压抖）
    python tools/hand_3d_d435/tools/render_keypoints_parquet.py chunk-000.parquet --smooth
    # 尺度修正（如源数据为真实尺度的 55%）+ 平移（腕点锁原点时可挪开）
    python tools/hand_3d_d435/tools/render_keypoints_parquet.py chunk-000.parquet \
        --scale 1.8 --shift 0.3,0.2,0.6
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from stereo_s80m.hand_3d.renderer_3d import RotatingSkeletonRenderer  # noqa: E402
from stereo_s80m.hand_3d.video_writer import create_video_sink       # noqa: E402

N_HANDS, N_KPTS = 2, 21
RENDER_SIZE = (1280, 720)
_COL_H3 = "observation.keypoints.hand_3d"
_COL_SM = "observation.keypoints.hand_3d_smoothed"
_NAMES = ("slot0", "slot1")


def _load(paths, col) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """读多个 parquet 拼帧 → ((N,2,21,3) float32, [(l0,l1), ...])。

    支持 (N,2,21,3) 与 (N,21,3) 两种展平（后者视为单手表、slot0 全 NaN）。
    """
    import pyarrow.parquet as pq

    mats, labels = [], []
    for p in paths:
        t = pq.read_table(p)
        names = t.column_names
        if col not in names:
            raise SystemExit(f"错误: {p} 无列 {col}。可用关键点列: "
                             f"{[c for c in names if 'keypoint' in c]}")
        flat = t[col].to_pylist()
        if not flat:
            raise SystemExit(f"错误: {p} 的 {col} 无数据")
        if len(flat[0]) == N_KPTS * 3:          # 单手 (N,21,3) → slot1
            arr = np.asarray(flat, np.float32).reshape(-1, 1, N_KPTS, 3)
            arr = np.concatenate([np.full_like(arr, np.nan), arr], axis=1)
        elif len(flat[0]) == N_HANDS * N_KPTS * 3:
            arr = np.asarray(flat, np.float32).reshape(-1, N_HANDS, N_KPTS, 3)
        else:
            raise SystemExit(f"错误: {p} 的 {col} 每行 {len(flat[0])} 维，"
                             f"非 2 手×21 点×3（126）或 1 手×21 点×3（63）")
        idx = None
        if "frame_index" in names:              # 乱序保险
            idx = np.argsort(t["frame_index"].to_pylist(), kind="stable")
            arr = arr[idx]
        mats.append(arr)
        # 逐帧 label（两槽）；缺失列填 ""
        labs0 = t["observation.keypoints.hand_0_label"].to_pylist() \
            if "observation.keypoints.hand_0_label" in names \
            else [""] * arr.shape[0]
        labs1 = t["observation.keypoints.hand_1_label"].to_pylist() \
            if "observation.keypoints.hand_1_label" in names \
            else [""] * arr.shape[0]
        if idx is not None:
            labs0, labs1 = [labs0[i] for i in idx], [labs1[i] for i in idx]
        labels += list(zip(labs0, labs1))
    return np.concatenate(mats, axis=0), labels


def _diagnose(h3: np.ndarray) -> None:
    """打印数据体检：手数、有效占比、腕点运动、尺度提示。"""
    n = h3.shape[0]
    print(f"── 数据体检（{n} 帧）──")
    for s, nm in enumerate(_NAMES):
        fin = np.isfinite(h3[:, s]).all(axis=-1)
        nf = int(fin.any(axis=-1).sum())
        print(f"  {nm}: {nf}/{n} 帧有效")
        if nf == 0:
            continue
        w = h3[fin[:, 0], s, 0]                     # 腕点（仅腕有效的帧）
        disp = np.linalg.norm(np.diff(w, axis=0), axis=1)
        print(f"    腕点逐帧位移 p50={np.median(disp) * 1000 if len(disp) else np.nan:.1f}mm "
              f"p95={np.percentile(disp, 95) * 1000 if len(disp) else np.nan:.1f}mm "
              f"{'(腕点锁原点：腕部相对系，无绝对运动轨迹)' if len(disp) and np.median(disp) < 1e-9 else ''}")
        bone = np.linalg.norm(h3[fin.all(axis=-1), s, 0]
                              - h3[fin.all(axis=-1), s, 9], axis=-1)
        if len(bone):
            print(f"    腕→中指MCP p50={np.median(bone) * 1000:.0f}mm "
                  f"(真手 ~100mm；明显偏小说明源数据带归一化尺度，可用 --scale 修正)")
    if h3.shape[0] == 0:
        raise SystemExit("错误: 无帧可渲染")


def _transform(h3: np.ndarray, scale: float, shift: tuple) -> np.ndarray:
    out = h3.copy()
    out *= scale
    out += np.asarray(shift, np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="parquet 路径（支持通配符，多文件按序拼接）")
    ap.add_argument("--col", default="hand_3d",
                    choices=("hand_3d", "hand_3d_smoothed"),
                    help="读取哪一列（默认 hand_3d）")
    ap.add_argument("--view", default="rotating", choices=("rotating", "static"),
                    help="旋转视角 / 静态视角（默认 rotating）")
    ap.add_argument("--revolutions", type=float, default=2.0,
                    help="旋转视角圈数（默认 2）")
    ap.add_argument("--yaw", type=float, default=180.0,
                    help="静态视角方位角 deg（180=正面，默认）")
    ap.add_argument("--elev", type=float, default=25.0,
                    help="静态视角俯仰角 deg（默认 25）")
    ap.add_argument("--smooth", action="store_true",
                    help="渲染前跑 offline_smooth 零相位平滑（scipy）")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="整体尺度修正（如 1.8）")
    ap.add_argument("--shift", default="0,0,0",
                    help="平移 x,y,z 米（腕点锁原点时挪到别处观察，如 0.3,0.2,0.6）")
    ap.add_argument("--fps", type=float, default=30.0, help="输出帧率（默认 30）")
    ap.add_argument("--frame", type=int, default=None,
                    help="只渲染第 N 帧出 PNG 预览（不写视频）")
    ap.add_argument("--out", help="输出 mp4（默认 <首个输入名>_3d.mp4）")
    ap.add_argument("--encoder", default="auto",
                    choices=("auto", "nvenc", "libx264", "mp4v"))
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        hits = sorted(glob.glob(pat)) or [pat]
        paths += hits
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"错误: 文件不存在: {p}")

    col = {"hand_3d": _COL_H3, "hand_3d_smoothed": _COL_SM}[args.col]
    h3, labels = _load(paths, col)
    _diagnose(h3)

    if args.smooth:
        from stereo_s80m.hand_3d.postprocess import offline_smooth
        rows = [{"observation.keypoints.hand_3d": f.reshape(-1).tolist()}
                for f in h3]
        h3 = offline_smooth(rows, sg_window=7, sg_poly=3, v0=0.08,
                            fps=args.fps, still_window=21)
        print(f"✓ offline_smooth 完成（fps={args.fps:.0f}）")

    if args.scale != 1.0 or args.shift != "0,0,0":
        shift = tuple(float(v) for v in args.shift.split(","))
        if len(shift) != 3:
            raise SystemExit("错误: --shift 需 x,y,z 三个数")
        h3 = _transform(h3, args.scale, shift)
        print(f"✓ 变换: scale={args.scale} shift={args.shift}")

    n = h3.shape[0]
    stem = os.path.splitext(os.path.basename(paths[0]))[0]
    title = f"{stem} ({col.split('.')[-1]}, {args.view})"

    if args.view == "static":
        renderer = RotatingSkeletonRenderer(*RENDER_SIZE, revolutions=1.0,
                                            elevation_deg=args.elev)
        # θ = 2π·rev·idx/(T−1) → idx = yaw/360·(T−1)/rev；T=360, rev=1
        frame_idx = int(round(args.yaw / 360.0 * (360 - 1)))
    else:
        renderer = RotatingSkeletonRenderer(*RENDER_SIZE,
                                            revolutions=args.revolutions,
                                            elevation_deg=args.elev)
        frame_idx = None

    if args.frame is not None:
        if not 0 <= args.frame < n:
            raise SystemExit(f"错误: --frame {args.frame} 越界（0~{n - 1}）")
        idx = frame_idx if args.view == "static" else args.frame
        img = renderer.render(np.asarray(h3[args.frame], np.float64),
                              labels[args.frame], (np.nan, np.nan),
                              idx, n, title)
        png = f"{stem}_frame{args.frame}.png"
        cv2.imwrite(png, img)
        print(f"✓ 预览帧: {png}")
        return

    out = args.out or f"{stem}_3d.mp4"
    sink = create_video_sink(out, args.fps, *RENDER_SIZE, encoder=args.encoder)
    for i in range(n):
        idx = frame_idx if args.view == "static" else i
        sink.write(renderer.render(np.asarray(h3[i], np.float64),
                                   labels[i], (np.nan, np.nan),
                                   idx, n, title))
    final = sink.close()
    if final:
        print(f"✓ 视频: {final}（{n} 帧 @{args.fps:.0f}fps）")
    else:
        raise SystemExit("错误: 视频写出失败")


if __name__ == "__main__":
    main()
