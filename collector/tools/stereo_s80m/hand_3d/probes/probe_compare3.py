#!/usr/bin/env python3
"""
双配置硬对比：纯 MP 单阶段 / 纯 MP 两阶段。

读两个 run 目录（同一会话、同默认后处理，仅 stage-2 不同）的 parquet + 视频：
1. 指标表（打印 + compare_report.csv）：err、双手帧、propagated 数、
   label 翻转次数、raw/offline 抖动中位（帧率/采纳率从 run 日志收集）。
2. 两个并排视频（montage/）：rect_side_by_side（0.5×）、rect_zoom4x
   （跟随 B 手部 ROI 4× 放大——亚像素精度差在此可见）、rot_side_by_side。

用法: venv/bin/python stereo_s80m/hand_3d/probes/probe_compare3.py \
        --a <A目录> --b <B目录> [--montage <输出目录>]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)  # stereo_s80m/hand_detection 已并入 tools/ 命名空间

from stereo_s80m.hand_3d.video_writer import create_video_sink        # noqa: E402

FPS = 25.0
NAMES = {"a": "A 纯MP单阶段", "b": "B 纯MP两阶段"}


# ── 指标 ──────────────────────────────────────────────────────

def _jitter(h3: np.ndarray) -> float:
    """_DispTracker 同口径：帧间共同有效点位移中位的一阶差分中位（mm）。"""
    n = len(h3)
    disp = np.full(n, np.nan)
    for t in range(1, n):
        a, b = h3[t - 1], h3[t]
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if ok.sum() >= 8:
            disp[t] = np.median(np.linalg.norm(a[ok] - b[ok], axis=1))
    jit = np.abs(np.diff(disp))
    jit = jit[np.isfinite(jit)]
    return float(np.median(jit)) * 1000 if jit.size else float("nan")


def compute_metrics(run_dir: str) -> dict:
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(run_dir, "hand_3d_refined",
                                      "chunk-000.parquet")).to_pylist()
    n = len(rows)
    h3 = np.stack([np.asarray(r["observation.keypoints.hand_3d"], np.float32)
                   .reshape(2, 21, 3) for r in rows])
    sm = np.stack([np.asarray(r["observation.keypoints.hand_3d_smoothed"], np.float32)
                   .reshape(2, 21, 3) for r in rows])
    prop = np.stack([np.asarray(r["observation.keypoints.propagated"], np.bool_)
                     for r in rows])
    errs = np.asarray([r["observation.keypoints.reprojection_error"] for r in rows],
                      np.float32).reshape(-1)
    errs = errs[np.isfinite(errs)]
    lab = [[r["observation.keypoints.hand_0_label"],
            r["observation.keypoints.hand_1_label"]] for r in rows]

    flips = [sum(1 for t in range(1, n)
                 if lab[t][s] != lab[t - 1][s] and lab[t][s] and lab[t - 1][s])
             for s in range(2)]
    both = sum(1 for r in rows
               if r["observation.keypoints.hand_0_present"]
               and r["observation.keypoints.hand_1_present"])
    return {
        "n": n,
        "err_mean": float(errs.mean()) if errs.size else float("nan"),
        "err_p95": float(np.percentile(errs, 95)) if errs.size else float("nan"),
        "both": both,
        "prop": int(prop.sum()),
        "flip0": flips[0], "flip1": flips[1],
        "jit_raw0": _jitter(h3[:, 0]), "jit_raw1": _jitter(h3[:, 1]),
        "jit_sm0": _jitter(sm[:, 0]), "jit_sm1": _jitter(sm[:, 1]),
    }


# ── 并排视频 ──────────────────────────────────────────────────

def _panel(img: np.ndarray, label: str, scale: float, color=(40, 255, 255)):
    small = cv2.resize(img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale != 1.0 else img
    cv2.putText(small, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)
    return small


def _hstack(imgs: list, labels: tuple, scale: float):
    return cv2.hconcat([_panel(im, lb, scale) for im, lb in zip(imgs, labels)])


def _montage_rect(videos: tuple, out_path: str, scale=0.5):
    caps = [cv2.VideoCapture(v) for v in videos]
    labels = ("A", "B")
    sink = None
    try:
        while True:
            frames = [c.read() for c in caps]
            if any(not ok for ok, _ in frames):
                break
            imgs = [f for _, f in frames]
            if sink is None:
                h0, w0 = imgs[0].shape[:2]
                sink = create_video_sink(out_path, FPS,
                                         int(w0 * scale * len(imgs)), int(h0 * scale))
            sink.write(_hstack(imgs, labels, scale))
    finally:
        for c in caps:
            c.release()
        if sink is not None:
            sink.close()
    return os.path.isfile(out_path)


def _montage_zoom(videos: tuple, roi_fn, out_path: str, win=(320, 240)):
    """roi_fn(frame_idx) -> (cx, cy) 左目矫正图坐标；无手返回 None。"""
    caps = [cv2.VideoCapture(v) for v in videos]
    labels = ("A", "B")
    sink = None
    t = 0
    try:
        while True:
            frames = [c.read() for c in caps]
            if any(not ok for ok, _ in frames):
                break
            imgs = [f for _, f in frames]
            roi = roi_fn(t)
            if sink is None:
                w0 = win[0] * 2
                sink = create_video_sink(out_path, FPS, w0 * len(imgs), win[1] * 2)
            if roi is None:
                gray = np.full((win[1] * 2, win[0] * 2, 3), 60, np.uint8)
                sink.write(_hstack([gray] * len(imgs), labels, 1.0))
            else:
                cx, cy = roi
                x0 = int(np.clip(cx - win[0] // 2, 0, imgs[0].shape[1] - win[0]))
                y0 = int(np.clip(cy - win[1] // 2, 0, imgs[0].shape[0] - win[1]))
                panels = []
                for im in imgs:
                    crop = im[y0:y0 + win[1], x0:x0 + win[0]]
                    panels.append(_panel(crop, "", 2.0))     # 2× = 4× 放大
                row = cv2.hconcat([_panel(p, lb, 1.0) for p, lb in zip(panels, labels)])
                sink.write(row)
            t += 1
    finally:
        for c in caps:
            c.release()
        if sink is not None:
            sink.close()
    return os.path.isfile(out_path)


def _montage_rot(videos: tuple, out_path: str, scale=0.5):
    return _montage_rect(videos, out_path, scale=scale)


# ── main ──────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="配置 A run 目录")
    ap.add_argument("--b", required=True, help="配置 B run 目录")
    ap.add_argument("--montage", default=None, help="montage 输出目录")
    args = ap.parse_args()

    dirs = {"a": args.a, "b": args.b}
    met = {k: compute_metrics(d) for k, d in dirs.items()}

    # ── 指标表 ──
    print("\n── 双配置指标（222_000008，同后处理仅 stage-2 不同）──")
    hdr = ["配置", "帧率(fps,日志)", "err mean px", "err p95 px", "双手帧",
           "propagated", "label翻转 h0/h1", "抖动raw h0/h1 mm", "抖动offline h0/h1 mm"]
    fps_log = {"a": "101.4", "b": "26.6"}
    rows = []
    for k in ("a", "b"):
        m = met[k]
        row = [NAMES[k], fps_log[k], f"{m['err_mean']:.2f}", f"{m['err_p95']:.2f}",
               f"{m['both']}/{m['n']}", str(m["prop"]),
               f"{m['flip0']}/{m['flip1']}",
               f"{m['jit_raw0']:.2f}/{m['jit_raw1']:.2f}",
               f"{m['jit_sm0']:.2f}/{m['jit_sm1']:.2f}"]
        rows.append(row)
        print("  " + "  ".join(f"{h}={v}" for h, v in zip(hdr[1:], row[1:])))
    out_csv = os.path.join(args.montage or ".", "compare_report.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(hdr)
        wr.writerows(rows)
    print(f"  ✓ {out_csv}")

    if not args.montage:
        return 0
    os.makedirs(args.montage, exist_ok=True)
    rect = [os.path.join(dirs[k], "stereo_triangulate_refined.mp4") for k in "ab"]
    rot = [os.path.join(dirs[k], "hand_3d_rotating.mp4") for k in "ab"]
    for v in rect + rot:
        if not os.path.isfile(v):
            print(f"[ERROR] 缺视频: {v}")
            return 1

    # ROI 函数：B 每帧 hand_0 的 stage-1 左目 2D 点中位（矫正图坐标）做中心，
    # 再 5 帧滑动中值平滑防跳变。不用 3D 投影——median 3D 质心会被深度离群点
    # 拉偏（实测中心 vs 2D 质心 p90 差 179px）；2D 与两配置 rect 骨架同源且无投影误差。
    import pyarrow.parquet as pq
    rows_b = pq.read_table(os.path.join(dirs["b"], "hand_3d_refined",
                                        "chunk-000.parquet")).to_pylist()
    k2_b = np.stack([np.asarray(r["observation.keypoints.stereo_left"],
                                np.float32).reshape(2, 21, 2) for r in rows_b])[:, 0]
    cents = np.full((len(k2_b), 2), np.nan)
    for t in range(len(k2_b)):
        ok = np.isfinite(k2_b[t]).all(axis=1)
        if ok.sum() >= 3:
            cents[t] = np.median(k2_b[t][ok], axis=0)
    smooth = cents.copy()
    for t in range(2, len(cents) - 2):
        w = cents[t - 2:t + 3]
        if np.isfinite(w).all():
            smooth[t] = np.median(w, axis=0)

    def roi_fn(t):
        if not np.isfinite(smooth[t]).all():
            return None
        return float(smooth[t, 0]), float(smooth[t, 1])

    print("\n── 生成并排视频 ──")
    for name, fn, vid_tup, sc in (
            ("rect_side_by_side.mp4", _montage_rect, rect, 0.5),
            ("rect_zoom4x.mp4", _montage_zoom, rect, None),
            ("rot_side_by_side.mp4", _montage_rot, rot, 0.5)):
        out = os.path.join(args.montage, name)
        ok = (fn(vid_tup, out, scale=sc) if sc is not None
              else fn(vid_tup, roi_fn, out))
        print(f"  {'✓' if ok else '[警告] 失败'} {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
