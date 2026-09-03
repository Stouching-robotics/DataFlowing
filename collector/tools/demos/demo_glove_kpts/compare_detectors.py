"""
黑手套检测器 A/B 对比 —— 对照检测器 vs 新训练模型，同一段视频逐帧同条件渲染。

两个 HandPipeline（仅检测器不同，其余 det_imgsz/conf/追踪/姿态门/渲染链完全
一致，即主程序部署口径）对同一帧各跑一遍，输出:
  - <out-dir>/old.mp4 / new.mp4          各自完整渲染视频
  - <out-dir>/side_by_side.mp4           左对照右新并排对比视频（带标签条）
  - <out-dir>/compare_report.csv         指标表（检出率/关键点置信度/抖动/跟踪）

--old 默认 "world"：内置 YOLO-World 零样本（prompt ["hand","glove"]，
imgsz 320——即代码里自带的黑手套检测方案）；也可传 .pt 路径对比旧训练模型。

指标口径（全部基于 process() 门后输出，即用户实际看到的渲染结果）:
  - 检出帧率        boxes 非空帧占比（姿态门已滤掉无手套误检框）
  - kpt均值中位数  所有被渲染手部 21 点置信度均值的中位数
  - 腕点抖动 p50/p95/max   同一 track 相邻帧腕点(点0)位移 px（渲染骨架跳动的直接度量）
  - ID 切换次数     新 track 出现而旧 track 仍活跃的次数
  - 平均 track 寿命 每个 track 被渲染的帧数均值

用法:
    python tools/demos/demo_glove_kpts/compare_detectors.py
    python tools/demos/demo_glove_kpts/compare_detectors.py --old <旧权重.pt|world> --new <新权重.pt>
                                               [--start N] [--frames N]
                                               [--display] [--no-transcode]
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

# 复用 demo_glove_video 的管线构造 / 打包 / 渲染 / ffmpeg 查找
from demo_glove_video import (  # noqa: E402
    build_pipeline,
    draw_kpts_overlay,
    _pack_hand_data,
    _find_ffmpeg,
    DEFAULT_VIDEO,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

OLD_DEFAULT = "world"   # 内置 YOLO-World 零样本（也可传 .pt 路径对比旧训练模型）
NEW_DEFAULT = os.path.join(_REPO_ROOT, "tools", "glove_package", "runs",
                           "hand_det", "weights", "best.pt")
OUT_DEFAULT = os.path.join(_REPO_ROOT, "keypoints_output", "ab_compare")

LABEL_H = 30   # 并排视频顶部标签条高度（偶数，保证 yuv420p 可用）


class ModelStats:
    """单模型逐帧指标收集器。"""

    def __init__(self, label: str, weights: str):
        self.label = label
        self.weights = weights
        self.n_frames = 0
        self.det_frames = 0
        self.hand_counts = []       # 每帧渲染手数
        self.kpt_means = []         # 每手 21 点置信度均值
        self.jitter = []            # 同 track 相邻帧腕点位移 px
        self.track_frames = {}      # track_id -> 被渲染帧数
        self.prev_wrist = {}        # track_id -> 上帧腕点 (x, y)
        self.prev_active = set()    # 上帧活跃 track（门后）
        self.id_switches = 0

    def add(self, boxes, kpts, scores, track_ids):
        self.n_frames += 1
        n = len(boxes)
        if n:
            self.det_frames += 1
        self.hand_counts.append(n)

        active = set(track_ids) if track_ids else set()
        if active and self.prev_active:
            new_ids = active - self.prev_active
            if new_ids:
                self.id_switches += len(new_ids)
        if active:
            self.prev_active = active

        if kpts is not None:
            for i, tid in enumerate(track_ids):
                self.kpt_means.append(float(scores[i][:21].mean()))
                self.track_frames[tid] = self.track_frames.get(tid, 0) + 1
                wrist = kpts[i][0]
                if tid in self.prev_wrist:
                    self.jitter.append(
                        float(np.hypot(*(wrist - self.prev_wrist[tid]))))
                self.prev_wrist[tid] = wrist.copy()

    def summary(self) -> dict:
        j = np.asarray(self.jitter) if self.jitter else np.zeros(1)
        km = np.asarray(self.kpt_means) if self.kpt_means else np.zeros(1)
        det = max(1, self.det_frames)
        return {
            "model": self.label,
            "weights": os.path.basename(self.weights),
            "frames": self.n_frames,
            "det_frames": f"{self.det_frames}/{self.n_frames}",
            "det_rate": round(self.det_frames / max(1, self.n_frames), 4),
            "avg_hands": round(float(np.mean(self.hand_counts)), 3),
            "single_hand_rate": round(
                sum(1 for c in self.hand_counts if c == 1) / det, 4),
            "kpt_mean_med": round(float(np.median(km)), 3),
            "jitter_p50_px": round(float(np.percentile(j, 50)), 2),
            "jitter_p95_px": round(float(np.percentile(j, 95)), 2),
            "jitter_max_px": round(float(j.max()), 2),
            "jitter_n": len(self.jitter),
            "avg_track_len": round(
                float(np.mean(list(self.track_frames.values()))), 1)
            if self.track_frames else 0,
            "id_switches": self.id_switches,
        }


def _as_pt(path: str, out_dir: str) -> str:
    """ultralytics 只认 .pt 后缀；非 .pt 权重复制成临时 .pt 再加载。"""
    if path.lower().endswith(".pt"):
        return path
    dst = os.path.join(out_dir, os.path.basename(path) + ".pt")
    if not os.path.isfile(dst):
        shutil.copyfile(path, dst)
    return dst


def _transcode(tmp: str, out: str, ffmpeg: str) -> bool:
    """mpeg4 临时文件 → H.264（与 demo_glove_video 相同配方）。"""
    if not ffmpeg:
        shutil.move(tmp, out)
        return False
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-i", tmp,
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           out]
    ok = subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    if ok:
        os.remove(tmp)
    else:
        shutil.move(tmp, out)
    return ok


def main():
    ap = argparse.ArgumentParser(description="黑手套检测器 A/B 渲染对比")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--old", default=OLD_DEFAULT,
                    help="对照检测器：'world' 用内置 YOLO-World 零样本（默认），"
                         "或传 .pt 路径")
    ap.add_argument("--new", default=NEW_DEFAULT, help="新模型权重路径")
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--det-device", default="cuda")
    ap.add_argument("--pose-device", default="cuda")
    ap.add_argument("--display", action="store_true", help="弹出实时显示窗口")
    ap.add_argument("--no-transcode", action="store_true",
                    help="跳过 ffmpeg 转码，直接输出 mpeg4")
    args = ap.parse_args()

    for path, tag in [(args.video, "video"), (args.new, "new")]:
        if not os.path.isfile(path):
            print(f"[error] {tag} 不存在: {path}")
            return 1
    if args.old != "world" and not os.path.isfile(args.old):
        print(f"[error] old 不存在: {args.old}")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[error] 无法打开视频: {args.video}")
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    max_frames = args.frames if args.frames > 0 else (total - args.start)

    # 标签条高度按偶数对齐（yuv420p 需要偶数尺寸）
    lab_h = LABEL_H if (height + LABEL_H) % 2 == 0 else LABEL_H + 1

    print(f"[input ] {args.video}")
    print(f"[input ] {width}x{height} @ {src_fps:.0f}fps, 处理 {max_frames} 帧")
    print(f"[out   ] {args.out_dir}")

    # ── 两条同参数管线，只差检测器 ────────────────────────
    old_arg = args.old if args.old == "world" else _as_pt(args.old, args.out_dir)
    new_pt = _as_pt(args.new, args.out_dir)
    t0 = time.perf_counter()
    pipe_old = build_pipeline(args.det_device, args.pose_device, old_arg)
    pipe_new = build_pipeline(args.det_device, args.pose_device, new_pt)
    print(f"[model ] OLD {pipe_old.detector_name} | NEW {pipe_new.detector_name}")
    print(f"[model ] 加载 {time.perf_counter() - t0:.1f}s")

    old_label = "WORLD" if args.old == "world" else "OLD"
    stats_old = ModelStats(
        old_label,
        "yolov8m-worldv2.pt" if args.old == "world" else args.old)
    stats_new = ModelStats("NEW", args.new)

    old_name = ("YOLO-World built-in" if args.old == "world"
                else os.path.basename(args.old))
    new_name = os.path.basename(args.new)

    def _writer(name):
        tmp = os.path.join(args.out_dir, name + ".mpeg4.tmp.mp4")
        return cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                               src_fps, (width, height)), tmp
    w_old, tmp_old = _writer("old")
    w_new, tmp_new = _writer("new")
    sb_tmp = os.path.join(args.out_dir, "side_by_side.mpeg4.tmp.mp4")
    w_sb = cv2.VideoWriter(sb_tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                           src_fps, (width * 2, height + lab_h))

    n = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and n >= max_frames):
                break

            # 同帧、同管线参数，只差检测器权重
            b_o, k_o, s_o, t_o = pipe_old.process(frame)
            b_n, k_n, s_n, t_n = pipe_new.process(frame)
            stats_old.add(b_o, k_o, s_o, t_o)
            stats_new.add(b_n, k_n, s_n, t_n)

            out_o = draw_kpts_overlay(frame, _pack_hand_data(b_o, k_o, t_o), t_o)
            out_n = draw_kpts_overlay(frame, _pack_hand_data(b_n, k_n, t_n), t_n)
            w_old.write(out_o)
            w_new.write(out_n)

            # 并排: 顶部标签条 + hconcat
            strip = np.full((lab_h, width * 2, 3), 20, np.uint8)
            cv2.putText(strip, old_name, (10, lab_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 255), 2, cv2.LINE_AA)
            cv2.putText(strip, new_name, (width + 10, lab_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 255, 90), 2, cv2.LINE_AA)
            cv2.putText(strip, f"frame {n + 1}/{max_frames}",
                        (width - 110, lab_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            side = np.vstack([strip, np.hstack([out_o, out_n])])
            w_sb.write(side)
            n += 1

            if n % 30 == 0:
                fps = n / (time.perf_counter() - t0)
                print(f"[proc  ] {n}/{max_frames} 帧  {fps:5.1f} fps  "
                      f"OLD检出 {stats_old.det_frames} | NEW检出 {stats_new.det_frames}")
            if args.display:
                small = cv2.resize(side, (width, (height + lab_h) // 2))
                cv2.imshow("A/B compare - q:quit", small)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        for w in (w_old, w_new, w_sb):
            w.release()
        cv2.destroyAllWindows()

    ffmpeg = "" if args.no_transcode else _find_ffmpeg()
    if ffmpeg:
        print(f"[ffmpeg] {ffmpeg}")
    names = [("old", tmp_old), ("new", tmp_new),
             ("side_by_side", sb_tmp)]
    for name, tmp in names:
        out = os.path.join(args.out_dir, name + ".mp4")
        _transcode(tmp, out, ffmpeg)
        print(f"[out   ] {out}")

    # ── 指标表 ──────────────────────────────────────────
    rows = [stats_old.summary(), stats_new.summary()]
    keys = list(rows[0].keys())
    print("\n" + "=" * 64)
    for k in keys:
        print(f"{k:<18} {rows[0][k]:<16} {rows[1][k]}")
    print("=" * 64)

    csv_path = os.path.join(args.out_dir, "compare_report.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(keys)
        for r in rows:
            wcsv.writerow([r[k] for k in keys])
    print(f"[csv   ] {csv_path}")

    elapsed = time.perf_counter() - t0
    print(f"[done  ] 处理 {n} 帧, 用时 {elapsed:.1f}s, 平均 {n / elapsed:.1f} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
