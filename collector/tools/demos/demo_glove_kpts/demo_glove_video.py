"""
黑手套关键点识别 + 渲染 Demo —— 对单个视频逐帧跑手套管线并叠加关键点。

完全复用主程序两条现成代码路径（行为与 GUI 会话后处理一致）:
  - core/hand_processor.py 内部调用 core.hand_tracking.process_session() 的
    glove 分支 → HandPipeline (YOLO best.pt 检测黑色手套 + RTMPose 21 关键点)
  - core.hand_tracking.draw_kpts_overlay() → hand_common.draw_hand 渲染

用法:
    python tools/demos/demo_glove_kpts/demo_glove_video.py
    python tools/demos/demo_glove_kpts/demo_glove_video.py --video <路径> [--out <路径>]
                                               [--detector <权重.pt>]
                                               [--start N] [--frames N]
                                               [--det-device cuda] [--pose-device cuda]
                                               [--no-display]

按键（显示窗口内）: q/Esc 退出, 空格 暂停/继续

输出视频默认写入本 demo 目录: <源文件名>_kpts.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np

from core.hand_tracking import (  # noqa: E402
    _lazy_import_pipeline,
    _pack_hand_data,
    draw_kpts_overlay,
)

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_VIDEO = os.path.join(
    _REPO_ROOT, "data", "recordings", "Project_Test10",
    "Project_Test10_000003", "videos", "D435_depth_rgb",
    "chunk-0000", "D435_depth_rgb.mp4")


def _find_ffmpeg() -> str:
    """找可用的 ffmpeg。优先 lerobot 环境（conda base 的 ffmpeg 因 openvino/tbb 失效）。"""
    for cand in [os.path.expanduser("~/miniconda3/envs/lerobot312/bin/ffmpeg"),
                 os.path.expanduser("~/miniconda3/envs/lerobot/bin/ffmpeg"),
                 shutil.which("ffmpeg") or "",
                 os.path.expanduser("~/miniconda3/bin/ffmpeg")]:
        if cand and os.path.isfile(cand):
            return cand
    return ""


def build_pipeline(det_device: str, pose_device: str, detector: str = ""):
    """与 core/hand_processor.py → process_session() 的 glove 分支同一套逻辑。"""
    try:
        import torch
        if not torch.cuda.is_available():
            if det_device == "cuda":
                det_device = "cpu"
            if pose_device == "cuda":
                pose_device = "cpu"
    except Exception:
        det_device = "cpu"
        pose_device = "cpu"

    HP = _lazy_import_pipeline()
    det_path = detector or os.path.join(_REPO_ROOT, "tools", "hand_detection", "best.pt")
    return HP(detector=det_path, det_device=det_device,
              pose_device=pose_device, max_hands=2)


def main():
    ap = argparse.ArgumentParser(description="黑手套视频关键点识别 + 渲染 Demo")
    ap.add_argument("--video", default=DEFAULT_VIDEO,
                    help="输入视频路径 (默认 Project_Test10_000003 D435 RGB)")
    ap.add_argument("--out", default="",
                    help="输出视频路径 (默认写入本 demo 目录: <源文件名>_kpts.mp4)")
    ap.add_argument("--detector", default="",
                    help="检测器权重路径 (默认 tools/hand_detection/best.pt)")
    ap.add_argument("--start", type=int, default=0, help="起始帧 (默认 0)")
    ap.add_argument("--frames", type=int, default=0,
                    help="处理帧数 (默认 0 = 到结尾)")
    ap.add_argument("--det-device", default="cuda", help="检测器设备 cuda/cpu")
    ap.add_argument("--pose-device", default="cuda", help="关键点设备 cuda/cpu")
    ap.add_argument("--no-display", action="store_true",
                    help="不弹显示窗口，只写输出视频")
    ap.add_argument("--no-transcode", action="store_true",
                    help="跳过 ffmpeg 转码，直接输出 cv2 的 mpeg4（部分播放器打不开）")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"[error] 视频不存在: {args.video}")
        return 1

    # ── 输出路径: 默认放本 demo 目录 ─────────────────────
    if args.out:
        out_path = args.out
    else:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(DEMO_DIR, f"{stem}_kpts.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # ── 打开视频 ────────────────────────────────────────
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

    print(f"[input ] {args.video}")
    print(f"[input ] {width}x{height} @ {src_fps:.0f}fps, "
          f"总帧 {total}, 从第 {args.start} 帧起处理 {max_frames} 帧")
    print(f"[output] {out_path}")

    # ── 加载管线（首帧含模型加载，较慢） ────────────────
    t_load = time.perf_counter()
    pipeline = build_pipeline(args.det_device, args.pose_device, args.detector)
    print(f"[model ] {pipeline.detector_name} | det={args.det_device} "
          f"pose={args.pose_device} | 加载 {time.perf_counter() - t_load:.1f}s")

    # cv2 的 FFmpeg 后端只支持 mpeg4 编码，播放器兼容性差 → 先写临时文件，
    # 结束后用 ffmpeg 转 H.264（libx264 + faststart，通用播放器都能开）
    tmp_path = out_path + ".mpeg4.tmp.mp4"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (width, height))
    display = not args.no_display
    paused = False
    n = 0
    t0 = time.perf_counter()
    n_hands_frames = 0

    try:
        while True:
            # ── 空格暂停时仍刷新窗口，但不读帧 ──────────
            if not paused:
                ok, frame = cap.read()
                if not ok or (max_frames and n >= max_frames):
                    break

                # 与 process_session() 相同的逐帧推理
                boxes, kpts, scores, track_ids = pipeline.process(frame)

                # 与主程序相同的打包 + 渲染（框 / Hand #ID / 骨骼 / 伸指）
                packed = _pack_hand_data(boxes, kpts, track_ids)
                out = draw_kpts_overlay(frame, packed, track_ids)

                if boxes:
                    n_hands_frames += 1
                writer.write(out)
                n += 1

                if n % 30 == 0:
                    fps = n / (time.perf_counter() - t0)
                    print(f"[proc  ] {n}/{max_frames} 帧  "
                          f"{fps:5.1f} fps  检出 {n_hands_frames} 帧")

                # ── 显示窗口（帧率限速在下方 waitKey 里做） ──
                if display:
                    cv2.imshow("glove kpts demo - q:quit  space:pause", out)
                    elapsed = time.perf_counter() - t0
                    # 按源帧率限速
                    wait_ms = max(1, int(1000.0 / src_fps * n - elapsed * 1000))
                else:
                    wait_ms = 1
            else:
                wait_ms = 1

            if display:
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):          # q / Esc
                    print("[demo  ] 用户退出")
                    break
                elif key == ord(" "):              # 空格 暂停/继续
                    paused = not paused
                    print(f"[demo  ] {'暂停' if paused else '继续'}")
    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    # ── 转码 H.264（cv2 只能写 mpeg4，部分播放器打不开） ──
    if n > 0:
        ffmpeg = "" if args.no_transcode else _find_ffmpeg()
        if ffmpeg:
            print(f"[ffmpeg] {ffmpeg}")
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-i", tmp_path,
                   "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   out_path]
            if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
                os.remove(tmp_path)
            else:
                print("[ffmpeg] 转码失败，保留 mpeg4 原始输出")
                shutil.move(tmp_path, out_path)
        else:
            if not args.no_transcode:
                print("[ffmpeg] 未找到可用 ffmpeg，输出为 mpeg4 "
                      "（部分播放器可能打不开）")
            shutil.move(tmp_path, out_path)

    elapsed = time.perf_counter() - t0
    print(f"[done  ] 处理 {n} 帧, 用时 {elapsed:.1f}s, "
          f"平均 {n / elapsed:.1f} fps, 检出帧 {n_hands_frames}/{n}")
    print(f"[done  ] 输出: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
