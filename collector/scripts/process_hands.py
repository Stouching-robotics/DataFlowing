#!/usr/bin/env python3
"""
手部关键点提取 CLI —— 对已录制会话离线处理。

用法:
    python scripts/process_hands.py kpts /path/to/session --mode bare
    python scripts/process_hands.py label /path/to/session
    python scripts/process_hands.py all /path/to/session --mode bare
"""

import argparse
import os
import sys
import subprocess
import shutil

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hand_tracking import process_session, label_session, load_hand_kpts, draw_kpts_overlay


def _progress_bar(cur: int, total: int, width: int = 40):
    ratio = min(cur / max(total, 1), 1.0)
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    sys.stderr.write(f"\r  [{bar}] {cur}/{total} ({ratio*100:.0f}%)")
    sys.stderr.flush()


def cmd_kpts(args):
    """提取手部关键点（2D + 可选 3D）。"""
    print(f"提取手部关键点: {args.session}")
    print(f"  模式: {args.mode}, 设备: {args.device}")

    sys.stderr.write("\n")
    result = process_session(
        session_path=args.session,
        mode=args.mode,
        det_device=args.device,
        pose_device=args.device,
        progress_cb=_progress_bar,
        status_cb=lambda msg: print(f"  [{msg}]"),
    )

    sys.stderr.write("\n")
    if result["success"]:
        print(f"✅ 完成: {result['frames']} 帧, "
              f"{result['elapsed']:.1f}s, {result['fps']:.1f} fps")
    else:
        print(f"❌ 失败: {result['error']}")
        sys.exit(1)


def cmd_label(args):
    """手势自动标注。"""
    print(f"手势标注: {args.session}")

    result = label_session(
        session_path=args.session,
        progress_cb=_progress_bar,
    )

    sys.stderr.write("\n")
    if result["success"]:
        print(f"✅ 完成: {result['frames']} 帧, {result['elapsed']:.1f}s")
    else:
        print(f"❌ 失败: {result['error']}")
        sys.exit(1)


def cmd_all(args):
    """先提取关键点，再标注。"""
    cmd_kpts(args)
    print()
    cmd_label(args)


def cmd_show(args):
    """显示已有关键点信息。"""
    kpts = load_hand_kpts(args.session)
    if kpts:
        n = len(kpts)
        sample = kpts[min(kpts.keys())]
        print(f"手部关键点数据: {n} 帧")
        print(f"  维度: {len(sample['hand_data'])}")
        print(f"  最多手数: {sample['num_hands']}")
    else:
        print("❌ 未找到手部关键点数据")
        sys.exit(1)


def _find_video(session_path: str) -> str:
    """在 session 目录下找第一个可用的 RGB 视频文件。"""
    from core.helpers import detect_session_format, egodata_video_path, egodata_metadata_path
    import json

    vdir = os.path.join(session_path, "videos")
    if not os.path.isdir(vdir):
        return ""

    fmt = detect_session_format(session_path)
    # 先尝试从 metadata 获取相机列表
    if fmt == "egodata":
        meta_path = egodata_metadata_path(session_path)
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                cameras = json.load(f).get("cameras", {})
            for cam_name in cameras:
                if cameras[cam_name].get("type") == "depth":
                    continue
                for vp in [egodata_video_path(session_path, cam_name),
                           os.path.join(vdir, f"{cam_name}.mp4")]:
                    if os.path.isfile(vp):
                        return vp

    # 直接扫描 videos/ 目录
    for entry in sorted(os.listdir(vdir)):
        if "depth" in entry.lower():
            continue
        full = os.path.join(vdir, entry)
        if os.path.isfile(full) and entry.endswith(".mp4"):
            return full
        # 子目录结构
        sub_dir = os.path.join(full, "chunk-0000")
        if os.path.isdir(sub_dir):
            for fname in os.listdir(sub_dir):
                if fname.endswith(".mp4"):
                    return os.path.join(sub_dir, fname)
    return ""


def cmd_render(args):
    """将手部关键点叠加到视频上，导出可视化视频。"""
    import time
    from core.helpers import keypoints_video_dir

    session_path = args.session
    kpts = load_hand_kpts(session_path)
    if not kpts:
        print("❌ 未找到手部关键点数据，请先运行 kpts 命令")
        sys.exit(1)

    video_path = _find_video(session_path)
    if not video_path:
        print("❌ 未找到可播放的视频文件")
        sys.exit(1)

    cam_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = keypoints_video_dir(session_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cam_name}_hand_kpts.mp4")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    ret, first = cap.read()
    if not ret:
        print("❌ 无法读取视频帧")
        sys.exit(1)
    h, w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ffmpeg 优先，cv2 回退
    writer = None
    ffmpeg_proc = None
    if shutil.which("ffmpeg"):
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", out_path,
        ]
        try:
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            ffmpeg_proc = None

    if not ffmpeg_proc:
        for codec in ["avc1", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if writer.isOpened():
                break

    print(f"输入视频: {video_path} ({total} 帧, {fps} fps)")
    print(f"手部关键点: {len(kpts)} 帧")
    print(f"输出: {out_path}")
    t0 = time.perf_counter()
    fi = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kpt = kpts.get(fi)
        if kpt is not None and np.any(kpt["hand_data"] > 0):
            frame = draw_kpts_overlay(frame, kpt["hand_data"], kpt["track_ids"])
        if ffmpeg_proc:
            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
        else:
            writer.write(frame)
        fi += 1
        if fi % 50 == 0:
            _progress_bar(fi, total)

    cap.release()
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    else:
        writer.release()

    elapsed = time.perf_counter() - t0
    _progress_bar(total, total)
    sys.stderr.write("\n")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"✅ 完成: {out_path}")
    print(f"   {fi} 帧, {elapsed:.1f}s ({fi / elapsed:.1f} fps), {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="手部关键点提取与标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/process_hands.py kpts   data/recordings/Test005/episode_000001 --mode bare
  python scripts/process_hands.py render data/recordings/Test005/episode_000001
  python scripts/process_hands.py all    data/recordings/Test005/episode_000001 --mode bare
  python scripts/process_hands.py show   data/recordings/Test005/episode_000001
        """,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    p_kpts = sub.add_parser("kpts", help="提取手部关键点")
    p_kpts.add_argument("session", help="会话目录路径")
    p_kpts.add_argument("--mode", default="glove",
                        choices=["glove", "bare"], help="追踪模式")
    p_kpts.add_argument("--device", default="cuda", help="推理设备")

    p_label = sub.add_parser("label", help="手势自动标注")
    p_label.add_argument("session", help="会话目录路径")

    p_all = sub.add_parser("all", help="提取 + 标注")
    p_all.add_argument("session", help="会话目录路径")
    p_all.add_argument("--mode", default="glove",
                       choices=["glove", "bare"], help="追踪模式")
    p_all.add_argument("--device", default="cuda", help="推理设备")

    p_show = sub.add_parser("show", help="查看已有数据")
    p_show.add_argument("session", help="会话目录路径")

    p_render = sub.add_parser("render", help="导出关键点可视化视频")
    p_render.add_argument("session", help="会话目录路径")

    args = parser.parse_args()

    if args.command == "kpts":
        cmd_kpts(args)
    elif args.command == "label":
        cmd_label(args)
    elif args.command == "all":
        cmd_all(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "render":
        cmd_render(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
