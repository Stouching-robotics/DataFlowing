#!/usr/bin/env python3
"""HEVC 压缩对比测试：原视频 vs 多种 HEVC/x264 压缩档 → 拼图对比视频。

用法:
    venv/bin/python tools/hevc_test/hevc_test.py [源视频.mp4] [输出子目录名]

第二个参数可选：产物写到 output/<子目录名>/（默认 output/），
多段视频对比时用它可以互不覆盖。

流程:
  1. 用 imageio 静态 ffmpeg（与主程序上传预压缩同款编码器）把源视频按
     x265 CRF 28/30/34/37 + x264 CRF30（旧上传方案对照）各压一遍
     → output/compressed/
  2. 解码全部变体与原件逐帧对齐，缩放到 640×360 拼成 3×2 网格，
     每格顶部标注格式与体积占比 → output/compare.mp4
  3. 逐变体计算 SSIM（vs 原件）与体积 → output/benchmark.txt

产物全部在 output/ 下，不写源目录。
"""

import os
import re
import subprocess
import sys

import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_OUT_DIR = os.path.join(_SCRIPT_DIR, "output")
_COMP_DIR = os.path.join(_OUT_DIR, "compressed")

_DEFAULT_SRC = os.path.join(
    _REPO_ROOT, "data", "recordings",
    "D435-裸手_3D手部关键点识别_AI标注",
    "videos", "chunk-000", "D435_head_rgb", "episode-007.mp4")

# (名字, 编码器, CRF)；orig 不编码，直接引用源文件
_VARIANTS = [
    ("x265_crf28", "libx265", "28"),
    ("x265_crf30", "libx265", "30"),
    ("x265_crf34", "libx265", "34"),
    ("x265_crf37", "libx265", "37"),
    ("x264_crf30", "libx264", "30"),
]
_CELL_W, _CELL_H = 640, 360       # 每格尺寸 → 3×2 网格 1920×720
_GRID_COLS = 3


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.isfile(exe):
            return exe
    except Exception:
        pass
    for name in ("ffmpeg", "ffmpeg.exe"):
        from shutil import which
        p = which(name)
        if p:
            return p
    raise RuntimeError("找不到可用的 ffmpeg")


def encode(ffmpeg, src, dst, enc, crf):
    """重编码一遍（veryfast，与上传预压缩同预设）；无音轨（源无音频）。"""
    cmd = [ffmpeg, "-y", "-i", src, "-an", "-c:v", enc, "-crf", crf,
           "-preset", "veryfast", "-pix_fmt", "yuv420p", dst]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    if r.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError(f"编码失败 {enc} crf{crf}: "
                           f"{r.stderr.decode('utf-8', 'ignore')[-400:]}")


def ssim(ffmpeg, src, comp):
    """comp 与源（原件解码帧）的整体 SSIM，解析 'All:xxxx'。"""
    r = subprocess.run(
        [ffmpeg, "-i", src, "-i", comp, "-lavfi", "ssim", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    m = re.search(r"All:([\d.]+)", r.stderr.decode("utf-8", "ignore"))
    return float(m.group(1)) if m else float("nan")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_SRC
    out_dir = (os.path.join(_OUT_DIR, sys.argv[2])
               if len(sys.argv) > 2 else _OUT_DIR)
    comp_dir = os.path.join(out_dir, "compressed")
    if not os.path.isfile(src):
        print(f"源视频不存在: {src}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(comp_dir, exist_ok=True)
    ffmpeg = find_ffmpeg()
    print(f"ffmpeg: {ffmpeg}")

    src_mb = os.path.getsize(src) / 1048576
    cap = cv2.VideoCapture(src)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"源视频: {src_mb:.1f}MB, {n_frames} 帧 @ {fps:.0f}fps")

    # ── 步骤 1: 逐档压缩 ──
    print("── 步骤 1: 逐档压缩 ──")
    paths = {}
    for name, enc, crf in _VARIANTS:
        dst = os.path.join(comp_dir, f"{name}.mp4")
        if not os.path.isfile(dst):
            encode(ffmpeg, src, dst, enc, crf)
        paths[name] = dst
        mb = os.path.getsize(dst) / 1048576
        print(f"  {name}: {mb:.2f}MB ({mb / src_mb * 100:.1f}% of orig)")

    # ── 步骤 2: SSIM（vs 原件）──
    print("── 步骤 2: SSIM vs 原件 ──")
    ssims = {}
    for name, p in paths.items():
        ssims[name] = ssim(ffmpeg, src, p)
        print(f"  {name}: SSIM {ssims[name]:.4f}")

    # ── 步骤 3: 解码拼图 ──
    print("── 步骤 3: 解码拼图（3×2 网格 → compare.mp4）──")
    order = [("orig", src)] + [(n, p) for n, p in paths.items()]
    caps = {k: cv2.VideoCapture(v) for k, v in order}
    last = {k: None for k in caps}
    grid_h, grid_w = _CELL_H * 2, _CELL_W * _GRID_COLS
    # -framerate 声明输入帧率（-r 历史歧义）+ -fps_mode passthrough 禁止
    # vsync cfr 抽/补帧：写多少帧就出多少帧（曾出现 1150 帧变 4600 帧
    # 的 4× 复制，就是默认 vsync 在补帧）
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{grid_w}x{grid_h}", "-framerate", str(int(round(fps))),
         "-i", "-", "-fps_mode", "passthrough", "-an",
         "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", os.path.join(out_dir, "compare.mp4")],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)

    def cell_label(k):
        if k == "orig":
            return "ORIG (CRF23, 100%)"
        pct = os.path.getsize(paths[k]) / os.path.getsize(src) * 100
        name, enc, crf = next((v for v in _VARIANTS if v[0] == k))
        return f"{enc} CRF{crf} ({pct:.1f}%)"

    for i in range(n_frames):
        row_frames = []
        for k, _v in order:
            ok, fr = caps[k].read()
            if not ok:
                fr = last[k]
            else:
                last[k] = fr
            if fr is None:
                fr = np.zeros((_CELL_H, _CELL_W, 3), np.uint8)
            else:
                fr = cv2.resize(fr, (_CELL_W, _CELL_H))
            row_frames.append(fr)
        grid = np.vstack([np.hstack(row_frames[0:3]),
                          np.hstack(row_frames[3:6])])
        labels = [cell_label(k) for k, _v in order]
        # 顶栏按格绘制（每格 640 宽）
        for idx, lab in enumerate(labels):
            r, c = divmod(idx, _GRID_COLS)
            x0, y0 = c * _CELL_W, r * _CELL_H
            grid[y0:y0 + 24, x0:x0 + _CELL_W] = (0, 0, 0)
            cv2.putText(grid, lab, (x0 + 8, y0 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)
        proc.stdin.write(grid.astype(np.uint8).tobytes())
        if i % 300 == 0:
            print(f"    frame {i}/{n_frames}")
    proc.stdin.close()
    proc.wait(timeout=120)
    for c in caps.values():
        c.release()
    print(f"  拼图完成: {os.path.join(out_dir, 'compare.mp4')}")

    # ── 步骤 4: 汇总 ──
    print("── 步骤 4: 汇总 ──")
    lines = ["HEVC 压缩对比（原件 = 录制 CRF23）", "-" * 60,
             f"原件: {src_mb:.1f}MB, SSIM 1.0000（基准）"]
    for name, enc, crf in _VARIANTS:
        mb = os.path.getsize(paths[name]) / 1048576
        lines.append(f"{enc} CRF{crf}: {mb:.2f}MB "
                     f"({mb / src_mb * 100:.1f}%), SSIM {ssims[name]:.4f}")
    bench = os.path.join(out_dir, "benchmark.txt")
    with open(bench, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for l in lines:
        print("  " + l)
    print(f"\n全部产物在 {out_dir}/")


if __name__ == "__main__":
    import numpy as np  # noqa: E402（main 内拼图用）
    main()
