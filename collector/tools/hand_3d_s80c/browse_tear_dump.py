#!/usr/bin/env python3
"""tear 导出帧人工浏览工具（撕裂排查 runbook 的判侧环节）。

用法：
    venv/bin/python tools/hand_3d_s80c/browse_tear_dump.py [目录]

目录默认 keypoints_output/live_d435/tear_exit_000（找不到时自动挑
keypoints_output 下最新的 tear_* 目录）。也可直接给 worker --raw-dump
导出的 raw_*.jpg 目录，对照"缝在 raw 还是 remap 后"。

判定准则（与 demo --tear-probe help 一致，判定权在人）：
    导出帧全干净（diff 只显整帧均匀运动） → 撕裂在显示/远程桌面层；
    帧内有水平缝（整条水平带内容与上下错位、带内是前一帧内容）→
        撕裂在数据/相机侧；
    斜向拉伸（随快速运动、静止无） → 滚动快门，相机固有。
水平缝在 d（diff）视图下是孤立水平亮带；h 行差提示条会在缝行位置
出现尖峰（提示仅供人参考，本工具不做自动判定——帧内缝检测器既往
实测不可靠，勿信机器判定）。

键位：
    ←/→ 或 a/f      上一/下一帧        Home/End      首/末帧
    PgUp/PgDn       ±10 帧             Space         播放/暂停
    [ / ]           步速 2/5/10/20fps   + / -         整体缩放
    r               复位视图           m             放大镜开关
    滚轮（m 开）     放大镜倍数 2×/4×/8×（鼠标处取样，INTER_NEAREST）
    d               diff 视图（vs 前一帧，×5 增益，缝=孤立水平亮带）
    h               行差提示条（底部条带：每列=图像一行，亮度=该行与
                    前一帧 mean absdiff；红线标最大行，仅供参考）
    t               标记当前帧（退出时打印标记帧号清单，供证据记录）
    s               截图当前视图存 PNG      q / Esc       退出
"""

import argparse
import glob
import os
import re
import sys
import time

import cv2
import numpy as np

_SPEEDS = (2, 5, 10, 20)
_MAG_FACTORS = (2, 4, 8)
_HINT_H = 44          # 行差提示条高度
_MAG_BOX = 176        # 放大镜小窗边长
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _num_key(name):
    """frame_0007.jpg / raw_123456789.jpg → 数字键（排序用）。"""
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else -1


def _find_dir(arg):
    if arg:
        return arg
    cands = sorted(glob.glob(os.path.join("keypoints_output",
                                          "**", "tear_*")),
                   key=os.path.getmtime, reverse=True)
    for c in cands:
        if os.path.isdir(c):
            return c
    print("keypoints_output 下没有任何 tear_* 导出目录。", file=sys.stderr)
    print("生成办法：跑 demo（--tear-probe 默认开）看到撕裂时按 q 退出", file=sys.stderr)
    print("（自动导出 tear_exit_000）或按 t 手动导出；也可把 worker", file=sys.stderr)
    print("--raw-dump 的导出目录直接作为参数传给本工具浏览 raw_*.jpg。",
          file=sys.stderr)
    return "keypoints_output/live_d435/tear_exit_000"


def _load(dirpath):
    files = sorted(glob.glob(os.path.join(dirpath, "*.jpg"))
                   + glob.glob(os.path.join(dirpath, "*.jpeg")),
                   key=_num_key)
    imgs = [(os.path.basename(f), cv2.imread(f, cv2.IMREAD_COLOR))
            for f in files]
    imgs = [(n, im) for n, im in imgs if im is not None]
    if not imgs:
        print(f"目录无 JPEG 帧: {dirpath}"
              + ("（目录不存在或还没导出过帧）"
                 if not os.path.isdir(dirpath) else ""),
              file=sys.stderr)
        sys.exit(1)
    print(f"已载入 {len(imgs)} 帧 ← {dirpath}")
    return imgs


def _row_diff_profile(prev, cur):
    """每行 mean absdiff（灰度，归一化 0-1）——提示条数据。"""
    p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prof = np.abs(c - p).mean(axis=1)
    mx = prof.max()
    return prof / mx if mx > 0 else prof


def _draw_hint_strip(canvas, prof, view_w, peak_row):
    """底部提示条：每列=图像一行，亮度=行差；红线标最大行。"""
    h, w = canvas.shape[:2]
    x0, y0 = 0, h - _HINT_H
    # 行剖面横向拉伸到 view_w（每列对应图像一行）
    xs = np.clip((np.arange(view_w) * len(prof) / max(view_w, 1))
                 .astype(np.int32), 0, len(prof) - 1)
    strip = np.zeros((_HINT_H, view_w, 3), np.uint8)
    for x in range(0, view_w, 2):        # 2px 步进足够，条带只是提示
        v = int(prof[xs[x]] * 220)
        cv2.line(strip, (x, _HINT_H), (x, _HINT_H - 6 - v),
                 (v, v, v), 2)
    if peak_row >= 0:
        px = int(peak_row * view_w / max(len(prof), 1))
        for yy in range(0, _HINT_H, 6):
            cv2.line(strip, (px, yy), (px, yy + 3), (60, 60, 255), 1)
    canvas[y0:y0 + _HINT_H, :view_w] = strip
    cv2.putText(canvas, f"row-diff (advisory)  peak={peak_row}",
                (8, h - 12), _FONT, 0.45, (120, 120, 120), 1, cv2.LINE_AA)


def _draw_magnifier(canvas, img, mx, my, factor):
    """鼠标处 2×/4×/8× 放大镜（INTER_NEAREST 看像素级缝）。"""
    ih, iw = img.shape[:2]
    half = _MAG_BOX // 2 // factor
    x0, x1 = max(0, mx - half), min(iw, mx + half)
    y0, y1 = max(0, my - half), min(ih, my + half)
    if x1 <= x0 or y1 <= y0:
        return
    mag = cv2.resize(img[y0:y1, x0:x1], (_MAG_BOX, _MAG_BOX),
                     interpolation=cv2.INTER_NEAREST)
    ox, oy = canvas.shape[1] - _MAG_BOX - 8, 8
    canvas[oy:oy + _MAG_BOX, ox:ox + _MAG_BOX] = mag
    cv2.rectangle(canvas, (ox, oy), (ox + _MAG_BOX, oy + _MAG_BOX),
                  (0, 255, 255), 1)
    cv2.putText(canvas, f"{factor}x", (ox + 6, oy + 22), _FONT, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default=None,
                    help="tear 导出目录（默认 keypoints_output/live_d435/"
                         "tear_exit_000，找不到自动挑最新 tear_*）")
    args = ap.parse_args()
    dirpath = _find_dir(args.dir)
    imgs = _load(dirpath)

    idx, play, speed_i = 0, False, 0
    zoom = 1.0
    mag_on, mag_i = False, 0
    diff_on, hint_on = False, False
    marks = set()
    mx, my = -1, -1
    wname = "tear browse"

    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, _param):
        nonlocal mx, my, mag_i
        mx, my = x, y
        if event == cv2.EVENT_MOUSEWHEEL and mag_on:
            mag_i = max(0, min(len(_MAG_FACTORS) - 1,
                               mag_i + (1 if flags > 0 else -1)))

    cv2.setMouseCallback(wname, on_mouse)

    def scaled():
        """当前帧按 zoom 缩放；diff 在缩放前原尺寸上算。"""
        name, img = imgs[idx]
        base = img
        if diff_on and idx > 0:
            _, prev = imgs[idx - 1]
            if prev.shape == img.shape:
                base = cv2.absdiff(prev, img)
                base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
                base = cv2.multiply(base, 5)
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            else:
                base = img
        h, w = base.shape[:2]
        view = base
        if abs(zoom - 1.0) > 1e-6:
            view = cv2.resize(base, (max(1, int(w * zoom)),
                                     max(1, int(h * zoom))),
                              interpolation=cv2.INTER_NEAREST if zoom > 1
                              else cv2.INTER_AREA)
        vh, vw = view.shape[:2]
        canvas = np.full((vh + _HINT_H, vw, 3), 20, np.uint8)
        canvas[:vh] = view
        prof, peak = None, -1
        if hint_on and idx > 0:
            _, prev = imgs[idx - 1]
            if prev.shape == img.shape:
                prof = _row_diff_profile(prev, img)
                peak = int(np.argmax(prof))
        if prof is not None:
            _draw_hint_strip(canvas, prof, vw, peak)
        if mag_on and mx >= 0 and my >= 0:
            ix = min(int(mx / zoom), img.shape[1] - 1)
            iy = min(int(my / zoom), img.shape[0] - 1)
            _draw_magnifier(canvas, img, ix, iy, _MAG_FACTORS[mag_i])
        flags = []
        if diff_on:
            flags.append("diff x5")
        if hint_on:
            flags.append("hint")
        if mag_on:
            flags.append(f"mag {_MAG_FACTORS[mag_i]}x")
        if play:
            flags.append(f"play {_SPEEDS[speed_i]}fps")
        if idx in marks:
            flags.append("MARKED")
        cv2.putText(canvas, f"{name}  [{idx + 1}/{len(imgs)}]  "
                            f"zoom={zoom:.1f}  {' '.join(flags)}",
                    (8, vh - 8), _FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return canvas

    last_shot = 0.0
    print("q/Esc 退出 | 见 --help 全键位")
    while True:
        canvas = scaled()
        cv2.imshow(wname, canvas)
        key = cv2.waitKey(1 if play else 30) & 0xFF
        now = time.monotonic()
        if play and now - last_shot >= 1.0 / _SPEEDS[speed_i]:
            idx = min(idx + 1, len(imgs) - 1)
            last_shot = now
            if idx == len(imgs) - 1:
                play = False
        if key in (ord("q"), 27):
            break
        if key in (81, 83) or key == ord("a"):       # ← / a
            idx = max(0, idx - 1)
            play = False
        elif key in (82, 84) or key == ord("f"):     # → / f
            idx = min(len(imgs) - 1, idx + 1)
            play = False
        elif key == ord(" "):
            play = not play
            if play:
                last_shot = time.monotonic()
        elif key == 0:                                # Home
            idx, play = 0, False
        elif key == 5:                                # End
            idx, play = len(imgs) - 1, False
        elif key == 2:                                # PgUp
            idx, play = max(0, idx - 10), False
        elif key == 3:                                # PgDn
            idx, play = min(len(imgs) - 1, idx + 10), False
        elif key in (ord("["),):
            speed_i = max(0, speed_i - 1)
        elif key in (ord("]"),):
            speed_i = min(len(_SPEEDS) - 1, speed_i + 1)
        elif key in (ord("+"), ord("=")):
            zoom = min(4.0, zoom * 1.25)
        elif key in (ord("-"), ord("_")):
            zoom = max(0.25, zoom / 1.25)
        elif key == ord("r"):
            zoom, mag_i = 1.0, 0
        elif key == ord("m"):
            mag_on = not mag_on
        elif key == ord("d"):
            diff_on = not diff_on
        elif key == ord("h"):
            hint_on = not hint_on
        elif key == ord("t"):
            if idx in marks:
                marks.discard(idx)
            else:
                marks.add(idx)
        elif key == ord("s"):
            _shot = f"browse_shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(_shot, canvas)
            print(f"截图已存: {_shot}")
    cv2.destroyAllWindows()
    if marks:
        print("标记帧号: " + ", ".join(str(m + 1) for m in sorted(marks))
              + f"（1 起；文件名 {[imgs[m][0] for m in sorted(marks)]}）")
    else:
        print("无标记帧。")


if __name__ == "__main__":
    main()
