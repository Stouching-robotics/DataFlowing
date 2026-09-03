#!/usr/bin/env python3
"""录制编码速度基准（v1.0.9）—— x265/x264/NVENC 各档在本机的实测编码 fps。

与主程序录制端同款口径：rawvideo bgr24 管道 → ffmpeg 子进程 → null muxer，
按并行流数同时起多个 ffmpeg 进程（对应主程序每路相机一个子进程、写盘
线程每 tick 串行喂全部槽的背压近似）。合成帧为全零帧（编码最快）——
真实帧含纹理会更慢，主程序 1.5x 门槛余量即为此而设。

输出纯 ASCII（Windows GBK 控制台安全）+ 落 output/encode_bench.txt。
结论区打印主程序自动选择判据：libx265 每流 fps >= 录制帧率 x 1.5 才算达标。

用法:
    venv/bin/python tools/hevc_test/encode_bench.py [并行流数] [每流帧数]
    默认 1,2,4 三种并行度、每流 90 帧；只测一种并行度可提速。
"""
import os
import shutil
import subprocess
import sys
import threading
import time

FPS = 30                     # 录制帧率
MIN_RATIO = 1.5              # x265 达标门槛 = FPS x 1.5（与主程序一致）
RESOLUTIONS = [(1280, 960), (1280, 800), (1280, 720)]
ENCODERS = [
    ("libx265", ["-crf", "30", "-preset", "veryfast", "-tag:v", "hvc1"]),
    ("libx265", ["-crf", "30", "-preset", "ultrafast", "-tag:v", "hvc1"]),
    ("libx264", ["-preset", "ultrafast", "-crf", "23"]),
]


def find_ffmpeg():
    """imageio 静态二进制 → PATH，-version 自检。"""
    cands = []
    try:
        import imageio_ffmpeg
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    p = shutil.which("ffmpeg")
    if p and p not in cands:
        cands.append(p)
    for exe in cands:
        try:
            if subprocess.run([exe, "-version"], capture_output=True,
                              timeout=10).returncode == 0:
                return exe
        except Exception:
            continue
    return None


def has_encoder(ffmpeg, enc):
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                           capture_output=True, timeout=30)
        text = ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", "ignore")
        return enc in text.split()
    except Exception:
        return False


def _feed(proc, frame_bytes, frames):
    try:
        for _ in range(frames):
            proc.stdin.write(frame_bytes)
        proc.stdin.flush()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass


def bench(ffmpeg, enc, args, w, h, streams, frames):
    """并行 streams 个 ffmpeg 各编 frames 帧（全零合成帧）→ null。

    返回 (每流fps, 成功)：每流 fps = frames / 总耗时（全部流并行跑满
    同一墙钟，与主程序探针口径一致）。
    """
    frame_bytes = bytes(w * h * 3)
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-frames:v", str(frames),
           "-c:v", enc, *args, "-f", "null", "-"]
    procs = []
    t0 = time.monotonic()
    ok = True
    try:
        for _ in range(streams):
            procs.append(subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL))
        feeders = [threading.Thread(target=_feed, args=(p, frame_bytes, frames),
                                    daemon=True) for p in procs]
        for t in feeders:
            t.start()
        for p in procs:
            p.wait(timeout=60)
    except Exception:
        ok = False
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
            ok = False
        elif p.returncode != 0:
            ok = False
    elapsed = time.monotonic() - t0
    if not ok or elapsed <= 0:
        return 0.0, False
    return frames / elapsed, True


def main():
    streams_list = [1, 2, 4]
    frames = 90
    if len(sys.argv) > 1:
        streams_list = [int(x) for x in
                        sys.argv[1].replace(" ", "").split(",") if x]
    if len(sys.argv) > 2:
        frames = int(sys.argv[2])

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("FAIL: ffmpeg not found")
        return 1

    encs = list(ENCODERS)
    if has_encoder(ffmpeg, "hevc_nvenc"):
        encs.append(("hevc_nvenc", ["-rc", "vbr", "-cq", "30",
                                    "-preset", "p4", "-tag:v", "hvc1"]))

    threshold = int(FPS * MIN_RATIO)
    lines = []
    p = lines.append
    p("encode_bench  (ffmpeg: %s)" % ffmpeg)
    p("gate: per-stream fps >= %d  ->  x265 OK for %dfps recording"
      % (threshold, FPS))
    p("")
    head = "%-12s %-20s" % ("resolution", "encoder")
    for s in streams_list:
        head += "  %5d-stream" % s
    p(head)
    p("-" * 76)

    for w, h in RESOLUTIONS:
        for enc, args in encs:
            row = "%-12s %-20s" % (f"{w}x{h}",
                                   f"{enc} {args[1]}")
            for s in streams_list:
                fps, ok = bench(ffmpeg, enc, args, w, h, s, frames)
                if not ok:
                    row += "  %10s" % "FAIL"
                else:
                    mark = "OK " if fps >= threshold else "LOW"
                    row += "  %6.1f(%s)" % (fps, mark)
            p(row)

    p("")
    p("OK  = per-stream fps >= %d (x265 gate for %dfps recording)"
      % (threshold, FPS))
    p("LOW = below gate -> auto-select falls back to x264")
    p("NOTE: synthetic all-zero frames encode fastest; real frames are slower")
    text = "\n".join(lines)
    print(text)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(here, "output"), exist_ok=True)
        out = os.path.join(here, "output", "encode_bench.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("\n[saved] %s" % out)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
