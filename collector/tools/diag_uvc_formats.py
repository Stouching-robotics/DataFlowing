"""UVC camera format/fps diagnostic (Windows + Linux).

For each camera index, try every FOURCC x resolution combination with the
same backends the main program uses, and MEASURE the real frame rate by
reading frames (~2 s per combination). Use it to find which combination
actually reaches ~30 fps (e.g. YUYV @ 1280x960 is hardware-limited to
5 fps on some cameras while MJPG reaches 30).

Usage:
    Windows:  venv\\Scripts\\python.exe tools\\diag_uvc_formats.py [max_index]
    Linux:    venv/bin/python3 tools/diag_uvc_formats.py [max_index]

Close the main program (and any app using the cameras) before running.
Output is ASCII-only so the Windows GBK console shows it correctly.
"""

import sys
import time

import cv2

BACKENDS = ([("DShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
            if sys.platform == "win32" else [("V4L2", cv2.CAP_V4L2)])
FOURCCS = ["MJPG", "YUYV"]
RESES = [(1280, 960), (960, 720), (800, 600), (640, 480)]
TARGET_FPS = 30
READ_FRAMES = 20          # 实测读帧数：5fps 下约 4s，30fps 下 <1s


def _is_sdk_device(index):
    """Linux: skip FTDI stereo nodes (need the vendor SDK, not UVC)."""
    if sys.platform == "win32":
        return False
    try:
        with open(f"/sys/class/video4linux/video{index}/name") as f:
            return "FTDI" in f.read()
    except OSError:
        return False


def measure(index, backend_val, fourcc, w, h):
    """Open camera with the given combo and measure the real fps."""
    cap = cv2.VideoCapture(index, backend_val)
    if not cap.isOpened():
        return None
    try:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        rb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rb_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        rb_fps = float(cap.get(cv2.CAP_PROP_FPS))
        # Windows DShow/MSMF: re-assert FOURCC after res/fps (media-type
        # rebuild can drop MJPG back to YUYV) -- same as the main program.
        if sys.platform == "win32" and fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        n, t0, ok = 0, 0.0, False
        for _ in range(READ_FRAMES):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            n += 1
            if n == 1:
                t0 = time.perf_counter()
        if n < 2:
            return None
        fps = (n - 1) / (time.perf_counter() - t0)
        return {"rb_w": rb_w, "rb_h": rb_h, "rb_fourcc": rb_fourcc,
                "rb_fps": rb_fps, "fps": fps}
    finally:
        cap.release()


def fourcc_name(v):
    return "".join(chr(c) if 32 <= c < 127 else "?"
                   for c in ((v >> 24) & 0xFF, (v >> 16) & 0xFF,
                             (v >> 8) & 0xFF, v & 0xFF))


def main():
    max_index = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"== UVC format/fps diagnostic (target {TARGET_FPS} fps) ==")
    for i in range(max_index):
        if _is_sdk_device(i):
            continue
        for bname, bval in BACKENDS:
            probe = cv2.VideoCapture(i, bval)
            opened = probe.isOpened()
            probe.release()
            if not opened:
                continue
            print(f"-- camera index {i} (backend {bname}) --")
            for fc in FOURCCS:
                for (w, h) in RESES:
                    r = measure(i, bval, fc, w, h)
                    if r is None:
                        print(f"   {fc:>4} {w}x{h}: open/read failed")
                        continue
                    star = "*" if r["fps"] >= TARGET_FPS * 0.8 else " "
                    print(f" {star} {fc:>4} {w}x{h}: real {r['fps']:5.1f} fps"
                          f"  (readback fourcc={fourcc_name(r['rb_fourcc'])}"
                          f" {r['rb_w']}x{r['rb_h']} fps={r['rb_fps']})")
            break   # per index: prefer the first backend that opens
    print("done.")


if __name__ == "__main__":
    main()
