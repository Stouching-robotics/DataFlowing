#!/usr/bin/env python3
"""
S80M 双目帧流协议后端消费示例（--pipe 模式）。

协议（全部大端）:
    [4B left_jpg_len][8B left_ts_ns][left_jpg]
    [4B right_jpg_len][8B right_ts_ns][right_jpg]
    [4B imu_count][imu_count × (8B ts_ns + 3×8B gyro + 3×8B acc)]   # 双精度

用法:
    ./run.sh --pipe - | python3 pipe_consumer.py -                # 实时消费
    python3 pipe_consumer.py out.bin --stats                      # 离线统计
    python3 pipe_consumer.py - --save-dir /tmp/frames             # 保存 JPEG
    python3 pipe_consumer.py - --show                             # 窗口显示（需图形环境）
"""

import argparse
import struct
import sys
import time

# ── 协议常量 ────────────────────────────────────────────────────
_HDR_LEFT = struct.Struct(">IQ")      # 左帧: 4B jpg 长度 + 8B 时间戳(ns)
_HDR_RIGHT = struct.Struct(">IQ")     # 右帧
_HDR_IMU_COUNT = struct.Struct(">I")  # 本帧周期 IMU 样本数
_HDR_IMU = struct.Struct(">Q6d")      # 每个 IMU 样本: 8B ts + 6×8B (gx,gy,gz,ax,ay,az)


def _read_exact(f, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None          # EOF
        buf += chunk
    return buf


def consume(source, show=False, save_dir=None, max_frames=None, stats=False):
    """source: 打开的文件对象（二进制）。"""
    import os
    n = 0
    n_imu = 0
    t0 = time.time()
    left_ts_last = right_ts_last = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if show:
        import cv2
        cv2.namedWindow("left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("right", cv2.WINDOW_NORMAL)

    while True:
        hl = _read_exact(source, _HDR_LEFT.size)
        if hl is None:
            break
        left_len, left_ts = _HDR_LEFT.unpack(hl)
        left_jpg = _read_exact(source, left_len)
        hr = _read_exact(source, _HDR_RIGHT.size)
        if hr is None:
            break
        right_len, right_ts = _HDR_RIGHT.unpack(hr)
        right_jpg = _read_exact(source, right_len)
        hc = _read_exact(source, _HDR_IMU_COUNT.size)
        if hc is None:
            break
        (imu_count,) = _HDR_IMU_COUNT.unpack(hc)
        imu_samples = []
        for _ in range(imu_count):
            raw = _read_exact(source, _HDR_IMU.size)
            if raw is None:
                break
            imu_samples.append(_HDR_IMU.unpack(raw))
        n += 1
        n_imu += len(imu_samples)
        left_ts_last, right_ts_last = left_ts, right_ts

        if show or save_dir:
            import cv2
            import numpy as np
            l_img = cv2.imdecode(np.frombuffer(left_jpg, np.uint8), cv2.IMREAD_COLOR)
            r_img = cv2.imdecode(np.frombuffer(right_jpg, np.uint8), cv2.IMREAD_COLOR)
            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"left_{left_ts:020d}.jpg"), l_img)
                cv2.imwrite(os.path.join(save_dir, f"right_{right_ts:020d}.jpg"), r_img)
            if show:
                cv2.imshow("left", l_img)
                cv2.imshow("right", r_img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        if stats and n % 100 == 0:
            el = time.time() - t0
            print(f"[consumer] {n} 帧  ({n/el:.1f} fps)  IMU {n_imu} 样本", flush=True)
        if max_frames and n >= max_frames:
            break

    el = time.time() - t0
    print(f"[consumer] 共 {n} 帧, IMU {n_imu} 样本, {el:.1f}s "
          f"({n/el:.1f} fps)" if n else "[consumer] 空流", flush=True)


def main():
    ap = argparse.ArgumentParser(description="S80M --pipe 帧流消费示例")
    ap.add_argument("source", help="'-' = 从 stdin 读（管道）；或 .bin 文件路径")
    ap.add_argument("--show", action="store_true", help="窗口显示")
    ap.add_argument("--save-dir", default=None, help="保存 JPEG 到目录")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stats", action="store_true", help="打印速率统计")
    args = ap.parse_args()

    if args.source == "-":
        f = sys.stdin.buffer
    else:
        f = open(args.source, "rb")
    try:
        consume(f, show=args.show, save_dir=args.save_dir,
                max_frames=args.max_frames, stats=args.stats)
    finally:
        if args.source != "-":
            f.close()


if __name__ == "__main__":
    main()
