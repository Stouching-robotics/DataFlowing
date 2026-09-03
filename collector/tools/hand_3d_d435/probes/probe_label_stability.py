#!/usr/bin/env python3
"""probe_label_stability.py —— 逐槽 label 翻转计数。

判据：全片 0 次翻转（连续 present 帧间 Left↔Right 切换）。
voter 7 帧多数票 + 槽位分配标签门应保证恒稳。

用法（venv）:
  python tools/hand_3d_d435/probes/probe_label_stability.py --parquet <chunk-000.parquet>
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True,
                    help="hand_3d_refined/chunk-000.parquet 路径")
    args = ap.parse_args()
    if not os.path.exists(args.parquet):
        sys.exit(f"错误: parquet 不存在: {args.parquet}")

    import pyarrow.parquet as pq
    t = pq.read_table(args.parquet)
    n = t.num_rows
    total = 0
    for s, name in enumerate(("slot0", "slot1")):
        present = t[f"observation.keypoints.hand_{s}_present"].to_numpy()
        labels = [str(x) for x in
                  t[f"observation.keypoints.hand_{s}_label"].to_pylist()]
        flips = 0
        flip_frames = []
        for i in range(1, n):
            if present[i] and present[i - 1] \
                    and labels[i] in ("Left", "Right") \
                    and labels[i - 1] in ("Left", "Right") \
                    and labels[i] != labels[i - 1]:
                flips += 1
                flip_frames.append(i)
        total += flips
        print(f"{name}: 翻转 {flips} 次 "
              + (f"（帧 {flip_frames[:8]}）" if flips else ""))
    print(f"\nlabel 稳定判据: {'PASS' if total == 0 else 'FAIL'}")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
