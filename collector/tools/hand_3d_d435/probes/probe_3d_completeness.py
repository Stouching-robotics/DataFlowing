#!/usr/bin/env python3
"""probe_3d_completeness.py —— 3D 输出完整性与翻面事件探针。

旋转 3D 渲染质量的直接判据：
  1. 骨架完整性：整手 21 点全有限的帧率（旧管线 3.3%/17.4%，渲染骨架
     支离破碎）；每帧每手有效点中位（旧 16/21）。
  2. 翻面事件：相邻帧某点 z 跳变 >300mm 的帧数（旧 5+2 起；物理上
     0.3m/帧 ≈ 9m/s，真实手速不可达 → 判边界深度混入背景的残留）。
  3. 腕点（点 0）有限率：旧 38%（腕是骨架锚点 + 掌心连线起点）。

用法（venv）: python tools/hand_3d_d435/probes/probe_3d_completeness.py --parquet <p>

判据：整手完整率 ≥50% 且 翻面事件 ≤2 起（新管线 band 采样 + 时序门 +
tracker 填充应把两者分别打到 ~90% 和 ~0；阈值放宽容忍场景差异）。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq

FLAP_M = 0.3        # 相邻帧 z 跳变阈值（米/帧，>9m/s 非人手速）
FULL_OK = 0.50      # 整手 21 点全有限帧率下限
FLAP_OK = 2         # 翻面事件上限（起）


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()

    t = pq.read_table(args.parquet)
    h3 = np.array(t["observation.keypoints.hand_3d_smoothed"]
                  .to_pylist()).reshape(-1, 2, 21, 3)
    n = len(h3)
    fin_all = np.isfinite(h3).all(axis=3)          # (N,2,21)
    nval = fin_all.sum(axis=2)                     # (N,2) 每帧每手有效点
    full = (nval == 21).mean(axis=0)               # 整手完整帧率
    med = np.median(nval[nval > 0]) if (nval > 0).any() else np.nan

    flaps = 0
    for s in range(2):
        z = h3[:, s, :, 2]
        dz = np.abs(np.diff(z, axis=0))
        flaps += int(((dz > FLAP_M) & np.isfinite(dz)).any(axis=1).sum())
    wrist = fin_all[:, :, 0].mean(axis=0)          # 腕点有限率

    ok = float(full.min()) >= FULL_OK and flaps <= FLAP_OK
    print(f"整手 21 点全有限帧率: slot0 {full[0] * 100:.1f}%  "
          f"slot1 {full[1] * 100:.1f}% (≥{FULL_OK * 100:.0f}%) "
          f"{'PASS' if full.min() >= FULL_OK else 'FAIL'}")
    print(f"每帧每手有效点中位: {med:.0f}/21（旧管线 16/21）")
    print(f"腕点(0) 有限率: slot0 {wrist[0] * 100:.1f}%  "
          f"slot1 {wrist[1] * 100:.1f}%（旧管线 38%/…）")
    print(f"翻面事件（z 跳变>{FLAP_M * 1000:.0f}mm）: {flaps} 起 "
          f"(≤{FLAP_OK}) {'PASS' if flaps <= FLAP_OK else 'FAIL'}（旧管线 7 起）")
    print(f"3D 完整性判据: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
