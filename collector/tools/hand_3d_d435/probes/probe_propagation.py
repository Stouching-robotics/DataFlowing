#!/usr/bin/env python3
"""probe_propagation.py —— propagated 比例 + 缺口长度直方图。

判据：每槽 propagated <15%；absent 缺口全 ≤15 帧（max_lost 硬顶），
超限段 3D 全 NaN 不幻觉（可数 '硬 absent' 帧验证）。

用法（venv）:
  python tools/hand_3d_d435/probes/probe_propagation.py --parquet <chunk-000.parquet>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _runs(mask: np.ndarray) -> list:
    """bool 数组 → 连续 True 段长度列表。"""
    out = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _hist(runs: list) -> str:
    bins = {"1": 0, "2-5": 0, "6-10": 0, "11-15": 0, ">15": 0}
    for r in runs:
        if r == 1:
            bins["1"] += 1
        elif r <= 5:
            bins["2-5"] += 1
        elif r <= 10:
            bins["6-10"] += 1
        elif r <= 15:
            bins["11-15"] += 1
        else:
            bins[">15"] += 1
    return ", ".join(f"{k}:{v}" for k, v in bins.items())


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
    h3 = np.asarray(t["observation.keypoints.hand_3d"].to_pylist(),
                    np.float32).reshape(n, 2, 21, 3)
    prop = np.asarray(t["observation.keypoints.propagated"].to_pylist(),
                      bool).reshape(n, 2)

    all_ok = True
    for s, name in enumerate(("slot0", "slot1")):
        present = t[f"observation.keypoints.hand_{s}_present"].to_numpy()
        n_pres = int(present.sum())
        n_prop = int((present & prop[:, s]).sum())
        frac = n_prop / n_pres if n_pres else 0.0
        absent_runs = _runs(~present)
        max_gap = max(absent_runs) if absent_runs else 0
        # 硬 absent：present=False 且 3D 全 NaN（未幻觉）
        hard = int(np.sum(~present
                          & ~np.isfinite(h3[:, s]).any(axis=(1, 2))))
        good = frac < 0.15 and max_gap <= 15
        all_ok &= good
        print(f"{name}: present {n_pres}/{n} propagated {n_prop} "
              f"({frac * 100:.1f}%)")
        print(f"  absent 缺口 [{_hist(absent_runs)}] 最大 {max_gap} 帧"
              f" | 硬 absent(全NaN) {hard} 帧 | "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\n传播判据: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
