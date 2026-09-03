#!/usr/bin/env python3
"""probe_bone_lengths.py —— 腕→中指MCP 骨长自检（标定无关硬判据）。

对 parquet 的 hand_3d 原始值（剔除 propagated 传播帧）逐槽统计骨长：
判据（双目管线验收经验值，中位带按 D435 正面手修正）：
  每槽中位 ∈ [72,95]mm、IQR<25mm、两槽中位差<10mm、<5% 帧出 [55,115]mm。

中位带从双目经验 [75,95] 放宽下界到 72：D435 单目补点（lift3d.complete）
把缺深腕点补到手深平面（zc），该帧骨长即正面投影值（实测 slot1 正面
投影 73.5mm、3D 混合中位 74.6mm，自洽）；双手持姿不同（slot0 腕更深
→ 82.8mm）也在带内。硬界 [55,115] 不变。

用法（venv）:
  python tools/hand_3d_d435/probes/probe_bone_lengths.py --parquet <chunk-000.parquet>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


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
    medians = []
    for s, name in enumerate(("slot0", "slot1")):
        present = t[f"observation.keypoints.hand_{s}_present"].to_numpy()
        real = present & ~prop[:, s]
        w = h3[real, s, 0, :]
        m = h3[real, s, 9, :]
        ok = np.isfinite(w).all(axis=1) & np.isfinite(m).all(axis=1)
        b = np.linalg.norm(w[ok] - m[ok], axis=1) * 1000.0     # mm
        if len(b) == 0:
            print(f"{name}: 无有效真检测帧")
            all_ok = False
            continue
        med = float(np.median(b))
        iqr = float(np.percentile(b, 75) - np.percentile(b, 25))
        out_frac = float(np.mean((b < 55.0) | (b > 115.0)))
        medians.append(med)
        good = (72.0 <= med <= 95.0 and iqr < 25.0 and out_frac < 0.05)
        all_ok &= good
        print(f"{name}: n={len(b)} 中位 {med:.1f}mm IQR {iqr:.1f}mm "
              f"出[55,115] {out_frac * 100:.1f}%  "
              f"{'PASS' if good else 'FAIL'}")
    if len(medians) == 2:
        diff = abs(medians[0] - medians[1])
        print(f"两槽中位差 {diff:.1f}mm {'PASS' if diff < 10 else 'FAIL'}")
        all_ok &= diff < 10
    print(f"\n骨长判据: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
