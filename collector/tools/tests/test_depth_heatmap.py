"""depth_to_heatmap / DepthHeatmapSmoother 单元测试（合成数据，无硬件）。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/test_depth_heatmap.py

覆盖:
  - 固定色标: near/far 同给时整幅 clip 0..255（demo 口径）——
    near→JET(0)、far→JET(255)、超远饱和 JET(255)、无效/近端内→JET(0) 深蓝
  - 帧内自适应: near/far=0 时 min/max 映射（S80M 传统行为不变）
  - smooth_k 中值滤波杀单像素椒盐噪点
  - DepthHeatmapSmoother EMA 收敛 + 形状变化重置
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2

from core.stereo_depth import depth_to_heatmap, DepthHeatmapSmoother

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def gray(h, w, val):
    return np.full((h, w), val, dtype=np.uint16)


# JET 色表端点色（cv2.applyColorMap 的 0/1/255 输出，作期望值）
JET_0 = cv2.applyColorMap(np.array([[0]], dtype=np.uint8),
                          cv2.COLORMAP_JET)[0, 0]
JET_1 = cv2.applyColorMap(np.array([[1]], dtype=np.uint8),
                          cv2.COLORMAP_JET)[0, 0]
JET_255 = cv2.applyColorMap(np.array([[255]], dtype=np.uint8),
                            cv2.COLORMAP_JET)[0, 0]


def main():
    print("── 1. 固定色标（near/far 同给，demo 口径 clip 0..255） ──")
    d = np.zeros((8, 8), dtype=np.uint16)
    d[0, 0] = 100          # near → JET(0) 深蓝
    d[0, 1] = 1000         # far → JET(255) 红
    d[0, 2] = 2000         # 超远 → 饱和 JET(255)
    d[0, 5] = 50           # 近端内（<near）→ JET(0) 深蓝
    d[0, 4] = 200          # 范围内 → 非黑
    heat = depth_to_heatmap(d, near_mm=100, far_mm=1000)
    check((heat[0, 0] == JET_0).all(), f"near 映射到 JET(0): {heat[0,0]}")
    check((heat[0, 1] == JET_255).all(), f"far 映射到 JET(255): {heat[0,1]}")
    check((heat[0, 2] == JET_255).all(),
          f"超远饱和 JET(255) 不置黑: {heat[0,2]}")
    check((heat[0, 3] == JET_0).all(),
          f"无效值(0)→JET(0) 深蓝不置黑: {heat[0,3]}")
    check((heat[0, 5] == JET_0).all(),
          f"近端内(<near)→JET(0) 深蓝: {heat[0,5]}")
    check((heat[0, 4] != 0).any(), f"范围内非黑: {heat[0,4]}")

    # 相同深度值在不同内容帧中颜色一致（真固定色标）
    a = gray(4, 4, 550)
    b = gray(4, 4, 550)
    b[0, 0] = 1000   # 加入一个 far 极值
    b[0, 1] = 100    # 加入一个 near 极值
    ha = depth_to_heatmap(a, 100, 1000)
    hb = depth_to_heatmap(b, 100, 1000)
    check((ha[1, 1] == hb[1, 1]).all(),
          f"固定色标下同距离同色（内容无关）: {ha[1,1]} vs {hb[1,1]}")

    print("── 2. 帧内自适应（near/far=0，S80M 行为） ──")
    d2 = gray(4, 4, 200)
    d2[0, 0] = 500
    h2 = depth_to_heatmap(d2, 0, 0)
    check((h2[1, 1] == JET_1).all(), f"自适应 min→JET(1): {h2[1,1]}")
    check((h2[0, 0] == JET_255).all(), f"自适应 max→JET(255): {h2[0,0]}")

    print("── 3. smooth_k 中值滤波 ──")
    d3 = gray(5, 5, 500)
    d3[2, 2] = 990          # 单像素椒盐噪点
    h3 = depth_to_heatmap(d3, 100, 1000, smooth_k=3)
    check((h3[2, 2] == h3[2, 1]).all(),
          f"中值滤波后椒盐点被抹平: {h3[2,2]} vs {h3[2,1]}")

    print("── 4. DepthHeatmapSmoother EMA ──")
    sm = DepthHeatmapSmoother(alpha=0.5)
    f1 = np.zeros((4, 4, 3), dtype=np.uint8)
    f2 = np.full((4, 4, 3), 200, dtype=np.uint8)
    check((sm.update(f1) == f1).all(), "首帧原样输出")
    out = sm.update(f2)
    check(int(out[0, 0, 0]) == 100,
          f"EMA 0.5: 0→200 收敛到 100: {out[0,0,0]}")
    out2 = sm.update(f2)
    check(int(out2[0, 0, 0]) == 150,
          f"EMA 0.5: 再帧收敛到 150: {out2[0,0,0]}")
    big = np.zeros((8, 8, 3), dtype=np.uint8)
    check((sm.update(big) == big).all(), "形状变化自动重置")

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 深度热力图单元测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
