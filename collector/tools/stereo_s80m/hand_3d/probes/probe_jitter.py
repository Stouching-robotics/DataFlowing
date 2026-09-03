#!/usr/bin/env python3
"""
P5 抖动验收探针：低速段 jitter 三列对比 + 快段保真 + 长缺口幻觉检查。

对 offline 模式跑出的 parquet（hand_3d=原始精修值、hand_3d_smoothed=
离线零相位平滑值），复现因果 One-Euro（与主循环 Hand3DSmoother 同参数
freq_min=3.0/beta=0.3，ts 用标称 25fps 帧间隔）作第三列。

指标（与 run_pipeline._DispTracker 同口径）：
  disp(t)   = 相邻帧共同有效点位移中位（帧内点中位）
  jitter(t) = |disp(t) − disp(t−1)|
速度分带（25fps）：低速 disp<1.2mm(30mm/s)、中速 1.2-6mm、快段 >6mm(150mm/s)
验收线：低速段 smoothed jitter 中位 <1.0mm；快段 |smoothed−raw| 中位 <1.5mm；
       长缺口（>15 帧）smoothed 保持 NaN 不幻觉。

用法: venv/bin/python stereo_s80m/hand_3d/probes/probe_jitter.py <parquet>
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)  # stereo_s80m/hand_detection 已并入 tools/ 命名空间

from hand_detection.hand_pipeline_mediapipe import OneEuroFilter3D  # noqa: E402

FPS = 25.0
N_HANDS, N_KPTS = 2, 21


def _load(path: str) -> tuple:
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    raw = np.stack([np.asarray(r["observation.keypoints.hand_3d"], np.float32)
                    .reshape(N_HANDS, N_KPTS, 3) for r in rows])
    has_sm = "observation.keypoints.hand_3d_smoothed" in rows[0]
    sm = (np.stack([np.asarray(r["observation.keypoints.hand_3d_smoothed"],
                               np.float32).reshape(N_HANDS, N_KPTS, 3)
                    for r in rows]) if has_sm else None)
    prop = np.stack([np.asarray(r["observation.keypoints.propagated"], np.bool_)
                     for r in rows])
    return raw, sm, prop


def _causal(raw: np.ndarray, labels: list) -> np.ndarray:
    """复现主循环因果 One-Euro（freq_min=3.0, beta=0.3，固定 40ms 帧间隔）。"""
    ts_ms = 1000.0 / FPS
    out = np.full_like(raw, np.nan, np.float32)
    filters = {}
    prev_label = [None, None]
    for t in range(len(raw)):
        for slot in range(2):
            if labels[t][slot] != prev_label[slot]:
                filters.pop(slot, None)
            prev_label[slot] = labels[t][slot]
            fs = filters.setdefault(slot, {})
            for k in range(N_KPTS):
                p = raw[t, slot, k]
                if not np.all(np.isfinite(p)):
                    continue
                f = fs.get(k)
                if f is None:
                    f = OneEuroFilter3D(3.0, 0.3, 1.0)
                    fs[k] = f
                out[t, slot, k] = f(p[0], p[1], p[2], (t + 1) * ts_ms)
    return out


def _band_stats(name: str, col: np.ndarray, mask: np.ndarray,
                raw: np.ndarray = None, disp: np.ndarray = None) -> None:
    """band mask（帧级）→ jitter 中位 + 可选对 raw 的保真中位。"""
    idx = np.where(mask)[0]
    n = len(idx)
    if n < 3:
        print(f"  [{name}] 帧数不足({n})，跳过")
        return
    jit = np.abs(np.diff(disp[idx]))              # disp 已按槽位逐帧
    line = f"  [{name}] n={n}  jitter中位={np.median(jit)*1000:.2f}mm " \
           f"p90={np.percentile(jit, 90)*1000:.2f}mm"
    if raw is not None:
        d = np.abs(col - raw)
        fin = np.isfinite(d)
        if fin.any():
            line += f"  |Δvs raw|中位={np.nanmedian(d[fin])*1000:.2f}mm"
    print(line)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not os.path.isfile(path):
        print(f"用法: {sys.argv[0]} <parquet>")
        return 1
    raw, sm, prop = _load(path)
    n = len(raw)
    import pyarrow.parquet as pq
    rows_meta = pq.read_table(path).to_pylist()
    labels = [[r["observation.keypoints.hand_0_label"],
               r["observation.keypoints.hand_1_label"]] for r in rows_meta]

    print(f"parquet: {path}  ({n} 帧)")
    if sm is not None:
        print("hand_3d_smoothed 列存在 ✓")

    # ── 传播/长缺口检查 ──
    n_prop0, n_prop1 = int(prop[:, 0].sum()), int(prop[:, 1].sum())
    print(f"propagated 帧: hand_0={n_prop0}  hand_1={n_prop1}")
    if sm is not None:
        for slot, name in ((0, "hand_0"), (1, "hand_1")):
            # 长缺口（>15 帧连续无效）不得被平滑幻觉
            finite = np.isfinite(sm[:, slot]).all(axis=(1, 2))
            runs = _runs_out(~finite)
            long = [r for r in runs if r[1] - r[0] + 1 > 15]
            for a, b in long:
                print(f"  [长缺口] {name} 帧 {a}-{b} ({b-a+1}帧) smoothed "
                      f"{'全 NaN ✓ 不幻觉' if not finite[a:b+1].any() else '!!! 有幻觉值 !!!'}")
            if not long:
                print(f"  [{name}] 无 >15 帧长缺口")

    # ── 三列 ──
    causal = _causal(raw, labels)
    cols = [("raw", raw), ("causal复现", causal)]
    if sm is not None:
        cols.append(("smoothed", sm))

    for slot, name in ((0, "hand_0"), (1, "hand_1")):
        # disp：帧间共同有效点位移中位（≥8 点才计）
        disp = np.full(n, np.nan)
        for t in range(1, n):
            a, b = raw[t - 1, slot], raw[t, slot]
            ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            if ok.sum() >= 8:
                disp[t] = np.median(np.linalg.norm(a[ok] - b[ok], axis=1))
        valid = np.isfinite(disp)
        lo = valid & (disp < 0.002)        # <50mm/s（3D 单位米：2mm/帧）
        mid = valid & (disp >= 0.002) & (disp <= 0.006)
        fast = valid & (disp > 0.006)      # >150mm/s
        qs = np.nanpercentile(disp, [10, 50, 90]) * 1000
        print(f"\n── {name} ── 低速帧 {lo.sum()} / 中速 {mid.sum()} / 快段 {fast.sum()}  "
              f"disp p10/50/90={qs[0]:.1f}/{qs[1]:.1f}/{qs[2]:.1f}mm")
        for tag, col in cols:
            if tag == "raw":
                print(f"  [{tag}] 帧间位移中位={np.nanmedian(disp)*1000:.2f}mm")
            for band, mask in (("低速<50mm/s", lo), ("中速", mid), ("快段>150mm/s", fast)):
                _band_stats(f"{tag} {band}", col, mask,
                            raw=raw if tag != "raw" else None, disp=disp)
    return 0


def _runs_out(mask: np.ndarray) -> list:
    """mask 为 True 的连续段 → [(a,b)]（探针自用，与 postprocess._runs 反义）。"""
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


# ── 合成静止手测试 ────────────────────────────────────────────

def synth_still(n: int = 200, sigma_mm: float = 1.2, seed: int = 7):
    """真实会话没有"手静止"片段（两会话 p5 位移 ≥2.1mm/帧），用合成信号
    验证平滑数学。

    噪声模型来自 P2 实测：GPU delegate 2D 抖动中位 0.72px（1280×800，
    fx_rect=362）→ z=0.5m 处 3D σ ≈ 0.72px×0.5m/362px×1.48 ≈ 1.2mm。
    抖动口径 = 静止窗口内逐点逐轴去趋势 std（用户所见"骨架哆嗦"幅度）。
    """
    from scipy.signal import savgol_filter
    sigma = sigma_mm / 1000.0
    rng = np.random.default_rng(seed)
    truth = np.tile(np.linspace(0, 1, 21)[:, None], (1, 3)) * 0.12   # (21,3) 静止姿态
    sig = np.tile(truth[None], (n, 1, 1)) + rng.normal(0, sigma, (n, 21, 3))
    # 快段：帧 120-160 匀速 40mm/帧 平移（台阶后恢复静止）
    sig[120:161] += np.arange(1, 42)[:, None, None] * 0.04
    sig[161:] += 42 * 0.04

    # 复刻 offline_smooth 新算法：sg 去噪输出测速 + v0=80mm/s + 静止长窗 21
    # + 跳变阻尼（帧间位移 >10mm/帧 的突变点 ±10 帧内保 raw，防长窗预振铃）
    out = sig.copy()
    fps = 25.0
    half = 21 // 2
    for k in range(21):
        for ax in range(3):
            seg = sig[:, k, ax]
            sg7 = savgol_filter(seg, 7, 3, mode="interp")
            sg21 = savgol_filter(seg, 21, 3, mode="interp")
            vel = np.abs(np.gradient(sg21)) * fps
            alpha = np.clip((0.08 - vel) / 0.06, 0.0, 1.0)
            sg = alpha * sg21 + (1.0 - alpha) * sg7
            wgt = np.exp(-(vel / 0.08) ** 2)
            jumps = np.where(np.abs(np.diff(seg)) > 0.010)[0]
            if jumps.size:
                t = np.arange(len(seg))
                near = np.zeros(len(seg), bool)
                for j in jumps:
                    near |= (np.abs(t - j) <= half) | (np.abs(t - (j + 1)) <= half)
                wgt = np.where(near, 0.0, wgt)
            out[:, k, ax] = wgt * sg + (1.0 - wgt) * seg

    def _wobble(x, a, b):
        """静止段逐点逐轴去均值 std（mm）。"""
        seg = x[a:b + 1]
        return float(np.nanstd(seg - seg.mean(axis=0)) * 1000)

    def _jit(x, a, b):
        """段内 disp 一阶差分中位（与 _DispTracker 同口径）。"""
        d = np.array([np.median(np.linalg.norm(x[t] - x[t - 1], axis=1))
                      for t in range(a + 1, b + 1)])
        return np.median(np.abs(np.diff(d))) * 1000

    still, fast = (0, 119), (120, 161)
    print(f"\n── 合成静止手（噪声 σ={sigma_mm}mm 来自 P2 GPU 2D 抖动实测）──")
    print(f"  静止段抖动(std): raw={_wobble(sig, *still):.2f}mm  "
          f"smoothed={_wobble(out, *still):.2f}mm  "
          f"(目标 <1.0mm)")
    print(f"  静止段 jitter(disp差): raw={_jit(sig, *still):.2f}mm  "
          f"smoothed={_jit(out, *still):.2f}mm")
    lag = np.median(np.abs(out[fast[1] + 1] - sig[fast[1] + 1])) * 1000
    print(f"  快段台阶后 1 帧零相位残差 中位={lag:.2f}mm（因果滤波会 >20mm 滞后）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--synth":
        sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
        sys.exit(synth_still(sigma_mm=sigma))
    sys.exit(main())
