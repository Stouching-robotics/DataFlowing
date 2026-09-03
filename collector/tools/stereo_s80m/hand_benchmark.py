#!/usr/bin/env python3
"""
S80M 手势检测参数基准测试 —— 分辨率 × 帧率 × 颜色 × 单双目

对会话视频离线跑 MediaPipe 手部关键点，扫参数组合，输出稳定性指标矩阵:

    分辨率   (1280×800) (960×600) (640×400) (480×300)
    帧率     25 / 12 / 6 / 3 fps (隔帧采样)
    颜色     rgb / gray (灰度 = GRAY2BGR 三通道复制, 信息等价)
    模式     mono(左目单目) / stereo(左右目 + 出厂标定三角化)

指标 (smooth 关闭, 测管线原始稳定性; One-Euro 平滑是后处理):
    det_rate       检出率    至少一帧只手 score≥0.5 的帧占比
    det2_rate      双手检出率
    mean_score     检出手平均置信度
    max_miss       最大连续未检出段 (视频原始帧数, 跨帧率可比)
    disp           相邻检出帧关键点位移中位 (归一化坐标×1000)
    jitter         位移序列一阶差分中位 (去趋势后的高频抖动残差)
    tri_rate       双目: 有效三角化帧占比 (valid_count≥8)
    tri_err        双目: 平均重投影误差 (px)
    ms_frame       处理耗时 (ms/帧, 单目只算左目)

用法:
    python stereo_s80m/hand_benchmark.py --session data/recordings/222/222_000008
    python stereo_s80m/hand_benchmark.py --session ... --workers 8 --out <dir>
"""

import argparse
import csv
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
sys.path.insert(0, _TOOLS_DIR)      # stereo_s80m/hand_detection 已并入 tools/ 命名空间

from hand_detection.hand_pipeline_mediapipe import MediaPipeHandPipeline  # noqa: E402
from stereo_s80m.stereo_triangulate import (                              # noqa: E402
    StereoTriangulator,
    load_stereo_calibration,
    match_hands,
)

MODEL_PATH = os.path.join(_REPO_ROOT, "tools", "models", "hand_landmarker.task")
DEFAULT_CALIB = os.path.join(_REPO_ROOT, "config", "s80m_stereo_calibration.json")

RESOLUTIONS = [(1280, 800), (960, 600), (640, 400), (480, 300)]
FPS_STEPS = [(25, 1), (12, 2), (6, 4), (3, 8)]     # (fps, 隔帧数)
COLORS = ["rgb", "gray"]
MODES = ["mono", "stereo"]


# ── 单组测试 (worker 进程内执行) ────────────────────────────────

def run_group(args):
    session, calib_path, swap, (w, h), (fps, step), color, mode = args
    vpl = os.path.join(session, "videos", "stereo_left", "chunk-0000",
                       "stereo_left.mp4")
    vpr = os.path.join(session, "videos", "stereo_right", "chunk-0000",
                       "stereo_right.mp4")

    # 左右目各自独立的 pipeline：共享实例交替喂帧会污染 VIDEO 模式的
    # 追踪先验（上帧是另一只眼的画面），导致双手检出率和三角化全部退化
    pipe_l = MediaPipeHandPipeline(model_path=MODEL_PATH, num_hands=2,
                                   mirror=False, smooth=False)  # 与 demo 一致
    pipe_r = None
    if mode == "stereo":
        pipe_r = MediaPipeHandPipeline(model_path=MODEL_PATH, num_hands=2,
                                       mirror=False, smooth=False)
    cap_l = cv2.VideoCapture(vpl)
    cap_r = cv2.VideoCapture(vpr) if mode == "stereo" else None

    tri = None
    if mode == "stereo":
        calib = load_stereo_calibration(calib_path=calib_path)
        if calib is None:
            return None
        tri = StereoTriangulator(calib, image_size=(w, h), swap_cams=swap)

    def prep(cap, frame):
        if step > 1:
            for _ in range(step - 1):
                cap.grab()
        ok, fr = cap.read()
        if not ok:
            return None
        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        if color == "gray":
            fr = cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),
                              cv2.COLOR_GRAY2BGR)
        return fr

    stats = {
        "n": 0, "det": 0, "det2": 0, "score": [], "miss_run": 0, "max_miss": 0,
        "disp": [], "jitter": [], "tri_ok": 0, "tri_err": [],
    }
    prev_lms = {}       # hand index → 上一检出帧 norm 关键点 (21,2)
    prev_disp = {}      # hand index → 上一次帧间位移中位
    t0 = time.perf_counter()

    while True:
        fl = prep(cap_l, None)
        if fl is None:
            break
        rl = pipe_l.process(fl)
        stats["n"] += 1

        det_hands = [h for h in rl.hands if h.score >= 0.5]
        if not det_hands:
            stats["miss_run"] += 1
            stats["max_miss"] = max(stats["max_miss"], stats["miss_run"])
            prev_lms.clear()
            prev_disp.clear()
            continue
        stats["miss_run"] = 0
        stats["det"] += 1
        if len(det_hands) >= 2:
            stats["det2"] += 1
        stats["score"].append(np.mean([h.score for h in det_hands]))

        # 帧间稳定性: 同 index 手的关键点归一化位移 + 一阶差分(抖动)
        cur = {h.index: np.array(h.norm_landmarks, dtype=np.float64)
               for h in det_hands}
        for idx, lm in cur.items():
            if idx in prev_lms:
                d = np.linalg.norm(lm - prev_lms[idx], axis=1)  # (21,)
                dmed = np.median(d)
                stats["disp"].append(dmed)
                if idx in prev_disp:
                    stats["jitter"].append(abs(dmed - prev_disp[idx]))
                prev_disp[idx] = dmed
        prev_lms = cur

        # 双目三角化
        if tri is not None:
            fr_r = prep(cap_r, None)
            if fr_r is None:
                break
            rr = pipe_r.process(fr_r)
            pairs = match_hands([h for h in rl.hands if h.score >= 0.5],
                                [h for h in rr.hands if h.score >= 0.5], tri)
            if pairs:
                stats["tri_ok"] += 1
                for p in pairs:
                    if p.result.valid_count >= 8 and p.result.mean_error < np.inf:
                        stats["tri_err"].append(p.result.mean_error)

    elapsed = time.perf_counter() - t0
    cap_l.release()
    if cap_r:
        cap_r.release()
    pipe_l.close()
    if pipe_r:
        pipe_r.close()

    n = max(stats["n"], 1)
    disp = np.array(stats["disp"]) if stats["disp"] else np.array([np.nan])
    jit = np.array(stats["jitter"]) if stats["jitter"] else np.array([np.nan])
    tri_err = np.array(stats["tri_err"]) if stats["tri_err"] else np.array([np.nan])
    return {
        "w": w, "h": h, "fps": fps, "color": color, "mode": mode,
        "n": stats["n"],
        "det_rate": stats["det"] / n,
        "det2_rate": stats["det2"] / n,
        "mean_score": float(np.mean(stats["score"])) if stats["score"] else 0.0,
        "max_miss": stats["max_miss"],
        "disp_med": float(np.nanmedian(disp)),
        "jitter_med": float(np.nanmedian(jit)),
        "tri_rate": stats["tri_ok"] / n if tri else float("nan"),
        "tri_err": float(np.nanmedian(tri_err)) if tri else float("nan"),
        "ms_frame": elapsed * 1000 / n,
    }


# ── 左右目配对自检 (stereo_swap_lr 隐患) ───────────────────────

def _detect_swap(session: str, calib_path: str, size=(640, 400)) -> bool:
    """ORB 极线误差法: 两种配对哪边 |y差| 中位数小用哪边。"""
    w, h = size
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    calib = load_stereo_calibration(calib_path=calib_path)
    if calib is None:
        return False

    def score(swap):
        tri = StereoTriangulator(calib, image_size=(w, h), swap_cams=swap)
        cap_l = cv2.VideoCapture(os.path.join(session, "videos", "stereo_left",
                                              "chunk-0000", "stereo_left.mp4"))
        cap_r = cv2.VideoCapture(os.path.join(session, "videos", "stereo_right",
                                              "chunk-0000", "stereo_right.mp4"))
        ys = []
        for _ in range(6):
            ok0, f0 = cap_l.read()
            ok1, f1 = cap_r.read()
            if not (ok0 and ok1):
                break
            l = cv2.resize(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY), (w, h),
                           interpolation=cv2.INTER_AREA)
            r = cv2.resize(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), (w, h),
                           interpolation=cv2.INTER_AREA)
            lr = cv2.remap(l, tri._maps["left"][0], tri._maps["left"][1],
                           cv2.INTER_LINEAR)
            rr = cv2.remap(r, tri._maps["right"][0], tri._maps["right"][1],
                           cv2.INTER_LINEAR)
            kl, dl = orb.detectAndCompute(lr, None)
            kr, dr = orb.detectAndCompute(rr, None)
            m = bf.knnMatch(dl, dr, k=2)
            for mm, nn in m:
                if mm.distance < 0.75 * nn.distance:
                    ys.append(kl[mm.queryIdx].pt[1] - kr[mm.trainIdx].pt[1])
        cap_l.release()
        cap_r.release()
        return np.median(np.abs(ys)) if ys else float("inf")

    s0, s1 = score(False), score(True)
    swap = s1 + 0.5 < s0
    print(f"  [方向自检] 常规 {s0:.2f}px  交换 {s1:.2f}px → "
          f"{'需要交换' if swap else '常规配对正确'}", flush=True)
    return swap


# ── 主流程 ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="CSV 输出目录 (默认 <script_dir>/hand_benchmark)")
    ap.add_argument("--modes", default="all", choices=["all", "mono", "stereo"],
                    help="只跑指定模式 (默认 all)")
    args = ap.parse_args()

    session = os.path.abspath(args.session)
    if not os.path.isdir(session):
        sys.exit(f"[错误] 会话目录不存在: {session}")

    swap = _detect_swap(session, args.calib)

    combos = list(itertools.product(RESOLUTIONS, FPS_STEPS, COLORS, MODES))
    if args.modes != "all":
        combos = [c for c in combos if c[3] == args.modes]
    tasks = [(session, args.calib, swap, res, fps_step, color, mode)
             for res, fps_step, color, mode in combos]
    print(f"共 {len(tasks)} 组参数组合, workers={args.workers}, "
          f"swap={swap}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(run_group, tasks, chunksize=1)):
            if r is not None:
                results.append(r)
            if (i + 1) % 8 == 0 or i == len(tasks) - 1:
                print(f"  {i + 1}/{len(tasks)} 组完成", flush=True)

    # ── 输出 CSV ──
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hand_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(session)
    suffix = "" if args.modes == "all" else f"_{args.modes}"
    csv_path = os.path.join(out_dir, f"benchmark_{tag}{suffix}.csv")
    cols = ["w", "h", "fps", "color", "mode", "n", "det_rate", "det2_rate",
            "mean_score", "max_miss", "disp_med", "jitter_med", "tri_rate",
            "tri_err", "ms_frame"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in cols})
    print(f"✓ CSV: {csv_path}")

    # ── 表格 ──
    print("\n=== 结果矩阵 ===")
    hdr = (f"{'res':>9} {'fps':>3} {'色':>4} {'模式':>6} "
           f"{'检出率':>6} {'双手':>5} {'置信':>5} {'最大漏检':>5} "
           f"{'位移×1e3':>7} {'抖动×1e3':>7} {'三角化':>6} {'重投影':>6} "
           f"{'ms/帧':>6}")
    print(hdr)
    for r in sorted(results, key=lambda x: (-x["det_rate"], x["jitter_med"])):
        tri_s = f"{r['tri_rate']*100:.0f}%" if r["mode"] == "stereo" else "  -  "
        err_s = f"{r['tri_err']:.1f}" if r["mode"] == "stereo" else "  -  "
        print(f"{r['w']:>4}x{r['h']:<4} {r['fps']:>3} "
              f"{r['color']:>4} {r['mode']:>6} "
              f"{r['det_rate']*100:5.1f}% {r['det2_rate']*100:4.1f}% "
              f"{r['mean_score']:5.2f} {r['max_miss']:>5} "
              f"{r['disp_med']*1000:7.1f} {r['jitter_med']*1000:7.1f} "
              f"{tri_s:>6} {err_s:>6} {r['ms_frame']:6.1f}")


if __name__ == "__main__":
    main()
