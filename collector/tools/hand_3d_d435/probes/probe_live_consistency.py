#!/usr/bin/env python3
"""probe_live_consistency.py —— live_demo 在线链路 vs 离线 parquet 一致性探针。

重跑 live_demo 的逐帧链（--fill 1 轮填洞 + tracker αβ + Hand3DSmoother
OneEuro），逐帧与离线产物 hand_3d_smoothed（fill_gaps + savgol，3 轮填洞）
比对。两路差异来源：填洞 1 vs 3 轮（同点 3×3 中位采样几乎不受影响，亚 mm 级）、
αβ+OneEuro vs 零相位 savgol（平滑器不同，静止段差 <2mm）、tracker 对缺失点
保持纯预测（离线那格是 NaN，不在比对集内）。

判据：两者都有限的手腕（点 0）深度差中位 <10mm、p95 <30mm。

用法（venv）: python tools/hand_3d_d435/probes/probe_live_consistency.py <session>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import pyarrow.parquet as pq                                     # noqa: E402

from hand_3d_d435.live_demo import (LiveAligner, ReplaySource,     # noqa: E402
                                    _nan_pair, _pred_pair)
from hand_3d_d435.depth_align import (load_calib,                 # noqa: E402
                                      load_session_depth_intr)
from hand_3d_d435.lift3d import lift_hand, gate_observations    # noqa: E402
from hand_3d_d435.mono_assign import assign_mono_slots          # noqa: E402
from stereo_s80m.hand_3d.detector import MediaPipeDetector       # noqa: E402
from stereo_s80m.hand_3d.identity import HandednessVoter         # noqa: E402
from stereo_s80m.hand_3d.track3d import HandSlotTracker          # noqa: E402
from stereo_s80m.hand_3d.smoother import Hand3DSmoother          # noqa: E402

MED_OK = 10.0       # 腕深差中位上限（mm）
P95_OK = 30.0       # 腕深差 p95 上限（mm）


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", help="录制会话目录")
    ap.add_argument("--calib", help="标定 JSON（默认 hand_3d_d435/calibration/）")
    args = ap.parse_args()

    session = args.session.rstrip("/")
    parquet = os.path.join(_REPO_ROOT, "keypoints_output",
                           os.path.basename(os.path.dirname(session)),
                           os.path.basename(session),
                           "hand_3d_refined", "chunk-000.parquet")
    if not os.path.isfile(parquet):
        sys.exit(f"错误: 离线 parquet 不存在: {parquet}")
    t = pq.read_table(parquet)
    off = np.array(t["observation.keypoints.hand_3d_smoothed"]
                   .to_pylist()).reshape(-1, 2, 21, 3)      # (T,2,21,3) m

    calib = load_calib(args.calib)
    sd = load_session_depth_intr(session)
    if sd is None:
        sd = calib["depth_intrinsics"]
    aligner = LiveAligner(calib["color_intrinsics"],
                          calib["depth_to_color"], sd, fill_passes=1)
    color_intr = (aligner.fx_c, aligner.fy_c, aligner.cx_c, aligner.cy_c)

    source = ReplaySource(session, pace=0)
    det = MediaPipeDetector(num_hands=2, delegate="cpu")
    voter = HandednessVoter()
    tracker = HandSlotTracker(max_lost=15)
    smoother = Hand3DSmoother()

    live = np.full_like(off, np.nan)
    lost_counts = [0, 0]
    n = 0
    try:
        while True:
            rgb, d = source.next()
            if rgb is None:
                break
            if d is None or d.shape[:2] != (aligner.dh, aligner.dw):
                aligned = np.zeros((aligner.ch, aligner.cw), np.float32)
            else:
                aligned = aligner.align_depth_to_color(d)

            hands = det.detect(rgb)
            if hands:
                voter.update(hands, frame_w=rgb.shape[1],
                             frame_h=rgb.shape[0], frame=n, cam="d435")
            pairs = [lift_hand(hd, aligner, aligned) for hd in hands]
            out = assign_mono_slots(pairs, tracker, n, color_intr,
                                    lost_counts=tuple(lost_counts))
            slot_pairs = []
            for s in range(2):
                if out[s] is not None:
                    p = out[s]
                    gated, wholesale = gate_observations(
                        p.result.points_3d, tracker.predict(s, n))
                    if wholesale:
                        tracker.observe_slot(s, "\x00reset",
                                             np.full((21, 3), np.nan), n)
                    tracker.observe_slot(s, p.left_label, gated, n)
                    p.result.points_3d = gated
                    lost_counts[s] = 0
                    slot_pairs.append(p)
                else:
                    pred = tracker.predict(s, n)
                    tracker.mark_lost(s, n)
                    lost_counts[s] += 1
                    if pred is not None:
                        slot_pairs.append(_pred_pair(pred,
                                                     tracker.slot_label(s)))
                    else:
                        slot_pairs.append(_nan_pair(tracker.slot_label(s)))
            labels = [slot_pairs[s].left_label if out[s] is not None
                      else tracker.slot_label(s) for s in range(2)]
            h3 = np.stack([np.asarray(p.result.points_3d, np.float64)
                           .reshape(21, 3) for p in slot_pairs])
            valids = [int(np.isfinite(h3[s]).all(axis=1).sum())
                      for s in range(2)]
            live[n] = smoother.update(h3, labels, valids)
            n += 1
    finally:
        source.close()
        det.close()

    # 比对：两者都有限的手腕（点 0）深度（m → mm）
    both = np.isfinite(off[:n, :, 0, 2]) & np.isfinite(live[:n, :, 0, 2])
    n_both = int(both.sum())
    if n_both == 0:
        print("无共同有效帧，无法比对（离线腕点 NaN 62% 帧，此为该数据已知缺陷）")
        sys.exit(1)
    d = np.abs(off[:n, :, 0, 2] - live[:n, :, 0, 2])[both] * 1000.0
    med, p95 = float(np.median(d)), float(np.percentile(d, 95))
    ok = med < MED_OK and p95 < P95_OK
    print(f"在线链路 {n} 帧跑完；腕深共同有效 {n_both} 帧（离线腕 NaN "
          f"{n * 2 - n_both}/{n * 2} 槽-帧）")
    print(f"腕深差: 中位 {med:.2f}mm (≤{MED_OK:.0f}mm)  "
          f"p95 {p95:.2f}mm (≤{P95_OK:.0f}mm)  →  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
