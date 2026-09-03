#!/usr/bin/env python3
"""从真机 D435 提取彩色内参 + depth→color 外参 → 固化到模块标定 JSON。

必须用 venv python（pyrealsense2 只装在 venv）：
    venv/bin/python \
        tools/hand_3d_d435/tools/extract_d435_color_calib.py [--session <session_dir>]

前置：设备已接入且未被主程序占用（主程序 D435 预览开启时设备被独占，
会报 "Device or resource busy"）。

外参语义（RealSense 约定）：P_color = R @ P_depth + t，t 单位米，
D435 深度→彩色横向平移典型 |t|≈25mm。录制链路不存彩色内参/外参，
离线对齐全靠本脚本固化的这份标定。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

MOD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(MOD_DIR, "calibration", "d435_color_calib.json")

COLOR_W, COLOR_H, DEPTH_W, DEPTH_H, FPS = 1280, 720, 848, 480, 30

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    print("错误: pyrealsense2 未安装。请用 venv python 运行本脚本：")
    print("  venv/bin/python tools/hand_3d_d435/tools/extract_d435_color_calib.py")
    sys.exit(1)


def _find_device():
    ctx = rs.context()
    for d in ctx.query_devices():
        if "D400" in d.get_info(rs.camera_info.product_line):
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", help="录制会话目录，与录制 head_stereo.json 深度内参交叉核对")
    ap.add_argument("--out", default=OUT_PATH, help=f"输出 JSON 路径（默认 {OUT_PATH}）")
    args = ap.parse_args()

    dev = _find_device()
    if dev is None:
        print("错误: 未检测到 D435 设备。请确认：")
        print("  1. 相机已接入 USB；")
        print("  2. 主程序未独占设备（打开 D435 预览时会占用）。")
        sys.exit(1)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)
    try:
        prof = pipe.start(cfg)
    except RuntimeError as e:
        print(f"错误: 打开设备失败（{e}）。")
        print("  若提示 Device or resource busy：主程序正在使用 D435，请先停止其预览。")
        sys.exit(1)

    try:
        fs = pipe.wait_for_frames(timeout_ms=5000)
        dp = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        cp = prof.get_stream(rs.stream.color).as_video_stream_profile()
        di, ci = dp.get_intrinsics(), cp.get_intrinsics()
        ext = dp.get_extrinsics_to(cp)          # P_color = R·P_depth + t（米）
        rot = ext.rotation                      # 2.58 API 是扁平 9 元组，旧版是 3×3
        if len(rot) == 9 and not isinstance(rot[0], (list, tuple)):
            rot = [[rot[0], rot[1], rot[2]],
                   [rot[3], rot[4], rot[5]],
                   [rot[6], rot[7], rot[8]]]
        sensor = prof.get_device().first_depth_sensor()

        out = {
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "extracted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "color_intrinsics": {
                "width": ci.width, "height": ci.height,
                "fx": ci.fx, "fy": ci.fy, "cx": ci.ppx, "cy": ci.ppy,
                "model": str(ci.model), "coeffs": [float(x) for x in ci.coeffs],
            },
            "depth_intrinsics": {
                "width": di.width, "height": di.height,
                "fx": di.fx, "fy": di.fy, "cx": di.ppx, "cy": di.ppy,
                "model": str(di.model), "coeffs": [float(x) for x in di.coeffs],
            },
            "depth_to_color": {
                "rotation": [[float(x) for x in row] for row in rot],
                "translation": [float(x) for x in ext.translation],
            },
            "depth_scale": float(sensor.get_depth_scale()),
        }

        # ── 交叉核对 ──
        t_norm = float(sum(x * x for x in ext.translation)) ** 0.5
        notes = []
        if not (0.01 <= t_norm <= 0.06):
            notes.append(f"depth→color 平移范数 {t_norm * 1000:.1f}mm 超出 [10,60]mm，"
                         f"疑似外参异常，请人工确认")
        if args.session:
            hp = os.path.join(args.session, "calibration", "head_stereo.json")
            if os.path.isfile(hp):
                with open(hp, encoding="utf-8") as f:
                    head = json.load(f)
                rec = head["depth_camera"]["intrinsic"]   # [fx, fy, cx, cy]
                for name, a, b in (("fx", rec[0], di.fx), ("fy", rec[1], di.fy),
                                   ("cx", rec[2], di.ppx), ("cy", rec[3], di.ppy)):
                    rel = abs(a - b) / max(abs(a), 1e-9)
                    if rel > 0.01:
                        notes.append(f"录制深度内参 {name}={a:.2f} 与设备 {b:.2f} 差 "
                                     f"{rel * 100:.1f}%——可能不是同一台设备录制，对齐精度受损")
                out["cross_check"] = {
                    "recorded_depth_intrinsic": list(rec),
                    "notes": notes,
                }
        if notes:
            print("警告:")
            for s in notes:
                print("  -", s)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"✓ 标定已固化: {args.out}")
        print(f"  serial={out['serial']} 彩色 fx={ci.fx:.3f} fy={ci.fy:.3f} "
              f"cx={ci.ppx:.3f} cy={ci.ppy:.3f}")
        print(f"  depth→color t=[{', '.join(f'{x:.4f}' for x in ext.translation)}] m "
              f"|t|={t_norm * 1000:.1f}mm")
        print(f"  深度内参 fx={di.fx:.3f} fy={di.fy:.3f} cx={di.ppx:.3f} cy={di.ppy:.3f}")
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
