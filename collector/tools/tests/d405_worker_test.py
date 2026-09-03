"""D405 采集冒烟测试（无头，真机）。

本机接入 D405 后验证：按 serial 锁定设备、1280×720 原生深度流、
帧信号形状/dtype、时间戳非负、标定提取（baseline 应显著小于 D435 的
≈0.050 m，D405 为短基线近景相机）。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/d405_worker_test.py
"""
import os
import sys
import time

# 脚本直跑时把项目根加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QCoreApplication

from config import settings
from core.d435_camera import D435Worker, list_d400_devices

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def main():
    app = QCoreApplication(sys.argv)

    target = next(((n, s) for (n, s) in list_d400_devices() if "405" in n), None)
    if target is None:
        print("SKIP: 未连接 D405")
        return 0
    model_name, serial = target
    prof = settings.realsense_profile(model_name)
    depth_w, depth_h = prof["depth_resolution"]
    rgb_w, rgb_h = prof["rgb_resolution"]
    print(f"目标: {model_name} {serial} 深度 {depth_w}x{depth_h} "
          f"RGB {rgb_w}x{rgb_h}@{prof['fps']}")

    worker = D435Worker(width=depth_w, height=depth_h, fps=prof["fps"],
                        rgb_width=rgb_w, rgb_height=rgb_h,
                        serial=serial, model_name=model_name)
    seen = {}
    errors = []
    worker.frames_ready.connect(
        lambda sid, f, ns, imu: seen.setdefault(sid, []).append((f, ns)))
    worker.error_occurred.connect(errors.append)
    worker.start()

    t_end = time.time() + 10
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)
        if (len(seen.get(settings.D435_SLOT_DEPTH, [])) >= 5
                and len(seen.get(settings.D435_SLOT_RGB, [])) >= 5):
            break

    d_frames = seen.get(settings.D435_SLOT_DEPTH, [])
    r_frames = seen.get(settings.D435_SLOT_RGB, [])
    check(len(d_frames) >= 5, f"深度帧流量: {len(d_frames)}")
    check(len(r_frames) >= 5, f"RGB 帧流量: {len(r_frames)}")
    if d_frames:
        f0 = d_frames[0][0]
        check(f0.dtype == np.uint16 and f0.shape == (depth_h, depth_w),
              f"深度帧形状/dtype: {f0.shape} {f0.dtype}")
        check(f0.max() > 0, f"深度帧含有效值 (max={f0.max()})")
    if r_frames:
        f1 = r_frames[0][0]
        check(f1.shape == (rgb_h, rgb_w, 3), f"RGB 帧形状: {f1.shape}")
    hw_ns = d_frames[0][1] if d_frames else -1
    check(isinstance(hw_ns, int) and hw_ns >= 0, f"时间戳非负: {hw_ns}")
    check(not errors, f"无错误: {errors[:2]}")

    calib = worker.get_calibration()
    if calib is None:
        check(False, "标定未就绪")
    else:
        check(calib.baseline < 0.04,
              f"baseline {calib.baseline:.5f} m < 0.04（D405 短基线）")
        check(abs(calib.depth_scale - 0.001) < 1e-6,
              f"标定 depth_scale 为输出格式刻度: {calib.depth_scale}")
        print(f"  depth 内参: {calib.depth_camera.intrinsic}")
    # D405 原生 0.1mm/单位 → worker 归一化到 mm（全链路 mm 约定）
    check(abs(worker._depth_unit_factor - 0.1) < 1e-3,
          f"深度单位归一化系数 ≈0.1: {worker._depth_unit_factor}")

    worker.stop()
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: D405 采集冒烟测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
