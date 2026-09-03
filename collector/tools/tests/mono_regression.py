"""单目模式回归测试（offscreen,真机摄像头）—— 验证 D435 改动不影响单目链路:
自动检测 → _add_camera_slot 真实路径 → 录制 5s → MP4 落盘、无深度目录
(DEPTH_ENABLED=False 默认行为不变)、时间戳单调。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/mono_regression.py
"""
import os
import sys
import json
import glob
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyarrow.parquet as pq
from PyQt5.QtWidgets import QApplication

from config import settings
from core.camera import detect_cameras
from ui.main_window import MainWindow
from core.helpers import (episode_video_files, task_dir_of, pooled_info_path,
                           pooled_data_parquet_path)

OUT_ROOT = "/tmp/mono_reg_test"
DURATION = 5.0


def wait(app, ms):
    t_end = time.time() + ms / 1000.0
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)


def main():
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    app = QApplication(sys.argv)
    assert settings.DEPTH_ENABLED is False, "默认 DEPTH_ENABLED 应为 False"

    cameras = detect_cameras(settings.MAX_CAMERAS)
    if not cameras:
        print("SKIP: 未检测到单目摄像头（无设备可回归）")
        return 0
    print(f"检测到: {cameras}")

    win = MainWindow()
    win.show()
    win._pipeline._output_dir = OUT_ROOT

    # 直接添加槽位（扫描按钮已改为仅刷新设备面板，不自动接入）
    for idx, backend in cameras:
        slot_id = settings._camera_slot_name(idx)
        if slot_id not in win.grid.slot_ids():
            win._add_camera_slot(slot_id, idx, backend)
    wait(app, 2000)  # 等 CameraWorker 首帧
    slots = set(win.grid.slot_ids())
    print(f"grid 槽位: {sorted(slots)}")
    if not slots:
        print("FAIL: 未添加任何单目槽位")
        win.close()
        return 1

    win._task_name_input.setText("")
    win._record_all()
    if not win._pipeline.is_recording:
        print("FAIL: 录制未启动")
        win.close()
        return 1
    wait(app, DURATION * 1000)
    win._stop_all()

    deadline = time.time() + 15
    while time.time() < deadline:
        app.processEvents()
        if not win._pipeline._session_path:
            break
        time.sleep(0.05)

    ep = task_dir_of(OUT_ROOT, "")
    if not os.path.isfile(pooled_info_path(ep)):
        print("FAIL: 未生成任务目录")
        win.close()
        return 1
    print(f"task: {ep}")

    errors = []

    # 1. 每槽位 MP4（池化 videos/chunk-000/<slot>/episode-000.mp4）
    files0 = episode_video_files(ep, 1)
    for sid in sorted(slots):
        v = files0.get(sid)
        print(f"  videos/{sid}: {'OK' if v else '缺失'}")
        if not v or not os.path.isfile(v):
            errors.append(f"{sid} MP4 缺失")

    # 2. DEPTH_ENABLED=False → videos 下无深度槽目录、无深度相机元数据
    depth_dirs = [d for d in os.listdir(os.path.join(ep, "videos", "chunk-000"))
                  if "depth" in d]
    if depth_dirs:
        print(f"  videos 深度目录: {depth_dirs}")
        errors.append("DEPTH_ENABLED=False 时不应有深度槽目录")
    else:
        print("  videos 深度目录: 无（符合 DEPTH_ENABLED=False）")

    meta = json.load(open(pooled_info_path(ep), "r", encoding="utf-8"))
    depth_cams = [name for name, info in meta.get("cameras", {}).items()
                  if info.get("type") == "depth" or "depth" in name]
    if depth_cams:
        errors.append(f"info.json 不应含深度相机: {depth_cams}")

    # 3. 时间戳单调非负（单目无硬件时钟 → 用会话相对 timestamp）
    t = pq.read_table(pooled_data_parquet_path(ep, 1))
    tt = np.asarray(t.column("timestamp").to_pylist(), dtype=np.float64)
    mono = bool(np.all(np.diff(tt) >= 0))
    neg = int((tt < 0).sum())
    print(f"  timestamp: {len(tt)} 行, 单调={mono}, 负值={neg}")
    if not mono or neg:
        errors.append("时间戳非单调或含负值")

    # 4. 标定: 单目写默认 head_stereo 为旧行为(writer 兜底默认值),
    #    但不允许出现 D435 深度字段泄漏
    cal = (meta.get("calibration") or {}).get("head_stereo") or {}
    if "depth_camera" in cal:
        errors.append("单目标定不应含 depth_camera 字段")

    win.close()
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: 单目回归通过（真实摄像头录制/落盘,深度链路默认关闭行为不变）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
