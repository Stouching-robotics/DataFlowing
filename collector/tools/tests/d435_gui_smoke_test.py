"""D435 GUI 冒烟测试（offscreen）—— 完整走设备面板开关路径 + 帧显示 + 录制链路。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/d435_gui_smoke_test.py
"""
import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyarrow.parquet as pq
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from config import settings
from ui.main_window import MainWindow, _D435_AVAILABLE
from core.helpers import (episode_video_files, task_dir_of, pooled_info_path,
                           pooled_data_parquet_path)

OUT_ROOT = "/tmp/d435_gui_test"
DURATION = 8.0


def wait(app, ms):
    t_end = time.time() + ms / 1000.0
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)


def main():
    import shutil
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    if not _D435_AVAILABLE:
        print("FAIL: D435 不可用")
        return 1

    # 通过面板开关路径打开 D435（等价于用户勾选设备开关）
    from core.d435_camera import list_d400_devices
    from core.device_detector import DeviceInfo
    devs = list_d400_devices()
    model_name, serial = devs[0]
    dev = DeviceInfo(key=f"d435:{serial}", kind="d435",
                     display_name=model_name, serial=serial)
    win._on_device_toggled(dev, True)
    wait(app, 1500)

    # 槽名随 GUI 用户命名（device_names.json 持久化），未命名回落 d435_*
    uname = settings.device_name(dev.key) or "d435"
    rgb_slot, depth_slot = f"{uname}_rgb", uname
    expect = {rgb_slot, depth_slot}
    slots = set(win.grid.slot_ids())
    print(f"grid 槽位: {sorted(slots)}")
    if slots != expect:
        print(f"FAIL: 槽位不符, 期望 {expect}")
        win.close()
        return 1

    # 帧接收计数（附加信号槽,验证信号链路含 hw_ns 无损;
    # 不能替换 _on_d435_frames —— 信号连接时已绑定原方法）
    counts = {"frames": 0, "hw_max": 0, "hw_neg": 0}

    def record(slot_id, frame, hw_ns=0, imu=None):
        counts["frames"] += 1
        counts["hw_max"] = max(counts["hw_max"], hw_ns)
        if hw_ns < 0:
            counts["hw_neg"] += 1

    win._workers[dev.key]["worker"].frames_ready.connect(record)

    wait(app, 4000)
    print(f"帧接收: {counts['frames']} 帧 (≈{(counts['frames']/2)/4:.1f} fps), "
          f"hw_ns 最大 {counts['hw_max']/1e6:.0f}ms, 负值 {counts['hw_neg']}")
    if counts["frames"] < 100:
        print("FAIL: 帧流量不足")
        win.close()
        return 1
    if counts["hw_neg"] > 0 or counts["hw_max"] <= 2 * 10**9:
        print("FAIL: hw_ns 异常（截断复现？）")
        win.close()
        return 1

    # 实时 FPS 显示：帧到达计数 → 1s 定时器刷新标签（不再是 "--"）
    fps_texts = {sid: win.grid.camera_widget(sid).fps_label.text()
                 for sid in expect}
    print(f"FPS 标签: {fps_texts}")
    if any(t == "FPS: --" for t in fps_texts.values()):
        print("FAIL: FPS 标签未刷新（D435 路径 _note_frame_arrival 断链）")
        win.close()
        return 1

    # 录制: 走 _record_all 真实路径（任务名留空跳过进度检查）
    win._pipeline._output_dir = OUT_ROOT
    win._task_name_input.setText("")
    win._record_all()
    if not win._pipeline.is_recording:
        print("FAIL: 录制未启动")
        win.close()
        return 1
    wait(app, DURATION * 1000)
    win._stop_all()
    wait(app, 4000)  # 等 _finish_async

    ep = task_dir_of(OUT_ROOT, "")
    files0 = episode_video_files(ep, 1)
    ok = all(os.path.isfile(p) for p in [
        files0.get(rgb_slot) or "\x00",
        files0.get(depth_slot) or "\x00",
        pooled_info_path(ep),
        pooled_data_parquet_path(ep, 1),
    ])
    for slot in ["d435_left", "d435_right"]:
        if os.path.isdir(os.path.join(ep, "videos", "chunk-000", slot)):
            ok = False
    pngs = glob.glob(os.path.join(ep, "videos", "chunk-000", depth_slot,
                                  "*.png"))
    print(f"task: {ep}")
    print(f"文件齐全: {ok}, 残留 PNG16: {len(pngs)}")
    if not ok or pngs:
        print("FAIL: 落盘不完整")
        win.close()
        return 1

    t = pq.read_table(pooled_data_parquet_path(ep, 1))
    hw = np.asarray(t.column("hardware_ns").to_pylist(), dtype=np.int64)
    mono = bool(np.all(np.diff(hw) >= 0))
    print(f"hardware_ns: {len(hw)} 行, 单调={mono}, 末={hw[-1]/1e6:.0f}ms")
    if not mono:
        print("FAIL: hardware_ns 非单调")
        win.close()
        return 1

    # 关闭开关 → D435 槽位应清空（只动自己，不残留）
    win._on_device_toggled(dev, False)
    wait(app, 800)
    leftover = set(win.grid.slot_ids()) & expect
    print(f"关闭开关后 D435 槽位残留: {leftover or '无'}")
    if leftover:
        print("FAIL: 关闭设备未清理")
        win.close()
        return 1

    win.close()
    wait(app, 300)
    print("PASS: GUI 冒烟测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
