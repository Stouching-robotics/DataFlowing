"""多路设备注册表测试（无真机）—— 槽名消歧 / 并存不互拆 / 冲突提示 /
OFF 只拆自己 / 多深度各写各的。

用法:
    QT_QPA_PLATFORM=offscreen PYTHONPYCACHEPREFIX=/tmp/pycache \
        venv/bin/python tools/tests/multi_device_registry_test.py

覆盖:
  1. 双 D435 槽名消歧（首台 d435_rgb/d435_depth，第 n 台 _2 后缀）
  2. UVC + D435 并存：互不拆槽；OFF 只拆自己
  3. S80M ↔ D435 冲突提示（后开者弹窗拒绝，无 worker 创建）
  4. 多深度槽各写各的（writer 级 mini episode，depth 槽名非 "depth"
     前缀亦由显式注册驱动；stereo_left 名义槽在多槽下不落盘）
"""
import os
import sys
import glob
import time
import shutil
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from config import settings
from core.device_detector import DeviceInfo
from ui.main_window import MainWindow

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def wait(app, ms):
    t_end = time.time() + ms / 1000.0
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)


FAKE_UVC = DeviceInfo(key="uvc:usb-FakeCam_12345678", kind="uvc",
                      display_name="FakeCam", serial="12345678",
                      video_index=40)
FAKE_D1 = DeviceInfo(key="d435:111111111111", kind="d435",
                     display_name="Intel RealSense D435",
                     serial="111111111111")
FAKE_D2 = DeviceInfo(key="d435:222222222222", kind="d435",
                     display_name="Intel RealSense D435",
                     serial="222222222222")
FAKE_D405 = DeviceInfo(key="d435:333333333333", kind="d435",
                       display_name="Intel RealSense D405",
                       serial="333333333333")
FAKE_S80M = DeviceInfo(key="s80m:ftdi", kind="s80m",
                       display_name="FaysSense S80M (fake)")

# _open_d435 按 serial 对 rs 上下文活体复查（2b5f4bf），假设备须同步
# 替身枚举结果，否则一律被"未检测到 RealSense 设备"拒绝。
FAKE_RS_ENUM = [("Intel RealSense D435", "111111111111"),
                ("Intel RealSense D435", "222222222222")]


class _Sig:
    """伪 Qt 信号（connect/disconnect 记账）。"""

    def __init__(self):
        self.slots = []

    def connect(self, f):
        self.slots.append(f)

    def disconnect(self, *a):
        self.slots.clear()


class FakeD435Worker:
    """D435Worker 替身（不碰真机）。"""

    def __init__(self, *a, **k):
        self.frames_ready = _Sig()
        self.error_occurred = _Sig()
        self.status_changed = _Sig()
        self.kwargs = k
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def deleteLater(self):
        pass

    def get_calibration(self):
        return None


def _patch_window(win, grid_slots):
    """栅格/管线记账替身 + UVC 建槽替身（真实采集不在本测试范围）。"""
    win.grid.add_camera = lambda sid, label="": grid_slots.add(sid) or None
    win.grid.remove_camera = lambda sid: grid_slots.discard(sid) or None
    win.grid.slot_ids = lambda: list(grid_slots)
    win.grid.camera_widget = lambda sid: None
    win._pipeline.register_external_source = lambda sid, dims, fps=None: None
    win._pipeline.unregister_external_source = lambda sid: None
    win._pipeline.set_depth_camera = (
        lambda name, res, fps=None, master_slot=None, **kw: None)
    win._pipeline.clear_depth_camera = lambda name=None: None
    win._pipeline.remove_camera = lambda sid: None
    uvc_calls = []

    def fake_add_slot(slot_id, camera_index, backend="", label=""):
        uvc_calls.append(slot_id)
        grid_slots.add(slot_id)

    win._add_camera_slot = fake_add_slot
    return uvc_calls


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    print("── 1. 双 D435 槽名消歧 ──")
    grid_slots = set()
    _patch_window(win, grid_slots)
    with patch("ui.main_window._D435_AVAILABLE", True), \
            patch("ui.main_window.D435Worker", FakeD435Worker), \
            patch("ui.main_window.list_d400_devices",
                  lambda: FAKE_RS_ENUM):
        win._on_device_toggled(FAKE_D1, True)
        e1 = win._workers.get(FAKE_D1.key)
        check(e1 is not None and e1["rgb_slot"] == settings.D435_SLOT_RGB
              and e1["depth_slot"] == settings.D435_SLOT_DEPTH,
              f"首台保旧槽名: {e1 and (e1['rgb_slot'], e1['depth_slot'])}")
        win._on_device_toggled(FAKE_D2, True)
        e2 = win._workers.get(FAKE_D2.key)
        check(e2 is not None
              and e2["rgb_slot"] == f"{settings.D435_SLOT_RGB}_2"
              and e2["depth_slot"] == f"{settings.D435_SLOT_DEPTH}_2",
              f"第 2 台槽名消歧: {e2 and (e2['rgb_slot'], e2['depth_slot'])}")
        check(grid_slots == {settings.D435_SLOT_RGB, settings.D435_SLOT_DEPTH,
                             f"{settings.D435_SLOT_RGB}_2",
                             f"{settings.D435_SLOT_DEPTH}_2"},
              f"4 槽并存: {sorted(grid_slots)}")
        # 关第一台 → 只拆自己
        win._on_device_toggled(FAKE_D1, False)
        check(FAKE_D1.key not in win._workers and FAKE_D2.key in win._workers,
              "关 D1 只拆自己，D2 仍在")
        check(grid_slots == {f"{settings.D435_SLOT_RGB}_2",
                             f"{settings.D435_SLOT_DEPTH}_2"},
              f"D2 槽未受影响: {sorted(grid_slots)}")
        win._on_device_toggled(FAKE_D2, False)
        check(not win._workers and not grid_slots, "全关后注册表/栅格清空")

    print("── 1b. 型号命名回落 + GUI 用户命名 ──")
    grid_slots = set()
    _patch_window(win, grid_slots)
    with patch("ui.main_window._D435_AVAILABLE", True), \
            patch("ui.main_window.D435Worker", FakeD435Worker), \
            patch("ui.main_window.list_d400_devices",
                  lambda: [("Intel RealSense D405", "333333333333")]), \
            patch("ui.main_window.settings.device_name", lambda k: ""):
        # 未命名 D405 → 回落型号名 d405_rgb/d405_depth
        win._on_device_toggled(FAKE_D405, True)
        e = win._workers.get(FAKE_D405.key)
        check(e is not None and e["rgb_slot"] == "d405_rgb"
              and e["depth_slot"] == "d405_depth",
              f"D405 未命名按型号: {e and (e['rgb_slot'], e['depth_slot'])}")
        check(grid_slots == {"d405_rgb", "d405_depth"},
              f"D405 槽进网格: {sorted(grid_slots)}")
        win._on_device_toggled(FAKE_D405, False)
        check(not win._workers and not grid_slots, "D405 关闭清空")

    with patch("ui.main_window._D435_AVAILABLE", True), \
            patch("ui.main_window.D435Worker", FakeD435Worker), \
            patch("ui.main_window.list_d400_devices",
                  lambda: [("Intel RealSense D435", "111111111111")]), \
            patch("ui.main_window.settings.device_name",
                  lambda k: "D435_depth" if k == FAKE_D1.key else ""):
        # GUI 命名 → RGB 槽 = 命名 + "_rgb"，深度槽 = 命名原样
        win._on_device_toggled(FAKE_D1, True)
        e = win._workers.get(FAKE_D1.key)
        check(e is not None and e["rgb_slot"] == "D435_depth_rgb"
              and e["depth_slot"] == "D435_depth",
              f"GUI 命名槽名: {e and (e['rgb_slot'], e['depth_slot'])}")
        check(grid_slots == {"D435_depth_rgb", "D435_depth"},
              f"命名槽进网格: {sorted(grid_slots)}")
        win._on_device_toggled(FAKE_D1, False)
        check(not win._workers and not grid_slots, "命名设备关闭清空")

    print("── 2. UVC + D435 并存，互不拆槽 ──")
    grid_slots = set()
    _patch_window(win, grid_slots)
    with patch("ui.main_window._D435_AVAILABLE", True), \
            patch("ui.main_window.D435Worker", FakeD435Worker), \
            patch("ui.main_window.list_d400_devices",
                  lambda: FAKE_RS_ENUM):
        win._on_device_toggled(FAKE_UVC, True)
        win._on_device_toggled(FAKE_D1, True)
        check({FAKE_UVC.key, FAKE_D1.key} <= set(win._workers),
              f"uvc+d435 并存: {sorted(win._workers)}")
        check(grid_slots == {settings._camera_slot_name(40),
                             settings.D435_SLOT_RGB, settings.D435_SLOT_DEPTH},
              f"三槽并存: {sorted(grid_slots)}")
        # 关 UVC → D435 槽不动
        win._on_device_toggled(FAKE_UVC, False)
        check(FAKE_UVC.key not in win._workers
              and {settings.D435_SLOT_RGB, settings.D435_SLOT_DEPTH}
              <= grid_slots,
              "OFF UVC 只拆自己，D435 槽仍在")
        # 关 D435 → 干净
        win._on_device_toggled(FAKE_D1, False)
        check(not win._workers and not grid_slots, "全关清空")

    print("── 3. S80M ↔ D435 冲突提示（后开者弹窗拒绝） ──")
    with patch("ui.main_window._D435_AVAILABLE", True), \
            patch("ui.main_window.D435Worker", FakeD435Worker), \
            patch("ui.main_window._STEREO_AVAILABLE", True), \
            patch("ui.main_window.list_d400_devices",
                  lambda: FAKE_RS_ENUM):
        win._on_device_toggled(FAKE_D1, True)
        # D435 已开 → 开 S80M 被拒（弹窗 + 无 worker）
        with patch("ui.main_window.QMessageBox.warning") as warn:
            win._on_device_toggled(FAKE_S80M, True)
        check(FAKE_S80M.key not in win._workers,
              "S80M 后开被拒（无 worker 创建）")
        check(warn.called, "冲突弹窗已提示")
        check(FAKE_S80M.key not in win._active_device_keys,
              f"被拒后勾选回退: {win._active_device_keys}")
        # 反向：开 S80M（替身脚本路径为真，但冲突检查在子进程启动之前）
        win._on_device_toggled(FAKE_D1, False)
        with patch("ui.main_window._STEREO_AVAILABLE", False):
            win._on_device_toggled(FAKE_S80M, True)
            check(FAKE_S80M.key not in win._workers,
                  "脚本缺失时 S80M 开启失败且无 worker")
        # 真实脚本存在时先插 s80m worker → D435 后开被拒
        win._workers[FAKE_S80M.key] = {"kind": "s80m", "slots": [],
                                       "label": "S80M fake"}
        with patch("ui.main_window.QMessageBox.warning") as warn2:
            win._on_device_toggled(FAKE_D1, True)
        check(FAKE_D1.key not in win._workers,
              "S80M 已开时 D435 后开被拒")
        check(warn2.called, "反向冲突弹窗已提示")
        win._workers.pop(FAKE_S80M.key)

    print("── 4. 多深度槽各写各的（writer 级） ──")
    from core.egodata_writer import EgoDataWriter
    out = tempfile.mkdtemp(prefix="multi_depth_")
    try:
        w = EgoDataWriter()
        cameras = {
            settings.D435_SLOT_RGB: (480, 848),
            settings.D435_SLOT_DEPTH: (480, 848),
            f"{settings.D435_SLOT_RGB}_2": (480, 848),
            f"{settings.D435_SLOT_DEPTH}_2": (480, 848),
            "stereo_left": (800, 1280),
            "stereo_right": (800, 1280),
        }
        depth_slots = [settings.D435_SLOT_DEPTH,
                       f"{settings.D435_SLOT_DEPTH}_2"]
        ok = w.start_episode(
            out, cameras, 30.0,
            depth_enabled=True,
            depth_heatmaps={
                s: {"near_mm": 300.0, "far_mm": 1000.0,
                    "smooth_k": 0, "temporal_alpha": 0.0}
                for s in depth_slots},
            depth_slots=depth_slots)
        check(ok, "start_episode（多深度槽）成功")
        frame = np.full((480, 848), 500, dtype=np.uint16)
        for i in range(5):
            w.write_depth_frame(i, frame, depth_slot=settings.D435_SLOT_DEPTH)
            w.write_depth_frame(i, frame,
                                depth_slot=f"{settings.D435_SLOT_DEPTH}_2")
            # stereo_left 名义槽在多槽场景下不落盘（无匹配 → 跳过）
            w.write_depth_frame(i, frame, depth_slot="stereo_left")
        w.end_episode()
        task_dir = w.task_dir
        d1 = os.path.join(task_dir, "videos", "chunk-000",
                          settings.D435_SLOT_DEPTH)
        d2 = os.path.join(task_dir, "videos", "chunk-000",
                          f"{settings.D435_SLOT_DEPTH}_2")
        check(os.path.isfile(os.path.join(d1, "episode-000.mp4"))
              or os.path.isfile(os.path.join(d1, "episode-000.mkv")),
              "槽 1 深度视频落盘（12-bit 灰 MP4，x265 无能力回落 FFV1 MKV）")
        check(os.path.isfile(os.path.join(d2, "episode-000.mp4"))
              or os.path.isfile(os.path.join(d2, "episode-000.mkv")),
              "槽 2 深度视频落盘（12-bit 灰 MP4，x265 无能力回落 FFV1 MKV）")
        check(len(glob.glob(os.path.join(d1, "*.png"))) == 0,
              "槽 1 不应再有 PNG16")
        check(len(glob.glob(os.path.join(d2, "*.png"))) == 0,
              "槽 2 不应再有 PNG16")
        check(len([f for f in os.listdir(d1)
                   if f.endswith((".mp4", ".mkv"))]) == 1,
              "单流直出后无合封临时件残留")
        # stereo_left 名义槽在多槽下不落深度（未注册深度槽 → 跳过；
        # 其 mp4 为 RGB 占位空文件，非深度流）
        check(len(glob.glob(os.path.join(task_dir, "videos", "chunk-000",
                                         "stereo_left", "*.mkv"))) == 0
              and len(glob.glob(os.path.join(task_dir, "videos", "chunk-000",
                                             "stereo_left", "*.png"))) == 0,
              "stereo_left 名义槽在多槽下不落深度")
    finally:
        shutil.rmtree(out, ignore_errors=True)

    win.close()
    wait(app, 300)

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 多路设备注册表测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
