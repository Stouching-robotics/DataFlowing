"""手套并入统一体系测试（mock BLE，无真机）。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/glove_widget_test.py

覆盖:
  1. GloveWidget: 连接流程 → 仿生手掌渲染帧落画面 + write_sensor 列名正确
  2. 主窗口集成: data_ble 开关 → 网格画面 + 传感器注册 + device_meta；
     关闭 → 画面/注册/条目全部撤销
  3. 无数据蓝牙开关 → 占位文案（不进传感器注册）
  4. 旧底部传感器 dock 已移除（_sensor_dock 不存在）
"""
import os
import sys
import time
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget

from config import settings
from config.i18n import tr

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


class FakePipeline:
    """录制管线替身：记录 write_sensor / record_event 调用。"""
    is_recording = True

    def __init__(self):
        self.sensor_writes = []
        self.events = []
        self._sensor_names = []

    def write_sensor(self, data, ts, sensor_name=""):
        self.sensor_writes.append((sensor_name, data.copy()))

    def record_event(self, dev, ev):
        self.events.append((dev, ev))

    def register_sensor(self, name):
        if name not in self._sensor_names:
            self._sensor_names.append(name)

    def unregister_sensor(self, name):
        if name in self._sensor_names:
            self._sensor_names.remove(name)


class MockBLE(QObject):
    """SensorBLEEngine 替身：真实 Qt 信号 + 合成压力矩阵。"""
    connected = pyqtSignal(str)
    disconnected = pyqtSignal()
    fps_updated = pyqtSignal(float)
    calibration_progress = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.addr = None
        self.is_calibrating = False
        self.hardware_fps = 30.0
        self.base_noise_gate = 500
        self.dynamic_noise_ratio = 0.0
        self.spatial_filter_enabled = True
        self.drift_baseline_val = 0
        self.latest_data_ts_us = int(time.time() * 1_000_000)

    def connect_device(self, address):
        self.addr = address
        self.connected.emit(address)

    def disconnect(self):
        self.disconnected.emit()

    def process_frame(self):
        data = np.random.rand(16, 16).astype(np.float32) * 3000
        return data, float(data.max())


class FakeGlove(QWidget):
    """主窗口集成测试用的手套控件替身。"""
    def __init__(self, slot, address, role, label, parent=None):
        super().__init__(parent)
        self.slot, self.address, self.role, self.label = slot, address, role, label
        self.started = None
        self.stopped = False
        self.pipeline = None

    def start(self, addr):
        self.started = addr

    def stop(self):
        self.stopped = True

    def set_pipeline(self, p):
        self.pipeline = p


def _mk_dev(key, kind, display_name, address=""):
    from core.device_detector import DeviceInfo
    return DeviceInfo(key=key, kind=kind, display_name=display_name,
                      address=address)


def main():
    app = QApplication(sys.argv)

    # ── 1. GloveWidget 渲染 + 录制列名 ──
    print("── 1. GloveWidget 仿生手掌渲染 / write_sensor ──")
    import ui.glove_widget as gw
    orig_cls = gw.SensorBLEEngine
    gw.SensorBLEEngine = MockBLE
    try:
        w = gw.GloveWidget("sensor:ble:AA:11:22:33:44:55",
                           "AA:11:22:33:44:55", "right_glove", "右手手套")
        pipe = FakePipeline()
        w.set_pipeline(pipe)
        w.start()
        app.processEvents()
        check(w._engine.addr == "AA:11:22:33:44:55",
              "start() 发起连接指定 MAC")
        check(w.video_widget._status_text
              == tr("已连接: {}…", "AA:11:22:33:44:55"[:12]),
              f"连接状态文案: {w.video_widget._status_text}")
        w._render_tick()
        app.processEvents()
        check(w.video_widget._has_frame, "仿生手掌渲染帧落画面")
        # 30ms 渲染定时器可能在 processEvents 期间已先行写入 → 只看最近一条
        last = pipe.sensor_writes[-1] if pipe.sensor_writes else None
        check(last is not None
              and last[0] == "right_glove"
              and last[1].shape == (16, 16),
              f"write_sensor 列名/形状: {(last[0], last[1].shape) if last else '-'}")
        w.stop()
        app.processEvents()
        check(w.video_widget._status_text == tr("已断开"),
              f"stop() 状态文案: {w.video_widget._status_text}")
        check(pipe.events and pipe.events[-1] == ("right_glove", "disconnected"),
              f"连接/断开事件: {pipe.events}")
        # 左手套 → 左配置文件（两文件键同值异：各自加载结果比对
        # 验证 role→文件路由 + DEFAULT_HAND 之外部位存在）
        from core.sensor_hand_config import load_sensor_hand_config
        from core.render_engine import DEFAULT_HAND
        wl = gw.GloveWidget("sensor:ble:BB:22:33:44:55:66",
                            "BB:22:33:44:55:66", "left_glove", "左手套")
        check(w.hand_config == load_sensor_hand_config("right_glove"),
              "right_glove 加载右配置文件")
        check(wl.hand_config == load_sensor_hand_config("left_glove")
              and len(wl.hand_config) > len(DEFAULT_HAND),
              f"left_glove 使用扩展仿生配置 ({len(wl.hand_config)} 部位)")
    finally:
        gw.SensorBLEEngine = orig_cls

    # ── 2/3/4. 主窗口集成（开关 → 网格/注册/meta → 关闭撤销；占位；dock 移除） ──
    print("── 2-4. 主窗口集成 ──")
    import ui.main_window as mw
    mw.GloveWidget = FakeGlove
    # 隔离设备命名持久化（避免污染真实 data/device_names.json）
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_path = tmp.name
    tmp.close()
    orig_names_file = settings.DEVICE_NAMES_FILE
    settings.DEVICE_NAMES_FILE = tmp_path
    win = mw.MainWindow()
    win.show()
    app.processEvents()
    try:
        check(not hasattr(win, "_sensor_dock"),
              "旧底部传感器 dock 已移除（无 _sensor_dock 属性）")

        glove = _mk_dev("ble:AA:11:22:33:44:55", "data_ble", "右手手套",
                        address="AA:11:22:33:44:55")
        win._on_device_toggled(glove, True)
        app.processEvents()
        slot = "sensor:ble:AA:11:22:33:44:55"
        check(slot in win.grid.slot_ids(), f"手套画面进主网格: {win.grid.slot_ids()}")
        entry = win._workers.get(glove.key, {})
        role = entry.get("sensor_column", "")
        check(entry.get("kind") == "data_ble" and role == "right_glove",
              f"worker 条目 kind/sensor_column: {entry}")
        check(role in win._pipeline._sensor_names,
              f"传感器列已注册: {win._pipeline._sensor_names}")
        meta = {d["key"]: d for d in win._build_device_meta()}
        check(meta[glove.key]["kind"] == "data_ble"
              and meta[glove.key]["slots"] == []
              and meta[glove.key].get("sensor_column") == "right_glove",
              f"device_meta 手套条目: {meta.get(glove.key)}")

        win._on_device_toggled(glove, False)
        app.processEvents()
        check(slot not in win.grid.slot_ids(), "关闭后网格画面移除")
        check(role not in win._pipeline._sensor_names, "关闭后传感器列注销")
        check(glove.key not in win._workers, "关闭后 worker 条目移除")

        # 无数据蓝牙占位
        ear = _mk_dev("ble:CC:33:44:55:66:77", "ble", "蓝牙耳机",
                      address="CC:33:44:55:66:77")
        win._on_device_toggled(ear, True)
        app.processEvents()
        bslot = f"ble:{ear.key}"
        bw = win.grid.camera_widget(bslot)
        check(bslot in win.grid.slot_ids()
              and bw is not None
              and bw.video_widget._status_text == tr("该设备无可视化数据"),
              f"无数据蓝牙占位: {bw.video_widget._status_text if bw else '-'}")
        check(not any(n in win._pipeline._sensor_names
                      for n in settings.SENSOR_NAMES),
              "占位设备不进传感器注册")
        win._on_device_toggled(ear, False)
        app.processEvents()
        check(bslot not in win.grid.slot_ids(), "占位关闭后移除")

        # teardown 全清
        win._on_device_toggled(glove, True)
        app.processEvents()
        win._teardown_all_workers()
        check(not win._workers and not win.grid.slot_ids(),
              "teardown 清理手套画面与条目")
    finally:
        settings.DEVICE_NAMES_FILE = orig_names_file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        win.close()
        app.processEvents()

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 手套并入统一体系测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
