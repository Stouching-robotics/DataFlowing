"""设备检测面板 GUI 冒烟测试（offscreen）—— 分组 / 开关 / 命名 / 锁死 / 多高亮 / 拔线。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/device_panel_gui_smoke_test.py

覆盖:
  1. DevicePanel 独立段：三组分组 + other_ble 折叠 + 空组隐藏 + 空列表占位
  2. 勾选 → device_toggled；set_checked_keys 恢复不发信号；多高亮
  3. 双击重命名 → 持久化（临时文件）+ 行文本更新 + device_renamed
  4. set_locked 录制锁死（子项去 ItemIsEnabled + tooltip）
  5. MainWindow 集成：扫描 → user_name 填充（重命名后仍显示新名）
  6. 开 UVC → 建槽；开手套 → 多设备并存；关手套 → 只动自己
  7. 相机互替（stereo 可用时）：开 S80M → UVC/手套取消勾选
  8. 录制中 toggle → 回退不建槽
  9. 拔线 → 占位；录制中拔线重插 → 恢复勾选
 10. 真机段（D435 在位）→ 开 d435 出双画面、serial 锁定
"""
import os
import sys
import tempfile
import time
from unittest.mock import patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from config import settings
from config.i18n import tr
from core.device_detector import DeviceInfo, detect_devices
from core.pipeline import CameraPipeline
from ui.device_panel import DevicePanel
from ui.main_window import MainWindow, _slot_base, _realsense_short

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
FAKE_D435 = DeviceInfo(key="d435:999888777666", kind="d435",
                       display_name="RealSense D435 (fake)",
                       serial="999888777666")
FAKE_S80M = DeviceInfo(key="s80m:ftdi", kind="s80m",
                       display_name="FaysSense S80M (fake)")
FAKE_GLOVE = DeviceInfo(key="ble:AA:11:22:33:44:55", kind="data_ble",
                        display_name="Matrix Glove R",
                        serial="AA:11:22:33:44:55",
                        address="AA:11:22:33:44:55")
FAKE_BT = DeviceInfo(key="ble:30:A9:98:57:4A:C2", kind="ble",
                     display_name="HUAWEI FreeBuds 5i",
                     serial="30:A9:98:57:4A:C2",
                     address="30:A9:98:57:4A:C2")
FAKES = [FAKE_UVC, FAKE_D435, FAKE_S80M, FAKE_GLOVE, FAKE_BT]


def group_rows(panel, group):
    """组子项列表 [(item, text)]（组不存在返回 []）。"""
    for i in range(panel._tree.topLevelItemCount()):
        top = panel._tree.topLevelItem(i)
        data = top.data(0, Qt.UserRole)
        if isinstance(data, tuple) and data[0] == "group" and data[1] == group:
            return [(top.child(j), top.child(j).text(0))
                    for j in range(top.childCount())]
    return []


def group_top(panel, group):
    """组顶层项（不存在返回 None）。"""
    for i in range(panel._tree.topLevelItemCount()):
        top = panel._tree.topLevelItem(i)
        data = top.data(0, Qt.UserRole)
        if isinstance(data, tuple) and data[0] == "group" and data[1] == group:
            return top
    return None


def main():
    app = QApplication(sys.argv)

    # 命名持久化全程指向临时文件（隔离真实 device_names.json）
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _orig_names_file = settings.DEVICE_NAMES_FILE
    settings.DEVICE_NAMES_FILE = tmp_path
    try:
        return _main(app)
    finally:
        settings.DEVICE_NAMES_FILE = _orig_names_file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _main(app):
    print("── 1. 分组 / 折叠 / 空组隐藏 / 占位 ──")
    panel = DevicePanel()
    panel.set_devices(FAKES)
    check(len(group_rows(panel, "camera")) == 3, "相机组 3 子项")
    check(len(group_rows(panel, "glove")) == 1, "手套组 1 子项")
    check(len(group_rows(panel, "other_ble")) == 1, "其他蓝牙组 1 子项")
    bt_top = group_top(panel, "other_ble")
    check(bt_top is not None and not bt_top.isExpanded(),
          "其他蓝牙组默认折叠")
    check(group_top(panel, "camera").isExpanded(), "相机组默认展开")
    texts = [t for _, t in group_rows(panel, "camera")]
    check(any("FakeCam" in t and "12345678" in t for t in texts),
          "相机条目含名称 + 序号")

    panel_ble_only = DevicePanel()
    panel_ble_only.set_devices([FAKE_GLOVE])
    check(group_top(panel_ble_only, "camera") is None,
          "空组隐藏（仅手套时无相机组）")

    panel_empty = DevicePanel()
    panel_empty.set_devices([])
    top = panel_empty._tree.topLevelItem(0)
    check(top is not None and top.text(0) == tr("未检测到设备"),
          f"空列表占位: {top.text(0) if top else None}")

    print("── 2. 勾选 → device_toggled；set_checked_keys 恢复；多高亮 ──")
    panel.set_devices(FAKES)
    got = []
    panel.device_toggled.connect(lambda d, on: got.append((d.key, on)))
    panel._items[FAKE_UVC.key].setCheckState(0, Qt.Checked)   # 模拟用户点击
    check(got == [(FAKE_UVC.key, True)],
          f"勾选发射 device_toggled: {got}")
    panel.set_checked_keys({FAKE_D435.key})
    check(panel.checked_keys() == {FAKE_D435.key}
          and got == [(FAKE_UVC.key, True)],
          "set_checked_keys 恢复不发信号")
    panel.set_active_keys({FAKE_UVC.key, FAKE_GLOVE.key})
    fg_uvc = panel._items[FAKE_UVC.key].foreground(0).color().name().lower()
    fg_s80m = panel._items[FAKE_S80M.key].foreground(0).color().name().lower()
    check(fg_uvc == settings.COLOR_STOPPED.lower()
          and fg_s80m == settings.COLOR_TEXT_PRIMARY.lower(),
          f"多高亮：激活绿({fg_uvc}) 未激活白({fg_s80m})")

    print("── 3. 双击重命名 → 持久化 + 行文本 + 信号 ──")
    renamed = []
    panel.device_renamed.connect(lambda d, n: renamed.append((d.key, n)))
    with patch("ui.device_panel.QInputDialog.getText",
               return_value=("右手手套", True)):
        panel._on_item_double_clicked(panel._items[FAKE_GLOVE.key], 0)
    check(settings.device_name("ble:AA:11:22:33:44:55") == "右手手套",
          "命名写入 device_names.json")
    glove_text = panel._items[FAKE_GLOVE.key].text(0)
    check("右手手套" in glove_text, f"行文本更新为用户命名: {glove_text}")
    check(renamed == [(FAKE_GLOVE.key, "右手手套")],
          f"device_renamed 发射: {renamed}")

    print("── 4. set_locked 录制锁死 ──")
    panel.set_locked(True)
    all_locked = all(not (item.flags() & Qt.ItemIsEnabled)
                     for item in panel._items.values())
    check(all_locked, "锁死后所有子项不可用（去 ItemIsEnabled）")
    check(panel._tree.toolTip() == tr("录制中不可更改设备"),
          "锁死 tooltip 提示")
    panel.set_locked(False)
    check(all(item.flags() & Qt.ItemIsEnabled for item in panel._items.values()),
          "解锁恢复 ItemIsEnabled")

    # ═══════════════════════════════════════════════════
    #  MainWindow 集成
    # ═══════════════════════════════════════════════════
    print("── 5. 扫描 → user_name 填充（重命名保持） ──")
    win = MainWindow()
    win.show()
    add_calls = []
    def fake_add_slot(slot_id, camera_index, backend="", label=""):
        add_calls.append((slot_id, camera_index, label))
    win._add_camera_slot = fake_add_slot
    # 兼容 fake 槽位生命周期:add 只记账(真实 UVC 采集不在本测试范围),
    # 槽位视图 = 真实 grid 槽 ∪ fake 记账;remove 均 no-op
    real_slot_ids = win.grid.slot_ids
    def fake_slot_ids():
        return list(set(real_slot_ids()) | {c[0] for c in add_calls})
    win.grid.slot_ids = fake_slot_ids
    win._pipeline.remove_camera = lambda sid: None
    _real_remove = win.grid.remove_camera
    def _fake_remove(sid):
        # 仅 fake UVC 槽（add 只记账）no-op；手套/占位等真实控件照常移除
        if sid != settings._camera_slot_name(40):
            _real_remove(sid)
    win.grid.remove_camera = _fake_remove

    with patch("core.device_detector.detect_devices", return_value=FAKES):
        win._page_stack.setCurrentIndex(1)
        t_end = time.time() + 3
        while len(win._device_panel._items) < 5 and time.time() < t_end:
            app.processEvents()
            time.sleep(0.01)
    win._device_timer.stop()
    check(len(win._device_panel._items) == 5, "面板列出 5 台设备（含蓝牙）")
    glove_text = win._device_panel._items[FAKE_GLOVE.key].text(0)
    check("右手手套" in glove_text,
          f"重命名后重新连接仍显示新命名: {glove_text}")

    print("── 6. 开 UVC → 建槽；开手套 → 并存；关手套 ──")
    win._device_panel._items[FAKE_UVC.key].setCheckState(0, Qt.Checked)
    check(len(add_calls) == 1
          and add_calls[0][0] == settings._camera_slot_name(40)
          and add_calls[0][2] == "FakeCam",
          f"按索引建槽: {add_calls}")
    check(win._active_device_keys == {FAKE_UVC.key},
          f"激活集合: {win._active_device_keys}")
    check(win._workers.get(FAKE_UVC.key, {}).get("kind") == "uvc",
          f"UVC 注册表条目: {list(win._workers)}")
    win._device_panel._items[FAKE_GLOVE.key].setCheckState(0, Qt.Checked)
    check(win._active_device_keys == {FAKE_UVC.key, FAKE_GLOVE.key},
          f"手套与相机并存: {win._active_device_keys}")
    win._device_panel._items[FAKE_GLOVE.key].setCheckState(0, Qt.Unchecked)
    check(win._active_device_keys == {FAKE_UVC.key}
          and FAKE_UVC.key in win._workers,
          "关手套只动自己，UVC 仍在显示")
    check("sensor:ble:AA:11:22:33:44:55" not in real_slot_ids(),
          "关手套后画面槽移除")
    check(win._device_panel.checked_keys() == {FAKE_UVC.key},
          "勾选集合与激活一致")

    print("── 7. 关 UVC → 只拆自己（注册表/勾选清空） ──")
    win._device_panel._items[FAKE_UVC.key].setCheckState(0, Qt.Unchecked)
    check(FAKE_UVC.key not in win._workers,
          f"UVC 注册表条目已拆: {list(win._workers)}")
    check(win._active_device_keys == set()
          and win._device_panel.checked_keys() == set(),
          "勾选集合清空")

    print("── 8. 录制中 toggle → 回退不建槽 ──")
    with patch.object(CameraPipeline, "is_recording",
                      new_callable=PropertyMock, return_value=True):
        win._device_panel._items[FAKE_UVC.key].setCheckState(0, Qt.Checked)
    check(win._device_panel._items[FAKE_UVC.key].checkState(0) == Qt.Unchecked,
          "录制中勾选被回退")
    check(len(add_calls) == 1, "未重复建槽")

    print("── 9. 拔线 → 占位；录制中拔线重插恢复勾选 ──")
    win._device_panel._items[FAKE_UVC.key].setCheckState(0, Qt.Checked)
    with patch.object(CameraPipeline, "is_recording",
                      new_callable=PropertyMock, return_value=True):
        win._on_devices_scanned([])
        top = win._device_panel._tree.topLevelItem(0)
        check(top is not None and top.text(0) == tr("未检测到设备"),
              f"拔线后占位: {top.text(0) if top else None}")
        check(FAKE_UVC.key in win._lost_device_keys,
              f"录制中拔线暂存 key: {win._lost_device_keys}")
        win._on_devices_scanned(FAKES)
    check(win._active_device_keys == {FAKE_UVC.key},
          f"重插恢复勾选: {win._active_device_keys}")
    check(win._device_panel._items[FAKE_UVC.key].checkState(0) == Qt.Checked,
          "重插后复选框恢复勾选")

    print("── 10. 真机段（RealSense 在位）→ 开设备出双画面 ──")
    real = detect_devices()
    d435s = [d for d in real if d.kind == "d435"]
    if not d435s:
        print("  SKIP: 本机无 D435")
    else:
        # 本测试全程隔离 device_names.json 到临时文件 → 命名回落型号名。
        # 期望槽名按与 _open_d435 同规则动态推导（命名规则本体由
        # multi_device_registry_test 断言；本段只验证面板开关→worker 接线）。
        def expect_slots(dev):
            base = _slot_base(settings.device_name(dev.key),
                              _realsense_short(dev.display_name).lower())
            n = sum(1 for e in win._workers.values()
                    if e["kind"] == "d435"
                    and e["rgb_slot"].startswith(f"{base}_rgb"))
            rgb = f"{base}_rgb" if n == 0 else f"{base}_rgb_{n + 1}"
            depth = base if base.endswith("_depth") else f"{base}_depth"
            if n > 0:
                depth = f"{depth}_{n + 1}"
            return rgb, depth

        win.grid.slot_ids = real_slot_ids   # 恢复真实槽位视图（结束 fake UVC 段）
        win._on_devices_scanned(real)
        d435_dev = d435s[0]
        # 期望须在开启前计算（n 统计现存同前缀 worker，不含目标自己）
        e_rgb, e_depth = expect_slots(d435_dev)
        win._device_panel._items[d435_dev.key].setCheckState(0, Qt.Checked)
        wait(app, 1500)
        slots = set(win.grid.slot_ids())
        check(slots == {e_rgb, e_depth}, f"双槽位出现: {sorted(slots)}")
        check(win._active_device_keys == {d435_dev.key},
              f"d435 高亮 key: {win._active_device_keys}")
        check(win._workers[d435_dev.key]["serial"] == d435_dev.serial,
              f"worker 锁定点击设备: {win._workers[d435_dev.key]['serial']}")
        others = [d for d in real if d.kind == "d435" and d.key != d435_dev.key]
        if others:
            # 首台已在 workers 中 → 同前缀才编号；异型号互不影响。
            # 期望同样须在开启前计算（不含目标自己）。
            o_rgb, o_depth = expect_slots(others[0])
            win._device_panel._items[others[0].key].setCheckState(0, Qt.Checked)
            wait(app, 1500)
            check(win._workers[others[0].key]["serial"] == others[0].serial,
                  f"开第二台 RealSense 独立 worker: {others[0].serial}")
            check(win._workers[others[0].key]["rgb_slot"] == o_rgb
                  and win._workers[others[0].key]["depth_slot"] == o_depth,
                  f"第二台槽名消歧: {win._workers[others[0].key]['rgb_slot']}/"
                  f"{win._workers[others[0].key]['depth_slot']}")
            # 关第一台 → 只拆自己，第二台不受影响（多路并存）
            win._device_panel._items[d435_dev.key].setCheckState(0, Qt.Unchecked)
            wait(app, 800)
            check(d435_dev.key not in win._workers
                  and others[0].key in win._workers,
                  "关一台只拆自己，另一台仍在")
        # 等真实帧流量
        counts = {"n": 0}
        active = win._workers[(others[0] if others else d435_dev).key]
        active["worker"].frames_ready.connect(
            lambda *a: counts.__setitem__("n", counts["n"] + 1))
        wait(app, 3000)
        check(counts["n"] > 50, f"D435 帧流量: {counts['n']} 信号/3s")

    win.close()   # 覆盖 closeEvent 清理路径
    wait(app, 300)

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 设备面板 GUI 冒烟测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
