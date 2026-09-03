"""每相机曝光控制测试（offscreen，无硬件）——

覆盖:
  1. settings device_params.json 曝光持久化 round-trip（merge-write）
  2. CameraWorker._apply_exposure_to:自动/手动两种写入序列
     （V4L2 菜单直通候选 0/3/2、1/0）
  3. CameraWorker 线程模型:UVC 固定自动曝光（开启/重连后首帧读出成功
     时自动应用，无任何外部调用）
  4. D435Worker exposure 构造注入 / set_exposure / exposure_info
  5. d435 _apply_color_exposure:自动只开关、手动写值、异常不抛出
  6. CameraWidget ☀ 按钮:默认隐藏、点击发信号、拖拽过滤器放行
  7. ExposureDialog:滑块映射 + 即时生效 + 自动曝光开关联动
  8. MainWindow 接线:_apply_exposure 按类型下发并持久化（UVC 无此
     功能）、S80M stdin 行协议、录制锁禁用/恢复曝光按钮

用法:
    QT_QPA_PLATFORM=offscreen PYTHONPYCACHEPREFIX=/tmp/pycache \
        venv/bin/python tools/tests/exposure_control_test.py
"""
import io
import os
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication

from config import settings
from config.i18n import tr
from core.camera import CameraWorker
from core.d435_camera import D435Worker, _apply_color_exposure
from core.pipeline import CameraPipeline
from ui.camera_grid import CameraGrid
from ui.exposure_dialog import ExposureDialog
from ui.main_window import MainWindow

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


# ── 假对象 ──────────────────────────────────────────────

class FakeCap:
    """CameraWorker 用的假 VideoCapture（记录 set 调用，read 出黑帧）。

    exp_get: 曝光相关属性的 get 回读值（如 {CAP_PROP_AUTO_EXPOSURE: 1,
    CAP_PROP_EXPOSURE: 62.0}），用于模拟「最一开始」基线捕获。
    reject_sets: (prop, value) 集合，模拟相机不支持的控制（set 返回
    False），如 DECXIN 相机不接受 auto_exposure 菜单值 0。
    auto_exposure 的 set 会记录当前值，get 读回最近一次成功写入
    （exp_get 优先，模拟 OpenCV 5 菜单直通回读）。
    """
    def __init__(self, exp_get=None, reject_sets=None):
        self.sets = []
        self.opened = True
        self._exp_get = exp_get or {}
        self._reject = reject_sets or set()
        self._auto_val = None

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.sets.append((prop, value))
        if (prop, value) in self._reject:
            return False
        if prop == _auto_prop():
            self._auto_val = value
        return True

    def get(self, prop):
        if prop in self._exp_get:
            return self._exp_get[prop]
        if prop == _auto_prop() and self._auto_val is not None:
            return self._auto_val
        return {cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480}.get(prop, 0)

    def read(self):
        time.sleep(0.01)
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.opened = False


class FakeSensor:
    """D435 color 传感器假对象（记录 set_option）。"""
    def __init__(self, fail=False):
        self.opts = {}
        self._fail = fail

    def set_option(self, opt, val):
        if self._fail:
            raise RuntimeError("fake option error")
        self.opts[opt] = val


def _auto_prop():
    return getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", 21)


# ── 各段测试 ────────────────────────────────────────────

def test_settings_roundtrip():
    print("── 1. settings 曝光持久化 ──")
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    orig = settings.DEVICE_PARAMS_FILE
    settings.DEVICE_PARAMS_FILE = tmp.name
    try:
        settings.save_device_exposure("uvc:a", False, 500.0)
        settings.save_device_exposure("d435:b", True, 0.0)   # merge-write
        check(settings.device_exposure("uvc:a")
              == {"auto": False, "value": 500.0},
              "手动曝光 round-trip")
        check(settings.device_exposure("d435:b")
              == {"auto": True, "value": 0.0},
              "自动曝光 round-trip（merge-write 保留 uvc:a）")
        check(settings.device_exposure("nope") is None,
              "无条目返回 None")

        # 「最一开始」基线：首次写入后锁定，永不覆盖
        check(settings.ensure_device_original("uvc:a", True, 0.0) is True,
              "原始基线首次写入")
        check(settings.ensure_device_original("uvc:a", False, 999.0) is False,
              "原始基线首见锁定，不覆盖")
        check(settings.device_original("uvc:a")
              == {"auto": True, "value": 0.0},
              "device_original 读回首见基线")
        check(settings.device_original("nope") is None,
              "无基线返回 None")
        check(settings.device_exposure("uvc:a")
              == {"auto": False, "value": 500.0},
              "基线写入 merge-write 保留曝光条目")
    finally:
        settings.DEVICE_PARAMS_FILE = orig
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def test_camera_worker_apply():
    print("── 2. CameraWorker 曝光写入 ──")
    # 默认 backend（""/Windows MSMF/DShow 分支）：约定 1=自动 0=手动
    cap = FakeCap()
    ok = CameraWorker._apply_exposure_to(cap, True, 0.0)
    check(ok[0] and ok[1], "自动模式返回成功")
    check((_auto_prop(), 1) in cap.sets
          and not any(p == cv2.CAP_PROP_EXPOSURE for p, _ in cap.sets),
          "自动:MSMF 约定写 auto=1 不写曝光值")

    cap = FakeCap()
    ok = CameraWorker._apply_exposure_to(cap, False, 500.0)
    idx_auto = cap.sets.index((_auto_prop(), 0))
    idx_val = cap.sets.index((cv2.CAP_PROP_EXPOSURE, 500.0))
    check(ok[0] and ok[1] and idx_auto < idx_val,
          "手动:先切手动(auto=0)再写值 500")

    cap = FakeCap()
    CameraWorker._apply_exposure_to(cap, False, 0.0)
    check(any(p == cv2.CAP_PROP_EXPOSURE and v >= 1.0
              for p, v in cap.sets),
          "手动值下限钳到 1")

    # V4L2 后端（OpenCV 5 菜单直通）：标准相机自动=0，写入 0 即成功
    cap = FakeCap()
    ok = CameraWorker._apply_exposure_to(cap, True, 0.0, backend="V4L2")
    check(ok[0] and ok[1] and (_auto_prop(), 0) in cap.sets
          and not any(p == cv2.CAP_PROP_EXPOSURE for p, _ in cap.sets),
          "V4L2 自动:标准相机写 0 成功、不写曝光值")

    # DECXIN 类相机：菜单无 0（自动=3 光圈优先），0 失败后候选落到 3
    cap = FakeCap(reject_sets={(_auto_prop(), 0)})
    ok = CameraWorker._apply_exposure_to(cap, True, 0.0, backend="V4L2")
    check(ok[0] and ok[1]
          and (_auto_prop(), 3) in cap.sets
          and not any(p == cv2.CAP_PROP_EXPOSURE for p, _ in cap.sets),
          "V4L2 自动:0 失败后候选 3（DECXIN 光圈优先）")

    # DECXIN 类相机手动：候选 1（标准手动）直接成功，随后写曝光值
    cap = FakeCap(reject_sets={(_auto_prop(), 0)})
    ok = CameraWorker._apply_exposure_to(cap, False, 500.0, backend="V4L2")
    check(ok[0] and ok[1]
          and (_auto_prop(), 1) in cap.sets
          and (cv2.CAP_PROP_EXPOSURE, 500.0) in cap.sets,
          "V4L2 手动:候选 1 成功 + 写值")


def test_camera_worker_thread_model():
    print("── 3. CameraWorker 固定自动曝光 + 重连重放 ──")
    # 无需任何外部调用：开启后在首帧读出成功后自动应用自动曝光
    fake = FakeCap()
    with patch("core.camera._try_open_camera", return_value=(fake, "V4L2")):
        w = CameraWorker(camera_index=5, resolution=(640, 480))
        w.start()
        time.sleep(0.3)
        w.stop()
        check((_auto_prop(), 0) in fake.sets
              and not any(p == cv2.CAP_PROP_EXPOSURE for p, _ in fake.sets),
              f"开启后自动应用自动曝光: {[(p, v) for p, v in fake.sets if p >= 15]}")

    # DECXIN 类相机（菜单无 0，自动=3 光圈优先）：候选落 3
    fake_dec = FakeCap(reject_sets={(_auto_prop(), 0)})
    with patch("core.camera._try_open_camera", return_value=(fake_dec, "V4L2")):
        w = CameraWorker(camera_index=5)
        w.start()
        time.sleep(0.3)
        w.stop()
        check((_auto_prop(), 3) in fake_dec.sets,
              f"DECXIN 类相机自动曝光落候选 3: "
              f"{[(p, v) for p, v in fake_dec.sets if p >= 15]}")

    # 重连重放：打开时只标记 pending 不直接写（部分 UVC 相机在 STREAMON
    # 时重置/忽略 AE 控制，开流前的写入会丢失 → 重连后画面发暗），
    # 采集循环首帧读出成功后补应用
    fake2 = FakeCap()
    with patch("core.camera._try_open_camera", return_value=(fake2, "V4L2")):
        w = CameraWorker(camera_index=6)
        w._open_camera()
        check((_auto_prop(), 0) not in fake2.sets,
              "重连重放:打开时不直接写曝光（推迟到首帧读出后）")
        check(w._exp_pending, "重连重放:打开后 pending 已标记")
    fake2b = FakeCap()
    with patch("core.camera._try_open_camera", return_value=(fake2b, "V4L2")):
        w = CameraWorker(camera_index=6)
        w.start()
        time.sleep(0.3)
        w.stop()
        check((_auto_prop(), 0) in fake2b.sets,
              "重连重放:首帧读出成功后自动曝光写入")


def test_d435_worker_exposure():
    print("── 4. D435Worker 曝光构造注入 / 下发 ──")
    w = D435Worker(exposure={"auto": False, "value": 8500.0})
    check(w.exposure_info() == (None, False, 8500.0),
          "构造注入手动曝光（量程未启动为 None）")
    w.set_exposure(True, 0.0)
    check(w.exposure_info() == (None, True, 0.0),
          "set_exposure 切换自动")
    w2 = D435Worker()
    check(w2.exposure_info() == (None, None, None),
          "无注入时三值全 None（不动相机默认）")


def test_d435_apply_color_exposure():
    print("── 5. _apply_color_exposure ──")
    import pyrealsense2 as rs
    s = FakeSensor()
    check(_apply_color_exposure(s, True, 0.0)
          and s.opts == {rs.option.enable_auto_exposure: 1},
          "自动:只开自动曝光")
    s = FakeSensor()
    check(_apply_color_exposure(s, False, 8500.0)
          and s.opts == {rs.option.enable_auto_exposure: 0,
                         rs.option.exposure: 8500.0},
          "手动:关自动 + 写值")
    check(_apply_color_exposure(FakeSensor(fail=True), False, 1.0) is False,
          "异常返回 False 不抛出")


def test_widget_button(app):
    print("── 6. ☀ 按钮 + 拖拽过滤器放行 ──")
    grid = CameraGrid()
    w = grid.add_camera("slot_x", "Cam X")
    grid.add_camera("slot_z", "Cam Z")   # 拖拽需 ≥2 画面
    check(w.exposure_btn.isHidden(), "按钮默认隐藏")
    w.set_exposure_button_visible(True)
    check(not w.exposure_btn.isHidden(), "set_exposure_button_visible 显示")
    clicks = []
    w.exposure_clicked.connect(clicks.append)
    w.exposure_btn.click()
    check(clicks == ["slot_x"], "点击发射 exposure_clicked(slot_id)")

    # 真实命中回归：覆盖条不得设 WA_TransparentForMouseEvents——Qt 语义
    # 是父控件连同子控件一起屏蔽鼠标，按钮上的 False 覆盖无效（曾致
    # 真机点击 ☀ 无效）。widgetAt 与鼠标分发走同一命中路径。
    check(not w.overlay.testAttribute(Qt.WA_TransparentForMouseEvents),
          "覆盖条未设鼠标穿透（否则子按钮收不到点击）")
    grid.show()
    for _ in range(8):
        app.processEvents()
    g = w.exposure_btn.mapToGlobal(w.exposure_btn.rect().center())
    check(QApplication.widgetAt(g) is w.exposure_btn,
          "命中测试落在 ☀ 按钮上（点击可被接收）")
    grid.hide()

    # 按钮按下：过滤器放行（不抓鼠标不消费）
    grid._drag_src = None
    ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(2, 2),
                     QPointF(2, 2), Qt.LeftButton, Qt.LeftButton,
                     Qt.NoModifier)
    check(grid.eventFilter(w.exposure_btn, ev) is False
          and grid._drag_src is None,
          "按钮按下不触发拖拽抓取")

    # 对照：信息条本体按下仍被消费（拖拽手柄语义不回归）
    gpos = w.overlay.mapToGlobal(QPoint(10, 10))
    ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(10, 10), gpos,
                     Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    check(grid.eventFilter(w.overlay, ev) is True
          and grid._drag_src is w,
          "信息条按下仍触发拖拽")
    grid._end_drag()

    # 录制锁语义
    w.set_exposure_enabled(False)
    check(not w.exposure_btn.isEnabled()
          and w.exposure_btn.toolTip() == tr("录制中不可调整曝光"),
          f"禁用 + tooltip 提示: {w.exposure_btn.toolTip()}")
    w.set_exposure_enabled(True)
    check(w.exposure_btn.isEnabled()
          and w.exposure_btn.toolTip() == tr("曝光设置"),
          "恢复启用 + tooltip 复原")


def test_dialog(app):
    print("── 7. ExposureDialog ──")
    dlg = ExposureDialog(None, "Cam X", 1.0, 1000.0, 500.0, False)
    check(tr("曝光设置") in dlg.windowTitle()
          and "Cam X" in dlg.windowTitle(),
          f"标题含设备名: {dlg.windowTitle()}")
    gets = []
    dlg.apply_requested.connect(lambda a, v: gets.append((a, v)))
    dlg._slider.sliderMoved.emit(250)
    check(gets == [(False, 250.75)], f"拖动即时下发: {gets}")
    dlg._auto_cb.setChecked(True)
    check(gets[-1][0] is True and not dlg._slider.isEnabled(),
          "勾选自动曝光立即下发 + 滑块禁用")
    dlg._slider.sliderMoved.emit(500)
    check(len(gets) == 2, "自动模式下拖动滑块不下发")
    dlg._auto_cb.setChecked(False)
    dlg._slider.setValue(1000)
    dlg._slider.sliderReleased.emit()
    check(gets[-1] == (False, 1000.0), "松开滑块按当前刻度下发")

    # 恢复默认：有基线时按钮可见，点击静默复位控件并下发一次
    dlg2 = ExposureDialog(None, "Cam X", 1.0, 1000.0, 800.0, False,
                          original=(False, 400.0))
    gets2 = []
    dlg2.apply_requested.connect(lambda a, v: gets2.append((a, v)))
    check(not dlg2._reset_btn.isHidden(), "有基线时恢复默认按钮可见")
    dlg2._reset_btn.click()
    check(gets2 == [(False, 400.0)] and not dlg2._auto_cb.isChecked()
          and dlg2._slider.value() == dlg2._to_ticks(400.0),
          f"恢复默认下发原始基线并复位控件: {gets2}")
    dlg3 = ExposureDialog(None, "Cam X", 1.0, 1000.0, 800.0, False)
    check(dlg3._reset_btn.isHidden(), "无基线时恢复默认按钮隐藏")
    dlg4 = ExposureDialog(None, "Cam X", 1.0, 1000.0, 800.0, False,
                          original=(True, 0.0))
    gets4 = []
    dlg4.apply_requested.connect(lambda a, v: gets4.append((a, v)))
    dlg4._reset_btn.click()
    check(gets4 == [(True, 0.0)] and dlg4._auto_cb.isChecked()
          and not dlg4._slider.isEnabled(),
          "恢复默认到自动曝光（滑块禁用）")


def test_main_window_wiring(app):
    print("── 8. MainWindow 接线 ──")
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    orig = settings.DEVICE_PARAMS_FILE
    settings.DEVICE_PARAMS_FILE = tmp.name
    win = MainWindow()
    try:
        # UVC：无曝光功能 —— _apply_exposure 不下发、不持久化
        class FakeCam:
            def __init__(self):
                self.calls = []
            def set_exposure(self, a, v):
                self.calls.append((a, v))

        cam = FakeCam()
        win._pipeline.get_slot = lambda sid: SimpleNamespace(camera=cam)
        win._apply_exposure("uvc:fake",
                            {"kind": "uvc", "slots": ["s1"]},
                            False, 500.0)
        check(cam.calls == []
              and settings.device_exposure("uvc:fake") is None,
              "UVC 无曝光功能:不下发不持久化")

        # D435：worker 下发
        class FakeWorker:
            def __init__(self):
                self.calls = []
            def set_exposure(self, a, v):
                self.calls.append((a, v))

        fw = FakeWorker()
        win._apply_exposure("d435:fake",
                            {"kind": "d435", "worker": fw},
                            True, 0.0)
        check(fw.calls == [(True, 0.0)]
              and settings.device_exposure("d435:fake")["auto"] is True,
              "D435 下发 + 持久化")

        # S80M：stdin 行协议
        buf = io.BytesIO()
        win._workers["s80m:fake"] = {"kind": "s80m", "stdin": buf}
        win._s80m_set_exposure("s80m:fake", False, 300.5)
        win._s80m_set_exposure("s80m:fake", True, 0.0)
        check(buf.getvalue() == b"SET_EXPOSURE 300.5\nSET_EXPOSURE -1.0\n",
              f"S80M stdin 行协议: {buf.getvalue()!r}")

        # 恢复默认基线：首见落盘锁定，之后 worker 捕获值不覆盖
        check(settings.ensure_device_original("uvc:fake", True, 0.0) is True
              and settings.ensure_device_original("uvc:fake", False, 999.0) is False
              and settings.device_original("uvc:fake")
              == {"auto": True, "value": 0.0},
              "原始基线首见锁定，不覆盖")

        # UVC：无曝光入口 —— ☀ 对话框不再创建
        dlgs = []
        def _fake_dlg(*a, **kw):
            dlgs.append((a, kw))
            return SimpleNamespace(exec_=lambda: None,
                                   apply_requested=SimpleNamespace(
                                       connect=lambda f: None))
        win._workers["uvc:fake"] = {"kind": "uvc", "slots": ["s1"],
                                    "label": "Cam"}
        with patch("ui.main_window.ExposureDialog", side_effect=_fake_dlg):
            win._open_exposure_dialog("s1")
        check(dlgs == [], "UVC 无曝光对话框")

        # 录制中：对话框入口拒绝
        win._workers["d435:fake"] = {"kind": "d435", "slots": ["s_d"],
                                     "worker": fw, "label": "D435"}
        with patch.object(CameraPipeline, "is_recording",
                          new_callable=PropertyMock, return_value=True):
            win._open_exposure_dialog("s_d")
        check(fw.calls == [(True, 0.0)],
              "录制中不弹对话框（无新下发）")

        # 录制锁：已显示按钮随状态禁用/恢复
        w = win.grid.add_camera("slot_y", "Cam Y")
        w.set_exposure_button_visible(True)
        win._set_exposure_buttons_enabled(False)
        check(not w.exposure_btn.isEnabled(), "录制锁禁用曝光按钮")
        win._set_exposure_buttons_enabled(True)
        check(w.exposure_btn.isEnabled(), "录制结束恢复曝光按钮")
    finally:
        settings.DEVICE_PARAMS_FILE = orig
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        win.close()


def main():
    app = QApplication(sys.argv)
    test_settings_roundtrip()
    test_camera_worker_apply()
    test_camera_worker_thread_model()
    test_d435_worker_exposure()
    try:
        import pyrealsense2  # noqa: F401
        test_d435_apply_color_exposure()
    except ImportError:
        print("── 5. _apply_color_exposure ──\n  SKIP: pyrealsense2 未安装")
    test_widget_button(app)
    test_dialog(app)
    test_main_window_wiring(app)

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 曝光控制测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
