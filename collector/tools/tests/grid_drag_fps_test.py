"""拖拽调位 + 实时 FPS 显示验证（offscreen，无硬件）。

Part A: CameraGrid 顶部信息条拖拽换位（事件过滤器 + 鼠标抓取）。
Part B: MainWindow 帧到达计数 → 每秒刷新 FPS 标签链路。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/grid_drag_fps_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt, QPointF, QEvent
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication

from ui.camera_grid import CameraGrid, DRAG_STRIP_H


def make_mouse(t, gpos, button=Qt.NoButton, buttons=Qt.NoButton):
    """构造 QMouseEvent（优先带 screenPos 的 7 参构造，兼容旧签名）。"""
    try:
        return QMouseEvent(t, QPointF(), QPointF(gpos), QPointF(gpos),
                           button, buttons, Qt.NoModifier)
    except TypeError:
        return QMouseEvent(t, QPointF(), QPointF(gpos),
                           button, buttons, Qt.NoModifier)


def wait(app, ms):
    t_end = time.time() + ms / 1000.0
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)


def part_a_grid_drag(app) -> int:
    grid = CameraGrid()
    wa = grid.add_camera("a", "A")
    wb = grid.add_camera("b", "B")
    wc = grid.add_camera("c", "C")
    grid.resize(900, 600)
    grid.show()
    wait(app, 200)
    assert grid.slot_ids() == ["a", "b", "c"], grid.slot_ids()

    # ── 1. 顶部信息条按下 → 开始拖拽（发送给视频子控件，过滤器拦截）──
    strip_pos = wa.mapToGlobal(wa.video_widget.rect().center())
    strip_pos.setY(wa.mapToGlobal(wa.rect().topLeft()).y() + 10)
    press = make_mouse(QEvent.MouseButtonPress, strip_pos,
                       Qt.LeftButton, Qt.LeftButton)
    consumed = QApplication.sendEvent(wa.video_widget, press)
    assert consumed, "信息条内按下应被过滤器消费（不触发缩放平移）"
    assert grid._drag_src is wa, "按下后应进入拖拽状态"

    # ── 2. 拖到 wc 上 → wc 高亮 ──
    target_pos = wc.mapToGlobal(wc.rect().center())
    move = make_mouse(QEvent.MouseMove, target_pos, Qt.NoButton, Qt.LeftButton)
    QApplication.sendEvent(wa, move)   # 抓取后鼠标事件直达源控件
    assert "4FC3F7" in (wc.styleSheet() or ""), "悬停目标应有高亮边框"

    # ── 3. 松开 → a/c 位置互换，高亮清除 ──
    release = make_mouse(QEvent.MouseButtonRelease, target_pos,
                         Qt.LeftButton, Qt.LeftButton)
    QApplication.sendEvent(wa, release)
    wait(app, 100)
    assert grid.slot_ids() == ["c", "b", "a"], grid.slot_ids()
    assert wc.styleSheet() == "", "松开后高亮应清除"
    assert grid._drag_src is None and grid._cursor_overridden is False

    # ── 4. 信息条下方按下 → 不进入拖拽（缩放平移区域不受影响）──
    bottom = wc.mapToGlobal(wc.rect().bottomLeft())
    bottom.setY(bottom.y() - 5)
    press2 = make_mouse(QEvent.MouseButtonPress, bottom,
                        Qt.LeftButton, Qt.LeftButton)
    QApplication.sendEvent(wc.video_widget, press2)
    assert grid._drag_src is None, "信息条下方按下不应进入拖拽状态"
    QApplication.sendEvent(wc.video_widget,
                           make_mouse(QEvent.MouseButtonRelease, bottom,
                                      Qt.LeftButton, Qt.LeftButton))
    wait(app, 100)
    assert grid.slot_ids() == ["c", "b", "a"], "未拖拽时顺序不应变化"

    # ── 5. 松开在原画面（拖到自身）→ 顺序不变 ──
    strip_pos2 = wb.mapToGlobal(wb.rect().topLeft())
    strip_pos2.setY(strip_pos2.y() + 10)
    QApplication.sendEvent(wb.video_widget,
                           make_mouse(QEvent.MouseButtonPress, strip_pos2,
                                      Qt.LeftButton, Qt.LeftButton))
    self_pos = wb.mapToGlobal(wb.rect().center())
    QApplication.sendEvent(wb, make_mouse(QEvent.MouseMove, self_pos,
                                          Qt.NoButton, Qt.LeftButton))
    QApplication.sendEvent(wb, make_mouse(QEvent.MouseButtonRelease, self_pos,
                                          Qt.LeftButton, Qt.LeftButton))
    wait(app, 100)
    assert grid.slot_ids() == ["c", "b", "a"], grid.slot_ids()

    # ── 6. 移除画面后无拖拽残留 ──
    grid.remove_camera("c")
    wait(app, 100)
    assert grid.slot_ids() == ["b", "a"]
    grid.close()
    print("PASS: Part A 拖拽调位（换位/高亮/信息条外不误触/自拖不动/移除清理）")
    return 0


def part_b_fps_labels(app) -> int:
    from ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    wait(app, 300)

    w = win.grid.add_camera("test_a", "T")
    assert w.fps_label.text() == "FPS: --", w.fps_label.text()

    # 3 帧到达 → 定时器回调应把标签刷成 3
    for _ in range(3):
        win._note_frame_arrival("test_a")
    win._update_fps_labels()
    assert w.fps_label.text() == "FPS: 3", w.fps_label.text()

    # 帧停 → 1s 后计数归零，标签归 0（模拟窗口老化）
    time.sleep(1.05)
    win._update_fps_labels()
    assert w.fps_label.text() == "FPS: 0", w.fps_label.text()

    # 设备关闭清理计数环
    win._fps_ring.pop("test_a", None)
    win.grid.remove_camera("test_a")
    win.close()
    wait(app, 300)
    print("PASS: Part B 帧到达计数 → FPS 标签刷新（含归零）")
    return 0


def main():
    app = QApplication(sys.argv)
    rc = part_a_grid_drag(app)
    if rc == 0:
        rc = part_b_fps_labels(app)
    return rc


if __name__ == "__main__":
    sys.exit(main())
