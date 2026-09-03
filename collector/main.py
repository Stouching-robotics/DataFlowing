"""
DAQ 视频管线 — 程序入口。

多路摄像机实时监控与录制系统：
  - 可拖拽调整大小的多画面网格布局
  - 每路摄像机独立的录制控制（正常停止保存 / 异常停止丢弃）
  - 录制时长实时显示在开始按钮上方
  - 录制历史记录追踪
  - 中英文界面切换

用法:
    python main.py
"""

import sys
import os

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)

# 先导入 torch，再导入 PyQt5（避免 DLL 加载冲突；torch 不可用时跳过）
try:
    import torch  # noqa: F401
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication, QDesktopWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase  # noqa: F401 — qt-material 需要
from qt_material import apply_stylesheet

from ui.main_window import MainWindow

# ── Qt 平台插件路径修复 ───────────────────────────────
# cv2（opencv-python 预编译轮子）在首次 import 时会把
# QT_QPA_PLATFORM_PLUGIN_PATH 覆盖成它自带的 qt/plugins 目录
# （见 venv/.../cv2/config-3.py），其 xcb 插件与 PyQt5 的 Qt 库
# 不兼容 → QApplication 创建即崩溃（"Could not load the Qt
# platform plugin xcb"）。必须在所有 import 之后、QApplication
# 创建之前把路径抢回 PyQt5 自带插件目录（cv2 只写一次，不会再覆盖）。
import PyQt5
_qt_platforms = os.path.join(
    os.path.dirname(os.path.abspath(PyQt5.__file__)),
    "Qt5", "plugins", "platforms")
if os.path.isdir(_qt_platforms):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_platforms


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("DAQ 视频管线")

    # ── Qt-Material 暗色主题 ──────────────────────────
    apply_stylesheet(app, theme='dark_teal.xml')

    window = MainWindow()
    # 先确定会话身份（账号登录/游客登录，身份决定任务页可见内容），
    # 确认前不显示主窗口（登录框弹在主窗口之前，不露出任务页），
    # 之后才是首次启动或经 start.bat/start.sh 启动（DAQ_SHOW_GUIDE=1）时的使用步骤窗口
    if not window.maybe_show_login():
        # 启动首屏关闭登录窗口 = 直接退出（不再静默进游客模式）
        window.close()   # 触发 closeEvent 释放已初始化的资源
        sys.exit(0)
    window.show()
    # 主窗口在登录框之后才 show，WM 不再把它当首窗居中（常落左上角）→ 显式居中
    geo = window.frameGeometry()
    geo.moveCenter(QDesktopWidget().availableGeometry().center())
    window.move(geo.topLeft())
    window.maybe_show_guide()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
