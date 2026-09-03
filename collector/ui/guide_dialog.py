"""
使用说明窗口 —— 首次启动（或经 start.bat / start.sh 启动）时自动弹出，
按步骤介绍: 服务器 URL 配置 → 设备连接 → 录制 → 回放 → 上传。
可随时从 帮助 → 使用说明 重新打开。

中英文切换：正文为整段 HTML 文档，按当前界面语言选择对应模板
（窗口每次打开时重建，模态期间无法切语言，无需实时刷新）。
"""

from __future__ import annotations

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QStyle, QStyleOptionButton, QTextBrowser, QVBoxLayout,
)

from config import settings
from config.i18n import tr, lang_manager

GUIDE_HTML_ZH = """
<h2>使用步骤总览</h2>
<ol>
<li><b>登录</b> —— 启动时在登录窗口填写服务器地址与账号，或点「游客登录」</li>
<li><b>连接设备</b> —— 插入相机，设备面板自动识别，点击预览</li>
<li><b>开始录制</b> —— 选择任务，每路相机独立开始 / 停止</li>
<li><b>回放检查</b> —— 在录制历史中回放</li>
<li><b>上传数据</b> —— 录制完成后上传服务器</li>
</ol>

<h3>1. 登录与服务器地址</h3>
<p>启动时弹出<b>登录窗口</b>：</p>
<ul>
<li><b>账号登录</b>：输入管理员分配的<b>服务器地址、用户名、密码</b>，登录后可看到分配给您的任务</li>
<li><b>游客登录</b>：无账号时直接进入，仅可查看公共任务（管理员未指定用户的任务）</li>
</ul>
<p>配置自动保存（data/server_config.json），无需重复输入；任务页顶部可随时「切换账号」。
若暂未分配服务器地址，可先跳过，仅本地录制。</p>
<p><b>配置错误的表现</b>：任务列表加载失败、上传失败 —— 点任务页顶部「切换账号」修改地址后重试。</p>

<h3>2. 连接设备</h3>
<ul>
<li><b>UVC 摄像头</b>：插入 USB 后，左侧设备面板约 2 秒内自动出现，点击即可预览</li>
<li><b>Intel RealSense D435 / D405</b>：需先运行 <code>start.bat extras</code> 安装组件，再插入设备</li>
<li><b>双目相机（S80C）</b>：插入 USB 后，面板自动出现 👁 双目条目（面板显示为 FaysSense S80M），点击开启左右两路画面与深度热力图（第三格）；录制时同时保存深度热力图视频与原始深度数据；与 RealSense 不能同时开启</li>
<li><b>触觉手套（蓝牙）</b>：打开手套电源，并开启电脑蓝牙 → 面板「🧤 手套」组自动出现设备 → 点击该设备，主画面出现仿生手掌并自动连接（左 / 右手按广播名 L / R 自动识别）</li>
</ul>
<p><b>看不到设备时</b>：</p>
<ul>
<li>Windows「设置 → 隐私和安全性 → 相机」允许应用访问相机</li>
<li>手套不出现：确认手套电源已开、电脑蓝牙已开启（面板自动刷新，等待几秒）</li>
<li>关闭占用相机的软件（微信、腾讯会议等）</li>
<li>换 USB 接口（优先主板直连 3.0 口）</li>
</ul>

<h3>3. 录制</h3>
<ul>
<li>在任务选择页选择任务（服务器同步或本地任务）</li>
<li>每路相机有独立的「⏺ 开始」「⏹ 完成」按钮，录制时长显示在按钮上方</li>
<li><b>正常停止</b>（⏹ 完成）= 保存本次录制；<b>异常停止</b>（⛔）= 丢弃本次录制</li>
<li>工具栏「⏺ 全部录制」可同时开始所有已连接相机</li>
</ul>

<h3>4. 回放</h3>
<ul>
<li>左侧「录制历史」列出本地全部录制会话，点击即可回放</li>
<li>回放窗口支持拖动分割条调整画面大小与位置</li>
</ul>

<h3>5. 上传</h3>
<ul>
<li>录制完成后按设置自动上传，也可在录制历史中手动上传</li>
<li>上传前提：服务器地址填写正确 + 网络可达</li>
<li>上传显示成功但服务器上看不到 → 联系管理员检查服务器导入日志</li>
<li>大文件上传耗时较长属正常现象，请耐心等待，勿中途退出</li>
</ul>

<p><i>更详细的异常处理与排查步骤见项目目录下的 使用说明.md
（双击 start.bat help 直接打开）。</i></p>
"""

GUIDE_HTML_EN = """
<h2>Quick Start Overview</h2>
<ol>
<li><b>Sign in</b> — enter the server address and account in the login window at startup, or click "Guest Login"</li>
<li><b>Connect devices</b> — plug in cameras; they appear in the device panel automatically; click to preview</li>
<li><b>Start recording</b> — choose a task, then start / stop each camera independently</li>
<li><b>Review playback</b> — play back from the recording history</li>
<li><b>Upload data</b> — upload to the server after recording</li>
</ol>

<h3>1. Sign In & Server Address</h3>
<p>A <b>login window</b> pops up at startup:</p>
<ul>
<li><b>Account login</b>: enter the <b>server address, username and password</b> issued by the administrator; after login you can see the tasks assigned to you</li>
<li><b>Guest login</b>: continue without an account; only public tasks (tasks not assigned to any user) are visible</li>
</ul>
<p>The configuration is saved automatically (data/server_config.json) — no need to re-enter.
You can switch accounts anytime via "Switch account" at the top of the task page.
If you have not received a server address yet, you can skip this and record locally.</p>
<p><b>Symptoms of a wrong configuration</b>: task list fails to load, upload fails —
click "Switch account" at the top of the task page, fix the address and retry.</p>

<h3>2. Connecting Devices</h3>
<ul>
<li><b>UVC camera</b>: after plugging in the USB, it appears in the device panel on the left within ~2 seconds; click to preview</li>
<li><b>Intel RealSense D435 / D405</b>: run <code>start.bat extras</code> first to install the components, then plug in the device</li>
<li><b>Stereo camera (S80C)</b>: after plugging in the USB, a 👁 stereo entry appears automatically (shown as FaysSense S80M in the panel); click it to open the left/right views plus a depth heatmap (third tile); recording also saves a depth heatmap video and raw depth data; cannot be enabled together with RealSense</li>
<li><b>Haptic glove (Bluetooth)</b>: power on the glove and enable Bluetooth on the PC → the device appears in the "🧤 Gloves" group automatically → click it; the bionic hand view opens and connects automatically (left / right hand is recognized by the broadcast name L / R)</li>
</ul>
<p><b>If a device does not appear</b>:</p>
<ul>
<li>Windows "Settings → Privacy &amp; security → Camera": allow apps to access the camera</li>
<li>Glove not showing: make sure the glove is powered on and the PC Bluetooth is enabled (the panel refreshes automatically; wait a few seconds)</li>
<li>Close other apps that are using the camera (WeChat, Tencent Meeting, etc.)</li>
<li>Try another USB port (prefer a direct motherboard 3.0 port)</li>
</ul>

<h3>3. Recording</h3>
<ul>
<li>Select a task on the task selection page (server-synced or local task)</li>
<li>Each camera has its own "⏺ Start" / "⏹ Finish" buttons; the recording duration is shown above the buttons</li>
<li><b>Normal stop</b> (⏹ Finish) = save this recording; <b>abnormal stop</b> (⛔) = discard this recording</li>
<li>The toolbar "⏺ Record All" starts all connected cameras at once</li>
</ul>

<h3>4. Playback</h3>
<ul>
<li>The "Recording History" panel on the left lists all local recording sessions; click one to play it back</li>
<li>Drag the splitter bars in the playback window to resize and rearrange the views</li>
</ul>

<h3>5. Upload</h3>
<ul>
<li>After recording, upload runs automatically per settings, or upload manually from the recording history</li>
<li>Requirements: correct server address + network reachable</li>
<li>Upload says success but nothing appears on the server → contact the administrator to check the server import logs</li>
<li>Large files take a long time to upload — this is normal; please wait and do not quit midway</li>
</ul>

<p><i>For more detailed troubleshooting, see 使用说明_EN.md in the project folder.</i></p>
"""


def guide_html() -> str:
    """按当前界面语言返回使用步骤正文（HTML）。"""
    return GUIDE_HTML_EN if lang_manager.current == "en" else GUIDE_HTML_ZH


class VisibleCheckBox(QCheckBox):
    """qt-material 暗色主题下勾选指示不明显：指示框加显式描边/填充色，
    勾选时在框内自绘白色 ✓（与未勾选的空心框一眼可辨）。"""

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked():
            return
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        r = self.style().subElementRect(QStyle.SE_CheckBoxIndicator, opt, self)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor("#ffffff"), 2.0)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline([
            QPoint(r.left() + 3, r.center().y()),
            QPoint(r.center().x() - 1, r.bottom() - 4),
            QPoint(r.right() - 2, r.top() + 4),
        ])
        p.end()


class GuideDialog(QDialog):
    """使用步骤说明窗口。dont_show_checked() 供调用方持久化"下次不再显示"。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("使用说明"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(760, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)

        title = QLabel(f"{settings.APP_NAME} · {tr('使用步骤')}  v{settings.APP_VERSION}")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        layout.addWidget(title)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(guide_html())
        layout.addWidget(browser, 1)

        bottom = QHBoxLayout()
        self._dont_show_cb = VisibleCheckBox(
            tr("下次启动不再自动显示（可随时从 帮助→使用说明 重新打开并取消勾选）"))
        self._dont_show_cb.setStyleSheet(
            "QCheckBox { spacing: 8px; }"
            "QCheckBox::indicator { width: 18px; height: 18px; image: none;"
            " border: 2px solid #607d8b; border-radius: 4px; background: #1e272c; }"
            "QCheckBox::indicator:hover { border-color: #90a4ae; }"
            "QCheckBox::indicator:checked { background: #26a69a; border-color: #26a69a; }")
        bottom.addWidget(self._dont_show_cb)
        bottom.addStretch(1)
        close_btn = QPushButton(tr("关闭"))
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    def dont_show_checked(self) -> bool:
        return self._dont_show_cb.isChecked()
