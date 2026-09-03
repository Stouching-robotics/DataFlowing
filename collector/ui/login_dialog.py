"""
登录对话框 —— 启动时（或任务页点「切换账号」）弹出，确定会话身份：

- 账号登录：输入服务器地址 + 后端派发的用户名/密码，点「登录」；
  对话框内同步校验（TaskService.verify_credentials，独立 Session），
  失败就地提示可重试，成功后携带 auth cookie 返回。
- 游客登录：无账号时直接进入，只能看到公共任务
  （后端未指定用户的任务）；指定用户的任务不可见。
- 关闭窗口（Esc / X）= 取消：启动首屏时由调用方退出应用；
  应用内重开（切换账号/登录过期）时按游客降级。

结果经访问器读取：choice() / server_url() / username() / password() /
remember_checked() / cookies()（仅登录成功时非 None）。
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QDialog, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout,
)

from config import settings
from config.i18n import tr
from core.task_service import TaskService
from ui.guide_dialog import VisibleCheckBox

_LINE_EDIT_STYLE = (
    f"QLineEdit {{ background:{settings.COLOR_BG_WIDGET}; "
    f"color:{settings.COLOR_TEXT_PRIMARY}; "
    f"border:1px solid {settings.COLOR_BORDER}; "
    f"border-radius:4px; padding:6px 10px; font-size:13px; }}"
    f"QLineEdit:focus {{ border-color:{settings.COLOR_BTN_START}; }}"
)


class LoginDialog(QDialog):
    """账号 / 游客登录窗口。exec_() 后经访问器读取结果。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._choice = "guest"          # 默认：游客登录
        self._cancelled = False         # X/Esc 关闭（区别于游客登录）
        self._cookies = None            # 登录成功后的 auth cookie

        self.setWindowTitle(tr("用户登录"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setModal(True)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 18)
        layout.setSpacing(10)

        # ── 标题 ──
        title = QLabel(tr("登录"))
        title.setStyleSheet(
            f"font-size:18px; font-weight:bold; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; background:transparent;")
        layout.addWidget(title)

        subtitle = QLabel(tr("输入后端派发的账号登录，或点「游客登录」直接进入（仅可见公共任务）。"))
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"color:{settings.COLOR_TEXT_SECONDARY}; font-size:12px; "
            f"background:transparent;")
        layout.addWidget(subtitle)

        # ── 表单 ──
        form = QGridLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(10)

        saved_url = settings.load_server_url()
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText(settings.SERVER_URL)
        # 与出厂默认相同时不回填，让 placeholder 显示默认值
        if saved_url != settings.SERVER_URL:
            self._url_edit.setText(saved_url)
        self._url_edit.setStyleSheet(_LINE_EDIT_STYLE)

        self._user_edit = QLineEdit()
        self._user_edit.setPlaceholderText(tr("用户名"))
        remembered = settings.load_remembered_username()
        if remembered:
            self._user_edit.setText(remembered)
        self._user_edit.setStyleSheet(_LINE_EDIT_STYLE)

        self._pwd_edit = QLineEdit()
        self._pwd_edit.setPlaceholderText(tr("密码"))
        self._pwd_edit.setEchoMode(QLineEdit.Password)
        self._pwd_edit.setStyleSheet(_LINE_EDIT_STYLE)
        self._pwd_edit.returnPressed.connect(self._on_login_clicked)

        self._remember_cb = VisibleCheckBox(tr("记住账号"))
        self._remember_cb.setChecked(True)
        self._remember_cb.setStyleSheet(
            "QCheckBox { spacing: 8px; }"
            "QCheckBox::indicator { width: 18px; height: 18px; image: none;"
            " border: 2px solid #607d8b; border-radius: 4px; background: #1e272c; }"
            "QCheckBox::indicator:hover { border-color: #90a4ae; }"
            "QCheckBox::indicator:checked { background: #26a69a; border-color: #26a69a; }")

        form.addWidget(self._label(tr("服务器地址:")), 0, 0)
        form.addWidget(self._url_edit, 0, 1)
        form.addWidget(self._label(tr("用户名")), 1, 0)
        form.addWidget(self._user_edit, 1, 1)
        form.addWidget(self._label(tr("密码")), 2, 0)
        form.addWidget(self._pwd_edit, 2, 1)
        form.addWidget(self._remember_cb, 3, 1)
        layout.addLayout(form)

        # ── 错误提示（初始隐藏）──
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(
            f"color:{settings.COLOR_ABNORMAL}; font-size:12px; "
            f"background:transparent;")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        # ── 按钮行 ──
        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        self._guest_btn = QPushButton(tr("游客登录"))
        self._guest_btn.setCursor(Qt.PointingHandCursor)
        self._guest_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; "
            f"color:{settings.COLOR_TEXT_PRIMARY}; font-weight:bold; "
            f"border:1px solid {settings.COLOR_BTN_START}; "
            f"border-radius:4px; padding:7px 20px; font-size:13px; }}"
            f"QPushButton:hover {{ background:{settings.COLOR_BTN_HOVER}; }}")
        self._guest_btn.clicked.connect(self._on_guest_clicked)
        buttons.addWidget(self._guest_btn)

        buttons.addStretch(1)

        self._login_btn = QPushButton(tr("登录"))
        self._login_btn.setDefault(True)
        self._login_btn.setCursor(Qt.PointingHandCursor)
        self._login_btn.setStyleSheet(
            f"QPushButton {{ background:{settings.COLOR_BTN_START}; "
            f"color:white; font-weight:bold; "
            f"border:none; border-radius:4px; padding:7px 28px; font-size:13px; }}"
            f"QPushButton:hover {{ background:#388E3C; }}"
            f"QPushButton:disabled {{ "
            f"background:{settings.COLOR_BTN_DISABLED_BG}; "
            f"color:{settings.COLOR_BTN_DISABLED_TEXT}; }}")
        self._login_btn.clicked.connect(self._on_login_clicked)
        buttons.addWidget(self._login_btn)

        layout.addLayout(buttons)

    # ── 访问器（exec_() 返回后调用）────────────────────

    def choice(self) -> str:
        """"login"（点登录且校验成功）、"guest"（游客登录）或
        "cancel"（X/Esc 关闭窗口）。"""
        return "cancel" if self._cancelled else self._choice

    def reject(self):
        """X/Esc 关闭：标记取消（不再静默等同游客登录，由调用方决定
        退出应用或游客降级）。"""
        self._cancelled = True
        super().reject()

    def server_url(self) -> str:
        url = self._url_edit.text().strip()
        return url if url else settings.SERVER_URL

    def username(self) -> str:
        return self._user_edit.text().strip()

    def password(self) -> str:
        return self._pwd_edit.text()

    def remember_checked(self) -> bool:
        return self._remember_cb.isChecked()

    def cookies(self):
        """登录成功后的 auth cookie（登录失败/游客为 None）。"""
        return self._cookies

    # ── 内部 ───────────────────────────────────────────

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{settings.COLOR_TEXT_SECONDARY}; font-size:13px; "
            f"background:transparent;")
        return lbl

    def _show_error(self, msg: str):
        self._error_label.setText(tr("登录失败: {}", msg[:120]))
        self._error_label.setVisible(True)

    def _on_guest_clicked(self):
        self._choice = "guest"
        self.accept()

    def _on_login_clicked(self):
        username = self.username()
        if not username:
            self._show_error(tr("请输入用户名"))
            return
        # 同步校验（独立 Session，≤8s）；先禁用按钮并刷新界面，避免误以为卡死
        self._login_btn.setText(tr("登录中…"))
        self._login_btn.setEnabled(False)
        self._guest_btn.setEnabled(False)
        self._error_label.setVisible(False)
        QApplication.processEvents()
        try:
            ok, msg, cookies = TaskService.verify_credentials(
                self.server_url(), username, self.password())
        finally:
            self._login_btn.setText(tr("登录"))
            self._login_btn.setEnabled(True)
            self._guest_btn.setEnabled(True)
        if ok:
            self._cookies = cookies
            self._choice = "login"
            self.accept()
        else:
            self._show_error(msg)
