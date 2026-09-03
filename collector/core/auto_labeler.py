"""
自动标注器 —— Qt 薄封装，核心逻辑在 core.hand_tracking。
"""

from __future__ import annotations
import threading
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from core.hand_tracking import label_session as _ht_label_session


class AutoLabeler(QObject):
    """基于手部关键点数据的自动标注器。

    信号
    ----
    progress(int, int)    — (已处理帧数, 总帧数)
    finished(str, str)    — (session_path, error_msg)
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def label_session(self, session_path: str):
        """启动后台标注一个录制会话。"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run, args=(session_path,),
            daemon=True, name="auto-labeler",
        )
        self._thread.start()

    def cancel(self):
        self._running = False

    def _run(self, session_path: str):
        def _progress(cur, total):
            self.progress.emit(cur, total)

        def _cancelled():
            return not self._running

        result = _ht_label_session(
            session_path=session_path,
            progress_cb=_progress,
            cancel_check=_cancelled,
        )

        if not self._running:
            return
        if result.get("success"):
            self.finished.emit(session_path, "")
        else:
            self.finished.emit(session_path, result.get("error", "未知错误"))

        self._running = False
