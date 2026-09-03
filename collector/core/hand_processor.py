"""
录制会话手部关键点后处理 —— Qt 薄封装，核心逻辑在 core.hand_tracking。
"""

from __future__ import annotations
import threading
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from core.hand_tracking import process_session as _ht_process_session


class SessionHandProcessor(QObject):
    """对已录制的会话视频逐帧跑手部关键点推理，结果写入 parquet。

    信号
    ----
    progress(int, int)         — (当前帧, 总帧数)
    finished(str, str)         — (session_path, error_msg)
                                  error_msg 为空表示成功
    status_changed(str)        — 状态变化通知
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str, str)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def process_session(self, session_path: str, detector: str = "",
                        det_device: str = "cuda", pose_device: str = "cuda",
                        mode: str = ""):
        """启动后台处理一个录制会话。"""
        if self._running:
            return

        self._track_mode = mode
        det_path = detector
        self._det_device = det_device
        self._pose_device = pose_device

        self._running = True
        self._thread = threading.Thread(
            target=self._process, args=(session_path, det_path),
            daemon=True, name="hand-processor",
        )
        self._thread.start()

    def cancel(self):
        """取消当前处理任务。"""
        self._running = False

    # ── 后台 ──────────────────────────────────────────

    def _process(self, session_path: str, det_path: str):
        def _progress(cur, total):
            self.progress.emit(cur, total)

        def _status(msg):
            self.status_changed.emit(msg)

        def _cancelled():
            return not self._running

        result = _ht_process_session(
            session_path=session_path,
            mode=getattr(self, '_track_mode', ''),
            detector=det_path or '',
            det_device=getattr(self, '_det_device', 'cuda'),
            pose_device=getattr(self, '_pose_device', 'cuda'),
            progress_cb=_progress,
            status_cb=_status,
            cancel_check=_cancelled,
        )

        if not self._running:
            self.status_changed.emit("cancelled")
        elif result.get("success"):
            self.finished.emit(session_path, "")
        else:
            self.finished.emit(session_path, result.get("error", "未知错误"))

        self._running = False
