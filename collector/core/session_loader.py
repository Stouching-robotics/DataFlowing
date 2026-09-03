"""
会话后台加载器 —— 会话时间线与手部关键点的 parquet 读取全部在 worker 线程。

与 UI 无关（QtCore QObject + 信号，不依赖 QtWidgets）：
  - SessionLoader  后台加载（gen 防过期协议由调用方维护）
  - 可选手部关键点经 core.hand_tracking 惰性导入
"""

from __future__ import annotations

import threading

from PyQt5.QtCore import QObject, pyqtSignal

from core.session_timeline import load_timeline, SensorTimeline

# 手部关键点（可选模块；core.hand_tracking 引入 ultralytics/torch 等重依赖）
try:
    from core.hand_tracking import load_hand_kpts as _load_hand_kpts
    from core.hand_tracking import load_hand_kpts_pooled as _load_hand_kpts_pooled
    _HAND_KPTS_AVAILABLE = True
except ImportError:
    _load_hand_kpts = None
    _load_hand_kpts_pooled = None
    _HAND_KPTS_AVAILABLE = False


class SessionLoader(QObject):
    """后台加载会话时间线与手部关键点（parquet 读取全部在 worker 线程）。

    信号
    ----
    finished(int, object)  — (gen, {"timeline": SensorTimeline, "hand_kpts": dict})
                             payload 按引用传递，worker emit 后不再触碰
    failed(int, str)       — (gen, error)

    VideoCapture 不在 worker 中创建/读取（OpenCV FFmpeg 后端不承诺跨线程
    安全），视频打开与播放全部留在调用方主线程。
    """

    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._pending = None    # 排队的下一次加载请求（最新覆盖旧的）
        self._lock = threading.Lock()
        self._thread: threading.Thread = None

    def start(self, gen: int, session_dir: str, sensor_names: list[str],
              load_kpts: bool = False, load_timeline: bool = True,
              episode_index: int = 0):
        """启动后台加载。

        已有加载在跑时把请求排队（新请求覆盖旧请求——过期结果本就会被
        gen 丢弃，只跑最新一次即可）。

        load_timeline=False → kpts-only 模式（手部处理完成后重载关键点）。
        episode_index > 0 时按池化布局读本 episode 的文件组。
        """
        with self._lock:
            if self._running:
                self._pending = (gen, session_dir, sensor_names,
                                 load_kpts, load_timeline, episode_index)
                return
            self._running = True
        self._spawn(gen, session_dir, sensor_names, load_kpts, load_timeline,
                    episode_index)

    def is_running(self) -> bool:
        return self._running

    def _spawn(self, gen, session_dir, sensor_names, load_kpts, load_timeline,
               episode_index):
        self._thread = threading.Thread(
            target=self._run,
            args=(gen, session_dir, sensor_names, load_kpts, load_timeline,
                  episode_index),
            daemon=True, name="session-loader",
        )
        self._thread.start()

    # ── 后台 ──────────────────────────────────────────

    def _run(self, gen: int, session_dir: str, sensor_names: list[str],
             load_kpts: bool, want_timeline: bool, episode_index: int):
        try:
            timeline = None
            if want_timeline:
                timeline = load_timeline(session_dir, sensor_names,
                                         episode_index)
            hand_kpts = {}
            if load_kpts and _HAND_KPTS_AVAILABLE:
                if episode_index > 0 and _load_hand_kpts_pooled:
                    # 池化布局：镜像目录 keypoints_output/<task>/episode_NNNNNN/
                    hand_kpts = (_load_hand_kpts_pooled(
                        session_dir, episode_index) or {})
                elif _load_hand_kpts:
                    hand_kpts = _load_hand_kpts(session_dir) or {}
            self.finished.emit(gen, {"timeline": timeline, "hand_kpts": hand_kpts})
        except Exception as e:
            self.failed.emit(gen, str(e)[:300])
        finally:
            with self._lock:
                self._running = False
                pending = self._pending
                self._pending = None
                if pending is not None:
                    self._running = True
            if pending is not None:
                self._spawn(*pending)
