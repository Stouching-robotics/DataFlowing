"""
后台任务轮询服务 —— 定时从后端拉取可用任务列表，通过 Qt 信号通知 UI 更新。

设计模式参照 SyncService：QTimer 定时触发 + 后台 threading.Thread 执行 HTTP 请求。

信号:
  tasks_updated(list)      — 任务列表已刷新
  connection_status(bool)  — 后端连接状态 (True=已连接)
  error_occurred(str)      — 轮询出错
"""

from __future__ import annotations
import threading

import requests
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from config import settings
from core import task_record

CONNECT_TIMEOUT = 10  # HTTP 请求超时（秒）


def _normalize_tasks(raw: list[dict]) -> list[dict]:
    """Map backend fields to internal field names.

    Backend → Internal:
      current_count → completed_count
      status "active" → "in_progress"
      assigned_user 透传（null/空串/缺失统一归一化为 None = 公共任务）
    """
    tasks = []
    for t in raw:
        status = t.get("status", "active")
        if status == "active":
            status = "in_progress"
        tasks.append({
            "id": t.get("id", ""),
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "status": status,
            "total_required": t.get("total_required", 0),
            "completed_count": t.get("current_count", 0),
            "assigned_at": t.get("assigned_at", ""),
            "assigned_user": t.get("assigned_user") or None,
            "params": t.get("params"),
        })
    return tasks


class TaskService(QObject):
    """后台任务轮询服务。

    定时向服务器请求任务列表，通过 Qt 信号通知 UI 刷新。
    轮询在后台线程执行，不阻塞 UI。
    支持 Cookie-based JWT 认证：先 POST /api/v1/auth/login，再用 Session 轮询。

    身份三态（_identity）: None=未决（启动未选身份）/ "guest"=游客 / 用户名=已登录。
    身份切换经 adopt_login()/set_guest() 完成，_epoch 代次守卫丢弃
    后台线程中过期身份的响应。
    """

    tasks_updated = pyqtSignal(list)         # list[dict] — 任务列表
    connection_status = pyqtSignal(bool)      # True=已连接, False=断开
    error_occurred = pyqtSignal(str)          # 错误描述
    login_result = pyqtSignal(bool, str)      # True=登陆成功, 附带消息
    identity_changed = pyqtSignal(str)        # 身份确定/切换后发: "guest" 或用户名
    identity_expired = pyqtSignal()           # 登录态轮询遇 401/403 —— 会话过期
    progress_synced = pyqtSignal()            # 进度增量上报成功后本地计数已更新

    LOGIN_TIMEOUT = 8  # 登录对话框同步校验超时（秒）

    def __init__(self, server_url: str = "", parent=None):
        super().__init__(parent)
        self._url = (server_url or settings.SERVER_URL).rstrip("/")
        self._username = ""
        self._password = ""
        self._running = False
        self._interval_ms = getattr(settings, 'TASK_POLL_INTERVAL_MS', 5000)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._trigger_poll)
        self._timer.setInterval(self._interval_ms)

        # HTTP Session 维持 auth cookie
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "DAQ-SDK/1.0"})

        # 缓存最近一次成功拉取的任务列表
        self._cached_tasks: list[dict] = []
        self._logged_in = False

        # 身份状态机
        self._identity: str | None = None   # None / "guest" / 用户名
        self._epoch = 0                     # 身份代次：每次身份变化 +1

        # 进度增量上报（多电脑协同聚合）
        self._progress_supported = True     # 后端 404 → False 静默降级本地口径
        self._flush_fail: dict[str, int] = {}  # task_id → 连续失败次数（防日志刷屏）
        self._flush_lock = threading.Lock()    # 防 flush_now 与轮询 tick 并发重复 POST
        self._poll_fail = 0                 # 轮询连续失败次数（防日志刷屏）

    # ── 公开接口 ──────────────────────────────────────

    def start(self):
        """启动定时轮询（身份由 adopt_login/set_guest 确定后调用）。"""
        if self._running:
            return
        self._running = True
        self._timer.start()

    def stop(self):
        """停止定时轮询。"""
        self._running = False
        self._timer.stop()

    def poll_now(self):
        """手动触发一次拉取（不改变定时器状态）。"""
        self._trigger_login_and_poll()

    @staticmethod
    def verify_credentials(url: str, username: str, password: str):
        """独立 Session 同步校验登录凭据（供登录对话框使用，不触碰实例状态）。

        返回 (ok: bool, msg: str, cookies: RequestsCookieJar)。
        成功时 cookies 含 auth cookie，由 adopt_login() 接管。
        """
        session = requests.Session()
        session.headers.update({"User-Agent": "DAQ-SDK/1.0"})
        try:
            r = session.post(
                f"{url.rstrip('/')}/api/v1/auth/login",
                json={
                    "username": username,
                    "password": password,
                    "remember_me": True,
                },
                timeout=TaskService.LOGIN_TIMEOUT,
            )
            if r.status_code == 200:
                data = r.json()
                return True, str(data.get("message", "Login successful")), session.cookies
            try:
                detail = r.json().get("detail", f"HTTP {r.status_code}")
            except Exception:
                detail = f"HTTP {r.status_code}"
            return False, str(detail), session.cookies
        except requests.RequestException as e:
            return False, str(e)[:200], session.cookies

    def adopt_login(self, url: str, username: str, cookies) -> None:
        """采纳已验证的登录：cookie 拷入实例 Session，切换为账号身份并立即轮询。

        登录对话框已用 verify_credentials 校验过凭据，此处免二次 login。
        """
        self._url = (url or self._url).rstrip("/")
        self._session.cookies.clear()
        self._session.cookies.update(cookies)
        self._username = username
        self._password = ""
        self._logged_in = True
        self._identity = username
        self._epoch += 1
        self.identity_changed.emit(username)
        self._trigger_poll()

    def set_guest(self, url: str | None = None) -> None:
        """切换为游客身份：清 cookie，立即轮询（只应拉到公共任务）。"""
        if url:
            self._url = url.rstrip("/")
        self._session.cookies.clear()
        self._username = ""
        self._password = ""
        self._logged_in = False
        self._identity = "guest"
        self._epoch += 1
        self.identity_changed.emit("guest")
        self._trigger_poll()

    def current_identity(self) -> str | None:
        """当前身份：None（未决）/ "guest" / 用户名。"""
        return self._identity

    def set_server_url(self, url: str):
        """动态修改服务器地址并立即重新轮询。"""
        self._url = url.rstrip("/")
        self._session.cookies.clear()
        self._logged_in = False
        self._progress_supported = True  # 新后端可能已实现进度端点
        self._trigger_login_and_poll()

    def set_credentials(self, username: str, password: str):
        """更新登陆凭据。"""
        self._username = username
        self._password = password

    def set_server_and_credentials(self, url: str, username: str, password: str):
        """同时更新服务器地址和登陆凭据。"""
        self._url = url.rstrip("/")
        self._username = username
        self._password = password
        self._session.cookies.clear()
        self._logged_in = False
        self._trigger_login_and_poll()

    def set_interval(self, ms: int):
        """动态修改轮询间隔（毫秒）。"""
        self._interval_ms = ms
        self._timer.setInterval(ms)

    def cached_tasks(self) -> list[dict]:
        """返回缓存的任务列表（不发起网络请求）。"""
        return list(self._cached_tasks)

    def flush_now(self):
        """录制完成后的即时进度上报（后台线程，不阻塞 UI）。

        本机显示不依赖它（increment 已本地重算），只为其他机器秒级传播；
        正确性由轮询 tick 兜底。
        """
        threading.Thread(target=self._flush_progress, daemon=True).start()

    # ── 内部 ──────────────────────────────────────────

    def _post_progress(self, task_id: str, session_id: str, increment: int,
                       device: str):
        """POST /api/v1/device/tasks/progress（幂等：后端按 session_id 去重）。

        返回 (新全局数 | None, 错误码)；错误码: "" 成功 / "404" / "auth"(401/403)
        / "net" / "http"。
        """
        try:
            r = self._session.post(
                f"{self._url}/api/v1/device/tasks/progress",
                json={"task_id": task_id, "session_id": session_id,
                      "increment": increment, "device_name": device},
                timeout=CONNECT_TIMEOUT,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and isinstance(data.get("completed_count"), int):
                    return data["completed_count"], ""
                return None, "http"
            if r.status_code == 404:
                return None, "404"
            if r.status_code in (401, 403):
                return None, "auth"
            return None, "http"
        except requests.RequestException:
            return None, "net"

    def _note_fail(self, tid: str):
        """进度上报失败计数：首次与每 10 次记一条日志，避免刷屏。"""
        n = self._flush_fail.get(tid, 0) + 1
        self._flush_fail[tid] = n
        if n == 1 or n % 10 == 0:
            self.error_occurred.emit(f"任务进度上报失败(连续{n}次): {tid[:8]}")

    def _flush_progress(self):
        """水位合并上报：每任务一次 POST increment = local - synced。

        幂等键含设备名与水位（重试不重复计数）；崩溃安全由 tasks.json 落盘
        保证；401/403 跳过本轮不标记死（重新登录后自然恢复）。
        """
        if not self._progress_supported:
            return
        if not self._flush_lock.acquire(blocking=False):
            return  # 已有一次 flush 在途（tick 或 flush_now），让给先到者
        try:
            device = getattr(settings, 'DEVICE_NAME', 'EGO_001')
            changed = False
            for p in task_record.pending_sync_tasks():
                tid = p["id"]
                inc = p["local_count"] - p["synced_count"]
                session_id = f"{device}:{tid}:{p['synced_count']}"
                new_backend, err = self._post_progress(tid, session_id, inc, device)
                if err == "404":
                    self._progress_supported = False  # 后端未升级 → 静默降级本地口径
                    return
                if new_backend is None:
                    self._note_fail(tid)  # auth/net/http：跳过本轮，下个 tick 重试
                    continue
                self._flush_fail.pop(tid, None)
                task_record.mark_synced(tid, p["synced_count"] + inc, new_backend)
                changed = True
            if changed:
                self.progress_synced.emit()
        finally:
            self._flush_lock.release()

    def _trigger_poll(self):
        """在后台线程执行一次 HTTP 轮询（仅拉取任务，不登陆）。"""
        threading.Thread(target=self._poll, daemon=True).start()

    def _trigger_login_and_poll(self):
        """在后台线程执行：先登陆再拉取任务。"""
        threading.Thread(target=self._login_and_poll, daemon=True).start()

    def _login_and_poll(self):
        """登陆 → 拉取任务列表。"""
        if self._username and self._password:
            ok = self._do_login()
            if not ok:
                self.connection_status.emit(False)
                return
        # 登陆成功（或无需认证），标记已连接
        self.connection_status.emit(True)
        self._poll()

    def _do_login(self) -> bool:
        """POST /api/v1/auth/login 获取 auth_token cookie。返回是否成功。"""
        try:
            r = self._session.post(
                f"{self._url}/api/v1/auth/login",
                json={
                    "username": self._username,
                    "password": self._password,
                    "remember_me": True,
                },
                timeout=CONNECT_TIMEOUT,
            )
            if r.status_code == 200:
                self._logged_in = True
                self._identity = self._username
                self._epoch += 1
                data = r.json()
                msg = data.get("message", "Login successful")
                self.login_result.emit(True, msg)
                self.identity_changed.emit(self._username)
                return True
            else:
                self._logged_in = False
                try:
                    detail = r.json().get("detail", f"HTTP {r.status_code}")
                except Exception:
                    detail = f"HTTP {r.status_code}"
                self.login_result.emit(False, str(detail))
                self.error_occurred.emit(f"登陆失败: {detail}")
                return False
        except requests.RequestException as e:
            self._logged_in = False
            self.login_result.emit(False, str(e)[:200])
            self.error_occurred.emit(f"登陆失败: {e}")
            return False

    def _poll(self):
        """Poll GET /api/v1/device/tasks?device_name=xxx and emit task list."""
        epoch = self._epoch
        try:
            device = getattr(settings, 'DEVICE_NAME', 'EGO_001')
            r = self._session.get(
                f"{self._url}/api/v1/device/tasks",
                params={"device_name": device},
                timeout=CONNECT_TIMEOUT,
            )
            if epoch != self._epoch:
                return  # 身份已切换，丢弃过期响应
            if r.status_code == 200:
                data = r.json()
                raw = data.get("tasks", []) if isinstance(data, dict) else []
                if self._identity == "guest":
                    # 游客态防御性兜底：契约上服务端只回公共任务，
                    # 万一返回指派任务也一律丢弃
                    raw = [t for t in raw if not (t.get("assigned_user") or None)]
                tasks = _normalize_tasks(raw)
                self._cached_tasks = tasks
                self.tasks_updated.emit(tasks)
                self.connection_status.emit(True)
                self._poll_fail = 0   # 成功即清零连续失败计数
                if epoch == self._epoch:
                    # GET 成功才 flush —— 断网时 POST 必然失败，不浪费请求；
                    # 恢复后下一个成功 tick 内收敛
                    self._flush_progress()
            elif r.status_code in (401, 403):
                if self._identity == "guest":
                    # 游客态 401：旧后端未遵循契约（匿名应 200）——只报错不断身份，
                    # 本地公共任务照常显示，可随时账号登录
                    self.connection_status.emit(False)
                    self.error_occurred.emit(
                        f"游客访问被服务器拒绝 (HTTP {r.status_code})，请尝试账号登录")
                elif self._identity is not None:
                    # 登录态认证失效 → 身份置空，通知 UI 重登/降级游客
                    self._logged_in = False
                    self._identity = None
                    self._epoch += 1
                    self.connection_status.emit(False)
                    self.error_occurred.emit(f"认证失效 HTTP {r.status_code}")
                    self.identity_expired.emit()
            else:
                # 服务器错误（如 500）不改变连接状态，只记录错误
                self.error_occurred.emit(f"HTTP {r.status_code}")
        except requests.RequestException as e:
            if epoch != self._epoch:
                return
            # 网络错误不改变连接状态（可能是暂时断网），只记录——
            # 连续失败只记首条与每 10 条（服务器卡死时 5s 一条会刷屏）
            self._poll_fail += 1
            if self._poll_fail == 1 or self._poll_fail % 10 == 0:
                self.error_occurred.emit(
                    f"{str(e)[:180]}（连续{self._poll_fail}次）")
