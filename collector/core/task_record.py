"""
Task record persistence — data/tasks.json is the single source of truth.

- 进度分账（多电脑协同，后端聚合口径）：
    local_count    本机累计完成段数（increment_task_completed 加它）
    synced_count   后端已确认的本机段数（水位，只前进不后退）
    backend_count  最近一次已知后端全局数（merge/flush 用后端值无条件替换）
    completed_count = backend_count + (local_count - synced_count)
      —— 派生显示值，字段名保留使 UI 零改动，所有写路径经 _recount() 重算。
- 迁移回填：旧数据没有上述字段时，local/synced/backend 一次性回填为
  旧 completed_count（目录扫描兜底）——历史不补报、显示连续；
  此后以持久化值为准，上传后删本地文件也不影响进度。
- Task metadata (id, name, description, total_required, etc.) is persisted.
- Backend tasks are merged into the local record, keeping local-only tasks.
"""

from __future__ import annotations
import json
import os
import threading
from datetime import datetime

from config import settings

_TASKS_FILE = os.path.join(settings.DATA_DIR, "tasks.json")
_lock = threading.RLock()  # reentrant — merge_backend_tasks calls save_tasks/load_tasks

_SEED_TASKS = [
    {
        "id": "task_001",
        "name": "cylinder_grasping",
        "description": "Grasp cylinder parts with right hand, record 16x16 tactile sensor data.",
        "status": "pending",
        "total_required": 1000,
        "assigned_at": "2026-08-07T09:00:00Z",
        "params": {"object": "cylinder", "hand": "right"},
    },
    {
        "id": "task_002",
        "name": "cube_placement",
        "description": "Place cube objects from point A to point B using both hands.",
        "status": "pending",
        "total_required": 500,
        "assigned_at": "2026-08-06T14:30:00Z",
        "params": {"object": "cube", "hand": "both"},
    },
    {
        "id": "task_003",
        "name": "valve_rotation",
        "description": "Rotate circular valve 90 degrees with single hand, record torque and tactile feedback.",
        "status": "pending",
        "total_required": 800,
        "assigned_at": "2026-08-05T11:00:00Z",
        "params": {"object": "valve", "hand": "right", "angle": 90},
    },
]


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════

def load_tasks(identity: str | None = None) -> list[dict]:
    """Load all tasks. Filters out hidden tasks.

    completed_count 以持久化值为准（increment_task_completed 累计）；
    文件中缺该字段（旧版本数据）时一次性按目录扫描回填并写回，
    此后删会话文件不再影响进度。

    Status is auto-computed: pending(0%), in_progress(>0%), completed(100%).

    排序: 按 assigned_at 从新到旧；null/空/解析失败排最后。

    identity（新增，默认 None 兼容旧调用）:
      None   → 返回全部可见任务（现状行为）
      "guest" → 仅公共任务（无 assigned_user 字段）
      用户名  → 公共任务 + assigned_user == 该用户的任务

    注意: 除迁移回填外不写回 tasks.json —— hidden 任务必须保留在文件中，
    否则后端再次推送时无法识别为已删除，任务会死而复生。
    """
    with _lock:
        tasks = _read_raw()
    need_backfill = False
    for t in tasks:
        if _ensure_sync_fields(t):
            need_backfill = True
        _recount(t)
        t["status"] = _compute_status(t)
    if need_backfill:
        with _lock:
            _write({"tasks": tasks, "updated_at": datetime.now().isoformat()})
    visible = [t for t in tasks if not t.get("hidden")]
    visible = filter_by_identity(visible, identity)
    visible.sort(key=_assigned_at_key, reverse=True)  # 新→旧；null/空/非法排最后
    return visible


def filter_by_identity(tasks: list[dict], identity: str | None = None) -> list[dict]:
    """按身份过滤任务列表（纯函数，供 UI 对已有列表即时过滤）。

    identity 语义与 load_tasks 相同；返回新列表，不修改原列表。
    """
    if identity is None:
        return list(tasks)
    if identity == "guest":
        return [t for t in tasks if not _assigned_user(t)]
    return [t for t in tasks
            if not _assigned_user(t) or _assigned_user(t) == identity]


def save_tasks(tasks: list[dict]):
    """Persist task metadata，含 completed_count（录制完成次数持久化权威值）。"""
    clean = [dict(t) for t in tasks]
    with _lock:
        _write({"tasks": clean, "updated_at": datetime.now().isoformat()})


def merge_backend_tasks(backend_tasks: list[dict], view_scope: str | None = None) -> list[dict]:
    """Merge backend tasks into local record. Returns merged list.

    - Backend tasks update existing entries by id, new ones are added.
    - Local-only tasks are preserved.
    - Invalid/incomplete tasks are filtered out.
    - 同名去重: 同一 (name, assigned_user) 组出现多个可见任务时，只保留
      assigned_at 最新的一个（后端推送的优先胜出——"根据后台指令更新任务细节"），
      其余标 hidden。公共任务（均无 assigned_user）分组结果与旧版等价。
    - view_scope（新增）: 当前会话身份。"guest"/None 不触发撤销清除；
      用户名时，本地 assigned_user == 该用户名、且本轮后端未返回的任务
      （已被后端撤销）从本地清除。游客轮询绝不触发撤销清除——防止把
      他人指派任务误删。
    """
    with _lock:
        local = _read_raw()

    backend_ids = {_tid(t) for t in backend_tasks}

    # ── 阶段 0: 撤销清除（仅账号轮询；游客/无身份完全跳过）──
    if view_scope not in (None, "guest"):
        local = [
            lt for lt in local
            if not (_assigned_user(lt) == view_scope and _tid(lt) not in backend_ids)
        ]

    candidates: list[dict] = []

    # ── 阶段 1: 后端任务（按 id 合并，逻辑与现状一致）──
    for bt in backend_tasks:
        if not _is_valid(bt):
            continue
        bid = _tid(bt)
        local_match = _find(local, bid)
        backend_global = max(0, int(bt.get("completed_count") or 0))
        if local_match:
            # 保留本地覆盖字段
            for key in ("total_required", "params", "assigned_user"):
                if key in local_match and key not in bt:
                    bt[key] = local_match[key]
            # 进度分账：后端全局数权威；本机完成数/已同步水位沿用本地
            _ensure_sync_fields(local_match)
            bt["local_count"] = local_match["local_count"]
            bt["synced_count"] = local_match["synced_count"]
            bt["backend_count"] = backend_global
            # 保留删除标记：用户删过的任务不能被后端推送复活
            if local_match.get("hidden"):
                bt["hidden"] = True
        else:
            # 新任务：本机 0 段全同步；backend_count 取后端全局数
            # （保留 current_count 作新任务初值的旧语义）
            bt["local_count"] = 0
            bt["synced_count"] = 0
            bt["backend_count"] = backend_global
        candidates.append(bt)

    # ── 阶段 2: 本地独有任务 ──
    for lt in local:
        if not _is_valid(lt):
            continue
        if _tid(lt) not in backend_ids:
            candidates.append(dict(lt))

    # ── 阶段 3: 同名去重（(name, assigned_user) 墓碑）──
    groups: dict[tuple, list[dict]] = {}
    for c in candidates:
        groups.setdefault((c["name"], _assigned_user(c)), []).append(c)
    for group in groups.values():
        visible = [g for g in group if not g.get("hidden")]
        if len(visible) <= 1:
            continue  # 无碰撞（含整组已 hidden）→ 不动
        backend_visible = [v for v in visible if _tid(v) in backend_ids]
        pool = backend_visible if backend_visible else visible
        winner = max(pool, key=_assigned_at_key)
        for g in visible:
            if g is not winner:
                g["hidden"] = True  # 持久化墓碑，防止复活

    for t in candidates:
        _ensure_sync_fields(t)
        _recount(t)
        t["status"] = _compute_status(t)
    save_tasks(candidates)
    return load_tasks()


def mark_hidden(task_id: str):
    """Mark a task as hidden so it no longer appears in the UI."""
    with _lock:
        tasks = _read_raw()
        for t in tasks:
            if _tid(t) == task_id:
                t["hidden"] = True
                break
        _write({"tasks": tasks, "updated_at": datetime.now().isoformat()})


def refresh_progress() -> list[dict]:
    """Reload task list with persisted completion counts."""
    return load_tasks()


def increment_task_completed(task_name: str, task_id: str | None = None,
                             assigned_user: str | None = None) -> dict | None:
    """一次录制完成：把对应任务的录制完成次数持久化 +1，返回更新后的任务。

    匹配优先级：
      1) task_id 提供时按 id 精确匹配（须 name 一致）
      2) (name, assigned_user) 精确匹配（assigned_user 为 None 时匹配公共任务）
      3) 同名条目唯一时回退唯一匹配（兼容旧数据/手动临时任务名）
      4) 多名同名且身份不匹配 → 返回 None 不计数（跨身份串扰防护）
    task_id/assigned_user 由调用方从所选任务（_current_task）提供，
    而非会话身份——避免"登录用户录制公共任务却按用户名匹配"的错配。
    任务不在 tasks.json（手动临时任务名）时不计数，返回 None。
    会话目录随后被删除也不影响该计数——进度以持久化值为准。
    """
    if not task_name:
        return None
    with _lock:
        tasks = _read_raw()
        target = None
        if task_id:
            for t in tasks:
                if _tid(t) == task_id and t.get("name") == task_name:
                    target = t
                    break
        if target is None:
            for t in tasks:
                if (t.get("name") == task_name
                        and _assigned_user(t) == (assigned_user or None)):
                    target = t
                    break
        if target is None:
            matches = [t for t in tasks if t.get("name") == task_name]
            if len(matches) == 1:
                target = matches[0]
        if target is None:
            return None
        _ensure_sync_fields(target)
        target["local_count"] = int(target.get("local_count", 0) or 0) + 1
        _recount(target)
        target["status"] = _compute_status(target)
        _write({"tasks": tasks, "updated_at": datetime.now().isoformat()})
        return dict(target)


def pending_sync_tasks() -> list[dict]:
    """所有 local_count > synced_count 的任务快照（含 hidden，不按身份过滤——
    身份不符由服务端 401/403 处理，跳过本轮即可）。

    纯读路径：直接 _read_raw()，跳过未迁移记录（无 local_count），
    绝不触发迁移写盘（flush 只上报，不改变本地计数语义）。
    """
    with _lock:
        tasks = _read_raw()
    out = []
    for t in tasks:
        local = t.get("local_count")
        if not isinstance(local, int):
            continue
        synced = int(t.get("synced_count", 0) or 0)
        tid = _tid(t)
        if local > synced and tid:
            out.append({"id": tid, "name": t.get("name", ""),
                        "local_count": local, "synced_count": synced})
    return out


def mark_synced(task_id: str, synced_count: int, backend_count: int) -> dict | None:
    """flush 成功后回写同步状态。

    synced_count 必须是「快照水位 + 本次增量」（不是当前 local_count——
    快照与响应之间 UI 线程可能又录了一段）；水位只前进不后退；
    backend_count 用响应值无条件替换（服务端权威，迁移基线可能高于真实
    全局数，不能 max-前向）。返回更新后的任务。
    """
    with _lock:
        tasks = _read_raw()
        target = _find(tasks, task_id)
        if target is None or not isinstance(target.get("local_count"), int):
            return None
        if synced_count > (int(target.get("synced_count", 0) or 0)):
            target["synced_count"] = synced_count
        target["backend_count"] = max(0, int(backend_count or 0))
        _recount(target)
        target["status"] = _compute_status(target)
        _write({"tasks": tasks, "updated_at": datetime.now().isoformat()})
        return dict(target)


def task_by_id(task_id: str) -> dict | None:
    tasks = load_tasks()
    return _find(tasks, task_id)


# ═══════════════════════════════════════════════════════════
#  Internal
# ═══════════════════════════════════════════════════════════

def _read_raw() -> list[dict]:
    if not os.path.isfile(_TASKS_FILE):
        _init_seed()
    try:
        with open(_TASKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tasks", [])
    except (json.JSONDecodeError, OSError):
        _init_seed()
        return [dict(t) for t in _SEED_TASKS]


def _write(data: dict):
    os.makedirs(os.path.dirname(_TASKS_FILE), exist_ok=True)
    tmp = _TASKS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _TASKS_FILE)


def _init_seed():
    _write({"tasks": [dict(t) for t in _SEED_TASKS], "updated_at": datetime.now().isoformat()})


def _tid(task: dict) -> str:
    return task.get("id", task.get("task_id", ""))


def _assigned_user(task: dict):
    """任务的指派用户；null/空串/缺失统一归一化为 None（= 公共任务）。"""
    return task.get("assigned_user") or None


def _assigned_at_key(task: dict) -> tuple:
    """排序键: 带有效时间的排前，null/空/解析失败排后。
    (0, raw) 与 (1, ts) 首元素不同，跨类不会比较第二元素。"""
    raw = task.get("assigned_at") or ""
    if not raw:
        return (0, "")
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return (1, dt.timestamp())
    except ValueError:
        return (0, raw)  # 解析失败: 排最后，按原始字符串保序


def _find(tasks: list[dict], task_id: str) -> dict | None:
    for t in tasks:
        if _tid(t) == task_id:
            return t
    return None


def _is_valid(task: dict) -> bool:
    name = task.get("name", task.get("task_name", ""))
    tid = task.get("id", task.get("task_id", ""))
    return bool(name and name.strip()) and bool(tid and tid.strip()) and task.get("total_required", 0) > 0


def _compute_status(task: dict) -> str:
    total = task.get("total_required", 0)
    completed = task.get("completed_count", 0)
    if total <= 0:
        return "pending"
    if completed >= total:
        return "completed"
    if completed > 0:
        return "in_progress"
    return "pending"


def _count_with_fallback(task: dict) -> int:
    """任务的录制完成次数：优先持久化值，缺字段（旧版本数据）回退目录扫描。"""
    cur = task.get("completed_count")
    if isinstance(cur, int) and cur >= 0:
        return cur
    return _count_sessions(task.get("name", ""))


def _count_sessions(task_name: str) -> int:
    """扫描录制目录统计 episode 数 —— 仅用于旧数据迁移回填初值，非进度权威来源。

    v1.1.0 池化布局：list_task_episodes（videos/data 文件组扫描，与
    episodes 每段文件一致——删除时两者同步移除）；旧格式任务目录
    回退按会话子目录计数。
    """
    if not task_name:
        return 0
    task_dir = os.path.join(settings.RECORDING_DIR, task_name)
    if not os.path.isdir(task_dir):
        return 0
    try:
        from core.helpers import detect_session_format, list_task_episodes
        if detect_session_format(task_dir) == "pooled":
            return len(list_task_episodes(task_dir))
    except ImportError:
        pass
    count = 0
    try:
        for entry in os.listdir(task_dir):
            ses_dir = os.path.join(task_dir, entry)
            if not os.path.isdir(ses_dir):
                continue
            if os.path.isfile(os.path.join(ses_dir, "meta", "info.json")):
                count += 1
    except OSError:
        pass
    return count


def _ensure_sync_fields(task: dict) -> bool:
    """补全 local/synced/backend 三字段（迁移）；返回是否发生迁移（需写回）。

    迁移（幂等，字段存在且为合法 int 即跳过）：
      local_count   = 旧 completed_count（目录扫描兜底）
      synced_count  = local_count（历史不补报，避免重复计数）
      backend_count = local_count（历史基线保显示连续；首次 merge/flush
                      用真实全局数无条件替换）
    """
    changed = False
    local = task.get("local_count")
    if not isinstance(local, int) or local < 0:
        task["local_count"] = _count_with_fallback(task)
        changed = True
    synced = task.get("synced_count")
    if not isinstance(synced, int) or synced < 0:
        task["synced_count"] = task["local_count"]
        changed = True
    backend = task.get("backend_count")
    if not isinstance(backend, int) or backend < 0:
        task["backend_count"] = task["local_count"]
        changed = True
    return changed


def _recount(task: dict) -> int:
    """显示值 = backend_count + (local_count - synced_count)，写回 completed_count。

    防御：synced 不超 local（异常水位按已同步处理，宁可少显示不可多显示）。
    """
    local = int(task.get("local_count", 0) or 0)
    synced = min(int(task.get("synced_count", 0) or 0), local)
    task["completed_count"] = int(task.get("backend_count", 0) or 0) + (local - synced)
    return task["completed_count"]
