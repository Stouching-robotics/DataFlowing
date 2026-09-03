"""热加载后端 — watchfiles 监听代码变化,自动重启 uvicorn(单进程,无 --reload)。

Windows 上 uvicorn --reload 的 spawn 子进程经常残留/端口冲突,导致
改完代码后运行中的还是旧版。本脚本改为:监听 app/web
文件变化 → 干净杀掉当前 uvicorn 进程 → 重新启动,避免手动重启。

用法: python scripts/hot_reload_backend.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import watchfiles

ROOT = Path(__file__).resolve().parent.parent
# .env 也在监视内:改配置(如 STORAGE_BACKEND)后自动重启生效
WATCH_DIRS = [ROOT / "app", ROOT / "web", ROOT / ".env"]
LOG_PATH = ROOT / "data" / "tmp" / "backend" / "backend-hot.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def start_server() -> subprocess.Popen:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app",
           "--host", "0.0.0.0", "--port", "8000"]
    log_file = open(LOG_PATH, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                            stdout=log_file, stderr=log_file)
    _log(f"started uvicorn pid={proc.pid}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    _log("stopping uvicorn...")
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def main() -> None:
    proc = start_server()
    last_restart = time.monotonic()
    try:
        for changes in watchfiles.watch(*WATCH_DIRS, step=500):
            if time.monotonic() - last_restart < 1.5:
                continue  # 防抖:合并连续变更
            files = sorted({str(p) for _, p in changes})
            _log("detected changes: " + ", ".join(files[:5]))
            stop_server(proc)
            proc = start_server()
            last_restart = time.monotonic()
    except KeyboardInterrupt:
        stop_server(proc)
        _log("hot reload stopped")


if __name__ == "__main__":
    main()
