"""热加载 Worker — watchfiles 监听代码变化,自动重启全部 worker 进程。

worker 本身无热重载(`python -m worker` 启动时加载模块代码,改代码必须重启)。
本脚本与 hot_reload_backend.py 同模式:监听 app/worker/.env 文件变化 →
干净杀掉全部 worker → 重新启动,避免"改了代码 worker 还在跑旧逻辑"。

中断安全:worker 执行中被 kill 的任务,后端靠租约(默认 90s)到期后重新
放回队列,由重启后的 worker 重新领取 —— 任务不丢,最多从头重跑一遍。

用法:
  python scripts/hot_reload_worker.py
  (API Key 未设置时自动从项目 .env 读 WORKER_API_KEY,不会打印)

可选环境变量:
  EGODATA_WORKER_COUNT=3        worker 数量(默认 3,与后端进程布局一致)
  EGODATA_WORKER_API_KEY=...    显式指定(优先于 .env)
  EGODATA_SERVER_URL=...        默认 http://127.0.0.1:8000
  EGODATA_DEVICE=auto           上报的设备类型
"""

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import watchfiles

ROOT = Path(__file__).resolve().parent.parent
# .env 也在监视内:改配置(如 WORKER_API_KEY / 临时目录)后自动重启生效
WATCH_DIRS = [ROOT / "app", ROOT / "worker", ROOT / ".env"]
LOG_PATH = ROOT / "data" / "tmp" / "worker" / "worker-hot.log"
WORKER_COUNT = int(os.getenv("EGODATA_WORKER_COUNT", "3"))


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _resolve_api_key() -> str:
    env = _read_env_file(ROOT / ".env")
    return (os.getenv("EGODATA_WORKER_API_KEY")
            or env.get("EGODATA_WORKER_API_KEY")
            or env.get("WORKER_API_KEY")
            or "")


API_KEY = _resolve_api_key()


def start_worker(index: int, generation: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))
    env["EGODATA_WORKER_API_KEY"] = API_KEY
    env.setdefault("EGODATA_SERVER_URL", "http://127.0.0.1:8000")
    env.setdefault("EGODATA_DEVICE", "auto")
    host = os.getenv("EGODATA_WORKER_ID_PREFIX", f"linux-{socket.gethostname()}")
    env["EGODATA_WORKER_ID"] = f"{host}-g{generation}-{index + 1}"
    # 每个 worker 独立临时子目录,避免共目录残留冲突
    env.setdefault("EGODATA_WORK_DIR",
                   str(ROOT / "data" / "tmp" / "worker" / f"w{index + 1}"))
    log_path = LOG_PATH.parent / f"worker-hot-{index + 1}.log"
    log_file = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "worker"],
                            cwd=str(ROOT), env=env,
                            stdout=log_file, stderr=log_file)
    _log(f"started worker #{index + 1} pid={proc.pid} id={env['EGODATA_WORKER_ID']}")
    return proc


def stop_worker(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
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
    if not API_KEY:
        print("[error] EGODATA_WORKER_API_KEY 未设置,且 .env 里没有 WORKER_API_KEY",
              file=sys.stderr)
        sys.exit(1)
    generation = uuid.uuid4().hex[:10]
    procs = [start_worker(i, generation) for i in range(WORKER_COUNT)]
    last_restart = time.monotonic()
    try:
        for changes in watchfiles.watch(*WATCH_DIRS, step=500):
            if time.monotonic() - last_restart < 1.5:
                continue  # 防抖:合并连续变更
            files = sorted({str(p) for _, p in changes})
            _log("detected changes: " + ", ".join(files[:5]))
            for p in procs:
                stop_worker(p)
            generation = uuid.uuid4().hex[:10]
            procs = [start_worker(i, generation) for i in range(WORKER_COUNT)]
            last_restart = time.monotonic()
    except KeyboardInterrupt:
        for p in procs:
            stop_worker(p)
        _log("hot reload stopped")


if __name__ == "__main__":
    main()
