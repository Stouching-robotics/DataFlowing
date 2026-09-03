#!/usr/bin/env python3
"""EgoData Data Acquisition 一键部署脚本.

跨平台 (Linux / Windows)，纯标准库实现：

    python3 deploy.py                      # 完整一键部署
    python3 deploy.py --check-only         # 只做硬件能力体检, 零副作用
    python3 deploy.py --no-services        # 前台监督模式 (不装 systemd/自启)
    python3 deploy.py --skip-vllm          # 跳过本地 VLM (vLLM)
    python3 deploy.py --download-model     # 顺带下载 Qwen3-VL-8B-FP8 权重 (9.9GB)
    python3 deploy.py --download-model --hf-mirror   # 国内网络走 hf-mirror

流程: 能力体检 → .env 补全 → venv 安装 → (vLLM) → 端口避让 → 手部骨骼冒烟测试
      → (workflows API 模式 patch) → 启动服务 → 打印访问报告.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"

VENV_NAME = ".venv-windows" if IS_WINDOWS else ".venv-linux"
VENV_DIR = PROJECT_ROOT / VENV_NAME
VENV_PY = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
VENV_PIP = [str(VENV_PY), "-m", "pip"]

VENV_LLM_DIR = PROJECT_ROOT / ".venv-llm"
VENV_LLM_BIN = VENV_LLM_DIR / ("Scripts" if IS_WINDOWS else "bin")

MODEL_DIR = PROJECT_ROOT / "models" / "llm" / "Qwen3-VL-8B-Instruct-FP8"
MODEL_REPO = "Qwen/Qwen3-VL-8B-Instruct-FP8"
GESTURE_TASK = PROJECT_ROOT / "models" / "gesture_recognizer.task"
GESTURE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/"
                    "gesture_recognizer/gesture_recognizer/float16/latest/"
                    "gesture_recognizer.task")
WORKFLOWS_JSON = PROJECT_ROOT / "data" / "state" / "workflows.json"

LOG_DIR = PROJECT_ROOT / "data" / "tmp" / "deploy"
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_BACKEND_PORT = 8000
DEFAULT_VLLM_PORT = 8001

# ---------------------------------------------------------------------------
# 输出辅助
# ---------------------------------------------------------------------------

def say(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"⚠️  {msg}", flush=True)


def fail(msg: str) -> str:
    """Print an error line and return it (caller decides whether to raise)."""
    print(f"❌  {msg}", flush=True)
    return msg


def ask(prompt: str) -> str:
    """交互确认; 非交互环境 (EOF/管道) 视为拒绝."""
    try:
        return input(prompt)
    except EOFError:
        return ""


def phase(title: str) -> None:
    say(f"\n{'─' * 62}\n── 阶段: {title}\n{'─' * 62}")


def run_cmd(cmd, cwd=None, env=None, timeout=None, check=False, capture=False,
            stdout=None, stderr=None):
    """Run a command, streaming output; returns CompletedProcess.

    ``stdout`` / ``stderr`` 可传文件对象 (与 capture 二选一), 用于把
    pip 等长输出重定向到日志文件。
    """
    say(f"    $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
        env=env, timeout=timeout, check=check,
        capture_output=capture, text=True,
        stdout=stdout, stderr=stderr,
    )


def run_quiet(cmd, cwd=None, env=None, timeout=60):
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
            env=env, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)


def venv_exists() -> bool:
    return VENV_PY.is_file()


def venv_imports(module: str) -> bool:
    if not venv_exists():
        return False
    rc, _, _ = run_quiet([str(VENV_PY), "-c", f"import {module}"], timeout=120)
    return rc == 0


# ---------------------------------------------------------------------------
# 网络 / 端口
# ---------------------------------------------------------------------------

def lan_ip() -> str:
    """探测本机局域网 IP (不发包)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        if IS_WINDOWS:
            rc, out, _ = run_quiet(["ipconfig"])
            for m in re.finditer(r"IPv4[^:]*:\s*([\d.]+)", out):
                if not m.group(1).startswith("127."):
                    return m.group(1)
        else:
            rc, out, _ = run_quiet(["hostname", "-I"])
            for tok in out.split():
                if not tok.startswith("127."):
                    return tok
    except Exception:
        pass
    return "127.0.0.1"


def port_free(port: int) -> bool:
    """试绑定即关闭, 判断端口是否空闲 (跨平台, 无需 root)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def port_owner_cmdline(port: int) -> str:
    """尽力识别占用端口的进程命令行; 识别不到返回空串."""
    try:
        if IS_WINDOWS:
            rc, out, _ = run_quiet(["netstat", "-ano"], timeout=30)
            pids = set()
            for line in out.splitlines():
                parts = line.split()
                if (len(parts) >= 5 and parts[0].upper() in ("TCP", "UDP")
                        and parts[1].endswith(f":{port}")
                        and parts[3] == "LISTENING"):
                    pids.add(parts[-1])
            for pid in pids:
                rc2, out2, _ = run_quiet(
                    ["wmic", "process", "where", f"ProcessId={pid}",
                     "get", "CommandLine"], timeout=30)
                if out2.strip():
                    return out2
        else:
            rc, out, _ = run_quiet(["ss", "-tlnp"], timeout=30)
            for line in out.splitlines():
                if f":{port} " in line and "pid=" in line:
                    m = re.search(r"pid=(\d+)", line)
                    if not m:
                        continue
                    try:
                        raw = Path(f"/proc/{m.group(1)}/cmdline").read_bytes()
                        return raw.replace(b"\0", b" ").decode(errors="replace")
                    except OSError:
                        continue
    except Exception:
        pass
    return ""


def port_owned_by_us(port: int, unit_name: str | None = None) -> bool:
    """端口被本项目自身进程占用 (systemd 单元 active 或 cmdline 含项目路径)."""
    if unit_name:
        rc, out, _ = run_quiet(
            ["systemctl", "--user", "is-active", unit_name], timeout=30)
        if rc == 0 and out.strip() == "active":
            return True
    owner = port_owner_cmdline(port)
    root = str(PROJECT_ROOT)
    return root in owner or VENV_NAME in owner or ".venv-llm" in owner


def pick_free_port(start: int, skip: set[int] | None = None,
                   limit: int = 200) -> int:
    """从 start 起向后找第一个空闲端口 (跳过 skip 集合)."""
    skip = skip or set()
    port = start
    for _ in range(limit):
        if port not in skip and port_free(port):
            return port
        port += 1
    return -1


def instances_running() -> list[str]:
    """检测本项目是否有实例正在运行 (backend/vllm/workers), 尽力而为."""
    running = []
    if not port_free(DEFAULT_BACKEND_PORT) \
            and port_owned_by_us(DEFAULT_BACKEND_PORT):
        running.append("backend")
    if not port_free(DEFAULT_VLLM_PORT) \
            and port_owned_by_us(DEFAULT_VLLM_PORT):
        running.append("vllm")
    try:
        if IS_WINDOWS:
            rc, out, _ = run_quiet(
                ["wmic", "process", "where", "name like '%python%'",
                 "get", "CommandLine"], timeout=30)
        else:
            rc, out, _ = run_quiet(
                ["pgrep", "-af", "hot_reload_worker.py"], timeout=30)
        if "hot_reload_worker.py" in out:
            running.append("workers")
    except Exception:
        pass
    return running


# ---------------------------------------------------------------------------
# 阶段 1: 能力体检
# ---------------------------------------------------------------------------

def _gpu_check() -> dict:
    """返回 {verdict, name, vram_mib, cc, driver, detail}."""
    info = {"verdict": "skip", "name": "", "vram_mib": 0,
            "cc": 0.0, "driver": "", "detail": ""}
    if IS_WINDOWS:
        info["detail"] = "Windows 不支持 vLLM, 本地 VLM 不可用"
        return info
    if not shutil.which("nvidia-smi"):
        info["detail"] = "未找到 nvidia-smi (无 NVIDIA 驱动)"
        return info
    rc, out, _ = run_quiet(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
         "--format=csv,noheader"], timeout=60)
    if rc != 0 or not out.strip():
        info["detail"] = "nvidia-smi 查询失败"
        return info
    row = out.strip().splitlines()[0]
    parts = [p.strip() for p in row.split(",")]
    try:
        name = parts[0]
        vram = int(re.sub(r"[^\d]", "", parts[1]))
        cc = float(parts[2])
        driver = parts[3] if len(parts) > 3 else ""
        info.update(name=name, vram_mib=vram, cc=cc, driver=driver)
    except (ValueError, IndexError):
        info["detail"] = f"无法解析 nvidia-smi 输出: {row!r}"
        return info
    # Qwen3-VL-8B-FP8: 权重 9.9GB + 16K KV cache + CUDA graphs;
    # FP8 需 Ampere+ (CC >= 8.0); 16GB 以上放心跑 (0.70 显存占比留 worker 份额)
    if vram >= 16000 and cc >= 8.0:
        info["verdict"] = "ok"
    elif (12000 <= vram < 16000) or (7.0 <= cc < 8.0):
        info["verdict"] = "ask"
        info["detail"] = ("显存/算力偏紧, 建议降配运行: "
                          "--gpu-memory-utilization 0.55 --max-model-len 8192")
    else:
        info["verdict"] = "skip"
        info["detail"] = f"显存 {vram}MB / CC {cc} 不足以跑 Qwen3-VL-8B-FP8"
    return info


def _python_dep_compat() -> list[str]:
    """检查 requirements.txt 中带版本锁定的关键包在当前 Python 下有无可用 wheel.

    用 pip --dry-run 实测 (零副作用, 不下载); 只检查已知存在平台/版本
    wheel 差异的包 (mediapipe 0.10.21 无 cp313 wheel 是已知坑), 其余
    交给 pip 安装阶段自行报错。网络错误不算不兼容。
    """
    problems = []
    try:
        text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    except OSError:
        return problems
    for m in re.finditer(r"^([\w.-]+)==([\w.*]+)", text, re.M):
        pkg, ver = m.group(1), m.group(2)
        if pkg != "mediapipe":
            continue
        rc, _, err = run_quiet(
            [sys.executable, "-m", "pip", "install", "--dry-run",
             "--only-binary", ":all:", "--no-deps", f"{pkg}=={ver}"],
            timeout=180)
        if rc != 0 and ("No matching distribution" in err
                        or "not find a version" in err):
            problems.append(
                f"{pkg}=={ver} 在当前 Python {sys.version.split()[0]} 无可用 wheel")
    return problems


def phase_check() -> dict:
    """硬件能力体检, 返回报告 dict (零副作用)."""
    report = {}

    report["os"] = f"{platform.system()} {platform.release()}"
    report["python"] = sys.version.split()[0]

    gpu = _gpu_check()
    report["gpu"] = gpu

    # 内存
    ram_detail = ""
    try:
        if IS_WINDOWS:
            rc, out, _ = run_quiet(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                timeout=30)
            m = re.search(r"(\d+)", out)
            if m:
                report["ram_gb"] = round(int(m.group(1)) / 1024 ** 3, 1)
                ram_detail = f"{report['ram_gb']}GB"
        else:
            rc, out, _ = run_quiet(["free", "-g"], timeout=30)
            #              total        used        free ...
            parts = out.splitlines()[1].split()
            report["ram_gb"] = int(parts[1])
            ram_detail = f"共 {parts[1]}GB / 可用 {parts[-1]}GB"
    except Exception:
        report["ram_gb"] = 0
    if report.get("ram_gb", 0) >= 16:
        report["ram"] = ("ok", ram_detail)
    elif report.get("ram_gb", 0) > 0:
        report["ram"] = ("warn", ram_detail)
    else:
        report["ram"] = ("unknown", "无法探测内存")

    # 磁盘
    du = shutil.disk_usage(PROJECT_ROOT)
    free_gb = round(du.free / 1024 ** 3, 1)
    pct = round(du.used / du.total * 100)
    report["disk"] = {
        "free_gb": free_gb, "pct": pct,
        "verdict": "warn" if free_gb < 12 else "ok",
        "detail": f"剩余 {free_gb}GB (已用 {pct}%)",
    }

    # ffmpeg (视频切片必需)
    ff = shutil.which("ffmpeg")
    if ff:
        rc, out, _ = run_quiet(["ffmpeg", "-version"], timeout=30)
        ver = out.splitlines()[0] if out else "ffmpeg"
        report["ffmpeg"] = ("ok", ver)
    else:
        report["ffmpeg"] = ("missing", "未找到 ffmpeg, 视频切片/骨骼视频渲染会失败")

    # 模型与手势资产 (models/ 被 gitignore, 新机器可能缺失)
    report["model_present"] = (MODEL_DIR / "config.json").is_file()
    report["gesture_present"] = GESTURE_TASK.is_file()

    # systemd 用户实例可用性 (Linux)
    report["systemd"] = "unavailable"
    if not IS_WINDOWS and shutil.which("systemctl"):
        rc, out, _ = run_quiet(["systemctl", "--user", "is-system-running"],
                               timeout=30)
        if rc in (0, 1) and out.strip() in ("running", "degraded", "starting"):
            report["systemd"] = out.strip()

    # 手部骨骼判定 (venv 安装后由 phase_setup_venv 回填 import 结果)
    report["hand_imports"] = None

    # 依赖 wheel 兼容性 (mediapipe 等平台敏感锁定)
    report["python_dep_compat"] = _python_dep_compat()

    # 本地 VLM 综合判定 = GPU 能力 + 平台 + 系统内存 (部署阶段按此联动,
    # 避免 Windows/无 GPU 机器仍提示下载 9.9GB 模型)
    if IS_WINDOWS:
        vlm = {"verdict": "skip", "detail": "Windows 不支持 vLLM"}
    elif gpu["verdict"] == "ok":
        if report.get("ram_gb", 0) and report["ram_gb"] < 16:
            vlm = {"verdict": "ask",
                   "detail": "显存充足但系统内存 <16GB, vLLM 可能 OOM"}
        else:
            vlm = {"verdict": "ok", "detail": ""}
    elif gpu["verdict"] == "ask":
        vlm = {"verdict": "ask", "detail": gpu["detail"]}
    else:
        vlm = {"verdict": "skip",
               "detail": gpu.get("detail") or "无可用 NVIDIA GPU"}
    report["vlm"] = vlm
    return report


def print_check_report(rep: dict) -> None:
    """打印能力体检结论表."""
    say("\n╔══════════════════════════════ 能力体检 ══════════════════════════════╗")
    say(f"  操作系统: {rep['os']}    Python: {rep['python']}")

    gpu = rep["gpu"]
    vlm = rep["vlm"]
    if vlm["verdict"] == "ok":
        say(f"  ✅ GPU: {gpu['name']}  {gpu['vram_mib']}MB  CC {gpu['cc']}  "
            f"driver {gpu['driver']} → 本地 VLM 可运行")
    elif vlm["verdict"] == "ask":
        say(f"  ⚠️  GPU: {gpu['name']}  {gpu['vram_mib']}MB  CC {gpu['cc']} → "
            f"{vlm['detail'] or gpu['detail']}")
    else:
        say(f"  ❌ 本地 VLM 不可用: {vlm['detail'] or '无可用 NVIDIA GPU'}"
            f" → AI 标注请切 API 模式")

    ram_v, ram_d = rep["ram"]
    say(f"  {'✅' if ram_v == 'ok' else '⚠️' if ram_v == 'warn' else '❓'} 内存: {ram_d}")

    disk = rep["disk"]
    disk_extra = ""
    if disk["verdict"] == "warn" and vlm["verdict"] != "skip":
        disk_extra = " (下载 9.9GB 模型可能失败)"
    say(f"  {'⚠️' if disk['verdict'] == 'warn' else '✅'} 磁盘: {disk['detail']}"
        + disk_extra)

    ff_v, ff_d = rep["ffmpeg"]
    say(f"  {'✅' if ff_v == 'ok' else '❌'} ffmpeg: {ff_d}")

    if not rep["model_present"]:
        if vlm["verdict"] == "skip":
            say(f"  ⓘ  本地模型权重缺失 → 本机不运行本地 VLM, 无需下载"
                f" ({vlm['detail']})")
        else:
            say("  ⚠️  本地模型权重缺失 (models/llm/Qwen3-VL-8B-Instruct-FP8)"
                " → 需 --download-model 或手动下载")
    if not rep["gesture_present"]:
        say("  ⚠️  models/gesture_recognizer.task 缺失 → 部署时自动下载"
            " (约 8.4MB); 下载失败则手势识别降级, 手部 21 关键点不受影响")

    for prob in rep.get("python_dep_compat") or []:
        say(f"  ❌ 依赖兼容: {prob} → 请改用 Python 3.12 运行部署,"
            f" 或调整 requirements.txt")

    if rep["systemd"] == "unavailable":
        say("  ⚠️  systemd 用户实例不可用 → 部署将采用前台监督模式")
    else:
        say(f"  ✅ systemd 用户实例: {rep['systemd']}")

    hand = rep.get("hand_imports")
    if hand is not None:
        ok_mods = [m for m, ok in hand.items() if ok]
        bad_mods = [m for m, ok in hand.items() if not ok]
        if not bad_mods:
            say(f"  ✅ 手部骨骼处理依赖可导入: {', '.join(ok_mods)}")
        else:
            say(f"  ❌ 手部骨骼依赖缺失: {', '.join(bad_mods)}"
                f" (已有: {', '.join(ok_mods) or '无'})")
    say("╚══════════════════════════════════════════════════════════════════════════╝")


# ---------------------------------------------------------------------------
# 阶段 2: .env 生成/补全
# ---------------------------------------------------------------------------

def _read_env_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _parse_env(lines: list[str]) -> dict[str, str]:
    """解析 KEY=VALUE, 忽略注释/空行. 返回 {key: value}."""
    out = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _set_env_value(lines: list[str], key: str, value: str) -> list[str]:
    """原位更新 KEY= 行 (保留注释与顺序); 不存在则追加. 返回是否改动."""
    new_lines = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _, _ = stripped.partition("=")
            if k.strip() == key:
                new_lines.append(f"{key}={value}")
                found = True
                continue
        new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    return new_lines


def phase_env(ip: str) -> dict:
    """生成/补全 .env, 永不覆盖已有值 (只动缺失/空/已知占位符)."""
    changes: dict[str, str] = {}
    created = False

    if not ENV_FILE.is_file():
        example = PROJECT_ROOT / ".env.example"
        if example.is_file():
            shutil.copy2(example, ENV_FILE)
            created = True
        else:
            ENV_FILE.write_text("", encoding="utf-8")

    lines = _read_env_lines(ENV_FILE)
    values = _parse_env(lines)

    def fix(key: str, new_value: str, replace_if=None) -> None:
        """replace_if: 值为这些占位符之一时才替换; None 表示值为空时替换."""
        nonlocal lines
        current = values.get(key, "")
        if replace_if is not None:
            if current in replace_if:
                lines = _set_env_value(lines, key, new_value)
                changes[key] = "(已生成, 不显示)"
                values[key] = new_value
        else:
            if not current:
                lines = _set_env_value(lines, key, new_value)
                changes[key] = new_value
                values[key] = new_value

    fix("STORAGE_BACKEND", "local")
    # 已有本项目实例在跑时, 轮换 key 会让旧实例鉴权失配 (现网中断), 故跳过
    live = instances_running()
    if live:
        warn(f"检测到本项目实例正在运行 ({', '.join(live)}), "
             "为不打断现网, 跳过 API_KEY/WORKER_API_KEY 轮换")
    else:
        fix("API_KEY", secrets.token_urlsafe(32), replace_if=("", "change-me"))
        fix("WORKER_API_KEY", secrets.token_urlsafe(32),
            replace_if=("", "change-me-worker-key"))
    # 缺失或等于示例默认值(过期的 IP/端口)时, 补为当前局域网地址
    fix("PUBLIC_BASE_URL", f"http://{ip}:{DEFAULT_BACKEND_PORT}",
        replace_if=("", "http://127.0.0.1:8000"))
    if values.get("STORAGE_BACKEND") == "sftp":
        warn(".env 使用 STORAGE_BACKEND=sftp, 本脚本不配置 SSH 隧道/SFTP, 请自行确保可用")

    if created or changes:
        if not created:  # 覆盖已有文件前备份
            shutil.copy2(ENV_FILE, PROJECT_ROOT /
                         f".env.bak-deploy-{int(time.time())}")
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            os.chmod(ENV_FILE, 0o600)
        except OSError:
            pass
        say(f"  .env {'已从示例生成' if created else '已补全'}: "
            f"{', '.join(changes.keys())}")
    else:
        say("  .env 已完整, 无需改动")
    return {"created": created, "changes": changes}


def env_port() -> int:
    """读取后端端口: 环境变量 PORT 优先, 其次 .env, 默认 8000."""
    v = os.getenv("PORT")
    if v:
        try:
            return int(v)
        except ValueError:
            pass
    if ENV_FILE.is_file():
        v = _parse_env(_read_env_lines(ENV_FILE)).get("PORT", "")
        try:
            return int(v)
        except ValueError:
            pass
    return DEFAULT_BACKEND_PORT


# ---------------------------------------------------------------------------
# 阶段 3: venv 安装
# ---------------------------------------------------------------------------

HAND_MODULES = ["mediapipe", "ultralytics", "rtmlib", "onnxruntime", "watchfiles"]


def phase_setup_venv(rep: dict) -> None:
    """创建/复用应用 venv 并安装依赖 (幂等)."""
    req = PROJECT_ROOT / (f"requirements-{'windows' if IS_WINDOWS else 'linux'}.txt")
    if venv_exists():
        if venv_imports("fastapi") and venv_imports("mediapipe"):
            say(f"  {VENV_NAME} 已就绪 (fastapi/mediapipe 可导入), 跳过安装")
            rep["hand_imports"] = {m: venv_imports(m) for m in HAND_MODULES}
            return
        say(f"  {VENV_NAME} 存在但不完整, 补装依赖...")
    else:
        if VENV_DIR.exists():
            warn(f"{VENV_NAME} 目录存在但缺 python.exe (上次创建失败残留),"
                 f" 清理后重建")
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        say(f"  创建虚拟环境 {VENV_NAME} ...")
        # check=True: venv 创建失败 (如文件被占用/杀软拦截) 立即停止,
        # 不再带着坏 venv 继续装依赖
        run_cmd([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # 不重定向 stdout: pip 检测到控制台时会显示原生进度条
    # (已下载/总量 + 速度 + 剩余时间), 用户能直观看到是否卡住;
    # --log 另存一份详细日志到 data/tmp/deploy/, 失败时便于排查
    say("  安装依赖 (首次约需数分钟, 下载进度条显示在本窗口)...")
    run_cmd(VENV_PIP + ["install", "--upgrade", "pip"], check=True)
    run_cmd(VENV_PIP + ["install", "-r", str(req),
                        "--log", str(LOG_DIR / f"pip-{'windows' if IS_WINDOWS else 'linux'}.log")],
            check=True)
    say("  依赖安装完成")

    rep["hand_imports"] = {m: venv_imports(m) for m in HAND_MODULES}


# ---------------------------------------------------------------------------
# 阶段 3.5: 模型资产补全
# ---------------------------------------------------------------------------

def phase_assets(rep: dict) -> None:
    """补全轻量模型资产 (gesture_recognizer.task, 约 8.4MB).

    与 9.9GB 的 Qwen 权重不同, 手势模型小而必需 (手势识别功能依赖),
    缺失时默认自动下载; 下载失败只降级不阻塞部署。
    """
    if GESTURE_TASK.is_file():
        return
    say("  gesture_recognizer.task 缺失, 自动下载 (约 8.4MB)...")
    try:
        import urllib.request

        socket.setdefaulttimeout(60)
        models = GESTURE_TASK.parent
        models.mkdir(parents=True, exist_ok=True)
        tmp = GESTURE_TASK.with_name(GESTURE_TASK.name + ".download")
        urllib.request.urlretrieve(GESTURE_TASK_URL, str(tmp))
        if tmp.stat().st_size > 100_000:   # 有效模型远大于 100KB
            tmp.replace(GESTURE_TASK)
            rep["gesture_present"] = True
            say("  ✅ gesture_recognizer.task 下载完成 (手势识别可用)")
        else:
            tmp.unlink(missing_ok=True)
            warn("  下载内容过小已丢弃; 手势识别将降级")
    except Exception as e:
        warn(f"  下载失败 ({e}); 手势识别将降级, 可手动放置模型到 "
             f"{GESTURE_TASK.parent}")


# ---------------------------------------------------------------------------
# 阶段 4: vLLM
# ---------------------------------------------------------------------------

def phase_vllm(rep: dict, args) -> bool:
    """vLLM venv/模型准备. 返回是否启用 vLLM 服务.

    消费体检的 vlm 综合判定 (GPU + 平台 + 内存): skip 时不下载模型、
    不建 .venv-llm, 与体检结论联动。
    """
    vlm = rep.get("vlm") or {"verdict": "skip", "detail": ""}
    if vlm["verdict"] == "skip":
        if args.download_model:
            warn(f"本机不运行本地 VLM ({vlm['detail']}),"
                 f" --download-model 无意义, 不下载模型")
        say(f"  本地 VLM 跳过: {vlm['detail']} → AI 标注请切 API 模式")
        return False
    if args.skip_vllm:
        say("  --skip-vllm → 跳过本地 VLM")
        return False
    if vlm["verdict"] == "ask":
        ans = ask("  ⚠️  vLLM 运行条件偏紧, 是否仍启用 (将降配运行)? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            say("  已跳过 vLLM")
            return False

    # 1) .venv-llm
    vllm_bin = VENV_LLM_BIN / ("vllm.exe" if IS_WINDOWS else "vllm")
    if not vllm_bin.is_file():
        say("  创建 .venv-llm 并安装 vLLM (约 8-12GB, 请耐心等待)...")
        run_cmd([sys.executable, "-m", "venv", str(VENV_LLM_DIR)], check=True)
        llm_pip = [str(VENV_LLM_BIN / ("python.exe" if IS_WINDOWS else "python")),
                   "-m", "pip"]
        run_cmd(llm_pip + ["install", "--upgrade", "pip"], check=True)
        run_cmd(llm_pip + ["install", "-r",
                           str(PROJECT_ROOT / "requirements-vllm.txt")],
                check=True)
    rc, out, _ = run_quiet([str(vllm_bin), "--version"], timeout=120)
    if rc != 0:
        fail("vllm 安装校验失败, 跳过本地 VLM")
        return False
    say(f"  vLLM 就绪: {out.strip().splitlines()[0] if out.strip() else 'ok'}")

    # 2) 模型权重
    if not (MODEL_DIR / "config.json").is_file():
        dl_cmd = (f'huggingface-cli download {MODEL_REPO} '
                  f'--local-dir "{MODEL_DIR}"')
        if args.download_model:
            du = shutil.disk_usage(PROJECT_ROOT)
            if du.free < 12 * 1024 ** 3:
                fail(f"磁盘剩余 {du.free / 1024**3:.1f}GB < 12GB, 无法下载模型")
                return False
            say(f"  下载模型权重 (9.9GB, 支持断点续传): {MODEL_REPO}")
            env = dict(os.environ)
            if args.hf_mirror:
                env["HF_ENDPOINT"] = "https://hf-mirror.com"
            hf_cli = VENV_LLM_BIN / ("huggingface-cli.exe" if IS_WINDOWS
                                     else "huggingface-cli")
            rc, _, err = run_quiet(
                [str(hf_cli), "download", MODEL_REPO, "--local-dir", str(MODEL_DIR)],
                env=env, timeout=7200)
            if rc != 0 or not (MODEL_DIR / "config.json").is_file():
                fail(f"模型下载失败 (可重跑同一命令续传): {err[-300:]}")
                return False
            say("  模型下载完成")
        else:
            warn(f"本地模型权重缺失. 手动下载 (或重跑加 --download-model):\n"
                 f"    {dl_cmd}\n"
                 f"    国内网络可先: export HF_ENDPOINT=https://hf-mirror.com")
            return False
    else:
        say(f"  模型就绪: {MODEL_DIR}")
    return True


# ---------------------------------------------------------------------------
# 阶段 5: 端口避让
# ---------------------------------------------------------------------------

def phase_ports(vllm_enabled: bool, ip: str) -> dict:
    """端口占用检测与自动避让. 返回 {backend_port, vllm_port}."""
    result = {"backend_port": env_port(), "vllm_port": DEFAULT_VLLM_PORT,
              "backend_switched": False, "vllm_switched": False}

    # 后端端口
    want = result["backend_port"]
    if port_free(want):
        say(f"  后端端口 {want} 空闲 ✓")
    elif port_owned_by_us(want, "egodata-backend.service"):
        say(f"  后端端口 {want} 由本项目自身占用 → 保持并复用")
    else:
        new_port = pick_free_port(want, skip={DEFAULT_VLLM_PORT})
        if new_port < 0:
            fail(f"端口 {want} 被占用且向后扫描 {want}+ 无空闲端口")
            raise SystemExit(1)
        say(f"  ⚠️  后端端口 {want} 被其他程序占用 → 改用 {new_port}")
        result["backend_port"] = new_port
        result["backend_switched"] = True

    # vLLM 端口
    if vllm_enabled:
        want = result["vllm_port"]
        if port_free(want):
            say(f"  vLLM 端口 {want} 空闲 ✓")
        elif port_owned_by_us(want, "egodata-vllm.service"):
            say(f"  vLLM 端口 {want} 由本项目自身占用 → 保持并复用")
        else:
            new_port = pick_free_port(want)
            say(f"  ⚠️  vLLM 端口 {want} 被其他程序占用 → 改用 {new_port}")
            result["vllm_port"] = new_port
            result["vllm_switched"] = True

    # 换端口则同步 .env (PORT / PUBLIC_BASE_URL), 保证应用配置与监听一致
    if result["backend_switched"]:
        lines = _read_env_lines(ENV_FILE)
        lines = _set_env_value(lines, "PORT", str(result["backend_port"]))
        lines = _set_env_value(lines, "PUBLIC_BASE_URL",
                               f"http://{ip}:{result['backend_port']}")
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        say("  已同步 .env: PORT / PUBLIC_BASE_URL")
    return result


# ---------------------------------------------------------------------------
# 阶段 6: 手部骨骼冒烟测试
# ---------------------------------------------------------------------------

def phase_smoke_test(rep: dict) -> None:
    """复用 scripts/test_mediapipe_local.py 跑端到端手部骨骼冒烟测试."""
    if rep["ffmpeg"][0] != "ok":
        warn("无 ffmpeg, 跳过冒烟测试")
        return
    hand = rep.get("hand_imports") or {}
    if hand.get("mediapipe") is False:
        fail("venv 中 mediapipe 导入失败, 跳过冒烟测试 (手部骨骼工作流将不可用)")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    smoke = LOG_DIR / "smoke.mp4"
    run_quiet(["ffmpeg", "-y", "-f", "lavfi", "-i",
               "color=c=blue:s=640x480:d=1", "-pix_fmt", "yuv420p", str(smoke)],
              timeout=120)
    if not smoke.is_file():
        warn("测试视频合成失败, 跳过冒烟测试")
        return

    say(f"  运行手部骨骼冒烟测试 ({smoke.name}) ...")
    rc, out, err = run_quiet(
        [str(VENV_PY), "scripts/test_mediapipe_local.py", str(smoke),
         "--device", "cpu", "--max-hands", "2"],
        cwd=PROJECT_ROOT, timeout=600)
    if rc == 0:
        try:
            payload = json.loads(out)
            gesture = payload.get("gesture_model")
            if gesture:
                say(f"  ✅ MediaPipe 手部骨骼流水线端到端通过"
                    f" (gesture_model: {gesture})")
            else:
                warn("流水线通过, 但 gesture_recognizer.task 缺失 → 手势识别降级")
            return
        except json.JSONDecodeError:
            pass
    fail(f"手部骨骼冒烟测试失败: {err[-500:] or out[-500:]}")


# ---------------------------------------------------------------------------
# 阶段 7: workflows API 模式 patch
# ---------------------------------------------------------------------------

def phase_workflows(vllm_enabled: bool, args) -> None:
    """vLLM 不可用时询问是否把工作流 local → api."""
    if vllm_enabled or not WORKFLOWS_JSON.is_file():
        return
    try:
        data = json.loads(WORKFLOWS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, list):
        return

    local_nodes = []
    for wf in data:
        for node in wf.get("graph", {}).get("nodes", []):
            nd = node.get("data", {})
            cfg = nd.get("config", {})
            if nd.get("nodeType") == "ai_annotation" \
                    and cfg.get("vlm_provider", "local") == "local":
                local_nodes.append((wf, node, cfg))

    if not local_nodes:
        return
    say(f"  检测到 {len(local_nodes)} 个 AI 标注节点使用 local 模式,"
        f" 本机 vLLM 不可用")
    ans = ask("  是否批量切换为 api 模式? (api_key 等留空, 之后在 Studio UI 填写) [y/N] ")
    if ans.strip().lower() not in ("y", "yes"):
        warn("未切换: local 模式任务将报 model_unavailable")
        return

    backup = PROJECT_ROOT / f"data/state/workflows.json.bak-deploy-{int(time.time())}"
    shutil.copy2(WORKFLOWS_JSON, backup)
    patched = 0
    for _wf, node, cfg in local_nodes:
        cfg["vlm_provider"] = "api"
        node["data"]["config"] = cfg
        patched += 1
    WORKFLOWS_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    say(f"  已 patch {patched} 个节点 (备份: {backup.name}),"
        f" 请在 Studio 中为各工作流填写 api_vendor/api_model/api_key")


# ---------------------------------------------------------------------------
# 阶段 8: 服务启动
# ---------------------------------------------------------------------------

def _service_env(ports: dict, vllm_enabled: bool) -> dict:
    """统一的子进程环境: 端口 + vLLM URL 联动."""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    env["PORT"] = str(ports["backend_port"])
    env["EGODATA_SERVER_URL"] = f"http://127.0.0.1:{ports['backend_port']}"
    if vllm_enabled:
        env["EGODATA_VLLM_PORT"] = str(ports["vllm_port"])
    env["EGODATA_VLLM_URL"] = (
        f"http://127.0.0.1:{ports['vllm_port']}/v1/chat/completions"
        if vllm_enabled else
        f"http://127.0.0.1:{DEFAULT_VLLM_PORT}/v1/chat/completions")
    return env


def _render_unit(template: Path, mapping: dict) -> str:
    text = template.read_text(encoding="utf-8")
    for key, value in mapping.items():
        text = text.replace(key, str(value))
    return text


def _systemd_available() -> bool:
    if IS_WINDOWS or not shutil.which("systemctl"):
        return False
    rc, out, _ = run_quiet(["systemctl", "--user", "is-system-running"],
                           timeout=30)
    return rc in (0, 1) and out.strip() in ("running", "degraded", "starting")


def phase_services_systemd(ports: dict, vllm_enabled: bool) -> None:
    """渲染并安装 systemd 用户级单元 + enable --now (开机自启)."""
    src_dir = PROJECT_ROOT / "scripts" / "systemd"
    dst_dir = Path.home() / ".config" / "systemd" / "user"
    dst_dir.mkdir(parents=True, exist_ok=True)

    vllm_url = f"http://127.0.0.1:{ports['vllm_port']}/v1/chat/completions"
    common = {
        "__PROJECT_ROOT__": str(PROJECT_ROOT),
        "__WANTED_BY__": "default.target",
        "__BACKEND_PORT__": str(ports["backend_port"]),
        "__VLLM_URL__": vllm_url,
        "__VLLM_PORT__": str(ports["vllm_port"]),
    }

    units = ["egodata-backend", "egodata-workers"]
    if vllm_enabled:
        units.append("egodata-vllm")

    for name in units:
        tpl = src_dir / f"{name}.service.in"
        if not tpl.is_file():
            fail(f"模板缺失: {tpl}")
            raise SystemExit(1)
        dst = dst_dir / f"{name}.service"
        dst.write_text(_render_unit(tpl, common), encoding="utf-8")
        say(f"  已渲染 {dst}")

    run_cmd(["systemctl", "--user", "daemon-reload"], check=True)

    # 端口被"手动启动的本项目实例"占用时, 只 enable 不 start:
    # 强行 start 会 bind 失败 → Restart 循环; 也不该抢手动实例的端口
    busy = set()
    if not port_free(ports["backend_port"]) \
            and not port_owned_by_us(ports["backend_port"], "egodata-backend.service"):
        busy.add("egodata-backend")
    if vllm_enabled and not port_free(ports["vllm_port"]) \
            and not port_owned_by_us(ports["vllm_port"], "egodata-vllm.service"):
        busy.add("egodata-vllm")

    for name in units:
        if name in busy:
            run_cmd(["systemctl", "--user", "enable", f"{name}.service"],
                    check=True)
            warn(f"{name}: 端口被手动启动的实例占用, 仅 enable 未 start; "
                 f"停掉手动实例后执行: systemctl --user start {name}.service")
        else:
            run_cmd(["systemctl", "--user", "enable", "--now", f"{name}.service"],
                    check=True)
    # 无登录开机自启
    rc, _, _ = run_quiet(["loginctl", "enable-linger", getpass.getuser()],
                         timeout=60)
    if rc != 0:
        warn(f"loginctl enable-linger 失败, 开机自启需要登录后生效; "
             f"或手动执行: sudo loginctl enable-linger {getpass.getuser()}")


def phase_services_foreground(ports: dict, vllm_enabled: bool,
                              rep: dict) -> None:
    """前台监督模式: 拉起全部子进程, Ctrl+C 全杀."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    live = instances_running()
    if live:
        warn(f"检测到本项目实例已在运行 ({', '.join(live)}), "
             "再启动一套会互相干扰 (建议先停掉旧实例, 或换端口)")
    env = _service_env(ports, vllm_enabled)
    procs: list[subprocess.Popen] = []

    def spawn(cmd, log_name, cwd=None, extra_env=None):
        e = dict(env)
        if extra_env:
            e.update(extra_env)
        log = open(LOG_DIR / log_name, "a", encoding="utf-8")
        log.write(f"\n── deploy 启动 {time.strftime('%F %T')} ──\n")
        log.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                             env=e, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        say(f"  已启动 {log_name.split('.')[0]} (pid={p.pid})")
        return p

    spawn([VENV_PY, "-m", "uvicorn", "app.main:app",
           "--host", "0.0.0.0", "--port", str(ports["backend_port"])],
          "backend.log")
    spawn([VENV_PY, "scripts/hot_reload_worker.py"], "workers.log")
    if vllm_enabled and not IS_WINDOWS:
        vllm = VENV_LLM_BIN / "vllm"
        spawn([vllm, "serve", str(MODEL_DIR),
               "--host", "127.0.0.1", "--port", str(ports["vllm_port"]),
               "--gpu-memory-utilization", "0.70", "--max-model-len", "16384"],
              "vllm.log", cwd="/tmp",
              extra_env={
                  "VLLM_USE_DEEP_GEMM": "0",   # RTX 5090 / SM120 FP8 必需
                  "VLLM_USE_FLASHINFER_SAMPLER": "0",  # 项目路径含空格, 跳过 JIT
                  "PATH": f"{VENV_LLM_BIN}{os.pathsep}{os.environ.get('PATH', '')}",
              })

    # 等待后端就绪 (健康检查)
    say("  等待后端就绪 ...")
    deadline = time.time() + 60
    up = False
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", ports["backend_port"]),
                                          timeout=1):
                up = True
                break
        except OSError:
            time.sleep(1)
    if up:
        say(f"  ✅ 后端已就绪: http://127.0.0.1:{ports['backend_port']}")
    else:
        warn("后端 60s 内未就绪, 请查看日志 data/tmp/deploy/backend.log")

    def _raise_kb(signum, frame):
        raise KeyboardInterrupt()

    say("  服务运行中 (前台监督), Ctrl+C 停止全部 ...")
    prev_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_kb)  # kill 时也走统一清理路径
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        say("  正在停止全部服务 ...")
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
        deadline = time.time() + 8
        while time.time() < deadline and any(p.poll() is None for p in procs):
            time.sleep(0.5)
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass
        say("  已全部停止")


def _write_windows_startup(ports: dict, vllm_enabled: bool) -> None:
    """写入启动文件夹 .bat (下次登录自启; 直调 python.exe 规避 PowerShell 策略)."""
    startup = Path(os.environ.get("APPDATA", str(Path.home()))) / "Microsoft" \
        / "Windows" / "Start Menu" / "Programs" / "Startup" / "egodata-start.bat"
    root = str(PROJECT_ROOT)
    py = str(VENV_PY)
    server = f"http://127.0.0.1:{ports['backend_port']}"
    content = (
        "@echo off\n"
        f'set PYTHONPATH={root}\n'
        f'set PORT={ports["backend_port"]}\n'
        f'set EGODATA_SERVER_URL={server}\n'
        f'set EGODATA_VLLM_URL=http://127.0.0.1:{ports["vllm_port"]}/v1/chat/completions\n'
        f'start "egodata-backend" cmd /c ""{py}" "{root}\\scripts\\hot_reload_backend.py"'
        f' >> "{root}\\data\\tmp\\deploy\\backend-hot.log" 2>&1"\n'
        f'start "egodata-workers" cmd /c ""{py}" "{root}\\scripts\\hot_reload_worker.py"'
        f' >> "{root}\\data\\tmp\\deploy\\workers-hot.log" 2>&1"\n'
    )
    startup.parent.mkdir(parents=True, exist_ok=True)
    startup.write_text(content, encoding="utf-8")
    say(f"  已写入开机自启: {startup}")


def phase_services(rep: dict, ports: dict, vllm_enabled: bool, args) -> None:
    """按 OS/参数选择启动模式."""
    if args.no_services:
        phase_services_foreground(ports, vllm_enabled, rep)
        return
    if IS_WINDOWS:
        _write_windows_startup(ports, vllm_enabled)
        say("  Windows: 开机自启已装好 (下次登录生效), 现在前台启动 ...")
        phase_services_foreground(ports, vllm_enabled, rep)
        return
    if _systemd_available():
        phase_services_systemd(ports, vllm_enabled)
    else:
        warn("systemd 用户实例不可用 → 退回前台监督模式")
        phase_services_foreground(ports, vllm_enabled, rep)


# ---------------------------------------------------------------------------
# 阶段 9: 报告
# ---------------------------------------------------------------------------

def read_versions() -> tuple[str, str]:
    """读取后端 (app/version.py) 与前端 (package.json) 版本号."""
    backend, web = "?", "?"
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app_version", PROJECT_ROOT / "app" / "version.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        backend = m.__version__
    except Exception:
        pass
    pkg = PROJECT_ROOT / "web" / "workflow-studio" / "package.json"
    try:
        web = json.loads(pkg.read_text(encoding="utf-8")).get("version", "?")
    except Exception:
        pass
    return backend, web


def _unit_state(name: str) -> str:
    rc, out, _ = run_quiet(["systemctl", "--user", "is-active", f"{name}.service"],
                           timeout=30)
    return out.strip() if rc == 0 else "inactive"


def phase_report(rep: dict, ports: dict, vllm_enabled: bool, args) -> None:
    ip = lan_ip()
    bp = ports["backend_port"]
    bver, wver = read_versions()
    say("\n╔════════════════════════════ EgoData 部署报告 ════════════════════════════╗")
    say(f"  版本: 后端 v{bver} · 前端 v{wver}")
    say(f"  局域网访问地址:  http://{ip}:{bp}")
    say(f"    登录页:        http://{ip}:{bp}/login")
    say(f"    任务页:        http://{ip}:{bp}/tasks")
    say(f"    后端健康:      http://{ip}:{bp}/health")
    if ports["backend_switched"]:
        say(f"    ⚠️  原端口被其他程序占用, 已自动改用 {bp}")
    if vllm_enabled:
        say(f"  vLLM (本地 VLM): http://127.0.0.1:{ports['vllm_port']}/v1/models"
            + ("   (⚠️ 已避开被占用的 8001)" if ports["vllm_switched"] else ""))

    if not IS_WINDOWS and not args.no_services:
        states = "  ".join(
            f"{n}: {_unit_state(n)}"
            for n in (["egodata-backend", "egodata-workers"]
                      + (["egodata-vllm"] if vllm_enabled else [])))
        say(f"  systemd 服务状态: {states}")
        say("  开机自启: 已启用 (loginctl linger)")

    say("  下一步:")
    say("    - 浏览器打开登录页 (首次空库账号由 EGODATA_BOOTSTRAP_* 配置)")
    if not vllm_enabled:
        say("    - AI 标注: 本机未启用本地 VLM → 请在 Studio 工作流节点选 api 模式")
    else:
        say("    - AI 标注: 本地 vLLM 已就绪; 也可按工作流切换 api 模式")
    if not rep["model_present"] and not args.download_model \
            and rep["vlm"]["verdict"] != "skip":
        say(f"    - 下载模型(可选): huggingface-cli download {MODEL_REPO} "
            f'--local-dir "{MODEL_DIR}"')
    say("╚══════════════════════════════════════════════════════════════════════════╝")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EgoData Data Acquisition 一键部署 (跨平台)")
    p.add_argument("--check-only", action="store_true",
                   help="只做硬件能力体检, 零副作用")
    p.add_argument("--download-model", action="store_true",
                   help="模型权重缺失时自动下载 (9.9GB, 断点续传)")
    p.add_argument("--hf-mirror", action="store_true",
                   help="下载模型走 hf-mirror.com 镜像")
    p.add_argument("--skip-vllm", action="store_true",
                   help="跳过本地 VLM (vLLM)")
    p.add_argument("--no-services", action="store_true",
                   help="不安装 systemd/自启, 前台监督运行")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 非终端输出 (管道/重定向/CI 日志): Python 默认按 locale 编码 stdout,
    # Windows 下是 GBK, emoji (❌ 等) 会 UnicodeEncodeError 直接崩溃;
    # 强制 UTF-8。终端 (isatty) 不动 —— WindowsConsoleIO 写宽字符,
    # 中文/emoji 本就正常。
    if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    say("═" * 62)
    say(f"  EgoData Data Acquisition 一键部署")
    say(f"  项目目录: {PROJECT_ROOT}")
    say(f"  系统: {platform.system()} {platform.release()}  "
        f"Python: {sys.version.split()[0]}")
    say("═" * 62)

    phase("能力体检")
    rep = phase_check()
    print_check_report(rep)
    if args.check_only:
        say("\n  (--check-only: 体检完成, 未做任何改动)")
        # A successful check is a successful command. Returning zero keeps
        # the Linux/Windows wrappers usable from CI and shell scripts.
        return

    ip = lan_ip()

    phase(".env 配置")
    phase_env(ip)

    phase("应用虚拟环境")
    phase_setup_venv(rep)
    print_check_report(rep)  # 回填手部骨骼依赖后重打判定表

    phase("模型资产")
    phase_assets(rep)

    phase("本地 VLM (vLLM)")
    vllm_enabled = phase_vllm(rep, args)

    phase("端口检测与避让")
    ports = phase_ports(vllm_enabled, ip)

    phase("手部骨骼冒烟测试")
    phase_smoke_test(rep)

    phase("AI 标注工作流检查")
    phase_workflows(vllm_enabled, args)

    phase("启动服务")
    phase_services(rep, ports, vllm_enabled, args)

    phase_report(rep, ports, vllm_enabled, args)


if __name__ == "__main__":
    main()
