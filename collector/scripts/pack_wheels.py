#!/usr/bin/env python3
"""生成 Windows 离线部署包（wheels/ 目录），供 start.bat 在无外网环境一键安装。

在任意**有网**的机器上运行（Linux 开发机即可，跨平台拉取 Windows 轮子），
把产出的 wheels/ 整个目录拷到客户项目根目录：
start.bat / start.sh 检测到 wheels\\*.whl 后自动改走纯离线安装。

跨平台解析用 uv（uv pip compile --python-platform windows，能正确按目标平台
评估环境标记，剔除 Linux-only 依赖如 bleak 的 dbus-fast）；无 uv 时回退纯
pip download（此时建议直接在一台 Windows 机器上运行本脚本）。

用法:
    python scripts/pack_wheels.py                 # 仅主程序必需依赖
    python scripts/pack_wheels.py --extras        # + mediapipe / pyrealsense2
    python scripts/pack_wheels.py --torch         # + CPU 版 torch（较大）
    python scripts/pack_wheels.py --out /tmp/wheels   # 自定义输出目录

产物:
    <out>/*.whl                        Windows 3.12 依赖轮子（含依赖闭包）
    <out>/python-3.12.10-amd64.exe     Python 安装包（客户机器无 Python 时用）
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PY_VERSION = "3.12"
PY_INSTALLER = "python-3.12.10-amd64.exe"
PY_INSTALLER_URLS = [
    f"https://mirrors.aliyun.com/python-release/windows/{PY_INSTALLER}",
    f"https://registry.npmmirror.com/-/binary/python/3.12.10/{PY_INSTALLER}",
    f"https://mirrors.huaweicloud.com/python/3.12.10/{PY_INSTALLER}",
]
ALIYUN = "https://mirrors.aliyun.com/pypi/simple/"
TUNA = "https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI = "https://pypi.org/simple"
MIRRORS = [ALIYUN, TUNA, PYPI]
EXTRAS = ["mediapipe", "pyrealsense2"]
TORCH_INDEXES = [
    ("https://mirrors.aliyun.com/pytorch-wheels/cpu/", ALIYUN),
    ("https://download.pytorch.org/whl/cpu", PYPI),
]
UV = shutil.which("uv")


def parse_requirements(path: Path) -> list[str]:
    """requirements.txt → 包规格列表（去注释、去空行）。"""
    specs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            specs.append(line)
    return specs


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _pip_download(lock: Path, out: Path, mirror: str) -> None:
    """按锁定文件抓取 Windows 轮子（--no-deps：锁定已含全部闭包，无需再解析）。"""
    _run([sys.executable, "-m", "pip", "download", "--no-deps",
          "-r", str(lock), "-d", str(out),
          "--platform", "win_amd64", "--python-version", PY_VERSION,
          "--only-binary=:all:", "-i", mirror])


def download_closure(specs: list[str], out: Path,
                     index: str | None = None,
                     extra_index: str | None = None,
                     lock_name: str = "requirements-win.txt") -> Path:
    """uv compile（目标 Windows）→ pip download 抓轮子。无 uv 时直接 pip download。"""
    lock = out / lock_name
    if UV is None:
        # 回退：在 Windows 机器上运行时 pip 会按本机平台评估标记，结果正确
        req = out / ".pack_wheels_req.tmp"
        req.write_text("\n".join(specs), encoding="utf-8")
        _run([sys.executable, "-m", "pip", "download", "-d", str(out),
              "-r", str(req), "--platform", "win_amd64",
              "--python-version", PY_VERSION, "--only-binary=:all:",
              "-i", index or MIRRORS[0]])
        req.unlink()
        return lock

    cmd = [UV, "pip", "compile", "--python-platform", "windows",
           "--python-version", PY_VERSION, "--output-file", str(lock)]
    if index:
        cmd += ["--index-url", index]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    req = out / ".pack_wheels_req.tmp"
    req.write_text("\n".join(specs), encoding="utf-8")
    cmd.append(str(req))
    _run(cmd)
    req.unlink()
    # 按锁抓轮子（锁内索引头由 pip 原样生效；传同一镜像兜底非锁内包的查找）
    _pip_download(lock, out, index or MIRRORS[0])
    return lock


def download_python_installer(out: Path) -> None:
    target = out / PY_INSTALLER
    if target.exists() and target.stat().st_size > 10_000_000:
        print(f"  已存在: {target.name}，跳过下载")
        return
    last_err: Exception | None = None
    for url in PY_INSTALLER_URLS:
        try:
            print(f"  下载 Python 安装包: {url}")
            with urllib.request.urlopen(url, timeout=120) as r:
                target.write_bytes(r.read())
            print(f"  完成: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as e:  # noqa: BLE001 — 逐源回退
            last_err = e
    raise RuntimeError(f"Python 安装包下载失败: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", action="store_true",
                        help="额外打包 mediapipe / pyrealsense2")
    parser.add_argument("--torch", action="store_true",
                        help="额外打包 CPU 版 torch（体积较大）")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "wheels",
                        help="输出目录（默认仓库根目录 wheels/）")
    parser.add_argument("--no-python-installer", action="store_true",
                        help="跳过 Python 安装包下载")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    specs = parse_requirements(repo_root / "requirements.txt")
    if not specs:
        print("[错误] requirements.txt 为空或不存在", file=sys.stderr)
        return 1
    if args.extras:
        specs = specs + EXTRAS

    print(f"[1/3] 解析 {len(specs)} 个包的 Windows {PY_VERSION} 依赖闭包 -> {out}/ ...")
    if UV is None:
        print("  [提示] 未检测到 uv，回退纯 pip 解析（建议在一台 Windows 机器上运行，"
              "或 pip install uv 后重试）")

    last_err: Exception | None = None
    for mirror in MIRRORS:
        try:
            download_closure(specs, out, index=mirror)
            break
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"  源 {mirror} 失败，尝试下一个 ...")
    else:
        print(f"[错误] 依赖下载失败: {last_err}", file=sys.stderr)
        if UV is None:
            print("  提示: 本机无 uv，且 pip 跨平台解析可能因 Linux-only 依赖失败；"
                  "请 pip install uv 后重试，或直接在一台 Windows 机器上运行本脚本。",
                  file=sys.stderr)
        return 1

    if args.torch:
        print("[2/3] 解析 CPU 版 torch（Windows）...")
        torch_done = False
        for index, extra in TORCH_INDEXES:
            try:
                download_closure(["torch"], out, index=index, extra_index=extra,
                                 lock_name="torch-win.txt")
                torch_done = True
                break
            except subprocess.CalledProcessError as e:
                last_err = e
                print(f"  源 {index} 失败，尝试下一个 ...")
        if not torch_done:
            print(f"[错误] torch 下载失败: {last_err}", file=sys.stderr)
            return 1
        print("[2/3] torch 完成")
    else:
        print("[2/3] 跳过 torch（--torch 可打包 CPU 版）")

    if not args.no_python_installer:
        print("[3/3] 拉取 Python 安装包（客户机器无 Python 时 start.bat 会用）...")
        try:
            download_python_installer(out)
        except RuntimeError as e:
            print(f"[警告] {e}；不影响 wheels 使用，手动放入同名文件即可",
                  file=sys.stderr)
    else:
        print("[3/3] 跳过 Python 安装包（--no-python-installer）")

    whls = sorted(out.glob("*.whl"))
    total = sum(w.stat().st_size for w in whls) / 1e6
    print()
    print("=" * 60)
    print(f"打包完成: {len(whls)} 个 wheel，共 {total:.0f} MB -> {out}/")
    print("交付方式: 把整个 wheels/ 目录拷到客户项目根目录，")
    print("          start.bat / start.sh 会自动改走纯离线安装。")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
