"""Read-only SSH inspection of the legacy server service layout."""

from __future__ import annotations

import os

import paramiko


def main() -> None:
    password = os.environ.get("EGODATA_SSH_PASSWORD")
    if not password:
        raise SystemExit("EGODATA_SSH_PASSWORD is required")

    commands = [
        "hostname",
        "docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}' 2>/dev/null || true",
        "docker ps -a --format '{{.Names}}|{{.Status}}|{{.Image}}' 2>/dev/null || true",
        "ps -ef | grep -E '2586|uvicorn|gunicorn|nginx|caddy|fastapi' | grep -v grep || true",
        "systemctl --no-pager --type=service --state=running 2>/dev/null | grep -E 'nginx|caddy|odoo|data|acq|uvicorn' || true",
        "command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:2586 -sTCP:LISTEN || true",
        "ss -ltnp 2>/dev/null | grep -E ':2586|:5432|:22' || true",
    ]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        os.environ.get("EGODATA_SSH_HOST", ""),
        port=22,
        username="Stouch",
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=10,
        banner_timeout=10,
        auth_timeout=10,
    )
    try:
        for command in commands:
            print(f"--- {command} ---")
            _, stdout, stderr = client.exec_command(command, timeout=15)
            print(stdout.read().decode("utf-8", errors="replace"), end="")
            print(stderr.read().decode("utf-8", errors="replace"), end="")
    finally:
        client.close()


if __name__ == "__main__":
    main()
