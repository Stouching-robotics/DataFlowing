"""Read-only inspection of a remote session directory and its DB mapping."""

from __future__ import annotations

import asyncio
import os
from pathlib import PurePosixPath

import asyncpg
import paramiko


REMOTE_SESSION = os.environ.get("EGODATA_REMOTE_SESSION", "/remote/path/to/session")


def print_tree(sftp: paramiko.SFTPClient, root: str) -> None:
    pending = [root]
    while pending:
        current = pending.pop(0)
        try:
            entries = sorted(sftp.listdir_attr(current), key=lambda item: item.filename)
        except OSError as exc:
            print(f"ERROR {current}: {exc}")
            continue
        for entry in entries:
            path = str(PurePosixPath(current) / entry.filename)
            mode = entry.st_mode or 0
            is_dir = bool(mode & 0o040000)
            if is_dir:
                print(f"DIR  {path}")
                pending.append(path)
            else:
                print(f"FILE {entry.st_size:>12} {path}")


async def print_db_mapping() -> None:
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=15432,
        user="odoo",
        password=os.environ.get("EGODATA_DB_PASSWORD", ""),
        database="data_acq",
    )
    try:
        sessions = await conn.fetch(
            """
            SELECT id::text, name, original_archive, episode_count
            FROM sessions
            WHERE name ILIKE '%episode_000014%'
               OR original_archive ILIKE '%episode_000014%'
            ORDER BY created_at DESC
            """
        )
        print("--- DB sessions ---")
        for row in sessions:
            print(dict(row))

        episodes = await conn.fetch(
            """
            SELECT id::text, session_id::text, name, camera_names, fps, frame_count,
                   status, meta
            FROM episodes
            WHERE name ILIKE '%episode_000014%'
               OR meta::text ILIKE '%episode_000014%'
               OR session_id IN (SELECT id FROM sessions WHERE original_archive ILIKE '%episode_000014%')
            ORDER BY created_at DESC
            """
        )
        print("--- DB episodes ---")
        for row in episodes:
            print(dict(row))
    finally:
        await conn.close()


def main() -> None:
    password = os.environ.get("EGODATA_SSH_PASSWORD")
    if not password:
        raise SystemExit("EGODATA_SSH_PASSWORD is required")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        os.environ.get("EGODATA_SSH_HOST", ""),
        username="Stouch",
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
    )
    try:
        with client.open_sftp() as sftp:
            print("--- Remote session tree ---")
            print_tree(sftp, REMOTE_SESSION)
    finally:
        client.close()

    asyncio.run(print_db_mapping())


if __name__ == "__main__":
    main()
