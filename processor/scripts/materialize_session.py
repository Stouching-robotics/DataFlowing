"""Materialize one server-backed session into the local cache (read-only remote)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import async_session
from app.storage import find_session_dir_async


async def main() -> None:
    async with async_session() as db:
        session_dir = await find_session_dir_async("103f5380-56d5-4341-9ba5-b367e68a2a2e", db)
    print(f"session_dir={session_dir}")
    if session_dir:
        for path in sorted(session_dir.rglob("*")):
            if path.is_file():
                print(f"{path} {path.stat().st_size}")
        for parquet in sorted(session_dir.rglob("*.parquet")):
            frame = pd.read_parquet(parquet)
            print(f"PARQUET {parquet.name}: rows={len(frame)} columns={list(frame.columns)}")


if __name__ == "__main__":
    asyncio.run(main())
