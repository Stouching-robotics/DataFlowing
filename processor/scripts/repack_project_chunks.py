#!/usr/bin/env python3
"""Repack a project into LeRobot v2.1's 1000-episode chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--drop-backup", action="store_true",
                        help="remove the pre-repack tree after verification")
    args = parser.parse_args()

    from app.project_dataset import repack_project_chunks

    result = repack_project_chunks(
        args.project_root,
        keep_backup=not args.drop_backup,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
