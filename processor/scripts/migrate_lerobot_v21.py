#!/usr/bin/env python3
"""Migrate all valid session projects to the project-level LeRobot v2.1 layout.

The command is deliberately backup-preserving.  A legacy project is rebuilt
under ``data/meta/videos`` and the old project tree is moved below
``<storage-root>/.backups/session-migrations``.  Existing canonical projects
with ``meta/depth`` PNG sidecars are converted to lossless HEVC depth videos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.project_dataset import (
    is_project_dataset,
    migrate_project_dataset,
    migrate_project_depth_pngs,
    verify_project_dataset,
)


def migrate(root: Path, project_name: str | None = None) -> list[dict]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Session root does not exist: {root}")
    projects = [root / project_name] if project_name else sorted(
        path for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    results = []
    for project in projects:
        if not project.is_dir():
            results.append({"project": project.name, "skipped": "missing"})
            continue
        try:
            if is_project_dataset(project):
                result = migrate_project_depth_pngs(project, keep_backup=True)
                if not result.get("changed"):
                    result["verification"] = verify_project_dataset(project)
            else:
                result = migrate_project_dataset(project, keep_backup=True)
            results.append(result)
        except Exception as exc:  # continue to report other projects
            results.append({"project": project.name, "error": str(exc)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions_root", type=Path)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()
    print(json.dumps(migrate(args.sessions_root, args.project),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
