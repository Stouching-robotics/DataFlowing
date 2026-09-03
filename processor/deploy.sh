#!/usr/bin/env bash
# One-click deploy/start entry (Linux). Shared logic lives in deploy.py.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 127
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/deploy.py" "$@"
