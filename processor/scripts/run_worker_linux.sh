#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="${PROJECT_ROOT}/.venv-linux/bin/python"
if [[ ! -x "${PYTHON_PATH}" ]]; then
  echo "Linux environment not found. Run scripts/setup_linux.sh first." >&2
  exit 1
fi

export EGODATA_SERVER_URL="${EGODATA_SERVER_URL:-http://127.0.0.1:8000}"
export EGODATA_WORKER_ID="${EGODATA_WORKER_ID:-linux-$(hostname)}"
export EGODATA_DEVICE="${EGODATA_DEVICE:-auto}"
export EGODATA_WORK_DIR="${EGODATA_WORK_DIR:-${PROJECT_ROOT}/data/tmp/worker}"

: "${EGODATA_WORKER_API_KEY:?Set EGODATA_WORKER_API_KEY before starting the worker}"
cd "${PROJECT_ROOT}"
exec "${PYTHON_PATH}" -m worker
