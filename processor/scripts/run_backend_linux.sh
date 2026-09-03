#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="${PROJECT_ROOT}/.venv-linux/bin/python"
if [[ ! -x "${PYTHON_PATH}" ]]; then
  echo "Linux environment not found. Run scripts/setup_linux.sh first." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
reload_args=()
case "${EGODATA_RELOAD:-1}" in
  0|false|False|no|off) ;;
  *)
    # Watch code/web only; data, videos, logs and SFTP cache do
    # not cause a pointless backend restart.
    reload_args=(
      --reload
      --reload-delay 0.5
      --reload-dir "${PROJECT_ROOT}/app"
      --reload-dir "${PROJECT_ROOT}/web/templates"
      --reload-dir "${PROJECT_ROOT}/web/static"
      --reload-dir "${PROJECT_ROOT}/web/workflow-studio/src"
    )
    ;;
esac
cd "${PROJECT_ROOT}"
exec "${PYTHON_PATH}" -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" "${reload_args[@]}"
