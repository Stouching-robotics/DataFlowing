#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv-linux"
PYTHON_PATH="${VENV_PATH}/bin/python"

if [[ ! -x "${PYTHON_PATH}" ]]; then
  echo "Creating Linux virtual environment: ${VENV_PATH}"
  python3 -m venv "${VENV_PATH}"
fi

"${PYTHON_PATH}" -m pip install --upgrade pip
"${PYTHON_PATH}" -m pip install -r "${PROJECT_ROOT}/requirements-linux.txt"
echo "Linux environment ready: ${PYTHON_PATH}"
