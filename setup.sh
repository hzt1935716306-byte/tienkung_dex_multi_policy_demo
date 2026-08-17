#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TARGET="${ROOT_DIR}"

if [[ "${1:-}" == "--voice" ]]; then
  INSTALL_TARGET="${ROOT_DIR}[voice]"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: ./setup.sh [--voice]"
  exit 2
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -e "${INSTALL_TARGET}"

if [[ "${1:-}" == "--voice" ]]; then
  echo "[INFO] Voice dependencies installed."
  echo "[INFO] If PyAudio failed to build on Ubuntu, run:"
  echo "       sudo apt update && sudo apt install -y portaudio19-dev"
  echo "       ./setup.sh --voice"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_setup.py"
