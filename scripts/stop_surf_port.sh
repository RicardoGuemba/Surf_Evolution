#!/usr/bin/env bash
# Encerra o processo que está à escuta na porta Gradio (evita ver UI antiga).
set -euo pipefail
PORT="${1:-${GRADIO_SERVER_PORT:-7860}}"
if ! command -v lsof >/dev/null 2>&1; then
  echo "Instale ou use: kill manualmente o PID na porta ${PORT}"
  exit 1
fi
PIDS=$(lsof -ti ":${PORT}" 2>/dev/null || true)
if [[ -z "${PIDS}" ]]; then
  echo "Nada a escutar na porta ${PORT}."
  exit 0
fi
echo "A terminar PID(s) na porta ${PORT}: ${PIDS}"
kill -9 ${PIDS} 2>/dev/null || true
echo "OK."
