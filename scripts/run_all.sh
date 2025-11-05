#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_VENVS=(
  "${PROJECT_ROOT}/venv"
  "${PROJECT_ROOT}/../.venv"
)

if [[ -z "${VENV_PATH:-}" ]]; then
  for candidate in "${DEFAULT_VENVS[@]}"; do
    if [[ -f "${candidate}/bin/activate" ]]; then
      VENV_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[All] Activating virtualenv: ${VENV_PATH}"
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "[All] No virtualenv activated (set VENV_PATH to override)."
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/..:${PYTHONPATH:-}"

AI_PORT="${AI_PORT:-8001}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
AI_HEALTHCHECK_URL="${AI_HEALTHCHECK_URL:-http://127.0.0.1:${AI_PORT}/healthz}"
BACKEND_HEALTHCHECK_URL="${BACKEND_HEALTHCHECK_URL:-http://127.0.0.1:${BACKEND_PORT}/healthz}"
WAIT_RETRIES="${WAIT_RETRIES:-30}"
WAIT_DELAY="${WAIT_DELAY:-1}"

MODEL_PATH="${MODEL_PATH:-models/license_plate/best.pt}"
DEFAULT_MODEL_BASE="${PROJECT_ROOT}"
if [[ ! -f "${DEFAULT_MODEL_BASE}/${MODEL_PATH}" ]]; then
  ALT_MODEL_BASE="$(cd "${PROJECT_ROOT}/.." && pwd)"
  if [[ -f "${ALT_MODEL_BASE}/${MODEL_PATH}" ]]; then
    DEFAULT_MODEL_BASE="${ALT_MODEL_BASE}"
  fi
fi
export MODEL_BASE_DIR="${MODEL_BASE_DIR:-${DEFAULT_MODEL_BASE}}"
export MODEL_PATH
export TESSDATA_DIR="${TESSDATA_DIR:-tessdata}"

cd "${PROJECT_ROOT}"

ensure_port_free() {
  local port="$1"
  if lsof -ti ":${port}" >/dev/null 2>&1; then
    echo "[All] Port ${port} is already in use. Stop the existing process or configure a different port." >&2
    exit 1
  fi
}

ensure_port_free "${AI_PORT}"
ensure_port_free "${BACKEND_PORT}"

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi

  if [[ -n "${AI_PID:-}" ]]; then
    kill "${AI_PID}" 2>/dev/null || true
    wait "${AI_PID}" 2>/dev/null || true
  fi

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

wait_for_health() {
  local url="$1"
  local retries="$2"
  local delay="$3"
  local pid="${4:-}"

  for ((i = 1; i <= retries; i++)); do
    if curl --silent --fail "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 2
    fi
    sleep "${delay}"
  done

  return 1
}

start_service() {
  local name="$1"
  shift
  echo "[All] Starting ${name}: $*" >&2
  "$@" &
  local pid=$!
  echo "${pid}"
}

AI_PID=$(start_service "AI service on port ${AI_PORT}" uvicorn ai.app.main:app --host 0.0.0.0 --port "${AI_PORT}")

echo "[All] Waiting for AI service health at ${AI_HEALTHCHECK_URL}"
health_status=0
wait_for_health "${AI_HEALTHCHECK_URL}" "${WAIT_RETRIES}" "${WAIT_DELAY}" "${AI_PID}" || health_status=$?
if [[ ${health_status} -ne 0 ]]; then
  if [[ ${health_status} -eq 2 ]]; then
    wait "${AI_PID}" 2>/dev/null || true
  fi
  echo "[All] AI service did not become ready in time" >&2
  exit 1
fi

BACKEND_PID=$(start_service "backend proxy on port ${BACKEND_PORT} -> http://127.0.0.1:${AI_PORT}" uvicorn backend.app:app --host 0.0.0.0 --port "${BACKEND_PORT}")

echo "[All] Waiting for backend health at ${BACKEND_HEALTHCHECK_URL}"
health_status=0
wait_for_health "${BACKEND_HEALTHCHECK_URL}" "${WAIT_RETRIES}" "${WAIT_DELAY}" "${BACKEND_PID}" || health_status=$?
if [[ ${health_status} -ne 0 ]]; then
  if [[ ${health_status} -eq 2 ]]; then
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
  echo "[All] Backend proxy did not become ready in time" >&2
  exit 1
fi

echo "[All] Both services are running."
echo "  AI service:      http://127.0.0.1:${AI_PORT}"
echo "  Backend proxy:   http://127.0.0.1:${BACKEND_PORT}"
echo
echo "[All] Press Ctrl+C to stop both services."

wait -n "${AI_PID}" "${BACKEND_PID}"
