#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -L "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
ROOT_DIR_REAL="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PID_FILE="$ROOT_DIR/.viz_server.pid"
LOG_FILE="$ROOT_DIR/viz_server.log"
PORT="${PORT:-18765}"
HOST="${HOST:-127.0.0.1}"
DEFAULT_BASE_URL="${DEFAULT_BASE_URL:-http://127.0.0.1:8080}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
APP_NAME="viz_server.py"

find_app_pids() {
  local pid command
  ps -axo pid=,command= | while read -r pid command; do
    [[ -n "$pid" ]] || continue
    [[ "$pid" != "$$" ]] || continue
    [[ "$command" == *"$APP_NAME"* ]] || continue
    if [[ "$command" == *"$ROOT_DIR/$APP_NAME"* ]] ||
       [[ "$command" == *"$ROOT_DIR_REAL/$APP_NAME"* ]] ||
       [[ "$command" == *"--static-dir $ROOT_DIR/viz"* ]] ||
       [[ "$command" == *"--static-dir $ROOT_DIR_REAL/viz"* ]]; then
      echo "$pid"
    fi
  done
}

stop_pids() {
  local pids=("$@")
  local pid
  [[ "${#pids[@]}" -gt 0 ]] || return 0

  kill "${pids[@]}" 2>/dev/null || true
  sleep 0.5
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

cleanup_old_instances() {
  local pids=()
  local pid
  while IFS= read -r pid; do
    pids+=("$pid")
  done < <(find_app_pids)

  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "stopping old viz_server instance(s): ${pids[*]}"
    stop_pids "${pids[@]}"
  fi

  rm -f "$PID_FILE"
}

start_server() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    cleanup_old_instances
  else
    rm -f "$PID_FILE"
  fi

  cleanup_old_instances

  nohup "$PYTHON_BIN" "$ROOT_DIR/viz_server.py" \
    --host "$HOST" \
    --port "$PORT" \
    --static-dir "$ROOT_DIR/viz" \
    --default-base-url "$DEFAULT_BASE_URL" \
    >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  sleep 0.2
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "viz_server failed to start; see $LOG_FILE"
    return 1
  fi

  echo "viz_server started on http://$HOST:$PORT (pid $(cat "$PID_FILE")), default base URL: $DEFAULT_BASE_URL"
}

stop_server() {
  local pids=()
  local pid

  while IFS= read -r pid; do
    pids+=("$pid")
  done < <(find_app_pids)

  if [[ "${#pids[@]}" -gt 0 ]]; then
    stop_pids "${pids[@]}"
    echo "viz_server stopped (pid(s) ${pids[*]})"
    rm -f "$PID_FILE"
    return 0
  fi

  if [[ ! -f "$PID_FILE" ]]; then
    echo "viz_server is not running"
    return 0
  fi
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "viz_server stopped (pid $PID)"
  else
    echo "viz_server pid file found but process is not running"
  fi
  rm -f "$PID_FILE"
}

status_server() {
  local pids=()
  local pid

  while IFS= read -r pid; do
    pids+=("$pid")
  done < <(find_app_pids)

  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "viz_server running (pid(s) ${pids[*]}) on http://$HOST:$PORT"
    exit 0
  fi

  rm -f "$PID_FILE"
  echo "viz_server is not running"
  exit 1
}

case "${1:-}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    status_server
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
