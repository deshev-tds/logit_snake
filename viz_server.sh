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
       [[ "$command" == *"--static-dir $ROOT_DIR_REAL/viz"* ]] ||
       [[ "$command" == *"$APP_NAME --"* && "$command" == *"--static-dir ./viz"* ]]; then
      echo "$pid"
    fi
  done
}

server_is_ready() {
  local pid="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -a -p "$pid" -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi

  curl --silent --fail --max-time 1 "http://$HOST:$PORT/" >/dev/null 2>&1
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

  PID="$(
    VIZ_PYTHON_BIN="$PYTHON_BIN" \
    VIZ_ROOT_DIR="$ROOT_DIR" \
    VIZ_HOST="$HOST" \
    VIZ_PORT="$PORT" \
    VIZ_DEFAULT_BASE_URL="$DEFAULT_BASE_URL" \
    VIZ_LOG_FILE="$LOG_FILE" \
    "$PYTHON_BIN" -c '
import os
import subprocess

log = open(os.environ["VIZ_LOG_FILE"], "ab", buffering=0)
cmd = [
    os.environ["VIZ_PYTHON_BIN"],
    os.path.join(os.environ["VIZ_ROOT_DIR"], "viz_server.py"),
    "--host",
    os.environ["VIZ_HOST"],
    "--port",
    os.environ["VIZ_PORT"],
    "--static-dir",
    os.path.join(os.environ["VIZ_ROOT_DIR"], "viz"),
    "--default-base-url",
    os.environ["VIZ_DEFAULT_BASE_URL"],
]
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    close_fds=True,
    start_new_session=True,
)
print(proc.pid)
'
  )"
  echo "$PID" > "$PID_FILE"

  for _ in {1..25}; do
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "viz_server failed to start; see $LOG_FILE"
      return 1
    fi
    if server_is_ready "$(cat "$PID_FILE")"; then
      echo "viz_server started on http://$HOST:$PORT (pid $(cat "$PID_FILE")), default base URL: $DEFAULT_BASE_URL"
      return 0
    fi
    sleep 0.2
  done

  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "viz_server did not become ready on http://$HOST:$PORT; see $LOG_FILE"
  return 1
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
