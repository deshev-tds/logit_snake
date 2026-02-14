#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/.viz_server.pid"
LOG_FILE="$ROOT_DIR/viz_server.log"
PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"

start_server() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "viz_server already running (pid $(cat "$PID_FILE"))"
    return 0
  fi
  nohup python3 "$ROOT_DIR/viz_server.py" --host "$HOST" --port "$PORT" --static-dir "$ROOT_DIR/viz" >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "viz_server started on http://$HOST:$PORT (pid $(cat "$PID_FILE"))"
}

stop_server() {
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
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "viz_server running (pid $(cat "$PID_FILE"))"
    exit 0
  fi
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
