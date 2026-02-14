#!/usr/bin/env python3
import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

CLIENTS = set()
CLIENTS_LOCK = threading.Lock()
PROC_LOCK = threading.Lock()
TELEMETRY_PROC = None
TELEMETRY_THREAD = None
DEFAULT_TELEMETRY = {}


def broadcast(payload):
    with CLIENTS_LOCK:
        targets = list(CLIENTS)
    for q in targets:
        q.put(payload)


class TelemetryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = queue.Queue()
            with CLIENTS_LOCK:
                CLIENTS.add(q)

            try:
                self.wfile.write(b":ok\n\n")
                self.wfile.flush()
                while True:
                    data = q.get()
                    if data is None:
                        break
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with CLIENTS_LOCK:
                    CLIENTS.discard(q)
            return

        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            running = False
            with PROC_LOCK:
                if TELEMETRY_PROC and TELEMETRY_PROC.poll() is None:
                    running = True
            payload = {"running": running}
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/start", "/api/stop"):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            body = {}

        if path == "/api/stop":
            stop_telemetry()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopped"}).encode("utf-8"))
            return

        prompt = (body or {}).get("prompt", "")
        base_url = (body or {}).get("base_url", DEFAULT_TELEMETRY.get("base_url"))
        if not prompt or not base_url:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "prompt and base_url are required"}).encode("utf-8"))
            return

        try:
            start_telemetry(prompt, base_url)
        except RuntimeError as err:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(err)}).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "started"}).encode("utf-8"))


def read_lines(stream, follow=False):
    if not follow:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            broadcast(line.encode("utf-8", errors="replace"))
        return

    while True:
        line = stream.readline()
        if not line:
            time.sleep(0.05)
            continue
        line = line.strip()
        if not line:
            continue
        broadcast(line.encode("utf-8", errors="replace"))


def read_process_output(proc):
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            broadcast(line.encode("utf-8", errors="replace"))
    finally:
        proc.stdout.close()


def stop_telemetry():
    global TELEMETRY_PROC
    with PROC_LOCK:
        proc = TELEMETRY_PROC
        TELEMETRY_PROC = None
    if not proc:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def start_telemetry(prompt, base_url):
    global TELEMETRY_PROC, TELEMETRY_THREAD
    with PROC_LOCK:
        if TELEMETRY_PROC and TELEMETRY_PROC.poll() is None:
            stop_telemetry()
        if not os.path.isfile(DEFAULT_TELEMETRY["script"]):
            raise RuntimeError(f"telemetry script not found: {DEFAULT_TELEMETRY['script']}")
        cmd = [
            sys.executable,
            DEFAULT_TELEMETRY["script"],
            "--base-url",
            base_url,
            "--prompt",
            prompt,
            "--emit-xyz",
            "--sample-tokens",
            str(DEFAULT_TELEMETRY["sample_tokens"]),
        ]
        if DEFAULT_TELEMETRY.get("n_predict") is not None:
            cmd.extend(["--n-predict", str(DEFAULT_TELEMETRY["n_predict"])])
        if DEFAULT_TELEMETRY.get("topk") is not None:
            cmd.extend(["--topk", str(DEFAULT_TELEMETRY["topk"])])
        if DEFAULT_TELEMETRY.get("window_size") is not None:
            cmd.extend(["--window-size", str(DEFAULT_TELEMETRY["window_size"])])
        if DEFAULT_TELEMETRY.get("no_embeddings"):
            cmd.append("--no-embeddings")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        TELEMETRY_PROC = proc
        TELEMETRY_THREAD = threading.Thread(target=read_process_output, args=(proc,), daemon=True)
        TELEMETRY_THREAD.start()


def stdin_reader(args):
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            read_lines(f, follow=args.follow)
    else:
        read_lines(sys.stdin, follow=args.follow)


def main():
    parser = argparse.ArgumentParser(description="Serve the 3D telemetry visualizer and stream JSONL via SSE")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    parser.add_argument("--static-dir", default="viz", help="directory with index.html/app.js")
    parser.add_argument("--input-file", help="JSONL input file (defaults to stdin)")
    parser.add_argument("--follow", action="store_true", help="follow input file/stdin for new lines")
    parser.add_argument("--telemetry-script", default="llama_telemetry.py", help="path to telemetry wrapper script")
    parser.add_argument("--telemetry-topk", type=int, default=20, help="top-k for telemetry wrapper")
    parser.add_argument("--telemetry-sample-tokens", type=int, default=8, help="sample tokens for telemetry wrapper")
    parser.add_argument("--telemetry-window-size", type=int, default=64, help="window size for telemetry wrapper drift")
    parser.add_argument("--telemetry-n-predict", type=int, default=256, help="n_predict for telemetry wrapper")
    parser.add_argument("--telemetry-no-embeddings", action="store_true", help="disable embeddings for telemetry wrapper")
    parser.add_argument("--telemetry-base-url", default="http://192.168.1.117:1234", help="default llama.cpp base URL")
    args = parser.parse_args()

    static_dir = os.path.abspath(args.static_dir)
    if not os.path.isdir(static_dir):
        raise SystemExit(f"Static dir not found: {static_dir}")

    DEFAULT_TELEMETRY.update({
        "script": os.path.abspath(args.telemetry_script),
        "topk": args.telemetry_topk,
        "sample_tokens": args.telemetry_sample_tokens,
        "window_size": args.telemetry_window_size,
        "n_predict": args.telemetry_n_predict,
        "no_embeddings": args.telemetry_no_embeddings,
        "base_url": args.telemetry_base_url,
    })

    if args.input_file or not sys.stdin.isatty():
        reader = threading.Thread(target=stdin_reader, args=(args,), daemon=True)
        reader.start()

    handler = lambda *h_args, **h_kwargs: TelemetryHandler(*h_args, directory=static_dir, **h_kwargs)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    sys.stderr.write(f"Serving on http://{args.host}:{args.port}\n")
    sys.stderr.write("Stream endpoint: /stream\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
