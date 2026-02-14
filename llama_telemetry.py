#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA_VERSION = "1.0"

DEFAULT_SCHEMA = {
    "version": SCHEMA_VERSION,
    "record_types": {
        "session_start": {
            "t_ms": {"type": "int", "unit": "ms_since_start"},
            "schema_version": {"type": "string"},
            "config": {"type": "object"},
            "model": {"type": "object"},
            "units": {"type": "object"},
        },
        "telemetry": {
            "t_ms": {"type": "int", "unit": "ms_since_start"},
            "token_index": {"type": "int", "unit": "token_index"},
            "token_id": {"type": "int", "unit": "vocab_id"},
            "token_text": {"type": "string", "unit": "token_text"},
            "position": {
                "x": {"type": "float", "unit": "normalized_0_1"},
                "y": {"type": "float", "unit": "normalized_0_1"},
                "z": {"type": "float", "unit": "normalized_0_1"},
            },
            "metrics": {
                "entropy": {"type": "float", "unit": "nats"},
                "logit_margin": {"type": "float", "unit": "logprob_delta"},
                "topk_mass": {"type": "float", "unit": "probability"},
                "drift": {"type": "float", "unit": "l2_per_token"},
            },
            "sample_reason": {"type": "string", "unit": "enum"},
            "event": {"type": "string", "unit": "enum"},
        },
        "text": {
            "t_ms": {"type": "int", "unit": "ms_since_start"},
            "text": {"type": "string", "unit": "utf8"},
            "token_index_start": {"type": "int", "unit": "token_index"},
            "token_index_end": {"type": "int", "unit": "token_index"},
        },
    },
}

UNITS = {
    "entropy": "nats",
    "logit_margin": "logprob_delta",
    "topk_mass": "probability",
    "drift": "l2_per_token",
    "position": "normalized_0_1",
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def build_url(base_url, path):
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def http_post_json(url, payload, api_key=None, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} for {url}: {body}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Failed to reach {url}: {err}") from err
    try:
        return json.loads(body)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Invalid JSON from {url}: {body[:500]}") from err


def http_get_json(url, api_key=None, timeout=30):
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} for {url}: {body}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Failed to reach {url}: {err}") from err
    try:
        return json.loads(body)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Invalid JSON from {url}: {body[:500]}") from err


def load_prompt(args):
    if args.prompt and args.prompt_file:
        raise SystemExit("Specify only one of --prompt or --prompt-file")
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    if args.prompt is None:
        raise SystemExit("Missing --prompt or --prompt-file")
    return args.prompt


def load_extra_params(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_commit_from_build_info(build_info):
    if not build_info:
        return None
    # Expected pattern: b<build>-<hash> or similar
    match = re.search(r"([0-9a-f]{7,40})", build_info)
    if not match:
        return None
    return match.group(1)


def maybe_write_commit_file(path, commit_hash):
    if not commit_hash:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(commit_hash + "\n")
    return True


def extract_probs(response):
    if isinstance(response, list):
        # Multiple prompts; take the first completion
        if not response:
            return None
        response = response[0]

    if "probs" in response:
        return response.get("probs")

    if "completion_probabilities" in response:
        probs = response.get("completion_probabilities")
        if isinstance(probs, list) and probs:
            # Some formats wrap per-completion objects
            first = probs[0]
            if isinstance(first, dict) and "probs" in first:
                return first.get("probs")
            if isinstance(first, list):
                return first
        return probs

    # OpenAI-style fallback
    choices = response.get("choices") if isinstance(response, dict) else None
    if choices:
        logprobs = choices[0].get("logprobs")
        if logprobs and "top_logprobs" in logprobs:
            return logprobs.get("top_logprobs")
    return None


def extract_content(response):
    if isinstance(response, list):
        if not response:
            return ""
        response = response[0]
    return response.get("content", "") if isinstance(response, dict) else ""


def extract_stop_flags(response):
    if isinstance(response, list):
        if not response:
            return {}
        response = response[0]
    if not isinstance(response, dict):
        return {}
    return {
        "stop": bool(response.get("stop")),
        "stopped_eos": response.get("stopped_eos"),
        "stopped_word": response.get("stopped_word"),
        "stopped_limit": response.get("stopped_limit"),
        "stopping_word": response.get("stopping_word"),
    }


def normalize_top_entries(entries):
    normed = []
    for entry in entries:
        if "logprob" in entry and entry["logprob"] is not None:
            logp = float(entry["logprob"])
            prob = math.exp(logp)
        elif "prob" in entry and entry["prob"] is not None:
            prob = float(entry["prob"])
            if prob <= 0.0:
                logp = float("-inf")
            else:
                logp = math.log(prob)
        else:
            continue
        normed.append({
            "id": entry.get("id"),
            "token": entry.get("token", ""),
            "logprob": logp,
            "prob": prob,
        })
    return normed


def compute_metrics(entry, topk_default):
    top_entries = entry.get("top_logprobs") or entry.get("top_probs") or []
    top_entries = normalize_top_entries(top_entries)

    # If top entries empty, try using this token only
    if not top_entries:
        top_entries = normalize_top_entries([{k: entry.get(k) for k in ("id", "token", "logprob", "prob") if k in entry}])

    # Sort by prob descending
    top_entries.sort(key=lambda x: x["prob"], reverse=True)

    topk_entries = top_entries[:topk_default] if top_entries else []
    topk_mass = sum(x["prob"] for x in topk_entries)

    entropy = 0.0
    for item in topk_entries:
        p = item["prob"]
        if p > 0.0:
            entropy -= p * math.log(p)
    p_rest = max(0.0, 1.0 - topk_mass)
    if p_rest > 0.0:
        entropy -= p_rest * math.log(p_rest)

    logit_margin = None
    if len(topk_entries) >= 2:
        logit_margin = topk_entries[0]["logprob"] - topk_entries[1]["logprob"]

    return entropy, logit_margin, topk_mass


def l2_distance(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    acc = 0.0
    for x, y in zip(a, b):
        d = x - y
        acc += d * d
    return math.sqrt(acc)


def normalize_unit(value, max_value):
    if value is None or max_value is None or max_value <= 0.0:
        return None
    if value <= 0.0:
        return 0.0
    return min(1.0, value / max_value)


def with_fallback(value, fallback, default=0.0):
    if value is None:
        return fallback if fallback is not None else default
    return value


def smooth_value(value, last, alpha):
    if last is None or alpha is None or alpha <= 0.0:
        return value
    return last + alpha * (value - last)


def read_commit_file(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip() or None


def emit_json(record):
    sys.stdout.write(json.dumps(record, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="llama.cpp telemetry wrapper (JSONL to stdout)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server base URL")
    parser.add_argument("--endpoint", default="/completion", help="completion endpoint path")
    parser.add_argument("--prompt", help="prompt text")
    parser.add_argument("--prompt-file", help="path to prompt file")
    parser.add_argument("--n-predict", type=int, default=128, help="total tokens to generate (-1 for unlimited)")
    parser.add_argument("--chunk-size", type=int, default=8, help="tokens to request per call")
    parser.add_argument("--topk", type=int, default=20, help="top-K for probability mass")
    parser.add_argument("--sample-tokens", type=int, default=8, help="emit sample every N tokens")
    parser.add_argument("--sample-ms", type=int, default=150, help="emit sample every N ms")
    parser.add_argument("--entropy-spike", type=float, default=0.5, help="entropy delta threshold for event sample")
    parser.add_argument("--margin-drop", type=float, default=1.0, help="logit margin drop threshold for event sample")
    parser.add_argument("--window-size", type=int, default=64, help="token window size for drift embedding")
    parser.add_argument("--embedding-endpoint", default="/embedding", help="embedding endpoint path")
    parser.add_argument("--embd-normalize", type=int, default=-1, help="embedding normalization for /embedding")
    parser.add_argument("--embeddings", dest="embeddings", action="store_true", default=True, help="enable embedding drift")
    parser.add_argument("--no-embeddings", dest="embeddings", action="store_false", help="disable embedding drift")
    parser.add_argument("--emit-xyz", dest="emit_xyz", action="store_true", default=True, help="emit normalized 3D position")
    parser.add_argument("--no-xyz", dest="emit_xyz", action="store_false", help="disable 3D position emission")
    parser.add_argument("--xyz-entropy-max", type=float, default=8.0, help="entropy max for X normalization")
    parser.add_argument("--xyz-margin-max", type=float, default=10.0, help="logit margin max for Y normalization")
    parser.add_argument("--xyz-drift-max", type=float, default=2.0, help="drift max for Z normalization")
    parser.add_argument("--xyz-alpha", type=float, default=0.2, help="EMA smoothing factor for XYZ (0 disables)")
    parser.add_argument("--api-key", default=None, help="API key for server (if required)")
    parser.add_argument("--id-slot", type=int, default=0, help="server slot id for KV cache reuse")
    parser.add_argument("--params", help="path to JSON file with extra generation params")
    parser.add_argument("--commit-file", default="llama_cpp_commit.txt", help="path to pinned llama.cpp commit file")
    parser.add_argument("--pin-commit", action="store_true", help="write commit hash from /props to commit file")
    parser.add_argument("--no-schema", action="store_true", help="do not emit schema/session_start record")
    args = parser.parse_args()

    prompt = load_prompt(args)
    extra_params = load_extra_params(args.params)
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")
    if args.topk < 1:
        raise SystemExit("--topk must be >= 1")
    if args.sample_tokens < 1:
        raise SystemExit("--sample-tokens must be >= 1")
    if args.sample_ms < 1:
        raise SystemExit("--sample-ms must be >= 1")
    if args.window_size < 1:
        raise SystemExit("--window-size must be >= 1")
    if args.emit_xyz:
        if args.xyz_entropy_max <= 0.0:
            raise SystemExit("--xyz-entropy-max must be > 0")
        if args.xyz_margin_max <= 0.0:
            raise SystemExit("--xyz-margin-max must be > 0")
        if args.xyz_drift_max <= 0.0:
            raise SystemExit("--xyz-drift-max must be > 0")
        if args.xyz_alpha < 0.0 or args.xyz_alpha > 1.0:
            raise SystemExit("--xyz-alpha must be between 0 and 1")

    # Session metadata from /props (best-effort)
    model_info = {}
    build_info = None
    try:
        props = http_get_json(build_url(args.base_url, "/props"), api_key=args.api_key, timeout=10)
        build_info = props.get("build_info")
        model_info = {
            "model_path": props.get("model_path"),
            "chat_template": props.get("chat_template"),
            "build_info": build_info,
        }
        if isinstance(props.get("modalities"), dict):
            model_info["modalities"] = props.get("modalities")
    except Exception as err:
        eprint(f"Warning: failed to read /props: {err}")

    commit_hash = read_commit_file(args.commit_file)
    if args.pin_commit and not commit_hash:
        commit_hash = parse_commit_from_build_info(build_info)
        if commit_hash:
            try:
                maybe_write_commit_file(args.commit_file, commit_hash)
            except Exception as err:
                eprint(f"Warning: failed to write commit file: {err}")

    t0 = time.monotonic()

    if not args.no_schema:
        emit_json({
            "type": "session_start",
            "schema_version": SCHEMA_VERSION,
            "t_ms": 0,
            "schema": DEFAULT_SCHEMA,
            "units": UNITS,
            "config": {
                "base_url": args.base_url,
                "endpoint": args.endpoint,
                "topk": args.topk,
                "chunk_size": args.chunk_size,
                "sample_tokens": args.sample_tokens,
                "sample_ms": args.sample_ms,
                "window_size": args.window_size,
                "embeddings": args.embeddings,
                "emit_xyz": args.emit_xyz,
                "xyz_entropy_max": args.xyz_entropy_max,
                "xyz_margin_max": args.xyz_margin_max,
                "xyz_drift_max": args.xyz_drift_max,
                "xyz_alpha": args.xyz_alpha,
            },
            "model": {
                **model_info,
                "llama_cpp_commit": commit_hash,
            },
        })

    generated_text = ""
    generated_tokens = []
    total_generated = 0
    last_emit_token_index = -1
    last_emit_time = t0
    last_emit_entropy = None
    last_emit_margin = None
    last_xyz = None
    prev_embedding = None
    prev_embedding_token_index = None

    remaining = args.n_predict
    stop = False
    embeddings_enabled = args.embeddings

    while not stop and (remaining == -1 or remaining > 0):
        chunk_size = args.chunk_size
        if remaining != -1:
            chunk_size = min(chunk_size, remaining)

        payload = {
            "prompt": prompt + generated_text,
            "n_predict": chunk_size,
            "n_probs": args.topk,
            "return_tokens": True,
            "stream": False,
            "cache_prompt": True,
            "id_slot": args.id_slot,
            "echo": False,
        }
        payload.update(extra_params)

        request_start = time.monotonic()
        response = http_post_json(build_url(args.base_url, args.endpoint), payload, api_key=args.api_key)
        request_end = time.monotonic()

        content = extract_content(response)
        probs = extract_probs(response)
        stop_flags = extract_stop_flags(response)
        stop = bool(stop_flags.get("stop"))

        if probs is None:
            raise RuntimeError("Missing probability data in response; ensure n_probs > 0 and stream=false")

        if not isinstance(probs, list):
            raise RuntimeError(f"Unexpected probs format: {type(probs)}")

        num_tokens = len(probs)
        if num_tokens == 0:
            break

        elapsed = max(0.0001, request_end - request_start)

        hard_stop = bool(stop_flags.get("stopped_eos")) or bool(stop_flags.get("stopped_word")) or bool(stop_flags.get("stopping_word"))
        if stop and not hard_stop and (remaining == -1 or remaining > 0):
            stop = False

        chunk_start_index = total_generated
        for i, entry in enumerate(probs):
            token_index = total_generated
            total_generated += 1
            if remaining != -1:
                remaining -= 1

            token_time = request_start + ((i + 1) / num_tokens) * elapsed
            t_ms = int((token_time - t0) * 1000)

            token_id = entry.get("id")
            token_text = entry.get("token", "")

            entropy, logit_margin, topk_mass = compute_metrics(entry, args.topk)

            tokens_since = token_index - last_emit_token_index
            time_since_ms = int((token_time - last_emit_time) * 1000)

            event = None
            if last_emit_entropy is not None and (entropy - last_emit_entropy) >= args.entropy_spike:
                event = "entropy_spike"
            if last_emit_margin is not None and logit_margin is not None:
                if (last_emit_margin - logit_margin) >= args.margin_drop:
                    event = "margin_drop"

            sample_reason = None
            if tokens_since >= args.sample_tokens:
                sample_reason = "interval_tokens"
            if time_since_ms >= args.sample_ms:
                sample_reason = "interval_time"
            if event is not None:
                sample_reason = "event"

            if sample_reason:
                drift = None
                position = None
                if embeddings_enabled:
                    window_tokens = generated_tokens[-args.window_size + 1:] if args.window_size > 1 else []
                    window_tokens.append({"text": token_text})
                    window_text = "".join(t["text"] for t in window_tokens)
                    try:
                        emb_payload = {"content": window_text, "embd_normalize": args.embd_normalize}
                        emb_resp = http_post_json(build_url(args.base_url, args.embedding_endpoint), emb_payload, api_key=args.api_key)
                        embedding = None
                        if "embedding" in emb_resp:
                            embedding = emb_resp.get("embedding")
                        elif "data" in emb_resp and emb_resp["data"]:
                            embedding = emb_resp["data"][0].get("embedding")
                        if isinstance(embedding, list):
                            dist = l2_distance(embedding, prev_embedding)
                            if dist is not None and prev_embedding_token_index is not None:
                                delta_tokens = max(1, token_index - prev_embedding_token_index)
                                drift = dist / float(delta_tokens)
                            prev_embedding = embedding
                            prev_embedding_token_index = token_index
                    except Exception as err:
                        msg = str(err)
                        if "HTTP 501" in msg or "not implemented" in msg.lower():
                            eprint("Warning: embedding endpoint unavailable; disabling embeddings for this session")
                            embeddings_enabled = False
                        else:
                            eprint(f"Warning: embedding call failed: {err}")

                if args.emit_xyz:
                    x_raw = normalize_unit(entropy, args.xyz_entropy_max)
                    y_raw = normalize_unit(logit_margin, args.xyz_margin_max)
                    z_raw = normalize_unit(drift, args.xyz_drift_max)

                    fallback_x = last_xyz[0] if last_xyz else None
                    fallback_y = last_xyz[1] if last_xyz else None
                    fallback_z = last_xyz[2] if last_xyz else None

                    x = with_fallback(x_raw, fallback_x, 0.0)
                    y = with_fallback(y_raw, fallback_y, 0.0)
                    z = with_fallback(z_raw, fallback_z, 0.0)

                    if args.xyz_alpha > 0.0 and last_xyz:
                        x = smooth_value(x, last_xyz[0], args.xyz_alpha)
                        y = smooth_value(y, last_xyz[1], args.xyz_alpha)
                        z = smooth_value(z, last_xyz[2], args.xyz_alpha)

                    position = {"x": x, "y": y, "z": z}
                    last_xyz = (x, y, z)

                record = {
                    "type": "telemetry",
                    "schema_version": SCHEMA_VERSION,
                    "t_ms": t_ms,
                    "token_index": token_index,
                    "token_id": token_id,
                    "token_text": token_text,
                    "metrics": {
                        "entropy": entropy,
                        "logit_margin": logit_margin,
                        "topk_mass": topk_mass,
                        "drift": drift,
                    },
                    "sample_reason": sample_reason,
                    "event": event,
                }
                if args.emit_xyz:
                    record["position"] = position
                emit_json(record)

                last_emit_token_index = token_index
                last_emit_time = token_time
                last_emit_entropy = entropy
                last_emit_margin = logit_margin

            generated_tokens.append({"id": token_id, "text": token_text})

        if content:
            emit_json({
                "type": "text",
                "schema_version": SCHEMA_VERSION,
                "t_ms": int((request_end - t0) * 1000),
                "text": content,
                "token_index_start": chunk_start_index,
                "token_index_end": total_generated - 1,
            })

        generated_text += content

    emit_json({
        "type": "session_end",
        "schema_version": SCHEMA_VERSION,
        "t_ms": int((time.monotonic() - t0) * 1000),
        "token_count": total_generated,
        "stop": stop,
    })


if __name__ == "__main__":
    main()
