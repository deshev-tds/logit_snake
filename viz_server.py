#!/usr/bin/env python3
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse


CLIENTS = set()
CLIENTS_LOCK = threading.Lock()

RUNS = {}
RUN_ORDER = []
RUNS_LOCK = threading.Lock()

GEN_LOCK = threading.Lock()
GEN_THREAD = None
GEN_STOP = None
GEN_RUN_ID = None

DEFAULTS = {}

ALT_TOKEN_POOL = [
    " the",
    " a",
    " and",
    " to",
    " of",
    " in",
    ",",
    ".",
    " that",
    " is",
    " with",
    " for",
    " on",
    " as",
    " by",
    " it",
    " this",
    " from",
    "\n",
]


def utc_now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def short_hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def build_url(base_url, path):
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def first_response(response):
    if isinstance(response, list):
        if not response:
            return {}
        first = response[0]
        return first if isinstance(first, dict) else {}
    return response if isinstance(response, dict) else {}


def http_post_json(url, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
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


def http_get_json(url, timeout=30):
    req = urllib.request.Request(url, method="GET")
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


def normalize_top_entries(entries):
    out = []
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        token_text = item.get("token", "")
        token_id = item.get("id")
        prob = item.get("prob")
        logprob = item.get("logprob")

        if logprob is None and prob is None:
            continue
        if logprob is None:
            try:
                prob = float(prob)
                logprob = float("-inf") if prob <= 0.0 else math.log(prob)
            except (TypeError, ValueError):
                continue
        else:
            try:
                logprob = float(logprob)
            except (TypeError, ValueError):
                continue
            if prob is None:
                prob = math.exp(logprob) if math.isfinite(logprob) else 0.0
            else:
                try:
                    prob = float(prob)
                except (TypeError, ValueError):
                    prob = math.exp(logprob) if math.isfinite(logprob) else 0.0

        out.append(
            {
                "token_id": token_id,
                "token_text": token_text,
                "logprob": logprob,
                "prob": max(0.0, float(prob)),
            }
        )
    out.sort(key=lambda x: x["prob"], reverse=True)
    return out


def compute_entropy_margin(topn):
    if not topn:
        return None, None
    mass = sum(max(0.0, t.get("prob") or 0.0) for t in topn)
    entropy = 0.0
    for item in topn:
        p = max(0.0, item.get("prob") or 0.0)
        if p > 0:
            entropy -= p * math.log(p)
    p_rest = max(0.0, 1.0 - mass)
    if p_rest > 0:
        entropy -= p_rest * math.log(p_rest)

    margin = None
    if len(topn) >= 2:
        a = topn[0].get("logprob")
        b = topn[1].get("logprob")
        if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
            margin = a - b
    return entropy, margin


def l2(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    acc = 0.0
    for x, y in zip(a, b):
        d = x - y
        acc += d * d
    return math.sqrt(acc)


def vector_sub(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    return [x - y for x, y in zip(a, b)]


def vector_norm(v):
    if not v:
        return 0.0
    return math.sqrt(sum(x * x for x in v))


def angle_between(a, b):
    if not a or not b or len(a) != len(b):
        return None
    na = vector_norm(a)
    nb = vector_norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    cos_theta = max(-1.0, min(1.0, dot / (na * nb)))
    return math.acos(cos_theta)


def stable_seed(*parts):
    joined = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def tokenize_fallback(text):
    if not text:
        return []
    parts = re.findall(r"\s+|[^\s]+", text, flags=re.UNICODE)
    return parts or [text]


def synthetic_topn(chosen_text, chosen_id, seed_value, n=5):
    if n < 1:
        n = 1
    rng = random.Random(seed_value)
    probs_template = [0.62, 0.17, 0.10, 0.07, 0.04]
    probs = probs_template[:n]
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]

    entries = [
        {
            "token_id": chosen_id,
            "token_text": chosen_text,
            "prob": probs[0],
            "logprob": math.log(probs[0]),
        }
    ]

    used = {chosen_text}
    for i in range(1, n):
        candidate = None
        for _ in range(8):
            candidate = ALT_TOKEN_POOL[rng.randrange(len(ALT_TOKEN_POOL))]
            if candidate not in used:
                break
        if candidate in used:
            candidate = f"{chosen_text}{i}"
        used.add(candidate)
        p = probs[i]
        entries.append(
            {
                "token_id": None,
                "token_text": candidate,
                "prob": p,
                "logprob": math.log(p),
            }
        )

    entries.sort(key=lambda x: x["prob"], reverse=True)
    return entries


def extract_content(response):
    resp = first_response(response)
    if "content" in resp and isinstance(resp.get("content"), str):
        return resp.get("content", "")

    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    return ""


def extract_probs(response):
    resp = first_response(response)
    if "probs" in resp:
        probs = resp.get("probs")
        return probs if isinstance(probs, list) else None

    comp_probs = resp.get("completion_probabilities")
    if isinstance(comp_probs, list):
        if not comp_probs:
            return []
        first = comp_probs[0]
        if isinstance(first, dict) and isinstance(first.get("probs"), list):
            return first.get("probs")
        if isinstance(first, list):
            return first
        if all(isinstance(x, dict) for x in comp_probs):
            return comp_probs

    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        logprobs = choices[0].get("logprobs") if isinstance(choices[0], dict) else None
        if isinstance(logprobs, dict):
            content = logprobs.get("content")
            if isinstance(content, list):
                out = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    top = item.get("top_logprobs") or []
                    top_entries = []
                    for t in top:
                        if not isinstance(t, dict):
                            continue
                        top_entries.append({"token": t.get("token", ""), "logprob": t.get("logprob")})
                    out.append(
                        {
                            "token": item.get("token", ""),
                            "logprob": item.get("logprob"),
                            "top_logprobs": top_entries,
                            "id": item.get("id"),
                        }
                    )
                return out
            top = logprobs.get("top_logprobs")
            if isinstance(top, list):
                return top
    return None


def extract_stop_flag(response):
    resp = first_response(response)
    if not isinstance(resp, dict):
        return False
    if resp.get("stop"):
        return True
    return bool(resp.get("stopped_eos") or resp.get("stopped_word") or resp.get("stopping_word"))


def sanitize_int(value, default_value, minimum=None, maximum=None):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default_value
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def sanitize_float(value, default_value, minimum=None, maximum=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default_value
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


class VectorProvider:
    def __init__(self, mode, base_url, dim, window_size, prompt_hash):
        self.mode = mode
        self.base_url = base_url
        self.dim = max(4, int(dim))
        self.window_size = max(1, int(window_size))
        self.prompt_hash = prompt_hash
        self.prev = None

    def _normalize(self, vec):
        norm = math.sqrt(sum(x * x for x in vec))
        if norm <= 1e-12:
            return [0.0 for _ in vec]
        return [x / norm for x in vec]

    def _placeholder(self, token_index, token_id, token_text):
        seed = stable_seed(self.prompt_hash, token_index, token_id, token_text)
        rng = random.Random(seed)
        raw = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        if self.prev is not None and len(self.prev) == self.dim:
            raw = [0.72 * p + 0.28 * r for p, r in zip(self.prev, raw)]
        vec = self._normalize(raw)
        self.prev = vec
        return vec

    def _real_embedding(self, token_history_text):
        payload = {"content": token_history_text}
        try:
            resp = http_post_json(build_url(self.base_url, "/embedding"), payload, timeout=20)
        except Exception:
            return None

        embedding = None
        if isinstance(resp, dict):
            if isinstance(resp.get("embedding"), list):
                embedding = resp.get("embedding")
            elif isinstance(resp.get("data"), list) and resp.get("data"):
                first = resp["data"][0]
                if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                    embedding = first.get("embedding")
        if not isinstance(embedding, list):
            return None

        clean = []
        for x in embedding[: self.dim]:
            try:
                clean.append(float(x))
            except (TypeError, ValueError):
                clean.append(0.0)
        if len(clean) < self.dim:
            clean.extend([0.0] * (self.dim - len(clean)))
        vec = self._normalize(clean)
        self.prev = vec
        return vec

    def vector_for(self, token_index, token_id, token_text, token_history):
        if self.mode == "real":
            window = token_history[-self.window_size :]
            text = "".join(window)
            vec = self._real_embedding(text)
            if vec is not None:
                return vec
            self.mode = "placeholder"
        return self._placeholder(token_index, token_id, token_text)



def choose_topn(entry, topn_n, prompt_hash, token_index):
    chosen_id = entry.get("id")
    chosen_text = entry.get("token", "")

    backend_top = normalize_top_entries(entry.get("top_logprobs") or entry.get("top_probs") or [])
    chosen_logprob = entry.get("logprob")
    chosen_prob = entry.get("prob")

    try:
        chosen_logprob = float(chosen_logprob) if chosen_logprob is not None else None
    except (TypeError, ValueError):
        chosen_logprob = None
    try:
        chosen_prob = float(chosen_prob) if chosen_prob is not None else None
    except (TypeError, ValueError):
        chosen_prob = None

    provider = "backend_logprobs"

    if chosen_logprob is None and chosen_prob is not None and chosen_prob > 0.0:
        chosen_logprob = math.log(chosen_prob)
    if chosen_prob is None and chosen_logprob is not None and math.isfinite(chosen_logprob):
        chosen_prob = math.exp(chosen_logprob)

    if not backend_top:
        provider = "deterministic_approx"
        backend_top = synthetic_topn(chosen_text, chosen_id, stable_seed(prompt_hash, token_index, chosen_text), n=topn_n)
        chosen = backend_top[0]
        chosen_logprob = chosen["logprob"]
        chosen_prob = chosen["prob"]
    else:
        found = None
        for item in backend_top:
            if chosen_id is not None and item.get("token_id") == chosen_id:
                found = item
                break
            if chosen_text and item.get("token_text") == chosen_text:
                found = item
                break
        if found is not None:
            if chosen_logprob is None:
                chosen_logprob = found.get("logprob")
            if chosen_prob is None:
                chosen_prob = found.get("prob")
        else:
            if chosen_logprob is None or chosen_prob is None:
                top0 = backend_top[0]
                chosen_logprob = top0.get("logprob")
                chosen_prob = top0.get("prob")

        chosen_entry = {
            "token_id": chosen_id,
            "token_text": chosen_text,
            "logprob": chosen_logprob,
            "prob": chosen_prob if chosen_prob is not None else 0.0,
        }
        has_chosen = any(
            (chosen_id is not None and e.get("token_id") == chosen_id)
            or (chosen_text and e.get("token_text") == chosen_text)
            for e in backend_top
        )
        if not has_chosen:
            backend_top.append(chosen_entry)
            backend_top.sort(key=lambda x: x.get("prob") or 0.0, reverse=True)

    topn = backend_top[: max(1, topn_n)]
    entropy, margin = compute_entropy_margin(topn)

    return {
        "chosen_token_id": chosen_id,
        "chosen_token_text": chosen_text,
        "logprob": chosen_logprob,
        "prob": chosen_prob,
        "topN": topn,
        "entropy": entropy,
        "margin": margin,
        "provider": provider,
    }


def token_record_from_entry(entry, topn_n, prompt_hash, token_index, t_ms):
    info = choose_topn(entry, topn_n, prompt_hash, token_index)
    return {
        "index": token_index,
        "t": t_ms,
        "text": info["chosen_token_text"],
        "chosen_token_id": info["chosen_token_id"],
        "chosen_token_text": info["chosen_token_text"],
        "logprob": info["logprob"],
        "prob": info["prob"],
        "entropy": info["entropy"],
        "margin": info["margin"],
        "topN": info["topN"],
        "topn_provider": info["provider"],
    }


def token_record_fallback(token_text, topn_n, prompt_hash, token_index, t_ms):
    topn = synthetic_topn(token_text, None, stable_seed(prompt_hash, token_index, token_text), n=topn_n)
    entropy, margin = compute_entropy_margin(topn)
    return {
        "index": token_index,
        "t": t_ms,
        "text": token_text,
        "chosen_token_id": None,
        "chosen_token_text": token_text,
        "logprob": topn[0]["logprob"],
        "prob": topn[0]["prob"],
        "entropy": entropy,
        "margin": margin,
        "topN": topn,
        "topn_provider": "deterministic_approx",
    }


def recompute_kinematics(tokens):
    prev_vec = None
    prev_delta = None
    for token in tokens:
        vec = token.get("embedding")
        velocity = l2(vec, prev_vec) if prev_vec is not None else 0.0
        delta = vector_sub(vec, prev_vec) if prev_vec is not None else None
        curvature = angle_between(prev_delta, delta) if prev_delta is not None and delta is not None else None
        token["velocity"] = velocity
        token["curvature"] = curvature
        prev_vec = vec
        prev_delta = delta


def detect_regime_markers(tokens):
    if len(tokens) < 4:
        return []
    velocities = [float(t.get("velocity") or 0.0) for t in tokens]
    entropy_slope = [0.0]
    for i in range(1, len(tokens)):
        e0 = tokens[i - 1].get("entropy")
        e1 = tokens[i].get("entropy")
        if e0 is None or e1 is None:
            entropy_slope.append(0.0)
        else:
            entropy_slope.append(abs(float(e1) - float(e0)))

    def stats(arr):
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / max(1, len(arr) - 1)
        return mean, math.sqrt(var)

    v_mean, v_std = stats(velocities)
    e_mean, e_std = stats(entropy_slope)
    v_thr = v_mean + 2.0 * v_std
    e_thr = e_mean + 2.0 * e_std

    markers = []
    last_idx = -10
    for i in range(1, len(tokens)):
        reasons = []
        if velocities[i] > v_thr:
            reasons.append("velocity_spike")
        if entropy_slope[i] > e_thr:
            reasons.append("entropy_slope_spike")
        if reasons and i - last_idx >= 3:
            markers.append({"index": i, "reasons": reasons})
            last_idx = i
    return markers


def run_summary(run):
    meta = run.get("meta", {})
    branch = meta.get("branch") or {}
    return {
        "run_id": run.get("run_id"),
        "label": meta.get("label"),
        "status": meta.get("status"),
        "token_count": len(run.get("tokens") or []),
        "model": meta.get("model"),
        "timestamp": meta.get("timestamp"),
        "parent_run_id": branch.get("parent_run_id"),
        "fork_index": branch.get("fork_index"),
    }


def send_json(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        return json.loads(raw.decode("utf-8")) if raw else {}
    except json.JSONDecodeError:
        return {}


def normalize_base_url(raw_value):
    raw = str(raw_value or "").strip()
    if not raw:
        return ""

    # Guard against accidental concatenation like:
    # http://127.0.0.1:8080http://192.168.1.117:1234/
    starts = []
    for prefix in ("http://", "https://"):
        idx = raw.find(prefix)
        while idx != -1:
            starts.append(idx)
            idx = raw.find(prefix, idx + 1)
    if len(starts) > 1:
        raw = raw[max(starts) :]

    parsed = urlparse(raw)
    if not parsed.scheme:
        raw = "http://" + raw
        parsed = urlparse(raw)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError("invalid base_url")

    path = parsed.path or ""
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")
    return normalized


def broadcast_event(event):
    payload = json.dumps(event, ensure_ascii=True).encode("utf-8")
    with CLIENTS_LOCK:
        queues = list(CLIENTS)
    stale = []
    for q in queues:
        try:
            q.put_nowait(payload)
        except queue.Full:
            stale.append(q)
    if stale:
        with CLIENTS_LOCK:
            for q in stale:
                CLIENTS.discard(q)


def fetch_model_name(base_url):
    props_model = None
    try:
        props = http_get_json(build_url(base_url, "/props"), timeout=8)
        if isinstance(props, dict):
            candidate = props.get("model_path") or props.get("model")
            if candidate and str(candidate).strip().lower() not in ("none", "null"):
                return str(candidate)
            props_model = candidate
    except Exception:
        pass

    # Router-style deployments can require explicit model names.
    try:
        models_resp = http_get_json(build_url(base_url, "/v1/models"), timeout=8)
        data = models_resp.get("data") if isinstance(models_resp, dict) else None
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                status = item.get("status")
                if isinstance(status, dict) and status.get("value") == "loaded" and item.get("id"):
                    return str(item.get("id"))
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    return str(item.get("id"))
    except Exception:
        pass

    if props_model and str(props_model).strip().lower() not in ("none", "null"):
        return str(props_model)
    return None


def sanitize_settings(raw_settings):
    raw_settings = raw_settings if isinstance(raw_settings, dict) else {}
    settings = {
        "max_tokens": sanitize_int(raw_settings.get("max_tokens", DEFAULTS["max_tokens"]), DEFAULTS["max_tokens"], minimum=1, maximum=4096),
        "chunk_size": sanitize_int(raw_settings.get("chunk_size", DEFAULTS["chunk_size"]), DEFAULTS["chunk_size"], minimum=1, maximum=256),
        "top_n": sanitize_int(raw_settings.get("top_n", DEFAULTS["top_n"]), DEFAULTS["top_n"], minimum=1, maximum=20),
        "n_probs": sanitize_int(raw_settings.get("n_probs", DEFAULTS["n_probs"]), DEFAULTS["n_probs"], minimum=1, maximum=100),
        "temperature": sanitize_float(raw_settings.get("temperature", DEFAULTS["temperature"]), DEFAULTS["temperature"], minimum=0.0, maximum=2.0),
        "top_p": sanitize_float(raw_settings.get("top_p", DEFAULTS["top_p"]), DEFAULTS["top_p"], minimum=0.0, maximum=1.0),
        "seed": sanitize_int(raw_settings.get("seed", DEFAULTS["seed"]), DEFAULTS["seed"]),
        "vector_mode": (raw_settings.get("vector_mode") or DEFAULTS["vector_mode"]).strip().lower(),
        "vector_dim": sanitize_int(raw_settings.get("vector_dim", DEFAULTS["vector_dim"]), DEFAULTS["vector_dim"], minimum=4, maximum=256),
        "vector_window": sanitize_int(raw_settings.get("vector_window", DEFAULTS["vector_window"]), DEFAULTS["vector_window"], minimum=1, maximum=256),
        "id_slot": sanitize_int(raw_settings.get("id_slot", DEFAULTS["id_slot"]), DEFAULTS["id_slot"], minimum=0, maximum=8),
    }
    if settings["vector_mode"] not in ("placeholder", "real"):
        settings["vector_mode"] = "placeholder"

    extra_params = raw_settings.get("extra_params")
    settings["extra_params"] = extra_params if isinstance(extra_params, dict) else {}
    return settings


def create_run_object(prompt, base_url, settings, label=None, branch_meta=None, prefill_tokens=None, inference_prompt=None):
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    prompt_hash = short_hash(prompt)
    model_name = fetch_model_name(base_url)
    base_tokens = copy.deepcopy(prefill_tokens or [])

    run = {
        "schema_version": "2.0",
        "run_id": run_id,
        "tokens": base_tokens,
        "bookmarks": [],
        "meta": {
            "label": label or ("Branch" if branch_meta else "Run"),
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "timestamp": utc_now_iso(),
            "base_url": base_url,
            "model": model_name,
            "status": "running",
            "generation_settings": settings,
            "branch": branch_meta,
            "inference_prompt": inference_prompt if inference_prompt is not None else prompt,
            "providers": {
                "topN": "backend_logprobs",
                "embedding": settings.get("vector_mode", "placeholder"),
            },
        },
        "analysis": {
            "regime_markers": [],
        },
        "summary": {},
    }
    return run


def append_token(run_id, token):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            return False
        run["tokens"].append(token)
        run["meta"]["providers"]["topN"] = token.get("topn_provider", run["meta"]["providers"].get("topN"))
    broadcast_event({"type": "token", "run_id": run_id, "token": token})
    return True


def generate_completion_chunk(base_url, prompt, chunk_size, settings, model_name=None):
    payload = {
        "prompt": prompt,
        "n_predict": chunk_size,
        "n_probs": max(settings.get("n_probs") or 5, settings.get("top_n") or 5),
        "return_tokens": True,
        "stream": False,
        "cache_prompt": True,
        "id_slot": settings.get("id_slot", 0),
        "echo": False,
        "temperature": settings.get("temperature"),
        "top_p": settings.get("top_p"),
        "seed": settings.get("seed"),
    }
    if model_name and str(model_name).strip().lower() not in ("none", "null"):
        payload["model"] = model_name
    extra = settings.get("extra_params") or {}
    payload.update(extra)
    return http_post_json(build_url(base_url, "/completion"), payload)


def initialize_prefill_tokens(run):
    tokens = run.get("tokens") or []
    meta = run.get("meta") or {}
    settings = meta.get("generation_settings") or {}
    provider = VectorProvider(
        settings.get("vector_mode", "placeholder"),
        meta.get("base_url", DEFAULTS["base_url"]),
        settings.get("vector_dim", DEFAULTS["vector_dim"]),
        settings.get("vector_window", DEFAULTS["vector_window"]),
        meta.get("prompt_hash", ""),
    )

    history_text = []
    for i, token in enumerate(tokens):
        token["index"] = i
        text = token.get("text") or token.get("chosen_token_text") or ""
        token["text"] = text
        if not isinstance(token.get("topN"), list) or not token.get("topN"):
            topn = synthetic_topn(text, token.get("chosen_token_id"), stable_seed(meta.get("prompt_hash", ""), i, text), n=settings.get("top_n", 5))
            token["topN"] = topn
            token["topn_provider"] = "deterministic_approx"
        entropy, margin = compute_entropy_margin(token.get("topN"))
        if token.get("entropy") is None:
            token["entropy"] = entropy
        if token.get("margin") is None:
            token["margin"] = margin

        if not isinstance(token.get("embedding"), list):
            token["embedding"] = provider.vector_for(i, token.get("chosen_token_id"), text, history_text)
        else:
            try:
                clean = [float(x) for x in token.get("embedding")]
                token["embedding"] = clean
                provider.prev = clean
            except Exception:
                token["embedding"] = provider.vector_for(i, token.get("chosen_token_id"), text, history_text)

        history_text.append(text)

    recompute_kinematics(tokens)


def run_worker(run_id, stop_event):
    started = time.monotonic()
    error_message = None

    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if run is None:
        return

    try:
        initialize_prefill_tokens(run)

        meta = run["meta"]
        settings = meta["generation_settings"]
        prompt_hash = meta["prompt_hash"]
        base_url = meta["base_url"]
        inference_prompt = meta.get("inference_prompt") or meta.get("prompt", "")
        model_name = meta.get("model")

        vector_provider = VectorProvider(
            settings.get("vector_mode", "placeholder"),
            base_url,
            settings.get("vector_dim", DEFAULTS["vector_dim"]),
            settings.get("vector_window", DEFAULTS["vector_window"]),
            prompt_hash,
        )

        with RUNS_LOCK:
            tokens = RUNS[run_id]["tokens"]
            history_text = [t.get("text") or "" for t in tokens]
            if tokens and isinstance(tokens[-1].get("embedding"), list):
                vector_provider.prev = tokens[-1]["embedding"]
            token_index = len(tokens)

        generated_text = ""
        remaining = max(0, settings.get("max_tokens", DEFAULTS["max_tokens"]) - token_index)

        while remaining > 0 and not stop_event.is_set():
            chunk_size = min(settings.get("chunk_size", DEFAULTS["chunk_size"]), remaining)
            try:
                response = generate_completion_chunk(
                    base_url,
                    inference_prompt + generated_text,
                    chunk_size,
                    settings,
                    model_name=model_name,
                )
            except Exception as req_err:
                req_msg = str(req_err)
                if "model name is missing" in req_msg.lower():
                    refreshed_model = fetch_model_name(base_url)
                    if refreshed_model:
                        model_name = refreshed_model
                        with RUNS_LOCK:
                            live_run = RUNS.get(run_id)
                            if live_run is not None:
                                live_run["meta"]["model"] = refreshed_model
                        response = generate_completion_chunk(
                            base_url,
                            inference_prompt + generated_text,
                            chunk_size,
                            settings,
                            model_name=model_name,
                        )
                    else:
                        raise
                else:
                    raise
            content = extract_content(response)
            probs = extract_probs(response)
            stopped = extract_stop_flag(response)

            emitted_this_chunk = 0
            chunk_time_ms = int((time.monotonic() - started) * 1000)

            if isinstance(probs, list) and probs:
                for entry in probs:
                    if stop_event.is_set() or remaining <= 0:
                        break
                    if not isinstance(entry, dict):
                        continue
                    t_ms = int((time.monotonic() - started) * 1000)
                    token = token_record_from_entry(entry, settings.get("top_n", DEFAULTS["top_n"]), prompt_hash, token_index, t_ms)
                    token_text = token.get("text", "")
                    token["embedding"] = vector_provider.vector_for(token_index, token.get("chosen_token_id"), token_text, history_text)

                    generated_text += token_text
                    history_text.append(token_text)

                    append_token(run_id, token)
                    token_index += 1
                    remaining -= 1
                    emitted_this_chunk += 1
            else:
                fallback_tokens = tokenize_fallback(content)
                for raw_token in fallback_tokens:
                    if stop_event.is_set() or remaining <= 0:
                        break
                    t_ms = int((time.monotonic() - started) * 1000)
                    token = token_record_fallback(raw_token, settings.get("top_n", DEFAULTS["top_n"]), prompt_hash, token_index, t_ms)
                    token["embedding"] = vector_provider.vector_for(token_index, token.get("chosen_token_id"), raw_token, history_text)
                    generated_text += raw_token
                    history_text.append(raw_token)

                    append_token(run_id, token)
                    token_index += 1
                    remaining -= 1
                    emitted_this_chunk += 1

            if emitted_this_chunk == 0:
                break
            if stopped:
                break

        with RUNS_LOCK:
            final_run = RUNS.get(run_id)
            if final_run is not None:
                recompute_kinematics(final_run["tokens"])
                final_run["analysis"]["regime_markers"] = detect_regime_markers(final_run["tokens"])
                final_run["meta"]["status"] = "stopped" if stop_event.is_set() else "complete"
                final_run["meta"]["completed_at"] = utc_now_iso()

                entropies = [t.get("entropy") for t in final_run["tokens"] if t.get("entropy") is not None]
                velocities = [t.get("velocity") for t in final_run["tokens"] if t.get("velocity") is not None]
                final_run["summary"] = {
                    "token_count": len(final_run["tokens"]),
                    "entropy_avg": (sum(entropies) / len(entropies)) if entropies else None,
                    "velocity_max": max(velocities) if velocities else None,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
    except Exception as err:
        error_message = str(err)
        sys.stderr.write(f"Run {run_id} failed: {error_message}\n")
        with RUNS_LOCK:
            failed = RUNS.get(run_id)
            if failed is not None:
                failed["meta"]["status"] = "error"
                failed["meta"]["completed_at"] = utc_now_iso()
                failed["meta"]["error"] = error_message
                failed["summary"] = {
                    "token_count": len(failed.get("tokens") or []),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "error": error_message,
                }

    with RUNS_LOCK:
        completed_run = copy.deepcopy(RUNS.get(run_id))

    if error_message:
        broadcast_event({"type": "run_error", "run_id": run_id, "error": error_message})
    if completed_run is not None:
        broadcast_event({"type": "run_completed", "run_id": run_id, "run": completed_run})

    global GEN_THREAD, GEN_STOP, GEN_RUN_ID
    with GEN_LOCK:
        if GEN_RUN_ID == run_id:
            GEN_THREAD = None
            GEN_STOP = None
            GEN_RUN_ID = None


def start_run_generation(run):
    global GEN_THREAD, GEN_STOP, GEN_RUN_ID

    stop_generation()

    run_id = run["run_id"]
    with RUNS_LOCK:
        RUNS[run_id] = run
        RUN_ORDER.append(run_id)

    stop_event = threading.Event()
    thread = threading.Thread(target=run_worker, args=(run_id, stop_event), daemon=True)
    with GEN_LOCK:
        GEN_THREAD = thread
        GEN_STOP = stop_event
        GEN_RUN_ID = run_id

    broadcast_event({"type": "run_started", "run": copy.deepcopy(run)})
    thread.start()


def stop_generation():
    global GEN_THREAD, GEN_STOP, GEN_RUN_ID

    with GEN_LOCK:
        thread = GEN_THREAD
        stop_event = GEN_STOP
        run_id = GEN_RUN_ID

    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=3)

    with GEN_LOCK:
        if GEN_THREAD is thread:
            GEN_THREAD = None
            GEN_STOP = None
            GEN_RUN_ID = None

    if run_id:
        with RUNS_LOCK:
            run = RUNS.get(run_id)
            if run and run.get("meta", {}).get("status") == "running":
                run["meta"]["status"] = "stopped"


def build_branch_run(body):
    run_id = body.get("run_id")
    fork_index = body.get("fork_index")

    if not run_id:
        raise ValueError("run_id is required")
    try:
        fork_index = int(fork_index)
    except (TypeError, ValueError):
        raise ValueError("fork_index must be an integer")

    with RUNS_LOCK:
        parent = copy.deepcopy(RUNS.get(run_id))
    if parent is None:
        raise ValueError("parent run not found")

    parent_tokens = parent.get("tokens") or []
    if not parent_tokens:
        raise ValueError("parent run has no tokens")
    if fork_index < 0 or fork_index >= len(parent_tokens):
        raise ValueError("fork_index out of range")

    target = parent_tokens[fork_index]
    topn = target.get("topN") or []

    alt_rank = body.get("alt_rank")
    chosen_alt = None
    if alt_rank is not None:
        try:
            alt_rank = int(alt_rank)
        except (TypeError, ValueError):
            raise ValueError("alt_rank must be an integer")
        if alt_rank < 0 or alt_rank >= len(topn):
            raise ValueError("alt_rank out of range")
        chosen_alt = topn[alt_rank]
    else:
        requested_id = body.get("alt_token_id")
        requested_text = body.get("alt_token_text")
        for item in topn:
            if requested_id is not None and item.get("token_id") == requested_id:
                chosen_alt = item
                break
            if requested_text is not None and item.get("token_text") == requested_text:
                chosen_alt = item
                break
        if chosen_alt is None and topn:
            chosen_alt = topn[0]

    if chosen_alt is None:
        raise ValueError("no alternative token available")

    prefix_tokens = copy.deepcopy(parent_tokens[:fork_index])
    forced_token = {
        "index": fork_index,
        "t": target.get("t"),
        "text": chosen_alt.get("token_text", ""),
        "chosen_token_id": chosen_alt.get("token_id"),
        "chosen_token_text": chosen_alt.get("token_text", ""),
        "logprob": chosen_alt.get("logprob"),
        "prob": chosen_alt.get("prob"),
        "entropy": target.get("entropy"),
        "margin": target.get("margin"),
        "topN": copy.deepcopy(topn[: parent.get("meta", {}).get("generation_settings", {}).get("top_n", 5)]),
        "topn_provider": target.get("topn_provider", "backend_logprobs"),
    }

    prefill_tokens = prefix_tokens + [forced_token]

    parent_meta = parent.get("meta", {})
    parent_settings = copy.deepcopy(parent_meta.get("generation_settings") or {})
    parent_prompt = parent_meta.get("prompt", "")

    forced_prefix = "".join(tok.get("text", "") for tok in prefill_tokens)
    inference_prompt = parent_prompt + forced_prefix

    branch_meta = {
        "parent_run_id": run_id,
        "fork_index": fork_index,
        "chosen_alt_token": {
            "token_id": chosen_alt.get("token_id"),
            "token_text": chosen_alt.get("token_text"),
            "logprob": chosen_alt.get("logprob"),
            "prob": chosen_alt.get("prob"),
        },
        "timestamp": utc_now_iso(),
        "forcing_strategy": "append_prefix_fallback",
    }

    label = body.get("label") or f"Branch@{fork_index}"

    return create_run_object(
        prompt=parent_prompt,
        base_url=parent_meta.get("base_url", DEFAULTS["base_url"]),
        settings=sanitize_settings(parent_settings),
        label=label,
        branch_meta=branch_meta,
        prefill_tokens=prefill_tokens,
        inference_prompt=inference_prompt,
    )


class TelemetryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = queue.Queue(maxsize=1000)
            with CLIENTS_LOCK:
                CLIENTS.add(q)

            try:
                self.wfile.write(b":connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        payload = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b":keepalive\n\n")
                        self.wfile.flush()
                        continue
                    if payload is None:
                        break
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with CLIENTS_LOCK:
                    CLIENTS.discard(q)
            return

        if path == "/api/status":
            with GEN_LOCK:
                running = bool(GEN_THREAD and GEN_THREAD.is_alive())
                active_run_id = GEN_RUN_ID
            send_json(self, 200, {"running": running, "active_run_id": active_run_id})
            return

        if path == "/api/runs":
            with GEN_LOCK:
                running = bool(GEN_THREAD and GEN_THREAD.is_alive())
                active_run_id = GEN_RUN_ID
            with RUNS_LOCK:
                items = [run_summary(RUNS[rid]) for rid in RUN_ORDER if rid in RUNS]
            send_json(self, 200, {"runs": items, "running": running, "active_run_id": active_run_id})
            return

        if path.startswith("/api/run/"):
            run_id = path[len("/api/run/") :].strip()
            with RUNS_LOCK:
                run = copy.deepcopy(RUNS.get(run_id))
            if run is None:
                send_json(self, 404, {"error": "run not found"})
            else:
                send_json(self, 200, {"run": run})
            return

        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        body = read_json_body(self)

        if path == "/api/stop":
            stop_generation()
            send_json(self, 200, {"status": "stopped"})
            return

        if path in ("/api/start", "/api/generate"):
            prompt = (body.get("prompt") or "").strip()
            raw_base_url = body.get("base_url") or DEFAULTS["base_url"]
            if not prompt:
                send_json(self, 400, {"error": "prompt is required"})
                return
            try:
                base_url = normalize_base_url(raw_base_url)
            except ValueError:
                send_json(self, 400, {"error": "base_url is invalid"})
                return
            if not base_url:
                send_json(self, 400, {"error": "base_url is required"})
                return

            settings = sanitize_settings(body.get("settings"))
            label = body.get("label") or "Run"
            run = create_run_object(prompt=prompt, base_url=base_url, settings=settings, label=label)
            start_run_generation(run)
            send_json(self, 200, {"status": "started", "run_id": run["run_id"]})
            return

        if path == "/api/branch":
            try:
                run = build_branch_run(body)
            except ValueError as err:
                send_json(self, 400, {"error": str(err)})
                return
            start_run_generation(run)
            send_json(self, 200, {"status": "started", "run_id": run["run_id"]})
            return

        if path == "/api/import-run":
            run = body.get("run")
            if not isinstance(run, dict):
                send_json(self, 400, {"error": "run object is required"})
                return

            run_copy = copy.deepcopy(run)
            run_id = run_copy.get("run_id") or f"run_{uuid.uuid4().hex[:12]}"
            run_copy["run_id"] = run_id
            if "meta" not in run_copy or not isinstance(run_copy["meta"], dict):
                run_copy["meta"] = {}
            run_copy["meta"].setdefault("timestamp", utc_now_iso())
            run_copy["meta"].setdefault("status", "complete")
            run_copy["meta"].setdefault("label", "Imported")
            run_copy.setdefault("tokens", [])
            run_copy.setdefault("analysis", {})
            run_copy.setdefault("summary", {})
            initialize_prefill_tokens(run_copy)
            run_copy["analysis"]["regime_markers"] = detect_regime_markers(run_copy["tokens"])
            run_copy["summary"]["token_count"] = len(run_copy["tokens"])

            with RUNS_LOCK:
                RUNS[run_id] = run_copy
                if run_id not in RUN_ORDER:
                    RUN_ORDER.append(run_id)

            broadcast_event({"type": "run_imported", "run": copy.deepcopy(run_copy)})
            send_json(self, 200, {"status": "imported", "run_id": run_id})
            return

        send_json(self, 404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="2D LLM run visualizer server")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    parser.add_argument("--static-dir", default="viz", help="directory with index.html/app.js")
    parser.add_argument("--default-base-url", default="http://127.0.0.1:8080", help="default backend base URL")
    parser.add_argument("--default-max-tokens", type=int, default=256, help="default generation length")
    parser.add_argument("--default-chunk-size", type=int, default=16, help="default generation chunk size")
    parser.add_argument("--default-topn", type=int, default=5, help="default token alternatives shown")
    parser.add_argument("--default-n-probs", type=int, default=20, help="default backend n_probs")
    parser.add_argument("--default-temperature", type=float, default=0.7, help="default temperature")
    parser.add_argument("--default-top-p", type=float, default=0.95, help="default top_p")
    parser.add_argument("--default-seed", type=int, default=1234, help="default seed")
    parser.add_argument("--default-vector-mode", default="placeholder", help="placeholder | real")
    parser.add_argument("--default-vector-dim", type=int, default=24, help="placeholder embedding dimension")
    parser.add_argument("--default-vector-window", type=int, default=32, help="embedding context window")
    parser.add_argument("--default-id-slot", type=int, default=0, help="backend id_slot")
    args = parser.parse_args()

    static_dir = os.path.abspath(args.static_dir)
    if not os.path.isdir(static_dir):
        raise SystemExit(f"Static dir not found: {static_dir}")

    DEFAULTS.update(
        {
            "base_url": args.default_base_url,
            "max_tokens": max(1, args.default_max_tokens),
            "chunk_size": max(1, args.default_chunk_size),
            "top_n": max(1, args.default_topn),
            "n_probs": max(1, args.default_n_probs),
            "temperature": args.default_temperature,
            "top_p": args.default_top_p,
            "seed": args.default_seed,
            "vector_mode": args.default_vector_mode,
            "vector_dim": max(4, args.default_vector_dim),
            "vector_window": max(1, args.default_vector_window),
            "id_slot": max(0, args.default_id_slot),
        }
    )

    handler = lambda *h_args, **h_kwargs: TelemetryHandler(*h_args, directory=static_dir, **h_kwargs)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    sys.stderr.write(f"Serving on http://{args.host}:{args.port}\n")
    sys.stderr.write("SSE stream: /stream\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_generation()


if __name__ == "__main__":
    main()
