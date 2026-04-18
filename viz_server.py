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
BACKEND_PROBE_CACHE = {}
BACKEND_PROBE_LOCK = threading.Lock()

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


def observed_mass(topn):
    if not topn:
        return None
    mass = sum(max(0.0, float(item.get("prob") or 0.0)) for item in topn)
    return max(0.0, min(1.0, mass))


def top_prob_gap(topn):
    if not topn:
        return None, None, None
    top1 = max(0.0, float(topn[0].get("prob") or 0.0))
    top2 = max(0.0, float(topn[1].get("prob") or 0.0)) if len(topn) > 1 else 0.0
    return top1, top2, max(0.0, top1 - top2)


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


def normalize_repeat_token(text):
    raw = str(text or "")
    compact = re.sub(r"\s+", " ", raw).strip().lower()
    return compact or raw.strip().lower() or raw.lower()


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


def extract_stop_info(response):
    resp = first_response(response)
    info = {
        "stop": False,
        "hard_stop": False,
        "stopped_limit": False,
        "finish_reason": None,
    }
    if not isinstance(resp, dict):
        return info

    stop_flag = bool(resp.get("stop"))
    stopped_eos = bool(resp.get("stopped_eos"))
    stopped_word = bool(resp.get("stopped_word") or resp.get("stopping_word"))
    stopped_limit = bool(resp.get("stopped_limit"))

    finish_reason = resp.get("finish_reason")
    if finish_reason is None:
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            if isinstance(choice0, dict):
                finish_reason = choice0.get("finish_reason")

    finish_reason_text = str(finish_reason).strip().lower() if finish_reason is not None else ""
    if finish_reason_text in ("length", "max_tokens"):
        stopped_limit = True

    hard_stop = stopped_eos or stopped_word or finish_reason_text in (
        "stop",
        "eos",
        "eos_token",
        "stop_sequence",
        "content_filter",
        "tool_calls",
    )

    info["stop"] = stop_flag or hard_stop or stopped_limit
    info["hard_stop"] = hard_stop
    info["stopped_limit"] = stopped_limit
    info["finish_reason"] = finish_reason
    return info


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


def prefill_tokens_from_text(text, prompt_hash, topn_n):
    tokens = []
    for idx, part in enumerate(tokenize_fallback(text)):
        tokens.append(token_record_fallback(part, topn_n, prompt_hash, idx, 0))
    return tokens


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


def enrich_decoder_diagnostics(tokens):
    previous_entropy = None
    seen_norm_tokens = []
    seen_bigrams = []

    for token in tokens:
        topn = token.get("topN") or []
        top1_prob, top2_prob, prob_gap = top_prob_gap(topn)
        mass = observed_mass(topn)
        entropy = token.get("entropy")

        if entropy is None:
            uncertainty = None
        else:
            denom = math.log(max(2, len(topn) + 1))
            uncertainty = max(0.0, min(1.0, float(entropy) / denom)) if denom > 0 else None

        entropy_delta = None
        if entropy is not None and previous_entropy is not None:
            entropy_delta = abs(float(entropy) - previous_entropy)
        previous_entropy = float(entropy) if entropy is not None else previous_entropy

        curr_norm = normalize_repeat_token(token.get("text") or token.get("chosen_token_text") or "")
        lookback_tokens = seen_norm_tokens[-12:]
        unigram_hits = sum(1 for item in lookback_tokens if item and item == curr_norm)
        repeat_unigram = (unigram_hits / len(lookback_tokens)) if lookback_tokens else 0.0

        prev_norm = seen_norm_tokens[-1] if seen_norm_tokens else ""
        current_bigram = f"{prev_norm}|{curr_norm}" if prev_norm or curr_norm else ""
        bigram_hits = sum(1 for item in seen_bigrams[-12:] if item and item == current_bigram)
        repeat_bigram = 1.0 if bigram_hits > 0 else 0.0

        repetition_pressure = max(0.0, min(1.0, 0.6 * repeat_unigram + 0.4 * repeat_bigram))
        gap_risk = None if prob_gap is None else max(0.0, min(1.0, 1.0 - prob_gap))
        volatility = None if entropy_delta is None else max(0.0, min(1.0, entropy_delta / 0.75))

        if any(x is not None for x in (uncertainty, gap_risk, volatility)):
            base = (
                0.38 * (uncertainty or 0.0)
                + 0.24 * (gap_risk or 0.0)
                + 0.22 * (volatility or 0.0)
                + 0.16 * repetition_pressure
            )
            decoder_risk = max(0.0, min(1.0, base))
        else:
            decoder_risk = None

        token["observed_mass"] = mass
        token["top1_prob"] = top1_prob
        token["top2_prob"] = top2_prob
        token["prob_gap"] = prob_gap
        token["uncertainty"] = uncertainty
        token["entropy_delta"] = entropy_delta
        token["repeat_unigram"] = repeat_unigram
        token["repeat_bigram"] = repeat_bigram
        token["repetition_pressure"] = repetition_pressure
        token["decoder_risk"] = decoder_risk

        seen_norm_tokens.append(curr_norm)
        seen_bigrams.append(current_bigram)


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


def detect_decoder_alerts(tokens):
    alerts = []
    last_idx = -10
    for i, token in enumerate(tokens):
        reasons = []
        uncertainty = token.get("uncertainty")
        prob_gap = token.get("prob_gap")
        entropy_delta = token.get("entropy_delta")
        repetition_pressure = token.get("repetition_pressure")
        decoder_risk = token.get("decoder_risk")

        if uncertainty is not None and uncertainty >= 0.72:
            reasons.append("high_uncertainty")
        if prob_gap is not None and prob_gap <= 0.12:
            reasons.append("weak_choice_gap")
        if entropy_delta is not None and entropy_delta >= 0.38:
            reasons.append("uncertainty_jump")
        if repetition_pressure is not None and repetition_pressure >= 0.45:
            reasons.append("repetition_pressure")

        if decoder_risk is not None and decoder_risk >= 0.64 and i - last_idx >= 3:
            alerts.append({"index": i, "risk": decoder_risk, "reasons": reasons or ["decoder_risk"]})
            last_idx = i
    return alerts


def run_summary(run):
    meta = run.get("meta", {})
    branch = meta.get("branch") or {}
    branch_history = meta.get("branch_history")
    if not isinstance(branch_history, list):
        branch_history = []
    return {
        "run_id": run.get("run_id"),
        "label": meta.get("label"),
        "status": meta.get("status"),
        "token_count": len(run.get("tokens") or []),
        "model": meta.get("model"),
        "timestamp": meta.get("timestamp"),
        "parent_run_id": branch.get("parent_run_id"),
        "fork_index": branch.get("fork_index"),
        "mutation_count": len(branch_history),
        "providers": meta.get("providers"),
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


def empty_backend_probe(base_url, probe_state="idle", probe_source=None):
    return {
        "base_url": base_url,
        "reachable": False,
        "completion": False,
        "token_probs": False,
        "embedding": False,
        "strict_token_forcing": "unknown",
        "model_required_explicit": "unknown",
        "stop_semantics": "unknown",
        "model": None,
        "error": None,
        "probe_state": probe_state,
        "probe_source": probe_source,
        "cached": False,
        "probed_at": None,
        "completion_ms": None,
        "embedding_ms": None,
        "probe_ms": None,
    }


def get_cached_backend_probe(base_url):
    with BACKEND_PROBE_LOCK:
        cached = copy.deepcopy(BACKEND_PROBE_CACHE.get(base_url))
    if cached is None:
        return None
    cached["cached"] = True
    return cached


def store_backend_probe(base_url, info):
    with BACKEND_PROBE_LOCK:
        BACKEND_PROBE_CACHE[base_url] = copy.deepcopy(info)


def publish_backend_probe(info):
    payload = copy.deepcopy(info)
    payload["cached"] = True
    broadcast_event({"type": "backend_probe_updated", "backend": payload})


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


def has_embedding_support(base_url):
    started = time.monotonic()
    try:
        resp = http_post_json(build_url(base_url, "/embedding"), {"content": "probe"}, timeout=8)
    except Exception:
        return False, int((time.monotonic() - started) * 1000)
    elapsed = int((time.monotonic() - started) * 1000)
    if isinstance(resp, dict):
        if isinstance(resp.get("embedding"), list):
            return True, elapsed
        data = resp.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return True, elapsed
    return False, elapsed


def infer_stop_semantics(response, stop_info):
    if not isinstance(stop_info, dict):
        return "unknown"
    if stop_info.get("stopped_limit") and not stop_info.get("hard_stop"):
        return "boundary-ambiguous"
    if stop_info.get("hard_stop"):
        return "hard"
    resp = first_response(response)
    if isinstance(resp, dict):
        stop_flag = bool(resp.get("stop"))
        tokens_predicted = sanitize_int(resp.get("tokens_predicted"), 0, minimum=0)
        requested = sanitize_int(
            (resp.get("generation_settings") or {}).get("n_predict"),
            sanitize_int(resp.get("generation_settings", {}).get("max_tokens"), 0, minimum=0) if isinstance(resp.get("generation_settings"), dict) else 0,
            minimum=0,
        )
        if stop_flag and not stop_info.get("hard_stop") and requested > 0 and tokens_predicted >= requested:
            return "boundary-ambiguous"
    return "unknown"


def try_strict_token_forcing_probe(base_url, model_name, base_payload, response):
    probs = extract_probs(response)
    if not isinstance(probs, list) or not probs:
        return "unknown"
    first = probs[0]
    if not isinstance(first, dict):
        return "unknown"

    probe_token = token_record_from_entry(first, 5, "probe_backend", 0, 0)
    topn = probe_token.get("topN") or []
    chosen_id = probe_token.get("chosen_token_id")
    alt = None
    for item in topn:
        token_id = item.get("token_id")
        if token_id is None:
            continue
        if chosen_id is not None and token_id == chosen_id:
            continue
        alt = item
        break
    if alt is None or chosen_id is None:
        return "unknown"

    payload = copy.deepcopy(base_payload)
    if model_name:
        payload["model"] = model_name
    payload["logit_bias"] = {
        str(alt.get("token_id")): 50.0,
        str(chosen_id): -50.0,
    }

    try:
        forced = http_post_json(build_url(base_url, "/completion"), payload, timeout=12)
    except Exception as err:
        msg = str(err).lower()
        if "logit_bias" in msg or "unknown field" in msg or "unknown parameter" in msg or "invalid type" in msg:
            return "no"
        return "unknown"

    forced_probs = extract_probs(forced)
    if isinstance(forced_probs, list) and forced_probs:
        forced_first = forced_probs[0]
        if isinstance(forced_first, dict):
            forced_token = token_record_from_entry(forced_first, 5, "probe_backend_forced", 0, 0)
            if forced_token.get("chosen_token_id") == alt.get("token_id"):
                return "maybe"
    return "unknown"


def probe_backend_uncached(base_url, probe_source="manual"):
    started = time.monotonic()
    info = empty_backend_probe(base_url, probe_state="ready", probe_source=probe_source)
    info["model"] = fetch_model_name(base_url)

    payload = {
        "prompt": "Probe:",
        "n_predict": 1,
        "n_probs": 5,
        "return_tokens": True,
        "stream": False,
        "cache_prompt": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "id_slot": DEFAULTS.get("id_slot", 0),
    }

    response = None
    completion_started = time.monotonic()
    try:
        response = http_post_json(build_url(base_url, "/completion"), payload, timeout=12)
        info["completion"] = True
        info["reachable"] = True
        info["model_required_explicit"] = "no"
    except Exception as err:
        msg = str(err)
        if "model name is missing" in msg.lower():
            info["model_required_explicit"] = "yes"
            retry_payload = copy.deepcopy(payload)
            if info["model"]:
                retry_payload["model"] = info["model"]
                try:
                    response = http_post_json(build_url(base_url, "/completion"), retry_payload, timeout=12)
                    info["completion"] = True
                    info["reachable"] = True
                except Exception as retry_err:
                    info["error"] = str(retry_err)
            else:
                info["error"] = msg
        else:
            info["error"] = msg
    info["completion_ms"] = int((time.monotonic() - completion_started) * 1000)

    if response is not None:
        info["token_probs"] = bool(extract_probs(response))
        info["stop_semantics"] = infer_stop_semantics(response, extract_stop_info(response))
        info["strict_token_forcing"] = try_strict_token_forcing_probe(base_url, info["model"], payload, response)

    try:
        info["embedding"], info["embedding_ms"] = has_embedding_support(base_url)
        if info["embedding"]:
            info["reachable"] = True
    except Exception:
        info["embedding"] = False

    info["probe_ms"] = int((time.monotonic() - started) * 1000)
    info["probed_at"] = utc_now_iso()
    return info


def probe_backend(base_url, refresh=False, probe_source="manual"):
    cached = None if refresh else get_cached_backend_probe(base_url)
    if cached is not None and cached.get("probe_state") in ("warming", "ready"):
        return cached

    info = probe_backend_uncached(base_url, probe_source=probe_source)
    store_backend_probe(base_url, info)
    publish_backend_probe(info)
    return copy.deepcopy(info)


def start_backend_warmup(base_url):
    if not base_url:
        return
    cached = get_cached_backend_probe(base_url)
    if cached is not None and cached.get("probe_state") in ("warming", "ready"):
        return

    warming = empty_backend_probe(base_url, probe_state="warming", probe_source="startup")
    warming["probed_at"] = utc_now_iso()
    store_backend_probe(base_url, warming)
    publish_backend_probe(warming)

    def worker():
        try:
            info = probe_backend_uncached(base_url, probe_source="startup")
        except Exception as err:
            info = empty_backend_probe(base_url, probe_state="ready", probe_source="startup")
            info["error"] = str(err)
            info["probed_at"] = utc_now_iso()
        store_backend_probe(base_url, info)
        publish_backend_probe(info)

    thread = threading.Thread(target=worker, name="backend-warmup", daemon=True)
    thread.start()


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


def create_run_object(
    prompt,
    base_url,
    settings,
    label=None,
    branch_meta=None,
    branch_history=None,
    prefill_tokens=None,
    inference_prompt=None,
):
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    prompt_hash = short_hash(prompt)
    model_name = fetch_model_name(base_url)
    base_tokens = copy.deepcopy(prefill_tokens or [])
    clean_branch_history = copy.deepcopy(branch_history) if isinstance(branch_history, list) else []

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
            "branch_history": clean_branch_history,
            "inference_prompt": inference_prompt if inference_prompt is not None else prompt,
            "providers": {
                "topN": "backend_logprobs",
                "embedding": settings.get("vector_mode", "placeholder"),
            },
        },
        "analysis": {
            "regime_markers": [],
            "decoder_alerts": [],
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
    enrich_decoder_diagnostics(tokens)
    run.setdefault("meta", {}).setdefault("providers", {})
    run["meta"]["providers"]["embedding"] = provider.mode


def finalize_completed_run(run, started, status="complete"):
    recompute_kinematics(run["tokens"])
    enrich_decoder_diagnostics(run["tokens"])
    run["analysis"]["regime_markers"] = detect_regime_markers(run["tokens"])
    run["analysis"]["decoder_alerts"] = detect_decoder_alerts(run["tokens"])
    run["meta"]["status"] = status
    run["meta"]["completed_at"] = utc_now_iso()

    entropies = [t.get("entropy") for t in run["tokens"] if t.get("entropy") is not None]
    velocities = [t.get("velocity") for t in run["tokens"] if t.get("velocity") is not None]
    risks = [t.get("decoder_risk") for t in run["tokens"] if t.get("decoder_risk") is not None]
    run["summary"] = {
        "token_count": len(run["tokens"]),
        "entropy_avg": (sum(entropies) / len(entropies)) if entropies else None,
        "velocity_max": max(velocities) if velocities else None,
        "decoder_risk_max": max(risks) if risks else None,
        "decoder_alert_count": len(run["analysis"]["decoder_alerts"]),
        "duration_ms": int((time.monotonic() - started) * 1000),
    }


def completion_with_model_retry(base_url, prompt, chunk_size, settings, model_name):
    try:
        return generate_completion_chunk(
            base_url,
            prompt,
            chunk_size,
            settings,
            model_name=model_name,
        ), model_name
    except Exception as req_err:
        req_msg = str(req_err)
        if "model name is missing" in req_msg.lower():
            refreshed_model = fetch_model_name(base_url)
            if refreshed_model:
                return generate_completion_chunk(
                    base_url,
                    prompt,
                    chunk_size,
                    settings,
                    model_name=refreshed_model,
                ), refreshed_model
        raise


def generate_single_token_step(base_url, prompt, settings, model_name, prompt_hash, token_index, history_text, vector_provider, started):
    response, model_name = completion_with_model_retry(base_url, prompt, 1, settings, model_name)
    content = extract_content(response)
    probs = extract_probs(response)
    stop_info = extract_stop_info(response)
    t_ms = int((time.monotonic() - started) * 1000)

    token = None
    if isinstance(probs, list) and probs:
        entry = probs[0]
        if isinstance(entry, dict):
            token = token_record_from_entry(entry, settings.get("top_n", DEFAULTS["top_n"]), prompt_hash, token_index, t_ms)
    if token is None:
        fallback_tokens = tokenize_fallback(content)
        if fallback_tokens:
            token = token_record_fallback(
                fallback_tokens[0],
                settings.get("top_n", DEFAULTS["top_n"]),
                prompt_hash,
                token_index,
                t_ms,
            )

    if token is not None:
        token_text = token.get("text", "")
        token["embedding"] = vector_provider.vector_for(token_index, token.get("chosen_token_id"), token_text, history_text)
    return token, stop_info, model_name


def add_token_local(run, token):
    run["tokens"].append(token)
    run["meta"]["providers"]["topN"] = token.get("topn_provider", run["meta"]["providers"].get("topN"))
    run["meta"]["providers"]["embedding"] = run["meta"]["providers"].get("embedding")


def joined_run_text(run):
    return "".join((token.get("text") or "") for token in (run.get("tokens") or []))


def token_char_spans(tokens):
    spans = []
    pos = 0
    for token in tokens or []:
        text = token.get("text") or ""
        start = pos
        end = start + len(text)
        spans.append({"start": start, "end": end})
        pos = end
    return spans


def strip_claim_prefix(text):
    return re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", str(text or "").strip())


def sentence_claim_spans(text):
    spans = []
    start = 0
    i = 0
    closers = "\"')]}”’"
    while i < len(text):
        ch = text[i]
        if ch in ".!?。！？":
            end = i + 1
            while end < len(text) and text[end] in closers:
                end += 1
            next_char = text[end] if end < len(text) else ""
            if not next_char or next_char.isspace():
                spans.append({"start": start, "end": end, "complete": True})
                while end < len(text) and text[end].isspace():
                    end += 1
                start = end
                i = end
                continue
        i += 1
    if start < len(text):
        spans.append({"start": start, "end": len(text), "complete": False})
    return spans


def extract_claim_spans(text):
    text = str(text or "")
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    spans = []

    if len(nonempty_lines) >= 2:
        offset = 0
        for raw_line in text.splitlines(True):
            stripped = raw_line.strip()
            if stripped:
                local_start = raw_line.find(stripped)
                start = offset + max(0, local_start)
                end = start + len(stripped)
                spans.append(
                    {
                        "start": start,
                        "end": end,
                        "complete": raw_line.endswith("\n") or bool(re.search(r"[.!?。！？][\"')\]}”’]*\s*$", stripped)),
                    }
                )
            offset += len(raw_line)
    else:
        spans = sentence_claim_spans(text)

    out = []
    for span in spans:
        raw_text = text[span["start"] : span["end"]]
        trimmed = raw_text.strip()
        if not trimmed:
            continue
        leading_trim = len(raw_text) - len(raw_text.lstrip())
        adjusted_start = span["start"] + leading_trim
        adjusted_end = adjusted_start + len(trimmed)
        out.append(
            {
                "start": adjusted_start,
                "end": adjusted_end,
                "text": strip_claim_prefix(trimmed),
                "complete": bool(span.get("complete")),
            }
        )
    return out


def claim_is_checkworthy(text):
    claim = strip_claim_prefix(text)
    normalized = re.sub(r"\s+", " ", claim).strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    banned = (
        "answer directly",
        "the prompt",
        "format:",
        "constraint:",
        "advanced logic",
        "conversational filler",
        "use open in snake scope",
        "this response",
        "the user asked",
    )
    if any(pat in lowered for pat in banned):
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", normalized)
    if len(words) < 4 and not re.search(r"\d|\"|“|”|'", normalized):
        return False
    has_signal = (
        bool(re.search(r"\d", normalized))
        or bool(re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", normalized))
        or bool(re.search(r"\b(is|are|was|were|has|have|had|includes|published|located|founded|wrote|won)\b", lowered))
        or bool(re.search(r"\"|“|”|ISBN|ISO|case\b|v\.", normalized))
    )
    return has_signal


def map_claim_tokens(tokens, claim_span):
    spans = token_char_spans(tokens)
    token_start = None
    token_end = None
    for idx, span in enumerate(spans):
        if span["end"] <= claim_span["start"]:
            continue
        if span["start"] >= claim_span["end"]:
            break
        if token_start is None:
            token_start = idx
        token_end = idx
    return token_start, token_end


def claim_for_token_index(run, focus_token_index, require_complete=True):
    tokens = run.get("tokens") or []
    if not tokens:
        return None
    focus_token_index = max(0, min(int(focus_token_index), len(tokens) - 1))
    text = joined_run_text(run)
    if not text:
        return None
    char_spans = token_char_spans(tokens)
    focus_char = max(0, char_spans[focus_token_index]["end"] - 1)
    candidates = []
    for span in extract_claim_spans(text):
        if require_complete and not span["complete"]:
            continue
        if span["start"] <= focus_char:
            candidates.append(span)
        elif span["start"] > focus_char:
            break
    if not candidates:
        return None
    best = None
    for span in reversed(candidates):
        if span["start"] <= focus_char < span["end"] and claim_is_checkworthy(span["text"]):
            best = span
            break
    if best is None:
        for span in reversed(candidates):
            if claim_is_checkworthy(span["text"]):
                best = span
                break
    if best is None:
        best = candidates[-1]

    token_start, token_end = map_claim_tokens(tokens, best)
    claim = copy.deepcopy(best)
    claim["token_start"] = token_start
    claim["token_end"] = token_end
    claim["checkworthy"] = claim_is_checkworthy(claim["text"])
    return claim


def probe_slot_for_settings(settings, offset):
    base = sanitize_int((settings or {}).get("id_slot", 0), 0, minimum=0, maximum=8)
    return min(8, max(0, base + offset))


def completion_text_with_retry(base_url, prompt, max_tokens, settings, model_name=None):
    response, used_model = completion_with_model_retry(base_url, prompt, max_tokens, settings, model_name)
    return extract_content(response).strip(), used_model


def extract_json_object(text):
    text = str(text or "")
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def normalize_probe_answer(text):
    answer = re.sub(r"\s+", " ", str(text or "")).strip()
    if answer:
        answer = answer.splitlines()[0].strip()
    answer = re.sub(r"^(answer|response)\s*:\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip(" \"'`.,;:()[]{}")


def heuristic_claim_probe_plan(claim, max_facts):
    claim = strip_claim_prefix(claim)
    facts = []
    lowered = claim.lower()

    isbn_match = re.search(r"\b(?:ISBN(?:-1[03])?\s*[:#]?\s*)?([0-9Xx][0-9Xx\-]{8,20})\b", claim)
    if isbn_match:
        facts.append({"fact": isbn_match.group(1), "kind": "number", "question": "What is the ISBN?"})

    year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b", claim)
    if year_match:
        facts.append({"fact": year_match.group(1), "kind": "date", "question": "What year is mentioned?"})

    label_match = re.match(r"^(Publisher|Author|Language|Publication Year|Quote|Direct Quotation)\s*:\s*(.+)$", claim, flags=re.IGNORECASE)
    if label_match:
        label = label_match.group(1).strip().lower()
        value = label_match.group(2).strip().strip('"')
        question_map = {
            "publisher": "Who is the publisher?",
            "author": "Who is the author?",
            "language": "What is the language?",
            "publication year": "What is the publication year?",
            "quote": "What is the quoted line?",
            "direct quotation": "What is the quoted line?",
        }
        facts.append(
            {
                "fact": value,
                "kind": "quote" if "quote" in label else "other",
                "question": question_map.get(label, f"What is the {label}?"),
            }
        )

    quote_match = re.search(r"[\"“](.+?)[\"”]", claim)
    if quote_match and len(quote_match.group(1)) >= 12:
        facts.append({"fact": quote_match.group(1).strip(), "kind": "quote", "question": "What is the quoted line?"})

    title_match = re.search(r"(?:monograph|book|paper|case|standard)\s+[\"“]([^\"”]+)[\"”]", lowered)
    if title_match:
        facts.append({"fact": title_match.group(1).strip(), "kind": "title", "question": "What is the title mentioned?"})

    person_match = re.search(r"\b(?:Prof\.|Professor|Dr\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", claim)
    if person_match and all(item["fact"] != person_match.group(1).strip() for item in facts):
        facts.append({"fact": person_match.group(1).strip(), "kind": "person", "question": "Which person is named in the claim?"})

    deduped = []
    seen = set()
    for item in facts:
        key = (item["fact"].strip().lower(), item["question"].strip().lower())
        if not item["fact"] or not item["question"] or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_facts:
            break

    return {
        "checkworthy": claim_is_checkworthy(claim) and bool(deduped),
        "standalone_claim": claim,
        "facts": deduped,
        "raw_response": "",
        "model": None,
    }


def build_claim_probe_plan(base_url, model_name, settings, claim, context_text, policy, progress_cb=None, progress_context=None):
    context = dict(progress_context or {})

    def emit(stage, **extra):
        if progress_cb is None:
            return
        payload = dict(context)
        payload.update(extra)
        payload["stage"] = stage
        progress_cb(payload)

    prompt = (
        "You are preparing factual probes for a generated claim.\n"
        "Return JSON only with this schema:\n"
        '{"checkworthy": true, "standalone_claim": "...", "facts": [{"fact": "...", "kind": "person|org|date|number|title|location|quote|other", "question": "..."}]}\n'
        "Rules:\n"
        "- Keep only explicit factual details already present in the claim.\n"
        "- If the claim depends on pronouns, rewrite it as a standalone claim using only the provided context.\n"
        "- If the claim is mostly stylistic, subjective, or instruction-following, set checkworthy to false and facts to [].\n"
        "- Do not use markdown fences or commentary.\n"
        f"- Return at most {policy['harness_max_facts']} facts.\n"
        "Context:\n"
        f"{context_text or '[none]'}\n\n"
        "Claim:\n"
        f"{claim}\n"
    )
    probe_settings = copy.deepcopy(settings)
    probe_settings["temperature"] = 0.15
    probe_settings["top_p"] = 0.9
    probe_settings["seed"] = stable_seed(prompt, probe_settings.get("seed", 0), "claim-plan")
    probe_settings["id_slot"] = probe_slot_for_settings(settings, 1)
    emit("plan_started", claim_text=claim)
    raw_text, used_model = completion_text_with_retry(
        base_url,
        prompt,
        policy["harness_json_tokens"],
        probe_settings,
        model_name=model_name,
    )
    parsed = extract_json_object(raw_text) or {}
    facts = parsed.get("facts") if isinstance(parsed.get("facts"), list) else []
    clean_facts = []
    for item in facts[: policy["harness_max_facts"]]:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        question = str(item.get("question") or "").strip()
        if not fact or not question:
            continue
        clean_facts.append(
            {
                "fact": fact,
                "kind": str(item.get("kind") or "other").strip().lower(),
                "question": question,
            }
        )
    standalone_claim = str(parsed.get("standalone_claim") or claim).strip() or claim
    checkworthy = bool(parsed.get("checkworthy")) if "checkworthy" in parsed else claim_is_checkworthy(standalone_claim)
    plan = {
        "checkworthy": checkworthy and bool(clean_facts),
        "standalone_claim": standalone_claim,
        "facts": clean_facts,
        "raw_response": raw_text,
        "model": used_model,
    }
    emit(
        "plan_ready",
        claim_text=standalone_claim,
        checkworthy=plan["checkworthy"],
        facts_count=len(clean_facts),
    )
    if not plan["facts"]:
        fallback = heuristic_claim_probe_plan(standalone_claim, policy["harness_max_facts"])
        if fallback["facts"]:
            fallback["raw_response"] = raw_text
            fallback["model"] = used_model
            emit(
                "plan_fallback",
                claim_text=fallback["standalone_claim"],
                checkworthy=fallback["checkworthy"],
                facts_count=len(fallback["facts"]),
            )
            return fallback
    return plan


def run_probe_answers(base_url, model_name, settings, question, policy, probe_index, progress_cb=None, progress_context=None):
    answers = []
    context = dict(progress_context or {})

    def emit(stage, **extra):
        if progress_cb is None:
            return
        payload = dict(context)
        payload.update(extra)
        payload["stage"] = stage
        progress_cb(payload)

    for sample_index in range(policy["harness_samples"]):
        prompt = (
            "Answer the factual question with a short phrase only.\n"
            'If the answer is unknown or unstable, answer exactly: unknown\n'
            f"Question: {question}\n"
            "Answer:"
        )
        probe_settings = copy.deepcopy(settings)
        probe_settings["temperature"] = max(0.75, float(settings.get("temperature") or 0.7))
        probe_settings["top_p"] = max(0.9, float(settings.get("top_p") or 0.95))
        probe_settings["seed"] = stable_seed(settings.get("seed", 0), question, probe_index, sample_index)
        probe_settings["id_slot"] = probe_slot_for_settings(settings, 1)
        emit("probe_sample_started", question=question, probe_index=probe_index, sample_index=sample_index + 1)
        raw_text, model_name = completion_text_with_retry(
            base_url,
            prompt,
            policy["harness_probe_tokens"],
            probe_settings,
            model_name=model_name,
        )
        answer_text = normalize_probe_answer(raw_text) or "unknown"
        answers.append({"raw": raw_text, "text": answer_text})
        emit(
            "probe_sample_completed",
            question=question,
            probe_index=probe_index,
            sample_index=sample_index + 1,
            answer_text=answer_text,
        )
    return answers, model_name


def judge_probe_fact(base_url, model_name, settings, claim, fact_detail, question, answers, policy, fact_index, progress_cb=None, progress_context=None):
    context = dict(progress_context or {})

    def emit(stage, **extra):
        if progress_cb is None:
            return
        payload = dict(context)
        payload.update(extra)
        payload["stage"] = stage
        progress_cb(payload)

    prompt = (
        "You are checking whether a factual detail from a generated claim is stable.\n"
        'Return JSON only with this schema: {"verdict":"supported|conflict|uncertain","agreement":"high|mixed|low","canonical_answer":"...","reason":"..."}\n'
        "Rules:\n"
        "- supported: the answers agree and support the original fact detail.\n"
        "- conflict: the answers disagree with the fact detail or clearly contradict each other.\n"
        "- uncertain: the answers are vague, say unknown, or do not reliably support the fact detail.\n"
        "- Keep reason short.\n\n"
        f"Standalone claim: {claim}\n"
        f"Fact detail: {fact_detail}\n"
        f"Probe question: {question}\n"
        f"Answer 1: {answers[0]['text'] if len(answers) > 0 else 'unknown'}\n"
        f"Answer 2: {answers[1]['text'] if len(answers) > 1 else 'unknown'}\n"
    )
    probe_settings = copy.deepcopy(settings)
    probe_settings["temperature"] = 0.1
    probe_settings["top_p"] = 0.9
    probe_settings["seed"] = stable_seed(settings.get("seed", 0), fact_detail, fact_index, "judge")
    probe_settings["id_slot"] = probe_slot_for_settings(settings, 1)
    emit("judge_started", fact_detail=fact_detail, fact_index=fact_index, question=question)
    raw_text, used_model = completion_text_with_retry(
        base_url,
        prompt,
        policy["harness_json_tokens"],
        probe_settings,
        model_name=model_name,
    )
    parsed = extract_json_object(raw_text) or {}
    verdict = str(parsed.get("verdict") or "uncertain").strip().lower()
    if verdict not in ("supported", "conflict", "uncertain"):
        verdict = "uncertain"
    agreement = str(parsed.get("agreement") or "mixed").strip().lower()
    if agreement not in ("high", "mixed", "low"):
        agreement = "mixed"
    canonical = normalize_probe_answer(parsed.get("canonical_answer") or "")
    reason = str(parsed.get("reason") or "").strip()

    if verdict == "supported":
        fact_risk = {"high": 0.14, "mixed": 0.24, "low": 0.32}[agreement]
    elif verdict == "conflict":
        fact_risk = {"high": 0.88, "mixed": 0.84, "low": 0.8}[agreement]
    else:
        if all((ans.get("text") or "").lower() == "unknown" for ans in answers):
            fact_risk = 0.52
        else:
            fact_risk = {"high": 0.46, "mixed": 0.58, "low": 0.68}[agreement]

    result = {
        "verdict": verdict,
        "agreement": agreement,
        "canonical_answer": canonical or None,
        "reason": reason,
        "fact_risk": fact_risk,
        "raw_response": raw_text,
        "model": used_model,
    }
    emit(
        "judge_completed",
        fact_detail=fact_detail,
        fact_index=fact_index,
        question=question,
        verdict=verdict,
        agreement=agreement,
        fact_risk=fact_risk,
    )
    return result


def run_claim_harness(run, focus_token_index, policy, progress_cb=None, progress_context=None):
    tokens = run.get("tokens") or []
    if not tokens:
        return {"status": "no_tokens"}

    context = dict(progress_context or {})

    def emit(stage, **extra):
        if progress_cb is None:
            return
        payload = dict(context)
        payload.update(extra)
        payload["stage"] = stage
        progress_cb(payload)

    focus_token_index = max(0, min(int(focus_token_index), len(tokens) - 1))
    cache = run.setdefault("analysis", {}).setdefault("claim_harness_cache", {})
    cache_key = str(focus_token_index)
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        emit("cache_hit", focus_token_index=focus_token_index, claim_risk=cached.get("claim_risk"), label=cached.get("label"))
        return copy.deepcopy(cached)

    claim_span = claim_for_token_index(run, focus_token_index, require_complete=True)
    if claim_span is None:
        result = {"status": "no_complete_claim", "focus_token_index": focus_token_index}
        cache[cache_key] = copy.deepcopy(result)
        return result

    base_url = run.get("meta", {}).get("base_url", DEFAULTS["base_url"])
    settings = copy.deepcopy(run.get("meta", {}).get("generation_settings") or {})
    model_name = run.get("meta", {}).get("model")
    text = joined_run_text(run)
    context_start = max(0, claim_span["start"] - 280)
    context_text = text[context_start : claim_span["start"]].strip()
    emit("claim_selected", focus_token_index=focus_token_index, claim_text=claim_span.get("text"), claim_token_start=claim_span.get("token_start"))

    if not claim_span.get("checkworthy"):
        result = {
            "status": "not_checkworthy",
            "focus_token_index": focus_token_index,
            "claim": claim_span,
        }
        cache[cache_key] = copy.deepcopy(result)
        return result

    plan = build_claim_probe_plan(
        base_url,
        model_name,
        settings,
        claim_span["text"],
        context_text,
        policy,
        progress_cb=progress_cb,
        progress_context={**context, "focus_token_index": focus_token_index},
    )
    if not plan.get("checkworthy") or not plan.get("facts"):
        result = {
            "status": "not_checkworthy",
            "focus_token_index": focus_token_index,
            "claim": claim_span,
            "plan": plan,
        }
        cache[cache_key] = copy.deepcopy(result)
        return result

    facts = []
    current_model = plan.get("model") or model_name
    for fact_index, item in enumerate(plan["facts"]):
        answers, current_model = run_probe_answers(
            base_url,
            current_model,
            settings,
            item["question"],
            policy,
            fact_index,
            progress_cb=progress_cb,
            progress_context={**context, "focus_token_index": focus_token_index, "fact_detail": item["fact"]},
        )
        judgement = judge_probe_fact(
            base_url,
            current_model,
            settings,
            plan["standalone_claim"],
            item["fact"],
            item["question"],
            answers,
            policy,
            fact_index,
            progress_cb=progress_cb,
            progress_context={**context, "focus_token_index": focus_token_index, "fact_detail": item["fact"]},
        )
        current_model = judgement.get("model") or current_model
        facts.append(
            {
                "fact": item["fact"],
                "kind": item["kind"],
                "question": item["question"],
                "answers": answers,
                "judgement": judgement,
            }
        )

    fact_risks = [float(f["judgement"]["fact_risk"]) for f in facts if f.get("judgement")]
    max_risk = max(fact_risks) if fact_risks else None
    mean_risk = (sum(fact_risks) / len(fact_risks)) if fact_risks else None
    claim_risk = None
    if fact_risks:
        claim_risk = max(0.0, min(1.0, 0.65 * max_risk + 0.35 * mean_risk))

    supported = sum(1 for f in facts if f["judgement"]["verdict"] == "supported")
    conflicted = sum(1 for f in facts if f["judgement"]["verdict"] == "conflict")
    uncertain = sum(1 for f in facts if f["judgement"]["verdict"] == "uncertain")
    if claim_risk is None:
        label = "no-evidence"
    elif claim_risk >= 0.72:
        label = "likely-unsupported"
    elif claim_risk >= 0.5:
        label = "unstable"
    else:
        label = "mostly-supported"

    result = {
        "status": "ok",
        "focus_token_index": focus_token_index,
        "claim": claim_span,
        "standalone_claim": plan["standalone_claim"],
        "plan": plan,
        "facts": facts,
        "supported_facts": supported,
        "conflicted_facts": conflicted,
        "uncertain_facts": uncertain,
        "claim_risk": claim_risk,
        "label": label,
    }
    emit(
        "harness_completed",
        focus_token_index=focus_token_index,
        claim_text=plan["standalone_claim"],
        claim_risk=claim_risk,
        label=label,
        supported_facts=supported,
        conflicted_facts=conflicted,
        uncertain_facts=uncertain,
    )
    cache[cache_key] = copy.deepcopy(result)
    return result


def infer_focus_object(prompt, answer_text=""):
    prompt_text = str(prompt or "")
    answer_text = str(answer_text or "")
    haystack = f"{prompt_text}\n{answer_text[:400]}"
    lowered = prompt_text.lower()
    object_type = "work"
    for label in ("monograph", "book", "paper", "article", "case", "standard", "report", "novel"):
        if label in lowered:
            object_type = label
            break

    title = None
    title_match = re.search(r"(?:monograph|book|paper|article|case|standard|report|novel)\s+[\"“]([^\"”]+)[\"”]", prompt_text, flags=re.IGNORECASE)
    if not title_match:
        title_match = re.search(r"[\"“]([^\"”]{6,160})[\"”]", prompt_text)
    if title_match:
        title = title_match.group(1).strip()

    author = None
    author_match = re.search(r"\bby\s+((?:Prof\.|Professor|Dr\.)?\s*[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)+)", prompt_text)
    if author_match:
        author = re.sub(r"\s+", " ", author_match.group(1).strip())

    year = None
    year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b", prompt_text)
    if year_match:
        year = year_match.group(1)

    requested_fields = []
    for label in ("publisher", "isbn", "quote", "quotation", "direct quotation", "holding", "scope", "requirement"):
        if label in lowered:
            requested_fields.append(label)

    summary_bits = [object_type]
    if title:
        summary_bits.append(f'"{title}"')
    if author:
        summary_bits.append(f"by {author}")
    if year:
        summary_bits.append(f"({year})")

    return {
        "object_type": object_type,
        "title": title,
        "author": author,
        "year": year,
        "requested_fields": requested_fields,
        "summary": " ".join(summary_bits).strip(),
        "raw_prompt": prompt_text,
        "answer_excerpt": answer_text[:500],
        "has_focus_object": bool(title or author or year or object_type != "work"),
        "object_key": "||".join(str(part or "") for part in (object_type, title, author, year)).strip("|"),
    }


def normalize_object_verdict(value):
    verdict = str(value or "uncertain").strip().lower().replace(" ", "_")
    if verdict in ("likely_invented", "invented", "fabricated", "unsupported"):
        return "likely_invented"
    if verdict in ("supported", "verified"):
        return "supported"
    return "uncertain"


def normalize_confidence(value):
    confidence = str(value or "mixed").strip().lower()
    if confidence not in ("high", "mixed", "low"):
        return "mixed"
    return confidence


def object_sample_risk(verdict, confidence):
    if verdict == "supported":
        return {"high": 0.18, "mixed": 0.28, "low": 0.36}[confidence]
    if verdict == "likely_invented":
        return {"high": 0.82, "mixed": 0.86, "low": 0.9}[confidence]
    return {"high": 0.52, "mixed": 0.62, "low": 0.7}[confidence]


def summarize_object_reason(samples):
    reasons = []
    for sample in samples:
        reason = str(sample.get("reason") or "").strip()
        if not reason:
            continue
        if reason not in reasons:
            reasons.append(reason)
    return reasons[0] if reasons else ""


def parse_object_harness_response(raw_text):
    parsed = extract_json_object(raw_text) or {}
    lowered = re.sub(r"\s+", " ", str(raw_text or "").strip().lower())

    verdict = normalize_object_verdict(parsed.get("verdict"))
    if "verdict" not in parsed:
        verdict_match = re.search(r'verdict"?\s*[:=]\s*"?(supported|uncertain|likely[_ -]?invented|invented|unsupported)', lowered)
        if verdict_match:
            verdict = normalize_object_verdict(verdict_match.group(1))
        elif any(term in lowered for term in ("hallucination", "hallucinated", "fabricated", "invented", "not real", "fictional")):
            verdict = "likely_invented"
        elif any(term in lowered for term in ("cannot verify", "cannot confirm", "no verifiable", "no reliable", "unsupported", "should be corrected")):
            verdict = "likely_invented"

    confidence = normalize_confidence(parsed.get("confidence"))
    if "confidence" not in parsed:
        confidence_match = re.search(r'confidence"?\s*[:=]\s*"?(high|mixed|low)', lowered)
        if confidence_match:
            confidence = normalize_confidence(confidence_match.group(1))
        elif verdict == "likely_invented":
            confidence = "mixed"

    abstain_recommended = bool(parsed.get("abstain_recommended"))
    if "abstain_recommended" not in parsed:
        abstain_match = re.search(r'abstain(?:_recommended)?"?\s*[:=]\s*(true|false|yes|no)', lowered)
        if abstain_match:
            abstain_recommended = abstain_match.group(1) in ("true", "yes")
        elif any(term in lowered for term in ("cannot verify", "cannot confirm", "should abstain", "avoid giving", "do not claim", "should be corrected")):
            abstain_recommended = True

    unstable_fields = parsed.get("unstable_fields") if isinstance(parsed.get("unstable_fields"), list) else []
    unstable_fields = [str(item).strip().lower() for item in unstable_fields if str(item).strip()]
    if not unstable_fields:
        for label in ("title", "author", "year", "publisher", "isbn", "quotation", "quote", "holding"):
            if label in lowered and ("unstable" in lowered or "hallucinat" in lowered or "invent" in lowered or "not valid" in lowered):
                unstable_fields.append(label)

    reason = str(parsed.get("reason") or "").strip()
    if not reason:
        for sentence in re.split(r"(?<=[.!?])\s+", str(raw_text or "").strip()):
            lower_sentence = sentence.lower()
            if any(term in lower_sentence for term in ("hallucination", "hallucinated", "invented", "cannot verify", "not valid", "should be corrected", "unsupported")):
                reason = sentence.strip()
                break

    return {
        "verdict": verdict,
        "confidence": confidence,
        "abstain_recommended": abstain_recommended,
        "canonical_title": str(parsed.get("canonical_title") or "").strip() or None,
        "canonical_author": str(parsed.get("canonical_author") or "").strip() or None,
        "canonical_year": str(parsed.get("canonical_year") or "").strip() or None,
        "unstable_fields": unstable_fields,
        "reason": reason,
    }


def run_object_harness(run, policy, progress_cb=None, progress_context=None):
    tokens = run.get("tokens") or []
    if not tokens:
        return {"status": "no_tokens"}

    context = dict(progress_context or {})

    def emit(stage, **extra):
        if progress_cb is None:
            return
        payload = dict(context)
        payload.update(extra)
        payload["stage"] = stage
        progress_cb(payload)

    prompt = ((run.get("meta") or {}).get("prompt")) or ""
    answer_text = joined_run_text(run)
    profile = infer_focus_object(prompt, answer_text)
    if not profile.get("has_focus_object"):
        return {"status": "no_focus_object", "profile": profile}

    cache = run.setdefault("analysis", {}).setdefault("object_harness_cache", {})
    cache_key = short_hash(f"{profile.get('object_key')}::{answer_text[:900]}")
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        emit(
            "cache_hit",
            profile=profile,
            object_risk=cached.get("object_risk"),
            label=cached.get("label"),
            mode_recommendation=cached.get("mode_recommendation"),
        )
        return copy.deepcopy(cached)

    base_url = run.get("meta", {}).get("base_url", DEFAULTS["base_url"])
    settings = copy.deepcopy(run.get("meta", {}).get("generation_settings") or {})
    model_name = run.get("meta", {}).get("model")
    prompt_text = prompt.strip() or "[none]"
    answer_excerpt = answer_text[:1200] or "[empty]"
    requested = ", ".join(profile.get("requested_fields") or []) or "none"

    emit("object_selected", profile=profile, object_summary=profile.get("summary"))

    samples = []
    current_model = model_name
    for sample_index in range(policy["harness_samples"]):
        sample_settings = copy.deepcopy(settings)
        sample_settings["temperature"] = 0.12
        sample_settings["top_p"] = 0.9
        sample_settings["seed"] = stable_seed(settings.get("seed", 0), profile.get("object_key"), sample_index, "object-harness")
        sample_settings["id_slot"] = probe_slot_for_settings(settings, 2)
        emit(
            "object_probe_started",
            sample_index=sample_index,
            profile=profile,
            object_summary=profile.get("summary"),
        )
        harness_prompt = (
            "You are auditing whether a referenced source or work should be treated as verified without external lookup.\n"
            "Return JSON only with this schema:\n"
            '{"verdict":"supported|uncertain|likely_invented","confidence":"high|mixed|low","abstain_recommended":true,'
            '"canonical_title":"...","canonical_author":"...","canonical_year":"...","unstable_fields":["title"],"reason":"..."}\n'
            "Rules:\n"
            "- Prefer likely_invented when multiple key bibliographic or source-identifying details appear unstable or unsupported.\n"
            "- Prefer uncertain when the object might exist but should not be asserted confidently from stable knowledge alone.\n"
            "- Set abstain_recommended=true when a cautious assistant should avoid giving concrete specifics such as ISBNs, publishers, dates, quotations, or holdings.\n"
            "- Do not invent catalog entries, citations, or alternate concrete specifics.\n\n"
            f"User request:\n{prompt_text}\n\n"
            f"Current answer draft:\n{answer_excerpt}\n\n"
            f"Focus object summary: {profile.get('summary')}\n"
            f"Requested fields: {requested}\n"
        )
        raw_text, current_model = completion_text_with_retry(
            base_url,
            harness_prompt,
            policy["harness_json_tokens"],
            sample_settings,
            model_name=current_model,
        )
        parsed = parse_object_harness_response(raw_text)
        verdict = parsed["verdict"]
        confidence = parsed["confidence"]
        unstable_fields = parsed["unstable_fields"]
        sample = {
            "verdict": verdict,
            "confidence": confidence,
            "abstain_recommended": bool(parsed.get("abstain_recommended")),
            "canonical_title": parsed.get("canonical_title"),
            "canonical_author": parsed.get("canonical_author"),
            "canonical_year": parsed.get("canonical_year"),
            "unstable_fields": unstable_fields,
            "reason": str(parsed.get("reason") or "").strip(),
            "risk": object_sample_risk(verdict, confidence),
            "raw_response": raw_text,
            "model": current_model,
        }
        if profile.get("requested_fields") and verdict != "supported":
            sample["risk"] = min(1.0, sample["risk"] + 0.03)
        if len(unstable_fields) >= 2 and verdict != "supported":
            sample["risk"] = min(1.0, sample["risk"] + 0.04)
        samples.append(sample)
        emit(
            "object_probe_completed",
            sample_index=sample_index,
            profile=profile,
            object_summary=profile.get("summary"),
            verdict=sample["verdict"],
            confidence=sample["confidence"],
            abstain_recommended=sample["abstain_recommended"],
            object_risk=sample["risk"],
            reason=sample["reason"],
        )

    sample_risks = [float(sample["risk"]) for sample in samples]
    mean_risk = sum(sample_risks) / len(sample_risks)
    max_risk = max(sample_risks)
    object_risk = max(0.0, min(1.0, 0.65 * max_risk + 0.35 * mean_risk))
    abstain_votes = sum(1 for sample in samples if sample.get("abstain_recommended"))
    likely_invented_votes = sum(1 for sample in samples if sample.get("verdict") == "likely_invented")
    supported_votes = sum(1 for sample in samples if sample.get("verdict") == "supported")
    if object_risk >= 0.72 or likely_invented_votes >= max(1, math.ceil(len(samples) / 2)):
        label = "likely-invented"
    elif object_risk >= 0.5:
        label = "unstable"
    else:
        label = "mostly-supported"
    abstain_recommended = abstain_votes >= max(1, math.ceil(len(samples) / 2)) or label == "likely-invented"
    mode_recommendation = "global_rewrite" if abstain_recommended or object_risk >= policy["object_trigger_threshold"] else "continue"

    result = {
        "status": "ok",
        "profile": profile,
        "samples": samples,
        "object_risk": object_risk,
        "label": label,
        "abstain_recommended": abstain_recommended,
        "mode_recommendation": mode_recommendation,
        "reason_summary": summarize_object_reason(samples),
        "supported_votes": supported_votes,
        "uncertain_votes": sum(1 for sample in samples if sample.get("verdict") == "uncertain"),
        "likely_invented_votes": likely_invented_votes,
    }
    emit(
        "object_harness_completed",
        profile=profile,
        object_summary=profile.get("summary"),
        object_risk=object_risk,
        label=label,
        abstain_recommended=abstain_recommended,
        mode_recommendation=mode_recommendation,
        reason=result["reason_summary"],
    )
    cache[cache_key] = copy.deepcopy(result)
    return result


def peak_risk_token_index(run):
    best_idx = 0
    best_risk = -1.0
    for idx, token in enumerate(run.get("tokens") or []):
        risk = token.get("decoder_risk")
        try:
            risk = float(risk)
        except (TypeError, ValueError):
            continue
        if risk > best_risk:
            best_risk = risk
            best_idx = idx
    return best_idx


def live_token_snapshot(token):
    if not isinstance(token, dict):
        return {}
    snapshot = {}
    for key in (
        "index",
        "t",
        "text",
        "chosen_token_id",
        "chosen_token_text",
        "logprob",
        "prob",
        "entropy",
        "margin",
        "uncertainty",
        "prob_gap",
        "entropy_delta",
        "repetition_pressure",
        "decoder_risk",
        "velocity",
        "curvature",
        "topN",
        "topn_provider",
    ):
        if key in token:
            snapshot[key] = copy.deepcopy(token[key])
    return snapshot


def decode_candidate_until_claim(run, focus_token_index, max_new_tokens=None):
    started = time.monotonic()
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

    history_text = [t.get("text") or "" for t in run["tokens"]]
    if run["tokens"] and isinstance(run["tokens"][-1].get("embedding"), list):
        vector_provider.prev = run["tokens"][-1]["embedding"]

    token_index = len(run["tokens"])
    generated_text = ""
    remaining = max(0, settings.get("max_tokens", DEFAULTS["max_tokens"]) - token_index)
    if max_new_tokens is not None:
        remaining = min(remaining, max(0, int(max_new_tokens)))

    while remaining > 0:
        token, stop_info, model_name = generate_single_token_step(
            base_url,
            inference_prompt + generated_text,
            settings,
            model_name,
            prompt_hash,
            token_index,
            history_text,
            vector_provider,
            started,
        )
        if model_name and meta.get("model") != model_name:
            meta["model"] = model_name
        if token is None:
            break

        token_text = token.get("text", "")
        generated_text += token_text
        history_text.append(token_text)
        meta["providers"]["embedding"] = vector_provider.mode
        add_token_local(run, token)
        recompute_kinematics(run["tokens"])
        enrich_decoder_diagnostics(run["tokens"])
        run["analysis"]["regime_markers"] = detect_regime_markers(run["tokens"])
        run["analysis"]["decoder_alerts"] = detect_decoder_alerts(run["tokens"])
        token_index += 1
        remaining -= 1

        claim = claim_for_token_index(run, focus_token_index, require_complete=True)
        if claim is not None and claim.get("token_end") is not None and claim["token_end"] >= focus_token_index:
            break

        if stop_info.get("hard_stop"):
            break
        if stop_info.get("stop") and remaining <= 0:
            break

    finalize_completed_run(run, started, status="complete")
    return run


def decode_run_local(run, max_new_tokens=None, progress_cb=None, phase=None):
    started = time.monotonic()
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

    history_text = [t.get("text") or "" for t in run["tokens"]]
    if run["tokens"] and isinstance(run["tokens"][-1].get("embedding"), list):
        vector_provider.prev = run["tokens"][-1]["embedding"]

    token_index = len(run["tokens"])
    generated_text = ""
    remaining = max(0, settings.get("max_tokens", DEFAULTS["max_tokens"]) - token_index)
    if max_new_tokens is not None:
        remaining = min(remaining, max(0, int(max_new_tokens)))

    while remaining > 0:
        token, stop_info, model_name = generate_single_token_step(
            base_url,
            inference_prompt + generated_text,
            settings,
            model_name,
            prompt_hash,
            token_index,
            history_text,
            vector_provider,
            started,
        )
        if model_name and meta.get("model") != model_name:
            meta["model"] = model_name
        if token is None:
            break

        token_text = token.get("text", "")
        generated_text += token_text
        history_text.append(token_text)
        meta["providers"]["embedding"] = vector_provider.mode
        add_token_local(run, token)
        recompute_kinematics(run["tokens"])
        enrich_decoder_diagnostics(run["tokens"])
        run["analysis"]["regime_markers"] = detect_regime_markers(run["tokens"])
        run["analysis"]["decoder_alerts"] = detect_decoder_alerts(run["tokens"])
        if progress_cb is not None:
            max_decoder_risk = None
            for item in run["tokens"]:
                try:
                    value = float(item.get("decoder_risk"))
                except (TypeError, ValueError):
                    continue
                max_decoder_risk = value if max_decoder_risk is None else max(max_decoder_risk, value)
            progress_cb(
                {
                    "phase": phase,
                    "run_id": run["run_id"],
                    "token_index": token_index,
                    "token_text": token_text,
                    "decoder_risk": token.get("decoder_risk"),
                    "max_decoder_risk": max_decoder_risk,
                    "alert_count": len(run["analysis"]["decoder_alerts"]),
                    "token": live_token_snapshot(token),
                    "text": joined_run_text(run),
                    "token_count": len(run["tokens"]),
                }
            )
        token_index += 1
        remaining -= 1

        if stop_info.get("hard_stop"):
            break
        if stop_info.get("stop") and remaining <= 0:
            break

    finalize_completed_run(run, started, status="complete")
    return run


def make_forced_token(target, chosen_alt, top_n_limit):
    topn = target.get("topN") or []
    return {
        "index": target.get("index"),
        "t": target.get("t"),
        "text": chosen_alt.get("token_text", ""),
        "chosen_token_id": chosen_alt.get("token_id"),
        "chosen_token_text": chosen_alt.get("token_text", ""),
        "logprob": chosen_alt.get("logprob"),
        "prob": chosen_alt.get("prob"),
        "entropy": target.get("entropy"),
        "margin": target.get("margin"),
        "topN": copy.deepcopy(topn[: top_n_limit]),
        "topn_provider": target.get("topn_provider", "backend_logprobs"),
    }


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
            stop_info = extract_stop_info(response)

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
                    with RUNS_LOCK:
                        live_run = RUNS.get(run_id)
                        if live_run is not None:
                            live_run["meta"]["providers"]["embedding"] = vector_provider.mode

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
                    with RUNS_LOCK:
                        live_run = RUNS.get(run_id)
                        if live_run is not None:
                            live_run["meta"]["providers"]["embedding"] = vector_provider.mode

                    append_token(run_id, token)
                    token_index += 1
                    remaining -= 1
                    emitted_this_chunk += 1

            if emitted_this_chunk == 0:
                break
            if stop_info.get("hard_stop"):
                break
            if stop_info.get("stop"):
                # Some backends mark each n_predict boundary as stop=true.
                # Continue unless this is a hard stop or we received a short final chunk.
                if remaining <= 0:
                    break
                if stop_info.get("stopped_limit"):
                    continue
                if emitted_this_chunk >= chunk_size:
                    continue
                break

        with RUNS_LOCK:
            final_run = RUNS.get(run_id)
            if final_run is not None:
                recompute_kinematics(final_run["tokens"])
                enrich_decoder_diagnostics(final_run["tokens"])
                final_run["analysis"]["regime_markers"] = detect_regime_markers(final_run["tokens"])
                final_run["analysis"]["decoder_alerts"] = detect_decoder_alerts(final_run["tokens"])
                final_run["meta"]["status"] = "stopped" if stop_event.is_set() else "complete"
                final_run["meta"]["completed_at"] = utc_now_iso()

                entropies = [t.get("entropy") for t in final_run["tokens"] if t.get("entropy") is not None]
                velocities = [t.get("velocity") for t in final_run["tokens"] if t.get("velocity") is not None]
                risks = [t.get("decoder_risk") for t in final_run["tokens"] if t.get("decoder_risk") is not None]
                final_run["summary"] = {
                    "token_count": len(final_run["tokens"]),
                    "entropy_avg": (sum(entropies) / len(entropies)) if entropies else None,
                    "velocity_max": max(velocities) if velocities else None,
                    "decoder_risk_max": max(risks) if risks else None,
                    "decoder_alert_count": len(final_run["analysis"]["decoder_alerts"]),
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
    chosen_rank = None
    if alt_rank is not None:
        try:
            alt_rank = int(alt_rank)
        except (TypeError, ValueError):
            raise ValueError("alt_rank must be an integer")
        if alt_rank < 0 or alt_rank >= len(topn):
            raise ValueError("alt_rank out of range")
        chosen_alt = topn[alt_rank]
        chosen_rank = alt_rank
    else:
        requested_id = body.get("alt_token_id")
        requested_text = body.get("alt_token_text")
        for rank, item in enumerate(topn):
            if requested_id is not None and item.get("token_id") == requested_id:
                chosen_alt = item
                chosen_rank = rank
                break
            if requested_text is not None and item.get("token_text") == requested_text:
                chosen_alt = item
                chosen_rank = rank
                break
        if chosen_alt is None and topn:
            chosen_alt = topn[0]
            chosen_rank = 0

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
    raw_parent_history = parent_meta.get("branch_history")
    parent_history = copy.deepcopy(raw_parent_history) if isinstance(raw_parent_history, list) else []
    branch_timestamp = utc_now_iso()
    branch_mutation = {
        "parent_run_id": run_id,
        "fork_index": fork_index,
        "alt_rank": chosen_rank,
        "from_token_id": target.get("chosen_token_id"),
        "from_text": target.get("text") or target.get("chosen_token_text") or "",
        "to_token_id": chosen_alt.get("token_id"),
        "to_text": chosen_alt.get("token_text") or "",
        "logprob": chosen_alt.get("logprob"),
        "prob": chosen_alt.get("prob"),
        "timestamp": branch_timestamp,
    }
    branch_history = parent_history + [branch_mutation]

    forced_prefix = "".join(tok.get("text", "") for tok in prefill_tokens)
    inference_prompt = parent_prompt + forced_prefix

    branch_meta = {
        "parent_run_id": run_id,
        "fork_index": fork_index,
        "alt_rank": chosen_rank,
        "chosen_alt_token": {
            "token_id": chosen_alt.get("token_id"),
            "token_text": chosen_alt.get("token_text"),
            "logprob": chosen_alt.get("logprob"),
            "prob": chosen_alt.get("prob"),
        },
        "timestamp": branch_timestamp,
        "forcing_strategy": "append_prefix_fallback",
    }

    label = body.get("label") or f"Branch@{fork_index}"

    return create_run_object(
        prompt=parent_prompt,
        base_url=parent_meta.get("base_url", DEFAULTS["base_url"]),
        settings=sanitize_settings(parent_settings),
        label=label,
        branch_meta=branch_meta,
        branch_history=branch_history,
        prefill_tokens=prefill_tokens,
        inference_prompt=inference_prompt,
    )


def sanitize_live_policy(raw_policy):
    raw_policy = raw_policy if isinstance(raw_policy, dict) else {}
    correction_loops = sanitize_int(raw_policy.get("correction_loops", raw_policy.get("max_interventions", 2)), 2, minimum=0, maximum=8)
    return {
        "risk_threshold": sanitize_float(raw_policy.get("risk_threshold", 0.64), 0.64, minimum=0.0, maximum=1.0),
        "risk_persistence": sanitize_int(raw_policy.get("risk_persistence", 2), 2, minimum=1, maximum=8),
        "rollback_buffer": sanitize_int(raw_policy.get("rollback_buffer", 1), 1, minimum=0, maximum=6),
        "branch_candidates": sanitize_int(raw_policy.get("branch_candidates", 3), 3, minimum=1, maximum=5),
        "branch_lookahead": sanitize_int(raw_policy.get("branch_lookahead", 6), 6, minimum=1, maximum=24),
        "max_interventions": correction_loops,
        "correction_loops": correction_loops,
        "min_risk_drop": sanitize_float(raw_policy.get("min_risk_drop", 0.08), 0.08, minimum=0.0, maximum=1.0),
        "cooldown_tokens": sanitize_int(raw_policy.get("cooldown_tokens", 4), 4, minimum=0, maximum=24),
        "candidate_harness_topk": sanitize_int(raw_policy.get("candidate_harness_topk", 2), 2, minimum=1, maximum=3),
        "harness_min_drop": sanitize_float(raw_policy.get("harness_min_drop", 0.12), 0.12, minimum=0.0, maximum=1.0),
        "harness_trigger_threshold": sanitize_float(raw_policy.get("harness_trigger_threshold", 0.52), 0.52, minimum=0.0, maximum=1.0),
        "object_trigger_threshold": sanitize_float(raw_policy.get("object_trigger_threshold", 0.68), 0.68, minimum=0.0, maximum=1.0),
        "object_min_drop": sanitize_float(raw_policy.get("object_min_drop", 0.1), 0.1, minimum=0.0, maximum=1.0),
        "harness_max_facts": sanitize_int(raw_policy.get("harness_max_facts", 2), 2, minimum=1, maximum=4),
        "harness_samples": sanitize_int(raw_policy.get("harness_samples", 2), 2, minimum=2, maximum=3),
        "harness_probe_tokens": sanitize_int(raw_policy.get("harness_probe_tokens", 48), 48, minimum=8, maximum=160),
        "harness_json_tokens": sanitize_int(raw_policy.get("harness_json_tokens", 220), 220, minimum=64, maximum=400),
    }


def rollback_index_for_run(run, policy):
    tokens = run.get("tokens") or []
    persistence = policy["risk_persistence"]
    threshold = policy["risk_threshold"]
    if len(tokens) < persistence:
        return None, None

    tail = tokens[-persistence:]
    if not all((t.get("decoder_risk") or 0.0) >= threshold for t in tail):
        return None, None

    onset = len(tokens) - persistence
    alert_index = None
    for alert in reversed(run.get("analysis", {}).get("decoder_alerts") or []):
        idx = alert.get("index")
        if isinstance(idx, int) and idx <= len(tokens) - 1 and idx >= max(0, len(tokens) - 6):
            alert_index = idx
    rollback_index = max(0, min(onset, alert_index if alert_index is not None else onset) - policy["rollback_buffer"])
    current_window = tokens[rollback_index:]
    current_avg_risk = None
    if current_window:
        risks = [float(t.get("decoder_risk") or 0.0) for t in current_window]
        current_avg_risk = sum(risks) / len(risks)
    return rollback_index, current_avg_risk


def run_candidate_branch(parent_run, rollback_index, alt_rank, lookahead):
    parent_tokens = parent_run.get("tokens") or []
    if rollback_index < 0 or rollback_index >= len(parent_tokens):
        return None
    target = parent_tokens[rollback_index]
    topn = target.get("topN") or []
    if alt_rank < 0 or alt_rank >= len(topn):
        return None
    chosen_alt = topn[alt_rank]

    prefix_tokens = copy.deepcopy(parent_tokens[:rollback_index])
    forced_token = make_forced_token(
        target,
        chosen_alt,
        parent_run.get("meta", {}).get("generation_settings", {}).get("top_n", 5),
    )
    prefill_tokens = prefix_tokens + [forced_token]
    prompt = parent_run.get("meta", {}).get("prompt", "")
    inference_prompt = prompt + "".join(tok.get("text", "") for tok in prefill_tokens)

    settings = copy.deepcopy(parent_run.get("meta", {}).get("generation_settings") or {})
    settings = sanitize_settings(settings)
    settings["id_slot"] = probe_slot_for_settings(settings, 2 + alt_rank)
    run = create_run_object(
        prompt=prompt,
        base_url=parent_run.get("meta", {}).get("base_url", DEFAULTS["base_url"]),
        settings=settings,
        label=f"Candidate@{rollback_index}:{alt_rank}",
        prefill_tokens=prefill_tokens,
        inference_prompt=inference_prompt,
    )
    decode_candidate_until_claim(run, rollback_index, max_new_tokens=lookahead)

    candidate_window = run.get("tokens", [])[rollback_index:]
    risks = [float(t.get("decoder_risk") or 0.0) for t in candidate_window] if candidate_window else []
    return {
        "run": run,
        "alt_rank": alt_rank,
        "alt_token_text": chosen_alt.get("token_text"),
        "alt_token_id": chosen_alt.get("token_id"),
        "avg_risk": (sum(risks) / len(risks)) if risks else None,
        "max_risk": max(risks) if risks else None,
        "token_count": len(run.get("tokens") or []),
        "strategy": "prefix_replay",
    }


def format_harness_evidence_for_prompt(harness):
    if not isinstance(harness, dict):
        return "- No structured evidence available."
    facts = harness.get("facts") or []
    if not facts:
        return "- Claim looks unstable, but no fact-level probe details were captured."
    lines = []
    for item in facts:
        judgement = item.get("judgement") or {}
        answers = item.get("answers") or []
        lines.append(
            f"- Detail: {item.get('fact') or '-'} | question: {item.get('question') or '-'} | "
            f"answer1: {answers[0].get('text') if len(answers) > 0 else 'unknown'} | "
            f"answer2: {answers[1].get('text') if len(answers) > 1 else 'unknown'} | "
            f"verdict: {judgement.get('verdict') or 'uncertain'} | "
            f"agreement: {judgement.get('agreement') or 'mixed'} | "
            f"risk: {judgement.get('fact_risk') if judgement.get('fact_risk') is not None else '-'}"
        )
    return "\n".join(lines)


def has_abstention_signal(text):
    lowered = re.sub(r"\s+", " ", str(text or "").strip().lower())
    patterns = (
        "cannot verify",
        "can't verify",
        "cannot confirm",
        "can't confirm",
        "unable to verify",
        "unable to confirm",
        "not enough reliable information",
        "no reliable evidence",
        "cannot determine",
        "do not have enough information",
    )
    return any(pattern in lowered for pattern in patterns)


def format_object_harness_evidence_for_prompt(object_harness):
    if not isinstance(object_harness, dict):
        return "- No object-level evidence available."
    profile = object_harness.get("profile") or {}
    lines = [
        f"- Focus object: {profile.get('summary') or '[unknown]'}",
        f"- Object verdict: {object_harness.get('label') or object_harness.get('status') or 'unknown'}",
        f"- Object risk: {object_harness.get('object_risk') if object_harness.get('object_risk') is not None else '-'}",
        f"- Abstain recommended: {'yes' if object_harness.get('abstain_recommended') else 'no'}",
    ]
    for index, sample in enumerate(object_harness.get("samples") or []):
        lines.append(
            f"- Sample {index + 1}: verdict {sample.get('verdict') or 'uncertain'} | "
            f"confidence {sample.get('confidence') or 'mixed'} | "
            f"abstain {'yes' if sample.get('abstain_recommended') else 'no'} | "
            f"risk {sample.get('risk') if sample.get('risk') is not None else '-'} | "
            f"reason {sample.get('reason') or '-'}"
        )
    return "\n".join(lines)


def run_policy_rewrite_candidate(parent_run, current_harness, rollback_index, policy, progress_cb=None):
    claim = (current_harness or {}).get("claim") or {}
    claim_token_start = claim.get("token_start")
    if claim_token_start is None:
        claim_token_start = rollback_index
    claim_token_start = max(0, int(claim_token_start))

    parent_tokens = copy.deepcopy(parent_run.get("tokens") or [])
    safe_prefix_tokens = copy.deepcopy(parent_tokens[:claim_token_start])
    safe_prefix_text = "".join(tok.get("text") or "" for tok in safe_prefix_tokens)
    prompt = parent_run.get("meta", {}).get("prompt", "")
    current_text = joined_run_text(parent_run)
    risky_claim_text = (current_harness.get("standalone_claim") or claim.get("text") or "").strip()
    evidence_text = format_harness_evidence_for_prompt(current_harness)

    rewrite_prompt = (
        "You are revising a draft answer to remove unsupported or unstable specifics.\n"
        "Write the revised answer in natural prose, not bullet commentary about the edit.\n"
        "Rules:\n"
        "- Keep the safe prefix unchanged.\n"
        "- Remove or soften unsupported specifics.\n"
        "- Do not invent new concrete details.\n"
        "- Prefer explicit uncertainty over swapping one concrete detail for another guess.\n"
        "- Do not replace one title, ISBN, publisher, date, quotation, or case identifier with a new unsupported one.\n"
        "- If multiple key details are unstable, say you cannot verify the underlying work, case, or source instead of fabricating substitutes.\n"
        "- If the central premise or a key bibliographic detail cannot be verified, say you cannot verify it.\n"
        "- Preserve the user's requested language and format when possible.\n\n"
        "Original user request:\n"
        f"{prompt}\n\n"
        "Draft answer so far:\n"
        f"{current_text}\n\n"
        "Risky claim:\n"
        f"{risky_claim_text}\n\n"
        "Probe evidence:\n"
        f"{evidence_text}\n\n"
        "Revised answer:\n"
        f"{safe_prefix_text}"
    )

    settings = copy.deepcopy(parent_run.get("meta", {}).get("generation_settings") or {})
    settings = sanitize_settings(settings)
    settings["temperature"] = min(float(settings.get("temperature") or 0.7), 0.2)
    settings["top_p"] = min(float(settings.get("top_p") or 0.95), 0.9)
    settings["seed"] = stable_seed(settings.get("seed", 0), risky_claim_text, "policy-rewrite")
    settings["id_slot"] = probe_slot_for_settings(settings, 6)
    prefill_tokens = prefill_tokens_from_text(safe_prefix_text, short_hash(prompt), settings.get("top_n", 5))
    run = create_run_object(
        prompt=prompt,
        base_url=parent_run.get("meta", {}).get("base_url", DEFAULTS["base_url"]),
        settings=settings,
        label=f"Rewrite@{claim_token_start}",
        prefill_tokens=prefill_tokens,
        inference_prompt=rewrite_prompt,
    )

    max_new_tokens = min(
        max(16, int(policy["branch_lookahead"]) * 4),
        max(0, settings.get("max_tokens", DEFAULTS["max_tokens"]) - len(prefill_tokens)),
    )
    decode_run_local(
        run,
        max_new_tokens=max_new_tokens,
        progress_cb=progress_cb,
        phase="rewrite_candidate",
    )

    candidate_window = run.get("tokens", [])[claim_token_start:]
    risks = [float(t.get("decoder_risk") or 0.0) for t in candidate_window] if candidate_window else []
    return {
        "run": run,
        "alt_rank": -1,
        "alt_token_text": "[policy rewrite]",
        "alt_token_id": None,
        "avg_risk": (sum(risks) / len(risks)) if risks else None,
        "max_risk": max(risks) if risks else None,
        "token_count": len(run.get("tokens") or []),
        "strategy": "policy_rewrite",
        "window_start": claim_token_start,
    }


def run_global_rewrite_candidate(parent_run, current_harness, object_harness, policy, progress_cb=None):
    prompt = parent_run.get("meta", {}).get("prompt", "")
    current_text = joined_run_text(parent_run)
    risky_claim_text = ((current_harness or {}).get("standalone_claim") or (((current_harness or {}).get("claim") or {}).get("text")) or "").strip()
    claim_evidence_text = format_harness_evidence_for_prompt(current_harness)
    object_evidence_text = format_object_harness_evidence_for_prompt(object_harness)
    profile = (object_harness or {}).get("profile") or {}
    object_summary = profile.get("summary") or "the referenced work or source"

    rewrite_prompt = (
        "You are rewriting an answer conservatively after factual audits flagged an unstable referenced work or source.\n"
        "Write a full replacement answer in normal prose or bullets that matches the user's requested format.\n"
        "Rules:\n"
        "- Keep supported generic content only if it still fits the user's request.\n"
        "- Remove unsupported specifics such as ISBNs, publishers, dates, quotations, holdings, case identifiers, or exact titles when they are not stable.\n"
        "- Do not replace one unsupported bibliographic detail with another unsupported guess.\n"
        "- If the referenced work or source cannot be verified confidently, say so plainly.\n"
        "- Do not mention telemetry, probes, audits, internal policies, or hidden checks.\n"
        "- Preserve the user's requested language and format when possible.\n\n"
        "Original user request:\n"
        f"{prompt}\n\n"
        "Current draft answer:\n"
        f"{current_text}\n\n"
        "Most unstable claim:\n"
        f"{risky_claim_text or '[none]'}\n\n"
        "Claim evidence summary:\n"
        f"{claim_evidence_text}\n\n"
        "Object-level evidence summary:\n"
        f"{object_evidence_text}\n\n"
        f"Rewrite the answer conservatively around {object_summary}:\n"
    )

    settings = copy.deepcopy(parent_run.get("meta", {}).get("generation_settings") or {})
    settings = sanitize_settings(settings)
    settings["temperature"] = min(float(settings.get("temperature") or 0.7), 0.15)
    settings["top_p"] = min(float(settings.get("top_p") or 0.95), 0.9)
    settings["seed"] = stable_seed(settings.get("seed", 0), profile.get("object_key"), risky_claim_text, "global-rewrite")
    settings["id_slot"] = probe_slot_for_settings(settings, 7)
    run = create_run_object(
        prompt=prompt,
        base_url=parent_run.get("meta", {}).get("base_url", DEFAULTS["base_url"]),
        settings=settings,
        label="Global Rewrite",
        inference_prompt=rewrite_prompt,
    )

    wrapped_progress = None
    if progress_cb is not None:
        wrapped_progress = lambda payload: progress_cb({**payload, "strategy": "global_rewrite"})

    decode_run_local(
        run,
        progress_cb=wrapped_progress,
        phase="rewrite_candidate",
    )

    candidate_window = run.get("tokens", [])
    risks = [float(t.get("decoder_risk") or 0.0) for t in candidate_window] if candidate_window else []
    return {
        "run": run,
        "alt_rank": -2,
        "alt_token_text": "[global rewrite]",
        "alt_token_id": None,
        "avg_risk": (sum(risks) / len(risks)) if risks else None,
        "max_risk": max(risks) if risks else None,
        "token_count": len(run.get("tokens") or []),
        "strategy": "global_rewrite",
        "window_start": 0,
    }


def run_live_correction_experiment(prompt, base_url, settings, policy, experiment_id=None):
    def progress(event_type, payload):
        if not experiment_id:
            return
        event = {"type": event_type, "experiment_id": experiment_id}
        event.update(payload)
        broadcast_event(event)

    status_lock = threading.Lock()
    status_state = {
        "phase": "setup",
        "message": "Preparing live experiment.",
        "since": time.monotonic(),
        "rewrite_pass": 1,
        "accepted_rewrites": 0,
        "loop_budget": policy.get("correction_loops", 0),
    }
    heartbeat_stop = threading.Event()

    def snapshot_status():
        with status_lock:
            snap = copy.deepcopy(status_state)
        since = snap.pop("since", time.monotonic())
        snap["waiting_ms"] = max(0, int((time.monotonic() - since) * 1000))
        return snap

    def update_status(message, **extra):
        with status_lock:
            status_state["message"] = message
            status_state["since"] = time.monotonic()
            status_state.update(extra)
            snap = copy.deepcopy(status_state)
        since = snap.pop("since", time.monotonic())
        snap["waiting_ms"] = max(0, int((time.monotonic() - since) * 1000))
        progress("live_experiment_status", snap)

    def heartbeat_worker():
        while not heartbeat_stop.wait(2.0):
            progress("live_experiment_status", snapshot_status())

    heartbeat_thread = threading.Thread(target=heartbeat_worker, name="live-experiment-heartbeat", daemon=True)
    heartbeat_thread.start()

    try:
        baseline = create_run_object(prompt=prompt, base_url=base_url, settings=sanitize_settings(copy.deepcopy(settings)), label="Baseline")
        corrected = create_run_object(prompt=prompt, base_url=base_url, settings=sanitize_settings(copy.deepcopy(settings)), label="Corrected")

        progress(
            "live_experiment_started",
            {
                "baseline_run_id": baseline["run_id"],
                "corrected_run_id": corrected["run_id"],
                "mode": "hybrid_correction_loop",
                "correction_loops": policy["correction_loops"],
            },
        )

        update_status("Decoding baseline path token by token.", phase="baseline", rewrite_pass=1, accepted_rewrites=0)
        decode_run_local(
            baseline,
            progress_cb=lambda payload: progress("live_experiment_progress", payload),
            phase="baseline",
        )

        started = time.monotonic()
        initialize_prefill_tokens(corrected)
        meta = corrected["meta"]
        cfg = meta["generation_settings"]
        prompt_hash = meta["prompt_hash"]
        model_name = meta.get("model")
        base_url = meta["base_url"]
        inference_prompt = meta.get("inference_prompt") or meta.get("prompt", "")

        vector_provider = VectorProvider(
            cfg.get("vector_mode", "placeholder"),
            base_url,
            cfg.get("vector_dim", DEFAULTS["vector_dim"]),
            cfg.get("vector_window", DEFAULTS["vector_window"]),
            prompt_hash,
        )
        history_text = [t.get("text") or "" for t in corrected["tokens"]]
        if corrected["tokens"] and isinstance(corrected["tokens"][-1].get("embedding"), list):
            vector_provider.prev = corrected["tokens"][-1]["embedding"]

        token_index = len(corrected["tokens"])
        generated_text = ""
        remaining = max(0, cfg.get("max_tokens", DEFAULTS["max_tokens"]) - token_index)
        interventions = []
        last_intervention_index = -999
        last_harness_claim_key = None
        update_status(
            "Decoding live replay path token by token.",
            phase="corrected",
            rewrite_pass=1,
            accepted_rewrites=0,
            loop_budget=policy["correction_loops"],
        )

        def describe_harness_stage(payload):
            stage = payload.get("stage")
            if stage == "cache_hit":
                return "Reusing cached claim evidence."
            if stage == "claim_selected":
                return "Selected a check-worthy claim near the risky region."
            if stage == "plan_started":
                return "Planning targeted factual probes for the risky claim."
            if stage == "plan_ready":
                return f"Prepared {payload.get('facts_count', 0)} factual probes for the claim."
            if stage == "plan_fallback":
                return "Fell back to heuristic fact extraction for the risky claim."
            if stage == "probe_sample_started":
                return f"Asking probe {payload.get('probe_index', 0) + 1}, sample {payload.get('sample_index', 1)}."
            if stage == "probe_sample_completed":
                return f"Probe {payload.get('probe_index', 0) + 1}, sample {payload.get('sample_index', 1)} returned: {payload.get('answer_text') or 'unknown'}."
            if stage == "judge_started":
                return f"Judging factual detail {payload.get('fact_index', 0) + 1}."
            if stage == "judge_completed":
                return (
                    f"Judged detail {payload.get('fact_index', 0) + 1}: "
                    f"{payload.get('verdict', 'uncertain')} at risk {payload.get('fact_risk') if payload.get('fact_risk') is not None else '-'}."
                )
            if stage == "harness_completed":
                return (
                    f"Claim evidence complete: {payload.get('label', 'unknown')} "
                    f"at {payload.get('claim_risk') if payload.get('claim_risk') is not None else '-'}."
                )
            return "Evaluating risky claim."

        def describe_object_stage(payload):
            stage = payload.get("stage")
            if stage == "cache_hit":
                return "Reusing cached object-level evidence."
            if stage == "object_selected":
                return f"Tracking object-level evidence for {payload.get('object_summary') or 'the referenced source'}."
            if stage == "object_probe_started":
                return f"Auditing whether {payload.get('object_summary') or 'the referenced source'} is verifiable. Sample {payload.get('sample_index', 0) + 1}."
            if stage == "object_probe_completed":
                return (
                    f"Object audit sample {payload.get('sample_index', 0) + 1}: "
                    f"{payload.get('verdict', 'uncertain')} at risk {payload.get('object_risk') if payload.get('object_risk') is not None else '-'}."
                )
            if stage == "object_harness_completed":
                recommendation = payload.get("mode_recommendation", "continue").replace("_", " ")
                return (
                    f"Object evidence complete: {payload.get('label', 'unknown')} "
                    f"at {payload.get('object_risk') if payload.get('object_risk') is not None else '-'}; "
                    f"recommended mode {recommendation}."
                )
            return "Evaluating whether the referenced object is stable enough to answer directly."

        while remaining > 0:
            token, stop_info, model_name = generate_single_token_step(
                base_url,
                inference_prompt + generated_text,
                cfg,
                model_name,
                prompt_hash,
                token_index,
                history_text,
                vector_provider,
                started,
            )
            if model_name and meta.get("model") != model_name:
                meta["model"] = model_name
            if token is None:
                break

            token_text = token.get("text", "")
            generated_text += token_text
            history_text.append(token_text)
            meta["providers"]["embedding"] = vector_provider.mode
            add_token_local(corrected, token)
            token_index += 1
            remaining -= 1

            recompute_kinematics(corrected["tokens"])
            enrich_decoder_diagnostics(corrected["tokens"])
            corrected["analysis"]["regime_markers"] = detect_regime_markers(corrected["tokens"])
            corrected["analysis"]["decoder_alerts"] = detect_decoder_alerts(corrected["tokens"])
            max_decoder_risk = None
            for item in corrected["tokens"]:
                try:
                    value = float(item.get("decoder_risk"))
                except (TypeError, ValueError):
                    continue
                max_decoder_risk = value if max_decoder_risk is None else max(max_decoder_risk, value)
            progress(
                "live_experiment_progress",
                {
                    "phase": "corrected",
                    "run_id": corrected["run_id"],
                    "token_index": token_index - 1,
                    "token_text": token_text,
                    "decoder_risk": token.get("decoder_risk"),
                    "max_decoder_risk": max_decoder_risk,
                    "alert_count": len(corrected["analysis"]["decoder_alerts"]),
                    "token": live_token_snapshot(token),
                    "text": joined_run_text(corrected),
                    "token_count": len(corrected["tokens"]),
                    "rewrite_pass": len(interventions) + 1,
                    "accepted_rewrites": len(interventions),
                    "loop_budget": policy["correction_loops"],
                },
            )

            if (
                len(interventions) < policy["max_interventions"]
                and len(corrected["tokens"]) - 1 >= last_intervention_index + policy["cooldown_tokens"]
            ):
                rollback_index, current_avg_risk = rollback_index_for_run(corrected, policy)
                if rollback_index is not None:
                    trigger_index = len(corrected["tokens"]) - 1
                    focus_claim = claim_for_token_index(corrected, rollback_index, require_complete=True)
                    claim_key = None
                    if focus_claim is not None:
                        claim_key = (focus_claim.get("start"), focus_claim.get("end"))
                    if claim_key is None or claim_key == last_harness_claim_key:
                        if stop_info.get("hard_stop"):
                            break
                        if stop_info.get("stop") and remaining <= 0:
                            break
                        continue

                    rewrite_pass = len(interventions) + 1

                    def harness_progress(payload):
                        update_status(
                            describe_harness_stage(payload),
                            phase="corrected",
                            rewrite_pass=rewrite_pass,
                            accepted_rewrites=len(interventions),
                            loop_budget=policy["correction_loops"],
                            rollback_index=rollback_index,
                            trigger_index=trigger_index,
                            harness_stage=payload.get("stage"),
                        )
                        progress(
                            "live_experiment_harness",
                            {
                                "phase": "corrected",
                                "run_id": corrected["run_id"],
                                "rollback_index": rollback_index,
                                "trigger_index": trigger_index,
                                "rewrite_pass": rewrite_pass,
                                "accepted_rewrites": len(interventions),
                                "loop_budget": policy["correction_loops"],
                                **payload,
                            },
                        )

                    current_harness = run_claim_harness(
                        corrected,
                        rollback_index,
                        policy,
                        progress_cb=harness_progress,
                        progress_context={"run_id": corrected["run_id"]},
                    )
                    last_harness_claim_key = claim_key
                    progress(
                        "live_experiment_harness",
                        {
                            "phase": "corrected",
                            "run_id": corrected["run_id"],
                            "rollback_index": rollback_index,
                            "trigger_index": trigger_index,
                            "rewrite_pass": rewrite_pass,
                            "accepted_rewrites": len(interventions),
                            "loop_budget": policy["correction_loops"],
                            "status": current_harness.get("status"),
                            "label": current_harness.get("label"),
                            "claim_risk": current_harness.get("claim_risk"),
                            "claim_text": (current_harness.get("standalone_claim") or ((current_harness.get("claim") or {}).get("text")) or ""),
                        },
                    )
                    current_object_harness = None
                    if current_harness.get("status") == "ok":
                        def object_progress(payload):
                            update_status(
                                describe_object_stage(payload),
                                phase="object_harness",
                                rewrite_pass=rewrite_pass,
                                accepted_rewrites=len(interventions),
                                loop_budget=policy["correction_loops"],
                                rollback_index=rollback_index,
                                trigger_index=trigger_index,
                                object_stage=payload.get("stage"),
                            )
                            progress(
                                "live_experiment_object",
                                {
                                    "phase": "corrected",
                                    "run_id": corrected["run_id"],
                                    "rollback_index": rollback_index,
                                    "trigger_index": trigger_index,
                                    "rewrite_pass": rewrite_pass,
                                    "accepted_rewrites": len(interventions),
                                    "loop_budget": policy["correction_loops"],
                                    **payload,
                                },
                            )

                        current_object_harness = run_object_harness(
                            corrected,
                            policy,
                            progress_cb=object_progress,
                            progress_context={"run_id": corrected["run_id"]},
                        )
                        progress(
                            "live_experiment_object",
                            {
                                "phase": "corrected",
                                "run_id": corrected["run_id"],
                                "rollback_index": rollback_index,
                                "trigger_index": trigger_index,
                                "rewrite_pass": rewrite_pass,
                                "accepted_rewrites": len(interventions),
                                "loop_budget": policy["correction_loops"],
                                "status": current_object_harness.get("status"),
                                "label": current_object_harness.get("label"),
                                "object_risk": current_object_harness.get("object_risk"),
                                "mode_recommendation": current_object_harness.get("mode_recommendation"),
                                "abstain_recommended": current_object_harness.get("abstain_recommended"),
                                "object_summary": ((current_object_harness.get("profile") or {}).get("summary")) or "",
                                "reason": current_object_harness.get("reason_summary"),
                            },
                        )

                    should_try_candidates = (
                        current_harness.get("status") == "ok"
                        and (
                            (
                                current_harness.get("claim_risk") is not None
                                and current_harness["claim_risk"] >= policy["harness_trigger_threshold"]
                            )
                            or (
                                isinstance(current_object_harness, dict)
                                and current_object_harness.get("status") == "ok"
                                and current_object_harness.get("object_risk") is not None
                                and current_object_harness["object_risk"] >= policy["object_trigger_threshold"]
                            )
                        )
                    )

                    if should_try_candidates:
                        candidate_results = []
                        target = corrected["tokens"][rollback_index]
                        topn = target.get("topN") or []
                        limit = min(len(topn), policy["branch_candidates"] + 1)
                        for alt_rank in range(1, limit):
                            alt_text = str((topn[alt_rank] or {}).get("token_text") or "")
                            if not alt_text.strip() or not any(ch.isalnum() for ch in alt_text):
                                continue
                            update_status(
                                f"Decoding prefix candidate {alt_rank + 1} from rollback token {rollback_index}.",
                                phase="candidate",
                                rewrite_pass=rewrite_pass,
                                accepted_rewrites=len(interventions),
                                loop_budget=policy["correction_loops"],
                                candidate_strategy="prefix_replay",
                                candidate_rank=alt_rank,
                            )
                            progress(
                                "live_experiment_candidate",
                                {
                                    "stage": "candidate_started",
                                    "strategy": "prefix_replay",
                                    "alt_rank": alt_rank,
                                    "alt_token_text": alt_text,
                                    "rollback_index": rollback_index,
                                    "trigger_index": trigger_index,
                                    "rewrite_pass": rewrite_pass,
                                    "accepted_rewrites": len(interventions),
                                    "loop_budget": policy["correction_loops"],
                                },
                            )
                            result = run_candidate_branch(corrected, rollback_index, alt_rank, policy["branch_lookahead"])
                            if result is not None:
                                candidate_results.append(result)
                                progress(
                                    "live_experiment_candidate",
                                    {
                                        "stage": "candidate_completed",
                                        "strategy": "prefix_replay",
                                        "alt_rank": alt_rank,
                                        "alt_token_text": alt_text,
                                        "avg_risk": result.get("avg_risk"),
                                        "max_risk": result.get("max_risk"),
                                        "rewrite_pass": rewrite_pass,
                                        "accepted_rewrites": len(interventions),
                                        "loop_budget": policy["correction_loops"],
                                    },
                                )

                        update_status(
                            "Generating a conservative policy rewrite candidate.",
                            phase="rewrite_candidate",
                            rewrite_pass=rewrite_pass,
                            accepted_rewrites=len(interventions),
                            loop_budget=policy["correction_loops"],
                            rollback_index=rollback_index,
                            trigger_index=trigger_index,
                        )
                        progress(
                            "live_experiment_candidate",
                            {
                                "stage": "candidate_started",
                                "strategy": "policy_rewrite",
                                "alt_rank": -1,
                                "alt_token_text": "[policy rewrite]",
                                "rollback_index": rollback_index,
                                "trigger_index": trigger_index,
                                "rewrite_pass": rewrite_pass,
                                "accepted_rewrites": len(interventions),
                                "loop_budget": policy["correction_loops"],
                            },
                        )
                        rewrite_candidate = run_policy_rewrite_candidate(
                            corrected,
                            current_harness,
                            rollback_index,
                            policy,
                            progress_cb=lambda payload: progress(
                                "live_experiment_progress",
                                {
                                    **payload,
                                    "rewrite_pass": rewrite_pass,
                                    "accepted_rewrites": len(interventions),
                                    "loop_budget": policy["correction_loops"],
                                },
                            ),
                        )
                        candidate_results.append(rewrite_candidate)
                        progress(
                            "live_experiment_candidate",
                            {
                                "stage": "candidate_completed",
                                "strategy": "policy_rewrite",
                                "alt_rank": -1,
                                "alt_token_text": "[policy rewrite]",
                                "avg_risk": rewrite_candidate.get("avg_risk"),
                                "max_risk": rewrite_candidate.get("max_risk"),
                                "rewrite_pass": rewrite_pass,
                                "accepted_rewrites": len(interventions),
                                "loop_budget": policy["correction_loops"],
                            },
                        )

                        global_rewrite_candidate = None
                        if (
                            isinstance(current_object_harness, dict)
                            and current_object_harness.get("status") == "ok"
                            and (
                                current_object_harness.get("abstain_recommended")
                                or (current_object_harness.get("object_risk") or 0.0) >= policy["object_trigger_threshold"]
                            )
                        ):
                            update_status(
                                "Switching to a whole-answer conservative rewrite candidate.",
                                phase="rewrite_candidate",
                                rewrite_pass=rewrite_pass,
                                accepted_rewrites=len(interventions),
                                loop_budget=policy["correction_loops"],
                                candidate_strategy="global_rewrite",
                                rollback_index=rollback_index,
                                trigger_index=trigger_index,
                            )
                            progress(
                                "live_experiment_candidate",
                                {
                                    "stage": "candidate_started",
                                    "strategy": "global_rewrite",
                                    "alt_rank": -2,
                                    "alt_token_text": "[global rewrite]",
                                    "rollback_index": rollback_index,
                                    "trigger_index": trigger_index,
                                    "rewrite_pass": rewrite_pass,
                                    "accepted_rewrites": len(interventions),
                                    "loop_budget": policy["correction_loops"],
                                },
                            )
                            global_rewrite_candidate = run_global_rewrite_candidate(
                                corrected,
                                current_harness,
                                current_object_harness,
                                policy,
                                progress_cb=lambda payload: progress(
                                    "live_experiment_progress",
                                    {
                                        **payload,
                                        "rewrite_pass": rewrite_pass,
                                        "accepted_rewrites": len(interventions),
                                        "loop_budget": policy["correction_loops"],
                                    },
                                ),
                            )
                            candidate_results.append(global_rewrite_candidate)
                            progress(
                                "live_experiment_candidate",
                                {
                                    "stage": "candidate_completed",
                                    "strategy": "global_rewrite",
                                    "alt_rank": -2,
                                    "alt_token_text": "[global rewrite]",
                                    "avg_risk": global_rewrite_candidate.get("avg_risk"),
                                    "max_risk": global_rewrite_candidate.get("max_risk"),
                                    "rewrite_pass": rewrite_pass,
                                    "accepted_rewrites": len(interventions),
                                    "loop_budget": policy["correction_loops"],
                                },
                            )

                        for item in candidate_results:
                            focus_index = item.get("window_start", rollback_index)

                            def candidate_harness_progress(payload, item=item):
                                update_status(
                                    describe_harness_stage(payload),
                                    phase="candidate_harness",
                                    rewrite_pass=rewrite_pass,
                                    accepted_rewrites=len(interventions),
                                    loop_budget=policy["correction_loops"],
                                    candidate_strategy=item.get("strategy"),
                                    candidate_rank=item.get("alt_rank"),
                                )
                                progress(
                                    "live_experiment_harness",
                                    {
                                        "phase": "candidate",
                                        "run_id": item["run"]["run_id"],
                                        "rollback_index": rollback_index,
                                        "trigger_index": trigger_index,
                                        "rewrite_pass": rewrite_pass,
                                        "accepted_rewrites": len(interventions),
                                        "loop_budget": policy["correction_loops"],
                                        "candidate_strategy": item.get("strategy"),
                                        "candidate_alt_rank": item.get("alt_rank"),
                                        **payload,
                                    },
                                )

                            item["harness"] = run_claim_harness(
                                item["run"],
                                focus_index,
                                policy,
                                progress_cb=candidate_harness_progress,
                                progress_context={"run_id": item["run"]["run_id"]},
                            )
                            def candidate_object_progress(payload, item=item):
                                update_status(
                                    describe_object_stage(payload),
                                    phase="candidate_object",
                                    rewrite_pass=rewrite_pass,
                                    accepted_rewrites=len(interventions),
                                    loop_budget=policy["correction_loops"],
                                    candidate_strategy=item.get("strategy"),
                                    candidate_rank=item.get("alt_rank"),
                                )
                                progress(
                                    "live_experiment_object",
                                    {
                                        "phase": "candidate",
                                        "run_id": item["run"]["run_id"],
                                        "rollback_index": rollback_index,
                                        "trigger_index": trigger_index,
                                        "rewrite_pass": rewrite_pass,
                                        "accepted_rewrites": len(interventions),
                                        "loop_budget": policy["correction_loops"],
                                        "candidate_strategy": item.get("strategy"),
                                        "candidate_alt_rank": item.get("alt_rank"),
                                        **payload,
                                    },
                                )

                            item["object_harness"] = run_object_harness(
                                item["run"],
                                policy,
                                progress_cb=candidate_object_progress,
                                progress_context={"run_id": item["run"]["run_id"]},
                            )

                        ranked_candidates = []
                        for item in candidate_results:
                            harness = item.get("harness")
                            harness_risk = harness.get("claim_risk") if isinstance(harness, dict) else None
                            object_harness = item.get("object_harness")
                            object_risk = object_harness.get("object_risk") if isinstance(object_harness, dict) else None
                            ranked_candidates.append({**item, "harness_risk": harness_risk, "object_risk": object_risk})

                        valid_candidates = [
                            item
                            for item in ranked_candidates
                            if item.get("harness_risk") is not None or item.get("object_risk") is not None
                        ]
                        baseline_object_risk = (
                            float(current_object_harness["object_risk"])
                            if isinstance(current_object_harness, dict) and current_object_harness.get("object_risk") is not None
                            else None
                        )
                        prefer_object_mode = (
                            baseline_object_risk is not None
                            and baseline_object_risk >= policy["object_trigger_threshold"]
                        ) or (isinstance(current_object_harness, dict) and current_object_harness.get("abstain_recommended"))
                        valid_candidates.sort(
                            key=lambda item: (
                                item.get("object_risk") if prefer_object_mode and item.get("object_risk") is not None else 1.0,
                                item.get("harness_risk"),
                                item.get("avg_risk") if item.get("avg_risk") is not None else 1.0,
                                item.get("max_risk") or 0.0,
                            )
                        )
                        best = valid_candidates[0] if valid_candidates else None

                        baseline_claim_risk = float(current_harness["claim_risk"])
                        harness_improved = (
                            best is not None
                            and best.get("harness_risk") is not None
                            and best["harness_risk"] <= (baseline_claim_risk - policy["harness_min_drop"])
                        )
                        risk_improved = (
                            best is not None
                            and current_avg_risk is not None
                            and best.get("avg_risk") is not None
                            and best["avg_risk"] <= (current_avg_risk - max(0.02, policy["min_risk_drop"] * 0.5))
                        )
                        rewrite_safe_enough = (
                            best is not None
                            and best.get("strategy") in ("policy_rewrite", "global_rewrite")
                            and best.get("avg_risk") is not None
                            and (
                                current_avg_risk is None
                                or best["avg_risk"] <= current_avg_risk + 0.05
                            )
                        )
                        object_improved = (
                            best is not None
                            and baseline_object_risk is not None
                            and best.get("object_risk") is not None
                            and best["object_risk"] <= (baseline_object_risk - policy["object_min_drop"])
                        )
                        rewrite_abstains = (
                            best is not None
                            and best.get("strategy") in ("policy_rewrite", "global_rewrite")
                            and has_abstention_signal(joined_run_text(best.get("run")))
                            and (
                                (
                                    best.get("harness_risk") is not None
                                    and best["harness_risk"] <= baseline_claim_risk + 0.03
                                )
                                or ((best.get("harness") or {}).get("status") in ("not_checkworthy", "no_complete_claim"))
                            )
                            and (
                                baseline_object_risk is None
                                or best.get("object_risk") is None
                                or best["object_risk"] <= baseline_object_risk + 0.03
                            )
                        )
                        accepted_best = best is not None and (
                            (harness_improved and (risk_improved or rewrite_safe_enough))
                            or (prefer_object_mode and (object_improved or rewrite_abstains) and rewrite_safe_enough)
                            or rewrite_abstains
                        )

                        if rewrite_candidate is not None:
                            rewrite_accepted = accepted_best and best is not None and best.get("strategy") == "policy_rewrite"
                            progress(
                                "live_experiment_candidate",
                                {
                                    "stage": "candidate_result",
                                    "strategy": "policy_rewrite",
                                    "accepted": rewrite_accepted,
                                    "candidate_run_id": rewrite_candidate["run"]["run_id"],
                                    "candidate_run": copy.deepcopy(rewrite_candidate["run"]),
                                    "current_corrected_run": copy.deepcopy(corrected),
                                    "rewrite_pass": rewrite_pass,
                                    "accepted_rewrites": len(interventions),
                                    "loop_budget": policy["correction_loops"],
                                },
                            )
                        if global_rewrite_candidate is not None:
                            global_rewrite_accepted = accepted_best and best is not None and best.get("strategy") == "global_rewrite"
                            progress(
                                "live_experiment_candidate",
                                {
                                    "stage": "candidate_result",
                                    "strategy": "global_rewrite",
                                    "accepted": global_rewrite_accepted,
                                    "candidate_run_id": global_rewrite_candidate["run"]["run_id"],
                                    "candidate_run": copy.deepcopy(global_rewrite_candidate["run"]),
                                    "current_corrected_run": copy.deepcopy(corrected),
                                    "rewrite_pass": rewrite_pass,
                                    "accepted_rewrites": len(interventions),
                                    "loop_budget": policy["correction_loops"],
                                },
                            )

                        if accepted_best:
                            chosen_harness = best.get("harness") or {}
                            corrected["tokens"] = best["run"]["tokens"]
                            corrected["meta"]["providers"]["embedding"] = best["run"]["meta"]["providers"].get(
                                "embedding",
                                corrected["meta"]["providers"].get("embedding"),
                            )
                            recompute_kinematics(corrected["tokens"])
                            enrich_decoder_diagnostics(corrected["tokens"])
                            corrected["analysis"]["regime_markers"] = detect_regime_markers(corrected["tokens"])
                            corrected["analysis"]["decoder_alerts"] = detect_decoder_alerts(corrected["tokens"])

                            history_text = [t.get("text") or "" for t in corrected["tokens"]]
                            token_index = len(corrected["tokens"])
                            generated_text = "".join(history_text)
                            remaining = max(0, cfg.get("max_tokens", DEFAULTS["max_tokens"]) - token_index)
                            vector_provider = VectorProvider(
                                corrected["meta"]["providers"].get("embedding", cfg.get("vector_mode", "placeholder")),
                                base_url,
                                cfg.get("vector_dim", DEFAULTS["vector_dim"]),
                                cfg.get("vector_window", DEFAULTS["vector_window"]),
                                prompt_hash,
                            )
                            if corrected["tokens"] and isinstance(corrected["tokens"][-1].get("embedding"), list):
                                vector_provider.prev = corrected["tokens"][-1]["embedding"]

                            interventions.append(
                                {
                                    "mode": best.get("strategy"),
                                    "trigger_index": trigger_index,
                                    "rollback_index": rollback_index,
                                    "baseline_avg_risk": current_avg_risk,
                                    "baseline_claim_risk": current_harness.get("claim_risk"),
                                    "baseline_claim_label": current_harness.get("label"),
                                    "baseline_claim": current_harness.get("standalone_claim") or ((current_harness.get("claim") or {}).get("text")),
                                    "baseline_object_risk": baseline_object_risk,
                                    "baseline_object_label": (current_object_harness or {}).get("label"),
                                    "baseline_object_summary": (((current_object_harness or {}).get("profile") or {}).get("summary")),
                                    "chosen_alt_rank": best["alt_rank"],
                                    "chosen_alt_text": best["alt_token_text"],
                                    "chosen_strategy": best.get("strategy"),
                                    "chosen_avg_risk": best["avg_risk"],
                                    "chosen_max_risk": best["max_risk"],
                                    "chosen_claim_risk": chosen_harness.get("claim_risk"),
                                    "chosen_claim_label": chosen_harness.get("label"),
                                    "chosen_claim": chosen_harness.get("standalone_claim") or ((chosen_harness.get("claim") or {}).get("text")),
                                    "chosen_object_risk": (best.get("object_harness") or {}).get("object_risk"),
                                    "chosen_object_label": (best.get("object_harness") or {}).get("label"),
                                    "chosen_object_summary": (((best.get("object_harness") or {}).get("profile") or {}).get("summary")),
                                    "baseline_harness": copy.deepcopy(current_harness),
                                    "chosen_harness": copy.deepcopy(chosen_harness),
                                    "baseline_object_harness": copy.deepcopy(current_object_harness),
                                    "chosen_object_harness": copy.deepcopy(best.get("object_harness")),
                                    "candidates": [
                                        {
                                            "alt_rank": item["alt_rank"],
                                            "alt_token_text": item["alt_token_text"],
                                            "strategy": item.get("strategy"),
                                            "avg_risk": item["avg_risk"],
                                            "max_risk": item["max_risk"],
                                            "claim_risk": item.get("harness_risk"),
                                            "claim_label": (item.get("harness") or {}).get("label"),
                                            "object_risk": item.get("object_risk"),
                                            "object_label": (item.get("object_harness") or {}).get("label"),
                                            "accepted": item.get("strategy") == best.get("strategy") and item.get("alt_rank") == best.get("alt_rank"),
                                        }
                                        for item in ranked_candidates
                                    ],
                                }
                            )
                            update_status(
                                f"Accepted {best.get('strategy', 'candidate').replace('_', ' ')} and moved to pass {len(interventions) + 1}.",
                                phase="corrected",
                                rewrite_pass=len(interventions) + 1,
                                accepted_rewrites=len(interventions),
                                loop_budget=policy["correction_loops"],
                            )
                            progress(
                                "live_experiment_intervention",
                                {
                                    "rollback_index": rollback_index,
                                    "trigger_index": trigger_index,
                                    "baseline_avg_risk": current_avg_risk,
                                    "chosen_alt_rank": best["alt_rank"],
                                    "chosen_alt_text": best["alt_token_text"],
                                    "chosen_strategy": best.get("strategy"),
                                    "chosen_avg_risk": best["avg_risk"],
                                    "chosen_max_risk": best["max_risk"],
                                    "baseline_claim_risk": current_harness.get("claim_risk"),
                                    "baseline_claim_label": current_harness.get("label"),
                                    "baseline_claim": current_harness.get("standalone_claim") or ((current_harness.get("claim") or {}).get("text")),
                                    "baseline_object_risk": baseline_object_risk,
                                    "baseline_object_label": (current_object_harness or {}).get("label"),
                                    "baseline_object_summary": (((current_object_harness or {}).get("profile") or {}).get("summary")),
                                    "chosen_claim_risk": chosen_harness.get("claim_risk"),
                                    "chosen_claim_label": chosen_harness.get("label"),
                                    "chosen_claim": chosen_harness.get("standalone_claim") or ((chosen_harness.get("claim") or {}).get("text")),
                                    "chosen_object_risk": (best.get("object_harness") or {}).get("object_risk"),
                                    "chosen_object_label": (best.get("object_harness") or {}).get("label"),
                                    "chosen_object_summary": (((best.get("object_harness") or {}).get("profile") or {}).get("summary")),
                                    "candidates": [
                                        {
                                            "alt_rank": item["alt_rank"],
                                            "alt_token_text": item["alt_token_text"],
                                            "strategy": item.get("strategy"),
                                            "avg_risk": item["avg_risk"],
                                            "max_risk": item["max_risk"],
                                            "claim_risk": item.get("harness_risk"),
                                            "claim_label": (item.get("harness") or {}).get("label"),
                                            "object_risk": item.get("object_risk"),
                                            "object_label": (item.get("object_harness") or {}).get("label"),
                                            "accepted": item.get("strategy") == best.get("strategy") and item.get("alt_rank") == best.get("alt_rank"),
                                        }
                                        for item in ranked_candidates
                                    ],
                                    "corrected_text": joined_run_text(corrected),
                                    "corrected_run_id": corrected["run_id"],
                                    "corrected_run": copy.deepcopy(corrected),
                                    "rewrite_pass": len(interventions) + 1,
                                    "accepted_rewrites": len(interventions),
                                    "loop_budget": policy["correction_loops"],
                                    "baseline_harness": copy.deepcopy(current_harness),
                                    "chosen_harness": copy.deepcopy(chosen_harness),
                                    "baseline_object_harness": copy.deepcopy(current_object_harness),
                                    "chosen_object_harness": copy.deepcopy(best.get("object_harness")),
                                },
                            )
                            last_intervention_index = len(corrected["tokens"]) - 1
                        else:
                            update_status(
                                "No candidate cleared the current acceptance gates; continuing replay path.",
                                phase="corrected",
                                rewrite_pass=rewrite_pass,
                                accepted_rewrites=len(interventions),
                                loop_budget=policy["correction_loops"],
                            )

            if stop_info.get("hard_stop"):
                break
            if stop_info.get("stop") and remaining <= 0:
                break

        finalize_completed_run(corrected, started, status="complete")
        corrected.setdefault("meta", {}).setdefault("live_experiment", {})
        corrected["meta"]["live_experiment"]["had_intervention"] = bool(interventions)
        corrected["meta"]["live_experiment"]["comparison_role"] = "corrected" if interventions else "replay_sample"
        if not interventions:
            corrected["meta"]["label"] = "Replay"
        corrected.setdefault("analysis", {})["interventions"] = copy.deepcopy(interventions)

        def final_harness_progress(payload, target_name, run_id):
            update_status(
                f"Final {target_name} evidence pass. {describe_harness_stage(payload)}",
                phase="finalize",
                target=target_name,
                rewrite_pass=len(interventions) + 1,
                accepted_rewrites=len(interventions),
                loop_budget=policy["correction_loops"],
            )
            progress(
                "live_experiment_harness",
                {
                    "phase": "finalize",
                    "run_id": run_id,
                    "target": target_name,
                    "rewrite_pass": len(interventions) + 1,
                    "accepted_rewrites": len(interventions),
                    "loop_budget": policy["correction_loops"],
                    **payload,
                },
            )

        update_status("Running final baseline evidence pass.", phase="finalize", target="baseline")
        baseline_harness = run_claim_harness(
            baseline,
            peak_risk_token_index(baseline),
            policy,
            progress_cb=lambda payload: final_harness_progress(payload, "baseline", baseline["run_id"]),
            progress_context={"run_id": baseline["run_id"]},
        )
        update_status("Running final corrected evidence pass.", phase="finalize", target="corrected")
        corrected_harness = run_claim_harness(
            corrected,
            peak_risk_token_index(corrected),
            policy,
            progress_cb=lambda payload: final_harness_progress(payload, "corrected", corrected["run_id"]),
            progress_context={"run_id": corrected["run_id"]},
        )
        baseline.setdefault("analysis", {})["focus_claim"] = copy.deepcopy(baseline_harness)
        corrected.setdefault("analysis", {})["focus_claim"] = copy.deepcopy(corrected_harness)

        def final_object_progress(payload, target_name, run_id):
            update_status(
                f"Final {target_name} object pass. {describe_object_stage(payload)}",
                phase="finalize",
                target=target_name,
                rewrite_pass=len(interventions) + 1,
                accepted_rewrites=len(interventions),
                loop_budget=policy["correction_loops"],
            )
            progress(
                "live_experiment_object",
                {
                    "phase": "finalize",
                    "run_id": run_id,
                    "target": target_name,
                    "rewrite_pass": len(interventions) + 1,
                    "accepted_rewrites": len(interventions),
                    "loop_budget": policy["correction_loops"],
                    **payload,
                },
            )

        update_status("Running final baseline object evidence pass.", phase="finalize", target="baseline_object")
        baseline_object_harness = run_object_harness(
            baseline,
            policy,
            progress_cb=lambda payload: final_object_progress(payload, "baseline", baseline["run_id"]),
            progress_context={"run_id": baseline["run_id"]},
        )
        update_status("Running final corrected object evidence pass.", phase="finalize", target="corrected_object")
        corrected_object_harness = run_object_harness(
            corrected,
            policy,
            progress_cb=lambda payload: final_object_progress(payload, "corrected", corrected["run_id"]),
            progress_context={"run_id": corrected["run_id"]},
        )
        baseline.setdefault("analysis", {})["focus_object"] = copy.deepcopy(baseline_object_harness)
        corrected.setdefault("analysis", {})["focus_object"] = copy.deepcopy(corrected_object_harness)

        with RUNS_LOCK:
            RUNS[baseline["run_id"]] = copy.deepcopy(baseline)
            RUNS[corrected["run_id"]] = copy.deepcopy(corrected)
            if baseline["run_id"] not in RUN_ORDER:
                RUN_ORDER.append(baseline["run_id"])
            if corrected["run_id"] not in RUN_ORDER:
                RUN_ORDER.append(corrected["run_id"])

        update_status("Packaging final experiment result.", phase="finalize", target="result")
        return {
            "mode": "hybrid_correction_loop",
            "baseline_run_id": baseline["run_id"],
            "corrected_run_id": corrected["run_id"],
            "baseline": baseline,
            "corrected": corrected,
            "interventions": interventions,
            "harness": {
                "baseline": baseline_harness,
                "corrected": corrected_harness,
            },
            "object_harness": {
                "baseline": baseline_object_harness,
                "corrected": corrected_object_harness,
            },
            "policy": policy,
            "metrics": {
                "baseline_max_risk": baseline.get("summary", {}).get("decoder_risk_max"),
                "corrected_max_risk": corrected.get("summary", {}).get("decoder_risk_max"),
                "baseline_alerts": baseline.get("summary", {}).get("decoder_alert_count"),
                "corrected_alerts": corrected.get("summary", {}).get("decoder_alert_count"),
                "baseline_claim_risk": baseline_harness.get("claim_risk") if isinstance(baseline_harness, dict) else None,
                "corrected_claim_risk": corrected_harness.get("claim_risk") if isinstance(corrected_harness, dict) else None,
                "baseline_object_risk": baseline_object_harness.get("object_risk") if isinstance(baseline_object_harness, dict) else None,
                "corrected_object_risk": corrected_object_harness.get("object_risk") if isinstance(corrected_object_harness, dict) else None,
                "intervention_count": len(interventions),
                "correction_loops": policy["correction_loops"],
                "rewrite_passes": len(interventions) + 1,
            },
        }
    finally:
        heartbeat_stop.set()


class TelemetryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            return

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
            backend_probe = get_cached_backend_probe(DEFAULTS.get("base_url"))
            if backend_probe is None:
                backend_probe = empty_backend_probe(DEFAULTS.get("base_url"), probe_state="idle")
            send_json(
                self,
                200,
                {
                    "running": running,
                    "active_run_id": active_run_id,
                    "defaults": copy.deepcopy(DEFAULTS),
                    "backend_probe": backend_probe,
                },
            )
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

        if path == "/api/backend-info":
            raw_base_url = body.get("base_url") or DEFAULTS["base_url"]
            refresh = bool(body.get("refresh"))
            try:
                base_url = normalize_base_url(raw_base_url)
            except ValueError:
                send_json(self, 400, {"error": "base_url is invalid"})
                return
            send_json(self, 200, {"backend": probe_backend(base_url, refresh=refresh, probe_source="manual")})
            return

        if path == "/api/live-experiment":
            prompt = (body.get("prompt") or "").strip()
            raw_base_url = body.get("base_url") or DEFAULTS["base_url"]
            experiment_id = body.get("experiment_id")
            if not prompt:
                send_json(self, 400, {"error": "prompt is required"})
                return
            try:
                base_url = normalize_base_url(raw_base_url)
            except ValueError:
                send_json(self, 400, {"error": "base_url is invalid"})
                return

            settings = sanitize_settings(body.get("settings"))
            settings["chunk_size"] = 1
            policy = sanitize_live_policy(body.get("policy"))
            try:
                result = run_live_correction_experiment(prompt, base_url, settings, policy, experiment_id=experiment_id)
            except Exception as err:
                send_json(self, 500, {"error": f"live experiment failed: {err}"})
                return

            broadcast_event({"type": "run_imported", "run": copy.deepcopy(result["baseline"])})
            broadcast_event({"type": "run_imported", "run": copy.deepcopy(result["corrected"])})
            if experiment_id:
                completed_event = {"type": "live_experiment_completed", "experiment_id": experiment_id}
                completed_event.update(copy.deepcopy(result))
                broadcast_event(completed_event)
            send_json(self, 200, result)
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
            if not isinstance(run_copy["meta"].get("branch_history"), list):
                run_copy["meta"]["branch_history"] = []
            run_copy.setdefault("tokens", [])
            run_copy.setdefault("analysis", {})
            run_copy.setdefault("summary", {})
            initialize_prefill_tokens(run_copy)
            run_copy["analysis"]["regime_markers"] = detect_regime_markers(run_copy["tokens"])
            run_copy["analysis"]["decoder_alerts"] = detect_decoder_alerts(run_copy["tokens"])
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
            "base_url": normalize_base_url(args.default_base_url),
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
    start_backend_warmup(DEFAULTS["base_url"])

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
