# llama.cpp telemetry wrapper (MVP)

Thin wrapper that talks to a local `llama-server` and emits JSONL telemetry to stdout.

## Why this exists

We want live, near-live observability into autoregressive inference without touching training or proprietary internals.
The point is to show that meaningful, stable telemetry can be derived from standard inference artifacts that every
runtime already computes: logits, token IDs, embeddings (and optionally attention weights).

This is not "model introspection" in the mystical sense. It is instrumentation of inference-time signals that already
exist in the forward pass. The demo is meant to prove that these signals can be captured cheaply, mapped into a 3D
trajectory, and used to visualize uncertainty, decisiveness, and semantic motion as text is generated.

## Scientific basis (what the metrics are and where they come from)

All metrics are derived from one of three standard inference stages:

1) Token sampling stage (mandatory in all autoregressive LLMs)
   - Source: logits and probabilities produced at each decoding step.
   - Entropy H(t) = -sum(p_i * log p_i). High H means many plausible continuations; low H means commitment.
   - Logit margin M(t) = logit_top1 - logit_top2 (or top1/top2 prob ratio). High margin means decisive choice.
   - Top-K mass = sum of top-K probabilities. High mass indicates distribution collapse / tunnel behavior.

2) Representation / embedding stage (mandatory in transformer-based LLMs)
   - Source: hidden-state embeddings (final layer for MVP).
   - Semantic drift velocity is computed by pooling the last N token embeddings and measuring L2 distance to the
     previous pooled window, normalized by token delta. Low drift means semantic continuity; high drift means
     semantic motion or instability.

3) Attention stage (optional, if exposed by runtime)
   - Source: attention matrices by head/position.
   - Attention concentration can be measured via entropy or Gini. If attention is unavailable, we approximate
     concentration using entropy and top-K mass as proxies.

These are standard statistical measures of uncertainty, separation, and representation drift. Nothing here requires
training-time access, gradients, or proprietary internals.

## What the demo is proving

- Every metric in the visualization is grounded in standard inference artifacts (logits, embeddings, optional attention).
- A deterministic, monotonic mapping from these metrics into 3D space yields a stable trajectory that reflects model
  uncertainty (entropy), decisiveness (margin), and semantic motion (drift) in real time.
- Regime boundaries ("tunnel", "spread", "vibe", etc.) are hypotheses layered on top of measured signals and can be
  tested with prompt-controlled experiments and clustering.
- Live observability is feasible without patching the runtime; deeper hooks can reduce latency or overhead later.

## What it does

- Calls `/completion` in small chunks (`--chunk-size`) to approximate near-live sampling without patching llama.cpp.
- Uses `n_probs` to compute entropy, logit margin, and top-K mass per generated token.
- Optionally calls `/embedding` on a rolling token window to compute semantic drift.
- Optionally emits normalized 3D coordinates (the "snake") for direct visualization.
- Emits JSONL records to stdout (one per telemetry sample) with a versioned schema.

## 3D trajectory mapping (the "snake")

For wider audiences, the demo is most compelling when the metrics become a visible, stable trajectory in 3D space.
By default, the wrapper can emit normalized coordinates in [0, 1] that map directly to the three core signals:

- X: entropy (uncertainty axis)
- Y: logit margin (decisiveness axis)
- Z: drift (semantic motion axis)

The mapping is deterministic and monotonic (no learned projection). Each axis is normalized by a fixed max so that
the geometry is comparable across runs. A light EMA can smooth jitter without hiding regime shifts.

Defaults (configurable):

- entropy_max = 8.0 nats
- margin_max = 10.0 logprob delta
- drift_max = 2.0 (L2 per token)
- alpha = 0.2 (EMA smoothing factor)

Disable or tune with:

```
python3 llama_telemetry.py \
  --prompt "test" \
  --emit-xyz \
  --xyz-entropy-max 8.0 \
  --xyz-margin-max 10.0 \
  --xyz-drift-max 2.0 \
  --xyz-alpha 0.2
```

## Requirements

- `llama-server` running locally (default `http://127.0.0.1:8080`).
- Start server with `--n-probs 20` (or higher) and `--embedding` if you want drift.

Example server launch (adjust model path and params):

```
./llama-server -m /path/to/model.gguf --n-probs 20 --embedding
```

## Usage

```
python3 llama_telemetry.py \
  --base-url http://127.0.0.1:8080 \
  --prompt "Write a short paragraph about telemetry." \
  --n-predict 128 \
  --chunk-size 8
```

## Live 3D visualizer (browser)

The viewer is a local web page powered by Three.js with mouse/trackpad orbit controls.
It consumes JSONL from stdin or a file, streams it over Server-Sent Events, and renders the
3D trajectory with token labels that always face the camera.

Start/stop the visualizer server:

```
./viz_server.sh start
```

Open:

```
http://127.0.0.1:8765
```

Use the UI to set the llama.cpp endpoint and prompt, then click Start.

Run telemetry and the visualizer together (pipe mode):

```
python3 llama_telemetry.py --prompt "test" --sample-tokens 1 --emit-xyz \
  | python3 viz_server.py --port 8765
```

Then open:

```
http://127.0.0.1:8765
```

Playback pacing is intentionally limited to ~6 tokens/sec (configurable via URL):

```
http://127.0.0.1:8765/?rate=6&scale=10&maxPoints=2000&bloomStrength=2.2&bloomRadius=0.65&bloomThreshold=0.55&labels=0
```

Optional:

```
python3 llama_telemetry.py \
  --prompt-file prompt.txt \
  --topk 20 \
  --sample-tokens 8 \
  --sample-ms 150 \
  --window-size 64 \
  --embeddings
```

Pass extra generation params (temperature, top_p, etc) via JSON:

```
python3 llama_telemetry.py \
  --prompt "test" \
  --params generation.json
```

## JSONL schema (proposed, v1.0)

A `session_start` record is emitted first unless `--no-schema` is set. It includes the schema and units.
Each telemetry record uses numeric types and units described in that schema.

Record types:

- `session_start`
  - `t_ms` (int, ms_since_start)
  - `schema_version` (string)
  - `schema` (object)
  - `units` (object)
  - `config` (object)
  - `model` (object)

- `telemetry`
  - `t_ms` (int, ms_since_start)
  - `token_index` (int, token_index)
  - `token_id` (int, vocab_id)
  - `token_text` (string)
  - `position.x` (float, normalized_0_1) [present when emit-xyz is enabled]
  - `position.y` (float, normalized_0_1) [present when emit-xyz is enabled]
  - `position.z` (float, normalized_0_1) [present when emit-xyz is enabled]
  - `metrics.entropy` (float, nats)
  - `metrics.logit_margin` (float, logprob_delta)
  - `metrics.topk_mass` (float, probability)
  - `metrics.drift` (float, l2_per_token)
  - `sample_reason` (string enum: interval_tokens | interval_time | event)
  - `event` (string enum: entropy_spike | margin_drop | null)

- `session_end`
  - `t_ms` (int, ms_since_start)
  - `token_count` (int)
  - `stop` (bool)

## Notes and limitations (MVP)

- Entropy is computed from top-K probabilities plus a single residual bucket (1 - topK). This is a lower-bound approximation of full-distribution entropy unless K is very large.
- Drift uses `/embedding` on the last `--window-size` tokens, which requires an extra embedding call per telemetry sample.
- `/completion` streaming does not include per-token probabilities in current llama.cpp, so the wrapper uses short non-streamed requests to approximate near-live sampling.
- XYZ coordinates are normalized by fixed maxima and smoothed with an EMA. Adjust `--xyz-*` knobs if your model operates outside the defaults.
- The visualizer loads Three.js from a CDN. If you need offline demo support, vendor the library locally.

## Pinning llama.cpp commit

The wrapper reads a pinned commit hash from `llama_cpp_commit.txt`. You can auto-fill it from the running server:

```
python3 llama_telemetry.py --prompt "test" --n-predict 1 --pin-commit
```

If `/props` exposes `build_info`, the script will write the commit hash into `llama_cpp_commit.txt`.
