# LLM Snake Scope (2D)

A local analyst-focused visualizer for autoregressive runs.

This version is intentionally **2D + time-first**:
- stable PCA projection of token vectors
- replay controls and freeze frame
- run-vs-run diff with alignment + delta summary
- regime-shift markers
- clickable token inspection (top-N alternatives)
- token branching and regeneration

## Quick Start

1. Start your inference backend (llama.cpp-compatible HTTP API).
   - It should expose `/completion`.
   - For best token alternatives, enable `n_probs` support.
   - Optional: expose `/embedding` for real vectors.

2. Start the visualizer server:

```bash
./viz_server.sh start
```

3. Open:

```text
http://127.0.0.1:8765
```

4. Enter endpoint + prompt and click **Generate Run**.

## Controls

- `Space`: play / pause
- `Left` / `Right`: step timeline
- `B`: bookmark current token index
- `D`: toggle diff overlay
- Click token: open alternatives popup
- `1..5`: choose alternative token in popup
- `Enter`: confirm branch generation

## What Changed vs the Old 3D Version

- Removed Three.js 3D camera-centric workflow from the main UX.
- Added deterministic 2D projection with cached PCA coordinates.
- Added run management (`Run A`, `Run B`), diff overlay, and alignment summary.
- Added per-token chips with top-N alternatives and branch regeneration.
- Added actionable time-series metrics (entropy, margin, velocity, curvature).

## Run JSON Schema (Expected Format)

The app accepts and emits this schema (version `2.0`):

```json
{
  "schema_version": "2.0",
  "run_id": "run_abc123",
  "meta": {
    "label": "Run",
    "prompt": "...",
    "prompt_hash": "...",
    "timestamp": "2026-02-14T12:00:00+00:00",
    "base_url": "http://127.0.0.1:8080",
    "model": "...",
    "status": "complete",
    "generation_settings": {
      "max_tokens": 256,
      "chunk_size": 16,
      "top_n": 5,
      "n_probs": 20,
      "temperature": 0.7,
      "top_p": 0.95,
      "seed": 1234,
      "vector_mode": "placeholder",
      "vector_dim": 24
    },
    "branch": {
      "parent_run_id": "run_parent",
      "fork_index": 42,
      "chosen_alt_token": {
        "token_id": 123,
        "token_text": " alternative",
        "logprob": -2.11,
        "prob": 0.12
      },
      "forcing_strategy": "append_prefix_fallback",
      "timestamp": "2026-02-14T12:05:00+00:00"
    }
  },
  "tokens": [
    {
      "index": 0,
      "t": 12,
      "text": "Hello",
      "chosen_token_id": 123,
      "chosen_token_text": "Hello",
      "logprob": -0.03,
      "prob": 0.97,
      "entropy": 0.45,
      "margin": 3.2,
      "velocity": 0.0,
      "curvature": null,
      "embedding": [0.11, -0.05, 0.2, "..."],
      "topN": [
        {"token_id": 123, "token_text": "Hello", "logprob": -0.03, "prob": 0.97},
        {"token_id": 998, "token_text": "Hi", "logprob": -3.1, "prob": 0.045}
      ]
    }
  ],
  "analysis": {
    "regime_markers": [
      {"index": 42, "reasons": ["velocity_spike", "entropy_slope_spike"]}
    ]
  },
  "summary": {
    "token_count": 256,
    "entropy_avg": 1.33,
    "velocity_max": 0.82,
    "duration_ms": 8120
  },
  "bookmarks": [
    {"index": 42, "label": "tone shift", "timestamp": "2026-02-14T12:06:00+00:00"}
  ]
}
```

## Loading Two Runs for Diff Mode

1. Generate runs from the UI, and/or load run JSON via **Load Run JSON**.
2. Select runs in **Run A** and **Run B** dropdowns.
3. Enable **Diff Overlay**.
4. Read the summary panel:
   - average aligned distance
   - max distance spike
   - first shift candidate over threshold

### Alignment Strategy

- If lengths match: index-to-index alignment.
- If lengths differ: monotonic sliding-window nearest-neighbor alignment in projected 2D space.

## Token Branching

### How to Trigger

1. Click a token chip in the token panel.
2. Choose an alternative token (mouse or keys `1..5`).
3. Press **Create Branch** or `Enter`.

### Backend Requirements

Preferred:
- `/completion` returns per-token probability data (`n_probs` / top-logprobs style).

Fallback behavior when detailed alternatives are unavailable:
- deterministic approximation for top-N alternatives.

### Branch Regeneration Strategy

Current implementation uses a deterministic, backend-compatible fallback:
1. Keep original prompt + original token prefix up to `i-1`.
2. Append chosen alternative token `i` to that prefix.
3. Continue generation from `i+1` onward with same generation settings.

Branch metadata is persisted in `meta.branch`.

## API Endpoints

- `GET /api/status`
- `GET /api/runs`
- `GET /api/run/<run_id>`
- `POST /api/generate` (alias: `/api/start`)
- `POST /api/branch`
- `POST /api/stop`
- `POST /api/import-run`
- `GET /stream` (SSE run events)

## Design Notes

### Projection Choice

- High-dimensional vectors are projected to 2D with deterministic PCA.
- In single-run mode: PCA fitted on Run A.
- In diff mode: PCA fitted on concatenated vectors from Run A + Run B, so both trajectories share one coordinate frame.
- Projection output is cached by `(run ids + token lengths)` for smooth replay.

### Regime Detection Heuristic

Markers are generated from token-index series:
- embedding velocity spikes (`||v[i]-v[i-1]||`)
- entropy slope spikes (`|H[i]-H[i-1]|`)

Thresholds use a simple `mean + 2*std` rule with a minimum index gap to avoid marker spam.

### Branch Forcing Approach

- Uses append-prefix continuation fallback for broad backend compatibility.
- Keeps settings deterministic (`seed`, `temperature`, `top_p`, etc.) from the parent run.
- Stores branch lineage (`parent_run_id`, `fork_index`, chosen alternative, timestamp).

## Legacy Compatibility

- `jsonl` telemetry logs from the old pipeline can still be loaded.
- A conversion layer maps legacy telemetry records into the new run schema.

## Known Gaps / TODO

- Optional strict token-forcing via backend-native logit bias when reliably supported.
- More robust DTW-style alignment for very different-length outputs.
- Explicit run persistence on disk (currently in-memory unless exported/imported).
