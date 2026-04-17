# LLM Snake Scope (2D)

A local analyst-focused visualizer for autoregressive runs.

This version is intentionally **2D + time-first**:
- stable PCA projection of token vectors
- optional 3D orbit view for the same runs
- replay controls and freeze frame
- run-vs-run diff with alignment + delta summary
- decoder alert markers
- clickable token inspection (top-N alternatives)
- token branching and regeneration
- backend probing and provenance display

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
- Drag in `3D Orbit`: rotate the projection
- Click token: open alternatives popup
- `1..5`: choose alternative token in popup
- `Enter`: confirm branch generation

## What Changed vs Earlier Versions

- Replaced the old camera-heavy 3D workflow with a cleaner 2D default plus an optional 3D orbit mode.
- Added deterministic PCA projection with cached coordinates.
- Added run management (`Run A`, `Run B`), diff overlay, and alignment summary.
- Added per-token chips with top-N alternatives and branch regeneration.
- Reframed the time-series panels around decoder diagnostics: uncertainty, choice gap, uncertainty jump, repetition pressure, decoder risk, and geometry motion.
- Added plain-language summary panels and backend/provenance checks.
- Added an experimental `Live Branch Lab` subpage for baseline-vs-corrected decoder-risk intervention runs.

## Implemented on 2026-04-18

The current repo now includes the decoder-diagnostics rewrite described above.

- Added `Probe Backend` and backend provenance so the UI shows whether token probabilities and embeddings are really available.
- Added plain-language summaries (`Current Risk`, `Why Now`, `What To Say`) next to the plots.
- Replaced the old metric stack with decoder-side diagnostics: uncertainty, choice gap, uncertainty jump, repetition pressure, decoder risk, and geometry motion.
- Restored 3D as an optional orbit mode instead of the main workflow.
- Fixed geometry provenance so a run requested with `real` embeddings falls back and displays `placeholder vectors` when the backend does not actually expose `/embedding`.
- Saved the next project backlog in [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).
- Added a separate `Live Branch Lab` page that runs baseline vs corrected decode and links the resulting runs back into the main visualizer.
- Added live-ish incremental updates for `Live Branch Lab` via SSE so the page shows baseline/corrected progress before the final response returns.
- Measured the current latency envelope on a fixed `max_tokens=18` / single-intervention benchmark:
  - backend mean completion time: about `4.08s` before SSE vs `4.12s` after SSE
  - first visible UI update: about `4.07s` before SSE vs `0.08s` after SSE
  - first visible token text: about `0.23s` after SSE

## Visual Evidence

### 2D Main View

This capture shows the plain-language summary, backend/provenance panel, run comparison panel, and the decoder diagnostics.

![2D main view evidence](docs/evidence/2026-04-18-main-ui-2d.png)

### 3D Orbit Mode

This capture shows the optional 3D orbit projection for the same run family.

![3D orbit view evidence](docs/evidence/2026-04-18-orbit-ui-3d.png)

## Live Branch Lab

An experimental second page is now available at `./live.html`.

What it does:
- runs a baseline decode and a corrected decode with the same prompt/settings
- watches decoder-risk token-by-token
- when the risk stays high long enough, it replays from an earlier prefix and evaluates a few alternative next-token branches
- chooses a lower-risk branch only if the measured reduction is large enough

Current limitation:
- this is a `prefix_replay` intervention, not exact KV-state rollback
- lower decoder risk after branching is useful evidence, but it is not proof of factual correctness

The repeatable evaluation plan for this page lives in [docs/LIVE_BRANCH_EVAL.md](docs/LIVE_BRANCH_EVAL.md).

## Research Recalibration (2026-04-17)

This project originally leaned on a stronger hypothesis than the current literature supports:

- abrupt changes in entropy, logit margin, vector velocity, or curvature might reveal the point where the model "starts hallucinating"

After reviewing recent peer-reviewed work, we no longer treat that as a calibrated claim.

What we now believe:
- single-pass token-level uncertainty signals (`entropy`, `logprob`, `margin`, `top-k mass`) contain useful uncertainty information, but are not strong enough on their own to reliably separate factual from non-factual generation
- semantic uncertainty across multiple sampled continuations is more informative than raw token entropy alone
- claim-level or span-level factuality is a better target than generic token anomaly detection
- if white-box access is available, cross-layer, hidden-state, and attention-derived features are more promising than final-layer-only telemetry
- broad hallucination detectors can fail out of distribution, so calibration and OOD validation matter as much as the detector itself

Operational consequence for this tool:
- the current charts should be read as uncertainty / instability diagnostics
- `regime_markers` are investigation cues, not factuality labels
- `velocity` and `curvature` are especially easy to over-interpret when the run uses placeholder vectors instead of real embeddings

Evidence base behind this correction:
- Kuhn et al., Nature 2024, "Detecting hallucinations in large language models using semantic entropy":
  https://www.nature.com/articles/s41586-024-07421-0
- Li et al., NAACL Findings 2025, "HALLUCANA: Fixing LLM Hallucination with A Canary Lookahead":
  https://aclanthology.org/2025.findings-naacl.12/
- Wu et al., NAACL Findings 2025, "Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models":
  https://aclanthology.org/2025.findings-naacl.217/
- Qin et al., ACL 2025, "Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation":
  https://aclanthology.org/2025.acl-short.93/
- Han et al., EMNLP Findings 2025, "Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation":
  https://aclanthology.org/2025.findings-emnlp.880/
- Kim et al., EMNLP 2025, "Detecting LLM Hallucination Through Layer-wise Information Deficiency":
  https://aclanthology.org/2025.emnlp-main.1644/
- Dubanowska et al., EMNLP Findings 2025, "Representation-based Broad Hallucination Detectors Fail to Generalize Out of Distribution":
  https://aclanthology.org/2025.findings-emnlp.952/
- Kulkarni et al., EMNLP Findings 2025, "Evaluating Evaluation Metrics - The Mirage of Hallucination Detection":
  https://aclanthology.org/2025.findings-emnlp.1035/

## Roadmap Toward Better Hallucination-Onset Detection

The goal is no longer "entropy spike means hallucination started here".
The goal is:

- estimate where the output becomes unsupported or confabulatory
- attach an explicit confidence score to that estimate
- make the estimate inspectable against evidence, not only against telemetry

What current science suggests is a more adequate direction:
- combine token-level uncertainty with claim-level / span-level factuality analysis
- prefer semantic uncertainty over multiple sampled continuations over single-pass entropy alone
- add retrieval or evidence-grounded verification so the UI can distinguish "uncertain" from "unsupported"
- when model internals are available, add cross-layer, hidden-state, and attention-derived features
- validate on human-aligned factuality labels and OOD settings, not only on in-domain heuristics

Planned expansion path:

### Phase 0: Make the Current Instrument Honest

- expose whether each run used real embeddings or placeholder vectors
- surface backend capability flags (`n_probs`, embeddings, strict token forcing support) in the UI and exported JSON
- label `regime_markers` explicitly as anomaly markers, not hallucination markers
- log additional decoder-side signals that are cheap and honest: repetition rates, stop reasons, sampled-vs-greedy path, top-k mass, entropy slope

### Phase 1: Add Black-Box Factuality Probes

- branch the same prefix into multiple stochastic continuations and compute semantic uncertainty around the next claim/span
- segment output into candidate claims and attach risk at the claim level rather than only the token level
- add retrieval-backed evidence checks for extracted claims
- render "supported / unsupported / unknown" overlays next to the current telemetry plots

### Phase 2: Add White-Box Research Mode

- capture cross-layer entropy or information-deficiency style features
- log hidden-state probes and attention/context-allocation signals around suspected onset regions
- compare onset estimates from final-layer telemetry versus multi-layer features
- measure whether white-box signals provide earlier and better-calibrated warnings than `entropy` / `margin` alone

### Phase 3: Calibrate the Detector

- build a labeled evaluation set where annotators mark the earliest unsupported claim/span
- train or calibrate a probabilistic onset score instead of showing only heuristic spikes
- report proper evaluation metrics such as AUROC, F1, ECE, lead time, and OOD degradation
- keep the visualizer usable as a microscope even when the detector is uncertain

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
      "uncertainty": 0.31,
      "prob_gap": 0.64,
      "entropy_delta": 0.08,
      "repetition_pressure": 0.0,
      "decoder_risk": 0.22,
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
    "decoder_alerts": [
      {"index": 42, "risk": 0.71, "reasons": ["high_uncertainty", "weak_choice_gap"]}
    ],
    "regime_markers": [
      {"index": 42, "reasons": ["velocity_spike", "entropy_slope_spike"]}
    ]
  },
  "summary": {
    "token_count": 256,
    "entropy_avg": 1.33,
    "velocity_max": 0.82,
    "decoder_risk_max": 0.71,
    "decoder_alert_count": 3,
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
- Optional `3D Orbit` mode projects the same run set into PCA-3 and allows interactive rotation.
- In single-run mode: PCA fitted on Run A.
- In diff mode: PCA fitted on concatenated vectors from Run A + Run B, so both trajectories share one coordinate frame.
- Projection output is cached by `(run ids + token lengths)` for smooth replay.

### Regime Detection Heuristic

Markers are generated from token-index series:
- embedding velocity spikes (`||v[i]-v[i-1]||`)
- entropy slope spikes (`|H[i]-H[i-1]|`)

Thresholds use a simple `mean + 2*std` rule with a minimum index gap to avoid marker spam.

These markers are currently heuristic anomaly indicators only.
They should not be interpreted as a calibrated answer to "hallucination started here".

### Decoder Diagnostics

The primary decoder-side signals are now:
- normalized next-token uncertainty
- probability gap between the top two token choices
- uncertainty jump from the previous token
- repetition pressure from recent local token reuse
- transparent decoder-risk blend over the above signals

These are still decoder-side heuristics, not claim-level factuality checks.

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
- Claim/span extraction and evidence-grounded factuality checks.
- Multi-sample semantic uncertainty rather than single-pass entropy only.
- White-box telemetry capture for cross-layer / hidden-state analysis when the backend allows it.
- Detector calibration and OOD evaluation against human-aligned factuality labels.
