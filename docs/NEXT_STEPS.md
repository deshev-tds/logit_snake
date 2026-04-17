# Next Steps

Saved on 2026-04-18 after the decoder-diagnostics rewrite, live experiment streaming, and Lab token-telemetry pass.

## Immediate

- Expose the exact decoder-risk formula, weights, and active thresholds in the UI and exported JSON so risk readouts stay inspectable.
- Add claim/span extraction on top of token replay so the tool can point to candidate unsupported statements, not only risky decoder regions.
- Run multi-sample continuations from the same prefix and compute semantic uncertainty around the next claim/span.
- Add retrieval-backed evidence checks and render `supported / unsupported / unknown` next to the current plots.

## Workflow

- Persist runs and screenshots to disk from the UI instead of keeping them mostly in memory.
- Add import/export support for multiple named sessions so long investigations can be saved and resumed.
- Add a lightweight screenshot refresh script for reproducible README evidence.

## Research

- Build a small labeled set where annotators mark the earliest unsupported claim/span.
- Calibrate the current decoder-risk heuristic against those labels and report AUROC, F1, ECE, and lead time.
- Compare black-box semantic uncertainty against white-box cross-layer / hidden-state signals when the backend allows it.
- Test out-of-distribution robustness across models, prompts, and domains before making stronger onset claims.
