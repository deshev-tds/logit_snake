# Next Steps

Saved on 2026-04-18 after the decoder-diagnostics rewrite, live experiment streaming, and the first claim-level harness pass.

## Immediate

- Expose the exact decoder-risk formula, weights, and active thresholds in the UI and exported JSON so risk readouts stay inspectable.
- Harden the current claim boundary detector, de-contextualization pass, and key-fact extraction so the harness focuses the right sentence more consistently.
- Add reusable probe templates and claim-family scoring for books, papers, standards, case law, identifiers, and quoted spans.
- Run multi-sample continuations from the same prefix and compute semantic uncertainty around the next claim/span.
- Add retrieval-backed evidence checks and render `supported / unsupported / unknown` next to the current plots.
- Reduce black-box harness latency through better probe dedupe, smaller secondary prompts, and optional async escalation.

## Workflow

- Persist runs and screenshots to disk from the UI instead of keeping them mostly in memory.
- Add import/export support for multiple named sessions so long investigations can be saved and resumed.
- Add a lightweight screenshot refresh script for reproducible README evidence.

## Research

- Build a small labeled set where annotators mark the earliest unsupported claim/span.
- Calibrate the current decoder-risk plus claim-harness stack against those labels and report AUROC, F1, ECE, and lead time.
- Compare black-box semantic uncertainty against white-box cross-layer / hidden-state signals when the backend allows it.
- Test out-of-distribution robustness across models, prompts, and domains before making stronger onset claims.
