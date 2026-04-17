# Next Steps

Saved on 2026-04-18 after the lecture/demo-oriented UX and decoder-diagnostics rewrite.

## Immediate

- Add claim/span extraction on top of token replay so the tool can point to candidate unsupported statements, not only risky decoder regions.
- Run multi-sample continuations from the same prefix and compute semantic uncertainty around the next claim/span.
- Add retrieval-backed evidence checks and render `supported / unsupported / unknown` next to the current plots.
- Expose the exact decoder-risk formula and thresholds in the UI so the lecture narrative stays inspectable.

## Productization

- Add a presentation mode with larger labels, bigger risk callouts, and a cleaner one-click demo flow.
- Save backend probe results and run provenance directly inside exported run JSON.
- Persist runs and screenshots to disk from the UI instead of keeping them mostly in memory.
- Add import/export support for multiple named sessions so conference demos can be prepared ahead of time.

## Research

- Build a small labeled set where annotators mark the earliest unsupported claim/span.
- Calibrate the current decoder-risk heuristic against those labels and report AUROC, F1, ECE, and lead time.
- Compare black-box semantic uncertainty against white-box cross-layer / hidden-state signals when the backend allows it.
- Test out-of-distribution robustness across models, prompts, and domains before making stronger onset claims.

## Repo Hygiene

- Keep generated Python bytecode out of git.
- Add a lightweight screenshot refresh script for reproducible README evidence.
