# Live Branch Evaluation Protocol

This document defines how to test whether the experimental `Live Branch Lab` actually helps reduce risky decoder trajectories in a repeatable way.

## What The Current Lab Does

- It decodes token-by-token (`n_predict=1`) so risk can be observed at each step.
- When decoder risk stays above a threshold for a configurable number of tokens, it rewinds approximately via `prefix_replay`.
- It evaluates a small set of alternative next tokens with short lookahead runs and picks the lower-risk continuation if the reduction is large enough.

This is **not** proof of factual correctness.
It is currently proof only of **decoder-risk intervention**.

## Core A/B Setup

For every prompt in the test suite, run:

1. Baseline decode
2. Corrected decode with the same prompt, model, and seed

Keep fixed:
- model file / model ID
- llama.cpp server version
- seed schedule
- temperature / top_p / samplers
- max tokens
- risk threshold and intervention policy

Report paired results, not anecdotal single runs.

## Test Groups

### Group A: No-Harm Control

Prompts where the model is usually stable and factual:
- capitals
- well-known university facts
- short biographies of very famous people
- simple geography

Goal:
- interventions should rarely trigger
- corrected mode should not reduce answer quality or obvious correctness

### Group B: Drift-Prone Factual Generation

Prompts that often drift into invented detail:
- biographies of moderately known people
- institution histories
- “write 3-5 factual sentences about X” prompts
- “include one surprising detail” prompts

Goal:
- corrected mode should reduce max decoder risk and alert count
- corrected mode should reduce unsupported claims after intervention

### Group C: Long-Form Pressure

Prompts that ask for longer factual continuations:
- mini encyclopedia paragraphs
- timelines
- “give 5 facts” prompts
- comparisons between entities

Goal:
- measure whether interventions happen early enough to matter
- measure latency and token-cost overhead

## Metrics

### Decoder-Side

- max decoder risk
- number of decoder alerts
- intervention count
- average risk in the post-trigger window
- branch acceptance rate

### Factuality

- claim support rate
- unsupported-claim count
- earliest unsupported claim/span
- intervention precision:
  how often the first intervention lands near the first unsupported claim/span
- intervention win rate:
  how often corrected output is more supported than baseline

### Cost / UX

- added latency
- extra tokens generated for lookahead branches
- percent of runs with no accepted branch
- percent of no-harm prompts that were unnecessarily altered

## Labeling Strategy

Use claim/span labels, not only answer-level labels.

For each output:
- segment into claims
- mark each claim as `supported`, `unsupported`, or `unclear`
- record the earliest unsupported claim/span

The key measurement is not just “was the answer bad?”
The key measurement is:
- did the intervention happen before or near the earliest unsupported claim/span?

## Recommended First Evidence Package

Start with:
- 50 no-harm prompts
- 50 drift-prone prompts
- 20 seeds per prompt

That gives a paired evaluation large enough to show whether the effect is real or mostly anecdotal.

## Stronger Follow-Up

After the decoder-risk PoC is stable:
- add claim extraction
- add retrieval-backed support checks
- compare `prefix_replay` against a more exact slot-restore path when llama.cpp slots are available
- report OOD behavior across at least two different models
