# Research Corpus

This directory stores the local artifacts behind the `2026-04-18` project pivot from telemetry-only onset cues toward claim-level harnessing.

## Contents

- `2026-04-18-harness-survey.md`
  - survey memo mapping papers and released code to concrete borrowing decisions for this repo
- `papers/`
  - downloaded PDF copies of the papers used for the current methodological pivot
- `code/`
  - vendored snapshots of public paper repos that were inspected locally
  - upstream `.git` metadata was intentionally stripped so these folders are committed as ordinary source snapshots, not nested repos
- `notes/paper_extracts.txt`
  - raw extraction notes taken while reading the papers and comparing them with the current codebase

## Papers Steering The Current Direction

- Agrawal et al. 2024, `Do Language Models Know When They're Hallucinating References?`
- Farquhar et al. 2024, `Detecting hallucinations in large language models using semantic entropy`
- Liang et al. 2024, `Learning to Trust Your Feelings`
- Liu et al. 2025, `Long-form Hallucination Detection with Self-elicitation`
- Gupta et al. 2025, `Consistency Is the Key`
- Han et al. 2025, `Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation`
- Azaria and Mitchell 2023, `The Internal State of an LLM Knows When It's Lying`

## Why It Lives In The Repo

The current lab work now depends on methodology details that are easy to misremember if they live only in external links.

Keeping the PDFs, notes, and inspected code snapshots locally makes it easier to:
- trace each product decision back to a paper or released implementation
- re-open the exact evidence base when the harness changes
- compare future white-box work against the current black-box baseline
