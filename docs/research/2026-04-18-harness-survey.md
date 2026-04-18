# Harness Survey for Live Hallucination Detection and Correction

This note records what was downloaded and inspected on `2026-04-18` for the next iteration of `Snake Scope` / `Live Branch Lab`.

The code directories listed below are committed as vendored snapshots inside this repo. Upstream `.git` metadata was intentionally stripped so the snapshots behave like ordinary project files.

## Local Artifacts

Downloaded PDFs:

- `docs/research/papers/agrawal_2024_hallucinating_references.pdf`
- `docs/research/papers/farquhar_2024_semantic_entropy.pdf`
- `docs/research/papers/liang_2024_trust_your_feelings.pdf`
- `docs/research/papers/liu_2025_self_elicitation.pdf`
- `docs/research/papers/gupta_2025_consistency_is_key.pdf`
- `docs/research/papers/han_2025_simple_factuality_probes.pdf`
- `docs/research/papers/liu_2025_attention_guided_self_reflection.pdf`
- `docs/research/papers/azaria_mitchell_2023_internal_state_lying.pdf`
- `docs/research/papers/alnuhait_2025_factcheckmate.pdf`

Cloned code repos:

- `docs/research/code/hallucinated-references`
- `docs/research/code/long_hallucinations`
- `docs/research/code/dreamcatcher`
- `docs/research/code/SelfElicit`
- `docs/research/code/fact-probe`

Supporting extraction notes:

- `docs/research/notes/paper_extracts.txt`

## Bottom Line

The most borrowable path for this repo is not a generic prompt like "are you sure you are not lying?".

The strongest black-box direction, compatible with `llama.cpp` as we use it now, is:

1. Use telemetry only as a trigger.
2. Stop at a claim or span boundary.
3. Extract check-worthy atomic claims or key facts.
4. Ask targeted factual probes about those facts.
5. Score internal consistency across answers, not only verbal confidence.
6. Rewrite or abstain when the probe evidence says the claim is unstable.

The strongest white-box direction, but not directly available through the current `llama.cpp` server surface, is:

1. hidden-state factuality probes
2. pre-decode hallucination detection
3. attention-guided query slicing

## Paper Inventory

| Paper | What it actually does | Code status | Borrow value for this repo |
| --- | --- | --- | --- |
| Agrawal et al. 2024, "Do Language Models Know When They're Hallucinating References?" | Detects hallucinated references by asking direct and indirect questions and measuring consistency across answers. | Official repo found and cloned: `microsoft/hallucinated-references` | Very high. The indirect-question pattern is a direct fit for `Live Branch Lab`. |
| Farquhar et al. 2024, "Detecting hallucinations ... using semantic entropy" | Generates multiple answers, clusters them by meaning using entailment, and computes entropy over semantic clusters. | Official reproduction repo found and cloned: `jlko/long_hallucinations` | High. Expensive, but a strong black-box secondary checker for risky claims. |
| Liang et al. 2024, "Learning to Trust Your Feelings" | Measures whether models know their knowledge boundary; uses DreamCatcher to combine self-consistency, answer similarity, and hidden-state probes; then uses RLKF for training. | Official repo found and cloned: `liangyuxin42/dreamcatcher` | Medium. Good as harness inspiration; less direct for live local inference because much of it is ranking/training-oriented. |
| Liu et al. 2025, "Long-form Hallucination Detection with Self-elicitation" | Filters check-worthy sentences, de-contextualizes them, extracts knowledge, then judges with self-elicited thoughts plus a self knowledge graph. | Official repo found and cloned: `lzhmarkk/SelfElicit` | Very high. This is the cleanest post-generation harness we can adapt right now. |
| Gupta et al. 2025, "Consistency Is the Key" | Extracts key facts, turns them into factual probes, regenerates answers, and checks consistency within and across models. | Paper says code exists, but I did not find a trustworthy public repo from the paper/web trail. | Very high conceptually. Especially useful for entity-centric probes inside our lab. |
| Han et al. 2025, "Simple Factuality Probes Detect Hallucinations ..." | Trains lightweight classifiers on hidden states to predict support at claim level. | Repo found and cloned: `JThh/fact-probe` | High research value, but future work for us unless backend exposes hidden states. |
| Alnuhait et al. 2025, "FACTCHECKMATE" | Predicts hallucination before decoding from hidden states and then intervenes by changing those hidden states. | Paper says code, data, and checkpoints will be released; I did not find a trustworthy public repo to inspect. | High future value, but only for a white-box backend path. |
| Liu et al. 2025, "Attention-guided Self-reflection ..." | Splits the query into attentive vs non-attentive parts using attention contribution, then compares consistency scores from both views. | No reliable public code repo found from the paper/web trail. | Medium. Interesting if we ever expose attention maps or token salience from the backend. |
| Azaria and Mitchell 2023, "The Internal State of an LLM Knows When It's Lying" | Trains a classifier on hidden-layer activations to predict truthfulness. | No reliable public code repo found from the paper/web trail. | Foundational, but mostly white-box background for this project. |

## What the Codebases Show

### 1. Agrawal 2024: indirect factual probes are simple and practical

Relevant code:

- `docs/research/code/hallucinated-references/code/src/cross_question.py`
- `docs/research/code/hallucinated-references/code/src/consistency_check.py`

What is reusable:

- Ask the model for a detail implied by the claim instead of asking "is this false?"
- Sample multiple independent answers.
- Compare the answers pairwise with a separate overlap or agreement prompt.
- Use the consistency score as evidence, not as proof.

Why this matters here:

- Our current lab already has a rollback trigger.
- The missing piece is a better judge for the risky span.
- Indirect probes are cheap enough to run only after a telemetry trigger.

### 2. Semantic entropy is stronger than token entropy, but more expensive

Relevant code:

- `docs/research/code/long_hallucinations/hallucination.py`
- `docs/research/code/long_hallucinations/models.py`

What is reusable:

- Break a long answer into factoids.
- Generate questions that each factoid could have answered.
- Re-answer those questions multiple times.
- Cluster answers by semantic equivalence via entailment.
- Compute entropy over meanings, not surface forms.

Why this matters here:

- This is a much better "escalation path" than raw token entropy.
- It is too heavy to run on every token in live mode.
- It is appropriate as an optional secondary pass on a small risky span.

### 3. DreamCatcher is useful mainly as a scoring pattern, not as a drop-in product

Relevant code:

- `docs/research/code/dreamcatcher/dreamcatcher.py`
- `docs/research/code/dreamcatcher/scorer/probe_scorer.py`
- `docs/research/code/dreamcatcher/readme.md`

What is reusable:

- Combine several weak signals instead of trusting one scalar.
- Distinguish "correct / uncertain / wrong" as separate buckets.
- Treat self-consistency as a feature, not as a final answer.

What is less reusable:

- It assumes either a correct answer or an offline labeling workflow in several places.
- The reward-learning side is outside the scope of our current live lab.

### 4. SelfElicit is the closest match to the current product direction

Relevant code:

- `docs/research/code/SelfElicit/methods/common/classify.py`
- `docs/research/code/SelfElicit/methods/common/context.py`
- `docs/research/code/SelfElicit/methods/elicit/extract.py`
- `docs/research/code/SelfElicit/methods/elicit/main.py`
- `docs/research/code/SelfElicit/methods/elicit/self_kg.py`

What is reusable:

- Sentence filtering for check-worthy content.
- De-contextualization when pronouns make a sentence hard to verify.
- Knowledge extraction from a sentence before judging it.
- Self-generated context bank / self knowledge graph to retain earlier validated knowledge.
- Iterative judge loop rather than one-shot confidence elicitation.

Why this matters here:

- It solves exactly the failure mode of long-form drift.
- It already separates claim extraction from claim judgment.
- It gives us a way to carry validated earlier claims into later correction loops.

### 5. CONFACTCHECK gives the best black-box harness template

Methodology from the paper:

- extract key entities or tags from generated text
- formulate factual questions around those key facts
- regenerate answers to those questions
- compare regenerated facts with the original output
- use intra-model and cross-model consistency as the score

What is reusable:

- a key-fact registry per sentence or claim
- templated factual probes per entity type
- contradiction / mismatch scoring for regenerated answers

Note:

- I did not find a trustworthy public implementation to inspect locally, so the method here is taken from the published paper, not from code.

### 6. Fact probes are real, but they need white-box access

Relevant code:

- `docs/research/code/fact-probe/long_fact_probes/train.py`
- `docs/research/code/fact-probe/long_fact_probes/eval.py`
- `docs/research/code/fact-probe/long_fact_probes/predict.py`

What is reusable in principle:

- claim-level support prediction from hidden states
- simple linear probes over specific layer groups
- rejection curves and calibrated confidence on claims

Why this does not drop into our current setup:

- the current `llama.cpp` server path we use does not expose hidden states
- our browser tool is currently black-box over HTTP plus token probabilities

### 7. FACTCHECKMATE is the right white-box mental model for future intervention

Methodology from the paper:

- predict hallucination before decoding from input-side hidden states
- if risk is high, intervene by modifying hidden states
- generate a more factual continuation with low extra inference overhead

Why it matters:

- conceptually, this is much closer to the "live correction" goal than our current prefix replay
- operationally, it is not an honest near-term target for the current stack because we do not have hidden-state access through the backend we are using
- it is a strong reference for a future `llama.cpp` fork or custom local runner

## What We Should Borrow Now

### Priority A: black-box harness, compatible with `llama.cpp` now

1. Claim boundary detector
   Trigger correction only after sentence or list-item boundaries, not at arbitrary mid-word risk spikes.

2. Check-worthy sentence filter
   Borrow the `SelfElicit` idea of filtering objective assertions from boilerplate or prompt restatement.

3. De-contextualization pass
   If a risky claim depends on pronouns or prior context, rewrite it into a standalone sentence before probing it.

4. Key-fact extraction
   Pull out entities, dates, numbers, titles, locations, identifiers, and quoted spans from the risky sentence.

5. Probe family instead of one self-report question
   For each risky claim, ask several specific probes.
   Example for a book-like claim:
   - Who is the author?
   - Who is the publisher?
   - What year was it published?
   - What is the ISBN?
   - Give one chapter title or one verifiable bibliographic detail.

6. Consistency scoring
   Score agreement between:
   - the original claim
   - repeated answers from the same model
   - optionally a second model if available

7. Rewrite policy
   If the probe score is weak, rewrite conservatively:
   - remove unsupported details
   - hedge
   - abstain

### Priority B: optional expensive verifier mode

Use semantic entropy only on a small risky claim bundle, not on the whole generation:

- sample 3 to 5 alternative answers
- cluster by semantic equivalence
- use entropy as a secondary decision signal

## What We Should Not Borrow As-Is

1. A single prompt like "Are you sure you are not lying?"
   The literature does not support that as a reliable detector.

2. Geometry-based motion as a factuality detector
   Useful for visualization, not for factual claims.

3. Pure token-level telemetry as the final judge
   Good trigger, weak verifier.

4. White-box methods without backend support
   Hidden-state probes and hidden-state intervention are promising, but not honest to present as "implemented" on the current HTTP-only path.

## Recommended Product Direction for This Repo

### Near-term: `Live Branch Lab` becomes a two-stage harness

Stage 1: telemetry trigger

- keep current `decoder_risk` logic
- keep live-ish UI
- detect risky windows and rollback candidates

Stage 2: claim verifier

- segment current text into claims
- choose the risky claim or nearest risky claim bundle
- run targeted probes
- compute a harness score:
  - probe consistency
  - contradiction count
  - optional semantic entropy
- only then decide to:
  - accept branch
  - rewrite the span
  - abstain / soften

### Mid-term: add correction loops with explicit stop conditions

- manual loop count: `1..N`
- stop if no claim-level improvement
- stop if the same claim re-triggers
- stop if rewritten text becomes shorter but not better supported

### Longer-term: white-box path

Only after backend changes:

- expose hidden states or attention maps
- train lightweight factuality probes
- optionally add pre-decode risk prediction

## Concrete Build Order

1. Add claim extraction and check-worthy filtering to `Live Branch Lab`.
2. Add a factual probe library for common risky entity types.
3. Add a consistency scorer and contradiction counter.
4. Add a correction policy: rewrite unsupported specifics or abstain.
5. Add optional semantic-entropy verification for high-risk cases.
6. Add a loop controller with bounded retries.
7. Only after that, consider white-box probes or backend instrumentation.

## Source Links

Primary papers:

- Agrawal et al. 2024: https://aclanthology.org/2024.findings-eacl.62/
- Farquhar et al. 2024: https://www.nature.com/articles/s41586-024-07421-0
- Liang et al. 2024: https://aclanthology.org/2024.knowledgenlp-1.4/
- Liu et al. 2025 SelfElicit: https://aclanthology.org/2025.findings-acl.211/
- Gupta et al. 2025 CONFACTCHECK: https://aclanthology.org/2025.findings-ijcnlp.129/
- Han et al. 2025 fact probes: https://aclanthology.org/2025.findings-emnlp.880/
- Alnuhait et al. 2025 FACTCHECKMATE: https://aclanthology.org/2025.findings-emnlp.663/
- Liu et al. 2025 AGSER: https://aclanthology.org/2025.emnlp-main.1063/
- Azaria and Mitchell 2023: https://aclanthology.org/2023.findings-emnlp.68/

Code repos inspected:

- https://github.com/microsoft/hallucinated-references
- https://github.com/jlko/long_hallucinations
- https://github.com/liangyuxin42/dreamcatcher
- https://github.com/lzhmarkk/SelfElicit
- https://github.com/JThh/fact-probe
