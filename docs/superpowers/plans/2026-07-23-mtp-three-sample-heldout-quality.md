# MTP Three-Sample and Held-Out Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TP=4 k=4 speculative decoding reproduce the MTP=0 greedy token
sequence on three fixed regression prompts, then pass three isolated held-out
prompts that were not used to choose the fix.

**Architecture:** Extend the existing validation matrix to load each engine
once per three-prompt phase by using the runner's `PROMPT_TEXTS_JSON` and
`EXPECTED_RUN_TOKEN_IDS` interfaces. Keep regression and held-out corpora in
separate JSONL files, reject overlap, and stop before held-out execution unless
regression is 3/3 exact. Diagnose the earliest mismatch with one candidate
switch changed per run and accept only input-independent fixes.

**Tech Stack:** Python 3.12, pytest, vLLM V1 TP=4, MetaX C500, PIECEWISE CUDA
graphs, JSONL corpora.

## Global Constraints

- TP=4, `GPU_MEM=0.9`, PIECEWISE graph mode, prefix caching enabled.
- Compare committed token IDs and finish reason against a fresh MTP=0 oracle.
- Each regression or held-out phase contains exactly three distinct prompts.
- Held-out prompts cannot influence implementation or candidate selection.
- MTP remains default-off on any mismatch, fallback, graph failure, or runtime
  error.
- Do not enable Torch, eager, or alternate-backend fallbacks.
- Preserve unrelated dirty-worktree changes and do not create commits unless
  explicitly requested.

---

### Task 1: Three-prompt runner contract

**Files:**

- Modify: `tools/debug/run_deepseek_v4_mtp_validation_matrix.py`
- Test: `tests/tools/test_run_deepseek_v4_mtp_validation_matrix.py`

**Interfaces:**

- Consumes: `PROMPT_TEXTS_JSON`, `RUN_MAX_TOKENS_JSON`,
  `EXPECTED_RUN_TOKEN_IDS` from `tools/tmp_deepseek_v4_mtp_generate.py`.
- Produces: one baseline run and one candidate run containing three token-ID
  rows and finish reasons.

- [x] Add a failing parser test for `RUN_TOKEN_IDS_JSON` and
  `RUN_FINISH_REASONS` containing three rows.
- [x] Run
  `pytest -q tests/tools/test_run_deepseek_v4_mtp_validation_matrix.py` and
  confirm the parser test fails because the fields are absent.
- [x] Parse both structured fields and add `build_group_run_env(prompts, ...)`
  that sets `BENCH_RUNS=3`, `PROMPT_TEXTS_JSON`, and the expected token matrix.
- [x] Replace per-prompt engine launches with one baseline and one candidate
  launch, then construct the existing per-prompt comparison rows.
- [x] Run the focused tests and Ruff until green.

### Task 2: Regression and held-out corpus isolation

**Files:**

- Create: `tools/debug/corpora/deepseek_v4_mtp_regression_3.jsonl`
- Create: `tools/debug/corpora/deepseek_v4_mtp_heldout_3_20260723.jsonl`
- Modify: `tools/debug/run_deepseek_v4_mtp_validation_matrix.py`
- Test: `tests/tools/test_run_deepseek_v4_mtp_validation_matrix.py`

**Interfaces:**

- Consumes: two JSONL files with `prompt_id` and `text`.
- Produces: distinct three-prompt phases and an overlap failure before model
  loading.

- [x] Add a failing test requiring exactly three unique prompts and rejecting
  duplicate IDs, duplicate text, or overlap between regression and held-out.
- [x] Create three regression prompts covering factual reasoning, numerical
  reasoning, and code reasoning.
- [x] Create three new held-out prompts in the same domains with different
  subjects, wording, and reasoning paths.
- [x] Implement corpus validation and run focused tests until green.

### Task 3: Reproduce and minimize k=4 divergence

**Files:**

- Artifacts: `.logs/deepseek_v4_mtp_quality_3sample_20260723/`
- Update: `docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md`

**Interfaces:**

- Consumes: the regression corpus and current k=4 candidate environment.
- Produces: a deterministic 3-row comparison with first-mismatch evidence.

- [x] Run the TP=4 k=4 regression phase with `MAX_TOKENS=100` and save the RED
  summary.
- [x] Re-run only the failing configuration to confirm the same prompt, token
  index, oracle token, and candidate token.
- [x] Test ranked hypotheses one at a time: disable selective native WQ_B;
  disable the MHC hybrid; then, only if needed, capture verifier/KV state around
  the first partial rejection.
- [x] Record the smallest input-independent configuration that removes the
  mismatch without a fallback.

### Task 4: Implement and gate the general fix

**Files:**

- Modify only the module identified by Task 3.
- Add the closest public-interface regression test under `tests/`.
- Update: `docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md`

**Interfaces:**

- Consumes: the minimized RED reproducer and MTP=0 oracle.
- Produces: TP=4 PIECEWISE k=4 exactness without prompt-dependent selectors.

- [x] Add one failing regression test for the identified state or numerical
  contract.
- [x] Implement the smallest input-independent fix and keep it default-off
  until all gates pass.
- [x] Run focused unit tests, Ruff, `py_compile`, and `git diff --check`.
- [x] Run the regression corpus and require `3/3` exact before proceeding.

### Task 5: Held-out acceptance

**Files:**

- Artifacts: `.logs/deepseek_v4_mtp_quality_heldout_20260723/`
- Update: `docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md`

**Interfaces:**

- Consumes: the frozen fix from Task 4 and held-out corpus from Task 2.
- Produces: an independent quality verdict.

- [x] Freeze code and candidate environment after regression reaches `3/3`.
- [x] Run the three held-out prompts once with fresh MTP=0 oracles.
- [x] Require `3/3` token IDs and finish reasons exact, graph sizes 1 through 5,
  and no fallback markers.
- [x] Confirm held-out did not fail, so no held-out prompt promotion or tuning
  loop was needed.
- [x] Document artifacts, dispatch evidence, and residual risks. Keep MTP
  default-off until the full gate passes.

Acceptance evidence (2026-07-23):

- Fixed compatibility decode so a row with `topk_lens=0` omits the physical
  all-invalid top-k stream instead of changing the native GEMM/softmax reduction
  width from `K=128` to `K=640`.
- Fixed mixed short/long speculative batches so the indexer compressor skips
  rows before the sliding-window boundary, matching single-token cache state.
- Regression corpus: `3/3` exact at
  `.logs/deepseek_v4_mtp_quality_3sample_20260723/promoted_systems_final_fixes_formal_3/summary.json`.
- Fresh post-fix held-out corpus: `3/3` exact at
  `.logs/deepseek_v4_mtp_quality_3sample_20260723/fresh_heldout_final_fixes_3/summary.json`.
- Both gates used TP=4, k=4, PIECEWISE graph mode, prefix caching, 8192 batched
  token limit, and 100 committed tokens. No Torch/eager fallback was enabled.
- A final scope smoke replayed the accepted token sequence exactly; artifact:
  `.logs/deepseek_v4_mtp_quality_3sample_20260723/final_scope_smoke/run.log`.
- Review found no remaining correctness issue in the two scalar branches:
  both run inside `attention_impl`'s breakable-graph eager segment. Their
  device-to-host synchronization cost remains for the performance phase.

Independent recheck on 2026-07-24 did not pass: a new three-prompt corpus
(`new_policy_reasoning`, `new_data_pipeline_reasoning`, and
`new_security_reasoning`) produced `0/3` exact under a clean baseline and an
explicit full candidate environment. First mismatches were token indices 1, 8,
and 27. The result is recorded at
`.logs/deepseek_v4_mtp_quality_3sample_20260724/new_heldout_3_explicit_full_v2/`;
MTP remains default-off and the general-quality gate is reopened.
