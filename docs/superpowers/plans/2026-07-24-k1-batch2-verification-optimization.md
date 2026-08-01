# K=1 Batch-2 Verification Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TP=4 DeepSeek-V4 K=1 target verification execute the two target rows as efficient batches while preserving the frozen greedy token sequence and at least 80% acceptance.

**Architecture:** Keep the exact K=1 umbrella default-off and replace one rowwise compatibility path at a time. Each slice first proves exactness against the frozen oracle, then measures call-count and cycle-latency reduction; failed slices stay disabled and their evidence is recorded. The final candidate combines only independently exact slices before the 100-prompt and normal-throughput gates.

**Tech Stack:** Python 3.12, PyTorch/vLLM V1, MetaX C500 TP=4, PIECEWISE CUDA graphs, TileLang MHC, pytest, Torch profiler.

## Global Constraints

- Acceptance is TP=4 with `NUM_SPECULATIVE_TOKENS=1`; TP=1 is diagnosis only.
- Keep MTP default-off and preserve exact greedy token IDs and finish reasons.
- Use `/root/vLLM-metax/.venv`, `source ./env.sh`, PIECEWISE graphs, prefix cache on, `MAX_NUM_BATCHED_TOKENS=8192`, and 100 output tokens for performance.
- Do not change model, TP, graph mode, prompt difficulty, output length, or memory settings to make a candidate pass.
- Do not use eager or Torch fallback as accepted performance paths.
- Preserve the dirty worktree and do not revert unrelated user changes.

---

### Task 1: Freeze Red-Capable Gates

**Files:**
- Modify: `tools/debug/parse_spec_decode_metrics.py`
- Test: `tests/tools/test_parse_spec_decode_metrics.py`
- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md`

**Interfaces:**
- Consumes: 100-prompt corpus and current exact K=1 manifest.
- Produces: exact acceptance aggregation and a fixed cycle/call-count comparison contract.

- [ ] **Step 1: Verify the acceptance aggregator**

Run:

```bash
source .venv/bin/activate
source ./env.sh
pytest -q tests/tools/test_parse_spec_decode_metrics.py
```

Expected: `2 passed` and aggregation of accepted/drafted counts remains exact.

- [ ] **Step 2: Record the frozen performance gates**

Record in Plan 08: acceptance `>=80%`, 100/100 exact prompts, K=1/base cycle ratio `<1.10x` interim and `<1.03x` final, MHC/MoE/all-reduce calls close to one batched call per layer, and D2D calls materially below the current 348 per step.

- [ ] **Step 3: Verify documentation**

Run:

```bash
markdownlint-cli2 docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md
git diff --check
```

Expected: zero errors.

### Task 2: Remove Duplicate Attention Projection Work

**Files:**
- Modify: `vllm_metax/models/deepseek_v4/attention.py`
- Modify: `vllm_metax/models/deepseek_v4/mtp_candidate.py`
- Modify: `vllm_metax/envs.py`
- Test: `tests/models/deepseek_v4/test_prefill_gemm_chunking.py`

**Interfaces:**
- Consumes: `_replace_fused_q_tokenwise(hidden_states, result)` and the K=1 correctness umbrella.
- Produces: a default-off batch-2 projection candidate that does not recompute both rows after the batched projection.

- [ ] **Step 1: Add one failing candidate-dispatch test**

Add a test that enables the K=1 correctness umbrella plus the new batch-2 projection candidate and asserts one `fused_wqa_wkv` invocation with a two-row tensor, while an explicit tokenwise environment override still produces two one-row calls.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
pytest -q tests/models/deepseek_v4/test_prefill_gemm_chunking.py -k 'k1 and attn_gemm'
```

Expected: failure because the candidate gate does not exist or still invokes rowwise projection.

- [ ] **Step 3: Implement the minimal default-off gate**

Add one candidate resolver in `mtp_candidate.py`/`envs.py` and make `attention.py` retain the batched projection result only when that resolver is enabled. Explicit `VLLM_METAX_DSV4_TOKENWISE_ATTN_GEMM=1` or `VLLM_METAX_DSV4_TOKENWISE_Q_ONLY=1` must continue to force rowwise behavior.

- [ ] **Step 4: Run focused unit tests and TP=4 fast exact gate**

Run:

```bash
pytest -q tests/models/deepseek_v4/test_prefill_gemm_chunking.py -k 'k1 or attn_gemm'
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k1_batch2_20260724/attn_projection \
  --num-speculative-tokens 1 --max-tokens 16 --max-model-len 1024 \
  --prefix-only \
  --cudagraph-mode PIECEWISE --no-diagnostic-enforce-eager \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_BATCH2_ATTN_PROJ_CANDIDATE=1
```

Expected: unit tests pass; TP=4 token IDs match the frozen oracle. A mismatch rejects this slice.

### Task 3: Restore Batched Indexer and Compressor

**Files:**
- Modify: `vllm_metax/customized/layers/sparse_attn_indexer/int8.py`
- Modify: `vllm_metax/models/deepseek_v4/compressor.py`
- Modify: `vllm_metax/models/deepseek_v4/mtp_candidate.py`
- Modify: `vllm_metax/envs.py`
- Test: `tests/customized/layers/sparse_attn_indexer/test_int8.py`
- Test: `tests/models/deepseek_v4/test_compressor_tokenwise.py`

**Interfaces:**
- Consumes: the current tokenwise indexer decode and compressor save/insert paths.
- Produces: independent default-off batch-2 candidates with explicit tokenwise overrides.

- [ ] **Step 1: Add a failing indexer dispatch test**

Assert that K=1 plus the indexer batch candidate invokes the native paged-logit operation once for two rows and preserves output shape, dtype, and top-k indices.

- [ ] **Step 2: Implement and verify the indexer candidate**

Run:

```bash
pytest -q tests/customized/layers/sparse_attn_indexer/test_int8.py -k 'k1 or tokenwise'
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k1_batch2_20260724/indexer \
  --num-speculative-tokens 1 --max-tokens 16 --max-model-len 1024 \
  --prefix-only \
  --cudagraph-mode PIECEWISE --no-diagnostic-enforce-eager \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_BATCH2_INDEXER_CANDIDATE=1
```

Expected: focused tests and frozen TP=4 tokens pass; otherwise reject the slice.

- [ ] **Step 3: Add a failing compressor dispatch test**

Assert that the compressor candidate performs one two-row save/compress call and that explicit tokenwise mode retains two one-row calls.

- [ ] **Step 4: Implement and verify the compressor candidate**

Run:

```bash
pytest -q tests/models/deepseek_v4/test_compressor_tokenwise.py
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k1_batch2_20260724/compressor \
  --num-speculative-tokens 1 --max-tokens 16 --max-model-len 1024 \
  --prefix-only \
  --cudagraph-mode PIECEWISE --no-diagnostic-enforce-eager \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_BATCH2_COMPRESSOR_CANDIDATE=1
```

Expected: focused tests and frozen TP=4 tokens pass; otherwise reject the slice.

### Task 4: Combine Exact Slices and Profile

**Files:**
- Modify: `tools/run_deepseek_v4_mtp_generate.sh` only if a stable public candidate switch is needed.
- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md`

**Interfaces:**
- Consumes: independently exact batch-2 candidates from Tasks 2 and 3.
- Produces: one combined default-off candidate manifest and before/after traces.

- [ ] **Step 1: Run the three-prompt exact gate**

Use the fixed three-prompt corpus, TP=4, 100 output tokens, PIECEWISE, prefix cache on, and every accepted candidate flag. Expected: all token IDs and finish reasons exactly match MTP=0.

- [ ] **Step 2: Profile five steady steps**

Collect four-rank traces for unchanged MTP=0 and combined K=1. Expected interim structural gates: MHC/MoE/all-reduce call counts no worse than the retained exact path; projection/indexer/compressor call counts and D2D copies fall according to the enabled slices.

- [ ] **Step 3: Reject non-improving complexity**

Remove any candidate code that passes exactness but produces no measurable cycle or call-count improvement. Keep artifact evidence in Plan 08.

### Task 5: Final 100-Prompt and Throughput Gates

**Files:**
- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md`

**Interfaces:**
- Consumes: combined exact K=1 candidate.
- Produces: acceptance, correctness, cycle, normal TPS, and per-GPU utilization evidence.

- [ ] **Step 1: Run the 100-prompt corpus**

Use `tools/debug/corpora/deepseek_v4_mtp_acceptance_100_20260724.jsonl`, TP=4, K=1, 100 output tokens, and the frozen manifest. Expected: 100/100 exact, acceptance at least 80%, and all finish reasons `length`.

- [ ] **Step 2: Run normal-serving performance**

Run MTP=0 and K=1 with identical workload settings, warmups, and repetitions. Expected final target: K=1/base cycle ratio below `1.03x`; report actual TPS even if the target is missed.

- [ ] **Step 3: Run project verification and review**

Run:

```bash
pytest -q tests/models/deepseek_v4/test_prefill_gemm_chunking.py \
  tests/models/deepseek_v4/test_compressor_tokenwise.py \
  tests/customized/layers/sparse_attn_indexer/test_int8.py
ruff check vllm_metax/models/deepseek_v4 vllm_metax/customized/layers/sparse_attn_indexer
git diff --check
markdownlint-cli2 docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md
```

Expected: all tests and static checks pass. A reviewer must confirm exactness, graph safety, fallback status, and evidence paths before enabling any candidate by default.
