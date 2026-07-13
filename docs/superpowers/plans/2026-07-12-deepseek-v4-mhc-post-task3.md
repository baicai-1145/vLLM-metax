# DeepSeek V4 MHC Exact Post Task 3 Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with a correctness checkpoint after every kernel change.

**Goal:** Implement a fixed-shape MetaX TileLang MHC post kernel that is bitwise identical to the captured Torch post boundary for all 340 real TP=4 payloads and is safe for CUDA graph replay.

**Architecture:** Keep the existing Torch post path as the oracle and do not dispatch the candidate until every payload passes. Replace the currently inaccurate scalar four-term reduction with a 16-aligned TileLang MMA `T.gemm` computation: pad the 4x4 comb matrix and four residual rows to K/M=16, compute the four output rows, add the post*x term in the epilogue, and cast exactly once to BF16. The kernel will use stable caller-owned output storage and fixed `(1,4,4096)` shapes.

**Tech Stack:** MetaX TileLang Cython backend, PyTorch BF16/FP32, CUDA graph replay, pytest.

## Global Constraints

- Keep TP=4 and the required greedy IDs `[260,5036,294,10588,14,790,342,2118,436,734,260,1894,5090,304,611,260]`.
- Do not modify `/root/vllm` or revert unrelated dirty changes.
- Preserve the Torch path as an explicit oracle; no silent fallback may be treated as success.
- Do not dispatch the candidate until direct `340/340` and graph replay both pass bitwise.
- Do not fuse raw GEMV/square-sum or alter the exact pre/Sinkhorn kernel in Task 3.

### Task 1: Freeze the oracle and candidate contract

**Files:**
- Modify: `vllm_metax/models/deepseek_v4/ops/mhc/tilelang_kernels.py`
- Modify: `vllm_metax/models/deepseek_v4/ops/mhc/debug_diff.py`
- Test: `tests/models/deepseek_v4/test_mhc_raw_diff.py`

- [ ] Add shape, dtype, contiguity, and output-pointer checks for `_mhc_post_exact_tl(x_flat, residual_flat, post_mix, comb_mix, out=None)`.
- [ ] Add a direct test that rejects `(T != 1)`, `H != 4096`, non-BF16 residuals, and non-FP32 mixes before kernel launch.
- [ ] Assert the capture directory contains 340 schema-v2 payloads, with rank counts `{0:85,1:85,2:85,3:85}`.
- [ ] Run `python -m py_compile ...` and the focused pytest module before changing arithmetic.

### Task 2: Implement the padded MMA post kernel

**Files:**
- Modify: `vllm_metax/models/deepseek_v4/ops/mhc/tilelang_kernels.py`
- Create: `tools/debug/probe_metax_mhc_post_mma.py`

- [ ] Define `_mhc_post_exact_mma()` with `T.Kernel(1, threads=128)` and `T.gemm` dimensions `M=16, N=4096, K=16`.
- [ ] Fill shared A as `A[output,input] = comb_mix[input,output]` for indices `<4`, zero elsewhere; fill shared B from the four BF16 residual rows converted to the MMA-supported input type and zero rows `4:16`.
- [ ] Accumulate into FP32 C, add `post_mix[output] * x[hidden]` using the verified MetaX rounding intrinsic, and store only rows `0:4` as BF16.
- [ ] Keep all hidden tiles statically sized and avoid dynamic allocations or pointer replacement so graph capture sees stable addresses.
- [ ] The probe must report first differing index, BF16 bit patterns, FP32 intermediate values, and the kernel configuration for one payload.
- [ ] Test the kernel on `rank0_call18.pt` first, then on one payload from each rank before running the full corpus.

### Task 3: Complete direct differential and graph gates

**Files:**
- Modify: `tools/debug/diff_deepseek_v4_mhc_raw.py`
- Test: `tests/models/deepseek_v4/test_mhc_raw_diff.py`

- [x] Add `--candidate post-mma` and route it only to `_mhc_post_exact_mma`.
- [x] Run:

```bash
python tools/debug/diff_deepseek_v4_mhc_raw.py \
  .logs/dsv4_mhc_fused_post_prenorm_corpus_tp4 \
  --candidate post-mma --require-bitwise --max-files 340
```

- [x] Require output `files=340`, `passed=340`, `failed=0`, and zero differing BF16 bits.
- [x] Capture two payloads into fixed input/output buffers and replay the kernel through `torch.cuda.CUDAGraph`; verify output pointers, input pointers, and bitwise outputs after both replays.
- [x] No payload failed; production dispatch remains unchanged and the probe remains available for arithmetic diagnostics.

**Task 3 verification (2026-07-12):**

- The `post-mma` differential run over `.logs/dsv4_mhc_fused_post_prenorm_corpus_tp4` reported `files=340`, `passed=340`, `failed=0`, `bitwise=true`, and no first failure.
- CUDA graph replay reported `passed=true`, `pointers_stable=true`, and `replays=2`.
- `./.venv/bin/pytest -q tests/models/deepseek_v4/test_mhc_raw_diff.py` reported `11 passed`.
- `py_compile` and `git diff --check` both exited successfully.

### Task 4: Production opt-in and correctness gate

**Files:**
- Modify: `vllm_metax/models/deepseek_v4/ops/mhc/tilelang.py`
- Modify: `vllm_metax/models/deepseek_v4/ops/mhc/backend.py`
- Modify: `tools/tmp_deepseek_v4_mtp_generate.py`
- Test: `tests/models/deepseek_v4/test_mhc_raw_diff.py`
- Test: `tests/compat/test_deepseek_v4_mhc_backend.py`

- [x] Add `VLLM_METAX_DSV4_MHC_EXACT_POST_MMA=1`, defaulting off; require the TileLang backend and a `post`/`fused` op selection.
- [x] When enabled for the fixed decode contract, run the MMA post kernel into caller-owned output storage, then use the existing exact raw/pre path. `VLLM_METAX_DSV4_MHC_EXACT_POST_MMA_DEBUG=1` compares the BF16 post result with the Torch oracle and raises on mismatch.
- [x] Fail closed on decode contract, device, contiguity, backend, and incompatible Torch-split settings. Non-decode prefill/dummy calls are explicitly logged as `torch_prefill` or `tilelang_prefill`, never mislabeled as exact decode.
- [x] TP=4 16-token eager and PIECEWISE gates both produced the exact ID sequence `[260,5036,294,10588,14,790,342,2118,436,734,260,1894,5090,304,611,260]`.

**Task 4 verification (2026-07-12):**

- TP=4 eager 2-token gate: IDs `[260,5036]`.
- TP=4 eager 16-token gate: exact IDs, `OUTPUT_TOKENS_PER_SECOND=3.798382`.
- TP=4 PIECEWISE 16-token gate: exact IDs, `OUTPUT_TOKENS_PER_SECOND=11.624672`.
- Single real payload with debug oracle enabled completed with `debug_post_oracle=passed`.

### Task 5: Post-kernel performance benchmark

- [x] Benchmark Torch post, scalar TileLang post, and MMA post with identical `(1,4,4096)` buffers, 20 warmups, and 100 eager/graph iterations.
- [x] Keep the candidate opt-in and default-off; all decode correctness and graph gates are exact, so no Torch fallback is used for the candidate path.

**Task 5 verification (2026-07-12):**

- Eager latency: Torch `0.0682 ms`, scalar TileLang `0.0654 ms`, exact MMA `0.0222 ms`.
- Graph latency: Torch `0.0453 ms`, scalar TileLang `0.0828 ms`, exact MMA `0.0350 ms`.
- Every benchmark output was bitwise equal to the captured oracle and output pointers remained stable.

### Task 6: TP=4 profiler capture and comparison

- [x] Add profiler delay/max/active/ignore-frontend controls to the local runner so the new capture matches the 2026-07-11 steady-steps configuration.
- [x] Capture the latest exact-post profile and retain the console/profile artifacts under `.logs/` and `/tmp/`.
- [x] Compare exact-post kernel, MCCL all-reduce, total Self CUDA, and greedy throughput against the prior steady-steps profile.

**Task 6 verification (2026-07-12):**

- Latest profile: `.logs/dsv4_mhc_exact_post_mma_steady_steps_profile_20260712.log`; trace directory: `/tmp/dsv4_mhc_exact_post_mma_steady_steps_profile_20260712`.
- Exact MMA kernel: `8.256 ms`, 430 calls, `19.201 us` average, `2.54%` Self CUDA.
- MCCL all-reduce: `14.515 ms`, 435 calls, `4.47%` Self CUDA.
- Total Self CUDA: `324.764 ms`; output throughput under profiler: `0.918715 token/s`; greedy IDs matched exactly.
- Prior 2026-07-11 steady-steps profile: Self CUDA `201.476 ms`, MCCL `5.596 ms`, throughput `2.065522 token/s`, with `mhc_pre_from_raw_exact_fuse_tl_kernel` at `61.189 ms`. The current profile does not contain that kernel because the exact-post opt-in uses the existing Torch exact raw/pre boundary; this is a measurement and optimization follow-up, not a correctness failure.

**Task 5/6 verification commands:**

- `./.venv/bin/python tools/debug/bench_deepseek_v4_mhc_post.py .logs/dsv4_mhc_fused_post_prenorm_corpus_tp4/rank0_call0.pt --warmup 20 --iterations 100 --json-out .logs/dsv4_mhc_post_bench_20260712.json`
- `PROFILE_DELAY_ITERATIONS=4 PROFILE_MAX_ITERATIONS=5 PROFILE_ACTIVE_ITERATIONS=5 PROFILE_IGNORE_FRONTEND=1 ./tools/run_deepseek_v4_mtp_generate.sh`
- Focused pytest, `py_compile`, and `git diff --check`.
