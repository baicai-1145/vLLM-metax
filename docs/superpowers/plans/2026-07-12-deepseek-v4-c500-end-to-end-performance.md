# DeepSeek V4 TP=4 C500 end-to-end performance plan

## Scope and non-negotiable gates

Goal: close the measured gap between 4x C500 and 4x A100 for
`DeepSeek-V4-Flash-W4A16-BF16Attn-MTP`, starting with MTP disabled. Optimize the
whole decode path rather than one MHC kernel in isolation.

- Keep TP=4 for acceptance runs; use TP=1 only as a diagnostic ablation.
- Keep the 16-token greedy IDs exactly equal to
  `[260,5036,294,10588,14,790,342,2118,436,734,260,1894,5090,304,611,260]`.
- Every new kernel must pass direct differential, graph replay, stable-pointer,
  and TP=4 model gates before performance claims.
- Do not modify `/root/vllm`; use `/root/vllm-0.25rc1` as the CUDA reference.
- Report normal inference throughput separately from profiler throughput.

## Current evidence

Normal C500 PIECEWISE inference is `11.6247 token/s`, or about `86.0 ms/token`.
The matching five-step steady profile reports `80.3 ms/step` inside
`execute_context`, which is consistent with the non-profiler result after normal
measurement noise.

| Aggregate over five profiled steps | Total | Approx. per step | Calls/step |
| --- | ---: | ---: | ---: |
| Elementwise/reduction top five kernels | 154.45 ms | 30.89 ms | thousands |
| Fused W4A16 MoE GEMM kernel | 23.13 ms | 4.63 ms | 86 |
| `aten::copy_` | 17.81 ms | 3.56 ms | 390 |
| MCCL BF16 all-reduce | 14.52 ms | 2.90 ms | 87 |
| FP32 `sgemvt` | 11.04 ms | 2.21 ms | 87 |
| Exact MHC post MMA | 8.26 ms | 1.65 ms | 86 |
| Attention `aten::bmm` | 3.03 ms | 0.61 ms | 86 |

The aggregate table is not a stage attribution: PyTorch operations from sparse
attention and exact MHC raw/pre share generic kernel names. Add stage ranges
before assigning the full 30.89 ms elementwise/reduction bucket.

### Confirmed CUDA/MetaX path gaps

1. **Sparse MLA decode is a Torch reference on MetaX.**
   `vllm_metax/models/deepseek_v4/flashmla.py` gathers KV rows in FP32 and runs
   `torch.matmul -> masked_fill -> softmax -> matmul`. The CUDA path calls
   `flash_mla_with_kvcache`, a dedicated sparse FlashMLA kernel.
2. **Sparse MLA prefill also falls back to Torch.**
   `vllm_metax/v1/attention/ops/flashmla.py` documents that the MetaX sparse
   kernel traps and routes to `torch_flash_mla_sparse_prefill`.
3. **CUDA fuses RMSNorm into MHC; MetaX does not.**
   CUDA passes `norm_weight/norm_eps` into the MHC TileLang operations. MetaX
   executes `attn_norm` and `ffn_norm` separately after MHC.
4. **CUDA O-proj uses fused inverse-RoPE + FP8 quant + FP8 einsum.**
   MetaX uses inverse-RoPE followed by BF16 einsum and `wo_b`, exposing several
   BF16 GEMV/GEMM and copy kernels.
5. **MetaX disables graph-aware custom all-reduce.**
   The runtime selects PYNCCL/MCCL. Decode performs about 87 TP reductions per
   token, while CUDA can use its custom graph-aware all-reduce chain.
6. **W4A16 MoE remains multi-stage at decode batch one.**
   Each layer launches alignment, two `fused_moe_kernel_gptq_awq` stages,
   activation/quantization, and routing helpers. Existing MCTLaSS fusion is not
   the W4A16 fast path.

The C500's authoritative BF16/TF32/INT4 peak, HBM bandwidth, and card-to-card
fabric figures are not available in the local or public material reviewed. Do
not explain the tenfold gap using unverified peak specifications. For reference,
NVIDIA publishes A100 80GB SXM at 2,039 GB/s HBM bandwidth and up to 600 GB/s
bidirectional NVLink, with mature Tensor Core, CUDA Graph, cuBLASLt, FlashMLA,
and NCCL paths.

## Phase 0: establish an A100/C500 differential benchmark

Use the same model checkpoint, prompt, commit, sampling parameters, TP=4,
batch=1, context length, graph mode, and MTP=0 on both systems.

Record all of the following after at least three warmups:

- prefill latency and tokens/s;
- decode median/p90 ms/token for 64 or more tokens;
- TP=1 and TP=4 decode latency;
- eager and graph latency;
- per-stage ranges for MHC, attention indexer, sparse MLA, O-proj, MoE, and TP
  collectives;
- kernel count, GPU-active time, inter-kernel gaps, achieved memory bandwidth,
  and occupancy where the vendor profiler exposes them.

Acceptance: repeated runs vary by less than 3%, output IDs match, and the A100
and C500 traces use equivalent model paths. A filename containing `tilelang` or
`flashmla` is not proof that the kernel was dispatched; require named kernel or
dispatch-counter evidence.

## Phase 1: replace Torch sparse MLA decode and prefill

This is the highest-priority implementation because the CUDA reference uses one
specialized sparse attention kernel while MetaX materializes FP32 gathered KV,
scores, softmax probabilities, and output through many Torch kernels.

Implementation sequence:

1. Build a real-payload differential corpus containing q, paged KV/SWA cache,
   sparse indices, lengths, scale, sink values, and output.
2. Repair or replace the trapping MetaX `sparse_attn_global_fwd_kernel`.
   Prefer a native MetaX/MCTLaSS kernel; use TileLang only if its paged gather,
   online-softmax, and graph capture behavior are proven.
3. Fuse paged gather, invalid-index masking, online softmax, and value reduction.
   Do not materialize FP32 `topk x heads` score/probability tensors globally.
4. Support the three observed compression ratios and the padded head layouts.
5. Re-enable the native prefill path only after real-cache differential and graph
   replay gates pass.

Targets: one planner kernel at most plus one attention kernel per layer; no
`aten::index`, `aten::bmm`, or standalone softmax in decode; at least 3x speedup
for the isolated sparse-MLA stage and no BF16 output differences.

## Phase 2: fuse exact MHC raw/pre with RMSNorm

The exact post MMA is already fast enough to be a secondary target. The missing
work is the raw/pre boundary and adjacent normalization.

1. Add stage-level differential capture for residual, raw GEMV, square sum,
   affine/sigmoid, Sinkhorn iterations, pre-mix, and normalized output.
2. Fuse post output consumption, raw GEMV, square-sum reduction, affine/sigmoid,
   and Sinkhorn into the minimum graph-safe kernel set.
3. Fuse pre-mix reduction and the following RMSNorm using the CUDA MHC API shape:
   pass `norm_weight` and `norm_eps` into the MetaX operation and remove the
   separate `attn_norm`/`ffn_norm` launches only after bitwise validation.
4. Keep exact output buffers caller-owned; prohibit silent Torch fallback and log
   explicit non-decode dispatches.

Targets: eliminate generic Torch `sum/div/mul/copy` clusters from MHC decode;
reduce the complete MHC+RMSNorm boundary to no more than three kernels; retain
the exact 16-token gate.

## Phase 3: optimize O-proj and W4A16 MoE

### O-proj

- First benchmark current inverse-RoPE, BF16 group einsum, and `wo_b` separately.
- If C500 has a supported FP8 tensor-core path, port fused inverse-RoPE + FP8
  quantization and scaled einsum from the CUDA design.
- Otherwise fuse inverse-RoPE directly into a BF16 Tensor Core/MCTLaSS GEMM and
  avoid intermediate BF16 copies. Do not emulate FP8 with slow software casts.

### W4A16 MoE

- Freeze the exact runtime shapes, expert histogram, and tuned config used by
  `E=256,N=256,device_name=MXC500,dtype=int4_w4a16.json`.
- Benchmark routing/alignment, stage-1 GEMM, activation, requantization, and
  stage-2 GEMM separately.
- Implement a decode-specialized persistent or grouped W4A16 kernel that fuses
  stage-1 epilogue, activation, and stage-2 input quantization.
- Fuse or remove per-layer expert alignment for the batch-one shape; reuse
  persistent metadata where routing permits it.

Targets: halve MoE kernel launches and reduce isolated MoE latency by at least
30%. Compare A100 against its standard FusedMoE path, not MegaMoE, because the
upstream MegaMoE implementation requires SM100.

## Phase 4: reduce TP communication and graph gaps

1. Measure the same tensors at TP=1 and TP=4. The difference is the upper bound
   for collective and TP synchronization work.
2. Extend the existing all-reduce microbenchmark to every runtime tensor size and
   compare eager, graph capture, and replay.
3. Implement a graph-safe low-latency C500 all-reduce for small BF16 decode
   tensors, or enable a verified vendor equivalent. Do not optimize bulk
   bandwidth first: 87 small reductions/token make latency dominant.
4. Fuse row-parallel reduction with the next residual/RMSNorm/MHC consumer where
   numerical ownership permits it.
5. Re-profile graph breaks and CPU-to-GPU gaps after removing Torch attention and
   MHC fragments. Attempt a larger decode graph only after every custom kernel
   has stable pointers and capture-safe allocation.

Targets: reduce MCCL/collective time below 1 ms/token and cut collective launches
or synchronization points by at least 2x. PIECEWISE graph must remain enabled;
the current graph run is already about 3x faster than eager.

## Execution order and decision gates

| Priority | Work item | Why first | Promotion gate |
| --- | --- | --- | --- |
| P0 | Native sparse MLA decode | Confirmed Torch fallback in every layer | real-cache differential + graph + TP=4 IDs |
| P0 | Native sparse MLA prefill | Confirmed trapping kernel/Torch fallback | mixed prefill-decode differential |
| P1 | Exact MHC raw/pre + RMSNorm | largest remaining launch/reduction cluster | full stage bitwise + graph |
| P1 | O-proj fused quant/GEMM | CUDA uses materially different FP8 path | isolated speedup + model IDs |
| P1 | W4A16 MoE decode kernel | 86 expert GEMM launches/token plus helpers | expert-shape corpus + TP=4 |
| P1 | Small-message custom all-reduce | 87 reductions/token | TP microbench + graph replay |
| P2 | Larger graph / overlap | only useful after capture-safe kernels | lower inter-step gap, no graph breaks |
| P3 | Re-enable MTP | base decode must be stable first | acceptance rate + net tokens/s |

After every phase, rerun the same A100/C500 matrix. Attribute cumulative speedup
to measured stage changes; do not combine profiler overhead with normal inference
throughput and do not claim the hardware gap is closed from microbenchmarks alone.
