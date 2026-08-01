# DeepSeek-V4 DSpark performance profiling handoff

> **Status:** historical profiling handoff. Its 2026-07-26 baseline and blocker
> are superseded by the accepted v104 cycle-reconstruction checkpoint at
> `18.208126 TPS`. Continue from
> [`12-dspark-cycle-reconstruction.md`](12-dspark-cycle-reconstruction.md).

## Start here

This document preserves the original attribution and intermediate coalescing
evidence. It is not the current checkpoint. Use the frozen contract and latest
accepted/active state in
[`12-dspark-cycle-reconstruction.md`](12-dspark-cycle-reconstruction.md).

Machine-readable evidence is in
`../../../../.logs/deepseek_v4_dspark_profile_20260726/profile_summary.json`.
Raw artifacts remain under
`../../../../.logs/deepseek_v4_dspark_profile_20260726/`.

## Frozen workload

| Setting | Value |
| --- | --- |
| Corpus | `tools/debug/corpora/deepseek_v4_mtp_new_heldout_3_20260724.jsonl` |
| Corpus SHA256 | `232696e55f01f85e3bb480e2550b277d9456c638964adaf22eadac722106be50` |
| Hardware | 4 x MetaX C500 |
| TP / speculative tokens | `4 / 5` |
| Output | 3 distinct prompts x 100 greedy tokens |
| Draft / KV | BF16 DSpark / `kv_cache_dtype=auto` with the accepted MetaX layout |
| Runtime | prefix cache ON, async ON, PIECEWISE, breakable graph |
| Limits | `MAX_MODEL_LEN=512`, `MAX_NUM_BATCHED_TOKENS=8192`, `GPU_MEM=0.9` |
| Sampling | warmup 1, measured runs 3 |
| Repository | `c399373a033b7779e6094a984ded262eae37d7b7`, dirty worktree preserved |

## Normal performance result

Both runs exited 0, generated exactly 100 tokens per prompt, ended with
`finish_reason=length`, and matched the frozen expected token IDs.

| Path | TPS | Decode seconds | P90 | GPU utilization |
| --- | ---: | --- | ---: | --- |
| MTP=0 target | 26.9972 | 3.8218, 3.7041, 3.5649 | 3.8218 s | Invalid: 1 s sampler aliased to zero |
| BF16 DSpark k=5 | 5.5939 | 21.8903, 17.8767, 12.5102 | 21.8903 s | 7.31-7.67% average, 24% maximum |

DSpark delivered 20.72% of target throughput, a 4.83x slowdown. Its P90
latency was 5.73x target. These are normal, non-instrumented measurements;
profiler TPS is intentionally excluded.

## Historical profiler attribution

> **Superseded for current exact-candidate counts.** The following 2026-07-26
> table describes the pre-coalescing trace. The current exact phase census is
> `111` all-reduces per cycle: `87` target, `19` draft backbone, and `5` draft
> sampling. There is no current unexplained 24-event gap. See the fresh exact
> evidence below and the cycle reconstruction plan.

The focused trace used one production-shape 100-token prompt and captured 15
steady decode steps after a 5-second delay. All four target and DSpark gzip
traces pass `gzip -t`.

| Rank-0 metric | Target | DSpark | Ratio |
| --- | ---: | ---: | ---: |
| Mean graph step | 48.68 ms | 841.18 ms | 17.28x |
| Scoped TP all-reduce events / step | 87 | 517 | 5.94x |
| TP all-reduce calls | 1,305 | 8,115 | 6.22x |
| All-reduce CUDA time | 70.45 ms | 6.441 s | 91.43x |
| `aten::mm` calls | 660 | 39,960 | 60.55x |
| MHC downstream RMS calls | 1,290 | 8,115 | 6.29x |
| Fused MoE kernel calls | 1,290 | 7,740 | 6.00x |
| Device-to-device copies | 675 | 12,105 | 17.93x |
| Self CPU time | 0.791 s | 13.831 s | 17.49x |

Post-handoff audit found that the DSpark named step scopes contain exactly
`7,755 = 517 * 15` all-reduces, while the full profiler table contains
`8,115 = 541 * 15`. The omitted 360 events account for 317.96 ms on rank 0 and
must be attributed by the per-layer/per-projection census; target reconciles at
`1,305 = 87 * 15`. Do not mix the scoped 5.94x ratio with the full-trace 6.22x
ratio. The same full trace contains `7,740 = 516 * 15` fused-MoE kernels, making
tokenwise FFN the first collective-reduction target.

The current exact five-cycle phase trace supersedes those counts for ongoing
optimization. Its per-rank total is `555 / 5 = 111` all-reduces per cycle,
reconciled as `435 / 5 = 87` target, `95 / 5 = 19` draft backbone, and
`25 / 5 = 5` draft sampling. Current artifacts:

- `.logs/deepseek_v4_dspark_cycle_reconstruction_20260727/fresh_exact_phase_5active/phase_allreduce_summary.tsv`
- `.logs/deepseek_v4_dspark_cycle_reconstruction_20260727/fresh_exact_phase_5active/allreduce_totals.tsv`
- `.logs/deepseek_v4_dspark_cycle_reconstruction_20260727/fresh_exact_shapes_5active/rank0_phase_mm_shape_census.tsv`

BF16 all-reduce consumed 70.68-75.44% of self CUDA time across TP ranks. Its
rank-0 mean latency rose from 53.98 us to 793.66 us; P90 was 1.522 ms and P99
was 6.239 ms. The kernels used grids of 4 or 16 blocks with reported occupancy
of 1% or 4%.

The device was not primarily waiting for the host: rank-0 device-busy union was
68.5% for DSpark versus 62.1% for target, and the mean largest device idle gap
was only 268 us. The timeline is busy with many tiny low-occupancy collectives.
Sparse MLA kernels were individually near 0.1% and are not the first target.

## Source map

The exactness adapter enables eight tokenwise controls in
`vllm_metax/patch/bugfix/dspark_v2_runner.py:15`.

| Multiplier | Current path | Optimization seam |
| --- | --- | --- |
| Q projection GEMMs | `vllm_metax/models/deepseek_v4/attention.py:1038` | Compute selected rows with a native row-exact batched kernel; keep row placement unchanged. |
| FFN and TP reductions | `vllm_metax/models/deepseek_v4/model.py:1070` | Preserve per-row FFN arithmetic while coalescing row-parallel reductions. Do not globally disable `reduce_results`. |
| O projection | `vllm_metax/models/deepseek_v4/flashmla.py:556` | Replace the Python row loop with a graph-safe row-exact native batch. |
| Compressor copies | `vllm_metax/models/deepseek_v4/compressor.py:504` | Batch row cache writes with the same positions and slot mapping. |
| Sparse MLA row loop | `vllm_metax/models/deepseek_v4/flashmla.py:425` | Lower priority; retain native dispatch and exact SWA-only semantics. |
| MHC RMS / copies | `vllm_metax/models/deepseek_v4/model.py:1318` | Batch exact helpers and reuse stable graph workspaces without changing reduction order. |

The current tokenwise tuple was introduced to recover exact greedy tokens. A
plain batched GEMM has previously changed target bytes, so “remove the loops” is
not an acceptable fix. The useful seam is to preserve the accepted per-row
arithmetic while reducing launches and TP synchronization.

## Historical recommended execution order

1. Build a per-step collective census by layer and projection, then reduce the
   517 all-reduces toward the target's 87 without changing hidden-state
   ownership or reduction order.
2. Implement and differentially validate row-exact native batching for WQ-B,
   O-projection, and FFN. The immediate isolated gate must compare each row to
   the current tokenwise oracle, including graph replay.
3. Batch MHC downstream RMS and device copies after the collective/GEMM work;
   they are secondary but measurable.
4. Run focused tests, native dispatch checks, TP=4 16-token fast exactness, then
   the frozen 3 x 100-token exactness and normal performance workload.
5. Re-profile only after normal exactness passes. Quantized draft work remains
   later because it cannot remove target-side collective multiplication.

## Acceptance gates

- Same target and staging checkpoints; no model rewrite.
- TP=4, BF16 draft, `k=5`, greedy, prefix cache ON, async ON, PIECEWISE.
- No Torch arithmetic, eager, backend, dtype, TP, or workload fallback.
- Stable graph capture/replay and observable native dispatch.
- Exact final token IDs for the frozen 3 x 100-token workload.
- Normal performance reported separately from profiler runs, with median/P90
  and per-GPU utilization.
- Acceptance quality rechecked after performance correctness passes; retain the
  established 100 x 32 corpus as the quality gate.

## Profiler caveat

The DSpark focused profiler process exited 1 **after** writing all four complete
traces and printing successful profiler-stop markers. Serializing four roughly
263 MB stack-enabled traces blocked TP workers long enough for the EngineCore
`sample_tokens` RPC to time out. This is an instrumentation teardown defect,
not a normal inference failure; the identical non-profiled run exited 0 and was
exact. Future traces should disable Python stack capture or use a dump method
that cannot block the TP control path.

Artifacts:

- Normal target: `.logs/deepseek_v4_dspark_profile_20260726/normal_target/`
- Normal DSpark: `.logs/deepseek_v4_dspark_profile_20260726/normal_dspark/`
- Target trace: `.logs/deepseek_v4_dspark_profile_20260726/profile_target_short/`
- DSpark trace: `.logs/deepseek_v4_dspark_profile_20260726/profile_dspark_short/`
- Summary: `.logs/deepseek_v4_dspark_profile_20260726/profile_summary.json`

## Collective coalescing follow-up (2026-07-26)

The opt-in FFN and O-projection candidates preserve rowwise native arithmetic
and coalesce only their final target-verifier TP reductions. Under the complete
frozen dispatch environment, FFN-only and FFN+O-projection both passed TP=4
PIECEWISE graph, 3 x 16 fast greedy exactness, and 3 x 100 greedy exactness.

FFN-only reached `5.796072 TPS`. The combined candidate reached
`6.033146 TPS` with P90 `20.005463 s`, a 7.85% throughput improvement over the
`5.593861 TPS` baseline. Reprofiling reduced full-trace all-reduces from
`8,115` to `1,665`, exactly 541 to 111 per step. Rank-0 all-reduce CUDA time,
however, changed only from `6.440545 s` to `6.362464 s`; mean latency increased
to `3.821300 ms`, P90 to `7.341056 ms`, and P99 to `7.568384 ms`. The next
performance problem is therefore rank-arrival skew at the remaining
collectives, followed by unchanged MHC/GEMM/D2D multipliers, not raw MCCL
payload bandwidth. A fail-closed TP=4 MCCL microbenchmark measured only
`42--105 us` median for aligned BF16 payloads from one to six rows; every rank
reported `MCCLLibrary` and PYNCCL dispatch.

The `541` to `111` comparison in this subsection remains the historical
pre-coalescing delta. It must not replace the fresh exact phase attribution
`111 = 87 + 19 + 5` above.

Artifacts:

- Control exactness:
  `.logs/deepseek_v4_dspark_collective_control_20260726/fast16_frozen_env/`
- FFN full gate:
  `.logs/deepseek_v4_dspark_ffn_collective_20260726/full3x100_frozen_env/`
- Combined full gate:
  `.logs/deepseek_v4_dspark_o_proj_collective_20260726/full3x100_frozen_env/`
- Candidate traces:
  `.logs/deepseek_v4_dspark_collective_profile_20260726/candidate_short/`
- Native MCCL curve:
  `.logs/deepseek_v4_dspark_mccl_curve_20260726/mccl_native.log`
- O-projection native differential:
  `.logs/dspark_collective_candidate_validation_20260726/o_proj_differential_probe_serial_retry.log`

The O-projection production differential is now complete on all four ranks.
Rows two and six were BF16 bitwise exact, finite, and repeat-bitwise; each
candidate invocation used one TP reduction. Fixed-six-row CUDAGraph capture and
two replays were bitwise with a stable output pointer.

## Current acceptance blocker (2026-07-26 evening rerun)

Fresh current-tree acceptance invalidated the earlier one-run exactness claim.
The combined candidate generated `5.986386 TPS`, but the data-pipeline sample
first diverged from the frozen oracle at zero-based output index 76. An
isolated replay also diverged, with different later tokens. Candidate and
uncoalesced control remain identical for each matched request history: hash
`3df594...` after policy-then-data ordering and `8fec197...` for an isolated
data prompt. Low-intrusion scheduler capture now shows that the two histories
are identical through verification cycle 29 and both commit `[554, 1749]`.
The second token is the target verifier bonus/correction token, while MTP=0
commits `366` after `554`. At cycle 30 the isolated run accepts zero drafts and
commits `[4067]`, while the ordered run accepts four and commits
`[366, 14114, 295, 270, 915]`; this history-dependent recovery is downstream
of the common wrong bonus token, not the root cause.

All prefix probes report zero hits for every KV group, including warmup and
measured requests. The blocker is therefore target-verifier row numerical
drift at cycle 29, independently of prefix reuse and FFN/O reduction
coalescing. No new throughput result is accepted until the six-row target
bonus row matches serial MTP=0 semantics.

The original `VLLM_METAX_DSV4_MTP_CAPTURE_DIR` probe is not observational for
this failure: its GPU-to-CPU tensor hashing/top-k reads change the generated
hashes. Use the independent
`VLLM_METAX_DSV4_PREFIX_CACHE_CAPTURE_DIR` and scheduler-side
`VLLM_METAX_SPEC_ACCEPTANCE_CAPTURE_DIR` probes for low-intrusion evidence.

Artifacts:

- Candidate 3 x 100 red rerun:
  `.logs/deepseek_v4_dspark_final_20260726/full3x100/`
- Candidate isolated replay:
  `.logs/deepseek_v4_dspark_final_20260726/replay_sample1/`
- Uncoalesced isolated control:
  `.logs/deepseek_v4_dspark_final_20260726/control_sample1/`
- Uncoalesced ordered control:
  `.logs/deepseek_v4_dspark_final_20260726/control_first2/`
- Ordered low-intrusion token-pair capture:
  `.logs/deepseek_v4_dspark_history_20260726/ordered_token_pairs/`
- Isolated low-intrusion token-pair capture:
  `.logs/deepseek_v4_dspark_history_20260726/isolated_token_pairs/`
- Invalid intrusive-capture diagnosis:
  `.logs/deepseek_v4_dspark_history_20260726/ordered_control_retry/`

Both coalescing candidates remain explicit opt-in. Their row-level native and
graph gates pass, and their historical performance evidence remains useful,
but they are not acceptance-ready while the unchanged control is not greedy
exact.

## Exactness resolution and accepted candidate (2026-07-27)

The blocker above is resolved by an opt-in target-verifier-only correction.
The tokenwise indexer Q path called a one-token native quant kernel while
passing all six `indexer_weights` rows. Because the native output follows the
weight tensor shape, the concatenated result did not contain one initialized
folded-weight row per verifier token. The first tempting fix, slicing rows in
both target and draft layers, was rejected by the 3 x 100 gate. The accepted
mode is `VLLM_METAX_DSV4_TOKENWISE_INDEXER_WEIGHT_ROWS=target`: target layers
slice the matching row, while DSpark draft layers (`num_hidden_layers + i`)
retain the frozen path.

Production native differential and graph replay are exact. Fast 3 x 16 and two
independent 3 x 100 runs under TP=4, BF16 draft, k=5, greedy, prefix cache ON,
async ON, and PIECEWISE graph matched all frozen oracle hashes. Normal TPS was
`5.965803` and `5.977872`, a `6.65%` to `6.87%` improvement over `5.593861`.
The sampled repeat averaged `16.959%`, `17.265%`, `15.714%`, and `17.061%` GPU
utilization over the 49-sample decode window. There were no fallback markers;
PYNCCL and native tokenwise dispatch were observed.

Artifacts:

- `artifacts/kernel_validation/indexer_weight_slice_gate.log`
- `.logs/deepseek_v4_dspark_indexer_weight_rows_target_20260727/fast16/`
- `.logs/deepseek_v4_dspark_indexer_weight_rows_target_20260727/full3x100/`
- `.logs/deepseek_v4_dspark_indexer_weight_rows_target_20260727/full3x100_repeat_util/`

The next profiler pass should use this accepted environment. WQ-B launch
multiplication and rank-arrival skew remain the leading known hypotheses; the
historical `39,960` WQ-B `aten::mm` count must be remeasured rather than assumed.
