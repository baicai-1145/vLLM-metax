# DeepSeek-V4 DSpark BF16 adaptation

> **Status:** Phase 0-5 BF16 functional and greedy-exactness gates passed on
> 2026-07-26. Profiling is complete; performance repair, long-prefill, and
> quantized-draft gates remain open. See
> [10-dspark-performance-profile-handoff.md](10-dspark-performance-profile-handoff.md).

## Goal

Run the official three-layer DeepSeek-V4 DSpark draft on four MetaX C500 GPUs
with TP=4, BF16 draft weights, BF16 KV cache, normal PIECEWISE graph mode, and
no silent dtype or attention fallback.

The merged checkpoint is
`/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-DSpark-staging`. It preserves
the target weights and adds three BF16 `mtp.*` shards. The original target
checkpoint remains unchanged.

## Implemented path

- `DSparkDraftModel` resolves to the MetaX adapter and constructs the local
  `DeepseekV4DecoderLayer` rather than the upstream NVIDIA decoder.
- Context KV accepts BF16 only and dispatches
  `_C::fused_deepseek_v4_qnorm_rope_kv_rope_insert` with the MetaX two-dimensional
  per-block cache ABI. FP8, INT8, FP32, and missing native op paths fail closed.
- DSpark non-causal SWA includes the trailing context window and all query
  tokens, including future positions. Its index width is padded to a multiple
  of 128 and its graph buffer is allocated once.
- The MetaX Triton 3.0 staged-writer patch preserves the upstream fused
  multi-group algorithm while replacing the runtime pointer dtype with the
  existing fixed `tl.int32` pointer helper.
- The DeepSeek runner accepts explicit `SPECULATIVE_METHOD=dspark` and
  `SPECULATIVE_MODEL`. Upstream DSpark loads draft weights from the merged
  target checkpoint, so the smoke uses the staging checkpoint as `MODEL`.

## Phase gates

| Phase | Result | Evidence |
| --- | --- | --- |
| 0: frozen oracle | Pass | TP=4, MTP=0, PIECEWISE, three identical 16-token hashes and finish reasons |
| 1: registry and schema | Pass | MetaX registry, local decoder, BF16-only dispatch, three real staging draft layers |
| 2: kernel and graph | Pass | Production `N=8`, local `H=16`, `D=512`, BF16 insert has zero error; context and SWA replay twice with stable pointers |
| 3: TP=4 smoke | Pass | Draft loaded 71 parameters; target and DSpark graphs captured; two identical 16-token generation replays |

Phase 0 artifacts:
`.logs/deepseek_v4_dspark_phase0_oracle_20260725/`.

Phase 2 artifacts:
`.logs/deepseek_v4_dspark_phase2_kernel_20260725/`.

Phase 3 first failure and final accepted run:
`.logs/deepseek_v4_dspark_phase3_smoke_20260725/run.log` and
`.logs/deepseek_v4_dspark_phase3_smoke_20260725/retry4/`.

The accepted run used TP=4, `MAX_MODEL_LEN=512`, `MAX_TOKENS=16`,
`MIN_TOKENS=16`, `MAX_NUM_BATCHED_TOKENS=8192`, prefix caching enabled,
`GPU_MEM=0.9`, `NUM_SPECULATIVE_TOKENS=5`, asynchronous scheduling, and
PIECEWISE graph mode. Its two output hashes were both
`35816a5e17750756ef57a80b6e647652e33b11e2052bd20e927860206a741241`.
The production cache is a packed multi-group view: blocks may have gaps, while
elements within each block remain contiguous. The BF16 insert validates this
layout and passes the actual block stride to the native kernel. Retry 2 exposed
and rejected an overly strict whole-tensor contiguity check; the corrected
packed-stride regression and retry 4 both pass.

## Phase 4: final-token exactness

Phase 4 passes on TP=4 with the unchanged 16-token correctness workload. Both
DSpark replays match the Phase 0 token IDs and `length` finish reason exactly.
The DSpark verifier width is six rows (`k=5` plus correction/bonus), so the
native tokenwise exact paths now admit six rows; the MHC N=6 differential is
bitwise exact for N=1, 2, 5, and 6, including 20 repeated executions and 20
PIECEWISE graph replays with stable pointers.

Artifacts:

- `.logs/deepseek_v4_dspark_phase4_20260725/exact_candidate/run.log`
- `.logs/deepseek_v4_dspark_phase4_20260725/mhc_n6_kernel/final_validation.json`
- `.logs/deepseek_v4_dspark_phase4_20260725/target_tensor_audit.json`

## Phase 5: BF16 acceptance quality

Phase 5 freezes
`tools/debug/corpora/deepseek_v4_mtp_new_heldout_3_20260724.jsonl`
(`sha256=232696e55f01f85e3bb480e2550b277d9456c638964adaf22eadac722106be50`)
and generates exactly 100 greedy tokens for each of its three prompts. A fresh
MTP=0 TP=4 PIECEWISE run from the current worktree is the per-prompt target
oracle. The DSpark candidate retains TP=4, PIECEWISE, prefix caching,
asynchronous scheduling, `GPU_MEM=0.9`, and `k=5`.

The hard gate requires all 300 committed token IDs and all finish reasons to
match the target oracle, with no crash, empty output, or ID-0 tail. Acceptance
rate, mean accepted length, per-position acceptance, and a 95% Wilson interval
are reported as descriptive BF16 baseline metrics; no minimum acceptance-rate
threshold is invented in this phase.

Phase 5 is **Pass**. All three 100-token outputs and all `length` finish reasons
match the frozen MTP=0 oracle exactly. The resulting token hashes are
`0e57b68b...17d8`, `7d7d327c...bc02`, and `97422c78...a869`, identical to the
oracle hashes. A stats-enabled replay also preserves all 300 token IDs and
reports 173 accepted of 550 drafted tokens (`31.45%`, 95% Wilson interval
`27.71%--35.45%`) across 110 verification cycles. Mean accepted draft length is
`1.57`; vLLM's mean acceptance length, including the target bonus token, is
`2.57`. Per-position acceptance is `65.45%`, `39.09%`, `27.27%`, `13.64%`, and
`11.82%`.

A larger acceptance-quality replay supersedes the three-prompt acceptance-rate
estimate. It freezes
`tools/debug/corpora/deepseek_v4_mtp_acceptance_100_20260724.jsonl`
(`sha256=704bc2fb76e762607d1fbd56f409e1aa3e46567466e29a98855ee7ba5658b85d`),
runs 100 distinct prompts for 32 greedy tokens each, and captures every
scheduler verification cycle after setting request warmup to zero. Under the
accepted prefix-cache-enabled workload, all 100 DSpark outputs match a fresh
MTP=0 oracle. The exact capture contains 1,049 cycles, 2,232 accepted of 5,245
drafted tokens (`42.55%`, 95% Wilson interval `41.22%--43.90%`), and a mean
acceptance length of `3.13` including the target bonus token. Per-position
acceptance is `73.88%`, `53.57%`, `38.51%`, `27.74%`, and `19.07%`.

The same 100-prompt diagnostic with prefix caching disabled has essentially the
same acceptance rate (`42.48%`) but is not exact: prompt index 58 diverges at
output token index 24, while the other 99 prompts match. The MTP=0 target is
invariant for that prompt with prefix caching enabled or disabled, so this is a
DSpark cache-disabled residual defect. It does not satisfy an acceptance gate
and is not used as the target workload.

A separate 20-prompt shared-prefix stress corpus
(`sha256=dff9c726c66e1e736218e167fd9c234caef715f14b924b0294a2b7e0313c78d3`)
forces real prefix-cache reuse. DSpark reaches an observed `54.7%` prefix-cache
hit rate and matches the MTP=0 oracle for all 20 outputs. Its acceptance capture
contains 208 cycles and 429 accepted of 1,040 drafted tokens (`41.25%`, 95%
Wilson interval `38.30%--44.27%`), with mean acceptance length `3.06` including
the target bonus token. This closes the prefix-cache-enabled exactness risk for
the current BF16 gate; it does not close the cache-disabled residual above.

The failure was cumulative target-verifier numerical drift, not asynchronous
rollback or sparse-MLA cache layout. At position 72, serial and six-row target
execution had identical causal indices, lens, and all 73 physical BF16 cache
rows read by sparse MLA. The remaining query difference was three BF16 values
after batched WQ-B projection; tokenwise target WQ-B made the query and native
attention output bitwise exact. At position 70, the next first difference was
six BF16 values after the batched O projection. Finally, batched initial MHC
mix coefficients polluted committed target KV rows in later layers. The retained
path therefore combines native row-exact initial MHC plus target-only tokenwise
WQ-B and O projection. DSpark draft layers retain their original batched MHC,
WQ-B, and O-projection behavior.

Artifacts:

- `.logs/deepseek_v4_dspark_phase5_20260725/target_oracle/retry1/summary.json`
- `.logs/deepseek_v4_dspark_phase5_20260725/dspark_bf16/run.log`
- `.logs/deepseek_v4_dspark_phase5_20260725/dspark_bf16/exactness_summary.json`
- `.logs/deepseek_v4_dspark_phase5_20260725/dspark_bf16/acceptance_metrics.json`
- `.logs/deepseek_v4_dspark_phase5_20260725/first_cycle_ffn_candidate/run.log`
- `.logs/deepseek_v4_dspark_phase5_20260725/force_reject_capture/capture/rank0.jsonl`
- `.logs/deepseek_v4_dspark_phase5_20260725/force_reject_sync_diagnosis/run.log`
- `.logs/deepseek_v4_dspark_phase5_20260725/layer_bisect_diff/`
- `.logs/deepseek_v4_dspark_phase5_20260725/layer_bisect_wqb_insert_diff/`
- `.logs/deepseek_v4_dspark_phase5_20260725/first_cycle_mhcpre_q6_candidate/run.log`
- `.logs/deepseek_v4_dspark_phase5_20260726/full_exact_wqb_oproj_gate/run.log`
- `.logs/deepseek_v4_dspark_phase5_20260726/final_heldout_3x100/run.log`
- `.logs/deepseek_v4_dspark_phase5_20260726/final_heldout_3x100_stats/run.log`
- `.logs/deepseek_v4_dspark_phase5_20260726/final_heldout_3x100_stats/acceptance_metrics.json`
- `.logs/deepseek_v4_dspark_acceptance_100x32_20260726/study_summary.json`
- `.logs/deepseek_v4_dspark_acceptance_100x32_20260726/formal_prefix_on/`
- `.logs/deepseek_v4_dspark_acceptance_100x32_20260726/formal_prefix_off/`
- `.logs/deepseek_v4_dspark_acceptance_100x32_20260726/target_oracle_prefix_on/`
- `.logs/deepseek_v4_dspark_acceptance_100x32_20260726/shared_prefix_stress/`
- `artifacts/dspark_mhc_initial_pre_row_equivalence_20260726.json`

The following claims remain outside Phase 4-5:

- 100-token normal throughput, latency, or GPU utilization;
- 1K and 10K prefill integration;
- W8A8 or W4A16 draft quality;
- a completely fallback-free MHC path.

The integration logs contain the existing, explicit `torch_prefill` MHC path for
non-decode shapes before the TileLang decode path. It is observable and was not
used to claim native-only acceptance. It must be replaced before a
fallback-free claim.

## Next action

Optimize the verification-row TP collective path first. The current profile
shows 517 all-reduce events per DSpark step versus 87 for target, with BF16
all-reduce consuming 70.68-75.44% of self CUDA time. Preserve the exact
tokenwise arithmetic while coalescing launches and synchronization; do not
replace it with the previously divergent plain batched path. Then address the
60.55x `aten::mm` and 17.93x device-copy call multiplication. Re-run the frozen
TP=4 exactness gate before any normal performance comparison. Long-prefill and
a completely fallback-free MHC path remain prerequisites for a native-only
claim. Evaluate W8A8 or W4A16 only after BF16 performance correctness passes.
