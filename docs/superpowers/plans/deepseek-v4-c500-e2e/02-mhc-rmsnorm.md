# MHC exact raw/pre 与 RMSNorm 融合

> **状态（2026-07-14）：已完成并接受 opt-in exact path；生产默认仍关闭。Plan 04 尚未开始。**

## 会话目标

保留已经完成的 exact MHC post MMA，继续把 Torch exact raw/pre、Sinkhorn、
pre-mix 和后续 RMSNorm 融合成 graph-safe MetaX kernel。

## 已完成，不要重复

- 真实 fused-post corpus：340 条，四 rank 各 85 条；
- `_mhc_post_exact_mma` 使用真正 MetaX TileLang MMA；
- direct differential：340/340 bitwise；
- CUDA graph replay：通过、输入输出指针稳定；
- TP=4 eager/PIECEWISE 16-token IDs：完全一致；
- post microbenchmark：`0.0222 ms` eager、`0.0350 ms` graph；
- 生产 opt-in：`VLLM_METAX_DSV4_MHC_EXACT_POST_MMA=1`，默认关闭。

详细证据：

- `../2026-07-12-deepseek-v4-mhc-post-task3.md`
- `.logs/dsv4_mhc_fused_post_prenorm_corpus_tp4/`
- `.logs/dsv4_mhc_post_bench_20260712.json`

## 当前缺口

exact-post opt-in 后仍使用 Torch exact raw/pre 边界。当前路径包含 FP32 linear/
GEMV、square-sum、affine、sigmoid、20 次 Sinkhorn、pre-mix、copy/cast；MetaX
模型还在 MHC 后分别执行 `attn_norm` 和 `ffn_norm`。CUDA 参考把 norm weight 和
eps 传入 MHC TileLang op，直接返回 normalized layer input。

当前已新增 opt-in model-boundary raw/pre/RMSNorm schema v2 capture，记录
layer/stage、生产 decode `residual_cur=[1,4,4096]`、raw GEMM 输入/输出、MHC
参数、pre-norm output、实际 norm weight/eps 和 normalized BF16 output。TP=4
PIECEWISE 两次调用/每 rank 的 compact corpus 位于
`.logs/dsv4_mhc_raw_norm_corpus_tp4/`，共 8 个约 `1.6 MiB` payload；capture run
保持 `[260,5036]` frozen IDs，日志为
`.logs/dsv4_mhc_raw_norm_capture_tp4_compact.log`。最初错误捕获 graph warmup 并
重复保存全 trace 的 `1.1 GiB/payload` corpus 已删除。下一步是让 raw replay CLI
支持 schema v2，并从输入重建中间 trace 与最终 RMSNorm oracle。

schema v2 replay 现已完成：CPU Torch raw/pre self-check `8/8` bitwise，实际模型
`ir.ops.rms_norm` 对 captured normalized BF16 output 也是 `8/8` bitwise。当前
`_mhc_pre_from_raw_exact_trace` 仍直接调用 Torch trace helper，因此所谓 TileLang
`8/8` 只是 oracle self-check，不能作为 native raw/pre kernel 证据。朴素 FP32
Torch RMSNorm 和 legacy `_C.rms_norm`
均在首个 payload 有 `1054/4096` BF16 元素不同、max abs `4.8828125e-4`，因此
不能作为融合 oracle。权威 summary 为
`.logs/dsv4_mhc_raw_norm_corpus_tp4_summary.json`、
`.logs/dsv4_mhc_raw_norm_corpus_tp4_ir_rms_summary.json` 和
`.logs/dsv4_mhc_raw_norm_corpus_tp4_tilelang_baseline.json`。

当前 replay CLI 已新增直接调用 `_mhc_pre_big_fuse` 的
`native-big-fuse` candidate，不经过 Torch exact helper。8 条生产 payload 的首次
native bitwise gate 为 `0/8`；8 条的第一个失败阶段均为最终
`sinkhorn_col_19`，而排在它之前比较的 `post_mix` 已通过。首条 payload 有 8 个
FP32 元素不同，首个差值 `9.313225746154785e-10`、最大差值
`5.960464477539063e-08`、最大 11 ULP。这把当前 native 缺口收窄到 Sinkhorn
softmax/反复行列归一化的算术顺序或数学 intrinsic；不能把这一结果表述成 native
raw/pre exact。证据位于
`.logs/dsv4_mhc_raw_norm_corpus_tp4_native_big_fuse_baseline.json` 和
`.logs/dsv4_mhc_raw_norm_corpus_tp4_native_big_fuse_baseline.log`。
尝试一次性输出全部 Sinkhorn 中间阶段的 TileLang debug kernel 时，MetaX
`LayoutInference` 在 lowering 阶段报 `no available layout found`；将输出展平并改为
compile-time stage probe 后仍相同，因此该 probe 没有产生数值证据，不能算作
kernel failure。失败日志为
`.logs/dsv4_mhc_raw_norm_corpus_tp4_native_sinkhorn_trace.log`。后续诊断必须使用
与现有 kernel 相同的固定小输出布局，逐阶段单独 probe。

## 基线与验证边界

- 当前 TP=4、MTP=0、100-token decode 基线为 `15.2334 tok/s`、`65.65 ms/token`，四卡利用率 `16.2–16.4%`；旧 `11.6247 tok/s`/`86 ms` 仅作历史参考。
- MTP1 的 `17.8358 tok/s` 仅作观察；出现 token divergence 时不得验收。
- 验证链固定为 kernel-validator（差分/graph gate）→ test-runner → profiler（按需归因）→ benchmark-runner（同 workload 的 E2E）。isolated kernel/模块 speedup 不等于 E2E。
- 10K prefill 的 Torch sparse MLA OOM 是独立阻断项，不能被本专题或 decode 提升掩盖。

## 代码入口

- `vllm_metax/models/deepseek_v4/ops/mhc/tilelang.py`
- `vllm_metax/models/deepseek_v4/ops/mhc/tilelang_kernels.py`
- `vllm_metax/models/deepseek_v4/ops/mhc/debug_diff.py`
- `vllm_metax/models/deepseek_v4/model.py`
- `tools/debug/diff_deepseek_v4_mhc_raw.py`
- `tests/models/deepseek_v4/test_mhc_raw_diff.py`
- CUDA 对照：`/root/vllm-0.25rc1/vllm/models/deepseek_v4/nvidia/model.py`

现有 focused 回归入口：

```bash
pytest -q tests/compat/test_deepseek_v4_mhc_backend.py \
  tests/models/deepseek_v4/test_mhc_raw_diff.py
```

## 实施顺序

1. 扩展真实 raw/pre corpus，保存 RMSNorm weight/eps 和最终 normalized 输出。
2. 给每个数学阶段保留 Torch trace，先找出必须严格保持的 FP32 顺序和 ULP。
3. 融合 raw GEMV 和 square-sum，优先消除 FP32 weight/raw GEMV 与重复 cast。
4. 融合 affine、sigmoid 和 Sinkhorn；不得用近似 softmax 改变 greedy IDs。
5. 融合 pre-mix reduction 与 RMSNorm，API 对齐 CUDA 的 `norm_weight/norm_eps`。
6. 从 `model.py` 删除独立 norm 调用前，先做端到端 differential 和 graph gate。
7. 保持 exact post caller-owned buffer，不重新引入 Torch post fallback。

允许拆成两到三个 kernel，因为全局 reduction 和 Sinkhorn 可能需要同步；目标是
最少的正确 kernel 集，而不是为了“一核”破坏可维护性或数值顺序。

## 验收指标

- raw/pre/RMSNorm 真实 corpus 全量通过；BF16 最终输出 bitwise；
- graph replay 和所有输出 pointer 稳定；
- MHC+RMSNorm decode 边界不超过三个 kernel；
- trace 中消除该阶段 generic Torch `sum/div/mul/copy` 簇；
- exact post 性能和 340/340 gate 不回退；
- TP=4 16-token IDs 完全一致，并给出每 token MHC 总延迟变化。

## Plan 02 16-token gate 证据（2026-07-13）

已实现默认关闭的 `VLLM_METAX_DSV4_MHC_EXACT_PRE_RMS=1` decode opt-in。该路径
只在 TP=4、单 token、`hc_mult=4`、`hidden=4096`、20 次 Sinkhorn 的 exact
contract 上启用；多 token dummy/prefill 保持原路径，其他单 token contract
fail closed。每层 exact workspace 由 Python caller 持有并复用，post 输出和
raw/pre/RMS 中间 buffer 在 graph replay 中保持稳定指针。

- native raw/pre/RMS corpus：8/8 bitwise，见
  `.logs/dsv4_mhc_raw_norm_corpus_tp4_native_three_kernel_exact_finalgate.json`；
- graph replay：2/2，`graph_replay=true`、`pointers_stable=true`，见
  `.logs/dsv4_mhc_raw_norm_corpus_tp4_native_three_kernel_exact_graph_finalgate.json`；
- C++ dtype/shape rejection：6/6，见
  `.logs/dsv4_mhc_native_three_kernel_rejection_finalgate.log`；
- native dispatch：`mhc_cast_sqrsum_kernel` -> `sgemvt_wave_kernel` ->
  `mhc_downstream_rms_kernel`，每个 rank 425 个连续 triple，无 generic Torch
  arithmetic 插入，见 `.logs/dsv4_mhc_plan02_production_profile.analysis.txt`；
- TP=4 PIECEWISE frozen 16-token gate：IDs 完全一致，见
  `.logs/dsv4_mhc_exact_pre_rms_tp4_piecewise_16tok_retry1.log`；
- exact post 历史 340/340、graph 和 post gate 保持通过，见已有
  `.logs/dsv4_mhc_fused_post_prenorm_corpus_tp4/` 与
  `.logs/dsv4_mhc_post_bench_20260712.json`。

上述证据只覆盖 16-token 快速 gate；profiler TPS 仅作 instrumented evidence，
不作为正常吞吐基线。

## 100-token 累计消融复核（2026-07-13）

Plan 02 默认关闭，`bc0aa3c -> 36078e9` 默认路径 median TPS 仅
`15.9486 -> 15.9160`（`-0.204%`），低于 run spread，按零收益处理。

显式启用 `MHC_BACKEND=tilelang`、`EXACT_POST_MMA=1` 和
`EXACT_PRE_RMS=1` 后，median TPS 为 `25.3571`，相对 default 提升
`59.318%`；但 TP=4 100-token IDs 从 0-based index 22 开始分叉。此前 16-token
gate 没有覆盖首次错误位置。

**历史状态（已由 2026-07-14 验收 superseded）：Plan 02 重新打开，correctness blocker。** 保持所有生产开关默认关闭；
本轮只做 current-reference comparison，没有独立 100-token frozen oracle。在建立
该 oracle 且 candidate 完全一致前，不得推广或把 59.318% 计入累计收益。
完整证据见
[`2026-07-13-plan01-03-cumulative-ablation.md`](2026-07-13-plan01-03-cumulative-ablation.md)
和 `.logs/plan123_ablation_wrapper_20260713/`。

## 2026-07-14 sigmoid 修复与 opt-in 最终验收

2026-07-13 的 index-22 divergence 根因已定位：`mhc_sigmoid` 使用
`__builtin_mxc_rcpf(1+expf(-x))`，其倒数近似改变了 exact 数值顺序。现已改为
`__fdiv_rn`。修复后的证据如下：

- 1892 个 rank0 真实 late payload 在每个已检查 stage 均 bitwise 一致，max abs/rel
  均为 `0`；`.logs/plan02_sigmoid_fdiv_stage_diff_20260714/`。
- graph gate `9/9`，指针稳定且无 allocation；`.logs/plan02_sigmoid_fdiv_graph_gate_20260714/`。
- TP=4 PIECEWISE exact-enabled 23-token gate 通过；fresh exact-off-vs-on
  100-token 的全部 token IDs 完全一致，first mismatch 为 none（修复前 index 22
  的 reference ID 为 `372`）；`.logs/plan02_sigmoid_fdiv_e2e23_20260714/`、
  `.logs/plan02_sigmoid_fdiv_e2e100_20260714/`。
- dispatch 保持 native path，无 fallback。

同一 workload 的正常、非 profiler、5-run benchmark：

| Path | Median TPS | P90 TPS | Median/P90 latency | GPU avg |
| ---- | ---------: | -------: | ------------------: | ------- |
| exact off | 15.965887 | 15.914479 | 6.263354 / 6.283586 s | `[16.13,16.03,16.16,16.10]%` |
| exact on | 25.875913 | 25.579406 | 3.864598 / 3.909395 s | `[24.50,24.60,24.20,24.50]%` |

exact on 相比 off 为 `+62.07%` TPS、`-38.30%` median latency。原始结果位于
`.logs/plan02_sigmoid_fdiv_benchmark_20260714/`。该结论仅适用于显式 opt-in
exact path；不得静默改变 runtime default。

## 2026-07-14 Plan02-on 最新 decode profile

在同一 frozen TP=4、PIECEWISE、MTP=0、`MAX_TOKENS=100`、exact-on workload
上，profiler 在 delay 4 后采集了 5 个 steady `execute_context`；trace 不含
token IDs，因此这里只报告 stage/时间归因，不声称具体 token 结果。rank0 device
self 为 `175.944 ms / 5 steps`：

- MHC exact grouped：`84.266 ms`（`47.9%`）；其中 downstream RMS 为
  `59.095 ms / 425`（`33.59%`，`139.048 us`/次）。
- MoE：`29.474 ms`（`16.8%`）；MCCL allreduce：`6.599 ms / 435`
  （`3.75%`）。
- graph internal gap：`3.925 ms/step`；inter-step gaps 合计
  `9.253 ms`；约 `2348` kernels/token、`61–62` memcopies/token、`44`
  graph launches/token。
- trace busy：graph 内约 `89.2–89.9%`，完整窗口 `85.9%`。

这是 profiler-instrumented TPS 证据，不替代同 workload 正常 benchmark 的
`25.8759 TPS`。下一项应验证 downstream RMS MHC kernel/fusion；不要因
`mx-smi` 低采样把该瓶颈误判为 host-only。原始 artifacts：
`.logs/plan02_on_decode_profile_20260714/{summary.txt,analysis.json,trace/}`。
Plan 04 仍未开始。

## 2026-07-14 downstream RMS Sinkhorn 优化验收

在主树 SHA `28ea10c` 上，`mhc_downstream_rms_kernel` 将 4x4 Sinkhorn 的
行/列归一化改为 `tid<4` 并行；每个四元素求和保持原操作顺序，保留
`__fdiv_rn` 和 20 次 repeat，未改变后续 BF16 pre-norm/RMS contract。该
优化通过全部验收，状态为 **accepted**（仅适用于显式 Plan 02-on path，
runtime 默认仍关闭；本记录不伪造 commit）。

- 21 tests passed；1892/1892 real late payloads bitwise，graph replay 9/9，
  pointers/allocations stable，无 fallback。
- Isolated latency：eager median/P90 `151.81/153.09 -> 81.92/83.456 us`；
  graph median/P90 `166.14/167.68 -> 97.024/98.816 us`。
- TP=4 PIECEWISE：23-token gate（index 22 为 `372`）和 100-token IDs 均
  byte-for-byte 通过，native dispatch，无 fallback。
- 同一正常 5-run workload：`25.875913 TPS`、`3.864598 s` median latency
  -> `30.640709 TPS`、`3.263632 s`；`+18.414%` TPS、`-15.551%` latency；
  GPU average `[29.7333, 29.5333, 29.5333, 29.6000]%`。

After-profile 对比显示 downstream RMS `59.095 -> 29.413 ms`（`-50.23%`，
`139.048 -> 69.208 us/call`），exact MHC grouped `84.266 -> 54.550 ms`
（`-35.26%`）。新 trace 的 MCCL residency 呈 rank-asymmetric：ranks 0-2
约 `48 ms`，rank 3 为 `3.289 ms`。该 residency 是同步/等待证据，不能
作为纯通信链路时间相加；下一诊断已拆分为
[`03.5-mccl-rank-skew.md`](03.5-mccl-rank-skew.md)，用于区分 rank arrival、
collective transfer tail 和 graph/host gap。Plan 04 仍未开始。

主要 artifacts：

- `.logs/mhc_downstream_sinkhorn_parallel_main_final_20260714/`
- `.logs/mhc_downstream_sinkhorn_parallel_main_e2e_20260714/`
- `.logs/mhc_downstream_sinkhorn_parallel_after_profile_20260714/`

## 会话任务提示

```text
完成 deepseek-v4-c500-e2e/02-mhc-rmsnorm.md。不要重新实现或调研 exact MHC
post；它已通过 340/340 和 graph gate。基于真实 raw/pre payload，把 Torch
GEMV、sqrsum、mix/Sinkhorn、pre-mix 和 RMSNorm 融合为 graph-safe MetaX
kernel，并保持 16-token IDs 完全一致。
```
