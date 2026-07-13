# Plan 01--03 累计消融审计

> **历史结论（2026-07-13，已由 2026-07-14 证据 superseded）：只有 Plan 01 对当前默认 decode 路径产生了可测的累计变化。Plan 02
> 默认关闭；显式启用虽快 59.3%，但从第 23 个 token 起分叉。Plan 03 为 no-go。**

## 审计结果

所有结果均为正常、非 profiler 的 TP=4 PIECEWISE 100-token decode，固定
`MTP=0`、`NUM_SPECULATIVE_TOKENS=0`、`GPU_MEM=0.9`、prefix cache 开启、
3 次 warmup 和 5 次 measured runs。每组均从对应 worktree cwd 调用官方
`tools/run_deepseek_v4_mtp_generate.sh`，并记录实际源码 import path。

| 版本 | Revision | Median TPS | Median / P90 latency | CV | Token IDs |
| ---- | -------- | ---------: | --------------------: | -: | --------- |
| current-before | `15a8ada` | 16.2325 | 6.160491 / 6.218778 s | 0.50% | current-reference |
| pre-Plan01 | `82e4c54` | 15.1398 | 6.605096 / 6.716849 s | 0.69% | mismatch @31 |
| Plan01 | `bc0aa3c` | 15.9486 | 6.270132 / 6.317445 s | 0.42% | current-reference |
| Plan02 default | `36078e9` | 15.9160 | 6.282978 / 6.286998 s | 0.20% | current-reference |
| Plan02 enabled | `36078e9` | 25.3571 | 3.943671 / 3.972285 s | 1.13% | mismatch @22 |
| current-after | `15a8ada` | 16.0112 | 6.245634 / 6.310178 s | 0.65% | current-reference |

current 前后夹具的 TPS 漂移为 `-1.363%`，小于两组 observed run spread；六组
测量期间存在有限时间漂移，所有小于约 2% 的版本差异按噪声处理。

## 如何解释增量

### Plan 01

`pre-Plan01 -> Plan01` 的 median TPS 变化为 `+5.342%`，median latency 变化为
`-5.071%`。Plan 01、Plan 02 default 和两个 current 夹具的 100-token IDs 完全
相同，因此 Plan 01 之后的默认生产序列稳定。

但是 pre-Plan01 从 0-based index 31 开始生成不同 token：旧序列局部为
`[270,2004,582,25910,20828,107265,35]`，current-reference 序列为
`[270,2004,582,57712,296,107265,671]`。MoE routing 会随 token/hidden state
变化，因此 `+5.342%` 不是严格的同输出序列 A/B；它是当前代码历史下的累计
端到端变化。Plan 01 已有相同 current-reference 序列下的 sparse-MLA stage
profiler 和
native dispatch 证据，可支持“Plan 01 是主要贡献来源”，但不能把全部 5.342%
都归因给单个 kernel。

### Plan 02（2026-07-13 历史结果，superseded）

Plan 02 的完成候选由以下显式开关控制，默认全部关闭：

```text
VLLM_METAX_DSV4_MHC_BACKEND=tilelang
VLLM_METAX_DSV4_MHC_TILELANG_OPS=fused
VLLM_METAX_DSV4_MHC_EXACT_POST_MMA=1
VLLM_METAX_DSV4_MHC_EXACT_PRE_RMS=1
VLLM_METAX_DSV4_MHC_REQUIRE_EXACT_TILELANG=1
```

因此 `Plan01 -> Plan02 default` 为 `-0.204%`，低于单组 run spread，应按零
收益处理。显式启用后 median TPS 提升 `59.318%`，但从 0-based index 22 开始
分叉：candidate 局部为 `[1499,1257,339,795,223,20,16]`，current-reference
序列为
`[1499,1257,339,372,1999,344,270]`。

此前 16-token gate 没有覆盖首次错误位置，所以不能证明 100-token correctness。
Plan 02 必须重新打开，保持默认关闭。本轮没有运行独立的 100-token frozen oracle；
在建立该 oracle 且 candidate 与其完全一致前，不得推广或计入累计性能。

### Plan 02 最终验收（2026-07-14）

2026-07-13 的 index-22 divergence 根因是 `mhc_sigmoid` 使用
`__builtin_mxc_rcpf(1+expf(-x))`；现已改为 `__fdiv_rn`。修复后，1892 个 rank0
真实 late payload 的每个已检查 stage 均 bitwise 一致，max abs/rel 均为 `0`；graph
gate `9/9` 稳定且无 allocation。TP=4 PIECEWISE exact-enabled 23-token gate
通过，fresh exact-off-vs-on 100-token 的全部 IDs 完全一致，first mismatch 为
none（修复前 index 22 的 reference ID 为 `372`），且无 fallback。证据目录为
`.logs/plan02_sigmoid_fdiv_stage_diff_20260714/`、
`.logs/plan02_sigmoid_fdiv_graph_gate_20260714/`、
`.logs/plan02_sigmoid_fdiv_e2e23_20260714/` 和
`.logs/plan02_sigmoid_fdiv_e2e100_20260714/`。

同一 workload 的正常、非 profiler、5-run benchmark：exact off median/P90 TPS
`15.965887/15.914479`，median/P90 latency `6.263354/6.283586 s`，GPU avg
`[16.13,16.03,16.16,16.10]%`；exact on 为
`25.875913/25.579406` TPS、`3.864598/3.909395 s` latency、
`[24.50,24.60,24.20,24.50]%`。即 `+62.07%` TPS、`-38.30%` median latency。
原始 benchmark 位于 `.logs/plan02_sigmoid_fdiv_benchmark_20260714/`。Plan 02
状态为完成/接受（仅 opt-in exact path），runtime default 不变。

### Plan 03

Plan 03 direct-BMM 的独立同口径正常基准为 `-0.906%`，候选已撤出；证据位于
`.logs/plan03_final_normal_baseline_20260713/` 和
`.logs/plan03_final_normal_direct_bmm_20260713/`。本次累计消融中，`36078e9`
default 与 current 夹具的差异也处于时间漂移范围内；Plan 03 对当前生产 TPS 的
贡献按零处理。

## 为什么此前看起来没有提升

1. Plan 01 的累计变化约为 5%，不是数量级提升，GPU 利用率仍然很低。
2. Plan 02 的高收益代码默认关闭，而且完整 100-token correctness 失败。
3. Plan 03 只消除了约 2 us/call 的 copy，正常端到端没有收益。
4. 早期结果混用了跨时段 baseline、profiler TPS 和 isolated latency。
5. 之前没有用 current-before/current-after 夹具约束系统漂移，也没有对所有候选跑
   100-token current-reference comparison。

## 后续门禁（2026-07-13 历史要求，superseded）

- 暂停 Plan 04，先修复或正式拒绝 Plan 02 的 index-22 divergence。
- 所有 decode 优化必须运行 TP=4 100-token current-reference comparison，并建立
  独立 frozen oracle；16-token 只作快速预检。
- 每个计划必须提供 current-before/current-after、至少 5 次逐次 latency 和 CV。
- 输出序列不同时，结果标记为不同 workload，不作为严格性能归因。
- profiler TPS、isolated kernel speedup 和正常端到端 TPS 必须分栏记录。

当前状态：Plan 02 opt-in exact path 已通过上述 2026-07-14 门禁；Plan 04 未开始。

## Artifacts

- 总比较：`.logs/plan123_ablation_wrapper_20260713/comparison.json`
- current 夹具：`.logs/plan123_ablation_wrapper_20260713/current_before/`、
  `.logs/plan123_ablation_wrapper_20260713/current_after/`
- Plan 01：`.logs/plan123_ablation_wrapper_20260713/pre_plan01/`、
  `.logs/plan123_ablation_wrapper_20260713/plan01/`
- Plan 02：`.logs/plan123_ablation_wrapper_20260713/plan02_default/`、
  `.logs/plan123_ablation_wrapper_20260713/plan02_enabled/`
- Plan 03：`.logs/plan03_final_normal_baseline_20260713/`、
  `.logs/plan03_final_normal_direct_bmm_20260713/`

正式累计消融的六个目录均包含 revision、源码 import path、官方 wrapper、
`TORCH_LIB`、最终 `LD_LIBRARY_PATH`、workload manifest、原始 stdout、逐次
`DECODE_RUN_SECONDS`、GPU telemetry 和 summary。Plan 03 artifacts 遵循其原始
独立 manifest，不包含上述 worktree 字段。`.logs/plan123_ablation_20260713/` 的 direct-Python
预跑和 `.logs/plan123_ablation_wrapper_20260713_invalid_cwd/` 的错误 cwd 尝试均
明确排除；正式六组从各自 worktree cwd 调用官方 wrapper，import path 门禁通过且
退出码均为 0。
