# Sparse attention O-projection 优化

## 会话目标

减少 MetaX O-proj 的 inverse-RoPE、BF16 group einsum、`wo_b` 和中间 copy，
建立适合 C500 的 fused quant/GEMM 路径。

## 已确认差异

MetaX：

- `vllm_metax/models/deepseek_v4/flashmla.py::_o_proj`
- `vllm_metax/models/deepseek_v4/ops/o_proj.py::deep_gemm_bf16_o_proj`
- inverse-RoPE 输出 BF16；
- `wo_a` 使用 BF16 grouped einsum；
- 再调用 `wo_b`。

CUDA：

- `/root/vllm-0.25rc1/vllm/models/deepseek_v4/nvidia/ops/o_proj.py`
- fused inverse-RoPE + FP8 quant；
- 使用带 scale 的 FP8 einsum；
- 架构相关 recipe 和 scale layout。

MetaX 默认没有被证明等价的 FP8 路径。不得为了复制 CUDA 结构而用软件 FP8
cast；只有 C500 原生 kernel 和真实测量证明收益后才能采用 FP8。

## 基线与验证边界

- 当前 TP=4、MTP=0、100-token decode 基线为 `15.2334 tok/s`、`65.65 ms/token`，四卡利用率 `16.2–16.4%`；旧 `11.6247 tok/s`/`86 ms` 仅作历史参考。
- MTP1 的 `17.8358 tok/s` 仅作观察；出现 token divergence 时不得验收。
- 验证链固定为 kernel-validator（差分/graph gate）→ test-runner → profiler（按需归因）→ benchmark-runner（同 workload 的 E2E）。isolated O-proj speedup 不等于 E2E。
- 10K prefill 的 Torch sparse MLA OOM 是独立阻断项，不能被 O-proj 或 decode 提升掩盖。

## 第一任务：阶段 benchmark 和 corpus

捕获真实 TP=4 O-proj 输入：attention output、positions、cos/sin cache、`wo_a`、
`wo_b`、group/head layout、当前中间 `o_bf16/z` 和最终输出。

分别测量：

1. inverse-RoPE；
2. `wo_a` grouped einsum；
3. `wo_b`；
4. 中间 allocation/copy/cast；
5. 整个 O-proj eager 和 graph replay。

建议新增：

- `tools/debug/diff_deepseek_v4_o_proj.py`
- `tools/debug/bench_deepseek_v4_o_proj.py`
- `tests/models/deepseek_v4/test_o_proj_diff.py`

## 实现路线

路线 A：若 C500 有经验证的原生 FP8 Tensor Core/DeepGEMM 路径，移植 CUDA 的
fused inverse-RoPE+quant 和 scaled einsum，同时保持 scale layout 固定且可捕获。

路线 B：若 FP8 不成熟，使用 BF16 MetaX MMA/MCTLaSS：

- inverse-RoPE 写入 GEMM 可直接消费的 shared/packed layout；
- 融合 inverse-RoPE 与 `wo_a` GEMM prologue；
- 将 `wo_a` epilogue 与 `wo_b` 输入布局衔接，消除中间 copy；
- 对 batch-one 固定 shape 做专用 kernel，不引入动态 autotune。

## 验收指标

- 真实 corpus 最终 BF16 输出满足模型 oracle；
- graph replay、指针和 workspace 稳定；
- O-proj kernel/拷贝数量至少减半；
- isolated O-proj 至少 2x 加速，或以数据证明受 `wo_b`/带宽硬上限限制；
- TP=4 greedy IDs 完全一致；
- 报告 FP8/BF16 路线的准确硬件证据，不使用未公开 C500 峰值推断。

## 当前证据（2026-07-13）

已增加默认关闭的真实输入 capture/replay hook。它只在
`VLLM_METAX_DSV4_O_PROJ_CAPTURE_DIR` 设置且不在 graph capture 时运行，并保存
attention output、实际位置与 compact cos/sin 行、`wo_a/wo_b`、`o_bf16`、`z`
和最终 BF16 输出。TP=4 PIECEWISE 16-token 运行得到两个 rank-0 payload：
`.logs/plan03_oproj_corpus_tp4/rank0_call0.pt` 和 `rank0_call1.pt`；运行日志为
`.logs/plan03_oproj_capture_tp4.log`。

离线 replay 两个 payload 均通过 Torch oracle，`o_bf16`、`z` 和 `output` 均
bitwise 相同；汇总为 `.logs/plan03_oproj_corpus_tp4/summary.json`。对应工具和
测试为 `tools/debug/diff_deepseek_v4_o_proj.py`、
`vllm_metax/models/deepseek_v4/ops/o_proj_debug.py` 和
`tests/models/deepseek_v4/test_o_proj_diff.py`。

同口径 profiler 基线（仅 profiler-instrumented，不作正常 TPS）位于
`.logs/plan03_oproj_baseline/summary.txt`：PIECEWISE inverse-RoPE 约
4.17 us、grouped `wo_a` 约 16.21 us、中间 copy 约 2.06 us、`wo_b` 约
15.62 us；每个 stage 215 次。当前没有 C500 原生 FP8 等价证据，也没有已验证
的 inverse-RoPE+GEMM 融合 kernel，因此尚未宣称 Plan 03 的性能验收完成。

曾实现 opt-in workspace 复用试验，并为 Triton inverse-RoPE 增加严格的
caller-owned layout 校验。同口径 TP=4 profile 的 O-proj kernel/copy 数量和
stage 时间没有改善，eager 反而回退；完整对比见
`.logs/plan03_oproj_workspace_benchmark/comparison.txt`。复审还确认单份缓存会在
并发 stream 或重叠 graph replay 间互相覆写，因此 workspace 复用实现和环境变量
已全部删除，不能作为 Plan 03 性能或正确性路径。

曾实现隔离的 C500 BF16 route-B probe，将 inverse-RoPE load 与 grouped `wo_a`
累加融合。它在 one-token `[1,16,512]` 上相对生产 inverse-RoPE + DeepGEMM
`wo_a` 只有 `0.082944 ms` 对 `0.089088 ms`（median，1.074x；P90 1.173x），
且真实非零输入不能 bitwise exact。该 probe、TileLang MMA 版本及其生产扩展编译
均已删除；历史验证保留在 `.logs/plan03_oproj_probe_validation_final.json`。

本机 GPU0 的实际持续 HBM roofline 已补测：4 GiB BF16 copy 的 median 为
`1.442 TB/s`，BF16 read/reduction 为 `1.596 TB/s`；`mx-smi` cropped window
观测约 `1.445 TB/s`。结果位于 `.logs/c500_bandwidth_roofline/results.json`。
按 `wo_b=[4096,2048]` BF16 的 16 MiB weight 计算，生产 profiler 的
`15.62 us` 约为 `1.07 TB/s`，说明 `wo_b` 已具有明显带宽约束，但这不能覆盖
前半段 candidate 的 correctness gate。

将 fused probe 临时接到单-token 生产路径后，TP=4 PIECEWISE frozen gate 从第二个
token 起分歧：实际 IDs 为
`[260,2112,1505,295,270,5148,294,260,2775,469,260,7231,396,344,1902,7451]`。
日志为 `.logs/plan03_fused_woa_tp4_piecewise_16tok.log`。因此生产 opt-in 接线已
移除；该候选是 correctness failure，不能用 kernel 数量或带宽证据验收。

数值根因诊断表明，scalar fused probe 有两处语义差异：首先遗漏了 inverse-RoPE
后的 BF16 舍入边界；补齐后，20-seed fuzz 从大范围误差收敛到少数 deterministic
1-ULP GEMV reduction-order 差异。即使按生产 `b16gemvt<64,8,4,8>` 的 launch、
vector 和 wave 拓扑近似，仍不能严格复现 vendor kernel 的 MMA/归约顺序；
TP=4 frozen IDs 继续分歧。因此 scalar 和 TileLang MMA probe 都保持隔离且不进入
生产 dispatch。

随后评估了 direct-BMM 候选：保留生产 inverse-RoPE BF16 输出，使用 caller-owned
`torch.bmm(..., out=z)` 调用 vendor BF16 GEMV，消除 compiled einsum 的
`triton_poi_fused_copy_0`。真实非零 payload、20/20 deterministic fuzz、CUDA
graph 10/10 replay 均 bitwise exact，稳定 pointer 且无 fallback；见
`.logs/plan03_direct_bmm_kernel_gate.json`。TP=4 PIECEWISE frozen 16-token IDs 完全
一致，见 `.logs/plan03_direct_bmm_tp4_piecewise_16tok.log`。

复审要求补齐 `T>1` 从 GEMV 进入 GEMM 形态后的交错 output stride 门禁。生产维度
`n_groups=2`、`heads_per_group=8`、`head_dim=512`、`o_lora_rank=1024` 下，
`T=[2,3,17,128]` 的真实 inverse-RoPE、`z` 和最终输出均与 grouped oracle
bitwise exact；每个 shape 的 eager 10 次与 CUDA graph 10 次 replay 全部稳定，
`z.transpose(0,1)` 的 caller-owned pointer 匹配，dispatch 为 CUDA
`aten::bmm` 且无 fallback。结果和 captures 位于
`.logs/plan03_direct_bmm_prefill_gate_20260713/`。

生产 trace 中 intermediate copy 从 215 次降为 0，但每次 O-proj 的主序列仅从
inverse-RoPE、`wo_a`、copy、`wo_b` 四个 stage 降为三个，未达到总 launch 数量
减半。O-proj median span 从 `86.430 us` 降至 `84.081 us`（-2.72%），P90 则从
`137.829 us` 升至 `149.042 us`。isolated prefix 为 1.738x，低于 2x；见
`.logs/plan03_direct_bmm_profile.summary.txt`。

早期跨时段正常基准曾显示 `16.111944 tok/s` 对旧基线 `15.233432 tok/s`，但 runner
没有输出逐次样本，不能用于最终推广。补齐 `DECODE_RUN_SECONDS` 和 P90 后，以两个
独立进程重跑完全相同的 TP=4、100-token、3 warmup、5 measured workload：

- baseline：median `16.137783 tok/s`，P90 `15.848810 tok/s`，CV `0.8545%`；
- direct-BMM：median `15.991597 tok/s`，P90 `15.906726 tok/s`，CV `0.8593%`；
- candidate 相对 baseline median TPS 为 `-0.906%`，两者最终 100 token IDs 相同。

完整 manifest、逐次 latency、telemetry 和原始输出位于
`.logs/plan03_final_normal_baseline_20260713/` 与
`.logs/plan03_final_normal_direct_bmm_20260713/`。因此不存在可验收的正常端到端
提速，direct-BMM 环境变量、生产 dispatch 和对应测试已删除。

本地 C500 API 审计没有发现可直接用于该阶段的 native fused O-proj API：
`/opt/maca/include/mcflashinfer/gemm/group_gemm.cuh` 和 `mctlassEx` 只提供
generic grouped/universal GEMM，已有 `csrc/metax_sparse/torch_bindings.cpp`
中的 fused RoPE 只覆盖 QK norm/KV-cache insertion。FP8 BMM API 也没有 inverse
RoPE 融合和模型 scale recipe。当前 venv 可直接导入 `deep_gemm.einsum`，但它只
覆盖 grouped GEMM，不覆盖 inverse-RoPE；`vllm_metax/utils/deep_gemm.py` 的
实际 worker dispatch 已由生产 trace 记录。没有 fused prologue/API 时，generic
GEMM 或软件 FP8 cast 不能替代精确路径；自定义 scalar/MMA 又无法复现 vendor GEMV
归约顺序。

## 最终验收结论（2026-07-13）

| 门禁 | 结果 | 结论 |
| ---- | ---- | ---- |
| 真实 corpus、graph、pointer、TP=4 IDs | direct-BMM 全部通过 | correctness 通过 |
| O-proj kernel/copy 总数至少减半 | 四个 stage 降为三个 | 未达到 |
| isolated 2x 或硬上限证据 | 1.738x；`wo_b` 有带宽约束但不是完整 O-proj 硬下限 | 未达到 |
| 正常 TP=4 端到端 | median TPS `-0.906%` | 未达到 |
| fallback-free 生产推广 | 候选已撤出 | 保持原生产路径 |

撤出候选后的默认生产路径已重新运行 TP=4 PIECEWISE 16-token frozen gate，退出码为
0，输出 IDs 与冻结序列完全一致，且没有设置任何 Plan 03 opt-in 环境变量。manifest、
stdout、summary 和 tracked diff hash 位于
`.logs/plan03_final_default_tp4_16tok_20260713/`。该 16-token 结果仅用于 correctness，
不报告为性能数据。

**状态：Plan 03 已完成，结论为 no-go。** 所有可用本机原生 API、BF16 scalar、
TileLang MMA、workspace 和 direct-BMM 路线均已完成验证；没有候选同时满足精确性和
性能门禁。生产保持原 inverse-RoPE + grouped einsum + `wo_b` 路径，不宣称任何
Plan 03 端到端提速。后续若出现带 inverse-RoPE prologue 的 C500 vendor GEMM API，
应建立新计划重新开放该优化，而不是复用已拒绝的 probe。

## 会话任务提示

```text
完成 deepseek-v4-c500-e2e/03-o-proj.md。先用真实 O-proj 输入拆分测量
inverse-RoPE、wo_a、wo_b 和 copy，再选择原生 FP8 或 BF16 fused MetaX 路线。
禁止用软件 FP8 模拟或只在随机输入上证明性能；必须通过 graph 和 TP=4 IDs。
```
