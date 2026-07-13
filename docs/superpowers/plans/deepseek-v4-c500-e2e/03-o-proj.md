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

另实现了隔离的 C500 BF16 route-B probe：
`csrc/metax_sparse/o_proj_probe.cu` 将 inverse-RoPE load 与 grouped `wo_a`
累加融合，注册为 `metax_o_proj_probe::fused_bf16_out`，不接入生产 dispatch。
通过项目正常 CMake/MACA 构建后，真实 corpus 两个 payload 均 bitwise exact；
7/7 malformed input rejection、caller-owned output、CUDA graph 5/5 replay 和
稳定 pointer 均通过，完整证据为 `.logs/plan03_oproj_probe_validation_final.json`。

但该 kernel 在 one-token `[1,16,512]` 上相对生产 inverse-RoPE + DeepGEMM
`wo_a` 只有 `0.082944 ms` 对 `0.089088 ms`（median，1.074x；P90 1.173x），
达不到 2x。两个真实 capture 的 `o`/`z` 为零；非零 synthetic 输入虽在宽松
allclose 下通过，但不是 bitwise exact。因此该 probe 不能进入生产路径，也不能
满足 Plan 03 的性能验收。

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

已增加默认关闭的精确路径 `VLLM_METAX_DSV4_O_PROJ_DIRECT_BMM=1`：保留生产
inverse-RoPE BF16 输出，随后使用 caller-owned `torch.bmm(..., out=z)` 直接调用
vendor BF16 GEMV，消除 compiled einsum 的 `triton_poi_fused_copy_0`。真实非零
payload、20/20 deterministic fuzz、CUDA graph 10/10 replay 均 bitwise exact，
稳定 pointer 且无 fallback；见 `.logs/plan03_direct_bmm_kernel_gate.json`。
TP=4 PIECEWISE frozen 16-token IDs 完全一致，见
`.logs/plan03_direct_bmm_tp4_piecewise_16tok.log`。

复审要求补齐 `T>1` 从 GEMV 进入 GEMM 形态后的交错 output stride 门禁。生产维度
`n_groups=2`、`heads_per_group=8`、`head_dim=512`、`o_lora_rank=1024` 下，
`T=[2,3,17,128]` 的真实 inverse-RoPE、`z` 和最终输出均与 grouped oracle
bitwise exact；每个 shape 的 eager 10 次与 CUDA graph 10 次 replay 全部稳定，
`z.transpose(0,1)` 的 caller-owned pointer 匹配，dispatch 为 CUDA
`aten::bmm` 且无 fallback。结果和 captures 位于
`.logs/plan03_direct_bmm_prefill_gate_20260713/`。

生产 trace 中 intermediate copy 从 215 次降为 0；O-proj median span 从
`86.430 us` 降至 `84.081 us`，但 all-reduce P90 波动使全段 P90 从
`137.829 us` 升至 `149.042 us`。isolated prefix median 为 `0.049920 ms`，
相对 baseline `0.086784 ms` 为 1.738x。证据位于
`.logs/plan03_direct_bmm_profile.summary.txt`。该路径解决了数值 correctness 和
中间 copy，但尚未单独证明 Plan 03 的全部端到端性能验收完成。

同一正常非 profiler TP=4 workload（100-token、3 warmup、5 measured runs）下，
direct-BMM median 为 `16.111944 tok/s`、`62.065757 ms/token`，相对冻结基线
`15.233432 tok/s` 提升 `5.7670%`；30 个持续 decode telemetry samples 的四卡
平均利用率为 `15.767%/15.833%/15.767%/15.867%`。原始日志、manifest 和
summary 位于 `.logs/plan03_direct_bmm_normal_benchmark_20260713/`。runner 未输出
decode 的逐次样本或 P90，因此不宣称正常 P90；TP=4 100-token 结果只用于性能，
数值结论仍由 frozen token gate 和上述 kernel gates 给出。

**状态：Plan 03 数值正确性已通过，性能验收仍在进行。** direct-BMM 继续默认
关闭；在 kernel/copy 数量减半指标得到明确解释并完成性能决策前，不进入 Plan 04。

本地 C500 API 审计没有发现可直接用于该阶段的 native fused O-proj API：
`/opt/maca/include/mcflashinfer/gemm/group_gemm.cuh` 和 `mctlassEx` 只提供
generic grouped/universal GEMM，已有 `csrc/metax_sparse/torch_bindings.cpp`
中的 fused RoPE 只覆盖 QK norm/KV-cache insertion。FP8 BMM API 也没有 inverse
RoPE 融合和模型 scale recipe。当前 venv 可直接导入 `deep_gemm.einsum`，但它只
覆盖 grouped GEMM，不覆盖 inverse-RoPE；`vllm_metax/utils/deep_gemm.py` 的
实际 worker dispatch 仍需在生产 profile 中单独记录。没有这些 native/dispatch 证据，不能
把 generic GEMM 或软件 FP8 cast 当作 Plan 03 的完成路径。

## 会话任务提示

```text
完成 deepseek-v4-c500-e2e/03-o-proj.md。先用真实 O-proj 输入拆分测量
inverse-RoPE、wo_a、wo_b 和 copy，再选择原生 FP8 或 BF16 fused MetaX 路线。
禁止用软件 FP8 模拟或只在随机输入上证明性能；必须通过 graph 和 TP=4 IDs。
```
