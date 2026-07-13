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

## 会话任务提示

```text
完成 deepseek-v4-c500-e2e/03-o-proj.md。先用真实 O-proj 输入拆分测量
inverse-RoPE、wo_a、wo_b 和 copy，再选择原生 FP8 或 BF16 fused MetaX 路线。
禁止用软件 FP8 模拟或只在随机输入上证明性能；必须通过 graph 和 TP=4 IDs。
```
