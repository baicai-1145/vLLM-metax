# CUDA Graph、kernel launch 和运行时间隙优化

## 会话目标

在主要计算 kernel capture-safe 后，减少 piecewise graph break、allocator/copy、
host dispatch 和 inter-kernel gap。不要用 graph 掩盖错误 kernel。

## 当前事实

- 历史参考：正常 TP=4 PIECEWISE/MTP0 `11.6247 tok/s`（约 `86 ms/token`），eager `3.7984 tok/s`；graph 当时约带来 3x 收益。
- 最新 TP=4、MTP=0、100-token decode 基线为 `15.2334 tok/s`、`65.65 ms/token`，四卡利用率 `16.2–16.4%`；profiler execute context 约 `80.3 ms/step` 仅作归因参考。
- Torch sparse attention 和 exact raw/pre 虽可被捕获，仍保留大量 GPU launches；
- custom kernel 必须使用稳定 caller-owned input/output/workspace；
- profiler 的 stack、trace 压缩和 stop 开销会把吞吐降到约 `0.9 token/s`，不能
  用作正常性能结论。
- vLLM graph 核心参考：`/root/vllm-0.25rc1/vllm/compilation/cuda_graph.py`；
- profiler 调度参考：`/root/vllm-0.25rc1/vllm/profiler/wrapper.py`。

## 基线与验证边界

- MTP1 的 `17.8358 tok/s` 仅作观察；出现 token divergence 时不得验收。
- 验证链固定为 kernel-validator（差分/graph gate）→ test-runner → profiler（按需归因）→ benchmark-runner（同 workload 的 E2E）。isolated kernel/graph speedup 不等于 E2E。
- 10K prefill 的 Torch sparse MLA OOM 是独立阻断项，不能被 CUDA Graph、专题优化或 decode 提升掩盖。

## 前置条件

至少完成 Sparse MLA decode 和主要 MHC raw/pre kernel，或者明确列出尚未完成的
graph break。否则本会话只能建立测量工具，不能推广更大的 graph。

## 测量任务

- 给 attention、MHC、norm、MoE、collective、sampler 添加稳定 NVTX ranges；
- 统计每 token kernel 数、graph node 数、graph segment 数和 break 原因；
- 量化 CPU submit gap、GPU idle gap、allocator、D2D/H2D copy；
- 比较 eager、PIECEWISE 和候选更大 graph 的正常 wall time；
- 记录每个 custom op capture 时是否发生 JIT、autotune、Python allocation 或 sync。

## 优化路线

1. 把所有 JIT/autotune 移到 warmup；固定 decode shape 的 kernel 使用缓存实例。
2. 预分配并复用 output、workspace、metadata 和 collective buffer。
3. 消除 graph 内 `.cpu()/.item()/synchronize()` 和动态 shape 分支。
4. 合并碎片化 copy/cast/elementwise epilogue，减少 graph node，而不是只减少 CPU API。
5. 在 stable-pointer gate 后扩大 graph segment；collective 必须证明 capture-safe。
6. 检查 replay 后输入更新和输出刷新，防止“graph 成功但使用旧 tensor”。

## 验收指标

- 两组不同真实 payload graph replay 后输出更新且 bitwise 正确；
- 所有输入、输出、workspace 指针稳定；
- 无 capture-time JIT/autotune/host sync；
- kernel/graph node 数显著下降，GPU idle gap 有 trace 证据下降；
- 100-token p50/p90 优于原 PIECEWISE，方差小于 3%；
- 不降低并行度、不关闭必要 collective、不改变模型输出。

## 会话任务提示

```text
完成 deepseek-v4-c500-e2e/06-cudagraph-launch.md。以已通过 differential 的
计算 kernel 为前提，建立 NVTX/graph-break/launch-gap 证据，固定所有 buffer 和
workspace，再扩大 graph。重点验证 replay 输入更新和输出刷新，禁止用 profiler
吞吐或关闭 TP 来宣称提升。
```
