# TP=4 小消息通信优化

## 会话目标

降低 decode 中约 87 次/token BF16 TP collective 的固定延迟，并保证 CUDA graph
capture/replay 安全。

## 当前事实

- C500 平台：`vllm_metax/platform.py` 禁用 custom all-reduce；
- runtime 使用 PYNCCL/MCCL；
- 最新五步 profile：435 次 BF16 all-reduce，`14.515 ms`，约 `2.90 ms/token`；
- 当前平均约 `33.37 us`/collective；
- CUDA upstream 有 graph-aware custom all-reduce dispatch chain；
- MetaX MoE all2all/AgRs：
  `vllm_metax/distributed/device_communicators/all2all.py`；
- 已有 microbenchmark：`tools/debug/bench_tp4_allreduce.py`。

## 基线与验证边界

- 当前 TP=4、MTP=0、100-token decode 基线为 `15.2334 tok/s`、`65.65 ms/token`，四卡利用率 `16.2–16.4%`；旧 `11.6247 tok/s`/`86 ms` 仅作历史参考。
- MTP1 的 `17.8358 tok/s` 仅作观察；出现 token divergence 时不得验收。
- 验证链固定为 kernel-validator（差分/graph gate）→ test-runner → profiler（按需归因）→ benchmark-runner（同 workload 的 E2E）。isolated collective speedup 不等于 E2E。
- 10K prefill 的 Torch sparse MLA OOM 是独立阻断项，不能被通信或 decode 提升掩盖。

## 第一任务：运行时 shape 清单

在真实 TP=4 decode 中记录每个 collective：

- caller/layer/stage；
- tensor shape、dtype、bytes；
- all-reduce、reduce-scatter、all-gather 类型；
- stream、前后依赖和是否在 graph 内；
- eager/capture/replay latency；
- 设备拓扑、peer access 和 MCCL algorithm/protocol。

扩展 microbenchmark 覆盖全部真实 size，而不是只测一个 4096 BF16 tensor。对每个
size 报告 p50/p90、warm/cold、TP=2/4、并发 stream 和 graph replay。

## 优化路线

1. 验证 MCCL LL 协议和 topology 是否正确选择；先排除配置问题。
2. 实现或接入 C500 graph-safe 小消息 custom all-reduce，使用预注册稳定 buffer。
3. 对小 tensor 优先优化 latency，不以大消息 GB/s 为主要指标。
4. 合并同一层可同时发起的 reduction，或融合 row-parallel reduction 与后续
   residual/RMSNorm/MHC consumer。
5. 检查 AgRs token<2048 分支的 reduce-scatterv + TP all-reduce，避免重复同步。
6. 用 TP=1 与 TP=4 同阶段差值验证通信上限，禁止把计算变化归因于 MCCL。

## 验收指标

- 所有真实 size 的 graph replay 正确且 pointer 稳定；
- collective 总时间低于 `1 ms/token`；
- collective/synchronization 次数至少降低 2x，或证明模型并行契约无法合并；
- 不引入 host sync、跨 rank 死锁或 graph capture failure；
- TP=4 token IDs 和 100-token 输出完全一致；
- 对照 TP=1/4 证明端到端收益来自通信。

## 会话任务提示

```text
完成 deepseek-v4-c500-e2e/05-tp-communication.md。先从真实 decode 记录全部
87 次/token collective 的 shape、caller 和 graph 状态，扩展现有 all-reduce
microbenchmark，再实现 graph-safe C500 小消息路径或融合边界。不得降低 TP 作为
最终结果，不得用大消息带宽代替小消息 latency。
```
