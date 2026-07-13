# DeepSeek V4 C500 端到端优化会话索引

本目录用于把 4x C500 相对 4x A100 的端到端性能优化拆成多个独立 Codex
会话。每个会话先完整读取 `00-shared-contract.md`，再读取自己的任务文档；不要
重新做已经记录的环境、TileLang、DeepGEMM、MCCL、Marlin 或 MHC post 调研。

## 推荐执行顺序

| 顺序 | 会话文档                       | 状态/目的                                | 依赖              |
| ---: | ------------------------------ | ---------------------------------------- | ----------------- |
|    0 | `00-shared-contract.md`        | 冻结 A100/C500 同口径基线和全局门禁      | 无                |
|    1 | `01-sparse-mla.md`             | 完成：累计约 +5%，含输出序列限制         | 00                |
|    2 | `02-mhc-rmsnorm.md`            | 完成/接受 opt-in exact path；默认关闭    | 00                |
|    3 | `03-o-proj.md`                 | 完成：精确融合 no-go，保持基线           | 00，建议 01 后    |
|  3.5 | `03.5-mccl-rank-skew.md`       | 未开始：归因 rank arrival，路由 04/05/06 | 00，01-03 后      |
|    4 | `04-w4a16-moe.md`              | 未开始                                   | 00，03.5 路由     |
|    5 | `05-tp-communication.md`       | 优化 87 次/token 小消息 TP collective    | 00，03.5 路由     |
|    6 | `06-cudagraph-launch.md`       | 减少 graph break、launch gap 和分配      | 00，03.5 路由     |
|    7 | `07-integration-acceptance.md` | 汇总收益、回归和最终推广决策             | 01-06             |

Sparse MLA、MHC、O-proj、MoE 的静态调研可以在文件写集合不重叠时并行推进；
Plan 04、05、06 的性能实现和推广必须等待 Plan 03.5 路由。执行上必须
decode-first，但 prefill 的 OOM blocker 要立即建立复现和修复门禁，不能等 decode
完成后再处理。TP 通信和 CUDA Graph 会影响所有阶段，应该在主要计算 kernel
稳定后再做最终推广。

## 当前事实

- 模型：`/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-MTP`
- **历史** TP=4 PIECEWISE、MTP=0、16-token 快速 gate：`11.624672 token/s`，约
  `86.0 ms/token`（仅作会话起始参考）
- 最新受控 MTP=0、TP=4 PIECEWISE、100-token decode 前后夹具为
  `16.2325/16.0112 tok/s`，漂移 `-1.363%`；累计消融见
  [`2026-07-13-plan01-03-cumulative-ablation.md`](2026-07-13-plan01-03-cumulative-ablation.md)
- 最新 1K prefill：`1321.61 input tok/s`、`0.774812 s`
- 最新 10K prefill：default `chunk=8192` 在
  `torch_flash_mla_sparse_prefill` 的 `torch.index_select` 尝试约 `10 GiB` 分配并
  OOM；`chunk=2048`、`GPU_MEM=0.8` 为 `1504.16 input tok/s`、median `6.648215 s`，
  四卡利用率约 `89.1%--89.2%`
- 已安装 `deep_gemm` 的 `int8_mqa_logits` 不接受 `backend` kwarg；动态 multi-head
  Triton loop 触发 `mcTriton SmallVector`。当前仓库的 3D head-grid atomic workaround
  已通过 differential、graph、16-token gate，但重复浮点差约 `1.5e-3`，暂不推广。
- MTP=1 测得 `17.8358 tok/s`，但从第二个 token 起偏离 greedy oracle；因此默认关闭，
  并以 hard fail 阻止验收。
- exact MHC post 已完成：340/340 bitwise、graph replay、稳定指针和 16-token gate
- exact post microbenchmark：`0.0222 ms` eager、`0.0350 ms` graph
- Plan02 2026-07-14 opt-in exact path 已接受：`mhc_sigmoid` 的
  `__builtin_mxc_rcpf(1+expf(-x))` 已改为 `__fdiv_rn`；1892 个 rank0 真实 late
  payload 每个已检查 stage bitwise、max abs/rel `0`，graph `9/9` 稳定且无分配。
  TP=4 PIECEWISE 23-token gate 及 fresh 100-token exact-off-vs-on IDs 全部一致，
  无 fallback。正常 5-run 同 workload 为 off `15.965887` median TPS / `6.263354 s`
  median latency，on `25.875913` / `3.864598 s`，即 `+62.07%` TPS、`-38.30%`
  latency；默认仍关闭。详见 `.logs/plan02_sigmoid_fdiv_stage_diff_20260714/`、
  `.logs/plan02_sigmoid_fdiv_graph_gate_20260714/`、`.logs/plan02_sigmoid_fdiv_e2e23_20260714/`、
  `.logs/plan02_sigmoid_fdiv_e2e100_20260714/` 和 `.logs/plan02_sigmoid_fdiv_benchmark_20260714/`。
- 当前最高风险路径是 MetaX sparse MLA decode/prefill 的显式 Torch reference
- 最新同口径 profiler：
  `.logs/dsv4_mhc_exact_post_mma_steady_steps_profile_20260712.log`
- 总体分析：
  `../2026-07-12-deepseek-v4-c500-end-to-end-performance.md`

## 新会话统一启动方式

把下面文本与对应任务文档末尾的“会话任务提示”一起作为新会话首条消息：

```text
在 /root/vLLM-metax 继续优化 DeepSeek-V4-Flash-W4A16-BF16Attn-MTP 的
TP=4、MTP=0 推理性能。先完整读取
docs/superpowers/plans/deepseek-v4-c500-e2e/00-shared-contract.md 和指定的
子任务文档。不要重新做文档中标为已完成的环境或 kernel 调研。使用当前工作树
为事实来源，不得修改 /root/vllm，不得降低 TP 验收，不得回退用户已有改动。
先建立真实输入 differential/benchmark feedback loop，再实现、验证和 profiler。
```

## 每个会话必须交付

- 修改文件和行为边界；
- 真实输入 corpus/harness，而不是随机 tensor 自测；
- direct differential、graph replay、稳定指针和 TP=4 16-token 结果；
- 正常推理和 profiler 分离的性能数据；
- kernel dispatch 证据、调用次数、单次和每 token 延迟；
- 与会话开始基线的差值，而不是只给绝对值；
- 未通过时保持 opt-in/default-off，明确 blocker，不得静默回退。

性能和正确性证据由专门角色产出：`benchmark-runner` 只负责保持不变 workload
的 normal throughput/利用率，`profiler` 负责 steady-state trace 与阶段归因，
`kernel-validator` 负责 differential、graph replay 和 kernel 微基准。三者均须
保留原始 artifact；会话摘要只能引用 artifact 路径、命令、commit、shape/dtype、
warmup/repetition、median/p90、利用率、token IDs、差异位置和 dispatch 证据，不能
用口头结论替代原始日志。
