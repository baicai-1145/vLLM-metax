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
|  3.5 | `03.5-mccl-rank-skew.md`       | 完成：aligned replay 三次稳定 `<5 ms`    | 00，01-03 后      |
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
- 2026-07-16 已修复 Sparse MLA scale-mask 越界、C4 local top-k 越界、compatibility
  top-k 读错 cache 和 BF16 compressor 行 stride 四个原生缺陷。TP=4、MTP=0、
  PIECEWISE、Plan02 exact-on 的 250+32 边界门禁为 32/32 非零，四 rank positions
  256--266 hidden/logits finite，无 fallback。证据见
  `.logs/deepseek_v4_flash_quality_eval_20260716/boundary_250_final_stride_postfix/`。
- 官方 non-thinking chat 三题在冻结 `max_tokens=256` 评分下仍为 `1/3`：事实题通过，
  数学和代码题正确推进但被输出上限截断，三题均无 token ID `0`。相同 prompt 的
  512-token 补充运行中，数学题完整得到 `960 liters`，代码题精确给出
  `[2, 1, 3, 2] [2, 3, 6, 8] [12, 16]`，两题均正常 stop，结果为 `2/2`。证据见
  `.logs/deepseek_v4_flash_quality_eval_20260716/real_qa_final_postfix/` 和
  `.logs/deepseek_v4_flash_quality_eval_20260716/real_qa_extended_512_postfix/`。该三题
  本身不替代完整 GSM8K 或动态 batch 一致性，不能据此宣称完整模型质量已验收。混合预算
  `3/3` 语义问答综合结论见
  `.logs/deepseek_v4_flash_quality_eval_20260716/real_qa_completed_postfix/`。
- 正常 native logits 的 seed42 100 题 GSM8K 已完成：batch=1 canonical/人工复核为
  `93%/96%`，batch=2 为 `94%/97%`，两路均无 invalid、length、token ID `0` 或
  runtime failure，达到官方公开 DeepSeek-V4-Flash Base `90.8` 的同一量级。官方
  使用 8-shot FP4/FP8 mixed checkpoint，不是本地 W4A16 exact oracle。证据见
  `.logs/deepseek_v4_flash_quality_eval_20260716/gsm8k_seed42_100_postfix/`。
- 动态 batch 确定性仍为 RED：100 题 token exact `17/100`、答案一致 `92/100`、
  correctness 一致 `93/100`。强制 FP32 logits 会降低 batch=2 语义准确率，已拒绝
  推广。Sparse MLA native/reference 差分只发现 layer0 C1 `7.629e-6` BF16 舍入差，
  probabilities/cache gather exact、无 NaN/Inf，仍不足以解释后续答案分叉。证据见
  `.logs/deepseek_v4_flash_quality_eval_20260716/sparse_batch_differential/`。
- 最新 10K prefill：default `chunk=8192` 在
  `torch_flash_mla_sparse_prefill` 的 `torch.index_select` 尝试约 `10 GiB` 分配并
  OOM；`chunk=2048`、`GPU_MEM=0.8` 为 `1504.16 input tok/s`、median `6.648215 s`，
  四卡利用率约 `89.1%--89.2%`
- 已安装 `deep_gemm` 的 `int8_mqa_logits` 不接受 `backend` kwarg；动态 multi-head
  Triton loop 触发 `mcTriton SmallVector`。当前仓库的 3D head-grid atomic workaround
  已通过 differential、graph、16-token gate，但重复浮点差约 `1.5e-3`，暂不推广。
- MTP=1 测得 `17.8358 tok/s`，但从第二个 token 起偏离 greedy oracle；因此默认关闭，
  并以 hard fail 阻止验收。
- Plan 03.5 三次 TP=4 profile 均为 4 ranks、5 active steps、87
  all-reduces/step、435 calls/rank。arrival-wait 谓词三次通过，但严格的
  correlation/runtime/host 链无法归因 top-decile 前序；completion-tail union
  仅占 active device window `6.413%/7.950%/7.024%`。Plan 04/05/06 均不满足，
  结果为 inconclusive，证据在 `.logs/plan35_mccl_rank_skew_20260714/`。
- Plan 03.5 后续硬目标为三次 unchanged TP=4 profile 的四个 rank 均低于
  `5 ms / 435 all-reduces / 5 active steps`。当前每次最差 rank 仍约为
  `51.768/71.006/48.244 ms`，正在建立快速红色回放门禁和 correlation-linked
  stage instrumentation；在根因证据完成前不启动 Plan 04/05/06。
- 首次 env-gated NVTX probe 保持了 TP=4 exact tokens 和 5x87 trace 完整性，
  但 active graph replay 中 `plan35.*` annotation 为 0，四 rank residency 仍为
  `42.553/39.602/42.173/3.540 ms`。该 probe 已判无效，下一步改用 profiler
  `record_function` 语义范围；artifact 在
  `.logs/plan35_mccl_rootcause_20260714/instrumented_run1/`。
- 第二次 `record_function` probe 同样保持 correctness 和完整 5x87 trace，但
  active replay 中 annotation 仍为 0，说明模型调用点只在 graph capture 执行；
  四 rank residency 为 `7.394/61.303/67.104/63.757 ms`，最大单次 arrival skew
  `4792.446 us`。下一 probe 转到 breakable graph replay 边界，artifact 在
  `.logs/plan35_mccl_rootcause_20260714/record_run1/`。
- 87 calls/token 已由源码固定映射为 embedding 1 次，加 43 层各一次
  attention o-proj 和一次 FFN down-proj。四组 trace 的 attention span-skew
  median/P90 为 `239.254/348.008 us`，FFN 仅 `2.407/3.917 us`；late-rank
  `b16gemvt` 时长无异常。708 个 `>50 us` attention 事件中 697 个由迟到 rank
  更晚调用 graph replay 引入，`mcGraphLaunch` API skew 仅 `2.173/7.962 us`。
  当前根因已收敛到 `mcGraphLaunch` 之前的 host replay invocation scheduling。
  当前系统没有 `numactl`，下一单变量实验使用 opt-in `sched_setaffinity`
  固定四个 TP worker 的 CPU sets，不改变 memory policy。
- 首次 direct-affinity run 的模型/trace gates 通过，但四条 worker affinity
  生效日志为 0，证明 general plugin 安装时机过晚；该 run 不作为候选证据。
  artifact 保存在 `.logs/plan35_mccl_rootcause_20260714/affinity_run1/`，正在把
  opt-in patch 安装提前到官方 runner 构造 `LLM` 之前再复测相同 CPU sets。
- client runner 提前安装后的第二次 affinity run 仍为 0 条 worker 生效日志，
  说明独立 EngineCore 不继承该 monkeypatch；artifact 在
  `.logs/plan35_mccl_rootcause_20260714/affinity_run2/`。实现正在改为 env-selected
  本地 GPU worker subclass，并在 worker 自身 `init_device()` 设置 affinity。
- subclass affinity run 已验证四个 rank 分别绑定 24 个独立 CPU cores，但
  residency 仍为 `50.939/43.428/41.819/3.709 ms`；attention span-skew
  median/P90 降至 `218.097/249.113 us`，仅属轻微抖动改善。vLLM 已将 worker
  Torch threads 从 96 降为 1，下一次用每 rank 单独一个物理核排除 CPU migration。
  artifact 在 `.logs/plan35_mccl_rootcause_20260714/affinity_run3/`。
- 单核结果为 `3.069/66.872/66.922/66.838 ms`，短 residency rank 变为 rank0，
  attention span-skew 反而升至 `239.492/319.221 us` 且出现 9 个 `>500 us`
  outliers。固定 GPU/link 和 CPU migration 均不是完整根因；当前转向分析 43 个
  breakable eager attention segments。artifact 在
  `.logs/plan35_mccl_rootcause_20260714/affinity_single_run1/`。
- opt-in 移除顶层 `attention_impl` eager break 虽能完成 graph capture/replay，
  但 100-token oracle 从第二 token 起立即分歧，属于 correctness failure；artifact
  在 `.logs/plan35_mccl_rootcause_20260714/capture_attention_run1/`。该方向不推广，
  下一候选保留 eager boundary，只减少 native sparse-MLA compatibility decode
  内部的 runtime launches。
- compatibility decode 全链已在 3 个真实 production corpus case 上通过 plain
  CUDAGraph feasibility gate：每个 case 12 次变异 replay 均与 eager-native
  bitwise exact，输出/workspace 指针稳定、无 NaN/Inf、无 Torch fallback；隔离
  median 从 `0.142/0.159/0.153 ms` 降至 `0.052/0.067/0.057 ms`。这只批准下一步
  opt-in 内层图缓存实验，尚未通过 TP=4 exact-token 或四 rank `<5 ms` 门禁。
  eager-native 对 frozen corpus 的既有漂移仍单独记录，artifact 在
  `.logs/plan35_sparse_compat_subgraph_validation_20260714/`。
- opt-in 产品实现 `VLLM_METAX_SPARSE_MLA_COMPAT_CUDAGRAPH=1` 已改用 exact
  pointer/layout key 和 dedicated graph pool。default-stream 实机门禁在同一
  进程创建 3 个 production keys，36 次变异 replay 全部 bitwise exact，cache
  `0 -> 3`，无 fallback/NaN/Inf/指针漂移；clear 后新 pool recapture 也通过。
  steady replay median 为 `0.054/0.067/0.059 ms`，对应 eager
  `0.141/0.161/0.157 ms`；首次 capture 每 key 约 `104--106 ms`。这仍只是 kernel
  级候选，TP=4 exact-token 和四 rank `<5 ms` 尚未运行。最终 artifact 在
  `.logs/plan35_sparse_compat_product_validation_20260714/summary_dedicated_consolidated.json`。
- 候选 normal TP=4 100-token gate 已 exact pass，5 次 latency 为
  `4.036655/3.254709/3.263282/3.246087/3.240143 s`，median normal TPS
  `30.724713`，与冻结 `30.640709` 基本持平；P90 TPS 仅 `24.772987`，四卡
  cropped utilization 为 `27.85%/28.25%/28.36%/28.13%`，首个 measured request
  的 capture outlier 仍需保留。第一条 unchanged profile 也通过四 rank、5x87、435/rank
  完整性，但 residency 仍为 `8.353/28.993/20.508/30.311 ms`。attention
  span-skew median 已从 `239.254 us` 降到 `57.873 us`，P90 却仍为
  `349.632 us`；剩余尾部集中在 attention o-proj 后的 outer graph replay
  transition。artifacts 在 `.logs/plan35_sparse_compat_e2e_20260714/normal_gate1/`
  和 `.logs/plan35_sparse_compat_profile_20260714/run1/`。
- residual transition 分解进一步显示 inner launch/device/host-gap median 仅
  `6.319/36.447/14.782 us`，且与 arrival skew 几乎不相关；outer API 也只有
  `7.046 us` median。真正的长区间是已排队 outer graph 到 collective 的
  `8.533 ms` median / `27.023 ms` P90，负相关 `-0.742` 表明早 rank 排队过深、
  迟 rank device queue starvation。下一 probe 仅启用现有 diagnostic sparse
  decode stream sync，验证 launch-ahead throttle，不改变 arithmetic/MCCL。
- stream-sync probe exact/trace gates 通过，但 residency 恶化为
  `42.232/44.169/6.012/37.065 ms`，aggregate 比 inner-graph run1 高 `46.8%`；
  attention span P90 升至 `377.917 us`，diagnostic latency 由 `3.264204` 增至
  `3.585157 s`。因此 local queue throttle 只轮换迟 rank，不会对齐 TP host
  progress；该 probe 拒绝，artifact 在
  `.logs/plan35_sparse_compat_profile_20260714/sync_probe1/`。
- narrow eager-boundary 候选仅把 metadata/KV/indexer/compressor 保持 eager，
  将 `forward_mqa` 与 o-proj 放回 outer graph，并通过 static `q_out` 避免 eager
  返回值冻结；但 TP=4 100-token gate 仍复现同型分歧
  `[260,37,25,70,5811,...]`。这证明 `forward_mqa` 也依赖不能直接 replay 的动态
  metadata/stream state。候选代码已移除，诊断性能无效，artifact 在
  `.logs/plan35_narrow_attention_e2e_20260714/normal_gate1/`。
- GC-disable probe 已通过 4 workers 生效、exact/trace gates，但 residency 变为
  `8.260/33.823/31.218/37.623 ms`，aggregate 比 inner-graph run1 高 `25.8%`，
  attention span P90 也升到 `370.702 us`。Python cyclic GC 不是主因；artifact
  在 `.logs/plan35_sparse_compat_profile_20260714/gc_probe1/`。
- GIL switch interval `0.005 -> 0.1 s` probe 覆盖 EngineCore+4 workers 并通过
  exact/trace gates；aggregate residency 小幅下降 `5.7%` 到 `83.107 ms`，
  completion/attention P90 分别降到 `129.562/332.775 us`，但四 rank 仍为
  `25.822/14.193/27.627/15.466 ms`。host-thread scheduling 有贡献但不是完整根因；
  artifact 在 `.logs/plan35_sparse_compat_profile_20260714/gil_probe1/`。
- GIL interval `1.0 s` follow-up residency 为
  `22.245/28.042/26.290/12.053 ms`，aggregate `88.630 ms` 比 run1 略差；即使
  completion P90 降到 `99.907 us`，attention P90 仍为 `351.120 us`。GIL 调参
  已饱和，artifact 在 `.logs/plan35_sparse_compat_profile_20260714/gil_probe2/`。
- `0.1 s` GIL + 已验证 24-core/rank affinity 组合也失败：residency
  `37.200/8.880/35.039/40.190 ms`，attention span 升至
  `67.876/414.950 us`，aggregate `121.309 ms`。Python scheduler/affinity 路线
  结束，artifact 在 `.logs/plan35_sparse_compat_profile_20260714/gil_affinity_probe1/`。
- 下一条更窄的 layer-3 Q/KV-insert 子图已通过 production-shape 可行性门禁：
  default stream 上仅捕获 BF16 `wq_b`、native fused Q norm/RoPE + 动态 SWA KV
  insert 和 Q padding。15/15 次交替 replay 覆盖 `qr/kv/positions/slot_mapping`
  内容变化、position-only、slot-only 与 invalid-slot，Q 和目标 cache row 全部
  bitwise exact，旧/非目标 cache row 不变，指针稳定且无 fallback/NaN/Inf。
  eager median/P90 为 `0.048128/0.057446 ms`，graph 为
  `0.051968/0.067123 ms`，因此它只验证减少 launch jitter 的可行性，不声称 kernel
  latency 加速。下一步按 TDD 做 opt-in、仅 layer 3 tracer；metadata/indexer/
  compressor/aux-stream/`forward_mqa`/o-proj/MCCL 均保持 eager。artifact 在
  `.logs/plan35_qkv_insert_subgraph_validation_20260714/`。
- layer-3 tracer 已实现为
  `VLLM_METAX_DSV4_Q_INSERT_CUDAGRAPH_LAYER=3`，保留 upstream eager-break，使用
  exact pointer/layout/weight/scalar/stream key 和 per-layer pool。TDD 先捕获了遗漏
  eager-break decorator 与 key 未绑定 `wq_b.weight` 两个问题；修正后 combined
  focused tests 为 23 passed，Ruff、`py_compile`、`git diff --check` 均通过。
- 首次产品 GPU gate 又发现 MetaX capture 记录本身不填充 graph-owned Q，导致首返
  未初始化（`max_abs 3.0245e35`），第二次 replay 才正确；现已增加首调 replay
  回归测试并修复。最终门禁首返及 15/15 动态 replay 全部 bitwise exact，旧/非目标
  cache row 不变、padding zero、指针稳定、weight-key recapture、clear fresh pool、
  invalid slot、nested fail-closed、frozen SHA 与 no-fallback 均通过。eager median/P90
  `0.047744/0.053427 ms`，graph `0.053760/0.089651 ms`，仍只作为 rank-jitter
  probe。RED 与最终 artifacts 均在
  `.logs/plan35_q_insert_product_validation_20260714/`；下一步是 unchanged TP=4
  100-token normal gate。
- layer-3 normal TP=4 gate exact pass，median/P90 为
  `30.505345/29.345272 TPS`（`3.278114/3.407704 s`），四卡 cropped utilization
  `29.05--29.25%`，无 fallback；artifact 在
  `.logs/plan35_q_insert_layer3_e2e_20260714/normal_gate1/`。
- layer-3 unchanged profile 的四 rank/5x87/435 matching 通过，但 residency 仍为
  `24.531/19.130/12.735/31.294 ms`。局部 ordinal 7 span median/P90 从
  `141.724/748.422 us` 降到 `3.552/9.439 us`，局部 residency `-94.47%`；整体
  aggregate 却只改善 `0.54%` 到 `87.691 ms`，attention P90 仍
  `354.657 us`。这证明边界有效但单层不足；下一步把完全相同的 graph 扩到所有
  eligible C128A 层，仍不捕获 C4A/indexer/metadata/compressor/`forward_mqa`/
  o-proj/MCCL。artifact 在 `.logs/plan35_q_insert_layer3_profile_20260714/run1/`。
- all-C128A 扩展的 focused suite 为 24 passed，normal exact gate median/P90
  `30.512913/29.866259 TPS`（`3.277301/3.348260 s`），四卡利用率
  `29.02--29.41%`，无 fallback；artifact 在
  `.logs/plan35_q_insert_all_c128_e2e_20260714/normal_gate1/`。
- all-C128A profile 四 rank residency 改善到
  `16.302/18.889/4.217/18.847 ms`，aggregate `58.256 ms`，相对 layer3
  `-33.57%`；20 个 C128A ordinal 全部改善且合计仅 `3.328 ms`（`-90.59%`）。
  仅 rank2 达标，剩余 top outliers 全为 C4A layer 2/24/38/28/16。下一步按 TDD
  把同一 Q/KV-insert graph 接入 C4A 3-way overlap，但 indexer/compressor/
  aux-events/top-k/`forward_mqa`/o-proj/MCCL 仍 eager。artifact 在
  `.logs/plan35_q_insert_all_c128_profile_20260714/run1/`。
- C4A 3-way overlap 接入的 RED→GREEN suite 为 25 passed；indexer/compressor
  仍各在原 eager aux 路径执行一次。all-attention normal exact gate median/P90
  提升到 `31.487544/30.683572 TPS`（`3.175859/3.259073 s`），四卡利用率
  `30.15--30.57%`，无 fallback；artifact 在
  `.logs/plan35_q_insert_all_attention_e2e_20260714/normal_gate1/`。
- 第一条 all-attention unchanged profile 已严格 GREEN：四 rank residency
  `3.718/4.117/4.288/3.854 ms`，全部低于 `5 ms / 435 calls / 5 steps`；
  aggregate `15.976 ms`，相对 all-C128A `-72.58%`。arrival median/P90
  `89.045/390.042 us`，completion `7.168/9.979 us`，attention span
  `3.635/7.517 us`。exact、4 ranks、5x87、435/rank、PIECEWISE、Plan02、native/
  no-fallback 全通过。这是所需 3 条 unchanged GREEN captures 的第 1 条；artifact
  在 `.logs/plan35_q_insert_all_attention_profile_20260714/run1/`。
- 首次三条稳定性尝试未被挑样：run3 rank1 为 `5.028 ms`，严格 RED；80.1% 超额来自
  embedding ordinal0，加入 ratio0 ordinal3 后覆盖 90.7%。失败 trace 与 gate 原样保留
  在 `.logs/plan35_q_insert_all_attention_profile_20260714/`。
- 最终将同一 graph 边界扩到 3 个无 indexer/compressor 的 ratio0 attention 层；focused
  suite 为 26 passed。all43 normal exact gate median/P90
  `31.668419/30.668487 TPS`（`3.157720/3.260676 s`），相对冻结 normal median TPS
  `+3.35%`，四卡利用率 `30.19--30.36%`，无 fallback；artifact 在
  `.logs/plan35_q_insert_all43_e2e_20260714/normal_gate1/`。
- 最终三条完全相同 profile 均 GREEN、无重跑：run1
  `3.539/4.095/3.906/4.072 ms`，run2 `3.539/4.432/4.141/3.911 ms`，run3
  `3.719/3.954/4.483/4.735 ms`；全局最大 `4.735 ms`，每次均 exact、4 ranks、
  5x87、435/rank、PIECEWISE、Plan02、native/no-fallback，并有 215 Q-insert graph
  calls/rank。raw traces、manifest、JSON/CSV 与 final decision 在
  `.logs/plan35_q_insert_all43_profile_20260714/`。
- 最终代码审计补齐 arbitrary layer selector、sparse graph stream key 和跨流 clear
  安全；trace analyzer 也改为 correlation→`mcGraphLaunch`→host execute context
  分步，重叠 parent 无法唯一消歧时 fail closed。缺 6 个 physical records 的 429-call
  trace 仍严格 RED，未补齐，artifact 在
  `.logs/plan35_q_insert_all43_final_profile_20260714/run2/`。
- all43 单独路径后续仍出现有效 RED `3.281/6.142/5.113/5.103 ms`；单独 current-stream
  drain 与单独 TP Gloo barrier 也分别在第二条 unchanged profile 失败，证明它们只会
  移动 late rank。最终 opt-in
  `VLLM_METAX_DSV4_PRE_OUTER_GRAPH_ALIGNED_REPLAY=1` 固定执行
  `current-stream drain -> TP Gloo CPU barrier -> outer replay`，且不新增 device MCCL。
- aligned 最终 current-code normal exact gate 为 `31.506830 TPS / 3.173915 s`
  median，比冻结 normal TPS 高 `2.83%`；最终三条 unchanged profile 分别为
  `3.337/3.579/3.473/3.885`、`3.364/3.554/3.573/3.502`、
  `3.433/3.578/3.737/3.532 ms`，全局最大 `3.884850 ms`。三次均 exact、4 ranks、
  5x87、435/rank、215 Q calls/rank、5 aligned markers/rank、Plan02、PIECEWISE、
  native/no-fallback，artifact 在
  `.logs/plan35_pre_outer_aligned_final_{e2e,profile}_20260715/`。
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
