# 主模型 Decode 优化 — 状态交接 (2026-08-01)

## 背景

DSpark 接受率问题的根因已确认（W4A16 target 量化分布 vs draft 训练时 MXFP4 分布不匹配），
等待官方/社区发布 W4A16 DSpark 量化权重即可解决。优化重点转回主模型本身。

## 当前 baseline (MTP=0, TP=4, PIECEWISE)

| 指标 | 值 | 来源 |
|------|-----|------|
| Decode 吞吐 | **27.2 TPS** (36.8 ms/token) | `.logs/deepseek_v4_mtp0_baseline_100_20260725/fresh1/summary.json` |
| 四卡 SM utilization | **30.3%** | 同上, `gpu_utilization_percent` |
| Prefill 1K | 1322 tok/s | shared-contract |
| Prefill 10K (chunk=2048) | 1504 tok/s | 同上 (default chunk OOM 是 blocker) |
| C500 HBM 带宽 | 5.3 TB/s | `artifacts/hardware/c500_maca_3.7.2/README.md` |

## 核心诊断 (中间结论, 已被下方"最终诊断"修正)

> **重要**: 本节是诊断过程中的中间结论 (occupancy + 带宽双证据)。后续实验证明
> "减少低 occupancy kernel 对 TPS 无影响", 真正瓶颈是 attention kernel 在 graph 外。
> 完整诊断链条见下方 [最终诊断](#最终诊断-2026-08-01-经过五轮实验收敛)。
> 本节保留作为诊断过程记录。

## 核心诊断 (中间): 小 grid kernel 长尾导致 SM 与带宽双重低效 (双证据交叉确认)

**根因结论 (2026-08-01 修正, 替换之前错误的 "kernel 数量爆炸 + 2.7% 带宽" 诊断):**
DeepSeek-V4 MTP=0 decode 的低 SM (30%) 与带宽未打满, 同源于一件事:
**decode 由大量单 block 小 grid kernel 串行组成, 每个 kernel 既填不满 SM 也填不满 HBM 带宽。** 这是两条独立硬证据交叉确认的, 不再依赖任何估算或 profiler 推断。

### 证据一: per-kernel occupancy (profiler `est. achieved occupancy %` 字段)

post-fusion steady-state step (5298 kernels) 的 occupancy 分布:

| occupancy | kernel 数 | device 时间占比 |
|-----------|----------|---------------|
| **0%** (grid=[1,1,1], 单 block) | 3482 | **46.2%** |
| 1-24% | 1579 | 29.9% |
| 25-49% | 149 | 18.5% |
| 50-74% | 43 | 2.3% |
| 75-100% | 45 | 3.1% |

**76% 的 device 时间花在 occupancy ≤ 24% 的 kernel 上, 46% 花在 occupancy=0% 的 kernel 上。** 0% occupancy 的 kernel 全是 grid=[1,1,1]、block≤1024 的小 kernel (elementwise / index / cast / reduce / slot-mapping), C500 有 148 个 SM, 单 block kernel 只占 1 个 SM。真正能吃满 SM 的 (≥50%, 主要是 `b16gemvt` attention GEMV 与 `fused_moe`) 只有 88 个, 占 device 时间 5.4%。

### 证据二: 硬件 HBM 带宽计数器 (`mx-smi --show-hbm-bandwidth`)

在 decode 运行期间用驱动级硬件计数器采样 (非 profiler, 非估算), 4 卡各白:
- **峰值带宽 ~640 GB/s = 5.3 TB/s 峰值的 12%** (不是之前文档声称的 2.7%)
- 带宽分布: **55% 时间 <1 GB/s (空闲)**, **43% 时间 1-100 GB/s**, 仅 1% 时间达到 0.5-1 TB/s, **从未接近 5.3 TB/s**
- 内存子系统确实未被打满, 与 SM 低利用率同源: 小 grid kernel 的访存模式达不到峰值带宽

### 为什么之前 "2.7% 带宽" 的诊断是错的

旧文档写的 `5.4 GB / 36.8ms = 2.7%` 有三层错误: (1) 5.4 GB 这个每步读取量未经实测验证 (估算低估了 attention/shared expert); (2) 带宽利用率的分母用了总 wall time 而非峰值窗口 (掩盖了是否存在满载瞬间); (3) 整个 "2.7% 带宽 → 不是访存瓶颈 → 一定是 kernel 启停" 的推理链悬空。实测峰值是 12% 不是 2.7%, 且带宽未打满的真正原因是小 kernel 访问模式, 不是 "kernel 太多"。

### 为什么之前 elementwise 融合 ROI 极低 (验证了上述根因)

fused sinkhorn (+3.5% TPS) 与 fused mhc_pre_norm (+0.36% TPS) 两个实验都不符合预期地低。原因现在清楚了: **把几个 occupancy=0% 的小 kernel 融合成一个 grid 仍为 [1,1,1] 的稍大 kernel, occupancy 还是 0%, SM 还是没填满, 带宽还是上不去。** 问题不是 kernel 数量, 是 **每个 kernel 的 grid 太小**。减少 kernel 数量不解决问题, 必须让单个 kernel 的 grid 足够大去占满 SM。

### 优化方向 (由根因直接推导)

真正的杠杆是把 decode 从 "~5000 个单 block 小 kernel" 变成 "几百个 grid 足够大的 kernel":
- FlashMLA 式的 attention 大融合 (一个 kernel 吃掉 q/kv/attn/o, grid 随 head 数×seq_len 足够大)
- MoE 的大融合 (把 topk/scatter/gemm/unscatter 融成少量大 grid kernel)
- 减少必须用单 block kernel 的操作 (把多个 norm/cast/quant 合并成 grid 更大的 kernel, 而非仅减少数量)

**仅减少 kernel 数量的小融合 (如已做的 sinkhorn/mhc_pre_norm) ROI 已耗尽, 不应继续。**

注意 (profiler 放大): torch profiler `with_stack=True` 在 1.5 万 kernel/token 下会显著放大 device wall 时间 (trace 内 in-span kernel ~57ms 远大于 baseline 36.8ms)。绝对时间不可直接引用 baseline 对比; 相对结构 (occupancy 分布、kernel 类型占比) 可信, 因为 occupancy 是 per-kernel 固有属性不受放大影响。任何性能接受必须用非 profiler 的 benchmark-runner 正常 TPS 验证。

证据来源:
- occupancy: `.logs/mtp0_piecewise_postfusion_20260801/trace/` (profiler `est. achieved occupancy %` 字段)
- HBM 带宽: `mx-smi --show-hbm-bandwidth` 驱动计数器, 采样脚本 `tools/debug/hbm_bandwidth_sampler.py`
- 融合实验验证: `.logs/deepseek_v4_mtp0_fused_sinkhorn_100_20260801/` (+3.5%), `.logs/deepseek_v4_mtp0_fused_mhc_pre_norm_100_20260801/` (+0.36%)

## DSpark 实验中获得的 target forward kernel 分解 (v123 profiler)

DSpark k=5 batched verify (M=6) 的 target_accept phase, GPU kernel 时间:

| 类别 | 时间/cycle | 占比 |
|------|-----------|------|
| MoE (W4A16 GEMM + helpers) | 5.2ms | 27% |
| Sparse MLA | 1.2ms | 6% |
| Attention/Other GEMM | 1.1ms | 6% |
| TP AllReduce | 0.5ms | 3% |
| Elementwise/Other | 11.1ms | 58% |

注意: 这是 M=6 (DSpark batched), 纯 GPU kernel 时间 3.8ms/cycle。
干净 MTP=0 trace (2026-08-01) 证实 M=1 的 MTP=0 **同样是小 grid kernel 主导**
(76% device 时间在 occupancy ≤24% 的 kernel, 46% 在 occupancy=0% 的单 block kernel),
这与下表 DSpark M=6 的 "Elementwise/Other 58%" 趋势一致。低效不在 graph gap 上,
而在 **每个 kernel 的 grid 太小、占不满 SM 也吃不满带宽** (见上文双证据)。

来源: `.logs/deepseek_v4_dspark_cycle_reconstruction_20260729/native_serial_attn_gemm_rows_profile_5active_v123/phase_kernel_summary.json`

## 最终诊断 (2026-08-01, 经过六轮实验收敛)

**FULL graph 下已接近 device-bound, 真正的杠杆是减少 device kernel time, 但大部分低-occupancy kernel 的 grid 小是 workload (单请求 M=1) 决定的, 非实现差。** 只有 FlashInfer 式整段 attention 融合 (让 grid = num_heads × seq × split_k) 能从根本上提升 SM。

### device-time 精确构成 (post-fusion step, 第六轮分析, 2026-08-01)

post-fusion steady-state step (~30ms) 的 occupancy × category 交叉表 (这是最精确的 device time 拆解):

| category | occupancy | count | ms | 可优化性 |
|----------|-----------|-------|-----|--------|
| aten_elem | **0%** | 2358 | **6.79** | block×grid=256/512/1024 元素, 单 block 合理 (M=1 固有) |
| moe | 25-49% | 82 | 4.40 | 已较优 (fused_moe) |
| other | 1-24% | 371 | 3.92 | 含 sgemvt/gemm routing |
| aten_elem | 1-24% | 749 | 3.00 | 部分 Pattern 1 (split+sigmoid) 可融合 |
| attn_gemv | 1-24% | 224 | 2.50 | M=1 固有小 grid |
| **sgemvt** | **0%** | 83 | **2.10** | grid=12 (12 head), M=1 固有 |
| attn_gemv | >=50% | 83 | 1.48 | 已优化 |
| **sinkhorn** | **0%** | 492 | **1.43** | grid=1 (16元素), 数学上必然小 |
| qnorm_rope | 0% | 121 | 0.56 | 小 |

**关键结论 (推翻 "融合小 kernel 减数量" 方向)**:
1. **FULL graph 下 TPS 30.18 ≈ device-bound 33.3 (1000/30ms) 的 91%** — host gap 仅 ~3ms, FULL 已接近 device-bound。故 FULL 下 device kernel time 是 TPS 的真杠杆 (与 PIECEWISE 相反)。
2. **但 0%-occupancy kernel 的 grid 小是 workload 决定的**: sinkhorn 处理 4×4=16 元素 (grid 必然 1), sgemvt 12 head (grid 必然 12), aten_elem 处理 256-1024 元素 (单 block 256 线程合理)。这些不是 "实现差", 是单请求 M=1 decode 的固有特性。
3. **"NV 单请求 99% SM" 的真正原因 = FlashInfer 整段 attention 融合**: 一个 kernel 吃 q/kv/score/o, grid = num_heads × seq_len × split_k, 单请求也很大。MetaX 把 attention 拆成 sgemvt(12) + softmax + 多个小 kernel。
4. **fused_sinkhorn 的教训**: 它减了 kernel 数 (134→6) 但 grid 仍 [1,1,1], 故 device -47% 但 TPS 只 +3.5%。证明 "减 kernel 数" 在单 block 小张量场景无效。

### 优化方向重排 (基于第六轮 device-time 构成)

1. **★★★★ FlashInfer 式 attention 整段融合** — 唯一能从根本提升 SM 的方向。需 tilelang V4 sparse+SWA 成熟 (当前只有 sm90 示例)。grid=num_heads×seq×split_k 能让单请求也占满 SM。工作量巨大, 但是对标 NV 的正道。
2. **★★ FULL graph 生产化** — 已验证 +8.4%, 待多 batch/长 context 稳定性确认。
3. **★ Pattern 1 融合 (split+sigmoid 3.93ms)** — 预期 FULL 下 +1-3% (小张量, 收益主要来自减 graph node)。低 ROI 但可作为练手。
4. **✗ 不再投入: 减 kernel 数量的小融合** — 三次证伪 (sinkhorn/mhc_pre_norm/exact path)。
 诊断链条 (每一步都有实验验证):

1. occupancy 分析: 76% device 时间在 occupancy ≤24% 的 kernel。但 high-occupancy kernel (≥50%, 主要是 attention GEMV) 只占 device 时间 5.9% (1.55ms)。
2. **MHC exact path 实验**: 把 MHC 的 kernel 砍掉 60% (5298→2116), 但 occupancy 结构不变 (0% bucket 46.7%→47.6%), TPS 不动 (+0.33%)。**证明: 减少低 occupancy kernel 对 TPS 无影响 — 它们在 graph 内被吸收, 是背景噪声。**
3. graph 外 kernel 分析: exact path 的 2116 个 kernel 里, **616 个 (29%) 在 graph 外**, 由 host 一个一个 launch (mcModuleLaunchKernel/mcLaunchKernel), 每个 kernel 之间 device 等 host 20-100μs, 累计 ~25ms device idle。这些 graph 外 kernel **几乎全是 attention**: sparse_mla 的 gather/cast/scale_mask/transpose (258个) + softmax + attention GEMV。
4. 根因定位: `attention_impl` (attention.py:1571) 被 `@eager_break_during_capture` 装饰。该装饰器在 PIECEWISE 模式下**主动把 attention 切到 graph 外 eager 执行** (breakable_cudagraph.py:101-115), 这是 PIECEWISE 的设计 — attention 是 graph break point。**这就是 attention 在 graph 外、贡献 25ms device idle 的直接原因。**

### 优化方向 (按已验证 ROI 排序, 2026-08-01)

### 1. ★★★★ FULL CUDA graph — 已验证最大提升, 应成为 production 默认
- **实验结果 (纯单变量, 2026-08-01)**: CUDAGRAPH_MODE=FULL vs PIECEWISE, 其余字节一致 (默认 torch MHC, 同 corpus):
  - **FULL 30.18 TPS / 35.76% SM  vs  PIECEWISE 27.84 TPS / 31.30% SM → +8.41% TPS, +4.46pp SM**
  - p90: 24.14 → 28.51 (+18.1%)。这是迄今所有实验中最大的单点提升。
- **机理**: FULL 模式下 `eager_break_during_capture` 不 break (breakable_cudagraph.py:102-104, mode==FULL 时 return fn), attention 进 graph, 616 个 graph 外 kernel 被吸收, 25ms device idle 消除。
- **正确性**: 16-token greedy hash 与 PIECEWISE baseline 完全一致 (`fa5810ac...6d696d`)。
- **重要机制澄清 (git-historian 2026-08-01 查证)**:
  - **实际运行的是 FULL_DECODE_ONLY**, 不是纯 FULL: upstream 自动降级 (MacaDeepseekV4FlashMLABackend 声明 `_cudagraph_support=AttentionCGSupport.UNIFORM_BATCH`, 不支持混合 prefill+decode 整图 capture)。decode 走整图 graph (attention 进 graph, +8.4%), prefill 走 eager。这是**更安全的形态** — prefill 本就变长难 capture。
  - **v159/v160 的 "attn_metadata freeze" 与 FULL 模式无关**: 它们是 PIECEWISE 实验在 capture 内手动跳过 eager break 导致的, 那个代码已 revert, 从未进入 FULL 路径。FULL 走 upstream `torch.cuda.graph()` 整体 capture + BatchDescriptor 精确匹配 replay, attn_metadata 的 in-place 更新在精确匹配下安全。**FULL 从来没有 v159/v160 那个 bug。**
  - **优雅降级兜底**: 任何未被 capture 的 shape (多 batch, 长序列, 混合) 自动 fallback 到 eager (CudaGraphManager.dispatch 在无匹配 descriptor 时返回 cg_mode=NONE)。推广 FULL 的风险是"性能退化", 不是"结果错误/corruption"。
  - **为什么 PIECEWISE 是默认**: 纯历史惯性 (commit fb00e89, 2026-07-11 加的 env passthrough, 默认 PIECEWISE)。FULL 从未做过默认, 2026-08-01 这次是 track 上第一次 FULL 实验。
- **待验证 (成为默认前)**: (a) FULL 在多 batch / 长序列 / 混合长度下的稳定性与 fallback 频率; (b) DSpark/MTP 恢复后与 FULL 的交互 (dspark_piecewise_cg.py 对 draft 强制 PIECEWISE, 但 target 用 FULL_DECODE_ONLY 应兼容); (c) 更长 context (当前只测 MAX_MODEL_LEN=512) 下的 graph 内存与 capture 时间。
- **证据**: `.logs/deepseek_v4_mtp0_full_graph_20260801/` (FULL), `.logs/deepseek_v4_mtp0_piecewise_control_20260801/` (控制组)。

### 2. ★ MHC tilelang 终态化 — ROI 已耗尽, 仅代码整洁价值
- exact path 实验证: MHC kernel 减 60% 但 TPS +0.33%。MHC kernel 在 graph 内, 减少无效。
- 可为代码整洁做 (修 sigmoid + 默认启用 exact/tilelang), 但**不应期待 TPS 提升**。
- 手写的 fused sinkhorn (+3.5%) / fused mhc_pre_norm (+0.36%) 同理 — 它们碰巧动了少量关键路径, 但整体 ROI 已证明极低。

### 3. ★ MLA fused kernel (对标 FlashInfer trtllm_batch_decode_sparse_mla_dsv4) — 降级为 FULL 稳定后的次要优化
- FULL graph 已解决最大问题 (attention 进 graph), fused MLA kernel 的核心价值 (让 attention 进 graph) 已被替代。
- 剩余价值: attention 仍是 258 个低 occupancy 子 kernel, fused 后可能进一步优化, 但预期 ROI 与 MHC exact path 类似 (低, 因为已在 graph 内)。
- **不建议现在投入**。等 FULL graph 稳定为默认后, 若仍有优化空间再评估。tilelang-metax 的 dense MLA (example_mla_decode_paged) 可作 spike 参考, 但 V4 sparse+SWA 版仍是未完成的 illustration。

### 4. Async scheduling / CPU-GPU overlap
- trace 显示 `host_ahead` 中位 22ms (host 跑完等 device)。FULL graph 已大幅压缩 host 串行, 此项优先级随 FULL 确立而下降。
- MTP=0 是否能开 async (DSSpark 强制 V2, MTP=0 默认 V1) 待评估。

### 4. ★ MoE kernel 效率
- INT4 GEMV 在 M=1 时的效率
- trace 显示 MoE GEMM (`fused_moe_kernel_gptq_awq`) 只占 8.1% kernel 时间,
  不是首要矛盾 (与原"27%"的估计不同, 该 27% 来自 DSpark M=6 场景, 不适用 M=1)

## 第一步已完成: 干净的 MTP=0 PIECEWISE profile (2026-08-01)

已完成一次干净 MTP=0 PIECEWISE trace 采集, 4 rank 原始 trace + 分析全部保留于
`.logs/mtp0_piecewise_profile_20260801/run1/`。结论见上方"核心诊断"。

四个问题答案:
1. **PIECEWISE graph 覆盖**: ~1 graph/层 (44 launches/step, 43 层), 覆盖率高;
   segment 间 device gap 仅 ~7.4ms/step。
2. **segment 间 gap**: 4 个 gap × ~1.9ms; host launch inter-arrival 中位 1.1μs。
   → 原假设的"graph-launch gap = 70%"不成立。
3. **FULL graph 是否 break**: 尚未验证 (次要项, 见优化方向 2)。
4. **async_scheduling**: host_ahead 中位 22ms 说明 host 不是瓶颈; 是否开启待实验。

采集命令 (已执行, 参考用):
```bash
PROFILE_DIR=/root/vLLM-metax/.logs/mtp0_piecewise_profile_20260801/run1/trace \
PROFILE_DELAY_ITERATIONS=4 \
PROFILE_MAX_ITERATIONS=5 \
PROFILE_ACTIVE_ITERATIONS=5 \
PROFILE_IGNORE_FRONTEND=1 \
NUM_SPECULATIVE_TOKENS=0 \
ENFORCE_EAGER=0 \
CUDAGRAPH_MODE=PIECEWISE \
MAX_TOKENS=100 MAX_MODEL_LEN=512 \
./tools/run_deepseek_v4_mtp_generate.sh
```

## 关键文件索引

### 已接受的 production patch
- `vllm_metax/patch/performance/dspark_piecewise_cg.py` — DSpark draft PIECEWISE CG (v165)
  - draft forward 7.5ms→2.9ms, DSpark 38→41.6 TPS
- `vllm_metax/patch/performance/grouped_topk_router.py` — MoE grouped topk
- `vllm_metax/patch/performance/gpu_model_runner_capture.py` — graph capture 增强
- `vllm_metax/patch/performance/pre_outer_graph_device_sync.py` — o-proj graph sync
- `vllm_metax/patch/performance/speculative_decode_perf.py` — spec decode perf

### DSpark 诊断 patch (可清理)
- `vllm_metax/patch/debug/perpos_accept_patch.py` — per-position accept rate (V2 RejectionSampler.__call__ hook)
- `vllm_metax/patch/debug/v1_cycle_host_patch.py` — V1 cycle host timing
- `vllm_metax/patch/debug/accept_count_patch.py` — accept count (spawn 问题, 不工作)
- `vllm_metax/patch/debug/timing_patch.py` — phase timing patch
- `vllm_metax/patch/debug/msafe_linear_patch.py` / `linear_msafe_probe.py` — MSAFE probe

### Benchmark/工具脚本
- `tools/debug/quality_benchmark_100_v2.py` — 100-prompt benchmark (主用)
- `tools/debug/dspark_cycle_timing.py` — DSpark cycle phase timing
- `tools/debug/arena_hard_accept_rate*.py` — accept rate benchmark
- `tools/debug/corpora/arena_hard_100.jsonl` — Arena-Hard 100 prompt corpus
- `tools/debug/corpora/deepseek_v4_mtp_new_heldout_3_20260724.jsonl` — 冻结 3-prompt corpus

### DSpark model override
- `vllm_metax/models/deepseek_v4/dspark.py` — DSpark model override

## 新官方权重

DeepSeek 官方 2025-07-31 发布 `deepseek-ai/DeepSeek-V4-Flash-0731`:
- 原始权重自带 DSpark
- 预期社区会做 W4A16 + DSpark 量化权重
- 届时 accept rate 问题自然解决 (draft 在正确量化分布上训练)

## 历史结论 (不需要重做的)

- **batched path (tokenwise=0) 已接受**: production path, batched oracle hashes 已冻结
- **max_num_seqs=1 正确**: DSpark 不适合并发, 已测试验证
- **DSpark accept rate 38% 不是 bug**: 根因是量化分布不匹配, 等 W4A16 DSpark 权重
- **Draft forward 已优化到极限**: 2.9ms (5%), 再降收益 <0.5 TPS
- **AllReduce 不是 decode 瓶颈**: 仅 3% (之前误判为 48%); 干净 MTP=0 trace 复测
  MCCL 仅 1.1ms/token (1.9%)
- **INT8 KV Cache 无价值**: 仅 0.4ms benefit
- **[2026-08-01 修正, 第二次修正] MTP=0 主瓶颈是小 grid kernel 长尾**, 不是
  kernel 数量、也不是 graph-launch starvation。双证据: (1) profiler occupancy 字段
  显示 76% device 时间在 occupancy ≤24% 的 kernel (46% 在 occupancy=0% 单 block
  kernel); (2) `mx-smi --show-hbm-bandwidth` 硬件计数器实测峰值带宽仅 12% (非旧文档
  的 2.7%), 43% 采样时间带宽落在 1-100 GB/s。低效源于每个 kernel grid 太小占不满
  148 个 SM, 非 kernel 数量。已验证: 小融合 (sinkhorn +3.5%, mhc_pre_norm +0.36%)
  ROI 极低, 因融合后 grid 仍为 [1,1,1]。原"70% GPU 空闲是 graph gap"与"2.7% 带宽"
  两版归因均作废。
- **[已作废] "W4A16 MoE GEMM 占 27%"**: 该 27% 来自 DSpark M=6 场景, 不适用 M=1。
  MTP=0 M=1 实测 MoE GEMM 仅 8.1% kernel 时间, 首要矛盾是 elementwise 不是 MoE。
