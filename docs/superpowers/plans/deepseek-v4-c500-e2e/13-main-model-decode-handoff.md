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

## 核心诊断: GPU starvation，不是 kernel 慢

每 token decode 需读取 ~5.4 GB 权重 (INT4 MoE + BF16 shared expert):
- 理论最短时间: 5.4 GB / 5.3 TB/s = **1.0 ms**
- 实际时间: **36.8 ms**
- **HBM 带宽利用率: ~2.7%**
- GPU 实际工作: ~11ms (30%)
- **GPU 空闲: ~25ms (70%)**

结论: 瓶颈是 graph launch overhead 和 CPU-GPU gap，GPU 大部分时间在等待。

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
MTP=0 的 M=1 应该更少。大量时间花在 kernel 间的 gap 上。

来源: `.logs/deepseek_v4_dspark_cycle_reconstruction_20260729/native_serial_attn_gemm_rows_profile_5active_v123/phase_kernel_summary.json`

## 优化方向 (按预期 ROI 排序)

### 1. ★★★ Graph 覆盖率 (最大机会, 理论 2-3x)
- 当前 PIECEWISE 只覆盖部分 segment
- 70% 时间是 GPU 空闲 → 扩大 graph 覆盖能直接消除 gap
- 需要确认: 当前哪些 segment 被 graph 捕获？哪些没有？
- 历史问题: v159/v160 attention graph capture 失败 (attn_metadata freeze)
- FULL graph 模式是否可行？

### 2. ★★★ Kernel 融合
- 当前 ~86 MoE kernel launches/token (43 layers × 2 stages)
- 加上 attention 的 rope/norm/softmax 等 elementwise kernel
- 融合能减少 launch overhead 和中间 buffer 读写
- 融合方向: MoE routing+align, attention norm+rope+softmax

### 3. ★★ Async scheduling / CPU-GPU overlap
- V2 runner 的 async_scheduling 已在 DSpark 验证可用
- MTP=0 是否也能开启？(DSSpark 强制 V2, MTP=0 默认 V1)
- 能 overlap host input prepare 和 GPU execute

### 4. ★ MoE kernel 效率
- INT4 GEMV 在 M=1 时的效率
- 当前 MoE 占 27% GPU time, 不是首要矛盾

## 需要的第一步: 干净的 MTP=0 profile

现有 profile 全是 DSpark 的。需要一次 MTP=0 的 profiler 采集来回答:
1. PIECEWISE graph 覆盖了多少层/segment？
2. segment 间 gap 有多少 μs？总共多少 ms？
3. FULL graph 模式是否会 break？根因是什么？
4. async_scheduling 对 MTP=0 是否可用？

Profile 命令 (参考 shared-contract):
```bash
PROFILE_DIR=/tmp/dsv4_mtp0_profile \
PROFILE_DELAY_ITERATIONS=4 \
PROFILE_MAX_ITERATIONS=5 \
PROFILE_ACTIVE_ITERATIONS=5 \
PROFILE_IGNORE_FRONTEND=1 \
NUM_SPECULATIVE_TOKENS=0 \
ENFORCE_EAGER=0 \
CUDAGRAPH_MODE=PIECEWISE \
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
- **AllReduce 不是 decode 瓶颈**: 仅 3% (之前误判为 48%)
- **INT8 KV Cache 无价值**: 仅 0.4ms benefit
- **W4A16 MoE GEMM 占 27%**: memory-bound at M=1, 但不是首要矛盾 (starvation 才是)
