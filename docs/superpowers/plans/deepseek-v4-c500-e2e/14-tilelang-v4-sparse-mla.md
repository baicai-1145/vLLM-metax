# tilelang V4 sparse MLA 整段融合 decode kernel

## 会话目标

对标 FlashInfer `trtllm_batch_decode_sparse_mla_dsv4`，用 tilelang-metax 实现
DeepSeek-V4 sparse MLA decode 的单融合 attention kernel，从根本上提升单请求
decode 的 SM 占用率（当前 36%，目标 70%+）。

这是经过六轮 device-time 分析后确认的**唯一能从根本提升 SM 的方向**。其他方向
（减 kernel 数量的小融合、MHC tilelang）已被三次实验证伪（fused_sinkhorn
device -47% 但 TPS +3.5%；fused_mhc_pre_norm TPS +0.36%）。

## 为什么是这个方向（诊断依据）

`13-main-model-decode-handoff.md` 的第六轮 device-time 构成分析结论：

1. FULL graph 下 TPS 30.18 ≈ device-bound 33.3（1000/30ms）的 **91%** —— host gap
   仅 ~3ms，FULL 下 **device kernel time 是 TPS 的真杠杆**（与 PIECEWISE 相反）。
2. 但 0%-occupancy kernel 的 grid 小是 **workload（单请求 M=1）决定的**：sinkhorn
   16 元素、sgemvt 12 head、aten 256-1024 元素——单 block 合理，不是实现差。
3. NV 单请求 99% SM 的秘密 = FlashInfer 整段 attention 融合：一个 kernel 吃
   q/kv/score/o，grid = num_heads × seq_len × split_k，单请求也很大。MetaX 把
   attention 拆成 sgemvt(12) + softmax + 多个小 kernel。
4. **fused_sinkhorn 的教训**：减了 kernel 数（134→6）但 grid 仍 [1,1,1]，故
   device -47% 但 TPS +3.5%。证明"减 kernel 数"在单 block 小张量场景无效。

**结论：必须把整段 attention 融成一个 grid 大的 kernel（split-KV 让单请求也占满
SM），而不是减 kernel 数量。**

## V4 sparse MLA 算法结构（可行性评估结论）

双 agent 评估（context-builder + researcher）确认的关键事实：

- **V4 没有 V3 的 kv_lora_rank 低秩吸收**。compressor 直接把每 token 压成 512 维
  KV entry，**K = V = C^Comp（MQA 共享）**。单 GEMM `Q[512]×KV[512]^T` 即可。
- RoPE **已预旋入 cache**（partial RoPE，最后 64 维），kernel 内**不需要任何
  RoPE 数学**。这比 tilelang 的 dense MLA 示例（需双 GEMM）还简单。
- 候选列表：SWA（128 token = 2 block）+ topk（512 token = 8 block）**拼接成单
  index 列表**，-1 padding 跳过无效 slot。
- attention_sink：softmax 分母加一个 `exp(sink_logit)` 项（V4-specific）。
- **indexer top-k selection 留在 kernel 外**（FlashInfer 也是这么做的）。

单 kernel 融合范围：gather（SWA ∪ topk 拼接索引）+ bmm1（512 维打分）+
softmax（+sink）+ bmm2（加权输出）+ split-KV merge。

## 单请求高 SM 的机制：split-KV

FlashInfer 单请求 99% SM 的核心 = **split-KV**：

- grid = `num_tokens × heads × KV_splits`
- V4: topk=512 候选 → 8 splits；乘以 16 heads（TP=4 每卡）= ~160 CTA
- 每 CTA 处理一个 query 的 64 条候选 KV，输出部分 O/LSE，再 merge
- C500 有 **104 SM**（manifest 权威值，不是 148），16 heads × 10 splits ≈ 160 CTA
  可填满（~1.5 wave）

split-KV reduce 必须用**确定性 merge kernel**（per-split scratch slot），**不能用
last-CTA atomic counter**——C500 的 atomic 非确定（manifest: scatter_add 50 次
replay 0 次 bitwise 相同）。

## C500 硬件约束（manifest 权威）

| 参数 | C500 值 | 影响 |
|------|---------|------|
| multi_processor_count | **104** | grid 目标 ≈ 104 CTA |
| shared_memory_per_multiprocessor | **64KB** | 远小于 H100 的 227KB，KV tile 受限 |
| warp_size | **64** | 不是 CUDA 的 32，线程组织需重排 |
| atomic determinism | **非确定** | split reduce 必须用 merge kernel |
| bf16_row_execution | 有已知 mismatch | 需差分验证 |

64KB shared 的缓解方案：block_N=32（32 token/块，32×512×2=32KB 单缓冲）或 KV
打包 FP8（584B/条）。

## MVP 已完成（2026-08-02，单卡 GPU0 实证）

`vllm_metax/kernels/tilelang_v4_sparse_mla_decode.py`（542 行）：

- ✅ tilelang 在 C500 **编译运行** sparse MLA decode kernel
- ✅ **BF16**（V4 native dtype）max_diff 0.001015（PASS < 1e-2）
- ✅ **10-seed BF16 全过**（0.0008-0.0011）
- ✅ **3-run 确定性**：bitwise 稳定（0.00e+00，无 atomic 竞争）
- ✅ **边界 case**：短序列（100 token）、长序列（3000 token/12 page）全过
- ✅ **split-k 触发**：num_split=9（10 blocks），combine 路径验证通过
- ✅ **graph capture 硬门控通过**：normal-vs-captured max_diff 0.00e+00，5x replay
  bitwise 稳定（证明 graph-safe，可集成 vLLM FULL/PIECEWISE）

工作配置：block_N=32（shared 48KB ≤ 64KB），block_H=16，threads=64，
num_stages=1，index_block=64，page_block_size=256，dtype=bf16。

grid = `(batch=1, heads/block_H=1, num_split=9)` split kernel +
`(heads, batch)` combine kernel。

实现中修复的 bug：attention-sink 原在 split kernel 内 per-split 加，combine 后
double-counted N×（num_split=9 时 error 0.036）。修复：sink 移到 combine kernel，
merge 完 LSE 后加一次。修正后 max_diff 1.5e-7。

## 当前缺口

MVP 是**功能正确**的，但还不是 production 可用：

1. **单请求 SM 未填满**：当前 grid `(1, 1, 9)` = 9 CTA，只占 104 SM 的 8.7%。
   需 **persistent/wave grid**（`total_tiles ≈ 104`）才能填满。评估报告：
   `example_mla_decode_persistent.py` 已有此机制（grid=sm_num + sync_grid combine）。
2. **未集成 vLLM**：当前是独立 kernel，未接入 attention.py dispatch。
3. **未过 TP=4 门控**：GPU2 硬件故障阻塞 TP=4 验证（见阻塞项）。
4. **性能未测**：无 TPS 数据（需 TP=4 benchmark-runner）。

## 阻塞项

- **GPU2 硬件故障**（2026-08-02）：`mx-smi` 显示 Not Available，compute queue
  `mxc_queue_acquire failed`，容器内 reset 失败（sysfs read-only）。需宿主机
  `mx-smi -i 2 -r` / driver reload / 冷重启。阻塞所有 TP=4 集成与验收。

## 实施顺序（GPU2 恢复后）

### Phase 1: persistent grid（单卡，1-2 天）

参考 `tilelang-metax/examples/deepseek_mla/example_mla_decode_persistent.py`：
- grid = `num_sm` 个 CTA，wave 调度（`waves = ceildiv(total_tiles, sm_num)`）
- `num_split` 作为自由调参轴，`total_tiles ≈ 104` 构造性 100% SM 填充
- split + combine 放**同一 kernel**（`T.sync_grid()`）
- 验证：单卡 profiler 看 SM occupancy 是否从 9 CTA → ~104 CTA

### Phase 2: vLLM 集成（2-3 天）

- 接入 `vllm_metax/models/deepseek_v4/attention.py` 的 attention_impl
- 构造 V4 的 attn_metadata（拼接 SWA+topk 索引、block_table、seqlens、sinks）
- env dispatch（`VLLM_METAX_DSV4_TILELANG_SPARSE_MLA=1`）
- 与现有 `sparse_mla_decode.py`（TORCH_COMPAT）共存，可回退

### Phase 3: TP=4 正确性 + 性能验收（2-3 天）

- kernel-validator：differential（对照 TORCH_COMPAT oracle）、graph replay、
  3-run 稳定性、生产 shape
- TP=4 16-token greedy hash 匹配冻结 baseline
  `fa5810ac3c5676dec33decf37da146499c3ec8610faa83d83bca7a7a796d696d`
- benchmark-runner：FULL graph 下 TPS（对照 30.18 基线）+ per-GPU SM%
- profiler：SM occupancy 是否从 36% 提升

## 验收指标

- TP=4 16-token greedy hash 匹配 baseline（正确性硬门控）
- FULL graph 下 median TPS > 30.18（+1% 才接受，否则记录原因）
- per-GPU SM% 提升（目标 50%+）
- 3-run bitwise 稳定 + graph replay 稳定
- 无 atomic 非确定性（用 merge kernel，不用 last-CTA atomic）

## 风险

1. **64KB shared + persistent grid 的张力**：persistent 多 CTA 共享 smem 分配，
   可能进一步压缩每 CTA 可用 smem。需实测 block_N/threads/stages 组合。
2. **tilelang fast-math 精度**：`exp2` 在 fast-math 下走近似实现（与已知
   `__builtin_mxc_rcpf` 同族），softmax 对 logits 误差敏感。MVP 已用 BF16 通过
   1e-2 容差，但 TP=4 greedy token 门控是更严的判据。
3. **BF16 gemm 确定性**：manifest 记录 bf16_row_execution 有 batched-vs-serial
   mismatch。MVP 3-run bitwise 稳定（0 diff），但 TP=4 多卡 allreduce 后需重验。

## 参考文件索引

| 文件 | 用途 |
|------|------|
| `vllm_metax/kernels/tilelang_v4_sparse_mla_decode.py` | **MVP**（本次产出，542 行）|
| `tilelang-metax/examples/blocksparse_attention/example_tilelang_sparse_gqa_decode_paged.py` | sparse GQA decode 骨架（split-k + flash tile）|
| `tilelang-metax/examples/deepseek_mla/example_mla_decode_persistent.py` | persistent grid + sync_grid combine（Phase 1 参考）|
| `tilelang-metax/examples/blocksparse_attention/heuristic.py` | split 数启发式 |
| `vllm_metax/kernels/sparse_mla_decode.py` | 当前 TORCH_COMPAT 实现（差分 oracle + 回退路径）|
| `artifacts/hardware/c500_maca_3.7.2/manifest.json` | C500 硬件权威参数（104 SM / 64KB smem / warp 64）|
| `.logs/deepseek_v4_mtp0_full_graph_20260801/` | FULL graph baseline（30.18 TPS）|
