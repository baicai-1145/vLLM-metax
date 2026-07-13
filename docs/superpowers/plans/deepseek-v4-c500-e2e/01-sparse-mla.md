# Sparse MLA decode/prefill 原生 MetaX kernel

## 会话目标

替换当前 MetaX sparse MLA decode 和 prefill 的 Torch reference 路径。这是整个
端到端计划的最高优先级，完成前不要继续微调 MHC post。执行顺序为 decode-first，
但 prefill OOM blocker 必须立即复现、修复并验收，不能后置。

## 已确认事实，不要重复调研

- MetaX decode：`vllm_metax/models/deepseek_v4/flashmla.py` 的
  `_torch_sparse_decode` 执行 FP32 gather、两次 matmul、mask、softmax 和 copy。
- MetaX decode 调用点：同文件 `_forward_decode` 直接调用该 Torch 函数。
- MetaX prefill：`vllm_metax/v1/attention/ops/flashmla.py` 明确记录原生
  `sparse_attn_global_fwd_kernel` trap，因此调用 `torch_flash_mla_sparse_prefill`。
- CUDA decode 参考：
  `/root/vllm-0.25rc1/vllm/models/deepseek_v4/nvidia/flashmla.py` 调用
  `flash_mla_with_kvcache`。
- CUDA prefill 参考：同文件调用 `flash_mla_sparse_fwd`。
- MetaX qnorm+RoPE+KV cache insertion 已有 fused op，不是第一优化对象。

## 最新证据与当前 blocker

- MTP=0、TP=4、PIECEWISE、100-token decode 为 `15.2334 tok/s`、`65.65 ms/token`，
  四卡利用率 `16.16%--16.42%`。
- 1K prefill 为 `1321.61 input tok/s`、`0.774812 s`。
- 10K default `chunk=8192` 在 `torch_flash_mla_sparse_prefill` 的
  `torch.index_select` 尝试约 `10 GiB` 分配并 OOM；`chunk=2048`、`GPU_MEM=0.8`
  可得 `1504.16 input tok/s`、median `6.648215 s`，四卡利用率约 `89.1%--89.2%`。
  该 workaround 不能替代 default-chunk 的 OOM 修复，prefill blocker 必须与 decode
  gate 同一会话持续跟踪。
- 已安装 `deep_gemm.int8_mqa_logits` 无 `backend` kwarg；动态 multi-head Triton
  loop 触发 `mcTriton SmallVector`。当前仓库 3D head-grid atomic workaround 已过
  differential、graph、16-token gate，但重复浮点差约 `1.5e-3`，因此保持 opt-in。
- Native sparse-MLA 合成测试 `23` 项通过；这不是真实 TP=4 corpus 或
  greedy-token gate 的替代。
- Attention-sink kernel validator 的 output `max_abs=7.8125e-3`、metadata
  `max_abs=5.72e-6`；3 次 graph replay 稳定、指针不变，未观察到 Torch
  `index_select`/matmul fallback。原始记录在
  `/tmp/sparse_sink_validation/pytest_targeted.log` 和
  `/tmp/sparse_sink_validation/inline_validation_v3.log`。该结果仅证明 sink
  kernel gate，不代表端到端 sparse-MLA 验收。
- TP=4、PIECEWISE 真实 prefill corpus 已在
  `.logs/dsv4_sparse_mla_corpus_tp4/` 生成 rank0--3 共 `16` 个 payload，
  data-only load 和独立 oracle replay `16/16` 通过。首次运行随后在
  DeepGEMM `fused_kv_cache.is_contiguous()` assertion 失败；根因为 indexer cache
  被跨层 packed allocation，实际 page stride `1660160` 而非连续的 `8448`。
  定向将 `...indexer.k_cache` 分配为 standalone tensor 后，四个 rank 均为
  `[9400,64,1,132]`、stride `[8448,132,132,1]`，TP=4 PIECEWISE 16-token
  诊断运行成功。这不代表 shared-contract 中的 frozen greedy oracle 已通过。
  原始证据在
  `.logs/dsv4_sparse_mla_capture_tp4_retry.log`、
  `.logs/dsv4_sparse_mla_capture_tp4_retry.run_manifest.json`、
  `.logs/dsv4_sparse_mla_capture_tp4_retry_manifest.json` 和
  `.logs/dsv4_indexer_layout_probe_tp4_standalone_indexer.log`。
- 真实 TP=4 eager decode corpus 已覆盖 SWA-only ratio 1 和 dual-cache ratio 4/128，
  位于 `.logs/dsv4_sparse_mla_eager_positive_decode_corpus_tp4/`。Native
  `torch_compat` 与 Torch oracle 在 `393216` 个 BF16 元素中仅 `10` 个不同，
  最大绝对差 `9.765625e-4`；但 TP=4 PIECEWISE token IDs 仍分叉，
  因此 native decode 不可 promotion。差分证据在
  `.logs/analyze_sparse_mla_eager_positive_bf16_diff.json` 和
  `.logs/dsv4_native_sparse_decode_tp4_gate_compat.log`。
- Scalar native prefill 已使 TP=4、PIECEWISE、default `chunk=8192`、10K prompt
  从原来约 `10 GiB` `index_select` OOM 变为可完成：`34.812055 s`、
  `287.256817 input tok/s`，证据在
  `.logs/dsv4_native_sparse_prefill_10k_chunk8192_gate.log`。Profiler 显示
  `_sparse_mla_prefill_kernel` 占首个 8192-token chunk device window 的 `86.64%`
  (`24.280846 s`)，见 `.logs/profile_native_sparse_prefill_10k_summary.json`。
  16-head tiled 实验退化到 `48.237275 s` 且生成 token 改变，已完整回退；
  当前仅能宣称 OOM blocker 解除，prefill 吞吐仍未验收。
- C500 launch sweep 显示 scalar kernel `num_warps=1,num_stages=2` 优于旧的
  `4,2`；产生实际 10K 端到端 `13.679399 s` / `731.026245 input tok/s`，
  1K 为 `1.650503 s` / `620.417001 input tok/s`。产生约束和 graph replay
  测试中的 output `max_abs=0.015625`、metadata `1.91e-5` 通过容差，
  但还未做同工作负载下的五次稳态吞吐验收。
- 共享 contract 的 TP=4 PIECEWISE gate 使用 native decode 时仍分叉；
  explicit `VLLM_METAX_DSV4_SPARSE_MLA_DECODE_BACKEND=torch_reference` 诊断后端则
  精确匹配 frozen IDs。这将问题隔离为 native decode 的 BF16 reduction
  顺序（填充 corpus 中 `393216` 个元素仅 `10` 个不同，但足以改变 token）。
  `torch_reference` 仅作 DIAGNOSTIC_ONLY，默认和验收仍为 native，因此 Sparse MLA 仍不能 promotion。
- 进一步的真实 late-decode stage dump 已将差异定位到 QK logits reduction，而非
  paged gather、value transpose 或 native FP32 GEMM：四个 rank 的 208 个有效 logits
  中分别有 `186/182/184/178` 个与 GPU Torch oracle 不同，最大绝对差
  `9.54e-7/1.43e-6/3.34e-6/1.43e-6`；无效 `-inf` 条目 `7360/7360` 精确。
  native probabilities、values、transpose 和 BF16 cast 的逐阶段证据保存在
  `.logs/dsv4_online_diff_stage_dump/late_decode_validation.json`，独立 logits
  probe 见 `.logs/dsv4_online_diff_stage_dump/late_decode_standalone_logits.log`。
  目前仍未证明 native TP=4 frozen greedy IDs 精确，不能进入 Plan 02。
- Native FP32 QK GEMM 加 MetaX persistent-softmax 候选在 TP=4 PIECEWISE
  2-token gate 精确得到 `[260,5036]`，但 16-token gate 从第 14 个 token
  分叉：expected 尾部 `[304,611,260]`，actual `[16,455,2004]`。在线逐层
  differential 的首次不一致分别出现在 rank1 call52/layer9、rank3
  call63/layer20、rank2 call96/layer10、rank0 call279/layer21，最大 BF16
  绝对差为 `0.001953125` 到 `0.00390625`。因此该集成已从默认 decode path
  移除，仅保留独立 native extension 和测试；证据在
  `.logs/dsv4_fp32_score_native_softmax_gate_16.log`、
  `.logs/dsv4_fp32_score_diff_run.log` 和 `.logs/dsv4_fp32_score_diff/`。
- 随后直接实例化已安装 MetaX PyTorch 的 `dispatch_softmax_forward` 模板，结合
  native FP32 QK/value GEMM 后，四个 rank 的真实 payload probabilities 均
  bitwise exact，graph replay 稳定；TP=4 PIECEWISE frozen 16-token gate 完全
  匹配。移除旧 prepare kernel 中重复的 QK/softmax，仅保留 value-only paged
  gather 后，正常 100-token decode 从 `12.569505 tok/s` 提升到
  `15.992848 tok/s`，超过会话起始 `15.2334 tok/s`。正确性与性能证据分别在
  `.logs/dsv4_fp32_score_metax_softmax_exact_gate_16.log`、
  `.logs/dsv4_value_only_gather_gate_16.log`、
  `.logs/dsv4_value_only_gather_benchmark_100.log`；旧 prepare kernel 的
  `407.696 us/call` / `22.49%` 归因在
  `.logs/profile_exact_sparse_decode_20260713/profiler_out_0.txt`。
- 最终 TP=4 prefill acceptance 禁用 prefix cache 以避免重复 prompt 命中缓存：
  1K 五次为 `[1.443799,1.430801,1.407319,1.426477,1.406168] s`，median
  `1.426477 s`、P90 `1.443799 s`、`717.852295 input tok/s`；10K、default
  `chunk=8192` 五次为
  `[12.970134,12.971603,12.946629,12.945564,12.941421] s`，median
  `12.946629 s`、P90 `12.971603 s`、`772.401833 input tok/s`，无 OOM。
  证据在 `.logs/dsv4_prefill_1k_five_run_acceptance_nocache.log` 和
  `.logs/dsv4_prefill_10k_chunk8192_five_run_acceptance_nocache.log`。此前启用
  prefix cache 的五次重复测量命中缓存并产生虚假的 `41K tok/s`，明确排除。

## 代码入口

- `vllm_metax/models/deepseek_v4/flashmla.py`
- `vllm_metax/v1/attention/ops/flashmla.py`
- `vllm_metax/models/deepseek_v4/sparse_mla.py`
- `vllm_metax/models/deepseek_v4/ops/fused_compress_quant_cache.py`
- `vllm_metax/v1/attention/backends/mla/flashmla_sparse.py`
- `tests/kernels/core/test_deepseek_v4_flashmla.py`
- CUDA 对照：`/root/vllm-0.25rc1/vllm/models/deepseek_v4/nvidia/flashmla.py`
- CUDA 公共布局：`/root/vllm-0.25rc1/vllm/models/deepseek_v4/sparse_mla.py`
- 可参考本地 kernel：`/root/TileOPs`、`/root/tilelang-metax`

## 第一任务：真实输入 differential harness

捕获真实 TP=4 decode/prefill 边界，payload 至少包含：

- q、output；
- paged SWA/KV cache 的实际 stride 和 padding；
- `swa_indices`、`topk_indices`、有效长度和无效索引；
- compression ratio、head padding、head/value dimension；
- softmax scale、attention sink；
- Torch oracle 输出和必要的 logits/LSE 中间值。

建议新增：

- `tools/debug/diff_deepseek_v4_sparse_mla.py`
- `tools/debug/probe_metax_sparse_mla.py`
- `tests/models/deepseek_v4/test_sparse_mla_diff.py`

保留并扩展现有测试入口：

```bash
pytest -q tests/kernels/core/test_deepseek_v4_flashmla.py
```

Corpus 必须覆盖 compression ratio 1、4、128，SWA-only、SWA+topk、边界长度、
无效 index 和至少四个 TP rank。不要只用随机 contiguous cache。

## Kernel 要求

- paged gather、index mask、online softmax 和 value reduction 在同一主 kernel；
- 不在 global memory 物化 FP32 score/probability；
- 支持当前 padded head layout 和 value dim 512；
- caller-owned output，禁止运行时替换指针；
- graph capture 内不得分配 Python/Torch 临时 tensor；
- accumulator 精度和 softmax 顺序先以 oracle 等价为准，再优化；
- 原生 kernel 失败时 fail closed，不能静默调用 Torch reference。
- 10K prefill 必须覆盖 default `chunk=8192` 的 OOM 复现和修复后行为；不得只报告
  `chunk=2048` workaround 的吞吐。

## 验收指标

- 所有真实 corpus 输出与 Torch oracle 满足约定精度；greedy IDs 完全一致；
- graph capture/replay 两组 payload，指针稳定；
- decode trace 中不再出现该阶段的 `aten::index`、`aten::bmm`、独立 softmax；
- decode attention 主路径不超过一个 planner 加一个 attention kernel；
- isolated sparse MLA 至少 3x 加速；
- TP=4 端到端正常推理相对会话起始基线有可重复提升。
- prefill 1K/10K 两档均通过，且 10K 不再触发约 `10 GiB` 的
  `torch.index_select` OOM；否则 Sparse MLA 不得推广。

当前以上 gate 已全部通过；Plan 01 完成。后续性能结果必须继续保留 frozen
TP=4 greedy gate，不得以吞吐回退本页已建立的 exact/native/default-chunk 证据。

## 会话任务提示

```text
完成 deepseek-v4-c500-e2e/01-sparse-mla.md。第一优先建立真实 paged-cache
differential corpus，然后替换 MetaX decode Torch sparse MLA；decode 完成并通过
graph/TP=4 gate 后再处理 prefill trapping kernel。不得用随机 tensor 或静默
Torch fallback 作为成功，不得修改 /root/vllm。
```
