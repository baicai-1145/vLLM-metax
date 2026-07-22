# DeepSeek V4 MTP k=4 无损 2x 加速实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to
> implement this plan task-by-task. Use TDD for every behavioral change and
> update each checkbox as evidence is produced.

**Goal:** 在 4x MetaX C500、TP=4、PIECEWISE graph 下完成 DeepSeek-V4-Flash
单 MTP head 的 k=4 迭代式 speculative decoding 适配；保持与 MTP=0 greedy 输出逐
token 完全一致，并在完全相同的正常 serving workload 上达到至少 `2.00x` 净
tokens/s。

**Architecture:** Target model 与 MTP layer 继续共用 TP=4 group，每个 rank 只加载
自己的 shard；同一个 MTP layer 迭代生成最多四个 draft token，target 一次验证最多
五个 token。保持 `max_num_seqs=1` 隔离尚未开放的多请求并发，但允许同一请求的
`1..5` token graph shapes。先建立 base/MTP 严格 differential 和 acceptance metrics，
再按 k=1→2→3→4 逐级通过正确性、graph 和性能门。

**Tech Stack:** Python 3.12、PyTorch/vLLM 0.25rc1、MetaX C500、TP=4、
PIECEWISE breakable CUDA Graph、DeepSeek V4 MHC、MTP/NextN、pytest、Ruff、
PyTorch profiler、JSON/JSONL artifacts。

## Global constraints

- 工作区固定为 `/root/vLLM-metax`；`/root/vllm-0.25rc1` 只读。
- 保留 dirty worktree；禁止 reset、revert、checkout 或覆盖已有修改。
- 模型固定为 `/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-MTP`。
- `_metax_sparse_C` SHA 固定为
  `28ea10c90d60189f47b38c41e24d3b3f59d6f31c847a6f4b23c19a8bccfaf821`。
- 验收只接受 TP=4；TP=1 只能用于明确标注的诊断。
- 正常模式固定为 Plan 02 exact-on、PIECEWISE、prefix cache on、`GPU_MEM=0.9`。
- MTP=0 正常基线为 `30.357923 TPS`、`32.940330 ms/token`；artifact 为
  `.logs/deepseek_v4_flash_quality_eval_20260716/batch_correctness_fix/post_fix_normal_tps/run.log`。
- 当前 checkpoint 为 `num_nextn_predict_layers=1`。k=4 必须复用同一个 MTP layer
  迭代 draft，禁止伪造或复制四个未训练 MTP layer。
- 当前 MTP=1 的 `17.8358 tok/s` 属于旧基线，且从第二个 token 起分叉，只能作为
  RED 证据，不能用于当前性能比较。
- 保持 `max_num_seqs=1`。本计划不重新开放多请求并发；k=4 的 token batch 仅来自
  同一请求的 draft/verification。
- 不允许 Torch arithmetic、eager、不同 attention backend 或静默 fallback 进入
  acceptance 数据。
- profiler TPS 不是正常 serving TPS；kernel microbenchmark 不能替代端到端结果。
- 模型加载 benchmark 必须交给 `benchmark-runner`，trace 必须交给 `profiler`，普通
  tests/lint 必须交给 `test-runner`，最终审查必须交给 `reviewer`。

---

## Success contract

“质量完全没问题”在本计划中定义为下列可审计条件全部成立，而不是仅比较答案分数：

1. MTP=0 base oracle 自身先通过三次 fresh-engine token exact 稳定性；若 base 不稳定，
   必须先修复 base，不得降低 MTP gate。
2. k=1、2、3、4 的每个 candidate 均与冻结 MTP=0 greedy oracle 逐 prompt、逐 token、
   逐 finish reason 完全一致；首个差异立即 hard fail。
3. 错误 draft 必须被 verifier 拒绝，不能改变 committed token、KV cache、position、
   RNG 或后续 hidden state。
4. 16-token、强制 100-token、GSM8K seed42 100 题和真实问答 corpus 均为 token exact
   `100%`；不能仅要求 correctness label 一致。
5. 三次 graph capture/replay、边界长度、提前 EOS 和重复运行均无 token ID `0` 尾巴、
   NaN/Inf、越界、旧 buffer replay 或 rank 分歧。
6. k=4 正常 serving median throughput 相对同一次、同 workload 的 MTP=0 baseline 至少
   `2.00x`；P90 speedup 至少 `1.80x`，五次 measured run 的 CV 小于 `3%`。
7. 若 fresh MTP=0 baseline 与 `30.357923 TPS` 的差异不超过 `3%`，k=4 median 还必须
   达到 `60.715846 TPS`；若基线漂移超过 `3%`，先解释并重跑，不得直接接受。
8. k=4 必须报告 mean acceptance length、平均 draft acceptance rate 和四个 position 的
   acceptance rate。社区参考是 draft 4 的平均接受长度约 `2.5--3.0`；该数字是性能
   诊断，不替代 exact-token gate。

本计划只对冻结的 greedy `temperature=0` serving contract 给出“无损”结论。随机
sampling 需要另建 RNG/distribution gate，不能从 greedy 结果外推。

## Required artifact layout

所有产物保存到：

```text
.logs/deepseek_v4_mtp_k4_exact_2x_YYYYMMDD/
  baseline/
  k1/
  k2/
  k3/
  k4/
  profiler/
  final/
```

每个目录至少包含 `command.txt`、`workload_manifest.json`、原始 stdout/stderr、
`exit_code.txt`、token JSON、acceptance JSON 和 summary。`final/` 还必须包含
`comparison.json`、`performance.csv`、`quality_decision.md` 和 `decision.md`。

### Task 1: 建立 MTP RED harness 和稳定 base oracle

**Files:**

- Create: `tools/debug/compare_deepseek_v4_mtp_consistency.py`
- Create: `tools/debug/parse_spec_decode_metrics.py`
- Create: `tests/tools/test_compare_deepseek_v4_mtp_consistency.py`
- Create: `tests/tools/test_parse_spec_decode_metrics.py`
- Modify: `tools/debug/evaluate_deepseek_v4_quality.py`
- Modify: `tests/tools/test_evaluate_deepseek_v4_quality.py`

**Interfaces:**

- `compare_mtp_consistency(base_paths, candidate_path) -> dict[str, Any]` 返回逐题
  `exact`、`first_diff`、finish reason、token count 和唯一 `pass/fail` decision。
- `parse_spec_decode_metrics(lines: Iterable[str]) -> dict[str, Any]` 返回 drafts、
  drafted/accepted tokens、mean acceptance length、平均和逐 position acceptance。
- quality evaluator 新增 `num_speculative_tokens: int = 0`，不再硬编码只接受0；当值
  大于0时 manifest 必须记录 method `mtp` 和 capture sizes。

- [ ] **Step 1: 写 comparator 的失败测试**

```python
def test_mtp_comparison_fails_at_first_divergent_token(tmp_path):
    base = write_artifact(tmp_path / "base.json", mtp=0, tokens=[10, 11, 12])
    candidate = write_artifact(
        tmp_path / "k1.json", mtp=1, tokens=[10, 99, 12]
    )

    result = compare_mtp_consistency([base], candidate)

    assert result["decision"] == "fail"
    assert result["comparisons"][0]["first_diff"] == 1
```

- [ ] **Step 2: 写 metrics parser 的失败测试**

```python
def test_parse_spec_decode_metrics_preserves_position_rates():
    line = (
        "SpecDecoding metrics: Mean acceptance length: 2.75, "
        "Accepted throughput: 50.00 tokens/s, Drafted throughput: 80.00 "
        "tokens/s, Accepted: 150 tokens, Drafted: 240 tokens, "
        "Per-position acceptance rate: 0.900, 0.700, 0.500, 0.400, "
        "Avg Draft acceptance rate: 62.5%"
    )

    result = parse_spec_decode_metrics([line])

    assert result["mean_acceptance_length"] == 2.75
    assert result["per_position_acceptance"] == [0.9, 0.7, 0.5, 0.4]
    assert result["avg_draft_acceptance_rate"] == 0.625
```

- [ ] **Step 3: 运行 tests 并确认 RED**

```bash
pytest -q \
  tests/tools/test_compare_deepseek_v4_mtp_consistency.py \
  tests/tools/test_parse_spec_decode_metrics.py \
  tests/tools/test_evaluate_deepseek_v4_quality.py
```

Expected: 新接口缺失导致失败，而不是 import 环境错误。

- [ ] **Step 4: 实现严格 comparator**

必须复用 `compare_deepseek_v4_quality_consistency.py` 的 provenance、prompt hash 和
token validation 规则，但只允许下列 manifest 字段在 base/candidate 间不同：

```python
_ALLOWED_MTP_DIFFERENCES = {
    "mtp",
    "num_speculative_tokens",
    "speculative_config",
    "compilation_config",
}
```

其余 workload、checkpoint、数据 SHA、sampling、TP、graph mode 和 evaluator SHA
任一不同都必须抛出 `ValueError`。

- [ ] **Step 5: 实现 metrics parser 和 evaluator 参数**

使用结构化正则解析现有 vLLM `SpecDecoding metrics:` 行。禁止从 TPS 反推接受率。
evaluator 的 MTP 配置必须是：

```python
speculative_config = None
if num_speculative_tokens:
    speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": num_speculative_tokens,
    }

capture_sizes = list(range(1, num_speculative_tokens + 2))
```

- [ ] **Step 6: 运行 unit tests、Ruff 和 py_compile**

```bash
pytest -q \
  tests/tools/test_compare_deepseek_v4_mtp_consistency.py \
  tests/tools/test_parse_spec_decode_metrics.py \
  tests/tools/test_evaluate_deepseek_v4_quality.py
ruff check tools/debug/compare_deepseek_v4_mtp_consistency.py \
  tools/debug/parse_spec_decode_metrics.py \
  tools/debug/evaluate_deepseek_v4_quality.py \
  tests/tools/test_compare_deepseek_v4_mtp_consistency.py \
  tests/tools/test_parse_spec_decode_metrics.py \
  tests/tools/test_evaluate_deepseek_v4_quality.py
python -m py_compile \
  tools/debug/compare_deepseek_v4_mtp_consistency.py \
  tools/debug/parse_spec_decode_metrics.py \
  tools/debug/evaluate_deepseek_v4_quality.py
```

- [x] **Step 7: 采集三次 fresh MTP=0 base oracle**

使用相同的固定 corpus、seed、TP=4、PIECEWISE、exact-on 和 serial request guard。
三次 artifact 必须 token exact；若当前已知的 `99/100` 运行间差异再次出现，停止 MTP
工作并将 base determinism 标为 blocker。

> **2026-07-18 current oracle:** 修复 sparse indexer 的 prefill/decode 边界后，三个
> fresh TP=4 PIECEWISE MTP=0 engine 均完成 85 prompts，三组两两比较均为 `85/85`
> token IDs + finish reason exact，`decision=pass`。冻结 oracle 为
> `baseline/indexer_prefill_decode_clean_masked_oracle3/run1/summary.json`；两两比较为同目录
> `run1_vs_run2.json`、`run1_vs_run3.json` 和 `run2_vs_run3.json`。以下 2026-07-17
> blocker chronology 作为历史诊断保留，但不再阻塞 k=1 入口。
> **2026-07-17 blocker evidence:** 两组独立 fresh-engine pair 均仅达到
> `98/100` token exact，且重复落在 Teresa（首差异 token 44）和 Gomer（首差异
> token 73）两个 prompt；因此第三次 oracle 和所有 `k>=1` gate 仍停止。生产输入上的
> native sparse indexer 已通过 eager/fresh-process/graph bitwise replay；exact-MHC 的四个
> native op 也已在 TP=4 corpus、两种 poison、20 次 fresh、20 次 cached 和 21 次 graph
> replay/rank 下通过全写入与 bitwise 稳定性验证。当前证据不支持修改 sampler、indexer
> 或盲目清零 MHC workspace。比较 artifact 位于
> `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/baseline/post_fix2_oracle2/`
> 和 `baseline/determinism_min/mhc_workspace_validation/`。
>
> 后续 graph-safe capture 在 request `84`、output token `73` 建立了不依赖全局 call
> index 的语义对齐 RED：layer 12 的四项 state bitwise exact，但同一 pair 的最终
> pre-`hc_head` residual 有 `14494/16384` 项不同；layer 15 pair 则在实际 token
> `12`/`438` 分叉时已出现 state 差异。layer 15 的 stage capture 显示 attention 返回时
> `residual`、`post_mix`、`res_mix` 已不同，而 attention 不修改这三项，因此首个坏边界
> 已缩小到 layer 13--15，并发生在 layer 15 attention 之前或其 MHC 输入。语义对齐
> comparison artifact 位于
> `baseline/determinism_min/residual_two_prompt_repro/`
> `request84_token73_layer{12,15}_comparison_run1_vs_run2.json` 和
> `request84_token73_layer15_stages_comparison_run1_vs_run2.json`。诊断 copy 会改变部分
> fresh-engine 分支选择，因此这些 artifact 只用于定位，不能替代三次无探针 oracle gate。
> layer 15 MHC-input copy 探针在三个 fresh engine 中都将 Gomer token 73 固定为
> `12`，且所有捕获 state exact；这是会改变 graph pool、地址布局或时序的
> Heisenbug，不是稳定性通过证据。对应 artifact 位于
> `baseline/determinism_min/residual_two_prompt_repro/`
> `request84_token73_layer15_mhc_input_run{1,2,3}/`。

同一 prefix-85 workload 的进一步 fresh-engine falsification 如下：

| 单一变量 | 运行间 exact | 首个差异 | artifact |
| --- | ---: | --- | --- |
| exact-MHC GEMV 改用显式 generic GEMM | `83/85` | Teresa 44：`270/327`；Gomer 73：`12/438` | `baseline/determinism_min/generic_gemm_fresh_pair/` |
| `USE_VLLM_TRITON_EXPERT=1` 诊断 fallback | `84/85` | Gomer 73：`12/438` | `baseline/determinism_min/vllm_triton_expert_fresh_pair/` |
| sparse-MLA decode 返回前 host synchronize | `83/85` | Teresa 44：`327/270`；Gomer 73：`438/12` | `baseline/determinism_min/sparse_mla_sync_fresh_pair/` |
| sparse-MLA compatibility graph | `84/85` | Teresa 44：`327/270` | `baseline/determinism_min/sparse_mla_compat_graph_fresh_pair/` |
| 关闭 shared-experts auxiliary stream | `84/85` | Teresa 44：`327/270` | `baseline/determinism_min/disable_shared_experts_stream_fresh_pair/` |

generic GEMM 和 Triton expert 仅是诊断路径，不可用于 acceptance；其余实验均保持
TP=4、PIECEWISE graph 和默认 MetaX MoE，无 eager/fallback marker。五个实验都未达到
fresh-engine exact，因此 exact-MHC GEMV、MoE 实现、sparse-MLA 完成同步、compatibility
graph 和 shared-experts overlap 均不是充分修复。下一诊断边界是 TP=4 BF16
all-reduce 以及生产形状 router GEMM + `topk_softplus_sqrt` 的 fresh-process/graph replay
bitwise 稳定性。这两个边界均已通过：all-reduce 使用捕获的 layer 12
`[1,4096]` BF16 输入，在 4 个 fresh TP=4 进程的每个 rank 上完成 21 次 eager 和
21 次 graph replay，通过 `MacaCommunicator`/PYNCCL 的输出跨 rank、跨进程
bitwise exact；router 在 layer 13/14、rank 0--3 的 8 个生产 case 上完成
3 个 fresh process、3 次 eager 和 21 次 graph replay，native FP32-output GEMM 与
`_moe_C.topk_softplus_sqrt` 的 logits、weight 和 expert ID 均 bitwise exact，无
fallback。artifact 位于 `baseline/determinism_min/tp_all_reduce_determinism/` 和
`baseline/determinism_min/router_topk_repeat_stability/`。下一单变量实验是完整模型
fresh-engine eager pair，用于判断分叉是否仅存在于 graph 路径；eager 结果不得
用于 acceptance。该 pair 也仅达到 `84/85`，并在 Teresa token 44 产生同样的
`327/270` 分叉，两次运行均记录 `enforce_eager=true`、`cudagraph_mode=NONE`；
因此分叉不是 CUDA Graph 独有问题。artifact 位于
`baseline/determinism_min/full_model_eager_fresh_pair/`。下一步在 eager layer 13--15
路径中分离尚未独立验证的 attention projection、o-proj 与 MoE 输出。

layer 14 的 routed-MoE 已使用真实捕获输入、四个 TP weight shard 和生产
`E=256,N=256,int4_w4a16` 配置完成独立验证。MetaX `fused_experts` 在 3 个 fresh
process、每个 rank 21 次 eager 和 21 次 graph replay 中均 bitwise stable，graph 与
eager 相同且 output pointer 稳定；四个 rank 的 routed partial 加四个捕获的 shared
partial 与生产 FFN 输出的最大绝对差为 `0.00348`。artifact 位于
`baseline/determinism_min/native_w4a16_moe_stability/`。shared-expert 的 gate/up 与 down
BF16 GEMM 及 clamp 激活重放也在 4 rank、3 fresh process 的 eager/graph 中 bitwise
stable，并逐 stage 匹配单次生产 capture；但保存的激活 harness 实际使用 Torch
`clamp/sigmoid/mul` 算术，不是 `SiluAndMulWithClamp.forward_native`。其 dispatch metadata
已更正为 `fallback=true`，所以该结果只能作为 GEMM/算术边界诊断，不能作为 native
activation 或 acceptance 的 fallback-free 证据。artifact 位于
`baseline/determinism_min/shared_expert_stability/`。

layer 15 的 ratio-128 C128A sparse-MLA decode 也用真实捕获的 q、index、cache 和输出，
在 native compatibility path 的 3 个 fresh process、21 次 eager 和 21 次 graph replay
中 bitwise stable，并 bitwise 匹配生产 capture。该重放使用 compact cloned cache，未保留
原生产 `swa_cache` 的 padded allocation stride，因此尚未排除地址或 stride 敏感性；不得
据此声明完整 sparse-MLA 路径已排除。artifact 位于
`baseline/determinism_min/layer15_sparse_decode_capture/` 和
`baseline/determinism_min/sparse_mla_decode_stability/`。

诊断 capture 本身会改变分叉：普通 layer 14 FFN capture 将首差提前到 frozen prefix-8
corpus 的 index 7/token 194，而 forward 内 CPU copy 和仅保留 tensor reference 的临时
模式都可令 fresh pair 变为 exact；reference-only 模式还出现过 3 次 fresh exact。两种
探针都会改变 allocator lifetime、地址布局或时序，相关 product code 和测试已撤销，
artifact 仅保留在 `baseline/determinism_min/eager_layer14_ffn_fresh_pair/`、
`eager_prefix8_layer14_ffn_fresh_pair/` 和 `eager_prefix8_layer14_ref_fresh_pair/`，不能作为
修复证据。当前下一验证边界是 layer 15 的 fused `wq_a/wkv`、`wq_b`、native
qnorm/RoPE/KV insert，以及 `inv_rope -> wo_a -> wo_b` projection chain；任何候选修复仍
必须回到无探针的三次 fresh MTP=0 oracle。

随后补采了 layer 15、token=1 的真实 O-proj payload：4 个 rank、decode position 0/634
共 8 个 case。native `inv_rope`、DeepGEMM `bf16_einsum` 和生产
`UnquantizedLinearMethod -> default_unquantized_gemm` 在 3 个 fresh process、每 case
21 次 eager 和 21 次 graph replay 中均 bitwise stable；`o_bf16`、`z`、`wo_b_local`
逐 stage 与 capture bitwise 相同，graph pointer 稳定，NaN-poisoned `z` 被完整覆写，且
无 fallback。采集运行本身为 TP=4 diagnostic eager，exit 0；由于 capture 会改变地址和
时序，该结果仍不是 fresh-engine oracle。artifact 位于
`baseline/determinism_min/layer15_o_proj_capture_eager/`。因此 O-proj 没有显示独立的
不稳定性，下一边界收紧为 layer 15 的 `wq_b` 与 native qnorm/RoPE/KV insert，以及保留
生产 padded cache stride 的 sparse-MLA replay。

layer 15 的 position 634、token=1 q pipeline 随后也完成 4 rank 真实捕获。生产
`default_unquantized_gemm` 的 `wq_b` 输出与捕获 `raw_q` bitwise 相同，native
`fused_deepseek_v4_qnorm_rope_kv_rope_insert` 输出与捕获 `post_q` bitwise 相同；3 个 fresh
process 中每 case 的 21 次 eager 和 21 次 graph replay 均 bitwise stable，graph 与 eager
相同、pointer 稳定、跨 fresh hash 相同，无 fallback。采集使用 TP=4 diagnostic eager 和
`MAX_TOKENS=2`，仅用于确保首个 decode forward 实际执行，不属于 acceptance workload。
artifact 位于 `baseline/determinism_min/layer15_q_pipeline_decode_capture_eager/`。attention
input/output projection 未显示独立不稳定性，当前最高优先级边界是用原生产 stride
`830080`（而非 clone 后的 `32768`）重放 layer 15 sparse-MLA cache。

该 padded-stride replay 也已通过：使用一个 `14,005,109,760` byte backing allocation
同时构造 `swa_cache` 与 `compressed_cache` view，二者 outer stride 均为生产记录的
`830080`。native compatibility sparse-MLA 在 3 个 fresh process、21 次 eager 和 21 次
graph replay 中 bitwise stable，输出与 capture bitwise 相同且跨 fresh hash 相同，graph
pointer 稳定，NaN-poison output 被完整覆写，无 fallback。artifact 位于
`baseline/determinism_min/sparse_mla_decode_padded_stride_stability/`。这排除了已知 cache
stride 差异作为充分原因，但不等同于复现相同虚拟地址。layer 15 attention 尚未独立验证
的输入边界只剩 fused `wq_a/wkv`、`q_norm/kv_norm` 和 compressor FP32-output GEMM；完成
这些边界后必须重新排序到 layer 13--14 或 allocator/address-sensitive 的组合路径，不能
继续重复已通过的 isolated kernel。

最后三个输入边界也已用同一 position 634 capture 完成：拼接 checkpoint 中的
`wq_a/wkv` 后调用生产 `default_unquantized_gemm`，再调用 Triton
`fused_q_kv_rmsnorm`，所得 normalized `qr/kv` 在 4 rank 上逐元素 bitwise 匹配 capture；
compressor 的生产 `torch.mm(out_dtype=torch.float32)` 输出在 3 fresh process、21 次
eager 和 21 次 graph replay 中稳定，三个输出的 graph pointer 稳定且跨 fresh hash
一致。artifact 与 q pipeline 共同位于
`baseline/determinism_min/layer15_q_pipeline_decode_capture_eager/`。至此 layer 15
attention 的所有独立 projection、norm、cache-insert、sparse-decode 和 cache-stride 边界
均未显示独立不稳定性。下一步不得再由 isolated-green 推断整层 green；应使用不做 CPU
copy、只改变 allocator lifetime 的无拷贝实验，在 layer 13--15 间定位组合性地址敏感
路径，并始终以无探针 fresh-engine pair 为 RED/green 判据。

无拷贝 lifetime bisection 随后在 frozen prefix-8 eager pair 建立了对照：无任何 probe 的
control 为 `7/8`，首差在 prompt 7/token 35；仅保留 layer 14 `after_attention` 的四个
tensor reference 为 `8/8`，继续缩减到只保留 `hidden_states` 仍为 `8/8`。在当前 stream
调用 `record_stream` 仍为 `7/8`，所有 layer 都保留 attention output 也仍为 `7/8`；只将
layer 14 的 token=1 attention output 保留到该 layer 下一次调用，则无诊断 env 的 product
candidate 为 `8/8`。artifact 位于 `baseline/determinism_min/`
`plain_prefix8_eager_fresh_pair/`、`retain_layer14_after_attention_fresh_pair/`、
`retain_layer14_attention_hidden_fresh_pair/`、`record_stream_attention_output_fresh_pair/`、
`retain_decode_attention_output_candidate/` 和
`retain_layer14_attention_output_candidate/`。该候选仍可能是 allocator-layout workaround，
不是已证明的底层 kernel root cause；只有 TP=4 PIECEWISE 的三次 fresh MTP=0 oracle 全部
exact 后才能进入 k=1。

该候选在真正的 TP=4 PIECEWISE gate 中失败：前两个 fresh engine 仅达到 `83/85`，
prompt 7/token 194 为 `14426/270`，Gomer token 73 为 `438/12`；因此按 gate 停止第三次
engine。artifact 位于 `baseline/post_lifetime_fix_oracle3/`。layer-14-only retention 只是
eager allocator-layout workaround，不是 graph-safe 修复；当前 stream 的 `record_stream`
同样无效。所有临时 reference-retention 和 candidate product code、对应 unit tests 已撤销，
仅保留 artifact。MTP=0 blocker 继续有效，下一步需要在 PIECEWISE graph 的 layer 14
attention-output 到 FFN exact-MHC 消费链上找到真实 stream/allocator ownership，而不是继续
用保活引用移动地址布局。

随后将 layer 14 token=1 attention output device-copy 到 persistent exact-MHC workspace，
再让 FFN MHC 消费稳定指针；该 graph-safe candidate 仍仅达到 prefix-8 `7/8`，首差移动到
prompt 7/token 145（`579/223`）。artifact 位于
`baseline/determinism_min/stable_workspace_layer14_ffn_graph_pair/`，candidate code 和临时
test 已撤销。这进一步排除了单一 FFN-MHC input pointer 失效，下一定位必须比较 PIECEWISE
graph segment 的 replay 输出边界。

为避免再次保活 graph tensor，随后加入了仅保存 Python `weakref`、在选定
`execute_model` replay 返回后才尝试 CPU clone 的边界探针。TP=4 PIECEWISE、frozen
prefix-8 的两个 fresh engine 均为 `8/8` exact，但 layer 13--15、三个 stage、四个 rank
的 8 个 payload 中 `weak_layer_outputs` 全部为空：graph capture 后 tensor wrapper 已在
post-replay serializer 运行前销毁，因此没有任何 stage tensor 可比较。artifact 位于
`baseline/determinism_min/weak_graph_layer13_15_prefix8_fresh_pair/`。该 exact pair 既不是
三次 fresh MTP=0 oracle，也没有取得 layer boundary evidence，不能解除 blocker；不得把
weak reference 替换成会改变 allocator lifetime 的强引用。下一定位机制必须由 graph
wrapper 暴露 replay-owned static output，或使用预分配、可证明不改变 tensor lifetime 的
最小 device-side observation，并先用无探针对照确认 RED 可复现。

当前 worktree 的无探针 TP=4 PIECEWISE control 随后再次稳定复现 blocker：两个 fresh
engine 均 exit 0、完成 85 prompts，但只有 `84/85` token IDs + finish reason exact；Gomer
request 84 在 output token 73 为 `438/12`，第三次 engine 按 fail-fast gate 未启动。artifact
位于 `baseline/current_unprobed_oracle3/`。这确认 weak-capture exact pair 只是较小样本未命中
分叉，不是修复。

shared-expert activation 的 provenance gap 也已关闭：MetaX 已安装
`torch.ops._C.silu_and_mul_with_clamp` 仍是 legacy 三参数 ABI，而 upstream v0.25 使用五参数
ABI，因此 OOT override 当前明确走 Torch arithmetic。legacy native op 在真实 layer 14
`[1,1024]` BF16 gate/up capture、4 rank、3 fresh process、21 次 eager + 21 次 graph replay
中 bitwise stable、poison/full-write 和 pointer gate 均通过；与现有 Torch output 的最大绝对
差为 `0.0078125`。但切换 product path 后 fresh oracle 退化到 `83/85`，出现两个新分叉并
改变大量 base sequences，因此 candidate code 和临时 test 已撤销。artifact 位于
`baseline/determinism_min/shared_expert_native_activation_stability/` 和
`baseline/native_shared_activation_oracle3/`；该 legacy kernel 既不是 MTP=0 修复，也不能
作为 acceptance path。

- [x] **Step 8: 复现当前 MTP=1 RED**

运行相同 corpus 的 k=1，保存首个 draft、target verification 和 committed token 差异。
验收 harness 必须稳定报告 `decision=fail` 和首个差异。

> **2026-07-18 current k=1 RED:** MetaX persistent batch-invariant matmul 已通过 native
> dispatch、production/boundary shape、20 次重复和 graph replay gate；开启全局
> `VLLM_BATCH_INVARIANT=1` 后，Melanie eager k=1 的原 index 108 分叉消失，但首差异移到
> index 190：frozen base/draft 为 `75242`，target/committed 为 `5328`，target logits
> 恰好同为 `35.5`，四 rank 一致。artifact 位于
> `k1/batch_invariant_persistent_metax_melanie_eager/` 和
> `k1/batch_invariant_persistent_metax_melanie_eager_capture/`。全局 invariant 的 MTP=0
> 会在 index 149 偏离 frozen oracle，因此它仍只是 k=1 诊断条件，不是新的 base oracle。
>
> 将“所有 exact-max tie 均接受 draft”接到实际 V1 rejection sampler 后，TP=4 eager
> Melanie 更早在 index 156 偏离：oracle/prior-k1 为 `34593`，candidate 为 `47593`；
> candidate 输出 221 tokens，oracle 为 212，finish reason 均为 `stop`。该策略未进入
> PIECEWISE gate，candidate code/test 已撤销。RED artifact 位于
> `k1/batch_invariant_v1_exact_tie_melanie_eager/`。这也验证 Task 4 的“不改 sampling”
> 约束必须保留；下一步继续定位 index 190 之前的最早数值边界，不得按 token ID 特判 tie。
>
> 单变量 RMSNorm probe 保留全局 invariant 的其他分支，只把 `RMSNorm.forward_cuda`
> 恢复到普通 native path；Melanie eager k=1 首差异反而提前到 index 26：oracle 为
> `509`，candidate 为 `850`，输出 206/212 tokens，finish reason 仍为 `stop`。probe
> code/test 已撤销，artifact 位于 `k1/batch_invariant_native_rms_norm_melanie_eager/`。
> 因此 index 190 不能归因于 RMSNorm 单点；全局 invariant 的改善与后续漂移存在跨算子耦合。
>
> LM-head-only probe 在全局 invariant 关闭时只尝试对最终 `ParallelLMHead` 使用 persistent
> batch-invariant linear；结果与原始 k=1 完全相同，仍为 195 tokens，首差异 index 108
> `28/1277`。该无效分支和 test 已撤销，artifact 位于
> `k1/batch_invariant_lm_head_only_melanie_eager/`。因此原 index 108 修复不来自最终
> unquantized vocab projection 单点。
>
> 对原 index 108 的无 invariant 路径完成了 position 741 语义对齐 capture。k=1 call 59
> row 0（inputs `[47593,28]`）与 base call 111（input `47593`）均与各自无探针序列完全
> 一致。layer 0 的 residual、pre-norm hidden 和 KV projection bitwise exact；低秩 Q 输入
> 仅 1 项相差 `9.31e-10`，raw/post Q 各 1 项相差 `1.49e-8`/`4.77e-7`，sparse attention
> 将其放大到 `0.001953125`，attention block 后 hidden 最大差为 `0.03125`。artifact 和
> 结构化比较位于 `k1/original_index108_layer_capture/`。
>
> 同一真实 position 的 same-input differential 进一步证明 44 个 attention layer 的
> `wq_b` M=2/M=1 均非 bitwise invariant（共 255 个 BF16 元素不同）；完整
> hidden→低秩 Q→RMSNorm→`wq_b` 的 rowwise M=1 结果在 layer 0 与 frozen base raw Q
> bitwise exact，而 batched 结果与 live k=1 raw Q bitwise exact。artifact 位于
> `k1/wq_b_same_input_diff/`。首个直接集成版本重复执行了 auxiliary projection/event，
> Melanie 在 index 14 `1527/260` 提前分叉，故 candidate code 已撤销；下一实现必须在
> 原 projection pipeline 内生成 rowwise Q，不能重入整段 multi-stream GEMM。
>
> 第二版 tokenwise full-Q probe 保留正常 upstream projection 和 auxiliary event，只将
> 2–5 token 的 fused Q projection 及 `wq_b` 替换为逐 row native call。TP=4 eager
> dispatch 证据确认 probe 生效，但 Melanie 仍在 index 26 偏离：frozen oracle 为 `509`，
> candidate 为 `850`；输出分别为 212/204 tokens，finish reason 均为 `stop`。因此该 probe
> 为 RED，未进入 PIECEWISE gate，candidate code 和两个专用 test 已撤销。artifact 位于
> `k1/tokenwise_full_q_v2_melanie_eager/`，其中 `exit_code.txt` 为 `0`，dispatch 记录位于
> `run.log`。
>
> 随后的 Q-only probe 保留 batched KV 和所有 auxiliary 输出，只将 fused projection 的
> 低秩 Q slice 改为逐 row native call。TP=4 eager 首差异由原 index 108 延后到 index 134：
> frozen oracle 为 `343`，candidate 为 `1527`；输出为 212/210 tokens，finish reason 均为
> `stop`。这证明低秩 Q projection 是必要但不充分的边界；下一单变量组合必须同时保留该
> Q-only 行为并将已证实 M=2/M=1 漂移的 `wq_b` 改为逐 row native call。artifact 位于
> `k1/tokenwise_q_only_melanie_eager/`。
>
> Q-only 与逐 row `wq_b` 的组合 probe 同时保留 batched KV，但再次在 index 14 提前
> 分叉：frozen oracle 为 `1527`，candidate 为 `260`；输出为 212/195 tokens，finish
> reason 均为 `stop`。两个 dispatch marker 均已命中，故完整 rowwise Q projection 仍不足以
> 恢复 exact sequence，candidate code/test 已撤销。artifact 位于
> `k1/tokenwise_q_only_wq_b_melanie_eager/`。下一边界转向 q_len=2 sparse FlashMLA 的
> actual cache length 与 causal masking。
>
> sparse FlashMLA 的 FP8/BF16 decode metadata 已改用真实 request sequence length，且两个
> native decode call 均显式传入 `causal=True`；CPU seam 在修改前 `4 failed`、修改后
> `4 passed`，TP=4 eager native run 也正常退出。但 Melanie token sequence 与原 k=1 完全
> 相同，仍在 index 108 `28/1277` 分叉，输出为 212/195 tokens。因此该语义修复保留，但
> 它不是当前首差异根因。artifact 位于 `k1/sparse_flashmla_causal_melanie_eager/`。
>
> 只替换 verifier row 0 的低秩 Q 与 `wq_b`、保留 row 1/batched KV/auxiliary 的 probe 仍在
> index 14 `1527/260` 提前分叉；限制到 target layers 0–42、排除 MTP drafter layer 43 后
> 结果不变，输出均为 197 tokens，finish reason 为 `stop`。因此 early RED 既非 row 1 也非
> drafter 污染，rowwise integrated substitution 不是可接受修复；candidate code/test 已撤销。
> artifacts 位于 `k1/tokenwise_q_row0_melanie_eager/` 和
> `k1/tokenwise_q_row0_target_only_melanie_eager/`。

### Task 2: 分离 request serialization 与 MTP verification graph sizes

**Files:**

- Modify: `vllm_metax/platform.py`
- Modify: `tests/compat/test_dsv4_serial_requests.py`

**Interfaces:**

- 新增 `_dsv4_safe_capture_sizes(vllm_config) -> list[int]`。
- `max_num_seqs` 继续固定为1；MTP k 的同请求 graph sizes 为 `1..k+1`。

- [ ] **Step 1: 写 graph size RED tests**

```python
def test_serial_requests_allow_k4_single_request_graphs(monkeypatch):
    config = dsv4_config(max_num_seqs=8, speculative_tokens=4)
    monkeypatch.delenv("VLLM_METAX_DSV4_ALLOW_UNSAFE_BATCHING", raising=False)

    assert _enforce_dsv4_serial_requests(config)
    assert config.scheduler_config.max_num_seqs == 1
    assert config.compilation_config.cudagraph_capture_sizes == [1, 2, 3, 4, 5]
    assert config.compilation_config.max_cudagraph_capture_size == 5
```

还必须测试 MTP=0 仍为 `[1]`，其他模型不变，非法 k 或未知 speculative method fail
closed。

- [ ] **Step 2: 运行测试确认当前实现错误地过滤 size 2--5**

```bash
pytest -q tests/compat/test_dsv4_serial_requests.py
```

- [ ] **Step 3: 实现最小配置修复**

只允许 `method == "mtp"` 且 `1 <= num_speculative_tokens <= 4` 时保留
`range(1, k + 2)`。不得使用 `VLLM_METAX_DSV4_ALLOW_UNSAFE_BATCHING=1`，因为该变量
会同时重新开放未经验证的多请求并发。

- [ ] **Step 4: 运行 focused tests**

```bash
pytest -q tests/compat/test_dsv4_serial_requests.py
ruff check vllm_metax/platform.py tests/compat/test_dsv4_serial_requests.py
```

### Task 3: 捕获并定位 k=1 第二 token 分叉

**Files:**

- Create: `vllm_metax/models/deepseek_v4/mtp_debug.py`
- Create: `tools/debug/diff_deepseek_v4_mtp.py`
- Create: `tests/models/deepseek_v4/test_mtp_debug.py`
- Modify: `vllm_metax/models/deepseek_v4/mtp.py`

**Interfaces:**

- `maybe_capture_mtp_stage(stage, spec_step_idx, tensors)` 仅在显式
  `VLLM_METAX_DSV4_MTP_CAPTURE_DIR` 下启用。
- schema 必须保存 rank、step、stage、shape/dtype/stride、input IDs、positions、draft
  IDs、target IDs、accepted count、pre/post hidden state hash 和 logits top-k。
- capture 在 CUDA graph capture 时必须 fail closed；先用 eager 定位，再用 profiler
  验证 graph 路径，不允许 capture helper 污染性能数据。

- [ ] **Step 1: 为 stage schema 和首差定位写 unit tests**

```python
def test_mtp_diff_reports_propose_verify_commit_boundary(tmp_path):
    write_stage(tmp_path, "propose", draft=[11], target=[12])
    write_stage(tmp_path, "verify", draft=[11], target=[12], accepted=0)
    write_stage(tmp_path, "commit", committed=[11])

    result = analyze_capture(tmp_path)

    assert result["first_invalid_boundary"] == "commit"
```

- [ ] **Step 2: 添加 opt-in capture points**

至少覆盖：MTP input RMS 前后、`h_proj/e_proj` 后、MTP block 后、hc_head/logits 后、
每个 `spec_step_idx` 返回前。不得默认写盘、同步或记录完整权重。

- [ ] **Step 3: 采集 k=1 eager red corpus**

比较以下边界：

1. target 保存的 pre-hc_head residual；
2. MTP 接收的 `previous_hidden_states` layout；
3. `inputs_embeds` 与 position 对齐；
4. draft top-1 token；
5. target verification token；
6. rejected draft 后实际 committed token；
7. 下一步 KV slot、position 和 hidden state。

- [ ] **Step 4: 写唯一根因 decision**

只能将分叉归到最早出现差异的边界。若 verifier 已判 reject、commit 却仍写入 draft，
修 verifier/commit patch；若 draft 前 hidden state 已错，修 V4 MTP layout/state；禁止同时
修改多个边界碰运气。

### Task 4: TDD 修复 k=1 exact greedy

**Files:**

- Modify when proven: `vllm_metax/models/deepseek_v4/mtp.py`
- Modify when proven: `vllm_metax/models/deepseek_v4/model.py`
- Create only for upstream verifier defect:
  `vllm_metax/patch/bugfix/spec_decode/deepseek_v4_mtp.py`
- Modify: `vllm_metax/patch/__init__.py`
- Modify: `tests/kernels/core/test_deepseek_v4_mtp.py`
- Create: `tests/v1/spec_decode/test_deepseek_v4_mtp.py`

- [ ] **Step 1: 把 Task 3 最小复现写成失败测试**

测试必须覆盖错误 draft 被 reject 后 committed token 等于 target token，并且下一个 step
使用 target token 的 position/KV state。

- [ ] **Step 2: 增加 fused MTP input RMS 边界 tests**

参数化 `num_tokens=[1, 2]`、position0 masking、非连续输入、BF16、NaN/Inf 和 output
buffer。reference 必须保持 FP32 RMS 计算顺序。

- [ ] **Step 3: 实现最小根因修复**

保持 V4 contract：target stash 为 flat `(T, hc_mult * hidden)` pre-hc_head residual；MTP
入口 reshape 为 `(T, hc_mult, hidden)`；每个 spec step 返回的 residual 必须能无损供下一
step 使用。不得改 checkpoint 权重、sampling 或 greedy oracle。

- [ ] **Step 4: 运行 k=1 unit、graph 和 TP=4 gates**

```bash
pytest -q \
  tests/kernels/core/test_deepseek_v4_mtp.py \
  tests/v1/spec_decode/test_deepseek_v4_mtp.py \
  tests/models/deepseek_v4/test_mtp_debug.py
```

随后运行 TP=4 eager 2-token、PIECEWISE 16-token、强制 100-token 和小型真实 corpus。
k=1 必须 token exact 后才允许进入 Task 5。

### Task 5: 按 k=2、3、4 扩展同一 MTP head

**Files:**

- Modify: `vllm_metax/models/deepseek_v4/mtp.py`
- Modify: `vllm_metax/models/deepseek_v4/model.py`
- Modify if required: `vllm_metax/models/deepseek_v4/ops/mhc/tilelang.py`
- Modify: `tests/kernels/core/test_deepseek_v4_mtp.py`
- Modify: `tests/models/deepseek_v4/test_mhc_raw_diff.py`
- Modify: `tests/v1/spec_decode/test_deepseek_v4_mtp.py`

- [ ] **Step 1: 写 spec step state-machine tests**

```python
@pytest.mark.parametrize("k", [2, 3, 4])
def test_each_spec_step_recycles_previous_mtp_residual(k):
    result = run_fake_mtp(k=k)
    assert result.spec_step_indices == list(range(k))
    assert result.input_state_hashes[1:] == result.output_state_hashes[:-1]
```

- [ ] **Step 2: 写 token batch 1..5 kernel tests**

exact MHC、MTP input RMS、attention metadata、KV slot mapping 和 logits 必须覆盖
`num_tokens=[1,2,3,4,5]`。batch result 必须与 stacked single-token oracle bitwise 相等，
除非已有 contract 明确规定 FP32 ULP tolerance。

- [ ] **Step 3: 写 partial verification tests**

覆盖 position1、2、3、4 分别 reject、全部4个接受、提前 EOS、距离 max model length
不足4 token。未使用 slot 必须保持无效，不能写 token ID `0` 或污染下一请求。

- [ ] **Step 4: 实现迭代 draft**

同一个 MTP layer 按 `spec_step_idx=0..k-1` 迭代；每一步只消费上一步真实返回的 state。
禁止为 k=4 复制权重或假设 checkpoint 有四个 MTP layer。

- [ ] **Step 5: 逐级运行 k=2、k=3、k=4 gates**

每一级必须先通过 eager，再通过 PIECEWISE graph，最后通过 TP=4 token oracle。任何一级
失败都停止，不得跳到更大的 k 掩盖问题。

### Task 6: 完成质量无损矩阵

**Files:**

- Modify: `tools/debug/evaluate_deepseek_v4_quality.py`
- Modify: `tools/debug/compare_deepseek_v4_mtp_consistency.py`
- Create: `tools/debug/evaluate_deepseek_v4_mtp_real_qa.py`
- Create: `tests/tools/test_evaluate_deepseek_v4_mtp_real_qa.py`

- [ ] **Step 1: 快速 frozen oracle**

运行 MTP=0 和 k=1..4 的同 prompt 16-token 与强制 100-token gate。要求所有 k 的 token
IDs、finish reason 和 token count 完全相同。

- [ ] **Step 2: GSM8K seed42 100题**

沿用已冻结 train/test SHA、5-shot prompt、`temperature=0`、`max_tokens=512`。base 与
k=4 必须 `100/100` token sequence exact；accuracy、invalid、length termination 和
token-zero-tail 必须完全相同。

- [ ] **Step 3: 真实问答 corpus**

至少覆盖事实、数学、代码、长推理、提前 stop 和接近512 token 输出。raw completion、
chat non-thinking、thinking prompt protocol 分开比较，不混合评分。

- [ ] **Step 4: 重复与 graph 安全**

三次 fresh engine、同 engine 连续请求、两种 prompt 长度交错。`max_num_seqs=1` 保持不
变；这里验证的是 MTP token batch，不是重新开放动态 request batching。

- [ ] **Step 5: kernel-validator gate**

验证 production shapes、BF16/W4 layouts、非对齐边界、stable pointers、capture 加至少
20次 replay。hidden fallback、allocation drift、token mismatch 或 graph failure 均为失败。

### Task 7: 测量 acceptance 并优化到正常 2x

**Files:**

- Modify: `tools/tmp_deepseek_v4_mtp_generate.py`
- Modify: `tools/run_deepseek_v4_mtp_generate.sh`
- Modify: `tests/tools/test_parse_spec_decode_metrics.py`
- Modify only when profile proves necessary:
  `vllm_metax/models/deepseek_v4/mtp.py`

- [ ] **Step 1: 冻结强制100 token 性能 workload**

runner 新增显式 `MIN_TOKENS=100` 支持。base 与 candidate 均设置：

```bash
TP=4 GPU_MEM=0.9 ENFORCE_EAGER=0 CUDAGRAPH_MODE=PIECEWISE \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 MAX_MODEL_LEN=512 \
MAX_NUM_BATCHED_TOKENS=8192 MAX_TOKENS=100 MIN_TOKENS=100 \
INPUT_TOKENS=0 ENABLE_PREFIX_CACHING=1 WARMUP_REQUESTS=3 BENCH_RUNS=5
```

分别运行 `NUM_SPECULATIVE_TOKENS=0` 和 `4`，其他变量必须完全相同。

- [ ] **Step 2: 记录 k=1..4 acceptance 曲线**

每个 k 保存 mean acceptance length、平均 draft acceptance、逐 position acceptance、
accepted/drafted throughput。k=4 若 mean acceptance length 小于 `2.5`，先报告模型/量化
接受率 blocker，不得通过隐藏低接受率美化 TPS。

- [ ] **Step 3: 建立 k=4 迭代预算**

正常基线 `32.940330 ms/token` 的2x目标为 `16.470165 ms/committed token`。若 mean
acceptance length 为 `L`，一次完整 draft+verify iteration 必须满足：

```text
iteration_latency_ms / L <= 16.470165
```

例如 `L=2.5` 时 iteration 必须不超过 `41.175 ms`；`L=3.0` 时不得超过
`49.410 ms`。

- [ ] **Step 4: 采集一组五 steady-step profiler**

profile 必须包含四 ranks、固定 active window、MTP draft steps、target verification、
sampling/commit 和 graph gaps。区分 MTP layer compute、verification batch、MCCL wait、
D2D copy 和 host gap，禁止相加重叠异步事件。

- [ ] **Step 5: 只优化 profiler 证明的最大 MTP 开销**

优先顺序由当前 trace 决定，候选包括：MTP layer graph capture、verification size5
graph、重复 embedding/head、KV metadata rebuild、draft step host round-trip。不得在没有
trace 证据时同时修改 target MHC、MoE 或 MCCL。

- [ ] **Step 6: benchmark-runner 正常验收**

同 workload 分别运行 base 和 k=4，报告五次 latency、median/P90 TPS、CV 和四卡平均
利用率。通过条件：median speedup `>=2.00x`、P90 speedup `>=1.80x`、CV `<3%`。

### Task 8: 最终推广、文档和审查

**Files:**

- Modify only after every gate passes: `tools/tmp_deepseek_v4_mtp_generate.py`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/00-shared-contract.md`
- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/07-integration-acceptance.md`
- Modify: `docs/superpowers/plans/deepseek-v4-c500-e2e/README.md`
- Update: this plan's checkboxes and final decision section

- [ ] **Step 1: 默认值推广门**

只有 Task 1--7 全部通过时，才能将官方 wrapper 的默认
`NUM_SPECULATIVE_TOKENS` 从0改为4。必须保留显式 `NUM_SPECULATIVE_TOKENS=0` rollback。
任一 gate 未通过则保持默认0，并在 decision 中写明 blocker。

- [ ] **Step 2: 完整静态和测试验证**

```bash
pytest -q \
  tests/kernels/core/test_deepseek_v4_mtp.py \
  tests/models/deepseek_v4/test_mhc_raw_diff.py \
  tests/v1/spec_decode/test_deepseek_v4_mtp.py \
  tests/compat/test_dsv4_serial_requests.py \
  tests/tools/test_compare_deepseek_v4_mtp_consistency.py \
  tests/tools/test_parse_spec_decode_metrics.py \
  tests/tools/test_evaluate_deepseek_v4_quality.py \
  tests/tools/test_evaluate_deepseek_v4_mtp_real_qa.py
mapfile -t py_files < <(
  { git diff --name-only -- '*.py'; \
    git ls-files --others --exclude-standard -- '*.py'; } | sort -u
)
if ((${#py_files[@]})); then
  ruff check "${py_files[@]}"
  python -m py_compile "${py_files[@]}"
fi
mapfile -t md_files < <(
  { git diff --name-only -- '*.md'; \
    git ls-files --others --exclude-standard -- '*.md'; } | sort -u
)
if ((${#md_files[@]})); then
  markdownlint-cli2 "${md_files[@]}"
fi
git diff --check
sha256sum vllm_metax/_metax_sparse_C.abi3.so
```

- [ ] **Step 3: 最终 reviewer gate**

reviewer 必须检查 verifier correctness、rejected draft state、KV/position、graph padding、
RNG、TP rank consistency、fallback、allocation、测试缺口和默认值回退。P0/P1/P2 finding
未解决不得推广。

- [ ] **Step 4: 写 final decision**

`decision.md` 必须列出：base/k4 token hashes、GSM8K和真实问答 exact 比较、acceptance
曲线、normal median/P90/CV、四卡利用率、profile decomposition、fallback 状态、binary
SHA、review 结果和 residual risks。

## 2026-07-19 continuation evidence

The opt-in rowwise native diagnostics now cover the complete TP=4 eager
speculative path: initial MHC pre, fused QKV projection, `wq_b`, Q/KV cache
insert, sparse MLA decode, output projection, and FFN. Unverified bonus
suppression remains enabled so committed output contains one token per step.

Focused regression coverage is `25 passed` across:
`test_mhc_tokenwise.py`, `test_prefill_gemm_chunking.py`,
`test_dsv4_drop_bonus.py`, and `test_flashmla_sparse_decode.py`.
Ruff, `py_compile`, and `git diff --check` pass for the touched modules.

The complete TP=4 eager candidate artifact is
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_complete_native_qkv_insert_wired_drop_bonus_melanie_eager/`.
All six rowwise dispatch warnings are present and the process exits 0, but the
token gate is RED: frozen oracle length 212 versus candidate length 206, first
mismatch index 26 (`509` versus `850`).

Aligned all-layer capture at call 28 / logical position 658 shows layers 0--2
bitwise exact after the rowwise stages. Follow-up row-selected captures at
logical position 651 refined the boundary: layers 0--2 attention inputs, Q
stages, and attention output inputs are exact for the selected row, but layer 2
`after_attention` first carries a small BF16 difference (132 elements, max
absolute difference `0.015625`) which propagates through layer 2 `after_ffn`
and becomes a layer 3 sparse-attention input difference. Layer 3 Q/KV cache
insert is therefore downstream of the mismatch: its selected-row `kv` input
already differs from base by 201 BF16 elements (max `0.015625`), matching the
later SWA cache row mismatch at logical position 651. Physical slot/block IDs
can differ across MTP=0 and MTP=1 allocation, so the observed one-block slot
difference is not by itself treated as a bug.

The sparse MLA decode capture hook now accepts optional logical positions and
token-to-request indices, with
`VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_POSITIONS` selecting only matching decode
rows while retaining full cache and block-table tensors for replay. This is
diagnostic-only and remains inert unless
`VLLM_METAX_DSV4_SPARSE_MLA_CAPTURE_DIR` is set. Focused validation for the new
capture filter is `20 passed` in `test_prefill_gemm_chunking.py`; the existing
torch-reference `_forward_decode` dispatch test also passes after the signature
update.

The first position-filtered layer 2 sparse MLA decode capture excludes sparse
decode as the selected-row source. Candidate artifacts emitted two calls for
logical position 651 and base emitted one. In both candidate calls, selected-row
`q` and sparse decode `output` are exact versus base. The SWA block table still
differs at logical block 10 (`181` versus `180`), SWA indices differ in the last
12 window entries by one physical block, and two top-k indices are swapped; none
of those metadata differences change the selected sparse output. The next
discriminating boundary is therefore layer 2 attention output projection or the
post-attention stage around it, not sparse MLA decode itself.

All-rank O-proj and sparse captures refine the source to rank 3 indexer/top-k
ordering. For layer 2 logical position 651, ranks 0--2 sparse outputs are exact.
Rank 3 sparse output differs in exactly one BF16 element (max `0.001953125`).
The O-proj input on rank 3 carries the same one-element difference, expands to
484 differing elements in local `wo_b`, and then all ranks receive the same
post-all-reduce O-proj output difference (132 BF16 elements, max `0.015625`).
Rank 3 SWA rows selected by the physical-block difference are byte-exact; the
actual sparse-output difference comes from compressed top-k order. Base maps
local top-k indices `[93, 118]` to global `[157, 182]`, while candidate maps
`[118, 93]` to `[182, 157]`. The matching indexer payloads are base layer 2
rank 3 `call18` row 0 versus candidate `call17` row 1 / `call18` row 0. K
cache, block table, and weights match at the compared rows, but the candidate
quantized indexer Q row has one element different, enough to swap the near-tie.

A diagnostic fix split the indexer Q projection and INT8 quantization
tokenwise under the existing `VLLM_METAX_DSV4_TOKENWISE_WQ_B=1` switch.
Focused unit coverage passes, but the fresh TP=4 eager token gate remains RED:
candidate length is 203 tokens versus oracle length 212, and the first mismatch
is still index 26 (`850` versus `509`). Therefore the tokenwise indexer Q split
is useful diagnostic narrowing, not an accepted exactness fix.

The current post-indexer-Q captures refine the boundary again. For layer 2
logical position 651, candidate `call1` matches the base sparse MLA path across
all ranks: top-k indices and sparse outputs are exact. Candidate `call0` for the
same logical position has exact `q` but differs in SWA physical block/window
metadata, compressed top-k, and sparse output by only a few BF16 elements
(rank 0: one element, max `3.0517578125e-05`; rank 3: four elements, max
`0.00048828125`). Layer 2/3 QKV-insert captures show both candidate calls write
the same physical slot; layer 3 `call1` `q`/`kv` is exact versus base and
overwrites the earlier differing `call0` row. This excludes QKV insert itself
as the remaining source and leaves row selection / verifier target-logit input
as the next discriminating boundary.

Fresh verifier capture with the current tokenwise-indexer-Q environment is
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/current_indexer_q_verifier_capture/`.
It reproduces the RED token stream (203 tokens, `tokens[24:30] =
[223, 20, 850, 304, 270, 4332]`) and records no rejected-draft commit bug:
rank 0 commit row 25 has `draft_ids=[850]`, `target_ids=[850]`,
`committed_ids=[850]`, and target-logit top-5 begins
`850:32.75`, `509:32.5`, `304:32.0`. Therefore the wrong token is already the
verifier target argmax for the committed row, not a sampler parse/bookkeeping
leak of an unverified draft.

The active diagnostic path is the V1 rejection sampler, not the V2 metadata
path (`VLLM_USE_V2_MODEL_RUNNER` is disabled by the MetaX platform plugin). A
diagnostic-only capture extension now records V2 verifier row metadata when
that path is active, and focused validation is `9 passed` in
`tests/models/deepseek_v4/test_mtp_debug.py` with Ruff and `py_compile`
passing. The follow-up TP=4 artifact
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/current_indexer_q_verifier_position_capture/`
confirms V1 is still the active capture path: proposal-stage records contain
positions, but verifier `commit` records do not yet have `row_indices` or
`expanded_local_pos`. The next probe should add a high-level V1 sampler capture
around `RejectionSampler.__call__` (or equivalent) that records
`metadata.logits_indices`, `target_logits_indices`, `bonus_logits_indices`, and
input positions for the target rows before running another TP=4 model load.

The V1 sampler/model-runner metadata probe is now present and has a focused
unit gate of `14 passed` across `test_mtp_debug.py` and
`test_dsv4_drop_bonus.py` with Ruff and `py_compile` passing for the touched
debug/patch files. Artifact
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/model_runner_metadata_capture/`
reproduces the RED stream (`tokens[24:30] =
[223, 20, 850, 304, 270, 4332]`). Around the first mismatch, rank 0 records
`input_ids=[20,850]`, `positions=[659,660]`,
`target_input_ids=[20]`, `target_positions=[659]`, and
`bonus_input_ids=[850]`, `bonus_positions=[660]`; the target top-5 for the
verifier row is already `850:32.75`, `509:32.5`, `304:32.0`. This confirms the
wrong committed token is not rejected-draft leakage or sampler bookkeeping: the
verifier target logits for input token `20` at position 659 already choose
`850`.

Position-659 layer captures correct the earlier row-selection ambiguity. The
default `LayerCaptureContext.save_stage` stores only the last row of a
multi-row candidate batch, while cloned `positions`/`input_ids` metadata remain
full; a capture enabled by position 659 can therefore store tensor row 660
unless `VLLM_METAX_DSV4_LAYER_CAPTURE_FULL_TENSORS=1` is used and rows are
matched by cloned position. The original clean artifacts are
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer0_3_stage_pos659_candidate/`
and
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer0_3_stage_pos659_base/`.
Both exit 0; candidate remains RED (`[223,20,850,304,270,4332]`) and base
matches the oracle window (`[223,20,509,270,4332,4200]`). A partial full-tensor
candidate rerun
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer0_3_stage_pos659_candidate_full/`
was intentionally interrupted after the needed call files were written, because
the unfiltered full-tensor capture continued copying later rows. Its selected
position-659 row shows candidate `call1` is bitwise exact versus base through
layers 0--3 for MHC state, attention inputs, Q stages, attention output, and
FFN output.

Narrowed all-layer diagnostic captures with `MAX_TOKENS=40` and call filters
were then used only to localize the mismatch, not as acceptance evidence:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer_all_call1_pos659_candidate_full_mt40/`
and
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer_all_call0_pos659_base_mt40/`.
Both wrote summaries and the required captures but were interrupted during
cleanup, so they have no clean exit-code gate. They show the first true
candidate/base divergence for the verifier target row is layer 4
`attention_output`: normalized attention input and Q stages are bitwise exact,
but rank-0 local attention output differs in four BF16 elements (max
`0.00048828125`) and the difference propagates through later layers.

Layer-4 sparse MLA captures isolate that first numerical difference further:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer4_sparse_pos659_candidate_mt40/`
and
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer4_sparse_pos659_base_mt40/`.
For candidate `call1` versus base `call0`, all ranks have exact `q` and exact
`topk_indices`. Ranks 1--3 sparse outputs are exact. Rank 0 differs by the same
four BF16 elements seen in layer capture (max `0.00048828125`). The rank-0 SWA
window differs only in physical mapping for the final 20 rows: candidate
`swa_block_table[0,10]=180` and indices `11520..11539`, versus base block `181`
and indices `11584..11603`; `swa_lens`, top-k lens, top-k indices, and every
referenced SWA cache row value are exact after mapping candidate and base
physical indices.
Kernel-validator replay then falsified the physical-index/layout-sensitive
sparse-decode hypothesis. Native replay of the captured candidate and base
rank-0 layer-4 sparse MLA inputs exactly reproduces each captured output, and
remapping SWA physical slot IDs while preserving identical row values remains
bitwise exact. Replacing only base `q` with candidate `q` does not move the
base output, but replacing only the base compressed cache with the candidate
compressed cache reproduces the same four-element BF16 output delta. The
remaining source is therefore compressed KV cache row drift before sparse
decode consumes it, not SWA physical mapping, verifier row selection, Q,
top-k, or logical SWA cache contents.
The two differing referenced compressed rows are slot `223` (block 3 offset
31, element 93 differs by `0.0001220703125`) and slot `226` (block 3 offset
34, element 357 differs by `1.52587890625e-05`). An opt-in diagnostic
`VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR=1` now slices both `save_partial_states`
and `compress_norm_rope_store_triton` rowwise for small speculative batches.
Focused unit coverage verifies rowwise slicing of `positions`, state
`slot_mapping`, `token_to_req_indices`, and compressed-KV `slot_mapping`, while
the default path still uses one launch. The TP=4 `MAX_TOKENS=40` diagnostic
gate remains RED with and without rowwise compressor slicing:
`tokens[24:30]=[223,20,850,304,270,4332]`. A focused layer-4 sparse recapture
under the rowwise compressor environment is bitwise identical to the previous
candidate sparse payload and retains the same compressed-slot `223`/`226`
drift versus base. The next discriminating boundary is the writer of those
compressed cache rows: capture compressed-KV writer inputs and before/after
cache rows for layer 4 slots `223` and `226`, including `positions`,
`slot_mapping`, `k_cache_metadata.slot_mapping`, `token_to_req_indices`,
`kv_score`, and `state_cache`.
That writer capture now exists behind
`VLLM_METAX_DSV4_COMPRESSOR_CAPTURE_DIR`, with optional layer, rank, head-dim,
physical-slot, and logical-position filters. The unit gate verifies physical
compressed-slot filtering, before/after cache-row capture, state-row metadata,
and the default non-capture path. The first generic-wrapper run used the wrong
prompt and produced no slot `223`/`226` captures, so it is not comparable to the
GSM8K RED harness. The GSM8K evaluator rerun with exact MHC/prefill flags,
rank `0`, head_dim `512`, and slots `223,226` captured exactly four BF16 sparse
compressed-cache writes:
position `639` -> physical slot `223` and position `651` -> physical slot
`226`, each with one changed writer call and one unchanged duplicate call.
However even this restricted capture changes the short diagnostic token window
to the frozen base window `[223,20,509,270,4332,4200]` instead of the current
RED window `[223,20,850,304,270,4332]`. Therefore the capture validates the
writer mapping and payload shape, but it is still perturbing evidence and
cannot be used to claim the RED mismatch is localized or fixed. The next probe
must reduce synchronization further, for example metadata-only first, or capture
device-side row hashes into preallocated buffers and copy them after generation.
After rechecking the current no-capture path, the active RED symptom returned
to the older index-108 branch rather than the layer-4 slot-drift branch:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/no_capture_exact_env_mt512_current/`
exits 0 with one Melanie GSM8K question, TP=4, k=1, eager, exact MHC/prefill
flags, and first mismatch `oracle[108]=28` versus candidate `1277`; candidate
length is 194 versus oracle length 212. This makes the current layer-4
compressed-cache slot capture secondary evidence only, because the unprobed
current path reaches a different first mismatch.
The latest single-variable probes around that current index-108 path are:

| diagnostic env added to exact MHC/prefill k=1 | first mismatch | effect |
| --- | ---: | --- |
| none | 108 (`28` vs `1277`) | current RED |
| `VLLM_METAX_DSV4_TOKENWISE_MHC_PRE=1` | 134 (`343` vs `1527`) | improves prefix; only useful isolated probe |
| `TOKENWISE_MHC_PRE=1` + `TOKENWISE_Q_ONLY=1` | 134 | neutral versus MHC-pre alone |
| `TOKENWISE_MHC_PRE=1` + `TOKENWISE_SPARSE_MLA_DECODE=1` | 134 | neutral |
| `TOKENWISE_MHC_PRE=1` + `TOKENWISE_QKV_INSERT=1` | 134 | neutral |
| `TOKENWISE_MHC_PRE=1` + `TOKENWISE_COMPRESSOR=1` | 134 | neutral |
| `TOKENWISE_MHC_PRE=1` + `TOKENWISE_FFN=1` | 26 (`509` vs `850`) | harmful; reintroduces early RED |
| `TOKENWISE_MHC_PRE=1` + `TOKENWISE_O_PROJ=1` | 26 (`509` vs `850`) | harmful; reintroduces early RED |
| `TOKENWISE_MHC_PRE=1` + target-only `TOKENWISE_TARGET_WQ_B=1` | 14 (`1527` vs `260`) | harmful; same regression as combined `TOKENWISE_WQ_B` |

Therefore the next boundary is the state produced after tokenwise initial MHC
pre but before the index-134 verifier token, not Q-only, sparse MLA decode,
Q/KV insert, or compressor. FFN and O-proj tokenwise substitutions remain
diagnostic regressions and must not be promoted.
Position-filtered layer captures for the index-134 boundary use logical
position `767`, inferred from the prior index-108 alignment
(`position 741 - token index 108 = prompt offset 633`). Candidate artifact
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index134_layer0_3_pos767_mhc_pre_candidate/`
preserves the RED at index 134 with `MAX_TOKENS=160`; matching MTP=0 base
artifact
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index134_layer0_3_pos767_base/`
matches the oracle prefix. At layer 0 rank 0, candidate calls for position
`767` have exact `before_attention` and `attention_inputs` versus base, but
`q_stages` first diverge (`raw_q` 6 BF16 elements, max `6.103515625e-05`;
`post_q` 5 BF16 elements, max `0.001953125`), followed by sparse attention
output and later MHC state divergence. Enabling Q-only replacement does not make
that stage exact, because the target `wq_b` GEMM remains batched; artifact
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index134_layer0_3_pos767_mhc_pre_q_only_candidate/`
preserves the same index-134 token mismatch and the same layer-0 q-stage
difference. A split diagnostic switch now separates target-attention
`VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B=1` from indexer
`VLLM_METAX_DSV4_TOKENWISE_INDEXER_WQ_B=1`, while preserving legacy combined
`VLLM_METAX_DSV4_TOKENWISE_WQ_B=1`. Target-only `wq_b` with tokenwise MHC pre
still regresses to index 14, and no indexer tokenwise marker appears in its
log, so the early regression is caused by target rowwise `wq_b` substitution
itself rather than the indexer half of the old combined switch.

An additional shadow-compare diagnostic now measures same-input batched target
`wq_b` versus rowwise target `wq_b` without replacing the live batched output.
It is gated by `VLLM_METAX_DSV4_WQ_B_SHADOW_COMPARE_DIR` and honors the existing
layer/rank/position filters before doing rowwise work. On the Melanie TP=4 k=1
eager exact-env run with tokenwise MHC pre, layer 0 rank 0 position `767`
captures kept the live stream at the MHC-pre RED boundary (first mismatch index
134, candidate length 210) and recorded small BF16 raw-Q drift:
`num_diff=7/9`, `max_abs=6.103515625e-05` for windows `[766,767]` and
`[767,768]`. Repeating the same shadow probe at inferred early-regression
position `647` also preserved first mismatch index 134, while recording
`num_diff=11/9`, `max_abs=0.000244140625` for windows `[646,647]` and
`[647,648]`. Therefore merely observing rowwise `wq_b` is not what causes the
index-14 regression; using rowwise target `wq_b` as the live attention Q is
sufficiently different from batched Q to remain a primary suspect. The next
diagnostic boundary should layer-bisect live target rowwise `wq_b` rather than
promoting the all-layer substitution.

The live target rowwise `wq_b` diagnostic now supports layer and position
selectors:
`VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_LAYERS` and
`VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS`. With tokenwise MHC pre, TP=4
k=1 eager exact-env Melanie runs show:

- layer `0` only regresses to first mismatch index 26 (`509` versus `850`),
  candidate length 204, so layer 0 alone is harmful but does not reproduce the
  all-layer target-rowwise index-14 branch;
- layer `0`, position `767` only preserves first mismatch index 134 and changes
  candidate length to 220, so local rowwise target `wq_b` at the observed
  layer-0 q-stage drift does not fix the index-134 token divergence.

This makes all current live rowwise target `wq_b` substitutions negative
diagnostics. The next boundary should compare the actual post-`wq_b`/post-qnorm
rows produced by the position-scoped live run against the MTP=0 base to learn
whether the rowwise local replacement made layer-0 Q exact but the mismatch
origin is already elsewhere, or whether native rowwise `wq_b` still fails to
match the base under live execution.

To shorten the Plan 08 quality feedback loop, a dedicated token gate runner now
wraps the existing GSM8K evaluator and direct frozen-oracle token comparison:

```bash
source .venv/bin/activate
source ./env.sh
tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/<name> \
  --env VLLM_METAX_DSV4_TOKENWISE_MHC_PRE=1
```

The runner selects the Melanie prompt from the frozen MTP=0 oracle, writes a
single-question fixture into the output directory, runs TP=4 k=1 eager exact-env
generation, writes `summary.json`, `run.log`, `token_gate_command.json`,
`comparison.json`, and exits `0` only for token-exact pass. A RED run exits `1`
and prints `first_mismatch`, oracle/candidate token counts, token windows, and
artifact paths. Smoke artifact
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/mhc_pre_smoke/`
correctly reproduced the current MHC-pre RED:
`first_mismatch=134`, oracle length 212, candidate length 210, oracle window
`[223,23,603,47593,28,343,24,90,565,223,20]`, candidate window
`[223,23,603,47593,28,1527,565,1527,17,21,565]`.

The runner now has an opt-in Plan08 debug loop that keeps the same TP=4 k=1
token verdict but automatically writes compact state-flow evidence:

```bash
tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/<name> \
  --plan08-debug-loop \
  --env VLLM_METAX_DSV4_TOKENWISE_MHC_PRE=1

tools/debug/summarize_deepseek_v4_plan08_debug_loop.py \
  .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/<name> \
  --output .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/<name>/debug_loop_summary.json
```

`--plan08-debug-loop` enables only opt-in diagnostics under that output root:
`mtp_capture/` for V1 proposer/verify/commit JSONL records,
`sparse_capture/` for layer-0 position-767 sparse MLA inputs, `qkv_capture/`
for position-767 QKV cache insertion, and `compressor_capture/` for the indexer
compressor write path when the selected layer/position/head-dim exists. The new
proposer records capture
`valid_sampled_tokens_count`, `num_rejected_tokens`, `token_indices_to_sample`,
positions, rejected/masked-token masks, query/seq metadata, block table, and
slot mapping, but not hidden states or full logits. The summarizer reports the
token-gate verdict, capture completeness, stage counts, and any direct evidence
that a rejected token still has a non-padding slot. A non-padding rejected slot
makes the next focus `audit_rejected_token_slot_mapping_and_cache_writes`;
otherwise the next boundary remains capture diff or a narrower state-flow probe.
Focused validation of this loop passed on 2026-07-19:
`pytest -q tests/tools/test_run_deepseek_v4_plan08_token_gate.py
tests/tools/test_summarize_deepseek_v4_plan08_debug_loop.py
tests/models/deepseek_v4/test_mtp_debug.py` (`23 passed`), plus `ruff check`
and `python -m py_compile` on the changed scripts/helpers.

Fresh TP=4 k=1 debug-loop runs on 2026-07-19 reproduced the same MHC-pre RED:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/plan08_debug_loop_mhc_pre_20260719_174104/`
and
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/plan08_debug_loop_compressor128_mhc_pre_20260719_180247/`
both failed at token index 134 (`343` versus `1527`, 212 versus 210 tokens).
The proposer/verify/commit capture shows no rejected draft was committed and no
rejected token retained a non-padding proposer slot. At the critical window,
draft `34593` was rejected and target `47593` was committed; the later
divergence occurs because the target verifier itself predicts/accepts `1527`
where the frozen base predicts `343`.

The matching MTP=0 debug-loop base run
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/plan08_debug_loop_base_qkv_sparse_20260719_181038/`
passed token exact and captured layer-0 position 767. Comparing it to the k=1
candidate shows:

- base position 767 writes SWA slot `16319` (block 254 offset 63), while k=1
  writes slot `14655` (block 228 offset 63);
- the candidate SWA block table is `[... 180, 228, 259]` versus base
  `[..., 181, 254]`, and `swa_indices` shift from `11584..` to `11520..`;
- the QKV capture's `kv` vector for position 767 is exact, but the flattened
  SWA cache row after insert differs in 8 BF16 elements (`max_abs=0.00048828125`);
- a position 766/767 candidate probe
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/plan08_debug_loop_qkv766_767_mhc_pre_20260719_181710/`
  confirms the rejected draft row at position 766 is overwritten by the
  correction row (`1023` elements changed, `max_abs=4.125`), so the current
  evidence is not "rejected KV was never overwritten."

Therefore the active boundary is narrower: the MetaX flattened
`fused_deepseek_v4_qnorm_rope_kv_rope_insert`/SWA row reuse path can leave a
small non-exact full cache row even when the captured `kv` input vector is exact.
The environment exposes only `_C.fused_deepseek_v4_qnorm_rope_kv_rope_insert`
and `_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`; upstream
`full_cache_bf16_insert` is not available, so a native differential or a
targeted full-row overwrite/zero-fill kernel gate is required before any product
fix can be claimed.

A companion offline q-stage diff utility now compares captured `raw_q` and
post-qnorm/RoPE `post_q` rows by logical position:

```bash
tools/debug/diff_deepseek_v4_q_stages.py \
  --base <base-capture-dir> \
  --candidate <candidate-capture-dir> \
  --position 767 --layer 0 --rank 0 \
  --output <q-stage-diff.json>
```

It reproduced the earlier MHC-pre layer-0 position-767 drift in
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index134_layer0_3_pos767_mhc_pre_candidate/q_stage_diff_layer0_pos767.json`
(`raw_q_exact_rows=0`, `post_q_exact_rows=0`). Running the new token gate with
position-scoped live target rowwise `wq_b` plus layer capture:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/live_target_wq_b_layer0_pos767_qstage/`
kept the token RED at index 134, but q-stage diff showed both candidate verifier
windows for layer 0 position `767` are now exact versus MTP=0 base
(`raw_q_exact_rows=2`, `post_q_exact_rows=2`). The same run's layer-0 stage
diff shows `attention_inputs.qr` and `attention_inputs.kv` are exact, while
`attention_output.output` first differs (`num_diff=9`,
`max_abs=0.0009765625`), then `after_attention.hidden_states` differs
(`num_diff=271`, `max_abs=0.03125`). Therefore the active boundary moved past
target `wq_b`/qnorm/RoPE for this row and into sparse MLA attention output.

Combining the same position-scoped live target rowwise `wq_b` with
`VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1` did not move the RED:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/live_wq_b_pos767_sparse_tokenwise_qstage/`
still has first mismatch index 134, candidate length 220, exact
`attention_inputs`/`q_stages`, and the same layer-0 attention output drift
(`num_diff=9`, `max_abs=0.0009765625`). This rules out simple q_len=2 sparse MLA
batching as the only cause at this boundary; the next probe should compare the
sparse MLA decode capture inputs for layer 0 position `767`, especially selected
SWA indices, top-k indices/lens, compressed/SWA cache rows, and metadata mapping.

That sparse decode pair loop is now available through the current schema-aware
`tools/debug/diff_deepseek_v4_sparse_mla.py --base ... --candidate ...` mode.
A fresh MTP=0 base run with layer-0/rank-0/position-767 sparse capture passed
the frozen token gate (`212/212` tokens, finish reason exact) and wrote:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_pair_base_pos767_20260719/`.
A matching k=1 candidate run with `TOKENWISE_MHC_PRE=1` and position-scoped live
target `wq_b` preserved the expected RED (`first_mismatch=134`, candidate
length 220) and wrote:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_pair_candidate_pos767_20260719/`.
Pair diff artifacts:

- `sparse_pair_diff_base_call0_vs_candidate_call0.json`
- `sparse_pair_diff_base_call0_vs_candidate_call1.json`

Both candidate verifier windows have exact layer-0 Q (`q num_diff=0`) and the
same logical position `[767]`, but both first differ in sparse MLA attention
inputs at `swa_indices`. The base SWA index row starts at
`[11584,11585,11586,11587,11588,11589,11590,11591]`; both candidate windows
start at `[11520,11521,11522,11523,11524,11525,11526,11527]`. `swa_lens` stays
exact at `128`, while `swa_block_table` differs in three entries and the full
captured `swa_cache` shape differs (`8436x64x512` base versus `6569x64x512`
candidate). The resulting attention output drift remains small but real
(`num_diff=9`, `max_abs=0.0009765625`). Therefore the next fix boundary is not
target `wq_b`/qnorm/RoPE or q_len batching; it is the MTP verifier's SWA
physical slot/block-table mapping for sparse MLA decode.

2026-07-19 feedback-loop update: a scoped target QKV probe was added with
`VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV=1`,
`VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_LAYERS`, and
`VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS`. The first live run failed
before token comparison because the diagnostic QKV slice returned a
non-contiguous `kv` tensor and the native Q/KV insert op requires contiguous
CUDA input. After making the diagnostic `qr` and `kv` slices contiguous, the
same TP=4 k=1 eager Melanie token gate passed with exact greedy tokens:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_insert_candidate_scoped_target_qkv_pos767_20260719_retry2/`
(`decision=pass`, `first_mismatch=null`, oracle/candidate token counts
`212/212`, finish reason `stop/stop`, elapsed `346.98s`). This result is now
known to be a raw-QKV compensation artifact, not valid localization evidence:
the probe ran at the `attention_impl` boundary, where normal `qr/kv` inputs have
already passed `q_norm/kv_norm`.

This is not a Plan08 acceptance pass and does not enable k=4/performance work:
it used diagnostic scoped env flags (`TOKENWISE_MHC_PRE=1`, target QKV layer
`0` position `767`, and target `wq_b` layer `0` position `767`) under eager
mode. The diagnostic helper was corrected to return normalized contiguous
`qr/kv`; with that fix, the same scoped target QKV + target `wq_b` gate is RED
again (`first_mismatch=134`, candidate length `210`) at:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_insert_candidate_scoped_target_qkv_normed_pos767_20260719/`.
This preserves the useful feedback loop but removes the false green.

Follow-up evidence also prevents overclaiming: the selected historical QKV
insert rows at positions
`658,674,685,696,706,711,722,734,738,749,760,763` still show the same
`kv[0,86]` BF16 drift (`base=-0.055908203125`,
`candidate=-0.055419921875`), and sparse pair diffs against the MTP=0 base still
show attention-input differences when comparing individual captured sparse
calls. A more correct pre-norm target QKV diagnostic was then added through
`VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_PRENORM=1`; scoped only to layer 0
position `767` with target `wq_b` still preserves the index-134 RED
(`candidate_prenorm_target_qkv_pos767_wqb767_20260719/`), while applying it to
the historical drift positions makes the run worse (`first_mismatch=31`,
`candidate_prenorm_target_qkv_histpos_wqb767_20260719/`). A KV-only variant
(`VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM=1`) also regresses to
`first_mismatch=26`
(`candidate_prenorm_target_kv_histpos_wqb767_20260719/`). Therefore QKV/KV
rowwise rewriting is currently a negative diagnostic, and the active fix
boundary remains SWA/KV-cache metadata or an upstream MTP verifier state-flow
issue rather than a simple target QKV product change.

2026-07-19 efficient feedback-loop update: the debug loop now has a single
orchestrator:

```bash
source .venv/bin/activate
source ./env.sh
tools/debug/run_deepseek_v4_plan08_feedback_loop.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/<name> \
  -- --env VLLM_METAX_DSV4_TOKENWISE_MHC_PRE=1
```

For already captured runs, use the seconds-level replay/summarize mode:

```bash
tools/debug/run_deepseek_v4_plan08_feedback_loop.py \
  --output-root <existing-debug-loop-artifact> \
  --summarize-existing
```

The orchestrator runs the TP=4 k=1 token gate with `--plan08-debug-loop`, always
writes `debug_loop_summary.json`, diffs
`sparse_capture/rank0_call2.pt` against the frozen MTP=0 debug-loop base, writes
`sparse_diff.json`, and emits `verdict.json` with a single `decision` plus
`next_action`. This makes each MTP hypothesis a one-command RED/GREEN loop
instead of a manual chain of token comparison, capture summary, sparse pair diff,
and handwritten conclusion. Focused validation passed:
`pytest -q tests/tools/test_run_deepseek_v4_plan08_feedback_loop.py
tests/tools/test_run_deepseek_v4_plan08_token_gate.py
tests/tools/test_summarize_deepseek_v4_plan08_debug_loop.py` (`12 passed`),
plus `ruff check` and `python -m py_compile` for the new script.

The inherited tokenwise QKV insert probe finished at
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/plan08_debug_loop_tokenwise_qkv_insert_mhc_pre_20260719_183301/`.
The new orchestrator summarized it and wrote `verdict.json`; the token gate is
still RED at first mismatch `134` (`343` versus `1527`, `212/210` tokens). Its
`sparse_diff.json` still shows non-exact layer-0 position-767 sparse attention
inputs: `q` differs in 5 BF16 elements (`max_abs=0.001953125`) and
`gathered_swa_rows` differs in 12 elements (`max_abs=0.00048828125`). Therefore
`VLLM_METAX_DSV4_TOKENWISE_QKV_INSERT=1` does not remove the historical gathered
SWA row drift; the current verdict is
`next_action=continue_sparse_swa_row_producer_diagnosis`. k=1 remains RED, so
k=4 and performance work remain blocked.

2026-07-19 sampler-boundary diagnostic update: a default-off diagnostic switch
`VLLM_METAX_MTP_FORCE_REJECT_DRAFTS=1` was added to the V1 greedy rejection
sampler wrapper. It forces sampler output to the target first-row argmax
correction and drops all draft/bonus columns, while leaving the verifier model
forward unchanged. Unit coverage is in
`tests/patch/bugfix/test_dsv4_drop_bonus.py`.

Two live TP=4 k=1 feedback-loop runs were used to validate the probe. The first
run exposed that this V1 path passes `cu_num_draft_tokens` as per-request counts
(`feedback_force_reject_drafts_20260719_185613/`, runtime error before verdict);
the helper was corrected to support both prefix and count forms. The corrected
run
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/feedback_force_reject_drafts_counts_20260719_191452/`
still failed at first mismatch `134` (`343` versus `1527`, `212/210` tokens)
with the same sparse diff signature: `q` differs in 5 BF16 elements and
`gathered_swa_rows` differs in 12 elements. Its verifier capture still reports
many `accepted_count=1` rows because the debug capture derives
`accepted_count` from target-vs-draft equality, not from the forced sampled
output; the committed token IDs remain the target argmax when target equals the
draft. Therefore the sampler/commit formatting layer is not the active root
cause. The current fix boundary is earlier: q_len=2 target verifier forward
state for accepted/correction rows must become逐 token exact with q_len=1 base
before k=1 can pass.

2026-07-19 feedback-loop closure: the layer-stage diff helper is now part of
the fast validation loop:

```bash
tools/debug/diff_deepseek_v4_layer_stages.py \
  --base <base-layer-capture-dir> \
  --candidate <candidate-layer-capture-dir> \
  --position 767 --layer 0 --stage before_attention \
  --output <candidate-artifact>/layer0_pos767_before_attention_diff.json
```

The helper compares full captured tensors by logical position, so it avoids the
earlier false comparison against only the last captured row. Focused validation
passed with the current environment:
`pytest -q tests/tools/test_run_deepseek_v4_plan08_feedback_loop.py
tests/tools/test_diff_deepseek_v4_layer_stages.py
tests/tools/test_diff_deepseek_v4_q_stages.py
tests/patch/bugfix/test_dsv4_drop_bonus.py` (`12 passed`).

Fresh TP=4 captures close the current engineering loop. The MTP=0 base artifact
is
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/layer0_pos767_base_full_20260719_193010/`
and passes the frozen greedy oracle (`212/212` tokens). The matching k=1
candidate artifact is
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/layer0_pos767_k1_full_20260719_193609/`
and remains RED at first mismatch `134` (`343` versus `1527`, `212/210`
tokens). The stage diffs show:

- `before_attention`: 2 candidate rows, both exact.
- `after_attention`: 2 candidate rows, both non-exact; first differing tensor is
  `hidden_states`, with 419 BF16 element differences and `max_abs=0.03125`.
- `after_ffn`: 2 candidate rows, both non-exact; drift propagates through
  `hidden_states`, `pre_norm`, `residual`, `post_mix`, and `res_mix`.

This completes the efficient RED feedback loop for the current blocker: the
next debug boundary is layer-0 attention/QKV/sparse execution for q_len=2
target verification versus q_len=1 base. Sampler/commit formatting and
pre-layer0 hidden-state flow are now lower-ranked hypotheses. k=1 remains RED;
do not advance to k=4 or performance measurement.

2026-07-19 continuation: the feedback loop was rerun with the combined
Plan08 debug-loop captures plus full layer-0 tensors:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/feedback_attention_boundary_layer0_pos767_20260719_194912/`.
It remains a stable k=1 RED at first mismatch `134` (`343` versus `1527`,
`212/210` tokens) and writes `comparison.json`, `debug_loop_summary.json`,
`sparse_diff.json`, `verdict.json`, `mtp_capture/`, `sparse_capture/`,
`qkv_capture/`, and `layer_capture/` in one artifact.

Additional seconds-level diffs were generated from that artifact:

- `layer0_pos767_attention_inputs_diff.json`: `qr` and `kv` are exact for both
  candidate verifier rows.
- `layer0_pos767_q_stage_diff.json`: without the scoped target-`wq_b`
  diagnostic, `raw_q` differs in 6 BF16 elements (`max_abs=6.103515625e-05`)
  and `post_q` differs in 5 elements (`max_abs=0.001953125`).
- `layer0_pos767_qkv_insert_diff.json`: comparing whole cache blocks is
  misleading because base and candidate use different physical blocks. Extracting
  the actual `cache_slot_offsets * head_dim` row shows the written KV row is
  exact, while `kv` is exact and `q` carries the same small drift.
- `layer0_pos767_attention_output_diff.json`: `output` first differs in 5 BF16
  elements (`max_abs=0.0009765625`).

The sparse diff tool now records first tensor-difference coordinates and values,
so verdicts can point directly at the next data row instead of only reporting a
count. For this artifact, `sparse_diff.json` reports:

- `q`: first diff at `[0, 4, 171]`, base `-0.06591796875`, candidate
  `-0.0654296875`, `num_diff=5`, `max_abs=0.001953125`.
- `swa_indices`: first diff at `[0, 0, 0]`, base `11584`, candidate `11520`;
  all 128 entries differ because the physical block table maps the same logical
  SWA window to different physical slots.
- `swa_block_table`: first diff at `[0, 10]`, base block `181`, candidate block
  `180`; block-table differences are `[10]`, `[11]`, and candidate-only future
  block `[12]`.
- `gathered_swa_rows`: first content diff at `[18, 86]`, base
  `-0.055908203125`, candidate `-0.055419921875`, `num_diff=12`,
  `max_abs=0.00048828125`.
  This row maps to historical logical position `658` in the position-767 SWA
  window: base slot `11602` (`block=181`, `offset=18`) versus candidate slot
  `11538` (`block=180`, `offset=18`).

Interpretation: the current combined loop proves that layer-0 pre-attention
inputs and current KV-row insertion are not enough to explain the token
divergence. The remaining active boundary is historical SWA cache row content
and physical block-table/slot trajectory under MTP verifier state flow. The
next useful probe is to capture the first accepted historical position whose
written SWA row later becomes `gathered_swa_rows[18]` at position `767`, then
compare its QKV insert row against the MTP=0 base.

2026-07-19 position-658 writeback closure: a dedicated offline row diff tool
now compares Q/KV-insert captures by logical position while extracting the
actual written cache row via `cache_slot_offsets * head_dim`, instead of
comparing whole physical cache blocks:

```bash
tools/debug/diff_deepseek_v4_qkv_insert_rows.py \
  --base <base-qkv-capture-dir> \
  --candidate <candidate-qkv-capture-dir> \
  --position 658 --layer 0 \
  --output <candidate-artifact>/layer0_pos658_qkv_insert_row_diff.json
```

Focused unit coverage is in
`tests/tools/test_diff_deepseek_v4_qkv_insert_rows.py`; it covers the case where
base and candidate use different physical blocks but the actual written row is
the comparable object.

Existing artifacts were sufficient, so no new model run was required:

- base:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_insert_base_pos658_763_20260719/qkv_capture/`
- candidate:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_insert_candidate_pos658_763_20260719/qkv_capture/`
- diff:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_insert_candidate_pos658_763_20260719/layer0_pos658_qkv_insert_row_diff.json`

For layer 0 logical position `658`, the selected candidate rows both differ
from the MTP=0 base before the cache write:

- `q`: 4 BF16 element differences, `max_abs=0.0001220703125`, first diff
  `[0,4,30]`.
- `kv`: 1 BF16 element difference, `max_abs=0.00048828125`, first diff
  `[0,86]`, base `-0.055908203125`, candidate `-0.055419921875`.
- `cache_after_row`: exactly the same one-element difference at `[86]`.
- physical slots differ by one SWA block, base slot `11602` versus candidate
  slot `11538`, but `cache_slot_offsets` are both `18`.

This proves the later position-767 `gathered_swa_rows[18,86]` drift is already
present at the historical position-658 Q/KV insert input and is faithfully
written into the candidate SWA cache. Therefore the immediate root boundary is
not the sparse MLA gather kernel and not a whole-block comparison artifact; it
is the producer of layer-0 position-658 `q/kv` for MTP verifier state.

The existing diagnostic variants were also diffed with the same tool. Plain
`TOKENWISE_QKV_INSERT` leaves the same `kv/cache_after_row` drift. Broader
`VLLM_METAX_DSV4_TOKENWISE_QKV=1` removes the position-658 `kv` and
`cache_after_row` drift, but the token gate becomes worse (`first_mismatch=26`,
candidate length `197`), so it remains a negative diagnostic rather than a
valid product fix. The next useful probe is a narrower layer-0 position-658
QKV-producer capture: compare the normalized hidden input, raw `wq_a/wkv_a`
outputs, and post-norm `qr/kv` for the same logical position under q_len=2
verifier versus q_len=1 base.

2026-07-19 QKV-producer boundary closure: the layer debug capture now includes
an opt-in `qkv_producer` stage immediately after `fused_q_kv_rmsnorm` and before
the attention implementation. It records the normalized `hidden_states`, raw
`qr_kv` split as `qr_pre_norm`/`kv_pre_norm`, and post-norm `qr`/`kv` rows using
the existing layer/rank/position capture filters, so it is inert unless
`VLLM_METAX_DSV4_LAYER_CAPTURE_DIR` is set.

Focused validation for the engineering loop passed:

```bash
pytest -q \
  tests/models/deepseek_v4/test_layer_debug.py::test_qkv_producer_capture_saves_pre_and_post_norm_rows \
  tests/tools/test_diff_deepseek_v4_layer_stages.py \
  tests/tools/test_diff_deepseek_v4_qkv_insert_rows.py

python -m py_compile \
  vllm_metax/models/deepseek_v4/layer_debug.py \
  vllm_metax/models/deepseek_v4/attention.py \
  tools/debug/diff_deepseek_v4_qkv_insert_rows.py \
  tools/debug/diff_deepseek_v4_layer_stages.py \
  tools/debug/run_deepseek_v4_plan08_feedback_loop.py \
  tools/debug/run_deepseek_v4_plan08_token_gate.py

git diff --check -- <touched feedback-loop files>
markdownlint-cli2 docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md
```

Results: `5 passed`, `py_compile` passed, `git diff --check` passed, and
Markdown lint reported `0 error(s)`.

Fresh TP=4 position-658 producer captures:

- base MTP=0:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_producer_base_pos658_20260719_202938/`
- candidate k=1 with `VLLM_METAX_DSV4_TOKENWISE_MHC_PRE=1`:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_producer_candidate_pos658_20260719_203654/`
- diff:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_producer_candidate_pos658_20260719_203654/layer0_pos658_qkv_producer_diff.json`

The base gate passes (`212/212` tokens). The candidate remains the same stable
k=1 RED at first mismatch `134`: oracle token `343`, candidate token `1527`,
lengths `212/210`. The producer-stage diff for layer 0 logical position `658`
shows two candidate verifier rows and neither row is exact, but the boundary is
now much narrower:

- `hidden_states`: exact, `num_diff=0`.
- `qr_pre_norm`: exact, `num_diff=0`.
- `qr`: exact, `num_diff=0`.
- `qr_kv`: 1 BF16 element differs, `max_abs=0.00006103515625`.
- `kv_pre_norm`: same 1 BF16 element differs, `max_abs=0.00006103515625`.
- `kv`: 1 BF16 element differs after RMSNorm, `max_abs=0.00048828125`.

Interpretation: the historical position-658 SWA row drift is not caused by
pre-layer0 hidden state, not by `qr`, and not by sparse gather. The next active
root boundary is the KV half of the fused QKV producer for q_len=2 verifier
execution versus q_len=1 base, before KV RMSNorm. The following fix/debug loop
should target batch-invariance of the `wkv_a`/`qr_kv` production path for
layer-0 position 658, with this command pair and diff as the red/green gate.
k=1 remains RED; do not proceed to k=4 or performance measurement.

2026-07-19 selected-row diagnostic update: the target rowwise diagnostics were
tightened so a position-scoped probe no longer rewrites the whole q_len=2
verifier batch. The following envs now preserve unselected rows and replace only
matching logical positions:

- `VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM=1` with
  `VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_POSITIONS=<positions>`.
- `VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B=1` with
  `VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B_POSITIONS=<positions>`.
- `VLLM_METAX_DSV4_TOKENWISE_O_PROJ=1` with
  `VLLM_METAX_DSV4_TOKENWISE_O_PROJ_POSITIONS=<positions>`.
- `VLLM_METAX_DSV4_TOKENWISE_FFN=1` with
  `VLLM_METAX_DSV4_TOKENWISE_FFN_POSITIONS=<positions>`.

Focused TDD/validation passed:

```bash
pytest -q \
  tests/models/deepseek_v4/test_mhc_tokenwise.py \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py \
  tests/models/deepseek_v4/test_o_proj_diff.py \
  tests/models/deepseek_v4/test_ffn_debug.py \
  tests/models/deepseek_v4/test_layer_debug.py::test_qkv_producer_capture_saves_pre_and_post_norm_rows \
  tests/tools/test_diff_deepseek_v4_layer_stages.py \
  tests/tools/test_diff_deepseek_v4_qkv_insert_rows.py \
  tests/tools/test_run_deepseek_v4_plan08_feedback_loop.py
```

Result: `71 passed`. `py_compile` and `git diff --check` also passed for the
edited model/debug files.

TP=4 evidence:

- Selective `KV@658` only:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/scoped_kv_prenorm_pos658_selective_20260719_205412/`
  remains RED and regresses to first mismatch `26` (`509/850`, `212/195`).
  However
  `layer0_pos658_qkv_producer_diff.json` shows position 658 is locally exact
  through `qkv_producer`.
- Selective `KV@658`, capture position 659:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/scoped_kv_prenorm_pos658_capture_pos659_20260719_210613/`
  remains RED at first mismatch `26`. Position 659 is exact through
  `qkv_producer`, but `q_stages.raw_q/post_q` differ.
- Selective `KV@658 + WQ_B@659`:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/selective_kv658_wqb659_20260719_211800/`
  remains RED at first mismatch `26`, but position 659 advances to exact through
  `q_stages` and `attention_output`; the first layer-stage difference moves to
  `after_attention`, i.e. the attention output projection boundary.
- Selective `KV@658 + WQ_B@659 + O_PROJ@659`:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/selective_kv658_wqb659_oproj659_20260719_213033/`
  is the best current diagnostic: first mismatch moves later to `156`
  (`34593/47593`, `212/220`). For position 659, all layer-0 stages through
  `after_attention` are exact; `after_ffn` has one BF16 difference
  (`max_abs=0.001953125`).
- Adding selective `FFN@659`:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/selective_kv658_wqb659_oproj659_ffn659_20260719_214240/`
  makes position 659 exact through `after_ffn`, but the global token gate
  regresses to first mismatch `134` (`343/1527`, `212/210`). Treat this as a
  negative diagnostic, not a candidate product fix.

Interpretation: selected rowwise replacement proves a chain of shape-sensitive
native projection differences under q_len=2 verifier execution: `KV@658` feeds
the later SWA row drift, `WQ_B@659` fixes the first Q/RoPE boundary, and
`O_PROJ@659` fixes the position-659 attention return boundary enough to move the
global RED from index `26` to `156`. The next useful probe should use the
three-point best diagnostic (`KV@658 + WQ_B@659 + O_PROJ@659`) and capture the
new mismatch window around logical position `789` (`prompt offset 633 + token
index 156`) instead of adding `FFN@659`.

2026-07-19 position-789 continuation: the new best RED window was captured with
the three-point diagnostic preserved. The fresh MTP=0 base artifact is
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_producer_base_pos789_20260719_215235/`
and passes the token gate (`212/212`). The matching candidate artifact is
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/selective_kv658_wqb659_oproj659_capture_pos789_20260719_215807/`
and reproduces the best RED at first mismatch `156`
(`34593/47593`, `212/220`).

Layer-0 position-789 diffs from that pair:

- `before_attention`: exact.
- `attention_inputs`: exact.
- `qkv_producer`: exact.
- `q_stages`: first difference `post_q`; `raw_q` differs in 2 elements with
  `max_abs=2.384185791015625e-07`, and `post_q` differs in 2 elements with
  `max_abs=7.62939453125e-06`.
- `attention_output`: non-exact, `output` differs in 85 BF16 elements with
  `max_abs=0.00390625`.
- `after_attention` and `after_ffn`: non-exact through `hidden_states`.

A follow-up selected-row gate extended `WQ_B` from position `659` to
`659,789` while keeping `KV@658` and `O_PROJ@659`:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/selective_kv658_wqb659_789_oproj659_20260719_220517/`.
It still fails at first mismatch `156` (`34593/47593`, `212/220`), but the
position-789 `q_stages` diff becomes exact. The first remaining layer-0
position-789 difference is therefore `attention_output`, with 85 BF16 output
differences and `max_abs=0.00390625`.

Interpretation: `WQ_B@789` is a useful local probe but does not move the global
token gate. The current active boundary under the best expanded diagnostic
(`KV@658 + WQ_B@659,789 + O_PROJ@659`) is the layer-0 sparse MLA attention
output for position 789. The next useful probe is a sparse MLA pair capture for
rank 0/layer 0/position 789 under the same expanded diagnostic, comparing `q`,
indices/block tables, gathered SWA/compressed rows, and output. Do not add
`FFN@659` back into the candidate path.

2026-07-19 sparse position-789 boundary: a fresh sparse MLA pair capture was
collected for rank 0/layer 0/position 789.

- base:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_base_pos789_20260719_221518/`
  (`rank0_call2.pt`, decode `swa`, position `[789]`, token index `[0]`, token
  gate pass `212/212`).
- candidate with expanded diagnostic `KV@658 + WQ_B@659,789 + O_PROJ@659`:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_candidate_pos789_expanded_20260719_222037/`
  (`rank0_call2.pt` and `rank0_call3.pt`, both decode `swa`, position `[789]`,
  token gate RED at first mismatch `156`, `34593/47593`, `212/220`).
- pair diffs:
  `sparse_diff_call2.json` and `sparse_diff_call3.json` in the candidate
  artifact.

The sparse pair proves:

- `q` is exact for both candidate rows (`num_diff=0`).
- first attention-input difference is `swa_indices`, not Q.
- `swa_indices` differs in 106 entries; first index maps base slot `11606` to
  candidate slot `11542`.
- `swa_block_table` differs at `[0,10]`: base block `181`, candidate block
  `180`; there are 2 differing block-table entries.
- `gathered_swa_rows` still has the familiar BF16 drift at column 86. For the
  expanded candidate, row 12 maps to logical position 674 in the position-789
  SWA window (`789 - 127 + 12 = 674`), with base `-0.055908203125` versus
  candidate `-0.055419921875`, `max_abs=0.00048828125`, and 15 differing
  gathered elements.
- `output` differs in 85 BF16 elements with `max_abs=0.00390625`.

A follow-up gate added `KV@674` to the selected KV-prenorm diagnostic:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/selective_kv658_674_wqb659_789_oproj659_20260719_222920/`.
This is a negative diagnostic: the token gate regresses to first mismatch `108`
(`1527/1277`, `212/229`). Its sparse pair still has `q` exact but keeps
`swa_indices`/`swa_block_table` differences and makes sparse output much worse
(`output` 31,910 BF16 diffs, `max_abs=0.6815185546875`; `gathered_swa_rows`
17,899 diffs, `max_abs=3.021484375`). Therefore individual historical KV-row
rewrites are not a safe cumulative product fix.

Interpretation: the current active root boundary is SWA cache metadata/slot
trajectory under the q_len=2 verifier, not the sparse MLA compute kernel and
not Q. The next useful probe should compare the SWA block-table construction
and slot mapping for logical positions 674 and 789 under MTP=0 versus the
expanded k=1 verifier path, rather than adding more rowwise KV fixes.

2026-07-19 efficient feedback-loop completion: the Plan08 feedback-loop
orchestrator now promotes sparse attention-input metadata differences into the
machine-readable verdict instead of leaving them buried in pair-diff JSON. The
seconds-level replay command:

```bash
tools/debug/run_deepseek_v4_plan08_feedback_loop.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_candidate_pos789_expanded_20260719_222037 \
  --summarize-existing \
  --base-output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_base_pos789_20260719_221518 \
  --base-sparse-call sparse_capture/rank0_call2.pt \
  --candidate-sparse-call sparse_capture/rank0_call2.pt
```

writes
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_candidate_pos789_expanded_20260719_222037/verdict.json`
and exits RED with `decision=fail`,
`next_action=audit_swa_metadata_slot_mapping`. The verdict records token
mismatch `156` (`34593/47593`, `212/220` tokens), `q` exact, first sparse
attention-input difference `swa_indices`, `swa_indices` first slot
`11606/11542`, `swa_block_table[0,10]` `181/180`, and the downstream
`gathered_swa_rows`/`output` BF16 drifts. This completes the intended fast
validation feedback engineering loop for the current blocker: each hypothesis
can now be tested by one command that emits a red/green token verdict plus the
next diagnostic boundary. k=1 remains RED; do not start k=4 or performance
gates.

2026-07-19 SWA mapping analyzer update: a companion offline analyzer now
explains each differing SWA global slot as
`logical_position -> logical_block -> block_table value -> physical slot`, then
compares the gathered cache row contents by logical position:

```bash
python tools/debug/analyze_deepseek_v4_swa_mapping_pair.py \
  --base .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_base_pos789_20260719_221518/sparse_capture/rank0_call2.pt \
  --candidate .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_candidate_pos789_expanded_20260719_222037/sparse_capture/rank0_call2.pt \
  --json-out .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_candidate_pos789_expanded_20260719_222037/swa_mapping_call2_analysis.json
```

The analysis shows both base and candidate SWA indices are internally
consistent with their own block tables: the first differing row maps logical
position `662`, logical block `10`, base slot `11606` from physical block
`181`, and candidate slot `11542` from physical block `180`; both
`slot_matches_block_table=true`. Candidate `rank0_call3.pt` has the same
mapping pattern. Therefore the physical block-table difference is not by itself
proof of a bad metadata formula; the remaining semantic difference is cache-row
content. The first content drift is logical position `674`, base slot `11618`
versus candidate slot `11554`, one BF16 element at column `86`
(`-0.055908203125` versus `-0.055419921875`, `max_abs=0.00048828125`), with 15
logical rows differing overall. The feedback-loop verdict now uses this
analysis and reports `next_action=compare_swa_cache_rows_by_logical_position`
when Q is exact, SWA physical indices differ only through internally consistent
block tables, and gathered SWA rows differ.

2026-07-19 position-674 producer boundary: existing QKV insert captures for the
historical drift positions already cover logical position `674`. Diffing
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_insert_base_pos658_763_20260719/qkv_capture`
against
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_insert_candidate_pos658_763_20260719/qkv_capture`
at layer 0/rank 0/position 674 writes
`qkv_insert_candidate_pos658_763_20260719/layer0_pos674_qkv_insert_row_diff.json`.
The first candidate row has exact `cache_before_row`, but `kv` and
`cache_after_row` both differ at column `86` by the same BF16 amount
(`-0.055908203125` versus `-0.055419921875`,
`max_abs=0.00048828125`). The insert slot differs by one physical block
(`11618` versus `11554`) but the cache row written exactly reflects the input
`kv`. Therefore the qnorm/RoPE/KV insert kernel is not the first producer of
the logical position-674 content drift.

Fresh matched qkv-producer captures were then collected:

- base:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_producer_base_pos674_20260719/`
  (`num_speculative_tokens=0`, token exact pass `212/212`);
- candidate:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_producer_candidate_pos674_expanded_20260719/`
  (`KV@658 + WQ_B@659,789 + O_PROJ@659`, token gate RED at mismatch `156`,
  `34593/47593`, `212/220`).

`tools/debug/diff_deepseek_v4_layer_stages.py` now includes first tensor
difference index/value in its JSON output. The layer-0/rank-0/position-674
`qkv_producer` diff writes
`qkv_producer_candidate_pos674_expanded_20260719/layer0_pos674_qkv_producer_diff.json`
and shows both candidate verifier rows have exact `hidden_states`,
`qr_pre_norm`, and `qr`, while `kv_pre_norm` differs in exactly one BF16 element
at `[0,86]` (`-0.0084228515625` versus `-0.00836181640625`,
`max_abs=0.00006103515625`). After KV RMSNorm this becomes the same `kv`
content drift at `[0,86]` (`-0.055908203125` versus `-0.055419921875`,
`max_abs=0.00048828125`) later observed in QKV insert and sparse gathered SWA
rows. The active boundary is now the target fused QKV projection's KV half under
q_len=2 verifier batching, not SWA block-table construction, sparse MLA compute,
or QKV cache insertion. Live rowwise KV replacements at historical positions
remain negative diagnostics and must not be promoted without a gate that
preserves the token sequence.

2026-07-19 pre-norm QKV shadow confirmation: a default-off diagnostic
`VLLM_METAX_DSV4_QKV_PRENORM_SHADOW_COMPARE_DIR` now compares the live batched
`fused_wqa_wkv` pre-norm `qr_kv` output with same-input rowwise projection for
selected layers/positions, without replacing the live tensor. The TP=4 k=1
candidate run
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/qkv_prenorm_shadow_pos674_expanded_20260719/`
kept the same RED at mismatch `156` (`34593/47593`, `212/220`), so the probe is
non-mutating for this boundary. Its shadow artifact
`qkv_prenorm_shadow/rank0_layer0_call0_qkv_prenorm_shadow_compare.pt` captures
positions `[674,675]` and reports:

- `qr`: exact (`num_diff=0`);
- `qr_kv`: one BF16 difference at `[0,1110]`, `max_abs=0.00006103515625`,
  batched `-0.00836181640625` versus rowwise `-0.0084228515625`;
- `kv`: the same one BF16 difference at `[0,86]`, batched
  `-0.00836181640625` versus rowwise `-0.0084228515625`.

This independently confirms the qkv-producer diff: the q_len=2 batched target
projection is the first observed source of the logical position-674 KV row
drift, while the QR half remains exact. The next gate should test a narrowly
scoped, non-regressing way to make target KV projection deterministic/exact for
q_len=2 verifier batches; historical live rowwise KV replacement at positions
`658,674` remains a negative diagnostic because it regressed the token gate to
mismatch `108`.

2026-07-19 fast feedback-loop gate hardening: the Plan08 feedback-loop
orchestrator now writes both `verdict.json` and a human-readable `decision.md`
for every run or `--summarize-existing` replay. `verdict.json` contains explicit
phase gates:

- `k1_exact`;
- `k4_correctness_unblocked`;
- `performance_unblocked`;
- `stage`.

It also embeds the exact rerun and summarize commands so a candidate hypothesis
can be validated by one command without reassembling environment state from
notes. Replaying the existing pos789 expanded sparse artifact:

```bash
source .venv/bin/activate
source ./env.sh
python tools/debug/run_deepseek_v4_plan08_feedback_loop.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_candidate_pos789_expanded_20260719_222037 \
  --summarize-existing \
  --base-output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_base_pos789_20260719_221518 \
  --base-sparse-call sparse_capture/rank0_call2.pt \
  --candidate-sparse-call sparse_capture/rank0_call2.pt
```

completed in `9.636s` and intentionally exited RED (`1`) with
`decision=fail`, `stage=blocked_on_k1_exact`, `k1_exact=fail`,
`k4_correctness_unblocked=false`, and `performance_unblocked=false`. The
generated decision artifact is
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/sparse_candidate_pos789_expanded_20260719_222037/decision.md`.
The current machine-selected `next_action` remains
`compare_swa_cache_rows_by_logical_position`; no k=4 or performance gate is
unblocked.

2026-07-20 target-QKV call selector probe: the default-off diagnostic target
QKV pre-norm tokenwise replacement now accepts
`VLLM_METAX_DSV4_TOKENWISE_TARGET_QKV_CALLS`, a comma-separated set of
nonnegative per-layer call indices or `all`. This tightens the rowwise KV
replacement experiment from layer/position-only to layer/call/position without
changing default serving behavior. Unit validation covered the case where the
same position appears in multiple calls and only the selected call is replaced.

Two TP=4 k=1 feedback-loop runs tested the hypothesis that earlier regressions
were caused by position-only replacement firing on the wrong verifier call:

- call0:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/selective_call0_kv674_wqb659_789_oproj659_20260720/`;
- call1:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/selective_call1_kv674_wqb659_789_oproj659_20260720/`.

Both runs stayed RED at first mismatch `26` (`509/850`, candidate length
`205`) and the sparse diff moved the first attention-input difference back to
`q` (`num_diff=8171`, `max_abs=2.71875`). The layer-0 qkv-producer captures
showed the same post-replacement trajectory for call0 and call1:
call0 positions `[673,674]` with col86 values
`[-0.00836181640625, 0.00830078125]`, then call1 positions `[674,675]` with
`[0.00830078125, -0.173828125]`. Therefore the single-row target KV pre-norm
replacement is a negative diagnostic even when call-selected; the next fix
direction should not be another historical row replacement. Continue from the
producer math/dispatch side of batched target `fused_wqa_wkv` determinism, or
build a non-mutating multi-position shadow over the full verifier batch before
attempting another live change.

Artifacts:

- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_all_layer_capture/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer3_sparse_input_capture/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer3_qkv_insert_pos651_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer3_qkv_insert_pos651_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_all_layer_qkv_insert_pos651_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_all_layer_qkv_insert_pos651_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer0_3_stage_pos651_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer0_3_stage_pos651_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer2_sparse_pos651_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer2_sparse_pos651_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer2_o_proj_pos651_allranks_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer2_o_proj_pos651_allranks_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer2_sparse_pos651_allranks_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer2_sparse_pos651_allranks_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer2_indexer_rank3_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index26_layer2_indexer_rank3_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_complete_native_indexer_q_melanie_eager/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_complete_native_qkv_insert_wired_drop_bonus_melanie_eager/`

- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/current_layer0_3_stage_pos651_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/current_layer2_sparse_pos651_allranks_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/current_layer2_3_qkv_insert_pos651_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/current_indexer_q_verifier_capture/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/current_indexer_q_verifier_position_capture/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/model_runner_metadata_capture/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer0_3_stage_pos659_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer0_3_stage_pos659_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer0_3_stage_pos659_candidate_full/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer_all_call1_pos659_candidate_full_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer_all_call0_pos659_base_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer4_sparse_pos659_candidate_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer4_sparse_pos659_base_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/replay_tmp/layer4_replay_results.json`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_compressor_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_compressor_save_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/layer4_sparse_pos659_tokenwise_compressor_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/compressor_slot_capture_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/compressor_slot_capture_gsm8k_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/compressor_slot_capture_exact_env_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/compressor_slot_capture_rank0_exact_env_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/compressor_slot_capture_rank0_h512_exact_env_mt40/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/no_capture_exact_env_mt40_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/no_capture_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_q_only_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_ffn_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_o_proj_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_sparse_mla_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_qkv_insert_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_compressor_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index134_layer0_3_pos767_mhc_pre_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index134_layer0_3_pos767_base/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index134_layer0_3_pos767_mhc_pre_q_only_candidate/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_wq_b_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/tokenwise_mhc_pre_target_wq_b_exact_env_mt512_current/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/wq_b_shadow_layer0_pos767_mhc_pre_melanie/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/wq_b_shadow_layer0_pos647_mhc_pre_melanie/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/live_target_wq_b_layers0_mhc_pre_melanie/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/live_target_wq_b_layer0_pos767_mhc_pre_melanie/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/mhc_pre_smoke/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/index134_layer0_3_pos767_mhc_pre_candidate/q_stage_diff_layer0_pos767.json`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/live_target_wq_b_layer0_pos767_qstage/`
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260716/k1/token_gate/live_wq_b_pos767_sparse_tokenwise_qstage/`

### 2026-07-20 NV-source parity and full k=1 gate

NV source comparison identified two missing DeepSeek V4 MTP parity fixes and
both are now ported locally:

- PR #48598 behavior: `MacaDeepseekV4Attention.forward` handles 3-D HC/MTP
  packed `hidden_states` by projecting and attending only slot 0
  (`hidden_states[:, 0, :]`).
- PR #48304 behavior: MTP draft-layer raw `compress_ratio=0` resolves to
  operational ratio `1` and selects unscaled/plain RoPE without mutating the
  shared config rope parameters. This matches the current model metadata:
  `num_hidden_layers=43`, `len(compress_ratios)=44`, and draft entry
  `compress_ratios[43]=0`.

Focused unit coverage now guards both behaviors. The full TP=4 k=1 feedback
gate was rerun with `num_speculative_tokens=1`, `max_tokens=512`,
`max_model_len=1024`, `gpu_memory_utilization=0.9`, `cudagraph_mode=PIECEWISE`,
and `--no-diagnostic-enforce-eager`:

- Artifact root:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_after_nv_attention_rope_full/`
- Result: still RED at first mismatch `108`; oracle/candidate token counts
  `212/194`.
- Finish reasons match (`stop/stop`) and the candidate answer is correct
  (`18`), but exact token IDs still diverge.
- Sparse diff reports first tensor difference at `token_indices`, first
  attention-input difference at `q`, with `q num_diff=8165` and
  `max_abs=4.8828125`.

The feedback-loop wrapper was also hardened so SWA mapping postprocessing shape
mismatches are non-blocking. The rerun now writes `verdict.json` and
`decision.md` even when mapping analysis records an error:
`The size of tensor a (64) must match the size of tensor b (16) at
non-singleton dimension 0`.

MTP remains default-off. No PIECEWISE exact gate, k=4 gate, graph/replay gate,
normal-serving benchmark, or 2x speedup claim is valid yet.

### 2026-07-20 PIECEWISE model capture seam and five-probe k=1 diagnostic

The model-level layer capture seam now works under PIECEWISE graph replay. The
first `model.py` attempt wrapped `maybe_layer_capture_context` with
`eager_break_during_capture`, but returning a Python `LayerCaptureContext` did
not preserve replay-time stage saves. The revised diagnostic wrapper creates the
context and saves the stage inside the eager-break function, while preserving
the existing capture env guard.

Patch2 model-stage artifacts:

- Base MTP=0 PASS:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/mtp0_base_pos659_l01_modelstage_20260720_after_patch2/`
- Candidate with target KV prenorm + WQ_B tokenwise remains RED at first
  mismatch `26` (`509/850`):
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_tokenwise_kvprenorm_wqb_pos659_l01_modelstage_20260720_after_patch2/`
- At position `659`, layer-0 QKV producer, attention inputs, and sparse
  attention output are exact. The first model-level divergence is
  `layer0 after_attention.hidden_states`, i.e. the O-projection tail after
  sparse attention.
- Enabling all-row tokenwise O-projection makes layer-0 `after_attention`
  exact, but `layer0 after_ffn.hidden_states` still diverges. Adding all-row
  tokenwise FFN still leaves the FFN-pre MHC mix-state difference visible.

The current diagnostic-only TP=4 PIECEWISE k=1 PASS requires all five rowwise
equivalence probes:

```text
VLLM_METAX_DSV4_TOKENWISE_MHC_PRE=1
VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM=1
VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B=1
VLLM_METAX_DSV4_TOKENWISE_O_PROJ=1
VLLM_METAX_DSV4_TOKENWISE_FFN=1
```

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_tokenwise_mhcpre_kvprenorm_wqb_oproj_ffn_all_l01_modelstage_20260720/`.
Result: PASS, exact token IDs, `212/212` tokens.

Minimization shows each probe is currently load-bearing for this diagnostic
PASS:

| removed probe | result | artifact |
| --- | --- | --- |
| MHC pre | RED at mismatch `26` | `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_tokenwise_kvprenorm_wqb_oproj_ffn_all_l01_modelstage_20260720/` |
| FFN | RED at mismatch `134` | `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_min_mhcpre_kvprenorm_wqb_oproj_all_20260720/` |
| O-projection | RED at mismatch `26` | `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_min_minus_oproj_mhcpre_kvprenorm_wqb_ffn_20260720/` |
| target KV prenorm | RED at mismatch `26` | `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_min_minus_kvprenorm_mhcpre_wqb_oproj_ffn_20260720/` |
| target WQ_B | RED at mismatch `26` | `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_min_minus_wqb_mhcpre_kvprenorm_oproj_ffn_20260720/` |

This is not production acceptance or performance evidence. It proves a
rowwise-equivalence diagnostic path can reproduce the frozen greedy sequence.
Next work is to turn the load-bearing probes into graph-safe production
behavior, or to prove which native batched kernels must match rowwise numerics.

### 2026-07-20 default-off single-switch k=1 correctness candidate

A default-off single switch now bundles the five load-bearing rowwise
equivalence probes for TP=4 PIECEWISE k=1 correctness diagnosis:
`VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1`. The switch is registered with
the vLLM environment validator through the MetaX platform plugin, so the
candidate flag no longer appears in unknown-env startup warnings.

The clean frozen token gate used only this candidate switch plus the existing
Plan08 exact-env defaults. The five individual probe envs and the related
position/layer/call selectors were explicitly unset; `plan08_debug_loop=false`,
so no capture envs were needed for the passing gate.

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_single_switch_candidate_clean3_20260720/`.
Result:

- `exit_code.txt`: `0`
- `comparison.json`: `decision=pass`, `exact=true`, `first_mismatch=null`
- Oracle/candidate token counts: `212/212`
- Finish reasons match
- `run.log`: PIECEWISE graph capture completed `2/2`, and markers confirm the
  implied MHC pre, target KV prenorm, target WQ_B, O-projection, and FFN
  tokenwise paths.

This is a k=1 correctness candidate only. It is not k=4 acceptance, not a
normal-serving throughput result, and not a 2x performance claim. MTP remains
default-off unless the candidate env is explicitly set.

### 2026-07-20 five-probe replay and native WQ_B RED evidence

The Plan08 captured-input replay harness now has one manifest-driven command:

```bash
python tools/debug/replay_deepseek_v4_plan08_probes.py \
  --manifest .logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_probe_replay_20260720/manifest.json \
  --output .logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_probe_replay_20260720/summary.json
```

The O-projection replay gap was caused by the old
`.logs/plan03_oproj_fail_seed1302.pt` schema, not by a proven O-projection
numeric mismatch. A TP=4 k=1 diagnostic eager gate produced current-schema
captures under
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_oproj_current_schema_k1_eager_20260720/`.
The gate was exact (`212/212`, `first_mismatch=null`) and wrote 8 layer-0
O-projection captures. Both offline replays passed:
`o_proj_replay_torch.json` and `o_proj_replay_native.json` have `failed=0`.
Because O-projection capture intentionally returns during CUDA graph capture,
this is eager diagnostic replay evidence, not PIECEWISE acceptance evidence.

Current five-probe replay summary:

| probe | status | evidence |
| --- | --- | --- |
| MHC_PRE | exact | 8/8 corpus files bitwise |
| KV_PRENORM / qkv_insert | divergent | layer 0 position 767, first tensor `q` |
| WQ_B q-stage | exact | layer 0 position 659, raw/post Q `2/2` |
| O_PROJ | exact | current-schema eager capture, 8 files, torch/native replay `failed=0` |
| FFN after_ffn | exact | layer 0 position 659, `2/2` rows |

A native/batched WQ_B replacement candidate
(`VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE=1`) was graph-runnable but not
correctness-equivalent. Artifact
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_wqb_candidate_20260720/`
failed the TP=4 k=1 frozen gate at `first_mismatch=26`; the run captured the
PIECEWISE graph and removed the tokenwise WQ_B marker, so WQ_B rowwise remains
load-bearing for the current k=1 exact path.

To avoid treating cache-slot mismatch as tensor divergence, the replay matrix
helper records QKV tensor equality separately from cache metadata alignment:

```bash
python tools/debug/summarize_deepseek_v4_plan08_replay_matrix.py \
  --base-layer-capture .logs/deepseek_v4_mtp_k4_exact_2x_20260720/mtp0_base_pos659_all_layers_20260720/layer_capture \
  --candidate-layer-capture .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_tokenwise_kvprenorm_wqb_pos659_all_layers_20260720/layer_capture \
  --base-qkv-capture .logs/deepseek_v4_mtp_k4_exact_2x_20260720/mtp0_base_pos659_all_layers_20260720/qkv_capture \
  --candidate-qkv-capture .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_tokenwise_kvprenorm_wqb_pos659_all_layers_20260720/qkv_capture \
  --position 659 --rank 0 --layers 0:42 \
  --layer-stage attention_inputs --layer-stage attention_output \
  --output .logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_replay_matrix_20260720/kvprenorm_wqb_pos659_attention_matrix.json
```

That matrix has no runtime errors: `total=172`, `exact=3`,
`alignment_mismatch=1`, `divergent=168`. Layer 0 has exact q-stage, exact
attention input/output, and qkv tensors exact with only `cache_block_indices`,
`slot_mapping`, and `token_indices` mismatching. From layer 1 onward, q-stage
and attention inputs diverge (`raw_q` / `kv`). Therefore the next candidate
should not be another blind WQ_B replacement; it should first capture or replay
the layer-0 output to layer-1 input transition under a cache-aligned pairing,
then target the earliest real tensor boundary that remains divergent.

The same matrix helper can now run model-stage-only analysis with
`--skip-q-stage --skip-qkv-insert`; this avoids unrelated qkv runtime errors
when an artifact contains only model stages. Two current artifacts summarize the
layer-0/layer-1 boundary at position 659:

- `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_replay_matrix_20260720/kvprenorm_wqb_pos659_l01_modelstage_only_matrix.json`
  (`KV_PRENORM + WQ_B` only): `total=6`, `divergent=6`. The first model-stage
  mismatch is already `layer0 before_attention.post_mix`, so missing MHC_PRE
  creates a tiny FP32 drift before the attention boundary.
- `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_replay_matrix_20260720/all5_pos659_l01_modelstage_only_matrix.json`
  (all five rowwise probes): `total=6`, `exact=4`, `divergent=2`. Layer 0 is
  exact through `after_ffn`, and layer 1 is exact at `before_attention`; the
  first remaining local mismatch is `layer1 after_attention.hidden_states`.

This moves the next root-cause target from WQ_B alone to the layer-1 attention
boundary under the full five-probe candidate. A production replacement should be
derived from same-input layer-1 attention replay evidence, not from another
WQ_B-only toggle.

### 2026-07-20 k=4 single-switch frozen token gate

The required TP=4 PIECEWISE k=4 frozen token gate was run with only the
default-off single switch enabled:
`VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1`. The five individual rowwise
envs were explicitly unset. Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k4_single_switch_candidate_20260720/`.

Result: RED correctness failure, not a graph or dispatch failure.

- `comparison.json`: `decision=fail`, `exact=false`, `first_mismatch=212`.
- Oracle token count: `212`; candidate token count: `213`.
- Finish reasons match: `stop` / `stop`.
- Candidate summary still has GSM8K `accuracy=1.0` and predicted answer `18`.
- The divergence is an extra tail token: oracle window
  `[223, 864, 271, 10375, 28]`, candidate window
  `[223, 864, 271, 10375, 28, 334]`.
- `run.log` shows PIECEWISE graph capture sizes `[1, 2, 3, 4, 5]` and graph
  capture completed `5/5`.
- `run.log` shows all five implied rowwise markers: initial MHC pre, target KV
  prenorm, WQ_B, O-projection, and FFN.

This satisfies the k=4 RED/green attempt requirement but does not satisfy k=4
exact acceptance. MTP must remain default-off. The next correctness question is
no longer whether the five rowwise paths can graph-capture for k=4; they can.
The remaining failure is end-of-sequence acceptance/commit behavior for k=4,
where the candidate emits one extra token after the MTP=0 oracle sequence.

A follow-up k=4 PIECEWISE debug-loop gate with the same workload and captures
enabled reproduced the same RED:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k4_single_switch_candidate_debugloop_20260720/`.
It wrote 5084 MTP capture records and no invalid MTP boundary was reported by
`diff_deepseek_v4_mtp.py`.

Tail MTP capture evidence falsifies the initial "all-accepted unverified bonus"
hypothesis. The extra `334` is a correction token after three accepted draft
tokens, not an all-accepted bonus:

- final relevant `verify`/`commit` record:
  `accepted_count=3`, `draft_ids=[271, 10375, 28, 455]`,
  `target_ids=[271, 10375, 28, 334]`,
  `committed_ids=[271, 10375, 28, 334]`.
- The corresponding proposer record has
  `target_ids=[864, 271, 10375, 28, 455]`,
  `next_token_ids=[334]`, `num_rejected_tokens=[1]`,
  `token_indices_to_sample=[3]`.

Therefore `VLLM_METAX_MTP_DROP_UNVERIFIED_BONUS=1` not covering the V2 sampler
is not sufficient to explain this RED. The next candidate should be stop-aware
speculative commit/truncation: once an accepted draft token completes the same
stop condition as the MTP=0 oracle, the following correction token must not be
surfaced even if the verifier can produce it.

### 2026-07-20 stop-aware output truncation candidate

The k=4 tail RED was traced to vLLM V1 output handling rather than the GPU
verifier. Upstream `OutputProcessor` passes a whole multi-token
`EngineCoreOutput.new_token_ids` batch to the detokenizer. Token-level
`check_stop` has already run in the scheduler, but stop strings are detected
later by the frontend detokenizer: it truncates `output_text` only and leaves
the full token-id list visible. For speculative batches this can surface a
correction token after a prior accepted token has already completed the stop
string. The observed DeepSeek tail decodes as:

- oracle tail token IDs `[223, 864, 271, 10375, 28]` ->
  `" 18\n\nQuestion:"`;
- candidate tail token IDs `[223, 864, 271, 10375, 28, 334]` ->
  `" 18\n\nQuestion: A"`.

Implemented candidate:

- `vllm_metax/patch/bugfix/plan08_stop_aware_output.py`
- default-off env:
  `VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1`
- registered in `vllm_metax/envs.py`
- imported from `vllm_metax/patch/bugfix/__init__.py`

The patch is output-side only: when a stop string is found within a multi-token
detokenizer update, it finds the first token prefix that completes the stop
string and trims the same-batch tail from `new_token_ids` and the detokenizer's
visible token IDs. It does not change GPU graph capture, rejection sampling, or
model logits.

Focused regression evidence:

```bash
source .venv/bin/activate && source ./env.sh
pytest -q tests/patch/bugfix/test_dsv4_drop_bonus.py \
  tests/patch/bugfix/test_plan08_stop_aware_output.py
```

Result: `9 passed`. The new tests directly reproduce the
`[223, 864, 271, 10375, 28, 334]` tail shape and verify that the candidate
trims `334` only when the env is enabled; default-off behavior keeps the tail.
They cover both `include_stop_str_in_output=True` and `False`.

Additional local checks:

```bash
python -m py_compile \
  vllm_metax/patch/bugfix/plan08_stop_aware_output.py \
  vllm_metax/patch/bugfix/dspark_greedy_punctuation_tie.py \
  vllm_metax/envs.py \
  tests/patch/bugfix/test_plan08_stop_aware_output.py
ruff check \
  vllm_metax/patch/bugfix/plan08_stop_aware_output.py \
  tests/patch/bugfix/test_plan08_stop_aware_output.py
pytest -q tests/tools/test_run_deepseek_v4_plan08_token_gate.py
```

Results: py-compile PASS, ruff PASS, token-gate tool tests `8 passed`.

TP=4 PIECEWISE gate evidence with the candidate enabled:

| Gate | Artifact | Decision | Tokens | Graph evidence |
| --- | --- | ---: | ---: | --- |
| k=1 no regression | `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_stop_aware_output_20260720/` | pass, `first_mismatch=null` | `212/212` | PIECEWISE capture `2/2`, graph finished |
| k=4 frozen token | `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k4_stop_aware_output_20260720/` | pass, `first_mismatch=null` | `212/212` | PIECEWISE capture `5/5`, graph finished |

Both gates used TP=4, the frozen oracle
`.logs/deepseek_v4_mtp_k4_exact_2x_20260716/baseline/indexer_prefill_decode_clean_masked_oracle3/run1/summary.json`,
`VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1`, and
`VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1`. Finish reasons matched
`stop/stop`; GSM8K answer accuracy remained `1.0` for the single frozen
question. These are correctness gates only; no throughput claim is made.

Residual risk: this proves the observed stop-string tail shape and the frozen
single-question k=4 gate. It does not prove broad multi-prompt MTP quality,
stop-string matches crossing batch boundaries, logprob trimming with
stop-string truncation, or any performance benefit. MTP should remain
default-off until the broader acceptance corpus and reviewer gate cover this
output-side candidate.

### 2026-07-20 native/batched O-projection replacement candidate

To satisfy the remaining rowwise-replacement requirement, an O-projection
candidate was tested because current-schema O-projection replay was exact while
the full five-probe model-stage matrix first diverged at the layer-1
after-attention boundary. The candidate is:

- env: `VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE=1`
- files:
    - `vllm_metax/models/deepseek_v4/mtp_candidate.py`
    - `vllm_metax/models/deepseek_v4/flashmla.py`
    - `vllm_metax/envs.py`
    - `tests/models/deepseek_v4/test_prefill_gemm_chunking.py`

Behavior: under `VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1`, the new env
disables the all-row tokenwise O-projection path and lets `_o_proj` use the
native batched `deep_gemm_bf16_o_proj` call. Explicit
`VLLM_METAX_DSV4_TOKENWISE_O_PROJ=1` still overrides the candidate and forces
rowwise O-projection.

Focused unit evidence:

```bash
source .venv/bin/activate && source ./env.sh
pytest -q \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py \
  tests/models/deepseek_v4/test_mhc_tokenwise.py \
  tests/patch/bugfix/test_plan08_stop_aware_output.py
```

Result: `56 passed`. The new O-projection test was red before implementation:
with the candidate env enabled it observed two rowwise native calls instead of
one batched call. After implementation it observes one batched call unless
`VLLM_METAX_DSV4_TOKENWISE_O_PROJ=1` is explicitly set.

TP=4 PIECEWISE k=1 frozen token gate:

- artifact:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_oproj_candidate_20260720/`
- envs:
  `VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1`,
  `VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE=1`,
  `VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1`
- graph evidence: PIECEWISE capture `2/2`, graph finished.
- dispatch evidence: `run.log` contains MHC_PRE, KV_PRENORM, WQ_B, and FFN
  rowwise markers; it does not contain the tokenwise O-projection marker.
- result: RED, `decision=fail`, `exact=false`, `first_mismatch=26`, oracle /
  candidate tokens `212/205`, finish reasons `stop/stop`.

This candidate is graph-runnable and does replace a rowwise path with native
batched O-projection, but it is not correctness-equivalent and therefore does
not satisfy the rowwise-replacement acceptance requirement. It independently
confirms the earlier min-minus O-projection RED; the next replacement attempt
should move earlier than O-projection, toward a cache-aligned MHC/KV/WQ_B
boundary rather than another all-batched O-projection toggle.

### 2026-07-20 cache-aligned attention-output boundary update

A narrower layer-stage replay was run after the native O-projection RED to
avoid another blind full-rowwise toggle. It compares the existing TP=4 PIECEWISE
MTP=0 base capture with the all-five-rowwise k=1 capture at position `659`:

```bash
source .venv/bin/activate && source ./env.sh
python tools/debug/summarize_deepseek_v4_plan08_replay_matrix.py \
  --base-layer-capture .logs/deepseek_v4_mtp_k4_exact_2x_20260720/mtp0_base_pos659_l01_modelstage_20260720_after_patch2/layer_capture \
  --candidate-layer-capture .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_tokenwise_mhcpre_kvprenorm_wqb_oproj_ffn_all_l01_modelstage_20260720/layer_capture \
  --base-qkv-capture .logs/deepseek_v4_mtp_k4_exact_2x_20260720/mtp0_base_pos659_l01_modelstage_20260720_after_patch2/qkv_capture \
  --candidate-qkv-capture .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_tokenwise_mhcpre_kvprenorm_wqb_oproj_ffn_all_l01_modelstage_20260720/qkv_capture \
  --position 659 --layers 0,1 \
  --skip-q-stage --skip-qkv-insert \
  --layer-stage before_attention \
  --layer-stage attention_inputs \
  --layer-stage attention_output \
  --layer-stage after_attention \
  --layer-stage after_ffn \
  --output .logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_replay_matrix_20260720/all5_pos659_l01_boundary_matrix.json
```

Result: `total=10`, `exact=7`, `divergent=3`.

| layer | stage | status | first different tensor |
| ---: | --- | --- | --- |
| 0 | before_attention | exact | - |
| 0 | attention_inputs | exact | - |
| 0 | attention_output | exact | - |
| 0 | after_attention | exact | - |
| 0 | after_ffn | exact | - |
| 1 | before_attention | exact | - |
| 1 | attention_inputs | exact | - |
| 1 | attention_output | divergent | `output` |
| 1 | after_attention | divergent | `hidden_states` |
| 1 | after_ffn | divergent | `hidden_states` |

The direct diff artifact
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_replay_matrix_20260720/all5_l1_attention_inputs_diff.json`
shows both candidate rows are bitwise exact for attention inputs (`qr` and
`kv`). The corresponding attention-output diff
`all5_l1_attention_output_diff.json` shows both candidate rows diverge with the
same first BF16 difference and `max_abs=0.015625`.

Interpretation: the earliest currently proven all-five-rowwise mismatch is the
raw sparse-MLA attention output before O-projection, not the O-projection or
FFN. This explains why full native WQ_B and full native O-projection toggles
remain poor next candidates. A useful next candidate must first account for
batched speculative sparse-MLA decode numerics or metadata at layer 1 position
`659`; serving with `VLLM_METAX_DSV4_SPARSE_MLA_DECODE_BACKEND=torch_reference`
is not acceptable because it is an explicit Torch fallback.

Existing sparse-MLA corpus replay is only partial evidence because the default
Plan08 debug loop captured layer 0 position `767`, not the layer 1 position
`659` that now carries the first boundary mismatch. For the all-five-rowwise
artifact, offline Torch-oracle replay of
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_tokenwise_mhcpre_kvprenorm_wqb_oproj_ffn_all_l01_modelstage_20260720/sparse_capture`
passes tolerance (`failed=0`) but is not bitwise exact; decode
`output_max_abs=0.0078125`. A pair diff between the layer-0 MTP=0 base and k=1
all-five sparse captures shows gathered SWA rows and captured output are exact
for that layer-0 position, so it does not prove the layer-1 issue.

An explicit diagnostic TP=4 PIECEWISE k=1 gate completed under
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_all5_tokenwise_sparse_l1_boundary_20260720/`
with `VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1`, layer capture for
position `659`, and sparse decode diff/capture for layer 1 position `659`.
The frozen token gate passed (`212/212`, `first_mismatch=null`) and PIECEWISE
graph capture completed `2/2`; `run.log` shows the tokenwise sparse MLA decode
marker. Sparse capture files are layer 1 position `659`, with decode rows
`rank0_call2.pt` and `rank0_call3.pt`; offline Torch-oracle replay passes
tolerance (`failed=0`) but remains non-bitwise (`output_max_abs=0.0078125`).
The run did not set `VLLM_METAX_DSV4_LAYER_CAPTURE_DIR`, so it cannot replace
the layer-stage matrix evidence above. It is diagnostic only and must not be
used as a performance result.

The follow-up non-blind O-projection replacement gate combined the native
O-projection candidate with tokenwise sparse MLA decode:

```bash
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_oproj_tokenwise_sparse_20260720 \
  --num-speculative-tokens 1 \
  --max-model-len 1024 \
  --no-diagnostic-enforce-eager \
  --plan08-debug-loop \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1 \
  --env VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1 \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_O_PROJ \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE
```

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_oproj_tokenwise_sparse_20260720/`.
Result: RED, `decision=fail`, `exact=false`, `first_mismatch=26`, oracle /
candidate tokens `212/205`, finish reasons `stop/stop`. The run captured the
PIECEWISE graph (`2/2`) and showed the tokenwise sparse MLA marker while the
tokenwise O-projection marker was absent. Therefore the previous native
O-projection RED was not explained solely by sparse-MLA attention-output
batching; O-projection rowwise remains independently load-bearing under the
current correctness umbrella.

The symmetric WQ_B replacement gate also combined the native WQ_B candidate with
tokenwise sparse MLA decode:

```bash
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_wqb_tokenwise_sparse_20260720 \
  --num-speculative-tokens 1 \
  --max-model-len 1024 \
  --no-diagnostic-enforce-eager \
  --plan08-debug-loop \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1 \
  --env VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1 \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE
```

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_wqb_tokenwise_sparse_20260720/`.
Result: RED, `decision=fail`, `exact=false`, `first_mismatch=26`, oracle /
candidate tokens `212/194`, finish reasons `stop/stop`. The PIECEWISE graph
captured successfully, the tokenwise sparse MLA marker was present, and the WQ_B
rowwise marker was absent. Therefore the previous native WQ_B RED also was not
explained solely by sparse-MLA attention-output batching; WQ_B rowwise remains
independently load-bearing under the current correctness umbrella.

Two additional default-off native candidates were added for the remaining
model-stage rowwise paths:

- `VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE=1`
- `VLLM_METAX_DSV4_MTP_K1_NATIVE_MHC_PRE_CANDIDATE=1`

Both mirror the WQ_B/O-projection candidate pattern: under
`VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1`, they suppress the corresponding
implicit all-row tokenwise path and use the native/batched implementation;
explicit `VLLM_METAX_DSV4_TOKENWISE_FFN=1` or
`VLLM_METAX_DSV4_TOKENWISE_MHC_PRE=1` still forces the rowwise diagnostic path.

Focused local validation:

```bash
source .venv/bin/activate && source ./env.sh
pytest -q \
  tests/models/deepseek_v4/test_mhc_tokenwise.py \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py
python -m py_compile \
  vllm_metax/models/deepseek_v4/mtp_candidate.py \
  vllm_metax/models/deepseek_v4/model.py \
  vllm_metax/envs.py \
  tests/models/deepseek_v4/test_mhc_tokenwise.py
ruff check \
  vllm_metax/models/deepseek_v4/mtp_candidate.py \
  vllm_metax/models/deepseek_v4/model.py \
  vllm_metax/envs.py \
  tests/models/deepseek_v4/test_mhc_tokenwise.py
```

Results: focused tests `55 passed`, py-compile PASS, Ruff PASS.

The FFN replacement gate combined the native FFN candidate with tokenwise sparse
MLA decode:

```bash
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_ffn_tokenwise_sparse_20260720 \
  --num-speculative-tokens 1 \
  --max-model-len 1024 \
  --no-diagnostic-enforce-eager \
  --plan08-debug-loop \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1 \
  --env VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1 \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_FFN \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE
```

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_ffn_tokenwise_sparse_20260720/`.
Result: RED, `decision=fail`, `exact=false`, `first_mismatch=134`, oracle /
candidate tokens `212/240`, finish reasons `stop/stop`. The graph captured
successfully, the tokenwise sparse MLA marker was present, and the FFN rowwise
marker was absent. Therefore FFN rowwise remains load-bearing for exact token
replay even after the sparse-MLA attention-output boundary is controlled.

The MHC_PRE replacement gate used the same controlled sparse-MLA setting:

```bash
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_mhcpre_tokenwise_sparse_20260720 \
  --num-speculative-tokens 1 \
  --max-model-len 1024 \
  --no-diagnostic-enforce-eager \
  --plan08-debug-loop \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_MHC_PRE_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1 \
  --env VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1 \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_MHC_PRE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE
```

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_mhcpre_tokenwise_sparse_20260720/`.
Result: RED, `decision=fail`, `exact=false`, `first_mismatch=26`, oracle /
candidate tokens `212/202`, finish reasons `stop/stop`. The graph captured
successfully, the tokenwise sparse MLA marker was present, and the initial MHC
pre rowwise marker was absent. Therefore MHC_PRE rowwise remains load-bearing
for exact token replay even after the sparse-MLA attention-output boundary is
controlled.

The KV_PRENORM replacement gate covered the last original rowwise path with the
same controlled sparse-MLA setting:

```bash
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_kvprenorm_tokenwise_sparse_20260720 \
  --num-speculative-tokens 1 \
  --max-model-len 1024 \
  --no-diagnostic-enforce-eager \
  --plan08-debug-loop \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_KV_PRENORM_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1 \
  --env VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1 \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_TARGET_KV_PRENORM \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_MHC_PRE_CANDIDATE
```

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_native_kvprenorm_tokenwise_sparse_20260720/`.
Result: RED, `decision=fail`, `exact=false`, `first_mismatch=26`, oracle /
candidate tokens `212/201`, finish reasons `stop/stop`. The graph captured
successfully, the tokenwise sparse MLA marker was present, and the KV_PRENORM
rowwise marker was absent.

Conclusion: under the controlled tokenwise sparse-MLA diagnostic, every
full-row native/batched replacement of an original load-bearing rowwise path is
now independently RED:

| replaced rowwise path | controlled sparse MLA? | first mismatch | artifact |
| --- | --- | ---: | --- |
| MHC_PRE | yes | 26 | `k1_native_mhcpre_tokenwise_sparse_20260720/` |
| KV_PRENORM | yes | 26 | `k1_native_kvprenorm_tokenwise_sparse_20260720/` |
| WQ_B | yes | 26 | `k1_native_wqb_tokenwise_sparse_20260720/` |
| O_PROJ | yes | 26 | `k1_native_oproj_tokenwise_sparse_20260720/` |
| FFN | yes | 134 | `k1_native_ffn_tokenwise_sparse_20260720/` |

The next candidate must therefore be selective and graph-safe, for example a
static layer-scoped native/batched replacement that keeps the early
load-bearing layers rowwise while replacing later layers. Position-dependent
selectors that call `.tolist()` on runtime tensors remain diagnostic only and
are not graph-safe acceptance candidates.

Selective WQ_B candidate plumbing was added as the first graph-safe
native/batched replacement mechanism:

- `VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE=1` without a layer selector
  keeps the previous full-native WQ_B behavior.
- `VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS=<layers>` makes only the listed
  nonnegative layer indices use native/batched WQ_B under the k=1 correctness
  umbrella; all unlisted layers remain rowwise.
- `VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS=all` is equivalent to the existing
  full-native replacement.
- Explicit `VLLM_METAX_DSV4_TOKENWISE_WQ_B=1` or
  `VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B=1` still overrides the candidate and
  forces rowwise WQ_B, preserving diagnostic control.

Focused validation:

```bash
pytest -q \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py::test_k1_native_wq_b_candidate_uses_batched_wq_b_unless_explicit \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py::test_k1_native_wq_b_candidate_layer_selector \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py::test_k1_candidate_target_selectors_are_all_rows \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py::test_tokenwise_target_wq_b_layer_selector
python -m py_compile \
  vllm_metax/models/deepseek_v4/mtp_candidate.py \
  vllm_metax/models/deepseek_v4/attention.py \
  vllm_metax/envs.py \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py
ruff check \
  vllm_metax/models/deepseek_v4/mtp_candidate.py \
  vllm_metax/models/deepseek_v4/attention.py \
  vllm_metax/envs.py \
  tests/models/deepseek_v4/test_prefill_gemm_chunking.py
```

Results: selector tests `4 passed`, py-compile PASS, Ruff PASS.

The first TP=4 PIECEWISE model gate for this selective mechanism is a
conservative late-layer smoke gate that keeps layers 0--41 rowwise and replaces
only layer 42 WQ_B:

```bash
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_selective_wqb_l42_tokenwise_sparse_20260720 \
  --num-speculative-tokens 1 \
  --max-model-len 1024 \
  --no-diagnostic-enforce-eager \
  --plan08-debug-loop \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS=42 \
  --env VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1 \
  --env VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1 \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_WQ_B \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_MHC_PRE_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_KV_PRENORM_CANDIDATE
```

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_selective_wqb_l42_tokenwise_sparse_20260720/`.
Result: GREEN, `decision=pass`, `exact=true`, `first_mismatch=null`, oracle /
candidate tokens `212/212`, finish reasons `stop/stop`, GSM8K accuracy `1.0`,
and exit code `0`. The run log shows PIECEWISE graph capture completed and the
tokenwise sparse MLA marker appeared. This proves the first graph-safe
selective native/batched WQ_B replacement candidate: late layer 42 can use
native/batched WQ_B while layers 0--41 remain rowwise under the frozen k=1
controlled sparse-MLA gate. This gate is a RED/green candidate-selection gate
only; it is not a throughput measurement.

The required follow-up k=4 frozen token gate used the same selective WQ_B
candidate:

```bash
python tools/debug/run_deepseek_v4_plan08_token_gate.py \
  --output-root .logs/deepseek_v4_mtp_k4_exact_2x_20260720/k4_selective_wqb_l42_tokenwise_sparse_20260720 \
  --num-speculative-tokens 4 \
  --max-model-len 1024 \
  --no-diagnostic-enforce-eager \
  --plan08-debug-loop \
  --env VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE=1 \
  --env VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS=42 \
  --env VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION=1 \
  --env VLLM_METAX_DSV4_TOKENWISE_SPARSE_MLA_DECODE=1 \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_TARGET_WQ_B \
  --unset-env VLLM_METAX_DSV4_TOKENWISE_WQ_B \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_MHC_PRE_CANDIDATE \
  --unset-env VLLM_METAX_DSV4_MTP_K1_NATIVE_KV_PRENORM_CANDIDATE
```

Artifact:
`.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k4_selective_wqb_l42_tokenwise_sparse_20260720/`.
Result: GREEN, `decision=pass`, `exact=true`, `first_mismatch=null`, oracle /
candidate tokens `212/212`, finish reasons match, GSM8K accuracy `1.0`, and
exit code `0`. The run log shows PIECEWISE graph capture for size 1--5
completed in 4 seconds and the tokenwise sparse MLA marker appeared. This is a
correctness-only RED/green gate, not a throughput claim.

The selective WQ_B suffix was then expanded under the same TP=4 PIECEWISE
correctness gate, keeping all earlier unlisted layers rowwise and preserving
tokenwise sparse MLA decode:

| native/batched WQ_B layers | k=1 gate | k=4 gate | artifact suffix |
| --- | --- | --- | --- |
| `42` | PASS, exact `212/212` | PASS, exact `212/212` | `selective_wqb_l42_tokenwise_sparse_20260720/` |
| `40--42` | PASS, exact `212/212` | PASS, exact `212/212` | `selective_wqb_l40_42_tokenwise_sparse_20260720/` |
| `38--42` | PASS, exact `212/212` | PASS, exact `212/212` | `selective_wqb_l38_42_tokenwise_sparse_20260720/` |
| `37--42` | PASS, exact `212/212` | PASS, exact `212/212` | `selective_wqb_l37_42_tokenwise_sparse_20260720/` |
| `36--42` | RED at index `134`, `212/240` | not run after k=1 RED | `k1_selective_wqb_l36_42_tokenwise_sparse_20260720/` |

Boundary conclusion: the largest currently verified exact-safe graph candidate
for selective native/batched WQ_B is the late-layer suffix `37--42`. Adding
layer 36 is already RED under k=1, with finish reasons still matching
`stop/stop`, so layer 36 is the first known unsafe addition in this suffix
search. This remains a correctness/boundary result only; no throughput
measurement or MTP-default enablement is claimed.

The layer-36 RED was then reproduced with full layer-stage capture while the
`37--42` suffix remained GREEN under the same TP=4 PIECEWISE k=1 gate:

- GREEN artifact:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_selective_wqb_l37_42_layer35_37_pos767_capture_20260720/`
- RED artifact:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/k1_selective_wqb_l36_42_layer35_37_pos767_capture_20260720/`
- Direct tensor diff:
  `.logs/deepseek_v4_mtp_k4_exact_2x_20260720/plan08_replay_matrix_20260720/selective_wqb_l37_green_vs_l36_red_direct_tensor_diff.json`

Prompt length for the frozen Melanie question is `634` tokens, so logical
position `767` is the decode state immediately before the first mismatching
output token at index `134`. Comparing rank 0, layers 35--37, and positions
`[766,767]` / `[767,768]` shows:

| boundary | result |
| --- | --- |
| layer 35 all captured stages | exact |
| layer 36 `before_attention` / `qkv_producer` / `attention_inputs` | exact |
| layer 36 `q_stages.raw_q` | first local difference: 2 BF16 elements for `[766,767]`, max abs `1.52587890625e-05` |
| layer 36 `q_stages.post_q` | same two elements after qnorm/RoPE, max abs `0.00048828125` |
| layer 36 `attention_output` | 5 BF16 elements differ, max abs `0.001953125` |
| layer 37 inputs and later stages | divergence has propagated |

This makes the current falsifiable root-cause hypothesis: layer 36 has exact
upstream inputs, but native/batched WQ_B produces a slightly different raw Q
than the rowwise WQ_B path, and that two-element BF16 delta is sufficient to
flip the verifier at output index `134`. The next probe should therefore be a
same-input layer-36 WQ_B shadow compare or a captured-input WQ_B replay; it
should not target O-projection, FFN, sparse MLA, or later layers first.

Separate normal decode benchmarking on the wrapper's default one-line prompt
was attempted with the current exact-safe k=4 candidate and `MAX_TOKENS=100`.
The run reached `GENERATED_TOKENS 100` but failed `EXPECTED_TOKEN_IDS`
comparison against the MTP=0 baseline, so it is **not** valid stability or
speedup evidence. Baseline output speed was `15.888831` tok/s
(`6.293729` s median, `6.303609` s P90); the failed candidate run reached
`11.842432` tok/s (`8.444212` s median, `8.460403` s P90) before exit code `1`.
This is a correctness failure on the normal decode workload, not an accepted
performance result.

The same normal decode workload was then run with `NUM_SPECULATIVE_TOKENS=1`
and the same selective native/batched WQ_B suffix `37--42`. It also reached
`GENERATED_TOKENS 100`, but failed the frozen MTP=0 `EXPECTED_TOKEN_IDS` check:
the first mismatch is output token index `67`, where the baseline has `1699`
and the k=1 candidate has `23393`. The measured failed-run speed was
`15.016847` tok/s (`6.659188` s median, `6.666799` s P90), below the
same-workload MTP=0 baseline `15.888831` tok/s. Because exactness failed, this
is not valid speedup evidence and does not justify enabling MTP. Artifact:
`.logs/deepseek_v4_mtp_perf_20260720/mtp1_decode100_exactsafe_wqb_l37_42/`.

A shorter `MAX_TOKENS=70`, `BENCH_RUNS=1`, same-prompt RED reproduced the k=1
normal-decode mismatch at the same output index `67` (`1699` vs `23393`) and
captured rank 0, position `78` around the producer state for that token. The
first compared layer-stage difference is not layer 36: layer 0
`before_attention`, `qkv_producer`, `q_stages`, `attention_inputs`,
`attention_output`, and `after_attention` are exact, while layer 0 `after_ffn`
already differs (`598` BF16 elements, max abs `0.00390625`). Sparse layer
bisection confirms the difference then propagates through layers `8`, `16`,
`24`, `32`, and `35`. This falsifies the normal-prompt variant of the
"layer-36 WQ_B is the first local difference" hypothesis; for this workload the
next probe should be a same-input layer-0 FFN/MoE shadow compare. A diagnostic
`VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE=1` run is not a fix: it moves the
normal-prompt first mismatch earlier to output index `11` (`4990` vs `14626`).
Artifacts:
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_red_capture_base_l0_8_16_24_32_35_pos78/`,
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_red_capture_candidate_l0_8_16_24_32_35_pos78/`,
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_red_capture_diff_l0_8_16_24_32_35_pos78/`,
and
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_native_ffn_probe_l0_8_16_24_32_35_pos78/`.

To split the layer-0 FFN boundary, layer capture was extended with an opt-in
`ffn_input` stage at the normalized MoE input. Re-running the same `MAX_TOKENS=70`
k=1 RED shows layer 0 `after_attention` remains exact at position `78`, but
`ffn_input` is already different before routed/shared MoE executes:
`hidden_states` has `6` BF16 differences with max abs `0.00048828125`; `pre_norm`
has `7` differences with max abs `0.0001220703125`; `post_mix` and `res_mix`
also differ by small FP32 amounts. The subsequent layer 0 `after_ffn` difference
is therefore an amplification of an earlier FFN-stage MHC post/pre plus RMSNorm
boundary difference, not first evidence against the MoE router. A direct
`VLLM_METAX_DSV4_MHC_EXACT_PRE_RMS=1` probe fails closed on the current default
MHC backend (`requires the TileLang fused backend for stage ffn`). Forcing
`VLLM_METAX_DSV4_MHC_BACKEND=tilelang`,
`VLLM_METAX_DSV4_MHC_TILELANG_OPS=fused`, and exact pre-RMS runs but changes the
normal-prompt sequence much earlier, at output index `3` (`6623` vs `100854`),
so it is not a candidate fix. Artifacts:
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_ffn_input_capture_base_l0_pos78/`,
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_ffn_input_capture_candidate_l0_pos78/`,
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_ffn_input_capture_diff_l0_pos78/`,
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_exact_pre_rms_probe_l0_pos78/`,
and
`.logs/deepseek_v4_mtp_perf_20260720/normal_prompt_k1_tilelang_exact_pre_rms_probe_l0_pos78/`.

### 2026-07-21 normal-prompt MHC/QKV same-input continuation

The fixed `normal_short_specdec` loop was rerun on the current worktree with
TP=4, PIECEWISE graph mode, `MAX_TOKENS=70`, and k=1. It reproduced the known
RED exactly at output index `67` (`1699` versus `23393`), with both sides
returning `70` tokens and finish reason `length`. Artifact:
`.logs/deepseek_v4_mtp_validation_matrix/normal_short_continue_20260721/`.

A debug-only, eager-break MHC-pre shadow now compares the live batched output
with rowwise `mhc_pre` outputs computed from the same `residual_cur`; it never
replaces the production result. At layer 0, logical position `78`, the batched
path differs from its rowwise shadow in all three pre outputs:

| tensor | differing elements | max abs |
| --- | ---: | ---: |
| `post_mix` | 8 | `2.9206276e-6` |
| `res_mix` | 32 | `5.1558018e-6` |
| `pre_norm` | 8 | `0.00048828125` |

The rowwise row 0 is bitwise exact to the corresponding MTP=0 layer capture
for `post_mix`, `res_mix`, and `pre_norm`. This proves that the position-78
FFN boundary is a batched MHC-pre numerical difference, not MHC post or
RMSNorm. Artifact:
`.logs/deepseek_v4_mtp_validation_matrix/normal_short_mhc_pre_shadow_20260721_075319/`.

Applying rowwise FFN MHC pre only to layer 0 is still not a fix: the token gate
moves earlier to output index `24`. A matched base/candidate capture at logical
position `35` shows layer 0 `before_attention` exact, but QKV production first
differs in one BF16 `qr_pre_norm` element at index `792` (`3.9815903e-5`
base versus `4.0054321e-5` candidate, max abs `2.3841858e-7`). KV remains
bitwise exact. Q normalization amplifies the corresponding raw-Q difference to
`0.00006103515625`; sparse attention output then differs in `31` elements with
max abs `0.00390625`. Artifacts:
`.logs/deepseek_v4_mtp_validation_matrix/normal_short_mhc_pre_layer0_20260721/`
and
`.logs/deepseek_v4_mtp_validation_matrix/normal_short_mhc_pre_layer0_pos35_capture_20260721/`.

The existing QKV same-input shadow independently confirms that position `35`
uses candidate positions `[34,35]` and that the batched fused QKV projection
differs from rowwise only at row 1, QR index `792`; the rowwise value equals the
MTP=0 base value exactly. Artifact:
`.logs/deepseek_v4_mtp_validation_matrix/normal_short_mhc_pre_layer0_qkv_shadow_pos35_20260721/`.

Decision: both all-layer and layer-0-only tokenwise MHC candidates remain RED,
so MTP stays default-off and no TPS from these diagnostic runs is acceptance
evidence. The next bounded probe is a scoped position-35 pre-norm QKV
replacement combined with the already proven position-78 MHC-pre correction;
it must not apply QKV rewriting to all layers or all positions because that
broader diagnostic is already known to regress earlier workloads.

Focused validation for the new shadow, layer selector, and separate baseline
harness environment completed with `49 passed`; Ruff, `py_compile`, diff check,
and Markdown lint are required again after this evidence update.

#### Scoped QKV/MHC candidate continuation

Two selectors were tightened without changing their default behavior:

- `VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST_POSITIONS` merges rowwise
  MHC-pre outputs only for selected logical positions and preserves the batched
  outputs for other rows.
- `VLLM_METAX_DSV4_MTP_K1_RESPECT_TARGET_QKV_SCOPE=1` makes the k1 candidate
  respect explicit target-QKV layer/position/call filters. The scoped path uses
  a device mask and never calls `torch.nonzero` or `.tolist()` during graph
  capture. Without this opt-in, the existing all-row k1 behavior is unchanged.

The first combined run initially failed during graph capture because the old
selected-index path called `torch.nonzero`; this was an instrumentation failure,
not token evidence. After the graph-safe device-mask fix, scoped pre-norm QKV
at layer 0 position `35` made the position-35 QKV, Q, attention input/output,
and `after_attention` captures bitwise exact. The remaining position-35
differences were only MHC-pre `post_mix` (`4` FP32 values, max
`1.9669533e-6`) and `res_mix` (`15` values, max `2.0600855e-6`). Adding MHC-pre
row correction at position `35` moved the token mismatch from output index
`24` to `49`; the matched position-60 capture again showed QKV and attention
bitwise exact with the first difference at `ffn_input`.

The bounded candidate with layer-0 QKV at position `35` and layer-0 MHC pre at
positions `35,60,78` is the first current-worktree k1 candidate to pass the
fixed TP=4 PIECEWISE `normal_short_specdec` 70-token gate exactly, including
finish reason. Artifact:
`.logs/deepseek_v4_mtp_validation_matrix/normal_short_scoped_qkv35_mhc35_60_78_20260721/`.

This is not a promotion result: the unchanged prompt at `MAX_TOKENS=100` is
still RED at output index `79`, corresponding to logical position `90`.
Artifact:
`.logs/deepseek_v4_mtp_validation_matrix/normal_short100_scoped_qkv35_mhc35_60_78_20260721/`.
MTP therefore remains default-off. The next bounded action is a matched
position-90 layer capture under this exact candidate; no performance or broader
quality gate is unblocked.

### Fixed normal-prompt MTP validation harness

Use
`tools/debug/run_deepseek_v4_mtp_validation_matrix.py` as the fixed normal
decode MTP validation entry point before any performance claim. The harness runs
the same prompt with `NUM_SPECULATIVE_TOKENS=0` to create a fresh TP=4
PIECEWISE greedy oracle, reruns the candidate with `EXPECTED_TOKEN_IDS`, checks
committed token IDs plus finish reason, and writes per-prompt artifacts plus a
machine-readable `summary.json`.

Quick RED/debug loop for the current first failing prompt:

```bash
source .venv/bin/activate
source ./env.sh
python tools/debug/run_deepseek_v4_mtp_validation_matrix.py \
  --output-root .logs/deepseek_v4_mtp_validation_matrix/normal_short_debug_$(date +%Y%m%d_%H%M%S) \
  --prompt-id normal_short_specdec \
  --max-tokens 70 \
  --num-speculative-tokens 1
```

Broader no-fail-fast normal-prompt matrix for candidate promotion:

```bash
source .venv/bin/activate
source ./env.sh
python tools/debug/run_deepseek_v4_mtp_validation_matrix.py \
  --output-root .logs/deepseek_v4_mtp_validation_matrix/full_$(date +%Y%m%d_%H%M%S) \
  --max-tokens 100 \
  --num-speculative-tokens 1 \
  --no-fail-fast
```

The fixed small real-inference smoke corpus is
`tools/debug/corpora/deepseek_v4_real_inference_4.jsonl`. It intentionally uses
only four domains: factual explanation, mathematical reasoning, code debugging,
and document synthesis. Run all four prompts with a fresh per-prompt MTP=0
oracle and forced 100-token output:

```bash
source .venv/bin/activate
source ./env.sh
python tools/debug/run_deepseek_v4_mtp_validation_matrix.py \
  --output-root .logs/deepseek_v4_mtp_validation_matrix/real4_$(date +%Y%m%d_%H%M%S) \
  --prompts-jsonl tools/debug/corpora/deepseek_v4_real_inference_4.jsonl \
  --max-tokens 100 \
  --num-speculative-tokens 1 \
  --no-fail-fast
```

This corpus is the minimum realistic smoke gate before GSM8K or the larger real
QA matrix. Position-scoped QKV/MHC variables are diagnostic-only and are
forbidden in this command; a pass obtained by enumerating positions from one
prompt is invalid.

The default candidate environment is the current Plan08 exact-safe k=1 setting:
generic correctness candidate, native WQ_B for layers `37--42`, stop-aware
truncation, and tokenwise sparse MLA decode. Use `--no-default-candidate-env`
plus explicit `--env KEY=VALUE` only when testing a different bounded
candidate. A run whose summary decision is `fail` is correctness evidence only;
its TPS must not be used as speedup evidence.

### 2026-07-22 k=1 correctness gate and EAGLE prefix-hit fix

The native initial-overlap clear fixed the previously stable seventh-request
failure. Current TP=4 evidence now passes all of the following with
`PIECEWISE`, prefix caching enabled, and serial-target k=1:

- same-engine factual replay: 20/20 exact;
- alternating short/long prompts: 20/20 exact;
- four-domain real prompt corpus: 4/4 exact;
- five-prompt Real-QA corpus: 5/5 token IDs and finish reasons exact, including
  one early-stop request;
- near-512 prompt: 174 input plus 338 output tokens, exact through total context
  length 512.

Primary artifacts:

- `.logs/deepseek_v4_mtp_validation_matrix/real_qa_5_disable_eagle_hit_20260722/`;
- `.logs/deepseek_v4_mtp_validation_matrix/real_qa_near512_mtp0_piecewise_20260721/`;
- `.logs/deepseek_v4_mtp_validation_matrix/real_qa_near512_disable_eagle_hit_20260722/`;
- `.logs/deepseek_v4_mtp_validation_matrix/gsm8k_first2_disable_eagle_hit_20260722/`;
- `.logs/deepseek_v4_mtp_validation_matrix/gsm8k_first2_final_code_20260722/`;
- `.logs/deepseek_v4_mtp_validation_matrix/gsm8k_first2_native_k1_acceptance_20260722/`;
- `.logs/deepseek_v4_mtp_validation_matrix/gsm8k_seed42_100_disable_eagle_hit_20260722/`.

The prior frozen GSM8K seed-42 RED is superseded for serial-target k=1. In the
minimized Melanie-to-Tom pair, MTP=0 reused the full 512-token shared prefix and
computed 113 target tokens, while MTP=1 was incorrectly reduced to a 256-token
hit and computed 369 target tokens. The hybrid coordinator's EAGLE group was
probing one additional 64-token lookahead block through token 576 before
discarding it. The two prompts share tokens only through 512, so this
speculative-only extra-block check incorrectly rejected a valid target prefix
hit and changed the prefill hidden state before the first divergent token.

The serial correctness candidate now disables `use_eagle` only while
`HybridKVCacheCoordinator.find_longest_cache_hit` computes the shared prefix,
then restores every mutable or immutable attention group in `finally`. The
target still recomputes the uncached 113-token suffix, so the change does not
skip target hidden-state production. The minimized gate is 2/2 exact, and the
full TP=4, PIECEWISE, prefix-cache-on GSM8K gate is 100/100 exact for token IDs,
output lengths, and finish reasons. MTP=0 and MTP=1 both report accuracy `0.76`,
zero invalid outputs, 33 length terminations, and 8041 output tokens; no
fallback or runtime error marker is present.

The earlier serial-target counter `accepted=0,rejected=8108` was not a native
acceptance result. AsyncScheduler populates `request.spec_token_ids` with `-1`
placeholders until the worker-side draft IDs are copied back, and the serial
diagnostic was mistakenly counting those placeholders as rejections. The
per-output pairing fix is retained, and negative placeholders are now reported
as `unavailable` rather than rejected.

Native TP=4, PIECEWISE, prefix-cache-on k=1 diagnostics report `39/45 = 86.7%`
accepted draft tokens on one prompt and `66/76 = 86.84%` across the first two
frozen GSM8K prompts, with `0.868` accepted tokens per step plus one bonus token
per step. This exceeds the interim 50% validation target. The outputs still
diverge from the frozen MTP=0 oracle (first mismatches at token indices 14 and
48), so acceptance rate and exact greedy correctness remain separate gates.
MTP remains default-off until the native acceptance path also passes the TP=4
exactness corpus; these are correctness diagnostics, not throughput claims.

Native acceptance artifact:

- `.logs/deepseek_v4_mtp_validation_matrix/gsm8k_first1_native_k1_baseline_20260722/`;
- `.logs/deepseek_v4_mtp_validation_matrix/gsm8k_first2_native_k1_acceptance_20260722/`.

The current normal k=1 benchmark reused the 30-TPS MTP=0 configuration:
TP=4, PIECEWISE, prefix cache on, MHC TileLang fused exact mode,
`max_num_batched_tokens=8192`, three warmups, five measured requests, and
`max_tokens=100`. Native MTP=1 measured median/P90 `33.718/32.522 tok/s` with
latencies `2.966/3.075 s`, 70.0% draft acceptance, and per-GPU average
utilization `25.42%--25.48%`. All five requests generated 100 tokens and ended
by length. The old MTP=0 `30.358 tok/s` reference generated only 78 tokens and
stopped, so these TPS values are not a strict same-output speedup comparison.
The MTP=1 tokens first differ from that reference at index 3; the run remains a
correctness RED and does not satisfy the 50 tok/s or exactness gates.

Performance artifact:

- `.logs/deepseek_v4_mtp_validation_matrix/normal_mtp1_prefix_tilelang_8192_20260722/`.

The current-code TP=4 k=4 frozen token smoke was rerun with iterative reuse of
the checkpoint's single MTP layer, selective native WQ_B on layer 42,
stop-aware output truncation, and tokenwise sparse MLA decode. PIECEWISE graph
capture completed for sizes 1 through 5. The candidate is exact `212/212`
tokens against the frozen oracle, finish reasons are `stop/stop`, GSM8K accuracy
is `1.0`, and no fallback or runtime error marker is present. This confirms that
k=4 starts, captures, and completes native four-token verification on the
controlled workload. It does not override the broader native k=1/k=4
correctness blockers and is not a performance acceptance result.

Current k=4 smoke artifact:

- `.logs/deepseek_v4_mtp_validation_matrix/k4_selective_wqb_l42_current_20260722/`.

The matching pre-fix stats-enabled k=4 replica remains exact `212/212` with
`stop/stop` and graph sizes 1 through 5. Across 71 verification steps it
accepted 132 of 284 proposed draft tokens, for overall draft acceptance
`46.48%`, or `1.859` accepted drafts per verification step. Mean committed
acceptance length is therefore `2.859` tokens per target verification. Weighted
acceptance by draft position is `94.36%`, `64.77%`, `26.76%`, and `0.00%` for
positions 1 through 4. This pre-fix artifact made the fourth iterative draft
appear to provide no accepted tokens; the corrected evidence below supersedes
that conclusion.

Current k=4 acceptance artifact:

- `.logs/deepseek_v4_mtp_validation_matrix/k4_selective_wqb_l42_acceptance_20260722/`.

The matching pre-fix k=3 stats run is also exact `212/212` with `stop/stop` and graph
sizes 1 through 4, but its acceptance profile is not the k=4 profile with the
unused fourth draft removed. It accepted 120 of 219 proposed drafts (`54.79%`)
over 73 verification steps, or `1.644` accepted drafts per step and acceptance
length `2.644`. Weighted position acceptance is `95.89%`, `68.49%`, and
`0.00%`. Relative to k=4 it saves 65 proposals (`22.89%`) but requires 73 rather
than 71 verification steps and accepts fewer drafts per step (`1.644` versus
`1.859`). This establishes lower scheduled draft/verification work, not an
end-to-end speedup; a timed same-workload comparison is still required.

Current k=3 comparison artifact:

- `.logs/deepseek_v4_mtp_validation_matrix/k3_selective_wqb_l42_acceptance_20260722/`.

### 2026-07-22 terminal-position acceptance fix

The pre-fix k=3 and k=4 runs above explicitly enabled
`VLLM_METAX_MTP_DROP_UNVERIFIED_BONUS=1`. On the active V1 sampler path this
mutated `result[:, K]` to `-1`. That slot contains the target bonus token only
when all K drafts are accepted, while the scheduler derives accepted drafts as
`len(generated_token_ids) - 1`. The mutation therefore capped the real
accepted count at K-1 and made the terminal position structurally zero. This is
separate from the earlier stop-string tail correction issue described above.

The fix removes the sampler mutation and removes the legacy flag from the token
gate's default exact environment. The stop-aware frontend truncation remains
responsible for hiding tokens after an already completed stop string. A
k=3/k=4 full-acceptance regression first failed with scheduler acceptance
`2 != 3`, then passed after the fix. The related test set reports `18 passed`;
Ruff, `py_compile`, and `git diff --check` also pass.

Both TP=4 PIECEWISE reruns deliberately kept the legacy flag explicitly set to
`1` to verify backward-compatible behavior. The k=3 run remains exact
`212/212`, `stop/stop`, captures graph sizes 1 through 4, and has no fallback or
strict marker. It accepted `135/213` drafts over 71 verification steps, or
`1.901` accepted drafts per step. Weighted position acceptance is `94.36%`,
`67.61%`, and `28.17%`.

The k=4 run remains exact `212/212`, `stop/stop`, captures graph sizes 1 through
5, and has no fallback or strict marker. It accepted `137/276` drafts over 69
verification steps, or `1.986` accepted drafts per step. Weighted position
acceptance is `95.65%`, `66.67%`, `28.99%`, and `7.25%`. The terminal draft is
now demonstrably accepted, although its marginal acceptance remains low.

Corrected artifacts:

- `.logs/deepseek_v4_mtp_validation_matrix/k3_last_position_fix_20260722/`;
- `.logs/deepseek_v4_mtp_validation_matrix/k4_last_position_fix_20260722/`.

This resolves the terminal-zero blocker on the frozen prompt only. It is not a
throughput result and does not replace the broader TP=4 exactness corpus, so MTP
remains default-off.

### 2026-07-23 strict 100-token k=4 baseline and first MHC optimization

The offline runner now accepts `MIN_TOKENS`, validates `1 <= MIN_TOKENS <=
MAX_TOKENS`, and applies it only to measured requests. Warmup and profiler
priming remain one-token requests. The focused runner suite reports `10 passed`.
This removes the earlier 78-token versus 100-token comparison error.

The frozen normal-serving workload is TP=4, PIECEWISE, prefix cache enabled,
`MAX_MODEL_LEN=512`, `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_TOKENS=100`,
`MIN_TOKENS=100`, three warmups, and five measured runs. A fresh MTP=0 run
measured median `31.0935 TPS`. The pre-optimization k=4 run measured median
`13.2586 TPS`, or `0.4264x` base, with nearly five times the target-model kernel
call count. Both paths were internally stable, but k=4 first diverged from base
at token index 44. This result is diagnostic only.

The first performance candidate batches the exact native MHC cast/sqrsum and
downstream RMS operations for N=1 through 5 while preserving the original
per-row native FP32 GEMV. A fully batched `mm_out` GEMV was rejected because it
was not bitwise equal for N=2 through 5. The accepted hybrid kernel gate is
bitwise equal for every N=1 through 5, overwrites caller buffers, rejects the
non-production hidden size, and passes 20 graph replays per shape. For N=5 its
median graph-replay latency fell from `0.5723 ms` to `0.2834 ms`, a `2.02x`
isolated speedup. The candidate is default-off and emits explicit hybrid
dispatch evidence.

The controlled TP=4 PIECEWISE token gate passes `212/212`, but the strict
100-token prompt still diverges from MTP=0 at token index 44 (`16` versus
`1901`). Normal k=4 throughput improves to median `15.9592 TPS` and P90
`15.6984 TPS`, which is `1.2037x` the prior k=4 path but only `0.5133x` MTP=0.
Average GPU utilization over the measured window is approximately `33.8%` on
each GPU. This is a useful first performance delta, not an accepted MTP result;
the broader exactness failure still blocks normal-serving acceptance.

Artifacts:

- `.logs/deepseek_v4_mtp_performance/k4_100token_20260723/benchmark_summary.json`;
- `artifacts/mhc_batched_post_validation.json`;
- `.logs/deepseek_v4_mtp_performance/k4_hybrid_exact_v2_20260723/`;
- `.logs/deepseek_v4_mtp_performance/k4_hybrid_100token_20260723/benchmark_summary.json`.

## Stop conditions

出现以下任一情况立即停止性能优化并保持 MTP default-off：

- MTP=0 base 三次运行不能形成稳定 exact oracle；
- k=1--4 任一级出现任何 committed token、finish reason 或 KV state 分歧；
- 必须启用多请求 unsafe batching、eager 或 Torch fallback 才能运行；
- k=4 graph size5 无法 capture/replay；
- 质量 gate 依赖降低 max tokens、TP、context、batch difficulty 或精度；
- 正常 serving speedup 未达到2x，即使 profiler 或 microbenchmark 达到；
- `_metax_sparse_C` SHA 改变但没有对应 kernel correctness 计划和验证。

## 新会话任务提示

```text
在 /root/vLLM-metax 完整执行
docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md。

开始前依次阅读：
1. /root/vLLM-metax/AGENTS.md
2. docs/superpowers/plans/deepseek-v4-c500-e2e/00-shared-contract.md
3. docs/superpowers/plans/deepseek-v4-c500-e2e/08-mtp-k4-exact-2x.md
4. docs/superpowers/plans/deepseek-v4-c500-e2e/07-integration-acceptance.md
5. docs/superpowers/plans/deepseek-v4-c500-e2e/README.md

使用 executing-plans、diagnosing-bugs 和 TDD 工作流，逐项更新 Plan 08 checkbox。
保留 dirty worktree，不要 reset、revert 或覆盖已有修改。

冻结条件：TP=4、Plan02 exact-on、PIECEWISE、prefix cache on、GPU_MEM=0.9、
max_num_seqs=1。当前 MTP=0 正常基线为 30.357923 TPS；当前 checkpoint 只有一个
MTP layer，k=4 必须迭代复用，不能复制四份权重。_metax_sparse_C SHA 必须保持
28ea10c90d60189f47b38c41e24d3b3f59d6f31c847a6f4b23c19a8bccfaf821。

目标是 k=4 与 MTP=0 greedy 逐 token 完全一致，GSM8K/真实问答/graph/replay 全部
无损，并在相同正常 serving workload 上 median 净吞吐至少2.00x、P90至少1.80x。
任何 token 分歧、fallback、graph failure 或低于2x都保持默认关闭并保存 blocker。

模型加载 benchmark 委派 benchmark-runner，profile 委派 profiler，普通测试委派
test-runner，最终审查委派 reviewer。不要把 profiler TPS 当正常 serving TPS。

从 Task 1 开始，先建立三次稳定 MTP=0 oracle 和可重复的当前 MTP=1 第二 token
分叉 RED harness；没有稳定 oracle 不得进入 MTP 性能优化。
```
