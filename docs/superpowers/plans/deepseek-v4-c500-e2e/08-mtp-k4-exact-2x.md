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

- [ ] **Step 7: 采集三次 fresh MTP=0 base oracle**

使用相同的固定 corpus、seed、TP=4、PIECEWISE、exact-on 和 serial request guard。
三次 artifact 必须 token exact；若当前已知的 `99/100` 运行间差异再次出现，停止 MTP
工作并将 base determinism 标为 blocker。

- [ ] **Step 8: 复现当前 MTP=1 RED**

运行相同 corpus 的 k=1，保存首个 draft、target verification 和 committed token 差异。
验收 harness 必须稳定报告 `decision=fail` 和第二 token 附近的首个差异。

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
