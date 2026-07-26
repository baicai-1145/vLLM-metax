# 端到端集成、回归和推广验收

## 会话目标

把 Sparse MLA、MHC、O-proj、MoE、collective 和 graph 优化集成到同一工作树，
按固定 A100/C500 矩阵验证累计收益，并决定默认推广或保持 opt-in。

## 输入要求

每个子会话必须提供：

- commit/diff 或明确 changed files；
- corpus 和 direct differential 结果；
- graph replay、pointer 结果；
- TP=4 token IDs；
- 正常 p50/p90 ms/token；
- profiler stage/kernel/launch delta；
- dispatch 证据和 default/opt-in 状态；
- residual risk。

artifact 必须包含 workload manifest、原始 stdout/stderr、commit/环境、warmup 与
重复次数、每次测量及汇总、token IDs/首个差异位置和 dispatch 证据。由
`benchmark-runner` 产出 normal 吞吐与四卡利用率，`profiler` 产出 steady trace
和阶段归因，`kernel-validator` 产出真实输入 differential、graph replay、稳定
指针和隔离 kernel 结果；角色之间不得以 profiler 或微基准数字替代端到端验收。

缺少上述任何一项的优化不能进入集成基线。

## 集成顺序

1. Sparse MLA decode（decode-first）；
2. 立即并行建立 Sparse MLA prefill OOM blocker 的复现和修复门禁；
3. MHC raw/pre+RMSNorm；
4. O-proj；
5. W4A16 MoE；
6. TP collective；
7. graph/launch；
8. 最后才重新启用 MTP；若第二个 token 起偏离 greedy oracle，立即 hard fail 并保持
   default-off。

decode-first 只规定执行优先级，不允许把 prefill blocker 后置：10K default
`chunk=8192` 在 `torch_flash_mla_sparse_prefill` 的 `torch.index_select` 约 `10 GiB`
OOM 必须在集成完成前解决或明确阻止推广。`chunk=2048`、`GPU_MEM=0.8` 的
`1504.16 input tok/s`、median `6.648215 s`（四卡约 `89.1%--89.2%`）仅是当前
workaround 证据；1K prefill 参考为 `1321.61 input tok/s`、`0.774812 s`。

每合入一项都重跑完整门禁；不要等全部合并后再定位错误。

## 完整回归矩阵

- TP=4 eager 2-token 和 16-token；
- TP=4 PIECEWISE 16-token correctness gate 和 100-token 性能测试；
- prefill 短/中/长 context；
- batch 1 为主，补充实际服务 batch；
- attention compression ratio 1/4/128；
- graph capture/replay 两组不同请求；
- MTP=0 基线；基础 decode 稳定后再测 MTP acceptance rate 和净 tokens/s；
- MTP=1 观测到 `17.8358 tok/s` 但第二个 token 起与 greedy oracle 分歧，必须记为
  hard fail，不得进入净吞吐比较；
- TP=1 仅用于计算/通信归因。

### 真实问答和动态 batch 正确性

- 使用 `tools/debug/evaluate_deepseek_v4_quality.py` 保存逐题 JSON artifact；
- GSM8K 数据哈希必须与 `00-shared-contract.md` 一致；
- TP=4、MTP=0、PIECEWISE、Plan02 exact-on 下先跑固定 seed 的小样本，再扩到完整
  数据集；eval TPS 不得作为 serving 性能；
- 增加 batch=2 长短回答交错 gate：短请求先 stop 后，长请求必须与 fresh 单请求逐
  token 一致，禁止出现 BOS/token ID `0` 填充尾巴；
- 同一题分别以 fresh batch=1、连续 batch=1 和单次 batch=2 运行，任何输出分歧均为
  correctness hard fail；
- raw completion、官方 DeepSeek chat/non-thinking prompt 和 thinking prompt 必须分开
  报告，不得把 prompt protocol 不兼容误记为模型准确率；
- eager、Plan02 off、Torch reference 或其他 fallback 结果只用于定位，不得进入推广。

当前状态（2026-07-16）：已修复 Sparse MLA scale-mask 越界、C4 local top-k 越界、
compatibility top-k 读错 cache 和 BF16 compressor 行 stride 四个原生路径缺陷。
TP=4、MTP=0、PIECEWISE、Plan02 exact-on 的 250+32 边界门禁为 32/32 非零，四 rank
positions 256--266 hidden/logits finite，且无 fallback。证据位于
`.logs/deepseek_v4_flash_quality_eval_20260716/boundary_250_final_stride_postfix/`。

独立 fresh-engine chat/non-thinking 三题在冻结 `max_tokens=256` 评分下为 `1/3`：事实
题通过，数学和代码题因输出上限截断而未出现最终答案，但不再出现 token ID `0` 或
数值退化。同 prompt 的 512-token 补充运行使数学和代码均正常 stop，分别得到
`960 liters` 和精确 stdout
`[2, 1, 3, 2] [2, 3, 6, 8] [12, 16]`，为 `2/2`。证据位于
`.logs/deepseek_v4_flash_quality_eval_20260716/real_qa_final_postfix/` 和
`.logs/deepseek_v4_flash_quality_eval_20260716/real_qa_extended_512_postfix/`。补充运行
不替代冻结的 256-token gate、完整 GSM8K 或动态 batch token 一致性。混合预算
`3/3` 语义问答综合结论见
`.logs/deepseek_v4_flash_quality_eval_20260716/real_qa_completed_postfix/`。

正常 native logits 的 seed42 100 题 GSM8K 已重跑：batch=1 canonical/人工复核分数为
`93%/96%`，batch=2 为 `94%/97%`，两路均无 invalid、length、token ID `0` 或 runtime
failure。该结果达到官方公开 DeepSeek-V4-Flash Base `90.8` 的同一量级，但官方结果
使用 8-shot FP4/FP8 mixed checkpoint，不能作为本地 W4A16 的 exact oracle。证据位于
`.logs/deepseek_v4_flash_quality_eval_20260716/gsm8k_seed42_100_postfix/`。

动态 batch token gate 仍失败：token exact `17/100`、解析答案一致 `92/100`、correctness
一致 `93/100`。FP32 logits 使 20 题 batch=2 从 `19/20` 降至 `18/20`，已拒绝推广。
Sparse MLA 四组 graph/eager、batch1/2 native/reference 差分只得到 layer0 C1
`7.629e-6` BF16 舍入差，probabilities/cache gather exact 且无 NaN/Inf，尚不能归因
后续答案分叉。集成推广继续被动态 batch 确定性和同 checkpoint A100 对照阻止。

## 性能归因表

为每一阶段维护同一张表：

| 阶段             | 开始 ms/token | 结束 ms/token | kernel/launch 差值 | 端到端增益 | 正确性 |
| ---------------- | ------------: | ------------: | -----------------: | ---------: | ------ |
| Sparse MLA       |               |               |                    |            |        |
| MHC+RMSNorm      |               |               |                    |            |        |
| O-proj           |               |               |                    |            |        |
| W4A16 MoE        |               |               |                    |            |        |
| TP communication |               |               |                    |            |        |
| Graph/runtime    |               |               |                    |            |        |

表中空值由集成会话根据实际日志填写，不能预估或用 microbenchmark 代替。

最新 MTP=0、TP=4、PIECEWISE、100-prompt x 100-token decode 参考为 median
`27.2195 tok/s`、P90 `42.879 ms/token`，四卡利用率 `30.32%--30.43%`。目录中更早的
`11.624672 token/s`/`86.0 ms/token` 必须标为历史起始基线，不得与新 workload 直接
比较。

## 推广条件

- 所有 greedy token、真实 corpus 和 graph gate 通过；
- normal inference 至少五次测量，方差小于 3%；
- 没有静默 Torch fallback、旧输入 replay 或异常 rank 分歧；
- default-on 路径支持实际 prefill/decode shape，而不是只支持单一 probe；
- profiler 显示目标阶段减少，而不是开销转移到 copy、sync 或其他 stream；
- 10K prefill default `chunk=8192` 不再因 `torch.index_select` 约 `10 GiB` 分配 OOM；
- 已安装 `deep_gemm.int8_mqa_logits` 的无 `backend` kwarg 和动态 multi-head Triton
  `mcTriton SmallVector` 问题有明确 dispatch 证据；3D head-grid atomic workaround
  的约 `1.5e-3` 重复浮点差未达到默认推广条件；
- `/root/vllm` 未修改，用户已有改动未回退。

若任何条件失败，保持 default-off，记录精确 blocker、复现命令和首个差异位置。

## 最终报告必须回答

1. 4x C500 相对 4x A100 的总差距从多少缩小到多少？
2. 每个阶段贡献了多少真实 ms/token，而不是百分比推测？
3. 剩余差距属于 kernel、带宽、互联、软件栈还是硬件上限？
4. MTP 当前状态为何是 Deferred，恢复工作需要满足哪些前置条件？
5. 哪些 opt-in 可以默认推广，哪些必须保留回退和显式告警？

## 会话任务提示

```text
完成 deepseek-v4-c500-e2e/07-integration-acceptance.md。收集各子会话的真实
artifact，按 Sparse MLA→MHC→O-proj→MoE→collective→graph 顺序逐项集成并重跑
门禁。当前只推进 MTP=0 的稳定 100-token A100/C500 同口径对比和 baseline 优化；
MTP 自 2026-07-25 起为 Deferred，不属于本轮推广范围。
不得用 microbenchmark 或 profiler 吞吐代替端到端 normal inference。
```
