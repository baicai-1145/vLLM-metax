# 共享基线、证据和验收契约

## 任务目标

为所有性能子会话冻结同一套 A100/C500 对比条件、正确性门禁和日志格式。任何
单点优化只有在本契约下通过，才可以进入端到端集成。

## 不可改变的条件

- 工作区：`/root/vLLM-metax`
- CUDA 参考代码：`/root/vllm-0.25rc1`，只读
- 禁止修改：`/root/vllm`
- 模型：`/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-MTP`
- 验收 TP：4；TP=1 只允许做归因实验
- 第一阶段关闭 MTP：`NUM_SPECULATIVE_TOKENS=0`
- decode：greedy，`temperature=0`
- 16-token oracle：
  `[260,5036,294,10588,14,790,342,2118,436,734,260,1894,5090,304,611,260]`
- 正常性能必须用 PIECEWISE graph；eager 只做差异归因
- 不得把 profiler 下的吞吐当成正常推理吞吐
- 不得通过降低精度、改变模型输出或减少层数获得性能结果

## 当前基线

```text
TP=4, MTP=0, PIECEWISE
GENERATED_TOKENS=16
GENERATE_SECONDS=1.376383
OUTPUT_TOKENS_PER_SECOND=11.624672
```

以上 16-token 数字是**历史快速 gate**，不得与新 workload 混报。最新同口径证据为：

| workload | 结果 | 备注 |
| -------- | ---- | ---- |
| MTP=0、TP=4、PIECEWISE、100-token decode | `15.2334 tok/s`、`65.65 ms/token` | 四卡利用率 `16.16%--16.42%` |
| 1K prefill | `1321.61 input tok/s`、`0.774812 s` | MTP=0、TP=4 |
| 10K prefill、`chunk=8192`、default | `torch.index_select` 尝试约 `10 GiB` 后 OOM | `torch_flash_mla_sparse_prefill` |
| 10K prefill、`chunk=2048`、`GPU_MEM=0.8` | `1504.16 input tok/s`、median `6.648215 s` | 四卡利用率约 `89.1%--89.2%` |

正式 decode 性能基线固定为 100 个 token、三个 warmup request 和五次测量，报告
median/p90 和方差；16-token 只用于快速 greedy correctness gate，不能作为性能结果。
prefill 必须同时保留 1K 和 10K 两个长度。10K default-chunk
OOM 是当前 blocker，chunk=2048 只是可复现的 workaround，不能把 OOM 从验收矩阵中
删除或推迟到 decode 完成后。

## 已知 blocker 与实现状态

- 已安装 `deep_gemm.int8_mqa_logits` 没有 `backend` kwarg；动态 multi-head Triton
  loop 会触发 `mcTriton SmallVector`。当前仓库 3D head-grid atomic workaround 已
  通过 differential、graph replay 和 TP=4 16-token gate，但重复浮点差约 `1.5e-3`，
  未达到默认推广条件。
- MTP=1 虽有 `17.8358 tok/s`，却从第二个 token 起与 greedy oracle 分歧；MTP=1
  必须 default-off，且任何分歧都 hard fail，不得以吞吐抵消错误。

## 必须建立的 A100/C500 矩阵

| 变量   | 值                     |
| ------ | ---------------------- |
| 设备   | 4x A100、4x C500       |
| TP     | 1（诊断）、4（验收）   |
| 图模式 | eager、PIECEWISE       |
| 阶段   | prefill、decode        |
| MTP    | 0                      |
| batch  | 1；其他 batch 另表报告 |

两个平台必须使用同一 checkpoint、prompt、最大长度、sampling、vLLM API 行为和
输出 token。若软件版本不同，记录 commit、环境和 CUDA/MACA 后端差异，禁止把
不同实现路径的结果直接归因于硬件。

## 反馈循环

快速 TP=4 gate：

```bash
MAX_TOKENS=16 \
EXPECTED_TOKEN_IDS='[260,5036,294,10588,14,790,342,2118,436,734,260,1894,5090,304,611,260]' \
NUM_SPECULATIVE_TOKENS=0 \
ENFORCE_EAGER=0 \
CUDAGRAPH_MODE=PIECEWISE \
./tools/run_deepseek_v4_mtp_generate.sh
```

Profiler 固定为五个 steady steps：

```bash
PROFILE_DIR=/tmp/dsv4_session_profile \
PROFILE_DELAY_ITERATIONS=4 \
PROFILE_MAX_ITERATIONS=5 \
PROFILE_ACTIVE_ITERATIONS=5 \
PROFILE_IGNORE_FRONTEND=1 \
NUM_SPECULATIVE_TOKENS=0 \
./tools/run_deepseek_v4_mtp_generate.sh
```

每次性能结果必须绑定同一份 workload manifest（checkpoint、prompt、TP、MTP、图
模式、chunk、GPU memory、warmup、重复次数和 commit），并将 normal benchmark 与
profiler 分开保存。

## 角色与 artifact 契约

| 角色 | 允许产出 | 必须保留的 artifact |
| ---- | -------- | ------------------- |
| `benchmark-runner` | 不变 workload 的 normal throughput、latency、四卡利用率 | 原始 stdout/stderr、命令与环境、workload manifest、每次测量和汇总 median/p90 |
| `profiler` | steady-state trace、host/device gap、阶段/kernel/launch 归因 | 原始 trace、profile log、时间窗口和 wall time；不得把 profiler 吞吐当 normal 吞吐 |
| `kernel-validator` | 真实输入 differential、graph capture/replay、稳定指针、隔离 kernel 微基准 | corpus/shape/dtype/stride/mask、oracle 与 tolerance、token IDs、差异位置、dispatch/graph 日志 |

artifact 路径必须在会话摘要中列出且可复现；缺少原始 artifact、workload manifest 或
首个差异位置时，结果只能记为未验收。kernel-validator 的微基准不得替代
benchmark-runner 的端到端结果。

## 正确性门禁

每个新 kernel 必须依次通过：

1. 真实输入 direct differential；BF16 输出要求 bitwise，FP32 中间值记录 ULP；
2. 至少两组不同 payload 的 CUDA graph capture/replay；
3. 输入和 caller-owned 输出 `data_ptr` 稳定；
4. focused pytest、`py_compile`、`git diff --check`；
5. TP=4 eager 2-token；
6. TP=4 eager 16-token；
7. TP=4 PIECEWISE 16-token；
8. 正常 100-token 性能和同口径 steady profiler。
9. MTP=1 若被测试，第二个 token 起的 greedy oracle 必须完全一致；否则立即 hard fail
   并保持 default-off。

## 证据规则

- 文件名含 `tilelang`、`flashmla` 或 `exact` 不是 dispatch 证据。
- 必须看到 kernel 名、显式 dispatch counter、NVTX range 或源代码路径证据。
- vLLM 对未知插件环境变量的 warning 不代表变量被删除；用实际 dispatch 验证。
- profiler 表的父子行和多 stream 时间不能相加；同时报告 wall time。
- C500 的 BF16/TF32/INT4 峰值、HBM 带宽和互联规格缺少权威公开数据，禁止猜测。

## 会话结束模板

```text
Changed files:
Correctness corpus/result:
Graph replay/result:
TP=4 token IDs:
Normal median/p90 ms/token:
Profiler stage delta:
Kernel launch delta:
Dispatch evidence:
Promotion status:
Residual risks:
Next dependent session:
```
