# Repository agent instructions

## Repository mission

This repository adapts and optimizes vLLM for MetaX GPUs. The active program is
DeepSeek-V4-Flash-W4A16-BF16Attn-MTP inference on four C500 GPUs.

The current optimization policy is:

1. Optimize TP=4 decode throughput. The MTP=0 decode baseline is `27.2 TPS`
   (`36.8 ms/token`) with four-card SM utilization of only `30.3%`. After FULL
   graph + fused sinkhorn/mhc_pre_norm landed (2026-08-01), the new baseline is
   `30.18 TPS` (`35.8%` SM), `+10.9%`. **The bottleneck is a mix of host
   scheduling (CPU-GPU gaps) and low-grid-count kernel implementation
   (torch aten fallback kernels with `grid=[1,1,1]`), NOT pure GPU starvation
   or single-kernel compute speed.** Three independent experiments confirm
   this:
   - `fused_sinkhorn` cut per-step device time 57ms→30ms (-47%) but TPS only
     +3.5% → device-kernel-time reduction does NOT translate to TPS under
     PIECEWISE (host scheduling is the binding constraint there).
   - FULL graph (attention captured in-graph) gave +8.4% TPS → host-launch
     gaps for the ~616 out-of-graph attention kernels ARE a real lever.
   - Per-kernel occupancy: 76% of device time is on `occupancy ≤24%` kernels;
   NV reaches 99% SM on single-request decode of the same model, so the 36%
   gap is software-implementation (kernel grid mapping), not workload shape.
   The earlier "HBM bandwidth below 3%" claim is **retracted** — hardware
   counters (`mx-smi --show-hbm-bandwidth`) show peak ~12% (640 GB/s of
   5.3 TB/s), not 3%. Primary levers, in order: (a) FULL graph coverage
   (productionize), (b) torch-aten fallback elimination via large-grid fusion
     (valid only under FULL graph), (c) async scheduling. See
   [`13-main-model-decode-handoff.md`](docs/superpowers/plans/deepseek-v4-c500-e2e/13-main-model-decode-handoff.md)
   for the full diagnosis, profiler evidence, and ranked optimization
   directions.
2. Fix blocking prefill correctness and memory failures early. In particular,
   the 10K sparse-MLA prefill OOM is not an optional throughput improvement.
3. Optimize prefill throughput after the blocking memory path is removed.
4. DSpark speculative decoding is **suspended pending official W4A16 DSpark
   weights**. DeepSeek released `deepseek-ai/DeepSeek-V4-Flash-0731` with
   native DSpark support; community W4A16 quantized variants are expected. The
   current `DSpark-staging` checkpoint has a `38%` arena-hard accept rate because
   the draft was trained on MXFP4 target distribution but deployed against a
   W4A16 INT4 target. Do not attempt to fix accept rate through engineering;
   wait for correctly quantized weights.

Treat
[`docs/superpowers/plans/deepseek-v4-c500-e2e/`](docs/superpowers/plans/deepseek-v4-c500-e2e/)
as the authoritative source for current baselines, blockers, workload details,
task plans, and acceptance gates. Update those documents when new evidence
supersedes an existing result.

## Hard constraints

- Acceptance is TP=4. TP=1 may be used only for explicitly labeled diagnosis.
- Treat `/root/vllm-0.25rc1` as a read-only upstream source tree. Implement all
  product changes in `/root/vLLM-metax`.
- Do not reduce TP, context length, output length, chunk size, memory settings,
  graph mode, or workload difficulty to make a result pass.
- Do not silently use Torch arithmetic, eager execution, a different backend,
  or any other fallback when a native kernel is required.
- A fallback may exist only when explicitly requested, observable in dispatch
  evidence, and excluded from native-kernel acceptance measurements.
- Do not revert, reset, unstage, overwrite, or reformat unrelated user changes.
- Preserve the dirty worktree and work with existing changes in files that the
  task must touch.
- Keep MTP off by default. A throughput improvement with divergent greedy token
  IDs is a correctness failure, not a valid optimization.
- Keep normal throughput separate from profiler-instrumented throughput.
  Profiler TPS is not a normal serving baseline.
- Do not claim exact, bitwise, graph-safe, fallback-free, or faster without
  current evidence from the relevant validation gate.
- Never call MetaX Cross IPC v1. In particular,
  `mcIpcGetMemHandleCross` and `mcIpcOpenMemHandleCross` are forbidden in
  product code, probes, tests, benchmarks, and one-off diagnosis. Cross IPC v1
  can hard-hang in the driver, freeze the entire container, and ignore timeout
  and SIGTERM; recovery may require restarting the container.
- Any MetaX Cross IPC implementation must use
  `mcIpcGetMemHandleCross_v2` and `mcIpcOpenMemHandleCross_v2` with
  `mcIpcCrossMemHandle_t`. Treat wrappers or compatibility aliases as v1 and
  forbidden unless their implementation is verified to dispatch to the v2
  APIs. Do not run even a single-process allocation/open smoke test through
  Cross IPC v1.

## Runtime environment and default commands

Run repository commands from `/root/vLLM-metax`. Use the repository virtual
environment and platform environment before invoking Python, vLLM, tests that
need MACA libraries, or benchmark scripts:

```bash
cd /root/vLLM-metax
source .venv/bin/activate
source ./env.sh
```

`env.sh` accepts an optional MACA root argument and defaults to `/opt/maca`. It
sets the MACA/cu-bridge paths, required dynamic-library paths, and
`VLLM_INSTALL_PUNICA_KERNELS=1`. Do not replace this setup with a system Python
or manually assembled library path unless diagnosing the environment itself.

The normal offline DeepSeek-V4 runner is:

```bash
./tools/run_deepseek_v4_mtp_generate.sh
```

Its defaults are the model
`/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-MTP`, `TP=4`, `GPU_MEM=0.9`,
`ENFORCE_EAGER=0`, `CUDAGRAPH_MODE=PIECEWISE`, and
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`. The generator defaults to `MAX_MODEL_LEN=512`,
`MAX_TOKENS=100`, prefix caching enabled, and `NUM_SPECULATIVE_TOKENS=0`.
All normal performance measurements must explicitly use `MAX_TOKENS=100` so
that the result represents intermediate steady-state decode rather than a
short-run first-token-biased measurement. `MAX_TOKENS=16` is reserved for the
explicit fast greedy correctness gate and must not be reported as performance.
Set every workload-affecting variable explicitly in a benchmark command rather
than relying on defaults. In particular, record `MAX_TOKENS`, `INPUT_TOKENS`,
`MAX_NUM_BATCHED_TOKENS`, `ENABLE_PREFIX_CACHING`, `WARMUP_REQUESTS`,
`BENCH_RUNS`, and `NUM_SPECULATIVE_TOKENS` whenever they apply.

Do not call `tools/tmp_deepseek_v4_mtp_generate.py` directly for acceptance
unless the wrapper's environment setup has been reproduced. Its standalone
defaults differ from the wrapper and are not the TP=4 benchmark defaults.

### C500 local evidence

Before MetaX kernel, graph, communication, or memory optimization, consult the
local reference and hardware evidence first:

- `/root/metax-reference/README.md` indexes the downloaded public MetaX/MACA
  manuals and source snapshots.
- `artifacts/hardware/c500_maca_3.7.2/README.md` summarizes the current local
  hardware characterization.
- `artifacts/hardware/c500_maca_3.7.2/manifest.json` is the machine-readable
  source of truth for per-GPU HBM, P2P, topology, MCCL, graph, atomic, and BF16
  row-execution evidence. Use its raw-log paths and `SHA256SUMS` when auditing
  a result.

The hardware manifest applies only to the recorded environment: MACA 3.7.2.0,
driver 3.8.23, PyTorch 2.8.0+metax3.5.3.9, and the current four-C500 host
topology. Re-run the characterization suite after changing the driver, MACA
SDK, PyTorch build, firmware, topology, power mode, GPU placement, or any
relevant runtime library. Do not carry old hardware values into a changed
environment without revalidation.

Keep hardware timing layers separate:

- Official MCCL perf timing measures the collective implementation directly.
- vLLM/PYNCCL timing includes wrapper, dispatch, event, and integration costs.
- Model traces additionally include producer arrival skew, scheduling, graph
  gaps, and critical-path dependencies.

Do not attribute a model-level wait directly to MCCL from only one of these
layers. Check the per-GPU and directed P2P matrices for a slow card or
asymmetric link before attributing a TP=4 regression to model code. Likewise,
do not compare TransferBench, standard peer-copy, official MCCL perf, and
vLLM/PYNCCL values as if they were the same measurement path.

P2P characterization may use only single-process standard peer-copy or an
official MetaX tool. Collective characterization must prove the intended
backend (`MCCLLibrary` for the vLLM/PYNCCL path) and retain correctness evidence.
Required evidence includes the exact payload and direction matrix, warmups and
repetitions, data checks or `#wrong=0`, raw logs, structured summaries, and
integrity hashes. A bandwidth number without those fields is not accepted.

Treat graph stability and mathematical equivalence as separate properties. A
stable ordinary `M=6` graph does not establish equality with the `6 x M=1`
oracle. Atomic reductions are not assumed deterministic; every production
atomic path requires repeated-run and graph-replay evidence at its actual
shape and dtype.

The Cross IPC v1 prohibition in the hard constraints overrides every local or
downloaded sample. Never adapt a sample or hardware probe to a forbidden v1
allocation or memory-handle path.

### Package installation and mirrors

Use the active virtual environment for package installation. The existing pip
configuration is `/root/.config/pip/pip.conf`; inspect it before installing or
changing packages. At the time of writing it configures a Huawei Cloud PyPI
index, not a USTC mirror. Repository installation uses:

```bash
python use_existing_metax.py
pip install -r requirements/build.txt
pip install . --no-build-isolation
```

For editable development installation, use:

```bash
pip install -v -e . --no-build-isolation
```

Do not change the pip index, trusted hosts, private MetaX package indexes, or
install CUDA/PyTorch replacements without an explicit task. If a USTC mirror
is required, update the actual pip configuration in a separate, explicit
environment change and verify package resolution; do not merely document an
unconfigured mirror in an agent instruction.

## Agent routing

Use the narrowest named agent whenever the runtime exposes it through
`agent_type`. Give agents bounded tasks, explicit file boundaries, prohibitions,
and the required compact output format. The parent agent owns final technical
decisions and evidence integration.

### AI-infra skill routing

- For DeepSeek-V4 main-model optimization, use `model-pr-history-knowledge`
  before selecting a new architecture, kernel, fusion, graph, or scheduling
  direction. Treat PR history as hypothesis input only; current C500 traces,
  native differential results, and frozen TP=4 evidence remain authoritative.
  Focus the search on decode graph coverage, kernel fusion, and
  CPU-GPU overlap rather than single-kernel micro-optimization.
- DSpark-specific skills (`vllm-sota-humanize-loop`, `llm-serving-auto-benchmark`)
  are not active while DSpark is suspended; see the DSpark track section.
- `vllm-sota-humanize-loop` may be used only as an optimization-loop
  methodology. Its cross-framework deployment search must not change the
  frozen model, backend, dtype, TP, graph mode, workload, or acceptance
  contract. Repository-specific correctness and evidence gates override it.
- Do not use `llm-serving-auto-benchmark` for acceptance measurements.
  It compares frameworks and deployment commands, while this track requires an
  unchanged vLLM/MetaX workload. Use it only for a separately requested and
  separately labeled cross-framework experiment.
- Use `llm-serving-capacity-planner` only for HBM, KV-cache capacity,
  concurrency, or OOM attribution. It does not establish decode bottlenecks or
  end-to-end throughput acceptance.
- Use `model-architecture-diagram` only when an architecture diagram is
  explicitly needed; diagrams are not implementation or performance evidence.
- Use `model-pr-diff-dossier` only when producing or revising a PR-history
  document that cites specific upstream PRs.
- Skills provide workflow and domain guidance; they do not replace the required
  `profiler`, `kernel-validator`, `benchmark-runner`, `test-runner`, and
  `reviewer` agents or their evidence gates.

### Retrieval and investigation agents

- Use `searcher` for a symbol, config, test, definition, or usage location that
  can be answered with a small number of focused searches.
- Use `doc-reader` for Markdown, README, design documents, plans, long logs, and
  other local text that would otherwise consume substantial parent context.
- Use `explorer` for cross-file call chains, backend selection, state flow,
  module boundaries, runtime shapes, dispatch, and behavioral attribution.
- Use `researcher` for external documentation, upstream implementations,
  version differences, release notes, and current authoritative facts.
- Use `git-historian` for blame, regression windows, commit provenance, and
  comparisons with earlier repository states.
- Use `dependency-auditor` only for manifests, lockfiles, versions, licenses,
  advisories, and supply-chain questions.
- Use reverse-engineering roles only when binary or IDA evidence is genuinely
  required; they are not substitutes for source inspection.

### Implementation and validation agents

- Use `worker` for a bounded implementation with explicit file ownership.
- Use `test-runner` for pytest, lint, formatting checks, type checks, builds,
  and ordinary non-performance correctness gates.
- Use `reviewer` for correctness, regression, numerical stability, concurrency,
  data safety, graph safety, and missing-test review.
- Do not delegate final architecture choices, destructive operations, or the
  final decision that an optimization is acceptable.

### End-to-end benchmark agent

- Always delegate model-loading throughput, latency, TTFT, TP scaling, and
  per-GPU utilization measurements to `benchmark-runner` first.
- Preserve model, prompt, token counts, TP, MTP, graph mode, prefix caching,
  chunk size, GPU memory utilization, warmups, repetitions, and measurement
  window exactly.
- `benchmark-runner` keeps raw output in artifact files and returns only a
  concise error or median/P90 TPS with per-GPU average utilization and paths.
- Do not reproduce complete model-loading logs in the parent context unless a
  failed benchmark must be diagnosed.
- A benchmark that changed workload parameters is a different experiment and
  must not be compared directly with the original baseline.

### Profiler agent

- Always delegate trace collection, operator timing, host/device gap analysis,
  communication attribution, and baseline/candidate trace comparison to
  `profiler` first.
- Preserve the assigned workload and steady-state profile window.
- Keep raw traces and tables in artifacts. Return ranked hotspots, one
  evidence-backed bottleneck conclusion, requested deltas, and artifact paths.
- Distinguish host time, device time, communication, synchronization, memory
  movement, and graph gaps. Do not naively sum overlapping asynchronous events.
- `profiler` does not implement fixes or make the final optimization decision.

### Profiler attribution method (mandatory)

Every profiler-based bottleneck conclusion must use the method below. A
single one of these errors previously produced a wrong "GPU starvation"
diagnosis that wasted a full optimization cycle, so treat each rule as a
hard gate, not advice.

1. **Wall time is defined by the host/device timeline overlap, never by
   device kernel time alone.** In every steady-state decode step, first
   reconstruct the full overlap structure: (a) the device stream busy
   interval (union of kernel intervals), (b) the host-side blocking
   synchronization calls (`mcEventSynchronize` / event waits on MetaX), and
   (c) the gaps between them. Report device-busy %, host-blocked %, and
   true-idle %. A conclusion stated as "X% of wall time" is only valid
   when X is measured against the wall window, not against device kernel
   time.
2. **Distinguish GPU-bound from CPU/GPU-starved before choosing a
   direction.** The decisive test is what the device stream is doing
   *during* the longest host-side `mcEventSynchronize` of the step: if the
   device is executing kernels throughout that window, the step is
   GPU-bound (the CPU is correctly waiting on a full device); if the
   device is idle, it is launch/starvation-bound. These demand opposite
   fixes. Never label a step "starved" while a blocking sync overlaps a
   busy device.
3. **Do not sum per-kernel micro-gaps and call it "GPU idle".** With
   ~15000 tiny kernels per decode step, there are ~15000 sub-microsecond
   inter-kernel bubbles; their sum looks large but represents scheduler
   jitter inside a continuously busy device, not macroscopic idle. Only
   gaps that fall *outside* a host-side blocking-sync window (i.e. where
   the device is idle and the host is NOT blocked on it) count as
   recoverable idle.
4. **Kernel-count share is not wall-time share, and not even device-time
   share.** A function can own 44% of kernels and <2% of device time
   (small-tensor sinkhorn). Always attribute by *aggregated kernel
   execution time*, grouped by normalized kernel name and by functional
   category, then rank. Counting kernels is only a launch-overhead proxy.
5. **In PIECEWISE / full CUDA-graph decode, the graph absorbs launch
   overhead, NOT kernel execution time.** Fusing many tiny graph-captured
   kernels into fewer kernels removes ~3-5 us of launch overhead each, but
   leaves their execution time. Theoretical speedups computed from
   eager-mode device time or profiler device-time share overestimate the
   e2e win by 10-60x under graph mode. Before trusting a predicted
   speedup, measure the target function's device time *and* its e2e delta;
   if they disagree by more than ~3x, the graph is absorbing the overhead
   and the function is not the real bottleneck.
6. **Isolated microbenchmark latency is never the e2e prediction.** A
   10x isolated kernel speedup that the graph already amortizes may yield
   <5% e2e. Always confirm any "this kernel is the bottleneck" claim
   with an unchanged-workload TP=4 PIECEWISE e2e measurement before
   committing an optimization direction.
7. **Re-validate the attribution after every optimization.** When an
   e2e result is much smaller than the profiler prediction, the most
   likely cause is that the attribution (not the optimization) was wrong.
   Re-read the trace with this method before assuming the fix failed or
   searching for a new bottleneck.

### Kernel validator agent

- Always delegate GPU-kernel differential correctness, graph capture/replay,
  repeated-run stability, and isolated microbenchmarks to `kernel-validator`
  first.
- Validate production shapes, dtypes, strides, layouts, scales, masks, real
  captured inputs when available, and relevant non-aligned boundary shapes.
- Benchmark a kernel only after correctness passes, unless reproducing a
  failure is the explicit task.
- Use warmups, device synchronization or device events, sufficient repetitions,
  and median/P90 latency.
- Hidden fallback, dispatch mismatch, compilation failure, token mismatch,
  graph failure, or tolerance failure makes validation fail.
- Never treat isolated-kernel speedup as end-to-end speedup.

## Required optimization workflow

For every substantial decode, prefill, communication, graph, or kernel change:

1. Freeze the workload and success criteria from the shared contract.
2. Locate the real call path, dispatch decision, shapes, dtypes, layouts, and
   graph behavior.
3. Establish or reuse a trusted oracle and a red-capable differential harness.
4. Use `kernel-validator` for kernel differential, boundary, stability, graph,
   and isolated-latency gates when a kernel changes.
5. Use `test-runner` for relevant project tests and static checks.
6. Run the TP=4 greedy-token correctness gate with the required normal graph
   mode. TP=1 results do not satisfy acceptance.
7. Use `profiler` when attribution or before/after trace evidence is required.
8. Use `benchmark-runner` for unchanged-workload normal throughput, latency,
   and per-GPU utilization.
9. Use `reviewer` before declaring a major optimization complete.
10. Update the relevant plan, evidence, blocker, and artifact references.

Do not skip correctness because performance improved. Do not optimize a stage
identified only by intuition when a current trace can test the hypothesis.

## Workload and evidence contract

Every reported performance result must record or inherit an unambiguous manifest
containing:

- model and repository state;
- TP, MTP, speculative-token count, and graph mode;
- prompt source, exact input length, and exact output length;
- prefix-cache setting, prefill chunk size, maximum model length, and GPU memory
  utilization;
- warmup count, measured repetitions, and timing window;
- normal or profiler-instrumented execution;
- fallback and dispatch status;
- median and P90 latency or throughput when samples support them;
- per-GPU utilization sampling interval and cropped benchmark window;
- raw log, trace, corpus, and summary artifact paths.

Performance comparisons must use the same workload manifest. If a setting must
change to avoid an OOM or compiler failure, report the original failure and
label the successful configuration as a separate workload.

Correctness evidence must include the applicable subset of:

- direct oracle comparison with stated absolute and relative tolerances;
- bitwise comparison only when explicitly required and actually achieved;
- real captured inputs and production shapes;
- non-aligned and boundary shapes;
- NaN/Inf and output-buffer checks;
- graph capture plus repeated replay;
- stable pointer and allocation behavior;
- dispatch evidence showing the intended native path;
- TP=4 greedy token IDs against the frozen oracle.

## Current blocker policy

- **Main-model decode is a mixed host-scheduling + kernel-implementation
  bottleneck, NOT pure GPU starvation.** The MTP=0 FULL-graph baseline is now
  `30.18 TPS` (`35.8%` SM). The prior "GPU-starved, 70% wall-time idle,
  ~5.4 GB weights/token" diagnosis is **retracted** — it was based on an
  unverified weight-read estimate and wall-time (not peak-window) bandwidth
  denominator. Hardware counters show peak HBM ~12% (640 GB/s of 5.3 TB/s),
  not 3%. Verified facts (2026-08-01): (1) device-kernel-time reduction does
  not translate to TPS under PIECEWISE (fused_sinkhorn: device -47%, TPS
  +3.5%); (2) FULL graph (attention in-graph) gives +8.4% TPS; (3) 76% of
  device time is on occupancy ≤24% kernels (torch aten fallback, grid=[1,1,1]).
  Levers ranked: FULL graph coverage → torch-aten large-grid fusion (valid
  only under FULL) → async scheduling. Use TPS (not profiler device time) as
  the acceptance signal, and profiler gap analysis + occupancy jointly for
  attribution.
- The long-prefill Torch sparse-MLA gather OOM is a blocking memory defect.
  Smaller chunks are a separately labeled workaround, not the final fix.
- The MetaX INT8 sparse-indexer workaround must retain differential, graph, and
  greedy-token gates; atomic reduction variability remains an explicit risk.
- MTP remains default-off. DSpark is suspended pending W4A16 weights (see
  DSpark track below).
- Historical profiler percentages and TPS remain useful only when labeled with
  their original workload. Do not apply them to the latest baseline without a
  new measurement. In particular, DSpark `target_accept` kernel-time breakdowns
  (M=6 batched) do not represent MTP=0 M=1 decode.

## DeepSeek-V4 DSpark track (suspended)

- DSpark speculative decoding is **suspended**. The `DSpark-staging` checkpoint
  achieves `41.6 TPS` (1.54x over MTP=0) but with only `38%` arena-hard accept
  rate. Root cause is confirmed: the draft model was trained on MXFP4 target
  distribution, but the deployed target uses W4A16 INT4 quantization, causing
  systematic hidden-state bias. This is not fixable through engineering; it
  requires correctly quantized DSpark weights.
- DeepSeek released `deepseek-ai/DeepSeek-V4-Flash-0731` with native DSpark
  support. Wait for community W4A16+DSpark quantized weights before resuming.
- `/root/models/DeepSeek-V4-Flash-DSpark` is a module source, not a complete
  model. It intentionally retains only official shards 46-48 containing
  `mtp.0`, `mtp.1`, and `mtp.2`, plus small metadata. Do not load it as a
  standalone model or redownload the base shards unless explicitly requested.
- `/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-DSpark-staging` is an
  experimental merged checkpoint. Do not overwrite
  `/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-MTP` with it or treat it as
  accepted without current TP=4 evidence.
- Established DSpark facts (do not re-investigate):
    - DSpark forces V2 Model Runner (`vllm/v1/worker/gpu/model_runner.py`), not
      V1. Spec config `method == "dspark"` triggers this in
      `vllm/config/vllm.py`.
    - `max_num_seqs=1` is correct for DSpark; concurrent batching was tested and
      rejected (slower, different tokens).
    - The batched path (`tokenwise=0`) is the accepted production path.
    - Draft PIECEWISE CG patch (`vllm_metax/patch/performance/dspark_piecewise_cg.py`)
      reduced draft forward from `7.5ms` to `2.9ms`; draft is no longer the
      bottleneck.
    - Per-position conditional accept rates: `pos0=72.6%, pos1=68.9%,
      pos2=67.3%, pos3=63.4%, pos4=58.2%`. The decay is gradual, confirming the
      mismatch is learnable systematic bias, not fundamental.
- DSpark acceptance requires the unchanged target model, TP=4, exact final
  greedy token IDs against MTP=0, stable graph replay, and observable native
  dispatch. Do not report DSpark speedup before these gates pass.
- Inspect the dirty worktree before DSpark changes. Preserve existing merger,
  test, and auxiliary-hidden-state work rather than regenerating or overwriting
  it.

## Editing and worktree rules

- Use `apply_patch` for manual file edits.
- Default to ASCII in code unless the file already uses another character set
  and the content requires it. All edited text files must remain UTF-8 without
  BOM.
- Add comments only where they explain non-obvious behavior or constraints.
- Make surgical changes and avoid unrelated refactoring, formatting, dependency
  updates, generated files, or metadata churn.
- Never run destructive Git commands such as `git reset --hard` or
  `git checkout --` on user work.
- Do not amend commits unless explicitly requested.
- Treat unfamiliar modifications as user work. Ignore unrelated changes and
  integrate with relevant ones.
- Keep long-running benchmark, profiler, or build sessions under control and do
  not finish while required sessions are still running.

## Verification and completion

- Use project-native tests, linters, type checks, build commands, smoke tests,
  differential harnesses, profiler tools, and benchmark scripts appropriate to
  the changed surface.
- Run `git diff --check` for edited tracked files and `markdownlint-cli2` for
  edited Markdown when available.
- A successful process exit is not sufficient when a shell pipeline can mask
  the underlying benchmark or worker exit code.
- Before completion, verify that required artifacts exist and contain the
  expected summary markers.
- Final reports state what changed, verification commands and results, impact
  scope, remaining risks, fallback status, and artifact paths.
- If verification cannot run, state exactly what is missing and do not claim the
  gate passed.
