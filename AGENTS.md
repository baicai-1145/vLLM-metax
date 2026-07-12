# Repository agent instructions

## Repository mission

This repository adapts and optimizes vLLM for MetaX GPUs. The active program is
DeepSeek-V4-Flash-W4A16-BF16Attn-MTP inference on four C500 GPUs.

The current optimization policy is:

1. Optimize TP=4 decode first because normal 100-token decode has low GPU
   utilization and dominates common conversational workloads.
2. Fix blocking prefill correctness and memory failures early. In particular,
   the 10K sparse-MLA prefill OOM is not an optional throughput improvement.
3. Optimize prefill throughput after the blocking memory path is removed.
4. Keep MTP disabled until speculative decoding produces the exact greedy
   token sequence required by the acceptance contract.

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

- Low decode GPU utilization is a primary optimization signal, not proof that a
  particular kernel is responsible. Use profiler evidence to rank stages.
- The long-prefill Torch sparse-MLA gather OOM is a blocking memory defect.
  Smaller chunks are a separately labeled workaround, not the final fix.
- The MetaX INT8 sparse-indexer workaround must retain differential, graph, and
  greedy-token gates; atomic reduction variability remains an explicit risk.
- MTP remains default-off until speculative verification reproduces the exact
  base greedy sequence under TP=4.
- Historical profiler percentages and TPS remain useful only when labeled with
  their original workload. Do not apply them to the latest baseline without a
  new measurement.

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
