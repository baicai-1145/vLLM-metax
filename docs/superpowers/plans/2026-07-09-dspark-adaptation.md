# vLLM 0.25rc1 MetaX DSpark Adaptation Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/root/vLLM-metax` run as the MetaX backend plugin for `/root/vllm-0.25rc1`, then validate upstream DSpark with `/root/models/Qwen3-8B` and `/root/models/dspark_qwen3_8b_block7`.

**Architecture:** Treat `/root/vllm-0.25rc1` as the upstream vLLM package and keep DSpark implementation in upstream, not reimplemented in vLLM-metax. Update vLLM-metax version binding, dependency/install flow, model overrides, platform registration, attention backends, custom ops, and monkey patches so the plugin cleanly loads against vLLM 0.25rc1; only add MetaX-specific compatibility patches where 0.25rc1 APIs or kernels need them.

**Tech Stack:** vLLM `v0.25.0rc1` from `/root/vllm-0.25rc1`, vLLM-metax plugin from `/root/vLLM-metax`, Python 3.12 venv `/root/vLLM-metax/.venv`, MACA SDK `/opt/maca`, torch `2.8.0+metax3.5.3.9`, mcoplib precompiled kernels, MetaX C500 64GB.

## Global Constraints

- Do not modify `/root/vllm` v0.23 for this work; it is no longer the active upstream target.
- Use `/root/vllm-0.25rc1` as the vLLM source tree and installed package.
- Keep upstream DSpark code in `/root/vllm-0.25rc1`; vLLM-metax should adapt around it.
- Do not wholesale replace vLLM-metax with upstream vLLM files.
- Keep production default `USE_PRECOMPILED_KERNEL=1` unless a task explicitly proves a local custom op must be rebuilt.
- Before runtime tests, use:
  ```bash
  source /root/vLLM-metax/.venv/bin/activate
  source /root/vLLM-metax/env.sh
  TORCH_LIB="$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")"
  export LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH}
  ```
- The first supported DSpark scope is Qwen3 only:
  - target: `/root/models/Qwen3-8B`
  - draft: `/root/models/dspark_qwen3_8b_block7`
  - draft config confirmed: `architectures=["Qwen3DSparkModel"]`, `model_type="qwen3"`, `block_size=7`, `markov_rank=256`, `vocab_size=151936`.
- Treat DeepSeek-V4 DSpark as a later milestone unless Qwen3 DSpark already passes.
- Preserve existing user worktree changes; current dirty files include `vllm_metax/patch/bugfix/deepseek_v4/__init__.py` and untracked `AGENTS.md`.
- Final acceptance requires both:
  - inference quality is unchanged versus non-speculative Qwen3-8B baseline under the same prompts/sampling settings;
  - DSpark acceptance metrics reach the target thresholds defined in the "Final DSpark Acceptance Criteria" section.

---

## Corrected Direction

The previous interpretation was wrong: this is not a backport of DSpark from 0.25rc1 into v0.23. The correct direction is:

```text
/root/vllm-0.25rc1        # upstream vLLM package, already contains DSpark
        +
/root/vLLM-metax          # MetaX platform/custom ops/model/patch plugin
        =
vLLM 0.25rc1 + MetaX backend + DSpark
```

That means implementation should focus on:

1. installing `/root/vllm-0.25rc1` into `.venv` with `VLLM_TARGET_DEVICE=empty`;
2. updating vLLM-metax metadata/version assumptions from `0.23.0` to `0.25.0rc1`;
3. making vLLM-metax plugin imports and monkey patches compatible with 0.25rc1;
4. validating the built-in upstream DSpark path on MetaX.

---

## File Structure

### Existing files likely to modify

- `setup.py`
  - Change `fixed_version_scheme()` from `0.23.0` to a 0.25rc1-compatible plugin version.
  - Re-check dependency discovery and build assumptions against `/root/vllm-0.25rc1`.
- `requirements/maca.txt`
  - Reconcile current MetaX torch/triton/mcoplib pins with 0.25rc1 requirements.
- `requirements/common.txt`
  - Compare with `/root/vllm-0.25rc1/requirements/common.txt` and avoid downgrading packages needed by 0.25rc1.
- `vllm_metax/__init__.py`
  - Validate platform/general/model entry points still match 0.25rc1 plugin loading.
- `vllm_metax/platform.py`
  - Validate `MacaPlatform` against 0.25rc1 platform API.
  - Validate attention backend registration API against 0.25rc1.
- `vllm_metax/models/__init__.py`
  - Remove or update registrations that conflict with upstream 0.25rc1 model names.
  - Keep MetaX-specific model overrides only where needed.
- `vllm_metax/patch/__init__.py`
- `vllm_metax/patch/bugfix/__init__.py`
- `vllm_metax/patch/bugfix/triton_support/__init__.py`
  - Gate or remove stale 0.23 monkey patches that break 0.25rc1.
- `vllm_metax/patch/bugfix/triton_support/*.py`
  - Revalidate each patch target symbol against 0.25rc1 before importing.
- `vllm_metax/patch/performance/speculative_decode_perf.py`
  - Revalidate against 0.25rc1 speculative decode internals.
- `vllm_metax/patch/bugfix/deepseek_v4/torch_fix.py`
  - Revalidate internal compiler/accelerator monkey patches against 0.25rc1.
- `vllm_metax/patch/plugin_enhancement/distributed/pynccl_wrapper.py`
- `vllm_metax/patch/plugin_enhancement/distributed/cuda_wrapper.py`
  - Revalidate distributed communication monkey patches against 0.25rc1.
- `vllm_metax/models/deepseek_v4/model.py`
  - Revalidate imports from upstream `vllm.models.deepseek_v4.nvidia.ops.*`.
- `env.sh`
  - Review stale variables such as `VLLM_INSTALL_PUNICA_KERNELS`.
- `vllm_metax/v1/attention/backends/*.py`
  - Revalidate backend classes and metadata structures against 0.25rc1.
- `vllm_metax/v1/attention/backends/mla/*.py`
  - Revalidate only if DeepSeek-V4/MTP tests are in scope.
- `vllm_metax/models/deepseek_v4/*`
  - Revalidate after Qwen3 DSpark is green; do not block first DSpark milestone unless imports fail.

### New files to create

- `tests/compat/test_vllm_025rc1_plugin_import.py`
  - Fast test that imports vLLM 0.25rc1 and loads vLLM-metax entry points.
- `tests/compat/test_vllm_025rc1_patch_targets.py`
  - Fast test that each monkey patch target exists before patching.
- `tests/v1/e2e/spec_decode/test_dspark_qwen3_metax.py`
  - Local hardware smoke for Qwen3 DSpark, skipped unless local model paths and MetaX device are available.
- `docs/superpowers/plans/2026-07-09-dspark-adaptation.md`
  - This plan.

---

## Task 1: Build a Clean 0.25rc1 Installation Baseline

**Files:**
- Read: `/root/vllm-0.25rc1/setup.py`
- Read: `/root/vllm-0.25rc1/pyproject.toml`
- Read: `/root/vLLM-metax/setup.py`
- Read: `/root/vLLM-metax/requirements/maca.txt`

**Interfaces:**
- Produces: a reproducible installed baseline where `python -c "import vllm; print(vllm.__version__)"` resolves to `/root/vllm-0.25rc1`, not `/root/vllm`.

**Steps:**

- [ ] Activate environment and load MACA:
  ```bash
  source /root/vLLM-metax/.venv/bin/activate
  source /root/vLLM-metax/env.sh
  TORCH_LIB="$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")"
  export LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH}
  ```
- [ ] Record current installed vLLM:
  ```bash
  python - <<'PY'
  import inspect, vllm
  print("version:", getattr(vllm, "__version__", None))
  print("path:", inspect.getfile(vllm))
  PY
  ```
- [ ] Install upstream 0.25rc1 in editable empty-target mode:
  ```bash
  cd /root/vllm-0.25rc1
  VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation
  ```
- [ ] Confirm Python imports `/root/vllm-0.25rc1/vllm/__init__.py`.
- [ ] Confirm DSpark files are importable:
  ```bash
  python - <<'PY'
  import importlib
  for name in [
      "vllm.config.speculative",
      "vllm.v1.worker.gpu.spec_decode.dspark.speculator",
      "vllm.model_executor.models.qwen3_dspark",
  ]:
      print(name, importlib.import_module(name).__file__)
  PY
  ```

**Verification:**
- Import path output points to `/root/vllm-0.25rc1`.
- DSpark imports do not fail before vLLM-metax is installed.

---

## Task 2: Update vLLM-metax Version Binding

**Files:**
- Modify: `setup.py`
- Optional modify: `README.md`
- Optional modify: `AGENTS.md`

**Interfaces:**
- Produces: vLLM-metax package metadata clearly indicates it targets vLLM `0.25.0rc1`.

**Steps:**

- [ ] Change `fixed_version_scheme()` in `setup.py`:
  ```python
  def fixed_version_scheme(version: ScmVersion) -> str:
      return "0.25.0rc1"
  ```
- [ ] Do not change MACA local-version suffix behavior in `get_plugin_version()`.
- [ ] Run:
  ```bash
  python setup.py --version
  ```
- [ ] Expected output begins with `0.25.0rc1+...` or `0.25.0rc1....` and includes the MACA/torch local suffix.
- [ ] Update docs that currently say vLLM-metax is `v0.23.0-dev` only after the install path is proven.

**Verification:**
- `python setup.py --version`

---

## Task 3: Install vLLM-metax Against `/root/vllm-0.25rc1`

**Files:**
- Read/modify as needed:
  - `requirements/maca.txt`
  - `requirements/common.txt`
  - `setup.py`

**Interfaces:**
- Produces: editable vLLM-metax install that loads with vLLM 0.25rc1.

**Steps:**

- [ ] Compare dependency versions:
  ```bash
  python - <<'PY'
  from pathlib import Path
  for p in [
      "/root/vLLM-metax/requirements/maca.txt",
      "/root/vLLM-metax/requirements/common.txt",
      "/root/vllm-0.25rc1/requirements/common.txt",
  ]:
      print("\\n---", p)
      print(Path(p).read_text()[:4000])
  PY
  ```
- [ ] Keep MetaX torch/triton/flash-attn/mcoplib pins unless 0.25rc1 import errors prove a conflict.
- [ ] Install plugin:
  ```bash
  cd /root/vLLM-metax
  USE_PRECOMPILED_KERNEL=1 pip install -e . --no-build-isolation
  ```
- [ ] Confirm package paths:
  ```bash
  python - <<'PY'
  import inspect, vllm, vllm_metax
  print("vllm:", inspect.getfile(vllm))
  print("vllm_metax:", inspect.getfile(vllm_metax))
  PY
  ```

**Verification:**
- `vllm` path is `/root/vllm-0.25rc1/vllm/...`.
- `vllm_metax` path is `/root/vLLM-metax/vllm_metax/...`.

---

## Task 4: Add Plugin Import and Entry Point Regression Tests

**Files:**
- Create: `tests/compat/test_vllm_025rc1_plugin_import.py`

**Interfaces:**
- Produces: fast red/green signal for plugin loading against vLLM 0.25rc1.

**Steps:**

- [ ] Add this test:
  ```python
  import inspect

  import vllm
  import vllm_metax


  def test_vllm_is_025rc1_checkout():
      assert "/root/vllm-0.25rc1/" in inspect.getfile(vllm)


  def test_vllm_metax_register_functions_import():
      assert vllm_metax.register() == "vllm_metax.platform.MacaPlatform"
      assert callable(vllm_metax.register_customized)
      assert callable(vllm_metax.register_model)
  ```
- [ ] Run:
  ```bash
  pytest -q tests/compat/test_vllm_025rc1_plugin_import.py
  ```

**Verification:**
- Test passes after Tasks 1-3.

---

## Task 5: Revalidate Platform API Against 0.25rc1

**Files:**
- Modify as needed: `vllm_metax/platform.py`
- Reference: `/root/vllm-0.25rc1/vllm/platforms/interface.py`
- Reference: `/root/vllm-0.25rc1/vllm/platforms/cuda.py`
- Reference: `/root/vllm-0.25rc1/vllm/v1/attention/backends/registry.py`

**Interfaces:**
- Produces: `MacaPlatform` imports and registers attention backends under 0.25rc1.

**Steps:**

- [ ] Run:
  ```bash
  python - <<'PY'
  from vllm_metax.platform import MacaPlatform
  print(MacaPlatform)
  print(getattr(MacaPlatform, "device_type", None))
  PY
  ```
- [ ] If import fails, diff the failing upstream API:
  ```bash
  diff -u /root/vllm/vllm/platforms/interface.py /root/vllm-0.25rc1/vllm/platforms/interface.py | sed -n '1,240p'
  ```
- [ ] Update only the broken methods or signatures in `vllm_metax/platform.py`.
- [ ] Run the import again.

**Verification:**
- `from vllm_metax.platform import MacaPlatform` succeeds.

---

## Task 6: Gate Stale Monkey Patches Before Importing Them

**Files:**
- Modify: `vllm_metax/patch/bugfix/__init__.py`
- Modify: `vllm_metax/patch/bugfix/triton_support/__init__.py`
- Modify as needed:
  - `vllm_metax/patch/bugfix/triton_support/eagle.py`
  - `vllm_metax/patch/bugfix/triton_support/rejection_sampler.py`
  - `vllm_metax/patch/performance/speculative_decode_perf.py`

**Interfaces:**
- Produces: vLLM-metax patch import succeeds under 0.25rc1 even when a 0.23-only patch target moved.

**Steps:**

- [ ] Create `tests/compat/test_vllm_025rc1_patch_targets.py` with:
  ```python
  import importlib


  def test_patch_package_imports_against_vllm_025rc1():
      importlib.import_module("vllm_metax.patch")
  ```
- [ ] Run:
  ```bash
  pytest -q tests/compat/test_vllm_025rc1_patch_targets.py
  ```
- [ ] For each import failure, classify the patch:
  - still needed and target moved: update imports/signature;
  - already fixed upstream 0.25rc1: skip importing it;
  - unrelated to Qwen3 DSpark: gate behind a version check and defer.
- [ ] Implement version gates with a small helper rather than broad try/except swallowing:
  ```python
  from packaging.version import Version
  import vllm

  VLLM_VERSION = Version(vllm.__version__.split("+", 1)[0])
  IS_VLLM_025_PLUS = VLLM_VERSION >= Version("0.25.0rc1")
  ```
- [ ] Re-run patch import test.
- [ ] Explicitly include these high-risk patches in the import test path:
  - `vllm_metax.patch.bugfix.deepseek_v4.torch_fix`
  - `vllm_metax.patch.bugfix.triton_support.eagle`
  - `vllm_metax.patch.bugfix.triton_support.rejection_sampler`
  - `vllm_metax.patch.plugin_enhancement.distributed.pynccl_wrapper`
  - `vllm_metax.patch.plugin_enhancement.distributed.cuda_wrapper`

**Verification:**
- `pytest -q tests/compat/test_vllm_025rc1_patch_targets.py`

---

## Task 7: Reconcile Model Registry Overrides

**Files:**
- Modify: `vllm_metax/models/__init__.py`
- Reference: `/root/vllm-0.25rc1/vllm/model_executor/models/registry.py`

**Interfaces:**
- Produces: MetaX-specific model registrations do not mask upstream 0.25rc1 DSpark Qwen3 registrations incorrectly.

**Steps:**

- [ ] Check whether upstream already registers `Qwen3DSparkModel`:
  ```bash
  rg -n '"Qwen3DSparkModel"|DSparkDraftModel|DFlashDraftModel' /root/vllm-0.25rc1/vllm/model_executor/models/registry.py
  ```
- [ ] Do not register `Qwen3DSparkModel` in vLLM-metax unless upstream registration fails on MetaX.
- [ ] Keep MetaX overrides for DeepSeek/MiMo/MTP only where they are still required.
- [ ] Add a smoke:
  ```bash
  python - <<'PY'
  import vllm_metax
  vllm_metax.register_model()
  from vllm import ModelRegistry
  print("model registry loaded")
  PY
  ```

**Verification:**
- Model registration smoke exits 0.

---

## Task 8: Verify Built-In Upstream DSpark Imports With Plugin Loaded

**Files:**
- No source change unless import fails.

**Interfaces:**
- Produces: upstream DSpark implementation remains importable after vLLM-metax patches load.

**Steps:**

- [ ] Run:
  ```bash
  python - <<'PY'
  import vllm_metax
  vllm_metax.register_customized()
  vllm_metax.register_model()

  import importlib
  for name in [
      "vllm.config.speculative",
      "vllm.v1.worker.gpu.spec_decode.dspark.speculator",
      "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
      "vllm.model_executor.models.qwen3_dspark",
      "vllm.model_executor.models.qwen3_dflash",
  ]:
      mod = importlib.import_module(name)
      print(name, mod.__file__)
  PY
  ```
- [ ] If a vLLM-metax patch breaks upstream DSpark import, fix that patch before editing DSpark code.

**Verification:**
- Every printed module path points to `/root/vllm-0.25rc1`.

---

## Task 9: Check DeepSeek-V4 Override Imports Do Not Break Qwen3 Startup

**Files:**
- Read/modify if needed: `vllm_metax/models/deepseek_v4/model.py`
- Reference: `/root/vllm-0.25rc1/vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`
- Reference: `/root/vllm-0.25rc1/vllm/models/deepseek_v4/nvidia/ops/*`

**Interfaces:**
- Produces: vLLM-metax model registration can run under 0.25rc1 without failing on DeepSeek-V4-only imports.

**Steps:**

- [ ] Run:
  ```bash
  python - <<'PY'
  import vllm_metax
  vllm_metax.register_model()
  import vllm_metax.models.deepseek_v4.model as m
  print(m.__file__)
  PY
  ```
- [ ] If this fails before Qwen3 DSpark can start, make DeepSeek-V4-only imports lazy or update the import path to the 0.25rc1 API.
- [ ] Do not attempt DeepSeek-V4 DSpark correctness in this task.
- [ ] Re-run the import smoke.

**Verification:**
- DeepSeek-V4 override imports do not block plugin model registration.

---

## Task 10: Constructor Smoke for Qwen3 DSpark

**Files:**
- Create: `tests/v1/e2e/spec_decode/test_dspark_qwen3_metax.py`

**Interfaces:**
- Produces: first hardware-aware smoke that builds an `LLM` with DSpark config.

**Steps:**

- [ ] Add a pytest that skips if model paths are missing:
  ```python
  from pathlib import Path

  import pytest


  TARGET = Path("/root/models/Qwen3-8B")
  DRAFT = Path("/root/models/dspark_qwen3_8b_block7")


  @pytest.mark.skipif(not TARGET.exists() or not DRAFT.exists(), reason="local DSpark models missing")
  def test_qwen3_dspark_constructor_metax():
      from vllm import LLM

      llm = LLM(
          model=str(TARGET),
          trust_remote_code=True,
          dtype="bfloat16",
          max_model_len=2048,
          enforce_eager=True,
          max_num_seqs=1,
          speculative_config={
              "method": "dspark",
              "model": str(DRAFT),
              "num_speculative_tokens": 7,
              "draft_sample_method": "greedy",
          },
      )
      assert llm is not None
  ```
- [ ] Run:
  ```bash
  pytest -q tests/v1/e2e/spec_decode/test_dspark_qwen3_metax.py::test_qwen3_dspark_constructor_metax -s
  ```
- [ ] If constructor fails, fix plugin/platform/patch compatibility first.

**Verification:**
- Constructor test passes or produces a concrete MetaX compatibility traceback.

---

## Task 11: Eager Greedy DSpark Generation Smoke

**Files:**
- Modify: `tests/v1/e2e/spec_decode/test_dspark_qwen3_metax.py`

**Interfaces:**
- Produces: non-empty generated output from upstream DSpark on MetaX.

**Steps:**

- [ ] Add:
  ```python
  @pytest.mark.skipif(not TARGET.exists() or not DRAFT.exists(), reason="local DSpark models missing")
  def test_qwen3_dspark_generate_greedy_metax():
      from vllm import LLM, SamplingParams

      llm = LLM(
          model=str(TARGET),
          trust_remote_code=True,
          dtype="bfloat16",
          max_model_len=2048,
          enforce_eager=True,
          max_num_seqs=1,
          speculative_config={
              "method": "dspark",
              "model": str(DRAFT),
              "num_speculative_tokens": 7,
              "draft_sample_method": "greedy",
          },
      )
      out = llm.generate(
          ["Briefly explain speculative decoding."],
          SamplingParams(max_tokens=32, temperature=0),
      )
      assert out[0].outputs[0].text.strip()
  ```
- [ ] Run the single test.
- [ ] If memory pressure occurs, lower `max_model_len` to `1024` for the test but keep serving docs at `8192` only after validated.

**Verification:**
- Test passes with non-empty output.

---

## Task 12: Validate DSpark Serve Command

**Files:**
- Optional modify: `AGENTS.md`
- Optional create: `docs/features/speculative_decoding/dspark_metax.md`

**Interfaces:**
- Produces: OpenAI-compatible server using `/root/vllm-0.25rc1` + vLLM-metax + DSpark.

**Steps:**

- [ ] Start server:
  ```bash
  vllm serve /root/models/Qwen3-8B \
    --max-model-len 8192 \
    --enforce-eager \
    --trust-remote-code \
    --dtype bfloat16 \
    --port 8000 \
    --speculative-config '{
      "method": "dspark",
      "model": "/root/models/dspark_qwen3_8b_block7",
      "num_speculative_tokens": 7,
      "draft_sample_method": "greedy"
    }'
  ```
- [ ] In another shell, send:
  ```bash
  curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "/root/models/Qwen3-8B",
      "messages": [{"role":"user","content":"用一句话解释 speculative decoding"}],
      "max_tokens": 32,
      "temperature": 0
    }' | jq .
  ```
- [ ] Confirm response content is non-empty.
- [ ] Stop the server cleanly.

**Verification:**
- curl response has non-empty `choices[0].message.content`.

---

## Task 13: Add Acceptance-Rate and Quality Regression Evaluation

**Files:**
- Create: `tests/v1/e2e/spec_decode/test_dspark_qwen3_acceptance_metax.py`
- Optional create: `tools/batched_test/configs/dspark_qwen3_acceptance.yaml`
- Reference: `/root/vllm-0.25rc1/tests/v1/e2e/spec_decode/test_spec_decode.py`
- Reference: `/root/vllm-0.25rc1/tests/v1/spec_decode/test_speculators_correctness.py`

**Interfaces:**
- Produces: final measurable pass/fail signal for DSpark quality and acceptance metrics.

**Steps:**

- [ ] Implement a helper equivalent to vLLM's metric extraction:
  ```python
  def compute_spec_decode_stats(metrics) -> dict:
      name2metric = {m.name: m for m in metrics}
      n_drafts = name2metric["vllm:spec_decode_num_drafts"].value
      n_draft_tokens = name2metric["vllm:spec_decode_num_draft_tokens"].value
      n_accepted = name2metric["vllm:spec_decode_num_accepted_tokens"].value
      per_pos_vec = name2metric["vllm:spec_decode_num_accepted_tokens_per_pos"].values
      acceptance_len = 1 + (n_accepted / n_drafts) if n_drafts > 0 else 1.0
      overall_acceptance_rate = (
          n_accepted / n_draft_tokens if n_draft_tokens > 0 else 0.0
      )
      per_pos_rates = [v / n_drafts for v in per_pos_vec] if n_drafts > 0 else []
      return {
          "num_drafts": n_drafts,
          "num_draft_tokens": n_draft_tokens,
          "num_accepted_tokens": n_accepted,
          "acceptance_len": acceptance_len,
          "overall_acceptance_rate": overall_acceptance_rate,
          "per_pos_acceptance_rates": per_pos_rates,
      }
  ```
- [ ] Build two `LLM` instances using identical target model and sampling settings:
  - baseline: no `speculative_config`;
  - DSpark: `method="dspark"`, draft `/root/models/dspark_qwen3_8b_block7`, `num_speculative_tokens=7`.
- [ ] Use deterministic prompts first with `temperature=0`, then a probabilistic GSM8K-style subset with fixed seed if probabilistic mode is enabled.
- [ ] For deterministic prompts, assert generated text is identical or token-equivalent after normalization.
- [ ] For benchmark prompts, assert task accuracy is not lower than baseline beyond a small tolerance:
  - exact-match style tasks: DSpark accuracy >= baseline accuracy - 1 percentage point;
  - open-ended chat smoke: manual/non-empty sanity only, not acceptance-rate gating.
- [ ] Assert acceptance metrics using the thresholds below.
- [ ] Print metrics in the test log:
  ```text
  acceptance_len
  overall_acceptance_rate
  per_pos_acceptance_rates
  num_drafts
  num_draft_tokens
  num_accepted_tokens
  ```

**Verification:**
- `pytest -q tests/v1/e2e/spec_decode/test_dspark_qwen3_acceptance_metax.py -s`

---

## Task 14: Re-enable Probabilistic and CUDA Graph Only After Greedy Eager Works

**Files:**
- Touch only if failures prove needed:
  - `vllm_metax/patch/bugfix/triton_support/rejection_sampler.py`
  - `vllm_metax/patch/performance/speculative_decode_perf.py`
  - `vllm_metax/v1/attention/backends/flash_attn.py`

**Interfaces:**
- Produces optional support for:
  - `draft_sample_method="probabilistic"`
  - DSpark without `--enforce-eager`

**Steps:**

- [ ] Run generation with `draft_sample_method="probabilistic"`.
- [ ] If it fails inside rejection sampler, compare vLLM-metax patch against 0.25rc1 sampler before editing.
- [ ] Run generation without `--enforce-eager`.
- [ ] If CUDA graph capture fails, record the exact failure and keep eager documented as the supported mode.
- [ ] Benchmark baseline Qwen3 vs DSpark only after correctness is stable.

**Verification:**
- Probabilistic mode either passes or is documented unsupported with traceback.
- CUDA graph mode either passes or is documented unsupported with traceback.

---

## First Success Criteria

The first real success is:

```bash
python - <<'PY'
import inspect, vllm, vllm_metax
print("vllm:", inspect.getfile(vllm))
print("vllm_metax:", inspect.getfile(vllm_metax))
PY
```

prints `/root/vllm-0.25rc1`, and:

```bash
pytest -q tests/v1/e2e/spec_decode/test_dspark_qwen3_metax.py::test_qwen3_dspark_generate_greedy_metax -s
```

generates non-empty text using DSpark.

## Final DSpark Acceptance Criteria

The final handoff is not complete until quality and acceptance metrics both pass.

### Quality criteria

- DSpark must preserve the target model output distribution.
- For `temperature=0` deterministic smoke prompts, DSpark output should match the non-speculative baseline exactly or be token-equivalent after tokenizer-normalized comparison.
- For benchmark/eval subsets, DSpark accuracy must not regress materially:
  - exact-match/math/code subsets: DSpark score >= baseline score - 1 percentage point;
  - if full benchmark runtime is too high, run a fixed representative subset and document sample size.
- Non-empty output alone is only a bring-up milestone, not final acceptance.

### Acceptance-rate criteria for `/root/models/dspark_qwen3_8b_block7`

Use vLLM metrics:

```text
acceptance_len = 1 + accepted_tokens / num_drafts
overall_acceptance_rate = accepted_tokens / draft_tokens
```

For block7, the conversion from accepted length to draft-token acceptance rate is:

```text
overall_acceptance_rate = (acceptance_len - 1) / 7
```

Target thresholds:

- GSM8K/math-style validation:
  - acceptance length >= 5.5
  - overall acceptance rate >= 0.66
- Mixed-task validation:
  - acceptance length >= 4.1
  - overall acceptance rate >= 0.44
- Chat-only prompts are not allowed to fail the whole validation by the math/code threshold; expected chat acceptance can be much lower, around 0.32-0.39 from public DSpark data.

Reference public DSpark Qwen3-8B block7 accepted lengths:

```text
GSM8K      6.17  -> rate 0.739
MATH       5.78  -> rate 0.683
AIME25     5.01  -> rate 0.573
MBPP       5.16  -> rate 0.594
HumanEval  5.52  -> rate 0.646
LCB        5.17  -> rate 0.596
MT-Bench   3.72  -> rate 0.389
Alpaca     3.58  -> rate 0.369
Arena-Hard 3.21  -> rate 0.316
```

The 9-task macro average is accepted length 4.81, equivalent to overall acceptance rate 0.545. Use this only for broad mixed-task reporting; do not use it to judge a single chat-only smoke.

## Residual Risks

- vLLM-metax monkey patches are version-sensitive; some 0.23 patches may actively break 0.25rc1 and should be gated or removed.
- vLLM 0.25rc1 may require dependency versions that conflict with current MetaX torch/triton/mcoplib pins.
- Upstream DSpark uses non-causal DFlash-style attention; MetaX attention backend support must be validated on real C500 hardware.
- `torch_fix.py` patches internal Torch/vLLM compiler APIs; this must be validated with 0.25rc1 rather than assumed compatible.
- vLLM 0.25rc1 includes upstream DeepSeek-V4; vLLM-metax DeepSeek-V4 overrides may import upstream NVIDIA helper APIs that changed.
- Distributed communication monkey patches may need 0.25rc1 signature updates before multi-card DSpark is attempted.
- `env.sh` may contain stale v0.23-era variables such as `VLLM_INSTALL_PUNICA_KERNELS`; clean only after verifying they are unused.
- CUDA graph and probabilistic rejection sampling are performance/quality hardening tasks, not the first correctness milestone.
- DeepSeek-V4 DSpark is out of first scope even though upstream 0.25rc1 contains it.
