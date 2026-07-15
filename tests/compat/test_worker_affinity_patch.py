import importlib
from types import SimpleNamespace

import pytest


MODULE_NAME = "vllm_metax.patch.plugin_enhancement.worker_affinity"


def _load_patch(monkeypatch, value):
    monkeypatch.setenv("VLLM_METAX_TP_WORKER_CPU_AFFINITY", value)
    return importlib.reload(importlib.import_module(MODULE_NAME))


def _config(worker_cls="auto"):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls=worker_cls),
        compilation_config=None,
        model_config=None,
        scheduler_config=SimpleNamespace(
            is_multimodal_model=False,
            disable_chunked_mm_input=False,
        ),
        attention_config=None,
    )


def test_platform_selects_affinity_worker_only_when_enabled(monkeypatch):
    from vllm_metax.platform import MacaPlatform

    monkeypatch.delenv("VLLM_METAX_TP_WORKER_CPU_AFFINITY", raising=False)
    config = _config()
    MacaPlatform.check_and_update_config(config)
    assert config.parallel_config.worker_cls == "vllm.v1.worker.gpu_worker.Worker"

    monkeypatch.setenv("VLLM_METAX_TP_WORKER_CPU_AFFINITY", "96-119,130")
    config = _config()
    MacaPlatform.check_and_update_config(config)
    assert (
        config.parallel_config.worker_cls
        == "vllm_metax.patch.plugin_enhancement.worker_affinity.AffinityWorker"
    )


def test_affinity_worker_applies_before_super_init_device(monkeypatch):
    module = _load_patch(monkeypatch, "96-98,130;140-141")
    events = []

    monkeypatch.setattr(
        module.os,
        "sched_setaffinity",
        lambda pid, cpus: events.append(("affinity", pid, cpus)),
    )

    def super_init_device(self):
        events.append("super")
        return "initialized"

    monkeypatch.setattr(module.Worker, "init_device", super_init_device)
    worker = object.__new__(module.AffinityWorker)
    worker.local_rank = 1

    assert worker.init_device() == "initialized"
    assert events == [("affinity", 0, {140, 141}), "super"]


def test_affinity_parser_supports_ranges_and_fails_closed(monkeypatch):
    module = _load_patch(monkeypatch, "96-98,130;140-141")
    assert module._cpus_for_rank("96-98,130;140-141", 0) == {
        96,
        97,
        98,
        130,
    }
    with pytest.raises(ValueError):
        module._cpus_for_rank("96-98", 1)
    with pytest.raises(ValueError):
        module._cpus_for_rank("96-", 0)


def test_disabled_affinity_worker_leaves_default_affinity(monkeypatch):
    module = importlib.reload(importlib.import_module(MODULE_NAME))
    monkeypatch.delenv("VLLM_METAX_TP_WORKER_CPU_AFFINITY", raising=False)
    calls = []
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda *args: calls.append(args))
    module._apply_affinity(0)
    assert calls == []
