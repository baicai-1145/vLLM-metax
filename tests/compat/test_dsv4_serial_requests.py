from types import SimpleNamespace


def _config(architecture: str, max_num_seqs: int = 8):
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=[architecture]),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 4],
            max_cudagraph_capture_size=4,
        ),
    )


def test_dsv4_requests_are_serialized_by_default(monkeypatch):
    from vllm_metax.platform import _enforce_dsv4_serial_requests

    monkeypatch.delenv("VLLM_METAX_DSV4_ALLOW_UNSAFE_BATCHING", raising=False)
    config = _config("DeepseekV4ForCausalLM")

    assert _enforce_dsv4_serial_requests(config)
    assert config.scheduler_config.max_num_seqs == 1
    assert config.compilation_config.cudagraph_capture_sizes == [1]
    assert config.compilation_config.max_cudagraph_capture_size == 1


def test_dsv4_unsafe_batching_requires_explicit_opt_in(monkeypatch):
    from vllm_metax.platform import _enforce_dsv4_serial_requests

    monkeypatch.setenv("VLLM_METAX_DSV4_ALLOW_UNSAFE_BATCHING", "1")
    config = _config("DeepseekV4ForCausalLM")

    assert not _enforce_dsv4_serial_requests(config)
    assert config.scheduler_config.max_num_seqs == 8
    assert config.compilation_config.cudagraph_capture_sizes == [1, 2, 4]
    assert config.compilation_config.max_cudagraph_capture_size == 4


def test_dsv4_serial_guard_also_filters_graph_sizes(monkeypatch):
    from vllm_metax.platform import _enforce_dsv4_serial_requests

    monkeypatch.delenv("VLLM_METAX_DSV4_ALLOW_UNSAFE_BATCHING", raising=False)
    config = _config("DeepseekV4ForCausalLM", max_num_seqs=1)

    assert _enforce_dsv4_serial_requests(config)
    assert config.compilation_config.cudagraph_capture_sizes == [1]
    assert config.compilation_config.max_cudagraph_capture_size == 1


def test_dsv4_serial_guard_limits_graph_max_without_explicit_sizes(monkeypatch):
    from vllm_metax.platform import _enforce_dsv4_serial_requests

    monkeypatch.delenv("VLLM_METAX_DSV4_ALLOW_UNSAFE_BATCHING", raising=False)
    config = _config("DeepseekV4ForCausalLM", max_num_seqs=1)
    config.compilation_config.cudagraph_capture_sizes = None

    assert _enforce_dsv4_serial_requests(config)
    assert config.compilation_config.cudagraph_capture_sizes is None
    assert config.compilation_config.max_cudagraph_capture_size == 1


def test_serial_request_guard_does_not_change_other_models(monkeypatch):
    from vllm_metax.platform import _enforce_dsv4_serial_requests

    monkeypatch.delenv("VLLM_METAX_DSV4_ALLOW_UNSAFE_BATCHING", raising=False)
    config = _config("Qwen3ForCausalLM")

    assert not _enforce_dsv4_serial_requests(config)
    assert config.scheduler_config.max_num_seqs == 8
