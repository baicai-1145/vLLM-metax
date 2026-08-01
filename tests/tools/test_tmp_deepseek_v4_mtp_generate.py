import pytest

from tools import tmp_deepseek_v4_mtp_generate as generate


def test_validate_expected_runs_checks_every_replay():
    with pytest.raises(SystemExit, match="run=1"):
        generate._validate_expected_runs(
            [[10, 11], [10, 99], [10, 11]],
            [10, 11],
        )


def test_validate_expected_runs_accepts_identical_replays():
    generate._validate_expected_runs(
        [[10, 11], [10, 11], [10, 11]],
        [10, 11],
    )


def test_validate_expected_run_matrix_checks_each_prompt_oracle():
    with pytest.raises(SystemExit, match="run=1"):
        generate._validate_expected_run_matrix(
            [[10, 11], [20, 99]],
            [[10, 11], [20, 21]],
        )


def test_parse_prompt_texts_requires_one_prompt_per_replay():
    with pytest.raises(ValueError, match="BENCH_RUNS=3"):
        generate._parse_prompt_texts_json('["short", "long"]', bench_runs=3)


def test_parse_expected_run_token_ids_requires_a_matrix():
    with pytest.raises(ValueError, match="JSON list of token-ID lists"):
        generate._parse_expected_run_token_ids_json("[10, 11]")


def test_parse_run_max_tokens_requires_one_positive_value_per_replay():
    with pytest.raises(ValueError, match="positive integer"):
        generate._parse_run_max_tokens_json("[100, 0]", bench_runs=2)


def test_validate_min_tokens_rejects_non_positive_values():
    with pytest.raises(ValueError, match="MIN_TOKENS must be at least 1"):
        generate._validate_min_tokens(0, [100])


def test_validate_min_tokens_rejects_run_max_tokens_overflow():
    with pytest.raises(ValueError, match="run=0 max_tokens=100"):
        generate._validate_min_tokens(101, [100, 200])


def test_validate_min_tokens_accepts_every_run_limit():
    generate._validate_min_tokens(100, [100, 200])


def test_serialize_run_token_ids_is_compact_json():
    assert generate._serialize_run_token_ids([[10, 11], [20]]) == "[[10,11],[20]]"


def test_async_scheduling_override_is_unset_by_default(monkeypatch):
    monkeypatch.delenv("ASYNC_SCHEDULING", raising=False)

    assert generate._async_scheduling_override() == {}


def test_async_scheduling_override_parses_explicit_false(monkeypatch):
    monkeypatch.setenv("ASYNC_SCHEDULING", "0")

    assert generate._async_scheduling_override() == {"async_scheduling": False}


def test_async_scheduling_override_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("ASYNC_SCHEDULING", "false")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        generate._async_scheduling_override()


def test_log_stats_override_is_unset_by_default(monkeypatch):
    monkeypatch.delenv("DISABLE_LOG_STATS", raising=False)

    assert generate._log_stats_override() == {}


def test_log_stats_override_enables_native_metrics(monkeypatch):
    monkeypatch.setenv("DISABLE_LOG_STATS", "0")

    assert generate._log_stats_override() == {"disable_log_stats": False}


def test_log_stats_override_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("DISABLE_LOG_STATS", "false")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        generate._log_stats_override()


def test_build_speculative_config_supports_explicit_dspark(monkeypatch):
    monkeypatch.setenv("SPECULATIVE_METHOD", "dspark")
    monkeypatch.setenv("SPECULATIVE_MODEL", "/models/deepseek-v4-dspark")

    assert generate._build_speculative_config(4) == {
        "method": "dspark",
        "model": "/models/deepseek-v4-dspark",
        "num_speculative_tokens": 4,
        "draft_sample_method": "greedy",
    }


def test_profiler_config_enables_record_shapes_explicitly(monkeypatch):
    monkeypatch.setenv("PROFILE_RECORD_SHAPES", "1")

    config = generate._build_profiler_config("/tmp/traces")

    assert config is not None
    assert config["torch_profiler_record_shapes"] is True


def test_print_bench_marker_includes_epoch_ns_without_changing_legacy_line(
    monkeypatch, capsys
):
    monkeypatch.setattr(generate.time, "time_ns", lambda: 1_785_146_789_123_456_789)

    generate._print_bench_marker("DECODE", "START")

    assert capsys.readouterr().out.splitlines() == [
        "DECODE_BENCH_START",
        "DECODE_BENCH_START_UNIX_NS 1785146789123456789",
    ]
