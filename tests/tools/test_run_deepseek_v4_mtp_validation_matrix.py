import json
from pathlib import Path

from tools.debug import run_deepseek_v4_mtp_validation_matrix as matrix


REAL_INFERENCE_CORPUS = (
    Path(__file__).parents[2]
    / "tools/debug/corpora/deepseek_v4_real_inference_4.jsonl"
)
REGRESSION_CORPUS = (
    Path(__file__).parents[2]
    / "tools/debug/corpora/deepseek_v4_mtp_regression_3.jsonl"
)
HELDOUT_CORPUS = (
    Path(__file__).parents[2]
    / "tools/debug/corpora/deepseek_v4_mtp_heldout_3_20260723.jsonl"
)


def test_default_candidate_keeps_exact_q_and_wq_b_paths():
    env = matrix.DEFAULT_CANDIDATE_ENV

    assert env["VLLM_METAX_DSV4_TOKENWISE_Q_ONLY"] == "1"
    assert env["VLLM_METAX_DSV4_TOKENWISE_COMPRESSOR"] == "1"
    assert "VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE" not in env
    assert "VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS" not in env


def test_parse_generate_log_extracts_tokens_and_timing(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "GEN_OK",
                "FINISH_REASON 'length'",
                "TOKEN_IDS [1, 2, 3]",
                "GENERATED_TOKENS 3",
                "GENERATE_SECONDS 0.500000",
                "OUTPUT_TOKENS_PER_SECOND 6.000000",
                "DECODE_RUN_SECONDS [0.5]",
                "TOKEN_IDS_MATCH_EXPECTED",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = matrix.parse_generate_log(log)

    assert result["token_ids"] == [1, 2, 3]
    assert result["finish_reason"] == "length"
    assert result["generated_tokens"] == 3
    assert result["output_tps"] == 6.0
    assert result["token_ids_match_expected"] is True


def test_parse_generate_log_extracts_three_run_matrix(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "RUN_TOKEN_IDS_JSON [[1,2],[3],[4,5]]",
                "RUN_FINISH_REASONS ['length', 'stop', 'length']",
                "RUN_MAX_TOKENS [100, 100, 100]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = matrix.parse_generate_log(log)

    assert result["run_token_ids"] == [[1, 2], [3], [4, 5]]
    assert result["run_finish_reasons"] == ["length", "stop", "length"]
    assert result["run_max_tokens"] == [100, 100, 100]


def test_parse_generate_log_ignores_shutdown_text_after_token_matrix(
    tmp_path: Path,
):
    log = tmp_path / "run.log"
    log.write_text(
        "RUN_TOKEN_IDS_JSON [[1,2],[3]]"
        "(EngineCore pid=123) INFO shutdown\n",
        encoding="utf-8",
    )

    result = matrix.parse_generate_log(log)

    assert result["run_token_ids"] == [[1, 2], [3]]


def test_real_inference_corpus_covers_four_fixed_domains():
    prompts = matrix.load_prompt_cases(REAL_INFERENCE_CORPUS)

    assert [prompt.prompt_id for prompt in prompts] == [
        "real_factual_explanation",
        "real_math_reasoning",
        "real_code_debugging",
        "real_document_synthesis",
    ]
    assert all(len(prompt.text) >= 120 for prompt in prompts)


def test_quality_corpora_are_three_prompt_and_disjoint():
    regression = matrix.load_prompt_cases(REGRESSION_CORPUS)
    heldout = matrix.load_prompt_cases(HELDOUT_CORPUS)

    matrix.validate_prompt_cases(regression, label="regression")
    matrix.validate_prompt_cases(heldout, label="heldout")
    matrix.validate_disjoint_prompt_cases(regression, heldout)

    assert len(regression) == 3
    assert len(heldout) == 3
    assert {prompt.prompt_id for prompt in regression}.isdisjoint(
        prompt.prompt_id for prompt in heldout
    )


def test_compare_tokens_reports_first_mismatch_and_windows():
    result = matrix.compare_tokens([10, 11, 12, 13], [10, 99, 12], window_radius=1)

    assert result["exact"] is False
    assert result["tokens_exact"] is False
    assert result["first_mismatch"] == 1
    assert result["expected_window"] == [10, 11, 12]
    assert result["actual_window"] == [10, 99, 12]


def test_compare_tokens_fails_on_finish_reason_mismatch():
    result = matrix.compare_tokens(
        [10, 11],
        [10, 11],
        window_radius=1,
        expected_finish_reason="length",
        actual_finish_reason="stop",
    )

    assert result["exact"] is False
    assert result["tokens_exact"] is True
    assert result["finish_reason_match"] is False


def test_build_run_env_records_explicit_workload():
    prompt = matrix.PromptCase(prompt_id="normal_short", text="hello")

    env = matrix.build_run_env(
        prompt,
        num_speculative_tokens=1,
        max_tokens=70,
        expected_token_ids=[1, 2],
        extra_env={"A": "B"},
    )

    assert env["PROMPT_TEXT"] == "hello"
    assert env["NUM_SPECULATIVE_TOKENS"] == "1"
    assert env["MAX_TOKENS"] == "70"
    assert env["TP"] == "4"
    assert env["EXPECTED_TOKEN_IDS"] == "[1, 2]"
    assert env["A"] == "B"


def test_build_group_run_env_uses_one_engine_for_three_prompts():
    prompts = [
        matrix.PromptCase(prompt_id="p0", text="alpha"),
        matrix.PromptCase(prompt_id="p1", text="beta"),
        matrix.PromptCase(prompt_id="p2", text="gamma"),
    ]

    env = matrix.build_group_run_env(
        prompts,
        num_speculative_tokens=4,
        max_tokens=100,
        expected_run_token_ids=[[1], [2], [3]],
        expected_finish_reasons=["length", "length", "length"],
        extra_env={"CANDIDATE": "1"},
    )

    assert json.loads(env["PROMPT_TEXTS_JSON"]) == ["alpha", "beta", "gamma"]
    assert json.loads(env["RUN_MAX_TOKENS_JSON"]) == [100, 100, 100]
    assert json.loads(env["EXPECTED_RUN_TOKEN_IDS"]) == [[1], [2], [3]]
    assert json.loads(env["EXPECTED_RUN_FINISH_REASONS"]) == [
        "length",
        "length",
        "length",
    ]
    assert env["BENCH_RUNS"] == "3"
    assert env["MIN_TOKENS"] == "100"
    assert env["NUM_SPECULATIVE_TOKENS"] == "4"
    assert env["CANDIDATE"] == "1"


def test_build_group_run_env_supports_ten_prompt_gate():
    prompts = [
        matrix.PromptCase(prompt_id=f"p{index}", text=f"prompt {index}")
        for index in range(10)
    ]
    expected = [[index, index + 1] for index in range(10)]

    matrix.validate_prompt_cases(prompts, label="ten-sample")
    env = matrix.build_group_run_env(
        prompts,
        num_speculative_tokens=4,
        max_tokens=100,
        expected_run_token_ids=expected,
        expected_finish_reasons=["length"] * 10,
    )

    assert env["BENCH_RUNS"] == "10"
    assert json.loads(env["PROMPT_TEXTS_JSON"]) == [
        f"prompt {index}" for index in range(10)
    ]
    assert json.loads(env["EXPECTED_RUN_TOKEN_IDS"]) == expected


def test_parser_accepts_shared_and_separate_environment_overrides():
    args = matrix.build_arg_parser().parse_args(
        [
            "--shared-env",
            "SHARED=1",
            "--base-env",
            "BASE=1",
            "--env",
            "CANDIDATE=1",
        ]
    )

    assert args.shared_env == ["SHARED=1"]
    assert args.base_env == ["BASE=1"]
    assert args.env == ["CANDIDATE=1"]


def test_write_summary_fails_on_any_prompt_mismatch(tmp_path: Path):
    rows = [
        {
            "prompt_id": "p0",
            "comparison": {"exact": True, "first_mismatch": None},
            "candidate": {"exit_code": 0},
        },
        {
            "prompt_id": "p1",
            "comparison": {"exact": False, "first_mismatch": 3},
            "candidate": {"exit_code": 1},
        },
    ]

    summary_path = matrix.write_summary(
        tmp_path / "summary.json",
        rows=rows,
        manifest={"max_tokens": 70},
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "fail"
    assert payload["num_prompts"] == 2
    assert payload["num_exact"] == 1
    assert payload["first_failure"]["prompt_id"] == "p1"


def test_main_loads_each_engine_once_for_three_prompts(
    tmp_path: Path, monkeypatch
):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps({"prompt_id": f"p{index}", "text": text})
            for index, text in enumerate(["alpha", "beta", "gamma"])
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run_generate_case(*, output_dir, env_overrides):
        calls.append((output_dir, env_overrides))
        return {
            "exit_code": 0,
            "artifact_dir": str(output_dir),
            "run_log": str(output_dir / "run.log"),
            "run_token_ids": [[10, 11], [20, 21], [30, 31]],
            "run_finish_reasons": ["length", "length", "length"],
        }

    monkeypatch.setattr(matrix, "run_generate_case", fake_run_generate_case)
    output_root = tmp_path / "out"

    exit_code = matrix.main(
        [
            "--output-root",
            str(output_root),
            "--prompts-jsonl",
            str(prompts),
            "--max-tokens",
            "100",
            "--num-speculative-tokens",
            "4",
            "--shared-env",
            "SHARED=1",
            "--base-env",
            "BASE=1",
            "--env",
            "CANDIDATE=1",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0][1]["NUM_SPECULATIVE_TOKENS"] == "0"
    assert calls[1][1]["NUM_SPECULATIVE_TOKENS"] == "4"
    assert calls[0][1]["SHARED"] == "1"
    assert calls[1][1]["SHARED"] == "1"
    assert calls[0][1]["BASE"] == "1"
    assert "BASE" not in calls[1][1]
    assert "CANDIDATE" not in calls[0][1]
    assert calls[1][1]["CANDIDATE"] == "1"
    summary = json.loads((output_root / "summary.json").read_text())
    assert summary["decision"] == "pass"
    assert summary["num_exact"] == 3
    assert [row["prompt_id"] for row in summary["rows"]] == ["p0", "p1", "p2"]
