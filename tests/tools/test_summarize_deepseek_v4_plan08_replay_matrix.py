from tools.debug.summarize_deepseek_v4_plan08_replay_matrix import (
    build_arg_parser,
    parse_layers,
    summarize_qkv_result,
)


def test_parse_layers_accepts_ranges_and_lists():
    assert parse_layers("0,2:4,2") == [0, 2, 3, 4]


def test_parser_supports_skipping_non_modelstage_probes(tmp_path):
    args = build_arg_parser().parse_args(
        [
            "--base-layer-capture",
            str(tmp_path / "base_layer"),
            "--candidate-layer-capture",
            str(tmp_path / "candidate_layer"),
            "--base-qkv-capture",
            str(tmp_path / "base_qkv"),
            "--candidate-qkv-capture",
            str(tmp_path / "candidate_qkv"),
            "--position",
            "659",
            "--skip-q-stage",
            "--skip-qkv-insert",
            "--output",
            str(tmp_path / "matrix.json"),
        ]
    )

    assert args.skip_q_stage
    assert args.skip_qkv_insert


def test_summarize_qkv_result_separates_alignment_mismatch():
    result = {
        "summary": {
            "num_candidate_rows": 1,
            "exact_candidate_rows": 0,
            "first_different_tensor": "slot_mapping",
        },
        "comparisons": [
            {
                "tensors": {
                    "q": {"exact": True},
                    "kv": {"exact": True},
                    "cache_before_row": {"exact": True},
                    "cache_after_row": {"exact": True},
                    "slot_mapping": {"exact": False},
                    "cache_block_indices": {"exact": False},
                    "cache_slot_offsets": {"exact": True},
                    "token_indices": {"exact": False},
                }
            }
        ],
    }

    summary = summarize_qkv_result(result)

    assert summary["status"] == "alignment_mismatch"
    assert summary["qkv_tensors_exact"]
    assert summary["tensor_mismatches"] == []
    assert summary["alignment_mismatches"] == [
        "cache_block_indices",
        "slot_mapping",
        "token_indices",
    ]


def test_summarize_qkv_result_reports_real_tensor_divergence():
    result = {
        "summary": {
            "num_candidate_rows": 1,
            "exact_candidate_rows": 0,
            "first_different_tensor": "q",
        },
        "comparisons": [
            {
                "tensors": {
                    "q": {"exact": False},
                    "kv": {"exact": True},
                    "cache_before_row": {"exact": True},
                    "cache_after_row": {"exact": True},
                    "slot_mapping": {"exact": True},
                    "cache_block_indices": {"exact": True},
                    "cache_slot_offsets": {"exact": True},
                    "token_indices": {"exact": True},
                }
            }
        ],
    }

    summary = summarize_qkv_result(result)

    assert summary["status"] == "divergent"
    assert not summary["qkv_tensors_exact"]
    assert summary["tensor_mismatches"] == ["q"]
