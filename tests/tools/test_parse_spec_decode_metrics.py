from tools.debug.parse_spec_decode_metrics import (
    aggregate_spec_decode_metrics,
    parse_spec_decode_metrics,
)


def test_parse_spec_decode_metrics_preserves_position_rates():
    line = (
        "SpecDecoding metrics: Mean acceptance length: 2.75, "
        "Accepted throughput: 50.00 tokens/s, Drafted throughput: 80.00 "
        "tokens/s, Accepted: 150 tokens, Drafted: 240 tokens, "
        "Per-position acceptance rate: 0.900, 0.700, 0.500, 0.400, "
        "Avg Draft acceptance rate: 62.5%"
    )

    result = parse_spec_decode_metrics([line])

    assert result["drafted_tokens"] == 240
    assert result["accepted_tokens"] == 150
    assert result["mean_acceptance_length"] == 2.75
    assert result["per_position_acceptance"] == [0.9, 0.7, 0.5, 0.4]
    assert result["avg_draft_acceptance_rate"] == 0.625


def test_aggregate_spec_decode_metrics_weights_counts_across_records():
    lines = [
        (
            "SpecDecoding metrics: Mean acceptance length: 3.40, "
            "Accepted throughput: 1.00 tokens/s, Drafted throughput: 1.00 "
            "tokens/s, Accepted: 24 tokens, Drafted: 40 tokens, "
            "Per-position acceptance rate: 0.900, 0.700, 0.500, 0.300, "
            "Avg Draft acceptance rate: 60.0%"
        ),
        (
            "SpecDecoding metrics: Mean acceptance length: 2.60, "
            "Accepted throughput: 1.00 tokens/s, Drafted throughput: 1.00 "
            "tokens/s, Accepted: 8 tokens, Drafted: 20 tokens, "
            "Per-position acceptance rate: 0.800, 0.400, 0.200, 0.200, "
            "Avg Draft acceptance rate: 40.0%"
        ),
    ]

    result = aggregate_spec_decode_metrics(lines)

    assert result["num_records"] == 2
    assert result["num_speculative_tokens"] == 4
    assert result["verification_cycles"] == 15
    assert result["drafted_tokens"] == 60
    assert result["accepted_tokens"] == 32
    assert result["covered_output_tokens"] == 47
    assert result["avg_draft_acceptance_rate"] == 32 / 60
    assert result["mean_acceptance_length"] == 1 + 32 / 15
    assert result["per_position_accepted_tokens"] == [13, 9, 6, 4]
    assert result["per_position_acceptance"] == [13 / 15, 9 / 15, 6 / 15, 4 / 15]
    low, high = result["acceptance_rate_95ci"]
    assert low < 32 / 60 < high
