from tools.debug.summarize_c500_mccl_perf import _parse_row


def test_parse_reduction_row():
    row = _parse_row(
        [
            "2048",
            "1024",
            "bfloat16",
            "sum",
            "-1",
            "9.24",
            "0.22",
            "0.33",
            "0",
            "9.22",
            "0.22",
            "0.33",
            "0",
        ],
        reduction=True,
    )
    assert row["size_bytes"] == 2048
    assert row["reduction"] == "sum"
    assert row["out_of_place"]["wrong"] == 0


def test_parse_non_reduction_row():
    row = _parse_row(
        [
            "2048",
            "256",
            "bfloat16",
            "-1",
            "9.24",
            "0.22",
            "0.33",
            "0",
            "9.22",
            "0.22",
            "0.33",
            "0",
        ],
        reduction=False,
    )
    assert row["count"] == 256
    assert row["root"] == -1
    assert row["in_place"]["time_us"] == 9.22
