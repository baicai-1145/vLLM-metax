import json

from tools.debug.summarize_c500_mccl import _benchmark_objects


def test_benchmark_objects_extracts_concatenated_rank_records():
    first = {"rank": 0, "collective_backend": "MCCLLibrary"}
    second = {"rank": 1, "collective_backend": "MCCLLibrary"}
    text = "noise\n" + json.dumps(first) + json.dumps(second) + "\nwarning"
    assert _benchmark_objects(text) == [first, second]


def test_benchmark_objects_ignores_other_json():
    text = json.dumps({"message": "noise"}) + json.dumps(
        {"rank": 0, "collective_backend": "MCCLLibrary"}
    )
    assert _benchmark_objects(text) == [
        {"rank": 0, "collective_backend": "MCCLLibrary"}
    ]
