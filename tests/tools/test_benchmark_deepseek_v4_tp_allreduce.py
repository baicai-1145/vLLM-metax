import argparse

import pytest

from tools.debug.benchmark_deepseek_v4_tp_allreduce import (
    _parse_rows,
    _percentile,
    _require_mccl_pynccl,
)


def test_percentile_uses_nearest_rank():
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.9) == 4.0


def test_parse_rows_is_strict():
    assert _parse_rows("1,3,6") == [1, 3, 6]
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_rows("1,0,6")


def test_require_mccl_pynccl_fails_closed():
    class Group:
        device_communicator = None

    with pytest.raises(RuntimeError, match="enabled PYNCCL"):
        _require_mccl_pynccl(Group())
