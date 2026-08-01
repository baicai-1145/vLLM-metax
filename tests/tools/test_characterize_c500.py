import pytest

from tools.debug.characterize_c500 import _percentile, _stats


def test_percentile_uses_nearest_rank():
    values = [4.0, 1.0, 3.0, 2.0]
    assert _percentile(values, 0.50) == 2.0
    assert _percentile(values, 0.90) == 4.0


def test_stats_reports_distribution():
    result = _stats([1.0, 2.0, 3.0, 4.0])
    assert result == {
        "repetitions": 4,
        "median": 2.5,
        "p90": 4.0,
        "p99": 4.0,
        "min": 1.0,
        "max": 4.0,
    }


@pytest.mark.parametrize("percentile", [0.5, 0.9, 0.99])
def test_percentile_rejects_empty_values(percentile):
    with pytest.raises(ValueError, match="must not be empty"):
        _percentile([], percentile)
