import asyncio

import pytest
import torch

from vllm_metax.models.deepseek_v4.collective_census import (
    assert_row_exact,
    collective_census,
    collective_census_context,
)


def test_row_gate_rejects_batch_sensitive_result_and_accepts_rows():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def batch_sensitive_projection(value):
        return value + value.shape[0]

    reference = torch.cat(
        [batch_sensitive_projection(row) for row in inputs.split(1)], dim=0
    )
    batched = batch_sensitive_projection(inputs)
    with pytest.raises(AssertionError, match=r"row 0.*max_abs"):
        assert_row_exact(reference, batched)
    assert_row_exact(
        reference,
        torch.cat([batch_sensitive_projection(row) for row in inputs.split(1)], dim=0),
    )


def test_row_gate_validates_shape_and_dtype():
    with pytest.raises(AssertionError, match="shape"):
        assert_row_exact(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(AssertionError, match="dtype"):
        assert_row_exact(
            torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float64)
        )


def test_row_gate_reports_global_error_with_first_bad_row():
    reference = torch.ones(2, 2)
    candidate = torch.tensor([[1.0, 1.5], [1.0, 4.0]])
    with pytest.raises(
        AssertionError,
        match=r"row 0 mismatch: max_abs=3 max_rel=3",
    ):
        assert_row_exact(reference, candidate)


def test_census_is_inert_and_nested_contexts_are_isolated():
    collective_census("", -1, -1, False)
    with collective_census_context() as outer:
        collective_census("wo_b", 2, 3, True)
        with collective_census_context() as inner:
            collective_census("ffn", 2, 1, False)
        collective_census("wo_b", 2, 1, True)
    assert inner == [
        {
            "layer_idx": 2,
            "projection": "ffn",
            "rows": 1,
            "expects_reduce": False,
        }
    ]
    assert outer == [
        {"layer_idx": 2, "projection": "wo_b", "rows": 3, "expects_reduce": True},
        {"layer_idx": 2, "projection": "wo_b", "rows": 1, "expects_reduce": True},
    ]


def test_census_rejects_writes_from_inherited_asyncio_context():
    async def run():
        with collective_census_context() as records:
            collective_census("ffn", 0, 1, True)

            async def child():
                collective_census("ffn", 1, 1, True)

            with pytest.raises(RuntimeError, match="across asyncio tasks"):
                await asyncio.create_task(child())
            return records

    assert asyncio.run(run()) == [
        {
            "layer_idx": 0,
            "projection": "ffn",
            "rows": 1,
            "expects_reduce": True,
        }
    ]


@pytest.mark.parametrize(
    "args",
    [
        ("", 0, 1, True),
        ("x", -1, 1, True),
        ("x", 0, -1, True),
        ("x", 0, 1, 1),
    ],
)
def test_census_validates_fields(args):
    with collective_census_context(), pytest.raises(ValueError):
        collective_census(*args)
