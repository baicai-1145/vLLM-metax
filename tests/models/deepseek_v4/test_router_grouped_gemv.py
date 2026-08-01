import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_router_ordinary_batched_gemm_is_not_row_exact():
    torch.manual_seed(137)
    device = torch.device("cuda")
    rows = torch.randn(6, 4096, dtype=torch.bfloat16, device=device)
    weight = torch.randn(256, 4096, dtype=torch.bfloat16, device=device)
    expected = torch.empty(6, 256, dtype=torch.float32, device=device)

    for row in range(6):
        row_slice = slice(row, row + 1)
        expected[row_slice] = torch.mm(
            rows[row_slice],
            weight.T,
            out_dtype=torch.float32,
        )

    ordinary_batched = torch.mm(
        rows,
        weight.T,
        out_dtype=torch.float32,
    )
    torch.cuda.synchronize()

    assert not torch.equal(ordinary_batched, expected)
