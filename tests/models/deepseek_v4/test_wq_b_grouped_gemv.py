import pytest
import torch

import vllm_metax._metax_sparse_C  # noqa: F401


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _inputs(seed: int = 41):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    rows = torch.randn(6, 1024, dtype=torch.bfloat16, device=device)
    weight = torch.randn(8192, 1024, dtype=torch.bfloat16, device=device)
    return rows, weight


def _serial(rows: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [torch.nn.functional.linear(rows[index : index + 1], weight) for index in range(6)]
    )


def test_wq_b_ordinary_batched_gemm_is_not_row_exact():
    rows, weight = _inputs()
    expected = _serial(rows, weight)
    ordinary_batched = torch.nn.functional.linear(rows, weight)
    torch.cuda.synchronize()

    assert not torch.equal(ordinary_batched, expected)


def test_wq_b_native_serial_rows_is_row_exact():
    rows, weight = _inputs()
    expected = _serial(rows, weight)
    actual = torch.empty_like(expected)

    torch.ops._metax_sparse_C.gemv_bf16_serial_rows_out(
        rows, weight, actual
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_wq_b_exact_grouped_rows_is_row_exact():
    rows, weight = _inputs(seed=67)
    expected = _serial(rows, weight)
    actual = torch.empty_like(expected)

    torch.ops._metax_sparse_C.gemv_bf16_exact_grouped_rows_out(
        rows, weight, actual
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_wq_b_exact_grouped_rows_graph_replay_is_row_exact():
    rows, weight = _inputs(seed=71)
    expected = _serial(rows, weight)
    actual = torch.empty_like(expected)
    op = torch.ops._metax_sparse_C.gemv_bf16_exact_grouped_rows_out

    op(rows, weight, actual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(rows, weight, actual)

    output_ptr = actual.data_ptr()
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        assert actual.data_ptr() == output_ptr
        assert torch.equal(actual, expected)


def test_native_serial_rows_matches_qkv_rowwise_shape():
    torch.manual_seed(81)
    device = torch.device("cuda")
    rows = torch.randn(6, 4096, dtype=torch.bfloat16, device=device)
    weight = torch.randn(1536, 4096, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [
            torch.nn.functional.linear(rows[index : index + 1], weight)
            for index in range(6)
        ]
    )
    actual = torch.full_like(expected, float("nan"))

    torch.ops._metax_sparse_C.gemv_bf16_serial_rows_out(rows, weight, actual)
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_native_serial_rows_graph_replay_matches_qkv_rowwise_shape():
    torch.manual_seed(83)
    device = torch.device("cuda")
    rows = torch.randn(6, 4096, dtype=torch.bfloat16, device=device)
    weight = torch.randn(1536, 4096, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [
            torch.nn.functional.linear(rows[index : index + 1], weight)
            for index in range(6)
        ]
    )
    actual = torch.empty_like(expected)
    op = torch.ops._metax_sparse_C.gemv_bf16_serial_rows_out

    op(rows, weight, actual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(rows, weight, actual)

    output_ptr = actual.data_ptr()
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        assert actual.data_ptr() == output_ptr
        assert torch.equal(actual, expected)


def test_bf16_fp32_serial_rows_matches_kv_score_rowwise_shape():
    torch.manual_seed(85)
    device = torch.device("cuda")
    rows = torch.randn(6, 4096, dtype=torch.bfloat16, device=device)
    weight = torch.randn(2048, 4096, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [
            torch.mm(
                rows[index : index + 1],
                weight.T,
                out_dtype=torch.float32,
            )
            for index in range(6)
        ]
    )
    actual = torch.full_like(expected, float("nan"))

    torch.ops._metax_sparse_C.gemv_bf16_fp32_serial_rows_out(
        rows, weight, actual
    )
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_bf16_fp32_serial_rows_graph_replay_matches_kv_score_rowwise_shape():
    torch.manual_seed(87)
    device = torch.device("cuda")
    rows = torch.randn(6, 4096, dtype=torch.bfloat16, device=device)
    weight = torch.randn(2048, 4096, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [
            torch.mm(
                rows[index : index + 1],
                weight.T,
                out_dtype=torch.float32,
            )
            for index in range(6)
        ]
    )
    actual = torch.empty_like(expected)
    op = torch.ops._metax_sparse_C.gemv_bf16_fp32_serial_rows_out

    op(rows, weight, actual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(rows, weight, actual)

    output_ptr = actual.data_ptr()
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        assert actual.data_ptr() == output_ptr
        assert torch.equal(actual, expected)


def test_o_proj_exact_grouped_rows_is_row_exact_at_production_shape():
    torch.manual_seed(73)
    device = torch.device("cuda")
    rows = torch.randn(6, 2048, dtype=torch.bfloat16, device=device)
    weight = torch.randn(4096, 2048, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [
            torch.nn.functional.linear(rows[index : index + 1], weight)
            for index in range(6)
        ]
    )
    actual = torch.empty_like(expected)

    torch.ops._metax_sparse_C.gemv_bf16_exact_oproj_grouped_rows_out(
        rows, weight, actual
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_o_proj_exact_grouped_rows_handles_singleton_tail_shape():
    torch.manual_seed(77)
    device = torch.device("cuda")
    rows = torch.randn(1, 2048, dtype=torch.bfloat16, device=device)
    weight = torch.randn(4096, 2048, dtype=torch.bfloat16, device=device)
    expected = torch.nn.functional.linear(rows, weight)
    actual = torch.empty_like(expected)

    torch.ops._metax_sparse_C.gemv_bf16_exact_oproj_grouped_rows_out(
        rows, weight, actual
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_o_proj_exact_grouped_rows_graph_replay_is_row_exact_at_production_shape():
    torch.manual_seed(79)
    device = torch.device("cuda")
    rows = torch.randn(6, 2048, dtype=torch.bfloat16, device=device)
    weight = torch.randn(4096, 2048, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [
            torch.nn.functional.linear(rows[index : index + 1], weight)
            for index in range(6)
        ]
    )
    actual = torch.empty_like(expected)
    op = torch.ops._metax_sparse_C.gemv_bf16_exact_oproj_grouped_rows_out

    op(rows, weight, actual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(rows, weight, actual)

    output_ptr = actual.data_ptr()
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        assert actual.data_ptr() == output_ptr
        assert torch.equal(actual, expected)


def test_o_proj_exact_row_list_is_row_exact_at_production_shape():
    torch.manual_seed(89)
    device = torch.device("cuda")
    rows = torch.randn(6, 2048, dtype=torch.bfloat16, device=device)
    weight = torch.randn(4096, 2048, dtype=torch.bfloat16, device=device)
    row_inputs = list(rows.split(1))
    expected = torch.cat(
        [
            torch.nn.functional.linear(row_inputs[index], weight)
            for index in range(6)
        ]
    )
    actual = torch.empty_like(expected)

    torch.ops._metax_sparse_C.gemv_bf16_exact_oproj_row_list_out(
        row_inputs, weight, actual
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_o_proj_exact_row_list_graph_replay_is_row_exact_at_production_shape():
    torch.manual_seed(91)
    device = torch.device("cuda")
    rows = torch.randn(6, 2048, dtype=torch.bfloat16, device=device)
    weight = torch.randn(4096, 2048, dtype=torch.bfloat16, device=device)
    row_inputs = list(rows.split(1))
    expected = torch.cat(
        [
            torch.nn.functional.linear(row_inputs[index], weight)
            for index in range(6)
        ]
    )
    actual = torch.empty_like(expected)
    op = torch.ops._metax_sparse_C.gemv_bf16_exact_oproj_row_list_out

    op(row_inputs, weight, actual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(row_inputs, weight, actual)

    output_ptr = actual.data_ptr()
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        assert actual.data_ptr() == output_ptr
        assert torch.equal(actual, expected)


def test_wq_b_native_serial_rows_graph_replay_is_row_exact():
    rows, weight = _inputs(seed=43)
    expected = _serial(rows, weight)
    actual = torch.empty_like(expected)
    op = torch.ops._metax_sparse_C.gemv_bf16_serial_rows_out

    op(rows, weight, actual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(rows, weight, actual)

    output_ptr = actual.data_ptr()
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        assert actual.data_ptr() == output_ptr
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("seed", "shape"),
    [(47, (2, 257, 193)), (53, (5, 1025, 511)), (59, (6, 512, 4096))],
)
def test_wq_b_native_serial_rows_handles_boundary_shapes(seed, shape):
    rows_count, input_width, output_width = shape
    torch.manual_seed(seed)
    device = torch.device("cuda")
    rows = torch.randn(
        rows_count, input_width, dtype=torch.bfloat16, device=device
    )
    weight = torch.randn(
        output_width, input_width, dtype=torch.bfloat16, device=device
    )
    expected = torch.cat(
        [
            torch.nn.functional.linear(rows[index : index + 1], weight)
            for index in range(rows_count)
        ]
    )
    actual = torch.full_like(expected, float("nan"))

    torch.ops._metax_sparse_C.gemv_bf16_serial_rows_out(
        rows, weight, actual
    )
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
