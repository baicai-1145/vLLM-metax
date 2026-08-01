import torch

from vllm_metax.models.deepseek_v4 import model
from vllm_metax.models.deepseek_v4.collective_census import (
    collective_census_context,
)


class _FakeMhcOps:
    def __init__(self):
        self.single_rows = []
        self.grouped_rows = []

    def mhc_gemv_fp32_out(self, input_tensor, _weight, _out):
        self.single_rows.append(input_tensor.shape[0])

    def mhc_gemv_fp32_grouped_out(self, input_tensor, _weight, _out):
        self.grouped_rows.append(input_tensor.shape[0])


def test_exact_mhc_gemv_grouped_candidate_is_opt_in(monkeypatch):
    from vllm_metax.models.deepseek_v4.ops.mhc import tilelang

    inputs = torch.empty(6, 16384, dtype=torch.float32)
    weight = torch.empty(24, 16384, dtype=torch.float32)
    out = torch.empty(6, 1, 24, dtype=torch.float32)

    default_ops = _FakeMhcOps()
    monkeypatch.delenv("VLLM_METAX_DSV4_MHC_GROUPED_GEMV", raising=False)
    tilelang._run_exact_mhc_gemv(default_ops, inputs, weight, out)

    assert default_ops.single_rows == [1] * 6
    assert default_ops.grouped_rows == []

    grouped_ops = _FakeMhcOps()
    monkeypatch.setenv("VLLM_METAX_DSV4_MHC_GROUPED_GEMV", "1")
    tilelang._run_exact_mhc_gemv(grouped_ops, inputs, weight, out)

    assert grouped_ops.single_rows == []
    assert grouped_ops.grouped_rows == [6]


def test_k1_candidate_enables_initial_mhc_pre_tokenwise(monkeypatch):
    post_mix = torch.empty(1, 4, 1)
    res_mix = torch.empty(1, 4, 4)
    hidden = torch.empty(1, 8)
    calls = []

    def native(x, *args):
        calls.append(x.shape[0])
        value = x[0, 0]
        post_mix.fill_(value)
        res_mix.fill_(value + 10)
        hidden.fill_(value + 20)
        return post_mix, res_mix, hidden

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setattr(model, "mhc_pre", native)

    x = torch.tensor([[2.0] * 8, [7.0] * 8])
    actual = model._mhc_pre_for_input(x, object())

    torch.testing.assert_close(
        actual[2], torch.stack([torch.full((8,), 22.0), torch.full((8,), 27.0)])
    )
    assert calls == [1, 1]


def test_k1_native_mhc_pre_candidate_uses_batched_mhc_unless_explicit(monkeypatch):
    calls = []

    def native(x, *args):
        calls.append(x.shape[0])
        values = x[:, 0]
        return (
            values[:, None, None].expand(-1, 4, 1).clone(),
            (values + 10)[:, None, None].expand(-1, 4, 4).clone(),
            (values + 20)[:, None].expand(-1, 8).clone(),
        )

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_MHC_PRE_CANDIDATE", "1")
    monkeypatch.setattr(model, "mhc_pre", native)
    x = torch.tensor([[2.0] * 8, [7.0] * 8])

    actual = model._mhc_pre_for_input(x, object())

    torch.testing.assert_close(
        actual[2], torch.stack([torch.full((8,), 22.0), torch.full((8,), 27.0)])
    )
    assert calls == [2]

    calls.clear()
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE", "1")

    actual = model._mhc_pre_for_input(x, object())

    torch.testing.assert_close(
        actual[2], torch.stack([torch.full((8,), 22.0), torch.full((8,), 27.0)])
    )
    assert calls == [1, 1]


def test_explicit_tokenwise_mhc_pre_keeps_post_batched(monkeypatch):
    post_calls = []
    pre_calls = []

    def native_post(x, residual, post_mix, res_mix):
        post_calls.append(x.shape[0])
        return residual + x[:, None, :]

    def native_pre(residual, *args, **kwargs):
        pre_calls.append(residual.shape[0])
        value = residual[:, :1, :1]
        return (
            value.expand(-1, 4, 1).clone(),
            value.expand(-1, 4, 4).clone(),
            residual[:, 0].clone(),
        )

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST", "1")
    monkeypatch.setattr(model, "mhc_post", native_post)
    monkeypatch.setattr(model, "mhc_pre", native_pre)
    monkeypatch.setattr(
        model,
        "mhc_fused_post_pre",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fused MHC should not run")
        ),
    )
    x = torch.tensor([[2.0] * 8, [7.0] * 8])
    residual = torch.zeros(2, 4, 8)
    post_mix = torch.zeros(2, 4, 1)
    res_mix = torch.zeros(2, 4, 4)

    actual = model._mhc_fused_post_pre_for_stage(
        "ffn", x, residual, post_mix, res_mix, object()
    )

    assert post_calls == [2]
    assert pre_calls == [1, 1]
    assert [value.shape[0] for value in actual] == [2, 2, 2, 2]
    torch.testing.assert_close(actual[0][0], torch.full((4, 8), 2.0))
    torch.testing.assert_close(actual[0][1], torch.full((4, 8), 7.0))


def test_explicit_tokenwise_mhc_pre_does_not_change_attention_stage(monkeypatch):
    fused_calls = []

    def native_fused(x, residual, post_mix, res_mix, *args, **kwargs):
        fused_calls.append(x.shape[0])
        return residual, post_mix, res_mix, x

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST", "1")
    monkeypatch.setattr(model, "mhc_fused_post_pre", native_fused)
    monkeypatch.setattr(
        model,
        "mhc_post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("attention MHC post should stay fused")
        ),
    )
    x = torch.zeros(2, 8)
    residual = torch.zeros(2, 4, 8)
    post_mix = torch.zeros(2, 4, 1)
    res_mix = torch.zeros(2, 4, 4)

    model._mhc_fused_post_pre_for_stage(
        "attn", x, residual, post_mix, res_mix, object()
    )

    assert fused_calls == [2]


def test_explicit_tokenwise_mhc_pre_can_select_attention_stage(monkeypatch):
    post_calls = []
    pre_calls = []

    def native_post(x, residual, post_mix, res_mix):
        post_calls.append(x.shape[0])
        return residual + x[:, None, :]

    def native_pre(residual, *args, **kwargs):
        pre_calls.append(residual.shape[0])
        value = residual[:, :1, :1]
        return (
            value.expand(-1, 4, 1).clone(),
            value.expand(-1, 4, 4).clone(),
            residual[:, 0].clone(),
        )

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST_ATTN_LAYERS", "1")
    monkeypatch.setattr(model, "mhc_post", native_post)
    monkeypatch.setattr(model, "mhc_pre", native_pre)
    monkeypatch.setattr(
        model,
        "mhc_fused_post_pre",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected attention MHC should not stay fused")
        ),
    )
    x = torch.tensor([[2.0] * 8, [7.0] * 8])
    residual = torch.zeros(2, 4, 8)
    post_mix = torch.zeros(2, 4, 1)
    res_mix = torch.zeros(2, 4, 4)

    actual = model._mhc_fused_post_pre_for_stage(
        "attn",
        x,
        residual,
        post_mix,
        res_mix,
        object(),
        layer_idx=1,
    )

    assert post_calls == [2]
    assert pre_calls == [1, 1]
    assert [value.shape[0] for value in actual] == [2, 2, 2, 2]


def test_tokenwise_mhc_pre_can_be_limited_to_selected_layers(monkeypatch):
    fused_calls = []
    pre_calls = []

    def native_fused(x, residual, post_mix, res_mix, *args, **kwargs):
        fused_calls.append(x.shape[0])
        return residual, post_mix, res_mix, x

    def native_pre(residual, *args, **kwargs):
        pre_calls.append(residual.shape[0])
        return (
            torch.zeros(residual.shape[0], 4, 1),
            torch.zeros(residual.shape[0], 4, 4),
            residual[:, 0].clone(),
        )

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST_LAYERS", "0")
    monkeypatch.setattr(model, "mhc_fused_post_pre", native_fused)
    monkeypatch.setattr(model, "mhc_post", lambda x, residual, *args: residual)
    monkeypatch.setattr(model, "mhc_pre", native_pre)
    x = torch.zeros(2, 8)
    residual = torch.zeros(2, 4, 8)
    post_mix = torch.zeros(2, 4, 1)
    res_mix = torch.zeros(2, 4, 4)

    model._mhc_fused_post_pre_for_stage(
        "ffn", x, residual, post_mix, res_mix, object(), layer_idx=1
    )
    model._mhc_fused_post_pre_for_stage(
        "ffn", x, residual, post_mix, res_mix, object(), layer_idx=0
    )

    assert fused_calls == [2]
    assert pre_calls == [1, 1]


def test_tokenwise_mhc_pre_replaces_only_selected_positions(monkeypatch):
    def native_pre(residual, *args, **kwargs):
        if residual.shape[0] == 2:
            return (
                torch.full((2, 4, 1), 100.0),
                torch.full((2, 4, 4), 200.0),
                torch.full((2, 8), 300.0),
            )
        value = residual[:, 0, :1]
        return (
            value[:, None].expand(-1, 4, 1).clone(),
            (value + 10)[:, None].expand(-1, 4, 4).clone(),
            (value + 20).expand(-1, 8).clone(),
        )

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE_AFTER_POST_POSITIONS", "78")
    monkeypatch.setattr(model, "mhc_post", lambda x, residual, *args: residual)
    monkeypatch.setattr(model, "mhc_pre", native_pre)
    residual = torch.tensor(
        [
            [[2.0] * 8] * 4,
            [[7.0] * 8] * 4,
        ]
    )

    actual = model._mhc_fused_post_pre_for_stage(
        "ffn",
        torch.zeros(2, 8),
        residual,
        torch.zeros(2, 4, 1),
        torch.zeros(2, 4, 4),
        object(),
        positions=torch.tensor([77, 78]),
    )

    torch.testing.assert_close(actual[1][0], actual[1][0].new_full((4, 1), 100.0))
    torch.testing.assert_close(actual[2][0], actual[2][0].new_full((4, 4), 200.0))
    torch.testing.assert_close(actual[3][0], actual[3][0].new_full((8,), 300.0))
    torch.testing.assert_close(actual[1][1], actual[1][1].new_full((4, 1), 7.0))
    torch.testing.assert_close(actual[2][1], actual[2][1].new_full((4, 4), 17.0))
    torch.testing.assert_close(actual[3][1], actual[3][1].new_full((8,), 27.0))


def test_mhc_pre_shadow_compares_batched_output_with_each_row(monkeypatch):
    pre_calls = []
    captured = []

    def native_pre(residual, *args, **kwargs):
        pre_calls.append(residual.shape[0])
        value = residual[:, :1, :1]
        return (
            value.expand(-1, 4, 1).clone(),
            value.expand(-1, 4, 4).clone(),
            residual[:, 0].clone(),
        )

    monkeypatch.setattr(model, "mhc_pre", native_pre)
    monkeypatch.setattr(
        model,
        "maybe_capture_mhc_pre_shadow_compare",
        lambda **kwargs: captured.append(kwargs),
    )
    residual = torch.tensor(
        [
            [[2.0] * 8] * 4,
            [[7.0] * 8] * 4,
        ]
    )
    actual = (
        torch.zeros(2, 4, 1),
        torch.zeros(2, 4, 4),
        torch.zeros(2, 8),
    )

    model._capture_mhc_pre_shadow_during_capture(
        layer_idx=0,
        positions=torch.tensor([78, 79]),
        residual_cur=residual,
        actual_outputs=actual,
        mhc_args=(object(),),
    )

    assert pre_calls == [1, 1]
    assert len(captured) == 1
    assert captured[0]["actual_outputs"] is actual
    rowwise = captured[0]["rowwise_outputs"]
    torch.testing.assert_close(rowwise[0][:, 0, 0], torch.tensor([2.0, 7.0]))
    torch.testing.assert_close(rowwise[2][:, 0], torch.tensor([2.0, 7.0]))


def test_k1_candidate_enables_all_row_tokenwise_ffn(monkeypatch):
    calls = []

    def ffn(x, input_ids):
        calls.append((x.shape[0], input_ids.tolist()))
        return x + x.shape[0] * 100 + input_ids[:, None]

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_FFN", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN_POSITIONS", "659")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")

    x = torch.tensor([[2.0] * 8, [7.0] * 8])
    input_ids = torch.tensor([3, 11])
    positions = torch.tensor([658, 659])
    actual = model._ffn_for_input(ffn, x, input_ids, positions)

    torch.testing.assert_close(
        actual, torch.stack([torch.full((8,), 105.0), torch.full((8,), 118.0)])
    )
    assert calls == [(1, [3]), (1, [11])]


def test_k1_native_ffn_candidate_uses_batched_ffn_unless_explicit(monkeypatch):
    calls = []

    def ffn(x, input_ids):
        calls.append((x.shape[0], input_ids.tolist()))
        return x + x.shape[0] * 100 + input_ids[:, None]

    monkeypatch.delenv("VLLM_METAX_DSV4_TOKENWISE_FFN", raising=False)
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE", "1")
    x = torch.tensor([[2.0] * 8, [7.0] * 8])
    input_ids = torch.tensor([3, 11])
    positions = torch.tensor([658, 659])

    actual = model._ffn_for_input(ffn, x, input_ids, positions)

    torch.testing.assert_close(
        actual, torch.stack([torch.full((8,), 205.0), torch.full((8,), 218.0)])
    )
    assert calls == [(2, [3, 11])]

    calls.clear()
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN", "1")

    actual = model._ffn_for_input(ffn, x, input_ids, positions)

    torch.testing.assert_close(
        actual, torch.stack([torch.full((8,), 105.0), torch.full((8,), 118.0)])
    )
    assert calls == [(1, [3]), (1, [11])]


def test_initial_mhc_pre_tokenwise_materializes_each_native_result(monkeypatch):
    post_mix = torch.empty(1, 4, 1)
    res_mix = torch.empty(1, 4, 4)
    hidden = torch.empty(1, 8)
    calls = []

    def native(x, *args):
        calls.append(x.shape[0])
        value = x[0, 0]
        post_mix.fill_(value)
        res_mix.fill_(value + 10)
        hidden.fill_(value + 20)
        return post_mix, res_mix, hidden

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_MHC_PRE", "1")
    monkeypatch.setattr(model, "mhc_pre", native)
    x = torch.tensor([[2.0] * 8, [7.0] * 8])

    actual = model._mhc_pre_for_input(x, object())

    torch.testing.assert_close(
        actual[0], torch.stack([torch.full((4, 1), 2.0), torch.full((4, 1), 7.0)])
    )
    torch.testing.assert_close(
        actual[1],
        torch.stack([torch.full((4, 4), 12.0), torch.full((4, 4), 17.0)]),
    )
    torch.testing.assert_close(
        actual[2], torch.stack([torch.full((8,), 22.0), torch.full((8,), 27.0)])
    )
    assert calls == [1, 1]


def test_exact_initial_mhc_pre_rms_dispatches_small_verifier_batch(monkeypatch):
    from vllm_metax.models.deepseek_v4.ops.mhc import tilelang

    sentinel = object()
    monkeypatch.setenv("VLLM_METAX_DSV4_MHC_EXACT_PRE_RMS", "1")
    monkeypatch.setattr(model, "get_mhc_backend_name", lambda: "tilelang")
    monkeypatch.setattr(
        tilelang,
        "_mhc_exact_initial_pre_rms_impl",
        lambda *args, **kwargs: sentinel,
    )

    result = model._mhc_exact_initial_pre_rms_for_input(
        torch.empty(6, 4, 4096, dtype=torch.bfloat16),
        norm_weight=torch.empty(4096, dtype=torch.bfloat16),
        workspace={},
    )

    assert result is sentinel


def test_exact_initial_mhc_pre_rms_is_target_only(monkeypatch):
    exact_calls = []
    exact_workspaces = []
    draft_calls = []

    def exact(residual, *args, **kwargs):
        exact_calls.append(residual.shape[0])
        exact_workspaces.append(kwargs["workspace"])
        value = residual[:, :1, :1]
        return (
            value.expand(-1, 4, 1).clone(),
            value.expand(-1, 4, 4).clone(),
            value[:, 0].expand(-1, 8).clone(),
            value[:, 0].expand(-1, 8).clone(),
        )

    def draft_batched(residual, *args):
        draft_calls.append(residual.shape[0])
        value = residual[:, :1, :1]
        return (
            value.expand(-1, 4, 1).clone() + 100,
            value.expand(-1, 4, 4).clone() + 200,
            value[:, 0].expand(-1, 8).clone(),
        )

    monkeypatch.setattr(model, "_mhc_exact_initial_pre_rms_for_input", exact)
    monkeypatch.setattr(model, "mhc_pre", draft_batched)
    residual = torch.tensor([[[2.0] * 8] * 4, [[7.0] * 8] * 4])
    initial_workspace = {}
    draft_workspace = {}

    target = model._mhc_initial_pre_for_layer(
        True,
        residual,
        object(),
        norm_weight=torch.empty(8),
        workspace=initial_workspace,
    )
    draft = model._mhc_initial_pre_for_layer(
        False,
        residual,
        object(),
        norm_weight=torch.empty(8),
        workspace=draft_workspace,
    )

    assert exact_calls == [2]
    assert exact_workspaces == [initial_workspace]
    assert exact_workspaces[0] is not draft_workspace
    assert draft_calls == [2]
    assert target[3] is not None
    assert draft[3] is None
    torch.testing.assert_close(target[0][0], torch.full((4, 1), 2.0))
    torch.testing.assert_close(target[0][1], torch.full((4, 1), 7.0))
    torch.testing.assert_close(target[1][0], torch.full((4, 4), 2.0))
    torch.testing.assert_close(target[1][1], torch.full((4, 4), 7.0))
    torch.testing.assert_close(draft[0][0], torch.full((4, 1), 102.0))
    torch.testing.assert_close(draft[0][1], torch.full((4, 1), 107.0))
    expected = torch.stack([torch.full((8,), 2.0), torch.full((8,), 7.0)])
    torch.testing.assert_close(target[2], expected)
    torch.testing.assert_close(draft[2], expected)


def test_exact_initial_mhc_pre_rms_is_inert_without_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_MHC_EXACT_PRE_RMS", raising=False)

    result = model._mhc_exact_initial_pre_rms_for_input(
        torch.empty(6, 4, 4096, dtype=torch.bfloat16),
        norm_weight=torch.empty(4096, dtype=torch.bfloat16),
        workspace={},
    )

    assert result is None


def test_tokenwise_ffn_slices_ids_and_materializes_each_native_result(monkeypatch):
    shared_buffer = torch.empty(1, 8)
    calls = []

    def ffn(x, input_ids):
        calls.append((x.shape[0], input_ids.tolist()))
        shared_buffer.copy_(x + input_ids[:, None])
        return shared_buffer

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN", "1")
    x = torch.tensor([[2.0] * 8, [7.0] * 8])
    input_ids = torch.tensor([3, 11])

    actual = model._ffn_for_input(ffn, x, input_ids)

    torch.testing.assert_close(
        actual, torch.stack([torch.full((8,), 5.0), torch.full((8,), 18.0)])
    )
    assert calls == [(1, [3]), (1, [11])]


def test_tokenwise_ffn_supports_six_row_dspark_verifier(monkeypatch):
    calls = []

    def ffn(x, input_ids):
        calls.append((x.shape[0], input_ids.tolist()))
        return x + input_ids[:, None]

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN", "1")
    x = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    input_ids = torch.arange(6)

    actual = model._ffn_for_input(ffn, x, input_ids)

    torch.testing.assert_close(actual, x + input_ids[:, None])
    assert calls == [(1, [index]) for index in range(6)]


def test_tokenwise_ffn_replaces_only_selected_positions(monkeypatch):
    calls = []

    def ffn(x, input_ids):
        calls.append((x.shape[0], input_ids.tolist()))
        return x + x.shape[0] * 100 + input_ids[:, None]

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN_POSITIONS", "659")
    x = torch.tensor([[2.0] * 8, [7.0] * 8])
    input_ids = torch.tensor([3, 11])
    positions = torch.tensor([658, 659])

    actual = model._ffn_for_input(ffn, x, input_ids, positions)

    torch.testing.assert_close(
        actual, torch.stack([torch.full((8,), 205.0), torch.full((8,), 118.0)])
    )
    assert calls == [(2, [3, 11]), (1, [11])]


def test_target_tokenwise_ffn_can_coalesce_only_the_final_reduction(monkeypatch):
    calls = []

    class Experts:
        is_internal_router = False

    class FFN:
        use_mega_moe = False
        layer_idx = 7
        experts = Experts()

        def gate(self, row):
            calls.append(("gate", row.clone()))
            return row + 9, None

        def __call__(self, *_args):
            raise AssertionError("the complete reduced FFN path must not run")

    def coalesce(
        runner,
        x,
        input_ids,
        router_logits_fn,
        *,
        group_rows,
        group_routed_experts,
        group_router,
    ):
        calls.append(("coalesce", runner, x.clone(), input_ids.clone()))
        assert group_rows == x.shape[0]
        assert not group_routed_experts
        assert not group_router
        torch.testing.assert_close(router_logits_fn(x[:1], input_ids[:1]), x[:1] + 9)
        return x + 100

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_COALESCE_TOKENWISE_FFN_REDUCE", "1")
    monkeypatch.setattr(model, "coalesce_moe_row_reductions", coalesce)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    input_ids = torch.tensor([3, 11])

    with collective_census_context() as census:
        actual = model._ffn_for_input(FFN(), x, input_ids, is_target_model=True)

    torch.testing.assert_close(actual, x + 100)
    assert calls[0][0] == "coalesce"
    assert calls[1][0] == "gate"
    assert census == [
        {
            "layer_idx": 7,
            "projection": "target_ffn",
            "rows": 2,
            "expects_reduce": True,
        }
    ]


def test_target_tokenwise_ffn_census_counts_rowwise_reductions(monkeypatch):
    class FFN:
        layer_idx = 9

        def __call__(self, x, input_ids):
            return x + input_ids[:, None]

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN", "1")
    monkeypatch.delenv("VLLM_METAX_DSV4_COALESCE_TOKENWISE_FFN_REDUCE", raising=False)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    input_ids = torch.tensor([3, 11])

    with collective_census_context() as census:
        model._ffn_for_input(FFN(), x, input_ids, is_target_model=True)

    assert census == [
        {
            "layer_idx": 9,
            "projection": "target_ffn",
            "rows": 1,
            "expects_reduce": True,
        },
        {
            "layer_idx": 9,
            "projection": "target_ffn",
            "rows": 1,
            "expects_reduce": True,
        },
    ]


def test_draft_tokenwise_ffn_does_not_use_target_collective_candidate(monkeypatch):
    calls = []

    def ffn(x, input_ids):
        calls.append((x.shape[0], input_ids.tolist()))
        return x + input_ids[:, None]

    monkeypatch.setenv("VLLM_METAX_DSV4_TOKENWISE_FFN", "1")
    monkeypatch.setenv("VLLM_METAX_DSV4_COALESCE_TOKENWISE_FFN_REDUCE", "1")
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    input_ids = torch.tensor([3, 11])

    actual = model._ffn_for_input(ffn, x, input_ids, is_target_model=False)

    torch.testing.assert_close(actual, x + input_ids[:, None])
    assert calls == [(1, [3]), (1, [11])]
