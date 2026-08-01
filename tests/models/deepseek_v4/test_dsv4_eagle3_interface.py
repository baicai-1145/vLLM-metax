import torch

from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm_metax.models.deepseek_v4 import model


def _uninitialized_model(*, aux_layers=()):
    dsv4 = model.DeepseekV4Model.__new__(model.DeepseekV4Model)
    torch.nn.Module.__init__(dsv4)
    dsv4.hc_mult = 2
    dsv4.use_mega_moe = False
    dsv4.start_layer = 0
    dsv4.end_layer = 3
    dsv4.aux_hidden_state_layers = aux_layers
    dsv4.layers = [lambda hidden, *args: (hidden, hidden, None, None)] * 3
    dsv4._mtp_hidden_buffer = torch.empty(4, 6)
    dsv4.hc_head_fn = torch.empty(1)
    dsv4.hc_head_scale = torch.empty(1)
    dsv4.hc_head_base = torch.empty(1)
    dsv4.rms_norm_eps = 1e-6
    dsv4.hc_eps = 1e-6
    dsv4.norm = lambda hidden: hidden
    return dsv4


def test_dsv4_exposes_eagle3_protocol_and_default_layers():
    dsv4 = _uninitialized_model()
    wrapper = model.DeepseekV4ForCausalLM.__new__(model.DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = dsv4
    dsv4.layers = [None] * 43

    assert isinstance(wrapper, SupportsEagle3)
    assert wrapper.get_eagle3_default_aux_hidden_state_layers() == (2, 21, 40)

    layers = (41, 42, 43)
    wrapper.set_aux_hidden_state_layers(layers)
    assert dsv4.aux_hidden_state_layers == layers


def test_dsv4_forward_captures_aux_and_reuses_final_reconstruction(monkeypatch):
    dsv4 = _uninitialized_model(aux_layers=(1, 3))
    calls = []

    def fake_pp_group():
        return type("PP", (), {"is_first_rank": True, "is_last_rank": True})()

    def fake_mhc_post(hidden, residual, post_mix, res_mix):
        calls.append(hidden)
        return torch.full_like(hidden, float(len(calls)))

    monkeypatch.setattr(model, "get_pp_group", fake_pp_group)
    monkeypatch.setattr(model, "mhc_post", fake_mhc_post)
    monkeypatch.setattr(
        model, "hc_head_fused_kernel", lambda hidden, *args: hidden[:, 0, :]
    )

    final, aux = dsv4(
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        None,
        inputs_embeds=torch.zeros(2, 3),
    )

    assert len(calls) == 2
    torch.testing.assert_close(aux[0], torch.ones(2, 3))
    torch.testing.assert_close(aux[1], torch.full((2, 3), 2.0))
    torch.testing.assert_close(final, torch.full((2, 3), 2.0))


def test_dsv4_forward_without_aux_keeps_plain_output(monkeypatch):
    dsv4 = _uninitialized_model()
    calls = []

    monkeypatch.setattr(
        model,
        "get_pp_group",
        lambda: type("PP", (), {"is_first_rank": True, "is_last_rank": True})(),
    )

    def fake_mhc_post(hidden, residual, post_mix, res_mix):
        calls.append(hidden)
        return hidden

    monkeypatch.setattr(model, "mhc_post", fake_mhc_post)
    monkeypatch.setattr(
        model, "hc_head_fused_kernel", lambda hidden, *args: hidden[:, 0, :]
    )

    output = dsv4(
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        None,
        inputs_embeds=torch.zeros(2, 3),
    )

    assert len(calls) == 1
    assert isinstance(output, torch.Tensor)
