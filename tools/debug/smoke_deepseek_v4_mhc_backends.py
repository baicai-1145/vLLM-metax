#!/usr/bin/env python3
import argparse
import importlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_metax.models.deepseek_v4.ops.mhc.debug_diff import tensor_diff


def _load_model_tensor(model: str, name: str) -> torch.Tensor:
    model_path = Path(model)
    index = json.loads(
        (model_path / 'model.safetensors.index.json').read_text()
    )
    shard = index['weight_map'][name]
    with safe_open(model_path / shard, framework='pt', device='cpu') as handle:
        return handle.get_tensor(name)


def build_inputs(
    device: str,
    num_tokens: int,
    hidden: int,
    model: str | None,
    layer: int,
    kind: str,
):
    torch.manual_seed(0)
    hc_mult = 4
    mix_hc = (2 + hc_mult) * hc_mult
    residual = torch.randn(num_tokens, hc_mult, hidden, device=device, dtype=torch.bfloat16)
    fn = torch.randn(mix_hc, hc_mult * hidden, device=device, dtype=torch.float32)
    scale = torch.randn(3, device=device, dtype=torch.float32)
    base = torch.randn(mix_hc, device=device, dtype=torch.float32)
    x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    head_fn = torch.randn(hc_mult, hc_mult * hidden, device=device, dtype=torch.float32)
    head_scale = torch.randn(1, device=device, dtype=torch.float32)
    head_base = torch.randn(hc_mult, device=device, dtype=torch.float32)
    if model is not None:
        prefix = f'layers.{layer}.hc_{kind}'
        fn = _load_model_tensor(model, f'{prefix}_fn').to(device, torch.float32)
        scale = _load_model_tensor(model, f'{prefix}_scale').to(device, torch.float32)
        base = _load_model_tensor(model, f'{prefix}_base').to(device, torch.float32)
        head_fn = _load_model_tensor(model, 'hc_head_fn').to(device, torch.float32)
        head_scale = _load_model_tensor(model, 'hc_head_scale').to(
            device, torch.float32
        )
        head_base = _load_model_tensor(model, 'hc_head_base').to(
            device, torch.float32
        )
    return residual, fn, scale, base, x, head_fn, head_scale, head_base


def _sync_and_check(name: str, outputs):
    torch.cuda.synchronize()
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    for index, value in enumerate(tensors):
        value_float = value.float()
        if not torch.isfinite(value_float).all().item():
            raise RuntimeError(f'{name}[{index}] contains non-finite values')
        print(
            f'{name}[{index}]: shape={tuple(value.shape)} stride={value.stride()} '
            f'max_abs={value_float.abs().max().item():.6g}'
        )


def run_backend(
    name: str,
    num_tokens: int,
    hidden: int,
    model: str | None,
    layer: int,
    kind: str,
):
    os.environ['VLLM_METAX_DSV4_MHC_BACKEND'] = name
    if name == 'tilelang':
        os.environ['VLLM_METAX_DSV4_MHC_TILELANG_OPS'] = 'pre,post,fused,head'
    mod = importlib.import_module('vllm_metax.models.deepseek_v4.ops.mhc.backend')
    mod = importlib.reload(mod)
    residual, fn, scale, base, x, head_fn, head_scale, head_base = build_inputs(
        'cuda', num_tokens, hidden, model, layer, kind
    )
    post, comb, layer_input = mod.mhc_pre(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 3)
    _sync_and_check('pre', (post, comb, layer_input))
    fused = mod.mhc_fused_post_pre(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        3,
    )
    _sync_and_check('fused_post_pre', fused)
    post_out = mod.mhc_post(x, residual, post, comb)
    _sync_and_check('post', post_out)
    head = mod.hc_head_fused_kernel(residual, head_fn, head_scale, head_base, 1e-6, 1e-6)
    _sync_and_check('head', head)
    return {
        'post': post,
        'comb': comb,
        'layer_input': layer_input,
        'post_out': post_out,
        'head': head,
        'fused_residual': fused[0],
        'fused_post': fused[1],
        'fused_comb': fused[2],
        'fused_layer_input': fused[3],
        'resolved_backend': mod.get_mhc_backend_name(),
    }


def check_graph_replay(hidden: int, raw_corpus: str | None = None) -> None:
    os.environ['VLLM_METAX_DSV4_MHC_BACKEND'] = 'tilelang'
    os.environ['VLLM_METAX_DSV4_MHC_TILELANG_OPS'] = 'fused,post,head'
    mod = importlib.import_module('vllm_metax.models.deepseek_v4.ops.mhc.backend')
    mod = importlib.reload(mod)
    residual, fn, scale, base, x, head_fn, head_scale, head_base = build_inputs(
        'cuda', 1, hidden, None, 0, 'attn'
    )
    post = torch.randn(1, 4, 1, device='cuda', dtype=torch.float32)
    comb = torch.randn(1, 4, 4, device='cuda', dtype=torch.float32)

    def sequence():
        fused = mod.mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
        )
        post_out = mod.mhc_post(x, residual, post, comb)
        head = mod.hc_head_fused_kernel(
            residual, head_fn, head_scale, head_base, 1e-6, 1e-6
        )
        return fused[0], fused[1], fused[2], fused[3], post_out, head

    for _ in range(2):
        outputs = sequence()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = sequence()
    torch.cuda.synchronize()
    output_ptrs = [output.data_ptr() for output in outputs]
    graph.replay()
    torch.cuda.synchronize()
    before = [output.detach().clone() for output in outputs]
    residual.zero_()
    x.zero_()
    expected_after = [output.detach().clone() for output in sequence()]
    graph.replay()
    torch.cuda.synchronize()
    if [output.data_ptr() for output in outputs] != output_ptrs:
        raise RuntimeError('TileLang graph replay changed output data_ptrs')
    if any(torch.equal(old, new) for old, new in zip(before, outputs)):
        raise RuntimeError(
            'TileLang graph replay did not refresh every fused/post/head output'
        )
    for index, (actual, expected) in enumerate(zip(outputs, expected_after)):
        diff = tensor_diff(actual, expected)
        if not diff['equal']:
            raise RuntimeError(f'graph replay output {index} mismatch: {diff}')
    print(
        'graph replay: refreshed fused_residual/fused_post/fused_comb/'
        'fused_layer_input/post/head'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare-tilelang', action='store_true')
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--model')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--kind', choices=('attn', 'ffn'), default='attn')
    parser.add_argument('--check-graph-replay', action='store_true')
    parser.add_argument('--raw-corpus')
    args = parser.parse_args()

    if args.check_graph_replay:
        check_graph_replay(args.hidden, args.raw_corpus)
        return

    torch_out = run_backend(
        'torch', args.tokens, args.hidden, args.model, args.layer, args.kind
    )
    print('torch backend:', torch_out['resolved_backend'])
    if not args.compare_tilelang:
        return

    tile_out = run_backend(
        'tilelang', args.tokens, args.hidden, args.model, args.layer, args.kind
    )
    print('tilelang backend:', tile_out['resolved_backend'])
    if tile_out['resolved_backend'] != 'tilelang':
        print('tilelang path not ready in current environment')
        return
    for key in (
        'post',
        'comb',
        'layer_input',
        'fused_residual',
        'fused_post',
        'fused_comb',
        'fused_layer_input',
        'post_out',
        'head',
    ):
        diff = (torch_out[key].float() - tile_out[key].float()).abs().max().item()
        print(f'{key}: max_abs_diff={diff}')


if __name__ == '__main__':
    main()
