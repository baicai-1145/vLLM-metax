#!/usr/bin/env python3
import argparse
import json

import torch

from vllm_metax.models.deepseek_v4.ops.mhc.tilelang_kernels import (
    _mhc_pre_big_fuse,
    _mhc_pre_mix_debug,
    mhc_fused_tilelang,
)
from vllm_metax.models.deepseek_v4.ops.mhc.torch import (
    mhc_pre_apply_mix_ref,
    mhc_pre_norm_fn_ref,
    mhc_pre_split_mixes_ref,
    sinkhorn_normalize_ref,
)


def _bf16_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> dict:
    mask = lhs.view(torch.int16) != rhs.view(torch.int16)
    if not mask.any().item():
        return {"equal": True}
    flat = int(mask.flatten().nonzero()[0].item())
    idx = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat, device=mask.device), mask.shape))
    return {
        "equal": False,
        "index": list(idx),
        "lhs": lhs[idx].float().item(),
        "rhs": rhs[idx].float().item(),
        "max_abs": (lhs.float() - rhs.float()).abs().max().item(),
        "num_diff": int(mask.sum().item()),
    }


def _float_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> dict:
    mask = lhs != rhs
    if not mask.any().item():
        return {"equal": True}
    flat = int(mask.flatten().nonzero()[0].item())
    idx = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat, device=mask.device), mask.shape))
    return {
        "equal": False,
        "index": list(idx),
        "lhs": lhs[idx].float().item(),
        "rhs": rhs[idx].float().item(),
        "max_abs": (lhs.float() - rhs.float()).abs().max().item(),
        "num_diff": int(mask.sum().item()),
    }


def _compute_exact_raw(
    residual_cur: torch.Tensor,
    fn: torch.Tensor,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_flat = residual_cur.reshape(
        residual_cur.shape[0], residual_cur.shape[1] * residual_cur.shape[2]
    ).float()
    fn = fn.float()
    assert residual_flat.shape[-1] % n_splits == 0
    split_size = residual_flat.shape[-1] // n_splits
    gemm_out_mul = torch.empty(
        n_splits,
        residual_flat.shape[0],
        fn.shape[0],
        dtype=torch.float32,
        device=residual_flat.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        residual_flat.shape[0],
        dtype=torch.float32,
        device=residual_flat.device,
    )
    for i_split in range(n_splits):
        start = i_split * split_size
        end = start + split_size
        residual_split = residual_flat[:, start:end]
        fn_split = fn[:, start:end]
        gemm_out_mul[i_split] = residual_split @ fn_split.t()
        gemm_out_sqrsum[i_split] = residual_split.square().sum(-1)
    return gemm_out_mul, gemm_out_sqrsum


def _run_big_fuse_with_exact_raw(
    residual_cur: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_splits = int(params["n_splits"])
    mhc_mult = residual_cur.shape[-2]
    hidden_size = residual_cur.shape[-1]
    gemm_out_mul, gemm_out_sqrsum = _compute_exact_raw(residual_cur, fn, n_splits)
    num_tokens = residual_cur.shape[0]
    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual_cur.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult * mhc_mult, dtype=torch.float32, device=residual_cur.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual_cur.device
    )
    _mhc_pre_big_fuse(
        hidden_size,
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"],
        params["sinkhorn_repeat"],
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix,
        comb_mix,
        layer_input,
    )
    torch.cuda.synchronize()
    return (
        post_mix.view(num_tokens, mhc_mult, 1),
        comb_mix.view(num_tokens, mhc_mult, mhc_mult),
        layer_input,
    )
def _run_big_fuse_with_raw(
    residual_cur: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_splits = int(params["n_splits"])
    mhc_mult = residual_cur.shape[-2]
    hidden_size = residual_cur.shape[-1]
    num_tokens = residual_cur.shape[0]
    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual_cur.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult * mhc_mult, dtype=torch.float32, device=residual_cur.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual_cur.device
    )
    _mhc_pre_big_fuse(
        hidden_size,
        params["rms_eps"],
        params["hc_pre_eps"],
        params["hc_sinkhorn_eps"],
        params["hc_post_mult_value"],
        params["sinkhorn_repeat"],
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix,
        comb_mix,
        layer_input,
    )
    torch.cuda.synchronize()
    return (
        post_mix.view(num_tokens, mhc_mult, 1),
        comb_mix.view(num_tokens, mhc_mult, mhc_mult),
        layer_input,
    )


def _run_pre_mix_debug(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual_cur: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_splits = int(params["n_splits"])
    mhc_mult = residual_cur.shape[-2]
    hidden_size = residual_cur.shape[-1]
    num_tokens = residual_cur.shape[0]
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    normalized_mixes = torch.empty(
        num_tokens,
        mhc_mult3,
        dtype=torch.float32,
        device=residual_cur.device,
    )
    pre_mix = torch.empty(
        num_tokens,
        mhc_mult,
        dtype=torch.float32,
        device=residual_cur.device,
    )
    _mhc_pre_mix_debug(
        hidden_size,
        params["rms_eps"],
        params["hc_pre_eps"],
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        normalized_mixes,
        pre_mix,
    )
    torch.cuda.synchronize()
    return normalized_mixes, pre_mix


def _scalar_apply_mix_value(
    residual_cur: torch.Tensor,
    pre_mix: torch.Tensor,
    token_idx: int,
    hidden_idx: int,
    order: tuple[int, ...],
) -> dict:
    acc = torch.zeros((), dtype=torch.float32, device=residual_cur.device)
    terms = []
    for mhc_idx in order:
        term = (
            pre_mix[token_idx, mhc_idx, 0]
            * residual_cur[token_idx, mhc_idx, hidden_idx].float()
        )
        acc = acc + term
        terms.append(
            {
                "mhc": int(mhc_idx),
                "pre": float(pre_mix[token_idx, mhc_idx, 0].item()),
                "residual": float(
                    residual_cur[token_idx, mhc_idx, hidden_idx].float().item()
                ),
                "term": float(term.item()),
                "acc": float(acc.item()),
            }
        )
    return {
        "value_fp32": float(acc.item()),
        "value_bf16": float(acc.bfloat16().float().item()),
        "terms": terms,
    }


def _layer_scalar_decomp(
    residual_cur: torch.Tensor,
    torch_pre: torch.Tensor,
    big_fuse_pre: torch.Tensor,
    torch_layer: torch.Tensor,
    big_fuse_layer: torch.Tensor,
) -> dict | None:
    diff = _bf16_diff(big_fuse_layer, torch_layer)
    if diff.get("equal"):
        return None
    hc_mult = residual_cur.shape[-2]
    torch_pre = torch_pre.reshape(-1, hc_mult, 1)
    big_fuse_pre = big_fuse_pre.reshape(-1, hc_mult, 1)
    idx = diff["index"]
    if len(idx) == 1:
        token_idx, hidden_idx = 0, idx[0]
    else:
        token_idx, hidden_idx = idx[-2], idx[-1]
    forward_order = tuple(range(hc_mult))
    reverse_order = tuple(reversed(forward_order))
    return {
        "index": idx,
        "torch_layer": float(torch_layer[tuple(idx)].float().item()),
        "big_fuse_layer": float(big_fuse_layer[tuple(idx)].float().item()),
        "torch_pre_forward": _scalar_apply_mix_value(
            residual_cur, torch_pre, token_idx, hidden_idx, forward_order
        ),
        "torch_pre_reverse": _scalar_apply_mix_value(
            residual_cur, torch_pre, token_idx, hidden_idx, reverse_order
        ),
        "big_fuse_pre_forward": _scalar_apply_mix_value(
            residual_cur, big_fuse_pre, token_idx, hidden_idx, forward_order
        ),
        "big_fuse_pre_reverse": _scalar_apply_mix_value(
            residual_cur, big_fuse_pre, token_idx, hidden_idx, reverse_order
        ),
    }


def _scalar_post_value(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    token_idx: int,
    out_mhc_idx: int,
    hidden_idx: int,
    comb_order: tuple[int, ...],
    x_first: bool,
) -> dict:
    acc = torch.zeros((), dtype=torch.float32, device=residual.device)
    x_term = post_mix[token_idx, out_mhc_idx] * x[token_idx, hidden_idx].float()
    terms = []
    if x_first:
        acc = acc + x_term
        terms.append({"name": "x", "term": float(x_term.item()), "acc": float(acc.item())})
    for in_mhc_idx in comb_order:
        term = (
            comb_mix[token_idx, in_mhc_idx, out_mhc_idx]
            * residual[token_idx, in_mhc_idx, hidden_idx].float()
        )
        acc = acc + term
        terms.append(
            {
                "name": "comb",
                "in_mhc": int(in_mhc_idx),
                "coef": float(comb_mix[token_idx, in_mhc_idx, out_mhc_idx].item()),
                "residual": float(
                    residual[token_idx, in_mhc_idx, hidden_idx].float().item()
                ),
                "term": float(term.item()),
                "acc": float(acc.item()),
            }
        )
    if not x_first:
        acc = acc + x_term
        terms.append({"name": "x", "term": float(x_term.item()), "acc": float(acc.item())})
    return {
        "value_fp32": float(acc.item()),
        "value_bf16": float(acc.bfloat16().float().item()),
        "terms": terms,
    }


def _post_scalar_decomp(
    inputs: dict,
    torch_residual_cur: torch.Tensor,
    tile_residual_cur: torch.Tensor,
) -> dict | None:
    diff = _bf16_diff(tile_residual_cur, torch_residual_cur)
    if diff.get("equal"):
        return None
    idx = diff["index"]
    token_idx, out_mhc_idx, hidden_idx = idx[-3], idx[-2], idx[-1]
    x = inputs["x"].to(torch_residual_cur.device).view(
        torch_residual_cur.shape[0], torch_residual_cur.shape[-1]
    )
    residual = inputs["residual"].to(torch_residual_cur.device).view_as(
        torch_residual_cur
    )
    post_mix = inputs["post_layer_mix"].to(torch_residual_cur.device).view(
        torch_residual_cur.shape[0], torch_residual_cur.shape[-2]
    )
    comb_mix = inputs["comb_res_mix"].to(torch_residual_cur.device).view(
        torch_residual_cur.shape[0],
        torch_residual_cur.shape[-2],
        torch_residual_cur.shape[-2],
    )
    hc_mult = torch_residual_cur.shape[-2]
    forward_order = tuple(range(hc_mult))
    reverse_order = tuple(reversed(forward_order))
    torch_term2 = torch.einsum("bmn,bmc->bnc", comb_mix, residual.float())
    torch_xterm = x.float().unsqueeze(-2) * post_mix.unsqueeze(-1)
    torch_total = torch_xterm + torch_term2
    return {
        "index": idx,
        "torch_residual_cur": float(torch_residual_cur[tuple(idx)].float().item()),
        "tile_residual_cur": float(tile_residual_cur[tuple(idx)].float().item()),
        "torch_einsum": {
            "term2": float(torch_term2[token_idx, out_mhc_idx, hidden_idx].item()),
            "xterm": float(torch_xterm[token_idx, out_mhc_idx, hidden_idx].item()),
            "total_fp32": float(torch_total[token_idx, out_mhc_idx, hidden_idx].item()),
            "total_bf16": float(
                torch_total[token_idx, out_mhc_idx, hidden_idx].bfloat16().float().item()
            ),
        },
        "comb_forward_x_last": _scalar_post_value(
            x, residual, post_mix, comb_mix, token_idx, out_mhc_idx, hidden_idx,
            forward_order, x_first=False
        ),
        "comb_reverse_x_last": _scalar_post_value(
            x, residual, post_mix, comb_mix, token_idx, out_mhc_idx, hidden_idx,
            reverse_order, x_first=False
        ),
        "x_first_comb_forward": _scalar_post_value(
            x, residual, post_mix, comb_mix, token_idx, out_mhc_idx, hidden_idx,
            forward_order, x_first=True
        ),
        "x_first_comb_reverse": _scalar_post_value(
            x, residual, post_mix, comb_mix, token_idx, out_mhc_idx, hidden_idx,
            reverse_order, x_first=True
        ),
    }


def _run_fused_raw(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = residual.shape[0]
    mhc_mult = residual.shape[1]
    hidden_size = residual.shape[2]
    mhc_mult3 = fn.shape[0]
    tile_n = 2 if num_tokens < 8 else 3
    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        mhc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual)
    mhc_fused_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        fn.view(mhc_mult3, mhc_mult, hidden_size),
        gemm_out_mul,
        gemm_out_sqrsum,
        residual_cur,
        mhc_mult,
        hidden_size,
        mhc_mult3,
        tile_n=tile_n,
        split_k=n_splits,
    )
    torch.cuda.synchronize()
    return residual_cur, gemm_out_mul, gemm_out_sqrsum


def _normalize_raw(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    rms_group_size: int,
    rms_eps: float,
) -> torch.Tensor:
    rms = torch.rsqrt(gemm_out_sqrsum.sum(dim=0) / rms_group_size + rms_eps)
    return gemm_out_mul.sum(dim=0) * rms.unsqueeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    dump = torch.load(args.dump, map_location=args.device)
    record = dump["record"]
    inputs = dump["inputs"]
    params = dump["params"]
    residual_cur = dump["torch_out"][0].to(args.device)
    tile_residual_cur = dump["tilelang_out"][0].to(args.device)
    torch_post = dump["torch_out"][1].to(args.device)
    torch_comb = dump["torch_out"][2].to(args.device)
    torch_layer = dump["torch_out"][3].to(args.device)
    tile_post = dump["tilelang_out"][1].to(args.device)
    tile_comb = dump["tilelang_out"][2].to(args.device)
    tile_layer = dump["tilelang_out"][3].to(args.device)

    mixes = mhc_pre_norm_fn_ref(
        residual_cur,
        inputs["fn"].to(args.device),
        None,
        params["rms_eps"],
    )
    pre, post, comb = mhc_pre_split_mixes_ref(
        mixes,
        inputs["hc_scale"].to(args.device),
        inputs["hc_base"].to(args.device),
        residual_cur.shape[-2],
        params["hc_post_mult_value"],
        params["hc_pre_eps"],
    )
    comb = sinkhorn_normalize_ref(
        comb,
        repeat=params["sinkhorn_repeat"],
        eps=params["hc_sinkhorn_eps"],
    )
    layer = mhc_pre_apply_mix_ref(residual_cur, pre).reshape_as(torch_layer)
    post = post.reshape_as(torch_post)
    comb = comb.reshape_as(torch_comb)

    result = {
        "record": record,
        "recomputed_post_vs_torch": _float_diff(post, torch_post),
        "recomputed_comb_vs_torch": _float_diff(comb, torch_comb),
        "recomputed_layer_vs_torch": _bf16_diff(layer, torch_layer),
        "tile_post_vs_torch": _float_diff(tile_post, torch_post),
        "tile_comb_vs_torch": _float_diff(tile_comb, torch_comb),
        "tile_layer_vs_torch": _bf16_diff(tile_layer, torch_layer),
        "tile_residual_vs_torch": _bf16_diff(tile_residual_cur, residual_cur),
        "tile_residual_scalar_decomp": _post_scalar_decomp(
            inputs,
            residual_cur,
            tile_residual_cur,
        ),
    }
    exact_post, exact_comb, exact_layer = _run_big_fuse_with_exact_raw(
        residual_cur,
        inputs["fn"].to(args.device),
        inputs["hc_scale"].to(args.device),
        inputs["hc_base"].to(args.device),
        params,
    )
    result.update(
        {
            "exact_raw_big_fuse_post_vs_torch": _float_diff(exact_post, torch_post),
            "exact_raw_big_fuse_comb_vs_torch": _float_diff(exact_comb, torch_comb),
            "exact_raw_big_fuse_layer_vs_torch": _bf16_diff(exact_layer, torch_layer),
        }
    )
    n_splits = int(params["n_splits"])
    fused_residual, fused_raw, fused_sqrsum = _run_fused_raw(
        inputs["x"].to(args.device).view(residual_cur.shape[0], residual_cur.shape[-1]),
        inputs["residual"].to(args.device).view_as(residual_cur),
        inputs["post_layer_mix"].to(args.device).view(
            residual_cur.shape[0], residual_cur.shape[-2]
        ),
        inputs["comb_res_mix"].to(args.device).view(
            residual_cur.shape[0], residual_cur.shape[-2], residual_cur.shape[-2]
        ),
        inputs["fn"].to(args.device),
        n_splits,
    )
    exact_raw, exact_sqrsum = _compute_exact_raw(
        residual_cur,
        inputs["fn"].to(args.device),
        n_splits,
    )
    big_fuse_mixes_from_exact, big_fuse_pre_from_exact = _run_pre_mix_debug(
        exact_raw,
        exact_sqrsum,
        inputs["hc_scale"].to(args.device),
        inputs["hc_base"].to(args.device),
        residual_cur,
        params,
    )
    raw_combo_diffs = {}
    for name, raw, sqrsum in (
        ("fused_raw_fused_sqrsum", fused_raw, fused_sqrsum),
        ("fused_raw_exact_sqrsum", fused_raw, exact_sqrsum),
        ("exact_raw_fused_sqrsum", exact_raw, fused_sqrsum),
        ("exact_raw_exact_sqrsum", exact_raw, exact_sqrsum),
    ):
        combo_post, combo_comb, combo_layer = _run_big_fuse_with_raw(
            residual_cur,
            raw,
            sqrsum,
            inputs["hc_scale"].to(args.device),
            inputs["hc_base"].to(args.device),
            params,
        )
        raw_combo_diffs[name] = {
            "post_vs_torch": _float_diff(combo_post, torch_post),
            "comb_vs_torch": _float_diff(combo_comb, torch_comb),
            "layer_vs_torch": _bf16_diff(combo_layer, torch_layer),
        }
    fused_mixes = _normalize_raw(
        fused_raw,
        fused_sqrsum,
        residual_cur.shape[-2] * residual_cur.shape[-1],
        params["rms_eps"],
    ).view_as(mixes)
    exact_mixes = _normalize_raw(
        exact_raw,
        exact_sqrsum,
        residual_cur.shape[-2] * residual_cur.shape[-1],
        params["rms_eps"],
    ).view_as(mixes)
    fused_pre, fused_post, fused_comb = mhc_pre_split_mixes_ref(
        fused_mixes,
        inputs["hc_scale"].to(args.device),
        inputs["hc_base"].to(args.device),
        residual_cur.shape[-2],
        params["hc_post_mult_value"],
        params["hc_pre_eps"],
    )
    exact_pre, exact_post_split, exact_comb_split = mhc_pre_split_mixes_ref(
        exact_mixes,
        inputs["hc_scale"].to(args.device),
        inputs["hc_base"].to(args.device),
        residual_cur.shape[-2],
        params["hc_post_mult_value"],
        params["hc_pre_eps"],
    )
    result.update(
        {
            "rerun_fused_residual_vs_torch": _bf16_diff(fused_residual, residual_cur),
            "rerun_fused_raw_vs_exact_raw": _float_diff(fused_raw, exact_raw),
            "rerun_fused_sqrsum_vs_exact_sqrsum": _float_diff(
                fused_sqrsum, exact_sqrsum
            ),
            "rerun_fused_mixes_vs_exact_mixes": _float_diff(
                fused_mixes, exact_mixes
            ),
            "rerun_fused_pre_vs_exact_pre": _float_diff(fused_pre, exact_pre),
            "rerun_fused_post_vs_exact_post": _float_diff(
                fused_post, exact_post_split
            ),
            "rerun_fused_comb_logits_vs_exact_comb_logits": _float_diff(
                fused_comb, exact_comb_split
            ),
            "exact_raw_big_fuse_pre_vs_torch_pre": _float_diff(
                big_fuse_pre_from_exact.view_as(pre),
                pre,
            ),
            "exact_raw_big_fuse_mixes_vs_torch_mixes": _float_diff(
                big_fuse_mixes_from_exact.view_as(mixes),
                mixes,
            ),
            "exact_raw_big_fuse_layer_scalar_decomp": _layer_scalar_decomp(
                residual_cur,
                pre,
                big_fuse_pre_from_exact.view_as(pre),
                torch_layer,
                exact_layer,
            ),
            "raw_combo_diffs": raw_combo_diffs,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
