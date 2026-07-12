#!/usr/bin/env python3
import argparse

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], execution_backend="cython")
def tl_gemm_kernel(M: int, N: int, K: int, block_M: int, block_N: int, block_K: int):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), T.float32),
        B: T.Tensor((K, N), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float32)
            B_shared = T.alloc_shared((block_K, block_N), T.float32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--disable-cache", action="store_true")
    args = parser.parse_args()

    if args.disable_cache:
        tilelang.disable_cache()
    torch.set_float32_matmul_precision("high")
    dump = torch.load(args.dump, map_location="cuda")
    residual = dump["torch_out"][0].cuda().flatten(1).float()
    padded = torch.zeros(
        16, residual.shape[1], dtype=torch.float32, device=residual.device
    )
    padded[: residual.shape[0]].copy_(residual)
    fn_t_24 = dump["inputs"]["fn"].cuda().t().contiguous()
    fn_t = torch.zeros(fn_t_24.shape[0], 32, dtype=torch.float32, device=fn_t_24.device)
    fn_t[:, : fn_t_24.shape[1]].copy_(fn_t_24)
    ref = torch.nn.functional.linear(residual, dump["inputs"]["fn"].cuda())
    kernel = tl_gemm_kernel(16, 32, residual.shape[1], 16, 32, args.block_k)
    out = kernel(padded, fn_t)[: residual.shape[0], : ref.shape[1]]
    torch.cuda.synchronize()
    mask = out != ref
    print(
        {
            "equal": not mask.any().item(),
            "num_diff": int(mask.sum().item()),
            "max_abs": (out - ref).abs().max().item(),
            "block_k": args.block_k,
        }
    )
    if mask.any().item():
        flat = int(mask.flatten().nonzero()[0].item())
        idx = tuple(
            int(v)
            for v in torch.unravel_index(torch.tensor(flat, device="cuda"), mask.shape)
        )
        print({"first": idx, "tilelang": out[idx].item(), "torch": ref[idx].item()})


if __name__ == "__main__":
    main()
