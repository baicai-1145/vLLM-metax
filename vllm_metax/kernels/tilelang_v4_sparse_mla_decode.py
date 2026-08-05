"""tilelang V4 sparse MLA decode kernel -- single-card functional MVP.

This is the first concrete step toward a FlashInfer-equivalent fused
attention kernel for DeepSeek-V4 on MetaX C500.  It is a kernel-only
prototype: it is NOT wired into ``attention.py`` yet.

V4 specifics implemented here (verified against the model config):

* head_dim = 512, single KV stream (K == V, MQA with heads_kv=1); the
  KV cache is rank-3 ``(num_pages, page_block_size, head_size)``.
* The index list concatenates sliding-window blocks (128 tokens = 2
  blocks of 64) and index-topk blocks (512 tokens = 8 blocks of 64);
  -1 pads invalid entries.  Each entry is a *logical* 64-token block
  index that maps through ``block_table`` to a physical page.
* attention_sink: the softmax denominator gains an extra ``exp(sink -
  max)`` term (sink logit per head); the sink token contributes zero
  value (it only dilutes the attention weights).  This matches the
  production triton kernel ``sparse_mla_decode.py`` semantics.
* BF16 KV (V4 cache dtype); fp16 is supported as a fallback via the
  ``dtype`` argument if tilelang/MACA cannot compile the BF16 GEMM.

Adapted from
``/root/tilelang-metax/examples/blocksparse_attention/example_tilelang_sparse_gqa_decode_paged.py``
with the C500 constraints from the feasibility evaluation:

* shared memory <= 64KB/SM: single KV_shared buffer reused for both the
  score GEMM and the value GEMM (K==V), ``num_stages=1``,
  ``block_N=32`` (32 x 512 x 2B = 32KB) + Q_shared 16KB = 48KB.
* threads=64 (proven), block_H=16 (matches 16 heads at TP=4).
* split-k combine kernel retained (correct with num_split >= 1).

Run (GPU0 only -- GPU2 is down, do NOT touch TP=4):

    cd /root/vLLM-metax
    source .venv/bin/activate && source ./env.sh
    export LD_LIBRARY_PATH=$(python -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).parent/"lib")'):$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=0
    python vllm_metax/kernels/tilelang_v4_sparse_mla_decode.py
"""

import argparse
import math
import random

import torch
import tilelang
import tilelang.language as T

LOG2E = 1.4426950408889634


def num_splits_heuristic(
    total_mblocks,
    num_sms,
    num_n_blocks,
    num_m_blocks,
    size_one_kv_head,
    is_causal_or_local,
    max_splits,
):
    """Split-k count heuristic (copied from tilelang blocksparse example)."""
    if total_mblocks >= 0.8 * num_sms:
        size_l2 = 50 * 1024 * 1024
        if size_one_kv_head > size_l2 and num_m_blocks >= num_sms * 2 and not is_causal_or_local:
            return min((size_one_kv_head + size_l2 - 1) // size_l2, max_splits)
        return 1
    if num_n_blocks <= 4:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    max_efficiency = 0.0
    efficiency = []
    for num_splits in range(1, max_splits + 1):
        n_waves = (total_mblocks * num_splits) / num_sms
        eff = n_waves / math.ceil(n_waves)
        max_efficiency = max(max_efficiency, eff)
        efficiency.append(eff)
    for num_splits in range(1, max_splits + 1):
        if efficiency[num_splits - 1] >= 0.85 * max_efficiency:
            return num_splits
    return 1


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def v4_sparse_mla_decode(
    batch,
    heads,
    dim,
    block_N,
    block_H,
    index_block,
    page_block_size,
    num_stages,
    threads,
    num_pages,
    dtype,
):
    """Fused sparse MLA decode: gather SWA+topk blocks, score, softmax+sink,
    weighted KV sum, split-k combine -- one fused tilelang kernel.

    Args (all compile-time):
        batch: number of sequences (1 for MVP).
        heads: number of Q heads (16 at TP=4).
        dim: KV/head dim (512 for V4; K==V shared).
        block_N: KV tile per GEMM (32 for 64KB shared budget).
        block_H: Q tile per CTA (16 == heads).
        index_block: logical block granularity in tokens (64).
        page_block_size: tokens per physical page (256).
        num_stages / threads: pipeline/thread config (1 / 64 proven).
        num_pages: physical pages allocated.
        dtype: "bf16" or "fp16" for the KV/Q stream.

    Tensor args:
        Q: [batch, heads, dim]
        KV: [num_pages, page_block_size, dim]  (K==V shared stream)
        block_indices: [batch, 1, max_selected_blocks] (logical block idx, -1 pad)
        cache_seqlens: [batch] int32
        block_table: [batch, max_num_blocks_per_seq] int32 (logical page -> physical)
        Sinks: [heads] fp32 attention-sink logits
        glse: [batch, heads, num_split] fp32 workspace
        Output_partial: [batch, heads, num_split, dim] fp32 workspace
        Output: [batch, heads, dim] (out_idx=-1)
    """
    scale = (1.0 / dim) ** 0.5 * LOG2E  # sm_scale * log2(e)
    TDT = T.bfloat16 if dtype == "bf16" else T.float16
    accum_dtype = T.float32

    kv_group_num = heads  # heads_kv == 1 -> kv_group_num == heads
    valid_block_H = min(block_H, kv_group_num)
    num_split = T.dynamic("num_split")
    max_num_blocks_per_seq = T.dynamic("max_num_blocks_per_seq")
    max_selected_blocks = T.dynamic("max_selected_blocks")
    sub_tiles = index_block // block_N  # 64/32 == 2
    assert index_block % block_N == 0
    assert block_N <= page_block_size and page_block_size % index_block == 0
    block_ratio = page_block_size // index_block  # pages per index block

    shape_q = [batch, heads, dim]
    shape_kv = [num_pages, page_block_size, dim]
    shape_indices = [batch, 1, max_selected_blocks]
    shape_block_table = [batch, max_num_blocks_per_seq]
    shape_o = [batch, heads, dim]
    part_shape = [batch, heads, num_split, dim]

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, TDT),
        KV: T.Tensor(shape_kv, TDT),
        block_indices: T.Tensor(shape_indices, T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        block_table: T.Tensor(shape_block_table, T.int32),
        Sinks: T.Tensor([heads], accum_dtype),
        glse: T.Tensor([batch, heads, num_split], accum_dtype),
        Output_partial: T.Tensor(part_shape, accum_dtype),
        Output: T.Tensor(shape_o, TDT),
    ):
        # flash-attn split kernel
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], TDT)
            KV_shared = T.alloc_shared([block_N, dim], TDT)  # reused K and V (K==V)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], TDT)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            has_valid_block = T.alloc_var(T.bool)

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            num_blocks = max_selected_blocks
            blocks_per_split = T.floordiv(num_blocks, num_split)
            remaining_blocks = T.floormod(num_blocks, num_split)
            loop_range = blocks_per_split + T.if_then_else(sid < remaining_blocks, 1, 0)
            start = blocks_per_split * sid + T.min(sid, remaining_blocks)
            has_valid_block = False
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                logical_block_idx = block_indices[bid, cur_kv_head, start + k]
                if logical_block_idx >= 0:
                    has_valid_block = True
                    block_table_idx = T.floordiv(logical_block_idx, block_ratio)
                    tile_idx = T.floormod(logical_block_idx, block_ratio)
                    physical_block_idx = block_table[bid, block_table_idx]
                    for sub in T.serial(sub_tiles):
                        kv_off = tile_idx * index_block + sub * block_N
                        T.copy(
                            KV[physical_block_idx, kv_off : kv_off + block_N, :],
                            KV_shared,
                        )
                        T.clear(acc_s)
                        T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        # Mask padding tokens beyond cache_seqlens (decode: no future mask).
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.if_then_else(
                                logical_block_idx * index_block + sub * block_N + j
                                >= cache_seqlens[bid],
                                -T.infinity(accum_dtype),
                                acc_s[i, j],
                            )
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_H):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            # guard against fully-masked tiles (all -inf)
                            scores_max[i] = T.if_then_else(
                                scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i]
                            )
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_H):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_H, dim):
                            acc_o[i, j] *= scores_scale[i]
                        # value GEMM reuses the same KV_shared (K==V)
                        T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            # NOTE: attention sink is NOT applied here per-split.  Adding it
            # per-split would double-count exp(sink) once per split in the
            # combine (bug fixed 2026-08-02).  The sink is added exactly once
            # in the combine kernel after the split LSEs are merged.
            if has_valid_block:
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            for i in T.Parallel(block_H):
                if i < valid_block_H:
                    glse[bid, hid * valid_block_H + i, sid] = logsum[i]
            for i, j in T.Parallel(block_H, dim):
                if i < valid_block_H:
                    Output_partial[bid, hid * valid_block_H + i, sid, j] = acc_o[i, j]

        # split-k combine
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dim], accum_dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)
            max_split = T.alloc_var(T.int32)
            sink_ref = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_local_split = glse[bz, by, k]
                if lse_local_split != 0:
                    max_split = k
                    lse_max_local = T.max(lse_max_local, glse[bz, by, k])

            for k in T.Pipelined(num_split, num_stages=1):
                if k <= max_split:
                    lse_local_split = glse[bz, by, k]
                    lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            # plain logsumexp over splits (log2 domain)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            # attention sink: add exp(sink) exactly once.  In log2 domain the
            # sink logit is sink*LOG2E; combine with the plain LSE safely.
            sink_ref = T.max(lse_logsum_local, Sinks[by] * LOG2E)
            lse_logsum_local = (
                T.log2(
                    T.exp2(lse_logsum_local - sink_ref)
                    + T.exp2(Sinks[by] * LOG2E - sink_ref)
                )
                + sink_ref
            )
            for k in T.serial(num_split):
                if k <= max_split:
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split = glse[bz, by, k]
                    scale_local = T.exp2(lse_local_split - lse_logsum_local)
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local
            for i in T.Parallel(dim):
                Output[bz, by, i] = o_accum_local[i]

    return main


class V4SparseMLADecode(torch.nn.Module):
    """Runner wrapper around the fused kernel (mirrors SparseFlashAttn)."""

    def __init__(
        self,
        batch,
        heads,
        dim,
        page_block_size,
        block_N,
        index_block,
        num_pages,
        dtype="bf16",
    ):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.dim = dim
        self.block_N = block_N
        self.index_block = index_block
        self.page_block_size = page_block_size
        self.num_pages = num_pages
        self.dtype = dtype
        self.block_H = heads  # 16 heads at TP=4
        props = torch.cuda.get_device_properties(torch.device("cuda:0"))
        self.num_sm = props.multi_processor_count

    def forward(self, q, kv, block_indices, cache_seqlens, block_table, sinks):
        batch = self.batch
        heads = self.heads
        dim = self.dim
        max_selected_blocks = block_indices.shape[-1]

        num_m_blocks = 1 * (heads // 1 + self.block_H - 1) // self.block_H
        num_n_blocks = max_selected_blocks
        size_one_kv_head = max_selected_blocks * self.index_block * (dim + dim) * 2
        total_mblocks = batch * 1 * num_m_blocks

        num_split = num_splits_heuristic(
            total_mblocks,
            self.num_sm,
            num_n_blocks,
            num_m_blocks,
            size_one_kv_head,
            is_causal_or_local=True,
            max_splits=128,
        )

        glse = torch.empty((batch, heads, num_split), dtype=torch.float32, device="cuda")
        output_partial = torch.empty((batch, heads, num_split, dim), dtype=torch.float32, device="cuda")

        kernel = v4_sparse_mla_decode(
            batch,
            heads,
            dim,
            self.block_N,
            self.block_H,
            self.index_block,
            self.page_block_size,
            num_stages=1,
            threads=64,
            num_pages=self.num_pages,
            dtype=self.dtype,
        )
        return kernel(
            q, kv, block_indices, cache_seqlens, block_table, sinks, glse, output_partial
        )


def v4_sparse_mla_decode_ref(
    q,
    kv_cache,
    block_indices,
    cache_seqlens,
    block_table,
    sinks,
    page_block_size,
    index_block,
    sm_scale=None,
):
    """Pure-torch differential oracle for the fused kernel.

    q: [batch, heads, dim]
    kv_cache: [num_pages, page_block_size, dim]  (K==V shared)
    block_indices: [batch, 1, max_selected_blocks] logical 64-token block idx
    cache_seqlens: [batch] int32
    block_table: [batch, max_num_blocks_per_seq] int32
    sinks: [heads] fp32
    """
    batch, heads, dim = q.shape
    num_pages, pbs, _ = kv_cache.shape
    if sm_scale is None:
        sm_scale = 1.0 / (dim**0.5)

    max_seq = int(cache_seqlens.max().item())
    kv_full = torch.zeros((batch, max_seq, dim), dtype=torch.float32, device=q.device)
    for b in range(batch):
        seqlen = int(cache_seqlens[b].item())
        for t in range(seqlen):
            page = int(block_table[b, t // pbs].item())
            off = t % pbs
            kv_full[b, t] = kv_cache[page, off].float()

    out = torch.zeros((batch, heads, dim), dtype=torch.float32, device=q.device)
    for b in range(batch):
        seqlen = int(cache_seqlens[b].item())
        selected = block_indices[b, 0]  # [max_selected_blocks]
        for h in range(heads):
            scores = torch.full((max_seq,), float("-inf"), dtype=torch.float32, device=q.device)
            for idx in selected:
                i = int(idx.item())
                if i >= 0:
                    start = i * index_block
                    end = min(start + index_block, seqlen)
                    for t in range(start, end):
                        scores[t] = torch.dot(q[b, h].float(), kv_full[b, t])
            logits = scores * sm_scale
            finite = logits != float("-inf")
            if not finite.any():
                out[b, h] = 0.0
                continue
            m = logits[finite].max()
            weights = torch.zeros_like(logits)
            weights[finite] = torch.exp(logits[finite] - m)
            sink_w = torch.exp(sinks[h].float() - m)
            norm = weights.sum() + sink_w
            out[b, h] = (weights @ kv_full[b]) / norm
    return out


def build_v4_case(
    batch=1,
    heads=16,
    dim=512,
    seqlen=600,
    page_block_size=256,
    index_block=64,
    num_pages=32,
    seed=42,
    dtype=torch.bfloat16,
):
    """Build a V4-shaped random case: SWA (128) + topk (512) blocks, -1 pad."""
    torch.manual_seed(seed)
    random.seed(seed)
    dev = "cuda"

    q = torch.randn((batch, heads, dim), dtype=dtype, device=dev)
    sinks = torch.randn((heads,), dtype=torch.float32, device=dev) * 2.0

    num_tiles = math.ceil(seqlen / index_block)  # 10 for 600/64
    swa_blocks = list(range(num_tiles - 2, num_tiles))  # last 128 tokens: [8, 9]
    # topk 512 tokens == 8 blocks, choose the oldest 8 distinct blocks
    topk_blocks = list(range(max(0, num_tiles - 10), num_tiles - 2))[-8:]
    selected = swa_blocks + [b for b in topk_blocks if b not in swa_blocks]
    max_selected = len(selected)  # 10

    # logical -> physical page mapping
    max_num_blocks_per_seq = math.ceil(seqlen / page_block_size)  # 3 for 600/256
    block_table = torch.zeros((batch, max_num_blocks_per_seq), dtype=torch.int32, device=dev)
    pages = list(range(max_num_blocks_per_seq))
    random.shuffle(pages)
    for i in range(max_num_blocks_per_seq):
        block_table[0, i] = pages[i]

    kv = torch.randn((batch, seqlen, dim), dtype=dtype, device=dev)
    kv_cache = torch.zeros((num_pages, page_block_size, dim), dtype=dtype, device=dev)
    for b in range(batch):
        for t in range(seqlen):
            page = int(block_table[b, t // page_block_size].item())
            kv_cache[page, t % page_block_size] = kv[b, t]

    block_indices = torch.full((batch, 1, max_selected), -1, dtype=torch.int32, device=dev)
    for j, blk in enumerate(selected):
        block_indices[0, 0, j] = blk

    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=dev)
    return q, kv, kv_cache, block_indices, cache_seqlens, block_table, sinks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=600)
    parser.add_argument("--page_block_size", type=int, default=256)
    parser.add_argument("--index_block", type=int, default=64)
    parser.add_argument("--num_pages", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args()

    tdt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q, kv, kv_cache, block_indices, cache_seqlens, block_table, sinks = build_v4_case(
        batch=args.batch,
        heads=args.heads,
        dim=args.dim,
        seqlen=args.seqlen,
        page_block_size=args.page_block_size,
        index_block=args.index_block,
        num_pages=args.num_pages,
        dtype=tdt,
    )

    model = V4SparseMLADecode(
        args.batch,
        args.heads,
        args.dim,
        args.page_block_size,
        block_N=32,
        index_block=args.index_block,
        num_pages=args.num_pages,
        dtype=args.dtype,
    )

    out = model(q, kv_cache, block_indices, cache_seqlens, block_table, sinks)
    ref = v4_sparse_mla_decode_ref(
        q,
        kv_cache,
        block_indices,
        cache_seqlens,
        block_table,
        sinks,
        args.page_block_size,
        args.index_block,
    )

    max_diff = torch.max(torch.abs(out.float() - ref)).item()
    mean_diff = torch.mean(torch.abs(out.float() - ref)).item()
    print(f"dtype={args.dtype} seqlen={args.seqlen} heads={args.heads} dim={args.dim}")
    print(f"num_split (heuristic): {model.num_sm=} -> blocks={block_indices.shape[-1]}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    if max_diff < args.tolerance:
        print("✓ Verification PASSED: results match within tolerance")
        return 0
    print("✗ Verification FAILED: results differ significantly")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
