# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import importlib
import importlib.util
import os
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

from .torch import (
    hc_head_fused_kernel as hc_head_fused_kernel_torch,
    mhc_fused_post_pre as mhc_fused_post_pre_torch,
    mhc_post as mhc_post_torch,
    mhc_pre as mhc_pre_torch,
)

logger = init_logger(__name__)

_MHC_BACKEND = os.getenv("VLLM_METAX_DSV4_MHC_BACKEND", "torch").strip().lower()
_TILELANG_OPS = {
    op.strip()
    for op in os.getenv(
        "VLLM_METAX_DSV4_MHC_TILELANG_OPS", "fused"
    ).split(",")
    if op.strip()
}
_VALID_TILELANG_OPS = {"pre", "post", "fused", "head"}
if not _TILELANG_OPS <= _VALID_TILELANG_OPS:
    unknown = sorted(_TILELANG_OPS - _VALID_TILELANG_OPS)
    logger.warning("Unknown TileLang MHC ops %s; ignoring them", unknown)
    _TILELANG_OPS &= _VALID_TILELANG_OPS
if _MHC_BACKEND not in ("torch", "tilelang"):
    logger.warning(
        "Unknown VLLM_METAX_DSV4_MHC_BACKEND=%s; falling back to torch",
        _MHC_BACKEND,
    )
    _MHC_BACKEND = "torch"


def get_mhc_backend_name() -> str:
    return _MHC_BACKEND


def _tilelang_runtime_ready() -> bool:
    if importlib.util.find_spec("tilelang") is None:
        logger.warning("TileLang MHC backend requested but tilelang is not installed")
        return False
    if importlib.util.find_spec("deep_gemm") is None:
        logger.warning("TileLang MHC backend requested but deep_gemm is not installed")
        return False
    try:
        dg = importlib.import_module("deep_gemm")
    except Exception as exc:
        logger.warning("TileLang MHC backend requested but deep_gemm import failed: %s", exc)
        return False
    if getattr(dg, "tf32_hc_prenorm_gemm", None) is None:
        logger.warning(
            "TileLang MHC backend requested but deep_gemm.tf32_hc_prenorm_gemm is missing"
        )
        return False
    return True


def _load_tilelang() -> tuple[Callable[..., Any], ...]:
    from .tilelang import (
        hc_head_fused_kernel_tilelang,
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_tilelang,
    )

    return (
        mhc_pre_tilelang,
        mhc_post_tilelang,
        mhc_fused_post_pre_tilelang,
        hc_head_fused_kernel_tilelang,
    )


if _MHC_BACKEND == "tilelang" and _tilelang_runtime_ready():
    try:
        (
            mhc_pre_tilelang,
            mhc_post_tilelang,
            mhc_fused_post_pre_tilelang,
            hc_head_fused_kernel_tilelang,
        ) = _load_tilelang()
        mhc_pre = mhc_pre_tilelang if "pre" in _TILELANG_OPS else mhc_pre_torch
        mhc_post = mhc_post_tilelang if "post" in _TILELANG_OPS else mhc_post_torch
        mhc_fused_post_pre = (
            mhc_fused_post_pre_tilelang
            if "fused" in _TILELANG_OPS
            else mhc_fused_post_pre_torch
        )
        hc_head_fused_kernel = (
            hc_head_fused_kernel_tilelang
            if "head" in _TILELANG_OPS
            else hc_head_fused_kernel_torch
        )
        if "pre" in _TILELANG_OPS:
            logger.warning(
                "TileLang mhc_pre uses MetaX DeepGEMM and is not graph-replay "
                "safe; enable it only with enforce_eager"
            )
        logger.info("DeepSeek V4 MHC backend: tilelang ops=%s", sorted(_TILELANG_OPS))
    except Exception as exc:
        logger.warning(
            "Failed to enable TileLang MHC backend (%s); falling back to torch",
            exc,
        )
        mhc_pre = mhc_pre_torch
        mhc_post = mhc_post_torch
        mhc_fused_post_pre = mhc_fused_post_pre_torch
        hc_head_fused_kernel = hc_head_fused_kernel_torch
        _MHC_BACKEND = "torch"
else:
    mhc_pre = mhc_pre_torch
    mhc_post = mhc_post_torch
    mhc_fused_post_pre = mhc_fused_post_pre_torch
    hc_head_fused_kernel = hc_head_fused_kernel_torch
    if _MHC_BACKEND == "tilelang":
        _MHC_BACKEND = "torch"
