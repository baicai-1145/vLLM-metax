# SPDX-License-Identifier: Apache-2.0
"""Default-off DeepSeek V4 MTP correctness candidate gates."""
from __future__ import annotations

import os

K1_CORRECTNESS_CANDIDATE_ENV = "VLLM_METAX_DSV4_MTP_K1_CORRECTNESS_CANDIDATE"
K1_NATIVE_WQ_B_CANDIDATE_ENV = "VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_CANDIDATE"
K1_NATIVE_WQ_B_LAYERS_ENV = "VLLM_METAX_DSV4_MTP_K1_NATIVE_WQ_B_LAYERS"
K1_NATIVE_O_PROJ_CANDIDATE_ENV = "VLLM_METAX_DSV4_MTP_K1_NATIVE_O_PROJ_CANDIDATE"
K1_NATIVE_FFN_CANDIDATE_ENV = "VLLM_METAX_DSV4_MTP_K1_NATIVE_FFN_CANDIDATE"
K1_NATIVE_MHC_PRE_CANDIDATE_ENV = "VLLM_METAX_DSV4_MTP_K1_NATIVE_MHC_PRE_CANDIDATE"
K1_NATIVE_KV_PRENORM_CANDIDATE_ENV = (
    "VLLM_METAX_DSV4_MTP_K1_NATIVE_KV_PRENORM_CANDIDATE"
)


def k1_correctness_candidate_enabled() -> bool:
    return os.getenv(K1_CORRECTNESS_CANDIDATE_ENV, "0") == "1"


def k1_native_wq_b_candidate_enabled() -> bool:
    return os.getenv(K1_NATIVE_WQ_B_CANDIDATE_ENV, "0") == "1"


def _parse_nonnegative_layer_set(env_name: str) -> set[int] | None:
    value = os.getenv(env_name)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return None
    try:
        selected = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            f"{env_name} must be 'all' or a comma-separated set of "
            "nonnegative integer layer indices"
        ) from exc
    if any(index < 0 for index in selected):
        raise ValueError(
            f"{env_name} must be 'all' or a comma-separated set of "
            "nonnegative integer layer indices"
        )
    return selected


def k1_native_wq_b_candidate_layer_scoped() -> bool:
    if not k1_native_wq_b_candidate_enabled():
        return False
    return _parse_nonnegative_layer_set(K1_NATIVE_WQ_B_LAYERS_ENV) is not None


def k1_native_wq_b_candidate_layer_enabled(layer_idx: int) -> bool:
    if not k1_native_wq_b_candidate_enabled():
        return False
    selected = _parse_nonnegative_layer_set(K1_NATIVE_WQ_B_LAYERS_ENV)
    return selected is None or layer_idx in selected


def k1_native_o_proj_candidate_enabled() -> bool:
    return os.getenv(K1_NATIVE_O_PROJ_CANDIDATE_ENV, "0") == "1"


def k1_native_ffn_candidate_enabled() -> bool:
    return os.getenv(K1_NATIVE_FFN_CANDIDATE_ENV, "0") == "1"


def k1_native_mhc_pre_candidate_enabled() -> bool:
    return os.getenv(K1_NATIVE_MHC_PRE_CANDIDATE_ENV, "0") == "1"


def k1_native_kv_prenorm_candidate_enabled() -> bool:
    return os.getenv(K1_NATIVE_KV_PRENORM_CANDIDATE_ENV, "0") == "1"


def env_or_k1_candidate_enabled(env_name: str) -> bool:
    return os.getenv(env_name, "0") == "1" or k1_correctness_candidate_enabled()
