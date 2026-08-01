# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


_ROW_INDICES_CACHE: dict[
    tuple[str, int | None, torch.dtype], torch.Tensor
] = {}


def _normalize_cache_device(device: torch.device) -> torch.device:
    if device.index is not None or device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        return device
    return torch.device(device.type, torch.cuda.current_device())


def get_cached_row_indices(
    length: int,
    device: torch.device,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Return cached row indices [0, length) for a device and dtype."""
    if length < 0:
        raise ValueError(f"length must be nonnegative, got {length}")

    device = _normalize_cache_device(torch.device(device))
    key = (device.type, device.index, dtype)
    cached = _ROW_INDICES_CACHE.get(key)
    if cached is None or cached.numel() < length:
        cached = torch.arange(length, device=device, dtype=dtype)
        _ROW_INDICES_CACHE[key] = cached
    return cached[:length]
