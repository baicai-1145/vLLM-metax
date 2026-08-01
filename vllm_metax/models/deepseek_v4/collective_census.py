"""Opt-in, process-local diagnostics for DSpark collective candidates."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class _CensusState:
    owner_task: asyncio.Task | None
    records: list[dict]


_ACTIVE: ContextVar[_CensusState | None] = ContextVar("dsv4_census", default=None)


def _current_task() -> asyncio.Task | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def collective_census(
    projection: str, layer_idx: int, rows: int, expects_reduce: bool
) -> None:
    """Record one invocation when an explicit census context is active."""
    state = _ACTIVE.get()
    if state is None:
        return
    if state.owner_task is not _current_task():
        raise RuntimeError(
            "collective census contexts cannot be shared across asyncio tasks"
        )
    if not isinstance(projection, str) or not projection:
        raise ValueError("projection must be a non-empty string")
    if not isinstance(layer_idx, int) or isinstance(layer_idx, bool) or layer_idx < 0:
        raise ValueError("layer_idx must be a non-negative integer")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ValueError("rows must be a non-negative integer")
    if not isinstance(expects_reduce, bool):
        raise ValueError("expects_reduce must be a bool")
    state.records.append(
        {
            "layer_idx": layer_idx,
            "projection": projection,
            "rows": rows,
            "expects_reduce": expects_reduce,
        }
    )


@contextmanager
def collective_census_context() -> Iterator[list[dict]]:
    """Collect synchronous records; nested contexts are isolated."""
    state = _CensusState(owner_task=_current_task(), records=[])
    token = _ACTIVE.set(state)
    try:
        yield state.records
    finally:
        _ACTIVE.reset(token)


def assert_row_exact(
    reference_rows: torch.Tensor,
    candidate_rows: torch.Tensor,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Raise with the first bad row and maximum absolute/relative errors."""
    if reference_rows.shape != candidate_rows.shape:
        raise AssertionError(
            f"row shape mismatch: reference={tuple(reference_rows.shape)} "
            f"candidate={tuple(candidate_rows.shape)}"
        )
    if reference_rows.dtype != candidate_rows.dtype:
        raise AssertionError(
            f"row dtype mismatch: reference={reference_rows.dtype} "
            f"candidate={candidate_rows.dtype}"
        )
    if reference_rows.ndim == 0:
        ref = reference_rows.reshape(1)
        cand = candidate_rows.reshape(1)
    else:
        ref, cand = reference_rows, candidate_rows
    diff = (cand - ref).abs()
    denom = ref.abs().clamp_min(
        torch.finfo(ref.dtype).tiny if ref.is_floating_point() else 1
    )
    rel = diff / denom
    allowed = diff <= atol + rtol * ref.abs()
    if bool(torch.all(allowed)):
        return
    bad = torch.nonzero(~allowed, as_tuple=False)
    row = int(bad[0, 0]) if bad.numel() else 0
    raise AssertionError(
        f"row {row} mismatch: max_abs={float(diff.max()):.9g} "
        f"max_rel={float(rel.max()):.9g}"
    )


__all__ = [
    "assert_row_exact",
    "collective_census",
    "collective_census_context",
]
