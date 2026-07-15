# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Opt-in CPU affinity for vLLM TP worker processes.
# -----------------------------------------------

"""Opt-in CPU affinity for local TP workers."""

from __future__ import annotations

import logging
import operator
import os
from typing import Any

from vllm.v1.worker.gpu_worker import Worker


_ENV_NAME = "VLLM_METAX_TP_WORKER_CPU_AFFINITY"
_logged_affinities: set[tuple[int, tuple[int, ...]]] = set()
logger = logging.getLogger(__name__)


def _parse_cpu_list(value: str) -> set[int]:
    """Parse one rank's CPU list, raising ``ValueError`` on bad input."""
    cpus: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError("CPU affinity contains an empty CPU entry")
        if "-" in token:
            bounds = [part.strip() for part in token.split("-")]
            if len(bounds) != 2 or not all(part.isdigit() for part in bounds):
                raise ValueError(f"Invalid CPU range: {token!r}")
            start, end = (int(part) for part in bounds)
            if start > end:
                raise ValueError(f"Invalid CPU range: {token!r}")
            cpus.update(range(start, end + 1))
        elif token.isdigit():
            cpus.add(int(token))
        else:
            raise ValueError(f"Invalid CPU entry: {token!r}")
    if not cpus:
        raise ValueError("CPU affinity list must not be empty")
    return cpus


def _cpus_for_rank(spec: str, local_rank: Any) -> set[int]:
    """Return the configured CPU set for ``local_rank``."""
    try:
        rank = operator.index(local_rank)
    except TypeError as exc:
        raise ValueError(f"Invalid local_rank for CPU affinity: {local_rank!r}") from exc
    if isinstance(local_rank, bool) or rank < 0:
        raise ValueError(f"Invalid local_rank for CPU affinity: {local_rank!r}")

    rank_specs = spec.split(";")
    if rank >= len(rank_specs):
        raise ValueError(
            f"CPU affinity has no entry for local_rank {rank} (configured ranks: "
            f"{len(rank_specs)})"
        )
    parsed_specs = [_parse_cpu_list(rank_spec) for rank_spec in rank_specs]
    return parsed_specs[rank]


def _log_affinity_once(local_rank: int, cpus: set[int]) -> None:
    key = (local_rank, tuple(sorted(cpus)))
    if key in _logged_affinities:
        return
    _logged_affinities.add(key)
    message = (
        "vLLM MetaX TP worker CPU affinity: "
        f"local_rank={local_rank} cpus={list(key[1])}"
    )
    logger.info(message)
    print(message, flush=True)


def _apply_affinity(local_rank: Any) -> None:
    """Apply the configured affinity, or leave the default unchanged."""
    spec = os.environ.get(_ENV_NAME)
    if spec is None or not spec.strip():
        return
    cpus = _cpus_for_rank(spec, local_rank)
    rank = operator.index(local_rank)
    os.sched_setaffinity(0, cpus)
    _log_affinity_once(rank, cpus)


class AffinityWorker(Worker):
    """GPU worker that applies opt-in CPU affinity before device setup."""

    def init_device(self):
        _apply_affinity(self.local_rank)
        return super().init_device()
