# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Default-off Plan08 output-side stop-string truncation candidate.
# -----------------------------------------------
from __future__ import annotations

import os

from vllm.v1.engine import detokenizer as _detokenizer


_ORIGINAL_BASE_DETOKENIZER_UPDATE = _detokenizer.BaseIncrementalDetokenizer.update


def _enabled() -> bool:
    return os.getenv("VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION") == "1"


def _decode_output_ids(detok, output_token_ids: list[int]) -> str | None:
    tokenizer = getattr(detok, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return None
    try:
        return tokenizer.decode(
            output_token_ids,
            skip_special_tokens=getattr(detok, "skip_special_tokens", False),
        )
    except TypeError:
        try:
            return tokenizer.decode(output_token_ids)
        except Exception:
            return None
    except Exception:
        return None


def _find_stop_prefix_count(
    detok,
    output_token_ids: list[int],
    first_new_output_count: int,
    stop_string: str,
    output_text: str,
) -> int | None:
    start = max(first_new_output_count + 1, 1)
    for keep_count in range(start, len(output_token_ids) + 1):
        decoded = _decode_output_ids(detok, output_token_ids[:keep_count])
        if decoded is None:
            return None
        if decoded == output_text:
            return keep_count
        stop_index = decoded.find(stop_string)
        if stop_index != -1:
            return keep_count
    return None


def _base_detokenizer_update(self, new_token_ids: list[int],
                             stop_terminated: bool) -> str | None:
    if not _enabled() or len(new_token_ids) <= 1:
        return _ORIGINAL_BASE_DETOKENIZER_UPDATE(self, new_token_ids,
                                                stop_terminated)

    first_new_output_count = self.num_output_tokens()
    stop_string = _ORIGINAL_BASE_DETOKENIZER_UPDATE(self, new_token_ids,
                                                   stop_terminated)
    if not stop_string:
        return stop_string

    output_token_ids = list(self.output_token_ids)
    keep_count = _find_stop_prefix_count(
        self,
        output_token_ids,
        first_new_output_count,
        stop_string,
        self.output_text,
    )
    if keep_count is None or keep_count >= len(output_token_ids):
        return stop_string

    num_tail_tokens = len(output_token_ids) - keep_count
    if num_tail_tokens <= 0 or num_tail_tokens > len(new_token_ids):
        return stop_string

    del new_token_ids[-num_tail_tokens:]
    del self.token_ids[-num_tail_tokens:]
    return stop_string


_detokenizer.BaseIncrementalDetokenizer.update = _base_detokenizer_update
