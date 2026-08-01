from types import SimpleNamespace
import json
from pathlib import Path

import pytest

import torch

from vllm_metax.patch.bugfix import mtp_target_runtime_capture as patch


@pytest.fixture(autouse=True)
def _rank_env(monkeypatch):
    monkeypatch.setenv("RANK", "0")


def _diag_path(tmp_path: Path, filename: str) -> Path:
    path = Path(filename)
    return tmp_path / f"{path.stem}.rank0{path.suffix or '.jsonl'}"


class _Buffer:
    def __init__(self, values):
        self.gpu = torch.as_tensor(values)
        self.cpu = self.gpu.clone()


def _runner(groups=2):
    kv_groups = [
        SimpleNamespace(
            block_table=_Buffer([[10 + i, 11 + i], [20 + i, 21 + i], [90, 91]]),
            slot_mapping=_Buffer([100 + i, 101 + i, 102 + i, 103 + i]),
        )
        for i in range(groups)
    ]
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=2,
            num_tokens=4,
            block_table=SimpleNamespace(block_tables=kv_groups),
        ),
        query_start_loc=_Buffer([0, 2, 4]),
        input_ids=_Buffer([31, 32, 41, 42]),
        positions=torch.tensor([7, 8, 9, 10]),
        seq_lens=torch.tensor([8, 10]),
        num_scheduled_tokens=_Buffer([2, 2]),
        req_indices=_Buffer([0, 0, 1, 1]),
        num_spec_tokens=1,
        prev_num_spec_tokens=1,
        execute_model_state=SimpleNamespace(
            sample_hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            logits=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            spec_decode_metadata=None,
        ),
    )


def test_inert_without_env_or_state(monkeypatch):
    calls = []
    monkeypatch.delenv(patch._CAPTURE_ENV, raising=False)
    monkeypatch.setattr(patch, "_ORIGINAL_SAMPLE_TOKENS", lambda *a, **k: calls.append(a) or "ok")
    runner = _runner()
    assert patch._sample_tokens(runner, "grammar") == "ok"
    assert calls and calls[0][1] == "grammar"
    assert not hasattr(runner, patch._COUNTER_ATTR)
    runner.execute_model_state = None
    monkeypatch.setenv(patch._CAPTURE_ENV, "/tmp/capture")
    assert patch._sample_tokens(runner, None) == "ok"
    assert not hasattr(runner, patch._COUNTER_ATTR)


def test_capture_preserves_call_and_records_mtp0_and_serial_mtp1(monkeypatch):
    records = []
    monkeypatch.setenv(patch._CAPTURE_ENV, "/tmp/capture")
    monkeypatch.setattr(patch, "maybe_capture_mtp_stage", lambda s, n, f: records.append((s, n, f)))
    expected = object()
    monkeypatch.setattr(patch, "_ORIGINAL_SAMPLE_TOKENS", lambda *a, **k: (a, k, expected))
    runner = _runner()
    assert patch._sample_tokens(runner, "g", value=3)[2] is expected
    runner.execute_model_state.spec_decode_metadata = None
    assert len(records) == 3
    assert records[0][0] == "v1_target_runtime"
    assert [item[0] for item in records[1:]] == [
        "v1_target_runtime_kv_group_0",
        "v1_target_runtime_kv_group_1",
    ]
    fields = records[0][2]
    assert fields["input_ids"].tolist() == [32, 42]
    assert fields["positions"].tolist() == [8, 10]
    assert fields["row_indices"].tolist() == [1, 3]
    assert fields["num_scheduled_tokens"].tolist() == [2, 2]
    assert fields["request_slot_mapping"].tolist() == [0, 0, 1, 1]
    assert fields["num_speculative_tokens"] == 1
    assert fields["prev_num_speculative_tokens"] == 1
    assert records[1][2]["block_table"].tolist() == [[10, 11], [20, 21]]
    assert records[1][2]["slot_mapping"].tolist() == [100, 101, 102, 103]


def test_non_serial_metadata_is_skipped(monkeypatch):
    records = []
    monkeypatch.setenv(patch._CAPTURE_ENV, "/tmp/capture")
    monkeypatch.setattr(patch, "maybe_capture_mtp_stage", lambda *a, **k: records.append(a))
    monkeypatch.setattr(patch, "_ORIGINAL_SAMPLE_TOKENS", lambda *a, **k: "ok")
    runner = _runner(1)
    runner.execute_model_state.spec_decode_metadata = object()
    assert patch._sample_tokens(runner, None) == "ok"
    assert records == []
    assert getattr(runner, patch._COUNTER_ATTR) == 1


def test_capture_step_limit_is_configurable(monkeypatch):
    attempts = []
    monkeypatch.setenv(patch._CAPTURE_ENV, "/tmp/capture")
    monkeypatch.setenv(patch._STEPS_ENV, "2")
    monkeypatch.setattr(patch, "_capture_target_runtime", lambda self, step: attempts.append(step))
    monkeypatch.setattr(patch, "_ORIGINAL_SAMPLE_TOKENS", lambda *a, **k: "ok")
    runner = _runner(0)
    for _ in range(4):
        patch._sample_tokens(runner, None)
    assert attempts == [0, 1]
    assert getattr(runner, patch._COUNTER_ATTR) == 4


def test_capture_step_window_preserves_global_step(monkeypatch):
    attempts = []
    monkeypatch.setenv(patch._CAPTURE_ENV, "/tmp/capture")
    monkeypatch.setenv(patch._START_STEP_ENV, "2")
    monkeypatch.setenv(patch._STEPS_ENV, "2")
    monkeypatch.setattr(patch, "_capture_target_runtime", lambda self, step: attempts.append(step))
    monkeypatch.setattr(patch, "_ORIGINAL_SAMPLE_TOKENS", lambda *a, **k: "ok")
    runner = _runner(0)
    for _ in range(5):
        patch._sample_tokens(runner, None)
    assert attempts == [2, 3]
    assert getattr(runner, patch._COUNTER_ATTR) == 5


def _kv_manager(probe):
    specs = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=16), is_eagle_group=False
            ),
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=32), is_eagle_group=True
            ),
        ]
    )
    return SimpleNamespace(
        coordinator=SimpleNamespace(
            find_longest_cache_hit_per_group=probe,
            kv_cache_config=specs,
        )
    )


def test_prefix_hit_diagnostic_is_inert_without_env(monkeypatch, tmp_path):
    probes = []
    calls = []
    monkeypatch.delenv(patch._CAPTURE_ENV, raising=False)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_GET_COMPUTED_BLOCKS",
        lambda manager, request: calls.append(request) or ("blocks", 7),
    )
    manager = _kv_manager(lambda *args: probes.append(args))
    request = SimpleNamespace(request_id="req-0", num_tokens=9, block_hashes=[1, 2])

    assert patch._get_computed_blocks(manager, request) == ("blocks", 7)
    assert calls == [request]
    assert probes == []
    assert not _diag_path(tmp_path, "prefix_hits.jsonl").exists()


def test_prefix_hit_diagnostic_records_per_group_hits(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    calls = []

    def probe(block_hashes, max_length):
        calls.append((block_hashes, max_length))
        return ([['a', 'b'], ['c']], (32, 32))

    monkeypatch.setattr(
        patch,
        "_ORIGINAL_GET_COMPUTED_BLOCKS",
        lambda manager, request: ("final-blocks", 24),
    )
    manager = _kv_manager(probe)
    request = SimpleNamespace(request_id="req-1", num_tokens=41, block_hashes=[3, 4])

    assert patch._get_computed_blocks(manager, request) == ("final-blocks", 24)
    assert calls == [([3, 4], 40)]
    records = [json.loads(line) for line in _diag_path(tmp_path, "prefix_hits.jsonl").read_text().splitlines()]
    assert records == [
        {
            "coordinator_type": "SimpleNamespace",
            "final_hit_length": 24,
            "final_num_computed_tokens": 24,
            "group_specs": [
                {"block_size": 16, "spec_type": "SimpleNamespace", "use_eagle": False},
                {"block_size": 32, "spec_type": "SimpleNamespace", "use_eagle": True},
            ],
            "group_spec_types": ["SimpleNamespace", "SimpleNamespace"],
            "group_block_sizes": [16, 32],
            "group_use_eagle": [False, True],
            "max_cache_hit_length": 40,
            "num_tokens": 41,
            "per_group_block_counts": [2, 1],
            "per_group_hit_lengths": [32, 32],
            "request_id": "req-1",
            "rank": 0,
            "use_eagle": None,
        }
    ]


def test_prefix_cache_capture_is_independent_from_mtp_tensor_capture(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(patch._CAPTURE_ENV, raising=False)
    monkeypatch.setenv(patch._PREFIX_CAPTURE_ENV, str(tmp_path))
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_GET_COMPUTED_BLOCKS",
        lambda manager, request: ("final-blocks", 0),
    )
    monkeypatch.setattr(patch, "_ORIGINAL_SAMPLE_TOKENS", lambda *a, **k: "ok")
    manager = _kv_manager(lambda block_hashes, max_length: ([[], []], (0, 0)))
    request = SimpleNamespace(request_id="req-prefix-only", num_tokens=9, block_hashes=[])
    runner = _runner(0)

    assert patch._get_computed_blocks(manager, request) == ("final-blocks", 0)
    assert patch._sample_tokens(runner, None) == "ok"
    assert _diag_path(tmp_path, "prefix_hits.jsonl").exists()
    assert not hasattr(runner, patch._COUNTER_ATTR)


def test_prefix_hit_probe_failure_does_not_change_original(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    calls = []

    def failing_probe(*args):
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(
        patch,
        "_ORIGINAL_GET_COMPUTED_BLOCKS",
        lambda manager, request: calls.append(request) or ("unchanged", 11),
    )
    manager = _kv_manager(failing_probe)
    request = SimpleNamespace(request_id="req-2", num_tokens=13, block_hashes=[])

    assert patch._get_computed_blocks(manager, request) == ("unchanged", 11)
    assert calls == [request]
    assert not _diag_path(tmp_path, "prefix_hits.jsonl").exists()


class _CacheBlock:
    def __init__(self, block_id, ref_cnt=0, is_null=False, block_hash=None, hash_tokens=None):
        self.block_id = block_id
        self.ref_cnt = ref_cnt
        self.is_null = is_null
        self.block_hash = block_hash
        self.block_hash_num_tokens = hash_tokens


class _CacheManager:
    def __init__(self):
        self.kv_cache_spec = SimpleNamespace(block_size=64)
        self.kv_cache_group_id = 3
        self.use_eagle = True
        self.scheduler_block_size = 128
        self.num_cached_block = {"req-cache": 5}
        self.req_to_blocks = {
            "req-cache": [
                _CacheBlock(0),
                _CacheBlock(1, ref_cnt=2, block_hash="h1", hash_tokens=64),
                _CacheBlock(2, ref_cnt=1, block_hash="h2", hash_tokens=128),
                _CacheBlock(3),
                _CacheBlock(4),
                _CacheBlock(5, ref_cnt=1, block_hash="h5", hash_tokens=320),
                _CacheBlock(6),
                _CacheBlock(7),
                _CacheBlock(8),
                _CacheBlock(9),
            ]
        }
        self.block_pool = SimpleNamespace(
            free_block_queue=SimpleNamespace(num_free_blocks=17)
        )

    def reachable_block_mask(self, **kwargs):
        assert kwargs["start_block"] == 5
        assert kwargs["end_block"] == 8
        return [True, False, True]


def test_cache_blocks_diagnostic_is_inert_and_preserves_original(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv(patch._CAPTURE_ENV, raising=False)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_CACHE_BLOCKS",
        lambda manager, request, num_tokens, retention_interval=None: calls.append(
            (request, num_tokens, retention_interval)
        ),
    )
    manager = _CacheManager()
    request = SimpleNamespace(request_id="req-cache", num_prompt_tokens=512)

    assert patch._cache_blocks(manager, request, 512, retention_interval=0) is None
    assert calls == [(request, 512, 0)]
    assert not _diag_path(tmp_path, "cache_blocks.jsonl").exists()


def test_cache_blocks_records_reachable_mask_and_before_after(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    calls = []

    def original(manager, request, num_tokens, retention_interval=None):
        calls.append((request, num_tokens, retention_interval))
        manager.num_cached_block[request.request_id] = 8
        manager.req_to_blocks[request.request_id][6].block_hash = "h6"
        manager.req_to_blocks[request.request_id][6].block_hash_num_tokens = 384
        manager.block_pool.free_block_queue.num_free_blocks = 14

    monkeypatch.setattr(patch, "_ORIGINAL_CACHE_BLOCKS", original)
    manager = _CacheManager()
    request = SimpleNamespace(request_id="req-cache", num_prompt_tokens=512)

    assert patch._cache_blocks(manager, request, 512, retention_interval=0) is None
    assert calls == [(request, 512, 0)]
    record = json.loads(_diag_path(tmp_path, "cache_blocks.jsonl").read_text())
    assert record["phase"] == "later_decode_or_replay"
    assert record["kv_cache_group_id"] == 3
    assert record["num_cached_blocks_before"] == 5
    assert record["num_cached_blocks_after"] == 8
    assert record["num_full_blocks"] == 8
    assert record["reachable_block_mask_indices"] == [5, 7]
    assert record["free_pool_count_before"] == 17
    assert record["free_pool_count_after"] == 14
    assert record["logical_blocks_before"]["5"]["hash_present"] is True
    assert record["logical_blocks_before"]["6"]["hash_present"] is False
    assert record["logical_blocks_after"]["6"]["hash_present"] is True


def test_cache_blocks_marks_initial_prefill_commit(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_CACHE_BLOCKS",
        lambda manager, request, num_tokens, retention_interval=None: None,
    )
    manager = _CacheManager()
    manager.num_cached_block["req-cache"] = 0
    request = SimpleNamespace(request_id="req-cache", num_prompt_tokens=512)

    patch._cache_blocks(manager, request, 512)
    record = json.loads(_diag_path(tmp_path, "cache_blocks.jsonl").read_text())
    assert record["phase"] == "initial_prefill_commit"


def test_cache_blocks_diagnostic_failure_does_not_change_original(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    calls = []
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_CACHE_BLOCKS",
        lambda manager, request, num_tokens, retention_interval=None: calls.append(1)
        or (_ for _ in ()).throw(RuntimeError("original failure")),
    )
    manager = _CacheManager()
    manager.reachable_block_mask = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("diagnostic failure")
    )
    request = SimpleNamespace(request_id="req-cache", num_prompt_tokens=512)

    try:
        patch._cache_blocks(manager, request, 512)
    except RuntimeError as exc:
        assert "original failure" in str(exc)
    else:
        raise AssertionError("original exception must be preserved")
    assert calls == [1]
    assert not _diag_path(tmp_path, "cache_blocks.jsonl").exists()


def test_eviction_diagnostic_is_inert_and_preserves_result(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv(patch._CAPTURE_ENV, raising=False)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_MAYBE_EVICT",
        lambda pool, block: calls.append(block) or False,
    )
    pool = SimpleNamespace()
    block = _CacheBlock(7, ref_cnt=0, block_hash=b"hash" + b"\x00" * 4, hash_tokens=320)

    assert patch._maybe_evict_cached_block(pool, block) is False
    assert calls == [block]
    assert not _diag_path(tmp_path, "evictions.jsonl").exists()


def test_eviction_diagnostic_records_before_after_and_group_key(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    block_hash = b"h" * 32 + (5).to_bytes(4, "big")
    block = _CacheBlock(7, ref_cnt=0, block_hash=block_hash, hash_tokens=320)
    alias = b"a" * 32 + (9).to_bytes(4, "big")
    pool = SimpleNamespace(
        cached_block_hashes_by_block={7: {alias}},
        free_block_queue=SimpleNamespace(num_free_blocks=12),
    )

    def original(pool_arg, block_arg):
        block_arg.block_hash = None
        block_arg.block_hash_num_tokens = None
        pool_arg.free_block_queue.num_free_blocks = 13
        return True

    monkeypatch.setattr(patch, "_ORIGINAL_MAYBE_EVICT", original)
    assert patch._maybe_evict_cached_block(pool, block) is True
    record = json.loads(_diag_path(tmp_path, "evictions.jsonl").read_text())
    assert record["evicted"] is True
    assert record["before"]["block_id"] == 7
    assert record["before"]["hash_present"] is True
    assert {item["group_id"] for item in record["before"]["hash_keys"]} == {5, 9}
    assert record["before"]["free_pool_count"] == 12
    assert record["after"]["hash_present"] is False
    assert record["after"]["free_pool_count"] == 13


def test_eviction_failure_preserves_original_exception(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    block = _CacheBlock(8, ref_cnt=0, block_hash=b"x" * 36, hash_tokens=640)
    pool = SimpleNamespace(cached_block_hashes_by_block={})

    def original(pool_arg, block_arg):
        raise RuntimeError("eviction failure")

    monkeypatch.setattr(patch, "_ORIGINAL_MAYBE_EVICT", original)
    try:
        patch._maybe_evict_cached_block(pool, block)
    except RuntimeError as exc:
        assert "eviction failure" in str(exc)
    else:
        raise AssertionError("original exception must be preserved")
    assert not _diag_path(tmp_path, "evictions.jsonl").exists()


def test_hash_removal_diagnostic_is_inert(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv(patch._CAPTURE_ENV, raising=False)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_REMOVE_CACHED_HASHES",
        lambda pool, block: calls.append(block) or [b"removed"],
    )
    pool = SimpleNamespace()
    block = _CacheBlock(4, block_hash=b"h" * 36, hash_tokens=320)

    assert patch._remove_cached_block_hashes(pool, block) == [b"removed"]
    assert calls == [block]
    assert not _diag_path(tmp_path, "hash_removals.jsonl").exists()


def test_hash_removal_records_caller_and_after_state(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    block_hash = b"h" * 32 + (4).to_bytes(4, "big")
    alias = b"a" * 32 + (6).to_bytes(4, "big")
    block = _CacheBlock(4, ref_cnt=0, block_hash=block_hash, hash_tokens=320)
    pool = SimpleNamespace(cached_block_hashes_by_block={4: {alias}})

    def call_from_cache_path():
        return patch._remove_cached_block_hashes(pool, block)

    def original(pool_arg, block_arg):
        pool_arg.cached_block_hashes_by_block.pop(block_arg.block_id, None)
        block_arg.block_hash = None
        block_arg.block_hash_num_tokens = None
        return [block_hash, alias]

    monkeypatch.setattr(patch, "_ORIGINAL_REMOVE_CACHED_HASHES", original)
    assert call_from_cache_path() == [block_hash, alias]
    record = json.loads(_diag_path(tmp_path, "hash_removals.jsonl").read_text())
    assert record["caller"]["caller"] == "call_from_cache_path"
    assert record["removed_hash_count"] == 2
    assert {item["group_id"] for item in record["removed_hash_keys"]} == {4, 6}
    assert record["before"]["hash_present"] is True
    assert record["after"]["hash_present"] is False


def test_hash_removal_preserves_original_exception(monkeypatch, tmp_path):
    monkeypatch.setenv(patch._CAPTURE_ENV, str(tmp_path))
    pool = SimpleNamespace(cached_block_hashes_by_block={})
    block = _CacheBlock(5, block_hash=b"h" * 36, hash_tokens=128)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_REMOVE_CACHED_HASHES",
        lambda pool_arg, block_arg: (_ for _ in ()).throw(
            RuntimeError("remove failure")
        ),
    )

    try:
        patch._remove_cached_block_hashes(pool, block)
    except RuntimeError as exc:
        assert "remove failure" in str(exc)
    else:
        raise AssertionError("original exception must be preserved")
    assert not _diag_path(tmp_path, "hash_removals.jsonl").exists()
