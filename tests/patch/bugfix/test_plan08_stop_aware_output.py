from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer

import vllm_metax.patch.bugfix.plan08_stop_aware_output  # noqa: F401


_TOKEN_TEXT = {
    223: " ",
    864: "18",
    271: "\n\n",
    10375: "Question",
    28: ":",
    334: " A",
}


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(_TOKEN_TEXT[token_id] for token_id in token_ids)


class _FakeDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, *, include_stop_str_in_output: bool = True):
        self.token_ids = []
        self.stop = ["\n\nQuestion:"]
        self.min_tokens = 0
        self.include_stop_str_in_output = include_stop_str_in_output
        self.stop_buffer_length = 0 if include_stop_str_in_output else 10
        self._last_output_text_offset = 0
        self.output_text = ""
        self.tokenizer = _FakeTokenizer()
        self.skip_special_tokens = False

    def decode_next(self, next_token_id: int) -> str:
        return _TOKEN_TEXT[next_token_id]


def test_stop_aware_output_truncates_same_batch_tail_token_ids(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION", "1")
    detokenizer = _FakeDetokenizer()
    new_token_ids = [223, 864, 271, 10375, 28, 334]

    stop_string = detokenizer.update(new_token_ids, stop_terminated=False)

    assert stop_string == "\n\nQuestion:"
    assert detokenizer.output_text == " 18\n\nQuestion:"
    assert new_token_ids == [223, 864, 271, 10375, 28]
    assert detokenizer.output_token_ids == [223, 864, 271, 10375, 28]


def test_stop_aware_output_truncates_tail_when_stop_string_excluded(monkeypatch):
    monkeypatch.setenv("VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION", "1")
    detokenizer = _FakeDetokenizer(include_stop_str_in_output=False)
    new_token_ids = [223, 864, 271, 10375, 28, 334]

    stop_string = detokenizer.update(new_token_ids, stop_terminated=False)

    assert stop_string == "\n\nQuestion:"
    assert detokenizer.output_text == " 18"
    assert new_token_ids == [223, 864]
    assert detokenizer.output_token_ids == [223, 864]


def test_stop_aware_output_is_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_DSV4_MTP_STOP_AWARE_OUTPUT_TRUNCATION",
                       raising=False)
    detokenizer = _FakeDetokenizer()
    new_token_ids = [223, 864, 271, 10375, 28, 334]

    stop_string = detokenizer.update(new_token_ids, stop_terminated=False)

    assert stop_string == "\n\nQuestion:"
    assert detokenizer.output_text == " 18\n\nQuestion:"
    assert new_token_ids == [223, 864, 271, 10375, 28, 334]
    assert detokenizer.output_token_ids == [223, 864, 271, 10375, 28, 334]
