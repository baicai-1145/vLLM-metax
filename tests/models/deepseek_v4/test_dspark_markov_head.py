import pytest
import torch
import torch.nn.functional as F

from vllm_metax.models.deepseek_v4.dspark import (
    _ReplicatedMarkovEmbedding,
    _ReplicatedMarkovHead,
)


def test_replicated_markov_head_matches_full_weight_arithmetic():
    torch.manual_seed(101)
    head = _ReplicatedMarkovHead(
        vocab_size=17,
        draft_vocab_size=19,
        rank=4,
        params_dtype=torch.bfloat16,
        prefix="markov_head",
    )
    w1 = torch.randn(17, 4, dtype=torch.bfloat16)
    w2 = torch.randn(19, 4, dtype=torch.bfloat16)
    _ReplicatedMarkovEmbedding.weight_loader(head.markov_w1.weight, w1)
    head.markov_w2.weight_loader(head.markov_w2.weight, w2)
    token_ids = torch.tensor([0, 8, 16])

    embedded = head.embed(token_ids)
    actual = head.bias(
        embedded,
        logits_processor=lambda *_args: (_ for _ in ()).throw(
            AssertionError("replicated head must not gather logits")
        ),
    )

    torch.testing.assert_close(embedded, F.embedding(token_ids, w1))
    torch.testing.assert_close(actual, F.linear(embedded, w2))
    assert set(head.state_dict()) == {"markov_w1.weight", "markov_w2.weight"}


def test_replicated_markov_embedding_rejects_wrong_weight_shape():
    embedding = _ReplicatedMarkovEmbedding(17, 4, torch.bfloat16)

    with pytest.raises(ValueError, match="weight shape mismatch"):
        embedding.weight_loader(
            embedding.weight, torch.empty(16, 4, dtype=torch.bfloat16)
        )

