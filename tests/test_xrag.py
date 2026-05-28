import numpy as np
import pytest
import torch
import torch.nn as nn

from llm_psychology.xrag import (
    DEFAULT_XRAG_LLM,
    DEFAULT_XRAG_RETRIEVER,
    XRAGConfig,
    compress_retrieval_embeddings,
    prepare_inputs_embeds,
)
from llm_psychology.xrag.compression import XRAGBridge


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4)

    def get_input_embeddings(self):
        return self.embedding


def test_xrag_bridge_projection_shape():
    bridge = XRAGBridge(retriever_hidden_size=3, lm_hidden_size=5, projector_type="mlp2x_gelu")
    projected = bridge(torch.zeros(2, 3))
    assert tuple(projected.shape) == (2, 5)


def test_native_xrag_defaults_match_source_repo():
    config = XRAGConfig()
    assert config.llm_name == DEFAULT_XRAG_LLM == "Hannibal046/xrag-7b"
    assert config.retriever_name == DEFAULT_XRAG_RETRIEVER == "Salesforce/SFR-Embedding-Mistral"


def test_compress_retrieval_embeddings_selects_nearest_raw_embedding():
    docs = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    query = np.array([1.8, 0.1], dtype=np.float32)

    result = compress_retrieval_embeddings(docs, query)

    assert result.selected_index == 1
    np.testing.assert_allclose(result.embedding, docs[1])
    assert result.distances.shape == (3,)


def test_prepare_inputs_embeds_replaces_xrag_token_embedding():
    torch.manual_seed(0)
    model = DummyModel()
    bridge = XRAGBridge(retriever_hidden_size=3, lm_hidden_size=4, projector_type="mlp1x_gelu")
    input_ids = torch.tensor([[1, 7, 2]])
    retrieval = torch.ones(1, 3)

    base_embeds = model.get_input_embeddings()(input_ids).detach().clone()
    inputs_embeds = prepare_inputs_embeds(
        model,
        input_ids,
        retrieval,
        xrag_token_id=7,
        projector=bridge,
    )

    assert tuple(inputs_embeds.shape) == (1, 3, 4)
    assert torch.allclose(inputs_embeds[:, 0], base_embeds[:, 0])
    assert not torch.allclose(inputs_embeds[:, 1], base_embeds[:, 1])


def test_prepare_inputs_embeds_requires_one_embedding_per_xrag_token():
    model = DummyModel()
    input_ids = torch.tensor([[7, 1, 7]])
    retrieval = torch.ones(1, 4)

    with pytest.raises(ValueError, match="number of <xRAG> tokens"):
        prepare_inputs_embeds(model, input_ids, retrieval, xrag_token_id=7)
