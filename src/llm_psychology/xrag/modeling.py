"""Model classes used by the xRAG checkpoint.

These classes are adapted from the local xRAG code in the
hippocampal-neocortical-RAG repository so the library can load the same
Hannibal046/xrag-7b weights directly.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn
from transformers import MistralConfig, MistralForCausalLM, MistralModel


class XMistralConfig(MistralConfig):
    def __init__(
        self,
        projector_type: str = "mlp2x_gelu",
        retriever_hidden_size: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", config.projector_type)
        if not mlp_gelu_match:
            raise ValueError(f"Unsupported projector_type: {config.projector_type!r}")
        mlp_depth = int(mlp_gelu_match.group(1))
        modules: list[nn.Module] = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.projector = nn.Sequential(*modules)

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        return self.projector(context_embedding)


class XMistralForCausalLM(MistralForCausalLM):
    config_class = XMistralConfig

    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, "retriever_hidden_size") and config.retriever_hidden_size > 0:
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size
        self.post_init()

    def set_xrag_token_id(self, token_id: int):
        self.xrag_token_id = token_id

    def prepare_inputs_embeds(self, input_ids: torch.Tensor, retrieval_embeds: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)

        num_xrag_tokens = torch.sum(input_ids == self.xrag_token_id).item()
        num_retrieval_embeds = retrieval_embeds.shape[0]
        assert num_xrag_tokens == num_retrieval_embeds, (num_xrag_tokens, num_retrieval_embeds)

        retrieval_embeds = self.projector(retrieval_embeds.to(inputs_embeds.dtype))
        inputs_embeds[input_ids == self.xrag_token_id] = retrieval_embeds
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        retrieval_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        at_the_beginning_of_generation = False
        if inputs_embeds is not None:
            assert not self.training
            assert retrieval_embeds is None
            at_the_beginning_of_generation = True

        if not at_the_beginning_of_generation and retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1], (inputs_embeds.shape, attention_mask.shape)

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        retrieval_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")

        if retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1], (inputs_embeds.shape, attention_mask.shape)
            return super().generate(attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

        return super().generate(attention_mask=attention_mask, input_ids=input_ids, **kwargs)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class SFR(MistralModel):
    def get_embed_dim(self) -> int:
        return self.config.hidden_size

    def get_embed_length(self) -> int:
        return 1

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return last_token_pool(outputs.last_hidden_state, attention_mask)

    def get_doc_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(input_ids, attention_mask)

    def get_query_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(input_ids, attention_mask)
