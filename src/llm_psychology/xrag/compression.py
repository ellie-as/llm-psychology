"""xRAG-style context compression.

This module ports the core xRAG mechanism used in the
hippocampal-neocortical-RAG codebase:

1. Register ``<xRAG>`` as a single tokenizer token.
2. Encode retrieved documents as dense retrieval vectors.
3. Project retrieval vectors into the language-model hidden space.
4. Replace the ``<xRAG>`` token embedding with the projected retrieval vector.

The high-level ``llm_psychology.xrag.XRAG`` wrapper loads the original
Hannibal046/xrag-7b and Salesforce/SFR-Embedding-Mistral weights. The helpers
here expose the same token replacement and retrieval-selection mechanics for
tests and lower-level experiments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


XRAG_TOKEN = "<xRAG>"


@dataclass
class XRAGCompressionResult:
    """One-token retrieval compression result."""

    embedding: np.ndarray
    selected_index: int
    distances: np.ndarray


class XRAGBridge(nn.Module):
    """Project retriever embeddings into the language-model hidden space.

    This mirrors the projector construction in the source xRAG model:
    ``mlpNx_gelu`` means one linear retriever-to-LM layer followed by ``N-1``
    GELU + LM-to-LM linear blocks.
    """

    def __init__(
        self,
        retriever_hidden_size: int,
        lm_hidden_size: int,
        *,
        projector_type: str = "mlp2x_gelu",
    ):
        super().__init__()
        match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if not match:
            raise ValueError(f"Unsupported projector_type: {projector_type!r}")
        depth = int(match.group(1))
        modules: list[nn.Module] = [nn.Linear(retriever_hidden_size, lm_hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(lm_hidden_size, lm_hidden_size))
        self.projector = nn.Sequential(*modules)
        self.retriever_hidden_size = int(retriever_hidden_size)
        self.lm_hidden_size = int(lm_hidden_size)
        self.projector_type = projector_type

    def forward(self, retrieval_embeddings: torch.Tensor) -> torch.Tensor:
        return self.projector(retrieval_embeddings)


def register_xrag_token(tokenizer, model=None, *, xrag_token: str = XRAG_TOKEN) -> int:
    """Register ``xrag_token`` as a single token and optionally resize a model."""
    if hasattr(tokenizer, "add_special_tokens"):
        added = tokenizer.add_special_tokens({"additional_special_tokens": [xrag_token]})
    else:
        added = tokenizer.add_tokens([xrag_token])
    if model is not None and added > 0:
        model.resize_token_embeddings(len(tokenizer))
    xrag_token_id = tokenizer.convert_tokens_to_ids(xrag_token)
    if xrag_token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"{xrag_token!r} mapped to UNK; token registration failed.")
    if model is not None and hasattr(model, "set_xrag_token_id"):
        model.set_xrag_token_id(xrag_token_id)
    probe_ids = tokenizer(f"Background: {xrag_token}", return_tensors="pt").input_ids
    if not (probe_ids == xrag_token_id).any().item():
        raise RuntimeError(f"{xrag_token!r} was not found as a single token in a prompt.")
    return int(xrag_token_id)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool decoder-style embedding models by the final non-padding token."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


@torch.no_grad()
def get_retrieval_embeds(retriever, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Call a retriever's document embedding API, matching the source xRAG code."""
    if hasattr(retriever, "get_doc_embedding"):
        return retriever.get_doc_embedding(input_ids=input_ids, attention_mask=attention_mask).view(input_ids.shape[0], -1)
    outputs = retriever(input_ids=input_ids, attention_mask=attention_mask)
    return last_token_pool(outputs.last_hidden_state, attention_mask)


@torch.no_grad()
def prepare_retrieval_embeds(
    backgrounds: Sequence[str],
    retriever,
    tokenizer,
    *,
    batch_size: int = 16,
    max_length: int = 180,
) -> list[torch.Tensor]:
    """Encode background strings as one dense retrieval embedding each."""
    device = next(retriever.parameters()).device
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(backgrounds), batch_size):
        batch = list(backgrounds[start : start + batch_size])
        tokenized = tokenizer(batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        embeds = get_retrieval_embeds(
            retriever,
            input_ids=tokenized["input_ids"].to(device),
            attention_mask=tokenized["attention_mask"].to(device),
        ).detach().cpu()
        embeddings.extend(embeds[index] for index in range(embeds.shape[0]))
    return embeddings


def xrag_prompt(question: str, *, xrag_token: str = XRAG_TOKEN) -> str:
    """Prompt template copied from the local xRAG wrapper."""
    return (
        "<s>[INST] Refer to the background document and answer the question.\n\n"
        f"Background: {xrag_token}\n\nQuestion: {question} [/INST] The answer is: "
    )


def prepare_inputs_embeds(
    model,
    input_ids: torch.Tensor,
    retrieval_embeds: torch.Tensor,
    *,
    xrag_token_id: int,
    projector: Optional[nn.Module] = None,
    retriever_hidden_size: Optional[int] = None,
) -> torch.Tensor:
    """Replace ``<xRAG>`` token embeddings with projected retrieval embeddings."""
    input_embedding_layer = model.get_input_embeddings()
    inputs_embeds = input_embedding_layer(input_ids)

    if retriever_hidden_size is None:
        retriever_hidden_size = int(retrieval_embeds.shape[-1])
    retrieval_embeds = retrieval_embeds.view(-1, retriever_hidden_size)

    num_xrag_tokens = torch.sum(input_ids == xrag_token_id).item()
    num_retrieval_embeds = retrieval_embeds.shape[0]
    if num_xrag_tokens != num_retrieval_embeds:
        raise ValueError(
            "The number of <xRAG> tokens must match retrieval embeddings: "
            f"{num_xrag_tokens} tokens vs {num_retrieval_embeds} embeddings."
        )

    if projector is not None:
        projector_param = next(projector.parameters())
        retrieval_embeds = projector(
            retrieval_embeds.to(device=projector_param.device, dtype=projector_param.dtype)
        ).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    elif retrieval_embeds.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError(
            "retrieval_embeds must already match the LM hidden size when no projector is supplied "
            f"({retrieval_embeds.shape[-1]} != {inputs_embeds.shape[-1]})."
        )
    else:
        retrieval_embeds = retrieval_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    inputs_embeds[input_ids == xrag_token_id] = retrieval_embeds
    return inputs_embeds


def build_xrag_prompt_inputs(
    model,
    tokenizer,
    prompt: str,
    retrieval_embeds: torch.Tensor | np.ndarray,
    *,
    projector: Optional[nn.Module] = None,
    xrag_token_id: Optional[int] = None,
    device: Optional[torch.device | str] = None,
) -> dict[str, torch.Tensor]:
    """Tokenize an xRAG prompt and build ``inputs_embeds`` for a forward pass."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    if xrag_token_id is None:
        xrag_token_id = register_xrag_token(tokenizer, model=model)
    if projector is not None:
        projector.to(device)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    retrieval_tensor = torch.as_tensor(retrieval_embeds, dtype=torch.float32, device=device)
    inputs_embeds = prepare_inputs_embeds(
        model,
        input_ids,
        retrieval_tensor,
        xrag_token_id=int(xrag_token_id),
        projector=projector,
    )
    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }


def append_target_tokens(model, tokenizer, xrag_inputs: dict[str, torch.Tensor], target_text: str) -> dict[str, torch.Tensor]:
    """Append target text and mask the prompt for bridge-training losses."""
    device = xrag_inputs["inputs_embeds"].device
    target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    target_embeds = model.get_input_embeddings()(target_ids)
    inputs_embeds = torch.cat([xrag_inputs["inputs_embeds"], target_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    labels = torch.full(attention_mask.shape, -100, dtype=torch.long, device=device)
    prompt_len = xrag_inputs["inputs_embeds"].shape[1]
    labels[:, prompt_len:] = target_ids
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@torch.no_grad()
def generate_with_xrag(
    model,
    tokenizer,
    prompt: str,
    retrieval_embeds: torch.Tensor | np.ndarray,
    *,
    projector: Optional[nn.Module] = None,
    xrag_token_id: Optional[int] = None,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    **generate_kwargs,
):
    """Generate with xRAG embeddings using the source method when available."""
    if xrag_token_id is None:
        xrag_token_id = register_xrag_token(tokenizer, model=model)
    device = next(model.parameters()).device
    if projector is not None:
        projector.to(device)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    retrieval_tensor = torch.as_tensor(retrieval_embeds, dtype=torch.float32, device=device)

    if hasattr(model, "set_xrag_token_id") and projector is None:
        model.set_xrag_token_id(int(xrag_token_id))
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            retrieval_embeds=retrieval_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )

    inputs_embeds = prepare_inputs_embeds(
        model,
        input_ids,
        retrieval_tensor,
        xrag_token_id=int(xrag_token_id),
        projector=projector,
    )
    return model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        **generate_kwargs,
    )


def compress_retrieval_embeddings(
    doc_embeddings: np.ndarray | torch.Tensor,
    query_embedding: np.ndarray | torch.Tensor,
    *,
    projector: Optional[nn.Module] = None,
) -> XRAGCompressionResult:
    """Select the nearest raw document embedding in xRAG space.

    This mirrors ``_nearest_doc_embed`` in the source wrapper: compare projected
    query and projected document embeddings, but return the raw document
    embedding because the language model applies the projector during forward.
    """
    doc_tensor = torch.as_tensor(doc_embeddings, dtype=torch.float32)
    query_tensor = torch.as_tensor(query_embedding, dtype=torch.float32).reshape(1, -1)
    if doc_tensor.ndim != 2:
        raise ValueError("doc_embeddings must be a 2D matrix.")
    if query_tensor.shape[1] != doc_tensor.shape[1]:
        raise ValueError("query_embedding and doc_embeddings must have the same hidden size.")

    with torch.no_grad():
        if projector is not None:
            device = next(projector.parameters()).device
            projected_docs = projector(doc_tensor.to(device)).detach().cpu().float()
            projected_query = projector(query_tensor.to(device)).detach().cpu().float()
        else:
            projected_docs = doc_tensor.float()
            projected_query = query_tensor.float()
        distances = torch.cdist(projected_query, projected_docs, p=2).reshape(-1)
        selected_index = int(torch.argmin(distances).item())
    return XRAGCompressionResult(
        embedding=doc_tensor[selected_index].detach().cpu().numpy().astype(np.float32),
        selected_index=selected_index,
        distances=distances.detach().cpu().numpy().astype(np.float32),
    )


def token_count(tokenizer, text: str) -> int:
    return int(len(tokenizer(text, add_special_tokens=False).input_ids))


def xrag_prompt_token_count(tokenizer, prompt: str, *, xrag_token: str = XRAG_TOKEN) -> int:
    xrag_id = tokenizer.convert_tokens_to_ids(xrag_token)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if xrag_id not in ids:
        raise ValueError(f"{xrag_token!r} is not present in the prompt.")
    return int(len(ids))


def compression_ratio(text_tokens: int, xrag_tokens: int = 1) -> float:
    if xrag_tokens <= 0:
        raise ValueError("xrag_tokens must be positive.")
    return float(text_tokens) / float(xrag_tokens)
