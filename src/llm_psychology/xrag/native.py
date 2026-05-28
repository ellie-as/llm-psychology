"""Native xRAG runner using the source experiment checkpoints."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from transformers import AutoTokenizer

from .compression import register_xrag_token, xrag_prompt
from .modeling import SFR, XMistralForCausalLM


DEFAULT_XRAG_LLM = "Hannibal046/xrag-7b"
DEFAULT_XRAG_RETRIEVER = "Salesforce/SFR-Embedding-Mistral"


@dataclass
class XRAGConfig:
    llm_name: str = DEFAULT_XRAG_LLM
    retriever_name: str = DEFAULT_XRAG_RETRIEVER
    torch_dtype: torch.dtype = torch.bfloat16
    retriever_batch_size: int = 16
    retriever_max_length: int = 256
    retriever_query_max_length: int = 180
    retriever_query_clip_chars: Optional[int] = None
    docs_per_datastore: Optional[int] = None
    min_new_tokens: int = 10
    no_repeat_ngram_size: int = 0


@dataclass
class XRAGDatastore:
    documents: list[str]
    raw_doc_embeddings: torch.Tensor
    xrag_doc_embeddings: torch.Tensor


@dataclass
class XRAGRetrieval:
    raw_embedding: torch.Tensor
    selected_index: int
    distances: torch.Tensor


@dataclass
class XRAGAnswer:
    prompt: str
    query_embedding: torch.Tensor
    retrieval: XRAGRetrieval
    text: str


def resolve_xrag_device(device: Optional[str | torch.device] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class XRAG:
    """Reusable xRAG wrapper matching the hippocampal-neocortical-RAG flow."""

    def __init__(self, config: Optional[XRAGConfig] = None, device: Optional[str | torch.device] = None):
        self.config = config or XRAGConfig()
        self.device = resolve_xrag_device(device)
        self.llm = None
        self.tokenizer = None
        self.retriever = None
        self.retriever_tokenizer = None
        self.xrag_token_id: Optional[int] = None

    def load(self):
        if self.llm is not None:
            return self

        self.llm = XMistralForCausalLM.from_pretrained(
            self.config.llm_name,
            torch_dtype=self.config.torch_dtype,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_name,
            add_eos_token=False,
            use_fast=False,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.xrag_token_id = register_xrag_token(self.tokenizer, self.llm)

        self.retriever = SFR.from_pretrained(
            self.config.retriever_name,
            torch_dtype=self.config.torch_dtype,
        ).to(self.device).eval()
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(self.config.retriever_name)
        return self

    def release(self):
        for attr in ("llm", "tokenizer", "retriever", "retriever_tokenizer"):
            setattr(self, attr, None)
        self.xrag_token_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _require_loaded(self):
        if self.llm is None or self.tokenizer is None or self.retriever is None or self.retriever_tokenizer is None:
            raise RuntimeError("Call XRAG.load() before using the xRAG model.")

    @torch.no_grad()
    def prepare_datastore(self, documents: Sequence[str]) -> XRAGDatastore:
        self._require_loaded()
        docs = list(documents)
        if self.config.docs_per_datastore is not None:
            docs = docs[: self.config.docs_per_datastore]
        if not docs:
            raise ValueError("documents must contain at least one item.")

        raw_chunks: list[torch.Tensor] = []
        xrag_chunks: list[torch.Tensor] = []
        batch_size = int(self.config.retriever_batch_size)

        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            encoded = self.retriever_tokenizer(
                batch,
                max_length=int(self.config.retriever_max_length),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            doc_emb = self.retriever.get_doc_embedding(**encoded)
            xrag_emb = self.llm.projector(doc_emb)
            raw_chunks.append(doc_emb.detach().cpu())
            xrag_chunks.append(xrag_emb.detach().cpu())
            del encoded, doc_emb, xrag_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return XRAGDatastore(
            documents=docs,
            raw_doc_embeddings=torch.cat(raw_chunks, dim=0),
            xrag_doc_embeddings=torch.cat(xrag_chunks, dim=0),
        )

    @torch.no_grad()
    def prepare_prompt(self, question: str) -> tuple[str, torch.Tensor]:
        self._require_loaded()
        query_text = str(question)
        if self.config.retriever_query_clip_chars is not None:
            query_text = query_text[: int(self.config.retriever_query_clip_chars)]
        else:
            query_text = query_text.split(".")[0]

        encoded = self.retriever_tokenizer(
            query_text,
            max_length=int(self.config.retriever_query_max_length),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        q_emb = self.retriever.get_query_embedding(**encoded)
        q_emb = self.llm.projector(q_emb)
        prompt = xrag_prompt(question)

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if self.xrag_token_id is not None and not (ids == self.xrag_token_id).any().item():
            raise RuntimeError("XRAG token id was not found in the prompt.")
        return prompt, q_emb

    def nearest_doc_embedding(self, query_embedding: torch.Tensor, datastore: XRAGDatastore) -> XRAGRetrieval:
        q_cpu = query_embedding.detach().cpu().float()
        distances = torch.cdist(q_cpu, datastore.xrag_doc_embeddings.float(), p=2).reshape(-1)
        selected_index = int(torch.argmin(distances).item())
        return XRAGRetrieval(
            raw_embedding=datastore.raw_doc_embeddings[selected_index].to(self.device),
            selected_index=selected_index,
            distances=distances.detach().cpu(),
        )

    @torch.no_grad()
    def run_xrag(
        self,
        prompt: str,
        raw_doc_embedding: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        min_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        **generate_kwargs,
    ) -> str:
        self._require_loaded()
        embedding = raw_doc_embedding
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.llm.generate(
            ids,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            min_new_tokens=self.config.min_new_tokens if min_new_tokens is None else min_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            retrieval_embeds=embedding.to(self.device),
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            **generate_kwargs,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def answer(self, question: str, datastore: XRAGDatastore, *, max_new_tokens: int = 64, **generate_kwargs) -> XRAGAnswer:
        prompt, query_embedding = self.prepare_prompt(question)
        retrieval = self.nearest_doc_embedding(query_embedding, datastore)
        text = self.run_xrag(prompt, retrieval.raw_embedding, max_new_tokens=max_new_tokens, **generate_kwargs)
        return XRAGAnswer(prompt=prompt, query_embedding=query_embedding, retrieval=retrieval, text=text)
