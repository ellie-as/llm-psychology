from typing import List

from .embed import LocalEmbedder
from .memory import VectorStore


class Retriever:
    def __init__(self, embedder: LocalEmbedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def index(self, texts: List[str]):
        embs = self.embedder.encode(texts)
        self.store.add(embs, texts)

    def retrieve(self, queries: List[str], top_k: int = 5):
        q = self.embedder.encode(queries)
        return self.store.search(q, top_k=top_k)


