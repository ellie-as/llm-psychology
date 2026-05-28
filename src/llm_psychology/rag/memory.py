from typing import List, Tuple

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []

    def add(self, embeddings: np.ndarray, texts: List[str]):
        assert embeddings.shape[0] == len(texts)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Tuple[float, str]]]:
        scores, idxs = self.index.search(query_embeddings, top_k)
        results: List[List[Tuple[float, str]]] = []
        for row_scores, row_idxs in zip(scores, idxs):
            results.append([(float(s), self.texts[i]) for s, i in zip(row_scores, row_idxs)])
        return results


