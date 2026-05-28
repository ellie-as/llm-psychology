from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return embs.astype(np.float32)


