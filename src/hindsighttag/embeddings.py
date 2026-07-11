"""Optional local embedding utility for adapters that need Rel(.,.) or retrieval.

Uses sentence-transformers when installed (``pip install hindsighttag[embeddings]``).
Default model: ``sentence-transformers/all-MiniLM-L6-v2`` (Apache 2.0, downloaded
from Hugging Face on first use). Results are cached per text so repeated
similarity calls are cheap and deterministic. If the dependency is missing, a
lightweight deterministic hashing fallback is used so the package still imports
and runs.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class EmbeddingModel:
    """Deterministic sentence embeddings with a per-text cache."""

    _shared: Dict[int, "EmbeddingModel"] = {}
    _cache: Dict[str, np.ndarray] = {}

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self._model = None
        self._backend = "hash"

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._backend = "sentence-transformers"
            except Exception:
                self._model = None
                self._backend = "hash"
        return self._model

    def _hash_embed(self, text: str, dim: int = 384) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((self.seed, text))) % (2**32))
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode_one(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        model = self._load()
        if model is not None:
            vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        else:
            vec = self._hash_embed(text)
        self._cache[text] = vec
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.vstack([self.encode_one(t) for t in texts])

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        a = self.encode_one(text_a)
        b = self.encode_one(text_b)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        sim = float(np.dot(a, b) / denom)
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def get_embedding_model(seed: int = 42) -> EmbeddingModel:
    if seed not in EmbeddingModel._shared:
        EmbeddingModel._shared[seed] = EmbeddingModel(seed=seed)
    return EmbeddingModel._shared[seed]
