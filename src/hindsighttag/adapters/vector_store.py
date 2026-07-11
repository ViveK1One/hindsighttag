"""Plain vector-store host (second baseline).

Stores everything, never decays, retrieves by pure cosine similarity. This is
encoding-time-monotonic by construction (Section 3) and useful as a contrast:
because nothing is ever forgotten, there is nothing for HindsightTag to rescue.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..embeddings import EmbeddingModel, get_embedding_model
from ..interface import HostMemorySystem, MemoryItem


class VectorStoreHost(HostMemorySystem):
    def __init__(
        self,
        consolidation_threshold: float = 0.4,
        embedder: Optional[EmbeddingModel] = None,
        seed: int = 42,
    ):
        self.consolidation_threshold = consolidation_threshold
        self.embedder = embedder or get_embedding_model(seed=seed)
        self._memories: Dict[str, MemoryItem] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

    def get_salience(self, item: MemoryItem) -> float:
        return float(item.salience)

    def get_timestamp(self, item: MemoryItem) -> float:
        return float(item.timestamp)

    def get_relatedness(self, item_a: MemoryItem, item_b: MemoryItem) -> float:
        return self.embedder.cosine_similarity(item_a.content, item_b.content)

    def get_consolidation_threshold(self) -> float:
        return self.consolidation_threshold

    def store(self, item: MemoryItem) -> None:
        self._memories[item.memory_id] = item
        self._embeddings[item.memory_id] = self.embedder.encode_one(item.content)

    def tick(self, current_time: float) -> None:
        pass

    def consolidate(self) -> None:
        pass

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        return self._memories.get(memory_id)

    def list_memories(self) -> List[MemoryItem]:
        return list(self._memories.values())

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if not self._memories:
            return []
        q = self.embedder.encode_one(query)
        scores = []
        for mid, emb in self._embeddings.items():
            denom = float(np.linalg.norm(q) * np.linalg.norm(emb))
            sim = float(np.dot(q, emb) / denom) if denom else 0.0
            scores.append((sim, mid))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [mid for _, mid in scores[:top_k]]

    def is_retrievable(self, memory_id: str, query: str, top_k: int = 5) -> bool:
        if memory_id not in self._memories:
            return False
        return memory_id in self.retrieve(query, top_k=top_k)
