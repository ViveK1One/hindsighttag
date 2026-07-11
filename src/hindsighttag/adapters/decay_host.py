"""Ebbinghaus-style decay host (encoding-time-monotonic baseline).

A concrete :class:`~hindsighttag.interface.HostMemorySystem` that models the
forgetting-based systems surveyed in the paper (Section 2.2): exponential decay,
salience gating, and *similarity-gated* suppression of older similar memories
(non-increasing only). It has **no** retroactive rescue -- that is exactly what
HindsightTag adds on top.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..embeddings import EmbeddingModel, get_embedding_model
from ..interface import HostMemorySystem, MemoryItem


class DecayMemoryHost(HostMemorySystem):
    def __init__(
        self,
        consolidation_threshold: float = 0.4,
        half_life: float = 14 * 24 * 3600.0,
        merge_similarity_threshold: float = 0.85,
        embedder: Optional[EmbeddingModel] = None,
        seed: int = 42,
    ):
        self.consolidation_threshold = consolidation_threshold
        self.half_life = half_life
        self.merge_similarity_threshold = merge_similarity_threshold
        self.embedder = embedder or get_embedding_model(seed=seed)
        self._memories: Dict[str, MemoryItem] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._current_time: float = 0.0
        self._last_decay_time: Dict[str, float] = {}

    # --- Section 4.8 accessors ---
    def get_salience(self, item: MemoryItem) -> float:
        return float(item.salience)

    def get_timestamp(self, item: MemoryItem) -> float:
        return float(item.timestamp)

    def get_relatedness(self, item_a: MemoryItem, item_b: MemoryItem) -> float:
        return self.embedder.cosine_similarity(item_a.content, item_b.content)

    def get_consolidation_threshold(self) -> float:
        return self.consolidation_threshold

    # --- decay internals ---
    def _decay_factor(self, elapsed: float) -> float:
        if elapsed <= 0:
            return 1.0
        return math.exp(-math.log(2) * elapsed / self.half_life)

    def _apply_decay(self, memory_id: str, current_time: float) -> None:
        item = self._memories.get(memory_id)
        if item is None or item.tombstoned or item.consolidated:
            return
        last = self._last_decay_time.get(memory_id, item.timestamp)
        elapsed = current_time - last
        if elapsed <= 0:
            return
        item.retention *= self._decay_factor(elapsed)
        self._last_decay_time[memory_id] = current_time
        if item.retention < 0.05:
            item.tombstoned = True

    def _maybe_merge_similar(self, item: MemoryItem) -> None:
        """Similarity-gated suppression (Section 2.2), non-increasing only."""
        for mid, other in self._memories.items():
            if mid == item.memory_id or other.tombstoned:
                continue
            if self.embedder.cosine_similarity(item.content, other.content) >= self.merge_similarity_threshold:
                other.retention *= 0.5

    # --- storage primitives ---
    def store(self, item: MemoryItem) -> None:
        self._current_time = max(self._current_time, item.timestamp)
        for mid in list(self._memories.keys()):
            self._apply_decay(mid, item.timestamp)
        if item.salience >= self.consolidation_threshold:
            item.retention = max(item.retention, item.salience)
        else:
            item.retention = min(item.retention, item.salience)
        self._memories[item.memory_id] = item
        self._embeddings[item.memory_id] = self.embedder.encode_one(item.content)
        self._last_decay_time[item.memory_id] = item.timestamp
        self._maybe_merge_similar(item)

    def tick(self, current_time: float) -> None:
        self._current_time = current_time
        for mid in list(self._memories.keys()):
            self._apply_decay(mid, current_time)

    def consolidate(self) -> None:
        for item in self._memories.values():
            if item.retention >= self.consolidation_threshold:
                item.consolidated = True

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        if memory_id in self._memories:
            self._apply_decay(memory_id, self._current_time)
        return self._memories.get(memory_id)

    def list_memories(self) -> List[MemoryItem]:
        for mid in list(self._memories.keys()):
            self._apply_decay(mid, self._current_time)
        return [m for m in self._memories.values() if not m.tombstoned]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        active = self.list_memories()
        if not active:
            return []
        q = self.embedder.encode_one(query)
        scores = []
        for item in active:
            emb = self._embeddings[item.memory_id]
            denom = float(np.linalg.norm(q) * np.linalg.norm(emb))
            sim = float(np.dot(q, emb) / denom) if denom else 0.0
            scores.append((sim * item.retention, item.memory_id))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [mid for _, mid in scores[:top_k]]

    def is_retrievable(self, memory_id: str, query: str, top_k: int = 5) -> bool:
        item = self.get_memory(memory_id)
        if item is None or item.tombstoned or item.retention < 0.05:
            return False
        return memory_id in self.retrieve(query, top_k=top_k)

    def retention_at(self, memory_id: str) -> float:
        item = self.get_memory(memory_id)
        if item is None or item.tombstoned:
            return 0.0
        return item.retention
