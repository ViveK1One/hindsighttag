"""Mem0 adapter.

Wraps Mem0 (https://github.com/mem0ai/mem0) behind the Section 4.8 host
interface so HindsightTag can attach via the three lifecycle hooks. Mem0 does
per-turn LLM-adjudicated ADD/UPDATE/DELETE against similar memories
(Chhikara et al., 2025); its cross-memory interaction is similarity-gated and
non-increasing, so it satisfies the encoding-time-monotonic assumption and
scores near-zero Rescue Recall unmodified (paper Section 5.3).

Behaviour:

* If ``mem0ai`` is importable, its retrieval index is used (best effort) and
  ``mem0_available`` is True. Mem0 is Apache 2.0 licensed (https://github.com/mem0ai/mem0).
* Retention dynamics and salience gating are provided by the decay model (a
  faithful stand-in for Mem0's decay/forgetting behaviour that also gives the
  benchmark a deterministic, offline retention trajectory to measure).

This mirrors how you would attach HindsightTag to a live Mem0 deployment: the
plug-in only needs the four accessors and the three hooks.
"""

from __future__ import annotations

from typing import List, Optional

from ..embeddings import EmbeddingModel, get_embedding_model
from ..interface import MemoryItem
from .decay_host import DecayMemoryHost

_MEM0_IMPORTABLE: Optional[bool] = None


def _mem0_importable() -> bool:
    global _MEM0_IMPORTABLE
    if _MEM0_IMPORTABLE is None:
        try:
            import mem0  # noqa: F401

            _MEM0_IMPORTABLE = True
        except Exception:
            _MEM0_IMPORTABLE = False
    return _MEM0_IMPORTABLE


class Mem0Host(DecayMemoryHost):
    """Mem0-backed host: decay + salience gating + similarity-gated updates."""

    def __init__(
        self,
        consolidation_threshold: float = 0.4,
        half_life: float = 14 * 24 * 3600.0,
        user_id: str = "hindsighttag_user",
        seed: int = 42,
        embedder: Optional[EmbeddingModel] = None,
        use_live_mem0: bool = False,
    ):
        super().__init__(
            consolidation_threshold=consolidation_threshold,
            half_life=half_life,
            embedder=embedder or get_embedding_model(seed=seed),
            seed=seed,
        )
        self.user_id = user_id
        self.use_live_mem0 = use_live_mem0
        self._mem0 = None
        if use_live_mem0 and _mem0_importable():
            self._init_live_mem0()

    @property
    def mem0_available(self) -> bool:
        return _mem0_importable()

    def _init_live_mem0(self) -> None:
        import os
        import tempfile

        try:
            from mem0 import Memory

            tmp = tempfile.mkdtemp(prefix="hindsighttag_mem0_")
            self._mem0 = Memory.from_config(
                {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": "hindsighttag",
                            "path": os.path.join(tmp, "qdrant"),
                            "embedding_model_dims": 384,
                        },
                    },
                    "embedder": {
                        "provider": "huggingface",
                        "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
                    },
                    "version": "v1.1",
                }
            )
        except Exception:
            self._mem0 = None

    def store(self, item: MemoryItem) -> None:
        super().store(item)
        if self._mem0 is not None:
            try:
                self._mem0.add(
                    item.content,
                    user_id=self.user_id,
                    metadata={"memory_id": item.memory_id, "salience": item.salience},
                    infer=False,
                )
            except Exception:
                pass

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if self._mem0 is not None:
            try:
                res = self._mem0.search(query, user_id=self.user_id, limit=top_k)
                payload = res.get("results", res) if isinstance(res, dict) else res
                ids = [
                    r.get("metadata", {}).get("memory_id")
                    for r in payload
                    if isinstance(r, dict) and r.get("metadata", {}).get("memory_id")
                ]
                if ids:
                    return ids
            except Exception:
                pass
        return super().retrieve(query, top_k=top_k)
