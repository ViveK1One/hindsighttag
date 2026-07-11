"""Host-system interface contract (paper Section 4.8).

HindsightTag treats the host memory system as a black box behind a four-function
accessor interface plus three lifecycle hooks (Section 4.6). Any host that
exposes these can attach the plug-in without modifying its internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    """A discrete, timestamped, auditable symbolic memory record.

    This is the unit HindsightTag operates over (contrast with prior
    computational STC models that operate on continuous synaptic weights).
    """

    memory_id: str
    content: str
    timestamp: float
    retention: float = 1.0          # v_i(t) in [0, 1]
    salience: float = 0.5           # Sal(.) in [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
    consolidated: bool = False
    tombstoned: bool = False
    provenance: List[Dict[str, Any]] = field(default_factory=list)


class HostMemorySystem(ABC):
    """Abstract host-system interface HindsightTag attaches to.

    A concrete host must implement the four accessors from Section 4.8 and the
    storage/retrieval primitives the benchmark and rescue operator rely on. The
    three lifecycle hooks (``store``, ``tick``, ``consolidate``) are driven by
    :class:`hindsighttag.core.HindsightTagPlugin`, which wraps the host.
    """

    # ------------------------------------------------------------------
    # Section 4.8 interface contract: the four required accessors.
    # ------------------------------------------------------------------
    @abstractmethod
    def get_salience(self, item: MemoryItem) -> float:
        """Return the host's own Sal(.) in [0, 1] for ``item``.

        Used both to decide whether a newly written item receives a tag
        (Section 4.2) and to detect capture events (Section 4.3).
        """

    @abstractmethod
    def get_timestamp(self, item: MemoryItem) -> float:
        """Return the write time tau_i, monotonically non-decreasing in arrival order."""

    @abstractmethod
    def get_relatedness(self, item_a: MemoryItem, item_b: MemoryItem) -> float:
        """Return an optional content-relatedness score in [0, 1] (Rel(.,.) in Eq. 4).

        If a host cannot supply this, HindsightTag must be run with
        ``omega_assoc = 0`` (pure heterosynaptic capture).
        """

    @abstractmethod
    def get_consolidation_threshold(self) -> float:
        """Return the scalar below which the host would let an item decay untouched.

        Used to decide tag eligibility at write time (Section 4.2).
        """

    # ------------------------------------------------------------------
    # Storage primitives the plug-in and benchmark drive.
    # ------------------------------------------------------------------
    @abstractmethod
    def store(self, item: MemoryItem) -> None:
        """Persist ``item`` in the host store (host's own encoding/gating)."""

    @abstractmethod
    def tick(self, current_time: float) -> None:
        """Advance the host's decay/reinforcement clock to ``current_time``."""

    @abstractmethod
    def consolidate(self) -> None:
        """Run the host's own consolidation pass."""

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Return the stored item, or ``None`` if unknown."""

    @abstractmethod
    def list_memories(self) -> List[MemoryItem]:
        """Return all currently live (non-tombstoned) items."""

    @abstractmethod
    def is_retrievable(self, memory_id: str, query: str, top_k: int = 5) -> bool:
        """Return whether ``memory_id`` appears in the top-k retrieval for ``query``."""

    def retention_at(self, memory_id: str) -> float:
        """Return current retention v_i for ``memory_id`` (0 if gone). Optional override."""
        item = self.get_memory(memory_id)
        if item is None or item.tombstoned:
            return 0.0
        return item.retention
