"""HindsightTag core mechanism (paper Section 4, Equations 2-6).

This module implements the three named objects from the paper:

* :class:`SynapticTag`      -- Eq. 2, the short-lived per-memory tag g_i(t).
* :class:`TemporalGraph`    -- Eq. 6, the content-independent co-allocation graph T,
                                stored as a bucketed structure of width ``E_window``
                                (Section 4.8 recommended data structures).
* :class:`HindsightTagPlugin` -- the lifecycle plug-in wiring Eqs. 2-6 into the host's
                                ``store()`` / ``tick()`` / ``consolidate()`` hooks
                                (Section 4.6), with ``T_active`` backed by a min-heap
                                keyed on tag-expiry time (Section 4.8).

The plug-in is *non-interfering*: if no capture event ever occurs for a tagged
memory, HindsightTag leaves that memory's trajectory untouched (Section 4.6).
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import HindsightTagConfig
from .interface import HostMemorySystem, MemoryItem


# ======================================================================
# Eq. 2 -- Synaptic tag and tag decay
# ======================================================================
@dataclass
class SynapticTag:
    """A short-lived synaptic tag attached to a low-salience memory (Eq. 2).

    The tag value decays exponentially with time constant ``T_tag`` and is
    defined only within the bounded window ``[tau_i, tau_i + W_capture]``;
    outside it the value is zero.
    """

    memory_id: str
    tau_i: float          # write time
    g0: float             # initial tag value g_0 = v_i(tau_i)
    T_tag: float          # decay time constant
    expiry_time: float    # tau_i + W_capture
    consolidated: bool = False

    def value(self, t: float) -> float:
        """Return g_i(t) per Eq. 2 (0 outside the capture window)."""
        if t < self.tau_i or t > self.expiry_time:
            return 0.0
        return self.g0 * math.exp(-(t - self.tau_i) / self.T_tag)

    def is_active(self, t: float) -> bool:
        return (not self.consolidated) and self.value(t) > 0.0


@dataclass(order=True)
class _HeapEntry:
    """Min-heap entry keyed on tag-expiry time (Section 4.8)."""

    expiry_time: float
    memory_id: str = field(compare=False)


# ======================================================================
# Eq. 6 -- Temporal co-allocation graph
# ======================================================================
class TemporalGraph:
    """Content-independent temporal co-allocation graph T (Eq. 6).

    Implemented as a bucketed structure indexed by fixed-width time windows of
    size ``E_window`` so a new memory's edges are computed by inspecting only
    its own and adjacent buckets, never the full store (Section 4.8).
    """

    def __init__(self, e_window: float, theta_link: float):
        self.e_window = e_window
        self.theta_link = theta_link
        self._buckets: Dict[int, Set[str]] = defaultdict(set)
        self._timestamps: Dict[str, float] = {}
        self._edges: Dict[Tuple[str, str], float] = {}

    def _bucket_index(self, timestamp: float) -> int:
        return int(timestamp // self.e_window)

    @staticmethod
    def link_weight(tau_i: float, tau_j: float, e_window: float) -> float:
        """Return L(i, j) per Eq. 6 (0 beyond the eligibility window)."""
        delta = abs(tau_i - tau_j)
        if delta > e_window:
            return 0.0
        return math.exp(-delta / e_window)

    def register(self, memory_id: str, timestamp: float) -> None:
        """Add a memory and create co-allocation edges to temporal neighbours."""
        bucket = self._bucket_index(timestamp)
        self._buckets[bucket].add(memory_id)
        self._timestamps[memory_id] = timestamp

        for neighbour_bucket in (bucket - 1, bucket, bucket + 1):
            for other_id in self._buckets.get(neighbour_bucket, set()):
                if other_id == memory_id:
                    continue
                weight = self.link_weight(
                    timestamp, self._timestamps[other_id], self.e_window
                )
                if weight >= self.theta_link:
                    pair = tuple(sorted((memory_id, other_id)))
                    self._edges[pair] = weight

    def neighbours(self, memory_id: str, k: int = 5) -> List[str]:
        """Return the top-k temporal co-allocates of ``memory_id`` by edge weight."""
        scored: List[Tuple[float, str]] = []
        for (a, b), weight in self._edges.items():
            if a == memory_id:
                scored.append((weight, b))
            elif b == memory_id:
                scored.append((weight, a))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [mid for _, mid in scored[:k]]

    @property
    def edges(self) -> Dict[Tuple[str, str], float]:
        return dict(self._edges)


@dataclass
class TombstoneRecord:
    """Permanent record of a tag that expired unrescued (Section 4.8)."""

    memory_id: str
    last_tag_value: float
    expired_at: float


# ======================================================================
# The lifecycle plug-in (Section 4.6) wiring Eqs. 2-6
# ======================================================================
class HindsightTagPlugin:
    """Attach HindsightTag to a host memory system via three lifecycle hooks.

    Parameters
    ----------
    host:
        Any :class:`~hindsighttag.interface.HostMemorySystem`.
    config:
        A :class:`~hindsighttag.config.HindsightTagConfig`. If omitted, the
        literature-informed calibration seeds are used.
    """

    def __init__(self, host: HostMemorySystem, config: Optional[HindsightTagConfig] = None):
        self.host = host
        self.config = config or HindsightTagConfig()
        self._tags: Dict[str, SynapticTag] = {}
        self._tag_heap: List[_HeapEntry] = []
        self.temporal_graph = TemporalGraph(self.config.E_window, self.config.theta_link)
        self._rescue_log: List[Dict[str, Any]] = []
        self._tombstones: List[TombstoneRecord] = []
        self._current_time: float = 0.0

    # ------------------------------------------------------------------
    # Eq. 2 helpers
    # ------------------------------------------------------------------
    def _maybe_assign_tag(self, item: MemoryItem) -> None:
        """Tag a newly written item iff its salience is below consolidation (Section 4.2)."""
        salience = self.host.get_salience(item)
        if salience >= self.host.get_consolidation_threshold():
            return  # host would consolidate it anyway; no tag needed.

        tau = self.host.get_timestamp(item)
        g0 = max(item.retention, salience)  # g_0 = v_i(tau_i)
        tag = SynapticTag(
            memory_id=item.memory_id,
            tau_i=tau,
            g0=g0,
            T_tag=self.config.T_tag,
            expiry_time=tau + self.config.W_capture,
        )
        self._tags[item.memory_id] = tag
        heapq.heappush(self._tag_heap, _HeapEntry(tag.expiry_time, item.memory_id))

    # ------------------------------------------------------------------
    # Eq. 3 -- T_active(t_e)
    # ------------------------------------------------------------------
    def _active_tags(self, t_e: float) -> List[SynapticTag]:
        """Return the currently-tagged, not-yet-consolidated memories (Eq. 3)."""
        window_start = t_e - self.config.W_capture
        active = [
            tag
            for tag in self._tags.values()
            if (not tag.consolidated)
            and window_start <= tag.tau_i < t_e
            and tag.value(t_e) > 0.0
        ]
        return active

    def _capture_mode(self) -> str:
        omega = self.config.omega_assoc
        if omega == 0.0:
            return "heterosynaptic"
        if omega == 1.0:
            return "associative"
        return "mixed"

    # ------------------------------------------------------------------
    # Eqs. 3-5 -- capture detection and backward rescue
    # ------------------------------------------------------------------
    def _execute_capture(self, event: MemoryItem, t_e: float) -> None:
        sal_e = self.host.get_salience(event)
        if sal_e < self.config.theta_capture:
            return  # not a capture event.

        candidates = self._active_tags(t_e)
        # Deterministic tie-breaking: increasing memory_id order (Section 4.8).
        candidates.sort(key=lambda tag: tag.memory_id)

        for tag in candidates:
            memory = self.host.get_memory(tag.memory_id)
            if memory is None or memory.tombstoned:
                continue

            g_i = tag.value(t_e)
            rel = self.host.get_relatedness(event, memory)
            # Eq. 4 -- capture strength
            c_i = g_i * sal_e * (
                self.config.omega_assoc * rel + (1.0 - self.config.omega_assoc)
            )
            if c_i < self.config.theta_rescue:
                continue

            # Eq. 5 -- rescue transition
            v_before = memory.retention
            v_after = v_before + self.config.delta_rescue * (1.0 - v_before) * c_i
            memory.retention = min(1.0, v_after)
            memory.consolidated = True
            tag.consolidated = True

            record = {
                "memory_id": memory.memory_id,
                "rescued_by": event.memory_id,
                "at": t_e,
                "capture_strength": c_i,
                "omega_assoc_used": self.config.omega_assoc,
                "mode": self._capture_mode(),
                "retention_before": v_before,
                "retention_after": memory.retention,
            }
            memory.provenance.append(record)
            self._rescue_log.append(record)

    # ------------------------------------------------------------------
    # Tag expiry (min-heap pop, Section 4.8)
    # ------------------------------------------------------------------
    def _expire_tags(self, current_time: float) -> None:
        while self._tag_heap and self._tag_heap[0].expiry_time <= current_time:
            entry = heapq.heappop(self._tag_heap)
            tag = self._tags.get(entry.memory_id)
            if tag is None or tag.consolidated:
                continue
            last_g = tag.value(current_time)
            if last_g <= 0.0:
                self._tombstones.append(
                    TombstoneRecord(
                        memory_id=tag.memory_id,
                        last_tag_value=last_g,
                        expired_at=current_time,
                    )
                )
                del self._tags[entry.memory_id]

    # ==================================================================
    # Lifecycle hooks (Section 4.6)
    # ==================================================================
    def store(self, item: MemoryItem) -> None:
        """``store(item)``: host-store, tag (Eq. 2), register in T (Eq. 6), and
        check whether ``item`` itself triggers a capture event (Eqs. 3-5)."""
        self.host.store(item)
        self._current_time = max(self._current_time, self.host.get_timestamp(item))
        self._maybe_assign_tag(item)
        self.temporal_graph.register(item.memory_id, self.host.get_timestamp(item))
        self._execute_capture(item, self._current_time)

    def tick(self, current_time: float) -> None:
        """``tick()``/``decay()``: advance host decay and expire stale tags (Eq. 2)."""
        self._current_time = current_time
        self.host.tick(current_time)
        self._expire_tags(current_time)

    def consolidate(self) -> None:
        """``consolidate()``: absorb surviving tags into the host; drop expired ones."""
        self._expire_tags(self._current_time)
        threshold = self.host.get_consolidation_threshold()
        for mid, tag in self._tags.items():
            if tag.consolidated:
                continue
            memory = self.host.get_memory(mid)
            if memory is not None and memory.retention >= threshold:
                tag.consolidated = True
        self.host.consolidate()

    # ------------------------------------------------------------------
    # Retrieval helper and audit accessors
    # ------------------------------------------------------------------
    def get_temporal_coallocates(self, memory_id: str, k: int = 5) -> List[str]:
        """Retrieve "what else was recorded around this time" (Eq. 6 retrieval axis)."""
        return self.temporal_graph.neighbours(memory_id, k=k)

    @property
    def rescue_log(self) -> List[Dict[str, Any]]:
        """Append-only provenance log of every rescue (Section 4.8 schema)."""
        return list(self._rescue_log)

    @property
    def tombstones(self) -> List[TombstoneRecord]:
        """Records of tags that expired unrescued (Section 4.8 tombstone schema)."""
        return list(self._tombstones)

    @property
    def active_tag_count(self) -> int:
        return sum(1 for t in self._tags.values() if not t.consolidated)
