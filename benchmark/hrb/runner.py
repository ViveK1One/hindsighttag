"""Replay an HRB stream through a host, with or without HindsightTag attached."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from hindsighttag.config import HindsightTagConfig
from hindsighttag.core import HindsightTagPlugin
from hindsighttag.interface import MemoryItem

from .dataset import StreamEvent


def replay_stream(
    host,
    events: List[StreamEvent],
    config: Optional[HindsightTagConfig] = None,
    use_hindsighttag: bool = False,
    end_time: Optional[float] = None,
) -> Tuple[Any, Optional[HindsightTagPlugin]]:
    """Store events in temporal order, ticking the clock between timestamps."""
    plugin = HindsightTagPlugin(host, config) if use_hindsighttag and config else None
    if not events:
        return host, plugin

    cutoff = end_time if end_time is not None else events[-1].timestamp + 1
    last_ts = events[0].timestamp

    for event in events:
        if event.timestamp > cutoff:
            break
        if event.timestamp > last_ts:
            (plugin or host).tick(event.timestamp)
            last_ts = event.timestamp
        item = MemoryItem(
            memory_id=event.memory_id,
            content=event.content,
            timestamp=event.timestamp,
            retention=event.retention,
            salience=event.salience,
            metadata=dict(event.metadata),
        )
        (plugin or host).store(item)

    (plugin or host).tick(cutoff)
    (plugin or host).consolidate()
    return host, plugin
