"""Host-system adapters for HindsightTag.

Each adapter implements :class:`hindsighttag.interface.HostMemorySystem`. To add
your own, see ``CONTRIBUTING.md``.
"""

from .decay_host import DecayMemoryHost
from .mem0_adapter import Mem0Host
from .vector_store import VectorStoreHost

__all__ = ["DecayMemoryHost", "Mem0Host", "VectorStoreHost"]
