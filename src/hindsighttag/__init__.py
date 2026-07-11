"""HindsightTag: a synaptic tagging-and-capture plug-in for LLM agent memory.

Retroactively rescues decaying, low-salience memories when a later high-salience
event triggers capture (Eqs. 2-5), and links memories by temporal proximity
independent of content (Eq. 6). Attaches to any host memory system via three
lifecycle hooks (Section 4.6).

This is a new, unvalidated mechanism released for community testing -- not a
proven production system. See the README "Status & honest caveats" section.
"""

from .config import HindsightTagConfig
from .core import (
    HindsightTagPlugin,
    SynapticTag,
    TemporalGraph,
    TombstoneRecord,
)
from .interface import HostMemorySystem, MemoryItem

__version__ = "0.1.0"

__all__ = [
    "HindsightTagConfig",
    "HindsightTagPlugin",
    "SynapticTag",
    "TemporalGraph",
    "TombstoneRecord",
    "HostMemorySystem",
    "MemoryItem",
    "__version__",
]
